"""engine.py - The single training loop. Shared by every architecture.

Extracted from the former monolithic train.py (plan.md 9: ONE training
system, split into readable modules; not parallel implementations).
"""

import os
import sys
import json
import time
import math
import collections
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import wandb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .families import expert_labels, ALL_OP_NAMES, NUM_OP_TYPES, NUM_EXPERTS_CANONICAL
from .data import MoREDataset
from .lang_data import MoRELanguageDataset, LANG_ROOT
from .model import MoEBlock, MoREWrapper, MoREModel
from .metrics import (compute_expert_load_entropy,
                      compute_pairwise_cosine_sim,
                      compute_token_exit_depths,
                      routing_accuracy_from_confusion,
                      permutation_invariant_routing_metrics,
                      paper_metrics_to_wandb,
                      make_routing_confusion_figure,
                      make_op_depth_bar_figure,
                      language_depth_metrics,
                      language_depth_to_wandb,
                      specialization_vs_control,
                      article_agreement_vs_topic,
                      control_comparison_to_wandb,
                      DEPTH_KEY_PROVENANCE,
                      uncovered_depth_keys)
from .run_context import RunContext, resolve_overrides, ProxyGuardError
from .seeding import (apply_seeding, make_generator, seed_worker,
                      nondeterministic_ops_observed)
from .model import (CAPACITY_POLICY, CANONICAL_ROUTING_MODE,
                    CANONICAL_ROUTER_NOISE,
                    ROUTER_NOISE_INIT_SCALE_DEFAULT,
                    ROUTER_NOISE_ANNEAL_STEPS_DEFAULT)
from .metrics import halting_supervision_loss, depth_allocation_error
from .families import op_target_depth_table
from .config import (resolve_halting_mode, CANONICAL_FAMILY_CLS_WEIGHT,
                     resolve_task, TASK_LANGUAGE)
# T-L3.2: the confusion diagonal is published under a task-dependent NAME. One
# helper, so the log dict, the console line, the results.tsv header and the
# metrics.json "N/A" contract cannot disagree about what it is called.
from .metrics import routing_agreement_key, routing_agreement_caption

# ---------------------------------------------------------------------------
# 4. Training loop
# ---------------------------------------------------------------------------

def train(cfg: dict, run_epochs: int | None = None, ctx: "RunContext | None" = None):
    """
    Full training procedure.

    Args:
        cfg        : parsed config dict
        run_epochs : override epoch count (used by verify_pipeline.py for
                     short diagnostic runs)
        ctx        : RunContext owning runs/<experiment_id>/. If None, one is
                     created here so that every entry point (including direct
                     `train(cfg)` calls from smoke_test.py / verify_pipeline.py)
                     gets per-run output paths instead of global filenames
                     (plan.md 2.2).

    Returns:
        best_val_loss
    """
    owns_ctx = ctx is None
    if ctx is None:
        resolved = resolve_overrides(cfg, epochs=run_epochs)
        ctx = RunContext.create(cfg, resolved)
    print(f"[Run] experiment_id   = {ctx.experiment_id}")
    print(f"[Run] experiment_group= {ctx.experiment_group}")
    print(f"[Run] output directory= {ctx.dir}")

    mc  = cfg["model"]
    tc  = cfg["training"]
    lw  = cfg["loss_weights"]
    # T8.3: family_cls was a bare 0.5 literal in the loss assembly until now, so
    # a config dict hand-built by a caller that does not go through
    # config.load_config (smoke_test.py, verify_pipeline.py, the regression
    # suites) has no such key and the assembly would KeyError. Default it to the
    # canonical weight HERE, into cfg itself, so the value reaches
    # resolved_config.json and provenance rather than being silently supplied at
    # the point of use -- the exact failure mode the literal had.
    lw.setdefault("family_cls", CANONICAL_FAMILY_CLS_WEIGHT)
    dc  = cfg["data"]
    log = cfg["logging"]
    # Resolved before the dataset because the dataset depends on it. `_task` below is
    # the same value from the same helper; this early binding exists only because the
    # dataset branch sits above the block where `_task` is set, and two names for one
    # value is cheaper than moving the halting-mode resolution.
    _task_early = resolve_task(cfg)

    # ---- T6.1 global seeding (plan.md §8.1) -----------------------------
    # FIRST, before the dataset, the model or the optimiser exist. Every one of
    # them draws from an RNG at construction time -- weight init, dropout masks,
    # shuffle order -- so seeding after any of them would leave that part of the
    # run unreproducible while the log still claimed a seed. The report is
    # written into cfg["provenance"], which IS ctx.resolved_cfg, and flushed
    # immediately so a crashed run still records how it was seeded.
    #
    # Seeded from ctx.resolved_cfg rather than cfg: on the CLI path the two are
    # the same object, but when train() creates its own context (smoke_test.py,
    # verify_pipeline.py) resolved_cfg is a deep copy, and writing the seed
    # provenance into the caller's dict would leave resolved_config.json silent
    # about the seeding of the run it describes.
    seed, seed_report = apply_seeding(ctx.resolved_cfg)
    ctx.write_resolved_config()

    epochs          = run_epochs if run_epochs is not None else tc["epochs"]
    batch_sz        = tc["batch_size"]
    lr              = tc["lr"]
    val_split       = tc["val_split"]
    grad_clip       = tc["grad_clip"]
    subset_fraction = float(dc.get("subset_fraction", 1.0))
    subset_seed     = int(dc.get("subset_seed", 42))
    max_steps       = mc["max_steps"]
    step_feat_dim   = mc["step_feat_dim"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Device: {device}")
    print(
        f"[Train] Effective settings: epochs={epochs}, lr={lr}, "
        f"routing_balance={lw['routing_balance']}, step_routing={lw['step_routing']}, "
        f"subset_fraction={subset_fraction}, subset_seed={subset_seed}"
    )

    # ---- Dataset -------------------------------------------------------
    # T-L7.4: the task chooses the dataset, exactly as `plan_language.md` §1 says the
    # axis works. Both classes return the same 7-tuple (T-L2.3), so everything below
    # this branch -- the DataLoader, the epoch loop, the metric accumulation -- is
    # shared, which is the ONE-training-system requirement of plan.md §9.
    if _task_early == TASK_LANGUAGE:
        _corpus = dc.get("corpus", "wikitext-103")
        train_ds = MoRELanguageDataset(_corpus, "train")
        if int(train_ds.seq_len) != int(dc.get("seq_len", train_ds.seq_len)):
            # A mismatch is a real defect rather than something to coerce: the blocks
            # on disk ARE `train_ds.seq_len` long, and a config asking for a different
            # length is asking for a corpus that was not built. Repacking is a
            # deliberate act (`build_language_dataset.py --stage pack --seq_len N`).
            raise ValueError(
                f"data.seq_len = {dc.get('seq_len')} but "
                f"data/lang/{_corpus}/ was packed at seq_len = {train_ds.seq_len} "
                f"({train_ds.dataset_version}). Repack the corpus or drop the "
                f"override -- silently using the stored length would make the "
                f"resolved config disagree with the data it describes."
            )
        if int(train_ds.vocab_size) != int(dc.get("vocab_size", train_ds.vocab_size)):
            raise ValueError(
                f"data.vocab_size = {dc.get('vocab_size')} but the corpus tokenizer "
                f"has V = {train_ds.vocab_size}. The tied LM head is "
                f"[V, d_model], so this is not a cosmetic disagreement."
            )
        print(f"[Dataset] {train_ds}")
        print(f"[Dataset] floor = {train_ds.primary_metric_floor} nats/token "
              f"(bigram, T-L2.5) | train_split_version = "
              f"{train_ds.train_split_version}")
    else:
        train_ds = MoREDataset(
            jsonl_path=dc.get("train_path", dc.get("jsonl_path")),
            max_steps=max_steps,
            step_feat_dim=step_feat_dim,
            max_val=dc["max_val"],
            pad_value=dc["pad_value"],
            num_experts=mc["num_experts"],
        )

    if 0.0 < subset_fraction < 1.0 and len(train_ds) > 1:
        subset_n    = max(1, int(len(train_ds) * subset_fraction))
        remainder_n = len(train_ds) - subset_n
        train_ds, _ = random_split(
            train_ds,
            [subset_n, remainder_n],
            generator=torch.Generator().manual_seed(subset_seed),
        )
        print(
            f"[Dataset] Using deterministic train subset: {subset_n} samples "
            f"({subset_fraction:.3f}), seed={subset_seed}"
        )

    val_ds   = None
    val_path = dc.get("val_path")
    if _task_early == TASK_LANGUAGE:
        # The corpus ships its own author-provided val split, and T-L2.7 verified zero
        # exact-content overlap with train across all 526,320 canonical train blocks.
        # So there is no random-split fallback here: falling back would carve a
        # validation set out of TRAIN and quietly destroy the one property that makes
        # the leakage audit meaningful.
        val_ds  = MoRELanguageDataset(dc.get("corpus", "wikitext-103"), "val")
        n_train = len(train_ds)
        n_val   = len(val_ds)
        print(f"[Dataset] {val_ds}")
    elif val_path and os.path.exists(val_path):
        print(f"[Dataset] Using validation set: {val_path}")
        val_ds = MoREDataset(
            jsonl_path=val_path,
            max_steps=max_steps,
            step_feat_dim=step_feat_dim,
            max_val=dc["max_val"],
            pad_value=dc["pad_value"],
            num_experts=mc["num_experts"],
        )
        n_train = len(train_ds)
        n_val   = len(val_ds)
    else:
        print("[Dataset] Validation file not found, falling back to random split.")
        dataset = train_ds
        n_val   = max(1, int(len(dataset) * val_split))
        n_train = len(dataset) - n_val
        if n_train < 1:
            n_train = len(dataset)
            n_val   = 0
        if n_val > 0:
            train_ds, val_ds = random_split(
                dataset,
                [n_train, n_val],
                generator=torch.Generator().manual_seed(42),
            )
        else:
            train_ds = dataset
            val_ds   = None

    # num_workers=0 avoids multiprocessing overhead on small datasets. The
    # generator and worker_init_fn are wired regardless (T6.1 / plan.md §8.1):
    # `generator` gives shuffling its own sub-stream, so batch order stops
    # depending on how many draws model init happened to consume, and
    # worker_init_fn is inert at num_workers=0 but correct the moment it is
    # raised -- a reproducibility hole that only opens under a config change is
    # the kind that gets shipped.
    num_workers = int(tc.get("num_workers", 0))
    train_loader = DataLoader(
        train_ds, batch_size=min(batch_sz, n_train),
        shuffle=True, num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        generator=make_generator(seed, "dataloader_shuffle"),
        worker_init_fn=seed_worker,
    )
    val_loader = (
        DataLoader(val_ds, batch_size=min(batch_sz, n_val),
                   shuffle=False, num_workers=num_workers,
                   worker_init_fn=seed_worker)
        if val_ds is not None else None
    )

    # ---- Model ---------------------------------------------------------
    model = MoREModel(
        step_feat_dim=step_feat_dim,
        d_model=mc["d_model"],
        num_experts=mc["num_experts"],
        max_depth=mc["max_depth"],
        num_blocks=mc["num_blocks"],
        dropout=mc["dropout"],
        fixed_depth=mc.get("fixed_depth", False),
        routing_mode=mc.get("routing_mode", CANONICAL_ROUTING_MODE),
        # T5.4: canonical is "none". A noisy run must say so in its config, which
        # then reaches resolved_config.json, W&B and the run directory name.
        router_noise=mc.get("router_noise", CANONICAL_ROUTER_NOISE),
        router_noise_init=mc.get("router_noise_init",
                                 ROUTER_NOISE_INIT_SCALE_DEFAULT),
        router_noise_anneal_steps=mc.get("router_noise_anneal_steps",
                                         ROUTER_NOISE_ANNEAL_STEPS_DEFAULT),
        # T6.7: FFN multiplier is read from config so provenance can report a
        # measured value. Default 4 reproduces every run made before this field
        # existed (verified: 6,358,553 params for MoE/MoRE, unchanged).
        ffn_mult=mc.get("ffn_mult", 4),
        # T8.3: the auxiliary label heads are sized by the DATASET's family label
        # space (always 6), not by num_experts. MoR runs with num_experts=1 but
        # still receives oracle family targets 0..5; sizing these heads by
        # num_experts made every MoR run die on a CUDA device-side assert
        # ('t >= 0 && t < n_classes') in the first batch. Passed explicitly rather
        # than left to the default so the coupling cannot silently return.
        num_families=NUM_EXPERTS_CANONICAL,
        # T-L4.0 / T-L5.0. Attention and the token embedding are constructed only when
        # the task asks for them, so an arithmetic state_dict stays bit-identical.
        # `max_seq_len` sizes the positional table and MUST be the packed length the
        # corpus was built at -- the mismatch guard above is what makes that true.
        attention=bool(mc.get("attention", False)),
        n_heads=int(mc.get("n_heads", 4)),
        max_seq_len=(int(dc["seq_len"]) if _task_early == TASK_LANGUAGE else None),
        task=_task_early,
        vocab_size=(int(dc["vocab_size"]) if _task_early == TASK_LANGUAGE else None),
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[Model] Total parameters: {total_params:,}")

    # ---- Optimiser & Scheduler -----------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=tc["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs
    )

    # ---- W&B -----------------------------------------------------------
    # Phase 0 provenance subset. The five fields below are exactly what the
    # Gate 0 proxy guard needs to be auditable after the fact: which run this
    # is, whether it claims canonical status, the config fingerprint, and the
    # two settings that were historically misreported (epochs and
    # subset_fraction). The FULL provenance field list of updated_rules.md 9 /
    # plan.md 8.7 is a Phase 6 task.
    # T3.3: halting supervision is an explicit, provenance-recorded option, not a
    # silent default. `updated_rules.md` 2.3 permits the operation-complexity
    # curriculum as supervision but requires that whichever mode is used be
    # reported. Off => pure unsupervised ACT.
    halting_supervision_enabled = bool(
        cfg.get("model", {}).get("halting_supervision", False)
    )
    lw.setdefault("halting_supervision", 0.0)
    if halting_supervision_enabled and lw["halting_supervision"] <= 0.0:
        raise ValueError(
            "model.halting_supervision is enabled but "
            "loss_weights.halting_supervision is 0.0, so the curriculum term "
            "would be computed, logged and then multiplied by zero -- the run "
            "would be recorded as supervised while training unsupervised. Set a "
            "non-zero loss_weights.halting_supervision, or disable the flag."
        )
    _target_depth_table = op_target_depth_table()
    # T8.2: derived by the ONE helper the Gate 0 guard also calls, so "which
    # halting objective ran" cannot mean one thing to the guard and another to
    # provenance. See more/config.py:resolve_halting_mode.
    halting_mode = resolve_halting_mode(cfg)
    # T-L3.2: resolved by the SAME helper the config layer and the proxy guard use,
    # so "which task ran" cannot mean one thing to the guard and another to the
    # metric names. Absence means arithmetic (config.py:resolve_task), so every
    # existing arithmetic run keeps emitting `val/routing_accuracy` unchanged.
    _task = resolve_task(cfg)
    _agree_key = routing_agreement_key(_task)
    # T-L6.8: the label manifest follows the task. `expert_labels` from `.families`
    # names the arithmetic op families; a language run must name POS families.
    if _task == TASK_LANGUAGE:
        from .lang_families import expert_labels as _expert_labels
    else:
        _expert_labels = expert_labels
    # T-L5.1: the trivial-baseline floor for THIS corpus, read from the dataset
    # manifest rather than from a config literal, so a run cannot quote a floor
    # measured on a different corpus. None when the dataset does not supply one --
    # reported as an absent key rather than as 0.0.
    _floor = None
    _tok_family = _log_freq = _surprisal = None
    _NUM_LANG_FAMILIES, _LANG_FAMILY_LABELS = 0, []
    if _task == TASK_LANGUAGE:
        # Unwrapped, because `random_split` returns a `Subset` and the attribute
        # lives on the underlying `MoRELanguageDataset`. Bounded loop rather than
        # recursion so a self-referential wrapper cannot hang the run.
        _ds = getattr(train_loader, "dataset", None)
        for _ in range(4):
            if _ds is None or hasattr(_ds, "primary_metric_floor"):
                break
            _ds = getattr(_ds, "dataset", None)
        _floor = getattr(_ds, "primary_metric_floor", None)
        # T-L6.1: the frozen corpus-difficulty vectors, resolved ONCE here rather
        # than per validation pass. They come from the dataset, so a run cannot
        # correlate depth against a frequency table built for another corpus.
        from .lang_families import (LANG_FAMILY_LABELS as _LFL,
                                    NUM_FAMILIES as _NLF)
        _LANG_FAMILY_LABELS, _NUM_LANG_FAMILIES = list(_LFL), _NLF
        if _ds is not None and hasattr(_ds, "log_freq"):
            _tok_family = _ds.token_family.numpy() if _ds.token_family is not None else None
            _log_freq = _ds.log_freq()
            _surprisal = _ds.unigram_surprisal()
        # T-L6.6 / T-L6.9: the two reference artifacts, loaded ONCE from the corpus
        # directory. Absent -> that axis is skipped and its keys are simply not written,
        # which an exporter reads as "not measured" rather than as a zero difference.
        _pos_control = _block_topic = None
        _cdir = os.path.join(LANG_ROOT, dc.get("corpus", "wikitext-103"))
        _pc = os.path.join(_cdir, "token_family_shuffled.npy")
        _bt = os.path.join(_cdir, "block_topic_val.npy")
        if os.path.exists(_pc):
            _pos_control = np.load(_pc)
        else:
            print(f"[Train] no shuffled control at {_pc}: routing_ami will be "
                  f"published WITHOUT its null band. Build it with "
                  f"data/lang/build_shuffled_control.py.", file=sys.stderr)
        if os.path.exists(_bt):
            _block_topic = np.load(_bt)
        else:
            print(f"[Train] no topic partition at {_bt}: the second reference axis is "
                  f"skipped. Build it with data/lang/build_block_topics.py.",
                  file=sys.stderr)
    if _task != "arithmetic":
        print(f"[Train] task={_task}: the confusion diagonal is published as "
              f"{_agree_key}")
        print(f"[Train] {routing_agreement_caption(_task)}")
    target_halt_weight = lw["halting"]
    print(f"[Train] halting: ponder_weight={target_halt_weight}, "
          f"supervision={'ON' if halting_supervision_enabled else 'OFF (pure ACT)'}"
          + (f", supervision_weight={lw['halting_supervision']}"
             if halting_supervision_enabled else ""))

    wandb_config = {**mc, **tc, **lw, **dc}
    wandb_config.update({
        "experiment_id":            ctx.experiment_id,
        "experiment_group":         ctx.experiment_group,
        "config_hash":              ctx.resolved_cfg.get("provenance", {}).get("config_hash"),
        "code_git_commit":          ctx.resolved_cfg.get("provenance", {}).get("code_git_commit"),
        "seed":                     ctx.resolved_cfg.get("provenance", {}).get("seed"),
        # T6.1. `seed` above is the DECLARED seed and is null for an
        # undeclared run; `resolved_seed` is the integer the RNGs actually got.
        # Both are logged because "which seed ran" and "did this run declare a
        # seed" are different questions, and only the second one decides whether
        # the run may enter the canonical table.
        "resolved_seed":            seed,
        "seed_source":              seed_report["seed_source"],
        "determinism_mode":         seed_report["determinism_mode"],
        "cudnn_deterministic":      seed_report["cudnn_deterministic"],
        "cudnn_benchmark":          seed_report["cudnn_benchmark"],
        "torch_deterministic_algorithms":
            seed_report["torch_deterministic_algorithms"],
        "determinism_exceptions":   "; ".join(seed_report["notes"]),
        # Values the loop ACTUALLY uses, not the values sitting in the file.
        "resolved_epochs":          epochs,
        "resolved_subset_fraction": subset_fraction,
        "resolved_batch_size":      batch_sz,
        # T3.3. `{**mc, **lw}` above contains a key collision: model.
        # halting_supervision (bool) and loss_weights.halting_supervision
        # (float) share a name, and lw wins the merge, so the bool would be
        # LOST from provenance. These three keys are collision-free and are the
        # fields updated_rules.md 2.3 requires be reported explicitly.
        "halting_mode":                halting_mode,
        "halting_supervision_enabled": halting_supervision_enabled,
        "halting_supervision_weight":  float(lw["halting_supervision"]),
        "ponder_weight":               float(target_halt_weight),
        # T5.4: the routing path, reported rather than assumed. `{**mc}` above
        # already carries these, but they are restated under explicit names so a
        # reader of the provenance block does not have to know that the model
        # section was splatted in, and so the pair cannot be lost to a key
        # collision the way halting_supervision was.
        "resolved_routing_mode": mc.get("routing_mode", CANONICAL_ROUTING_MODE),
        "resolved_router_noise": mc.get("router_noise", CANONICAL_ROUTER_NOISE),
        # T6.7 (updated_rules.md 9). Every field below is required provenance
        # that `{**mc, **tc, **lw, **dc}` above does NOT supply under a name a
        # reader would look for:
        #   architecture / variant  live at the top level of the config, so the
        #     splats miss them entirely -- and a W&B table without them cannot
        #     tell MoE from MoRE, or canonical from an ablation.
        #   routing_supervision_*   `lw["step_routing"]` is the weight, but the
        #     rules ask for enabled/weight as a pair, and 0.0 vs "off" should not
        #     have to be inferred by the reader.
        #   router_noise_scale      the mode alone does not say how much noise.
        #   halt_target_mode        which halting objective ran: pure ACT or the
        #     complexity curriculum. The most consequential switch in the Phase 7
        #     comparison, and the one an ablation is likeliest to differ on.
        #   ffn_mult                arrives via **mc now that it is a config
        #     field; restated so the provenance block is self-contained.
        "architecture":  ctx.resolved_cfg.get("architecture"),
        "variant":       ctx.resolved_cfg.get("provenance", {}).get("variant"),
        "run_name":      log.get("run_name"),
        "dataset_version":     dc.get("dataset_version"),
        "train_split_version": dc.get("train_split_version"),
        "ffn_mult":            mc.get("ffn_mult", 4),
        "routing_supervision_enabled": float(lw.get("step_routing", 0.0)) != 0.0,
        "routing_supervision_weight":  float(lw.get("step_routing", 0.0)),
        # T8.3: the family-supervision pair, reported the same way as the routing
        # pair and for the same reason -- 0.0 vs "off" should not have to be
        # inferred. Until T8.3 this coefficient was a literal in the loss
        # assembly, so the W&B table could not distinguish a run that had
        # whole-program family supervision from one that did not.
        "family_supervision_enabled": float(lw.get("family_cls", 0.0)) != 0.0,
        "family_cls_weight":          float(lw.get("family_cls", 0.0)),
        "router_noise_scale": (
            float(mc.get("router_noise_init", ROUTER_NOISE_INIT_SCALE_DEFAULT))
            if mc.get("router_noise", CANONICAL_ROUTER_NOISE) != "none" else 0.0
        ),
        "halt_target_mode": halting_mode,
        # Parameter count belongs in provenance, not just stdout: Phase 9's
        # parameter-matching question (MoE/MoRE ~6.36M vs MoR ~1.10M) is asked of
        # the run record, not of a scrollback buffer.
        "total_params": total_params,
    })
    wandb.init(
        project=log["wandb_project"],
        name=log.get("run_name"),
        group=ctx.experiment_group,
        config=wandb_config,
        reinit=True,
    )
    wandb.watch(model, log="gradients", log_freq=50)

    # Keep resolved_config.json in sync with the numbers the loop is using.
    ctx.resolved_cfg.setdefault("provenance", {}).update({
        "resolved_epochs":          epochs,
        "resolved_subset_fraction": subset_fraction,
        "resolved_batch_size":      batch_sz,
        "wandb_run_id":             getattr(wandb.run, "id", None),
        "device":                   str(device),
        "halting_mode":                halting_mode,
        "halting_supervision_enabled": halting_supervision_enabled,
        "halting_supervision_weight":  float(lw["halting_supervision"]),
        "ponder_weight":               float(target_halt_weight),
        "resolved_routing_mode": mc.get("routing_mode", CANONICAL_ROUTING_MODE),
        "resolved_router_noise": mc.get("router_noise", CANONICAL_ROUTER_NOISE),
        # T6.7: the same required fields written to resolved_config.json, so a
        # run can be audited from its own directory with no W&B access.
        "ffn_mult":                    mc.get("ffn_mult", 4),
        "routing_supervision_enabled": float(lw.get("step_routing", 0.0)) != 0.0,
        "routing_supervision_weight":  float(lw.get("step_routing", 0.0)),
        "family_supervision_enabled":  float(lw.get("family_cls", 0.0)) != 0.0,
        "family_cls_weight":           float(lw.get("family_cls", 0.0)),
        "router_noise_scale": (
            float(mc.get("router_noise_init", ROUTER_NOISE_INIT_SCALE_DEFAULT))
            if mc.get("router_noise", CANONICAL_ROUTER_NOISE) != "none" else 0.0
        ),
        "halt_target_mode":            halting_mode,
        "total_params":                total_params,
    })
    ctx.write_resolved_config()

    best_val_loss = float("inf")
    results_rows  = []

    # ---- Halt-loss linear warmup ----------------------------------------
    # Halt pressure is zero for the first 20 % of training steps so the
    # router can explore freely before being pushed toward early exits.
    # After warmup it ramps linearly up to the configured halting weight.
    total_steps        = epochs * len(train_loader)
    warmup_steps       = max(1, int(0.2 * total_steps))
    # target_halt_weight is resolved above, before wandb.init, because the
    # halting mode has to reach the provenance record (T3.3).

    current_step       = 0

    # ---- Training epochs -----------------------------------------------
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_total      = 0.0
        epoch_task       = 0.0
        epoch_cls        = 0.0
        epoch_bal        = 0.0
        epoch_halt       = 0.0
        epoch_halt_sup      = 0.0
        epoch_halt_stats: dict = {}
        epoch_depth_abs_err = 0.0
        epoch_depth_rel_err = 0.0
        epoch_depth_err_n   = 0
        epoch_step_route = 0.0
        # T-L5.3: the probe CE accumulated separately from the blended
        # `step_routing_loss`, because on language it is a MEASUREMENT of
        # decodability rather than a term that trains anything -- reported
        # under `probe/family_ce`, never folded into a total.
        epoch_probe_ce   = 0.0
        epoch_start      = time.perf_counter()
        total_tokens     = 0
        depth_hist       = torch.zeros(mc["max_depth"])

        # --- Shannon entropy tracking (activated) -----------------------
        # epoch_expert_counts: hard-argmax token counts per expert, across all
        # batches in this epoch.  Used to compute:
        #   (a) per-expert load % (logged individually to W&B)
        #   (b) overall Shannon entropy via compute_expert_load_entropy()
        epoch_expert_counts: torch.Tensor = torch.zeros(mc["num_experts"], device=device)
        # Raw per-depth argmax tensors — fed into compute_expert_load_entropy
        epoch_all_expert_idx: list = []
        # T2.3: per-epoch dispatch accounting. Reset each epoch so the logged
        # overflow rate describes THIS epoch, not a running total.
        epoch_route_stats: dict = {}

        for batch_idx, (x, step_mask, step_experts, step_ops, family, depth, target) in enumerate(train_loader):
            x            = x.to(device)
            step_mask    = step_mask.to(device)
            step_experts = step_experts.to(device)
            step_ops     = step_ops.to(device)
            family       = family.to(device)
            target       = target.to(device)

            optimizer.zero_grad(set_to_none=True)

            (
                reg_out, cls_out, step_cls_out,
                bal_loss, ponder_cost,
                depth_exits, avg_depth,
                batch_expert_idx,
                oracle_routing_ce,
                _first_route,
                batch_route_stats,
                expected_depth,
                batch_halt_stats,
            ) = model(x, step_mask, step_experts, step_ops)

            # T2.3 dispatch accounting, accumulated over the epoch.
            for _k, _v in batch_route_stats.items():
                if not isinstance(_v, (int, float)):
                    continue
                if _k in ("max_load_fraction", "experts_called"):
                    epoch_route_stats[_k] = max(epoch_route_stats.get(_k, 0.0), _v)
                else:
                    epoch_route_stats[_k] = epoch_route_stats.get(_k, 0.0) + _v

            # Accumulate expert routing counts for Shannon entropy logging
            if batch_expert_idx is not None:
                for idx_tensor in batch_expert_idx:
                    for e in range(mc["num_experts"]):
                        epoch_expert_counts[e] += (idx_tensor == e).float().sum()
                # Feed raw tensors to compute_expert_load_entropy at epoch end
                epoch_all_expert_idx.extend(batch_expert_idx)

            # --- Losses -------------------------------------------------
            # 1. Primary task loss.
            #
            # T-L5.1: the two tasks compute genuinely different quantities from
            # slot 0, and the branch is here rather than inside the model so the
            # model stays a pure function of its inputs. ARITHMETIC: MSE on the
            # scalar answer. LANGUAGE: causal-LM cross-entropy in NATS/token, with
            # the shift done here (`logits[:, :-1]` predicts `input_ids[:, 1:]`)
            # rather than stored twice in the dataset (T-L2.3).
            #
            # NATS, not bits, and that is not a presentation choice: every other
            # term in the weighted sum below -- balance, ponder, probe -- is in
            # nats, and a bits/nats mix inside one weighted sum is a silent
            # ln(2) = 0.693 scaling bug on whichever term is the odd one out.
            # Perplexity is reported from it and never optimised.
            if _task == TASK_LANGUAGE:
                _V = reg_out.shape[-1]
                task_loss = F.cross_entropy(
                    reg_out[:, :-1, :].reshape(-1, _V),
                    x[:, 1:].reshape(-1),
                )
            else:
                task_loss = F.mse_loss(reg_out.squeeze(-1), target)

            # 2. Whole-program family classification (auxiliary supervision)
            # family == -1 means "multi-operation program, no single family";
            # it is ignored, never treated as a 7th class (plan.md 7.2).
            # If a batch happens to be entirely MIXED, cross_entropy would
            # average over zero elements and return NaN, silently poisoning
            # every downstream loss. Emit an exact zero instead.
            #
            # T-L5.2: on language `cls_out` is None -- a packed LM block has no
            # whole-sequence family, so the head is not constructed. The term is an
            # exact structural zero, not a zero-weighted computation, and
            # `apply_task` already refuses a non-zero `loss_weights.family_cls`
            # there (§7.3), so this branch cannot silently drop a term someone
            # meant to use.
            if cls_out is None:
                cls_loss = torch.zeros((), device=device)
            elif bool((family >= 0).any()):
                cls_loss = F.cross_entropy(cls_out, family, ignore_index=-1)
            else:
                cls_loss = cls_out.sum() * 0.0

            # 3. Per-step routing loss (two complementary signals):
            #    (a) oracle_routing_ce: direct CE on router_logits at each depth step.
            #        Short gradient path — router weights updated immediately.
            #        Teaches the MoEBlock router: ADD→Expert0, MULT→Expert1, etc.
            #    (b) step_cls CE on step_cls_head(h): longer path but helps separate
            #        representations so the router's linear classifier is feasible.
            valid_mask   = step_experts.reshape(-1) >= 0                # [B*S]
            # T5.1 (plan.md 7.1): width derives from the HEAD, never from a
            # literal. Hard-coded 7, this reshape silently reinterpreted a
            # [B, S, 6] tensor as [B*S*6/7, 7] whenever the head width and the
            # literal disagreed -- a view, so no error, just step tokens blended
            # into each other's logits. It only stayed invisible because the head
            # was also 7.
            # T8.3: the literal was then num_experts, which is 1 for MoR while
            # step_experts holds oracle family indices 0..5. Read the width off
            # step_cls_out itself (== model.num_families) so the reshape cannot
            # disagree with the layer that produced the tensor.
            step_logits  = step_cls_out.reshape(-1, step_cls_out.shape[-1])[valid_mask]
            step_targets = step_experts.reshape(-1)[valid_mask]
            step_cls_loss = (
                F.cross_entropy(step_logits, step_targets)
                if step_logits.shape[0] > 0
                else torch.tensor(0.0, device=device)
            )
            # T-L5.3: on language `step_cls_out` is `family_probe(h.detach())`, so
            # this cross-entropy is `probe/family_ce` -- a READ-ONLY measurement of
            # how linearly decodable the POS family is from the trunk. It is
            # reported under its own key below and its weight is scientifically
            # inert: the detach means no value of `loss_weights.step_routing` can
            # change a single trunk gradient (verified bit-identically by
            # `test_lang_heads.py` TLH.3b). On arithmetic the same tensor comes from
            # `step_cls_head(h)`, is IN the objective, and does shape the trunk --
            # which is the confound §7.3 refuses to inherit.
            probe_family_ce = step_cls_loss
            # Blend: oracle CE (0.01) is secondary; step_cls CE (0.3) is primary
            step_routing_loss = 0.01 * oracle_routing_ce + 0.3 * step_cls_loss

            # 3b. Halting supervision against the operation-complexity
            #     curriculum (T3.3). Kept as its OWN term, never folded into the
            #     ponder cost: the ponder cost pushes depth down unconditionally
            #     while this pulls each token toward its curriculum target, so a
            #     combined number would hide which one the model is following
            #     (plan.md 5.4). Off by default -- pure ACT is the unsupervised
            #     path; enabling it is an explicit, provenance-recorded choice.
            if halting_supervision_enabled and expected_depth is not None:
                halt_sup_loss = halting_supervision_loss(
                    expected_depth, step_ops, step_mask,
                    _target_depth_table, mc["max_depth"],
                )
            else:
                halt_sup_loss = torch.zeros((), device=device)

            # 4. Routing balance + halting (halt weight linearly warmed up)
            current_step += 1
            current_halt_weight = min(
                target_halt_weight,
                target_halt_weight * (current_step / warmup_steps)
            )
            # T5.4: the L2-on-noise-scale term exists ONLY in the "trainable"
            # router-noise ablation, which is the only variant that HAS a noise
            # parameter. In canonical it is an exact differentiable zero and no
            # third pressure acts on the router. CLAUDE.md 2: this penalty is not
            # a proven fix for a growing noise scale and must not be called one.
            if any(hasattr(b.moe_block, "router_noise_scale") for b in model.blocks):
                noise_reg_loss = 0.001 * sum(
                    (block.moe_block.router_noise_scale ** 2)
                    for block in model.blocks
                    if hasattr(block.moe_block, "router_noise_scale")
                )
            else:
                noise_reg_loss = torch.zeros((), device=device)
            # T8.3: `0.5` was a bare literal here. It is the coefficient on the
            # 6-way whole-program FAMILY cross-entropy -- the second-largest term
            # in the objective -- and because it lived in source rather than in
            # config it appeared in no resolved_config, no provenance block and no
            # results table, so no run could be labelled by it and the only way to
            # ablate it was to edit this line (which produces an UNLABELLED
            # variant, forbidden by CLAUDE.md 6). It is now
            # loss_weights.family_cls, canonical 0.5, with 0.0 the T10.H
            # no_family_supervision ablation that resolve_variant tags.
            # NOTE it stays INSIDE the lw["task"] factor, exactly as the literal
            # did, so this change is numerically identical at the canonical value
            # and no existing run's loss is retroactively reinterpreted.
            total_loss = (
                lw["task"]              * (task_loss
                                           + lw["family_cls"] * cls_loss)
                + lw["step_routing"]   * step_routing_loss    # per-step oracle CE (direct)
                + lw["routing_balance"] * bal_loss            # maximise entropy, minimise collapse
                + current_halt_weight   * ponder_cost         # differentiable ACT ponder cost
                + lw["halting_supervision"] * halt_sup_loss   # curriculum target depth
                + noise_reg_loss
            )

            total_loss.backward()
            router_grad = model.blocks[0].moe_block.router.weight.grad
            if batch_idx == 0:
                # T5.4: report the scale that was ACTUALLY applied, via the
                # block's own accessor, so "none" prints 0.0000 instead of
                # crashing on a parameter that no longer exists in canonical.
                noise_scales = [
                    f"{b.moe_block.current_router_noise_scale():.4f}"
                    for b in model.blocks
                ]
                print(f"[Epoch {epoch:03d} Batch 0] Router grad norm: "
                      f"{router_grad.norm().item():.6f} | "
                      f"router_noise={model.router_noise} scales: {noise_scales}")
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            # Accumulate metrics
            bs = x.shape[0]
            epoch_total      += total_loss.item()
            epoch_task       += task_loss.item()
            # T8.3: the family CE was never accumulated, so `classification_loss`
            # -- which CLAUDE.md 4 names explicitly in the list of components that
            # must be tracked separately -- appeared in no metrics.json and no W&B
            # panel, even though at weight 0.5 it is the second-largest term in
            # total_loss. Its absence also made the T10.H ablation unreadable:
            # there was no series to compare against.
            epoch_cls        += cls_loss.item()
            epoch_bal        += bal_loss.item()
            epoch_halt       += ponder_cost.item()
            epoch_halt_sup   += halt_sup_loss.item()
            for _k, _v in batch_halt_stats.items():
                epoch_halt_stats[_k] = epoch_halt_stats.get(_k, 0.0) + _v
            _da, _dr = depth_allocation_error(
                expected_depth, step_ops, step_mask, _target_depth_table
            ) if expected_depth is not None else (None, None)
            if _da is not None:
                epoch_depth_abs_err += _da
                epoch_depth_rel_err += _dr
                epoch_depth_err_n   += 1
            epoch_step_route += step_routing_loss.item()
            epoch_probe_ce   += probe_family_ce.item()
            total_tokens     += bs
            if depth_exits is not None:
                depth_hist += depth_exits.detach().cpu().sum(dim=0)

        scheduler.step()
        elapsed    = time.perf_counter() - epoch_start
        throughput = total_tokens / max(elapsed, 1e-6)
        n_batches  = max(len(train_loader), 1)

        # ---- Validation + paper metrics --------------------------------
        val_loss = float("nan")
        paper_log_dict: dict = {}
        # T11.1 (audit item 7): validation-pass depth metrics. Declared out here,
        # not inside the `if`, so a non-logging epoch leaves it empty rather than
        # NameError-ing at the merge below.
        val_depth_log: dict = {}
        pim: dict = {}
        if val_loader is not None and (epoch % log["log_interval"] == 0 or epoch == epochs):
            model.eval()
            val_total = 0.0
            val_n     = 0
            # T5.1 (plan.md 7.1): every one of these is num_experts wide, not 7.
            # A 7x7 confusion matrix over 6 experts carries a permanently empty
            # row and column, which drags the diagonal fraction below the true
            # routing accuracy and breaks the CLAUDE.md 4 invariant that the two
            # must agree within tolerance. The family depth vectors had the same
            # phantom slot: `avg_depth_by_family` would have reported a 7th
            # family with count 0.
            #
            # T8.3: the two AXES are not the same width. Rows are indexed by the
            # ORACLE FAMILY (the dataset's label space, always 6); columns by the
            # PREDICTED EXPERT (a model property, 1 for MoR). Sizing the row axis
            # by num_experts made every MoR validation pass die with
            # "IndexError: index 4 is out of bounds for dimension 0 with size 1"
            # -- after two full epochs of training, so it burned the whole run.
            # For MoE/MoRE the matrix is square 6x6 exactly as before; only MoR
            # becomes rectangular, and both routing-metric functions already
            # return N/A there (num_experts < 2), so no number changes.
            _E = mc["num_experts"]
            _F = int(getattr(model, "num_families", NUM_EXPERTS_CANONICAL))
            confusion = torch.zeros(
                _F, _E, dtype=torch.float64
            )
            op_depth_sum   = torch.zeros(NUM_OP_TYPES, dtype=torch.float64)
            op_depth_count = torch.zeros(NUM_OP_TYPES, dtype=torch.float64)
            # Per-family depth is indexed by the oracle family too, and it is a
            # metric MoR genuinely has (MoR allocates depth; it just does not
            # route). It must therefore be _F wide, not _E.
            family_depth_sum   = torch.zeros(_F, dtype=torch.float64)
            family_depth_count = torch.zeros(_F, dtype=torch.float64)

            # T11.1 (audit item 7): depth metrics measured ON THE VALIDATION PASS.
            # Every depth number before this was `depth/*` and `train/avg_recursion
            # _steps`, accumulated inside the TRAINING loop -- i.e. under dropout,
            # on training data, with the halt head mid-update. A paper sentence of
            # the form "MoRE allocates depth in accordance with the curriculum" is
            # a claim about the trained model's behaviour on held-out data, which
            # the training-time number does not measure. The model already returns
            # `val_expected_depth` and `val_halt_stats` from this same forward pass
            # (they were unpacked and discarded), so this costs no extra compute --
            # only accumulation. The training-time keys are KEPT and unchanged: the
            # two are different measurements, not a correction of one by the other.
            val_depth_abs_err = 0.0
            val_depth_rel_err = 0.0
            val_depth_err_n   = 0
            val_exp_depth_sum = 0.0
            val_exp_depth_n   = 0
            val_halt_acc: dict[str, float] = {}
            val_halt_batches  = 0
            # T-L6.1: per-token exit depth, id and loss, accumulated over the whole
            # validation split so the correlations are over the split rather than over
            # whichever batch happened to be last.
            _lang_depth_ed: list = []
            _lang_depth_ids: list = []
            _lang_depth_loss: list = []
            _lang_route: list = []
            # All positions, unlike `_lang_depth_ids` which drops the last
            # of each block (no next-token target). The routing comparison
            # needs whole blocks, so it needs its own id vector.
            _lang_ids_full: list = []

            with torch.no_grad():
                for x_v, sm_v, se_v, so_v, fam_v, _, tgt_v in val_loader:
                    x_v, sm_v, se_v, so_v, fam_v, tgt_v = (
                        x_v.to(device), sm_v.to(device), se_v.to(device),
                        so_v.to(device), fam_v.to(device), tgt_v.to(device)
                    )
                    (
                        reg_v, _, _, _, _, depth_exits, _, _, _,
                        first_route, _val_route_stats,
                        val_expected_depth, val_halt_stats,
                    ) = model(x_v, sm_v, se_v, so_v)
                    l_v = (
                        F.cross_entropy(
                            reg_v[:, :-1, :].reshape(-1, reg_v.shape[-1]),
                            x_v[:, 1:].reshape(-1),
                        )
                        if _task == TASK_LANGUAGE
                        else F.mse_loss(reg_v.squeeze(-1), tgt_v)
                    )
                    val_total += l_v.item() * x_v.shape[0]
                    val_n     += x_v.shape[0]

                    # T-L6.1: exit depth, token id and that token's OWN loss, all
                    # over the identical token set. The set is the positions the loss
                    # is actually taken over -- `[:, :-1]`, since the last position
                    # has no next-token target -- so the three vectors describe the
                    # same population, which is what T-L6.1 requires.
                    #
                    # `reduction="none"` rather than a second forward pass, so the
                    # per-token losses are the same numbers the mean above came from.
                    if _task == TASK_LANGUAGE and depth_exits is not None:
                        _ed = compute_token_exit_depths(
                            depth_exits, mc["max_depth"]
                        ).reshape(x_v.shape[0], x_v.shape[1])
                        _pt = F.cross_entropy(
                            reg_v[:, :-1, :].reshape(-1, reg_v.shape[-1]),
                            x_v[:, 1:].reshape(-1),
                            reduction="none",
                        )
                        _lang_depth_ed.append(_ed[:, :-1].reshape(-1).cpu())
                        _lang_depth_ids.append(x_v[:, :-1].reshape(-1).cpu())
                        _lang_depth_loss.append(_pt.detach().cpu())

                    # Same helper the training loop calls (metrics.depth_allocation
                    # _error), so the two passes cannot disagree on the definition.
                    if val_expected_depth is not None:
                        _vda, _vdr = depth_allocation_error(
                            val_expected_depth, so_v, sm_v, _target_depth_table
                        )
                        if _vda is not None:
                            val_depth_abs_err += _vda
                            val_depth_rel_err += _vdr
                            val_depth_err_n   += 1
                        _vm = sm_v.reshape(-1)
                        if bool(_vm.any()):
                            val_exp_depth_sum += float(
                                val_expected_depth.reshape(-1)[_vm].sum().item()
                            )
                            val_exp_depth_n += int(_vm.sum().item())
                    if val_halt_stats:
                        for _k, _v in val_halt_stats.items():
                            val_halt_acc[_k] = val_halt_acc.get(_k, 0.0) + _v
                        val_halt_batches += 1

                    # T-L6.6 / T-L6.9: the first-step routing decision over WHOLE
                    # blocks, in split order. Whole blocks because the article-level
                    # reduction needs every token of a block, and `updated_rules.md` §8
                    # makes the FIRST step the authoritative routing decision, so this is
                    # the same signal the confusion matrix below is built from rather
                    # than a second opinion.
                    if _task == TASK_LANGUAGE and first_route is not None:
                        _lang_route.append(first_route.reshape(-1).detach().cpu())
                        _lang_ids_full.append(x_v.reshape(-1).detach().cpu())

                    if first_route is None or depth_exits is None:
                        continue

                    flat_mask   = sm_v.reshape(-1)
                    flat_oracle = se_v.reshape(-1)
                    flat_route  = first_route.reshape(-1)
                    flat_ops    = so_v.reshape(-1)
                    exit_depths = compute_token_exit_depths(
                        depth_exits, mc["max_depth"]
                    ).reshape(-1)

                    route_valid = flat_mask & (flat_oracle >= 0) & (flat_route >= 0)
                    if route_valid.any():
                        oracle_idx = flat_oracle[route_valid].long()
                        pred_idx   = flat_route[route_valid].long()
                        confusion.index_put_(
                            (oracle_idx, pred_idx),
                            torch.ones(oracle_idx.shape[0], dtype=torch.float64),
                            accumulate=True,
                        )

                    depth_valid = flat_mask & (flat_ops >= 0)
                    if depth_valid.any():
                        ops    = flat_ops[depth_valid].long().cpu()
                        depths = exit_depths[depth_valid].double().cpu()
                        for op_id, depth_val in zip(ops, depths):
                            op_depth_sum[op_id]   += depth_val
                            op_depth_count[op_id] += 1.0

                    family_valid = flat_mask & (flat_oracle >= 0)
                    if family_valid.any():
                        fams   = flat_oracle[family_valid].long().cpu()
                        depths = exit_depths[family_valid].double().cpu()
                        for fam_id, depth_val in zip(fams, depths):
                            family_depth_sum[fam_id]   += depth_val
                            family_depth_count[fam_id] += 1.0

            val_loss = val_total / max(val_n, 1)

            # T11.1 (audit item 7): the validation-pass depth block. Absent, never
            # 0.0, when the quantity does not exist -- `val_expected_depth` is None
            # for a non-adaptive model (the fixed_depth ablation), and a zero here
            # would read as perfect allocation. Keys are namespaced `val/` so no
            # reader can mistake them for the `depth/*` training-time numbers.
            if val_depth_err_n > 0:
                val_depth_log["val/depth_allocation_error_abs"] = (
                    val_depth_abs_err / val_depth_err_n
                )
                val_depth_log["val/depth_allocation_error_rel"] = (
                    val_depth_rel_err / val_depth_err_n
                )
            # T-L6.0: on LANGUAGE these two keys are structurally undefined, not
            # merely unmeasured. There is no per-token ground-truth depth for English
            # (`plan_language.md` §5.1), `lang_families.op_target_depth_table()` is
            # empty by construction, so `val_depth_err_n` is 0 and the branch above
            # never fires. Writing "N/A" makes the absence a STATEMENT rather than a
            # gap: an exporter joining arithmetic and language rows on a common column
            # set has to decide what a missing column means, and the whole point of
            # "N/A" is that it never has to. A 0.0 or -1 here would read as perfect
            # allocation against a curriculum that does not exist -- the single most
            # misleading number this migration could produce.
            if _task == TASK_LANGUAGE:
                val_depth_log["val/depth_allocation_error_abs"] = "N/A"
                val_depth_log["val/depth_allocation_error_rel"] = "N/A"
            if val_exp_depth_n > 0:
                val_depth_log["val/avg_recursion_steps"] = (
                    val_exp_depth_sum / val_exp_depth_n
                )
            _vfe = val_halt_acc.get("forced_exits", 0.0)
            _vee = val_halt_acc.get("early_exits",  0.0)
            if (_vfe + _vee) > 0:
                val_depth_log["val/forced_exit_rate"] = _vfe / (_vfe + _vee)
                val_depth_log["val/early_exit_rate"]  = _vee / (_vfe + _vee)
                val_depth_log["val/mean_remainder"]   = (
                    val_halt_acc.get("mean_remainder", 0.0)
                    / max(val_halt_batches, 1)
                )

            # T-L6.6 / T-L6.9: BOTH reference axes, each against its own null. Wired
            # here because a bare `routing_ami` is uninterpretable: the AMI floor for a
            # 6-way partition with these marginals is ~0.05, not 0, so an unaccompanied
            # 0.06 reads as weak specialization when it is at chance. POS first (§4.4
            # makes it primary), topic second and labelled INDUCED -- the question is
            # "what does it organize by", not "does it reproduce POS", so both always.
            if (_task == TASK_LANGUAGE and _lang_route
                    and mc["num_experts"] > 1):
                try:
                    _route = torch.cat(_lang_route).numpy()
                    _ids_full = torch.cat(_lang_ids_full).numpy()
                    if _pos_control is not None and _tok_family is not None:
                        _pos_cmp = specialization_vs_control(
                            _route, _ids_full, _tok_family, _pos_control,
                            mc["num_experts"])
                        val_depth_log.update(control_comparison_to_wandb(
                            _pos_cmp, prefix="val/routing_control_pos"))
                    if _block_topic is not None:
                        _nb = _route.size // int(dc["seq_len"])
                        _top_cmp = article_agreement_vs_topic(
                            _route, _block_topic[:_nb],
                            int(dc["seq_len"]), mc["num_experts"], n_draws=10,
                            seed=seed)
                        val_depth_log.update(control_comparison_to_wandb(
                            _top_cmp, prefix="val/routing_control_topic"))
                except Exception as _exc:      # pragma: no cover
                    print(f"[Train] reference-axis comparison unavailable: "
                          f"{type(_exc).__name__}: {_exc}", file=sys.stderr)
                    val_depth_log["val/routing_control_error"] = (
                        f"{type(_exc).__name__}: {_exc}")

            # T-L6.1 / T-L6.2: the correlational depth report that REPLACES
            # allocation error on language. Computed once per validation pass over the
            # whole split, each rho with its permutation null band -- a near-constant
            # depth vector can produce a nonzero rho from tie-breaking alone, so a
            # depth correlation without its null is uninterpretable.
            _lang_depth: dict = {}
            if _task == TASK_LANGUAGE and _lang_depth_ed:
                try:
                    _lang_depth = language_depth_metrics(
                        torch.cat(_lang_depth_ed).numpy(),
                        torch.cat(_lang_depth_ids).numpy(),
                        torch.cat(_lang_depth_loss).numpy(),
                        _tok_family, _log_freq, _surprisal,
                        mc["max_depth"], _NUM_LANG_FAMILIES, _LANG_FAMILY_LABELS,
                        seed=seed,
                    )
                    val_depth_log.update(language_depth_to_wandb(_lang_depth))
                except Exception as _exc:      # pragma: no cover
                    # Reported, not swallowed: a failed depth report must not take
                    # the run down, but it must not look like a run without one.
                    print(f"[Train] language depth metrics unavailable: "
                          f"{type(_exc).__name__}: {_exc}", file=sys.stderr)
                    val_depth_log["val/depth_report_error"] = (
                        f"{type(_exc).__name__}: {_exc}"
                    )

            # The single definition of routing accuracy (T6.2), so the scalar
            # and the confusion matrix cannot drift. Returns
            # NaN -> "N/A" when num_experts < 2, because MoR makes no routing
            # decision and its 0.1005 was not a routing measurement.
            routing_acc = routing_accuracy_from_confusion(
                confusion, mc["num_experts"]
            )
            # T6.3 (updated_rules.md §8.3): the raw diagonal answers "did the
            # router pick the oracle's own index for this family?". Expert indices
            # carry no semantic identity, so a perfect but relabelled partition
            # scores 0.0 there. Computed once here, from the same matrix, and
            # passed down so W&B, metrics.json and the console line cannot report
            # three different numbers.
            pim = permutation_invariant_routing_metrics(
                confusion, mc["num_experts"]
            )
            op_avg_depth: dict[str, float] = {}
            for i, name in enumerate(ALL_OP_NAMES):
                if op_depth_count[i] > 0:
                    op_avg_depth[name] = (
                        op_depth_sum[i] / op_depth_count[i]
                    ).item()
            family_avg_depth: dict[str, float] = {}
            # T5.1: labels derive from the one manifest helper, never from a
            # literal. T8.3: from the FAMILY label space (_F), not num_experts --
            # at E=1 an _E-wide vector indexed out of range, and at E=7 it
            # invented a seventh family. These are oracle family labels, so all
            # three architectures report the same six rows.
            #
            # T-L6.8: and from the TASK's manifest. Before this fix a language run
            # published `recursion/avg_depth_by_family/E1_ADD_SUB` -- arithmetic labels
            # on language values -- because `expert_labels` is imported from
            # `.families` at module scope. The numbers were right and the row names
            # were from the other study, which is exactly the kind of thing that
            # reaches a paper table unnoticed. Found in the first completed language
            # run, `runs/langB_MoRE_seed42__91c9bba1`.
            for i, label in enumerate(_expert_labels(_F)):
                if family_depth_count[i] > 0:
                    family_avg_depth[label] = (
                        family_depth_sum[i] / family_depth_count[i]
                    ).item()

            paper_log_dict = paper_metrics_to_wandb(
                confusion.numpy(),
                routing_acc,
                op_avg_depth,
                family_avg_depth,
                mc["num_experts"],
                pim=pim,
                task=_task,
            )

        # ---- Expert metrics (no grad) ----------------------------------
        with torch.no_grad():
            # Both are None when num_experts < 2 (the MoR baseline): a single
            # expert has no pair, so the statistic does not exist (T6.5).
            mean_cos, max_cos = compute_pairwise_cosine_sim(model)

        # Depth histogram as fractions
        depth_total = depth_hist.sum().clamp(min=1.0)
        depth_frac  = (depth_hist / depth_total).tolist()

        # --- Compute Shannon entropy from epoch_expert_counts -----------
        # epoch_expert_counts holds the hard-argmax token assignments summed
        # over ALL depth steps of ALL batches this epoch — activating this
        # block gives us:
        #   (a) per_expert_load[e]: fraction of all routed tokens → expert e
        #   (b) load_entropy: Shannon entropy of the load distribution
        total_expert_calls = epoch_expert_counts.sum().clamp(min=1.0)
        per_expert_load    = (epoch_expert_counts / total_expert_calls).cpu()  # [E]

        # load_entropy is NORMALIZED (H / log E) and is None for E = 1 or when
        # no tokens were routed. It is a load-balance diagnostic, not a quality
        # metric, and 0.0 must never be substituted for "undefined" (T6.4).
        load_entropy = None
        if epoch_all_expert_idx:
            _ent = compute_expert_load_entropy(
                depth_exits,            # used only for .device; last batch is fine
                epoch_all_expert_idx,   # every per-depth argmax tensor this epoch
                mc["num_experts"],
            )
            load_entropy = None if _ent is None else _ent.item()

        # ---- W&B logging -----------------------------------------------
        log_dict = {
            "train/total_loss":          epoch_total      / n_batches,
            "train/task_loss":           epoch_task       / n_batches,
            # T8.3: the whole-program family CE, reported under BOTH the name
            # CLAUDE.md 4 uses ("classification_loss") and the name of the config
            # key that weights it ("family_cls_loss"), so a reader can go from the
            # chart to loss_weights.family_cls without guessing. This is the raw,
            # UNWEIGHTED component; the weight is in provenance as
            # family_cls_weight.
            "train/classification_loss": epoch_cls        / n_batches,
            "train/family_cls_loss":     epoch_cls        / n_batches,
            "train/aux_routing_loss":    epoch_bal        / n_batches,
            # T4.1: this is now a MEAN over (block, depth) calls, not a sum, so
            # it is comparable across max_depth and num_blocks settings.
            # `aux_routing_loss` is kept as the historical alias.
            "train/routing_balance_loss": epoch_bal       / n_batches,
            # CLAUDE.md 4 requires ponder_cost and halting_supervision_loss be
            # reported SEPARATELY. `train/halting_loss` is retained as an alias
            # for train/ponder_cost so existing readers keep working, but the
            # unambiguous names are the two below it.
            "train/halting_loss":            epoch_halt       / n_batches,
            "train/ponder_cost":             epoch_halt       / n_batches,
            # T11.0b: N/A, not 0.0, when the term was never computed. This key
            # read 0.0 in all 15 canonical Phase B runs -- halting supervision is
            # off in canonical (loss_weights.halting_supervision = 0.0), so
            # epoch_halt_sup was never accumulated and the division produced a
            # clean zero that is indistinguishable from "supervised, and the
            # predicted depths matched the curriculum exactly". Opposite
            # meanings, same rendering (CLAUDE.md 4). The gate is the enabling
            # flag, not the value, so a genuinely-converged 0.0 still prints 0.0.
            "train/halting_supervision_loss": (
                epoch_halt_sup / n_batches
                if halting_supervision_enabled else "N/A"
            ),
            "train/step_routing_loss":   epoch_step_route / n_batches,
            # T-L5.3. On language this is a read-only probe on h.detach(),
            # so it reports how linearly decodable the POS family is and
            # cannot have created that decodability. On arithmetic the same
            # CE IS in the objective at weight 0.5, so the two are published
            # under different names to keep the distinction visible.
            ("probe/family_ce" if _task == TASK_LANGUAGE
             else "train/step_cls_ce"): epoch_probe_ce / n_batches,
            "train/avg_recursion_steps": (
                avg_depth.item() if avg_depth is not None else float("nan")
            ),
            "perf/throughput_tokens_sec":   throughput,
            "epoch": epoch,
        }

        # Undefined metrics are OMITTED from the W&B payload rather than logged
        # as 0.0 or -1.0. A charted zero is indistinguishable from a measured
        # zero, which is precisely how a sentinel becomes a published number
        # (CLAUDE.md 4). metrics.json records the string "N/A" instead.
        if load_entropy is not None:
            log_dict["train/expert_load_entropy_normalized"] = load_entropy
        if max_cos is not None:
            log_dict["diag/max_pairwise_cosine_sim"]  = max_cos
            log_dict["diag/mean_pairwise_cosine_sim"] = mean_cos

        # ---- T2.3 dispatch / capacity accounting ---------------------------
        # Every one of these is a real measurement. Under
        # CAPACITY_POLICY = "no_capacity_limit" the overflow count is a measured
        # zero -- nothing is dropped -- which plan.md 4.3 requires be stated
        # explicitly rather than left silent. max_load_fraction is the real
        # capacity pressure: the largest share of tokens any single expert took.
        if epoch_route_stats.get("dispatched", 0.0) > 0:
            _d = epoch_route_stats["dispatched"]
            log_dict["dispatch/tokens_dispatched"]  = _d
            log_dict["dispatch/overflow_tokens"]    = epoch_route_stats["overflow"]
            log_dict["dispatch/overflow_rate"]      = epoch_route_stats["overflow"] / _d
            log_dict["dispatch/expert_evaluations"] = epoch_route_stats["expert_evaluations"]
            log_dict["dispatch/evals_per_token"]    = (
                epoch_route_stats["expert_evaluations"] / _d
            )
            log_dict["dispatch/max_load_fraction"]  = epoch_route_stats["max_load_fraction"]
            log_dict["dispatch/experts_called_max"] = epoch_route_stats["experts_called"]
            log_dict["dispatch/capacity_policy"]    = CAPACITY_POLICY
            log_dict["dispatch/routing_mode"]       = mc.get(
                "routing_mode", CANONICAL_ROUTING_MODE
            )
            # T5.4: which noise variant ran, and the scale actually applied this
            # epoch. Canonical logs "none" / 0.0; the annealed ablation logs a
            # decaying number, which is the only way to tell from the logs
            # whether exploration was still active when a metric was recorded.
            log_dict["dispatch/router_noise"] = mc.get(
                "router_noise", CANONICAL_ROUTER_NOISE
            )
            log_dict["dispatch/router_noise_scale"] = max(
                b.moe_block.current_router_noise_scale() for b in model.blocks
            )

        # ---- T3.1 / T3.4 halting diagnostics -------------------------------
        # These are the externally visible evidence that ACT halting is real:
        # a mix of early and forced exits (never 100% forced -- that would mean
        # every token ran to max_depth and halting decided nothing), and a
        # remainder distribution that is not degenerate. The exit RATES are
        # recomputed from the summed counts, not averaged from the per-batch
        # rates, because a mean-of-ratios is not the ratio over the epoch.
        _fe = epoch_halt_stats.get("forced_exits", 0.0)
        _ee = epoch_halt_stats.get("early_exits",  0.0)
        _tot_exits = _fe + _ee
        if _tot_exits > 0:
            log_dict["halt/forced_exits"]     = _fe
            log_dict["halt/early_exits"]      = _ee
            log_dict["halt/forced_exit_rate"] = _fe / _tot_exits
            log_dict["halt/early_exit_rate"]  = _ee / _tot_exits
            log_dict["halt/mean_remainder"]   = (
                epoch_halt_stats.get("mean_remainder", 0.0) / n_batches
            )
            log_dict["halt/mean_halt_mass"]   = (
                epoch_halt_stats.get("mean_halt_mass", 0.0) / n_batches
            )

        # ---- T3.3 depth allocation error -----------------------------------
        # CLAUDE.md 4: this is DEPTH ALLOCATION ERROR against the predefined
        # operation-complexity curriculum in families.OP_TARGET_DEPTH. It is NOT
        # "compute efficiency" -- no FLOPs are in it. Absent (N/A) rather than
        # 0.0 when no batch contained a labelled operation step, because a zero
        # here would read as perfect allocation.
        if epoch_depth_err_n > 0:
            log_dict["depth/allocation_error_abs"] = (
                epoch_depth_abs_err / epoch_depth_err_n
            )
            log_dict["depth/allocation_error_rel"] = (
                epoch_depth_rel_err / epoch_depth_err_n
            )

        # ---- T4.2 balance-objective decomposition --------------------------
        # CLAUDE.md 4: report the entropy term and the Switch-style auxiliary
        # term SEPARATELY. `routing_balance_loss = -entropy_term +
        # switch_aux_term`, and the two pull in opposite directions, so the
        # single number can sit flat while both components drift a long way.
        # Both arrive already averaged over (block, depth) calls (T4.1); the
        # division here is only over batches.
        #
        # `balance_to_task_ratio` is the T4.2 acceptance measurement: the
        # WEIGHTED contribution of the balance term relative to the task loss.
        # plan.md 6.2 requires the balance objective be "clearly subordinate to
        # task learning" -- it was ~146x larger before normalization. Logged as
        # a ratio rather than left to be eyeballed from two separate charts.
        if mc["num_experts"] > 1 and "entropy_term" in epoch_route_stats:
            _ent_t = epoch_route_stats["entropy_term"]    / n_batches
            _sw_t  = epoch_route_stats["switch_aux_term"] / n_batches
            log_dict["train/entropy_term"]    = _ent_t
            log_dict["train/switch_aux_term"] = _sw_t
            # Normalized as CLAUDE.md 4 requires: H / log(E), so it is
            # comparable across expert counts. E == 1 is excluded above (log 1
            # = 0); for E == 1 entropy is N/A, never a comparative 0.0.
            log_dict["train/entropy_term_normalized"] = (
                _ent_t / math.log(mc["num_experts"])
            )
        _task_mag = abs(epoch_task / n_batches)
        if lw["routing_balance"] != 0.0 and _task_mag > 0.0:
            log_dict["train/balance_to_task_ratio"] = (
                abs(lw["routing_balance"] * (epoch_bal / n_batches)) / _task_mag
            )

        # Per-expert load distribution (from the now-activated epoch_expert_counts)
        for e in range(mc["num_experts"]):
            log_dict[f"expert_load/expert_{e}_pct"] = per_expert_load[e].item() * 100.0

        # Depth distribution histogram
        depth_hist_wandb = {
            f"depth_dist/step_{d+1}_pct": depth_frac[d] * 100.0
            for d in range(mc["max_depth"])
        }
        log_dict.update(depth_hist_wandb)

        if not math.isnan(val_loss):
            # CLAUDE.md 4: the PRIMARY predictive metric is validation TASK
            # loss, and total_loss must never be reported as the quality metric.
            # `val_loss` above is pure F.mse_loss on the regression head -- no
            # auxiliary, balance, halting or routing-supervision term is in it.
            # `val/loss` is kept as an alias so existing readers do not break,
            # but `val/task_loss` is the name that cannot be misread as a total.
            log_dict["val/loss"]      = val_loss
            log_dict["val/task_loss"] = val_loss
            # T-L5.1 / T-L5.4: perplexity is a MONOTONE TRANSFORM of the loss and
            # carries no extra information, so it is reported and never optimised
            # and never used for checkpoint selection -- the verdict is always
            # taken on nats. Emitted only on language, because exp() of an
            # arithmetic MSE is not a perplexity and a column that silently meant
            # two things would be worse than a missing one.
            if _task == TASK_LANGUAGE and math.isfinite(val_loss):
                log_dict["val/perplexity"] = math.exp(val_loss)
                log_dict["val/bits_per_token"] = val_loss / math.log(2)
                # The distance from the trivial floor, which is the only form in
                # which a language loss number means anything (T-L2.5). Absent
                # rather than 0.0 when the floor was not supplied.
                if _floor is not None:
                    log_dict["val/nats_below_bigram_floor"] = _floor - val_loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), ctx.path("checkpoint.pt"))

        log_dict.update(paper_log_dict)
        log_dict.update(val_depth_log)

        wandb.log(log_dict, step=epoch)

        routing_acc_str = (
            f"{paper_log_dict[_agree_key]:.4f}"
            if _agree_key in paper_log_dict
            else "N/A"
        )

        # One renderer for every metric that can be undefined. "N/A" is a fact
        # ("this quantity does not exist for this architecture"); 0.0 or -1.0 in
        # the same column is a fabricated measurement (CLAUDE.md 4).
        def _na(v, fmt=".4f") -> str:
            return "N/A" if v is None else format(v, fmt)

        entropy_str  = _na(load_entropy)
        max_cos_str  = _na(max_cos)
        mean_cos_str = _na(mean_cos)
        depth_str    = _na(avg_depth.item() if avg_depth is not None else None, ".2f")
        # T6.3: the two permutation-invariant headline numbers travel in the
        # per-epoch table beside the raw accuracy, not only in W&B. The exporter
        # reads this file, and a reviewer comparing MoE with MoRE needs the matched
        # number in the same row as the raw one -- quoting one without the other is
        # what updated_rules.md §8.3 forbids.
        hung_str = _na(pim.get("hungarian_accuracy"))
        ami_str  = _na(pim.get("ami"))

        # Append to results.tsv (per-run; plan.md 2.2)
        # T9.0: an unevaluated epoch writes the string "N/A", not `nan`. Nothing
        # was ever *reported* as a measurement here (the `isnan` guard below
        # keeps it out of best_val_loss), but results.tsv used `nan` for "not
        # evaluated" while metrics.json uses "N/A" for the same idea. Two
        # conventions for "no number here" in one run directory is what a
        # plotting script misreads as a measured zero or a diverged loss.
        val_str = "N/A" if math.isnan(val_loss) else f"{val_loss:.6f}"
        # T-L5.4: the three language columns are APPENDED, never inserted. The
        # archived arithmetic `results.tsv` files and every existing reader index by
        # POSITION, so reordering would silently reinterpret published columns.
        # Each is "N/A" rather than 0.0 on arithmetic: exp() of an MSE is not a
        # perplexity, and a 0.0 in a perplexity column is a fabricated measurement
        # (CLAUDE.md §4).
        _ppl_str = (
            f"{math.exp(val_loss):.4f}"
            if (_task == TASK_LANGUAGE and not math.isnan(val_loss))
            else "N/A"
        )
        _below_str = (
            f"{_floor - val_loss:+.6f}"
            if (_task == TASK_LANGUAGE and _floor is not None
                and not math.isnan(val_loss))
            else "N/A"
        )
        # depth_rho_model_loss is Phase L-6's measurement (correlation between the
        # depth a token received and the loss it incurred). The column exists now so
        # the header never has to be reordered later; it reads "N/A" until L-6
        # computes it, which is the honest value for "not measured yet".
        _rho_str = _na(val_depth_log.get("val/depth_rho_model_loss"), ".6f")
        results_rows.append(
            f"{epoch}\t{epoch_task/n_batches:.6f}\t{val_str}\t"
            f"{entropy_str}\t{depth_str}\t"
            f"{mean_cos_str}\t{max_cos_str}\t{routing_acc_str}\t"
            f"{hung_str}\t{ami_str}\t"
            f"{_ppl_str}\t{_below_str}\t{_rho_str}"
        )

        # Flush results to disk every epoch so a partial run is still readable.
        # T-L3.2: the routing column is NAMED from `_agree_key`, so a language
        # results.tsv reads `routing_agreement_with_pos` and an arithmetic one
        # reads `routing_accuracy` -- and no exporter can join the two columns by
        # position and silently compare an agreement figure against an accuracy.
        with open(ctx.path("results.tsv"), "w") as f:
            f.write(
                "epoch\ttrain_task_loss\tval_loss\t"
                "expert_entropy_normalized\tavg_depth\t"
                "mean_cos_sim\tmax_cos_sim\t"
                f"{_agree_key.split('/', 1)[1]}\t"
                "routing_hungarian_acc\trouting_ami\t"
                # T-L5.4: APPENDED, never inserted -- see the row builder above for
                # why position matters to the archived arithmetic files.
                "val_perplexity\tnats_below_bigram_floor\tdepth_rho_model_loss\n"
            )
            f.write("\n".join(results_rows) + "\n")

        if epoch % log["log_interval"] == 0 or epoch == epochs:
            # T8.1: `bal` and `halt` go through the same _na renderer as the
            # routing columns. The console previously printed MoR's balance term
            # as bal=1.0000 and MoE's ponder cost as halt=1.0000 while
            # metrics.json wrote "N/A" for both -- the artifact and the log
            # disagreed about the same quantity, and 1.0000 in a console column
            # next to MoRE's -0.6128 reads as "MoR is maximally imbalanced".
            # Both are constants of E == 1 / max_depth == 1, not measurements.
            _bal_defined  = mc["num_experts"] > 1
            _halt_defined = mc["max_depth"] > 1 and mc.get("adaptive_halting", True)
            print(
                f"[Epoch {epoch:03d}/{epochs}] "
                f"task={epoch_task/n_batches:.4f}  "
                f"step_route={epoch_step_route/n_batches:.4f}  "
                f"bal={_na(epoch_bal/n_batches if _bal_defined else None)}  "
                f"halt={_na(epoch_halt/n_batches if _halt_defined else None)}  "
                f"val={val_loss:.4f}  "
                f"route_acc={routing_acc_str}  "
                f"route_hung={hung_str}  route_ami={ami_str}  "
                f"entropy_norm={entropy_str}  "
                f"cos_max={max_cos_str}  "
                f"tok/s={throughput:.0f}"
            )

    print(f"\n[Train] Complete. Best val_loss = {best_val_loss:.6f}")

    # Final metric dump -> runs/<experiment_id>/metrics.json  (plan.md 2.2)
    try:
        metrics = dict(log_dict)
        # Explicit N/A in the artifact, so a downstream exporter cannot mistake
        # an absent key for an unrecorded run (CLAUDE.md 4: emit N/A, never
        # fabricate).
        if load_entropy is None:
            metrics["train/expert_load_entropy_normalized"] = "N/A"
        if max_cos is None:
            metrics["diag/max_pairwise_cosine_sim"]  = "N/A"
            metrics["diag/mean_pairwise_cosine_sim"] = "N/A"
        # Depth allocation error is undefined, not zero, when no labelled
        # operation step was seen. 0.0 would read as perfect allocation.
        if epoch_depth_err_n == 0:
            metrics["depth/allocation_error_abs"] = "N/A"
            metrics["depth/allocation_error_rel"] = "N/A"
        # MoE runs at max_depth == 1: every token forced-exits at step 1, so
        # "early exit rate" is 0 by construction and is not a measurement of
        # halting behaviour. Report N/A rather than a comparable-looking zero.
        if mc["max_depth"] <= 1 or not mc.get("adaptive_halting", True):
            # T11.0b: the two raw COUNTS were missing from this list while the
            # four derived rates were in it, so a MoE metrics.json carried
            # halt/forced_exits as a real integer (every token, forced at step 1)
            # beside halt/forced_exit_rate = "N/A". A count is as much a halting
            # measurement as a rate; at max_depth == 1 neither is one, because
            # the exit is a property of the configuration and nothing was
            # decided. Emitting the count invites an exporter to divide it by
            # itself and re-derive the very rate this block refuses to state.
            for _k in ("halt/forced_exits", "halt/early_exits",
                       "halt/forced_exit_rate", "halt/early_exit_rate",
                       "halt/mean_remainder", "halt/mean_halt_mass"):
                metrics[_k] = "N/A"
            # At max_depth == 1 the ponder cost is (N + R) / (max_depth + 1) =
            # (1 + 1) / 2 = 1.0 for EVERY token, always. That is a constant of
            # the configuration, not a measured cost, and 1.0 sitting in a
            # ponder-cost column beside MoR's 0.27 and MoRE's 0.47 would read as
            # "MoE ponders hardest" -- exactly the sentinel-as-measurement
            # failure of T6.4 (CLAUDE.md 4). Same for depth allocation error:
            # with depth pinned to 1 the error is a property of the dataset's
            # operation mix, not of anything the model allocated.
            for _k in ("train/ponder_cost", "train/halting_loss",
                       "depth/allocation_error_abs", "depth/allocation_error_rel"):
                metrics[_k] = "N/A"
            # T11.1 (audit item 7): the validation-pass twins of the same four.
            # `val_expected_depth` is None at max_depth == 1 / non-adaptive, so
            # these keys are simply absent from log_dict there; stating "N/A"
            # explicitly keeps "does not apply to this architecture" distinct from
            # "this run predates the metric", which is the distinction the
            # exporter's absent-vs-N/A boundary rests on.
            for _k in ("val/depth_allocation_error_abs",
                       "val/depth_allocation_error_rel",
                       "val/avg_recursion_steps",
                       "val/forced_exit_rate", "val/early_exit_rate",
                       "val/mean_remainder"):
                metrics[_k] = "N/A"
        metrics["halting_mode"] = halting_mode
        # T4.1/T4.2: with a single expert there is nothing to balance. The
        # objective still evaluates to a fixed constant -- entropy of a
        # one-element distribution is 0 and the Switch term is E * 1 * 1 = 1, so
        # routing_balance_loss is exactly 1.0 for every batch, forever. That is a
        # property of E == 1, not a measured load imbalance, and MoR's weight is
        # already 0.0 so it never enters the objective. Tabled next to MoRE's
        # value it would read as "MoR is maximally imbalanced" (CLAUDE.md 4).
        if mc["num_experts"] <= 1:
            for _k in ("train/routing_balance_loss", "train/aux_routing_loss",
                       "train/entropy_term", "train/switch_aux_term",
                       "train/entropy_term_normalized",
                       "train/balance_to_task_ratio"):
                metrics[_k] = "N/A"
            # T8.1: the FLAT routing keys were simply ABSENT from a MoR
            # metrics.json, while the nested routing_permutation_invariant block
            # wrote "N/A" for the same quantities. Two conventions for one fact
            # in one file: the nested block's own comment states the rule ("the
            # field list is the same for every architecture and a missing key can
            # never be mistaken for an unrecorded run") and the flat keys did not
            # follow it. Absence is not fabrication, so this was not a wrong
            # number -- but an exporter joining three architectures on a common
            # column set has to decide what a missing column means, and the whole
            # point of writing "N/A" is that it never has to.
            for _k in (_agree_key, "val/routing_hungarian_accuracy",
                       "val/routing_ami", "val/routing_purity",
                       "val/routing_macro_recall",
                       "val/routing_matched_macro_recall"):
                metrics[_k] = "N/A"
        # The normalizer that makes the term depth-invariant, recorded so a
        # reader can confirm which aggregation produced the number.
        metrics["routing_balance_normalization"] = (
            "mean over depth calls per block, then mean over blocks"
        )
        # T-L3.2: the caption travels WITH the number, in the artifact the exporter
        # and the paper draft read -- not only on the console, which nothing keeps.
        # `plan_language.md` §4.4 requires the language figure to be captioned as
        # agreement with a linguistic prior rather than as accuracy, and a caption
        # that lives in a prose file is a caption that can be omitted from a table.
        metrics["routing_agreement_metric_key"] = _agree_key
        # T-L6.8: four key families all read as "the exit-depth distribution" and two of
        # them differ in the fifth decimal. Renaming is not available -- two of the four
        # are in the arithmetic contract Gate L0 checks -- so the provenance ships AS DATA
        # in every run directory, and `uncovered_depth_keys` refuses to let a new depth key
        # escape it. Written for both tasks: the ambiguity is not language-specific.
        metrics["depth_key_provenance"] = DEPTH_KEY_PROVENANCE
        _uncov = uncovered_depth_keys(metrics.keys())
        if _uncov:
            # Recorded rather than raised: a run must not die because a metric name is
            # undocumented, but it must not look documented either.
            metrics["depth_key_provenance_uncovered"] = _uncov
            print(f"[Train] depth keys with no provenance entry: {_uncov} -- add them to "
                  f"metrics.DEPTH_KEY_PROVENANCE", file=sys.stderr)
        metrics["routing_agreement_caption"] = routing_agreement_caption(_task)
        # T6.3: the full permutation-invariant routing report, in the artifact the
        # exporter reads. Written even when undefined (E == 1) so the field list is
        # the same for every architecture and a missing key can never be mistaken
        # for an unmeasured one.
        _pim_out = {}
        for _k in ("raw_accuracy", "hungarian_accuracy", "macro_recall",
                   "matched_macro_recall", "ami", "purity"):
            _v = pim.get(_k)
            _pim_out[_k] = "N/A" if _v is None else _v
        _pim_out["hungarian_assignment"] = (
            pim.get("hungarian_assignment") or "N/A"
        )
        _pim_out["collapsed_experts"] = (
            pim.get("collapsed_experts") if pim.get("collapsed_experts") else
            ("none" if pim.get("hungarian_accuracy") is not None else "N/A")
        )
        _pim_out["per_family"] = {
            _lab: {_s: ("N/A" if _v is None else _v) for _s, _v in _st.items()}
            for _lab, _st in (pim.get("per_family") or {}).items()
        } or "N/A"
        # T6.8: the relabelling-invariant per-family table. Written beside the
        # identity one, never in place of it, so the exporter can choose by
        # supervision mode instead of guessing.
        _pim_out["per_family_matched"] = {
            _lab: {_s: ("N/A" if _v is None else _v) for _s, _v in _st.items()}
            for _lab, _st in (pim.get("per_family_matched") or {}).items()
        } or "N/A"
        _pim_out["note"] = (
            "expert indices carry no semantic identity; read the gap between "
            "raw_accuracy and hungarian_accuracy, not either alone "
            "(updated_rules.md 8.3). per_family is under the IDENTITY mapping "
            "and is only interpretable when step_routing supervised the index; "
            "per_family_matched applies hungarian_assignment first and is the "
            "table to report for an unsupervised router (T6.8)."
        )
        metrics["routing_permutation_invariant"] = _pim_out
        metrics["best_val_loss"]   = best_val_loss
        metrics["experiment_id"]   = ctx.experiment_id
        metrics["experiment_group"] = ctx.experiment_group
        # Architecture and expert count belong IN the artifact, not only in the
        # run name: --run_name is free text, so a directory called "MoRE_..."
        # proves nothing about what trained. Anything reading metrics.json to
        # build a table needs to know which architecture produced it, and
        # num_experts is what decides whether the balance keys above are
        # measurements or "N/A" (CLAUDE.md 5, provenance).
        metrics["architecture"]    = ctx.resolved_cfg.get("architecture")
        metrics["num_experts"]     = mc["num_experts"]
        # T5.4: the routing path travels with the numbers. Without these, two
        # metrics.json files differing only by "noise was on" are
        # indistinguishable to any exporter reading them.
        metrics["routing_mode"]    = mc.get("routing_mode", CANONICAL_ROUTING_MODE)
        metrics["router_noise"]    = mc.get("router_noise", CANONICAL_ROUTER_NOISE)
        metrics["config_hash"]     = ctx.resolved_cfg.get("provenance", {}).get("config_hash")
        # T6.1: which ops actually ran non-deterministically, as observed rather
        # than as predicted. seed-time `determinism_exceptions` cannot know this:
        # a kernel without a deterministic implementation only warns when it is
        # first called, mid-training.
        nondet = nondeterministic_ops_observed()
        metrics["nondeterministic_ops_observed"] = nondet
        ctx.resolved_cfg.setdefault("provenance", {})[
            "nondeterministic_ops_observed"] = nondet
        ctx.write_resolved_config()
        if nondet != ["none"]:
            print("[Seed] ops without a deterministic implementation were used "
                  f"in this run: {', '.join(nondet)}. Losses may not reproduce "
                  "bitwise; this is recorded in provenance rather than hidden.")
        written = ctx.write_metrics(metrics)
        print(f"[Run] metrics written to {written}")
    except Exception as e:
        print(f"[WARN] Failed to write metrics.json: {e}")

    wandb.finish()
    if owns_ctx:
        ctx.close()
    return best_val_loss


