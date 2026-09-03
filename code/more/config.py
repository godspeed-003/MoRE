"""config.py - Config loading and precedence.

Extracted from the former monolithic train.py (plan.md 9: ONE training
system, split into readable modules; not parallel implementations).
"""

import os
import json

# ---------------------------------------------------------------------------
# 0. Configuration loader
# ---------------------------------------------------------------------------

from .model import (CANONICAL_ROUTING_MODE, DENSE_ABLATION_ROUTING_MODE,
                    ROUTING_MODES, CANONICAL_ROUTER_NOISE, ROUTER_NOISE_MODES,
                    ROUTER_NOISE_INIT_SCALE_DEFAULT,
                    ROUTER_NOISE_ANNEAL_STEPS_DEFAULT)
from .families import NUM_EXPERTS_CANONICAL

# The canonical expert count, E1..E6; the E7 catch-all is removed
# (updated_rules.md 2, plan.md 7.2). Defined HERE, above the defaults that use
# it, and taken from the one family manifest rather than restated as a literal
# -- T5.1: a second copy of the number is how `num_experts` and the family
# label list drift apart in the first place.
CANONICAL_NUM_EXPERTS = NUM_EXPERTS_CANONICAL


# ---------------------------------------------------------------------------
# The TASK axis  (T-L1.0 / plan_language.md §1)
# ---------------------------------------------------------------------------
# `task` is ORTHOGONAL to `architecture` and the two must never be merged into a
# single enum. They answer different questions:
#
#   architecture -> num_experts / max_depth / adaptive_halting / ffn_mult
#                   i.e. WHICH computation module and HOW MUCH depth
#   task         -> input adapter, output head, task loss, oracle-label source,
#                   metric vocabulary, dataset
#
# All nine (task, architecture) pairs are meaningful, which is the whole point:
# the arithmetic study's MoE/MoR/MoRE comparison is re-run on language with the
# same engine, so a difference between the two studies is a property of the data
# and not of a second implementation (CLAUDE.md §9: ONE training system).
#
# Defined at the top of the module, above the defaults that use them, for the
# same reason as CANONICAL_NUM_EXPERTS above.
#
# NAME COLLISION, and it is unavoidable: `cfg["task"]` is this axis, while
# `cfg["loss_weights"]["task"]` is the WEIGHT on the task-loss term and predates
# the migration. They are different quantities at different depths -- one selects
# the objective, the other scales one term of it. Neither is renamed: renaming the
# loss weight would move every arithmetic config_hash, which is the exact harm this
# axis is built to avoid. `test_language_task_axis.py:TL1.0a` asserts that
# `loss_weights.task` is the ONLY `task` key in a resolved arithmetic config, so
# the collision cannot quietly become a leak.
TASK_ARITHMETIC = "arithmetic"
TASK_LANGUAGE   = "language"
TASKS = (TASK_ARITHMETIC, TASK_LANGUAGE)

# The task whose absence is assumed. See resolve_task() for why absence -- rather
# than an injected default key -- is what "default arithmetic" means here.
CANONICAL_TASK = TASK_ARITHMETIC

# Language shape fields, stamped by apply_task.
#
# `attention` is DEFINITIONAL and is assigned, not defaulted: language is not
# learnable without context (plan_language.md §1.3), and `MoREWrapper(attention=
# False)` constructs no attention module at all, so a language run with it off is
# not a weaker language model -- it is a model that cannot see any other token.
#
# The three NUMBERS are `setdefault`, not assignment, and the difference matters.
# `seq_len` is one of the protocol fields that canonical_spec_language.json holds
# at `null` until GATE L5 freezes it FROM MEASUREMENT (plan_language.md §10.2) --
# 6 GB of VRAM is the binding constraint on this phase (ENVIRONMENT.md §2), so the
# real value comes out of T-L7.0's throughput probe. If apply_task assigned it,
# editing config_language.json or the frozen spec would have no effect and the
# gate's own value would be silently overwritten by a literal in this file. The
# value below exists only so that `--task language` alone builds a RUNNABLE model
# instead of raising KeyError; it is provisional and is not the protocol.
LANGUAGE_SEQ_LEN_DEFAULT    = 256     # provisional -- GATE L5 owns the real value
LANGUAGE_VOCAB_SIZE_DEFAULT = 8192    # T-L2.1: byte-level BPE fitted on train only
LANGUAGE_N_HEADS_DEFAULT    = 4       # plan_language.md §6.6, frozen at d_model 256


def resolve_task(cfg: dict) -> str:
    """
    The task this config declares: "arithmetic" (the default) or "language".

    ABSENCE MEANS ARITHMETIC, and that is a deliberate design choice rather than
    laziness about writing a setdefault. `run_context.config_hash()` is SHA-256
    over the config with only `provenance` and `logging` removed, so injecting a
    top-level `task` key into every config would move the hash of every arithmetic
    config -- including the fifteen canonical Phase-B runs. Their stored
    resolved_config.json would keep the old hash while a re-resolution of the same
    file produced a new one, so `export_results.py` (which refuses to mix config
    hashes) would read a re-run as a different experiment, and run_context's
    property that "a re-run of the same config is recognisably the same run" would
    quietly stop holding.

    So the key is present only when someone actually declared it. An arithmetic
    config resolves byte-identically to what it resolved to before this axis
    existed (T-L1.0's Verify criterion, checked by test_gateL0_task_axis.py), and a
    language config carries `"task": "language"` and hashes differently -- which is
    correct, because it IS a different experiment.

    Every run still REPORTS its task: engine.py publishes `provenance.task`, and
    `provenance` is outside the hash. Same shape as `variant` and `halting_mode` --
    derived, published, never hashed.

    Raises ValueError on an unrecognised value rather than falling back to
    arithmetic: silently running the arithmetic pipeline because someone typed
    `--task lang` would produce a complete, plausible, wrong run (CLAUDE.md §2:
    an unmapped value must raise, never fall back).
    """
    task = cfg.get("task")
    if task is None:
        return CANONICAL_TASK
    task = str(task).lower()
    if task not in TASKS:
        raise ValueError(
            f"task must be one of {TASKS}, got {cfg.get('task')!r}. Absence means "
            f"{CANONICAL_TASK!r}; an unrecognised value is refused rather than "
            "defaulted, because defaulting would run the arithmetic pipeline on a "
            "config that asked for something else and report it as arithmetic."
        )
    return task


def load_config_defaults(cfg: dict | None = None) -> dict:
    """
    Fill in every default key, in place, and return cfg.

    Split out of load_config so the defaults can be exercised without a file on
    disk. This matters for the gates: a test that hand-builds a dict would be
    asserting against its own copy of the defaults rather than against the ones
    the training run actually uses, and the two could drift apart silently.
    """
    cfg = {} if cfg is None else cfg

    # T-L1.0 (plan_language.md §1). NOTE WHAT IS *NOT* HERE: there is no
    # `cfg.setdefault("task", TASK_ARITHMETIC)`. That was the obvious
    # implementation and it is wrong -- it would move the config_hash of every
    # arithmetic run, including the fifteen canonical Phase-B ones. The default is
    # the ABSENCE of the key; `resolve_task()` carries the full reasoning, and the
    # language default block is applied at the end of this function.
    #
    # T-L1.0 / T5.3 again: "the user declared this" must stay distinguishable from
    # "we filled it in". `loss_weights.family_cls` gets the arithmetic default 0.5
    # below, so by the time the language block runs, a language config that
    # explicitly asked for 0.5 is indistinguishable from one that said nothing --
    # and those two must behave differently. The first is a copied arithmetic
    # config and must be REFUSED (plan_language.md §7.3); the second is an ordinary
    # language run and must be set to 0.0 and proceed. Snapshotted here, before any
    # setdefault runs, for exactly the reason _user_subset is.
    _user_family_cls = (
        cfg["loss_weights"]["family_cls"]
        if isinstance(cfg.get("loss_weights"), dict)
        and "family_cls" in cfg["loss_weights"]
        else None
    )


    # T5.3 (plan.md 7.3): snapshot what the CALLER actually wrote for the
    # duplicated subset keys, BEFORE any setdefault below runs. Once the
    # defaults are in, `"subset_fraction" in cfg["training"]` is true
    # unconditionally and "the user asked for this" is indistinguishable from
    # "we filled it in" -- which is precisely how the removed
    # backward-compatibility shim came to overwrite a configured
    # `data.subset_fraction` with the default 1.0 on every single load.
    _user_subset = {
        key: {
            section: cfg[section][key]
            for section in ("training", "data")
            if isinstance(cfg.get(section), dict) and key in cfg[section]
        }
        for key in ("subset_fraction", "subset_seed")
    }

    # Provide safe defaults for any key the agent may have deleted
    cfg.setdefault("model", {})
    cfg["model"].setdefault("d_model", 256)
    # T5.1 (plan.md 7.1): the canonical expert count, from the one manifest.
    # This default was the literal 7 while CANONICAL_NUM_EXPERTS was 6, so any
    # config that omitted the key -- or any caller of load_config_defaults that
    # never went through apply_architecture, which includes several tests --
    # silently built a 7-expert model with a permanently unreachable 7th class.
    cfg["model"].setdefault("num_experts", CANONICAL_NUM_EXPERTS)
    cfg["model"].setdefault("routing_mode", CANONICAL_ROUTING_MODE)
    # T5.4 (plan.md 7.4): router exploration noise, canonical OFF. Previously
    # there was no key at all and a trainable noise scale of 0.1 was always
    # active, so no config, resolved config or W&B record mentioned that every
    # training-time routing decision was made on perturbed logits.
    cfg["model"].setdefault("router_noise", CANONICAL_ROUTER_NOISE)
    cfg["model"].setdefault("router_noise_init", ROUTER_NOISE_INIT_SCALE_DEFAULT)
    cfg["model"].setdefault("router_noise_anneal_steps",
                            ROUTER_NOISE_ANNEAL_STEPS_DEFAULT)
    # NOTE: these two 7s are NOT the expert count and must not be "fixed" to 6.
    # max_depth is the recursion budget (how many times the shared block may be
    # re-applied); max_steps is the number of step-tokens the dataset packs per
    # program. They coincide with the old expert count only by accident, and
    # that coincidence is what made the T5.1 landmines hard to spot.
    cfg["model"].setdefault("max_depth", 7)
    cfg["model"].setdefault("max_steps", 7)        # number of step-tokens per program
    cfg["model"].setdefault("step_feat_dim", 12)   # features per step-token
    # T8.2: 1, not 2. canonical_spec.json freezes num_blocks=1 (plan.md Phase 8:
    # "one recursive MoRE block initially"), and code/config.json says 1. This
    # default said 2, making load_config_defaults a THIRD source of truth that
    # disagreed with the other two. It only bit callers that build a config
    # without reading config.json -- which is every gate test and every
    # hand-built dict -- so the tests were exercising a 2-block model while the
    # canonical run is 1-block. Same reasoning for batch_size below.
    cfg["model"].setdefault("num_blocks", 1)
    # Halting supervision (T3.3). DEFAULT OFF = pure unsupervised ACT.
    # When True the halt heads additionally receive an MSE target from the
    # predefined operation-complexity curriculum (families.OP_TARGET_DEPTH).
    # updated_rules.md 2.3 permits either, but requires the choice be reported
    # explicitly -- so this flag is recorded in provenance, never inferred.
    cfg["model"].setdefault("halting_supervision", False)
    cfg["model"].setdefault("dropout", 0.1)

    cfg.setdefault("training", {})
    cfg["training"].setdefault("lr", 1e-3)
    cfg["training"].setdefault("weight_decay", 1e-4)
    cfg["training"].setdefault("batch_size", 768)   # T8.2: was 128; see num_blocks
    cfg["training"].setdefault("epochs", 50)
    cfg["training"].setdefault("val_split", 0.1)
    cfg["training"].setdefault("grad_clip", 1.0)
    cfg["training"].setdefault("subset_fraction", 1.0)
    cfg["training"].setdefault("subset_seed", 42)

    cfg.setdefault("loss_weights", {})
    cfg["loss_weights"].setdefault("task", 1.0)
    # T4.2 (plan.md 6.2): FROZEN at 0.001 from measurement, not inherited. The
    # historical 0.05 put the weighted balance term at 0.41x the task loss after
    # warmup and, projected onto the converged task loss this repo has actually
    # reached (0.001666), at ~20x it -- i.e. it re-dominates by the end of a
    # 50-epoch run, which is the defect Phase 4 exists to remove. At 0.001 the
    # ratio is 0.017 after warmup and ~0.82 at that converged task loss, and the
    # hard expert load stays spread (9.5-29.3%, no expert starved). This default
    # must agree with code/config.json; see changelog.md "Phase 4" and
    # ARCHITECTURE.md 5b for the full measurement table.
    cfg["loss_weights"].setdefault("routing_balance", 0.001)
    cfg["loss_weights"].setdefault("halting", 0.001)
    # T8.0b: 0.0, NOT 0.5. engine.py builds step_routing_loss as
    # 0.01*oracle_routing_ce + 0.3*step_cls_loss, so any non-zero weight trains
    # the router against the ORACLE EXPERT INDEX. plan.md E: supervision OFF is
    # canonical, supervision ON is an explicitly labelled oracle-routing
    # ablation (T10.E). The old 0.5 default made the canonical config an
    # ablation, and drove routing_accuracy = hungarian = AMI = purity = 1.0 with
    # the identity assignment recovered -- a measurement of the supervision
    # signal, not of learned specialization. A default that silently turns
    # supervision on is the more dangerous direction of error, so the default
    # matches the canonical value. Must agree with code/config.json.
    cfg["loss_weights"].setdefault("step_routing", 0.0)
    # Kept SEPARATE from "halting" (the ponder cost). plan.md 5.4: the two pull
    # in opposite directions -- ponder cost pushes depth down, curriculum
    # supervision pushes it toward the target -- so they must never be summed
    # into one opaque number. 0.0 by default because supervision is off.
    cfg["loss_weights"].setdefault("halting_supervision", 0.0)
    # T8.3: the whole-program family classification term. Was a bare `0.5`
    # literal inside engine.py's total_loss expression -- `lw["task"] *
    # (task_loss + 0.5 * cls_loss)` -- so it appeared in no config, no
    # provenance block and no results table, while being the second-largest
    # term in the objective. Two consequences: the paper could not state the
    # objective it actually optimized, and the term could not be ablated without
    # editing source (which produces an unlabelled variant, forbidden by
    # CLAUDE.md 6).
    #
    # It is EXTERNAL SUPERVISION, not self-supervision: the targets are the
    # "family": "E1".."E6" string literals that data/script.py stamps into every
    # JSONL record at generation time, mapped through the hand-written
    # families.py:FAMILY_TO_IDX manifest. Verified independent of the model --
    # the label histogram is byte-identical at num_experts=6 and num_experts=1.
    # Canonical value is the historical 0.5 so every completed run is
    # reproducible; 0.0 is the T10.H no-family-supervision ablation.
    cfg["loss_weights"].setdefault("family_cls", 0.5)

    cfg.setdefault("data", {})
    # An explicitly configured single-file dataset (`jsonl_path`) must win over
    # the split defaults. Without this, setdefault below injects
    # ../data/train.jsonl and the engine's `train_path or jsonl_path`
    # precedence silently trains on a file the config never named.
    if "jsonl_path" in cfg["data"] and "train_path" not in cfg["data"]:
        cfg["data"]["train_path"] = cfg["data"]["jsonl_path"]
    cfg["data"].setdefault("train_path", "../data/train.jsonl")
    cfg["data"].setdefault("val_path", "../data/val.jsonl")
    cfg["data"].setdefault("test_path", "../data/test.jsonl")
    cfg["data"].setdefault("jsonl_path", cfg["data"]["train_path"])
    cfg["data"].setdefault("subset_fraction", 1.0)
    cfg["data"].setdefault("subset_seed", 42)
    cfg["data"].setdefault("max_val", 1e6)
    cfg["data"].setdefault("pad_value", 0.0)

    cfg.setdefault("logging", {})
    cfg["logging"].setdefault("wandb_project", "micro-MoRE-poc")
    # T9.0: log_interval also gates VALIDATION (engine.py: `epoch %
    # log_interval == 0 or epoch == epochs`), so it is a protocol field, not a
    # cosmetic one. At the old value of 10 a 20-epoch run measured val twice, and
    # "best epoch by val loss" was a choice between two candidates -- which on a
    # still-falling curve always returns the last epoch, making checkpoint
    # selection indistinguishable from "take the final model". This default must
    # stay equal to config.json's value: a defensive default that disagrees with
    # canonical is exactly the trap T8.3 hit with family_cls.
    cfg["logging"].setdefault("log_interval", 2)

    # T5.3 (plan.md 7.3): ONE source of truth for the subset keys.
    #
    # `subset_fraction` / `subset_seed` are readable in two sections for
    # historical reasons: older fast_config files wrote them under `training`,
    # while every CONSUMER reads `data` -- engine.py builds the Subset from
    # `data.subset_fraction`, run_context.py's proxy guard refuses a canonical
    # claim from it, and provenance publishes it as `resolved_subset_fraction`.
    #
    # What used to be here was `if "subset_fraction" in cfg["training"]:
    # cfg["data"][...] = cfg["training"][...]`, whose test is ALWAYS true
    # because of the setdefault above. So a config that said
    # `data.subset_fraction: 0.5` trained on the full dataset and reported 1.0,
    # with nothing in the output disagreeing with itself. The two sections are
    # now reconciled instead:
    #   * given in neither          -> the default, mirrored into both
    #   * given in exactly one      -> that value, mirrored into both
    #   * given in both and equal   -> that value (harmless duplication)
    #   * given in both and DIFFER  -> raise
    # The last case refuses rather than picking a winner: either choice trains
    # on a dataset size the config does not state, and `resolved_subset_fraction`
    # then records that size as though it had been asked for.
    for _key in ("subset_fraction", "subset_seed"):
        _given  = _user_subset[_key]
        _values = set(_given.values())
        if len(_values) > 1:
            raise ValueError(
                f"Conflicting {_key}: training.{_key}={_given['training']!r} "
                f"but data.{_key}={_given['data']!r}. These are the same "
                "quantity read from two sections; the old shim silently let "
                "`training` win, so a configured `data` value never took effect "
                "and the run reported a subset size it did not use "
                "(plan.md 7.3). Set exactly one of them."
            )
        # No user value anywhere -> fall through to the default already applied
        # by setdefault. `training` and `data` carry the same default, so which
        # one is read here does not matter.
        _resolved = _values.pop() if _values else cfg["data"][_key]
        cfg["training"][_key] = _resolved
        cfg["data"][_key]     = _resolved

    # T-L1.0 (plan_language.md §7 table, `load_config_defaults` row): the language
    # block runs LAST, after every shared default, so it can react to values the
    # shared defaults just filled in (`family_cls`) and so an arithmetic config
    # never touches it at all. Guarded on the declared task, so this is a no-op --
    # not a cheap no-op, literally an unentered branch -- for every arithmetic
    # config in the repository.
    if resolve_task(cfg) == TASK_LANGUAGE:
        _apply_language_block(cfg, declared_family_cls=_user_family_cls)

    return cfg


def load_config(path: str = "config.json") -> dict:
    """Load the agent-modifiable config file, apply defaults, resolve paths."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"config.json not found at '{path}'. "
            "Create it before running train.py."
        )
    with open(path, "r") as f:
        cfg = json.load(f)

    cfg = load_config_defaults(cfg)

    # Resolve relative dataset paths against the config file directory.
    cfg_dir = os.path.dirname(os.path.abspath(path))
    for key in ("train_path", "val_path", "test_path", "jsonl_path"):
        p = cfg["data"].get(key)
        if isinstance(p, str) and p and not os.path.isabs(p):
            cfg["data"][key] = os.path.normpath(os.path.join(cfg_dir, p))

    return cfg




# ---------------------------------------------------------------------------
# Architecture selection  (plan.md 9 / updated_rules.md 6)
# ---------------------------------------------------------------------------
# ONE training system. `architecture` derives only the settings that are
# *definitionally* different between the three baselines; everything else --
# dataset, split, optimizer, lr, weight decay, batch size, epochs, seed set,
# input representation -- stays identical so the comparison is fair
# (updated_objective.md "CANONICAL HYPERPARAMETER PRINCIPLE").

ARCHITECTURES = ("moe", "mor", "more")

# T6.6 (updated_rules.md 9 / canonical_spec.json run_name_template). Run names
# must be STABLE and readable: phaseB_MoE_seed42, not phaseB_moe or
# more_run_final_v3. The display spelling is separate from the config value
# because the config value is a lowercase enum used for dispatch, while the run
# label is what appears in W&B, the run directory and every figure caption.
ARCH_DISPLAY = {"moe": "MoE", "mor": "MoR", "more": "MoRE"}
CANONICAL_RUN_PREFIX = "phaseB"

# T-L1.1 (plan_language.md §10.2). The language study's run-name prefix. A
# SEPARATE prefix, not a suffix or a tag, because the prefix is the first thing
# sorted in a `runs/` listing, in a W&B project sidebar and in an exporter table --
# `langB_MoRE_seed42` and `phaseB_MoRE_seed42` can never be confused at a glance or
# by a glob, and the two studies are two studies (plan_language.md T-L10.2: not
# merged, not averaged, not a trend across two points).
CANONICAL_RUN_PREFIX_LANGUAGE = "langB"
RUN_PREFIX_BY_TASK = {
    TASK_ARITHMETIC: CANONICAL_RUN_PREFIX,
    TASK_LANGUAGE:   CANONICAL_RUN_PREFIX_LANGUAGE,
}

# T6.7 (updated_rules.md 9). `variant` is a required provenance field and must
# never be hand-written, because the whole point of it is to make an ablation
# impossible to mistake for the canonical setting in a results table. It is
# DERIVED from the resolved config: "canonical" only when every deviation-
# carrying field sits at its canonical value, otherwise a sorted, joined list of
# the deviations that are actually active. A new ablation must be added here or
# it will silently be reported as canonical.
CANONICAL_VARIANT = "canonical"
# T8.3. The canonical weight of the whole-program family-classification term.
# 0.5 is the value every run in this repository has used since the term was
# written as a literal inside engine.py's total_loss; it is frozen at that value
# so the field's introduction changes no number, and 0.0 is the labelled T10.H
# ablation that measures how much of the routing partition it is responsible for.
CANONICAL_FAMILY_CLS_WEIGHT = 0.5

# T-L1.1. The canonical family-classification weight is TASK-DEPENDENT, and this
# is a defect fix, not a new feature. resolve_variant compared `family_cls`
# against the single arithmetic constant above, so an unmodified canonical
# language config -- where 0.0 is REQUIRED (plan_language.md §7.3 / §10.1) --
# resolved to `variant = "no_family_supervision"`: a canonical run wearing an
# ablation label, in the one field that exists to stop an ablation being read as
# canonical. Observed while writing test_language_task_axis.py, before any
# language run existed to be mislabelled.
CANONICAL_FAMILY_CLS_WEIGHT_BY_TASK = {
    TASK_ARITHMETIC: CANONICAL_FAMILY_CLS_WEIGHT,
    # 0.0 is not "the ablation" on language -- it is the only coherent value. The
    # head reads h.detach(), so the weight cannot change what the trunk learns, and
    # a packed LM sequence has no whole-sequence family label for the term to use.
    TASK_LANGUAGE:   0.0,
}

# T10.C. Canonical recursive-block count. Frozen at 1 in canonical_spec.json's
# enforced_fields; named here so resolve_variant can LABEL a deviation instead of
# letting a 2-block run print `variant = canonical` in an ablation table.
CANONICAL_NUM_BLOCKS = 1

# T8.3. Canonical FFN multiplier per architecture -- the ONE place the
# parameter-budget policy is written, imported by resolve_variant here and
# mirrored in canonical_spec.json's architecture_variants (which the Gate 0 guard
# enforces). MoE/MoRE: 4, unchanged since the field existed. MoR: 24, i.e. 4 x 6,
# so MoR's single shared FFN receives the total hidden width of MoRE's six
# experts. That is a principled construction rather than a fitted one, and it
# lands the budgets within 0.12% (571,150 -> 3,197,710 against MoRE's 3,201,555),
# inside T9.2's 5% bound. MoR at 4 remains available as the labelled
# under-budgeted small baseline.
CANONICAL_FFN_MULT = {"moe": 4, "mor": 24, "more": 4}


def resolve_variant(cfg: dict) -> str:
    """
    Derive the `variant` label from the resolved config.

    Returns "canonical", or a '+'-joined sorted list of active deviations, e.g.
    "dense_routing+supervised_curriculum". Never raises: an unknown value is
    reported as a deviation rather than dropped, since silence is the failure
    mode this field exists to prevent.
    """
    mc = cfg.get("model", {}) or {}
    lw = cfg.get("loss_weights", {}) or {}
    tags: list[str] = []

    # T-L1.1 (plan_language.md §1). The TASK goes in first, and it is the one tag
    # that is not a "deviation" -- it is which study the run belongs to. Without it
    # a canonical language run resolved to `variant = "canonical"`, i.e. a run on a
    # different dataset, with a different objective and a different output head,
    # wearing the same label as the fifteen published arithmetic runs. `variant` is
    # read by the exporter and printed in results tables, so that is precisely the
    # confusion this field exists to prevent.
    #
    # Sorting: tags are sorted below, and "arithmetic"/"language" would not
    # reliably sort first, so the task is prepended AFTER the sort (see the return).
    task = resolve_task(cfg)

    routing_mode = mc.get("routing_mode", CANONICAL_ROUTING_MODE)
    if routing_mode != CANONICAL_ROUTING_MODE:
        tags.append(f"routing_{routing_mode}")

    router_noise = mc.get("router_noise", CANONICAL_ROUTER_NOISE)
    if router_noise != CANONICAL_ROUTER_NOISE:
        tags.append(f"router_noise_{router_noise}")

    if mc.get("fixed_depth", False):
        tags.append("fixed_depth")

    # T10.C: num_blocks is an ENFORCED field, so the Gate 0 guard already refuses
    # a canonical_phase_b claim from a 2-block run. But `variant` is what an
    # ablation TABLE prints, and without this tag a 2-block run read
    # `variant = canonical` -- a run that differs from canonical in a pinned
    # architectural field, wearing the canonical label in the one place a reader
    # looks. Additive only: canonical num_blocks is 1, so the 15 T9.1 runs are
    # unaffected and their variant stays `canonical`.
    num_blocks = int(mc.get("num_blocks", CANONICAL_NUM_BLOCKS))
    if num_blocks != CANONICAL_NUM_BLOCKS:
        tags.append(f"num_blocks_{num_blocks}")

    # Supervised depth curriculum vs pure ACT. MoE is excluded: it runs at
    # max_depth 1, where there is no depth to allocate, so "pure ACT" would be a
    # claim about a mechanism the architecture does not have.
    if mc.get("halting_supervision", False) and float(
            lw.get("halting_supervision", 0.0)) != 0.0:
        tags.append("supervised_curriculum")

    # T8.2 BUGFIX. This line read `lw.get("routing_oracle", 0.0)` -- a key that
    # exists NOWHERE in this repository: not in config.json, not in any of the
    # config_*.json variants, not in load_config_defaults, not in cli.py, and not
    # in engine.py's loss assembly. So the test was `0.0 != 0.0`, permanently
    # False, and the `oracle_routing` tag could never be emitted.
    #
    # The real routing-supervision weight is `loss_weights.step_routing`
    # (engine.py: step_routing_loss = 0.01*oracle_routing_ce + 0.3*step_cls_loss,
    # then `+ lw["step_routing"] * step_routing_loss`). Any non-zero value trains
    # the router against the ORACLE EXPERT INDEX.
    #
    # Consequence while the typo stood: every oracle-supervised run was stamped
    # variant="canonical". Verified on runs/routedec_oracle_supervised_router_
    # seed42__4bd62d20, whose provenance says routing_supervision_enabled=true /
    # routing_supervision_weight=0.5 and variant="canonical" in the same block.
    # No MEASUREMENT was wrong -- the provenance pair carried the truth, which is
    # how T8.0b validated its arms -- but `variant` is precisely the field that
    # exists to stop an ablation being read as canonical in a results table
    # (plan.md E / T10.E: supervision OFF is canonical, ON is a labelled
    # ablation, never mixed in one table), and it was inoperative for the one
    # ablation the routing half of the paper turns on.
    if float(lw.get("step_routing", 0.0)) != 0.0:
        tags.append("oracle_routing")

    # T8.3. The family-classification weight, lifted out of engine.py's
    # total_loss where it was a bare 0.5 literal. CANONICAL_FAMILY_CLS_WEIGHT is
    # the historical value, so every run made before the field existed still
    # resolves to "canonical"; a run that turns the term off is the T10.H
    # ablation and must be labelled, because "MoRE discovers operation families"
    # is a different claim depending on whether a 6-way family cross-entropy was
    # in the objective.
    #
    # T-L1.1: compared against the canonical weight FOR THIS TASK, not against the
    # arithmetic 0.5 alone. On language the canonical value is 0.0 and a non-zero
    # value is refused outright by apply_task, so the only reachable value is the
    # canonical one and no tag is emitted -- where the shared constant used to emit
    # `no_family_supervision` on every canonical language run.
    _canon_fcw = CANONICAL_FAMILY_CLS_WEIGHT_BY_TASK.get(
        task, CANONICAL_FAMILY_CLS_WEIGHT)
    if float(lw.get("family_cls", _canon_fcw)) != _canon_fcw:
        if float(lw.get("family_cls", _canon_fcw)) == 0.0:
            tags.append("no_family_supervision")
        else:
            tags.append(
                f"family_cls_{float(lw['family_cls']):g}".replace(".", "p"))

    # T8.3. ffn_mult is architecture-scoped: canonical MoE/MoRE are 4, canonical
    # MoR is 24 (4 x 6 -- MoR's single FFN is given the total hidden width of
    # MoRE's six experts, which is what makes the parameter budgets comparable to
    # 0.12%). A deviation must therefore be judged against the run's OWN
    # architecture, not against a single shared number. MoR at ffn_mult 4 is the
    # deliberately under-budgeted small baseline; it is a legitimate ablation and
    # gets a label rather than being refused silently.
    arch = str((cfg.get("provenance", {}) or {}).get("architecture")
               or cfg.get("architecture") or "").lower()
    if arch in CANONICAL_FFN_MULT:
        want_fm = CANONICAL_FFN_MULT[arch]
        got_fm = int(mc.get("ffn_mult", 4))
        if want_fm is not None and got_fm != want_fm:
            tags.append(f"ffn_mult_{got_fm}")

    # T-L1.1. The task leads, then the sorted deviations. So:
    #   arithmetic, no deviations -> "canonical"      (unchanged, all 15 runs)
    #   language,   no deviations -> "language"       (canonical FOR that task)
    #   language,   fixed depth   -> "language+fixed_depth"
    # The word "canonical" is deliberately never emitted for a non-default task:
    # it is the label the arithmetic study's published table already uses, and a
    # language row must not be able to carry it.
    if task != CANONICAL_TASK:
        return "+".join([task] + sorted(tags))
    return "+".join(sorted(tags)) if tags else CANONICAL_VARIANT


# T8.2. ONE definition of "which halting objective ran", imported by both
# engine.py (which logs it as provenance.halting_mode / halt_target_mode) and
# run_context.py (whose Gate 0 guard enforces the canonical value frozen by
# T8.0a). The guard runs BEFORE engine.train() builds its provenance block, so it
# cannot read the value off provenance and must derive it -- and a second copy of
# the rule living in the guard is how a guard comes to pass a run whose objective
# no longer matches (the T5.4 trap).
HALTING_MODE_PURE_ACT   = "pure_act"
HALTING_MODE_SUPERVISED = "supervised_curriculum"


def resolve_halting_mode(cfg: dict) -> str:
    """
    "pure_act" or "supervised_curriculum", from the resolved config.

    Requires BOTH the model flag and a non-zero loss weight, matching engine.py:
    a flag set with a zero weight computes the curriculum term and multiplies it
    away, which engine.py refuses outright rather than silently recording as
    supervised.
    """
    mc = cfg.get("model", {}) or {}
    lw = cfg.get("loss_weights", {}) or {}
    supervised = bool(mc.get("halting_supervision", False)) and float(
        lw.get("halting_supervision", 0.0)) != 0.0
    return HALTING_MODE_SUPERVISED if supervised else HALTING_MODE_PURE_ACT



def canonical_run_name(architecture: str, seed: int | None,
                       task: str = TASK_ARITHMETIC) -> str:
    """
    The stable run name required by updated_rules.md 9: phaseB_<Arch>_seed<S>,
    or langB_<Arch>_seed<S> on the language task (T-L1.1 / plan_language.md §10.2).

    Seed omitted when no seed is declared -- an undeclared-seed run must not be
    given a name that claims a seed it never set (T6.1).

    `task` is a keyword with the arithmetic default so that every existing caller
    -- apply_architecture and several tests -- keeps producing byte-identical
    names. A required third positional would have been the same change to the
    output for arithmetic, but a much easier one to get wrong at one call site
    and not notice, because the wrong answer is still a plausible run name.
    """
    arch = str(architecture).lower()
    if arch not in ARCHITECTURES:
        raise ValueError(
            f"architecture must be one of {ARCHITECTURES}, got {architecture!r}"
        )
    t = str(task).lower()
    if t not in TASKS:
        raise ValueError(f"task must be one of {TASKS}, got {task!r}")
    stem = f"{RUN_PREFIX_BY_TASK[t]}_{ARCH_DISPLAY[arch]}"
    return stem if seed is None else f"{stem}_seed{int(seed)}"


def stamp_seed_into_run_name(cfg: dict) -> str:
    """
    Append `_seed<S>` to logging.run_name when a seed is declared and the name
    does not already carry one, and return the final name.

    Why this is a separate step from apply_architecture: the seed is not known
    until resolve_overrides has applied `--seed`, and the name is not final
    until `--run_name` has been applied. Stamping earlier would label every run
    with the config-file seed. Custom names are kept and extended rather than
    replaced, so a diagnostic run stays identifiable while still recording which
    seed produced it.
    """
    log = cfg.setdefault("logging", {})
    seed = cfg.get("provenance", {}).get("seed")
    name = str(log.get("run_name") or "run")
    if seed is not None and "_seed" not in name:
        name = f"{name}_seed{int(seed)}"
        log["run_name"] = name
    return name


# CANONICAL_NUM_EXPERTS is defined at the top of this module (T5.1) so the
# defaults in load_config_defaults can use it instead of a second literal.


def apply_architecture(cfg: dict, architecture: str) -> dict:
    """
    Stamp the architecture-specific fields into cfg, in place, and return it.

    moe  : 6 experts, depth 1, no adaptive halting, learned top-1 routing
    mor  : 1 expert,  canonical depth, adaptive halting, routing N/A
    more : 6 experts, canonical depth, adaptive halting, learned top-1 routing
    """
    arch = str(architecture).lower()
    if arch not in ARCHITECTURES:
        raise ValueError(
            f"architecture must be one of {ARCHITECTURES}, got {architecture!r}"
        )

    mc = cfg.setdefault("model", {})
    lw = cfg.setdefault("loss_weights", {})
    canonical_depth = mc.get("max_depth", mc.get("max_steps", 7))

    if arch == "moe":
        mc["num_experts"] = CANONICAL_NUM_EXPERTS
        mc["max_depth"] = 1
        mc["adaptive_halting"] = False
        lw["halting"] = 0.0          # no ponder cost when there is no recursion
        # max_depth == 1 means every token forced-exits at step 1. A depth
        # curriculum target of 2, 3 or 4 is then unreachable by construction, so
        # supervising it would inject a constant, irreducible loss and a
        # gradient that can never be satisfied. MoE has no depth to allocate.
        mc["halting_supervision"] = False
        lw["halting_supervision"] = 0.0
    elif arch == "mor":
        mc["num_experts"] = 1
        mc["max_depth"] = canonical_depth
        mc["adaptive_halting"] = True
        lw["routing_balance"] = 0.0  # nothing to balance with a single expert
        lw["step_routing"] = 0.0     # no routing target with a single expert
    else:  # more
        mc["num_experts"] = CANONICAL_NUM_EXPERTS
        mc["max_depth"] = canonical_depth
        mc["adaptive_halting"] = True

    # T8.3 PARAMETER BUDGET. ffn_mult is stamped here, alongside num_experts and
    # max_depth, because it is now definitionally architecture-scoped: MoR has one
    # FFN where MoRE has six, so equal ffn_mult means an unequal parameter budget.
    # Left at the shared 4, MoR is 571,150 parameters against MoRE's 3,201,555 --
    # an 82.2% shortfall, so any MoRE-beats-MoR result would be confounded with
    # capacity and T9.2's 5% budget-matching bound fails outright. At 24 (= 4 x 6,
    # MoR's single FFN given the total hidden width of MoRE's six) MoR is 3,197,710
    # and the gap is 0.12%.
    #
    # Stamped rather than left to config.json so `--architecture mor` alone
    # produces a budget-matched model; an explicit `--ffn_mult 4` still wins,
    # because resolve_overrides runs after this and produces the labelled
    # `ffn_mult_4` small-baseline ablation (resolve_variant).
    _canonical_fm = CANONICAL_FFN_MULT.get(arch)
    if _canonical_fm is not None:
        mc["ffn_mult"] = _canonical_fm

    cfg["architecture"] = arch
    # T-L1.1: named for the task the config declares. On the CLI path apply_task
    # has NOT run yet at this point (cli.py calls it immediately after this
    # function), so a `--task language` run is stamped `phaseB_*` here and repaired
    # by _apply_language_block -- which only replaces a name that is still exactly
    # this default, so a config-file or `--run_name` label is never clobbered.
    cfg.setdefault("logging", {}).setdefault(
        "run_name", canonical_run_name(arch, None, resolve_task(cfg)))
    # NOTE: enforce_routing_mode is deliberately NOT called here. At this point
    # the run_name is still the default stamped one line above -- the CLI's
    # --run_name override has not been applied yet -- so gating on the label
    # here refuses even a correctly labelled dense ablation. The gate runs in
    # cli.py AFTER resolve_overrides, against the label the run will really use.
    return cfg


# ---------------------------------------------------------------------------
# Task selection  (T-L1.0 / plan_language.md §1, §6.6, §7.3, §10.1)
# ---------------------------------------------------------------------------

def _apply_language_block(cfg: dict, declared_family_cls=None) -> dict:
    """
    Stamp the language task's shape fields, in place, and refuse the one
    combination that cannot mean anything. ONE implementation, two callers:
    `load_config_defaults` (a config file that declares `task: "language"`) and
    `apply_task` (a `--task language` on the CLI). A second copy of this rule is
    how the two entry points come to disagree -- the T5.4 trap.

    `declared_family_cls` is the value the INCOMING config declared, or None if it
    declared nothing. It cannot be read off `cfg` here, because the shared defaults
    have already filled in the arithmetic 0.5.
    """
    mc = cfg.setdefault("model", {})
    lw = cfg.setdefault("loss_weights", {})
    dc = cfg.setdefault("data", {})

    # plan_language.md §7.3, enforced rather than intended. On language the family
    # head is a DETACHED probe: it reads h.detach(), its gradient reaches only its
    # own Linear, and its loss weight is therefore scientifically inert. Weighting
    # it into the trunk would re-import the exact confound the arithmetic study has
    # to caveat -- "MoRE's representation separates families" is a much weaker
    # statement when a family cross-entropy was being minimised on that
    # representation the whole time.
    #
    # There is also no whole-sequence family label in a packed LM sequence for the
    # term to use, so a non-zero value here can only be a copied arithmetic config.
    # Hence: refuse a DECLARED non-zero weight, and set an undeclared one to 0.0.
    # The distinction is the point -- silently rewriting a declared 0.5 to 0.0 would
    # run something the config did not ask for, and raising on an undeclared 0.5
    # would make `--task language` alone impossible (T-L1.2's Verify).
    if declared_family_cls is not None and float(declared_family_cls) != 0.0:
        raise ValueError(
            f"loss_weights.family_cls = {declared_family_cls!r} with "
            f"task = {TASK_LANGUAGE!r}. On language the family head is a DETACHED "
            "probe (plan_language.md §7.3): it reads h.detach(), so its gradient "
            "never reaches the trunk and its weight cannot change what is learned. "
            "A non-zero value therefore claims a supervision term that does not "
            "exist, and there is no whole-sequence family label in a packed LM "
            "sequence for such a term to use -- this is almost certainly a copied "
            "arithmetic config. Set loss_weights.family_cls to 0.0 (it is REQUIRED "
            "at 0.0 in config_language.json) or drop the key and let the language "
            "block set it."
        )
    lw["family_cls"] = 0.0

    # DEFINITIONAL, so assigned rather than defaulted. MoREWrapper(attention=False)
    # constructs NO attention module (plan_language.md §6.5, which is also what
    # keeps the arithmetic state_dict bit-identical), so a language run with this
    # off is not a weaker language model -- it is a model in which no token can see
    # any other token. There is no legitimate `task = language, attention = False`
    # run, so this is not left to a config file to get right.
    mc["attention"] = True

    # PROTOCOL NUMBERS, so setdefault: config_language.json, the frozen
    # canonical_spec_language.json and the CLI all win over these. See the
    # LANGUAGE_*_DEFAULT comments -- seq_len in particular is GATE L5's to freeze,
    # from T-L7.0's measurement on 6 GB of VRAM, not this file's to assert.
    dc.setdefault("seq_len", LANGUAGE_SEQ_LEN_DEFAULT)
    dc.setdefault("vocab_size", LANGUAGE_VOCAB_SIZE_DEFAULT)
    mc.setdefault("n_heads", LANGUAGE_N_HEADS_DEFAULT)
    # plan_language.md §5: the LM head is tied to the token embedding. Named here
    # so `--task language` alone builds the model the plan specifies; a config may
    # still untie it as a labelled ablation.
    mc.setdefault("tie_lm_head", True)

    # Caught here rather than inside nn.MultiheadAttention, which raises the same
    # arithmetic constraint much later and with no mention of which config field is
    # wrong.
    d_model = int(mc.get("d_model", 256))
    n_heads = int(mc["n_heads"])
    if n_heads < 1 or d_model % n_heads != 0:
        raise ValueError(
            f"model.n_heads = {n_heads} does not divide model.d_model = {d_model}. "
            "Causal self-attention splits d_model into n_heads equal head "
            f"dimensions. Canonical is n_heads = {LANGUAGE_N_HEADS_DEFAULT} at "
            "d_model = 256 (plan_language.md §6.6), identical across MoE, MoR and "
            "MoRE -- attention hyperparameters are part of the frozen protocol and "
            "are not tuned per architecture."
        )

    # OWED TO L-4, recorded rather than half-done: the shared defaults still fill in
    # `model.max_steps`, `model.step_feat_dim`, `data.max_val` and `data.pad_value`,
    # which are arithmetic concepts that plan_language.md §10.1 lists as ABSENT from
    # config_language.json. Deleting them belongs with the engine/model migration
    # that stops reading them; removing them now would make a resolved language
    # config honest about four keys while the code still expects them, which trades
    # a cosmetic inaccuracy for a KeyError.

    # T-L1.1 RUN-NAME REPAIR. cli.py runs apply_architecture BEFORE apply_task, so
    # a `--task language` run has already had `phaseB_<Arch>` stamped in as the
    # default name by the time this runs. Replace it -- but ONLY when it is still
    # exactly that stamped default, character for character. Anything else is a name
    # someone chose (a config file, a diagnostic label) and overwriting it would
    # rename a run its author had already named, which is how a dense-routing
    # ablation loses the label enforce_routing_mode requires.
    log  = cfg.setdefault("logging", {})
    arch = str(cfg.get("architecture") or "").lower()
    if arch in ARCHITECTURES:
        stamped_arith = canonical_run_name(arch, None, TASK_ARITHMETIC)
        if log.get("run_name") in (None, stamped_arith):
            log["run_name"] = canonical_run_name(arch, None, TASK_LANGUAGE)
    return cfg


def apply_task(cfg: dict, task: str) -> dict:
    """
    Stamp the task-definitional fields into cfg, in place, and return it.

    arithmetic : no-op by construction -- the key is not even written, so the
                 resolved config is byte-identical to a pre-task-axis resolution
                 and no arithmetic config_hash moves (resolve_task, T-L1.0).
    language   : causal self-attention on, LM shape fields present, family_cls
                 forced to 0.0 and refused if declared non-zero (§7.3).

    Deliberately parallel to apply_architecture: same in-place-and-return shape,
    same "stamp definitional fields, let later overrides win" contract, and the
    same placement rule -- cli.py calls it immediately after apply_architecture and
    BEFORE the override blocks, so an explicit `--family_cls` on a language run
    hits a refusal instead of quietly winning (T-L1.2).
    """
    t = str(task).lower()
    if t not in TASKS:
        raise ValueError(f"task must be one of {TASKS}, got {task!r}")

    if t == TASK_ARITHMETIC:
        # Not `cfg["task"] = "arithmetic"`. Writing the default in is the one thing
        # this axis must never do: it would move every arithmetic config_hash. An
        # explicit `--task arithmetic` and no flag at all must produce the same
        # bytes, so they produce the same bytes.
        cfg.pop("task", None)
        return cfg

    cfg["task"] = t
    # declared_family_cls: at THIS entry point the caller is the CLI, and
    # load_config_defaults has already run, so a 0.5 sitting in loss_weights is
    # indistinguishable from the arithmetic default and must be treated as
    # undeclared. cli.py's --family_cls block does the refusing for an explicit
    # flag, where the declaration is unambiguous (T-L1.2).
    return _apply_language_block(cfg, declared_family_cls=None)


# ---------------------------------------------------------------------------
# Routing-mode gate  (T2.4 / updated_rules.md 1.1 ablation F / plan.md 4.1)
# ---------------------------------------------------------------------------

DENSE_ABLATION_LABEL = "dense_routing_ablation"


def enforce_routing_mode(cfg: dict) -> str:
    """
    Resolve model.routing_mode and refuse dense routing on any unlabelled run.

    plan.md 4.1 states the dense evaluate-all-and-blend path "may remain only
    as: MoRE - Dense Routing Ablation". The danger is not that the dense path
    exists -- it is that a dense run gets written up as if it were MoE, because
    nothing in the output distinguishes them. So the gate is on the RUN LABEL,
    not on a quiet boolean: to use dense routing you must name the run
    something containing "dense_routing_ablation", which then appears in the
    run directory, the W&B run name, and every provenance record downstream.

    Raises ValueError when routing_mode is dense but the label does not say so.
    """
    mc   = cfg.setdefault("model", {})
    mode = str(mc.setdefault("routing_mode", CANONICAL_ROUTING_MODE)).lower()

    if mode not in ROUTING_MODES:
        raise ValueError(
            f"model.routing_mode must be one of {ROUTING_MODES}, got {mode!r}."
        )

    if mode == DENSE_ABLATION_ROUTING_MODE:
        label = " ".join(str(v) for v in (
            cfg.get("logging", {}).get("run_name", ""),
            cfg.get("experiment_group", ""),
            cfg.get("variant", ""),
        )).lower()
        if DENSE_ABLATION_LABEL not in label:
            raise ValueError(
                f"model.routing_mode = {DENSE_ABLATION_ROUTING_MODE!r} evaluates "
                "every expert on every token and blends by router probability. "
                "That is not MoE: no token is routed, every expert sees every "
                "token, and expert independence is destroyed. plan.md 4.1 and "
                "updated_rules.md 1.1 permit it ONLY as an explicitly labelled "
                f"ablation, so the run label must contain "
                f"{DENSE_ABLATION_LABEL!r}. Got run_name="
                f"{cfg.get('logging', {}).get('run_name', '')!r}, "
                f"experiment_group={cfg.get('experiment_group', '')!r}, "
                f"variant={cfg.get('variant', '')!r}. Set the run name (e.g. "
                f"--run_name phaseB_more_{DENSE_ABLATION_LABEL}) so the dense "
                "path cannot be mistaken for canonical MoE in any table."
            )
        mc["variant"] = cfg.setdefault("variant", DENSE_ABLATION_LABEL)

    mc["routing_mode"] = mode
    return mode
