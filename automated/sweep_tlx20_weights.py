# -*- coding: utf-8 -*-
"""sweep_tlx20_weights.py -- T-LX.20: why MoRE saturates depth on the canonical corpus.

    python automated/sweep_tlx20_weights.py
    python automated/sweep_tlx20_weights.py --axes balance        # just one axis
    python automated/sweep_tlx20_weights.py --subset 0.25         # bigger budget

EXPLORATORY. Every point declares `experiment_group = lang_sweep_tlx20_*`, which
`export_results.py` refuses for the canonical table, and `main()` asserts the prefix
before launching anything (CLAUDE.md 9: drivers in `automated/` must never launch a
canonical run).

WHY THIS EXISTS, AND WHY IT IS NOT THE SWEEP THAT WAS ASKED FOR
---------------------------------------------------------------
T-LX.19 observed `train/entropy_term_normalized = 0.9999` on the canonical MoRE arm and
read it as the balance loss over-driving the router. Before running anything, check what
the repository already measured: `code/lang_calibration_weights.json` swept
`routing_balance` over {0.001, 0.05, 0.21, 1.0} at T-L7.1 and **0.001 won on both
`val/task_loss` and AMI, and it is the LOWEST value ever tested**. So "lower the balance
weight" is not an untested hypothesis, it is a hypothesis that was tested in the only
direction available and the answer was "0.001 is already the best of the grid".

That makes the balance axis here worth running only because it extends BELOW that floor,
to 1e-4 and to **0.0**. The 0.0 point is the decisive one and it is a falsification test
of T-LX.19's own claim: with the term switched off entirely, if per-token router entropy
is still ~1.0, then the balance loss never caused the near-uniform router and that
sentence in the changelog is wrong.

THE AXIS THE DATA ACTUALLY POINTS AT IS HALTING, AND THE EVIDENCE IS ALREADY ON DISK.
Three independent points, same halting weight 0.001, differing only in how many optimizer
steps the model got:

    dev corpus, 3 epochs    ->  avg_depth 2.70 of 7   (lang_calibration_weights.json)
    dev corpus, 8 epochs    ->  avg_depth 5.16 of 7   (calibrate_lang_weights.py docstring)
    wikitext-103, 3 epochs  ->  avg_depth 6.96 of 7   (the canonical arm, T-LX.19)

Depth is monotone in training steps at fixed ponder weight. The halting weight was frozen
on a corpus ~60x smaller than the one the paper reports, where it produced genuinely
adaptive depth, and it does not transfer. That is a far better explanation of the canonical
arm's 97% forced-exit rate than anything about the router, and it is the difference between
"MoRE cannot allocate depth" (an architectural claim) and "this ponder weight cannot hold
depth down at this corpus scale" (a calibration claim). The two have opposite implications
for the paper, so the experiment has to separate them.

WHAT EACH AXIS DECIDES
----------------------
`balance`  routing_balance in {0.0, 1e-4, 1e-3} at halting 1e-3.
           Read `train/entropy_term_normalized`. If 0.0 does not move it off ~1.0, the
           T-LX.19 diagnosis is falsified and the near-uniform router is a property of
           the task, not of the auxiliary loss.

`halting`  halting in {1e-3, 1e-2, 3e-2, 1e-1} at balance 1e-3, ON THE CANONICAL CORPUS.
           Read `depth/mean`, `halt/forced_exit_rate` and
           `depth/spearman_vs_model_loss` together. The dev grid showed depth DOES respond
           to this knob (2.70 -> 1.48 -> 1.07 at 0.001 -> 0.107 -> 0.5), so the question is
           not whether it works but where it has to sit at 60x the data, and what the
           recovered adaptivity costs in val loss and AMI. The dev grid also showed the
           coupling that makes this non-obvious: raising halting DESTROYED routing
           structure (AMI 0.2907 -> 0.0025), so there may be no weight at which both
           behaviours are alive. A grid that shows that cleanly is a result either way.

`subset`   subset_fraction in {0.025, 0.1, 0.4} at the frozen weights.
           The mechanism check, and the only axis that holds corpus identity fixed while
           varying step count. If depth climbs monotonically here, depth saturation is a
           training-budget effect and the three-point trend above is confirmed under
           control rather than inferred across two different corpora.

PROTOCOL NOTES THAT MATTER FOR READING THE NUMBERS
--------------------------------------------------
- CORPUS is the CANONICAL wikitext-103, not the dev corpus. That is deliberate and is the
  whole point: the saturation being diagnosed does not occur on wikitext-2. It also means
  these numbers may NOT be quoted as results -- `subset_fraction < 1` and a single seed
  make every point a proxy (CLAUDE.md 6). They pick a hypothesis; they do not test one.
- `subset_fraction` subsets TRAIN only, deterministically, by `subset_seed`
  (`engine.py:187`). The author-provided validation split is untouched, so every point
  here is scored on the same val set as the canonical arm.
- One seed (44, the calibration seed). Seed variance is not estimated and no point here
  carries an error bar. A difference smaller than the canonical arm's own seed spread
  (0.0134 nats on MoRE) means nothing.
- One axis at a time, not a cross product, following T-L7.1: the terms act on different
  parts of the model and a full grid costs 3x the runs to answer separable questions.
- The POS partition's own load entropy is read from THIS corpus's manifest (0.8942 for
  wikitext-103), never pasted. T-LX.6: the dev value is 0.8884 and the difference flips
  the sign of the "is the router more uniform than the data" comparison.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
CODE = os.path.join(REPO, "code")

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# THE CANONICAL CORPUS. See the docstring -- the effect under study does not exist on the
# dev corpus, so calibrating on the dev corpus is what produced the defect.
CORPUS = "wikitext-103"
SEQ_LEN, BATCH = 256, 64          # batch 64 at seq 256 is T-L7.0's 6 GB recommendation
EPOCHS = 3
SEED = 44
GROUP_PREFIX = "lang_sweep_tlx20"

BASE_BALANCE, BASE_HALTING, BASE_SUBSET = 0.001, 0.001, 0.1

AXIS_BALANCE = [0.0, 1e-4, 1e-3]
AXIS_HALTING = [1e-3, 1e-2, 3e-2, 1e-1]
AXIS_SUBSET = [0.025, 0.1, 0.4]

OUT = os.path.join(CODE, "lang_sweep_tlx20.json")

# Keys pulled from each point's metrics.json. Grouped by the question they answer so a
# reader of the output JSON can tell which column settles which axis.
METRIC_KEYS = {
    # quality
    "val_task_loss": "val/task_loss",
    "nats_below_floor": "val/nats_below_bigram_floor",
    "val_perplexity": "val/perplexity",
    # depth -- the halting axis
    "depth_mean": "depth/mean",
    "depth_std": "depth/std",
    "forced_exit_rate": "halt/forced_exit_rate",
    "early_exit_rate": "halt/early_exit_rate",
    "mean_remainder": "halt/mean_remainder",
    "spearman_depth_vs_loss": "depth/spearman_vs_model_loss",
    "train_avg_depth": "train/avg_recursion_steps",
    "ponder_cost": "train/ponder_cost",
    # router -- the balance axis
    "entropy_term_normalized": "train/entropy_term_normalized",
    "load_entropy_normalized": "train/expert_load_entropy_normalized",
    "routing_balance_loss": "train/routing_balance_loss",
    "max_load_fraction": "dispatch/max_load_fraction",
    # specialization, against its own permutation control
    "ami": "val/routing_ami",
    "ami_control_mean": "val/routing_control_pos/ami_control_mean",
    "ami_delta_z": "val/routing_control_pos/ami_delta_z",
    "hungarian": "val/routing_hungarian_accuracy",
    "purity": "val/routing_purity",
    "collapsed_experts": "val/routing_collapsed_experts",
}


def manifest_const(key, fallback=None):
    """A corpus constant read from CORPUS's own manifest. See T-LX.6 in the docstring."""
    path = os.path.join(REPO, "data", "lang", CORPUS, "dataset_meta.json")
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh).get(key, fallback)
    except (OSError, ValueError):
        return fallback


def write_config(dest, routing_balance, halting, subset_fraction):
    """One config per point. Only `--routing_balance` has a CLI flag, so `--config` is the
    supported path for the other two (same reasoning as calibrate_lang_weights.py)."""
    with open(os.path.join(CODE, "config.json"), "r", encoding="utf-8") as fh:
        cfg = json.load(fh)
    cfg["loss_weights"]["routing_balance"] = routing_balance
    cfg["loss_weights"]["halting"] = halting
    # apply_task REFUSES a non-zero family_cls on language (plan_language.md 7.3) and the
    # base config declares the arithmetic 0.5, so this is required, not cosmetic.
    cfg["loss_weights"]["family_cls"] = 0.0
    cfg["training"]["lr"] = 0.001
    cfg["training"]["weight_decay"] = 0.0001
    cfg["model"]["dropout"] = 0.0
    # Written in BOTH sections on purpose: config.py reads `data.subset_fraction` for the
    # proxy guard and `training.subset_fraction` is the one engine.py resolves. Setting
    # only one is the T8-era defect where a run trained on a subset and reported 1.0.
    cfg["data"]["subset_fraction"] = subset_fraction
    cfg["training"]["subset_fraction"] = subset_fraction
    # Three epochs at the default interval of 2 gives two validation points and the
    # last-epoch number is what decides every column here.
    cfg["logging"]["log_interval"] = 1
    with open(dest, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2)
    return dest


def run_point(tag, cfg_path, env):
    """Train one point. Returns (metrics, run_dir_name) or None with the reason printed."""
    group = f"{GROUP_PREFIX}_{tag}"
    assert not group.startswith("canonical"), group
    cmd = [env.get("LANG_PY", sys.executable), "train.py",
           "--architecture", "more", "--task", "language", "--corpus", CORPUS,
           "--config", cfg_path, "--epochs", str(EPOCHS),
           "--batch_size", str(BATCH), "--seq_len", str(SEQ_LEN),
           "--seed", str(SEED), "--experiment_group", group]
    t0 = time.time()
    res = subprocess.run(cmd, cwd=CODE, capture_output=True, text=True, env=env)
    mins = (time.time() - t0) / 60.0
    if res.returncode != 0:
        tail = res.stderr.strip().splitlines()[-1][:200] if res.stderr.strip() else ""
        print(f"  [{tag}] FAILED rc={res.returncode} after {mins:.1f} min: {tail}")
        return None
    # Find the run directory by mtime: the name is a config hash, and reconstructing it
    # here would duplicate run_context's logic and drift from it.
    runs = os.path.join(REPO, "runs")
    cands = [os.path.join(runs, d) for d in os.listdir(runs) if d.startswith("langB_")]
    newest = max(cands, key=os.path.getmtime)
    mp = os.path.join(newest, "metrics.json")
    if not os.path.exists(mp):
        print(f"  [{tag}] no metrics.json in {os.path.basename(newest)} "
              f"(run did not finish)")
        return None
    with open(mp, "r", encoding="utf-8") as fh:
        return json.load(fh), os.path.basename(newest), mins


def flatten(metrics):
    """metrics.json stores slash-joined keys at the top level in some places and nested
    dicts in others. Flatten both so METRIC_KEYS can be a single flat lookup."""
    out = {}

    def walk(obj, prefix=""):
        if isinstance(obj, dict):
            for k, v in obj.items():
                walk(v, f"{prefix}/{k}" if prefix else k)
        else:
            out[prefix] = obj

    walk(metrics)
    return out


def row_for(flat, **fixed):
    row = dict(fixed)
    for name, key in METRIC_KEYS.items():
        row[name] = flat.get(key)
    return row


def save(rows, note):
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump({
            "task": "T-LX.20",
            "status": "EXPLORATORY -- subset_fraction < 1 and one seed. These points PICK "
                      "a hypothesis and may never be quoted as a result (CLAUDE.md 6).",
            "corpus": CORPUS,
            "dataset_version": manifest_const("dataset_version"),
            "corpus_bigram_floor": manifest_const("primary_metric_floor"),
            "pos_partition_own_load_entropy": manifest_const(
                "shuffled_control_marginal_entropy_real"),
            "pos_partition_own_load_entropy_corpus": CORPUS,
            "epochs": EPOCHS, "seq_len": SEQ_LEN, "batch_size": BATCH, "seed": SEED,
            "baseline": {"routing_balance": BASE_BALANCE, "halting": BASE_HALTING,
                         "subset_fraction": BASE_SUBSET},
            "canonical_reference": {
                "note": "The canonical MoRE arm for comparison. Full corpus, 5 seeds, "
                        "batch 48 -- NOT comparable point-to-point with the rows below, "
                        "which are subset proxies at batch 64. Trend only.",
                "val_task_loss": 3.636966, "depth_mean": 6.9552,
                "forced_exit_rate": 0.9734, "spearman_depth_vs_loss": -0.0230,
                "entropy_term_normalized": 0.9999, "ami": 0.1803},
            "note": note,
            "rows": rows,
        }, fh, indent=2)
        fh.write("\n")


HDR = (f"{'axis':8s} {'bal':>7s} {'halt':>6s} {'subs':>5s} {'val':>8s} {'depth':>6s} "
       f"{'forced':>7s} {'rho':>7s} {'H_tok':>6s} {'H_load':>7s} {'ami':>6s} {'min':>6s}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--axes", default="balance,halting,subset",
                    help="comma list of axes to run")
    ap.add_argument("--subset", type=float, default=BASE_SUBSET,
                    help="subset_fraction for the balance and halting axes")
    args = ap.parse_args(argv)

    axes = [a.strip() for a in args.axes.split(",") if a.strip()]
    base_subset = args.subset

    env = dict(os.environ)
    env.setdefault("WANDB_MODE", "offline")
    env.setdefault("WANDB_SILENT", "true")
    env.setdefault("WANDB_DIR", os.environ.get("TEMP", "."))
    env.setdefault("PYTHONIOENCODING", "utf-8")

    tmp = os.path.join(os.environ.get("TEMP", "."), "more_sweep_tlx20")
    os.makedirs(tmp, exist_ok=True)

    # (axis, balance, halting, subset). Deduplicated on the weight triple so the shared
    # baseline point is trained once and attributed to the first axis that wants it.
    plan, seen = [], set()
    if "balance" in axes:
        for b in AXIS_BALANCE:
            key = (b, BASE_HALTING, base_subset)
            if key not in seen:
                seen.add(key)
                plan.append(("balance",) + key)
    if "halting" in axes:
        for h in AXIS_HALTING:
            key = (BASE_BALANCE, h, base_subset)
            if key not in seen:
                seen.add(key)
                plan.append(("halting",) + key)
    if "subset" in axes:
        for s in AXIS_SUBSET:
            key = (BASE_BALANCE, BASE_HALTING, s)
            if key not in seen:
                seen.add(key)
                plan.append(("subset",) + key)

    print(f"[tlx20] {len(plan)} points on {CORPUS}, {EPOCHS} epochs, seed {SEED}, "
          f"batch {BATCH}, seq {SEQ_LEN}")
    print(f"[tlx20] group prefix {GROUP_PREFIX}_* -- EXPLORATORY, never canonical")
    print(f"[tlx20] -> {OUT}")
    print()
    print(HDR)
    print("-" * len(HDR))

    rows = []
    for axis, bal, halt, subs in plan:
        tag = f"b{bal:g}_h{halt:g}_s{subs:g}".replace(".", "p").replace("-", "m")
        cfg = write_config(os.path.join(tmp, f"cfg_{tag}.json"), bal, halt, subs)
        got = run_point(tag, cfg, env)
        if not got:
            # Save what exists rather than losing the completed points to one failure.
            save(rows, f"INCOMPLETE -- point {tag} failed; {len(rows)} of {len(plan)} done")
            continue
        metrics, run_dir, mins = got
        flat = flatten(metrics)
        row = row_for(flat, axis=axis, routing_balance=bal, halting=halt,
                      subset_fraction=subs, run=run_dir, wall_minutes=round(mins, 2))
        rows.append(row)

        def n(key, w=7, p=4):
            v = row.get(key)
            return f"{v:>{w}.{p}f}" if isinstance(v, (int, float)) else f"{'N/A':>{w}s}"

        print(f"{axis:8s} {bal:>7.4g} {halt:>6.3g} {subs:>5.3g} "
              f"{n('val_task_loss', 8)} {n('depth_mean', 6, 3)} "
              f"{n('forced_exit_rate')} {n('spearman_depth_vs_loss')} "
              f"{n('entropy_term_normalized', 6, 3)} {n('load_entropy_normalized')} "
              f"{n('ami', 6, 3)} {mins:>6.1f}")
        save(rows, f"{len(rows)} of {len(plan)} points complete")

    save(rows, f"COMPLETE -- {len(rows)} of {len(plan)} points")
    print()
    print(f"[tlx20] {len(rows)}/{len(plan)} points -> {OUT}")
    return 0 if len(rows) == len(plan) else 1


if __name__ == "__main__":
    sys.exit(main())
