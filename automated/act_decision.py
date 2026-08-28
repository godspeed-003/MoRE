"""
The controlled ACT decision experiment.

The question
-----------
Does the canonical MoRE halt under **pure unsupervised ACT**, or under the
**supervised operation-complexity curriculum**?  This has to be settled before
`canonical_spec.json` is frozen, because it changes what the depth numbers in the
paper are allowed to mean (ARCHITECTURE.md 5a).

The decision rule is fixed HERE, before any run, on purpose
-----------------------------------------------------------
Writing it into the driver instead of deciding after looking at the table is the
difference between an experiment and tuning (CLAUDE.md 6).

  1. PRIMARY criterion: validation task loss, mean +- std over the seeds.
     `supervised_curriculum` is selected only if its mean beats `pure_act` by
     more than the pooled seed std -- i.e. by more than seed noise.  Otherwise
     `pure_act` wins by default.

  2. Depth allocation error is REPORTED but is NOT a selection criterion.  The
     supervised variant is handed `families.OP_TARGET_DEPTH` as a target, so it
     wins that column by construction.  Selecting on it would be selecting the
     variant that was told the answer, and would simultaneously destroy the
     column's value as evidence: a supervised run's allocation error says
     nothing about what the architecture found.

  3. `pure_act` is the DEFAULT WINNER on ties, because only an unsupervised run
     supports the claim in updated_objective.md.  A tie is a reason to keep the
     honest variant, not a reason to pick the flattering one.

  4. If `pure_act` shows no depth differentiation at all -- per-operation depth
     spread ~ 0, or early-exit rate pinned at ~0 or ~1 -- that is Outcome C for
     adaptive computation (CLAUDE.md 8) and must be REPORTED.  It is not to be
     "fixed" by switching supervision on; supervision would then be a labelled
     ablation whose purpose is to show what the unsupervised model failed to do.

What is matched
---------------
Everything except the halting objective: same seeds, epochs, d_model, blocks,
batch size, lr, weight decay, routing_balance, step_routing, dataset and split.
Only `--halting_supervision` / `--halting_supervision_weight` differ, so the
comparison isolates one variable (CLAUDE.md 6: no causal claim without an
experiment that isolates the variable).

What this is NOT
----------------
These runs are PROXIES.  They use a reduced epoch count and they run before
`lr` / `d_model` / `weight_decay` / `routing_balance` are frozen, so
`experiment_group` stays `exploratory` and the Gate 0 guard would refuse a
canonical claim from them.  No number produced here may enter the headline
table; the only output that survives is the halting-mode decision itself.

Usage
-----
    python automated/act_decision.py                    # run + report
    python automated/act_decision.py --report-only      # re-read existing runs
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE = os.path.join(REPO, "code")
RUNS = os.path.join(REPO, "runs")
PY = sys.executable

sys.path.insert(0, CODE)
from more.families import OP_TARGET_DEPTH          # noqa: E402

# --- the matched protocol -------------------------------------------------
# blocks and batch_size are taken from canonical_spec.json's already-frozen
# enforced_fields, so the decision is made under the canonical protocol as far
# as that protocol exists. epochs is deliberately NOT canonical (50) -- this is
# a decision proxy, and it is labelled as one.
SEEDS = (42, 43, 44)
EPOCHS = 20
BLOCKS = 1
BATCH_SIZE = 768
SUPERVISION_WEIGHT = 0.1     # the value used by the Phase 3 verification runs

VARIANTS = {
    "pure_act": [],
    "supervised_curriculum": ["--halting_supervision",
                              "--halting_supervision_weight",
                              str(SUPERVISION_WEIGHT)],
}

PREFIX = "actdec"


def run_name(variant: str, seed: int) -> str:
    return f"{PREFIX}_{variant}"


def launch(variant: str, seed: int) -> tuple[bool, float, str]:
    """One training run. Returns (ok, wall_seconds, tail_of_stderr)."""
    cmd = [PY, "train.py",
           "--architecture", "more",
           "--epochs", str(EPOCHS),
           "--blocks", str(BLOCKS),
           "--batch_size", str(BATCH_SIZE),
           "--seed", str(seed),
           "--run_name", run_name(variant, seed)] + VARIANTS[variant]
    env = dict(os.environ, WANDB_MODE="disabled")
    t0 = time.time()
    p = subprocess.run(cmd, cwd=CODE, env=env, capture_output=True, text=True)
    return p.returncode == 0, time.time() - t0, (p.stderr or "")[-500:]


def newest_run_dir(variant: str, seed: int) -> str | None:
    """Newest run directory for this (variant, seed).

    Selected by mtime, never alphabetically: a stale directory from an earlier
    attempt sorts unpredictably against the current one and would be read
    silently in its place.
    """
    stem = f"{run_name(variant, seed)}_seed{seed}"
    cands = [d for d in glob.glob(os.path.join(RUNS, stem + "*"))
             if os.path.exists(os.path.join(d, "metrics.json"))]
    if not cands:
        return None
    return max(cands, key=os.path.getmtime)


def read_run(variant: str, seed: int) -> dict | None:
    rd = newest_run_dir(variant, seed)
    if rd is None:
        return None
    m = json.load(open(os.path.join(rd, "metrics.json"), encoding="utf-8"))
    rc = json.load(open(os.path.join(rd, "resolved_config.json"),
                        encoding="utf-8"))
    prov = rc.get("provenance", {})

    # The run must AGREE that it is the variant we asked for. A silently
    # ignored flag would otherwise show up as "supervision makes no
    # difference", which is a conclusion, not a bug report.
    declared = prov.get("halting_mode") or prov.get("halt_target_mode")
    return {
        "dir": os.path.basename(rd),
        "declared_mode": declared,
        "variant_label": prov.get("variant"),
        "config_hash": prov.get("config_hash"),
        "resolved_seed": prov.get("resolved_seed"),
        "val_task_loss": m.get("val/task_loss"),
        "train_task_loss": m.get("train/task_loss"),
        "avg_depth": m.get("train/avg_recursion_steps"),
        "alloc_err_abs": m.get("depth/allocation_error_abs"),
        "alloc_err_rel": m.get("depth/allocation_error_rel"),
        "early_exit": m.get("halt/early_exit_rate"),
        "forced_exit": m.get("halt/forced_exit_rate"),
        "halt_mass": m.get("halt/mean_halt_mass"),
        "ponder": m.get("train/ponder_cost"),
        "halt_sup_loss": m.get("train/halting_supervision_loss"),
        "route_acc": m.get("val/routing_accuracy"),
        "route_hung": m.get("val/routing_hungarian_accuracy"),
        "per_op_depth": {k.rsplit("/", 1)[1]: v for k, v in m.items()
                         if k.startswith("recursion/avg_depth_by_op/")},
        "depth_dist": {k.rsplit("/", 1)[1]: v for k, v in m.items()
                       if k.startswith("depth_dist/")},
        "metrics": m,
    }


def mean_std(xs: list[float]) -> tuple[float | None, float | None]:
    """Mean and SAMPLE std (n-1). None when undefined, never 0.0 as a stand-in."""
    vals = [float(x) for x in xs if isinstance(x, (int, float))]
    if not vals:
        return None, None
    mu = sum(vals) / len(vals)
    if len(vals) < 2:
        return mu, None          # std of one sample is undefined, not zero
    var = sum((v - mu) ** 2 for v in vals) / (len(vals) - 1)
    return mu, math.sqrt(var)


def fmt(v, spec=".6f"):
    if v is None:
        return "N/A"
    if isinstance(v, str):
        return v
    return format(v, spec)


def depth_spread(per_op: dict[str, float]) -> float | None:
    """Std of the per-operation mean depths.

    This is the differentiation measure: a model that gives every operation the
    same depth has spread 0 and is not allocating adaptively, however good its
    task loss is. Reported separately from allocation error because a model can
    have low spread AND low error by sitting on the target mean.
    """
    _, sd = mean_std(list(per_op.values()))
    return sd


def curriculum_alignment(per_op: dict[str, float]) -> float | None:
    """Pearson r between measured per-operation depth and OP_TARGET_DEPTH.

    ADDED AFTER THE RUNS, and recorded as such. It is a reported diagnostic, not
    a selection criterion, and it does not change the decision -- `pure_act` was
    already selected by the pre-registered primary criterion and the tie rule.

    Why it had to be added: rule 4 above tests depth SPREAD, i.e. whether the
    model gives different operations different depths. Spread is necessary but
    not sufficient. A model can differentiate depth strongly and still allocate
    it to the wrong operations, and the spread number cannot tell the two apart.
    Alignment can, because it asks whether the differentiation points the way the
    curriculum does. For a supervised run r is high by construction and carries
    no information; for `pure_act` it is the whole question.

    Read it across seeds, never on one: a differentiation that is real but
    arbitrary shows up as an r that changes SIGN between seeds, which no single
    run can reveal.
    """
    common = [o for o in OP_TARGET_DEPTH if o in per_op]
    if len(common) < 3:
        return None
    tgt = [float(OP_TARGET_DEPTH[o]) for o in common]
    got = [float(per_op[o]) for o in common]
    mt, mg = sum(tgt) / len(tgt), sum(got) / len(got)
    num = sum((a - mt) * (b - mg) for a, b in zip(tgt, got))
    dt = math.sqrt(sum((a - mt) ** 2 for a in tgt))
    dg = math.sqrt(sum((b - mg) ** 2 for b in got))
    if dt == 0.0 or dg == 0.0:
        return None          # undefined, not 0.0 (T6.4: no sentinel)
    return num / (dt * dg)


def report(rows: dict[str, dict[int, dict]]) -> int:
    print("\n" + "=" * 78)
    print("ACT DECISION EXPERIMENT -- pure_act vs supervised_curriculum")
    print("=" * 78)
    print(f"matched protocol: architecture=more  epochs={EPOCHS}  "
          f"blocks={BLOCKS}  batch_size={BATCH_SIZE}  seeds={list(SEEDS)}")
    print(f"only difference : halting_supervision "
          f"(weight {SUPERVISION_WEIGHT} when on)")
    print("PROXY RUNS -- reduced epochs, hyperparameters not yet frozen. "
          "No number below may enter the headline table.")

    problems = []
    summary = {}

    for variant in VARIANTS:
        got = rows.get(variant, {})
        print(f"\n-- {variant} " + "-" * (74 - len(variant)))
        print(f"{'seed':>5}  {'val_task_loss':>13}  {'avg_depth':>9}  "
              f"{'alloc_abs':>9}  {'alloc_rel':>9}  {'early':>7}  "
              f"{'forced':>7}  {'op_spread':>9}  {'align_r':>8}")
        for seed in SEEDS:
            r = got.get(seed)
            if r is None:
                print(f"{seed:>5}  {'MISSING':>13}")
                problems.append(f"{variant} seed {seed}: no run directory")
                continue
            # A flag that was silently ignored would look like "supervision
            # makes no difference", which is a conclusion rather than a bug.
            if r["declared_mode"] != variant:
                problems.append(
                    f"{variant} seed {seed}: run declares halting_mode="
                    f"{r['declared_mode']!r}")
            if r["resolved_seed"] != seed:
                problems.append(
                    f"{variant} seed {seed}: resolved_seed="
                    f"{r['resolved_seed']}")
            print(f"{seed:>5}  {fmt(r['val_task_loss']):>13}  "
                  f"{fmt(r['avg_depth'], '.3f'):>9}  "
                  f"{fmt(r['alloc_err_abs'], '.3f'):>9}  "
                  f"{fmt(r['alloc_err_rel'], '.3f'):>9}  "
                  f"{fmt(r['early_exit'], '.4f'):>7}  "
                  f"{fmt(r['forced_exit'], '.4f'):>7}  "
                  f"{fmt(depth_spread(r['per_op_depth']), '.3f'):>9}  "
                  f"{fmt(curriculum_alignment(r['per_op_depth']), '+.3f'):>8}")

        present = [got[s] for s in SEEDS if s in got]
        vl_mu, vl_sd = mean_std([r["val_task_loss"] for r in present])
        ae_mu, ae_sd = mean_std([r["alloc_err_abs"] for r in present])
        sp_mu, sp_sd = mean_std([depth_spread(r["per_op_depth"])
                                 for r in present])
        al = [curriculum_alignment(r["per_op_depth"]) for r in present]
        al_mu, al_sd = mean_std(al)
        dp_mu, _ = mean_std([r["avg_depth"] for r in present])
        summary[variant] = {"n": len(present),
                            "val_mu": vl_mu, "val_sd": vl_sd,
                            "alloc_mu": ae_mu, "alloc_sd": ae_sd,
                            "spread_mu": sp_mu, "spread_sd": sp_sd,
                            "align": al, "align_mu": al_mu, "align_sd": al_sd,
                            "depth_mu": dp_mu}
        print(f"  mean+-std  val_task_loss = {fmt(vl_mu)} +- {fmt(vl_sd)}"
              f"   alloc_err_abs = {fmt(ae_mu, '.3f')} +- {fmt(ae_sd, '.3f')}"
              f"   op_depth_spread = {fmt(sp_mu, '.3f')} +- {fmt(sp_sd, '.3f')}"
              f"   align_r = {fmt(al_mu, '+.3f')} +- {fmt(al_sd, '.3f')}")

    # -- the decision, applying the rule stated at the top of this file ----
    print("\n" + "=" * 78)
    print("DECISION")
    print("=" * 78)

    if problems:
        print("The comparison is NOT valid. Fix these first (CLAUDE.md 7: stop, "
              "report, identify, fix, re-run):")
        for p in problems:
            print("  * " + p)
        return 1

    pa, sc = summary["pure_act"], summary["supervised_curriculum"]
    if pa["val_mu"] is None or sc["val_mu"] is None:
        print("No val_task_loss recorded for one arm; cannot decide.")
        return 1

    gap = pa["val_mu"] - sc["val_mu"]          # > 0 means supervised is better
    pooled = max(x for x in (pa["val_sd"] or 0.0, sc["val_sd"] or 0.0))
    print(f"primary criterion -- validation task loss")
    print(f"  pure_act              {fmt(pa['val_mu'])} +- {fmt(pa['val_sd'])}")
    print(f"  supervised_curriculum {fmt(sc['val_mu'])} +- {fmt(sc['val_sd'])}")
    print(f"  gap (pure - supervised) = {gap:+.6f}   "
          f"largest seed std = {pooled:.6f}")

    if gap > pooled:
        choice = "supervised_curriculum"
        why = ("supervised_curriculum beats pure_act on validation task loss by "
               "more than seed noise")
    else:
        choice = "pure_act"
        why = ("the val-task-loss gap does not exceed seed noise, and pure_act "
               "is the default winner on ties (rule 3): only an unsupervised "
               "run supports the adaptive-computation claim")

    print(f"\n  -> CANONICAL HALTING MODE = {choice}")
    print(f"     because {why}.")

    print("\nreported, NOT used to select (rule 2) -- the supervised arm is "
          "handed the target depths:")
    print(f"  alloc_err_abs   pure_act {fmt(pa['alloc_mu'], '.3f')} "
          f"vs supervised {fmt(sc['alloc_mu'], '.3f')}")
    print(f"  op_depth_spread pure_act {fmt(pa['spread_mu'], '.3f')} "
          f"vs supervised {fmt(sc['spread_mu'], '.3f')}")
    print(f"  avg_depth       pure_act {fmt(pa['depth_mu'], '.3f')} "
          f"vs supervised {fmt(sc['depth_mu'], '.3f')}")
    print(f"  align_r         pure_act {fmt(pa['align_mu'], '+.3f')} "
          f"vs supervised {fmt(sc['align_mu'], '+.3f')}  "
          f"(supervised is high BY CONSTRUCTION -- it was given the targets)")

    # rule 4 -- the honesty check, in the two parts it actually has
    spread = pa["spread_mu"]
    if spread is not None and spread < 0.10:
        print("\n  OUTCOME C SIGNAL (rule 4, spread): under pure_act the "
              f"per-operation mean depths have std {spread:.3f} -- effectively "
              "one depth for every operation. The unsupervised model is NOT "
              "allocating adaptively.")

    al = [r for r in pa["align"] if r is not None]
    if al and (min(al) < 0.0 < max(al) or abs(pa["align_mu"] or 0.0) < 0.5):
        print("\n  OUTCOME C SIGNAL (alignment, diagnostic added after the "
              "runs): under pure_act the per-operation depths DO differ "
              f"(spread {fmt(spread, '.3f')}), but their correlation with "
              f"OP_TARGET_DEPTH is {fmt(pa['align_mu'], '+.3f')} over seeds "
              f"{[round(r, 3) for r in al]}"
              + ("  -- the SIGN FLIPS between seeds" if min(al) < 0.0 < max(al)
                 else "")
              + ".\n     Reading: unsupervised ACT learns a non-trivial but "
              "essentially ARBITRARY depth partition, not the "
              "operation-complexity curriculum. Report it that way. It is not "
              "to be fixed by switching supervision on; supervision would be a "
              "labelled ablation demonstrating exactly this gap.")

    json.dump({"seeds": list(SEEDS), "epochs": EPOCHS, "blocks": BLOCKS,
               "batch_size": BATCH_SIZE,
               "supervision_weight": SUPERVISION_WEIGHT,
               "summary": summary, "decision": choice, "reason": why},
              open(os.path.join(REPO, "automated", "act_decision_result.json"),
                   "w", encoding="utf-8"), indent=2)
    print("\nwritten: automated/act_decision_result.json")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report-only", action="store_true",
                    help="Re-read existing run directories without training")
    args = ap.parse_args()

    if not args.report_only:
        total = len(VARIANTS) * len(SEEDS)
        i = 0
        for variant in VARIANTS:
            for seed in SEEDS:
                i += 1
                print(f"[{i}/{total}] {variant} seed {seed} ...",
                      flush=True)
                ok, secs, err = launch(variant, seed)
                print(f"        {'ok' if ok else 'FAILED'} in {secs/60:.1f} min",
                      flush=True)
                if not ok:
                    print("        stderr tail: " + err, flush=True)

    rows = {v: {s: r for s in SEEDS
                if (r := read_run(v, s)) is not None}
            for v in VARIANTS}
    return report(rows)


if __name__ == "__main__":
    sys.exit(main())
