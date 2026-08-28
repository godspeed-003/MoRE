"""
The controlled routing-supervision experiment (T8.0b, and the measurement for
T10.E).

The question
-----------
`code/config.json` ships `loss_weights.step_routing = 0.5`, and engine.py:485
builds that term as `0.01 * oracle_routing_ce + 0.3 * step_cls_loss`.  Any
non-zero weight therefore trains the router against the ORACLE EXPERT INDEX.

`plan.md` E already states the rule: **supervision OFF is canonical, supervision
ON is an explicitly labelled oracle-routing ablation, and the two may not share a
headline table.**  So this experiment does not decide the canonical value -- the
plan decided it.  What it has to establish is the thing the paper actually needs:
WHAT THE ROUTER LEARNS WITHOUT THE ORACLE, and what the cost of removing the
oracle is.

Why this had to be run before freezing the spec
-----------------------------------------------
Every routing number measured so far was measured at `step_routing = 0.5`, where
`raw_accuracy = hungarian_accuracy = ami = purity = 1.0` and the recovered
Hungarian assignment is the identity permutation.  Under T6.3's interpretation
contract that is the "raw ~= matched -> the index itself is supervised" case:
expected, not a finding.  Freezing the spec at 0.5 would freeze a headline
specialization claim that measures the supervision signal.

The decision rule, fixed HERE, before any run
---------------------------------------------
  1. CANONICAL VALUE IS NOT UP FOR SELECTION.  `step_routing = 0.0` is canonical
     because plan.md E says so.  No table produced below can change that; a
     supervised arm that wins on every column is still an ablation.

  2. What IS being measured, and reported either way:
     (a) the task-loss cost of removing the oracle -- val task loss, mean +- std;
     (b) whether an UNSUPERVISED router partitions the operation space at all,
         read through T6.3's contract, NOT through raw accuracy alone.

  3. The reading of (b) is fixed in advance, since expert indices carry no
     semantic identity:
       * hungarian_accuracy / AMI / purity near chance (1/6 ~ 0.167, AMI ~ 0)
         -> NO partition.  Outcome C for the specialization claim.  The paper
         then reports that MoRE's experts do not differentiate by operation
         family without supervision, and every specialization number becomes a
         supervised ablation result, labelled as such.
       * hungarian_accuracy well above chance with AMI > 0 while RAW accuracy
         stays low -> a REAL partition exists under relabelling.  This is the
         publishable specialization result, and it is only credible from the
         unsupervised arm.
       * raw ~= hungarian ~= 1.0 -> the index is supervised.  Only the ON arm may
         legitimately show this, and there it is a sanity check, not a result.

  4. The ON arm exists to bound the gap, not to be chosen.  If the unsupervised
     router finds no partition, the ON arm shows what the architecture is capable
     of WHEN TOLD -- i.e. it becomes the labelled ablation that demonstrates the
     gap, exactly as with `supervised_curriculum` in the ACT experiment.

  5. Sub-criterion on collapse.  `collapsed_experts` is reported for both arms.
     An unsupervised router that routes everything to one expert has AMI ~ 0 for
     a different reason than one that partitions arbitrarily, and the two must
     not be reported as the same failure.

What is matched
---------------
Everything except `loss_weights.step_routing`: same seeds, epochs, d_model,
blocks, batch size, lr, weight decay, routing_balance, halting mode (pure_act,
per the T8.0a decision), dataset and split.  Only `--step_routing` differs
(CLAUDE.md 6: no causal claim without an experiment that isolates the variable).

What this is NOT
----------------
PROXY RUNS.  Reduced epoch count, and `lr` / `d_model` / `weight_decay` /
`routing_balance` are still null in canonical_spec.json, so `experiment_group`
stays `exploratory` and the Gate 0 guard would refuse a canonical claim from
them.  No number here may enter the headline table.  What survives is the
decision and the qualitative reading of (b).

Usage
-----
    python automated/routing_supervision_decision.py               # run + report
    python automated/routing_supervision_decision.py --report-only
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

# --- the matched protocol -------------------------------------------------
# Identical to the ACT decision proxy so the two decisions are made under the
# same conditions and can be read against each other. Halting is pure_act, which
# is what T8.0a selected -- deciding routing supervision under a halting mode
# that was already rejected would confound the two.
SEEDS = (42, 43, 44)
EPOCHS = 20
BLOCKS = 1
BATCH_SIZE = 768
SUPERVISED_WEIGHT = 0.5      # the value currently sitting in code/config.json
CHANCE = 1.0 / 6.0           # six canonical experts

VARIANTS = {
    "unsupervised_router": ["--step_routing", "0.0"],
    "oracle_supervised_router": ["--step_routing", str(SUPERVISED_WEIGHT)],
}

PREFIX = "routedec"


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
    """Newest run dir for this (variant, seed), by mtime -- never alphabetically.

    A stale directory from an earlier attempt sorts unpredictably against the
    current one and would be read silently in its place.
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
    pim = m.get("routing_permutation_invariant", {}) or {}

    def num(v):
        """N/A and sentinels stay None; they are not measurements (T6.4)."""
        return float(v) if isinstance(v, (int, float)) else None

    return {
        "dir": os.path.basename(rd),
        # The run must AGREE it is the arm we launched. A silently ignored flag
        # would surface as "supervision makes no difference", which is a
        # conclusion, not a bug report.
        "sup_enabled": prov.get("routing_supervision_enabled"),
        "sup_weight": num(prov.get("routing_supervision_weight")),
        "halting_mode": prov.get("halting_mode") or prov.get("halt_target_mode"),
        "resolved_seed": prov.get("resolved_seed"),
        "config_hash": prov.get("config_hash"),
        "val_task_loss": num(m.get("val/task_loss")),
        "train_task_loss": num(m.get("train/task_loss")),
        "raw_acc": num(pim.get("raw_accuracy")),
        "hung_acc": num(pim.get("hungarian_accuracy")),
        "ami": num(pim.get("ami")),
        "purity": num(pim.get("purity")),
        "macro_recall": num(pim.get("macro_recall")),
        "assignment": pim.get("hungarian_assignment"),
        "collapsed": pim.get("collapsed_experts"),
        "per_family": pim.get("per_family") or {},
        "entropy_norm": num(m.get("train/entropy_term_normalized")),
        "load_entropy": num(m.get("train/expert_load_entropy_normalized")),
        "cos_max": num(m.get("diag/max_pairwise_cosine_sim")),
        "step_route_loss": num(m.get("train/step_routing_loss")),
        "avg_depth": num(m.get("train/avg_recursion_steps")),
        "metrics": m,
    }


def mean_std(xs) -> tuple[float | None, float | None]:
    """Mean and SAMPLE std (n-1). None when undefined -- never 0.0 as a stand-in."""
    vals = [float(x) for x in xs if isinstance(x, (int, float))]
    if not vals:
        return None, None
    mu = sum(vals) / len(vals)
    if len(vals) < 2:
        return mu, None
    var = sum((v - mu) ** 2 for v in vals) / (len(vals) - 1)
    return mu, math.sqrt(var)


def fmt(v, spec=".6f"):
    if v is None:
        return "N/A"
    if isinstance(v, str):
        return v
    return format(v, spec)


def report(rows: dict[str, dict[int, dict]]) -> int:
    print("\n" + "=" * 84)
    print("ROUTING-SUPERVISION EXPERIMENT (T8.0b / T10.E)")
    print("unsupervised_router (step_routing=0.0, CANONICAL per plan.md E)  vs")
    print(f"oracle_supervised_router (step_routing={SUPERVISED_WEIGHT}, "
          "LABELLED ABLATION)")
    print("=" * 84)
    print(f"matched protocol: architecture=more  halting=pure_act  "
          f"epochs={EPOCHS}  blocks={BLOCKS}  batch_size={BATCH_SIZE}  "
          f"seeds={list(SEEDS)}")
    print("only difference : loss_weights.step_routing")
    print("PROXY RUNS -- reduced epochs, hyperparameters not yet frozen. "
          "No number below may enter the headline table.")
    print(f"chance level for 6 experts = {CHANCE:.3f}")

    problems = []
    summary = {}

    for variant in VARIANTS:
        got = rows.get(variant, {})
        want_sup = variant == "oracle_supervised_router"
        print(f"\n-- {variant} " + "-" * (80 - len(variant)))
        print(f"{'seed':>5}  {'val_task_loss':>13}  {'raw_acc':>8}  "
              f"{'hung_acc':>8}  {'ami':>7}  {'purity':>7}  {'H_load':>7}  "
              f"{'collapsed':>12}  assignment")
        for seed in SEEDS:
            r = got.get(seed)
            if r is None:
                print(f"{seed:>5}  {'MISSING':>13}")
                problems.append(f"{variant} seed {seed}: no run directory")
                continue
            # A silently ignored --step_routing would look like "supervision
            # makes no difference", which is a conclusion rather than a bug.
            if bool(r["sup_enabled"]) != want_sup:
                problems.append(
                    f"{variant} seed {seed}: run declares "
                    f"routing_supervision_enabled={r['sup_enabled']!r}")
            exp_w = SUPERVISED_WEIGHT if want_sup else 0.0
            if r["sup_weight"] != exp_w:
                problems.append(
                    f"{variant} seed {seed}: routing_supervision_weight="
                    f"{r['sup_weight']} (expected {exp_w})")
            if r["halting_mode"] != "pure_act":
                problems.append(
                    f"{variant} seed {seed}: halting_mode="
                    f"{r['halting_mode']!r}, expected pure_act -- the two "
                    "decisions would be confounded")
            if r["resolved_seed"] != seed:
                problems.append(
                    f"{variant} seed {seed}: resolved_seed={r['resolved_seed']}")
            print(f"{seed:>5}  {fmt(r['val_task_loss']):>13}  "
                  f"{fmt(r['raw_acc'], '.4f'):>8}  "
                  f"{fmt(r['hung_acc'], '.4f'):>8}  "
                  f"{fmt(r['ami'], '.4f'):>7}  "
                  f"{fmt(r['purity'], '.4f'):>7}  "
                  f"{fmt(r['load_entropy'], '.4f'):>7}  "
                  f"{str(r['collapsed']):>12}  {r['assignment']}")

        present = [got[s] for s in SEEDS if s in got]
        agg = {}
        for key in ("val_task_loss", "raw_acc", "hung_acc", "ami", "purity",
                    "macro_recall", "load_entropy", "avg_depth"):
            mu, sd = mean_std([r[key] for r in present])
            agg[key] = (mu, sd)
        summary[variant] = {"n": len(present),
                            **{k: {"mean": v[0], "std": v[1]}
                               for k, v in agg.items()},
                            "collapsed": [r["collapsed"] for r in present],
                            "assignments": [r["assignment"] for r in present]}
        print(f"  mean+-std  val_task_loss = {fmt(agg['val_task_loss'][0])} "
              f"+- {fmt(agg['val_task_loss'][1])}")
        print(f"             raw_acc = {fmt(agg['raw_acc'][0], '.4f')} "
              f"+- {fmt(agg['raw_acc'][1], '.4f')}   "
              f"hung_acc = {fmt(agg['hung_acc'][0], '.4f')} "
              f"+- {fmt(agg['hung_acc'][1], '.4f')}   "
              f"ami = {fmt(agg['ami'][0], '.4f')} "
              f"+- {fmt(agg['ami'][1], '.4f')}   "
              f"purity = {fmt(agg['purity'][0], '.4f')} "
              f"+- {fmt(agg['purity'][1], '.4f')}")

    print("\n" + "=" * 84)
    print("READING (rules 1-5, fixed before the runs)")
    print("=" * 84)

    if problems:
        print("The comparison is NOT valid. Fix these first (CLAUDE.md 7: stop, "
              "report, identify, fix, re-run):")
        for p in problems:
            print("  * " + p)
        return 1

    un = summary["unsupervised_router"]
    sup = summary["oracle_supervised_router"]

    print("rule 1 -- the canonical value is NOT selected here. plan.md E fixes "
          "it:\n  -> CANONICAL loss_weights.step_routing = 0.0 "
          "(routing supervision OFF)")
    print(f"  -> step_routing = {SUPERVISED_WEIGHT} is retained as the labelled "
          "oracle-routing ablation (T10.E), never in the same table.")

    ul, sl = un["val_task_loss"]["mean"], sup["val_task_loss"]["mean"]
    pooled = max(x for x in (un["val_task_loss"]["std"] or 0.0,
                             sup["val_task_loss"]["std"] or 0.0))
    print("\nrule 2a -- task-loss cost of removing the oracle")
    print(f"  unsupervised      {fmt(ul)} +- "
          f"{fmt(un['val_task_loss']['std'])}")
    print(f"  oracle-supervised {fmt(sl)} +- "
          f"{fmt(sup['val_task_loss']['std'])}")
    if ul is not None and sl is not None:
        d = ul - sl
        print(f"  cost = {d:+.6f}   largest seed std = {pooled:.6f}   "
              + ("INSIDE seed noise: removing the oracle costs nothing "
                 "measurable here." if abs(d) <= pooled else
                 ("OUTSIDE seed noise: the oracle term is doing measurable work "
                  "on task loss." if d > 0 else
                  "OUTSIDE seed noise: the unsupervised arm is BETTER on task "
                  "loss.")))

    print("\nrule 2b/3 -- does an UNSUPERVISED router partition the operation "
          "space?")
    ha, am, pu = (un["hung_acc"]["mean"], un["ami"]["mean"],
                  un["purity"]["mean"])
    ra = un["raw_acc"]["mean"]
    print(f"  unsupervised: raw {fmt(ra, '.4f')}  hungarian {fmt(ha, '.4f')}  "
          f"ami {fmt(am, '.4f')}  purity {fmt(pu, '.4f')}  "
          f"(chance {CHANCE:.3f})")
    print(f"  supervised  : raw {fmt(sup['raw_acc']['mean'], '.4f')}  "
          f"hungarian {fmt(sup['hung_acc']['mean'], '.4f')}  "
          f"ami {fmt(sup['ami']['mean'], '.4f')}  "
          f"purity {fmt(sup['purity']['mean'], '.4f')}   "
          "-- high BY CONSTRUCTION, it was given the indices")

    verdict = None
    if ha is None or am is None:
        verdict = "undecidable"
        print("\n  Routing metrics are N/A for the unsupervised arm; cannot "
              "read rule 3.")
    elif ha < CHANCE + 0.10 and (am is None or am < 0.05):
        verdict = "no_partition"
        print("\n  OUTCOME C FOR THE SPECIALIZATION CLAIM (rule 3, case 1): "
              f"hungarian accuracy {ha:.4f} is at chance ({CHANCE:.3f}) and AMI "
              f"{fmt(am, '.4f')} ~ 0. Without the oracle term the router does "
              "NOT partition the operation space.\n     Consequence for the "
              "paper: every specialization number belongs to the labelled "
              "oracle-routing ablation. MoRE does not discover expert "
              "specialization on this task at this scale. Report it.")
    elif ra is not None and ha - ra > 0.15 and am > 0.05:
        verdict = "real_partition_under_relabelling"
        print(f"\n  REAL PARTITION UNDER RELABELLING (rule 3, case 2): raw "
              f"{ra:.4f} << hungarian {ha:.4f}, AMI {am:.4f} > 0. The "
              "unsupervised router groups operations by family but numbers the "
              "experts arbitrarily -- which is exactly what an expert index is "
              "allowed to do (updated_rules.md 8.3).\n     This is the "
              "publishable specialization result, and it is only credible from "
              "this arm.")
    elif ra is not None and abs(ha - ra) < 0.05 and ha > 0.9:
        verdict = "index_supervised"
        print("\n  RAW ~= HUNGARIAN ~= 1 on the UNSUPERVISED arm (rule 3, "
              "case 3). That should be impossible without a supervision "
              "signal: an unsupervised router has no reason to prefer the "
              "oracle's numbering.\n     Treat this as a LEAK to be found, not "
              "a result -- check whether the expert index reaches the input "
              "features (CLAUDE.md 3) before reading anything else here.")
        problems.append("unsupervised arm reproduces the oracle numbering; "
                        "possible label leak")
    else:
        verdict = "weak_partition"
        print(f"\n  WEAK / INTERMEDIATE (rule 3, none of the three cases "
              f"cleanly): hungarian {ha:.4f} vs chance {CHANCE:.3f}, AMI "
              f"{fmt(am, '.4f')}, raw {fmt(ra, '.4f')}. Above chance but not a "
              "clean partition. Report the numbers and the ambiguity; do not "
              "round it up into a specialization claim.")

    print("\nrule 5 -- collapse check (a collapsed router has AMI ~ 0 for a "
          "DIFFERENT reason than an arbitrary one)")
    print(f"  unsupervised collapsed_experts: {un['collapsed']}")
    print(f"  unsupervised load entropy H/log(E) = "
          f"{fmt(un['load_entropy']['mean'], '.4f')} +- "
          f"{fmt(un['load_entropy']['std'], '.4f')}  "
          "(1.0 = perfectly balanced, 0.0 = one expert takes everything)")
    print(f"  supervised   collapsed_experts: {sup['collapsed']}")

    print("\nrule 4 -- the supervised arm's role: it bounds what the "
          "architecture can do WHEN TOLD the answer. It is the labelled "
          "ablation that demonstrates the gap, not a candidate for canonical.")

    out = {"seeds": list(SEEDS), "epochs": EPOCHS, "blocks": BLOCKS,
           "batch_size": BATCH_SIZE, "chance": CHANCE,
           "supervised_weight": SUPERVISED_WEIGHT,
           "canonical_step_routing": 0.0,
           "canonical_authority": "plan.md E -- supervision OFF is canonical",
           "summary": summary, "verdict": verdict,
           "note": "PROXY RUNS -- experiment_group=exploratory, no headline use"}
    json.dump(out, open(os.path.join(REPO, "automated",
                                     "routing_supervision_result.json"),
                        "w", encoding="utf-8"), indent=2)
    print("\nwritten: automated/routing_supervision_result.json")
    return 1 if problems else 0


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
                print(f"[{i}/{total}] {variant} seed {seed} ...", flush=True)
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
