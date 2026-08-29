"""
The family-supervision ablation (T10.H): how much of MoRE's expert partition is
`cls_loss` responsible for?

The question
-----------
T8.0b established that with `step_routing = 0.0` the router still recovers a
real partition of the operation space under relabelling (Hungarian accuracy
0.5147 +- 0.0047, AMI 0.4940, against chance 0.167).  That was reported as
specialization emerging "without direct routing supervision".

But `loss_weights.family_cls = 0.5` is still live in that arm, and it is NOT
self-supervision.  `data/script.py` stamps a `"family"` string ("E1".."E6") into
every JSONL record at generation time; `more/families.py:FAMILY_TO_IDX` is a
hand-written manifest; `more/data.py` reads `rec.get("family")` straight off
disk.  So `cls_loss` is a 6-way cross-entropy against an EXTERNAL annotation
that names exactly the partition the routing metrics then measure.  It does not
touch the router logits, but it shapes the shared trunk representation the
router reads.

This experiment measures how much of the partition survives when that term is
removed entirely -- i.e. what MoRE does with no oracle-derived label anywhere in
the objective.

The decision rule, fixed HERE, before any run
---------------------------------------------
  1. THIS DOES NOT SELECT THE CANONICAL VALUE.  Canonical `family_cls = 0.5` is
     frozen in `canonical_spec.json:enforced_fields`, and the freeze is recorded
     as INHERITED, not selected.  `family_cls = 0.0` is the labelled T10.H
     ablation (`variant = no_family_supervision`) and can never be tabled as
     canonical, whatever it scores.

  2. What is measured, and reported either way:
     (a) the task-loss cost of removing family supervision (the user's
         theoretical claim is that it is load-bearing for prediction, not only
         for interpretability);
     (b) how much of the Hungarian / AMI / purity partition it accounts for.

  3. The reading of (b), fixed in advance.  Chance is 1/6 = 0.167.
       * hungarian and AMI fall to chance without `cls_loss`
         -> the family label is responsible for essentially the WHOLE partition.
            The paper's specialization claim must then read "under whole-program
            family supervision", and truly-unsupervised expert discovery is
            OUTCOME C.  Report it.
       * hungarian and AMI stay well above chance without `cls_loss`
         -> the partition does not depend on any oracle-derived label.  This is
            a materially STRONGER claim than T8.0b's and must be measured before
            it is made, not assumed.
       * partial retention -> report the retained fraction and the ambiguity.
         Do not round it up.

  4. Collapse is a distinct failure from an arbitrary partition (T8.0b rule 5).
     `collapsed_experts` and normalized load entropy are reported for both arms
     so an AMI near zero from collapse is never read as an AMI near zero from
     arbitrary numbering.

Why BOTH arms are re-run rather than reusing the T8.0b runs
-----------------------------------------------------------
The `family_cls = 0.5` arm of this comparison appears to exist already: the three
`routedec_unsupervised_router_seed{42,43,44}` runs from T8.0b used exactly this
protocol.  Reusing them would have saved three 20-epoch runs.

They are not admissible, for two independent reasons, and the second is the
disqualifying one.

  1. They predate T8.3, when `family_cls` was a bare `0.5` literal inside
     engine.py's loss assembly.  Their `resolved_config.json` has no
     `loss_weights.family_cls` key and their provenance records
     `family_cls_weight: null`.  T8.3 moved that literal into the config at the
     same position inside the `lw["task"]` factor, which is numerically
     identical at 0.5 -- but "should be identical" is not evidence.

  2. They predate T6.8, which added the permutation-invariant per-family table.
     They carry `per_family` (identity mapping) but no `per_family_matched` and
     no `matched_macro_recall`.  Comparing them against a current ablation run
     would put the two arms on DIFFERENT METRIC LAYERS -- and the matched
     per-family table is precisely the evidence T6.3/T6.8 exist to supply for an
     unsupervised router.  A missing metric in one arm is not a matched
     comparison.

So both arms are trained on the current code.  The legacy runs are kept as a
REPRODUCTION CROSS-CHECK only: the driver reports the seed-42 val-task-loss delta
between the fresh baseline and the T8.0b run, which measures whether T8.3 +
T6.8 disturbed the objective.  A delta inside 1e-9 says they did not; a larger
delta is reported as a finding and does NOT invalidate this experiment, because
both arms here come from one code state either way.


What this is NOT
----------------
PROXY RUNS.  20 epochs, and `lr` / `d_model` / `weight_decay` /
`routing_balance` are still null in `canonical_spec.json`, so
`experiment_group` stays `exploratory` and the Gate 0 guard refuses a canonical
claim from any of them.  No number here may enter the headline table.  What
survives is the ablation reading.

Usage
-----
    python automated/family_cls_ablation.py                 # run + report
    python automated/family_cls_ablation.py --report-only
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

# --- the matched protocol, copied from routing_supervision_decision.py --------
# It must stay copied rather than imported: if that driver's constants are ever
# retuned, this comparison's baseline arm would silently stop matching the runs
# on disk that it reuses.
SEEDS = (42, 43, 44)
EPOCHS = 20
BLOCKS = 1
BATCH_SIZE = 768
CANONICAL_FAMILY_CLS = 0.5
CHANCE = 1.0 / 6.0

# The T8.0b arm, kept as a reproduction cross-check only -- NOT as an arm.
LEGACY_STEM = "routedec_unsupervised_router_seed"
BASELINE_NAME = "famcls_fam"
ABLATION_NAME = "famcls_nofam"
REPLICATION_TOL = 1e-9


# Fields that determine the objective and the optimization trajectory. The
# baseline arm is admissible only if every one of these agrees with the ablation
# arm. config_hash and code_git_commit are deliberately NOT in this list -- both
# changed at T8.3 for reasons the replication check exists to prove inert.
MATCH_FIELDS = ("resolved_epochs", "resolved_batch_size", "resolved_seed",
                "halt_target_mode", "routing_supervision_weight",
                "resolved_routing_mode", "resolved_router_noise",
                "resolved_subset_fraction", "dataset_version",
                "train_split_version", "ponder_weight", "total_params")
MATCH_MODEL = ("d_model", "num_experts", "max_depth", "num_blocks", "ffn_mult",
               "dropout")


def launch(run_name: str, seed: int, family_cls: float) -> tuple[bool, float, str]:
    """One training run. Returns (ok, wall_seconds, tail_of_stderr)."""
    cmd = [PY, "train.py",
           "--architecture", "more",
           "--epochs", str(EPOCHS),
           "--blocks", str(BLOCKS),
           "--batch_size", str(BATCH_SIZE),
           "--step_routing", "0.0",
           "--family_cls", str(family_cls),
           "--seed", str(seed),
           "--run_name", run_name]
    env = dict(os.environ, WANDB_MODE="disabled")
    t0 = time.time()
    p = subprocess.run(cmd, cwd=CODE, env=env, capture_output=True, text=True)
    return p.returncode == 0, time.time() - t0, (p.stderr or "")[-800:]


def newest_run_dir(stem: str) -> str | None:
    """Newest run dir for a stem, by mtime -- never alphabetically.

    A stale directory from an earlier attempt sorts unpredictably against the
    current one and would be read silently in its place.
    """
    cands = [d for d in glob.glob(os.path.join(RUNS, stem + "*"))
             if os.path.exists(os.path.join(d, "metrics.json"))]
    return max(cands, key=os.path.getmtime) if cands else None


def _num(v):
    """N/A and sentinels stay None; they are not measurements (CLAUDE.md 4)."""
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) \
        else None


def read_run(stem: str) -> dict | None:
    rd = newest_run_dir(stem)
    if rd is None:
        return None
    m = json.load(open(os.path.join(rd, "metrics.json"), encoding="utf-8"))
    rc = json.load(open(os.path.join(rd, "resolved_config.json"),
                        encoding="utf-8"))
    prov = rc.get("provenance", {})
    pim = m.get("routing_permutation_invariant", {}) or {}
    return {
        "dir": os.path.basename(rd),
        # A run that does not AGREE which arm it is makes "the ablation changed
        # nothing" a conclusion rather than a bug report.
        "fam_enabled": prov.get("family_supervision_enabled"),
        "fam_weight": _num(prov.get("family_cls_weight")),
        "lw_fam": _num((rc.get("loss_weights") or {}).get("family_cls")),
        "match": {k: prov.get(k) for k in MATCH_FIELDS},
        "match_model": {k: (rc.get("model") or {}).get(k) for k in MATCH_MODEL},
        "variant": prov.get("variant"),
        "experiment_group": prov.get("experiment_group"),
        "val_task_loss": _num(m.get("val/task_loss")),
        "train_task_loss": _num(m.get("train/task_loss")),
        "cls_loss": _num(m.get("train/classification_loss")),
        "raw_acc": _num(pim.get("raw_accuracy")),
        "hung_acc": _num(pim.get("hungarian_accuracy")),
        "ami": _num(pim.get("ami")),
        "purity": _num(pim.get("purity")),
        "macro_recall": _num(pim.get("macro_recall")),
        "matched_macro_recall": _num(pim.get("matched_macro_recall")),
        "assignment": pim.get("hungarian_assignment"),
        "collapsed": pim.get("collapsed_experts"),
        "load_entropy": _num(m.get("train/expert_load_entropy_normalized")),
        "cos_max": _num(m.get("diag/max_pairwise_cosine_sim")),
        "avg_depth": _num(m.get("train/avg_recursion_steps")),
        "abs_depth_err": _num(m.get("val/depth_allocation_error_abs")),
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


def check_replication(legacy42: dict | None, fresh42: dict | None) -> dict:
    """Cross-check the fresh baseline against the pre-T8.3/T6.8 T8.0b run.

    Diagnostic only. Both arms of THIS experiment come from one code state
    regardless of the outcome, so a mismatch is reported as a finding about
    T8.3/T6.8 and does not invalidate the ablation.
    """
    print("\n" + "=" * 84)
    print("REPRODUCTION CROSS-CHECK -- did T8.3 / T6.8 disturb the objective?")
    print("=" * 84)
    print("The T8.0b runs are NOT used as an arm here (they predate T6.8's "
          "matched per-family\ntable). This only asks whether the fresh "
          "family_cls=0.5 baseline reproduces them.")
    if legacy42 is None or fresh42 is None:
        print("  -> not available (one side missing).")
        return {"state": "unavailable", "delta": None}
    a, b = legacy42["val_task_loss"], fresh42["val_task_loss"]
    print(f"  T8.0b     {legacy42['dir']:52} val/task_loss = {fmt(a, '.12f')}")
    print(f"  fresh 0.5 {fresh42['dir']:52} val/task_loss = {fmt(b, '.12f')}")
    if a is None or b is None:
        print("  -> a val/task_loss is missing.")
        return {"state": "unavailable", "delta": None}
    d = abs(a - b)
    print(f"  |delta| = {d:.3e}   tolerance = {REPLICATION_TOL:.0e}")
    if d <= REPLICATION_TOL:
        print("  -> REPRODUCED. Moving family_cls out of the loss assembly and "
              "into the config\n     (T8.3) and adding the matched metric layer "
              "(T6.8) left the objective intact.")
        return {"state": "reproduced", "delta": d}
    print("  -> NOT REPRODUCED. Recorded as a finding: something between the "
          "T8.0b runs and\n     now changed the trajectory. T8.0b's numbers may "
          "not be quoted alongside\n     current runs. This experiment is "
          "unaffected -- both its arms are current.")
    return {"state": "not_reproduced", "delta": d}



def arm_table(title: str, rows: dict[int, dict], want_enabled: bool | None,
              want_weight: float | None, problems: list[str],
              strict_arm: bool) -> dict:
    print(f"\n-- {title} " + "-" * max(4, 80 - len(title)))
    print(f"{'seed':>5}  {'val_task':>10}  {'cls_loss':>9}  {'raw':>7}  "
          f"{'hung':>7}  {'ami':>7}  {'purity':>7}  {'mac_rec':>7}  "
          f"{'H_load':>7}  {'depth':>6}  {'collapsed':>10}  assignment")
    for seed in SEEDS:
        r = rows.get(seed)
        if r is None:
            print(f"{seed:>5}  {'MISSING':>10}")
            problems.append(f"{title} seed {seed}: no run directory")
            continue
        if strict_arm:
            if bool(r["fam_enabled"]) != bool(want_enabled):
                problems.append(
                    f"{title} seed {seed}: run declares "
                    f"family_supervision_enabled={r['fam_enabled']!r}")
            if r["fam_weight"] != want_weight:
                problems.append(
                    f"{title} seed {seed}: family_cls_weight="
                    f"{r['fam_weight']} (expected {want_weight})")
        if r["match"].get("resolved_seed") != seed:
            problems.append(f"{title} seed {seed}: resolved_seed="
                            f"{r['match'].get('resolved_seed')}")
        print(f"{seed:>5}  {fmt(r['val_task_loss']):>10}  "
              f"{fmt(r['cls_loss'], '.6f'):>9}  "
              f"{fmt(r['raw_acc'], '.4f'):>7}  {fmt(r['hung_acc'], '.4f'):>7}  "
              f"{fmt(r['ami'], '.4f'):>7}  {fmt(r['purity'], '.4f'):>7}  "
              f"{fmt(r['matched_macro_recall'], '.4f'):>7}  "
              f"{fmt(r['load_entropy'], '.4f'):>7}  "
              f"{fmt(r['avg_depth'], '.3f'):>6}  "
              f"{str(r['collapsed']):>10}  {r['assignment']}")
    present = [rows[s] for s in SEEDS if s in rows]
    agg = {}
    for key in ("val_task_loss", "train_task_loss", "raw_acc", "hung_acc",
                "ami", "purity", "macro_recall", "matched_macro_recall",
                "load_entropy", "avg_depth", "cos_max", "cls_loss"):
        mu, sd = mean_std([r[key] for r in present])
        agg[key] = {"mean": mu, "std": sd}
    print(f"  mean+-std  val_task_loss = {fmt(agg['val_task_loss']['mean'])} "
          f"+- {fmt(agg['val_task_loss']['std'])}")
    print(f"             hung = {fmt(agg['hung_acc']['mean'], '.4f')} "
          f"+- {fmt(agg['hung_acc']['std'], '.4f')}   "
          f"ami = {fmt(agg['ami']['mean'], '.4f')} "
          f"+- {fmt(agg['ami']['std'], '.4f')}   "
          f"purity = {fmt(agg['purity']['mean'], '.4f')} "
          f"+- {fmt(agg['purity']['std'], '.4f')}")
    return {"n": len(present),
            "collapsed": [r["collapsed"] for r in present],
            "assignments": [r["assignment"] for r in present],
            **agg}


def retained(with_v: float | None, without_v: float | None,
             floor: float) -> float | None:
    """Fraction of the above-chance partition that survives the ablation.

    Descriptive only. It is a ratio of two point estimates, not a variance
    decomposition, and it is undefined when the supervised arm is itself at the
    floor -- in which case there is no partition to attribute.
    """
    if with_v is None or without_v is None:
        return None
    denom = with_v - floor
    if denom <= 1e-6:
        return None
    return (without_v - floor) / denom


def read_matching(sup: dict[int, dict], abl: dict[int, dict],
                  problems: list[str]) -> None:
    """Every objective-relevant field must agree between the arms."""
    print("\n-- matched-protocol check " + "-" * 56)
    bad = 0
    compared = 0
    for seed in SEEDS:
        a, b = sup.get(seed), abl.get(seed)
        if a is None or b is None:
            continue
        for k in MATCH_FIELDS:
            compared += 1
            if a["match"].get(k) != b["match"].get(k):
                bad += 1
                problems.append(
                    f"seed {seed}: {k} differs between arms -- "
                    f"{a['match'].get(k)!r} vs {b['match'].get(k)!r}")
        for k in MATCH_MODEL:
            compared += 1
            if a["match_model"].get(k) != b["match_model"].get(k):
                bad += 1
                problems.append(
                    f"seed {seed}: model.{k} differs between arms -- "
                    f"{a['match_model'].get(k)!r} vs {b['match_model'].get(k)!r}")
    # Zero comparisons is an ABSENCE of evidence, never evidence -- the same
    # defect class as reporting a sentinel as a measurement (CLAUDE.md 4).
    if compared == 0:
        problems.append("matched-protocol check compared 0 fields: at least one "
                        "arm has no runs on disk")
        print("  0 fields compared -- NOT a pass. See problems below.")
        return
    print(f"  {compared} field comparisons across "
          f"{len(MATCH_FIELDS)} provenance + {len(MATCH_MODEL)} model fields "
          f"per seed: {'ALL MATCH' if bad == 0 else str(bad) + ' MISMATCH'}")
    if bad == 0:
        print("  -> loss_weights.family_cls is the only variable that differs "
              "(CLAUDE.md 6).")



def report(sup: dict[int, dict], abl: dict[int, dict],
           legacy42: dict | None) -> int:
    problems: list[str] = []
    print("\n" + "=" * 84)
    print("FAMILY-SUPERVISION ABLATION (T10.H)")
    print(f"family_cls = {CANONICAL_FAMILY_CLS} (CANONICAL, frozen in "
          "canonical_spec.json)  vs")
    print("family_cls = 0.0 (LABELLED ABLATION, variant=no_family_supervision)")
    print("=" * 84)
    print(f"matched protocol: architecture=more  halting=pure_act  "
          f"step_routing=0.0  epochs={EPOCHS}  blocks={BLOCKS}  "
          f"batch_size={BATCH_SIZE}  seeds={list(SEEDS)}")
    print("only difference : loss_weights.family_cls")
    print("both arms trained on ONE code state -- see the module docstring for "
          "why the T8.0b\nruns are not reused as the baseline arm.")
    print("PROXY RUNS -- 20 epochs, hyperparameters not yet frozen. No number "
          "below may enter the headline table.")
    print(f"chance level for 6 experts = {CHANCE:.3f}")

    s = arm_table(f"family_cls = {CANONICAL_FAMILY_CLS} (canonical)", sup,
                  True, CANONICAL_FAMILY_CLS, problems, strict_arm=True)
    a = arm_table("family_cls = 0.0 (ablation)", abl,
                  False, 0.0, problems, strict_arm=True)
    read_matching(sup, abl, problems)
    xcheck = check_replication(legacy42, sup.get(42))

    print("\n" + "=" * 84)
    print("READING (rules 1-4, fixed before the runs)")
    print("=" * 84)
    if problems:
        print("The comparison is NOT valid. Fix these first (CLAUDE.md 7: "
              "stop, report, identify, fix, re-run):")
        for p in problems:
            print("  * " + p)
        return 1


    print("rule 1 -- the canonical value is NOT selected here. "
          f"family_cls = {CANONICAL_FAMILY_CLS} stays canonical;")
    print("  family_cls = 0.0 is the labelled T10.H ablation, never in the same "
          "table.")

    sl, al = s["val_task_loss"]["mean"], a["val_task_loss"]["mean"]
    pooled = max(x for x in (s["val_task_loss"]["std"] or 0.0,
                             a["val_task_loss"]["std"] or 0.0))
    print("\nrule 2a -- task-loss cost of removing family supervision")
    print(f"  family_cls = {CANONICAL_FAMILY_CLS}  {fmt(sl)} +- "
          f"{fmt(s['val_task_loss']['std'])}")
    print(f"  family_cls = 0.0  {fmt(al)} +- "
          f"{fmt(a['val_task_loss']['std'])}")
    if sl is not None and al is not None:
        d = al - sl
        print(f"  cost = {d:+.6f}   largest seed std = {pooled:.6f}   "
              + ("INSIDE seed noise: family supervision does no measurable work "
                 "on task loss." if abs(d) <= pooled else
                 ("OUTSIDE seed noise: family supervision measurably HELPS task "
                  "loss." if d > 0 else
                  "OUTSIDE seed noise: family supervision measurably HURTS task "
                  "loss.")))

    print("\nrule 2b/3 -- how much of the partition is family supervision "
          "responsible for?")
    rh = retained(s["hung_acc"]["mean"], a["hung_acc"]["mean"], CHANCE)
    ra_ = retained(s["ami"]["mean"], a["ami"]["mean"], 0.0)
    rp = retained(s["purity"]["mean"], a["purity"]["mean"], CHANCE)
    print(f"  {'metric':<22} {'with 0.5':>10} {'with 0.0':>10} "
          f"{'floor':>7} {'retained':>9}")
    for nm, key, fl, rt in (("hungarian accuracy", "hung_acc", CHANCE, rh),
                            ("AMI", "ami", 0.0, ra_),
                            ("purity", "purity", CHANCE, rp)):
        print(f"  {nm:<22} {fmt(s[key]['mean'], '.4f'):>10} "
              f"{fmt(a[key]['mean'], '.4f'):>10} {fl:>7.3f} "
              + (f"{rt*100:>8.1f}%" if rt is not None else f"{'N/A':>9}"))
    print("  'retained' = (ablation - floor) / (canonical - floor). Descriptive "
          "ratio of two\n  point estimates, NOT a variance decomposition.")

    hb, ab_ = a["hung_acc"]["mean"], a["ami"]["mean"]
    if hb is None or ab_ is None:
        verdict = "undecidable"
        print("\n  Routing metrics are N/A on the ablation arm; rule 3 cannot be "
              "read.")
    elif hb < CHANCE + 0.10 and ab_ < 0.05:
        verdict = "family_label_responsible_for_whole_partition"
        print(f"\n  RULE 3, CASE 1 -- the family label is responsible for "
              f"essentially the WHOLE\n  partition: without it hungarian "
              f"{hb:.4f} is at chance ({CHANCE:.3f}) and AMI "
              f"{ab_:.4f} ~ 0.\n  Consequence: the specialization claim MUST "
              "read 'under whole-program family\n  supervision'. "
              "Truly-unsupervised expert discovery is OUTCOME C on this task at "
              "this\n  scale. Report it as such -- do not restate T8.0b's "
              "result as unsupervised.")
    elif rh is not None and rh >= 0.60:
        verdict = "partition_survives_without_any_oracle_label"
        print(f"\n  RULE 3, CASE 2 -- the partition SURVIVES with no "
              f"oracle-derived label anywhere in\n  the objective: hungarian "
              f"{hb:.4f} vs chance {CHANCE:.3f}, AMI {ab_:.4f}, "
              f"{rh*100:.0f}% of the\n  above-chance partition retained. This "
              "is a materially stronger claim than\n  T8.0b's and is now "
              "measured rather than assumed.")
    else:
        verdict = "partial_retention"
        print(f"\n  RULE 3, CASE 3 -- PARTIAL RETENTION: hungarian {hb:.4f} vs "
              f"chance {CHANCE:.3f}, AMI\n  {ab_:.4f}, retained "
              + (f"{rh*100:.0f}%" if rh is not None else "N/A")
              + " of the above-chance partition. Above chance but\n  materially "
              "degraded. Report both numbers and the ambiguity; do not round it "
              "up\n  into an unsupervised-specialization claim.")

    print("\nrule 4 -- collapse is a different failure from arbitrary numbering")
    print(f"  canonical collapsed_experts: {s['collapsed']}   "
          f"H_load = {fmt(s['load_entropy']['mean'], '.4f')}")
    print(f"  ablation  collapsed_experts: {a['collapsed']}   "
          f"H_load = {fmt(a['load_entropy']['mean'], '.4f')}")
    print("  (H_load 1.0 = perfectly balanced, 0.0 = one expert takes "
          "everything)")
    print(f"\n  Hungarian assignments, canonical: {s['assignments']}")
    print(f"  Hungarian assignments, ablation : {a['assignments']}")
    print("  Assignments differing across seeds at stable accuracy is a real "
          "partition under\n  relabelling; identical assignments at ~1.0 "
          "accuracy would mean the index itself is\n  supervised.")

    out = {"seeds": list(SEEDS), "epochs": EPOCHS, "blocks": BLOCKS,
           "batch_size": BATCH_SIZE, "chance": CHANCE,
           "canonical_family_cls": CANONICAL_FAMILY_CLS,
           "canonical_authority":
               "canonical_spec.json:enforced_fields -- INHERITED, not selected",
           "reproduction_cross_check": xcheck,
           "retained_fraction": {"hungarian": rh, "ami": ra_, "purity": rp},
           "arms": {"family_cls_0.5": s, "family_cls_0.0": a},
           "verdict": verdict,
           "note": "PROXY RUNS -- experiment_group=exploratory, no headline use"}
    dest = os.path.join(REPO, "automated", "family_cls_ablation_result.json")
    json.dump(out, open(dest, "w", encoding="utf-8"), indent=2)
    print(f"\nwritten: {os.path.relpath(dest, REPO)}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report-only", action="store_true",
                    help="Re-read existing run directories without training")
    args = ap.parse_args()

    if not args.report_only:
        # Interleaved by seed rather than arm-by-arm: if the sweep is cut short
        # (quota, reboot), a partial result still has both arms at the seeds it
        # reached instead of one complete arm and one empty one.
        jobs = [(nm, s, fc) for s in SEEDS
                for nm, fc in ((BASELINE_NAME, CANONICAL_FAMILY_CLS),
                               (ABLATION_NAME, 0.0))]
        for i, (nm, seed, fc) in enumerate(jobs, 1):
            print(f"[{i}/{len(jobs)}] {nm} seed {seed} family_cls={fc} ...",
                  flush=True)
            ok, secs, err = launch(nm, seed, fc)
            print(f"        {'ok' if ok else 'FAILED'} in {secs/60:.1f} min",
                  flush=True)
            if not ok:
                print("        stderr tail: " + err, flush=True)

    sup = {s: r for s in SEEDS
           if (r := read_run(f"{BASELINE_NAME}_seed{s}")) is not None}
    abl = {s: r for s in SEEDS
           if (r := read_run(f"{ABLATION_NAME}_seed{s}")) is not None}
    legacy42 = read_run(f"{LEGACY_STEM}42")
    return report(sup, abl, legacy42)


if __name__ == "__main__":
    sys.exit(main())




