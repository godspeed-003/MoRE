"""summarize_opdiff_matrix.py - T-OD.6. Aggregate the opdiff matrix.

    <interpreter> code/summarize_opdiff_matrix.py
    <interpreter> code/summarize_opdiff_matrix.py --out results/opdiff/summary.json

The ONE aggregation command for the opdiff task. It is deliberately NOT
`code/export_results.py`: that exporter is spec-driven, and the opdiff spec is
UNFROZEN (every enforced field is null), so there is nothing for it to admit
rows against. This script therefore carries the same discipline in its own
right -- it refuses to mix dataset versions, it refuses to mix config hashes
within an arm, it reports N/A rather than a sentinel, and it never writes a
number it did not read out of a run's own `metrics.json`.

WHAT IT REFUSES, and why each refusal is a correctness rule and not caution:

  * A run whose `data.dataset_version` differs from the declared dataset's is
    NOT admitted. The opdiff dataset is not tracked in git, so it is
    regenerated on each machine, and a regeneration that sampled different
    rows is a DIFFERENT dataset. Mixing such a run into a mean with the
    others would produce a number that describes no dataset at all.
  * Two runs of the SAME arm with different `config_hash` are refused as a
    set. A config that changed between two seeds of one arm means the arm is
    no longer one arm, and the mean across it is a mean of two experiments.
  * A run with fewer than one completed seed is reported as N/A, not as a
    mean over what happened to be there. One seed is a single observation and
    the study reports mean +- std over five.

THE PRIMARY METRIC IS `val/task_loss` -- the masked cross-entropy over the
generator's scored positions (the answer digits and the EOS), in nats per
scored token. `total_loss` is never reported as a quality number (CLAUDE.md 4):
it contains the balance, ponder and probe terms, and a negative weighted total
is not evidence of anything.
"""

import argparse
import glob
import json
import math
import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from more.config import load_config, TASK_OPDIFF
from more.opdiff_data import opdiff_versions, resolve_opdiff_root

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

CODE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(CODE)
RUNS = os.path.join(REPO, "runs")
OPDIFF_CONFIG = os.path.join(CODE, "config_opdiff.json")
ARMS = ["moe", "mor", "more"]

# The run NAME uses the DISPLAY spelling (MoE, MoR, MoRE -- config.py's
# ARCH_DISPLAY), not the config enum (moe, mor, more). Matching the wrong one
# is how a summariser silently reports zero admissible runs.
RUN_DIR_RE = re.compile(r"^opdiffB_(?P<arch>moe|mor|more)_seed(?P<seed>\d+)(__.*)?$",
                        re.IGNORECASE)


# ---------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------

def expected_version():
    """The dataset version the CONFIGURED dataset has, or None."""
    root = load_config(OPDIFF_CONFIG)["data"]["opdiff_root"]
    dsv, _ = opdiff_versions(root)
    return dsv


def collect(expect_version=None):
    """The admissible runs, as a list of dicts. Never a computed number."""
    out, skipped = [], []
    for d in sorted(glob.glob(os.path.join(RUNS, "opdiffB_*"))):
        base = os.path.basename(d)
        m = RUN_DIR_RE.match(base)
        if not m:
            skipped.append((base, "not an opdiffB_<arch>_seed<n> directory"))
            continue
        mp = os.path.join(d, "metrics.json")
        rp = os.path.join(d, "resolved_config.json")
        if not (os.path.exists(mp) and os.path.exists(rp)):
            skipped.append((base, "no finished metrics.json (interrupted run)"))
            continue
        try:
            cfg = json.load(open(rp, "r", encoding="utf-8"))
            met = json.load(open(mp, "r", encoding="utf-8"))
        except (OSError, ValueError) as e:
            skipped.append((base, f"unreadable: {e}"))
            continue
        if (cfg.get("task") or "") != TASK_OPDIFF:
            skipped.append((base, f"task={cfg.get('task')!r}"))
            continue
        dsv = (cfg.get("data") or {}).get("dataset_version")
        if expect_version and dsv != expect_version:
            skipped.append((base, f"dataset_version {dsv!r} != {expect_version!r} "
                                  f"(a different generation, or the smoke dataset)"))
            continue
        out.append({
            "dir": base,
            "arch": str(cfg.get("architecture") or m.group("arch")).lower(),
            "seed": int((cfg.get("provenance") or {}).get("seed", -1)),
            "config_hash": (cfg.get("provenance") or {}).get("config_hash"),
            "dataset_version": dsv,
            "group": (cfg.get("logging") or {}).get("experiment_group"),
            "metrics": met,
        })
    return out, skipped


def _num(v):
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def _ms(vals):
    """(mean, std, n) over the MEASURED values only. N/A when there are none."""
    xs = [v for v in vals if v is not None]
    if not xs:
        return None, None, 0
    m = sum(xs) / len(xs)
    if len(xs) == 1:
        # One observation has no spread. Reporting 0.0 would claim a certainty
        # the data does not contain; reporting the sample std of one point
        # (0.0) is the same lie. None, and the table prints N/A.
        return m, None, 1
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return m, math.sqrt(var), len(xs)


def _fmt(mean, std, digits=4):
    if mean is None:
        return "N/A"
    if std is None:
        return f"{mean:.{digits}f} (n=1, no spread)"
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------

HEADLINE = [
    ("val/task_loss", "val/task_loss", 4),
    ("val/avg_recursion_steps", "avg recursion steps", 2),
    ("val/routing_agreement_with_group", "agreement with group", 4),
    ("val/routing_hungarian_accuracy", "Hungarian accuracy", 4),
    ("val/routing_ami", "AMI", 4),
    ("val/routing_purity", "purity", 4),
    ("train/expert_load_entropy_normalized", "load entropy H/log(E)", 4),
    ("probe/family_ce", "group probe CE", 4),
]


def main(argv=None):
    ap = argparse.ArgumentParser(description="Aggregate the opdiff matrix.")
    ap.add_argument("--out", default=None, help="write the summary as JSON here")
    ap.add_argument("--all-datasets", action="store_true",
                    help="admit every dataset version instead of only the "
                         "configured one (for a diagnostic, never for a table)")
    args = ap.parse_args(argv)

    expect = None if args.all_datasets else expected_version()
    print("\n=== opdiff matrix " + "=" * 55)
    print(f"  dataset_version = {expect or '(any)'}")
    runs, skipped = collect(expect)
    for base, why in skipped:
        print(f"  skipped  {base}  -- {why}")
    if not runs:
        print("\n  no admissible runs. Nothing to summarise.")
        return 1

    # REFUSAL 1: one dataset version across the whole set.
    versions = sorted({r["dataset_version"] for r in runs})
    if len(versions) > 1:
        print(f"\n  REFUSED: the runs span {len(versions)} dataset versions "
              f"({versions}). A mean across two datasets describes neither.")
        return 2
    # REFUSAL 2: one config hash per arm.
    by_arm = defaultdict(set)
    for r in runs:
        by_arm[r["arch"]].add(r["config_hash"])
    bad = {a: sorted(h) for a, h in by_arm.items() if len(h) > 1}
    if bad:
        print(f"\n  REFUSED: these arms have more than one config hash: {bad}. "
              "The config changed between two seeds of the same arm, so the arm "
              "is two experiments.")
        return 2

    print(f"\n  {len(runs)} admissible run(s)\n")
    print(f"  {'arm':<6} {'n':>2}  {'val/task_loss':>22}  {'depth':>16}  "
          f"{'Hungarian':>16}")
    summary = {"dataset_version": versions[0], "arms": {}}
    for arch in ARMS:
        rows = [r for r in runs if r["arch"] == arch]
        if not rows:
            print(f"  {arch:<6} {0:>2}  {'N/A':>22}")
            continue
        vl = _ms([_num(r["metrics"].get("val/task_loss")) for r in rows])
        dp = _ms([_num(r["metrics"].get("val/avg_recursion_steps")) for r in rows])
        hg = _ms([_num(r["metrics"].get("val/routing_hungarian_accuracy"))
                  for r in rows])
        print(f"  {arch:<6} {vl[2]:>2}  {_fmt(vl[0], vl[1]):>22}  "
              f"{_fmt(dp[0], dp[1], 2):>16}  {_fmt(hg[0], hg[1]):>16}")
        summary["arms"][arch] = {
            "n_seeds": vl[2],
            "seeds": sorted(r["seed"] for r in rows),
            "config_hash": sorted(by_arm[arch]),
            "runs": [r["dir"] for r in rows],
            "metrics": {},
        }
        for key, label, _d in HEADLINE:
            vals = [_num(r["metrics"].get(key)) for r in rows]
            m, s, n = _ms(vals)
            summary["arms"][arch]["metrics"][key] = {
                "mean": m, "std": s, "n": n,
                "per_seed": {str(r["seed"]): _num(r["metrics"].get(key))
                             for r in rows},
            }

    # THE FULL HEADLINE TABLE, one row per measured quantity.
    print("\n  every measured quantity (mean ± std over the seeds that have it)")
    print(f"  {'arm':<6} {'quantity':<28} {'value':>26} {'n':>3}")
    for arch in ARMS:
        if arch not in summary["arms"]:
            continue
        for key, label, _d in HEADLINE:
            e = summary["arms"][arch]["metrics"][key]
            print(f"  {arch:<6} {label:<28} {_fmt(e['mean'], e['std']):>26} "
                  f"{e['n']:>3}")

    # THE PER-OPERATION DEPTH TABLE -- the opdiff study's depth measurement.
    # It is a MEAN over the seeds of the per-operation average exit depth, and
    # it is the table the "does the model spend more computation on the
    # operations that need more" question is answered from.
    print("\n  average recursion depth by operation (mean over seeds)")
    op_names = sorted({k.rsplit("/", 1)[1]
                       for r in runs for k in r["metrics"]
                       if k.startswith("recursion/avg_depth_by_op/")})
    if op_names:
        hdr = f"  {'operation':<10}" + "".join(f"{a:>12}" for a in ARMS)
        print(hdr)
        for op in op_names:
            line = f"  {op:<10}"
            for arch in ARMS:
                rows = [r for r in runs if r["arch"] == arch]
                vals = [_num(r["metrics"].get(f"recursion/avg_depth_by_op/{op}"))
                        for r in rows]
                m, s, n = _ms(vals)
                line += (f"{m:>8.2f} ±{s:>3.2f}" if m is not None and s is not None
                         else (f"{m:>12.2f}" if m is not None else f"{'N/A':>12}"))
            print(line)

    print("\n  average recursion depth by expert group (mean over seeds)")
    fam_names = sorted({k.rsplit("/", 1)[1]
                        for r in runs for k in r["metrics"]
                        if k.startswith("recursion/avg_depth_by_family/")})
    if fam_names:
        print(f"  {'group':<18}" + "".join(f"{a:>12}" for a in ARMS))
        for fam in fam_names:
            line = f"  {fam:<18}"
            for arch in ARMS:
                rows = [r for r in runs if r["arch"] == arch]
                vals = [_num(r["metrics"].get(f"recursion/avg_depth_by_family/{fam}"))
                        for r in rows]
                m, s, n = _ms(vals)
                line += (f"{m:>8.2f} ±{s:>3.2f}" if m is not None and s is not None
                         else (f"{m:>12.2f}" if m is not None else f"{'N/A':>12}"))
            print(line)

    print("\n  'agreement with group' is agreement with a DESIGNED PRIOR, not")
    print("  accuracy. Read the Hungarian / AMI / purity columns first.")
    print("  Every opdiff run is exploratory: the protocol is not frozen yet, so")
    print("  no number here may head a table labelled canonical.")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2, sort_keys=True, default=str)
        print(f"\n  written: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
