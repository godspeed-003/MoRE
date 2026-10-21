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
# The SMOKE runs are named `opdiffB_smoke_<Arch>_seed<S>` -- deliberately
# distinguishable from a matrix cell, so a smoke run can never be admitted to
# a matrix table by accident. They are aggregated with `--kind smoke`, which is
# how the pilot's own numbers are read before the matrix exists.
SMOKE_DIR_RE = re.compile(
    r"^opdiffB_smoke_(?P<arch>moe|mor|more)(?:_(?P<variant>ffn\d+))?_seed(?P<seed>\d+)(__.*)?$",
    re.IGNORECASE)
# The CAPACITY ABLATIONS: `opdiffB_<Arch>_ffn<N>_seed<S>`. The width is IN THE
# NAME, so an under-budgeted arm can never be admitted to the canonical table
# -- and the resolved config's `variant` string says the same thing
# ("opdiff+ffn_mult_4"). Two independent labels, because a table that mixes a
# 0.8 M-parameter arm with a 3.9 M one and calls them "MoR" is the exact
# confusion the parameter-budget rule exists to prevent.
ABLATION_DIR_RE = re.compile(
    r"^opdiffB_(?P<arch>moe|mor|more)_(?P<variant>ffn\d+)_seed(?P<seed>\d+)(__.*)?$",
    re.IGNORECASE)


# ---------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------

def expected_version():
    """The dataset version the CONFIGURED dataset has, or None."""
    root = load_config(OPDIFF_CONFIG)["data"]["opdiff_root"]
    dsv, _ = opdiff_versions(root)
    return dsv


def collect(expect_version=None, kind="matrix"):
    """The admissible runs, as a list of dicts. Never a computed number."""
    out, skipped = [], []
    pattern = {"matrix": RUN_DIR_RE, "smoke": SMOKE_DIR_RE,
               "ablation": ABLATION_DIR_RE}.get(kind, RUN_DIR_RE)
    for d in sorted(glob.glob(os.path.join(RUNS, "opdiffB_*"))):
        base = os.path.basename(d)
        m = pattern.match(base)
        if not m:
            skipped.append((base, f"not an opdiffB_<arch>_seed<n> {kind} directory"))
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
            "variant": m.groupdict().get("variant"),
            "variant_tag": (cfg.get("provenance") or {}).get("variant"),
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
    ("val/accuracy", "val accuracy (per digit)", 4),
    ("val/exact_match", "val exact match (per example)", 4),
    ("val/avg_recursion_steps", "avg recursion steps", 2),
    ("val/routing_agreement_with_group", "agreement with group", 4),
    ("val/routing_hungarian_accuracy", "Hungarian accuracy", 4),
    ("val/routing_ami", "AMI", 4),
    ("val/routing_purity", "purity", 4),
    ("train/expert_load_entropy_normalized", "load entropy H/log(E)", 4),
    ("probe/family_ce", "group probe CE", 4),
    ("ood_interp/task_loss", "ood_interp task_loss", 4),
    ("ood_interp/accuracy", "ood_interp accuracy", 4),
    ("ood_interp/exact_match", "ood_interp exact match", 4),
    ("ood_interp/avg_recursion_steps", "ood_interp avg depth", 2),
    ("ood_extrap/task_loss", "ood_extrap task_loss", 4),
    ("ood_extrap/accuracy", "ood_extrap accuracy", 4),
    ("ood_extrap/exact_match", "ood_extrap exact match", 4),
    ("ood_extrap/avg_recursion_steps", "ood_extrap avg depth", 2),
]


def main(argv=None):
    ap = argparse.ArgumentParser(description="Aggregate the opdiff matrix.")
    ap.add_argument("--out", default=None, help="write the summary as JSON here")
    ap.add_argument("--all-datasets", action="store_true",
                    help="admit every dataset version instead of only the "
                         "configured one (for a diagnostic, never for a table)")
    ap.add_argument("--kind", choices=("matrix", "smoke", "ablation"),
                    default="matrix",
                    help="'matrix' (the default) reads the matrix cells; "
                         "'smoke' reads the opdiffB_smoke_* runs -- the pilot's "
                         "own numbers, before the matrix exists; 'ablation' "
                         "reads the capacity ablations (opdiffB_<Arch>_ffnN_*)")
    args = ap.parse_args(argv)

    expect = None if args.all_datasets else expected_version()
    print("\n=== opdiff matrix " + "=" * 55)
    print(f"  dataset_version = {expect or '(any)'}   kind = {args.kind}")
    runs, skipped = collect(expect, args.kind)
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
    # REFUSAL 2: one config hash per (arm, variant). The VARIANT is part of
    # the key because a capacity ablation is a DIFFERENT experiment from the
    # canonical arm -- and it is a LABELLED one ("opdiff+ffn_mult_4"). Two
    # runs of the SAME (arm, variant) with different config hashes is the real
    # defect: the config changed between two seeds of one experiment.
    by_arm = defaultdict(set)
    for r in runs:
        by_arm[(r["arch"], r["variant"])].add(r["config_hash"])
    bad = {a: sorted(h) for a, h in by_arm.items() if len(h) > 1}
    if bad:
        print(f"\n  REFUSED: these (arm, variant) pairs have more than one "
              f"config hash: {bad}. The config changed between two seeds of the "
              "same experiment, so it is two experiments.")
        return 2

    print(f"\n  {len(runs)} admissible run(s)\n")
    print(f"  {'arm':<16} {'n':>2}  {'val/task_loss':>22}  {'depth':>16}  "
          f"{'Hungarian':>16}")
    summary = {"dataset_version": versions[0], "arms": {}}
    # The headline table is per (arm, variant): a capacity ablation is a
    # DIFFERENT experiment from the canonical arm, and it is a LABELLED one, so
    # it gets its own row and can never be averaged into the canonical arm's.
    _arm_groups = {}
    for r in runs:
        _arm_groups.setdefault((r["arch"], r["variant"]), []).append(r)
    for (arch, variant), rows in sorted(
            _arm_groups.items(), key=lambda kv: (kv[0][0], kv[0][1] or "")):
        _label = arch if not variant else f"{arch}/{variant}"
        vl = _ms([_num(r["metrics"].get("val/task_loss")) for r in rows])
        dp = _ms([_num(r["metrics"].get("val/avg_recursion_steps")) for r in rows])
        hg = _ms([_num(r["metrics"].get("val/routing_hungarian_accuracy"))
                  for r in rows])
        print(f"  {_label:<16} {vl[2]:>2}  {_fmt(vl[0], vl[1]):>22}  "
              f"{_fmt(dp[0], dp[1], 2):>16}  {_fmt(hg[0], hg[1]):>16}")
        summary["arms"][_label] = {
            "n_seeds": vl[2],
            "seeds": sorted(r["seed"] for r in rows),
            "config_hash": sorted(by_arm[(arch, variant)]),
            "runs": [r["dir"] for r in rows],
            "metrics": {},
        }
        for key, label, _d in HEADLINE:
            vals = [_num(r["metrics"].get(key)) for r in rows]
            m, s, n = _ms(vals)
            summary["arms"][_label]["metrics"][key] = {
                "mean": m, "std": s, "n": n,
                "per_seed": {str(r["seed"]): _num(r["metrics"].get(key))
                             for r in rows},
            }

    # THE FULL HEADLINE TABLE, one row per measured quantity.
    print("\n  every measured quantity (mean ± std over the seeds that have it)")
    print(f"  {'arm':<16} {'quantity':<28} {'value':>26} {'n':>3}")
    for _label in sorted(summary["arms"]):
        for key, label, _d in HEADLINE:
            e = summary["arms"][_label]["metrics"][key]
            print(f"  {_label:<16} {label:<28} {_fmt(e['mean'], e['std']):>26} "
                  f"{e['n']:>3}")

    # THE PER-OPERATION DEPTH TABLE -- the opdiff study's depth measurement.
    # It is a MEAN over the seeds of the per-operation average exit depth, and
    # it is the table the "does the model spend more computation on the
    # operations that need more" question is answered from.
    #
    # The columns are (arm, variant) groups, not arms: an under-budgeted MoR
    # averaged into the canonical MoR's column would be a table of two
    # experiments wearing one arm's name.
    _cols = sorted(summary["arms"])

    def _col_rows(label):
        for (_a, _v), _rows in _arm_groups.items():
            if (_a if not _v else f"{_a}/{_v}") == label:
                return _rows
        return []

    # THE PER-OPERATION DEPTH TABLE -- the opdiff study's depth measurement.
    # It is a MEAN over the seeds of the per-operation average exit depth, and
    # it is the table the "does the model spend more computation on the
    # operations that need more" question is answered from.
    print("\n  average recursion depth by operation (mean over seeds)")
    op_names = sorted({k.rsplit("/", 1)[1]
                       for r in runs for k in r["metrics"]
                       if k.startswith("recursion/avg_depth_by_op/")})
    if op_names:
        print(f"  {'operation':<10}" + "".join(f"{c:>16}" for c in _cols))
        for op in op_names:
            line = f"  {op:<10}"
            for c in _cols:
                vals = [_num(r["metrics"].get(f"recursion/avg_depth_by_op/{op}"))
                        for r in _col_rows(c)]
                m, s, n = _ms(vals)
                line += (f"{m:>10.2f} ±{s:>3.2f}" if m is not None and s is not None
                         else (f"{m:>16.2f}" if m is not None else f"{'N/A':>16}"))
            print(line)

    print("\n  average recursion depth by expert group (mean over seeds)")
    fam_names = sorted({k.rsplit("/", 1)[1]
                        for r in runs for k in r["metrics"]
                        if k.startswith("recursion/avg_depth_by_family/")})
    if fam_names:
        print(f"  {'group':<18}" + "".join(f"{c:>16}" for c in _cols))
        for fam in fam_names:
            line = f"  {fam:<18}"
            for c in _cols:
                vals = [_num(r["metrics"].get(f"recursion/avg_depth_by_family/{fam}"))
                        for r in _col_rows(c)]
                m, s, n = _ms(vals)
                line += (f"{m:>10.2f} ±{s:>3.2f}" if m is not None and s is not None
                         else (f"{m:>16.2f}" if m is not None else f"{'N/A':>16}"))
            print(line)

    # THE CAPACITY LADDER. The question it answers: does the MoR-vs-MoRE gap
    # depend on the MoR arm's PARAMETER BUDGET, or on the architecture? The
    # canonical MoR is budget-matched to MoRE (7 x 4 = 28 hidden width); the
    # other widths are LABELLED ablations and are NOT matched to anything, and
    # the non-embedding parameter count is printed next to every row precisely
    # so that no reader can mistake one for another.
    _groups = {}
    for r in runs:
        _groups.setdefault((r["arch"], r["variant"]), []).append(r)
    if len(_groups) > 1 or any(v for _a, v in _groups):
        print("\n  the capacity ladder (an UNMATCHED width is a labelled "
              "ablation, never a MoR cell)")
        print(f"  {'arm':<6} {'width':<10} {'variant':<22} {'non-emb params':>15} "
              f"{'val/task_loss':>14} {'exact':>7} {'depth':>7} {'n':>3}")
        for (_arch, _var), _rows in sorted(
                _groups.items(), key=lambda kv: (kv[0][0], kv[0][1] or "")):
            _ne = _rows[0]["metrics"].get("non_embedding_params")
            _vl = _ms([_num(x["metrics"].get("val/task_loss")) for x in _rows])
            _ex = _ms([_num(x["metrics"].get("val/exact_match")) for x in _rows])
            _dp = _ms([_num(x["metrics"].get("val/avg_recursion_steps")) for x in _rows])
            print(f"  {_arch:<6} {(_var or 'canonical'):<10} "
                  f"{str(_rows[0].get('variant_tag') or 'canonical'):<22} "
                  f"{(_ne if _ne else 0):>15,} {_fmt(_vl[0], _vl[1]):>14} "
                  f"{_fmt(_ex[0], _ex[1], 3):>7} {_fmt(_dp[0], _dp[1], 2):>7} "
                  f"{_vl[2]:>3}")
        print("  The canonical MoR row is the budget-matched one (7 x 4 = 28);")
        print("  every other width is an explicitly labelled ablation and is")
        print("  NOT comparable with MoRE on capacity -- which is the point of")
        print("  running it.")

    # THE COMPUTE-VERSUS-k TABLE. This is the one the study's OOD claim is
    # read off, and it is the diagnostic the dataset's own caveat asks for: the
    # SEEN k values and the HELD-OUT ones in the same table, so a reader can
    # see whether the model fails on the held-out k values because it cannot
    # READ the k slot or because it cannot SOLVE an F row at all.
    _k_names = sorted({k.rsplit("/", 1)[1]
                        for r in runs for k in r["metrics"]
                        if k.startswith("val/accuracy_by_op/F")},
                       key=lambda s: int(s[1:]))
    if _k_names:
        print("\n  the compute-versus-k table (F rows only, mean over seeds)")
        print("  'seen' = the k was trained on; 'HELD OUT' = it was not.")
        print(f"  {'k':>4} {'status':>10}  {'acc':>8} {'exact':>8} {'depth':>8}   "
              f"{'ood acc':>8} {'ood exact':>10} {'ood depth':>10}")

        def _axis_mean(arch, prefix, name):
            rows = [r for r in runs if r["arch"] == arch]
            m, s, n = _ms([_num(r["metrics"].get(f"{prefix}/{name}"))
                           for r in rows])
            return m

        for k in _k_names:
            for arch in ARMS:
                rows = [r for r in runs if r["arch"] == arch]
                if not rows:
                    continue
                _acc = _axis_mean(arch, "val/accuracy_by_op", k)
                _ex = _axis_mean(arch, "val/exact_match_by_op", k)
                _dp = _axis_mean(arch, "recursion/avg_depth_by_op", k)
                _oacc = _axis_mean(arch, "ood_interp/accuracy_by_op", k)
                _oex = _axis_mean(arch, "ood_interp/exact_match_by_op", k)
                _odp = _axis_mean(arch, "ood_interp/avg_depth_by_op", k)
                if _oacc is None:
                    _oacc = _axis_mean(arch, "ood_extrap/accuracy_by_op", k)
                    _oex = _axis_mean(arch, "ood_extrap/exact_match_by_op", k)
                    _odp = _axis_mean(arch, "ood_extrap/avg_depth_by_op", k)
                _held = _oacc is not None
                def _f(v, d=3):
                    return f"{v:.{d}f}" if v is not None else "N/A"
                print(f"  {k:>4} {'HELD OUT' if _held else 'seen':>10}  "
                      f"{_f(_acc):>8} {_f(_ex):>8} {_f(_dp, 2):>8}   "
                      f"{_f(_oacc):>8} {_f(_oex):>10} {_f(_odp, 2):>10}   "
                      f"[{arch}]")
        print("\n  A HELD-OUT row whose 'acc' is the same as the seen rows' says")
        print("  the model has not learned to READ the k slot at all -- that is")
        print("  a task problem, not an architecture problem, and the fix is more")
        print("  trained k values (see OPDIFF_DATASET.md).")

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
