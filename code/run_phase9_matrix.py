"""
T9.1 -- THE CANONICAL PHASE-B MATRIX.  MoE / MoR / MoRE x seeds 42-46.

This is the only driver in the repository allowed to launch canonical runs.
`automated/` is barred from doing so (CLAUDE.md 9), which is why this lives in
`code/` beside train.py.

FOUR RULES, FIXED BEFORE THE FIRST RUN.

1. NO CLI OVERRIDES BEYOND --architecture, --seed, --run_name, --experiment_group.
   Every other field must come from config.json so the Gate 0 guard can compare
   it against canonical_spec.json. The moment this driver passes --epochs or
   --batch_size it is asserting the value rather than inheriting it, and a typo
   becomes a silent proxy wearing a canonical label. If config.json disagrees
   with the spec, the correct fix is to change config.json in the same commit as
   the spec -- never to paper over it here.
   --experiment_group is the exception that PROVES the rule: it is not a config
   value at all, it is the canonical claim the guard adjudicates. Omitting it does
   not raise -- the run is stamped `exploratory` and quietly excluded from the
   headline. See launch().

2. A RUN THAT IS NOT canonical_phase_b IS NOT A ROW.
   After each run this driver reads provenance and requires
   experiment_group == canonical_phase_b, variant == canonical and
   seed_declared == True. Anything else is recorded as a problem and excluded
   from the aggregate. The guard refusing a run is the guard working.

3. ARCHITECTURE-INAPPLICABLE METRICS AGGREGATE TO "N/A", NEVER 0.0.
   MoR has one expert, so it has no routing accuracy, no AMI, no load entropy,
   no pairwise cosine. MoE has max_depth 1, so its depth metrics are constants.
   `mean_std()` returns None for a metric whose per-seed values are the string
   "N/A", and the report prints N/A. A 0.0 in place of an N/A is a fabricated
   data point (CLAUDE.md 4, results_exp.md 0).

4. EVERY NUMBER IS mean +- std OVER FIVE SEEDS, AND EVERY VERDICT COMES FROM AN
   EXACT TWO-SIDED RANDOMIZATION TEST.
   The three architectures are known to sit in a narrow band (~0.0635 against a
   predict-the-mean floor of 0.080914, R^2 ~ 0.215 at 20 epochs), so the reading
   rule decides what this driver is able to see. It used to be "a difference
   smaller than the larger arm's seed std is not a difference", which is wrong:
   a difference of means is judged against the standard error OF THE DIFFERENCE,
   ~sqrt(5) smaller at five seeds per arm. That rule silently suppressed two of
   the three pairwise findings. Replaced at T11.0b by the exhaustive permutation
   test in code/seed_stats.py, which needs no variance estimate at all; the
   report prints the p-value beside the smallest p the arm sizes can produce, so
   a result sitting at the resolution floor is visible as such. It still does not
   rank the architectures.

Resumable: a seed/architecture pair whose run directory already holds a complete
metrics.json is skipped, so a 4-hour matrix survives an interrupted session.

Writes code/phase9_matrix_result.json.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import seed_stats as _seed_stats                                   # noqa: E402

CODE = Path(__file__).resolve().parent
ROOT = CODE.parent
RUNS = ROOT / "runs"
PY = sys.executable

SEEDS = (42, 43, 44, 45, 46)
ARCHES = ("moe", "mor", "more")
CANONICAL_GROUP = "canonical_phase_b"
FLOOR = 0.080914          # predict-the-train-mean on the real val split
CHANCE_6 = 1.0 / 6.0
RESULT = CODE / "phase9_matrix_result.json"

# Metrics pulled per run. `None` in the aggregate means the architecture does not
# have the quantity -- printed as N/A, never as a number.
SCALARS = (
    "val/task_loss",
    "best_val_loss",
    "train/task_loss",
    "train/classification_loss",
    "val/routing_accuracy",
    "val/routing_hungarian_accuracy",
    "val/routing_ami",
    "val/routing_purity",
    "val/routing_matched_macro_recall",
    "train/expert_load_entropy_normalized",
    "train/entropy_term_normalized",
    "train/avg_recursion_steps",
    "depth/allocation_error_abs",
    "halt/early_exit_rate",
    "halt/forced_exit_rate",
    "dispatch/evals_per_token",
    "diag/max_pairwise_cosine_sim",
    "perf/throughput_tokens_sec",
    "total_params",
)

# Quantities that are a tautology rather than a measurement for one architecture,
# and that engine.py still emits as a float. See aggregate().
SUPPRESS_CONSTANT = {
    "moe": ("train/avg_recursion_steps",     # max_depth = 1, so depth is 1 by construction
            "val/avg_recursion_steps"),      # T11.1 item 7: same, on the val pass
}

# Provenance fields that must be identical across all 15 runs. Architecture and
# seed are expected to differ; config_hash is expected to differ BY architecture
# and is checked per-architecture instead.
INVARIANT = (
    "experiment_group", "variant", "dataset_version", "train_split_version",
    "resolved_epochs", "resolved_batch_size", "resolved_subset_fraction",
    "halt_target_mode", "routing_supervision_weight", "family_cls_weight",
    "resolved_routing_mode", "resolved_router_noise",
)

# Config keys that the seed legitimately writes into. `config_hash()` hashes the
# whole config minus `provenance` and `logging` (run_context.py:82), and --seed
# lands in `data.subset_seed` / `training.subset_seed`, which are INSIDE that.
# So config_hash necessarily differs across the five seeds of one architecture.
#
# The first version of this driver treated that as a defect ("seeds must not
# change the config") and reported three problems on an otherwise clean 15-run
# matrix. The predicate was wrong, not the runs. The property that actually has
# to hold -- so that five seeds may share one row -- is that the configs are
# identical *once the seed is factored out*, which is what SEED_KEYS lets
# seed_blind_diff() check by naming the differing field paths rather than
# comparing two opaque hashes.
#
# config_hash() itself is deliberately NOT changed: subset_seed genuinely
# selects a different data subset whenever subset_fraction < 1.0, so dropping it
# from the hash would let two different datasets share one identity. On these 15
# runs subset_fraction is 1.0, so the field is provably inert.
SEED_KEYS = ("data.subset_seed", "training.subset_seed")


def run_name(arch: str) -> str:
    """canonical_spec.json:run_name_template -- train.py appends _seed{N}__{hash}."""
    return f"phaseB_{arch}"


def find_dir(arch: str, seed: int, canonical_only: bool = False) -> Path | None:
    """Locate a finished run for this pair.

    `canonical_only` is what the resume path uses: a directory stamped
    `exploratory` must NOT satisfy the skip check, or the undeclared-group failure
    becomes permanent -- the driver would skip the pair forever and report it as
    excluded on every subsequent invocation.
    """
    hits = sorted(RUNS.glob(f"{run_name(arch)}_seed{seed}__*"))
    for h in hits:
        if not (h / "metrics.json").exists():
            continue
        if canonical_only:
            prov = load(h, "resolved_config.json").get("provenance", {})
            if prov.get("experiment_group") != CANONICAL_GROUP:
                continue
        return h
    return None


def load(path: Path, name: str) -> dict:
    try:
        return json.loads((path / name).read_text())
    except Exception:
        return {}


def launch(arch: str, seed: int) -> tuple[bool, float]:
    """Rule 1: --architecture, --seed, --run_name, --experiment_group. NOTHING ELSE.

    `--experiment_group` is NOT a config override -- it is the canonical *claim*
    the Gate 0 guard then validates against canonical_spec.json. Without it
    `logging.experiment_group` is None and the guard returns the default
    'exploratory' silently, with no error and no refusal, because exploratory work
    is meant to stay cheap. The first attempt at this matrix omitted the flag and
    produced six perfectly-canonical-looking runs stamped `exploratory` -- every
    enforced field matched and `variant` read `canonical`, but the group field
    (the one thing the exporter filters on) said the runs were proxies. They were
    archived, not re-stamped: a group is declared at launch and validated by the
    guard, never edited onto a finished run.
    """
    cmd = [PY, "train.py",
           "--architecture", arch,
           "--seed", str(seed),
           "--run_name", run_name(arch),
           "--experiment_group", CANONICAL_GROUP]
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(CODE))
    return proc.returncode == 0, (time.time() - t0) / 60.0


def mean_std(values: list) -> dict | None:
    """Delegates to `seed_stats.mean_std`. See that module for the n-1 reasoning.

    T11.0b removed the local reimplementation. It was correct, but it was a third
    copy: this driver, `automated/phase10_ablations.py` and the exporter each had
    their own, and the reporting layer had already drifted once (it divided by n
    while three ablation drivers divided by n-1). One definition now.
    """
    return _seed_stats.mean_std(values)


def fmt(agg: dict | None, prec: int = 6, width: int = 20) -> str:
    if agg is None:
        return "N/A".rjust(width)
    if agg.get("std") is None:
        return f"{agg['mean']:.{prec}f} +- N/A(n=1)".rjust(width)
    return f"{agg['mean']:.{prec}f} +- {agg['std']:.{prec}f}".rjust(width)


def gap_reading(a_vals: list, b_vals: list) -> str:
    """Rule 4, REPLACED AT T11.0b. Exact two-sided randomization test.

    Was: "pooled std is the larger of the two arms' seed stds -- the honest
    yardstick, since a gap has to clear the noisier arm to mean anything." That
    reasoning is wrong, and it cost this project a real result. `max(std_a, std_b)`
    is the spread of ONE arm's five seed values; a difference of MEANS must be
    compared against the standard error OF THE DIFFERENCE,
    sqrt(s_a^2/n_a + s_b^2/n_b), which at five seeds per arm is about sqrt(5)
    smaller. Concretely: MoRE - MoR printed "1.93x the larger seed std -> inside
    seed noise" and is in fact 3.94 standard errors, exact p = 0.0159.

    Note the direction of the error. A too-CONSERVATIVE rule feels safe, so it went
    unexamined for the whole project; what it actually did was suppress two of the
    three pairwise findings in the headline matrix and two Phase 10 arms. Being
    wrong toward silence is still being wrong.

    Takes the RAW per-seed lists, not the aggregates: a permutation test needs the
    individual values, and it needs no variance estimate at all, so there is no
    pooled-std choice left to get wrong. `gap_over_std` is still printed so the
    old necessary condition can be checked against the new verdict.
    """
    res = _seed_stats.perm_test(a_vals, b_vals)
    if res is None:
        return "N/A -- fewer than 2 seeds in one arm, no test is possible"
    at_floor = " AT-FLOOR" if res["p_value"] <= res["min_p"] + 1e-12 else ""
    verdict = "SIGNIFICANT" if res["p_value"] < 0.05 else "not significant"
    return (f"{res['gap']:+.6f}  se={res['se_diff']:.6f} d={res['cohens_d']:+.2f} "
            f"p={res['p_value']:.4f} (floor {res['min_p']:.4f}{at_floor}, "
            f"{res['n_perms']} perms, {res['gap_over_std']:.2f}x seed std) "
            f"-> {verdict}")


def flatten(d: dict, prefix: str = "") -> dict:
    """`{'a': {'b': 1}}` -> `{'a.b': 1}`, so a mismatch can be named exactly."""
    out: dict = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        else:
            out[key] = v
    return out


def scientific_config(run_dir: Path) -> dict:
    """The part of resolved_config.json that defines the experiment.

    Mirrors `config_hash()` (drop `provenance` and `logging`) and then also drops
    the seed fields, because a seed is not a configuration -- it is the thing the
    five rows of one arm are meant to differ in. See SEED_KEYS.
    """
    cfg = load(run_dir, "resolved_config.json")
    flat = flatten({k: v for k, v in cfg.items()
                    if k not in ("provenance", "logging")})
    return {k: v for k, v in flat.items() if k not in SEED_KEYS}


def seed_blind_diff(dirs: list[Path]) -> list[str]:
    """Field paths on which any two of these runs disagree, seed excluded.

    Empty means the five seeds of this architecture are the same experiment and
    may be averaged into one row. Non-empty names the offending field, which is
    strictly more useful than "N distinct config_hash values".
    """
    if len(dirs) < 2:
        return []
    ref = scientific_config(dirs[0])
    bad: dict[str, set] = {}
    for d in dirs[1:]:
        cur = scientific_config(d)
        for k in set(ref) | set(cur):
            a, b = ref.get(k, "<absent>"), cur.get(k, "<absent>")
            if a != b:
                bad.setdefault(k, set()).update((repr(a), repr(b)))
    return [f"{k}: {' vs '.join(sorted(v))}" for k, v in sorted(bad.items())]


def collect() -> tuple[dict, list[str]]:
    """Read every run off disk. Rule 2: admissibility is decided here, from
    provenance, not from whether the process exited 0."""
    problems: list[str] = []
    per_arch: dict = {}

    for arch in ARCHES:
        runs: dict = {}
        for seed in SEEDS:
            d = find_dir(arch, seed)
            if d is None:
                problems.append(f"{arch} seed {seed}: no run directory with metrics.json")
                continue
            m = load(d, "metrics.json")
            prov = load(d, "resolved_config.json").get("provenance", {})

            group = prov.get("experiment_group")
            variant = prov.get("variant")
            declared = prov.get("seed_declared")
            if group != CANONICAL_GROUP:
                problems.append(f"{arch} seed {seed}: experiment_group={group!r} "
                                f"(need {CANONICAL_GROUP!r}) -- EXCLUDED from aggregate")
                continue
            if variant != "canonical":
                problems.append(f"{arch} seed {seed}: variant={variant!r} -- EXCLUDED")
                continue
            if declared is not True:
                problems.append(f"{arch} seed {seed}: seed_declared={declared!r} -- EXCLUDED")
                continue

            runs[seed] = {"dir": d.name, "metrics": m, "prov": prov}
            # total_params lives in provenance, not metrics.json (it is an
            # admissibility field, not a measurement -- the parameter-matching
            # claim rests on it). Surfaced into the metric dict so the report can
            # print it in the primary table beside the loss it has to be read with.
            m["total_params"] = prov.get("total_params")
        per_arch[arch] = runs

    # Invariant provenance across every admitted run (CLAUDE.md 6: one table may
    # not mix configurations).
    admitted = [(a, s, r) for a, rs in per_arch.items() for s, r in rs.items()]
    compared = 0
    if admitted:
        ref_a, ref_s, ref = admitted[0]
        for field in INVARIANT:
            base = ref["prov"].get(field)
            for a, s, r in admitted[1:]:
                compared += 1
                if r["prov"].get(field) != base:
                    problems.append(
                        f"protocol mismatch on {field}: {ref_a} seed {ref_s} = "
                        f"{base!r} but {a} seed {s} = {r['prov'].get(field)!r}")
    # Absence is never a pass -- same defect class as a sentinel reported as a
    # measurement (CLAUDE.md 4) and as the suite's EMPTY verdict.
    if compared == 0:
        problems.append("invariant check compared 0 fields: fewer than 2 admitted runs")

    # config_hash: expected to differ by architecture AND by seed (the seed is
    # inside the hashed config -- see SEED_KEYS). Recorded for provenance, but the
    # admissibility question is asked seed-blind, on named fields.
    hashes = {}
    seed_blind: dict = {}
    for arch, rs in per_arch.items():
        hs = {r["prov"].get("config_hash") for r in rs.values()}
        hashes[arch] = sorted(h[:8] if h else None for h in hs)
        dirs = [RUNS / r["dir"] for _, r in sorted(rs.items())]
        diffs = seed_blind_diff(dirs)
        seed_blind[arch] = diffs
        for d in diffs:
            problems.append(f"{arch}: seeds disagree on a non-seed config field -- {d}")
        if len(dirs) < 2:
            problems.append(f"{arch}: seed-blind config comparison had fewer than "
                            f"2 admitted runs -- nothing was compared")

    return {"per_arch": per_arch, "compared": compared, "hashes": hashes,
            "seed_blind": seed_blind}, problems


def aggregate(per_arch: dict) -> dict:
    out: dict = {}
    for arch, runs in per_arch.items():
        agg: dict = {}
        raw: dict = {}
        for key in SCALARS:
            vals = [r["metrics"].get(key) for r in runs.values()]
            agg[key] = mean_std(vals)
            # T11.0b: the raw per-seed vector is kept alongside the aggregate
            # because the randomization test in gap_reading() needs the individual
            # draws, not mean/std. Filtered to numbers here so callers never have
            # to re-handle the "N/A" strings engine.py writes.
            raw[key] = [v for v in vals if isinstance(v, (int, float))]
        # Rule 3, the one case metrics.json does not already handle. Every other
        # inapplicable quantity is written as the string "N/A" by engine.py, but
        # MoE's train/avg_recursion_steps comes out as the float 1.0 -- which is
        # not a measurement, it is a restatement of max_depth = 1. Printed as
        # "1.0000 +- 0.0000" beside MoRE's 2.13 it reads as a comparison of
        # learned depths. Suppressed here rather than in engine.py so no existing
        # run's metrics.json changes meaning (CLAUDE.md 4, results_exp.md 5).
        for key in SUPPRESS_CONSTANT.get(arch, ()):
            agg[key] = None
            raw[key] = []
        out[arch] = {"n_seeds": len(runs), "seeds": sorted(runs), "metrics": agg,
                     "raw": raw}
    return out


def report(data: dict, agg: dict, problems: list[str]) -> None:
    W = 96
    print()
    print("=" * W)
    print("T9.1 -- CANONICAL PHASE-B MATRIX")
    print("MoE / MoR / MoRE  x  seeds 42-46  |  experiment_group = canonical_phase_b")
    print("=" * W)
    print(f"admitted runs: " + "  ".join(
        f"{a}={agg[a]['n_seeds']}/5" for a in ARCHES))
    print(f"config_hash per architecture (differs by architecture AND by seed --")
    print(f"    the seed is inside the hashed config; identity is checked seed-blind below):")
    for a in ARCHES:
        print(f"    {a:5s} {data['hashes'].get(a)}")
    print(f"seed-blind config identity (non-seed fields, all 5 seeds of one arch):")
    for a in ARCHES:
        d = data["seed_blind"].get(a) or []
        print(f"    {a:5s} {'identical' if not d else d}")
    print(f"invariant provenance comparisons: {data['compared']}")
    print(f"predict-the-train-mean floor on val = {FLOOR:.6f}   "
          f"6-expert routing chance = {CHANCE_6:.3f}")

    print()
    print("-- PRIMARY METRIC: val/task_loss (lower is better) " + "-" * 44)
    print(f"{'arch':6s} {'val/task_loss':>24s} {'best_val_loss':>24s} "
          f"{'vs floor (R^2)':>16s} {'params':>12s}")
    for a in ARCHES:
        m = agg[a]["metrics"]
        vt = m["val/task_loss"]
        r2 = "N/A" if vt is None else f"{1.0 - vt['mean'] / FLOOR:.4f}"
        pm = m["total_params"]
        pstr = "N/A" if pm is None else f"{int(round(pm['mean'])):,}"
        print(f"{a:6s} {fmt(vt, 6, 24)} {fmt(m['best_val_loss'], 6, 24)} "
              f"{r2:>16s} {pstr:>12s}")

    print()
    print("-- RULE 4: exact two-sided randomization test on val/task_loss " + "-" * 33)
    for x, y in (("more", "moe"), ("more", "mor"), ("moe", "mor")):
        print(f"  {x:>4s} - {y:<4s}  "
              f"{gap_reading(agg[x]['raw']['val/task_loss'], agg[y]['raw']['val/task_loss'])}")
    print("  Verdicts come from the permutation p-value, never from a k x std")
    print("  threshold (superseded T11.0b -- see gap_reading.__doc__). `floor` is the")
    print("  smallest p this arm size can produce; a p AT-FLOOR is the test's")
    print("  resolution limit, not a strong result, and will not survive a")
    print("  multiple-comparison correction. This driver does not rank the")
    print("  architectures; it reports the three pairwise tests.")

    groups = (
        ("ROUTING -- specialization (N/A for MoR: one expert)", (
            "val/routing_accuracy", "val/routing_hungarian_accuracy",
            "val/routing_ami", "val/routing_purity",
            "val/routing_matched_macro_recall")),
        ("DEPTH -- adaptive computation (N/A for MoE: max_depth 1)", (
            "train/avg_recursion_steps", "depth/allocation_error_abs",
            "halt/early_exit_rate", "halt/forced_exit_rate")),
        ("DIAGNOSTICS -- not specialization, read beside Hungarian", (
            "train/expert_load_entropy_normalized", "train/entropy_term_normalized",
            "diag/max_pairwise_cosine_sim", "train/classification_loss")),
        ("SPARSITY AND COST", (
            "dispatch/evals_per_token", "perf/throughput_tokens_sec")),
    )
    for title, keys in groups:
        print()
        print(f"-- {title} " + "-" * max(0, 92 - len(title)))
        print(f"{'metric':40s} {'MoE':>20s} {'MoR':>20s} {'MoRE':>20s}")
        for k in keys:
            prec = 1 if k == "perf/throughput_tokens_sec" else 4
            print(f"{k:40s} " + " ".join(
                fmt(agg[a]["metrics"][k], prec, 20) for a in ARCHES))

    print()
    print("-- READING NOTES (results_exp.md) " + "-" * 61)
    print("  * routing chance = 0.167. Read raw accuracy ONLY beside the Hungarian")
    print("    number; a large raw<<Hungarian gap with AMI>0 is a real partition,")
    print("    arbitrarily numbered. raw ~ Hungarian ~ 1.0 on a canonical run means a")
    print("    label leak, not a result (routing_supervision_weight is 0.0 here).")
    print("  * dispatch/evals_per_token must be exactly 1.0 for MoE/MoRE -- that is")
    print("    what distinguishes Top-1 sparse dispatch from dense blend.")
    print("  * throughput is machine- and batch-specific. An engineering observation,")
    print("    never an architectural efficiency claim.")
    print("  * high load entropy is balance, not specialization; low cosine is")
    print("    'consistent with differentiated parameterizations', not orthogonality.")

    print()
    print("=" * W)
    if problems:
        print(f"PROBLEMS ({len(problems)}) -- the matrix is NOT clean:")
        for p in problems:
            print(f"  - {p}")
    else:
        print("No problems: 15/15 runs admitted as canonical_phase_b, invariant")
        print("provenance identical, and the five seeds of each architecture differ")
        print("in the seed alone (seed-blind config comparison found no differing field).")
    print("=" * W)


def main() -> int:
    report_only = "--report-only" in sys.argv

    if not report_only:
        # Interleaved by seed: a matrix killed halfway holds all three
        # architectures at every completed seed, which is a readable partial
        # result. Grouped by architecture it would hold five MoE runs and
        # nothing to compare them against.
        jobs = [(a, s) for s in SEEDS for a in ARCHES]
        total = len(jobs)
        for i, (arch, seed) in enumerate(jobs, 1):
            existing = find_dir(arch, seed, canonical_only=True)
            if existing is not None:
                print(f"[{i}/{total}] {arch} seed {seed} -- already on disk "
                      f"({existing.name}), skipping")
                continue
            print(f"[{i}/{total}] {arch} seed {seed} ...", flush=True)
            ok, mins = launch(arch, seed)
            print(f"        {'ok' if ok else 'FAILED'} in {mins:.1f} min", flush=True)
            if not ok:
                print(f"        train.py exited non-zero. Continuing so the rest of "
                      f"the matrix still runs; this pair will show as missing.",
                      flush=True)

    data, problems = collect()
    agg = aggregate(data["per_arch"])
    report(data, agg, problems)

    payload = {
        "task": "T9.1",
        "seeds": list(SEEDS),
        "architectures": list(ARCHES),
        "canonical_group": CANONICAL_GROUP,
        "val_floor_predict_train_mean": FLOOR,
        "routing_chance_6_experts": CHANCE_6,
        "config_hash_per_architecture": data["hashes"],
        "seed_blind_config_diff_per_architecture": data["seed_blind"],
        "invariant_comparisons": data["compared"],
        "aggregate": agg,
        "per_run_dirs": {a: {s: r["dir"] for s, r in rs.items()}
                         for a, rs in data["per_arch"].items()},
        "problems": problems,
        "clean": not problems,
    }
    RESULT.write_text(json.dumps(payload, indent=2))
    print(f"\nwritten: {RESULT.relative_to(ROOT)}")
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())




