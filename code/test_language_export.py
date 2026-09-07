"""test_language_export.py - T-L10.0. The language results/aggregation path.

Run with the CPU interpreter:
    C:/Users/vedan/anaconda3/python.exe code/test_language_export.py

WHY THIS FILE EXISTS, AND IT IS NOT A HYPOTHETICAL.

HANDOFF.md told the operator to aggregate 39 GPU-hours of canonical language runs
with two commands:

    python code/seed_stats.py    --group canonical_lang_b
    python code/export_results.py --group canonical_lang_b

Both were wrong, and both were wrong SILENTLY. `seed_stats.py` is a pure library
with no `main()` and no argparse: it exited 0 and printed nothing, which an operator
reads as "no differences found". `export_results.py` never had a `--group` flag; it
ignored the argument, read `canonical_spec.json`, and exported the ARITHMETIC matrix
into `results/` -- 15 admitted rows of mean squared error under a heading the
operator would have read as their language result. Neither failure raises.

That is the class of defect this file exists to make impossible: a reporting path
that is only exercised once, after the expensive part, by someone who cannot tell a
right answer from a wrong one because they have never seen the right answer.

WHAT IS TESTED AND WHAT IS FAKED.

Faked: the run directories. There is no canonical language matrix yet -- that is the
point of running this BEFORE the matrix -- so 15 fixture runs are synthesized in a
temp directory and `export_results.RUNS`/`OUT` are pointed at it.

Not faked: the metric key shapes. Every fixture's `metrics.json` is built from a
REAL language run's metrics.json (221 scalar keys, 4 non-scalar), so the union,
the N/A boundary and the prefix suppression are exercised against the keys the
engine actually writes -- including the three the engine already derives
(`val/perplexity`, `val/bits_per_token`, `val/nats_below_bigram_floor`), the two it
writes as the string `"N/A"` (`depth/allocation_error_*`), and the 54-key `depth/`
block that only exists on this task. A hand-written 6-key fixture would pass while
the real thing failed on any of those.

Also not faked: every line of exporter logic. `admit`, `consistency_errors`,
`aggregate`, `cell`, `pairwise`, `derived` and all four writers are the shipped
functions, called on the fixtures.

THE ARGV CONTRACT. `export_results` resolves its task from `sys.argv` at IMPORT
time, because module-level defaults (`pairwise(agg, key=PRIMARY)`) bind the spec
when the function is defined. So this file sets `sys.argv` before the import. If
that looks fragile: the alternative, a `configure()` called from `main()`, leaves
`PRIMARY` bound to the arithmetic spec while `CANONICAL_GROUP` points at the
language one, and that is a wrong number rather than a crash.
"""

import json
import math
import os
import shutil
import sys
import tempfile

CODE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(CODE)
sys.path.insert(0, CODE)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# MUST precede the import. See THE ARGV CONTRACT above.
sys.argv = ["export_results.py", "--task", "language"]
import export_results as ex                                        # noqa: E402

_p, _f, _s = [], [], []


def check(name, cond, detail=""):
    (_p if cond else _f).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    return bool(cond)


def skip(name, why):
    _s.append(name)
    print(f"  [SKIP] {name}  -- {why}")


# ===========================================================================
print("\n=== the exporter resolved the LANGUAGE spec " + "=" * 34)
# ===========================================================================

check("task is 'language'", ex.TASK == "language", f"got {ex.TASK!r}")
check("spec file is canonical_spec_language.json",
      ex.SPEC_PATH.name == "canonical_spec_language.json", ex.SPEC_PATH.name)
check("canonical group is canonical_lang_b",
      ex.CANONICAL_GROUP == "canonical_lang_b", ex.CANONICAL_GROUP)
# The arithmetic runs are stamped variant='canonical', the language ones
# variant='language'. Hard-coding either breaks the other task's admission.
check("canonical variant is read from the spec, not hard-coded to 'canonical'",
      ex.CANONICAL_VARIANT == "language", ex.CANONICAL_VARIANT)
check("primary metric is val/task_loss", ex.PRIMARY == "val/task_loss", ex.PRIMARY)
check("floor is the wikitext-103 backoff bigram 4.98494701553838 (T-L2.5), NOT the "
      "dev corpus 5.3983",
      abs(ex.FLOOR - 4.98494701553838) < 1e-12, f"{ex.FLOOR!r}")
check("output directory is results/language, so a language export cannot overwrite "
      "the published arithmetic tables",
      ex.OUT.as_posix().endswith("results/language"), ex.OUT.as_posix())
check("frozen seeds are 42..46", list(ex.FROZEN_SEEDS) == [42, 43, 44, 45, 46],
      str(ex.FROZEN_SEEDS))


# ===========================================================================
print("\n=== fixtures: 15 runs built from a REAL language metrics.json " + "=" * 16)
# ===========================================================================

# The donor. Any finished language run will do -- what is borrowed is the KEY SET,
# not the values, and every value that matters to an assertion is overwritten below.
#
# NEWEST BY MTIME, not first alphabetically. The first attempt at this took the first
# directory in sorted order and drew `langB_MoRE_seed42__91c9bba1`, a run from before
# the Phase L-6 depth work: 158 scalar keys, 34 under `depth/`, and NO
# `depth_key_provenance` registry. The suite then "passed" its N/A-boundary checks
# against a key set the canonical matrix will not produce, and failed only the one
# assertion that happened to name the missing key. A fixture built from the oldest
# available run tests the exporter against history instead of against the contract.
DONOR = None
_cands = []
for name in sorted(os.listdir(os.path.join(ROOT, "runs"))):
    d = os.path.join(ROOT, "runs", name)
    if (name.startswith("langB_") and os.path.isfile(os.path.join(d, "metrics.json"))
            and os.path.isfile(os.path.join(d, "resolved_config.json"))):
        rc = json.load(open(os.path.join(d, "resolved_config.json"), encoding="utf-8"))
        if rc.get("task") == "language":
            _cands.append((os.path.getmtime(os.path.join(d, "metrics.json")), d))
if _cands:
    DONOR = max(_cands)[1]

TMP = None
if DONOR is None:
    skip("the whole fixture suite",
         "no finished language run under runs/ to borrow a key set from; run "
         "`python code/run_language_matrix.py --smoke` first")
else:
    print(f"  donor key set: {os.path.basename(DONOR)}")
    donor_m = json.load(open(os.path.join(DONOR, "metrics.json"), encoding="utf-8"))
    donor_rc = json.load(open(os.path.join(DONOR, "resolved_config.json"),
                              encoding="utf-8"))
    n_scalar = sum(1 for v in donor_m.values() if not isinstance(v, (dict, list)))
    check("the donor carries a realistic language key set (>150 scalars, incl. the "
          "depth/ block)",
          n_scalar > 150 and any(k.startswith("depth/") for k in donor_m),
          f"{n_scalar} scalar keys, "
          f"{sum(1 for k in donor_m if k.startswith('depth/'))} under depth/")
    # The current metric contract. If the newest language run on disk lacks this, the
    # fixture is being built from a pre-Phase-L-6 key set and the suppression and
    # provenance assertions below are testing history rather than the contract.
    check("the donor carries the depth_key_provenance registry, so the fixture is a "
          "post-Phase-L-6 key set",
          isinstance(donor_m.get("depth_key_provenance"), dict),
          f"{len(donor_m.get('depth_key_provenance') or {})} registered depth key "
          f"prefixes")

    # Per-arm loss levels chosen so the arms are separable and the ordering is
    # arbitrary: this file tests the PLUMBING, and a fixture whose numbers looked
    # like a result would be the more dangerous artifact.
    LOSS = {"moe": 4.60, "mor": 4.55, "more": 4.50}
    JITTER = {42: 0.000, 43: 0.004, 44: -0.003, 45: 0.002, 46: -0.001}
    PARAMS = {"moe": 5_584_908, "mor": 5_581_063, "more": 5_584_908}
    COMMIT = "0" * 40

    # Keys a one-expert arm cannot have. Not an exhaustive reproduction of what
    # engine.py omits for MoR -- it does not need to be. What is being tested is
    # that ABSENT keys export as N/A and are counted as n_na, and one prefix that
    # is genuinely absent for one arm exercises that boundary completely.
    MOR_ABSENT = ("expert_load/", "val/routing_precision/", "val/routing_recall/",
                  "val/routing_f1/", "val/routing_hungarian")

    def fixture(root, arch, seed, *, group=None, dsv=None, task="language",
                loss=None, margin_key=True, suffix=""):
        """One run directory, complete enough for `admit` to accept or refuse it."""
        rid = f"langB_{arch}_seed{seed}{suffix}"
        d = os.path.join(root, rid)
        os.makedirs(d, exist_ok=True)
        vl = LOSS[arch] + JITTER[seed] if loss is None else loss
        m = {}
        for k, v in donor_m.items():
            if arch == "mor" and k.startswith(MOR_ABSENT):
                continue
            m[k] = v
        m["val/task_loss"] = vl
        m["val/loss"] = vl + 0.01
        m["best_val_loss"] = vl - 0.02
        m["val/perplexity"] = math.exp(vl)
        m["val/bits_per_token"] = vl / math.log(2)
        if margin_key:
            m["val/nats_below_bigram_floor"] = 4.98494701553838 - vl
        else:
            m.pop("val/nats_below_bigram_floor", None)
        m["architecture"] = arch
        m["experiment_id"] = rid
        m["experiment_group"] = group or "canonical_lang_b"
        # MoE at max_depth 1: the depth block is written, and every value in it is a
        # restatement of the architecture. Left in the fixture ON PURPOSE so the
        # prefix suppression has something to suppress.
        if arch == "moe":
            for k in list(m):
                if k.startswith("depth/") and isinstance(m[k], (int, float)):
                    m[k] = 1.0
            m["train/avg_recursion_steps"] = 1.0
        rc = json.loads(json.dumps(donor_rc))
        rc["task"] = task
        rc["architecture"] = arch
        prov = rc.setdefault("provenance", {})
        prov.update({
            "experiment_id": rid, "experiment_group": group or "canonical_lang_b",
            "architecture": arch, "variant": "language", "seed_declared": True,
            "resolved_seed": seed, "seed": seed,
            "dataset_version": dsv or ex.CANON_DSV,
            "train_split_version": ex.CANON_SPLIT,
            "config_hash": f"{arch}{seed}" + "f" * 56,
            "code_git_commit": COMMIT, "code_git_dirty": False,
            "total_params": PARAMS[arch], "resolved_epochs": 3,
            "resolved_subset_fraction": 1.0,
        })
        # Seed-blind config identity: everything outside SEED_KEYS must match within
        # an arm or `consistency_errors` (correctly) refuses the whole export.
        rc.setdefault("data", {})["subset_seed"] = seed
        rc.setdefault("training", {})["seed"] = seed
        json.dump(m, open(os.path.join(d, "metrics.json"), "w", encoding="utf-8"))
        json.dump(rc, open(os.path.join(d, "resolved_config.json"), "w",
                           encoding="utf-8"))
        return d

    TMP = tempfile.mkdtemp(prefix="more_lang_export_")
    RUNS_DIR = os.path.join(TMP, "runs")
    OUT_DIR = os.path.join(TMP, "results", "language")
    os.makedirs(RUNS_DIR)
    for arch in ex.ARCHES:
        for seed in ex.FROZEN_SEEDS:
            fixture(RUNS_DIR, arch, seed)

    # --- the negative controls, in the same directory as the good runs ----------
    # A refusal that only fires in a clean directory is not a refusal.
    fixture(RUNS_DIR, "more", 42, group="lang_calib_b0p05", suffix="__calib")
    fixture(RUNS_DIR, "more", 43, dsv="lang-wikitext-2-bpe8192-len256-9f870794",
            suffix="__wt2")
    fixture(RUNS_DIR, "more", 44, task="arithmetic", suffix="__arith")
    unfinished = os.path.join(RUNS_DIR, "langB_more_seed45__nometrics")
    os.makedirs(unfinished)
    shutil.copy(os.path.join(RUNS_DIR, "langB_more_seed45", "resolved_config.json"),
                os.path.join(unfinished, "resolved_config.json"))

    from pathlib import Path                                        # noqa: E402
    ex.RUNS = Path(RUNS_DIR)
    ex.OUT = Path(OUT_DIR)

    rows, refusals = ex.scan()
    by_reason = {n: why for n, why in refusals}

    check("15 fixture runs admitted, 4 refused",
          len(rows) == 15 and len(refusals) == 4,
          f"admitted {len(rows)}, refused {len(refusals)}: {sorted(by_reason)}")
    check("a non-canonical experiment_group is refused BY GROUP",
          "canonical_lang_b" in by_reason.get("langB_more_seed42__calib", ""),
          by_reason.get("langB_more_seed42__calib"))
    check("a dev-corpus (wikitext-2) run is refused BY dataset_version -- the dev "
          "numbers select, they never report",
          "dataset_version" in by_reason.get("langB_more_seed43__wt2", ""),
          by_reason.get("langB_more_seed43__wt2"))
    # The check that keeps an MSE out of a cross-entropy table even if a group name
    # is wrong. Everything about this fixture is canonical EXCEPT `task`.
    check("a run stamped task=arithmetic is refused from the language export even "
          "with a canonical group, variant, seed and dataset_version",
          "task=" in by_reason.get("langB_more_seed44__arith", ""),
          by_reason.get("langB_more_seed44__arith"))
    check("a directory with a config but no metrics.json is refused as unfinished",
          "metrics.json" in by_reason.get("langB_more_seed45__nometrics", ""),
          by_reason.get("langB_more_seed45__nometrics"))
    check("every arm has all five seeds",
          all(len([r for r in rows if r["architecture"] == a]) == 5
              for a in ex.ARCHES),
          str({a: sorted(r["seed"] for r in rows if r["architecture"] == a)
               for a in ex.ARCHES}))

    errs = ex.consistency_errors(rows)
    check("the 15 admitted rows pass every consistency rule", errs == [], str(errs))


# ===========================================================================
print("\n=== the N/A boundary on language keys " + "=" * 40)
# ===========================================================================

if TMP is None:
    skip("N/A boundary", "no fixtures")
else:
    keys = ex.metric_keys(rows)
    agg = ex.aggregate(rows, keys)
    idx = {(r["architecture"], r["seed"]): r for r in rows}
    moe, more, mor = idx[("moe", 42)], idx[("more", 42)], idx[("mor", 42)]

    check("the metric union is taken, not the intersection (MoR's absent routing "
          "keys still appear as rows)",
          any(k.startswith("expert_load/") for k in keys),
          f"{len(keys)} keys in union")

    # The language-specific suppression. MoE at max_depth 1 writes ~54 depth keys and
    # every one is 'MoE does not recurse' restated; a 1.0 in a table beside MoRE's
    # 2.6 reads as a measured comparison.
    for k in ("depth/mean", "depth/std", "depth/hist/step_1",
              "depth/mean_by_family/L1_FUNCTION"):
        if k not in more["metrics"]:
            continue
        check(f"MoE's `{k}` is suppressed to N/A (depth is not a measurement at "
              f"max_depth 1)", ex.cell(moe, k) == "N/A", repr(ex.cell(moe, k)))
        check(f"MoRE's `{k}` is NOT suppressed", ex.cell(more, k) != "N/A",
              repr(ex.cell(more, k)))
    check("MoR's depth keys are NOT suppressed -- MoR is the arm depth is the whole "
          "point of", ex.cell(mor, "depth/mean") != "N/A",
          repr(ex.cell(mor, "depth/mean")))

    check("engine.py's own string \"N/A\" exports as N/A, not as 0.0 "
          "(depth/allocation_error_abs has no ground truth on English)",
          ex.cell(more, "depth/allocation_error_abs") == "N/A",
          repr(donor_m.get("depth/allocation_error_abs")))
    check("an ABSENT key exports as N/A for the arm that lacks it and as a number "
          "for the arm that has it",
          ex.cell(mor, "expert_load/expert_0_pct") == "N/A"
          and isinstance(ex.cell(more, "expert_load/expert_0_pct"), (int, float)),
          f"mor={ex.cell(mor, 'expert_load/expert_0_pct')!r}")
    a_mor = agg["mor"]["metrics"]["expert_load/expert_0_pct"]
    check("an all-absent key aggregates to n=0, n_na=5 -- distinguishable from one "
          "seed failing to record it",
          a_mor["n"] == 0 and a_mor["n_na"] == 5 and a_mor["mean"] is None,
          f"n={a_mor['n']}, n_na={a_mor['n_na']}, mean={a_mor['mean']!r}")
    check("a string-valued metric (routing_hungarian_assignment) never reaches a "
          "numeric cell", ex.cell(more, "val/routing_hungarian_assignment") == "N/A")


# ===========================================================================
print("\n=== derived columns: cross-entropy is not variance explained " + "=" * 17)
# ===========================================================================

if TMP is None:
    skip("derived columns", "no fixtures")
else:
    check("the language derived columns are bits/perplexity/margin and there is NO "
          "R^2 column (1 - CE/floor is not a variance ratio)",
          ex.DERIVED_COLUMNS == ("bits/token", "perplexity",
                                 "margin vs floor (nats)"),
          str(ex.DERIVED_COLUMNS))
    mean_more = agg["more"]["metrics"][ex.PRIMARY]["mean"]
    dv = ex.derived(mean_more)
    check("bits/token = nats / ln 2",
          abs(float(dv["bits/token"]) - mean_more / math.log(2)) < 5e-5,
          f"{dv['bits/token']} vs {mean_more / math.log(2):.6f}")
    check("perplexity = exp(nats)",
          abs(float(dv["perplexity"]) - math.exp(mean_more)) < 0.01,
          f"{dv['perplexity']} vs {math.exp(mean_more):.4f}")
    check("margin = floor - nats, and is POSITIVE when the run beats the bigram",
          abs(float(dv["margin vs floor (nats)"]) - (ex.FLOOR - mean_more)) < 5e-5
          and float(dv["margin vs floor (nats)"]) > 0,
          dv["margin vs floor (nats)"])
    check("a None mean yields N/A in every derived column, never a 0.0",
          set(ex.derived(None).values()) == {"N/A"}, str(ex.derived(None)))

    # The trap this guards: the spec's floor and the corpus manifest's floor are in
    # SEPARATE files, and the dev/canonical pair differ by 0.41 nats -- larger than
    # any architecture gap this study can resolve.
    bad = ex.fixture_floor_probe = None
    rows_bad = [dict(r) for r in rows]
    rows_bad[0] = dict(rows_bad[0])
    rows_bad[0]["metrics"] = dict(rows_bad[0]["metrics"])
    rows_bad[0]["metrics"]["val/nats_below_bigram_floor"] = (
        5.398304166059712 - rows_bad[0]["metrics"]["val/task_loss"])
    errs_bad = ex.consistency_errors(rows_bad)
    check("a run that measured its margin against the DEV corpus floor is a hard "
          "consistency failure, not a footnote",
          any("floor margin" in e for e in errs_bad),
          str(errs_bad)[:160])
    check("...and the same rows without that tampering still pass",
          ex.consistency_errors(rows) == [])


# ===========================================================================
print("\n=== pairwise verdicts come from the exact test " + "=" * 32)
# ===========================================================================

if TMP is None:
    skip("pairwise", "no fixtures")
else:
    pw = ex.pairwise(agg)
    check("all three ordered pairs are tested", len(pw) == 3,
          str([p["pair"] for p in pw]))
    check("every pair reports p, min_p and the permutation count",
          all({"p_value", "min_p", "n_perms"} <= set(p) for p in pw))
    check("min_p at 5v5 is 1/252 = 0.0040, the resolution limit five seeds can buy",
          all(abs(p["min_p"] - 1 / 252) < 1e-9 for p in pw),
          f"{pw[0]['min_p']:.6f}")
    check("no p-value is below its own floor",
          all(p["p_value"] >= p["min_p"] - 1e-12 for p in pw))
    check("the test is the randomization test, never a k x std threshold",
          all(p["test"] == "exact two-sided randomization" for p in pw))


# ===========================================================================
print("\n=== the four artifacts, written and read back " + "=" * 33)
# ===========================================================================

if TMP is None:
    skip("writers", "no fixtures")
else:
    ex.OUT.mkdir(parents=True, exist_ok=True)
    abl = ex.load_ablations()
    # The cross-task leak with no run directory to refuse it: the Phase 10 file
    # holds arithmetic MSE gaps against an arithmetic MoRE baseline.
    check("the arithmetic Phase 10 ablation file is NOT ingested into a language "
          "export, even though it exists on disk",
          abl is None and ex.PHASE10_RESULT.exists(),
          f"load_ablations()={type(abl).__name__}, "
          f"file exists={ex.PHASE10_RESULT.exists()}")

    p_runs = ex.write_runs_csv(rows, keys)
    p_agg = ex.write_agg_csv(agg, keys)
    p_md = ex.write_md(rows, refusals, agg, keys, pw, abl)
    p_json = ex.write_json(rows, refusals, agg, keys, pw, abl)
    check("all four artifacts written under results/language/",
          all(p.exists() for p in (p_runs, p_agg, p_md, p_json)),
          ", ".join(p.name for p in (p_runs, p_agg, p_md, p_json)))

    import csv as _csv
    with p_runs.open(encoding="utf-8") as fh:
        rr = list(_csv.DictReader(fh))
    check("results.csv has one row per admitted run and a `task` column stating the "
          "task on every row",
          len(rr) == 15 and all(r["task"] == "language" for r in rr),
          f"{len(rr)} rows, tasks={sorted({r['task'] for r in rr})}")
    check("results.csv carries the full provenance list, so a cell can be traced to "
          "a run directory",
          all(k in rr[0] for k in ex.PROV_KEYS) and "run_dir" in rr[0])

    md = p_md.read_text(encoding="utf-8")
    check("the markdown states nats/token and the bigram floor",
          "nats/token" in md and "4.9849" in md)
    check("the markdown does NOT contain an R^2 column on the language task",
          "R^2 vs floor" not in md)
    check("the markdown carries the perplexity Jensen caveat rather than implying "
          "exp(mean) == mean(exp)",
          "the mean of the five" in md and "convex" in md)
    check("the markdown forbids mixing the arithmetic MSE and this cross-entropy in "
          "one table", "may never appear in one table" in md)
    check("the depth section says the language depth block is already a validation "
          "measurement, not a training-time one",
          "already a validation-pass" in md)
    check("section 4 records that the arithmetic ablations were deliberately not "
          "ingested", "Not applicable on the language task" in md)

    # --- T-LX.5: the machine record keeps "N/A"; the human table drops all-dead
    # rows and NAMES them. These are different requests and conflating them undoes
    # the eleven-flat-routing-keys fix, so both halves are asserted here.
    dead = ex.dead_rows(rows, agg, keys)
    alloc = [k for k in keys if "allocation_error" in k]
    # The footnote is itself a markdown table, so "is this key rendered as a row"
    # has to be asked of the METRIC tables only -- everything above the footnote
    # heading. Asking it of the whole document conflates "dropped from the tables"
    # with "not mentioned anywhere", and the second is exactly what T-LX.5 forbids.
    _FOOT = "### Keys omitted from the tables above"
    check("the footnote heading is present, which is what makes the split below "
          "meaningful", _FOOT in md)
    md_tables = md.split(_FOOT)[0]
    md_foot = md.split(_FOOT)[-1]
    _reasons = {}
    for _d in dead.values():
        _reasons[_d["reason"]] = _reasons.get(_d["reason"], 0) + 1
    print(f"    dead rows on language: {len(dead)} of {len(keys)} union keys "
          f"-> {_reasons}")
    print(f"    structural (no arm has the quantity): "
          f"{sorted(k for k, d in dead.items() if d['reason'] == 'structural')}")
    check("there are allocation-error keys in the union at all (if this fails the "
          "rest of T-LX.5 is vacuous)", bool(alloc), f"{alloc}")
    check("every allocation-error key is classified STRUCTURAL: English has no "
          "per-token ground-truth depth, so the quantity does not exist rather than "
          "being unrecorded",
          all(dead.get(k, {}).get("reason") == "structural" for k in alloc),
          f"{ {k: dead.get(k, {}).get('reason') for k in alloc} }")
    check("no allocation-error key is rendered as a table row in the markdown",
          not any(f"| `{k}` |" in md_tables for k in alloc),
          f"still rendered: {[k for k in alloc if f'| `{k}` |' in md_tables]}")
    check("every dropped key is NAMED in the footnote, so dropping is not hiding",
          all(f"`{k}`" in md_foot for k in dead),
          f"unnamed: {[k for k in dead if f'`{k}`' not in md_foot]}")
    check("the footnote states the reason a structural key has no number, in the "
          "words that forbid filling it with 0.0",
          "curriculum that does not exist" in md)

    with p_runs.open(encoding="utf-8") as fh:
        rr2 = list(_csv.DictReader(fh))
    check("the MACHINE record still carries every dropped key: results.csv has a "
          "column per allocation-error key and the value is literally N/A",
          all(k in rr2[0] and all(r[k] == "N/A" for r in rr2) for k in alloc),
          f"columns present={[k in rr2[0] for k in alloc]}")
    with p_agg.open(encoding="utf-8") as fh:
        ag_keys = {r["metric"] for r in _csv.DictReader(fh)}
    check("results_aggregate.csv still carries a row per dropped key per arm",
          all(k in ag_keys for k in alloc))

    # Rule 3 must survive: a key that is N/A for ONE arm and numeric for another is
    # the asymmetry the union exists to show, and must NOT be dropped. `depth/mean`
    # is exactly that -- suppressed for MoE at max_depth 1, real on MoR and MoRE.
    check("a key that is N/A for MoE only is NOT dropped -- that asymmetry is Rule 3",
          "depth/mean" in keys and "depth/mean" not in dead
          and "| `depth/mean` |" in md,
          f"in keys={'depth/mean' in keys}, dead={'depth/mean' in dead}")
    check("and its MoE cell in that row still reads N/A rather than a number",
          any(l.startswith("| `depth/mean` |") and l.split("|")[2].strip() == "N/A"
              for l in md.splitlines()),
          next((l for l in md.splitlines()
                if l.startswith("| `depth/mean` |")), "(row missing)"))
    cat = {k: d for k, d in dead.items() if d["reason"] == "categorical"}
    small = sorted(k for k, d in cat.items() if len(d["values"]) <= 3)
    big = sorted(k for k, d in cat.items() if len(d["values"]) > 3)
    if small:
        k0 = small[0]
        check("a CATEGORICAL dead key prints its recorded value, because `N/A` in "
              "this document means does-not-exist and that would be false for it",
              all(f"`{v}`" in md_foot for v in cat[k0]["values"]),
              f"{k0} -> {cat[k0]['values']}")
    else:
        skip("categorical dead-key footnote", "no low-cardinality categorical key")
    if big:
        k1 = big[0]
        # config_hash and experiment_id have one value per run. Pasting 15 of them,
        # two being 64-char hashes, is less readable than the N/A row it replaced.
        check("a per-run identifier is summarised by count plus a pointer to "
              "results.csv rather than pasted 15 times into the footnote",
              f"{len(cat[k1]['values'])} distinct values" in md_foot
              and not any(len(v) > 40 and f"`{v}`" in md_foot
                          for v in cat[k1]["values"]),
              f"{k1} -> {len(cat[k1]['values'])} values")
    else:
        skip("high-cardinality footnote summary", "no per-run categorical key")
    check("all four refusals are listed in the document -- an exporter that hid them "
          "would be indistinguishable from one that found nothing wrong",
          all(n in md for n, _ in refusals), f"{len(refusals)} refusals")
    check("the spec version is recorded in the document",
          ex.SPEC["spec_version"] in md, ex.SPEC["spec_version"])

    blob = json.loads(p_json.read_text(encoding="utf-8"))
    check("results.json records the task and the units of the primary metric",
          blob["task"] == "language"
          and "cross-entropy" in blob["primary_metric_units"],
          blob.get("primary_metric_units"))
    check("results.json carries the raw per-seed vector for the primary metric, so a "
          "verdict can be recomputed without re-reading runs/",
          all(len(blob["aggregate"][a]["metrics"][ex.PRIMARY]["raw"]) == 5
              for a in ex.ARCHES))
    check("results.json reports derived_vs_floor and not r2_vs_floor",
          "derived_vs_floor" in blob["aggregate"]["more"]
          and "r2_vs_floor" not in blob["aggregate"]["more"])
    check("the non-scalar artifacts survive into results.json (depth_key_provenance "
          "is how a reader tells a train-pass depth key from a val-pass one)",
          any("depth_key_provenance" in r.get("metrics_nonscalar", {})
              for r in blob["admitted_runs"]),
          f"nonscalar keys on the first admitted run: "
          f"{sorted(blob['admitted_runs'][0].get('metrics_nonscalar', {}))}")
    check("results.json records what the markdown elided, so the elision is "
          "auditable against the per-run metrics in the same file",
          set(blob["markdown_omitted_all_na_keys"]) == set(dead)
          and all(blob["admitted_runs"][0]["metrics"][k] == "N/A" for k in alloc),
          f"{len(blob.get('markdown_omitted_all_na_keys', {}))} recorded "
          f"vs {len(dead)} dropped")

    # Headline sanity, read back from the written file rather than from memory.
    hl = [l for l in md.splitlines() if l.startswith("| MoRE |")]
    check("the headline MoRE row is present and carries a mean +- std, not a bare "
          "number", bool(hl) and "+-" in hl[0], hl[0] if hl else "(missing)")
    print(f"\n  headline rows as written:")
    for l in md.splitlines():
        if l.startswith(("| architecture", "| MoE |", "| MoR |", "| MoRE |")):
            print("    " + l)

    shutil.rmtree(TMP, ignore_errors=True)


# ===========================================================================
print("\n" + "=" * 78)
print(f"  T-L10.0 language export:  {len(_p)} passed, {len(_f)} failed, "
      f"{len(_s)} skipped")
if _f:
    print("\n  FAILED:")
    for n in _f:
        print(f"    - {n}")
print("=" * 78)
sys.exit(1 if _f else 0)
