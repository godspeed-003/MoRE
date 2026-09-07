"""T11.1 -- the results exporter. The ONLY sanctioned path from run directories
to a number in a table, a figure or the paper.

WHY THIS FILE EXISTS. Every previous results table in this project was assembled
by a human or an agent reading `metrics.json` and typing. That is how
`archive/pre_finalization/` came to contain tables that mixed a 3-epoch proxy with
a 20-epoch run, reported a `0.0` placeholder as a measurement, and quoted an
`expert_entropy` from a run whose dataset version no longer existed. None of those
were dishonesty; they were transcription. This module removes the transcription
step: it reads run directories, refuses anything it cannot certify, and writes the
CSV/JSON/Markdown that downstream documents include verbatim.

FOUR RULES, in the order they bind.

1. A RUN IS A ROW ONLY IF IT CERTIFIES ITSELF AS CANONICAL.
   Admission requires, from `resolved_config.json`, all of:
   `provenance.experiment_group == canonical_phase_b`, `variant == canonical`,
   `seed_declared == True`, `resolved_seed` in the frozen seed set, and
   `dataset_version` / `train_split_version` equal to the spec's. A directory that
   fails any check is recorded in `refusals` with the field that failed and is not
   aggregated. **A refusal is this module working**, not an error to route around.
   The archived `runs/phaseB_moe__seedNA__f83aa42f` (experiment_group
   `exploratory`, `provenance.architecture = None`) is the standing negative test:
   it must produce a refusal line and must not crash the exporter, because a
   crash is indistinguishable from "there were no bad runs".

2. CONSISTENCY IS A HARD ERROR, NOT A FOOTNOTE.
   Admitted rows must share one `dataset_version` and one `train_split_version`;
   within an architecture the non-seed config fields must be identical
   (seed-blind hash); no `(architecture, seed)` may appear twice. Any violation
   aborts with a non-zero exit and writes nothing. A table that silently spans two
   dataset versions is worse than no table, because it looks finished.

3. ABSENT, "N/A" AND 0.0 ARE THREE DIFFERENT THINGS.
   The canonical 15 runs do not all carry the same metric keys: MoE is missing the
   six `depth_dist/step_2..7_pct` keys and MoR is missing 43 routing keys
   (per-expert load, per-family precision/recall/f1 and their `_matched` variants,
   collapse count, Hungarian assignment) -- 49 absent slots in total. Absent means
   "this architecture has no such quantity"; `engine.py` writes the string "N/A"
   for the same reason where the key does exist. Both export as `N/A`. A `0.0`
   would export as `0.0` only if a run actually measured zero. This is CLAUDE.md
   4's no-sentinel rule applied at the table boundary.

4. NO VERDICT COMES FROM A k x std THRESHOLD.
   Pairwise comparisons use the exact two-sided randomization test in
   `code/seed_stats.py`, and every p-value is exported beside `min_p`, the smallest
   p the two arm sizes can produce. See `seed_stats` and T11.0b for why the old
   `2 x max(std_a, std_b)` rule was wrong in the direction of silence.

The primary metric is the spec's `protocol.primary_metric` read under the spec's
`protocol.checkpoint_selection` -- last epoch, not the best-epoch order statistic.
`best_val_loss` is exported as a secondary column so the selection-bias argument
stays checkable, never as the headline.

On ARITHMETIC, R^2 is derived here and never stored by a run:
`1 - val/task_loss / primary_metric_floor` against the frozen predict-the-train-mean
floor. It exists so a reader can see that all three architectures explain ~22% of
target variance and that the entire between-architecture spread is a few percent of
that -- the single most important piece of context for reading the matrix, and the
one most easily lost when a table shows six decimal places of loss and nothing else.
On LANGUAGE that same expression is not an R^2 and is not computed; see `derived()`.

Ablation arms are NOT re-derived here. They are ingested from
`automated/phase10_ablations_result.json`, which owns arm discovery, and are
written into a separately headed section that names its own reading rule and
resolution floor. Canonical and exploratory numbers never share a table
(CLAUDE.md 6).

TWO TASKS, ONE EXPORTER. `--task language` swaps the spec to
`canonical_spec_language.json` and the output directory to `results/language/`.
Everything else -- the admission filter, the consistency rules, the N/A boundary,
the randomization test -- is shared, because those four things are the reason this
module exists and a second copy of them would drift from this one. What the task
DOES change is the derived headline column: on arithmetic the floor is a
predict-the-train-mean MSE and `1 - loss/floor` is an R^2, on language the floor is
a bigram cross-entropy in nats and that same expression is not a variance ratio at
all. See `derived()`.

An arithmetic run and a language run can never land in one table: they differ in
`experiment_group`, in `dataset_version` and in `task`, and all three are admission
checks. That redundancy is deliberate -- one is a mean squared error and the other a
per-token cross-entropy, and CLAUDE.md 6 forbids the mixture absolutely.

Writes results/results.csv, results/results_aggregate.csv, results/results.json,
results/results_tables.md (under results/language/ for `--task language`).
Read-only with respect to runs/.

Usage:
    python code/export_results.py                       # arithmetic export
    python code/export_results.py --check               # admissions + consistency only
    python code/export_results.py --task language       # the language matrix
    python code/export_results.py --task language --check
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parent
ROOT = CODE.parent
RUNS = ROOT / "runs"
PHASE10_RESULT = ROOT / "automated" / "phase10_ablations_result.json"

sys.path.insert(0, str(CODE))
import seed_stats as _seed_stats                                    # noqa: E402

# Which task's canonical spec this invocation exports. Parsed from argv at IMPORT
# time, not inside main(), because the spec supplies CANONICAL_GROUP / PRIMARY /
# FLOOR and those are read by module-level defaults -- `pairwise(agg, key=PRIMARY)`
# binds its default when the function is defined. A `configure()` called from main()
# would leave that default pointing at the arithmetic spec while everything else
# pointed at the language one, which is the silent-wrong-number failure mode this
# whole module is built to prevent. A test that wants the language spec therefore
# sets `sys.argv` before importing; `test_language_export.py` does exactly that and
# says why.
TASKS = {
    "arithmetic": {"spec": "canonical_spec.json", "out": ("results",)},
    "language": {"spec": "canonical_spec_language.json",
                 "out": ("results", "language")},
}


def _task_from_argv(argv=None) -> str:
    argv = list(sys.argv[1:] if argv is None else argv)
    task = "arithmetic"          # absence means arithmetic, as everywhere else
    for i, a in enumerate(argv):
        if a == "--task" and i + 1 < len(argv):
            task = argv[i + 1]
        elif a.startswith("--task="):
            task = a.split("=", 1)[1]
    if task not in TASKS:
        raise SystemExit(f"--task must be one of {sorted(TASKS)}, got {task!r}")
    return task


TASK = _task_from_argv()
SPEC_PATH = CODE / TASKS[TASK]["spec"]
OUT = ROOT.joinpath(*TASKS[TASK]["out"])
SPEC = json.loads(SPEC_PATH.read_text())
CANONICAL_GROUP = SPEC["canonical_group"]
# The arithmetic spec has no `canonical_variant` key and its runs are stamped
# `variant = canonical`; the language spec names `language`. Read it from the spec
# so the admission check cannot be right for one task by hard-coding.
CANONICAL_VARIANT = SPEC.get("canonical_variant") or "canonical"
FROZEN_SEEDS = tuple(SPEC["seed_set"])
CANON_DSV = SPEC["enforced_fields"]["dataset_version"]
CANON_SPLIT = SPEC["enforced_fields"]["train_split_version"]
PRIMARY = SPEC["protocol"]["primary_metric"]
FLOOR = SPEC["protocol"]["primary_metric_floor"]
CHECKPOINT_RULE = SPEC["protocol"]["checkpoint_selection"]
ARCHES = ("moe", "mor", "more")
ARCH_LABEL = {"moe": "MoE", "mor": "MoR", "more": "MoRE"}

# The run-recorded margin over the baseline floor, if the task records one. Language
# runs write it from the corpus manifest; arithmetic runs have no such key, so the
# cross-check in `consistency_errors` simply finds nothing to check there.
FLOOR_MARGIN_KEY = ("val/nats_below_bigram_floor" if TASK == "language" else None)

# Rule 3's one case that metrics.json cannot express. MoE's max_depth is 1, so
# train/avg_recursion_steps is written as the float 1.0 -- true, but it is a
# restatement of the architecture, not a measurement of learned depth, and beside
# MoRE's 2.06 it reads as a comparison. Suppressed to N/A at the table boundary so
# no run's metrics.json has to change meaning. Kept identical to the same table in
# run_phase9_matrix.py; if you add an entry there, add it here.
#
# Task-independent on purpose: these keys are written by `engine.py`'s depth block,
# which does not know what task it is running, so a key that is a constant for MoE
# on arithmetic is a constant for MoE on language too.
SUPPRESS_CONSTANT = {
    "moe": ("train/avg_recursion_steps", "train/ponder_cost",
            "depth/allocation_error_abs", "depth/allocation_error_rel",
            # T11.1 audit item 7 added validation-pass twins of the depth block.
            # engine.py already writes "N/A" for these at max_depth == 1, so this
            # is belt-and-braces: it also covers a hand-assembled metrics.json.
            "val/avg_recursion_steps",
            "val/depth_allocation_error_abs", "val/depth_allocation_error_rel",
            "val/forced_exit_rate", "val/early_exit_rate", "val/mean_remainder"),
}

# The same argument by PREFIX, which the language metric layer made necessary. The
# language runs carry ~40 depth keys the arithmetic runs never had -- `depth/mean`,
# `depth/std`, `depth/hist/step_N`, `depth/mean_by_family/*`,
# `depth/embed_norm_logfreq_partial_norm`, `depth/by_document/*` -- and at
# max_depth 1 every one of them is a restatement of "MoE does not recurse", not a
# measurement. Enumerating them would guarantee that the next key added to the depth
# block is exported as a real MoE number by omission; the prefix cannot be forgotten.
#
# Safe on arithmetic as well, and applied there: the only two `depth/`-prefixed keys
# an arithmetic run writes are the two allocation-error keys already listed above, so
# this changes no published arithmetic cell.
SUPPRESS_PREFIX = {
    "moe": ("depth/",),
}

# Provenance fields copied verbatim into every exported row. updated_rules.md 9
# requires the full list on the W&B run; the same list has to survive into the
# offline table or a number in the paper cannot be traced back to a run directory.
#
# `task` is last and is resolved with a fallback: the language runs stamp the task at
# the config top level but `provenance.task` is currently None, and an arithmetic run
# carries no `task` key at all (T-L1.0's absence-means-arithmetic rule, which is what
# keeps the 15 published config hashes intact). Exported anyway so that a row in a
# CSV states which task's loss it is, instead of that being inferable only from the
# group name.
PROV_KEYS = (
    "experiment_id", "experiment_group", "architecture", "variant", "seed_declared",
    "resolved_seed", "dataset_version", "train_split_version", "config_hash",
    "code_git_commit", "code_git_dirty", "total_params", "resolved_epochs",
    "resolved_subset_fraction", "task",
)

# The config fields that legitimately differ across seeds, plus the run-identity
# fields that are seed-derived by construction. Everything else in
# resolved_config.json must be bit-identical within an architecture.
#
# Comparing raw config_hash cannot show that, because the seed is inside the hash,
# so all five seeds of one arm hash differently by construction. The exclusion list
# is therefore load-bearing and must stay minimal and justified per entry -- an
# over-broad list would let a real config difference through, which is the one
# failure mode of this check. `data.subset_seed` / `training.subset_seed` are the
# same two fields run_phase9_matrix.py:134 excludes (and see the comment there for
# why config_hash() itself is not changed); the rest are the seed itself, names that
# embed the seed, and per-run identity/timestamps that carry no scientific content.
SEED_KEYS = (
    "data.subset_seed", "training.subset_seed", "training.seed", "provenance.seed",
    "provenance.resolved_seed", "logging.run_name", "provenance.run_name",
    "provenance.experiment_id", "provenance.config_hash", "provenance.run_dir",
    "provenance.wandb_run_id", "provenance.wandb_run_name",
    "provenance.started_at", "provenance.finished_at", "provenance.hostname",
    "provenance.timestamp",
)


def flatten(obj, prefix: str = "") -> dict:
    """dotted-path flatten, so two configs can be diffed field by field rather than
    by hash. A list is compared as a whole (tuple), never element-wise: an ordering
    difference inside a list IS a config difference."""
    out: dict = {}
    if isinstance(obj, dict) and obj:
        for k, v in obj.items():
            out.update(flatten(v, f"{prefix}{k}."))
        return out
    out[prefix.rstrip(".")] = tuple(obj) if isinstance(obj, list) else obj
    return out


def load_json(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception as exc:                       # noqa: BLE001
        return {"__error__": f"{type(exc).__name__}: {exc}"}


def admit(d: Path) -> tuple[dict | None, str | None]:
    """Rule 1. Returns (row, None) for an admitted run or (None, reason) otherwise.

    Every failure path returns a reason string naming the field that failed. Nothing
    here may raise on malformed input: the whole point of the negative test is that a
    directory with `provenance.architecture = None` is REFUSED, and a refusal that
    arrives as a traceback is not a refusal, it is an outage.
    """
    rc_p, mt_p = d / "resolved_config.json", d / "metrics.json"
    if not rc_p.exists():
        return None, "no resolved_config.json (run never reached provenance write)"
    if not mt_p.exists():
        return None, "no metrics.json (run did not finish)"
    rc, mt = load_json(rc_p), load_json(mt_p)
    for name, blob in (("resolved_config.json", rc), ("metrics.json", mt)):
        if not isinstance(blob, dict) or "__error__" in blob:
            return None, f"{name} unreadable: {blob.get('__error__') if isinstance(blob, dict) else 'not an object'}"
    prov = rc.get("provenance") or {}
    if not isinstance(prov, dict):
        return None, "provenance is not an object"

    group = prov.get("experiment_group")
    if group != CANONICAL_GROUP:
        return None, f"experiment_group={group!r} != {CANONICAL_GROUP!r}"
    variant = prov.get("variant")
    if variant != CANONICAL_VARIANT:
        return None, f"variant={variant!r} != {CANONICAL_VARIANT!r}"
    # The task the run actually trained on. `provenance.task` is not stamped (the
    # language runs carry it at the config top level, arithmetic runs carry no `task`
    # key at all -- T-L1.0's absence-means-arithmetic rule), so it is resolved here
    # from the config rather than trusted from provenance. This check is REDUNDANT
    # with the group and dataset_version checks above and is here anyway: one task's
    # loss is a mean squared error and the other's a per-token cross-entropy, and a
    # single mislabelled group name must not be the only thing standing between them.
    run_task = rc.get("task") or "arithmetic"
    if run_task != TASK:
        return None, f"task={run_task!r} != {TASK!r} (exporting the {TASK} matrix)"
    if prov.get("seed_declared") is not True:
        return None, f"seed_declared={prov.get('seed_declared')!r} is not True"
    # architecture may live at the top level, inside provenance, or (for the
    # invalidated seedNA run) be present-but-None in both.
    arch = prov.get("architecture") or rc.get("architecture")
    if arch not in ARCHES:
        return None, f"architecture={arch!r} not in {ARCHES}"
    seed = prov.get("resolved_seed")
    if seed not in FROZEN_SEEDS:
        return None, f"resolved_seed={seed!r} not in frozen seed set {FROZEN_SEEDS}"
    if prov.get("dataset_version") != CANON_DSV:
        return None, f"dataset_version={prov.get('dataset_version')!r} != spec"
    if prov.get("train_split_version") != CANON_SPLIT:
        return None, f"train_split_version={prov.get('train_split_version')!r} != spec"
    if mt.get(PRIMARY) is None or isinstance(mt.get(PRIMARY), str):
        return None, f"primary metric {PRIMARY} absent or non-numeric ({mt.get(PRIMARY)!r})"

    # T11.1 audit item 7: the offline validation-pass depth sidecar, if present.
    # `val_depth_offline.json` is written by code/eval_val_depth.py from the run's
    # best-val checkpoint AFTER the run finished; it is NOT part of the run's own
    # metrics.json and is NOT measured at the same point in training as the
    # last-epoch primary metric. It is merged here rather than left to be read by
    # hand -- a number a human copies out of 15 JSON files is exactly the failure
    # this exporter exists to remove -- but it keeps its own `val_offline/` prefix
    # and its own `_offline_measured_at` field so no table can present it as if it
    # came from the training run. A run without the sidecar is still admitted: the
    # keys are simply absent, which the N/A boundary already renders correctly.
    side = load_json(d / "val_depth_offline.json")
    off_metrics: dict = {}
    off_at = None
    if isinstance(side, dict):
        off_at = side.get("measured_at")
        off_metrics = {k: v for k, v in side.items()
                       if k.startswith("val_offline/")}

    return {
        "run_dir": d.name,
        "architecture": arch,
        "seed": seed,
        "task": run_task,
        "offline_measured_at": off_at,
        "provenance": {k: prov.get(k) for k in PROV_KEYS}
        | {"architecture": arch, "task": run_task},
        "seed_blind_config": {k: v for k, v in flatten(rc).items()
                              if k not in SEED_KEYS},
        "metrics": {k: v for k, v in mt.items() if not isinstance(v, (dict, list))}
        # total_params lives in provenance, not metrics.json, but it is the one
        # provenance field that belongs in a results table: the MoE/MoRE parameter
        # identity and MoR's 0.120% deficit are what make the pairwise comparisons
        # interpretable. Injected under a `prov/` prefix so it aggregates and
        # exports like any other scalar without pretending to be a measured metric.
        | {"prov/total_params": prov.get("total_params")}
        | off_metrics,
        # Non-scalar metrics (the confusion matrix, the Hungarian assignment vector)
        # cannot go in a CSV cell, but dropping them would lose the two artifacts a
        # reader most needs to check the routing claims against. They are carried
        # here and written into results.json only; section 3 of the Markdown names
        # them so their absence from the tables is visible rather than silent.
        "metrics_nonscalar": {k: v for k, v in mt.items()
                              if isinstance(v, (dict, list))},
    }, None


def scan() -> tuple[list[dict], list[tuple[str, str]]]:
    rows, refusals = [], []
    for d in sorted(p for p in RUNS.iterdir() if p.is_dir()):
        row, reason = admit(d)
        if row is None:
            refusals.append((d.name, reason or "unknown"))
        else:
            rows.append(row)
    return rows, refusals


def consistency_errors(rows: list[dict]) -> list[str]:
    """Rule 2. Anything returned here aborts the export.

    The three checks are not interchangeable. Duplicate (arch, seed) means two run
    directories claim the same cell and the aggregate would double-count one seed.
    A split dataset_version means the table spans two datasets. A seed-blind config
    difference means one arm's five "replicates" are not replicates of the same
    configuration -- the most dangerous of the three, because it is invisible in
    every downstream artifact and turns a seed std into a config sweep.
    """
    errs: list[str] = []
    seen: dict[tuple, str] = {}
    for r in rows:
        key = (r["architecture"], r["seed"])
        if key in seen:
            errs.append(f"duplicate cell {key}: {seen[key]} and {r['run_dir']}")
        seen[key] = r["run_dir"]

    for field in ("dataset_version", "train_split_version", "code_git_commit"):
        vals = sorted({str(r["provenance"].get(field)) for r in rows})
        if len(vals) > 1:
            # code_git_commit is a WARNING-shaped fact but a hard error here: the
            # 15 canonical runs were launched from one working tree, so more than
            # one commit means the matrix was assembled across a code change and
            # the arms are not comparable until that diff is inspected.
            errs.append(f"admitted rows span {len(vals)} values of {field}: {vals}")

    for arch in ARCHES:
        arm = [r for r in rows if r["architecture"] == arch]
        if len(arm) < 2:
            continue
        ref = arm[0]["seed_blind_config"]
        for r in arm[1:]:
            diff = sorted(k for k in set(ref) | set(r["seed_blind_config"])
                          if ref.get(k) != r["seed_blind_config"].get(k))
            if diff:
                errs.append(f"{arch}: seed-blind config differs between "
                            f"{arm[0]['run_dir']} and {r['run_dir']}: {diff}")

    # The floor the RUNS measured themselves against must be the floor the SPEC
    # names, or every derived column in this document is computed against a
    # different baseline than the runs report. Language runs record
    # `val/nats_below_bigram_floor` from the corpus manifest's own floor, and that
    # quantity is linear in the loss, so `spec_floor - val/task_loss` must reproduce
    # it exactly for every admitted run.
    #
    # This is not redundant with the dataset_version check, and the difference is the
    # point: the spec and `data/lang/<corpus>/dataset_meta.json` are SEPARATE files.
    # Editing `primary_metric_floor` in the spec without rebuilding the corpus leaves
    # every admission check passing and silently shifts the margin column. Measured
    # size of the trap: the dev corpus floor is 5.3983 and the canonical one 4.9849,
    # so a mix-up moves every margin by 0.41 nats -- larger than any architecture gap
    # this study can hope to resolve.
    off = []
    for r in rows:
        rec, loss = r["metrics"].get(FLOOR_MARGIN_KEY), r["metrics"].get(PRIMARY)
        if not isinstance(rec, (int, float)) or isinstance(rec, bool):
            continue
        if not isinstance(loss, (int, float)) or isinstance(loss, bool):
            continue
        if abs((FLOOR - loss) - rec) > 1e-6:
            off.append(f"{r['run_dir']}: recorded {FLOOR_MARGIN_KEY}={rec:.6f} but "
                       f"spec floor {FLOOR:.6f} - {PRIMARY} {loss:.6f} = "
                       f"{FLOOR - loss:+.6f}")
    if off:
        errs.append("run-recorded floor margin disagrees with the spec's "
                    f"primary_metric_floor in {len(off)} run(s) -- the runs and this "
                    f"table are not measuring against the same baseline: "
                    + "; ".join(off[:3]) + ("; ..." if len(off) > 3 else ""))
    return errs


def metric_keys(rows: list[dict]) -> list[str]:
    """Union of every metric key across admitted rows, sorted. The union -- not the
    intersection -- is what makes Rule 3 visible: taking the intersection would make
    the 49 architecture-inapplicable slots disappear from the table entirely, which
    reads as "not measured" rather than "does not apply"."""
    keys: set[str] = set()
    for r in rows:
        keys.update(r["metrics"])
    return sorted(keys)


def cell(row: dict, key: str):
    """The single place absent / "N/A" / suppressed-constant become the string N/A.

    Booleans pass through unchanged: a flag like `dispatch/top1_verified` is a
    recorded fact, not a measurement, and turning it into N/A would hide it. It is
    non-numeric, so `aggregate()` excludes it from mean +- std by the same filter
    that excludes "N/A".
    """
    if key in SUPPRESS_CONSTANT.get(row["architecture"], ()):
        return "N/A"
    if key.startswith(SUPPRESS_PREFIX.get(row["architecture"], ())):
        return "N/A"
    if key not in row["metrics"]:
        return "N/A"                      # absent: this architecture has no such quantity
    v = row["metrics"][key]
    if isinstance(v, bool):
        return v
    if v is None or isinstance(v, str):
        return "N/A"                      # engine.py already refused to state it
    return v


def aggregate(rows: list[dict], keys: list[str]) -> dict:
    out: dict = {}
    for arch in ARCHES:
        arm = sorted((r for r in rows if r["architecture"] == arch),
                     key=lambda r: r["seed"])
        per_key: dict = {}
        for k in keys:
            vals = [cell(r, k) for r in arm]
            nums = [v for v in vals if isinstance(v, (int, float))
                    and not isinstance(v, bool)]
            ms = _seed_stats.mean_std(nums)
            per_key[k] = {
                "mean": None if ms is None else ms["mean"],
                "std": None if ms is None else ms["std"],
                "n": 0 if ms is None else ms["n"],
                "raw": nums,
                # n_na counts cells that export literally as "N/A" -- absent keys,
                # engine.py's own refusals, and suppressed constants. It is what a
                # reader needs to tell "this architecture has no such quantity"
                # (n_na == n_seeds) from "one seed failed to record it" (0 < n_na
                # < n_seeds). Booleans are neither: they are excluded from the mean
                # but are not N/A, so they are counted separately.
                "n_na": sum(1 for v in vals if v == "N/A"),
                "n_bool": sum(1 for v in vals if isinstance(v, bool)),
            }
        out[arch] = {"seeds": [r["seed"] for r in arm], "n_seeds": len(arm),
                     "metrics": per_key}
    return out


def pairwise(agg: dict, key: str = PRIMARY) -> list[dict]:
    """Rule 4. All three ordered pairs, each with its own resolution floor."""
    res = []
    for a, b in (("more", "moe"), ("more", "mor"), ("moe", "mor")):
        t = _seed_stats.perm_test(agg[a]["metrics"][key]["raw"],
                                  agg[b]["metrics"][key]["raw"])
        res.append({"pair": f"{ARCH_LABEL[a]} - {ARCH_LABEL[b]}", "metric": key,
                    "a": a, "b": b, "test": "exact two-sided randomization",
                    **({} if t is None else t),
                    "at_floor": bool(t and t["p_value"] <= t["min_p"] + 1e-12),
                    "significant_alpha05": bool(t and t["p_value"] < 0.05)})
    return res


# The derived headline columns, and they are NOT the same quantity on the two tasks.
#
# ARITHMETIC: the floor is a predict-the-train-mean MSE, so `1 - loss/floor` is the
# fraction of target variance explained -- a genuine R^2, and the context that stops
# a six-decimal loss table from reading as a large difference.
#
# LANGUAGE: the floor is a backoff-BIGRAM CROSS-ENTROPY in nats (4.9849, = 7.1918
# bits, = perplexity 146.2). `1 - CE/floor` is not a variance ratio and calling it
# R^2 would be a fabricated statistic wearing a familiar name -- a reviewer would
# read "R^2 = 0.13" as 13% of variance explained, which is not a claim the number
# supports. What IS honest is what the spec's own `primary_metric_note` names: the
# monotone transforms (bits/token = nats / ln 2, perplexity = exp nats) plus the
# margin over the floor in nats, POSITIVE when the run beats the baseline. The
# transforms carry no information the nats do not, so the verdict is still taken on
# nats -- they are here because a reviewer reads perplexity.
DERIVED_COLUMNS = {
    "arithmetic": ("R^2 vs floor",),
    "language": ("bits/token", "perplexity", "margin vs floor (nats)"),
}[TASK]


def derived(mean) -> dict:
    """Task-appropriate derived columns for the headline. Computed here and stored by
    no run, so they can never drift from the loss they are derived from.

    ONE CAVEAT THAT MUST BE PRINTED, NOT ASSUMED. The perplexity column is
    `exp(mean over seeds of nats)`, which is NOT the mean of the five per-seed
    perplexities -- exp is convex, so the second is always the larger. Each run also
    records its own `val/perplexity`, which appears in section 3 aggregated the other
    way round, and the two will not agree in the last digits. Neither is wrong; they
    answer different questions, and the verdict is taken on nats either way. The
    margin column has no such problem: it is linear in the loss, which is why
    `consistency_errors` can cross-check it against the run-recorded value."""
    if mean is None:
        return {k: "N/A" for k in DERIVED_COLUMNS}
    if TASK == "language":
        return {"bits/token": f"{mean / math.log(2):.4f}",
                "perplexity": f"{math.exp(mean):.2f}",
                "margin vs floor (nats)": f"{FLOOR - mean:+.4f}"}
    return {"R^2 vs floor": f"{1.0 - mean / FLOOR:.4f}"}


def load_ablations() -> dict | None:
    """Ingest the Phase 10 arms rather than rediscovering them. That driver owns arm
    discovery, its own per-arm seed sets and its own resolution floors; duplicating
    that here would create a second definition of the ablation table, which is the
    exact failure this module exists to prevent.

    ARITHMETIC ONLY, and the guard is load-bearing rather than tidy:
    `automated/phase10_ablations_result.json` holds arithmetic MSE gaps against an
    arithmetic MoRE baseline. Without the task check a language export would ingest
    that file and print those gaps into a cross-entropy document -- the exact
    unit-mixing CLAUDE.md 6 forbids, arriving through a path with no run directory to
    refuse. Language ablations, when Phase L-9 produces them, get their own result
    file and their own entry here."""
    if TASK != "arithmetic" or not PHASE10_RESULT.exists():
        return None
    blob = load_json(PHASE10_RESULT)
    return None if "__error__" in blob else blob


def fmt_ms(a: dict, prec: int = 6) -> str:
    if a["n"] == 0:
        return "N/A"
    if a["std"] is None:
        return f"{a['mean']:.{prec}f} +- N/A (n=1)"
    return f"{a['mean']:.{prec}f} +- {a['std']:.{prec}f}"


def write_runs_csv(rows: list[dict], keys: list[str]) -> Path:
    """Long format, one row per run: provenance then every metric. This is the file a
    reviewer re-analyses from; it contains no aggregates and no verdicts, so it cannot
    encode a reading rule."""
    path = OUT / "results.csv"
    header = list(PROV_KEYS) + ["run_dir"] + keys
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for r in sorted(rows, key=lambda r: (ARCHES.index(r["architecture"]), r["seed"])):
            prov = r["provenance"]
            w.writerow([prov.get(k) if prov.get(k) is not None else "N/A"
                        for k in PROV_KEYS]
                       + [r["run_dir"]] + [cell(r, k) for k in keys])
    return path


def write_agg_csv(agg: dict, keys: list[str]) -> Path:
    """One row per (metric, architecture). `n` and `n_na` are columns, not footnotes:
    a mean over 3 of 5 seeds and a mean over 5 must not look alike."""
    path = OUT / "results_aggregate.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["metric", "architecture", "mean", "std", "n_numeric", "n_na",
                    "n_bool"])
        for k in keys:
            for arch in ARCHES:
                a = agg[arch]["metrics"][k]
                w.writerow([k, arch,
                            "N/A" if a["mean"] is None else f"{a['mean']:.10g}",
                            "N/A" if a["std"] is None else f"{a['std']:.10g}",
                            a["n"], a["n_na"], a["n_bool"]])
    return path


def write_md(rows, refusals, agg, keys, pw, abl) -> Path:
    """Markdown tables for inclusion in T11.2's results document and the paper.

    Grouped by metric-key prefix rather than by a hand-curated list of "interesting"
    metrics. A curated list is a place to quietly drop an unflattering number; the
    prefix grouping is mechanical, so every key in the union appears exactly once.
    """
    L: list[str] = []
    A = L.append
    A(f"# Exported results -- canonical {'language Phase L-B' if TASK == 'language' else 'Phase B'}")
    A("")
    A("GENERATED BY `code/export_results.py`. Do not edit by hand and do not copy")
    A("numbers out of a run directory into this file -- regenerate it.")
    A("")
    A(f"- task: `{TASK}`  |  spec: `{SPEC_PATH.name}` at "
      f"`{SPEC.get('spec_version')}`")
    A(f"- canonical group: `{CANONICAL_GROUP}`  |  frozen seeds: {list(FROZEN_SEEDS)}")
    A(f"- dataset_version: `{CANON_DSV}`  |  train_split_version: `{CANON_SPLIT}`")
    A(f"- primary metric: `{PRIMARY}` under checkpoint rule **{CHECKPOINT_RULE}**")
    if TASK == "language":
        A(f"- baseline floor on val: `{FLOOR:.6f}` nats/token -- the backoff BIGRAM "
          f"(T-L2.5), = {FLOOR / math.log(2):.4f} bits, perplexity "
          f"{math.exp(FLOOR):.1f}. A run that does not beat it has learned nothing a "
          f"two-column count table could not. The derived columns below are monotone "
          f"transforms of the nats; no verdict is taken on them.")
        A("- **this is a per-token cross-entropy and the arithmetic tables are a mean "
          "squared error. They may never appear in one table (CLAUDE.md 6).**")
    else:
        A(f"- predict-the-train-mean floor on val: `{FLOOR:.6f}` (R^2 below is derived "
          f"from it, not stored)")
    A(f"- admitted runs: " + ", ".join(
        f"{ARCH_LABEL[a]} {agg[a]['n_seeds']}/{len(FROZEN_SEEDS)} "
        f"(seeds {agg[a]['seeds']})" for a in ARCHES))
    A(f"- git commit of the admitted runs: "
      f"`{rows[0]['provenance'].get('code_git_commit')}` "
      f"(dirty={rows[0]['provenance'].get('code_git_dirty')})")
    A(f"- refused directories: {len(refusals)} (listed at the end; a refusal is the "
      f"admission filter working)")
    A("")
    A("## 1. Headline")
    A("")
    A(f"| architecture | {PRIMARY} (mean +- std) | best_val_loss (secondary) | "
      + " | ".join(DERIVED_COLUMNS) + " | params |")
    A("|---|---|---|" + "---|" * (len(DERIVED_COLUMNS) + 1))
    for a in ARCHES:
        m = agg[a]["metrics"]
        prm = m.get("prov/total_params", {"mean": None})
        pstr = "N/A" if prm["mean"] is None else f"{int(round(prm['mean'])):,}"
        dv = derived(m[PRIMARY]["mean"])
        A(f"| {ARCH_LABEL[a]} | {fmt_ms(m[PRIMARY])} "
          f"| {fmt_ms(m.get('best_val_loss', {'n': 0}))} | "
          + " | ".join(dv[k] for k in DERIVED_COLUMNS) + f" | {pstr} |")
    A("")
    if TASK == "language":
        A("The derived columns are transforms of the mean nats, in that order: "
          "`bits = nats / ln 2`, `perplexity = exp(nats)`, "
          f"`margin = {FLOOR:.4f} - nats` (positive BEATS the bigram baseline).")
        A("`exp` is convex, so the perplexity above is **not** the mean of the five")
        A("per-seed perplexities -- each run's own `val/perplexity` is in section 3 and")
        A("aggregates the other way round. The verdict is taken on nats either way.")
        A("")
    A("`best_val_loss` is the minimum over evaluated epochs -- an order statistic")
    A("whose downward bias scales with each arm's per-epoch validation noise, which")
    A("is not matched across architectures. It is a secondary column so that")
    A("selection bias stays checkable; it is never the headline.")
    A("")
    A("## 2. Pairwise tests on the primary metric")
    A("")
    A("| pair | gap | se(diff) | Cohen d | p (exact) | min_p | perms | verdict |")
    A("|---|---|---|---|---|---|---|---|")
    for p in pw:
        if "p_value" not in p:
            A(f"| {p['pair']} | N/A | N/A | N/A | N/A | N/A | N/A | "
              f"fewer than 2 seeds in an arm |")
            continue
        v = ("SIGNIFICANT" if p["significant_alpha05"] else "not significant")
        if p["at_floor"]:
            v += " (AT RESOLUTION FLOOR)"
        A(f"| {p['pair']} | {p['gap']:+.6f} | {p['se_diff']:.6f} "
          f"| {p['cohens_d']:+.2f} | {p['p_value']:.4f} | {p['min_p']:.4f} "
          f"| {p['n_perms']} | {v} |")
    A("")
    A("Exact two-sided randomization test (`code/seed_stats.py`); no k x std")
    A("threshold is used anywhere (superseded, T11.0b). `min_p` is the smallest")
    A("p-value these arm sizes can produce: a p at the floor is the test's")
    A("resolution limit and will not survive a multiple-comparison correction.")
    A("")
    _write_md_depth(L, rows, agg)
    return _write_md_tail(L, refusals, agg, keys, abl)


# T11.1 audit item 7. The keys, in report order, and the pairs worth testing.
# MoE is absent from both: at max_depth 1 there is no depth to allocate, so it has
# no row here rather than a row of zeros.
OFFLINE_DEPTH_KEYS = (
    "val_offline/avg_recursion_steps",
    "val_offline/depth_allocation_error_abs",
    "val_offline/depth_allocation_error_rel",
    "val_offline/early_exit_rate",
    "val_offline/forced_exit_rate",
    "val_offline/mean_remainder",
)


def _write_md_depth(L: list, rows: list[dict], agg: dict) -> None:
    """
    Section 2b: depth measured on the VALIDATION pass.

    Why this section exists as its own block rather than merged into section 3:
    every other depth number in this document (`depth/*`, `train/avg_recursion
    _steps`, `halt/*`) was accumulated inside the TRAINING loop -- under dropout,
    on training data, with the halt head mid-update. The paper's depth claim is
    about the trained model on held-out data, and the training-time number does
    not measure it. These do.

    They are also measured at a DIFFERENT point in training than the headline
    loss: the sidecar is computed from `checkpoint.pt`, which is the best-val
    checkpoint, while the primary metric is last-epoch. That caveat is printed,
    not implied.

    THE FIRST PARAGRAPH IS AN ARITHMETIC STATEMENT. On language the depth block was
    rebuilt in Phase L-6 to run on the validation pass and to record its own
    provenance per key (`depth_key_provenance` in every language metrics.json), so
    the premise "every other depth number here is training-time" is false there. The
    language branch below says so instead of repeating it; printing the arithmetic
    caveat over a language table would understate evidence that exists, which is the
    same class of error as overstating evidence that does not.
    """
    A = L.append
    have = [r for r in rows if r.get("offline_measured_at")]
    A("## 2b. Depth on the validation pass (offline, from checkpoint)")
    A("")
    if TASK == "language":
        # The paragraph below is an ARITHMETIC statement and would be false here.
        # On language the depth block was rebuilt in Phase L-6 to run on the
        # validation pass, and every key records its own pass and population in the
        # run's `depth_key_provenance` registry -- so the offline sidecar is a
        # redundancy on this task, not the only held-out depth source.
        A("On the language task the `depth/*` block is **already a validation-pass**")
        A("measurement: each key records its own pass, population and caveats in the")
        A("run's `depth_key_provenance` (carried into `results.json` under each run's")
        A("`metrics_nonscalar`). Read that registry before comparing two depth keys --")
        A("`depth_dist/*` is the training loop, `depth/hist/*` is validation over a")
        A("different population, and the two are documented not to agree.")
        A("")
        if not have:
            A("No `val_depth_offline.json` sidecar present, which on this task removes")
            A("a redundancy rather than the evidence: the held-out depth numbers are the")
            A("`depth/*` rows in section 3.")
            A("")
            return
    elif not have:
        A("No `val_depth_offline.json` sidecar found in any admitted run. Run")
        A("`python code/eval_val_depth.py` to produce them. Until then every depth")
        A("number in this document is a TRAINING-TIME measurement and any claim about")
        A("held-out depth allocation is unsupported.")
        A("")
        return
    if not have:
        return
    at = sorted({r["offline_measured_at"] for r in have})
    A(f"Source: `val_depth_offline.json` in {len(have)}/{len(rows)} admitted runs, "
      f"written by `code/eval_val_depth.py`.")
    A(f"Measured at: **{', '.join(at)}** -- NOT the last-epoch model the primary")
    A("metric comes from. The two numbers describe the same run at two different")
    A("points in training, so a depth figure and a loss figure from this document")
    A("may not be captioned as coming from one model state.")
    A("")
    A("| metric | " + " | ".join(ARCH_LABEL[a] for a in ARCHES) + " |")
    A("|---|" + "---|" * len(ARCHES))
    for k in OFFLINE_DEPTH_KEYS:
        cells = []
        for a in ARCHES:
            m = agg[a]["metrics"].get(k, {"n": 0})
            cells.append(fmt_ms(m, 4))
        A(f"| `{k}` | " + " | ".join(cells) + " |")
    A("")
    A("| metric | pair | gap | Cohen d | p (exact) | min_p | verdict |")
    A("|---|---|---|---|---|---|---|")
    for k in OFFLINE_DEPTH_KEYS[:3]:
        for p in pairwise(agg, key=k):
            if "p_value" not in p:
                continue
            v = "SIGNIFICANT" if p["significant_alpha05"] else "not significant"
            if p["at_floor"]:
                v += " (AT FLOOR)"
            A(f"| `{k}` | {p['pair']} | {p['gap']:+.4f} | {p['cohens_d']:+.2f} "
              f"| {p['p_value']:.4f} | {p['min_p']:.4f} | {v} |")
    A("")
    A("Pairs involving MoE are omitted rather than reported as N/A rows: MoE runs at")
    A("max_depth 1, so it allocates no depth and the quantity does not exist for it.")
    A("Reading these together: whether MoRE's extra recursion buys any better")
    A("agreement with the operation-complexity curriculum than MoR's is exactly the")
    A("`depth_allocation_error_abs` row, and it is the row a depth claim rests on.")
    A("")


def _write_md_tail(L: list, refusals, agg, keys, abl) -> Path:
    """Sections 3-5: the full metric union, the ablations, the refusals."""
    A = L.append
    A("## 3. All metrics (union over admitted runs, grouped by key prefix)")
    A("")
    A("`N/A` means either the key is absent for that architecture (the quantity does")
    A("not exist -- MoR has one expert, MoE has max_depth 1) or `engine.py` wrote the")
    A("string `N/A` because it refused to state a value. Neither is `0.0`.")
    A("")
    A("Non-scalar metrics (the routing confusion matrix, the Hungarian assignment")
    A("vector) are not table cells and are not shown here; they are carried in")
    A("`results.json` under each run's `metrics_nonscalar`.")
    A("")
    groups: dict[str, list[str]] = {}
    for k in keys:
        groups.setdefault(k.split("/")[0] if "/" in k else "(no prefix)", []).append(k)
    for g in sorted(groups):
        A(f"### `{g}`")
        A("")
        A("| metric | " + " | ".join(ARCH_LABEL[a] for a in ARCHES) + " |")
        A("|---|" + "---|" * len(ARCHES))
        for k in groups[g]:
            cells = []
            for a in ARCHES:
                m = agg[a]["metrics"][k]
                s = fmt_ms(m, 4)
                if m["n"] and m["n"] < agg[a]["n_seeds"]:
                    s += f" [{m['n_na']} N/A]"
                cells.append(s)
            A(f"| `{k}` | " + " | ".join(cells) + " |")
        A("")

    A("## 4. Ablation arms -- EXPLORATORY, NOT CANONICAL")
    A("")
    if abl is None:
        A("`automated/phase10_ablations_result.json` not present; nothing ingested."
          if TASK == "arithmetic" else
          "Not applicable on the language task: the Phase 10 ablation file holds "
          "arithmetic MSE gaps and is deliberately NOT ingested here (see "
          "`load_ablations`). Language ablations get their own result file.")
    else:
        A("Ingested verbatim from `automated/phase10_ablations_result.json`. These arms")
        A("are stamped `exploratory` and MUST NOT be merged into section 1 or 2: they")
        A("vary one field each against the canonical MoRE baseline and several carry")
        A("fewer seeds, so they have their own resolution floors.")
        A("")
        A(f"reading rule of record: {abl.get('reading_rule', 'N/A')}")
        A("")
        n_arms = len(abl.get("arms") or {}) or 1
        bonf = 0.05 / n_arms
        A("| arm | task | varied field | value | n arm / n base | gap vs MoRE "
          "| Cohen d | p (exact) | min_p | verdict |")
        A("|---|---|---|---|---|---|---|---|---|---|")
        for name, arm in sorted((abl.get("arms") or {}).items()):
            p = arm.get("primary") if isinstance(arm, dict) else None
            if not isinstance(p, dict) or "p_value" not in p:
                A(f"| `{name}` | {arm.get('task', 'N/A') if isinstance(arm, dict) else 'N/A'} "
                  f"| N/A | N/A | N/A | N/A | N/A | N/A | N/A | no test possible |")
                continue
            v = "SIGNIFICANT" if p["p_value"] < 0.05 else "not significant"
            if p["p_value"] <= p["min_p"] + 1e-12:
                v += " (AT FLOOR)"
            if p["p_value"] < 0.05:
                v += ", WORSE" if p["gap"] > 0 else ", BETTER"
                v += ("; survives Bonferroni" if p["p_value"] < bonf
                      else "; FAILS Bonferroni")
            A(f"| `{name}` | {arm.get('task', 'N/A')} | `{arm.get('field', 'N/A')}` "
              f"| `{arm.get('value', 'N/A')}` | {p.get('n_arm')} / {p.get('n_base')} "
              f"| {p['gap']:+.6f} | {p['cohens_d']:+.2f} | {p['p_value']:.4f} "
              f"| {p['min_p']:.4f} | {v} |")
        A("")
        A("`gap` is arm minus canonical MoRE on the same primary metric, so a positive")
        A("gap is a HIGHER validation loss, i.e. the ablation is worse. Bonferroni over")
        A(f"the {n_arms} arms of this phase is alpha = {bonf:.4f}. An arm at n=3 cannot")
        A("produce a p below 0.0179, so an AT-FLOOR result at that seed count is the")
        A("test's resolution limit and cannot clear the corrected threshold at all.")
        probs = abl.get("problems") or []
        if probs:
            A("")
            A("Recorded problems in the ablation export:")
            for p in probs:
                A(f"- {p}")
    A("")

    A("## 5. Refused directories")
    A("")
    A("Each line is a directory under `runs/` that did not certify itself as")
    A("canonical. This section is part of the result: an exporter that silently")
    A("skipped these would be indistinguishable from one that found nothing wrong.")
    A("")
    A("| directory | reason refused |")
    A("|---|---|")
    for name, reason in refusals:
        A(f"| `{name}` | {reason} |")
    A("")

    path = OUT / "results_tables.md"
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
    return path


def write_json(rows, refusals, agg, keys, pw, abl) -> Path:
    """The machine artifact. Carries the raw per-seed vectors alongside every
    aggregate so any downstream consumer can recompute a verdict without re-reading
    runs/, and so a stale aggregate cannot outlive the numbers it came from."""
    payload = {
        "generated_by": "code/export_results.py (T11.1)",
        "task": TASK,
        "spec_file": SPEC_PATH.name,
        "spec_version": SPEC.get("spec_version"),
        "canonical_group": CANONICAL_GROUP,
        "canonical_variant": CANONICAL_VARIANT,
        "frozen_seeds": list(FROZEN_SEEDS),
        "dataset_version": CANON_DSV,
        "train_split_version": CANON_SPLIT,
        "primary_metric": PRIMARY,
        "checkpoint_selection": CHECKPOINT_RULE,
        "primary_metric_floor": FLOOR,
        "primary_metric_units": ("nats/token cross-entropy" if TASK == "language"
                                 else "mean squared error"),
        "reading_rule": ("exact two-sided randomization test (code/seed_stats.py); "
                        "k x std thresholds are superseded (T11.0b)"),
        "admitted_runs": [
            {"run_dir": r["run_dir"], "architecture": r["architecture"],
             "seed": r["seed"], "task": r.get("task"), "provenance": r["provenance"],
             "offline_measured_at": r.get("offline_measured_at"),
             "metrics": {k: cell(r, k) for k in keys},
             "metrics_nonscalar": r["metrics_nonscalar"]}
            for r in sorted(rows, key=lambda r: (ARCHES.index(r["architecture"]),
                                                 r["seed"]))],
        "aggregate": {
            a: {"seeds": agg[a]["seeds"], "n_seeds": agg[a]["n_seeds"],
                "derived_vs_floor": derived(agg[a]["metrics"][PRIMARY]["mean"]),
                "metrics": agg[a]["metrics"]}
            for a in ARCHES},
        "pairwise": pw,
        # T11.1 audit item 7: pairwise tests on the OFFLINE validation-pass depth
        # keys, kept in their own block so no consumer can iterate `pairwise` and
        # silently mix a last-epoch loss test with a best-checkpoint depth test.
        "pairwise_offline_depth": {
            k: pairwise(agg, key=k) for k in OFFLINE_DEPTH_KEYS
            if any(agg[a]["metrics"].get(k, {"n": 0})["n"] for a in ARCHES)
        },
        "offline_depth_note": (
            "val_offline/* comes from code/eval_val_depth.py reading each run's "
            "best-val checkpoint.pt after the fact -- NOT from the training run, "
            "and NOT at the same point in training as the last-epoch primary metric"
        ),
        "refusals": [{"run_dir": n, "reason": why} for n, why in refusals],
        "ablations_exploratory": abl,
    }
    path = OUT / "results.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def main() -> int:
    check_only = "--check" in sys.argv
    print(f"task = {TASK}   spec = {SPEC_PATH.name} ({SPEC.get('spec_version')})")
    print(f"canonical group = {CANONICAL_GROUP}   variant = {CANONICAL_VARIANT}   "
          f"seeds = {list(FROZEN_SEEDS)}")
    print(f"output dir = {OUT.relative_to(ROOT)}")
    if not RUNS.exists():
        print(f"REFUSED: {RUNS} does not exist")
        return 2

    rows, refusals = scan()
    print(f"scanned {len(rows) + len(refusals)} directories under runs/")
    print(f"  admitted: {len(rows)}     refused: {len(refusals)}")
    for name, reason in refusals:
        print(f"    REFUSED  {name}\n             {reason}")

    if not rows:
        print("\nREFUSED: no run certified itself as canonical -- nothing to export.")
        return 2

    errs = consistency_errors(rows)
    if errs:
        print("\nCONSISTENCY FAILURE -- nothing written:")
        for e in errs:
            print(f"  - {e}")
        return 2
    print("consistency: one dataset_version, one split version, one commit, "
          "no duplicate cells, seed-blind configs identical within each arm")

    incomplete = [a for a in ARCHES
                  if len([r for r in rows if r['architecture'] == a]) != len(FROZEN_SEEDS)]
    if incomplete:
        print(f"WARNING: incomplete arms {incomplete} -- exported with n< "
              f"{len(FROZEN_SEEDS)}; every table states its own n")

    keys = metric_keys(rows)
    agg = aggregate(rows, keys)
    pw = pairwise(agg)
    n_na = sum(agg[a]["metrics"][k]["n_na"] for a in ARCHES for k in keys)
    print(f"metric keys in union: {len(keys)}     cells exported as N/A: {n_na}")

    if check_only:
        print("\n--check: admissions and consistency only, nothing written.")
        return 0

    OUT.mkdir(parents=True, exist_ok=True)
    abl = load_ablations()
    written = [write_runs_csv(rows, keys), write_agg_csv(agg, keys),
               write_md(rows, refusals, agg, keys, pw, abl),
               write_json(rows, refusals, agg, keys, pw, abl)]
    print()
    for p in written:
        print(f"written: {p.relative_to(ROOT)}")
    print()
    for p in pw:
        if "p_value" in p:
            print(f"  {p['pair']:>14s}  gap={p['gap']:+.6f}  d={p['cohens_d']:+.2f}  "
                  f"p={p['p_value']:.4f} (floor {p['min_p']:.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
