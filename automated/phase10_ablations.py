"""
T10 -- PHASE 10 ABLATIONS.  MoRE, one field changed at a time, against the
canonical MoRE arm from T9.1.

WHY THIS EXISTS.  T9.1 came back Outcome C on the primary metric: MoR (one shared
block, no routing) has the lowest val task loss, and MoRE is indistinguishable
from MoE. So the question Phase 10 has to answer is no longer "how much does MoRE
win by" but "which of MoRE's two mechanisms is doing anything at all". Each arm
below removes or perturbs exactly one mechanism and asks whether the primary
metric moves by more than the seed noise.

THE BASELINE IS NOT RE-RUN.  The canonical MoRE runs from T9.1 (seeds 42-46, 50
epochs) are the baseline arm. That is only legitimate because they were produced
on the current code state and nothing in the model has changed since -- the T10.H
lesson was that two arms must not straddle a code change. `verify_baseline_state()`
enforces it by refusing if the model source has been modified since those runs.
Ablation arms use seeds 42-44, and the baseline is restricted to the SAME three
seeds for every comparison so the n matches; the 5-seed baseline value is printed
beside it for reference only.

ONE FIELD, PROVED BY CONSTRUCTION.  Each arm is a full config file generated from
config.json with exactly one leaf changed, and `write_arm_configs()` asserts the
flattened diff against the canonical MoRE config is exactly that leaf. This is
stronger than a CLI flag: `--config` means the arm's entire configuration is on
disk and diffable, and three of the five fields have no CLI flag at all.

THESE ARE NOT CANONICAL RUNS.  No arm passes --experiment_group, so every one is
stamped `exploratory` -- correct, because an ablation is by definition not the
frozen spec. `variant` carries the deviation name (`fixed_depth`,
`routing_dense_blend`, ...), which is what the ablation table prints.

READING RULE, FIXED BEFORE ANY RUN.  An arm has moved the primary metric only if
|arm - baseline| >= 2x the larger of the two seed stds. Anything smaller is
reported as inside seed noise and does NOT support a claim in either direction.
Same rule as T9.1 Rule 4. A mechanism whose removal costs nothing is a finding,
not a failure -- and it is the finding this project most needs to report honestly.

Writes automated/phase10_ablations_result.json.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

AUTO = Path(__file__).resolve().parent
ROOT = AUTO.parent
CODE = ROOT / "code"
RUNS = ROOT / "runs"
CFGDIR = AUTO / "phase10_configs"
PY = sys.executable
RESULT = AUTO / "phase10_ablations_result.json"

# T11.0b: the statistics live in ONE place. This driver used to carry its own
# `mean_std` and its own `2 x std` reading rule; both are now delegated to
# code/seed_stats.py so that this file, run_phase9_matrix.py and the results
# exporter can never disagree about what "significant" means. See that module's
# docstring for why the k x std rule was abandoned rather than re-tuned.
sys.path.insert(0, str(CODE))
import seed_stats as _seed_stats                     # noqa: E402

# Arms were pre-registered at n=3. T11.0b extended `dense_routing` and
# `routing_supervision` to seeds 45-46, because those two were the only arms with
# a significant result and both sat EXACTLY at the 3-vs-5 resolution floor
# (p = 0.0179 = 1/56), so neither could survive Bonferroni across six arms. Five
# seeds drops the floor to 1/252 = 0.0040. Every other arm remains at n=3, so the
# seed set is now per-arm and is DISCOVERED FROM DISK rather than assumed -- see
# `arm_seeds()`. CANDIDATE_SEEDS is the frozen set a run may draw from at all.
CANDIDATE_SEEDS = (42, 43, 44, 45, 46)
SEEDS = (42, 43, 44)
BASELINE_SEEDS_FULL = (42, 43, 44, 45, 46)
ARCH = "more"
BASELINE_GLOB = "phaseB_more_seed{seed}__*"

# task id -> (arm name, section, key, value, what the arm asks)
# Ordered by scientific value, because a batch killed halfway should have answered
# the important questions first. Dense-vs-sparse and adaptive-vs-fixed directly
# interrogate MoRE's two claims; noise and block count are secondary.
ARMS = (
    ("T10.F", "dense_routing", "model", "routing_mode", "dense_blend",
     "Does Top-1 sparse dispatch cost anything against evaluate-all-and-blend? "
     "If dense is no better, sparsity is free; if dense is much better, the "
     "canonical architecture is paying for sparsity it does not need."),
    ("T10.B", "fixed_depth", "model", "fixed_depth", True,
     "Does learned adaptive halting beat a fixed depth? T9.1 measured a depth "
     "allocation error of ~1.0 step, so the depths ACT chooses may be doing "
     "nothing that a constant could not."),
    ("T10.E", "routing_supervision", "loss_weights", "step_routing", 0.5,
     "Does supervising the router with the oracle expert index improve the task "
     "metric, or only the routing metric? Canonical is 0.0 (T8.0b)."),
    ("T10.D", "router_noise", "model", "router_noise", "fixed_annealed",
     "Does exploration noise in the router change the partition or the loss? "
     "Canonical is none (CLAUDE.md 2). `fixed_annealed` and not `trainable`: the "
     "L2-on-trainable-noise-scale mechanism is explicitly not a proven fix, so an "
     "ablation of it would confound noise with that unproven machinery."),
    ("T10.C", "two_blocks", "model", "num_blocks", 2,
     "Does a second recursive block help? Varies ONLY num_blocks -- the old "
     "1-vs-2-block headline moved three factors at once and is uninterpretable."),
    ("T10.J", "ponder_cost_low", "loss_weights", "halting", 0.0001,
     "T10.B found fixed depth BETTER than learned halting at 2.10x seed std, but "
     "[PRE-REGISTRATION -- LEFT VERBATIM. The 2.10x was a population-std artifact "
     "(T11.0a: mean_std divided by n, not n-1); corrected it is 1.88x, and under "
     "the exact randomization test that replaced the k x std rule entirely "
     "(T11.0b, code/seed_stats.py) T10.B is p = 0.125 -- NOT significant. The "
     "verdict this text rests on is withdrawn; the arm's own rationale below "
     "still holds, because 'is the coefficient mis-tuned' is worth answering "
     "whether or not fixed depth won.] "
     "that arm also raised the compute budget 2.05 -> 7.0 steps, so it cannot "
     "separate 'adaptive computation does not work' from 'our ponder cost is "
     "mis-weighted'. This arm reduces the ponder coefficient 10x (0.001 -> 0.0001) "
     "and changes NOTHING else, so halting stays adaptive and differentiable while "
     "the task term is allowed to dominate. Reading: if avg_recursion_steps rises "
     "from ~2.05 toward max_depth and val/task_loss moves to the fixed_depth number "
     "(0.062822), the ACT objective was mis-weighted and adaptive computation is "
     "viable at lower compute than fixed depth. If depth rises but loss does not "
     "follow, the halt head is not learning anything useful and Outcome C stands "
     "for the depth half of the architecture. 0.0001 and not 0.0: at zero there is "
     "no pressure to exit at all, which degenerates into T10.B and re-measures a "
     "number we already have."),
)

SCALARS = (
    "val/task_loss", "best_val_loss", "train/task_loss",
    "val/routing_accuracy", "val/routing_hungarian_accuracy", "val/routing_ami",
    "val/routing_purity", "val/routing_matched_macro_recall",
    "train/expert_load_entropy_normalized", "train/entropy_term_normalized",
    "train/avg_recursion_steps", "depth/allocation_error_abs",
    "halt/early_exit_rate", "halt/forced_exit_rate",
    "dispatch/evals_per_token", "diag/max_pairwise_cosine_sim",
    "perf/throughput_tokens_sec", "total_params",
    # T11.1 audit item 7: depth measured on the VALIDATION pass, recovered from
    # each run's best-val checkpoint by code/eval_val_depth.py and merged in
    # metrics_of() below. Every other depth key here is a TRAINING-time average
    # over the final epoch; these are the held-out numbers a depth claim needs.
    # Absent (-> N/A) for any run whose sidecar has not been computed, and for
    # the fixed_depth arm, which allocates nothing.
    "val_offline/avg_recursion_steps", "val_offline/depth_allocation_error_abs",
    "val_offline/depth_allocation_error_rel", "val_offline/early_exit_rate",
)

PRIMARY = "val/task_loss"

# Files whose contents define the COMPUTATION -- the model, the loss assembly, the
# data pipeline. The baseline arm is reused from T9.1 rather than re-run, so if any
# of these changed after those runs finished, the baseline and the ablations were
# trained by different code and no comparison between them means anything (the
# T10.H lesson). Checked by mtime against the baseline runs.
MODEL_SOURCES = ("more/model.py", "more/engine.py", "more/data.py")

# Plumbing: defaults, labels, run directories, argument parsing. These are NOT
# mtime-checked, because a label-only edit (T10.C added a `num_blocks_2` variant
# tag to resolve_variant) cannot change what trains, and refusing on it would make
# the guard cry wolf. What matters about them is checked on the ARTEFACT instead:
# verify_baseline_state() proves today's config.json still resolves to the same
# experiment the baseline runs recorded. All of them are fingerprinted for the
# prospective mid-phase check.
CODE_SOURCES = MODEL_SOURCES + ("more/config.py", "more/run_context.py",
                                "more/cli.py")
FINGERPRINT = AUTO / "phase10_code_fingerprint.json"

# Fields that change the architecture but exist ONLY as a bare literal default in
# source (`mc.get("fixed_depth", False)` at engine.py:210, config.py:321), so they
# appear in no config file and in no provenance block. Same defect class as the
# 0.5 family_cls literal that T8.3 surfaced. Filled in on both sides of every
# comparison here so `absent` and `the canonical value` are one thing: without
# this, the fixed_depth arm is the only one whose diff is an ADDITION, and every
# OTHER arm reports a spurious second difference against the baseline.
IMPLICIT_DEFAULTS = {"model.fixed_depth": False}


sys.path.insert(0, str(CODE))
from more.config import load_config, apply_architecture  # noqa: E402
from more.model import ROUTING_MODES, ROUTER_NOISE_MODES  # noqa: E402

# Enum-valued fields, checked at CONFIG-GENERATION time. The first draft of this
# driver set router_noise = "gaussian", which does not exist -- the legal values are
# ("none", "fixed_annealed", "trainable"). Nothing caught it until the model
# constructor raised, and resolve_variant had happily labelled the arm
# `router_noise_gaussian`, i.e. the label layer will name a value that cannot run.
# A 1-epoch smoke test found it; on the real batch it would have surfaced after two
# completed arms. Validated here against the model's own tuples so the two cannot
# drift apart.
LEGAL_VALUES = {
    "model.routing_mode": ROUTING_MODES,
    "model.router_noise": ROUTER_NOISE_MODES,
}


def flatten(d: dict, prefix: str = "") -> dict:
    """`{'a': {'b': 1}}` -> `{'a.b': 1}`, so a diff can name the exact leaf."""
    out: dict = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        else:
            out[key] = v
    return out


def jload(path: Path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def materialized_base() -> dict:
    """
    config.json with every canonical default made EXPLICIT, MoRE already stamped.

    Why materialize instead of editing config.json directly: three of the five
    ablated fields (routing_mode, router_noise, fixed_depth) are absent from
    config.json and only appear via `setdefault` inside load_config_defaults. A
    diff against the raw file would therefore show an ADDITION for those arms and
    a value CHANGE for the other two -- two different kinds of evidence. Running
    the same loader the trainer runs gives one base in which all five fields are
    present, so every arm is provably a one-value change from a stated value.

    `logging.run_name` is dropped: the run label is the CLI's job, and the dense
    arm's label is what enforce_routing_mode gates on.
    """
    cfg = load_config(str(CODE / "config.json"))
    apply_architecture(cfg, ARCH)
    cfg.pop("_README", None)
    cfg.get("logging", {}).pop("run_name", None)
    for leaf, val in IMPLICIT_DEFAULTS.items():
        sec, key = leaf.split(".", 1)
        cfg.setdefault(sec, {}).setdefault(key, val)
    return cfg


def write_arm_configs() -> dict:
    """
    Generate one full config file per arm and PROVE the one-field claim.

    The assertion is the point of this function. `arm.json` is written, read back
    through the same flattener as the base, and the set of disagreeing leaves must
    be exactly {section.key}. If a future edit to config.json or to
    load_config_defaults makes an arm differ in two places, this raises before any
    GPU time is spent, instead of producing a two-factor "ablation" that looks
    like a one-factor one in the results table.
    """
    base = materialized_base()
    CFGDIR.mkdir(parents=True, exist_ok=True)
    base["_README"] = [
        "GENERATED by automated/phase10_ablations.py -- do not hand-edit.",
        "code/config.json with every canonical default made explicit and the MoRE",
        "architecture already stamped. Each phase10_configs/<arm>.json is this file",
        "with exactly ONE leaf changed; that diff is asserted before any run.",
    ]
    with open(CFGDIR / "_base_more.json", "w", encoding="utf-8") as fh:
        json.dump(base, fh, indent=2)
    flat_base = flatten({k: v for k, v in base.items() if k not in DOC_KEYS})

    paths, notes = {}, {}
    for tid, name, section, key, value, question in ARMS:
        leaf = f"{section}.{key}"
        if leaf not in flat_base:
            raise SystemExit(
                f"[REFUSED] arm {name!r} varies {leaf!r}, which is not present in "
                "the materialized base config. Either the field moved or the "
                "loader stopped defaulting it; in both cases the arm would not be "
                "a one-field change from a known value."
            )
        if flat_base[leaf] == value:
            raise SystemExit(
                f"[REFUSED] arm {name!r} sets {leaf} = {value!r}, which is already "
                "the canonical value. The arm would be a duplicate of the baseline "
                "and its result would be reported as an ablation effect of zero."
            )
        if leaf in LEGAL_VALUES and value not in LEGAL_VALUES[leaf]:
            raise SystemExit(
                f"[REFUSED] arm {name!r} sets {leaf} = {value!r}, which is not one "
                f"of {LEGAL_VALUES[leaf]}. The model constructor would raise, but "
                "only after the data had loaded -- and resolve_variant would "
                "already have produced a variant label naming a value that cannot "
                "run."
            )
        arm = json.loads(json.dumps(base))
        arm[section][key] = value
        flat_arm = flatten({k: v for k, v in arm.items() if k not in DOC_KEYS})
        diff = {k for k in set(flat_base) | set(flat_arm)
                if flat_base.get(k, "<absent>") != flat_arm.get(k, "<absent>")}
        if diff != {leaf}:
            raise SystemExit(
                f"[REFUSED] arm {name!r} was meant to differ from canonical MoRE "
                f"in exactly {leaf!r}, but the flattened diff is {sorted(diff)}."
            )
        p = CFGDIR / f"{name}.json"
        arm["_README"] = [
            f"{tid} ABLATION ARM: {name}. GENERATED -- do not hand-edit.",
            f"Sole deviation from canonical MoRE: {leaf} = {flat_base[leaf]!r} -> "
            f"{value!r}.",
            f"Question: {question}",
            "Runs WITHOUT --experiment_group, so it is stamped `exploratory` and "
            "excluded from the headline table. resolve_variant labels it.",
        ]
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(arm, fh, indent=2)
        paths[name] = p
        notes[name] = f"{leaf}: {flat_base[leaf]!r} -> {value!r}"
        print(f"[cfg] {name:<20s} {notes[name]}")
    return paths, notes


def baseline_dirs(seeds=SEEDS) -> dict:
    """
    The T9.1 canonical MoRE run directory per seed. Refuses anything not admitted.

    An ablation is only interpretable against a baseline that was itself canonical,
    so `experiment_group` must be `canonical_phase_b` and `variant` must be
    `canonical`. The six T9.1a runs that were silently stamped `exploratory` are in
    archive/ and cannot be picked up here, but the check is kept because the failure
    it guards against -- reading an exploratory run as the baseline -- is exactly
    what cost 1.6 h in T9.1a.
    """
    out = {}
    for s in seeds:
        hits = sorted(RUNS.glob(BASELINE_GLOB.format(seed=s)))
        hits = [h for h in hits if (h / "metrics.json").exists()]
        if len(hits) != 1:
            raise SystemExit(
                f"[REFUSED] expected exactly 1 completed canonical MoRE run for "
                f"seed {s}, found {len(hits)}: {[h.name for h in hits]}"
            )
        prov = jload(hits[0] / "resolved_config.json").get("provenance", {})
        if prov.get("experiment_group") != "canonical_phase_b":
            raise SystemExit(
                f"[REFUSED] baseline {hits[0].name} is stamped "
                f"{prov.get('experiment_group')!r}, not canonical_phase_b."
            )
        if prov.get("variant") != "canonical":
            raise SystemExit(
                f"[REFUSED] baseline {hits[0].name} has variant "
                f"{prov.get('variant')!r}, not canonical."
            )
        out[s] = hits[0]
    return out


def verify_baseline_state(bdirs: dict) -> dict:
    """
    Refuse if the baseline arm and the ablation arms cannot share one code state.

    Three independent checks, because a reused baseline needs both directions:

      RETROACTIVE, SOURCE -- no file in MODEL_SOURCES may be newer than the oldest
      baseline run's metrics.json. If it is, that source was edited after the
      baseline was trained, so the ablations would run code the baseline never saw.

      RETROACTIVE, ARTEFACT -- today's config.json, resolved through the same
      loader the trainer uses, must equal what each baseline run actually recorded
      in resolved_config.json (seed- and label-blind). This is the check that
      matters for the plumbing files: it proves the CONFIGURATION is unchanged
      regardless of how the loader was edited, which a source hash cannot show.

      PROSPECTIVE -- a sha256 fingerprint of every source file is pinned on the
      first launch and re-checked on every later one, so an edit landing between
      two ablation arms is caught even though it postdates the baseline check.
    """
    import hashlib

    fp, newest = {}, 0.0
    for rel in CODE_SOURCES:
        p = CODE / rel
        fp[rel] = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
        if rel in MODEL_SOURCES:
            newest = max(newest, p.stat().st_mtime)

    oldest_run = min((d / "metrics.json").stat().st_mtime for d in bdirs.values())
    if newest > oldest_run:
        stale = [rel for rel in MODEL_SOURCES
                 if (CODE / rel).stat().st_mtime > oldest_run]
        raise SystemExit(
            "[REFUSED] model/training source is NEWER than the reused T9.1 "
            f"baseline runs: {stale}. The baseline arm and the ablation arms would "
            "straddle a code change, which is the one thing an ablation may never "
            "do (T10.H). Either re-run the MoRE baseline on the current code, or "
            "revert the change."
        )

    want = sci_flat(materialized_base())
    for s, d in sorted(bdirs.items()):
        got = sci_config(d)
        diff = sorted(k for k in set(want) | set(got)
                      if want.get(k, "<absent>") != got.get(k, "<absent>"))
        if diff:
            raise SystemExit(
                f"[REFUSED] today's config.json no longer resolves to the "
                f"experiment baseline seed {s} ({d.name}) recorded. Differs on "
                + "; ".join(f"{k}: run {got.get(k, '<absent>')!r} vs now "
                            f"{want.get(k, '<absent>')!r}" for k in diff)
                + ". Every ablation arm is generated from config.json, so each "
                  "would differ from the baseline in this field TOO, and no arm "
                  "would be a one-field ablation."
            )

    if FINGERPRINT.exists():
        old = jload(FINGERPRINT)
        drift = {k for k in fp if old.get("sha", {}).get(k) != fp[k]}
        if drift:
            raise SystemExit(
                f"[REFUSED] source changed mid-Phase-10: {sorted(drift)}. Arms "
                "already run and arms not yet run would not share a code state. "
                f"Delete {FINGERPRINT.name} only if you intend to restart the "
                "whole phase, including the baseline."
            )
    else:
        with open(FINGERPRINT, "w", encoding="utf-8") as fh:
            json.dump({"sha": fp, "pinned_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                       "baseline_runs": {str(k): v.name
                                         for k, v in bdirs.items()}}, fh, indent=2)
        print(f"[state] pinned code fingerprint over {len(fp)} source files")
    return fp


def arm_label(name: str) -> str:
    """
    The --run_name for an arm, WITHOUT the seed (stamp_seed_into_run_name adds it).

    The dense arm's label must contain `dense_routing_ablation` or
    enforce_routing_mode refuses the run outright -- the gate is on the label, not
    on a boolean, precisely so a dense run can never be tabled as if it were MoE.
    """
    if name == "dense_routing":
        return "t10_more_dense_routing_ablation"
    return f"t10_more_{name}"


def find_arm_dir(name: str, seed: int) -> Path | None:
    """The completed run directory for one arm/seed, or None if it has not run."""
    hits = [h for h in sorted(RUNS.glob(f"{arm_label(name)}_seed{seed}__*"))
            if (h / "metrics.json").exists()]
    return hits[-1] if hits else None


def launch(name: str, cfg_path: Path, seed: int) -> tuple[bool, float]:
    """
    Train one arm/seed.

    NO --experiment_group: an ablation is not the frozen spec, so it must be
    stamped `exploratory` and excluded from the headline table. Passing the flag
    here would make the Gate 0 guard refuse the run anyway (routing_mode,
    router_noise, num_blocks and the loss weights are all enforced fields), which
    is the guard working correctly -- but relying on a refusal to keep an ablation
    out of the headline would be relying on an error path for correctness.

    Everything that defines the experiment is in `cfg_path`, so the only CLI
    arguments are the architecture, the seed and the label.
    """
    cmd = [PY, "train.py", "--architecture", ARCH, "--config", str(cfg_path),
           "--seed", str(seed), "--run_name", arm_label(name)]
    t0 = time.time()
    print(f"\n[run] {name} seed={seed}  ({' '.join(cmd[1:])})", flush=True)
    rc = subprocess.run(cmd, cwd=str(CODE)).returncode
    dt = (time.time() - t0) / 60.0
    print(f"[run] {name} seed={seed} exit={rc} in {dt:.1f} min", flush=True)
    return rc == 0, dt


# --seed is written into both of these, and config_hash() hashes them, so two
# seeds of one arm legitimately differ here and nowhere else (T9.1b). Excluded
# from every config comparison below.
SEED_KEYS = ("data.subset_seed", "training.subset_seed")

# Label fields, not experiment fields. `variant` is the deviation NAME, so of
# course it differs between an arm and the baseline -- that is the arm working.
# enforce_routing_mode also copies it into model.variant for the dense arm.
LABEL_KEYS = ("variant", "model.variant", "architecture")

# Prose, not configuration. Dropped from every comparison: the baseline runs carry
# config.json's own _README, and each arm file below carries an arm-specific one
# saying what was changed and why -- which is the point of writing the arm config
# to disk at all, and must not read as a config difference.
DOC_KEYS = ("_README",)


def sci_config(run_dir: Path) -> dict:
    """resolved_config.json reduced to the leaves that define the experiment."""
    cfg = jload(run_dir / "resolved_config.json")
    return sci_flat(cfg)


def sci_flat(cfg: dict) -> dict:
    """Shared reduction, so a run directory and a live config are read the same
    way. Drops provenance/logging (not experiment fields), drops the seed and the
    label fields, and fills the implicit defaults that live only in source."""
    flat = flatten({k: v for k, v in cfg.items()
                    if k not in ("provenance", "logging")})
    drop = set(SEED_KEYS) | set(LABEL_KEYS) | set(DOC_KEYS)
    out = {k: v for k, v in flat.items() if k not in drop}
    for leaf, val in IMPLICIT_DEFAULTS.items():
        out.setdefault(leaf, val)
    return out


def one_field_check(arm_dir: Path, base_dir: Path, leaf: str) -> list:
    """
    The arm's TRAINED config must differ from the baseline's in exactly `leaf`.

    write_arm_configs() proves the intent on the config files; this proves the
    outcome on what actually trained. The two are not the same claim: between the
    file and the model sit load_config_defaults, apply_architecture,
    resolve_overrides and enforce_routing_mode, any of which could stamp a second
    field. This is the check that makes the ablation table's one-factor claim true.
    """
    a, b = sci_config(arm_dir), sci_config(base_dir)
    diff = {k for k in set(a) | set(b)
            if a.get(k, "<absent>") != b.get(k, "<absent>")}
    out = []
    if leaf not in diff:
        out.append(f"{arm_dir.name}: {leaf} does NOT differ from the baseline -- "
                   f"the arm trained the canonical value {b.get(leaf)!r}")
    for k in sorted(diff - {leaf}):
        out.append(f"{arm_dir.name}: unintended second difference {k}: "
                   f"baseline {b.get(k, '<absent>')!r} vs arm {a.get(k, '<absent>')!r}")
    return out


def mean_std(values: list) -> dict | None:
    """Delegates to `code/seed_stats.mean_std` -- see the import note at the top.

    Kept as a module-level name because `collect()` and `report()` call it in a
    dozen places. The local reimplementation was removed at T11.0b so that this
    driver, `run_phase9_matrix.py` and the exporter cannot drift apart on what
    "mean +- std" means; the previous copy was correct (n-1) but a copy.
    """
    return _seed_stats.mean_std(values)


def fmt(agg: dict | None, prec: int = 6, width: int = 20) -> str:
    if agg is None:
        return "N/A".rjust(width)
    if agg.get("std") is None:
        return f"{agg['mean']:.{prec}f} +- N/A(n=1)".rjust(width)
    return f"{agg['mean']:.{prec}f} +- {agg['std']:.{prec}f}".rjust(width)


def rule4(arm_vals: list, base_vals: list) -> dict:
    """
    THE READING RULE, REPLACED AT T11.0b. Exact two-sided randomization test.

    Was: "a gap counts only if |arm - baseline| >= 2x the LARGEST available seed
    std." That rule is wrong in a way that mattered, and the error ran toward
    silence rather than toward over-claiming. `max(std_a, std_b)` is the spread of
    ONE arm's seed values; a difference of MEANS has to be compared against the
    standard error OF THE DIFFERENCE, sqrt(s_a^2/n_a + s_b^2/n_b), which at
    n = 3-vs-5 is roughly sqrt(n) smaller. Two arms of this phase were being
    reported as "inside seed noise" while their exact p-values were 0.0179:
    `dense_routing` and `routing_supervision`. See the module docstring of
    `code/seed_stats.py` for the full derivation and the numbers.

    The replacement enumerates every regrouping of the pooled per-seed values,
    so there is no threshold to justify, no normality assumption and no pooled-std
    yardstick to choose. It also removes the old function's most awkward
    compromise -- borrowing the 5-seed baseline std as the noise estimate while
    comparing means at n=3. A permutation test needs no variance estimate, so the
    arm and the baseline are now compared at whatever matched seed set both
    actually have on disk, and nothing is borrowed.

    READ THE FLOOR, NOT JUST THE p. `min_p = 1/C(n_a+n_b, n_a)` is 0.0179 at
    3-vs-5 and 0.0040 at 5-vs-5. An arm whose p EQUALS its floor is as extreme as
    its seed count can express; that is not the same claim as "significant", and
    it will not survive a Bonferroni correction across this phase's six arms
    (alpha = 0.05/6 = 0.0083). The verdict string says AT-FLOOR when that happens.
    """
    res = _seed_stats.perm_test(arm_vals, base_vals)
    if res is None:
        return {"verdict": "N/A -- fewer than 2 seeds in one arm, no test exists"}
    gap = res["gap"]
    d = "WORSE than canonical MoRE" if gap > 0 else "BETTER than canonical MoRE"
    at_floor = res["p_value"] <= res["min_p"] + 1e-12
    # Bonferroni over this phase's six arms. Stated explicitly for every
    # significant arm, not only for the AT-FLOOR ones: after seeds 45-46 landed,
    # dense_routing and routing_supervision moved from 3-vs-5 (p = 0.0179, at the
    # floor, hopeless against 0.0083) to 5-vs-5 (p = 0.0079), which clears the
    # corrected threshold by 0.0004. A margin that thin must be printed, not
    # inferred by a reader comparing two numbers in different paragraphs.
    bonf = 0.05 / len(ARMS)
    if res["p_value"] < 0.05:
        verdict = (f"SIGNIFICANT (p={res['p_value']:.4f}"
                   f"{', AT FLOOR' if at_floor else ''}, d={res['cohens_d']:+.2f})"
                   f" -- {d}")
        verdict += (f"; survives Bonferroni at {len(ARMS)} arms "
                    f"(alpha={bonf:.4f})" if res["p_value"] < bonf
                    else f"; fails Bonferroni at {len(ARMS)} arms "
                         f"(alpha={bonf:.4f})")
    else:
        verdict = (f"not significant (p={res['p_value']:.4f}, floor "
                   f"{res['min_p']:.4f}) -- supports no claim")
    return {"gap": gap, "p_value": res["p_value"], "min_p": res["min_p"],
            "n_perms": res["n_perms"], "cohens_d": res["cohens_d"],
            "se_diff": res["se_diff"], "gap_over_std": res["gap_over_std"],
            "n_arm": len(arm_vals), "n_base": len(base_vals),
            "test": "exact two-sided randomization on the difference of means",
            "verdict": verdict}



def arm_seeds(name: str) -> list:
    """The seeds of `name` that actually have a completed run directory.

    T11.0b: the seed set is per-arm, because `dense_routing` and
    `routing_supervision` were extended to 45-46 and the other four arms were
    not. Discovering it from disk rather than from a constant means a partially
    launched extension reports its true n instead of silently comparing an n=4
    arm against a 5-seed baseline.
    """
    return [s for s in CANDIDATE_SEEDS if find_arm_dir(name, s) is not None]


def collect() -> tuple[dict, list]:
    """Aggregate every arm that has completed. Returns (payload, problems)."""
    problems: list = []
    bdirs_full = baseline_dirs(BASELINE_SEEDS_FULL)

    def metrics_of(d: Path) -> dict:
        m = jload(d / "metrics.json")
        prov = jload(d / "resolved_config.json").get("provenance", {})
        m["total_params"] = prov.get("total_params")
        # T11.1 audit item 7: fold in the offline validation-pass depth sidecar if
        # code/eval_val_depth.py has been run for this directory. Keys keep their
        # `val_offline/` prefix, so a reader cannot mistake a best-checkpoint depth
        # number for one this run recorded while training. Missing sidecar -> the
        # keys are simply absent and mean_std reports N/A.
        side = d / "val_depth_offline.json"
        if side.exists():
            sc = jload(side)
            m.update({k: v for k, v in sc.items() if k.startswith("val_offline/")})
        return m

    base_full_m = {s: metrics_of(d) for s, d in bdirs_full.items()}
    base_agg5 = {k: mean_std([base_full_m[s].get(k) for s in BASELINE_SEEDS_FULL])
                 for k in SCALARS}
    # The 3-seed baseline aggregate is retained for continuity with the pre-T11.0b
    # tables and is what the secondary-metric columns are printed against. It no
    # longer feeds any verdict: the exact test below uses all five baseline seeds.
    base_agg = {k: mean_std([base_full_m[s].get(k) for s in SEEDS])
                for k in SCALARS}

    arms: dict = {}
    for tid, name, section, key, value, question in ARMS:
        leaf = f"{section}.{key}"
        seeds_here = arm_seeds(name)
        if not seeds_here:
            continue
        per_seed, dirs = {}, {}
        for s in seeds_here:
            d = find_arm_dir(name, s)
            problems.extend(one_field_check(d, bdirs_full[s], leaf))
            per_seed[s] = metrics_of(d)
            dirs[s] = d.name
        agg = {k: mean_std([per_seed[s].get(k) for s in seeds_here])
               for k in SCALARS}

        # The two vectors the exact test consumes: this arm's per-seed primary
        # values, and the baseline's over the FULL frozen seed set.
        #
        # WHY THE BASELINE IS NOT RESTRICTED TO THE ARM'S SEEDS. A randomization
        # test does not require n_a == n_b -- it enumerates the C(n_a+n_b, n_a)
        # regroupings of whatever two groups it is given. Restricting the baseline
        # to the arm's three seeds was tried and discarded: it throws away two
        # perfectly valid baseline draws and collapses the resolution floor from
        # 1/56 = 0.0179 to 1/20 = 0.0500, at which NO arm can ever reach
        # alpha = 0.05. The old `2 x std` rule needed matched n because its
        # yardstick was a single arm's std; this test needs no variance estimate,
        # so the constraint that motivated matching is gone with it.
        arm_vals = [per_seed[s].get(PRIMARY) for s in seeds_here]
        arm_vals = [v for v in arm_vals if isinstance(v, (int, float))]
        base_vals = [base_full_m[s].get(PRIMARY) for s in BASELINE_SEEDS_FULL]
        base_vals = [v for v in base_vals if isinstance(v, (int, float))]
        primary = rule4(arm_vals, base_vals)
        primary["arm_seeds"] = seeds_here
        primary["baseline_seeds"] = list(BASELINE_SEEDS_FULL)

        if len(seeds_here) < len(CANDIDATE_SEEDS):
            problems.append(
                f"{name}: n={len(seeds_here)} of {len(CANDIDATE_SEEDS)} frozen "
                f"seeds ({seeds_here}). The exact test's resolution floor is "
                f"{primary.get('min_p', float('nan')):.4f}, so this arm cannot "
                f"report a p below that value no matter how large the effect. "
                f"This is a design limit, not a defect -- but a p AT the floor "
                f"must not be presented as clearing a Bonferroni correction.")

        arms[name] = {"task": tid, "field": leaf, "value": value,
                      "question": question, "seeds": seeds_here,
                      "dirs": dirs, "agg": agg, "primary": primary}

    # Absence is never a pass. Without this, a report run before anything has
    # trained prints the clean verdict "every completed arm differs in exactly one
    # field" -- true and worthless, because zero arms were checked. Same defect
    # class as the T9.1b assertion that could not fail.
    if not arms:
        problems.append("no ablation arm has completed -- nothing was compared, so "
                        "no verdict below is evidence of anything")

    return {"baseline_dirs": {str(s): d.name for s, d in bdirs_full.items()},
            "baseline_agg_3seed": base_agg, "baseline_agg_5seed": base_agg5,
            "arms": arms}, problems



SECONDARY = (
    ("routing (permutation-invariant; expert indices carry no identity)",
     ("val/routing_hungarian_accuracy", "val/routing_ami", "val/routing_purity",
      "val/routing_matched_macro_recall", "val/routing_accuracy")),
    ("depth allocation (NOT compute efficiency -- no FLOPs in these numbers)",
     ("train/avg_recursion_steps", "depth/allocation_error_abs",
      "halt/early_exit_rate", "halt/forced_exit_rate")),
    ("depth allocation ON HELD-OUT DATA (offline, from each run's best-val "
     "checkpoint -- NOT the last-epoch model the primary metric comes from)",
     ("val_offline/avg_recursion_steps", "val_offline/depth_allocation_error_abs",
      "val_offline/depth_allocation_error_rel", "val_offline/early_exit_rate")),
    ("balance / differentiation / cost",
     ("train/expert_load_entropy_normalized", "train/entropy_term_normalized",
      "diag/max_pairwise_cosine_sim", "dispatch/evals_per_token",
      "perf/throughput_tokens_sec", "total_params")),
)


def report(data: dict, problems: list) -> None:
    arms, base = data["arms"], data["baseline_agg_3seed"]
    print("\n" + "=" * 78)
    print("T10 -- PHASE 10 ABLATIONS (MoRE, one field at a time)")
    print("=" * 78)
    print(f"baseline arm : canonical MoRE from T9.1, seeds {list(SEEDS)} "
          f"(reused, not re-run)")
    print(f"arms         : per-arm seed set, experiment_group=exploratory")
    print(f"reading rule : exact two-sided randomization test (code/seed_stats.py);"
          f" alpha = 0.05, floor 1/C(n_a+n_b, n_a) reported per arm\n")

    print(f"PRIMARY METRIC: {PRIMARY}")
    print(f"{'arm':<22s}{'field changed':<30s}{'val/task_loss':>21s}   verdict")
    b = base[PRIMARY]
    print(f"{'canonical MoRE':<22s}{'--':<30s}{fmt(b, 6, 21)}   baseline "
          f"(5-seed: {fmt(data['baseline_agg_5seed'][PRIMARY], 6, 1).strip()})")
    for name, a in arms.items():
        chg = f"{a['field']} = {a['value']}"
        print(f"{name:<22s}{chg:<30s}{fmt(a['agg'][PRIMARY], 6, 21)}   "
              f"n={len(a['seeds'])}  {a['primary']['verdict']}")

    for title, keys in SECONDARY:
        print(f"\n{title}")
        print(f"{'metric':<42s}{'canonical MoRE':>21s}" +
              "".join(f"{n[:19]:>21s}" for n in arms))
        for k in keys:
            row = f"{k:<42s}{fmt(base.get(k), 4, 21)}"
            for a in arms.values():
                row += fmt(a["agg"].get(k), 4, 21)
            print(row)

    print("\nWHAT EACH ARM ASKED")
    for name, a in arms.items():
        print(f"  [{a['task']}] {name}: {a['question']}")

    print("\n" + "-" * 78)
    if problems:
        print(f"PROBLEMS ({len(problems)}) -- the table above is NOT admissible "
              f"until these are resolved:")
        for p in problems:
            print(f"  * {p}")
    else:
        print("No problems. Every completed arm differs from the canonical MoRE "
              "baseline in exactly one config leaf, verified on the config that "
              "actually trained, and all arms share one code state.")

    with open(RESULT, "w", encoding="utf-8") as fh:
        json.dump({"generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
                   "candidate_seeds": list(CANDIDATE_SEEDS),
                   "seeds": list(SEEDS),
                   "primary_metric": PRIMARY,
                   "reading_rule": ("exact two-sided randomization test on the "
                                    "difference of means, code/seed_stats.py; "
                                    "alpha = 0.05; report min_p alongside p"),
                   "reading_rule_superseded": ("|arm - baseline| >= 2 * max(seed "
                                               "stds) -- withdrawn at T11.0b, "
                                               "wrong denominator"),
                   "problems": problems, **data}, fh, indent=2)
    print(f"\nWrote {RESULT.relative_to(ROOT)}")


def main(argv=None) -> int:
    import argparse
    p = argparse.ArgumentParser(description="T10 Phase 10 MoRE ablations")
    p.add_argument("--report-only", action="store_true",
                   help="Aggregate what is already on disk. Launches nothing.")
    p.add_argument("--configs-only", action="store_true",
                   help="Generate and verify the per-arm config files, then stop.")
    p.add_argument("--arms", type=str, default=None,
                   help="Comma-separated arm names. Default: all, in the priority "
                        "order declared in ARMS (dense and fixed_depth first, "
                        "because they test MoRE's two actual claims).")
    p.add_argument("--seeds", type=str, default=None,
                   help=f"Comma-separated seeds. Default {','.join(map(str, SEEDS))}.")
    args = p.parse_args(argv)

    cfgs, notes = write_arm_configs()
    if args.configs_only:
        print(f"\n{len(cfgs)} arm configs written to "
              f"{CFGDIR.relative_to(ROOT)}; one-field diff verified for each.")
        return 0

    order = [a[1] for a in ARMS]
    if args.arms:
        want = [s.strip() for s in args.arms.split(",") if s.strip()]
        unknown = [w for w in want if w not in order]
        if unknown:
            print(f"[REFUSED] unknown arm(s) {unknown}; known: {order}")
            return 2
        order = want
    seeds = ([int(s) for s in args.seeds.split(",")] if args.seeds else list(SEEDS))

    if not args.report_only:
        bdirs = baseline_dirs(BASELINE_SEEDS_FULL)
        verify_baseline_state(bdirs)
        print(f"[state] baseline arm = 5 canonical MoRE runs, code state verified")

        todo = [(n, s) for n in order for s in seeds
                if find_arm_dir(n, s) is None]
        print(f"[plan] {len(todo)} run(s) to launch, "
              f"{len(order) * len(seeds) - len(todo)} already on disk")
        t0 = time.time()
        for i, (n, s) in enumerate(todo, 1):
            print(f"\n===== {i}/{len(todo)}  arm={n}  seed={s}  "
                  f"elapsed={(time.time()-t0)/60:.0f} min =====", flush=True)
            ok, _dt = launch(n, cfgs[n], s)
            if not ok:
                print(f"[STOP] {n} seed {s} failed. Not launching the rest: a "
                      f"partial arm cannot be compared, and CLAUDE.md 7 says stop "
                      f"and report on failure rather than continue.")
                break
        print(f"\n[plan] launch phase done in {(time.time()-t0)/60:.1f} min")

    data, problems = collect()
    report(data, problems)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
