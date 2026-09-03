"""
run_context.py — per-run output directories, resolved-config provenance, and the
Gate 0 proxy guard.

Phase 0 of plan.md requires three things from this module:

  §2.2  Stop writing fixed global filenames (best_model.pt, results.tsv,
        final_run_metrics.json, *_results.csv).  Every run owns
        runs/<experiment_id>/ containing:

            config.json           — the config exactly as loaded from disk
            resolved_config.json  — after CLI overrides, with provenance block
            metrics.json          — final metric dump
            results.tsv           — per-epoch table
            checkpoint.pt         — best-val checkpoint
            stdout.log            — tee'd stdout + stderr

  §2.3  Canonical experiments must be explicitly specified, never silently
        launched with proxy settings.

  Gate 0 — "final experiments cannot silently use proxy settings."  Enforced by
        `assert_not_silent_proxy()`: a run may only claim
        experiment_group == "canonical_phase_b" if every field that
        canonical_spec.json pins is both set and matching.  Any run that does
        not claim that group is stamped with a non-canonical group so the
        Phase-9 exporter can exclude it mechanically.

Scope note: global seeding lives in `seeding.py` (T6.1) and is applied by
`engine.train()`, which then extends the provenance block written here with
`resolved_seed` / `seed_source` / the determinism fields.  This module owns only
`provenance.seed` -- the DECLARED seed, null when `--seed` was not passed --
because that is the field the Gate 0 guard tests; it must never be defaulted.
Still outstanding here: the full W&B provenance field list (§8.7).
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys

# The canonical routing path, imported rather than restated. A second copy of
# "canonical is top1_sparse with no noise" living in the guard is exactly how a
# guard comes to pass a run the model no longer implements (T5.4). No import
# cycle: model.py does not import this module.
from .model import CANONICAL_ROUTING_MODE, CANONICAL_ROUTER_NOISE
# T8.2: the guard must decide "which halting objective ran" and "is any ablation
# active" using the SAME code that stamps provenance, not a second copy of the
# rule. config.py imports model.py and families.py only, so there is no cycle.
from .config import resolve_halting_mode, resolve_variant, CANONICAL_VARIANT
# T-L1.3: which canonical spec applies is decided by the TASK, so the guard reads
# the task with the same helper the config layer does. resolve_task returns
# CANONICAL_TASK for a config with no `task` key, which is what keeps every
# arithmetic run -- and every archived resolved config -- resolving exactly as
# before (T-L1.0).
from .config import resolve_task, CANONICAL_TASK, TASK_ARITHMETIC, TASK_LANGUAGE


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

_PKG_DIR  = os.path.dirname(os.path.abspath(__file__))
_CODE_DIR = os.path.dirname(_PKG_DIR)
_REPO_DIR = os.path.dirname(_CODE_DIR)
RUNS_ROOT = os.path.join(_REPO_DIR, "runs")
CANONICAL_SPEC_PATH = os.path.join(_CODE_DIR, "canonical_spec.json")

CANONICAL_GROUP = "canonical_phase_b"

# --- T-L1.3: one frozen spec and one canonical group PER TASK --------------- #
#
# Why the group name has to differ, and not merely the file: 'canonical_phase_b'
# is the exact string export_results.py admits into the headline arithmetic
# table. A language run inside it would contribute a row whose primary metric is
# a per-token cross-entropy in nats sitting in a column of arithmetic MSEs --
# numerically plausible, scientifically meaningless. So a language run claiming
# 'canonical_phase_b' is REFUSED outright (see assert_not_silent_proxy) rather
# than quietly stamped exploratory, which is what would happen to any other
# unrecognised label.
#
# CANONICAL_GROUP and CANONICAL_SPEC_PATH keep their names and values: they are
# imported elsewhere (test_phase6_seeding.py) and are the arithmetic entries of
# the two maps below.
CANONICAL_SPEC_PATH_LANGUAGE = os.path.join(_CODE_DIR, "canonical_spec_language.json")
CANONICAL_GROUP_LANGUAGE = "canonical_lang_b"

CANONICAL_SPEC_PATH_BY_TASK = {
    TASK_ARITHMETIC: CANONICAL_SPEC_PATH,
    TASK_LANGUAGE:   CANONICAL_SPEC_PATH_LANGUAGE,
}
CANONICAL_GROUP_BY_TASK = {
    TASK_ARITHMETIC: CANONICAL_GROUP,
    TASK_LANGUAGE:   CANONICAL_GROUP_LANGUAGE,
}
# Every string that means "this is a headline run" on SOME task. Used only to
# tell a cross-task claim (refuse) from an exploratory label (allow).
_ALL_CANONICAL_GROUPS = frozenset(CANONICAL_GROUP_BY_TASK.values())

# Any run that does not explicitly and validly claim CANONICAL_GROUP is
# stamped with this, so the exporter's filter can never pick it up by accident.
NONCANONICAL_GROUP_DEFAULT = "exploratory"


# --------------------------------------------------------------------------- #
# Hashing / provenance primitives
# --------------------------------------------------------------------------- #

def canonical_json(obj) -> str:
    """Stable JSON serialization for hashing: sorted keys, no incidental space."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def config_hash(cfg: dict) -> str:
    """
    SHA-256 over the config with the `provenance` and `logging` blocks removed.

    `provenance` contains the hash itself and the run identity, so including it
    would make the hash irreproducible.

    T6.7: `logging` is excluded because it holds the run LABEL. With it inside
    the hash, `phaseB_MoRE_seed42` and `phaseB_MoRE_seed43` -- the same
    experiment at two seeds, which is exactly what has to be averaged in the
    final matrix -- produced different config hashes, and the results exporter
    refuses to mix config hashes. The hash must identify the scientific
    configuration (architecture, model, training, loss weights, data), not what
    the run was called or which W&B project it went to.
    """
    stripped = {k: v for k, v in cfg.items()
                if k not in ("provenance", "logging")}
    return hashlib.sha256(canonical_json(stripped).encode("utf-8")).hexdigest()


def git_commit() -> str:
    """Current HEAD sha, or 'unknown' outside a git checkout."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_REPO_DIR, capture_output=True, text=True, timeout=15,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return "unknown"


def git_dirty() -> bool | None:
    """True if the working tree has uncommitted changes; None if undeterminable."""
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=_REPO_DIR, capture_output=True, text=True, timeout=15,
        )
        if out.returncode == 0:
            return bool(out.stdout.strip())
    except Exception:
        pass
    return None


def _repo_relative(path: str) -> str:
    """
    `path` expressed relative to the repository root with forward slashes, or as
    an absolute path when no relative expression exists.

    T-L0.3: this was `os.path.relpath(directory, _REPO_DIR)` inline, and on
    Windows `relpath` RAISES when the two paths are on different drives:

        ValueError: path is on mount 'C:', start on mount 'D:'

    That is not a hypothetical. `test_phase6_seeding.py` passes a
    `tempfile.mkdtemp()` runs root, which lands on C:\\Users\\...\\Temp while
    this repository is on D:, so `RunContext.create` raised before writing
    anything and Gate 4 died mid-suite with 51 checks counted and no [FAIL]
    marker -- a crash that the runner can only report as "output format not
    recognised". It passed on the machine the POC was developed on solely
    because the repo and the temp directory happened to share a drive there.

    A run directory outside the repository is legitimate: a test using a temp
    root, or a user putting `runs/` on a data disk. Provenance should then
    record where the run actually is. The relative form is preserved whenever it
    exists, because that is the short, portable string `export_results.py` and
    `depth_null_model.py` put in their tables for runs under `runs/`.
    """
    try:
        return os.path.relpath(path, _REPO_DIR).replace("\\", "/")
    except ValueError:
        return os.path.abspath(path).replace("\\", "/")


def file_sha256(path: str) -> str | None:
    """SHA-256 of a file, or None if unreadable."""
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Gate 0 — proxy guard
# --------------------------------------------------------------------------- #

class ProxyGuardError(RuntimeError):
    """Raised when a run claims canonical status but does not satisfy the spec."""


def load_canonical_spec(path: str | None = None,
                        task: str | None = None) -> dict:
    """
    Load the canonical spec for a task.

    Fields whose value is null are 'not yet frozen'.  A run cannot claim
    canonical status while any required field is unfrozen — that is what stops
    a canonical matrix from being launched before the config is frozen.

    T-L1.3: WHICH spec is chosen by the TASK, not by the architecture. `task`
    None means CANONICAL_TASK, so `load_canonical_spec()` with no arguments still
    reads canonical_spec.json and every existing caller is unaffected. An
    explicit `path` still wins over the task selection, which is what lets a test
    point the guard at a fixture without monkeypatching the module.

    The two specs are separate FILES rather than two blocks of one file because
    canonical_spec.json has three readers besides this one -- export_results.py
    and the two Phase-6 gates parse it directly -- so restructuring it would put
    all of them on a language commit for no scientific gain. See that file's
    _README, and canonical_spec_language.json's.
    """
    if path is None:
        t = task or CANONICAL_TASK
        if t not in CANONICAL_SPEC_PATH_BY_TASK:
            raise ValueError(
                f"No canonical spec is registered for task {t!r}. Known tasks: "
                f"{sorted(CANONICAL_SPEC_PATH_BY_TASK)}. A new task must bring "
                "its own frozen spec -- falling back to another task's would let "
                "it inherit numbers that were measured on different data."
            )
        path = CANONICAL_SPEC_PATH_BY_TASK[t]
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _effective(resolved: dict) -> dict:
    """
    Pull the values the guard checks out of a resolved config.

    Keys here must match canonical_spec.json:enforced_fields exactly. Where a
    spec key would collide across sections (`halting` is a loss weight, but
    `halting_supervision` is both a model bool and a loss weight) the spec key
    carries a `_weight` suffix and is read from loss_weights explicitly.
    """
    mc = resolved.get("model", {})
    tc = resolved.get("training", {})
    dc = resolved.get("data", {})
    lw = resolved.get("loss_weights", {})
    prov = resolved.get("provenance", {})
    return {
        "epochs":          tc.get("epochs"),
        "batch_size":      tc.get("batch_size"),
        "subset_fraction": dc.get("subset_fraction"),
        "dataset_version": dc.get("dataset_version"),
        "seed":            prov.get("seed"),
        "num_experts":     mc.get("num_experts"),
        "max_depth":       mc.get("max_depth"),
        "num_blocks":      mc.get("num_blocks"),
        "d_model":         mc.get("d_model"),
        "lr":              tc.get("lr"),
        "weight_decay":    tc.get("weight_decay"),
        "routing_balance": lw.get("routing_balance"),
        # T5.4: the routing path itself. Both default to their canonical value,
        # so a run that simply omits the keys still reads as canonical -- what
        # the guard catches is a run that turns noise or dense blending ON and
        # then claims canonical status anyway.
        "routing_mode":    mc.get("routing_mode"),
        "router_noise":    mc.get("router_noise"),
        # ---- T8.2 additions -------------------------------------------------
        # The user-named list for T8.1/T8.2 was: d_model, lr, weight_decay,
        # routing_balance, num_blocks, batch_size, num_experts, max_depth,
        # halting mode, router mode, and ALL loss weights. Everything above this
        # line was already covered; everything below was named in the spec, or in
        # config.json, but never actually checked -- so a run could differ on it
        # and still be stamped canonical_phase_b.
        "train_split_version": dc.get("train_split_version"),
        "dropout":       mc.get("dropout"),
        "grad_clip":     tc.get("grad_clip"),
        "max_steps":     mc.get("max_steps"),
        "step_feat_dim": mc.get("step_feat_dim"),
        # Derived, not read off provenance: the guard runs before engine.train()
        # writes the provenance block. resolve_halting_mode is the same helper
        # engine.py calls, so guard and provenance cannot disagree.
        "halting_mode":  resolve_halting_mode(resolved),
        "task_weight":   lw.get("task"),
        # T8.3: the family-supervision weight. It belongs in enforced_fields
        # rather than in architecture_variants because apply_architecture does
        # NOT vary it -- all three architectures receive the same six oracle
        # family labels off disk and the same 6-wide cls_head, so 0.5 is a shared
        # canonical constant. Until T8.3 it was a bare literal in engine.py, which
        # meant the second-largest term in the objective was invisible to this
        # guard: a run could zero it by editing source and still be stamped
        # canonical_phase_b.
        "family_cls_weight": lw.get("family_cls"),
        # ---- T-L1.3: the language shape fields ------------------------------
        # Read unconditionally and with .get, so on an arithmetic config they are
        # simply None. That is safe because the guard iterates over the SPEC's
        # enforced_fields, not over this dict: canonical_spec.json does not list
        # them, so nothing compares them, and no arithmetic behaviour changes.
        # canonical_spec_language.json does list them, and a key it lists that
        # this function did not produce would be reported as
        # "requires 8192, run has None" -- a real check failing for a fake
        # reason, which is the trap this block exists to avoid.
        "seq_len":      dc.get("seq_len"),
        "vocab_size":   dc.get("vocab_size"),
        "n_heads":      mc.get("n_heads"),
        "attention":    mc.get("attention"),
        "tie_lm_head":  mc.get("tie_lm_head"),
        # Architecture-scoped (checked against architecture_variants[arch], not
        # against enforced_fields).
        "architecture":       resolved.get("architecture"),
        "adaptive_halting":   mc.get("adaptive_halting"),
        "ffn_mult":           mc.get("ffn_mult"),
        "routing_balance_weight":     lw.get("routing_balance"),
        "step_routing_weight":        lw.get("step_routing"),
        "halting_weight":            lw.get("halting"),
        "halting_supervision_weight": lw.get("halting_supervision"),
    }


# Descriptive-only keys in an architecture_variants block: not enforced, because
# the enforceable facts behind them are already covered elsewhere. `top1_routing`
# is implied by `routing_mode` (shared) plus `num_experts` (per-architecture);
# MoR's `false` is a consequence of having one expert, not a separate switch.
_ARCH_VARIANT_DESCRIPTIVE_ONLY = ("top1_routing",)



def assert_not_silent_proxy(resolved: dict, spec: dict | None = None) -> str:
    """
    Gate 0 enforcement.

    Returns the experiment_group the run is allowed to use.

    - A run that does not set logging.experiment_group to its task's canonical
      group is simply stamped non-canonical.  No error: exploratory work stays
      cheap.
    - A run that DOES claim it must satisfy every pinned field in that task's
      spec, must not be missing a seed, and must not run on a data subset.
      Otherwise ProxyGuardError aborts before any compute.
    - T-L1.3: a run that claims ANOTHER task's canonical group is refused
      outright.  Stamping it exploratory would be the wrong mercy -- the operator
      asked for a headline run and would get a silently downgraded one.
    """
    task = resolve_task(resolved)
    if spec is None:
        spec = load_canonical_spec(task=task)

    # Read from the spec, not from the module constants, so the two tasks cannot
    # drift apart in code: adding a task means adding a file, not an if-branch.
    canonical_group   = spec.get("canonical_group") or CANONICAL_GROUP
    canonical_variant = spec.get("canonical_variant") or CANONICAL_VARIANT
    # Which file the operator has to edit to fix a complaint below. Derived from
    # the task rather than hard-coded, so a language failure does not send the
    # reader to the arithmetic spec.
    spec_file = os.path.basename(
        CANONICAL_SPEC_PATH_BY_TASK.get(task, CANONICAL_SPEC_PATH))

    claimed = resolved.get("logging", {}).get("experiment_group")

    if claimed != canonical_group and claimed in _ALL_CANONICAL_GROUPS:
        raise ProxyGuardError(
            f"Gate 0 (proxy guard) refused this run: WRONG GROUP FOR THE TASK.\n"
            f"  task                     = {task!r}\n"
            f"  experiment_group claimed = {claimed!r}\n"
            f"  the group for this task   = {canonical_group!r}\n\n"
            f"{claimed!r} is the headline group of a different task, and each "
            "task's headline table is filtered on that exact string. Admitting "
            "this run would put its primary metric in a column of a different "
            "quantity -- a per-token cross-entropy in nats beside an arithmetic "
            "MSE, or the reverse. Set logging.experiment_group to "
            f"{canonical_group!r} if this really is a headline run for "
            f"task={task!r}, or to an exploratory label otherwise."
        )

    if claimed != canonical_group:
        return claimed or NONCANONICAL_GROUP_DEFAULT

    eff = spec.get("enforced_fields", {})
    got = _effective(resolved)
    problems: list[str] = []

    # 1. Nothing may be unfrozen.
    unfrozen = sorted(k for k, v in eff.items() if v is None)
    if unfrozen:
        problems.append(
            "the canonical spec has unfrozen fields (still null): "
            + ", ".join(unfrozen)
            + ". The canonical configuration must be frozen before any "
              "run may claim experiment_group='" + canonical_group + "'."
        )

    # 2. Every frozen field must match exactly.
    for key, want in eff.items():
        if want is None:
            continue
        have = got.get(key)
        if have != want:
            problems.append(f"{key}: canonical requires {want!r}, run has {have!r}")

    # 2b. T8.2 -- the architecture-specific half of the contract.
    #
    # Why this is separate from the loop above: config.py:apply_architecture
    # legitimately varies num_experts, max_depth, adaptive_halting and four loss
    # weights by architecture. MoE has no recursion, so its ponder weight is 0.0;
    # MoR has one expert, so its balance and routing-supervision weights are 0.0.
    # Putting any of those in enforced_fields pins one architecture's value onto
    # all three and refuses two thirds of the Phase 9 matrix. Before T8.2,
    # `routing_balance` sat in enforced_fields with a null; freezing it to 0.001
    # would have silently made every canonical MoR run impossible.
    #
    # Only the block matching THIS run's architecture is consulted, so a null in
    # (say) mor.ffn_mult blocks canonical MoR without blocking MoE or MoRE.
    variants = spec.get("architecture_variants", {}) or {}
    arch = got.get("architecture")
    if arch is None:
        problems.append(
            "config['architecture'] is not set. A canonical run must name its "
            f"architecture so the guard can enforce architecture_variants; see "
            f"config.py:apply_architecture."
        )
    else:
        arch_spec = variants.get(arch)
        if not isinstance(arch_spec, dict):
            problems.append(
                f"architecture={arch!r} has no block in "
                f"{spec_file}:architecture_variants, so there is nothing "
                "to enforce it against. Add the block or drop the canonical claim."
            )
        else:
            for key, want in arch_spec.items():
                if key.startswith("_") or key in _ARCH_VARIANT_DESCRIPTIVE_ONLY:
                    continue
                if want is None:
                    problems.append(
                        f"architecture_variants.{arch}.{key} is null (NOT YET "
                        f"FROZEN), so no {arch} run may claim canonical status. "
                        f"Freeze it in {spec_file} first -- see that "
                        "file's architecture_variants._note for why it is open."
                    )
                    continue
                have = got.get(key)
                if have != want:
                    problems.append(
                        f"{key}: canonical {arch} requires {want!r}, "
                        f"run has {have!r}"
                    )

    # 2c. T8.2 -- no ablation may hide inside the canonical group.
    #
    # The field-by-field checks above pin the values the spec happens to list.
    # This one closes the complement: resolve_variant() inspects every
    # deviation-carrying field (dense routing, router noise, fixed depth,
    # supervised curriculum, oracle routing supervision) and returns
    # "canonical" only when all of them sit at their canonical value. So a new
    # ablation added to resolve_variant is refused here automatically, without
    # anyone remembering to also add it to enforced_fields.
    #
    # This check is what made the resolve_variant typo consequential: it read a
    # loss-weight key (`routing_oracle`) that exists nowhere in the repo, so the
    # oracle-routing tag could never fire and an oracle-supervised run reported
    # variant="canonical". Fixed in T8.2; see config.py:resolve_variant.
    got_variant = resolve_variant(resolved)
    if got_variant != canonical_variant:
        problems.append(
            f"variant={got_variant!r} -- canonical for task={task!r} is "
            f"{canonical_variant!r}, so this run has at least one ablation "
            f"active and may not claim experiment_group='{canonical_group}'. "
            "plan.md E and CLAUDE.md 6 forbid mixing an ablation into the "
            "headline table; label it as the ablation it is."
        )


    # 3. A canonical run is never a subset run.
    if got["subset_fraction"] not in (None, 1.0, 1):
        problems.append(
            f"subset_fraction={got['subset_fraction']!r} - a canonical run must "
            "use the full dataset (subset_fraction=1.0)."
        )

    # 4. A canonical run must declare its seed.
    if got["seed"] is None:
        problems.append(
            "seed is not set. Canonical runs must pass --seed from the frozen "
            f"seed set (see {spec_file}:seed_set)."
        )
    elif got["seed"] not in spec.get("seed_set", []):
        problems.append(
            f"seed={got['seed']} is not in the frozen seed set "
            f"{spec.get('seed_set')}."
        )

    # 5. A canonical run must declare which dataset it used.
    if not got["dataset_version"]:
        problems.append(
            "data.dataset_version is not set. Canonical runs must name the "
            "dataset version they consumed (Phase 1 writes this into "
            "data/dataset_meta.json)."
        )

    # 6. T5.4 / T2.4: canonical is deterministic top-1 routing with no
    #    exploration noise. These two are checked here rather than only in
    #    canonical_spec.json because they are ARCHITECTURAL, not tuning: a
    #    dense-blend or noisy run is a different model, and updated_rules.md 1.1
    #    admits both solely as labelled ablations. Unlike the enforced_fields
    #    loop above, these fire even while the spec is unfrozen.
    if got["routing_mode"] not in (None, CANONICAL_ROUTING_MODE):
        problems.append(
            f"model.routing_mode={got['routing_mode']!r} - canonical is "
            f"{CANONICAL_ROUTING_MODE!r}. Dense blending is ablation F and may "
            "not claim canonical status (updated_rules.md 1.1, plan.md 4.1)."
        )
    if got["router_noise"] not in (None, CANONICAL_ROUTER_NOISE):
        problems.append(
            f"model.router_noise={got['router_noise']!r} - canonical is "
            f"{CANONICAL_ROUTER_NOISE!r}. Router exploration noise is ablation D "
            "and may not claim canonical status (updated_rules.md 1.1, "
            "plan.md 7.4)."
        )

    if problems:
        raise ProxyGuardError(
            "Gate 0 (proxy guard) refused this run.\n"
            f"A run with task={task!r} claiming experiment_group='"
            + canonical_group + "' must be fully canonical, as defined by "
            + spec_file + ". Problems:\n  - "
            + "\n  - ".join(problems)
            + "\n\nEither fix the config, or drop the canonical claim by setting "
              "logging.experiment_group to an exploratory label."
        )

    return canonical_group


# --------------------------------------------------------------------------- #
# stdout / stderr tee
# --------------------------------------------------------------------------- #

class _Tee:
    """Duplicate a text stream to a file handle. Never swallows the original."""

    def __init__(self, stream, handle):
        self._stream = stream
        self._handle = handle

    def write(self, data):
        self._stream.write(data)
        try:
            self._handle.write(data)
        except Exception:
            pass
        return len(data)

    def flush(self):
        try:
            self._stream.flush()
        except Exception:
            pass
        try:
            self._handle.flush()
        except Exception:
            pass

    def isatty(self):
        try:
            return self._stream.isatty()
        except Exception:
            return False

    def __getattr__(self, name):
        return getattr(self._stream, name)


# --------------------------------------------------------------------------- #
# RunContext
# --------------------------------------------------------------------------- #

class RunContext:
    """
    Owns runs/<experiment_id>/ for a single training run.

    Usage:
        ctx = RunContext.create(raw_cfg, resolved_cfg)
        ...
        torch.save(state, ctx.path("checkpoint.pt"))
        ctx.write_metrics(metrics_dict)
        ctx.close()
    """

    def __init__(self, experiment_id: str, directory: str,
                 resolved_cfg: dict, experiment_group: str):
        self.experiment_id = experiment_id
        self.dir = directory
        self.resolved_cfg = resolved_cfg
        self.experiment_group = experiment_group
        self._log_handle = None
        self._saved_stdout = None
        self._saved_stderr = None

    # -- construction ------------------------------------------------------ #

    @classmethod
    def create(cls, raw_cfg: dict, resolved_cfg: dict,
               runs_root: str = RUNS_ROOT, tee_stdout: bool = True) -> "RunContext":
        """
        Validate against Gate 0, allocate runs/<experiment_id>/, and write both
        config snapshots.

        Raises ProxyGuardError before creating anything if the run claims
        canonical status without satisfying canonical_spec.json.
        """
        group = assert_not_silent_proxy(resolved_cfg)
        resolved_cfg.setdefault("logging", {})["experiment_group"] = group

        experiment_id = cls._allocate_id(resolved_cfg, runs_root)
        directory = os.path.join(runs_root, experiment_id)
        os.makedirs(directory, exist_ok=False)

        prov = resolved_cfg.setdefault("provenance", {})
        prov["experiment_id"]   = experiment_id
        prov["experiment_group"] = group
        prov["code_git_commit"] = git_commit()
        prov["code_git_dirty"]  = git_dirty()
        prov["config_hash"]     = config_hash(resolved_cfg)
        prov["run_dir"]         = _repo_relative(directory)

        # T6.7 (updated_rules.md 9): architecture, variant, run_name and the two
        # data versions are REQUIRED provenance. They already live elsewhere in
        # the resolved config, but a reader of the provenance block should not
        # have to know that -- and the results exporter reads this block, not the
        # whole file. `variant` is derived, never declared, so an ablation cannot
        # be written up as canonical by omission.
        from .config import resolve_variant
        data = resolved_cfg.get("data", {}) or {}
        prov["architecture"]        = resolved_cfg.get("architecture")
        prov["variant"]             = resolve_variant(resolved_cfg)
        prov["run_name"]            = resolved_cfg.get("logging", {}).get("run_name")
        prov["dataset_version"]     = data.get("dataset_version")
        prov["train_split_version"] = data.get("train_split_version")

        ctx = cls(experiment_id, directory, resolved_cfg, group)

        with open(ctx.path("config.json"), "w", encoding="utf-8") as f:
            json.dump(raw_cfg, f, indent=2, default=str)
        ctx.write_resolved_config()

        if tee_stdout:
            ctx.open_log()
        return ctx

    @staticmethod
    def _allocate_id(resolved_cfg: dict, runs_root: str) -> str:
        """
        Deterministic, collision-safe run id:  <run_name>__seed<S>__<hash8>

        No timestamp: the same config re-run is recognisably the same
        experiment, and a numeric suffix disambiguates repeats.
        """
        log = resolved_cfg.get("logging", {})
        prov = resolved_cfg.get("provenance", {})
        base = log.get("run_name") or "run"
        base = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in str(base))
        seed = prov.get("seed")
        seed_part = f"seed{seed}" if seed is not None else "seedNA"
        h8 = config_hash(resolved_cfg)[:8]

        # T6.6: the run NAME now carries the seed (phaseB_MoRE_seed42), so
        # appending it again would produce phaseB_MoRE_seed42__seed42__hash.
        # The seed segment is still added when the name does not already state
        # it, because the directory must be readable without opening a file.
        stem = (f"{base}__{h8}" if seed_part in base
                else f"{base}__{seed_part}__{h8}")
        os.makedirs(runs_root, exist_ok=True)
        candidate, n = stem, 1
        while os.path.exists(os.path.join(runs_root, candidate)):
            n += 1
            candidate = f"{stem}__r{n}"
        return candidate

    # -- paths and writers -------------------------------------------------- #

    def path(self, filename: str) -> str:
        """Absolute path to a file inside this run's directory."""
        return os.path.join(self.dir, filename)

    def write_resolved_config(self) -> None:
        with open(self.path("resolved_config.json"), "w", encoding="utf-8") as f:
            json.dump(self.resolved_cfg, f, indent=2, default=str)

    def write_metrics(self, metrics: dict) -> str:
        """Serialize a metrics dict to metrics.json, coercing tensors/objects."""
        out = {}
        for k, v in metrics.items():
            if hasattr(v, "item"):
                try:
                    out[k] = v.item()
                    continue
                except Exception:
                    pass
            if isinstance(v, (int, float, str, bool, type(None), list, dict)):
                out[k] = v
            else:
                # Do NOT str() it. The log dict also carries wandb.Image /
                # wandb.Table objects for the paper figures; str() wrote
                # "<wandb.sdk.data_types.image.Image object at 0x...>" into
                # metrics.json, where a memory address masquerades as a
                # recorded value. Figures belong in W&B, not in the metric
                # dump, so drop them and name them instead.
                out.setdefault("_non_scalar_keys_omitted", []).append(
                    f"{k} ({type(v).__name__})"
                )
        target = self.path("metrics.json")
        with open(target, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        return target

    # -- stdout capture ----------------------------------------------------- #

    def open_log(self) -> None:
        if self._log_handle is not None:
            return
        self._log_handle = open(self.path("stdout.log"), "w", encoding="utf-8",
                                buffering=1)
        self._saved_stdout, self._saved_stderr = sys.stdout, sys.stderr
        sys.stdout = _Tee(self._saved_stdout, self._log_handle)
        sys.stderr = _Tee(self._saved_stderr, self._log_handle)

    def close(self) -> None:
        """Restore streams and flush the log. Safe to call more than once."""
        if self._saved_stdout is not None:
            sys.stdout = self._saved_stdout
            self._saved_stdout = None
        if self._saved_stderr is not None:
            sys.stderr = self._saved_stderr
            self._saved_stderr = None
        if self._log_handle is not None:
            try:
                self._log_handle.flush()
                self._log_handle.close()
            except Exception:
                pass
            self._log_handle = None

    def __enter__(self) -> "RunContext":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


# --------------------------------------------------------------------------- #
# Config resolution
# --------------------------------------------------------------------------- #

def resolve_overrides(cfg: dict, *, epochs: int | None = None,
                      blocks: int | None = None,
                      batch_size: int | None = None,
                      seed: int | None = None,
                      run_name: str | None = None,
                      experiment_group: str | None = None) -> dict:
    """
    Apply CLI overrides into a deep copy of cfg so the resolved config is the
    single truthful record of what actually ran.

    This fixes the historical defect where `--epochs 1` still logged
    `epochs: 50` to W&B, because the override never reached the config that
    was serialized.
    """
    out = copy.deepcopy(cfg)
    out.setdefault("model", {})
    out.setdefault("training", {})
    out.setdefault("data", {})
    out.setdefault("logging", {})
    prov = out.setdefault("provenance", {})

    if blocks is not None:
        out["model"]["num_blocks"] = blocks
    if batch_size is not None:
        out["training"]["batch_size"] = batch_size
    if epochs is not None:
        out["training"]["epochs"] = epochs
    if seed is not None:
        out["training"]["subset_seed"] = seed
        out["data"]["subset_seed"] = seed
        prov["seed"] = seed
    else:
        prov.setdefault("seed", None)
    if run_name is not None:
        out["logging"]["run_name"] = run_name
    if experiment_group is not None:
        out["logging"]["experiment_group"] = experiment_group

    # Truthful record of the values the training loop will actually use.
    prov["resolved_epochs"] = out["training"].get("epochs")
    prov["resolved_subset_fraction"] = out["data"].get("subset_fraction")
    prov["resolved_batch_size"] = out["training"].get("batch_size")
    return out
