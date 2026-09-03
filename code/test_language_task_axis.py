"""test_language_task_axis.py - T-L1.0..T-L1.3 verification for the `task` axis.

Run:
    C:/Users/vedan/anaconda3/python.exe code/test_language_task_axis.py

WHY THIS IS NOT IN `run_correctness_suite.py:GATES` (yet). Gate L0
(plan_language.md §11) asserts the arithmetic suite still prints
`TOTAL 356 356 0 0` -- an UNCHANGED total is the whole content of the gate, so
registering new checks there would make the gate's own criterion unmeetable and
destroy the one number that says "the published arithmetic study still holds".
This file is therefore run directly, and is registered when the language gate
table (L1..L8) is built. It already emits the `[PASS]`/`[FAIL]` markers the
runner counts, so registration is a one-line change with no edits here.

WHAT IS ACTUALLY AT RISK, and why the central check is a hash and not a field
comparison. `run_context.config_hash()` is SHA-256 over the resolved config with
only `provenance` and `logging` removed. So ANY new key that reaches the resolved
config -- however inert at runtime -- changes the identity of every run that
resolves through it. The failure mode is not a wrong number; it is that
`runs/phaseB_MoRE_seed42__<hash8>` stops being the directory a re-run of that
config produces, `export_results.py` (which refuses to mix config hashes) reads a
re-run as a different experiment, and fifteen published canonical runs quietly
stop being reproducible from their own recorded configs. That is why T-L1.0's
Verify says "byte-identically ... diff the two resolved dicts, not just spot
fields", and why the default task is the ABSENCE of the key rather than an
injected `"arithmetic"`.

The evidence used here is the 18 tracked `runs/phaseB_*` directories: each stores
the resolved config it ran with AND the hash it recorded at the time, so
re-hashing the stored dict compares today's hash function against a value written
before this axis existed. No run is re-executed (T-LX.2).
"""

import copy
import glob
import hashlib
import json
import os
import sys

sys.path.insert(0, __file__.rsplit("\\", 1)[0].rsplit("/", 1)[0])

from more.config import (load_config_defaults, load_config, apply_architecture,
                         apply_task, resolve_task, resolve_variant,
                         canonical_run_name, stamp_seed_into_run_name,
                         TASKS, TASK_ARITHMETIC, TASK_LANGUAGE, CANONICAL_TASK,
                         CANONICAL_VARIANT, CANONICAL_RUN_PREFIX,
                         CANONICAL_RUN_PREFIX_LANGUAGE,
                         LANGUAGE_N_HEADS_DEFAULT, LANGUAGE_SEQ_LEN_DEFAULT,
                         LANGUAGE_VOCAB_SIZE_DEFAULT, ARCHITECTURES)
from more.run_context import config_hash, canonical_json

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

CODE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(CODE)

PASS, FAIL = [], []


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))


def _raises_valueerror(fn, needle=""):
    """True iff fn() raises ValueError whose message contains `needle`.

    The needle matters: "it raised" is satisfied by a KeyError from a typo two
    frames deeper, which is not the refusal the check is asserting.
    """
    try:
        fn()
        return False
    except ValueError as e:
        return needle in str(e)
    except Exception:
        return False


# ===========================================================================
print("\n--- T-L1.0  the arithmetic resolved config is byte-identical")
# ===========================================================================

# 1. No `task` key anywhere in an arithmetic resolution. Checked at every level,
#    not just the top: a nested `model.task` would also enter the hash.
arith = load_config_defaults({})


def _find_key(obj, key, path="cfg"):
    hits = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == key:
                hits.append(f"{path}.{k}")
            hits += _find_key(v, key, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            hits += _find_key(v, key, f"{path}[{i}]")
    return hits


hits = _find_key(arith, "task")
# `loss_weights.task` is NOT the task axis -- it is the WEIGHT on the task-loss
# term, and it has existed since before this migration. The name collision is
# real and is flagged in config.py where the axis is defined: `cfg["task"]`
# selects the dataset and the objective, `cfg["loss_weights"]["task"]` scales one
# term of it. Anything ELSE named `task` in a resolved arithmetic config would be
# the axis leaking in and moving the hash.
PREEXISTING_TASK_KEYS = {"cfg.loss_weights.task"}
leaked = sorted(set(hits) - PREEXISTING_TASK_KEYS)
check("TL1.0a arithmetic resolution contains no task-AXIS key at any depth",
      leaked == [],
      f"leaked at {leaked}" if leaked
      else f"only the pre-existing task-loss weight {sorted(hits)}")

check("TL1.0b absence resolves to the arithmetic task",
      resolve_task(arith) == TASK_ARITHMETIC == CANONICAL_TASK,
      f"resolve_task({{}}) = {resolve_task(arith)!r}")

# 2. `--task arithmetic` and no flag at all must produce THE SAME BYTES. This is
#    the check that would fail under a `setdefault("task", "arithmetic")`
#    implementation: the explicit form would carry the key and the implicit form
#    would too, but every pre-existing config on disk resolves without it.
implicit = load_config_defaults({})
explicit = apply_task(load_config_defaults({}), TASK_ARITHMETIC)
check("TL1.0c apply_task(cfg, 'arithmetic') is byte-identical to no task at all",
      canonical_json(implicit) == canonical_json(explicit),
      f"{len(canonical_json(implicit))} bytes both" if
      canonical_json(implicit) == canonical_json(explicit) else "DIFFER")

check("TL1.0d and therefore has the same config_hash",
      config_hash(implicit) == config_hash(explicit),
      config_hash(implicit)[:16] + "...")

# 3. apply_task(arithmetic) REMOVES a task key rather than rewriting it, so an
#    arithmetic re-run of a language config cannot inherit a hash-changing key.
downgraded = apply_task(load_config_defaults({"task": TASK_LANGUAGE}),
                        TASK_ARITHMETIC)
check("TL1.0e apply_task(cfg, 'arithmetic') pops an existing `task` key",
      "task" not in downgraded, f"keys = {sorted(downgraded.keys())}")

# 4. The real config file on disk, resolved through the current code, still has
#    the top-level key set it had before the axis existed. `_README` is a comment
#    block the config file has carried for a long time; it is hashed like any
#    other key, which is fine because it is stable -- it is listed here so that a
#    NEW key is what this check catches, rather than being lost in a hand-written
#    allowlist that nobody trusts.
HISTORICAL_TOP_KEYS = {"_README", "architecture", "data", "logging",
                       "loss_weights", "model", "training"}
disk = load_config(os.path.join(CODE, "config.json"))
extra = set(disk.keys()) - HISTORICAL_TOP_KEYS - {"provenance", "variant"}
check("TL1.0f code/config.json resolves with no new top-level key",
      extra == set(), f"unexpected: {sorted(extra)}" if extra
      else f"keys = {sorted(disk.keys())}")

# 5. THE DECISIVE CHECK. Re-hash every tracked arithmetic run's stored resolved
#    config and compare against the hash that run recorded. A changed hashed key
#    set shows up here as a mismatch on real evidence written before this axis
#    existed.
#
#    Scoped to the SEEDED runs, which are the evidence: the canonical matrix is
#    seeds 42-46 x {MoE, MoR, MoRE} = 15 runs, and an undeclared-seed run is not
#    citable (T6.1). The three `__seedNA__` directories are handled by TL1.0g2
#    below, which explains why they cannot match and proves it is not this change.
def _rehash(dirs):
    ok, bad, unhashed = [], [], 0
    for d in dirs:
        rc = os.path.join(d, "resolved_config.json")
        if not os.path.exists(rc):
            continue
        with open(rc, "r", encoding="utf-8") as fh:
            stored = json.load(fh)
        recorded = (stored.get("provenance", {}) or {}).get("config_hash")
        if not recorded:
            unhashed += 1
            continue
        if config_hash(stored) == recorded:
            ok.append(os.path.basename(d))
        else:
            bad.append((os.path.basename(d), recorded[:8], config_hash(stored)[:8]))
    return ok, bad, unhashed


run_dirs = sorted(glob.glob(os.path.join(REPO, "runs", "phaseB_*")))
seeded = [d for d in run_dirs if "seedNA" not in os.path.basename(d)]
unseeded = [d for d in run_dirs if "seedNA" in os.path.basename(d)]

ok, mismatched, unhashed = _rehash(seeded)
check("TL1.0g every SEEDED phaseB run's recorded config_hash is still "
      "re-derivable from its own resolved config",
      len(ok) > 0 and not mismatched,
      f"{len(ok)} runs re-hashed and matched, {unhashed} without a recorded "
      f"hash, {len(mismatched)} mismatched"
      + (f": {mismatched[:3]}" if mismatched else ""))

# TL1.0g2. The three unseeded runs record a hash computed when config_hash
# excluded `provenance` ONLY; `logging` was added to the exclusion set later, and
# correctly so -- run_name and experiment_group are labels, and a run's config
# identity must not change when it is renamed. So those three CANNOT re-derive
# under today's function, and that is a dated historical fact, not a defect and
# not something this change caused. Asserted rather than narrated: if a SEEDED run
# ever needed the old exclusion set to verify, that would be a real finding.
old_excl_matches = []
for d in unseeded:
    rc = os.path.join(d, "resolved_config.json")
    if not os.path.exists(rc):
        continue
    with open(rc, "r", encoding="utf-8") as fh:
        stored = json.load(fh)
    recorded = (stored.get("provenance", {}) or {}).get("config_hash")
    legacy = hashlib.sha256(canonical_json(
        {k: v for k, v in stored.items() if k != "provenance"}
    ).encode("utf-8")).hexdigest()
    if recorded and legacy == recorded:
        old_excl_matches.append(os.path.basename(d))

_, unseeded_bad, _ = _rehash(unseeded)
check("TL1.0g2 the only runs that do not re-derive are the unseeded ones, and "
      "they match under the pre-`logging`-exclusion hash",
      len(old_excl_matches) == len(unseeded_bad),
      f"{len(unseeded_bad)} unseeded runs do not re-derive today; "
      f"{len(old_excl_matches)} of them reproduce exactly under the old "
      f"provenance-only exclusion -> the mismatch predates the task axis")

# The run directory name carries the first 8 hex of the hash, so a moved hash
# also breaks the name<->identity link the exporter relies on.
namebreak = []
for d in run_dirs:
    rc = os.path.join(d, "resolved_config.json")
    if not os.path.exists(rc):
        continue
    with open(rc, "r", encoding="utf-8") as fh:
        stored = json.load(fh)
    recorded = (stored.get("provenance", {}) or {}).get("config_hash")
    if recorded and not os.path.basename(d).endswith(recorded[:8]):
        namebreak.append((os.path.basename(d), recorded[:8]))
check("TL1.0h and still matches the hash8 suffix in its directory name",
      not namebreak, f"{len(run_dirs)} dirs checked"
      if not namebreak else f"broken: {namebreak[:3]}")


# ===========================================================================
print("\n--- T-L1.0  the language block stamps and refuses")
# ===========================================================================

lang = load_config_defaults({"task": TASK_LANGUAGE})
check("TL1.0i language resolution declares the task explicitly",
      lang.get("task") == TASK_LANGUAGE and resolve_task(lang) == TASK_LANGUAGE,
      f"task = {lang.get('task')!r}")

check("TL1.0j attention is ASSIGNED on (not left to a config file)",
      lang["model"]["attention"] is True)

check("TL1.0k language and arithmetic hash differently",
      config_hash(lang) != config_hash(arith),
      f"{config_hash(lang)[:8]} vs {config_hash(arith)[:8]}")

check("TL1.0l shape fields present: seq_len / vocab_size / n_heads / tie_lm_head",
      lang["data"]["seq_len"] == LANGUAGE_SEQ_LEN_DEFAULT
      and lang["data"]["vocab_size"] == LANGUAGE_VOCAB_SIZE_DEFAULT
      and lang["model"]["n_heads"] == LANGUAGE_N_HEADS_DEFAULT
      and lang["model"]["tie_lm_head"] is True,
      f"seq_len={lang['data']['seq_len']} (provisional, GATE L5 owns it), "
      f"V={lang['data']['vocab_size']}, n_heads={lang['model']['n_heads']}")

# The three numbers are setdefault, not assignment: config_language.json and the
# frozen spec must win, or GATE L5's measured seq_len is overwritten by a literal.
declared = load_config_defaults({
    "task": TASK_LANGUAGE,
    "data": {"seq_len": 512, "vocab_size": 4096},
    "model": {"n_heads": 8},
})
check("TL1.0m a declared seq_len / vocab_size / n_heads wins over the default",
      declared["data"]["seq_len"] == 512
      and declared["data"]["vocab_size"] == 4096
      and declared["model"]["n_heads"] == 8,
      f"seq_len={declared['data']['seq_len']}, "
      f"V={declared['data']['vocab_size']}, "
      f"n_heads={declared['model']['n_heads']}")

check("TL1.0n family_cls is forced to 0.0 on language (detached probe, §7.3)",
      float(lang["loss_weights"]["family_cls"]) == 0.0,
      f"family_cls = {lang['loss_weights']['family_cls']!r}, "
      f"arithmetic keeps {arith['loss_weights']['family_cls']!r}")

# The distinction that makes the refusal meaningful: a DECLARED non-zero weight is
# a copied arithmetic config and is refused; an undeclared one is the arithmetic
# default and is silently correct to zero.
raised = None
try:
    load_config_defaults({"task": TASK_LANGUAGE,
                          "loss_weights": {"family_cls": 0.5}})
except ValueError as e:
    raised = str(e)
check("TL1.0o a DECLARED non-zero family_cls on language raises",
      raised is not None and "7.3" in raised,
      (raised or "did not raise").splitlines()[0][:90])

check("TL1.0p an UNDECLARED family_cls does not raise (`--task language` alone "
      "must work)",
      float(load_config_defaults({"task": TASK_LANGUAGE})
            ["loss_weights"]["family_cls"]) == 0.0)

check("TL1.0q family_cls = 0.0 declared explicitly is accepted",
      float(load_config_defaults(
          {"task": TASK_LANGUAGE, "loss_weights": {"family_cls": 0.0}}
      )["loss_weights"]["family_cls"]) == 0.0)

# An unrecognised task must RAISE, not fall back to arithmetic: a fallback runs
# the wrong pipeline to completion and reports it as the right one.
raised = None
try:
    resolve_task({"task": "lang"})
except ValueError as e:
    raised = str(e)
check("TL1.0r an unrecognised task raises instead of defaulting",
      raised is not None and "arithmetic" in raised,
      (raised or "did not raise")[:80])

raised = None
try:
    apply_task({}, "wikitext")
except ValueError as e:
    raised = str(e)
check("TL1.0s apply_task refuses an unknown task",
      raised is not None, (raised or "did not raise")[:60])

# n_heads must divide d_model -- caught in the config layer, where the field name
# is known, not inside nn.MultiheadAttention.
raised = None
try:
    load_config_defaults({"task": TASK_LANGUAGE, "model": {"n_heads": 5}})
except ValueError as e:
    raised = str(e)
check("TL1.0t n_heads that does not divide d_model raises in the config layer",
      raised is not None and "n_heads" in raised,
      (raised or "did not raise")[:70])


# ===========================================================================
print("\n--- T-L1.0  the task axis is orthogonal to the architecture axis")
# ===========================================================================

# All six (task, architecture) pairs must resolve, and `architecture` must stamp
# exactly what it stamped before on both tasks: the two axes answer different
# questions (which module / how much depth vs which data and objective) and a
# cross-term here would mean one had leaked into the other.
ARCH_EXPECT = {"moe": (6, 1, False), "mor": (1, None, True), "more": (6, None, True)}
ok_pairs, bad = 0, []
for t in TASKS:
    for a in ARCHITECTURES:
        cfg = load_config_defaults({"task": t} if t != TASK_ARITHMETIC else {})
        cfg = apply_architecture(cfg, a)
        cfg = apply_task(cfg, t)
        want_E, want_depth, want_halt = ARCH_EXPECT[a]
        mc = cfg["model"]
        good = (mc["num_experts"] == want_E
                and mc["adaptive_halting"] is want_halt
                and (want_depth is None or mc["max_depth"] == want_depth)
                and resolve_task(cfg) == t)
        # attention is a property of the TASK, on all three architectures:
        # plan_language.md §6.6, "MoR keeps attention" -- one expert is not the
        # same claim as no context.
        good = good and (mc.get("attention", False) is (t == TASK_LANGUAGE))
        ok_pairs += bool(good)
        if not good:
            bad.append((t, a, mc.get("num_experts"), mc.get("max_depth"),
                        mc.get("adaptive_halting"), mc.get("attention")))
check("TL1.0u all 6 (task, architecture) pairs resolve with both axes intact",
      ok_pairs == len(TASKS) * len(ARCHITECTURES),
      f"{ok_pairs}/{len(TASKS) * len(ARCHITECTURES)}"
      + (f", bad: {bad}" if bad else "; attention on iff task=language, MoR included"))

# apply_task must be idempotent and order-independent with apply_architecture,
# because cli.py fixes one order and the tests use the other.
c1 = apply_task(apply_architecture(load_config_defaults({}), "more"), TASK_LANGUAGE)
c2 = apply_architecture(apply_task(load_config_defaults({}), TASK_LANGUAGE), "more")
check("TL1.0v apply_task and apply_architecture commute",
      canonical_json(c1) == canonical_json(c2),
      "identical" if canonical_json(c1) == canonical_json(c2) else "DIFFER")

c3 = apply_task(copy.deepcopy(c1), TASK_LANGUAGE)
check("TL1.0w apply_task is idempotent",
      canonical_json(c1) == canonical_json(c3))


# ===========================================================================
print("\n--- T-L1.1  the variant string and the run name name the task")
# ===========================================================================

# The defect this box exists to prevent, and it was live: resolve_variant compared
# family_cls against the arithmetic canonical 0.5, so an unmodified canonical
# language config -- where 0.0 is REQUIRED -- resolved to
# `variant = "no_family_supervision"`. A canonical run wearing an ablation label,
# in the exact field that exists to stop an ablation being read as canonical.
lang_more = apply_architecture(load_config_defaults({"task": TASK_LANGUAGE}), "more")
v_lang = resolve_variant(lang_more)
check("TL1.1a an unmodified language config's variant names the task",
      v_lang == TASK_LANGUAGE, f"variant = {v_lang!r}")

check("TL1.1b and does NOT carry the arithmetic-canonical family_cls ablation tag",
      "no_family_supervision" not in v_lang, f"variant = {v_lang!r}")

check("TL1.1c a language run can never be labelled 'canonical'",
      CANONICAL_VARIANT not in v_lang.split("+"), f"variant = {v_lang!r}")

# Unchanged for arithmetic: the 15 published runs must still read "canonical".
arith_more = apply_architecture(load_config_defaults({}), "more")
check("TL1.1d an unmodified arithmetic config still resolves to 'canonical'",
      resolve_variant(arith_more) == CANONICAL_VARIANT,
      f"variant = {resolve_variant(arith_more)!r}")

# A real deviation on language must still be labelled, with the task leading.
lang_fixed = apply_architecture(
    load_config_defaults({"task": TASK_LANGUAGE, "model": {"fixed_depth": True}}),
    "more")
v_fixed = resolve_variant(lang_fixed)
check("TL1.1e a language ablation reads task-then-deviation",
      v_fixed == f"{TASK_LANGUAGE}+fixed_depth", f"variant = {v_fixed!r}")

# A DECLARED 0.0 on arithmetic is still the T10.H ablation -- the task-dependent
# canonical must not have made that tag unreachable.
arith_nofam = apply_architecture(
    load_config_defaults({"loss_weights": {"family_cls": 0.0}}), "more")
check("TL1.1f arithmetic family_cls = 0.0 is still tagged no_family_supervision",
      resolve_variant(arith_nofam) == "no_family_supervision",
      f"variant = {resolve_variant(arith_nofam)!r}")

# Run naming.
check("TL1.1g canonical_run_name(moe, 42, language) = langB_MoE_seed42",
      canonical_run_name("moe", 42, TASK_LANGUAGE) == "langB_MoE_seed42",
      canonical_run_name("moe", 42, TASK_LANGUAGE))

check("TL1.1h all three language arms are langB_<Arch>_seed42",
      [canonical_run_name(a, 42, TASK_LANGUAGE) for a in ARCHITECTURES]
      == ["langB_MoE_seed42", "langB_MoR_seed42", "langB_MoRE_seed42"],
      ", ".join(canonical_run_name(a, 42, TASK_LANGUAGE) for a in ARCHITECTURES))

check("TL1.1i arithmetic names are unchanged and the default is arithmetic",
      canonical_run_name("more", 42) == "phaseB_MoRE_seed42"
      == canonical_run_name("more", 42, TASK_ARITHMETIC),
      canonical_run_name("more", 42))

check("TL1.1j the two prefixes cannot collide under a glob",
      not "langB".startswith(CANONICAL_RUN_PREFIX)
      and not CANONICAL_RUN_PREFIX.startswith(CANONICAL_RUN_PREFIX_LANGUAGE)
      and CANONICAL_RUN_PREFIX_LANGUAGE == "langB",
      f"{CANONICAL_RUN_PREFIX}_* vs {CANONICAL_RUN_PREFIX_LANGUAGE}_*")

check("TL1.1k an undeclared seed still yields no _seed suffix (T6.1)",
      canonical_run_name("mor", None, TASK_LANGUAGE) == "langB_MoR",
      canonical_run_name("mor", None, TASK_LANGUAGE))

# The cli.py ordering: apply_architecture stamps the name first, apply_task must
# repair it. Both orders must end at langB, because the ordering is a call-site
# detail and a run's name is not.
cli_order = apply_task(
    apply_architecture(load_config_defaults({}), "more"), TASK_LANGUAGE)
check("TL1.1l apply_task repairs the phaseB name apply_architecture stamped",
      cli_order["logging"]["run_name"] == "langB_MoRE",
      f"run_name = {cli_order['logging']['run_name']!r}")

check("TL1.1m and the config-file order gives the same name",
      apply_architecture(load_config_defaults({"task": TASK_LANGUAGE}), "more")
      ["logging"]["run_name"] == "langB_MoRE")

# ... but a name someone CHOSE must survive. Overwriting it would strip the label
# enforce_routing_mode requires from a dense-routing ablation.
chosen = apply_task(
    apply_architecture(
        load_config_defaults({"logging": {"run_name": "probe_dense_routing_ablation"}}),
        "more"),
    TASK_LANGUAGE)
check("TL1.1n a run_name the config chose is NOT overwritten by apply_task",
      chosen["logging"]["run_name"] == "probe_dense_routing_ablation",
      f"run_name = {chosen['logging']['run_name']!r}")

# The seed suffix is stamped later, from provenance, and must work on either task.
seeded_cfg = apply_task(apply_architecture(load_config_defaults({}), "moe"),
                        TASK_LANGUAGE)
seeded_cfg.setdefault("provenance", {})["seed"] = 44
check("TL1.1o stamp_seed_into_run_name yields langB_MoE_seed44",
      stamp_seed_into_run_name(seeded_cfg) == "langB_MoE_seed44",
      stamp_seed_into_run_name(seeded_cfg))


# ===========================================================================
print("\n--- T-L1.2  the CLI: --task / --seq_len / --vocab_size")
# ===========================================================================
# These drive cli.main() end to end -- argparse, apply_architecture, apply_task,
# every override block in order, resolve_overrides, enforce_routing_mode and the
# seed stamp -- with only RunContext.create and train() replaced by stubs.
#
# Stubbing those two is not weakening the check. The entire argument contract is
# resolved before either runs, and running them for real would need the language
# dataset and model (owed to L-2 and L-4) and would deposit a runs/ directory that
# a later reader could mistake for evidence. What RunContext.create does check --
# the proxy guard -- is exercised directly in the T-L1.3 block below, on configs
# built the same way.
#
# WHY THE REFUSALS SURFACE AS EXCEPTIONS, NOT EXIT CODES: main() catches
# ProxyGuardError only. A bad flag combination is an operator error that must
# arrive as a traceback naming the field, not as a bare non-zero status. The
# ledger's "exits non-zero" is satisfied either way -- an uncaught exception from
# train.py exits 1 -- and the checks below assert on the message, which is the
# part that has to be right.
import io
import contextlib

import more.cli as cli_mod

CFG_PATH = os.path.join(CODE, "config.json")


class _FakeCtx:
    """No run directory: this suite must not write anything under runs/."""

    def close(self):
        pass


class _FakeRunContext:
    @staticmethod
    def create(raw_cfg, resolved_cfg, **kw):
        return _FakeCtx()


def run_cli(*flags):
    """(rc, resolved_or_None, exception_or_None) for `train.py <flags>`."""
    box = {}

    def _fake_train(resolved, ctx=None):
        box["resolved"] = resolved

    real_train, real_ctx = cli_mod.train, cli_mod.RunContext
    cli_mod.train, cli_mod.RunContext = _fake_train, _FakeRunContext
    sink = io.StringIO()
    try:
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            rc = cli_mod.main(["--config", CFG_PATH, *flags])
        return rc, box.get("resolved"), None
    except SystemExit as e:                       # argparse rejected the flags
        return (e.code if isinstance(e.code, int) else 1), box.get("resolved"), e
    except Exception as e:
        return 1, box.get("resolved"), e
    finally:
        cli_mod.train, cli_mod.RunContext = real_train, real_ctx


# --- the two checks the ledger box names, first ----------------------------
rc, res, err = run_cli("--architecture", "more", "--task", "language",
                       "--family_cls", "0.5")
check("TL1.2a --task language --family_cls 0.5 exits non-zero",
      rc != 0 and res is None,
      f"rc={rc}, reached train()={res is not None}")
check("TL1.2b ... with a refusal naming plan_language.md §7.3",
      isinstance(err, ValueError) and "7.3" in str(err)
      and "detach" in str(err).lower(),
      f"{type(err).__name__}: {str(err)[:70]}")

rc, res, err = run_cli("--architecture", "more", "--task", "language")
check("TL1.2c --task language alone succeeds",
      rc == 0 and err is None and res is not None,
      f"rc={rc}, err={type(err).__name__ if err else None}")
check("TL1.2d ... and resolves to a complete language config",
      bool(res) and res.get("task") == TASK_LANGUAGE
      and res["model"]["attention"] is True
      and res["loss_weights"]["family_cls"] == 0.0
      and res["model"]["n_heads"] == LANGUAGE_N_HEADS_DEFAULT
      and res["data"]["vocab_size"] == LANGUAGE_VOCAB_SIZE_DEFAULT,
      "task/attention/family_cls/n_heads/vocab_size all as specified")
check("TL1.2e ... named langB_MoRE, not phaseB_MoRE",
      bool(res) and res["logging"]["run_name"] == "langB_MoRE",
      f"run_name = {res['logging']['run_name']!r}" if res else "no config")

# --- the arithmetic half: the CLI must not move an existing hash -----------
rc_a, res_a, _ = run_cli("--architecture", "more")
rc_b, res_b, _ = run_cli("--architecture", "more", "--task", "arithmetic")
check("TL1.2f no --task and --task arithmetic are byte-identical through the CLI",
      rc_a == 0 and rc_b == 0
      and canonical_json(res_a) == canonical_json(res_b),
      f"{len(canonical_json(res_a))} vs {len(canonical_json(res_b))} bytes")
check("TL1.2g ... and neither carries a `task` key or a language shape field",
      "task" not in res_a and "task" not in res_b
      and "seq_len" not in res_a["data"] and "n_heads" not in res_a["model"],
      f"top-level keys = {sorted(res_a)}")

rc, res, err = run_cli("--architecture", "more", "--task", "language",
                       "--seed", "44")
check("TL1.2h --seed reaches the language run label",
      rc == 0 and res["logging"]["run_name"] == "langB_MoRE_seed44",
      f"run_name = {res['logging']['run_name']!r}" if res else f"rc={rc}")

# --- the shape flags -------------------------------------------------------
rc, res, err = run_cli("--architecture", "more", "--task", "language",
                       "--seq_len", "512")
check("TL1.2i --seq_len overrides the provisional default (Gate L5's field)",
      rc == 0 and res["data"]["seq_len"] == 512,
      f"seq_len = {res['data']['seq_len']}" if res else f"rc={rc}, {err}")

rc, res, err = run_cli("--architecture", "more", "--seq_len", "512")
check("TL1.2j --seq_len on arithmetic is REFUSED, not ignored",
      rc != 0 and isinstance(err, ValueError) and "--task language" in str(err),
      f"{type(err).__name__}: {str(err)[:60]}")

rc, res, err = run_cli("--architecture", "moe", "--vocab_size", "16384")
check("TL1.2k --vocab_size on arithmetic is REFUSED too",
      rc != 0 and isinstance(err, ValueError) and "--task language" in str(err),
      f"{type(err).__name__}: {str(err)[:60]}")

# uint16 storage: 65536 distinct ids fit, 65537 do not. Checked at the boundary
# because an off-by-one here wraps id 65536 to id 0 -- a valid id for a different
# token, so the corpus is corrupted with no error anywhere.
rc, res, err = run_cli("--architecture", "more", "--task", "language",
                       "--vocab_size", "70000")
check("TL1.2l --vocab_size above the uint16 ceiling is refused",
      rc != 0 and isinstance(err, ValueError) and "uint16" in str(err),
      f"{type(err).__name__}: {str(err)[:60]}")

rc, res, err = run_cli("--architecture", "more", "--task", "language",
                       "--vocab_size", "65536")
check("TL1.2m ... and exactly 65536 is accepted (0..65535 fits)",
      rc == 0 and res["data"]["vocab_size"] == 65536,
      f"vocab_size = {res['data']['vocab_size']}" if res else f"rc={rc}, {err}")

rc, res, err = run_cli("--architecture", "more", "--task", "language",
                       "--seq_len", "0")
check("TL1.2n --seq_len 0 is refused",
      rc != 0 and isinstance(err, ValueError) and ">= 1" in str(err),
      f"{type(err).__name__}: {str(err)[:60]}")

rc, res, err = run_cli("--architecture", "more", "--task", "arithmetics")
check("TL1.2o an unknown --task is refused by argparse (exit 2)",
      rc == 2 and isinstance(err, SystemExit),
      f"rc={rc}, {type(err).__name__}")

# All three arms through the real CLI, MoR included -- the CLI-level form of
# TL1.0u. MoR keeping attention is the claim that makes MoR-vs-MoRE a comparison
# of routing rather than of context access (plan_language.md §6.6).
_cli_arms = {}
for _arch in ARCHITECTURES:
    rc, res, err = run_cli("--architecture", _arch, "--task", "language",
                           "--seed", "42")
    _cli_arms[_arch] = (rc, res, err)
check("TL1.2p all three arms resolve through the CLI with attention on",
      all(rc == 0 and res["model"]["attention"] is True
          for rc, res, err in _cli_arms.values()),
      ", ".join(f"{a}:rc={v[0]}" for a, v in _cli_arms.items()))
check("TL1.2q ... each named langB_<Arch>_seed42",
      [_cli_arms[a][1]["logging"]["run_name"] for a in ("moe", "mor", "more")]
      == ["langB_MoE_seed42", "langB_MoR_seed42", "langB_MoRE_seed42"],
      str([_cli_arms[a][1]["logging"]["run_name"] for a in ARCHITECTURES]))


# ===========================================================================
print("\n--- T-L1.3  the canonical spec is selected by the task")
# ===========================================================================
from more.run_context import (assert_not_silent_proxy, load_canonical_spec,
                              ProxyGuardError, CANONICAL_GROUP,
                              CANONICAL_GROUP_LANGUAGE,
                              CANONICAL_SPEC_PATH,
                              CANONICAL_SPEC_PATH_LANGUAGE)

spec_a = load_canonical_spec()
spec_t = load_canonical_spec(task=TASK_ARITHMETIC)
spec_l = load_canonical_spec(task=TASK_LANGUAGE)

check("TL1.3a load_canonical_spec() with no arguments still reads the "
      "arithmetic spec",
      spec_a == spec_t and spec_a["canonical_group"] == CANONICAL_GROUP
      and spec_a["spec_version"] == "1.1-T8.2-frozen",
      f"group={spec_a['canonical_group']}, version={spec_a['spec_version']}")

# Identity fields, per the ledger's "assert on the loaded dict's identity
# fields": the arithmetic spec still names the arithmetic dataset and the phaseB
# template, so nothing about it was edited to make room for the language one.
check("TL1.3b the arithmetic spec's identity fields are untouched",
      spec_a["run_name_template"] == "phaseB_{architecture}_seed{seed}"
      and spec_a["enforced_fields"]["dataset_version"].startswith("more6-v1-")
      and spec_a["enforced_fields"]["family_cls_weight"] == 0.5
      and "canonical_variant" not in spec_a,
      "template/dataset_version/family_cls_weight unchanged, no new key")

check("TL1.3c the language task loads the language spec",
      spec_l["canonical_group"] == CANONICAL_GROUP_LANGUAGE
      and spec_l["task"] == TASK_LANGUAGE
      and spec_l["run_name_template"] == "langB_{architecture}_seed{seed}",
      f"group={spec_l['canonical_group']}, "
      f"template={spec_l['run_name_template']}")

check("TL1.3d the two groups are different strings",
      CANONICAL_GROUP != CANONICAL_GROUP_LANGUAGE
      and spec_a["canonical_group"] != spec_l["canonical_group"],
      f"{CANONICAL_GROUP!r} vs {CANONICAL_GROUP_LANGUAGE!r}")

check("TL1.3e ... and the two paths are different files that both exist",
      CANONICAL_SPEC_PATH != CANONICAL_SPEC_PATH_LANGUAGE
      and os.path.exists(CANONICAL_SPEC_PATH)
      and os.path.exists(CANONICAL_SPEC_PATH_LANGUAGE))

check("TL1.3f an unregistered task raises instead of falling back to a spec",
      _raises_valueerror(lambda: load_canonical_spec(task="translation"),
                         "canonical spec"),
      "ValueError, no silent fallback to the arithmetic numbers")

# The language spec's clean-run variant must be exactly what resolve_variant
# returns for a clean language config, or the guard would refuse every language
# run for a reason that has nothing to do with the run.
check("TL1.3g the language spec's canonical_variant matches resolve_variant",
      spec_l["canonical_variant"]
      == resolve_variant(apply_task(apply_architecture(
          load_config_defaults({}), "more"), TASK_LANGUAGE)),
      f"spec says {spec_l['canonical_variant']!r}")

# Every key the language spec enforces must be a key _effective() produces.
# A spec key the guard cannot see reports as "requires 8192, run has None" --
# a real check failing for a fake reason.
from more.run_context import _effective

_eff_keys = set(_effective(load_config_defaults({})))
_lang_keys = set(spec_l["enforced_fields"]) | {
    k for a, b in spec_l["architecture_variants"].items()
    if not a.startswith("_") for k in b}
check("TL1.3h every field the language spec enforces is visible to the guard",
      not (_lang_keys - _eff_keys - {"top1_routing"}),
      f"unreadable = {sorted(_lang_keys - _eff_keys - {'top1_routing'})}")


def _claim(group, task=None, **overrides):
    """A resolved config that claims `group`, otherwise as canonical as possible."""
    cfg = load_config_defaults({})
    apply_architecture(cfg, "more")
    if task is not None:
        apply_task(cfg, task)
    cfg.setdefault("logging", {})["experiment_group"] = group
    cfg["data"]["dataset_version"] = "dummy"
    cfg["data"]["train_split_version"] = "dummy"
    cfg.setdefault("provenance", {})["seed"] = 42
    for k, v in overrides.items():
        cfg["model"][k] = v
    try:
        return None, assert_not_silent_proxy(cfg)
    except ProxyGuardError as e:
        return str(e), None


# THE check the ledger names: wrong group for the task.
_msg, _grp = _claim(CANONICAL_GROUP, task=TASK_LANGUAGE)
check("TL1.3i a language run claiming canonical_phase_b is REFUSED",
      _msg is not None and "WRONG GROUP FOR THE TASK" in _msg,
      "refused" if _msg else f"admitted as {_grp!r}")
check("TL1.3j ... and the message names the task and the right group",
      _msg is not None and TASK_LANGUAGE in _msg
      and CANONICAL_GROUP_LANGUAGE in _msg,
      "names task='language' and canonical_lang_b" if _msg else "not refused")

# The symmetric case. It is not hypothetical laziness to check it: an arithmetic
# run mislabelled with the language group would otherwise be stamped
# "canonical_lang_b" and land in the language table.
_msg, _grp = _claim(CANONICAL_GROUP_LANGUAGE, task=None)
check("TL1.3k an arithmetic run claiming canonical_lang_b is REFUSED too",
      _msg is not None and "WRONG GROUP FOR THE TASK" in _msg,
      "refused" if _msg else f"admitted as {_grp!r}")

# ... and the right group for the right task must fail for the RIGHT reason:
# the language spec is deliberately unfrozen, so the refusal must be about null
# fields, never about the group.
_msg, _grp = _claim(CANONICAL_GROUP_LANGUAGE, task=TASK_LANGUAGE)
check("TL1.3l a language run claiming canonical_lang_b is refused for "
      "UNFROZEN fields, not for its group",
      _msg is not None and "WRONG GROUP" not in _msg
      and "unfrozen fields (still null)" in _msg,
      "refused on nulls" if _msg else f"admitted as {_grp!r}")
check("TL1.3m ... and the refusal names the language spec, not the arithmetic one",
      _msg is not None and "canonical_spec_language.json" in _msg,
      "names canonical_spec_language.json" if _msg else "not refused")
check("TL1.3n ... listing exactly the enforced_fields nulls",
      _msg is not None and all(
          f in _msg for f in ("epochs", "batch_size", "lr", "weight_decay",
                              "seq_len", "dropout")),
      "epochs/batch_size/lr/weight_decay/seq_len/dropout all named"
      if _msg else "not refused")

# An exploratory label is still free on either task -- the cross-task refusal
# must not have turned every unrecognised group into an error.
_msg, _grp = _claim("langB_smoke", task=TASK_LANGUAGE)
check("TL1.3o an exploratory label is still allowed on language",
      _msg is None and _grp == "langB_smoke",
      f"group={_grp!r}" if _msg is None else "refused an exploratory label")

# And the arithmetic path is bit-for-bit the old behaviour: same refusal, same
# reasons, on the config shape test_phase5_dimensions.py:G4.37 uses.
_msg, _grp = _claim(CANONICAL_GROUP, task=None, router_noise="trainable")
check("TL1.3p the arithmetic guard still refuses router noise (G4.37's case)",
      _msg is not None and "router_noise" in _msg and "WRONG GROUP" not in _msg,
      "refused, naming model.router_noise" if _msg else "let it through")



# ===========================================================================
print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    print("FAILED:")
    for f in FAIL:
        print(f"  - {f}")
sys.exit(1 if FAIL else 0)
