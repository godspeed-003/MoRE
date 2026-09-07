"""test_language_spec_freeze.py - T-L7.1 / T-L7.1a verification.

Run:
    C:/Users/vedan/anaconda3/python.exe code/test_language_spec_freeze.py

TWO THINGS ARE UNDER TEST, and they are in one file because they fail together.

T-L7.1 -- THE FROZEN PROTOCOL IS ENFORCEABLE IN BOTH DIRECTIONS. The ledger's
Verify is "no null remains in the spec; a language run with the frozen config is
accepted as canonical; the same run with any single protocol field perturbed is
REFUSED". The second clause is the one that can rot silently: a spec whose fields
the guard cannot read (a typo'd key, a field `_effective()` does not surface)
still passes the acceptance half, because a field the guard never compares can
never mismatch. So every enforced field is perturbed INDIVIDUALLY and the refusal
is required to NAME it -- not merely to occur. `test_language_task_axis.py`
TL1.3h checks readability structurally; this file checks it behaviourally, which
is strictly stronger.

T-L7.1a -- THE PROVENANCE DEFECT. Before this fix a `--task language` run recorded
`dataset_version = "more6-v1-seed42-n70000-3c1087b6aad9"` and
`train_split_version = "fixed-file-splits-v1"` -- the ARITHMETIC 70 k-record
dataset -- while reading 526,320 blocks of WikiText. Both strings came from
`config.json`'s shared `data` block; `_apply_language_block` never replaced them;
and `MoRELanguageDataset`, which knows the right answer, is constructed inside
`engine.train()`, long after `RunContext.create` has hashed the config and stamped
provenance. Fourteen language runs on disk carry the wrong id.

WHY THAT DEFECT IS TESTED HERE RATHER THAN IN test_language_data.py. It is not a
dataset bug. The dataset object was always right; the failure was one of ORDERING
-- a measurement reaching the config after the thing that reads the config had
already run. The regression that would reintroduce it is someone moving the
`stamp_language_dataset_versions` call, or adding a `--corpus`-like override after
it, so the test has to exercise the same call ORDER `cli.py` uses. That places it
next to the guard, not next to the loader.

THE DOUBLE-STAMP IS THE SUBTLE HALF. `apply_task` stamps once; `cli.py` stamps
again after `--corpus` is applied, because `--corpus` lands AFTER `apply_task`. A
`--corpus wikitext-2` run stamped only once carries the wikitext-103 version
string -- i.e. a dev-corpus run claiming canonical-corpus provenance, which is
exactly the confusion T-L2.0 (wikitext-2 and wikitext-103 share their validation
split byte-for-byte) makes dangerous. TP4 below is that case.

CORPUS-DEPENDENT CHECKS SKIP RATHER THAN FAIL when `data/lang/<corpus>/` is
absent. `data/lang/**/*.npy` is not tracked, so a fresh clone legitimately has no
corpus, and a hard failure there would report a missing download as a code
defect. The absence path itself IS asserted (TP5): a missing manifest must yield
None, never a guess and never the arithmetic string, because the proxy guard
refuses a canonical claim on a null and silently accepts a wrong non-null.
"""

import copy
import json
import os
import sys

sys.path.insert(0, __file__.rsplit("\\", 1)[0].rsplit("/", 1)[0])

from more.config import (load_config, load_config_defaults, apply_architecture,
                         apply_task, resolve_task, resolve_variant,
                         stamp_language_dataset_versions,
                         TASK_LANGUAGE, ARCHITECTURES)
from more.lang_data import corpus_versions, TRAIN_SPLIT_VERSION, LANG_ROOT
from more.run_context import (assert_not_silent_proxy, ProxyGuardError,
                              load_canonical_spec, config_hash, _effective)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

CODE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(CODE)
LANG_CONFIG = os.path.join(CODE, "config_language.json")

PASS, FAIL, SKIP = [], [], []


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))


def skip(name, why):
    SKIP.append(name)
    print(f"  [SKIP] {name}  -- {why}")


SPEC = load_canonical_spec(task=TASK_LANGUAGE)
CANON_GROUP = SPEC["canonical_group"]


def frozen_config(arch, seed=42, group=None):
    """The resolved config a canonical language run of `arch` actually uses.

    Built through the SAME call order as `cli.py`: load the file, apply the
    architecture, apply the task, re-stamp the corpus versions. Hand-building a
    dict here would test this file's idea of the protocol rather than the one
    `train.py` resolves, which is the failure mode `load_config_defaults`'
    docstring warns about.
    """
    cfg = load_config(LANG_CONFIG)
    apply_architecture(cfg, arch)
    apply_task(cfg, TASK_LANGUAGE)
    stamp_language_dataset_versions(cfg)          # cli.py's second call
    cfg.setdefault("provenance", {})["seed"] = seed
    cfg.setdefault("logging", {})["experiment_group"] = group or CANON_GROUP
    return cfg


def claim(cfg):
    """(error message or None, granted group or None)."""
    try:
        return None, assert_not_silent_proxy(copy.deepcopy(cfg))
    except ProxyGuardError as e:
        return str(e), None


CORPUS = "wikitext-103"
HAVE_CORPUS = os.path.exists(
    os.path.join(LANG_ROOT, CORPUS, "dataset_meta.json"))

print("\n=== T-L7.1  the frozen spec has no unfrozen field left ===")

_nulls = sorted(k for k, v in SPEC["enforced_fields"].items() if v is None)
_arch_nulls = sorted(
    f"{a}.{k}" for a, b in SPEC["architecture_variants"].items()
    if not a.startswith("_") for k, v in b.items() if v is None)
check("TL7.1a enforced_fields is fully frozen",
      not _nulls, f"nulls = {_nulls}" if _nulls else "0 nulls")
check("TL7.1b all three architecture_variants blocks are fully frozen",
      not _arch_nulls, f"nulls = {_arch_nulls}" if _arch_nulls else "0 nulls")
check("TL7.1c protocol.val_interval is frozen",
      SPEC["protocol"].get("val_interval") is not None,
      f"val_interval = {SPEC['protocol'].get('val_interval')}")

# A frozen number with no justification is indistinguishable from an inherited
# guess, which is the single failure this file exists to prevent (spec _NULLS_NOTE).
_no_why = sorted(set(SPEC["enforced_fields"]) - set(SPEC["frozen_by"]))
check("TL7.1d every enforced field carries a frozen_by entry",
      not _no_why, f"missing = {_no_why}" if _no_why else "1:1")
_thin = sorted(k for k, v in SPEC["frozen_by"].items() if len(str(v)) < 40)
check("TL7.1e ... and none of them is a placeholder",
      not _thin, f"suspiciously short = {_thin}" if _thin else "all substantive")
check("TL7.1f spec_version records the freeze",
      SPEC["spec_version"] == "L7.1-language-frozen",
      f"spec_version = {SPEC['spec_version']!r}")


print("\n=== T-L7.1  the frozen config is ACCEPTED, on all three arms ===")

for arch in ARCHITECTURES:
    cfg = frozen_config(arch)
    if not HAVE_CORPUS:
        # dataset_version is stamped from the manifest, so with no corpus it is
        # None and the guard correctly refuses. That is the right behaviour, not
        # an acceptance failure -- assert it explicitly and move on.
        msg, grp = claim(cfg)
        skip(f"TL7.1g[{arch}] frozen config accepted as {CANON_GROUP}",
             f"data/lang/{CORPUS}/ not built on this machine; guard "
             f"{'refused on the null dataset_version (correct)' if msg else 'ADMITTED IT (wrong)'}")
        check(f"TL7.1g2[{arch}] with no corpus the claim is refused, not guessed",
              msg is not None and "dataset_version" in msg,
              "refused, naming dataset_version" if msg else "admitted without a corpus")
        continue
    msg, grp = claim(cfg)
    check(f"TL7.1g[{arch}] the frozen config is accepted as {CANON_GROUP}",
          msg is None and grp == CANON_GROUP,
          f"group={grp!r}" if msg is None else f"REFUSED: {msg.splitlines()[0]}")
    check(f"TL7.1h[{arch}] ... and resolves to the canonical variant",
          resolve_variant(cfg) == SPEC["canonical_variant"],
          f"variant={resolve_variant(cfg)!r}")
    check(f"TL7.1i[{arch}] ... on the language task",
          resolve_task(cfg) == TASK_LANGUAGE, f"task={resolve_task(cfg)!r}")


print("\n=== T-L7.1  ONE perturbed protocol field is REFUSED, and NAMED ===")

# Where each enforced field lives in the resolved config, so a perturbation lands
# in the same slot the guard reads. Derived from `_effective`, not guessed: a
# field this map placed in the wrong section would be perturbed where nobody looks
# and the refusal check would pass for the wrong reason.
SECTION = {
    "epochs": "training", "batch_size": "training", "lr": "training",
    "weight_decay": "training", "grad_clip": "training",
    "dropout": "model", "d_model": "model", "n_heads": "model",
    "attention": "model", "tie_lm_head": "model", "num_blocks": "model",
    "routing_mode": "model", "router_noise": "model",
    "seq_len": "data", "vocab_size": "data",
    "dataset_version": "data", "train_split_version": "data",
    # `data`, NOT `training`, and this cost a false FAIL while writing the test.
    # subset_fraction is readable in both sections; every CONSUMER reads `data`
    # (engine.py's Subset, this guard, provenance.resolved_subset_fraction), and
    # load_config_defaults reconciles the two AT LOAD TIME. Perturbing the
    # already-resolved `training` copy therefore changes nothing the guard reads
    # and the run is admitted -- which looks like a guard defect and is not one.
    # The operator-facing path is checked separately at TL7.1o/TL7.1o2 below.
    "subset_fraction": "data",
}


def perturb(value):
    """A different value of a type the field will accept."""
    if isinstance(value, bool):
        return not value
    if isinstance(value, (int, float)):
        return type(value)(value * 2 + 1)
    return f"{value}-PERTURBED"


if not HAVE_CORPUS:
    skip("TL7.1j.. one-field perturbation refusals",
         f"needs data/lang/{CORPUS}/; the acceptance baseline is unavailable, so "
         f"a refusal would not distinguish the perturbation from the missing corpus")
else:
    _unreachable = []
    for field, want in sorted(SPEC["enforced_fields"].items()):
        section = SECTION.get(field)
        if section is None:
            # Not perturbable through a config section: halting_mode, task_weight
            # and family_cls_weight are computed or forced. Recorded, not skipped
            # silently -- an enforced field with no route to perturbation is a
            # field this test cannot defend.
            _unreachable.append(field)
            continue
        cfg = frozen_config("more")
        cfg.setdefault(section, {})[field] = perturb(want)
        msg, grp = claim(cfg)
        check(f"TL7.1j[{field}] perturbing {section}.{field} is refused and named",
              msg is not None and field in msg,
              f"refused, names {field}" if msg
              else f"ADMITTED as {grp!r} with {field}={perturb(want)!r}")
    check("TL7.1k every enforced field perturbable through a config section "
          "was actually exercised",
          set(_unreachable) <= {"halting_mode", "task_weight",
                                "family_cls_weight"},
          f"unexercised = {sorted(_unreachable)}")

    # The per-arm blocks, one field each, on the arm they belong to. num_experts
    # and max_depth are the two that define what MoE/MoR/MoRE MEAN, so a spec that
    # stopped enforcing them would admit a 1-expert run calling itself MoRE.
    for arch, field in (("more", "num_experts"), ("more", "max_depth"),
                        ("mor", "max_depth"), ("more", "ffn_mult")):
        want = SPEC["architecture_variants"][arch][field]
        cfg = frozen_config(arch)
        cfg["model"][field] = perturb(want)
        msg, grp = claim(cfg)
        check(f"TL7.1l[{arch}.{field}] perturbing it is refused and named",
              msg is not None and field in msg,
              f"refused, names {field}" if msg else f"ADMITTED as {grp!r}")

    # Loss weights are the T-L7.1 measurement. A spec that froze 0.001 but did not
    # enforce it would let the matrix run at the ratio-matched 0.107 that collapses
    # depth to 1.49 and AMI to 0.0025 -- the whole reason the calibration was run.
    for arch, key, field in (("more", "halting", "halting_weight"),
                             ("more", "routing_balance",
                              "routing_balance_weight")):
        want = SPEC["architecture_variants"][arch][field]
        cfg = frozen_config(arch)
        cfg["loss_weights"][key] = perturb(want)
        msg, grp = claim(cfg)
        check(f"TL7.1m[{arch}.{field}] perturbing loss_weights.{key} is refused",
              msg is not None and (field in msg or key in msg),
              "refused" if msg else f"ADMITTED as {grp!r}")

    # A seed outside the frozen set, and a subset run. Both are Gate 0's original
    # job and must not have been weakened by the language spec's arrival.
    cfg = frozen_config("more", seed=99)
    msg, _ = claim(cfg)
    check("TL7.1n a seed outside seed_set is refused",
          msg is not None and "seed" in msg.lower(),
          "refused" if msg else "admitted seed 99")

    cfg = frozen_config("more")
    cfg["data"]["subset_fraction"] = 0.1
    msg, _ = claim(cfg)
    check("TL7.1o a subset run may not claim canonical",
          msg is not None and "subset" in msg.lower(),
          "refused" if msg else "admitted a 10% subset")

    # THE OPERATOR-FACING VERSION of the same thing, which is the one that would
    # actually happen: someone edits the defaults FILE. Both sections must be
    # unable to sneak a subset past the guard, and they fail differently --
    # `data` mirrors into `training` and the guard refuses; `training` conflicts
    # with the file's own `data: 1.0` and load_config RAISES. Either is a hard
    # stop, and asserting both stops a future "harmless" reconciliation change
    # from turning one of them into a silent 10% canonical run.
    import tempfile

    def _file_variant(section, value):
        raw = json.load(open(LANG_CONFIG, "r", encoding="utf-8"))
        raw.setdefault(section, {})["subset_fraction"] = value
        p = os.path.join(tempfile.mkdtemp(), "config_language_variant.json")
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(raw, fh)
        try:
            c = load_config(p)
        except ValueError as e:
            return "RAISED", str(e)
        apply_architecture(c, "more")
        apply_task(c, TASK_LANGUAGE)
        stamp_language_dataset_versions(c)
        c.setdefault("provenance", {})["seed"] = 42
        c.setdefault("logging", {})["experiment_group"] = CANON_GROUP
        m, g = claim(c)
        return ("REFUSED", m) if m else ("ADMITTED", g)

    _verdict, _why = _file_variant("data", 0.1)
    check("TL7.1o2 a subset written into the defaults file's data block is refused",
          _verdict == "REFUSED" and "subset" in _why.lower(),
          f"{_verdict}")
    _verdict, _why = _file_variant("training", 0.1)
    check("TL7.1o3 ... and one written into its training block is a hard error",
          _verdict == "RAISED" and "subset_fraction" in _why,
          f"{_verdict}: ValueError naming both sections" if _verdict == "RAISED"
          else f"{_verdict}")

    # And the escape hatch still works: exploratory work stays cheap.
    cfg = frozen_config("more", group="langB_smoke")
    cfg["training"]["epochs"] = 1
    msg, grp = claim(cfg)
    check("TL7.1p an exploratory label is unaffected by the freeze",
          msg is None and grp == "langB_smoke", f"group={grp!r}")


print("\n=== T-L7.1a  the corpus's own version strings reach the config ===")

_dsv, _tsv = corpus_versions(CORPUS)
if not HAVE_CORPUS:
    skip("TP1 wikitext-103 stamps its own dataset_version",
         f"data/lang/{CORPUS}/dataset_meta.json absent")
    skip("TP2 ... and train_split_version alongside it", "same reason")
else:
    cfg = frozen_config("more")
    check("TP1 wikitext-103 stamps its own dataset_version",
          cfg["data"]["dataset_version"] == _dsv
          and _dsv == SPEC["enforced_fields"]["dataset_version"],
          f"{cfg['data']['dataset_version']!r}")
    check("TP2 ... and train_split_version alongside it",
          cfg["data"]["train_split_version"] == TRAIN_SPLIT_VERSION,
          f"{cfg['data']['train_split_version']!r}")
    # THE DEFECT, stated as its own assertion so a regression reads as itself
    # rather than as a mismatch somewhere in the guard.
    arith_defaults = load_config(os.path.join(CODE, "config.json"))
    check("TP3 the ARITHMETIC dataset id does not survive into a language config",
          cfg["data"]["dataset_version"]
          != arith_defaults["data"]["dataset_version"],
          f"arithmetic id {arith_defaults['data']['dataset_version']!r} not present")

_dev = "wikitext-2"
if not os.path.exists(os.path.join(LANG_ROOT, _dev, "dataset_meta.json")):
    skip("TP4 --corpus wikitext-2 re-stamps rather than keeping wikitext-103's id",
         f"data/lang/{_dev}/ absent")
else:
    # cli.py's ORDER exactly: apply_task first, THEN the --corpus override, THEN
    # the second stamp. Stamping only inside apply_task leaves the wikitext-103
    # string on a wikitext-2 run -- a dev-corpus run claiming canonical-corpus
    # provenance, which T-L2.0's shared validation split makes actively misleading.
    cfg = load_config(LANG_CONFIG)
    apply_architecture(cfg, "more")
    apply_task(cfg, TASK_LANGUAGE)
    stamped_once = cfg["data"]["dataset_version"]
    cfg["data"]["corpus"] = _dev                       # the --corpus override
    stamp_language_dataset_versions(cfg)               # cli.py's second call
    dev_dsv, _ = corpus_versions(_dev)
    check("TP4 --corpus wikitext-2 re-stamps rather than keeping wikitext-103's id",
          cfg["data"]["dataset_version"] == dev_dsv
          and cfg["data"]["dataset_version"] != stamped_once,
          f"{stamped_once!r} -> {cfg['data']['dataset_version']!r}")
    check("TP4b ... and the two corpora are distinguishable ids, not one string",
          dev_dsv != _dsv, f"{dev_dsv!r} != {_dsv!r}")

# The absence path, asserted rather than assumed. This is the branch a fresh clone
# takes, and the one where a "helpful" default would be most damaging.
_absent_dsv, _absent_tsv = corpus_versions("no-such-corpus-xyz")
check("TP5 an unbuilt corpus yields None, never a guess",
      _absent_dsv is None and _absent_tsv is None,
      f"({_absent_dsv!r}, {_absent_tsv!r})")
check("TP5b ... and the guard refuses a canonical claim on that None",
      True if not HAVE_CORPUS else
      (lambda m: m is not None and "dataset_version" in m)(
          claim({**frozen_config("more"),
                 "data": {**frozen_config("more")["data"],
                          "dataset_version": None,
                          "train_split_version": None}})[0]),
      "a null dataset_version blocks the canonical claim")

# The arithmetic path must be untouched: the stamper is reached only through
# _apply_language_block, so an arithmetic config keeps config.json's own strings
# and every published phaseB_* hash is preserved.
arith = load_config(os.path.join(CODE, "config.json"))
apply_architecture(arith, "more")
_h_before = config_hash(arith)
arith_again = load_config(os.path.join(CODE, "config.json"))
apply_architecture(arith_again, "more")
check("TP6 the arithmetic path never reaches the language stamper",
      arith["data"]["dataset_version"] == "more6-v1-seed42-n70000-3c1087b6aad9"
      and arith["data"]["train_split_version"] == "fixed-file-splits-v1",
      f"{arith['data']['dataset_version']!r}")
check("TP6b ... and its config_hash is stable across resolutions",
      config_hash(arith_again) == _h_before, f"hash8 = {_h_before[:16]}")

# Idempotence. The double call is deliberate; if it were not idempotent the second
# call would be a silent config mutation between the guard and the hash.
if HAVE_CORPUS:
    cfg = frozen_config("more")
    h1 = config_hash(cfg)
    stamp_language_dataset_versions(cfg)
    stamp_language_dataset_versions(cfg)
    check("TP7 stamping is idempotent, so the double call cannot move the hash",
          config_hash(cfg) == h1, f"hash8 = {h1[:16]}")
else:
    skip("TP7 stamping is idempotent", f"data/lang/{CORPUS}/ absent")


print("\n=== T-L7.1  config_language.json and the spec agree ===")

lang_defaults = load_config(LANG_CONFIG)
apply_architecture(lang_defaults, "more")
apply_task(lang_defaults, TASK_LANGUAGE)
eff = _effective(lang_defaults)
_disagree = []
for field, want in SPEC["enforced_fields"].items():
    if field in ("dataset_version", "train_split_version"):
        continue          # stamped from the corpus, checked above
    got = eff.get(field)
    if got != want:
        _disagree.append(f"{field}: config={got!r} spec={want!r}")
check("TL7.1q the defaults file matches the frozen spec field for field",
      not _disagree, "; ".join(_disagree) if _disagree
      else f"{len(SPEC['enforced_fields']) - 2} fields agree")

check("TL7.1r logging.log_interval matches protocol.val_interval",
      int(lang_defaults["logging"]["log_interval"])
      == int(SPEC["protocol"]["val_interval"]),
      f"log_interval = {lang_defaults['logging']['log_interval']}")

# family_cls must be forced to 0.0 by the task block, and it must not be written
# in the defaults file -- an explicit 0.5 there is a copied arithmetic config and
# is refused, and an explicit 0.0 would record a supervision term that does not
# exist (spec:frozen_by.family_cls_weight).
_raw = json.load(open(LANG_CONFIG, "r", encoding="utf-8"))
check("TL7.1s the defaults file does not declare loss_weights.family_cls",
      "family_cls" not in _raw.get("loss_weights", {}),
      "absent, as required")
check("TL7.1t ... and the task block forces it to 0.0 anyway",
      float(lang_defaults["loss_weights"]["family_cls"]) == 0.0,
      "family_cls = 0.0")


print(f"\n{len(PASS)} passed, {len(FAIL)} failed, {len(SKIP)} skipped")
if FAIL:
    print("FAILED:")
    for f in FAIL:
        print(f"  - {f}")
sys.exit(1 if FAIL else 0)
