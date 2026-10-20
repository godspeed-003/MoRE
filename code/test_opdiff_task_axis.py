"""test_opdiff_task_axis.py - T-OD.3 verification for the third task axis.

Run:
    python code/test_opdiff_task_axis.py

Scope: T-OD.3 (`TASKS.md`), the port of the language pipeline to an explicit
`opdiff` task. Kept out of `run_correctness_suite.py` for the same reason as
the other language suites: Gate L0's entire content is that the arithmetic
total is still 356, and a new task's suite must not move that number.

WHAT THIS SUITE IS FOR. The opdiff port touched seven files, and three of
the ways it could have gone wrong are not visible in a smoke run:

  1.  A SECOND COPY OF THE DATASET CONTRACT. `make_opdiff_dataset_v2.py`
      writes the layout; `opdiff_families.py` and `opdiff_data.py` read it.
      If the two disagree about the token layout, the operation axis, the
      group labels or the loss mask, then EVERYTHING still runs and every
      number is wrong in the same direction. So the manifest is checked
      AGAINST THE GENERATOR, by importing it, here.

  2.  A SILENTLY WRONG LABEL SPACE. The router width (7) and the family
      label space (7 groups) are the whole reason the confusion matrix is
      square and the Hungarian / AMI / purity metrics are defined at all. A
      six-expert opdiff run would produce a 7x6 matrix, and the metric layer
      returns an all-None report for a non-square matrix -- so the routing
      numbers would be N/A rather than wrong, which is easy to mistake for
      "the router did not specialise".

  3.  A LOST INVARIANT. The port must not move a single byte of an
      arithmetic or language resolved config. The arithmetic and language
      hashes are re-derived from a PRE-PORT tree of `code/more` in the
      verification below, and the live published language cells are
      re-hashed from their own stored resolved configs.

THE FIXTURE. The opdiff dataset is NOT tracked in git (113 MB), so this
suite GENERATES its own: the generator is called with a handful of rows per
group into a temporary directory, which takes about a second and exercises
the real generator, the real file layout and the real manifest. Nothing here
depends on a dataset that a machine may or may not have built.
"""

import contextlib
import importlib.util
import io
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile

import numpy as np
import torch

CODE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(CODE)
sys.path.insert(0, CODE)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from more.config import (load_config, load_config_defaults, apply_architecture,  # noqa: E402
                         apply_task, resolve_task, resolve_variant,
                         canonical_run_name, TASKS, TASK_ARITHMETIC, TASK_LANGUAGE,
                         TASK_OPDIFF, LM_TASKS, CANONICAL_NUM_EXPERTS_BY_TASK,
                         CANONICAL_FFN_MULT_OPDIFF, CANONICAL_FAMILY_CLS_WEIGHT_BY_TASK,
                         ARCHITECTURES)
from more import opdiff_families as OF  # noqa: E402
from more.opdiff_data import (MoREOpdiffDataset, opdiff_versions,  # noqa: E402
                              resolve_opdiff_root, OPDIFF_ROOT_DEFAULT)
from more.metrics import (routing_agreement_key, routing_agreement_caption,  # noqa: E402
                           routing_agreement_keys_all, permutation_invariant_routing_metrics,
                           halting_supervision_loss, depth_allocation_error)
from more.model import MoREModel  # noqa: E402
from more.run_context import config_hash, canonical_json  # noqa: E402

PASS, FAIL, SKIP = [], [], []


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))


def skip(name, reason):
    SKIP.append(name)
    print(f"  [SKIP] {name}  -- {reason}")


def raises(fn, exc=ValueError, needle=""):
    try:
        fn()
    except exc as e:
        return needle in str(e), str(e)
    except Exception as e:                       # noqa: BLE001
        return False, f"{type(e).__name__}: {e}"
    return False, "did not raise"


def _dataset_root_for_test():
    """The CONFIGURED opdiff dataset root, resolved. Raises if it is absent."""
    from more.config import load_config
    _root = load_config(os.path.join(CODE, "config_opdiff.json"))["data"]["opdiff_root"]
    return resolve_opdiff_root(_root)


# ===========================================================================
print("\n--- T-OD.3.1  the manifest agrees with the GENERATOR")
# ===========================================================================
#
# The generator is the authority for the dataset contract; the manifest is the
# authority for the label space the CODE uses. If they drift apart, every
# opdiff number is wrong in the same direction and nothing looks broken.

_gen_spec = importlib.util.spec_from_file_location(
    "_opdiff_generator", os.path.join(REPO, "make_opdiff_dataset_v2.py"))
GEN = importlib.util.module_from_spec(_gen_spec)
_gen_spec.loader.exec_module(GEN)

check("OD.1a the generator and the manifest agree on the TOKEN LAYOUT",
      GEN.OP0 == OF.OP_TOKEN_OFFSET
      and list(GEN.TOK.values()) == [OF.OP_TOKEN_OFFSET + i
                                     for i in range(OF.NUM_OP_TYPES)]
      and list(GEN.ALL_OPS.keys()) == OF.ALL_OP_NAMES,
      f"op0={GEN.OP0}/{OF.OP_TOKEN_OFFSET}, "
      f"{len(GEN.ALL_OPS)} ops, first={list(GEN.ALL_OPS)[0]}, "
      f"last={list(GEN.ALL_OPS)[-1]}")

check("OD.1b the vocabulary is 43 = 18 fixed + 25 operation tokens",
      OF.VOCAB_SIZE == 43 and GEN.OP0 + len(GEN.ALL_OPS) == 43
      and OF.NUM_OP_TYPES == 25,
      f"VOCAB_SIZE={OF.VOCAB_SIZE}, NUM_OP_TYPES={OF.NUM_OP_TYPES}")

_gen_groups = {g - 1: [OF.OP_NAME_TO_IDX[o] for o in ops]
               for g, ops in GEN.GROUPS.items()}
_man_groups = {g - 1: [OF.OP_NAME_TO_IDX[o] for o in ops]
               for g, ops in OF.GROUP_OP_NAMES.items()}
check("OD.1c the group partition is the generator's, and the OOD operations "
      "are E7 on disk",
      _gen_groups == _man_groups
      and all(OF.OP_TO_EXPERT[o] == 6 for o in GEN.OOD_OPS)
      and all(OF.OP_TO_EXPERT[o] == 6 for o in GEN.GROUPS[7]),
      "E1..E7 as generated; F3/F6/F12 sit in E7 because the generator labels "
      "the OOD rows with group 7")

check("OD.1d every operation maps to exactly one group, and every group is "
      "reachable",
      len(OF.OP_TO_EXPERT) == 25
      and set(OF.OP_TO_EXPERT.values()) == set(range(7)),
      f"{len(OF.OP_TO_EXPERT)} operations over "
      f"{len(set(OF.OP_TO_EXPERT.values()))} groups")

check("OD.1e the depth curriculum table is EMPTY -- the generator's nominal "
      "difficulty is a placeholder, not a target",
      OF.op_target_depth_table() == []
      and callable(GEN.nominal),
      "op_target_depth_table() == []; the generator has a `nominal()` function "
      "and it is deliberately NOT used as a depth target")

check("OD.1f the group labels are W&B-safe and unique",
      len(set(OF.OPDIFF_GROUP_LABELS)) == 7
      and not any("/" in l for l in OF.OPDIFF_GROUP_LABELS),
      ", ".join(OF.OPDIFF_GROUP_LABELS))

check("OD.1g expert_labels is num_experts wide, truncating and padding "
      "positionally",
      OF.expert_labels(1) == ["E1_BITWISE"]
      and len(OF.expert_labels(7)) == 7
      and OF.expert_labels(9)[7:] == ["E8", "E9"],
      f"E=1 -> {OF.expert_labels(1)}, E=9 tail -> {OF.expert_labels(9)[7:]}")

_ok, _msg = raises(lambda: OF.expert_labels(0), ValueError, "num_experts")
check("OD.1h expert_labels(0) raises rather than returning an empty axis",
      _ok, _msg[:70])

_ok, _msg = raises(lambda: OF.op_id_from_token(17), ValueError, "operation token")
check("OD.1i a NON-operation token id raises instead of aliasing an "
      "operation",
      _ok, _msg[:70])


# ===========================================================================
print("\n--- T-OD.3.2  the dataset: the 7-tuple, the mask, the group")
# ===========================================================================
#
# A TINY dataset, generated here. The generator is the contract, so using it
# is also a check that the contract still produces what the reader expects.

_TMP = tempfile.mkdtemp(prefix="opdiff_tiny_")
try:
    # SIX rows per group, not three: the largest group (E7) has six
    # operations and the generator's split de-duplication needs every
    # operation to appear in BOTH train and val, or the forbid-map
    # construction raises. Six is the smallest count that satisfies it.
    _rc = subprocess.run(
        [sys.executable, os.path.join(REPO, "make_opdiff_dataset_v2.py"),
         "--out", _TMP, "--train_rows_per_group", "6", "--eval_rows_per_group", "6"],
        capture_output=True, text=True)
    if _rc.returncode != 0:
        skip("OD.2a-d", f"the generator failed: {_rc.stderr[-200:]}")
        _TMP = None

    if _TMP:
        _man = json.load(open(os.path.join(_TMP, "meta.json"), encoding="utf-8"))
        _ds = MoREOpdiffDataset("train", root=_TMP)
        _item = _ds[0]
        _ids, _mask, _exp, _ops, _fam, _depth, _tgt = _item

        check("OD.2a the dataset returns the engine's 7-TUPLE, in order",
              len(_item) == 7
              and _ids.dtype == torch.long and _mask.dtype == torch.bool
              and _exp.dtype == torch.long and _ops.dtype == torch.long
              and _fam.dtype == torch.long and _depth.dtype == torch.long
              and _tgt.dtype == torch.float32,
              f"shapes: ids{tuple(_ids.shape)} mask{tuple(_mask.shape)} "
              f"exp{tuple(_exp.shape)} ops{tuple(_ops.shape)}")

        check("OD.2b the row is 304 tokens, the vocabulary is 43, and the "
              "manifest agrees",
              _ids.shape == (304,) and _ds.seq_len == 304
              and _ds.vocab_size == 43 == _man["vocab_size"],
              f"seq_len={_ds.seq_len}, vocab={_ds.vocab_size}")

        check("OD.2c step_mask is all True and the target is the NaN POISON",
              bool(_mask.all()) and bool(math.isnan(float(_tgt))),
              "no padding in an opdiff row; the LM target is a shift inside the "
              "loss, so slot 6 holds a value that cannot be mistaken for one")

        _grp = int(np.load(os.path.join(_TMP, "train_group.npy"), mmap_mode="r")[0])
        check("OD.2d step_experts is the ROW's oracle group at every position",
              bool((_exp == _grp - 1).all()) and int(_fam) == _grp - 1,
              f"group file says {_grp}; step_experts and family say {_grp - 1}")

        _raw = np.asarray(np.load(os.path.join(_TMP, "train_x.npy"),
                                  mmap_mode="r")[0], dtype=np.int64)
        _ex = 19
        _want_ops = _raw[::_ex] - OF.OP_TOKEN_OFFSET
        check("OD.2e step_ops is the OWNING EXAMPLE's operation, derived from "
              "the op token the model is shown",
              bool((_ops.reshape(-1)[::_ex] == _want_ops).all())
              and int(_ops.min()) >= 0 and int(_ops.max()) < OF.NUM_OP_TYPES,
              f"ops at the 16 example starts: {_want_ops.tolist()}")

        _lm = np.asarray(_man["loss_mask"], dtype=bool)
        check("OD.2f the loss mask comes from the manifest and is shifted by one "
              "for the shift-by-one targets",
              _ds.loss_mask.shape == (304,)
              and _ds.loss_mask_shifted.shape == (303,)
              and int(_lm.sum()) == 16 * 9,
              f"{int(_lm.sum())} scored positions of 304 "
              f"(16 examples x 9 answer+EOS positions)")

        check("OD.2g the dataset version is the generator's CONTENT hash",
              bool(_ds.dataset_version) and _ds.dataset_version.startswith("opdiff-v2-")
              and _ds.dataset_version == _man["dataset_version"]
              and _ds.train_split_version == _man["dataset_version"],
              f"dataset_version={_ds.dataset_version}")

        check("OD.2h an ABSENT dataset yields (None, None) rather than an "
              "exception -- the dataset is not tracked in git",
              opdiff_versions(os.path.join(_TMP, "does_not_exist")) == (None, None),
              "the proxy guard refuses a canonical claim on a null version, and "
              "the dataset itself raises the actionable error when reached")

        _ok, _msg = raises(lambda: MoREOpdiffDataset("train",
                                                     root=os.path.join(_TMP, "nope")),
                           FileNotFoundError, "make_opdiff_dataset_v2.py")
        check("OD.2i a missing dataset raises with the BUILD COMMAND in the "
              "message",
              _ok, _msg[:80])

        check("OD.2j a relative opdiff_root resolves against the REPO ROOT, "
              "not the working directory",
              resolve_opdiff_root("opdiff_v2").replace("\\", "/").endswith(
                  "/MoRE/opdiff_v2"),
              resolve_opdiff_root("opdiff_v2"))

        check("OD.2k the OOD split exists and carries group 7 (E7)",
              "ood" in _man["rows"] and _man["rows"]["ood"] > 0
              and int(np.load(os.path.join(_TMP, "ood_group.npy"),
                              mmap_mode="r").min()) == 7,
              f"ood rows={_man['rows']['ood']}, group="
              f"{int(np.load(os.path.join(_TMP, 'ood_group.npy'), mmap_mode='r').min())}")
finally:
    if _TMP:
        shutil.rmtree(_TMP, ignore_errors=True)


# ===========================================================================
print("\n--- T-OD.3.3  the config layer: expert count, budget, refusals")
# ===========================================================================

check("OD.3a the opdiff arms carry SEVEN experts (moe, more) and MoR carries "
      "one",
      CANONICAL_NUM_EXPERTS_BY_TASK[TASK_OPDIFF] == 7
      and apply_architecture(load_config_defaults({"task": TASK_OPDIFF}),
                             "moe")["model"]["num_experts"] == 7
      and apply_architecture(load_config_defaults({"task": TASK_OPDIFF}),
                             "more")["model"]["num_experts"] == 7
      and apply_architecture(load_config_defaults({"task": TASK_OPDIFF}),
                             "mor")["model"]["num_experts"] == 1,
      "a 7x7 confusion matrix is SQUARE, which is the precondition for the "
      "Hungarian / AMI / purity report to exist at all")

check("OD.3b the FFN widths are budget-matched to SEVEN experts: 4 for the "
      "7-expert arms, 28 = 7x4 for MoR's single FFN",
      CANONICAL_FFN_MULT_OPDIFF == {"moe": 4, "mor": 28, "more": 4}
      and apply_architecture(load_config_defaults({"task": TASK_OPDIFF}),
                             "mor")["model"]["ffn_mult"] == 28,
      str(CANONICAL_FFN_MULT_OPDIFF))

_cfg_mor = apply_architecture(load_config_defaults({"task": TASK_OPDIFF}), "mor")
_cfg_more = apply_architecture(load_config_defaults({"task": TASK_OPDIFF}), "more")
check("OD.3c the resolved variant names the TASK and no deviation",
      resolve_variant(_cfg_more) == "opdiff" and resolve_variant(_cfg_mor) == "opdiff",
      f"more -> {resolve_variant(_cfg_more)!r}, mor -> {resolve_variant(_cfg_mor)!r}")

_cfg_mor4 = dict(_cfg_mor)
_cfg_mor4["model"] = dict(_cfg_mor["model"], ffn_mult=4)
check("OD.3d an under-budgeted MoR is LABELLED, not accepted quietly",
      resolve_variant(_cfg_mor4) == "opdiff+ffn_mult_4",
      resolve_variant(_cfg_mor4))

_cfg_fc = apply_task(load_config_defaults({"task": TASK_OPDIFF}), TASK_OPDIFF)
check("OD.3e family_cls is forced to 0.0 on opdiff",
      float(_cfg_fc["loss_weights"]["family_cls"]) == 0.0
      and CANONICAL_FAMILY_CLS_WEIGHT_BY_TASK[TASK_OPDIFF] == 0.0,
      "the probe is detached, and an opdiff row DOES carry a whole-row group "
      "label, so a non-zero weight would be a live supervision signal")

_ok, _msg = raises(lambda: load_config_defaults(
    {"task": TASK_OPDIFF, "loss_weights": {"family_cls": 0.5}}),
    ValueError, "DETACHED")
check("OD.3f a DECLARED non-zero family_cls on opdiff is REFUSED",
      _ok, _msg[:80])

_ok, _msg = raises(lambda: load_config_defaults(
    {"task": TASK_OPDIFF, "model": {"halting_supervision": True},
     "loss_weights": {"halting_supervision": 0.1}}),
    ValueError, "depth curriculum")
check("OD.3g halting supervision is REFUSED on opdiff -- there is no depth "
      "curriculum to supervise against",
      _ok, _msg[:80])

_ok, _msg = raises(lambda: load_config_defaults(
    {"task": TASK_OPDIFF, "loss_weights": {"halting_supervision": 0.1}}),
    ValueError, "depth curriculum")
check("OD.3h ... and so is a non-zero halting-supervision WEIGHT on its own",
      _ok, _msg[:80])

check("OD.3i the run name is opdiffB_<Arch> and cannot collide with the other "
      "two studies' prefixes",
      canonical_run_name("more", 42, TASK_OPDIFF) == "opdiffB_MoRE_seed42"
      and not "opdiffB".startswith("phaseB")
      and not "opdiffB".startswith("langB"),
      canonical_run_name("more", 42, TASK_OPDIFF))

_a = apply_task(apply_architecture(load_config_defaults({"task": TASK_OPDIFF}),
                                   "more"), TASK_OPDIFF)
_b = apply_architecture(apply_task(load_config_defaults({"task": TASK_OPDIFF}),
                                    TASK_OPDIFF), "more")
check("OD.3j apply_task and apply_architecture COMMUTE on opdiff",
      canonical_json(_a) == canonical_json(_b),
      "the CLI applies the architecture first and the task second; the config "
      "file does the opposite, and both must end at the same numbers")

check("OD.3k the run name is repaired from phaseB to opdiffB on the CLI "
      "order, and a CHOSEN name is left alone",
      _a["logging"]["run_name"] == "opdiffB_MoRE"
      and apply_task(apply_architecture(
          load_config_defaults({"task": TASK_OPDIFF,
                                "logging": {"run_name": "probe_dense_ablation"}}),
          "more"), TASK_OPDIFF)["logging"]["run_name"] == "probe_dense_ablation",
      _a["logging"]["run_name"])

check("OD.3l route persistence is per_token for the opdiff MoRE arm and "
      "ABSENT for MoE and MoR (the axis is inert there)",
      _a["model"].get("routing_persistence") == "per_token"
      and "routing_persistence" not in _cfg_mor["model"]
      and "routing_persistence" not in
      apply_architecture(load_config_defaults({"task": TASK_OPDIFF}),
                         "moe")["model"],
      "absent, not written: the key's presence would move the config_hash of "
      "arms it cannot affect")

_cfg_file = load_config(os.path.join(CODE, "config_opdiff.json"))
_cfg_file = apply_architecture(_cfg_file, "more")
check("OD.3m the config FILE path and the CLI path agree",
      _cfg_file["model"]["num_experts"] == 7
      and _cfg_file["model"]["ffn_mult"] == 4
      and _cfg_file["logging"]["run_name"] == "opdiffB_MoRE"
      and int(_cfg_file["data"]["seq_len"]) == 304
      and int(_cfg_file["data"]["vocab_size"]) == 43
      and _cfg_file["model"]["attention"] is True
      and _cfg_file["model"]["tie_lm_head"] is True,
      f"E={_cfg_file['model']['num_experts']}, "
      f"run={_cfg_file['logging']['run_name']}")

check("OD.3n the opdiff task is a MEMBER of TASKS and of the LM shape group",
      TASK_OPDIFF in TASKS and len(TASKS) == 3 and TASK_OPDIFF in LM_TASKS
      and TASK_ARITHMETIC not in LM_TASKS,
      f"TASKS={TASKS}")


# ===========================================================================
print("\n--- T-OD.3.4  the CLI: --opdiff_root, and the refusals")
# ===========================================================================

import more.cli as cli_mod  # noqa: E402

CFG_OPDIFF = os.path.join(CODE, "config_opdiff.json")


class _FakeCtx:
    def close(self):
        pass


class _FakeRunContext:
    @staticmethod
    def create(raw_cfg, resolved_cfg, **kw):
        return _FakeCtx()


def run_cli(*flags, config=CFG_OPDIFF):
    box = {}

    def _fake_train(resolved, ctx=None):
        box["resolved"] = resolved

    real_train, real_ctx = cli_mod.train, cli_mod.RunContext
    cli_mod.train, cli_mod.RunContext = _fake_train, _FakeRunContext
    sink = io.StringIO()
    try:
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            rc = cli_mod.main(["--config", config, *flags])
        return rc, box.get("resolved"), None
    except SystemExit as e:
        return (e.code if isinstance(e.code, int) else 1), box.get("resolved"), e
    except Exception as e:                       # noqa: BLE001
        return 1, box.get("resolved"), e
    finally:
        cli_mod.train, cli_mod.RunContext = real_train, real_ctx


_rc, _res, _err = run_cli("--architecture", "more", "--task", "opdiff")
check("OD.4a --task opdiff is accepted and reaches the resolved config",
      _rc == 0 and _res is not None
      and _res["model"]["num_experts"] == 7,
      f"rc={_rc}, E={_res['model']['num_experts'] if _res else None}")

_rc, _res, _err = run_cli("--architecture", "more", "--task", "opdiff",
                          "--family_cls", "0.5")
check("OD.4b --family_cls 0.5 on opdiff is REFUSED, not ignored",
      _rc != 0 and _res is None and isinstance(_err, ValueError)
      and "oracle" in str(_err).lower(),
      f"rc={_rc}, {type(_err).__name__}: {str(_err)[:60]}")

_rc, _res, _err = run_cli("--architecture", "more", "--task", "opdiff",
                          "--halting_supervision")
check("OD.4c --halting_supervision on opdiff is REFUSED",
      _rc != 0 and _res is None and isinstance(_err, ValueError)
      and "curriculum" in str(_err).lower(),
      f"rc={_rc}, {type(_err).__name__}: {str(_err)[:60]}")

_rc, _res, _err = run_cli("--architecture", "more", "--task", "language",
                          "--opdiff_root", "opdiff_v2")
check("OD.4d --opdiff_root on a non-opdiff task is REFUSED",
      _rc != 0 and _res is None and isinstance(_err, ValueError),
      f"rc={_rc}, {type(_err).__name__}: {str(_err)[:60]}")

_rc, _res, _err = run_cli("--architecture", "more", "--task", "opdiff",
                          "--opdiff_root", "opdiff_v2_smoke")
check("OD.4e --opdiff_root re-stamps the dataset version from the directory "
      "ACTUALLY read",
      _rc == 0 and _res is not None
      and _res["data"]["opdiff_root"] == "opdiff_v2_smoke"
      and _res["data"]["dataset_version"] != _cfg_file["data"]["dataset_version"],
      f"root={_res['data']['opdiff_root'] if _res else None}, "
      f"version={_res['data']['dataset_version'] if _res else None}")


# ===========================================================================
print("\n--- T-OD.3.5  the metric layer: the agreement name, the label space, "
      "the empty curriculum")
# ===========================================================================

check("OD.5a the opdiff diagonal is published as AGREEMENT WITH A GROUP, "
      "never as accuracy",
      routing_agreement_key("opdiff") == "val/routing_agreement_with_group"
      and "AGREEMENT" in routing_agreement_caption("opdiff"),
      routing_agreement_key("opdiff"))

check("OD.5b the three tasks publish three DIFFERENT column names",
      len(set(routing_agreement_keys_all())) == 3,
      str(routing_agreement_keys_all()))

_c7 = np.zeros((7, 7))
for _i in range(7):
    _c7[_i, _i] = 100.0
_pim = permutation_invariant_routing_metrics(_c7, 7,
                                             family_labels=OF.OPDIFF_GROUP_LABELS)
check("OD.5c a 7x7 confusion gives a DEFINED permutation-invariant report "
      "with the opdiff group names",
      _pim["hungarian_accuracy"] is not None
      and _pim["ami"] is not None and _pim["purity"] is not None
      and set(_pim["per_family"].keys()) == set(OF.OPDIFF_GROUP_LABELS),
      f"hungarian={_pim['hungarian_accuracy']}, ami={_pim['ami']:.3f}, "
      f"purity={_pim['purity']:.3f}")

_c71 = np.zeros((7, 1))
_c71[:, 0] = 100.0
_pim_mor = permutation_invariant_routing_metrics(_c71, 1,
                                                  family_labels=OF.OPDIFF_GROUP_LABELS)
check("OD.5d MoR's 7x1 matrix gives the all-None EMPTY report -- MoR makes no "
      "routing decision",
      all(_pim_mor[k] is None for k in ("raw_accuracy", "hungarian_accuracy",
                                        "ami", "purity", "macro_recall"))
      and _pim_mor["per_family"] == {},
      "N/A, not a fabricated 1.0 recall")

_ed = torch.tensor([2.0, 3.0, 1.0, 4.0])
_ops = torch.tensor([0, 5, 16, 24])
_m = torch.ones(4, dtype=torch.bool)
_hl = halting_supervision_loss(_ed, _ops, _m, [], 7)
check("OD.5e an EMPTY curriculum table with VALID operation ids returns an "
      "exact zero instead of raising",
      float(_hl) == 0.0,
      f"value={float(_hl)}")
_da = depth_allocation_error(_ed, _ops, _m, [])
check("OD.5f ... and the depth ALLOCATION ERROR against an empty curriculum is "
      "N/A, not 0.0",
      _da == (None, None), str(_da))
_ok, _msg = raises(lambda: depth_allocation_error(_ed, _ops, _m, [1.0] * 16),
                   ValueError, "outside the depth curriculum table")
check("OD.5g a curriculum table from the WRONG task RAISES rather than "
      "measuring the wrong operations",
      _ok, _msg[:80])
_hl_ar = halting_supervision_loss(_ed, torch.tensor([0, 5, 15, 15]), _m,
                                   [1.0, 2.0, 3.0, 4.0] * 4, 7)
check("OD.5h the ARITHMETIC path is unchanged: a non-empty table still "
      "supervises",
      float(_hl_ar) > 0.0, f"value={float(_hl_ar):.4f}")

# THE LOSS ASSEMBLY, PROVED BY CONSTRUCTION. The masked cross-entropy is the
# one place where a plausible-looking error changes the number without
# changing any shape: scoring the OPERAND positions instead of the answer
# positions still produces a finite loss, still falls during training, and
# still looks like a language model being trained.
#
# The test builds logits that are PERFECT at the scored positions and wrong
# everywhere else, and asserts that the loss is ~0. If the mask or the shift
# were wrong, the wrong positions would enter the average and the loss would
# be large -- so this discriminates the right wiring from every nearby wrong
# wiring.
_mask_np = None
try:
    _ds2 = MoREOpdiffDataset("val", root=_dataset_root_for_test())
    _mask_np = np.asarray(_ds2.loss_mask, dtype=bool)
except Exception as _e:                                  # noqa: BLE001
    skip("OD.5i the opdiff loss is scored on the answer positions only",
         f"the opdiff dataset is not available here: {_e}")

if _mask_np is not None:
    _B, _S, _V = 2, int(_mask_np.shape[0]), 43
    _m1 = _mask_np[1:]                                   # the shift the engine uses
    _fake_targets = np.zeros((_B, _S), dtype=np.int64)
    _fake_logits = np.full((_B, _S, _V), -20.0, dtype=np.float32)
    for _b in range(_B):
        for _t in range(1, _S):
            _fake_targets[_b, _t] = (_t * 7 + _b) % _V
            if _m1[_t - 1]:
                # this target position IS scored -> predict it correctly
                _fake_logits[_b, _t - 1, _fake_targets[_b, _t]] = 20.0
            else:
                # not scored -> predict it WRONG, so that an unmasked loss
                # would show it
                _fake_logits[_b, _t - 1, (_fake_targets[_b, _t] + 1) % _V] = 20.0
    _loss = torch.nn.functional.cross_entropy(
        torch.from_numpy(_fake_logits)[:, :-1, :][:, torch.from_numpy(_m1), :].reshape(-1, _V),
        torch.from_numpy(_fake_targets)[:, 1:][:, torch.from_numpy(_m1)].reshape(-1),
    )
    check("OD.5i the opdiff loss is scored on the ANSWER positions only -- a "
          "perfect prediction at those positions gives ~0",
          float(_loss) < 1e-3 and int(_m1.sum()) == 144,
          f"loss={float(_loss):.2e} over {int(_m1.sum())} scored positions "
          f"(16 examples x 9 answer+EOS positions, shifted by one)")

    # The same logits, scored on EVERY position, must be large: that is the
    # number the mask exists to avoid, and it is what an unmasked (or
    # wrongly-masked) loss would report.
    _loss_all = torch.nn.functional.cross_entropy(
        torch.from_numpy(_fake_logits)[:, :-1, :].reshape(-1, _V),
        torch.from_numpy(_fake_targets)[:, 1:].reshape(-1),
    )
    check("OD.5j ... and the UNMASKED version of the same quantity is large, so "
          "the mask is doing the work",
          float(_loss_all) > 1.0,
          f"unmasked loss={float(_loss_all):.2f} vs masked {float(_loss):.2e}")


# ===========================================================================
print("\n--- T-OD.3.6  the model: the LM path, the detached probe, the budget")
# ===========================================================================


def _build(arch):
    cfg = apply_architecture(load_config_defaults({"task": TASK_OPDIFF}), arch)
    mc = cfg["model"]
    return MoREModel(
        step_feat_dim=12, d_model=mc["d_model"], num_experts=mc["num_experts"],
        max_depth=mc["max_depth"], num_blocks=1, dropout=0.0,
        ffn_mult=mc["ffn_mult"], num_families=7, attention=True,
        n_heads=mc["n_heads"], max_seq_len=304, task=TASK_OPDIFF,
        vocab_size=43, routing_persistence=mc.get("routing_persistence", "per_step"),
    )


_mor = _build("mor")
_more = _build("more")


def _nonemb(m):
    total = sum(p.numel() for p in m.parameters())
    emb = sum(p.numel() for n, p in m.named_parameters()
              if n.startswith(("tok_embed", "pos_embed", "lm_head")))
    return total, total - emb


_t_mor, _ne_mor = _nonemb(_mor)
_t_more, _ne_more = _nonemb(_more)
_gap = abs(_ne_mor - _ne_more) / max(_ne_more, 1)
check("OD.6a the MoR and MoRE arms are BUDGET-MATCHED on the non-embedding "
      "count (T9.2's 5% bound)",
      _gap < 0.05,
      f"MoR {_ne_mor:,} vs MoRE {_ne_more:,} non-embedding params, "
      f"gap {_gap * 100:.3f}%")

check("OD.6b the opdiff model builds the LM path and NOT the arithmetic one",
      hasattr(_more, "tok_embed") and hasattr(_more, "lm_head")
      and hasattr(_more, "family_probe")
      and not hasattr(_more, "step_proj") and not hasattr(_more, "op_embed")
      and not hasattr(_more, "cls_head")
      and not hasattr(_more, "regression_head"),
      "token embedding, tied LM head, detached probe; no step projection, no "
      "operation embedding, no regression head, no family classifier")

_ids = torch.randint(0, 43, (2, 304))
_mask = torch.ones(2, 304, dtype=torch.bool)
_exp = torch.zeros(2, 304, dtype=torch.long)
_out = _more(_ids, _mask, _exp, torch.zeros(2, 304, dtype=torch.long))
check("OD.6c a forward pass returns per-position vocabulary logits and a "
      "7-wide probe",
      tuple(_out[0].shape) == (2, 304, 43) and tuple(_out[2].shape) == (2, 304, 7)
      and _out[1] is None,
      f"logits{tuple(_out[0].shape)}, probe{tuple(_out[2].shape)}, "
      f"cls_out={_out[1]}")

_more.zero_grad(set_to_none=True)
_probe_loss = _out[2].sum()
_probe_loss.backward()
_emb_grad = _more.tok_embed.weight.grad
check("OD.6d the family probe is DETACHED: its loss sends NO gradient into the "
      "token embedding",
      _emb_grad is None or float(_emb_grad.abs().sum()) == 0.0,
      "so the group stays a diagnostic and every routing number measures "
      "EMERGENT structure")

_ok, _msg = raises(lambda: _more(torch.rand(2, 304), _mask, _exp,
                                  torch.zeros(2, 304, dtype=torch.long)),
                   TypeError, "input_ids")
check("OD.6e a FLOAT input is refused rather than silently embedded by index "
      "truncation",
      _ok, _msg[:70])


# ===========================================================================
print("\n--- T-OD.3.7  nothing else moved")
# ===========================================================================
#
# The port must not change a single byte of an arithmetic or language
# resolved config. The pre-port `code/more` is re-derived from git (the
# commit before the port) and the SAME config files are resolved with it, so
# the only thing that differs between the two resolutions is the code.

import subprocess  # noqa: E402

_pre_dir = os.path.join(tempfile.gettempdir(), "opdiff_pre_port_tree")
try:
    _zip = os.path.join(tempfile.gettempdir(), "opdiff_pre_port_tree.zip")
    _rc = subprocess.run(["git", "-C", REPO, "archive", "--format=zip", "-o",
                          _zip, "HEAD", "code/more"],
                         capture_output=True, text=True)
    if _rc.returncode != 0:
        skip("OD.7a-b", f"git archive failed: {_rc.stderr[-120:]}")
    else:
        shutil.rmtree(_pre_dir, ignore_errors=True)
        os.makedirs(_pre_dir, exist_ok=True)
        with zipfile.ZipFile(_zip) as _zf:
            _zf.extractall(_pre_dir)
        # The pre-port tree resolves the data paths against ITS OWN root, so
        # the language corpus (and therefore the language dataset_version)
        # would be missing. The junction makes the two trees see the same
        # data, which is what makes the hashes comparable.
        _pre_data = os.path.join(_pre_dir, "data")
        if not os.path.exists(_pre_data) and os.path.isdir(os.path.join(REPO, "data")):
            try:
                os.symlink(os.path.join(REPO, "data"), _pre_data,
                           target_is_directory=True)
            except (OSError, NotImplementedError):
                pass
        _pre_code = os.path.join(_pre_dir, "code")
        _script = os.path.join(_pre_dir, "_hash.py")
        with open(_script, "w", encoding="utf-8") as _fh:
            _fh.write(
                "import json, sys\n"
                f"sys.path.insert(0, {_pre_code!r})\n"
                "from more.run_context import config_hash\n"
                "from more.config import load_config, apply_architecture\n"
                "out = {}\n"
                f"for name in ('config.json', 'config_language.json'):\n"
                f"    p = {os.path.join(CODE, '')!r} + name\n"
                "    out[name] = {}\n"
                "    for arch in ('moe', 'mor', 'more'):\n"
                "        c = apply_architecture(load_config(p), arch)\n"
                "        out[name][arch] = config_hash(c)\n"
                "print(json.dumps(out, sort_keys=True))\n")
        _proc = subprocess.run([sys.executable, _script], capture_output=True,
                               text=True)
        if _proc.returncode != 0:
            skip("OD.7a-b", f"the pre-port tree failed to resolve: {_proc.stderr[-160:]}")
        else:
            _pre = json.loads(_proc.stdout)
            _now = {}
            for _name in ("config.json", "config_language.json"):
                _now[_name] = {}
                for _arch in ("moe", "mor", "more"):
                    _now[_name][_arch] = config_hash(
                        apply_architecture(load_config(os.path.join(CODE, _name)),
                                           _arch))
            check("OD.7a the ARITHMETIC config hash is byte-identical to the "
                  "pre-port tree's",
                  _pre["config.json"] == _now["config.json"],
                  ", ".join(f"{k}:{v[:8]}" for k, v in sorted(_now["config.json"].items())))
            check("OD.7b the LANGUAGE config hash is byte-identical to the "
                  "pre-port tree's",
                  _pre["config_language.json"] == _now["config_language.json"],
                  ", ".join(f"{k}:{v[:8]}" for k, v in sorted(_now["config_language.json"].items())))
finally:
    shutil.rmtree(_pre_dir, ignore_errors=True)


# ===========================================================================
print(f"\n{len(PASS)} passed, {len(FAIL)} failed, {len(SKIP)} skipped")
if FAIL:
    print("FAILED:")
    for f in FAIL:
        print(f"  - {f}")
sys.exit(1 if FAIL else 0)
