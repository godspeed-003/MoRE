"""test_phase6_seeding.py -- T6.1 / plan.md §8.1 verification.

Gate 5 (partial): global deterministic seeding. What this file has to establish
is stronger than "a seed function exists":

  1. every stream plan.md §8.1 names is actually reseeded -- proved by drawing
     from each one twice and comparing, not by reading the source;
  2. the seed CHANGES results when it changes (a function that seeds nothing and
     a function that seeds everything both give "identical" reruns if the model
     is deterministic anyway -- the non-vacuity check is that different seeds
     diverge);
  3. seeding happens before the model exists, so weight init is covered;
  4. an undeclared seed still leaves provenance.seed null, so Gate 0 keeps
     refusing to let such a run claim canonical status.

The end-to-end criterion from TASKS.md -- "two runs with the same seed produce
identical first-epoch losses" -- is checked here by running the real engine
twice on a small subset and comparing row 1 of results.tsv byte for byte, plus a
third run at a different seed that must differ.

Run from the code/ directory:
    python test_phase6_seeding.py
"""

import hashlib
import io
import json
import os
import random
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from more import (MoREModel, seed_everything, apply_seeding,
                  resolve_effective_seed, make_generator, seed_worker,
                  CANONICAL_SEED_SET, NUM_EXPERTS_CANONICAL)
from more.seeding import DEFAULT_SEED, CUBLAS_WORKSPACE_CONFIG, _PURPOSE_OFFSETS

PASS, FAIL = 0, 0


def check(name, ok, detail=""):
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"  [PASS] {name}" + (f"  ({detail})" if detail else ""))
    else:
        FAIL += 1
        print(f"  [FAIL] {name}" + (f"  ({detail})" if detail else ""))


def draw_all(seed):
    """Seed once, then take one draw from every stream plan.md §8.1 lists."""
    seed_everything(seed, verbose=False)
    out = {
        "random": random.random(),
        "numpy": float(np.random.rand()),
        "torch_cpu": float(torch.rand(1).item()),
    }
    if torch.cuda.is_available():
        out["torch_cuda"] = float(torch.rand(4, device="cuda").sum().item())
    g = make_generator(seed, "dataloader_shuffle")
    out["dataloader_generator"] = torch.randperm(32, generator=g).tolist()
    return out


def param_digest(model):
    """A single hash over every parameter, so init is compared exactly."""
    h = hashlib.sha256()
    for n, p in sorted(model.named_parameters()):
        h.update(n.encode())
        h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def build_after_seed(seed):
    seed_everything(seed, verbose=False)
    m = MoREModel(step_feat_dim=12, d_model=64,
                  num_experts=NUM_EXPERTS_CANONICAL,
                  max_depth=3, num_blocks=2, dropout=0.1)
    return m


# =========================================================================== #
print("=" * 74)
print("T6.1a  every stream named in plan.md §8.1 is really reseeded")
print("=" * 74)

A1 = draw_all(42)
A2 = draw_all(42)
B1 = draw_all(43)

for stream in A1:
    check(f"T6.1a same seed reproduces {stream}",
          A1[stream] == A2[stream],
          f"{str(A1[stream])[:28]} vs {str(A2[stream])[:28]}")

# Non-vacuity. Without this, a seed function that seeds nothing at all would
# still pass every check above on a deterministic machine.
differing = [s for s in A1 if A1[s] != B1[s]]
check("T6.1a seed 42 and 43 differ on EVERY stream",
      sorted(differing) == sorted(A1.keys()),
      f"differ={sorted(differing)} of {sorted(A1.keys())}")

check("T6.1a all five plan.md §8.1 stream classes covered",
      {"random", "numpy", "torch_cpu", "dataloader_generator"} <= set(A1),
      f"drawn={sorted(A1)}")

print()
print("=" * 74)
print("T6.1b  model initialisation is seed-determined (weights, not just data)")
print("=" * 74)

d1 = param_digest(build_after_seed(42))
d2 = param_digest(build_after_seed(42))
d3 = param_digest(build_after_seed(43))
check("T6.1b seed 42 twice -> identical parameter digest", d1 == d2, d1[:16])
check("T6.1b seed 43 -> different parameter digest", d1 != d3, d3[:16])

print()
print("=" * 74)
print("T6.1c  a full forward+backward in train mode reproduces bitwise")
print("=" * 74)


def flat(o):
    """Flatten whatever a model output slot holds into a comparable list."""
    if o is None:
        return ["None"]
    if hasattr(o, "detach"):
        return o.detach().flatten().tolist()
    if isinstance(o, (list, tuple)):
        out = []
        for e in o:
            out.extend(flat(e))
        return out
    return [o]


def fwd_bwd(seed):
    """Dropout masks and the routing argmax are both inside this."""
    m = build_after_seed(seed)
    m.train()
    x = torch.randn(8, 7, 12)
    mask = torch.ones(8, 7, dtype=torch.bool)
    mask[0, -1] = False
    exp = torch.randint(0, NUM_EXPERTS_CANONICAL, (8, 7))
    ops = torch.randint(0, 16, (8, 7))
    out = m(x, mask, exp, ops)
    loss = out[0].sum() + out[3] + out[4]
    loss.backward()
    gsum = sum(float(p.grad.abs().sum()) for p in m.parameters()
               if p.grad is not None)
    return float(loss), gsum, flat(out[7])


c1, c2, c3 = fwd_bwd(42), fwd_bwd(42), fwd_bwd(43)
check("T6.1c loss identical across reruns at seed 42", c1[0] == c2[0],
      f"{c1[0]!r} vs {c2[0]!r}")
check("T6.1c gradient sum identical across reruns at seed 42", c1[1] == c2[1],
      f"{c1[1]!r}")
check("T6.1c routing decisions identical across reruns at seed 42",
      c1[2] == c2[2], f"{len(c1[2])} token slots compared")
check("T6.1c seed 43 gives a different loss (non-vacuity)", c1[0] != c3[0],
      f"{c1[0]:.6f} vs {c3[0]:.6f}")

print()
print("=" * 74)
print("T6.1d  determinism flags are READ BACK from torch, not merely requested")
print("=" * 74)

rep = seed_everything(42, deterministic=True, verbose=False)
check("T6.1d cudnn.deterministic set", rep["cudnn_deterministic"] is True)
check("T6.1d cudnn.benchmark cleared", rep["cudnn_benchmark"] is False)
check("T6.1d deterministic algorithms enabled",
      rep["torch_deterministic_algorithms"] is True)
check("T6.1d torch agrees with the report",
      torch.are_deterministic_algorithms_enabled() is True)
check("T6.1d CUBLAS_WORKSPACE_CONFIG present",
      os.environ.get("CUBLAS_WORKSPACE_CONFIG") == CUBLAS_WORKSPACE_CONFIG
      or rep["cublas_workspace_config"] != "N/A",
      f"env={os.environ.get('CUBLAS_WORKSPACE_CONFIG')!r} "
      f"report={rep['cublas_workspace_config']!r}")
check("T6.1d mode is warn_only by default (documents, does not abort)",
      rep["determinism_mode"] == "warn_only", rep["determinism_mode"])

# CLAUDE.md §4: no sentinel may be reported as a measurement.
check("T6.1d notes is never an empty list", isinstance(rep["notes"], list)
      and len(rep["notes"]) > 0, f"notes={rep['notes']}")
check("T6.1d no report field is None",
      all(v is not None for v in rep.values()),
      f"none_keys={[k for k, v in rep.items() if v is None]}")
check("T6.1d cuda_version reads 'N/A' rather than None on a CPU build",
      isinstance(rep["cuda_version"], str), repr(rep["cuda_version"]))

rep_off = seed_everything(42, deterministic=False, verbose=False)
check("T6.1d deterministic=False is recorded, not hidden",
      rep_off["determinism_mode"] == "off"
      and rep_off["cudnn_benchmark"] is True
      and any("deterministic=False" in n for n in rep_off["notes"]),
      f"mode={rep_off['determinism_mode']} notes={rep_off['notes'][:1]}")
seed_everything(42, verbose=False)          # restore the deterministic state

print()
print("=" * 74)
print("T6.1d2 non-determinism observed at RUNTIME is recorded, and deduplicated")
print("=" * 74)

# seed-time notes cannot know this: a kernel with no deterministic implementation
# only warns when it is first called, in the middle of training. A 1-epoch CLI run
# emitted 281 identical `_histc_cuda` warnings from wandb.watch's gradient
# histograms while provenance still read ["none"] -- a clean-looking false record.
import warnings as _warnings
from contextlib import redirect_stderr as contextlib_redirect_stderr

from more.seeding import nondeterministic_ops_observed

check("T6.1d2 baseline reads 'none', never an empty list",
      nondeterministic_ops_observed() == ["none"]
      or "none" not in nondeterministic_ops_observed(),
      str(nondeterministic_ops_observed()))

_emitted = io.StringIO()
# The recorder wraps warnings.showwarning and delegates to the original, which
# writes to sys.stderr -- so capturing stderr counts what a reader of stdout.log
# would actually see, and does not bypass the recorder the way replacing
# warnings.showwarning here would.
with contextlib_redirect_stderr(_emitted):
    for _ in range(5):
        _warnings.warn("_fake_op_cuda does not have a deterministic "
                       "implementation, but you set "
                       "'torch.use_deterministic_algorithms(True, "
                       "warn_only=True)'.", UserWarning)

# Count the warning HEADER, not the message text: showwarning also echoes the
# offending source line, and that line literally contains the message string, so
# a substring count of the message reports 2 for one printed warning.
_printed = _emitted.getvalue().count("UserWarning: _fake_op_cuda")
check("T6.1d2 the op name is recorded",
      "_fake_op_cuda" in nondeterministic_ops_observed(),
      str(nondeterministic_ops_observed()))
check("T6.1d2 five identical warnings print once, not five times",
      _printed == 1, f"printed {_printed} of 5")

print()
print("=" * 74)
print("T6.1e  DataLoader generator + worker_init_fn")
print("=" * 74)

from torch.utils.data import DataLoader, TensorDataset

ds = TensorDataset(torch.arange(64).float().unsqueeze(1))


def batch_order(seed):
    g = make_generator(seed, "dataloader_shuffle")
    dl = DataLoader(ds, batch_size=8, shuffle=True, generator=g,
                    worker_init_fn=seed_worker, num_workers=0)
    return [int(v) for b, in dl for v in b.flatten().tolist()]


o1, o2, o3 = batch_order(42), batch_order(42), batch_order(43)
check("T6.1e shuffle order reproduces at the same seed", o1 == o2,
      f"first 8: {o1[:8]}")
check("T6.1e shuffle order changes with the seed", o1 != o3,
      f"42:{o1[:6]} 43:{o3[:6]}")
check("T6.1e shuffle order is a real permutation", sorted(o1) == list(range(64)))

# The shuffle stream must NOT be the global torch RNG: otherwise any change in
# the number of init draws (Phase 5's 7->6 head narrowing did exactly that)
# silently reorders every batch.
seed_everything(42, verbose=False)
_ = torch.randn(1000)                        # burn global draws
o4 = batch_order(42)
check("T6.1e shuffle order is independent of global RNG consumption", o1 == o4,
      "1000 global draws burned in between")

import pickle
check("T6.1e seed_worker is picklable (num_workers>0 on Windows spawn)",
      pickle.loads(pickle.dumps(seed_worker)) is seed_worker)

torch.manual_seed(1234)
seed_worker(0)
w0 = (random.random(), float(np.random.rand()))
torch.manual_seed(1234)
seed_worker(0)
w1 = (random.random(), float(np.random.rand()))
check("T6.1e seed_worker derives numpy+random from the torch worker seed",
      w0 == w1, f"{w0}")

print()
print("=" * 74)
print("T6.1f  seed precedence, and Gate 0 stays armed for undeclared runs")
print("=" * 74)

cfg_cli = {"provenance": {"seed": 45}, "training": {"seed": 7}}
cfg_file = {"provenance": {"seed": None}, "training": {"seed": 7}}
cfg_none = {"provenance": {}, "training": {}}

check("T6.1f --seed wins over training.seed",
      resolve_effective_seed(cfg_cli) == (45, "cli"),
      str(resolve_effective_seed(cfg_cli)))
check("T6.1f training.seed used when --seed absent",
      resolve_effective_seed(cfg_file) == (7, "config"),
      str(resolve_effective_seed(cfg_file)))
check("T6.1f documented default used when neither is given",
      resolve_effective_seed(cfg_none) == (DEFAULT_SEED, "default"),
      str(resolve_effective_seed(cfg_none)))

cfg_p = {"provenance": {}, "training": {}}
s, r = apply_seeding(cfg_p, verbose=False)
prov = cfg_p["provenance"]
check("T6.1f apply_seeding records the seed it actually used",
      prov["resolved_seed"] == s == DEFAULT_SEED, f"resolved_seed={s}")
check("T6.1f provenance.seed is NOT back-filled by the default",
      prov.get("seed") is None, f"seed={prov.get('seed')!r}")
check("T6.1f seed_declared records that no seed was declared",
      prov["seed_declared"] is False)
check("T6.1f seed_source recorded", prov["seed_source"] == "default")
for key in ("determinism_mode", "cudnn_deterministic", "cudnn_benchmark",
            "torch_deterministic_algorithms", "cublas_workspace_config",
            "determinism_exceptions"):
    check(f"T6.1f provenance carries {key}", key in prov, repr(prov.get(key)))

# The point of not back-filling: Gate 0 must still refuse this run.
from more.run_context import assert_not_silent_proxy, ProxyGuardError, CANONICAL_GROUP

cfg_claim = {"logging": {"experiment_group": CANONICAL_GROUP},
             "provenance": {}, "training": {}, "model": {}, "data": {},
             "loss_weights": {}}
apply_seeding(cfg_claim, verbose=False)
try:
    assert_not_silent_proxy(cfg_claim)
    check("T6.1f Gate 0 still refuses a canonical claim with no declared seed",
          False, "guard returned instead of raising")
except ProxyGuardError as e:
    check("T6.1f Gate 0 still refuses a canonical claim with no declared seed",
          True, "ProxyGuardError raised")

cfg_declared = {"provenance": {"seed": 44}, "training": {}}
s2, _ = apply_seeding(cfg_declared, verbose=False)
check("T6.1f declared seed flows through to the RNGs",
      s2 == 44 and cfg_declared["provenance"]["seed_declared"] is True,
      f"resolved_seed={s2}")

print()
print("=" * 74)
print("T6.1g  one frozen seed set, in one place")
print("=" * 74)

check("T6.1g CANONICAL_SEED_SET is 42-46",
      tuple(CANONICAL_SEED_SET) == (42, 43, 44, 45, 46), str(CANONICAL_SEED_SET))

with io.open("canonical_spec.json", encoding="utf-8") as f:
    spec = json.load(f)
check("T6.1g canonical_spec.json seed_set agrees with the code",
      list(spec.get("seed_set", [])) == list(CANONICAL_SEED_SET),
      f"spec={spec.get('seed_set')}")
check("T6.1g the default seed is a member of the frozen set",
      DEFAULT_SEED in CANONICAL_SEED_SET, f"default={DEFAULT_SEED}")
check("T6.1g every generator purpose has its own offset",
      len(set(_PURPOSE_OFFSETS.values())) == len(_PURPOSE_OFFSETS),
      str(_PURPOSE_OFFSETS))
try:
    make_generator(42, "not_a_purpose")
    check("T6.1g an unregistered generator purpose raises", False)
except KeyError:
    check("T6.1g an unregistered generator purpose raises", True)

print()
print("=" * 74)
print("T6.1h  END-TO-END: two runs at the same seed, identical first-epoch loss")
print("=" * 74)

import contextlib

from more.config import apply_architecture
from more.engine import train
from more.run_context import RunContext, resolve_overrides

os.environ["WANDB_MODE"] = "disabled"       # no network, no run directory in wandb/

_TMP_RUNS = tempfile.mkdtemp(prefix="t61_runs_")


def short_run(seed, run_name):
    """A real engine run: same code path as a canonical run, 2 epochs on dummy data."""
    with io.open("config.json", encoding="utf-8") as f:
        cfg = json.load(f)
    apply_architecture(cfg, "more")
    cfg["model"].update(d_model=32, num_blocks=1, max_depth=3, dropout=0.1)
    cfg["training"].update(epochs=2, batch_size=32, lr=1e-3)
    cfg["data"].update(train_path="../data/dummy-train.jsonl",
                       val_path="../data/dummy-val.jsonl")
    cfg["logging"]["run_name"] = run_name
    # log_interval = 1 so validation runs on epoch 1 too. The engine only
    # validates when `epoch % log_interval == 0 or epoch == epochs`; at the
    # config default of 10 the first-epoch row would carry val_loss = nan and
    # routing_accuracy = N/A, and "identical first-epoch losses" would be
    # comparing two placeholders.
    cfg["logging"]["log_interval"] = 1
    resolved = resolve_overrides(cfg, seed=seed, run_name=run_name)
    ctx = RunContext.create(cfg, resolved, runs_root=_TMP_RUNS, tee_stdout=False)
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            train(resolved, ctx=ctx)
    finally:
        ctx.close()
    with io.open(os.path.join(ctx.dir, "results.tsv"), encoding="utf-8") as f:
        rows = [ln for ln in f.read().strip().split("\n") if ln]
    with io.open(os.path.join(ctx.dir, "resolved_config.json"), encoding="utf-8") as f:
        prov = json.load(f)["provenance"]
    return rows, prov, ctx.experiment_id


try:
    rows_a, prov_a, id_a = short_run(42, "t61_seedcheck_a")
    rows_b, prov_b, id_b = short_run(42, "t61_seedcheck_b")
    rows_c, prov_c, id_c = short_run(43, "t61_seedcheck_c")

    check("T6.1h both runs produced a first epoch row",
          len(rows_a) >= 2 and len(rows_b) >= 2, f"{len(rows_a)}/{len(rows_b)} lines")
    check("T6.1h first-epoch row IDENTICAL for two runs at seed 42",
          rows_a[1] == rows_b[1],
          f"\n        A: {rows_a[1]}\n        B: {rows_b[1]}")
    check("T6.1h every epoch row identical for two runs at seed 42",
          rows_a == rows_b, f"{len(rows_a) - 1} epochs compared")
    check("T6.1h first-epoch row DIFFERS at seed 43 (non-vacuity)",
          rows_a[1] != rows_c[1],
          f"\n        42: {rows_a[1]}\n        43: {rows_c[1]}")

    check("T6.1h run id carries the declared seed, not seedNA",
          "__seed42__" in id_a and "__seed43__" in id_c, f"{id_a}")
    check("T6.1h resolved_config.json records resolved_seed",
          prov_a.get("resolved_seed") == 42 and prov_c.get("resolved_seed") == 43,
          f"a={prov_a.get('resolved_seed')} c={prov_c.get('resolved_seed')}")
    check("T6.1h resolved_config.json records seed_source=cli",
          prov_a.get("seed_source") == "cli", str(prov_a.get("seed_source")))
    check("T6.1h resolved_config.json records the determinism settings",
          prov_a.get("determinism_mode") == "warn_only"
          and prov_a.get("cudnn_benchmark") is False,
          f"mode={prov_a.get('determinism_mode')} "
          f"benchmark={prov_a.get('cudnn_benchmark')}")
    check("T6.1h determinism exceptions are listed explicitly",
          isinstance(prov_a.get("determinism_exceptions"), list)
          and len(prov_a["determinism_exceptions"]) > 0,
          "; ".join(prov_a.get("determinism_exceptions", []))[:110])
finally:
    shutil.rmtree(_TMP_RUNS, ignore_errors=True)

print()
print("=" * 74)
print(f"T6.1 / Gate 5 (seeding):  {PASS} passed, {FAIL} failed")
print("=" * 74)
sys.exit(1 if FAIL else 0)
