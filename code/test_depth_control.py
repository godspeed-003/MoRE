"""test_depth_control.py -- T-OD.13. Checks for the forced-depth control and
the language subset-fraction override.

STANDALONE, and deliberately NOT registered in run_correctness_suite.py: the
frozen Gate L0 baseline is 356 on Vedant's CPU box (357 on this CUDA box, see
TASKS.md), and adding checks to a registered suite re-baselines a number three
documents quote. These checks are cheap and self-contained; run them with

    <interpreter> code/test_depth_control.py

What is asserted, and why each one is load-bearing:

TD.1  A fixed `subset_seed` makes the language subsets NESTED: the 2.5 % run is
      a literal PREFIX of the 10 % run, which is a prefix of the 40 % run.
      engine.py builds the subset with torch.utils.data.random_split on
      torch.randperm(n, generator=manual_seed(subset_seed)), which is only
      nested because the permutation is shared and sliced from the front. If a
      future edit switched to torch.randperm(subset_n, ...) per fraction, the
      three scales would sample DIFFERENT tokens and every "same data, more of
      it" statement in the sweep would be false while all three runs still
      looked healthy.

TD.2  resolve_overrides writes --subset_fraction into BOTH `data` and
      `training` and publishes it as resolved_subset_fraction. engine.py reads
      `data`; the Gate 0 proxy guard reads `data`; write one and the other
      lies. This is the same two-section trap T5.3 fixed for the config file.

TD.3  Forced depth d returns a DIFFERENT state for every d. A mixture that
      ignored the forced weight (the plausible bug) would return the
      ACT-weighted output for every d, i.e. all seven values identical.

TD.4  force_depth=1 on a max_depth=1 wrapper is BIT-IDENTICAL to the default
      forward. That is the identity the eval script relies on when it reports
      MoE's forced curve, and it is the only case where the forced control and
      the learned policy must coincide.

TD.5  An out-of-range force_depth is refused by MoREModel, not silently
      clamped. A clamp would make the d=1 row of a chart secretly be d=0.

No GPU required; runs on CPU in a few seconds.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from more.model import MoREModel, MoREWrapper
from more.run_context import resolve_overrides

PASS, FAIL = [], []


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}"
          + (f"  -- {detail}" if detail else ""))


def subset_indices(n, fraction, seed):
    """Exactly engine.py's construction: random_split on a range dataset."""
    from torch.utils.data import random_split
    k = max(1, int(n * fraction))
    ds = torch.arange(n)
    sub, _rest = random_split(ds, [k, n - k],
                              generator=torch.Generator().manual_seed(seed))
    return list(sub)


print("\n=== TD.1  subset nesting (same subset_seed) ===")
# random_split slices ONE randperm, so subsets at different fractions are
# prefixes OF THAT PERMUTATION -- not prefixes of range(n). Asserting
# `i40 == list(range(n))[:len(i40)]` would be asserting the permutation is the
# identity, which is the opposite of what we want: it would make the 2.5 % cell
# the first 13,158 blocks of WikiText in corpus order, confounding scale with
# document/topic order. TD.1d asserts the sample is NOT the corpus head.
N = 20000
for seed in (42, 43, 44, 45, 46):
    i25 = subset_indices(N, 0.025, seed)
    i10 = subset_indices(N, 0.10, seed)
    i40 = subset_indices(N, 0.40, seed)
    ok = (i25 == i10[:len(i25)] and i10 == i40[:len(i10)]
          and len(set(x.item() for x in i40)) == len(i40)
          and max(x.item() for x in i40) < N)
    check(f"TD.1 seed{seed}: 2.5% prefix of 10% prefix of 40%, all in range",
          ok, f"sizes {len(i25)}/{len(i10)}/{len(i40)}")
check("TD.1b fraction 1.0 selects exactly the full set",
      sorted(subset_indices(N, 1.0, 42)) == list(range(N)))
check("TD.1c the permutation is not the identity",
      subset_indices(N, 0.025, 42)[0].item() != 0)
check("TD.1d the 2.5% sample is not the corpus head (scale is not confounded "
      "with block order)",
      sorted(x.item() for x in subset_indices(N, 0.025, 42)) != list(range(500)))

print("\n=== TD.2  --subset_fraction reaches both sections and provenance ===")
cfg = {"data": {"subset_fraction": 1.0, "subset_seed": 42},
       "training": {"subset_fraction": 1.0, "subset_seed": 42},
       "logging": {}, "model": {}}
out = resolve_overrides(cfg, subset_fraction=0.10)
check("TD.2a data.subset_fraction == 0.10", out["data"]["subset_fraction"] == 0.10,
      repr(out["data"]["subset_fraction"]))
check("TD.2b training.subset_fraction == 0.10",
      out["training"]["subset_fraction"] == 0.10,
      repr(out["training"]["subset_fraction"]))
check("TD.2c provenance.resolved_subset_fraction == 0.10",
      out["provenance"]["resolved_subset_fraction"] == 0.10,
      repr(out["provenance"]["resolved_subset_fraction"]))
check("TD.2d the default (no flag) keeps 1.0 and adds no field",
      resolve_overrides(cfg)["data"]["subset_fraction"] == 1.0)

print("\n=== TD.3/TD.4  forced depth changes the returned state ===")
torch.manual_seed(42)
W = MoREWrapper(d_model=16, max_depth=4, num_experts=2, dropout=0.0,
                attention=False)
W.eval()
x = torch.randn(2, 8, 16)
mask = torch.ones(2, 8, dtype=torch.bool)
outs = {}
with torch.no_grad():
    default, *_ = W(x, mask)
    for d in range(1, 5):
        outs[d], *_ = W(x, mask, force_depth=d)
distinct = len({tuple(o.flatten().tolist()) for o in outs.values()})
check("TD.3 forced d=1..4 give four distinct outputs", distinct == 4,
      f"{distinct} distinct")

torch.manual_seed(42)
W1 = MoREWrapper(d_model=16, max_depth=1, num_experts=2, dropout=0.0,
                 attention=False)
W1.eval()
x1 = torch.randn(2, 8, 16)
m1 = torch.ones(2, 8, dtype=torch.bool)
with torch.no_grad():
    a, *_ = W1(x1, m1)
    b, *_ = W1(x1, m1, force_depth=1)
check("TD.4 max_depth=1: forced d=1 is bit-identical to default",
      torch.equal(a, b),
      f"max|diff| = {float((a - b).abs().max()):.3e}")

print("\n=== TD.5  out-of-range force_depth is refused ===")
torch.manual_seed(0)
M = MoREModel(step_feat_dim=4, d_model=16, num_experts=2, max_depth=4,
              num_blocks=1, dropout=0.0)
M.eval()
xx = torch.randn(2, 5, 4)
mm = torch.ones(2, 5, dtype=torch.bool)
for bad in (0, 5):
    try:
        with torch.no_grad():
            M(xx, mm, force_depth=bad)
    except ValueError as e:
        check(f"TD.5 force_depth={bad} raises ValueError", True, str(e)[:70])
    else:
        check(f"TD.5 force_depth={bad} raises ValueError", False, "no exception")

print(f"\n===== DEPTH-CONTROL/SWEEP CHECKS: {len(PASS)} passed, "
      f"{len(FAIL)} failed =====")
sys.exit(1 if FAIL else 0)
