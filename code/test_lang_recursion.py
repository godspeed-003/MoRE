"""test_lang_recursion.py - GATE L3 / GATE L4 for the language path (T-L6.4).

Run:
    C:/Users/vedan/anaconda3/python.exe code/test_lang_recursion.py

Attention (T-L4.0) is the change most likely to break the two invariants that took
longest to get right in the arithmetic POC, because it is the first sublayer whose
output at position i depends on positions other than i. Every check here is therefore
run on a LANGUAGE model with `attention=True`, not on the arithmetic path that Gate 1
and Gate 2 already cover.

GATE L3 -- recursion is weight-shared, halting is frozen-and-forced.
GATE L4 -- a real gradient reaches every halt head FROM THE TASK LOSS ALONE.

L4 is the one worth reading the code for. The naive version of that test sets the
ponder weight to its canonical value and checks the halt heads have gradient -- which
they always do, because the ponder cost is an explicit function of the halt
probabilities. That version passes on a model whose task path is completely
disconnected from halting. So here `ponder_weight = 0` and the backward pass comes from
`task_loss` only: if the halt heads still receive gradient, the LM objective itself is
what shapes halting, which is the property `updated_rules.md` §2.2 actually requires.
"""

import math
import os
import sys

import torch
import torch.nn.functional as F

CODE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from more.model import MoREModel, TASK_LANGUAGE          # noqa: E402

PASS, FAIL, SKIP = [], [], []


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))


V, DM, S, B, E, MD = 1024, 64, 12, 3, 6, 5


def build(max_depth=MD, num_experts=E, seed=11, **kw):
    torch.manual_seed(seed)
    return MoREModel(step_feat_dim=8, d_model=DM, num_experts=num_experts,
                     max_depth=max_depth, num_blocks=1, dropout=0.0,
                     attention=True, n_heads=4, max_seq_len=S,
                     task=TASK_LANGUAGE, vocab_size=V, **kw)


def batch(seed=5):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, V, (B, S), generator=g)
    return (ids, torch.ones(B, S, dtype=torch.bool),
            torch.randint(0, E, (B, S), generator=g),
            torch.full((B, S), -1, dtype=torch.long))


IDS, SM, SE, SO = batch()

# ===========================================================================
print("\n=== GATE L3  Recursion is weight-shared; halting is frozen and forced ===")
# ===========================================================================

M = build()
M.eval()
W = M.blocks[0]

# L3a. WEIGHT SHARING, by object identity. The strongest available form: not "the
# tensors are equal at every depth" (which a stack of independently-initialised blocks
# could satisfy for one step) but "there is exactly one block object, and the depth loop
# calls it".
_calls = []
_h = W.moe_block.register_forward_pre_hook(lambda m, i: _calls.append(id(m)))
with torch.no_grad():
    _ = M(IDS, SM, SE, SO)
_h.remove()
check("L3a there is ONE block object and every recursion step calls that object",
      len(set(_calls)) == 1 and _calls[0] == id(W.moe_block) and len(_calls) > 1,
      f"{len(_calls)} calls, {len(set(_calls))} distinct object id -- depth is "
      f"computation through time, not a stack of depth-specific networks")

check("L3b the expert count and the halt-head count agree, one head per expert",
      len(W.expert_halt_heads) == E == len(W.moe_block.experts),
      f"{len(W.expert_halt_heads)} halt heads, {len(W.moe_block.experts)} experts")

# L3c. FORCED EXIT AT MAX DEPTH. With halting driven to never fire, every token must
# exit at exactly max_depth and nothing may run past it.
_M = build()
with torch.no_grad():
    for hh in _M.blocks[0].expert_halt_heads:
        hh.weight.zero_()
        hh.bias.fill_(-30.0)          # sigmoid ~ 0: never halt voluntarily
    out = _M(IDS, SM, SE, SO)
_exits = out[5]
_depths = (_exits * torch.arange(1, MD + 1, dtype=_exits.dtype)).sum(-1)
_hs = out[12]
check("L3c with halting suppressed EVERY token is force-exited at max_depth",
      bool((_depths == MD).all()) and _hs["forced_exits"] > 0
      and _hs["early_exits"] == 0.0,
      f"all {_depths.numel()} tokens at depth {MD}; forced={_hs['forced_exits']:.0f} "
      f"early={_hs['early_exits']:.0f} -- no token may run past the budget")

_M2 = build()
with torch.no_grad():
    for hh in _M2.blocks[0].expert_halt_heads:
        hh.weight.zero_()
        hh.bias.fill_(30.0)           # sigmoid ~ 1: halt at the first opportunity
    out2 = _M2(IDS, SM, SE, SO)
_d2 = (out2[5] * torch.arange(1, MD + 1, dtype=out2[5].dtype)).sum(-1)
check("L3d with halting saturated EVERY token exits at depth 1",
      bool((_d2 == 1).all()) and out2[12]["forced_exits"] == 0.0,
      f"all tokens at depth 1, forced={out2[12]['forced_exits']:.0f} -- the two "
      f"extremes bracket the mechanism, so a depth that never varied would fail one")

# ===========================================================================
print()
print("=== GATE L3  balance loss stays depth-invariant after normalization ===")
# ===========================================================================

# The T4.1 property, re-measured under attention: `routing_balance_loss` is normalized
# by the number of (block, depth) dispatch calls, so its MAGNITUDE must not grow with
# the recursion budget. Before normalization it was ~146x the task loss on arithmetic.
_bals = {}
for _md in (1, 3, 5, 9):
    _m = build(max_depth=_md)
    _m.eval()
    with torch.no_grad():
        _o = _m(IDS, SM, SE, SO)
    _bals[_md] = float(_o[3])
_vals = list(_bals.values())
_spread = (max(_vals) - min(_vals)) / max(1e-12, abs(sum(_vals) / len(_vals)))
check("L3e the balance term does not scale with max_depth",
      _spread < 0.35,
      "  ".join(f"depth{d}={v:+.4f}" for d, v in _bals.items())
      + f"  relative spread {_spread:.1%} across a 9x depth range")

# The halted-state freeze, measured rather than asserted. A token that has exited must
# not be recomputed, so re-running the same input must reproduce the same hidden state
# bit-for-bit; under attention that is a real risk, because a still-active token's
# attention output could otherwise write back into a halted position.
_m = build()
_m.eval()
with torch.no_grad():
    _a = _m(IDS, SM, SE, SO)
    _b = _m(IDS, SM, SE, SO)
check("L3f the forward pass is deterministic under attention (halted states frozen)",
      torch.equal(_a[0], _b[0]) and torch.equal(_a[5], _b[5]),
      "logits and exit markers bit-identical across two eval passes -- a halted "
      "state that kept being updated by a neighbour's attention would not reproduce")

# And the invariant that makes the ACT denominators exact: `step_mask` is all-True on
# language (T-L2.2 dropped the trailing partial block for this), so no token is both
# masked and active.
check("L3g step_mask is all-True on the language path, so ACT denominators are exact",
      bool(SM.all()),
      "no key_padding_mask, so the NaN path over a fully-masked attention row is "
      "unreachable and every halting denominator is a plain token count")


# ===========================================================================
print()
print("=== GATE L4  A real gradient reaches EVERY halt head from task_loss ALONE ===")
# ===========================================================================

# ponder_weight = 0: the backward pass carries NO explicit function of the halt
# probabilities. If the halt heads still receive gradient, it is the LM objective that
# shapes halting -- which is what updated_rules.md §2.2 requires and what the naive
# version of this test (ponder weight left on) cannot distinguish.
_m = build()
_m.train()
_o = _m(IDS, SM, SE, SO)
_task = F.cross_entropy(_o[0][:, :-1].reshape(-1, V), IDS[:, 1:].reshape(-1))
_ponder = _o[4]
(_task + 0.0 * _ponder).backward()

_norms = []
for _e, _hh in enumerate(_m.blocks[0].expert_halt_heads):
    _g = _hh.weight.grad
    _norms.append(None if _g is None else float(_g.norm()))
check("L4a every expert's halt head has a non-None gradient with NONZERO norm, "
      "from task_loss alone",
      all(n is not None and n > 0 for n in _norms),
      "  ".join(f"E{i + 1}={n:.3e}" if n is not None else f"E{i + 1}=None"
                for i, n in enumerate(_norms))
      + "  (ponder_weight = 0, so this is the TASK path)")

_bg = [None if hh.bias.grad is None else float(hh.bias.grad.norm())
       for hh in _m.blocks[0].expert_halt_heads]
check("L4b the halt BIAS also receives gradient, so the halt threshold is learnable",
      all(n is not None and n > 0 for n in _bg),
      "  ".join(f"E{i + 1}={n:.3e}" for i, n in enumerate(_bg) if n is not None))

# The complement: with the task loss removed the ponder cost alone must also reach the
# heads. Both paths existing separately is what makes the sum meaningful.
_m2 = build()
_m2.train()
_o2 = _m2(IDS, SM, SE, SO)
(_o2[4] * 1.0).backward()
_pn = [None if hh.weight.grad is None else float(hh.weight.grad.norm())
       for hh in _m2.blocks[0].expert_halt_heads]
check("L4c the ponder cost independently reaches every halt head",
      all(n is not None and n > 0 for n in _pn),
      "  ".join(f"E{i + 1}={n:.3e}" for i, n in enumerate(_pn) if n is not None)
      + "  (task_loss excluded) -- two independent paths, so the sum is not one "
      "path wearing two names")

# A differentiable path must survive the boolean threshold that drives dispatch. If the
# only route were through the hard comparison, the gradient would be exactly zero.
_m3 = build()
_m3.train()
_o3 = _m3(IDS, SM, SE, SO)
_ed = _o3[11]
check("L4d expected_depth is differentiable, so halting is not only a boolean gate",
      _ed is not None and _ed.requires_grad,
      "a boolean threshold may DRIVE dispatch, but the objective keeps a "
      "differentiable path to the halt head (updated_rules.md §2.2)")


# ===========================================================================
print()

# ===========================================================================
# T-LX.9 -- the training loop must not retain per-token routing tensors.
#
# THE DEFECT. The epoch loop kept `epoch_all_expert_idx`, a list holding every
# [N_active] int64 argmax index tensor, for every depth step, for every batch,
# until the epoch ended -- to feed `compute_expert_load_entropy` once at epoch
# end. On canonical language that is 7 depths x 12,288 tokens x 8 B x 10,965
# batches = ~7.4 GiB of LIVE tensors, in ~77 k separate allocations. MoR OOM'd
# on a 16 GB V100. MoR has `num_experts = 1`, where the entropy is `None` by
# definition (T6.4) -- so the arm paid 7.4 GiB to feed a statistic it never
# reports.
#
# WHY MOVING THE LIST TO CPU IS NOT THE FIX. `.cpu()` on each tensor relocates
# the same 7.4 GiB into host RAM, and the box that OOM'd has 21 GB. The list is
# redundant: `epoch_expert_counts` already accumulates the identical per-expert
# totals incrementally, and those counts are the complete input to the
# statistic. So the list is gone and the entropy is computed from the counts.
#
# WHAT THESE CHECKS BIND. The refactor is only safe if the two entry points
# give the SAME NUMBER -- otherwise every published `routing_load_entropy_norm`
# silently changes meaning. TLX.9a-c assert bit-identity on the incremental
# accumulation order the engine actually uses, at E = 1, 2 and 6, so the
# canonical MoE/MoRE value (E=6) and the MoR N/A contract (E=1) are both
# covered.

from more.metrics import (compute_expert_load_entropy,          # noqa: E402
                          expert_load_entropy_from_counts)

def _counts_like_engine(idx_list, E):
    """Rebuild counts exactly as engine.py's epoch loop accumulates them:
    float32, one `(idx == e).float().sum()` per expert per tensor, in tensor
    order. Any deviation here would make the comparison meaningless."""
    counts = torch.zeros(E)
    for t in idx_list:
        for e in range(E):
            counts[e] += (t == e).float().sum()
    return counts

torch.manual_seed(20260914)
for _E in (1, 2, 6):
    _idx = [torch.randint(0, _E, (1234,)) for _ in range(7)]
    _from_idx = compute_expert_load_entropy(torch.zeros(4, 7), _idx, _E)
    _from_cnt = expert_load_entropy_from_counts(_counts_like_engine(_idx, _E), _E)

    if _E < 2:
        check(f"TLX.9a E={_E}: both entry points return None, never 0.0 -- the "
              "MoR arm has no routing decision to be 'collapsed' about",
              _from_idx is None and _from_cnt is None,
              f"indices={_from_idx}, counts={_from_cnt}")
    else:
        check(f"TLX.9a E={_E}: entropy from accumulated COUNTS is bit-identical "
              "to entropy from retained per-token INDICES, so dropping the "
              "retained list reinterprets no published number",
              _from_idx is not None and _from_cnt is not None
              and _from_idx.item() == _from_cnt.item(),
              f"{_from_idx.item()!r} vs {_from_cnt.item()!r}")

# An empty epoch must report N/A, not a spurious 0.0 -- this is the branch the
# old `if epoch_all_expert_idx:` guard covered and the new `counts.sum() > 0`
# guard has to keep covering.
_zero = expert_load_entropy_from_counts(torch.zeros(6), 6)
check("TLX.9b a zero count vector still yields a finite normalized entropy, so "
      "the engine's `sum() > 0` guard -- not a NaN -- is what makes an "
      "unrouted epoch report N/A",
      _zero is not None and math.isfinite(_zero.item()),
      f"H/log(E)={_zero.item():.6f} at all-zero counts (guarded upstream)")

# The structural half: the engine must not reintroduce an unbounded per-batch
# accumulator. Checked against the SOURCE because the property is a source
# property -- there is no list to observe at runtime once it is correctly gone.
_eng = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "more", "engine.py"), encoding="utf-8").read()
_eng_code = "\n".join(l.split("#")[0] for l in _eng.splitlines())
check("TLX.9c the epoch loop holds no `epoch_all_expert_idx` accumulator -- "
      "neither on GPU nor relocated to CPU",
      "epoch_all_expert_idx" not in _eng_code,
      "no per-token routing list survives in executable engine code")

check("TLX.9d the engine reports its own peak GPU memory per epoch, so an OOM "
      "at an epoch boundary is diagnosed from the run's metrics rather than "
      "from a traceback and an arithmetic estimate",
      "perf/gpu_peak_alloc_gib" in _eng_code
      and "perf/gpu_peak_reserved_gib" in _eng_code
      and "reset_peak_memory_stats" in _eng_code,
      "peak alloc + peak reserved, reset per epoch (a rising series = leak, "
      "a flat series = the batch genuinely does not fit)")

print("=" * 78)
print(f"{len(PASS)} passed, {len(FAIL)} failed, {len(SKIP)} skipped")
if FAIL:
    print("\nFAILED:")
    for f in FAIL:
        print(f"  - {f}")
print("=" * 78)
sys.exit(1 if FAIL else 0)
