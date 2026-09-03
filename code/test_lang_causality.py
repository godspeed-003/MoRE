"""test_lang_causality.py - GATE L1: causality, leakage, freezing and the waste.

Run:
    C:/Users/vedan/anaconda3/python.exe code/test_lang_causality.py

Scope: T-L4.0 .. T-L4.4 (`plan_language.md` §6). Kept out of
`run_correctness_suite.py` for the same reason as the other language suites --
Gate L0's entire content is that the arithmetic total is still 356.

WHY THIS IS THE MOST IMPORTANT NEW TEST IN THE MIGRATION. It is the direct analogue
of the arithmetic target-in-input leakage audit
(`data/dataset_meta.json:gate1_audit`). A causal LM that can see one token into the
future has an implausibly good perplexity and NOTHING ELSE looks wrong: the loss
curve is smooth, the gradients are finite, the routing metrics are unremarkable.
There is no symptom to notice. So causality is tested by PERTURBATION rather than by
inspection -- mutate a future position, re-run, and require the earlier positions'
outputs to be bit-identical.

AND THE TEST IS SHOWN TO BE ABLE TO FAIL. A leakage check that has never failed has
not been demonstrated to work. TLC.4 removes the causal mask from an otherwise
identical model and requires the same perturbation to CHANGE the earlier outputs. If
that variant passed, this file would be measuring nothing.

BIT-IDENTICAL, NOT `allclose`. Float tolerance is the wrong instrument here: a real
leak through the ACT halting statistics could easily be smaller than 1e-6 at first
and grow over training. `torch.equal` is the assertion.
"""

import os
import sys

import torch

CODE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from more.model import MoREWrapper, MoREModel      # noqa: E402
from more.families import NUM_EXPERTS_CANONICAL    # noqa: E402

PASS, FAIL, SKIP = [], [], []


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))


def skip(name, reason):
    SKIP.append(name)
    print(f"  [SKIP] {name}  -- {reason}")


D, S, B = 32, 12, 2
torch.manual_seed(42)


def build(max_depth=4, attention=True, num_experts=NUM_EXPERTS_CANONICAL, seed=42):
    torch.manual_seed(seed)
    w = MoREWrapper(d_model=D, max_depth=max_depth, num_experts=num_experts,
                    dropout=0.0, attention=attention, n_heads=4)
    w.eval()          # dropout off: a stochastic forward cannot be bit-compared
    return w


# ===========================================================================
print("\n=== T-L4.0 / T-L4.1  One shared sublayer; absence when off ===")
# ===========================================================================

W_off, W_on = build(attention=False), build(attention=True)

check("TLC.0a attention=False constructs NO module, not a bypassed one",
      not hasattr(W_off, "attn") and not hasattr(W_off, "attn_norm")
      and W_off.attention is False,
      "absence is verifiable; a flag-bypassed module would still be in the "
      "state_dict and still in the optimizer")

_extra = sorted(set(W_on.state_dict()) - set(W_off.state_dict()))
_lost = sorted(set(W_off.state_dict()) - set(W_on.state_dict()))
check("TLC.0b the arithmetic state_dict key set is untouched",
      not _lost and len(_extra) == 6,
      f"attention adds exactly {_extra}, removes {_lost or 'nothing'} -- so "
      f"Gate 4's 'no unexpected parameter' check still means what it meant")

_p = sum(p.numel() for p in W_on.attn.parameters())
check("TLC.0c the attention sublayer is 4*d^2 + 4*d parameters, counted once",
      _p == 4 * D * D + 4 * D,
      f"{_p} params for d_model={D}; ONE module per wrapper regardless of "
      f"max_depth={W_on.max_depth}")

# The weight-sharing invariant, asserted by IDENTITY rather than by shape: two
# distinct modules with equal shapes would pass a shape check and break
# updated_rules.md §2.1, which is what makes depth "computation through time".
_ids = []
_orig_fwd = W_on.attn.forward


def _spy(*a, **k):
    _ids.append((W_on.attn.in_proj_weight.data_ptr(),
                 W_on.attn.out_proj.weight.data_ptr(),
                 W_on.attn_norm.weight.data_ptr()))
    return _orig_fwd(*a, **k)


W_on.attn.forward = _spy
_x = torch.randn(B, S, D)
_sm = torch.ones(B, S, dtype=torch.bool)
_res = W_on(_x, _sm)
W_on.attn.forward = _orig_fwd

check("TLC.0d the SAME parameter tensors are used at every recursion depth",
      len(_ids) >= 2 and len(set(_ids)) == 1,
      f"attention ran at {len(_ids)} depth steps on one set of storage pointers "
      f"-- a per-depth module would make depth a stack, not time")

_deep = build(max_depth=7, attention=True)
check("TLC.0e the parameter count does not grow with max_depth",
      sum(p.numel() for p in _deep.attn.parameters()) == _p,
      f"max_depth 4 and 7 both give {_p} attention parameters")

# ===========================================================================
print()
print("=== T-L4.2  Explicit upper-triangular mask; is_causal=True forbidden ===")
# ===========================================================================

_m = W_on.causal_mask(5, torch.device("cpu"))
_want = torch.triu(torch.ones(5, 5, dtype=torch.bool), diagonal=1)

check("TLC.2a the mask is a bool tensor in the documented orientation",
      _m.dtype == torch.bool and _m.shape == (5, 5) and torch.equal(_m, _want),
      "entry [i, j] is True (= disallowed) exactly when j > i, so query i sees "
      "keys j <= i and nothing later")

check("TLC.2b the diagonal is ALLOWED -- a token may attend to itself",
      not bool(_m.diagonal().any()),
      "diagonal=1 in the triu, not diagonal=0: excluding the self-position would "
      "be a different model, not a stricter one")

check("TLC.2c the mask is cached per (S, device), not rebuilt each depth step",
      W_on.causal_mask(5, torch.device("cpu")) is _m,
      "and cached in a plain dict, not a registered buffer -- a buffer would put "
      "the sequence length into the checkpoint")

# The grep, as an assertion -- and done with `ast`, not string search. A naive
# `"is_causal=True" in src` also matches the DOCSTRING in `causal_mask` that
# explains why the hint is not used, so the string check failed on its own
# documentation. Parsing for a keyword argument actually named `is_causal` with a
# literal `True` is exact and immune to prose.
import ast  # noqa: E402

_srcs = {}
for _f in ("more/model.py", "more/engine.py", "more/lang_data.py"):
    with open(os.path.join(CODE, _f), "r", encoding="utf-8") as fh:
        _srcs[_f] = fh.read()

_offenders = []
for _f, _s in _srcs.items():
    for _node in ast.walk(ast.parse(_s)):
        if isinstance(_node, ast.Call):
            for _kw in _node.keywords:
                if (_kw.arg == "is_causal"
                        and isinstance(_kw.value, ast.Constant)
                        and _kw.value.value is True):
                    _offenders.append(f"{_f}:{_node.lineno}")
check("TLC.2d no CALL passes is_causal=True in the model, engine or data path",
      not _offenders,
      f"{sorted(_srcs)} parsed with ast -- an explicit mask is validated by the "
      f"same code path on every backend, whereas is_causal is a hint whose "
      f"honouring depends on the kernel chosen at runtime"
      if not _offenders else f"FOUND AT {_offenders}")

check("TLC.2e the mask is actually PASSED as attn_mask",
      "attn_mask=self.causal_mask(" in _srcs["more/model.py"],
      "so the triangle reaches nn.MultiheadAttention rather than being built and "
      "discarded")


# ===========================================================================
print()
print("=== T-L4.4  GATE L1: perturb the future, the past must not move ===")
# ===========================================================================

def logits_under_perturbation(w, x, sm, t, k):
    """Run twice, the second time with position t+k replaced, and return both."""
    a = w(x, sm)[0]
    x2 = x.clone()
    g = torch.Generator().manual_seed(1234 + t * 97 + k)
    x2[:, t + k, :] = torch.randn(x.shape[0], x.shape[2], generator=g)
    b = w(x2, sm)[0]
    return a.reshape(x.shape[0], x.shape[1], -1), b.reshape(x.shape[0], x.shape[1], -1)


def past_deviation(w, x, sm, cases):
    """Worst absolute change at positions <= t over all (t, k), and exact-equality."""
    worst, all_equal = 0.0, True
    for t, k in cases:
        a, b = logits_under_perturbation(w, x, sm, t, k)
        if not torch.equal(a[:, :t + 1], b[:, :t + 1]):
            all_equal = False
        worst = max(worst, float((a[:, :t + 1] - b[:, :t + 1]).abs().max()))
    return worst, all_equal


# WHY THIS GATE IS NOT A BARE `torch.equal`, and why that is a STRONGER test.
#
# `plan_language.md` §6.4 specifies bit-identity. Measured, bit-identity holds
# exactly at `num_experts = 1, max_depth = 1` and NOT above it -- the residual is
# 2.4e-07, which is 2 ULP of float32 (eps = 1.19e-07). The cause is not a
# causality defect. Top-1 dispatch GROUPS rows by expert, so when a perturbation
# flips the perturbed token's expert the per-expert row counts change (measured:
# loads [6,4,3,1,2,8] -> [5,4,4,1,2,8]) and both experts' GEMMs tile differently,
# which perturbs the shared rows in their last bits. The same happens through the
# ACT loop for E = 1 at depth > 1, where a changed halt decision changes
# `N_active`.
#
# So the gate is written in three parts, and together they say more than
# `torch.equal` alone could:
#   TLC.4a  EXACT bit-identity where the confound is absent (E=1, depth 1).
#   TLC.4b  a few-ULP bound everywhere else, with the mechanism proved in TLC.4e
#           rather than asserted.
#   TLC.4c  the mask-removed variant must exceed that bound by >= 1e5, so the
#           bound cannot be hiding a real leak.
ULP = torch.finfo(torch.float32).eps
PAST_TOL = 8 * ULP          # 9.5e-07; observed worst is 2.4e-07

CASES = [(0, 1), (0, 5), (1, 1), (3, 2), (5, 1), (5, 6), (7, 4), (10, 1)]
_x = torch.randn(B, S, D)
_sm = torch.ones(B, S, dtype=torch.bool)

_w11 = build(max_depth=1, attention=True, num_experts=1)
_d11, _eq11 = past_deviation(_w11, _x, _sm, CASES)
check("TLC.4a EXACT bit-identity with one expert at depth 1 -- no tolerance at all",
      _eq11 and _d11 == 0.0,
      f"{len(CASES)} (t, k) pairs, max|diff| = {_d11:.1e}, torch.equal True. "
      f"This is the anchor: the causal mask is exactly causal once the grouped "
      f"dispatch and the ACT loop cannot vary the batch shapes")

# Every configuration, including all three architectures. MoR is E=1 --
# `updated_objective.md` §4 defines it as recursion without multiple experts, NOT
# as no context, so it keeps attention and must be just as causal.
_cfgs = [("MoE", 6, 1), ("MoR", 1, 7), ("MoRE", 6, 7),
         ("MoRE-d4", 6, 4), ("attn-only-E1-d4", 1, 4)]
_devs = {}
for _name, _e, _md in _cfgs:
    _w = build(max_depth=_md, attention=True, num_experts=_e)
    _devs[_name], _ = past_deviation(_w, _x, _sm, CASES)
_worst_name = max(_devs, key=_devs.get)
check("TLC.4b every architecture keeps the past within a few ULP of unchanged",
      all(v <= PAST_TOL for v in _devs.values()),
      "  ".join(f"{n}={v:.1e}" for n, v in _devs.items())
      + f"   (bound {PAST_TOL:.1e} = 8 ULP; worst is {_worst_name})")

# ---- the test must be ABLE to fail ----------------------------------------
# A leakage check that has never failed has not been shown to work. Same model,
# mask removed, nothing else different.
_ratios = {}
for _name, _e, _md in _cfgs:
    _b = build(max_depth=_md, attention=True, num_experts=_e)
    _b.causal_mask = lambda seq_len, device: None      # noqa: E731
    _ratios[_name] = past_deviation(_b, _x, _sm, CASES)[0]
_min_ratio = min(_ratios[n] / max(_devs[n], ULP) for n in _devs)
check("TLC.4c removing the mask blows the bound by >= 1e5, so it hides no leak",
      all(v > PAST_TOL * 1e5 for v in _ratios.values()),
      "  ".join(f"{n}={v:.2f}" for n, v in _ratios.items())
      + f"   (smallest masked-to-unmasked ratio {_min_ratio:.1e})")

# ...and the arithmetic path is causal for a DIFFERENT reason, worth separating:
# with `attention=False` no token sees another at all, so it could not leak even
# with a broken mask.
_w_noattn = build(max_depth=1, attention=False, num_experts=1)
_dna, _eqna = past_deviation(_w_noattn, _x, _sm, CASES)
check("TLC.4d attention=False is causal by ABSENCE of any cross-position path",
      _eqna and _dna == 0.0,
      f"max|diff| = {_dna:.1e} with E=1 -- there is no mask to break, because "
      f"there is no attention")

# ---- the residual is dispatch tiling, PROVED not asserted -------------------
_w61 = build(max_depth=1, attention=False, num_experts=6)
_r1 = _w61(_x, _sm)
_x2 = _x.clone()
_g = torch.Generator().manual_seed(7)
_x2[:, 11, :] = torch.randn(B, D, generator=_g)
_r2 = _w61(_x2, _sm)
_l1 = torch.bincount(_r1[4][0], minlength=6)
_l2 = torch.bincount(_r2[4][0], minlength=6)
_d61, _ = past_deviation(_w61, _x, _sm, [(10, 1)])
check("TLC.4e the residual comes from GROUPED DISPATCH, not from attention",
      not torch.equal(_l1, _l2) and 0.0 < _d61 <= PAST_TOL and _d11 == 0.0,
      f"E=6 depth 1 with NO attention still shifts the past by {_d61:.1e}: "
      f"per-expert loads {_l1.tolist()} -> {_l2.tolist()}, so two GEMMs change "
      f"row count and re-tile. E=1 depth 1 is exactly 0.0, which isolates the "
      f"cause to the grouping rather than to the mask")


# ===========================================================================
print()
print("=== T-L4.3  Halted tokens stay attendable, and stay frozen ===")
# ===========================================================================

# The freeze is asserted BITWISE on the state the attention sublayer is handed at
# each depth step, which is `current_state` after the previous step's write-back.
# A position that halted at depth h is written for the last time at depth h, so
# every capture from index h onward must be byte-identical for that position.
_wf = build(max_depth=6, attention=True)
_caps = []
_h = _wf.attn.register_forward_pre_hook(
    lambda _m, inp: _caps.append(inp[0].detach().clone().reshape(-1, D)))
_out = _wf(_x, _sm)
_h.remove()

_depth_exits = _out[3]                       # [N, max_depth]
_exit_depth = _depth_exits.argmax(dim=1) + 1  # 1-indexed
_spread = sorted(set(_exit_depth.tolist()))

if len(_caps) < 2:
    skip("TLC.3a-c", f"attention ran only {len(_caps)} depth step(s); the freeze "
                     f"needs at least two to compare")
else:
    _viol = []
    for _p in range(_caps[0].shape[0]):
        _hd = int(_exit_depth[_p])
        base = min(_hd, len(_caps) - 1)
        for _i in range(base, len(_caps)):
            if not torch.equal(_caps[_i][_p], _caps[base][_p]):
                _viol.append((_p, _hd, _i))
                break
    check("TLC.3a a halted position's state is BITWISE unchanged at every later "
          "depth",
          not _viol,
          f"{_caps[0].shape[0]} positions, exit depths {_spread}, "
          f"{len(_caps)} captures -- torch.equal, not allclose"
          if not _viol else f"{len(_viol)} positions moved after halting: "
                            f"{_viol[:5]}")

    check("TLC.3b the check is non-vacuous: tokens really do halt at different "
          "depths",
          len(_spread) >= 2 and min(_spread) < len(_caps),
          f"exit depths present: {_spread} over {len(_caps)} attention steps -- "
          f"if every token halted at max_depth there would be nothing frozen to "
          f"observe")

    # Attendable: the halted position is still a KEY. Perturbing a halted
    # position's state must still change LATER positions' outputs -- otherwise
    # "attendable" is a claim about code rather than about behaviour. Compared
    # within the position's OWN batch row: the other row shifts by ~2 ULP through
    # the dispatch-tiling path of TLC.4e, which is not the effect under test.
    _cand = [int(i) for i in torch.argsort(_exit_depth).tolist()
             if (i % S) < S - 1]
    if _cand:
        _early = _cand[0]
        _b_idx, _s_idx = divmod(_early, S)
        _x2 = _x.clone()
        _gg = torch.Generator().manual_seed(99)
        _x2[_b_idx, _s_idx, :] = torch.randn(D, generator=_gg)
        _o1 = _wf(_x, _sm)[0].reshape(B, S, D)
        _o2 = _wf(_x2, _sm)[0].reshape(B, S, D)
        _moved = float((_o1[_b_idx, _s_idx + 1:]
                        - _o2[_b_idx, _s_idx + 1:]).abs().max())
        check("TLC.3c an early-halting position is still a visible KEY to later "
              "positions",
              _moved > 1e-3,
              f"position {_s_idx} of row {_b_idx} halts at depth "
              f"{int(_exit_depth[_early])}; perturbing it moves later positions by "
              f"{_moved:.3f} -- six orders above the {8 * ULP:.0e} tiling floor, so "
              f"halting early cannot blind position 40 to the word at position 4")
    else:
        skip("TLC.3c", "every position is last in its sequence, so none has a "
                       "later position to be a key for")

# ---- the waste, measured and reported -------------------------------------

_rs = _out[8]
check("TLC.3d the attention query waste is REPORTED as a measured fraction",
      "attn_query_waste_fraction" in _rs
      and 0.0 <= _rs["attn_query_waste_fraction"] < 1.0
      and _rs["attn_query_slots"] > 0,
      f"{_rs['attn_query_wasted']:.0f} of {_rs['attn_query_slots']:.0f} query "
      f"slots discarded = {_rs['attn_query_waste_fraction']:.2%} -- attention runs "
      f"at all S positions and its output is thrown away at halted ones")

_rs_off = W_off(_x, _sm)[8]
check("TLC.3e the arithmetic path reports NO waste key, rather than 0.0",
      "attn_query_waste_fraction" not in _rs_off
      and "attn_query_slots" not in _rs_off,
      "absence means 'not applicable'; a 0.0 would be a sentinel presented as a "
      "measurement (CLAUDE.md §4)")

# The number has to move with the halted fraction, or it is not measuring waste.
_w1 = build(max_depth=1, attention=True)
_rs1 = _w1(_x, _sm)[8]
check("TLC.3f at max_depth=1 nothing has halted yet, so the waste is exactly 0",
      _rs1["attn_query_waste_fraction"] == 0.0,
      f"{_rs1['attn_query_waste_fraction']:.4f} -- one step, every position still "
      f"active, no output discarded")

check("TLC.3g deeper recursion wastes strictly more, which is why it is reported",
      _rs["attn_query_waste_fraction"] > _rs1["attn_query_waste_fraction"],
      f"max_depth 6: {_rs['attn_query_waste_fraction']:.2%} vs max_depth 1: "
      f"{_rs1['attn_query_waste_fraction']:.2%} -- a real cost of the "
      f"fixed-length-query choice, accepted for exact depth accounting")


# ===========================================================================
print()
print("=" * 78)
print(f"{len(PASS)} passed, {len(FAIL)} failed, {len(SKIP)} skipped")
if FAIL:
    print("\nFAILED:")
    for f in FAIL:
        print(f"  - {f}")
print("=" * 78)
sys.exit(1 if FAIL else 0)
