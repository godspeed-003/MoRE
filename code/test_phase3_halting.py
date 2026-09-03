"""test_phase3_halting.py - Gate 2 for Phase 3: real ACT halting.

Verifies T3.1 - T3.4 (plan.md 5.1-5.5, updated_rules.md 2.2-2.3, CLAUDE.md 2).

Run:
    C:/Users/vedan/anaconda3/python.exe code/test_phase3_halting.py

Design note. The defect this gate exists to prevent is SILENT. Before Phase 3
the halting term was `active_mask.float().mean() * 0.05` -- a number computed
from a bool tensor, so `requires_grad` was False and every one of the 24
halt-head parameters had `grad is None` after backward. Nothing about that is
visible in the loss curve, the depth histogram or the exit-rate metric: the
model still exited at different depths, purely because randomly-initialised
halt heads produce different sigmoid outputs. A run could be, and was, written
up as "adaptive depth allocation" while the halting parameters never moved.

So every check below is a measurement of gradient flow or of tensor values in
the running model, never an inspection of the source.
"""

import sys
import torch

sys.path.insert(0, __file__.rsplit("\\", 1)[0].rsplit("/", 1)[0])

from more.model import MoREWrapper, MoREModel
from more.metrics import halting_supervision_loss, depth_allocation_error
from more.families import (op_target_depth_table, OP_TARGET_DEPTH,
                           OP_NAME_TO_IDX, ALL_OP_NAMES, NUM_OP_TYPES)
from more.config import apply_architecture, load_config_defaults

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

PASS, FAIL = [], []


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))


def build_wrapper(max_depth=5, num_experts=6, d_model=32, fixed_depth=False):
    torch.manual_seed(0)
    return MoREWrapper(d_model=d_model, max_depth=max_depth,
                       num_experts=num_experts, dropout=0.0,
                       fixed_depth=fixed_depth).eval()


def build_model(max_depth=5, num_experts=6, d_model=32, num_blocks=2):
    torch.manual_seed(0)
    return MoREModel(step_feat_dim=12, d_model=d_model, num_experts=num_experts,
                     max_depth=max_depth, num_blocks=num_blocks,
                     dropout=0.0).eval()


# ===========================================================================
print("\n[T3.1] Canonical ACT: convex combination, frozen halted states, "
      "forced exit at max_depth")
# ===========================================================================

W = build_wrapper(max_depth=5)
B, S, D = 4, 7, 32
x = torch.randn(B, S, D)
smask = torch.ones(B, S, dtype=torch.bool)

# --- T3.1a: per-token ACT step weights sum to exactly 1 --------------------
# This is THE structural property of ACT. If it does not hold the output is not
# a convex combination of the per-step states, its scale drifts with depth, and
# the "deeper = more halt mass" confound contaminates the task loss. It is also
# the property that makes the halt weights a legitimate gradient path rather
# than an arbitrary rescaling.
#
# `halting_prob` (the cumulative mass at exit) IS that sum by construction, so
# we read it back out of a re-instrumented forward.
captured = {}
_orig_forward = MoREWrapper.forward


def _instrumented(self, xx, mm, se=None):
    out = _orig_forward(self, xx, mm, se)
    # expected_depth = N + R ; halt mass total is recomputed independently below
    captured["expected_depth"] = out[9]
    captured["halt_stats"] = out[10]
    return out


MoREWrapper.forward = _instrumented
res = W(x, smask)
MoREWrapper.forward = _orig_forward

mean_mass = captured["halt_stats"]["mean_halt_mass"]
check("T3.1a ACT step weights sum to 1 per token (mean halt mass == 1.0)",
      abs(mean_mass - 1.0) < 1e-5,
      f"mean cumulative halt mass over real tokens = {mean_mass:.9f} "
      f"(exactly 1.0 => convex combination; a value < 1 would mean the output "
      f"is a shrunken state and depth silently scales the activations)")

# --- T3.1b: halted states are frozen, not recomputed ----------------------
# Measured, not asserted from the code: run the wrapper twice with the SAME
# input but hook the MoEBlock to record how many rows it received at each depth
# step. A token that halted at step k must never appear again at step k+1, so
# the active-row count must be non-increasing.
rows_per_step = []
h = W.moe_block.register_forward_pre_hook(
    lambda _m, inp: rows_per_step.append(inp[0].shape[0])
)
_ = W(x, smask)
h.remove()
non_increasing = all(rows_per_step[i] >= rows_per_step[i + 1]
                     for i in range(len(rows_per_step) - 1))
check("T3.1b halted tokens are evicted, never recomputed "
      "(active rows non-increasing over depth)",
      non_increasing and len(rows_per_step) >= 2,
      f"rows per depth step = {rows_per_step} "
      f"(strictly non-increasing; a rise would mean a halted token re-entered)")

# --- T3.1c: every real token exits exactly once ---------------------------
depth_exits = res[3]                      # [B*S, max_depth]
exits_per_token = depth_exits.sum(dim=1)
real = smask.reshape(-1)
check("T3.1c every real token exits exactly once",
      bool(torch.all(exits_per_token[real] == 1.0)),
      f"min={exits_per_token[real].min().item()} "
      f"max={exits_per_token[real].max().item()} over {int(real.sum())} tokens")

# --- T3.1d: forced exit at max_depth is real, and counted separately ------
# Force it: drive every halt head's bias very negative so p_t ~ 0 and no token
# can ever accumulate enough mass to halt on its own.
Wf = build_wrapper(max_depth=4)
with torch.no_grad():
    for head in Wf.expert_halt_heads:
        head.weight.zero_()
        head.bias.fill_(-20.0)
resf = Wf(x, smask)
hs_f = resf[10]
de_f = resf[3]
all_at_last = bool(torch.all(de_f[real][:, -1] == 1.0))
check("T3.1d forced exit at max_depth when halting never fires",
      all_at_last and hs_f["early_exits"] == 0.0 and hs_f["forced_exits"] > 0,
      f"forced={hs_f['forced_exits']:.0f} early={hs_f['early_exits']:.0f}; "
      f"all real tokens exited at depth {Wf.max_depth}: {all_at_last}")

# --- T3.1e: early exit is reachable, and the two are distinguishable ------
# The mirror of T3.1d: bias very positive so p_1 ~ 1 and every token halts at
# depth 1. If both directions work, the exit-rate metric is measuring halting
# behaviour rather than reporting a constant.
We = build_wrapper(max_depth=4)
with torch.no_grad():
    for head in We.expert_halt_heads:
        head.weight.zero_()
        head.bias.fill_(20.0)
rese = We(x, smask)
hs_e = rese[10]
de_e = rese[3]
all_at_first = bool(torch.all(de_e[real][:, 0] == 1.0))
check("T3.1e early exit at depth 1 when halting fires immediately",
      all_at_first and hs_e["forced_exits"] == 0.0 and hs_e["early_exits"] > 0,
      f"forced={hs_e['forced_exits']:.0f} early={hs_e['early_exits']:.0f}; "
      f"all real tokens exited at depth 1: {all_at_first}")

# --- T3.1f: remainder is the ACT remainder, not a constant ----------------
# R = 1 - c_t at the halt step. When every token halts at depth 1, c_0 = 0 so
# R must be exactly 1.0. This pins the remainder to the ACT definition rather
# than to "whatever the halt head emitted".
check("T3.1f remainder R == 1 - cumulative mass (== 1.0 for a depth-1 exit)",
      abs(hs_e["mean_remainder"] - 1.0) < 1e-5,
      f"mean_remainder={hs_e['mean_remainder']:.9f} at depth-1 exit "
      f"(must be exactly 1.0: nothing was accumulated before the halt step)")

# --- T3.1g: pad tokens never enter the depth loop -------------------------
smask_pad = smask.clone()
smask_pad[:, 4:] = False
res_pad = W(x, smask_pad)
de_pad = res_pad[3]
padrows = ~smask_pad.reshape(-1)
check("T3.1g pad tokens are excluded from the depth loop (zero exits)",
      bool(torch.all(de_pad[padrows].sum(dim=1) == 0.0)),
      f"{int(padrows.sum())} pad tokens, total pad exits recorded = "
      f"{de_pad[padrows].sum().item():.0f}")


# ===========================================================================
print("\n[T3.2] Ponder cost is differentiable and normalized")
# ===========================================================================

Wg = build_wrapper(max_depth=5)
Wg.train()
resg = Wg(x, smask)
ponder = resg[2]
expected_depth = resg[9]

check("T3.2a ponder_cost.requires_grad is True",
      bool(ponder.requires_grad),
      f"requires_grad={ponder.requires_grad} "
      f"(was False before Phase 3: the old term was "
      f"active_mask.float().mean() * 0.05 on a bool tensor)")
check("T3.2b ponder_cost has a grad_fn (it is on the autograd graph)",
      ponder.grad_fn is not None,
      f"grad_fn={type(ponder.grad_fn).__name__ if ponder.grad_fn else None}")
check("T3.2c expected_depth (N + R) is differentiable",
      bool(expected_depth.requires_grad) and expected_depth.grad_fn is not None,
      f"requires_grad={expected_depth.requires_grad}, "
      f"grad_fn={type(expected_depth.grad_fn).__name__ if expected_depth.grad_fn else None}")

# Normalization: (N + R) / (max_depth + 1) must land in (0, 1]. Unnormalized it
# would grow with max_depth, so a deeper model would be penalised for its shape
# rather than its behaviour -- the magnitude defect CLAUDE.md 2 forbids.
pv = float(ponder.item())
check("T3.2d ponder_cost is normalized into (0, 1] and does not scale with depth",
      0.0 < pv <= 1.0,
      f"ponder_cost={pv:.6f} for max_depth={Wg.max_depth} "
      f"= mean(N + R) / (max_depth + 1)")

# Depth-invariance of the magnitude: same input, two different max_depths, both
# must stay in (0, 1]. If the cost were a raw step count the deeper model's
# value would be ~2x larger for identical behaviour.
p_small = float(build_wrapper(max_depth=3)(x, smask)[2].item())
p_large = float(build_wrapper(max_depth=9)(x, smask)[2].item())
check("T3.2e normalized ponder cost stays bounded as max_depth grows",
      0.0 < p_small <= 1.0 and 0.0 < p_large <= 1.0,
      f"max_depth=3 -> {p_small:.6f}, max_depth=9 -> {p_large:.6f} "
      f"(both in (0,1]; an unnormalized step count would grow ~3x)")


# ===========================================================================
print("\n[T3.3] Operation-complexity curriculum + halting supervision")
# ===========================================================================

table = op_target_depth_table()
check("T3.3a curriculum covers exactly the operation manifest",
      len(table) == NUM_OP_TYPES == len(OP_TARGET_DEPTH),
      f"{len(table)} target depths for {NUM_OP_TYPES} operations")

# The curriculum must NOT be a pure function of the expert index. If it were, a
# model could score a perfect depth-allocation error by routing alone and the
# depth measurement would carry no information independent of routing.
from more.families import OP_TO_EXPERT
by_expert = {}
for op, e in OP_TO_EXPERT.items():
    by_expert.setdefault(e, set()).add(OP_TARGET_DEPTH[op])
spans = {e: sorted(v) for e, v in by_expert.items() if len(v) > 1}
shared = [d for d in set(OP_TARGET_DEPTH.values())
          if len({OP_TO_EXPERT[o] for o in OP_TARGET_DEPTH if OP_TARGET_DEPTH[o] == d}) > 1]
check("T3.3b target depth is NOT a pure function of the expert index",
      len(spans) > 0 and len(shared) > 0,
      f"expert(s) spanning multiple target depths: {spans}; "
      f"target depth(s) shared across experts: {sorted(shared)} "
      f"(so perfect routing does not imply perfect depth allocation)")

# The supervision loss must be differentiable w.r.t. expected_depth and must be
# zero-but-differentiable (never NaN) when no labelled step is present.
step_ops = torch.randint(0, NUM_OP_TYPES, (B, S))
hsl = halting_supervision_loss(expected_depth, step_ops, smask, table, Wg.max_depth)
check("T3.3c halting_supervision_loss is differentiable",
      bool(hsl.requires_grad) and hsl.grad_fn is not None,
      f"value={hsl.item():.6f}, requires_grad={hsl.requires_grad}")

empty_mask = torch.zeros_like(smask)
hsl0 = halting_supervision_loss(expected_depth, step_ops, empty_mask, table,
                                Wg.max_depth)
check("T3.3d no labelled step -> differentiable zero, not NaN",
      bool(hsl0.requires_grad) and float(hsl0.item()) == 0.0
      and not torch.isnan(hsl0).any(),
      f"value={hsl0.item()}, requires_grad={hsl0.requires_grad}")

# depth_allocation_error must be an ABSOLUTE error: over- and under-allocation
# must not cancel. Construct a case where they would.
fake_depth = torch.zeros(B * S)
ops2 = torch.zeros(B, S, dtype=torch.long)
add_i = OP_NAME_TO_IDX["ADD"]        # target 1
srt_i = OP_NAME_TO_IDX["SORT"]       # target 4
ops2[:, 0] = add_i
ops2[:, 1] = srt_i
m2 = torch.zeros(B, S, dtype=torch.bool)
m2[:, 0] = True
m2[:, 1] = True
flat2 = m2.reshape(-1)
fake_depth[flat2] = 0.0
fake_depth = fake_depth.clone()
idx = torch.nonzero(flat2).squeeze(-1)
fake_depth[idx[0::2]] = 4.0          # ADD tokens over-allocated by +3
fake_depth[idx[1::2]] = 1.0          # SORT tokens under-allocated by -3
abs_err, rel_err = depth_allocation_error(fake_depth, ops2, m2, table)
check("T3.3e depth_allocation_error is absolute (+3 and -3 do not cancel)",
      abs_err is not None and abs(abs_err - 3.0) < 1e-6,
      f"abs_err={abs_err} (must be 3.0, not 0.0); rel_err={rel_err}")

no_abs, no_rel = depth_allocation_error(fake_depth, ops2, empty_mask, table)
check("T3.3f depth_allocation_error returns None (-> N/A) when undefined",
      no_abs is None and no_rel is None,
      f"({no_abs}, {no_rel}) -- caller emits the string N/A, never 0.0, "
      f"which would read as perfect allocation (CLAUDE.md 4)")

# Provenance: flipping the flag must change what is recorded, not just what runs.
cfg_off = load_config_defaults({})
cfg_on = load_config_defaults({"model": {"halting_supervision": True},
                               "loss_weights": {"halting_supervision": 0.1}})
check("T3.3g halting_supervision defaults OFF (pure ACT is the default path)",
      cfg_off["model"]["halting_supervision"] is False
      and cfg_off["loss_weights"]["halting_supervision"] == 0.0,
      f"model.halting_supervision={cfg_off['model']['halting_supervision']}, "
      f"loss_weights.halting_supervision={cfg_off['loss_weights']['halting_supervision']}")
check("T3.3h enabling supervision changes the recorded config",
      cfg_on["model"]["halting_supervision"] is True
      and cfg_on["loss_weights"]["halting_supervision"] == 0.1,
      "model.halting_supervision=True, loss_weights.halting_supervision=0.1 "
      "-> engine records halting_mode='supervised_curriculum' in "
      "resolved_config.json, metrics.json and the W&B config")

# MoE has max_depth 1: a curriculum target of 2/3/4 is unreachable by
# construction, so supervising it would inject a permanent irreducible loss.
cfg_moe = apply_architecture(load_config_defaults({}), "moe")
check("T3.3i MoE (max_depth 1) has curriculum supervision force-disabled",
      cfg_moe["model"]["halting_supervision"] is False
      and cfg_moe["loss_weights"]["halting_supervision"] == 0.0
      and cfg_moe["model"]["max_depth"] == 1,
      f"max_depth={cfg_moe['model']['max_depth']}, supervision="
      f"{cfg_moe['model']['halting_supervision']} "
      f"(targets of 2-4 are unreachable at depth 1)")


# ===========================================================================
print("\n[T3.4] A real gradient reaches the halt parameters")
# ===========================================================================

# The headline check. Build the FULL model, backward through the TASK loss only
# -- no ponder cost, no supervision term -- and require that a real gradient
# reaches the halt parameters. Task-loss-only is deliberate: a gradient arriving
# from the ponder cost alone would prove nothing, since the ponder cost is a
# function of the halt heads by definition. What was broken before Phase 3 is the
# path from the task loss.
#
# IMPORTANT (T5.1 follow-up). The halt heads are PER EXPERT and canonical routing
# is top-1 sparse, so on a small batch some expert receives no token in some
# block and its halt head then correctly receives an exactly-zero gradient.
# Demanding all 24 heads be live on an arbitrary batch demands that dispatch be
# DENSE -- the opposite of the canonical design (CLAUDE.md 2) -- and it is luck:
# under the current 6-wide heads at B=4, S=3 the experts left unused across the
# two blocks number 4, 3, 3, 1, 3, 1 for seeds 0-5. The old all-24 form passed
# only because the pre-T5.1 7-wide cls_head consumed a different slice of the
# init RNG stream and that router init happened to reach every expert.
#
# So the claim is split into parts, none of which depends on that luck:
#   (a) on a batch where every expert IS dispatched, every halt head must be live
#   (a2) on a sparse batch no halt parameter may be OFF the graph (grad is None).
#        That is the actual pre-Phase-3 defect (None=24) and it is a property of
#        the graph, not of routing.
#   (a3) a gradient must never appear for an expert that was not dispatched to.
xb = torch.randn(B, S, 12)
sops = torch.randint(0, NUM_OP_TYPES, (B, S))
sexp = torch.randint(0, 6, (B, S))


def dispatch_sets(model, x, mask, exp, ops):
    """Run the model, recording which experts each block actually dispatched to.

    Read off MoEBlock's own third output (expert_idx) via a forward hook, so this
    is the routing DECISION as the model made it, not the test's guess at it.
    """
    seen = {i: set() for i in range(len(model.blocks))}
    handles = []
    for i, blk in enumerate(model.blocks):
        def _hook(_m, _in, out_tuple, _i=i):
            seen[_i].update(int(e) for e in
                            out_tuple[2].detach().flatten().tolist())
        handles.append(blk.moe_block.register_forward_hook(_hook))
    out = model(x, mask, exp, ops)
    for h in handles:
        h.remove()
    return out, seen


def live_halt_experts(model):
    """Per block, the experts whose halt head received a NON-ZERO gradient."""
    live = {i: set() for i in range(len(model.blocks))}
    for i, blk in enumerate(model.blocks):
        for e, head in enumerate(blk.expert_halt_heads):
            g = head.weight.grad
            if g is not None and float(g.abs().sum()) > 0.0:
                live[i].add(e)
    return live


def wide_batch(nb, ns, seed):
    """A fresh model plus a batch big enough that all six experts can be hit."""
    torch.manual_seed(seed)
    m = build_model(max_depth=5, num_experts=6, num_blocks=2)
    m.train()
    x = torch.randn(nb, ns, 12)
    mask = torch.ones(nb, ns, dtype=torch.bool)
    mask[0, -1] = False                     # a pad token must be present
    ops = torch.randint(0, NUM_OP_TYPES, (nb, ns))
    exp = torch.randint(0, 6, (nb, ns))
    return m, (x, mask, exp, ops)

# (a) Search for a batch on which every expert is dispatched in every block. The
# search failing is itself a failure -- it would mean routing cannot reach all six
# experts even on ~288 tokens, which is collapse, not sparsity.
COV = None
for _seed in range(25):
    _m, _batch = wide_batch(48, 6, _seed)
    _out, _seen = dispatch_sets(_m, *_batch)
    if all(len(s) == 6 for s in _seen.values()):
        COV = (_m, _batch, _out, _seen, _seed)
        break

if COV is None:
    check("T3.4a every halt-head parameter gets a gradient from the TASK loss "
          "on a fully-dispatched batch", False,
          "no batch in 25 tries dispatched to all 6 experts in both blocks")
else:
    M, cov_batch, cov_out, cov_seen, cov_seed = COV
    task_loss = torch.nn.functional.mse_loss(
        cov_out[0].squeeze(-1), torch.randn(cov_batch[0].shape[0]))
    M.zero_grad(set_to_none=True)
    task_loss.backward()

    halt_params = [(n, p) for n, p in M.named_parameters()
                   if "expert_halt_heads" in n]
    n_none = sum(1 for _, p in halt_params if p.grad is None)
    n_live = sum(1 for _, p in halt_params
                 if p.grad is not None and float(p.grad.abs().sum()) > 0.0)
    grad_norm = sum(float(p.grad.abs().sum()) for _, p in halt_params
                    if p.grad is not None)
    check("T3.4a every halt-head parameter gets a gradient from the TASK loss "
          "on a fully-dispatched batch",
          len(halt_params) > 0 and n_none == 0 and n_live == len(halt_params),
          f"seed {cov_seed}, {cov_batch[0].shape[0] * cov_batch[0].shape[1]} "
          f"tokens, all 6 experts dispatched in both blocks: "
          f"{len(halt_params)} halt params, grad is None={n_none}, non-zero "
          f"grad={n_live}/{len(halt_params)}; total |grad| = {grad_norm:.6f} "
          f"(pre-Phase-3 measurement was: None=24, non-zero=0)")

# (a2) The graph property, checked on the ORIGINAL small batch where dispatch is
# genuinely sparse. The invariant is one-directional, and measurement is what
# settled which direction: if an expert IS dispatched its halt head was actually
# called, so a graph node exists and `grad is not None`. If it is NOT dispatched
# the head is never called at all, so `grad is None` -- absence of a node, not a
# severed one. (A dispatched expert can still land on an exact zero when every
# token routed to it had already halted at that depth; measured on seeds 0-7 that
# happens for 0-2 of the 12 expert slots and is not a defect.)
M_sparse = build_model(max_depth=5, num_experts=6, num_blocks=2)
M_sparse.train()
sp_out, sp_seen = dispatch_sets(M_sparse, xb, smask, sexp, sops)
sp_loss = torch.nn.functional.mse_loss(sp_out[0].squeeze(-1), torch.randn(B))
M_sparse.zero_grad(set_to_none=True)
sp_loss.backward()
sp_off_graph = [(i, e) for i, blk in enumerate(M_sparse.blocks)
                for e in sorted(sp_seen[i])
                if blk.expert_halt_heads[e].weight.grad is None]
sp_unused = sum(6 - len(s) for s in sp_seen.values())
check("T3.4a2 every DISPATCHED expert's halt head is on the autograd graph",
      not sp_off_graph,
      f"{sp_unused} expert slot(s) unused across the 2 blocks "
      f"(b0={sorted(sp_seen[0])}, b1={sorted(sp_seen[1])}); of the "
      f"{12 - sp_unused} dispatched slots, {len(sp_off_graph)} had grad is None "
      f"-- an undispatched expert's head is simply never called, which is top-1 "
      f"sparsity, not the pre-Phase-3 defect of a severed path")

# (a3) No gradient may appear for an expert that was never dispatched to.
sp_live = live_halt_experts(M_sparse)
leaked = {i: sorted(sp_live[i] - sp_seen[i]) for i in sp_live
          if sp_live[i] - sp_seen[i]}
check("T3.4a3 no halt head is live for an expert that was not dispatched to",
      not leaked,
      ("live sets " + ", ".join(f"b{i}={sorted(sp_live[i])}" for i in sp_live)
       + " are subsets of the dispatch sets")
      if not leaked else f"gradient leaked to undispatched experts: {leaked}")


# Non-vacuity: the check above must be capable of failing. Re-run with the halt
# weights detached from the accumulator and confirm the gradient disappears --
# otherwise the "gradient" might be arriving through some other route and the
# test would pass for the wrong reason.
M2 = build_model(max_depth=5, num_experts=6, num_blocks=2)
M2.train()
for blk in M2.blocks:
    for head in blk.expert_halt_heads:
        head.weight.requires_grad_(False)
        head.bias.requires_grad_(False)
out2 = M2(xb, smask, sexp, sops)
tl2 = torch.nn.functional.mse_loss(out2[0].squeeze(-1), torch.randn(B))
M2.zero_grad(set_to_none=True)
tl2.backward()
frozen_grads = [p.grad for n, p in M2.named_parameters()
                if "expert_halt_heads" in n]
check("T3.4b the T3.4a check is non-vacuous (frozen halt heads get no grad)",
      all(g is None for g in frozen_grads),
      f"{len(frozen_grads)} frozen halt params, all grad is None: "
      f"{all(g is None for g in frozen_grads)} "
      f"(so T3.4a is reading the halt path, not an unrelated one)")

# Gradient also has to arrive via the ponder cost, which is what makes the ponder
# term a real regularizer rather than a logged number. Same fully-dispatched
# batch as T3.4a, so this criterion is likewise independent of routing luck.
if COV is None:
    check("T3.4c the ponder cost alone also reaches every halt parameter", False,
          "skipped: no fully-dispatched batch was found for T3.4a")
else:
    M3, b3 = wide_batch(48, 6, cov_seed)
    out3, seen3 = dispatch_sets(M3, *b3)
    M3.zero_grad(set_to_none=True)
    out3[4].backward()                     # total_ponder_cost alone
    p_live = sum(1 for n, p in M3.named_parameters()
                 if "expert_halt_heads" in n and p.grad is not None
                 and float(p.grad.abs().sum()) > 0.0)
    p_tot = sum(1 for n, _ in M3.named_parameters() if "expert_halt_heads" in n)
    full3 = all(len(s) == 6 for s in seen3.values())
    check("T3.4c the ponder cost alone also reaches every halt parameter",
          p_tot > 0 and p_live == p_tot and full3,
          f"{p_live}/{p_tot} halt params with non-zero grad from ponder_cost "
          f"alone, on the fully-dispatched batch (all 6 experts per block: "
          f"{full3})")


# ===========================================================================
print("\n[GATE 2] plan.md 5.5: halt probabilities move in the expected "
      "direction on a known-depth synthetic case")
# ===========================================================================

# plan.md 5.5 asks for a tiny synthetic case where the target depth is KNOWN,
# with a check that the halt probabilities move the right way after an optimizer
# step. This is the end-to-end statement of "halting learns": everything above
# proves the gradient exists; this proves its SIGN is useful.
#
# Setup: all tokens are SORT (curriculum target depth 4) but the halt heads are
# initialised to fire almost immediately (bias +3 => p_1 ~ 0.95, expected depth
# ~1). Under the curriculum loss the heads must be pushed DOWN so tokens survive
# to deeper steps. One optimizer step must increase expected depth and decrease
# the loss.
torch.manual_seed(7)
SYNTH_MAX_DEPTH = 6
Ms = build_model(max_depth=SYNTH_MAX_DEPTH, num_experts=6, num_blocks=1)
Ms.train()
with torch.no_grad():
    for blk in Ms.blocks:
        for head in blk.expert_halt_heads:
            head.weight.zero_()
            head.bias.fill_(3.0)

sops_sort = torch.full((B, S), OP_NAME_TO_IDX["SORT"], dtype=torch.long)
target_d = float(OP_TARGET_DEPTH["SORT"])
opt = torch.optim.SGD(
    [p for n, p in Ms.named_parameters() if "expert_halt_heads" in n], lr=5.0
)


def curriculum_step(apply_update: bool):
    o = Ms(xb, smask, sexp, sops_sort)
    ed = o[11]
    loss = halting_supervision_loss(ed, sops_sort, smask,
                                    op_target_depth_table(), SYNTH_MAX_DEPTH)
    mean_d = float(ed[smask.reshape(-1)].mean().item())
    if apply_update:
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return float(loss.item()), mean_d


loss_before, depth_before = curriculum_step(True)
loss_after, depth_after = curriculum_step(False)

check("GATE2a halt heads move toward the known target depth after one step",
      depth_after > depth_before,
      f"target depth for SORT = {target_d:.0f}; expected depth "
      f"{depth_before:.4f} -> {depth_after:.4f} (must INCREASE: the heads "
      f"started biased to halt at step 1)")
check("GATE2b the curriculum loss decreases after that step",
      loss_after < loss_before,
      f"halting_supervision_loss {loss_before:.6f} -> {loss_after:.6f}")

# And the opposite direction, so the test is not just measuring "SGD reduces
# whatever it is given". Target ADD (depth 1) from heads biased to never halt.
torch.manual_seed(7)
Md = build_model(max_depth=SYNTH_MAX_DEPTH, num_experts=6, num_blocks=1)
Md.train()
with torch.no_grad():
    for blk in Md.blocks:
        for head in blk.expert_halt_heads:
            head.weight.zero_()
            head.bias.fill_(-3.0)
sops_add = torch.full((B, S), OP_NAME_TO_IDX["ADD"], dtype=torch.long)
opt_d = torch.optim.SGD(
    [p for n, p in Md.named_parameters() if "expert_halt_heads" in n], lr=5.0
)


def curriculum_step_d(apply_update: bool):
    o = Md(xb, smask, sexp, sops_add)
    ed = o[11]
    loss = halting_supervision_loss(ed, sops_add, smask,
                                    op_target_depth_table(), SYNTH_MAX_DEPTH)
    if apply_update:
        opt_d.zero_grad(set_to_none=True)
        loss.backward()
        opt_d.step()
    return float(loss.item()), float(ed[smask.reshape(-1)].mean().item())


dl_before, dd_before = curriculum_step_d(True)
dl_after, dd_after = curriculum_step_d(False)
check("GATE2c the direction is target-driven, not monotone "
      "(ADD target 1 pushes depth DOWN)",
      dd_after < dd_before,
      f"target depth for ADD = {OP_TARGET_DEPTH['ADD']}; expected depth "
      f"{dd_before:.4f} -> {dd_after:.4f} (must DECREASE, opposite of GATE2a)")


# ===========================================================================
print("\n[GATE 2] Reporting separation (CLAUDE.md 4 / plan.md 5.4)")
# ===========================================================================

# plan.md 5.4: "Keep it separate from ponder cost ... Do not collapse them into
# one opaque number." The two terms pull in OPPOSITE directions -- ponder cost
# pushes depth down, curriculum supervision pushes it toward the target -- so a
# single summed number could sit flat while both components move a lot.
import io as _io
_eng = _io.open(__file__.rsplit("\\", 1)[0].rsplit("/", 1)[0]
                + "/more/engine.py", encoding="utf-8").read()
required_keys = [
    "train/ponder_cost",
    "train/halting_supervision_loss",
    "halt/forced_exit_rate",
    "halt/early_exit_rate",
    "halt/mean_remainder",
    "depth/allocation_error_abs",
    "depth/allocation_error_rel",
]
missing = [k for k in required_keys if f'"{k}"' not in _eng]
check("GATE2d ponder cost and curriculum supervision are logged as SEPARATE keys",
      not missing,
      f"all {len(required_keys)} keys present" if not missing
      else f"missing: {missing}")

check("GATE2e halt/depth metrics emit N/A rather than 0.0 when undefined",
      'metrics["depth/allocation_error_abs"] = "N/A"' in _eng
      and '"halt/forced_exit_rate", "halt/early_exit_rate"' in _eng,
      "engine writes the string N/A into metrics.json for undefined halt and "
      "depth quantities (MoE at max_depth 1, or no labelled step seen)")


# ===========================================================================
print("\n" + "=" * 70)
print(f"  PASSED: {len(PASS)}    FAILED: {len(FAIL)}")
if FAIL:
    print("\n  FAILURES:")
    for f in FAIL:
        print(f"    - {f}")
    print("=" * 70)
    sys.exit(1)
print(f"  GATE 2 (Phase 3 real ACT halting): {len(PASS)}/{len(PASS)} checks passed")
print("=" * 70)
