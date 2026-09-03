"""test_phase2_routing.py - Gate for Phase 2: true Top-1 sparse routing.

Verifies T2.1 - T2.4 (plan.md 4.1-4.3, updated_rules.md 1.1 ablation F).

Run:
    C:/Users/vedan/anaconda3/python.exe code/test_phase2_routing.py

Design note. Every check here is a MEASUREMENT of the running code, not an
inspection of it. The dense-vs-sparse distinction is invisible in the loss curve
and invisible in the metrics -- both produce a [N, d_model] output and both
report an argmax. The only way to tell them apart from outside is to count how
many tokens each expert's forward() actually received, so that is what these
tests do, via forward hooks on the expert modules themselves.
"""

import sys
import io
import torch

sys.path.insert(0, __file__.rsplit("\\", 1)[0].rsplit("/", 1)[0])

from more.model import (MoEBlock, MoREModel, ROUTING_MODES, CAPACITY_POLICY,
                        CANONICAL_ROUTING_MODE, DENSE_ABLATION_ROUTING_MODE)
from more.config import enforce_routing_mode, DENSE_ABLATION_LABEL

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

PASS, FAIL = [], []


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))


def instrument(block):
    """Hook every expert; record the input rows each one actually receives."""
    seen = {e: [] for e in range(block.num_experts)}
    handles = []
    for e, expert in enumerate(block.experts):
        def hook(_mod, inp, _out, _e=e):
            seen[_e].append(inp[0].detach().clone())
        handles.append(expert.register_forward_hook(hook))
    return seen, handles


# ---------------------------------------------------------------------------
print("\n[T2.1] Sparse dispatch: each token enters exactly ONE expert")
# ---------------------------------------------------------------------------
torch.manual_seed(0)
E, D, N = 6, 32, 64

block = MoEBlock(num_experts=E, d_model=D, dropout=0.0,
                 routing_mode=CANONICAL_ROUTING_MODE).eval()
x = torch.randn(N, D)
seen, handles = instrument(block)
out, bal, expert_idx, logits, stats = block(x)
for h in handles:
    h.remove()

rows_per_expert = {e: sum(t.shape[0] for t in v) for e, v in seen.items()}
total_rows = sum(rows_per_expert.values())
check(
    "T2.1a each token evaluated by exactly one expert",
    total_rows == N,
    f"experts received {total_rows} rows for {N} tokens "
    f"(dense would be {N * E}); per-expert {rows_per_expert}",
)

# Non-vacuity: if only one expert were ever selected the count would also be N,
# and the test would pass while routing was fully collapsed. Require >1 expert.
check(
    "T2.1a-nv non-vacuous: more than one expert was actually called",
    sum(1 for v in rows_per_expert.values() if v > 0) > 1,
    f"{sum(1 for v in rows_per_expert.values() if v > 0)}/{E} experts called",
)

# Dense comparison run: proves the check above can fail, i.e. it discriminates.
dense = MoEBlock(num_experts=E, d_model=D, dropout=0.0,
                 routing_mode=DENSE_ABLATION_ROUTING_MODE).eval()
seen_d, handles_d = instrument(dense)
dense(x)
for h in handles_d:
    h.remove()
dense_rows = sum(sum(t.shape[0] for t in v) for v in seen_d.values())
check(
    "T2.1a-disc the check discriminates (dense path fails it)",
    dense_rows == N * E,
    f"dense path routed {dense_rows} rows = N*E = {N * E}, sparse routed {total_rows}",
)

# ---- gradient isolation --------------------------------------------------
torch.manual_seed(1)
g_block = MoEBlock(num_experts=E, d_model=D, dropout=0.0,
                   routing_mode=CANONICAL_ROUTING_MODE).eval()
x1 = torch.randn(1, D)
out1, _, idx1, _, _ = g_block(x1)
out1.sum().backward()
chosen = int(idx1.item())
grad_norms = {}
for e, expert in enumerate(g_block.experts):
    tot = 0.0
    for p in expert.parameters():
        if p.grad is not None:
            tot += float(p.grad.abs().sum())
    grad_norms[e] = tot
others = [v for e, v in grad_norms.items() if e != chosen]
check(
    "T2.1b non-selected experts receive NO gradient from that token",
    grad_norms[chosen] > 0 and all(v == 0.0 for v in others),
    f"chosen expert {chosen} grad={grad_norms[chosen]:.6f}; "
    f"others max={max(others):.6f}",
)

# ---- the router itself must still be trained by the task loss ------------
router_grad = float(g_block.router.weight.grad.abs().sum())
check(
    "T2.1c gradient reaches the router through the gate multiply",
    router_grad > 0,
    f"router weight grad = {router_grad:.6f} (0.0 would mean the argmax cut the "
    "router off from the task loss, leaving it trained only by aux/oracle terms)",
)

# ---- the selected gate probability must actually scale the output --------
torch.manual_seed(2)
s_block = MoEBlock(num_experts=E, d_model=D, dropout=0.0,
                   routing_mode=CANONICAL_ROUTING_MODE).eval()
xs = torch.randn(8, D)
out_s, _, idx_s, logits_s, _ = s_block(xs)
probs_s = torch.softmax(logits_s, dim=-1)
gate_s = probs_s.gather(-1, idx_s.unsqueeze(-1)).squeeze(-1)
raw = torch.stack([s_block.experts[int(idx_s[i])](xs[i:i + 1]).squeeze(0)
                   for i in range(xs.shape[0])])
expected = raw * gate_s.unsqueeze(-1)
check(
    "T2.1d output == selected expert output * selected gate probability",
    torch.allclose(out_s, expected, atol=1e-6),
    f"max abs diff = {float((out_s - expected).abs().max()):.3e}; "
    f"gate range [{float(gate_s.min()):.4f}, {float(gate_s.max()):.4f}]",
)

# ---------------------------------------------------------------------------
print("\n[T2.1 cost] Compute cost reported honestly per path")
# ---------------------------------------------------------------------------
# Both paths report the same `dispatched` count (one per token). Only
# `expert_evaluations` distinguishes them, and it must match the hook-counted
# rows exactly -- otherwise a dense ablation could be tabled next to a sparse
# run as if it cost the same.
stats_d = dense(x)[4]
check(
    "T2.1e expert_evaluations equals hook-counted rows on BOTH paths",
    (stats["expert_evaluations"] == float(total_rows)
     and stats_d["expert_evaluations"] == float(dense_rows)),
    f"sparse reports {stats['expert_evaluations']} vs {total_rows} counted; "
    f"dense reports {stats_d['expert_evaluations']} vs {dense_rows} counted",
)
check(
    "T2.1f dense costs exactly num_experts x sparse per token",
    stats_d["expert_evaluations"] == stats["expert_evaluations"] * E,
    f"sparse={stats['expert_evaluations']}, dense={stats_d['expert_evaluations']}, "
    f"ratio={stats_d['expert_evaluations'] / stats['expert_evaluations']:.1f} = E = {E}",
)

# ---------------------------------------------------------------------------
print("\n[T2.2] The scored index is the index that controls dispatch")
# ---------------------------------------------------------------------------
# Match each row an expert received back to its position in x by exact value
# equality, then confirm the expert that received it equals expert_idx.
mismatch, matched = 0, 0
for e, chunks in seen.items():
    for chunk in chunks:
        for r in range(chunk.shape[0]):
            row = chunk[r]
            hit = (x == row).all(dim=1).nonzero(as_tuple=True)[0]
            if hit.numel() != 1:
                continue
            matched += 1
            if int(expert_idx[int(hit.item())]) != e:
                mismatch += 1
check(
    "T2.2a reported expert_idx == expert that actually ran the token",
    mismatch == 0 and matched == N,
    f"{matched}/{N} tokens traced to an expert forward call, {mismatch} mismatches",
)

# Confusion diagonal must equal mean(pred == oracle) to 1e-6.
from more.metrics import routing_accuracy_from_confusion

torch.manual_seed(3)
oracle = torch.randint(0, E, (N,))
pred = expert_idx
conf = torch.zeros(E, E, dtype=torch.float64)
conf.index_put_((oracle.long(), pred.long()),
                torch.ones(N, dtype=torch.float64), accumulate=True)
direct = float((pred == oracle).float().mean())
from_conf = routing_accuracy_from_confusion(conf, E)
check(
    "T2.2b confusion diagonal fraction == mean(pred == oracle) within 1e-6",
    abs(direct - from_conf) < 1e-6,
    f"direct={direct:.10f}  from_confusion={from_conf:.10f}  "
    f"delta={abs(direct - from_conf):.2e}",
)

# ---------------------------------------------------------------------------
print("\n[T2.3] Capacity / overflow policy is measured, not assumed")
# ---------------------------------------------------------------------------
check(
    "T2.3a capacity policy is declared explicitly",
    CAPACITY_POLICY == "no_capacity_limit",
    f"CAPACITY_POLICY = {CAPACITY_POLICY!r} (plan.md 4.3: prefer the simple "
    "implementation when the distribution does not require capacity limiting)",
)
check(
    "T2.3b every routed token is dispatched; overflow is a measured number",
    (stats["dispatched"] == stats["tokens"] == float(N)
     and isinstance(stats["overflow"], float) and stats["overflow"] == 0.0),
    f"tokens={stats['tokens']}, dispatched={stats['dispatched']}, "
    f"overflow={stats['overflow']} (measured zero, not a placeholder)",
)
check(
    "T2.3c capacity pressure reported as a real fraction",
    0.0 < stats["max_load_fraction"] <= 1.0,
    f"max_load_fraction = {stats['max_load_fraction']:.4f} "
    f"(uniform would be {1.0 / E:.4f}, full collapse 1.0)",
)

# Model-level aggregation must surface an overflow_rate, not a sentinel.
torch.manual_seed(4)
model = MoREModel(step_feat_dim=12, d_model=32, num_experts=E, max_depth=3,
                  num_blocks=2, dropout=0.0).eval()
B, S = 4, 7
xb = torch.randn(B, S, 12)
smask = torch.ones(B, S, dtype=torch.bool)
sops = torch.randint(0, 16, (B, S))
sexp = torch.randint(0, E, (B, S))
res = model(xb, smask, sexp, sops)
# Index 10, not -1. Phase 3 appended `expected_depth` and `halt_stats` to the
# MoREModel return tuple, so `res[-1]` silently became the halt-stats dict and
# these two checks failed with a KeyError-shaped miss rather than a wrong number.
# Positional indices into a growing tuple are the hazard; name the slot.
rs = res[10]
check(
    "T2.3e model reports evals_per_token (1.0 == true sparse)",
    isinstance(rs.get("evals_per_token"), float)
    and abs(rs["evals_per_token"] - 1.0) < 1e-9,
    f"evals_per_token={rs.get('evals_per_token')!r} "
    f"(1.0 = every token seen by exactly one expert; {float(E)} would mean dense)",
)
check(
    "T2.3d model reports overflow_rate as a real number",
    isinstance(rs.get("overflow_rate"), float) and rs["overflow_rate"] == 0.0,
    f"overflow_rate={rs.get('overflow_rate')!r}, dispatched={rs.get('dispatched')}, "
    f"dispatch_steps={rs.get('dispatch_steps')}, policy={rs.get('capacity_policy')!r}",
)

# ---------------------------------------------------------------------------
print("\n[T2.4] Dense routing reachable only as a labelled ablation")
# ---------------------------------------------------------------------------
base = {"model": {"routing_mode": DENSE_ABLATION_ROUTING_MODE},
        "logging": {"run_name": "phaseB_more"}}
try:
    enforce_routing_mode(base)
    check("T2.4a unlabelled dense run is REFUSED", False, "no exception raised")
except ValueError as e:
    check("T2.4a unlabelled dense run is REFUSED", DENSE_ABLATION_LABEL in str(e),
          f"ValueError mentions the required label: ...{str(e)[-90:].strip()}")

ok = {"model": {"routing_mode": DENSE_ABLATION_ROUTING_MODE},
      "logging": {"run_name": f"phaseB_more_{DENSE_ABLATION_LABEL}"}}
try:
    mode = enforce_routing_mode(ok)
    check("T2.4b labelled dense run is ALLOWED", mode == DENSE_ABLATION_ROUTING_MODE,
          f"routing_mode resolved to {mode!r}, variant={ok.get('variant')!r}")
except ValueError as e:
    check("T2.4b labelled dense run is ALLOWED", False, str(e)[:90])

canon = {"model": {}, "logging": {"run_name": "phaseB_more"}}
check("T2.4c canonical default is top1_sparse",
      enforce_routing_mode(canon) == CANONICAL_ROUTING_MODE,
      f"default resolved to {canon['model']['routing_mode']!r}")

bad = {"model": {"routing_mode": "top2"}, "logging": {"run_name": "x"}}
try:
    enforce_routing_mode(bad)
    check("T2.4d unknown routing_mode raises", False, "no exception raised")
except ValueError:
    check("T2.4d unknown routing_mode raises", True,
          f"only {ROUTING_MODES} accepted")

# ---------------------------------------------------------------------------
print("\n" + "-" * 70)
print(f"  Phase 2 gate: {len(PASS)}/{len(PASS) + len(FAIL)} checks passed")
print("-" * 70)
if FAIL:
    for f in FAIL:
        print(f"  FAILED: {f}")
    sys.exit(1)
print("  Phase 2 (T2.1-T2.4) verified.")
