"""test_phase4_balance.py - Gate 3 for Phase 4: the balance objective.

Verifies T4.1 - T4.3 (plan.md 6.1-6.3, CLAUDE.md 2 "Balance loss", CLAUDE.md 4).

Run:
    C:/Users/vedan/anaconda3/python.exe code/test_phase4_balance.py

Design note. The defect this gate exists to prevent is a MAGNITUDE defect, and
magnitude defects do not crash anything. The balance loss was SUMMED over every
(block, depth) call, so a 2-block 7-depth model produced up to 14 copies of a
term that a 1-block 1-depth model produced once -- for identical routing
behaviour. Two consequences, both silent:

  1. The balance objective stopped being subordinate. Measured pre-fix:
     0.05 * (-4.861) = -0.243 against a task loss of 0.001666, i.e. the
     auxiliary term was ~146x the primary one.
  2. Every depth and block-count comparison became confounded. A deeper model
     is not just deeper, it is also more heavily regularised, so "depth helped"
     and "regularisation helped" cannot be separated after the fact.

So the checks below measure the aggregation itself (is the returned number the
MEAN of the per-call values?), the invariance it is supposed to buy, and the
direction of the gradient it produces. Nothing is inspected from the source.

This phase does NOT redesign the objective: the per-call formula stays
-entropy_term + switch_aux exactly as before. Only the aggregation over calls,
the reporting split, and the weight change.
"""

import math
import sys
import torch

sys.path.insert(0, __file__.rsplit("\\", 1)[0].rsplit("/", 1)[0])

from more.model import MoEBlock, MoREWrapper, MoREModel
from more.config import apply_architecture, load_config_defaults

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

PASS, FAIL = [], []


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))

D_MODEL, E = 32, 6


def build_wrapper(max_depth=5, num_experts=E, fixed_depth=False):
    torch.manual_seed(0)
    return MoREWrapper(d_model=D_MODEL, max_depth=max_depth,
                       num_experts=num_experts, dropout=0.0,
                       fixed_depth=fixed_depth).eval()


def build_model(max_depth=5, num_experts=E, num_blocks=2):
    torch.manual_seed(0)
    return MoREModel(step_feat_dim=12, d_model=D_MODEL, num_experts=num_experts,
                     max_depth=max_depth, num_blocks=num_blocks,
                     dropout=0.0).eval()


class CallRecorder:
    """Records every per-call balance value the real MoEBlock produces.

    Wraps MoEBlock.forward rather than reimplementing it, so the recorded values
    are by construction the same tensors that go into the aggregate. A test that
    recomputed the balance formula itself would pass even if the aggregation and
    the formula disagreed.
    """

    def __init__(self):
        self.values = []
        self._orig = MoEBlock.forward
        rec = self

        def patched(blk, x):
            out = rec._orig(blk, x)
            rec.values.append(float(out[1].detach().item()))
            return out

        MoEBlock.forward = patched

    def __enter__(self):
        return self

    def __exit__(self, *a):
        MoEBlock.forward = self._orig
        return False

# ===========================================================================
print("\n[T4.1] Balance loss is a MEAN over (block, depth) calls, not a sum")
# ===========================================================================

B, S = 4, 7
torch.manual_seed(1)
x_in  = torch.randn(B, S, D_MODEL)
smask = torch.ones(B, S, dtype=torch.bool)

# --- T4.1a: the wrapper's returned value IS the mean of its per-call values --
# Definitional, and exact to float tolerance. If someone reintroduces the sum,
# or divides by max_depth instead of the realised call count, this fails.
W = build_wrapper(max_depth=5, fixed_depth=True)
with CallRecorder() as rec:
    res = W(x_in, smask)
bal_w = float(res[1].detach().item())
mean_calls = sum(rec.values) / len(rec.values)
check("T4.1a wrapper balance == mean of per-call values",
      abs(bal_w - mean_calls) < 1e-6,
      f"returned {bal_w:.9f}, mean of {len(rec.values)} calls {mean_calls:.9f}, "
      f"sum would be {sum(rec.values):.6f}")

# --- T4.1b: and it is NOT the sum -----------------------------------------
# Non-vacuity: with 5 calls the mean and the sum are far apart, so T4.1a is
# discriminating rather than trivially satisfied.
check("T4.1b returned value is not the sum (non-vacuous)",
      len(rec.values) > 1 and abs(bal_w - sum(rec.values)) > 1e-3,
      f"{len(rec.values)} calls; |mean - sum| = {abs(bal_w - sum(rec.values)):.6f}")

# --- T4.1c: the model's value is the mean over blocks of the block means ----
# The halt biases are driven to -20 so no token ever halts early: both blocks
# then run exactly max_depth steps, which makes the mean-of-means numerically
# equal to a flat mean over all calls and lets one assertion cover both. (Where
# blocks run unequal depths the two differ, and equal-per-block weighting is the
# deliberate choice -- see model.MoREModel.forward.)
M = build_model(max_depth=5, num_blocks=2)
with torch.no_grad():
    for blk in M.blocks:
        for hh in blk.expert_halt_heads:
            hh.bias.fill_(-20.0)
xs = torch.randn(B, S, 12)
# step_ops is mandatory since Phase 1 (operation identity is the only signal for
# WHICH function a step computes). Values are irrelevant to the balance
# objective; they just have to be valid op codes.
ops = torch.zeros(B, S, dtype=torch.long)
with CallRecorder() as rec_m:
    res_m = M(xs, smask, None, ops)
bal_m = float(res_m[3].detach().item())
n_blocks_calls = len(rec_m.values)
grand_mean = sum(rec_m.values) / n_blocks_calls
check("T4.1c model balance == mean over blocks of per-block means",
      abs(bal_m - grand_mean) < 1e-6 and n_blocks_calls == 10,
      f"returned {bal_m:.9f}, flat mean over {n_blocks_calls} calls "
      f"{grand_mean:.9f}; sum would be {sum(rec_m.values):.4f}")


# --- T4.1d: depth invariance, max_depth 3 vs 7 (TASKS.md: < 5%) -------------
# Same seed, so both wrappers get bit-identical parameters, and the same input.
# fixed_depth=True forces 3 and 7 real calls respectively, which is the worst
# case for the old sum: it differed by 7/3 = 2.33x on identical routing.
bals, sums = {}, {}
for d in (3, 7):
    Wd = build_wrapper(max_depth=d, fixed_depth=True)
    with CallRecorder() as r:
        out_d = Wd(x_in, smask)
    bals[d] = float(out_d[1].detach().item())
    sums[d] = sum(r.values)
rel = abs(bals[7] - bals[3]) / max(abs(bals[3]), 1e-12)
old_rel = abs(sums[7] - sums[3]) / max(abs(sums[3]), 1e-12)
check("T4.1d balance loss depth-invariant (max_depth 3 vs 7, < 5%)",
      rel < 0.05,
      f"depth3 {bals[3]:.6f}, depth7 {bals[7]:.6f}, delta {rel*100:.3f}%  "
      f"(unnormalized sums {sums[3]:.4f} vs {sums[7]:.4f} = {old_rel*100:.1f}%)")

# --- T4.1e: the pre-fix behaviour is reproduced, so the fix is load-bearing --
# The sums must differ by roughly the depth ratio. Without this the < 5% above
# could be passing because the balance term is simply insensitive to depth for
# some unrelated reason.
check("T4.1e unnormalized sum WOULD scale with depth (fix is load-bearing)",
      old_rel > 0.5,
      f"sum ratio depth7/depth3 = {sums[7]/sums[3]:.3f} (~7/3 = 2.333 expected)")

# --- T4.1f: block-count invariance, 1 vs 4 blocks ---------------------------
# Block 0 is bit-identical across the two models (same seed, same construction
# order); the deeper blocks see different inputs, so exact equality is not
# expected. What must hold is that the value does not scale with the block
# count: summed, 4 blocks gave ~4x. Tolerance is loose on purpose -- the claim
# is "does not scale with num_blocks", not "is identical".
bb = {}
for nb in (1, 4):
    Mb = build_model(max_depth=5, num_blocks=nb)
    with torch.no_grad():
        for blk in Mb.blocks:
            for hh in blk.expert_halt_heads:
                hh.bias.fill_(-20.0)
    with CallRecorder() as r:
        ob = Mb(xs, smask, None, ops)
    bb[nb] = (float(ob[3].detach().item()), sum(r.values), len(r.values))
ratio_norm = bb[4][0] / bb[1][0]
ratio_sum  = bb[4][1] / bb[1][1]
check("T4.1f balance loss does not GROW with num_blocks (1 vs 4)",
      ratio_norm < 1.25,
      f"1 block {bb[1][0]:.6f} ({bb[1][2]} calls), 4 blocks {bb[4][0]:.6f} "
      f"({bb[4][2]} calls), ratio {ratio_norm:.4f}; summed ratio "
      f"{ratio_sum:.3f}. Below 1.0 is behavioural, not a normalization "
      f"artifact: deeper blocks route more confidently, so their per-call "
      f"balance value is genuinely lower.")

# --- T4.1g: the aggregate is inside the hull of its parts -------------------
# Exact and behaviour-independent, unlike a tolerance band: a mean of values
# must lie between their min and max, and a sum of same-signed values cannot.
# This is the check that still holds when the blocks disagree, which is why it
# exists alongside T4.1c.
Mh = build_model(max_depth=5, num_blocks=4)
with CallRecorder() as r:
    oh = Mh(xs, smask, None, ops)
agg = float(oh[3].detach().item())
lo, hi = min(r.values), max(r.values)
check("T4.1g aggregate lies within [min, max] of the per-call values",
      lo - 1e-6 <= agg <= hi + 1e-6 and len(r.values) > 4,
      f"aggregate {agg:.6f} in [{lo:.6f}, {hi:.6f}] over {len(r.values)} calls; "
      f"sum {sum(r.values):.4f} is outside it")


# ===========================================================================
print("\n[T4.2] The two halves of the objective are reported separately")
# ===========================================================================

# --- T4.2a: both terms are present in route_stats and reconstruct the loss ---
# CLAUDE.md 4 requires entropy_term and switch_aux_term be reported separately.
# The reconstruction identity is what makes them trustworthy: if the reported
# pair did not sum back to the loss, they would be describing something other
# than the objective actually being minimised.
Wr = build_wrapper(max_depth=1, fixed_depth=True)
out_r = Wr(x_in, smask)
rs = out_r[8]
have_both = "entropy_term" in rs and "switch_aux_term" in rs
recon = (-rs["entropy_term"] + rs["switch_aux_term"]) if have_both else None
check("T4.2a entropy_term and switch_aux_term reported, and -H + aux == loss",
      have_both and abs(recon - float(out_r[1].item())) < 1e-6,
      f"entropy {rs.get('entropy_term'):.6f}, switch_aux "
      f"{rs.get('switch_aux_term'):.6f}, -H+aux {recon:.6f} vs loss "
      f"{float(out_r[1].item()):.6f}")

# --- T4.2b: the reported terms are per-call means too, not sums --------------
W5 = build_wrapper(max_depth=5, fixed_depth=True)
o5 = W5(x_in, smask)
rs5 = o5[8]
check("T4.2b reported terms are per-call means (5 calls, entropy <= log E)",
      rs5["entropy_term"] <= math.log(E) + 1e-6,
      f"entropy_term {rs5['entropy_term']:.6f} <= log({E}) = {math.log(E):.6f}; "
      f"a sum over 5 calls would be ~{5 * rs5['entropy_term']:.4f}")

# --- T4.2c: switch_aux is in [1, E] and entropy in [0, log E] ---------------
# The bounds of the two terms are what make the weight selection in T4.2
# auditable: they say exactly how large the balance term can ever get, so a
# coefficient can be chosen against a worst case instead of a lucky batch.
check("T4.2c both terms inside their analytic bounds",
      0.0 <= rs5["entropy_term"] <= math.log(E) + 1e-6
      and 1.0 - 1e-6 <= rs5["switch_aux_term"] <= E + 1e-6,
      f"entropy in [0, {math.log(E):.4f}] -> {rs5['entropy_term']:.6f}; "
      f"switch_aux in [1, {E}] -> {rs5['switch_aux_term']:.6f}")

# ===========================================================================
print("\n[T4.3] Gradient direction (plan.md 6.3)")
# ===========================================================================

N_TOK = 240


def fresh_block():
    torch.manual_seed(3)
    return MoEBlock(E, D_MODEL, 0.0).eval()   # eval -> router noise off


def collapsed_setup(strength=1.5):
    """Router that sends every token to expert 0, escapably.

    Every token carries a constant positive value in dim 0 and zero-mean noise
    elsewhere; only row 0 of the router reads dim 0. So dispatch starts 100%
    collapsed, but the noise dimensions give the other rows something
    discriminative to learn -- the router CAN escape. `strength` is modest on
    purpose: past roughly 2.0 the softmax saturates to one-hot in float32,
    entropy underflows to exactly 0.0, and the gradient vanishes. T4.3f pins
    that down as a measured limitation rather than leaving it to be rediscovered.
    """
    blk = fresh_block()
    with torch.no_grad():
        blk.router.weight.zero_()
        blk.router.weight[0, 0] = strength
    torch.manual_seed(4)
    xc = torch.randn(N_TOK, D_MODEL)
    xc[:, 0] = 2.0
    return blk, xc


def uniform_setup():
    """Uniform in BOTH senses: soft probabilities and hard dispatch.

    Both matter. The Switch term is E * sum_e(f_e * P_e) with f the hard load;
    it is at its optimum only when f is uniform as well, so a setup with uniform
    softmax but all argmax ties landing on expert 0 is NOT the optimum and does
    have gradient. Tokens are therefore given an epsilon-sized preference for
    expert (i mod E): the dispatch is exactly uniform while the probabilities
    stay uniform to O(eps).
    """
    blk = fresh_block()
    eps = 1e-3
    with torch.no_grad():
        blk.router.weight.zero_()
        for e in range(E):
            blk.router.weight[e, e] = eps
    torch.manual_seed(5)
    xu = torch.randn(N_TOK, D_MODEL)
    xu[:, :E] = 0.0
    for i in range(N_TOK):
        xu[i, i % E] = 1.0
    return blk, xu

# --- T4.3a: collapsed routing -> gradient moves AWAY from collapse ----------
# Optimising the balance loss alone must undo the collapse. This is the check
# that the sign is right: a flipped sign would minimise entropy and the term
# would actively drive the very collapse it exists to prevent, which no loss
# curve would reveal.
blk_c, xc = collapsed_setup()
o = blk_c(xc)
ent_before  = o[4]["entropy_term"]
aux_before  = o[4]["switch_aux_term"]
load_before = o[4]["max_load_fraction"]

opt = torch.optim.SGD([blk_c.router.weight], lr=0.05)
grad0 = None
for i in range(200):
    opt.zero_grad(set_to_none=True)
    out = blk_c(xc)
    out[1].backward()          # balance loss ONLY
    if i == 0:
        grad0 = blk_c.router.weight.grad.norm().item()
    opt.step()
o = blk_c(xc)
ent_after  = o[4]["entropy_term"]
aux_after  = o[4]["switch_aux_term"]
load_after = o[4]["max_load_fraction"]

check("T4.3a collapsed routing: balance gradient increases entropy",
      ent_after > ent_before + 1e-4,
      f"entropy_term {ent_before:.6f} -> {ent_after:.6f} "
      f"(max {math.log(E):.6f}), first-step |grad| {grad0:.5f}")
check("T4.3b collapsed routing: switch_aux term decreases",
      aux_after < aux_before - 1e-4,
      f"switch_aux_term {aux_before:.6f} -> {aux_after:.6f} (min 1.0)")
check("T4.3c collapsed routing: hard dispatch collapse is reduced",
      load_after < load_before - 1e-2,
      f"max_load_fraction {load_before:.4f} -> {load_after:.4f} "
      f"(uniform would be {1.0/E:.4f})")

# --- T4.3d: measured limitation -- entropy saturates before dispatch does ----
# Not a failure of the fix; a property of the objective that must be reported.
# The soft entropy term reaches its maximum while a large majority of tokens are
# still dispatched to one expert, because entropy is computed on the MEAN softmax
# and is blind to the hard argmax. Consequence for reporting: normalized entropy
# near 1.0 is NOT evidence that load is balanced (CLAUDE.md 4 already forbids
# reading it as specialization). Read dispatch/max_load_fraction alongside it.
check("T4.3d near-maximal entropy coexists with heavy hard imbalance (limitation)",
      ent_after / math.log(E) > 0.99 and load_after > 2.0 / E,
      f"entropy_normalized {ent_after / math.log(E):.4f} while "
      f"max_load_fraction is {load_after:.4f} vs uniform {1.0/E:.4f}")

# --- T4.3e: uniform routing -> the objective is at its optimum --------------
# plan.md 6.3: "uniform routing -> balance objective has the expected optimum".
# The analytic optimum of -H + E*sum(f*P) with both f and P uniform is
# -log(E) + 1 exactly. Checking the VALUE against the closed form (not just
# "it's small") is what catches a rescaled or re-signed term.
blk_u, xu = uniform_setup()
ou = blk_u(xu)
ent_u  = ou[4]["entropy_term"]
aux_u  = ou[4]["switch_aux_term"]
load_u = ou[4]["max_load_fraction"]
bal_u  = float(ou[1].item())
expected_opt = -math.log(E) + 1.0
check("T4.3e uniform routing sits at the analytic optimum -log(E) + 1",
      abs(bal_u - expected_opt) < 1e-3
      and abs(load_u - 1.0 / E) < 1e-6,
      f"balance {bal_u:.6f} vs -log({E})+1 = {expected_opt:.6f}; "
      f"entropy {ent_u:.6f} vs log(E) {math.log(E):.6f}; switch_aux "
      f"{aux_u:.6f} vs 1.0; max_load {load_u:.6f} vs {1.0/E:.6f}")

# --- T4.3f: at the optimum the gradient is near zero ------------------------
# The other half of "has the expected optimum": the value is right AND the
# gradient stops pushing. Compared against the collapsed gradient so the claim
# is a ratio, not an absolute number that means nothing on its own.
blk_u.zero_grad(set_to_none=True)
blk_u(xu)[1].backward()
g_uniform = blk_u.router.weight.grad.norm().item()
blk_c2, xc2 = collapsed_setup()
blk_c2.zero_grad(set_to_none=True)
blk_c2(xc2)[1].backward()
g_collapsed = blk_c2.router.weight.grad.norm().item()
check("T4.3f gradient magnitude near zero at uniform, large at collapse",
      g_uniform < 0.01 * g_collapsed,
      f"|grad| uniform {g_uniform:.8f} vs collapsed {g_collapsed:.6f} "
      f"(ratio {g_uniform / max(g_collapsed, 1e-12):.2e})")

# --- T4.3g: saturated router -> no gradient at all (limitation) -------------
# Measured, and recorded because it bounds what the balance loss can be claimed
# to do: it discourages collapse, it does not rescue a router that has already
# saturated. If a run ever reports entropy exactly 0.0, no amount of balance
# weight will move it -- the fix is initialisation or noise, not this term.
blk_s, xs_sat = collapsed_setup(strength=10.0)
os_ = blk_s(xs_sat)
blk_s.zero_grad(set_to_none=True)
os_[1].backward()
g_sat = blk_s.router.weight.grad.norm().item()
check("T4.3g fully saturated router has numerically zero balance gradient",
      os_[4]["entropy_term"] < 1e-5 and g_sat < 1e-5,
      f"entropy_term {os_[4]['entropy_term']:.3e}, |grad| {g_sat:.3e} "
      f"-- the balance term cannot recover from full saturation")

# ===========================================================================
print("\n[T4.2 cont] Config, architecture interaction and CLI refusals")
# ===========================================================================

# --- T4.2d: MoR sets the weight to zero, because E == 1 has nothing to balance
cfg_mor = apply_architecture(load_config_defaults({}), "mor")
check("T4.2d architecture mor zeroes routing_balance (E == 1)",
      cfg_mor["loss_weights"]["routing_balance"] == 0.0
      and cfg_mor["model"]["num_experts"] == 1,
      f"num_experts={cfg_mor['model']['num_experts']}, "
      f"routing_balance={cfg_mor['loss_weights']['routing_balance']}")

# --- T4.2e: MoE and MoRE keep it, because both route ------------------------
cfg_moe  = apply_architecture(load_config_defaults({}), "moe")
cfg_more = apply_architecture(load_config_defaults({}), "more")
check("T4.2e moe and more keep a non-zero routing_balance",
      cfg_moe["loss_weights"]["routing_balance"] > 0.0
      and cfg_more["loss_weights"]["routing_balance"] > 0.0,
      f"moe={cfg_moe['loss_weights']['routing_balance']}, "
      f"more={cfg_more['loss_weights']['routing_balance']}")

# --- T4.2h: the selected coefficient is FROZEN, and frozen in ONE place -----
# plan.md 6.2: "The exact final coefficient must be documented and frozen before
# the canonical matrix." Frozen means a future edit has to trip this check rather
# than silently changing what every run optimises. It is asserted in two places
# because there ARE two: more/config.py's setdefault and code/config.json. If they
# drift, runs that pass --config get one value and runs that do not get another,
# and the difference never appears in a results table.
FROZEN_ROUTING_BALANCE = 0.001

import json as _json
import os as _os

_cfg_json_path = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                               "config.json")
with open(_cfg_json_path) as _f:
    _w_file = _json.load(_f)["loss_weights"]["routing_balance"]
_w_default = load_config_defaults({})["loss_weights"]["routing_balance"]

check("T4.2h routing_balance frozen at the measured value in both places",
      _w_file == FROZEN_ROUTING_BALANCE and _w_default == FROZEN_ROUTING_BALANCE,
      f"config.json={_w_file}, config.py default={_w_default}, "
      f"frozen={FROZEN_ROUTING_BALANCE} "
      f"(measured ratio 0.0173 after warmup vs 0.4146 at the historical 0.05)")


from more import cli as more_cli

def cli_refuses(argv):
    try:
        more_cli.main(argv)
        return False, "no exception"
    except SystemExit as e:
        return False, f"SystemExit {e.code}"
    except ValueError as e:
        return True, str(e).split(".")[0]
    except Exception as e:                      # noqa: BLE001
        return False, f"{type(e).__name__}: {e}"

ok, why = cli_refuses(["--architecture", "mor", "--routing_balance", "0.01",
                       "--epochs", "1", "--config", _cfg_json_path])
check("T4.2f CLI refuses --routing_balance on mor (single expert)", ok, why)

ok, why = cli_refuses(["--architecture", "more", "--routing_balance", "-0.01",
                       "--epochs", "1", "--config", _cfg_json_path])
check("T4.2g CLI refuses a negative routing_balance (sign flip)", ok, why)

# ===========================================================================
print("\n[GATE 3] Verified against real run artifacts (plan.md 6.3)")
# ===========================================================================
# The first two gate criteria are already covered by measurements above
# (T4.1d depth-invariance, T4.3a-g gradient direction). The third -- "task loss
# dominates" -- is a property of a TRAINING RUN, not of a synthetic tensor, so it
# is checked against what the run actually wrote to disk. metrics.json is what a
# results table is built from, so that is the correct thing to assert on.

import json
import os

RUNS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs"
)


def load_metrics(substr, arch=None):
    """Newest runs/<id>/metrics.json whose directory name contains `substr`.

    `arch` filters on the architecture the run actually trained, read from the
    run's own resolved_config.json rather than from the directory name -- a
    --run_name is free text and cannot be trusted to describe the run.
    """
    if not os.path.isdir(RUNS_DIR):
        return None, None
    cands = []
    for d in os.listdir(RUNS_DIR):
        if substr not in d:
            continue
        p = os.path.join(RUNS_DIR, d, "metrics.json")
        if not os.path.exists(p):
            continue
        if arch is not None:
            rc = os.path.join(RUNS_DIR, d, "resolved_config.json")
            if not os.path.exists(rc):
                continue
            with open(rc) as f:
                if json.load(f).get("architecture") != arch:
                    continue
        cands.append((os.path.getmtime(p), d, p))
    if not cands:
        return None, None
    _, name, path = max(cands)
    with open(path) as f:
        return name, json.load(f)

BAL_KEYS = (
    "train/routing_balance_loss", "train/aux_routing_loss",
    "train/entropy_term", "train/switch_aux_term",
    "train/entropy_term_normalized", "train/balance_to_task_ratio",
)

# ---- G3.1  task loss dominates, in a real MoRE run ------------------------
more_name, more_m = load_metrics("phase4_", arch="more")

if more_m is None:
    check("G3.1 MoRE run artifact present", False,
          "no runs/<id>/metrics.json found -- run train.py --architecture more "
          "--epochs 1 first; the gate cannot be judged from synthetic tensors")
else:
    ratio = more_m.get("train/balance_to_task_ratio")
    check("G3.1 balance term subordinate to task loss in a real run",
          isinstance(ratio, (int, float)) and ratio < 1.0,
          f"{more_name}: |w*balance|/|task| = {ratio} (pre-fix ~146x)")

    # G3.2  the two halves are reported separately (CLAUDE.md 4). They pull in
    # opposite directions, so a single fused number can sit flat while both
    # components drift; the decomposition is what makes that visible.
    missing = [k for k in ("train/entropy_term", "train/switch_aux_term",
                           "train/entropy_term_normalized") if k not in more_m]
    check("G3.2 entropy_term and switch_aux_term reported separately",
          not missing,
          f"entropy={more_m.get('train/entropy_term')} "
          f"switch_aux={more_m.get('train/switch_aux_term')} "
          f"H/log(E)={more_m.get('train/entropy_term_normalized')}"
          if not missing else f"missing {missing}")

    # G3.3  the aggregation is recorded IN the artifact. Without this a future
    # reader cannot tell whether a tabled balance number is a mean or a sum,
    # which is exactly the ambiguity that produced the defect.
    norm = more_m.get("routing_balance_normalization")
    check("G3.3 normalization scheme stamped into metrics.json",
          norm == "mean over depth calls per block, then mean over blocks",
          f"{norm!r}")

    # G3.4  normalized entropy is in [0, 1] by construction (H / log E). A value
    # outside it means the wrong E or a double-normalized term.
    en = more_m.get("train/entropy_term_normalized")
    check("G3.4 normalized entropy within [0, 1]",
          isinstance(en, (int, float)) and 0.0 <= en <= 1.0 + 1e-9,
          f"H/log(E) = {en}")

# ---- G3.5  E == 1 is N/A, never a comparative number ----------------------
# With one expert the objective is a CONSTANT: entropy of a one-element
# distribution is 0 and the Switch term is exactly 1, so routing_balance_loss is
# 1.0 for every batch forever. Tabled next to MoRE's value that constant reads as
# "MoR is maximally imbalanced", which is a fabricated comparison (CLAUDE.md 4:
# no sentinel may be reported as a measurement).
mor_name, mor_m = load_metrics("", arch="mor")
if mor_m is None:
    check("G3.5 MoR run artifact present", False,
          "no MoR metrics.json found -- run train.py --architecture mor --epochs 1")
else:
    bad = {k: mor_m[k] for k in BAL_KEYS if k in mor_m and mor_m[k] != "N/A"}
    check("G3.5 single-expert run reports balance keys as N/A",
          not bad,
          f"{mor_name}: num_experts=1, all {len(BAL_KEYS)} balance keys N/A"
          if not bad else f"numeric values leaked: {bad}")

# ===========================================================================
print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    print("FAILED:")
    for f in FAIL:
        print(f"  - {f}")
sys.exit(1 if FAIL else 0)










