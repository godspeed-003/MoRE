# -*- coding: utf-8 -*-
"""test_route_persistence.py - T-LX.12 verification.

Run:
    C:/Users/vedan/anaconda3/python.exe code/test_route_persistence.py

WHAT IS UNDER TEST. `MoREWrapper.forward` used to call `self.moe_block(...)` inside
the depth loop unconditionally, so a token's expert was re-chosen by `argmax` at
every recursion step. T-LX.11 measured the consequence on a canonical-corpus proxy:
agreement with the depth-1 expert fell 1.000 -> 0.844 -> 0.771 -> 0.635 -> 0.573 ->
0.521 -> 0.427 across the seven depths, a 0.30 switch rate per step. CLAUDE.md 1
defines MoRE as "a token is routed to a specialized expert, and THAT EXPERT'S weights
are recursively reused for an adaptive number of steps" -- so the shipped model was
not MoRE. T-LX.12 adds `routing_persistence`: `per_token` routes once per token per
forward and reuses the decision, `per_step` is the legacy behaviour kept as a
labelled ablation.

WHY BOTH THE INDEX AND THE GATE HAVE TO PERSIST, and why a test must check the
second one. The obvious fix persists only the argmax and lets the router re-derive
the gate probability at each depth. That fix is defeatable: the router can drive the
persisted expert's gate toward 0 at depths >= 2, which multiplies the expert output
by ~0 and makes the block a near-no-op, i.e. re-routing through the back door while
`expert_idx` reports a fixed index and every persistence metric looks perfect.
TLX12e is what makes that visible -- it asserts the router is CALLED once, which no
amount of gate manipulation can fake.

WHY THE LEGACY CONTROL IS ASSERTED TOO (TLX12c, TLX12j2). A persistence test that
only checks `per_token` passes vacuously if the depth loop stops re-routing for some
unrelated reason (a shape bug that collapses the active set, an `argmax` over a
constant tensor). Each invariant is therefore paired with the legacy run on the SAME
weights and the SAME input, where the defect must still be present. If a control
ever starts passing, the `per_token` result above it has stopped being evidence.

SEEDED, CPU, NO TRAINING. Every model here is randomly initialised and run forward
only, because route persistence is a property of the forward graph, not of a learned
policy. That is also why these numbers may be quoted about the ARCHITECTURE and never
about the trained models: a random router's decay curve is not the canonical one.
"""

import copy
import json
import os
import random
import sys

import torch

sys.path.insert(0, __file__.rsplit("\\", 1)[0].rsplit("/", 1)[0])

from more.model import (MoREModel, MoEBlock,                       # noqa: E402
                        ROUTING_PERSISTENCE_PER_TOKEN,
                        ROUTING_PERSISTENCE_LEGACY,
                        ROUTING_PERSISTENCE_MODES,
                        CANONICAL_ROUTING_MODE, DENSE_ABLATION_ROUTING_MODE)
from more.config import (load_config, apply_architecture, apply_task,   # noqa: E402
                         resolve_variant, canonical_routing_persistence,
                         stamp_language_dataset_versions,
                         TASK_LANGUAGE, TASK_ARITHMETIC, ARCHITECTURES)
from more.run_context import (config_hash, assert_not_silent_proxy,  # noqa: E402
                              ProxyGuardError, load_canonical_spec)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

CODE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(CODE)

PASS, FAIL, SKIP = [], [], []


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))


def skip(name, why):
    SKIP.append(name)
    print(f"  [SKIP] {name}  -- {why}")


def raises(fn, needle):
    """(did it raise?, does the message name `needle`?, the message)."""
    try:
        fn()
    except Exception as e:                       # noqa: BLE001 - the type is asserted below
        return True, (needle in str(e)), f"{type(e).__name__}: {str(e).splitlines()[0][:110]}"
    return False, False, "no exception"


E, MAXD, D_MODEL, SEQ = 6, 7, 64, 8
BATCH = 24


def seeded():
    """Identical initialisation for the per_token and per_step models.

    Global seeding, not just a generator, because `nn.Linear` draws from the global
    RNG -- CLAUDE.md 5 requires this everywhere and it is load-bearing here: the two
    models must be the SAME model differing only in persistence, otherwise a
    difference in the decay curve could be a difference in weights.
    """
    torch.manual_seed(1234)
    random.seed(1234)


def build(persistence, fixed_depth=False, num_experts=E, routing_mode=None, **kw):
    seeded()
    return MoREModel(
        step_feat_dim=8, d_model=D_MODEL, num_experts=num_experts, max_depth=MAXD,
        num_blocks=1, dropout=0.0, fixed_depth=fixed_depth, ffn_mult=2,
        num_families=E, attention=False, n_heads=4, max_seq_len=SEQ,
        routing_mode=routing_mode or CANONICAL_ROUTING_MODE,
        routing_persistence=persistence, **kw)


def batch():
    seeded()
    x = torch.randn(BATCH, SEQ, 8)
    return (x, torch.ones(BATCH, SEQ, dtype=torch.bool), None,
            torch.zeros(BATCH, SEQ, dtype=torch.long))


def run(model, inp=None):
    x, mask, experts, ops = inp or batch()
    out = model(x, mask, step_experts=experts, step_ops=ops)
    return {"bal": out[3], "depth_exits": out[5], "per_depth_expert": out[7],
            "first_route": out[9], "route_stats": out[10], "halt": out[12],
            "logits": out[0]}


def agreement_curve(per_depth):
    """P(expert@d == expert@d1) per depth, truncated where the active set narrows.

    A shrinking active set means row i of depth d is a DIFFERENT token from row i of
    depth 1, so comparing them would manufacture disagreement. Truncating is why the
    persistence checks run with `fixed_depth=True`: there the width is constant and
    the comparison is exact at all seven depths.
    """
    base = per_depth[0]
    out = []
    for idx in per_depth:
        if idx.numel() != base.numel():
            break
        out.append(float((idx == base).float().mean()))
    return out


# ---------------------------------------------------------------------------
print("\n=== T-LX.12  the route persists through recursion ===")

fx_tok = run(build(ROUTING_PERSISTENCE_PER_TOKEN, fixed_depth=True))
fx_leg = run(build(ROUTING_PERSISTENCE_LEGACY, fixed_depth=True))
a_tok, a_leg = agreement_curve(fx_tok["per_depth_expert"]), agreement_curve(fx_leg["per_depth_expert"])

check("TLX12a all seven depths are comparable under fixed_depth",
      len(a_tok) == MAXD and len(a_leg) == MAXD,
      f"per_token {len(a_tok)}/{MAXD}, per_step {len(a_leg)}/{MAXD}")
check("TLX12b per_token: every token keeps its depth-1 expert at every depth",
      all(abs(v - 1.0) < 1e-9 for v in a_tok),
      " ".join(f"{v:.3f}" for v in a_tok))
# The control. If this ever passes, TLX12b has stopped being evidence.
check("TLX12c per_step still re-routes on the same weights (control)",
      len(a_leg) == MAXD and a_leg[-1] < 0.95,
      " ".join(f"{v:.3f}" for v in a_leg))
# `first_route_block0` was a reporting-only capture before T-LX.12 (it was written
# and never read). Under per_token it is the decision the recursion actually uses, so
# it must equal the per-depth index -- a mismatch would mean the metrics describe a
# different assignment from the one the experts saw.
check("TLX12d ... and the reported depth-1 route is the one the recursion used",
      bool(torch.equal(fx_tok["first_route"].to(fx_tok["per_depth_expert"][-1].dtype),
                       fx_tok["per_depth_expert"][-1])),
      "first_route_block0 == expert_idx at depth 7")


print("\n=== T-LX.12  one routing DECISION, max_depth dispatches ===")

rs_tok, rs_leg = fx_tok["route_stats"], fx_leg["route_stats"]
check("TLX12e per_token calls the router once while dispatching max_depth times",
      rs_tok["router_calls"] == 1.0 and rs_tok["dispatch_steps"] == float(MAXD),
      f"router_calls={rs_tok['router_calls']:.0f} dispatch_steps={rs_tok['dispatch_steps']:.0f}")
check("TLX12e2 per_step calls it once per dispatch (control)",
      rs_leg["router_calls"] == float(MAXD) and rs_leg["dispatch_steps"] == float(MAXD),
      f"router_calls={rs_leg['router_calls']:.0f} dispatch_steps={rs_leg['dispatch_steps']:.0f}")

# THE NORMALIZATION. `total_bal_loss /= bal_calls` counts the calls that actually
# ROUTED. The tempting implementation -- have a persisted-route call return 0.0 rather
# than None -- divides one real balance term by max_depth and silently weakens the
# auxiliary 7x (CLAUDE.md 2: the balance loss must be normalized so its magnitude does
# not grow with recursion depth; being SHRUNK by depth is the same defect mirrored).
# Captured per call, so the aggregate is checked against its own parts rather than
# against a number typed in here.
_CAP = []
_orig = MoEBlock.forward


def _capture(self, x, route=None):
    out = _orig(self, x, route=route)
    # `.get`, not `[...]`: on a persisted-route call the block reports NO balance
    # statistics at all, so `entropy_term` and `switch_aux_term` are absent rather
    # than zero. That absence is the contract -- a 0.0 would be averaged in by
    # `/= bal_calls` and drag the reported entropy toward zero for free.
    st = out[4]
    _CAP.append((out[1], st["router_calls"], st.get("entropy_term"),
                 st.get("switch_aux_term")))
    return out


MoEBlock.forward = _capture
try:
    _CAP.clear()
    cap_tok = run(build(ROUTING_PERSISTENCE_PER_TOKEN, fixed_depth=True))
    calls_tok = list(_CAP)
    _CAP.clear()
    cap_leg = run(build(ROUTING_PERSISTENCE_LEGACY, fixed_depth=True))
    calls_leg = list(_CAP)
finally:
    MoEBlock.forward = _orig

routed_tok = [c for c in calls_tok if c[1] == 1.0]
check("TLX12f per_token: exactly one block call reports a balance term",
      len(calls_tok) == MAXD and len(routed_tok) == 1
      and all(c[0] is None for c in calls_tok[1:]),
      f"{len(calls_tok)} calls, {len(routed_tok)} routed, "
      f"{sum(1 for c in calls_tok if c[0] is None)} return None")
check("TLX12g ... and the aggregate equals that ONE term, undiluted",
      abs(float(cap_tok['bal']) - float(routed_tok[0][0])) < 1e-6,
      f"aggregate={float(cap_tok['bal']):.6f} single_call={float(routed_tok[0][0]):.6f}")
# ABSENT, not 0.0. A 0.0 is a number the aggregator would happily average in; an
# absent key forces every consumer to decide explicitly what a non-routing step means.
check("TLX12f2 ... and reports no entropy/switch statistics on those calls",
      all(c[2] is None and c[3] is None for c in calls_tok[1:])
      and calls_tok[0][2] is not None,
      f"depth1 H={calls_tok[0][2]:.4f}; depths 2..{MAXD} report None")
check("TLX12g2 ... which is NOT the same as that term divided by max_depth",
      abs(float(cap_tok['bal']) - float(routed_tok[0][0]) / MAXD) > 1e-4,
      f"/max_depth would be {float(routed_tok[0][0]) / MAXD:.6f}")
_ent_mean = sum(c[2] for c in calls_leg) / len(calls_leg)
check("TLX12h per_step aggregates the mean over its max_depth terms (control)",
      len(calls_leg) == MAXD
      and abs(cap_leg["route_stats"]["entropy_term"] - _ent_mean) < 1e-5,
      f"{len(calls_leg)} routed calls, mean H={_ent_mean:.6f}, "
      f"reported {cap_leg['route_stats']['entropy_term']:.6f}")
# Normalized entropy is reported as H/log(E) elsewhere; here the raw term only has to
# be finite and positive -- a None leaking through the new conditional branches would
# surface as a TypeError or a nan, which is the regression this catches.
check("TLX12h2 both aggregates are finite, not None-contaminated",
      all(torch.isfinite(torch.as_tensor(float(v)))
          for v in (cap_tok["bal"], cap_leg["bal"],
                    rs_tok["entropy_term"], rs_tok["switch_aux_term"])),
      f"per_token H={rs_tok['entropy_term']:.4f} aux={rs_tok['switch_aux_term']:.4f}")


print("\n=== T-LX.12  persistence does not orphan the router ===")

# The failure this rules out: if the router fires once and its gate is then reused as
# a constant, a careless implementation (e.g. caching `gate.detach()`) leaves the
# router with no path from the task loss at all, and the balance loss becomes its only
# supervision. The router would then optimise load balance alone -- collapse dressed
# up as specialization. Gradient magnitude comparable to legacy is the evidence that
# task gradient still arrives.
grads = {}
for mode in (ROUTING_PERSISTENCE_PER_TOKEN, ROUTING_PERSISTENCE_LEGACY):
    m = build(mode, fixed_depth=True)
    out = run(m)
    loss = out["logits"].pow(2).mean() + out["bal"]
    m.zero_grad(set_to_none=True)
    loss.backward()
    g = m.blocks[0].moe_block.router.weight.grad
    grads[mode] = None if g is None else float(g.abs().sum())
check("TLX12i per_token still delivers gradient to the router",
      grads[ROUTING_PERSISTENCE_PER_TOKEN] is not None
      and grads[ROUTING_PERSISTENCE_PER_TOKEN] > 1e-6,
      f"|grad|_1 = {grads[ROUTING_PERSISTENCE_PER_TOKEN]}")
_ratio = (grads[ROUTING_PERSISTENCE_PER_TOKEN]
          / max(1e-12, grads[ROUTING_PERSISTENCE_LEGACY]))
check("TLX12i2 ... of the same order as legacy, so nothing was detached away",
      0.1 < _ratio < 10.0,
      f"per_token {grads[ROUTING_PERSISTENCE_PER_TOKEN]:.2f} vs "
      f"per_step {grads[ROUTING_PERSISTENCE_LEGACY]:.2f} (ratio {_ratio:.3f})")


print("\n=== T-LX.12  the halt head follows the persisted expert ===")

# THE COMPOUNDING HALF OF T-LX.11. Halt heads are per-expert and selected by
# `expert_idx`, so re-routing re-drew the halting policy too and no head ever saw one
# token's full trajectory (98.4% forced-exit on the canonical MoRE arm).
#
# The probe makes the heads constant and mutually contradictory: expert 0's head says
# "halt now" (bias +20, p ~ 1), every other head says "never halt" (bias -20, p ~ 0),
# and all weights are zeroed so the state cannot influence the decision. Under
# per_token the exit depth is then FULLY DETERMINED by the depth-1 expert: 1 if it is
# expert 0, else forced out at max_depth. Under per_step a token that starts on expert
# 3 can be handed expert 0's head later and halt there, so the prediction breaks --
# which is asserted as the control.
def halt_probe(persistence):
    m = build(persistence, fixed_depth=False)
    with torch.no_grad():
        for e, head in enumerate(m.blocks[0].expert_halt_heads):
            head.weight.zero_()
            head.bias.fill_(20.0 if e == 0 else -20.0)
    out = run(m)
    exit_depth = out["depth_exits"].argmax(dim=1) + 1        # [N]
    first = out["first_route"].long()
    want = torch.where(first == 0, torch.ones_like(exit_depth),
                       torch.full_like(exit_depth, MAXD))
    return exit_depth, want, first, out["halt"]


ed_t, want_t, first_t, halt_t = halt_probe(ROUTING_PERSISTENCE_PER_TOKEN)
ed_l, want_l, first_l, halt_l = halt_probe(ROUTING_PERSISTENCE_LEGACY)
_n0 = int((first_t == 0).sum())
check("TLX12j the probe is informative (both halt policies are exercised)",
      0 < _n0 < first_t.numel(),
      f"{_n0}/{first_t.numel()} tokens start on the halt-now expert")
check("TLX12j1 per_token: exit depth is decided by the depth-1 expert's head",
      bool(torch.equal(ed_t, want_t)),
      f"{int((ed_t == want_t).sum())}/{ed_t.numel()} tokens match the prediction")
check("TLX12j2 per_step: the head is re-drawn with the expert (control)",
      not bool(torch.equal(ed_l, want_l)),
      f"{int((ed_l == want_l).sum())}/{ed_l.numel()} match; "
      f"{int((ed_l != want_l).sum())} tokens halted on a head they were not routed to")
# Same probe, read through the metric the matrix actually reports, so the fix is
# visible in the artifact and not only in a tensor: with 5/6 of the population on
# "never halt" heads the per_token forced-exit rate must be exactly 1 - P(expert 0).
_want_forced = 1.0 - _n0 / first_t.numel()
check("TLX12j3 ... and forced_exit_rate reflects the persisted policy exactly",
      abs(halt_t["forced_exit_rate"] - _want_forced) < 1e-9,
      f"reported {halt_t['forced_exit_rate']:.4f}, predicted {_want_forced:.4f}")


print("\n=== T-LX.12  misuse is refused, not silently accepted ===")

did, named, msg = raises(
    lambda: build(ROUTING_PERSISTENCE_PER_TOKEN, routing_mode=DENSE_ABLATION_ROUTING_MODE),
    "routing_persistence")
check("TLX12k per_token + dense routing is refused",
      did and named, msg)
did, named, msg = raises(lambda: build("per_forward"), "routing_persistence")
check("TLX12k2 an unknown persistence mode is refused",
      did and named, msg)
# A route tensor whose row count disagrees with the active set means the caller
# (a diagnostic, a future refactor) has lost track of which tokens are still running.
# Dispatching on it would mix tokens' experts, so it must raise rather than broadcast.
_blk = build(ROUTING_PERSISTENCE_PER_TOKEN).blocks[0].moe_block
_x = torch.randn(5, D_MODEL)
did, named, msg = raises(
    lambda: _blk(_x, route=(torch.zeros(4, dtype=torch.long), torch.ones(4))),
    "route")
check("TLX12k3 a route/activation row-count mismatch is refused",
      did and named, msg)
check("TLX12k4 a well-formed route of the right width is accepted",
      _blk(_x, route=(torch.zeros(5, dtype=torch.long), torch.ones(5)))[2].numel() == 5,
      "5 rows in, 5 expert indices out")
# The diagnostic wraps MoEBlock.forward and must forward `route`; a wrapper with the
# old (self, x) signature raises TypeError on a per_token model. Asserting the keyword
# by name keeps diag_route_stability.py runnable.
check("TLX12k5 MoEBlock.forward still takes `route` by keyword",
      "route" in __import__("inspect").signature(MoEBlock.forward).parameters,
      "signature(self, x, route=None)")


print("\n=== T-LX.12  config identity: who gets the key, and who must not ===")

LANG_CONFIG = os.path.join(CODE, "config_language.json")
ARITH_CONFIG = os.path.join(CODE, "config.json")


def resolved(path, arch, task):
    """cli.py's call order: load, apply_architecture, apply_task."""
    cfg = load_config(path)
    apply_architecture(cfg, arch)
    apply_task(cfg, task)
    if task == TASK_LANGUAGE:
        stamp_language_dataset_versions(cfg)
    return cfg


for arch in ARCHITECTURES:
    cfg = resolved(LANG_CONFIG, arch, TASK_LANGUAGE)
    got = cfg["model"].get("routing_persistence")
    want = canonical_routing_persistence(TASK_LANGUAGE, arch)
    if arch == "more":
        check(f"TLX12l[{arch}] canonical language MoRE is stamped per_token",
              got == ROUTING_PERSISTENCE_PER_TOKEN, f"routing_persistence={got!r}")
    else:
        # ABSENT, not `per_step`. The key is part of the config_hash input, so writing
        # a legacy default into MoE and MoR would move the hash of all 10 already-
        # published cells for an axis that cannot affect them (MoE has max_depth=1 so
        # there is no second depth to persist to; MoR has num_experts=1 so argmax is
        # constant). Same reasoning as `load_config_defaults` deliberately not
        # setdefault-ing `task` for arithmetic (config.py).
        check(f"TLX12l[{arch}] language {arch.upper()} does NOT carry the key",
              got is None and want == ROUTING_PERSISTENCE_LEGACY,
              f"absent; canonical value is the legacy default {want!r}")
    check(f"TLX12l2[{arch}] ... and still resolves to the canonical variant",
          resolve_variant(cfg) == "language", f"variant={resolve_variant(cfg)!r}")

for arch in ARCHITECTURES:
    cfg = resolved(ARITH_CONFIG, arch, TASK_ARITHMETIC)
    check(f"TLX12m[{arch}] arithmetic is untouched by the language freeze",
          cfg["model"].get("routing_persistence") is None
          and resolve_variant(cfg) == "canonical",
          f"key absent, variant={resolve_variant(cfg)!r}")

# THE INTENDED ASYMMETRY, asserted so it cannot be mistaken for a bug later. The five
# published L7.1 MoRE cells store `variant = "language"`; resolving their architecture
# TODAY yields a deviation tag, because the canonical value for that arm has moved.
# That is what marks them as the labelled per-step ablation (CLAUDE.md 6) rather than
# stale canonical rows. Their resolved_config.json is never rewritten.
_legacy = resolved(LANG_CONFIG, "more", TASK_LANGUAGE)
_legacy["model"]["routing_persistence"] = ROUTING_PERSISTENCE_LEGACY
check("TLX12n a per_step MoRE now resolves as a labelled deviation",
      resolve_variant(_legacy) == "language+route_per_step",
      f"variant={resolve_variant(_legacy)!r}")
_mor_dev = resolved(LANG_CONFIG, "mor", TASK_LANGUAGE)
_mor_dev["model"]["routing_persistence"] = ROUTING_PERSISTENCE_PER_TOKEN
check("TLX12n2 ... and setting it on MoR is also tagged, not silently absorbed",
      resolve_variant(_mor_dev) == "language+route_per_token",
      f"variant={resolve_variant(_mor_dev)!r}")

# The proxy guard reads the field, so a per_step MoRE cannot claim canonical any more.
# "Refused" is not enough -- it must NAME the field, or an operator cannot tell which
# of nineteen enforced fields they perturbed (the TL7.1j contract).
SPEC = load_canonical_spec(task=TASK_LANGUAGE)
_claim = copy.deepcopy(_legacy)
_claim.setdefault("provenance", {})["seed"] = 42
_claim.setdefault("logging", {})["experiment_group"] = SPEC["canonical_group"]
did, named, msg = raises(lambda: assert_not_silent_proxy(copy.deepcopy(_claim)),
                         "routing_persistence")
if _claim["data"].get("dataset_version") is None:
    skip("TLX12o a per_step MoRE is refused a canonical claim",
         "wikitext-103 not built here, so the null dataset_version refuses first")
else:
    check("TLX12o a per_step MoRE is refused a canonical claim, by name",
          did and named, msg)
    _ok = copy.deepcopy(_claim)
    _ok["model"]["routing_persistence"] = ROUTING_PERSISTENCE_PER_TOKEN
    did2, _, msg2 = raises(lambda: assert_not_silent_proxy(copy.deepcopy(_ok)), "")
    check("TLX12o2 ... while the frozen per_token MoRE is accepted",
          not did2, "accepted" if not did2 else msg2)


print("\n=== T-LX.12  no published run's config_hash moved ===")

# The real regression test for a schema change: recompute the hash of every canonical
# run on disk and compare it with the one stamped at run time. A change that moves a
# published hash silently detaches those artifacts from their own provenance, and the
# exporter's refusal to mix config hashes then fires on runs that never changed.
_checked, _moved = 0, []
for name in sorted(os.listdir(os.path.join(REPO, "runs"))) if os.path.isdir(
        os.path.join(REPO, "runs")) else []:
    rc_path = os.path.join(REPO, "runs", name, "resolved_config.json")
    if not os.path.exists(rc_path):
        continue
    try:
        rc = json.loads(open(rc_path, encoding="utf-8").read())
    except Exception:
        continue
    prov = rc.get("provenance", {})
    if prov.get("experiment_group") not in ("canonical_phase_b", "canonical_lang_b"):
        continue
    stamped = prov.get("config_hash")
    if not stamped:
        continue
    _checked += 1
    if config_hash(rc) != stamped:
        _moved.append(name)
if _checked == 0:
    skip("TLX12p canonical config_hashes are unchanged", "no canonical runs in runs/")
else:
    check(f"TLX12p all {_checked} canonical runs still hash to their stamped value",
          not _moved, f"MOVED: {_moved[:4]}" if _moved else "0 moved")


print(f"\n{len(PASS)} passed, {len(FAIL)} failed, {len(SKIP)} skipped")
if FAIL:
    print("FAILED:")
    for n in FAIL:
        print(f"  - {n}")
sys.exit(1 if FAIL else 0)
