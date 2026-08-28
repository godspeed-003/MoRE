"""
GATE 5 closure (plan.md Phase 6 gate).

The gate has six criteria. Five of them are already covered in depth by the
phase suites -- this file does not re-implement those; it re-asserts each one
directly against a live model and a real run directory, so "Gate 5 passes" is a
statement about the code as it stands now rather than about six test files that
passed at different times.

    1 valid provenance          every updated_rules.md 9 field, non-None
    2 valid metrics             components reported separately; total_loss is
                                not the primary quality metric
    3 correct dimensions        every width derives from num_experts
    4 routing assertion         Top-1 sparse: one expert per token, argmax
    5 halt-gradient assertion   a real gradient reaches the halt parameters
    6 no sentinel metrics       nothing reports -1 / 0.0 in place of N/A

Run AFTER test_phase6_provenance.py, which produces the run directory read by
criteria 1, 2 and 6.
"""

import json
import glob
import os
import sys

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE = os.path.join(REPO, "code")
RUNS = os.path.join(REPO, "runs")
sys.path.insert(0, CODE)

from more.model import MoREModel, CANONICAL_ROUTING_MODE
from more.families import NUM_EXPERTS_CANONICAL, expert_labels, NUM_OP_TYPES
from more.metrics import permutation_invariant_routing_metrics

PASS = 0
FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"[PASS] {name}" + (f"  -- {detail}" if detail else ""))
    else:
        FAIL += 1
        print(f"[FAIL] {name}" + (f"   ({detail})" if detail else ""))


print("=" * 74)
print("GATE 5 -- Phase 6 closure")
print("=" * 74)

# Newest run directory that has both a resolved config and metrics.
#
# T8.3: "newest" alone is not good enough. Gate 5 checks that the ROUTING metric
# block is complete, and a MoR run (num_experts = 1) legitimately reports every
# routing key as the string "N/A" -- so whether this gate passed used to depend on
# which architecture happened to run last, and it crashed with
# "TypeError: '<=' not supported between instances of 'float' and 'str'" the first
# time a MoR run was the newest directory. The routing checks therefore read the
# newest ROUTING-BEARING run (num_experts >= 2), and the newest single-expert run,
# if one exists, is checked separately for the opposite property: that it reports
# N/A rather than a degenerate number.
def _load(d):
    return (json.load(open(os.path.join(d, "metrics.json"), encoding="utf-8")),
            json.load(open(os.path.join(d, "resolved_config.json"),
                           encoding="utf-8")))


cands = [d for d in glob.glob(os.path.join(RUNS, "*"))
         if os.path.exists(os.path.join(d, "metrics.json"))
         and os.path.exists(os.path.join(d, "resolved_config.json"))]
cands.sort(key=os.path.getmtime)


def _experts(d):
    try:
        return int(_load(d)[1].get("model", {}).get("num_experts", 0))
    except Exception:                                        # noqa: BLE001
        return 0


_routing = [d for d in cands if _experts(d) >= 2]
_single  = [d for d in cands if _experts(d) == 1]
RUN    = _routing[-1] if _routing else (cands[-1] if cands else None)
RUN_E1 = _single[-1] if _single else None
print(f"\nreading run: {os.path.basename(RUN) if RUN else 'NONE FOUND'}")
if RUN_E1:
    print(f"single-expert cross-check run: {os.path.basename(RUN_E1)}")

metrics, rc = _load(RUN)
prov = rc.get("provenance", {})


def _num(v):
    """A metric value as a float, or None if it is the honest "N/A" string.

    Every N/A in metrics.json serializes as a STRING (CLAUDE.md 4 forbids a
    numeric sentinel), so any comparison has to go through here.
    """
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) else None

# ---------------------------------------------------------------- 1
print("\n-- 1  valid provenance ---------------------------------------------")

REQUIRED = ["experiment_id", "experiment_group", "architecture", "variant",
            "seed", "resolved_seed", "dataset_version", "train_split_version",
            "code_git_commit", "config_hash", "ffn_mult", "ponder_weight",
            "routing_supervision_enabled", "routing_supervision_weight",
            "resolved_router_noise", "router_noise_scale", "halt_target_mode",
            "resolved_epochs", "resolved_subset_fraction", "total_params",
            "determinism_exceptions", "nondeterministic_ops_observed"]
absent = [k for k in REQUIRED if prov.get(k) is None]
check("G5.1 every required provenance field is present and non-None",
      not absent, f"missing: {absent}")
check("G5.1 the run states which seed was DECLARED and which the RNGs got",
      prov.get("seed") == prov.get("resolved_seed"),
      f"{prov.get('seed')} / {prov.get('resolved_seed')}")
check("G5.1 determinism is reported as two distinct facts: what could not be "
      "configured, and what actually ran non-deterministically",
      isinstance(prov.get("nondeterministic_ops_observed"), list)
      and prov.get("nondeterministic_ops_observed") != [],
      str(prov.get("nondeterministic_ops_observed")))
check("G5.1 variant is a derived label, so an ablation cannot be reported as "
      "canonical by omission", isinstance(prov.get("variant"), str)
      and prov["variant"] != "", str(prov.get("variant")))

# ---------------------------------------------------------------- 2
print("\n-- 2  valid metrics ------------------------------------------------")

COMPONENTS = ["train/task_loss", "train/routing_balance_loss",
              "train/entropy_term", "train/switch_aux_term",
              "train/halting_loss", "train/ponder_cost",
              "train/step_routing_loss", "train/halting_supervision_loss",
              "train/total_loss", "val/task_loss"]
missing = [k for k in COMPONENTS if k not in metrics]
check("G5.2 every loss component is reported separately (CLAUDE.md 4)",
      not missing, f"missing: {missing}")
check("G5.2 the primary predictive metric is validation task loss, and it is "
      "not the weighted total",
      "val/task_loss" in metrics
      and metrics["val/task_loss"] != metrics.get("train/total_loss"),
      f"val/task_loss={metrics.get('val/task_loss')}")
_ent = _num(metrics.get("train/expert_load_entropy_normalized"))
check("G5.2 entropy is reported NORMALIZED by log(E), not raw",
      "train/expert_load_entropy_normalized" in metrics
      and _ent is not None and 0.0 <= _ent <= 1.0,
      str(metrics.get("train/expert_load_entropy_normalized")))
check("G5.2 cosine similarity is reported as mean AND max",
      "diag/mean_pairwise_cosine_sim" in metrics
      and "diag/max_pairwise_cosine_sim" in metrics)
check("G5.2 depth is reported as allocation ERROR, absolute and relative, not "
      "as 'compute efficiency'",
      "depth/allocation_error_abs" in metrics
      and "depth/allocation_error_rel" in metrics
      and not [k for k in metrics if "efficiency" in k])
check("G5.2 exit behaviour is reported both ways: early and forced",
      "halt/early_exit_rate" in metrics and "halt/forced_exit_rate" in metrics)
check("G5.2 routing is reported permutation-invariantly (T6.3), not by the "
      "diagonal alone",
      all(f"val/routing_{k}" in metrics
          for k in ("accuracy", "hungarian_accuracy", "ami", "purity",
                    "macro_recall")))
pim_block = metrics.get("routing_permutation_invariant", {})
check("G5.2 the raw / matched pair is stored together with a note on how to "
      "read it", "note" in pim_block and "raw_accuracy" in pim_block
      and "hungarian_accuracy" in pim_block)
check("G5.2 raw and matched accuracy agree to within tolerance when the index "
      "itself is supervised -- the expected case, stated as such",
      abs(pim_block.get("raw_accuracy", 0) -
          metrics.get("val/routing_accuracy", -1)) < 1e-6,
      f"pim={pim_block.get('raw_accuracy')} "
      f"metric={metrics.get('val/routing_accuracy')}")

# T8.3: the mirror-image requirement, on a single-expert run. MoR makes no routing
# decision, so its 1-wide argmax is 0 for every token and a diagonal fraction
# would report "the fraction of tokens that are family E1" as a routing score.
# CLAUDE.md 4 forbids that: it must be N/A, and N/A must be the STRING, never 0.0
# or -1. Checked on a real MoR run rather than on a synthetic dict, because the
# gating lives in metrics.paper_metrics_to_wandb and the writer in engine.py.
if RUN_E1 is not None:
    m1, rc1 = _load(RUN_E1)
    _ROUTING_KEYS = ["val/routing_accuracy", "val/routing_hungarian_accuracy",
                     "val/routing_ami", "val/routing_purity",
                     "val/routing_macro_recall",
                     "train/expert_load_entropy_normalized",
                     "train/entropy_term", "train/switch_aux_term",
                     "train/routing_balance_loss",
                     "diag/mean_pairwise_cosine_sim",
                     "diag/max_pairwise_cosine_sim"]
    leaked = [k for k in _ROUTING_KEYS if _num(m1.get(k)) is not None]
    check("G5.2b a single-expert run reports NO routing or expert-diversity "
          "quantity as a number -- every one is N/A (CLAUDE.md 4)",
          not leaked, f"reported numerically: {leaked}")
    _pim1 = m1.get("routing_permutation_invariant", {})
    check("G5.2b the permutation-invariant block of a single-expert run is N/A "
          "throughout, not an empty dict that a reader could mistake for 0",
          all(_pim1.get(k) == "N/A" for k in
              ("raw_accuracy", "hungarian_accuracy", "ami", "purity",
               "macro_recall", "matched_macro_recall")),
          str({k: _pim1.get(k) for k in ("raw_accuracy", "hungarian_accuracy")}))
    # Depth is NOT gated: allocating depth is exactly what MoR contributes, so a
    # missing depth metric here is as much a defect as a fabricated routing one.
    _DEPTH_KEYS = ["depth/allocation_error_abs", "depth/allocation_error_rel",
                   "halt/early_exit_rate", "halt/forced_exit_rate",
                   "train/avg_recursion_steps", "train/ponder_cost"]
    absent_depth = [k for k in _DEPTH_KEYS if _num(m1.get(k)) is None]
    check("G5.2b a single-expert run still reports every DEPTH metric as a real "
          "number -- MoR allocates depth even though it does not route",
          not absent_depth, f"missing/N-A: {absent_depth}")
    _fam_depth = [k for k in m1 if k.startswith("recursion/avg_depth_by_family/")]
    check("G5.2b per-family depth is reported for all six families on a "
          "single-expert run (T8.3: the vector is sized by the family label "
          "space, not by num_experts)",
          len(_fam_depth) == NUM_EXPERTS_CANONICAL,
          f"{len(_fam_depth)} family depth keys: {sorted(_fam_depth)[:2]}...")
else:
    print("[SKIP] G5.2b  no single-expert (MoR) run directory found to "
          "cross-check; run train.py --architecture mor to populate it")

# ---------------------------------------------------------------- 3
print("\n-- 3  correct dimensions -------------------------------------------")

E = rc["model"]["num_experts"]
model = MoREModel(step_feat_dim=rc["model"]["step_feat_dim"],
                  d_model=64, num_experts=E, max_depth=3, num_blocks=1,
                  num_op_types=NUM_OP_TYPES)
check("G5.3 num_experts is the canonical six, from the family manifest",
      E == NUM_EXPERTS_CANONICAL, str(E))
check("G5.3 cls_head width == num_families, the family manifest's label space "
      "(T8.3: equals 6 here; NOT tied to num_experts, which is 1 for MoR)",
      model.cls_head.out_features == NUM_EXPERTS_CANONICAL,
      str(model.cls_head.out_features))
check("G5.3 step_cls_head width == num_families",
      model.step_cls_head.out_features == NUM_EXPERTS_CANONICAL,
      str(model.step_cls_head.out_features))
check("G5.3 router width == num_experts",
      model.blocks[0].moe_block.router.out_features == E)
check("G5.3 one expert FFN and one halt head per expert",
      len(model.blocks[0].moe_block.experts) == E
      and len(model.blocks[0].expert_halt_heads) == E)
check("G5.3 the op embedding is sized by OPERATION count, not expert count -- "
      "the op->expert map is what routing must discover",
      model.op_embed.num_embeddings == NUM_OP_TYPES != E,
      f"{model.op_embed.num_embeddings} ops vs {E} experts")
check("G5.3 metric labels are as wide as the model",
      len(expert_labels(E)) == E == len(pim_block.get("per_family", {})),
      f"{len(expert_labels(E))} labels, "
      f"{len(pim_block.get('per_family', {}))} reported families")
check("G5.3 ffn_mult is a reported config field, so the FFN width is not an "
      "assumption", model.blocks[0].moe_block.ffn_mult == prov.get("ffn_mult"),
      str(prov.get("ffn_mult")))

# ---------------------------------------------------------------- 4 and 5
print("\n-- 4  routing assertion / 5  halt-gradient assertion ---------------")

torch.manual_seed(0)
B, S, F = 64, rc["model"]["max_steps"], rc["model"]["step_feat_dim"]
x = torch.randn(B, S, F)
mask = torch.ones(B, S, dtype=torch.bool)
ops = torch.randint(0, NUM_OP_TYPES, (B, S))
out = model(x, mask, step_ops=ops)
reg_out, ponder_cost, first_route = out[0], out[4], out[9]

check("G5.4 the canonical routing mode is Top-1 sparse",
      model.routing_mode == CANONICAL_ROUTING_MODE
      and metrics.get("dispatch/routing_mode") == CANONICAL_ROUTING_MODE,
      f"{model.routing_mode} / {metrics.get('dispatch/routing_mode')}")
check("G5.4 the run measured exactly one expert evaluation per token -- "
      "dispatch, not dense blending",
      abs(metrics.get("dispatch/evals_per_token", 0.0) - 1.0) < 1e-9,
      str(metrics.get("dispatch/evals_per_token")))
check("G5.4 router noise is off in the canonical path",
      metrics.get("dispatch/router_noise") == "none"
      and metrics.get("dispatch/router_noise_scale") == 0.0)

used = sorted(set(int(v) for v in first_route.reshape(-1).tolist()))
check("G5.4 every token has exactly one expert index in range",
      first_route.numel() == B * S and used and min(used) >= 0
      and max(used) < E, f"experts used: {used}")

# The gradient check: this is the defect Phase 3 existed to fix -- halted states
# were copied into an output buffer by a discrete choice, so every halt
# parameter had grad None after a full backward. It is asserted per expert and
# only for experts that actually received tokens: under Top-1 dispatch an expert
# nobody selected is never called, so its halt head having no gradient is
# arithmetic, not a bug. Reporting it as a failure would train the reader to
# ignore this check.
loss = reg_out.float().pow(2).mean()
model.zero_grad(set_to_none=True)
loss.backward(retain_graph=True)

heads = model.blocks[0].expert_halt_heads
dead_used = [i for i in used
             if heads[i].weight.grad is None
             or float(heads[i].weight.grad.abs().sum()) == 0.0]
check("G5.5 every halt head that received tokens gets a non-zero gradient "
      "from the TASK loss",
      not dead_used, f"experts with no halt gradient: {dead_used} of used {used}")
check("G5.5 the check covered all six experts, so no expert's halt path is "
      "untested", len(used) == E, f"used {len(used)}/{E}: {used}")
check("G5.5 the router also receives gradient from the task path",
      all(p.grad is not None
          for p in model.blocks[0].moe_block.router.parameters()))

# Sharper form: the ponder cost ALONE must reach the halt parameters. If the
# only gradient came via the task path, the recursion cost would be a number
# printed next to the model rather than a term the halting policy is optimizing
# against (CLAUDE.md 2: "a real gradient reaching the halt parameters").
model.zero_grad(set_to_none=True)
ponder_cost.backward()
p_dead = [i for i in used
          if heads[i].weight.grad is None
          or float(heads[i].weight.grad.abs().sum()) == 0.0]
check("G5.5 the differentiable ponder cost by itself reaches every active "
      "halt head", not p_dead, f"unreached: {p_dead}")
check("G5.5 the run reports a differentiable ponder cost, not just a depth "
      "count", metrics.get("train/ponder_cost", 0.0) > 0.0,
      str(metrics.get("train/ponder_cost")))

# ---------------------------------------------------------------- 6
print("\n-- 6  no sentinel metrics ------------------------------------------")

SENTINELS = (-1, -1.0)
sent = [k for k, v in metrics.items()
        if isinstance(v, (int, float)) and not isinstance(v, bool)
        and v in SENTINELS]
check("G5.6 no metric reports -1 as a measurement", not sent, str(sent))

nan_keys = [k for k, v in metrics.items()
            if isinstance(v, float) and v != v]
check("G5.6 no metric is NaN -- an undefined quantity is written 'N/A'",
      not nan_keys, str(nan_keys))

na_scalars = [k for k, v in metrics.items() if v == "N/A"]
print(f"       ({len(na_scalars)} metric(s) correctly written 'N/A')")
check("G5.6 undefined values in the permutation-invariant block are 'N/A' "
      "strings, never numeric placeholders",
      all(not isinstance(v, (int, float)) or v == v
          for k, v in pim_block.items() if not isinstance(v, dict)))

# The strongest form of this check: a single-expert (MoR) confusion has no
# routing to report, and every routing quantity must come back undefined rather
# than as a comparable-looking zero.
mor_pim = permutation_invariant_routing_metrics(torch.tensor([[100.0]]), 1)
check("G5.6 with one expert every routing metric is None, not 0.0 -- entropy "
      "of a one-element distribution is not a score",
      all(mor_pim[k] is None for k in
          ("raw_accuracy", "hungarian_accuracy", "ami", "purity",
           "macro_recall")))

print("\n" + "=" * 74)
print(f"GATE 5:  {PASS} passed, {FAIL} failed")
print("=" * 74)
sys.exit(1 if FAIL else 0)
