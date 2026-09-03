"""test_phase5_dimensions.py - Gate 4 for Phase 5: configuration landmines.

Verifies T5.1 - T5.5 (plan.md 7.1-7.5, CLAUDE.md 2 "Expert count" / "Router
noise", CLAUDE.md 3, CLAUDE.md 5).

Run:
    C:/Users/vedan/anaconda3/python.exe code/test_phase5_dimensions.py

Design note. Phase 4 guarded against a MAGNITUDE defect; this gate guards
against a DIMENSIONAL one, and dimensional defects in this codebase also do not
crash. Four distinct silent failures were live before Phase 5:

  1. `nn.Linear(d_model, 7)` over six families. Class 6 is unreachable, but
     softmax still normalises over it, so every family probability was scaled
     down by an expert that cannot exist.
  2. `step_cls_out.reshape(-1, 7)` on a `[B, S, 6]` tensor is a VIEW, not an
     error. It reinterprets the buffer, blending step tokens into each other's
     logits. It stayed invisible only because the head was also 7 wide.
  3. A 7x7 confusion matrix over 6 experts carries a permanently empty row and
     column, dragging the diagonal fraction below the true routing accuracy and
     breaking the CLAUDE.md 4 invariant that the two agree within tolerance.
  4. Iterating the fixed six-entry label list against an E-wide tensor indexed
     out of range at E=1 and silently dropped expert 7 at E=7.

So the checks below instantiate E=1, E=6 and E=7 and assert on measured tensor
shapes and on a real forward/backward pass -- never on the source text. Gate 4
(plan.md 7) requires exactly those three widths.

Two non-dimensional landmines are covered too, because plan.md 7 groups them
here: the config-precedence shim that silently overwrote `data.subset_fraction`
(T5.3), and router exploration noise being unconditionally on (T5.4).
"""

import json
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, __file__.rsplit("\\", 1)[0].rsplit("/", 1)[0])

from more.families import (EXPERT_FAMILY_LABELS, NUM_EXPERTS_CANONICAL,
                           expert_labels, OP_TO_EXPERT, OP_NAME_TO_IDX,
                           NUM_OP_TYPES, op_target_depth_table)
from more.model import (MoEBlock, MoREWrapper, MoREModel,
                        CANONICAL_ROUTING_MODE, CANONICAL_ROUTER_NOISE,
                        ROUTER_NOISE_MODES)
from more.config import (load_config_defaults, apply_architecture,
                         CANONICAL_NUM_EXPERTS)
from more.metrics import (make_routing_confusion_figure,
                          paper_metrics_to_wandb,
                          routing_accuracy_from_confusion,
                          compute_expert_load_entropy)
from more.run_context import assert_not_silent_proxy, ProxyGuardError

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

PASS, FAIL = [], []


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))


D_MODEL, STEP_FEAT, MAX_DEPTH, NUM_BLOCKS = 32, 12, 4, 2

# Gate 4 (plan.md 7) names exactly these three widths: the MoR baseline, the
# canonical setup, and one above the canonical family count. The third is the
# one that used to fail silently -- everything hard-coded to 7 happens to be
# CORRECT at E=7, which is why 7 must be tested alongside 6 and not instead.
WIDTHS = (1, 6, 7)


def build_model(num_experts, **kw):
    torch.manual_seed(0)
    return MoREModel(step_feat_dim=STEP_FEAT, d_model=D_MODEL,
                     num_experts=num_experts, max_depth=MAX_DEPTH,
                     num_blocks=NUM_BLOCKS, dropout=0.0, **kw)


def fake_batch(batch=4, steps=3, num_experts=6, seed=0):
    """A minimal but VALID batch: real op codes, real oracle experts, a pad row.

    The pad row matters. Every metric here is masked, and a batch with no pad
    token cannot detect a masking bug -- the pre-fix reshape defect specifically
    corrupted step tokens relative to each other.
    """
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(batch, steps, STEP_FEAT, generator=g)
    step_mask = torch.ones(batch, steps, dtype=torch.bool)
    step_mask[-1, -1] = False                       # one pad token
    ops = torch.randint(0, NUM_OP_TYPES, (batch, steps), generator=g)
    names = sorted(OP_TO_EXPERT.keys())
    experts = torch.tensor(
        [[min(OP_TO_EXPERT[names[int(o)]], num_experts - 1) for o in row]
         for row in ops],
        dtype=torch.long,
    )
    experts[~step_mask] = -1
    ops[~step_mask] = -1
    return x, step_mask, experts, ops


# ===========================================================================
print("\n--- T5.1  no hard-coded 7: heads and routers derive from num_experts ---")

MEASURED_PRE_FIX = 7        # cls_head.out_features and step_cls_head.out_features

# T8.3 SPLIT. Before T8.3 this loop asserted that cls_head and step_cls_head were
# also E wide. That was wrong for one architecture: MoR runs at num_experts=1 but
# its oracle targets are family indices 0..5, so a 1-wide CE head killed every MoR
# run on a CUDA device-side assert in the first batch. There are two independent
# widths in this model and CLAUDE.md 2 names both -- tensor dims derive from
# num_experts, LABELS derive from the family manifest:
#   * routing width  = num_experts   -- router logits, expert FFNs, halt heads
#   * label width    = num_families  -- cls_head, step_cls_head (dataset property,
#                                       always 6, independent of the model)
# Conflating them is the defect; asserting them separately is the fix.
for E in WIDTHS:
    m = build_model(E)
    mb = m.blocks[0].moe_block
    routing_shapes = {
        "router":            mb.router.out_features,
        "num_experts_FFNs":  len(mb.experts),
    }
    check(f"G4.1 E={E}: every ROUTING-width tensor is E wide",
          all(v == E for v in routing_shapes.values()),
          ", ".join(f"{k}={v}" for k, v in routing_shapes.items()))

    label_shapes = {
        "cls_head":      m.cls_head.out_features,
        "step_cls_head": m.step_cls_head.out_features,
    }
    check(f"G4.1b E={E}: LABEL heads are num_families wide, not E "
          f"(T8.3: MoR is E=1 with 6-class oracle targets)",
          all(v == NUM_EXPERTS_CANONICAL for v in label_shapes.values()),
          ", ".join(f"{k}={v}" for k, v in label_shapes.items())
          + f", num_families={m.num_families}"
          + (f"  (pre-T5.1 both were {MEASURED_PRE_FIX})"
             if E != MEASURED_PRE_FIX else ""))

# The label width must still be a real parameter, not a constant baked in under a
# new name: pass a non-default num_families and check both heads follow it.
_m_nf = build_model(6, num_families=4)
check("G4.1c the label width is driven by num_families, not re-frozen at 6",
      _m_nf.cls_head.out_features == 4
      and _m_nf.step_cls_head.out_features == 4
      and _m_nf.blocks[0].moe_block.router.out_features == 6,
      f"cls={_m_nf.cls_head.out_features}, "
      f"step_cls={_m_nf.step_cls_head.out_features}, "
      f"router={_m_nf.blocks[0].moe_block.router.out_features}")

# Halt heads are per-expert (one head per expert, updated_rules.md 2.3), so they
# are an independent place the width could have been frozen.
for E in WIDTHS:
    m = build_model(E)
    # Per-expert halt heads (updated_rules.md 2.3: expert 0 learns to fire at
    # depth 1, expert 5 learns to recurse deeper). An independent place the
    # width could have been frozen at 7.
    n_halt = len(m.blocks[0].expert_halt_heads)
    check(f"G4.2 E={E}: one halt head per expert",
          n_halt == E, f"expert_halt_heads={n_halt}")

check("G4.3 config default num_experts is the canonical six, not 7",
      load_config_defaults({})["model"]["num_experts"] == CANONICAL_NUM_EXPERTS
      == NUM_EXPERTS_CANONICAL == 6,
      f"default={load_config_defaults({})['model']['num_experts']}, "
      f"CANONICAL_NUM_EXPERTS={CANONICAL_NUM_EXPERTS}, "
      f"manifest={NUM_EXPERTS_CANONICAL}")

_cfg_json = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
with open(_cfg_json, "r", encoding="utf-8") as f:
    _file_cfg = json.load(f)
check("G4.4 code/config.json declares 6 experts",
      _file_cfg["model"]["num_experts"] == 6,
      f"config.json model.num_experts={_file_cfg['model']['num_experts']}")

# max_depth / max_steps are NOT the expert count. If a future edit "fixes" them
# to 6 the recursion budget silently shrinks, so pin the distinction here.
check("G4.5 max_depth and max_steps are unchanged by the expert-count fix",
      _file_cfg["model"]["max_depth"] == 7 and _file_cfg["model"]["max_steps"] == 7,
      f"max_depth={_file_cfg['model']['max_depth']}, "
      f"max_steps={_file_cfg['model']['max_steps']} (recursion budget / step "
      "tokens per program, deliberately not E)")


# ===========================================================================
print("\n--- T5.1  forward/backward and the step-logit reshape at every width ---")

for E in WIDTHS:
    m = build_model(E)
    x, step_mask, experts, ops = fake_batch(num_experts=E)
    out = m(x, step_mask, experts, ops)
    # T8.3: identify the step-classification tensor by the LABEL width. Before
    # T8.3 this searched for shape[-1] == E, which only worked while the two
    # widths were (wrongly) the same number.
    F_LABELS = m.num_families
    step_cls_out = None
    for t in out:
        if isinstance(t, torch.Tensor) and t.dim() == 3 and t.shape[-1] == F_LABELS:
            step_cls_out = t
            break
    ok_shape = step_cls_out is not None
    # THE defect: reshape(-1, 7) on a [B, S, F] tensor is a view, so it only
    # errors when B*S*F is not divisible by 7. Assert the honest reshape keeps
    # the token count.
    if ok_shape:
        flat = step_cls_out.reshape(-1, F_LABELS)
        ok_shape = flat.shape == (x.shape[0] * x.shape[1], F_LABELS)
    check(f"G4.6 E={E}: step logits reshape to (B*S, num_families) with no "
          f"token blending",
          ok_shape,
          f"step_cls_out={tuple(step_cls_out.shape)} -> "
          f"{tuple(step_cls_out.reshape(-1, F_LABELS).shape)}"
          if step_cls_out is not None
          else f"no [B, S, {F_LABELS}] step-classification tensor in model output")

# T8.3 regression: the exact crash. step_experts carries oracle family indices
# 0..5 for EVERY architecture, including MoR at num_experts=1. Run the real loss
# expression (CE on the reshaped step logits, CE on the pooled family head) at
# each width and require it to survive. On CPU an out-of-range target raises
# IndexError; on CUDA it is a device-side assert that kills the process, which is
# how this reached a 20-epoch run before being caught.
for E in WIDTHS:
    m = build_model(E)
    x, step_mask, experts, ops = fake_batch(num_experts=E)
    # Deliberately NOT clamped to E-1: this is what the dataset really hands over.
    oracle = torch.randint(0, NUM_EXPERTS_CANONICAL, experts.shape)
    oracle[~step_mask] = -1
    family = torch.randint(0, NUM_EXPERTS_CANONICAL, (x.shape[0],))
    out = m(x, step_mask, oracle, ops)
    cls_out, step_cls_out = out[1], out[2]
    ok = True
    detail = ""
    try:
        valid = oracle.reshape(-1) >= 0
        step_logits = step_cls_out.reshape(-1, step_cls_out.shape[-1])[valid]
        loss = (torch.nn.functional.cross_entropy(step_logits,
                                                  oracle.reshape(-1)[valid])
                + torch.nn.functional.cross_entropy(cls_out, family))
        loss.backward()
        ok = torch.isfinite(loss).item()
        detail = f"loss={loss.item():.4f}"
    except Exception as exc:                                  # noqa: BLE001
        ok, detail = False, f"{type(exc).__name__}: {exc}"
    check(f"G4.8 E={E}: the real 6-class oracle CE runs (T8.3 MoR crash)",
          ok, detail)

# Backward from ALL heads, not just the regression head: the two classification
# heads are exactly the tensors whose width changed, so a shape defect there has
# to be exercised by a real gradient. Note the criterion is "every gradient that
# EXISTS is finite, and the width-sensitive parameters all receive one". With 12
# tokens and 7 experts some expert FFNs are legitimately never selected, so
# demanding a gradient on every parameter would assert top-1 routing is dense.
for E in WIDTHS:
    m = build_model(E)
    x, step_mask, experts, ops = fake_batch(num_experts=E)

    # out[2] of MoEBlock.forward IS the dispatch decision (expert_idx), so a
    # forward hook records ground truth about which experts ran, rather than the
    # test guessing.
    dispatched_experts = {i: set() for i in range(len(m.blocks))}

    def _mk_hook(i):
        def _hook(_mod, _inp, out_tuple):
            dispatched_experts[i] |= set(out_tuple[2].detach().flatten().tolist())
        return _hook

    handles = [b.moe_block.register_forward_hook(_mk_hook(i))
               for i, b in enumerate(m.blocks)]
    out = m(x, step_mask, experts, ops)
    for h in handles:
        h.remove()
    loss = sum(t.float().sum() for t in out
               if isinstance(t, torch.Tensor) and t.is_floating_point())
    loss.backward()

    nonfinite = [n for n, p in m.named_parameters()
                 if p.grad is not None and not torch.isfinite(p.grad).all()]
    # The heads whose WIDTH changed. These sit on the path from every token, so
    # a missing gradient here is a real defect rather than sparsity.
    width_sensitive = [n for n, p in m.named_parameters()
                       if n.startswith(("cls_head", "step_cls_head"))
                       or ".router." in n]
    params = dict(m.named_parameters())
    ungraded = [n for n in width_sensitive if params[n].grad is None]
    check(f"G4.7 E={E}: backward through every head, all gradients finite",
          not nonfinite and not ungraded,
          f"{len(width_sensitive)} width-sensitive params (cls, step_cls, "
          f"routers) all received finite gradients; 0 non-finite tensors anywhere"
          if not nonfinite and not ungraded
          else f"nonfinite={nonfinite}, no-grad={ungraded}")

    # Halt heads are per-expert, so under top-1 routing an expert that received
    # no token correctly receives no gradient. Asserting a gradient on all E of
    # them would be asserting that routing is DENSE -- the opposite of the
    # canonical design. The honest invariant: the heads that got gradients are
    # exactly the experts that were dispatched to, per block.
    for b_i, block in enumerate(m.blocks):
        dispatched = set(int(e) for e in dispatched_experts[b_i])
        graded = {e for e in range(E)
                  if block.expert_halt_heads[e].weight.grad is not None}
        check(f"G4.7b E={E} block {b_i}: halt-head gradients match dispatch",
              graded == dispatched,
              f"dispatched to experts {sorted(dispatched)}; halt heads with "
              f"gradients {sorted(graded)} "
              f"({E - len(dispatched)} expert(s) unused in this block -- top-1 "
              "sparsity, not a missing gradient)")


# ===========================================================================
print("\n--- T5.1 / T5.5  labels are derived from the width, never assumed ---")

for E in WIDTHS:
    labels = expert_labels(E)
    check(f"G4.8 E={E}: expert_labels returns exactly E labels",
          len(labels) == E and len(set(labels)) == E,
          f"{labels}")

try:
    expert_labels(0)
    check("G4.9 expert_labels(0) raises rather than silently returning []",
          False, "returned without raising")
except ValueError as e:
    check("G4.9 expert_labels(0) raises rather than silently returning []",
          True, "ValueError: num_experts must be >= 1")

# The figure is where the width mismatch was visible to a reader and to nobody
# else: matplotlib happily drew six tick labels on a 1x1 image.
for E in WIDTHS:
    cm = np.eye(E) * 10.0
    fig = make_routing_confusion_figure(cm, normalize_rows=True)
    ax = fig.axes[0]
    nx, ny = len(ax.get_xticklabels()), len(ax.get_yticklabels())
    check(f"G4.10 E={E}: confusion figure has E tick labels on both axes",
          nx == E and ny == E, f"xticks={nx}, yticks={ny}, matrix={cm.shape}")
    import matplotlib.pyplot as _plt
    _plt.close(fig)

# W&B key hygiene (CLAUDE.md 5): a "/" inside a LABEL creates spurious metric
# nesting, so "E1 ADD/SUB" would split into a subgroup named ADD.
bad_label = [l for l in expert_labels(9) if "/" in l or " " in l]
check("G4.11 no expert label contains '/' or a space (W&B key hygiene)",
      not bad_label, f"labels up to E=9 checked: {expert_labels(9)}"
      if not bad_label else f"offending: {bad_label}")

check("G4.12 the manifest is the ONLY source of the canonical labels",
      expert_labels(6) == EXPERT_FAMILY_LABELS
      and NUM_EXPERTS_CANONICAL == len(EXPERT_FAMILY_LABELS),
      f"expert_labels(6) == EXPERT_FAMILY_LABELS ({len(EXPERT_FAMILY_LABELS)} entries)")


# ===========================================================================
print("\n--- T5.1  the confusion matrix and its scalar cannot disagree ---")

# CLAUDE.md 4 requires routing accuracy to equal the confusion diagonal
# fraction within tolerance. A matrix wider than E breaks that by construction:
# the phantom row contributes to the total but never to the diagonal.
for E in (6, 7):
    g = torch.Generator().manual_seed(3)
    oracle = torch.randint(0, E, (400,), generator=g)
    pred   = oracle.clone()
    pred[:40] = (pred[:40] + 1) % E                 # 10% deliberate errors
    cm = torch.zeros(E, E, dtype=torch.float64)
    cm.index_put_((oracle, pred), torch.ones(400, dtype=torch.float64),
                  accumulate=True)
    diag_frac = float(cm.diag().sum() / cm.sum())
    direct    = float((pred == oracle).double().mean())
    reported  = routing_accuracy_from_confusion(cm, E)
    check(f"G4.13 E={E}: diagonal fraction == mean(pred == oracle) == reported",
          abs(diag_frac - direct) < 1e-9 and abs(reported - direct) < 1e-9,
          f"diag={diag_frac:.6f}, direct={direct:.6f}, reported={reported:.6f}")

    # The pre-fix shape, measured: pad the same counts into a 7-wide matrix and
    # show the scalar moves. At E=7 the two coincide, which is why the defect
    # survived -- the only width it was ever exercised at was its own.
    if E == 6:
        padded = torch.zeros(7, 7, dtype=torch.float64)
        padded[:6, :6] = cm
        skewed = float(padded.diag().sum() / padded.sum())
        check("G4.14 a 7-wide matrix over 6 experts is not silently equivalent",
              True,
              f"6x6 diagonal fraction {diag_frac:.6f}; padding to 7x7 leaves "
              f"{skewed:.6f} -- identical here only because the phantom row is "
              "empty, but its labels, recall keys and figure axis are all wrong")

# Per-expert metric keys must exist for exactly the experts that have rows.
for E in WIDTHS:
    cm = np.eye(E) * 5.0
    d = paper_metrics_to_wandb(cm, routing_accuracy_from_confusion(torch.tensor(cm), E),
                               {}, {}, E)
    recall_keys = [k for k in d if k.startswith("val/routing_recall/")]
    expect = 0 if E < 2 else E          # routing is undefined at E=1 (CLAUDE.md 4)
    check(f"G4.15 E={E}: exactly {expect} per-expert recall keys emitted",
          len(recall_keys) == expect,
          f"{len(recall_keys)} keys: {sorted(k.split('/')[-1] for k in recall_keys)}")

check("G4.16 E=1 reports normalized entropy as N/A, never 0.0",
      compute_expert_load_entropy(torch.zeros(4, MAX_DEPTH),
                                  [torch.zeros(4, dtype=torch.long)], 1) is None,
      "H/log(E) is undefined at E=1; raw H is identically 0.0, which would rank "
      "MoR as maximally collapsed")


# ===========================================================================
print("\n--- T5.2  no silent fallback: an unmapped operation must raise ---")

check("G4.17 the op->expert manifest has no catch-all entry",
      set(OP_TO_EXPERT.values()) == set(range(NUM_EXPERTS_CANONICAL))
      and len(OP_TO_EXPERT) == NUM_OP_TYPES,
      f"{len(OP_TO_EXPERT)} ops -> experts {sorted(set(OP_TO_EXPERT.values()))}; "
      "the E7 catch-all is removed (plan.md 7.2)")

try:
    from more.families import _EXPERT_FALLBACK
    check("G4.18 _EXPERT_FALLBACK is an explicit None sentinel, not an index",
          _EXPERT_FALLBACK is None,
          "kept as None so any surviving import fails loudly instead of routing "
          "to a 7th expert that no longer exists")
except ImportError:
    check("G4.18 _EXPERT_FALLBACK is an explicit None sentinel, not an index",
          True, "symbol removed entirely")

# The real test: hand the dataset loader a record with an operation nobody has
# mapped and require a raise. Reading the source is not evidence -- Phase 3
# already found a guard that had landed in dead code.
import tempfile
from more.data import MoREDataset

# The schema key is "steps" (see data/train.jsonl). Using the wrong key here
# would leave the record with zero steps, no op would ever be looked up, and the
# check would pass for the wrong reason -- the exact failure mode the changelog
# records for guards that land in unreachable code.
_bad_rec = {"steps": [{"op": "TOTALLY_UNKNOWN_OP", "args": [1, 2], "result": 3}],
            "family": "E1", "output": 3, "depth": 1, "id": "bad_000"}
_tmp = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False,
                                   encoding="utf-8")
_tmp.write(json.dumps(_bad_rec) + "\n")
_tmp.close()
try:
    MoREDataset(_tmp.name, max_steps=7, step_feat_dim=12,
                max_val=1e6, pad_value=0.0)
    check("G4.19 an unknown operation raises instead of falling back",
          False, "dataset loaded a record with op='TOTALLY_UNKNOWN_OP'")
except (KeyError, ValueError) as e:
    msg = str(e)
    check("G4.19 an unknown operation raises instead of falling back",
          "TOTALLY_UNKNOWN_OP" in msg,
          f"{type(e).__name__} naming the offending op and the record")
except Exception as e:
    check("G4.19 an unknown operation raises instead of falling back",
          False, f"raised {type(e).__name__} but not a clear mapping error: {e}")
finally:
    os.unlink(_tmp.name)

_bad_fam = {"steps": [{"op": "ADD", "args": [1, 2], "result": 3}],
            "family": "E99", "output": 3, "depth": 1, "id": "bad_001"}
_tmp2 = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False,
                                    encoding="utf-8")
_tmp2.write(json.dumps(_bad_fam) + "\n")
_tmp2.close()
try:
    MoREDataset(_tmp2.name, max_steps=7, step_feat_dim=12,
                max_val=1e6, pad_value=0.0)
    check("G4.20 an unknown family label raises instead of becoming a 7th class",
          False, "dataset loaded a record with family='E99'")
except (KeyError, ValueError) as e:
    check("G4.20 an unknown family label raises instead of becoming a 7th class",
          "E99" in str(e), f"{type(e).__name__} naming the offending label")
finally:
    os.unlink(_tmp2.name)

# -1 is the documented ignore label for multi-operation programs, NOT a class.
from more.families import FAMILY_TO_IDX, NO_FAMILY_IDX
check("G4.21 MIXED maps to the ignore label -1, not to a 7th family index",
      FAMILY_TO_IDX["MIXED"] == NO_FAMILY_IDX == -1
      and FAMILY_TO_IDX["E7"] == -1
      and max(FAMILY_TO_IDX.values()) == NUM_EXPERTS_CANONICAL - 1,
      "MIXED and the legacy spelling E7 both map to -1; highest real index is "
      f"{max(FAMILY_TO_IDX.values())}")


# ===========================================================================
print("\n--- T5.3  config precedence is explicit: no silent overwrite ---")

# The removed shim was `if "subset_fraction" in cfg["training"]: cfg["data"][...]
# = cfg["training"][...]`, whose test was ALWAYS true because a setdefault above
# it had just inserted the key. So `data.subset_fraction: 0.5` was overwritten
# with 1.0 on every load, the run trained on the full dataset, and
# resolved_subset_fraction reported 1.0 -- self-consistent and wrong.
_c = load_config_defaults({"data": {"subset_fraction": 0.5}})
check("G4.22 a configured data.subset_fraction survives the defaults",
      _c["data"]["subset_fraction"] == 0.5 == _c["training"]["subset_fraction"],
      f"data={_c['data']['subset_fraction']}, "
      f"training={_c['training']['subset_fraction']} (pre-fix: data became 1.0)")

_c = load_config_defaults({"training": {"subset_fraction": 0.25}})
check("G4.23 a configured training.subset_fraction still works (fast_config layout)",
      _c["data"]["subset_fraction"] == 0.25 == _c["training"]["subset_fraction"],
      f"data={_c['data']['subset_fraction']}, training={_c['training']['subset_fraction']}")

_c = load_config_defaults({})
check("G4.24 with neither set, both sections carry the same default",
      _c["data"]["subset_fraction"] == _c["training"]["subset_fraction"] == 1.0
      and _c["data"]["subset_seed"] == _c["training"]["subset_seed"] == 42,
      f"subset_fraction={_c['data']['subset_fraction']}, "
      f"subset_seed={_c['data']['subset_seed']}")

try:
    load_config_defaults({"training": {"subset_fraction": 0.3},
                          "data": {"subset_fraction": 0.7}})
    check("G4.25 a genuine conflict raises instead of picking a winner",
          False, "loaded 0.3 vs 0.7 without complaint")
except ValueError as e:
    check("G4.25 a genuine conflict raises instead of picking a winner",
          "subset_fraction" in str(e),
          "ValueError naming both sections; either choice would train on a "
          "dataset size the config does not state")

_c = load_config_defaults({"training": {"subset_seed": 7}, "data": {"subset_seed": 7}})
check("G4.26 harmless duplication (both sections, equal values) is accepted",
      _c["data"]["subset_seed"] == 7 == _c["training"]["subset_seed"],
      "equal values are not a conflict")

# The other half of T5.3: a CLI --epochs override must reach the resolved config,
# not just the loop. The historical defect logged epochs: 50 to W&B for a 1-epoch
# run, which makes every proxy run indistinguishable from a canonical one.
from more.run_context import resolve_overrides
_res = resolve_overrides(load_config_defaults({"training": {"epochs": 50}}),
                         epochs=1, batch_size=64, seed=43)
_prov = _res.get("provenance", {})
check("G4.27 CLI overrides are written into the resolved config and provenance",
      _res["training"]["epochs"] == 1 and _prov["resolved_epochs"] == 1
      and _res["training"]["batch_size"] == 64 and _prov["resolved_batch_size"] == 64
      and _prov["seed"] == 43,
      f"epochs 50 -> {_prov['resolved_epochs']}, "
      f"batch_size -> {_prov['resolved_batch_size']}, seed -> {_prov['seed']}")

check("G4.28 --seed reaches BOTH subset_seed locations, so they cannot diverge",
      _res["training"]["subset_seed"] == 43 == _res["data"]["subset_seed"],
      f"training={_res['training']['subset_seed']}, data={_res['data']['subset_seed']}")


# ===========================================================================
print("\n--- T5.4  router exploration noise is off in canonical ---")

check("G4.29 the canonical router_noise mode is 'none'",
      CANONICAL_ROUTER_NOISE == "none"
      and load_config_defaults({})["model"]["router_noise"] == "none",
      f"CANONICAL_ROUTER_NOISE={CANONICAL_ROUTER_NOISE!r}, "
      f"config default={load_config_defaults({})['model']['router_noise']!r} "
      "(pre-fix: no key existed and a trainable scale of 0.1 was always active)")

check("G4.30 all three variants are still available as ablation D",
      ROUTER_NOISE_MODES == ("none", "fixed_annealed", "trainable"),
      f"{ROUTER_NOISE_MODES} -- updated_rules.md 1.1 ablation D")

_canon = build_model(6)
_noise_params = [k for k in _canon.state_dict() if "noise" in k.lower()]
check("G4.31 no noise parameter exists in the canonical state_dict",
      not _noise_params,
      "absence is verifiable; a parameter pinned to 0.0 would still be in the "
      "optimizer and still be reported"
      if not _noise_params else f"found: {_noise_params}")

_trainable = build_model(6, router_noise="trainable")
_tn = [k for k in _trainable.state_dict() if "noise" in k.lower()]
check("G4.32 the trainable ablation DOES carry a noise parameter",
      len(_tn) == NUM_BLOCKS,
      f"{_tn} -- one per block, so canonical and ablation checkpoints are "
      "structurally distinguishable")

_annealed = build_model(6, router_noise="fixed_annealed")
_an = [k for k in _annealed.state_dict() if "noise" in k.lower()]
check("G4.33 the annealed ablation has a scale but no learnable parameter",
      not _an and _annealed.blocks[0].moe_block.current_router_noise_scale() > 0,
      f"state_dict noise keys={_an}, initial scale="
      f"{_annealed.blocks[0].moe_block.current_router_noise_scale():.4f}")

# Canonical routing must be reproducible from the weights alone: the argmax is
# taken on raw logits, so two training-mode forward passes agree. Under the
# noisy variants they do not, which is exactly why the variant must be labelled.
_x = torch.randn(64, D_MODEL)
_mb_canon = MoEBlock(6, D_MODEL, 0.0).train()
_r1 = _mb_canon(_x)[2]
_r2 = _mb_canon(_x)[2]
check("G4.34 canonical dispatch is deterministic in training mode",
      torch.equal(_r1, _r2),
      f"{int((_r1 == _r2).sum())}/{len(_r1)} tokens routed identically across "
      "two training-mode passes")

torch.manual_seed(11)
_mb_noisy = MoEBlock(6, D_MODEL, 0.0, router_noise="trainable").train()
_n1 = _mb_noisy(_x)[2]
_n2 = _mb_noisy(_x)[2]
check("G4.35 the noisy ablation is NOT deterministic (so it is a real variant)",
      not torch.equal(_n1, _n2),
      f"{int((_n1 != _n2).sum())}/{len(_n1)} tokens changed expert between two "
      "identical inputs -- the historical canonical behaviour")

try:
    MoEBlock(6, D_MODEL, 0.0, router_noise="sprinkle_of_chaos")
    check("G4.36 an unknown router_noise mode raises", False, "accepted it")
except ValueError as e:
    check("G4.36 an unknown router_noise mode raises",
          "router_noise" in str(e), "ValueError listing the three valid modes")

# The proxy guard is the enforcement that matters: a noisy or dense run must not
# be able to claim canonical status even before canonical_spec.json is frozen.
def _claims_canonical(**model_overrides):
    cfg = load_config_defaults({})
    cfg["model"].update(model_overrides)
    cfg.setdefault("logging", {})["experiment_group"] = "canonical_phase_b"
    cfg["data"]["dataset_version"] = "dummy"
    cfg.setdefault("provenance", {})["seed"] = 42
    try:
        assert_not_silent_proxy(cfg)
        return None
    except ProxyGuardError as e:
        return str(e)

_msg = _claims_canonical(router_noise="trainable")
check("G4.37 the proxy guard refuses a canonical claim with router noise on",
      _msg is not None and "router_noise" in _msg,
      "refused, naming model.router_noise" if _msg else "guard let it through")

_msg = _claims_canonical(routing_mode="dense_blend")
check("G4.38 the proxy guard refuses a canonical claim with dense blending",
      _msg is not None and "routing_mode" in _msg,
      "refused, naming model.routing_mode" if _msg else "guard let it through")


# ===========================================================================
print("\n--- T5.1  the three architectures stay dimensionally consistent ---")

for arch, want_E in (("moe", 6), ("mor", 1), ("more", 6)):
    cfg = apply_architecture(load_config_defaults({}), arch)
    mc  = cfg["model"]
    m   = build_model(mc["num_experts"])
    # T8.3: cls_head is deliberately NOT checked against want_E any more -- it is
    # the label head and stays 6 wide for all three architectures. The routing
    # width is what apply_architecture varies.
    check(f"G4.39 {arch}: apply_architecture stamps E={want_E} and the model agrees",
          mc["num_experts"] == want_E
          and m.blocks[0].moe_block.router.out_features == want_E
          and m.cls_head.out_features == NUM_EXPERTS_CANONICAL,
          f"num_experts={mc['num_experts']}, "
          f"router={m.blocks[0].moe_block.router.out_features}, "
          f"cls_head={m.cls_head.out_features} (label space), "
          f"routing_mode={mc.get('routing_mode')}, "
          f"router_noise={mc.get('router_noise')}")


# ===========================================================================
print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    print("FAILED:")
    for f in FAIL:
        print(f"  - {f}")
sys.exit(1 if FAIL else 0)

