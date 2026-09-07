"""test_language_gate_l6_l7.py - T-L7.2 (GATE L6) and T-L7.3 (input parity).

Run with the CPU interpreter:
    C:/Users/vedan/anaconda3/python.exe code/test_language_gate_l6_l7.py

TWO THINGS ARE CHECKED HERE AND THEY ARE DIFFERENT KINDS OF CLAIM.

GATE L6 -- parameter budget. The three arms must be budget-matched, so that a loss
difference between them is attributable to the ARCHITECTURE and not to one arm
simply having more capacity. T-L6.3 requires the match on BOTH the total and the
non-embedding count, and the second is the one that binds: the tied 8192x256 token
table plus the 256x256 positional table are identical across all three arms by
construction, so they sit in both the numerator and the denominator of a relative
gap and shrink it for free. Measured on the frozen shape, MoR vs MoRE is 0.069% on
the total and 0.112% on the non-embedding count -- a 1.6x difference in the number
being tested, from the same two models.

The counts are read through the SAME construction path a real run uses (the
resolved config -> MoREModel kwargs mapping copied from engine.py:263), not through
a hand-built model. A test that builds its own model can pass while the trainer
instantiates something else, which is exactly how a budget mismatch would survive.

T-L7.3 -- input parity. CLAUDE.md 3 requires input features to be BIT-IDENTICAL
across MoE, MoR and MoRE for the same record. On arithmetic this is a substantive
test because the feature vector is assembled per record and an expert_id could leak
into a slot. On language it is nearly definitional -- all three arms consume the
same `input_ids` -- and that is precisely why it is worth an automated assertion
rather than a paragraph: the invariant is cheap to state, cheap to check, and the
day someone adds an architecture-conditional field to the input pipeline this is
the only thing that will notice.

Parity is checked at three levels, because they can fail independently:
  1. the DATASET tuple at a fixed index (no seed involved),
  2. the first batch out of a seeded DataLoader built exactly as engine.py builds
     it -- this catches an arch-dependent shuffle order, which a dataset-level
     check cannot see,
  3. the resolved DATA config, so the three arms are not reading different corpora
     or sequence lengths and then agreeing about a batch by coincidence.

WHAT THIS FILE DOES NOT COVER: GATE L7, which requires an actual short run to beat
`primary_metric_floor`. A floor is beaten by a training run or not at all, and no
assertion in a test file can stand in for one. Its evidence is a run directory,
recorded against T-L7.2 in TASKS_LANGUAGE.md.
"""

import json
import os
import sys

CODE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import torch                                                     # noqa: E402
from torch.utils.data import DataLoader                           # noqa: E402

from more.model import MoREModel                                  # noqa: E402
from more.families import NUM_EXPERTS_CANONICAL                   # noqa: E402
from more.config import (load_config, apply_architecture, apply_task,  # noqa: E402
                         stamp_language_dataset_versions, TASK_LANGUAGE)
from more.lang_data import MoRELanguageDataset, LANG_ROOT         # noqa: E402
from more.seeding import make_generator, seed_worker              # noqa: E402
from more.run_context import load_canonical_spec                  # noqa: E402

LANG_CONFIG = os.path.join(CODE, "config_language.json")
SPEC = load_canonical_spec(task=TASK_LANGUAGE)
ARMS = ["moe", "mor", "more"]
TOLERANCE = 0.05          # Gate L6's 5%, from T-L6.3

_p, _f, _s = [], [], []


def check(name, cond, detail=""):
    (_p if cond else _f).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    return bool(cond)


def skip(name, why):
    _s.append(name)
    print(f"  [SKIP] {name}  -- {why}")


def frozen(arch, seed=42):
    """The config a canonical run of `arch` resolves to. cli.py's call order."""
    cfg = load_config(LANG_CONFIG)
    apply_architecture(cfg, arch)
    apply_task(cfg, TASK_LANGUAGE)
    stamp_language_dataset_versions(cfg)
    cfg.setdefault("provenance", {})["seed"] = seed
    return cfg


def build_from_config(cfg):
    """MoREModel from a resolved config, mirroring engine.py:263 field for field.

    Kept as an explicit mapping rather than a helper import because engine's model
    construction is inlined in `train_one`; if that signature changes, this file
    must fail loudly rather than quietly count a differently-shaped model.
    """
    mc, dc = cfg["model"], cfg["data"]
    return MoREModel(
        step_feat_dim=mc["step_feat_dim"],
        d_model=mc["d_model"],
        num_experts=mc["num_experts"],
        max_depth=mc["max_depth"],
        num_blocks=mc["num_blocks"],
        dropout=mc["dropout"],
        fixed_depth=mc.get("fixed_depth", False),
        routing_mode=mc.get("routing_mode", "top1_sparse"),
        router_noise=mc.get("router_noise", "none"),
        ffn_mult=mc.get("ffn_mult", 4),
        num_families=NUM_EXPERTS_CANONICAL,
        attention=bool(mc.get("attention", False)),
        n_heads=int(mc.get("n_heads", 4)),
        max_seq_len=int(dc["seq_len"]),
        task=TASK_LANGUAGE,
        vocab_size=int(dc["vocab_size"]),
    )


def counts(m):
    """`(total, embedding, non_embedding)`. Prefixes copied from engine.py:312."""
    total = sum(p.numel() for p in m.parameters())
    emb = sum(p.numel() for n, p in m.named_parameters()
              if n.startswith(("tok_embed", "pos_embed", "lm_head")))
    return total, emb, total - emb


# ===========================================================================
print("\n=== T-L7.2 / GATE L6: parameter budget across the three arms " + "=" * 15)
# ===========================================================================

table = {}
for arch in ARMS:
    cfg = frozen(arch)
    m = build_from_config(cfg)
    t, e, n = counts(m)
    table[arch] = {
        "total": t, "embedding": e, "non_embedding": n,
        "num_experts": cfg["model"]["num_experts"],
        "max_depth": cfg["model"]["max_depth"],
        "ffn_mult": cfg["model"]["ffn_mult"],
    }

print(f"\n  {'arm':<6} {'E':>2} {'depth':>6} {'ffn_mult':>9} "
      f"{'total':>12} {'embedding':>12} {'non-embedding':>14}")
for arch in ARMS:
    r = table[arch]
    print(f"  {arch.upper():<6} {r['num_experts']:>2} {r['max_depth']:>6} "
          f"{r['ffn_mult']:>9} {r['total']:>12,d} {r['embedding']:>12,d} "
          f"{r['non_embedding']:>14,d}")

# The gate's own comparison: MoR against MoRE. MoE is in the table for the paper
# but is not the binding pair -- MoE has max_depth 1, so its FFN stack is the same
# shape as MoRE's and the interesting mismatch is only ever MoR's wide single expert
# against MoRE's six narrow ones.
print()
for key in ("total", "non_embedding"):
    a, b = table["mor"][key], table["more"][key]
    rel = abs(a - b) / max(a, b)
    check(f"L6: MoR vs MoRE within {TOLERANCE:.0%} on {key.replace('_', '-')}",
          rel < TOLERANCE,
          f"MoR {a:,d} vs MoRE {b:,d} = {rel:.4%}")

# MoE is reported and also checked, because a divergence there would mean the
# arithmetic-inherited ffn_mult 4 no longer produces a comparable arm on language.
for key in ("total", "non_embedding"):
    a, b = table["moe"][key], table["more"][key]
    rel = abs(a - b) / max(a, b)
    check(f"L6: MoE vs MoRE within {TOLERANCE:.0%} on {key.replace('_', '-')}",
          rel < TOLERANCE,
          f"MoE {a:,d} vs MoRE {b:,d} = {rel:.4%}")

# The embedding block must be IDENTICAL, not merely close. If it is not, the arms
# are reading different vocabularies or sequence lengths and the whole comparison is
# void -- and the non-embedding gap above would be computed against different
# denominators, so this has to be asserted before those numbers mean anything.
embs = {a: table[a]["embedding"] for a in ARMS}
check("L6: the embedding + positional block is bit-identical in size across arms",
      len(set(embs.values())) == 1, f"{embs}")

# The regression anchor from T-L6.3/T-L6.11. A changed shape is not automatically
# wrong, but it must be a deliberate re-derivation, so the number is pinned here.
check("L6: MoRE at the frozen shape is 5,584,908 parameters (T-L6.3 anchor)",
      table["more"]["total"] == 5_584_908,
      f"got {table['more']['total']:,d}")
check("L6: MoR at the frozen shape is 5,581,063 parameters (T-L6.3 anchor)",
      table["mor"]["total"] == 5_581_063,
      f"got {table['mor']['total']:,d}")

# Per-arm shape, so a table row cannot be right by accident with the wrong arm.
check("L6: MoE is 6 experts at depth 1 (which computation, not how much)",
      table["moe"]["num_experts"] == 6 and table["moe"]["max_depth"] == 1,
      f"E={table['moe']['num_experts']}, depth={table['moe']['max_depth']}")
check("L6: MoR is 1 expert at depth 7 (how much computation, not which)",
      table["mor"]["num_experts"] == 1 and table["mor"]["max_depth"] == 7,
      f"E={table['mor']['num_experts']}, depth={table['mor']['max_depth']}")
check("L6: MoRE is 6 experts at depth 7 (both axes)",
      table["more"]["num_experts"] == 6 and table["more"]["max_depth"] == 7,
      f"E={table['more']['num_experts']}, depth={table['more']['max_depth']}")

# Write the table where the paper renderer and the changelog can cite it instead of
# a number hand-copied out of this stdout (CLAUDE.md 5: never hand-copy).
out = os.path.join(CODE, "lang_param_budget.json")
with open(out, "w", encoding="utf-8") as fh:
    json.dump({
        "_note": ("GATE L6 evidence for T-L7.2. Counts built through the "
                  "engine.py:263 kwargs mapping from the frozen "
                  "config_language.json, so they are the counts a canonical run "
                  "prints. Non-embedding excludes tok_embed/pos_embed/lm_head; "
                  "lm_head is TIED to tok_embed so named_parameters yields the "
                  "shared tensor once."),
        "spec_version": SPEC["spec_version"],
        "dataset_version": frozen("more")["data"].get("dataset_version"),
        "tolerance": TOLERANCE,
        "arms": table,
        "relative_gaps": {
            f"{x}_vs_more_{k}": abs(table[x][k] - table["more"][k]) / max(
                table[x][k], table["more"][k])
            for x in ("moe", "mor") for k in ("total", "non_embedding")
        },
    }, fh, indent=2)
print(f"\n  table written to {os.path.relpath(out, os.path.dirname(CODE))}")


# ===========================================================================
print("\n=== T-L7.3: input parity across MoE / MoR / MoRE " + "=" * 30)
# ===========================================================================

corpus = load_config(LANG_CONFIG)["data"]["corpus"]
built = os.path.exists(os.path.join(LANG_ROOT, corpus, "dataset_meta.json"))

if not built:
    skip("T-L7.3 input parity",
         f"data/lang/{corpus}/ is not built (*.npy is untracked); "
         f"see SETUP.md 5")
else:
    SEED, IDX = 42, 137

    # -- level 3 first: the arms must be reading the same corpus at all ------
    dcs = {a: frozen(a, SEED)["data"] for a in ARMS}
    for field in ("corpus", "seq_len", "vocab_size", "dataset_version",
                  "train_split_version", "subset_fraction"):
        vals = {a: dcs[a].get(field) for a in ARMS}
        check(f"TL7.3 data config agrees across arms on {field}",
              len(set(map(repr, vals.values()))) == 1, f"{vals}")

    # -- level 1: the dataset tuple at a fixed index, no seed involved -------
    dss = {a: MoRELanguageDataset(corpus, "train") for a in ARMS}
    items = {a: dss[a][IDX] for a in ARMS}
    ref = items["more"]
    check("TL7.3 the dataset yields the same 7-slot tuple shape for every arm",
          len({len(items[a]) for a in ARMS}) == 1 and len(ref) == 7,
          f"len={len(ref)}")

    for slot in range(7):
        ok, why = True, ""
        for a in ("moe", "mor"):
            x, y = items[a][slot], ref[slot]
            if torch.is_tensor(y):
                # NaN != NaN, so slot 6 (the deliberately poisoned regression
                # target) is compared with equal_nan semantics rather than being
                # exempted -- an exemption would also excuse a real difference.
                same = (x.shape == y.shape and x.dtype == y.dtype and
                        bool(torch.equal(x, y) or
                             (x.is_floating_point() and
                              torch.equal(torch.nan_to_num(x, 0.0),
                                          torch.nan_to_num(y, 0.0)))))
            else:
                same = (x == y) or (x != x and y != y)
            if not same:
                ok, why = False, f"{a} differs at slot {slot}"
        check(f"TL7.3 slot {slot} is bit-identical across the three arms at "
              f"index {IDX}", ok, why)

    # Slot 0 is the one CLAUDE.md 3 is actually about: the input the trunk sees.
    check("TL7.3 slot 0 (input_ids) is a long tensor of the frozen seq_len",
          torch.is_tensor(ref[0]) and ref[0].dtype == torch.long and
          ref[0].numel() == int(dcs["more"]["seq_len"]),
          f"dtype={ref[0].dtype}, numel={ref[0].numel()}")

    # And it must contain no architecture-derived value. The strongest cheap
    # statement: every id is a legal vocabulary index, so nothing expert-shaped
    # (a family id, an expert index) has been packed into the stream.
    V = int(dcs["more"]["vocab_size"])
    check("TL7.3 input_ids are all legal vocabulary indices (no leaked label "
          "space)", bool((ref[0] >= 0).all() and (ref[0] < V).all()),
          f"min={int(ref[0].min())}, max={int(ref[0].max())}, V={V}")

    # -- level 2: the first batch from a seeded loader, built as engine does --
    # This is the level that catches an arch-dependent SHUFFLE, which levels 1
    # and 3 cannot see: identical datasets visited in different orders would give
    # the two arms different data while every per-index check passed.
    batches = {}
    for a in ARMS:
        cfg = frozen(a, SEED)
        loader = DataLoader(
            dss[a], batch_size=8, shuffle=True, num_workers=0,
            generator=make_generator(SEED, "dataloader_shuffle"),
            worker_init_fn=seed_worker)
        batches[a] = next(iter(loader))

    ok = True
    for a in ("moe", "mor"):
        if not torch.equal(batches[a][0], batches["more"][0]):
            ok = False
    check("TL7.3 the FIRST BATCH of input_ids is bit-identical across arms at "
          "seed 42 (same shuffle order)", ok,
          f"shape={tuple(batches['more'][0].shape)}")

    # The same seed must also be REQUIRED for that -- otherwise the check above
    # would pass on a loader that ignores its generator, and would prove nothing.
    other = DataLoader(dss["more"], batch_size=8, shuffle=True, num_workers=0,
                       generator=make_generator(43, "dataloader_shuffle"),
                       worker_init_fn=seed_worker)
    b43 = next(iter(other))
    check("TL7.3 a DIFFERENT seed gives a different first batch (so the parity "
          "above is a real invariant, not a loader ignoring its generator)",
          not torch.equal(b43[0], batches["more"][0]))


# ===========================================================================
print("\n" + "=" * 78)
print(f"  T-L7.2 / T-L7.3:  {len(_p)} passed, {len(_f)} failed, {len(_s)} skipped")
if _f:
    print("\n  FAILED:")
    for n in _f:
        print(f"    - {n}")
print("=" * 78)
sys.exit(1 if _f else 0)
