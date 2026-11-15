"""families.py - Single operation/expert family manifest. All labels derive from here.

Extracted from the former monolithic train.py (plan.md 9: ONE training
system, split into readable modules; not parallel implementations).
"""

# ---------------------------------------------------------------------------
# 1. Dataset
# ---------------------------------------------------------------------------

# Whole-program family index. There are exactly SIX experts (updated_rules.md 2).
# Multi-operation programs have no single family: their steps route to E1-E6
# individually, so they carry index -1 = "no whole-program family", which the
# classification loss ignores. -1 is a documented ignore label, never a 7th
# class and never a measurement. "E7" is accepted only as the legacy spelling of
# MIXED emitted by pre-Phase-1 dataset files.
FAMILY_TO_IDX = {"E1": 0, "E2": 1, "E3": 2, "E4": 3, "E5": 4, "E6": 5,
                 "MIXED": -1, "E7": -1}
NO_FAMILY_IDX = -1
NUM_FAMILIES = 6

# Per-operation expert index: maps each op-code to its expert (0-5).
# Operations within the same family share an expert index. There is no
# catch-all: an unmapped operation must raise (updated_rules.md 2, plan.md 7.2).
OP_TO_EXPERT = {
    "ADD":     0, "SUB":     0,                    # E0 ≡ E1 family (ADD/SUB)
    "MULT":    1, "DIV":     1,                    # E1 ≡ E2 family (MULT/DIV)
    "MOD":     2, "POW":     2,                    # E2 ≡ E3 family (MOD/POW)
    "AND":     3, "OR":      3,
    "XOR":     3, "NOT":     3,                    # E3 ≡ E4 family (LOGIC)
    "SHIFT_L": 4, "SHIFT_R": 4,                    # E4 ≡ E5 family (SHIFT)
    "SORT":    5, "MEDIAN":  5,
    "MAX":     5, "MIN":     5,                    # E5 ≡ E6 family (SORT/STAT)
}
# The E7 catch-all expert is REMOVED (plan.md 7.2). Kept as an explicit sentinel
# so any surviving import fails loudly instead of silently routing to a 7th
# expert that no longer exists.
_EXPERT_FALLBACK = None

# Paper-figure labels: row/column names for the 7×7 routing confusion matrix.
# W&B-safe: no "/" inside a label, which would create spurious metric nesting
# (updated_rules.md 5). Six entries, one per canonical expert.
EXPERT_FAMILY_LABELS = [
    "E1_ADD_SUB",
    "E2_MULT_DIV",
    "E3_MOD_POW",
    "E4_LOGIC",
    "E5_SHIFT",
    "E6_SORT_STAT",
]
NUM_EXPERTS_CANONICAL = len(EXPERT_FAMILY_LABELS)


def expert_labels(num_experts: int) -> list[str]:
    """Exactly `num_experts` axis labels, derived from the one manifest.

    T5.1 / Gate 4 (plan.md 7.1). Every confusion matrix, heat map and per-expert
    metric key is `num_experts` wide, but `EXPERT_FAMILY_LABELS` is fixed at the
    canonical six. Using the manifest list directly to label a matrix of a
    different width is the exact failure Gate 4 exists to catch: at E=1 (the MoR
    baseline) matplotlib silently drew six tick labels on a 1x1 image, and at E=7
    the seventh row had no label at all.

    Below six, the manifest is truncated -- E=1 means "one expert, which is
    family E1" and that is a real statement about the run. Above six there is no
    canonical family, so the extra slots are named `E7`, `E8`, ... positionally
    and are honest about carrying no family semantics; that configuration exists
    only as a dimensional-consistency ablation, never as canonical
    (updated_rules.md 2: canonical is six).
    """
    if num_experts <= 0:
        raise ValueError(
            f"num_experts must be >= 1, got {num_experts}. There is no such "
            "thing as a router over zero experts; this is a config error, not a "
            "degenerate case to be handled silently."
        )
    if num_experts <= len(EXPERT_FAMILY_LABELS):
        return list(EXPERT_FAMILY_LABELS[:num_experts])
    return list(EXPERT_FAMILY_LABELS) + [
        f"E{i + 1}" for i in range(len(EXPERT_FAMILY_LABELS), num_experts)
    ]


# Per-operation labels for the recursion-depth bar chart (sorted for stable indexing).
ALL_OP_NAMES = sorted(OP_TO_EXPERT.keys())
OP_NAME_TO_IDX = {name: i for i, name in enumerate(ALL_OP_NAMES)}
NUM_OP_TYPES = len(ALL_OP_NAMES)


# ---------------------------------------------------------------------------
# Operation-complexity depth curriculum  (updated_rules.md 2.3, plan.md 5.4)
# ---------------------------------------------------------------------------
#
# READ THIS BEFORE CITING ANY DEPTH NUMBER.
#
# This table is a DESIGN CHOICE, not a measurement and not a property the model
# discovers. It states how many sequential sub-steps each operation takes under a
# naive algorithmic decomposition, and it is the reference against which
# "depth allocation error" is computed.
#
# The only defensible claim is the one updated_rules.md 2.3 mandates:
#
#   "The model learns to allocate recursion depth in accordance with the
#    predefined operation-complexity curriculum."
#
# NEVER write that the model "discovers intrinsic mathematical complexity". The
# target depth is a property of this table, which a human wrote.
#
# Rationale for the assignment (naive sequential decomposition):
#   1  single primitive machine operation
#   2  one primitive applied repeatedly, or a single pairwise comparison
#   3  a division-plus-remainder or repeated-multiplication chain
#   4  requires establishing an ordering over all arguments
#
# The curriculum is deliberately CORRELATED WITH BUT NOT IDENTICAL TO the expert
# families: E4 LOGIC and E5 SHIFT share target depth 1, and E6 SORT/STAT spans
# 2 (MAX/MIN) and 4 (SORT/MEDIAN). If target depth were a pure function of the
# expert index, a model could score perfectly on depth by routing alone and the
# depth measurement would carry no independent information.
OP_TARGET_DEPTH = {
    "ADD":     1, "SUB":     1,   # one primitive add/subtract
    "AND":     1, "OR":      1,
    "XOR":     1, "NOT":     1,   # one primitive bitwise operation
    "SHIFT_L": 1, "SHIFT_R": 1,   # one primitive shift
    "MULT":    2, "DIV":      2,   # repeated addition / repeated subtraction
    "MAX":     2, "MIN":      2,   # single pairwise comparison
    "MOD":     3, "POW":      3,   # divide-and-remainder / repeated multiply
    "MEDIAN":  4, "SORT":     4,   # requires an ordering over all arguments
}

# The curriculum must cover exactly the operation manifest -- no more, no fewer.
# A missing entry would otherwise silently receive a fallback target depth, which
# is the same class of defect as the removed E7 catch-all (plan.md 7.2).
_missing = set(OP_TO_EXPERT) - set(OP_TARGET_DEPTH)
_extra   = set(OP_TARGET_DEPTH) - set(OP_TO_EXPERT)
if _missing or _extra:
    raise ValueError(
        "OP_TARGET_DEPTH must cover exactly the operations in OP_TO_EXPERT. "
        f"Missing target depth for {sorted(_missing)}; "
        f"target depth given for unknown operations {sorted(_extra)}. "
        "Every operation needs an explicit target depth: an unmapped operation "
        "must raise, never fall back to a default (plan.md 7.2)."
    )

MIN_TARGET_DEPTH = min(OP_TARGET_DEPTH.values())
MAX_TARGET_DEPTH = max(OP_TARGET_DEPTH.values())


def op_target_depth_table():
    """
    Target depth per operation CODE, ordered by OP_NAME_TO_IDX.

    Returns a plain list so this module stays torch-free; the caller converts it
    to a tensor on the right device. Index i corresponds to
    `ALL_OP_NAMES[i]`, i.e. to the integer stored in the dataset's `step_ops`.
    """
    return [float(OP_TARGET_DEPTH[name]) for name in ALL_OP_NAMES]



