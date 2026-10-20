"""opdiff_families.py - Expert-group manifest for `task = opdiff`.

The opdiff counterpart of `families.py` (arithmetic) and `lang_families.py`
(language), and deliberately the same SHAPE: the same accessor names, the same
`-1`-means-ignore convention, the same import-time guards. A module that
already imports `expert_labels`, `ALL_OP_NAMES`, `NUM_OP_TYPES` and
`NUM_EXPERTS_CANONICAL` from `families.py` can import the identical four names
from here.

THE DATASET THIS DESCRIBES. `make_opdiff_dataset_v2.py` writes a row = 16
packed examples x 19 tokens = 304 uint8 token ids. Each example is

    [OP] a3 a2 a1 a0  b3 b2 b1 b0  [SEP]  r7 r6 r5 r4 r3 r2 r1 r0  [EOS]
     0    1..4         5..8          9       10..17                  18

with the OP token naming WHICH function to compute, A and B being 16-bit
operands rendered as 4 hex digits each, and the answer being 8 hex digits.
Every row belongs to EXACTLY ONE expert group (E1..E7), and a row never mixes
groups. The group is recorded in `{split}_group.npy` and is NEVER an input
feature: it is the oracle label the routing metrics are scored against, and
the supervision target for the labelled `oracle_routing` ablation.

TOKEN LAYOUT (from `meta.json`, measured, not assumed):

    SEP 0, EOS 1, hex digits 2..17, operation tokens 18..42, vocab 43.

The 25 operation tokens are the generator's `ALL_OPS` INSERTION order, which
is NOT sorted and NOT the group order:

    18 AND  19 OR   20 XOR  21 NOT  22 SHL  23 SHR  24 REV  25 MIN
    26 MAX  27 ADD  28 SUB  29 MUL  30 DIV  31 MOD  32 SORT 33 MEDIAN
    34 GCD  35 F1   36 F2   37 F4   38 F8   39 F16
    40 F3   41 F6   42 F12                      <- OOD, never trained on

So `op_id = token_id - 18` is a dense 0..24 axis, and THAT is the axis the
per-operation depth breakdown is reported on.

WHY SEVEN GROUPS AND NOT SIX. The arithmetic study's six families are a
functional partition of 16 operations; the opdiff partition is a DESIGNED
partition of 25 operations, given by the generator, and it is a THIRD study's
label space. Two consequences follow and both are deliberate:

1.  `num_experts` is 7 for the MoE and MoRE arms on this task. That is a
    different router width from the arithmetic and language studies, which
    both run 6 experts, so a difference between studies is not confounded
    with a difference in expert count ONLY if the comparison is made against
    the same-task MoR/MoE arms. Every opdiff number in this phase is
    therefore reported against opdiff's own MoE and MoR arms, never against
    the arithmetic table. Seven is also what keeps the confusion matrix
    SQUARE (7 oracle groups x 7 experts), which is the precondition for the
    Hungarian / AMI / purity metrics to be defined at all: the metric layer
    returns an all-None "empty" report for a non-square matrix, which would
    have made every permutation-invariant routing number N/A on this task.
    MoR stays at 1 expert, so its matrix is 7x1 and its routing metrics are
    N/A by construction -- the same contract arithmetic and language have.

2.  THE GROUP IS A PRIOR, NOT A GROUND TRUTH OF THE DATA. `families.OP_TO_EXPERT`
    on arithmetic is functional truth: ADD really does belong with SUB. Here
    E2 bundles SHL, SHR, REV, MIN and MAX -- shift, byte-reverse, minimum and
    maximum -- which is a plausible engineering grouping and not a fact. A
    router that separates the shifts from the byte-reverse may be the better
    partition while scoring low against this one. So the agreement number is
    named `routing_agreement_with_group`, never `routing_accuracy`, and the
    permutation-invariant metrics are primary. This is the same discipline
    `lang_families.py` applies to POS, and the same reason.

THREE THINGS DIFFER FROM ARITHMETIC, and they are the reason this file exists
rather than a second table in `families.py`:

1.  THERE IS A REAL OPERATION AXIS, and it is 25 wide, not 16. It sizes the
    per-operation depth breakdown (`recursion/avg_depth_by_op/<OP>`), which is
    where the "does the model spend more computation on the operations that
    need more of it" question is actually answered for this dataset.

2.  THERE IS NO DEPTH CURRICULUM. `op_target_depth_table()` returns `[]`, and
    the emptiness is load-bearing: an EMPTY table cannot supervise depth no
    matter which loss weight a future config sets, and `halting_supervision`
    is REFUSED on this task for the same reason language refuses it. The
    generator's `{split}_diff.npy` is a NOMINAL difficulty (the group number,
    or 7 + log2(k) for the F ops) and is explicitly a placeholder pending
    dense-baseline calibration -- it is NOT a depth target and is never read
    as one.

3.  THE OOD OPS BELONG TO E7 ON DISK. `make_opdiff_dataset_v2.py` builds the
    `ood` split by calling `gen_group(rng, 7, ...)` with the OOD operation
    list, so those rows carry group 7 exactly like the E7 rows they are
    meant to be compared against. The oracle mapping below therefore places
    F3, F6 and F12 in E7, and the OOD SPLIT (not a different label) is what
    separates them from the trained F1/F2/F4/F8/F16. The distinction that
    matters -- "iteration counts never seen in training" -- is a property of
    the split, and it is recorded as such.

Deliberately free of `torch`, `numpy` and any dataset dependency: this is the
manifest. `make_opdiff_dataset_v2.py` is the authority for the dataset
contract; this file is the authority for the LABEL SPACE the code uses, and
the two are cross-checked by `code/test_opdiff_task_axis.py`.
"""

# ---------------------------------------------------------------------------
# 1. The group manifest
# ---------------------------------------------------------------------------

# Paper-figure labels: row/column names for the 7x7 routing confusion matrix.
# W&B-safe -- no "/" inside a label, which would create spurious metric nesting
# (updated_rules.md 9). This is why the arithmetic labels read `E1_ADD_SUB` and
# not `E1 ADD/SUB`, and the same rule is applied here from the start.
OPDIFF_GROUP_LABELS = [
    "E1_BITWISE",
    "E2_SHIFT_MINMAX",
    "E3_ADD_SUB",
    "E4_MULT",
    "E5_DIV_MOD",
    "E6_SORT_MEDIAN",
    "E7_ITER_GCD",
]

# `EXPERT_FAMILY_LABELS` is the name `families.py` exports and `__init__.py`
# re-exports; aliased rather than renamed so the manifests stay drop-in
# interchangeable at the import site and a `task`-conditional import is the only
# thing that differs.
EXPERT_FAMILY_LABELS = OPDIFF_GROUP_LABELS

NUM_FAMILIES = len(OPDIFF_GROUP_LABELS)
NUM_EXPERTS_CANONICAL = len(OPDIFF_GROUP_LABELS)

# The documented ignore label. Same value and same meaning as
# `families.NO_FAMILY_IDX`: not an eighth class, not a measurement, and never
# reported as one.
NO_FAMILY_IDX = -1

# Short-code -> index, mirroring `families.FAMILY_TO_IDX`. `UNMAPPED` is an
# explicit key for -1, so a caller writes `FAMILY_TO_IDX["UNMAPPED"]` rather
# than a bare literal.
FAMILY_TO_IDX = {
    "E1": 0, "E2": 1, "E3": 2, "E4": 3, "E5": 4, "E6": 5, "E7": 6,
    "UNMAPPED": NO_FAMILY_IDX,
}
IDX_TO_FAMILY = {v: k for k, v in FAMILY_TO_IDX.items() if v >= 0}


# ---------------------------------------------------------------------------
# 2. The operation axis
# ---------------------------------------------------------------------------

# The generator's `ALL_OPS` insertion order, which is the TOKEN order. Do not
# sort this list: `op_id = token_id - OP_TOKEN_OFFSET` is the index the dataset
# uses, and a sorted list would silently relabel the axis.
#
#   0 AND   1 OR    2 XOR   3 NOT   4 SHL   5 SHR   6 REV   7 MIN
#   8 MAX   9 ADD  10 SUB  11 MUL  12 DIV  13 MOD  14 SORT 15 MEDIAN
#  16 GCD  17 F1   18 F2   19 F4   20 F8   21 F16  22 F3   23 F6   24 F12
ALL_OP_NAMES: list[str] = [
    "AND", "OR", "XOR", "NOT",
    "SHL", "SHR", "REV", "MIN", "MAX",
    "ADD", "SUB",
    "MUL",
    "DIV", "MOD",
    "SORT", "MEDIAN",
    "GCD", "F1", "F2", "F4", "F8", "F16",
    # OOD: present in the data, never trained on.
    "F3", "F6", "F12",
]
OP_NAME_TO_IDX: dict[str, int] = {n: i for i, n in enumerate(ALL_OP_NAMES)}
NUM_OP_TYPES = len(ALL_OP_NAMES)

# The first operation token id. `meta.json:tokens.op0` is 18, and the 18 ids
# below it are SEP (0), EOS (1) and the 16 hex digits (2..17).
OP_TOKEN_OFFSET = 18

# The full vocabulary is the 18 fixed tokens plus the 25 operation tokens.
# `meta.json:vocab_size` is 43 and the LM head is [43, d_model], so a
# disagreement here is a disagreement about the size of the output layer.
NUM_FIXED_TOKENS = OP_TOKEN_OFFSET
VOCAB_SIZE = NUM_FIXED_TOKENS + NUM_OP_TYPES

# The OOD operation names, which the generator holds out of train/val/test.
OOD_OP_NAMES: list[str] = ["F3", "F6", "F12"]

# The generator's `GROUPS`: group number (1..7) -> its operation names. This
# is the DESIGNED partition, reproduced verbatim; `OP_TO_EXPERT` below is
# derived from it rather than typed out a second time, so the two cannot drift.
GROUP_OP_NAMES: dict[int, list[str]] = {
    1: ["AND", "OR", "XOR", "NOT"],
    2: ["SHL", "SHR", "REV", "MIN", "MAX"],
    3: ["ADD", "SUB"],
    4: ["MUL"],
    5: ["DIV", "MOD"],
    6: ["SORT", "MEDIAN"],
    7: ["F1", "F2", "F4", "F8", "F16", "GCD"],
}

# Operation name -> 0-based GROUP index. Derived from GROUP_OP_NAMES, with the
# three OOD operations placed in E7 (group 7) because the generator labels the
# OOD rows with group 7 on disk -- see the module docstring, point 3.
OP_TO_EXPERT: dict[str, int] = {}
for _g, _ops in GROUP_OP_NAMES.items():
    for _op in _ops:
        OP_TO_EXPERT[_op] = _g - 1
for _op in OOD_OP_NAMES:
    OP_TO_EXPERT[_op] = 6

# `OP_TO_GROUP` is the same map under the name that says what it is on THIS
# task: the oracle GROUP, which is the supervision target and the confusion
# matrix's row axis. On arithmetic the analogous map is called OP_TO_EXPERT
# because the expert IS the family there; here they are the same thing too
# (the oracle expert index of a token is its row's group), and the alias
# exists so that a reader does not have to know which name this task uses.
OP_TO_GROUP = OP_TO_EXPERT


# ---------------------------------------------------------------------------
# 3. Accessors
# ---------------------------------------------------------------------------

def expert_labels(num_experts: int) -> list[str]:
    """Exactly `num_experts` axis labels, derived from the one manifest.

    Same contract as `families.expert_labels`, including the failure it exists
    to prevent: at E=1 (the MoR baseline) matplotlib silently drew six tick
    labels on a 1x1 image, and above six the extra rows had no label at all
    (Gate 4, plan.md 7.1). Every confusion matrix and per-expert metric key is
    `num_experts` wide while the manifest is fixed at seven, so the width must
    come from here and never from the manifest list used directly.

    Below seven the manifest is truncated -- E=1 means "one expert, which is
    group E1", and that is a real statement about the run. Above seven there
    is no canonical group, so the extra slots are named positionally and are
    honest about carrying no group semantics; that configuration exists only
    as a dimensional-consistency ablation, never as canonical.
    """
    if num_experts <= 0:
        raise ValueError(
            f"num_experts must be >= 1, got {num_experts}. There is no such "
            "thing as a router over zero experts; this is a config error, not a "
            "degenerate case to be handled silently."
        )
    if num_experts <= len(OPDIFF_GROUP_LABELS):
        return list(OPDIFF_GROUP_LABELS[:num_experts])
    return list(OPDIFF_GROUP_LABELS) + [
        f"E{i + 1}" for i in range(len(OPDIFF_GROUP_LABELS), num_experts)
    ]


def op_token_id(op_name: str) -> int:
    """The token id of one operation, e.g. "ADD" -> 27.

    The inverse of `op_id = token - OP_TOKEN_OFFSET`, spelled out because the
    dataset stores token ids and the code stores op ids, and a mix-up between
    them is a silent 18-wide offset rather than an error.
    """
    if op_name not in OP_NAME_TO_IDX:
        raise KeyError(
            f"unknown operation {op_name!r}. Known operations: "
            f"{sorted(OP_NAME_TO_IDX)}. An unmapped operation must raise, never "
            "fall back to another one (updated_rules.md 2: no catch-all class)."
        )
    return OP_NAME_TO_IDX[op_name] + OP_TOKEN_OFFSET


def op_id_from_token(token_id: int) -> int:
    """The dense 0..24 op index of a token id. Raises outside the op range."""
    if not (OP_TOKEN_OFFSET <= token_id < OP_TOKEN_OFFSET + NUM_OP_TYPES):
        raise ValueError(
            f"token id {token_id} is not an operation token. Operation tokens "
            f"are {OP_TOKEN_OFFSET}..{OP_TOKEN_OFFSET + NUM_OP_TYPES - 1}; "
            f"{OP_TOKEN_OFFSET - 1} is the last hex digit and {OP_TOKEN_OFFSET} "
            "is the first operation. A row whose op slot holds something else "
            "was generated against a different layout."
        )
    return int(token_id) - OP_TOKEN_OFFSET


def op_group(op_id: int) -> int:
    """The oracle group index (0..6) of one op index (0..24)."""
    if not (0 <= op_id < NUM_OP_TYPES):
        raise ValueError(
            f"op_id must be in 0..{NUM_OP_TYPES - 1}, got {op_id}. The op axis "
            f"is {NUM_OP_TYPES} wide; an id outside it is an indexing bug, and "
            "clamping it would silently attribute a depth measurement to the "
            "wrong operation."
        )
    return OP_TO_EXPERT[ALL_OP_NAMES[op_id]]


def op_target_depth_table() -> list[float]:
    """EMPTY, and the emptiness is the constraint made mechanical.

    `engine.py` binds this table unconditionally at setup, so it must exist and
    must return something. For opdiff the honest something is nothing: the
    generator's `{split}_diff.npy` is a NOMINAL difficulty placeholder pending
    dense-baseline calibration, not a curriculum, and a curriculum invented
    here would be a depth target fitted to the model that is supposed to be
    measured against it.

    An EMPTY table cannot supervise depth no matter which loss weight a
    future config sets, which is strictly stronger than a zero-filled table of
    length 25 (which would silently supervise every token toward depth 0).
    Downstream, `epoch_depth_err_n` stays 0, so `depth/allocation_error_abs`
    and `depth/allocation_error_rel` are absent from the log dict and reported
    as `N/A` rather than as a suspiciously perfect 0.0.

    NOTE: opdiff DOES have a valid `step_ops` axis (0..24), unlike language.
    That is why `metrics.halting_supervision_loss` and
    `metrics.depth_allocation_error` guard on an EMPTY TABLE as well as on an
    empty valid mask: with 25 valid op ids and a 0-length table, indexing
    raises IndexError, and with the arithmetic table (16 entries) it would
    silently measure F1..F12 against the wrong operations.
    """
    return []


# ---------------------------------------------------------------------------
# 4. Import-time consistency guards
# ---------------------------------------------------------------------------
#
# `families.py` raises at import when `OP_TARGET_DEPTH` and `OP_TO_EXPERT`
# disagree, because a silently missing entry would receive a fallback and that
# is the same defect class as the removed E7 catch-all. The equivalent
# invariants here are checked the same way, at import, so a bad edit fails on
# the first import rather than after a dataset build.

_bad_labels = [lbl for lbl in OPDIFF_GROUP_LABELS if "/" in lbl]
if _bad_labels:
    raise ValueError(
        f"Group labels must not contain '/': {_bad_labels}. A '/' inside a W&B "
        "metric label creates spurious nesting (updated_rules.md 9), which is "
        "why the arithmetic labels read 'E1_ADD_SUB' and not 'E1 ADD/SUB'."
    )

if NUM_FAMILIES != 7:
    raise ValueError(
        f"NUM_FAMILIES must be 7, got {NUM_FAMILIES}. Seven is what the opdiff "
        "generator partitions its rows into, and it is what keeps the routing "
        "confusion matrix SQUARE for the 7-expert arms -- a non-square matrix "
        "makes the Hungarian / AMI / purity report all-N/A, which would have "
        "removed every permutation-invariant routing number from this task."
    )

if len(set(OPDIFF_GROUP_LABELS)) != NUM_FAMILIES:
    raise ValueError(
        f"duplicate group labels: {OPDIFF_GROUP_LABELS}. Each group needs its "
        "own name; two groups sharing a label would merge their rows in every "
        "per-family metric key."
    )

# Every operation must belong to exactly one group, and every group must
# contain at least one operation. An operation in no group is the E7
# catch-all defect; a group with no operation is a structurally empty
# confusion-matrix row, which reads as a routing failure rather than as a
# manifest defect.
_missing = [op for op in ALL_OP_NAMES if op not in OP_TO_EXPERT]
if _missing:
    raise ValueError(
        f"operations with no group: {_missing}. Every operation must be "
        "assigned to exactly one group -- an unmapped operation must raise, "
        "never fall back (updated_rules.md 2)."
    )
_extra = [op for op in OP_TO_EXPERT if op not in OP_NAME_TO_IDX]
if _extra:
    raise ValueError(
        f"group map names operations that do not exist: {_extra}. The map is "
        "derived from ALL_OP_NAMES, so this can only mean the two lists were "
        "edited apart."
    )

_unreachable = sorted(
    set(range(NUM_FAMILIES)) - {g for g in OP_TO_EXPERT.values()})
if _unreachable:
    raise ValueError(
        "No operation maps to groups "
        f"{[OPDIFF_GROUP_LABELS[i] for i in _unreachable]}. A group nothing can "
        "be assigned to gives a structurally empty confusion-matrix row."
    )

# The vocabulary arithmetic: 18 fixed tokens (SEP, EOS, 16 hex digits) plus
# 25 operation tokens is 43, which is the `meta.json:vocab_size` the config
# must declare and the width of the tied LM head.
if VOCAB_SIZE != 43:
    raise ValueError(
        f"VOCAB_SIZE is {VOCAB_SIZE}, expected 43 "
        f"({NUM_FIXED_TOKENS} fixed + {NUM_OP_TYPES} operations). The tied LM "
        "head is [vocab_size, d_model], so a wrong vocabulary is a wrong output "
        "layer, not a cosmetic mismatch."
    )

# The operation axis must be dense and complete: every id in 0..24 must be a
# distinct operation, or `op_id = token - 18` would alias two operations onto
# one depth bucket.
if len(set(OP_NAME_TO_IDX.values())) != NUM_OP_TYPES:
    raise ValueError(
        "OP_NAME_TO_IDX does not cover 0..%d uniquely. The op id is the token "
        "id minus %d, so it must be a bijection with the token range."
        % (NUM_OP_TYPES - 1, OP_TOKEN_OFFSET)
    )

# The generator holds F3, F6 and F12 out of training. If one of them ever
# appears in GROUP_OP_NAMES, the OOD split stops being an OOD split and this
# guard is the place that says so.
_leaked = [op for op in OOD_OP_NAMES if op in GROUP_OP_NAMES.get(7, [])]
if _leaked:
    raise ValueError(
        f"OOD operations appear in the trained group partition: {_leaked}. The "
        "generator holds F3/F6/F12 out of train/val/test; listing them as "
        "trained operations would make the OOD split meaningless."
    )
