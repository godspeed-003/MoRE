"""opdiff_families.py - Expert-group manifest for `task = opdiff`.

The opdiff counterpart of `families.py` (arithmetic) and `lang_families.py`
(language), and deliberately the same SHAPE: the same accessor names, the same
`-1`-means-ignore convention, the same import-time guards. A module that
already imports `expert_labels`, `ALL_OP_NAMES`, `NUM_OP_TYPES` and
`NUM_EXPERTS_CANONICAL` from `families.py` can import the identical four names
from here.

THE DATASET THIS DESCRIBES (v4). `make_opdiff_dataset_v4.py` writes a row =
16 packed examples x 24 tokens = 384 uint8 token ids. Each example is

    [OP] k4 k3 k2 k1 k0  a3 a2 a1 a0  b3 b2 b1 b0  [SEP]  r7 r6 r5 r4 r3 r2 r1 r0  [EOS]
     0    1..5            6..9         10..13         14     15..22                  23

with the OP token naming WHICH function to compute, a five-token slot holding
the ITERATION COUNT k of the F operation in binary (five NULL tokens for every
other operation), A and B being 16-bit operands rendered as 4 hex digits each,
and the answer being 8 hex digits. Every row belongs to EXACTLY ONE expert
group (E1..E7), and a row never mixes groups. The group is recorded in
`{split}_group.npy` and is NEVER an input feature: it is the oracle label the
routing metrics are scored against.

TOKEN LAYOUT (from `meta.json`, measured, not assumed):

    SEP 0, EOS 1, NULL 2, hex digits 3..18, operation tokens 19..36, vocab 37.

The 18 operation tokens are the generator's `OPS` INSERTION order, which is
NOT sorted and NOT the group order:

    19 AND  20 OR   21 XOR  22 NOT  23 SHL  24 SHR  25 REV  26 MIN  27 MAX
    28 ADD  29 SUB  30 MUL  31 DIV  32 MOD  33 SORT 34 MEDIAN 35 GCD  36 F

WHY THE k SLOT EXISTS, because it is the whole point of v3. In v2 the F
operations were separate tokens F1, F2, F4, ..., F16, and the OOD split was
rows of F3, F6, F12. That could not test anything: **F3's token never appears
in training, so its embedding is never trained**, and no architecture can
know that "F3" means three iterations. The accuracy on those rows would be
near chance for every model, and the "does the router extrapolate to unseen
iteration counts?" claim would have been vacuous.

In v3 there is ONE F operation token, and k is written out in the five-token
binary slot using the two hex-digit tokens the model has already learned
(token 3 = '0', token 4 = '1'). The trained k values are 1, 2, 4, 5, 7, 8, 10,
12, 13, 15, 16; the held-out ones are 3, 6, 9, 11, 14 (INTERPOLATION -- inside
the trained range, never seen) and 17, 20 (EXTRAPOLATION -- beyond it). The
bit patterns overlap: k=3 combines bits the model has seen together in k=7
and k=15, so a held-out k is inferable from trained tokens. A held-out k
therefore tests a real capability -- reading a composition out of a slot and
spending the matching amount of computation -- instead of testing whether an
untrained embedding can be guessed.

THE OPERATION AXIS, AND WHY IT IS 35 WIDE AND NOT 18. The per-operation depth
breakdown (`recursion/avg_depth_by_op/<OP>`) is what answers "does the model
spend more computation on the operations that need more of it?". For the F
operation the interesting quantity is not the operation but the ITERATION
COUNT: the claim is that depth grows with k. So the axis carries one entry
per (operation, k) INSTANCE: the 17 non-F operations, then one entry per k
value, named `F1`…`F17`, `F20`. The axis index of a token is derived from the INPUT --
the operation token and the k slot the model is shown -- and never from a
label file, so the axis is legitimate operation identity and not an oracle.

THERE IS NO DEPTH CURRICULUM. `op_target_depth_table()` returns `[]`, and the
emptiness is load-bearing: an EMPTY table cannot supervise depth no matter
which loss weight a future config sets. The F operation's true algorithmic
depth IS k, but the model's recursion budget is 7 steps, so a target of k = 16
would be unreachable by construction -- an irreducible loss, the same trap the
arithmetic study's curriculum had. The compute-versus-k claim is therefore
MEASURED (the per-k average depth) rather than supervised, and the
depth-allocation error against a curriculum is N/A, correctly.

THREE THINGS DIFFER FROM ARITHMETIC, and they are the reason this file exists
rather than a second table in `families.py`:

1.  THERE IS A REAL OPERATION AXIS, 35 wide, and it is 35 wide because the F
    operation is a family of instances rather than one operation.
2.  THERE IS NO DEPTH CURRICULUM (above).
3.  THE SEVEN GROUPS ARE A PRIOR, NOT GROUND TRUTH. `families.OP_TO_EXPERT` on
    arithmetic is functional truth: ADD really does belong with SUB. Here E2
    bundles SHL, SHR, REV, MIN and MAX -- shift, byte-reverse, minimum and
    maximum -- which is a plausible engineering grouping and not a fact. A
    router that separates the shifts from the byte-reverse may be the better
    partition while scoring low against this one. So the agreement number is
    named `routing_agreement_with_group`, never `routing_accuracy`, and the
    permutation-invariant metrics are primary.

Deliberately free of `torch`, `numpy` and any dataset dependency: this is the
manifest. `make_opdiff_dataset_v4.py` is the authority for the dataset
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

# The generator's `OPS` insertion order, which is the TOKEN order. Do not sort
# this list: the op id is the token id minus OP_TOKEN_OFFSET, and a sorted
# list would silently relabel the axis.
#
#  0 AND   1 OR    2 XOR   3 NOT   4 SHL   5 SHR   6 REV   7 MIN   8 MAX
#  9 ADD  10 SUB  11 MUL  12 DIV  13 MOD  14 SORT 15 MEDIAN 16 GCD  17 F
ALL_OP_NAMES: list[str] = [
    "AND", "OR", "XOR", "NOT",
    "SHL", "SHR", "REV", "MIN", "MAX",
    "ADD", "SUB",
    "MUL",
    "DIV", "MOD",
    "SORT", "MEDIAN",
    "GCD",
    # The F operation: ONE token, with the iteration count in the k slot.
    "F",
]
OP_NAME_TO_IDX: dict[str, int] = {n: i for i, n in enumerate(ALL_OP_NAMES)}
NUM_OP_TYPES = len(ALL_OP_NAMES)

# The first operation token id. `meta.json:tokens.op0` is 19, and the 19 ids
# below it are SEP (0), EOS (1), NULL (2) and the 16 hex digits (3..18).
OP_TOKEN_OFFSET = 19

# The full vocabulary is the 19 fixed tokens plus the 18 operation tokens.
# `meta.json:vocab_size` is 37 and the LM head is [37, d_model], so a
# disagreement here is a disagreement about the size of the output layer.
NUM_FIXED_TOKENS = OP_TOKEN_OFFSET
VOCAB_SIZE = NUM_FIXED_TOKENS + NUM_OP_TYPES

# The F operation's name, and the number of binary tokens its k occupies.
F_OP_NAME = "F"
K_BITS = 5

# EVERY k VALUE THE DATASET CAN CONTAIN, sorted: the trained ones, the
# interpolation ones and the extrapolation ones. The axis is sized by this
# list, so a k value that is not in it has no axis entry -- and the dataset
# raises rather than clamping, because a clamped k would attribute a measured
# depth to the wrong iteration count.
#
# T-OD.10. The list is the v4 one, and it is set from the MEASURED
# learnability frontier: the dense calibration says at which k a plain
# transformer still learns the F operation at all, and the dial has to live
# inside that frontier, because a dial that overshoots the model's depth
# budget measures nothing but the budget. The v3 list (1..16 plus 17 and 20)
# is the record of the ORIGINAL design, whose F operation was not learnable
# at ANY k -- see make_opdiff_dataset_v3.py and the changelog.
# The axis is a LABEL SPACE, so it is a SUPERSET of the current dataset's k
# set. The v4 design uses 1..7; the v3 design used 1..16 plus 17 and 20, and
# the v3 dataset is kept for the record (it is the source of the "the F
# operation is not learnable at ANY k" measurement), so a run over it still
# needs an axis entry for its k values. The DATA's own k list is checked
# against the DATA's own manifest in opdiff_data.py, so a wide axis cannot
# hide a typo in a dataset.
K_VALUES: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                             15, 16, 17, 20)
K_VALUE_TO_IDX: dict[int, int] = {k: i for i, k in enumerate(K_VALUES)}

# The generator's `GROUPS`: group number (1..6) -> its operation names, and
# E7 built separately (the F operation with every trained k, plus GCD).
#
# E7's membership is the OPERATION names, not the instances: "F" covers every
# iteration count, because the k is the instance axis and the group says which
# family of operations the row belongs to. The generator's own description of
# E7 -- "F(k) k in TRAIN_K, plus GCD" -- is the same set written the other way
# round.
GROUP_OP_NAMES: dict[int, list[str]] = {
    1: ["AND", "OR", "XOR", "NOT"],
    2: ["SHL", "SHR", "REV", "MIN", "MAX"],
    3: ["ADD", "SUB"],
    4: ["MUL"],
    5: ["DIV", "MOD"],
    6: ["SORT", "MEDIAN"],
    7: ["F", "GCD"],
}

# Operation name -> 0-based GROUP index, DERIVED from the table above rather
# than typed out a second time (the two cannot drift). This is the oracle map:
# which group an operation belongs to. It is a DIAGNOSTIC and a supervision
# target -- never an input feature.
OP_TO_GROUP: dict[str, int] = {}
for _g, _ops in GROUP_OP_NAMES.items():
    for _op in _ops:
        OP_TO_GROUP[_op] = _g - 1
# `OP_TO_EXPERT` is the name `families.py` exports for the arithmetic task,
# where the expert IS the family. Here the two are the same thing too (the
# oracle expert index of a token is its row's group), and the alias exists so
# that a reader does not have to know which name this task uses.
OP_TO_EXPERT = OP_TO_GROUP

# THE DEPTH AXIS. The 17 non-F operations, then one entry per k value.
#
# This is the axis the per-operation depth breakdown is reported on, and its
# width is what the engine's depth tensors are sized by. The F entries are
# named `F<k>`, which is what makes the compute-versus-k claim a slice of the
# table rather than a separate mechanism.
_NON_F_OPS: list[str] = [op for op in ALL_OP_NAMES if op != F_OP_NAME]
OP_AXIS_NAMES: list[str] = _NON_F_OPS + [f"F{k}" for k in K_VALUES]
NUM_OP_AXIS = len(OP_AXIS_NAMES)

# The axis index of each non-F operation, and of each k.
_OP_AXIS_INDEX_OF_OP: dict[str, int] = {op: i for i, op in enumerate(_NON_F_OPS)}
_K_AXIS_OFFSET = len(_NON_F_OPS)
# Which axis entries are F instances, and which k each carries.
AXIS_K_OF_INDEX: dict[int, int] = {
    _K_AXIS_OFFSET + j: k for j, k in enumerate(K_VALUES)
}


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
    """The token id of one operation, e.g. "ADD" -> 28.

    The inverse of `op_id = token - OP_TOKEN_OFFSET`, spelled out because the
    dataset stores token ids and the code stores op ids, and a mix-up between
    them is a silent 19-wide offset rather than an error.
    """
    if op_name not in OP_NAME_TO_IDX:
        raise KeyError(
            f"unknown operation {op_name!r}. Known operations: "
            f"{sorted(OP_NAME_TO_IDX)}. An unmapped operation must raise, never "
            "fall back to another one (updated_rules.md 2: no catch-all class)."
        )
    return OP_NAME_TO_IDX[op_name] + OP_TOKEN_OFFSET


def op_id_from_token(token_id: int) -> int:
    """The dense 0..17 op index of a token id. Raises outside the op range."""
    if not (OP_TOKEN_OFFSET <= token_id < OP_TOKEN_OFFSET + NUM_OP_TYPES):
        raise ValueError(
            f"token id {token_id} is not an operation token. Operation tokens "
            f"are {OP_TOKEN_OFFSET}..{OP_TOKEN_OFFSET + NUM_OP_TYPES - 1}; "
            f"{OP_TOKEN_OFFSET - 1} is the last hex digit and {OP_TOKEN_OFFSET} "
            "is the first operation. A row whose op slot holds something else "
            "was generated against a different layout."
        )
    return int(token_id) - OP_TOKEN_OFFSET


def op_axis_index(op_name: str, k: int = 0) -> int:
    """The DEPTH-AXIS index of one (operation, k) instance.

    For every operation except F this is the operation's own index. For F it
    is the index of the k value, because the interesting measurement is the
    depth of the ITERATION COUNT, not of the operation.

    Raises for an unknown operation or an unknown k. A clamped k would
    attribute a measured depth to the wrong iteration count, which is the
    exact failure the compute-versus-k claim cannot survive.
    """
    if op_name == F_OP_NAME:
        if k not in K_VALUE_TO_IDX:
            raise ValueError(
                f"k={k} is not one of the dataset's k values {K_VALUES}. An "
                "unknown k has no axis entry; clamping it would report the "
                "depth of a different iteration count."
            )
        return _K_AXIS_OFFSET + K_VALUE_TO_IDX[k]
    if op_name not in _OP_AXIS_INDEX_OF_OP:
        raise KeyError(
            f"unknown operation {op_name!r}. Known operations: "
            f"{sorted(OP_NAME_TO_IDX)}."
        )
    return _OP_AXIS_INDEX_OF_OP[op_name]


def op_axis_k(axis_index: int) -> int | None:
    """The k an axis entry carries, or None for a non-F entry."""
    return AXIS_K_OF_INDEX.get(int(axis_index))


def op_target_depth_table() -> list[float]:
    """EMPTY, and the emptiness is the constraint made mechanical.

    `engine.py` binds this table unconditionally at setup, so it must exist and
    must return something. For opdiff the honest something is nothing.

    WHY NOT "TARGET DEPTH = k FOR THE F OPERATION", which is the obvious
    temptation because k IS the true algorithmic depth of that operation. Two
    reasons, and both are load-bearing:

    1.  The recursion budget is 7 steps, and the trained k values run to 16.
        A curriculum target of 16 is unreachable BY CONSTRUCTION, so the term
        would be an irreducible loss -- the same trap the arithmetic study's
        curriculum had, and the reason the arithmetic spec's own target depths
        are bounded by max_steps.
    2.  The point of the measurement is to find out WHETHER the model allocates
        depth in proportion to k. Supervising it toward k would manufacture the
        answer and then report it as a discovery.

    So the compute-versus-k claim is measured -- the per-k average exit depth
    -- and the depth-allocation error against a curriculum is N/A, correctly.
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

# The axis must be a bijection: every axis entry must be reachable and every
# operation and k must land somewhere.
if NUM_OP_AXIS != len(_NON_F_OPS) + len(K_VALUES):
    raise ValueError(
        f"NUM_OP_AXIS is {NUM_OP_AXIS}, expected "
        f"{len(_NON_F_OPS) + len(K_VALUES)}."
    )
if len(set(OP_AXIS_NAMES)) != NUM_OP_AXIS:
    raise ValueError(
        f"duplicate axis labels: {OP_AXIS_NAMES}. Two axis entries sharing a "
        "name would merge their depth measurements."
    )
if len(set(K_VALUE_TO_IDX.values())) != len(K_VALUES):
    raise ValueError("K_VALUES does not map to a dense 0..N-1 index range.")

# Every operation must be in exactly one group, and every group must contain
# at least one operation. An operation in no group is the E7 catch-all defect;
# a group with no operation is a structurally empty confusion-matrix row.
_mapped = set()
for _g, _ops in GROUP_OP_NAMES.items():
    for _op in _ops:
        if _op in _mapped:
            raise ValueError(
                f"operation {_op!r} appears in more than one group. A row must "
                "belong to exactly one group, or the oracle label is ambiguous."
            )
        if _op not in OP_NAME_TO_IDX and not _op.startswith("F(k)"):            raise ValueError(
                f"group {_g} lists unknown operation {_op!r}. The group map is "
                "derived from the operation list, and the two must agree."
            )
        _mapped.add(_op)
if F_OP_NAME not in _mapped:
    raise ValueError("the F operation is in no group; E7 would be empty.")

# The vocabulary arithmetic: 19 fixed tokens (SEP, EOS, NULL, 16 hex digits)
# plus 18 operation tokens is 37, which is the `meta.json:vocab_size` the
# config must declare and the width of the tied LM head.
if VOCAB_SIZE != 37:
    raise ValueError(
        f"VOCAB_SIZE is {VOCAB_SIZE}, expected 37 "
        f"({NUM_FIXED_TOKENS} fixed + {NUM_OP_TYPES} operation tokens). The tied "
        "LM head is [vocab_size, d_model], so a wrong vocabulary is a wrong "
        "output layer, not a cosmetic mismatch."
    )

# The op id must be a bijection with the token range.
if len(set(OP_NAME_TO_IDX.values())) != NUM_OP_TYPES:
    raise ValueError(
        "OP_NAME_TO_IDX does not cover 0..%d uniquely. The op id is the token "
        "id minus %d, so it must be a bijection with the token range."
        % (NUM_OP_TYPES - 1, OP_TOKEN_OFFSET)
    )
