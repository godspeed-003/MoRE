"""lang_families.py - Single POS/expert family manifest for `task = language`.

The language counterpart of `families.py`, and deliberately the same SHAPE: the
same accessor names, the same six-wide label space, the same `-1`-means-ignore
convention. A module that already imports `expert_labels`, `ALL_OP_NAMES`,
`NUM_OP_TYPES` and `NUM_EXPERTS_CANONICAL` from `families.py` can import the
identical four names from here and needs no new plumbing (`plan_language.md`
§4.1, T-L3.0).

Six families, not five and not eight, so `num_experts = 6` stays canonical and
every `E > 1` metric is comparable with the arithmetic result. A language study
with a different expert count would confound every difference between the two
studies with a difference in router width.

    L1 FUNCTION      determiners, prepositions, conjunctions, pronouns,
                     particles, auxiliary/copular verbs
    L2 NOUN          common and proper nouns
    L3 VERB          main verbs, participles, gerunds
    L4 MODIFIER      adjectives, adverbs
    L5 PUNCT_SYM     punctuation and symbols
    L6 NUM_SUBWORD   numerals, and subword continuation pieces with no
                     standalone POS

THREE THINGS DIFFER FROM ARITHMETIC. Read them before citing a language routing
number.

1.  THIS PARTITION IS A PRIOR, NOT A GROUND TRUTH. `families.OP_TO_EXPERT` is
    functional truth: `ADD` really does belong with `SUB`. A POS partition is a
    linguistic hypothesis about what a useful expert split would be. Nothing says
    the optimal partition for next-token prediction is noun-versus-verb; a router
    that separated "begins a rare multi-piece name" from "continues one" could be
    the better partition while scoring near chance against POS. So the agreement
    number is named `routing_agreement_with_pos`, never `routing_accuracy`, and
    the permutation-invariant metrics are primary (`plan_language.md` §4.4,
    T-L3.2).

2.  UNMAPPED TYPES ARE IGNORED, NOT RAISED. `updated_rules.md` §2 requires an
    unmapped *operation* to raise, because the arithmetic op set is closed at 16.
    A byte-level BPE vocabulary has no such closure -- it legitimately contains
    fragments with no part of speech. Raising would make the build impossible; a
    fallback family would re-create the E7 catch-all the rules removed. So such
    types get `NO_FAMILY_IDX = -1`, the ignore label that `ignore_index=-1` and
    the `(family >= 0)` masks in `engine.py` and `model.py` already honour. What
    replaces `raise` is a published BUDGET: Gate L2 fails the build if the
    unmapped share exceeds 2% of train tokens (`plan_language.md` §4.3). An
    ignore label whose share is unmeasured is the catch-all in another costume;
    one whose share is bounded and published is a measurement.

3.  THERE IS NO OPERATION AXIS. See `ALL_OP_NAMES` below.

Deliberately free of `torch`, `nltk` and any dataset dependency: this is the
manifest. Building `token_family[V]` from it is T-L3.1.

That is a claim about THIS FILE's own imports, and it is not the same as
"importing it is cheap": `import more.lang_families` runs `more/__init__.py`
first, which imports `engine`/`model` and therefore torch. A caller that wants the
manifest without the framework must load it by path. TL3.0w in
`code/test_language_families.py` checks the file-level claim for that reason.
"""

# ---------------------------------------------------------------------------
# 1. The family manifest
# ---------------------------------------------------------------------------

# Paper-figure labels: row/column names for the 6x6 routing confusion matrix.
# W&B-safe -- no "/" inside a label, which would create spurious metric nesting
# (updated_rules.md §9). This is why the arithmetic labels read `E1_ADD_SUB` and
# not `E1 ADD/SUB`, and the same rule is applied here from the start.
LANG_FAMILY_LABELS = [
    "L1_FUNCTION",
    "L2_NOUN",
    "L3_VERB",
    "L4_MODIFIER",
    "L5_PUNCT_SYM",
    "L6_NUM_SUBWORD",
]

# `EXPERT_FAMILY_LABELS` is the name `families.py` exports and `__init__.py`
# re-exports; aliased rather than renamed so the two manifests are drop-in
# interchangeable at the import site and a `task`-conditional import is the only
# thing that differs.
EXPERT_FAMILY_LABELS = LANG_FAMILY_LABELS

NUM_FAMILIES = len(LANG_FAMILY_LABELS)
NUM_EXPERTS_CANONICAL = len(LANG_FAMILY_LABELS)

# The documented ignore label. Same value and same meaning as
# `families.NO_FAMILY_IDX`: not a seventh class, not a measurement, and never
# reported as one. See §2 of the module docstring for why language *uses* it
# where arithmetic raises.
NO_FAMILY_IDX = -1

# Short-code -> index, mirroring `families.FAMILY_TO_IDX`. `UNMAPPED` is the
# language spelling of arithmetic's `MIXED`: an explicit key for -1, so a caller
# writes `FAMILY_TO_IDX["UNMAPPED"]` rather than a bare literal.
FAMILY_TO_IDX = {
    "L1": 0, "L2": 1, "L3": 2, "L4": 3, "L5": 4, "L6": 5,
    "UNMAPPED": NO_FAMILY_IDX,
}
IDX_TO_FAMILY = {v: k for k, v in FAMILY_TO_IDX.items() if v >= 0}


# ---------------------------------------------------------------------------
# 1a. Names that exist in `families.py` and must NOT exist here
# ---------------------------------------------------------------------------
#
# `more/__init__.py` re-exports eight names from `families.py`. Seven of them have
# a language meaning and are defined above. `OP_TO_EXPERT` does not: it maps the 16
# arithmetic op codes to experts, and language has no op codes.
#
# The tempting fix is `OP_TO_EXPERT = {}`, and it is wrong. An empty dict makes
# `OP_TO_EXPERT[op]` raise a bare `KeyError` at whatever line happens to index it,
# which reads as a missing operation rather than as a caller that reached for the
# arithmetic axis while running language. Raising here instead puts the diagnosis
# at the point of the mistake. PEP 562 module `__getattr__` is what makes that
# possible without defining the name.
#
# NOTE the leak this does NOT close: `more/__init__.py` imports from `.families`
# unconditionally, so `from more import OP_TO_EXPERT` still yields the arithmetic
# mapping no matter what this module does. Only `lang_families.OP_TO_EXPERT` is
# guarded. Closing the other path means making the package export
# task-conditional, which is Phase L-5's problem, not this manifest's.
_ARITHMETIC_ONLY = {
    "OP_TO_EXPERT": (
        "OP_TO_EXPERT is arithmetic-only: it maps the 16 op codes to experts, and "
        "`task = language` has no operation axis (see ALL_OP_NAMES). If you want "
        "the token -> family mapping, that is `token_family[V]` from T-L3.1, "
        "indexed by vocabulary id, not by op name."
    ),
    "OP_TARGET_DEPTH": (
        "OP_TARGET_DEPTH is arithmetic-only: plan_language.md §5 forbids canonical "
        "language runs from inventing a depth curriculum. `op_target_depth_table()` "
        "exists and returns [] so engine.py's unconditional call still works; there "
        "is no per-operation target depth to read."
    ),
}


def __getattr__(name: str):
    """PEP 562 hook: refuse the arithmetic-only names with the reason."""
    if name in _ARITHMETIC_ONLY:
        raise AttributeError(f"{__name__}.{name}: {_ARITHMETIC_ONLY[name]}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")



def expert_labels(num_experts: int) -> list[str]:
    """Exactly `num_experts` axis labels, derived from the one manifest.

    Same contract as `families.expert_labels`, including the failure it exists to
    prevent: at E=1 (the MoR baseline) matplotlib silently drew six tick labels on
    a 1x1 image, and above six the extra rows had no label at all (Gate 4,
    plan.md §7.1). Every confusion matrix and per-expert metric key is
    `num_experts` wide while the manifest is fixed at six, so the width must come
    from here and never from the manifest list used directly.

    Below six the manifest is truncated -- E=1 means "one expert, which is family
    L1", and that is a real statement about the run. Above six there is no
    canonical family, so the extra slots are named positionally and are honest
    about carrying no family semantics; that configuration exists only as a
    dimensional-consistency ablation, never as canonical.
    """
    if num_experts <= 0:
        raise ValueError(
            f"num_experts must be >= 1, got {num_experts}. There is no such "
            "thing as a router over zero experts; this is a config error, not a "
            "degenerate case to be handled silently."
        )
    if num_experts <= len(LANG_FAMILY_LABELS):
        return list(LANG_FAMILY_LABELS[:num_experts])
    return list(LANG_FAMILY_LABELS) + [
        f"L{i + 1}" for i in range(len(LANG_FAMILY_LABELS), num_experts)
    ]


# ---------------------------------------------------------------------------
# 2. There is no operation axis
# ---------------------------------------------------------------------------
#
# Arithmetic has 16 op codes, and they do three jobs: they size `model.op_embed`,
# they index the per-operation depth bar chart, and they index
# `OP_TARGET_DEPTH`. Language has none of the three.
#
#   - No op embedding: the language input adapter is a token embedding over V.
#     WHICH computation to perform is not side information handed to the model;
#     it is the whole prediction problem.
#   - No per-operation depth chart: the depth breakdown for language is by
#     FAMILY (and, for ablation F only, by frequency decile), never by "operation".
#   - No depth curriculum at all: `plan_language.md` §5 is explicit that canonical
#     language runs invent no depth target, because unlike arithmetic there is no
#     defensible a-priori number of sequential steps a token "should" take.
#
# So these are EMPTY, not absent, and the emptiness is load-bearing:
# `engine.py:803` iterates `enumerate(ALL_OP_NAMES)` to build `op_avg_depth`, and
# an empty list yields an empty dict, which the metric layer then reports as
# absent rather than as 0.0 -- exactly the required behaviour under CLAUDE.md §4
# ("no sentinel value may be reported as a measurement"). A one-element dummy op
# would instead have produced a real-looking per-op depth number for an operation
# that does not exist.
ALL_OP_NAMES: list[str] = []
OP_NAME_TO_IDX: dict[str, int] = {}
NUM_OP_TYPES = 0


def op_target_depth_table() -> list[float]:
    """Empty, and the emptiness is the constraint made mechanical.

    `engine.py:265` calls this unconditionally at setup, so it must exist and must
    return something. For language the honest something is nothing:
    `plan_language.md` §5 forbids canonical language runs from inventing a depth
    curriculum, and an EMPTY table cannot supervise depth no matter which loss
    weight a future config sets. That is stronger than a zero-filled table of
    length V, which would silently supervise every token toward depth 0.

    Downstream, `epoch_depth_err_n` stays 0, so `depth/allocation_error_abs` and
    `depth/allocation_error_rel` are absent from the log dict and reported as
    `N/A` rather than as a suspiciously perfect 0.0.

    NOTE FOR PHASE L-5: `depth_allocation_error(...)` is handed this table
    together with `step_ops`. Language has no `step_ops`, so the language path must
    mask on `step_ops >= 0` (or skip the call) BEFORE indexing -- an empty table
    indexed by anything raises. That raise is preferable to a silent fallback and
    is the reason this returns `[]` rather than a padded list.

    T-L2.6's `token_target_depth[V]` is a DIFFERENT table: per vocabulary id, from
    frequency deciles, built so ablation F is runnable and documented as unused by
    the canonical configuration. It does not live here.
    """
    return []


# ---------------------------------------------------------------------------
# 3. Penn Treebank tag -> family
# ---------------------------------------------------------------------------
#
# nltk's averaged-perceptron tagger emits Penn Treebank tags, so the mapping from
# a tagger decision to a family is a lookup on that tagset. Every tag the tagger
# can emit MUST appear either here or in `PENN_UNMAPPED` -- the guard at the foot
# of this module raises if a tag is in neither, so a tagset the mapping does not
# cover fails at import rather than at build time.
PENN_TO_FAMILY = {
    # -- L1 FUNCTION: closed-class grammatical items -------------------------
    "CC":   0,   # coordinating conjunction: and, but, or
    "DT":   0,   # determiner: the, a, some
    "PDT":  0,   # predeterminer: all, both, half
    "EX":   0,   # existential there
    "IN":   0,   # preposition / subordinating conjunction: in, of, that
    "TO":   0,   # infinitival "to" and the preposition
    "MD":   0,   # modal: can, could, will, would, must
    "POS":  0,   # possessive ending: 's
    "PRP":  0,   # personal pronoun: he, they, it
    "PRP$": 0,   # possessive pronoun: his, their
    "RP":   0,   # particle: up, off, out (phrasal-verb particle)
    "WDT":  0,   # wh-determiner: which, that
    "WP":   0,   # wh-pronoun: who, what
    "WP$":  0,   # possessive wh-pronoun: whose
    # WRB (how, where, when, why) is an ADVERB by Penn's tagset but a closed-class
    # grammatical operator by function, and §4.1 assigns pronouns/determiners to
    # L1. Placing wh-adverbs with the other wh-words keeps the L1/L4 boundary at
    # "closed grammatical class vs open modifier class", which is the boundary the
    # partition is meant to express. Arguable; recorded as a choice.
    "WRB":  0,
    # UH (interjection: oh, well, yes) is closed-class and non-referential. The
    # alternative was -1, which would spend ignore budget on a class that is
    # vanishingly rare in encyclopaedic prose.
    "UH":   0,

    # -- L2 NOUN -------------------------------------------------------------
    "NN":   1, "NNS":  1,     # common noun, singular / plural
    "NNP":  1, "NNPS": 1,     # proper noun, singular / plural (§4.1: both)

    # -- L3 VERB -------------------------------------------------------------
    # Auxiliary and copular verbs belong to L1 per §4.1, but Penn does not
    # distinguish them from main verbs -- "is" is VBZ either way. The correction
    # is `AUXILIARY_SURFACE_FORMS` below, applied before this table.
    "VB":   2, "VBD":  2, "VBG": 2,
    "VBN":  2, "VBP":  2, "VBZ": 2,

    # -- L4 MODIFIER ---------------------------------------------------------
    "JJ":   3, "JJR":  3, "JJS": 3,     # adjective, comparative, superlative
    "RB":   3, "RBR":  3, "RBS": 3,     # adverb, comparative, superlative

    # -- L5 PUNCT_SYM --------------------------------------------------------
    ".":    4, ",":    4, ":":   4,
    "``":   4, "''":   4,
    "(":    4, ")":    4,
    "-LRB-": 4, "-RRB-": 4,             # bracket spellings some taggers emit
    "#":    4, "$":    4, "SYM": 4,
    # LS (list-item marker: "1.", "a)") is a marker, not a numeral in running
    # text, so it sits with the symbols rather than in L6.
    "LS":   4,

    # -- L6 NUM_SUBWORD ------------------------------------------------------
    "CD":   5,   # cardinal number. Subword pieces reach L6 by surface class, not
                 # by tag -- see `surface_class_family`.
}

# Tags that are deliberately NOT mapped. Listed explicitly rather than left to
# fall through, so the import guard can tell "considered and excluded" from
# "forgotten".
PENN_UNMAPPED = {
    # A foreign word has no English part of speech. Mapping it to L2_NOUN would
    # assert a lexical class the tagger did not find, and FW is rare enough in
    # wikitext that the ignore budget absorbs it.
    "FW",
    "NIL",       # the tagger's own "no tag"
    "-NONE-",    # Penn's null element; should never reach a surface token
}

# ---------------------------------------------------------------------------
# 3a. The auxiliary/copular override
# ---------------------------------------------------------------------------
#
# §4.1 puts "auxiliary/copular verbs" in L1 FUNCTION, but Penn tags them VB*, the
# same as main verbs. So an override is unavoidable. It is kept to the BE
# paradigm alone, deliberately:
#
#   BE has no main-verb use in English other than the copula, and §4.1 assigns the
#   copula to L1. So overriding BE forms adds no judgement the manifest has not
#   already made.
#
#   HAVE and DO are NOT here, even though both are common auxiliaries, because
#   both have genuine main-verb uses ("I have a car", "I do my homework").
#   Deciding between those uses is exactly the job of the type-level majority
#   vote in T-L3.1; hard-coding it would bypass the mechanism whose whole purpose
#   is to make that call from the corpus rather than from an author's intuition.
#   The consequence is stated plainly: HAVE and DO land in L3 VERB.
#
# The negative particle is included. Penn tags "not"/"n't" as RB, which would put
# negation in L4 MODIFIER alongside open-class adverbs; it is a closed-class
# grammatical operator and §4.1's L1 list names "particles". wikitext is
# pre-tokenized with clitics split, so "n't" is its own surface token.
AUXILIARY_SURFACE_FORMS = frozenset({
    "be", "am", "is", "are", "was", "were", "been", "being",
    "'s", "'re", "'m",            # contracted copula (wikitext splits clitics)
    "not", "n't",                 # negative particle
})


def penn_to_family(tag: str, surface: str | None = None) -> int:
    """Family index for one tagged word occurrence, or `NO_FAMILY_IDX`.

    `surface` is the word as it appeared. When given, the BE/negation override is
    applied FIRST, so an auxiliary tagged VBZ lands in L1 rather than L3. Passing
    `surface=None` gives the pure tag mapping, which is what the unit checks use
    to test the table in isolation.

    Returns `NO_FAMILY_IDX` for a tag in `PENN_UNMAPPED`. RAISES for a tag in
    neither table -- an unknown tag means the tagger emitted something this
    manifest has not considered, and inventing a family for it is the E7 mistake.
    The import guard makes this unreachable for the standard Penn tagset, so a
    raise here means a non-standard tagger.
    """
    if surface is not None and surface.strip().lower() in AUXILIARY_SURFACE_FORMS:
        return FAMILY_TO_IDX["L1"]
    if tag in PENN_TO_FAMILY:
        return PENN_TO_FAMILY[tag]
    if tag in PENN_UNMAPPED:
        return NO_FAMILY_IDX
    raise KeyError(
        f"Penn tag {tag!r} is in neither PENN_TO_FAMILY nor PENN_UNMAPPED. "
        "An unmapped tag must be classified explicitly, never given a fallback "
        "family (updated_rules.md §2: no catch-all expert). Add it to whichever "
        "table is correct and say why in the comment."
    )


# ---------------------------------------------------------------------------
# 3b. Surface class, for types with no standalone-POS evidence
# ---------------------------------------------------------------------------

_PUNCT_SYM_CHARS = frozenset(
    "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
    "–—‘’“”…·«»"  # – — ‘ ’ “ ” … · « »
    "§°±×÷†‡′″€£¥"
)
_NUMERIC_EXTRA_CHARS = frozenset(".,:/-+%–—")


def surface_class_family(surface: str) -> int:
    """Family for a vocabulary type the corpus never shows as a standalone word.

    The §4.1 phrase this implements is "numerals, and subword continuation pieces
    with no standalone POS" (L6), against §4.3's "fragments that have no POS"
    (-1). The two overlap, so the boundary is drawn here and stated:

      digits          -> L6.  "1998" and the piece "199" are both numeral material.
      punct/symbol    -> L5.  A type made only of punctuation is punctuation
                              whether or not the tagger ever saw it alone.
      contains a letter -> L6. This is the "subword continuation piece" case:
                              `ing`, `tion`, `pre` are word-internal material with
                              no part of speech of their own.
      anything else   -> -1.  Whitespace-only pieces, lone continuation bytes of a
                              multi-byte character, unused vocabulary entries.
                              These are §4.3's fragments, and they are what the 2%
                              budget is measured against.

    This is a DESIGN CHOICE, not a measurement. The alternative -- sending every
    no-evidence type to -1 -- would make the ignore share large and the budget
    check vacuous; the alternative in the other direction, sending everything to
    L6, would make L6 a catch-all and reproduce E7. The share each branch produces
    is written to the manifest by T-L3.1 so the split can be audited.
    """
    s = surface.strip()
    if not s:
        return NO_FAMILY_IDX
    if any(ch.isdigit() for ch in s) and all(
        ch.isdigit() or ch in _NUMERIC_EXTRA_CHARS for ch in s
    ):
        return FAMILY_TO_IDX["L6"]
    if all(ch in _PUNCT_SYM_CHARS for ch in s):
        return FAMILY_TO_IDX["L5"]
    if any(ch.isalpha() for ch in s):
        return FAMILY_TO_IDX["L6"]
    return NO_FAMILY_IDX


# ---------------------------------------------------------------------------
# 4. The unmapped budget
# ---------------------------------------------------------------------------

# `plan_language.md` §4.3. The number that replaces arithmetic's `raise`: Gate L2
# fails the build if more than this fraction of TRAIN TOKENS (not types -- a
# family can be 2% of types and 40% of tokens) carries `NO_FAMILY_IDX`. Kept here
# rather than in the gate so the manifest writer and the gate read one constant.
UNMAPPED_TOKEN_BUDGET = 0.02


# ---------------------------------------------------------------------------
# 5. Import-time consistency guards
# ---------------------------------------------------------------------------
#
# `families.py` raises at import when `OP_TARGET_DEPTH` and `OP_TO_EXPERT`
# disagree, because a silently missing entry would receive a fallback and that is
# the same defect class as the removed E7 catch-all. The equivalent invariants
# here are checked the same way, at import, so a bad edit fails on the first
# import rather than after a 6-minute dataset build.

_bad_labels = [lbl for lbl in LANG_FAMILY_LABELS if "/" in lbl]
if _bad_labels:
    raise ValueError(
        f"Family labels must not contain '/': {_bad_labels}. A '/' inside a W&B "
        "metric label creates spurious nesting (updated_rules.md §9), which is "
        "why the arithmetic labels read 'E1_ADD_SUB' and not 'E1 ADD/SUB'."
    )

if NUM_FAMILIES != 6:
    raise ValueError(
        f"NUM_FAMILIES must be 6, got {NUM_FAMILIES}. Six is what keeps the MoE "
        "axis width identical to the arithmetic study, so a difference between "
        "the two studies is not a difference in expert count."
    )

_overlap = set(PENN_TO_FAMILY) & set(PENN_UNMAPPED)
if _overlap:
    raise ValueError(
        f"Penn tags appear in both PENN_TO_FAMILY and PENN_UNMAPPED: "
        f"{sorted(_overlap)}. A tag is either mapped or explicitly ignored; "
        "being both makes `penn_to_family` order-dependent."
    )

_bad_idx = {t: f for t, f in PENN_TO_FAMILY.items()
            if not (0 <= f < NUM_FAMILIES)}
if _bad_idx:
    raise ValueError(
        f"PENN_TO_FAMILY maps tags outside 0..{NUM_FAMILIES - 1}: {_bad_idx}. "
        "Use NO_FAMILY_IDX via PENN_UNMAPPED to ignore a tag; never encode "
        "'ignore' as an out-of-range family index."
    )

_unreachable = sorted(set(range(NUM_FAMILIES)) - set(PENN_TO_FAMILY.values()))
if _unreachable:
    raise ValueError(
        "No Penn tag maps to families "
        f"{[LANG_FAMILY_LABELS[i] for i in _unreachable]}. A family nothing can "
        "be assigned to gives a structurally empty confusion-matrix row, which "
        "reads as a routing failure rather than as a manifest defect."
    )

# The standard Penn Treebank tagset -- the 36 word tags plus the 9 punctuation
# tags. The tagger cannot emit anything outside this set, so full coverage here is
# what makes `penn_to_family`'s KeyError unreachable in normal operation.
_PENN_TAGSET = {
    "CC", "CD", "DT", "EX", "FW", "IN", "JJ", "JJR", "JJS", "LS", "MD",
    "NN", "NNS", "NNP", "NNPS", "PDT", "POS", "PRP", "PRP$",
    "RB", "RBR", "RBS", "RP", "SYM", "TO", "UH",
    "VB", "VBD", "VBG", "VBN", "VBP", "VBZ",
    "WDT", "WP", "WP$", "WRB",
    ".", ",", ":", "``", "''", "(", ")", "#", "$",
}
_uncovered = sorted(_PENN_TAGSET - set(PENN_TO_FAMILY) - set(PENN_UNMAPPED))
if _uncovered:
    raise ValueError(
        f"Penn tags with no family decision: {_uncovered}. Every tag the tagger "
        "can emit must be either mapped to a family or listed in PENN_UNMAPPED "
        "with a stated reason -- leaving one out would send real tokens to a "
        "KeyError mid-build, or worse, to a fallback."
    )
