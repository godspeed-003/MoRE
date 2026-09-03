"""test_language_families.py - Phase L-3 verification for the POS family manifest.

Run:
    C:/Users/vedan/anaconda3/python.exe code/test_language_families.py

Scope: T-L3.0 .. T-L3.3 (`plan_language.md` §4). Kept out of
`run_correctness_suite.py` for the same reason as `test_language_data.py` and
`test_language_task_axis.py` -- Gate L0's entire content is that the arithmetic
total is still 356, so adding checks there would destroy the one number that says
the published arithmetic study still holds. Registered when the L1..L8 language
gate table is built.

WHAT THIS FILE IS FOR. `lang_families.py` is a manifest, so almost every defect it
can have is a *silent* one: a family no tag can reach gives an empty
confusion-matrix row that reads as a routing failure; a `/` in a label splits one
W&B metric into two; an accessor `metrics.py` imports from `families.py` but not
from here fails only when a language run reaches the metric layer, an hour in.

Two checks here are deliberately not restatements of the module. TL3.0e/f read the
import statements out of `metrics.py` and `__init__.py` with a regex and require
`lang_families` to satisfy whatever those files actually ask for -- so adding an
import there without adding the accessor here fails immediately, rather than the
check drifting out of date. TL3.0s re-imports the module in a clean subprocess and
inspects `sys.modules`, because "this module is dependency-free" is a claim about
import side effects that cannot be tested from inside a process that has already
imported torch.
"""

import json
import math
import os
import re
import subprocess
import sys

CODE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(CODE)
LANG = os.path.join(REPO, "data", "lang")
sys.path.insert(0, CODE)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

PASS, FAIL, SKIP = [], [], []


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))


def skip(name, reason):
    SKIP.append(name)
    print(f"  [SKIP] {name}  -- {reason}")


# ===========================================================================
print("\n=== T-L3.0  Six POS families, same shape as the arithmetic manifest ===")
# ===========================================================================

from more import lang_families as lf      # noqa: E402
from more import families as af           # noqa: E402

EXPECTED_LABELS = ["L1_FUNCTION", "L2_NOUN", "L3_VERB",
                  "L4_MODIFIER", "L5_PUNCT_SYM", "L6_NUM_SUBWORD"]

check("TL3.0a NUM_FAMILIES is exactly 6",
      lf.NUM_FAMILIES == 6, f"NUM_FAMILIES = {lf.NUM_FAMILIES}")

check("TL3.0b no family label contains '/' (W&B nesting rule)",
      not any("/" in lbl for lbl in lf.LANG_FAMILY_LABELS),
      " ".join(lf.LANG_FAMILY_LABELS))

check("TL3.0c the six labels are the §4.1 names, in order, and distinct",
      lf.LANG_FAMILY_LABELS == EXPECTED_LABELS
      and len(set(lf.LANG_FAMILY_LABELS)) == 6,
      "matches plan_language.md §4.1")

check("TL3.0d the expert axis is the SAME WIDTH as arithmetic's",
      lf.NUM_EXPERTS_CANONICAL == af.NUM_EXPERTS_CANONICAL == 6,
      f"language {lf.NUM_EXPERTS_CANONICAL} == arithmetic "
      f"{af.NUM_EXPERTS_CANONICAL} (so a study difference is not a width "
      f"difference)")


def _imported_names(path, module):
    """Names a file imports from `.<module>`, read from its source.

    Deliberately parsed rather than hard-coded: the point of TL3.0e/f is that
    `lang_families` satisfies whatever the consumer ACTUALLY asks for today, so
    the required list must come from the consumer, not from this test.
    """
    with open(path, "r", encoding="utf-8") as fh:
        src = fh.read()
    m = re.search(rf"from \.{module} import \(?([^)\n]*(?:\n[^)]*)?)\)?\n", src)
    if not m:
        return []
    return [n.strip() for n in m.group(1).replace("\n", " ").split(",")
            if n.strip() and n.strip() != "#"]


_metrics_needs = _imported_names(os.path.join(CODE, "more", "metrics.py"),
                                 "families")
_missing_m = [n for n in _metrics_needs if not hasattr(lf, n)]
check("TL3.0e every name metrics.py imports from families.py exists here",
      _metrics_needs and not _missing_m,
      f"needs {_metrics_needs}" + (f"  MISSING {_missing_m}" if _missing_m else ""))

_init_needs = _imported_names(os.path.join(CODE, "more", "__init__.py"),
                              "families")
# A name may be satisfied two ways: defined here, or explicitly declared
# arithmetic-only in `_ARITHMETIC_ONLY`. Both are decisions; what fails the check
# is a name in NEITHER bucket, i.e. one nobody has classified. So adding an import
# to `__init__.py` breaks this until someone says which kind of name it is.
_missing_i = [n for n in _init_needs
              if not hasattr(lf, n) and n not in lf._ARITHMETIC_ONLY]
_refused_here = sorted(set(lf._ARITHMETIC_ONLY) & set(_init_needs))
check("TL3.0f every name __init__.py re-exports is defined here or declared "
      "arithmetic-only",
      _init_needs and not _missing_i,
      f"{len(_init_needs)} names: {len(_init_needs) - len(_refused_here)} "
      f"defined, {_refused_here} refused"
      if not _missing_i else f"UNCLASSIFIED {_missing_i}")

_refused = []
for _n in sorted(lf._ARITHMETIC_ONLY):
    try:
        getattr(lf, _n)
    except AttributeError as exc:
        _refused.append(_n if "arithmetic-only" in str(exc) else None)
check("TL3.0f2 the arithmetic-only names RAISE with the reason, and are not "
      "defined as empty",
      _refused == sorted(lf._ARITHMETIC_ONLY),
      f"{sorted(lf._ARITHMETIC_ONLY)} -- an empty OP_TO_EXPERT would raise "
      f"KeyError at the indexing line instead of naming the mistake")


check("TL3.0g the ignore label is -1, identical to arithmetic's",
      lf.NO_FAMILY_IDX == af.NO_FAMILY_IDX == -1
      and lf.FAMILY_TO_IDX["UNMAPPED"] == -1,
      "so ignore_index=-1 and the (family >= 0) masks need no change")

check("TL3.0h expert_labels truncates below 6 and names extras positionally",
      lf.expert_labels(1) == ["L1_FUNCTION"]
      and lf.expert_labels(6) == EXPECTED_LABELS
      and lf.expert_labels(7) == EXPECTED_LABELS + ["L7"],
      "E=1 -> 1 label (the Gate 4 defect), E=7 -> 7 labels")

_e0_raised = False
try:
    lf.expert_labels(0)
except ValueError:
    _e0_raised = True
check("TL3.0i expert_labels(0) raises rather than returning []",
      _e0_raised, "a router over zero experts is a config error, not a case")

check("TL3.0j there is NO operation axis, and it is empty rather than absent",
      lf.ALL_OP_NAMES == [] and lf.NUM_OP_TYPES == 0
      and lf.OP_NAME_TO_IDX == {} and lf.op_target_depth_table() == [],
      "engine.py:803 enumerate([]) -> empty op_avg_depth -> reported N/A, "
      "not 0.0")

check("TL3.0k FAMILY_TO_IDX maps L1..L6 to 0..5 and nothing else to a family",
      [lf.FAMILY_TO_IDX[f"L{i + 1}"] for i in range(6)] == list(range(6))
      and set(lf.IDX_TO_FAMILY) == set(range(6)),
      "IDX_TO_FAMILY excludes -1, so a reverse lookup cannot name the ignore "
      "label as a family")


# ---------------------------------------------------------------------------
print()
print("=== T-L3.0  Penn tagset coverage: every tag decided, none defaulted ===")
# ---------------------------------------------------------------------------

check("TL3.0l PENN_TO_FAMILY and PENN_UNMAPPED are disjoint",
      not (set(lf.PENN_TO_FAMILY) & set(lf.PENN_UNMAPPED)),
      f"{len(lf.PENN_TO_FAMILY)} mapped, {len(lf.PENN_UNMAPPED)} explicitly "
      f"ignored")

_decided = set(lf.PENN_TO_FAMILY) | set(lf.PENN_UNMAPPED)
_uncovered = sorted(lf._PENN_TAGSET - _decided)
check("TL3.0m every standard Penn tag has an explicit decision",
      not _uncovered,
      f"all {len(lf._PENN_TAGSET)} tags (36 word + 9 punctuation)"
      if not _uncovered else f"UNCOVERED {_uncovered}")

_reachable = sorted(set(lf.PENN_TO_FAMILY.values()))
check("TL3.0n every one of the six families is reachable from some tag",
      _reachable == list(range(6)),
      "no structurally empty confusion-matrix row"
      if _reachable == list(range(6))
      else f"reachable = {[lf.LANG_FAMILY_LABELS[i] for i in _reachable]}")

_unknown_raised = False
try:
    lf.penn_to_family("NOT_A_PENN_TAG")
except KeyError:
    _unknown_raised = True
check("TL3.0o an unknown tag RAISES instead of getting a fallback family",
      _unknown_raised,
      "updated_rules.md §2: no catch-all. -1 is for tags decided to be ignored, "
      "not for tags nobody considered")

check("TL3.0p a tag in PENN_UNMAPPED returns -1, not a raise",
      lf.penn_to_family("FW") == -1,
      "FW has no English POS, so it is ignored -- a decision, unlike the above")

# The auxiliary override, and the case it deliberately does NOT cover.
_be = {s: lf.penn_to_family("VBZ", s) for s in ("is", "was", "been", "'s")}
check("TL3.0q BE forms tagged VB* land in L1 FUNCTION, not L3 VERB",
      all(v == 0 for v in _be.values()),
      " ".join(f"{k}->{lf.IDX_TO_FAMILY[v]}" for k, v in _be.items()))

_hd = {s: lf.penn_to_family("VBP", s) for s in ("have", "do")}
check("TL3.0r HAVE and DO are NOT overridden, so they land in L3 VERB",
      all(v == 2 for v in _hd.values()),
      "documented consequence: both have genuine main-verb uses, and choosing "
      "between them is the type-level majority vote's job, not a hard-coded one")

check("TL3.0s the negative particle is L1, not an L4 adverb",
      lf.penn_to_family("RB", "not") == 0
      and lf.penn_to_family("RB", "n't") == 0
      and lf.penn_to_family("RB", "quickly") == 3,
      "Penn tags 'not' as RB; negation is a closed-class particle (§4.1)")

check("TL3.0t the override needs the surface form -- tag alone is the pure map",
      lf.penn_to_family("VBZ") == 2 and lf.penn_to_family("VBZ", "is") == 0,
      "penn_to_family(tag) is the table; penn_to_family(tag, surface) applies "
      "the override first")


# ---------------------------------------------------------------------------
print()
print("=== T-L3.0  Surface class: the L6-vs-(-1) boundary of §4.1 vs §4.3 ===")
# ---------------------------------------------------------------------------

_L5, _L6, _IGN = 4, 5, -1
_cases = [
    ("1998",  _L6, "a whole numeral"),
    ("199",   _L6, "a numeral fragment is still numeral material"),
    ("3.14",  _L6, "digits with a decimal separator"),
    ("ing",   _L6, "subword continuation piece, §4.1's 'no standalone POS'"),
    ("tion",  _L6, "subword continuation piece"),
    ("zx",    _L6, "§4.3's example fragment -- alphabetic, so L6 not -1"),
    ("...",   _L5, "punctuation-only type"),
    (" ,",    _L5, "leading-space punctuation (byte-level BPE keeps the space)"),
    ("   ",   _IGN, "whitespace-only piece has no class at all"),
    ("",      _IGN, "empty surface"),
]
_bad = [(s, lf.surface_class_family(s), want) for s, want, _ in _cases
        if lf.surface_class_family(s) != want]
check("TL3.0u the surface-class fallback splits L6 from -1 as documented",
      not _bad,
      f"{len(_cases)} cases: numerals and lettered fragments -> L6, "
      f"punctuation -> L5, whitespace/empty -> -1"
      if not _bad else f"WRONG {_bad}")

check("TL3.0v the unmapped budget is the §4.3 number, in one place",
      lf.UNMAPPED_TOKEN_BUDGET == 0.02,
      f"{lf.UNMAPPED_TOKEN_BUDGET:.0%} of TRAIN TOKENS -- read by both the "
      f"manifest writer and Gate L2, so they cannot disagree")


# ---------------------------------------------------------------------------
print()
print("=== T-L3.0  The manifest file itself imports nothing heavy ===")
# ---------------------------------------------------------------------------

# Loaded BY PATH, not as `more.lang_families`. Importing it through the package
# runs `more/__init__.py`, which imports engine/model and therefore torch -- so
# the package route would fail this check for a reason that has nothing to do with
# this file. The claim under test is the one the module docstring makes: the
# manifest's OWN imports are empty, so a build script or a bare checkout can read
# it without the framework.
_probe = (
    "import importlib.util, sys\n"
    "spec = importlib.util.spec_from_file_location('_lf', r'%s')\n"
    "mod = importlib.util.module_from_spec(spec)\n"
    "spec.loader.exec_module(mod)\n"
    "assert mod.NUM_FAMILIES == 6\n"
    "print(','.join(m for m in ('torch','nltk','numpy','datasets','tokenizers')\n"
    "               if m in sys.modules))\n"
) % os.path.join(CODE, "more", "lang_families.py")
_res = subprocess.run([sys.executable, "-c", _probe], cwd=CODE,
                      capture_output=True, text=True)
_heavy = _res.stdout.strip()
check("TL3.0w the manifest file loads with no torch, nltk, numpy or datasets",
      _res.returncode == 0 and _heavy == "",
      "clean subprocess, loaded by path: sys.modules has none of them"
      if _res.returncode == 0 and _heavy == ""
      else f"PULLED IN {_heavy or _res.stderr.strip()[-200:]}")



# ---------------------------------------------------------------------------
print()
print("=== T-L3.1  token_family[V]: frozen from majority POS, audited ===")
# ---------------------------------------------------------------------------

import numpy as np      # noqa: E402

CORPORA = ("wikitext-2", "wikitext-103")


def _manifest(corpus):
    p = os.path.join(LANG, corpus, "dataset_meta.json")
    if not os.path.exists(p):
        return None
    with open(p, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _sha256_file(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


_built = [(c, m) for c in CORPORA
          if (m := _manifest(c)) is not None and "token_family_sha256" in m]

if not _built:
    skip("TL3.1a-l", "no corpus has a family lookup yet; run "
                     "data/lang/build_family_lookup.py")

for corpus, man in _built:
    tag = f"[{corpus}]"
    path = os.path.join(LANG, corpus, "token_family.npy")
    V = man["vocab_size"]
    table = np.load(path)

    check(f"TL3.1a {tag} the lookup has length V and dtype int8",
          table.shape == (V,) and table.dtype == np.int8,
          f"shape {table.shape} dtype {table.dtype} (V = {V})")

    check(f"TL3.1b {tag} every entry is a real family or the ignore label",
          set(np.unique(table).tolist()) <= set(range(6)) | {-1},
          f"values = {sorted(np.unique(table).tolist())}")

    check(f"TL3.1c {tag} the recorded hash matches the file on disk",
          _sha256_file(path) == man["token_family_sha256"],
          f"sha256 {man['token_family_sha256'][:12]}")

    check(f"TL3.1d {tag} the EOT token is ignored, not given a family",
          int(table[man["eot_id"]]) == -1,
          f"token_family[{man['eot_id']}] = {int(table[man['eot_id']])} "
          f"-- a document separator is not a lexical class")

    # -- the -1 share, recounted rather than read ---------------------------
    n_unmapped_types = int((table == -1).sum())
    check(f"TL3.1e {tag} the manifest's unmapped TYPE count matches the array",
          n_unmapped_types == man["family_counts_types_unmapped"],
          f"{n_unmapped_types:,} of {V:,} types "
          f"({100 * n_unmapped_types / V:.2f}%)")

    recount = {lbl: int((table == f).sum())
               for f, lbl in enumerate(man["family_labels"])}
    check(f"TL3.1f {tag} every per-family TYPE count matches the array",
          recount == man["family_counts_types"],
          " ".join(f"{k.split('_')[0]}={v:,}" for k, v in recount.items()))

    check(f"TL3.1g {tag} the type counts plus the unmapped ones account for V",
          sum(recount.values()) + n_unmapped_types == V,
          f"{sum(recount.values()):,} + {n_unmapped_types:,} = {V:,}")

    # -- token-level, recomputed from the packed arrays ---------------------
    # `plan_language.md` §4.3's budget is over TOKENS, so the check that matters
    # gathers the lookup through the real corpus rather than trusting the writer.
    tok_ok, tok_detail = True, []
    for split in ("train", "val", "test"):
        arr = np.load(os.path.join(LANG, corpus, f"{split}.npy"), mmap_mode="r")
        counts = np.zeros(7, dtype=np.int64)
        rows = max(1, (1 << 22) // arr.shape[1])
        for i in range(0, arr.shape[0], rows):
            fam = table[np.asarray(arr[i:i + rows], dtype=np.int64)]
            counts += np.bincount(fam.ravel() + 1, minlength=7)
        rec = man["family_counts_tokens"][split]
        same = (int(counts.sum()) == rec["total_tokens"]
                and int(counts[0]) == rec["unmapped"]
                and all(int(counts[f + 1]) == rec["by_family"][lbl]
                        for f, lbl in enumerate(man["family_labels"])))
        tok_ok = tok_ok and same
        tok_detail.append(f"{split} {int(counts[0]):,}/{int(counts.sum()):,}"
                          f"={int(counts[0]) / max(1, int(counts.sum())):.4%}")
    check(f"TL3.1h {tag} token-level family counts reproduce from the .npy files",
          tok_ok, "  ".join(tok_detail))

    # -- the §4.3 budget: measured, published, and bounded -------------------
    share = man["unmapped_token_share_train"]
    check(f"TL3.1i {tag} the unmapped TRAIN-TOKEN share is inside the 2% budget",
          share is not None and share <= lf.UNMAPPED_TOKEN_BUDGET
          and man["unmapped_within_budget"] is True,
          f"{share:.4%} <= {lf.UNMAPPED_TOKEN_BUDGET:.0%} -- an ignore label "
          f"whose share is unmeasured is the E7 catch-all in another costume")

    check(f"TL3.1j {tag} the budget itself is recorded, not just the verdict",
          man.get("unmapped_token_budget") == lf.UNMAPPED_TOKEN_BUDGET,
          "so a later reader can tell a passing build from a loosened budget")

    # -- type-level and token-level are genuinely different ------------------
    # T-L2.4 requires both denominators on the grounds that "a family can be 2% of
    # types and 40% of tokens". This checks that the justification is a measured
    # fact about this corpus rather than a hypothetical.
    mapped_tok = (man["family_counts_tokens"]["train"]["total_tokens"]
                  - man["family_counts_tokens"]["train"]["unmapped"])
    mapped_typ = sum(recount.values())
    gaps = {lbl: (man["family_counts_tokens"]["train"]["by_family"][lbl] / mapped_tok
                  - recount[lbl] / mapped_typ)
            for lbl in man["family_labels"]}
    worst = max(gaps, key=lambda k: abs(gaps[k]))
    check(f"TL3.1k {tag} type share and token share diverge by more than 10 points "
          f"for at least one family",
          abs(gaps[worst]) > 0.10,
          f"{worst}: {100 * recount[worst] / mapped_typ:.1f}% of types vs "
          f"{100 * man['family_counts_tokens']['train']['by_family'][worst] / mapped_tok:.1f}% "
          f"of tokens -- a load-balance number read against the wrong denominator "
          f"is uninterpretable")

    # -- the oracle partition is NOT balanced, and that has consequences -----
    p = [man["family_counts_tokens"]["train"]["by_family"][lbl] / mapped_tok
         for lbl in man["family_labels"]]
    H = -sum(x * math.log(x) for x in p if x > 0) / math.log(6)
    check(f"TL3.1l {tag} the oracle partition's own normalized entropy is < 1",
          H < 0.95,
          f"H/log(6) = {H:.4f}, largest/smallest family token ratio "
          f"{max(p) / min(p):.2f}x -- so a router pushed to H=1.0 by the balance "
          f"loss must DISAGREE with POS. Read every language load-entropy number "
          f"against {H:.4f}, not against 1.0")

    # -- provenance of each decision, and convergence ------------------------
    prov = man["family_type_provenance"]
    check(f"TL3.1m {tag} every vocabulary entry is accounted for by provenance",
          prov["voted"] + prov["surface"] + prov["special"] == V,
          f"{prov['voted']:,} voted + {prov['surface']:,} surface-class + "
          f"{prov['special']:,} special = {V:,}")

    vs = man["family_vote_stats"]
    check(f"TL3.1n {tag} the words that could NOT vote are counted, not discarded "
          f"silently",
          vs["single_id"] + vs["multi_piece"] == vs["words"]
          and vs["multi_piece"] > 0,
          f"{vs['words']:,} words: {vs['single_id']:,} single-id "
          f"({100 * vs['single_id'] / vs['words']:.1f}%), "
          f"{vs['multi_piece']:,} multi-piece (no vote, by design), "
          f"{vs['tag_ignored']:,} tag-ignored")

    stab = man["family_vote_stability"]
    check(f"TL3.1o {tag} the majority survives a first-half / second-half split",
          stab["disagreement_rate"] is not None
          and stab["disagreement_rate"] < 0.10,
          f"{stab['disagreement_rate']:.4%} of {stab['types_in_both_halves']:,} "
          f"types flip -- the corpus decided the majority, it is not sampling "
          f"noise")

    check(f"TL3.1p {tag} the tokenization and tagger used are on the record",
          man.get("family_lookup_word_tokenization", "").startswith("whitespace")
          and "perceptron" in man.get("family_lookup_tagger", ""),
          f"{man.get('family_lookup_tagger')} over "
          f"{man.get('family_lookup_word_tokenization')}")

# `majority` decides ties by lowest family index rather than by insertion order,
# which is what lets the parallel and serial builds agree. Checked on the function
# because reproducing it end-to-end means two full corpus passes; the
# workers=1-vs-6 byte-identity was verified once by hand and recorded in the
# ledger.
sys.path.insert(0, LANG)
try:
    import build_family_lookup as bfl      # noqa: E402
    check("TL3.1q the majority vote breaks ties deterministically, lowest family "
          "index first",
          bfl.majority({3: 5, 1: 5, 5: 5}) == 1
          and bfl.majority({5: 9, 0: 2}) == 5,
          "a tie resolved by dict order would make the parallel and serial "
          "builds disagree")
except Exception as exc:      # pragma: no cover
    check("TL3.1q majority tie-break", False, f"{type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
print()
print("=== T-L3.2 / T-L3.3  not yet built ===")
# ---------------------------------------------------------------------------

skip("TL3.2 routing_agreement_with_pos replaces routing_accuracy",
     "metric-layer change not made yet")
skip("TL3.3 shuffled-control partition (ablation G)",
     "evaluation-time only; built after the metric layer is renamed")



# ===========================================================================
print()
print("=" * 78)
print(f"{len(PASS)} passed, {len(FAIL)} failed, {len(SKIP)} skipped")
if FAIL:
    print("\nFAILED:")
    for f in FAIL:
        print(f"  - {f}")
print("=" * 78)
sys.exit(1 if FAIL else 0)
