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
print("=== T-L3.2  routing_accuracy is DEMOTED to routing_agreement_with_pos ===")
# ---------------------------------------------------------------------------

from more import metrics as M      # noqa: E402

check("TL3.2a the language key is not named routing_accuracy",
      M.routing_agreement_key("language") == "val/routing_agreement_with_pos"
      and "routing_accuracy" not in M.routing_agreement_key("language"),
      M.routing_agreement_key("language"))

check("TL3.2b the ARITHMETIC key is unchanged, so Gate L0's contract still holds",
      M.routing_agreement_key("arithmetic") == "val/routing_accuracy"
      and M.routing_agreement_key() == "val/routing_accuracy",
      "G5.2b checks this exact string on a single-expert run; the default has to "
      "stay arithmetic for absence-means-arithmetic to survive")

check("TL3.2c the two spellings are DIFFERENT, so no exporter can join them",
      len(set(M.routing_agreement_keys_all())) == 2,
      f"{M.routing_agreement_keys_all()} -- one column of accuracies and one of "
      f"agreements cannot be silently averaged together")

_task_raised = False
try:
    M.routing_agreement_key("vision")
except KeyError:
    _task_raised = True
check("TL3.2d an unknown task raises rather than defaulting to a name",
      _task_raised,
      "a default here would publish a language agreement number under "
      "val/routing_accuracy, silently, in the key the exporter joins on")

# The real code path, not just the constant: build a synthetic confusion matrix and
# ask the log-dict builder for both tasks.
_conf = np.array([[8, 1, 0, 0, 0, 1],
                  [1, 7, 1, 0, 1, 0],
                  [0, 1, 9, 0, 0, 0],
                  [0, 0, 1, 6, 2, 1],
                  [1, 0, 0, 1, 8, 0],
                  [0, 1, 0, 1, 0, 8]], dtype=np.int64)
_acc = M.routing_accuracy_from_confusion(_conf, 6)
_ld_lang = M.paper_metrics_to_wandb(_conf, _acc, {}, {}, 6, task="language")
_ld_arith = M.paper_metrics_to_wandb(_conf, _acc, {}, {}, 6, task="arithmetic")
_ld_default = M.paper_metrics_to_wandb(_conf, _acc, {}, {}, 6)

_lang_offenders = [k for k in _ld_lang if "routing_accuracy" in k]
check("TL3.2e a language log dict contains NO key spelled routing_accuracy",
      not _lang_offenders
      and "val/routing_agreement_with_pos" in _ld_lang,
      f"{len(_ld_lang)} keys emitted, agreement published as "
      f"val/routing_agreement_with_pos = {_ld_lang['val/routing_agreement_with_pos']:.4f}"
      if not _lang_offenders else f"OFFENDERS {_lang_offenders}")

check("TL3.2f the number itself is unchanged -- only the name moved",
      abs(_ld_lang["val/routing_agreement_with_pos"]
          - _ld_arith["val/routing_accuracy"]) < 1e-15
      and abs(_ld_lang["val/routing_agreement_with_pos"] - _acc) < 1e-15,
      f"both {_acc:.6f}, and equal to the confusion diagonal fraction")

check("TL3.2g omitting `task` reproduces the arithmetic dict key-for-key",
      set(_ld_default) == set(_ld_arith)
      and all(_ld_default[k] == _ld_arith[k] for k in _ld_default
              if isinstance(_ld_default[k], (int, float, str))),
      f"{len(_ld_default)} keys identical -- an existing arithmetic caller that "
      f"passes no task is byte-identical to before T-L3.2")

# -- "primary" made mechanical ------------------------------------------------
# §4.4 says the permutation-invariant metrics become PRIMARY for language and the
# POS-agreement number is supplementary. `LANGUAGE_SPECIALIZATION_ORDER` is what
# that means as an object a results writer can consume, rather than a convention
# living in a prose file: the agreement key is LAST in it.
_order = M.LANGUAGE_SPECIALIZATION_ORDER
check("TL3.2h the permutation-invariant metrics come FIRST and agreement LAST",
      _order[-1] == M.routing_agreement_key("language")
      and {"val/routing_hungarian_accuracy", "val/routing_ami",
           "val/routing_purity"} <= set(_order[:-1]),
      " -> ".join(k.replace("val/routing_", "") for k in _order))

check("TL3.2i every name in that order is a key the log dict actually emits",
      all(k in _ld_lang for k in _order),
      f"all {len(_order)} present in a real language log dict, so the order is "
      f"consumable rather than aspirational")

# -- the caption travels with the number --------------------------------------
_cap = M.routing_agreement_caption("language")
_cap_l = _cap.lower()
check("TL3.2j the language caption says agreement-with-a-prior, NOT accuracy",
      "not accuracy" in _cap_l and "prior" in _cap_l
      and "linguistic hypothesis" in _cap_l
      and "functional ground truth" in _cap_l,
      _cap[:120] + "...")

check("TL3.2k the arithmetic caption still claims a functional ground truth",
      "functional ground truth" in M.routing_agreement_caption("arithmetic")
      and "NOT ACCURACY" not in M.routing_agreement_caption("arithmetic"),
      "the two captions make different claims, which is the whole point")

# metrics.json is written by a real run, which needs the Phase L-4 model. What is
# checkable now is that the writer names the caption at all -- asserted on the
# source, because a caption that lives only on the console is a caption that never
# reaches a table.
with open(os.path.join(CODE, "more", "engine.py"), "r", encoding="utf-8") as fh:
    _eng = fh.read()
check("TL3.2l engine.py writes the key AND the caption into metrics.json",
      'metrics["routing_agreement_caption"]' in _eng
      and 'metrics["routing_agreement_metric_key"]' in _eng
      and "routing_agreement_key(_task)" in _eng,
      "so the artifact the exporter and the paper draft read carries the caveat, "
      "not just stdout")

check("TL3.2m no site in engine.py still hard-codes the routing key",
      _eng.count('"val/routing_accuracy"') == 0
      and _eng.count("'val/routing_accuracy'") == 0,
      "the log dict, console line, results.tsv header and metrics.json N/A "
      "contract all resolve it through routing_agreement_key(_task)")

skip("TL3.2n results.md ordering",
     "the language results writer does not exist yet (Phase L-9); "
     "LANGUAGE_SPECIALIZATION_ORDER is the contract it must consume, and TL3.2h "
     "pins it")


# ---------------------------------------------------------------------------
print()
print("=== T-L3.3  Shuffled control (ablation G): the null the POS number needs ===")
# ---------------------------------------------------------------------------

_ctrl_built = [(c, m) for c, m in _built
               if "token_family_shuffled_sha256" in m]
if not _ctrl_built:
    skip("TL3.3a-k", "control not built; run data/lang/build_shuffled_control.py")

for corpus, man in _ctrl_built:
    tag = f"[{corpus}]"
    real = np.load(os.path.join(LANG, corpus, "token_family.npy"))
    cpath = os.path.join(LANG, corpus, "token_family_shuffled.npy")
    draws = np.load(cpath)
    V = man["vocab_size"]
    nd = man["shuffled_control_n_draws"]

    check(f"TL3.3a {tag} the control is (n_draws, V) int8 with its hash recorded",
          draws.shape == (nd, V) and draws.dtype == np.int8
          and _sha256_file(cpath) == man["token_family_shuffled_sha256"],
          f"shape {draws.shape}, {nd} independent draws, "
          f"sha256 {man['token_family_shuffled_sha256'][:12]}")

    check(f"TL3.3b {tag} every draw uses the same family alphabet as the real one",
          all(set(np.unique(draws[d]).tolist()) <= set(range(6)) | {-1}
              for d in range(nd)),
          "no draw invents a seventh family or a second ignore label")

    # The -1 population must be identical, or the two metric sets are computed over
    # different token populations and the pair is not comparable.
    check(f"TL3.3c {tag} the unmapped types are EXACTLY preserved in every draw",
          all(np.array_equal(draws[d] < 0, real < 0) for d in range(nd))
          and man["shuffled_control_unmapped_preserved"] is True,
          f"{int((real < 0).sum())} types stay at -1, so both metric sets are "
          f"computed over the same tokens")

    check(f"TL3.3d {tag} no draw is the real partition, or a copy of another draw",
          all(not np.array_equal(draws[d], real) for d in range(nd))
          and len({draws[d].tobytes() for d in range(nd)}) == nd,
          f"{nd} distinct partitions, none equal to POS")

    # -- the marginals, recomputed from the arrays -----------------------------
    _tr = np.load(os.path.join(LANG, corpus, "train.npy"), mmap_mode="r")
    _cnt = np.zeros(V, dtype=np.int64)
    _rows = max(1, (1 << 22) // _tr.shape[1])
    for _i in range(0, _tr.shape[0], _rows):
        _cnt += np.bincount(np.asarray(_tr[_i:_i + _rows], dtype=np.int64).ravel(),
                            minlength=V)
    _tot = float(_cnt[real >= 0].sum())
    _share = lambda a: np.array([_cnt[a == f].sum() / _tot for f in range(6)])
    _real_sh = _share(real)
    _err = max(float(np.abs(_share(draws[d]) - _real_sh).max()) for d in range(nd))

    check(f"TL3.3e {tag} the control matches the real TOKEN proportions",
          _err < 5e-3 and abs(_err - man["shuffled_control_worst_token_share_error"])
          < 1e-9,
          f"worst per-family token-share error {_err:.2e} over {nd} draws "
          f"(recomputed here from train.npy, matches the manifest)")

    # The quantity that actually sets AMI's chance level.
    _H = lambda p: float(-(p[p > 0] * np.log(p[p > 0])).sum() / math.log(6))
    _hr, _hc = _H(_real_sh), [_H(_share(draws[d])) for d in range(nd)]
    check(f"TL3.3f {tag} the control's marginal ENTROPY matches, which is what "
          f"sets AMI's chance level",
          abs(_hr - float(np.mean(_hc))) < 2e-3,
          f"real {_hr:.6f} vs control {np.mean(_hc):.6f} "
          f"(+-{np.std(_hc):.1e}) -- so a difference in AMI cannot be an artifact "
          f"of differently balanced partitions")

    # T-L3.3's wording says TYPE counts; §4.4 says TOKEN proportions. The manifest
    # records which was implemented and why; this pins that the departure is real
    # and declared rather than an accident.
    check(f"TL3.3g {tag} the type-vs-token specification conflict is declared",
          "TOKEN proportions" in man["shuffled_control_matched"]
          and "T-L3.3" in man["shuffled_control_matched_note"]
          and "§4.4" in man["shuffled_control_matched_note"],
          "type counts differ from the real partition BY DESIGN "
          f"(L1: {int((real == 0).sum())} real vs "
          f"{int((draws[0] == 0).sum())} control), and the manifest says why")

# -- the control actually DISCRIMINATES, shown on synthetic routers -----------
# The point of ablation G is that it separates a real partition from the metric's
# floor. That is testable now, with no trained model: score two synthetic routers
# against POS and against the control and require the control to behave as a floor
# for one and not the other. A control that could not do this would be decoration.
if _ctrl_built:
    _c, _m = _ctrl_built[0]
    _real = np.load(os.path.join(LANG, _c, "token_family.npy"))
    _draws = np.load(os.path.join(LANG, _c, "token_family_shuffled.npy"))
    _val = np.load(os.path.join(LANG, _c, "val.npy"), mmap_mode="r")
    _ids = np.asarray(_val[:400], dtype=np.int64).ravel()

    _rng = np.random.default_rng(7)
    _random_router = _rng.integers(0, 6, size=_ids.size)
    _oracle_router = np.clip(_real[_ids], 0, 5)

    _cmp_rand = M.specialization_vs_control(_random_router, _ids, _real, _draws, 6)
    _cmp_pos = M.specialization_vs_control(_oracle_router, _ids, _real, _draws, 6)

    _r_ami = _cmp_rand["metrics"]["ami"]
    _p_ami = _cmp_pos["metrics"]["ami"]

    check("TL3.3h a RANDOM router scores at the floor against BOTH partitions",
          abs(_r_ami["delta"]) < 0.02,
          f"AMI vs POS {_r_ami['real']:.4f}, vs control "
          f"{_r_ami['control_mean']:.4f} +- {_r_ami['control_std']:.4f}, "
          f"delta {_r_ami['delta']:+.4f} -- no apparent specialization, correctly")

    check("TL3.3i a POS-PERFECT router beats the control by a wide margin",
          _p_ami["real"] > 0.9 and _p_ami["delta"] > 0.5,
          f"AMI vs POS {_p_ami['real']:.4f}, vs control "
          f"{_p_ami['control_mean']:.4f}, delta {_p_ami['delta']:+.4f} "
          f"(z = {_p_ami['delta_z']:.1f}) -- so the control is a floor, not a cap")

    check("TL3.3j the control separates the two routers, which is its whole job",
          _p_ami["delta"] - _r_ami["delta"] > 0.5,
          f"delta(POS-perfect) {_p_ami['delta']:+.4f} vs delta(random) "
          f"{_r_ami['delta']:+.4f}: an AMI of {_r_ami['real']:.3f} against POS "
          f"would look like weak specialization WITHOUT the control")

    check("TL3.3k every compared metric reports mean, std, delta and z together",
          all(set(_cmp_pos["metrics"][k]) ==
              {"real", "control_mean", "control_std", "delta", "delta_z"}
              for k in M.CONTROL_COMPARED_METRICS)
          and _cmp_pos["n_control_draws"] == _draws.shape[0],
          f"{list(M.CONTROL_COMPARED_METRICS)} over "
          f"{_cmp_pos['n_control_draws']} draws -- T-L3.3 requires the difference "
          f"with its uncertainty, not the raw pair")

    _log = M.control_comparison_to_wandb(_cmp_pos)
    check("TL3.3l the comparison flattens to log keys with None omitted, not zeroed",
          all(v is not None for v in _log.values())
          and any(k.endswith("_delta") for k in _log)
          and any(k.endswith("_control_std") for k in _log),
          f"{len(_log)} keys under val/routing_control/ -- an undefined difference "
          f"is absent, because 0.0 there would read as 'POS is no better than noise'")





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
