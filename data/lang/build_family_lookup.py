"""build_family_lookup.py - T-L3.1: freeze `token_family[V]` from majority POS.

Run (dev corpus, ~2 min):
    C:/Users/vedan/anaconda3/python.exe data/lang/build_family_lookup.py --corpus wikitext-2

Run (canonical corpus; use --workers, the tagger is pure Python):
    C:/Users/vedan/anaconda3/python.exe data/lang/build_family_lookup.py --corpus wikitext-103

Produces `data/lang/<corpus>/token_family.npy` -- an `int8` array of length V --
and appends the statistics behind every decision to `dataset_meta.json`.

WHY TYPE-LEVEL (`plan_language.md` §4.2). The oracle family is a property of the
BPE id, not of the occurrence. A per-occurrence oracle would make the "correct"
expert depend on context the router cannot see at its own input, so disagreement
would measure the tagger's context sensitivity rather than the router's behaviour --
a different and much stronger claim than "the router groups tokens by lexical
class". Type-level is the weaker, honest, cheap choice, and the training loop then
gets its labels from a single gather, `token_family[input_ids]`.

HOW A TYPE EARNS A VOTE. The train text is split on whitespace -- NOT with a
treebank tokenizer, measured reason below -- and tagged. Each word is encoded with
its leading space (byte-level BPE distinguishes " the" from "the"), and if it
encodes to exactly ONE id, that id receives one vote for the word's family. A word
that splits into several pieces votes for nothing: its part of speech is a property
of the word, and attributing it to an arbitrary piece would invent information.
Types with no votes fall to `lang_families.surface_class_family`, and the share
each branch produces is written to the manifest so the fallback can be audited
rather than trusted.

WHY WHITESPACE AND NOT `TreebankWordTokenizer`. Measured on wikitext-2: treebank
yields 1.0251x more tokens, and the excess is entirely units that DO NOT EXIST in
the byte stream the BPE was fitted on -- it splits WikiText's own `@-@` / `@.@` /
`@,@` escapes into `@`+`-`+`@` (2,830 occurrences per ~330 k tokens), rewrites `"`
as Penn's `` `` ``, splits `cannot` into `can`+`not`, and strips the final period
off `U.S.`. Tagging a unit that never appears in the corpus would break the
word-to-id alignment the vote depends on. WikiText is already Moses-tokenized with
spaces around punctuation, so whitespace IS its tokenization.

DETERMINISM. Vote counts merge associatively, so the parallel and serial paths must
agree exactly; `code/test_language_families.py` checks that `--workers 1` and
`--workers 4` produce a byte-identical lookup. Ties in the majority vote go to the
lowest family index, never to insertion order.
"""

import argparse
import collections
import hashlib
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

LANG = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(LANG))
sys.path.insert(0, os.path.join(REPO, "code"))
sys.path.insert(0, LANG)

import build_language_dataset as bld                      # noqa: E402


def _load_manifest_module():
    """Load `lang_families.py` BY PATH, not as `more.lang_families`.

    Importing it through the package runs `more/__init__.py`, which imports
    `engine`/`model` and therefore torch -- ~350 MB of resident set for a module
    that is 300 lines of dicts. On Windows there is no `fork`, so every
    `multiprocessing` worker re-imports this file and would pay that cost again:
    measured at 6 x 358 MB before this change, against 6 x ~40 MB after.

    This is the loading pattern `code/test_language_families.py` TL3.0w exists to
    keep available -- the manifest's own imports are empty, so it can be read
    without the framework, and here that is worth 1.9 GB.
    """
    import importlib.util
    path = os.path.join(REPO, "code", "more", "lang_families.py")
    spec = importlib.util.spec_from_file_location("_lang_families", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_LF = _load_manifest_module()
LANG_FAMILY_LABELS = _LF.LANG_FAMILY_LABELS
NUM_FAMILIES = _LF.NUM_FAMILIES
NO_FAMILY_IDX = _LF.NO_FAMILY_IDX
UNMAPPED_TOKEN_BUDGET = _LF.UNMAPPED_TOKEN_BUDGET
penn_to_family = _LF.penn_to_family
surface_class_family = _LF.surface_class_family


# Lines per work unit. Large enough that the per-chunk tagger/tokenizer lookup and
# the pickling of the returned Counter are negligible against the tagging itself;
# small enough that a 1.1 M-line corpus still gives every worker many units, so a
# slow chunk cannot leave one core finishing alone.
CHUNK_LINES = 2000

# Worker-global handles. Built once per process on first use rather than passed in:
# the tagger's weight table does not pickle cheaply, and on Windows `spawn` would
# re-send it for every chunk.
_TAGGER = None
_TOKENIZER = None
_TOK_PATH = None
_TEXT_PATH = None


def _worker_init(tokenizer_path, text_path):
    """Set the per-worker paths (Windows has no fork, so this runs per process)."""
    global _TOK_PATH, _TEXT_PATH
    _TOK_PATH = tokenizer_path
    _TEXT_PATH = text_path


def _handles():
    global _TAGGER, _TOKENIZER
    if _TAGGER is None:
        from nltk.tag import PerceptronTagger
        _TAGGER = PerceptronTagger()
    if _TOKENIZER is None:
        from tokenizers import Tokenizer
        _TOKENIZER = Tokenizer.from_file(_TOK_PATH)
    return _TAGGER, _TOKENIZER



def tag_chunk(work):
    """Tag one byte range of the train text and return the votes it produced.

    `work` is `(index, start_byte, end_byte)`; the worker reads its own slice, so
    no corpus text is ever pickled or held by the parent. Both boundaries fall on
    newline positions, which are single-byte in UTF-8, so slicing cannot split a
    character.

    Returns `(index, votes, stats)` where `votes` is `{(token_id, family): count}`
    and `stats` counts what happened to each word occurrence. The index is returned
    so the caller can split the corpus into halves for the convergence check
    without depending on completion order.

    A word contributes a vote only when (a) it encodes to exactly one id and (b)
    its tag maps to a real family. Both exclusions are counted, because "how many
    occurrences were unusable" is the number that says whether the vote had enough
    evidence.
    """
    idx, start, end = work
    tagger, tok = _handles()
    with open(_TEXT_PATH, "rb") as fh:
        fh.seek(start)
        text = fh.read(end - start).decode("utf-8")

    votes = collections.Counter()
    n_words = n_single = n_multi = n_tag_ignored = 0

    for line in text.splitlines():
        words = line.split()
        if not words:
            continue
        for j, (word, tag) in enumerate(tagger.tag(words)):
            n_words += 1
            # Leading space for every word but the first on the line: that is how
            # the byte-level BPE saw it in the stream, and " the" and "the" are
            # different types.
            ids = tok.encode((" " if j > 0 else "") + word,
                             add_special_tokens=False).ids
            if len(ids) != 1:
                n_multi += 1
                continue
            n_single += 1
            fam = penn_to_family(tag, word)
            if fam == NO_FAMILY_IDX:
                n_tag_ignored += 1
                continue
            votes[(ids[0], fam)] += 1

    stats = {"words": n_words, "single_id": n_single, "multi_piece": n_multi,
             "tag_ignored": n_tag_ignored}
    return idx, votes, stats


def chunk_offsets(path, chunk_lines=CHUNK_LINES):
    """`([(idx, start_byte, end_byte)], n_lines)` -- byte ranges on line boundaries.

    The parent keeps ~900 integer triples for a 539 MB corpus instead of the corpus.
    The first version of this builder did `list(iter_chunks(...))` and materialized
    1.8 M lines of wikitext-103 as Python strings, measured at ~1.0 GB resident
    before a single word was tagged; feeding a lazy generator to
    `imap_unordered` would not have fixed it either, because the pool's task
    handler drains its input as fast as it can and the queue is unbounded.
    """
    offsets = []
    n_lines = 0
    start = 0
    pos = 0
    lines_in_chunk = 0
    with open(path, "rb") as fh:
        while True:
            block = fh.read(1 << 22)
            if not block:
                break
            base = pos
            i = block.find(b"\n")
            while i != -1:
                n_lines += 1
                lines_in_chunk += 1
                if lines_in_chunk >= chunk_lines:
                    end = base + i + 1
                    offsets.append((len(offsets), start, end))
                    start = end
                    lines_in_chunk = 0
                i = block.find(b"\n", i + 1)
            pos += len(block)
    if start < pos:
        offsets.append((len(offsets), start, pos))
    return offsets, n_lines



def collect_votes(train_text, tokenizer_path, workers, progress_every=50):
    """Tag the whole train split and return `(votes, per_half_votes, stats)`.

    `per_half_votes` is the same tally split by whether the chunk index fell in the
    first or second half of the corpus. It costs one extra Counter and buys the
    convergence evidence in `vote_stability`: a majority that flips between halves
    is a majority the corpus did not actually decide. Halves rather than an
    interleaved split deliberately -- interleaving samples the same distribution
    twice and would converge trivially, hiding any drift across the corpus.
    """
    votes = collections.Counter()
    halves = [collections.Counter(), collections.Counter()]
    stats = collections.Counter()
    work, n_lines = chunk_offsets(train_text)
    n_chunks = len(work)
    mid = n_chunks / 2.0
    t0 = time.time()
    print(f"[vote] {n_lines:,} lines -> {n_chunks:,} byte ranges of "
          f"{CHUNK_LINES} lines", flush=True)

    def _absorb(result):
        idx, v, s = result
        votes.update(v)
        halves[0 if idx < mid else 1].update(v)
        stats.update(s)

    if workers <= 1:
        _worker_init(tokenizer_path, train_text)
        for i, chunk in enumerate(work):
            _absorb(tag_chunk(chunk))
            if progress_every and (i + 1) % progress_every == 0:
                _report(i + 1, n_chunks, stats, t0)
    else:
        with mp.Pool(workers, initializer=_worker_init,
                     initargs=(tokenizer_path, train_text)) as pool:
            for i, result in enumerate(
                    pool.imap_unordered(tag_chunk, work, chunksize=1)):
                _absorb(result)
                if progress_every and (i + 1) % progress_every == 0:
                    _report(i + 1, n_chunks, stats, t0)

    print(f"[vote] {stats['words']:,} words tagged in {time.time() - t0:.1f}s "
          f"({n_chunks:,} chunks, {workers} worker(s))", flush=True)
    return votes, halves, dict(stats)





def _report(done, total, stats, t0):
    el = time.time() - t0
    rate = stats["words"] / el if el else 0
    eta = (total - done) * el / done if done else 0
    print(f"  [vote] chunk {done:,}/{total:,}  {stats['words']:,} words  "
          f"{rate:,.0f} w/s  eta {eta / 60:.1f} min", flush=True)


def _by_type(votes, vocab_size):
    """`{token_id: {family: count}}` from the flat `(id, family)` tally."""
    per = collections.defaultdict(dict)
    for (tid, fam), n in votes.items():
        if 0 <= tid < vocab_size:
            per[tid][fam] = n
    return per


def majority(family_votes):
    """Winning family for one type. Ties go to the LOWEST family index.

    Deterministic by construction, so the parallel and serial paths cannot
    disagree: `max` on `(count, -family)` never consults insertion order, which a
    naive `max(d, key=d.get)` would.
    """
    return max(family_votes.items(), key=lambda kv: (kv[1], -kv[0]))[0]


def build_lookup(votes, tokenizer, vocab_size, eot_id):
    """Freeze `token_family[V]`, and report how each entry was decided.

    Three provenances, counted separately because they carry different weight:
    `voted` is corpus evidence, `surface` is the documented fallback of
    `plan_language.md` §4.1/§4.3, and `special` is the EOT token, which is not a
    linguistic type at all and is ignored rather than assigned.
    """
    per = _by_type(votes, vocab_size)
    table = np.full(vocab_size, NO_FAMILY_IDX, dtype=np.int8)
    provenance = {"voted": 0, "surface": 0, "special": 0, "unmapped": 0}
    surface_breakdown = collections.Counter()
    contested = 0

    for v in range(vocab_size):
        if v == eot_id:
            provenance["special"] += 1
            continue
        fv = per.get(v)
        if fv:
            table[v] = majority(fv)
            provenance["voted"] += 1
            if len(fv) > 1:
                contested += 1
            continue
        fam = surface_class_family(tokenizer.decode([v]))
        table[v] = fam
        provenance["surface"] += 1
        surface_breakdown[fam] += 1
        if fam == NO_FAMILY_IDX:
            provenance["unmapped"] += 1

    return table, provenance, dict(surface_breakdown), contested


def vote_stability(halves, vocab_size):
    """Would the first half of the corpus alone have given the same answer?

    Computed over the types that received votes in BOTH halves, since a type seen
    only in one half has nothing to disagree with. This is the convergence evidence
    the plan's "majority over the train split" claim needs: if the two halves
    disagree on a large share of types, the vote is reporting sampling noise, and a
    later rebuild on a different slice would produce a different oracle.
    """
    a, b = (_by_type(h, vocab_size) for h in halves)
    both = sorted(set(a) & set(b))
    if not both:
        return {"types_in_both_halves": 0, "agree": 0, "disagreement_rate": None}
    agree = sum(1 for v in both if majority(a[v]) == majority(b[v]))
    return {
        "types_in_both_halves": len(both),
        "agree": agree,
        "disagreement_rate": (len(both) - agree) / len(both),
    }


def token_level_counts(corpus, table, splits):
    """Family counts over ACTUAL corpus occurrences, per split.

    Type-level counts and token-level counts answer different questions and the
    plan requires both (T-L2.4): a family can be 2% of types and 40% of tokens, and
    a load-balance number read against the wrong denominator is uninterpretable.
    This is a `bincount` over the packed arrays -- no tagger, so it is exact over
    every stored token rather than over a sample.
    """
    out = {}
    for split in splits:
        arr = np.load(os.path.join(LANG, corpus, f"{split}.npy"), mmap_mode="r")
        counts = np.zeros(NUM_FAMILIES + 1, dtype=np.int64)   # +1 for the -1 bin
        # Chunked so a 269 MB uint16 array is never gathered into an int64 copy.
        rows = max(1, (1 << 22) // arr.shape[1])
        for i in range(0, arr.shape[0], rows):
            fam = table[np.asarray(arr[i:i + rows], dtype=np.int64)]
            counts += np.bincount(fam.ravel() + 1, minlength=NUM_FAMILIES + 1)
        total = int(counts.sum())
        out[split] = {
            "total_tokens": total,
            "unmapped": int(counts[0]),
            "unmapped_share": (int(counts[0]) / total) if total else None,
            "by_family": {LANG_FAMILY_LABELS[f]: int(counts[f + 1])
                          for f in range(NUM_FAMILIES)},
        }
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corpus", default=bld.CANONICAL_CORPUS,
                    choices=sorted(bld.CORPORA))
    ap.add_argument("--workers", type=int, default=0,
                    help="0 = cpu_count()-1; 1 = serial (the determinism "
                         "reference the parallel path is checked against)")
    ap.add_argument("--out-suffix", default="",
                    help="write token_family<suffix>.npy and skip the manifest "
                         "update; used by the workers=1-vs-N determinism check")
    args = ap.parse_args(argv)

    corpus = args.corpus
    cdir = bld.corpus_dir(corpus)
    man = bld.load_manifest(corpus)
    if man is None or "vocab_size" not in man:
        raise SystemExit(
            f"{corpus}: no tokenizer in the manifest. Run "
            f"build_language_dataset.py --corpus {corpus} --stage tokenizer first."
        )
    if "splits" not in man:
        raise SystemExit(
            f"{corpus}: not packed yet. Run build_language_dataset.py --corpus "
            f"{corpus} --stage pack first -- the token-level counts read the "
            f".npy files."
        )

    workers = args.workers or max(1, (os.cpu_count() or 2) - 1)
    tok_path = os.path.join(cdir, "tokenizer.json")
    train_text = os.path.join(cdir, "train_text.txt")
    if not os.path.exists(train_text):
        raise SystemExit(
            f"{train_text} is missing. It is a build intermediate and is not "
            f"tracked by git; re-run --stage tokenizer to materialize it."
        )

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(tok_path)
    V = man["vocab_size"]
    eot = man["eot_id"]

    print(f"[lookup] {corpus}: V={V} eot={eot} workers={workers}")
    votes, halves, vstats = collect_votes(train_text, tok_path, workers)
    table, prov, surf, contested = build_lookup(votes, tok, V, eot)
    stability = vote_stability(halves, V)
    tokens = token_level_counts(corpus, table, bld.SPLITS)

    out_path = os.path.join(cdir, f"token_family{args.out_suffix}.npy")
    np.save(out_path, table)
    sha = bld._sha256_file(out_path)

    type_counts = {LANG_FAMILY_LABELS[f]: int((table == f).sum())
                   for f in range(NUM_FAMILIES)}
    n_unmapped_types = int((table == NO_FAMILY_IDX).sum())

    print(f"[lookup] types: {prov['voted']:,} voted, {prov['surface']:,} by "
          f"surface class, {prov['special']:,} special, {n_unmapped_types:,} "
          f"unmapped ({100 * n_unmapped_types / V:.2f}% of V)")
    print(f"[lookup] contested types (>1 family seen): {contested:,} of "
          f"{prov['voted']:,} voted "
          f"({100 * contested / max(1, prov['voted']):.1f}%)")
    if stability["disagreement_rate"] is not None:
        print(f"[lookup] half-vs-half majority disagreement: "
              f"{stability['disagreement_rate']:.4%} over "
              f"{stability['types_in_both_halves']:,} types")
    for split in bld.SPLITS:
        t = tokens[split]
        print(f"[lookup] {split:5s} unmapped tokens {t['unmapped']:,} of "
              f"{t['total_tokens']:,} = {t['unmapped_share']:.4%}")
    print("[lookup] " + "  ".join(
        f"{lbl.split('_')[0]}={100 * tokens['train']['by_family'][lbl] / tokens['train']['total_tokens']:.1f}%"
        for lbl in LANG_FAMILY_LABELS))

    train_share = tokens["train"]["unmapped_share"]
    over_budget = train_share is not None and train_share > UNMAPPED_TOKEN_BUDGET

    if args.out_suffix:
        print(f"[lookup] suffix build -> {out_path} (manifest NOT updated)")
        return 0

    bld.update_manifest(corpus, {
        "family_manifest": "code/more/lang_families.py",
        "family_labels": list(LANG_FAMILY_LABELS),
        "token_family_file": f"data/lang/{corpus}/token_family.npy",
        "token_family_sha256": sha,
        "token_family_dtype": str(table.dtype),
        "token_family_len": int(table.shape[0]),
        "family_lookup_generator": "data/lang/build_family_lookup.py",
        "family_lookup_generator_commit": bld._git_commit(),
        "family_lookup_word_tokenization": "whitespace (WikiText is Moses-tokenized)",
        "family_lookup_tagger": "nltk averaged_perceptron_tagger_eng",
        "family_vote_stats": vstats,
        "family_vote_stability": stability,
        "family_type_provenance": prov,
        "family_surface_fallback_by_family": {
            (LANG_FAMILY_LABELS[f] if f >= 0 else "UNMAPPED"): int(n)
            for f, n in sorted(surf.items())
        },
        "family_contested_types": int(contested),
        "family_counts_types": type_counts,
        "family_counts_types_unmapped": n_unmapped_types,
        "family_counts_tokens": tokens,
        "unmapped_token_budget": UNMAPPED_TOKEN_BUDGET,
        "unmapped_token_share_train": train_share,
        "unmapped_within_budget": not over_budget,
    })
    print(f"[lookup] manifest -> {bld.manifest_path(corpus)}")

    if over_budget:
        # Gate L2 owns the verdict, but failing loudly here means a build that
        # blows the §4.3 budget cannot be mistaken for a good one just because the
        # file appeared. The artifact and the manifest are kept either way, so the
        # share can be inspected rather than re-derived.
        print(f"[lookup] FAIL: unmapped train-token share {train_share:.4%} "
              f"exceeds the §4.3 budget of {UNMAPPED_TOKEN_BUDGET:.0%}",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    mp.freeze_support()
    sys.exit(main())
