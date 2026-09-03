"""build_baseline_floors.py - T-L2.5: the numbers a language loss must beat.

Run:
    C:/Users/vedan/anaconda3/python.exe data/lang/build_baseline_floors.py --corpus wikitext-2

Writes `uniform_ce`, `unigram_ce`, `bigram_ce` and `primary_metric_floor` into
`dataset_meta.json`, all in nats per token, all fitted on TRAIN and evaluated on
VAL.

WHY THIS EXISTS. A validation loss of 5.4 nats/token is a number with no meaning on
its own. `plan_language.md` §3 requires the floors so that every later language
result can be stated as a distance from something trivial:

    uniform_ce = ln(8192) = 9.0109      knows nothing but the vocabulary size
    unigram_ce                          knows how often each type occurs
    bigram_ce                           knows the previous token

A model that does not beat `bigram_ce` has not learned anything a count-based table
could not, and on a 10-20 M-parameter budget over 134 M tokens that is a real
possibility rather than a formality -- so the floor is computed before the runs,
not after a disappointing number arrives.

TRAIN-ONLY, ASSERTED NOT INTENDED. Every count comes from `train.npy`; the val
array is opened exactly once, read-only, inside `evaluate_*`, and never during a
fit. `code/test_language_data.py` re-derives all three floors from the arrays
itself rather than reading the manifest, so a fit that had touched val would have
to reproduce that leak to pass.

THE BLOCK BOUNDARY IS NOT A SEQUENCE BOUNDARY. `pack_split` cuts one contiguous
token stream into fixed-length rows, so row i's last token and row i+1's first
token really were adjacent in the corpus. The bigram model therefore reads each
split as one flat stream via `.ravel()`, which is the same stream the tokenizer
produced. Treating rows as independent sequences would discard one bigram per
block and, worse, would not match how a trained model sees the data.

DISCOUNTING. Katz backoff in its original form uses Good-Turing discounted counts.
What is implemented here is the absolute-discounting variant of the same backoff
structure -- `max(c(v,w) - D, 0) / c(v)` for seen bigrams, with the leftover mass
distributed over the unigram -- and the manifest records
`bigram_smoothing = "backoff, absolute discounting D=0.75"` so nobody reads
"Katz" and assumes Good-Turing. The choice is not load-bearing: this is a floor,
and any reasonable discount puts it within a few hundredths of a nat.
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np

LANG = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, LANG)

import build_language_dataset as bld      # noqa: E402

# Absolute-discounting constant. 0.75 is the conventional default and is recorded
# in the manifest so the floor is reproducible; see the module docstring on why the
# exact value does not matter for a floor.
DISCOUNT = 0.75

# Rows per read. 4 Mi tokens at seq_len 256 is 16 k rows, ~8 MB of uint16 -- small
# enough that the int64 index arrays the counting needs stay under 100 MB on a
# 526 k-row corpus.
CHUNK_TOKENS = 1 << 22


def _split_path(corpus, split):
    return os.path.join(bld.corpus_dir(corpus), f"{split}.npy")


def iter_flat(corpus, split, chunk_tokens=CHUNK_TOKENS):
    """Yield contiguous int64 slices of one split, as ONE flat token stream.

    The `mmap_mode="r"` handle means a 269 MB train split is never resident. Each
    yielded slice overlaps the previous one by a single token, so a bigram spanning
    a chunk boundary is counted exactly once by the caller (which skips the first
    element of every slice after the first).
    """
    arr = np.load(_split_path(corpus, split), mmap_mode="r")
    n_rows, seq_len = arr.shape
    rows = max(1, chunk_tokens // seq_len)
    carry = None
    for i in range(0, n_rows, rows):
        flat = np.asarray(arr[i:i + rows], dtype=np.int64).ravel()
        if carry is None:
            yield flat, 0
        else:
            yield np.concatenate(([carry], flat)), 1
        carry = int(flat[-1])


def fit_unigram(corpus, vocab_size):
    """Train-split token counts, and the CE-ready log probabilities.

    Add-1 only if some type is unseen -- on 134 M tokens over 8192 types that does
    not happen, and smoothing a distribution with no zeros would loosen the floor
    for nothing. Which branch ran is recorded, because "unigram CE" means two
    different numbers depending on it.
    """
    counts = np.zeros(vocab_size, dtype=np.int64)
    for flat, skip in iter_flat(corpus, "train"):
        counts += np.bincount(flat[skip:] if skip else flat,
                              minlength=vocab_size)
    n_zero = int((counts == 0).sum())
    smoothing = "none (MLE)" if n_zero == 0 else "add-1 (Laplace)"
    c = counts.astype(np.float64) + (0.0 if n_zero == 0 else 1.0)
    return counts, c / c.sum(), smoothing, n_zero


def fit_bigram(corpus, vocab_size):
    """Sparse train bigram counts as CSR, plus the per-context row totals.

    CSR rather than a dense `(8192, 8192)` array: dense int32 is 268 MB and mostly
    zeros, and the val evaluation only ever asks for the contexts it actually
    meets. Accumulated per chunk and summed, so peak memory is one chunk's COO plus
    the running CSR rather than both corpora at once.
    """
    from scipy import sparse
    total = sparse.csr_matrix((vocab_size, vocab_size), dtype=np.int64)
    n_bigrams = 0
    t0 = time.time()
    for flat, skip in iter_flat(corpus, "train"):
        prev, cur = flat[:-1], flat[1:]
        n_bigrams += prev.size
        chunk = sparse.coo_matrix(
            (np.ones(prev.size, dtype=np.int64), (prev, cur)),
            shape=(vocab_size, vocab_size),
        ).tocsr()
        total = total + chunk
        del chunk
    total.sum_duplicates()
    row_total = np.asarray(total.sum(axis=1)).ravel().astype(np.float64)
    row_types = np.diff(total.indptr).astype(np.float64)
    print(f"[floors] bigram: {n_bigrams:,} bigrams, {total.nnz:,} distinct "
          f"contexts-continuations in {time.time() - t0:.1f}s")
    return total, row_total, row_types, n_bigrams


def evaluate_uniform(vocab_size, n_tokens):
    """CE of the distribution that knows only V. Exactly ln(V), by construction."""
    return math.log(vocab_size)


def evaluate_unigram(corpus, split, uni_p):
    """Mean negative log unigram probability over every token of `split`."""
    logp = np.log(uni_p)
    total, n = 0.0, 0
    for flat, skip in iter_flat(corpus, split):
        tokens = flat[skip:] if skip else flat
        total += float(logp[tokens].sum())
        n += tokens.size
    return -total / n, n


def evaluate_bigram(corpus, split, bg, row_total, row_types, uni_p):
    """CE of the backoff bigram, streamed over `split`.

        seen (v, w):    P = (c(v,w) - D) / c(v)  +  lambda(v) * P_uni(w)
        unseen (v, w):  P =                         lambda(v) * P_uni(w)
        lambda(v)    =  D * types_after(v) / c(v)

    The interpolated form is used rather than strict Katz renormalization over the
    unseen set: it is the same backoff structure, it cannot divide by a zero
    unseen-mass, and for a floor the difference is far below the precision anyone
    quotes. `lambda` is exactly the discount mass removed from row `v`, so each row
    sums to 1 -- checked in `main`.

    The FIRST token of the split has no predecessor and is scored by the unigram.
    That is one token in ~281 k, and it is stated rather than dropped so the token
    count matches `evaluate_unigram`'s.
    """
    total, n = 0.0, 0
    first = True
    for flat, skip in iter_flat(corpus, split):
        if first:
            total += float(np.log(uni_p[flat[0]]))
            n += 1
            first = False
        prev, cur = flat[:-1], flat[1:]
        ctx_total = row_total[prev]
        lam = np.where(ctx_total > 0, DISCOUNT * row_types[prev] / np.maximum(ctx_total, 1.0), 1.0)
        seen = np.asarray(bg[prev, cur]).ravel().astype(np.float64)
        disc = np.where(ctx_total > 0,
                        np.maximum(seen - DISCOUNT, 0.0) / np.maximum(ctx_total, 1.0),
                        0.0)
        p = disc + lam * uni_p[cur]
        total += float(np.log(p).sum())
        n += cur.size
    return -total / n, n


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corpus", default=bld.CANONICAL_CORPUS,
                    choices=sorted(bld.CORPORA))
    ap.add_argument("--eval-split", default="val", choices=["val", "test"],
                    help="val is canonical; test exists only so the floor can be "
                         "quoted once at the end of the study")
    args = ap.parse_args(argv)

    corpus, split = args.corpus, args.eval_split
    man = bld.load_manifest(corpus)
    if man is None or "splits" not in man:
        raise SystemExit(f"{corpus}: not packed yet. Run build_language_dataset.py "
                         f"--corpus {corpus} --stage pack first.")
    V = man["vocab_size"]

    t0 = time.time()
    counts, uni_p, uni_smoothing, n_zero = fit_unigram(corpus, V)
    print(f"[floors] unigram fitted on train in {time.time() - t0:.1f}s "
          f"({int(counts.sum()):,} tokens, {n_zero} unseen types, "
          f"smoothing: {uni_smoothing})")

    bg, row_total, row_types, n_bigrams = fit_bigram(corpus, V)

    uniform_ce = evaluate_uniform(V, None)
    unigram_ce, n_uni = evaluate_unigram(corpus, split, uni_p)
    bigram_ce, n_big = evaluate_bigram(corpus, split, bg, row_total, row_types,
                                       uni_p)

    floor = min(uniform_ce, unigram_ce, bigram_ce)
    ordered = uniform_ce >= unigram_ce >= bigram_ce

    print(f"[floors] evaluated on {split} ({n_uni:,} tokens)")
    print(f"[floors]   uniform_ce = {uniform_ce:.4f} nats  "
          f"({uniform_ce / math.log(2):.4f} bits)")
    print(f"[floors]   unigram_ce = {unigram_ce:.4f} nats  "
          f"({unigram_ce / math.log(2):.4f} bits)")
    print(f"[floors]   bigram_ce  = {bigram_ce:.4f} nats  "
          f"({bigram_ce / math.log(2):.4f} bits)")
    print(f"[floors]   primary_metric_floor = {floor:.4f} nats   "
          f"ordered uniform>=unigram>=bigram: {ordered}")

    bld.update_manifest(corpus, {
        "floors_generator": "data/lang/build_baseline_floors.py",
        "floors_generator_commit": bld._git_commit(),
        "floors_eval_split": split,
        "floors_eval_tokens": int(n_uni),
        "floors_fitted_on": "train",
        "uniform_ce": uniform_ce,
        "unigram_ce": unigram_ce,
        "bigram_ce": bigram_ce,
        "primary_metric_floor": floor,
        "floors_ordered_uniform_ge_unigram_ge_bigram": bool(ordered),
        "unigram_smoothing": uni_smoothing,
        "unigram_unseen_types": int(n_zero),
        "bigram_smoothing": f"backoff, absolute discounting D={DISCOUNT}",
        "bigram_train_bigrams": int(n_bigrams),
        "bigram_distinct_pairs": int(bg.nnz),
        "floors_units": "nats per token",
    })
    print(f"[floors] manifest -> {bld.manifest_path(corpus)}")

    if not ordered:
        # A floor that is not ordered is a broken floor: each model strictly
        # refines the one above it, so an inversion means a fitting or evaluation
        # bug, not a surprising corpus. Gate L2 owns the verdict; failing here
        # means the number cannot be quoted by accident in the meantime.
        print("[floors] FAIL: floors are not ordered uniform >= unigram >= bigram",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
