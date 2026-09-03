"""build_depth_deciles.py - T-L2.6: the frequency-decile table, built and UNUSED.

Run:
    C:/Users/vedan/anaconda3/python.exe data/lang/build_depth_deciles.py --corpus wikitext-2

Writes `data/lang/<corpus>/token_decile.npy` -- an `int8` array of length V giving
each vocabulary id a train-frequency decile 0..9 -- and the statistics behind it to
`dataset_meta.json`.

THIS TABLE IS DELIBERATELY NOT PART OF THE CANONICAL CONFIGURATION.
`plan_language.md` §5 is explicit: canonical language runs invent no depth target.
Arithmetic could defend one, because a naive sequential decomposition gives a real
number of sub-steps per operation; nothing gives a principled number of recursion
steps for predicting the next token. The table exists so ablation F
(`supervised_curriculum`) is runnable and comparable with its arithmetic
counterpart, and for no other reason. Canonical language runs keep
`loss_weights.halting_supervision = 0.0` and pure-ACT halting.

MEASUREMENT AND CHOICE ARE KEPT IN DIFFERENT PLACES, on purpose. This file produces
`token_decile[V]` -- which decile each id falls in, a measured property of the train
split. The decile-to-depth mapping is a design choice and lives in
`code/more/lang_families.py:decile_target_depth`, which also takes `max_depth` as an
argument so this artifact does not hard-code a recursion budget it cannot know at
dataset-build time.

DECILES BY TOKEN MASS, NOT BY TYPE COUNT. Equal-type-count deciles would put ~90% of
corpus occurrences in the single most-frequent decile, because the vocabulary is
Zipfian: the resulting "depth curriculum" would assign one depth to almost every
token actually seen. Cutting at equal token mass instead gives ten deciles each
carrying ~10% of the corpus, so the supervision signal is spread across the data the
loss actually weights. The realised mass per decile is written to the manifest,
because integer type counts cannot hit exactly 10% and the deviation should be
visible rather than assumed away.

THE CONFOUND, MEASURED. `families.OP_TARGET_DEPTH` was built to be correlated with
but not a function of the expert index, so a model could not score well on depth by
routing alone. A frequency decile has no such protection: function words are
frequent and subword pieces are rare, so the decile is substantially predictable
from the POS family. The mutual information between decile and family is computed
here and stored as `decile_family_mutual_information` (normalized), so ablation F's
depth numbers can be read against it instead of as independent evidence.
"""

import argparse
import json
import math
import os
import sys

import numpy as np

LANG = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(LANG))
sys.path.insert(0, LANG)

import build_language_dataset as bld      # noqa: E402


def _load_manifest_module():
    """Load `lang_families.py` by path -- see `build_family_lookup.py` for why."""
    import importlib.util
    path = os.path.join(REPO, "code", "more", "lang_families.py")
    spec = importlib.util.spec_from_file_location("_lang_families", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_LF = _load_manifest_module()
N_DECILES = _LF.N_FREQUENCY_DECILES
NUM_FAMILIES = _LF.NUM_FAMILIES
LANG_FAMILY_LABELS = _LF.LANG_FAMILY_LABELS


def train_counts(corpus, vocab_size):
    """Per-type occurrence counts over the packed train split."""
    arr = np.load(os.path.join(bld.corpus_dir(corpus), "train.npy"), mmap_mode="r")
    counts = np.zeros(vocab_size, dtype=np.int64)
    rows = max(1, (1 << 22) // arr.shape[1])
    for i in range(0, arr.shape[0], rows):
        counts += np.bincount(
            np.asarray(arr[i:i + rows], dtype=np.int64).ravel(),
            minlength=vocab_size)
    return counts


def assign_deciles(counts, n_deciles=N_DECILES):
    """`token_decile[V]` cut at equal TOKEN MASS, most frequent first.

    Types are ordered by descending count, with the id as a tie-break so the result
    is deterministic for the many types that share a count. The cumulative mass then
    decides the boundaries: an id joins decile `d` when the mass before it has passed
    `d / n_deciles` of the total.

    Unseen types (count 0) all land in the last decile, which is correct in the only
    sense available -- they are the rarest thing in the corpus -- and is recorded
    separately so the number is never mistaken for a measurement of them.
    """
    order = np.lexsort((np.arange(counts.size), -counts))
    sorted_counts = counts[order]
    cum = np.cumsum(sorted_counts)
    total = int(cum[-1])
    # Position of each sorted entry as a fraction of total mass BEFORE it.
    before = (cum - sorted_counts) / total
    dec_sorted = np.minimum((before * n_deciles).astype(np.int64), n_deciles - 1)
    deciles = np.empty(counts.size, dtype=np.int8)
    deciles[order] = dec_sorted.astype(np.int8)
    return deciles, total


def decile_family_mi(deciles, families, counts):
    """Normalized mutual information between decile and POS family, token-weighted.

    Token-weighted rather than type-weighted because the question is about the
    supervision signal the loss actually sees. Unmapped types (`family = -1`) are
    excluded: they carry no family, so including them would measure the ignore
    label's frequency profile instead.

    Returned as `I(D;F) / min(H(D), H(F))`, so 0 means the decile says nothing about
    the family and 1 means one determines the other. This is the number ablation F's
    depth results have to be read against -- a high value means "allocated depth
    correctly" and "routed correctly" are close to the same claim.
    """
    mask = families >= 0
    w = counts[mask].astype(np.float64)
    if w.sum() == 0:
        return None
    d = deciles[mask].astype(np.int64)
    f = families[mask].astype(np.int64)
    joint = np.zeros((N_DECILES, NUM_FAMILIES), dtype=np.float64)
    np.add.at(joint, (d, f), w)
    joint /= joint.sum()
    pd = joint.sum(axis=1)
    pf = joint.sum(axis=0)

    def _H(p):
        p = p[p > 0]
        return float(-(p * np.log(p)).sum())

    nz = joint > 0
    mi = float((joint[nz] * np.log(joint[nz]
                                   / (pd[:, None] * pf[None, :])[nz])).sum())
    denom = min(_H(pd), _H(pf))
    return {
        "mutual_information_nats": mi,
        "entropy_decile_nats": _H(pd),
        "entropy_family_nats": _H(pf),
        "normalized": (mi / denom) if denom > 0 else None,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corpus", default=bld.CANONICAL_CORPUS,
                    choices=sorted(bld.CORPORA))
    args = ap.parse_args(argv)

    corpus = args.corpus
    man = bld.load_manifest(corpus)
    if man is None or "splits" not in man:
        raise SystemExit(f"{corpus}: not packed yet. Run build_language_dataset.py "
                         f"--corpus {corpus} --stage pack first.")
    V = man["vocab_size"]

    counts = train_counts(corpus, V)
    deciles, total = assign_deciles(counts)
    n_unseen = int((counts == 0).sum())

    out = os.path.join(bld.corpus_dir(corpus), "token_decile.npy")
    np.save(out, deciles)
    sha = bld._sha256_file(out)

    # T-L6.1: the raw per-type train counts, frozen as their own artifact.
    # `depth/spearman_vs_logfreq` and `depth/spearman_vs_unigram_surprisal` are
    # non-circular only because both are properties of the corpus fixed BEFORE any
    # model runs (`plan_language.md` §5.2). Storing the counts rather than the two
    # derived vectors keeps one source of truth: `lang_data` derives log-frequency
    # and surprisal from this array, so they cannot disagree about the smoothing.
    counts_path = os.path.join(bld.corpus_dir(corpus), "token_train_count.npy")
    np.save(counts_path, counts)
    counts_sha = bld._sha256_file(counts_path)

    per = []
    for d in range(N_DECILES):
        sel = deciles == d
        per.append({
            "decile": d,
            "types": int(sel.sum()),
            "tokens": int(counts[sel].sum()),
            "token_share": float(counts[sel].sum() / total),
            "min_count": int(counts[sel].min()) if sel.any() else None,
            "max_count": int(counts[sel].max()) if sel.any() else None,
        })

    print(f"[decile] {corpus}: V={V} train tokens {total:,} "
          f"({n_unseen} types unseen in train)")
    print(f"[decile] {'d':>2s} {'types':>7s} {'tokens':>14s} {'share':>7s} "
          f"{'count range':>20s}")
    for p in per:
        print(f"[decile] {p['decile']:2d} {p['types']:7,d} {p['tokens']:14,d} "
              f"{p['token_share']:6.2%} "
              f"{(str(p['min_count']) + '..' + str(p['max_count'])):>20s}")

    mi = None
    fam_path = os.path.join(bld.corpus_dir(corpus), "token_family.npy")
    if os.path.exists(fam_path):
        fams = np.load(fam_path)
        mi = decile_family_mi(deciles, fams, counts)
        if mi and mi["normalized"] is not None:
            print(f"[decile] decile-vs-family normalized MI = "
                  f"{mi['normalized']:.4f}  (I={mi['mutual_information_nats']:.4f} "
                  f"nats, H(D)={mi['entropy_decile_nats']:.4f}, "
                  f"H(F)={mi['entropy_family_nats']:.4f})")
    else:
        print("[decile] token_family.npy absent -- MI against the POS partition "
              "not computed. Re-run after build_family_lookup.py to record the "
              "ablation-F confound.")

    bld.update_manifest(corpus, {
        "decile_generator": "data/lang/build_depth_deciles.py",
        "decile_generator_commit": bld._git_commit(),
        "token_decile_file": f"data/lang/{corpus}/token_decile.npy",
        "token_decile_sha256": sha,
        "token_train_count_file": f"data/lang/{corpus}/token_train_count.npy",
        "token_train_count_sha256": counts_sha,
        "token_train_count_total": int(counts.sum()),
        "token_decile_dtype": str(deciles.dtype),
        "token_decile_len": int(deciles.shape[0]),
        "n_frequency_deciles": N_DECILES,
        "decile_cut": "equal train TOKEN mass, most frequent = decile 0",
        "decile_table": per,
        "decile_types_unseen_in_train": n_unseen,
        "decile_family_mutual_information": mi,
        "decile_is_canonical": False,
        "decile_purpose": (
            "ablation F (supervised_curriculum) only. plan_language.md §5: "
            "canonical language runs invent no depth target, so "
            "loss_weights.halting_supervision stays 0.0 and halting stays pure ACT. "
            "The decile->depth mapping is a DESIGN CHOICE and lives in "
            "code/more/lang_families.py:decile_target_depth, not in this artifact."
        ),
    })
    print(f"[decile] manifest -> {bld.manifest_path(corpus)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
