"""build_block_topics.py - T-L6.5: an INDUCED topic partition, as a second axis.

Run:
    C:/Users/vedan/anaconda3/python.exe data/lang/build_block_topics.py --corpus wikitext-2

Writes `data/lang/<corpus>/block_topic_<split>.npy` -- one `int8` topic id per stored
block -- and the cluster statistics into `dataset_meta.json`.

WHY THIS EXISTS. `plan_language.md` §4.1a. A production MoE has experts that look like
*law / science / math / code*, and the fair question is whether a small recursive MoE
organizes by SUBJECT rather than by lexical class. After T-L3.2 and T-L5.3 the oracle
partition is a read-only measurement reference -- `family_probe` reads `h.detach()`, and
trunk gradients are bitwise identical across probe weights -- so adding a second
reference costs one build script and changes nothing the model sees. It turns the question
from "does the router reproduce POS?" into "**what, if anything, does it organize by?**",
which is the better question and the one a reader will ask.

**INDUCED, NOT GROUND TRUTH, and that is the headline caveat.** WikiText ships no topic
labels. These clusters are k-means over TF-IDF document vectors -- OUR construction, with
our k and our seed -- and they are reported exactly like the shuffled control: a reference
partition with its own null, never as truth. A cluster's "name" is whatever its top terms
suggest, assigned by a human reading the printout, and is not evidence of anything.

k = 6, NOT 4 OR 8. Six so the topic axis is directly comparable with the POS axis and with
the arithmetic study's router width; a different k would confound "organizes by topic"
with "has a different number of things to organize into".

FITTED ON TRAIN ONLY, like everything else here. The vectorizer and the k-means are fit on
the train documents and then *applied* to val and test. Fitting the topic model on the
split being scored would make the reference partition depend on the data it is used to
evaluate -- the same defect as a tokenizer fitted on validation text (T-L2.1).

DOCUMENT BOUNDARIES COME FROM THE STORED STREAM, not from a re-segmentation.
`_encode_stream_to_bin` inserts `eot_id` BETWEEN documents and not after the last, so the
document index of a token is the number of EOTs before it. Verified on both corpora: the
packed train split contains exactly `train_documents - 1` EOT tokens (629 of 630;
29,444 of 29,445). A block then takes the topic of the document supplying most of its
tokens, because a 256-token block can straddle a boundary.
"""

import argparse
import collections
import os
import sys

import numpy as np

LANG = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, LANG)

import build_language_dataset as bld      # noqa: E402

# Six, matching NUM_FAMILIES and the canonical router width. See the module docstring.
N_TOPICS = 6

# A data artifact, so its seed is deliberately NOT one of the frozen run seeds (42..46) --
# tying a reference partition to a training seed would invite "which arm's topics are
# these". Same constant family as the shuffled control's 20260903.
TOPIC_SEED = 20260904

# TF-IDF settings, recorded in the manifest because they are choices, not measurements.
# `min_df=5` drops terms appearing in fewer than five documents (mostly OCR noise and
# one-off proper nouns); `max_df=0.5` drops terms in more than half the corpus, which on
# Wikipedia is exactly the boilerplate that makes every article look alike.
MAX_FEATURES = 20000
MIN_DF = 5
MAX_DF = 0.5

# LSA before k-means, and this was added AFTER a measured failure -- recorded that way
# on purpose, because "changed the method after seeing the result" is a pattern that
# needs its reasoning in the open.
#
# WHAT FAILED. TF-IDF -> k-means directly, k=6, on wikitext-103's 29,445 documents put
# **64.2% of train blocks in a single cluster** whose top terms were
# `species, war, century, army, king, race, south, city, church, british` -- a residual
# "everything else" bucket rather than a topic -- and left T3 (highways) at 1.72%, below
# T-L6.5's pre-registered 2% floor. A reference axis whose majority class is two thirds
# of the corpus is not a reference axis: a router "agreeing" with it would mostly be
# agreeing with "is this the majority class".
#
# WHY THIS IS A FIX AND NOT TUNING. The 2% criterion was pre-registered in the ledger
# BEFORE the build, the criterion is unchanged, and no routing number exists yet -- there
# is nothing downstream to select on. TF-IDF -> TruncatedSVD -> k-means is also the
# standard text-clustering pipeline (it is sklearn's own documented recipe): k-means uses
# Euclidean distance, which concentrates badly in 20,000 sparse dimensions, and reducing
# to a dense low-rank space is the ordinary remedy rather than a trick. 100 components is
# the conventional LSA width; seeded, because an unseeded SVD would break the
# same-seed-reproduces clause.
SVD_COMPONENTS = 100


def doc_iter(corpus, split, cache_dir=None):
    """Yield document strings for one split, ONE AT A TIME.

    A generator rather than a list: wikitext-103's train split is 539 MB of text and
    holding it as ~29 k Python strings is over a gigabyte, which is the same trap
    `build_family_lookup.chunk_offsets` exists to avoid. `TfidfVectorizer.fit_transform`
    consumes an iterable in a single pass, so streaming works.
    """
    splits = bld.fetch_splits(corpus, cache_dir)
    for doc in bld.iter_documents(splits[split]):
        yield doc


def block_doc_index(corpus, split, eot_id, seq_len):
    """`block_doc[n_blocks]` -- which document supplies most of each block's tokens.

    The document index of a token is the number of `eot_id` tokens before it, because the
    encoder inserts the separator BETWEEN documents only. A 256-token block can straddle a
    boundary, so the block takes the majority document rather than its first.
    """
    arr = np.load(os.path.join(bld.corpus_dir(corpus), f"{split}.npy"), mmap_mode="r")
    n_blocks = arr.shape[0]
    out = np.empty(n_blocks, dtype=np.int64)
    seen = 0
    rows = max(1, (1 << 22) // seq_len)
    for i in range(0, n_blocks, rows):
        chunk = np.asarray(arr[i:i + rows], dtype=np.int64)
        # Running document index for every token position in this chunk.
        is_eot = (chunk == eot_id).ravel()
        doc_of_token = (seen + np.cumsum(is_eot) - is_eot).reshape(chunk.shape)
        seen += int(is_eot.sum())
        for j in range(chunk.shape[0]):
            vals, counts = np.unique(doc_of_token[j], return_counts=True)
            out[i + j] = int(vals[int(np.argmax(counts))])
    return out, seen + 1


def fit_topics(corpus, cache_dir=None):
    """Fit TF-IDF -> LSA -> k-means on TRAIN documents. Returns the fitted pipeline.

    Top terms are recovered by projecting each k-means centroid back through the SVD
    components (`centroid @ svd.components_`), so they are still the TF-IDF vocabulary a
    human can read -- the reduction changes the geometry k-means works in, not what a
    cluster is described by.
    """
    from sklearn.cluster import KMeans
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import Normalizer

    vec = TfidfVectorizer(max_features=MAX_FEATURES, min_df=MIN_DF, max_df=MAX_DF,
                          stop_words="english", lowercase=True,
                          strip_accents="unicode")
    X = vec.fit_transform(doc_iter(corpus, "train", cache_dir))
    n_comp = min(SVD_COMPONENTS, X.shape[1] - 1, max(2, X.shape[0] - 1))
    svd = TruncatedSVD(n_components=n_comp, random_state=TOPIC_SEED)
    nrm = Normalizer(copy=False)
    Z = nrm.fit_transform(svd.fit_transform(X))
    # `n_init=10` and an explicit `random_state` are both required for the T-L6.5
    # reproducibility clause: k-means is seeded, and leaving n_init to the sklearn
    # default would make the result depend on the installed version.
    km = KMeans(n_clusters=N_TOPICS, random_state=TOPIC_SEED, n_init=10)
    labels = km.fit_predict(Z)
    terms = np.array(vec.get_feature_names_out())
    centroids_tfidf = km.cluster_centers_ @ svd.components_
    top = {}
    for c in range(N_TOPICS):
        order = np.argsort(centroids_tfidf[c])[::-1][:12]
        top[c] = [str(t) for t in terms[order]]
    explained = float(svd.explained_variance_ratio_.sum())
    return (vec, svd, nrm, km), labels, top, X.shape, n_comp, explained


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corpus", default=bld.CANONICAL_CORPUS,
                    choices=sorted(bld.CORPORA))
    ap.add_argument("--cache_dir", default=None)
    ap.add_argument("--out-suffix", default="",
                    help="write block_topic<suffix>_<split>.npy and skip the manifest; "
                         "used by the same-seed reproducibility check")
    args = ap.parse_args(argv)

    corpus = args.corpus
    man = bld.load_manifest(corpus)
    if man is None or "splits" not in man:
        raise SystemExit(f"{corpus}: not packed yet. Run build_language_dataset.py "
                         f"--corpus {corpus} --stage pack first.")
    eot, seq_len = man["eot_id"], man["seq_len"]

    (vec, svd, nrm, km), train_labels, top, shape, n_comp, explained = fit_topics(
        corpus, args.cache_dir)
    print(f"[topic] {corpus}: TF-IDF {shape[0]:,} train docs x {shape[1]:,} terms "
          f"-> LSA {n_comp} components ({explained:.1%} variance) -> k-means "
          f"k={N_TOPICS}, seed={TOPIC_SEED}")
    for c in range(N_TOPICS):
        print(f"[topic] T{c}  n_docs={int((train_labels == c).sum()):>6,d}  "
              f"top: {', '.join(top[c][:10])}")

    per_split, block_counts = {}, {}
    for split in bld.SPLITS:
        if split == "train":
            doc_topic = train_labels
        else:
            # TRANSFORM, not fit: the reference partition must not depend on the split
            # it is later used to score.
            doc_topic = km.predict(nrm.transform(svd.transform(
                vec.transform(doc_iter(corpus, split, args.cache_dir)))))
        bdoc, n_docs = block_doc_index(corpus, split, eot, seq_len)
        if bdoc.size and int(bdoc.max()) >= doc_topic.size:
            raise SystemExit(
                f"{corpus}/{split}: block_doc references document "
                f"{int(bdoc.max())} but only {doc_topic.size} documents were "
                f"segmented. The EOT reconstruction and iter_documents disagree, "
                f"which means one of them changed -- do not paper over this."
            )
        bt = doc_topic[bdoc].astype(np.int8)
        out = os.path.join(bld.corpus_dir(corpus),
                           f"block_topic{args.out_suffix}_{split}.npy")
        np.save(out, bt)
        counts = {str(c): int((bt == c).sum()) for c in range(N_TOPICS)}
        share = {c: n / max(1, bt.size) for c, n in counts.items()}
        per_split[split] = {
            "blocks": int(bt.size), "documents": int(doc_topic.size),
            "sha256": bld._sha256_file(out),
            "blocks_by_topic": counts,
            "block_share_by_topic": share,
            "min_topic_share": float(min(share.values())) if share else None,
        }
        block_counts[split] = counts
        print(f"[topic] {split:5s} {bt.size:>7,d} blocks over {doc_topic.size:>6,d} "
              f"docs | shares " + " ".join(f"T{c}={100 * s:.1f}%"
                                           for c, s in sorted(share.items())))

    # NON-DEGENERACY IS A PROPERTY OF THE FITTED PARTITION, so it is checked on TRAIN.
    # Measured reason, not a convenience: wikitext-2's val split contains only 61
    # documents and test 63 -- and wikitext-103 shares those splits BYTE-FOR-BYTE
    # (T-L2.0), so the canonical corpus has exactly the same 61. With k = 6, one small
    # cluster contributing a single val document is 1.6% of documents and ~1.1% of
    # blocks, so a 2% floor applied to val could never pass for this corpus family no
    # matter how good the clustering is. The question the floor is asking -- did k-means
    # produce six usable groups -- is answered by the split it was fitted on.
    #
    # The val/test shares are printed and stored anyway, because their thinness is a real
    # limitation of T-L6.6: article-level agreement on val is computed over 61 documents.
    train_worst = per_split["train"]["min_topic_share"]
    worst_any = min(v["min_topic_share"] for v in per_split.values())
    degenerate = train_worst < 0.02
    print(f"[topic] smallest TRAIN topic share: {100 * train_worst:.2f}% "
          f"({'DEGENERATE, below the 2% floor' if degenerate else 'above the 2% floor'})")
    print(f"[topic] smallest share on ANY split: {100 * worst_any:.2f}% -- val/test hold "
          f"only {per_split['val']['documents']}/{per_split['test']['documents']} "
          f"documents, so a small cluster is thin there by sample size, not by defect")

    if args.out_suffix:
        print(f"[topic] suffix build (manifest NOT updated)")
        return 1 if degenerate else 0

    bld.update_manifest(corpus, {
        "block_topic_generator": "data/lang/build_block_topics.py",
        "block_topic_generator_commit": bld._git_commit(),
        "block_topic_files": {s: f"data/lang/{corpus}/block_topic_{s}.npy"
                              for s in bld.SPLITS},
        "n_topics": N_TOPICS,
        "topic_seed": TOPIC_SEED,
        "topic_is_induced": True,
        "topic_note": (
            "INDUCED, NOT GROUND TRUTH. k-means over TF-IDF document vectors -- our k, "
            "our seed, our vectorizer settings. WikiText ships no topic labels. Reported "
            "beside the POS axis and against its own shuffled null (T-L6.6), never as "
            "truth. Cluster names are whatever a human reads off the top terms and are "
            "not evidence. Fitted on TRAIN documents and APPLIED to val/test, so the "
            "reference partition does not depend on the split it scores."
        ),
        "topic_vectorizer": {
            "kind": "TfidfVectorizer", "max_features": MAX_FEATURES,
            "min_df": MIN_DF, "max_df": MAX_DF, "stop_words": "english",
            "strip_accents": "unicode", "n_terms_fitted": int(shape[1]),
        },
        "topic_lsa": {"kind": "TruncatedSVD", "n_components": int(n_comp),
                      "explained_variance_ratio": explained,
                      "random_state": TOPIC_SEED, "then": "L2 Normalizer",
                      "why": ("added after a MEASURED failure: TF-IDF -> k-means "
                              "directly put 64.2% of wikitext-103 train blocks in one "
                              "residual cluster and left a topic at 1.72%, below the "
                              "pre-registered 2% floor. k-means uses Euclidean distance, "
                              "which concentrates badly in 20,000 sparse dimensions; "
                              "LSA first is the standard remedy, not a trick. The "
                              "criterion was not changed and no routing number existed "
                              "yet, so there was nothing downstream to select on.")},
        "topic_kmeans": {"n_clusters": N_TOPICS, "n_init": 10,
                         "random_state": TOPIC_SEED, "fitted_on": "train documents"},
        "topic_top_terms": {str(c): top[c] for c in range(N_TOPICS)},
        "block_topic_by_split": per_split,
        "topic_min_share_train": float(train_worst),
        "topic_min_share_any_split": float(worst_any),
        "topic_non_degenerate": not degenerate,
        "topic_non_degeneracy_note": (
            "The 2% floor is applied to the TRAIN split, which is where the clustering "
            "is fitted and where 'six usable groups' is a property of the partition. "
            "val/test hold only 61/63 documents and wikitext-103 shares them "
            "byte-for-byte with wikitext-2, so one small cluster contributing a single "
            "val document is ~1.1% of val blocks and a 2% floor there could never pass "
            "for this corpus family. Their shares are stored above because that "
            "thinness is a real limitation of the article-level agreement in T-L6.6."
        ),
        "block_topic_document_assignment": (
            "majority document per block; document index = number of eot_id tokens "
            "before the token, which is exact because the encoder inserts the "
            "separator between documents only (verified: packed train holds "
            "train_documents - 1 EOTs on both corpora)"
        ),
    })
    print(f"[topic] manifest -> {bld.manifest_path(corpus)}")
    return 1 if degenerate else 0


if __name__ == "__main__":
    sys.exit(main())
