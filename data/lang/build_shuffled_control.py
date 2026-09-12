"""build_shuffled_control.py - T-L3.3 / ablation G: the null the POS number needs.

Run:
    C:/Users/vedan/anaconda3/python.exe data/lang/build_shuffled_control.py --corpus wikitext-103

Writes `data/lang/<corpus>/token_family_shuffled.npy` -- an `int8` array of shape
`(n_draws, V)` -- and its marginal statistics into `dataset_meta.json`.

WHY THIS IS MANDATORY AND NOT AN EXTRA. `plan_language.md` §4.4: AMI against POS is
evidence of linguistic specialization ONLY if it exceeds AMI against a meaningless
partition of the same shape. Without the control, a POS agreement of 0.45 is
uninterpretable -- it could be the metric's floor for a 6-way partition with these
marginals. The arithmetic study had no such control and did not need one, because
`OP_TO_EXPERT` is functional truth; here it is the difference between a measurement
and a number.

Evaluation-time only: it costs no extra training runs, just a second confusion matrix
per validation pass.

TWO SECTIONS OF THE PLAN SPECIFY THIS DIFFERENTLY, AND THE DISAGREEMENT MATTERS.
T-L3.3 says "permuting the family assignment across types while holding the per-family
TYPE counts fixed". §4.4 says "a random type-level 6-way partition with the observed
family TOKEN proportions". Those are not the same partition, and on this corpus they
are very far apart: L1_FUNCTION is 2.6% of types and 27.3% of tokens.

§4.4's specification is the correct one, and this file implements it. The reason is
what the metrics are computed over: every entry of the routing confusion matrix is a
TOKEN, so AMI's and Hungarian accuracy's chance levels depend on the TOKEN marginals.
A control that matched type counts would carry a token profile nothing like the real
partition's -- roughly uniform instead of ~5x imbalanced (5.29x on wikitext-2, 5.17x on
wikitext-103; the exact ratio is per corpus and is computed at build time rather than
written down here, T-LX.6) -- and the comparison would
then confound "meaningless" with "differently balanced", which is precisely the
confusion the control exists to remove. The realised type counts are recorded anyway,
so the departure from T-L3.3's looser wording is visible rather than silent.

HOW A DRAW IS MADE. Types are visited in a seeded random order and each is assigned to
whichever family is currently furthest below its token-mass target. That gives token
proportions matched to a few parts in 10^4 while the assignment itself carries no POS
information -- a uniform permutation would match neither marginal. Randomising the
visit order is what stops the greedy step from reconstructing POS structure through the
frequency-family correlation (measured at normalized MI 0.27, T-L2.6).

`-1` TYPES STAY `-1`. They carry no family in either partition and are masked out of
every metric, so moving them would change the token count the two metric sets are
computed over and make the pair incomparable.

N_DRAWS, NOT ONE. T-L3.3 requires the difference to be "reported with its uncertainty,
not just the raw pair". One control partition gives a point estimate with no spread, so
the artifact holds `N_DRAWS` independent draws and the metric layer reports
mean +- std across them.
"""

import argparse
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
NUM_FAMILIES = _LF.NUM_FAMILIES
NO_FAMILY_IDX = _LF.NO_FAMILY_IDX
LANG_FAMILY_LABELS = _LF.LANG_FAMILY_LABELS

# Independent control partitions. Ten is enough for a mean +- std that is not itself
# noise, and the artifact is 10 x 8192 int8 = 80 kB, so there is no reason to
# economise. Fixed rather than a CLI knob: the number of draws is part of what the
# reported uncertainty MEANS, so it belongs in the artifact's definition.
N_DRAWS = 10

# The control is a dataset artifact, not a run, so its seed is not one of the frozen
# run seeds (42..46) -- using those would tie a data artifact to a training seed and
# invite the question of which arm's control is which. A separate documented constant
# keeps the two seed spaces from touching.
CONTROL_SEED = 20260903


def train_token_counts(corpus, vocab_size):
    """Per-type occurrence counts over the packed TRAIN split.

    Train, not val: the control's token marginals must match the partition the model
    was trained against, and reading val here would make the null depend on the split
    it is later evaluated on.
    """
    arr = np.load(os.path.join(bld.corpus_dir(corpus), "train.npy"), mmap_mode="r")
    counts = np.zeros(vocab_size, dtype=np.int64)
    rows = max(1, (1 << 22) // arr.shape[1])
    for i in range(0, arr.shape[0], rows):
        counts += np.bincount(
            np.asarray(arr[i:i + rows], dtype=np.int64).ravel(),
            minlength=vocab_size)
    return counts


def _repair(out, counts, target, total, tol=1e-4, max_iters=20000):
    """Drive the token-share error down by moving or SWAPPING single types.

    The one-pass greedy assignment leaves a tail: with 3 types carrying ~4% of the
    canonical corpus each, one arriving late in the random visit order dumps a large
    lump into whichever family had the deficit and overshoots. Measured before this
    step, the worst per-family error over 10 canonical draws was 2.0e-2 -- 2
    percentage points -- against a mean nearer 4e-3.

    Each iteration takes the family most OVER target and the one most UNDER and tries
    two operations, keeping whichever reduces the pair's worst error more:

      MOVE  one type over -> under, choosing the mass closest to the amount that
            would balance the pair;
      SWAP  a type from each, choosing the pair whose mass DIFFERENCE is closest to
            that amount.

    The swap is what makes the loop reliable rather than usually-fine. A move alone
    gets stuck whenever the over-family holds no type small enough to transfer without
    overshooting -- measured on canonical draw 0, which stalled at 1.4e-2 with moves
    only while the other nine reached ~5e-5. A swap can transfer an arbitrarily small
    NET mass out of two large types, so the granularity floor disappears.

    Selection is by MASS only and never consults the real partition, so the repair
    cannot reintroduce POS structure. An operation is accepted only if it strictly
    reduces the pair's worst error, which makes the loop monotone and lets it stop on
    its own well before `max_iters`.
    """
    have = np.array([counts[out == f].sum() / total for f in range(NUM_FAMILIES)],
                    dtype=np.float64)
    mass = counts / total
    for _ in range(max_iters):
        err = have - target
        if np.abs(err).max() < tol:
            break
        over, under = int(np.argmax(err)), int(np.argmin(err))
        if over == under:
            break
        want = min(err[over], -err[under])
        c_over = np.flatnonzero(out == over)
        c_under = np.flatnonzero(out == under)
        if c_over.size == 0:
            break
        before = max(abs(err[over]), abs(err[under]))

        # -- candidate MOVE ---------------------------------------------------
        m_over = mass[c_over]
        i_mv = int(c_over[int(np.argmin(np.abs(m_over - want)))])
        d_mv = mass[i_mv]
        best = (max(abs(err[over] - d_mv), abs(err[under] + d_mv)),
                i_mv, None, d_mv)

        # -- candidate SWAP ---------------------------------------------------
        if c_under.size:
            m_und = mass[c_under]
            srt = np.argsort(m_und, kind="stable")
            m_srt, idx_srt = m_und[srt], c_under[srt]
            # For each over-type i we want an under-type j with m_j ~ m_i - want.
            pos = np.clip(np.searchsorted(m_srt, m_over - want), 0, m_srt.size - 1)
            d_sw = m_over - m_srt[pos]
            cost = np.maximum(np.abs(err[over] - d_sw), np.abs(err[under] + d_sw))
            k = int(np.argmin(cost))
            if cost[k] < best[0]:
                best = (cost[k], int(c_over[k]), int(idx_srt[pos[k]]), d_sw[k])

        if best[0] >= before:
            break
        _, i, j, d = best
        out[i] = under
        if j is not None:
            out[j] = over
        have[over] -= d
        have[under] += d
    return out


def one_draw(real, counts, rng):
    """One POS-independent partition whose TOKEN proportions match `real`'s.

    Types are visited in random order; each goes to whichever family is furthest
    below its running token-mass target, and `_repair` then removes the tail the
    greedy pass leaves. The random visit order is load-bearing -- a
    frequency-ordered pass would place the highest-frequency types first and, since
    frequency and family are correlated at normalized MI 0.27, would partly rebuild
    the real partition.

    `-1` types are copied through unchanged. They are masked out of every metric, so
    reassigning them would silently change the token population the real and control
    metric sets are computed over.
    """
    mapped = real >= 0
    total = float(counts[mapped].sum())
    target = np.array(
        [counts[real == f].sum() / total for f in range(NUM_FAMILIES)],
        dtype=np.float64)

    out = np.full(real.shape, NO_FAMILY_IDX, dtype=np.int8)
    have = np.zeros(NUM_FAMILIES, dtype=np.float64)
    order = np.flatnonzero(mapped)
    rng.shuffle(order)
    for t in order:
        # Deficit against the target, in token-mass fraction. argmax on a plain
        # array breaks ties toward the lowest family index, which is deterministic
        # given the seed.
        f = int(np.argmax(target - have))
        out[t] = f
        have[f] += counts[t] / total
    out = _repair(out, counts, target, total)
    return out, target


def draw_stats(draw, real, counts):
    """Type counts and token shares for one partition, for both marginal profiles."""
    mapped = real >= 0
    total = float(counts[mapped].sum())
    return {
        "types": {LANG_FAMILY_LABELS[f]: int((draw == f).sum())
                  for f in range(NUM_FAMILIES)},
        "token_share": {LANG_FAMILY_LABELS[f]:
                        float(counts[draw == f].sum() / total)
                        for f in range(NUM_FAMILIES)},
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corpus", default=bld.CANONICAL_CORPUS,
                    choices=sorted(bld.CORPORA))
    args = ap.parse_args(argv)

    corpus = args.corpus
    cdir = bld.corpus_dir(corpus)
    man = bld.load_manifest(corpus)
    fam_path = os.path.join(cdir, "token_family.npy")
    if not os.path.exists(fam_path):
        raise SystemExit(
            f"{fam_path} is missing. The control is a permutation OF the real "
            f"partition, so run build_family_lookup.py --corpus {corpus} first."
        )
    real = np.load(fam_path)
    V = man["vocab_size"]
    counts = train_token_counts(corpus, V)

    rng = np.random.default_rng(CONTROL_SEED)
    draws = np.empty((N_DRAWS, V), dtype=np.int8)
    stats, target = [], None
    for d in range(N_DRAWS):
        draws[d], target = one_draw(real, counts, rng)
        stats.append(draw_stats(draws[d], real, counts))

    out = os.path.join(cdir, "token_family_shuffled.npy")
    np.save(out, draws)
    sha = bld._sha256_file(out)

    real_stats = draw_stats(real, real, counts)
    # Worst per-family token-share error across every draw: the number that says
    # whether the null really is marginal-matched.
    worst = max(
        abs(s["token_share"][lbl] - real_stats["token_share"][lbl])
        for s in stats for lbl in LANG_FAMILY_LABELS)

    # The quantity that actually sets AMI's and Hungarian accuracy's chance level is
    # the marginal ENTROPY, not any single family's share. Reported so the control's
    # validity is a number rather than an argument about how close 2e-4 is.
    def _norm_entropy(share_map):
        p = np.array([share_map[lbl] for lbl in LANG_FAMILY_LABELS])
        p = p[p > 0]
        return float(-(p * np.log(p)).sum() / np.log(NUM_FAMILIES))

    h_real = _norm_entropy(real_stats["token_share"])
    h_ctrl = [_norm_entropy(s["token_share"]) for s in stats]

    # T-LX.6. The largest/smallest family token ratio is CORPUS-SPECIFIC, and the note
    # written into the manifest below used to quote the literal `5.29x` -- which is the
    # wikitext-2 dev figure, where canonical wikitext-103 is 5.17x. Computed from this
    # corpus's own marginals so the artifact's justification can never describe a
    # different corpus than the one it was built from. Same class of defect as the
    # partition-entropy constant: a per-corpus quantity frozen into prose.
    _shares = [real_stats["token_share"][lbl] for lbl in LANG_FAMILY_LABELS]
    imbalance = (max(_shares) / min(_shares)) if min(_shares) > 0 else float("inf")

    print(f"[control] {corpus}: {N_DRAWS} draws, seed {CONTROL_SEED}, "
          f"{int((real >= 0).sum()):,} mapped types "
          f"({int((real < 0).sum()):,} left at -1)")
    print(f"[control] {'family':16s} {'real types':>11s} {'ctrl types':>11s} "
          f"{'real tok%':>10s} {'ctrl tok%':>10s}")
    for lbl in LANG_FAMILY_LABELS:
        ct = np.mean([s["types"][lbl] for s in stats])
        cs = np.mean([s["token_share"][lbl] for s in stats])
        print(f"[control] {lbl:16s} {real_stats['types'][lbl]:11,d} "
              f"{ct:11,.1f} {100 * real_stats['token_share'][lbl]:9.2f}% "
              f"{100 * cs:9.2f}%")
    print(f"[control] worst per-family token-share error over all draws: "
          f"{worst:.2e}")
    print(f"[control] normalized token-marginal entropy: real {h_real:.6f}  "
          f"control {np.mean(h_ctrl):.6f} +- {np.std(h_ctrl):.1e}  "
          f"(this is what sets AMI's chance level)")

    bld.update_manifest(corpus, {
        "shuffled_control_generator": "data/lang/build_shuffled_control.py",
        "shuffled_control_generator_commit": bld._git_commit(),
        "token_family_shuffled_file":
            f"data/lang/{corpus}/token_family_shuffled.npy",
        "token_family_shuffled_sha256": sha,
        "token_family_shuffled_shape": [int(draws.shape[0]), int(draws.shape[1])],
        "shuffled_control_n_draws": N_DRAWS,
        "shuffled_control_seed": CONTROL_SEED,
        "shuffled_control_matched": "train TOKEN proportions (plan_language.md §4.4)",
        "shuffled_control_matched_note": (
            "T-L3.3's wording says per-family TYPE counts; §4.4 says observed TOKEN "
            "proportions, and this artifact implements §4.4. Confusion-matrix "
            "entries are tokens, so AMI's and Hungarian accuracy's chance levels "
            "depend on the token marginals; a type-count-matched control would be "
            f"roughly token-uniform against this corpus's real {imbalance:.2f}x "
            "largest/smallest family token imbalance and would confound "
            "'meaningless' with 'differently balanced'. Realised type "
            "counts are recorded below so the departure is visible."
        ),
        # T-LX.6: recorded rather than left implicit in the note, so a reader comparing
        # corpora gets the ratio as a number instead of parsing a sentence.
        "shuffled_control_real_token_imbalance_ratio": float(imbalance),
        "shuffled_control_target_token_share": {
            LANG_FAMILY_LABELS[f]: float(target[f]) for f in range(NUM_FAMILIES)},
        "shuffled_control_real_marginals": real_stats,
        "shuffled_control_draw_marginals": stats,
        "shuffled_control_worst_token_share_error": float(worst),
        "shuffled_control_marginal_entropy_real": h_real,
        "shuffled_control_marginal_entropy_draws": [float(x) for x in h_ctrl],
        "shuffled_control_unmapped_preserved": True,
    })
    print(f"[control] manifest -> {bld.manifest_path(corpus)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
