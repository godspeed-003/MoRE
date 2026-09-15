"""metrics.py - Entropy, cosine similarity, depth and routing metrics; paper figures.

Extracted from the former monolithic train.py (plan.md 9: ONE training
system, split into readable modules; not parallel implementations).
"""

import os
import sys
import json
import time
import math
import collections
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import wandb

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .families import (expert_labels, ALL_OP_NAMES, NUM_OP_TYPES,
                       NUM_EXPERTS_CANONICAL)
from .model import MoREModel

# ---------------------------------------------------------------------------
# 3. Metric utilities
# ---------------------------------------------------------------------------

def expert_load_entropy_from_counts(
    counts: torch.Tensor,
    num_experts: int,
) -> torch.Tensor:
    """
    Normalized Shannon entropy H / log(E) of an ALREADY-ACCUMULATED load vector.

    `counts[e]` is the number of hard-argmax token assignments to expert `e`.
    This is the whole input the statistic needs -- the per-token indices that
    produced those counts are not required and must not be retained to get here
    (T-LX.9; see `compute_expert_load_entropy` below).

    This is a LOAD-BALANCE DIAGNOSTIC ONLY. High entropy means routing is
    uniform, i.e. no expert is starved; it is NOT evidence of specialization
    and must never be tuned toward (CLAUDE.md 2, 4). Returns None when E < 2.
    """
    total   = counts.sum().clamp(min=1.0)
    probs   = counts / total
    entropy = -(probs * torch.log(probs + 1e-8)).sum()

    # T6.4 (updated_rules.md 8.4 / CLAUDE.md 4): entropy is reported NORMALIZED,
    # H / log(E), so it is comparable across different expert counts. For E = 1
    # (the MoR baseline) log(E) = 0, the quantity is undefined, and raw H is
    # identically 0.0 -- reporting that 0.0 would rank MoR as "maximally
    # collapsed" against MoE/MoRE when in fact it has no routing decision to
    # make. Return None, which callers must render as N/A, never as a number.
    if num_experts < 2:
        return None
    return entropy / math.log(num_experts)


def compute_expert_load_entropy(
    depth_exits: torch.Tensor,
    all_expert_idx: list,
    num_experts: int,
) -> torch.Tensor:
    """
    Normalized expert-load entropy from RAW per-depth argmax index tensors.

    Kept because it is the form the correctness suite exercises (G4.16) and the
    form a caller with indices already in hand wants. It counts, then defers to
    `expert_load_entropy_from_counts`, so there is exactly one definition of the
    statistic and the two entry points cannot drift apart.

    T-LX.9: the TRAINING LOOP must NOT use this entry point. Reaching it
    requires holding every per-depth index tensor of the epoch, which on the
    language task is ~7.4 GiB of live tensors by the last batch. `engine.py`
    already accumulates the identical `counts` vector incrementally and calls
    `expert_load_entropy_from_counts` instead.
    """
    counts = torch.zeros(num_experts, device=depth_exits.device)
    for idx_tensor in all_expert_idx:
        for e in range(num_experts):
            counts[e] += (idx_tensor == e).sum().float()
    return expert_load_entropy_from_counts(counts, num_experts)


def compute_pairwise_cosine_sim(model: MoREModel):
    """
    MEAN and MAX cosine similarity between every pair of expert w1 weight
    vectors, across all MoRE blocks. Both are required (CLAUDE.md 4): the max
    detects any single collapsing pair, the mean describes the population.

    Approaching 1.0 indicates experts are converging to identical
    parameterizations. Low values are "consistent with differentiated
    parameterizations" -- they do NOT prove orthogonality.

    Returns (mean, max), or (None, None) when there are fewer than 2 experts.
    """
    sims: list[float] = []
    for block in model.blocks:
        weights = [
            exp[0].weight.detach().flatten()    # first Linear in each expert Sequential
            for exp in block.moe_block.experts
        ]
        for i, j in itertools.combinations(range(len(weights)), 2):
            sims.append(F.cosine_similarity(
                weights[i].unsqueeze(0), weights[j].unsqueeze(0)
            ).item())

    # T6.5: with a single expert there is no PAIR, so both statistics are
    # undefined. The old code initialised max_sim = -1.0 and returned that
    # untouched for E = 1, publishing the sentinel -1.0 as if it were a
    # measured cosine similarity (prohibited, CLAUDE.md 4).
    if not sims:
        return None, None
    return sum(sims) / len(sims), max(sims)


# ---------------------------------------------------------------------------
# T-L3.2  What the confusion diagonal is CALLED, per task
# ---------------------------------------------------------------------------
#
# The quantity is the same on both tasks -- `mean(predicted_expert ==
# oracle_expert)`, equal to the confusion diagonal within tolerance. What differs is
# what it licenses you to say.
#
# ARITHMETIC. `families.OP_TO_EXPERT` is a FUNCTIONAL ground truth: `ADD` really does
# belong with `SUB`, so disagreeing with it is genuinely worse routing.
# `updated_rules.md` §8 makes this the authoritative first-step routing accuracy, and
# that name is correct there.
#
# LANGUAGE. A POS partition is a LINGUISTIC PRIOR. Nothing says the optimal expert
# split for next-token prediction is noun-versus-verb; a router that separated
# "begins a rare multi-piece name" from "continues one" could be the better partition
# while scoring near chance against POS. Calling the number "accuracy" would assert
# something the experiment cannot support, so on language it is named for what it
# measures (`plan_language.md` §4.4).
#
# There is also a measured reason the two cannot be read the same way. The language
# oracle partition is NOT near-uniform: its own normalized load entropy is ~0.89, with
# a ~5.2x largest/smallest family token ratio (T-L3.1). A router that reproduced POS
# exactly would score ~0.89 on load entropy, so the balance term and this agreement
# number pull against each other -- which was not true on arithmetic, where the oracle
# families were near-uniform by construction.
#
# THE EXACT VALUE IS PER CORPUS AND THE CORPORA DO NOT AGREE (T-LX.6):
# wikitext-2 (dev) 0.8884, wikitext-103 (canonical) 0.8942. That gap is small in
# absolute terms and decisive in direction -- a run measuring 0.890 is just BELOW the
# partition on canonical and just ABOVE it on dev, i.e. "matching the balance the data
# has" versus "buying uniformity it does not", which is the exact reading T-L7.1 used
# to justify routing_balance_weight = 0.001. So no constant is written here: every
# language run publishes its own corpus's value as
# `val/routing_pos_partition_load_entropy`, read from that corpus's
# `dataset_meta.json` by `MoRELanguageDataset`.
ROUTING_AGREEMENT_KEYS = {
    "arithmetic": "val/routing_accuracy",
    "language":   "val/routing_agreement_with_pos",
}

ROUTING_AGREEMENT_CAPTIONS = {
    "arithmetic": (
        "Routing accuracy: mean(predicted_expert == oracle_expert) against "
        "families.OP_TO_EXPERT, which is a functional ground truth for the "
        "arithmetic op set. Equals the confusion diagonal fraction within tolerance."
    ),
    "language": (
        "AGREEMENT WITH A PRIOR, NOT ACCURACY. mean(predicted_expert == "
        "token_family[input_id]) against a majority-POS partition, which is a "
        "linguistic hypothesis about a useful expert split rather than a functional "
        "ground truth. A router may be better for next-token prediction and still "
        "score low here. Read the permutation-invariant metrics first "
        "(see LANGUAGE_SPECIALIZATION_ORDER), and read load entropy against this "
        "corpus's own partition entropy -- published per run as "
        "val/routing_pos_partition_load_entropy, ~0.89 and NOT 1.0. It differs "
        "between corpora (dev 0.8884, canonical 0.8942), so use the run's value."
    ),
}


# T-L3.2 makes the permutation-invariant metrics PRIMARY for language, and this list
# is what "primary" means mechanically: the order any language specialization report
# must present them in, with the POS-agreement number LAST. Kept here rather than in
# the results writer so the requirement is one checkable object instead of a
# convention in a prose file -- `code/test_language_families.py` asserts the
# agreement key is last, and the language results writer consumes this list.
LANGUAGE_SPECIALIZATION_ORDER = (
    "val/routing_hungarian_accuracy",
    "val/routing_ami",
    "val/routing_purity",
    "val/routing_matched_macro_recall",
    "val/routing_agreement_with_pos",
)


def routing_agreement_key(task: str = "arithmetic") -> str:
    """The metric key the confusion diagonal is published under, for `task`.

    RAISES on an unknown task rather than defaulting. A default here would publish a
    language run's agreement number under `val/routing_accuracy`, which is the one
    outcome T-L3.2 exists to prevent -- and it would do so silently, in a key the
    exporter then joins across tasks.
    """
    if task not in ROUTING_AGREEMENT_KEYS:
        raise KeyError(
            f"unknown task {task!r}: expected one of "
            f"{sorted(ROUTING_AGREEMENT_KEYS)}. The routing-metric NAME is "
            "task-dependent (plan_language.md §4.4), so there is no safe default."
        )
    return ROUTING_AGREEMENT_KEYS[task]


def routing_agreement_caption(task: str = "arithmetic") -> str:
    """The caption that must accompany the number wherever it is presented."""
    if task not in ROUTING_AGREEMENT_CAPTIONS:
        raise KeyError(f"unknown task {task!r}")
    return ROUTING_AGREEMENT_CAPTIONS[task]


def routing_agreement_keys_all() -> tuple[str, ...]:
    """Every task's spelling, for code that must recognise the number generically."""
    return tuple(ROUTING_AGREEMENT_KEYS[t] for t in sorted(ROUTING_AGREEMENT_KEYS))


# ---------------------------------------------------------------------------
# T-L3.3 / ablation G  The shuffled control, evaluated beside the real partition
# ---------------------------------------------------------------------------

# Which metrics are compared against the control. The agreement figure is included
# because its floor is exactly the thing in question, and the permutation-invariant
# three because they are what §4.4 makes primary.
CONTROL_COMPARED_METRICS = ("raw_accuracy", "hungarian_accuracy", "ami", "purity")

# Below this, the control draws are identical and the "spread" is floating-point residue
# rather than a null distribution, so `delta / std` is undefined rather than large. See the
# comment at its use for the run that made this necessary.
_CONTROL_STD_FLOOR = 1e-12


def confusion_from_pairs(oracle, predicted, num_experts: int):
    """`[num_experts, num_experts]` counts of (oracle family, predicted expert).

    Rows are the oracle family, columns the predicted expert, matching
    `routing_accuracy_from_confusion`'s diagonal convention. Tokens whose oracle
    family is negative are DROPPED, not counted as a class -- `-1` is the documented
    ignore label, and giving it a row would make the ignore share look like a family
    the router failed at.
    """
    oracle = np.asarray(oracle).ravel()
    predicted = np.asarray(predicted).ravel()
    keep = oracle >= 0
    o, p = oracle[keep].astype(np.int64), predicted[keep].astype(np.int64)
    flat = np.bincount(o * num_experts + p, minlength=num_experts * num_experts)
    return flat.reshape(num_experts, num_experts)


def specialization_vs_control(predicted_expert, token_ids, token_family,
                              control_draws, num_experts: int) -> dict:
    """Both metric sets side by side, with the difference and its uncertainty.

    T-L3.3. `control_draws` is the `(n_draws, V)` artifact
    `data/lang/build_shuffled_control.py` writes: independent POS-independent
    partitions with the same token marginals. The same predicted assignments are
    scored against the real partition once and against each draw, so the only thing
    that varies is the partition being agreed with.

    WHAT THIS IS FOR, restated because a number without it is misleading. AMI of 0.31
    against POS means nothing on its own -- it could be the floor for a 6-way
    partition with these marginals. It is evidence of linguistic specialization only
    if it exceeds AMI against a meaningless partition of the same shape. The
    arithmetic study needed no such control because `OP_TO_EXPERT` is functional
    truth; here it is the difference between a measurement and a number.

    Returns, for each metric in `CONTROL_COMPARED_METRICS`:
        real                the value against POS
        control_mean/std    across the draws
        delta               real - control_mean
        delta_z             delta / control_std, or None when the control has no
                            spread (a z with a zero denominator is not a large
                            effect, it is an undefined one)
    `None` propagates rather than being filled: at E == 1 the permutation-invariant
    metrics are undefined and must stay that way (CLAUDE.md §4).
    """
    tf = np.asarray(token_family)
    ids = np.asarray(token_ids).ravel()
    pred = np.asarray(predicted_expert).ravel()
    draws = np.atleast_2d(np.asarray(control_draws))
    return partition_agreement_vs_control(
        tf[ids], pred, [draws[d][ids] for d in range(draws.shape[0])], num_experts)


def partition_agreement_vs_control(oracle, predicted, control_oracles,
                                   num_experts: int) -> dict:
    """The comparison itself, over any UNIT -- a token, or a whole article.

    Factored out of `specialization_vs_control` for T-L6.6, which scores the same
    predictions against a per-BLOCK topic partition instead of a per-token POS one. The
    two callers differ only in what a unit is and how the oracle label reaches it; the
    statistics must be identical, and one implementation is how that stays true. Two
    copies of this would be the two-copies-that-diverge failure `changelog.md` already
    records once.

    `oracle` and `predicted` are paired label vectors of the same length;
    `control_oracles` is a list of same-length null label vectors.
    """
    real_pim = permutation_invariant_routing_metrics(
        confusion_from_pairs(oracle, predicted, num_experts), num_experts)
    ctrl_pims = [
        permutation_invariant_routing_metrics(
            confusion_from_pairs(c, predicted, num_experts), num_experts)
        for c in control_oracles
    ]

    out = {"n_control_draws": len(ctrl_pims), "n_units": int(np.asarray(oracle).size),
           "metrics": {}}
    for key in CONTROL_COMPARED_METRICS:
        rv = real_pim.get(key)
        cvs = [p.get(key) for p in ctrl_pims]
        if rv is None or not cvs or any(v is None for v in cvs):
            out["metrics"][key] = {"real": rv, "control_mean": None,
                                  "control_std": None, "delta": None,
                                  "delta_z": None}
            continue
        cm = float(np.mean(cvs))
        cs = float(np.std(cvs, ddof=1)) if len(cvs) > 1 else 0.0
        delta = float(rv) - cm
        out["metrics"][key] = {
            "real": float(rv), "control_mean": cm, "control_std": cs,
            "delta": delta,
            # `cs > 0` is NOT a sufficient guard, and a real run proved it: a router whose
            # block-majority expert is CONSTANT gives every control draw the identical
            # score, so the spread is floating-point residue -- measured at 2.1e-31 in
            # `runs/langB_MoRE_seed44__2ce26c0d`, which turned a delta of -2e-31 into a
            # z of -0.95. A z near one from a 1e-31 denominator is not a small effect,
            # it is no effect at all, and it would read as the former. Below the floor
            # the ratio is undefined and is reported as such.
            "delta_z": ((delta / cs) if cs > _CONTROL_STD_FLOOR else None),
        }
    return out


def block_majority_expert(predicted_expert, seq_len: int) -> np.ndarray:
    """`block_expert[n_blocks]` -- the expert most of a block's tokens went to.

    T-L6.6's unit of analysis. "Did this article's tokens concentrate on one expert" is
    the article-level question, and a majority vote is the honest reduction: a block whose
    tokens split evenly across six experts has no article-level expert, and the majority
    then records whichever won rather than pretending to a concentration that is not
    there. `block_topic_concentration` below is what says how meaningful the majority was.
    """
    p = np.asarray(predicted_expert).ravel()
    if p.size % seq_len != 0:
        raise ValueError(
            f"{p.size} predictions do not divide into blocks of {seq_len}. T-L6.6 needs "
            "the per-token predictions of WHOLE blocks to reduce them per article."
        )
    rows = p.reshape(-1, seq_len)
    n_e = int(rows.max()) + 1 if rows.size else 1
    counts = np.stack([(rows == e).sum(axis=1) for e in range(n_e)], axis=1)
    return counts.argmax(axis=1).astype(np.int64)


def block_topic_concentration(predicted_expert, seq_len: int) -> float | None:
    """Mean share of a block's tokens that went to the block's MAJORITY expert.

    Reported beside every article-level agreement number, because the agreement is
    computed on a majority label and a majority of 1/6 means the article was not routed
    anywhere in particular. At chance over E experts this is ~1/E plus sampling; at 1.0
    every article went entirely to one expert. Without it, a high article-level AMI could
    describe a partition of coin flips.
    """
    p = np.asarray(predicted_expert).ravel()
    if p.size == 0 or p.size % seq_len != 0:
        return None
    rows = p.reshape(-1, seq_len)
    n_e = int(rows.max()) + 1
    counts = np.stack([(rows == e).sum(axis=1) for e in range(n_e)], axis=1)
    return float((counts.max(axis=1) / seq_len).mean())


def make_label_permutation_controls(labels, n_draws: int, seed: int = 0) -> list:
    """`n_draws` permutations of `labels`, destroying the pairing only.

    For a per-BLOCK partition this is exact rather than approximate, which the per-type
    case (T-L3.3) had to work for: every stored block holds exactly `seq_len` tokens, so
    permuting block labels preserves the token marginals of every topic EXACTLY. No
    mass-matching repair is needed here.
    """
    a = np.asarray(labels).ravel()
    rng = np.random.default_rng(seed)
    return [rng.permutation(a) for _ in range(int(n_draws))]


# T-L6.6: the two reference axes, and the order they must be presented in. POS first
# because §4.4 makes the permutation-invariant metrics primary against it, topic second
# because it is INDUCED (our k, our seed) and carries a weaker claim -- but both always,
# because the question is "what does it organize by", not "does it reproduce POS".
REFERENCE_AXES = ("pos", "topic")


def article_agreement_vs_topic(predicted_expert, block_topic, seq_len: int,
                               num_experts: int, n_draws: int = 10,
                               seed: int = 0) -> dict:
    """T-L6.6: did this ARTICLE's tokens concentrate on one expert, and is that real?

    The article-level counterpart of `specialization_vs_control`. Per-token predictions
    are reduced to one expert per block by majority vote, scored against the induced
    topic partition, and compared with `n_draws` label permutations of that partition --
    which is an exact marginal-matched null here, because every block holds exactly
    `seq_len` tokens.

    `concentration` is reported alongside and must be read first: an article-level
    agreement computed on a majority label means nothing if the majority was 1/6.
    """
    bt = np.asarray(block_topic).ravel()
    be = block_majority_expert(predicted_expert, seq_len)
    if be.size != bt.size:
        raise ValueError(
            f"{be.size} blocks of predictions against {bt.size} topic labels. The topic "
            "array is per stored block, so the predictions must cover whole blocks of "
            "the same split in the same order."
        )
    out = partition_agreement_vs_control(
        bt, be, make_label_permutation_controls(bt, n_draws, seed), num_experts)
    out["concentration"] = block_topic_concentration(predicted_expert, seq_len)
    out["n_blocks"] = int(bt.size)
    out["axis"] = "topic"
    out["axis_is_induced"] = True
    return out


def control_comparison_to_wandb(cmp: dict, prefix: str = "val/routing_control") -> dict:
    """Flatten `specialization_vs_control` into log keys, `None` omitted.

    Omitted rather than written as 0.0: an undefined difference is not a zero
    difference, and a 0.0 in a "real minus control" column reads as "the POS
    partition is no better than noise", which is a finding, not a missing value.
    """
    log: dict = {}
    for key, rec in cmp.get("metrics", {}).items():
        for field in ("real", "control_mean", "control_std", "delta", "delta_z"):
            if rec.get(field) is not None:
                log[f"{prefix}/{key}_{field}"] = rec[field]
    log[f"{prefix}/n_draws"] = cmp.get("n_control_draws")
    # T-L6.6 extras, present only on the article-level (topic) comparison. Concentration
    # is not optional decoration: an article-level agreement computed on a majority label
    # is uninterpretable without knowing how large the majority was.
    for k in ("concentration", "n_blocks"):
        if cmp.get(k) is not None:
            log[f"{prefix}/{k}"] = cmp[k]
    return log


def routing_accuracy_from_confusion(confusion, num_experts: int) -> float:
    """
    THE authoritative routing accuracy. Every caller must use this function.    Defined as mean(predicted_expert == oracle_expert) over exactly the tokens
    the confusion matrix was built from, computed as the diagonal fraction so
    the reported scalar and the reported matrix cannot disagree by construction
    (updated_rules.md 8.2 / T6.2). Before this existed, the value was computed
    independently in `evaluate_paper_metrics` (deleted at T11.1 -- see the note
    lower in this file) and again inline in engine.py's validation loop -- two
    copies, of which only the engine's ran.

    Returns NaN (rendered "N/A", never 0.0) when the quantity does not exist:
      * num_experts < 2 -- the MoR baseline makes no routing decision at all.
        Its argmax is 0 for every token, so a diagonal fraction would report
        "the fraction of tokens that happen to be family E1" (measured 0.1005)
        as though MoR had routed badly.
      * no tokens were routed.
    """
    import torch as _torch
    total = float(confusion.sum().item() if hasattr(confusion, "sum") else 0.0)
    if num_experts < 2 or total <= 0:
        return float("nan")
    diag = (confusion.diag().sum().item() if isinstance(confusion, _torch.Tensor)
            else float(confusion.diagonal().sum()))
    return diag / total


# --------------------------------------------------------------------------- #
# T6.3 -- permutation-invariant routing metrics  (plan.md §8.3, CLAUDE.md §4)
# --------------------------------------------------------------------------- #
# Expert indices carry NO semantic identity. Nothing in the architecture says
# "expert 3 is the MOD/POW expert"; that association is whatever the router
# happens to learn. So raw accuracy -- the confusion DIAGONAL -- answers the
# question "did the router pick the index the oracle table happens to use for
# this family?", which is not the scientific question. A router that had learned
# a perfect but relabelled partition (every ADD/SUB token to expert 4, every
# MULT/DIV token to expert 0, ...) would score 0.0 raw and is in fact a complete
# success. updated_rules.md §8.3 therefore makes permutation-invariant metrics
# mandatory alongside the raw number.
#
# Read the RAW-vs-MATCHED GAP, not either number alone:
#
#   raw ≈ matched   the router uses the oracle's own indexing -- expected while
#                   `loss_weights.step_routing > 0`, because that term supervises
#                   the index directly, so raw accuracy is meaningful there.
#   raw ≪ matched   a real partition under a relabelling. This is the case the
#                   unsupervised-routing ablation exists to detect, and the case
#                   where quoting raw accuracy alone would be simply wrong.
#   both low        no partition. AMI near 0 confirms it (AMI is corrected for
#                   chance, so "0.9 accuracy" on an imbalanced set cannot hide
#                   there).
#
# Everything below is computed from the SAME E×E confusion matrix the figure and
# the raw accuracy use, so the numbers cannot disagree with the matrix or need a
# second pass over the loader. The assignment and the AMI correction are
# implemented here, exactly, rather than approximated -- a greedy match is not the
# Hungarian match and must not be labelled as one.
#
# T-L0.0: this comment used to justify the local implementations with "no
# scipy/sklearn in this environment (checked)". That was false, or had stopped
# being true: sklearn 1.3.2 and scipy 1.14.1 are both importable from the
# interpreters in ENVIRONMENT.md. The real reasons the local versions stay are
# that they are exact at E<=15 and checked against brute force in
# test_phase6_routing_metrics.py, and that the metric layer then imports nothing
# beyond torch/numpy -- so a metrics-only environment cannot silently produce a
# different Hungarian accuracy than the training environment did. Swapping
# working exact code for a library call would be churn, not a fix; only the
# stated reason was wrong.


def _to_contingency(confusion) -> np.ndarray:
    """Confusion (torch or numpy) -> float64 numpy contingency: rows=oracle."""
    if isinstance(confusion, torch.Tensor):
        return confusion.detach().cpu().double().numpy()
    return np.asarray(confusion, dtype=np.float64)


def _max_weight_assignment(weight: np.ndarray) -> tuple[list[int], float]:
    """
    EXACT maximum-weight one-to-one assignment (the Hungarian objective), by
    subset DP: dp[mask] = best total for the first popcount(mask) rows using
    exactly the columns in `mask`. O(E^2 · 2^E) -- 6 experts is 2304 steps.

    Exact, not greedy: greedy row-by-row selection is what makes a "Hungarian
    accuracy" quietly become a lower bound, and a metric that is sometimes a
    bound and sometimes the truth is not reportable.
    """
    n = weight.shape[0]
    if n != weight.shape[1]:
        raise ValueError(f"assignment needs a square matrix, got {weight.shape}")
    if n > 15:
        raise ValueError(
            f"num_experts={n}: the subset DP is exponential and would allocate "
            f"2**{n} states. Canonical is 6 (updated_rules.md §2); if a wider "
            "ablation is ever needed, bring in a real Hungarian implementation "
            "rather than silently degrading to greedy."
        )
    NEG = -np.inf
    dp = np.full(1 << n, NEG)
    dp[0] = 0.0
    choice = np.full(1 << n, -1, dtype=np.int64)
    for mask in range(1 << n):
        if dp[mask] == NEG:
            continue
        row = bin(mask).count("1")
        if row >= n:
            continue
        for col in range(n):
            if mask & (1 << col):
                continue
            nxt = mask | (1 << col)
            cand = dp[mask] + weight[row, col]
            if cand > dp[nxt]:
                dp[nxt] = cand
                choice[nxt] = col
    full = (1 << n) - 1
    assignment = [-1] * n
    mask = full
    while mask:
        col = int(choice[mask])
        assignment[bin(mask).count("1") - 1] = col
        mask &= ~(1 << col)
    return assignment, float(dp[full])


def _entropy_of_counts(counts: np.ndarray, n: float) -> float:
    """Shannon entropy (nats) of a partition given its block sizes."""
    p = counts[counts > 0] / n
    return float(-(p * np.log(p)).sum())


def _mutual_information(c: np.ndarray, n: float) -> float:
    """MI(oracle, predicted) in nats, from the contingency table."""
    a = c.sum(axis=1, keepdims=True)
    b = c.sum(axis=0, keepdims=True)
    nz = c > 0
    if not nz.any():
        return 0.0
    outer = (a @ b)[nz]
    return float(((c[nz] / n) * (np.log(c[nz] * n) - np.log(outer))).sum())


def _expected_mutual_information(a: np.ndarray, b: np.ndarray, n: int) -> float:
    """
    E[MI] under the permutation model with the marginals held fixed -- the
    correction term that turns MI into AMI.

    Without it, MI (like raw accuracy) rewards splitting: a router that scattered
    tokens into many small groups would score above one that found the true
    partition. The expectation is over the hypergeometric distribution of each
    cell, evaluated in log space because n is O(10^4-10^5) and the factorials
    overflow float64 long before that.

    log(k!) comes from one cumulative-sum table rather than per-element lgamma
    calls: the inner range can be tens of thousands of cells wide, and a Python
    loop over lgamma there costs seconds per validation pass -- enough to make an
    honest metric expensive enough to be switched off, which is its own failure
    mode.
    """
    logfact = np.concatenate(([0.0], np.cumsum(np.log(np.arange(1, n + 1,
                                                               dtype=np.float64)))))
    log_n_fact = logfact[n]
    ln_n = math.log(n)
    emi = 0.0
    for ai in a.astype(np.int64):
        ai = int(ai)
        if ai <= 0:
            continue
        for bj in b.astype(np.int64):
            bj = int(bj)
            if bj <= 0:
                continue
            lo = max(1, ai + bj - n)
            hi = min(ai, bj)
            if hi < lo:
                continue
            nij = np.arange(lo, hi + 1, dtype=np.int64)
            fnij = nij.astype(np.float64)
            term = (fnij / n) * (np.log(fnij) + ln_n
                                 - math.log(ai) - math.log(bj))
            log_w = (logfact[ai] + logfact[bj]
                     + logfact[n - ai] + logfact[n - bj]
                     - log_n_fact
                     - logfact[nij]
                     - logfact[ai - nij]
                     - logfact[bj - nij]
                     - logfact[n - ai - bj + nij])
            emi += float((term * np.exp(log_w)).sum())
    return emi


def permutation_invariant_routing_metrics(confusion, num_experts: int) -> dict:
    """
    T6.3 / updated_rules.md §8.3. The full routing-quality report, all of it
    derived from the one E×E confusion matrix.

    Keys (every one is `None` when the quantity does not exist -- never 0.0,
    never -1, per CLAUDE.md §4):

      raw_accuracy          the diagonal fraction, via the authoritative
                            `routing_accuracy_from_confusion`
      hungarian_accuracy    best accuracy over all E! relabellings of the
                            experts; >= raw_accuracy by construction
      hungarian_assignment  the matching itself, oracle family i -> expert
                            `assignment[i]`. Report it: "0.99 matched accuracy"
                            is only interpretable next to the permutation, which
                            is also what tells you whether the router merely
                            renamed the experts or genuinely mixed families.
      macro_recall          unweighted mean per-family recall. The dataset is not
                            perfectly balanced across families, so a
                            token-weighted accuracy can be carried by the largest
                            family alone.
      ami                   adjusted mutual information, chance-corrected: 0 is
                            what a random router of the same marginals scores,
                            1 is an exact partition up to relabelling. Can be
                            slightly negative -- that means worse than chance and
                            is reported as measured, not clipped to 0.
      purity                sum over experts of the largest family inside that
                            expert, over all tokens. Unlike AMI it is NOT
                            chance-corrected and rises trivially with E, so it is
                            reported beside AMI, never instead of it.
      per_family            {label: {precision, recall, f1, support}} under the
                            IDENTITY mapping -- the oracle's own indexing, which
                            is what `step_routing` supervises. READ THIS ONLY ON
                            A SUPERVISED RUN. Without supervision the expert
                            index is arbitrary, so these numbers describe the
                            router's numbering, not its partition: measured
                            unsupervised they read 0.014-0.162 macro recall while
                            the matched accuracy is a stable 0.515.
      per_family_matched    the same four quantities computed AFTER applying
                            `hungarian_assignment`, plus `matched_expert`. This
                            is the per-family table that is invariant to expert
                            relabelling, and therefore the only one that may be
                            reported for an unsupervised router (T6.8).
      matched_macro_recall  unweighted mean of the MATCHED per-family recalls.
                            Stands in the same relation to `macro_recall` as
                            `hungarian_accuracy` does to `raw_accuracy`.
      collapsed_experts     expert SLOTS that received no token at all, named
                            `expert_<j>` by INDEX. Not by family label: expert 3
                            is not "the LOGIC expert", it is whichever slot the
                            router used, and naming a collapsed slot after a
                            family asserts the very identification the rest of
                            this function exists to avoid (T6.8). A high purity
                            with 5 of 6 slots collapsed is the classic
                            flattering artifact.

    Undefined for num_experts < 2 (MoR makes no routing decision) or when no
    token was routed: every value comes back None.

    Caveat on ties, recorded because it bounds what the matched table can claim:
    the optimal assignment need not be unique. The matched TOTAL is invariant
    across tied optima -- it is the optimum -- but the split of that total across
    families can differ between two equally-optimal permutations. So
    `hungarian_accuracy` is exact, while an individual family's matched recall is
    exact only up to the tie. Ties are vanishingly unlikely on O(10^4) tokens and
    would show as a degenerate confusion matrix; the honest statement is that this
    is a property of the matching, not a defect of the implementation.
    """
    c = _to_contingency(confusion)
    n_tok = float(c.sum())
    labels = expert_labels(num_experts)
    empty = {
        "raw_accuracy": None, "hungarian_accuracy": None,
        "hungarian_assignment": None, "macro_recall": None, "ami": None,
        "purity": None, "per_family": {}, "per_family_matched": {},
        "matched_macro_recall": None, "collapsed_experts": None,
    }
    if num_experts < 2 or n_tok <= 0 or c.shape[0] != c.shape[1]:
        return empty

    raw_acc = routing_accuracy_from_confusion(confusion, num_experts)
    assignment, matched = _max_weight_assignment(c)
    row_tot = c.sum(axis=1)
    col_tot = c.sum(axis=0)

    per_family: dict[str, dict] = {}
    recalls: list[float] = []
    per_family_matched: dict[str, dict] = {}
    matched_recalls: list[float] = []

    def _prf(tp: float, support: float, predicted: float) -> dict:
        """precision / recall / f1 for one family, given its TP cell.

        Shared by the identity and the matched tables so the two cannot drift
        apart in their handling of the undefined cases: recall is undefined when
        the family has no oracle tokens, precision when nothing was predicted
        into the paired expert. F1 is N/A only when one of its inputs is itself
        undefined -- when both are defined and zero, 0.0 is a MEASUREMENT (the
        expert got nothing right) and reporting it as N/A would hide a total miss.
        """
        rec = float(tp / support) if support > 0 else None
        prec = float(tp / predicted) if predicted > 0 else None
        if prec is None or rec is None:
            f1 = None
        elif prec + rec == 0.0:
            f1 = 0.0
        else:
            f1 = 2 * prec * rec / (prec + rec)
        return {"precision": prec, "recall": rec, "f1": f1,
                "support": float(support)}

    for i, label in enumerate(labels):
        # (a) IDENTITY mapping -- oracle family i paired with expert index i.
        # Meaningful only when something forced the router onto the oracle's
        # numbering, i.e. on a step_routing-supervised run.
        stats = _prf(float(c[i, i]), float(row_tot[i]), float(col_tot[i]))
        if stats["recall"] is not None:
            recalls.append(stats["recall"])
        per_family[label] = stats

        # (b) MATCHED mapping -- oracle family i paired with the expert the
        # Hungarian matching gave it. Invariant under relabelling of the expert
        # indices: permuting the columns by pi carries assignment[i] to
        # pi(assignment[i]), and both c[i, assignment[i]] and col_tot[
        # assignment[i]] travel with it, so every number below is unchanged.
        # This is the table to report for an unsupervised router (T6.8).
        j = int(assignment[i])
        mstats = _prf(float(c[i, j]), float(row_tot[i]), float(col_tot[j]))
        if mstats["recall"] is not None:
            matched_recalls.append(mstats["recall"])
        # The paired index is recorded WITH the stats: a matched recall is not
        # interpretable without knowing which expert it was matched to, and
        # keeping them in one dict stops the two being joined wrongly later.
        mstats["matched_expert"] = j
        per_family_matched[label] = mstats

    n_int = int(round(n_tok))
    mi = _mutual_information(c, n_tok)
    emi = _expected_mutual_information(row_tot, col_tot, n_int)
    h_oracle = _entropy_of_counts(row_tot, n_tok)
    h_pred = _entropy_of_counts(col_tot, n_tok)
    denom = 0.5 * (h_oracle + h_pred) - emi
    if abs(denom) < 1e-12:
        # Both partitions are a single block: MI, EMI and both entropies are 0,
        # so the ratio is 0/0. The two clusterings agree exactly (trivially), and
        # sklearn's AMI returns 1.0 here for the same reason.
        ami = 1.0 if abs(mi - emi) < 1e-12 else None
    else:
        ami = float((mi - emi) / denom)

    return {
        "raw_accuracy": (None if math.isnan(raw_acc) else float(raw_acc)),
        "hungarian_accuracy": float(matched / n_tok),
        "hungarian_assignment": [int(v) for v in assignment],
        "macro_recall": (float(sum(recalls) / len(recalls)) if recalls else None),
        "matched_macro_recall": (float(sum(matched_recalls)
                                       / len(matched_recalls))
                                 if matched_recalls else None),
        "ami": ami,
        "purity": float(c.max(axis=0).sum() / n_tok),
        "per_family": per_family,
        "per_family_matched": per_family_matched,
        # T6.8: named by INDEX, not by expert_labels(). A collapsed slot has no
        # family identity by definition -- it received nothing -- so calling
        # index 3 "E4_LOGIC" reports a family that in fact went somewhere else.
        # Measured on an unsupervised run this printed
        # "collapsed: ['E4_LOGIC']" while LOGIC tokens were being routed
        # perfectly well to slot 1.
        "collapsed_experts": [f"expert_{j}" for j in range(num_experts)
                              if col_tot[j] == 0],
    }


def compute_token_exit_depths(
    depth_exits: torch.Tensor,
    max_depth: int,
) -> torch.Tensor:
    """Convert [N, max_depth] one-hot exit markers into per-token exit depths."""
    steps = torch.arange(
        1, max_depth + 1, device=depth_exits.device, dtype=depth_exits.dtype
    )
    return (depth_exits * steps.unsqueeze(0)).sum(dim=-1)


# ---------------------------------------------------------------------------
# T-L6.1 / T-L6.2  Depth correlations, with ties and a null band
# ---------------------------------------------------------------------------
#
# WHY CORRELATIONS AT ALL. `plan_language.md` §5.1: there is no per-token
# ground-truth depth for English, so `depth/allocation_error_*` is `N/A` on language
# (T-L6.0) and the Second Objective -- does MoRE learn adaptive computation? -- has to
# be tested against properties of the corpus we did NOT choose. Writing a per-token
# depth target and then reporting agreement with it would make a designed correlation
# look like a discovery.
#
# `spearman_vs_logfreq` and `spearman_vs_unigram_surprisal` are non-circular because
# both are fixed at dataset-build time, before any model exists.
# `spearman_vs_model_loss` is the interesting one: it asks whether the model spends
# more computation where IT finds the task hard, which is a behavioural claim rather
# than a designed one.

# ---------------------------------------------------------------------------
# T-L6.8  Which pass and which tokens each depth key covers
# ---------------------------------------------------------------------------
#
# FOUR key families all read as "the exit-depth distribution", and two of them differ in
# the fifth decimal. That is the two-copies-that-diverge hazard `changelog.md` already
# records once, and it was found in the first completed language run: 2.0034485 under one
# name and 2.0034746 under another, both labelled "mean depth by family".
#
# They are legitimately different measurements -- different passes, different token
# populations -- so the fix is not to delete one but to make it impossible to read them as
# the same quantity. Renaming is not available: `recursion/avg_depth_by_family` and
# `depth_dist/step_N_pct` are in the published arithmetic contract that Gate L0 checks. So
# the provenance is published as data, written into every `metrics.json`, and
# `code/test_lang_depth_keys.py` asserts every emitted depth key is covered by it.
#
# Longest prefix wins, so `depth/mean_by_family/` is matched by its own entry rather than
# by `depth/`.
DEPTH_KEY_PROVENANCE = {
    "depth_dist/step_": {
        "pass": "train",
        "population": "every token of every training batch, all recursion steps",
        "note": ("Accumulated in the training loop, so it includes tokens that hit the "
                 "max-depth forced exit -- step_7_pct matches halt/forced_exit_rate "
                 "exactly. NOT comparable with depth/hist, which is the validation pass."),
    },
    "depth/hist/step_": {
        "pass": "validation",
        "population": ("every position except each block's last, which has no next-token "
                       "target and therefore no per-token loss"),
        "note": ("The token-level exit-depth histogram over the whole validation split. "
                 "Counts, not percentages, and over a different pass and population than "
                 "depth_dist/*, so the two will not agree."),
    },
    "depth/mean_by_family/": {
        "pass": "validation",
        "population": ("positions with a mapped POS family (-1 excluded), last position "
                       "of each block excluded"),
        "note": ("Companion to depth/hist. Differs from "
                 "recursion/avg_depth_by_family in the fifth decimal precisely because "
                 "that one includes each block's last position."),
    },
    "recursion/avg_depth_by_family/": {
        "pass": "validation",
        "population": ("positions with step_mask true and a mapped oracle family, "
                       "INCLUDING each block's last position"),
        "note": ("The pre-existing arithmetic key, kept unchanged because Gate L0 checks "
                 "the contract it belongs to. The last-position difference is the whole "
                 "of its disagreement with depth/mean_by_family."),
    },
    "recursion/avg_depth_by_op/": {
        "pass": "validation",
        "population": "positions with a mapped operation code",
        "note": ("Arithmetic only. Empty on language, because lang_families.ALL_OP_NAMES "
                 "is [] -- there is no operation axis."),
    },
    "depth/spearman_": {
        "pass": "validation",
        "population": ("the same positions as depth/hist, so all three correlations are "
                       "over one identical token set (T-L6.1)"),
        "note": ("Each carries its own permutation null band. At n ~ 2.8e5 the band is "
                 "about +-0.004, so exceeds_null says almost nothing about EFFECT SIZE -- "
                 "read it with depth/std and depth/hist or not at all."),
    },
    "depth/by_document/": {
        "pass": "validation",
        "population": ("all positions except each block's last, grouped by document via "
                       "the count of eot_id tokens before each position"),
        "note": ("T-L6.7's within- vs between-document variance decomposition. `total`, "
                 "`within` and `between` are token-weighted POPULATION variances and sum "
                 "exactly, which `sum_matches_total` reports rather than assumes. "
                 "`between_share` is the passage-level allocation number; per-document "
                 "means are summarised (std/min/max) rather than logged one key each."),
    },
    "depth/embed_norm_": {
        "pass": "validation",
        "population": ("the same positions as depth/hist; the norm is per TYPE and is "
                       "gathered by token id, read at THIS epoch"),
        "note": ("T-L6.10's confound diagnostics. `logfreq_partial_norm` is the CORRECTED "
                 "frequency-depth correlation and the number to quote; `depth_vs_norm` and "
                 "`logfreq_vs_norm` are the confound's strength, published beside it so the "
                 "correction is readable. `logfreq_within_bins_*` is a secondary "
                 "assumption-free cross-check that UNDER-corrects (see "
                 "spearman_within_bins), so a low value from it is informative and a high "
                 "one is not conclusive."),
    },
    "depth/allocation_error_": {
        "pass": "n/a",
        "population": "none",
        "note": ("Structurally undefined on language: there is no per-token ground-truth "
                 "depth for English (plan_language.md §5.1), so the value is the string "
                 "N/A. A 0.0 here would read as perfect allocation against a curriculum "
                 "that does not exist."),
    },
}


def depth_key_provenance(key: str):
    """The provenance record for one depth key, longest-prefix match, else None."""
    best = None
    for prefix, rec in DEPTH_KEY_PROVENANCE.items():
        if key.startswith(prefix) and (best is None or len(prefix) > len(best[0])):
            best = (prefix, rec)
    return None if best is None else {"prefix": best[0], **best[1]}


def uncovered_depth_keys(keys) -> list:
    """Depth-ish keys with no provenance entry -- must be empty.

    "Depth-ish" is deliberately broad: anything whose name could be read as an exit-depth
    quantity. A key that escapes the registry is exactly the ambiguity this exists to
    prevent, so the net is cast wide and the registry is what narrows it.
    """
    out = []
    for k in keys:
        if not (k.startswith("depth") or k.startswith("recursion/avg_depth")):
            continue
        # Scalars that describe the report itself rather than a per-unit measurement.
        # `depth_key_provenance_uncovered` is here because the flag's own name starts with
        # `depth`, so without it the registry reports itself as uncovered forever -- which
        # it did, on the first run that populated the flag.
        if k in ("depth/mean", "depth/std", "depth/n_tokens",
                 "depth/distinct_exit_depths", "depth/report_error",
                 "depth_key_provenance", "depth_key_provenance_uncovered"):
            continue
        if depth_key_provenance(k) is None:
            out.append(k)
    return sorted(out)


# Percentiles of the permutation null. Two-sided 95%: a rho inside this band is
# indistinguishable from the same depth marginal paired at random.
NULL_BAND_PERCENTILES = (2.5, 97.5)
NULL_PERMUTATIONS = 200


def _average_ranks(a: np.ndarray) -> np.ndarray:
    """Ranks of `a` with TIES AVERAGED, which is what makes this Spearman.

    Ties are the whole difficulty here and they are not rare -- exit depth is an
    integer in `1..max_depth`, so on a collapsed-depth model almost every value is
    tied. `argsort().argsort()` would give ordinal ranks and break ties by array
    position, which manufactures a correlation out of whatever order the batch
    happened to arrive in: the arithmetic POC put 93-97% of tokens at exactly 2 steps,
    and ordinal ranks over that would have produced a confident non-zero rho from
    nothing.
    """
    order = np.argsort(a, kind="stable")
    ranks = np.empty(a.size, dtype=np.float64)
    sorted_a = a[order]
    i = 0
    while i < a.size:
        j = i
        while j + 1 < a.size and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def spearman_rho(x, y):
    """Spearman rank correlation with averaged ties, or None when undefined.

    `None` rather than 0.0 or nan when either side is constant: a constant vector has
    no rank variance, so the correlation does not exist, and a 0.0 there would read as
    "measured no relationship" (CLAUDE.md §4). A collapsed-depth model is exactly the
    case that produces it, so this branch is reached in practice.

    Implemented rather than taken from scipy so the metric layer keeps its
    torch/numpy-only import set, matching the hand-rolled Hungarian and AMI above.
    `code/test_lang_depth.py` checks it against `scipy.stats.spearmanr` on random and
    heavily-tied data, so "exact" is verified rather than claimed.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if x.size != y.size:
        raise ValueError(
            f"spearman_rho needs paired inputs, got {x.size} and {y.size}. A "
            "length mismatch here would silently correlate different token sets."
        )
    if x.size < 2:
        return None
    rx, ry = _average_ranks(x), _average_ranks(y)
    sx, sy = rx.std(), ry.std()
    if sx == 0.0 or sy == 0.0:
        return None
    return float(((rx - rx.mean()) * (ry - ry.mean())).mean() / (sx * sy))


def spearman_with_null(x, y, n_perm=NULL_PERMUTATIONS, seed=0):
    """`{rho, n, null_low, null_high, null_mean, exceeds_null}` for one pair.

    T-L6.2, and it is mandatory rather than decorative. A near-constant depth vector
    can still produce a nonzero rho through tie-breaking alone, so a depth correlation
    published without its null band is uninterpretable. The null destroys ONLY the
    pairing -- `y` is permuted, so both marginals, including the tie structure that
    causes the problem, are preserved exactly.

    `exceeds_null` is the verdict: True when the observed rho falls outside the
    two-sided 95% band. It is None when rho itself is None, never False -- "undefined"
    and "inside the band" are different findings.
    """
    rho = spearman_rho(x, y)
    n = int(np.asarray(x).size)
    out = {"rho": rho, "n": n, "n_permutations": int(n_perm),
           "null_low": None, "null_high": None, "null_mean": None,
           "exceeds_null": None}
    if rho is None or n < 2:
        return out
    rng = np.random.default_rng(seed)
    ys = np.asarray(y, dtype=np.float64).ravel()
    draws = []
    for _ in range(n_perm):
        r = spearman_rho(x, rng.permutation(ys))
        if r is not None:
            draws.append(r)
    if not draws:
        return out
    lo, hi = np.percentile(draws, NULL_BAND_PERCENTILES)
    out.update(null_low=float(lo), null_high=float(hi),
               null_mean=float(np.mean(draws)),
               exceeds_null=bool(rho < lo or rho > hi))
    return out

def partial_spearman(x, y, z):
    """Spearman rho between `x` and `y` with `z` PARTIALLED OUT, or None.

    T-L6.10. The frequency-depth correlation came out POSITIVE in both smoke runs --
    frequent tokens received MORE recursion, MoR giving L1_FUNCTION 1.85 steps against
    L2_NOUN 1.11 -- which is the opposite direction to the hypothesis the metric exists to
    test. Before that can be interpreted either way, one mundane explanation has to be
    excluded: with a WEIGHT-TIED head, `tok_embed.weight` is also the output projection, so
    a frequent type's row norm grows faster during training, the halt head reads a hidden
    state that still carries that embedding component, and the halt logit becomes
    scale-correlated with frequency BY CONSTRUCTION. That is an initialization-and-training
    artifact, not adaptive computation.

        rho(x,y|z) = (r_xy - r_xz * r_yz) / sqrt((1 - r_xz^2) * (1 - r_yz^2))

    on RANKS, so it is the partial Spearman rather than the partial Pearson. `None` when any
    input has no rank variance or when the denominator vanishes -- a partial correlation
    with a zero denominator is undefined, not 1.0, and the collapsed-depth case reaches it.
    """
    r_xy, r_xz, r_yz = spearman_rho(x, y), spearman_rho(x, z), spearman_rho(y, z)
    if r_xy is None or r_xz is None or r_yz is None:
        return None
    denom = math.sqrt(max(0.0, (1.0 - r_xz ** 2) * (1.0 - r_yz ** 2)))
    if denom <= 1e-12:
        return None
    return float((r_xy - r_xz * r_yz) / denom)


# 50 rather than 10, measured. See `spearman_within_bins` for why the ledger's "or
# equivalently, within norm deciles" is NOT an equivalence.
WITHIN_BIN_DEFAULT = 50


def spearman_within_bins(x, y, z, n_bins=WITHIN_BIN_DEFAULT):
    """Token-weighted mean of rho(x,y) computed WITHIN bins of `z`.

    THE LEDGER CALLS THIS EQUIVALENT TO PARTIALLING OUT. IT IS NOT, AND THE DIFFERENCE IS
    MEASURED. On a synthetic case where the x-y correlation is entirely mediated by `z`
    (raw rho +0.9993, true partial +0.0156), the within-bin mean is:

        n_bins    10      20      50     100     200
        rho     +0.941  +0.807  +0.451  +0.214  +0.106

    Binning removes only BETWEEN-bin variation, so when the mediator is continuous and the
    relationship is tight, the residual within-bin variation still carries the confound. At
    ten bins -- "norm deciles" -- it under-corrects so badly it would pass a fully mediated
    correlation straight through. On a genuine correlation it is stable (+0.978 at 10 bins,
    +0.963 at 200), so it does not over-correct.

    Therefore: `partial_spearman` is the PRIMARY diagnostic and this is a secondary,
    assumption-free cross-check whose known failure mode is under-correction. Its value is
    that it makes no monotonicity assumption where the partial formula does, so a
    disagreement between them is worth investigating rather than either being trusted alone.

    Bins are quantiles of `z`, so they are equal-count rather than equal-width -- the
    embedding-norm distribution is skewed, and equal-width bins would put almost every
    token in one of them.
    """
    xa = np.asarray(x, dtype=np.float64).ravel()
    ya = np.asarray(y, dtype=np.float64).ravel()
    za = np.asarray(z, dtype=np.float64).ravel()
    if not (xa.size == ya.size == za.size) or xa.size < n_bins * 2:
        return None
    edges = np.quantile(za, np.linspace(0.0, 1.0, n_bins + 1))
    idx = np.clip(np.searchsorted(edges[1:-1], za, side="right"), 0, n_bins - 1)
    rhos, weights = [], []
    for b in range(n_bins):
        sel = idx == b
        if int(sel.sum()) < 2:
            continue
        r = spearman_rho(xa[sel], ya[sel])
        if r is not None:
            rhos.append(r)
            weights.append(int(sel.sum()))
    if not rhos:
        return None
    w = np.asarray(weights, dtype=np.float64)
    return {"mean": float((np.asarray(rhos) * w).sum() / w.sum()),
            "n_bins_used": len(rhos),
            "min": float(min(rhos)), "max": float(max(rhos))}


def language_depth_metrics(exit_depth, token_ids, per_token_loss,
                           token_family, log_freq, unigram_surprisal,
                           max_depth, num_families, family_labels,
                           seed=0, n_perm=NULL_PERMUTATIONS,
                           embed_norm=None) -> dict:
    """The five §5.2 depth metrics, each with its permutation null band.

    Args are all over the SAME flat token set, and that is a requirement rather than a
    convenience: T-L6.1 asks for the correlations "computed over the identical token
    set", so a caller that masked one of them differently would be comparing
    populations. The lengths are checked.

        exit_depth        [N] the depth each token actually exited at
        token_ids         [N] vocabulary id per token
        per_token_loss    [N] that token's own cross-entropy, or None
        token_family      [V] int8 lookup, -1 = ignore
        log_freq          [V] log(1 + train count) per type, fixed at build time
        unigram_surprisal [V] -log p_unigram per type, frozen at build time

    `logfreq` and `surprisal` are TYPE-level properties, so their correlations are
    computed over token OCCURRENCES via a gather -- which weights each type by how
    often the model actually met it. The alternative, one point per type, would give a
    hapax the same weight as `the` and would answer a different question.
    """
    ed = np.asarray(exit_depth, dtype=np.float64).ravel()
    ids = np.asarray(token_ids).ravel().astype(np.int64)
    if ed.size != ids.size:
        raise ValueError(
            f"exit_depth ({ed.size}) and token_ids ({ids.size}) must describe the "
            "same tokens; T-L6.1 requires every correlation over one token set."
        )
    out: dict = {"n_tokens": int(ed.size), "max_depth": int(max_depth)}

    # Descriptive, no target. The histogram is over 1..max_depth so a collapsed
    # allocation is visible as a spike rather than inferred from a low variance.
    hist = np.bincount(ed.astype(np.int64), minlength=max_depth + 1)[1:]
    out["hist"] = {str(d + 1): int(hist[d]) for d in range(max_depth)}
    out["mean"] = float(ed.mean()) if ed.size else None
    out["std"] = float(ed.std()) if ed.size else None
    # The number that says whether any correlation CAN be meaningful. A model with
    # one distinct exit depth has no depth variance to correlate with anything, and
    # `spearman_rho` returns None there rather than a number.
    out["distinct_exit_depths"] = int(np.unique(ed).size)

    lf = np.asarray(log_freq, dtype=np.float64)
    us = np.asarray(unigram_surprisal, dtype=np.float64)
    out["spearman_vs_logfreq"] = spearman_with_null(
        ed, lf[ids], n_perm=n_perm, seed=seed)
    out["spearman_vs_unigram_surprisal"] = spearman_with_null(
        ed, us[ids], n_perm=n_perm, seed=seed + 1)

    if per_token_loss is not None:
        pl = np.asarray(per_token_loss, dtype=np.float64).ravel()
        if pl.size != ed.size:
            raise ValueError(
                f"per_token_loss ({pl.size}) does not match exit_depth ({ed.size})."
            )
        out["spearman_vs_model_loss"] = spearman_with_null(
            ed, pl, n_perm=n_perm, seed=seed + 2)
    else:
        out["spearman_vs_model_loss"] = None

    # T-L6.10: the embedding-norm confound, excluded before any depth-allocation claim.
    # With a WEIGHT-TIED head `tok_embed.weight` is also the output projection, so a
    # frequent type's row norm grows faster during training, the halt head reads a state
    # still carrying that component, and the halt logit becomes scale-correlated with
    # frequency by construction. `embed_norm` is per TYPE and gathered by id, so the
    # diagnostic is over the same token occurrences as the raw correlation.
    if embed_norm is not None:
        en = np.asarray(embed_norm, dtype=np.float64)[ids]
        out["embed_norm"] = {
            "depth_vs_norm": spearman_rho(ed, en),
            "logfreq_vs_norm": spearman_rho(lf[ids], en),
            # THE number. Primary diagnostic; see spearman_within_bins for why the
            # binned version is a weaker cross-check rather than an equivalent.
            "logfreq_partial_norm": partial_spearman(ed, lf[ids], en),
            "logfreq_within_norm_bins": spearman_within_bins(ed, lf[ids], en),
            "surprisal_partial_norm": partial_spearman(ed, us[ids], en),
            "norm_min": float(en.min()), "norm_max": float(en.max()),
            "norm_mean": float(en.mean()),
        }
    else:
        out["embed_norm"] = None

    fam = np.asarray(token_family).ravel()[ids]
    by_family = {}
    for f in range(num_families):
        sel = fam == f
        by_family[family_labels[f]] = (float(ed[sel].mean()) if sel.any() else None)
    out["mean_by_family"] = by_family
    # Unmapped tokens get their own entry rather than being folded into a family:
    # -1 is the ignore label, and averaging it into L1 would move a published number.
    _unm = fam < 0
    out["mean_unmapped"] = float(ed[_unm].mean()) if _unm.any() else None
    return out


def document_index_from_ids(token_ids, eot_id: int) -> np.ndarray:
    """`doc_index[N]` -- how many documents began before each token.

    T-L6.7. The encoder inserts `eot_id` BETWEEN documents and not after the last, so the
    document index of a token is the number of EOTs strictly before it. That makes this
    exact rather than a heuristic, and it is derived from the STORED token stream rather
    than from a re-segmentation of the raw text -- which is what T-L6.7 asks for, because
    a second segmentation could drift from the one the packing used.

    The separator token itself is counted as belonging to the document it CLOSES, which is
    the arbitrary half of an otherwise exact definition and is stated rather than hidden:
    it is one token per document, ~0.4% of a 256-token block.
    """
    ids = np.asarray(token_ids).ravel()
    is_eot = (ids == eot_id)
    return (np.cumsum(is_eot) - is_eot).astype(np.int64)


def depth_variance_decomposition(exit_depth, doc_index) -> dict:
    """Split exit-depth variance into WITHIN-document and BETWEEN-document parts.

    T-L6.7, and the point is `between_share`. Per-token recursion was kept (§4.1b) rather
    than moving to segment-level halting, so the fair question is whether the model
    nonetheless spends more computation on harder PASSAGES. The law of total variance
    answers it directly:

        Var(depth) = E[Var(depth | doc)]  +  Var(E[depth | doc])
                     within-document         between-document

    If `between_share` is negligible, the model is not allocating at the passage level and
    that is the honest finding -- it is not evidence against per-token allocation, which
    `depth/spearman_vs_model_loss` measures separately.

    Both parts are POPULATION variances weighted by document token count, so they sum to
    the total exactly (checked by the caller and by `code/test_lang_heads.py`). Using the
    unweighted mean of per-document variances instead would not sum, and the discrepancy
    would look like a bug in whichever term was quoted second.
    """
    d = np.asarray(exit_depth, dtype=np.float64).ravel()
    g = np.asarray(doc_index).ravel().astype(np.int64)
    if d.size != g.size:
        raise ValueError(
            f"exit_depth ({d.size}) and doc_index ({g.size}) must cover the same tokens."
        )
    if d.size == 0:
        return {"n_documents": 0, "total": None, "within": None, "between": None,
                "between_share": None, "per_document_mean_std": None,
                "sum_matches_total": None}
    n_docs = int(g.max()) + 1
    counts = np.bincount(g, minlength=n_docs).astype(np.float64)
    sums = np.bincount(g, weights=d, minlength=n_docs)
    sq = np.bincount(g, weights=d * d, minlength=n_docs)
    seen = counts > 0
    means = np.zeros_like(sums)
    means[seen] = sums[seen] / counts[seen]
    # E[x^2] - (E[x])^2 per document, then token-weighted.
    var_in = np.zeros_like(sums)
    var_in[seen] = np.maximum(sq[seen] / counts[seen] - means[seen] ** 2, 0.0)
    w = counts[seen] / counts[seen].sum()
    grand = float((counts[seen] * means[seen]).sum() / counts[seen].sum())
    within = float((w * var_in[seen]).sum())
    between = float((w * (means[seen] - grand) ** 2).sum())
    total = float(d.var())
    return {
        "n_documents": int(seen.sum()),
        "total": total, "within": within, "between": between,
        "between_share": (between / total) if total > 0 else None,
        "per_document_mean_std": float(means[seen].std()),
        "per_document_mean_min": float(means[seen].min()),
        "per_document_mean_max": float(means[seen].max()),
        # The identity is reported, not assumed: a mismatch means the weighting is wrong,
        # and a silently wrong decomposition is worse than none.
        "sum_matches_total": bool(abs(within + between - total) < 1e-9 * max(1.0, total)),
    }


def depth_by_document_to_wandb(dv: dict, prefix: str = "depth/by_document") -> dict:
    """Flatten the decomposition, `None` omitted.

    Per-document MEANS are not logged individually: there are 61 documents in the
    validation split and 29,445 in train, and 61 keys that no plot reads is clutter that
    makes the three numbers that matter harder to find. The spread, the range and the
    between-share are what the passage-level question is answered with.
    """
    out = {}
    for k in ("n_documents", "total", "within", "between", "between_share",
              "per_document_mean_std", "per_document_mean_min",
              "per_document_mean_max"):
        if dv.get(k) is not None:
            out[f"{prefix}/{k}"] = dv[k]
    if dv.get("sum_matches_total") is not None:
        out[f"{prefix}/sum_matches_total"] = int(dv["sum_matches_total"])
    return out


def language_depth_to_wandb(dm: dict, prefix: str = "depth") -> dict:
    """Flatten `language_depth_metrics` into log keys, `None` OMITTED not zeroed.

    An undefined correlation is absent. A 0.0 in `depth/spearman_vs_model_loss` would
    read as "measured: the model allocates compute unrelated to difficulty", which is
    a finding, not a missing value -- and it is the specific finding the arithmetic POC
    actually made, so the two must stay distinguishable.
    """
    log: dict = {}
    for k in ("n_tokens", "mean", "std", "distinct_exit_depths"):
        if dm.get(k) is not None:
            log[f"{prefix}/{k}"] = dm[k]
    for d, n in dm.get("hist", {}).items():
        log[f"{prefix}/hist/step_{d}"] = n
    for lbl, v in (dm.get("mean_by_family") or {}).items():
        if v is not None:
            log[f"{prefix}/mean_by_family/{lbl}"] = v
    en = dm.get("embed_norm")
    if en:
        for k in ("depth_vs_norm", "logfreq_vs_norm", "logfreq_partial_norm",
                  "surprisal_partial_norm", "norm_min", "norm_max", "norm_mean"):
            if en.get(k) is not None:
                log[f"{prefix}/embed_norm_{k}"] = en[k]
        wb = en.get("logfreq_within_norm_bins")
        if wb:
            for k, v in wb.items():
                log[f"{prefix}/embed_norm_logfreq_within_bins_{k}"] = v
    for key in ("spearman_vs_logfreq", "spearman_vs_unigram_surprisal",
                "spearman_vs_model_loss"):
        rec = dm.get(key)
        if not rec or rec.get("rho") is None:
            continue
        log[f"{prefix}/{key}"] = rec["rho"]
        for f in ("null_low", "null_high", "null_mean"):
            if rec.get(f) is not None:
                log[f"{prefix}/{key}_{f}"] = rec[f]
        if rec.get("exceeds_null") is not None:
            log[f"{prefix}/{key}_exceeds_null"] = int(rec["exceeds_null"])
    return log



# T11.1 (audit item, ex-"item 11"): `evaluate_paper_metrics` USED TO LIVE HERE and
# has been deleted. It was a 109-line second implementation of the validation-pass
# metric accumulation, unreferenced by anything (engine.py imported it and never
# called it), and by the time it was removed its `model(...)` unpack expected 11
# return values while MoREModel returns 13 -- so calling it would have raised
# rather than produced a second opinion. Every number in every run directory came
# from the inline loop in `engine.train()`; that loop is the only implementation.
# If you want an offline metric harness, do NOT resurrect a parallel copy: reuse
# `routing_accuracy_from_confusion`, `permutation_invariant_routing_metrics`,
# `compute_token_exit_depths` and `depth_allocation_error` from this module, which
# is exactly what engine.py calls. The two-copies-that-diverge failure is already
# in changelog.md (the tolerance guard landed in the copy that never ran).


def make_routing_confusion_figure(
    confusion: np.ndarray,
    normalize_rows: bool = True,
) -> plt.Figure:
    """Row-normalized E×E heatmap: actual expert family vs chosen expert.

    T5.1: the axis size is read off the matrix, and the tick labels are derived
    from it, so the figure cannot disagree with the data it is drawing.
    """
    cm = confusion.astype(np.float64)
    if normalize_rows:
        row_sums = cm.sum(axis=1, keepdims=True)
        display = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums > 0)
        vmax = 1.0
        cbar_label = "Row fraction"
        title = "Routing Confusion Matrix (row-normalized)"
    else:
        display = cm
        vmax = None
        cbar_label = "Token count"
        title = "Routing Confusion Matrix (counts)"

    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    im = ax.imshow(display, cmap="Blues", vmin=0.0, vmax=vmax)
    _labels = expert_labels(display.shape[0])
    ax.set_xticks(range(len(_labels)))
    ax.set_yticks(range(len(_labels)))
    ax.set_xticklabels([f"E{i + 1}" for i in range(len(_labels))])
    ax.set_yticklabels(_labels, fontsize=9)
    ax.set_xlabel("Chosen Expert (router at depth 1, block 0)")
    ax.set_ylabel("Actual Operation Family (oracle)")
    ax.set_title(title)

    n = display.shape[0]
    for row in range(n):
        for col in range(n):
            val = display[row, col]
            if cm[row].sum() == 0 and normalize_rows:
                text = "—"
            elif normalize_rows:
                text = f"{val:.0%}" if val >= 0.005 else ""
            else:
                text = f"{int(val)}" if val > 0 else ""
            color = "white" if val > 0.55 else "black"
            ax.text(col, row, text, ha="center", va="center", color=color, fontsize=8)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=cbar_label)
    fig.tight_layout()
    return fig


def make_op_depth_bar_figure(op_avg_depth: dict[str, float]) -> plt.Figure | None:
    """Bar chart of average recursion depth grouped by operation type."""
    if not op_avg_depth:
        return None

    preferred_order = [
        "ADD", "SUB", "MULT", "DIV", "MOD", "POW",
        "AND", "OR", "XOR", "NOT",
        "SHIFT_L", "SHIFT_R",
        "SORT", "MEDIAN", "MAX", "MIN",
    ]
    labels = [op for op in preferred_order if op in op_avg_depth]
    labels += sorted(op for op in op_avg_depth if op not in labels)
    values = [op_avg_depth[op] for op in labels]

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.55), 5))
    bars = ax.bar(labels, values, color="#4C72B0", edgecolor="white")
    ax.set_xlabel("Operation")
    ax.set_ylabel("Average Recursion Depth")
    ax.set_title("Adaptive Compute: Operation vs Avg Recursion Depth")
    ax.set_ylim(0, max(values) * 1.15 if values else 1.0)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.05,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.tight_layout()
    return fig


def paper_metrics_to_wandb(
    confusion: np.ndarray,
    routing_acc: float,
    op_avg_depth: dict[str, float],
    family_avg_depth: dict[str, float],
    num_experts: int,
    pim: dict | None = None,
    task: str = "arithmetic",
) -> dict:
    """
    Build W&B log dict with paper-ready figures and scalar diagnostics.

    Every ROUTING quantity is gated on num_experts >= 2. With one expert (MoR)
    the argmax is 0 for every token, so the 1x1 confusion matrix yields
    recall(E1) = 1.0 and recall(E2..E6) = 0.0 -- which reads in a table as "MoR
    routes E1 perfectly and starves the rest" when in fact MoR performs no
    routing at all. Those are artifacts of the degenerate shape, not
    measurements, and CLAUDE.md 4 forbids reporting them. DEPTH metrics are not
    gated: MoR does allocate depth, and that is exactly its contribution.

    T-L3.2: `task` selects only the NAME the confusion diagonal is published under
    (`routing_agreement_key`), never the arithmetic. It defaults to "arithmetic" so
    every existing caller keeps emitting `val/routing_accuracy` byte-identically --
    Gate L0's 356 includes G5.2b, which checks that exact key.
    """
    log_dict: dict = {}
    routing_defined = num_experts >= 2

    if routing_defined and not math.isnan(routing_acc):
        log_dict[routing_agreement_key(task)] = routing_acc

    if routing_defined:
        row_totals = confusion.sum(axis=1)
        # T5.1: iterate over exactly the experts the matrix HAS. The old loop
        # walked the canonical six and relied on an `i < shape[0]` guard, which
        # silently emitted nothing for expert 7 in a 7-expert run.
        for i, label in enumerate(expert_labels(num_experts)):
            if row_totals[i] > 0:
                log_dict[f"val/routing_recall/{label}"] = (
                    confusion[i, i] / row_totals[i]
                )

        # T6.3: expert indices carry no semantic identity, so the raw diagonal is
        # never the whole story. Logged next to it, never instead of it, so the
        # raw-vs-matched gap stays visible in the run history. The caller passes
        # the dict in when it has already computed it -- the AMI chance correction
        # is the one non-trivial cost here and should be paid once per validation.
        if pim is None:
            pim = permutation_invariant_routing_metrics(confusion, num_experts)
        for key in ("hungarian_accuracy", "macro_recall",
                    "matched_macro_recall", "ami", "purity"):
            if pim[key] is not None:
                log_dict[f"val/routing_{key}"] = pim[key]
        if pim["hungarian_assignment"] is not None:
            # A string, not six numeric keys: the permutation is one fact and
            # each element is meaningless alone.
            log_dict["val/routing_hungarian_assignment"] = "->".join(
                str(v) for v in pim["hungarian_assignment"]
            )
        if pim["collapsed_experts"] is not None:
            log_dict["val/routing_collapsed_experts"] = (
                ",".join(pim["collapsed_experts"]) or "none"
            )
        for label, stats in pim["per_family"].items():
            for stat in ("precision", "f1"):
                if stats[stat] is not None:
                    log_dict[f"val/routing_{stat}/{label}"] = stats[stat]

        # T6.8: the MATCHED per-family table, logged under its own prefix so it
        # can never be confused with the identity one above. On an unsupervised
        # run the identity table describes the router's arbitrary NUMBERING and
        # the matched table describes its PARTITION -- measured, those disagree
        # wildly (macro recall 0.014-0.162 identity vs a stable 0.515 matched),
        # so a reader who picks the wrong prefix draws the opposite conclusion.
        # `_matched` is on the metric name rather than folded into the label,
        # because updated_rules.md 9 forbids '/' inside a label.
        for label, stats in (pim.get("per_family_matched") or {}).items():
            for stat in ("precision", "recall", "f1"):
                if stats.get(stat) is not None:
                    log_dict[f"val/routing_{stat}_matched/{label}"] = stats[stat]

    for op, depth_val in op_avg_depth.items():
        log_dict[f"recursion/avg_depth_by_op/{op}"] = depth_val

    for label, depth_val in family_avg_depth.items():
        log_dict[f"recursion/avg_depth_by_family/{label}"] = depth_val

    if routing_defined:
        cm_fig = make_routing_confusion_figure(confusion, normalize_rows=True)
        log_dict["paper/routing_confusion_matrix"] = wandb.Image(cm_fig)
        plt.close(cm_fig)

        cm_count_fig = make_routing_confusion_figure(confusion, normalize_rows=False)
        log_dict["paper/routing_confusion_matrix_counts"] = wandb.Image(cm_count_fig)
        plt.close(cm_count_fig)

    depth_fig = make_op_depth_bar_figure(op_avg_depth)
    if depth_fig is not None:
        log_dict["paper/op_avg_recursion_depth"] = wandb.Image(depth_fig)
        plt.close(depth_fig)

    if op_avg_depth:
        table = wandb.Table(columns=["operation", "avg_recursion_depth"])
        for op in sorted(op_avg_depth, key=lambda k: op_avg_depth[k]):
            table.add_data(op, op_avg_depth[op])
        log_dict["paper/op_avg_recursion_depth_table"] = table

    return log_dict




# ---------------------------------------------------------------------------
# Halting supervision + depth allocation error  (T3.3, plan.md 5.4 / 12.x)
# ---------------------------------------------------------------------------

def halting_supervision_loss(expected_depth, step_ops, step_mask,
                             target_depth_table, max_depth: int):
    """
    Explicit supervised-halting term against the operation-complexity curriculum.

    plan.md 5.4 requires this be kept SEPARATE from the ponder cost and never
    collapsed into one opaque number, because they pull in opposite directions:
    the ponder cost pushes depth DOWN unconditionally, while this term pushes
    each token toward its own curriculum target. Reporting their sum would hide
    which one the model is actually responding to.

    `expected_depth` is ACT's differentiable N(t) + R(t) per token. The target
    comes from `families.OP_TARGET_DEPTH`, which is a stated design choice, not
    something the model discovers -- see the warning in that module.

    Both sides are divided by max_depth so the loss is scale-free and cannot
    grow simply because max_depth was raised.

    Returns a scalar tensor. Emits an exact differentiable zero when no token
    carries a valid operation code, rather than a NaN from a zero-size mean.
    """
    import torch as _torch

    flat_ops  = step_ops.reshape(-1)
    flat_mask = step_mask.reshape(-1)
    valid     = flat_mask & (flat_ops >= 0)
    if not bool(valid.any()):
        return expected_depth.sum() * 0.0

    table  = _torch.as_tensor(target_depth_table, device=expected_depth.device,
                              dtype=expected_depth.dtype)
    target = table[flat_ops[valid].long()]
    pred   = expected_depth[valid]
    return _torch.nn.functional.mse_loss(pred / max_depth, target / max_depth)


def depth_allocation_error(expected_depth, step_ops, step_mask,
                           target_depth_table):
    """
    Absolute and relative depth allocation error against the curriculum.

    CLAUDE.md 4: call this *depth allocation error*, never "compute efficiency" --
    no FLOPs are in it. It is signed-free (mean absolute) so over- and
    under-allocation cannot cancel out to a flattering zero.

    Returns (absolute, relative) as floats, or (None, None) when no token carries
    a valid operation code, so the caller reports N/A rather than a sentinel 0.0.
    """
    import torch as _torch

    flat_ops  = step_ops.reshape(-1)
    flat_mask = step_mask.reshape(-1)
    valid     = flat_mask & (flat_ops >= 0)
    if not bool(valid.any()):
        return None, None

    table  = _torch.as_tensor(target_depth_table, device=expected_depth.device,
                              dtype=expected_depth.dtype)
    target = table[flat_ops[valid].long()]
    pred   = expected_depth[valid].detach()
    abs_err = float((pred - target).abs().mean().item())
    rel_err = float(((pred - target).abs() / target.clamp(min=1e-6)).mean().item())
    return abs_err, rel_err
