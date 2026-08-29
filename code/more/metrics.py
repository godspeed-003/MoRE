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

def compute_expert_load_entropy(
    depth_exits: torch.Tensor,
    all_expert_idx: list,
    num_experts: int,
) -> torch.Tensor:
    """
    Normalized Shannon entropy of expert utilisation, H / log(E) in [0, 1].

    This is a LOAD-BALANCE DIAGNOSTIC ONLY. High entropy means routing is
    uniform, i.e. no expert is starved; it is NOT evidence of specialization
    and must never be tuned toward (CLAUDE.md 2, 4). Returns None when E < 2.
    Uses hard argmax assignments, not soft probabilities.
    """
    counts = torch.zeros(num_experts, device=depth_exits.device)
    for idx_tensor in all_expert_idx:
        for e in range(num_experts):
            counts[e] += (idx_tensor == e).sum().float()
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


def routing_accuracy_from_confusion(confusion, num_experts: int) -> float:
    """
    THE authoritative routing accuracy. Every caller must use this function.

    Defined as mean(predicted_expert == oracle_expert) over exactly the tokens
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
# second pass over the loader. No scipy/sklearn in this environment (checked),
# so the assignment and the AMI correction are implemented here rather than
# approximated -- a greedy match is not the Hungarian match and must not be
# labelled as one.


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
    """
    log_dict: dict = {}
    routing_defined = num_experts >= 2

    if routing_defined and not math.isnan(routing_acc):
        log_dict["val/routing_accuracy"] = routing_acc

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
