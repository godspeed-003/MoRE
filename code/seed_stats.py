"""Seed-level statistics for the MoRE results layer.

WHY THIS MODULE EXISTS
----------------------
Until the T11.0b audit every verdict in this repository was read off a
`k x std` heuristic: take `max(std_a, std_b)` over the per-seed values and call
the difference real at `>= 2 x`. That denominator is wrong. `max(std)` is the
spread of ONE arm's seeds; the quantity a difference of means must be compared
against is the standard error OF THE DIFFERENCE,

    se = sqrt(s_a**2 / n_a + s_b**2 / n_b)

which at n = 5 per arm is about sqrt(5) smaller. Concretely, for
MoRE - MoR on `val/task_loss` the heuristic printed "1.93x seed std" and called
it noise; the same gap is 3.94 standard errors and the exact test gives
p = 0.0159. Two Phase 10 arms and one Phase 9 pair were being suppressed.

Rather than pick a new `k`, use an exact randomization test. With five seeds per
arm there are only C(10,5) = 252 ways to split the pooled values, so the null
distribution can be enumerated exactly -- no threshold to choose, no normality
assumption, and it is the test a reviewer will ask for.

RESOLUTION LIMITS -- read these before quoting a p-value
--------------------------------------------------------
THE FLOOR IS NOT 1 / C(n_a + n_b, n_a) WHEN THE ARMS ARE THE SAME SIZE. It was
reported that way here until T-LX.18, and the wrong value reached
`ARCHITECTURE.md`, both `canonical_spec*.json` protocol notes, the paper, and a
reviewer, who caught it.

The test is TWO-SIDED on |mean(a) - mean(b)|. When n_a == n_b, the complement of
any group-a subset is itself a valid group-a subset, and swapping the two groups
negates the difference while preserving its absolute value. So every arrangement
is paired with a mirror of identical statistic, the count at the extreme can
never be 1, and the floor is 2 / C. When n_a != n_b the complement has the wrong
size and is not an admissible regrouping, so no mirror is forced and the floor
really is 1 / C.

    n=5 vs n=5  ->  2/252  = 0.00794  (clears Bonferroni at 6; margin 0.0004)
    n=7 vs n=7  ->  2/3432 = 0.00058  (clears Bonferroni at 7 and beyond)
    n=3 vs n=5  ->  1/56   = 0.0179   (does NOT clear Bonferroni at 6 arms)
    n=3 vs n=3  ->  2/20   = 0.1000   (cannot reach alpha = 0.05 AT ALL, even
                                       uncorrected -- the old table said 0.0500
                                       and "after correction", both too kind)

Verify, do not trust this table -- `test_seed_stats.py` re-derives every row by
enumerating the most extreme data the design admits.

So a Phase 10 arm at n=3 vs n=5 that reports p = 0.0179 is AT the floor: it is
as extreme as the design can show, and adding seeds is the only way to
strengthen it. Report the floor alongside the p-value; `perm_test` returns it as
`min_p`. The 5-vs-5 language cells at p = 0.00794 are likewise AT the floor --
before T-LX.18 they printed without the AT-FLOOR marker, because the detector
compared them against a floor that was 2x too small to ever be reached.

MULTIPLE COMPARISONS
--------------------
A per-test alpha = 0.05 is not a verdict when several comparisons are read off
one results table. Use `holm()` over a family DECLARED BEFORE THE RESULTS EXIST
(`code/confirmatory_tests.json`). Holm-Bonferroni is uniformly more powerful
than plain Bonferroni and controls the same familywise error rate, so there is
no reason to use the plain version: at m = 4 its easiest threshold is
alpha/4 = 0.0125 against a 5-vs-5 floor of 0.00794, a margin of 0.0046, where
plain Bonferroni at m = 7 gives 0.00714 -- BELOW the floor, i.e. a design that
cannot produce a significant result however large the effect.

Do NOT use the paired sign-flip variant at five seeds: it enumerates 2**5 = 32
sign assignments, so its minimum two-sided p is 2/32 = 0.0625 and it can never
reach alpha = 0.05. Pairing is weakly motivated here anyway -- the train/val
split is fixed by file (`train_split_version = fixed-file-splits-v1`), so a seed
varies only initialization and shuffle order, not the data.

`canonical_spec.json:protocol.seed_reporting` freezes "a difference smaller than
the seed std is not a difference". That is a NECESSARY condition and this module
still reports it (`gap_over_std`) so the frozen rule can be checked, but the
verdict comes from `p_value`.
"""

from __future__ import annotations

import itertools
import math
import statistics as st

__all__ = ["mean_std", "se_of_difference", "perm_test", "cohens_d", "holm",
           "format_row"]


def mean_std(values) -> dict | None:
    """Mean and SAMPLE (n-1) std of the numeric entries, or None.

    Returns None when nothing numeric is present, so a metric that is "N/A" for
    an architecture prints N/A instead of inventing 0.0 (CLAUDE.md 4). `std` is
    None at n = 1: one seed has no spread, and 0.0 would read as perfect
    agreement.
    """
    nums = [v for v in values if isinstance(v, (int, float))
            and not isinstance(v, bool)]
    if not nums:
        return None
    m = st.mean(nums)
    if len(nums) < 2:
        return {"mean": m, "std": None, "n": 1}
    return {"mean": m, "std": st.stdev(nums), "n": len(nums)}


def se_of_difference(a: list, b: list) -> float | None:
    """sqrt(s_a^2/n_a + s_b^2/n_b) -- the denominator the k x std rule got wrong.

    Returned for reporting only; the verdict comes from `perm_test`, which needs
    no variance estimate at all.
    """
    if len(a) < 2 or len(b) < 2:
        return None
    return math.sqrt(st.variance(a) / len(a) + st.variance(b) / len(b))


def cohens_d(a: list, b: list) -> float | None:
    """Standardised effect size on the pooled SD. Sign follows mean(a) - mean(b).

    Report this NEXT TO the p-value, never instead of it, and next to the raw gap
    -- |d| here reaches 4.1 on differences of 0.0005, which is 3% of the variance
    the models explain. A large d on a tiny gap is exactly the case a reader must
    not be allowed to misread.
    """
    if len(a) < 2 or len(b) < 2:
        return None
    pooled = math.sqrt((st.variance(a) + st.variance(b)) / 2)
    if pooled == 0.0:
        return None
    return (st.mean(a) - st.mean(b)) / pooled


def perm_test(a: list, b: list) -> dict | None:
    """Exact two-sided randomization test on the difference of means.

    Enumerates every way of splitting the pooled `a + b` values into groups of
    the original sizes and counts the fraction whose |mean difference| is at
    least the observed one. Exhaustive, so the result is deterministic -- there
    is no sampling error and no seed.

    Returns `min_p`, the smallest p the two arm sizes can produce. That is
    2 / C(n_a+n_b, n_a) for equal arms and 1 / C(n_a+n_b, n_a) otherwise -- see
    the module docstring for why the two cases differ. A p equal to min_p means
    the observed split is as extreme as the design admits, i.e. the design is at
    its resolution limit and only more seeds can strengthen the claim.

    `n_perms` is C(n_a+n_b, n_a); it is 252 at 5-vs-5 and 3432 at 7-vs-7, so
    exhaustive enumeration is cheap for every design this project uses.
    """
    if len(a) < 2 or len(b) < 2:
        return None
    obs = abs(st.mean(a) - st.mean(b))
    pool = list(a) + list(b)
    n = len(a)
    idx_all = range(len(pool))
    at_least = 0
    total = 0
    for idx in itertools.combinations(idx_all, n):
        sel = set(idx)
        ga = [pool[j] for j in idx]
        gb = [pool[j] for j in idx_all if j not in sel]
        total += 1
        # -1e-15 so the observed arrangement itself always counts, and so float
        # round-trips of an identical regrouping are not dropped.
        if abs(st.mean(ga) - st.mean(gb)) >= obs - 1e-15:
            at_least += 1
    # Equal arms: the complement of a group-a subset is also a group-a subset and
    # has the same |mean difference|, so extremes come in mirror pairs and the
    # count can never be 1. Unequal arms: the complement has the wrong size, is
    # not an admissible regrouping, and no mirror is forced. Do NOT "simplify"
    # this to a single branch -- a blanket 2/total inflates the floor for the
    # n=3-vs-5 Phase 10 arms and makes them spuriously AT-FLOOR, and a blanket
    # 1/total is the T-LX.18 defect this comment exists to prevent recurring.
    mirrored = 2.0 if len(a) == len(b) else 1.0
    return {
        "gap": st.mean(a) - st.mean(b),
        "p_value": at_least / total,
        "min_p": mirrored / total,
        "n_perms": total,
        "se_diff": se_of_difference(a, b),
        "cohens_d": cohens_d(a, b),
        # The frozen `seed_reporting` necessary condition, kept so it can still
        # be checked -- NOT the verdict. See the module docstring.
        "gap_over_std": (abs(st.mean(a) - st.mean(b))
                         / max(st.stdev(a), st.stdev(b))),
    }


def holm(tests: dict, alpha: float = 0.05, family_size: int | None = None) -> dict:
    """Holm-Bonferroni over a DECLARED family of comparisons.

    `tests` maps a label to a `perm_test` result (or None when an arm had n < 2).
    Returns the same labels mapped to the input dict plus `holm_threshold`,
    `holm_reject`, `holm_rank`, `floor_above_threshold` and `at_floor`.

    Procedure: sort the computable p-values ascending and compare the i-th
    (0-based) against alpha / (m - i). Reject while that holds; at the first
    failure, stop and retain that test and every larger p. The step-down is what
    makes Holm uniformly more powerful than plain Bonferroni (which would use
    alpha/m for all m) while controlling the same familywise error rate.

    `m` IS THE DECLARED FAMILY SIZE, NOT THE NUMBER OF TESTS THAT RAN. If a
    comparison in the pre-registered family could not be computed, its slot is
    still consumed. Shrinking m after seeing the data is exactly the degree of
    freedom pre-registration exists to remove, so dropping a failed arm must not
    be allowed to make the survivors easier to call significant. Pass
    `family_size` explicitly from `confirmatory_tests.json`; it defaults to
    `len(tests)` and MUST NOT be smaller than it (that would be a correction
    weaker than the number of comparisons actually read).

    `floor_above_threshold` is the one every reader of this repo must check: it
    is True when `min_p > holm_threshold`, meaning the design CANNOT produce a
    significant result for that comparison however large the true effect. At
    5-vs-5 the floor is 0.00794, so a family of 7 corrected at 0.05/7 = 0.00714
    is unachievable by construction -- the test is not weak evidence, it is no
    test at all, and must be reported as N/A rather than "not significant"
    (CLAUDE.md 4: no sentinel may be reported as a measurement).
    """
    m = len(tests) if family_size is None else int(family_size)
    if m < len(tests):
        raise ValueError(
            f"declared family_size={m} is smaller than the {len(tests)} "
            "comparisons supplied. Holm would then correct for fewer tests than "
            "are being read off the table, which inflates the familywise error "
            "rate. Declare the full family in code/confirmatory_tests.json."
        )
    out = {label: (dict(res) if res is not None else None)
           for label, res in tests.items()}
    computable = sorted(((res["p_value"], label) for label, res in tests.items()
                         if res is not None),
                        key=lambda t: t[0])
    still_rejecting = True
    for rank, (p, label) in enumerate(computable):
        thr = alpha / (m - rank)
        if p > thr:
            still_rejecting = False
        row = out[label]
        row["holm_rank"] = rank
        row["holm_threshold"] = thr
        row["holm_reject"] = still_rejecting
        row["at_floor"] = p <= row["min_p"] + 1e-12
        row["floor_above_threshold"] = row["min_p"] > thr
    out["_family"] = {"alpha": alpha, "family_size": m,
                      "n_computable": len(computable),
                      "n_rejected": sum(1 for _, lb in computable
                                        if out[lb]["holm_reject"])}
    return out


def format_row(label: str, res: dict | None, alpha: float = 0.05) -> str:
    """One fixed-width line. Emits N/A rather than a fabricated verdict.

    When `res` carries Holm fields (i.e. it came through `holm()`), the verdict
    is the FAMILYWISE one and the threshold is printed. Without them the line is
    marked UNCORRECTED, because a bare per-test alpha read off a table of
    several comparisons is not a familywise verdict and must not look like one.
    """
    if res is None:
        return f"{label:34s}  n<2 in one arm -- no test is possible (N/A)"
    at_floor = " AT-FLOOR" if res["p_value"] <= res["min_p"] + 1e-12 else ""
    if "holm_threshold" in res:
        thr = res["holm_threshold"]
        if res.get("floor_above_threshold"):
            # Reporting "not significant" here would blame the data for a limit
            # of the design. The comparison was never decidable.
            verdict = (f"N/A -- floor {res['min_p']:.5f} exceeds Holm threshold "
                       f"{thr:.5f}; undecidable at this n")
        else:
            verdict = (("significant" if res["holm_reject"]
                        else "not significant")
                       + f" (Holm, thr {thr:.5f})")
    else:
        verdict = (("significant" if res["p_value"] < alpha
                    else "not significant") + " (UNCORRECTED)")
    return (f"{label:34s} gap={res['gap']:+.6f} "
            f"d={res['cohens_d']:+.2f} p={res['p_value']:.4f} "
            f"(floor {res['min_p']:.5f}{at_floor})  {verdict}")


if __name__ == "__main__":
    # THIS MODULE IS A LIBRARY AND HAS NO CLI, ON PURPOSE: it computes a test from
    # two lists of numbers and knows nothing about run directories, admission or
    # provenance. Reading runs/ is `export_results.py`'s job, and it is the only
    # sanctioned path from a run directory to a number in a table.
    #
    # This block exists because the alternative is WORSE THAN AN ERROR. Before it,
    # `python code/seed_stats.py --group canonical_lang_b` -- a command that was
    # written into HANDOFF.md and printed by run_language_matrix.py -- exited 0 and
    # printed absolutely nothing. An operator who had just spent 39 GPU-hours would
    # have read that silence as "no differences found". A non-zero exit naming the
    # right command cannot be misread.
    import sys as _sys

    print(__doc__.strip().splitlines()[0])
    print()
    print("code/seed_stats.py is a LIBRARY, not a command. It has no CLI and")
    print("running it computes nothing. To aggregate runs into a results table:")
    print()
    print("    python code/export_results.py                    # arithmetic matrix")
    print("    python code/export_results.py --task language    # language matrix")
    print()
    print("That exporter calls perm_test()/mean_std() from this file, applies the")
    print("admission and consistency rules, and writes results/ (results/language/")
    print("for --task language). Its stdout carries the same pairwise verdict lines")
    print("this module formats.")
    if len(_sys.argv) > 1:
        print()
        print(f"ignored arguments: {_sys.argv[1:]}")
    raise SystemExit(2)

