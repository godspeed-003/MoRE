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
The smallest attainable p is 1 / C(n_a + n_b, n_a):

    n=5 vs n=5  ->  1/252  = 0.0040   (clears Bonferroni for 3 or 6 comparisons)
    n=3 vs n=5  ->  1/56   = 0.0179   (does NOT clear Bonferroni at 6 arms)
    n=3 vs n=3  ->  1/20   = 0.0500   (cannot clear alpha = 0.05 after correction)

So a Phase 10 arm at n=3 that reports p = 0.0179 is AT the floor: it is as
extreme as the design can show, and adding seeds is the only way to strengthen
it. Report the floor alongside the p-value; `perm_test` returns it as `min_p`.

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

__all__ = ["mean_std", "se_of_difference", "perm_test", "cohens_d", "format_row"]


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

    Returns `min_p`, the floor 1 / C(n_a+n_b, n_a). A p equal to min_p means the
    observed split is the single most extreme arrangement, i.e. the design is at
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
    return {
        "gap": st.mean(a) - st.mean(b),
        "p_value": at_least / total,
        "min_p": 1.0 / total,
        "n_perms": total,
        "se_diff": se_of_difference(a, b),
        "cohens_d": cohens_d(a, b),
        # The frozen `seed_reporting` necessary condition, kept so it can still
        # be checked -- NOT the verdict. See the module docstring.
        "gap_over_std": (abs(st.mean(a) - st.mean(b))
                         / max(st.stdev(a), st.stdev(b))),
    }


def format_row(label: str, res: dict | None, alpha: float = 0.05) -> str:
    """One fixed-width line. Emits N/A rather than a fabricated verdict."""
    if res is None:
        return f"{label:34s}  n<2 in one arm -- no test is possible (N/A)"
    at_floor = " AT-FLOOR" if res["p_value"] <= res["min_p"] + 1e-12 else ""
    verdict = ("significant" if res["p_value"] < alpha else "not significant")
    return (f"{label:34s} gap={res['gap']:+.6f} "
            f"d={res['cohens_d']:+.2f} p={res['p_value']:.4f} "
            f"(floor {res['min_p']:.4f}{at_floor})  {verdict}")

