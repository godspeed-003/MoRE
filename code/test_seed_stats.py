# -*- coding: utf-8 -*-
"""test_seed_stats.py - T-LX.18 verification of the statistics layer.

Run:
    C:/Users/vedan/anaconda3/python.exe code/test_seed_stats.py

WHAT IS UNDER TEST, AND WHY IT IS NOT COSMETIC.

`perm_test` returned `min_p = 1 / C(n_a+n_b, n_a)` for every design. The test is
TWO-SIDED on |mean(a) - mean(b)|, and when the two arms are the same size the
complement of any group-a subset is also a valid group-a subset carrying the
SAME absolute difference. Extremes therefore come in mirror pairs, the count at
the extreme can never be 1, and the true floor is 2 / C. The reported floor was
2x too small for every equal-arm comparison in the repository, which is every
headline comparison in both papers.

THE CONSEQUENCE WAS A SILENTLY DEAD GUARD, not a wrong decimal. `format_row` and
`export_results.py` both flag AT-FLOOR with `p_value <= min_p + 1e-12`. At 5v5
that asked `0.00794 <= 0.00397`, which is false by construction, so the flag
could never fire. The two headline language comparisons (MoRE-vs-MoE and
MoR-vs-MoE, both p = 0.00794) sit EXACTLY on the floor and were both printed as
though they had headroom -- i.e. a result at the design's resolution limit was
reported as an ordinary significant result. Absence of evidence rendered as
evidence, the same defect class as reporting a sentinel as a measurement
(CLAUDE.md 4).

WHY THE FIX IS TWO-BRANCHED AND MUST STAY THAT WAY (TLX18d, TLX18e). The
tempting fix is a blanket `2.0 / total`. That is also wrong: at n=3 vs n=5 the
complement of a 3-subset has 5 elements and is NOT an admissible regrouping, so
no mirror is forced and 1/56 really is attainable. A blanket 2x would inflate
the floor for exactly the Phase 10 arms whose verdicts depend on it and mark
them spuriously AT-FLOOR. Both branches are asserted against brute-force
enumeration rather than against the formula, so neither can be "simplified"
back into the other without a failure.

WHY THE FLOOR TABLE IS RE-DERIVED RATHER THAN ASSERTED (TLX18a). The old wrong
numbers propagated into the module docstring, ARCHITECTURE.md, both
canonical_spec files and the paper, and survived there because every copy was
hand-written from the same wrong formula. TLX18a enumerates the most extreme
data each design admits and compares the measured minimum against the docstring
table, so prose and code cannot drift apart again.

WHAT THE HOLM TESTS COVER. Holm-Bonferroni is a step-down procedure, and the two
ways to get it wrong are both tested: stopping at the first non-rejection
(TLX18h -- a naive loop that keeps testing would reject a large p after a small
one failed) and taking the family size from the tests that happened to run
rather than from the declared family (TLX18j -- the degree of freedom
pre-registration exists to remove). TLX18l asserts the undecidable case prints
N/A rather than "not significant", because blaming the data for a limit of the
design is the dishonest reading.
"""

from __future__ import annotations

import itertools
import json
import math
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from seed_stats import (cohens_d, format_row, holm, mean_std,  # noqa: E402
                        perm_test, se_of_difference)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

CODE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(CODE)

PASS, FAIL, SKIP = [], [], []


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))


def skip(name, why):
    SKIP.append(name)
    print(f"  [SKIP] {name}  -- {why}")


def raises(fn, needle):
    """(did it raise?, does the message name `needle`?, the message)."""
    try:
        fn()
    except Exception as e:                      # noqa: BLE001 - type asserted below
        return True, (needle in str(e)), f"{type(e).__name__}: {str(e).splitlines()[0][:110]}"
    return False, False, "no exception"


def brute_force_min_p(na: int, nb: int) -> float:
    """The smallest two-sided p the design admits, by exhaustive search.

    Independent of seed_stats: re-implements the enumeration from scratch over
    maximally separated data, so it cannot inherit the defect under test. With
    arm a strictly above arm b and no overlap, the observed split IS the most
    extreme arrangement, so the p it yields is the attainable minimum.
    """
    a = [float(1000 + i) for i in range(na)]
    b = [float(i) for i in range(nb)]
    pool = a + b
    obs = abs(st.mean(a) - st.mean(b))
    at_least = total = 0
    for idx in itertools.combinations(range(len(pool)), na):
        sel = set(idx)
        ga = [pool[j] for j in idx]
        gb = [pool[j] for j in range(len(pool)) if j not in sel]
        total += 1
        if abs(st.mean(ga) - st.mean(gb)) >= obs - 1e-15:
            at_least += 1
    return at_least / total


print("\n-- TLX18a..c  the floor table is re-derived, not quoted")

# The table the module docstring publishes. If a line here changes, the
# docstring must change in the same commit -- and vice versa.
DOC_TABLE = {(5, 5): 0.00794, (7, 7): 0.00058, (3, 5): 0.0179, (3, 3): 0.1000}

for (na, nb), documented in sorted(DOC_TABLE.items()):
    measured = brute_force_min_p(na, nb)
    check(f"TLX18a docstring floor for n={na} vs n={nb} matches enumeration",
          abs(measured - documented) < 5e-5,
          f"docstring {documented:.5f}, enumerated {measured:.5f}")

for (na, nb) in sorted(DOC_TABLE):
    measured = brute_force_min_p(na, nb)
    reported = perm_test([float(1000 + i) for i in range(na)],
                         [float(i) for i in range(nb)])
    check(f"TLX18b perm_test min_p is attainable at n={na} vs n={nb}",
          abs(reported["min_p"] - measured) < 1e-12,
          f"min_p {reported['min_p']:.6f}, enumerated {measured:.6f}")
    check(f"TLX18c perm_test p equals its own floor on extremal data "
          f"at n={na} vs n={nb}",
          abs(reported["p_value"] - reported["min_p"]) < 1e-12,
          f"p {reported['p_value']:.6f} vs floor {reported['min_p']:.6f}")


print("\n-- TLX18d..f  the equal/unequal asymmetry, both directions")

_eq = perm_test([float(1000 + i) for i in range(5)], [float(i) for i in range(5)])
check("TLX18d equal arms get the MIRRORED floor 2/C, not 1/C",
      abs(_eq["min_p"] - 2.0 / 252) < 1e-12
      and abs(_eq["min_p"] - 1.0 / 252) > 1e-6,
      f"min_p {_eq['min_p']:.6f}; 2/252={2/252:.6f}, 1/252={1/252:.6f}")

_un = perm_test([float(1000 + i) for i in range(3)], [float(i) for i in range(5)])
check("TLX18e unequal arms keep the UNMIRRORED floor 1/C -- a blanket 2x "
      "would break the Phase 10 3v5 arms",
      abs(_un["min_p"] - 1.0 / 56) < 1e-12,
      f"min_p {_un['min_p']:.6f}; 1/56={1/56:.6f}, 2/56={2/56:.6f}")

# n=3 vs n=3 cannot reach alpha = 0.05 even uncorrected. The old docstring said
# 0.0500 and blamed "correction"; the truth is the design is undecidable at 0.05.
_33 = brute_force_min_p(3, 3)
check("TLX18f n=3 vs n=3 cannot reach alpha=0.05 at all, corrected or not",
      _33 > 0.05, f"floor {_33:.4f} > 0.05")


print("\n-- TLX18g  the AT-FLOOR flag fires on the real language numbers")

# The five published per-seed val/task_loss values per arm (runs/langB_*).
_real = perm_test([3.5031 - 0.0133, 3.5031, 3.5031 + 0.0133, 3.5031, 3.5031],
                  [3.9529 - 0.0396, 3.9529, 3.9529 + 0.0396, 3.9529, 3.9529])
check("TLX18g a maximally-separated 5v5 is flagged AT-FLOOR in format_row",
      "AT-FLOOR" in format_row("more_vs_moe", _real),
      format_row("more_vs_moe", _real).strip()[:150])
check("TLX18g2 that same comparison was NOT flagged under the old 1/C floor",
      _real["p_value"] > 1.0 / _real["n_perms"] + 1e-12,
      f"p {_real['p_value']:.6f} > old floor {1.0/_real['n_perms']:.6f}")


print("\n-- TLX18h..k  Holm-Bonferroni step-down")

def _arm(mean, spread=0.01):
    return [mean - spread, mean - spread / 2, mean, mean + spread / 2, mean + spread]

_big = perm_test(_arm(1.0), _arm(2.0))      # hugely separated -> p at floor
_mid = perm_test(_arm(1.0, 0.30), _arm(1.22, 0.30))
_null = perm_test(_arm(1.0), _arm(1.0005))  # overlapping -> p near 1

fam = holm({"big": _big, "mid": _mid, "null": _null}, alpha=0.05, family_size=4)
check("TLX18h Holm thresholds step down as alpha/(m-rank)",
      abs(fam["big"]["holm_threshold"] - 0.05 / 4) < 1e-12,
      f"rank0 thr {fam['big']['holm_threshold']:.5f} (expect {0.05/4:.5f})")
check("TLX18h2 Holm stops at the first non-rejection and retains every "
      "larger p",
      fam["big"]["holm_reject"] and not fam["null"]["holm_reject"],
      f"big reject={fam['big']['holm_reject']}, null reject={fam['null']['holm_reject']}")

# A naive implementation that keeps testing after a failure would reject `null`
# if its own threshold happened to be loose. Assert monotonicity explicitly.
_ranked = sorted((r["holm_rank"], r["holm_reject"]) for lb, r in fam.items()
                 if lb != "_family" and r is not None)
_seen_fail = False
_mono = True
for _, rej in _ranked:
    if not rej:
        _seen_fail = True
    elif _seen_fail:
        _mono = False
check("TLX18i no test is rejected after an earlier one failed (step-down "
      "monotonicity)", _mono, f"sequence {[r for _, r in _ranked]}")

_did, _named, _msg = raises(
    lambda: holm({"a": _big, "b": _mid, "c": _null}, family_size=2),
    "smaller than")
check("TLX18j holm refuses a family_size smaller than the tests supplied",
      _did and _named, _msg)

fam_none = holm({"a": _big, "b": None}, alpha=0.05, family_size=4)
check("TLX18k a non-computable comparison still consumes its declared slot",
      fam_none["b"] is None and fam_none["_family"]["family_size"] == 4
      and abs(fam_none["a"]["holm_threshold"] - 0.05 / 4) < 1e-12,
      f"m={fam_none['_family']['family_size']}, "
      f"computable={fam_none['_family']['n_computable']}, "
      f"thr={fam_none['a']['holm_threshold']:.5f}")


print("\n-- TLX18l..m  an undecidable design reports N/A, never 'not significant'")

# Plain Bonferroni over 7 comparisons gives 0.00714, BELOW the 5v5 floor of
# 0.00794: no effect of any size could be called significant.
fam7 = holm({f"t{i}": perm_test(_arm(1.0), _arm(1.0 + 0.5 * (i == 0)))
             for i in range(7)}, alpha=0.05, family_size=7)
_worst = fam7["t0"]
check("TLX18l a 5v5 floor above its Holm threshold is marked undecidable",
      _worst["floor_above_threshold"] is True,
      f"floor {_worst['min_p']:.5f} vs thr {_worst['holm_threshold']:.5f}")
_line = format_row("t0", _worst)
check("TLX18m format_row prints N/A for it, not a significance verdict",
      "N/A" in _line and "not significant" not in _line, _line.strip()[:150])

_line_ok = format_row("mid", fam["mid"])
check("TLX18m2 a decidable Holm row still prints a verdict with its threshold",
      "Holm" in _line_ok and "N/A" not in _line_ok, _line_ok.strip()[:150])

_line_raw = format_row("raw", _mid)
check("TLX18m3 a row that never went through holm() is marked UNCORRECTED",
      "UNCORRECTED" in _line_raw, _line_raw.strip()[:150])


print("\n-- TLX18n..q  the pre-registered family file is self-consistent")

_pre_path = os.path.join(CODE, "confirmatory_tests.json")
if not os.path.isfile(_pre_path):
    skip("TLX18n confirmatory_tests.json exists", "file missing")
else:
    pre = json.loads(open(_pre_path, encoding="utf-8").read())
    check("TLX18n confirmatory_tests.json parses and declares Holm",
          pre.get("correction") == "holm-bonferroni" and pre.get("alpha") == 0.05,
          f"correction={pre.get('correction')}, alpha={pre.get('alpha')}")

    fams = pre.get("families", {})
    check("TLX18o exactly the two declared families are present",
          set(fams) == {"A_predictive_quality", "B_adaptive_depth_mechanism"},
          f"{sorted(fams)}")

    _bad = [k for k, v in fams.items()
            if int(v.get("family_size", -1)) != len(v.get("tests", []))]
    check("TLX18p every family_size equals its number of declared tests",
          not _bad, f"mismatched: {_bad}" if _bad else
          ", ".join(f"{k}={v['family_size']}" for k, v in sorted(fams.items())))

    # Every declared 5v5 slot must be decidable at its own easiest threshold,
    # or the family is too large for the design and the paper cannot use it.
    _undecidable = []
    for k, v in fams.items():
        m = int(v["family_size"])
        for t in v["tests"]:
            n = int(t.get("n_per_arm", 0))
            if n < 2:
                continue
            floor = 2.0 / math.comb(2 * n, n)       # equal arms, mirrored
            if floor > 0.05 / m:
                _undecidable.append(f"{k}/{t['id']}")
    check("TLX18q no declared family is too large for its own resolution floor",
          not _undecidable,
          f"undecidable: {_undecidable}" if _undecidable else
          f"5v5 floor {2.0/math.comb(10,5):.5f} < easiest thr "
          f"{0.05/max(int(v['family_size']) for v in fams.values()):.5f}")

    _ids = [t["id"] for v in fams.values() for t in v["tests"]]
    check("TLX18q2 test ids are unique across both families",
          len(_ids) == len(set(_ids)), f"{len(_ids)} ids, {len(set(_ids))} unique")


print("\n-- TLX18r..t  the untouched helpers still behave (regression guard)")

check("TLX18r mean_std returns None, not 0.0, when nothing is numeric",
      mean_std([None, "N/A"]) is None, "None")
check("TLX18r2 mean_std leaves std None at n=1 rather than reporting 0.0",
      mean_std([1.5]) == {"mean": 1.5, "std": None, "n": 1}, f"{mean_std([1.5])}")
check("TLX18s se_of_difference is sqrt(s_a^2/n_a + s_b^2/n_b)",
      abs(se_of_difference([1.0, 2.0, 3.0], [1.0, 3.0, 5.0])
          - math.sqrt(1.0 / 3 + 4.0 / 3)) < 1e-12,
      f"{se_of_difference([1.0,2.0,3.0],[1.0,3.0,5.0]):.6f}")
check("TLX18s2 cohens_d returns None on zero pooled spread, not inf",
      cohens_d([2.0, 2.0, 2.0], [1.0, 1.0, 1.0]) is None, "None")
check("TLX18t perm_test returns None when an arm has n<2",
      perm_test([1.0], [1.0, 2.0]) is None, "None")
check("TLX18t2 format_row emits N/A for an untestable pair",
      "N/A" in format_row("x", None), format_row("x", None).strip()[:90])
check("TLX18t3 gap sign follows mean(a) - mean(b)",
      perm_test(_arm(2.0), _arm(1.0))["gap"] > 0
      and perm_test(_arm(1.0), _arm(2.0))["gap"] < 0, "both signs correct")


print(f"\n{len(PASS)} passed, {len(FAIL)} failed, {len(SKIP)} skipped")
if FAIL:
    print("FAILED:")
    for n in FAIL:
        print(f"  - {n}")
sys.exit(1 if FAIL else 0)
