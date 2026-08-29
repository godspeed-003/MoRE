"""
run_correctness_suite.py -- T7.2 (plan.md Phase 7 Step 10).

ONE runnable command that collects the Gate 1-5 assertions and prints a
per-gate pass table. Exit code is 0 only if every gate passed.

    python run_correctness_suite.py            # all gates
    python run_correctness_suite.py --gate 5   # one gate
    python run_correctness_suite.py -v         # stream child output

WHY A SUBPROCESS RUNNER AND NOT pytest / import-and-call
--------------------------------------------------------
Each suite is a standalone script that seeds global RNGs, mutates
torch.backends flags, builds CUDA models and calls sys.exit(). Importing them
into one process would let one suite's determinism settings silently decide
another suite's verdict -- exactly the class of cross-contamination Gate 4
(seeding) exists to rule out. A subprocess per suite keeps each verdict a
statement about that suite alone.

WHY THE ORDER IS FIXED AND NOT ALPHABETICAL
-------------------------------------------
test_gate5.py reads a real run directory out of runs/ and asserts against its
metrics.json and resolved_config.json. test_phase6_provenance.py is what
produces that directory. Running Gate 5 first makes it either skip its
provenance checks or judge a stale directory from an earlier code version, so
provenance is pinned ahead of it inside the same gate.

CONTRACT WITH THE CHILD SUITES
------------------------------
Each suite prints one line per assertion beginning "[PASS]" or "[FAIL]" (and
"[SKIP]" where a check has no data to run against), and exits non-zero if any
assertion failed. This runner reads those markers rather than re-implementing
the assertions, so there is exactly one definition of each check and this file
can never drift from the gate it reports on.

Two details of that contract are load-bearing and were both got wrong on the
first attempt:

  * The marker may be INDENTED. The Phase 2-5 suites print theirs at column 0
    but the Phase 3 halting suite indents some inside grouped sections, and an
    anchored r"^\\[" counted zero checks for four of the five gates.
  * A gate that counts ZERO assertions must FAIL, never pass. With the anchor
    bug above, gates 1-4 each printed "0 checks ... PASS", which is the same
    defect as reporting a sentinel as a measurement: an absence of evidence
    rendered as evidence. Zero counted checks now means the contract broke.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time

CODE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(CODE)

# gate number -> (title, [suite files in execution order])
GATES: dict[int, tuple[str, list[str]]] = {
    1: ("Routing is Top-1 sparse dispatch, not dense blending",
        ["test_phase2_routing.py"]),
    2: ("Halting is differentiable ACT with a real halt gradient",
        ["test_phase3_halting.py"]),
    3: ("Balance loss is depth- and block-normalized",
        ["test_phase4_balance.py"]),
    4: ("Every tensor width derives from num_experts / num_families; "
        "seeding is global",
        ["test_phase5_dimensions.py", "test_phase6_seeding.py"]),
    5: ("Phase 6 closure: provenance, permutation-invariant routing metrics, "
        "no sentinels",
        ["test_phase6_provenance.py", "test_phase6_routing_metrics.py",
         "test_gate5.py"]),
}

MARKER = re.compile(r"^[ \t]*\[(PASS|FAIL|SKIP)\]", re.MULTILINE)


def run_suite(path: str, verbose: bool) -> dict:
    """Run one suite as a child process and count its assertion markers."""
    t0 = time.perf_counter()
    env = dict(os.environ)
    # The suites are diagnostics, never canonical runs: no W&B traffic, and a
    # stable hash seed so a dict-ordering difference cannot change a verdict.
    env["WANDB_MODE"] = "disabled"
    env.setdefault("PYTHONHASHSEED", "0")
    proc = subprocess.run(
        [sys.executable, path],
        cwd=CODE, env=env, capture_output=True, text=True,
        encoding="utf-8", errors="replace",
    )
    out = (proc.stdout or "") + (proc.stderr or "")
    if verbose:
        print(out)
    counts = {"PASS": 0, "FAIL": 0, "SKIP": 0}
    for m in MARKER.finditer(out):
        counts[m.group(1)] += 1
    return {
        "file": os.path.basename(path),
        "rc": proc.returncode,
        "elapsed": time.perf_counter() - t0,
        "output": out,
        **counts,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="T7.2 correctness suite: Gates 1-5 in one command")
    ap.add_argument("--gate", type=int, action="append", choices=sorted(GATES),
                    help="Run only this gate (repeatable). Default: all.")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="Stream each suite's full output as it runs.")
    args = ap.parse_args(argv)

    wanted = sorted(set(args.gate)) if args.gate else sorted(GATES)

    print("=" * 78)
    print("CORRECTNESS SUITE (T7.2) -- Gates " +
          ", ".join(str(g) for g in wanted))
    print(f"python: {sys.executable}")
    print(f"cwd:    {CODE}")
    print("=" * 78)

    results: dict[int, list[dict]] = {}
    for gate in wanted:
        title, files = GATES[gate]
        print(f"\n--- GATE {gate}  {title}")
        results[gate] = []
        for fname in files:
            path = os.path.join(CODE, fname)
            if not os.path.exists(path):
                # A missing suite is a FAILURE, not a skip: the gate's evidence
                # does not exist, so the gate cannot be reported as passing.
                print(f"    {fname:34s}  MISSING")
                results[gate].append({"file": fname, "rc": 127, "PASS": 0,
                                      "FAIL": 1, "SKIP": 0, "elapsed": 0.0,
                                      "output": "file not found"})
                continue
            print(f"    {fname:34s}  running ...", end="", flush=True)
            r = run_suite(path, args.verbose)
            results[gate].append(r)
            print(f"\r    {fname:34s}  {r['PASS']:3d} pass  "
                  f"{r['FAIL']:3d} fail  {r['SKIP']:2d} skip  "
                  f"rc={r['rc']}  {r['elapsed']:6.1f}s")

    # ---- per-gate table --------------------------------------------------
    print("\n" + "=" * 78)
    print("PER-GATE PASS TABLE")
    print("=" * 78)
    print(f"{'gate':<5} {'checks':>7} {'pass':>6} {'fail':>6} {'skip':>6} "
          f"{'time':>8}  verdict  description")
    print("-" * 78)

    all_ok = True
    for gate in wanted:
        rs = results[gate]
        p = sum(r["PASS"] for r in rs)
        f = sum(r["FAIL"] for r in rs)
        s = sum(r["SKIP"] for r in rs)
        t = sum(r["elapsed"] for r in rs)
        # A gate passes only if no assertion failed AND every child exited 0
        # AND it actually counted at least one assertion. The three are checked
        # separately on purpose: a suite that dies on an exception before its
        # last check prints no [FAIL] line at all, and a suite whose marker
        # format this runner fails to recognise prints none either -- counting
        # markers alone would read both as clean passes.
        ok = (f == 0) and all(r["rc"] == 0 for r in rs) and (p > 0)
        all_ok &= ok
        verdict = "PASS " if ok else "FAIL "
        if p + f == 0:
            verdict = "EMPTY"
        print(f"{gate:<5} {p + f:>7} {p:>6} {f:>6} {s:>6} {t:>7.1f}s  "
              f"{verdict}   {GATES[gate][0][:34]}")
    print("-" * 78)
    tp = sum(r["PASS"] for rs in results.values() for r in rs)
    tf = sum(r["FAIL"] for rs in results.values() for r in rs)
    ts = sum(r["SKIP"] for rs in results.values() for r in rs)
    print(f"{'TOTAL':<5} {tp + tf:>7} {tp:>6} {tf:>6} {ts:>6}")

    # ---- failure detail --------------------------------------------------
    # Printed AFTER the table so the table is never scrolled off by a
    # traceback, and only for suites that actually failed (CLAUDE.md 7: report
    # the failure plainly, with the actual output).
    if not all_ok:
        print("\n" + "=" * 78)
        print("FAILURE DETAIL")
        print("=" * 78)
        for gate in wanted:
            for r in results[gate]:
                if r["FAIL"] == 0 and r["rc"] == 0 and r["PASS"] > 0:
                    continue
                print(f"\n-- gate {gate} / {r['file']} (rc={r['rc']}, "
                      f"{r['PASS']} pass / {r['FAIL']} fail)")
                for line in r["output"].splitlines():
                    if line.strip().startswith("[FAIL]"):
                        print("   " + line)
                if r["FAIL"] == 0:
                    print("   no [FAIL] marker was printed -- the suite either "
                          "crashed or its output format is not recognised. "
                          "Tail of its output:")
                    for line in r["output"].splitlines()[-25:]:
                        print("   | " + line)

    print("\n" + "=" * 78)
    print(f"CORRECTNESS SUITE: {'ALL GATES PASS' if all_ok else 'GATE FAILURE'}")
    print("=" * 78)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
