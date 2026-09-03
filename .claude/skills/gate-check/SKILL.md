---
name: gate-check
description: Run the MoRE correctness suite (Gate L0) or one individual gate, interpret the output against the frozen 356-check baseline, and follow the plan.md §20 failure protocol. Use before and after every code change, and whenever a gate count or a [FAIL] needs interpreting.
---

# Running a gate

## The command

Full suite — this is **Gate L0**, the regression fence for the language migration:

```bash
cd "D:/res/git/MoRE/.claude/worktrees/english-language-dataset-migration-92946e/code" && "C:/Users/vedan/anaconda3/python.exe" run_correctness_suite.py
```

Use the **CPU** interpreter (`C:/Users/vedan/anaconda3/python.exe`). The gate
counts were measured on it and the gates do not need CUDA. See
[ENVIRONMENT.md](../../../ENVIRONMENT.md).

Runtime is roughly 3–6 minutes: gates 1–4 are fast, **gate 5 launches two real
1-epoch training runs** (T6.7c `more`, T6.7d `mor`). Run it in the background and
do other work while it goes.

One gate at a time, when iterating on a specific area:

```bash
cd "D:/res/git/MoRE/.claude/worktrees/english-language-dataset-migration-92946e/code" && "C:/Users/vedan/anaconda3/python.exe" test_phase6_provenance.py
```

Each suite file is standalone and exits non-zero on any `[FAIL]`.

## The baseline

**`TOTAL 356 356 0 0` — 5 gates, ALL GATES PASS.** Anything else needs an
explanation before you continue.

| number | meaning |
|---|---|
| 356 | checks on this machine, established in T-L0.3 |
| 350 | the count at the close of the arithmetic study, and the number still quoted in older notes |
| +1 | gate 4: the seeding suite now completes instead of crashing at check 51 (T-L0.3a) |
| +5 | gate 5: the new T6.7d single-expert provenance producer (T-L0.3b) |

A gate that reports **zero** assertions **FAILS** by design — `run_correctness_suite.py`
treats an empty gate as a broken gate, not a passing one. If a count *drops*, first
suspect that a suite file stopped being discovered or crashed before its first
`[PASS]`, not that assertions were legitimately removed.

`SKIP` is not `PASS`. Gate 5's G5.2b skips rather than passes when no
single-expert run at HEAD exists; the skip message names the commit of the
historical run it declined to grade.

## Reading the output

The runner scans for `^[ \t]*\[(PASS|FAIL|SKIP)\]`. Two failure shapes:

- **`[FAIL] <name> (<detail>)`** — a real assertion failure. The detail is the
  measured value. Trust it.
- **"output format not recognised"** — the suite file *crashed* before emitting a
  marker. There is no `[FAIL]` line to read. Re-run that one file directly and
  read the traceback; this is how the cross-drive `relpath` defect presented.

## When a gate fails — plan.md §20, in order

1. **STOP.** Do not start the next task.
2. **Report the failure** with the actual output, not a paraphrase.
3. **Identify the cause.**
4. **Fix it.**
5. **Re-run the gate.**

Never revert a correctness fix because the old number looked better. Never
proceed past a failing gate.

### Before you conclude the code is wrong

Two defects in this repo were *test-selection* bugs, not code bugs, and both
looked exactly like a real regression:

- **Is the test grading a run produced by the code under test?** `runs/` is
  tracked, so `git worktree add` rewrites every directory mtime into checkout
  order. Selecting "the newest run" by `os.path.getmtime` can hand a test a run
  from before the fix it is testing. Select on
  `provenance.code_git_commit == git_commit()`.
- **Is a path crossing a drive letter?** `os.path.relpath` *raises* on Windows
  when the two paths are on different mounts. The repo is on `D:`;
  `tempfile.mkdtemp()` returns `C:`.

**Filesystem metadata is not experiment metadata.** An mtime or a drive letter is
a property of the substrate, and using one as evidence is the same error class as
reporting a sentinel as a measurement.

## After the run

- Record the count in the task's Evidence line in
  [TASKS_LANGUAGE.md](../../../TASKS_LANGUAGE.md) — the number and the verdict,
  e.g. `356/356 ALL GATES PASS`.
- Gate 5 leaves `runs/t67_provenance_check_seed44__*` and
  `runs/t67d_provenance_check_mor_seed44__*` behind. That is expected: they are
  1-epoch smoke runs, they are **not** canonical, and they must never enter a
  results table. Do not delete them (`CLAUDE.md` §6: archive, never delete) and
  do not clean `runs/` to tidy up.
