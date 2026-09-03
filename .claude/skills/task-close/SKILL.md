---
name: task-close
description: Close out a completed MoRE task — tick its box in TASKS_LANGUAGE.md with real evidence, append the changelog entry in the house format, and commit ledger plus code together. Use every time a task's Verify criterion has actually passed.
---

# Closing a task

Three things happen together, in one commit: the code, the ledger tick, and the
changelog entry. `TASKS_LANGUAGE.md` T-LX.1 and T-LX.3 require it — a ledger that
lags the code is worse than no ledger, because it is trusted.

## 1. Tick the box — only if Verify actually ran

Read the task's own **Verify:** line in
[TASKS_LANGUAGE.md](../../../TASKS_LANGUAGE.md) and execute exactly that. Then:

| mark | meaning |
|---|---|
| `[x]` | Verify executed and passed. Evidence recorded inline. |
| `[~]` | in progress — partial work landed, Verify not yet green |
| `[ ]` | not started |
| `[!]` | blocked, with the blocker named |
| `[✗]` | deliberately not done, with the reason |

**A box may only be checked when its Verify has actually executed and passed.**
Not "the code looks right", not "the change is obviously correct". If you did not
run it, it is `[~]`.

**Never convert `[✗]` to `[x]`.** A deferred decision that later gets done is a
new task, so the record of the deferral survives.

Add an `Evidence:` line under the box carrying the *number*, not an adjective:

```markdown
- [x] **T-L0.2** Write `ENVIRONMENT.md`.
      Evidence: both interpreters probed 2026-09-03; torch 2.6.0+cpu / 2.6.0+cu126,
      RTX 3050 6143 MiB cc 8.6 driver 596.08; Gate L0 356/356 ALL GATES PASS.
```

## 2. Append the changelog entry

Newest-**last** in [changelog.md](../../../changelog.md), so the file reads in the
order work happened. The house format, all five fields:

```markdown
## T-L0.2 — ENVIRONMENT.md, the machine this repo actually runs on

**What.** One-paragraph statement of the change, at a level a reader can reason
about without opening the diff.

**Method.** How, and *why this way* — including the alternative that was rejected
and the reason. Skip only when the change is a one-liner.

**Failure tracing.** The symptom a future agent will see if this area breaks, and
what it means. This is the field people actually grep.

**Where to look.** The file and symbol to open. `file.py:symbol` or `doc.md §N`.
Every entry names a real code location — this is the whole purpose of the file.

**Verified by.** The command that was run and its actual output. A count, a
verdict, a measured value.
```

Write this at full engineering depth (`CLAUDE.md` §10): file:line references,
exact metric keys, exact config paths, the wrong output verbatim, the rejected
alternative. Under-documenting a trap is the more expensive error. The *chat*
message is where you compress; the files are where you do not.

## 3. Commit

Stage specific files, never `git add .`:

```bash
cd "D:/res/git/MoRE/.claude/worktrees/english-language-dataset-migration-92946e" && git add TASKS_LANGUAGE.md changelog.md <the files the task touched> && git status --short
```

Commit message: `<task id>: <what changed>`, then a body naming the verification.
End with the `Co-Authored-By: Claude <noreply@anthropic.com>` trailer.

Commit only when the user has asked for commits, or when a standing instruction
covers it. Do not push to `main`.

## 4. Re-run Gate L0

Per T-LX.0, after every commit: **356/356**. Use the `gate-check` skill. A
language change that moves an arithmetic gate count has broken something the
arithmetic study published, and that is a STOP.

## What does not go in

- Do not re-run or delete arithmetic artifacts to make a number tidy (T-LX.2).
- Do not rewrite history in `changelog.md`, `TASKS.md`, `results/results.md` or
  `runs/` to match a corrected present. Those record what was true when written;
  corrections go in the *new* entry and reference the old one. `CLAUDE.md` §6:
  archive, never delete, evidence.
- Do not tick a downstream task because an upstream one made it "obviously fine".
