---
name: research-writing
description: The two-register rule for this repository — full engineering depth in files and the paper, researcher-level abstraction in chat messages to the user. Use whenever writing a chat message that reports work done, run results, a defect, or a decision; and whenever writing a changelog entry, docstring, ledger Evidence line, or paper text. Also carries the progress-report template for autonomous loops.
---

# Two registers, one standard of rigour

`CLAUDE.md` §10 is the rule. This skill is the operational form of it: what to
actually type, with the self-check that catches the failure mode.

The failure mode is **not** being too technical. It is being technical about the
wrong axis — spending the reader's attention on build mechanics when the thing
they have to decide is architectural. Stated by the user, 2026-09-06:

> "as an AI Researcher I'll like to know what's going on in high level of
> engineering but lower level of actual architecture underneath so that I can make
> big decisions without clouding my memory too much with engineering choices"
>
> "I won't have to focus on `stdout.log carried 281 copies of _histc_cuda does not
> have a deterministic implementation` and can talk about main decision like how we
> earlier talked about cls_loss and about MoR fixes."

Read that as an axis swap, not a volume cut. Depth on **architecture, objective,
measurement validity, and what a number licenses us to claim** stays. Depth on
**plumbing** moves to the files.

---

## 1. Files and the paper — unchanged, maximum depth

Applies to: `changelog.md`, `TASKS*.md` Evidence lines, code comments and
docstrings, `ARCHITECTURE.md`, `HANDOFF.md`, `canonical_spec*.json` notes,
`results_exp.md`, and the paper draft.

Keep all of it:

- `file.py:symbol` / `file.py:line` references
- exact metric keys (`val/routing_control_pos/ami_control_mean`), exact config
  paths (`loss_weights.routing_balance_weight`)
- the wrong output verbatim, including the sentinel that was mistaken for a
  measurement
- the alternative fix that was rejected, and why
- hashes, counts, seeds, run-directory names, dataset versions

Rationale: the reader of a file is a future agent or a future self debugging a bad
number six months out, with no conversational context. Under-documenting a trap
costs more than over-documenting one. **This skill never licenses trimming a
file.**

---

## 2. Chat to the user — researcher register

### The shape of a good message

1. **The finding or the decision, first sentence.** What is now known that was not
   known before, or what was chosen.
2. **The number that supports it**, with its uncertainty or its comparison point.
   A number without a baseline is not a finding.
3. **What it implies** — for the architecture, for the claims the paper can make,
   or for the next experiment.
4. **The open decision**, if there is one, with the options and their costs.
5. *Optional, one line:* mechanism, only when the user needs it to decide.

### Belongs in chat

- the scientific result and the comparison that makes it a result
- what it means for MoE vs MoR vs MoRE, for the objective, for outcome A/B/C
- a defect's **consequence** ("those 14 runs can't be cited, ~2 h lost") — not its
  fix
- wall-clock and GPU cost when it changes a choice
- decisions taken autonomously, stated as taken, so they can be overruled cheaply
- tables of *results*

### Does not belong in chat

- log lines, warning text, stack traces, line counts
- `file:line` unless the user is about to open that line
- metric-key strings, config field paths, config hashes, run-directory names
- how a fix was implemented, unless the user must pick between fixes
- restating what a test asserts — say it passed and what it covers
- interpreter paths, background task ids, file sizes, commit SHAs

### Worked contrasts

> ✗ "`stamp_language_dataset_versions` in `config.py` is called twice because
> `cli.py` applies `--corpus` after `apply_task`; without the second call a
> `--corpus wikitext-2` run carries `lang-wikitext-103-bpe8192-len256-800d6154`."
>
> ✓ "Every language run so far recorded the *arithmetic* dataset id in its
> provenance while actually reading WikiText. Fixed. Consequence: the 14 exploratory
> language runs on disk can't be quoted in the paper as-is — no numbers are wrong,
> the label is. The canonical matrix hasn't started, so nothing published is
> affected."

> ✗ "`--halting_weight 0.107` yields `val/task_loss 5.3345`, `depth/avg 1.49`,
> `routing/ami 0.0025` at `logging.log_interval=2`."
>
> ✓ "The 0.017× loss-weight rule we inherited from arithmetic breaks on language:
> at that halting weight, depth collapses to 1.5 of a 7 budget and routing structure
> goes to zero — it switches off both behaviours the study exists to measure. 0.001
> on both terms wins on validation loss *and* on routing structure. The invariant
> was absolute gradient magnitude, not the ratio to task loss."

Both ✓ versions are longer in ideas and shorter in tokens the user has to hold.

---

## 3. Self-check before sending a chat message

Four questions. Any "no" means rewrite, not append.

1. **Does the first sentence say what was learned or chosen?** Not what was worked
   on.
2. **Could every remaining number be checked against a baseline by the reader?**
   Strip numbers that only describe plumbing.
3. **Is every identifier that survives one the user would type or open?** File
   paths, hashes, task ids, metric keys otherwise go.
4. **Would this message change how they think about the architecture or the
   paper?** If it only proves diligence, compress it to one line of status.

---

## 4. Progress reports in an autonomous loop

When the user is away and asked to be kept updated, each message is a checkpoint,
not a log. Structure:

**Done since last message** — one line per unit of work, phrased as outcomes:
what is now true, not what commands ran.

**Results** — a table if there are numbers. Every column needs a comparison point
(a floor, a baseline, a previous value).

**Decisions I made for you** — anything frozen, chosen, or rejected without
asking, each with the one-line reason. State them as decided and overrulable. This
is the section that keeps autonomy honest: a decision buried in a file is a
decision the user cannot revisit.

**Blocked / needs you** — only genuine blockers. Do not park work here that could
proceed under a stated assumption.

**Next** — the immediate next step, one line.

Report failures with the same prominence as successes. `CLAUDE.md` §8: outcome C
is a valid result and must be reported honestly; a gate that failed is stated in
the first sentence, not in a closing caveat.

---

## 5. When the register inverts

**A mechanism question gets the mechanism, at full depth.** "Why is the GPU idle",
"how does the halting gradient reach the halt head", "why does `config_hash` not
cover logging" — answer those completely, with the code path. §2 sets the default
for *unprompted* reporting; it is not a ceiling on answering.

Likewise, if the user is about to run or open something, give the exact path, the
exact command, the exact field name. Precision is not the thing being trimmed —
irrelevance is.
