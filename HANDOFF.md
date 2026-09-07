# HANDOFF.md — instructions for the agent driving the canonical language matrix

You are on **Ayan's machine** (RTX 4060 Laptop, 8 GB), the best GPU available to the
project. Everything up to and including **Phase L-7.3 is finished, committed and
pushed, and the protocol is FROZEN.** Your job is one thing: **run the 15-run
canonical matrix, record it, push it.** No design decisions are open. If you find
yourself wanting to change a hyperparameter, read §"Why nothing here is tunable"
before you do.

Read in this order: `SETUP.md` (environment + corpus build), this file, then
`CLAUDE.md`. `TASKS_LANGUAGE.md` is the ledger — the first unticked box is
**T-L8.0**. `plan_language.md` has the design; `updated_rules.md` and
`updated_objective.md` are authoritative and override anything here.

---

## The one command

```bash
C:/Users/Hp/anaconda3/envs/more_env/python.exe code/run_language_matrix.py --preflight
```

```bash
C:/Users/Hp/anaconda3/envs/more_env/python.exe code/run_language_matrix.py --all
```

`--preflight` checks everything that can be wrong before a GPU-hour is spent — CUDA
usable (it runs a real matmul, not just `is_available()`), corpus built, POS family
lookup present, the spec frozen, the frozen config **accepted by the proxy guard on
all three arms**, W&B settled, disk headroom. It refuses rather than warns. Run it
first; it takes about fifteen seconds.

`--all` then runs all 15 runs in order MoE → MoR → MoRE, seeds 42–46, skipping any
that already completed. **It is interruption-safe**: close the laptop, re-run
`--all`, and it picks up where it stopped. A run counts as complete only if its
directory has the right arm, the right seed, the canonical group stamp *and* a
finished `metrics.json`; a directory with a checkpoint but no metrics is treated as
interrupted and re-run, because a checkpoint at an unknown epoch is exactly the kind
of artifact that ends up in a table by accident.

Other entry points: `--smoke` (one short non-canonical run on the dev corpus, to
prove the machine works — ~10 min, cannot contaminate the matrix because wikitext-2
carries its own `dataset_version`), `--arch more --seed 42` (one run),
`--dry-run` (print the commands).

**Settle W&B before you start**, or the first run dies in `wandb.init`:

```bash
export WANDB_MODE=offline
```

Offline is a completely legitimate answer — `runs/<id>/metrics.json` is the record,
W&B is convenience. Preflight refuses if neither a key nor `WANDB_MODE` is present.

---

## Budget

| | per epoch (measured, 6 GB 3050) | × 3 epochs | × 5 seeds |
|---|---|---|---|
| MoE | 0.41 h | 1.2 h | 6.2 h |
| MoR | 1.03 h | 3.1 h | 15.5 h |
| MoRE | 1.16 h | 3.5 h | 17.4 h |
| **total** | | | **≈ 39 h** |

Those are 3050 numbers. **Your 8 GB card should be meaningfully faster** — the 3050
runs the MoR arm at 87% of VRAM, where it is memory-bound rather than
compute-bound. Treat 39 h as an upper bound and record what you actually get.
`run_language_matrix.py` prints elapsed hours as it goes.

One epoch is ~135 M tokens (526,320 blocks × 256). Three epochs is a **fixed
budget, not convergence** — no arm is trained to convergence and the paper must say
so. All three arms get the identical budget, which is what makes the comparison
valid.

---

## The frozen protocol

`code/canonical_spec_language.json` is at `spec_version L7.1-language-frozen` — zero
nulls, and every field carries a `frozen_by` entry naming the measurement that set
it. `code/config_language.json` is the operator-facing defaults file that resolves to
it. Changing a value in either without the other makes the guard refuse every
canonical run, which is the intended direction of failure.

| field | value | why |
|---|---|---|
| `epochs` | 3 | average depth was still moving between epochs 2 and 3 at every calibration point; a 2-epoch matrix would report an unfinished depth allocation |
| `batch_size` | 48 | measured optimum. 32 costs 7 h more; 64 pushes MoR over 6 GB into a soft cliff where its epoch time goes *up* 1.03 → 2.31 h |
| `seq_len` | 256 | the packed length of the corpus; the positional table is sized from it |
| `lr` | 0.001 | best of {5e-4, 1e-3, 2e-3} and the only value stable across both dropout settings |
| `dropout` | 0.1 | a wash on the dev corpus (0.014 nats at one seed); frozen by decision, with a cheap labelled ablation flagged |
| `routing_balance_weight` | 0.001 (MoE, MoRE), 0.0 (MoR) | measured, see below |
| `halting_weight` | 0.001 (MoR, MoRE), 0.0 (MoE) | measured, see below |
| `ffn_mult` | MoE 4, MoR 24, MoRE 4 | re-derived on language, not copied |
| `max_depth` | MoE 1, MoR 7, MoRE 7 | definitional |

### Why nothing here is tunable

The single most expensive mistake available to you is "improving" a loss weight.
Here is the measurement that says so.

The arithmetic study chose `0.001` so the weighted balance term sat at 0.017× the
task loss. Language cross-entropy starts near 9 nats against arithmetic's ~0.06 MSE,
so the **inherited advice was to scale the weight up by ~100× to preserve that
ratio.** That advice was written into the spec's own notes as a prediction. It is
**wrong, and measurably so** — this is a paper-level finding, not a tuning detail:

| balance weight | val loss | AMI (control 0.062) | load entropy |
|---|---|---|---|
| **0.001** | **5.241** | **0.291** | **0.857** |
| 0.05 | 5.299 | 0.307 | — |
| 0.21 *(ratio-matched)* | 5.310 | 0.269 | 0.99+ |
| 1.0 | 5.346 | 0.181 | 0.99+ |

| halting weight | val loss | avg depth | AMI |
|---|---|---|---|
| **0.001** | **5.241** | **2.63** | **0.291** |
| 0.03 | 5.255 | 2.30 | 0.005 |
| 0.107 *(ratio-matched)* | 5.335 | 1.49 | 0.003 |
| 0.5 | 5.331 | 1.07 | 0.021 |

The ratio-matched weights **switch off both behaviours the study exists to
measure** — depth collapses to 1.49 and routing structure vanishes. A matrix run at
those weights would have produced a clean-looking Outcome C that was an artifact of
a loss weight. What transfers between tasks is **absolute gradient magnitude**, not
the ratio to the task loss.

Two secondary readings worth carrying into how you interpret your results:

- 0.001 sits at load entropy 0.857, just *below* the oracle POS partition's own
  0.884. Every higher weight overshoots to 0.99+. Entropy and AMI move in
  **opposite** directions across this sweep — which is exactly why CLAUDE.md §2
  forbids tuning toward maximal entropy.
- Raising the **halting** weight destroys **routing** structure (AMI 0.291 →
  0.003). Depth and expert differentiation are **not independent knobs** in MoRE.

All calibration numbers above are **dev-corpus (wikitext-2)** and may never be
quoted as results. They select; they do not report.

---

## What you are measuring against

**`primary_metric_floor = 4.9849` nats/token** on wikitext-103 val (7.1918 bits,
perplexity 146.2). It is the backoff bigram — the lowest of uniform 9.0109 /
unigram 7.1982 / bigram 4.9849. **A run that does not beat 4.9849 has learned
nothing a two-column count table could not.** The dev-corpus figure is 5.3983 and is
**not** interchangeable with it.

Four things already measured that change how you read your numbers:

- **The oracle POS partition's own normalized load entropy is 0.884, not 1.0**, with
  a 5.29× largest/smallest family token ratio. A router that perfectly reproduced
  POS would score 0.884. Quote every language load-entropy figure against 0.884, not
  against 1.0.
- **AMI has a nonzero floor and it is PER RUN.** Every language `metrics.json`
  carries `val/routing_control_pos/ami_control_mean` and its std beside `ami_real`.
  Use those; do not use a fixed constant. A `delta_z` of `None` means the control had
  no spread, which happens when the prediction is constant.
- **Depth allocation is real and its direction is counter-intuitive.** T-L6.10 is
  closed: the frequency–depth partial correlation *strengthens* across epochs
  (+0.193 → +0.473 at epochs 2/4/6/8), so it is learned, not an embedding-norm
  artifact. **Frequent tokens receive more recursion.** Use
  `depth/embed_norm_logfreq_partial_norm`, which is also a per-epoch column in
  `results.tsv`. Allocation is **within**-passage, not passage-level:
  `depth/by_document/between_share` stays near 0.010 however much total depth grows.
- **A mis-shifted LM loss reads ~5.6 nats at epoch 0** — below the uniform floor and
  close enough to 4.98 to look like fast learning. If an early loss looks too good,
  suspect the shift before believing it.

Also retracted, so you do not repeat them: "depth collapses" was **under-training**,
not the architecture (`avg_depth` climbs 1.96 → 5.16 of a 7 budget over 8 epochs);
and the margin over the floor is not thin (+0.042 nats at one epoch became **+0.666**
at eight).

### Gate L6 is passed and the numbers are pinned

```bash
C:/Users/Hp/anaconda3/envs/more_env/python.exe code/test_language_gate_l6_l7.py
```

| arm | E | depth | ffn_mult | total params | non-embedding |
|---|---|---|---|---|---|
| MoE | 6 | 1 | 4 | 5,584,908 | 3,422,220 |
| MoR | 1 | 7 | 24 | 5,581,063 | 3,418,375 |
| MoRE | 6 | 7 | 4 | 5,584,908 | 3,422,220 |

MoR vs MoRE is 0.069% on the total and 0.112% on the non-embedding count, against a
5% tolerance. **MoE and MoRE are exactly parameter-identical** — weight sharing means
depth costs no parameters, so those two arms differ *only* in how much computation
they do. That is the cleanest comparison in the study and worth stating in the paper
that way. The table is written to `code/lang_param_budget.json`; cite that file, not
this one.

**If your run prints a different parameter count than the table above, stop.** The
arms are not budget-matched and nothing downstream means anything.

The same file also runs T-L7.3, the input-parity invariant (CLAUDE.md §3): all three
arms see bit-identical `input_ids` at the same seed and index, checked at the
dataset tuple, the seeded DataLoader's first batch, and the resolved data config.
28 checks, all passing.

---

## Rules that matter more than speed

1. **Gate L0 is the stop condition.** After **every** commit:

   ```bash
   C:/Users/Hp/anaconda3/envs/more_env/python.exe code/run_correctness_suite.py
   ```

   It must print `TOTAL 356 356 0 0` and `ALL GATES PASS`. Any other count: STOP,
   report the actual output, find the cause, fix, re-run (`plan.md` §20). Never
   revert a correctness fix because the old number looked better. **One caveat for
   your machine specifically:** you are on torch 2.5.1, this baseline was
   established on 2.6.0, and a *different* count is not automatically a regression —
   diagnose it as a version difference before treating it as a code defect, and
   record which it was.

2. **A box is `[x]` only when its own Verify line has actually been executed and
   passed**, with the number recorded inline as Evidence. A ticked box whose
   evidence exists only in chat is the failure mode the ledger exists to prevent.

3. **Report Outcome C honestly.** If MoRE gives no advantage, that is a valid result
   and the paper says so. Do not tune until the desired conclusion appears
   (CLAUDE.md §8). Given the calibration above, an Outcome C at these weights would
   be a *real* Outcome C, which is precisely why the weights were measured first.

4. **Never touch the arithmetic evidence.** `runs/phaseB_*`, `archive/`,
   `data/dataset_meta.json`, `data/*.jsonl` are the published study. Do not re-run,
   delete or move them. Never put an arithmetic number and a language number in the
   same table — one is an MSE, the other a per-token cross-entropy.

5. **A crash mid-matrix is a STOP, not a retry.** The runner stops at the first
   failure and names the run that died, deliberately. Diagnose it; do not loop
   around it. Completed runs are intact and will be skipped when you resume.

---

## Recording results so the paper can cite them

`code/more/run_context.py` enforces this, so mostly you must not fight it. Every run
writes **only** into `runs/<experiment_id>/`: `config.json`, `resolved_config.json`,
`metrics.json`, `results.tsv`, `checkpoint.pt`, `stdout.log`. Never a global
`best_model.pt` / `results.tsv` / `final_run_metrics.json`.

Provenance carries `experiment_id`, `experiment_group`, `architecture`, `variant`,
`seed`, `dataset_version`, `train_split_version`, `code_git_commit`, `config_hash`
and the resolved epochs / subset_fraction. That is what lets a figure be traced to
an exact run six months from now.

**Never hand-copy a number into a table.** After the matrix, one command:

```bash
C:/Users/Hp/anaconda3/envs/more_env/python.exe code/export_results.py --task language
```

It writes `results/language/{results.csv,results_aggregate.csv,results.json,results_tables.md}`
and prints the pairwise verdicts. `--task language` is what selects
`canonical_spec_language.json`, the `canonical_lang_b` group and the nats-based
derived columns; **without it you get the arithmetic matrix in `results/`**, which is
15 rows of mean squared error under a heading that looks like your result. Add
`--check` to see admissions and consistency without writing anything.

`code/seed_stats.py` is a **library, not a command** — the exporter calls its exact
randomization test. Running it directly now prints what to run instead and exits 2.
Until T-L10.0 it exited 0 and printed nothing, and the earlier version of this
section told you to run it; if you have an older checkout in front of you, that is
why.

Report **mean ± std**; one favourable seed is never evidence. Verdicts come from the
exact randomization test, never from a k×std threshold. The exporter refuses to mix
dataset versions or config hashes, refuses a run stamped `task = arithmetic`, and
emits `N/A` rather than fabricating.

Three things it will refuse the whole export for, so they are worth avoiding rather
than debugging at hour 39:

- **more than one `code_git_commit` across the 15 runs.** Pull before you start the
  matrix and then do not pull again until it is finished. A fix landing at hour 20
  splits the matrix across a code change, and the arms stop being comparable.
- **a seed-blind config difference within an arm** — the five seeds of an arm must
  differ *only* in the seed. This is why you use `--all` rather than 15 hand-written
  commands.
- **a floor mismatch**: every run records its own margin over the bigram floor, and
  the exporter checks it reproduces `spec floor − val/task_loss` exactly. The dev
  corpus floor is 5.3983 and the canonical one 4.9849 — a 0.41 nat difference, larger
  than any architecture gap this study can resolve.

`stdout.log` and `*.pt` are git-ignored, so the citable pushed record is
`config.json` + `resolved_config.json` + `metrics.json` + `results.tsv`. If a
`stdout.log` contains something a reader needs, quote it into the changelog rather
than citing the file.

Keep `"N/A"` in `metrics.json` and `results.tsv` exactly as it is. It means "this
quantity does not exist for this configuration" — it is not a gap to fill, and a 0.0
there is a fabricated measurement. On language, `depth/allocation_error_*` is `N/A`
structurally: there is no per-token ground-truth depth for English, so a 0.0 would
read as perfect allocation against a curriculum that does not exist. Dropping dead
columns is the *paper renderer's* job (T-LX.5), not the machine record's.

---

## Pushing back

```bash
git add -A && git commit -m "<what you measured, and the number>"
```

```bash
git push origin claude/english-language-dataset-migration-92946e
```

Ledger + changelog + code in the **same** commit. One `changelog.md` entry per
completed task, appended before `<!-- APPEND-MARKER-CL -->`, naming the code
location to open when that area misbehaves. Push after every gate that a **real run**
validated — not after a gate that only pre-written assertions passed.

For the matrix specifically: push after each **arm** finishes its five seeds, rather
than once at the end. 39 hours is long enough that a disk failure at hour 30 should
not cost all of it.

## If you get stuck

Mark the box `[!]` with the blocker named, move to the next independent task, and
say so in the commit message. Do not silently skip, and do not invent a value to get
past a gate. `changelog.md` records the traps already hit in this codebase — guards
landing in dead code, label checks running before override resolution, sentinels
reported as measurements — and it is worth reading before changing routing, halting,
the loss assembly or the metric layer.

**A note on whose machine a traceback came from.** `C:\Users\Hp\` is yours;
`C:\Users\vedan\` is the other author's, and paths under it appear in committed
notes and in `ENVIRONMENT.md` §1–7 legitimately. Do not "fix" one into the other.
`ENVIRONMENT.md` §8 describes your machine.
