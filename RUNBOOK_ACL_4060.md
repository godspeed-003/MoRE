# RUNBOOK_ACL_4060.md — the 8-day plan for the ACL submission, on Ayan's 4060

**Deadline: ACL, 12th. Today is the 4th.** This file is the execution plan for the window.
It is written for an agent session on **Ayan's machine** and assumes no prior context.
Read [PAPER_GUIDE.md](PAPER_GUIDE.md) first for the repo map, the authoritative numbers and
the claim ledger; this file is only *what to run, in what order, and why*.

**Interpreter:** `C:\Users\Hp\anaconda3\envs\more_env\python.exe`. Prefix every command with
`PYTHONIOENCODING=utf-8`. Use absolute paths. No rented GPU and no spend — everything here
runs on the local 4060 8 GB.

---

## 1. The one fact that determines the whole plan

**Only the five MoRE checkpoints survive. MoE and MoR language weights are gone** — they
were trained on the first rented V100 and were never copied off before the instance was
released. Verify it yourself in one command:

```bash
PYTHONIOENCODING=utf-8 C:/Users/Hp/anaconda3/envs/more_env/python.exe -c "import glob,os,json; print(sum(os.path.exists(os.path.join(d,'checkpoint.pt')) for d in glob.glob('runs/langB_*') if os.path.exists(os.path.join(d,'resolved_config.json')) and json.load(open(os.path.join(d,'resolved_config.json'),encoding='utf-8')).get('provenance',{}).get('experiment_group')=='canonical_lang_b'), 'of 15 canonical cells have a checkpoint')"
```

Expect `5 of 15`.

> **Superseded as of T-LA.1 (2026-10-04): MoE has been retrained on this 4060 and all five
> of its cells now carry `checkpoint.pt`.** The count above reads `11 of 21` while both the
> V100 and 4060 MoE cells are on disk, and `10 of 16` after the V100 MoE cells are archived.
> MoE reproduced at 3.9483 ± 0.0242 against the published 3.9529 ± 0.0396 (−0.117 σ), so the
> §5 reproduction gate has **passed for MoE**. Only the MoR arm still needs weights.

**Consequence.** Every evaluation worth adding to this paper is a *checkpoint* evaluation —
held-out test split, difficulty-stratified loss, out-of-domain perplexity. None of them can
be run for MoE or MoR until those two arms are retrained. So the critical path is: retrain
two arms → run the evals on all three → write.

The published `val/task_loss` numbers for MoE and MoR remain valid and are **not** being
replaced. The retrain exists to recover *weights*, and a re-run at the same frozen config
should land within seed noise of the published mean. **If it does not, that is a finding and
you stop** (`CLAUDE.md` §7) — do not quietly adopt the new numbers.

---

## 2. Budget, and why this fits

Published V100 training-loop hours per cell, from `RUNBOOK_V100.md`: MoE 0.61 h, MoR 2.52 h,
MoRE 2.60 h. The 4060 is Ada `sm_89` with comparable FP32 throughput to a V100 but it is a
*laptop* card with thermal throttling, and the numbers above exclude validation, the
depth/routing metric pass and checkpoint writes. Budget **1.3× for the card and +25% for the
non-training passes**:

| arm | est. h / cell | × 5 seeds | note |
|---|---|---|---|
| MoE | ~1.0 | **~5 h** | cheapest, do first |
| MoR | ~4.1 | **~20 h** | the long pole |
| MoRE | ~4.2 | ~21 h | **not needed — checkpoints already exist** |

**Plan A (recommended): MoE + MoR only ≈ 25 h.** Over 8 days that is ~3–4 h/day, or two
unattended overnight sessions. Leaves 5 days of writing time.

**Plan B (only if Plan A finishes by the 7th): also re-run MoRE**, giving a single-card,
single-commit 15-cell matrix. Adds ~21 h. Scientifically cleaner but it is a *nice-to-have*,
and burning the writing window on it would be the wrong trade with a hard deadline.

`batch_size` stays at **48**. It is frozen in `canonical_spec_language.json`, it was tuned
for exactly this 8 GB card (T-L7.0/T-L7.1), and changing it moves `config_hash` and makes
the new cells incomparable with the kept ones. Do not raise it to use spare VRAM.

---

## 3. The schedule

| day | work | GPU |
|---|---|---|
| **4th** | Gate L0 must pass. Launch MoE (5 cells, ~5 h). Start writing methods — it does not depend on any run. | 5 h |
| **5th** | Verify MoE reproduced within seed noise (§5). Launch MoR (~20 h, let it run overnight into the 6th). | 20 h |
| **6th** | MoR finishes. Re-export. Run the three evaluations in §6. | ~1 h |
| **7th** | Results and mechanism sections written against the real numbers. Optional: Plan B. | — |
| **8th–10th** | Full draft: intro, related work, discussion, limitations. | — |
| **11th** | Internal review against the §12 forbidden-sentence checklist in `PAPER_GUIDE.md`. Regenerate every table. | — |
| **12th** | Submit with a buffer. Do not plan to finish on the 12th. | — |

**Write the methods section on the 4th, in parallel with the MoE run.** It is the largest
section that depends on nothing: architecture definitions, the ACT forward path
(`PAPER_REVISION.md` §1 is still the valid agenda for this and is the paper's single biggest
blocking item), the statistics layer, and the pre-registration. Do not wait for runs.

---

## 4. Launching the retrains

Gate first — if this is not 357/357 the numbers are not trustworthy and nothing else matters:

```bash
PYTHONIOENCODING=utf-8 C:/Users/Hp/anaconda3/envs/more_env/python.exe code/run_correctness_suite.py
```

> **Expect `TOTAL 357 357 0 0`, not 356.** One check was added after this file was first
> written (measured 2026-10-04, T-LA.1). A count *above* 357 with 0 fail / 0 skip is
> likewise fine; any non-zero fail or skip column is a STOP.

Then set the environment once per shell:

```bash
export WANDB_MODE=offline && export WANDB_SILENT=true
```

**MoE is already retrained and reproduced** (T-LA.1: 3.9483 ± 0.0242 against the published
3.9529 ± 0.0396, a −0.117 σ difference). All five cells carry `checkpoint.pt`. Do not re-run
it. The line below is kept only for the case where those cells are lost again:

```bash
PYTHONIOENCODING=utf-8 C:/Users/Hp/anaconda3/envs/more_env/python.exe code/run_language_matrix.py --arch moe --force
```

Then MoR. **`--force` is mandatory and `--dry-run` first is mandatory:**

```bash
PYTHONIOENCODING=utf-8 C:/Users/Hp/anaconda3/envs/more_env/python.exe code/run_language_matrix.py --arch mor --dry-run
```

```bash
PYTHONIOENCODING=utf-8 C:/Users/Hp/anaconda3/envs/more_env/python.exe code/run_language_matrix.py --arch mor --force
```

### Why `--force`, and why the plain form is a silent no-op

**An earlier revision of this section told you to run `--arch mor` with no flags. That
trains nothing and exits 0.** `run_language_matrix.py:finished_run()`
(code/run_language_matrix.py:284-311) decides completeness from `metrics.json` +
`resolved_config.json` + canonical group + arch + seed + a non-null `val/task_loss`.
**`checkpoint.pt` is not one of the clauses.** The V100 MoR cells lost *only* their weights,
so they satisfy every clause and all five seeds print

    skip  mor seed 42  -- already complete: langB_MoR_seed42__547e435b__r4
    ...
    nothing to do; every requested run is already complete.

The runner is interruption-safe in the *opposite* direction — it re-runs any directory
holding a checkpoint without a `metrics.json`, because a checkpoint at an unknown epoch is
exactly the artifact that ends up in a table by accident. The weights-lost case is that
case's mirror and the completeness test does not cover it. Always `--dry-run` first and read
the skip lines before committing 15 h.

**Use `--force` rather than archiving the V100 cells first.** Archiving also unblocks the
runner, but it removes the admitted MoR arm from `runs/` for the whole retrain, during which
the exporter cannot regenerate the published table at all. `--force` writes the new cells
alongside and the duplicate state is resolved afterwards (§4 below).

Long runs belong in a **separate Windows Terminal tab**, not in an agent tool call: a 15 h
job outlives any single call, and the operator needs to be able to read it. `_launch_mor_retrain.ps1`
in the repo root is the launcher — it sets `PYTHONIOENCODING`, `WANDB_MODE=offline` and
`WANDB_SILENT` (the Terminal tool types one literal PowerShell line into a fresh tab, so
Git-Bash `export VAR=... && python ...` does not work and env vars do not persist across
tabs) and tees to `runs/_mor_retrain_console.log`.

**Do not pass `--arch more` and do not pass `--all`.** MoRE's five cells are complete, they
carry their checkpoints, and a re-run produces a second cell for the same `(arch, seed)` —
which the exporter rejects as a duplicate. That refusal already cost one debugging cycle;
see §Phase 6 of `RUNBOOK_V100.md`.

### The duplicate-cell trap, which you will hit

The new MoE/MoR cells will have the same `(arch, seed)` as the published ones. Two
canonical cells per seed makes `export_results.py` refuse the whole export with
`duplicate cell ('moe', 42)` and write nothing. That guard is correct — it is what stops
two different runs being averaged into one row.

Retire the superseded directories into `archive/`, and **use `git mv`, never plain `mv`**:

```bash
mkdir -p archive/pre_finalization/lang_v100_no_checkpoints
```

```bash
git mv runs/langB_MoE_seed42__1958b9a5 archive/pre_finalization/lang_v100_no_checkpoints/
```

`mv` removes the directory from the working tree but leaves it in the index at the old path,
so `git add <newpath>` stages the copy without the deletion and the commit carries the cell
at **both** paths. The local export still succeeds and the duplication only surfaces on the
next machine to pull. Confirm the deletions are staged before committing:

```bash
git status --short | grep "^D"
```

> **This check reports 0 after a correct `git mv`, and that is not a failure.** When the
> content is byte-identical git records a **rename** (`R old -> new`) rather than a
> delete-plus-add, which carries the same guarantee the check exists to verify: the old path
> leaves the index. Measured at T-LA.1 — the five MoE cells staged as 20 `R` entries and 0
> `D` entries. Use this instead, expecting one line per moved file:
>
> ```bash
> git status --short | grep -cE "^R"
> ```
>
> The real `mv` trap shows up as a surviving `runs/...` path in `git status`, so the
> unambiguous test is that no retired `runs/` path appears there at all.

**The MoE half of this is already done** (T-LA.1): the five V100 MoE cells are in
`archive/pre_finalization/lang_v100_no_checkpoints/`. The MoR half is still pending and must
happen after the retrain completes, not before.

Archive, never delete (`CLAUDE.md` §6). The V100 cells are the provenance record for the
numbers currently in the paper.

---

## 5. The reproduction check — this is a gate, not a formality

Before using any new cell, confirm the retrain reproduced. Published means:
**MoE 3.9529 ± 0.0396**, **MoR 3.5202 ± 0.0406** on `val/task_loss`.

```bash
PYTHONIOENCODING=utf-8 C:/Users/Hp/anaconda3/envs/more_env/python.exe code/export_results.py --task language --check
```

A new arm mean inside ±1 published std is a reproduction — say so in the paper and move on.
**Outside ±2 std, stop.** A V100-to-4060 difference that large is not numerics; it means a
config, data or code difference, and finding it is more important than the deadline. Report
it with the actual numbers (`CLAUDE.md` §7).

**One caveat you must carry into the paper.** `perf/throughput_items_sec` for the retrained
arms is measured on a 4060 while MoRE's is on a V100, so the three columns are no longer
one hardware platform. Fix it cheaply rather than caveating it — there is a harness:

```bash
PYTHONIOENCODING=utf-8 C:/Users/Hp/anaconda3/envs/more_env/python.exe code/measure_lang_throughput.py
```

Measure all three arms on the 4060 and report the cost table from that single-card
measurement, with the training-time throughputs cited separately as per-run provenance.
`val/task_loss` is not hardware-dependent beyond numerics, so the quality table is unaffected.

---

## 6. The evaluations — your best shot at a positive finding

> **Division of labour changed 2026-10-04 (T-LA.1): every evaluation in this section now
> runs on Vedant's 3050, not here.** This 4060 is occupied by the MoR retrain, and the two
> jobs cannot share one card without contending for VRAM and corrupting any timing number.
> That is the reason the canonical checkpoints are force-added into git (see `.gitignore`
> §"Per-run output directories"): the weights have to *travel*, and they are the only
> artifact that does not regenerate from a config. MoE's five are pushed; MoR's follow when
> the retrain lands. Benchmarks and specialized tests are Vedant's machine's work — do not
> start them here.
>
> **BLiMP specifically is in scope on that machine and is not caught by the warning below.**
> The ban is on MMLU/HellaSwag/ARC/PIQA, which need world knowledge and sit at chance at this
> scale. BLiMP scores a *relative* preference inside a matched minimal pair, so it degrades
> gracefully rather than collapsing — its own paper's baselines include an n-gram model above
> 50%. Two conditions on it, both load-bearing: (a) the BPE-8192 WikiText vocab tokenizes the
> two members of a pair to *different lengths* whenever they differ at rare morphology, and a
> summed log-prob then measures token count rather than grammaticality — score only the
> diverging positions, and report how many pairs fail an equal-length assertion; (b) BLiMP is
> **not** in the pre-registered families and `code/confirmatory_tests.json` forbids adding a
> family after results exist, so it reports uncorrected effect sizes, labelled exploratory,
> and stays out of every significance claim in the abstract.

This is the answer to "where can MoRE shine." **Not** on general benchmarks. On the thing
its architecture is actually for.

### Why not the frontier-model benchmark suite

At 5.6M parameters, an 8192-token BPE vocabulary and wikitext-103 only, MMLU, HellaSwag,
ARC, PIQA and the rest sit at chance. GPT-2 small at 124M — twenty-two times larger — is
already near-chance on MMLU. Three arms reporting three chance-level scores produces
differences that are noise, and a reviewer reads it as padding. **Do not run them.** The
honest sentence is that the scale precludes zero-shot task evaluation, stated once in
limitations.

### What does work, and matches the architecture's premise

Your System-1 intuition translates precisely: do not measure general capability, measure
**the mechanism's own premise**. MoRE claims to spend more compute where a token is harder.
So the evaluation that can make it shine is **per-stratum loss**, and the question becomes:
*does the adaptive-depth arm's advantage concentrate on the hardest tokens?* A model can be
worse on average and better where its mechanism is supposed to help — and that is a real,
reportable, mechanism-matched result.

**The assets are already built.** `data/lang/wikitext-103/` contains `token_decile.npy`
(frequency deciles — the difficulty axis), `token_train_count.npy`, `token_family.npy` (POS
families), `token_family_shuffled.npy` (the permutation control) and `block_topic_val.npy`.
Nothing new needs constructing.

Run these three, in this order of value:

**(a) Difficulty-stratified validation loss, per arm.** Per-token loss grouped by
`token_decile.npy`, reported as a 10-row × 3-column table. This is the headline candidate.
Three outcomes, all publishable: MoRE's deficit is uniform across deciles (the mechanism
does nothing); it shrinks on rare tokens (the mechanism works but is outweighed); or it
reverses on the hardest decile (the mechanism works and the mean hides it). Pair it with
`depth/hist` per decile so you can say whether depth actually varied by stratum — and
remember that with 97.9% of MoRE tokens at step 7 the honest finding may be that it could
not have varied.

**(b) Held-out test-split evaluation.** `code/eval_test_split.py` already exists and is
validated. `data/test.jsonl` has never been read by any training run — `engine.py` reads
only the train and val paths. This closes the most serious objection in the previous review
round (headline numbers reported on validation while a test split was declared), and it is
pure upside: it costs one eval pass per checkpoint and converts a methodological concern
into a reported number.

**(c) Out-of-domain perplexity transfer.** Zero-shot perplexity on a corpus the models never
saw is valid at any scale and is standard in LM papers. **Do not use wikitext-2** — T-L2.0
established that it shares its validation split byte-for-byte with wikitext-103, so it is
not held out. PTB or `text8` are legitimate. This is the lowest-priority item; drop it first
if the schedule slips.

Optional if there is time: **loss by position within the 256-token block.** Recursion should
matter most where context is shortest, so an early-position advantage for the recursive arms
would be a clean mechanism signal.

### Reporting rules for all of the above

These are exploratory unless you add them to the pre-registered families, and you may
**not** add a family after results exist (`code/confirmatory_tests.json` prohibits it
explicitly). So report them with effect sizes and intervals, **uncorrected and labelled as
such**, and keep them out of the abstract's significance claims. A stratified result is a
mechanism observation; the confirmatory verdicts stay with the declared families.

---

## 7. What not to do, in priority order

1. **Do not retrain MoRE** unless Plan A finished early. Its checkpoints exist.
2. **Do not run frontier benchmark suites.** §6.
3. **Do not change `batch_size`, `lr`, `weight_decay`, `dropout`, `epochs` or any frozen
   spec field.** Every one moves `config_hash` and breaks comparability with the kept cells.
4. **Do not alter the architecture to improve a number** (`CLAUDE.md` §6). If you find an
   improvement, preserve the original run and add the change as an explicitly labelled
   variant.
5. **Do not hand-copy a number into the paper.** Regenerate through
   `code/export_results.py --task language`.
6. **Do not delete the retired per-step MoRE cells** in
   `archive/pre_finalization/more_l7_1_per_step/`. They are a labelled ablation the paper
   reports.
7. **Do not add a third confirmatory family or promote an exploratory item into one.**

---

## 8. If the schedule slips

Cut in this order. The paper survives all four cuts.

1. Drop out-of-domain perplexity (§6c).
2. Drop the position-stratified loss.
3. Drop the MoR retrain and run the stratified and test-split evals on **MoE and MoRE
   only** — a two-arm stratified comparison still tests whether routing-plus-recursion
   beats routing alone on hard tokens, which is a real question. MoE is only ~5 h.
4. Drop every retrain and submit on the existing 15-cell `val/task_loss` matrix plus the
   mechanism argument from `code/lang_calibration_weights.json`. **This version is already
   submittable** — the decomposition result, the ponder-weight-transfer finding and the
   depth/specialization coupling are all established without a single new run. Everything
   above is strengthening, not rescuing.

Write the paper as if cut 4 is what you get, then add what lands. That ordering is what
makes the deadline safe.

---

## 9. One loose thread from Vedant's machine

A weight sweep (`automated/sweep_tlx20_weights.py`, T-LX.20) was launched on the 3050 and
writes to `code/lang_sweep_tlx20.json` after every point. Pull and check whether that file
has rows before writing the mechanism section — it tests the ponder-weight and
balance-weight axes on the **canonical** corpus rather than the dev corpus, which
materially strengthens §6 Cause 2 and Cause 3 of `PAPER_GUIDE.md`. Every row is a
`subset_fraction < 1`, single-seed proxy and may never be quoted as a canonical result.
If the file is absent or partial, write the mechanism section from
`code/lang_calibration_weights.json` and state that the canonical-corpus confirmation was
not run.
