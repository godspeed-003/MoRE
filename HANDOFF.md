# HANDOFF.md — instructions for the agent driving the language runs

You are picking up the MoRE English-language migration on the machine that has the
project's best GPU (RTX 4060 Laptop, 8 GB). Everything up to and including **Phase L-6**
is finished, committed and pushed. Your job is **Phase L-7 onward**: measure, freeze,
run, record, push.

Read in this order before doing anything: `SETUP.md`, then `CLAUDE.md`, then
`TASKS_LANGUAGE.md` (the ledger — find the first box that is not `[x]`), then
`plan_language.md` for the design of whatever task that is. `updated_rules.md` and
`updated_objective.md` are authoritative on constraints and override anything here.

---

## The five rules that matter most

1. **Gate L0 is the stop condition.** `python code/run_correctness_suite.py` must print
   `TOTAL 356 356 0 0` / `ALL GATES PASS`, and it is re-run **after every commit**. Any
   other count: STOP, report the actual output, find the cause, fix, re-run
   (`plan.md` §20). Never revert a correctness fix because the old number looked better.

2. **A box is `[x]` only when its own Verify line has actually been executed and
   passed**, with the number recorded inline as Evidence. A ticked box with evidence
   only in chat is the failure mode the ledger exists to prevent.

3. **Report Outcome C honestly.** If MoRE gives no advantage, or fails to learn adaptive
   depth, that is a valid result and the paper says so. Do not tune until the desired
   conclusion appears. `CLAUDE.md` §8 is not decoration.

4. **Never touch the arithmetic evidence.** `runs/phaseB_*`, `archive/`,
   `data/dataset_meta.json`, `data/*.jsonl` are the published study. Do not re-run,
   delete, or move them. Never put an arithmetic number and a language number in the
   same table — one is an MSE, the other a per-token cross-entropy.

5. **Push after every gate that a REAL RUN validated** — not after a gate that only
   pre-written assertions passed. Gate L0 qualifies (Gate 5 launches two actual
   1-epoch runs). Commit the ledger and the changelog in the *same* commit as the code.

---

## Where things stand

Done and verified: the corpus (both WikiText sizes, Gate L2 passing), the POS family
lookup, the trivial floors, the shuffled control, the language dataset class, the
token-embedding + tied LM head, the causal-LM loss, the detached family probe, the depth
correlations with permutation nulls, and Gates L1 / L2 / L3 / L4.

**The number everything is measured against:** `primary_metric_floor = 4.9849`
nats/token on wikitext-103 val (7.1918 bits, perplexity 146.2). This is the backoff
bigram — the lowest of uniform 9.0109 / unigram 7.1982 / bigram 4.9849. **A run that
does not beat 4.9849 has learned nothing a two-column count table could not.** The dev
corpus figure is 5.3983 and is **not** interchangeable with it.

Three things already measured that will change how you read your results:

- **The oracle POS partition's own normalized load entropy is 0.8884, not 1.0**, with a
  5.29× largest/smallest family token ratio. A router that perfectly reproduced POS
  would score 0.8884. So the balance term and the POS-agreement number pull against each
  other, and every language load-entropy figure must be quoted against 0.8884.
- **AMI has a nonzero floor, and it is PER RUN — do not use a fixed constant.** The
  shuffled control is now wired (T-L6.9), so every language `metrics.json` carries
  `val/routing_control_pos/ami_control_mean` and its std beside `ami_real`. Use those. An
  earlier note here said to compare against 0.0527 by hand; that figure is the AMI a
  *POS-perfect* router scores against the control and is **not** a universal floor — the
  control mean depends on both partitions and was 0.0069 in the seed-44 run. Read the
  delta and its z, and treat a `delta_z` of `None` as "the control had no spread", which
  happens when the prediction is constant.
- **A mis-shifted LM loss reads ~5.6 nats at epoch 0** — below the uniform floor, close
  enough to 4.98 to look like fast learning. If an early loss looks too good, suspect the
  shift before believing it.

### Two 1-epoch CPU smoke runs already exist. Read them as hypotheses, not results.

`runs/langB_MoRE_seed42__91c9bba1` and `runs/langB_MoR_seed43__c1873a66`. Different
seeds, one epoch, dev corpus — so any difference could be seed variance. What they show:

| | MoRE (s42) | MoR (s43) |
|---|---|---|
| `val_loss` vs dev floor 5.3983 | 5.3567 (**+0.042 better**) | 5.6595 (**−0.261 worse**) |
| `depth/std` | 0.071 | **0.497** |
| depth hist, steps 1/2 | 1,088 / 281,619 | 156,157 / 126,893 |
| per-family depth spread | 0.011 steps | **0.74 steps** |

MoR allocates depth and predicts badly; MoRE predicts (barely) above the floor with depth
collapsed to 99.5% at one step. If that survives real seeds and epochs it is an Outcome B
shape. **Do not quote it before it does.**

**Depth allocation: T-L6.10 is CLOSED, and the answer changed the picture.** An 8-epoch
GPU run (`runs/langB_MoRE_seed44__22474619`) shows the frequency-depth partial correlation
STRENGTHENING across epochs: +0.193, +0.411, +0.442, +0.473 at epochs 2/4/6/8. The test was
"an artifact should weaken as training equalises the embedding row norms, a real behaviour
should not", so the artifact hypothesis is rejected -- this is learned. The direction is
contrary to the original hypothesis and stands as measured: **frequent tokens receive more
recursion.**

Two things I told you earlier are retracted by that run:

- **"Depth collapses" was under-training, not the architecture.** `avg_depth` climbs
  1.96 to 5.16 of a 7 budget over 8 epochs and `depth/std` reaches 1.174, against 0.071 at
  one epoch. Do not quote the 99.5%-at-one-step histogram; it was epoch 1 of an untrained
  model.
- **The margin over the floor is not thin.** +0.042 nats at one epoch became **+0.666** at
  eight (val_loss 4.7318 against the dev bigram floor 5.3983, perplexity 113.6 vs 221.0).

Robust across the run: `depth/by_document/between_share` stays near **0.010**, so allocation
is **within**-passage, not passage-level, however much total depth grows.

Use `depth/embed_norm_logfreq_partial_norm`, which is also a per-epoch column in
`results.tsv`. Note the ledger's Verify line offers "or equivalently, within norm deciles" --
that is **false**, and measured: on a fully mediated synthetic case whose true partial is
+0.016, the within-decile value is **+0.941**, so ten bins would pass the confound straight
through. Treat `..._within_bins_mean` (default 50 bins) as a cross-check where a LOW value is
informative and a high one is not conclusive.


`ffn_mult` for the language arms is frozen at `{moe: 4, mor: 24, more: 4}`, re-derived
(not copied) with rel(non-embedding) 0.112%. MoRE at that shape is **5,584,908
parameters**; if your run prints a different number the arms are not budget-matched —
stop.

---

## What to do, in order

### Step 1 — Phase L-7.0: measure throughput and VRAM on the 4060

This is the task that is *blocked on your hardware* and is why the work moved to your
machine. `batch_size` and `seq_len` are `null` in
`code/canonical_spec_language.json` precisely so the proxy guard refuses every canonical
claim until they are measured — that refusal is the mechanism working.

Sweep on **wikitext-2** (fast) with the CUDA interpreter, a fixed small step count, and
record for each `(batch_size, seq_len)`: tokens/s, peak VRAM in MiB, and whether the step
OOMs. Use `torch.cuda.max_memory_allocated()`. Both `--batch_size` and `--seq_len` are
CLI flags. Note that changing `seq_len` requires the corpus to have been **packed** at
that length:

```bash
python data/lang/build_language_dataset.py --corpus wikitext-2 --stage pack --seq_len 512 --cache_dir data/lang/_hf_cache
```

Record the table as Evidence under T-L7.0 in the ledger. **Your 8 GB replaces the 6 GB
figure in every note that mentions it** — say so explicitly where you write the numbers,
because a later reader will otherwise compare your batch size against a 6 GB constraint.

### Step 2 — Phase L-7.1: freeze the protocol in ONE commit

From the T-L7.0 measurements, fill in every remaining `null` in
`canonical_spec_language.json`: `epochs`, `batch_size`, `lr`, `weight_decay`, `dropout`,
`seq_len`, `dataset_version`, `train_split_version`, and the per-arm `max_depth`,
`halting_weight`, `routing_balance_weight`.

Two of those need care rather than a copied value:

- **`routing_balance_weight` and `halting_weight` must be re-measured, not carried.** The
  arithmetic 0.001 was chosen so the weighted balance term sat at 0.017× the task loss.
  The language task loss starts near 9.0 nats against arithmetic's ~0.06 MSE — **more
  than two orders of magnitude larger** — so 0.001 would put the balance term at ~1e-4×
  the task loss, i.e. effectively off, in the arm whose central failure mode is router
  collapse. Do what arithmetic's T4.2 did: pick the weight that puts the weighted term at
  a stated fraction of the task loss after warmup, and record the fraction.
- **`dropout`** was inherited as 0.1 on 70 k arithmetic records. With ~103 M training
  tokens against a 5.6 M-parameter model the language arm is underfitting by
  construction, so 0.1 is a guess in the wrong direction. Measure or argue it; do not
  carry it.

Add a `frozen_by` entry naming the task or measurement for every field you freeze, in the
same edit. A frozen number with no `frozen_by` is indistinguishable from an inherited
guess, which is what that file exists to prevent.

Then verify both directions: a run with the frozen config is **accepted** as
`canonical_lang_b`, and the same run with any single protocol field perturbed is
**refused**.

### Step 3 — Phase L-7.2: Gate L6 / Gate L7

- **L6:** publish the three arms' parameter counts as a table and confirm the MoR/MoRE
  gap is < 5% on both the total and the non-embedding count.
- **L7:** one short run must beat `primary_metric_floor` (4.9849 on the canonical
  corpus). **If it does not, the matrix is not worth running** — apply the gate-failure
  protocol. That is the entire purpose of having a floor.

### Step 4 — Phase L-8: the canonical matrix

Three architectures × five seeds `{42, 43, 44, 45, 46}`, `--experiment_group
canonical_lang_b`. Report **mean ± std**; one favourable seed is never evidence. Verdicts
come from the exact randomization test in `code/seed_stats.py`, never from a k×std
threshold. Never hand-copy a number into a table — the exporter refuses to mix dataset
versions or config hashes and emits `N/A` rather than fabricating.

---

## Recording results so the paper can cite them

This is already enforced by `code/more/run_context.py`, so mostly you just must not fight
it. Every run writes **only** into `runs/<experiment_id>/`: `config.json`,
`resolved_config.json`, `metrics.json`, `results.tsv`, `checkpoint.pt`, `stdout.log`.
Never write a global `best_model.pt` / `results.tsv` / `final_run_metrics.json`.

Provenance carries `experiment_id`, `experiment_group`, `architecture`, `variant`,
`seed`, `dataset_version`, `train_split_version`, `code_git_commit`, `config_hash`, and
the resolved epochs / subset_fraction. That is what lets a figure be traced to an exact
run six months from now.

`stdout.log` and `*.pt` are **git-ignored**, so when you push, the citable record is
`config.json` + `resolved_config.json` + `metrics.json` + `results.tsv`. If a run's
`stdout.log` contains something a reader needs, quote it into the changelog rather than
citing the file.

Keep `"N/A"` in `metrics.json` and `results.tsv` exactly as it is. It means "this
quantity does not exist for this configuration", it is not a gap to fill, and a 0.0 there
is a fabricated measurement. Dropping dead columns is the *paper renderer's* job
(T-LX.5), not the machine record's.

## Pushing back

```bash
git add -A && git commit -m "<what you measured, and the number>"
```

```bash
python code/run_correctness_suite.py
```

```bash
git push origin claude/english-language-dataset-migration-92946e
```

Ledger + changelog + code in the same commit. One `changelog.md` entry per completed
task, appended before `<!-- APPEND-MARKER-CL -->`, naming the code location to open when
that area misbehaves.

## If you get stuck

Mark the box `[!]` with the blocker named, move to the next independent task, and say so
in the commit message. Do not silently skip, and do not invent a value to get past a
gate. `changelog.md` records the traps already hit in this codebase — guards landing in
dead code, label checks running before override resolution, sentinels reported as
measurements — and it is worth reading before changing routing, halting, the loss
assembly, or the metric layer.
