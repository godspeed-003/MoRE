# HANDOFF.md — the opdiff matrix, for an agent with no other context

**What this is.** The third experimental study of this repository, on a new
dataset: `opdiff` — rows of 16 packed 24-token arithmetic examples, 25
operations (one of them with a *dial*: an iteration count written in binary in
the input), seven expert groups, 112.9 M training tokens per epoch, and two
held-out splits that test the dial on iteration counts the model has never
seen. The question it asks is the one the whole repository exists for: does a
top-1-routed mixture of experts that also recurses (MoRE) beat the two
ablations (MoE = routing without recursion, MoR = recursion without routing) on
a task where the operations have genuinely different difficulty — and can it do
so on the held-out iteration counts too?

**Who is reading this.** An agent on the co-author's machine, with no access to
this conversation. Every command below has been executed at least once, on the
smoke path, on the gate machine. The numbers in §9 are measurements, not
estimates.

**Read §8 (the never-change list) before you run anything.** It is short and
it is the list of things that would silently invalidate the study.

**IF ANY RUNS ALREADY EXIST ON AN EARLIER DATASET** (v3, before the F
operation was redesigned), keep them and do NOT mix them with v4 runs: the
aggregator refuses to average across dataset versions and says so, and a
`dataset_version` that differs means the rows were sampled differently, so the
two sets are two experiments. The port's operation axis covers both datasets'
k values, so an older run directory is still readable and aggregable — on its
own.

**The dataset itself is explained in [OPDIFF_DATASET.md](OPDIFF_DATASET.md).**
Read §2 of that file before interpreting any number that comes out of a run:
it defines the row layout, the k slot, the loss mask, the per-row labels and
the two OOD splits.

---

## 1. The machine and the interpreter

| | |
|---|---|
| Machine | the co-author's laptop, **RTX 4060 8 GB** |
| Interpreter | `C:\Users\Hp\anaconda3\envs\more_env\python.exe` |
| Working directory | the repository root (the directory containing `code/`, `runs/`, `TASKS.md`) |
| Repo state | `main`, at the commit that carries this file |

That interpreter path is not interchangeable with anything else. The
co-author's Anaconda **base** and `C:\Python314` have no `torch` at all, so a
bare `python` fails at `import torch` instead of falling back to CPU.

**Use the 8 GB card for this matrix, and treat the 6 GB card as unusable for
it.** The measured peak of the heaviest arm (MoR) is **5.80 GiB allocated /
6.38 GiB reserved** — the reserved figure is *over* the 6 GB card's memory, so
the arm barely fits there and the allocator is thrashing. On the 8 GB card the
same arm is 73 % allocated / 80 % reserved, which fits. Nothing else may use
the GPU while a run is in flight.

The full environment record is `AGENTS.md` §9. The college-lab RTX 4060 is
deliberately excluded from this matrix: interrupted epochs put a per-machine
confound inside the seed variance the study reports.

---

## 2. The dataset: check, regenerate, verify

The opdiff dataset is **not tracked in git** (113 MB), so it is regenerated per
machine. It is deterministic at seed 0, but a different NumPy RNG can produce
different sampled rows, so **the dataset identity is a fingerprint of the bytes
and it must be recorded**.

**Check first:**

```bash
python -c "import json;m=json.load(open('opdiff_v4/meta.json'));print(m['dataset_version'], m['rows'], m['vocab_size'], m['row_len'], m['train_k'], m['ood_interp_k'], m['ood_extrap_k'])"
```

Expected: rows = 294,000 train / 7,000 val / 7,000 test / 1,000 per held-out
k in ood_interp / 1,000 per held-out k in ood_extrap; `vocab_size` 37;
`row_len` 384; `train_k`, `ood_interp_k` and `ood_extrap_k` as the generator
declares them (`TRAIN_K`, `INTERP_K`, `EXTRAP_K` at the top of
`make_opdiff_dataset_v4.py`) — currently `train_k = [1, 2, 3, 4]`,
`ood_interp_k = [5]`, `ood_extrap_k = [6, 7]`.

**Those lists are a DESIGN choice, not a calibration.** The design's reasoning:
one F iteration costs about one layer (the eight digits update in parallel), so
the dial has to live inside the model's depth budget of 7 — a k the model cannot
do in 7 steps is not a difficulty, it is an arithmetic wall, and the v3
measurement showed exactly that (the old F operation was at chance at EVERY k,
including k = 1). A separate calibration run was started and then abandoned: it
was measuring the dataset's learnability, which is a dataset-design question,
and the matrix's own per-k table reports it honestly for free. The lists live in
the generator, not in a config.

**If it is missing, build it** (about 10 s):

```bash
python make_opdiff_dataset_v3.py --out opdiff_v4

**This is the v4 dataset, and its F operation is not the v3 one.**
v3's F was `f(x) = (x^2 + 12345) mod 4294967291` on a 32-bit
`x = A||B`; the full-scale measurement showed it was **not learnable
at ANY k, including k = 1** (0.161-0.172 per-digit accuracy and
0.0 exact match, against 0.850 for the other six groups), because
one iteration of a 32-bit modular squaring is a 32x32-bit
multiplication. v4's F is a per-nibble nonlinear recurrence over the
eight nibbles of the state, `d'_i = ((d_i XOR d_{(i+1) mod 8}) + 3) mod 16`,
applied k times: one iteration is a 256-case per-digit table
(learnable), the k-fold composition has no closed form over a
4.3e9 state space (so it cannot be memorised or shortcut), and the
eight digits update in parallel, so one iteration costs about one
layer and the dial spans the depth budget. See
[OPDIFF_DATASET.md](OPDIFF_DATASET.md) for the full argument;
`make_opdiff_dataset_v3.py` is kept because the measurement on it is
what justifies the change.
```

**Verify after building** — the things that must agree, and why each matters:

| check | expected | why it matters |
|---|---|---|
| `meta.json:vocab_size` | 37 | 19 fixed tokens (SEP, EOS, NULL, 16 hex digits) + 18 operation tokens; this is the width of the tied LM head |
| `meta.json:row_len` | 384 | 16 examples × 24 tokens; the positional table is sized by it |
| `meta.json:rows.train` | 294,000 | 42,000 rows per group × 7 groups |
| `meta.json:train_k` / `ood_interp_k` / `ood_extrap_k` | as above | the held-out iteration counts ARE the OOD evaluation; if they are missing, the run would report an in-distribution number and call it the OOD result |
| the five splits' `*_x.npy`, `*_group.npy`, `*_diff.npy`, `*_k.npy`, `*_sub.npy` | all present | a half-generated dataset fails at the first batch with a file name in the message |

Do **not** edit `make_opdiff_dataset_v3.py`. It is the dataset contract: the
token layout, the k slot, the loss mask, the group partition and the split
definitions are all defined there, and `code/test_opdiff_task_axis.py`
cross-checks the code's manifest against it by importing it.

**The smoke subset** (1 % of the data, for proving the machine works):

```bash
python make_opdiff_dataset_v3.py --out opdiff_v4_smoke --train_rows_per_group 512 --eval_rows_per_group 128 --ood_rows_per_k 128

**This is the v4 dataset, and its F operation is not the v3 one.**
v3's F was `f(x) = (x^2 + 12345) mod 4294967291` on a 32-bit
`x = A||B`; the full-scale measurement showed it was **not learnable
at ANY k, including k = 1** (0.161-0.172 per-digit accuracy and
0.0 exact match, against 0.850 for the other six groups), because
one iteration of a 32-bit modular squaring is a 32x32-bit
multiplication. v4's F is a per-nibble nonlinear recurrence over the
eight nibbles of the state, `d'_i = ((d_i XOR d_{(i+1) mod 8}) + 3) mod 16`,
applied k times: one iteration is a 256-case per-digit table
(learnable), the k-fold composition has no closed form over a
4.3e9 state space (so it cannot be memorised or shortcut), and the
eight digits update in parallel, so one iteration costs about one
layer and the dial spans the depth budget. See
[OPDIFF_DATASET.md](OPDIFF_DATASET.md) for the full argument;
`make_opdiff_dataset_v3.py` is kept because the measurement on it is
what justifies the change.
```

---

## 3. The preflight — run this first, always

```bash
python code/run_opdiff_matrix.py --preflight
```

It checks, before any GPU-hour is spent: a usable CUDA device (it runs a
matmul, it does not read a flag), the dataset's presence and its five splits,
that the trained and held-out k lists are disjoint, that the proxy guard
stamps the runs `exploratory` for all three arms, that W&B is either logged
in or explicitly offline, and the disk headroom for 15 checkpoints.

**A non-zero exit means the matrix does not start.** The message names what to
fix and how.

Set `WANDB_MODE=offline` if you have no W&B account — the metrics land in
`runs/<id>/metrics.json` either way, and W&B is convenience, not the record.

---

## 4. The pilot — 3 arms × seed 42

```bash
python code/run_opdiff_matrix.py --pilot
```

This is **seed 42 of the matrix**, not extra work: the pilot's runs are the
matrix's seed-42 cells and are kept.

**Acceptance checks, in this order.** A pilot run that does not pass them is a
failed pilot, and the next step is to diagnose it, not to continue.

1. **The run finished**: `runs/opdiffB_<Arch>_seed42__<hash>/metrics.json`
   exists and contains `val/task_loss` as a number.
2. **The loss fell**: `results.tsv` shows `val_loss` lower at epoch 3 than at
   epoch 1.
3. **The routing numbers are defined for the routed arms and N/A for MoR**:
   `val/routing_agreement_with_group` and `val/routing_hungarian_accuracy`
   are numbers for `moe` and `more`, and the string `"N/A"` for `mor`. A
   number for MoR's routing is a defect, not a result.
4. **The confusion matrix is square for the routed arms**: `metrics.json`
   contains exactly SEVEN `val/routing_recall/<group>` keys, named
   `E1_BITWISE` … `E7_ITER_GCD`. Seven oracle groups × seven experts. The
   Hungarian / AMI / purity report is all-`None` for a non-square matrix, so a
   6-expert run would report N/A everywhere and look like "the router did not
   specialise".
5. **The depth numbers are live for the adaptive arms**: `val/avg_recursion_steps`
   is a number for `mor` and `more` and `"N/A"` for `moe` (MoE has no
   recursion at all).
6. **THE OOD SPLITS WERE EVALUATED**: `ood_interp/task_loss`,
   `ood_interp/accuracy`, `ood_interp/exact_match`, `ood_extrap/task_loss`
   and the rest are present in `metrics.json`. If they are missing, the
   dataset's held-out splits were not evaluated and the OOD claim has no
   evidence behind it.
7. **THE F ROWS ARE BEING LEARNED** — this is the check that decides whether
   the rest of the OOD analysis means anything. Look at
   `val/accuracy_by_op/F<k>` and `val/exact_match_by_op/F<k>` for the
   **trained** k values. If those are near zero, the model cannot solve an F
   row at all, and the held-out numbers are then measuring "the model cannot
   do F" rather than "the model cannot extrapolate". The dataset's own caveat
   says exactly this: *"If it fails on ood_interp, check whether it also fails
   on the seen k values before blaming your architecture."*
   **The chance level is exact and is the thing to compare against**: an F row's
   answer is 8 hex digits plus the EOS, so a model that has learned nothing
   about the operation scores `(8/16 + 1)/9 = 0.1667` per scored position.
   The v3 F operation measured 0.161–0.172 — i.e. exactly chance, at every k.
   The v4 F operation is designed to be learnable, so a trained k sitting at
   0.1667 after the full budget means something is wrong with the run, not with
   the dataset.
   **If this check fails, REPORT IT AND CONTINUE.** It is a finding, not a
   reason to change the protocol: the matrix also measures the primary
   three-arm comparison, which does not depend on the F rows at all. Do NOT
   shrink `TRAIN_K`, do NOT change the epochs, do NOT re-freeze anything —
   report the numbers and let the user decide.
8. **The VRAM peak is as measured**: `perf/gpu_peak_alloc_gib` ≈ 0.7 (MoE),
   ≈ 2.4 (MoRE), ≈ 5.8 (MoR) GiB. If MoR's peak is materially above 5.8 GiB,
   **stop** and report it.
9. **The epoch time is the number §9 predicts**, within ~20 %. If it is
   materially slower, the budget changes and the user must be told before the
   remaining ten runs start.

---

## 5. The remaining ten runs

```bash
python code/run_opdiff_matrix.py --all
```

The driver **skips completed runs** (a run directory with a finished
`metrics.json`, for that arm and seed, on the same dataset version) and
**stops at the first failure** (§7). Re-running `--all` after a failure
resumes: the completed cells are skipped.

The ten remaining cells are seeds 43–46 × {MoE, MoR, MoRE}.

**Approval gate.** The full matrix was APPROVED by the user on 2026-10-21 at
the budget in §9 (3 epochs × 5 seeds, ≈ 32 h on this card), and the approval
is recorded as a tick in `TASKS.md` under T-OD.5. So the pilot AND the
remaining ten cells may run. If that box is ever unticked — e.g. on a second
machine, or after a re-freeze — run §3 and §4 only and report the measured
epoch times back.

### 5b. No capacity ladder — and why that is the right answer

**There is no ladder to run.** The question a ladder asks — "does the MoR-vs-MoRE
result depend on the MoR arm's parameter budget?" — is answered by a *design
fact* that this repository already measures and publishes: MoR's single FFN is
given the total hidden width of MoRE's seven experts (7 x 4 = 28), and the
measured gap in non-embedding parameters is **0.117 %**. That is the same
reason MoR is a legitimate arm of the published comparison at all
(6 x 4 = 24 on the six-group task). A ladder would re-derive a number the
design already fixes, on a *baseline* arm, on a task whose claim is about the
composition. It is not run.

**IF A LADDER IS EVER WANTED**, the widths are 4 (792 K non-embedding
parameters, 20 % of the budget), 24 (3.42 M -- the six-group width, i.e. the
width the arithmetic study gives MoR) and 28 (3.94 M -- the matched one). All
three must be measured on the SAME card, in the same protocol, or the
comparison has the card changed as well as the budget. `resolve_variant` labels
them `opdiff+ffn_mult_4` / `opdiff+ffn_mult_24`, and their run names say so,
and the aggregator's matrix pattern cannot match them -- so an ablation can
never be admitted into the canonical table by accident.

## 6. Aggregation, and what to report back

```bash
python code/summarize_opdiff_matrix.py
python code/summarize_opdiff_matrix.py --out results/opdiff/summary.json
```

This is the **only** aggregation command for this task. It reads the runs'
own `metrics.json` files and refuses to produce a number it did not read. It
refuses outright if the runs span more than one dataset version, or if one
arm's runs have more than one config hash (either would mean a mean over two
different experiments).

It prints three tables:

- **the headline** (val loss, accuracy, exact match, depth, routing) per arm;
- **every measured quantity** (mean ± std over the seeds that have it),
  including the OOD splits' loss, accuracy, exact match and depth;
- **the compute-versus-k table**: every k value in one row, with the SEEN
  values and the HELD-OUT ones side by side, and the depth the model spent on
  each. That table is where the study's claim is read off.

And with `--kind ablation` (or `--kind smoke`, which includes the smoke
ablations) it prints **the capacity ladder**: each (arm, width) with its
non-embedding parameter count next to its loss, so a 0.8 M-parameter arm and
a 3.9 M one can never be read as the same arm.

**Report back exactly these, per arm, as mean ± std over the seeds:**

| quantity | key | reading |
|---|---|---|
| **primary** | `val/task_loss` | nats/token over the generator's scored positions (answer digits + EOS). Lower is better. |
| accuracy | `val/accuracy` | per-digit accuracy over the scored positions |
| exact match | `val/exact_match` | per-example: every digit right. The strict reading of "the right answer". |
| depth | `val/avg_recursion_steps` | the adaptive arms only; MoE is N/A by construction |
| routing | `val/routing_hungarian_accuracy` | the permutation-invariant number — **primary for routing** |
| routing | `val/routing_ami`, `val/routing_purity` | chance-corrected and not; read them together |
| agreement | `val/routing_agreement_with_group` | agreement with a **designed prior**, NOT accuracy |
| balance | `train/expert_load_entropy_normalized` | a load-balance diagnostic, not evidence of specialization |
| **OOD** | `ood_interp/task_loss`, `ood_interp/accuracy`, `ood_interp/exact_match` | the held-out iteration counts inside the trained range — the realistic target |
| **OOD** | `ood_extrap/…` | the held-out counts beyond the trained range — the stretch result |
| **compute vs k** | `val/accuracy_by_op/F<k>`, `val/exact_match_by_op/F<k>`, `ood_interp/accuracy_by_op/F<k>`, … | the seen and the held-out k values in one table |

**Do not** quote `total_loss`. It is never a quality metric (CLAUDE.md §4): it
contains the balance, ponder and probe terms, and a low weighted total is not
evidence of a good model.

**Do not** compare the OOD loss with the validation loss. The OOD splits are
F rows only; the validation split is a mixture of all seven groups. The
comparable number is the **per-k F-row** loss/accuracy on the validation
split, which is why the aggregator prints the per-k table.

---

## 7. If a run fails: stop, report, do not continue

The driver stops at the first failure and prints which run died. That is
deliberate (plan.md §20): a crash mid-matrix is a diagnosis, not something to
paper over with a retry loop.

**The protocol:**

1. **STOP.** Do not launch the next run.
2. **Report the failure with its actual output.** The run's own
   `runs/<id>/stdout.log` has the full log; the driver's console has the
   exit code and the run's wall-clock. Quote the error verbatim.
3. **Find the cause.** The most likely causes, in order of how often they
   have actually happened in this repository: an OOM (check
   `perf/gpu_peak_alloc_gib` against §9), a dataset version that changed
   mid-matrix (the summariser will refuse to aggregate, which is the
   symptom), or a config that differs from `code/config_opdiff.json`.
4. **Fix it, then re-run `--all`.** The completed cells are skipped.

**Never** edit a completed run's `metrics.json`, `resolved_config.json` or
`results.tsv` to make it look finished, and never relabel a run after the
fact (CLAUDE.md §6). A half-finished run is re-run, not repaired.

---

## 8. The never-change list

These are the things that make the study a study. Changing any one of them
does not produce a better run; it produces a different study that cannot be
compared to the other arms, and the change will not be visible in the run
label.

| do not change | where it lives | why |
|---|---|---|
| the frozen seeds **42–46** | `code/canonical_spec_opdiff.json:seed_set` | the study reports mean ± std over five seeds; one seed is one observation |
| `config_opdiff.json` | the defaults file | every run's `config_hash` is derived from it; the summariser refuses a matrix whose arms have two hashes |
| `num_experts = 7` for MoE/MoRE | `config.py:CANONICAL_NUM_EXPERTS_BY_TASK` | seven is the oracle partition's width; six makes the confusion matrix non-square and every permutation-invariant routing number N/A |
| `ffn_mult = 28` for MoR | `config.py:CANONICAL_FFN_MULT_OPDIFF` | 7 × 4 = 28 is the same budget rule arithmetic and language use; the measured gap is 0.117 % and at 24 it would be 14 % short |
| `batch_size = 48` | `config_opdiff.json` | measured VRAM headroom per arm; a different batch is a different experiment and is not labelled by anything |
| `seq_len = 384`, `vocab_size = 37` | `config_opdiff.json` | the dataset's own row length and vocabulary; the engine refuses a mismatch with the files on disk |
| `loss_weights.family_cls = 0.0` | forced by the task block | the group probe is **detached**; a non-zero weight would put the oracle partition into the trunk's objective — the one confound this study exists to exclude |
| `halting_supervision` off | forced by the task block | opdiff has no depth curriculum; the generator's nominal difficulty is a placeholder awaiting dense-baseline calibration. The compute-versus-k claim is MEASURED, not supervised |
| `step_routing = 0.0` | `config_opdiff.json` | the unsupervised router is the arm under study; a non-zero value is a legitimate **labelled** oracle-routing ablation and may never share a table with an unsupervised run |
| the canonical MoR's `ffn_mult = 28` | `config.py:CANONICAL_FFN_MULT_OPDIFF` | 7 × 4 is what makes MoR budget-matched to MoRE; changing it turns the canonical arm into an ablation |
| `experiment_group = exploratory` | the driver | the opdiff spec is **unfrozen by design**; the proxy guard refuses a `canonical_opdiff_b` claim, and a run that claims canonical status before the protocol is ratified is a claim nobody can audit |
| the dataset itself | `make_opdiff_dataset_v3.py` | it is the contract: token layout, k slot, loss mask, group partition, the three k lists. The manifest in the code is cross-checked against it by the test suite |
| the k lists | the generator's `TRAIN_K` / `INTERP_K` / `EXTRAP_K` | they define what "held out" means. A k that is both trained and held out is not held out, and the preflight refuses it |
| `canonical_spec_opdiff.json`'s nulls | the spec file | they are what makes the guard refuse a canonical claim. Filling them in is a **deliberate act** with a `frozen_by` entry, not a convenience |

**Also never:** put the opdiff numbers in the same table as the arithmetic or
language ones. Three studies, three quantities (`val/task_loss` here is nats
per scored token; arithmetic's is an MSE), three run-name prefixes
(`opdiffB_`, `phaseB_`, `langB_`).

---

## 9. What is already measured (do not re-measure it)

Measured on the **gate machine** (RTX 3050 6 GB), the **smoke subset**
(3,584 train rows, 896 val, 640 ood_interp, 256 ood_extrap), batch 48, 3
epochs, seed 42, on 2026-10-21. The full dataset is **82× larger**, so the
epoch times below are an extrapolation from the measured token rate — the
rates themselves are direct measurements.

| arm | tokens/s | val/task_loss (1 ep → 3 ep) | val accuracy | val exact match | depth | Hungarian | AMI | peak alloc / reserved |
|---|---|---|---|---|---|---|---|---|
| MoE | 161,081 | 1.4303 → 1.3630 | 0.564 | 0.622 | N/A | 0.2513 | 0.0749 | 0.72 / 2.16 GiB |
| MoR | 17,301 | 1.4210 → 1.3550 | 0.565 | 0.536 | 3.67 | N/A | N/A | **5.80 / 6.38 GiB** |
| MoRE | 30,874 | 1.4225 → 1.3430 | 0.566 | 0.636 | 4.88 | 0.2084 | 0.0323 | 2.44 / 2.66 GiB |

**The per-epoch cost, MEASURED, and it is larger than an FLOPs estimate
suggests.** The two recursive arms are *kernel-launch-bound*, not
compute-bound: a depth-7 step over 48 rows of 304 tokens issues hundreds of
tiny kernels, and the measured batch cost is ~2.0 s where the FLOPs say ~0.08 s.
The GPU reports 70–80 % "utilisation" throughout, which is any-kernel
activity, not occupancy.

| arm | tokens/s (measured) | per epoch, training pass | + the evaluation |
|---|---|---|---|
| MoE | 151,081 | 12.5 min | ~14 min |
| MoR (28) | 17,301 | ~110–150 min | ~155 min |
| MoRE | 30,874 | 61 min | ~63 min |

**Budget (the arithmetic above is a floor; the per-epoch times are what
matter):** at 3 epochs x 5 seeds, MoE 3.1 h + MoR ~38 h + MoRE 15.3 h ≈ **56 h
on the 6 GB card**; at 5 epochs x 5 seeds ≈ **92 h on the 6 GB card**. On the
8 GB 4060, scale by the language calibration's measured ~1.5x: **≈ 37 h at 3
epochs and ≈ 61 h at 5 epochs**.

**The earlier "≈ 32 h" figure in this file was wrong** and is corrected here: it
was built from the token rate alone and did not carry the per-step launch
overhead, and it was measured with an evaluation that cost 18 minutes per epoch
(the evaluation is now ~1.5 min, after the per-token Python loops in the depth
accumulation were replaced with a `scatter_add_`). **The pilot's measured epoch
time is the number to plan against**, and it replaces these estimates as soon
as it lands.

**These rates are why the matrix is affordable at all**: MoR is 9.3× slower
than MoE per token, because MoR evaluates one dense FFN of width 28 for every
token while MoE evaluates one expert of width 4. MoR's memory peak is the
same story, and it is why the matrix belongs on the 8 GB card.

**The per-epoch figures above are TRAINING time** (the throughput is measured
over the training loop), and the evaluation is a separate ~1.5 min per epoch
after the T-OD.10 fix — it used to be ~18 min per epoch, which was more than
the training it was measuring, and which is why the first dense reference
appeared to hang for 88 minutes without finishing an epoch. The budget above
was measured with the slow evaluation, so it is the conservative figure.

---

## 10. What this study may and may not claim

**May.** A comparison of MoE, MoR and MoRE on a 25-operation dataset with
seven expert groups, over five seeds, with matched parameter budgets, at fixed
epochs — and a held-out-iteration-count evaluation in two parts
(interpolation and extrapolation).

**May not.**

- Call the routing number "accuracy". It is **agreement with a designed
  prior** — the generator's partition of the operations into seven groups is a
  plausible engineering grouping, not functional truth. A router that
  separates SHL from REV may be the better partition while scoring low here.
  The Hungarian / AMI / purity columns are the primary routing evidence.
- Call high load entropy specialization. Entropy is a **load-balance
  diagnostic**; a uniform router has maximal entropy and no structure at all.
- Call a lower `total_loss` a better model. The primary metric is
  `val/task_loss`.
- Call the depth numbers "compute efficiency". No FLOPs are in them, and
  there is **no depth curriculum** on this task, so the depth-allocation
  metrics are N/A by construction — the measured depth is a *behaviour*, and
  the question "does it spend more on the harder operations" is answered by
  the per-operation and per-k depth tables, not by an error against a target.
- Report the OOD loss against the validation loss. The OOD splits are F rows
  only; the comparison is with the **per-k F rows** of the validation split.
- Report the OOD result without the **seen** k values next to it. If the model
  cannot solve the trained F rows, the held-out numbers say nothing about
  extrapolation, and the table that shows both is the one that decides it.
- Put these numbers in the same table as the arithmetic or language study.
- Report any single seed. The study is mean ± std over five.

---

## 11. The commands, in order, in one place

```bash
# 0. the interpreter (Windows path; adjust the shell)
PY="C:\Users\Hp\anaconda3\envs\more_env\python.exe"

# 1. the dataset
$PY make_opdiff_dataset_v3.py --out opdiff_v4

**This is the v4 dataset, and its F operation is not the v3 one.**
v3's F was `f(x) = (x^2 + 12345) mod 4294967291` on a 32-bit
`x = A||B`; the full-scale measurement showed it was **not learnable
at ANY k, including k = 1** (0.161-0.172 per-digit accuracy and
0.0 exact match, against 0.850 for the other six groups), because
one iteration of a 32-bit modular squaring is a 32x32-bit
multiplication. v4's F is a per-nibble nonlinear recurrence over the
eight nibbles of the state, `d'_i = ((d_i XOR d_{(i+1) mod 8}) + 3) mod 16`,
applied k times: one iteration is a 256-case per-digit table
(learnable), the k-fold composition has no closed form over a
4.3e9 state space (so it cannot be memorised or shortcut), and the
eight digits update in parallel, so one iteration costs about one
layer and the dial spans the depth budget. See
[OPDIFF_DATASET.md](OPDIFF_DATASET.md) for the full argument;
`make_opdiff_dataset_v3.py` is kept because the measurement on it is
what justifies the change.
$PY make_opdiff_dataset_v3.py --out opdiff_v4_smoke --train_rows_per_group 512 --eval_rows_per_group 128 --ood_rows_per_k 128

**This is the v4 dataset, and its F operation is not the v3 one.**
v3's F was `f(x) = (x^2 + 12345) mod 4294967291` on a 32-bit
`x = A||B`; the full-scale measurement showed it was **not learnable
at ANY k, including k = 1** (0.161-0.172 per-digit accuracy and
0.0 exact match, against 0.850 for the other six groups), because
one iteration of a 32-bit modular squaring is a 32x32-bit
multiplication. v4's F is a per-nibble nonlinear recurrence over the
eight nibbles of the state, `d'_i = ((d_i XOR d_{(i+1) mod 8}) + 3) mod 16`,
applied k times: one iteration is a 256-case per-digit table
(learnable), the k-fold composition has no closed form over a
4.3e9 state space (so it cannot be memorised or shortcut), and the
eight digits update in parallel, so one iteration costs about one
layer and the dial spans the depth budget. See
[OPDIFF_DATASET.md](OPDIFF_DATASET.md) for the full argument;
`make_opdiff_dataset_v3.py` is kept because the measurement on it is
what justifies the change.

# 2. the preflight (must exit 0)
$PY code/run_opdiff_matrix.py --preflight

# 3. the pilot = seed 42 (only after the T-OD.5 approval for the FULL matrix)
$PY code/run_opdiff_matrix.py --pilot

# 4. the remaining ten cells
$PY code/run_opdiff_matrix.py --all

# 5. the aggregation
$PY code/summarize_opdiff_matrix.py --out results/opdiff/summary.json

# 6. the gates, before and after (the arithmetic total must stay 356)
$PY code/run_correctness_suite.py
$PY code/test_opdiff_task_axis.py
```

**Do not** run two matrices at once, and do not run anything else on the GPU
during a run.

---

## 12. What to send back

1. The three pilot runs' measured epoch times (from `results.tsv` and
   `perf/throughput_tokens_sec`) and the corrected full-matrix budget.
2. The full aggregation output, as printed — including the compute-versus-k
   table.
3. The per-operation and per-group depth tables.
4. Anything that failed, with its verbatim output, and what was done about it.
5. The `dataset_version` your machine produced — if it differs from the gate
   machine's, say so, because the rows were sampled differently and the runs
   are on a different dataset.

Do not paste numbers into a table by hand. The summariser is the table.
