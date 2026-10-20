# HANDOFF.md — the opdiff matrix, for an agent with no other context

**What this is.** The third experimental study of this repository, on a new
dataset: `opdiff` — rows of 16 packed 19-token arithmetic examples, 25
operations, seven expert groups, 110.7 M training tokens per epoch. The
question it asks is the one the whole repository exists for: does a
top-1-routed mixture of experts that also recurses (MoRE) beat the two
ablations (MoE = routing without recursion, MoR = recursion without routing) on
a task where the operations have genuinely different difficulty?

**Who is reading this.** An agent on the co-author's machine, with no access
to this conversation. Every command below has been executed at least once, on
the smoke path, on the gate machine. The numbers in §9 are measurements, not
estimates.

**Read §8 (the never-change list) before you run anything.** It is short and
it is the list of things that would silently invalidate the study.

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
bare `python` fails at `import torch` instead of falling back to CPU. If you
see `ModuleNotFoundError: No module named 'torch'`, you are on the wrong
interpreter.

**Use the 8 GB card for this matrix.** The measured peak of the heaviest arm
(MoR) is **5,491 MiB** — 69 % of 8 GB, but 91 % of the 6 GB gate card, which
is the occupancy at which the language calibration found the epoch time going
*up* instead of the run failing. Nothing else may use the GPU while a run is
in flight.

The full environment record is `AGENTS.md` §9. The college-lab RTX 4060 is
deliberately excluded from this matrix: interrupted epochs put a per-machine
confound inside the seed variance the study reports.

---

## 2. The dataset: check, regenerate, verify

The opdiff dataset is **not tracked in git** (113 MB), so it is regenerated per
machine. It is deterministic at seed 0, but a different NumPy RNG can produce
different sampled rows, so **the dataset identity is a content hash and it must
be recorded**.

**Check first:**

```bash
python -c "import json;m=json.load(open('opdiff_v2/meta.json'));print(m['dataset_version'], m['rows'], m['vocab_size'], m['row_len'])"
```

Expected: a version string like `opdiff-v2-<12 hex>`, `rows` = 364,000 train /
7,000 val / 7,000 test / 3,000 ood, `vocab_size` 43, `row_len` 304.

**If it is missing, build it** (about 15 s):

```bash
python make_opdiff_dataset_v2.py --out opdiff_v2
```

**Verify after building** — the four things that must agree, and the reason
each one matters:

| check | expected | why it matters |
|---|---|---|
| `meta.json:vocab_size` | 43 | 18 fixed tokens (SEP, EOS, 16 hex digits) + 25 operation tokens; this is the width of the tied LM head |
| `meta.json:row_len` | 304 | 16 examples × 19 tokens; the positional table is sized by it |
| `meta.json:rows.train` | 364,000 | 52,000 rows per group × 7 groups |
| `meta.json:dataset_version` | present | the content identity that every run's provenance records |

Do **not** edit `make_opdiff_dataset_v2.py`. It is the dataset contract: the
token layout, the loss mask, the group partition and the OOD split are all
defined there, and `code/test_opdiff_task_axis.py` cross-checks the code's
manifest against it by importing it.

**The smoke subset** (1 % of the data, for proving the machine works):

```bash
python make_opdiff_dataset_v2.py --out opdiff_v2_smoke --train_rows_per_group 512 --eval_rows_per_group 128
```

---

## 3. The preflight — run this first, always

```bash
python code/run_opdiff_matrix.py --preflight
```

It checks, before any GPU-hour is spent: a usable CUDA device (it runs a
matmul, it does not read a flag), the dataset's presence and version, that the
proxy guard stamps the runs `exploratory` for all three arms, that W&B is
either logged in or explicitly offline, and the disk headroom for 15
checkpoints.

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
   epoch 1. (Measured on the smoke subset: MoE 1.426 → 1.333, MoR 1.413 →
   1.325, MoRE 1.385 → 1.309.)
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
6. **The VRAM peak is as measured**: `perf/gpu_peak_alloc_gib` ≈ 0.6 (MoE),
   ≈ 5.5 (MoR), ≈ 2.1 (MoRE) GiB. If MoR's peak is materially above 5,491 MiB,
   **stop** and report it — the matrix would not fit the 8 GB card either.
7. **The epoch time is the number §9 predicts**, within ~20 %. If it is
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

**Approval gate.** The full matrix may only start after the user has
approved the budget in the T-OD.5 thread. The approval is recorded as a tick
in `TASKS.md` under T-OD.5. If that box is not ticked, run §3 and §4 only,
then report the measured epoch times back — that is what the approval is
based on.

---

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

**Report back exactly these, per arm, as mean ± std over the seeds:**

| quantity | key | reading |
|---|---|---|
| **primary** | `val/task_loss` | nats/token over the generator's scored positions (answer digits + EOS). Lower is better. |
| depth | `val/avg_recursion_steps` | the adaptive arms only; MoE is N/A by construction |
| routing | `val/routing_hungarian_accuracy` | the permutation-invariant number — **primary for routing** |
| routing | `val/routing_ami`, `val/routing_purity` | chance-corrected and not; read them together |
| agreement | `val/routing_agreement_with_group` | agreement with a **designed prior**, NOT accuracy |
| balance | `train/expert_load_entropy_normalized` | a load-balance diagnostic, not evidence of specialization |

Plus, from the summariser's second table, the **average recursion depth per
operation** (22 rows) and **per expert group** (7 rows) — that is where
"does the model spend more computation on the operations that need more of
it" is answered.

**Do not** quote `total_loss`. It is never a quality metric (CLAUDE.md §4): it
contains the balance, ponder and probe terms, and a low weighted total is not
evidence of a good model.

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
| `ffn_mult = 28` for MoR | `config.py:CANONICAL_FFN_MULT_OPDIFF` | 7 × 4 = 28 is the same budget rule arithmetic and language use; the measured non-embedding gap is 0.117 % and at 24 it would be 14 % short |
| `batch_size = 48` | `config_opdiff.json` | measured VRAM headroom per arm; a different batch is a different experiment and is not labelled by anything |
| `seq_len = 304`, `vocab_size = 43` | `config_opdiff.json` | the dataset's own row length and vocabulary; the engine refuses a mismatch with the files on disk |
| `loss_weights.family_cls = 0.0` | forced by the task block | the group probe is **detached**; a non-zero weight would put the oracle partition into the trunk's objective — the one confound this study exists to exclude |
| `halting_supervision` off | forced by the task block | opdiff has no depth curriculum; the generator's nominal difficulty is a placeholder awaiting dense-baseline calibration, not a target |
| `step_routing = 0.0` | `config_opdiff.json` | the unsupervised router is the arm under study; a non-zero value is a legitimate **labelled** oracle-routing ablation and may never share a table with an unsupervised run |
| `experiment_group = exploratory` | the driver | the opdiff spec is **unfrozen by design**; the proxy guard refuses a `canonical_opdiff_b` claim, and a run that claims canonical status before the protocol is ratified is a claim nobody can audit |
| the dataset itself | `make_opdiff_dataset_v2.py` | it is the contract: token layout, loss mask, group partition, OOD split. The manifest in the code is cross-checked against it by the test suite |
| `canonical_spec_opdiff.json`'s nulls | the spec file | they are what makes the guard refuse a canonical claim. Filling them in is a **deliberate act** with a `frozen_by` entry, not a convenience |

**Also never:** put the opdiff numbers in the same table as the arithmetic or
language ones. Three studies, three quantities (`val/task_loss` here is nats
per scored token; arithmetic's is an MSE), three run-name prefixes
(`opdiffB_`, `phaseB_`, `langB_`).

---

## 9. What is already measured (do not re-measure it)

Measured on the **gate machine** (RTX 3050 6 GB), the **smoke subset**
(3,584 train rows, 896 val rows), batch 48, 3 epochs, seed 42, on
2026-10-21. The full dataset is **101.6× larger**, so the epoch times below
are an extrapolation from the measured token rate — the rates themselves are
direct measurements.

| arm | items/s | tokens/s | val/task_loss (1 ep → 3 ep) | depth | Hungarian | AMI | peak VRAM |
|---|---|---|---|---|---|---|---|
| MoE | 498.4 | 151,504 | 1.4261 → 1.3327 | N/A | 0.2241 | 0.0738 | 0.59 GiB |
| MoR | 52.4 | 15,930 | 1.4129 → 1.3254 | 4.17 | N/A | N/A | **5.49 GiB** |
| MoRE | 95.3 | 28,957 | 1.3853 → 1.3087 | 5.43 | 0.1846 | 0.0140 | 2.10 GiB |

**Extrapolated epoch time on the full dataset (364,000 train + 7,000 val
rows), 3 epochs per seed, 5 seeds:**

| arm | per epoch | per seed (3 ep) | 5 seeds |
|---|---|---|---|
| MoE | 12.4 min | 37 min | 3.1 h |
| MoR | 118 min | 5.9 h | 29.5 h |
| MoRE | 65 min | 3.2 h | 16.2 h |
| **all three** | | **9.8 h** | **≈ 49 h** |

On the 8 GB 4060 the same matrix should take **≈ 33 h** (the language
calibration measured the 4060 at ~1.5× the 3050's per-epoch time). Treat that
as an estimate and **correct it with the pilot's measured epoch time** before
the remaining ten runs — the pilot is the measurement, and it is the first
thing in §4.

**These rates are why the matrix is affordable at all**: MoR is 9.5× slower
than MoE per token, because MoR evaluates one dense FFN of width 28 for every
token while MoE evaluates one expert of width 4. MoR's memory peak is the
same story. This is the parameter-budget rule doing its job, not a defect.

---

## 10. What this study may and may not claim

**May.** A comparison of MoE, MoR and MoRE on a 25-operation dataset with
seven expert groups, over five seeds, with matched parameter budgets, at
fixed epochs. The routing numbers are permutation-invariant, so they measure
the router's **partition**, not its arbitrary numbering.

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
  the per-operation depth table, not by an error against a target.
- Put these numbers in the same table as the arithmetic or language study.
- Report any single seed. The study is mean ± std over five.

---

## 11. The commands, in order, in one place

```bash
# 0. the interpreter (Windows path; adjust the shell)
PY="C:\Users\Hp\anaconda3\envs\more_env\python.exe"

# 1. the dataset
$PY make_opdiff_dataset_v2.py --out opdiff_v2
$PY make_opdiff_dataset_v2.py --out opdiff_v2_smoke --train_rows_per_group 512 --eval_rows_per_group 128

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
2. The full aggregation output, as printed.
3. The per-operation and per-group depth tables.
4. Anything that failed, with its verbatim output, and what was done about it.
5. The `dataset_version` your machine produced — if it differs from
   `opdiff-v2-a899c743eab0`, say so, because the rows were sampled
   differently and the runs are on a different dataset.

Do not paste numbers into a table by hand. The summariser is the table.
