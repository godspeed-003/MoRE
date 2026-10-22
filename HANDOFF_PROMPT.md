# The prompt to give Ayan's agent

Copy everything between the two lines. The repository is the brief —
`HANDOFF.md` is the authority and this prompt is the pointer.

---

You are running the **opdiff matrix** for the MoRE repository — a task-level
ablation on a synthetic arithmetic dataset with 25 operations, seven expert
groups, a binary iteration-count slot in the input, and two held-out splits
(interpolation + extrapolation). The repository is the brief; this prompt is the
pointer.

**0. What changed since the last prompt — read this first.**
- **The dataset is now `opdiff_v4`, not `opdiff_v3`.** v3's F operation was
  `f(x) = (x² + 12345) mod 4294967291` on a 32-bit `x = A‖B`, i.e. a 32×32-bit
  modular multiplication, and the measurement says it is **not learnable at any
  k**: 0.161–0.172 per-digit accuracy against an exactly computable chance
  level of 0.1667, at every k *including k = 1*. v4's F is a per-nibble
  nonlinear recurrence, `d'_i = ((d_i XOR d_{(i+1) mod 8}) + 3) mod 16`,
  iterated k times: one iteration is a 256-case per-digit table (learnable), the
  k-fold composition has no closed form over a 4.3 × 10⁹ state space (so it
  cannot be memorised or shortcut), and the eight digits update in parallel, so
  one iteration costs about one layer. **The layout, loss mask, vocabulary,
  operation tokens, seven groups, E7 structure, split definitions and per-row
  label files are unchanged.**
- The code now defaults to `opdiff_v4` (`code/config_opdiff.json`).
- **If you already have v3 runs, keep them and do NOT mix them with v4 runs.**
  The aggregator refuses to average across dataset versions and says so; a
  different `dataset_version` means the rows were sampled differently, so the
  two sets are two experiments. The operation axis covers both datasets' k
  values, so an older run directory is still readable — on its own.
- **There is no capacity ladder to run.** The question it would ask ("does the
  MoR-vs-MoRE result depend on MoR's parameter budget?") is answered by a
  design fact the repo already measures: MoR's single FFN is given the total
  hidden width of MoRE's seven experts (7 × 4 = 28), a measured **0.117 %** gap
  in non-embedding parameters — which is the same reason MoR is a legitimate
  arm of the published comparison at all. Do not run MoR at other widths.
- **The budget is corrected.** The depth-7 arms are kernel-launch-bound, not
  compute-bound (~2.0 s per batch where the FLOPs say ~0.08 s), so **5 epochs
  × 5 seeds ≈ 61 h on the 8 GB card** — the earlier "≈ 32 h" ignored that.
  Per epoch, measured: MoE ~14 min, MoR ~155 min, MoRE ~63 min. The pilot's
  actual epoch time replaces these as soon as it lands.

**1. Orient first.**
```bash
cd <the repo>              # the directory containing code/, runs/, TASKS.md
git pull origin main       # commit faca237 or later
```
Then read, IN FULL, IN THIS ORDER:
- **`HANDOFF.md`** — the authority: machine and interpreter, the dataset, the
  preflight, the pilot and its nine acceptance checks, the remaining ten runs,
  the aggregation, the stop-at-first-failure protocol, the never-change list,
  the measured numbers, and what the study may and may not claim.
- **`OPDIFF_DATASET.md`** — what the data is: the row layout, the binary k
  slot, the loss mask, the per-row labels, the F_k difficulty dial, the two
  OOD splits, and why the F operation was redesigned.
- **`TASKS.md`, the T-OD.5 box** — the budget is approved, so the pilot and
  the remaining ten cells may run.

**2. Environment — three things that will otherwise waste hours.**
- Interpreter: `C:\Users\Hp\anaconda3\envs\more_env\python.exe`. Not
  `python` — the Anaconda base and `C:\Python314` have no torch at all.
- The 8 GB card, and nothing else may use the GPU during a run: the MoR arm
  peaks at 5.80 GiB allocated / 6.38 GiB reserved, which is over the 6 GB
  card's memory.
- Git Bash is the shell; for long commands, write a script file rather than
  fight the quoting.

**3. Run, in this order.**
```bash
PY="C:\Users\Hp\anaconda3\envs\more_env\python.exe"
$PY make_opdiff_dataset_v4.py --out opdiff_v4        # the dataset is NOT in git (~10 s)
$PY code/run_opdiff_matrix.py --preflight            # must exit 0
$PY code/run_opdiff_matrix.py --pilot                # = the matrix's seed-42 cells (kept)
$PY code/run_opdiff_matrix.py --all                   # the remaining ten; skips done ones, stops at the first failure
$PY code/summarize_opdiff_matrix.py --out results/opdiff/summary.json
$PY code/run_correctness_suite.py                    # the arithmetic total must still be 356
$PY code/test_opdiff_task_axis.py
```

**4. The pilot's nine acceptance checks** are in `HANDOFF.md` §4 — go through
all of them and report each. **Check 7 decides how the OOD result may be
read**, and the comparison is against an exact chance level: an F row's answer is
8 hex digits plus the EOS, so a model that has learned nothing about the
operation scores `(8/16 + 1)/9 = 0.1667` per scored position. Compare
`val/accuracy_by_op/F<k>` and `val/exact_match_by_op/F<k>` for the trained k
values against the held-out ones in the compute-versus-k table. If check 7
fails, **REPORT IT AND CONTINUE** — it is a finding, not a protocol failure.
Do not shrink `TRAIN_K`, do not change the epochs, do not re-freeze anything.

**5. If anything fails: STOP.** No retry, no editing a finished run's
`metrics.json` / `resolved_config.json` / `results.tsv`, no changing the config
to make it fit, no continuing around a failure. `HANDOFF.md` §7 has the
protocol: report the failure with its verbatim output, the run directory, and
what you checked. Completed cells are intact and `--all` resumes by skipping
them.

**6. Never change** (`HANDOFF.md` §8 has the list with reasons): seeds 42–46;
`code/config_opdiff.json`; `num_experts` 7 for MoE/MoRE and `ffn_mult` 28 for
MoR; `batch_size` 48; `seq_len` 384 and `vocab_size` 37;
`loss_weights.family_cls` 0.0; halting supervision off; `step_routing` 0.0;
`experiment_group` exploratory; the dataset and its three k lists; the nulls in
`code/canonical_spec_opdiff.json`.

**7. Report back** (`HANDOFF.md` §12): the pilot's measured epoch times and
the corrected budget; the full summariser output including the
compute-versus-k table; the per-operation and per-group depth tables; any
failure with its verbatim output; and the `dataset_version` your machine
produced — if it differs from the gate machine's, say so, because the rows
were sampled differently and the runs are on a different dataset. Don't paste
numbers into a table by hand: the summariser is the table.

---
