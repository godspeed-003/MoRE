# The prompt to give Ayan's agent

Copy everything between the two lines. The repository is the authority —
`HANDOFF.md` is the brief, and this prompt is the pointer plus the three
things an agent with no context would otherwise get wrong (the interpreter,
the GPU, and what to do when something fails).

---

You are running the **opdiff matrix** for the MoRE repository — the third
experimental study, on a new dataset with 25 operations, seven expert groups
and two held-out splits. The repository is the brief; this prompt is just the
pointer.

**1. Orient (do this first).**
```bash
cd <the repo>              # the directory containing code/, runs/, TASKS.md
git pull origin main       # the work is on main (commit fb093b2 or later)
```
Then read, IN FULL and IN THIS ORDER:
- `HANDOFF.md` — **the authority**: the machine and interpreter, the dataset,
  the preflight, the pilot and its nine acceptance checks, the remaining ten
  runs, the aggregation, the stop-at-first-failure protocol, the
  never-change list, and the already-measured numbers.
- `OPDIFF_DATASET.md` — what the data is: the row layout, the binary
  iteration-count slot, the loss mask, the per-row labels, the F_k difficulty
  dial, and the two OOD splits (interpolation and extrapolation).
- `TASKS.md`, the `T-OD.5` box — the budget is **approved** (2026-10-21,
  ≈ 32 h on this card), so the pilot **and** the remaining ten cells may run.

**2. Environment — three things that will otherwise waste hours.**
- Interpreter: `C:\Users\Hp\anaconda3\envs\more_env\python.exe`. Not `python`
  — the Anaconda base and `C:\Python314` have no `torch` at all.
- The **8 GB** card, and nothing else may use the GPU while a run is in
  flight. The MoR arm peaks at 5.80 GiB allocated / 6.38 GiB reserved, which
  is OVER the 6 GB card's memory: this card is the only one that fits.
- Git Bash is the shell. Long commands: write them to a file, do not fight
  the quoting.

**3. Run, in this order.**
```bash
PY="C:\Users\Hp\anaconda3\envs\more_env\python.exe"

# a. the dataset, if it is not there yet (~10 s; it is not in git)
$PY make_opdiff_dataset_v3.py --out opdiff_v3

# b. the preflight — must exit 0, and it prints what to fix if it does not
$PY code/run_opdiff_matrix.py --preflight

# c. the pilot = the matrix's seed-42 cells (kept, not extra work)
$PY code/run_opdiff_matrix.py --pilot

# d. the remaining ten cells (skips the completed ones, stops at the first failure)
$PY code/run_opdiff_matrix.py --all

# e. the aggregation — the only aggregation command for this task
$PY code/summarize_opdiff_matrix.py --out results/opdiff/summary.json

# f. the gates, before and after: the arithmetic total must still be 356
$PY code/run_correctness_suite.py
$PY code/test_opdiff_task_axis.py
```

**4. The pilot's acceptance checks** are in `HANDOFF.md` §4 — go through all
nine and report each. Check 7 is the one that decides how the OOD result may
be read: look at `val/accuracy_by_op/F<k>` and `val/exact_match_by_op/F<k>`
for the **trained** k values, and compare them with the held-out ones in the
compute-versus-k table. **If check 7 fails, REPORT IT AND CONTINUE** — it is
a finding, not a protocol failure. Do NOT shrink `TRAIN_K`, do NOT change the
epochs, do NOT re-freeze anything: the matrix also measures the primary
three-arm comparison, which does not depend on the F rows at all.

**5. If anything fails: STOP.** Do not retry, do not edit a finished run's
`metrics.json` / `resolved_config.json` / `results.tsv`, do not change the
config to make it fit, and do not continue the matrix around a failure.
`HANDOFF.md` §7 has the protocol: report the failure with its verbatim
output, the run directory, and what you checked. The completed cells are
intact and `--all` resumes by skipping them.

**6. Never change** (`HANDOFF.md` §8 has the full list with reasons): the
seeds 42–46; `code/config_opdiff.json`; `num_experts` 7 for MoE/MoRE and
`ffn_mult` 28 for MoR; `batch_size` 48; `seq_len` 384 and `vocab_size` 37;
`loss_weights.family_cls` 0.0; halting supervision off; `step_routing` 0.0;
`experiment_group` exploratory; the dataset and its three k lists; the nulls
in `code/canonical_spec_opdiff.json`. Each of these is a protocol field, and
a run that differs from the others is a different study that cannot be
compared to them.

**7. Report back** (`HANDOFF.md` §12): the pilot's measured epoch times and
the corrected budget; the full summariser output **including the
compute-versus-k table**; the per-operation and per-group depth tables; any
failure with its verbatim output; and the `dataset_version` your machine
produced — if it differs from the gate machine's, say so, because the rows
were sampled differently and the runs are on a different dataset. Do not
paste numbers into a table by hand: the summariser is the table.

---
