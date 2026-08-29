# INVALIDATED — first T9.1 matrix attempt, 2026-08-28

These six runs are **not** admissible as canonical Phase-B rows and were archived
rather than re-stamped.

## What they are

The first attempt at the T9.1 canonical matrix (`code/run_phase9_matrix.py`).
Five completed runs plus one killed mid-training:

| dir | epochs | metrics.json | experiment_group |
|---|---|---|---|
| `phaseB_moe_seed42__c35fe8b6`  | 50 | yes | `exploratory` |
| `phaseB_mor_seed42__87d75af9`  | 50 | yes | `exploratory` |
| `phaseB_more_seed42__989e89cc` | 50 | yes | `exploratory` |
| `phaseB_moe_seed43__33afa654`  | 50 | yes | `exploratory` |
| `phaseB_mor_seed43__93eb15bc`  | 50 | yes | `exploratory` |
| `phaseB_more_seed43__909a3dc4` | killed at epoch 44 | **no** | `exploratory` |

Every *enforced* field matched the frozen spec: `variant = canonical`,
`seed_declared = True`, `resolved_epochs = 50`, `resolved_batch_size = 768`,
`resolved_subset_fraction = 1.0`. Only the group field is wrong — and that is
the one field the exporter filters on.

## Why they are invalid

`code/run_phase9_matrix.py` launched them without `--experiment_group
canonical_phase_b`. That flag is not a config override; it is the *claim* the
Gate 0 guard adjudicates. With `logging.experiment_group = None`,
`assert_not_silent_proxy` (`code/more/run_context.py:256`) returns
`NONCANONICAL_GROUP_DEFAULT` — no error, no refusal, by design, so exploratory
work stays cheap. The result is a run that looks canonical in every respect and
is stamped a proxy.

The guard was never broken. Verified in-process: with
`logging.experiment_group = 'canonical_phase_b'` and `provenance.seed = 42`, the
guard returns `'canonical_phase_b'` for moe, mor and more at config defaults.

## Why they were archived and not re-stamped

An experiment group is **declared at launch and validated by the guard**. Editing
`experiment_group` onto a finished `resolved_config.json` would produce a
canonical label that no guard ever adjudicated — exactly the silent-proxy failure
the guard exists to prevent (CLAUDE.md §5, §6). Evidence is archived, never
deleted; these runs remain readable as an exploratory 50-epoch reference.

## The fix

`launch()` in `code/run_phase9_matrix.py` now passes
`--experiment_group canonical_phase_b`, and `find_dir(..., canonical_only=True)`
gates the resume path so an `exploratory` directory can never satisfy the skip
check (otherwise the failure would become permanent — the driver would skip the
pair forever and report it as excluded on every invocation).

Cost of the defect: ~1.6 h of compute. See `changelog.md` (T9.1) and
`TASKS.md` (T9.1).
