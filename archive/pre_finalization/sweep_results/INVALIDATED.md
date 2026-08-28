# INVALIDATED FOR SCIENTIFIC COMPARISON

Reason: confirmed target leakage and/or broken halting/routing implementation.

**Scope: every file in `archive/pre_finalization/sweep_results/` and all subdirectories.**

## Why

The pipeline that produced these artifacts had six confirmed defects:

1. **Target leakage** — each computation step's `result` was written
   into that step's input feature row, so the regression target was
   present in the input for 100% of validation records. Reported
   losses are worse than a closed-form copy baseline.
2. **Untrained halting** — the halting loss had no `grad_fn`; every
   halt-head parameter received `grad = None`. Depth allocation was
   never learned.
3. **Dense routing** — every expert was evaluated and blended, so
   these are not Top-1 sparse MoE/MoRE results.
4. **Objective scaling** — the balance term reached ~146x the task
   loss and accumulated per recursion depth, so total loss was
   dominated by, and confounded with, depth.
5. **Dimension hard-coding** — 7-class heads in 6-expert runs.
6. **Input mismatch** — MoE / MoR / MoRE received different feature
   scalings, and the oracle expert label occupied input slot 0.

## What this means

- These numbers are **research history**, not results.
- They must not appear in any table, figure, or baseline.
- The exporter must refuse them (`experiment_group` filter).
- They are kept because deleting evidence is prohibited
  (`updated_rules.md` 12).

## Authoritative replacements

`plan.md`, `updated_rules.md`, `updated_objective.md`, and — once the
corrected canonical matrix has run — `runs/<experiment_id>/` plus
`results.md`.
