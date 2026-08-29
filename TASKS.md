# TASKS.md — MoRE Execution Task List

Derived from [plan.md](plan.md), constrained by [updated_rules.md](updated_rules.md)
and [updated_objective.md](updated_objective.md). Operating rules: [CLAUDE.md](CLAUDE.md).

> **Tooling note.** A `TaskCreate` function is not available in this
> environment, so this file is the task register and is maintained with ordinary
> file edits.

**Ticking rule.** A box may only be checked when its **Verify** command/criterion
has actually been executed and passed. Record the observed evidence inline. If a
gate fails: STOP, report, identify cause, fix, re-run the gate (`plan.md` §20).

**Legend** — `[x]` verified complete · `[~]` in progress · `[ ]` not started ·
`[!]` blocked · `[✗]` **closed WITHOUT running it** — the task's scope was
deliberately narrowed and the reason is recorded inline. A `[✗]` is not a failure
and not a silent drop: it is a written decision that the experiment would not
answer its own question, or would answer a question that cannot enter the paper.
Introduced at T11.0c. Never convert a `[✗]` to `[x]`; if the arm is later run,
open a new task so the narrowing decision stays legible.

---

## Phase 0 — Quarantine the Invalid Experimental Record  (`plan.md` §2)

- [x] **T0.0 Governing documents.** Create a root `CLAUDE.md` condensed from
  `updated_rules.md` + `updated_objective.md` so the rules are always in
  context (`plan.md` §1).
  **Verify:** `CLAUDE.md` exists at repo root, names the three authoritative
  documents, and states that the archived `rules.md`/`objective.md` are
  superseded.
  **Evidence:** `CLAUDE.md` written, 9 sections. A root `CLAUDE.md` is
  auto-loaded, so no `.claude/` copy is needed.

- [x] **T0.1 Mark the archived record invalidated** (`plan.md` §2.1). Every
  artifact under `archive/pre_finalization/` carries the exact header
  `INVALIDATED FOR SCIENTIFIC COMPARISON` / `Reason: confirmed target leakage
  and/or broken halting/routing implementation.` Evidence preserved, not
  deleted.
  **Verify:** `python archive/mark_invalidated.py --check` exits 0 with
  0 unmarked artifacts; measurement files remain byte-identical.
  **Evidence:** `already marked: 1745`, 0 gaps. Text docs (`.md`/`.py`/
  `final_plan`) have the header prepended in place; `.csv`/`.tsv`/`.json`/`.pt`
  are untouched and carry sibling `*.INVALIDATED.txt` markers; 5 directory-level
  `INVALIDATED.md` notices. `cannonic.md` — which previously read "use this data
  as the source of truth" — now opens with the invalidation block.

- [x] **T0.2 Per-run output directories** (`plan.md` §2.2). No global
  `best_model.pt` / `results.tsv` / `final_run_metrics.json` /
  `*_results.csv`. Every run owns `runs/<experiment_id>/` with `config.json`,
  `resolved_config.json`, `metrics.json`, `results.tsv`, `checkpoint.pt`,
  `stdout.log`.
  **Verify:** run `train.py` and confirm all six files appear in the run
  directory and none of the three global filenames appear in `code/`.
  **Evidence:** 2-epoch dummy run produced all six files in
  `runs/phase0_probe__seed42__b5257aa6/`; `code/` contains none of the three.
  Stale copies moved to `archive/pre_finalization/stale_global_outputs/`.
  Implemented in [code/run_context.py](code/run_context.py).

- [x] **T0.3 Resolved configuration is the single source of truth.** CLI
  overrides are written *into* the config that trains and is logged, fixing the
  defect where `--epochs 1` still logged `epochs: 50`.
  **Verify:** run with `--epochs 2` against a config saying 50; confirm
  `resolved_config.json` reports 2 while `config.json` preserves 50.
  **Evidence:** `resolved_config.json: training.epochs = 2`,
  `provenance.resolved_epochs = 2`; `config.json: training.epochs = 50`.

- [x] **T0.4 Autonomous canonical experimentation disabled** (`plan.md` §2.3).
  Sweep/autoresearch drivers quarantined off the canonical path.
  **Verify:** no proxy-search driver remains in `code/`; `code/` holds only
  `train.py`, `run_context.py`, `verify_pipeline.py`, `smoke_test.py` + configs.
  **Evidence:** `automated/` holds `autoresearch_runner.py`, `run_sweeps.py`,
  `rerun_mor.py`, `run_remaining_tests.py`, `clean_csv.py`.

- [x] **T0.5 Gate 0(c) proxy guard.** A run cannot silently claim canonical
  status. `code/canonical_spec.json` is the frozen definition; unfrozen (`null`)
  fields block every canonical claim until Phase 8 freezes them.
  **Verify:** (a) exploratory run allowed and stamped non-canonical;
  (b) `--experiment_group canonical_phase_b` with proxy settings refused with a
  non-zero exit *before any compute*; (c) a fully compliant run against a
  frozen spec accepted.
  **Evidence:** (a) group `exploratory`; (b) exit code **2**, five problems
  named (unfrozen fields, epochs 2≠50, batch 8≠768, blocks 2≠1, missing
  `dataset_version`), and **no run directory created**; (c) accepted as
  `canonical_phase_b`. Bad seed (99) and `subset_fraction=0.5` also correctly
  refused.

- [x] **T0.6 `.gitignore`.** Exclude `wandb/`, `__pycache__/`, checkpoints and
  logs; **keep** the small per-run provenance/metric files tracked so figures
  stay traceable to run IDs (`updated_rules.md` §9).
  **Verify:** `.gitignore` exists; `git status` does not list `wandb/` or
  `__pycache__/`; already-tracked archive `.pt` evidence is unaffected.

- [x] **T0.7 Repair the pre-flight harness broken by T0.2.**
  `verify_pipeline.py` copied only `train.py` into its sandbox and asserted a
  sandbox-local `results.tsv`.
  **Verify:** harness copies `run_context.py` + `canonical_spec.json`, locates
  `runs/<experiment_id>/`, asserts all six artifacts, asserts the three global
  filenames are absent, and cleans up the sandbox run directory.

### GATE 0 — status: **PASS** (see verification report in chat)

- [x] (a) historical results clearly marked invalidated
- [x] (b) canonical output paths are per-run
- [x] (c) final experiments cannot silently use proxy settings

---

## Phase 1 — Repair the Task Before the Model  (`plan.md` §3) — Step 2, 3

Acceptance test for the whole phase: [code/audit_leakage.py](code/audit_leakage.py),
measurements in [runs/gate1_audit.json](runs/gate1_audit.json). **8/8 checks pass**
(`python audit_leakage.py --json ../runs/gate1_audit.json`, exit 0).

- [x] **T1.1 Remove target leakage** (§3.1). The per-step tokenizer wrote each
  step's `result` into that step's feature row; `data/script.py`'s `verify()`
  guarantees `output == steps[-1]["result"]`, so the target was in the input.
  Fixed in [more/data.py](code/more/data.py): `numeric_vals` is built from
  `flat_args` only; `scalar_result` is computed, `del`eted, and never written.
  **Verified — mutation test (decisive), not a threshold.** Multiplying every
  step `result` and `output` by 3 and adding 137 changes **368/400 targets**
  (check A0: the test is not vacuous) and **0/400 input feature rows**
  (check A). Pre-fix, every record's features changed.
  Copy baseline rose from **0.0** to **0.081853** ≥ predict-train-mean
  **0.080914** (check E).
  *Reported diagnostic, not a failure:* the target still coincides with some
  argument in 33.45% of val / 33.54% of train records, worst single slot 16.06%
  (pre-fix: 100% of records, worst slot 74.15%). For MAX/MIN/MEDIAN/SORT the
  answer **is** one of the arguments — that is the task, not a leak.

- [x] **T1.2 Remove the oracle family ID from the input** (§3.2). Deleted
  `row[0] = expert_id / max(num_experts - 1, 1)`. Operation identity now travels
  in `step_ops` and is consumed by `nn.Embedding(NUM_OP_TYPES=16, d_model)` in
  [more/model.py](code/more/model.py) — an embedding over *operations*, never
  over experts. `step_ops` is a required argument: `forward` raises without it.
  **Verified — permutation test, stronger than the planned probe.** Expert
  indices are arbitrary labels, so cyclically relabelling `OP_TO_EXPERT`
  (`e -> (e+1) % 6`) must leave the input untouched: **400/400 records changed
  `step_experts`** (check B0, not vacuous) and **0/400 changed features**
  (check B). A probe bounds recoverability empirically; permutation proves the
  information is absent by construction. Check B2 additionally confirms no
  low-cardinality slot maps 1:1 onto the expert label (majority-class rate
  0.3026, offending slots `[]`).

- [x] **T1.3 Input parity test** (§3.3). **Verified:** features, `step_mask`,
  `step_ops` and `target` are bit-identical (`torch.equal`) across
  `E = 1 / 5 / 6` over 200 val records — 0 mismatches (check C). Slot-0 range is
  now `[-1, 1]` for all three. *Pre-fix:* `[0, 5.0]` at E=1, `[0, 1.0]` at E=6,
  `[0, 0.833]` at E=7 — not parity, so MoE/MoR/MoRE were not solving the same task.

- [x] **T1.4 Canonical dataset manifest** (§3.4). Regenerated via
  [data/script.py](data/script.py) with global program dedup.
  **Verified:** `dataset_version = more6-v1-seed42-n70000-3c1087b6aad9`;
  70,000 verified examples, **133 duplicate programs rejected**; splits
  59,500 / 5,250 / 5,250; all three manifest SHA-256s **MATCH** the files on
  disk; `dataset_version` is frozen into
  `canonical_spec.json:enforced_fields.dataset_version` and **is consumed by the
  guard** — a canonical claim omitting `data.dataset_version` is refused citing
  it, and a claim supplying it is no longer refused for that reason (the guard
  still refuses, correctly, on the four fields Phase 8 has yet to freeze:
  `d_model`, `lr`, `weight_decay`, `routing_balance`).
  **E7 redefinition (documented, not silent):** the generator's E7 slot is the
  multi-operation CHAIN family at 0.40 weight (28,000 records, the only source
  of depth 4–7), not a seventh expert. Per §3.4 the mass is **redefined, not
  dropped**: records emit `family: "MIXED"` with whole-program index **−1**,
  which the classification loss ignores (`ignore_index=-1`,
  [more/engine.py](code/more/engine.py)); their individual steps still route to
  E1–E6. Dropping the mass would have deleted every depth 4–7 program and made
  adaptive-depth allocation — the phenomenon under study — unmeasurable. Rule
  recorded in the manifest under `family_label_redefinition`.
  Per-family average depth: E1 1.81 · E2 2.21 · E3 3.01 · E4 1.81 · E5 1.80 ·
  E6 3.40 · MIXED 5.50.

- [x] **T1.5 Leakage audit + trivial baselines** (§3.5).
  **Verified:** `audit_leakage.py` exits **0**, 8/8 checks. Split overlap on
  `(op, args, result)` program keys is `{train|val: 0, train|test: 0,
  val|test: 0}` (check D) — it was `{test|train: 22, train|val: 14}` before
  regeneration, which is the failure T1.4 was run to fix. All three baselines
  are recorded in the manifest under `trivial_baselines`:
  predict-zero **0.081853**, predict-train-mean **0.080914**,
  best-single-feature-copy **0.081853** (slot 8), val target variance 0.080899.
  *Pre-fix reference:* copy was **0.0** — the task was solvable by copying.

### GATE 1 — PASSED (8/8)
- [x] no leakage · [x] input parity holds · [x] trivial baselines recorded

**End-to-end confirmation on the regenerated data** (not part of the gate, but it
proves the new labels train): `train_more.py --epochs 1` exits 0, 6,359,069
params, `task=0.0820 val=0.0782 route_acc=0.4274 entropy=1.790`. No NaN from
all-MIXED batches — [more/engine.py](code/more/engine.py) emits an exact zero
when a batch's family targets are entirely ignored, instead of letting
`cross_entropy` average over zero elements.
Two known-open observations, both owned by later phases and **not** silently
accepted: `bal=-8.7716` is the unnormalized balance loss (Phase 4, T4.1), and
entropy 1.790 ≈ `log(6)=1.7918` means routing is still near-uniform, which is
the load-balance diagnostic — not evidence of specialization (§4 of CLAUDE.md).

---

## Phase 2 — True Top-1 Sparse Routing  (`plan.md` §4) — Step 5

**Gate:** `python code/test_phase2_routing.py` → **19/19 checks passed**.
Every check is a *measurement* of the running code via forward hooks on the
expert modules, not an inspection of it — the dense/sparse difference is
invisible in the loss, the entropy and the confusion matrix, so counting the rows
each expert's `forward()` actually receives is the only external evidence.

- [x] **T2.1 Sparse dispatch** (§4.1). Replace the dense blend
  (`train.py` ~404–405) with argmax dispatch to one expert, multiplied by the
  selected gate probability.
  **Verify:** instrument expert forward calls — each token enters exactly one
  expert; a gradient-flow test shows non-selected experts receive no gradient
  from that token.
  **Evidence:** implemented in `more/model.py` `MoEBlock.forward`, branch
  `routing_mode == "top1_sparse"`.
  - T2.1a experts received **64 rows for 64 tokens** (dense would be 384);
    per-expert `{0:5, 1:7, 2:14, 3:13, 4:8, 5:17}`.
  - T2.1a-nv non-vacuous: **6/6** experts actually called, so the count of 64
    is not a collapsed router.
  - T2.1a-disc the check **discriminates** — the dense path delivers
    `384 = N×E` rows to the same hooks.
  - T2.1b gradient isolation: chosen expert 3 grad `572.019427`, all others
    **exactly 0.000000**.
  - T2.1c router grad through the gate multiply `30.010744` — non-zero, so the
    argmax did not sever the router from the task loss.
  - T2.1d `out == selected_expert(x) × gate` to **5.960e-08**; gate range
    `[0.2806, 0.4429]`.
  - T2.1e/f compute honesty: `expert_evaluations` equals the hook-counted rows
    on **both** paths (sparse 64, dense 384) and dense is exactly `E×` sparse.

- [x] **T2.2 Route output and routing metrics agree** (§4.2).
  **Verify:** the expert index used for dispatch is the same tensor reported by
  the routing metrics; confusion-matrix diagonal fraction equals
  `mean(pred == oracle)` within 1e-6.
  **Evidence:** `expert_idx` is one tensor, used for dispatch and returned to the
  metrics — the divergence is made impossible rather than tested for.
  - T2.2a **64/64** tokens traced from an expert's forward call back to their row
    in `x`, matched against the reported index: **0 mismatches**.
  - T2.2b direct `0.1562500000` vs `routing_accuracy_from_confusion`
    `0.1562500000`, delta **0.00e+00** (tolerance 1e-6).

- [x] **T2.3 Capacity / overflow policy** (§4.3). Document and test the chosen
  policy (drop / no-capacity-limit); report the overflow rate.
  **Verify:** overflow rate logged as a real number, never a sentinel.
  **Evidence:** `CAPACITY_POLICY = "no_capacity_limit"` in `more/model.py`, with
  the `plan.md` §4.3 justification in-source (single-machine synthetic setup;
  capacity factors bound per-device buffers in distributed expert-parallel
  training, which this is not; a dropped token would also have no honest exit
  depth). `overflow = 0` is therefore a **measured** zero, not "not implemented".
  - T2.3b `tokens = dispatched = 64.0`, `overflow = 0.0`.
  - T2.3c real capacity pressure `max_load_fraction = 0.2656` (uniform 0.1667,
    full collapse 1.0) — a collapsing router stays visible.
  - T2.3d/e model-level `overflow_rate = 0.0`, `evals_per_token = 1.0`,
    `dispatch_steps = 6.0`, `capacity_policy = 'no_capacity_limit'`.
  - Logged per epoch under `dispatch/*` and written to `metrics.json`.

- [x] **T2.4 Keep dense as a labelled ablation only** (`updated_rules.md` §1.1,
  ablation F). **Verify:** dense path reachable only via an explicit flag whose
  run label contains `dense_routing_ablation`.
  **Evidence:** `enforce_routing_mode()` in `more/config.py`, called from
  `more/cli.py` **after** override resolution (running it in
  `apply_architecture` tested the default label and refused correctly-labelled
  runs). Gating is on the run label, not a boolean, so the dense path appears in
  the run directory, the W&B name and all provenance.
  - T2.4a unlabelled dense **REFUSED**, error names the required label.
  - T2.4b labelled dense **ALLOWED**, `variant='dense_routing_ablation'`.
  - T2.4c canonical default resolves to `top1_sparse`.
  - T2.4d unknown mode (`"top2"`) raises.
  - Ablation run dir `phaseB_more_dense_routing_ablation__seedNA__cd905ca5`,
    `dispatch/routing_mode='dense_blend'` — un-confusable with canonical MoE.

**Regression check after Phase 2:** `smoke_test.py` ALL TESTS PASSED,
`verify_pipeline.py` 35/35, `audit_leakage.py` Gate 1: 8/8.

---

## Phase 3 — Real ACT Halting  (`plan.md` §5) — Step 6

- [x] **T3.1 Implement ACT halting** (§5.2): per-step halt probability,
  cumulative halting mass, remainder, frozen halted states, forced exit at
  `max_depth`.
  **Verified:** `code/more/model.py` `MoREWrapper.forward` now implements the
  canonical Graves-2016 / Universal-Transformer recurrence
  (`plan.md` §5.2: "do not invent a novel halting equation") — 7/7 checks:
  - T3.1a per-token step weights sum to **1.000000000** (convex combination).
  - T3.1b active rows per depth step `[28, 28, 16, 5, 1]`, non-increasing — a
    halted token never re-enters the block. Measured with a forward pre-hook on
    the `MoEBlock`, not read off the source.
  - T3.1c every real token exits exactly once (min = max = 1.0 over 28 tokens).
  - T3.1d halt bias forced to −20 → `forced=28 early=0`, all tokens exit at
    `max_depth`.
  - T3.1e halt bias forced to +20 → `forced=0 early=28`, all exit at depth 1.
    d and e together prove the exit-rate metric measures behaviour, not a
    constant.
  - T3.1f remainder at a depth-1 exit = **1.000000000** = `1 − c_0`, pinning R
    to the ACT definition.
  - T3.1g 12 pad tokens, **0** exits recorded — pads never enter the loop.

- [x] **T3.2 Differentiable ponder cost** (§5.3), replacing
  `active_mask.float().mean() * 0.05`.
  **Verified:** 5/5 — `ponder_cost.requires_grad = True`,
  `grad_fn = DivBackward0`, `expected_depth` (N + R) `grad_fn = AddBackward0`.
  Normalized `(N + R) / (max_depth + 1)`: `0.480394` at `max_depth=5`;
  `max_depth=3 → 0.720968`, `max_depth=9 → 0.303979`, both inside (0, 1], so the
  term's magnitude does not grow with depth (CLAUDE.md §2). Averaged, not summed,
  over blocks for the same reason.

- [x] **T3.3 Operation-complexity curriculum** (§5.4) wired as an explicit,
  reportable supervision option (`updated_rules.md` §2.3).
  **Verified:** 9/9. Curriculum is `families.OP_TARGET_DEPTH`, 16 targets for 16
  operations, declared in-source as a **design choice, not a measurement**.
  - T3.3b target depth is **not** a pure function of the expert index: E6
    spans `[2, 4]`, and depths 1 and 2 are each shared across experts — so a
    model cannot score perfect depth allocation by routing alone.
  - T3.3e depth allocation error is **absolute**: a +3 and a −3 token give
    `3.0`, not `0.0`.
  - T3.3f returns `None` → caller writes `"N/A"`, never `0.0` (which would read
    as perfect allocation).
  - T3.3g/h default is **OFF** (pure ACT); enabling writes
    `halting_mode='supervised_curriculum'`, `halting_supervision_enabled`,
    `halting_supervision_weight` and `ponder_weight` into `resolved_config.json`,
    `metrics.json` and the W&B config. Confirmed in two real runs:
    `pure_act` vs `supervised_curriculum`.
  - T3.3i MoE (`max_depth = 1`) force-disables it; `--halting_supervision` on
    MoE is **refused**, and `--halting_supervision_weight` without the flag is
    refused, so a run can never be labelled supervised while training otherwise.

- [x] **T3.4 Halt-gradient acceptance test** (§5.5).
  **Verified:** 3/3, backward through the **task loss only**:
  **24/24** `expert_halt_heads.*` parameters have `grad is not None` with
  total |grad| = **0.649250**; `grad is None = 0`, all-zero grad = 0.
  *Measured pre-fix:* `halt_loss.requires_grad = False`, `grad_fn = None`,
  `grad is None` on **all 24** halt-head parameters — halting was never trained.
  - T3.4b non-vacuity: freezing the halt heads gives `grad is None` on all 24,
    so T3.4a reads the halt path and not an unrelated one.
  - T3.4c the ponder cost alone also reaches 24/24, so it is a real regularizer.

### GATE 2 — [x] halting is genuinely trained; halt-gradient test passes

`python code/test_phase3_halting.py` → **29/29 checks passed.**

`plan.md` §5.5's known-target-depth synthetic case, both directions:
- GATE2a/b heads biased to halt at step 1, all tokens `SORT` (target depth 4):
  one optimizer step moves expected depth `2.0474 → 2.0618` **up** and the
  curriculum loss `0.105904 → 0.104354` **down**.
- GATE2c heads biased never to halt, all tokens `ADD` (target depth 1):
  expected depth `6.7629 → 2.2913` **down** — opposite direction, so the test
  measures a target-driven gradient and not "SGD reduces whatever it is given".
- GATE2d/e ponder cost and curriculum supervision are logged under **separate**
  keys (`plan.md` §5.4: "do not collapse them into one opaque number"), and
  undefined halt/depth quantities are written as `"N/A"`.

**End-to-end on the full dataset (1 epoch, unseeded, offline W&B, all exit 0).**
Halting is now measurably active and the three modes are distinguishable:

| | avg depth | early-exit rate | forced-exit rate | halt mass | ponder cost | depth alloc err |
|---|---|---|---|---|---|---|
| MoE  | 1.000 | N/A | N/A | N/A | N/A | N/A |
| MoR  | 2.000 | 0.99998 | 0.000025 | 1.0000000 | 0.274614 | 0.930909 |
| MoRE | 4.756 | 0.87586 | 0.124137 | 1.0000000 | 0.474371 | 2.718111 |

MoE's column is `N/A` by construction, not by omission: at `max_depth = 1` the
ponder cost is `(1+1)/2 = 1.0` for every token always, and reporting that beside
MoR's `0.27` would read as "MoE ponders hardest" — the sentinel-as-measurement
failure of T6.4.

**Supervised vs unsupervised halting, measured** (MoRE, 1 epoch each):

| | halting_mode | val task loss | avg depth | depth alloc err (abs) |
|---|---|---|---|---|
| pure ACT | `pure_act` | **0.072163** | 4.756 | 2.718 |
| curriculum | `supervised_curriculum` | 0.074127 | 2.206 | **0.729** |

Read honestly: unsupervised ACT does **not** learn the curriculum on its own —
2.72 steps of allocation error on a 1–4 target range is close to uninformative.
Supervision cuts that to 0.73 at a small cost in val task loss. Neither number is
publishable (1 epoch, unseeded); the point is that the two modes now produce
*different, correctly labelled* artifacts.

**Regression check after Phase 3:** `smoke_test.py` ALL TESTS PASSED,
`verify_pipeline.py` 35/35, `audit_leakage.py` Gate 1: 8/8,
`test_phase2_routing.py` 19/19.

---

## Phase 4 — Fix the Balance Objective  (`plan.md` §6) — Step 7

- [x] **T4.1 Normalize over (block, depth) calls** (§6.1). Replace the
  accumulation `total_bal_loss = total_bal_loss + b_loss` inside the depth loop
  (~line 563) with a mean over calls.
  **Verify:** with `max_depth` 3 vs 7 and the same router state, balance loss
  differs by < 5%.
  *Measured pre-fix:* `aux_routing_loss = -4.861` ≈ 5 × the per-depth value,
  and `0.05 × (-4.861) = -0.243` ≈ **146×** the task loss (0.001666).
  **DONE.** Mean on both axes: over the *realised* depth calls inside
  `MoREWrapper.forward` (not `max_depth` — ACT can break early), then over
  blocks in `MoREModel.forward`. `metrics.json` now stamps
  `routing_balance_normalization`. Measured, `code/test_phase4_balance.py`:
  T4.1a wrapper returns `-0.749462724` == mean of 5 calls `-0.749462700` (sum
  would be `-3.747313`); T4.1c model returns `0.300454140` == flat mean over 10
  calls `0.300454104`; **T4.1d depth 3 vs 7 delta 0.180 %** (< 5 % required)
  against **132.9 %** for the unnormalized sum on identical routing;
  T4.1e the sum really did scale (ratio `2.329` ≈ 7/3), so the tolerance is not
  vacuously satisfied; T4.1f does not grow with `num_blocks` (ratio `0.5832`
  normalized vs `2.333` summed); T4.1g the aggregate lies inside
  `[min, max]` of its 14 per-call values (`0.026618` in
  `[-0.474102, 0.343925]`) — a sum cannot.

- [x] **T4.2 Re-select the balance weight** (§6.2) after normalization.
  **Verify:** `|balance_term| / |task_loss| < 1` throughout a smoke run; the
  entropy term and Switch-aux term are logged separately.
  **DONE — frozen at `routing_balance = 0.001`** in both `code/config.json` and
  the `more/config.py` default (T4.2h asserts the two cannot drift). Selected
  from two 1-epoch full-dataset MoRE runs, identical but for the weight:

  | | `w = 0.05` (historical) | `w = 0.001` (frozen) |
  |---|---|---|
  | `train/task_loss` | 0.078803 | 0.078808 |
  | `train/routing_balance_loss` | −0.653436 | +1.365229 |
  | **`train/balance_to_task_ratio`** | **0.414599** | **0.017323** |
  | projected ratio at converged task 0.001666 | ≈ 19.6 | ≈ 0.82 |
  | `train/entropy_term_normalized` (soft) | 0.986127 | 0.694558 |
  | `train/switch_aux_term` | 1.113466 | 2.609710 |
  | `train/expert_load_entropy_normalized` (hard) | 0.989215 | 0.955612 |
  | hard expert load min–max % | 13.4 – 23.6 | 9.5 – 29.3 |
  | `val/routing_accuracy` | 0.922945 | **0.997934** |
  | `val/task_loss` | 0.072073 | 0.074507 |

  Runs: `runs/phase4_measure_w005__seedNA__f01c4d6a`,
  `runs/phase4_confirm_w0001__seedNA__ffdbb048`.
  Both satisfy the stated `< 1` criterion after warmup; `0.001` was chosen
  because it is the only tested weight that stays subordinate when projected
  onto the converged task loss this repo has actually reached (0.001666) —
  `0.05` re-dominates at ≈ 19.6× by the end of a 50-epoch run, which is the
  defect Phase 4 exists to remove. No expert is starved at `0.001` (min 9.5 %).
  Entropy and Switch-aux are logged separately as
  `train/entropy_term`, `train/switch_aux_term`,
  `train/entropy_term_normalized`.
  *Recorded honestly:* `val/task_loss` is 3.4 % worse at `0.001` (0.074507 vs
  0.072073) at 1 epoch, 1 seed, with global seeding still absent (**T6.1**).
  That is not a basis for a quality claim either way, and weight-vs-quality is a
  seeded hyperparameter question for Phase 7, not a Phase 4 decision.

- [x] **T4.3 Gradient-direction tests** (§6.3).
  **Verify:** with an artificially collapsed router, the balance gradient
  increases entropy; with a uniform router the gradient magnitude is near zero.
  **DONE**, 7 checks. T4.3a collapsed router: `entropy_term 0.820246 →
  1.789653` (max `1.791759`), first-step `|grad| 3.20431`; T4.3b
  `switch_aux 4.804092 → 1.059087` (min 1.0); T4.3c hard
  `max_load_fraction 1.0000 → 0.8083`; T4.3e uniform routing sits exactly on the
  analytic optimum `−log(6) + 1 = −0.791759` with `switch_aux = 1.000000` and
  `max_load = 0.166667`; T4.3f `|grad|` uniform `1e-8` vs collapsed `3.204310`
  (ratio `1.86e-09`). Two limitations recorded rather than hidden: T4.3d soft
  entropy `0.9988` coexists with hard `max_load_fraction 0.8083` (entropy is
  blind to the argmax), and T4.3g a fully saturated router has numerically zero
  balance gradient (`entropy 1.879e-07`, `|grad| 2.352e-07`) — the term
  discourages collapse, it cannot reverse saturation (that is **T5.4**).

### GATE 3 — [x] balance loss depth-invariant · [x] gradient direction correct · [x] task loss dominates

**27/27 checks pass** — `C:/Users/Hp/anaconda3/envs/more_env/python.exe code/test_phase4_balance.py`.
The three gate criteria map to: T4.1a–g (depth- and block-invariance, measured
0.180 % across `max_depth` 3 vs 7), T4.3a–g (gradient direction and its two
limits), and G3.1 (`train/balance_to_task_ratio = 0.017323 < 1` read out of a
real run's `metrics.json`, not a synthetic tensor). G3.2–G3.4 assert the
decomposition and the normalization stamp survive into the artifact a results
table is built from; G3.5 asserts a single-expert run writes `"N/A"` for all six
balance keys — with `E = 1` the objective is the constant `1.0` forever
(`bal = 1.0000` measured in `runs/phase4_mor_nagate__seedNA__3148ae84`), and
tabled beside MoRE's value that constant would read as "MoR is maximally
imbalanced".
No regression: Gate 1 19/19, Gate 2 29/29 re-run after the change.

---

## Phase 5 — Dimensions and Config Precedence  (`plan.md` §7) — Step 4, 8

- [x] **T5.1 `num_experts` genuinely dynamic** (§7.1). Replace
  `nn.Linear(d_model, 7)` (~lines 698, 703) and
  `step_cls_out.reshape(-1, 7)` (~1267) and `torch.zeros(7, 7)` (~1323–1325).
  **Verify:** at `num_experts = 6`, `cls_head.out_features == 6` and
  `step_cls_head.out_features == 6`. *Measured pre-fix: both were 7.*
  **DONE.** Measured at three widths (Gate 4 G4.1–G4.7, G4.39):
  `E=1: cls=1 step_cls=1 router=1 experts=1 halt_heads=1`;
  `E=6: 6/6/6/6/6`; `E=7: 7/7/7/7/7`. `step_cls_out` reshapes
  `(4,3,E) -> (12,E)` at all three, so step tokens can no longer be blended into
  one another's logits by a 7-wide view. `max_depth`/`max_steps` deliberately
  left at 7 — a different quantity that shares the value (G4.5). `smoke_test.py`
  now prints `cls_out (4,6)`, `step_cls_out (4,7,6)`: 7 steps, 6 experts.

- [x] **T5.2 Remove the E7 fallback** (§7.2). Delete `_EXPERT_FALLBACK = 6`
  (~line 160); an unmapped operation must raise.
  **Verify:** a synthetic unknown op raises a clear error rather than routing to
  a catch-all.
  **DONE.** G4.17–G4.21: 16 ops map onto experts {0..5} with no catch-all;
  `_EXPERT_FALLBACK is None`; a record carrying `op="TOTALLY_UNKNOWN_OP"` raises
  `KeyError` naming the op and the record; family label `E99` raises; `MIXED` and
  the legacy spelling `E7` both map to the ignore label −1 with highest real
  index 5.

- [x] **T5.3 Config precedence, one source of truth** (§7.3). Remove the
  `load_config` shim (lines ~101–129) whose `setdefault` + `if key in` pattern
  makes `training.*` unconditionally overwrite `data.*`, which silently
  discarded `data.subset_fraction: 0.5` in all six 50-epoch configs.
  **Verify:** a config specifying `data.subset_fraction = 0.5` results in
  `resolved_config.json` reporting 0.5, and the data loader consuming 0.5.
  **DONE.** G4.22–G4.28: `data`-only 0.5 → 0.5/0.5; `training`-only 0.25 →
  0.25/0.25; neither → 1.0/1.0; both-equal 0.3 accepted; conflicting 0.3 vs 0.7
  raises instead of picking a winner. CLI overrides reach the resolved config
  (`epochs 50 → 1`, `batch_size → 64`, `seed → 43`) and `--seed` lands in both
  `subset_seed` locations so they cannot diverge.

- [x] **T5.4 Disable router noise in canonical** (Step 8). Remove the trainable
  `router_noise_scale` (~382) and its injection (~399) from the canonical path;
  keep `none | fixed-annealed | trainable` as ablation D.
  **Verify:** canonical resolved config records `router_noise = none`; noise
  parameters absent from the canonical model's `state_dict`.
  **DONE.** G4.29–G4.38: canonical `router_noise='none'`; the canonical
  `state_dict` contains **zero** noise keys (absence is verifiable — a parameter
  pinned to 0.0 would still sit in the optimizer); the trainable ablation carries
  exactly one per block; canonical dispatch is deterministic in training mode
  (64/64 tokens identical across two passes) while the noisy variant moved 9/64;
  an unknown mode raises; and the Gate 0 proxy guard refuses a canonical claim
  for `router_noise="trainable"` or `routing_mode="dense_blend"` **even while
  `canonical_spec.json` is unfrozen**.

- [x] **T5.5 Six-expert family manifest.** One manifest; W&B-safe labels
  (`E1_ADD_SUB`, not `E1 ADD/SUB`).
  **Verify:** no metric key contains `/` inside a label; all labels derive from
  the single manifest.
  **DONE.** G4.8–G4.16: `expert_labels(E)` returns exactly E labels at E=1/6/7
  and raises at E=0; `expert_labels(6) == EXPERT_FAMILY_LABELS`; no label up to
  E=9 contains `/` or a space; confusion figures carry exactly E tick labels on
  both axes; per-expert recall keys number 0 at E=1 (routing undefined) and E at
  E=6/7; the confusion diagonal fraction equals `mean(pred == oracle)` equals the
  reported scalar to 1e-9 in float64.

### GATE 4 — [x] no hard-coded 7 · [x] no silent fallback · [x] config precedence explicit

`code/test_phase5_dimensions.py`: **62/62 checks passed** (G4.1–G4.39, several
run at each of E = 1, 6, 7 — 7 is tested *alongside* 6 because everything
hard-coded to 7 is accidentally correct at E=7 and only a three-width sweep can
tell the two apart).

No regression, all re-run after the Phase 5 changes:
Gate 1 `audit_leakage.py` 8/8 · `test_phase2_routing.py` 19/19 ·
Gate 2 `test_phase3_halting.py` 31/31 · Gate 3 `test_phase4_balance.py` 27/27 ·
`verify_pipeline.py` 35/35 · `smoke_test.py` passing at the canonical width.

Phase 3's T3.4 did fail (2/29) immediately after T5.1 and was **fixed rather than
waived**: it demanded a non-zero gradient on all 24 per-expert halt parameters,
which asserts that top-1 dispatch is dense. Narrowing `cls_head` from 7 to 6
shifted the init RNG stream, so a different set of experts won the argmax and one
was starved — the old pass was luck, not a property (unused experts across the
two blocks measured 4, 3, 3, 1, 3, 1 for seeds 0–5). It is now stated as three
luck-independent claims: on a batch where all six experts *are* dispatched all 24
params are live; a dispatched expert's halt head is always on the autograd graph;
and a gradient never appears for an expert that was not dispatched to.

---

## Phase 6 — Seeding, Metrics and Provenance  (`plan.md` §8) — Step 9

- [x] **T6.1 Global seeding** (§8.1) for `random`, `numpy`, `torch`, CUDA, and
  DataLoader workers — currently only `random_split` generators are seeded.
  **Verify:** `grep -c "torch.manual_seed(" code/train.py` > 0 *(pre-fix: 0)*;
  two runs with the same seed produce identical first-epoch losses.
  **Done.** New module [code/more/seeding.py](code/more/seeding.py) —
  `seed_everything` / `apply_seeding` / `make_generator` / `seed_worker`, plus the
  frozen `CANONICAL_SEED_SET = (42, 43, 44, 45, 46)`. `engine.train()` calls
  `apply_seeding(ctx.resolved_cfg)` as its first act, before the dataset, the
  model or the optimiser exist; the DataLoader gets `generator=` and
  `worker_init_fn=`.
  *Verification path note:* the literal grep target is stale — Phase 1 split the
  1.6k-line `train.py` into `more/`, leaving `code/train.py` a 19-line launcher,
  so the count there is 0 by design and putting a seed call in it would seed the
  wrong process stage. Measured on the real path instead:
  `grep -c "manual_seed(" code/more/seeding.py` = **3** (CPU/CUDA + the
  DataLoader generator), and `engine.py:93` is the single call site.
  **Measured (`code/test_phase6_seeding.py`, 61/61 PASS, CUDA present):**
  same seed reproduces all five stream classes and seeds 42/43 differ on *every*
  one of them; parameter digest identical at 42 (`15a7bd18…`), different at 43
  (`57ff38c9…`); full forward+backward bitwise identical (loss
  1.1138908863067627, |grad| 1991.3999771339586, 137 routing slots) and different
  at 43; shuffle order reproducible, seed-sensitive, and unchanged after 1000
  global RNG draws are burned in between (the Phase-5 RNG-coupling trap);
  `seed_worker` picklable for Windows spawn. Two real 2-epoch engine runs at seed
  42 produced the byte-identical first-epoch row
  `1  0.134741  0.062896  0.7943  2.21  0.0030  0.0309  0.0241` — train task
  loss, val loss, entropy, depth, cosines and routing accuracy all matching —
  while seed 43 gave `1  0.137417  0.049722  0.8968  2.48  -0.0024  0.0231
  0.0964`. Run ids now read `…__seed42__…`, not `seedNA`.
  *Determinism honesty (§8.1 last paragraph):* `warn_only` mode, so a kernel with
  no deterministic implementation is recorded rather than fatal;
  `provenance.determinism_exceptions` lists what failed **at seeding time**,
  `CUBLAS_WORKSPACE_CONFIG=:4096:8` is exported before CUDA init, and every
  determinism field in the report is read back from torch rather than assumed.
  *Runtime exceptions are a separate field, found by running it.* A real 1-epoch
  CLI run recorded `determinism_exceptions = ["none"]` while `stdout.log` held
  **281** `_histc_cuda does not have a deterministic implementation` warnings — a
  clean-looking but false record, because a missing kernel only announces itself
  when it is first *called*, mid-training, long after seeding. Fixed by recording
  observed warnings by op name and writing them to
  `provenance.nondeterministic_ops_observed` and `metrics.json` at end of run,
  deduplicated to one printed line per op. Re-measured on
  `runs/t61_nondet_check__seed42__42b858b3`: **1** warning line instead of 281,
  `nondeterministic_ops_observed = ["_histc_cuda"]`, and the epoch row
  `1  0.081314  0.073893  0.9529  1.98  -0.0006  0.0061  1.0000` is unchanged from
  the pre-fix run, so the recorder does not touch the training math. The op comes
  from `wandb.watch`'s gradient histograms — logging, not the loss path.
  *Gate 0 unaffected:* an undeclared seed still leaves `provenance.seed` null —
  the default 42 is recorded only as `resolved_seed` / `seed_source="default"` —
  so the proxy guard still refuses such a run's canonical claim (asserted).

- [x] **T6.2 Authoritative routing accuracy** (§8.2) — one computation, shared
  by the confusion matrix.
  **Verify:** diagonal fraction equals `mean(pred == oracle)` within 1e-6.
  *Pre-fix:* accuracy 0.0403 at near-maximal entropy, i.e. below chance (1/7 =
  0.1429), with two recalls at ~0.
  **DONE.** There were two independent implementations of the routing accuracy:
  one in `metrics.evaluate_paper_metrics` and one inlined in `engine.py`'s
  validation loop. Only the engine's ran — `grep -rn "evaluate_paper_metrics"
  --include=*.py` returned a single hit, the import at `engine.py:31`, and no
  call site. A guard added to the `metrics.py` copy therefore had *no effect on
  any run*, which is how this was caught. Both now delegate to
  `metrics.routing_accuracy_from_confusion(confusion, num_experts)`, so the
  scalar is the confusion diagonal fraction *by construction* and the 1e-6
  tolerance cannot be violated. `evaluate_paper_metrics` was retained but marked
  UNUSED in-source so the next agent fixes the copy that runs; **at T11.1 it was
  deleted outright** (109 lines) and replaced by an in-source note naming the
  helpers to reuse — by then its `model(...)` unpack expected 11 return values
  against `MoREModel`'s 13, so it could not have run even if called.
  *Measured:* `test_phase6_routing_metrics.py` builds 200 random token streams
  (50–400 tokens, 60 % oracle-agreement), accumulates a confusion matrix and an
  independent `mean(pred == oracle)`, and reports **max gap 0.000e+00** across
  all 200 — the diagonal fraction, the engine's scalar and the
  permutation-invariant block's `raw_accuracy` are one number, not three that
  agree.

- [x] **T6.3 Permutation-invariant routing metrics** (§8.3): Hungarian-matched
  accuracy, AMI, cluster purity, per-family precision/recall.
  **Verify:** on a synthetic perfectly-permuted assignment, Hungarian accuracy
  = 1.0 while raw accuracy is at chance.
  **DONE — `test_phase6_routing_metrics.py` 42/42 PASS.** The stated criterion,
  measured on a cyclic (derangement) permutation of six experts:
  `hungarian_accuracy = 1.0`, `raw_accuracy = 0.0`, recovered assignment
  `[1,2,3,4,5,0]` — the planted permutation. AMI and purity are 1.0 and
  bit-identical to the identity-labelled table, which is what permutation
  invariance means operationally.
  *No scipy and no sklearn in `more_env`*, so both non-trivial algorithms are
  implemented in `more/metrics.py` and checked against brute force rather than
  against a library: the exact assignment (subset DP, `O(E²·2^E)`) matches
  exhaustive enumeration over all `n!` permutations on 40 random matrices
  (`max |Δ| < 1e-9`), and the AMI chance correction matches the literal
  permutation-model expectation enumerated over 720 permutations (agreement to
  1e-10).
  *Interpretation contract, recorded in `metrics.json` alongside the numbers:*
  raw ≈ matched means the index itself is supervised (`step_routing = 0.5`, the
  current canonical setting) — expected, not a finding; raw ≪ matched means a
  real partition exists under relabelling; both low with AMI ≈ 0 means no
  partition. On a chance-level (product) table AMI ≤ 0 while purity stays
  ≈ 1/E — reported side by side because purity is *not* chance-corrected.
  *Measured on a real 1-epoch MoRE run* (`runs/t63_pim_check__seed42__b19853cd`):
  `route_hung=1.0000 route_ami=1.0000` in the console line, two new
  `results.tsv` columns (`routing_hungarian_acc`, `routing_ami`), the
  `routing_permutation_invariant` block in `metrics.json`, and 24 `val/routing_*`
  W&B keys with **no `/` inside any label**. Task and val loss are byte-identical
  to the pre-T6.3 run (0.081314 / 0.073893) — the metric observes without
  perturbing the optimization.

- [x] **T6.4 Normalized entropy** (§8.4): report `H / log(E)`; `N/A` for `E = 1`.
  **Verify:** at `E = 6`, uniform load gives `H_norm = 1.0`; MoR runs report
  `N/A`, never `0`.
  **DONE.** `compute_expert_load_entropy` now divides by `log(E)` and returns
  `None` for `E < 2`; `engine.py` renders `None` as the string `N/A` in the
  console line, in `results.tsv`, and in `metrics.json`.
  *Measured, 1 epoch, full dataset:* MoE (E=6) `entropy_norm=0.9869`
  (near-uniform load, i.e. no expert starved — a load-balance diagnostic, **not**
  evidence of specialization); MoR (E=1) `entropy_norm=N/A`.
  *Pre-fix MoR printed `entropy=-0.000`* — raw `H` is identically 0 for one
  expert, so MoR was being reported as maximally collapsed against MoE/MoRE when
  it has no routing decision to make.

- [x] **T6.5 Cosine similarity** (§8.5): mean **and** max; `N/A` for `E = 1`.
  **Verify:** no run emits `-1.0` as a similarity. *Pre-fix:* `max_sim = -1.0`
  was published three times for MoR.
  **DONE.** `compute_pairwise_cosine_sim` returns `(mean, max)` over all
  `C(E,2)` pairs in every block, or `(None, None)` when no pair exists.
  *Measured:* MoE `cos_mean=0.000251  cos_max=0.006477`; MoR both `N/A`.
  *Pre-fix MoR printed `cos=-1.000`* — the `max_sim = -1.0` initialiser returned
  untouched, publishing a sentinel as a measured similarity.
  Two further sentinel-as-measurement defects were found and fixed while
  verifying this, both in the same class:
  1. `paper_metrics_to_wandb` logged `val/routing_recall/E1_ADD_SUB = 1.0` and
     `E2..E6 = 0.0` for MoR. With `E = 1` the argmax is 0 for every token, so
     those are artifacts of the degenerate 1x1 confusion shape, and in a table
     they read as "MoR routes E1 perfectly, starves the rest". All routing
     quantities (accuracy, per-family recall, both confusion figures) are now
     gated on `num_experts >= 2`. Depth metrics are deliberately **not** gated —
     MoR does allocate depth and that is its contribution.
  2. `RunContext.write_metrics` fell back to `str(v)` for non-JSON values, which
     wrote `"<wandb.sdk.data_types.image.Image object at 0x000001C09782C8E0>"`
     into `metrics.json` — a memory address stored as if it were a recorded
     value. Figure objects are now dropped and named under
     `_non_scalar_keys_omitted` instead.
  *Verified:* `metrics.json` for both architectures contains zero
  `"object at 0x"` strings; MoR emits no `val/routing_*` key at all; MoE emits
  the full set.

- [x] **T6.6 W&B naming** (§8.6): `phaseB_{arch}_seed{seed}`; no `/` in labels.
  **DONE — `test_phase6_provenance.py` 32/32 PASS (T6.6 section: 8 checks).**
  Naming is no longer typed by hand at the call site: `more/config.py` holds
  `ARCH_DISPLAY = {"moe": "MoE", "mor": "MoR", "more": "MoRE"}` (the *paper*
  spelling, deliberately separate from the lowercase config enum),
  `canonical_run_name(arch, seed)` and `stamp_seed_into_run_name(cfg)`.
  *Measured:* the three canonical names come out exactly
  `['phaseB_MoE_seed42', 'phaseB_MoR_seed42', 'phaseB_MoRE_seed42']` and match
  `canonical_spec.json`'s `run_name_template` when formatted from it — the
  template and the code cannot drift apart silently. An unknown architecture
  **raises** instead of producing a plausible-looking name; an undeclared seed
  yields `phaseB_MoRE` rather than a name that claims a seed it does not have.
  Stamping is applied in `cli.py` *after* `resolve_overrides` (only then are
  both the seed and the run name final), is idempotent (no
  `phaseB_MoRE_seed43_seed43`), and *extends* a custom `--run_name` rather than
  overwriting it (`t63_pim_check` → `t63_pim_check_seed42`), so a diagnostic run
  is still identifiable.
  *Label hygiene:* T6.3's suite asserts no `/` appears in any expert label
  (`E1_ADD_SUB`, not `E1 ADD/SUB`), and the real run emitted 24 `val/routing_*`
  keys with no spurious nesting.

- [x] **T6.7 Full provenance** (§8.7 / `updated_rules.md` §9). Extend the
  Phase-0 subset to the complete field list, including `architecture`,
  `variant`, `train_split_version`, FFN multiplier, halting weight, routing
  supervision weight, router noise mode, halt target mode.
  **Verify:** a smoke run's W&B config contains every listed field, none `None`.
  **DONE — `test_phase6_provenance.py` 32/32 PASS.** Verified against a **real
  CLI run**, not a constructed dict: `python train.py --architecture more
  --epochs 1 --seed 44 --run_name t67_provenance_check` → exit 0 →
  `runs/t67_provenance_check_seed44__c3571d3a/resolved_config.json`, then every
  one of the 27 `updated_rules.md` §9 fields resolved from the block a reader
  would actually open, with the failure printed as a `missing/None` list.
  Result: **`missing/None: []`**. The run is end-to-end because the fields are
  assembled across three files (`cli.py` → `run_context.py` → `engine.py`) and a
  unit test of any one of them would not catch a field dropped between them.
  Three things had to be built to make the list satisfiable:
  1. **`variant` is derived, never declared** (`more/config.py:resolve_variant`).
     An unmodified MoRE config reports `canonical`; a dense-routing, router-noise
     or fixed-depth run cannot be reported as canonical *by omission*; two
     deviations compose into one stable sorted label
     (`fixed_depth+router_noise_gaussian`). A curriculum flag with **zero**
     weight is correctly *not* a variant — the term contributes nothing.
  2. **`ffn_mult` is a real config field** threaded through
     `MoREModel → MoREWrapper → MoEBlock` instead of a hard-coded `* 4`, so the
     FFN width is recorded rather than assumed (and Phase 9's parameter matching
     has a lever). Default 4 reproduces every earlier run exactly:
     `total_params == 6,358,553`, unchanged.
  3. **`router_noise_scale` is resolved, not inherited** — under the canonical
     `router_noise = "none"` it is written `0.0`, not the init value it would
     have had if noise were on.
  *Also verified:* declared seed 44 == `resolved_seed` 44 (the seed reached the
  RNGs, not just the log), `halt_target_mode` names the halting objective, no
  provenance value is the *string* `"None"`, and the T6.3 block is present in the
  same run's `metrics.json`.
  *Two latent defects fixed here, both of which would have surfaced only in the
  final matrix:* `config_hash` included the `logging` block, so two seeds of one
  experiment hashed differently and the exporter — which refuses to mix hashes —
  would have refused to average them (`run_context.py:config_hash` now strips
  `logging` as well as `provenance`); and `_allocate_id` appended the seed a
  second time once T6.6 put it in the run name
  (`..._seed44__seed44__c3571d3a`), now suppressed when the base already carries
  it.

### GATE 5 — [x] valid provenance · [x] valid metrics · [x] correct dimensions · [x] routing assertion · [x] halt-gradient assertion · [x] no sentinel metrics

**CLOSED — `code/test_gate5.py` 34/34 PASS.** The gate is re-asserted in one
file against a live model and a real run directory, so "Gate 5 passes" is a
statement about the code as it stands now rather than about six test files that
passed at different times. Run: `python code/test_gate5.py` (after
`test_phase6_provenance.py`, which produces the run directory it reads).

1. **valid provenance** — 21 required fields present and non-None; declared seed
   == resolved seed; determinism reported as *two* distinct facts (what could not
   be configured vs. `nondeterministic_ops_observed`, what actually ran
   non-deterministically); `variant` non-empty.
2. **valid metrics** — all 10 loss components reported separately;
   `val/task_loss` present and distinct from `train/total_loss`; entropy
   normalized to `[0,1]` by `log(E)`; cosine similarity mean **and** max; depth
   as `allocation_error_abs`/`_rel` with **no key containing "efficiency"**;
   early **and** forced exit rate; the full permutation-invariant key set; and
   `pim.raw_accuracy == val/routing_accuracy` to < 1e-6.
3. **correct dimensions** — `num_experts == 6` from the family manifest, and
   `cls_head`, `step_cls_head`, `router`, the expert list and the halt-head list
   all measured equal to it (no hard-coded 7). The op embedding is
   `NUM_OP_TYPES = 16 ≠ 6` — sized by *operation* count, since the op→expert map
   is precisely what routing must discover.
4. **routing assertion** — `routing_mode == "top1_sparse"` in both the model and
   the run's metrics; measured `dispatch/evals_per_token == 1.0` (dispatch, not
   dense blending); router noise `none` / scale `0.0`; every one of 64×7 token
   indices in `[0, 6)`.
5. **halt-gradient assertion** — a real gradient reaches the halt parameters,
   asserted per expert. *Scoped deliberately:* under Top-1 dispatch an expert no
   token selected is never called, so its halt head having `grad is None` is
   arithmetic, not a bug — reporting it as a failure would train the reader to
   ignore the check. So the assertion runs over the experts that actually
   received tokens (from the forward's returned first-route indices) **and**
   separately asserts coverage (`len(used) == 6`) so it cannot go silently weak.
   Then the sharper form: `ponder_cost.backward()` **alone** reaches every active
   halt head, so the recursion cost is a term the halting policy optimizes
   against, not a number printed next to the model.
6. **no sentinel metrics** — no metric equals `-1`, none is NaN, undefined
   entries are the string `"N/A"`; and the strongest case, a single-expert (MoR)
   confusion, returns `None` for *every* routing quantity rather than a
   comparable-looking `0.0`.

*Full regression at closure:* Gate 1 8/8 · Phase 2 19/19 · Gate 2 31/31 ·
Phase 4 27/27 · Phase 5 62/62 · T6.1 61/61 · T6.3 42/42 · T6.6/T6.7 32/32 ·
Gate 5 34/34 · `verify_pipeline.py` 35/35 · `smoke_test.py` PASS.

---

## Phase 7 — One Training Pipeline  (`plan.md` §9)

- [x] **T7.1 `architecture = moe | mor | more`** as an explicit mode in the
  single `train.py`. Do **not** create `run_moe.py` / `run_mor.py`.
  **Verify:** all three architectures instantiate from one entry point with one
  tokenizer; `code/` contains no per-architecture training script.
  **Evidence (done early, ahead of Phase 1, to make the remaining phases
  cheap to edit):** the 1615-line monolith is split into `code/more/`
  (`config` 139, `families` 44, `data` 183, `model` 465, `metrics` 303,
  `engine` 551, `run_context` 480, `cli` 83). `train.py` is a 19-line entry
  point taking `--architecture`; `train_moe.py` / `train_mor.py` /
  `train_more.py` are 16-line launchers that only set the architecture and call
  `more.cli.main` — they contain **no** model, data or training code, so there
  is still exactly one pipeline, one tokenizer and one evaluation path.
  `apply_architecture()` (`more/config.py`) derives only the definitionally
  different fields. Observed resolved configs:
  `moe E=6 depth=1 halt=False`, `mor E=1 depth=7 halt=True`,
  `more E=6 depth=7 halt=True`; all three exited 0 on a 1-epoch dummy run.
  `smoke_test.py` passes; `verify_pipeline.py` **35/35**; the Gate 0 guard still
  exits **2** on a canonical claim with no run directory created.
  Three defects fixed at source during the split: (a) `load_config` injected a
  `train_path` default that overrode an explicit `data.jsonl_path`, so the
  engine trained on a file the config never named (same class as T5.3);
  (b) `metrics.py` was missing `import wandb` — package now pyflakes-clean,
  0 undefined names; (c) `verify_pipeline.py` crashed in its own banner on the
  cp1252 Windows console.

- [x] **T7.2 Correctness suite** (Step 10). Collect the Gate 1–5 assertions
  into one runnable suite.
  **Verify:** suite exits 0 and prints a per-gate pass table.
  **DONE — `code/run_correctness_suite.py`.** One command,
  `python run_correctness_suite.py`, runs the eight existing suites as eight
  child processes in a fixed order, counts their `[PASS]`/`[FAIL]`/`[SKIP]`
  markers, and prints the table below. `--gate N` runs one gate; `-v` streams
  child output; exit code is 0 only if every gate passes.

  | gate | checks | pass | fail | verdict | what it closes |
  |---|---|---|---|---|---|
  | 1 | 19 | 19 | 0 | PASS | Top-1 sparse dispatch, not dense blending |
  | 2 | 31 | 31 | 0 | PASS | differentiable ACT, real halt gradient |
  | 3 | 27 | 27 | 0 | PASS | balance loss depth/block-normalized |
  | 4 | 130 | 130 | 0 | PASS | widths derive from `num_experts`/`num_families`; global seeding |
  | 5 | 139 | 139 | 0 | PASS | provenance, permutation-invariant routing metrics, no sentinels |
  | **TOTAL** | **346** | **346** | **0** | **exit 0** | |

  Three design points, each forced by a defect hit while building it:
  *subprocess per suite* — the suites seed global RNGs and mutate
  `torch.backends` flags, so importing them into one process would let one
  suite's determinism settings decide another's verdict, the exact
  cross-contamination Gate 4 exists to rule out; *fixed order, not
  alphabetical* — `test_gate5.py` reads a real run directory that
  `test_phase6_provenance.py` produces, so Gate 5 pins provenance ahead of it;
  *zero counted checks is a FAILURE, not a pass* — the first version anchored
  its marker regex at column 0, the Phase 2–5 suites indent some of theirs, and
  four of the five gates printed "0 checks … PASS". An absence of evidence
  rendered as evidence is the same defect as a sentinel reported as a
  measurement (CLAUDE.md §4), so the runner now requires `pass > 0`, `fail ==
  0` **and** `rc == 0` before it will call a gate passed, and prints `EMPTY`
  where no marker was recognised.

---

## Phase 8 — Freeze the Canonical Configuration  (`plan.md` §10) — Step 11, 12

- [x] **T8.0a The controlled ACT decision experiment** — a prerequisite for
  T8.2, since `canonical_spec.json` cannot record a halting mode that has not
  been chosen on evidence.
  **Verify:** two arms differing in exactly one field, ≥3 seeds each, mean ± std,
  and a decision rule fixed before the runs.
  **DONE — `automated/act_decision.py`, 6 runs.** Matched protocol
  (`architecture=more`, 20 epochs, `blocks=1`, `batch_size=768`, seeds 42/43/44,
  identical lr / d_model / weight decay / `routing_balance` / `step_routing`);
  the only difference is `--halting_supervision --halting_supervision_weight 0.1`.
  The decision rule is written at the top of the driver *before* the runs —
  primary criterion is validation task loss, allocation error is reported but
  explicitly **not** a selection criterion (the supervised arm is given
  `OP_TARGET_DEPTH` and wins it by construction), and `pure_act` wins ties.

  | mean ± std, 3 seeds | val task loss | avg depth | alloc err abs | op depth spread | corr. w/ `OP_TARGET_DEPTH` |
  |---|---|---|---|---|---|
  | `pure_act` | **0.063676 ± 0.000283** | 2.292 | 1.025 ± 0.059 | 0.452 ± 0.044 | **+0.204 ± 0.554** |
  | `supervised_curriculum` | 0.063953 ± 0.000190 | 2.057 | **0.445 ± 0.012** | 0.983 ± 0.084 | +0.957 ± 0.027 |

  Gap `−0.000277` against a largest seed std of `0.000283` — inside noise, so the
  tie rule decides: **canonical halting mode = `pure_act`**, recorded in
  `automated/act_decision_result.json`. The driver refuses to report at all if a
  run's `provenance.halting_mode` disagrees with the arm it was launched as, so a
  silently ignored flag cannot masquerade as "supervision makes no difference".
  **Outcome C for the adaptive-computation claim.** Under `pure_act` the
  per-operation depths do differ (spread 0.452), but their correlation with the
  curriculum is `+0.608 / −0.428 / +0.431` across seeds — **the sign flips**.
  Unsupervised ACT learns a non-trivial but *arbitrary* depth partition, not the
  operation-complexity curriculum. Reported, not fixed by enabling supervision.
  Also recorded: the pre-registered rule tested depth *spread*, which passed
  (0.452 ≫ 0.10) on a model whose allocation is arbitrary — spread is necessary
  but not sufficient, so the correlation diagnostic was added **after** the runs
  and is labelled as such in-source. It does not change the decision.

- [x] **T8.0b The routing-supervision decision** — also a prerequisite for T8.2.
  `plan.md` §E states the rule already: *supervision OFF = canonical, supervision
  ON = explicitly labelled oracle-routing ablation.* The current `config.json`
  has `loss_weights.step_routing = 0.5`, and `engine.py:485` makes that term
  `0.01 * oracle_routing_ce + 0.3 * step_cls_loss` — i.e. the router **is** being
  cross-entropy-trained against the oracle expert index. That is why T6.3's run
  reports `routing_accuracy = hungarian = AMI = 1.0`: the index is supervised, the
  exact case T6.3's interpretation note calls "expected, not a finding".
  So the canonical config as it stands would be an oracle-routing ablation by
  `plan.md`'s own definition, and every specialization number in it is supervised.
  **Verify:** matched arms at `step_routing` on vs off, ≥3 seeds, mean ± std, and
  the resulting routing metrics read through T6.3's raw-vs-matched contract.
  Doubles as the measurement for **T10.E**.
  **DONE — `automated/routing_supervision_decision.py`, 6 runs,
  `automated/routing_supervision_result.json`.** Matched protocol
  (`architecture=more`, 20 epochs, `blocks=1`, `batch_size=768`, seeds 42/43/44,
  pure ACT, identical everything else); the only difference is
  `loss_weights.step_routing` 0.0 vs 0.5.

  | mean ± std, 3 seeds | val task loss | raw acc | Hungarian acc | AMI | purity | macro recall | load entropy | avg depth |
  |---|---|---|---|---|---|---|---|---|
  | `step_routing = 0.0` (router unsupervised) | **0.063521 ± 0.000428** | 0.0800 ± 0.0593 | 0.5147 ± 0.0047 | 0.4940 ± 0.0395 | 0.5881 | 0.0991 ± 0.0766 | 0.9540 | 2.128 |
  | `step_routing = 0.5` (oracle-supervised) | 0.063676 ± 0.000283 | 0.9998 ± 0.0003 | 0.9998 | 0.9992 | 0.9998 | 0.9997 | 0.9263 | 2.292 |

  **Decision: canonical `step_routing = 0.0`.** Two independent reasons, and the
  measurement is the weaker of them.

  *The measurement.* Oracle supervision **costs** 0.000155 val task loss against
  a 0.000428 seed std — inside noise, with the unsupervised arm marginally
  ahead. It buys nothing on the primary metric while making AMI 0.9992 true by
  construction: raw accuracy 0.0800 → 0.9998 is the router reproducing a label
  it was trained on. In the unsupervised arm the Hungarian assignment differs
  every seed (`[4,0,5,1,3,2]`, `[5,3,4,1,0,2]`, `[0,1,4,5,3,2]`) while
  Hungarian accuracy stays at 0.5147 ± 0.0047 — a **real partition recovered
  under relabelling**, which is the finding. In the supervised arm the identity
  permutation is recovered every time, which is not.

  *The definitional point, which is what actually settles it.* The instruction
  for this task was to accept routing supervision **if it is
  self-supervised**. For `step_routing` that antecedent is false, and this was
  checked in source rather than assumed: `engine.py` assembles the term as
  `0.01 * oracle_routing_ce + 0.3 * step_cls_loss`, and `oracle_routing_ce` is
  cross-entropy **directly on the router logits** against `step_experts` — the
  oracle expert index that `data/script.py` writes into every record as a
  `"family"` string and `more/data.py` reads back through the hand-written
  `OP_TO_EXPERT` manifest. It is external annotation, not a target the model
  generates. The proof is that the label histogram is byte-identical at
  `num_experts = 6` and `num_experts = 1`: it does not depend on the model at
  all. The configuration described as *"driven by `cls_loss` and ACT"* is
  precisely `step_routing = 0.0` — the router receives no direct label, the
  shared representation is shaped by the family CE at 0.5, and depth comes from
  pure ACT with no external depth target. Applying the stated rule to the
  verified facts therefore selects `step_routing = 0.0`, which coincides with
  `plan.md` §E and with the measurement.
  `step_routing = 0.5` is retained as the labelled **T10.E oracle-routing
  ablation** (variant tag `oracle_routing`); it may not share a headline table
  with canonical runs.

  **Honesty note carried into the paper.** `step_routing = 0.0` removes the
  *direct* oracle CE on the router, but it does **not** make the system
  unsupervised: `loss_weights.family_cls = 0.5` still trains `cls_head` on the
  same oracle family label from the same manifest, and it is that pressure the
  router's partition rides on. The routing claim must be phrased as
  *specialization emerges without direct routing supervision, under
  whole-program family supervision* — never as unsupervised expert discovery.
  See T10.H, which measures exactly how much of the partition that term is
  responsible for: **~24%**. With `family_cls = 0.0` as well — no oracle-derived
  label anywhere in the objective — Hungarian accuracy is still 0.4300 vs chance
  0.167 (AMI 0.3568), i.e. 76% of the above-chance partition is retained. So the
  phrasing above stays mandatory for canonical runs, but the stronger claim
  *"the partition does not require any oracle label"* is now measured and
  available as a labelled ablation result.

- [x] **T8.1 Smoke calibration.** 1–2 epochs on each of MoE / MoR / MoRE;
  inspect every metric by hand.
  **Verify:** no NaN, no sentinel, entropy and depth in plausible ranges,
  task loss below the trivial baselines.
  **DONE — runs `t81c_{moe,mor,more}_seed42`, 2 epochs, seed 42, full dataset,
  `experiment_group = t81_smoke`.** Every metric read by hand out of
  `metrics.json`, not off the console.

  | | MoE | MoR | MoRE |
  |---|---|---|---|
  | params | 3,201,555 | 3,197,710 | 3,201,555 |
  | `ffn_mult` | 4 | 24 | 4 |
  | `val/task_loss` | 0.071424 | 0.071318 | 0.072134 |
  | `train/classification_loss` | 0.001648 | 0.000453 | 0.000490 |
  | Hungarian acc | 0.5385 | N/A | 0.4910 |
  | AMI | 0.4578 | N/A | 0.4177 |
  | `entropy_norm` | 0.8881 | N/A | 0.9141 |
  | avg depth | 1.000 | 2.000 | 2.593 |
  | tok/s | 7919 | 8221 | 2411 |

  *No NaN, no sentinel.* All three: `dispatch/evals_per_token = 1.0` (sparse
  Top-1, not dense), `routing_mode = top1_sparse`, `router_noise = none`,
  `overflow_rate = 0.0`, `variant = canonical`, `family_cls_weight = 0.5`,
  `routing_supervision_weight = 0.0`, `halt_target_mode = pure_act`,
  `seed_declared = true`. MoR emits **no** routing quantity as a number and every
  depth metric as one (`abs 0.928`, `rel 0.578`, early-exit 1.0, forced 0.0);
  MoE, at `max_depth 1`, correctly emits every depth metric as `"N/A"` because
  they are constants there, and no routing metric as `"N/A"`. Determinism:
  MoR re-run at `--seed 42` reproduced `val_loss 0.071318` exactly.

  *Task loss below the trivial baselines.* Computed on the real split
  (val n = 5250, target mean 0.030892, std 0.284455, range [−1, 1]):

  | predictor | val MSE |
  |---|---|
  | predict train mean | 0.080914 |
  | predict train median | 0.081853 |
  | predict zero | 0.081853 |
  | **measured, 2 epochs** | **0.0713 – 0.0721** |
  | *measured, 20 epochs (T8.0b)* | *0.0635* |

  All three clear all three baselines, so the criterion passes — but **only by
  ~12% at 2 epochs and ~21% at 20 epochs** (R² ≈ 0.215 against predict-the-mean).
  That is a modest fraction of target variance and must be stated plainly in the
  paper rather than presented as strong regression performance. It is a property
  of the task and the 3.2M-parameter scale, not evidence against any of the three
  architectures — but it does mean the MoE/MoR/MoRE task-loss differences sit in
  a narrow band, which is exactly why seed variance is reported alongside.

  *Four defects this smoke existed to catch, all fixed:* MoR's flat
  `val/routing_*` keys were **absent** from `metrics.json` rather than `"N/A"`
  (engine.py's own nested block writes `"N/A"` precisely so the field list is
  architecture-independent); `test_gate5.py`'s `train/ponder_cost > 0.0` check
  would `TypeError` whenever the newest routing-bearing run was MoE, i.e. the
  gate's verdict depended on which architecture ran last; check G5.2b accepted
  mere absence where it must require the string `"N/A"`; and the console epoch
  line printed MoR `bal=1.0000` / MoE `halt=1.0000` while the artifact wrote
  `"N/A"` for both — `1.0000` beside MoRE's `−0.6128` reads as "MoR is maximally
  imbalanced". Correctness suite after the fixes: **347/347, exit 0.**

- [x] **T8.2 Freeze `code/canonical_spec.json`.** Fill the currently-null
  fields: `dataset_version`, `d_model`, `lr`, `weight_decay`,
  `routing_balance`. Starting point from `plan.md` §10: `batch_size 768`,
  `50 epochs`, full dataset, 1 recursive block, 6 experts (1 for MoR).
  Do not copy old hyperparameters blindly.
  **Verify:** `--experiment_group canonical_phase_b` is *accepted* for a
  compliant run and still refused for any proxy variant.
  **DONE — `spec_version = 1.1-T8.2-frozen`.** Two structural rules were applied
  rather than filling fields by hand: anything `apply_architecture` varies per
  architecture belongs in `architecture_variants` (a null there blocks only that
  architecture); anything shared belongs in `enforced_fields` (a null there
  blocks *all* canonical runs, which is the intended direction of failure).

  Resolved this task: `architecture_variants.mor.ffn_mult` null → **24**, and
  `enforced_fields.family_cls_weight` → **0.5**, recorded in `frozen_by` as
  **INHERITED, not selected** — it was a bare literal in engine.py for the whole
  project history, so no sweep informed it and it must not be written up as a
  tuned coefficient. `code/config.json` was made literal and complete in the same
  commit so a run launched with **no CLI overrides at all** is already
  canonical-compliant; before T8.2 the defaults file said `num_blocks 2` /
  `batch_size 128` against the spec's frozen `1` / `768`, meaning the default run
  was silently a proxy and only an explicitly-overridden run could be canonical.

  Gate 0 guard, measured:

  | run | `canonical_phase_b` | reason given |
  |---|---|---|
  | moe / mor / more, defaults, seed 42 | **ACCEPTED** | — (MoR was refused before, on the null `ffn_mult`) |
  | `--family_cls 0.0` | REFUSED | `family_cls_weight: canonical requires 0.5, run has 0.0`; `variant='no_family_supervision'` |
  | `--ffn_mult 4` (MoR) | REFUSED | `ffn_mult: canonical mor requires 24, run has 4` |
  | `--epochs 2` | REFUSED | epoch count is not the frozen 50 |
  | `--seed 99` | REFUSED | `not in the frozen seed set [42, 43, 44, 45, 46]` |
  | `--experiment_group exploratory` | accepted as non-canonical | guard does not apply |

  **No field in the spec is null.** `enforced_fields` carries 18 pinned values
  and `architecture_variants` 10 per architecture, all populated — so the Gate 0
  guard is fully armed and nothing blocks T9.1 from declaring
  `canonical_phase_b`. Three of them are frozen as **VALIDATED-STABLE, not
  sweep-selected** — `d_model 256`, `lr 0.001`, `weight_decay 0.0001`, each
  justified in `frozen_by` by the 12 completed 20-epoch runs (6 `actdec` + 6
  `routedec`) converging to val task loss ≈ 0.0635 without divergence. No
  comparative width, lr or weight-decay sweep was ever run under the corrected
  Top-1 / ACT / normalized-balance regime. The paper must describe them that way
  and must not present them as tuned. `family_cls_weight` is weaker still —
  INHERITED from a literal. Only `halting_mode` and `step_routing_weight` were
  selected by an actual controlled experiment (T8.0a, T8.0b).

### PHASE 8 — status: **CLOSED** (T8.0a, T8.0b, T8.1, T8.2 all ticked)

The canonical configuration is frozen at `spec_version = 1.1-T8.2-frozen` with no
null fields, the Gate 0 guard accepts `canonical_phase_b` for all three
architectures at defaults and refuses every proxy variant tested, the parameter
budget is matched to 0.120%, and the correctness suite passes 347/347. **No
further architecture changes.** What each frozen value rests on differs sharply
and the paper must not flatten that: two values (`halting_mode`,
`step_routing_weight`) come from controlled 3-seed experiments; three
(`d_model`, `lr`, `weight_decay`) are validated-stable but never swept; one
(`family_cls_weight`) is inherited from a code literal; the rest come from
`plan.md` or `updated_rules.md` by rule.


---

## Phase 9 — Primary Scientific Matrix  (`plan.md` §11) — Step 13, 14

- [x] **T9.1 Primary comparison** (§11.1): MoE / MoR / MoRE × seeds
  42–46 = 15 canonical runs.
  **Verify:** 15 run directories with `experiment_group = canonical_phase_b`,
  identical `config_hash` apart from architecture and seed.
  **DONE — 15/15 admitted, 236 min (3.9 h), report exit 0, no problems.**
  Driver `code/run_phase9_matrix.py`; aggregate `code/phase9_matrix_result.json`;
  re-derive without retraining via `python run_phase9_matrix.py --report-only`.
  Provenance: `variant = canonical`, `seed_declared = True` on all 15, and **168
  invariant provenance comparisons with 0 mismatches** over the 12 protocol fields.
  Seed-blind config identity: `moe / mor / more` all `identical` (see T9.1b below
  on why raw `config_hash` is *expected* to differ per seed).

  | arch | `val/task_loss` | `best_val_loss` | R² vs floor 0.080914 | params |
  |---|---|---|---|---|
  | MoE | 0.063260 ± 0.000137 | 0.062819 ± 0.000173 | 0.2182 | 3,201,555 |
  | MoR | **0.062706 ± 0.000100** | 0.062527 ± 0.000047 | 0.2250 | 3,197,710 |
  | MoRE | 0.063186 ± 0.000223 | 0.062892 ± 0.000193 | 0.2191 | 3,201,555 |

  Rule 4 (a gap under 2× the larger seed std is not a difference):
  MoRE − MoE = −0.000074 (0.33× → **inside seed noise**);
  MoRE − MoR = +0.000481 (2.16× → resolved);
  MoE − MoR = +0.000554 (4.04× → resolved).

  **Outcome C on the primary metric.** MoR — one shared block, no routing — has the
  lowest val task loss and its margin over both MoE and MoRE clears the seed noise;
  MoRE is indistinguishable from MoE. Caveats both ways: the gap is ~0.8% of the
  loss and all three beat the predict-the-mean floor by only ~22%, so this is a
  ranking among weak models; and MoR has 3,845 *fewer* parameters, so the 0.120%
  budget residual favours the arms that lost.

  Routing (no oracle label in the objective, `routing_supervision_weight = 0.0`):
  Hungarian 0.5004 ± 0.0501 (MoE) / 0.5258 ± 0.0547 (MoRE) against chance 0.167,
  AMI 0.5132 / 0.4907, purity 0.5715 / 0.6047, matched macro recall 0.5133 / 0.5491;
  raw accuracy 0.2016 ± 0.1411 / 0.1629 ± 0.1668 — the raw ≪ Hungarian signature of
  a real partition that is arbitrarily numbered, the large raw std being the
  permutation varying by seed. Depth: avg steps 2.0169 ± 0.0191 (MoR) / 2.0642 ±
  0.0520 (MoRE), allocation error 0.9509 / 1.0055, early exit 0.9998, forced exit
  0.0002 — halting is alive but depth does **not** track the complexity curriculum.
  Sparsity: `dispatch/evals_per_token = 1.0000 ± 0.0000` for all three. Throughput
  8485 ± 324 / 4574 ± 637 / 2225 ± 246 tok/s on one RTX 4060 Laptop — an engineering
  observation, not an efficiency claim. Full reading in `changelog.md` (T9.1c).

  *Launched with* `--architecture`, `--seed`, `--run_name`, `--experiment_group`
  and nothing else; every other field inherited from `config.json` so the Gate 0
  guard adjudicates it.

  **T9.1b — the matrix's own verification rule was the only thing it flagged.**
  The first aggregation exited 1 with three problems, one per architecture:
  `N distinct config_hash values across seeds -- seeds must not change the config`.
  The runs were fine; the predicate was wrong. `config_hash()`
  ([run_context.py:82](code/more/run_context.py:82)) hashes the config minus
  `provenance` and `logging`, and `--seed` is written into **`data.subset_seed`**
  and **`training.subset_seed`**, which are inside it — the 8-char hash *is* the
  run-directory suffix, which is why `seed42__c35fe8b6` and `seed43__33afa654`
  never collide. Field-by-field diff of the five hashed configs per architecture:
  those two fields differ and nothing else, and `resolved_subset_fraction = 1.0`
  on all 15, so `subset_seed` provably could not have changed the data.
  A test had been asserting the opposite and passing: T6.7b in
  `test_phase6_provenance.py` claimed *"two seeds of one experiment share a
  config_hash"* while varying only `logging.run_name` — a check that cannot fail.
  **`config_hash()` was deliberately not changed** (`subset_seed` really does
  select a different subset when `subset_fraction < 1.0`, and changing the hash
  would break comparability with these 15 runs). Instead `seed_blind_diff()` in
  the driver compares the hashed config *minus* `SEED_KEYS` and names the
  differing **field path** rather than comparing opaque hashes; T6.7b was rewritten
  into four honest assertions including one that varies the real seed. Suite
  re-run: **350/350, all five gates PASS** (up from 347). Matrix re-aggregated
  `--report-only`: **exit 0, no problems**, no run re-executed.

  *Ready to launch:* the Gate 0 guard now **accepts** `canonical_phase_b` for all
  three architectures at defaults (T8.2), so all 15 runs are launchable with no
  CLI overrides beyond `--architecture`, `--seed`, `--run_name` and
  `--experiment_group`. Beyond the 15, Phase 10 now
  owns 9 labelled ablation arms (T10.A–T10.I), which are separate tables and
  never merged into the headline.

  **T9.1a — the undeclared-group defect (first launch attempt discarded).** The
  first attempt ran ~1.6 h and produced 5 complete 50-epoch runs + 1 killed, all
  stamped `experiment_group = exploratory` and therefore **inadmissible**, because
  the driver omitted `--experiment_group canonical_phase_b`. That flag is not a
  config override, it is the canonical *claim*; with
  `logging.experiment_group = None` the Gate 0 guard returns
  `NONCANONICAL_GROUP_DEFAULT` with **no error and no refusal** by design
  ([run_context.py:256](code/more/run_context.py:256)). Every enforced field
  matched — `variant: canonical`, `seed_declared: True`, `resolved_epochs: 50` —
  so this is a failure Gate 0 cannot catch by construction. The guard itself was
  verified sound in-process (returns `canonical_phase_b` for moe/mor/more at
  defaults once `logging.experiment_group` and `provenance.seed` are both set).
  The 6 directories are archived at `archive/t9_1_undeclared_group_runs/` with an
  `INVALIDATED.md`, **not re-stamped** — a group is declared at launch and
  validated by the guard, never edited onto a finished run. Fix: `launch()` passes
  the flag, and `find_dir(..., canonical_only=True)` gates the resume path so an
  `exploratory` directory can never satisfy the skip check (otherwise the failure
  becomes permanent). **Cheapest future check: read the `[Run]
  experiment_group=` line in the first run's banner, ~60 s after launch.**

  **T9.0 — protocol fixed before launch** (recorded in
  `canonical_spec.json:protocol`, a new non-enforced block so the rules live in
  one file instead of only in a changelog entry):
  - `val_interval = 2` (`logging.log_interval`, raised from 10). It gates
    validation at [engine.py:629](code/more/engine.py:629), so at 10 a 20-epoch
    run measured val **twice** — and "best epoch by val loss" over two candidates
    on a still-falling curve always returns the last epoch, making checkpoint
    selection indistinguishable from "take the final model". Measured on the six
    T10.H runs: val fell −0.002010 ± 0.000443 between epochs 10 and 20, same sign
    in all six, ~4.7× the seed std — still descending at the end of training, so
    the coarse cadence was hiding the curve, not summarising it.
  - **Checkpoint selection:** lowest `val/task_loss` among evaluated epochs, no
    early stopping, full frozen epoch count every run, argmin taken post hoc.
  - Changing `val_interval` changes `config_hash`, so it was set **before** the
    matrix launched. A mid-matrix change would produce two hash groups that may
    not share a table.
  - `results.tsv` now writes `"N/A"` for an unevaluated epoch instead of `nan`,
    matching `metrics.json`. Nothing was ever reported as a measurement (the
    `isnan` guard kept it out of `best_val_loss`), but two conventions for "no
    number here" in one run directory is what a plotting script misreads.

- [x] **T9.2 Parameter-matched MoR** (§11.2): `|P_MoR − P_MoRE| / P_MoRE < 0.05`.
  **Verify:** recorded parameter counts satisfy the bound.
  **RESOLVED at T8.2 — gap 0.120%, inside the 5% bound.** Measured on the
  current code, from `provenance.total_params` of real builds:

  | architecture | `ffn_mult` | parameters |
  |---|---|---|
  | MoE | 4 | 3,201,555 |
  | MoR | **24** | 3,197,710 |
  | MoRE | 4 | 3,201,555 |

  |3,197,710 − 3,201,555| / 3,201,555 = **0.00120**.

  The policy is to equalise **total FFN hidden width**, not the per-FFN
  multiplier: MoR has one FFN where MoRE has six, so `ffn_mult = 24 = 4 × 6` is
  principled rather than fitted. Measured candidates: 4 → 571,150 (82.2% short),
  23 → 3,066,382 (4.22%), **24 → 3,197,710 (0.120%)**, 25 → 3,329,038 (3.98%).
  The residual 0.120% is the router plus the six per-expert halt heads MoR does
  not have, and is irreducible by any integer multiplier — that is the
  "unavoidable difference" the comparison protocol has to state, and it favours
  MoRE by 3,845 parameters (0.12%), which is far too small to explain any
  task-loss difference. `ffn_mult = 4` is retained as the labelled T10.I
  under-budgeted arm.
  *Superseded, on the old code:* MoE 3,724,055 = MoRE 3,724,055, MoR 567,569 →
  **84.8%** mismatch.

  **CONFIRMED on the 15 canonical runs (T9.1):** `provenance.total_params`
  aggregated over five seeds is constant within each architecture — MoE
  3,201,555, MoR 3,197,710, MoRE 3,201,555 — so the bound is satisfied by the
  runs that actually produced the headline table, not only by a build-time check.
  Note the direction: the residual favours **MoR by 3,845 parameters**, and MoR is
  the arm with the lowest val loss, so the budget difference cannot be invoked to
  explain away its margin.


- [x] **T9.3 Report mean ± std** for best/final val loss, avg depth, normalized
  entropy, routing accuracy, depth allocation error, throughput.
  **DONE — every quantity in this list is emitted as mean ± std over seeds 42–46
  by `run_phase9_matrix.py`**, with architecture-inapplicable entries printed as
  `N/A` rather than 0.0: final val loss `val/task_loss`, best val loss
  `best_val_loss`, avg depth `train/avg_recursion_steps` (N/A for MoE —
  `max_depth = 1` makes it a restatement, not a measurement, so the driver
  suppresses the float 1.0 it would otherwise print), normalized entropy both
  `train/entropy_term_normalized` and `train/expert_load_entropy_normalized`
  (N/A for MoR — E = 1), routing accuracy raw **and** Hungarian-matched plus AMI,
  purity and matched macro recall (N/A for MoR), depth allocation error
  `depth/allocation_error_abs` (N/A for MoE), throughput
  `perf/throughput_tokens_sec`. Numbers in T9.1 above; machine-readable in
  `code/phase9_matrix_result.json`; interpretation guide in `results_exp.md`.
  Nothing was hand-copied — the table is generated from the run directories.

---

## Phase 10 — Ablations  (`plan.md` §12) — Step 15

- [ ] **T10.A** Oracle routing vs learned routing (diagnostic upper bound only —
  never in the headline). **Still open — requires new code** (a hard dispatch
  path that bypasses the router with the manifest label). T10.E is the *soft*
  version of this question and already answers it in the negative, which lowers
  T10.A's value considerably: a router driven to 100% family accuracy produced
  *worse* task loss, so the oracle upper bound is unlikely to be an upper bound.

**T10.B–T10.F were executed as one batch** by `automated/phase10_ablations.py`,
5 arms × seeds 42/43/44 × 50 epochs, ~6.1 h GPU. The canonical MoRE arm from
T9.1 is **reused, not retrained**, restricted to the same three seeds
(`val/task_loss` 0.063290 ± 0.000050; the 5-seed std 0.000223 is admitted into
the noise yardstick because the 3-seed std is ~4.5× under-sampled). Reading rule
fixed before any run: a gap counts as resolved only at `≥ 2 ×` the largest
available seed std. All arms are `experiment_group = exploratory`; none may enter
the headline table. Aggregate: `automated/phase10_ablations_result.json`.

- [x] **T10.B** Adaptive depth vs fixed depth. `model.fixed_depth False → True`.
  **Not significant: −0.000364, Cohen d = −1.90, exact p = 0.125** (0.062822 vs
  baseline 0.063186 ± 0.000249). Both earlier readings — the original "RESOLVED
  2.10× BETTER" (a population-std artifact, **T11.0a**) and the corrected
  "1.88×, inside noise" — reach the same conclusion as the exact test, so this
  arm's verdict is stable; see **T11.0b** for why the intermediate reasoning was
  still wrong. Defensible claim: canonical MoRE matches a fixed 7-step schedule on
  task loss while spending 2.05 steps, a **3.4× recursion-compute saving at no
  measurable quality cost**. What it does *not* show is that the halting *policy*
  is good — `depth/allocation_error_abs ≈ 1.0` says it does not track the
  complexity curriculum, T11.0b found **96.5% of tokens exit at exactly step 2**
  with only a 0.11-step spread across sixteen operations, and T10.J shows loss is
  flat over 2.0–7.0 steps. A learned constant of 2 is not excluded — it is what
  the depth distribution actually shows. `depth/allocation_error_abs = 5.8850` in
  this arm is uninformative by construction.
- [x] **T10.C** One vs two recursive blocks — **only `num_blocks` varies**
  (1 → 2), which the old three-factor headline never achieved.
  **Inside seed noise, 0.08× — the flattest arm in the phase**: 0.063258 ±
  0.000423, at **1.99× the parameters** (6,358,553 vs 3,201,555) and 0.77× the
  throughput. The model is not capacity-limited; this also retires the
  uninterpretable pre-T6.8 1-vs-2-block comparison.
- [✗] **T10.D** Router noise: none vs fixed-annealed vs trainable.
  **SCOPE NARROWED AT T11.0c — the `trainable` arm will not be run.** What was
  measured stands: `none → fixed_annealed` is **not significant, exact p = 0.4107**
  (0.063363 ± 0.000399 vs 0.063186 ± 0.000249, n=3 vs n=5; the old "0.18× seed
  std" phrasing came from the withdrawn `2 × std` rule — see **T11.0b**), so the
  canonical `router_noise = none`, chosen on principle in T4, is also empirically
  free. The task is closed at two arms rather than three because the third cannot
  answer the question it is named for: CLAUDE.md §2 records the
  L2-on-trainable-noise-scale mechanism as **unproven**, so a `trainable` arm
  varies the noise schedule *and* introduces that machinery in the same run. Its
  result would be uninterpretable in either direction — a win could not be
  attributed to trainable noise, and a loss could not be attributed to it either.
  A one-field ablation whose one field is two fields is not a one-field ablation.
  The written scope of T10.D is therefore **"none vs fixed-annealed"**, and the
  paper states that the trainable variant was excluded by design with this reason,
  not that it was tried. If it is ever wanted, it needs its own task and an
  explicit two-field label.
- [x] **T10.E** Routing supervision off vs on. `loss_weights.step_routing
  0.0 → 0.5`. **Routing supervision HURTS: +0.000585, Cohen d = 2.77, exact
  p = 0.0179** (0.063772 vs baseline 0.063186 ± 0.000249, n=3 vs n=5). The T11.0a
  downgrade to "buys nothing" is **withdrawn** — it came from the `2 × std`
  reading rule, which is ~√n too conservative; see **T11.0b**. The original
  "WORSE" verdict was right for the wrong reason. Every routing metric
  hits its ceiling (Hungarian / AMI / purity / matched macro recall / raw accuracy
  all exactly 1.0000 ± 0.0000, from 0.5538 / 0.5092 / 0.6224 / 0.5712 / 0.0790).
  A router driven to 100% family accuracy is **significantly worse** on the
  primary metric than one 8% correct, so family identity is not merely useless to
  the router — optimizing for it actively trades against the task. Also
  the cleanest available validation of the T6.3 Hungarian layer: raw and matched
  accuracy coincide exactly when the permutation is identity. Load becomes less
  uniform (`load_entropy` 0.9860 → 0.9518) because the family distribution is not
  uniform; nothing collapses.
  **Caveat RESOLVED — seeds 45–46 landed (T11.1 window).** At the full 5-vs-5 the
  arm reads **+0.000565, Cohen d = 2.89, exact p = 0.0079** against a floor of
  0.0040, so it **survives Bonferroni** across the six Phase 10 arms
  (α = 0.05/6 = 0.0083) with 0.0004 to spare. The direction, the effect size and the
  routing-ceiling story are unchanged from n=3; only the resolution improved. The
  thin margin is printed in the driver's verdict string rather than left for a
  reader to infer, and the 5-seed mean is 0.063751 ± 0.000121.
- [x] **T10.F** Dense vs Top-1 sparse routing. `model.routing_mode
  top1_sparse → dense_blend`. **Dense is significantly WORSE: +0.000500,
  Cohen d = 1.43, exact p = 0.0179** — the earlier "inside seed noise, 1.14×"
  reading was the `2 × std` artifact (see **T11.0b**). So the canonical Top-1
  choice is now supported on **quality** as well as on cost: dense runs **6.0
  expert evaluations per token** instead of 1.0, and is **1.40×
  faster** in wall clock (2912 ± 145 vs 2073 ± 170 tok/s) — the loop is
  launch-overhead-bound, so one batched six-expert matmul beats a one-expert
  gather/scatter. Routing quality degrades too (Hungarian 0.5538 → 0.4963). The arm
  cannot be mislabelled as MoE: `enforce_routing_mode` refuses any `dense_blend`
  run whose label lacks `dense_routing_ablation`.
  **Caveat RESOLVED — seeds 45–46 landed (T11.1 window).** At the full 5-vs-5:
  **+0.000529, Cohen d = 1.90, exact p = 0.0079**, floor 0.0040 — **survives
  Bonferroni** at α = 0.0083. 5-seed mean 0.063716 ± 0.000305. Both of the two
  significant Phase 10 arms are now Bonferroni-robust, which was the whole purpose
  of running the extra seeds.
  **Verify (all): PASS.** The driver asserts the one-field difference twice —
  once on the generated arm config and once on the `resolved_config.json` that
  actually trained — and pins a code fingerprint across all six sources so no arm
  straddles a code change. Final verdict: *"No problems. Every completed arm
  differs from the canonical MoRE baseline in exactly one config leaf, verified
  on the config that actually trained, and all arms share one code state."*

- [x] **T10.J** Ponder-cost sensitivity — `loss_weights.halting` 0.001 → 0.0001,
  3 seeds × 50 epochs, added specifically because T10.B could not separate
  *"adaptive computation does not work here"* from *"our ponder cost is
  mis-weighted"*. Reading pre-registered in the arm docstring before any seed ran.
  **Result: the coefficient is exonerated, and the benchmark cannot see depth.**
  Depth responded — `avg_recursion_steps` 2.0524 ± 0.0749 → **2.5120 ± 0.0876**,
  ≈5.2× seed std, which also re-confirms a live gradient into the halt parameters —
  but it released only 0.46 of the 4.95 steps to the fixed-depth budget, and
  `val/task_loss` did not move (0.063630 ± 0.000473, **0.72×, flat**). Read with
  the corrected T10.B, three depth points (2.05 / 2.51 / 7.00) all give the same
  task loss, so **loss on this benchmark is flat in recursion depth across the
  whole usable range**. The halting machinery works; the task cannot discriminate
  allocation policies. That is a finding about the experimental design as much as
  the architecture, and it means MoRE's depth half is **untestable here, not
  refuted**. `depth/allocation_error_abs ≈ 1.0` remains the only allocation-quality
  evidence and says the policy does not track the curriculum.
  `halting = 0.0` considered and rejected: at zero there is no pressure to exit,
  so the arm either drifts to `max_depth` and re-measures T10.B or it does not,
  and the 0.001 → 0.0001 dose-response already bounds the coefficient's influence
  as small against a 4.95-step gap.

- [✗] **T10.G** Supplementary sweeps (§13): batch size, capacity/model size.
  **SCOPE NARROWED AT T11.0c — neither sweep will be run.** The capacity half is
  already answered by **T10.C** (`num_blocks 1 → 2`, i.e. ~2× the recursive
  parameters): **not significant, exact p = 0.7500**. Doubling the model changes
  nothing measurable, so a wider capacity sweep would be spending GPU to re-derive
  a null at more points. The batch-size half is closed on a different ground: batch
  size moves **throughput, not a claim**. It cannot enter the headline table
  (`canonical_spec.json:enforced_fields.batch_size = 768` is frozen, so any other
  value is by definition non-canonical), and this project's binding constraint is a
  single consumer laptop GPU — wall-clock spent characterising throughput is
  wall-clock not spent on arms that answer a scientific question about MoRE. The
  written scope of T10.G is therefore **closed as "capacity answered by T10.C;
  batch size out of scope as a throughput-only lever"**, and the paper says exactly
  that rather than implying a sweep was performed. If a reviewer asks for
  batch-size robustness, it becomes a new task with its own label.


- [x] **T10.H** Family supervision off vs on — `loss_weights.family_cls`
  0.0 vs canonical 0.5, MoRE, `step_routing = 0.0`, 3 seeds × 20 epochs.
  Measures how much of the expert partition the **external** whole-program
  family label is responsible for, and what removing it costs task loss.
  Driver: `automated/family_cls_ablation.py`. Both arms are trained on one code
  state — the T8.0b runs are *not* reused as the baseline arm, because they
  predate T6.8 and carry no `per_family_matched` / `matched_macro_recall`, which
  would put the two arms on different metric layers. The T8.0b seed-42 run is
  kept as a reproduction cross-check only.
  **Verify:** every field in the matched-protocol list identical across arms,
  `family_cls_weight` in provenance agrees with the arm, and the reading follows
  the decision rule fixed in the driver docstring before any run.

  **Result — 6 runs, exit 0, ~61 min.** Both arms MoRE / `pure_act` /
  `step_routing 0.0` / 20 epochs / blocks 1 / batch 768, seeds 42–44. Mean ± std,
  chance = 0.167. Raw output: `automated/family_cls_ablation_result.json`.

  | metric | `family_cls = 0.5` (canonical) | `family_cls = 0.0` (ablation) | floor | retained |
  |---|---|---|---|---|
  | `val/task_loss` | 0.063521 ± 0.000428 | 0.063421 ± 0.000157 | 0.080914 | — |
  | `routing_hungarian_accuracy` | **0.5147 ± 0.0047** | **0.4300 ± 0.0293** | 0.167 | **75.7%** |
  | `routing_ami` | 0.4940 ± 0.0395 | 0.3568 ± 0.0431 | 0.0 | 72.2% |
  | `routing_purity` | 0.5881 ± 0.0135 | 0.5041 ± 0.0539 | 0.167 | 80.1% |
  | `routing_matched_macro_recall` | 0.4819 ± 0.0053 | 0.4286 ± 0.0327 | — | — |
  | `routing_accuracy` (raw, identity) | 0.0800 ± 0.0593 | 0.1379 ± 0.0635 | — | — |
  | `train/classification_loss` | 0.000087 ± 0.000010 | 1.7801 ± 0.1331 | — | — |
  | `expert_load_entropy_normalized` | 0.9540 ± 0.0149 | 0.9796 ± 0.0064 | — | — |
  | `avg_recursion_steps` | 2.128 ± 0.088 | 2.085 ± 0.046 | — | — |
  | `max_pairwise_cosine_sim` | 0.0260 ± 0.0137 | 0.0493 ± 0.0069 | — | — |
  | collapsed experts | none / none / none | none / none / none | — | — |

  **Rule 1 (canonical value not selected here).** `family_cls = 0.5` remains
  frozen and INHERITED; `family_cls = 0.0` is the labelled variant
  `no_family_supervision` and never shares a table with a canonical row. Both
  arms are `experiment_group = exploratory` proxies at 20 epochs.

  **Rule 2a — task-loss cost of removing family supervision: none measurable.**
  cost = **−0.000100** (the ablation is nominally *lower*) against a largest-arm
  seed std of 0.000428. Inside seed noise, so the 6-way family CE does no
  measurable work on the primary predictive metric. The term shapes the trunk's
  representation, not its accuracy.

  **Rule 2b/3 — CASE 2, pre-registered: the partition survives.** With
  `family_cls = 0.0` there is **no oracle-derived label anywhere in the
  objective** (`step_routing = 0.0` already, now `family_cls = 0.0` too), and
  Hungarian accuracy is still 0.4300 vs chance 0.167 with AMI 0.3568 —
  **76% of the above-chance partition retained**. The remaining ~24% is
  attributable to the family CE. Reported as measured, not rounded up: family
  supervision *contributes to* the partition but is not what creates it.

  **Rule 4 — no collapse in either arm.** Load entropy 0.954 / 0.980, no expert
  starved, and the Hungarian assignment differs on every seed in both arms
  (canonical `[4,0,5,1,3,2]`, `[5,3,4,1,0,2]`, `[0,1,4,5,3,2]`; ablation
  `[5,2,3,1,4,0]`, `[1,3,0,4,2,5]`, `[1,3,0,5,4,2]`). Stable accuracy under a
  different relabelling each seed is the evidence of a real partition; identical
  assignments at ~1.0 would have meant the index was supervised.

  **Reproduction cross-check passed exactly.** Fresh `famcls_fam_seed42`
  `val/task_loss = 0.064011740289` vs T8.0b `routedec_unsupervised_router_seed42`
  `0.064011740289`, |delta| = 0.000e+00 at tol 1e-9 — moving `family_cls` out of
  the loss assembly into config (T8.3) and adding the matched metric layer (T6.8)
  left the objective bit-identical.

  **Matched-protocol check: 54 comparisons (12 provenance + 6 model fields × 3
  seeds), 0 mismatches** — `loss_weights.family_cls` is the only field that
  differs. Note: the captured console banner reads "T10.F" because the process
  was launched before the T10.F→T10.H rename landed on disk; the driver source
  and all future runs print T10.H. The JSON result carries no label.

- [x] **T10.I** MoR at `ffn_mult = 4` — the *under-budgeted* MoR baseline
  (567k-class parameter count, 82.2% short of MoRE), variant tag `ffn_mult_4`.
  Canonical MoR is `ffn_mult = 24`, which equalises **total FFN hidden width**
  against MoRE's six experts (T9.2 gap 0.120%). This arm answers the different
  question of what MoR does at MoRE's *per-FFN* multiplier, and may never be
  tabled as canonical.
  **Verify:** run resolves to `variant = ffn_mult_4` and the Gate 0 guard
  refuses `canonical_phase_b` for it.
  **SCOPED AT T11.0c — a 1-epoch proxy discharges this task.** The `Verify` clause
  above is a statement about the **guard**, not about task loss: it asks that the
  variant resolve and that `canonical_phase_b` be refused. Both are decided at
  `run_context.assert_not_silent_proxy()` before the first optimizer step, so a
  1-epoch run answers it as completely as fifty epochs would, at ~1/50 the GPU.
  The *scientific* question the arm could also ask — what MoR does at 82% under
  budget — is not needed: the headline defence of the parameter budget is already
  stronger without it, because canonical MoR wins on `val/task_loss` with **fewer**
  parameters than MoRE (3,197,710 vs 3,201,555, −0.120%), so no reviewer can
  attribute MoR's result to capacity. An under-budgeted arm would answer a question
  nobody is asking. The proxy run must be stamped `experiment_group = exploratory`
  and must never appear in a loss table; it exists to exercise the refusal path,
  and it doubles as the negative test **T11.1** needs for the exporter.
  **DONE — both halves of the Verify clause pass.**
  *Half 1, the variant resolves.* `runs/t10i_mor_ffn_mult_4_proxy_seed42__f2e3e60d`,
  1 epoch, ~1 min: `provenance.variant = ffn_mult_4`, `model.ffn_mult = 4`,
  `total_params = 571,150` — the 567k-class figure, **82.2% short** of MoRE's
  3,201,555, confirming the arm is the under-budgeted baseline it claims to be and
  that `ffn_mult` reaches the model rather than only the config. Stamped
  `experiment_group = exploratory`, `resolved_epochs = 1`; it must never appear in a
  loss table and its `val = 0.0733` is recorded here only as evidence the run
  trained, not as a measurement.
  *Half 2, Gate 0 refuses the canonical claim.* The same command with
  `--experiment_group canonical_phase_b` is **refused before the first optimizer
  step**, naming all three violations independently: `epochs: canonical requires
  50, run has 1`; `ffn_mult: canonical mor requires 24, run has 4`; and
  `variant='ffn_mult_4' -- this run has at least one ablation active, so it may not
  claim experiment_group='canonical_phase_b'`. No run directory is created. The
  third check is the one that matters for this task: even at 50 epochs with every
  other field canonical, the variant tag alone would block the claim.
  *Third use, free.* The proxy directory is now also a refusal row in the T11.1
  exporter. It refuses on `experiment_group='exploratory' != 'canonical_phase_b'`,
  which is the *first* check in the admission order — the variant check never runs.
  That ordering is correct (the group is the claim; a run that makes no canonical
  claim needs no further examination) but it means this directory does **not**
  exercise the exporter's variant path. Nothing under `runs/` currently does: no run
  has ever been stamped `canonical_phase_b` with a non-canonical variant, because
  Gate 0 refuses that combination at launch, as half 2 above just demonstrated. The
  exporter's variant check is therefore defence in depth against a future
  hand-edited provenance block, not a path with live coverage.

---

## Phase 11 — Export and Report  (`plan.md` §14, §15) — Step 16, 17

- [x] **T11.0a Reporting-layer statistics defect + checkpoint-rule ruling.**
  Found while auditing the comparison protocol (audit item 3) by recomputing the
  matrix independently. `mean_std()` in `code/run_phase9_matrix.py` and
  `automated/phase10_ablations.py` divided by `n`, not `n−1`, understating the seed
  std 22% at n=3 / 12% at n=5 and inflating every `2 × std` ratio by the same
  factor — always in the direction of manufacturing a resolution. The three earlier
  decision drivers (`act_decision.py:181`, `family_cls_ablation.py:230`,
  `routing_supervision_decision.py:203`) had always used `n−1`, so the reporting
  layer was the outlier. **Fixed in both, `n < 2 → std = None` (never `0.0`), both
  reports regenerated from the run directories.**
  **Four verdicts changed:** `more − mor` 2.16 → **1.93×**, T10.B 2.10 → **1.88×**,
  T10.E 2.16 → **1.94×** (all now inside noise); T10.J depth 6.4 → 5.2× (still
  resolved). `moe − mor` 3.62× survives. **The canonical matrix now holds exactly
  one resolved difference — MoR beats MoE — and MoRE is indistinguishable from
  both baselines.**
  **Checkpoint rule declared: last-epoch is canonical.** `best_val_loss` is a
  minimum-of-25 order statistic whose downward bias scales with each
  architecture's per-epoch validation noise (MoR std 0.000052 vs MoE 0.000194), so
  selecting on it confounds quality with curve noise. Under best-val **nothing** in
  the matrix resolves (`moe − mor` falls to 1.51×), so the one surviving claim was
  rule-dependent and undeclared. Last-epoch is what every report already used.
  **Must be disclosed in the paper:** `checkpoint.pt` is the best-val checkpoint
  and is therefore not the model whose metrics are reported; both columns go in the
  results file.
  **Verify:** both drivers re-run clean; `grep '/ len(nums)'` returns nothing in
  either. See `changelog.md` → T11.0a.

- [x] **T11.0b Blocking pre-Phase-11 audit → `PHASE11_AUDIT.md`.** Discharged
  audit items 3–7. **350/350 regression checks pass; leakage 8/8; the comparison
  protocol passes cleanly.** Three findings changed conclusions.
  **(1) The reading rule was wrong, not the measurements.** Every verdict in this
  repo compared a difference of means against `max(std_a, std_b)` — one arm's
  spread — at a `2 ×` threshold. The standard error of a difference is
  `sqrt(s_a²/n_a + s_b²/n_b)`, ~√n smaller; MoRE − MoR is **3.94 SE**, not the
  "1.93 × std" printed. Replaced with an **exact randomization test** (all
  C(10,5)=252 regroupings) in the new shared module `code/seed_stats.py`, which
  also returns `min_p`, the design's resolution floor.
  **Canonical matrix, exact p:** MoR − MoE **0.0079**, MoR − MoRE **0.0159**,
  MoRE − MoE 0.659. So **MoR beats both**, and MoRE ≈ MoE — which, since MoE and
  MoRE are parameter-identical to the unit (3,201,555), is the clean isolation of
  recursion. **Phase 10:** `dense_routing` p = 0.0179 (worse) and
  `routing_supervision` p = 0.0179 (worse) both resolve; the other four do not.
  Both sit at the n=3 floor (1/56) and fail Bonferroni at six arms.
  **(2) The frozen checkpoint rule is not the rule that ran.**
  `canonical_spec.json → protocol.checkpoint_selection` freezes best-val; every
  report used last-epoch, and the spec's own instruction to re-check their
  agreement on the matrix now **fails** (all 15 runs diverge). The qualitative
  conclusion is identical under both rules, so this is a declaration problem —
  **but amending a frozen field is the researcher's call and is left open.**
  `checkpoint.pt` is not the model whose metrics are reported; disclose it.
  **(3) Two Rule 4 violations** in the reporting layer: `halt/early_exits` /
  `halt/forced_exits` report numbers for MoE where the rates are `"N/A"`
  (`engine.py:875` guards on the exit tally, not on `max_depth > 1`), and
  `train/halting_supervision_loss = 0.0` in all 15 runs for a term never computed
  (`engine.py:814`). Both must become `"N/A"`; `test_phase3_halting.py:626` and
  `test_gate5.py:156` must be relaxed to accept it.
  Also established: MoE == MoRE params **exactly**; the MoR gap is 3,845
  (0.120 %) = 1,285 halt heads + 1,280 router + **1,280 FFN biases the frozen
  `parameter_budget_note` omits**, and it runs *against* the result so MoRE's loss
  cannot be blamed on capacity; **all three architectures explain only 21.8–22.7 %
  of target variance** against the frozen floor 0.080914, and the entire
  between-architecture spread is **3.1 % of that**; the depth headline 2.05 is a
  *training*-time statistic (no `val/avg_recursion_steps` exists); **50 metric keys
  are absent rather than `"N/A"`** and the exporter must not render them `0.0`;
  all 15 canonical runs are `code_git_dirty: true`, the weakest link in provenance.
  **Retracted:** the earlier claim that `lr`/`weight_decay`/`d_model`/`num_blocks`/
  `num_experts`/`max_depth`/`routing_balance` were provenance gaps — they reach
  both `resolved_config.json` and W&B via `engine.py:277`.
  **Verify:** `code/run_correctness_suite.py` → 350/350, Gates 1–5.
  `code/audit_leakage.py` → 8/8. `code/seed_stats.py` reproduces the audit's §7.1
  table from the run directories and returns `None`, never `0.0`, on its N/A
  paths. See `PHASE11_AUDIT.md` §8 for the 12-row required-action table and
  `changelog.md` → T11.0b.

- [x] **T11.1 `code/export_results.py`.** Filter
  `experiment_group == canonical_phase_b AND dataset_version == CANONICAL`;
  refuse mismatched dataset versions or config hashes; emit `N/A` rather than
  fabricate; never hand-copy numbers.
  **Verify:** feeding it an archived invalidated run causes a refusal, not a row.
  **DONE.** Admission requires `experiment_group == canonical_phase_b`,
  `variant == canonical`, `seed_declared == True`, `resolved_seed` in the frozen
  set, and `dataset_version` / `train_split_version` equal to `canonical_spec.json`.
  Of 120 directories under `runs/`, **15 admitted, 105 refused**, each refusal
  carrying the field that failed and exported in section 5 of the Markdown — a
  silently-skipped bad run is indistinguishable from no bad runs.
  **Verify clause passed:** `runs/phaseB_moe__seedNA__f83aa42f` (the invalidated
  undeclared-group run, `provenance.architecture = None`) produces
  `experiment_group='exploratory' != 'canonical_phase_b'` and **no row**; it does not
  raise, because a refusal arriving as a traceback is an outage, not a refusal.
  Consistency is a hard error that writes nothing: one `dataset_version`, one
  `train_split_version`, one `code_git_commit`, no duplicate `(arch, seed)` cell,
  and seed-blind config identity within each arm (all 15 pass).
  **Discharges audit items 5 and 6.** 130 scalar keys in the union; **550 cells
  export as `N/A`**, distinguishing absent (the architecture has no such quantity)
  from `engine.py`'s own refusal, with `n_na` and `n_bool` as columns so a mean over
  3 of 5 seeds cannot look like a mean over 5. R² is derived from the frozen
  `primary_metric_floor = 0.080914`, never stored: MoE 0.2182, MoR 0.2250,
  MoRE 0.2191.
  Primary metric read as **last epoch** per the amended `protocol`; `best_val_loss`
  exported as a labelled secondary column. Verdicts come from
  `seed_stats.perm_test` only, and reproduce `run_phase9_matrix.py` to the digit
  (MoRE−MoE p=0.6587, MoRE−MoR p=0.0159, MoE−MoR p=0.0079, floor 0.0040).
  Phase 10 arms are **ingested** from `automated/phase10_ablations_result.json`
  into a separately headed exploratory section with its own floor — never merged
  into the canonical tables.
  Writes `results/results.csv` (long, per run, no aggregates or verdicts, so it
  cannot encode a reading rule), `results/results_aggregate.csv`,
  `results/results.json` (raw per-seed vectors beside every aggregate),
  `results/results_tables.md`. Read-only with respect to `runs/`.

  **T11.1 EXTENSION — the last three audit items closed (same window).**
  - **Audit item 7 (depth claims were training-time only) — CLOSED, and it changed
    a result.** Every depth number in the project came from inside the training
    loop (`depth/allocation_error_abs` at `engine.py:905`, under dropout, on train
    data, halt head mid-update), while the paper's depth claim is about the trained
    model on held-out data. Two fixes landed. (a) `engine.py`'s validation pass now
    emits `val/avg_recursion_steps`, `val/depth_allocation_error_{abs,rel}`,
    `val/{forced,early}_exit_rate`, `val/mean_remainder` — the model already
    returned `val_expected_depth` / `val_halt_stats` from that same forward pass
    and both were unpacked and discarded, so this costs no extra compute. The
    training-time keys are unchanged: two measurements, not a correction.
    (b) `code/eval_val_depth.py` recovers the same quantities for runs that predate
    the metric, from each run's `checkpoint.pt`, into a `val_depth_offline.json`
    sidecar (`metrics.json` is never modified). **Verified:** MoRE/MoR/MoE 1-epoch
    smoke runs give real values / real values / `N/A` respectively; 15/15 canonical
    runs and 22/22 Phase 10 arm runs have sidecars; correctness suite **350/350,
    5/5 gates PASS**.
    **The result:** on held-out data at the best-val checkpoint, MoRE spends
    significantly *more* depth than MoR (2.353 ± 0.109 vs 2.128 ± 0.019 steps,
    +0.225, d = +2.88, p = 0.0079) while its absolute depth allocation error is
    statistically indistinguishable (0.994 ± 0.105 vs 0.942 ± 0.027, +0.052,
    d = +0.67, **p = 0.357**) and its *relative* error is significantly worse
    (+0.0995, p = 0.0159). MoRE's extra recursion buys no better agreement with the
    operation-complexity curriculum. The T10.J arm reads the same way: dropping the
    ponder coefficient 10× raised held-out depth 2.38 → 2.77 and made allocation
    error *worse* (1.03 → 1.15) without moving the task loss (p = 0.16) — depth rose,
    loss did not follow, which is the pre-registered Outcome-C branch for the depth
    half. Every number is measured at the best-val checkpoint, **not** the last-epoch
    model the primary metric comes from; that caveat is printed in
    `measured_at: best_val_checkpoint`, in `results_tables.md` §2b, and in
    `results.json:offline_depth_note`, so it cannot be lost in transcription.
    Ingested by `export_results.py` (§2b + `pairwise_offline_depth`) and by
    `phase10_ablations.py` (its own held-out depth block) rather than read by hand.
  - **Audit item 10 (`config.json` did not contain three fields its `_README`
    claimed) — CLOSED.** `model.routing_mode = "top1_sparse"`,
    `model.router_noise = "none"` and `data.subset_fraction = 1.0` are now written
    out. **Verified hash-neutral:** `load_config()` already injected all three
    before `config_hash()` is taken, so the default config still hashes to
    `b7b17489cdc2` and the 15 completed canonical runs stay comparable with anything
    launched later. `halting_mode` remains absent from the file by design — it is
    derived by `resolve_halting_mode()`, which is the guard's own source.
  - **Audit item 11 (stale T10.J pre-registration text) — CLOSED**; the bracketed
    `[PRE-REGISTRATION — LEFT VERBATIM. … p = 0.125 — NOT significant]` annotation
    is in place at `automated/phase10_ablations.py:101` with the pre-registration
    itself unrewritten. `TASKS.md`'s reference to the T8.0b driver corrected to
    `automated/routing_supervision_decision.py`.
  - **Deferred cleanup landed:** the dead 109-line `metrics.evaluate_paper_metrics`
    and its `engine.py:31` import are deleted, replaced by an in-source note naming
    the four shared helpers to reuse instead of resurrecting a parallel copy.
  - **`ARCHITECTURE.md` refreshed to the current repository (same window).** It had
    drifted to a pre-Phase-8 state and was actively misleading: it reported
    **6,359,069 / 1,098,259** parameters (now 3,201,555 / 3,197,710), quoted
    `routing_balance = 0.05` and `step_routing = 0.5` as the architecture-defining
    weights (now 0.001 and **0.0**), listed module line counts from the refactor
    (`engine.py` 594 → 1280), stopped its status section at Phase 6 with "Next: the
    controlled ACT decision experiment", described `evaluate_paper_metrics` as
    "retained but marked UNUSED in-source" after it had been deleted, and said five
    `enforced_fields` entries were still `null` when the spec has been frozen since
    Phase 8. Added: the canonical result table with the exact-test p-values and the
    three framing confounds; the §5 subsection stating that the canonical router is
    **unsupervised** (`step_routing = 0.0`) and that raw `val/routing_accuracy`
    (0.163 ± 0.186) must therefore be read through Hungarian/AMI/purity; the
    `family_cls` disclosure with its measured retained-fraction (76% Hungarian, 72%
    AMI, 80% purity with the term off); the §5a three-prefix table separating
    `train/` from `val/` from `val_offline/` depth with the best-val-checkpoint
    caveat and the held-out matrix; `mor.ffn_mult = 24` as the T8.3 budget decision;
    the T10.A–T10.J arm roster with its seed counts and the α = 0.0083 /
    floor-0.0179 arithmetic; the §6 randomization-test reading rule; the read-side
    call graph (`export_results.py` → `seed_stats.py` → `results/`); and
    `run_correctness_suite.py` as the one command whose gate numbers may be quoted
    (**350/350**, re-run to confirm after these edits).
    **Verify:** `run_correctness_suite.py` 350/350 5/5 PASS; `smoke_test.py` ALL
    TESTS PASSED; `verify_pipeline.py` 35/35; every number in the new sections read
    out of `results/results_aggregate.csv`, `code/canonical_spec.json` or
    `automated/family_cls_ablation_result.json`, none transcribed from prose.
  - **Still open:** audit item 9 (the single end-of-Phase-11 commit hash in the
    results files), deferred by the researcher's git policy. All 15 canonical runs
    remain `code_git_dirty: true` at `f7166b4`, so until that commit lands
    provenance rests on `config_hash` + `resolved_config.json`, and the results
    document must say so.

- [x] **T11.2 `results.csv` / `results.json` / `results.md`** with the 19
  prescribed sections (`plan.md` §15).
  **Verify:** every section present; every number traceable to an
  `experiment_id`; measured facts separated from interpretation.

  **DONE (T11.2 window).** `results/results.md` is 1,625 lines, §1–§19 all present,
  no placeholder markers remain (`grep -nE "<!--CHUNK|TODO|TBD|PLACEHOLDER"` returns
  nothing). Every figure was copied from a generated artifact, never typed from a log:
  `results/results.csv` / `.json` / `results_tables.md` (`code/export_results.py`,
  15 admitted / 114 refused of 129 scanned), `results/seed_stats.json`,
  `runs/*/val_depth_offline.json` (`code/eval_val_depth.py`),
  `automated/phase10_ablations_result.json` (`--report-only`),
  `automated/family_cls_ablation_result.json`, `results/depth_null_model.json`
  (`code/depth_null_model.py`, new this window), and `results/bench_capacity.log`
  (`code/bench_capacity.py`, new this window). §16 names the script behind each one.
  Register is enforced by an explicit MEASUREMENT / INTERPRETATION / HYPOTHESIS table
  in the preamble and those labels are used inline throughout.

  **Two sections that did not exist when this task was written.** §9.1 is the
  best-constant-depth null model — the missing null for the adaptive-depth claim, and
  the reason §19 lands on Outcome C for the halting half. §13 reports the capacity
  microbenchmark and **rejects its own width sweep** as unusable (non-monotonic:
  `d_model = 512` timed *faster* than 256, violating the benchmark's own stated
  invariant), keeping only the batch sweep, which is a clean launch-bound signature.

  **Corrections to the FRAMING RULING text below, left verbatim per convention.**
  Two figures in the ruling are wrong and the corrected values are used in
  `results.md`: (a) "21.8–22.7% of target variance" — the measured R² values are
  0.2182 / 0.2250 / 0.2191, so the range is **21.8–22.5%**; (b) "target mean 1.875" —
  1.8750 is the mean over the 16 operation *types*, but depth metrics are token-level
  and the **token-weighted** curriculum mean on held-out data is **2.1082**
  (`results/depth_null_model.json:curriculum_mean_token_weighted`). The ruling's
  substance is unaffected; §9.1 states both quantities and labels which governs.

  **FRAMING RULING — T11.0c, researcher's decision. Binding on T11.2, T11.3 and
  the paper draft.** This is recorded here because the Phase 9/10 results under the
  corrected reading rule invite a framing the paper must NOT take.

  1. **The subject of the paper is MoRE.** This is an exploratory paper on the
     proposed architecture. MoR outperforming it is a *result within* that paper,
     not the paper's thesis. "MoR is good" is not a contribution — recursive
     depth-sharing is established prior work; the novelty being reported is the
     composition of learned expert routing with adaptive recursive reuse.
  2. **Do not restructure around MoR because it won.** Rewriting the paper's
     subject to match whichever arm scored best is choosing the conclusion after
     seeing the data. The comparison is reported at full strength — MoR beats MoE
     (p = 0.0079) and beats MoRE (p = 0.0159), MoRE ≈ MoE (p = 0.659) at identical
     parameter count — and then interpreted, not promoted to thesis.
  3. **MoR's win is confounded by three factors that must be stated, each as a
     limitation on the CONCLUSION rather than as an excuse for MoRE.**
     - **Scale.** Every run is 3.2 M parameters, `d_model = 256`, one block, on a
       single consumer laptop GPU (RTX 4060 Laptop). Expert specialisation is a
       capacity-partitioning mechanism: six experts of width 1024 each see ~1/6 of
       the tokens, where MoR's single width-6144 FFN sees all of them. At this
       width, partitioning may simply cost more than specialisation buys. The
       result is "at ~3 M parameters on this task", and nothing wider was tested.
     - **The dataset.** `more6-v1` is **synthetic and custom**, generated by
       `data/script.py` using Python's `random` over 16 hand-chosen operations with
       a hand-authored depth curriculum. Both of MoRE's mechanisms are being asked
       to discover structure that was *put there by construction*, on a task whose
       difficulty profile we chose. All three architectures explain only
       **21.8–22.7%** of target variance against the frozen predict-the-mean floor
       (0.080914), and the entire between-architecture spread is ~3% of what they
       explain — a regime where the task, not the architecture, is the binding
       constraint. No natural-data or held-out-distribution evidence exists.
     - **The evaluation.** The primary metric is a single scalar regression loss
       (`val/task_loss`) at one fixed train/val split, 5 seeds, 50 epochs, one
       optimizer setting frozen as VALIDATED-STABLE rather than swept per
       architecture. A metric that rewards only final-answer accuracy cannot see
       the properties MoRE is *for* (interpretable routing, per-input compute
       allocation), so it is structurally incapable of crediting them.
  4. **Do not downplay MoRE, and do not inflate it either.** The negative findings
     are reported plainly — recursion adds nothing at matched parameters, and the
     halt head learned an unconditional constant of ≈2 steps (target mean 1.875)
     rather than an adaptive policy. What must NOT happen is presenting those as
     settled properties of the architecture when the compute ceiling is the honest
     reason the design space went unexplored: one laptop GPU, so no width sweep, no
     per-architecture tuning, no larger `d_model`, no second dataset. State the
     limitation as a limitation and let the negative result stand as
     *what was observed at this scale*, per CLAUDE.md §8 Outcome C.
  5. **Register.** Measured facts and interpretation stay visibly separated
     (CLAUDE.md §8). Every confound above belongs in the limitations section with
     its own evidence, not sprinkled as hedging next to individual numbers.

- [x] **T11.3 Rewrite `README.md`.** §2–§3 still publish invalidated results,
  and §1.4 self-contradictorily claims L2 regularization prevents
  `router_noise_scale` from collapsing to 0 (L2 drives it *to* 0).
  **Verify:** no invalidated number remains in `README.md`.

  **DONE (T11.2/T11.3 window).** Rewritten 98 → 203 lines against
  `results/results.md`. Removed as invalidated: the `0.002320` val loss and the
  "87.26% reduction", the `25,470 tok/s` table, load entropy quoted against
  `ln(7)` (canonical is six experts), "simple ops halt at 1.03 / complex at 6.68
  steps" (measurement says 93–97% of tokens sit at exactly 2 steps in *both*
  recursive arms), the oracle-routing weight `0.01` (canonical `step_routing = 0.0`),
  the claim that L2 on a trainable noise scale prevents collapse, the
  "fewer than 7 experts / fallback expert" language, and
  `python run_sweeps.py` → `sweep_results.csv` (that path does not exist and that
  filename is a banned global output).

  **Invalidated numbers are named, not erased.** README §5 lists the old figures
  explicitly *as invalidated*, with the reason each is not comparable (proxy runs,
  7-expert config, dense evaluate-all-and-blend routing, pre-leakage-audit,
  pre-provenance-guard, pre-permutation-invariant-metrics). This satisfies both
  "archive, never delete, evidence" and this task's verification, since nothing
  there is presented as a result. §5 also names the two claims that *inverted*:
  dense routing went from "the fix for blocked router gradients" to a labelled
  ablation that measures significantly worse, and the L2-on-noise-scale claim is
  withdrawn outright.

  README §4 carries the metric-reading rules (never quote `total_loss`; entropy is
  normalized and is a load-balance diagnostic only; raw routing accuracy is
  uninformative under an unsupervised router; "depth allocation error", never
  "compute efficiency"; no sentinel is a measurement), so a reader who never opens
  CLAUDE.md still cannot misread the table in §1. Every command in §3 was executed
  from `code/` before being published.

- [ ] **T11.4 Human scientific audit** (Step 17) before any number enters the
  paper. Report Outcome A, B, or **C** honestly.

---

## Standing constraints (apply to every task above)

- Primary metric is **validation task loss**, never `total_loss`.
- High entropy is a load-balance diagnostic, **not** evidence of specialization.
- No sentinel value (`-1`, placeholder `0.0`) may be reported as a measurement —
  use `N/A`.
- Archive, never delete, evidence.
- Never revert a correctness fix because the previous number looked better.
- Do not optimize toward a predetermined conclusion.
