# changelog.md — what changed, why, and where to look when it breaks

**Purpose.** One entry per completed task. Each entry says what the task was,
how it was done at a level you can reason about without reading the diff, and —
most importantly — **the file and symbol to open if a future experiment
misbehaves or the architecture has to change**.

**How to use this file when something goes wrong.**

1. Find the symptom in the *Failure tracing* line of the relevant entry.
2. Open the file named there. Every entry names a real code location.
3. Check the *Verified by* command still passes before you change anything —
   if it already fails, the regression is in that entry's area, not yours.

Authority order is unchanged: [plan.md](plan.md) →
[updated_rules.md](updated_rules.md) → [updated_objective.md](updated_objective.md)
→ [CLAUDE.md](CLAUDE.md). This file is a record, not a rule.

Entries are newest-last, so the file reads as the order work actually happened.

---

## Refactor — 1615-line `train.py` split into `code/more/`

**What.** The single monolithic `train.py` became one package, `code/more/`, plus
four thin launchers (`train.py`, `train_moe.py`, `train_mor.py`,
`train_more.py`, 16–19 lines each).

**Method.** Split by *responsibility*, not by architecture. That distinction is
the whole point: there is one training system and `architecture = moe | mor |
more` is a runtime mode, so `engine.py` holds the only training loop that exists
and all three baselines run through it. Splitting per-architecture instead would
have produced three copies of the data loader and loss assembly, and any measured
MoE-vs-MoRE difference would then be confounded with an implementation
difference. `plan.md` §9 forbids it for exactly that reason.

The package is named `more/` because MoRE is the *project*. It is not the "more"
mode. There is deliberately no `moe/` or `mor/` folder — see
[ARCHITECTURE.md](ARCHITECTURE.md) §1, which carries the field-by-field
`resolved_config.json` diff proving the three modes share one pipeline
(47 fields: 35 identical, 6 run identity, 6 the architecture definition itself).

**Failure tracing.**
- Loss assembly / training step wrong for *all three* architectures →
  [code/more/engine.py](code/more/engine.py), the epoch loop. There is only one.
- Wrong for *one* architecture only → [code/more/config.py](code/more/config.py)
  `apply_architecture()`. That function is the entire definition of what makes
  MoE, MoR and MoRE different. If a difference is not in that function, it is a
  bug, not a design.
- Input tensors wrong → [code/more/data.py](code/more/data.py).
- A launcher containing logic → it must not. `train_*.py` are pure dispatch.

**Verified by.** All three launchers exit 0; `smoke_test.py` ALL TESTS PASSED;
`verify_pipeline.py` 35/35.

---

## Phase 1 / Gate 1 — input leakage removed

**What.** Three prohibited quantities were removed from the model input: the
oracle expert/family index, the regression target, and any slot that reproduced
the routing label.

**Method.** The dataset previously handed the model features from which the
answer was recoverable — the "copy one input slot" baseline achieved MSE
**exactly 0.000000**, meaning the task was solvable without learning anything.
The fix was to restrict `x` to numeric arguments only and move operation identity
into a separate `step_ops` channel consumed by an embedding over *operations*
(16 op codes), never over experts.

Operation identity is legitimate task information — it tells the model which
function to compute. The oracle *expert* index is not; it is the answer to the
routing question being studied. Keeping them in separate channels is what makes
that line enforceable.

Verification is **mutation-based with non-vacuity assertions**, so a check cannot
pass by testing nothing: perturb the targets and assert the features do not move,
permute the op→expert map and assert the features do not move, and separately
assert that the perturbation actually changed something.

**Failure tracing.**
- A suspiciously low val loss, or val loss near zero → suspect leakage first.
  Run `audit_leakage.py`. Read the loss against the **predict-train-mean floor
  0.080914**, not against zero.
- Feature construction → [code/more/data.py](code/more/data.py)
  `MoREDataset.__getitem__`, the 7-tuple contract.
- If a new operation is added and something silently buckets it →
  [code/more/families.py](code/more/families.py). An unmapped op **raises** by
  design; do not add a fallback, that is how the removed E7 catch-all worked.

**Verified by.** `python code/audit_leakage.py` → **Gate 1: 8/8**. Copy baseline
went from `0.000000` to `0.081853`, now *worse* than predict-mean `0.080914`.
Residual honest diagnostic: the target coincides with an argument in 33.45% of
val records, inherent to MAX/MIN/MEDIAN/SORT where the answer *is* an input.

---

## T6.4 / T6.5 / T6.2 — sentinel values stopped being reported as measurements

**What.** MoR (`E = 1`) was publishing numbers for quantities that do not exist
for it. Fixed to emit `N/A`.

**Method.** The rule (`CLAUDE.md` §4) is that no sentinel may be reported as a
measurement. The concrete danger was specific and severe: MoR was reporting
**routing accuracy 0.1005**. MoR performs no routing at all — its argmax is 0 for
every token, so that figure measures "what fraction of tokens happen to belong to
E1". Put in a comparison table it would have read as *MoR is bad at routing*,
which is not a finding, it is an artifact of a 1×1 confusion matrix.

Four quantities were gated on `num_experts >= 2`: normalized entropy (`H/log(E)`,
and `log(1) = 0`), mean/max pairwise cosine (there is no *pair*), routing
accuracy, and per-family recall. Depth metrics are deliberately **not** gated —
MoR does allocate depth, and that is its actual contribution.

Routing accuracy was then collapsed to **one implementation**,
`metrics.routing_accuracy_from_confusion()`, which is the confusion diagonal
fraction by construction, so the reported scalar and the reported matrix cannot
disagree.

**The trap worth remembering.** The first version of this fix did not work and
appeared to. The guard was added to `metrics.evaluate_paper_metrics()`, the run
was repeated, and MoR still printed `route_acc=0.1005`. Cause: `engine.py`
contained its own inlined duplicate, and `evaluate_paper_metrics` was imported
but **never called**. The guard was live code in a dead function.

**Failure tracing.**
- A routing metric is wrong → fix the **engine validation loop**, not
  `metrics.evaluate_paper_metrics`, which is retained but marked UNUSED in-source
  precisely so nobody repeats the above.
- Scalar and confusion matrix disagree → they can't, unless a caller bypassed
  `routing_accuracy_from_confusion()`. Find that caller.
- `metrics.json` contains an object repr with a memory address →
  [code/more/run_context.py](code/more/run_context.py) `write_metrics`. It used
  to `str()` non-JSON values, which wrote a `wandb.Image` repr into the metrics
  file; it now records the key under `_non_scalar_keys_omitted` instead.

**Verified by.** MoR one-epoch run reports `route_acc=N/A`, `entropy_norm=N/A`,
`cos=N/A`; per-family recall keys absent; `metrics.json` free of object reprs.

---

## Phase 2 (T2.1–T2.4) — true Top-1 sparse routing

**What.** The canonical routing path now dispatches each token to exactly one
expert. Previously it evaluated **every** expert on **every** token and blended
the outputs by router probability. That dense path still exists, but only as an
explicitly labelled ablation that the config layer refuses to run unlabelled.

**Why it mattered.** Dense evaluate-all-and-blend is not MoE. No token is routed
anywhere, every expert sees every token, and expert independence — the thing the
paper claims to measure — is destroyed, because every expert receives gradient
from every token and there is no pressure for any of them to specialize. Worse,
the defect is **invisible from the outside**: both paths emit a `[N, d_model]`
output and both report an argmax, so the loss curve, the entropy and the
confusion matrix all look normal. A dense run could have been written up as MoE
with nothing in the artifacts to contradict it.

**Method, in four parts.**

1. **Sparse dispatch** (`MoEBlock.forward`). `logits → softmax → argmax →
   gather rows for expert *e* → run only expert *e* on them → scatter back →
   multiply by the selected gate probability`. The gate multiply is not
   cosmetic: `argmax` has no gradient, so without it the router would be trained
   only by the auxiliary and oracle terms and never by the task loss. The gate
   is the router's only path to the task objective.

2. **One index governs both dispatch and scoring.** The tensor returned to the
   metrics *is* the tensor that selected the expert. `plan.md` §4.2's concern is
   scoring an argmax that does not control computation; making it literally the
   same object removes the possibility rather than testing for it.

3. **Capacity policy = `no_capacity_limit`, declared explicitly.** Every routed
   token is evaluated; nothing is dropped. Justified by `plan.md` §4.3 (prefer
   the simple implementation when the distribution does not require capacity
   limiting) — this is single-machine training on synthetic arithmetic, and
   capacity factors exist to bound per-device buffers in distributed
   expert-parallel training. Dropping tokens would also corrupt the depth
   measurement, since a dropped token has no honest exit depth. The reporting
   consequence is the point: `overflow = 0` is now a **measured** zero, not a
   placeholder meaning "not implemented". Real capacity pressure is reported as
   `max_load_fraction`.

4. **Dense gated on the run label, not a boolean.** To use dense routing the run
   name must contain `dense_routing_ablation`, which then propagates into the run
   directory, the W&B run name, and every provenance record. A quiet
   `--dense=true` flag would satisfy the letter of the rule while leaving the
   output indistinguishable from canonical MoE, which is the actual risk.

**Honest compute accounting.** Both paths report the same `dispatched` count (one
per token), so a dense ablation could have been tabled beside a sparse run as if
it cost the same. `expert_evaluations` and `evals_per_token` were added: **1.0**
for true sparse, **num_experts** for dense. That single number is what
distinguishes the two paths in any log or table.

**Two problems hit on the way, recorded because they will recur.**

- `model.py` was found **broken on disk** — it referenced `ROUTING_MODES` at
  line 45 with no definition, so `MoEBlock(...)` raised
  `NameError: name 'ROUTING_MODES' is not defined` and no model could be built
  at all. It was a half-landed version of this task whose direction matched
  `plan.md` §4.1, so it was finished rather than reverted. A duplicate
  balance-loss block (27 dead lines after a `return`, ending in a stale 4-tuple
  `return`) was deleted at the same time, and a tuple-arity mismatch fixed —
  `MoEBlock.forward` returned 5 values while the depth loop unpacked 4.
- **The T2.4 gate initially refused correctly-labelled runs.** It ran inside
  `apply_architecture()`, which executes *before* the CLI applies `--run_name`,
  so it tested the default label `phaseB_more` instead of the real one. Moved to
  [code/more/cli.py](code/more/cli.py) after `resolve_overrides`. The general
  lesson: **any check on a run label must run after override resolution.**

**Failure tracing.**
- Suspect the model is secretly dense → check `dispatch/evals_per_token` in the
  run's `metrics.json`. `1.0` = true sparse, `6.0` = dense. Do not try to infer
  it from the loss curve; you cannot.
- Change the dispatch mechanism → [code/more/model.py](code/more/model.py)
  `MoEBlock.forward`, the `if self.routing_mode == "top1_sparse"` branch. The
  routing-mode constants are at the top of the same file.
- Router not learning / expert collapse → the gradient path into the router is
  the **gate multiply** at the end of that branch. If it is removed or detached,
  the router silently stops receiving task gradient and nothing else in the
  pipeline will complain.
- Add a new routing mode → it must be added to `ROUTING_MODES` in `model.py`
  *and* handled in `enforce_routing_mode()` in
  [code/more/config.py](code/more/config.py); an unknown mode raises by design.
- Overflow or capacity behaviour needs to change → the policy constant and its
  full justification are `CAPACITY_POLICY` in `model.py`. Changing it means
  changing what `overflow = 0` *means*, so update the reporting with it.

**Verified by.** `python code/test_phase2_routing.py` → **19/19 checks passed**.
The checks are measurements of the running code, not inspections of it: forward
hooks on each expert count the rows it actually receives, since that is the only
externally visible difference between the two paths. Key figures — sparse
delivered **64 rows for 64 tokens** where dense delivered **384 = N×E** (so the
check discriminates rather than passing trivially); 6/6 experts called
(non-vacuous); selected expert gradient **572.019427** with all others exactly
**0.000000**; router gradient through the gate **30.010744**; output equals
expert output × gate to **5.96e-08**; 64/64 tokens traced back to the expert that
ran them with 0 mismatches; confusion diagonal vs `mean(pred==oracle)` delta
**0.00e+00**; unlabelled dense **refused**, labelled dense **allowed**.

`smoke_test.py`, `verify_pipeline.py` (35/35) and `audit_leakage.py` (8/8) all
still pass.

**Reading caveat that belongs with these numbers.** MoRE reports
`max_load_fraction = 1.0`. This is a real measurement but it is the **worst
single step across all blocks × depths**, and at deep recursion steps very few
tokens remain active — one expert legitimately takes 100% of a 1-token step. It
is not evidence of router collapse. MoR's `1.0` is `1.0` by definition, since it
has one expert.

**End-to-end confirmation on the full dataset** (1 epoch, unseeded, offline W&B,
both exit 0) — the two paths are now distinguishable from their artifacts alone:

| | `evals_per_token` | `expert_evaluations` | val task loss | route acc |
|---|---|---|---|---|
| canonical sparse | **1.0** | 666,658 | 0.074766 | 0.2445 |
| dense ablation | **6.0** | 5,003,142 | 0.075369 | 0.5468 |

`5,003,142 = 833,857 × 6` exactly, so the dense ablation can no longer be tabled
as if it cost what sparse costs.

**Do not read the route-acc column as a finding.** Unseeded 1-epoch MoRE runs gave
`0.5090`, `0.1689` and `0.2445` on the canonical path. The spread across seeds is
larger than any difference the paper would claim, which is why seeds `42–46` with
mean ± std are mandatory and why **T6.1 (global seeding) blocks every headline
number.** Nothing in this table may enter a paper.

---

## Naming — `val/task_loss` added as the unambiguous primary metric key

**What.** The validation loss was logged only as `val/loss`. It now also logs as
`val/task_loss`; `val/loss` is kept as an alias so nothing downstream breaks.

**Why.** `CLAUDE.md` §4 makes validation **task** loss the primary predictive
metric and forbids ever reporting `total_loss` as the quality metric. The value
behind `val/loss` was verified correct — it is `F.mse_loss` on the regression head
with no auxiliary, balance, halting or routing-supervision term mixed in — but the
*name* invites exactly the misreading the rule exists to prevent, and a reader
assembling a table from `metrics.json` has no way to tell from the key alone.

**Failure tracing.** If a val number ever looks implausibly low, or negative,
confirm what is inside it at the `l_v = F.mse_loss(...)` line in
[code/more/engine.py](code/more/engine.py)'s validation loop. A negative weighted
total is not automatically a bug, but it is never evidence of quality, and it must
never be the number in a table.

---

## Phase 3 (T3.1–T3.4) — halting became a trained mechanism instead of a number

**What.** The recursion-halting mechanism was replaced with canonical ACT. Before
this, the "halting loss" was `active_mask.float().mean() * 0.05` — a scalar
computed from a *boolean* tensor. It had `requires_grad = False`.

**Why it is the most serious defect found so far.** Measured before the fix, on
the real objective, after a full backward pass: **all 24 halt-head parameters had
`grad is None`.** The halt heads were never trained. Not weakly trained — not
trained at all. And the run looked completely normal: tokens still exited at
different depths (randomly-initialised sigmoid heads produce different outputs),
the depth histogram still had spread, the exit-rate metric still moved. Any prior
result described as "adaptive depth allocation" was reporting the behaviour of
random initialisation.

**Method — the fix is one structural change, not a new equation.** `plan.md` §5.2
says explicitly: do not invent a novel halting equation. So this is Graves 2016 /
Universal Transformer as written: `p_t = sigmoid(halt_head(state))`, cumulative
mass `c_t`, a token halts at the step where `c_t + p_t` would exceed
`threshold = 1 - eps`, and its final step carries the **remainder** `R = 1 - c_t`
instead of `p_t`.

The single line that matters: the output is now the weighted sum of the states of
the steps a token actually took, and those weights sum to **exactly 1**. The old
code copied the state at exit time into an output buffer. Copying is a discrete
choice — the halt probability decided *which* state was copied and then vanished
from the graph. Multiplying each step's state by its halt weight is what puts the
halt parameters on the path from the **task** loss. Everything else in Phase 3
follows from that.

`eps > 0` is not cosmetic: with a threshold of exactly 1.0, a token with
`p_1 = 0.999` would still be dragged into a second step, so single-step exit would
be unreachable and E1/E4/E5 could never learn their target depth of 1.

**The ponder cost is now `(N + R) / (max_depth + 1)`.** `N` is the discrete step
count and carries no gradient; `R` does, and is the entire differentiable path
into the halt heads from the regularizer side. The normalizer matters for a
reason unrelated to taste: unnormalized, the term grows with `max_depth`, so a
deeper model is penalised for its *shape* rather than its *behaviour*. It is
averaged over blocks, not summed, for the same reason (CLAUDE.md §2).

**The operation-complexity curriculum had to be written from scratch.**
`updated_rules.md` §2.3 refers to it as though it exists; the dataset does not
contain it. Records carry only `depth`, which is program chain length — a
property of the *program*, not of the *operation*. So `OP_TARGET_DEPTH` is new,
lives in [code/more/families.py](code/more/families.py), and is labelled in-source
as **a design choice a human wrote, not a measurement and not something the model
discovers.** The permitted claim is the one the rules mandate: *the model learns
to allocate recursion depth in accordance with the predefined
operation-complexity curriculum.*

One deliberate property of that table is worth knowing before trusting any depth
number: it is **correlated with but not identical to** the expert families. E4
LOGIC and E5 SHIFT share target depth 1; E6 SORT/STAT spans 2 (MAX/MIN) and 4
(SORT/MEDIAN). If target depth were a pure function of the expert index, a model
could score a perfect depth-allocation error *by routing alone*, and the depth
metric would carry no information independent of routing accuracy.

**Supervision is an option, not the default, and the mode is in provenance.**
`updated_rules.md` §2.3 permits either supervised curriculum halting or pure
unsupervised ACT but requires the choice be reported. Default is **off** = pure
ACT. Enabling it writes `halting_mode`, `halting_supervision_enabled`,
`halting_supervision_weight` and `ponder_weight` into `resolved_config.json`,
`metrics.json` and the W&B config. Two guards exist because a mislabelled run is
worse than a wrong one: enabling the flag with a zero weight **raises** (the run
would be recorded as supervised while training unsupervised), and enabling it on
MoE **raises** (at `max_depth = 1` a curriculum target of 2–4 is unreachable, so
the term would be a permanent irreducible loss with an unsatisfiable gradient).

**The measured answer to "does ACT learn the curriculum on its own?" is no.**
Two 1-epoch MoRE runs, identical except for the flag:

| | val task loss | avg depth | depth alloc error (abs) |
|---|---|---|---|
| `pure_act` | **0.072163** | 4.756 | 2.718 |
| `supervised_curriculum` | 0.074127 | 2.206 | **0.729** |

2.72 steps of error against a 1–4 target range is close to uninformative, so
unsupervised ACT allocates depth, but not *this* depth. Supervision fixes the
allocation at a small cost in task loss. **Neither row is publishable** — 1 epoch,
unseeded — but the two modes now produce different, correctly labelled artifacts,
which is what T3.3 required. **Which of the two is canonical is a
`canonical_spec.json` decision and has not been made.**

**Failure tracing.**
- Depth metrics look suspiciously good, or suspiciously constant → check
  `halting_mode` in the run's `metrics.json` **first**. A
  `supervised_curriculum` run was *told* the answer; its depth allocation error
  is not evidence that the architecture discovered anything.
- Halt heads not learning → [code/more/model.py](code/more/model.py)
  `MoREWrapper.forward`, the line that multiplies `updated` by `step_weight`
  before adding it into `accumulator`. That multiply is the *only* path from the
  task loss to the halt parameters. If it is removed, detached, or replaced by a
  copy-at-exit, the heads silently stop training and **nothing else in the
  pipeline complains** — that is exactly the bug this phase fixed.
- Output magnitude drifts with depth → the step weights no longer sum to 1. Check
  `halt/mean_halt_mass` in `metrics.json`; it must be 1.0 to ~1e-7. A value below
  1 means the output is a shrunken state and depth is silently rescaling
  activations into the task loss.
- Ponder cost grows when you increase `max_depth` or `num_blocks` → the
  normalizer or the block-averaging was lost. `MoREWrapper.forward` divides by
  `max_depth + 1`; `MoREModel.forward` divides by the block count.
- Depth targets need to change → [code/more/families.py](code/more/families.py)
  `OP_TARGET_DEPTH`. An operation missing from it **raises** at import; do not
  add a fallback, that is how the removed E7 catch-all worked. Changing a target
  invalidates every previously reported depth-allocation number, because the
  metric is defined *against this table*.
- A halt or depth metric reads `N/A` → that is deliberate for MoE. At
  `max_depth = 1` the ponder cost is `(1+1)/2 = 1.0` for every token always. Put
  in a table beside MoR's `0.27` it would read as "MoE ponders hardest", which is
  the T6.4 sentinel-as-measurement failure. Gated in
  [code/more/engine.py](code/more/engine.py) where `metrics.json` is assembled.
- Separating the two halting terms → they are `train/ponder_cost` and
  `train/halting_supervision_loss` and must never be summed for reporting
  (`plan.md` §5.4). They pull in **opposite** directions: ponder cost pushes depth
  down, curriculum pushes it toward the target. A single combined number can sit
  flat while both components move a great deal.

**A trap this phase added, recorded because it already bit once.** The
`MoREModel.forward` return tuple grew from 11 to 13 elements.
`test_phase2_routing.py` read the dispatch stats as `res[-1]`, which silently
became the new halt-stats dict, and two Phase 2 checks failed in a way that looked
like a Phase 2 regression. It was not. **Do not index this tuple from the end.**
If it grows again the same failure will reappear in the same shape somewhere else.

**End-to-end on the full dataset** (1 epoch each, unseeded, offline W&B, all exit
0). Halting is measurably active and the three modes are distinguishable from
their artifacts alone:

| | avg depth | early-exit rate | forced-exit rate | halt mass | ponder cost | depth alloc err |
|---|---|---|---|---|---|---|
| MoE  | 1.000 | N/A | N/A | N/A | N/A | N/A |
| MoR  | 2.000 | 0.99998 | 0.000025 | 1.0000000 | 0.274614 | 0.930909 |
| MoRE | 4.756 | 0.87586 | 0.124137 | 1.0000000 | 0.474371 | 2.718111 |

**Verified by.** `python code/test_phase3_halting.py` → **29/29 checks passed**,
every one a measurement of the running model rather than an inspection of source.
Key figures — step weights sum to **1.000000000**; active rows per depth step
`[28, 28, 16, 5, 1]` (non-increasing, measured with a forward pre-hook, so
eviction is observed rather than asserted); halt bias −20 gives
`forced=28 early=0` and +20 gives `forced=0 early=28`, so the exit-rate metric
discriminates instead of reporting a constant; remainder at a depth-1 exit exactly
**1.0**; 12 pad tokens produce **0** exits; **24/24** halt parameters carry
gradient from the task loss alone, total |grad| **0.649250**, versus `None` on all
24 before the fix, and 0/24 when the heads are frozen (so the check is
non-vacuous); `plan.md` §5.5's synthetic case moves expected depth **up** toward a
target of 4 (`2.0474 → 2.0618`, loss `0.105904 → 0.104354`) and **down** toward a
target of 1 (`6.7629 → 2.2913`) — opposite directions from the same code, so the
gradient is target-driven rather than merely monotone.

`smoke_test.py` ALL TESTS PASSED, `verify_pipeline.py` 35/35,
`audit_leakage.py` 8/8 and `test_phase2_routing.py` 19/19 all still pass.

**Where this is written down for future agents.**
[ARCHITECTURE.md](ARCHITECTURE.md) §5a is the reference section for halting: the
ACT pseudocode, why a convex combination rather than copy-at-exit, the `halt_eps`
rationale, the two-terms-never-summed table, and the instruction to **read
`halting_mode` before reading any depth number**. §7 carries the Gate 2 check
table, §10 the post-Phase-3 halting numbers. Two entries were added to §10's
known-defect list, because neither is fixed by this phase and both will mislead
someone otherwise: that **unsupervised ACT does not learn the curriculum**
(2.718 vs 0.729 allocation error) and that the canonical-mode choice is still
open, and that the 13-element `MoREModel.forward` tuple must never be indexed
from the end. Task-level evidence per check is in [TASKS.md](TASKS.md) Phase 3.





---

## Phase 4 (T4.1–T4.3) — the balance objective stopped outweighing the task

**What.** The routing-balance auxiliary loss was **summed** over every
`(block, depth)` router call. It is now the **mean** over those calls, its two
halves are reported separately, and its coefficient was re-selected from
measurement and frozen. The objective's own formula was not touched — per call it
is still `-entropy_term + switch_aux`. Only the aggregation, the reporting, and
the weight changed.

**Why this one was easy to miss.** It is a *magnitude* defect, and magnitude
defects crash nothing. Every run completed, every metric moved, nothing looked
wrong. Two things were quietly true:

1. **The auxiliary was not subordinate.** Measured before the fix,
   `0.05 × (-4.861) = -0.243` against a task loss of `0.001666` — the balance
   term was roughly **146×** the thing the model is supposed to be learning.
2. **Every depth and block-count comparison was confounded.** A 2-block,
   7-depth model produced up to 14 copies of a term a 1-block, 1-depth model
   produced once, *for identical routing behaviour*. So a deeper model was not
   only deeper, it was also more heavily regularised, and "depth helped" could
   never be separated from "more regularisation helped" after the fact. That
   makes it a threat to the paper's central comparison, not just to a number.

**Method — T4.1, the normalization.** Divide by the number of calls actually
made, on both axes: inside `MoREWrapper.forward` by the realised depth-call count,
then in `MoREModel.forward` by the block count. The divisor is the **realised**
count, not `max_depth` — under ACT the depth loop breaks early once every token
has halted, so dividing by `max_depth` would leave a residual depth dependence.
This is deliberately the same aggregation the Phase 3 ponder cost already uses.
Mean-of-means equals a flat mean over all calls whenever blocks run equal depth;
where they do not, equal weight per block is the chosen convention. `metrics.json`
now records `routing_balance_normalization` so no future reader has to guess
whether a tabled balance number is a mean or a sum — that ambiguity is what let
the defect live.

**Method — T4.2, the weight.** `plan.md` §6.2 says not to inherit the historical
`0.05`, so it was selected by running the same 1-epoch full-dataset MoRE
configuration twice, changing only the weight:

| | `w = 0.05` (historical) | `w = 0.001` (frozen) |
|---|---|---|
| `train/task_loss` | 0.078803 | 0.078808 |
| `train/routing_balance_loss` | −0.653436 | +1.365229 |
| **weighted balance / task** | **0.414599** | **0.017323** |
| same ratio projected to converged task 0.001666 | ≈ 19.6 | ≈ 0.82 |
| soft entropy `H / log E` | 0.986127 | 0.694558 |
| hard load entropy `H / log E` | 0.989215 | 0.955612 |
| hard expert load, min–max % | 13.4 – 23.6 | 9.5 – 29.3 |
| `val/routing_accuracy` | 0.922945 | **0.997934** |
| `val/task_loss` | 0.072073 | 0.074507 |

Both weights satisfy the stated `ratio < 1` criterion *after warmup*. `0.001` was
chosen because it is the only tested weight that still satisfies it **at
convergence**: task loss falls over 50 epochs, and projected onto the best
converged task loss this repo has actually reached, `0.05` climbs back to ≈ 19.6×
— i.e. the defect returns by the end of training. Nothing is starved at `0.001`
(no expert below 9.5 % of tokens). Frozen in `code/config.json` *and* the
`more/config.py` default, with a gate check asserting the two cannot drift apart:
if they did, runs that pass `--config` would optimise a different objective from
runs that do not, and the difference would never surface in a results table.

**The finding that matters more than the weight.** Lowering the balance pressure
*improved* routing: `val/routing_accuracy` went from **0.9229 to 0.9979** while
soft entropy *fell* from 0.9861 to 0.6946. Pushing routing toward uniform makes
the router agree with the oracle operation families **less**. This is the concrete
demonstration of why `CLAUDE.md` §2 forbids tuning toward maximal entropy and why
the old `expert_entropy > baseline` criterion is void: on this task the two
objectives are in direct tension, and entropy is the one that is not measuring
specialization.

**Method — T4.2, the reporting split.** `entropy_term` and `switch_aux_term` are
now logged separately (plus `entropy_term_normalized = H / log E`), because they
pull in opposite directions — entropy up is good, Switch-aux up is bad — so the
fused number can sit flat while both components drift together. The two runs above
show exactly that risk in miniature: balance moved −0.65 → +1.37, and the split
is what tells you it was entropy dropping *and* Switch-aux rising, not one or the
other.

Two further reporting facts were made explicit rather than left as folklore. A
**negative balance loss is expected**: at perfectly uniform routing the objective
equals `-log(E) + 1 = -0.791759` for `E = 6`, not 0, so most of the reported
magnitude is a constant offset that carries essentially no gradient (measured
`|grad| ≈ 1e-8` at uniform). And with **`E = 1` the objective is a constant** —
entropy of a one-element distribution is 0, the Switch term is exactly 1, so MoR
reports `bal = 1.0000` on every batch forever. `engine.py` now writes `"N/A"` for
all six balance keys when `num_experts <= 1`, because printed next to MoRE's value
that constant reads as "MoR is maximally imbalanced" — a fabricated comparison,
the same sentinel-as-measurement failure as the halting columns in Phase 3.

**Method — T4.3, the gradient tests.** `plan.md` §6.3 asks for three properties
and all three are measured on the real `MoEBlock`, not on a reimplementation of
the formula: from an artificially collapsed router the gradient raises entropy
`0.820 → 1.790` against a `log 6` ceiling of `1.792`, drives Switch-aux
`4.804 → 1.059` against a floor of 1.0, and reduces hard peak load
`1.0000 → 0.8083`; a uniform router sits exactly on the analytic optimum
`−0.791759`; and depth-invariance is 0.180 % across `max_depth` 3 vs 7.

**Two limitations were recorded instead of smoothed over.** Soft entropy read
`0.9988` while one expert held **81 %** of the hard dispatch — entropy is computed
on the mean softmax and is blind to the argmax that actually routes, so a normalized
entropy near 1.0 is *not* evidence of balanced load. And a fully saturated router
(softmax exactly one-hot in float32) has numerically **zero** balance gradient
(`|grad| 2.352e-07`): the term discourages collapse from a soft state, it cannot
reverse a hard one. That is an initialization/noise problem — **T5.4**, next phase.

**Explicitly not done, per instruction.** The halting formulation was not touched,
no supervision was added, and `canonical_spec.json` was not resolved: whether the
canonical matrix runs `pure_act` or `supervised_curriculum` is still an open
decision, and the proxy guard correctly keeps refusing canonical claims while it
is unset.

**Where this is written down for future agents.**
[ARCHITECTURE.md](ARCHITECTURE.md) **§5b** is the new reference section for the
balance objective: the formula, the two-axis normalization and why the divisor is
the realised call count, the `-log(E) + 1` offset that explains the negative sign,
the reporting split, the `E = 1` constant, and a table distinguishing the **three**
different entropy-ish keys — `train/entropy_term_normalized` (soft, what the
gradient sees), `train/expert_load_entropy_normalized` (hard argmax, what actually
dispatched), and `dispatch/max_load_fraction` (a peak-over-all-steps capacity
metric that pins to `1.0` in any ACT run and is therefore **not** the epoch-level
balance companion). §10's known-defect list lost the now-fixed
"balance loss grows with depth × block count" entry and gained three: the
entropy-is-not-load warning, the saturated-router limitation, and the measured
balance-weight-versus-routing-accuracy trade-off.

**If a balance number looks wrong, open these in order.** Per-call formula and
`route_stats`: `MoEBlock.forward` in [code/more/model.py](code/more/model.py).
Depth-axis mean: end of `MoREWrapper.forward`. Block-axis mean: end of
`MoREModel.forward` — note the return tuple is 13 elements and must be indexed by
slot number, never from the end. Reporting, the `balance_to_task_ratio`
measurement and the `N/A` gating: [code/more/engine.py](code/more/engine.py).
Coefficient: `loss_weights.routing_balance` in
[code/config.json](code/config.json), overridable per run with
`--routing_balance`, which refuses `num_experts <= 1` and refuses negative values
(a negative coefficient would *minimise* entropy and actively drive the collapse
the term exists to prevent). Gate: `code/test_phase4_balance.py`, 27/27; Gate 1
(19/19) and Gate 2 (29/29) re-run with no regression. Per-check evidence is in
[TASKS.md](TASKS.md) Phase 4.


---

## Phase 5 (T5.1–T5.5) — the expert count became a real variable, and the config stopped lying about itself

**What was wrong.** Nothing crashed, which is why this class of defect survived so
long. Three of the model's heads were built as `nn.Linear(d_model, 7)` regardless
of `num_experts`, the per-step logits were reshaped with a literal
`reshape(-1, 7)`, and the confusion matrix was allocated `torch.zeros(7, 7)`. At
the canonical six experts that leaves a seventh logit no family can ever occupy,
which softmax still normalises over — so every routing probability was divided by
a competitor that does not exist. The reshape is worse than wasteful: viewing a
`[B, S, 6]` tensor as `(-1, 7)` is a *view*, so step tokens were folded into each
other's logit rows, and the phantom row of the confusion matrix contributed to its
total but never to its diagonal, which breaks the CLAUDE.md §4 invariant that
routing accuracy equals the diagonal fraction. Alongside that, `_EXPERT_FALLBACK`
silently routed any unmapped operation to a seventh catch-all expert that the
six-family manifest had already abolished; `load_config` ran
`setdefault("subset_fraction", 1.0)` on `training` *before* the guard
`if "subset_fraction" in cfg["training"]`, making that guard unconditionally true
so `data.subset_fraction` was overwritten on every single load; and a trainable
`router_noise_scale` initialised to 0.1 was injected into the router logits on
every training forward pass while appearing in no config file, no resolved config,
and no W&B record.

**Methodology.** Each of the five defects was fixed at its single source and then
verified by *measuring the running model*, never by reading the source. The expert
count is now taken from one place — the family manifest — and threaded through as
a parameter: `CANONICAL_NUM_EXPERTS` in `code/more/config.py` is defined as
`NUM_EXPERTS_CANONICAL` imported from `families.py`, not restated as a literal,
because a second copy of the number is exactly how the width and the label list
drift apart. The config fix replaced the shim with an explicit reconciliation:
`subset_fraction` and `subset_seed` are snapshotted from both sections *before*
any default is applied, a value given in either section wins, both sections are
then written with the same resolved value, and a genuine disagreement **raises**
rather than silently letting `training` win. Router noise became a three-valued
mode — `none` (canonical) | `fixed_annealed` | `trainable` (ablation D) — where
the canonical setting creates no noise parameter at all, so its absence is
verifiable in the `state_dict`; a parameter pinned to 0.0 would still sit in the
optimizer and still have to be reported.

**Why the gate tests E = 1, 6 *and* 7.** Everything hard-coded to 7 is
accidentally correct at E = 7, and a 7-wide head over 6 families produces no error
message at E = 6. Only a sweep across all three widths distinguishes "derives its
width" from "happens to match". E = 1 additionally covers the MoR baseline, where
routing does not exist and normalized entropy, per-expert recall and cosine
similarity must all report `N/A` rather than the sentinel values that would rank
MoR as maximally collapsed.

**The one real regression, fixed rather than waived.** Narrowing `cls_head` from 7
to 6 changes how many draws the initialiser takes from the RNG stream, so every
parameter created afterwards — including the routers — gets different values, a
different set of experts wins the argmax, and Phase 3's T3.4 dropped to 27/29. That
check demanded a non-zero gradient on all 24 per-expert halt parameters, which is
an assertion that top-1 dispatch is *dense* — the opposite of the canonical
design. A seed sweep confirmed the old pass was luck: at B = 4, S = 3 the number
of expert slots left unused across the two blocks is 4, 3, 3, 1, 3, 1 for seeds
0–5. The criterion, not the code, was wrong, and it was restated as three claims
that do not depend on that luck: on a batch large enough that all six experts *are*
dispatched, all 24 halt parameters must be live; a dispatched expert's halt head
must always be on the autograd graph; and a gradient must never appear for an
expert that was not dispatched to. Dispatch is read from `MoEBlock`'s own
`expert_idx` output through a forward hook, so the test measures the routing
decision the model made instead of guessing at it. The pre-Phase-3 defect this
check exists to catch — all 24 gradients `None` because the halting term was
computed from a bool tensor — is still caught, now on the fully-dispatched batch.

**Verification.** `code/test_phase5_dimensions.py` (Gate 4) — **62/62**. Headline
measurements: heads, router and expert list all width `E` at E = 1/6/7;
`step_cls_out` reshaping `(4,3,E) -> (12,E)` at every width; the confusion
diagonal fraction equal to `mean(pred == oracle)` equal to the reported scalar in
float64; an unknown op and an unknown family label both raising; `data`-only 0.5
surviving the defaults and a 0.3-vs-0.7 conflict raising; zero noise keys in the
canonical `state_dict` against exactly one per block in the trainable ablation;
canonical dispatch identical for 64/64 tokens across two training-mode passes
where the noisy variant moved 9/64; and the Gate 0 proxy guard refusing a
canonical claim for either `router_noise="trainable"` or
`routing_mode="dense_blend"` — that last check fires even while
`canonical_spec.json` is still unfrozen, because those two are architectural
choices rather than tuning. Full regression after the change: Gate 1
`audit_leakage.py` 8/8, `test_phase2_routing.py` 19/19, Gate 2
`test_phase3_halting.py` 31/31, Gate 3 `test_phase4_balance.py` 27/27,
`verify_pipeline.py` 35/35, and `smoke_test.py` now printing `cls_out (4,6)` /
`step_cls_out (4,7,6)` — 7 steps, 6 experts, the two 7s finally distinguished.

**What deliberately did *not* change.** `max_depth` and `max_steps` are still 7.
They are the recursion budget and the number of steps per program, a different
quantity that happens to share the value with the old expert count; "fixing" them
to 6 would silently shorten every program. `canonical_spec.json` remains unfrozen
and the `pure_act` versus `supervised_curriculum` decision remains open, both by
explicit instruction.

**If a dimension or a config value looks wrong, open these in order.** Head and
router widths, the noise modes and `current_router_noise_scale()`:
[code/more/model.py](code/more/model.py) — the constants `CANONICAL_ROUTING_MODE`,
`CANONICAL_ROUTER_NOISE` and `ROUTER_NOISE_MODES` live at the top and are imported
everywhere else rather than restated. Defaults, the `subset_*` reconciliation and
`CANONICAL_NUM_EXPERTS`: [code/more/config.py](code/more/config.py). Label
derivation: `expert_labels(num_experts)` in
[code/more/families.py](code/more/families.py) — every consumer of a confusion
matrix or a per-expert metric key must call this, and reaching for
`EXPERT_FAMILY_LABELS` instead is the defect Gate 4 exists to catch, which is why
both are exported from [code/more/__init__.py](code/more/__init__.py). Model
construction, the `resolved_routing_mode` / `resolved_router_noise` provenance
fields and the noise-penalty gating: [code/more/engine.py](code/more/engine.py).
Refusal of a non-canonical routing path: `assert_not_silent_proxy` check 6 in
[code/more/run_context.py](code/more/run_context.py). Declared expert count:
`model.num_experts` in [code/config.json](code/config.json), now 6. Per-check
evidence is in [TASKS.md](TASKS.md) Phase 5.


---

## T6.1 (Phase 6) — the seed became the only thing a rerun changes

**What was wrong.** The repository claimed seeded experiments and did not have
them. The single seeding call anywhere in the training path was the generator
handed to `random_split` for the data subset. Weight initialisation, dropout
masks, the DataLoader shuffle order, cuDNN's algorithm choice and every CUDA
kernel's RNG were all left on whatever state the interpreter happened to be in.
Two runs launched with the same `--seed` therefore started from different weights
and saw batches in a different order, and `runs/` directory names read `seedNA`.
That matters beyond tidiness: `updated_rules.md` §5 asks for mean ± std over
seeds 42-46, and a spread is only attributable to the seed if the seed is the
only input that varied. It also makes every earlier "the number moved" claim
unfalsifiable, because a rerun could not be repeated.

**Methodology.** One module, one entry point, everything else imports it:
[code/more/seeding.py](code/more/seeding.py).

- `seed_everything(seed, deterministic=True, strict=False)` seeds `random`,
  `numpy`, the torch CPU generator and all CUDA generators, sets
  `cudnn.deterministic`, clears `cudnn.benchmark`, turns on
  `torch.use_deterministic_algorithms`, and exports
  `CUBLAS_WORKSPACE_CONFIG=:4096:8` — the last one *before* the CUDA context is
  created, because cuBLAS reads it at context creation and setting it afterwards
  does nothing.
- `make_generator(seed, "dataloader_shuffle")` gives shuffling its own
  sub-stream instead of drawing from the global torch RNG. This is a direct
  lesson from Phase 5: narrowing `cls_head` from 7 to 6 outputs changed how many
  initialiser draws happened, which shifted every later draw and flipped which
  experts the router picked. Batch order must not be hostage to that.
- `seed_worker` is a module-level function, not a closure, so it survives
  pickling — on Windows the DataLoader spawn start method would otherwise fail
  the moment `num_workers` is raised above 0. It is wired unconditionally even
  though the loaders currently run with 0 workers.
- `apply_seeding(cfg)` resolves the seed with an explicit precedence —
  `--seed` > `training.seed` > documented default 42 — seeds the process, and
  writes what actually happened into `provenance`.

`engine.train()` calls `apply_seeding` as its **first** action, before the
dataset, the model or the optimiser exist, and immediately flushes
`resolved_config.json` so even a crashed run records how it was seeded. Anything
seeded after model construction would leave weight initialisation unreproducible
while the log still advertised a seed.

**Two decisions worth knowing about.**

*The declared seed and the resolved seed are different fields.* A run launched
without `--seed` is now bit-reproducible under the default 42, but
`provenance.seed` stays `null`; the 42 is recorded separately as
`resolved_seed` with `seed_source = "default"`. Back-filling the declared field
would have been the tidier-looking choice and would have quietly disarmed the
Gate 0 proxy guard, whose canonical check is precisely "was a seed declared, and
is it in the frozen set". The test asserts the guard still refuses such a run.

*Determinism is `warn_only`, not `strict`.* The top-1 dispatch path uses
scatter/index writes whose CUDA kernels have no deterministic implementation in
torch 2.5. `plan.md` §8.1 says to document the exception rather than pretend
exact determinism, so a non-deterministic op logs a warning (forced through
`warnings.filterwarnings("always", ...)` so it is not deduplicated away after the
first batch, and captured in the run's `stdout.log` via the existing stderr tee)
and the list is written to `provenance.determinism_exceptions`. It is never an
empty list: with no exceptions it reads `["none"]`, per the no-sentinels rule.
Every determinism field in the report is read back out of torch rather than
echoed from the request, so a setting that failed to apply shows as `False`
instead of as a claim. `training.deterministic_strict = true` is available if a
future run wants a hard failure instead.

**A false-clean record the tests could not have caught, found by running it.**
The first real 1-epoch CLI run wrote `determinism_exceptions = ["none"]` into
provenance while its own `stdout.log` carried **281** copies of
`_histc_cuda does not have a deterministic implementation`. Both halves were
wrong. The record was false because `seed_everything`'s notes can structurally
only report what failed *while seeding*, and a kernel without a deterministic
implementation does not announce itself until it is first **called**, somewhere in
the middle of training — so the honest-looking `["none"]` was written before the
evidence existed. And the log was unreadable because forcing `always` through the
warning filter meant torch re-warned on every single call.

Both are fixed by one mechanism, in `seeding.py`: a wrapper around
`warnings.showwarning` that extracts the op name from the warning text, records it
in a set, prints the *first* occurrence and swallows the rest. The set is read at
the end of the run and written to `provenance.nondeterministic_ops_observed` and
`metrics.json`, with a `[Seed]` line in the log when it is non-empty. One line per
distinct op is documentation; 281 copies is noise that buries everything else.
Re-measured on `runs/t61_nondet_check__seed42__42b858b3`: **1** warning line
instead of 281, `nondeterministic_ops_observed = ["_histc_cuda"]`, and an epoch
row identical to the pre-fix run — the recorder observes, it does not perturb.
Worth knowing *which* op it is: `_histc_cuda` comes from `wandb.watch`'s gradient
histograms, i.e. the logging path, not the loss or routing math. So the two
fields answer different questions and both are needed: `determinism_exceptions` =
"what could not be configured", `nondeterministic_ops_observed` = "what actually
ran non-deterministically".

**Verification** — [code/test_phase6_seeding.py](code/test_phase6_seeding.py),
**61/61 PASS** with CUDA present. The structure of the test matters as much as
the count: for every "same seed reproduces X" check there is a paired "seed 43
changes X", because a seeding function that seeds nothing at all also produces
identical reruns on a deterministic machine. Highlights:

- all five stream classes reproduce at seed 42 and *all five* differ at seed 43;
- parameter digest identical at 42, different at 43 (so init is covered, not
  just data order);
- a full forward+backward in train mode is bitwise identical (loss
  `1.1138908863067627`, |grad| `1991.3999771339586`, 137 routing slots);
- shuffle order is reproducible, seed-sensitive, and unchanged after 1000 global
  RNG draws are burned in between — the Phase-5 coupling is closed;
- five synthetic "no deterministic implementation" warnings are recorded by op
  name and print **once**, not five times (the 281-line regression above);
- two real 2-epoch engine runs at seed 42 produced a byte-identical first-epoch
  results row, `1  0.134741  0.062896  0.7943  2.21  0.0030  0.0309  0.0241`
  (train task loss, val loss, normalized entropy, avg depth, mean/max cosine,
  routing accuracy), against `1  0.137417  0.049722  0.8968  2.48  -0.0024
  0.0231  0.0964` at seed 43.

No regression: Gate 1 8/8, Phase 2 19/19, Gate 2 31/31, Gate 3 27/27, Gate 4
62/62, `verify_pipeline.py` 35/35, `smoke_test.py` passing.

**What deliberately did not change.** No architectural change of any kind (the
model file is untouched). The fixed train/val split generator stays pinned at 42
so the split does not move with the run seed. `subset_seed` remains tied to
`--seed` by `resolve_overrides`, as before — it is inert for canonical runs,
which use the full dataset. The ACT variant choice and `canonical_spec.json`
remain open by instruction.

**If a rerun does not reproduce, open these in order.** First
`runs/<id>/resolved_config.json` → `provenance`: `resolved_seed` and
`seed_source` say which seed the RNGs actually got (not `seed`, which only says
whether one was declared), and `determinism_exceptions` (what could not be
configured), `nondeterministic_ops_observed` (what actually ran
non-deterministically) plus `cublas_workspace_config` say what could not be made
deterministic. Then
[code/more/seeding.py](code/more/seeding.py) for what is seeded and the
`_PURPOSE_OFFSETS` sub-stream map. Then the call site,
[code/more/engine.py](code/more/engine.py) line ~93 — if anything that draws
random numbers is ever constructed *above* that line, it is outside the seed's
reach, and that is the first thing to check. DataLoader wiring
(`generator=` / `worker_init_fn=`) is ~90 lines below it. If losses match on CPU
but not on GPU, suspect a warn_only exception: read
`provenance.nondeterministic_ops_observed` (or `stdout.log` for
"does not have a deterministic implementation") and set
`training.deterministic_strict = true` to convert it into a hard error. Per-check
evidence is in [TASKS.md](TASKS.md) Phase 6.

---

## T6.2 / T6.3 (Phase 6) — routing quality stopped being measured by the expert's index number

**What was wrong.** Routing accuracy was the fraction of tokens whose predicted
expert index equalled the oracle's index. That answers "did the router pick
expert *number* 3?" — but an expert index carries no semantic identity. A model
that had perfectly discovered the six operation families and merely numbered them
differently would score 0.0, and we would have reported "routing failed" about a
model that had in fact solved the problem. The converse is just as bad: a
favourable index alignment would read as specialization. Separately, T6.2's
authoritative-accuracy requirement was only half-satisfied — the accuracy scalar
and the confusion matrix agreed by convention, not by construction.

**What changed.** `more/metrics.py` gained
`permutation_invariant_routing_metrics(confusion, num_experts)`, which reports,
from the *same* confusion matrix the accuracy comes from: raw accuracy,
Hungarian-matched accuracy (the best possible relabelling of experts), the
recovered assignment itself, AMI (chance-corrected agreement between the router's
partition and the oracle's), cluster purity, and per-family
precision/recall/F1/support. The engine writes the whole block to
`metrics.json`, adds `routing_hungarian_acc` and `routing_ami` columns to
`results.tsv`, and logs the scalars to W&B. T6.2 was closed by making both
implementations delegate to one function, so the scalar *is* the confusion
diagonal fraction rather than merely equalling it.

`more_env` has neither scipy nor sklearn, so both non-trivial algorithms are
local: exact assignment by subset DP (`O(E²·2^E)`, which refuses `n > 15` rather
than silently degrading to a greedy approximation that would *overstate* nothing
but *understate* the match), and AMI's chance correction by hypergeometric
expectation computed in log space over a cumulative log-factorial table.

**How to read the numbers — this is the part that matters for the paper.** The
three cases are written into `metrics.json` as a `note`, because the same pair of
numbers means different things:

- raw ≈ matched → the *index itself* is supervised. This is the current canonical
  setting (`loss_weights.step_routing = 0.5`), so it is the expected reading, not
  a finding. Do not present it as evidence of discovered structure.
- raw ≪ matched → a real partition exists under relabelling. This is the
  interesting case, and the only one that supports an unsupervised
  specialization claim.
- both low with AMI ≈ 0 → no partition. Purity may still look respectable at
  ≈ 1/E because purity is **not** chance-corrected; that is why AMI and purity
  are always reported side by side.

**Verification** — [code/test_phase6_routing_metrics.py](code/test_phase6_routing_metrics.py),
**42/42 PASS**. The suite is built so that a metric which quietly did nothing
would fail:

- the criterion from `plan.md` §8.3 verbatim: a cyclic permutation of six
  experts gives `hungarian_accuracy = 1.0` against `raw_accuracy = 0.0`, and the
  recovered assignment is the planted `[1,2,3,4,5,0]`. Every permutation-invariant
  quantity is bit-identical to the identity-labelled control — which is what
  permutation invariance *means* operationally;
- the subset DP matches exhaustive enumeration over all `n!` permutations on 40
  random matrices (max |Δ| < 1e-9), and refuses `n > 15` and non-square input;
- the closed-form AMI chance correction matches literal enumeration of the
  permutation model over 720 permutations (agreement to 1e-10);
- `hungarian ≥ raw` on 60 random tables (an inequality that must hold by
  definition, so a sign error anywhere shows up here);
- an independent (product) table gives MI exactly 0 and AMI ≤ 0 while purity
  stays high — the chance-correction is doing work;
- total collapse (all tokens to one expert) gives AMI 0 and
  `hungarian = purity = 1/E`, not a flattering score;
- a hand-built imbalanced 3×3 table checks recall 8/10, precision 8/14, the F1
  harmonic mean, per-family support, `N/A` for a zero-support family, and
  `macro_recall = 0.75` against `raw = 22/30` — deliberately made *unequal*, so
  the macro average cannot pass by coinciding with the raw figure;
- `E = 1` and an all-zero confusion return `None` for every quantity and `{}` for
  `per_family`;
- **T6.2:** over 200 random token streams the confusion diagonal fraction, the
  engine's `mean(pred == oracle)` and the block's `raw_accuracy` differ by
  **max 0.000e+00**;
- no expert label contains `/` (W&B would turn `E1 ADD/SUB` into spurious
  nesting).

*On a real run* (`runs/t63_pim_check__seed42__b19853cd`, 1 epoch MoRE):
`route_acc=1.0000 route_hung=1.0000 route_ami=1.0000`, the full block in
`metrics.json`, 24 clean `val/routing_*` W&B keys — and train/val loss
**byte-identical** to the pre-T6.3 run (0.081314 / 0.073893), so the metric layer
observes without perturbing the optimization.

**Trap hit here, worth remembering.** The per-family F1 used
`if prec and rec: f1 = ...`, i.e. truthiness. A *measured* precision of 0.0 is
falsy, so an expert that got everything wrong was reported as `N/A` — the one
case where the reader most needs a number. F1 is now `None` only when precision
or recall is itself *undefined* (no oracle tokens for the family, or nothing
predicted into that index); when both are defined and zero, `0.0` is a
measurement. The same distinction applies anywhere else a metric is skipped:
check `is None`, never truthiness.

**If routing numbers look wrong, open these in order.**
[code/more/metrics.py](code/more/metrics.py) →
`permutation_invariant_routing_metrics` (and `_hungarian_max_assignment` /
`_adjusted_mutual_info` beneath it) for the computation;
[code/more/engine.py](code/more/engine.py) validation loop for the confusion
matrix that feeds it — if the matrix is accumulated over a different token set
than the accuracy, T6.2's guarantee is gone and everything downstream inherits
the error. `metrics.evaluate_paper_metrics` still exists and is marked UNUSED
in-source: if you fix a routing metric there, nothing changes in any run.
Per-check evidence in [TASKS.md](TASKS.md) Phase 6.

---

## T6.6 / T6.7 + GATE 5 (Phase 6 closure) — every run can now say what it was

**Why this mattered.** A number in a paper is worthless if you cannot say which
code, config, data split and seed produced it. Two specific failure modes were
open: a provenance field that exists but is `None` (reported, unusable, easy to
miss in a wide W&B table), and an **ablation whose provenance is
indistinguishable from canonical** — which is exactly how a variant ends up in a
headline table.

**What changed.**

1. **Run naming became a function, not a typed string** (`more/config.py`).
   `ARCH_DISPLAY` holds the *paper* spelling (`MoE`/`MoR`/`MoRE`), deliberately
   separate from the lowercase config enum, so the two can never drift.
   `canonical_run_name(arch, seed)` produces `phaseB_MoRE_seed42` and **raises**
   on an unknown architecture instead of returning a plausible-looking name.
   `stamp_seed_into_run_name(cfg)` is called from `cli.py` *after*
   `resolve_overrides` — only at that point are both the seed and the run name
   final. It is idempotent, and it *extends* a custom `--run_name` rather than
   replacing it, so a diagnostic run keeps its identity
   (`t63_pim_check` becomes `t63_pim_check_seed42`). With no seed declared the
   name does not claim one.

2. **`variant` is derived, never declared** (`more/config.py:resolve_variant`).
   An unmodified config resolves to `canonical`; a dense-routing, router-noise,
   fixed-depth, oracle-routing or supervised-curriculum run gets a label naming
   the deviation, and multiple deviations compose into one sorted, stable string
   (`fixed_depth+router_noise_gaussian`). A curriculum *flag* with **zero
   weight** is correctly not a variant, because the term contributes nothing to
   the loss. This is the structural fix for "ablation reported as canonical by
   omission": there is no way to omit the field, because nobody writes it.

3. **`ffn_mult` became a real config field** threaded
   `MoREModel` to `MoREWrapper` to `MoEBlock`, replacing a hard-coded `* 4`. The
   FFN width is now recorded rather than assumed, and Phase 9's parameter
   matching (MoE/MoRE 6,358,553 vs MoR 1,098,259) has a lever to pull. Default 4
   reproduces every earlier run exactly — `total_params` is still
   **6,358,553**, which is how we know the refactor is behaviour-identical and
   not merely plausible.

4. **The provenance block was completed** across `cli.py`, `run_context.py` and
   `engine.py`, adding architecture, variant, run_name, dataset_version,
   train_split_version, ffn_mult, routing_supervision_enabled/weight,
   `router_noise_scale`, halt_target_mode and total_params to both the W&B
   config and `resolved_config.json`. `router_noise_scale` is *resolved*: under
   the canonical `router_noise = "none"` it is written `0.0`, not the init value
   it would carry if noise were on.

**Verification** — [code/test_phase6_provenance.py](code/test_phase6_provenance.py),
**32/32 PASS**. The provenance half is checked against a **real CLI run**, not a
constructed dict, because the fields are assembled in three different files and a
unit test of any one of them would not catch a field lost between them:
`python train.py --architecture more --epochs 1 --seed 44 --run_name
t67_provenance_check` exits 0, and all 27 `updated_rules.md` §9 fields resolved
from `resolved_config.json` give **`missing/None: []`**. Also confirmed: declared
seed 44 == `resolved_seed` 44 (the seed reached the RNGs, not just the log),
`variant == "canonical"`, `ffn_mult == 4`, `router_noise_scale == 0.0`, the T6.3
block present in the same run's `metrics.json`, and no provenance value equal to
the *string* `"None"`.

**Two latent defects were found here that would otherwise have surfaced only in
the final matrix.**

- `config_hash` hashed the whole config including the `logging` block — and the
  run name lives in `logging`. So `phaseB_MoRE_seed42` and `phaseB_MoRE_seed43`
  produced *different* hashes, and the exporter (which by rule refuses to mix
  config hashes) would have refused to average the five seeds of one experiment.
  `run_context.py:config_hash` now strips `logging` as well as `provenance`;
  tested both ways — two seeds share a hash, while an `lr` or `ffn_mult` change
  still changes it.
- `_allocate_id` appended the seed a second time once T6.6 put it in the run
  name, giving `t67_provenance_check_seed44__seed44__c3571d3a`. Now suppressed
  when the base already carries the seed, with a "seed appears once" assertion.
  Note the test selects the run directory by **mtime**, not alphabetically —
  alphabetical ordering picked up the stale doubled directory and would have kept
  passing against it.

**GATE 5 — CLOSED, [code/test_gate5.py](code/test_gate5.py) 34/34 PASS.** The six
criteria are re-asserted in one file against a live model and a real run
directory, so "Gate 5 passes" is a claim about the code as it stands now rather
than about six test files that passed at different times:

1. provenance complete and non-None, with determinism reported as two distinct
   facts — what could not be configured vs. what actually ran
   non-deterministically;
2. all ten loss components separate, `val/task_loss` primary and distinct from
   the weighted total, entropy normalized by `log(E)`, cosine mean **and** max,
   depth as *allocation error* with no key containing "efficiency", early **and**
   forced exit rates, and the permutation-invariant block's `raw_accuracy`
   matching `val/routing_accuracy` to under 1e-6;
3. every width measured equal to `num_experts = 6`, with the op embedding sized
   by `NUM_OP_TYPES = 16` instead — the op-to-expert map is what routing must
   discover, so it must not be handed to the model as a matching dimension;
4. `top1_sparse` with measured `evals_per_token == 1.0`, i.e. dispatch rather
   than dense blending, noise off, and every token index in range;
5. halt gradients, below;
6. no `-1`, no NaN, undefined values as `"N/A"`, and a single-expert (MoR)
   confusion returning `None` for every routing quantity rather than a
   comparable-looking `0.0`.

**The halt-gradient check, and why its scope is the interesting part.** This is
the defect Phase 3 existed to fix: halted states were copied into an output buffer
by a discrete choice, so every halt parameter had `grad is None` after a full
backward. The first version of this check failed 2/12 — and that was the *test*
being wrong, not the model. Under Top-1 dispatch an expert that no token selected
is never called, so its halt head having no gradient is arithmetic. The check now
runs over the experts that actually received tokens (taken from the forward's
returned first-route indices) **and separately asserts coverage**
(`len(used) == 6`), so it cannot go silently weak by testing one expert. Then the
sharper form: `ponder_cost.backward()` **alone** reaches every active halt head,
meaning the recursion cost is a term the halting policy is optimizing against,
not a number printed next to the model. Anyone tightening this check later should
keep both halves — dropping the coverage assertion makes it vacuous, and dropping
the used-expert scoping makes it fail on correct code, which trains readers to
ignore it.

**Full regression at closure:** Gate 1 8/8, Phase 2 19/19, Gate 2 31/31, Phase 4
27/27, Phase 5 62/62, T6.1 61/61, T6.3 42/42, T6.6/T6.7 32/32, Gate 5 34/34,
`verify_pipeline.py` 35/35, `smoke_test.py` PASS.

**If a run cannot be reproduced or identified, open these in order.**
`runs/<id>/resolved_config.json` and its `provenance` block first: it names the
git commit, config hash, dataset and split versions, resolved seed and `variant`.
If two runs of the same experiment will not average, check `config_hash` —
[code/more/run_context.py](code/more/run_context.py) `config_hash` must exclude
both `logging` and `provenance`, or seeds stop being comparable. If a run's label
or directory looks wrong, [code/more/config.py](code/more/config.py)
`canonical_run_name` / `stamp_seed_into_run_name`, then
`run_context._allocate_id`. If a run is labelled `canonical` that should not be,
`resolve_variant` in the same file is the only place that decision is made.
Provenance assembly itself is split between `RunContext.create` and the
`wandb_config` block in [code/more/engine.py](code/more/engine.py) — a field
present in one and missing from the other is the failure mode to look for.
Per-check evidence in [TASKS.md](TASKS.md) Phase 6.


---

## The controlled ACT decision experiment — canonical halting is `pure_act`, and unsupervised depth allocation does not track the curriculum

**The question.** `canonical_spec.json` could not be frozen without settling
whether canonical MoRE halts under pure unsupervised ACT or under the supervised
operation-complexity curriculum. The choice changes what the depth numbers in the
paper are *allowed to mean*: a supervised run was told the target depths, so its
allocation error is not evidence the architecture found anything.

**Method.** [automated/act_decision.py](automated/act_decision.py). Two arms,
three seeds each (42/43/44), everything matched except the halting objective:
`architecture=more`, 20 epochs, `blocks=1`, `batch_size=768`, same lr, d_model,
weight decay, `routing_balance` and `step_routing`, same dataset and split. The
only difference is `--halting_supervision --halting_supervision_weight 0.1`. Six
runs, ~7 min each.

**The decision rule is written at the top of the driver, before the runs.** That
placement is the point — deciding after looking at the table is tuning, not an
experiment (`CLAUDE.md` §6). The rule, in full:

1. Primary criterion is validation task loss, mean ± std. `supervised_curriculum`
   is selected only if it beats `pure_act` by more than seed noise.
2. Depth allocation error is reported but is **not** a selection criterion. The
   supervised arm is handed `families.OP_TARGET_DEPTH`, so it wins that column by
   construction; selecting on it would be selecting the variant that was told the
   answer, and would simultaneously destroy the column's value as evidence.
3. `pure_act` wins ties, because only an unsupervised run supports the claim in
   `updated_objective.md`. A tie is a reason to keep the honest variant.
4. If `pure_act` shows no depth differentiation, that is Outcome C and gets
   reported, not fixed by switching supervision on.

The driver also refuses to report at all if any run's
`provenance.halting_mode` disagrees with the arm it was launched as, or if
`resolved_seed` does not match the requested seed. A silently ignored flag would
otherwise present itself as "supervision makes no difference" — a conclusion where
the truth is a bug.

**Result.**

| MoRE, mean ± std over 3 seeds | val task loss | avg depth | alloc err (abs) | per-op depth spread | corr. with `OP_TARGET_DEPTH` |
|---|---|---|---|---|---|
| `pure_act` | **0.063676 ± 0.000283** | 2.292 | 1.025 ± 0.059 | 0.452 ± 0.044 | **+0.204 ± 0.554** |
| `supervised_curriculum` | 0.063953 ± 0.000190 | 2.057 | **0.445 ± 0.012** | 0.983 ± 0.084 | +0.957 ± 0.027 |

Gap on the primary criterion: `−0.000277`, against a largest seed std of
`0.000283`. Inside noise, so rule 3 decides: **canonical halting mode =
`pure_act`**. Written to `automated/act_decision_result.json`.

**The finding that matters more than the decision.** Under `pure_act` the
per-operation depths do differ — spread 0.452 is not a flat allocation — but their
correlation with the curriculum across seeds 42/43/44 is `+0.608 / −0.428 /
+0.431`. **The sign flips between seeds.** So unsupervised ACT learns a
non-trivial but essentially *arbitrary* depth partition, not the
operation-complexity curriculum. That is **Outcome C for the adaptive-computation
claim** (`CLAUDE.md` §8) and is to be reported as such. The contrast is the useful
result: `r = +0.957` when the targets are supplied, `+0.204` when they are not.

**A gap in our own pre-registered rule, recorded rather than quietly patched.**
Rule 4 tested depth *spread*. Spread is necessary but not sufficient: a model can
differentiate depth strongly and still point it at the wrong operations, and the
spread number cannot tell the two cases apart — it passed here (0.452 ≫ the 0.10
threshold) on a model whose allocation is arbitrary. The correlation diagnostic
was therefore added to the driver **after** the runs, and both the docstring and
the printed output say so. It does not change the decision, which the primary
criterion and the tie rule had already settled. Any future pre-registered depth
criterion should test alignment, not just spread.

**What this means for the paper.** The honest framing is that MoRE's recursion
depth is adaptive in the mechanical sense — tokens exit at different steps, the
halt heads are trained (Gate 2 / Gate 5 criterion 5), and the ponder cost reaches
them — but the *allocation* it settles on is seed-dependent and not the designed
curriculum. Never write "discovers intrinsic mathematical complexity"; on this
evidence, do not write "allocates depth in accordance with the predefined
curriculum" either, because at `r = +0.204 ± 0.554` it does not.

**Where to look if this needs revisiting.**
[automated/act_decision.py](automated/act_decision.py) holds the rule, the matched
protocol and the report; re-run it with `--report-only` to re-derive the table from
the existing `runs/actdec_*` directories without retraining. The mechanism itself
is `MoREWrapper.forward` in [code/more/model.py](code/more/model.py) (the ACT
recursion and the `state_t * w_t` convex combination) and the two halting terms in
[code/more/engine.py](code/more/engine.py) — `train/ponder_cost` and
`train/halting_supervision_loss`, never summed. `families.OP_TARGET_DEPTH` is the
hand-written curriculum the correlation is measured against; it is a design
choice, not data, which is why a low correlation is a fact about the model rather
than a fact about mathematics. If a longer canonical run changes this picture,
the number to recompute first is the per-seed correlation, not the allocation
error.


---

## T6.8 — the per-family table became permutation-invariant, so an unsupervised router can finally be read

**The problem.** T6.3 had already established the rule that expert indices carry
no semantic identity, and it added Hungarian accuracy, AMI and purity — all
permutation-invariant scalars. But the *per-family* breakdown (precision, recall,
F1 per operation family) was still computed under the **identity** mapping:
family `E3_MOD_POW` was scored against expert index 3. For a supervised router
that is correct, because supervision pins the numbering. For an unsupervised
router it is meaningless — the router may have learned a clean MOD/POW expert and
numbered it 5, in which case the identity table reports recall 0.0 for a family
the model actually separates. That is exactly the number a reader would quote as
"MoRE fails on MOD/POW", and it would be an artefact of numbering.

**What changed.** The metric layer now emits two per-family tables side by side.
`per_family` keeps the identity mapping and is explicitly labelled as
interpretable only when `step_routing` supervised the index. `per_family_matched`
applies `hungarian_assignment` **first** and then computes precision / recall /
F1 / support per family, and records which expert each family was matched to
(`matched_expert`). A scalar `matched_macro_recall` accompanies it, alongside the
pre-existing identity `macro_recall`. The same pair is flattened into the W&B key
space as `val/routing_{precision,recall,f1}_matched/<FAMILY>` beside the
unmatched keys, and the nested block's `note` field states in words which table
to report for which regime.

**Why both are kept rather than replacing one.** The gap between the two tables is
itself the evidence. Identity macro-recall 0.126 against matched macro-recall
0.418 on the same run *is* the finding "a partition exists, arbitrarily
numbered". Deleting the identity table would remove the contrast; deleting the
matched table (the state before T6.8) makes the finding unreportable. Regression
tests pin invariance directly: relabelling the predicted expert indices by an
arbitrary permutation must leave every matched quantity unchanged while the
identity quantities move.

**Where to look if this needs revisiting.** The computation is in
[code/more/metrics.py](code/more/metrics.py) — the Hungarian assignment is solved
there (no scipy in this environment, so the assignment is solved directly) and
consumed by both tables. The writer that flattens the nested block into
`metrics.json` and W&B keys is in
[code/more/engine.py](code/more/engine.py). The invariance tests are in
[code/test_phase6_routing_metrics.py](code/test_phase6_routing_metrics.py).
**Practical consequence for comparisons:** any run produced before T6.8 has
`per_family` but no `per_family_matched` and no `matched_macro_recall`, so it
cannot be placed in a table beside a current run — the two arms would sit on
different metric layers. This is the reason the T10.H family-supervision ablation
retrains its own baseline arm instead of reusing the T8.0b runs.

---

## T7.2 — one command now runs all five correctness gates, and an empty gate fails

**The problem.** Gates 1–5 existed as eight separate test files that had to be
remembered and run by hand. Nothing tied them together, so "the gates pass" was
an assertion about whatever someone last ran, and a suite that silently stopped
asserting anything would look identical to a suite that passed.

**What changed.** `code/run_correctness_suite.py` runs all eight suites grouped
into the five gates, counts the `[PASS]` / `[FAIL]` / `[SKIP]` markers each one
emits, prints a per-gate table, and exits non-zero if any gate fails. Current
state: **347 checks, 347 pass, 0 fail, exit 0.**

Three design decisions, each forced by a defect rather than chosen for style.
*Subprocess per suite*, because the suites mutate global RNG state and
`torch.backends` flags; imported into one process they contaminate each other and
the failure looks like a model bug. *Fixed, non-alphabetical order*, because
`test_gate5.py` reads the run directory that `test_phase6_provenance.py`
produces — alphabetical ordering happens to work today and would break silently
if a file were renamed. *A gate that counted zero assertions FAILS* with verdict
`EMPTY`, never passes.

That third rule exists because the runner's own first version reported "0 checks
… PASS" for gates 1–4: its marker regex was anchored at column 0 while the Phase
2–5 suites indent their markers, so it matched nothing and called that success.
This is the same defect class as reporting a sentinel as a measurement — an
absence of evidence rendered as evidence — and it is the one the whole T6.4/T6.5
line of work was about.

**Where to look if this needs revisiting.**
[code/run_correctness_suite.py](code/run_correctness_suite.py); the `GATES` dict
is the gate→file mapping and the `MARKER` regex is the counter. If a new suite is
added it must be registered in `GATES` or it will not run, and if a suite changes
its marker format the count will drop to zero and the gate will fail loudly
rather than pass quietly. Run it before ticking any gate box and before any
canonical launch.

---

## T8.0b — canonical routing supervision is OFF, and the router does partition the operation space without it

**The decision that had to be made before the spec could freeze.** `config.json`
shipped `loss_weights.step_routing = 0.5`, and that term is built as
`0.01 * oracle_routing_ce + 0.3 * step_cls_loss` where `oracle_routing_ce` is a
cross-entropy **directly on the router logits** against the oracle expert index.
Every routing number measured before this task was therefore measured with the
router being told the answer: raw accuracy, Hungarian accuracy, AMI and purity all
sat at ~1.0 and the recovered assignment was the identity permutation. Freezing
the spec at 0.5 would have frozen a headline specialization claim that measures
the supervision signal rather than the architecture.

**Method.** Two arms, `step_routing` 0.0 vs 0.5, everything else matched — MoRE,
`pure_act` halting (the T8.0a selection, so the two decisions are not confounded),
20 epochs, 1 block, batch 768, seeds 42/43/44, same dataset and splits. The
reading rules were fixed in the driver docstring **before** any run, including
what "no partition" and "a real partition under relabelling" would each look like
numerically, and including a rule that a collapsed router (everything to one
expert) is a different failure from an arbitrarily-numbered partition and must not
be reported as the same thing.

**Result.** Removing the oracle **costs 0.000155** val task loss against a seed
std of 0.000428 — inside noise, with the unsupervised arm marginally ahead. So
supervision buys nothing on the primary metric while making AMI 0.9992 true by
construction. In the unsupervised arm the Hungarian assignment differs on every
seed (`[4,0,5,1,3,2]`, `[5,3,4,1,0,2]`, `[0,1,4,5,3,2]`) while Hungarian accuracy
stays at **0.5147 ± 0.0047** and AMI at 0.4940 against chance 0.167 — a real
partition recovered under relabelling, which is the publishable form of the
result. `step_routing = 0.5` is retained as the labelled T10.E oracle-routing
ablation and may never share a table with canonical runs.

**The honesty constraint this created.** `step_routing = 0.0` removes the *direct*
routing supervision. It does **not** make the system unsupervised:
`loss_weights.family_cls = 0.5` still trains `cls_head` on the same oracle family
label, read off disk from the same hand-written manifest. So the claim must read
*"specialization emerges without direct routing supervision, under whole-program
family supervision"* — never as unsupervised expert discovery. T10.H exists to
measure how much of the 0.5147 that remaining term accounts for.

**Where to look if this needs revisiting.**
[automated/routing_supervision_decision.py](automated/routing_supervision_decision.py)
holds the pre-registered rules, the protocol and the report; `--report-only`
re-derives the table from the existing `runs/routedec_*` directories without
retraining, and the machine-readable summary is
`automated/routing_supervision_result.json`. The term itself is assembled in
[code/more/engine.py](code/more/engine.py) (`step_routing_loss`) and the oracle
index it trains against comes from `families.OP_TO_EXPERT` via
[code/more/data.py](code/more/data.py). If a routing number ever comes back at
~1.0 on a canonical run, check `provenance.routing_supervision_weight` first — at
0.0 that number would mean a label leak, not a result.

---

## T8.3 — the second-largest term in the objective stopped being an undeclared literal

**The problem.** `engine.py`'s loss assembly contained a bare `0.5 * cls_loss`.
That term — a 6-way cross-entropy on `cls_head(pooled)` against the whole-program
family label — was the second-largest contributor to `total_loss`, and it appeared
in **no** config file, **no** provenance block, **no** results table and **no**
ablation label. It could not be varied from the command line, could not be
reported, and could not be turned off to find out what it was doing. Worse,
`cls_loss` was never accumulated per epoch either, so it was not even logged as a
separate loss component — a direct violation of the rule that every loss term is
tracked separately and `total_loss` is never the headline number.

**Why it matters scientifically, not just tidiness.** That label is **external
annotation, not self-supervision**: `data/script.py` stamps a `"family"` string
into every JSONL record at generation time, `more/families.py:FAMILY_TO_IDX` is a
hand-written manifest, and `more/data.py` reads `rec.get("family")` straight off
disk. Proof that it is model-independent: the label histogram is byte-identical at
`num_experts = 6` and `num_experts = 1`. So an undeclared 0.5-weighted oracle
signal was shaping the shared trunk representation that the router reads, in every
run in the project's history, while the write-up would have described the routing
result as emerging without supervision.

**What changed.** The literal became `loss_weights.family_cls`, declared in
`config.json`, overridable by `--family_cls`, defaulted defensively in the engine
*into `cfg` itself* so the value reaches `resolved_config.json` rather than being
supplied silently at the point of use (the exact failure mode the literal had),
recorded in both provenance blocks as `family_cls_weight` and
`family_supervision_enabled`, pinned in `canonical_spec.json:enforced_fields` at
0.5, checked by the Gate 0 guard, accumulated per epoch as
`train/classification_loss`, and labelled by `resolve_variant` — `0.0` resolves to
`no_family_supervision`, any other non-canonical value to `family_cls_<v>`, so an
ablation run can never be tabled as canonical. The new term sits at the same
position inside the `lw["task"]` factor as the literal did, so at 0.5 the
objective is numerically unchanged.

**Where to look if this needs revisiting.** The assembly is in
[code/more/engine.py](code/more/engine.py) (`total_loss`), the default and the
variant labelling in [code/more/config.py](code/more/config.py), the CLI override
and its refuse-don't-ignore validation in [code/more/cli.py](code/more/cli.py),
the guard field in [code/more/run_context.py](code/more/run_context.py), and the
freeze — recorded as **INHERITED, not selected** — in
[code/canonical_spec.json](code/canonical_spec.json). If a future run's routing
metrics change unexpectedly, `provenance.family_cls_weight` is now the first field
to read; before T8.3 there was nothing to read.

---

## T8.1 / T8.2 — the canonical configuration is frozen, the parameter budget is matched to 0.12%, and the default run is finally canonical

**T8.1, the smoke.** Two epochs of each architecture at seed 42 on the full
dataset, with every metric read by hand out of `metrics.json` rather than off the
console. MoE 3,201,555 params / val task 0.071424; MoR 3,197,710 / 0.071318; MoRE
3,201,555 / 0.072134. No NaN, no numeric sentinel. All three report
`evals_per_token = 1.0` (sparse Top-1, not dense), `overflow_rate = 0.0`,
`routing_mode = top1_sparse`, `router_noise = none`, `halt_target_mode =
pure_act`, `variant = canonical`. MoR emits no routing quantity as a number and
every depth metric as one; MoE, at `max_depth 1`, emits every depth metric as
`"N/A"` because they are constants there. A re-run at an explicit `--seed 42`
reproduced `val_loss 0.071318` to all digits.

Against the trivial baselines computed on the real split: predict-train-mean MSE
0.080914, predict-median and predict-zero 0.081853. All three architectures clear
all three, so the criterion passes — **but only by ~12% at 2 epochs and ~21% at 20
epochs (R² ≈ 0.215 against predict-the-mean)**. That is a modest fraction of
target variance and must be stated plainly rather than presented as strong
regression performance. It also means the three architectures' task losses sit in
a narrow band, which is precisely why seed variance is reported beside every one
of them.

**Four defects the smoke caught, which is what a smoke is for.** MoR's flat
`val/routing_*` keys were **absent** from `metrics.json` instead of `"N/A"`, even
though the nested block writes `"N/A"` specifically so the field list is
architecture-independent — absence is a third convention and mixing three
conventions in one file breaks any consumer. `test_gate5.py`'s
`train/ponder_cost > 0.0` check raised `TypeError` whenever the newest
routing-bearing run was MoE (which correctly writes `"N/A"` at `max_depth 1`), so
the gate's verdict depended on which architecture happened to run last; the run
selector now requires both routing and depth. Check G5.2b accepted mere absence
where it must require the string. And the console epoch line printed MoR
`bal=1.0000` and MoE `halt=1.0000` while the artifact wrote `"N/A"` for both —
`1.0000` beside MoRE's `−0.6128` reads as "MoR is maximally imbalanced", so both
now route through the same `_na()` renderer the artifact uses. Correctness suite
after the fixes: 347/347, exit 0.

**T8.2, the freeze, and the parameter budget.** Two structural rules replaced
filling fields by hand: whatever `apply_architecture` varies per architecture
belongs in `architecture_variants` (a null there blocks only that architecture);
whatever is shared belongs in `enforced_fields` (a null there blocks *all*
canonical runs, which is the intended direction of failure). `mor.ffn_mult` was
resolved to **24** and `enforced_fields.family_cls_weight` to 0.5, and
`config.json` was made literal and complete in the same commit — before this, the
defaults file said `num_blocks 2` / `batch_size 128` against the spec's frozen
`1` / `768`, so the *default* run was silently a proxy and only an explicitly
overridden run could ever be canonical.

`ffn_mult = 24` equalises **total FFN hidden width**, not the per-FFN multiplier:
MoR has one FFN where MoRE has six, so 24 = 4 × 6 is principled rather than
fitted. Measured: 4 → 571,150 (82.2% short), 23 → 3,066,382 (4.22%), **24 →
3,197,710 (0.120%)**, 25 → 3,329,038 (3.98%). The residual 0.120% is the router
plus the six per-expert halt heads MoR does not have and is irreducible by any
integer multiplier — that is the unavoidable difference T9.2 requires be stated,
and it favours MoRE by 3,845 parameters, far too small to explain any task-loss
difference. `ffn_mult = 4` survives as the labelled T10.I under-budgeted arm.

Gate 0 measured: `canonical_phase_b` **accepted** for all three architectures at
defaults (MoR was refused before, on the null `ffn_mult`), and refused for
`--family_cls 0.0`, `--ffn_mult 4`, `--epochs 2` and `--seed 99`, each with the
field and the required value named in the refusal.

**What the frozen values actually rest on — do not flatten this in the paper.**
`halting_mode` and `step_routing_weight` come from controlled 3-seed experiments
(T8.0a, T8.0b). `d_model`, `lr` and `weight_decay` are **validated-stable but
never swept** — 12 completed 20-epoch runs converge to ≈ 0.0635 without
divergence, and `frozen_by` says so in those words; no width, lr or weight-decay
sweep has ever been run under the corrected Top-1 / ACT / normalized-balance
regime. `family_cls_weight` is weaker still: inherited from a code literal. The
rest follow from `plan.md` or `updated_rules.md` by rule.

**Where to look if this needs revisiting.**
[code/canonical_spec.json](code/canonical_spec.json) is the frozen contract and
its `frozen_by` / `architecture_variants_frozen_by` blocks record *why* each value
holds — read those before quoting any hyperparameter as tuned.
[code/config.json](code/config.json) is the defaults file and must be changed in
the same commit as the spec or the guard will refuse every canonical run. The
guard itself and the per-architecture stamping are in
[code/more/run_context.py](code/more/run_context.py) and
[code/more/config.py](code/more/config.py) (`apply_architecture`). If a canonical
run is unexpectedly refused, the refusal message names the field; if one is
unexpectedly *accepted*, the bug is in `_effective()` not exposing that field to
the guard, which is how `family_cls` stayed invisible until T8.3.

---

## T10.H — Family-supervision ablation: how much of the expert partition does the oracle family label actually own?

**The question.** Canonical MoRE sets `step_routing = 0.0`, so nothing trains the
router on the oracle expert index directly. But `family_cls = 0.5` still trains
`cls_head` on the whole-program family label read off a hand-written manifest —
the same label the routing metrics are scored against. So the honest claim after
T8.0b was hedged: *specialization emerges without direct routing supervision,
under whole-program family supervision*. That hedge names an unmeasured quantity.
T10.H measures it by turning the term off.

**Method.** `automated/family_cls_ablation.py` trains six MoRE runs — two arms
(`family_cls = 0.5`, `family_cls = 0.0`) × seeds 42/43/44 — at 20 epochs,
1 block, batch 768, `pure_act`, `step_routing = 0.0`. Three design decisions
matter for anyone re-reading the numbers later:

1. **Both arms are trained fresh on one code state.** The obvious shortcut was to
   reuse the T8.0b `routedec` runs as the baseline. They are inadmissible twice
   over: they predate T8.3 (no `family_cls` key in their resolved config), and —
   the disqualifying one — they predate T6.8, so they carry only the identity-
   mapped `per_family` table and no `per_family_matched` / `matched_macro_recall`.
   Comparing them against a current run would put the two arms on **different
   metric layers**, and the matched table is precisely the evidence being argued
   over. A missing metric in one arm is not a matched comparison.
2. **Jobs are interleaved by seed**, so a sweep killed halfway still has both
   arms at every completed seed rather than three baselines and no ablation.
3. **The decision rule was written into the module docstring before any run** —
   four pre-registered readings, including "if the partition falls to chance the
   family label owns all of it and truly-unsupervised discovery is Outcome C" and
   an explicit *do not round the retained fraction up*.

**Result — the partition survives with no oracle label in the objective.**
Hungarian accuracy 0.5147 ± 0.0047 with the term, **0.4300 ± 0.0293 without it**,
against chance 0.167; AMI 0.4940 → 0.3568; purity 0.5881 → 0.5041. That is
**~76% of the above-chance partition retained**, so the family CE contributes
roughly a quarter of it and is not what creates it. Removing it costs nothing
measurable on the primary metric: −0.000100 against a seed std of 0.000428, i.e.
inside noise. No expert collapsed in either arm (load entropy 0.954 / 0.980), and
the Hungarian assignment differs on every seed in both arms — stable accuracy
under a different relabelling each time is the evidence of a real partition;
identical assignments at ~1.0 would have meant a supervised index instead.

**What this changes and what it does not.** It does **not** reselect the canonical
value: `family_cls = 0.5` stays frozen in `canonical_spec.json`, both arms are
`experiment_group = exploratory` 20-epoch proxies, and no number above may enter
the headline table. What it does is convert a hedge into a measurement. Canonical
rows keep the "under whole-program family supervision" phrasing, and the stronger
sentence — *the partition does not require any oracle-derived label* — is now
available as a labelled ablation with a number attached.

**A cross-check worth keeping.** The fresh `family_cls = 0.5` seed-42 run
reproduced the T8.0b run's `val/task_loss` to **0.000e+00** at tol 1e-9
(0.064011740289 both). Two refactors sat between them — T8.3 moving `family_cls`
out of the loss assembly into config, and T6.8 adding the matched metric layer —
and neither perturbed the objective. If a future refactor of the loss assembly or
metric layer breaks that equality, this is the run pair to diff against.

**Where to look if this needs revisiting.** The driver is
[automated/family_cls_ablation.py](automated/family_cls_ablation.py); its
`MATCH_FIELDS` / `MATCH_MODEL` lists are what proves only one field differed (54
comparisons, 0 mismatches), and `read_matching()` now refuses to report a pass
when it compared zero fields — the same "absence is not a pass" defect class as
the correctness suite's `EMPTY` verdict. The weight itself lives in
`loss_weights.family_cls` in [code/config.json](code/config.json), is applied in
[code/more/engine.py](code/more/engine.py), and is stamped into provenance as
`family_cls_weight`; read that field first when a routing number looks surprising.
Machine-readable result: `automated/family_cls_ablation_result.json`. One cosmetic
artefact: the captured console banner reads "T10.F" because the process was
launched before the T10.F→T10.H rename landed on disk — the source and all future
runs print T10.H.

---

## T9.0 — Comparison protocol fixed before the canonical matrix launched

**What was wrong.** `logging.log_interval` was 10, and the same field gates
*validation* at `engine.py:629` (`epoch % log_interval == 0 or epoch == epochs`).
So a 20-epoch run measured val exactly twice, and the 50-epoch canonical runs
would have measured it six times. On top of that, "best epoch by val loss" over
two candidates on a still-descending curve always returns the last epoch — the
checkpoint-selection rule and "just take the final model" were operationally the
same thing, and nothing in the repo said which one was intended.

**How it was found.** Reading the T10.H `results.tsv` files rather than only their
`metrics.json`. Eight of twenty rows per run held `nan` in the `val_loss` column.
Those were unevaluated epochs, not failures, but they made the val curve
unreadable. Between the only two measured points val fell **−0.002010 ± 0.000443**
with the same sign in all six runs — roughly 4.7× the seed std, so validation was
still improving at the end of training and the coarse cadence was hiding the curve
rather than summarising it.

**What changed.** `log_interval` 10 → 2 in `config.json`, and the defensive
`setdefault` in `config.py` moved to 2 in the same commit — a default that
disagrees with canonical is precisely the trap T8.3 hit with `family_cls`. The
protocol itself is now recorded in a new non-enforced `protocol` block in
`canonical_spec.json`: `val_interval`, `checkpoint_selection`
(*lowest `val/task_loss` among evaluated epochs, no early stopping, full epoch
count every run, argmin taken post hoc*), the parameter-budget policy with its
0.120% residual, the seed-reporting rule, and the primary-metric floor. It sits
outside `enforced_fields` deliberately: the Gate 0 guard does not check these, but
the protocol audit requires them written down somewhere other than a changelog
entry. `results.tsv` also now writes the string `"N/A"` for an unevaluated epoch
instead of `nan`, matching `metrics.json` — nothing was ever *reported* as a
measurement (the `isnan` guard kept it out of `best_val_loss`), but two
conventions for "no number here" inside one run directory is what a plotting
script misreads as a diverged loss.

**Why this had to happen before Phase 9 and not after.** `log_interval` is not in
`enforced_fields`, so changing it does not break the T8.2 freeze — but it *does*
change `config_hash`. Changed partway through the matrix it would have produced
two hash groups that may never share a table (CLAUDE.md §5), silently splitting
the headline comparison in half.

**Verified.** Correctness suite **347/347, exit 0** after the edits. A 4-epoch
MoRE smoke (`t90_valcadence_check_seed42__d167dc3d`) confirmed both behaviours:
val measured at epochs 2 and 4, `"N/A"` written at epochs 1 and 3, and
`experiment_group = exploratory` because 4 ≠ the frozen 50 — the guard refusing a
short run is the guard working.

**Where to look if this needs revisiting.** The gate is one condition at
[code/more/engine.py:629](code/more/engine.py:629); the TSV renderer is ~350 lines
below it; the value lives in `logging.log_interval` in
[code/config.json](code/config.json) with its default at
[code/more/config.py:172](code/more/config.py:172). If a future run shows a
suspiciously sparse val curve, that condition is the only thing controlling it.
The written protocol is `canonical_spec.json:protocol` — read it before quoting
any comparison rule as established.

---

## T9.1 — The canonical Phase-B matrix: driver and the 50-epoch correction

**A correction that changes how every earlier number must be labelled.** The
frozen canonical epoch count is **50**, not 20. Every multi-seed run this project
has produced so far — T8.0a's ACT decision, T8.0b's routing-supervision decision,
the T8.1 smokes, the six T10.H family-supervision runs — is a **20-epoch proxy**.
They were all correctly stamped `experiment_group = exploratory` by the guard, so
nothing was mislabelled on disk, but the distinction has to survive into the
paper: those runs decided *design questions* under a matched protocol, and none of
them is a headline row.

**The driver.** `code/run_phase9_matrix.py`, deliberately in `code/` and not
`automated/` — CLAUDE.md §9 bars `automated/` from launching canonical runs. Four
rules are fixed in its docstring before the first run:

1. **No CLI overrides beyond `--architecture`, `--seed`, `--run_name`,
   `--experiment_group`.** Every other field must be *inherited* from
   `config.json` so the Gate 0 guard can compare it against the spec. A driver
   that passes `--epochs` is asserting the value instead of inheriting it, and one
   typo becomes a proxy wearing a canonical label. `--experiment_group` is the
   exception that proves the rule: it is not a config value, it is the canonical
   *claim* the guard adjudicates. Omitting it cost 1.6 hours of compute — see the
   next entry.
2. **Admissibility is read from provenance, not from the exit code.** Each run
   must show `experiment_group = canonical_phase_b`, `variant = canonical` and
   `seed_declared = True`; anything else is recorded as a problem and excluded
   from the aggregate rather than quietly averaged in.
3. **Architecture-inapplicable metrics aggregate to `N/A`.** `mean_std()` skips
   the string `"N/A"` and returns `None`, which prints as `N/A`. One extra
   suppression was needed: MoE emits `train/avg_recursion_steps` as the float
   `1.0`, which is a restatement of `max_depth = 1`, not a measurement — beside
   MoRE's 2.13 it would read as a comparison of learned depths. Suppressed in the
   driver, not in `engine.py`, so no existing run's `metrics.json` changes meaning.
4. **Every number is mean ± std over five seeds, and the report does not rank the
   architectures.** Each pairwise gap is printed beside the larger of the two
   arms' seed stds and labelled `RESOLVED` (≥ 2×) or `inside seed noise`. The
   three architectures are known to sit in a narrow band, so this is the only
   honest way to present the primary comparison.

The driver also cross-checks that `config_hash` is *constant within* each
architecture (a seed must not change the config) while differing *across* them,
and refuses to call an invariant check passed when it compared zero fields — the
same "absence is not a pass" rule as the correctness suite's `EMPTY` verdict.
Jobs are interleaved by seed, so a matrix interrupted halfway holds all three
architectures at every completed seed instead of five MoE runs with nothing to
compare against. Runs already on disk are skipped, which makes a ~4-hour matrix
survive an interrupted session.

**Where to look if this needs revisiting.** `SCALARS` in
[code/run_phase9_matrix.py](code/run_phase9_matrix.py) is the exact metric-key
list, and it was validated against a real `metrics.json` before launch — one key
(`total_params`) turned out to live in provenance rather than metrics and is now
read from there. If the matrix reports a whole column as `N/A`, suspect a renamed
metric key there before suspecting the runs. Machine-readable output:
`code/phase9_matrix_result.json`, whose `clean` flag is false whenever any of the
15 runs was excluded.


## T9.1a — The undeclared-group defect: six perfectly-canonical runs stamped `exploratory`

**The failure.** The first launch of the T9.1 matrix ran for ~1.6 hours and
produced five complete 50-epoch runs (moe/mor/more at seed 42, moe/mor at seed 43)
plus one killed mid-training. Every one of them is inadmissible as a headline row.
The launch banner said it out loud on run 1 — `[Run] experiment_group=
exploratory` — and provenance confirmed it: `variant: canonical`,
`seed_declared: True`, `resolved_epochs: 50`, `resolved_batch_size: 768`,
`resolved_subset_fraction: 1.0`, and `experiment_group: exploratory`. Every
*enforced* field matched the frozen spec. The single field the exporter filters on
said "proxy".

**The cause, and why no guard caught it.** `--experiment_group` is a CLI flag with
`default=None` ([code/more/cli.py:107](code/more/cli.py:107)). The Gate 0 guard
`assert_not_silent_proxy` reads the claim from `resolved["logging"]
["experiment_group"]` ([code/more/run_context.py:256](code/more/run_context.py:256));
when that is not `canonical_phase_b` it **returns
`NONCANONICAL_GROUP_DEFAULT` with no error and no refusal.** That is deliberate
design, not a bug: exploratory work must stay cheap, so an undeclared run is
simply stamped exploratory. The consequence is a failure mode Gate 0 *cannot*
catch by construction — a run can satisfy every enforced field and still be a
proxy, because the group is a **claim the launcher makes**, not a value the guard
derives. My driver's Rule 1 ("no CLI overrides") had classified the flag as an
override and omitted it.

**The guard was never broken.** Verified in-process before touching anything:
setting `provenance.experiment_group` returns `exploratory` (wrong input path);
setting `logging.experiment_group` alone refuses with *seed is not set* (the seed
is read from `provenance.seed`, [run_context.py:181](code/more/run_context.py:181));
setting `logging.experiment_group = 'canonical_phase_b'` **and**
`provenance.seed = 42` returns `'canonical_phase_b'` for moe, mor and more at
config defaults. The declaration was the only thing missing.

**The fix, in two parts.** `launch()` now passes `--experiment_group
canonical_phase_b`. And `find_dir()` gained `canonical_only`, used on the resume
path: without it an `exploratory` directory would keep satisfying the skip check,
so the failure would become **permanent** — the driver would skip that pair
forever and report it as excluded on every subsequent invocation. That second half
matters more than the first; a resumable driver can cache its own bugs.

**Archived, not re-stamped.** The six directories are at
`archive/t9_1_undeclared_group_runs/` with an `INVALIDATED.md` recording what they
are and why. Editing `experiment_group` onto a finished `resolved_config.json`
would manufacture a canonical label that no guard ever adjudicated — precisely the
silent proxy the guard exists to prevent. A group is declared at launch and
validated by the guard, never written onto a finished run (CLAUDE.md §5, §6).

**Where to look if this recurs.** Read the *launch banner*, not the config: the
line `[Run] experiment_group=` on the first run of any canonical batch is the
cheapest possible check, and it appears within ~60 seconds. If it says
`exploratory`, kill the batch immediately. Any new canonical driver must pass
`--experiment_group canonical_phase_b`; this belongs in the pre-Phase-7 audit as a
protocol defect rather than a code defect.


## T9.1b — The matrix landed 15/15, and the one thing it flagged was its own verification rule

**The matrix ran.** All 15 canonical runs completed with exit 0 in **236 min**
(3.9 h), matching the ~4 h estimate. Per-run wall clock: MoE ≈ 7.7 min, MoR ≈ 13.4
min, MoRE ≈ 25.8 min — MoRE is ~3.4× MoE because it recurses *and* routes.
All 15 admitted: `experiment_group = canonical_phase_b`, `variant = canonical`,
`seed_declared = True`, and **168 invariant provenance comparisons with 0
mismatches** across the 12 protocol fields.

**But the driver exited 1 with three problems**, one per architecture:

```
moe: 5 distinct config_hash values across seeds -- seeds must not change the config
```

**The predicate was wrong, not the runs.** `config_hash()`
([run_context.py:82](code/more/run_context.py:82)) hashes the whole config minus
`provenance` and `logging`. `--seed` is written into `data.subset_seed` **and**
`training.subset_seed`, which are inside that. So config_hash differs across the
five seeds of one architecture *by construction* — indeed the 8-char hash is the
run-directory suffix, which is why `phaseB_moe_seed42__c35fe8b6` and
`phaseB_moe_seed43__33afa654` never collide. I checked what actually differs by
diffing the hashed portion of the five `resolved_config.json` files field by
field: exactly two fields, `data.subset_seed` and `training.subset_seed`, and
nothing else. And `resolved_subset_fraction = 1.0` on all 15 runs, so
`subset_seed` provably **could not have changed the data** in this matrix.

**A test had been asserting the opposite and passing.** `test_phase6_provenance.py`
T6.7b read *"two seeds of one experiment share a config_hash, so the exporter can
average them instead of refusing to mix hashes"* — while varying only
`logging.run_name`. It never exercised the claim it made, so it passed while the
claim was false. This is the same defect class as the guard landing in dead code
and the `EMPTY` verdict: **a check that cannot fail is not a check.**

**Resolution, and what was deliberately *not* changed.** `config_hash()` is
untouched. `subset_seed` genuinely selects a different data subset whenever
`subset_fraction < 1.0`, so dropping it from the hash would let two different
datasets share one identity — a worse defect than the one being fixed, and it
would silently change the hash of every future run, breaking comparability with
these 15. Instead the *predicate* was corrected to ask the question that actually
matters:

- `scientific_config()` reduces a run to the hashed config **minus the seed
  fields**, flattened to dotted paths.
- `seed_blind_diff()` reports the differing **field paths** across an
  architecture's five seeds, not two opaque hashes. `moe / mor / more` all report
  `identical`.
- Fewer than 2 admitted runs is itself a problem — absence is not a pass.
- The report now prints both: config_hash per architecture (expected to differ by
  architecture *and* seed) and the seed-blind identity verdict.

T6.7b was rewritten into four honest assertions: the run label is outside the hash;
changing the **real** seed *does* change the hash (so the matrix may not verify
seeds by hash); two seeds are identical once the seed is factored out; and the
seed-blind comparison still catches a real change (`lr 0.001 → 0.002`). Re-run:
**350/350 checks, all five gates PASS** (up from 347 — the three new assertions).
The matrix re-aggregated `--report-only` to **exit 0, no problems**, with no run
re-executed.

**Where to look if this needs revisiting.** `SEED_KEYS` in
[code/run_phase9_matrix.py](code/run_phase9_matrix.py) is the list of config paths
the seed is allowed to write into. If a future change routes the seed into a new
field, the matrix will report `seeds disagree on a non-seed config field -- <path>`
and naming that path in `SEED_KEYS` is the fix — *after* confirming the field
really is seed plumbing and not a configuration difference. If a real exporter is
ever written, it must compare seed-blind too; comparing raw `config_hash` would
refuse to average the five seeds it exists to average.


## T9.1c — What the canonical matrix actually measured (the headline result)

**This is the first table in the project's history that is allowed in the paper.**
15 runs, 50 epochs, full dataset, seeds 42–46, `experiment_group =
canonical_phase_b`, 0 protocol mismatches. Every number is mean ± std over five
seeds. Machine-readable: `code/phase9_matrix_result.json`.

| arch | `val/task_loss` | `best_val_loss` | R² vs floor | params |
|---|---|---|---|---|
| MoE | 0.063260 ± 0.000137 | 0.062819 ± 0.000173 | 0.2182 | 3,201,555 |
| MoR | **0.062706 ± 0.000100** | 0.062527 ± 0.000047 | 0.2250 | 3,197,710 |
| MoRE | 0.063186 ± 0.000223 | 0.062892 ± 0.000193 | 0.2191 | 3,201,555 |

Floor (predict the train mean on the real val split) = 0.080914.

**The primary comparison, read against seed noise:**

| gap | value | vs larger seed std | reading |
|---|---|---|---|
| MoRE − MoE | −0.000074 | 0.33× | **inside seed noise** |
| MoRE − MoR | +0.000481 | 2.16× | resolved |
| MoE − MoR | +0.000554 | 4.04× | resolved |

**This is Outcome C on the primary metric, and it is reported as such.** MoR — one
shared block, one expert, no routing at all — has the lowest validation task loss,
and its margin over both MoE and MoRE clears 2× the larger seed std. MoRE is
indistinguishable from MoE. Adding six specialized experts to a recursive block
did **not** improve prediction; it cost 0.000481 in val loss and 3.8× in
throughput. Two honest caveats in the other direction: the gap is ~0.8% of the
loss and every architecture beats the predict-the-mean floor by only ~22%, so all
three are weak models of this task and the ranking is a ranking among weak models;
and MoR carries 3,845 *fewer* parameters, so the residual 0.120% budget difference
favours the two arms that lost.

**Routing does partition the operation space, without any oracle label in the
objective** (`routing_supervision_weight = 0.0`):

| metric | MoE | MoRE |
|---|---|---|
| Hungarian-matched accuracy | 0.5004 ± 0.0501 | 0.5258 ± 0.0547 |
| AMI | 0.5132 ± 0.0758 | 0.4907 ± 0.0659 |
| purity | 0.5715 ± 0.0563 | 0.6047 ± 0.0552 |
| matched macro recall | 0.5133 ± 0.0493 | 0.5491 ± 0.0881 |
| raw accuracy | 0.2016 ± 0.1411 | 0.1629 ± 0.1668 |

Hungarian ≈ 0.51–0.53 against chance 0.167 with AMI ≈ 0.49–0.51: **a real
partition, arbitrarily numbered.** That is exactly the raw ≪ Hungarian signature
`results_exp.md` says to read as permutation, not failure — and the huge raw std
(±0.14, ±0.17) is the permutation itself varying by seed, which is why raw accuracy
must never be quoted alone. MoRE and MoE partition about equally well; specialization
is not what separates them.

**Adaptive depth is measured but is not tracking the curriculum:**

| metric | MoR | MoRE |
|---|---|---|
| avg recursion steps | 2.0169 ± 0.0191 | 2.0642 ± 0.0520 |
| depth allocation error (abs) | 0.9509 ± 0.0159 | 1.0055 ± 0.1033 |
| early-exit rate | 0.9998 ± 0.0005 | 0.9998 ± 0.0002 |
| forced-exit rate | 0.0002 ± 0.0005 | 0.0002 ± 0.0002 |

Halting is alive (99.98% of tokens exit early, forced exit ≈ 0), but average depth
sits at ~2.0 of `max_depth` for both, and an absolute allocation error of ~1.0 step
means depth is **not** aligned with the predefined operation-complexity curriculum.
Consistent with the earlier ACT decision experiment. The correct phrasing remains
*the model allocates recursion depth, but not in accordance with the curriculum* —
not "discovers intrinsic complexity".

**Sparsity and cost.** `dispatch/evals_per_token = 1.0000 ± 0.0000` for all three,
which is the Top-1 sparse-dispatch invariant holding exactly. Throughput 8485 ± 324
(MoE) / 4574 ± 637 (MoR) / 2225 ± 246 (MoRE) tokens/s — a machine-specific
engineering observation on one RTX 4060 Laptop, never an architectural efficiency
claim. Diagnostics: load entropy ≈ 0.988 and per-token router entropy ≈ 0.997 are
**balance, not specialization** (a near-uniform router); max pairwise cosine 0.051
(MoE) / 0.073 (MoRE) is *consistent with differentiated parameterizations*, not
proof of orthogonality.

**Where to look if this needs revisiting.** The 15 directories are
`runs/phaseB_{moe,mor,more}_seed4{2..6}__*`, each with `resolved_config.json`,
`metrics.json`, `results.tsv` and `stdout.log`. `code/phase9_matrix_result.json`
carries the aggregate with a `clean` flag. To re-derive the table without
retraining: `python run_phase9_matrix.py --report-only`. Do not merge these rows
with any pre-T6.8 run, any 20-epoch run (T8.0a/T8.0b/T10.H), or any Phase-10
ablation arm — different epoch count or different loss configuration.

---

## T10.0a — Two defects that would have quietly corrupted the Phase 10 table

Both found while preparing Phase 10, both fixed before any ablation ran.

**`train.py --help` crashed.** `ValueError: unsupported format character ')'`. Two
literal `%` signs in the `--ffn_mult` help text at `code/more/cli.py:96–105`
("0.12%", "82.2% short") were passed through argparse's `%`-formatting. Escaped to
`%%`. Trivial to fix, but worth an entry for what it *concealed*: because `--help`
raised, nobody had ever read the flag list end to end, and the list turns out to
have no flag for `fixed_depth`, `router_noise` or `routing_mode`. Three of the five
Phase 10 arms vary a field that cannot be set from the command line at all. That is
why the driver generates a per-arm config file and passes `--config` instead of
building a flag string — a design decision that came directly out of a broken
help message.

**A 2-block run printed `variant = canonical`.** `resolve_variant` in
`code/more/config.py` tagged deviations in `routing_mode`, `router_noise`,
`fixed_depth`, `step_routing` and `ffn_mult`, but not `num_blocks`. The Gate 0
guard already refuses a `canonical_phase_b` *claim* from a 2-block run —
`num_blocks` is in `canonical_spec.json:enforced_fields` — so nothing unsafe could
enter the headline table. But `variant` is the string an **ablation** table prints,
and Phase 10's whole job is to print one row per deviation. Without the tag the
T10.C row would have been labelled identically to the baseline row it was meant to
be compared against. Fixed additively: `CANONICAL_NUM_BLOCKS = 1` is named as a
constant and `resolve_variant` appends `num_blocks_{n}` when it differs. Canonical
is 1, so all 15 T9.1 runs still resolve to `canonical` and nothing already on disk
changed meaning.

Verified all six Phase 10 configurations now label distinctly: `canonical`,
`fixed_depth`, `num_blocks_2`, `router_noise_fixed_annealed`, `oracle_routing`,
`routing_dense_blend`.

**Where to look if labels look wrong.** `resolve_variant` in
`code/more/config.py` is the single place a deviation becomes a printable name. It
validates nothing — see T10.0 below for what that cost.

---

## T10.0b — CLAUDE.md §10: two audiences, two registers

Not a code change; a documentation contract, added because agent output was being
written at one altitude for two different readers.

**The problem.** Files in this repository are read by the next agent, months later,
trying to work out why a guard exists or what a metric key means. Chat messages are
read by a researcher deciding whether to spend six GPU-hours. The same prose cannot
serve both. Reporting `stdout.log carried 281 copies of "_histc_cuda does not have
a deterministic implementation"` into a chat message spends the reader's working
memory on a build detail and crowds out the decision they actually have to make —
the register in which the `cls_loss` debate and the MoR findings were discussed.

**The rule.** `CLAUDE.md §10` now states: full engineering depth — `file:line`
references, exact metric keys, verbatim failure output, rejected alternatives,
provenance detail — in **files** (changelog, TASKS.md, code comments and docstrings,
ARCHITECTURE.md, results_exp.md, the paper). In **chat**: the finding first, then
its implication for the architecture or the paper, the cost, and the decision
needed. Leave out log lines, stack traces, config paths, hashes and directory names
unless the user is about to act on them. Results tables in chat are welcome — they
*are* the finding. Implementation tables are not.

The section closes with an explicit ceiling-not-default clause: when the user asks
a mechanism question, answer it at full depth.

**Where to look.** `CLAUDE.md §10`, with a worked ✗/✓ contrast on the
`log_interval` fix.

---

## T10.0 — The Phase 10 ablation driver: one field, proved by construction

`automated/phase10_ablations.py`. Five arms, each removing or perturbing exactly
one mechanism of canonical MoRE, against the canonical MoRE runs from T9.1.

**Why the questions changed.** T9.1 came back Outcome C on the primary metric: MoR
(one shared block, no routing) has the lowest val task loss, and MoRE is
indistinguishable from MoE. So Phase 10 is no longer asking *how much does MoRE win
by* — it is asking *which of MoRE's two mechanisms is doing anything at all*. The
arms, in the order the driver runs them (highest scientific value first, so a batch
killed halfway has answered the important questions):

| task | arm | field changed | question |
|---|---|---|---|
| T10.F | `dense_routing` | `model.routing_mode` `top1_sparse → dense_blend` | does Top-1 sparse dispatch cost anything against evaluate-all-and-blend? |
| T10.B | `fixed_depth` | `model.fixed_depth` `False → True` | does learned adaptive halting beat a constant depth, given the ~1.0-step allocation error T9.1 measured? |
| T10.E | `routing_supervision` | `loss_weights.step_routing` `0.0 → 0.5` | does supervising the router with the oracle expert index improve the *task* metric, or only the routing metric? |
| T10.D | `router_noise` | `model.router_noise` `none → fixed_annealed` | does exploration noise change the partition or the loss? |
| T10.C | `two_blocks` | `model.num_blocks` `1 → 2` | does a second recursive block help, varying *only* `num_blocks`? |

`num_blocks` last on purpose: the old 1-vs-2-block headline moved three factors at
once and is uninterpretable. This is the run that makes that claim decidable.

**`--config`, not flags.** Each arm is a full config file under
`automated/phase10_configs/`, generated from `code/config.json` with one leaf
changed. `write_arm_configs()` reads the file back, flattens it, and asserts the
diff against the canonical MoRE base is *exactly* that leaf — so a future edit to
`config.json` or to `load_config_defaults` that makes an arm differ in two places
raises before any GPU time is spent, instead of producing a two-factor "ablation"
that reads as one-factor in the table. Three of the five fields have no CLI flag
(T10.0a), and a config file has the further advantage of being diffable on disk.

**The base config is materialized, not raw.** `routing_mode`, `router_noise` and
`fixed_depth` are absent from `config.json` and appear only via `setdefault` inside
`load_config_defaults`. A diff against the raw file would show an *addition* for
those three arms and a value *change* for the other two — two different kinds of
evidence. `materialized_base()` runs the same loader the trainer runs, so all five
fields are present and every arm is provably a one-value change from a stated
value. `fixed_depth` needed one more step: it exists only as a bare literal default
(`mc.get("fixed_depth", False)` at `more/engine.py:210` and `more/config.py:321`),
in no config file and in no provenance block — the same defect class T8.3 surfaced
with the `0.5` `family_cls` literal. `IMPLICIT_DEFAULTS` fills it on both sides of
every comparison so *absent* and *the canonical value* are one thing.

**The intent check and the outcome check are different claims.** Between the config
file and the model sit `load_config_defaults`, `apply_architecture`,
`resolve_overrides` and `enforce_routing_mode`, any of which can stamp a second
field. So `one_field_check()` re-runs the comparison on `resolved_config.json` —
what actually trained — seed-blind and label-blind, and requires the differing set
to be exactly the intended leaf. Validated against the smoke runs, where it
correctly reported `training.epochs: baseline 50 vs arm 1` as an unintended second
difference. This is the check that makes the table's one-factor claim true.

**The baseline is reused, and that is guarded three ways.** The T9.1 canonical MoRE
runs (seeds 42–46) are the baseline arm rather than being re-trained — legitimate
only because nothing in the model has changed since, the T10.H lesson being that two
arms must not straddle a code change. `verify_baseline_state()` enforces it:

1. *Retroactive, source.* No file in `more/{model,engine,data}.py` may be newer than
   the oldest baseline run's `metrics.json`.
2. *Retroactive, artefact.* Today's `config.json`, resolved through the trainer's own
   loader, must equal what every baseline run recorded in `resolved_config.json`.
   This is the check that matters for the plumbing files — `config.py`, `cli.py`,
   `run_context.py` are deliberately **not** mtime-checked, because a label-only edit
   (T10.0a's `num_blocks_2` tag) cannot change what trains and refusing on it would
   make the guard cry wolf. Comparing the artefact proves the configuration is
   unchanged regardless of how the loader was edited, which a source hash cannot show.
3. *Prospective.* A sha256 fingerprint of all six sources is pinned on first launch
   (`automated/phase10_code_fingerprint.json`) and re-checked on every later one, so
   an edit landing *between two ablation arms* is caught even though it postdates the
   baseline.

Ablation arms use seeds 42–44 and the baseline mean is restricted to the **same
three seeds** so the n matches; the 5-seed value is printed beside it for reference.

**The reading rule, fixed before any run.** `|arm − baseline| ≥ 2 ×` the largest
available seed std, same as T9.1 Rule 4. One refinement: the 3-seed baseline subset's
std is ~4.5× *smaller* than the 5-seed std (0.000050 vs 0.000223 on
`val/task_loss`) — three draws under-sample the seed spread, and taking the smaller
number would make every gap look more resolved than it is. So the mean is compared
at matched n=3 while the noise yardstick admits all five seeds. Below 2× the verdict
is `inside seed noise`, which supports **no** claim in either direction — not "a
small improvement", not "no effect".

**Not canonical, by construction.** No arm passes `--experiment_group`, so every one
is stamped `exploratory` and excluded from the headline. That is correct — an
ablation is by definition not the frozen spec — and it is done by omission rather
than by relying on the Gate 0 guard's refusal, because relying on an error path for
correctness is how T9.1a happened.

**Absence is not a pass.** A report run before anything has trained used to print
the clean verdict *"every completed arm differs in exactly one field"* — true and
worthless, since zero arms were checked. `collect()` now records that as a problem.
Same defect class as the T9.1b assertion that could not fail.

**What the 1-epoch smoke test caught.** All five arms were run for one epoch before
committing the batch. `router_noise` had been declared as `"gaussian"`, which does
not exist — `more/model.py:71` defines `ROUTER_NOISE_MODES = ("none",
"fixed_annealed", "trainable")` and the `MoEBlock` constructor raised. The failure
mode worth recording is not the typo: `resolve_variant` had already produced the
label `router_noise_gaussian` without complaint, because it string-formats whatever
it is handed and validates nothing. The label layer will happily name a
configuration that cannot run. The driver now validates enum fields against
`more.model`'s own tuples at config-generation time (`LEGAL_VALUES`), so the two
cannot drift apart. `fixed_annealed` was chosen over `trainable` deliberately:
CLAUDE.md §2 records that the L2-on-trainable-noise-scale mechanism is not a proven
fix, so ablating `trainable` would confound router noise with unproven machinery.
On the real batch this would have surfaced after two completed arms, roughly two
hours in. Smoke directories are archived at `archive/t10_smoke_1epoch/` with an
`INVALIDATED.md`; at `epochs = 1` they are proxies in the strictest sense and may
never appear in any table.

**Cost, sized from the smoke throughput.** ≈6.4 h for 5 arms × 3 seeds: dense
routing 17.7 min/run, routing supervision 19.6, fixed depth 24.6, router noise 26.0,
two blocks 41.0. Note dense routing came out *faster* than Top-1 sparse — the
workload is launch-overhead-bound (T9.1c: 0–30% GPU utilisation at ~2 W), so
evaluating all six experts as one batched matmul can cost less wall clock than
gathering and scattering for one. A hypothesis to confirm on the real arm, not a
result: one epoch on a downclocked laptop GPU is not a throughput measurement.

**Where to look if a Phase 10 row looks wrong.** `automated/phase10_ablations.py`
is the whole driver; `automated/phase10_configs/<arm>.json` is the exact
configuration that trained, each carrying its own `_README` naming the sole
deviation; `automated/phase10_ablations_result.json` carries the aggregate and the
problem list. To re-derive the table without retraining:
`python automated/phase10_ablations.py --report-only`. To verify the one-field claim
without running anything: `--configs-only`. Arms already on disk are skipped, so an
interrupted batch resumes.

---

## T10.1 — What the ablations measured: neither of MoRE's two mechanisms is earning its place

> **CORRECTED by T11.0a.** Every `± std` and every verdict first written in this
> entry came from a population std (÷n) instead of the sample std (÷n−1), which
> understates the spread 22% at n=3 and inflated the `2 × std` ratios by the same
> 22%. **Both RESOLVED verdicts below were artifacts and are now `inside seed
> noise` (1.88× and 1.94×).** The tables in this entry have been corrected in
> place; the reasoning that did not depend on the loss threshold is unchanged.
> Read T11.0a before quoting any number from here.


15 runs (5 arms × seeds 42–44), 50 epochs, against the canonical MoRE arm from T9.1
restricted to the same three seeds. `automated/phase10_ablations_result.json`.
Driver reported no problems: every arm's `resolved_config.json` differs from the
baseline's in exactly the intended leaf, and all arms share one code state.

### Primary metric — `val/task_loss`

| arm | field changed | val/task_loss | verdict (2× seed std rule) |
|---|---|---|---|
| canonical MoRE | — | 0.063290 ± 0.000061 | baseline (5-seed: 0.063186 ± 0.000249) |
| fixed_depth | `model.fixed_depth` `False → True` | 0.062822 ± 0.000108 | inside seed noise, **1.88×** |
| routing_supervision | `loss_weights.step_routing` `0.0 → 0.5` | 0.063772 ± 0.000166 | inside seed noise, **1.94×** |
| dense_routing | `model.routing_mode` `top1_sparse → dense_blend` | 0.063687 ± 0.000426 | inside seed noise, 0.93× |
| router_noise | `model.router_noise` `none → fixed_annealed` | 0.063363 ± 0.000399 | inside seed noise, 0.18× |
| two_blocks | `model.num_blocks` `1 → 2` | 0.063258 ± 0.000423 | inside seed noise, 0.08× |

**No arm resolves on the primary metric.** Two came within 6% of the threshold and
were originally mis-read as resolved (see the banner above). The scientifically
important consequence is not "nothing happened" — it is that **`val/task_loss` on
this dataset cannot discriminate any of these five mechanisms at n=3**, which is a
statement about the benchmark's resolving power and belongs in the paper as such.
The findings that survive live in the *non-loss* metrics below, where several arms
move by many multiples of the seed std.


### T10.B — adaptive halting costs no measurable accuracy, and saves 3.4× the depth

`fixed_depth = True` runs every token to `max_depth = 7`
(`train/avg_recursion_steps = 7.0000 ± 0.0000`, `halt/forced_exit_rate = 1.0`,
`depth/allocation_error_abs = 5.8850` — the error is large *by construction* here
and carries no information, since a constant 7 cannot track a 2–4 curriculum). It
lands 0.000468 below canonical MoRE at **1.88×** the seed std — under the
pre-registered threshold, so it **supports no claim of superiority.**

**The direction this points has reversed relative to the first reading, and the
reversal matters.** With the threshold not crossed, the defensible statement is:
*canonical MoRE reaches the same task loss as a fixed 7-step schedule while
spending 2.05 steps — a 3.4× reduction in recursion compute for no measurable
quality cost.* That is what adaptive computation is supposed to buy, and it is a
positive result for the halting machinery's *efficiency*.

What it is **not** is evidence that the halting *policy* is good. T10.J (below)
shows depth anywhere in 2.0–2.5 gives the same loss, and
`depth/allocation_error_abs ≈ 1.0 step` says the policy does not track the
complexity curriculum. So the compute saving is real, and the allocation rule is
still unvalidated: a constant 2 would plausibly do as well, and no arm run so far
excludes that. **That is the arm to run if one more is affordable** —
`max_depth = 2` fixed, which isolates "is adaptive allocation worth anything over
a small constant" and is the one question this phase left genuinely open.

For the same reason the earlier MoR comparison must be softened: fixed-depth MoRE
(0.062822 ± 0.000108) sits 1.04× the seed std from canonical MoR
(0.062706 ± 0.000112), and canonical MoRE itself is 1.93× from MoR — both inside
noise. Recorded as an informal observation and NOT as a table row (different n,
different experiment group, CLAUDE.md §6). MoRE is not measurably worse than MoR
on this benchmark; the earlier "MoRE loses to MoR" reading was the same std
artifact.


### T10.E — perfect routing buys nothing

`loss_weights.step_routing = 0.5` supervises the router against the family manifest.
It works exactly as intended and drives every routing metric to its ceiling:

| metric | canonical MoRE | routing_supervision |
|---|---|---|
| `routing/hungarian_accuracy` | 0.5538 ± 0.0468 | **1.0000 ± 0.0000** |
| `routing/ami` | 0.5092 | **1.0000** |
| `routing/cluster_purity` | 0.6224 | **1.0000** |
| `routing/matched_macro_recall` | 0.5712 | **1.0000** |
| `routing/accuracy` (raw, index-aligned) | 0.0790 | **1.0000** |
| `val/task_loss` | 0.063290 | 0.063772 (**1.94×, inside noise**) |

Raw accuracy rising 0.0790 → 1.0000 while Hungarian rises 0.5538 → 1.0000 is the
expected signature: supervision pins expert *indices* to families, so the arbitrary
permutation that the unsupervised router is free to choose disappears. This is also
the cleanest available confirmation that the Hungarian layer added in T6.3 is sound —
the two accuracies coincide exactly when and only when the permutation is identity.

Balance moves the other way: `routing/load_entropy` 0.9860 → 0.9518 and
`loss/entropy_term` 0.9985 → 0.8208, i.e. supervision makes the load *less* uniform,
because the family distribution in the dataset is not uniform. Nothing is collapsing;
the balance loss is simply no longer the thing choosing the assignment.

**The consequence for the paper's premise, which survives the correction.** A routing
head that is 100% correct about operation family produces **no better** task loss
than one that is 8% correct — the point estimate is worse, at 1.94× the seed std, so
"hurts" is not supportable but "buys nothing" is, and the effect on the routing
metrics is a 12-σ move against a 1.9-σ move on loss. Whatever the router contributes,
it is not identifying the operation family. The specialization story cannot be the
mechanism, because handing the mechanism to the model for free changes the primary
metric not at all. Combined with T8.3 (`cls_loss` coefficient found to be 0.0 in
canon) this is two independent results saying supervised family identity is not the
useful signal.


### T10.F — Top-1 sparsity buys nothing at this scale, and costs wall clock

`dense_blend` evaluates all six experts and blends by gate probability
(`compute/expert_evals_per_token` 1.0 → **6.0**). Task loss is indistinguishable
(0.93×), and it is **1.40× faster** in wall clock (2912 ± 145 vs 2073 ± 170 tok/s) —
confirming on the real arm the hypothesis the 1-epoch smoke test raised
(`archive/t10_smoke_1epoch/INVALIDATED.md`): the loop is launch-overhead-bound, so
one batched six-expert matmul beats a gather/scatter for one expert. Routing quality
*degrades* under dense (Hungarian 0.5538 → 0.4963, purity 0.6224 → 0.5404), which is
consistent with a blend having no pressure to commit to an expert.

The arm never risked being mislabelled as MoE: `enforce_routing_mode` refuses any
`dense_blend` run whose label lacks `dense_routing_ablation`, so its run name is
`t10_more_dense_routing_ablation_seed{42,43,44}`.

### T10.D and T10.C — the two flattest results in the phase

`router_noise = fixed_annealed` (0.18× seed std) changes nothing measurable, which
supports keeping `router_noise = none` as canonical — the current default was chosen
on principle in T4, and this is the first evidence that the choice is also empirically
free.

`num_blocks = 2` is the flattest arm in the entire phase at **0.08× the seed std**,
while carrying **1.99× the parameters** (6,358,553 vs 3,201,555) and running at 0.77×
the throughput (1587 ± 153 tok/s). Two things follow:

1. **The model is not capacity-limited.** Doubling parameters produces a change 11×
   smaller than seed noise. Any future "make the model bigger to go faster / do
   better" proposal now has a measured answer, and a capacity sweep is a low-value
   experiment.
2. **The old confounded 1-vs-2-block headline is finally decidable.** Pre-T6.8 runs
   varied block count alongside other fields, so that comparison never isolated the
   variable. Isolated, the second recursive block does nothing.

### Cost and provenance

~6.1 h of GPU time against a 6.4 h estimate, in two phases: 10 runs / 222 min, then
`t10_more_router_noise_seed43` died 0.5 min in with exit `3221226505`
(0xC0000409, STATUS_STACK_BUFFER_OVERRUN) immediately after W&B setup and with no
Python traceback. The `[STOP]` guard halted the batch as designed rather than
continuing with a hole in the matrix. Judged transient — seed 42 of the same arm had
completed clean in 28.4 min — the dead directory was archived to
`archive/t10_crashed_runs/` rather than deleted, and the driver's
`find_arm_dir()`/`[plan]` logic resumed with `5 run(s) to launch, 10 already on disk`.
All five completed exit 0 in 142 min. If that exit code recurs on a *specific* arm
rather than randomly, suspect the W&B init path, not the model.

Every arm is `experiment_group = exploratory` by omission of `--experiment_group`,
never by relying on the guard's error path (which is how T9.1a was invalidated).
No Phase 10 run may enter the headline table; these are ablations and are labelled as
such, with the varying field named per arm.

**Where to look.** Driver and arm table: `automated/phase10_ablations.py`
(`ARMS`, `rule4()`, `one_field_check()`). Aggregate numbers:
`automated/phase10_ablations_result.json`. Per-arm intent: the `_README` key in each
`automated/phase10_configs/*.json`. Halting mechanics if the fixed-depth result needs
re-examining: `more/engine.py:210` (`fixed_depth` read) and the ponder-cost assembly
in the same file.

---

## T10.J — The ponder coefficient is exonerated, and the benchmark cannot see depth

> **CORRECTED by T11.0a** (population-vs-sample std, see that entry). The original
> title of this entry was "*the halt head is the defect*" — that conclusion depended
> on T10.B's fixed-depth arm being a resolved improvement, which it is not. Numbers
> and conclusion below are the corrected ones.


3 seeds × 50 epochs, `loss_weights.halting` `0.001 → 0.0001` (10× weaker), nothing
else changed. Same driver, same baseline, same reading rule. ~85 min.

This arm exists because T10.B could not distinguish two very different claims. Fixed
depth beat learned halting, but fixed depth also raised the compute budget 3.4×, so
"adaptive computation does not work here" and "our ponder cost is mis-weighted" both
predicted that result. **The reading was pre-registered in the arm's docstring before
any seed ran** (`automated/phase10_ablations.py`, `ARMS` entry `T10.J`):

> *if `avg_recursion_steps` rises from ~2.05 toward `max_depth` and `val/task_loss`
> moves to the fixed_depth number (0.062822), the ACT objective was mis-weighted [...]
> If depth rises but loss does not follow, the halt head is not learning anything
> useful and Outcome C stands for the depth half of the architecture.*

| metric | canonical MoRE | ponder ×0.1 | fixed depth | change |
|---|---|---|---|---|
| `train/avg_recursion_steps` | 2.0524 ± 0.0749 | **2.5120 ± 0.0876** | 7.0000 | **+0.46, ≈5.2× seed std — real** |
| `val/task_loss` | 0.063290 ± 0.000061 | 0.063630 ± 0.000473 | 0.062822 | **0.72× seed std — flat** |
| `halt/forced_exit_rate` | 0.0003 ± 0.0003 | 0.0104 ± 0.0089 | 1.0000 | ~35× but still ~1% |
| `halt/early_exit_rate` | 0.9997 | 0.9896 ± 0.0089 | 0.0000 | still exits early ~99% |

**The second branch fired.** Depth responded to the coefficient — the move is ~5.2×
the seed std, so the halt head *is* sensitive to the penalty and the gradient path
into it is live (which independently re-confirms the T6 halting-gradient fix). But a
10× cut released only 0.46 of the 4.95 steps separating canonical MoRE from the
fixed-depth budget, and task loss did not move at all. So the ponder cost is not what
pins depth near 2: relax it by an order of magnitude and the model still chooses ~2.5
steps, and choosing those extra 0.46 steps buys nothing measurable.

**Conclusion for the paper, as revised by T11.0a.** Read together with the corrected
T10.B — where a fixed 7-step schedule is *also* indistinguishable on loss — the
finding is not "the halt head fails" but something sharper and more troublesome:
**task loss on this benchmark is flat in recursion depth over the whole range
2.0 → 7.0.** Three points now say so (2.05, 2.51, 7.00). A depth-allocation policy
therefore cannot be evaluated on this task at all, in either direction: the halting
machinery demonstrably works (depth responds to its cost term at 5.2σ), and the
benchmark cannot tell a good allocation from a bad one. `depth/allocation_error_abs
≈ 1.0` remains the only evidence about allocation *quality*, and it says the policy
does not track the complexity curriculum.

That is the honest claim, and it is a claim about the experimental design as much as
the architecture: **testing adaptive computation needs a task whose loss is sensitive
to depth.** This one is not, so MoRE's depth half is untestable here rather than
refuted. Any follow-up should first establish depth sensitivity in the baseline —
e.g. `max_depth = 2` fixed vs 7 fixed — before another halting variant is trained.


**Why no `halting = 0.0` arm.** Considered and rejected; the reason is already in the
T10.J docstring: at zero there is no pressure to exit at all, so the arm either drifts
to `max_depth` and re-measures T10.B, or it does not, and either way the dose-response
between 0.001 and 0.0001 has already shown the coefficient's influence is small
relative to the 4.95-step gap. Spending ~1.4 h of GPU to place a third point on a
curve whose slope is already known does not change any claim in the paper.

Side observation, **not** a claim: this arm has the best unsupervised routing numbers
in the phase (Hungarian 0.5654 ± 0.0500, purity 0.6495 ± 0.0117, AMI 0.5409 ± 0.0315
vs canonical 0.5538 / 0.6224 / 0.5092) — all inside seed noise, so it supports
nothing, but if a future arm needs a hypothesis, "more recursion steps give the router
more chances to differentiate" is the one to test.

**Where to look.** Arm definition and pre-registered reading: the `T10.J` entry in
`ARMS`, `automated/phase10_ablations.py`. Config: `automated/phase10_configs/ponder_cost_low.json`.
Runs: `runs/t10_more_ponder_cost_low_seed{42,43,44}__*`. Launcher log:
`automated/phase10_ponder_launch.log`. If the halt head is ever reworked, the
regression to beat is 2.5120 steps at 0.063630 — an improvement must move depth
*and* loss together.

---

## T11.0a — The reporting layer used a population std, and it changed four verdicts

Found while auditing the comparison protocol before Phase 11 (audit item 3), by
recomputing the matrix independently with `statistics.stdev` and getting different
ratios than the driver printed.

**The defect.** `mean_std()` in both reporting drivers computed
`sqrt(Σ(x−m)² / n)` — the *population* standard deviation. The seeds are a sample
from the run-to-run distribution, so the unbiased estimator is `n−1`. Dividing by
`n` understates the spread by **22% at n=3** and **12% at n=5**
(`√(n/(n−1))` = 1.2247, 1.1180). Every verdict in this project is read against a
`|gap| ≥ 2 × std` threshold, so a 22% understated std inflates every ratio by 22%
and manufactures resolutions the data does not support.

Two things make it unambiguously a bug rather than a defensible convention choice:

1. **The repo already had a convention, and the reporting layer broke it.**
   `automated/act_decision.py:181`, `automated/family_cls_ablation.py:230` and
   `automated/routing_supervision_decision.py:203` all divide by `len(vals) - 1`.
   Only the two drivers whose output was destined for the paper —
   `code/run_phase9_matrix.py` and `automated/phase10_ablations.py` — divided by
   `len`. So the ACT decision, the family-supervision decision and the
   routing-supervision decision were all read on the unbiased estimator, and the
   canonical matrix and the ablation phase were not.
2. **The bias is in the one direction this project cannot afford** — it only ever
   turns "inside noise" into "RESOLVED", never the reverse.

**Fixed** in both files: `/(len(nums) - 1)`, with `n < 2 → std = None` rather than
`0.0`, because a single seed has no spread and `0.0` reads as "perfectly
reproducible" (CLAUDE.md §4, no sentinel as a measurement). `fmt()` prints
`± N/A(n=1)`; `rule4()` and `gap_reading()` filter `None` out of the yardstick and
return *cannot judge* if nothing is left. Both reports were regenerated from the
run directories — no number was edited by hand.

### Four verdicts changed

| comparison | was | now |
|---|---|---|
| T9.1 `more − mor` | 2.16× — **RESOLVED, MoR better** | **1.93× — inside seed noise** |
| T10.B fixed_depth | 2.10× — **RESOLVED, better** | **1.88× — inside seed noise** |
| T10.E routing_supervision | 2.16× — **RESOLVED, worse** | **1.94× — inside seed noise** |
| T10.J depth response | 6.4× | 5.2× (still resolved) |

`moe − mor` at 3.62× (was 4.04×) and every "inside noise" verdict are unaffected.

**Consequence for the paper.** The canonical matrix now contains **exactly one
resolved difference**: MoR beats MoE by 0.000554, 3.62× the larger seed std. MoRE is
statistically indistinguishable from *both* single-mechanism baselines
(`more − moe` 0.30×, `more − mor` 1.93×). The previously-headline claim that MoRE
loses to MoR does not survive, and neither does either Phase 10 resolution. This is
Outcome C in the strict sense of CLAUDE.md §8 and must be reported as such.

### Second finding: the verdicts are also sensitive to the checkpoint-selection rule

`metrics.json` carries two candidate primary numbers and the protocol never declared
which one is canonical:

- `val/task_loss` — the **final-epoch** (epoch 50) validation task loss. This is the
  number every report in this repository has used.
- `best_val_loss` — the **minimum** over the 25 validated epochs
  (`engine.py:951-953`), and the epoch at which `checkpoint.pt` was written.

Both are pure task loss (`engine.py:942-950`: `val_loss` is `F.mse_loss` on the
regression head with no auxiliary term), so this is *not* a total-loss contamination
issue. It is a selection issue:

| arch | last-epoch (reported) | best-val (checkpoint) |
|---|---|---|
| MoE | 0.063260 ± 0.000153 | 0.062819 ± 0.000194 |
| MoR | 0.062706 ± 0.000112 | 0.062527 ± 0.000052 |
| MoRE | 0.063186 ± 0.000249 | 0.062892 ± 0.000216 |

| comparison | last-epoch | best-val |
|---|---|---|
| `moe − mor` | **3.62× RESOLVED** | 1.51× inside noise |
| `more − mor` | 1.93× inside noise | 1.69× inside noise |
| `more − moe` | 0.30× inside noise | 0.33× inside noise |

**Under best-val selection nothing in the canonical matrix resolves at all.** The
one surviving claim in the paper depends on a protocol choice that was never
declared, which is precisely the class of defect that invalidates a comparison.

**Ruling, and why.** **Last-epoch is canonical.** `best_val_loss` is a
minimum-of-25 order statistic, so it is biased downward by an amount that scales
with each architecture's *per-epoch validation noise* — and that noise is not
matched across architectures (MoR's best-val std is 0.000052 against MoE's
0.000194). Selecting on it therefore mixes "how good is this model" with "how noisy
is its validation curve", which is a confound; last-epoch under a fixed 50-epoch
budget with no early stopping is the same unbiased estimator for all three arms.
Every number already reported uses it, so nothing is re-derived by this ruling —
it is now *declared* rather than implicit.

**Disclosure the paper must carry**: `checkpoint.pt` is the best-val checkpoint and
is therefore **not** the model whose metrics are reported. Fixing that would mean
re-running all 15 canonical runs (~5 h) to also save a final-epoch checkpoint; the
cheaper and equally honest course is to state it, report both columns, and note
that the two orderings differ. Recorded here so the discrepancy is never discovered
by a reader instead of by us.

**Where to look.** `code/run_phase9_matrix.py:mean_std` and
`automated/phase10_ablations.py:mean_std` (both carry the full rationale in the
docstring now). Checkpoint rule: `code/more/engine.py:942-953`. Regenerated
aggregates: `code/phase9_matrix_result.json`,
`automated/phase10_ablations_result.json`. If a future driver reports `± std`, it
must divide by `n−1`; grep for `/ len(nums)` before trusting any new report.

---

## T11.0b — The blocking pre-Phase-11 audit: the protocol is sound, the reading rule was not

**What the task was.** Discharge the five open items of the standing pre-Phase-11
audit directive before the T11.1 exporter is allowed to emit paper numbers:
(3) the comparison protocol and parameter matching, (4) every paper metric,
(5) dataset and feature leakage, (6) contradictions across the governing
documents, (7) the full regression suite. Output: `PHASE11_AUDIT.md`.

**The headline, and it reverses two earlier conclusions.** Nothing measured in
this project turned out to be wrong. The *reading rule* was. Every verdict ever
produced here compared a difference of means against `max(std_a, std_b)` — one
arm's per-seed spread — and called the difference real at `≥ 2 ×`. That is not
the standard error of a difference. The correct denominator at n = 5 per arm is
`sqrt(s_a²/n_a + s_b²/n_b)`, roughly √n smaller: for MoRE − MoR the driver
printed "1.93 × seed std, inside noise" for a gap that is **3.94 standard
errors**. The heuristic was suppressing real differences, and the T11.0a ddof fix
(≈1.12× in the other direction) did not come close to cancelling it.

**What replaced it.** Not a new `k`. With five matched seeds per arm the pooled
values can be split C(10,5) = 252 ways, so the null distribution is enumerable
and the **exact two-sided randomization test** is available: deterministic, no
normality assumption, no threshold to choose, and it is what a reviewer asks for.
Implemented once, in `code/seed_stats.py`, so the exporter and every future
driver share it instead of each rolling its own `mean_std`. The module also
returns `min_p = 1/C(n_a+n_b, n_a)` — the design's resolution floor — because a
p-value sitting *at* that floor means "as extreme as this many seeds can show",
which is a different statement from "significant" and must not be conflated.

**Consequence 1 — the canonical matrix has two real differences, not one.**
MoR beats MoE (p = 0.0079) *and* MoR beats MoRE (p = 0.0159); MoRE and MoE are
indistinguishable (p = 0.659). Since MoE and MoRE are parameter-identical **to
the unit** (3,201,555 both), that third comparison is the clean isolation of
recursion, and recursion contributes nothing. The reading: adding expert routing
on top of recursion recovers nothing recursion alone gives, and costs relative to
recursion alone. Effect sizes are large (|d| 2.1–4.1) but the magnitudes are
small — MoR's edge over MoRE is 0.00048, **2.7 % of the variance the models
explain**. All three facts go in the paper together.

**Consequence 2 — two Phase 10 arms resolve after all.** `dense_routing` is
significantly *worse* than Top-1 sparse (+0.000500, p = 0.0179), which supports
the canonical routing choice on quality as well as on its 6× dispatch cost. And
routing supervision **hurts** (+0.000585, d = 2.77, p = 0.0179): driving the
router to perfect family assignment — Hungarian accuracy, AMI, purity and matched
macro-recall all exactly 1.0000 ± 0.0000 — makes task loss significantly worse.
The T11.0a retraction of "perfect routing hurts" down to "buys nothing" was
itself an artifact of the reading rule and is **withdrawn**; the stronger claim
holds. Both arms sit exactly at the n = 3 floor (1/56 = 0.0179), so neither
survives Bonferroni across six arms — the n = 3 design cannot support a
multiplicity-corrected claim at all, and seeds 45–46 on these two arms (4 runs,
≈2.8 h) would drop the floor to 0.004 and fix that.

**Item 3 — the comparison protocol passes, and more cleanly than expected.**
Flattening the three seed-42 resolved configs gives 44 fields of which **36 are
bit-identical**; the 8 that differ are `architecture`, `run_name`, `num_experts`,
`max_depth`, `adaptive_halting`, `ffn_mult`, and the two loss weights whose terms
are undefined for the arm that zeroes them (`halting` for MoE, `routing_balance`
for MoR). **No free hyperparameter differs between arms** — same `lr`,
`weight_decay`, `dropout`, `batch_size`, `epochs`, `d_model`, `num_blocks`,
`grad_clip`, both data versions, `log_interval`, and the remaining loss weights.

Parameter counts were re-derived by instantiating `MoREModel` from each run's own
recorded model block and summing `named_parameters()`; all three match the
`provenance.total_params` the run wrote at train time. MoE 3,201,555 /
MoR 3,197,710 / MoRE 3,201,555. MoE == MoRE exactly because `adaptive_halting`
is not a constructor argument — the six halt heads are allocated in MoE too and
merely receive no gradient, so MoE carries 1,542 params (0.048 %) of dead weight
and the recursion-isolating comparison is exactly matched. The MoR gap of 3,845
(0.120 %) decomposes completely: **1,285** halt heads (6 × 257 vs 1 × 257),
**1,280** router (256 × 6 vs 256 × 1), and **1,280 that the frozen note in
`canonical_spec.json` omits** — a bias-count difference, 6 × (1024 + 256) = 7,680
biases against 1 × (6144 + 256) = 6,400. The weight matrices match exactly
(6 × 2 × 256 × 1024 = 2 × 256 × 6144); only the biases cannot, at any integer
`ffn_mult`. Crucially the gap runs *against* the result — MoR wins with fewer
parameters — so MoRE's loss cannot be blamed on capacity. Also disclosed: MoR's
256-parameter router is vestigial (softmax over one logit is identically 1.0).

**Item 4 — two Rule 4 violations, both in the reporting layer only.** MoE emits
`halt/early_exit_rate = "N/A"` (correct) but `halt/early_exits = 0.0` and
`halt/forced_exits = 224450.0` as numbers: with `max_depth = 1` the code counts
every single-step token as a forced exit, so a reader sees "MoE never exits
early", which reads as a finding about halting behaviour MoE does not have. The
guard at `engine.py:875` is `if _tot_exits > 0`, which is true for MoE, and the
applicability layer that stamps `"N/A"` covers `halt/*_rate` but not
`halt/*_exits` — the same shape as every past defect in this repo where the guard
was on the wrong quantity. Second: `train/halting_supervision_loss = 0.0` in all
15 runs, because `engine.py:814` divides the accumulator by `n_batches`
unconditionally while canonical never computes the term. `0.0` for a loss invites
"the curriculum objective was satisfied perfectly". Both must become `"N/A"`, and
`test_phase3_halting.py:626` / `test_gate5.py:156` must be relaxed to accept
`"N/A"` or they will fail the fix. Not defects: `dispatch/overflow_*` (real, with
`capacity_policy = no_capacity_limit` recorded beside it) and
`dispatch/router_noise_scale` (canonical noise is genuinely none).

Also from item 4, two scoping problems the exporter must handle. **50 keys are
absent rather than `"N/A"`** for the arm they do not apply to — 24 per-family
routing metrics, 5 `expert_load` slots, the Hungarian assignment and collapsed-expert
count (all MoR), and `depth_dist/step_{2..7}_pct` (MoE) — so absent must map to
N/A and never to 0.0, or MoR shows zero per-family recall. And **no run records
R²**: `canonical_spec.json` freezes `primary_metric_floor = 0.080914` but nothing
joins it, so the paper would print 0.0632 against no scale. Against the floor all
three architectures explain **21.8–22.7 %** of target variance and the entire
between-architecture spread is **3.1 % of what they explain** — the single most
important piece of context in the results section.

**Item 4, continued — what the depth distribution says.** Read off the raw
metrics for the first time in this audit, and it is the sharpest negative result
in the project: **96.51 % of MoRE tokens exit at exactly step 2** (step 1 0.005 %,
step 3 2.85 %, steps 4–7 together 0.64 %), and per-operation average depth spans
only **1.999 (SORT) to 2.112 (AND) — 0.11 steps across sixteen operations** whose
curriculum target depths span **1–4**, not 1–7: `families.OP_TARGET_DEPTH` assigns
1 to eight operations (ADD SUB AND OR XOR NOT SHIFT_L SHIFT_R), 2 to four
(MULT DIV MAX MIN), 3 to two (MOD POW) and 4 to two (MEDIAN SORT). The 7 is
`model.max_depth`, the configuration's ceiling, and no operation ever targets it —
an earlier draft of this entry conflated the two. The corrected table sharpens the
reading rather than softening it. The target mean is **1.875**, and the model's
constant ≈2.05 is essentially that mean: the halt head is performing
*unconditional mean prediction*. That also explains the magnitude of
`depth/allocation_error_abs = 1.010` exactly — a constant-at-the-mean predictor on
this skewed table gives a mean absolute error of ≈0.90, and because half the
operations want depth 1, almost all of that error is **over-computation on the
easy operations**, not under-computation on the hard ones. So the cost of the
failure is wasted compute, not lost accuracy on MEDIAN/SORT. The honest statement is that
**the halting mechanism learned a constant depth of 2, not an adaptive policy.**
It is not broken — T10.J moved depth to 2.51 by scaling the ponder coefficient, so
gradient does reach the halt head — it simply has no incentive to differentiate,
because 1 vs 2 vs 7 steps makes no measurable difference to loss on this
benchmark. Separately: the 2.05 figure everyone quotes is
`train/avg_recursion_steps`, a **training-time** mean; there is no
`val/avg_recursion_steps`, and dropout is on in training and off in eval, so the
depth claims must either be relabelled "training-time" or a validation-pass depth
metric must be added.

**Item 5 — leakage passes 8/8, three of them non-vacuity controls.** Mutating the
step result changes 368/400 targets and **0** input features; permuting expert
labels changes 400/400 `step_experts` and **0** input features; no low-cardinality
slot maps 1:1 onto the expert label (majority-class rate 0.303); input features
are **bit-identical across E = 1 / 5 / 6**, which is the `updated_rules.md` §3
invariant that licenses comparing MoR against MoE/MoRE at all; zero
`(op, args, result)` overlap across all three split pairs. Residual
target-equals-an-argument coincidence is 33.45 % of val records and is inherent —
MAX/MIN/MEDIAN/SORT return an argument — and is made safe by the control that the
best single-feature copy scores 0.081853, *worse* than predicting the train mean
(0.080914). Pre-fix those figures were 100 % and 0.000000.

**Item 6 — the load-bearing contradiction: the frozen checkpoint rule is not the
rule that was used.** `canonical_spec.json → protocol.checkpoint_selection`
freezes "lowest `val/task_loss` among evaluated epochs" and every report used the
**final epoch** instead. The frozen note itself said the T10.H agreement between
the two "is a measurement, not a guarantee, and must be re-checked on the matrix"
— re-checked here, and **it fails**: all 15 canonical runs diverge (MoRE seed 42,
0.063307 final vs 0.063097 best). Both are pure `F.mse_loss`, so this is
selection, not contamination; `best_val_loss` is a minimum over 25 validated
epochs whose downward bias scales with each arm's validation-curve noise, and that
noise is *not* matched (MoR 0.000052 vs MoE 0.000194), which is why last-epoch is
the better rule. **But amending a frozen field is the researcher's call, not the
auditor's**, so `PHASE11_AUDIT.md` §7.1 reports the matrix under both. The
reassuring result: the qualitative conclusion is **identical under both rules** —
MoR beats MoE (p 0.0079 / 0.0397) and beats MoRE (p 0.0159 / 0.0079), MoRE ≈ MoE
(p 0.659 / 0.611) — so this is a declaration problem, not a scientific one. Either
way the paper must disclose that **`checkpoint.pt` is not the model whose metrics
are reported**.

Three smaller item-6 findings: the frozen `seed_reporting` rule is **1 ×** std
while practice used 2 ×, and read literally the frozen text states a *necessary*
condition, so 2 × was a defensible but undeclared strengthening sitting exactly
where the MoRE − MoR result lives. `subset_fraction`, `routing_mode` and
`router_noise` are absent from `config.json` and supplied by defaults that happen
to match the spec, so `resolved_config.json:_README`'s claim that every enforced
field "holds the same value here" is overstated for three of eighteen.
`canonical_spec.json` has **no null fields**, so the proxy guard is armed and all
15 runs earned `experiment_group = canonical_phase_b` through it.

**Retracted from an earlier audit note:** `num_experts`, `max_depth`,
`num_blocks`, `d_model`, `lr`, `weight_decay` and `routing_balance` were listed as
"provenance gaps". They are not. They are absent from the `provenance` sub-block
but recorded at the top level of the same `resolved_config.json` **and** in W&B,
because `engine.py:277` builds `wandb_config = {**mc, **tc, **lw, **dc}` before
adding the provenance fields. `updated_rules.md` §9 asks that the W&B run record
them; it does.

**The weakest link, disclosed:** all 15 canonical runs carry
`code_git_commit = f7166b4` *and* `code_git_dirty = true`, so the recorded commit
does not pin the code that trained them. The 15 share one code state with each
other and the 18 Phase 10 runs share one with each other, but neither is
recoverable from git alone. Commit the tree, or ship content fingerprints for the
canonical matrix the way `automated/phase10_code_fingerprint.json` does.

**Failure tracing.** If a future report calls a difference "inside noise" and it
looks too clean, check whether the driver used `code/seed_stats.py:perm_test` or
rolled its own `k × std`: the latter is ~√n too conservative. If a p-value equals
`min_p` exactly, the design is at its resolution floor and the answer is more
seeds, not a stronger claim. If MoE shows a numeric early/forced-exit count, the
fix at `engine.py:875` did not land. If a results table shows `0.0` for a MoR
per-family recall, the exporter is mapping absent to zero instead of N/A. If the
headline loss numbers ever shift by ~0.0004 with no code change, someone switched
between `val/task_loss` and `best_val_loss`.

**Where to look.** `PHASE11_AUDIT.md` (all seven items, with the 12-row
required-action table at §8). New shared statistics module:
`code/seed_stats.py` — `perm_test`, `se_of_difference`, `mean_std`, and the
resolution-floor table in its docstring. Metric defects:
`code/more/engine.py:875` (exit counts) and `:814`
(`halting_supervision_loss`). Frozen protocol fields:
`code/canonical_spec.json → protocol` (`checkpoint_selection`,
`seed_reporting`, `parameter_budget_note`, `primary_metric_floor`). Leakage
controls: `code/audit_leakage.py`.

**Verified by.** `code/run_correctness_suite.py` → **350/350 checks pass**,
Gates 1–5 (19 + 31 + 27 + 130 + 143), 152.8 s, no failures or skips.
`code/audit_leakage.py` → **8/8**. `code/seed_stats.py` reproduces the §7.1
table from the run directories, and its N/A paths return `None` rather than 0.0
at n = 1 and for all-`"N/A"` inputs. Every number in `PHASE11_AUDIT.md` was
recomputed from `runs/*/metrics.json` and `runs/*/resolved_config.json` in this
audit — nothing was copied from a prior report or changelog entry.

---

## T11.0c — Landing the audit's blocking fixes: the protocol is amended, the reading rule is now the only reading rule

**What this task did.** `T11.0b` diagnosed; this entry is the repair. Eight of the
twelve required actions in `PHASE11_AUDIT.md` §8 are closed here. Two (the
exporter's absent→N/A mapping and its R²/floor columns) are deferred to **T11.1**
because the exporter does not exist yet, and two are held pending the researcher's
decision. The audit document was also renamed `PRE_PHASE7_AUDIT.md` →
`PHASE11_AUDIT.md`; the old name predated the phase renumbering and was actively
misleading about which gate the document blocks. All four referring files were
updated in step.

**The frozen protocol was amended, deliberately and with the reasoning recorded.**
`canonical_spec.json:protocol.checkpoint_selection` moved from
`"lowest val/task_loss among evaluated epochs"` to `"final epoch"`. This is the one
change in this entry that alters a *frozen* field, so the note beside it carries the
full argument rather than a pointer: `best_val_loss` is a **minimum over ~13
evaluated epochs**, an order statistic whose downward bias grows with the arm's
per-epoch validation noise — and that noise is **not matched across arms**
(per-seed std: MoR 0.000052, MoRE 0.000216, MoE 0.000194). Selecting on it hands
the noisiest architecture the largest free improvement, which is a selection bias
that *varies by architecture*, the one kind a three-way comparison cannot absorb.
Last-epoch is noisier per run but unbiased and identically defined for all three.
The old rule's own note had required re-checking, on the matrix, whether the argmin
coincided with the final epoch as it had on the six 20-epoch T10.H runs; it does
not — all 15 canonical runs diverge — which is what forced the choice. **Verified
harmless to every finding:** all three pairwise verdicts are identical under both
rules, so this changes the declared protocol, not a result. `best_val_loss` stays
in `metrics.json` as a secondary column.

**`protocol.seed_reporting` was amended too, and this is the more consequential
one.** It had frozen "a difference smaller than the seed std is not a difference".
That sentence is the `k × std` rule, and the rule is wrong in its denominator, so
freezing it froze the defect. Replaced with the exact randomization test, with the
derivation and the resolution-floor table (0.0040 at 5v5, 0.0179 at 3v5, 0.0500 at
3v3) written into `seed_reporting_note`, including the explicit prohibition on the
paired sign-flip variant (minimum two-sided p = 2/32 = 0.0625, can never reach
α = 0.05) and the standing requirement to print `min_p` next to every p.

**`protocol.parameter_budget_note` gained the component it was missing.** The
0.120 % MoR deficit now decomposes in the file as 1,285 halt heads + 1,280 router +
**1,280 FFN biases**, with the arithmetic shown: six experts carry
6 × (1024 + 256) = 7,680 bias parameters against MoR's 6,144 + 256 = 6,400, while
the weight *matrices* match to the parameter (6 × 2 × 256 × 1024 = 2 × 256 × 6144).
The bias term is irreducible at any integer `ffn_mult`, and the whole gap runs
*against* the headline result. The note also now records that MoE and MoRE are
parameter-identical **to the unit** (3,201,555), because `adaptive_halting` is not
a `MoREModel` constructor argument, so the six halt heads are allocated under
`architecture = moe` as well and simply never receive gradient — which is what makes
MoRE-vs-MoE the exactly-matched isolation of recursion.

**Two Rule 4 violations closed in `engine.py`.**
- `halt/forced_exits` and `halt/early_exits` were **absent from the N/A list at
  `engine.py:1053`** while the four derived rates were in it, so a MoE
  `metrics.json` carried a real integer forced-exit count beside
  `halt/forced_exit_rate = "N/A"`. A count is as much a halting measurement as a
  rate, and emitting it invites an exporter to divide it by itself and re-derive
  the very rate the block refuses to state. Both keys added to that tuple.
- `train/halting_supervision_loss` was written **unconditionally** at
  `engine.py:814`, so all 15 canonical runs recorded `0.0` for a term that was
  never accumulated (canonical has `loss_weights.halting_supervision = 0.0`).
  `0.0` there is indistinguishable from "supervised, and predicted depths matched
  the curriculum exactly" — opposite meanings, one rendering. Now gated on
  `halting_supervision_enabled` (the *flag*, not the value), so a genuinely
  converged zero still prints as a number.

**The two tests the audit flagged needed no change** — checked rather than assumed.
`test_gate5.py:156` asserts *presence* of the loss-component keys, and `"N/A"` is
present; `test_phase3_halting.py:626` greps `engine.py`'s **source text** for the
key strings, which are still there. Correctness suite re-run after all edits:
**350/350, 5/5 gates PASS.**

**`automated/phase10_ablations.py` was rewired to the shared statistics module.**
Its local `mean_std` and its `rule4` (the `2 × std` reading rule) are gone; both now
delegate to `code/seed_stats.py`, so this driver, `run_phase9_matrix.py` and the
future exporter cannot drift on what "significant" means. Three design changes came
with it:
- The **seed set is per-arm and discovered from disk** (`arm_seeds()`), because
  `dense_routing` and `routing_supervision` were extended to seeds 45–46 while the
  other four arms stay at n=3. A constant would have let a partially-completed
  extension compare an n=4 arm as though it were n=3.
- The **baseline is no longer restricted to the arm's seeds.** This was tried first
  and discarded: a randomization test does not require `n_a == n_b`, and matching
  threw away two valid baseline draws while collapsing the floor from 1/56 = 0.0179
  to 1/20 = 0.0500 — at which *no* arm can reach α = 0.05. The constraint that
  motivated matched n was the old rule's single-arm-std yardstick, and it left with
  the rule.
- An arm at fewer than the five frozen seeds now **raises a `problems` entry
  naming its own resolution floor**, so a p sitting *at* the floor can never be
  read as clearing a Bonferroni correction.

**Verified:** `--report-only` reproduces `PHASE11_AUDIT.md` §7.2 exactly —
`dense_routing` p = 0.0179 (d +1.43, AT FLOOR), `routing_supervision` p = 0.0179
(d +2.77, AT FLOOR), `fixed_depth` 0.1250, `ponder_cost_low` 0.1607,
`router_noise` 0.4107, `two_blocks` 0.7500.

**The depth-curriculum range was wrong in the T11.0b entry and is corrected.**
Targets span **1–4**, not 1–7: `families.OP_TARGET_DEPTH` assigns 1 to eight
operations, 2 to four, 3 to two, 4 to two. The 7 is `model.max_depth`, a ceiling no
operation targets. This *sharpens* the finding. Target mean is **1.875**, the
model's learned constant is **≈2.05**, so the halt head is doing unconditional
**mean prediction** — and that reproduces `depth/allocation_error_abs = 1.010`
arithmetically (a constant at the mean gives ≈0.90 MAE on this skewed table).
Because eight of sixteen operations want depth 1, essentially all of the error is
**over-computation on easy operations**, not under-computation on MEDIAN/SORT. The
cost of the failure is wasted compute, not lost accuracy.

**Still open, and deliberately not actioned here.**
- **Item 9, `code_git_dirty: true` on all 15 canonical runs.** Committing the tree
  retroactively defines "the code the paper's numbers came from", so it is not an
  unattended action.
- **T10.D `trainable` / T10.G batch size.** Recommended for closure by *narrowing
  the task's scope in writing* rather than by GPU: CLAUDE.md §2 records the
  L2-on-trainable-noise-scale mechanism as unproven, so that arm would vary two
  things at once, and batch size moves throughput rather than a claim. Not written
  in yet — narrowing a pre-registered scope is a scientific decision.
- **Item 10** (three absent `config.json` literals) held until the seed-45/46 runs
  finish: writing them mutates `config_hash` and the one-field diff check compares
  live arms against the canonical baselines.
- **Item 11** (delete the dead `metrics.evaluate_paper_metrics`, 109 lines,
  unreferenced, unpack expects 11 values where the model returns 13) held for the
  same reason — the remaining arms import `metrics.py` on launch.

**Failure tracing.** If a Phase 10 verdict looks wrong, the test itself is
`code/seed_stats.py:perm_test` and the arm/baseline vectors are assembled in
`phase10_ablations.py:collect()`; the per-arm `primary` dict records `arm_seeds`,
`baseline_seeds`, `n_perms` and `min_p`, so a suspicious p can be re-derived by
hand. If a MoE metrics file shows a halting number that should be N/A, the list is
`engine.py:1053`. If a checkpoint-selection number disagrees with a report, the
declared rule is `canonical_spec.json:protocol.checkpoint_selection` and its note
explains which column is primary.

**Where to look.** `code/canonical_spec.json` (`protocol` block: three amended
fields, each with its reasoning in the adjacent `_note`); `code/more/engine.py`
(`:814` supervision gating, `:1053` halting N/A list);
`automated/phase10_ablations.py` (`arm_seeds`, `rule4`, `collect`);
`code/seed_stats.py` (unchanged, now the single source);
`PHASE11_AUDIT.md` (renamed, §8 action table).

**Verified by.** `run_correctness_suite.py` 350/350, 5/5 gates PASS, after all
edits. `canonical_spec.json` re-parsed as JSON. `phase10_ablations.py
--report-only` reproduces the audit's §7.2 p-values to the digit.

---

## T11.0d — The framing ruling, and three tasks closed without running them

**What this task did.** Recorded two researcher decisions that shape everything
Phase 11 writes, and closed three Phase 10 tasks — two by narrowing their written
scope, one by scoping it down to a 1-epoch guard test. No GPU was spent and no
measurement changed. This entry exists because both decisions are the kind that
look arbitrary six months later unless the reasoning is on disk.

**A new TASKS.md marker: `[✗]` — closed WITHOUT running it.** The legend had `[x]`,
`[~]`, `[ ]`, `[!]` and no way to say "we decided not to do this, and here is why".
The absence mattered: T10.D and T10.G had been sitting at `[ ]` with explanatory
prose, which is indistinguishable from a backlog item nobody got to. `[✗]` means the
scope was deliberately narrowed with the reason inline. It is not a failure and not
a silent drop. **Never convert a `[✗]` to `[x]`** — if the arm is later run it gets
a new task, so the original decision stays legible.

**T10.D `[✗]` — the `trainable` router-noise arm will not be run.** What was
measured stands: `none → fixed_annealed` is not significant, exact p = 0.4107, so
the canonical `router_noise = none` chosen on principle at T4 is also empirically
free. The third arm is refused because it **cannot answer the question the task is
named for**: CLAUDE.md §2 records the L2-on-trainable-noise-scale mechanism as
unproven, so a `trainable` run varies the noise schedule *and* introduces that
machinery simultaneously. Neither outcome would be attributable — a win could not
be credited to trainable noise, a loss could not be blamed on it. A one-field
ablation whose one field is two fields is not a one-field ablation. T10.D's written
scope is now "none vs fixed-annealed", and the paper says the trainable variant was
excluded by design, not that it was tried and failed.

**T10.G `[✗]` — neither supplementary sweep will be run.** The capacity half is
already answered by T10.C (`num_blocks 1 → 2`, ~2× recursive parameters): not
significant, p = 0.7500. Doubling the model changes nothing measurable, so a wider
capacity sweep would spend GPU re-deriving a null at more points. Batch size is
closed on different grounds — it moves **throughput, not a claim**, and it cannot
enter the headline table at all because `enforced_fields.batch_size = 768` is
frozen, so any other value is non-canonical by definition. With a single consumer
laptop GPU as the binding constraint, wall-clock spent characterising throughput is
wall-clock not spent on arms that ask something about MoRE.

**T10.I scoped to a 1-epoch proxy.** Its `Verify` clause — "run resolves to
`variant = ffn_mult_4` and the Gate 0 guard refuses `canonical_phase_b`" — is a
statement about the **guard**, and the guard fires in
`run_context.assert_not_silent_proxy()` before the first optimizer step. One epoch
discharges it as completely as fifty, at ~1/50 the GPU. The *scientific* question
(what MoR does at 82% under budget) is not needed, because the parameter-budget
defence is already stronger without it: canonical MoR wins on `val/task_loss` with
**fewer** parameters than MoRE (3,197,710 vs 3,201,555), so MoR's result cannot be
attributed to capacity, and an under-budgeted arm would answer a question nobody
asked. The proxy must be stamped `experiment_group = exploratory`, must never appear
in a loss table, and doubles as the negative test T11.1 needs for the exporter's
refusal path.

**THE FRAMING RULING — recorded in full at `TASKS.md` T11.2, binding on T11.2,
T11.3 and the paper draft.** The corrected reading rule turned Phase 9 from "one
resolved difference" into "MoR beats both MoE and MoRE", and that invites a framing
the paper must not take. The ruling, in five parts:

1. **The subject of the paper is MoRE.** MoR outperforming it is a result *within*
   the paper, not the thesis. "MoR is good" is not a contribution — recursive
   depth-sharing is established prior work. The novelty is the composition of
   learned expert routing with adaptive recursive reuse.
2. **Do not restructure around MoR because it won.** Rewriting the subject to match
   whichever arm scored best is choosing a conclusion after seeing the data. The
   comparison is reported at full strength and then *interpreted*, not promoted.
3. **MoR's win is confounded three ways, and each is a limitation on the
   CONCLUSION, not an excuse for MoRE.** (a) **Scale** — 3.2 M parameters,
   `d_model = 256`, one block, one RTX 4060 Laptop. Six experts of width 1024 each
   see ~1/6 of the tokens where MoR's single width-6144 FFN sees all of them, so at
   this width partitioning may cost more than specialisation buys; nothing wider was
   tested. (b) **The dataset** — `more6-v1` is synthetic and custom, generated by
   `data/script.py` with Python's `random` over 16 hand-chosen operations and a
   hand-authored depth curriculum, so both MoRE mechanisms are asked to discover
   structure placed there by construction, on a difficulty profile we chose. All
   three architectures explain only 21.8–22.7% of target variance against the
   frozen floor 0.080914, and the whole between-architecture spread is ~3% of that —
   a regime where the task, not the architecture, binds. (c) **The evaluation** — a
   single scalar regression loss at one fixed split, 5 seeds, 50 epochs, one
   optimizer setting frozen as VALIDATED-STABLE rather than swept per architecture.
   A metric rewarding only final-answer accuracy is structurally incapable of
   crediting interpretable routing or per-input compute allocation, which is what
   MoRE is for.
4. **Do not downplay MoRE, and do not inflate it.** The negative findings are
   reported plainly — recursion adds nothing at matched parameters; the halt head
   learned an unconditional ≈2 steps against a target mean of 1.875. What must not
   happen is presenting those as settled properties of the architecture when the
   compute ceiling is the honest reason the design space went unexplored: one laptop
   GPU, so no width sweep, no per-architecture tuning, no larger `d_model`, no
   second dataset. The negative result stands as *what was observed at this scale*
   (CLAUDE.md §8, Outcome C).
5. **Register.** Measured facts stay visibly separated from interpretation, and the
   confounds live in the limitations section with their own evidence rather than as
   hedging beside individual numbers.

**Git policy, recorded so it is not re-litigated.** No commit and no push until all
experiments have run and results are logged, i.e. until the repository is ready for
paper writing. This closes `PHASE11_AUDIT.md` §8 item 9 as **deferred, not
ignored**: the 15 canonical runs carry `code_git_dirty: true`, so
`code_git_commit` does not yet identify the code that produced them, and the fix is
a single commit at the end of Phase 11 whose hash is then written into the results
files. Until then, provenance for those runs rests on `config_hash` plus
`resolved_config.json`, and the results document must say so.

**Failure tracing.** If a Phase 11 document reads as a paper about MoR, or if a
limitation has quietly become an excuse, the binding text is the five-part ruling at
`TASKS.md` T11.2 — it is the authority, this entry is the record of why. If a `[✗]`
task is questioned, its inline paragraph carries the argument; the legend at the top
of `TASKS.md` defines the marker.

**Where to look.** `TASKS.md` (legend, T10.D, T10.G, T10.I, T11.2 framing ruling);
`PHASE11_AUDIT.md` §8 item 9 (git provenance, now deferred by policy);
`code/canonical_spec.json:enforced_fields.batch_size` (why a batch-size sweep can
never be canonical).

**Verified by.** Documentation-only task — no code path changed, so no gate was
re-run. `TASKS.md` marker counts after the edit: 8 `[ ]`, 2 `[✗]`, and the T10.I
entry still `[ ]` pending its 1-epoch proxy run.

## T11.1 — The results exporter: 15 rows admitted, 105 directories refused, and no number typed by hand

Every results table this project produced before today was assembled by reading
`metrics.json` and typing. That is how `archive/pre_finalization/` came to hold
tables that mixed a 3-epoch proxy with a 20-epoch run, printed a `0.0` placeholder
as a measurement, and quoted an `expert_entropy` from a run whose dataset version
no longer exists. None of that was dishonesty; it was transcription.
`code/export_results.py` removes the transcription step. It is now the only
sanctioned path from a run directory to a number in a table, a figure or the paper.

**What it does, in the order the rules bind.**

*Admission (Rule 1).* A directory becomes a row only if it certifies itself:
`provenance.experiment_group == canonical_phase_b`, `variant == canonical`,
`seed_declared == True`, `resolved_seed` in `canonical_spec.json:seed_set`, and
`dataset_version` / `train_split_version` equal to the spec's enforced values. The
primary metric must be present and numeric. Of **120** directories under `runs/`,
**15 admitted and 105 refused**, each refusal recorded with the field that failed
and written into section 5 of the Markdown. Refusals are part of the result: an
exporter that silently skipped them would be indistinguishable from one that found
nothing wrong.

*Consistency (Rule 2).* Admitted rows must share one `dataset_version`, one
`train_split_version` and one `code_git_commit`; no `(architecture, seed)` cell may
appear twice; and within an architecture the non-seed config fields must be
bit-identical. Any violation aborts with exit 2 and writes nothing — a table that
silently spans two dataset versions is worse than no table, because it looks
finished. The seed-blind check is the dangerous one to get wrong in either
direction: raw `config_hash` cannot do it (the seed is inside the hash, so all five
seeds of an arm hash differently by construction), so the comparison runs over a
dotted-path flatten of `resolved_config.json` minus an explicit exclusion list
(`SEED_KEYS`). That list is load-bearing and each entry is justified in place: the
seed itself, names that embed the seed, `data.subset_seed` / `training.subset_seed`
(the same two `run_phase9_matrix.py:134` excludes), and per-run identity/timestamps.
An over-broad list would let a real config difference through, which is this
check's only failure mode. The first run of the exporter did exactly that in
reverse — it reported all 15 runs as config-divergent because `logging.run_name`,
`provenance.run_name` and `provenance.seed` were not yet excluded. The check
failing loudly on its own first invocation is the behaviour we want.

*Absent vs "N/A" vs 0.0 (Rule 3, audit item 5).* 130 scalar keys in the union
across admitted runs — the union, never the intersection, because taking the
intersection would make the architecture-inapplicable slots vanish from the table,
which reads as "not measured" rather than "does not apply". **550 cells export as
`N/A`**: keys absent for an architecture (MoE has no `depth_dist/step_2..7_pct`;
MoR has no per-expert load, no per-family precision/recall/f1 or their `_matched`
variants, no collapse count, no Hungarian assignment), keys where `engine.py`
itself wrote the string `N/A`, and the four MoE depth constants suppressed at the
table boundary. `n_na` and `n_bool` are aggregate columns, not footnotes, so a mean
over 3 of 5 seeds cannot look like a mean over 5. Non-scalar metrics (confusion
matrix, Hungarian assignment vector) are not table cells but are carried in
`results.json` under each run's `metrics_nonscalar` rather than dropped, and
section 3 says so.

*Verdicts (Rule 4).* Pairwise comparisons come from `seed_stats.perm_test` only,
and every p-value is exported beside `min_p`. Reproduces `run_phase9_matrix.py` to
the digit: MoRE−MoE p=0.6587 (d −0.36), MoRE−MoR p=0.0159 (d +2.49), MoE−MoR
p=0.0079 (d +4.13), floor 0.0040. The primary metric is read under the amended
`protocol.checkpoint_selection` — **last epoch** — and `best_val_loss` is exported
as an explicitly labelled secondary column so the selection-bias argument stays
checkable without ever being the headline.

*Audit item 6.* R² is **derived here and stored nowhere**:
`1 − val/task_loss / primary_metric_floor` against the frozen 0.080914. MoE 0.2182,
MoR 0.2250, MoRE 0.2191. It is in the headline table because it is the single most
important piece of context for reading the matrix, and the piece most easily lost
when a table shows six decimals of loss and nothing else.

*Ablations are ingested, not re-derived.* `automated/phase10_ablations_result.json`
owns arm discovery, per-arm seed sets and per-arm resolution floors; reimplementing
that here would create a second definition of the ablation table, which is the exact
failure this module exists to prevent. The arms land in a separately headed
`EXPLORATORY, NOT CANONICAL` section that names its own reading rule and floor, and
are never merged into sections 1–2 (CLAUDE.md §6).

**Outputs.** `results/results.csv` (long format, one row per run, provenance then
every metric — deliberately contains no aggregates and no verdicts, so it cannot
encode a reading rule and a reviewer can re-analyse from it),
`results/results_aggregate.csv` (one row per metric × architecture with `n_numeric`,
`n_na`, `n_bool`), `results/results.json` (raw per-seed vectors beside every
aggregate, so a stale aggregate cannot outlive the numbers it came from), and
`results/results_tables.md` (five sections, tables grouped mechanically by
metric-key prefix rather than by a curated list — a curated list is a place to
quietly drop an unflattering number). Read-only with respect to `runs/`.

**Verify clause, passed.** `runs/phaseB_moe__seedNA__f83aa42f` — the invalidated
undeclared-group run with `provenance.architecture = None` — yields
`experiment_group='exploratory' != 'canonical_phase_b'` in the refusal list and no
row, and does not raise. Every admission path returns a reason string rather than
propagating an exception, because a refusal that arrives as a traceback is an
outage, not a refusal, and would be indistinguishable from "there were no bad runs".

**Also landed here: audit item 2 is now complete.** `run_phase9_matrix.py` no longer
carries its own statistics. Its `mean_std` (`:183`) delegates to `seed_stats`, and
its `gap_reading` (`:202`) is now the exact randomization test. That change forced a
signature change: the old function took the two aggregate dicts and compared
`|gap|` to `max(std_a, std_b)`, whereas a permutation test needs the individual
draws — so `aggregate()` now stores a `raw` per-seed vector beside every aggregate
and `report()` passes `agg[x]['raw'][...]` instead of `agg[x]['metrics'][...]`. The
docstring records why the superseded rule was wrong *in the direction of silence*:
`max(std_a, std_b)` is one arm's seed spread, not the standard error of a
difference, so at five seeds per arm it was ~√5 too conservative and printed
`more − mor` as "1.93× the larger seed std → inside seed noise" when the gap is
3.94 standard errors, exact p = 0.0159. A too-conservative rule feels safe, which is
why it went unexamined for the whole project. The module docstring's Rule 4 and the
report's trailing prose were rewritten to match; the report now prints
`p`, `min_p`, the permutation count and an `AT-FLOOR` marker.

**Failure tracing.** If an expected run is missing from a table, look at section 5
of `results/results_tables.md` or `refusals` in `results.json` — the reason names
the field. If the exporter aborts, the message names which of the four consistency
invariants broke. If a number in the paper disagrees with a run directory,
regenerate rather than edit: nothing in `results/` is hand-maintained. If a new
metric appears as `N/A` for every architecture, check whether `engine.py` writes it
as a string; if it appears as `N/A` for one architecture only, that is Rule 3
working. If a legitimate config field starts tripping the seed-blind check, do not
extend `SEED_KEYS` reflexively — that check is the only thing standing between a
config sweep and a seed std.

**Where to look.** `code/export_results.py` (`admit` for admission,
`consistency_errors` for the four invariants, `cell` for the single N/A boundary,
`aggregate` / `pairwise` for the statistics, `_write_md_tail` for sections 3–5);
`code/seed_stats.py` for every statistic; `code/canonical_spec.json:protocol` for
the primary metric, checkpoint rule and floor; `code/run_phase9_matrix.py:183,202`
for the rewired reading rule.

**Verified by.** `python code/export_results.py --check` (15 admitted, 105 refused,
all consistency invariants pass); the full export writes four files; pairwise
p-values identical to `run_phase9_matrix.py --report-only`; correctness suite
**350/350, 5 gates PASS** after the matrix-driver rewiring.

## T10.E / T10.F extension — seeds 45–46 landed, and both significant ablation arms are now Bonferroni-robust

The researcher's decision at the T11.0b audit was to make the two significant
Phase 10 arms survive a multiple-comparison correction rather than leave them
sitting on the resolution floor. `dense_routing` and `routing_supervision` were
run at seeds 45 and 46 (four runs, ~54 min wall clock each pair on the RTX 4060
Laptop), taking both arms from 3-vs-5 to 5-vs-5.

**Result: both survive.** The floor drops from 1/C(8,3) = 0.0179 to
1/C(10,5) = 0.0040, and both arms land at **p = 0.0079** against a Bonferroni
threshold of 0.05/6 = **0.0083** — clearing it by 0.0004.

| arm | n | gap vs canonical MoRE | Cohen d | p (exact) | floor | Bonferroni |
|---|---|---|---|---|---|---|
| `routing_supervision` (T10.E) | 5v5 | +0.000565 | +2.89 | 0.0079 | 0.0040 | survives |
| `dense_routing` (T10.F) | 5v5 | +0.000529 | +1.90 | 0.0079 | 0.0040 | survives |

Both are still *worse* than canonical MoRE, and the direction, effect sizes and
mechanism stories are unchanged from n=3 — only the resolution improved. So the two
architectural choices CLAUDE.md §2 fixes on principle are now supported on the
primary metric at corrected significance: **Top-1 sparse dispatch beats
evaluate-all-and-blend**, and **supervising the router toward the oracle family
index actively trades against the task** even though it drives every routing metric
to exactly 1.0000.

**A margin of 0.0004 must be printed, not inferred.** `rule4()` in
`automated/phase10_ablations.py` previously appended a Bonferroni note only in the
AT-FLOOR case, which was adequate when every significant arm was hopeless against
the corrected threshold. Now that two arms clear it narrowly, the note is emitted
for every significant arm and states the direction explicitly — `survives Bonferroni
at 6 arms (alpha=0.0083)` or `fails …` — with the arm count and alpha taken from
`len(ARMS)` rather than written as a literal, so adding a seventh arm re-derives the
threshold instead of silently invalidating the sentence. The exporter's section 4
carries the same column.

**One trap worth recording.** The result JSON written when the background launch
finished was in the OLD format (`primary` absent, `n=3`, no per-arm test), because
the launcher process had been started *before* the T11.0c driver rewrite and was
still running the previously-imported module in memory. The stale file was
faithfully ingested by the exporter, which then published a 3-seed ablation table
alongside a 5-seed canonical one. **After any long background launch whose driver
source changed mid-flight, re-run `--report-only` before trusting the result JSON,
and re-run the exporter after it.** The file's own `generated` timestamp is the
tell; the code fingerprint pinned across arms does not catch this, because the arms
themselves were fine — it was the *reporting* that was stale.

**Failure tracing.** Arm p-values, floors and Bonferroni verdicts:
`automated/phase10_ablations.py:rule4` and `collect()` (per-arm seed discovery via
`arm_seeds()`, which reads the disk rather than a constant, so new seeds are picked
up with no code change). The published tables: `results/results_tables.md` §4 and
`results/results.json:ablations_exploratory`, both regenerated by
`code/export_results.py`.

**Verified by.** `python automated/phase10_ablations.py --report-only` (all six arms,
n printed per arm, no problems reported: every arm still differs from the canonical
baseline in exactly one config leaf, verified on the `resolved_config.json` that
actually trained, and all arms share one code fingerprint); exporter re-run and
section 4 regenerated from the fresh JSON.

## T10.I — The under-budgeted MoR arm, discharged by a 1-epoch proxy because its question is about the guard

T10.I asks what MoR looks like at MoRE's *per-FFN* multiplier (`ffn_mult = 4`)
rather than at the multiplier that equalises total FFN hidden width
(`ffn_mult = 24`, canonical). Its `Verify` clause, read literally, is a statement
about the **guard**, not about task loss: *the run resolves to `variant =
ffn_mult_4` and Gate 0 refuses `canonical_phase_b` for it.* Both facts are settled
in `run_context.assert_not_silent_proxy()` before the first optimizer step, so a
1-epoch run answers it as completely as fifty would at ~1/50 the GPU. The
researcher's ruling at T11.0d scoped the task accordingly.

**Why the 50-epoch science was not needed.** The scientific question — what MoR does
at 82% under budget — would exist to defend the parameter budget against a reviewer
who suspects MoR won on capacity. That defence is already stronger without the arm:
canonical MoR achieves the **lowest** `val/task_loss` with **fewer** parameters than
MoRE (3,197,710 vs 3,201,555, −0.120%). The budget residual runs *against* the
result, so an under-budgeted arm answers a question nobody is asking.

**Half 1 — the variant resolves.** `runs/t10i_mor_ffn_mult_4_proxy_seed42__f2e3e60d`,
1 epoch, ~1 min: `provenance.variant = ffn_mult_4`, `model.ffn_mult = 4`,
`total_params = 571,150`. That is the 567k-class count, **82.2% short** of MoRE, which
confirms `--ffn_mult` reaches the model rather than only the config file. Stamped
`experiment_group = exploratory`, `resolved_epochs = 1`. Its `val = 0.0733` is
recorded as evidence the run trained and is **not a measurement**; the run may never
appear in a loss table.

**Half 2 — Gate 0 refuses the canonical claim.** The same command with
`--experiment_group canonical_phase_b` is refused before the first optimizer step and
enumerates all three violations independently rather than stopping at the first:
`epochs: canonical requires 50, run has 1`; `ffn_mult: canonical mor requires 24,
run has 4`; and `variant='ffn_mult_4' -- this run has at least one ablation active,
so it may not claim experiment_group='canonical_phase_b'`. No run directory is
created. The third is the check this task exercises: even at 50 epochs with every
other enforced field canonical, the variant tag alone blocks the claim, which is
what makes "label it as the ablation it is" enforceable rather than a convention.

**One thing this run does NOT cover.** It was expected to double as the exporter's
variant-path negative test. It does not: the exporter checks `experiment_group`
first, so the proxy refuses on `experiment_group='exploratory' !=
'canonical_phase_b'` and the variant check never executes. The ordering is right —
the group is the canonical *claim*, and a run that makes no claim needs no further
examination — but it means no directory under `runs/` exercises the exporter's
variant branch, and none ever will while Gate 0 refuses that combination at launch.
That branch is defence in depth against a future hand-edited provenance block, not a
path with live coverage. Recorded here so nobody later reads the branch as tested.

**Failure tracing.** The refusal text and its three checks:
`code/more/run_context.py:assert_not_silent_proxy()`, reading
`code/canonical_spec.json:enforced_fields` and `architecture_variants`. If a future
ablation is silently admitted to a canonical table, that function and the
`variant != canonical` clause inside it are the first place to look. The exporter's
independent second line of defence: `code/export_results.py:admit()`.

**Verified by.** Both commands above, run back to back on the same interpreter;
provenance read from the written `resolved_config.json`, not from stdout; exporter
re-run (15 admitted, 106 refused — the new proxy is refusal 106).

---

## T11.1 (extension) — Depth on held-out data: the metric every depth claim needed and no run had

**What was wrong.** Every depth number this project has ever reported was
accumulated inside the *training* loop. `depth/allocation_error_abs` and
`depth/allocation_error_rel` are written at `engine.py:974` from `expected_depth`
tensors reduced at `engine.py:608`, inside the batch loop, i.e. under dropout, on
training data, with the halt head being updated between batches;
`train/avg_recursion_steps` and the whole `halt/*` block are the same. The sentence
the paper wants to write — CLAUDE.md §4's sanctioned phrasing, "the model learns to
allocate recursion depth in accordance with the predefined operation-complexity
curriculum" — is a claim about the *trained* model on *held-out* data. A
training-time average over the final epoch is not a measurement of that. Nothing in
the repo measured it. This was audit item 7, and it was the last non-cosmetic item
open before Phase 11 proper.

**Methodology.** Two halves, deliberately kept separate.

*Half one, for every future run.* `engine.py`'s validation pass already called
`model(x_v, sm_v, se_v, so_v)` and unpacked `val_expected_depth, val_halt_stats`
into names it never used — the quantities were computed on the val set and thrown
away every epoch. The accumulation is now wired: `val/avg_recursion_steps`,
`val/depth_allocation_error_abs`, `val/depth_allocation_error_rel`,
`val/forced_exit_rate`, `val/early_exit_rate`, `val/mean_remainder`. Cost: zero
extra forward passes. The depth block sits *before* the
`if first_route is None or depth_exits is None: continue` guard, which matters —
MoR has `first_route = None` and would otherwise have been skipped for a metric it
genuinely has. `depth_allocation_error` is the same helper the training loop calls,
so the two passes cannot acquire different definitions. The training-time keys are
untouched: these are two different measurements of two different things, not a
correction of one by the other, and a reader who finds both in one table needs both
to still be there.

*Half two, for the 37 runs that already exist.* Re-training the canonical matrix
for a metric that requires no training would have cost hours of GPU for nothing, so
`code/eval_val_depth.py` recovers the same quantities offline from each run's saved
`checkpoint.pt`, writing a `val_depth_offline.json` sidecar into the run directory.
`metrics.json` is never modified — it is the artifact of the run, and a number
computed weeks later by a different script does not belong inside it. Model
reconstruction mirrors `engine.py:202` field for field (including the `fixed_depth`,
`routing_mode`, `router_noise`, `ffn_mult` and `num_families` defaults); the val
loader mirrors `engine.py:194`; data paths resolve against `code/`, not cwd, because
that is what `../data/val.jsonl` in a `resolved_config.json` means.

**The caveat that had to be printed, not implied.** `checkpoint.pt` is the
*best-validation-loss* checkpoint, while the canonical primary metric is *last
epoch*. The offline depth numbers therefore describe the same run at a different
point in training than the headline loss. Rather than bury that, `measured_at:
"best_val_checkpoint"` is written into every sidecar, `results_tables.md` §2b states
it in the section header and again in prose ("a depth figure and a loss figure from
this document may not be captioned as coming from one model state"), and
`results.json` carries `offline_depth_note`. Anyone assembling a figure caption from
these files is told twice.

**What it changed scientifically.** On held-out data, MoRE spends significantly more
depth than MoR (2.3531 ± 0.1087 vs 2.1283 ± 0.0188 steps; +0.2248, d = +2.88,
p = 0.0079, floor 0.0040) while its absolute depth allocation error is
statistically indistinguishable (0.9940 ± 0.1054 vs 0.9423 ± 0.0266; +0.0517,
d = +0.67, **p = 0.3571**), and its relative error is significantly *worse*
(0.6934 ± 0.0931 vs 0.5939 ± 0.0225; +0.0995, p = 0.0159). The extra recursion buys
no better agreement with the curriculum. Both architectures early-exit essentially
always (≥ 0.9986), so nothing is being forced at `max_depth`. The T10.J arm reads
the same way on held-out data: the 10× lower ponder coefficient raised depth
2.3810 → 2.7721 and made allocation error *worse* (1.0282 → 1.1495) while the task
loss did not follow (p = 0.1607) — precisely the pre-registered branch "if depth
rises but loss does not follow, the halt head is not learning anything useful and
Outcome C stands for the depth half."

**Why the numbers are ingested and not quoted.** `export_results.py` folds the
sidecar into each admitted row under its `val_offline/` prefix (`admit()`, beside
the `prov/total_params` injection), aggregates it like any other scalar, prints
§2b, and emits `pairwise_offline_depth` as a *separate* block from `pairwise` so no
consumer can iterate one list and silently mix a last-epoch loss test with a
best-checkpoint depth test. `phase10_ablations.py` folds it in at `metrics_of()`
and prints its own held-out depth group. Neither rediscovers the arms or recomputes
the statistic; there is still exactly one definition of each.

**Failure tracing.** If a held-out depth number looks wrong: `val/*` keys come from
the accumulation block in `engine.train()`'s validation pass (declared as
`val_depth_log` next to `paper_log_dict`, merged into `log_dict` right after it);
`val_offline/*` keys come from `code/eval_val_depth.py:evaluate`. If a `val_offline`
number disagrees with the matching `val` number for the same run, the most likely
cause is not a bug but the checkpoint: one is the best-val model, the other the
last-epoch model. If MoE shows anything other than `N/A`, the `adaptive` gate in
`evaluate()` (`max_depth > 1 and adaptive_halting and not fixed_depth`) has been
weakened — at `max_depth = 1` a zero there reads as perfect allocation, which is the
sentinel-as-measurement failure CLAUDE.md §4 forbids. If a sidecar is missing,
nothing breaks: the keys are absent, which the exporter's absent-vs-N/A boundary
already renders as `N/A`, and §2b says outright that until the sidecars exist every
depth number in the document is training-time and any held-out claim is unsupported.

**Where to look.** `code/eval_val_depth.py` (offline harness, module docstring
carries the caveat); `engine.py` validation pass (live accumulation, and the N/A
list for `max_depth ≤ 1` that now covers the six `val/*` twins);
`code/export_results.py` `OFFLINE_DEPTH_KEYS` / `_write_md_depth`;
`automated/phase10_ablations.py` `SCALARS` + `metrics_of` + `SECONDARY`.

**Verified by.** Three 1-epoch smoke runs (MoRE real values, MoR real values, MoE
all six keys `N/A`); `run_correctness_suite.py` **350/350, 5/5 gates PASS** after
the deletion below; `eval_val_depth.py` 15/15 canonical + 22/22 Phase 10 arm
sidecars written; exporter re-run (15 admitted, 137 keys, 580 N/A cells) and
`phase10_ablations.py --report-only` re-run, both agreeing with the hand-checked
permutation tests to the digit.

---

## T11.1 (extension) — Audit items 10 and 11, and the deletion of `evaluate_paper_metrics`

**Audit item 10 — three fields `config.json` claimed to hold and did not.**
`canonical_spec.json:enforced_fields` pins 18 fields, and `config.json:_README` says
"every field that `canonical_spec.json:enforced_fields` pins now holds the same
value here". For `model.routing_mode`, `model.router_noise` and
`data.subset_fraction` the value was not there to hold: `load_config()` supplied
them from code defaults that happened to match the spec. The claim was true of the
*resolved* config and false of the file's own text. All three are now written out
literally.

*Why this was safe to land at the end of the project, and how that was checked.*
The obvious objection is that editing `config.json` mutates `config_hash` and
therefore breaks comparability with the 15 completed canonical runs. It does not:
`config_hash()` hashes the *resolved* config (`run_context.py:82`), and
`load_config()` injects those three defaults before the hash is taken, so the
default config hashes to `b7b17489cdc2` both before and after the edit — verified
directly. What *did* move the hash was an earlier draft of this change that also
added explanatory lines to the `_README` list: `config_hash()` strips only
`provenance` and `logging`, so **the prose in `config.json` is inside the hash**,
and `_README` is present in every `resolved_config.json` on disk. That is worth
knowing before anyone reformats a comment in that file. The explanation lives here
instead, and the `_README` text is byte-identical to what the 15 runs recorded.
`halting_mode` remains absent from both files by design — `resolve_halting_mode()`
(`code/more/config.py:410`) derives it, and that helper is the guard's own source,
so there is nothing to duplicate.

**Audit item 11 — the stale T10.J pre-registration.** Already annotated in place at
`automated/phase10_ablations.py:101`: the pre-registration text stands verbatim,
with a bracketed correction stating that the 2.10× was a population-std artifact,
that the corrected figure is 1.88×, and that under the exact randomization test
T10.B is p = 0.125, i.e. not significant. Confirmed present and left alone — a
pre-registration that gets rewritten after the fact is no longer a
pre-registration. `TASKS.md`'s reference to the T8.0b driver was pointing at
`automated/routing_supervision.py`, which does not exist; corrected to
`automated/routing_supervision_decision.py`.

**The deletion.** `metrics.evaluate_paper_metrics` (109 lines) is gone, along with
its import at `engine.py:31`. It was a second implementation of the validation-pass
metric accumulation, referenced by nothing — `engine.py` imported it and never
called it — and its `model(...)` unpack expected 11 return values while
`MoREModel` returns 13, so calling it would have raised rather than provided a
second opinion. Every number in every run directory came from the inline loop in
`engine.train()`. A comment now stands where the function was, naming the four
helpers to reuse (`routing_accuracy_from_confusion`,
`permutation_invariant_routing_metrics`, `compute_token_exit_depths`,
`depth_allocation_error`) so that the next person wanting an offline harness
reuses the live definitions instead of forking them again. The reason this matters
is recorded earlier in this file: the last time two copies existed, a tolerance
guard was added to the copy that never ran.

**Failure tracing.** If a canonical run is suddenly refused by the Gate 0 guard
after someone edits `config.json`, compare `_effective(resolved)`
(`run_context.py:162`) against `canonical_spec.json:enforced_fields` — that pairing,
not the file text, is what the guard reads. If `config_hash` changes without any
scientific field changing, look for an edit to `_README`.

**Where to look.** `code/config.json` (`model.routing_mode`, `model.router_noise`,
`data.subset_fraction`); `code/more/run_context.py:82` (`config_hash`, and what it
does *not* strip); `code/more/metrics.py` (the note where the dead function was);
`automated/phase10_ablations.py:101` (the annotated pre-registration).

**Verified by.** `config_hash(load_config('config.json'))` = `b7b17489cdc2` before
and after; `_effective()` vs `enforced_fields` mismatch list empty;
`run_correctness_suite.py` **350/350, 5/5 gates PASS** after the deletion.

---

## T11.1 (extension) — ARCHITECTURE.md brought back in sync with the repository

**Why this counts as a defect and not housekeeping.** `CLAUDE.md` tells every
incoming agent to read `ARCHITECTURE.md` *first*. It had drifted to a pre-Phase-8
state, so the first thing a new agent learned was wrong in ways that would change
its decisions:

- parameter counts **6,359,069 / 1,098,259** — the pre-T8.3 figures, off by 2× and
  3× respectively, and they were the basis of a "the comparison is not
  parameter-matched" defect entry that has since been resolved to a 0.12% gap;
- `loss_weights.routing_balance = 0.05` and `step_routing = 0.5` presented as the
  architecture-defining weights, when canonical is 0.001 and **0.0** — an agent
  reading that would have concluded the router is oracle-supervised, which is the
  opposite of the truth and would have made every routing number in the repo look
  broken rather than unsupervised;
- module line counts from the refactor (`engine.py` 594, `model.py` 501) against
  1280 and 1076 now, and a repo map missing eleven files including every gate test,
  `seed_stats.py`, `export_results.py` and `results/`;
- a status section ending at Phase 6 with "Next: the controlled ACT decision
  experiment, then freezing `canonical_spec.json`, then the final matrix" — all
  three long since done;
- `evaluate_paper_metrics` described as "retained but marked UNUSED in-source",
  written before the deletion recorded above;
- Gate 0 described as still refusing every canonical claim because five
  `enforced_fields` entries were `null`, when the spec has been frozen since
  Phase 8 and 15 runs have been admitted through it.

**Methodology.** Every replacement number was read out of a generated artifact, not
out of another document: `results/results_aggregate.csv` for the canonical table and
the routing/depth aggregates, `code/canonical_spec.json` for the frozen fields and
the `architecture_variants` blocks, `automated/family_cls_ablation_result.json` for
the retained-fraction figures, `wc -l` for the line counts, and a fresh
`run_correctness_suite.py` for the gate counts. Where an old number was a *finding*
rather than a stale transcription — the 1-epoch reference tables, the T6.x closure
counts — it is marked superseded and kept, not deleted, per `CLAUDE.md` §6.

**What was added, and why each addition is load-bearing.**

- **§5, "The canonical router is UNSUPERVISED: `step_routing = 0.0`."** The most
  consequential thing about this repo's routing numbers is that no loss term tells
  the router which expert is correct. Without that stated up front,
  `val/routing_accuracy = 0.163 ± 0.186` (a std larger than the mean) reads as a
  broken router instead of as an arbitrary index assignment, and the mandatory
  permutation-invariant metrics look optional. The section names the three to read
  instead, with their values.
- **The `family_cls` disclosure.** `loss_weights.family_cls = 0.5` was *inherited*
  into the spec rather than selected, and it is the only term carrying family
  information. The ablation says the partition survives without it (76% of
  Hungarian, 72% of AMI, 80% of purity retained; task loss unmoved) — so the honest
  claim is "mostly not an artifact of the label", not "fully unsupervised".
- **§5a's three-prefix depth table** (`train/` vs `val/` vs `val_offline/`) with the
  best-val-checkpoint caveat, plus the held-out matrix. This is the distinction the
  entry above exists to protect; a document that lists depth numbers without the
  prefix invites exactly the conflation.
- **§10's canonical result** with the exact p-values and the three framing confounds
  (scale, dataset, evaluation) stated as limitations on the *conclusion*. The
  framing ruling is recorded in `TASKS.md` under T11.2; repeating its substance here
  keeps an agent from "discovering" MoR's win and restructuring around it.
- **§3's read-side call graph.** The write side was documented and the read side was
  not, which is how a hand-copied number gets into a table. There is one definition
  of admissible (`export_results.admit`) and one of significant
  (`seed_stats.perm_test`).
- **§6's reading rule** (exact randomization test, sample std, resolution floor
  `1/C(n_a+n_b, n_a)`, and the fact that a 3-seed arm cannot clear α = 0.0083).
- **§7's single command.** Gate counts were quoted per-subsection and had all
  drifted upward as the suites grew. The section now leads with
  `run_correctness_suite.py`'s table and says outright that the historical
  per-subsection counts are lower and should not be quoted.

**Failure tracing.** If a number in `ARCHITECTURE.md` disagrees with `results/`,
`results/` wins — it is generated and the document is written. If it disagrees with
`canonical_spec.json` about what is enforced, the spec wins. The document is
descriptive by its own first paragraph; `plan.md`, `updated_rules.md` and
`updated_objective.md` remain authoritative on conflict.

**Where to look.** `ARCHITECTURE.md` §1 (architecture-defining field table +
`ffn_mult` budget), §2 (repo map), §3 (both call graphs), §5 (`step_routing`,
`family_cls`), §5a (depth prefixes + held-out matrix), §6 (reading rule), §7 (gate
table), §10 (canonical result, arm roster, defect list).

**Verified by.** `run_correctness_suite.py` **350/350, 5/5 gates PASS**;
`smoke_test.py` ALL TESTS PASSED; `verify_pipeline.py` 35/35. Stale-reference sweep:
`grep -rn "evaluate_paper_metrics"` now returns only the in-source deletion note,
one historical mention in a `metrics.py` docstring marked as deleted, and the
TASKS.md/changelog history.

---

## T10 (extension) — `fixed_depth` ablation extended to seeds 45–46; the arm that ignores the halting mechanism entirely fits best

**Why this arm and not the other three.** Three Phase 10 arms were still at n=3
(`router_noise`, `two_blocks`, `ponder_cost_low`). Only `fixed_depth` was worth GPU
time, and the reason is a resolution argument rather than a hunch: the exact
two-sided randomization test in `code/seed_stats.py` cannot return a p below
`2 / C(8,3) = 0.0357` at 3-vs-5, and the Phase 10 family-wise threshold is
Bonferroni α = 0.05/6 = **0.0083**. At n=3 `fixed_depth` sat at p = 0.125 with
d = −1.90 — the largest effect of any arm and the *only* one pointing in the
direction that would falsify the halting claim — yet it could not have reached
significance at any effect size whatsoever. That is a measurement floor, not a
null result, and it is the only kind of "insignificant" finding that more seeds can
legitimately move. The other three arms were left at n=3 and are reported as
direction-only.

**Result at n=5.** `val/task_loss` **0.062886 ± 0.000143** vs canonical MoRE
0.063186 ± 0.000249. Gap **−0.000301** (se 0.000128), **d = −1.48, p = 0.0476**,
resolution floor now 0.0040. Nominally significant; **fails Bonferroni**. Reported
in `results/results.md` §11.3 as exactly that, with both thresholds named — a
p of 0.0476 against a family-wise α of 0.0083 is not a positive finding and must
never be quoted as one. `best_val_loss` agrees in direction
(0.062715 ± 0.000166 vs 0.062892 ± 0.000216), which matters because the two
statistics are computed at *different model states* and could have disagreed.

**The finding that reshaped §19.** Forcing all 7 recursion steps for every token
costs nothing measurable in wall clock: **2198 ± 274 vs 2225 ± 274 records/sec**,
indistinguishable, despite 3.4× the recursion steps actually executed. So the
learned halting mechanism bought neither quality nor throughput on this task at
this scale. Combined with the §9.1 null model and the `ponder_cost_low` arm, that is
three independent lines converging on Outcome C for the adaptive-depth half of the
architecture — and this one is the strongest, because it is a direct intervention
rather than a comparison against a baseline policy.

**The metric that had to be reinterpreted.** `depth/allocation_error_abs` for this
arm is **5.8850 ± 0.0002**, necessarily so: every token sits at depth 7 against a
token-weighted curriculum target mean of 2.1082. The arm that is *maximally wrong*
about the curriculum fits the task *best*. `results/results.md` §15.13 now treats
`depth_allocation_error` as agreement-with-a-hand-authored-assumption, not as
correctness, and that reading is propagated to §9, §11.3 and §19. This is the
single most important interpretive change in Phase 10 and it came from an ablation
that was expected to be a formality.

**The trap: a canonical-guard "failure" that was the guard working.** Launching
seeds 45–46 tripped the proxy guard, and the instinct — the wrong one — is to relax
the guard. The guard was correct: an ablation arm is not `canonical_phase_b`, and
`fixed_depth` declares `experiment_group = t10_ablation` with
`variant = fixed_depth`. Nothing in `code/canonical_spec.json` or
`code/more/run_context.py` was weakened. This is recorded because a future agent
adding a seed to an ablation will hit the identical message and must reach for the
declaration, not the guard.

**The trap: verifying that a resumed launcher reproduces the original arm.** Seeds
45–46 were launched months of edits after seeds 42–44, so the arm's config identity
had to be proven unchanged rather than assumed. Method: a 1-epoch run at an existing
seed under the current launcher, then a seed-blind config comparison against the
stored `resolved_config.json` of the original run. The seed-blind flattening is
required because `config_hash` includes the seed by construction
(`code/more/run_context.py:82–99` strips only `provenance` and `logging`), so the
five seeds of any arm hash differently and hash equality is the wrong test. Without
this step a silent config drift between seed 42 and seed 46 would have been reported
as seed variance.

**Failure tracing.** If `fixed_depth` ever reports a depth number other than `N/A`,
the halting bypass has stopped being a bypass — `code/eval_val_depth.py` deliberately
emits `N/A` for every depth key on this arm because there is no halting distribution
to measure, while still emitting `offline/val_task_loss`. A numeric depth there means
the arm is no longer what its name says. If the n=5 verdict moves, re-check that the
arm's five run directories are the ones named below and that none is a re-run.

**Where to look.** `automated/phase10_ablations.py` (arm definitions, `--report-only`
regeneration, per-arm baseline seed-count matching);
`automated/phase10_ablations_result.json` (the verdict table, PROBLEMS list now 3
entries rather than 4); `code/seed_stats.py` (`mean_std`, exact randomization test,
`min_p`); `code/eval_val_depth.py` (the `N/A` policy); the five run directories
`runs/t10_more_fixed_depth_seed{42..46}__{5bcde9fd,66a417f3,ba544f66,be64ebdc,557b8dcf}`.

**Verified by.** Both new runs completed 50 epochs (`results.tsv` 51 lines each,
`metrics.json` present, `val_depth_offline.json` written with depth keys `N/A` and
`offline/val_task_loss` populated). `automated/phase10_ablations.py --report-only`
regenerated the verdict table with `fixed_depth` at n=5. `run_correctness_suite.py`
**350/350, 5/5 gates PASS** after the extension.

---

## T11 (new tool) — `code/depth_null_model.py`: the missing null for the adaptive-depth claim

**The gap this closes.** Every adaptive-depth number in the project was previously
reported against the hand-authored curriculum target only: average depth,
per-operation depth, `depth_allocation_error_abs/_rel`, early-exit and forced-exit
rates. All of those answer "how far is the model from the curriculum". None answers
the question that decides whether *adaptivity* exists: **would a policy that ignores
its input entirely do better?** A model can post a respectable allocation error while
being, in effect, a constant. Without this null the sentence "the model learns to
allocate recursion depth in accordance with the predefined operation-complexity
curriculum" is unfalsifiable.

**Method.** Sweep a constant depth `c` over the held-out split, score
`mean |c − target|` against the same token-level targets and the same tokens the
model metrics use, and take the minimum. Best constant is **c = 2.00 at abs error
0.8703**. Both recursive arms lose to it: MoR 0.9423 ± 0.0266 (**+8.3% worse**, 5/5
seeds lose) and MoRE 0.9940 ± 0.1054 (**+14.2% worse**, 5/5 seeds lose). The
per-seed unanimity matters more than the means — it removes the possibility that one
bad seed carries the result.

**Token-weighted, not per-operation-type.** The curriculum mean over the 16 operation
*types* is 1.8750, but depth metrics are token-level and the operation mix in the
split is not uniform, so the correct comparison mean is the **token-weighted 2.1082**
(`curriculum_mean_token_weighted` in the output). Using 1.8750 would shift the null's
optimum and understate how close the models already are to constant behaviour. This
is the same units trap that has bitten the throughput and entropy metrics elsewhere
in this project: the aggregation weight is part of the metric definition.

**What this null can and cannot conclude.** Because `|·|` is convex, beating the
model with a single constant rules out depth variation that is *aligned* with the
curriculum. It does **not** distinguish "no structure" from "structure misaligned
with the curriculum" — a model could vary its depth informatively along some axis the
curriculum does not encode and still lose to a constant. `results/results.md` §9.1
states this limitation in the same breath as the result, and §15.13 carries the
complementary point that the curriculum itself is not independently validated by
task performance.

**Failure tracing.** If the best constant is ever reported as something other than
≈2.00, check the target mean first — a best constant near 1.9 means the script has
reverted to unweighted per-operation-type targets, and a best constant near 7 means
it is reading a fixed-depth arm's directories. If the model arms suddenly *beat* the
null, confirm the sidecars being read are `val_depth_offline.json` (held-out) and not
training-time `depth/` keys; the training-time distribution is measured at a
different model state and is not comparable.

**Where to look.** `code/depth_null_model.py` (sweep + token weighting),
`results/depth_null_model.json` (`curriculum_mean_token_weighted`, best `c`, per-arm
per-seed comparison), `code/eval_val_depth.py` (produces the held-out sidecars this
consumes), `results/results.md` §9.1 (result + convexity limitation), §15.13 (the
curriculum-is-an-assumption point).

**Verified by.** `python depth_null_model.py` reproduces c = 2.00 / 0.8703 and the
5/5-seeds-lose count for both arms; the per-arm means it prints match
`results/results.json` for MoR and MoRE to the digits quoted in §9.1.

---

## T11.2 — `results/results.md`: the 19-section results narrative

**What it is.** 1,625 lines, §1–§19 per `plan.md` §15, written *over* the generated
artifacts rather than alongside them. No number was typed from a log or a chat
transcript; each is copied from `results/results.csv` / `.json` /
`results_tables.md`, `results/seed_stats.json`, `runs/*/val_depth_offline.json`,
`automated/phase10_ablations_result.json`,
`automated/family_cls_ablation_result.json`, `results/depth_null_model.json` or
`results/bench_capacity.log`. §16 is the manifest that names the producing script for
each. Register is enforced by an explicit MEASUREMENT / INTERPRETATION / HYPOTHESIS
convention declared in the preamble and applied inline, so a reader can always see
which sentences are load-bearing.

**The headline it had to carry.** MoR 0.062706 ± 0.000112 beats MoE
(d = 4.13, p = 0.0079) and beats MoRE (d = 2.49, p = 0.0159); MoRE vs MoE is
unresolved (d = −0.36, p = 0.6587). That last contrast is the one that isolates the
composition, because MoE *is* MoRE at `max_depth = 1` with identical parameter
tensors. §1 and §19 therefore report Outcome C for the composition, Outcome C for
adaptive depth on three independent lines (§9.1 null model, §11.3 `fixed_depth`,
§11.5 `ponder_cost_low`), and B-leaning-but-not-clean for specialization — because
MoE shows the same partition strength with no recursion at all, so specialization
cannot be credited to the composition either.

**The context every architecture number is set in.** All three arms explain only
**21.8–22.5%** of held-out target variance against the predict-the-mean floor. The
entire between-architecture spread is ~3% of what any single arm explains. §6 states
this before the comparisons, because a reader who sees the significance stars first
will over-read a difference that is small relative to how much of the task is
unexplained by every arm.

**Two sections that did not exist when the task was written.** §9.1 (the null model,
above) and §13, which reports the capacity microbenchmark and then **rejects its own
width sweep**: `d_model = 512` timed at 72.47 ms/step against 256 at 143.88 ms/step,
which violates the benchmark's own printed invariant that a wider model can never
finish sooner. Cause is laptop-GPU clock ramp under too few warmup iterations. Only
the batch sweep is kept, and it is a clean launch-bound signature — batch 96→6144
(64×) costs 2.5× wall clock per step, ms per step-token falling monotonically
162.37 → 6.40. §13 also records that `batch_size` is a hashed config field, so the
canonical runs must **not** be re-run at a larger batch to exploit that headroom.

**The trap: a unit error that nearly published a ~70× fabrication.** §13.3 was about
to compare the microbenchmark's 37,395 tok/s against training's ~2,225 tok/s. The two
are different units. `code/more/engine.py:615` accumulates `total_tokens += bs` and
`:621` divides by elapsed, so training throughput is **records/sec**;
`code/bench_capacity.py:94` sets `tokens = batch * STEPS`, so the benchmark reports
**step-tokens/sec**. Like-for-like is 37,395 ÷ 7 = **5,342 records/s** against the
`fixed_depth` arm's 2,198 records/s — a real ~2.4× gap, not 70×. The reconciliation is
written into §13.3 itself so the trap is documented where the next reader will hit it.

**The trap: baselines are seed-count-matched, so one baseline metric has two correct
values.** `automated/phase10_ablations.py` prints a baseline column matched to each
arm's seed count. Canonical MoRE held-out abs depth allocation error is therefore
1.0282 ± 0.1320 over seeds 42–44 and 0.9940 ± 0.1054 over all five — both correct.
Three n=5 arms in an early §11 draft were compared against the 3-seed column. Fixed,
and §11's design section now opens with a "read the baseline numbers carefully"
paragraph so the two-valued baseline cannot read as a contradiction.

**Two other corrections made before publication.** §16 originally claimed the
exporter requires one `config_hash` per architecture — false, since the seed is inside
the hash, so all 15 canonical runs hash differently by construction; the real check is
the seed-blind config comparison plus duplicate-`(architecture, seed)` detection and
single-`dataset_version`/`train_split_version`/`code_git_commit` checks. §12
originally said no run exists above 3.2 M params, which the 6,358,553-param
`two_blocks` arm falsifies; corrected to *no trained run at a larger FFN width*, with
`two_blocks` named and the reason it does not fill the gap (it adds a block rather
than widening the FFN, and it is MoRE, not MoR).

**Failure tracing.** If a number in `results.md` disagrees with `results/*.csv|json`,
the generated artifact wins and the prose is stale — regenerate with
`export_results.py` and re-check the section. If a section quotes a figure with no
producing script named in §16, treat it as unverified. If the exporter's admitted
count drops below 15, a provenance field regressed; the refusal reasons are recorded
per directory and a refusal is the tool working.

**Where to look.** `results/results.md` §6 (main table + variance context), §9.1
(null model), §11.1–§11.7 (six Phase 10 arms + the `family_cls` proxy), §13.3 (the
throughput unit reconciliation), §15.1–§15.13 (failure cases, including §15.8 the
undisclosed `family_cls` term, §15.9 depth and loss measured at different model
states, §15.13 the curriculum-as-assumption point), §16 (run manifest + script
provenance), §17/§18 (claims supported vs not), §19 (verdict).

**Verified by.** `grep -nE "<!--CHUNK|TODO|TBD|PLACEHOLDER|XXX" results/results.md`
returns nothing; all 19 section headings present. Derived artifacts regenerated after
the `fixed_depth` extension: `export_results.py` scanned 129 directories, admitted 15,
refused 114, 137 metric keys in union, 580 cells `N/A`, consistency line clean.
`run_correctness_suite.py` **350/350, 5/5 gates PASS**; `verify_pipeline.py` 35/35;
`smoke_test.py` ALL TESTS PASSED.

---

## T11.3 — `README.md` rewritten; invalidated numbers named as invalidated rather than deleted

**Why this was urgent.** `README.md` was the only document in the repository still
*publishing* pre-audit results as current, and it is the first file any reader opens.
Its §2–§3 tables predated the leakage audit, the provenance guard and the
permutation-invariant routing metrics, and its §1.4 asserted that L2 regularization
prevents `router_noise_scale` from collapsing to 0 — the opposite of what an L2
penalty does.

**Removed as invalidated:** the `0.002320` validation loss and the "87.26% reduction",
the `25,470 tok/s` table, expert load entropy quoted against `ln(7)` (canonical is six
experts, and entropy is reported normalized as `H / log(E)`), "simple ops halt at 1.03
steps / complex at 6.68 steps" — directly contradicted by measurement, which puts
93–97% of tokens at exactly 2 steps in *both* recursive arms — the oracle-routing
weight `0.01` (canonical `step_routing = 0.0`, the router is unsupervised), the
L2-prevents-noise-collapse claim, the "fewer than 7 experts / fallback expert"
language (the E7 catch-all is removed and an unmapped operation must raise), and
`python run_sweeps.py` writing `sweep_results.csv` — that path does not exist under
`code/` and that filename is a banned global output.

**The decision worth recording: named, not erased.** Deleting those figures would
have satisfied the task's verification ("no invalidated number remains") while
violating "archive, never delete, evidence" — and would have left a reader who
remembers the old README with no way to learn that it was wrong. README §5 therefore
lists the old numbers **explicitly as invalidated**, each with the reason it is not
comparable to the current table (proxy runs, 7-expert config, dense
evaluate-all-and-blend routing, pre-leakage-audit, pre-guard,
pre-permutation-invariant-metrics). Nothing there is presented as a result, so both
constraints hold. §5 additionally names the two claims that *inverted* rather than
merely aged: dense evaluate-all-and-blend went from "the fix for blocked router
gradients" to a labelled ablation that measures significantly worse than canonical
Top-1 sparse (§11.1), and the L2-on-trainable-noise-scale claim is withdrawn outright.

**Guardrails shipped with the numbers.** §4 carries the metric-reading rules inline —
never quote `total_loss`; normalized entropy is a load-balance diagnostic that a
router ignoring its input maximizes, so it cannot show specialization; raw routing
accuracy is uninformative under an unsupervised router (its seed std exceeds its mean),
quote Hungarian-matched/AMI/purity; low cosine similarity is "consistent with
differentiated parameterizations", never proof of orthogonality; "depth allocation
error", never "compute efficiency"; no sentinel is a measurement. A reader who never
opens CLAUDE.md still cannot misread §1. §1 also states the specialization result
together with the fact that supervising the router drives every routing metric to
exactly 1.0000 ± 0.0000 *while making task loss significantly worse*, so routing
quality is never quotable as evidence of architectural quality.

**Failure tracing.** If a README number disagrees with `results/results.md`, the
README is stale — it is hand-written prose over the same artifacts and has no
generator. If a figure from the §5 invalidated list ever reappears outside §5, the
rewrite has regressed; those exact strings are the ones to grep for.

**Where to look.** `README.md` §1 (headline finding, main table, depth null-model
table, "What bounds all of this"), §2 (repo layout), §3 (verified commands), §4
(metric-reading rules), §5 (invalidated history + the two inversions);
`results/results.md` is the source for every figure in §1.

**Verified by.** 98 → 203 lines. Every command published in §3 was executed from
`code/` first (`run_correctness_suite.py`, `export_results.py`, `depth_null_model.py`,
`eval_val_depth.py --glob 'phaseB_*'`, `../automated/phase10_ablations.py
--report-only`), and the flag names were checked against `code/more/cli.py` and
`code/train.py`. Cross-check of §1 against `results/results.json`: task losses, R²
values (0.2182 / 0.2250 / 0.2191 → the range is 21.8–22.5%, not the 22.7% carried in
an earlier draft), parameter counts and all six Phase 10 verdicts match.

---

# Language migration (branch `claude/english-language-dataset-migration-92946e`)

Everything below belongs to the **English-language study**, not the arithmetic
one. The arithmetic study is closed: `TASKS.md` is closed, its runs in `runs/` are
never re-run or deleted, and its numbers are never merged or averaged with
language numbers (`TASKS_LANGUAGE.md` T-L10.2, T-LX.2). The two are separate
studies that share one codebase.

The plan is [plan_language.md](plan_language.md); the resumable ledger and resume
point is [TASKS_LANGUAGE.md](TASKS_LANGUAGE.md).

## T-L0.1 — CUDA interpreter, because the documented one does not exist here

**What.** Built `D:\res\git\MoRE\.venv_cuda` with `torch 2.6.0+cu126` on top of
the Anaconda base interpreter, and verified CUDA end to end on the RTX 3050.

**Method.** `python -m venv --system-site-packages`, then torch alone from the
cu126 index. `--system-site-packages` is the point: numpy, scipy, sklearn, pandas,
wandb and the whole HuggingFace stack stay in exactly one place, so the CPU and
CUDA interpreters cannot drift apart in anything except torch. The venv's own
`site-packages` shadows base for torch only.

CUDA was verified by running a **real device matmul**, not by trusting
`torch.cuda.is_available()`, which returns `True` on drivers that then fail to
launch a kernel.

The install printed four dependency-conflict warnings — `google-genai`,
`pymilvus`, `s3fs`, `streamlit`. All four are unrelated base packages with loose
pins; none is imported anywhere in this repository. Verified harmless rather than
assumed harmless. Base was confirmed unmodified afterwards
(`base torch 2.6.0+cpu cuda False`), so the venv is fully reversible: delete the
directory and nothing else changes. It is git-ignored.

**Failure tracing.** `No such file or directory` on the interpreter path means you
are reading a doc written on the old development machine — see T-L0.0.
`torch.cuda.is_available() == False` from `.venv_cuda` means the venv's torch was
shadowed by base's CPU build; check that `torch.__version__` ends in `+cu126`, not
`+cpu`. A CUDA OOM at 6 GB is expected territory in the language phase, not a bug —
that is what T-L7.0 exists to measure before the protocol is frozen.

**Where to look.** [ENVIRONMENT.md](ENVIRONMENT.md) §1 and §4 (both interpreters,
the build commands, the conflict warnings); `CLAUDE.md` §9; `ARCHITECTURE.md` §9.

**Verified by.** Probe of both interpreters, 2026-09-03: CPU
`C:\Users\vedan\anaconda3\python.exe` Python 3.12.7 `torch 2.6.0+cpu`
`cuda False`; CUDA `D:\res\git\MoRE\.venv_cuda\Scripts\python.exe` Python 3.12.7
`torch 2.6.0+cu126` `cuda True` CUDA 12.6,
`NVIDIA GeForce RTX 3050 6GB Laptop GPU` 6143 MiB, compute capability 8.6, driver
596.08.

## T-L0.3a — `os.path.relpath` raises across Windows drives, and gate 4 died with no `[FAIL]`

**What.** `RunContext.create` recorded the run directory as a repo-relative path.
On this machine that **raised** and took gate 4 down at check 51. Fixed by falling
back to the absolute path when no relative expression exists. Gate 4 went from
`51 pass + crash` to `129 checks 129 pass 0 fail 0 skip PASS`.

**Method.** The line was `os.path.relpath(directory, _REPO_DIR)`. On Windows,
`relpath` does not degrade gracefully across mounts — it raises
`ValueError: path is on mount 'C:', start on mount 'D:'`. The repository is on
`D:`; `test_phase6_seeding.py:388` creates its runs root with
`tempfile.mkdtemp(prefix="t61_runs_")`, which lands in `C:\Users\...\Temp`.

Extracted `_repo_relative(path)` in
[code/more/run_context.py](code/more/run_context.py) (just above `file_sha256`):
keep the relative form with forward slashes when one exists, otherwise the
absolute path, also forward-slashed. A run directory outside the repository is
**legitimate** — both consumers of this field only echo the string — so the fix is
to record the truth, not to force a relative path that cannot exist.

Not a new defect. It was latent from the beginning and invisible on the machine
the POC was developed on, because there the repository and the temp directory
happened to share a drive.

The secondary damage is worth remembering: the crash produced **no `[PASS]`,
`[FAIL]` or `[SKIP]` marker at all**, so `run_correctness_suite.py` could only
report *"output format not recognised"*. A suite whose failure mode is an
unparseable gate tells you less than a suite that fails loudly.

**Failure tracing.** *"output format not recognised"* from the suite runner means a
gate file **crashed before its first marker** — re-run that one file directly and
read the traceback; there is no `[FAIL]` line to find. Any `ValueError: path is on
mount` means a path is being made relative across drives; the repo is on `D:` and
`tempfile` is on `C:`.

**Where to look.** [code/more/run_context.py](code/more/run_context.py)
`_repo_relative` and its single call site setting `prov["run_dir"]`;
[code/test_phase6_seeding.py](code/test_phase6_seeding.py):388 (`_TMP_RUNS`);
`code/run_correctness_suite.py` `MARKER`.

**Verified by.** `C:/Users/vedan/anaconda3/python.exe code/test_phase6_seeding.py`
→ 60 pass, rc 0 (was 51 pass then crash). Gate 4:
`129 checks 129 pass 0 fail 0 skip 21.1s PASS`.

## T-L0.3b — gate 5 graded a run from before the fix it was testing, because `git worktree add` rewrote every mtime

**What.** Gate 5's G5.2b block reported five `val/routing_*` keys as *absent* on a
single-expert run, which would mean the T8.1 sentinel contract had regressed. It
had not. **The writers are correct at HEAD** — a MoR run built at `08e90427` emits
all eleven routing and expert-diversity quantities as the string `'N/A'`, present
rather than absent. The fault was the test's *selection key*. Fixed two ways, and
the fix closed a real blind spot as well. Gate 5:
`150 checks 150 pass 0 fail 0 skip PASS`.

**Method.** `test_gate5.py` chose its single-expert run as the newest matching
directory by `os.path.getmtime`. But `runs/` is **tracked** (555 files), so
`git worktree add` wrote all 132 run directories fresh, in checkout order — a
measured mtime spread of 2026-09-03 01:54:39 → 02:30:05, 2125.8 s, carrying no
relation whatever to when the experiments ran. That put
`t83_mor_headfix_seed42__dd11e6dc__r2` last: commit `e5b0f6df`, from **before** the
T8.1 fix that made those keys present-and-`"N/A"`. The test was grading
pre-fix history and reporting a defect the current code does not have.

Audited all 18 single-expert runs to be sure the archive itself was sound: 9
satisfy the contract (all five `phaseB_mor_seed4*`, plus `t81b`/`t81c` at
`30831ff6`), 9 predate the fix. `t81_mor` (0 `N/A` keys) and `t81b_mor_seed42`
(5 `N/A` keys) sit at the same commit and are exactly T8.1's before/after pair — so
the archive is behaving as intended and nothing there needs correcting.

Fix 1, **selection by provenance instead of by filesystem**: `_pick(pred)` in
[code/test_gate5.py](code/test_gate5.py) prefers directories whose
`provenance.code_git_commit` equals `git_commit()`, returns an at-HEAD flag, and
falls back to older runs only for the diagnostic print. G5.2b now runs only when
`RUN_E1_AT_HEAD`, and its `SKIP` message **names the commit** of the historical run
it declined to grade — a skip that explains itself instead of a pass that lies.

Fix 2, **give the slot a producer**: new **T6.7d** section in
[code/test_phase6_provenance.py](code/test_phase6_provenance.py) runs a 1-epoch
`mor` job through the real CLI via a new `_smoke_run()` helper. Gate 5 runs
provenance *before* `test_gate5.py`, so a fresh at-HEAD single-expert run always
exists and the dependence on timestamps is removed entirely rather than merely
worked around.

T6.7d also closed a genuine gap: the `updated_rules.md` §9 provenance list had only
ever been asserted on a `more` run. `E=1` and `max_depth=1` are exactly where
architecture-specific defects hide — cf. T8.3, where a head sized by `num_experts`
passed at E=6 and hit a CUDA device-side assert at E=1. The 27-field list was
extracted into `_required_fields(rc)` so both architectures are held to the
identical list, and one of the new checks is that MoR's legitimately **zero**
`routing_balance` is recorded as `0.0` and not dropped — `if not lw.get(...)` would
read it as missing, which is the falsy-zero form of reporting a sentinel as a
measurement.

Cost: one extra ~49 s CPU MoR run inside the suite. Worth it.

**The principle.** *Filesystem metadata is not experiment metadata.* An mtime, a
directory ordering or a drive letter is a property of the substrate. Using one as
evidence is the same error class as reporting a sentinel as a measurement — a value
that looks like data but is an artifact of where it was stored. Any future test that
picks "the latest run" must pick it by provenance.

**Failure tracing.** A gate-5 routing-key failure on a MoR run: check *which
directory it graded* before touching the metric writers. Print the run's
`provenance.code_git_commit` and compare with `git_commit()`. If the suite reports
`G5.2b SKIP`, no single-expert run at HEAD exists — T6.7d did not run or did not
produce a directory, which is a gate-5 ordering problem, not a metrics problem.

**Where to look.** [code/test_gate5.py](code/test_gate5.py) `_commit`, `_HEAD`,
`_at_head`, `_pick`, `RUN_E1_AT_HEAD` and the G5.2b guard;
[code/test_phase6_provenance.py](code/test_phase6_provenance.py) `_required_fields`,
`_smoke_run`, the T6.7d block; `code/run_correctness_suite.py` `GATES` (provenance
is pinned ahead of `test_gate5.py` on purpose).

**Verified by.** Gate 5 `150 checks 150 pass 0 fail 0 skip PASS`.
`test_phase6_provenance.py` 38 → 43 pass. `test_gate5.py` 38 pass / 1 fail → 39
pass / 0 fail. The eleven-key contract confirmed directly on
`runs/t_l03_mor_timing_seed44__fa9339bc` (all eleven present, all the string
`'N/A'`).

## T-L0.3 — an executed baseline for "arithmetic unchanged": **356**, not 350

**What.** Established Gate L0, the regression fence the whole language migration is
measured against: the full correctness suite, run to completion on this machine,
**`TOTAL 356 356 0 0`, 5 gates, ALL GATES PASS**. Recorded before any language code
exists, so "the arithmetic study still passes" is a falsifiable claim rather than an
assurance.

**Method.** Run the suite first, fix what it finds, run it again — do not write new
code on top of a suite whose current state is unknown. Two pre-existing defects had
to be fixed to reach a completing run: T-L0.3a (cross-drive `relpath` crashed gate
4) and T-L0.3b (gate 5 graded pre-fix history). Both are recorded above.

Both were **environment-dependent in the same way**, which is the transferable
lesson: each was invisible on the machine the POC was developed on because that
machine's filesystem happened to satisfy an assumption the code made silently — the
repository and the temp directory on one drive; mtimes that tracked experiment
order. Neither was a mistake in the architecture, the metrics or the science. They
were mistakes about the substrate.

The suite total therefore moved **350 → 356**:

| | |
|---|---|
| 350 | the count at the close of the arithmetic study, still quoted in older notes |
| +1 | gate 4 — the seeding suite now completes instead of crashing at check 51 |
| +5 | gate 5 — the new T6.7d single-expert provenance producer |
| **356** | measured on `C:\Users\vedan\anaconda3\python.exe` |

Every stale `350` in `plan_language.md` (§0, §9, §11), `TASKS_LANGUAGE.md` and
`README.md` was corrected to 356, each with the derivation beside it rather than a
bare number — a reference count with no derivation is the thing that goes stale.

Separately confirmed, so that no engine or metrics change is needed for the language
phase: the whole `N/A` writer contract is intact at HEAD. All eleven routing and
expert-diversity keys on a MoR run are **present** and are the **string** `"N/A"`.
The arithmetic POC's published behaviour stands as written.

**Failure tracing.** A gate count other than 356 is a STOP (`plan.md` §20), and the
first question is *which* gate moved, not which language change caused it. A count
that **drops** usually means a suite file stopped being discovered or crashed before
its first marker — a gate reporting zero assertions FAILS by design. A count that
**rises** without a new task is an accidentally duplicated check.

**Where to look.** `code/run_correctness_suite.py` (`GATES`, `MARKER`, the
zero-assertion rule); [ENVIRONMENT.md](ENVIRONMENT.md) §5; `plan_language.md` §0 for
the 350 → 356 derivation; `TASKS_LANGUAGE.md` T-LX.0 for the obligation to re-run
after every commit.

**Verified by.** `C:/Users/vedan/anaconda3/python.exe code/run_correctness_suite.py`
→ `TOTAL 356 356 0 0`, 5 gates, `ALL GATES PASS`.

## T-L0.0 / T-L0.2 — the documented environment was another machine's, and ENVIRONMENT.md now exists

**What.** `CLAUDE.md` §9 and `ARCHITECTURE.md` §9 instructed the reader to use
`C:\Users\Hp\anaconda3\envs\more_env\python.exe` (torch 2.5.1) on an RTX 4060
Laptop 8 GB. Neither exists here. Both were corrected to name the two real
interpreters and the RTX 3050, a false in-source justification was corrected in
three places, and [ENVIRONMENT.md](ENVIRONMENT.md) was written so a resumed session
does not have to rediscover any of it.

**Method.** The correction had to distinguish two kinds of stale reference, and
this is the part worth remembering:

*Instructions* were corrected. `CLAUDE.md` §9, `ARCHITECTURE.md` §9 (including all
four quick-check bash blocks), the usage docstrings of
`test_phase2_routing.py` / `test_phase3_halting.py` / `test_phase4_balance.py` /
`test_phase5_dimensions.py`, and `README.md` §6 all tell a reader what to run. A
wrong path there costs the next session its first command.

*Records* were left alone. `changelog.md`, `TASKS.md`, `results/results.md`, and
`ARCHITECTURE.md`'s scale caveat describe the machine the **arithmetic runs were
actually produced on**, which really was the RTX 4060 with torch 2.5.1. Rewriting
those would falsify the provenance of published numbers. `CLAUDE.md` §6: archive,
never delete, evidence. The scale caveat gained one clause pointing at §9 so the
apparent contradiction resolves for a reader rather than looking like a defect.

The false-absence claim — *"No scipy/sklearn in this environment (checked)"* —
appeared in `code/more/metrics.py` above `_to_contingency`, in `ARCHITECTURE.md`
§T6.3, and in `code/test_phase6_routing_metrics.py`'s docstring, in each case as the
**reason** the exact Hungarian assignment and the AMI chance correction are
hand-written. sklearn 1.3.2 and scipy 1.14.1 are both importable from both
interpreters, so the stated reason was false and the code looked like a workaround
for a constraint that does not exist — an invitation for a future agent to "simplify"
it into a library call.

The implementations **stay**, with the real reasons recorded: they are exact at
`E <= 15` (a greedy match is not the Hungarian match and must not be labelled as
one), they are checked against brute force in `test_phase6_routing_metrics.py`, and
keeping them local means the metric layer imports nothing beyond torch/numpy — so a
metrics-only environment cannot silently produce a different Hungarian accuracy than
the training environment did. Replacing working exact code to use a library is churn,
not a fix. Only the justification was wrong.

Three `automated/*.py` drivers hard-code the dead interpreter and were **deliberately
not repointed**. Each also writes an absolute `c:\Users\Hp\Desktop\Waste\MoRE\...`
path and a global `results.tsv` / `final_run_metrics.json` — outputs `CLAUDE.md` §5
prohibits and `run_context.py` actively refuses. Repointing the interpreter would put
a forbidden-output script one edit from runnable, which is worse than leaving it
obviously broken. Each got a NOT-RUNNABLE banner naming why and saying what to write
instead. `run_remaining_tests.py`'s banner also states that, despite the name, it is
not part of the correctness suite.

`ENVIRONMENT.md` records two facts that change later phases, both measured rather
than assumed:

1. **The HuggingFace stack is already installed** in both interpreters
   (`datasets 4.6.1`, `transformers 4.51.3`, `tokenizers 0.21.0`,
   `huggingface_hub 0.30.2`, `pyarrow 23.0.1`). Phase L-2 needs no installation and
   no new pin. WikiText itself is **not** cached, so T-L2.0 is a real download.
2. **`nltk 3.9.1` is present**, with `taggers/averaged_perceptron_tagger` and
   `tokenizers/punkt` already downloaded to
   `C:\Users\vedan\AppData\Roaming\nltk_data` — but **`corpora/universal_tagset` is
   MISSING**. That settles T-L3.0's tagger question: use `nltk.pos_tag`'s Penn
   Treebank tags with an explicit PTB→six-family map checked into
   `code/more/lang_families.py`, **not** `pos_tag(tagset="universal")`, which would
   hit the missing corpus and trigger a download at dataset-build time — making the
   family definition a function of network state. The explicit map is also the better
   artifact: it is the thing a reader can disagree with, and an unmapped tag can
   raise instead of falling back to a catch-all (`CLAUDE.md` §2).

**Failure tracing.** `No such file or directory` on an interpreter path means the doc
was written on the old development machine; check `ENVIRONMENT.md` §1 first. A future
"why is this Hungarian solver hand-written when scipy is installed" question is
answered in `metrics.py` above `_to_contingency` — the answer is exactness plus
import surface, not availability. If a family assignment ever changes without the
code changing, suspect the nltk tagger version or its data package: the manifest
pins both (T-L2.4) precisely so that is detectable.

**Where to look.** [ENVIRONMENT.md](ENVIRONMENT.md) §1 (which interpreter), §2
(measured versions), §3 (HF stack + the nltk/`universal_tagset` finding), §6 (the
probe); `CLAUDE.md` §9; `ARCHITECTURE.md` §9 and §T6.3;
[code/more/metrics.py](code/more/metrics.py) above `_to_contingency`;
`automated/rerun_mor.py` for the NOT-RUNNABLE banner the other two reference.

**Verified by.** `C:/Users/vedan/anaconda3/python.exe code/run_correctness_suite.py`
after all edits → gate 1 19, gate 2 31, gate 3 27, gate 4 129, gate 5 150,
`TOTAL 356 356 0 0`, `ALL GATES PASS`, 174 s. Both interpreters probed 2026-09-03:
CPU `torch 2.6.0+cpu` / `cuda False`; CUDA `torch 2.6.0+cu126` / `cuda True` /
CUDA 12.6 / `NVIDIA GeForce RTX 3050 6GB Laptop GPU` 6143 MiB cc 8.6 driver 596.08;
identical in numpy 1.26.4, scipy 1.14.1, sklearn 1.3.2, pandas 2.2.1,
matplotlib 3.10.0, wandb 0.25.0, datasets 4.6.1, transformers 4.51.3,
tokenizers 0.21.0, huggingface_hub 0.30.2, pyarrow 23.0.1, nltk 3.9.1;
spacy/stanza/flair absent in both.

## T-L1.0 / T-L1.1 — the `task` axis, and why its default is the *absence* of a key

**What changed.** `code/more/config.py` gained a `task` axis orthogonal to
`architecture`: `TASK_ARITHMETIC`/`TASK_LANGUAGE` constants, a `TASKS` tuple
beside `ARCHITECTURES`, `resolve_task(cfg)`, `apply_task(cfg, task)` and the
`_apply_language_block` helper that stamps the language shape fields
(`seq_len`, `vocab_size`, `n_heads`, `attention`, `tie_lm_head`) and the language
loss weights. `resolve_variant` and `canonical_run_name` became task-aware.

**The ledger box's literal wording is not implementable, and this is the one
design decision of the phase worth remembering.** T-L1.0 said "add `task` to
`load_config_defaults` with default `arithmetic`". Doing that —
`cfg.setdefault("task", TASK_ARITHMETIC)` — fails the box's *own* Verify
criterion, and the mechanism is `run_context.py:82`: `config_hash` is a SHA-256
over the resolved config with only `provenance` and `logging` removed. A new
top-level key with a value, even a value that changes no behaviour, changes the
hash of **every arithmetic config**. That would rename every future arithmetic run
directory, break the `config_hash` equality the proxy guard and
`export_results.py` both rely on, and make the 15 admitted rows of the Phase 11
headline table unreproducible by any later run — for a key whose only content is
"this is the thing it has always been".

So **the arithmetic default is the absence of the key.** `resolve_task` reads
`cfg.get("task", CANONICAL_TASK)`; nothing writes `task` unless
`apply_task(cfg, "language")` is called. Arithmetic resolved configs and hashes
are byte-identical to pre-axis, which T-L1.4 then demonstrated on a real run
(`fa9339bc…`, 2014 bytes, all 64 hex digits). The cost is that `task` is not
self-documenting in an arithmetic `resolved_config.json`; the compensation is that
`resolve_task` is the single reader and there is no second code path.

**Two defects found and fixed while wiring this up.**

1. **`resolve_variant` compared `family_cls` against the arithmetic default on
   language runs.** Language sets `family_cls_weight = 0.0` by design — on
   language the family classifier is a **detached probe** (`h.detach()`), so a
   non-zero weight would train a head on gradients that never reach the trunk,
   i.e. it would be either a no-op or a lie depending on which you believed. But
   `resolve_variant`'s ablation detection read `0.0 != 0.5` and returned
   `no_family_cls`, so *every* language run would have been labelled an ablation
   of a term it does not have. Fixed by making the expected default per-task. The
   general shape of this trap: **a variant detector that hard-codes one task's
   defaults reports the other task's canonical settings as deviations.**

2. **The declared-vs-defaulted ambiguity.** Once the language default is `0.0`,
   `--family_cls 0.0` on a language run and no flag at all produce the same
   config — but only one of them is a user asserting something. `cli.py` takes a
   `_user_family_cls` snapshot **before** defaults are applied, so an explicit
   `--family_cls` on language is *refused* (naming `plan_language.md` §7.3 and the
   detached probe) rather than silently agreed with. Refuse-don't-ignore, same as
   every other override.

**Ordering matters and is fragile.** `apply_task` must run *after*
`apply_architecture` and *before* the override blocks. After, because the run name
template is task-dependent (`phaseB_{Arch}` vs `langB_{Arch}`) and
`apply_architecture` sets the architecture the template interpolates. Before,
because an override that lands ahead of `apply_task` gets overwritten by the
task's `setdefault`s instead of being honoured or refused.

**Where to look.** [code/more/config.py](code/more/config.py) — `resolve_task`
(`:84`), `TASKS` (`:57`), `resolve_variant` (`:460`), `_apply_language_block`
(`:744`), `apply_task` (`:848`); [code/more/cli.py](code/more/cli.py) `main`
(`:155`) for the ordering and the `_user_family_cls` snapshot;
`code/test_language_task_axis.py` TL1.0*/TL1.1*/TL1.2* for the checks.

**Verified by.** `code/test_language_task_axis.py` → **72 passed, 0 failed**.
TL1.2a–q drive `cli.main()` end to end with only `RunContext.create` and `train`
stubbed: no `--task` and `--task arithmetic` are byte-identical through the CLI
(2077 bytes, no `task` key, no shape field); `--task language` resolves to
`langB_MoRE` with `seq_len 256`; `--task language --family_cls 0.5` exits 1 with a
`ValueError` naming §7.3 and never reaches `train()`; the shape flags are refused
on arithmetic rather than ignored; `--vocab_size` is bounded at the uint16 ceiling
(65536 accepted, 70000 refused — the tokenized arrays are `uint16` and id 65536
would wrap to id 0, a *valid* id for a different token, corrupting the corpus with
no error anywhere).

---

## T-L1.3 — one canonical spec per task, and a refusal that used to give the wrong reason

**What changed.** `code/canonical_spec_language.json` is new;
`run_context.load_canonical_spec(path=None, task=None)` selects by task
(`CANONICAL_SPEC_PATH_BY_TASK`), each task has its own canonical group
(`canonical_phase_b` / `canonical_lang_b`), and the proxy guard refuses a
cross-task canonical claim as its own first-class check.
`code/canonical_spec.json` is **untouched**, as are its three other direct readers
(`export_results.py`, `test_phase6_provenance.py`, `test_phase6_seeding.py`) —
which is why this is a second *file* rather than a `task` sub-object inside the
existing one.

**The cross-task refusal already worked, and that was the problem.** Because
T-L1.1 makes `resolve_variant` return `"language"` and never `"canonical"`, check
2c already refused a language run claiming `canonical_phase_b` — with the message
*"variant='language' … this run has at least one ablation active"*. That is false,
and it would send the next reader hunting for an ablation that does not exist.
The check is now explicit, named ("WRONG GROUP FOR THE TASK"), symmetric in both
directions, and placed **before** the field loop so it cannot be buried under a
list of null-field complaints
([code/more/run_context.py:370](code/more/run_context.py:370)). **A guard that
refuses for the wrong reason is a guard that will be worked around.**

**Why separate group strings.** `canonical_phase_b` is the exact string
`export_results.py` filters the headline arithmetic table on. A language row
admitted there would put a per-token cross-entropy in nats into a column of
arithmetic MSEs — a units error that no downstream check would catch because both
are floats near 0.1. Two group names make the filter do the separating.

**`_effective()` had to grow five keys** (`seq_len`, `vocab_size` from `data`;
`n_heads`, `attention`, `tie_lm_head` from `model`). The guard iterates the
**spec's** `enforced_fields`, so a field the spec pins but `_effective` cannot
produce reports as *"requires 8192, run has None"*: a real check failing for a
fake reason, and one that would be debugged as a config bug. All five are read
with `.get`, so on arithmetic they are `None` and nothing compares them —
`canonical_spec.json` does not list them. TL1.3h asserts the closure directly
(`unreadable == []`) rather than trusting the list to stay in sync.

**The language spec leaves eleven fields null, not the eight T-L1.3 named.** The
three extra, with the reasoning in the file's `_NULLS_NOTE`: `dropout` (0.1 was
*inherited* on 70 k arithmetic records; ~103 M tokens against ~3.2 M parameters
underfits, so the inherited value points the wrong way), and
`routing_balance_weight` + `halting_weight`. Those last two are the substantive
ones: T4.2 chose `0.001` because it put the weighted balance term at `0.017×` the
task loss, and a per-token cross-entropy near `ln(8192) = 9.0` nats at init is two
orders of magnitude larger than arithmetic's `~0.06` MSE. The same coefficient
would put the balance term at `~1e-4 ×` the task loss — **effectively off, in the
arm whose central failure mode is router collapse** — and an equally invisible
ponder cost would silently turn adaptive depth into always-max depth. A loss
coefficient calibrated against one task's loss scale does not transfer to
another's. A null makes the guard strictly stricter, never looser, so leaving them
null costs nothing but a later measurement.

`attention: true` sits in `enforced_fields`, shared by all three arms, and
deliberately **not** in `architecture_variants`. That placement *is* the claim
that MoR keeps attention: one expert is not the same as no context, and a MoR arm
without attention would be a bag-of-tokens model losing to two sequence models.

**Where to look.** [code/canonical_spec_language.json](code/canonical_spec_language.json)
`_README` (why a second file, the four-condition contract, why `canonical_variant`
is `"language"`) and `_NULLS_NOTE` (the eleven nulls);
[code/more/run_context.py](code/more/run_context.py) — the per-task constants
after `CANONICAL_GROUP`, `load_canonical_spec`, `_effective`,
`assert_not_silent_proxy`'s opening block; `code/verify_pipeline.py` for the
sandbox dep tuple (the language spec is copied in so a future language run in the
sandbox fails for a real reason rather than a `FileNotFoundError`).

**Verified by.** TL1.3a–p inside the 72-check language suite. `load_canonical_spec()`
with no arguments still reads the arithmetic spec, whose identity fields are
unchanged and which gained no key (TL1.3a/b); an unregistered task **raises**
rather than falling back to another task's numbers (TL1.3f); the cross-task claim
is refused in both directions naming the task and the right group (TL1.3i–k); a
language run claiming its own group is refused for **unfrozen fields**, not for
its group, and the message names `canonical_spec_language.json` (TL1.3l–n);
`test_phase5_dimensions.py:G4.37`'s router-noise case is re-checked from the
language suite because that gate's message assertions are what the rewrite could
most easily have broken (TL1.3p).

---

## T-L1.4 — GATE L0 after the task axis: 356, identical params, and a bit-identical run

**Why the gate runs here.** The task axis is the change most likely to silently
alter arithmetic behaviour, and it lands before any language code exists — so a
regression caught here has exactly one possible cause.

**All three criteria, executed.**

1. `C:\Users\vedan\anaconda3\python.exe code/run_correctness_suite.py` →
   gate 1 19, gate 2 31, gate 3 27, gate 4 129, gate 5 150,
   `TOTAL 356 356 0 0`, `CORRECTNESS SUITE: ALL GATES PASS`, ~146 s.

2. **Parameter counts unchanged for all three arms.** Measured by extracting the
   T-L0.3 tree with `git archive 112809a code` into a scratch directory and
   building all three arms from a resolved config in *both* trees, same
   interpreter, model constructed exactly as `engine.py:202-229` does. Compared as
   JSON including a **per-tensor `numel` map**, because a compensating pair of
   shape changes would leave the total intact: `diff` **empty**.
   moe `3,201,555` / 54 tensors, mor `3,197,710` / 24, more `3,201,555` / 54;
   the MoR/MoRE gap is 3,845 params = 0.12%, the T4.x parameter-matching result.
   **`engine.py:220`'s comment citing `6,358,553` is the pre-T8.2 two-block
   figure and is not the baseline** — the baseline is what the T-L0.3 tree
   produces, which is what was measured.

3. **The 1-epoch arithmetic run is bit-identical, not merely close.**
   `--architecture mor --epochs 1 --seed 44` against the pre-change reference
   `runs/t_l03_mor_timing_seed44__fa9339bc` (built at `08e90427`):
   `config_hash` identical to all 64 hex digits
   (`fa9339bc279495e8b5bb8c604833526d90adf3a6a0808319d0cf927ffed90f73`, which is
   why the re-run auto-named itself `…__r3`), resolved config byte-identical at
   **2014 bytes** with `task` absent from both, and **all 93 comparable
   `metrics.json` keys equal** — `train/task_loss = 0.09949329204093187`,
   `val/task_loss = best_val_loss = 0.07614141606433052`,
   `train/total_loss = 0.12512240410806277`,
   `nondeterministic_ops_observed = ['none']` in both. Excluded:
   `experiment_id` (carries the `__r3` suffix by construction),
   `perf/throughput_tokens_sec` (1940 → 1599 tok/s — machine load, not a property
   of the model), `_non_scalar_keys_omitted`.

**One operational finding that will bite Phase L-7.** The first attempt died in
`wandb.init` with `UsageError: No API key configured`, **after**
`RunContext.create` had already made the run directory — leaving
`runs/t_l03_mor_timing_seed44__fa9339bc__r2` with a `config.json`,
`resolved_config.json` and a `stdout.log` holding the traceback, but no
`metrics.json`. It is **kept, not deleted** (CLAUDE.md §6: archive, never delete;
a directory with no `metrics.json` cannot be mistaken for a result), and its
`resolved_config.json` is in fact the artifact that proves the hash claim above.
The `stdout.log` is on-disk only — the standing `runs/**/stdout.log` rule excludes
it — which is why the error line is quoted here rather than cited.
Re-run with `WANDB_MODE=offline` and `WANDB_DIR` outside the repo, so no `wandb/`
directory landed in the tree. **A canonical language run will hit the same wall**
and leave the same directory-shaped debris; W&B credentials need settling before
the matrix, not during it.

**Where to look.** `TASKS_LANGUAGE.md` T-L1.4 Evidence for the tables;
`runs/t_l03_mor_timing_seed44__fa9339bc__r3/metrics.json` versus
`runs/t_l03_mor_timing_seed44__fa9339bc/metrics.json` for the comparison itself.

## T-L2.0 / T-L2.1 / T-L2.2 — the language corpus, and the two ways it can lie

`data/lang/build_language_dataset.py` (new), `code/test_language_data.py` (new),
`.gitignore` (new `data/lang/` section). Both corpora built end to end;
`code/test_language_data.py` → **65 passed, 0 failed, 0 skipped** on the CPU
interpreter. Gate L0 unaffected — the language suites are deliberately **not**
registered in `run_correctness_suite.py`, and the reason is written into both
files' docstrings: Gate L0's entire content is "the arithmetic total is still
356", so adding checks there would destroy the one number that says the published
arithmetic study still holds.

**Design: three stages that APPEND to one manifest.** `--stage fetch |
tokenizer | pack`, each reading `dataset_meta.json`, adding its own keys, and
writing it back (`save_manifest`). A stage never rewrites another stage's fields,
so a re-run of `pack` cannot silently change the recorded `hf_revision` — and the
canonical build, which takes ~6 minutes of encode, can be resumed without
re-downloading 539 MB.

**Every check is on measured file contents, never on the builder's intent.** This
is the whole design principle of `test_language_data.py`, because a dataset defect
is invisible at training time: a corpus with `<unk>` in it, a tokenizer fitted on
validation text, a short trailing block that makes `step_mask` conditional — none
of these raise, and all of them silently change what a loss number means. So:
`<unk>` is *counted* by reading every line back (0 in all six split/corpus pairs,
which confirms the `-raw-` choice empirically rather than on the dataset card);
block length is asserted on `a.shape[1]` and block count on `a.shape[0]`; the
tokenizer's training-file list is compared to the train path rather than trusted;
and the "no OOV" property is checked as *no `<unk>` is representable* (every
printable ASCII byte still encodes) rather than as *no `<unk>` was emitted*.

**The train-only tokenizer property is proved by an ABSENCE.** `write_train_text`
materializes only the train split's text, so TL2.1e can glob `*_text.txt` in the
corpus directory and require the set to be exactly `{train_text.txt}`. That is
strictly stronger than reading `tokenizer_training_files`: a held-out file cannot
have been passed through a path the manifest omits, because no such file exists.
Anyone tempted to dump a `val_text.txt` next to it for convenience will break this
check — that is intentional, and the scratch script that hashed the val/test rows
(below) reads through `fetch_splits()` in memory for exactly this reason.

**Two findings that will corrupt a results table if forgotten.**

1. **wikitext-2 and wikitext-103 ship byte-identical val and test splits.** Not
   just equal sizes — equal SHA-256 over the row text:
   `val 8ef74978…`, `test bbf94c53…`, 3,760 / 4,358 rows, in both corpora. So
   wikitext-2 is a *correctness* corpus and never a model-selection corpus: any
   hyperparameter or stopping point chosen on its val loss has been chosen on the
   canonical corpus's val set. `plan_language.md` §3 called it the "dev loop"
   without saying this.
2. **Per-token loss is not comparable across the two corpora.** Same val bytes,
   different BPE: 284,328 tokens under wikitext-2's tokenizer versus 281,335 under
   wikitext-103's (4.024 vs 4.067 bytes/token). The canonical tokenizer is the more
   efficient one, so a naive nats/token comparison flatters wikitext-103 by ~1% for
   nothing. Cross-corpus statements must be in bits per byte.

**Memory, because 134.7 M tokens is where the naive version dies.** `fetch_splits`
returns HF `Dataset` objects rather than lists (1.8 M rows of Python strings is
~1 GB); `_encode_stream_to_bin` streams uint16 to a raw `.bin` rather than
accumulating ids (a list of 134.7 M ints is ~3.6 GB); `pack_split` then reshapes
via `np.lib.format.open_memmap` in row batches and deletes the `.bin`. Canonical
train: 134,737,951 tokens → 526,320 blocks of 256, dropped tail **31**, 269 MB
`train.npy`, 194.6 s. The accounting identity
`tokens == blocks × seq_len + dropped_tail` holds exactly for all six pairs, so
the only loss is 325 tokens of 135.3 M across the canonical corpus, 2.4 × 10⁻⁶.

**Document segmentation is lossless, and that is measured, not assumed.**
`_is_doc_start` is a heuristic on ` = title = ` headings, and a mis-split would put
the EOT separators in the wrong places — a defect no downstream check would catch.
`train_text_bytes` equals the independent raw byte scan exactly (10,914,845 and
539,295,549), so joining rows with `""` reproduces the corpus byte-for-byte.

**`.gitignore`: the definition is tracked, the payload is not.**
`dataset_meta.json` (every number a language result is conditioned on, including
the per-split hashes the tests grade the arrays against) and `tokenizer.json` are
tracked; `*.npy`, `*_text.txt`, `*.bin` are not. `tokenizer.json` is tracked
*despite* being regenerable because BPE merge order depends on the `tokenizers`
version's tie-breaking, so a rebuilt tokenizer is not guaranteed byte-identical —
and a different tokenizer makes every recorded loss number incomparable. It is the
one build artifact cheaper to store than to trust. As everywhere else in that file,
this declines tracking, not existence (CLAUDE.md §6).

**Where to look.** `data/lang/build_language_dataset.py` `iter_documents` /
`_is_doc_start` when EOT placement looks wrong; `pack_split` when a block count
disagrees with the manifest; `code/test_language_data.py` TL2.1e when someone
wonders why a `val_text.txt` breaks the build's guarantees;
`TASKS_LANGUAGE.md` T-L2.0–T-L2.2 Evidence for the full tables.

## T-L3.0 — the POS family manifest, and two names it refuses to define

`code/more/lang_families.py` (new), `code/test_language_families.py` (new).
**24 passed, 0 failed, 3 skipped** (the skips are T-L3.1/3.2/3.3). Gate L0 after
the preceding commit `0c3a883`: `TOTAL 356 356 0 0`, ALL GATES PASS.

**Design: the same SHAPE as `families.py`, so nothing downstream needs plumbing.**
Same six-wide label space, same `expert_labels` contract including the E=1 and E=7
behaviour Gate 4 exists to protect, same `-1`-means-ignore convention with the same
value, `EXPERT_FAMILY_LABELS` aliased rather than renamed. The intended end state is
that a `task`-conditional import is the *only* difference between the arithmetic and
language metric paths.

**Two names are deliberately NOT defined, and they raise.** `more/__init__.py`
re-exports eight names from `families.py`; `OP_TO_EXPERT` has no language meaning,
because it maps the 16 arithmetic op codes to experts. `OP_TO_EXPERT = {}` for
parity was rejected: an empty dict turns `OP_TO_EXPERT[op]` into a bare `KeyError`
at whatever line indexes it, which reads as a missing operation rather than as a
caller reaching for the arithmetic axis mid-language-run. A PEP 562 module
`__getattr__` raises `AttributeError` with that explanation instead, driven by an
`_ARITHMETIC_ONLY` registry (`OP_TO_EXPERT`, `OP_TARGET_DEPTH`).

**The leak this does not close — Phase L-5 will need it.**
`more/__init__.py` imports from `.families` unconditionally, so
`from more import OP_TO_EXPERT` still yields the *arithmetic* mapping no matter what
`lang_families` does. Only the direct attribute is guarded. Closing the other path
means making the package export task-conditional, which is a Phase L-5 change and
is written down here so it is not discovered by a wrong number.

**The op axis is empty, not absent, and the emptiness is load-bearing.**
`ALL_OP_NAMES = []`, `NUM_OP_TYPES = 0`, `op_target_depth_table() → []`.
`engine.py:803` iterates `enumerate(ALL_OP_NAMES)`, so empty yields an empty
`op_avg_depth`, which the metric layer reports as *absent* rather than `0.0` —
CLAUDE.md §4's "no sentinel reported as a measurement", obtained for free rather
than by a special case. An empty depth table also beats a zero-filled one: it
cannot supervise depth whatever weight a future config sets, which turns
`plan_language.md` §5 from a written rule into a mechanical one. A one-element dummy
op would instead have produced a real-looking per-op depth number for an operation
that does not exist. **Phase L-5 note in the docstring:**
`depth_allocation_error(...)` receives this table with `step_ops`, so the language
path must mask on `step_ops >= 0` or skip the call before indexing — an empty table
indexed by anything raises, and that raise is preferable to a silent fallback.

**The auxiliary override is the BE paradigm and nothing else.** §4.1 assigns
auxiliary/copular verbs to L1, but Penn tags them `VB*` like main verbs, so some
override is unavoidable. It covers `be/am/is/are/was/were/been/being` and the
clitics `'s/'re/'m`, on the ground that BE has no English main-verb use other than
the copula — so the override asserts nothing the manifest has not already asserted.
`have` and `do` are **excluded on purpose** and land in L3 VERB: both have real
main-verb uses, and choosing between them is precisely what T-L3.1's type-level
majority vote is for. Hard-coding it would bypass the mechanism whose entire purpose
is to make that call from the corpus rather than from an author's intuition. TL3.0r
pins the consequence so a later reader cannot "fix" it without deciding. `not`/`n't`
*are* overridden to L1: Penn tags them `RB`, which would file negation with
open-class adverbs.

**The L6-vs-`-1` boundary, which the plan leaves overlapping.** §4.1 sends "subword
continuation pieces with no standalone POS" to L6; §4.3 sends "fragments that have
no POS" to `-1`. `surface_class_family` draws the line and says it is a design
choice: digits → L6, punctuation-only → L5, contains a letter → L6, everything else
(whitespace-only, lone continuation bytes, unused vocabulary entries) → `-1`.
Sending every no-evidence type to `-1` would make the 2% budget check vacuous;
sending everything to L6 would make L6 a catch-all and reproduce E7. T-L3.1 writes
each branch's share to the manifest so the split can be audited rather than trusted.

**Five guards raise at import**, following `families.py`'s discipline: no `/` in a
label, `NUM_FAMILIES == 6`, mapped and ignored tag sets disjoint, every mapped index
in `0..5`, every family reachable from at least one tag, and full coverage of the
45-tag standard Penn set. The reachability guard is the non-obvious one — a family
no tag can reach yields a structurally empty confusion-matrix row, which reads as a
routing failure rather than as a manifest defect.

**Two test checks are not restatements of the module, and both caught something.**
TL3.0e/f *parse* the `from .families import (...)` statements out of `metrics.py`
and `__init__.py`, so the required name list comes from the consumer rather than
from the test — adding an import there fails until someone classifies the new name.
That is what surfaced `OP_TO_EXPERT`. TL3.0w loads the file **by path** in a clean
subprocess and inspects `sys.modules`; the first version imported
`more.lang_families` and reported torch and numpy, which was correct — the package
`__init__` pulls them in. The docstring's "deliberately free of torch" was true of
the file and false of the import, and a reader would have read it as "loading this
is cheap"; both the probe and the docstring were corrected.

**Where to look.** `code/more/lang_families.py` `_ARITHMETIC_ONLY` when an
`AttributeError` names a family accessor; `PENN_TO_FAMILY` when a family's token
share looks wrong; `surface_class_family` when the unmapped share moves;
`op_target_depth_table` when a language run reports a depth allocation error at all
(it should report `N/A`).

## T-L2.5 / T-L2.6 / T-L2.7 / T-L3.1 — GATE L2 passes, and the floor is 4.9849

Four builders and two suites. `code/test_language_data.py` → **104 passed, 0 failed,
0 skipped**; `code/test_language_families.py` → **57 passed, 0 failed, 2 skipped**
(T-L3.2/3.3 unbuilt); `code/test_language_task_axis.py` → **72 passed, 0 failed**.
New: `data/lang/build_family_lookup.py`, `data/lang/build_baseline_floors.py`,
`data/lang/build_depth_deciles.py`. Changed:
`data/lang/build_language_dataset.py` (`update_manifest`),
`code/more/lang_families.py` (corpus markers, deciles),
`code/canonical_spec_language.json` (`protocol.primary_metric_floor`).

**THE NUMBER: `primary_metric_floor = 4.9849` nats/token** on wikitext-103 val
(7.1918 bits, perplexity 146.2). Fitted on train, evaluated on val:
uniform 9.010913 = ln 8192 = 13.0000 bits exactly, unigram 7.198213, backoff bigram
4.984947. A canonical language run that does not beat 4.9849 has learned nothing a
two-column count table could not. Frozen into
`canonical_spec_language.json:protocol.primary_metric_floor`, whose note previously
named the *unigram* as the language floor — which `plan_language.md` §3 contradicts —
and had `ln(8192) = 9.0106`, wrong in the fourth decimal. Both corrected.

**Naming corrected rather than glossed.** The plan says "Katz-backoff"; what is
implemented is the **absolute-discounting** variant of the same backoff structure,
`D = 0.75`, with `λ(v) = D·types_after(v)/c(v)` of leftover mass on the unigram — not
Katz's Good–Turing discounting. `bigram_smoothing` records the actual method so
nobody reads "Katz" and assumes Good–Turing. Unigram smoothing is picked by
measurement, not default: 80 of 8192 types are unseen in canonical train, so add-1
applies and is recorded; with no zeros the builder uses MLE and says so.

**The train-only property is proved by interception.** TL2.5g wraps `numpy.load` for
the duration of `fit_unigram` *and* `fit_bigram` and records every path: `called 2x,
all on ['train.npy']`. A fit that read val through *any* path, including one the
manifest does not mention, shows up. TL2.5f re-derives `unigram_ce` inside the test
file and requires 1e-9 agreement, catching a writer that computed one number and
recorded another. The bigram is deliberately not re-derived — re-implementing its
discounting in the test would check only that the same author wrote the same formula
twice.

**GATE L2's overlap check is exact content, not hashes.** Raw 512-byte block bytes as
dict keys: 2,354 held-out canonical blocks is 1.2 MB, so exactness costs nothing and
there is no collision argument to make. All **526,320** canonical train blocks
streamed against that set — zero overlap, and val/test share no block with each other
either (2,354 distinct from 2,354 stored, so no duplicates inside the held-out data
at all). "The manifest hash is reproducible" is read as the one hash the manifest
publishes: `dataset_version` re-derives from the recorded per-split SHA-256s.

**A SILENT MANIFEST CLOBBER, found by a SKIP.** `build_baseline_floors.py --corpus
wikitext-103` finished while `build_family_lookup.py --corpus wikitext-103` was still
tagging; the lookup then wrote back the manifest snapshot it had loaded 25 minutes
earlier, **erasing all four floor keys**. Nothing raised, the JSON stayed valid, and
no check failed — TL2.5 reported `[SKIP] floors not built for this corpus`, and the
reason string is what made the absence visible instead of merely absent. That is the
argument for skipping loudly rather than conditionally not checking. Fixed with
`bld.update_manifest(corpus, new_fields)`, which re-reads the on-disk manifest
immediately before merging; all three stage builders use it. The residual race is in
that docstring rather than papered over: overlapping read-and-write windows can still
lose keys, the window is now milliseconds instead of minutes, and two stage builders
must not run concurrently on the same corpus.

**FINDING — the oracle POS partition is not balanced, so the balance loss now fights
the specialization metric.** `H/log(6) = 0.8884` for the partition itself;
largest/smallest family token ratio **5.29×**. A router that perfectly reproduced POS
scores 0.8884, so a router driven toward 1.0 by the balance term must *disagree* with
POS by construction. On arithmetic the oracle families were near-uniform and this
question did not arise. Every language load-entropy number must be quoted against the
partition's own entropy, never against 1.0. TL3.1l pins it.

**FINDING — the type/token divergence T-L2.4 guards against is extreme.**
L1_FUNCTION is 2.6% of types and 27.3% of tokens (10.7×); L5_PUNCT_SYM is 0.57% of
types and 8.9% of tokens. TL3.1k requires at least one family to diverge by >10
points, so the manifest's two denominators are justified by measurement.

**Whitespace tokenization, chosen by measurement.** `TreebankWordTokenizer` yields
1.0251× more tokens, and every extra unit is one that **does not exist in the byte
stream the BPE was fitted on**: it splits WikiText's `@-@`/`@.@`/`@,@` escapes into
`@`+`-`+`@` (2,830 per ~330 k tokens), rewrites `"` as Penn's ` `` `, splits `cannot`,
strips the period off `U.S.`. Tagging a unit absent from the corpus breaks the
word-to-id alignment the vote depends on. Those markers are ~0.86% of tokens and got
an explicit `CORPUS_MARKER_FORMS → L5` override, because the tagger has no opinion
worth having about them — `@-@` is not English, so the perceptron falls back on shape
features and emits whatever the context suggests.

**Two memory defects fixed before the canonical build, both measured.**
`list(iter_chunks(...))` materialized 1.8 M lines of the 539 MB corpus as Python
strings: **1,013 MB resident before a word was tagged**. A lazy generator would not
have fixed it — `Pool.imap_unordered` drains its input into an unbounded queue. Now
the parent passes `(index, start_byte, end_byte)` triples and each worker reads its
own slice. Separately, each of 6 workers was **358 MB** because importing
`more.lang_families` runs `more/__init__.py` and pulls in torch — the exact leak
TL3.0w documents — so the manifest is now loaded by path. After: parent 45 MB,
workers 161 MB, **1,025 MB total against ~3,100 MB**, and throughput rose to 104 k
words/s. The refactor was gated on reproducing the pre-refactor lookup byte-for-byte,
which it did (`sha256 731fc370ecd08d87`, also identical between `--workers 1` and
`--workers 6`).

**Deciles are cut at equal TOKEN mass**, and the Zipf numbers show why that is not
stylistic: 3 types carry 10.5% of the canonical corpus while 4,265 types (52% of V)
carry the last 10%. Equal-type-count deciles would have assigned one depth to nearly
every token the loss weights. The ablation-F confound is measured — decile-vs-family
normalized MI **0.2724** — so depth allocation and routing are correlated but not
near-equivalent claims, provided the number is quoted against 0.27 rather than 0.
Measurement and choice are split across files on purpose: `token_decile.npy` is
measured, `lang_families.decile_target_depth(decile, max_depth)` is the choice and
takes `max_depth` as an argument so the artifact hard-codes no recursion budget.

**Deliberately still null in `canonical_spec_language.json`.** `dataset_version` and
`seq_len` stay unfrozen even though the corpus is built and Gate L2 passes: the
version string encodes `-len256-`, and `seq_len` is Gate L5's to set from measured
throughput and VRAM headroom on the actual GPU. Freezing now would either pre-empt
that measurement or guarantee a re-freeze. Both `frozen_by` notes were rewritten to
say the corpus IS built and what specifically still blocks each field, so a future
reader does not read the nulls as an oversight.

**Where to look.** `build_family_lookup.py` `chunk_offsets` / `_load_manifest_module`
for the memory fixes; `majority` when parallel and serial builds disagree;
`build_language_dataset.py` `update_manifest` when a manifest key goes missing;
`build_baseline_floors.py` `evaluate_bigram` for the discounting formula;
`lang_families.CORPUS_MARKER_FORMS` when L5's token share moves;
`TASKS_LANGUAGE.md` T-L2.5–T-L2.7 and T-L3.1 Evidence for the full tables and the
40-types-per-family spot sample.

## T-L2.3 / T-L2.4 — Phase L-2 closed: the loader, the schema, and a rebuild

`code/more/lang_data.py` (new), `code/more/__init__.py`,
`data/lang/build_language_dataset.py` (`MANIFEST_SCHEMA`, `validate_manifest`,
`set_out_root`, `_repo_relative`), `code/test_language_data.py`.
**133 passed, 0 failed, 0 skipped.** Every box in Phase L-2 is now ticked.

**`MoRELanguageDataset` keeps the engine's 7-tuple arity rather than introducing a
signature.** `engine.py:438` and the validation loop at 686 both unpack positionally;
a second arity would force a second training loop, which `plan.md` §9 forbids. TL2.3a
reads the expected arity out of `MoREDataset.__getitem__`'s own source, so a change
on the arithmetic side fails the language check instead of drifting past it. Slots
with no language meaning carry the documented ignore label: `step_ops` all `-1`
(the operation axis is *absent*, so every `step_ops >= 0` mask is empty and per-op
metrics report `N/A`), `family` `-1` (a 256-token block has no lexical class, which is
also why `family_cls_weight = 0.0`).

**Slot 6 is NaN, and that is a poison value rather than a placeholder.** The LM target
is derived by shifting `input_ids` inside the loss so the ids are not stored twice. The
slot still has to hold something collatable, and `0.0` or `-1.0` would let a mis-wired
language loss train silently against a constant and draw a plausible curve. NaN
surfaces that on the first optimizer step. It is never logged, so this is the opposite
of a sentinel reported as a measurement — it is a value that cannot be mistaken for one.

**The memmap is opened per process, keyed on pid.** A `np.memmap` held as an attribute
pickles by materializing, so `num_workers > 0` under Windows spawn would hand every
worker its own **269 MB** copy of the canonical train split. The family lookup is
validated on load rather than trusted: wrong length means a different tokenizer, and a
family index >= 6 would silently widen the oracle label space — the T8.3 defect.

**Determinism is checked on emitted block CONTENT, not on indices.** An index
permutation that were reproducible while `__getitem__` was not would pass an
index-level check and still be non-deterministic. Two epochs at seed 42 give identical
first batches, seed 43 differs, and with `shuffle=False` block 0 of the loader is block
0 of the file — so the manifest's per-split hash describes exactly what the model
reads. No new generator purpose was added; `make_generator(seed,
"dataloader_shuffle")` is the arithmetic one.

**`more/__init__.py` re-exports nothing from `.lang_families`, on purpose.** The eight
names above come from `.families` unconditionally; putting a second manifest's labels
in the same namespace is how `EXPERT_FAMILY_LABELS` ends up meaning whichever module
imported last. Language code imports `more.lang_families` by name.

**"The documented schema" is now an object.** `MANIFEST_SCHEMA` maps
`key -> (stage, type, required)` across 6 build stages and 50 keys, so a missing key
names *which stage* failed to write it, and `validate_manifest` returns every problem
rather than raising on the first. The `bool`-is-a-subclass-of-`int` trap is handled
both ways. The comment beside it writes out the correspondence with
`data/dataset_meta.json` instead of leaving "mirrors" as an assertion — including the
four fields language legitimately lacks (`operation_counts`, `seed`, `train_frac`,
`test_frac`). That last group reinterprets T-L2.4's Verify: there is **no seed** to
rebuild "from the same seed" with, because nothing is sampled, so the check is plain
determinism — the stronger requirement.

**The rebuild clause is executed.** TL2.4d builds wikitext-2 from scratch through all
three stages into a scratch out-root and compares nine fields: `dataset_version`, all
three per-split SHA-256s, token and block counts, dropped tails, and — the one that
was not guaranteed — **`tokenizer.json` byte-identical**. So on a fixed `tokenizers`
version the BPE fit is deterministic; the `.gitignore` note about merge-order
tie-breaking is a cross-version caution and now says so.

**The T-L0.3a cross-drive defect was still latent in this file, and this check fired
it.** `fit_tokenizer` recorded its training-file list with
`os.path.relpath(train_text, _REPO)`, and Windows `relpath` **raises** across drives:
`ValueError: path is on mount 'C:', start on mount 'D:'`. The repo is on `D:`, the
scratch out-root under `C:\Users\...\Temp`. Invisible while every build wrote under
`data/lang/`. Fixed with a local `_repo_relative()` mirroring
`more/run_context.py:_repo_relative`. An off-drive rebuild therefore records absolute
paths in ITS manifest, which is correct, and is why TL2.1d grades only the canonical
location.

**A ledger number was wrong and is corrected.** T-L2.0's Evidence table gave
wikitext-2's non-empty line counts as 31,175 / 3,213 / 3,760; the manifest and
TL2.0l's own output say **23,767 / 2,461 / 2,891**. The test read the right values
throughout — only the hand-written table was wrong, which is the case for
`updated_rules.md`'s "never hand-copy numbers into a results table".

**Where to look.** `lang_data.py` `_blocks` when a DataLoader worker balloons;
`_load_family_lookup` when a lookup/tokenizer mismatch is suspected; `__getitem__`'s
slot comments before changing the engine's unpacking;
`build_language_dataset.MANIFEST_SCHEMA` when a manifest key is added;
`set_out_root` before rebuilding anything in place.

## T-L3.2 / T-L3.3 — the POS number is demoted, and given a floor to be read against

`code/more/metrics.py`, `code/more/engine.py`, `data/lang/build_shuffled_control.py`
(new), `code/test_language_families.py`, `plan_language.md` §3.1.
**89 passed, 0 failed, 1 skipped.** Gate L0 after the metric-layer change:
`TOTAL 356 356 0 0`, ALL GATES PASS — arithmetic untouched.

**One helper owns the routing key's NAME, and five call sites read it.**
`routing_agreement_key(task)` gives `val/routing_accuracy` for arithmetic and
`val/routing_agreement_with_pos` for language, and is the only place either string
appears: the W&B log dict, the console line, the `results.tsv` header, and the
`metrics.json` `"N/A"` contract. TL3.2m asserts `engine.py` hard-codes it nowhere.
The NUMBER is untouched — TL3.2f pins the language value equal to the arithmetic one
and to the confusion diagonal at 1e-15. Only the name moved.

**The default stays arithmetic, and that is load-bearing.** G5.2b checks the literal
`val/routing_accuracy` on a single-expert run, and `resolve_task` makes absence mean
arithmetic so no existing config's hash moves. TL3.2g calls the log-dict builder with
no `task` and requires all 46 keys to match the explicit-arithmetic call. An unknown
task **raises** (TL3.2d): a default would publish a language agreement figure under
`val/routing_accuracy`, silently, in the key the exporter joins on. The two spellings
are deliberately different strings so no exporter can average a column of accuracies
against a column of agreements.

**"Primary" became an object.** `LANGUAGE_SPECIALIZATION_ORDER` is
`hungarian_accuracy -> ami -> purity -> matched_macro_recall -> agreement_with_pos`
with the agreement key LAST, and TL3.2i checks every name in it is a key a real
language log dict emits — consumable, not aspirational. The Phase L-9 results writer
consumes this list rather than reimplementing the convention.

**The caption travels in the artifact, not in prose.** `metrics.json` carries
`routing_agreement_metric_key` and `routing_agreement_caption`. A caveat that lives
only in a plan file gets omitted from a table; one that lives only on stdout is not
kept at all. The language caption opens `AGREEMENT WITH A PRIOR, NOT ACCURACY`, names
the POS partition a linguistic hypothesis rather than a functional ground truth, and
carries the T-L3.1 finding forward: read load entropy against the partition's own
0.8884, not against 1.0.

**T-L3.3's Verify clause was not implemented as written, and the departure is the
substance.** The ledger says "holding the per-family TYPE counts fixed";
`plan_language.md` §4.4 says "the observed family TOKEN proportions". Those are
different partitions and on this corpus they are far apart — L1_FUNCTION is 2.6% of
types and 27.3% of tokens. **§4.4 is correct and is what got built.** Every entry of
the routing confusion matrix is a TOKEN, so AMI's and Hungarian accuracy's chance
levels depend on the token marginals; a type-count-matched control would be roughly
token-uniform against the real partition's 5.29x imbalance, and the comparison would
confound "meaningless" with "differently balanced" — the exact confusion the control
exists to remove. Realised type counts are recorded anyway (L1: 243 real vs ~2,506
control) so the departure is visible, and `shuffled_control_matched_note` states it.

**THE NUMBER THE PAPER NEEDS: the AMI floor for a 6-way partition with these
marginals is 0.0527, not 0.** Verified on two synthetic routers with no trained model:
a random 6-way router scores AMI 0.0000 against POS and -0.0000 against the control
(delta +0.0000); a POS-perfect router scores 1.0000 against POS and 0.0527 against the
control (delta +0.9473, z = 47.5). So the control behaves as a floor for one and not
the other — it separates a real partition from chance, which is its whole job. An
observed AMI of 0.06 against POS would be AT the floor, and without this control it
would have read as weak specialization.

**Matching, measured rather than argued.** Worst per-family token-share error
9.74e-05 (wikitext-2) and 3.77e-03 (wikitext-103) over ten draws, recomputed in the
test from `train.npy` rather than read from the manifest. The quantity that actually
sets AMI's chance level is the marginal ENTROPY, and it agrees to 1e-4: 0.894156 real
vs 0.894249 +- 2.9e-04 control. Unmapped types are EXACTLY preserved in every draw,
so both metric sets are computed over the same token population — otherwise the pair
is not comparable.

**The repair pass needed a SWAP, not just a move.** Types are visited in seeded random
order and assigned to whichever family is furthest below its token-mass target;
randomising the order is load-bearing, because a frequency-ordered pass would rebuild
part of the real partition through the frequency-family correlation (normalized MI
0.27). The greedy pass leaves a tail: with 3 types carrying ~4% of the canonical corpus
each, one arriving late overshoots. One-way moves fixed nine of ten draws to ~5e-05 and
left canonical draw 0 stalled at 1.4e-02, because the over-family held no type small
enough to transfer without overshooting. A swap moves an arbitrarily small NET mass out
of two large types, so the granularity floor disappears. Every selection is by mass
alone and never consults the real partition, so the repair cannot reintroduce POS
structure.

**Ten draws, not one.** T-L3.3 requires the difference "with its uncertainty, not just
the raw pair", and one control gives a point estimate with no spread.
`specialization_vs_control` returns `real`, `control_mean`, `control_std`, `delta` and
`delta_z` per metric; `control_comparison_to_wandb` OMITS undefined entries rather than
writing 0.0, because a 0.0 in a "real minus control" column reads as "POS is no better
than noise", which is a finding rather than a missing value. `delta_z` is `None` when
the control has no spread — a z with a zero denominator is undefined, not large. Seed
20260903, kept out of the frozen run-seed space so a data artifact is never tied to a
training seed.

**Corpus decision recorded in `plan_language.md` §3.1: FineWeb-Edu rejected as
canonical, kept as the preferred Phase L-10 robustness ablation.** The case for it is
real and it is about downstream capability — HuggingFace's ablations show 12-24%
relative gains on MMLU/ARC/OpenBookQA over general web snapshots — but those ran at
~1.8 B parameters, and this spec's frozen shape (`d_model` 256, 4 heads,
`num_blocks` 1, `V` 8192, tied head) is single-digit millions. A model that size sits
at chance on MMLU whatever it trained on, so the one axis FineWeb-Edu measurably wins
on is one this paper cannot measure. Switching would cost the three properties that
ARE load-bearing here: author-provided document-disjoint splits that Gate L2 audits as
the dataset's property rather than ours; a floor a reader can situate in the WikiText
literature; and no sampling decision for the proxy guard to enforce. The legitimate
part of the objection — that a partition learned on encyclopedic register may be an
artifact of it — is answered by a second corpus as a labelled ablation, not by moving
the canonical arm.

**Where to look.** `metrics.routing_agreement_key` when a routing key is misspelled in
an artifact; `LANGUAGE_SPECIALIZATION_ORDER` before writing the language results
section; `build_shuffled_control._repair` when a control draw's marginals drift;
`specialization_vs_control` when a delta or its z looks wrong;
`TASKS_LANGUAGE.md` T-L3.2/T-L3.3 Evidence for the synthetic-router table.

## T-L4.0 .. T-L4.4 — GATE L1 passes, and bit-identity turned out to need a footnote

`code/more/model.py`, `code/test_lang_causality.py` (new), `plan_language.md` §6.4.
**22 passed, 0 failed, 0 skipped.** Gate L0 after the change: `TOTAL 356 356 0 0`,
ALL GATES PASS, and the arithmetic per-tensor parameter map is byte-identical to the
T-L0.3 baseline (moe 3,201,555 / 54 tensors; mor 3,197,710 / 24; more 3,201,555 / 54).

**Why this change had to happen at all.** The model had no attention anywhere. For
arithmetic that was the right minimal design; for causal LM it is degenerate — with no
cross-position path the best achievable model is the unigram distribution, every arm of
the matrix would converge to `unigram_ce`, and the MoE/MoR/MoRE comparison would
measure nothing.

**One `nn.MultiheadAttention` plus its own norm per wrapper, and the sharing is
asserted by STORAGE IDENTITY.** 4,224 params at `d_model = 32` = 4·d² + 4·d exactly,
unchanged between `max_depth` 4 and 7. TLC.0d hooks `attn.forward` and records
`data_ptr()` for `in_proj_weight`, `out_proj.weight` and `attn_norm.weight` at every
depth step: one distinct set across four steps. Two distinct modules with equal shapes
would pass a shape check and silently break `updated_rules.md` §2.1 — the invariant
that makes depth computation through time rather than a stack, i.e. the invariant that
makes the MoR and MoRE arms mean what they claim.

**`attention=False` builds NOTHING, following the `router_noise_scale` precedent.**
`hasattr(w, "attn")` is False; the state_dict gains exactly six keys when the flag is
on and loses none. Compared per-tensor rather than by total, because a compensating pair
of shape changes would leave a total intact.

**THE INTERESTING PART: `plan_language.md` §6.4's "bit-identical" is achievable only
at `num_experts = 1, max_depth = 1`, and the residual above it is not a leak.**

| configuration | masked | mask removed |
|---|---|---|
| E=1, depth 1 | **0.0 exactly** | 0.37 |
| E=1, depth 4 / 7 | 1.2e-07 (1 ULP) | 0.93 / 0.95 |
| E=6, depth 1 / 4 / 7 | 2.4e-07 (2 ULP) | 0.37 / 0.51 / 0.51 |

The cause is **grouped Top-1 dispatch**. A perturbation that flips the perturbed
token's expert changes the per-expert row counts — measured `[9,6,4,3,2,0] ->
[8,7,5,2,2,0]` — so two `nn.Linear` GEMMs get different shapes, tile differently, and
move the shared rows in their last one or two bits. For `E = 1` at depth > 1 the same
thing happens through `N_active` when a halt decision changes.

**The mechanism is proved rather than assumed.** TLC.4e runs `E=6, depth 1` with
attention ENTIRELY ABSENT and still measures a 2.4e-07 shift in the past, while
`E=1, depth 1` is exactly 0.0. That isolates the cause to the grouping and clears the
mask. Without this comparison, "we allow 2.4e-07" would have been an unexplained
tolerance in the one test where a tolerance is dangerous.

So the gate is three parts, which together assert MORE than `torch.equal` alone:
exact bit-identity where the confound is absent; an 8-ULP bound (9.5e-07) for every
architecture, worst observed 2.4e-07; and the mask-removed variant exceeding that
bound by >= 1e5 in all five configurations (smallest ratio 2.1e+06). §6.4 now carries
the amendment and the table.

**A second test bug worth recording, because the naive form is the obvious one.**
TLC.2d's first version was `"is_causal=True" in src` — and it FAILED, on the
`causal_mask` docstring that explains why the hint is not used. It now walks the `ast`
for a call keyword actually named `is_causal` with a literal `True`. A string search
over source for a code property is a check that can be defeated by a comment.

**Halted tokens: frozen bitwise, attendable behaviourally.** The freeze is asserted
with `torch.equal` on the state the attention sublayer receives at each depth
(forward pre-hook): 24 positions, exit depths {2,3,4} over 4 captures, zero
violations, and TLC.3b confirms the tokens really do halt at different depths so the
assertion is not empty. "Attendable" is checked as behaviour: perturbing an
early-halting position (depth 2) moves later positions in its own sequence by
**0.720**, six orders above the tiling floor — so halting early cannot blind position
40 to the word at position 4, which would couple one token's prediction to another
token's halting decision.

**The waste is reported and it moves.** `route_stats.attn_query_waste_fraction` is
39.58% at `max_depth = 6` (38 of 96 query slots) and exactly 0.0 at `max_depth = 1`,
where nothing has halted — so it tracks the halted fraction rather than reporting a
constant. Aggregated across blocks as one fraction over summed slots, not a mean of
per-block fractions, so a block that ran seven depth steps contributes seven steps'
worth. On the arithmetic path the key is ABSENT rather than 0.0: there is no attention
sublayer, so there is no waste, and a 0.0 would be a sentinel standing in for "not
applicable".

**Position embeddings sit in `MoREModel`, not the wrapper**, under the same flag, and
are added ONCE before the depth loop — position is a property of a token's place in the
sequence, not of how much computation it has received, and re-adding per step would
make the positional signal grow with depth. `attention=True` requires `max_seq_len` and
raises without it; a default would silently truncate or oversize the table.
Learned-absolute over RoPE/ALiBi because both of those modify attention scores, and a
positional signal that varied with step count would confound the depth analysis.

**Where to look.** `MoREWrapper.causal_mask` for the mask orientation and why
`is_causal` is refused; the depth-loop step 0 block for the attend-then-freeze order;
`route_stat_totals["attn_query_*"]` when the waste number looks wrong;
`test_lang_causality.py` TLC.4e first, whenever a future Gate L1 failure needs
splitting into "real leak" versus "dispatch tiling".

## T-L5.0 .. T-L5.4 — the LM head, and two init defects the ln(V) check caught

`code/more/model.py`, `code/more/engine.py`, `code/more/config.py`,
`code/test_lang_heads.py` (new). **25 passed, 0 failed, 0 skipped.** Gate L0:
`TOTAL 356 356 0 0`, ALL GATES PASS.

**Heads are now constructed per task, not shared as a superset with some weighted to
zero.** Language builds `tok_embed`, `lm_head` (tied), `family_probe`, `pos_embed`;
arithmetic keeps `step_proj`, `op_embed`, `regression_head`, `cls_head`,
`step_cls_head`. Each set is ABSENT on the other path, which is the T-L5.2 argument:
an unused head still contributes parameters to the budget comparison and still invites
a future reader to weight it. `op_embed` was the one that mattered most to drop — on
language it would have been `nn.Embedding(0, d_model)`, since
`lang_families.NUM_OP_TYPES` is 0. Legal to build, raises only when indexed.

**THE `task_loss ~ ln(V)` CHECK PAID FOR ITSELF ON THE FIRST RUN. With
`nn.Embedding`'s default N(0, 1) the initial loss measured 167.6 nats against a floor
of 9.011** — an 18x overshoot, perplexity ~1e73. Weight tying makes the embedding's
init scale an OUTPUT-LOGIT scale: `logits = h @ W.T`, so at unit-RMS `h` the logit
spread is `std(W) * sqrt(d_model) = 1 * 16`, and a 16-nat spread over 8192 classes is a
confidently wrong distribution rather than a uniform one. A model starting there spends
its first epochs undoing its own initialisation, which shows up as a suspiciously steep
early loss curve and never as an error. Fixed with std=0.02 (GPT-2 convention) on both
`tok_embed` and `pos_embed` — the same scale on both because they are ADDED, so an
N(0,1) positional table beside an N(0,0.02) token table would make position 50x louder
than identity at init. After: 9.0277..9.0898 over the five frozen seeds, worst relative
error 0.88%.

**A second number worth having: a mis-shifted language run would report ~5.6 nats at
epoch 0.** Scoring the UNSHIFTED target gives 5.5552 — 3.46 nats below the uniform
floor, before any training — because a tied head over a residual trunk already peaks
`logits_i` at `x_i`. That is the copy shortcut tying gives for free, and 5.6 is close
enough to the 4.98 bigram floor to be genuinely deceiving. Gate L1 detects the general
case; TLH.1d pins the magnitude.

**T-L5.3's real test is the gradient comparison, not the `requires_grad` check.**
`assert not h.requires_grad` at the probe input is necessary and insufficient — a
detached tensor that still routed gradient to the trunk through a second path would
pass it. **Trunk gradients are BITWISE identical at probe weight 0.0 / 0.5 / 1.0: 47
tensors, `torch.equal`, zero differences.** And the check is not passing because the
probe is inert: the probe's own gradients are exactly zero at weight 0, non-zero at 1,
and exactly half the weight-1 values at 0.5.

What that buys is the removal of a standing caveat. On arithmetic `family_cls` sits in
the objective at weight 0.5 and shapes the trunk, so "MoRE's representation separates
operation families" is permanently weaker there — which is why that study needs the
`no_family_supervision` ablation. Language does not inherit the confound at all: every
routing / Hungarian / AMI / purity number on the language arm measures emergent
structure with no family signal anywhere in the trunk's objective.

**Weight tying measured against an untied copy rather than against a formula:**
5,523,468 tied vs 7,620,620 untied, exactly 2,097,152 saved. The expert stack plus
attention is the other 3,426,316, which is what makes a MoR/MoRE budget comparison a
statement about them rather than about lookup tables. The tie is real backward too: one
backward pass puts gradient on 8,192 of 8,192 embedding rows against only 64 distinct
ids in the batch — the excess is the output path.

**`results.tsv` gained three APPENDED columns**, never inserted: `val_perplexity`,
`nats_below_bigram_floor`, `depth_rho_model_loss`. Verified against a Gate-5-written
arithmetic file that the ten pre-L5 columns are unchanged IN POSITION and all three new
ones read `N/A` — `exp()` of an MSE is not a perplexity. `depth_rho_model_loss` is
Phase L-6's measurement and the column is reserved so the header never has to be
reordered later.

`val/nats_below_bigram_floor` reads the floor from the DATASET MANIFEST via
`MoRELanguageDataset.primary_metric_floor`, not from a config literal, so a run cannot
quote a floor measured on a different corpus. The probe CE is published as
`probe/family_ce` on language and `train/step_cls_ce` on arithmetic — one tensor, two
names, because on one task it is a measurement and on the other it trains the trunk.

**One structural refactor, forced and worth recording.** `MoREModel` must branch on the
task to decide which heads exist, and `config.py` imports `model.py` and never the
reverse (`run_context.py:52`). So `TASK_ARITHMETIC` / `TASK_LANGUAGE` moved DOWN into
`model.py`, with `config.py` re-exporting them unchanged. The alternative was
duplicating two string literals across two modules, against the one-manifest principle.
No call site moved.

**Where to look.** `model.py`'s task branch in `__init__` when a head is unexpectedly
present or absent; the `nn.init.normal_(..., std=0.02)` comment when an initial loss is
not near ln(V); `engine.py`'s `task_loss` branch for the shift; `_floor` for where the
baseline comes from; `test_lang_heads.py` TLH.3b before touching anything about the
probe.

## T-L6.0 … T-L6.4 — GATE L3 and GATE L4 pass; depth gets a null band and no target

`code/more/metrics.py`, `code/more/engine.py`, `code/more/lang_data.py`,
`code/more/config.py`, `data/lang/build_depth_deciles.py`,
`code/derive_lang_ffn_mult.py` (new), `code/test_lang_recursion.py` (new),
`code/canonical_spec_language.json`. **11 passed, 0 failed** on the new gate suite;
Gate L0 unchanged at `TOTAL 356 356 0 0`.

**T-L6.0. The allocation-error keys are written as the STRING `"N/A"`, not omitted.**
They are structurally undefined on language: `op_target_depth_table()` is empty by
construction, so `val_depth_err_n` is 0 and the arithmetic branch never fires. Writing
`"N/A"` makes the absence a statement an exporter can join on. A 0.0 or -1 there would
read as perfect allocation against a curriculum that does not exist — §5.1's "single
most misleading number this migration could produce".

**T-L6.1. Spearman is hand-rolled, and the ties are the whole difficulty.** Exit depth
is an integer, so on a collapsed-depth model almost every value is tied;
`argsort().argsort()` gives ordinal ranks that break ties by ARRAY POSITION and would
manufacture a confident rho out of batch arrival order. The arithmetic POC put 93-97% of
tokens at exactly 2 steps, so this is the regime, not a corner case. Averaged ranks
instead, **verified against `scipy.stats.spearmanr` to 1.7e-16 over 200 trials** with a
third of them heavily tied — exactness measured rather than claimed, and the metric layer
keeps its torch/numpy-only imports. Returns **`None`, not 0.0**, when either side is
constant, because "measured no relationship" and "no relationship is definable" are
different findings and the collapsed-depth model produces the second.

New artifact `token_train_count.npy` (per-type train counts, hash in the manifest) so
`log_freq` and `unigram_surprisal` derive from ONE source and cannot disagree about
smoothing — the surprisal reads `unigram_smoothing` from the manifest so it describes the
same distribution as the `unigram_ce` floor the run is quoted against. `log1p` rather
than a masked log: 80 of 8192 canonical types are unseen, and `log(0)` in a rank vector
is a NaN generator rather than an extreme value.

**T-L6.2. The null band, and it discriminates:**

| depth vector | rho | null band | exceeds? |
|---|---|---|---|
| 100% at depth 2 | **None** | — | undefined, correctly |
| 97% at depth 2 (the POC's regime) | -0.0277 | [-0.0448, +0.0381] | **No** |
| genuinely correlated | +0.8577 | [-0.0374, +0.0342] | **Yes** |

The permutation destroys ONLY the pairing, so both marginals — including the tie
structure that causes the problem — are preserved exactly. Ten... two hundred
permutations per rho, `exceeds_null` is `None` rather than `False` when rho itself is
undefined. `language_depth_to_wandb` omits undefined entries rather than zeroing them: a
0.0 in `depth/spearman_vs_model_loss` would read as "measured: the model allocates
compute unrelated to difficulty", which is a finding — and it is the specific finding the
arithmetic POC made, so the two must stay distinguishable.

**T-L6.3. Re-derived, and the answer is 24 — the same as arithmetic, for a structural
reason.** `derive_lang_ffn_mult.py` prints the table and writes
`lang_ffn_mult_derivation.json`. MoRE reference at E=6/mult 4: 5,584,908 total,
3,422,220 non-embedding. Only ffn_mult 23/24/25 satisfy the 5% requirement on BOTH
counts; 24 gives rel(total) 0.069% and rel(non-embedding) 0.112%. Attention (~263 K) and
the tied 2,097,152-element embedding are added EQUALLY to every arm and cancel in
`P_MoR - P_MoRE`, leaving the expert-stack condition `6 experts x mult 4` against
`1 x mult 24` unchanged from arithmetic. `rel(non-emb) > rel(total)` at every row, which
is T-L6.3's premise confirmed: the shared embedding sits in both numerator and
denominator and shrinks the relative gap for free, so the total alone is too easy.
**Insensitive to `seq_len`** (128/256/512/1024 all give 24) because the positional table
is also identical across arms — Gate L5's seq_len decision and this one are independent.

**GATE L3 PASSES, under attention.** One block object, 5 recursion calls, **1 distinct
object id**: identity rather than tensor equality, because a stack of
independently-initialised blocks could satisfy equality for a single step. Halting
suppressed (`bias = -30`) → every token force-exits at `max_depth`, `early_exits = 0`;
saturated (`bias = +30`) → every token exits at depth 1, `forced_exits = 0`. The two
extremes bracket the mechanism, so a depth that never varied would fail one of them.
Balance term across a 9x depth range (1/3/5/9): -0.7400 / -0.7056 / -0.6082 / -0.6082,
relative spread 19.8% — T4.1's normalization property re-measured with attention in the
loop. Two eval passes give bit-identical logits and exit markers, so no halted state is
being rewritten by a still-active neighbour's attention output, which is the specific new
risk attention introduces.

**GATE L4 PASSES, and only the strong form was accepted.** The naive version of this test
leaves `ponder_weight` at its canonical value and checks the halt heads have gradient —
which they always do, because the ponder cost is an explicit function of the halt
probabilities. **That version passes on a model whose task path is completely
disconnected from halting.** So `ponder_weight = 0` and the backward pass comes from
`task_loss` alone: all six halt heads have non-`None` weight gradients with nonzero norm
(`8.42e-03 … 1.52e-02`), biases too, so the threshold is learnable. Separately the ponder
cost alone reaches every head (`3.14e-02 … 7.26e-02`), so the two are independent paths
rather than one wearing two names, and `expected_depth.requires_grad` is True — a boolean
threshold may drive dispatch, but the objective keeps a differentiable route.

**Where to look.** `metrics._average_ranks` when a depth correlation looks too confident;
`spearman_with_null` when a rho is quoted without a band; `derive_lang_ffn_mult.py` when
a parameter-budget gate fails; `test_lang_recursion.py` L4a before believing any claim
that halting is learned from the task.

## T-L7.4 (part) — the engine now actually loads the language corpus, and a run starts

`code/more/engine.py`, `code/more/config.py`, `code/more/cli.py`, `SETUP.md` (new),
`HANDOFF.md` (new). Gate L0 `TOTAL 356 356 0 0`; all six language suites unchanged
(25 / 11 / 22 / 133 / 89 / 72 passed, 0 failed).

**THE GAP THIS CLOSED, stated plainly because it was not obvious from the ledger.**
Every piece of the language path existed and was gate-verified -- dataset class, tied LM
head, causal-LM loss, detached probe, depth correlations, Gates L1-L4 -- but
`engine.train()` still built `MoREDataset(jsonl_path=...)` unconditionally. **A
`--task language` run would have loaded the arithmetic JSONL.** Nothing raised; the
config said `task: language`, the model would have been built for language, and the data
would have been arithmetic. No ledger box covered the wiring, which is how it stayed
invisible through five phases of green suites.

**What was wired.** `resolve_task` is now resolved before the dataset (`_task_early`,
named separately only because the dataset branch sits above where `_task` was already
being set), and the dataset, the val split and the model constructor all branch on it.
Both dataset classes return the same 7-tuple, so everything below the branch -- the
DataLoader, the epoch loop, the metric accumulation -- stays shared, which is plan.md
§9's one-training-system requirement.

**No random-split fallback on language, deliberately.** The arithmetic path falls back to
carving a validation set out of train when `val_path` is missing. Language must not: the
corpus ships author-provided splits and T-L2.7 verified zero exact-content overlap across
all 526,320 canonical train blocks, so a fallback would quietly destroy the one property
that makes the leakage audit meaningful.

**Two mismatch guards rather than coercion.** A `data.seq_len` that disagrees with the
packed length RAISES, naming both values and the `dataset_version`, because the blocks on
disk *are* that length and a config asking for another is asking for a corpus that was
not built. Same for `vocab_size` against the tokenizer's V -- the tied LM head is
`[V, d_model]`, so that disagreement is not cosmetic.

**`data.corpus` defaults to the CANONICAL corpus, not the dev one.** A silent fallback to
wikitext-2 would quote the loss against the wrong floor -- 5.398 instead of 4.985 -- and
the two are not comparable, because the corpora tokenize the identical val text to
different token counts (T-L2.2). `--corpus` is plumbed through the same
refuse-on-arithmetic path as `--seq_len`, and the positivity check in that loop is now
guarded on `isinstance(_val, int)`: `--corpus` is a name, and `"wikitext-2" < 1` raised
`TypeError: '<' not supported between instances of 'str' and 'int'`.

**FIRST REAL LANGUAGE RUN STARTS CORRECTLY.**
`runs/langB_MoRE_seed42__91c9bba1` -- note the `langB_` prefix, so it can never be read
as one of the fifteen `phaseB_*` arithmetic runs. It resolved
`task=language, corpus=wikitext-2, variant=language, family_cls=0.0,
step_routing=0.0`, loaded `10,527 blocks x 256 tokens` with `floor = 5.3983`, built
**5,584,908 parameters** -- exactly the MoRE reference from the T-L6.3 derivation -- and
printed a non-zero router gradient norm (0.009294) on the first batch. It was launched on
the CPU interpreter and has NOT completed, so there is no `metrics.json` and no number
from it is quoted anywhere. The directory is kept rather than deleted (CLAUDE.md §6): a
run directory with no `metrics.json` cannot be mistaken for a result.

**`SETUP.md` and `HANDOFF.md`** are written for the 4060 8 GB machine that will run the
canonical matrix. `SETUP.md` is the environment build, the dataset rebuild (`*.npy` is not
tracked; `token_family.npy` is, because it needs a specific nltk model), and a
symptom-to-cause table. The `--ignore-installed torch` step is called out because without
it pip sees the inherited CPU torch, decides the requirement is satisfied, and yields a
venv that silently trains on CPU. `HANDOFF.md` carries the three numbers that change how
results are read -- the 4.9849 floor, the partition's own 0.8884 load entropy, and the
0.0527 AMI floor -- plus the warning that a mis-shifted loss reads ~5.6 nats at epoch 0,
which is close enough to the floor to look like fast learning.

**Where to look.** `engine.py` `_task_early` for the dataset branch and the two mismatch
guards; `config.LANGUAGE_CORPUS_DEFAULT` when a run reads the wrong corpus; `cli.py`'s
override loop when a string-valued language flag is added next.

## First completed language run — the pipeline learns, and two defects surfaced

`runs/langB_MoRE_seed42__91c9bba1` (MoRE, wikitext-2, 1 epoch, batch 8, seed 42, CPU,
7 tok/s, `experiment_group = lang_smoke` so it can never enter a table). Not a
scientific result -- one epoch on the DEV corpus -- but the first end-to-end language
number, and it earned its keep by exposing two defects a green test suite had not.

**It beats the floor, thinly.** `val_loss = 5.3567` nats/token against the dev corpus's
bigram floor of 5.3983: `nats_below_bigram_floor = +0.0416`, perplexity 212.0, 7.728
bits/token. So the LM path learns something a two-column count table does not -- after one
epoch, by 0.04 nats. The canonical floor is 4.9849 and is the one that matters.

**DEPTH COLLAPSED, and the instrumentation reported it correctly rather than flattering
it.** Of 283,050 validation tokens: 1,088 at depth 1, **281,619 at depth 2**, 343 at depth
3, and **zero** at depths 4-7 out of a budget of 7. `depth/std = 0.071`,
`distinct_exit_depths = 3`. Mean depth by family spans **0.011 steps** across all six
families (1.9928 … 2.0035) -- no per-family differentiation at all. This is the arithmetic
POC's signature reproduced on language (93-97% at exactly 2 steps there, 99.5% here).

**A METHODOLOGICAL CATCH worth recording.** All three depth correlations report
`exceeds_null = 1`: logfreq +0.0948, surprisal -0.0948, model_loss +0.0431, against null
bands of about +-0.004. Those are not false positives -- the tendency is real -- but at
n = 283,050 the permutation null shrinks to +-0.004, so **statistical significance here
carries almost no information about effect size**. With 99.5% of tokens at one depth, a
rho of 0.095 is driven by the ~1,400 tokens that exited elsewhere. The null band needs its
companion, and `depth/std`, `distinct_exit_depths` and `depth/hist` are it: the pair is
interpretable, either number alone is not. Anyone quoting `exceeds_null` without the
histogram is quoting the wrong half.

Two internal consistency confirmations fell out of the same numbers.
`spearman_vs_logfreq` and `spearman_vs_unigram_surprisal` are **exact negatives to 16
digits**, which they must be (surprisal is monotone decreasing in frequency) -- and which
also means the plan lists two metrics that are not independent evidence. And the sign is
**contrary to the hypothesis**: rho vs log-frequency is POSITIVE, so more frequent tokens
received slightly *more* depth, not less. Tiny, but the wrong direction.

**No evidence of POS-aligned specialization.** `routing_agreement_with_pos = 0.0600`,
`routing_hungarian_acc = 0.2973`, **`routing_ami = 0.0181`** -- below the **0.0527**
chance level measured on synthetic routers with these marginals (T-L3.3). Load entropy
0.7600 against the partition's own 0.8884, so the router is *more* imbalanced than POS is.
Expert cosine similarity 0.028 mean / 0.044 max: barely differentiated after one epoch.
`probe/family_ce = 1.803` nats against `ln(6) = 1.792`, i.e. the detached probe is at
chance too -- which is the honest reading of a trunk that has had no family signal in its
objective (T-L5.3) for one epoch.

**DEFECT 1, FIXED HERE: arithmetic labels on a language run.** The run published
`recursion/avg_depth_by_family/E1_ADD_SUB`, `E2_MULT_DIV`, `E3_MOD_POW`, ... on LANGUAGE
values, because `expert_labels` is imported from `.families` at engine module scope. The
numbers were right and the row names came from the other study. Nothing raised, no test
covered it, and it would have put `E2_MULT_DIV` in a language paper table. The engine now
resolves the label helper from the task (`lang_families.expert_labels` on language).

**DEFECT 2, OPEN as T-L6.8(b): four keys that all read as "the exit-depth
distribution".** `recursion/avg_depth_by_family/*` = 2.0034485 and the new
`depth/mean_by_family/*` = 2.0034746 -- close but not equal, because mine excludes the
last position (no next-token target). Likewise `depth_dist/step_7_pct = 3.79%` (train
pass, and it matches `halt/forced_exit_rate` exactly) against `depth/hist/step_7 = 0`
(validation pass). Both pairs are legitimate separate measurements of different
populations, and that is precisely why the naming is dangerous: this is the
two-copies-that-diverge failure already recorded once in this file. Every depth key needs
to state its pass and its token population.

**GAP, OPEN as T-L6.9:** `specialization_vs_control` is built and unit-tested but nothing
calls it from `engine.py`, so `routing_ami` is published with **no null band beside it**.
Same class of gap as the dataset wiring -- and it bites immediately, because 0.0181
without the band reads as weak-but-present specialization rather than as at-or-below the
floor.

**Where to look.** `engine.py` `_expert_labels` for the label fix;
`runs/langB_MoRE_seed42__91c9bba1/metrics.json` for every number above;
`TASKS_LANGUAGE.md` T-L6.8 / T-L6.9 for what is still owed.

## MoR language smoke — the label fix confirmed, and a trade-off shape worth testing

`runs/langB_MoR_seed43__c1873a66` (MoR, wikitext-2, 1 epoch, batch 64, seed 43, CPU,
`experiment_group = lang_smoke2`). Resolved `num_experts = 1, ffn_mult = 24, max_depth = 7,
variant = language`, so it IS the budget-matched arm the T-L6.3 derivation specifies.

**The label fix is confirmed on a real run.** Every family key now reads
`L1_FUNCTION … L6_NUM_SUBWORD`; no `E1_ADD_SUB` anywhere. Both
`depth/mean_by_family/*` and `recursion/avg_depth_by_family/*` carry the language
manifest's names.

**Every routing metric is correctly `N/A` on the one-expert arm** --
`routing_agreement_with_pos`, entropy, both cosine similarities, all six
permutation-invariant keys. MoR makes no routing decision, so the arithmetic-era 0.1005
would have been a fabricated measurement; the CLAUDE.md §4 contract is doing its job on the
new task without a special case.

**THE PRELIMINARY SHAPE, and it is a trade-off rather than a win.** One epoch each,
DIFFERENT SEEDS, dev corpus, CPU -- this is not evidence, it is a hypothesis with a number
attached:

| | MoRE (seed 42) | MoR (seed 43) |
|---|---|---|
| `val_loss` nats/token | **5.3567** | 5.6595 |
| vs dev bigram floor 5.3983 | **+0.042 better** | **-0.261 worse** |
| perplexity | 212.0 | 287.0 |
| `depth/std` | 0.071 | **0.497** |
| depth histogram (steps 1/2) | 1,088 / 281,619 | **156,157 / 126,893** |
| per-family depth spread | 0.011 steps | **0.74 steps** |
| `spearman_vs_logfreq` | +0.095 | **+0.392** |
| `spearman_vs_model_loss` | +0.043 | **+0.130** |
| `probe/family_ce` (chance = ln 6 = 1.792) | 1.803 | 1.742 |

So MoR **allocates depth substantially** -- a 55/45 split between depths 1 and 2, seven
times MoRE's depth variance, and a 0.74-step spread across POS families -- while
**predicting worse than a bigram table**. MoRE **beats the bigram table, barely**, with
depth **collapsed** to 99.5% at one step. If that survives proper seeds and epochs it is
an Outcome B shape: neither arm dominates, and the two behaviours it is supposed to
combine appear in the two arms separately rather than together. It could equally be seed
variance at one epoch.

**A CONFOUND THAT MUST BE EXCLUDED BEFORE ANY DEPTH CLAIM (now T-L6.10).** The
frequency-depth correlation is **POSITIVE in both arms**: frequent tokens get MORE
recursion, and MoR gives L1_FUNCTION 1.85 steps against L2_NOUN 1.11. That is the opposite
direction to the hypothesis, and there is a mundane explanation that has not been ruled
out: after one epoch a tied token embedding has systematically larger row norms for
frequent types, the halt head reads a hidden state still carrying that embedding
component, so the halt logit is scale-correlated with frequency BY CONSTRUCTION. That is
an initialization artifact wearing the shape of adaptive computation. The test is to
partial out the per-token embedding norm and to track the correlation across epochs -- an
artifact should weaken as training equalises the norms. **Until then no depth-allocation
claim may be made in either direction**, and the `exceeds_null = 1` on all six
correlations does not help, because at n = 283,050 the null band is +-0.004.

**GAP (now T-L6.11): `metrics.json` carries no parameter count at all.** Gate L6 has to
publish the three arms' totals and the MoR/MoRE gap on both the total and non-embedding
counts, but the count only reaches `stdout.log` -- which is git-ignored, so on a pushed run
it is unrecoverable.

**Where to look.** `runs/langB_MoR_seed43__c1873a66/metrics.json` beside
`runs/langB_MoRE_seed42__91c9bba1/metrics.json` for the table above; `engine.py`
`_expert_labels` for the label fix; `TASKS_LANGUAGE.md` T-L6.10 / T-L6.11.

## T-L6.5 — an induced topic axis, and a pre-registered floor that failed first

`data/lang/build_block_topics.py` (new), `code/test_language_families.py`.
**105 passed, 0 failed, 1 skipped.** `plan_language.md` §4.1a's second reference axis:
`block_topic_<split>.npy`, one `int8` topic per stored block, both corpora.

**INDUCED, NOT GROUND TRUTH, and the artifact says so about itself.** WikiText ships no
topic labels, so these clusters are k-means over TF-IDF document vectors with OUR k, OUR
seed and OUR vectorizer settings. `topic_is_induced = True` and `topic_note` carry the
caveat into the manifest, and TL6.5g asserts the wording is there -- a caveat that lives
only in a plan file is one that gets dropped from a table. It is reported beside POS and
against its own null (T-L6.6), never as truth. Cluster *names* are whatever a human reads
off the top terms and are not evidence of anything.

**Fitted on TRAIN, applied to val/test.** The vectorizer, the SVD and the k-means are all
fit on train documents and then `transform`/`predict` on the held-out splits. Fitting the
topic model on the split it is used to score is the same defect as a tokenizer fitted on
validation text (T-L2.1).

**THE 2% FLOOR FAILED ON THE FIRST BUILD.** TF-IDF -> k-means directly, k=6, put **64.2%
of canonical train blocks in a single cluster** (top terms `species, war, century, army,
king, race, south, city, church, british` -- a residual bucket, not a topic) and left the
highways cluster at **1.72%**, under the pre-registered floor. A reference axis whose
majority class is two thirds of the corpus is not a reference axis: a router "agreeing"
with it would mostly be agreeing with *is this the majority class*.

Fixed by inserting **TruncatedSVD (100 components, seeded) + L2 normalization** between
TF-IDF and k-means. This is the standard LSA text-clustering pipeline and it is sklearn's
own documented recipe: k-means uses Euclidean distance, which concentrates badly in 20,000
sparse dimensions, so reducing to a dense low-rank space is the ordinary remedy rather
than a trick. **The distinction from tuning matters and is recorded in the code:** the 2%
criterion was pre-registered in `TASKS_LANGUAGE.md` before the build, the criterion was
not changed, and no routing number exists yet -- there was nothing downstream to select
on. The `topic_lsa.why` field in the manifest carries that reasoning too.

| | before | after |
|---|---|---|
| largest canonical train topic | **64.2%** | 34.8% |
| smallest canonical train topic | **1.72%** (FAIL) | **8.97%** (pass) |
| dev-corpus smallest | 2.51% | 4.26% |

**The six canonical clusters** -- five clean, one honestly mixed: military/naval
(`ship, guns, aircraft, fleet, squadron`), TV & film (`episode, series, character`), music
(`album, song, band, chart`), history/politics (`century, king, government, church,
court`), sport (`game, team, league, player, football`), and **T4, which genuinely mixes
biology, weather and roads** (`species, storm, highway, tropical, hurricane, route`) and
should be described that way rather than named. The set lands close to the categories
§4.1a was asked about.

**A SECOND SCOPE DECISION, and it is a LOOSENING, so it is flagged rather than quietly
applied.** The floor is checked on the **train** split, not on all three. Measured reason:
wikitext-2's val split holds only **61** documents and test **63**, and wikitext-103
shares those splits BYTE-FOR-BYTE (T-L2.0), so one small cluster contributing a single val
document is ~1.1% of val blocks -- a 2% floor on val could never pass for this corpus
family however good the clustering is. Non-degeneracy is a property of the fitted
partition, so train is where it belongs. `topic_min_share_any_split` is stored anyway
(4.30% canonical) because that thinness is **a real limitation of T-L6.6**: article-level
agreement on val is computed over 61 documents, which is a small sample for a 6-way
partition, and it belongs in the paper rather than in a reviewer's question.

**Document boundaries come from the STORED stream.** The encoder inserts `eot_id` between
documents and not after the last, so a token's document index is the number of EOTs before
it. TL6.5h re-derives the count from the packed array rather than trusting it: 29,444 EOTs
for 29,445 documents canonical, 629 for 630 dev. A block then takes the topic of the
document supplying most of its tokens, because a 256-token block can straddle a boundary.

Same-seed rebuild is **byte-identical on all six arrays**, both corpora -- which required
`random_state` on the SVD as well as the k-means, and an explicit `n_init=10` so the
result does not depend on the installed sklearn's default.

**Where to look.** `build_block_topics.SVD_COMPONENTS` for the failure that motivated LSA;
`fit_topics` for how top terms are recovered through the reduction
(`centroid @ svd.components_`, so they stay readable TF-IDF vocabulary); `block_doc_index`
when a block's topic looks wrong.

## T-L6.6 / T-L6.9 — two reference axes, one implementation, and a null for each

`code/more/metrics.py`, `code/more/engine.py`, `code/test_language_families.py`.
**114 passed, 0 failed, 1 skipped.** Gate L0 `TOTAL 356 356 0 0`.

**THE SEPARABILITY DEMONSTRATION, which had to come before either axis is quoted.**
Three synthetic routers, each scored on both axes:

| router | topic AMI (delta) | POS AMI (delta) | concentration |
|---|---|---|---|
| routes by topic | **1.0000** (+0.9963) | 0.0012 | 1.000 |
| routes by POS | 0.0227 (+0.0231) | **1.0000** (+0.9473) | 0.325 |
| random | -0.0029 (-0.0066) | 0.0000 (-0.0000) | 0.199 |

Each axis is beaten by its own router and not by the other's, and a random router is at
chance on both -- so neither is a metric that scores something highly by default. **The
POS router's topic delta is +0.023: small but NOT zero.** The two axes are separable
without being orthogonal, because different subjects use different noun/verb mixes. Said
plainly here because otherwise a genuine topic result could be partly POS leakage, and the
number to subtract is on the record.

**`concentration` is not decoration and must be read first.** Article-level agreement is
computed on a per-block MAJORITY expert, so a majority of 1/6 means the article was not
routed anywhere in particular. Random scores 0.199 against chance 1/6 = 0.167; the topic
router 1.000 by construction. Without it a high article-level AMI could describe a
partition of coin flips, which is exactly the reading a reviewer would challenge.

**The topic null is EXACT, unlike the per-type one.** Every stored block holds exactly
`seq_len` tokens, so permuting block labels preserves each topic's token mass precisely --
none of the greedy mass-matching plus swap-repair that T-L3.3 needed. Worth noting because
it means the article-level band is tighter and cleaner than the token-level one.

**ONE implementation, two callers.** `partition_agreement_vs_control(oracle, predicted,
control_oracles, num_experts)` was factored out of `specialization_vs_control`; the
per-token POS path and the per-article topic path differ only in what a unit is and how
the oracle label reaches it. TL6.6i asserts there is exactly one definition. Two copies of
these statistics would be the two-copies-that-diverge failure this file already records
once -- and that failure landed the tolerance guard in the copy that never ran.

**T-L6.9's substance is wired in the same commit.** The engine loads
`token_family_shuffled.npy` and `block_topic_val.npy` once per run and writes both
comparisons under `val/routing_control_pos/*` and `val/routing_control_topic/*`. A missing
artifact prints a NAMED warning to stderr and skips that axis rather than writing a zero --
absence is "not measured", a 0.0 would be "POS is no better than noise". The signal is
`first_route`, i.e. `updated_rules.md` §8's authoritative first-step decision, so it is the
same one the confusion matrix is built from rather than a second opinion. The routing
accumulator collects WHOLE blocks in split order, which is why it needs its own id vector
rather than reusing the depth one (that drops each block's last position, which has no
next-token target).

**T-L6.6 IS NOT TICKED.** Its Verify has two clauses. The separability clause is executed
and passing; the "both agreement numbers and both null bands appear for every language
run" clause needs a completed language run, and the one exercising it is still training on
CPU. The box stays `[ ]` until that has actually run.

**Where to look.** `metrics.partition_agreement_vs_control` for the shared statistics;
`block_majority_expert` / `block_topic_concentration` for the article-level reduction and
the number that makes it interpretable; `engine.py` `_lang_route` / `_lang_ids_full` for
the accumulators and why there are two.

## T-L6.6 closed by a real run — and the topic axis answers the question: no

`runs/langB_MoRE_seed44__2ce26c0d` (MoRE, wikitext-2, 1 epoch, batch 96, seed 44, CPU)
wrote **21 POS-axis keys and 20 topic-axis keys**, no `routing_control_error`, so the
every-run clause of T-L6.6 is executed and the box is ticked. Suite 114 passed, 0 failed,
1 skipped; Gate L0 `TOTAL 356 356 0 0`.

**POS axis: the PARTITION carries POS information, the LABELLING does not.** AMI 0.1041
against a control mean of 0.0069 -- delta **+0.097**, z 28.5. Hungarian 0.3617 vs 0.2937,
purity 0.3687 vs 0.2998, both about +0.068. But **raw agreement 0.1424 against a control
of 0.2403, delta -0.098** -- below its own null. That is exactly the raw-versus-Hungarian
gap `updated_rules.md` §8.3 exists for: quoting raw agreement alone would have reported the
OPPOSITE conclusion from the permutation-invariant metrics on the same run. This is the
first language run where the specialization metrics disagree with each other, and the
ordering §4.4 imposed (permutation-invariant first, agreement last) is what makes it
readable rather than confusing.

**Topic axis: no article-level preference at all, reported cleanly.** AMI -9.7e-16, delta
exactly 0.0, `control_std` exactly 0.0; purity and Hungarian both **0.6351351351**, which
is the largest val topic's block share (0.6351351351351351) **to the last digit**. The
block-majority expert is CONSTANT across all 1,110 val blocks.

And this is *not* token-level collapse: token load entropy is 0.832 and
`routing_collapsed_experts = none`. So the router varies expert within a block while having
no preference between articles. After one epoch, "did the right article go to the right
expert" has a measured answer, and it is **no** -- which is the honest negative the second
axis was added to be able to give.

**A DEFECT THE RUN EXPOSED, fixed here.** `delta_z` was guarded on `control_std > 0`, which
is not sufficient. With a constant prediction every control draw scores identically, so the
spread is floating-point residue -- **measured at 2.1e-31** -- and a delta of -2e-31 became
**z = -0.95**. A z near one from a 1e-31 denominator is not a small effect, it is no effect,
and it reads as the former. Now guarded on a `1e-12` floor with `None` below it, and the
comment names the run that made it necessary.

**A CORRECTION worth its own paragraph, because it was written into `HANDOFF.md` as
guidance.** The 0.0527 figure from T-L3.3 is the AMI a *POS-perfect* router scores against
the control -- it is NOT a universal AMI floor. The control mean depends on BOTH partitions,
and for this router it was 0.0069. So "compare a run's AMI against 0.0527 by hand" was wrong
advice; the per-run band in `metrics.json` is the right baseline and is now always present.
`HANDOFF.md` has been corrected, since it is the document another agent will follow.

**Where to look.** `metrics._CONTROL_STD_FLOOR` for the z guard;
`runs/langB_MoRE_seed44__2ce26c0d/metrics.json` for both axes side by side;
`TASKS_LANGUAGE.md` T-L6.6 Evidence for the numbers.

## T-L6.8 / T-L6.9 — depth-key provenance shipped as data, and a Verify line corrected

`code/more/metrics.py`, `code/more/engine.py`, `code/test_lang_heads.py`.
**30 passed, 0 failed, 1 skipped.**

**T-L6.8. Renaming was not available, so the provenance ships AS DATA.** Two of the four
ambiguous keys -- `recursion/avg_depth_by_family` and `depth_dist/step_N_pct` -- belong to
the arithmetic contract Gate L0 checks, so they cannot be renamed. Instead
`metrics.DEPTH_KEY_PROVENANCE` maps seven key prefixes to `{pass, population, note}` and
the engine writes it into every run's `metrics.json`, for BOTH tasks, because the ambiguity
was never language-specific.

The two disagreements are now named rather than left to be rediscovered:
**2.0034746 vs 2.0034485 is the last-position exclusion and nothing else** (the last
position of a block has no next-token target, so the new key drops it and the old one
keeps it), and **`depth_dist/step_7_pct = 3.79%` against `depth/hist/step_7 = 0` is
train-versus-validation** -- which is also why the former matches `halt/forced_exit_rate`
exactly.

`uncovered_depth_keys` casts a deliberately wide net -- anything whose name could be read
as an exit-depth quantity -- and the engine writes `depth_key_provenance_uncovered` plus a
named stderr warning if a key escapes the registry. **Zero uncovered keys across all three
completed language runs.** Matching is longest-prefix so `depth/mean_by_family/` is
described by its own entry rather than by a shorter `depth/` one.

TL6.8f is the single skip, and it is skipped rather than passed or failed on purpose: all
three completed runs predate the writer, so none carries the block. It reports that with
the run count, passes as soon as one run has it, and **fails loudly if any run flags an
uncovered key** -- which would mean the writer ran and found a gap.

**T-L6.9. Wired, and its Verify line was WRONG IN TWO WAYS, both corrected rather than
worked around.**

1. The key prefix is `val/routing_control_pos/*`, not `val/routing_control/*` -- there are
   two axes now, so the POS one is named for what it compares against.
2. **"matches the manual 0.0527 comparison" is the wrong test and must not be performed.**
   0.0527 is the AMI a *POS-perfect* router scores against the control. The control mean
   depends on BOTH partitions, and for the seed-44 router it was **0.0069**. Against the
   fixed constant, an AMI of 0.104 would have looked unremarkable; against its own null it
   is delta +0.097 at z 28.5. `HANDOFF.md` carried the bad advice and was corrected,
   because another agent follows that file.

A missing reference artifact prints a named stderr warning and skips that axis rather than
writing a zero: absence means "not measured", 0.0 would mean "POS is no better than noise".

**Where to look.** `metrics.DEPTH_KEY_PROVENANCE` before adding any depth metric;
`uncovered_depth_keys` for what counts as depth-ish; `TASKS_LANGUAGE.md` T-L6.8 for the
prefix/pass/population table.

<!-- APPEND-MARKER-CL -->







































