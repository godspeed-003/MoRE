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
