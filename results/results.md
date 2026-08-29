# MoRE Canonical Research Results

**Status.** T11.2 deliverable. This document is the single narrative reading of
the canonical Phase B matrix and the Phase 10 ablations. It is **hand-authored
prose over machine-generated numbers**: every figure below is read out of one of
the artifacts listed in §16, and none is transcribed from an earlier prose
document. Where a number disagrees with anything in `README.md`,
`results_exp.md`, or any file under `archive/`, this document and the artifacts
it cites are correct and the other is stale.

**Register.** Each section separates three kinds of statement explicitly:

| label | meaning |
|---|---|
| **MEASUREMENT** | A number produced by a run and read from an artifact. Falsifiable by re-running. |
| **INTERPRETATION** | What we take the measurement to mean. Depends on assumptions stated with it. |
| **HYPOTHESIS** | A candidate explanation that this evidence does *not* settle. Named so it is not mistaken for a finding. |

No sentinel is reported as a measurement. `N/A` means the quantity does not
exist for that architecture, or the code refused to state it — never `0.0`
standing in for "not implemented" (`CLAUDE.md` §4).

**Reading rule of record.** Exact two-sided randomization test on the difference
of means (`code/seed_stats.py`). Sample standard deviation (n−1). Every p is
reported with `min_p = 1/C(n_a+n_b, n_a)`, the smallest value the arm sizes can
produce: **0.0040 at 5-vs-5, 0.0179 at 3-vs-5.** A p equal to its floor is the
test's resolution limit, not a significance claim. The superseded
`|gap| ≥ 2 × max(seed std)` rule is withdrawn (T11.0b) and appears nowhere here.

---

## 1. Executive Summary

**MEASUREMENT.** At matched parameter count (~3.2 M), on the synthetic `more6-v1`
arithmetic-reasoning task, 5 seeds each, last-epoch validation task loss:

| architecture | `val/task_loss` | R² vs floor | params |
|---|---|---|---|
| MoE | 0.063260 ± 0.000153 | 0.2182 | 3,201,555 |
| **MoR** | **0.062706 ± 0.000112** | **0.2250** | 3,197,710 |
| MoRE | 0.063186 ± 0.000249 | 0.2191 | 3,201,555 |

MoR is lowest. MoR beats MoE (gap 0.000554, d = 4.13, p = 0.0079) and beats MoRE
(gap 0.000481, d = 2.49, p = 0.0159). MoRE and MoE are indistinguishable
(gap −0.000074, d = −0.36, p = 0.6587).

On held-out data, from each run's best-val checkpoint, MoRE spends
significantly **more** recursion than MoR (2.3531 ± 0.1087 vs 2.1283 ± 0.0188
steps, d = 2.88, p = 0.0079) and gets **no better** agreement with the
operation-complexity curriculum for it (absolute depth allocation error
0.9940 ± 0.1054 vs 0.9423 ± 0.0266, p = 0.3571; relative error significantly
*worse*, 0.6934 vs 0.5939, p = 0.0159).

Both recursive architectures place 93–97% of tokens at exactly 2 steps, with
early-exit rates of 0.9986–0.9997. The decisive test is against the **best
constant-depth policy** — a model that ignores its input and emits one number for
every token (`results/depth_null_model.json`, §9). On held-out data that null sits
at absolute depth allocation error **0.8703 at c = 2.00**, and neither trained halt
head beats it: MoR 0.9423 (+8.3% worse), MoRE 0.9940 (+14.2% worse), with **5 of 5
seeds losing to the constant in both architectures** (exact sign test p = 0.0625,
its floor at n = 5; 10 of 10 pooled, p = 0.00195). The curriculum's token-weighted
mean target depth is 2.1082, so a constant 2 is already close to the right
*average* — what the halt head fails to supply is the per-operation *variation*
around it.

Two ablations sharpen this into architectural conclusions (§11, six arms,
family-wise α = 0.05/6 = 0.0083). **Forcing every token to the maximum depth of 7
gives *better* validation loss than learned halting** — 0.062886 ± 0.000143 vs
0.063186 ± 0.000249, d = −1.48, p = 0.0476, nominally significant and failing the
family-wise threshold — at statistically indistinguishable throughput (2198 ± 274 vs
2225 ± 274 tok/s) despite 3.4× the recursion steps. So the halting mechanism bought
neither quality nor wall clock. And **supervising the router with the oracle expert
index drives every routing metric to exactly 1.0000 ± 0.0000 while making the task
loss significantly worse** (0.063751 ± 0.000121, d = +2.89, p = 0.0079,
Bonferroni-surviving): the oracle partition is learnable to perfection, and learning
it hurts. Routing quality is therefore not a proxy for architectural quality anywhere
in this work.

Expert routing is above chance but far from clean: Hungarian-matched routing
accuracy 0.5258 ± 0.0612 for MoRE and 0.5004 ± 0.0560 for MoE, against a 6-way
chance level of 0.1667. Raw (unmatched) routing accuracy is 0.1629 ± 0.1864 for
MoRE — a standard deviation larger than the mean — because the canonical router
is unsupervised (`loss_weights.step_routing = 0.0`) and expert indices carry no
semantic identity.

**INTERPRETATION — this is `CLAUDE.md` §8 Outcome C, on both halves of the
architecture.** At this scale and on this task, expert routing buys nothing over
a single equally-sized FFN (MoRE ≈ MoE), and adding recursion on top of routing
costs quality relative to recursion alone (MoR < MoRE). The halt head learned an
approximately unconditional constant of ≈2 steps rather than an input-dependent
policy — quantified, not asserted: it is beaten by the best constant on every
seed. So "adaptive computation" is present mechanically — the gradient reaches
the halt parameters, the exits are real, the ponder cost is differentiable — but
not behaviourally.

**INTERPRETATION — what MoR winning does and does not license.** MoR being
lowest-loss is a result *within* an exploratory paper about MoRE, not a thesis
about MoR. Recursive depth-sharing is established prior work; the novelty under
test is the *composition* of learned expert routing with adaptive recursive
reuse, and the composition did not pay off here. Three confounds bound this
conclusion, and each is a limitation on the conclusion rather than an excuse for
MoRE (§15, §19):

1. **Scale.** Every run is 3.2 M parameters, `d_model = 256`, one recursive
   block, on one RTX 4060 Laptop. Six experts of FFN width 1024 each see ~1/6 of
   the tokens; MoR's single width-6144 FFN sees all of them. Partitioning may
   simply cost more than specialization buys at this width. Nothing wider was
   tested (§13).
2. **The dataset.** `more6-v1` is synthetic and custom (`data/script.py`, Python
   `random`, 16 hand-chosen operations, hand-authored depth curriculum). Both
   MoRE mechanisms are being asked to recover structure that was put there by
   construction. All three architectures explain only 21.8–22.5% of target
   variance, and the entire between-architecture spread is ~3% of what they
   explain — a regime where the task, not the architecture, is the binding
   constraint.
3. **The evaluation.** One scalar regression loss, one fixed split, 5 seeds, 50
   epochs, one optimizer setting frozen as VALIDATED-STABLE rather than swept per
   architecture. A metric that rewards only final-answer accuracy is structurally
   incapable of crediting the properties MoRE is *for*.

**HYPOTHESIS (not settled here).** That MoRE's deficit is a capacity-partitioning
artifact which would invert at larger `d_model` or wider experts. Testing it
requires the width sweep that was never run.

---

## 2. Research Question and Architecture Definition

**The question.** Does routing a token to a *specialized* expert and then reusing
that expert's weights for an *adaptive* number of recursion steps do something
that neither mechanism achieves alone, at matched parameter count?

The three architectures answer three different questions and must never be
conflated (`CLAUDE.md` §1):

| | mechanism | answers | `num_experts` | `max_depth` |
|---|---|---|---|---|
| **MoE** | multiple independent expert FFNs, learned Top-1 sparse token routing | *which computation module?* | 6 | 1 |
| **MoR** | one shared block reused across recursion steps with adaptive halting | *how much computation?* | 1 | 7 |
| **MoRE** | route to an expert, then recursively reuse that expert's weights with adaptive halting | *both* | 6 | 7 |

All three are one training system with an explicit `architecture` mode
(`code/more/cli.py`, `code/more/engine.py`). There is no `moe/` or `mor/`
directory and no `run_moe.py` / `run_mor.py`: the three arms share one tokenizer,
one data pipeline, one optimizer setup and one evaluation path by construction, so
a difference between them cannot be a difference in the harness.

**Architectural commitments actually implemented and verified** (Gates 1–5, 350
assertions, §16):

- **Routing** is `Linear router → softmax → argmax → dispatch to that expert only
  → multiply by the selected gate probability`. `dispatch/evals_per_token` =
  1.0000 ± 0.0000 for all three arms, which is the sparsity claim measured rather
  than asserted. Dense evaluate-all-and-blend exists only as the labelled
  `dense_routing` ablation (§11), which reports `evals_per_token` = 6.
- **Expert independence**: six independent FFNs, no cross-expert communication.
- **Weight sharing**: the *same* block parameters at every recursion depth. Depth
  is computation through time, not a stack of depth-specific networks
  (`num_blocks = 1`).
- **Halting** is differentiable ACT: a learned per-step halt probability, early
  exit for active tokens, halted states frozen and not recomputed, forced exit at
  `max_depth = 7`, a differentiable ponder cost, and a real gradient reaching the
  halt parameters (Gate 2, 31 assertions).
- **Balance loss** is normalized so its magnitude does not grow with recursion
  depth or block count (Gate 3, 27 assertions), and the entropy term and the
  Switch-style auxiliary term are reported separately (§7).
- **Six experts**, no E7 catch-all, all tensor dims derived from `num_experts`,
  all labels from one family manifest, unmapped operations raise (Gate 4, 130
  assertions).

**Input integrity.** `expert_id` / `family_id` is not an input feature, no oracle
routing class is encoded into an input slot, and the regression target is not
embedded in the input. Operation identity is present as an operation embedding —
legitimate task information — not as the oracle expert index. Input features are
**bit-identical** across MoE, MoR and MoRE for the same record; this is a tested
invariant, not an aspiration (Gate 1, §4).

---

## 3. Canonical Configuration

**Source of authority.** `code/canonical_spec.json` is the single frozen
definition of "canonical". A run may declare `experiment_group =
canonical_phase_b` only if it matches that spec exactly, uses the full dataset,
and names a seed from the frozen set `{42, 43, 44, 45, 46}`. All 18
`enforced_fields` are non-null, so the guard is live rather than permissive:

| field | canonical value | field | canonical value |
|---|---|---|---|
| `epochs` | 50 | `grad_clip` | 1.0 |
| `batch_size` | 768 | `max_steps` | 7 |
| `subset_fraction` | 1.0 | `step_feat_dim` | 12 |
| `num_blocks` | 1 | `routing_mode` | `top1_sparse` |
| `d_model` | 256 | `router_noise` | `none` |
| `lr` | 0.001 | `halting_mode` | `pure_act` |
| `weight_decay` | 0.0001 | `task_weight` | 1.0 |
| `dropout` | 0.1 | `family_cls_weight` | 0.5 |
| `dataset_version` | `more6-v1-seed42-n70000-3c1087b6aad9` | `train_split_version` | `fixed-file-splits-v1` |

`halting_mode` is deliberately absent from `code/config.json` and derived by
`config.resolve_halting_mode()` (`code/more/config.py:410`), so a supervised
halting run cannot be mistaken for an unsupervised one by reading the config file
alone.

**Fields that legitimately differ by architecture** — every other field is
bit-identical across the three arms:

| field | MoE | MoR | MoRE | why it differs |
|---|---|---|---|---|
| `num_experts` | 6 | 1 | 6 | MoR is one shared block by definition |
| `max_depth` | 1 | 7 | 7 | MoE has no depth to allocate |
| `model.ffn_mult` | 4 | **24** | 4 | parameter-budget match, T8.3 (below) |
| `loss_weights.halting` | 0.0 | 0.001 | 0.001 | no ponder cost exists at depth 1 |
| `loss_weights.routing_balance` | 0.001 | 0.0 | 0.001 | one expert has nothing to balance |

**The `ffn_mult = 24` decision (T8.3).** MoR has one FFN where MoRE has six, so
matching `ffn_mult` would have compared 3.2 M parameters against 0.57 M and
called the difference architectural. `24 = 4 × 6` gives MoR's single FFN the
total hidden width of MoRE's six experts. Measured totals:

| `mor.ffn_mult` | total params | shortfall vs MoRE's 3,201,555 |
|---|---|---|
| 4 | 571,150 | 82.2% |
| 23 | 3,066,382 | 4.22% |
| **24** | **3,197,710** | **0.12%** |
| 25 | 3,329,038 | −3.98% (over) |

The residual 3,845-parameter gap is fully accounted for in §5, and none of it is
FFN weight matrix: MoR and MoRE hold **exactly** 3,145,728 FFN weight-matrix
parameters each. `ffn_mult = 4` survives as the explicitly labelled
under-budgeted baseline `ffn_mult_4` (§12), never as canonical.

**The router is UNSUPERVISED.** `loss_weights.step_routing = 0.0` in
`code/config.json`, and pinned at 0.0 for all three architectures in
`canonical_spec.json:architecture_variants` (enforced by Gate 0 since T8.2). The
partition in §8 is therefore learned from the task objective, not taught from the
oracle expert index. `train/step_routing_loss` is still *computed* and logged
(0.5868 ± 0.0451 for MoRE) so the term's magnitude is visible, but it is
multiplied by zero and contributes no gradient.

**Disclosure — `family_cls` is external annotation, not self-supervision.**
`loss_weights.family_cls = 0.5` weights a 6-way whole-program family
cross-entropy whose labels come from a `"family"` string that `data/script.py`
stamps into every record and `code/more/data.py` reads off disk. It is the
second-largest term in the objective. It was a bare `0.5` literal inside
`engine.py` until T8.3, i.e. it appeared in no config, no provenance block and no
results table. §11 reports the ablation that measures how much of the routing
partition depends on it.

---

## 4. Dataset and Leakage Audit

**MEASUREMENT — provenance.** `more6-v1-seed42-n70000-3c1087b6aad9`, generated by
`data/script.py` at commit `e5b0f6df0ecc53e70e52baf424277a570fd29718` with
generator seed 42. 70,000 records, split by fixed files (`train_split_version =
fixed-file-splits-v1`) into 59,500 / 5,250 / 5,250, each split carrying its own
sha256 in `data/dataset_meta.json`. The split is a file boundary, not a
per-seed `random_split`, so seed-to-seed variance in §14 is initialization,
dropout and batch order only — never a different train set.

**MEASUREMENT — the splits do not overlap.** Program-key overlap is exactly 0 for
all three pairs (`train|val`, `train|test`, `val|test`), and unique-program counts
equal record counts in every split (59,500 / 5,250 / 5,250), so there are no
duplicate programs *within* a split either.

**MEASUREMENT — Gate 1 leakage audit, 8 of 8 checks passed**
(`code/audit_leakage.py` → `runs/gate1_audit.json`):

| check | result |
|---|---|
| mutating the step *result* changes any input feature | 0 features change |
| permuting the oracle expert index changes any input feature | 0 features change |
| input slot-0 range at `num_experts` = 1 / 5 / 6 | identical `[−1.0, 1.0]` |
| split overlap (all three pairs) | 0 |

The second and third rows are the two claims that matter most. Zero feature
changes under expert permutation is the machine check that the oracle expert index
is not encoded anywhere in the input; identical ranges across `num_experts` is the
machine check behind "input features are bit-identical across MoE, MoR and MoRE"
(`CLAUDE.md` §3). Per-step features hold only that step's numeric arguments,
clamped to ±`max_val` = 1e6 and divided by it; the step *result* is excluded
because for the final step it **is** the regression target. Operation identity
enters separately as `step_ops` through an operation embedding.

**MEASUREMENT — the trivial-baseline floor.** On the normalized target:

| baseline | MSE |
|---|---|
| predict 0 | 0.081853 |
| **predict the train mean (0.034806)** | **0.080914** |
| best single-input-feature copy (slot 8) | 0.081853 |
| val target variance | 0.080899 |

Every R² in this document is `1 − val/task_loss / 0.080914` against the middle
row. All three architectures beat all three baselines; none beats them by much
(§6).

**MEASUREMENT — one third of targets coincide with an argument.** The target
equals one of the step's own arguments for 33.54% of train and 33.45% of val
records — an unavoidable consequence of operations like `MAX`, `MIN` and `MOD`
whose answer often *is* an input. **INTERPRETATION:** this is not an exploitable
shortcut. Copying the best single feature scores 0.081853, i.e. *worse* than
predicting the constant train mean, so a copy strategy is strictly dominated and
the coincidence carries no free signal. It is disclosed because a reader
computing "how often could the model cheat?" will find the number and should find
our reading of it too.

**MEASUREMENT — the generator's E7 slot is redefined, not dropped.**
`data/script.py` emits seven family slots with mass `E1 0.06, E2 0.10, E3 0.14,
E4 0.06, E5 0.06, E6 0.18, E7 0.40`. Canonical MoRE has exactly six experts and
routes **per step**, so E7 — the multi-operation CHAIN family — is not a seventh
expert: its steps are individually routed to E1–E6. Those records are emitted with
`family = 'MIXED'` and `whole_program_family_index = −1`, i.e. no whole-program
family label. Train counts: E1 3,614 / E2 5,954 / E3 8,335 / E4 3,548 / E5 3,574 /
E6 10,791 / **MIXED 23,684**.

**INTERPRETATION.** Dropping the 40% instead would have deleted every program of
depth 4–7 and made adaptive-depth allocation unmeasurable — the phenomenon under
study. The cost of keeping it is that the family classifier in §7 is trained on
the 60% that carry a whole-program label, and `−1` is a real sentinel that must be
masked, not a class. It is masked (`code/more/data.py`), and MIXED never enters
the 6-way family accuracy.

**MEASUREMENT — the depth curriculum is not a function of the expert index.**
`families.OP_TARGET_DEPTH` assigns target depth 1 to the eight ADD/SUB/LOGIC/SHIFT
operations, 2 to MULT/DIV/MAX/MIN, 3 to MOD/POW, 4 to MEDIAN/SORT. E4 LOGIC and E5
SHIFT therefore share depth 1, and E6 SORT/STAT spans depth 2 (MAX/MIN) and 4
(SORT/MEDIAN). Held-out program-length distribution is 1: 627, 2: 974, 3: 759,
4: 1,042, 5: 802, 6: 509, 7: 537.

**INTERPRETATION.** This is a deliberate design property, and §9 depends on it: if
target depth were a pure function of the expert index, a model could score
perfectly on depth by routing alone and the depth measurement would carry no
information independent of §8.

**Scope limit, stated once and applying to every number in this document.** This
is a synthetic dataset of 16 hand-chosen integer operations with a hand-authored
depth curriculum, generated by Python `random`. Raw targets span
`[−6.96e13, 3.29e15]` and are clamped to ±1e6 before normalization, so most of the
normalized target mass sits near zero — which is why the variance to be explained
is only 0.0809 and why "R² = 0.22" is a statement about a heavily compressed
target, not about a natural distribution. Both MoRE mechanisms are being asked to
recover structure that was put there by construction. Nothing here transfers to
language modelling without being re-measured.

---

## 5. Parameter Counts

**MEASUREMENT — totals, by module, from instantiated models.** Every count below
is `sum(p.numel())` over `named_parameters()` of a model built with the canonical
per-architecture fields of §3:

| module | MoE | MoR | MoRE |
|---|---|---|---|
| `step_proj` | 3,840 | 3,840 | 3,840 |
| `op_embed` | 4,096 | 4,096 | 4,096 |
| `blocks` | 3,156,998 | 3,153,153 | 3,156,998 |
| `regression_head` | 33,537 | 33,537 | 33,537 |
| `cls_head` (family) | 1,542 | 1,542 | 1,542 |
| `step_cls_head` | 1,542 | 1,542 | 1,542 |
| **TOTAL** | **3,201,555** | **3,197,710** | **3,201,555** |

**MEASUREMENT — MoE and MoRE have identical parameter sets.** Not merely equal
totals: every module is the same shape, because both are `num_experts = 6`,
`ffn_mult = 4`, `num_blocks = 1`. The only configuration difference is
`max_depth`, 1 versus 7, and weight sharing means depth adds no parameters.

**INTERPRETATION — this makes MoRE − MoE the cleanest contrast in the matrix.**
MoE *is* MoRE with the recursion switched off. The §6 gap between them therefore
isolates adaptive recursion alone, with no parameter, shape, initialization-shape
or optimizer difference to argue about. Everything MoRE has that MoE lacks is
*computation*, not capacity. Its measured value is −0.000074 ± 0.000131,
p = 0.6587.

**MEASUREMENT — the MoRE − MoR gap of 3,845 parameters, itemized.** All of it sits
inside the recursive block:

| parameter | MoR | MoRE | delta | what it is |
|---|---|---|---|---|
| `moe_block.experts` | 3,152,128 | 3,153,408 | +1,280 | bias vectors only (below) |
| `moe_block.router` | 256 | 1,536 | +1,280 | `Linear(256 → E)`, no bias |
| `expert_halt_heads` | 257 | 1,542 | +1,285 | `Linear(256 → 1)` per expert |
| `layer_norm` | 512 | 512 | 0 | — |
| **total** | 3,153,153 | 3,156,998 | **+3,845** | |

**MEASUREMENT — the FFN weight matrices are exactly equal.** Strip the bias
vectors and both architectures hold **3,145,728** FFN weight parameters:
MoR `2 × 256 × 6144`, MoRE `6 × 2 × 256 × 1024`. The +1,280 in the `experts` row
is bias arithmetic — MoRE carries six `(1024 + 256)` bias pairs (7,680) where MoR
carries one `(6144 + 256)` pair (6,400).

**INTERPRETATION.** The parameter match is not approximate in the part that
matters. The three architectures differ by 0.12% overall and by **0.00%** in FFN
weight capacity; the entire residual is one extra router row per expert, one extra
halt head per expert, and bias vectors. No result in §6 can be attributed to a
capacity difference. What differs is how that identical capacity is *partitioned*:
six width-1024 experts each seeing ~1/6 of the tokens, versus one width-6144 FFN
seeing all of them. That partitioning is the architecture, and it is the thing
being tested (§19, and the HYPOTHESIS in §1).

**MEASUREMENT — `expert_evaluations` per epoch** (train split, from the run
metrics): MoE 224,450; MoR 451,445 ± 3,951; MoRE 463,991 ± 12,427. `evals_per_token`
is 1.0000 ± 0.0000 for all three, so these are step-token counts times realized
depth, not dense-blend counts. 224,450 is exactly the number of valid step-tokens
in the train split, which is the arithmetic check that MoE evaluates each
step-token exactly once.

---

## 6. Main MoE / MoR / MoRE Results

**The primary metric is last-epoch `val/task_loss`** — the regression MSE alone, on
held-out data, at the end of the fixed 50-epoch budget. Not `total_loss`, which
mixes in the balance and ponder terms and can go negative (`CLAUDE.md` §4). Not
`best_val_loss`, which is reported below as a secondary because "best" is selected
on the same split it is read from.

**MEASUREMENT — per seed, n = 5 per arm:**

| seed | MoE | MoR | MoRE |
|---|---|---|---|
| 42 | 0.063090 | 0.062623 | 0.063307 |
| 43 | 0.063195 | 0.062872 | 0.063341 |
| 44 | 0.063492 | 0.062702 | 0.063222 |
| 45 | 0.063203 | 0.062746 | 0.063313 |
| 46 | 0.063320 | 0.062586 | 0.062748 |
| **mean ± std** | 0.063260 ± 0.000153 | **0.062706 ± 0.000112** | 0.063186 ± 0.000249 |
| seed range | 0.000402 | 0.000286 | **0.000593** |
| R² vs floor | 0.2182 | **0.2250** | 0.2191 |
| `best_val_loss` | 0.062819 ± 0.000194 | 0.062527 ± 0.000052 | 0.062892 ± 0.000216 |

**MEASUREMENT — pairwise, exact two-sided randomization test, 252 permutations,
`min_p = 0.0040`:**

| contrast | gap | se of difference | Cohen's d | p |
|---|---|---|---|---|
| MoRE − MoE | −0.000074 | 0.000131 | −0.36 | 0.6587 |
| MoRE − MoR | +0.000481 | 0.000122 | +2.49 | **0.0159** |
| MoE − MoR | +0.000554 | 0.000085 | +4.13 | **0.0079** |

**INTERPRETATION — three statements, in decreasing strength.**

1. **MoR is lowest, and it beats both others.** Positive gaps mean worse (higher
   loss). MoR beats MoE at p = 0.0079 with d = 4.13 — a gap 6.5× the standard
   error of the difference — and beats MoRE at p = 0.0159 with d = 2.49. Neither p
   is at the 0.0040 floor, so these are resolved results, not resolution limits.
2. **MoRE and MoE are indistinguishable.** p = 0.6587, d = −0.36, and the gap is
   smaller than its own standard error. Since §5 establishes that MoE is exactly
   MoRE with `max_depth = 1` and the identical parameter tensors, this is the
   direct measurement of what adaptive recursion contributed on top of routing:
   nothing detectable. Read the sign with care — MoRE's mean is nominally *lower*,
   but at d = −0.36 the sign is not evidence of anything.
3. **MoRE has the widest seed spread**, 0.000593 versus MoR's 0.000286, driven by
   seed 46 at 0.062748 — an outlier low enough to be MoR-competitive on its own.
   One favourable seed is never evidence (`CLAUDE.md` §5), and it is named here
   precisely so nobody quotes it alone.

**INTERPRETATION — the effect sizes are large and the effects are small.** These
two facts are not in tension and both belong in any honest reading. d = 4.13 is
large because the seed-to-seed noise is tiny (std ~1.5e-4 on a loss of 6.3e-2,
i.e. 0.24%), not because the architectures are far apart. In absolute terms all
three sit inside a window of 0.00055, while the distance from the predict-the-mean
floor down to the best model is 0.0182. **The entire between-architecture spread
is 3.0% of the variance any of them explains.** Whatever separates these
architectures is a third-order effect next to what none of them has learned about
this task.

**INTERPRETATION — what this does *not* license (the framing ruling, T11.0c/d).**
The subject of this work is MoRE. MoR being lowest-loss is a *result inside* an
exploratory study of MoRE, not a thesis about MoR, and this document is not
restructured around MoR because it won. Recursive depth-sharing is established
prior work; what is novel and under test here is the *composition* of learned
expert routing with adaptive recursive reuse. The composition did not pay off at
this scale, on this task, under this evaluation. Three bounds on that conclusion
are stated in §1 and carried through §13, §15 and §19 — they are limitations on
the conclusion, not excuses for MoRE.

**HYPOTHESIS (not settled here).** That MoR's advantage is a
capacity-partitioning effect: one width-6144 FFN sees every token, while each of
six width-1024 experts sees roughly a sixth of them, so per-expert effective
sample size is ~1/6 and gradient signal per expert correspondingly thinner.
§5 shows the *total* FFN weight capacity is identical, so partitioning is the only
remaining structural difference — but "the only remaining difference" is not a
mechanism, and nothing here measures the width dependence. The sweep that would
test it was not run (§13).

---

## 7. Training Dynamics

**MEASUREMENT — the ordering on train does not match the ordering on val.**
Last-epoch, n = 5:

| | `train/task_loss` | `val/task_loss` | generalization gap |
|---|---|---|---|
| MoE | 0.060453 ± 0.000214 | 0.063260 ± 0.000153 | 0.002807 |
| MoR | 0.061501 ± 0.000173 | **0.062706 ± 0.000112** | **0.001205** |
| MoRE | **0.060237 ± 0.000360** | 0.063186 ± 0.000249 | 0.002949 |

**INTERPRETATION — MoRE fits the training set best and generalizes worst.** It has
the lowest training loss of the three and the second-worst validation loss; MoR has
the *highest* training loss and the best validation loss, with a generalization gap
2.4× smaller than either six-expert model. This is the single most informative
pattern in the training-side metrics, and it reframes the §6 result: MoRE's deficit
is not a failure to fit. The capacity is being used — it is being used to fit
training-set detail that does not transfer.

**INTERPRETATION — this is a coherent reading of the partition, and it is still not
a causal claim.** Six experts each specialize on a ~1/6 slice, and a slice of
59,500 records affords more room to memorize than the whole of it does per unit of
shared weight. The gap ordering (MoRE 0.0029 ≈ MoE 0.0028 ≫ MoR 0.0012) tracks
*expert count*, not depth: MoE has no recursion and the same gap as MoRE. What
would settle it is a width or dataset-size sweep, and neither was run (§13).

**MEASUREMENT — the auxiliary terms, and their scale relative to the task.**

| term | MoE | MoR | MoRE | note |
|---|---|---|---|---|
| `train/entropy_term` | 1.7858 ± 0.0019 | N/A | 1.7882 ± 0.0025 | raw, nats |
| `train/entropy_term_normalized` | 0.9966 ± 0.0010 | N/A | 0.9980 ± 0.0014 | `H / log 6` |
| `train/switch_aux_term` | 0.9853 ± 0.0110 | N/A | 1.0192 ± 0.0050 | 1.0 = perfectly balanced |
| `train/routing_balance_loss` | −0.8005 ± 0.0093 | N/A | −0.7690 ± 0.0070 | the assembled term |
| `train/balance_to_task_ratio` | 0.0132 ± 0.0002 | N/A | 0.0128 ± 0.0002 | weighted, vs task |
| `train/ponder_cost` | N/A | 0.2691 ± 0.0028 | 0.2940 ± 0.0057 | = `halting_loss` |
| `train/family_cls_loss` | 0.0001 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | solved |
| `train/step_routing_loss` | 0.5892 ± 0.0458 | 0.5715 ± 0.0349 | 0.5868 ± 0.0451 | **weight 0.0** |

Reported separately, never summed, per `CLAUDE.md` §2/§4. `N/A` on the MoR column
is structural: one expert has no load to balance and no entropy over experts to
report — not a zero.

**INTERPRETATION — four readings, in order of consequence.**

1. **The balance term is a diagnostic, not a driver.** Weighted, it is 1.3% of the
   task term. Normalized entropy at 0.9966–0.9980 out of 1.0 means the router is
   very close to uniform load, and `switch_aux_term` ≈ 1.0 says the same thing from
   the Switch-style side. High entropy is *load balance*, and per `CLAUDE.md` §2 it
   is explicitly **not** evidence of specialization — §8 measures that separately
   and finds it middling. The old primary criterion `expert_entropy > baseline` is
   void.
2. **The negative sign on `routing_balance_loss` is expected, not a bug.** The term
   is assembled as the Switch auxiliary minus the raw entropy, and the reported
   numbers satisfy that identity exactly: 0.9853 − 1.7858 = −0.8005 (MoE),
   1.0192 − 1.7882 = −0.7690 (MoRE). A near-uniform router makes the entropy bonus
   dominate, so the assembled value is negative. This is one concrete reason
   `total_loss` may not be the headline metric — a negative auxiliary can drag a
   weighted total down while the task term is unchanged.
3. **`family_cls` is fully solved and therefore contributes almost no gradient by
   the end.** At 1e-4 to 1e-5 it has stopped teaching. This matters for reading §11:
   the ablation removes a term that is *solved*, not one that is actively shaping
   late training, so the routing partition it supports was established early.
4. **MoRE pays 9.3% more ponder cost than MoR** (0.2940 vs 0.2691) for the depth
   allocation §9 shows is no better. The `step_routing_loss` row is computed and
   logged at all three arms purely so its magnitude is visible; it is multiplied by
   `loss_weights.step_routing = 0.0` and contributes no gradient anywhere (§3).

---

## 8. Expert Routing and Specialization

**Why permutation-invariant metrics are mandatory here.** The canonical router is
unsupervised (`loss_weights.step_routing = 0.0`, §3). Nothing ties expert index 3
to family E3; a run that partitions the operations perfectly but labels the parts in
a different order scores near chance on raw accuracy. So raw accuracy is reported —
and then set aside in favour of Hungarian matching, AMI and purity
(`CLAUDE.md` §4).

**MEASUREMENT — n = 5, held-out, 6-way chance = 0.1667:**

| metric | MoE | MoRE |
|---|---|---|
| raw routing accuracy | 0.2016 ± 0.1578 | 0.1629 ± 0.1864 |
| **Hungarian-matched accuracy** | 0.5004 ± 0.0560 | **0.5258 ± 0.0612** |
| matched macro recall | 0.5133 ± 0.0551 | 0.5491 ± 0.0984 |
| AMI | **0.5132 ± 0.0847** | 0.4907 ± 0.0736 |
| cluster purity | 0.5715 ± 0.0630 | **0.6047 ± 0.0618** |
| `expert_load_entropy_normalized` | 0.9886 | 0.9875 |
| per-expert load range | 15.41 – 19.09% | 14.46 – 18.32% |
| mean pairwise cosine similarity | 0.0227 ± 0.0075 | 0.0261 ± 0.0050 |
| max pairwise cosine similarity | 0.0512 ± 0.0239 | 0.0733 ± 0.0261 |

**MEASUREMENT — raw accuracy has a standard deviation larger than its mean** for
MoRE (0.1629 ± 0.1864) and nearly so for MoE. **INTERPRETATION:** this is the
predicted signature of an unsupervised router, not instability in the partition
itself — the *same* runs are stable under Hungarian matching (std 0.0612). Any
paper text quoting raw routing accuracy for these runs would be reporting label
permutation noise.

**INTERPRETATION — the partition is real, incomplete, and no better in MoRE than in
MoE.** Hungarian-matched accuracy of 0.5258 is 3.2× chance, so an unsupervised
router did discover structure aligned with the operation families from the task
objective alone. It is also roughly half-wrong. MoRE leads MoE on Hungarian
(+0.0254) and purity (+0.0332) but trails on AMI (−0.0225), all well inside one
standard deviation of each other — three permutation-invariant metrics disagreeing
on the sign is the definition of no measurable difference. Recursion neither helps
nor harms the routing partition.

**MEASUREMENT — load is near-uniform and no expert is dead.** Normalized load
entropy 0.9875–0.9886; every expert carries 14.5–19.1% of tokens against a uniform
16.67%. No collapse in any of the 10 six-expert runs.

**MEASUREMENT — `dispatch/max_load_fraction` = 1.0000 for MoR and MoRE.**
**INTERPRETATION — this is an artifact of how the metric aggregates, not a collapse
signal, and it must not be quoted as one.** The quantity is aggregated as the
`max` over recursion steps — the *worst* step. With `max_depth = 7` and 93–97% of
tokens exiting at step 2, the deep steps hold a handful of stragglers, so the
busiest expert trivially holds 100% of a nearly empty step. The per-expert loads in
the table above are the informative measurement and they are near-uniform. For MoE
(`max_depth = 1`) there is only one step and the number is meaningful.

**INTERPRETATION — cosine similarity is consistent with differentiated
parameterizations and proves nothing stronger.** Mean pairwise 0.023–0.026, max
0.051–0.073, i.e. the six expert weight matrices are close to mutually orthogonal
in direction. Per `CLAUDE.md` §4 the correct phrasing stops there: near-zero cosine
similarity between independently initialized matrices is the *expected* state in
256 dimensions, so this is a check that experts did not converge to copies of each
other, not evidence that they learned complementary functions. The functional
claim is the Hungarian/AMI/purity block above, and that one is middling.

**MEASUREMENT — MoRE's max pairwise cosine similarity is 43% higher than MoE's**
(0.0733 vs 0.0512), the largest relative difference in this section.
**HYPOTHESIS (not settled here).** That recursive reuse pulls experts toward each
other — every expert must remain a sensible function of its own output at depths
2…7, which is a shared constraint that MoE's single-pass experts never face. Both
values are small in absolute terms and the standard deviations (0.024–0.026)
overlap; distinguishing this from noise needs more seeds or a direct
representational-drift measurement, and neither exists here.

---

## 9. Adaptive Recursion and Halting

**What is measured, and at which model state.** Every depth number in this section
is `val_offline/*` — recovered from each run's saved `checkpoint.pt` by
`code/eval_val_depth.py`, on the held-out split, in `model.eval()`. The alternative,
the `train/*` and `depth/*` keys, measures the model *while it was being updated*,
under dropout, on training data, with the halt head mid-optimization; it cannot
support a claim about a trained model's policy. **Caveat carried on every figure
here:** `checkpoint.pt` is the best-*validation* checkpoint, while §6's headline
loss is last-epoch, so a depth figure and a loss figure from this document are not
guaranteed to describe the same weights.

**MEASUREMENT — per seed and aggregate, n = 5:**

| | MoR abs err | MoR steps | MoRE abs err | MoRE steps |
|---|---|---|---|---|
| 42 | 0.9891 | 2.1522 | 1.0566 | 2.3549 |
| 43 | 0.9332 | 2.1430 | 1.1437 | 2.5301 |
| 44 | 0.9340 | 2.1134 | 0.8843 | 2.2581 |
| 45 | 0.9325 | 2.1242 | 0.9622 | 2.2702 |
| 46 | 0.9226 | 2.1088 | 0.9231 | 2.3523 |
| **mean ± std** | 0.9423 ± 0.0266 | 2.1283 ± 0.0188 | 0.9940 ± 0.1054 | 2.3531 ± 0.1087 |

| metric | MoE | MoR | MoRE | contrast (MoRE − MoR) |
|---|---|---|---|---|
| avg recursion steps | N/A (`max_depth`=1) | 2.1283 ± 0.0188 | 2.3531 ± 0.1087 | +0.2248, d 2.88, **p 0.0079** |
| depth alloc. error, abs | N/A | 0.9423 ± 0.0266 | 0.9940 ± 0.1054 | +0.0517, d 0.67, p 0.3571 |
| depth alloc. error, rel | N/A | 0.5939 ± 0.0225 | 0.6934 ± 0.0931 | +0.0995, d 1.47, **p 0.0159** |
| early-exit rate | N/A | 0.9986 ± 0.0030 | 0.9997 ± 0.0007 | |
| forced-exit rate | N/A | 0.0014 | 0.0003 | |
| mean ACT remainder | N/A | 0.1283 ± 0.0402 | 0.2651 ± 0.0599 | |

`N/A` for MoE is structural — at `max_depth = 1` there is no allocation to measure,
and a 0.0 here would read as perfect allocation.

**MEASUREMENT — depth distribution.** MoE places 100% of tokens at step 1 by
construction. MoR places 97.12% ± 1.06 at exactly step 2. MoRE places 92.66% ± 3.18
at step 2 and 4.88% ± 3.79 at step 3. Forced exits at `max_depth = 7` are
essentially absent (0.03–0.14%), so the ponder cost is binding and the depth ceiling
is not.

**INTERPRETATION — MoRE buys more computation and gets nothing for it.** It spends
0.2248 more recursion steps per token than MoR, a difference resolved at p = 0.0079,
and in exchange its absolute allocation error is not better (p = 0.3571, and the
point estimate is *worse*) while its relative error is significantly **worse**
(p = 0.0159). More depth, no better matched to the curriculum. On the halting half
of the architecture this is `CLAUDE.md` §8 **Outcome C**.

### 9.1 The null model: is any of this adaptive?

An absolute error of 0.94 steps has no scale on its own. The null for "the model
learns to allocate recursion depth in accordance with the predefined
operation-complexity curriculum" is the **best constant-depth policy** — a policy
that ignores its input entirely and emits one number for every token.
`code/depth_null_model.py` computes it on the same split, through the same loader,
with the same `metrics.depth_allocation_error` function and the same
per-batch accumulation as the per-run sidecars, so the numbers are directly
comparable. It reads no checkpoint: it is a property of the data and the curriculum
table, identical for every seed and architecture
(`results/depth_null_model.json`).

**MEASUREMENT — the curriculum, over the 19,843 valid held-out step-tokens.**
Target depth 1: 38.10% of tokens (7,561), depth 2: 28.20% (5,596), depth 3: 18.46%
(3,664), depth 4: 15.23% (3,022). Token-weighted mean target depth **2.1082**. The
unweighted mean over the 16 operation *types* is 1.8750 — a different quantity, and
the wrong one to compare a token-level error against.

**MEASUREMENT — the null, scanned over c ∈ [0.50, 7.00] at step 0.01:**

| policy | abs error | rel error |
|---|---|---|
| oracle (`pred = target`) | 0.0000 | 0.0000 |
| **best constant, c = 2.00** | **0.8703** | 0.5186 |
| best constant for rel error, c = 1.00 | 1.1085 | **0.3784** |
| constant c = 3.00 | 1.1964 | 0.9410 |
| **MoR, measured** | 0.9423 ± 0.0266 | 0.5939 ± 0.0225 |
| **MoRE, measured** | 0.9940 ± 0.1054 | 0.6934 ± 0.0931 |

**MEASUREMENT — neither trained halt head beats the constant, on any seed.**

| | abs error | vs null 0.8703 | seeds worse | sign test p (2-sided) |
|---|---|---|---|---|
| MoR | 0.9423 | **+0.0720 (+8.3%)** | **5 / 5** | 0.0625 (its floor at n = 5) |
| MoRE | 0.9940 | **+0.1237 (+14.2%)** | **5 / 5** | 0.0625 (floor) |
| pooled | | | **10 / 10** | 0.00195 |

Relative error is worse against the null too: every one of the 10 runs exceeds both
0.3784 (the rel-optimal constant) and 0.5186 (the abs-optimal constant's rel error).

**MEASUREMENT — the comparison survives matching the operating point.** A constant
policy set to each architecture's *own* measured mean depth still wins: at c = 2.13
the null scores 0.9127 against MoR's 0.9423 (+0.0296, +3.2%); at c = 2.35 it scores
0.9844 against MoRE's 0.9940 (+0.0096, +1.0%).

**INTERPRETATION — the halt head learned a level, not a policy.** Both models sit
close to the right *average* depth — MoR's 2.1283 is within 1% of the curriculum's
token-weighted mean 2.1082 — and both are beaten by a policy with no input at all.
Matching each model against a constant at its own mean rules out the obvious
defence that it merely chose a different operating point: what remains is the
per-token *variation*, and that variation makes allocation worse rather than better.

**What that does and does not establish.** Any variation independent of the target
also increases mean absolute error (|·| is convex), so this result cleanly rules out
the claim that the halt head's per-token variation is *aligned* with the curriculum;
it does not separate actively-misaligned variation from unstructured noise. Either
way the conclusion for the paper is the same: **the per-token variation carries no
usable curriculum signal.** A constant-2 policy is wrong by at least one step on
71.8% of tokens, and the trained adaptive policies are wrong by more.

**INTERPRETATION — the mechanism works; the behaviour did not materialize.** Gate 2
verifies with 31 assertions that halting is real ACT: a learned per-step halt
probability, genuine early exit, halted states frozen and not recomputed, a forced
exit at `max_depth`, a differentiable ponder cost, and a real gradient reaching the
halt parameters. All of that holds. What the ponder cost plus task loss selected for,
at this scale on this task, was an approximately unconditional ≈2 steps. Mechanical
adaptivity is not behavioural adaptivity, and this document does not claim the
latter anywhere.

**HYPOTHESIS (not settled here).** That the halt head is under-incentivized rather
than incapable: with `loss_weights.halting = 0.001`, the ponder cost is ~0.5% of the
weighted objective, and the task loss barely rewards correct depth because the
regression target is reachable at almost any depth on a compressed target (§4). A
supervised-halting arm (`halting_mode ≠ pure_act`, against
`families.OP_TARGET_DEPTH`) would separate "cannot" from "was not asked to". That
arm exists in the code and was not run as part of the canonical matrix; the
`ponder_cost_low` ablation (§11) varies the weight but not the supervision.

---

## 10. Compute and Throughput

**MEASUREMENT — n = 5, one RTX 4060 Laptop (8 GB), batch 768, identical data
pipeline:**

| | throughput (tok/s) | vs MoE | `expert_evaluations` / epoch | `evals_per_token` |
|---|---|---|---|---|
| MoE | 8485 ± 362 | 1.00× | 224,450 | 1.0000 ± 0.0000 |
| MoR | 4574 ± 713 | 0.54× | 451,445 ± 3,951 | 1.0000 ± 0.0000 |
| MoRE | 2225 ± 274 | 0.26× | 463,991 ± 12,427 | 1.0000 ± 0.0000 |

`evals_per_token` = 1.0000 exactly is the measured sparsity claim: Top-1 dispatch
evaluates one expert per token per step, never a dense blend. The labelled
`dense_routing` ablation reports 6 on the same key (§11), which is how we know the
metric can distinguish the two.

**MEASUREMENT — `expert_evaluations` counts invocations, not arithmetic.** Scaling
each invocation by its expert's hidden width (MoR 6144, MoE/MoRE 1024) gives a
first-order FFN arithmetic proxy per epoch: MoE 2.30e8 width-units, MoRE 4.75e8
(2.07× MoE), MoR 2.77e9 (**12.1× MoE, 5.8× MoRE**).

**INTERPRETATION — MoRE does one sixth of MoR's FFN arithmetic and takes twice as
long.** That is the central compute finding, and it is not about FLOPs. MoR runs
5.8× more FFN arithmetic than MoRE at 2.06× MoRE's throughput; combining the two,
MoR converts arithmetic into wall clock about 12× more efficiently. The cost is
Top-1 dispatch: gathering tokens per expert, running six small matmuls instead of
one large one, and scattering results back — repeated at every one of up to 7
recursion steps. At `d_model = 256` and expert width 1024 the matmuls are far too
small to amortize that overhead, so MoRE pays the routing tax 7 times per token and
banks none of the arithmetic saving.

**INTERPRETATION — sparsity bought nothing measurable here, on either axis.** MoRE
is statistically indistinguishable from MoE in quality (§6, p = 0.6587) while
running 3.81× slower, and it is worse than MoR in quality while doing a sixth of the
arithmetic. The usual justification for sparse routing — more parameters at constant
FLOPs — cannot apply in this study by construction, because §5 pins all three arms
to the same parameter count. What is left of the sparsity argument at matched
parameters is that MoRE should be *faster*, and it is 3.81× slower.

**Scope limit on every number in this section.** This measures one un-optimized
PyTorch implementation on one consumer GPU. Dispatch uses index/scatter operations,
not fused grouped-GEMM kernels, and no expert-parallel or capacity-factor batching
is implemented. A production MoE kernel would change these ratios substantially.
The *quality* results in §6 do not depend on any of this; the throughput ratios
are properties of this implementation and must be labelled as such wherever they
appear. What they legitimately establish is that at this scale sparse routing has
no wall-clock case to offset its quality result — not that sparse routing is
intrinsically slow.

## 11. Ablation Studies

**Design.** Six arms, each varying **exactly one** field of the canonical MoRE
configuration against the same MoRE baseline, at the frozen seeds. The baseline is
the canonical MoRE runs themselves, not a re-trained control, and each arm is
compared at matching seed counts. Reading rule is the exact two-sided randomization
test on the difference of means, with each arm's resolution floor `min_p` printed
beside its p. Six arms means the family-wise threshold is **α = 0.05/6 = 0.0083**;
an arm at nominal p < 0.05 that exceeds 0.0083 is reported as nominal only.

**Read the baseline numbers carefully.** Because baselines are seed-count-matched, the
same canonical MoRE metric appears with two different values in this section — the
3-seed baseline for the n = 3 arms and the 5-seed baseline for the n = 5 arms. Held-out
depth allocation error, for example, is 1.0282 ± 0.1320 over seeds 42–44 and
0.9940 ± 0.1054 over all five (the figure §9 reports). Every comparison below states
which baseline it uses; none mixes seed counts.

| Arm | Field varied | n | Δ val task loss | se | d | p | floor | Verdict |
|---|---|---|---|---|---|---|---|---|
| `dense_routing` | `routing_mode = dense_blend` | 5 | **+0.000529** | 0.000176 | +1.90 | **0.0079** | 0.0040 | worse; survives Bonferroni |
| `routing_supervision` | `step_routing = 0.5` | 5 | **+0.000565** | 0.000124 | +2.89 | **0.0079** | 0.0040 | worse; survives Bonferroni |
| `fixed_depth` | `fixed_depth = True` | 5 | **−0.000301** | 0.000128 | −1.48 | 0.0476 | 0.0040 | better; nominal only, fails Bonferroni |
| `ponder_cost_low` | `halting = 0.0001` | 3 | +0.000444 | 0.000295 | +1.17 | 0.1607 | 0.0179 | unresolved |
| `router_noise` | `router_noise = fixed_annealed` | 3 | +0.000176 | 0.000256 | +0.53 | 0.4107 | 0.0179 | unresolved |
| `two_blocks` | `num_blocks = 2` | 3 | +0.000072 | 0.000269 | +0.21 | 0.7500 | 0.0179 | unresolved |

Positive Δ means the ablation is **worse** than canonical MoRE. The three n = 3 arms
sit at p = 0.16–0.75, far above their 0.0179 floor, so additional seeds would not
have changed their verdicts and were not spent; `fixed_depth` was extended from 3 to
5 seeds precisely because it was the one arm close enough to its floor for the
verdict to move (§15.10, §15.11).

### 11.1 Sparse Top-1 dispatch is not paying for itself — but dense is worse

Replacing argmax dispatch with evaluate-all-and-blend raises `evals_per_token` from
1.0 to 6.0 and makes quality **worse** (0.063716 ± 0.000305, p = 0.0079,
Bonferroni-surviving). It also runs **faster**: 3051 ± 311 vs 2225 ± 274 tok/s.

**INTERPRETATION.** Six times the expert arithmetic executed 1.37× faster than
routing each token to one expert, which is direct confirmation of §10's account —
the cost at this scale is per-step gather/scatter over six narrow experts, not the
FFN arithmetic. So sparsity is *not* free (it costs wall clock), and it is also *not
wasted* (dense scores worse). Dense routing also degraded the partition, compared at
matching seed count against the 5-seed canonical MoRE baseline: Hungarian
0.4717 ± 0.0497 vs 0.5258 ± 0.0612 and AMI 0.3927 ± 0.0260 vs 0.4907 ± 0.0736 —
consistent with a blended path giving the router weaker pressure to commit.

### 11.2 Supervising the router perfects the routing metric and hurts the task

Setting `step_routing = 0.5` drives every routing metric to **1.0000 ± 0.0000** —
raw accuracy, Hungarian, AMI, purity, matched macro recall, all exactly perfect —
while validation task loss gets significantly **worse** (0.063751 ± 0.000121,
d = +2.89, p = 0.0079, Bonferroni-surviving). Expert load entropy falls to
0.9481 ± 0.0082 and the normalized entropy term to 0.8321 ± 0.0285.

**INTERPRETATION — this is the most decision-relevant ablation in the set.** The
oracle partition is learnable to perfection and learning it makes predictions worse.
Two consequences. First, routing quality cannot be used as a proxy for architectural
quality anywhere in this work: the two are not merely weakly correlated, they are
anti-correlated when routing is optimized directly. Second, it bounds the ceiling of
§8's unsupervised partition story — reaching Hungarian 1.0 was never the goal, because
the configuration that achieves it is worse at the task. The likely mechanism (stated
as hypothesis, not finding) is that the oracle family partition is not the
token-level partition that minimizes regression error, so the supervision term pulls
the router away from a better-for-the-task assignment.

### 11.3 Fixed maximum depth beats learned halting on quality

Forcing all seven recursion steps for every token gives **0.062886 ± 0.000143**
against canonical MoRE's 0.063186 ± 0.000249 — better by 0.000301, d = −1.48,
**p = 0.0476**, which is nominally significant and **fails** the family-wise
threshold 0.0083. Best-val loss agrees in direction (0.062715 ± 0.000166 vs
0.062892 ± 0.000216). Depth allocation error is, necessarily, 5.8850 ± 0.0002.
Throughput is **2198 ± 274 tok/s — indistinguishable from canonical MoRE's
2225 ± 274** despite running 3.4× the recursion steps.

**INTERPRETATION.** Three things, in decreasing strength. (i) Learned halting did not
buy quality; the direction favours the fixed policy and the family-wise correction
declines to certify it, so the honest statement is *no evidence that adaptive halting
helps, weak evidence that it hurts.* (ii) Learned halting did not buy **compute**
either: the halting machinery's per-step overhead consumed the wall-clock saving from
exiting at ~2.1 steps instead of 7, leaving throughput unchanged. Adaptive
computation is normally justified by a quality-per-FLOP argument, and at this scale
neither side of that ratio improved. (iii) The arm that is maximally wrong about the
depth curriculum is the arm that fits best, which is why §15.13 treats
`depth_allocation_error` as agreement-with-an-assumption rather than correctness.

### 11.4 Lowering the ponder cost raises depth without improving loss

`ponder_cost_low` (halting weight 0.001 → 0.0001, nothing else changed) was
pre-registered with an explicit reading: if depth rises toward `max_depth` **and**
loss moves to the fixed-depth number, the ACT objective was mis-weighted; if depth
rises and loss does not follow, the halt head is not learning anything useful.

Depth rose — train 2.5120 ± 0.0876 vs the matched 3-seed baseline's 2.0524 ± 0.0749,
held-out 2.7721 ± 0.0746 vs 2.3810 ± 0.1379, forced-exit rate 0.0104 ± 0.0089 vs
0.0003 ± 0.0003 — and loss did **not** follow: 0.063630 ± 0.000473, *worse* than
baseline, p = 0.1607 at floor 0.0179. Depth allocation error also worsened, to
1.1495 ± 0.0568 held-out against the baseline's 1.0282 ± 0.1320.

**INTERPRETATION.** The pre-registered branch that fired is the second one, and its
stated conclusion is Outcome C for the depth half of the architecture. Two
qualifications keep this honest: at n = 3 the test cannot resolve the loss difference
at all, so the *direction* is what fired, not a significant result; and the arm rules
out the specific "our ponder coefficient is 10× too strong" explanation, not every
possible mis-specification of the halting objective.

### 11.5 A second recursive block doubled the parameters and changed nothing

`num_blocks = 2` takes total parameters from 3,201,555 to 6,358,553 (+98.6%) for
Δ = +0.000072 at p = 0.7500, while throughput falls to 1587 ± 187 tok/s (−29%).
**INTERPRETATION:** at n = 3 this is unresolved, but the effect size (d = 0.21) is
small enough that the *interesting* reading is not "two blocks is worse" — it is that
doubling capacity produced no measurable movement in either direction, which is what
§15.5's ceiling (best R² 0.2250) would predict if the binding constraint is the task
representation rather than model capacity.

### 11.6 Router exploration noise had no measurable effect

`router_noise = fixed_annealed` gives Δ = +0.000176 at p = 0.4107, with the partition
essentially unchanged (Hungarian 0.4853 ± 0.0237 vs the matched 3-seed baseline's
0.5538 ± 0.0468; AMI 0.4633 ± 0.0105 vs 0.5092 ± 0.0620). **INTERPRETATION:** unresolved at n = 3, and no reason from this evidence to
revisit the canonical `router_noise = none`. The `trainable` variant was deliberately
*not* ablated, because its L2-on-noise-scale mechanism is not an established fix and
testing it would confound exploration noise with that unproven machinery.

### 11.7 How much of the partition depends on the external family labels

**Status: EXPLORATORY PROXY — 20 epochs, 3 seeds, `experiment_group = exploratory`.**
These runs are not admissible to any headline table and are reported here only
because the question in §15.8 has no canonical answer on disk.

| | `family_cls = 0.5` (canonical) | `family_cls = 0.0` | Chance-corrected retained |
|---|---|---|---|
| Hungarian accuracy | 0.5147 ± 0.0047 | 0.4300 ± 0.0293 | **75.7%** |
| AMI | 0.4940 ± 0.0395 | 0.3568 ± 0.0431 | **72.2%** |
| Purity | 0.5881 ± 0.0135 | 0.5041 ± 0.0539 | **80.1%** |
| val task loss | 0.063521 ± 0.000428 | 0.063421 ± 0.000157 | — |
| Collapse | none / none / none | none / none / none | — |

Retained fractions are chance-corrected — `(arm − 0.1667)/(canonical − 0.1667)` for
Hungarian and purity, and the ratio directly for AMI, which is already
chance-corrected.

**INTERPRETATION.** Removing the family-label term entirely costs roughly a quarter of
the chance-corrected partition quality and leaves ~72–80% of it intact, with no
collapse and no change in task loss. So §8's partition is **not** an artifact of the
external annotation — a majority of it survives with no oracle label anywhere in the
objective — but it is also not fully self-supervised, and the unqualified claim is
withdrawn (§18, item 11). The honest form is: *most of the measured partition
survives removal of the only oracle-derived loss term, measured at reduced epochs on
three seeds.*



---

## 12. Parameter-Matched Comparison

**MEASUREMENT — the match, verified by instantiation.** Re-measured for this
document by building each model and summing `named_parameters()`:

| `mor.ffn_mult` | total params | vs MoRE's 3,201,555 |
|---|---|---|
| 4 | 571,150 | +82.16% short |
| 23 | 3,066,382 | +4.22% short |
| **24 (canonical)** | **3,197,710** | **+0.12% short** |
| 25 | 3,329,038 | −3.98% over |

`24 = 4 × 6` equalizes total FFN hidden width: MoR's one FFN is as wide as MoRE's
six experts combined. §5 itemizes the residual 3,845 parameters and shows the FFN
weight matrices are **exactly equal** at 3,145,728 each, so the match is not
approximate in the component that carries the capacity.

**INTERPRETATION — what the match rules out.** No §6 result can be explained by a
parameter-count difference, and specifically not by MoR having more FFN weight than
MoRE. The three arms also share tokenizer, data pipeline, optimizer settings,
epoch budget, batch protocol, seeds and evaluation path (§2), so the surviving
differences are exactly three: how many experts the FFN capacity is split into, how
many times the block is applied, and the two auxiliary loss weights that only exist
for one of those choices (§3).

**MEASUREMENT — what is *not* measured: no trained under-budgeted baseline.**
The only `mor.ffn_mult = 4` run on disk is
`t10i_mor_ffn_mult_4_proxy_seed42`, a **1-epoch, single-seed, `exploratory`** run.
Per `CLAUDE.md` §6 a proxy cannot enter a results table, so no row for it appears
anywhere in this document, and the `ffn_mult_4` baseline exists here as a
*parameter count only*.

**INTERPRETATION — the consequence, stated as a gap and not softened.** The
parameter match is established structurally but not empirically bracketed. We can
say MoR wins at equal parameters; we cannot show the shape of the curve on either
side of the match, because no *trained* run exists at 0.57 M (`ffn_mult = 4`) or at
any larger FFN width. The one arm above 3.2 M is `two_blocks` at 6.36 M (§11.5), and
it does not fill this gap: it adds a second recursive block rather than widening the
FFN, and it is MoRE rather than MoR. A reader asking "is MoR's advantage a width
effect or a weight-sharing effect?" gets no answer from these experiments. The two
runs that would answer it — MoR at expert width 1024, and MoRE with six width-6144
experts — were never launched, and that is the same missing sweep as §13.

## 13. Preliminary Capacity Sweep

**Status: NOT AN EXPERIMENT.** This is a wall-clock microbenchmark
(`code/bench_capacity.py`, log at `results/bench_capacity.log`). It builds the real
`MoREModel` and times forward + backward + optimizer step on **synthetic** batches of
the canonical shape — deliberately, to remove the dataloader and leave model compute
plus Python/kernel-launch overhead. It reads no run directory, trains nothing, and
**measures no quality at any width.** Nothing here can be quoted as a result about
architecture quality, and the scale confound in §19.3 is *not* resolved by it.

It exists to answer one operational question: the canonical matrix ran with the GPU at
~0–30% utilization and ~2 W, so would a bigger model use the GPU properly and
therefore finish *faster*?

### 13.1 Batch sweep at canonical `d_model = 256` — usable, and conclusive

| batch | ms/step | ×canonical ms | ms per step-token | peak MiB |
|---|---|---|---|---|
| 96 | 109.11 | 0.76 | 162.37 | 136 |
| 192 | 115.15 | 0.80 | 85.68 | 208 |
| 384 | 117.02 | 0.81 | 43.54 | 362 |
| **768 (canonical)** | **143.76** | **1.00** | **26.74** | **695** |
| 1536 | 153.03 | 1.06 | 14.23 | 1293 |
| 3072 | 179.57 | 1.25 | 8.35 | 2447 |
| 6144 | 275.36 | 1.92 | 6.40 | 4732 |

**MEASUREMENT.** A 64× increase in batch size costs 2.5× the wall clock per step, and
per-token time falls monotonically by 25×. From canonical, 8× the batch costs 1.92×.

**INTERPRETATION.** This is a textbook launch-bound signature and it explains the 2 W
idle GPU directly: at the canonical batch the step is dominated by kernel-launch and
Python overhead, not by arithmetic. Practical consequence — the canonical batch of 768
leaves large free headroom, and a future sweep can raise it substantially at
near-constant wall clock per step. It does **not** follow that the canonical runs
should be re-run at a larger batch: batch size is a hashed config field, so changing it
changes run identity and would invalidate every comparison in this document.

### 13.2 The width sweep is unusable and is reported as such

The width sweep at fixed batch 768 produced: `d_model` 256 → 143.88 ms, 384 → 268.54,
**512 → 72.47**, 768 → 197.53, 1024 → 160.07.

**This is rejected as a measurement.** A 12.76 M-parameter model cannot run a step in
half the time of a 3.20 M-parameter model on the same loop — it violates the
benchmark's own stated invariant that *"a wider model can never make a run finish
sooner."* The cause is named in the script's own measurement-hygiene note: a laptop GPU
idling at 2 W is heavily downclocked, and per-configuration warmups of 12 iterations
with 3 repeats are not enough to hold the clock steady across a five-point sweep, so
the numbers measure clock ramp as much as throughput. Reporting a width scaling curve
from this would be fabrication. **No width-vs-cost claim is made, and no width-vs-
quality number exists at any width other than canonical.**

### 13.3 One honest cross-check on the training loop

Units differ between the two measurements and must be reconciled before comparing:
the microbenchmark's `tok/s` counts **step-tokens** (`batch × 7`), while the training
loop's `perf/throughput_tokens_sec` counts **records**. At the canonical batch the
microbenchmark's 37,395 step-tokens/s is **5,342 records/s**. The microbenchmark also
always runs all seven steps with no early exit, so the like-for-like training
comparator is the `fixed_depth` arm at **2,198 ± 274 records/s** (§11.3).

**INTERPRETATION.** Roughly 2.4× of the training loop's wall clock sits outside
forward + backward + step — dataloading, loss assembly, metric accumulation and
logging. That is a tractable engineering gap, and it means the throughput figures in
§10 and §11 measure *this training loop*, not the architectures' intrinsic cost. The
§10 caveat stands unchanged: the 3.81× MoRE-vs-MoE penalty is a property of one
un-optimized implementation at one scale.



---

## 14. Statistical Stability

**The reading rule, stated once.** Exact two-sided randomization test on the
difference of means (`code/seed_stats.py`), sample standard deviation (n−1), every
p reported with `min_p = 1 / C(n_a + n_b, n_a)`: **0.0040 at 5-vs-5, 0.0179 at
3-vs-5.** A p equal to its floor is the test's resolution limit, not a significance
claim. The superseded rule `|gap| ≥ 2 × max(seed std)` is withdrawn (T11.0b) —
it compared a gap against a single arm's spread instead of the standard error of the
difference, which systematically overstates significance when the two arms have
unequal variance. It appears nowhere in this document.

**MEASUREMENT — seed spread on the primary metric, n = 5:**

| | std | range | std as % of mean | widest seed |
|---|---|---|---|---|
| MoE | 0.000153 | 0.000402 | 0.24% | 44 (high) |
| MoR | **0.000112** | **0.000286** | **0.18%** | 43 (high) |
| MoRE | 0.000249 | 0.000593 | 0.39% | **46 (low, 0.062748)** |

**INTERPRETATION — MoRE is the least stable arm, by a factor of ~2.**
Its seed std is 2.2× MoR's and 1.6× MoE's, and its range is 2.1× MoR's. Adding a
second learned discrete decision (which expert) on top of a learned continuous one
(how deep) plausibly widens the basin of outcomes, but with n = 5 per arm this is an
observation about spread, not a tested claim about variance — an F-test on 5-vs-5
samples has almost no power, and none is reported.

**MEASUREMENT — seed 46 is MoRE's best run and it is an outlier.** At 0.062748 it
sits 0.000438 below MoRE's own mean (1.8 sample std) and only 0.000042 above MoR's
mean. Excluding it, MoRE's mean is 0.063296 ± 0.000051. **INTERPRETATION:** this is
reported to make one thing impossible — quoting seed 46 as "MoRE matches MoR". One
favourable seed is never evidence (`CLAUDE.md` §5), and the n = 5 test that includes
it already returns p = 0.0159 against MoR.

**MEASUREMENT — where the resolution floor binds.** All three canonical pairwise
tests are 5-vs-5, floor 0.0040, and none of the three reported p values sits at it
(0.0079, 0.0159, 0.6587), so all three are resolved. The Phase 10 arms are a
different matter: with Bonferroni over six arms the threshold is
α = 0.05/6 = 0.0083, and a 3-vs-5 arm has floor 0.0179 — **more than twice the
threshold.** A 3-seed arm therefore *cannot* be declared significant no matter how
large its effect. That is a property of the design, not of the arms, and §11 states
it per arm rather than leaving a reader to compare two numbers in different
paragraphs.

**INTERPRETATION — what n = 5 can and cannot support.** It resolves the
between-architecture ordering on this task: three tests, effect sizes 0.36 to 4.13,
two of them below α = 0.05 without correction and the MoE−MoR result at p = 0.0079
also below a 6-way-corrected threshold. It does not support any claim about
variance, any claim about a subgroup of seeds, or any claim that a null result
(MoRE ≈ MoE at p = 0.6587) is an equivalence. The standard error of that particular
difference is 0.000131, so a true gap of 0.0002 — 1.5 standard errors, and 36% of
the MoE−MoR gap we *did* resolve — would very likely have gone undetected.
**"Indistinguishable at n = 5" is the honest phrasing; "equivalent" is not.**

---

## 15. Failure Cases and Negative Results

This section exists because `CLAUDE.md` §8 requires Outcome C to be reported
honestly, and because a reader who finds a problem here that we did not name would
be right to distrust everything else. Model-level failures first, then measurement
and process failures.

### 15.1 The adaptive-computation mechanism did not produce adaptive behaviour

**The strongest negative result in this work, and it now rests on three independent
lines.** Both recursive arms are beaten by a constant-depth policy on held-out data,
on every seed, and the comparison holds after matching each model's own mean depth
(§9.1). Forcing all seven steps for every token gives *better* task loss at
indistinguishable throughput (§11.3). Weakening the ponder cost tenfold raises depth
without improving loss, which is that arm's own pre-registered Outcome-C branch
(§11.4). The halt head learned a level (≈2 steps), not an input-dependent policy.
Everything the architecture requires mechanically is present and gate-verified; the
behaviour the mechanism exists to produce is absent.

### 15.2 The composition of routing and recursion added nothing

MoRE is indistinguishable from MoE (p = 0.6587) despite §5 establishing that MoE is
*literally* MoRE with `max_depth = 1` and identical parameter tensors, and MoRE is
significantly worse than MoR (p = 0.0159). Both halves of the novel contribution
came back null-or-negative on this task at this scale. This is the finding, not a
setback to be worked around.

### 15.3 MoRE fits best and generalizes worst

Lowest training loss of the three (0.060237) and a generalization gap of 0.002949 —
2.4× MoR's 0.001205 (§7). The extra capacity is being used; it is not transferring.
The gap ordering tracks expert count, not depth.

### 15.4 MoRE is the least stable arm

Seed std 0.000249, 2.2× MoR's, range 2.1× MoR's, with seed 46 far enough below its
own arm's mean to be MoR-competitive alone (§14). Named explicitly so it cannot be
quoted in isolation.

### 15.5 The task is barely learned by anything

R² is 0.2182 / 0.2250 / 0.2191 against the predict-the-train-mean floor. Between
73% and 78% of held-out target variance is unexplained by every architecture tested,
and the entire between-architecture spread is 3.0% of what they do explain (§6).
**INTERPRETATION:** the binding constraint in this study is the task and the scale,
not the architecture. Any claim of the form "architecture X is better for
arithmetic reasoning" is unsupportable from a regime where none of them explains a
quarter of the variance.

### 15.6 Sparse routing cost 3.81× wall clock for no quality gain

And it did so while performing one sixth of MoR's FFN arithmetic (§10). Caveat
carried from §10: this measures one un-optimized implementation, not the intrinsic
cost of sparse routing.

### 15.7 Two metrics are structurally uninformative and must not be quoted

**MEASUREMENT.** Raw routing accuracy is 0.1629 ± 0.1864 for MoRE — standard
deviation larger than the mean. `dispatch/max_load_fraction` is 1.0000 for both
recursive arms while their per-expert loads are 14.5–18.3%.

**INTERPRETATION — both are measurement artifacts with known causes, not model
pathologies.** Raw accuracy is meaningless under an unsupervised router because
expert indices carry no semantic identity; the permutation-invariant metrics on the
same runs are stable (§8). `max_load_fraction` is aggregated as the `max` over
recursion steps, so with 93–97% of tokens gone by step 2 the deep steps hold a
handful of stragglers and the busiest expert trivially holds 100% of almost nothing.
Neither number indicates collapse, and neither belongs in a paper table for the
recursive arms. They are reported here so that a reader who finds them in the raw
metrics has the explanation.

### 15.8 `family_cls` was an undisclosed objective term for most of the project

**MEASUREMENT.** `loss_weights.family_cls = 0.5` weights a 6-way whole-program
family cross-entropy whose labels are read off disk from a `"family"` string stamped
into every record by the generator. It is the second-largest term in the objective
after the task loss. Until T8.3 it was a bare `0.5` literal inside `engine.py`: it
appeared in no config file, no provenance block, and no results table.

**INTERPRETATION — this is a methodological failure, and it is ours.** For most of
the project the models had access to external annotation that no reported
configuration disclosed. It is now in `canonical_spec.json:enforced_fields`, so a
run cannot claim canonical status without declaring it, and §11 reports an ablation
measuring how much of the routing partition depends on it. The reason it matters
beyond bookkeeping: a reader assessing whether §8's partition is "unsupervised"
needs to know that while the *router* receives no oracle gradient
(`step_routing = 0.0`), the *encoder* was trained against family labels that
correlate with the routing target.

### 15.9 Depth and loss are measured at different model states

**MEASUREMENT.** Every `val_offline/*` figure in §9 comes from `checkpoint.pt`, the
best-*validation* checkpoint. The primary metric in §6 is last-epoch
`val/task_loss`. **INTERPRETATION:** a depth figure and a loss figure from this
document may not describe the same weights, so no sentence anywhere may caption them
as one model. `measured_at: "best_val_checkpoint"` is written into every sidecar so
the distinction cannot be lost in transcription. Runs launched after T11.1 emit
`val/*` depth keys from the same validation pass that produces the headline loss;
none of the 15 canonical runs predating that change has them, and re-training for a
metric that needs no training was refused as a waste of GPU.

### 15.10 Three of six Phase 10 arms cannot reach their own significance threshold

**MEASUREMENT.** With Bonferroni over six arms, α = 0.05/6 = 0.0083. A 3-vs-5 arm
has resolution floor 0.0179 — more than twice α. **INTERPRETATION:** those arms are
un-decidable by construction, not merely inconclusive, and no amount of effect size
would change that. §11 states the floor next to every arm's p rather than leaving a
reader to compare two numbers in different paragraphs. The one arm whose verdict
could still move with more seeds was extended to n = 5 for exactly this reason; the
other three sit at p = 0.16–0.75, nowhere near the floor, so seeds would not have
changed their verdicts and were not spent.

### 15.11 A code-state guard produced a false positive, and was not weakened

**MEASUREMENT.** `automated/phase10_ablations.py` refuses to launch an arm when any
model source file is newer than the oldest reused baseline run, or when its sha256
has drifted from the pinned fingerprint. On 2026-08-29 it refused the extension in
§15.10. The refusal was a false positive: the only changed file was `engine.py`, and
the only change was the addition of validation-pass depth accumulation *inside*
`torch.no_grad()` in the eval loop — pure reductions over tensors the forward pass
already returned, consuming no RNG and touching no parameter or optimizer state.

**INTERPRETATION — the response was to prove the claim, not to edit the guard.**
Neither the guard nor the fingerprint was modified. Instead the arm's config was
re-run for one epoch on current code and compared against the recorded 50-epoch run
at epoch 1: `train_task_loss` 0.100573 vs 0.100573, plus exact agreement on
normalized expert entropy, average depth and both cosine-similarity diagnostics.
Epoch 1 is independent of total epoch count *for this arm specifically* — the LR
scheduler steps only after epoch 1, and the halt-loss warmup scales a ponder cost
that a fixed-depth arm does not produce. Aggregation still ran through the driver,
so the field-equality check and the resolution-floor warning were not bypassed;
only the launch precondition was. The evidence is recorded in
`automated/phase10_code_state_proof.json`, which ends by telling the next agent
that the guard will still refuse and should.

### 15.12 Non-determinism and refused directories, disclosed

**MEASUREMENT.** `torch.use_deterministic_algorithms(True, warn_only=True)` is set,
and `_histc_cuda` — reached only by W&B's parameter-histogram logging — has no
deterministic implementation. The warning is recorded in provenance rather than
suppressed. Separately, the exporter refused 110 run directories, three of which are
*named* `phaseB_moe__seedNA__*`, `phaseB_mor__seedNA__*` and
`phaseB_more_dense_routing_ablation__seedNA__*` and carry a training task loss of
0.0825 — essentially the 0.0809 trivial floor.

**INTERPRETATION.** Selection is on provenance fields, never on directory names, and
those three directories are the concrete reason: a name-glob would have pulled
barely-trained pre-provenance runs into the headline table. The nondeterminism
affects a logging histogram, not training; it is disclosed because "deterministic"
appears in the provenance block and must not be read as unqualified.

### 15.13 The depth curriculum itself is not validated by the task

**MEASUREMENT.** The `fixed_depth` ablation (§11) runs every token to `max_depth = 7`,
giving a depth allocation error of 5.8850 ± 0.0002 — the worst possible score against
a curriculum whose token-weighted mean target is 2.1082. It nonetheless achieved a
*lower* validation task loss than canonical MoRE: 0.062886 ± 0.000143 vs
0.063186 ± 0.000249, d = −1.48, p = 0.0476.

**INTERPRETATION.** Depth allocation error and predictive quality are not merely
weakly related here, they point in opposite directions: the arm that is maximally
wrong about the curriculum is the arm that fits held-out data best. Two readings are
available and this evidence does not separate them — either the hand-authored
per-operation depth targets are not the depths this task actually rewards, or extra
recursion helps for reasons unrelated to per-operation complexity (more compute is
simply better here). Either way, `depth_allocation_error` measures agreement with an
assumption, not correctness, and §9's depth numbers must be read that way.

## 16. Reproducibility and Run Manifest

**MEASUREMENT.** Every number in this document traces to a run directory under
`runs/`, and every run directory is admitted or refused by provenance fields only.
The exporter (`code/export_results.py`) reads `metrics.json` and
`resolved_config.json` and requires `experiment_group == canonical_phase_b`,
`variant == canonical`, `seed_declared == True`, a `resolved_seed` in the frozen set
`{42, 43, 44, 45, 46}`, and `dataset_version` / `train_split_version` equal to the
spec's; it emits `N/A` rather than a substitute when a key is absent. It refused 114
of the 129 directories it scanned. The 15 it admitted are:

| Architecture | Seeds | Run directories |
|---|---|---|
| MoE | 42–46 | `phaseB_moe_seed{42..46}__{c35fe8b6, 33afa654, d522a4d0, 5a7cb215, 0c558a88}` |
| MoR | 42–46 | `phaseB_mor_seed{42..46}__{87d75af9, 93eb15bc, 5bb04148, e5955ef2, 2ffafc33}` |
| MoRE | 42–46 | `phaseB_more_seed{42..46}__{989e89cc, 909a3dc4, ba0c483d, 891228d4, 0a9dce12}` |

The hash suffix is `config_hash[:8]`, so a directory name that survives a config
change is impossible by construction: change any hashed field and the directory name
changes with it. **The seed is inside that hash**, which is why all five seeds of one
arm hash differently and why the exporter cannot certify "one configuration per arm"
by comparing hashes. It instead compares a **seed-blind** flattening of
`resolved_config.json` — every field except the seed itself, the seed-derived subset
seeds, the run name, and per-run identity/timestamps — and refuses the whole export
if two seeds of one architecture differ in any remaining field. It additionally
refuses a duplicated `(architecture, seed)` cell, and a table spanning more than one
`dataset_version`, `train_split_version` or `code_git_commit`. **INTERPRETATION:**
without the seed-blind check a silent config drift between seed 42 and seed 46 would
turn a reported seed standard deviation into an unlabelled configuration sweep, which
is the specific way a five-seed table stops meaning what its caption says.

**What a third party needs to reproduce this.** The interpreter is
`more_env` (torch 2.5.1 + CUDA) on Windows 11, one RTX 4060 Laptop 8 GB. Data is
regenerated by `data/script.py`; its output is fingerprinted in
`data/dataset_meta.json` (record counts, per-split hashes, generator seed), and a
regenerated corpus that does not reproduce those hashes is a different dataset and
must be given a new `dataset_version`. The canonical definition lives in
`code/canonical_spec.json`; the guard refuses to stamp `canonical_phase_b` on any
run that deviates from it, which is why no proxy run appears above. Launch is
`python train.py --architecture {moe|mor|more} --seed <s> --run_name phaseB_<arch>`
from `code/`.

**Derived artifacts and the scripts that write them.** `results/results.csv`,
`results/results.json` and `results/results_tables.md` are machine-generated by
`code/export_results.py`; `results/seed_stats.json` by `code/seed_stats.py`;
`runs/*/val_depth_offline.json` by `code/eval_val_depth.py`; the Phase 10 arm
verdicts by `automated/phase10_ablations.py --report-only`; and
`results/depth_null_model.json` by `code/depth_null_model.py`. This document is the
only hand-written file in `results/`, and every figure in it was copied from one of
those. **INTERPRETATION:** the division is deliberate — a number that cannot be
regenerated by naming a script is a number a reader has to take on trust, and none
of the numbers here require that.

**Known reproducibility limits.** `torch.use_deterministic_algorithms(True,
warn_only=True)`: one CUDA histogram kernel used by W&B parameter logging has no
deterministic implementation (§15.12), so bit-exact replay across machines is not
claimed. Seed-to-seed variation is reported everywhere as sample std over the five
frozen seeds, which is the quantity a replication attempt should be compared
against. **The single end-of-phase commit hash is not yet stamped into the generated
result files** — it is deferred to the end of Phase 11 by explicit decision, and
until it lands a reader cannot pin these results to a source revision. That is a
real gap and it is named here rather than in a footnote.

## 17. Claims This Evidence Supports

Each claim below is stated in the form the paper may use, with the number that
carries it and the section to check. A claim absent from this list is not licensed by
this evidence, whether or not it appears elsewhere in the literature.

**On quality.**

1. *On this task at this scale, composing sparse expert routing with adaptive
   recursion did not improve predictive quality over either component alone.* MoRE
   0.063186 ± 0.000249 vs MoE 0.063260 ± 0.000153 (p = 0.6587) and vs MoR
   0.062706 ± 0.000112 (p = 0.0159, MoR better). §6, §14.
2. *One shared recursive block with adaptive halting achieved the lowest validation
   task loss of the three architectures, and the gap to sparse routing is resolvable
   at five seeds.* MoE − MoR = +0.000554, d = 4.13, p = 0.0079 against a floor of
   0.0040. §6.
3. *All three architectures explain a small and nearly equal fraction of held-out
   target variance.* R² = 0.2182 / 0.2250 / 0.2191 against the predict-train-mean
   floor 0.080914. The entire between-architecture spread is ~3% of what any one arm
   explains. §4, §6.
4. *MoRE achieved the lowest training loss and the largest generalization gap of the
   three.* train 0.060237 ± 0.000360, gap 0.002949, vs MoR's gap 0.001205. §7.

**On depth.**

5. *The trained halting policy did not allocate recursion depth better than the best
   input-independent constant-depth policy on held-out data.* Best constant c = 2.00
   scores abs 0.8703; MoR 0.9423 ± 0.0266 (+8.3%), MoRE 0.9940 ± 0.1054 (+14.2%),
   with 5/5 seeds worse in each arm and 10/10 pooled. Holds after matching each arm's
   own mean depth. §9.1.
6. *Recursion depth was near-degenerate: the halting mechanism ran, but it converged
   to a narrow band around two steps.* MoR 2.1283 ± 0.0188 with 97.12% of tokens
   exiting at step 2; MoRE 2.3531 ± 0.1087 with 92.66% at step 2 and 4.88% at step 3;
   forced-exit ≤ 0.0014. §9.
7. *Adding expert routing to recursion measurably increased mean depth.* +0.2248
   steps, d = 2.88, p = 0.0079. This is a real, resolvable effect on the depth
   distribution that did not translate into quality. §9, §15.2.

**On specialization.**

8. *Both routed architectures learned a token-to-expert partition well above chance
   under permutation-invariant measurement, without any routing supervision.*
   Hungarian-matched accuracy 0.5004 ± 0.0560 (MoE) and 0.5258 ± 0.0612 (MoRE)
   against chance 0.1667; AMI 0.5132 / 0.4907; purity 0.5715 / 0.6047, with
   `step_routing` weight fixed at 0.0. §8.
9. *Expert load stayed balanced; no collapse occurred.* Normalized load entropy
   0.9886 / 0.9875, per-expert loads 14.46–19.09%. §8.
10. *Expert parameterizations remained near-orthogonal in the weak sense of low
    pairwise cosine similarity* — mean 0.0227 / 0.0261, max 0.0512 / 0.0733. This is
    *consistent with* differentiated parameterizations and is not evidence of
    orthogonality or of functional specialization. §8.

**On cost.**

11. *Sparse Top-1 dispatch inside a recursive loop was expensive at this scale and
    bought no quality.* MoRE 2225 ± 274 tok/s vs MoE 8485 ± 362 (3.81× slower) and
    MoR 4574 ± 713, while running ~5.8× less FFN arithmetic than MoR — roughly a 12×
    worse arithmetic-to-wall-clock conversion, attributable to per-step gather/scatter
    over six narrow experts. §10.
12. *MoE is exactly MoRE at `max_depth = 1`, with identical parameter tensors, so the
    MoRE − MoE contrast isolates computation rather than capacity.* Totals 3,201,555
    each; FFN weight matrices identical at 3,145,728 across all three arms. §5, §12.

## 18. Claims This Evidence Does Not Support

These are the sentences that would be easy to write from the numbers above and that
the numbers do not license. Each names what is missing, not merely that it is
missing.

1. **"MoRE discovers the intrinsic computational complexity of arithmetic
   operations."** Prohibited on two grounds. The depth targets are a hand-authored
   table in `families.py`, so the ceiling is a design choice, not a property of
   arithmetic; and the measured policy loses to a constant (§9.1), so there is no
   discovery to attribute. The admissible phrasing — *learns to allocate recursion
   depth in accordance with the predefined operation-complexity curriculum* — is
   itself **not supported here**, because §9.1 shows it did not.
2. **"MoRE learns adaptive computation."** The mechanism is present and
   differentiable, gradient reaches the halt parameters, early exit fires on
   99.97% of tokens — and the resulting allocation is beaten by a fixed number.
   *Mechanically adaptive, behaviourally near-constant* is the supportable claim.
3. **"The halting policy allocates depth in a misaligned way"** — i.e. structured but
   wrong. Not separable from unstructured noise by this evidence. Mean absolute error
   is convex, so *any* input-dependent variation uncorrelated with the target raises
   it; losing to a constant rules out **aligned** variation and says nothing about
   which of the two remaining explanations holds. Distinguishing them needs a
   per-operation depth-vs-target correlation, which is not reported here.
4. **"Low cosine similarity proves the experts are orthogonal / functionally
   specialized."** Near-zero cosine between weight matrices is expected at
   initialization and under weight decay; it is a diagnostic, not a specialization
   result. §8's Hungarian/AMI/purity figures are the specialization evidence, and the
   MoRE-vs-MoE cosine difference (max 0.0733 vs 0.0512, +43%) is flagged as
   hypothesis, not finding.
5. **"High expert entropy shows the router specialized."** Normalized entropy 0.988
   is a *load-balance* statistic; it is maximized by a router that ignores its input
   and spreads tokens uniformly. It cannot distinguish specialization from random
   assignment, and the retired primary criterion that used it is void.
6. **"Raw routing accuracy is 0.20 (MoE) / 0.16 (MoRE)."** Structurally
   uninformative: expert indices carry no semantic identity under an unsupervised
   router, and the seed std exceeds the mean. Quote the permutation-invariant
   figures. Likewise **`dispatch/max_load_fraction`** for the recursive arms — it is
   a max over recursion steps, and with 93–97% of tokens gone by step 2 the busiest
   expert trivially owns a near-empty step. §15.7.
7. **"MoRE is worse than MoE."** Not resolved: −0.000074 at p = 0.6587. The
   supportable statement is that the difference is smaller than this design can
   detect, with the detectable floor stated (§14).
8. **"MoR is the better architecture."** Supported *on this task, at 3.2 M
   parameters, `d_model = 256`, one block, one dataset, one split, one scalar
   regression objective, at a shared VALIDATED-STABLE hyperparameter setting that was
   not swept per architecture.* Strip any of those qualifiers and the claim exceeds
   the evidence — in particular, a configuration tuned for a sparse router is not
   neutral ground, and none of the three arms was tuned for itself.
9. **"These results generalize to language modelling / to scale."** No experiment
   here varies scale, task family, or objective. The single capacity probe available
   is a wall-clock-only preliminary sweep with no quality number at any other width
   (§13), and the one non-canonical width run on disk is a 1-epoch proxy that cannot
   enter a table.
10. **"The task is hard and the models learned it well."** Nothing learned it well:
    best R² 0.2250 against a constant-prediction floor. Any architectural conclusion
    drawn here is a conclusion about behaviour in a regime where 78% of target
    variance is unexplained by every arm.
11. **"The routing partition is entirely self-supervised."** The router receives no
    oracle gradient, but the encoder was trained against externally annotated family
    labels at weight 0.5 (§15.8). §11 reports the ablation that bounds this; the
    unqualified claim is not available.

## 19. Final Scientific Assessment

### 19.1 The verdict

`CLAUDE.md` §8 defines three admissible outcomes. **This work lands on Outcome C for
the halting half of MoRE and on Outcome C for the composition claim, while the
routing half lands between B and C.** Stated without hedging:

- **Composition — Outcome C.** MoRE is not better than either component alone. It
  does not separate from MoE (Δ = −0.000074, p = 0.6587) and it is worse than MoR
  (Δ = +0.000481, d = +2.49, p = 0.0159). There is no measured quantity in this work
  on which the combination beats both parents. §6, §15.2.
- **Adaptive depth — Outcome C, on three independent lines of evidence.** The learned
  policy loses to the best constant-depth policy on held-out allocation error, 5/5
  seeds in each recursive arm (§9.1). Forcing maximum depth for every token gives
  *better* task loss at nominally significant p = 0.0476 (§11.3). Weakening the ponder
  cost 10× raises depth without improving loss, which is the pre-registered
  Outcome-C branch of that arm's own reading (§11.4). The mechanism is real —
  differentiable, gradient-carrying, early-exiting on 99.97% of tokens — and its
  behaviour is near-constant.
- **Expert specialization — B-leaning, not clean.** Both routed arms learn a
  token-to-expert partition far above chance under permutation-invariant measurement
  (Hungarian ≈ 0.50–0.53 vs chance 0.167, AMI ≈ 0.49–0.51) with balanced load and no
  collapse, and ~72–80% of it survives removing the only oracle-derived loss term
  (§11.7). That is a real, interpretable, reproducible phenomenon. It is not Outcome B
  as written, because B requires the combination to show something *neither component
  alone gives*, and this partition is present in MoE at equal strength without any
  recursion.

**The one thing MoRE does that neither parent does** is allocate measurably more
recursion depth than MoR (+0.2248 steps, d = 2.88, p = 0.0079) — a resolvable
interaction between routing and halting. It bought nothing: the extra depth came with
worse allocation error and no quality gain. An honest paper reports this as a
mechanism that composes and a benefit that does not.

### 19.2 What is unambiguously true here

Independently of any architectural conclusion, the following are measurements and will
survive a re-analysis: MoE and MoRE are the same parameter tensors at
`max_depth = 1` vs 7, so their contrast isolates computation from capacity (§5, §12);
a Top-1 sparse recursive loop costs 3.81× MoE's wall clock at this scale and dense
evaluate-all is 1.37× *faster* than the sparse dispatch it replaces, so the cost is
dispatch, not arithmetic (§10, §11.1); perfect oracle routing is learnable and makes
predictions worse (§11.2); and every arm explains 21.8–22.5% of held-out target
variance, so the entire between-architecture spread is ~3% of what any one arm
explains (§6, §15.5).

### 19.3 What bounds this conclusion

Three confounds bound **the conclusion**, and none of them is a reason to soften the
verdict above. They are stated as limits on generality, not as reasons MoRE deserved
better.

**Scale.** 3.2 M parameters, `d_model = 256`, one recursive block, one 8 GB laptop
GPU. Every conclusion about *cost* is scale-specific in a way that is measurable
rather than speculative: the dominant term is per-step dispatch overhead over six
1024-wide experts, and that term amortizes as expert width grows. At a width where
FFN arithmetic dominates launch overhead, the 3.81× penalty in §10 would shrink, and
nothing here predicts by how much. Conclusions about *quality* are bounded differently
— doubling the parameter count moved the loss by 0.000072 at p = 0.75 (§11.5), so
within the range tested, capacity was not the binding constraint.

**The dataset.** `more6-v1` is synthetic, generated by `data/script.py`: 16
hand-chosen operations, a hand-authored per-operation depth curriculum, six
hand-assigned expert families, one scalar regression target. Two specific
consequences. The depth claim is scored against an assumption, and §11.3 shows the
task prefers depths the assumption calls wrong (§15.13) — so "the model failed to
match the curriculum" and "the curriculum is not what this task rewards" are both
live, and this evidence does not separate them. And the specialization claim is scored
against a partition the generator wrote down; §11.2 shows that partition is not the
one that minimizes error.

**The evaluation.** One scalar regression loss, one fixed split, five seeds, 50
epochs, and — the most important item — a single shared hyperparameter setting
(`VALIDATED-STABLE`) that was **not** swept per architecture. A configuration
validated on one arm is not neutral ground for three. MoR winning under a shared
setting is a real result at that setting; it is not a claim that MoR is the better
architecture under per-arm tuning, and §18 item 8 states the qualifiers that must
travel with it. Statistical resolution is 0.000131–0.000249 in loss units at n = 5;
differences smaller than that are invisible here and several reported non-results are
non-results *at that resolution* (§14).

### 19.4 What would change the verdict

Concrete and ordered by how much they would move the conclusion, not by cost.

1. **A task whose depth requirement is not hand-authored.** The single largest
   weakness is that adaptive depth is being graded against a table someone wrote. A
   task where required computation is a *derivable* property of the input — iterated
   application until a fixed point, variable-length dependency chains — would make the
   depth claim falsifiable rather than assumption-relative.
2. **Per-architecture hyperparameter sweeps.** Until each of MoE/MoR/MoRE is tuned for
   itself, the ordering in §6 is an ordering at one shared setting. This is the
   cheapest item on the list and it directly gates the strongest comparative claim.
3. **Scale on expert width.** The dispatch-bound regime is identified and quantified
   (§10, §11.1); the question of whether MoRE's cost penalty is architectural or
   incidental to a 1024-wide expert on a laptop GPU is answerable by one width sweep
   with quality measured, which §13 explicitly does not provide.
4. **A per-operation depth-vs-target correlation.** §9.1 rules out *aligned*
   variation in the halting policy but cannot distinguish misaligned structure from
   unstructured noise (§18 item 3). That distinction decides whether the halt head
   learned the wrong thing or nothing, which are different repairs.
5. **A non-regression objective.** A single scalar target with 78% unexplained
   variance is a weak instrument for detecting architectural differences. Any of the
   above at a higher signal-to-noise ratio would resolve gaps this design cannot.

### 19.5 Closing statement

MoRE, as specified and implemented here, is a working architecture that does what its
definition says: tokens are routed Top-1 to independent experts, and the selected
expert's weights are reused for an adaptively halted number of steps, with a real
gradient reaching the halt parameters. Both mechanisms function. On this task, at this
scale, under this evaluation, **neither mechanism earned its cost, and their
combination was not better than either alone.** The specialization behaviour is
genuine and mostly survives removing its only oracle-derived supervision; the adaptive
computation behaviour is mechanically present and behaviourally near-constant, and a
fixed-depth policy is at least as good on quality and no worse on wall clock.

That is Outcome C for the questions this work set out to answer, and the design of the
measurement — the null model, the ablation matrix, the family-wise correction, the
refusal-based export — was built so that a negative result would be visible rather
than absorbable. It was.





















