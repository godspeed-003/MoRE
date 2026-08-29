# PHASE11_AUDIT.md

Blocking audit run before Phase 11 (results export). Covers the seven items of
the standing pre-Phase-7 audit directive. Items 1–2 (routing/halting mechanism)
were discharged by Gates 1–3 and are re-verified here only through the
regression suite.

Audit date: 2026-08-29. Code state: working tree at `f7166b4` **+ uncommitted
changes** (see §6.5). All 15 canonical runs live in `runs/phaseB_{moe,mor,more}_seed4{2..6}__*`.

**Headline: the comparison protocol is sound and the dataset is clean. The
*reading rule* was not.** Every verdict in this repository was produced by
comparing a difference of means against a single arm's standard deviation. That
is not the standard error of a difference, and replacing it with an exact
permutation test changes the conclusion of both Phase 9 and Phase 10. See §6.1
and §7 — this is the finding that matters.

---

## 1. Verdict table

| item | scope | verdict |
|---|---|---|
| 3 | comparison protocol / parameter matching | **PASS** — clean, one confound-free diff, gap 0.120% and it disfavours MoRE |
| 4 | paper metric audit | **2 DEFECTS** — `halt/early_exits` and `train/halting_supervision_loss` report `0.0` as a measurement |
| 5 | dataset & feature leakage | **PASS** — 8/8, including three non-vacuity controls |
| 6 | cross-document contradictions | **3 FOUND** — one load-bearing (checkpoint rule), one methodological (std reading rule), one cosmetic |
| 7 | regression suite | **PASS** — 350/350 checks, Gates 1–5 |

Nothing here blocks Phase 11's exporter from being *written*. Two things block
the numbers it exports: the §4 defects (cosmetic but rule-violating) and the §6.1
reading rule (changes the paper's conclusion).

---

## 2. Method

Every number below was recomputed from `runs/*/metrics.json` and
`runs/*/resolved_config.json` in this audit. Nothing was copied from a prior
report, a changelog entry, or `phase9_matrix_result.json`. Parameter counts were
obtained by re-instantiating `MoREModel` from each run's own recorded
`model` block and summing `named_parameters()`, then checked against the
`provenance.total_params` the run recorded at train time.

---

## 3. Item 3 — comparison protocol and parameter matching

### 3.1 The three arms differ in exactly eight config leaves

Flattening `resolved_config.json` (excluding `provenance`) for the seed-42 run of
each architecture gives 44 fields, of which **36 are bit-identical** and 8 differ:

| field | MoE | MoR | MoRE | why it must differ |
|---|---|---|---|---|
| `architecture` | moe | mor | more | the independent variable |
| `logging.run_name` | phaseB_moe_seed42 | phaseB_mor_seed42 | phaseB_more_seed42 | label only |
| `model.num_experts` | 6 | 1 | 6 | definitional — MoR has no expert set |
| `model.max_depth` | 1 | 7 | 7 | definitional — MoE has no recursion |
| `model.adaptive_halting` | False | True | True | definitional |
| `model.ffn_mult` | 4 | 24 | 4 | **parameter-matching lever**, see 3.2 |
| `loss_weights.halting` | 0.0 | 0.001 | 0.001 | MoE has no ponder cost to weight |
| `loss_weights.routing_balance` | 0.001 | 0.0 | 0.001 | balance over 1 expert is undefined |

Every one of the eight is either the definition of the architecture or a forced
consequence of it. **There is no free hyperparameter differing between the arms** —
`lr`, `weight_decay`, `dropout`, `batch_size`, `epochs`, `d_model`,
`num_blocks`, `step_feat_dim`, `grad_clip`, `subset_fraction`, both data
versions, `loss_weights.task`, `loss_weights.family_cls`,
`loss_weights.step_routing`, `loss_weights.halting_supervision`,
`routing_mode`, `router_noise` and `log_interval` are all identical. This is the
strongest form of the claim the audit directive asked for.

The two zeroed loss weights deserve one note: they are zeroed because the
corresponding *term* is undefined for that architecture, not to tune it. A
non-zero `loss_weights.halting` on MoE would multiply a ponder cost over a
single step, and a non-zero `routing_balance` on MoR would penalise imbalance
across a set of size one.

### 3.2 Parameter counts, fully decomposed

`canonical_spec.json → protocol.parameter_budget_policy` freezes the rule:
*equalise total FFN hidden width, not per-FFN multiplier.* MoR has one FFN where
MoE/MoRE have six, so `ffn_mult` 24 = 4 × 6.

Built from each run's own recorded model block; `built == provenance.total_params`
for all three, so the count in the run record is the count that trained:

| module | MoE | MoR | MoRE | MoE − MoR |
|---|---:|---:|---:|---:|
| `blocks.experts` (all) | 3,153,408 | 3,152,128 | 3,153,408 | **+1,280** |
| `blocks.moe_block` (router) | 1,536 | 256 | 1,536 | **+1,280** |
| `blocks.expert_halt_heads` | 1,542 | 257 | 1,542 | **+1,285** |
| `blocks.layer_norm` | 512 | 512 | 512 | 0 |
| `op_embed` | 4,096 | 4,096 | 4,096 | 0 |
| `step_proj` | 3,840 | 3,840 | 3,840 | 0 |
| `regression_head` | 33,537 | 33,537 | 33,537 | 0 |
| `cls_head` | 1,542 | 1,542 | 1,542 | 0 |
| `step_cls_head` | 1,542 | 1,542 | 1,542 | 0 |
| **TOTAL** | **3,201,555** | **3,197,710** | **3,201,555** | **+3,845** |

Three facts follow, and the paper must state all three.

**(a) MoE and MoRE are parameter-identical to the unit.** Not approximately —
`3,201,555 == 3,201,555`. This is because `adaptive_halting` is not a
constructor argument: the six per-expert halt heads are allocated in MoE too and
simply receive no gradient (`loss_weights.halting = 0.0`, `max_depth = 1`). MoE
therefore carries 1,542 params of dead weight, 0.048% of its total. The
*MoRE-vs-MoE* comparison — which is the one that isolates recursion, since the
two arms differ only in depth and halting — is exactly parameter-matched.

**(b) The MoR gap is 3,845 params = 0.120%, and it decomposes completely.**
The frozen note in `canonical_spec.json` attributes it to "the router plus six
per-expert halt heads". That accounts for 2,565 of it. The remaining **1,280 is a
bias-count difference and the frozen note omits it**: six FFNs contribute
6 × (1024 + 256) = 7,680 bias parameters where one wide FFN contributes
1 × (6144 + 256) = 6,400. It is irreducible by any integer `ffn_mult` — the
weight matrices match exactly (6 × 2 × 256 × 1024 = 2 × 256 × 6144), only the
biases cannot. `canonical_spec.json:protocol.parameter_budget_note` should be
amended to name this third component.

**(c) The gap runs against the result.** MoR has *fewer* parameters and *lower*
loss. MoRE's failure to beat MoR cannot be explained by a capacity deficit,
because MoRE has 0.120% more capacity. State it this way round in the paper; it
converts a caveat into a strengthening.

### 3.3 A vestigial module worth disclosing

MoR's `blocks.moe_block` router is a 256 × 1 projection: a softmax over one
logit is identically 1.0, so those 256 parameters cannot influence any output.
They are harmless (0.008% of the model) and `loss_weights.routing_balance = 0.0`
correctly prevents the balance loss from acting on them, but a reviewer counting
parameters will find them. Same for MoR's single `expert_halt_heads` entry —
that one *is* live and is MoR's ACT head.

---

## 4. Item 4 — paper metric audit

133 metric keys across the 15 canonical runs. Per-architecture applicability
already works for the large majority: MoE emits `"N/A"` for 8 halting/depth keys,
MoR emits `"N/A"` for 15 routing/entropy/cosine keys, MoRE for none (all 133
apply to it). Two keys escaped that layer.

### 4.1 DEFECT — `halt/early_exits` reports 0.0 as a measurement for MoE

MoE seed 42: `halt/early_exit_rate = "N/A"` (correct) but
`halt/early_exits = 0.0` and `halt/forced_exits = 224450.0` (both numbers).
With `max_depth = 1` there is no such thing as an early or a forced exit; the
code classifies all 224,450 single-step tokens as forced. A reader sees "MoE
never exits early", which reads as a finding about MoE's halting behaviour when
MoE has no halting.

Cause: `engine.py:875` gates the whole exit block on `if _tot_exits > 0`, and for
MoE `_tot_exits = 224450 > 0`, so the block populates the counts. The
applicability layer that later stamps `"N/A"` covers `halt/*_rate` but not
`halt/*_exits`. Same shape as the historical defect of guards landing in dead
code — the guard was on the wrong quantity.

Fix: the counts must follow the rates. Gate them on the architecture having
`max_depth > 1`, not on the exit tally being positive.

### 4.2 DEFECT — `train/halting_supervision_loss` reports 0.0 in all 15 runs

Canonical has `model.halting_supervision = False` and
`loss_weights.halting_supervision = 0.0`, so the curriculum term is never
computed. `engine.py:814` divides the accumulator by `n_batches`
unconditionally, yielding `0.0`. That is a placeholder, not a measurement, and
`0.0` for a *loss* invites the reading "the curriculum objective was satisfied
perfectly". Must be `"N/A"` for all three architectures under canonical config.
Note `test_phase3_halting.py:626` and `test_gate5.py:156` both require the key to
*exist* — they must be relaxed to accept `"N/A"`, or they will fail the fix.

### 4.3 NOT defects — three zeros that are real

- `dispatch/overflow_rate = 0.0`, `dispatch/overflow_tokens = 0.0` — true
  measurements. `dispatch/capacity_policy = "no_capacity_limit"` is recorded
  alongside them, so the zero is interpretable.
- `dispatch/router_noise_scale = 0.0` — canonical `router_noise = "none"`; the
  configured value genuinely is zero.

### 4.4 The exporter must map *absent* to N/A for 50 keys

50 keys are **absent** rather than `"N/A"` in the architecture they do not apply
to: 24 per-family routing precision/recall/F1 (raw and Hungarian-matched), 5
`expert_load/expert_{1..5}_pct`, `val/routing_hungarian_assignment` and
`val/routing_collapsed_experts` — all absent for MoR; and
`depth_dist/step_{2..7}_pct` absent for MoE. Absent is honest, but the T11.1
exporter must treat a missing key as `N/A` and never as `0.0`, or MoR will appear
to have zero per-family recall and MoE zero depth beyond step 1.

Note `depth_dist/step_1_pct` *is* present for MoE (100%) and
`expert_load/expert_0_pct` *is* present for MoR (100%) — the absences are
consistently the inapplicable tail, not a random gap.

### 4.5 No run records R², and the depth headline is a *training* statistic

Two scoping problems that are not sentinels but will mislead the paper.

**R² is computable but never computed.** `canonical_spec.json` freezes
`primary_metric_floor = 0.080914` (predict-the-train-mean on the real 5,250-record
val split), but no run joins it. `val/*` contains exactly two keys —
`val/loss` and `val/task_loss` — so the paper would report `0.0632` against no
scale. Against the frozen floor, **all three architectures explain 21.8–22.7% of
target variance**, and the entire between-architecture spread (0.00055) is **3.1%
of the explained variance** (0.0177). The exporter must carry the floor and derive
R², because the size of what is being compared is the most important context in
the results section.

**The depth number everyone quotes is `train/avg_recursion_steps`.** There is no
`val/avg_recursion_steps`. The 2.05-step MoRE figure is a training-time mean, and
dropout is active in training but not in evaluation, so halting behaviour is not
guaranteed identical. Either add a validation-pass depth metric or label the
existing claims "training-time average depth" throughout. The per-operation table
(`recursion/avg_depth_by_op/*`) is a separate evaluation-pass artifact and is the
stronger evidence anyway — see 4.6.

### 4.6 What the depth distribution actually shows (MoRE, seed 42)

Not a defect; recorded here because it is the sharpest negative result in the
project and the audit is where it was first read off the raw metrics.

`depth_dist`: **96.51% of tokens exit at exactly step 2.** Step 1 takes 0.005%,
step 3 2.85%, and steps 4–7 together 0.64%. Per-operation average depth spans
**1.999 (SORT) to 2.112 (AND) — a total spread of 0.11 steps** across sixteen
operations whose target depths in the complexity curriculum span **1–4**, not the
full 1–7. Corrected here: `families.OP_TARGET_DEPTH` is `{1: eight ops, 2: four,
3: two, 4: two}`; the 7 is `model.max_depth`, a configuration ceiling no operation
targets. Target mean **1.875**.

`depth/allocation_error_abs = 1.010` is then fully explained by the skew: a
constant predictor sitting at the target mean gives ≈0.90 mean absolute error on
this table, and since eight of sixteen operations want depth 1, essentially all of
that error is **over-computation on the easy operations**. The failure costs
wasted compute, not accuracy on MEDIAN/SORT.

The honest statement: **the halting mechanism has learned a constant depth of 2,
not an adaptive policy.** It is not broken — T10.J showed depth responds to the
ponder coefficient (2.05 → 2.51 at ×0.1), so gradient does reach the halt head —
but it does not differentiate by operation. Combined with §7, where 2 versus 7
versus 1 makes no difference to loss, the mechanism has no incentive to
differentiate on this benchmark.

---

## 5. Item 5 — dataset and feature leakage

`code/audit_leakage.py`, run against the canonical dataset
(`more6-v1-seed42-n70000-3c1087b6aad9`, 59,500 train / 5,250 val):
**8/8 checks pass.** Three of the eight are non-vacuity controls, which is what
makes the other five worth anything.

| check | result |
|---|---|
| A0 non-vacuity: result mutation actually moved targets | 368/400 targets changed |
| A no step result reaches the input | 400 records mutated, **0** feature changes (pre-fix: all 400) |
| B0 non-vacuity: permutation actually relabelled experts | 400/400 changed `step_experts` |
| B no oracle expert label in input | **0**/400 feature changes (pre-fix slot 0 was `expert_id/(E−1)`) |
| B2 no low-cardinality slot maps 1:1 onto the expert label | offending slots `[]`, majority-class rate 0.303 |
| C input features bit-identical across E = 1 / 5 / 6 | 200 records × 3 expert counts, **0** mismatches |
| D no `(op, args, result)` overlap across splits | train\|val 0, test\|train 0, test\|val 0 |
| E copy baseline is no longer free | copy 0.081853 ≥ predict-mean 0.080914 (pre-fix copy was 0.000000) |

Check C is the `updated_rules.md` §3 input-parity invariant, and it passing at
E = 1 / 5 / 6 is what licenses comparing MoR (E=1) against MoE/MoRE (E=6) at all.

**Residual target–argument coincidence is inherent, not leakage.** In 33.45% of
val records the target equals one of the arguments, because MAX/MIN/MEDIAN/SORT
*return* an argument. Worst single input slot is 16.06%. Pre-fix these were 100%
and 74.15%. Check E is the control that makes this safe: the best single-feature
copy scores 0.081853, *worse* than predicting the train mean, so no copy strategy
is a shortcut.

**The MSE floor for the results section.** predict-zero 0.081853,
predict-train-mean 0.080914, best-single-feature-copy 0.081853, val target
variance 0.080899. The frozen `primary_metric_floor` matches the recomputed
predict-train-mean exactly.

---

## 6. Item 6 — contradictions between the governing documents and practice

Reported, not silently resolved, per the audit directive.

### 6.1 LOAD-BEARING — the reading rule is not a standard error

Every verdict in this repository — Phase 9's matrix, all six Phase 10 arms, and
the three earlier decision drivers — was produced by comparing a **difference of
means** against `max(std_a, std_b)`, the larger of the two arms' per-seed
standard deviations, and calling the difference real at `≥ 2 ×`.

`max(std)` is not the standard error of a difference of means. With n = 5 per arm
the correct denominator is `sqrt(s_a²/n_a + s_b²/n_b)`, which for MoRE−MoR is
0.000122, not the 0.000249 the rule used. The gap 0.000481 is therefore **3.94
standard errors**, not the "1.93 ×" the driver printed. The heuristic was roughly
√n too conservative, and the earlier ddof=0 defect (fixed at T11.0a) was ~1.12×
too liberal, so the net effect was that real differences were being reported as
noise.

`canonical_spec.json → protocol.seed_reporting` freezes a **1 ×** rule ("a
difference smaller than the seed std is not a difference"), and every report used
**2 ×**. Read literally the frozen text states a *necessary* condition, not a
sufficient one, so 2× is a defensible strengthening rather than a violation — but
it is an undeclared threshold that sits exactly where the MoRE−MoR result lives,
and it is the thing that suppressed that result.

**Resolution adopted: stop using a `k × std` heuristic.** With five matched seeds
per arm the exact randomization test is available and assumption-free: enumerate
all C(10,5) = 252 regroupings, count how many give a mean difference at least as
extreme as observed. No threshold to choose, no normality assumption, and it is
what a reviewer will ask for. Results in §7. The `k × std` yardstick is retained
in the drivers only as a *screening* display and must not appear in the paper.

One negative result about power, worth recording: the **paired** sign-flip test
over 2⁵ = 32 sign assignments has a minimum attainable p of 2/32 = 0.0625 and can
therefore *never* reach α = 0.05 at five seeds. Pairing on seed is weakly
motivated here anyway — the train/val split is fixed by file, so a seed varies
only initialization and shuffle order. Use the unpaired permutation test.

### 6.2 LOAD-BEARING — the frozen checkpoint rule is not the rule that was used

`canonical_spec.json → protocol.checkpoint_selection` freezes
**"lowest `val/task_loss` among evaluated epochs"** and states that
`best_val_loss` in `metrics.json` is that minimum. Every report in this
repository instead used the **final-epoch** `val/task_loss`.

The frozen note anticipated exactly this and left an instruction:

> "Measured at T10.H: on all six 20-epoch runs the final epoch WAS the argmin, so
> the rule and 'take the final model' agreed — that agreement is a measurement,
> not a guarantee, **and must be re-checked on the matrix**."

Re-checked here, and **the agreement does not hold on the 50-epoch canonical
matrix.** MoRE seed 42: `val/task_loss` 0.063307 versus `best_val_loss` 0.063097.
All 15 runs diverge. So the repository has been reporting a quantity that is not
its own declared primary metric.

Both are pure `F.mse_loss` on the regression head — this is a *selection*
difference, not contamination. `best_val_loss` is a minimum over 25 validated
epochs, i.e. an order statistic whose downward bias scales with each
architecture's per-epoch validation noise, and that noise is **not matched across
arms** (MoR std 0.000052 versus MoE 0.000194). Selecting on it therefore mixes
quality with curve noise, which is why last-epoch is the better rule.

**But that is a change to a frozen field and is not the auditor's call to make.**
§7 reports the matrix under both rules. The reassuring finding is that the
*conclusion is identical under both* once §6.1 is fixed, so the decision is about
what to declare, not about what is true. Required action: either amend
`protocol.checkpoint_selection` to last-epoch with the order-statistic reasoning,
or keep best-val and re-export every number. Do not leave the field contradicting
practice.

Either way the paper must disclose that **`checkpoint.pt` is not the model whose
metrics are reported** — it is saved at the argmin epoch, while the reported
metrics come from the final epoch.

### 6.3 COSMETIC — three enforced spec fields are absent from `config.json`

`canonical_spec.json:enforced_fields` pins 18 fields. 14 hold the identical value
in both `config.json` defaults and the actual MoRE seed-42 run.
`subset_fraction`, `routing_mode` and `router_noise` are **absent** from
`config.json` and supplied by constructor/loader defaults, which happen to match
the spec (`1.0`, `top1_sparse`, `none`); `halting_mode` is absent from both
because it is *derived* by `resolve_halting_mode()`, which is correct by design.

Behaviour is right, but `resolved_config.json:_README` claims "every field that
`canonical_spec.json:enforced_fields` pins now holds the same value here", and for
three of them the value is not there to hold. Write the three literals into
`config.json` so the claim becomes true, or soften the claim.

`canonical_spec.json` contains **no null fields**, so the proxy guard is armed,
and all 15 canonical runs carry `experiment_group = canonical_phase_b`, which the
guard only grants on an exact spec match.

### 6.4 COSMETIC — two stale references, already known, still open

- `TASKS.md` names the T8.0b driver `automated/routing_supervision.py`; the file
  is `automated/routing_supervision_decision.py`.
- The `config_hash()` docstring refers to a "results exporter" that does not
  exist as a separate module — `code/run_phase9_matrix.py` is it, until T11.1
  creates `code/export_results.py`.
- The T10.J pre-registration block in `automated/phase10_ablations.py` (~line 85)
  still reads "T10.B found fixed depth BETTER than learned halting at 2.10x seed
  std". It is a pre-registration and must not be rewritten, but it needs a
  bracketed `[CORRECTED: 1.88x after the T11.0a ddof fix; p = 0.125 under the
  exact test — not significant]` annotation.

### 6.5 DISCLOSURE — all 15 canonical runs are `code_git_dirty: true`

Every canonical run records `code_git_commit = f7166b4` **and**
`code_git_dirty = true`, so the commit does not pin the code that trained them.
The 18 Phase 10 runs share one code state with each other (verified by the driver's
fingerprint check), and the canonical 15 share one with each other, but neither is
recoverable from git history alone. The paper must either commit the current tree
and note that the recorded hash is a parent, or ship
`automated/phase10_code_fingerprint.json`-style content hashes for the canonical
matrix too. **This is the weakest link in the provenance chain.**

Not a defect: `num_experts`, `max_depth`, `num_blocks`, `d_model`, `lr`,
`weight_decay` and `routing_balance` are absent from the `provenance` sub-block
but *are* recorded — at the top level of the same `resolved_config.json`, and in
W&B, because `engine.py:277` builds `wandb_config = {**mc, **tc, **lw, **dc}`
before adding the provenance fields. `updated_rules.md` §9 asks that the W&B run
record them, and it does. An earlier audit note calling these "provenance gaps"
was wrong.

---

## 7. Item 7 — regression suite, and the corrected results

`code/run_correctness_suite.py`: **350/350 checks pass**, Gates 1–5
(19 + 31 + 27 + 130 + 143), 152.8 s. No failures, no skips.

### 7.1 The canonical matrix under an exact test

n = 5 per arm (seeds 42–46), exact two-sided randomization test over all 252
regroupings, Cohen's d on the pooled SD. Floor = 0.080914.

**Last-epoch `val/task_loss`** (what every report used):

| arch | mean | sample std | R² vs floor |
|---|---:|---:|---:|
| MoE | 0.063260 | 0.000153 | 0.2182 |
| MoR | **0.062706** | 0.000112 | **0.2250** |
| MoRE | 0.063186 | 0.000249 | 0.2191 |

| pair | gap | Cohen d | exact p |
|---|---:|---:|---:|
| MoR − MoE | −0.000554 | −4.13 | **0.0079** |
| MoR − MoRE | −0.000481 | −2.49 | **0.0159** |
| MoRE − MoE | −0.000074 | −0.36 | 0.659 |

**Best-val `best_val_loss`** (what `canonical_spec.json` freezes):

| arch | mean | sample std | R² vs floor |
|---|---:|---:|---:|
| MoE | 0.062819 | 0.000194 | 0.2236 |
| MoR | **0.062527** | 0.000052 | **0.2272** |
| MoRE | 0.062892 | 0.000216 | 0.2227 |

| pair | gap | Cohen d | exact p |
|---|---:|---:|---:|
| MoR − MoE | −0.000292 | −2.06 | **0.0397** |
| MoR − MoRE | −0.000365 | −2.32 | **0.0079** |
| MoRE − MoE | +0.000072 | +0.35 | 0.611 |

**The qualitative conclusion is identical under both selection rules, which is
what makes §6.2 a declaration problem rather than a scientific one:**

1. **MoR is significantly better than both MoE and MoRE** (p ≤ 0.04 in all four
   comparisons; p ≤ 0.016 under the last-epoch rule).
2. **MoRE and MoE are statistically indistinguishable** (p ≈ 0.6 both ways) —
   and they are parameter-identical to the unit, so this is the clean isolation of
   recursion, and recursion contributes nothing.
3. Therefore **adding expert routing to recursion recovers nothing that recursion
   alone gives, and costs relative to recursion alone.** Under Bonferroni across
   the three pairs (α = 0.0167) the last-epoch MoR−MoE and best-val MoR−MoRE
   results survive; the other two significant cells sit just outside.

Effect sizes are large (|d| 2.1–4.1) but the magnitudes are small in absolute
terms: MoR's advantage over MoRE is 0.00048, which is **2.7% of the explained
variance**. Real, reproducible across five seeds, and small — all three
statements belong in the paper.

### 7.2 Phase 10 arms under an exact test

n = 3 per arm against the 5-seed canonical MoRE baseline; C(8,3) = 56
regroupings, so the **smallest attainable p is 1/56 = 0.0179**.

| arm | n | mean | gap | Cohen d | exact p |
|---|---:|---:|---:|---:|---:|
| dense_routing (`routing_mode = dense_blend`) | 3 | 0.063687 | +0.000500 | +1.43 | **0.0179** |
| routing_supervision (`step_routing = 0.5`) | 3 | 0.063772 | +0.000585 | +2.77 | **0.0179** |
| fixed_depth (`fixed_depth = True`) | 3 | 0.062822 | −0.000364 | −1.90 | 0.125 |
| ponder_cost_low (`halting = 0.0001`) | 3 | 0.063630 | +0.000444 | +1.17 | 0.161 |
| router_noise (`fixed_annealed`) | 3 | 0.063363 | +0.000176 | +0.53 | 0.411 |
| two_blocks (`num_blocks = 2`) | 3 | 0.063258 | +0.000072 | +0.21 | 0.750 |

**Two arms resolve that the `2 × std` rule had called noise:**

- **dense_routing is worse than Top-1 sparse** (+0.000500, p = 0.0179). This
  supports the canonical architectural choice on quality grounds, in addition to
  the 6× dispatch cost (`dispatch/evals_per_token` 6.0 versus 1.0).
- **routing supervision hurts** (+0.000585, d = 2.77, p = 0.0179). Driving the
  router to *perfect* family assignment — Hungarian accuracy, AMI, purity and
  matched macro-recall all exactly 1.0000 ± 0.0000, a ~12σ move on routing
  quality — makes task loss significantly **worse**. The earlier retraction of
  "perfect routing hurts" to "perfect routing buys nothing" was itself an
  artifact of the reading rule and is withdrawn: the stronger claim holds.

Both sit exactly at the design's resolution floor (1/56), so neither survives
Bonferroni across six arms (α = 0.0083). **The n = 3 design cannot support a
multiplicity-corrected claim at all.** Adding seeds 45 and 46 to these two arms
(4 runs, ≈ 2.8 h) drops the floor to 1/252 = 0.004 and makes both findings
multiplicity-robust. That is the highest-value remaining GPU spend in the project.

---

## 8. Required actions before the T11.1 exporter emits paper numbers

| # | action | blocking? |
|---|---|---|
| 1 | Resolve §6.2: amend `protocol.checkpoint_selection`, or re-export on best-val | **yes** — the spec currently contradicts every number |
| 2 | Replace the `k × std` verdict layer with the exact permutation test (§6.1) in the exporter; keep `k × std` as screening only | **yes** — it changed four conclusions |
| 3 | Fix `halt/early_exits`/`halt/forced_exits` to N/A when `max_depth == 1` (§4.1), and relax the two tests that require the key to exist | yes, cosmetic but Rule 4 |
| 4 | Fix `train/halting_supervision_loss` to N/A when the term is not computed (§4.2) | yes, cosmetic but Rule 4 |
| 5 | Exporter maps absent → `N/A` for the 50 keys in §4.4; never `0.0` | **yes** |
| 6 | Exporter carries `primary_metric_floor` and derives R² (§4.5) | **yes** — the numbers are uninterpretable without it |
| 7 | Label depth claims "training-time", or add a validation-pass depth metric (§4.5) | yes |
| 8 | Amend `parameter_budget_note` with the 1,280-param bias component (§3.2b) | no |
| 9 | Commit the tree or ship content hashes for the canonical 15 (§6.5) | **yes** — provenance |
| 10 | Write the three absent literals into `config.json` (§6.3) | no |
| 11 | Annotate the stale T10.J pre-registration text (§6.4) | no |
| 12 | Optional: seeds 45–46 on dense_routing and routing_supervision (§7.2), ≈2.8 h | no — strengthens two claims |

Items 1, 2, 5, 6 and 9 must land before any number leaves the exporter. Items 3,
4 and 7 must land before the metric table is typeset. Nothing here requires a
change to the model, the data, or the training loop: **no measured quantity in
this audit was found to be wrong — only the way differences were read, and the
way inapplicable metrics were rendered.**

