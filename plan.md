# MoRE — Finalization Plan After Repository Audit

**Audience:** Claude Code / senior research coding agent

**Purpose:** This is the immediate execution plan after the deep repository audit. It supersedes the older exploratory `rules.md` / `objective.md` documents and operationalizes the corrected scientific protocol. The audit found that the previous empirical results are not safe to use as paper evidence because several core mechanisms were broken or confounded. Do not start another large training matrix until the gates in this document pass.

---

# 0. Read This First

The project is studying:

- **MoE = Mixture of Experts:** sparse learned Top-1 routing among independent experts.
- **MoR = Mixture of Recursion:** reuse of one shared computational block through multiple recursion steps with adaptive ACT-style halting.
- **MoRE = Mixture of Recursive Experts:** the proposed combination: route a token to a specialized expert and recursively reuse that expert/block for an adaptive number of steps.

The objective is NOT “make MoRE win validation loss by any means.” The objective is to determine whether the combined architecture can be implemented correctly and whether it exhibits a meaningful combination of expert specialization and adaptive computation.

The deep audit found six blocking defects:

1. target leakage — the answer is present in the input;
2. halting has no gradient;
3. canonical routing is dense rather than sparse Top-1;
4. the balance objective dominates task loss and scales with recursion depth;
5. the six-expert configuration is internally hard-coded to seven classes in multiple places;
6. MoR and MoRE receive different feature distributions and the routing oracle is directly encoded in the input.

It also found provenance, seeding, metric, and W&B namespace defects.

Therefore **all old loss-based results are research-history only** until the corrected pipeline passes the gates below. Do not use old numbers to justify a claim about arithmetic ability, learned halting, or MoRE superiority.

---

# 1. Governing Documents

Create the following  documents:

built a skill.md and/or claude.md based on the following rules and objectives. make sure that is used always. give user instructions if you need to copy it to .claude folder and can't access it yourself.

Use the new updated `rules.md` and `objective.md` supplied with this handoff as the authoritative versions.

The older documents are considered superseded because they placed expert entropy ahead of predictive correctness and allowed proxy-oriented optimization behavior. The audit confirmed that this was unsafe: the old implementation could maximize entropy while producing a near-uninformative router.

Do not use the old objective “expert_entropy > baseline” as the primary success criterion. follow these new plan and objective.

fyi already updated the files and these are now named as updated_rules.md
updated_objective.md
plan.md
---

# 2. Phase 0 — Quarantine the Invalid Experimental Record

Before touching the architecture, preserve the evidence but prevent accidental reuse.

## 2.1 Archive

Create an archive location such as:

```text
archive/pre_finalization/
```

Move/copy historical result artifacts there with an explicit header:

```text
INVALIDATED FOR SCIENTIFIC COMPARISON
Reason: confirmed target leakage and/or broken halting/routing implementation.
```

Keep the history. Do not delete evidence.

this is also already done by the user so consider this comleted.

## 2.2 Run directories

Stop writing these globally:

```text
best_model.pt
results.tsv
final_run_metrics.json
1block_results.csv
2block_results.csv
```

Use:

```text
runs/<experiment_id>/
```

with:

```text
config.json
resolved_config.json
metrics.json
results.tsv
checkpoint.pt
stdout.log
```

## 2.3 Disable autonomous canonical experimentation

Do not let an autoresearch loop mutate configs and silently launch canonical runs.

Proxy search may remain on a separate exploratory path, but the canonical matrix must be explicitly specified and reproducible.

### Gate 0

Do not continue until:

- historical results are clearly marked invalidated;
- canonical output paths are per-run;
- final experiments cannot silently use proxy settings.

---

fyi this is already done as we aren't using the autoresearch file but keep it albeit in a seperate folder like one named automated.

# 3. Phase 1 — Repair the Task Before Repairing the Model

This is the highest priority.

## 3.1 Remove target leakage

The audit established that the generator guarantees:

```text
final step result == regression target
```

and the tokenizer writes every step result into the input features.

This makes the current task a target-copy task rather than an arithmetic-computation task.

### Required change

The canonical tokenizer MUST NOT feed step results to the model.

At minimum:

```python
numeric_vals = flat_args
```

rather than:

```python
flat_args + [scalar_result]
```

Also ensure the final-step input does not contain the answer indirectly through another slot.

Do not invent a second target leakage path while removing the first.

## 3.2 Do not encode the oracle family in the input

The current encoding contains:

```python
row[0] = expert_id / (...)
```

This gives the router the answer to the routing problem.

Remove it.

Represent the operation identity using a legitimate operation representation independent of `num_experts`, e.g. an operation embedding or one-hot operation token encoding.

The operation token is task input. The expert-family label is an oracle routing target.

They are not the same thing.

## 3.3 Input parity test

For one fixed record, tokenize it with:

```text
E=1
E=6
E=5
```

and assert the feature tensor is identical.

`num_experts` must not change the semantics or scale of model input.

## 3.4 Canonical dataset manifest

Create/update:

```text
dataset_meta.json
```

with:

```text
dataset_version
generator commit/identifier
seed
train count
validation count
test count
family counts
operation counts
target-depth counts
normalization range
input feature definition
SHA-256 of each split
```

The current dataset metadata reports 70,000 records split 59,500/5,250/5,250 with seed 42. The family weights currently contain an E7=0.40 component, so the regenerated six-expert dataset must explicitly redistribute/redefine that mass rather than silently dropping it. fileciteturn7file2L2-L19

Document the redistribution rule.

## 3.5 Leakage audit

Create a hard-fail audit that checks every split for:

- target present in any input feature;
- ID overlap;
- exact input-expression overlap;
- duplicate operation/argument/result records when relevant.

Also calculate:

```text
MSE(predict zero)
MSE(predict train mean)
MSE(best single input feature)
```

The learned task must beat the trivial baselines for the claim “the model learned the task” to be meaningful.

### Gate 1

Do not proceed until all of these pass:

- no target leakage;
- no oracle family label in input;
- train/val/test leakage audit passes;
- feature representation is identical across model variants;
- manifest hashes match files;
- trivial baselines are logged;
- the task cannot be solved by copying a single feature.

---

# 4. Phase 2 — Correct the MoE Mechanism

## 4.1 Implement true Top-1 sparse routing

The audit found that the current implementation computes every expert and takes a dense weighted sum. That violates the intended MoE definition and makes the “routing” argmax mainly diagnostic.

Canonical route must be:

```text
router logits
 -> softmax
 -> top-1 expert
 -> selected expert only
 -> selected gate probability multiplies selected expert output
```

Use the implementation that preserves a real gradient path into the router.

A dense implementation may remain only as:

```text
MoRE – Dense Routing Ablation
```

## 4.2 Route output and routing metrics must agree

The expert that determines the forward computation must be the expert whose index is logged and evaluated.

Do not score an argmax that does not control dispatch.

## 4.3 Capacity and overflow

If capacity limiting is used, explicitly measure:

```text
capacity factor
number of dropped/overflow tokens
fraction of tokens receiving residual fallback
```

Do not silently discard tokens.

If the controlled dataset distribution does not require capacity limiting, prefer a simple implementation that is easy to verify.

---

# 5. Phase 3 — Correct the Recursion / Halting Mechanism

## 5.1 The audit found halting is currently not trained

The current “halting loss” is based on a boolean mask converted to float. It has no gradient path to the halt head.

This means prior adaptive-depth plots cannot be treated as learned halting evidence.

## 5.2 Implement genuine ACT-style halting

The canonical recursion must include:

```text
halt probability p_t
 -> cumulative halting mass
 -> remainder / halting probability
 -> differentiable ponder cost
 -> frozen state after halt
```

The exact standard ACT formulation may be used; do not invent a novel halting equation.

The dispatch/evaluation threshold can remain discrete, but training must include a differentiable path to the halt head.

## 5.3 Ponder cost

Return a real recursion/ponder cost.

This must:

- receive gradients;
- discourage unconditional max-depth recursion;
- be normalized so its magnitude is interpretable;
- be logged separately from task loss.

## 5.4 Operation-complexity curriculum

If the canonical experiment uses per-operation target depths, implement that as explicit `halting_supervision_loss`.

Keep it separate from ponder cost:

```text
halting_supervision_loss
ponder_cost
```

Do not collapse them into one opaque number.

## 5.5 Acceptance test

One backward pass must prove:

```text
for every halt-head parameter:
    grad is not None
    abs(grad).sum() > 0
```

Also run a tiny synthetic case where the target depth is known and verify that the halt probabilities move in the expected direction after one optimizer step.

### Gate 2

No canonical training until real gradients reach the halting mechanism.

---

# 6. Phase 4 — Fix the Balance Objective

## 6.1 Normalize over block/depth calls

The audit found that balance loss was summed across recursion depths and blocks, so deeper models receive proportionally larger balance rewards/penalties.

This makes fixed-depth and block-count comparisons mathematically confounded.

Replace:

```text
sum over all (block, depth)
```

with:

```text
mean over all valid (block, depth) calls
```

or another explicitly depth-invariant normalization.

## 6.2 Weight selection

Do not blindly use the historical `routing_balance=0.05`.

First measure:

```text
task_loss magnitude
balance_loss magnitude
ponder magnitude
```

after warmup.

Then select a routing-balance coefficient whose contribution is clearly subordinate to task learning rather than ~146× larger.

The exact final coefficient must be documented and frozen before the canonical matrix.

## 6.3 Gradient-direction tests

Synthetic tests must verify:

```text
collapsed routing -> gradient away from collapse
uniform routing -> balance objective has the expected optimum
balance(depth=1) ≈ balance(depth=5)
```

### Gate 3

Do not begin hyperparameter search until the balance objective passes these tests.

---

# 7. Phase 5 — Remove Configuration Landmines

## 7.1 Make `num_experts` genuinely dynamic

Every head, confusion matrix, classifier, manifest lookup, and tensor reshape must derive dimensions from `num_experts`.

The current code has hard-coded seven-class structures even when configured for six.

Fix all of them.

## 7.2 Remove E7 fallback

There is no canonical E7 in the final six-family operation-level setup.

An unknown operation must raise an explicit error.

## 7.3 Config precedence

There must be one source of truth for:

```text
subset_fraction
seed
epochs
batch_size
```

Do not silently overwrite data configuration with backward-compatibility shims.

A command-line `--epochs` override must be written into the resolved W&B config.

### Gate 4

A smoke test must instantiate E=1, E=6, and E=7 and verify all architecture heads/metrics are dimensionally consistent.

---

# 8. Phase 6 — Rebuild Reproducibility and Metrics

## 8.1 Global seeding

Implement a real seed function covering:

```text
random
numpy
torch
torch.cuda
DataLoader generator/worker
```

Use the canonical seed set:

```text
42, 43, 44, 45, 46
```

Determinism should be enabled where compatible with the chosen GPU operations. If strict deterministic execution conflicts with a necessary kernel, document the exception instead of pretending exact determinism.

## 8.2 Routing accuracy

There is one authoritative implementation.

Calculate direct accuracy and confusion-matrix diagonal accuracy from the same token arrays and assert equality.

## 8.3 Permutation-invariant routing

Because expert indices are exchangeable, report:

- raw routing accuracy;
- Hungarian-matched accuracy;
- AMI;
- cluster purity.

This is particularly important if the router learns a consistent partition whose expert numbering does not match the manually named oracle families.

## 8.4 Expert entropy

Report:

```text
raw entropy
normalized entropy
per-expert load percentages
```

Do not use entropy as evidence of semantic specialization without routing/cluster evidence.

## 8.5 Cosine similarity

Report:

```text
mean pairwise cosine
max pairwise cosine
```

Return `N/A` for one-expert MoR rather than `-1`.

## 8.6 W&B naming

Never use `/` inside family labels.

Use:

```text
E1_ADD_SUB
E2_MULT_DIV
E3_MOD_POW
E4_LOGIC
E5_SHIFT
E6_SORT_STAT
```

## 8.7 Provenance

Every canonical W&B run gets:

```text
experiment_id
experiment_group
architecture
variant
seed
dataset_version
code_git_commit
config_hash
resolved_epochs
resolved_subset_fraction
all critical model/loss settings
```

### Gate 5

A two-epoch smoke run must produce:

- valid provenance;
- valid metrics;
- correct dimensions;
- routing assertion pass;
- halt-gradient assertion pass;
- no sentinel metrics.

---

# 9. Phase 7 — Build a Single Canonical Training Pipeline

Do not maintain separate long-term implementations such as:

```text
train.py
run_more_baseline.py
run_moe.py
run_mor.py
```

Instead use one training system with an explicit architecture mode:

```text
architecture = moe | mor | more
```

and one canonical metric/export pipeline.

The exact same dataset tokenizer and input representation must be used for all three.

Architecture-specific changes are derived only from the model definition.

---

# 10. Phase 8 — Canonical Configuration

Before the large runs, create one resolved canonical configuration.

Suggested starting point from the previously successful compute protocol:

```text
batch_size = 768
50 epochs
full canonical dataset
one recursive MoRE block initially
6 experts for MoE/MoRE
one expert for MoR
same d_model/FFN base unless parameter-matched run is explicitly requested
```

Do not copy old hyperparameters blindly. The corrected Top-1 routing, ACT halting, leakage-free task, and normalized balance loss change the optimization regime.

Perform only a small, controlled smoke calibration after correctness gates. Do not launch broad autoresearch.

---

# 11. Phase 9 — Primary Scientific Matrix

Once Gates 0–5 pass:

## 11.1 Primary comparison

Run five seeds each:

```text
MoE
MoR
MoRE
```

using the same:

```text
dataset
batch size
optimizer
learning rate
weight decay
epochs
input representation
seed set
```

Primary configurations:

| Architecture | Experts | Max depth | Halting | Routing |
|---|---:|---:|---|---|
| MoE | 6 | 1 | fixed | learned Top-1 |
| MoR | 1 | canonical | adaptive ACT | N/A |
| MoRE | 6 | canonical | adaptive ACT | learned Top-1 |

## 11.2 Parameter-matched MoR

Run a secondary five-seed MoR configuration whose trainable parameter count is within 5% of MoRE.

Report it separately.

This avoids the criticism that the basic MoR baseline is simply much smaller.

---

# 12. Phase 10 — Ablations

Each ablation must change one declared factor.

## A. Oracle routing

```text
MoRE learned router
vs
MoRE oracle expert dispatch
```

Purpose:

> estimate the performance upper bound of expert assignment.

This is NOT the headline model.

## B. Fixed-depth recursion

Compare:

```text
fixed depth 1
fixed depth 3
fixed depth canonical maximum
adaptive depth
```

Purpose:

> isolate whether adaptive halting provides useful compute allocation.

## C. One vs two recursive blocks

Only the number of blocks changes.

Do not simultaneously change max depth, routing supervision, or loss weights.

## D. Router noise

```text
none
fixed annealed
trainable
```

Only if the canonical no-noise model needs this diagnostic.

## E. Routing supervision

```text
supervision OFF = canonical
supervision ON = explicitly labeled oracle-routing ablation
```

Do not mix the two in the same headline table.

## F. Dense vs Top-1

Dense routing is an implementation ablation only.

The canonical model remains sparse Top-1.

---

# 13. Phase 11 — Secondary Studies

## 13.1 Batch size sweep

Keep it supplementary.

Measure:

- throughput;
- wall-clock time;
- peak memory;
- validation loss.

Do not tune different architectures with different batch sizes for the primary comparison.

## 13.2 Capacity / model-size sweep

If time permits, run Nano/Micro/Mini with the corrected implementation.

Call it:

> preliminary capacity sweep

not:

> scaling law

unless enough model/data scales are actually trained to support that claim.

---

# 14. Results Exporter

Create:

```text
code/export_results.py
```

It must select only:

```text
experiment_group == canonical_phase_b
AND dataset_version == CANONICAL
```

and refuse to produce a headline table if:

- config hashes differ unexpectedly;
- dataset versions differ;
- a run lacks a seed;
- epoch count is wrong;
- subset fraction is wrong;
- required metrics are missing.

Output:

```text
results.csv
results.json
results.md
```

Never fabricate missing metrics. Use `N/A`.

---

# 15. Required `results.md` Content

Claude must generate `results.md` with this exact high-level organization:

```text
# MoRE Canonical Research Results

## 1. Executive Summary
## 2. Research Question and Architecture Definition
## 3. Canonical Configuration
## 4. Dataset and Leakage Audit
## 5. Parameter Counts
## 6. Main MoE / MoR / MoRE Results
## 7. Training Dynamics
## 8. Expert Routing and Specialization
## 9. Adaptive Recursion / Halting
## 10. Compute and Throughput
## 11. Ablation Studies
## 12. Parameter-Matched Comparison
## 13. Preliminary Capacity Sweep
## 14. Statistical Stability
## 15. Failure Cases and Negative Results
## 16. Reproducibility / Run Manifest
## 17. Paper-Safe Claims
## 18. Unsupported Claims
## 19. Final Scientific Assessment
```

Every numerical result must include source run IDs or source artifact paths.

The summary must distinguish:

```text
measurement
interpretation
hypothesis
```

Do not turn a correlation into a causal claim.

---

# 16. Paper-Safe Interpretation Framework

The paper should ultimately evaluate three possible stories.

## Story A — Strong

MoRE approaches or improves predictive quality while retaining sparse expert specialization and using adaptive depth to reduce computation.

## Story B — Exploratory but useful

MoRE does not beat MoE on raw loss but demonstrates that expert specialization and adaptive recursion can be learned jointly, with measurable and reproducible operation-dependent compute allocation.

## Story C — Negative / falsifying

After correcting leakage, halting, routing, and objective issues, MoRE fails to produce meaningful adaptive-computation or compute/accuracy benefits.

If Story C occurs, report it honestly. Do not tune until it becomes Story A.

---

# 17. Claims That Are Explicitly Forbidden Unless New Evidence Supports Them

Do NOT write:

- “MoRE beats MoE” unless the corrected primary results establish it.
- “MoRE discovers intrinsic computational complexity” — the current depth target is a designed supervision curriculum.
- “High entropy proves specialization.”
- “Low cosine similarity proves orthogonality.”
- “Adaptive computation is unsupervised” if target-depth supervision is enabled.
- “MoRE scales to billions of parameters” from a tiny POC.
- “MoRE is more efficient” based only on one hardware throughput measurement.
- “MoR collapses” when E=1 — a one-expert model cannot collapse across experts.
- Any claim supported only by historical pre-fix runs.

---

# 18. Researcher Decision Gate

At the end, `results.md` must explicitly answer:

1. Did the corrected task require genuine computation rather than copying?
2. Did Top-1 routing actually determine expert execution?
3. Did the halt heads receive gradient and learn?
4. Did the balance objective avoid dominating task optimization?
5. Did expert routing remain non-collapsed?
6. Did the router learn meaningful specialization beyond chance?
7. Did recursion depth track the predefined operation-complexity curriculum?
8. Did MoRE differ meaningfully from MoE-only and MoR-only?
9. Did the adaptive mechanism reduce compute at acceptable predictive cost?
10. Are the results stable across five seeds?
11. Are MoR comparisons parameter-fair?
12. Is there enough evidence for a NeurIPS workshop submission?

The final answer must be evidence-based, and a “No” answer is acceptable.

---

# 19. Immediate Execution Order — Do This Now

Claude should execute exactly this order:

### Step 1
Archive/quarantine old invalid results.

### Step 2
Fix dataset target leakage and remove oracle family ID from model features.

### Step 3
Regenerate canonical six-expert dataset and manifest.

### Step 4
Fix `num_experts` generality and all hard-coded seven-class structures.

### Step 5
Implement true sparse Top-1 dispatch with selected gate multiplier.

### Step 6
Implement real ACT-style halting with ponder cost and gradient flow.

### Step 7
Normalize balance loss across depth/block calls and verify gradient direction.

### Step 8
Disable router noise in the canonical configuration; keep alternatives as ablations.

### Step 9
Implement global seed/provenance/output-directory system.

### Step 10
Build and run the full correctness suite.

### Step 11
Run only a 1–2 epoch smoke test on each of:

```text
MoE
MoR
MoRE
```

and verify all metrics manually.

### Step 12
Freeze the corrected canonical configuration.

### Step 13
Run five seeds for the three primary architectures.

### Step 14
Run parameter-matched MoR.

### Step 15
Run the required ablations.

### Step 16
Run the exporter and generate `results.md`.

### Step 17
Perform a human scientific audit of `results.md` before using any number in the paper.

---

# 20. Final Instruction to the Coding Agent

Do not optimize for a positive paper conclusion.

Your job is to make the experiment correct enough that the result can be trusted.

Do not silently reinterpret an existing number to rescue it.

When a test fails:

```text
STOP
REPORT THE FAILURE
IDENTIFY THE CAUSE
FIX IT
RE-RUN THE GATE
```

When a result gets worse after a correctness fix, do not revert the fix simply because the old result looked better.

The scientific value of this project comes from determining whether MoRE works **after the architecture is implemented as actually intended**.

The canonical experiment begins only after all correctness gates pass.
