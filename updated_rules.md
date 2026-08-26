# SYSTEM DIRECTIVE: MoRE (Mixture of Recursive Experts) — Final Research Rules

## 0. Purpose and Terminology

You are an AI research coding assistant working on a controlled proof-of-concept for **MoRE: Mixture of Recursive Experts**.

Use these definitions throughout the repository:

- **MoE (Mixture of Experts):** multiple independent expert networks with a learned router selecting an expert for each token. In this micro-PoC the canonical routing mechanism is **Top-1 sparse routing**.
- **MoR (Mixture of Recursion):** one shared computational block is reused across recursion steps/depth, with adaptive halting determining how many times a token is processed.
- **MoRE (Mixture of Recursive Experts):** the proposed combination of the two: a token is routed to a specialized expert and that expert/block is recursively reused for an adaptive number of steps.

The research question is whether **expert specialization (MoE axis)** and **adaptive recursive computation (MoR axis)** can be learned jointly in one architecture.

This repository is a scientific experiment, not an open-ended product optimization project. Preserve the mathematical meaning of MoE, MoR, and MoRE while making the implementation correct, reproducible, and auditable.

---

## 1. Non-Negotiable MoE Constraints

### 1.1 Routing

The canonical MoE and MoRE routing path MUST use standard **Top-1 sparse routing**:

```text
x
 -> Linear router
 -> Softmax probabilities
 -> argmax / top-1 expert index
 -> dispatch ONLY to that expert
 -> optional multiplication by the selected gate probability
```

Do NOT replace this with dense evaluation of every expert and a weighted sum in the canonical configuration.

A dense routing implementation may exist only as an explicitly labeled ablation such as:

```text
MoRE – Dense Routing Ablation
```

It must never silently become the canonical path.

### 1.2 Expert independence

Experts are independent FFNs. No cross-expert communication or expert-specific skip connections may be introduced unless explicitly defined as a labeled ablation.

### 1.3 Load balancing

A standard auxiliary load-balancing objective is required for the canonical sparse router. Its role is to discourage pathological expert collapse, not to become the primary optimization objective.

The balance objective MUST be normalized so that its magnitude does not grow merely because the model executes more recursion steps or contains more blocks.

### 1.4 Balance-loss semantics

The implementation must make the direction of optimization explicit and tested:

- collapsed routing: low entropy, poor load balance;
- approximately uniform routing: high entropy, healthy utilization;
- the gradient must push a collapsed router away from collapse;
- balance loss must be approximately depth-invariant after normalization.

Report the entropy term and Switch-style auxiliary term separately.

Do NOT tune the system toward maximal entropy blindly. High entropy is evidence of balanced utilization, not proof of specialization or useful routing.

---

## 2. Non-Negotiable MoR / Recursion Constraints

### 2.1 Weight sharing

The same recursive block parameters MUST be reused at every recursion depth. Do not instantiate separate expert weights for each depth.

Depth is computation through time, not a stack of independent depth-specific expert networks.

### 2.2 Halting

The canonical recursion mechanism must use a differentiable ACT/Universal-Transformer-style halting formulation.

Requirements:

- a learned halting probability is produced at each recursion step;
- active tokens can leave the recursion early;
- halted states are frozen and are not recursively recomputed;
- tokens that reach maximum depth are forced to exit;
- a differentiable ponder/recursion cost exists;
- the task path or halting supervision must provide a real gradient to the halting parameters.

A boolean threshold may be used for dispatch decisions, but the training objective must retain a differentiable path to the halt head.

### 2.3 Halting supervision

For this controlled mathematical POC, the dataset contains an explicit **operation-complexity depth curriculum**. This may be used as a supervised halting ablation and, if enabled in the canonical configuration, must be reported explicitly.

Do NOT claim that the model discovers intrinsic computational complexity without qualification.

Use precise wording such as:

> The model learns to allocate recursion depth in accordance with the predefined operation-complexity curriculum.

If unsupervised ACT is tested, label it as a separate ablation.

---

## 3. MoRE Architecture Invariant

Canonical MoRE combines the two axes:

```text
                 MoE axis
                     |
input step -> router -> specialized expert
                              |
                              v
                         recursive block
                              |
                    halt / continue decision
                              |
                              v
                        next recursion step
```

The same expert-block weights are reused at every recursion depth.

Changing expert identity and changing recursion depth are conceptually separate mechanisms:

- MoE answers **which computation module?**
- MoR answers **how much computation?**
- MoRE jointly answers both.

Do not collapse those concepts in variable names, comments, plots, or paper claims.

---

## 4. Input and Oracle-Label Integrity

The routing answer must NOT be embedded directly in the model input.

The following are prohibited in the canonical dataset representation:

```text
expert_id / family_id as a numerical feature
oracle routing class encoded into an input slot
final answer / regression target embedded in input features
```

The operation identity itself is legitimate task information. It may be represented with an operation embedding or equivalent encoding.

Oracle labels remain available for diagnostics and explicitly labeled oracle-routing experiments.

The input representation MUST be identical across MoE, MoR, and MoRE for the same record.

---

## 5. Expert Count and Family Manifest

The canonical operation-level POC has **six experts**:

```text
E1 ADD/SUB
E2 MULT/DIV
E3 MOD/POW
E4 LOGIC
E5 SHIFT
E6 SORT/STAT
```

The old E7 catch-all/CHAIN expert is removed from the canonical operation-level experiment.

Rules:

- no hard-coded 7-class heads in a 6-expert run;
- all tensor dimensions derive from `num_experts`;
- all labels derive from one family manifest;
- an unmapped operation must raise an error rather than silently route to a fallback expert.

---

## 6. Fair Baseline Requirements

The primary architecture comparison consists of:

### MoE

```text
num_experts = 6
max_depth = 1
adaptive halting = off
learned top-1 routing = on
```

### MoR

```text
num_experts = 1
canonical max_depth
adaptive halting = on
routing = N/A
```

### MoRE

```text
num_experts = 6
canonical max_depth
adaptive halting = on
learned top-1 routing = on
```

All three must receive **bit-identical input features for the same record**.

A secondary parameter-matched MoR comparison is required if compute permits. Target parameter difference:

```text
abs(P_MoR - P_MoRE) / P_MoRE < 0.05
```

Do not infer architectural superiority from a parameter-unmatched comparison alone.

---

## 7. Router Noise

The canonical model defaults to:

```text
router_noise = none
```

Noise may be evaluated as an explicit ablation:

- none;
- fixed, annealed Gaussian noise;
- trainable Gaussian noise.

Do not add multiple new routing regularizers simultaneously.

Do not add router z-loss to the canonical configuration unless a separate controlled experiment demonstrates it is needed and its effect is clearly isolated.

The old L2 regularization of a trainable noise scale is not assumed to be beneficial and must not be described as a proven fix.

---

## 8. Loss and Metric Rules

Never use `total_loss` as the primary scientific quality metric.

Track separately:

```text
task_loss
classification_loss (if retained)
routing_balance_loss
entropy_term
switch_aux_term
halting_loss / ponder_cost
routing_supervision_loss
weighted_total_loss
```

Validation task loss is the primary predictive metric.

A negative weighted total loss is not automatically a bug, but it must never be interpreted as model quality. The weighting and signs must be documented and tested.

### Routing metrics

The authoritative first-step routing accuracy is:

```text
mean(predicted_expert == oracle_expert)
```

over identical valid tokens used by the confusion matrix.

The confusion-matrix diagonal fraction MUST equal this metric within tolerance.

Also report:

- per-family precision;
- per-family recall;
- macro routing accuracy;
- Hungarian-matched routing accuracy;
- AMI / cluster purity where appropriate.

Permutation-invariant metrics are necessary because expert indices themselves have no semantic identity.

### Expert entropy

Always report normalized entropy:

```text
H_normalized = H / log(E)
```

Raw entropy may be reported in supplementary material.

For `E=1`, entropy is `N/A`, not zero as a comparative quality score.

### Expert similarity

Report mean and maximum pairwise cosine similarity when `E > 1`.

Use:

> consistent with differentiated parameterizations

Do NOT write:

> proves orthogonality

unless the precise mathematical claim is actually established.

### Depth allocation

Report:

- average recursion depth;
- per-operation average depth;
- absolute depth allocation error;
- relative depth allocation error;
- early-exit rate;
- forced-exit rate.

Call the metric **depth allocation error**, not “compute efficiency,” unless actual FLOPs/cost are incorporated.

---

## 9. Reproducibility and Provenance

Every canonical W&B run MUST record:

```text
experiment_id
experiment_group
architecture
variant
seed
dataset_version
train_split_version
code_git_commit
config_hash
num_experts
max_depth
num_blocks
d_model
FFN multiplier
batch_size
lr
weight_decay
routing_balance
halting weight
routing supervision enabled/weight
router noise mode/scale
halt target mode
resolved epochs
resolved subset_fraction
```

Runs must use stable names:

```text
phaseB_MoE_seed42
phaseB_MoR_seed42
phaseB_MoRE_seed42
```

Every figure and table must be traceable to exact run IDs.

Never manually copy numbers into the canonical results table.

---

## 10. Canonical Dataset Integrity

The canonical dataset must pass all of these checks before training:

- no input feature contains the regression target;
- no oracle expert label is embedded in the input;
- no train/val/test record-ID overlap;
- no expression-string overlap;
- no exact `(op,args,result)` tuple overlap when used as a leakage test;
- train/val/test hashes are recorded;
- operation counts and target-depth distributions are recorded.

Trivial predictors must be computed:

```text
predict zero
predict train mean
best single-feature copy baseline
```

A learned model's validation loss is not interpretable without these controls.

---

## 11. Experimental Integrity Rules

No proxy run may enter the headline table.

A proxy is anything with non-canonical:

```text
epoch count
subset fraction
dataset version
batch protocol
loss configuration
architecture
```

No table may mix configurations unless the table is explicitly labeled an ablation and the varying field is stated.

No sentinel values such as `-1` may be reported as measurements.

Use `N/A`.

No causal claim without an experiment that isolates the variable.

Do not alter architecture merely to obtain a favorable result. If an improvement is discovered, preserve the original run and add it as an explicit variant.

---

## 12. Repository Safety

Before the canonical matrix:

- archive invalid historical results rather than deleting the evidence;
- use per-run directories instead of fixed output filenames;
- do not let an autoresearch loop silently modify configs during canonical runs;
- canonical experiments must use an explicit resolved configuration;
- the exporter must refuse to combine mismatched dataset versions or config hashes.

---

## 13. Final Scientific Principle

The purpose of the experiment is to determine whether the MoRE hypothesis is supported.

The expected outcomes are all scientifically valid:

1. MoRE improves accuracy and/or compute trade-offs.
2. MoRE does not improve accuracy but demonstrates stable joint specialization and adaptive computation.
3. MoRE fails to provide a meaningful advantage once the task and implementation are corrected.

The coding agent MUST NOT optimize toward a predetermined outcome.

If the corrected experiment falsifies the original hypothesis, report that result honestly.
