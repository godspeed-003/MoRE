# Objective: Canonical Scientific Evaluation of MoRE

## 0. What We Are Actually Testing

The project is a proof-of-concept study of **MoRE: Mixture of Recursive Experts**.

The terminology is fixed:

- **MoE = Mixture of Experts:** learned Top-1 token routing among independent experts.
- **MoR = Mixture of Recursion:** reuse of a shared computation block over multiple steps with adaptive halting.
- **MoRE = Mixture of Recursive Experts:** a combination in which each token is routed to a specialized expert and that expert/block may recursively process the token for an adaptive number of steps.

The scientific question is:

> Can expert specialization and adaptive recursive computation be learned jointly in one architecture, and what computational/accuracy trade-off does that combination produce compared with MoE-only and MoR-only baselines?

This is an **exploratory architectural POC**, not an unconditional accuracy-optimization contest.

--------------------------------------------------------

## PRIMARY OBJECTIVE

Determine whether the MoRE architecture can train correctly after all task leakage, routing, halting, objective-scaling, and provenance defects are removed.

Acceptance criteria:

- non-trivial predictive learning above trivial baselines;
- no target leakage;
- no expert-collapse pathologies;
- real gradient flow to the halting mechanism;
- reproducible behavior across seeds.

The primary predictive metric is validation task loss.

--------------------------------------------------------

## SECONDARY OBJECTIVE

Determine whether MoRE actually learns **adaptive computation**.

Measure:

- average recursion depth;
- per-operation average depth;
- depth allocation error against the predefined operation-complexity curriculum;
- early-exit rate;
- forced-exit rate.

Success means that learned depth tracks the controlled target-depth curriculum better than a fixed-depth baseline while maintaining useful predictive performance.

Do not call this discovery of intrinsic mathematical complexity. The target depth is a property of the dataset/supervision design.

--------------------------------------------------------

## THIRD OBJECTIVE

Determine whether MoRE retains meaningful expert specialization without expert collapse.

Measure:

- normalized load entropy;
- expert load distribution;
- first-step routing accuracy;
- Hungarian-matched routing accuracy;
- AMI / cluster purity;
- per-family precision/recall;
- mean and maximum pairwise expert cosine similarity.

High entropy is only a load-balance diagnostic.

It is NOT, by itself, a success criterion for useful specialization.

A router that assigns tokens uniformly but incorrectly has high entropy and poor specialization.

--------------------------------------------------------

## FOURTH OBJECTIVE

Compare the two axes of the architecture independently and jointly.

### MoE-only

```text
specialization without recursion
```

### MoR-only

```text
recursion without multiple experts
```

### MoRE

```text
specialization + recursion
```

The main scientific question is not whether MoRE must have the lowest validation loss.

The important question is whether the combination produces behavior that is qualitatively and quantitatively distinct from either component alone.

--------------------------------------------------------

## FIFTH OBJECTIVE

Characterize the compute/accuracy trade-off.

Measure:

- throughput;
- wall-clock training time;
- average recursion depth;
- normalized or analytically calculated FLOPs/token when possible;
- GPU memory;
- validation loss.

Do not claim computational efficiency from throughput alone. Throughput is hardware- and implementation-dependent.

The correct question is:

> At a given predictive quality, how much computation does each architecture require in this controlled implementation?

--------------------------------------------------------

## SIXTH OBJECTIVE

Determine reproducibility.

Use seeds:

```text
42, 43, 44, 45, 46
```

For each primary architecture report:

```text
mean ± standard deviation
```

for:

- best validation loss;
- final validation loss;
- average recursion depth;
- normalized entropy;
- routing accuracy (where applicable);
- depth allocation error;
- throughput.

A single favorable seed is never sufficient evidence for the main claim.

--------------------------------------------------------

## REQUIRED ABLATIONS

1. MoRE learned routing vs MoRE oracle routing.
2. MoRE adaptive depth vs fixed depth.
3. One recursive block vs two.
4. Router noise: none vs fixed annealed vs trainable, if needed.
5. Routing supervision disabled vs enabled.
6. Dense routing vs canonical Top-1 sparse routing, if the implementation was historically dense.
7. Parameter-matched MoR.

Oracle routing is an upper-bound/capacity diagnostic only.

It must never replace the learned-router MoRE result in the headline comparison.

--------------------------------------------------------

## CANONICAL HYPERPARAMETER PRINCIPLE

Do not independently tune each architecture until it wins.

The primary baseline matrix must use a single resolved training protocol:

```text
same dataset
same split
same optimizer
same learning rate
same weight decay
same batch size
same epoch count
same seed set
same input representation
```

The architecture-specific changes should be only those necessary to instantiate:

```text
MoE
MoR
MoRE
```

A secondary parameter-matched MoR analysis is required when feasible.

--------------------------------------------------------

## DATASET OBJECTIVE

The controlled synthetic dataset exists to isolate the architectural mechanisms, not to claim general mathematical reasoning ability.

The dataset must therefore be:

- leakage-free;
- split-clean;
- reproducible;
- documented;
- equipped with an explicit operation-complexity depth curriculum.

The final paper must explicitly state that the dataset is a controlled synthetic benchmark.

--------------------------------------------------------

## WHAT COUNTS AS SUCCESS

### Outcome A — Strong

MoRE retains or approaches baseline predictive quality while demonstrating:

- stable expert specialization;
- adaptive depth;
- useful compute/accuracy behavior;
- low seed variance.

### Outcome B — Publishable exploratory result

MoRE is not the lowest-loss architecture but demonstrates a stable and interpretable combination of specialization and adaptive recursion that neither MoE nor MoR alone provides.

### Outcome C — Negative result

Once leakage, halting, routing, and objective bugs are fixed, MoRE does not provide a meaningful advantage or does not learn adaptive computation.

Outcome C is valid and must be reported honestly.

--------------------------------------------------------

## PROHIBITED OPTIMIZATION BEHAVIOR

Do not:

- select a favorable subset after seeing results;
- change target-depth labels to improve a metric after the fact;
- add oracle routing to the main model;
- hide parameter mismatches;
- treat high entropy as proof of specialization;
- treat low cosine similarity as proof of orthogonality;
- call a controlled depth curriculum “intrinsic complexity discovery”;
- merge proxy runs with canonical runs;
- tune the architecture until the desired conclusion appears.

The objective is to learn what the architecture does, not what we want it to do.

--------------------------------------------------------

## FINAL DELIVERABLE

The final experiment must produce:

```text
results.csv
results.json
results.md
```

`results.md` is the canonical paper-facing empirical record and must include:

1. experiment summary;
2. exact configuration;
3. dataset statistics and leakage audit;
4. parameter counts;
5. MoE/MoR/MoRE main comparison;
6. training dynamics;
7. expert routing analysis;
8. adaptive-depth analysis;
9. ablations;
10. parameter-matched comparison;
11. capacity/scale sweep if available;
12. negative results and failure cases;
13. mean ± std stability statistics;
14. full provenance;
15. paper-safe claims;
16. explicitly unsupported claims;
17. final go/no-go assessment.

The final report must distinguish measured facts from interpretation and must never silently use invalidated historical runs.
