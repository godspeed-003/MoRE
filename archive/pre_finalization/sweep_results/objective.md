> **INVALIDATED FOR SCIENTIFIC COMPARISON**
>
> Reason: confirmed target leakage and/or broken halting/routing implementation.
>
> Retained as research history only. Superseded by `plan.md`,
> `updated_rules.md`, `updated_objective.md`. Nothing in this file may
> be cited, exported, or used as a baseline for scientific comparison.

---

# Objective: Improve the MoRE Architecture

The goal of the autoresearch system is NOT to minimize validation loss alone.

The goal is to discover architectural modifications that improve the routing behaviour of Mixture of Recursive Experts while preserving predictive performance.

The baseline architecture has already converged after a full 50-epoch run.

Each automated experiment performs only a 5-epoch proxy evaluation on a deterministic 10% subset of the dataset.

The purpose of this proxy is to determine whether a proposed architectural modification deserves a full training run.

--------------------------------------------------------

PRIMARY OBJECTIVE

Prevent expert collapse.

Target:

expert_entropy > baseline

Never accept routing collapse.

--------------------------------------------------------

SECONDARY OBJECTIVE

Healthy recursion.

Preferred range:

1.5 <= avg_depth <= 3.0

Reject

avg_depth == 1

or

avg_depth == max_depth

--------------------------------------------------------

THIRD OBJECTIVE

Maintain expert diversity.

Minimize

max_pairwise_cosine_similarity

Never allow experts to become identical.

--------------------------------------------------------

FOURTH OBJECTIVE

Increase early convergence speed.

Compare the loss slope across the first five epochs.

A steeper decrease is preferred even if the final loss is not yet optimal.

--------------------------------------------------------

LAST OBJECTIVE

Reduce validation loss.

Validation loss is important but must never be improved by sacrificing routing quality.

--------------------------------------------------------

Allowed modifications

- routing loss weights
- halting loss
- dropout
- router temperature
- router noise
- label smoothing
- routing regularization
- expert implementation
- gating implementation

Avoid changing

- dataset
- evaluation procedure
- logging
- metric definitions

Always produce one small modification per iteration.