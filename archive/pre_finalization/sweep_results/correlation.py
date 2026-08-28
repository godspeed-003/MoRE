# INVALIDATED FOR SCIENTIFIC COMPARISON
# Reason: confirmed target leakage and/or broken halting/routing implementation.
#
# Retained as research history only. Do not run against the canonical
# pipeline; paths and assumptions here predate plan.md Phase 0.

###correlation.py
# Group your evaluation dataset by difficulty. For example:

#     Low Complexity: Simple operations (a + b).

#     Medium Complexity: Multi-step expressions (a * b + c).

#     High Complexity: Deeply nested equations (((a + b) * c) >> d).

# Plot a curve showing the Average Execution Depth against these complexity tiers. A successful MoRE architecture will show a distinct step-ladder pattern: simple tokens exit at depth 1 or 2, while complex tokens utilize the full 6 or 7 steps. This proves your primary efficiency thesis—that the model successfully allocates compute dynamically based on task hardness.
