# INVALIDATED FOR SCIENTIFIC COMPARISON
# Reason: confirmed target leakage and/or broken halting/routing implementation.
#
# Retained as research history only. Do not run against the canonical
# pipeline; paths and assumptions here predate plan.md Phase 0.

import numpy as np
import matplotlib.pyplot as plt

num_runs = 100
num_steps = 7
num_experts = 7

heatmap = np.zeros((num_experts, num_steps))

for _ in range(num_runs):
    routing_path = np.random.randint(0, num_experts, size=num_steps)
    for step, expert in enumerate(routing_path):
        heatmap[expert, step] += 1

plt.imshow(heatmap, cmap="hot")
plt.colorbar(label="Selection Count")

plt.xticks(range(num_steps), [f"S{i+1}" for i in range(num_steps)])
plt.yticks(range(num_experts), [f"E{i+1}" for i in range(num_experts)])

plt.title("Routing Frequency Heatmap (100 Runs)")
plt.xlabel("Recursion Depth")
plt.ylabel("Experts")
plt.gca().invert_yaxis()
plt.show()