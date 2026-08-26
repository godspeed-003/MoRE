# Mixture of Recursive Experts (MoRE)

This repository contains the official PyTorch implementation and ablation sweeps for **Mixture of Recursive Experts (MoRE)**, a novel neural network architecture that combines dynamic Mixture of Experts (MoE) token routing with adaptive Mixture of Recursion (MoR) compute loops.

---

## 1. Core Architectural Fixes & Optimization

Following a detailed analysis of the initial implementation, four critical issues that caused training instability, gradient blockage, and expert collapse were resolved:

1. **Fixed Gradient Flow (Soft Routing)**: 
   - *Problem*: The original code used hard boolean masking to dispatch tokens to experts (`out[mask] = expert(x[mask])`). This discrete selection blocked gradients from flowing from the task loss back to the routing weights (`self.router.weight`).
   - *Fix*: Implemented differentiable soft routing where all experts are evaluated and blended using gating probabilities:
     ```python
     expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
     out = (router_probs.unsqueeze(-1) * expert_outputs).sum(dim=1)
     ```
   - *Result*: Router gradient norms are now non-zero and stable (typically `0.07`–`8.93`), allowing successful routing weight optimization.

2. **Corrected Routing Balance Loss Formula & Sign**:
   - *Problem*: The balance loss formula was defined as `entropy_term - switch_aux` and subtracted from the objective function. This incorrectly rewarded routing collapse (low entropy) and penalised uniform distribution.
   - *Fix*: Corrected the signs to penalise low entropy and reward balanced token distribution:
     ```python
     balance_loss = -entropy_term + switch_aux
     total_loss = ... + balance_loss
     ```

3. **Task-Driven Gating (Oracle Downscaling)**:
   - *Problem*: Direct oracle family classification CE was weighted at `1.0`, forcing the router to act as a static classifier rather than optimizing for prediction difficulty.
   - *Fix*: Scaled down the oracle routing weight to `0.01`, allowing routing choices to be dynamically driven by task complexity.

4. **Noise Scale Regularization**:
   - *Problem*: The trainable `router_noise_scale` parameter converged to 0, leaving the router vulnerable to collapse.
   - *Fix*: Added an L2 weight regularization term ($0.001 \times \text{noise\_scale}^2$) to the loss function to prevent noise scale collapse.

5. **CUDA Device-Side Assert & Out-of-Bounds Fixes**:
   - Resolved shape mismatches and out-of-bounds indices in classification heads and validation loops when running with fewer than 7 experts (such as 1 expert in MoR) or indexing fallback experts.

---

## 2. Canonical Baseline Performance (50-Epoch Results)

Training the corrected implementation on the micro-baseline task configurations yielded an **87.26% reduction in validation loss** and a **212% throughput acceleration** due to healthy adaptive early exiting.

| Metric | Original 1-Block Baseline (Old Code) | Fixed 1-Block Baseline (New Code) | Fixed 2-Block Model |
| :--- | :---: | :---: | :---: |
| **Best Val Loss** | `0.018212` | **`0.002320`** | **`0.002285`** |
| **Val Loss (Epoch 50)** | `0.018212` | **`0.002332`** | **`0.002292`** |
| **Token Throughput (tok/s)** | `8,148.70` | **`25,470.02`** | **`14,679.10`** |
| **Average Recursion Depth** | `1.4537` | **`2.2599`** | **`2.8023`** |
| **Expert Load Entropy** | `0.0` (collapsed) | **`1.9347`** | **`1.6837`** |

### Workload Division (Task-Driven Routing)
* **Simple operations** (e.g., addition, subtraction) halt early at an average depth of **`1.03`** steps.
* **Complex operations** (e.g., logic, shift operations) recurse deeply to an average depth of **`6.68`** steps (approaching the ceiling of `max_depth = 7`).

---

## 3. Proxy Sweep Results

A comprehensive 20-run proxy sweep suite (5 epochs, 10% data subset) was executed to test the codebase across varying settings:

### Structural Ablation
Stacking two physical blocks sequential doubles recursion traversal depth (`3.17` vs `1.92` steps) but increases computational overhead (`205s` vs `148s`).

### Fixed-Depth Ablation
True fixed-depth configurations (Fixed Depth 1 and 5) **no longer cause expert routing collapse** under the new implementation; routing entropy remains extremely high ($>1.92$ out of $\ln(7) \approx 1.94$). Adaptive compute reduces steps significantly compared to Fixed Depth 5 (saving step traversal compute) while maintaining comparable performance.

### Batch Size Sweep
Batch sizes between 128 and 2560 were swept. **768** remains the optimal throughput/convergence sweet spot for training on consumer-grade hardware.

### Scaling Laws
- **MoE** (mixture of experts, single step) scales sharply with dimensions.
- **MoR** (single expert, adaptive recursion) performs poorly due to capacity constraints.
- **MoRE** (mixture of recursive experts) demonstrates highly stable validation loss and robust, balanced routing ($Entropy > 1.93$) across all parameter counts (Nano, Micro, Mini scales).

---

## 4. Getting Started

### Environment Setup
Activate the environment:
```bash
conda activate more_env
```

### Running a Train Job
Run training using a JSON config file:
```bash
python train.py --config config_1block_fixed.json
```

### Running Ablation Sweeps
To execute the suite of ablation sweep runs:
```bash
python run_sweeps.py
```
This writes sequential execution records to `sweep_results.csv`.
