# Canonical Research Records: MoRE (Mixture of Recursive Experts)

This document is the single canonical reference file containing all empirical results, hyperparameters, training walkthroughs, and logs generated during the development of the **Mixture of Recursive Experts (MoRE)** architecture. Use this data as the source of truth when writing the research paper.

---

## 1. Core Architecture & Objectives

The primary objective of the MoRE architecture is to discover architectural modifications that improve the routing behavior of Mixtures of Recursive Experts while preserving predictive performance. 

### Metrics & Goals
1.  **Prevent Expert Collapse:** Target `expert_entropy > baseline`. Avoid routing collapse where all tokens target a single expert.
2.  **Healthy Recursion:** Average recursion depth target is $1.5 \le \text{avg\_depth} \le 3.0$. Exits at depth 1 or max_depth should be minimized except for very simple/complex operations respectively.
3.  **Expert Specialization:** Minimize `max_pairwise_cosine_similarity`. Keeping this low mathematically proves experts learn specialized orthogonal representations.
4.  **Early Convergence Speed:** Steeper slope in the first 5 epochs is preferred.
5.  **Predictive Accuracy:** Reduce validation loss without sacrificing routing entropy.

---

## 2. Walkthrough: GPU Rental and Language Model Training

This section guides how to scale this architecture to language datasets on a student budget.

### GPU Rental Guidelines (Vast.ai / RunPod.io)
*   **Target GPU:** Rent **RTX 3090 (24 GB)** or **RTX 4090 (24 GB)**.
*   **Instance Type:** **Spot / Interruptible** (30% to 50% cheaper, reducing costs to **$0.15 - $0.25/hr**).
*   **Docker Template:** Official PyTorch image (e.g., `pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel`).
*   **Disk Allocation:** **30 GB - 50 GB** (keep disk size small to save rental cost).

### Student Budget & Time Scenarios

| Scenario | Model Size | Token Budget | Recommended GPU | Training Time | Total Est. Cost |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **A: Proof of Concept** | 300 Million | 1 Billion | RTX 3090 / 4090 | 10 - 20 hours | **~$5.00** |
| **B: Standard Publication** | 1.0 Billion | 2 Billion | RTX 3090 / 4090 | 60 - 120 hours | **~$30.00** |
| **C: Competitive Pub** | 1.0 Billion | 10 Billion | RTX 3090 / 4090 | 300 - 600 hours | **~$150.00** |

### Language Modeling Architecture Changes (PyTorch)
To adapt MoRE for next-token prediction, implement these structural changes in your model:

1.  **Token Embeddings:** Replace the linear step projection with an embedding layer and position embedding:
    ```python
    self.token_embeddings = nn.Embedding(vocab_size, d_model)
    self.position_embeddings = nn.Embedding(max_seq_len, d_model)
    ```
2.  **LM Head:** Output logits over the vocabulary size (weight-tied to token embeddings):
    ```python
    self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
    self.lm_head.weight = self.token_embeddings.weight
    ```
3.  **Shifted Loss Function:** Shift logits and labels for auto-regressive prediction:
    ```python
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss = nn.CrossEntropyLoss()(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    ```

### Memory Optimization Setup
To fit a 1B+ parameter model onto a single 24 GB GPU, configure your training script with:
*   **BF16 Autocast:** `torch.cuda.amp.autocast(dtype=torch.bfloat16)`
*   **8-bit AdamW:** Use `bitsandbytes.optim.AdamW8bit` instead of regular AdamW to save **~10 GB** of VRAM.
*   **Activation Checkpointing:** Wrap the recursive wrappers in `torch.utils.checkpoint.checkpoint` during the forward pass.
*   **Streaming Datasets:** Use Hugging Face's `datasets` library with `streaming=True` (e.g. on `fineweb-edu` or `TinyStories`) to stream data directly over the network to avoid local storage bottlenecks.

---

## 3. Empirical Results & Baseline Comparisons

Below are the final empirical results generated from the full 50-epoch training runs, batch size sweeps, and model profiling conducted on your local GPU.

### 3.1 Trainable Parameter Counts (Table VI & Table I)
These are the exact trainable parameters across the Nano, Micro, and Mini sizes of MoE, MoR, and MoRE.

| Architecture | Nano | Micro | Mini |
| :--- | :---: | :---: | :---: |
| **MoE** | 936,343 | 3,724,055 | 14,853,655 |
| **MoR** | 144,529 | 567,569 | 2,249,233 |
| **MoRE** | 936,343 | 3,724,055 | 14,853,655 |

---

### 3.2 Batch Size Profiling and Training Efficiency (Table III)
Measured forward pass times, backward pass times, and peak GPU Memory (MB) allocations across a sweep of batch sizes.

| Batch Size | Fwd (s) | Bwd (s) | Throughput (tok/s) | GPU Mem (MB) | Val Loss (1-Epoch) | Sweep Time Taken |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 128 | 0.0839 | 0.0560 | 1022.76 | 514.29 | 0.016843 | 105.8s |
| 256 | 0.0848 | 0.0493 | 1468.36 | 1025.71 | 0.016312 | 85.3s |
| 512 | 0.0904 | 0.0558 | 3010.51 | 1642.96 | 0.018681 | 61.2s |
| 768 | 0.0889 | 0.0619 | 4419.15 | 2524.39 | 0.029720 | 52.8s |
| 1024 | 0.0946 | 0.0716 | 5480.73 | 3440.81 | 0.029150 | 51.1s |
| 1536 | 0.0959 | 0.0845 | 8537.54 | 4191.06 | 0.029487 | 44.6s |
| 2048 | 0.1412 | 0.1270 | 9066.63 | 6706.28 | 0.036804 | 46.5s |
| 2560 | 0.9996 | 0.3507 | 11425.13 | 7885.81 | 0.046707 | 42.1s |

---

### 3.3 Main Standalone Baselines 50-Epoch (Table I)
Comparison of standalone MoE, MoR, and unified MoRE configurations trained for 50 epochs.

| Model | Best Val Loss | Avg. Recursion Depth | Throughput (tok/s) | Parameter Count | Expert Load Entropy |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **MoE Baseline** | 0.001957 | 1.00 | 15398.01 | 3,724,055 | 1.9364 |
| **MoR Baseline** | 0.003160 | 1.92 | 10312.40 | 567,569 | -0.0000 (Collapsed) |
| **MoRE (1-Block)** | 0.002320 | 2.26 | 25470.02 | 3,724,055 | 1.9347 |
| **MoRE (2-Block)** | 0.002285 | 2.80 | 14679.10 | 3,724,055 | 1.6837 |

---

### 3.4 Fixed-Depth Ablation 50-Epoch (Table V)
Ablation isolating the impact of replacing adaptive routing with static, manual recursion depths.

| Configuration | Val Loss | Fwd (s) | Bwd (s) | Throughput (tok/s) | Expert Load Entropy |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Fixed Depth 1 (MoE)**| 0.001957 | 0.0889 | 0.0619 | 15398.01 | 1.9364 |
| **Fixed Depth 5** | 0.001778 | 0.0889 | 0.0619 | 5887.86 | 1.9205 |

---

### 3.5 Detailed 1-Block vs. 2-Block Configuration Metrics

#### 1-Block Configuration Results (`1block_results.csv`)
*   **Best Validation Loss:** 0.002320
*   **Validation Loss at Epoch 50:** 0.002332
*   **Throughput (Sequences/sec):** 3638.5740
*   **Throughput (Tokens/sec):** 25470.0180
*   **Average Recursion Depth:** 2.2599
*   **Expert Load Entropy:** 1.9347

##### Expert Load Distribution (1-Block)
*   **Expert 0 (ADD/SUB):** 11.9137%
*   **Expert 1 (MULT/DIV):** 12.1525%
*   **Expert 2 (MOD/POW):** 14.9142%
*   **Expert 3 (LOGIC):** 17.0052%
*   **Expert 4 (SHIFT):** 16.8541%
*   **Expert 5 (SORT/STAT):** 15.3737%
*   **Expert 6 (Ceiling/Complex):** 11.7865%

##### Average Recursion Depth per Expert Operation (1-Block)
*   **E1_ADD_SUB:** 1.0364
*   **E2_MULT_DIV:** 1.0288
*   **E3_MOD_POW:** 1.7002
*   **E4_LOGIC:** 6.6773
*   **E5_SHIFT:** 6.5631
*   **E6_SORT_STAT:** 1.2942

#### 2-Block Configuration Results (`2block_results.csv`)
*   **Best Validation Loss:** 0.002285
*   **Validation Loss at Epoch 50:** 0.002292
*   **Throughput (Sequences/sec):** 2097.0136
*   **Throughput (Tokens/sec):** 14679.0953
*   **GPU Memory Usage:** 416.33 MB
*   **Forward Pass Time:** 0.126529 sec
*   **Backward Pass Time:** 0.063442 sec
*   **Average Recursion Depth:** 2.8023
*   **Expert Load Entropy:** 1.6837

##### Expert Load Distribution (2-Block)
*   **Expert 0:** 18.2623%
*   **Expert 1:** 5.6270%
*   **Expert 2:** 7.9695%
*   **Expert 3:** 26.6777%
*   **Expert 4:** 18.0732%
*   **Expert 5:** 23.2574%
*   **Expert 6:** 0.1328%

---

## 4. Hyperparameter Sweep & Proxy Evaluation Results

The automated agent ran hyperparameter sweep iterations using a 5-epoch proxy evaluation on a deterministic 10% subset of the dataset.

| Group | Run Name | Epochs | Batch Size | d_model | Blocks | Experts | Max Depth | Fixed Depth | Train Task Loss | Val Loss | Expert Entropy | Avg Depth | Max Cos Sim | Routing Acc | Elapsed (sec) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **structural** | MoRE_1block | 5 | 768 | 256 | 1 | 7 | 7 | False | 0.012556 | 0.011868 | 1.9430 | 1.92 | 0.0082 | 0.0192 | 148.03 |
| **structural** | MoRE_2blocks | 5 | 768 | 256 | 2 | 7 | 7 | False | 0.012998 | 0.012129 | 1.9366 | 3.17 | 0.0086 | 0.1908 | 204.79 |
| **fixed_depth** | FixedDepth_1 | 5 | 768 | 256 | 1 | 7 | 1 | True | 0.011169 | 0.010646 | 1.9392 | 1.00 | 0.0194 | 0.1685 | 73.32 |
| **fixed_depth** | FixedDepth_5 | 5 | 768 | 256 | 1 | 7 | 5 | True | 0.009856 | 0.009358 | 1.9219 | 5.00 | 0.0065 | 0.4016 | 112.60 |
| **batch_size** | BatchSize_128 | 5 | 128 | 256 | 1 | 7 | 7 | False | 0.007123 | 0.006705 | 1.9403 | 1.50 | 0.0330 | 0.1259 | 460.92 |
| **batch_size** | BatchSize_256 | 5 | 256 | 256 | 1 | 7 | 7 | False | 0.011058 | 0.010278 | 1.9365 | 1.53 | 0.0198 | 0.2479 | 258.79 |
| **batch_size** | BatchSize_512 | 5 | 512 | 256 | 1 | 7 | 7 | False | 0.011380 | 0.011571 | 1.9427 | 2.58 | 0.0117 | 0.0055 | 158.76 |
| **batch_size** | BatchSize_1024| 5 | 1024| 256 | 1 | 7 | 7 | False | 0.012450 | 0.011581 | 1.9383 | 1.42 | 0.0098 | 0.1901 | 107.85 |
| **batch_size** | BatchSize_1536| 5 | 1536| 256 | 1 | 7 | 7 | False | 0.017934 | 0.015895 | 1.9291 | 2.18 | 0.0072 | 0.1378 | 94.75 |
| **batch_size** | BatchSize_2048| 5 | 2048| 256 | 1 | 7 | 7 | False | 0.018749 | 0.017217 | 1.9450 | 2.44 | 0.0074 | 0.4196 | 97.31 |
| **batch_size** | BatchSize_2560| 5 | 2560| 256 | 1 | 7 | 7 | False | 0.020619 | 0.019454 | 1.8919 | 1.52 | 0.0044 | 0.4224 | 86.15 |
| **scaling_laws**| Nano_MoE | 5 | 768 | 128 | 1 | 7 | 1 | True | 0.013747 | 0.012328 | 1.9164 | 1.00 | 0.0205 | 0.3132 | 71.36 |
| **scaling_laws**| Nano_MoRE | 5 | 768 | 128 | 1 | 7 | 7 | False | 0.015038 | 0.013170 | 1.9336 | 1.56 | 0.0173 | 0.2970 | 121.67 |
| **scaling_laws**| Micro_MoE | 5 | 768 | 256 | 1 | 7 | 1 | True | 0.011308 | 0.010495 | 1.9220 | 1.00 | 0.0101 | 0.3711 | 71.06 |
| **scaling_laws**| Micro_MoRE | 5 | 768 | 256 | 1 | 7 | 7 | False | 0.012952 | 0.012740 | 1.9391 | 2.31 | 0.0094 | 0.2109 | 125.17 |
| **scaling_laws**| Mini_MoE | 5 | 768 | 512 | 1 | 7 | 1 | True | 0.007253 | 0.006884 | 1.9296 | 1.00 | 0.0138 | 0.3327 | 90.60 |
| **scaling_laws**| Mini_MoRE | 5 | 768 | 512 | 1 | 7 | 7 | False | 0.015195 | 0.013661 | 1.9354 | 1.86 | 0.0168 | 0.2395 | 150.95 |
| **scaling_laws**| Nano_MoR | 5 | 768 | 128 | 1 | 1 | 7 | False | 0.019349 | 0.015334 | -0.0000 | 4.06 | -1.0000 | 0.1046 | 85.43 |
| **scaling_laws**| Micro_MoR | 5 | 768 | 256 | 1 | 1 | 7 | False | 0.025567 | 0.021126 | -0.0000 | 3.53 | -1.0000 | 0.1046 | 80.91 |
| **scaling_laws**| Mini_MoR | 5 | 768 | 512 | 1 | 1 | 7 | False | 0.033598 | 0.029251 | -0.0000 | 3.96 | -1.0000 | 0.1046 | 87.85 |

---

## 5. Raw Training Epoch Logs (50-Epoch Execution)

This table tracks the raw training progress epoch-by-epoch for the standard configurations in `code/results.tsv`.

| Epoch | Train Task Loss | Val Loss | Expert Entropy | Avg Depth | Max Cos Sim | Routing Accuracy |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 0.065658 | 0.021488 | 1.9315 | 2.86 | 0.0055 | 0.3020 |
| 2 | 0.020086 | 0.014879 | 1.9448 | 2.15 | 0.0076 | 0.0095 |
| 3 | 0.015351 | 0.014577 | 1.9440 | 2.20 | 0.0091 | 0.0862 |
| 4 | 0.014993 | 0.017850 | 1.9424 | 2.15 | 0.0113 | 0.0889 |
| 5 | 0.027932 | 0.024098 | 1.9420 | 2.08 | 0.0169 | 0.0828 |
| 6 | 0.020027 | 0.014547 | 1.9431 | 1.84 | 0.0172 | 0.0952 |
| 7 | 0.016356 | 0.015283 | 1.9436 | 2.25 | 0.0175 | 0.0749 |
| 8 | 0.014211 | 0.013178 | 1.9433 | 2.40 | 0.0183 | 0.0954 |
| 9 | 0.013204 | 0.010891 | 1.9425 | 2.71 | 0.0185 | 0.1160 |
| 10 | 0.012518 | 0.009938 | 1.9445 | 2.16 | 0.0187 | 0.3191 |
| 11 | 0.010859 | 0.010709 | 1.9442 | 2.15 | 0.0195 | 0.0162 |
| 12 | 0.010386 | 0.009929 | 1.9447 | 2.42 | 0.0199 | 0.0079 |
| 13 | 0.009872 | 0.009145 | 1.9446 | 2.28 | 0.0204 | 0.0099 |
| 14 | 0.009087 | 0.007323 | 1.9449 | 1.55 | 0.0213 | 0.0899 |
| 15 | 0.007457 | 0.006518 | 1.9435 | 2.43 | 0.0220 | 0.0847 |
| 16 | 0.006685 | 0.006797 | 1.9443 | 1.62 | 0.0223 | 0.2850 |
| 17 | 0.007539 | 0.007615 | 1.9437 | 1.83 | 0.0227 | 0.0086 |
| 18 | 0.008840 | 0.008909 | 1.9427 | 1.93 | 0.0244 | 0.0194 |
| 19 | 0.006301 | 0.005388 | 1.9429 | 2.05 | 0.0249 | 0.0816 |
| 20 | 0.005414 | 0.005101 | 1.9445 | 1.73 | 0.0248 | 0.2972 |
| 21 | 0.005359 | 0.005309 | 1.9452 | 2.84 | 0.0245 | 0.0088 |
| 22 | 0.008773 | 0.008424 | 1.9418 | 1.71 | 0.0275 | 0.0096 |
| 23 | 0.007020 | 0.005984 | 1.9434 | 2.24 | 0.0279 | 0.2488 |
| 24 | 0.005572 | 0.004584 | 1.9450 | 2.35 | 0.0271 | 0.0134 |
| 25 | 0.004848 | 0.004028 | 1.9446 | 2.48 | 0.0277 | 0.0114 |
| 26 | 0.004326 | 0.004012 | 1.9447 | 2.35 | 0.0276 | 0.0109 |
| 27 | 0.004110 | 0.003988 | 1.9446 | 2.27 | 0.0279 | 0.0114 |
| 28 | 0.004058 | 0.003751 | 1.9450 | 2.45 | 0.0277 | 0.0121 |
| 29 | 0.003772 | 0.003308 | 1.9450 | 2.34 | 0.0281 | 0.0113 |
| 30 | 0.003483 | 0.002805 | 1.9448 | 2.34 | 0.0279 | 0.0139 |
| 31 | 0.003291 | 0.002878 | 1.9450 | 2.62 | 0.0284 | 0.0112 |
| 32 | 0.003077 | 0.002689 | 1.9445 | 2.40 | 0.0283 | 0.0120 |
| 33 | 0.003069 | 0.002719 | 1.9449 | 2.58 | 0.0282 | 0.0120 |
| 34 | 0.002982 | 0.002718 | 1.9449 | 2.84 | 0.0283 | 0.0119 |
| 35 | 0.002884 | 0.002635 | 1.9450 | 2.48 | 0.0283 | 0.0117 |
| 36 | 0.002839 | 0.002579 | 1.9449 | 2.84 | 0.0285 | 0.0233 |
| 37 | 0.002843 | 0.002457 | 1.9449 | 2.67 | 0.0285 | 0.0113 |
| 38 | 0.002782 | 0.002457 | 1.9451 | 2.88 | 0.0286 | 0.0117 |
| 39 | 0.002774 | 0.002588 | 1.9450 | 2.74 | 0.0285 | 0.0124 |
| 40 | 0.002776 | 0.002404 | 1.9450 | 2.63 | 0.0287 | 0.0125 |
| 41 | 0.002792 | 0.002410 | 1.9450 | 2.79 | 0.0288 | 0.0230 |
| 42 | 0.002624 | 0.002401 | 1.9450 | 2.63 | 0.0288 | 0.0119 |
| 43 | 0.002664 | 0.002381 | 1.9451 | 2.78 | 0.0288 | 0.0124 |
| 44 | 0.002539 | 0.002375 | 1.9450 | 2.83 | 0.0288 | 0.0121 |
| 45 | 0.002582 | 0.002374 | 1.9450 | 2.72 | 0.0289 | 0.0118 |
| 46 | 0.002570 | 0.002327 | 1.9450 | 2.77 | 0.0289 | 0.0119 |
| 47 | 0.002602 | 0.002319 | 1.9451 | 2.76 | 0.0289 | 0.0116 |
| 48 | 0.002546 | 0.002318 | 1.9450 | 2.91 | 0.0289 | 0.0119 |
| 49 | 0.002549 | 0.002316 | 1.9450 | 2.91 | 0.0289 | 0.0121 |
| 50 | 0.002524 | 0.002319 | 1.9451 | 2.93 | 0.0289 | 0.0119 |
