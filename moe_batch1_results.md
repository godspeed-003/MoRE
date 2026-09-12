> **NOT AN ADMISSIBLE RESULT — hand-written narration, kept as evidence (T-LX.7).**
>
> This file is a prose summary written by hand after the arm finished. It is archived
> rather than deleted (`CLAUDE.md` §6) because `changelog.md` cites it, but **no number
> in it may enter a results table** and it is superseded on three points:
>
> 1. **The five `runs/langB_MoE_seed4*__<hash>/` directories are not in this
>    repository.** Until they are pushed, `export_results.py --task language` cannot
>    admit the arm, the proxy guard cannot verify `config_hash` or `dataset_version`,
>    and `CLAUDE.md` §5 forbids hand-copying these numbers into a table.
> 2. **Every `±` below is a population std (n).** The repository uses the sample std
>    (n−1) via `seed_stats.mean_std`, so each figure here understates seed variance by
>    `sqrt(4/5) = 0.894`, i.e. 10.6 %. Correct values: val loss ± 0.0242, load entropy
>    ± 0.1326, Hungarian ± 3.27, AMI ± 0.0478, AMI control ± 0.0107, cosine ± 0.0043.
>    The **means** are all correct, as is every per-seed number.
> 3. **§1 quotes the partition entropy as `0.884`.** That is the wikitext-2 *dev*
>    constant; on canonical wikitext-103 it is `0.894156` (T-LX.6). Each run publishes
>    its own corpus's value as `val/routing_pos_partition_load_entropy`.
>
> The authoritative version of this arm is whatever
> `results/language/results_aggregate.csv` contains once the run directories land.

# MoE (Mixture of Experts) Language Benchmark Results

This report compiles the complete experimental results for **Batch 1** of the canonical language matrix on **WikiText-103**: the **MoE baseline** across 5 global random seeds (`42, 43, 44, 45, 46`).

---

## 1. Executive Summary & Key Findings

- **Validation Task Loss:** **$3.9483 \pm 0.0216$ nats/token** (Perplexity: **$51.86 \pm 1.13$**; Bits/token: **$5.696 \pm 0.031$**).
- **Margin vs. Trivial Bigram Floor:** Every seed strongly outperforms the canonical WikiText-103 backoff bigram baseline ($4.9849$ nats/token) by an average margin of **$+1.0366 \pm 0.0216$ nats/token**.
- **Specialization & Routing Alignment:**
  - **Permutation-Invariant AMI (Adjusted Mutual Information):** **$0.1637 \pm 0.0427$** (evaluated against an empirical permutation control floor of $0.0364 \pm 0.0096$).
  - **Hungarian-Matched Accuracy against POS:** **$36.16\% \pm 2.93\%$** (control floor: $29.66\% \pm 1.13\%$).
  - **Cluster Purity:** **$0.4345 \pm 0.0389$**.
- **Expert Parameter Differentiation:**
  - Mean pairwise cosine similarity across expert FFN weights is low at **$0.0304 \pm 0.0038$** (Max pairwise: **$0.0572 \pm 0.0087$**), consistent with differentiated parameterizations without expert collapse.
- **Normalized Load Entropy:** **$0.6003 \pm 0.1186$** (against the oracle POS partition's theoretical entropy of $0.884$).

---

## 2. Combined Per-Seed Benchmark Table

| Seed | Run ID | Val Loss (nats) | Val PPL | Bits/token | Margin vs Floor | Load Entropy | Hungarian Acc | AMI (real) | AMI Control | Mean Cos Sim |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **42** | `langB_MoE_seed42__51000018` | 3.9629 | 52.61 | 5.717 | +1.0220 | 0.6520 | 31.29% | 0.1421 | 0.0225 | 0.0351 |
| **43** | `langB_MoE_seed43__7d85fe81` | 3.9829 | 53.67 | 5.746 | +1.0020 | 0.3709 | 35.92% | 0.1066 | 0.0326 | 0.0315 |
| **44** | `langB_MoE_seed44__c348b39a` | 3.9400 | 51.42 | 5.684 | +1.0449 | 0.6102 | 36.76% | 0.1652 | 0.0337 | 0.0294 |
| **45** | `langB_MoE_seed45__64800f08` | 3.9311 | 50.96 | 5.671 | +1.0538 | 0.6633 | 36.35% | 0.1673 | 0.0421 | 0.0237 |
| **46** | `langB_MoE_seed46__3d9d12a9` | 3.9245 | 50.63 | 5.662 | +1.0604 | 0.7049 | 40.48% | 0.2371 | 0.0509 | 0.0324 |
| **Aggregate** | **5 Seeds (Mean $\pm$ Std)** | **$3.9483 \pm 0.0216$** | **$51.86 \pm 1.13$** | **$5.696 \pm 0.031$** | **$+1.0366 \pm 0.0216$** | **$0.6003 \pm 0.1186$** | **$36.16\% \pm 2.93\%$** | **$0.1637 \pm 0.0427$** | **$0.0364 \pm 0.0096$** | **$0.0304 \pm 0.0038$** |

---

## 3. Epoch-by-Epoch Convergence Trajectory

All runs executed for the frozen budget of **3 epochs** (~135M tokens/epoch). Average validation task loss progressed as follows:

| Epoch | Avg Train Task Loss | Avg Val Loss (nats) | Avg Val Perplexity | Avg Routing Hungarian Acc | Avg Normalized Entropy |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | ~4.72 | ~4.31 | ~74.4 | ~28.5% | ~0.52 |
| **2** | 4.214 | 4.039 | 56.77 | 33.48% | 0.450 |
| **3** | **4.101** | **3.948** | **51.86** | **36.16%** | **0.600** |

*Key observations:*
1. **Steady Generalization:** Validation loss dropped monotonically across all 5 seeds from ~4.31 nats to ~3.95 nats.
2. **Emergence of Syntactic Specialization:** Unsupervised routing alignment (AMI and Hungarian accuracy against grammatical families) strengthened with training without any oracle routing labels injected into the model.

---

## 4. Architectural & Provenance Specifications

- **Task:** English Language Modeling (`wikitext-103`).
- **Architecture:** `MoE` (Top-1 sparse token dispatch).
- **Parameters:** $5,584,908$ total ($3,422,220$ non-embedding). Exactly matched with MoRE.
- **Hyperparameters:**
  - Number of experts: $E=6$
  - Recursion depth: $1$ (no recurrence in MoE)
  - FFN expansion multiplier: $4$
  - Sequence length: $256$
  - Batch size: $48$
  - Learning rate: $0.001$ (AdamW)
  - Dropout: $0.1$
  - Balance loss weight: $0.001$
  - Halting weight: $0.0$ (definitional for MoE)
  - Routing supervision: `0.0` (unsupervised routing)
- **Primary Metric Floor:** $4.9849$ nats/token (WikiText-103 backoff bigram).
- **Hardware Platform:** NVIDIA GeForce RTX 4060 Laptop GPU (8 GB VRAM).
- **Throughput:** ~335 tokens/sec (~1.1 hours per seed).
