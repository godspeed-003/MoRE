# Rental GPU Readiness Analysis & Onboarding Flags (`rental_onboarding.md`)

This document provides an architectural and execution-level readiness audit of the **MoRE** repository for rented enterprise cloud GPUs (**NVIDIA H100 SXM/PCIe/NVL, H200, B200, A100**).

---

## 1. Executive Verdict: Are We in a Ready State?

### **Yes, the core code is in a fully ready state.**
1. **One Unified Codebase:** All three architectures (`moe`, `mor`, `more`) share the identical training engine (`engine.py`), model definitions (`model.py`), tokenizer, and dataset structures.
2. **Dynamic Hardware Detection:** Compute device binding uses `torch.device("cuda" if torch.cuda.is_available() else "cpu")` and dynamic index allocation without any hard-coded GPU device IDs or machine hostname checks.
3. **Automated Driver (`run_language_matrix.py`):**
   - Incorporates a preflight check (`--preflight`) that verifies real CUDA kernel execution, VRAM headroom, corpus manifest integrity, syntactic lookup presence, and proxy guard compliance before running any compute.
   - Includes skip-on-completion logic so that if the 5 completed MoE runs are pulled onto the rental instance, it skips them automatically and only trains the remaining MoR and MoRE runs.
4. **Frozen Scientific Protocol:** The protocol is locked in `code/canonical_spec_language.json` (`L7.1-language-frozen`). Any inadvertent flag deviation will be blocked by the proxy guard (`assert_not_silent_proxy`) before any GPU time is wasted.

---

## 2. Rental Onboarding Flags & Execution Traps

While the architecture is ready, there are **critical operational and environment flags** that must be observed on a newly rented cloud GPU:

### Flag 1: PyTorch Version & Deterministic Flags
- **Observed Behavior:** The repository uses strict global seeding (`seeding.py` with `deterministic_algorithms=True`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `cudnn.deterministic=True`).
- **Hopper (H100/H200) & Blackwell (B200) Compatibility:**
  - Modern CUDA 12.x attention implementations (`sdpa` / memory-efficient attention) trigger non-deterministic warnings or minor backward algorithm fallback.
  - In `engine.py`, `torch.use_deterministic_algorithms(True, warn_only=True)` is set. **Do NOT pass `warn_only=False`**, or Hopper memory-efficient attention kernels will abort with a runtime exception.
- **Recommended Container / Image:**
  - `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel` (exact PyTorch version parity with local repo) or `pytorch/pytorch:2.4.x / 2.5.x`.

### Flag 2: W&B Mode (MANDATORY OFFLINE)
- **The Trap:** If `WANDB_API_KEY` is not configured on the remote rental machine and `WANDB_MODE` is unset, `train.py` will prompt interactively or raise `UsageError: No API key configured` during `wandb.init`, failing the run after initial setup.
- **The Requirement:** Always set:
  ```bash
  export WANDB_MODE=offline
  ```
  *(All run logs, scalar curves, and final metrics land locally in `runs/<id>/metrics.json` and `results.tsv`).*

### Flag 3: Dataset Arrays Are Not in Git (`data/lang/` Rebuild Required)
- **The Trap:** `data/lang/**/*.npy` (269 MB canonical corpus token array) is git-ignored. Cloned repositories on rental instances will fail preflight if launched immediately.
- **The Requirement:** The dataset and its metric artifacts must be built immediately after clone (~3–4 min on NVMe):
  ```bash
  python data/lang/build_language_dataset.py --corpus wikitext-103 --stage all --cache_dir data/lang/_hf_cache
  python data/lang/build_baseline_floors.py --corpus wikitext-103
  python data/lang/build_depth_deciles.py --corpus wikitext-103
  python data/lang/build_shuffled_control.py --corpus wikitext-103
  ```

### Flag 4: Do NOT Modify Frozen Hyperparameters for H100
- **The Trap:** Because an H100 has 80 GB of VRAM, an operator might be tempted to increase `batch_size` (e.g. from 48 to 128 or 256) or change `lr`.
- **Scientific Invariant:** **Do NOT touch batch_size or learning rate.**
  - `batch_size = 48`, `lr = 0.001`, `seq_len = 256`, and `epochs = 3` are frozen in `canonical_spec_language.json`.
  - Changing batch size changes gradient noise scale, optimizer trajectory, and will trigger a hard `ProxyGuardError` that rejects the run from the canonical group (`canonical_lang_b`).
  - The H100 speedup comes from massive memory bandwidth (3.35 TB/s) executing the depth-7 recursion in sub-milliseconds, not from changing batch dynamics.

### Flag 5: Preserve Completed MoE Runs
- We already completed all 5 MoE runs (`langB_MoE_seed42` through `seed46`).
- When launching the matrix on the cloud instance, we should pass `--arch mor` then `--arch more` so it only runs the remaining 10 runs.

---

## 3. GPU Pricing & Selection Matrix (From Your Provider Listings)

### Provider A: Vast.ai (On-Demand Cloud Spot / Verified)
From the dashboard options provided:

| GPU Model | Architecture | VRAM | Bandwidth | Vast.ai Hourly Rate | Est. 10-Run Time | Est. Cost (10 Runs) | Verdict |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **H100 SXM** | Hopper | **80 GB HBM3** | **3.35 TB/s** | **$1.73 – $2.71 / hr** | **~10 – 11 hrs** | **~$19 – $30 USD** | **BEST VALUE & FASTEST** |
| **H100 NVL** | Hopper | 80 GB HBM3 | 3.90 TB/s | **$1.96 / hr** | ~10 – 11 hrs | ~$21 – $24 USD | Excellent alternative |
| **H100 PCIe**| Hopper | 80 GB HBM2e| 2.00 TB/s | **$2.13 / hr** | ~12 – 14 hrs | ~$25 – $30 USD | Slightly slower memory bus |
| **H200**     | Hopper | 141 GB HBM3e| 4.80 TB/s | **$1.97 / hr** | ~9 – 10 hrs | ~$19 – $22 USD | Outstanding if available |
| **B200**     | Blackwell | 192 GB | 8.00 TB/s | **$5.31 / hr** | ~6 – 7 hrs | ~$35 – $40 USD | Overkill for a 5.5M param model |

### Provider B: E2E Networks (Indian Cloud)
From the E2E Networks pricing table provided:

| GPU Model | VRAM | RAM | Hourly Rate (INR) | Equivalent USD | Est. 10-Run Cost | Notes |
|:---|:---:|:---:|:---:|:---:|:---:|:---|
| **NVIDIA H100** | **80 GB** | **250 GB** | **₹362.00 / hr** | ~$4.32 / hr | **₹3,982 – ₹4,700 INR** | Best Indian domestic option |
| **NVIDIA RTX PRO 6000** | 96 GB | 170 GB | **₹182.00 / hr** | ~$2.17 / hr | ~₹3,600 INR (~18-20 hrs) | GDDR6 (slower on recursion) |
| **NVIDIA A100 (80GB)** | 80 GB | 115 GB | **₹189.00 / hr** | ~$2.25 / hr | ~₹3,400 INR (~16-18 hrs) | Good middle ground |
| **NVIDIA L40S** | 48 GB | 220 GB | **₹102.00 / hr** | ~$1.22 / hr | ~₹2,200 INR (~22 hrs) | Budget option (GDDR6) |

> **Strategic Selection:**
> - **Top Pick for Speed & Lowest Dollar Cost:** **Vast.ai H100 SXM ($1.73 – $2.71/hr)**. Completes everything in ~10 hours for under $30.
> - **Top Pick for Domestic Indian Billing (UPI / RuPay / GST):** **E2E Networks NVIDIA H100 (₹362/hr)**. Completes in ~11 hours for ~₹4,000 INR.

---

## 4. Exact Launch Protocol for the Rental Instance

Once SSH terminal access to the rented instance is open, run these exact command blocks:

### Step 1: System Dependencies & Environment
```bash
apt update && apt install -y git tmux htop nvtop

# Clone and checkout
git clone https://github.com/godspeed-003/MoRE.git
cd MoRE
git checkout rtx4060-english

# Dependencies
pip install -r requirements.txt
pip install wandb nltk scipy scikit-learn datasets tokenizers
python -c "import nltk; nltk.download('averaged_perceptron_tagger_eng')"
```

### Step 2: Build the Corpus & Linguistic Tables
```bash
python data/lang/build_language_dataset.py --corpus wikitext-103 --stage all --cache_dir data/lang/_hf_cache
python data/lang/build_baseline_floors.py --corpus wikitext-103
python data/lang/build_depth_deciles.py --corpus wikitext-103
python data/lang/build_shuffled_control.py --corpus wikitext-103
```

### Step 3: Run the Preflight Verification
```bash
export WANDB_MODE=offline
python code/run_language_matrix.py --preflight
```
*Verify that every line reports `[ok ]`.*

### Step 4: Launch via `tmux` (Disconnect-Safe)
```bash
tmux new -s more_matrix

# Inside tmux:
export WANDB_MODE=offline

# 1. MoR Arm (5 seeds: 42, 43, 44, 45, 46) -> ~5.0 hours
python code/run_language_matrix.py --arch mor

# 2. MoRE Arm (5 seeds: 42, 43, 44, 45, 46) -> ~4.5 hours
python code/run_language_matrix.py --arch more
```
*(Detach with `Ctrl+B` then `D`).*

### Step 5: Export & Commit
```bash
# Compute all statistical comparisons, permutation tests, and markdown tables:
python code/export_results.py --task language

# Commit and push
git add runs/langB_MoR_* runs/langB_MoRE_* results/language/
git commit -m "feat(matrix): complete canonical MoR and MoRE matrix on H100"
git push origin rtx4060-english
```
