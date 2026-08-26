# Walkthrough: Renting GPUs and Training MoRE on Language Data (Student Budget Guide)

This walkthrough provides a step-by-step guide to adapting the **Mixture of Recursive Experts (MoRE)** architecture for auto-regressive language modeling (next-token prediction), renting cheap hourly GPUs, and training a standard publication-size model on a student budget.

---

## Table of Contents
1. [GPU Rental Strategy (RunPod vs. Vast.ai)](#1-gpu-rental-strategy-runpod-vs-vastai)
2. [VRAM Math & Budget Scenarios](#2-vram-math--budget-scenarios)
3. [Adapting MoRE for Language Modeling (Next-Token Prediction)](#3-adapting-more-for-language-modeling-next-token-prediction)
4. [Streaming Language Datasets (Zero Disk Space Setup)](#4-streaming-language-datasets-zero-disk-space-setup)
5. [VRAM-Saving Code Implementation (BF16, 8-bit Adam, Checkpointing)](#5-vram-saving-code-implementation-bf16-8-bit-adam-checkpointing)
6. [Step-by-Step Setup & Running the Job](#6-step-by-step-setup--running-the-job)

---

## 1. GPU Rental Strategy (RunPod vs. Vast.ai)

For student budgets, traditional cloud providers (AWS, GCP, Azure) are too expensive. Instead, use decentralized or hobbyist-friendly GPU clouds:

### Option A: RunPod.io (Highly Recommended)
*   **Pros:** Clean UI, very reliable, easy to spin up pre-configured templates (e.g., PyTorch template), has a built-in terminal and web-based VS Code/Jupyter.
*   **Cons:** Slightly more expensive than Vast.ai (usually $0.05 - $0.10/hr more).
*   **Average RTX 3090 Spot Price:** ~$0.22 - $0.28 / hour.

### Option B: Vast.ai
*   **Pros:** Cheapest possible prices on the internet; highly customizable.
*   **Cons:** You are renting from individuals hosting rigs, so machine reliability/network bandwidth varies.
*   **Average RTX 3090 Spot Price:** ~$0.15 - $0.22 / hour.

### Step-by-Step Selection Guide
1.  **Select Spot / Interruptible:** Always choose "Spot" rather than "On-Demand". It is 30% to 50% cheaper.
2.  **Select the Docker Template:** Choose the official **PyTorch** template (e.g., `pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel` or latest).
3.  **VRAM Target:** Choose a card with **24 GB VRAM** (RTX 3090 or RTX 4090). Avoid older cards like Tesla V100/RTX 2080 Ti (no native BF16 support) and Tesla T4 (extremely slow).
4.  **Allocate Disk Space:** Since we will be streaming the language dataset, you only need **30 GB - 50 GB** of disk space (enough for PyTorch docker layers, cache, and saving checkpoints).

---

## 2. VRAM Math & Budget Scenarios

During training, memory is split into static parameters (weights + gradients + optimizer states) and dynamic activation states.
$$\text{Static VRAM} \approx 16 \text{ bytes} \times \text{Number of Parameters } (P)$$

Here are the optimal training configurations based on your project target:

| Scenario | Target Model Size | Dataset Size (Tokens) | Recommended GPU | Approx. Training Time | Total Dollar Cost |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **A: Proof of Concept** | 300 Million | 1 Billion | RTX 3090 / 4090 | 10 - 20 hours | **~$5.00** |
| **B: Standard Pub Target** | 1.0 Billion | 2 Billion | RTX 3090 / 4090 | 60 - 120 hours | **~$30.00** |
| **C: Competitive Pub** | 1.0 Billion | 10 Billion | RTX 3090 / 4090 | 300 - 600 hours | **~$150.00** |

---

## 3. Adapting MoRE for Language Modeling (Next-Token Prediction)

Currently, `train.py` is set up for regression (math operations data). To train a language model, we must adapt the architecture:
1.  Replace `step_proj` (Linear) with `nn.Embedding(vocab_size, d_model)`.
2.  Replace the regression/classification heads with an `lm_head` (Linear) mapping `d_model` back to `vocab_size`.
3.  Compute standard Cross-Entropy Loss on shifted tokens.

### Architecture Adaptation Blueprint
Here is a code design of how you can adapt the model for language data:

```python
import torch
import torch.nn as nn
from transformers import AutoTokenizer

class MoREForCausalLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        num_experts: int = 8,
        max_depth: int = 6,
        num_blocks: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        
        # 1. Text token embedding instead of step linear projection
        self.token_embeddings = nn.Embedding(vocab_size, d_model)
        self.position_embeddings = nn.Embedding(2048, d_model) # e.g. 2048 max context length
        
        # 2. Re-use your MoRE recursive expert blocks
        self.blocks = nn.ModuleList([
            MoREWrapper(
                d_model=d_model, 
                max_depth=max_depth, 
                num_experts=num_experts, 
                dropout=dropout
            )
            for _ in range(num_blocks)
        ])
        
        self.layer_norm = nn.LayerNorm(d_model)
        
        # 3. LM head: map representations back to vocabulary
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        # Weight tie embeddings with LM head to save VRAM and improve training
        self.lm_head.weight = self.token_embeddings.weight
        
    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor = None):
        # input_ids shape: [BatchSize, SeqLen]
        B, S = input_ids.shape
        device = input_ids.device
        
        # Compute embeddings
        pos = torch.arange(0, S, dtype=torch.long, device=device).unsqueeze(0)
        h = self.token_embeddings(input_ids) + self.position_embeddings(pos)
        
        # Create a boolean step mask (True for all tokens since we don't pad active sequence tokens)
        mask = torch.ones((B, S), dtype=torch.bool, device=device)
        
        # Forward pass through recursive MoRE wrappers
        total_bal_loss = torch.tensor(0.0, device=device)
        total_halt_loss = torch.tensor(0.0, device=device)
        
        for block in self.blocks:
            # We pass our token hidden states through recursive expert layers
            h, bal_loss, halt_loss, _, _, _, _, _ = block(h, mask)
            total_bal_loss += bal_loss
            total_halt_loss += halt_loss
            
        h = self.layer_norm(h)
        logits = self.lm_head(h) # [B, S, vocab_size]
        
        # Compute auto-regressive cross-entropy loss if labels are provided
        loss = None
        if labels is not None:
            # Shift logits and labels for causal next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss_fct = nn.CrossEntropyLoss()
            task_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            # Combine task loss with routing regularization losses
            loss = task_loss + (0.05 * total_bal_loss) + (0.001 * total_halt_loss)
            
        return logits, loss
```

---

## 4. Streaming Language Datasets (Zero Disk Space Setup)

Downloading 100GB+ datasets (e.g. SlimPajama or FineWeb) requires expensive disk volume rental and hours of download time. Instead, use Hugging Face's `datasets` library in **streaming mode** to load data on the fly.

### Tokenized Streaming Dataset Code
Add this file as `dataset.py` in your code directory:

```python
import torch
from torch.utils.data import IterableDataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer

class StreamingLanguageDataset(IterableDataset):
    def __init__(self, dataset_name: str, split: str, tokenizer_name: str, max_seq_length: int = 1024):
        self.dataset = load_dataset(dataset_name, split=split, streaming=True)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_seq_length = max_seq_length

    def __iter__(self):
        # Accumulate text to form fixed-length context windows
        buffer = []
        for example in self.dataset:
            text = example.get("text", "")
            tokens = self.tokenizer.encode(text)
            buffer.extend(tokens)
            
            # Yield chunks of max_seq_length
            while len(buffer) >= self.max_seq_length:
                chunk = buffer[:self.max_seq_length]
                buffer = buffer[self.max_seq_length:]
                
                input_ids = torch.tensor(chunk, dtype=torch.long)
                # For causal LM, labels are identical to input_ids (loss shifts them internally)
                yield {
                    "input_ids": input_ids,
                    "labels": input_ids.clone()
                }

# Usage Example:
# train_dataset = StreamingLanguageDataset(
#     dataset_name="HuggingFaceFW/fineweb-edu", 
#     split="train", 
#     tokenizer_name="gpt2", 
#     max_seq_length=1024
# )
# train_loader = DataLoader(train_dataset, batch_size=8)
```

---

## 5. VRAM-Saving Code Implementation (BF16, 8-bit Adam, Checkpointing)

To successfully train a **1 Billion parameter model** on a single **RTX 3090 (24 GB)**, apply these memory optimizations:

### Optimization 1: Native PyTorch BF16 Autocast
Ensure your training step uses `torch.cuda.amp.autocast`:
```python
scaler = torch.cuda.amp.GradScaler(enabled=False) # BF16 does not need scaling, unlike FP16

for batch in dataloader:
    optimizer.zero_grad()
    inputs = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)
    
    # Run forward pass under autocast context
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        logits, loss = model(inputs, labels=labels)
        
    # Backward pass
    loss.backward()
    
    # Gradient clipping to prevent gradient explosion
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
```

### Optimization 2: 8-Bit Adam Optimizer (bitsandbytes)
Replace default PyTorch AdamW to save **~10 GB of VRAM** for a 1B parameter model:
```bash
pip install bitsandbytes
```
```python
import bitsandbytes as bnb

# Replace standard optimizer:
# optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=1e-4, betas=(0.9, 0.95), weight_decay=0.1)
```

### Optimization 3: Activation Checkpointing
Add checkpointing to the loop inside your recursive expert block wrappers. This frees up activation states by recomputing them on backward pass:
```python
from torch.utils.checkpoint import checkpoint

# Inside your model forward loop (for blocks):
# Replace: h, bal_loss, halt_loss, ... = block(h, mask)
# With a checkpoint wrapper:
def create_custom_forward(block_module):
    def custom_forward(hidden_states, mask_tensor):
        return block_module(hidden_states, mask_tensor)
    return custom_forward

# Run forward pass with activation checkpointing
h, bal_loss, halt_loss, _, _, _, _, _ = checkpoint(
    create_custom_forward(block), 
    h, 
    mask, 
    use_reentrant=False
)
```

---

## 6. Step-by-Step Setup & Running the Job

Once your instance is running:

### Step 1: Install Dependencies
Open the container terminal and run:
```bash
pip install torch transformers datasets bitsandbytes wandb
```

### Step 2: Authenticate Tools
Log in to Weights & Biases (Wandb) to monitor your training metrics remotely:
```bash
wandb login
```

### Step 3: Run Training in the Background
To prevent your training job from stopping when you close your SSH terminal connection, run the script in the background using `nohup` or `screen`:

```bash
# Run in background, redirect outputs to train.log, and write PID to pid.txt
nohup python -u code/train.py --config code/config.json > train.log 2>&1 & echo $! > train.pid
```

*   To monitor logs in real-time: `tail -f train.log`
*   To check if training is running: `ps -ef | grep python`
*   To terminate training: `kill $(cat train.pid)`

### Step 4: Setting up Auto-Resuming Checkpoints
Since spot instances can be terminated unexpectedly, write your checkpoint saving logic to save the optimizer state as well:

```python
# Save complete state dict
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}
torch.save(checkpoint, 'checkpoints/latest_checkpoint.pt')
```

And check for checkpoints at startup to auto-resume:
```python
import os
if os.path.exists('checkpoints/latest_checkpoint.pt'):
    checkpoint = torch.load('checkpoints/latest_checkpoint.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    print(f"Resumed training from epoch {start_epoch}")
```
