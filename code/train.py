"""
train.py — Mixture of Recursive Experts (MoRE) Training Script
==============================================================
Architecture:
  • 7 feedforward expert networks, one per operation family
    (E1: ADD/SUB, E2: MULT/DIV, E3: MOD/POW, E4: LOGIC,
     E5: SHIFT, E6: SORT/STAT, E7: CHAIN)
  • Top-1 routing per token via a learned linear router
  • Halting router per token: exits early when halt_prob > 0.5
  • 2 physical MoRE blocks stacked end-to-end
  • Reads all hyperparameters from config.json (agent-modifiable)
  • Logs to W&B and writes best validation loss to results.tsv

Dataset (dummy.jsonl format):
  Each line is a JSON object with keys:
    input   : human-readable expression string
    steps   : list of {op, args, result}
    output  : final scalar (or list) answer
    family  : "E1" … "E7"  (ground-truth expert label)
    depth   : number of operation steps (1–7)
    id      : record identifier

Tokenisation strategy (no external vocab packages):
  - Flatten the numeric arguments from every step into a fixed-length
    float vector of length SEQ_LEN.
  - Normalise by max_val so all values are in [-1, 1].
  - Family label → integer class index 0–6.
  - Target output → scalar float (or mean of list for multi-value outputs).
"""

import os
import sys
import json
import time
import math
import argparse
import collections
import itertools
import traceback

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import wandb

# ---------------------------------------------------------------------------
# 0. Configuration loader
# ---------------------------------------------------------------------------

def load_config(path: str = "config.json") -> dict:
    """Load and validate the agent-modifiable config file."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"config.json not found at '{path}'. "
            "Create it before running train.py."
        )
    with open(path, "r") as f:
        cfg = json.load(f)
    # Provide safe defaults for any key the agent may have deleted
    cfg.setdefault("model", {})
    cfg["model"].setdefault("d_model", 256)
    cfg["model"].setdefault("num_experts", 7)
    cfg["model"].setdefault("max_depth", 7)
    cfg["model"].setdefault("num_blocks", 2)
    cfg["model"].setdefault("dropout", 0.1)

    cfg.setdefault("training", {})
    cfg["training"].setdefault("lr", 1e-3)
    cfg["training"].setdefault("weight_decay", 1e-4)
    cfg["training"].setdefault("batch_size", 128)
    cfg["training"].setdefault("seq_len", 64)
    cfg["training"].setdefault("epochs", 50)
    cfg["training"].setdefault("val_split", 0.1)
    cfg["training"].setdefault("grad_clip", 1.0)

    cfg.setdefault("loss_weights", {})
    cfg["loss_weights"].setdefault("task", 1.0)
    cfg["loss_weights"].setdefault("routing_balance", 0.01)
    cfg["loss_weights"].setdefault("halting", 0.001)

    cfg.setdefault("data", {})
    cfg["data"].setdefault("jsonl_path", "dummy.jsonl")
    cfg["data"].setdefault("max_val", 1e6)
    cfg["data"].setdefault("pad_value", 0.0)

    cfg.setdefault("logging", {})
    cfg["logging"].setdefault("wandb_project", "micro-MoRE-poc")
    cfg["logging"].setdefault("log_interval", 10)

    return cfg


# ---------------------------------------------------------------------------
# 1. Dataset
# ---------------------------------------------------------------------------

FAMILY_TO_IDX = {"E1": 0, "E2": 1, "E3": 2, "E4": 3, "E5": 4, "E6": 5, "E7": 6}

class MoREDataset(Dataset):
    """
    Reads dummy.jsonl and vectorises each record into a fixed-length
    float tensor without using any external vocabulary libraries.

    Encoding:
      - Collect all numeric values from every step's args and result.
      - Flatten to a 1-D list, clamp to max_val, normalise to [-1, 1].
      - Pad / truncate to seq_len.
      - family label → integer (0-6) for cross-entropy routing supervision.
      - output scalar → float for regression target.
    """

    def __init__(self, jsonl_path: str, seq_len: int, max_val: float, pad_value: float):
        self.seq_len = seq_len
        self.max_val = max(max_val, 1.0)   # guard against 0
        self.pad_value = pad_value
        self.records = []

        if not os.path.exists(jsonl_path):
            raise FileNotFoundError(
                f"Dataset file not found: '{jsonl_path}'. "
                "Place dummy.jsonl alongside train.py or set data.jsonl_path in config.json."
            )

        with open(jsonl_path, "r") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"[WARN] Skipping malformed JSON on line {line_no}: {e}")
                    continue

                # --- Vectorise arguments --------------------------------
                nums = []
                for step in rec.get("steps", []):
                    args = step.get("args", [])
                    result = step.get("result", 0)
                    # args can be scalars or lists (E6/E7 multi-arg)
                    for a in args:
                        if isinstance(a, (int, float)):
                            nums.append(float(a))
                        elif isinstance(a, list):
                            nums.extend(float(x) for x in a)
                    if isinstance(result, (int, float)):
                        nums.append(float(result))
                    elif isinstance(result, list):
                        nums.extend(float(x) for x in result)

                # --- Target scalar --------------------------------------
                raw_out = rec.get("output", 0)
                if isinstance(raw_out, list):
                    target_val = float(sum(raw_out)) / max(len(raw_out), 1)
                else:
                    target_val = float(raw_out)

                # Clamp and normalise
                nums = [max(-self.max_val, min(self.max_val, v)) for v in nums]
                nums = [v / self.max_val for v in nums]

                # Pad / truncate to seq_len
                if len(nums) < seq_len:
                    nums += [pad_value] * (seq_len - len(nums))
                else:
                    nums = nums[:seq_len]

                family_idx = FAMILY_TO_IDX.get(rec.get("family", "E7"), 6)
                depth = int(rec.get("depth", 1))

                self.records.append({
                    "x": torch.tensor(nums, dtype=torch.float32),
                    "family": torch.tensor(family_idx, dtype=torch.long),
                    "depth": torch.tensor(depth, dtype=torch.long),
                    "target": torch.tensor(
                        max(-self.max_val, min(self.max_val, target_val)) / self.max_val,
                        dtype=torch.float32,
                    ),
                })

        if len(self.records) == 0:
            raise RuntimeError(
                "Dataset is empty after parsing. Check your JSONL file."
            )
        print(f"[Dataset] Loaded {len(self.records)} records from '{jsonl_path}'.")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        r = self.records[idx]
        return r["x"], r["family"], r["depth"], r["target"]


# ---------------------------------------------------------------------------
# 2. Model
# ---------------------------------------------------------------------------

class MoEBlock(nn.Module):
    """
    Single Mixture-of-Experts block with Top-1 routing.

    Contains `num_experts` independent 2-layer feedforward networks
    (one per operation family).  A learned linear router assigns each
    token to exactly one expert.  An entropy-based auxiliary loss
    encourages uniform expert utilisation.

    Args:
        num_experts : number of expert networks (default 7)
        d_model     : hidden / embedding dimension
        dropout     : dropout probability applied inside each expert
    """

    def __init__(self, num_experts: int = 7, d_model: int = 256, dropout: float = 0.1):
        super().__init__()
        self.num_experts = num_experts
        self.d_model = d_model

        # 7 independent FFN experts
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 4, d_model),
                nn.Dropout(dropout),
            )
            for _ in range(num_experts)
        ])

        # Top-1 router: projects hidden state → num_experts logits
        self.router = nn.Linear(d_model, num_experts, bias=False)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x : [N, d_model]  (flattened active tokens)
        Returns:
            out         : [N, d_model]  expert-processed tokens
            balance_loss: scalar — entropy balancing auxiliary loss
            expert_idx  : [N] long tensor of chosen expert per token
        """
        router_logits = self.router(x)                         # [N, E]
        router_probs  = F.softmax(router_logits, dim=-1)       # [N, E]
        expert_idx    = torch.argmax(router_probs, dim=-1)     # [N]

        out = torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            mask = (expert_idx == i)
            if mask.any():
                out[mask] = expert(x[mask])

        # Entropy-based load-balance loss: maximise entropy → uniform routing
        avg_probs    = router_probs.mean(dim=0)                # [E]
        balance_loss = -torch.sum(avg_probs * torch.log(avg_probs + 1e-8))

        return out, balance_loss, expert_idx


class MoREWrapper(nn.Module):
    """
    Wraps a MoEBlock in an active-token-mask loop of up to `max_depth`
    iterations.  Each depth step:
      1. Processes only ACTIVE tokens through the MoEBlock (Top-1).
      2. Updates their hidden state with a residual connection.
      3. A halting router outputs halt_prob ∈ (0,1) per active token.
         Tokens with halt_prob > 0.5 are locked into `final_output`
         and evicted from the active set.
      4. Any tokens still active when depth == max_depth are force-exited.

    Gradients flow through every depth step because we use boolean masks
    (no in-place scatter on leaf tensors with grad).

    Args:
        d_model     : hidden dimension
        max_depth   : maximum recursion depth (default 7)
        num_experts : number of expert networks (default 7)
        dropout     : dropout in experts
    """

    def __init__(
        self,
        d_model: int = 256,
        max_depth: int = 7,
        num_experts: int = 7,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.max_depth   = max_depth
        self.num_experts = num_experts
        self.d_model     = d_model

        self.moe_block      = MoEBlock(num_experts, d_model, dropout)
        self.halting_router = nn.Linear(d_model, 1)

        # LayerNorm applied after residual at each depth step
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x : [B, S, d_model]
        Returns:
            final_output    : [B, S, d_model]
            total_bal_loss  : scalar
            total_halt_loss : scalar
            depth_exits     : [B*S, max_depth] float — 1 where token exited
            all_expert_idx  : list[Tensor[N_active]] per depth step
            avg_depth       : scalar float (mean recursion steps used)
        """
        B, S, D = x.shape
        N = B * S

        flat_x        = x.reshape(N, D)
        current_state = flat_x.clone()

        # Accumulated output buffer — accumulate contributions via masking
        final_output = torch.zeros_like(flat_x)

        # Boolean tracker: True = token is still recursing
        active_mask = torch.ones(N, dtype=torch.bool, device=x.device)

        total_bal_loss  = torch.tensor(0.0, device=x.device)
        total_halt_loss = torch.tensor(0.0, device=x.device)

        # [N, max_depth] — records which step each token exited
        depth_exits    = torch.zeros(N, self.max_depth, device=x.device)
        all_expert_idx = []                # list of [N_active] per step
        tokens_per_step = []               # number of active tokens each step

        for depth in range(1, self.max_depth + 1):
            if not active_mask.any():
                break

            # ---- 1. Gather active tokens --------------------------------
            active_inputs = current_state[active_mask]   # [N_active, D]
            N_active = active_inputs.shape[0]
            tokens_per_step.append(N_active)

            # ---- 2. MoE forward ----------------------------------------
            moe_out, b_loss, expert_idx = self.moe_block(active_inputs)
            all_expert_idx.append(expert_idx)

            # ---- 3. Residual + LayerNorm --------------------------------
            updated = self.layer_norm(active_inputs + moe_out)

            # Write updated state back (only active positions)
            # We use index-based write to avoid in-place on grad tensors
            active_positions = active_mask.nonzero(as_tuple=True)[0]  # [N_active]
            current_state = current_state.clone()
            current_state[active_positions] = updated

            total_bal_loss = total_bal_loss + b_loss

            # ---- 4. Halting decision -----------------------------------
            halt_logits = self.halting_router(updated).squeeze(-1)  # [N_active]
            halt_probs  = torch.sigmoid(halt_logits)                # [N_active]

            # Force exit if this is the last allowed depth
            if depth == self.max_depth:
                stop_local = torch.ones(N_active, dtype=torch.bool, device=x.device)
            else:
                stop_local = halt_probs > 0.5                       # [N_active]

            # ---- 5. Lock in exiting tokens ----------------------------
            stop_global = active_positions[stop_local]              # indices in [N]
            final_output = final_output.clone()
            final_output[stop_global] = current_state[stop_global]
            depth_exits[stop_global, depth - 1] = 1.0

            # ---- 6. Evict stopped tokens from active set ---------------
            active_mask = active_mask.clone()
            active_mask[stop_global] = False

            # Penalise late halting — encourage early exit when possible
            total_halt_loss = total_halt_loss + halt_probs.mean() * 0.01

        # --- Compute average recursion depth ----------------------------
        exit_steps   = (depth_exits * torch.arange(
            1, self.max_depth + 1, device=x.device, dtype=torch.float32
        ).unsqueeze(0)).sum(dim=-1)            # [N]
        avg_depth    = exit_steps.mean()

        return (
            final_output.reshape(B, S, D),
            total_bal_loss,
            total_halt_loss,
            depth_exits,
            all_expert_idx,
            avg_depth,
        )


class MoREModel(nn.Module):
    """
    Full MoRE model stacking `num_blocks` MoREWrapper layers.

    Input projection: seq_len → d_model (treats the sequence as a
    single token to keep parameter count in the 10–20M range).
    After MoRE blocks, a linear head maps d_model → 1 for regression
    and d_model → num_experts for auxiliary routing classification.

    Total params target: ~10–20M.
    """

    def __init__(
        self,
        seq_len: int,
        d_model: int = 256,
        num_experts: int = 7,
        max_depth: int = 7,
        num_blocks: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model

        # Input projection: flat vector → [B, 1, d_model]
        self.input_proj = nn.Sequential(
            nn.Linear(seq_len, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        # Stack of MoRE blocks
        self.blocks = nn.ModuleList([
            MoREWrapper(d_model, max_depth, num_experts, dropout)
            for _ in range(num_blocks)
        ])

        # Regression head: predict normalised output scalar
        self.regression_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

        # Auxiliary classification head: predict expert family label
        self.cls_head = nn.Linear(d_model, num_experts)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x : [B, seq_len]
        Returns:
            reg_out         : [B, 1]
            cls_out         : [B, num_experts]
            total_bal_loss  : scalar
            total_halt_loss : scalar
            depth_exits     : [B, max_depth]  (from the last block)
            avg_depth       : scalar float (from the last block)
        """
        # [B, seq_len] → [B, 1, d_model]
        h = self.input_proj(x).unsqueeze(1)

        total_bal_loss  = torch.tensor(0.0, device=x.device)
        total_halt_loss = torch.tensor(0.0, device=x.device)
        depth_exits_last = None
        avg_depth_last   = None

        for block in self.blocks:
            h, b_loss, h_loss, depth_exits, _, avg_depth = block(h)
            total_bal_loss  = total_bal_loss  + b_loss
            total_halt_loss = total_halt_loss + h_loss
            depth_exits_last = depth_exits
            avg_depth_last   = avg_depth

        # [B, 1, d_model] → [B, d_model]
        h = h.squeeze(1)

        reg_out = self.regression_head(h)           # [B, 1]
        cls_out = self.cls_head(h)                  # [B, num_experts]

        return (
            reg_out,
            cls_out,
            total_bal_loss,
            total_halt_loss,
            depth_exits_last,
            avg_depth_last,
        )


# ---------------------------------------------------------------------------
# 3. Metric utilities
# ---------------------------------------------------------------------------

def compute_expert_load_entropy(
    depth_exits: torch.Tensor,
    all_expert_idx: list,
    num_experts: int,
) -> torch.Tensor:
    """
    Compute Shannon entropy of expert utilisation across the batch.
    Higher entropy = more uniform routing = healthier MoE.
    """
    counts = torch.zeros(num_experts, device=depth_exits.device)
    for idx_tensor in all_expert_idx:
        for e in range(num_experts):
            counts[e] += (idx_tensor == e).sum().float()
    total = counts.sum().clamp(min=1.0)
    probs = counts / total
    entropy = -(probs * torch.log(probs + 1e-8)).sum()
    return entropy


def compute_max_pairwise_cosine_sim(model: MoREModel) -> float:
    """
    Compute the maximum cosine similarity between any pair of expert w1
    weight vectors across ALL MoRE blocks.  Approaches 1.0 → experts
    are collapsing to identical representations.
    """
    max_sim = -1.0
    for block in model.blocks:
        weights = [
            exp[0].weight.detach().flatten()          # first Linear in Sequential
            for exp in block.moe_block.experts
        ]
        for i, j in itertools.combinations(range(len(weights)), 2):
            sim = F.cosine_similarity(
                weights[i].unsqueeze(0), weights[j].unsqueeze(0)
            ).item()
            if sim > max_sim:
                max_sim = sim
    return max_sim


# ---------------------------------------------------------------------------
# 4. Training loop
# ---------------------------------------------------------------------------

def train(cfg: dict, run_epochs: int | None = None):
    """
    Full training procedure.

    Args:
        cfg        : parsed config dict
        run_epochs : override epoch count (used by verify_pipeline.py for
                     short diagnostic runs)
    """
    mc  = cfg["model"]
    tc  = cfg["training"]
    lw  = cfg["loss_weights"]
    dc  = cfg["data"]
    log = cfg["logging"]

    epochs    = run_epochs if run_epochs is not None else tc["epochs"]
    batch_sz  = tc["batch_size"]
    seq_len   = tc["seq_len"]
    lr        = tc["lr"]
    val_split = tc["val_split"]
    grad_clip = tc["grad_clip"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Device: {device}")

    # ---- Dataset -------------------------------------------------------
    dataset = MoREDataset(
        jsonl_path=dc["jsonl_path"],
        seq_len=seq_len,
        max_val=dc["max_val"],
        pad_value=dc["pad_value"],
    )
    n_val   = max(1, int(len(dataset) * val_split))
    n_train = len(dataset) - n_val
    if n_train < 1:
        n_train = len(dataset)
        n_val   = 0

    if n_val > 0:
        train_ds, val_ds = random_split(
            dataset, [n_train, n_val],
            generator=torch.Generator().manual_seed(42),
        )
    else:
        train_ds = dataset
        val_ds   = None

    # Use num_workers=0 to avoid multiprocessing overhead on small datasets
    train_loader = DataLoader(
        train_ds, batch_size=min(batch_sz, n_train),
        shuffle=True, num_workers=0, pin_memory=(device.type == "cuda")
    )
    val_loader = (
        DataLoader(val_ds, batch_size=min(batch_sz, n_val),
                   shuffle=False, num_workers=0)
        if val_ds is not None else None
    )

    # ---- Model ---------------------------------------------------------
    model = MoREModel(
        seq_len=seq_len,
        d_model=mc["d_model"],
        num_experts=mc["num_experts"],
        max_depth=mc["max_depth"],
        num_blocks=mc["num_blocks"],
        dropout=mc["dropout"],
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[Model] Total parameters: {total_params:,}")

    # ---- Optimiser & Scheduler -----------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=tc["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs
    )

    # ---- W&B -----------------------------------------------------------
    wandb.init(
        project=log["wandb_project"],
        config={**mc, **tc, **lw, **dc},
        reinit=True,
    )
    wandb.watch(model, log="gradients", log_freq=50)

    best_val_loss = float("inf")
    results_rows  = []

    # ---- Training epochs -----------------------------------------------
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_total  = 0.0
        epoch_task   = 0.0
        epoch_bal    = 0.0
        epoch_halt   = 0.0
        epoch_start  = time.perf_counter()
        total_tokens = 0
        depth_hist   = torch.zeros(mc["max_depth"])

        for batch_idx, (x, family, depth, target) in enumerate(train_loader):
            x, family, target = (
                x.to(device), family.to(device), target.to(device)
            )
            depth = depth.to(device)

            t0 = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)

            (
                reg_out, cls_out,
                bal_loss, halt_loss,
                depth_exits, avg_depth,
            ) = model(x)

            # --- Losses -------------------------------------------------
            # 1. Primary regression loss
            task_loss = F.mse_loss(reg_out.squeeze(-1), target)

            # 2. Auxiliary family classification (supervision signal)
            cls_loss  = F.cross_entropy(cls_out, family)

            # 3. Routing balance + halting
            total_loss = (
                lw["task"] * (task_loss + 0.5 * cls_loss)
                + lw["routing_balance"] * bal_loss
                + lw["halting"] * halt_loss
            )

            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            # Accumulate metrics
            bs = x.shape[0]
            epoch_total  += total_loss.item()
            epoch_task   += task_loss.item()
            epoch_bal    += bal_loss.item()
            epoch_halt   += halt_loss.item()
            total_tokens += bs
            if depth_exits is not None:
                depth_hist += depth_exits.detach().cpu().sum(dim=0)

        scheduler.step()
        elapsed = time.perf_counter() - epoch_start
        throughput = total_tokens / max(elapsed, 1e-6)
        n_batches  = max(len(train_loader), 1)

        # ---- Validation ------------------------------------------------
        val_loss = float("nan")
        if val_loader is not None and (epoch % log["log_interval"] == 0 or epoch == epochs):
            model.eval()
            val_total = 0.0
            val_n     = 0
            with torch.no_grad():
                for x_v, fam_v, _, tgt_v in val_loader:
                    x_v, fam_v, tgt_v = (
                        x_v.to(device), fam_v.to(device), tgt_v.to(device)
                    )
                    reg_v, cls_v, b_v, h_v, _, _ = model(x_v)
                    l_v = F.mse_loss(reg_v.squeeze(-1), tgt_v)
                    val_total += l_v.item() * x_v.shape[0]
                    val_n     += x_v.shape[0]
            val_loss = val_total / max(val_n, 1)

        # ---- Expert metrics (no grad) ----------------------------------
        with torch.no_grad():
            max_cos = compute_max_pairwise_cosine_sim(model)

        # Depth histogram as fractions
        depth_total = depth_hist.sum().clamp(min=1.0)
        depth_frac  = (depth_hist / depth_total).tolist()

        # Load entropy from depth_exits (proxy; full per-step calc only in val)
        # We re-use the last batch's depth_exits for the epoch log
        if depth_exits is not None:
            load_counts = depth_exits.sum(dim=1).float()   # tokens per step
            load_probs  = load_counts / load_counts.sum().clamp(min=1.0)
            load_entropy = -(load_probs * torch.log(load_probs + 1e-8)).sum().item()
        else:
            load_entropy = float("nan")

        # ---- W&B logging -----------------------------------------------
        log_dict = {
            "train/total_loss":       epoch_total / n_batches,
            "train/task_loss":        epoch_task  / n_batches,
            "train/aux_routing_loss": epoch_bal   / n_batches,
            "train/halting_loss":     epoch_halt  / n_batches,
            "train/expert_load_entropy": load_entropy,
            "train/avg_recursion_steps": (
                avg_depth.item() if avg_depth is not None else float("nan")
            ),
            "diag/max_pairwise_cosine_sim": max_cos,
            "perf/throughput_tokens_sec": throughput,
            "epoch": epoch,
        }

        # Depth distribution histogram (W&B native histogram)
        depth_hist_wandb = {
            f"depth_dist/step_{d+1}_pct": depth_frac[d] * 100.0
            for d in range(mc["max_depth"])
        }
        log_dict.update(depth_hist_wandb)

        if not math.isnan(val_loss):
            log_dict["val/loss"] = val_loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), "best_model.pt")

        wandb.log(log_dict, step=epoch)

        # Append to results.tsv (autoresearch_runner.py reads this)
        results_rows.append(
            f"{epoch}\t{epoch_task/n_batches:.6f}\t{val_loss:.6f}\t"
            f"{load_entropy:.4f}\t"
            f"{avg_depth.item() if avg_depth is not None else 0:.2f}\t"
            f"{max_cos:.4f}"
        )

        # Flush results to disk every epoch so runner can read partial runs
        with open("results.tsv", "w") as f:
            f.write(
                "epoch\ttrain_task_loss\tval_loss\texpert_entropy\t"
                "avg_depth\tmax_cos_sim\n"
            )
            f.write("\n".join(results_rows) + "\n")

        if epoch % log["log_interval"] == 0 or epoch == epochs:
            print(
                f"[Epoch {epoch:03d}/{epochs}] "
                f"task={epoch_task/n_batches:.4f}  "
                f"bal={epoch_bal/n_batches:.4f}  "
                f"halt={epoch_halt/n_batches:.4f}  "
                f"val={val_loss:.4f}  "
                f"entropy={load_entropy:.3f}  "
                f"cos={max_cos:.3f}  "
                f"tok/s={throughput:.0f}"
            )

    print(f"\n[Train] Complete. Best val_loss = {best_val_loss:.6f}")
    wandb.finish()
    return best_val_loss


# ---------------------------------------------------------------------------
# 5. Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MoRE model")
    parser.add_argument(
        "--config", default="config.json",
        help="Path to config.json (default: config.json)"
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override epochs from config (used for diagnostic runs)"
    )
    args = parser.parse_args()

    try:
        cfg = load_config(args.config)
        train(cfg, run_epochs=args.epochs)
    except Exception as e:
        print(f"\n[FATAL] Training failed: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
