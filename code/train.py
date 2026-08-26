"""
train.py — Mixture of Recursive Experts (MoRE) Training Script  v2
==================================================================
Architecture (v2 — per-step tokenisation):
  • 7 feedforward expert networks, one per operation family
    (E0: ADD/SUB, E1: MULT/DIV, E2: MOD/POW, E3: LOGIC,
     E4: SHIFT, E5: SORT/STAT, E6: catch-all for unknown ops)
  • Per-step tokenisation: each computation step → its own token.
    Input shape: [B, max_steps, step_feat_dim]  (default [B, 7, 12])
    S = number of real steps (1–7), not always 1.
  • Top-1 routing per token via a learned linear router.
  • Per-expert halt heads: each expert learns its own halt threshold.
    Expert 0 (ADD/SUB) learns to halt fast; Expert 5 (SORT) recurses.
  • step_routing_ce loss: CrossEntropy on per-step oracle expert labels,
    teaching the router "ADD tokens → Expert 0, MULT → Expert 1, etc."
  • balance_loss (soft-entropy + Switch-aux): encourages uniform load.
  • 2 physical MoRE blocks stacked end-to-end.
  • Reads all hyperparameters from config.json (agent-modifiable).
  • Logs to W&B; writes best validation loss to results.tsv.
  • Paper metrics (val set, every log_interval epoch):
      - paper/routing_confusion_matrix — 7×7 oracle vs router heatmap
      - paper/op_avg_recursion_depth — operation vs avg exit depth bar chart
      - val/routing_accuracy — diagonal fraction of confusion matrix
      - recursion/avg_depth_by_op/{OP} — per-operation depth scalars

Dataset (dummy.jsonl format):
  Each line is a JSON object with keys:
    input   : human-readable expression string
    steps   : list of {op, args, result}
    output  : final scalar (or list) answer
    family  : "E1" … "E7"  (ground-truth expert label)
    depth   : number of operation steps (1–7)
    id      : record identifier

Tokenisation strategy (per-step, no external vocab packages):
  - Each step → row: [op_expert_norm, arg0/max, arg1/max, ..., result/max, 0...]
    of length step_feat_dim (default 12).
    Slot 0   : OP_TO_EXPERT[op] / (num_experts-1)  ∈ [0,1]  — op-type signal.
    Slots 1..: up to (step_feat_dim-1) normalised numeric values from
               flatten(args) + [scalar_result], truncated to fit.
    Remaining: 0.0 padding.
  - Stack real step rows; pad to [max_steps, step_feat_dim] with zero rows.
  - step_mask [max_steps] bool: True for real steps, False for pad rows.
  - step_experts [max_steps] long: OP_TO_EXPERT[op] per real step, -1 for pad.
  - Family label → integer (0-6) for whole-program cls_head supervision.
  - Target output → normalised scalar float.
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

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
    cfg["model"].setdefault("max_steps", 7)        # number of step-tokens per program
    cfg["model"].setdefault("step_feat_dim", 12)   # features per step-token
    cfg["model"].setdefault("num_blocks", 2)
    cfg["model"].setdefault("dropout", 0.1)

    cfg.setdefault("training", {})
    cfg["training"].setdefault("lr", 1e-3)
    cfg["training"].setdefault("weight_decay", 1e-4)
    cfg["training"].setdefault("batch_size", 128)
    cfg["training"].setdefault("epochs", 50)
    cfg["training"].setdefault("val_split", 0.1)
    cfg["training"].setdefault("grad_clip", 1.0)
    cfg["training"].setdefault("subset_fraction", 1.0)
    cfg["training"].setdefault("subset_seed", 42)

    cfg.setdefault("loss_weights", {})
    cfg["loss_weights"].setdefault("task", 1.0)
    cfg["loss_weights"].setdefault("routing_balance", 0.01)
    cfg["loss_weights"].setdefault("halting", 0.001)
    cfg["loss_weights"].setdefault("step_routing", 0.5)   # per-step oracle CE weight

    cfg.setdefault("data", {})
    cfg["data"].setdefault("train_path", "../data/train.jsonl")
    cfg["data"].setdefault("val_path", "../data/val.jsonl")
    cfg["data"].setdefault("test_path", "../data/test.jsonl")
    cfg["data"].setdefault("jsonl_path", cfg["data"]["train_path"])
    cfg["data"].setdefault("subset_fraction", 1.0)
    cfg["data"].setdefault("subset_seed", 42)
    cfg["data"].setdefault("max_val", 1e6)
    cfg["data"].setdefault("pad_value", 0.0)

    cfg.setdefault("logging", {})
    cfg["logging"].setdefault("wandb_project", "micro-MoRE-poc")
    cfg["logging"].setdefault("log_interval", 10)

    # Backward-compatible subset config support:
    # keep existing fast_config layout (training.*) authoritative when present.
    if "subset_fraction" in cfg["training"]:
        cfg["data"]["subset_fraction"] = cfg["training"]["subset_fraction"]
    if "subset_seed" in cfg["training"]:
        cfg["data"]["subset_seed"] = cfg["training"]["subset_seed"]

    # Resolve relative dataset paths against the config file directory.
    cfg_dir = os.path.dirname(os.path.abspath(path))
    for key in ("train_path", "val_path", "test_path", "jsonl_path"):
        p = cfg["data"].get(key)
        if isinstance(p, str) and p and not os.path.isabs(p):
            cfg["data"][key] = os.path.normpath(os.path.join(cfg_dir, p))

    return cfg


# ---------------------------------------------------------------------------
# 1. Dataset
# ---------------------------------------------------------------------------

FAMILY_TO_IDX = {"E1": 0, "E2": 1, "E3": 2, "E4": 3, "E5": 4, "E6": 5, "E7": 6}

# Per-operation expert index: maps each op-code to its expert (0-6).
# Operations within the same family share an expert index.
# Expert 6 is the catch-all for any op not explicitly listed (E7 chains).
OP_TO_EXPERT = {
    "ADD":     0, "SUB":     0,                    # E0 ≡ E1 family (ADD/SUB)
    "MULT":    1, "DIV":     1,                    # E1 ≡ E2 family (MULT/DIV)
    "MOD":     2, "POW":     2,                    # E2 ≡ E3 family (MOD/POW)
    "AND":     3, "OR":      3,
    "XOR":     3, "NOT":     3,                    # E3 ≡ E4 family (LOGIC)
    "SHIFT_L": 4, "SHIFT_R": 4,                    # E4 ≡ E5 family (SHIFT)
    "SORT":    5, "MEDIAN":  5,
    "MAX":     5, "MIN":     5,                    # E5 ≡ E6 family (SORT/STAT)
}
_EXPERT_FALLBACK = 6   # catch-all expert for any op not in OP_TO_EXPERT

# Paper-figure labels: row/column names for the 7×7 routing confusion matrix.
EXPERT_FAMILY_LABELS = [
    "E1 ADD/SUB",
    "E2 MULT/DIV",
    "E3 MOD/POW",
    "E4 LOGIC",
    "E5 SHIFT",
    "E6 SORT/STAT",
    "E7 CHAIN/OTHER",
]

# Per-operation labels for the recursion-depth bar chart (sorted for stable indexing).
ALL_OP_NAMES = sorted(OP_TO_EXPERT.keys())
OP_NAME_TO_IDX = {name: i for i, name in enumerate(ALL_OP_NAMES)}
NUM_OP_TYPES = len(ALL_OP_NAMES)


class MoREDataset(Dataset):
    """
    Reads dummy.jsonl and encodes each record as a per-step token matrix.

    Encoding (per-step, structured):
      Each step → row [step_feat_dim]:
        Slot 0   : OP_TO_EXPERT[op] / (num_experts-1) ∈ [0,1]  — op-type signal
        Slots 1..: flatten(args) + [scalar_result], normalised, truncated to fit
        Remaining: 0.0
      Stack real step rows; pad to [max_steps, step_feat_dim] with zero rows.

    Returns 6-tuple per record:
      x            [max_steps, step_feat_dim]  float  — per-step token matrix
      step_mask    [max_steps]                 bool   — True = real step
      step_experts [max_steps]                 long   — oracle expert per step (-1=pad)
      step_ops     [max_steps]                 long   — op-type index per step (-1=pad)
      family       scalar long                        — whole-program family (0-6)
      depth        scalar long                        — number of real steps
      target       scalar float                       — normalised output value
    """

    def __init__(
        self,
        jsonl_path: str,
        max_steps: int,
        step_feat_dim: int,
        max_val: float,
        pad_value: float,
        num_experts: int = 7,
    ):
        self.max_steps     = max_steps
        self.step_feat_dim = step_feat_dim
        self.max_val       = max(max_val, 1.0)   # guard against 0
        self.pad_value     = pad_value
        self.num_experts   = num_experts
        self.records       = []

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

                steps = rec.get("steps", [])
                num_real_steps = min(len(steps), max_steps)

                # --- Per-step token matrix ----------------------------------
                x_rows       = []
                step_exp_ids = []
                step_op_ids  = []

                for s_idx in range(num_real_steps):
                    step   = steps[s_idx]
                    op     = step.get("op", "")
                    args   = step.get("args", [])
                    result = step.get("result", 0)

                    # Op-code → expert index
                    expert_id = OP_TO_EXPERT.get(op, _EXPERT_FALLBACK)
                    op_id     = OP_NAME_TO_IDX.get(op, -1)

                    # Flatten args (handle scalar or list args for SORT/STAT)
                    flat_args = []
                    for a in args:
                        if isinstance(a, (int, float)):
                            flat_args.append(float(a))
                        elif isinstance(a, list):
                            flat_args.extend(float(v) for v in a)

                    # Scalar result (mean of list for SORT output etc.)
                    if isinstance(result, list):
                        scalar_result = float(sum(result)) / max(len(result), 1)
                    else:
                        scalar_result = float(result)

                    # Clamp + normalise: all values into [-1, 1]
                    numeric_vals = flat_args + [scalar_result]
                    numeric_vals = [
                        max(-self.max_val, min(self.max_val, v)) / self.max_val
                        for v in numeric_vals
                    ]

                    # Build row: slot 0 = op-type signal, slots 1.. = numerics
                    row    = [0.0] * step_feat_dim
                    row[0] = expert_id / max(num_experts - 1, 1)      # ∈ [0,1]
                    for i, v in enumerate(numeric_vals[: step_feat_dim - 1]):
                        row[i + 1] = v

                    x_rows.append(row)
                    step_exp_ids.append(expert_id)
                    step_op_ids.append(op_id)

                # --- Build padded tensors -----------------------------------
                x_tensor     = torch.zeros(max_steps, step_feat_dim, dtype=torch.float32)
                step_mask    = torch.zeros(max_steps, dtype=torch.bool)
                step_experts = torch.full((max_steps,), -1, dtype=torch.long)
                step_ops     = torch.full((max_steps,), -1, dtype=torch.long)

                for s_idx in range(num_real_steps):
                    x_tensor[s_idx]     = torch.tensor(x_rows[s_idx], dtype=torch.float32)
                    step_mask[s_idx]    = True
                    step_experts[s_idx] = step_exp_ids[s_idx]
                    step_ops[s_idx]     = step_op_ids[s_idx]

                # --- Target scalar ------------------------------------------
                raw_out = rec.get("output", 0)
                if isinstance(raw_out, list):
                    target_val = float(sum(raw_out)) / max(len(raw_out), 1)
                else:
                    target_val = float(raw_out)
                target_val = (
                    max(-self.max_val, min(self.max_val, target_val)) / self.max_val
                )

                family_idx = FAMILY_TO_IDX.get(rec.get("family", "E7"), 6)
                depth      = int(rec.get("depth", 1))

                self.records.append({
                    "x":            x_tensor,
                    "step_mask":    step_mask,
                    "step_experts": step_experts,
                    "step_ops":     step_ops,
                    "family":       torch.tensor(family_idx, dtype=torch.long),
                    "depth":        torch.tensor(depth,      dtype=torch.long),
                    "target":       torch.tensor(target_val, dtype=torch.float32),
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
        return (
            r["x"],             # [max_steps, step_feat_dim]  float
            r["step_mask"],     # [max_steps]                 bool
            r["step_experts"],  # [max_steps]                 long  (-1 for pad)
            r["step_ops"],      # [max_steps]                 long  (-1 for pad)
            r["family"],        # scalar long
            r["depth"],         # scalar long
            r["target"],        # scalar float
        )


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
        self.d_model     = d_model

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

        # Trainable exploration noise: the optimiser adapts the scale; starts
        # at 0.1 and can grow or shrink. Applied only during training so that
        # inference is deterministic. Breaks early argmax lock-in without
        # changing the Top-1 hard-routing design.
        self.router_noise_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor):
        """
        Args:
            x : [N, d_model]  (flattened active tokens)
        Returns:
            out         : [N, d_model]  expert-processed tokens
            balance_loss: scalar — entropy balancing auxiliary loss
            expert_idx  : [N] long tensor of chosen expert per token
        """
        router_logits = self.router(x)                          # [N, E]
        if self.training:
            # Trainable Gaussian noise injected before softmax/argmax.
            # Prevents the router from permanently locking onto one expert
            # in the first few batches. Gradient flows through noise_scale
            # so the network learns how much exploration it actually needs.
            noise         = torch.randn_like(router_logits) * self.router_noise_scale.abs()
            router_logits = router_logits + noise
        router_probs = F.softmax(router_logits, dim=-1)         # [N, E]
        expert_idx   = torch.argmax(router_probs, dim=-1)       # [N]

        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        out = (router_probs.unsqueeze(-1) * expert_outputs).sum(dim=1)

        # --- Two-term balance loss -------------------------------------------
        # Term 1 – Soft entropy: gradient always present via softmax, but blind
        #   to hard routing collapse because softmax is never exactly one-hot.
        avg_probs    = router_probs.mean(dim=0)                          # [E]
        entropy_term = -torch.sum(avg_probs * torch.log(avg_probs + 1e-8))

        # Term 2 – Switch-Transformer auxiliary loss:
        #   f_e  = fraction of tokens dispatched to expert e (hard argmax,
        #          detached so no gradient through the discrete choice).
        #   P_e  = mean soft router probability for expert e (differentiable).
        #   Loss = E * Σ_e(f_e * P_e)  — minimised at uniform routing (value=1),
        #          maximised at full collapse (value=E=7).  Must be ADDED to total
        #          loss (minimised), not subtracted.  We subtract balance_loss in
        #          total_loss, so we store: balance_loss = entropy_term - switch_aux
        #          → -entropy_term (maximise entropy) + switch_aux (minimise collapse).
        load = torch.zeros(self.num_experts, device=x.device)
        for e in range(self.num_experts):
            load[e] = (expert_idx == e).float().sum()
        load       = load / load.sum().clamp(min=1.0)                    # [E]
        switch_aux = self.num_experts * (load.detach() * avg_probs).sum()

        balance_loss = -entropy_term + switch_aux

        # Return router_logits (pre-softmax, after noise) so callers can
        # apply oracle CE directly on the router — much shorter gradient path
        # than going through step_cls_head after pooling.
        return out, balance_loss, expert_idx, router_logits


class MoREWrapper(nn.Module):
    """
    Wraps a MoEBlock in an active-token-mask loop of up to `max_depth`
    iterations.  Each depth step:
      1. Processes only ACTIVE tokens through the MoEBlock (Top-1).
      2. Updates their hidden state with a residual connection + LayerNorm.
      3. Per-expert halt heads output halt_prob ∈ (0,1) per active token.
         Each expert has its own Linear(d_model, 1):
           Expert 0 (ADD/SUB) → learns to fire at depth=1 (simple, 1-step ops).
           Expert 5 (SORT/STAT) → learns to recurse deeper (complex aggregation).
         Tokens with halt_prob > 0.5 are locked into `final_output`
         and evicted from the active set.
      4. Any tokens still active when depth == max_depth are force-exited.
      5. Pad tokens (step_mask == False) are excluded from depth=1 onward.

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
        fixed_depth: bool = False,
    ):
        super().__init__()
        self.max_depth   = max_depth
        self.num_experts = num_experts
        self.d_model     = d_model
        self.fixed_depth = fixed_depth

        self.moe_block = MoEBlock(num_experts, d_model, dropout)

        # Per-expert halting heads — replaces the single shared halting_router.
        # Each expert independently learns when its type of operation is "done".
        # Expert 0 (ADD/SUB) learns: halt fast   (1-step operations).
        # Expert 5 (SORT/STAT) learns: recurse    (needs multiple refinements).
        self.expert_halt_heads = nn.ModuleList([
            nn.Linear(d_model, 1) for _ in range(num_experts)
        ])

        # LayerNorm applied after residual at each depth step
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        step_mask: torch.Tensor,
        step_experts: torch.Tensor | None = None,
    ):
        """
        Args:
            x            : [B, S, d_model]
            step_mask    : [B, S] bool — True for real steps, False for padding.
                           Pad tokens are permanently excluded from the depth loop.
            step_experts : [B, S] long — oracle expert index per token (-1 for pad).
                           When provided, oracle CE is computed directly on router
                           logits at each depth step (short gradient path to router).
        Returns:
            final_output      : [B, S, d_model]
            total_bal_loss    : scalar
            total_halt_loss   : scalar
            depth_exits       : [B*S, max_depth] float — 1 where token exited
            all_expert_idx    : list[Tensor[N_active]] per depth step
            avg_depth         : scalar float (mean recursion steps, real tokens only)
            oracle_routing_ce : scalar — direct CE on router logits (0 if no labels)
            first_route_flat  : [B*S] long — expert chosen at depth 1 (-1 for pad)
        """
        B, S, D = x.shape
        N = B * S

        flat_x        = x.reshape(N, D)
        current_state = flat_x.clone()

        # Accumulated output buffer — contributions written via boolean masking
        final_output = torch.zeros_like(flat_x)

        # Initialise active_mask from step_mask so pad tokens NEVER enter the loop
        active_mask = step_mask.reshape(N)   # [N] bool — False for pad positions

        # Flat oracle labels — [N], -1 for pad.  None if not provided.
        flat_experts = step_experts.reshape(N) if step_experts is not None else None

        total_bal_loss    = torch.tensor(0.0, device=x.device)
        total_halt_loss   = torch.tensor(0.0, device=x.device)
        oracle_routing_ce = torch.tensor(0.0, device=x.device)
        oracle_depth_count = 0

        # [N, max_depth] — records at which depth step each token exited
        depth_exits     = torch.zeros(N, self.max_depth, device=x.device)
        all_expert_idx  = []          # list of [N_active] per depth step
        tokens_per_step = []          # diagnostic: active count per step
        first_route_flat = torch.full((N,), -1, dtype=torch.long, device=x.device)

        for depth in range(1, self.max_depth + 1):
            if not active_mask.any():
                break

            # ---- 1. Gather active tokens --------------------------------
            active_inputs = current_state[active_mask]    # [N_active, D]
            N_active      = active_inputs.shape[0]
            tokens_per_step.append(N_active)
            active_positions = active_mask.nonzero(as_tuple=True)[0]   # [N_active]

            # ---- 2. MoE forward ----------------------------------------
            moe_out, b_loss, expert_idx, router_logits = self.moe_block(active_inputs)
            all_expert_idx.append(expert_idx)

            if depth == 1:
                first_route_flat = first_route_flat.clone()
                first_route_flat[active_positions] = expert_idx

            # ---- 3. Residual + LayerNorm --------------------------------
            updated = self.layer_norm(active_inputs + moe_out)

            # Write updated state back; clone to avoid in-place on grad tensors
            current_state    = current_state.clone()
            current_state[active_positions] = updated

            total_bal_loss = total_bal_loss + b_loss

            # ---- Oracle routing CE — direct on router logits -----------
            # This is the SHORT gradient path: router_logits → oracle CE.
            # The router learns immediately which expert each op type belongs to,
            # without waiting for gradients to travel back through the full
            # MoREWrapper depth loop via step_cls_head.
            if flat_experts is not None:
                active_oracle = flat_experts[active_positions]    # [N_active]
                valid_oracle  = (active_oracle >= 0) & (active_oracle < self.num_experts)
                if valid_oracle.any():
                    oracle_routing_ce = oracle_routing_ce + F.cross_entropy(
                        router_logits[valid_oracle], active_oracle[valid_oracle]
                    )
                    oracle_depth_count += 1

            # ---- 4. Per-expert halting decision --------------------------
            # Dispatch each token to the halt head of the expert that just
            # processed it.  Vectorised: loop over experts same as dispatch.
            halt_logits = torch.zeros(N_active, device=x.device)
            for e, halt_head in enumerate(self.expert_halt_heads):
                emask = (expert_idx == e)
                if emask.any():
                    halt_logits[emask] = halt_head(updated[emask]).squeeze(-1)
            halt_probs = torch.sigmoid(halt_logits)                 # [N_active]

            # Force exit at the last allowed depth
            if self.fixed_depth:
                stop_local = torch.zeros(N_active, dtype=torch.bool, device=x.device) if depth < self.max_depth else torch.ones(N_active, dtype=torch.bool, device=x.device)
            else:
                if depth == self.max_depth:
                    stop_local = torch.ones(N_active, dtype=torch.bool, device=x.device)
                else:
                    stop_local = halt_probs > 0.5                        # [N_active]

            # ---- 5. Lock in exiting tokens ----------------------------
            stop_global = active_positions[stop_local]               # indices in [N]
            final_output = final_output.clone()
            final_output[stop_global] = current_state[stop_global]
            depth_exits[stop_global, depth - 1] = 1.0

            # ---- 6. Evict stopped tokens from active set ---------------
            active_mask = active_mask.clone()
            active_mask[stop_global] = False

            # Penalise late halting — encourage early exit when possible
            total_halt_loss = total_halt_loss + active_mask.float().mean() * 0.05

        # Normalise oracle CE by number of depth steps that contributed labels
        if oracle_depth_count > 0:
            oracle_routing_ce = oracle_routing_ce / oracle_depth_count

        # --- Compute average recursion depth (real tokens only) --------
        real_token_mask = step_mask.reshape(N)   # [N] bool
        exit_steps = (
            depth_exits
            * torch.arange(1, self.max_depth + 1, device=x.device, dtype=torch.float32)
            .unsqueeze(0)
        ).sum(dim=-1)                            # [N]
        real_exit_steps = exit_steps[real_token_mask]
        avg_depth = (
            real_exit_steps.mean()
            if real_exit_steps.numel() > 0
            else torch.tensor(0.0, device=x.device)
        )

        return (
            final_output.reshape(B, S, D),
            total_bal_loss,
            total_halt_loss,
            depth_exits,
            all_expert_idx,
            avg_depth,
            oracle_routing_ce,
            first_route_flat,
        )


class MoREModel(nn.Module):
    """
    Full MoRE model stacking `num_blocks` MoREWrapper layers.

    Input projection:
      step_proj: Linear(step_feat_dim → d_model) + LayerNorm + GELU
      Applied per-row to [B, max_steps, step_feat_dim] → [B, max_steps, d_model].
      S is now real (1–7 steps), NOT always 1.

    After MoRE blocks, masked mean-pooling over real steps → [B, d_model].

    Three output heads:
      regression_head : d_model → 1            — normalised scalar prediction
      cls_head        : d_model → num_experts  — whole-program family class
      step_cls_head   : d_model → num_experts  — per-step expert class
                        (used for step_routing_ce supervision; applied before pool)

    Total params target: ~10–20M.
    """

    def __init__(
        self,
        step_feat_dim: int,
        d_model: int = 256,
        num_experts: int = 7,
        max_depth: int = 7,
        num_blocks: int = 2,
        dropout: float = 0.1,
        fixed_depth: bool = False,
    ):
        super().__init__()
        self.d_model     = d_model
        self.num_experts = num_experts

        # Per-step projection: each step row [step_feat_dim] → [d_model].
        # PyTorch applies Linear to the last dim, so [B, S, F] → [B, S, d_model].
        self.step_proj = nn.Sequential(
            nn.Linear(step_feat_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        # Stack of MoRE blocks
        self.blocks = nn.ModuleList([
            MoREWrapper(d_model, max_depth, num_experts, dropout, fixed_depth)
            for _ in range(num_blocks)
        ])

        # Regression head: predict normalised output scalar (from pooled repr)
        self.regression_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

        # Whole-program auxiliary classification head (whole-program family)
        self.cls_head = nn.Linear(d_model, 7)

        # Per-token step classification head — applied per-step token BEFORE
        # pooling.  Supervised by per-step oracle labels (step_routing_ce loss).
        # This teaches the router: ADD token → Expert 0, MULT → Expert 1, etc.
        self.step_cls_head = nn.Linear(d_model, 7)

    def forward(
        self,
        x: torch.Tensor,
        step_mask: torch.Tensor,
        step_experts: torch.Tensor | None = None,
    ):
        """
        Args:
            x            : [B, max_steps, step_feat_dim]
            step_mask    : [B, max_steps] bool
            step_experts : [B, max_steps] long — oracle expert labels (-1 for pad).
                           Threaded into each MoREWrapper to compute direct oracle
                           CE on router logits (short, clean gradient path).
        Returns:
            reg_out           : [B, 1]
            cls_out           : [B, num_experts]
            step_cls_out      : [B, max_steps, num_experts]  (diagnostic head)
            total_bal_loss    : scalar
            total_halt_loss   : scalar
            depth_exits       : [B*max_steps, max_depth]  (from last block)
            avg_depth         : scalar float              (from last block)
            expert_idx_last   : list of routing indices from last block
            oracle_routing_ce : scalar — direct CE on router logits across all blocks
            first_route_flat  : [B*max_steps] long — block-0 depth-1 router choice
        """
        # [B, max_steps, step_feat_dim] → [B, max_steps, d_model]
        h = self.step_proj(x)

        # Zero out pad positions after projection so they carry no signal
        h = h * step_mask.unsqueeze(-1).float()

        total_bal_loss       = torch.tensor(0.0, device=x.device)
        total_halt_loss      = torch.tensor(0.0, device=x.device)
        total_oracle_routing = torch.tensor(0.0, device=x.device)
        depth_exits_last     = None
        avg_depth_last       = None
        expert_idx_last      = None
        first_route_block0   = None

        for block_idx, block in enumerate(self.blocks):
            (
                h, b_loss, h_loss, depth_exits, block_expert_idx,
                avg_depth, block_oracle_ce, first_route,
            ) = block(h, step_mask, step_experts)
            total_bal_loss       = total_bal_loss       + b_loss
            total_halt_loss      = total_halt_loss      + h_loss
            total_oracle_routing = total_oracle_routing + block_oracle_ce
            depth_exits_last     = depth_exits
            avg_depth_last       = avg_depth
            expert_idx_last      = block_expert_idx
            if block_idx == 0:
                first_route_block0 = first_route

        # Per-token step classification head — diagnostic / secondary supervision.
        # step_cls_head applies to h after MoRE blocks; used to monitor whether
        # the learned representations are separable by expert class.
        step_cls_out = self.step_cls_head(h)   # [B, max_steps, num_experts]

        # Masked mean-pool over real steps (exclude pad positions)
        mask_f   = step_mask.unsqueeze(-1).float()                    # [B, S, 1]
        h_pooled = (h * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)

        reg_out = self.regression_head(h_pooled)   # [B, 1]
        cls_out = self.cls_head(h_pooled)           # [B, num_experts]

        return (
            reg_out,
            cls_out,
            step_cls_out,
            total_bal_loss,
            total_halt_loss,
            depth_exits_last,
            avg_depth_last,
            expert_idx_last,
            total_oracle_routing,
            first_route_block0,
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
    Uses hard argmax assignments, not soft probabilities.
    """
    counts = torch.zeros(num_experts, device=depth_exits.device)
    for idx_tensor in all_expert_idx:
        for e in range(num_experts):
            counts[e] += (idx_tensor == e).sum().float()
    total   = counts.sum().clamp(min=1.0)
    probs   = counts / total
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
            exp[0].weight.detach().flatten()    # first Linear in each expert Sequential
            for exp in block.moe_block.experts
        ]
        for i, j in itertools.combinations(range(len(weights)), 2):
            sim = F.cosine_similarity(
                weights[i].unsqueeze(0), weights[j].unsqueeze(0)
            ).item()
            if sim > max_sim:
                max_sim = sim
    return max_sim


def compute_token_exit_depths(
    depth_exits: torch.Tensor,
    max_depth: int,
) -> torch.Tensor:
    """Convert [N, max_depth] one-hot exit markers into per-token exit depths."""
    steps = torch.arange(
        1, max_depth + 1, device=depth_exits.device, dtype=depth_exits.dtype
    )
    return (depth_exits * steps.unsqueeze(0)).sum(dim=-1)


def evaluate_paper_metrics(
    model: MoREModel,
    loader: DataLoader,
    device: torch.device,
    num_experts: int,
    max_depth: int,
) -> tuple[np.ndarray, float, dict[str, float], dict[str, float]]:
    """
    Accumulate validation-set metrics for paper figures:
      • 7×7 routing confusion matrix (oracle expert vs block-0 depth-1 router)
      • Per-operation average recursion depth (from last MoRE block)
    """
    confusion = torch.zeros(num_experts, num_experts, dtype=torch.float64)
    op_depth_sum   = torch.zeros(NUM_OP_TYPES, dtype=torch.float64)
    op_depth_count = torch.zeros(NUM_OP_TYPES, dtype=torch.float64)
    family_depth_sum   = torch.zeros(num_experts, dtype=torch.float64)
    family_depth_count = torch.zeros(num_experts, dtype=torch.float64)

    model.eval()
    with torch.no_grad():
        for x, step_mask, step_experts, step_ops, _, _, _ in loader:
            x            = x.to(device)
            step_mask    = step_mask.to(device)
            step_experts = step_experts.to(device)
            step_ops     = step_ops.to(device)

            _, _, _, _, _, depth_exits, _, _, _, first_route = model(
                x, step_mask, step_experts
            )
            if first_route is None or depth_exits is None:
                continue

            flat_mask   = step_mask.reshape(-1)
            flat_oracle = step_experts.reshape(-1)
            flat_route  = first_route.reshape(-1)
            flat_ops    = step_ops.reshape(-1)
            exit_depths = compute_token_exit_depths(depth_exits, max_depth).reshape(-1)

            route_valid = flat_mask & (flat_oracle >= 0) & (flat_route >= 0)
            if route_valid.any():
                oracle_idx = flat_oracle[route_valid].long()
                pred_idx   = flat_route[route_valid].long()
                confusion.index_put_(
                    (oracle_idx, pred_idx),
                    torch.ones(oracle_idx.shape[0], dtype=torch.float64),
                    accumulate=True,
                )

            depth_valid = flat_mask & (flat_ops >= 0)
            if depth_valid.any():
                ops    = flat_ops[depth_valid].long().cpu()
                depths = exit_depths[depth_valid].double().cpu()
                for op_id, depth_val in zip(ops, depths):
                    op_depth_sum[op_id]   += depth_val
                    op_depth_count[op_id] += 1.0

            family_valid = flat_mask & (flat_oracle >= 0)
            if family_valid.any():
                fams   = flat_oracle[family_valid].long().cpu()
                depths = exit_depths[family_valid].double().cpu()
                for fam_id, depth_val in zip(fams, depths):
                    family_depth_sum[fam_id]   += depth_val
                    family_depth_count[fam_id] += 1.0

    total = confusion.sum().item()
    routing_acc = (
        confusion.diag().sum().item() / total if total > 0 else float("nan")
    )

    op_avg_depth: dict[str, float] = {}
    for i, name in enumerate(ALL_OP_NAMES):
        if op_depth_count[i] > 0:
            op_avg_depth[name] = (op_depth_sum[i] / op_depth_count[i]).item()

    family_avg_depth: dict[str, float] = {}
    for i, label in enumerate(EXPERT_FAMILY_LABELS):
        if family_depth_count[i] > 0:
            family_avg_depth[label] = (
                family_depth_sum[i] / family_depth_count[i]
            ).item()

    return (
        confusion.numpy(),
        routing_acc,
        op_avg_depth,
        family_avg_depth,
    )


def make_routing_confusion_figure(
    confusion: np.ndarray,
    normalize_rows: bool = True,
) -> plt.Figure:
    """Row-normalized 7×7 heatmap: actual expert family vs chosen expert."""
    cm = confusion.astype(np.float64)
    if normalize_rows:
        row_sums = cm.sum(axis=1, keepdims=True)
        display = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums > 0)
        vmax = 1.0
        cbar_label = "Row fraction"
        title = "Routing Confusion Matrix (row-normalized)"
    else:
        display = cm
        vmax = None
        cbar_label = "Token count"
        title = "Routing Confusion Matrix (counts)"

    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    im = ax.imshow(display, cmap="Blues", vmin=0.0, vmax=vmax)
    ax.set_xticks(range(len(EXPERT_FAMILY_LABELS)))
    ax.set_yticks(range(len(EXPERT_FAMILY_LABELS)))
    ax.set_xticklabels([f"E{i + 1}" for i in range(len(EXPERT_FAMILY_LABELS))])
    ax.set_yticklabels(EXPERT_FAMILY_LABELS, fontsize=9)
    ax.set_xlabel("Chosen Expert (router at depth 1, block 0)")
    ax.set_ylabel("Actual Operation Family (oracle)")
    ax.set_title(title)

    n = display.shape[0]
    for row in range(n):
        for col in range(n):
            val = display[row, col]
            if cm[row].sum() == 0 and normalize_rows:
                text = "—"
            elif normalize_rows:
                text = f"{val:.0%}" if val >= 0.005 else ""
            else:
                text = f"{int(val)}" if val > 0 else ""
            color = "white" if val > 0.55 else "black"
            ax.text(col, row, text, ha="center", va="center", color=color, fontsize=8)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=cbar_label)
    fig.tight_layout()
    return fig


def make_op_depth_bar_figure(op_avg_depth: dict[str, float]) -> plt.Figure | None:
    """Bar chart of average recursion depth grouped by operation type."""
    if not op_avg_depth:
        return None

    preferred_order = [
        "ADD", "SUB", "MULT", "DIV", "MOD", "POW",
        "AND", "OR", "XOR", "NOT",
        "SHIFT_L", "SHIFT_R",
        "SORT", "MEDIAN", "MAX", "MIN",
    ]
    labels = [op for op in preferred_order if op in op_avg_depth]
    labels += sorted(op for op in op_avg_depth if op not in labels)
    values = [op_avg_depth[op] for op in labels]

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.55), 5))
    bars = ax.bar(labels, values, color="#4C72B0", edgecolor="white")
    ax.set_xlabel("Operation")
    ax.set_ylabel("Average Recursion Depth")
    ax.set_title("Adaptive Compute: Operation vs Avg Recursion Depth")
    ax.set_ylim(0, max(values) * 1.15 if values else 1.0)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.05,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.tight_layout()
    return fig


def paper_metrics_to_wandb(
    confusion: np.ndarray,
    routing_acc: float,
    op_avg_depth: dict[str, float],
    family_avg_depth: dict[str, float],
) -> dict:
    """Build W&B log dict with paper-ready figures and scalar diagnostics."""
    log_dict: dict = {}

    if not math.isnan(routing_acc):
        log_dict["val/routing_accuracy"] = routing_acc

    row_totals = confusion.sum(axis=1)
    for i, label in enumerate(EXPERT_FAMILY_LABELS):
        if row_totals[i] > 0:
            log_dict[f"val/routing_recall/{label}"] = (
                confusion[i, i] / row_totals[i]
            )

    for op, depth_val in op_avg_depth.items():
        log_dict[f"recursion/avg_depth_by_op/{op}"] = depth_val

    for label, depth_val in family_avg_depth.items():
        log_dict[f"recursion/avg_depth_by_family/{label}"] = depth_val

    cm_fig = make_routing_confusion_figure(confusion, normalize_rows=True)
    log_dict["paper/routing_confusion_matrix"] = wandb.Image(cm_fig)
    plt.close(cm_fig)

    cm_count_fig = make_routing_confusion_figure(confusion, normalize_rows=False)
    log_dict["paper/routing_confusion_matrix_counts"] = wandb.Image(cm_count_fig)
    plt.close(cm_count_fig)

    depth_fig = make_op_depth_bar_figure(op_avg_depth)
    if depth_fig is not None:
        log_dict["paper/op_avg_recursion_depth"] = wandb.Image(depth_fig)
        plt.close(depth_fig)

    if op_avg_depth:
        table = wandb.Table(columns=["operation", "avg_recursion_depth"])
        for op in sorted(op_avg_depth, key=lambda k: op_avg_depth[k]):
            table.add_data(op, op_avg_depth[op])
        log_dict["paper/op_avg_recursion_depth_table"] = table

    return log_dict


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

    epochs          = run_epochs if run_epochs is not None else tc["epochs"]
    batch_sz        = tc["batch_size"]
    lr              = tc["lr"]
    val_split       = tc["val_split"]
    grad_clip       = tc["grad_clip"]
    subset_fraction = float(dc.get("subset_fraction", 1.0))
    subset_seed     = int(dc.get("subset_seed", 42))
    max_steps       = mc["max_steps"]
    step_feat_dim   = mc["step_feat_dim"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Device: {device}")
    print(
        f"[Train] Effective settings: epochs={epochs}, lr={lr}, "
        f"routing_balance={lw['routing_balance']}, step_routing={lw['step_routing']}, "
        f"subset_fraction={subset_fraction}, subset_seed={subset_seed}"
    )

    # ---- Dataset -------------------------------------------------------
    train_ds = MoREDataset(
        jsonl_path=dc.get("train_path", dc.get("jsonl_path")),
        max_steps=max_steps,
        step_feat_dim=step_feat_dim,
        max_val=dc["max_val"],
        pad_value=dc["pad_value"],
        num_experts=mc["num_experts"],
    )

    if 0.0 < subset_fraction < 1.0 and len(train_ds) > 1:
        subset_n    = max(1, int(len(train_ds) * subset_fraction))
        remainder_n = len(train_ds) - subset_n
        train_ds, _ = random_split(
            train_ds,
            [subset_n, remainder_n],
            generator=torch.Generator().manual_seed(subset_seed),
        )
        print(
            f"[Dataset] Using deterministic train subset: {subset_n} samples "
            f"({subset_fraction:.3f}), seed={subset_seed}"
        )

    val_ds   = None
    val_path = dc.get("val_path")
    if val_path and os.path.exists(val_path):
        print(f"[Dataset] Using validation set: {val_path}")
        val_ds = MoREDataset(
            jsonl_path=val_path,
            max_steps=max_steps,
            step_feat_dim=step_feat_dim,
            max_val=dc["max_val"],
            pad_value=dc["pad_value"],
            num_experts=mc["num_experts"],
        )
        n_train = len(train_ds)
        n_val   = len(val_ds)
    else:
        print("[Dataset] Validation file not found, falling back to random split.")
        dataset = train_ds
        n_val   = max(1, int(len(dataset) * val_split))
        n_train = len(dataset) - n_val
        if n_train < 1:
            n_train = len(dataset)
            n_val   = 0
        if n_val > 0:
            train_ds, val_ds = random_split(
                dataset,
                [n_train, n_val],
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
        step_feat_dim=step_feat_dim,
        d_model=mc["d_model"],
        num_experts=mc["num_experts"],
        max_depth=mc["max_depth"],
        num_blocks=mc["num_blocks"],
        dropout=mc["dropout"],
        fixed_depth=mc.get("fixed_depth", False),
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
        name=log.get("run_name"),
        config={**mc, **tc, **lw, **dc},
        reinit=True,
    )
    wandb.watch(model, log="gradients", log_freq=50)

    best_val_loss = float("inf")
    results_rows  = []

    # ---- Halt-loss linear warmup ----------------------------------------
    # Halt pressure is zero for the first 20 % of training steps so the
    # router can explore freely before being pushed toward early exits.
    # After warmup it ramps linearly up to the configured halting weight.
    total_steps        = epochs * len(train_loader)
    warmup_steps       = max(1, int(0.2 * total_steps))
    target_halt_weight = lw["halting"]
    current_step       = 0

    # ---- Training epochs -----------------------------------------------
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_total      = 0.0
        epoch_task       = 0.0
        epoch_bal        = 0.0
        epoch_halt       = 0.0
        epoch_step_route = 0.0
        epoch_start      = time.perf_counter()
        total_tokens     = 0
        depth_hist       = torch.zeros(mc["max_depth"])

        # --- Shannon entropy tracking (activated) -----------------------
        # epoch_expert_counts: hard-argmax token counts per expert, across all
        # batches in this epoch.  Used to compute:
        #   (a) per-expert load % (logged individually to W&B)
        #   (b) overall Shannon entropy via compute_expert_load_entropy()
        epoch_expert_counts: torch.Tensor = torch.zeros(mc["num_experts"], device=device)
        # Raw per-depth argmax tensors — fed into compute_expert_load_entropy
        epoch_all_expert_idx: list = []

        for batch_idx, (x, step_mask, step_experts, step_ops, family, depth, target) in enumerate(train_loader):
            x            = x.to(device)
            step_mask    = step_mask.to(device)
            step_experts = step_experts.to(device)
            step_ops     = step_ops.to(device)
            family       = family.to(device)
            target       = target.to(device)

            optimizer.zero_grad(set_to_none=True)

            (
                reg_out, cls_out, step_cls_out,
                bal_loss, halt_loss,
                depth_exits, avg_depth,
                batch_expert_idx,
                oracle_routing_ce,
                _first_route,
            ) = model(x, step_mask, step_experts)

            # Accumulate expert routing counts for Shannon entropy logging
            if batch_expert_idx is not None:
                for idx_tensor in batch_expert_idx:
                    for e in range(mc["num_experts"]):
                        epoch_expert_counts[e] += (idx_tensor == e).float().sum()
                # Feed raw tensors to compute_expert_load_entropy at epoch end
                epoch_all_expert_idx.extend(batch_expert_idx)

            # --- Losses -------------------------------------------------
            # 1. Primary regression loss
            task_loss = F.mse_loss(reg_out.squeeze(-1), target)

            # 2. Whole-program family classification (auxiliary supervision)
            cls_loss = F.cross_entropy(cls_out, family)

            # 3. Per-step routing loss (two complementary signals):
            #    (a) oracle_routing_ce: direct CE on router_logits at each depth step.
            #        Short gradient path — router weights updated immediately.
            #        Teaches the MoEBlock router: ADD→Expert0, MULT→Expert1, etc.
            #    (b) step_cls CE on step_cls_head(h): longer path but helps separate
            #        representations so the router's linear classifier is feasible.
            valid_mask   = step_experts.reshape(-1) >= 0                # [B*S]
            step_logits  = step_cls_out.reshape(-1, 7)[valid_mask]
            step_targets = step_experts.reshape(-1)[valid_mask]
            step_cls_loss = (
                F.cross_entropy(step_logits, step_targets)
                if step_logits.shape[0] > 0
                else torch.tensor(0.0, device=device)
            )
            # Blend: oracle CE (0.01) is secondary; step_cls CE (0.3) is primary
            step_routing_loss = 0.01 * oracle_routing_ce + 0.3 * step_cls_loss

            # 4. Routing balance + halting (halt weight linearly warmed up)
            current_step += 1
            current_halt_weight = min(
                target_halt_weight,
                target_halt_weight * (current_step / warmup_steps)
            )
            noise_reg_loss = 0.001 * sum((block.moe_block.router_noise_scale ** 2) for block in model.blocks)
            total_loss = (
                lw["task"]              * (task_loss + 0.5 * cls_loss)
                + lw["step_routing"]   * step_routing_loss    # per-step oracle CE (direct)
                + lw["routing_balance"] * bal_loss            # maximise entropy, minimise collapse
                + current_halt_weight   * halt_loss           # encourage early exit
                + noise_reg_loss
            )

            total_loss.backward()
            router_grad = model.blocks[0].moe_block.router.weight.grad
            if batch_idx == 0:
                noise_scales = [f"{b.moe_block.router_noise_scale.item():.4f}" for b in model.blocks]
                print(f"[Epoch {epoch:03d} Batch 0] Router grad norm: {router_grad.norm().item():.6f} | Noise scales: {noise_scales}")
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            # Accumulate metrics
            bs = x.shape[0]
            epoch_total      += total_loss.item()
            epoch_task       += task_loss.item()
            epoch_bal        += bal_loss.item()
            epoch_halt       += halt_loss.item()
            epoch_step_route += step_routing_loss.item()
            total_tokens     += bs
            if depth_exits is not None:
                depth_hist += depth_exits.detach().cpu().sum(dim=0)

        scheduler.step()
        elapsed    = time.perf_counter() - epoch_start
        throughput = total_tokens / max(elapsed, 1e-6)
        n_batches  = max(len(train_loader), 1)

        # ---- Validation + paper metrics --------------------------------
        val_loss = float("nan")
        paper_log_dict: dict = {}
        if val_loader is not None and (epoch % log["log_interval"] == 0 or epoch == epochs):
            model.eval()
            val_total = 0.0
            val_n     = 0
            confusion = torch.zeros(
                7, 7, dtype=torch.float64
            )
            op_depth_sum   = torch.zeros(NUM_OP_TYPES, dtype=torch.float64)
            op_depth_count = torch.zeros(NUM_OP_TYPES, dtype=torch.float64)
            family_depth_sum   = torch.zeros(7, dtype=torch.float64)
            family_depth_count = torch.zeros(7, dtype=torch.float64)

            with torch.no_grad():
                for x_v, sm_v, se_v, so_v, fam_v, _, tgt_v in val_loader:
                    x_v, sm_v, se_v, so_v, fam_v, tgt_v = (
                        x_v.to(device), sm_v.to(device), se_v.to(device),
                        so_v.to(device), fam_v.to(device), tgt_v.to(device)
                    )
                    (
                        reg_v, _, _, _, _, depth_exits, _, _, _,
                        first_route,
                    ) = model(x_v, sm_v, se_v)
                    l_v = F.mse_loss(reg_v.squeeze(-1), tgt_v)
                    val_total += l_v.item() * x_v.shape[0]
                    val_n     += x_v.shape[0]

                    if first_route is None or depth_exits is None:
                        continue

                    flat_mask   = sm_v.reshape(-1)
                    flat_oracle = se_v.reshape(-1)
                    flat_route  = first_route.reshape(-1)
                    flat_ops    = so_v.reshape(-1)
                    exit_depths = compute_token_exit_depths(
                        depth_exits, mc["max_depth"]
                    ).reshape(-1)

                    route_valid = flat_mask & (flat_oracle >= 0) & (flat_route >= 0)
                    if route_valid.any():
                        oracle_idx = flat_oracle[route_valid].long()
                        pred_idx   = flat_route[route_valid].long()
                        confusion.index_put_(
                            (oracle_idx, pred_idx),
                            torch.ones(oracle_idx.shape[0], dtype=torch.float64),
                            accumulate=True,
                        )

                    depth_valid = flat_mask & (flat_ops >= 0)
                    if depth_valid.any():
                        ops    = flat_ops[depth_valid].long().cpu()
                        depths = exit_depths[depth_valid].double().cpu()
                        for op_id, depth_val in zip(ops, depths):
                            op_depth_sum[op_id]   += depth_val
                            op_depth_count[op_id] += 1.0

                    family_valid = flat_mask & (flat_oracle >= 0)
                    if family_valid.any():
                        fams   = flat_oracle[family_valid].long().cpu()
                        depths = exit_depths[family_valid].double().cpu()
                        for fam_id, depth_val in zip(fams, depths):
                            family_depth_sum[fam_id]   += depth_val
                            family_depth_count[fam_id] += 1.0

            val_loss = val_total / max(val_n, 1)

            total_routed = confusion.sum().item()
            routing_acc = (
                confusion.diag().sum().item() / total_routed
                if total_routed > 0
                else float("nan")
            )
            op_avg_depth: dict[str, float] = {}
            for i, name in enumerate(ALL_OP_NAMES):
                if op_depth_count[i] > 0:
                    op_avg_depth[name] = (
                        op_depth_sum[i] / op_depth_count[i]
                    ).item()
            family_avg_depth: dict[str, float] = {}
            for i, label in enumerate(EXPERT_FAMILY_LABELS):
                if family_depth_count[i] > 0:
                    family_avg_depth[label] = (
                        family_depth_sum[i] / family_depth_count[i]
                    ).item()

            paper_log_dict = paper_metrics_to_wandb(
                confusion.numpy(),
                routing_acc,
                op_avg_depth,
                family_avg_depth,
            )

        # ---- Expert metrics (no grad) ----------------------------------
        with torch.no_grad():
            max_cos = compute_max_pairwise_cosine_sim(model)

        # Depth histogram as fractions
        depth_total = depth_hist.sum().clamp(min=1.0)
        depth_frac  = (depth_hist / depth_total).tolist()

        # --- Compute Shannon entropy from epoch_expert_counts -----------
        # epoch_expert_counts holds the hard-argmax token assignments summed
        # over ALL depth steps of ALL batches this epoch — activating this
        # block gives us:
        #   (a) per_expert_load[e]: fraction of all routed tokens → expert e
        #   (b) load_entropy: Shannon entropy of the load distribution
        total_expert_calls = epoch_expert_counts.sum().clamp(min=1.0)
        per_expert_load    = (epoch_expert_counts / total_expert_calls).cpu()  # [E]

        if epoch_all_expert_idx:
            load_entropy = compute_expert_load_entropy(
                depth_exits,            # used only for .device; last batch is fine
                epoch_all_expert_idx,   # every per-depth argmax tensor this epoch
                mc["num_experts"],
            ).item()
        else:
            load_entropy = 0.0

        # ---- W&B logging -----------------------------------------------
        log_dict = {
            "train/total_loss":          epoch_total      / n_batches,
            "train/task_loss":           epoch_task       / n_batches,
            "train/aux_routing_loss":    epoch_bal        / n_batches,
            "train/halting_loss":        epoch_halt       / n_batches,
            "train/step_routing_loss":   epoch_step_route / n_batches,
            "train/expert_load_entropy": load_entropy,
            "train/avg_recursion_steps": (
                avg_depth.item() if avg_depth is not None else float("nan")
            ),
            "diag/max_pairwise_cosine_sim": max_cos,
            "perf/throughput_tokens_sec":   throughput,
            "epoch": epoch,
        }

        # Per-expert load distribution (from the now-activated epoch_expert_counts)
        for e in range(mc["num_experts"]):
            log_dict[f"expert_load/expert_{e}_pct"] = per_expert_load[e].item() * 100.0

        # Depth distribution histogram
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

        log_dict.update(paper_log_dict)

        wandb.log(log_dict, step=epoch)

        routing_acc_str = (
            f"{paper_log_dict['val/routing_accuracy']:.4f}"
            if "val/routing_accuracy" in paper_log_dict
            else "n/a"
        )

        # Append to results.tsv (autoresearch_runner.py reads this)
        results_rows.append(
            f"{epoch}\t{epoch_task/n_batches:.6f}\t{val_loss:.6f}\t"
            f"{load_entropy:.4f}\t"
            f"{avg_depth.item() if avg_depth is not None else 0:.2f}\t"
            f"{max_cos:.4f}\t{routing_acc_str}"
        )

        # Flush results to disk every epoch so runner can read partial runs
        with open("results.tsv", "w") as f:
            f.write(
                "epoch\ttrain_task_loss\tval_loss\texpert_entropy\t"
                "avg_depth\tmax_cos_sim\trouting_accuracy\n"
            )
            f.write("\n".join(results_rows) + "\n")

        if epoch % log["log_interval"] == 0 or epoch == epochs:
            print(
                f"[Epoch {epoch:03d}/{epochs}] "
                f"task={epoch_task/n_batches:.4f}  "
                f"step_route={epoch_step_route/n_batches:.4f}  "
                f"bal={epoch_bal/n_batches:.4f}  "
                f"halt={epoch_halt/n_batches:.4f}  "
                f"val={val_loss:.4f}  "
                f"route_acc={routing_acc_str}  "
                f"entropy={load_entropy:.3f}  "
                f"cos={max_cos:.3f}  "
                f"tok/s={throughput:.0f}"
            )

    print(f"\n[Train] Complete. Best val_loss = {best_val_loss:.6f}")
    
    # Save final run metrics to a JSON file for automated analysis
    final_metrics_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "final_run_metrics.json")
    try:
        serializable_log = {}
        for k, v in log_dict.items():
            if hasattr(v, "item"):
                serializable_log[k] = v.item()
            elif isinstance(v, (int, float, str, bool, list, dict)):
                serializable_log[k] = v
            else:
                serializable_log[k] = str(v)
        serializable_log["best_val_loss"] = best_val_loss
        with open(final_metrics_path, "w") as f:
            json.dump(serializable_log, f, indent=2)
    except Exception as e:
        print(f"[WARN] Failed to write final_run_metrics.json: {e}")

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
    parser.add_argument("--blocks", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    args = parser.parse_args()

    try:
        cfg = load_config(args.config)
        if args.blocks is not None:
            cfg["model"]["num_blocks"] = args.blocks
        if args.batch_size is not None:
            cfg["training"]["batch_size"] = args.batch_size
        if args.seed is not None:
            cfg["training"]["subset_seed"] = args.seed
            cfg["data"]["subset_seed"] = args.seed
        if args.run_name is not None:
            cfg["logging"]["run_name"] = args.run_name
        train(cfg, run_epochs=args.epochs)
    except Exception as e:
        print(f"\n[FATAL] Training failed: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
