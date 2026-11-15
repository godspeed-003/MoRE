"""data.py - Dataset and per-step tokenisation.

Extracted from the former monolithic train.py (plan.md 9: ONE training
system, split into readable modules; not parallel implementations).
"""

import os
import sys
import json
import time
import math
import collections
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np

from .families import (FAMILY_TO_IDX, OP_TO_EXPERT, NO_FAMILY_IDX,
                       OP_NAME_TO_IDX, NUM_OP_TYPES)

class MoREDataset(Dataset):
    """
    Reads dummy.jsonl and encodes each record as a per-step token matrix.

    Encoding (per-step, structured):
      Each step → row [step_feat_dim]:
        Slots 0..: flatten(args), clamped and normalised into [-1, 1]
        Remaining: 0.0
      The step `result` and the oracle expert index are BOTH excluded
      (updated_rules.md 3). Operation identity travels in `step_ops`.
      Stack real step rows; pad to [max_steps, step_feat_dim] with zero rows.

    Returns 6-tuple per record:
      x            [max_steps, step_feat_dim]  float  — per-step token matrix
      step_mask    [max_steps]                 bool   — True = real step
      step_experts [max_steps]                 long   — oracle expert per step (-1=pad)
      step_ops     [max_steps]                 long   — op-type index per step (-1=pad)
      family       scalar long                        — whole-program family (0-5, -1=MIXED)
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
        num_experts: int = 6,
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

                    # Op-code -> expert index. No catch-all: an unmapped
                    # operation raises (plan.md 7.2 / updated_rules.md 2).
                    op_id = OP_NAME_TO_IDX.get(op, -1)
                    if op_id < 0 or op not in OP_TO_EXPERT:
                        raise KeyError(
                            f"Unknown operation '{op}' in record {line_no} of "
                            f"'{jsonl_path}'. Operation identity is the model's "
                            "only signal for WHICH function to compute, so an "
                            "unmapped op must raise rather than silently become "
                            "a zero embedding (updated_rules.md 2)."
                        )

                    expert_id = OP_TO_EXPERT[op]

                    # Flatten args (handle scalar or list args for SORT/STAT)
                    flat_args = []
                    for a in args:
                        if isinstance(a, (int, float)):
                            flat_args.append(float(a))
                        elif isinstance(a, list):
                            flat_args.extend(float(v) for v in a)

                    # Scalar result (mean of list for SORT output etc.) — kept
                    # for reference only; it is NEVER written into the input.
                    if isinstance(result, list):
                        scalar_result = float(sum(result)) / max(len(result), 1)
                    else:
                        scalar_result = float(result)
                    del scalar_result

                    # Clamp + normalise the ARGUMENTS into [-1, 1].
                    #
                    # T1.1 (plan.md 3.1): the step `result` is deliberately
                    # excluded. data/script.py's verify() guarantees
                    # output == steps[-1]["result"], so appending results put the
                    # regression target directly into the input — measured in
                    # 5250/5250 validation records, which made a closed-form
                    # copy baseline reach MSE 0.0.
                    numeric_vals = [
                        max(-self.max_val, min(self.max_val, v)) / self.max_val
                        for v in flat_args
                    ]

                    # Row = numeric arguments only, from slot 0.
                    #
                    # T1.2 (plan.md 3.2): slot 0 used to hold
                    # `expert_id / (num_experts - 1)` — the oracle routing label,
                    # prohibited as an input feature (updated_rules.md 3).
                    # Operation identity IS legitimate task information, so it
                    # now travels separately in `step_ops` and is consumed by an
                    # operation embedding inside the model.
                    #
                    # T1.3: because the row no longer references `num_experts`,
                    # features are bit-identical across MoE / MoR / MoRE for the
                    # same record — the input-parity invariant is now testable.
                    row = [0.0] * step_feat_dim
                    for i, v in enumerate(numeric_vals[:step_feat_dim]):
                        row[i] = v

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

                fam_label = rec.get("family")
                if fam_label not in FAMILY_TO_IDX:
                    raise KeyError(
                        f"Unknown family '{fam_label}' in record {line_no} of "
                        f"'{jsonl_path}'. Known: {sorted(FAMILY_TO_IDX)}. There is "
                        "no 7th catch-all family (plan.md 7.2)."
                    )
                family_idx = FAMILY_TO_IDX[fam_label]
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


