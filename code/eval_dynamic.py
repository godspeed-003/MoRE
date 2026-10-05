# -*- coding: utf-8 -*-
"""eval_dynamic.py -- the across-depth dynamic probe (velocity + update orthogonality).

    python code/eval_dynamic.py                      # all arms with a checkpoint
    python code/eval_dynamic.py --arch more --n_blocks 48

WHY THIS EXISTS
---------------
The final-state subspace probe (eval_subspace.py) refuted trapping in the END
representation but could not see the TRAJECTORY. This is the dynamic form Gemini
asked for: does a token actually travel through latent space across the recursion,
or vibrate in place / bounce in a recursive eddy -- which would explain why ACT
never halts (the state never converges to a confident point).

TWO QUANTITIES, PER RECURSION STEP k (on the MoREWrapper layer_norm output h_k,
the post-residual state written back each step, model.py:878)
--------------------------------------------------------------------------------
* velocity_k = mean_tokens || h_k - h_{k-1} ||_2, and the cosine distance
  1 - cos(h_k, h_{k-1}). A healthy recursive refinement shows meaningful velocity
  that DECAYS as the representation settles (which is what lets ACT halt). A
  collapsed one shows microscopic velocity (stuck) or a non-decaying / oscillating
  profile (eddy).
* update_collinearity_k = mean_tokens cos(h_k - h_{k-1}, h_{k-1}). If the per-step
  update Delta h_k is collinear with the current state, the expert is SCALING the
  existing representation rather than adding new features -- the mechanism that
  would produce the high participation ratio and near-collinear experts the
  final-state probe measured. Near 0 = orthogonal (new features); near +-1 =
  pure rescale.

WHY fixed_depth IS FORCED ON
----------------------------
With adaptive halting, the active-token set shrinks step to step, so layer_norm's
output changes BOTH length and row-to-token mapping across steps and consecutive
states cannot be differenced. Running with fixed_depth=True pushes EVERY token
through all max_depth steps, so each step's layer_norm output is [N, d] in a fixed
token order and h_k - h_{k-1} is well defined. For MoRE this barely changes the
measured object -- it already forces 97% of tokens to depth 7 -- and it is exactly
Gemini's "force exit at depth k" framing. MoE has max_depth 1 (one step, no
trajectory) and is reported as N/A.

EXPLORATORY, like every eval outside the two declared families. Per-arm means over
seeds; no abstract-level verdict.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

CODE = Path(os.path.dirname(os.path.abspath(__file__)))
REPO = CODE.parent
sys.path.insert(0, str(CODE))

from eval_test_split import build_model                       # noqa: E402
from more.config import TASK_LANGUAGE                         # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

CANONICAL_GROUP = "canonical_lang_b"


def canonical_runs(explicit, arch_filter):
    out = []
    runs = REPO / "runs"
    for d in sorted(runs.iterdir()) if runs.is_dir() else []:
        if explicit and d.name not in explicit:
            continue
        if not ((d / "resolved_config.json").exists() and (d / "checkpoint.pt").exists()):
            continue
        try:
            rc = json.loads((d / "resolved_config.json").read_text(encoding="utf-8"))
        except ValueError:
            continue
        prov = rc.get("provenance", {}) or {}
        if prov.get("experiment_group") != CANONICAL_GROUP:
            continue
        if arch_filter and prov.get("architecture") != arch_filter:
            continue
        out.append(d)
    return out


def find_recursion_layer_norm(model):
    """The MoREWrapper's residual LayerNorm, applied once per recursion step. Matched
    by the attribute name `.layer_norm` so the step_proj/attn LayerNorms are excluded."""
    hits = [m for name, m in model.named_modules()
            if name.endswith("layer_norm") and isinstance(m, nn.LayerNorm)]
    if len(hits) != 1:
        raise RuntimeError(f"expected exactly one *.layer_norm LayerNorm, found {len(hits)}")
    return hits[0]


@torch.no_grad()
def probe(run_dir, device, n_blocks, batch=16):
    rc = json.loads((run_dir / "resolved_config.json").read_text(encoding="utf-8"))
    mc, dc = dict(rc.get("model", {})), rc.get("data", {})
    task = rc.get("task") or (rc.get("provenance", {}) or {}).get("task")
    prov = rc.get("provenance", {}) or {}
    if task != TASK_LANGUAGE:
        raise RuntimeError(f"{run_dir.name}: not a language run")
    if "step_feat_dim" not in mc:
        raise KeyError(f"{run_dir.name}: no model.step_feat_dim")
    K = int(mc["max_depth"])
    if K < 2:
        # MoE: one step, no trajectory.
        return {"run": run_dir.name, "arch": prov.get("architecture"),
                "seed": int(prov.get("seed", -1)), "max_depth": K,
                "velocity": None, "cos_distance": None, "update_collinearity": None,
                "note": "max_depth 1 -- single step, no across-depth trajectory"}

    mc["fixed_depth"] = True              # force every token through all K steps
    model = build_model(mc, dc, task, int(mc["step_feat_dim"]), device)
    model.load_state_dict(torch.load(run_dir / "checkpoint.pt", map_location=device))
    model.eval()

    captured = []
    h = find_recursion_layer_norm(model).register_forward_hook(
        lambda m, inp, out: captured.append(out.detach()))

    from more.lang_data import MoRELanguageDataset
    ds = MoRELanguageDataset(dc.get("corpus") or dc.get("dataset_version"), "val")
    loader = DataLoader(ds, batch_size=batch, shuffle=False)

    # Accumulators per transition k-1 -> k (K-1 of them).
    vel = [[] for _ in range(K - 1)]
    cosd = [[] for _ in range(K - 1)]
    coll = [[] for _ in range(K - 1)]
    seen = 0
    for b in loader:
        x, sm, se, so = [t.to(device) if torch.is_tensor(t) else t for t in b[:4]]
        captured.clear()
        model(x, sm, se, so)
        if len(captured) < K:
            # Fewer calls than K would mean fixed_depth did not force full depth;
            # skip rather than mis-align. Should not happen with fixed_depth=True.
            continue
        steps = captured[:K]                     # each [N, d], fixed token order
        for k in range(1, K):
            prev, cur = steps[k - 1], steps[k]
            if prev.shape != cur.shape:
                break
            delta = cur - prev
            vel[k - 1].append(float(delta.norm(dim=-1).mean()))
            cosd[k - 1].append(float((1 - torch.cosine_similarity(cur, prev, dim=-1)).mean()))
            coll[k - 1].append(float(torch.cosine_similarity(delta, prev, dim=-1).mean()))
        seen += x.shape[0]
        if seen >= n_blocks:
            break
    h.remove()

    mean = lambda L: [st.mean(c) if c else float("nan") for c in L]
    return {"run": run_dir.name, "arch": prov.get("architecture"),
            "seed": int(prov.get("seed", -1)), "max_depth": K,
            "transition_k": list(range(2, K + 1)),
            "velocity": mean(vel), "cos_distance": mean(cosd),
            "update_collinearity": mean(coll)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--arch", default=None, choices=["moe", "mor", "more"])
    ap.add_argument("--run", action="append", default=None)
    ap.add_argument("--n_blocks", type=int, default=48)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runs = canonical_runs(args.run, args.arch)
    if not runs:
        print("no canonical cell with a checkpoint found.")
        return 2
    print(f"device={device}  runs={len(runs)}  n_blocks={args.n_blocks}  "
          "(fixed_depth forced for clean trajectories)")
    print("EXPLORATORY -- the across-depth dynamic probe.\n")

    cells = []
    for d in runs:
        c = probe(d, device, args.n_blocks)
        cells.append(c)
        if c["velocity"] is None:
            print(f"  {c['arch']:5s} s{c['seed']}  N/A ({c['note']})")
        else:
            v = " ".join(f"{x:.3f}" for x in c["velocity"])
            co = " ".join(f"{x:+.2f}" for x in c["update_collinearity"])
            print(f"  {c['arch']:5s} s{c['seed']}  vel[{v}]  coll[{co}]")

    # Per-arm mean curves across seeds.
    agg, arms = {}, sorted({c["arch"] for c in cells if c["velocity"] is not None})
    for a in arms:
        sel = [c for c in cells if c["arch"] == a and c["velocity"] is not None]
        if not sel:
            continue
        K1 = len(sel[0]["velocity"])
        agg[a] = {
            "n_seeds": len(sel),
            "transition_k": sel[0]["transition_k"],
            "velocity_mean": [st.mean([c["velocity"][i] for c in sel]) for i in range(K1)],
            "cos_distance_mean": [st.mean([c["cos_distance"][i] for c in sel]) for i in range(K1)],
            "update_collinearity_mean": [st.mean([c["update_collinearity"][i] for c in sel]) for i in range(K1)],
        }

    out = Path(args.out) if args.out else (REPO / "results" / "language" / "dynamic.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "probe": "across-depth dynamic (velocity + update collinearity)",
        "status": "EXPLORATORY. fixed_depth forced so per-step states align. velocity "
                  "= mean L2 of consecutive post-norm states; update_collinearity = "
                  "mean cos(h_k - h_{k-1}, h_{k-1}) (near 0 orthogonal/new features, "
                  "near +-1 pure rescale). MoE is single-step N/A. Uncorrected.",
        "by_arm": agg, "cells": cells,
    }, indent=2) + "\n", encoding="utf-8")

    print("\nby arm (mean velocity per transition 2..K):")
    for a in arms:
        print(f"  {a:5s} vel  " + " ".join(f"{x:.3f}" for x in agg[a]["velocity_mean"]))
        print(f"  {a:5s} coll " + " ".join(f"{x:+.2f}" for x in agg[a]["update_collinearity_mean"]))
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
