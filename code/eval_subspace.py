# -*- coding: utf-8 -*-
"""eval_subspace.py -- the subspace-trapping probe (the specialized test).

    python code/eval_subspace.py                     # MoE vs MoRE, val sample
    python code/eval_subspace.py --n_blocks 80
    python code/eval_subspace.py --arch more

THE HYPOTHESIS UNDER TEST
-------------------------
Gemini's reading of the MoRE result: because a token is routed ONCE and then recurses
on that single expert's weights for up to 7 steps, its representation is confined
("trapped") to one expert's subspace and cannot make the syntax -> shallow-semantics
-> retrieval phase transitions that a stack of DIFFERENT layers affords. The halting
then saturates to max depth in a doomed attempt to reduce loss from inside that
subspace. MoE applies an expert ONCE per token and never recurses, so if trapping is
real, MoRE's final representations should be MORE confined and its experts MORE
mutually separated than MoE's at matched parameters.

WHAT IS MEASURED (on the final hidden state `h`, the input to `lm_head`)
-----------------------------------------------------------------------
* participation_ratio -- effective dimensionality of `h`, PR = (sum λ)^2 / sum λ^2
  over the covariance eigenvalues, range [1, d_model]. LOWER = more confined. The
  global PR and the per-expert PR (tokens grouped by their routed expert) are both
  reported. Trapping predicts MoRE < MoE on global PR and on mean per-expert PR.
* between_expert_separation -- mean pairwise COSINE between the six expert-mean
  vectors (lower cosine = more separated) and the between/total variance ratio (a
  Fisher-style number: higher = experts occupy more distinct regions). Trapping
  predicts MoRE more separated than MoE.

This is a FINAL-STATE confinement probe, which is tractable with a single hook on
`lm_head`'s input. The stronger DYNAMIC form -- how little the representation moves
from recursion step t to t+1 -- needs per-step capture across the shrinking active
set and is left as the post-ACL follow-up (POST_ACL_TODO.md); state that limitation
rather than implying this measures the across-depth trajectory directly.

EXPLORATORY, like every eval outside the two declared families. Reported with the
exact randomization test, UNCORRECTED, no abstract-level verdict.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

CODE = Path(os.path.dirname(os.path.abspath(__file__)))
REPO = CODE.parent
sys.path.insert(0, str(CODE))

from eval_test_split import build_model                       # noqa: E402
from more.config import TASK_LANGUAGE                         # noqa: E402
import seed_stats as _seed_stats                              # noqa: E402

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


def participation_ratio(x: torch.Tensor) -> float:
    """Effective dimensionality of rows of x [n, d]: (sum λ)^2 / sum λ^2 of the
    covariance. 1.0 = a line, d = isotropic. Centered; needs n >= 2."""
    if x.shape[0] < 2:
        return float("nan")
    xc = (x - x.mean(0, keepdim=True)).double()
    cov = (xc.T @ xc) / (x.shape[0] - 1)
    ev = torch.linalg.eigvalsh(cov).clamp(min=0)
    s1, s2 = float(ev.sum()), float((ev * ev).sum())
    return (s1 * s1) / s2 if s2 > 0 else float("nan")


@torch.no_grad()
def probe(run_dir, device, n_blocks, batch=16):
    rc = json.loads((run_dir / "resolved_config.json").read_text(encoding="utf-8"))
    mc, dc = rc.get("model", {}), rc.get("data", {})
    task = rc.get("task") or (rc.get("provenance", {}) or {}).get("task")
    if task != TASK_LANGUAGE:
        raise RuntimeError(f"{run_dir.name}: not a language run")
    if "step_feat_dim" not in mc:
        raise KeyError(f"{run_dir.name}: no model.step_feat_dim")
    model = build_model(mc, dc, task, int(mc["step_feat_dim"]), device)
    model.load_state_dict(torch.load(run_dir / "checkpoint.pt", map_location=device))
    model.eval()
    E = int(mc["num_experts"])

    captured = []
    h_pre = model.lm_head.register_forward_pre_hook(
        lambda m, inp: captured.append(inp[0].detach().reshape(-1, inp[0].shape[-1])))

    from more.lang_data import MoRELanguageDataset
    corpus = dc.get("corpus") or dc.get("dataset_version")
    ds = MoRELanguageDataset(corpus, "val")
    loader = DataLoader(ds, batch_size=batch, shuffle=False)

    H, R = [], []
    seen = 0
    for b in loader:
        x, sm, se, so = [t.to(device) if torch.is_tensor(t) else t for t in b[:4]]
        captured.clear()
        out = model(x, sm, se, so)
        h = captured[-1]                                   # [B*L, d]
        route = out[9]                                     # [B*L] routed expert
        route = route.reshape(-1) if torch.is_tensor(route) else None
        H.append(h.float().cpu())
        R.append(route.cpu() if route is not None else torch.full((h.shape[0],), -1))
        seen += x.shape[0]
        if seen >= n_blocks:
            break
    h_pre.remove()

    H = torch.cat(H)                                       # [T, d]
    R = torch.cat(R)                                       # [T]
    prov = rc.get("provenance", {}) or {}

    global_pr = participation_ratio(H)
    per_expert_pr, means, counts = {}, [], []
    for e in range(E):
        sel = R == e
        n = int(sel.sum())
        counts.append(n)
        if n >= 2:
            he = H[sel]
            per_expert_pr[e] = participation_ratio(he)
            means.append(he.mean(0))
    # Between-expert separation: mean pairwise cosine of expert means (lower =
    # more separated), and between/total variance ratio.
    sep_cos = float("nan")
    if len(means) >= 2:
        M = torch.stack(means)
        Mn = M / M.norm(dim=1, keepdim=True).clamp(min=1e-8)
        C = Mn @ Mn.T
        iu = torch.triu_indices(len(means), len(means), offset=1)
        sep_cos = float(C[iu[0], iu[1]].mean())
    total_var = float(((H - H.mean(0)) ** 2).sum(1).mean())
    between_var = float(((torch.stack(means) - H.mean(0)) ** 2).sum(1).mean()) \
        if len(means) >= 2 else float("nan")
    between_ratio = between_var / total_var if total_var > 0 else float("nan")

    valid_pr = [v for v in per_expert_pr.values() if v == v]
    return {
        "run": run_dir.name, "arch": prov.get("architecture"),
        "seed": int(prov.get("seed", -1)), "n_tokens": int(H.shape[0]),
        "d_model": int(H.shape[1]),
        "global_participation_ratio": global_pr,
        "mean_per_expert_pr": st.mean(valid_pr) if valid_pr else float("nan"),
        "per_expert_pr": {str(k): v for k, v in per_expert_pr.items()},
        "expert_token_counts": counts,
        "between_expert_mean_cosine": sep_cos,
        "between_over_total_variance": between_ratio,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--arch", default=None, choices=["moe", "mor", "more"])
    ap.add_argument("--run", action="append", default=None)
    ap.add_argument("--n_blocks", type=int, default=48,
                    help="val blocks to pool for the covariance (>= ~20 for d=256)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runs = canonical_runs(args.run, args.arch)
    if not runs:
        print("no canonical cell with a checkpoint found.")
        return 2

    print(f"device={device}  runs={len(runs)}  n_blocks={args.n_blocks}")
    print("EXPLORATORY -- final-state confinement probe; per-step dynamic form is "
          "the post-ACL follow-up.\n")

    cells = []
    for d in runs:
        c = probe(d, device, args.n_blocks)
        cells.append(c)
        print(f"  {c['arch']:5s} s{c['seed']}  globalPR {c['global_participation_ratio']:6.2f}  "
              f"meanExpertPR {c['mean_per_expert_pr']:6.2f}  "
              f"sepCos {c['between_expert_mean_cosine']:+.3f}  "
              f"betw/tot {c['between_over_total_variance']:.4f}")

    agg, arms = {}, sorted({c["arch"] for c in cells})
    for a in arms:
        sel = [c for c in cells if c["arch"] == a]
        def ms(key):
            vals = [c[key] for c in sel if c[key] == c[key]]
            return (st.mean(vals), st.stdev(vals) if len(vals) > 1 else None) if vals \
                else (float("nan"), None)
        agg[a] = {"n_seeds": len(sel),
                  "global_pr": ms("global_participation_ratio"),
                  "mean_per_expert_pr": ms("mean_per_expert_pr"),
                  "between_expert_mean_cosine": ms("between_expert_mean_cosine"),
                  "between_over_total_variance": ms("between_over_total_variance")}

    pairwise = {}
    for i, a in enumerate(arms):
        for b in arms[i + 1:]:
            r = _seed_stats.perm_test(
                [c["global_participation_ratio"] for c in cells if c["arch"] == a],
                [c["global_participation_ratio"] for c in cells if c["arch"] == b])
            if r is not None:
                pairwise[f"{a}_vs_{b}_globalPR"] = {
                    "gap": r["gap"], "p_value": r["p_value"], "min_p": r["min_p"],
                    "status": "UNCORRECTED -- exploratory"}

    out = Path(args.out) if args.out else (REPO / "results" / "language" / "subspace.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "probe": "subspace-trapping (Gemini hypothesis): final-state confinement",
        "status": "EXPLORATORY. Final-state participation ratio and inter-expert "
                  "separation on lm_head input. LOWER global/per-expert PR and HIGHER "
                  "separation for MoRE than MoE would support trapping. The across-depth "
                  "dynamic form is the post-ACL follow-up. Uncorrected.",
        "by_arm": {a: {k: {"mean": v[0], "std": v[1]} if isinstance(v, tuple) else v
                       for k, v in d.items()} for a, d in agg.items()},
        "pairwise_uncorrected": pairwise, "cells": cells,
    }, indent=2) + "\n", encoding="utf-8")

    print("\nby arm (mean over seeds):")
    for a in arms:
        g = agg[a]["global_pr"]
        print(f"  {a:5s} globalPR {g[0]:.2f}" + (f" +- {g[1]:.2f}" if g[1] else ""))
    for k, v in pairwise.items():
        print(f"  {k}: gap {v['gap']:+.2f}, p={v['p_value']:.4f} (floor {v['min_p']:.4f})")
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
