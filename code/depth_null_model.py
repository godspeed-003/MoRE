"""
The null model for the adaptive-depth claim (T11.2, results.md 9 / 15 / 17).

WHY THIS EXISTS
---------------
`val_offline/depth_allocation_error_abs` is reported per run (code/eval_val_depth.py)
but a raw error of 0.94 steps is uninterpretable on its own: it has no scale and no
null. The paper's depth claim is that the model "learns to allocate recursion depth
in accordance with the predefined operation-complexity curriculum" (CLAUDE.md 4).
The correct null for that claim is the BEST CONSTANT-DEPTH POLICY -- a model that
ignores its input entirely and emits one number for every token. If a trained
adaptive halt head cannot beat that, "adaptive computation" is mechanical only, and
the honest verdict is CLAUDE.md 8 Outcome C for the halting half of MoRE.

This computes, on the held-out split, `depth_allocation_error(c, ...)` for every
constant c on a fine grid, using:
  * the SAME dataset class and the SAME loader settings as eval_val_depth.py:139,
  * the SAME `metrics.depth_allocation_error` the live training loop calls,
  * the SAME per-batch-mean-then-average-batches accumulation as
    eval_val_depth.py:184 (unweighted over batches, so the short final batch counts
    equally). Mirroring the accumulation matters: weighting batches by size would
    move the null by ~1e-4 while the measured runs stayed where they are, and the
    comparison would silently stop being like-for-like.

No model is loaded and no checkpoint is read. This is a property of the DATA plus
the curriculum table, so it is constant across seeds and architectures -- one
artifact serves every run.

OUTPUT
------
`results/depth_null_model.json`. Nothing else is written; no run directory is
touched.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

from more.data import MoREDataset                                # noqa: E402
from more.families import (OP_TARGET_DEPTH, ALL_OP_NAMES,        # noqa: E402
                           op_target_depth_table)
from more.metrics import depth_allocation_error                  # noqa: E402
from seed_stats import mean_std                                  # noqa: E402

CODE_DIR = Path(__file__).resolve().parent
OUT = CODE_DIR.parent / "results" / "depth_null_model.json"


def _out(s: str) -> None:
    """cp1252-safe stdout."""
    sys.stdout.buffer.write(s.encode("utf-8"))
    sys.stdout.buffer.write(b"\n")
    sys.stdout.flush()


def batches(split_path: Path, cfg: dict):
    """The exact loader eval_val_depth.py builds, minus the model."""
    ds = MoREDataset(
        jsonl_path=str(split_path),
        max_steps=cfg["max_steps"],
        step_feat_dim=cfg["step_feat_dim"],
        max_val=cfg["max_val"],
        pad_value=cfg["pad_value"],
        num_experts=cfg["num_experts"],
    )
    loader = DataLoader(ds, batch_size=min(int(cfg["batch_size"]), len(ds)),
                        shuffle=False, num_workers=0)
    return ds, loader


def compare_runs(best_abs, best_rel) -> dict:
    """
    Read the canonical runs' offline depth sidecars and state, per architecture,
    whether the trained halt head beat the best constant. Reads only
    `runs/phaseB_*/val_depth_offline.json`; writes nothing there.

    The test is the exact SIGN test over seeds, not the randomization test used for
    between-architecture gaps: the null here is a single deterministic constant with
    no seed variance, so there is no second sample to permute against. Its two-sided
    floor at n = 5 is 2 * 0.5**5 = 0.0625 -- report it with the floor, same
    discipline as seed_stats.py's min_p.
    """
    runs_dir = CODE_DIR.parent / "runs"
    out: dict = {
        "_note": [
            "Per-architecture comparison of val_offline/depth_allocation_error_* "
            "against the best constant policy above. measured_at is each run's "
            "BEST-VAL checkpoint (eval_val_depth.py), not the last-epoch model the "
            "headline loss comes from -- see that script's docstring.",
            "sign_test_p_two_sided floor at n=5 is 0.0625; k/n = n/n means every "
            "seed lost to the constant, which is the strongest statement this "
            "design can make.",
        ],
        "null_abs_error": best_abs[1][0],
        "null_abs_constant": float(best_abs[0]),
        "null_rel_error_at_abs_optimal_constant": best_abs[1][1],
        "null_rel_error_best": best_rel[1][1],
        "null_rel_constant": float(best_rel[0]),
    }
    for arch in ("mor", "more"):
        rows = []
        for p in sorted(runs_dir.glob(f"phaseB_{arch}_*/val_depth_offline.json")):
            d = json.loads(p.read_text(encoding="utf-8"))
            a = d.get("val_offline/depth_allocation_error_abs")
            r = d.get("val_offline/depth_allocation_error_rel")
            s = d.get("val_offline/avg_recursion_steps")
            if isinstance(a, (int, float)) and isinstance(r, (int, float)):
                rows.append({"seed": d.get("resolved_seed"), "abs": a, "rel": r,
                             "steps": s, "run_dir": d.get("run_dir")})
        if not rows:
            out[arch] = "N/A -- no sidecars found"
            continue
        n = len(rows)
        worse_abs = sum(1 for x in rows if x["abs"] > out["null_abs_error"])
        worse_rel = sum(1 for x in rows if x["rel"] > out["null_rel_error_best"])
        # seed_stats.mean_std returns a dict (or None) so an all-"N/A" metric
        # cannot silently become 0.0; unpack it explicitly.
        d_abs = mean_std([x["abs"] for x in rows]) or {}
        d_rel = mean_std([x["rel"] for x in rows]) or {}
        d_st = mean_std([x["steps"] for x in rows]) or {}
        m_abs, s_abs = d_abs.get("mean"), d_abs.get("std")
        m_rel, s_rel = d_rel.get("mean"), d_rel.get("std")
        m_st, s_st = d_st.get("mean"), d_st.get("std")
        out[arch] = {
            "n_seeds": n,
            "per_seed": rows,
            "abs_error_mean": m_abs, "abs_error_std": s_abs,
            "rel_error_mean": m_rel, "rel_error_std": s_rel,
            "avg_recursion_steps_mean": m_st, "avg_recursion_steps_std": s_st,
            "abs_error_minus_null": m_abs - out["null_abs_error"],
            "abs_error_pct_worse_than_null":
                100.0 * (m_abs - out["null_abs_error"]) / out["null_abs_error"],
            "seeds_worse_than_null_abs": f"{worse_abs}/{n}",
            "seeds_worse_than_null_rel": f"{worse_rel}/{n}",
            "sign_test_p_two_sided_abs": 2.0 * 0.5 ** n if worse_abs in (0, n)
                else "not unanimous -- see per_seed",
            "beats_null": worse_abs == 0,
            "verdict": ("adaptive policy LOSES to the best constant policy"
                        if worse_abs == n else
                        "adaptive policy beats the best constant policy"
                        if worse_abs == 0 else "mixed across seeds"),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="val", choices=("val", "test"))
    args = ap.parse_args()

    cfg_all = json.loads((CODE_DIR / "config.json").read_text(encoding="utf-8"))
    cfg = {
        "max_steps": cfg_all["model"]["max_steps"],
        "step_feat_dim": cfg_all["model"]["step_feat_dim"],
        "max_val": cfg_all["data"]["max_val"],
        "pad_value": cfg_all["data"]["pad_value"],
        "num_experts": 6,
        "batch_size": cfg_all["training"]["batch_size"],
    }
    split_path = (CODE_DIR / cfg_all["data"][f"{args.split}_path"]).resolve()
    ds, loader = batches(split_path, cfg)
    table = op_target_depth_table()

    # Cache every batch's (step_ops, step_mask) once; the grid scan reuses them.
    cached = [(so, sm) for _x, sm, _se, so, _f, _r, _t in loader]
    _out(f"[null] {args.split}: {len(ds)} records, {len(cached)} batches")

    # Token-level curriculum statistics over exactly the tokens the error uses.
    counts: dict[int, int] = {}
    for so, sm in cached:
        flat_ops, flat_mask = so.reshape(-1), sm.reshape(-1)
        valid = flat_mask & (flat_ops >= 0)
        for op_code in flat_ops[valid].tolist():
            d = int(table[op_code])
            counts[d] = counts.get(d, 0) + 1
    n_tok = sum(counts.values())
    weighted_mean = sum(d * n for d, n in counts.items()) / n_tok

    def err(c: float) -> tuple[float, float]:
        a_sum = r_sum = 0.0
        for so, sm in cached:
            # model.py returns expected_depth already FLAT over (batch * max_steps)
            # step-tokens -- depth_allocation_error indexes it with a flat mask, so
            # a (B, max_steps) tensor here would raise. Mirror the flat shape.
            pred = torch.full((so.numel(),), float(c), dtype=torch.float32)
            a, r = depth_allocation_error(pred, so, sm, table)
            a_sum += a
            r_sum += r
        return a_sum / len(cached), r_sum / len(cached)

    grid = [round(0.50 + 0.01 * i, 2) for i in range(int((7.00 - 0.50) / 0.01) + 1)]
    scan = {f"{c:.2f}": err(c) for c in grid}
    best_abs = min(scan.items(), key=lambda kv: kv[1][0])
    best_rel = min(scan.items(), key=lambda kv: kv[1][1])

    payload = {
        "_README": [
            "Best-constant-depth NULL MODEL for the adaptive-depth claim.",
            "An adaptive halt head only demonstrates input-dependent allocation if",
            "it beats best_constant_policy.abs_error below. Same split, same loader,",
            "same depth_allocation_error function, same batch accumulation as",
            "code/eval_val_depth.py, so the numbers are directly comparable to",
            "val_offline/depth_allocation_error_{abs,rel} in every run's sidecar.",
            "No checkpoint is read: this depends only on the data and the curriculum.",
        ],
        "source": "code/depth_null_model.py",
        "split": args.split,
        "split_file": str(split_path.name),
        "dataset_version": cfg_all["data"]["dataset_version"],
        "records": len(ds),
        "valid_step_tokens": n_tok,
        "curriculum_table": {k: OP_TARGET_DEPTH[k] for k in sorted(OP_TARGET_DEPTH)},
        "operations": len(ALL_OP_NAMES),
        "target_depth_token_counts": {str(k): counts[k] for k in sorted(counts)},
        "target_depth_token_fractions": {
            str(k): counts[k] / n_tok for k in sorted(counts)
        },
        "curriculum_mean_unweighted_over_operations": (
            sum(OP_TARGET_DEPTH.values()) / len(OP_TARGET_DEPTH)
        ),
        "curriculum_mean_token_weighted": weighted_mean,
        "oracle_policy": {"abs_error": 0.0, "rel_error": 0.0,
                          "note": "pred == target by construction"},
        "best_constant_policy_by_abs": {
            "constant": float(best_abs[0]), "abs_error": best_abs[1][0],
            "rel_error": best_abs[1][1]},
        "best_constant_policy_by_rel": {
            "constant": float(best_rel[0]), "abs_error": best_rel[1][0],
            "rel_error": best_rel[1][1]},
        "integer_constants": {
            str(c): {"abs_error": scan[f"{c:.2f}"][0],
                     "rel_error": scan[f"{c:.2f}"][1]}
            for c in (1, 2, 3, 4, 5, 6, 7)
        },
        "grid": {"lo": 0.50, "hi": 7.00, "step": 0.01, "points": len(grid)},
        "scan_abs": {k: v[0] for k, v in scan.items()},
        "scan_rel": {k: v[1] for k, v in scan.items()},
        "measured_vs_null": compare_runs(best_abs, best_rel),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    _out(f"[null] token-weighted curriculum mean depth : {weighted_mean:.4f}")
    _out(f"[null] unweighted over 16 operations        : "
         f"{payload['curriculum_mean_unweighted_over_operations']:.4f}")
    _out(f"[null] best constant (abs) c={best_abs[0]} -> abs {best_abs[1][0]:.4f}  "
         f"rel {best_abs[1][1]:.4f}")
    _out(f"[null] best constant (rel) c={best_rel[0]} -> abs {best_rel[1][0]:.4f}  "
         f"rel {best_rel[1][1]:.4f}")
    for c in (1, 2, 3):
        _out(f"[null]   c={c}: abs {scan[f'{c:.2f}'][0]:.4f}  "
             f"rel {scan[f'{c:.2f}'][1]:.4f}")
    _out(f"[null] wrote {OUT}")
    cmp_ = payload["measured_vs_null"]
    for arch in ("mor", "more"):
        c = cmp_.get(arch)
        if isinstance(c, dict):
            _out(f"[null] {arch:<4} abs {c['abs_error_mean']:.4f} "
                 f"+/- {c['abs_error_std']:.4f}  "
                 f"delta vs null {c['abs_error_minus_null']:+.4f} "
                 f"({c['abs_error_pct_worse_than_null']:+.1f}%)  "
                 f"seeds worse {c['seeds_worse_than_null_abs']}  -> {c['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
