"""
Offline validation-pass depth metrics for runs that predate them (T11.1, audit item 7).

WHY THIS EXISTS
---------------
Every depth number in the canonical matrix was accumulated inside the TRAINING
loop: `train/avg_recursion_steps`, `depth/allocation_error_abs`,
`depth/allocation_error_rel`, `halt/*`. Those are measurements of the model while
it was being updated, under dropout, on training data. The paper's depth claim --
"the model learns to allocate recursion depth in accordance with the predefined
operation-complexity curriculum" (CLAUDE.md §4 phrasing) -- is a claim about the
TRAINED model on HELD-OUT data. The training-time number does not measure it.

engine.py now emits `val/avg_recursion_steps`, `val/depth_allocation_error_{abs,rel}`
and `val/{forced,early}_exit_rate` from its existing validation forward pass, so
every FUTURE run carries both. The 15 completed canonical runs and the Phase 10
arms do not, and re-training them for a metric that needs no training would cost
hours of GPU for nothing. This script recovers the same quantities from each run's
saved `checkpoint.pt`.

WHAT IT MEASURES, EXACTLY
-------------------------
The checkpoint written by engine.py is the BEST-VALIDATION-LOSS checkpoint, not
the last epoch. The canonical primary metric is last-epoch `val/task_loss`, so
these depth numbers are measured at a *different* point in training than the
headline loss. That is a real caveat and must be stated wherever these numbers
appear: `measured_at: "best_val_checkpoint"` is written into every output file so
the distinction cannot be lost in transcription. With log_interval = 2 over 50
epochs the best checkpoint is typically late, but "typically" is not "is".

Model reconstruction mirrors engine.py:202 field for field, and the val loader
mirrors engine.py:194 (no shuffle, so batch composition is irrelevant anyway).
The depth quantities come from `metrics.depth_allocation_error` and the model's
own halt stats -- the same functions the live loop calls, so there is no second
definition to drift.

OUTPUT
------
`<run_dir>/val_depth_offline.json`, one file per run. metrics.json is NEVER
modified: it is the artifact of the run, and a number computed months later by a
different script does not belong in it. The exporter may read the sidecar and
must label it as offline.

USAGE
-----
  python eval_val_depth.py                    # all runs/phaseB_* dirs
  python eval_val_depth.py --glob 't10_more_*' # a Phase 10 arm family
  python eval_val_depth.py --force            # recompute existing sidecars
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

from more.data import MoREDataset                                    # noqa: E402
from more.families import NUM_EXPERTS_CANONICAL, op_target_depth_table  # noqa: E402
from more.metrics import depth_allocation_error                      # noqa: E402
from more.model import (MoREModel, CANONICAL_ROUTING_MODE,           # noqa: E402
                        CANONICAL_ROUTER_NOISE,
                        ROUTER_NOISE_INIT_SCALE_DEFAULT,
                        ROUTER_NOISE_ANNEAL_STEPS_DEFAULT)

CODE_DIR = Path(__file__).resolve().parent
RUNS_DIR = CODE_DIR.parent / "runs"
SIDECAR = "val_depth_offline.json"


def _out(s: str) -> None:
    """cp1252-safe stdout: this file prints en-dashes and check marks."""
    sys.stdout.buffer.write(s.encode("utf-8"))
    sys.stdout.buffer.write(b"\n")
    sys.stdout.flush()


def _resolve_path(raw: str) -> Path:
    """
    Data paths in resolved_config.json are relative to code/ (`../data/val.jsonl`),
    because that is the cwd a run is launched from. Resolve against code/, not
    against cwd, so this script works from anywhere.
    """
    p = Path(raw)
    return p if p.is_absolute() else (CODE_DIR / p).resolve()


def build_model(mc: dict, step_feat_dim: int, device: torch.device) -> MoREModel:
    """Mirror of engine.py:202. Every default here is the engine's default."""
    return MoREModel(
        step_feat_dim=step_feat_dim,
        d_model=mc["d_model"],
        num_experts=mc["num_experts"],
        max_depth=mc["max_depth"],
        num_blocks=mc["num_blocks"],
        dropout=mc["dropout"],
        fixed_depth=mc.get("fixed_depth", False),
        routing_mode=mc.get("routing_mode", CANONICAL_ROUTING_MODE),
        router_noise=mc.get("router_noise", CANONICAL_ROUTER_NOISE),
        router_noise_init=mc.get("router_noise_init",
                                 ROUTER_NOISE_INIT_SCALE_DEFAULT),
        router_noise_anneal_steps=mc.get("router_noise_anneal_steps",
                                         ROUTER_NOISE_ANNEAL_STEPS_DEFAULT),
        ffn_mult=mc.get("ffn_mult", 4),
        num_families=NUM_EXPERTS_CANONICAL,
    ).to(device)


def evaluate(run_dir: Path, device: torch.device) -> tuple[dict | None, str | None]:
    """
    Returns (payload, None) or (None, reason). Never raises on a bad run dir:
    a sweep over 120 directories must not die on the one that is half-written.
    """
    rc_path = run_dir / "resolved_config.json"
    ck_path = run_dir / "checkpoint.pt"
    if not rc_path.exists():
        return None, "no resolved_config.json"
    if not ck_path.exists():
        return None, "no checkpoint.pt"
    try:
        rc = json.loads(rc_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return None, f"unreadable resolved_config.json ({exc})"

    mc = rc.get("model", {})
    tc = rc.get("training", {})
    dc = rc.get("data", {})
    prov = rc.get("provenance", {})

    val_path = _resolve_path(dc.get("val_path", "../data/val.jsonl"))
    if not val_path.exists():
        return None, f"val file missing: {val_path}"

    ds = MoREDataset(
        jsonl_path=str(val_path),
        max_steps=mc["max_steps"],
        step_feat_dim=mc["step_feat_dim"],
        max_val=dc["max_val"],
        pad_value=dc["pad_value"],
        num_experts=mc["num_experts"],
    )
    loader = DataLoader(ds, batch_size=min(int(tc["batch_size"]), len(ds)),
                        shuffle=False, num_workers=0)

    model = build_model(mc, mc["step_feat_dim"], device)
    try:
        state = torch.load(ck_path, map_location=device, weights_only=True)
        model.load_state_dict(state)
    except Exception as exc:
        # A shape mismatch here means the checkpoint and resolved_config disagree,
        # which is a provenance defect worth surfacing, not swallowing.
        return None, f"checkpoint/state_dict mismatch ({exc})"
    model.eval()

    table = op_target_depth_table()
    abs_sum = rel_sum = 0.0
    err_n = 0
    depth_sum = 0.0
    depth_n = 0
    halt_acc: dict[str, float] = {}
    halt_batches = 0
    loss_sum = 0.0
    loss_n = 0

    with torch.no_grad():
        for x, sm, se, so, _fam, _, tgt in loader:
            x, sm, se, so, tgt = (x.to(device), sm.to(device), se.to(device),
                                  so.to(device), tgt.to(device))
            (reg, _, _, _, _, _depth_exits, _, _, _, _first_route,
             _rstats, expected_depth, halt_stats) = model(x, sm, se, so)
            loss_sum += float(
                torch.nn.functional.mse_loss(reg.squeeze(-1), tgt).item()
            ) * x.shape[0]
            loss_n += x.shape[0]

            if expected_depth is not None:
                a, r = depth_allocation_error(expected_depth, so, sm, table)
                if a is not None:
                    abs_sum += a
                    rel_sum += r
                    err_n += 1
                m = sm.reshape(-1)
                if bool(m.any()):
                    depth_sum += float(expected_depth.reshape(-1)[m].sum().item())
                    depth_n += int(m.sum().item())
            if halt_stats:
                for k, v in halt_stats.items():
                    halt_acc[k] = halt_acc.get(k, 0.0) + v
                halt_batches += 1

    # "N/A", never 0.0: at max_depth == 1 / fixed depth there is no allocation to
    # measure, and a zero here would read as perfect allocation (CLAUDE.md §4).
    adaptive = mc["max_depth"] > 1 and mc.get("adaptive_halting", True) \
        and not mc.get("fixed_depth", False)
    fe = halt_acc.get("forced_exits", 0.0)
    ee = halt_acc.get("early_exits", 0.0)

    payload = {
        "_README": (
            "Offline validation-pass depth metrics, computed by "
            "code/eval_val_depth.py from checkpoint.pt AFTER the run finished. "
            "NOT part of the run's own metrics.json. measured_at is the "
            "best-validation-loss checkpoint, which is NOT necessarily the "
            "last-epoch model the canonical primary metric comes from."
        ),
        "measured_at": "best_val_checkpoint",
        "source": "code/eval_val_depth.py",
        "run_dir": run_dir.name,
        "experiment_id": prov.get("experiment_id"),
        "architecture": prov.get("architecture") or rc.get("architecture"),
        "resolved_seed": prov.get("resolved_seed", prov.get("seed")),
        "config_hash": prov.get("config_hash"),
        "val_records": len(ds),
        # Recomputed here purely as a consistency handle: it should sit at or
        # below the run's own best_val_loss (same model, same data, eval mode).
        "offline/val_task_loss": loss_sum / max(loss_n, 1),
        "val_offline/avg_recursion_steps": (
            depth_sum / depth_n if (adaptive and depth_n > 0) else "N/A"
        ),
        "val_offline/depth_allocation_error_abs": (
            abs_sum / err_n if (adaptive and err_n > 0) else "N/A"
        ),
        "val_offline/depth_allocation_error_rel": (
            rel_sum / err_n if (adaptive and err_n > 0) else "N/A"
        ),
        "val_offline/forced_exit_rate": (
            fe / (fe + ee) if (adaptive and (fe + ee) > 0) else "N/A"
        ),
        "val_offline/early_exit_rate": (
            ee / (fe + ee) if (adaptive and (fe + ee) > 0) else "N/A"
        ),
        "val_offline/mean_remainder": (
            halt_acc.get("mean_remainder", 0.0) / halt_batches
            if (adaptive and halt_batches > 0) else "N/A"
        ),
    }
    return payload, None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--glob", default="phaseB_*",
                    help="run-directory glob under runs/ (default: phaseB_*)")
    ap.add_argument("--force", action="store_true",
                    help="recompute even where val_depth_offline.json exists")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dirs = sorted(d for d in RUNS_DIR.glob(args.glob) if d.is_dir())
    if not dirs:
        _out(f"No run directories match runs/{args.glob}")
        return 1

    _out(f"[eval_val_depth] {len(dirs)} run dirs, device={device.type}")
    done = skipped = failed = 0
    rows: list[dict] = []
    for d in dirs:
        sc = d / SIDECAR
        if sc.exists() and not args.force:
            rows.append(json.loads(sc.read_text(encoding="utf-8")))
            skipped += 1
            continue
        payload, reason = evaluate(d, device)
        if payload is None:
            _out(f"  SKIP {d.name}: {reason}")
            failed += 1
            continue
        sc.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        rows.append(payload)
        done += 1
        _out(f"  ok   {d.name}")

    _out(f"\n[eval_val_depth] computed {done}, reused {skipped}, skipped {failed}")

    if rows:
        _out("\nrun                              arch  depth  abs_err  rel_err  early")
        for r in sorted(rows, key=lambda r: (r.get("architecture") or "", r["run_dir"])):
            def f(k: str, w: int = 7, p: int = 4) -> str:
                v = r.get(k)
                return f"{v:>{w}.{p}f}" if isinstance(v, (int, float)) else f"{'N/A':>{w}}"
            _out(f"{r['run_dir'][:32]:<32} {(r.get('architecture') or '?'):<5}"
                 f"{f('val_offline/avg_recursion_steps', 6, 3)}"
                 f"{f('val_offline/depth_allocation_error_abs', 9)}"
                 f"{f('val_offline/depth_allocation_error_rel', 9)}"
                 f"{f('val_offline/early_exit_rate', 7, 3)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
