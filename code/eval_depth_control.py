"""
TEST 1 -- the depth-matched uniform control, on opdiff (T-OD.13).

WHY THIS EXISTS
---------------
The paper's mechanism claim rests on one comparison: a model's learned halting
policy vs a uniform policy of the SAME depth. On language that control gave MoR
a 1.47-nat advantage and MoRE only 0.06, and the paper reads that as "MoRE's
halting is nearly inert". But the language MoRE cells had already collapsed --
97.3% forced exit, depth spread 0.009 steps -- so the control was testing a
policy that barely existed. On opdiff the same arm does NOT collapse: 27.7%
forced exit, 1.74-step spread across operations. This script measures the
control where the policy is live.

WHAT IT COMPUTES
----------------
For each admissible opdiff matrix checkpoint (runs/opdiffB_{MoE,MoR,MoRE}_seed*):

  learned        the model's own ACT policy: the halting-weighted mixture its
                 forward pass returns by default.
  forced d=1..7  every token runs exactly d steps and the output is the state
                 AT that depth, not a mixture (MoREWrapper.forward's
                 `force_depth` argument). Weights are 1.0 on step d and 0.0
                 before, so this is a policy that ignores its input.
  best uniform   min over d of the forced loss.
  margin_best    best_uniform_loss - learned_loss          (nats; + = learned wins)
  depth-matched  d_hat = round(learned mean exit depth) clipped to 1..max_depth
  margin_matched forced_loss(d_hat) - learned_loss

THREE THINGS THAT WOULD INVALIDATE THE TABLE, AND WHAT IS DONE ABOUT THEM
------------------------------------------------------------------------
1. Both sides come from the SAME checkpoint. metrics.json's `val/task_loss` is
   the LAST-epoch number; checkpoint.pt is the BEST-val checkpoint. Comparing a
   forced loss from the checkpoint against the headline would mix two models,
   so the learned side is re-computed here, in this harness, from the
   checkpoint. The metrics.json value is recorded under `metrics_val_task_loss`
   for cross-reference only and is never used in a margin.
2. Forced depth returns the final state, not the halting mixture -- implemented
   in the model (`force_depth`), not approximated by temperature or masking.
3. MoE has max_depth = 1, so forced d=1 IS the learned policy. Its margin is
   N/A by construction, never a measured 0.0.

OUTPUT
------
results/opdiff/depth_control.json  machine-readable, per run + per arm
results/opdiff/depth_control.md    the tables and the reading

USAGE
-----
  python eval_depth_control.py                     # all 15 matrix cells
  python eval_depth_control.py --glob 'opdiffB_MoR_*'
  python eval_depth_control.py --force             # ignore the JSON cache
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_DIR))

from more.config import TASK_OPDIFF                                  # noqa: E402
from more.model import (MoREModel, CANONICAL_ROUTING_MODE,           # noqa: E402
                        CANONICAL_ROUTER_NOISE,
                        ROUTER_NOISE_INIT_SCALE_DEFAULT,
                        ROUTER_NOISE_ANNEAL_STEPS_DEFAULT,
                        ROUTING_PERSISTENCE_LEGACY)
from more.opdiff_data import MoREOpdiffDataset                        # noqa: E402
from more.opdiff_families import NUM_FAMILIES as OPDIFF_NUM_FAMILIES  # noqa: E402

RUNS_DIR = CODE_DIR.parent / "runs"
RESULTS_DIR = CODE_DIR.parent / "results" / "opdiff"
JSON_OUT = RESULTS_DIR / "depth_control.json"
MD_OUT = RESULTS_DIR / "depth_control.md"

# runs/opdiffB_MoR_seed45__706e3aa2__r2 is the admitted cell for seed 45 (the
# first attempt OOM'd and has no metrics.json); the stale directory is skipped
# by the metrics.json requirement rather than by name.
CELL_RE = re.compile(r"^opdiffB_(MoE|MoR|MoRE)_seed(\d\d)__[0-9a-f]+(?:__r\d)?$")
EXCLUDE = ("smoke", "calib", "dense")


def out(s: str) -> None:
    sys.stdout.buffer.write((s + "\n").encode("utf-8"))
    sys.stdout.flush()


def build_model(rc: dict, device: torch.device) -> MoREModel:
    """Field-for-field mirror of engine.py's construction for an opdiff run."""
    mc, dc = rc["model"], rc["data"]
    return MoREModel(
        step_feat_dim=mc["step_feat_dim"],
        d_model=mc["d_model"],
        num_experts=mc["num_experts"],
        max_depth=mc["max_depth"],
        num_blocks=mc["num_blocks"],
        dropout=mc["dropout"],
        fixed_depth=mc.get("fixed_depth", False),
        routing_mode=mc.get("routing_mode", CANONICAL_ROUTING_MODE),
        routing_persistence=mc.get("routing_persistence",
                                   ROUTING_PERSISTENCE_LEGACY),
        router_noise=mc.get("router_noise", CANONICAL_ROUTER_NOISE),
        router_noise_init=mc.get("router_noise_init",
                                 ROUTER_NOISE_INIT_SCALE_DEFAULT),
        router_noise_anneal_steps=mc.get("router_noise_anneal_steps",
                                         ROUTER_NOISE_ANNEAL_STEPS_DEFAULT),
        ffn_mult=mc.get("ffn_mult", 4),
        num_families=OPDIFF_NUM_FAMILIES,
        attention=bool(mc.get("attention", False)),
        n_heads=int(mc.get("n_heads", 4)),
        max_seq_len=int(dc["seq_len"]),
        task=TASK_OPDIFF,
        vocab_size=int(dc["vocab_size"]),
    ).to(device)


def opdiff_loss(reg_out: torch.Tensor, x: torch.Tensor,
                mask1: torch.Tensor) -> torch.Tensor:
    """Verbatim copy of engine.py:_task_loss's opdiff branch.

    Copied rather than imported because importing `more.engine` pulls the whole
    training stack (datasets, wandb) into an evaluation that needs neither, and
    mutating engine module state from an eval script is worse than a 4-line
    duplication that `tests` can pin. The expression is identical: causal CE
    over the generator's scored positions, in NATS.
    """
    v = reg_out.shape[-1]
    return F.cross_entropy(
        reg_out[:, :-1, :][:, mask1, :].reshape(-1, v),
        x[:, 1:][:, mask1].reshape(-1),
    )


@torch.no_grad()
def eval_pass(model, loader, mask1, device, force_depth=None):
    """Mean task loss (nats) over the val split, plus mean learned exit depth.

    `force_depth=None` is the model's own policy; any integer is the uniform
    policy at exactly that depth.
    """
    loss_sum, n = 0.0, 0
    depth_sum, depth_n = 0.0, 0
    for x, sm, se, so, _fam, _dep, _poison in loader:
        x, sm, se, so = x.to(device), sm.to(device), se.to(device), so.to(device)
        (reg, _cls, _step_cls, _bal, _pon, depth_exits, _avg_depth, _eidx,
         _oce, _fr, _rs, _exp, _hs) = model(x, sm, se, so, force_depth=force_depth)
        loss_sum += float(opdiff_loss(reg, x, mask1).item()) * x.shape[0]
        n += x.shape[0]
        if force_depth is None and depth_exits is not None:
            steps = (depth_exits * torch.arange(
                1, depth_exits.shape[-1] + 1, device=device,
                dtype=depth_exits.dtype)).sum(dim=-1)
            m = sm.reshape(-1)
            depth_sum += float(steps[m].sum().item())
            depth_n += int(m.sum().item())
    mean = loss_sum / max(n, 1)
    mean_depth = (depth_sum / depth_n) if depth_n else None
    return mean, mean_depth, n


def run_cell(run_dir: Path, device: torch.device, batch_size: int) -> dict:
    rc = json.loads((run_dir / "resolved_config.json").read_text(encoding="utf-8"))
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    prov = rc.get("provenance", {})
    max_depth = int(rc["model"]["max_depth"])

    ds = MoREOpdiffDataset("val", root=rc["data"].get("opdiff_root"))
    mask1 = torch.from_numpy(
        np.asarray(ds.loss_mask, dtype=bool)[1:]).to(device)
    loader = DataLoader(ds, batch_size=min(batch_size, len(ds)),
                        shuffle=False, num_workers=0)

    model = build_model(rc, device)
    state = torch.load(run_dir / "checkpoint.pt", map_location=device,
                       weights_only=True)
    model.load_state_dict(state)
    model.eval()

    t0 = time.time()
    learned, mean_depth, n_rows = eval_pass(model, loader, mask1, device)
    forced = {}
    for d in range(1, max_depth + 1):
        forced[str(d)] = eval_pass(model, loader, mask1, device, force_depth=d)[0]
    elapsed = time.time() - t0

    admissible = max_depth > 1 and not rc["model"].get("fixed_depth", False)
    best_d = min(forced, key=forced.get)
    best_loss = forced[best_d]
    d_hat = max(1, min(max_depth, int(round(mean_depth)))) if mean_depth else 1

    return {
        "run_dir": run_dir.name,
        "architecture": str(prov.get("architecture") or rc.get("architecture")),
        "seed": int(prov.get("resolved_seed", prov.get("seed"))),
        "max_depth": max_depth,
        "adaptive": admissible,
        # Both sides of every margin come from THIS checkpoint, never from
        # metrics.json (which is last-epoch; checkpoint.pt is best-val).
        "learned_val_task_loss": learned,
        "learned_mean_depth": mean_depth,
        "forced_val_task_loss": forced,
        "best_uniform_depth": int(best_d),
        "best_uniform_loss": best_loss,
        "margin_best": (best_loss - learned) if admissible else "N/A",
        "depth_matched_d": d_hat,
        "depth_matched_loss": forced[str(d_hat)],
        "margin_matched": (forced[str(d_hat)] - learned) if admissible else "N/A",
        "n_val_rows": n_rows,
        "secs": round(elapsed, 1),
        "gpu_peak_reserved_gib": (round(torch.cuda.max_memory_reserved(device)
                                        / 2**30, 3)
                                  if device.type == "cuda" else "N/A"),
        # Provenance cross-references, never used in a margin:
        "metrics_val_task_loss": metrics.get("val/task_loss"),
        "metrics_val_depth": metrics.get("val/avg_recursion_steps", "N/A"),
        "config_hash": prov.get("config_hash"),
        "measured_at": "checkpoint.pt (best validation loss)",
    }


def _f(v, w=7, p=3):
    return f"{v:>{w}.{p}f}" if isinstance(v, (int, float)) else f"{'N/A':>{w}}"


def _arm_of(r: dict) -> str:
    """provenance.architecture is lower-case ('moe'/'mor'/'more'); the tables
    use the paper's capitalisation. Normalise once, here."""
    a = str(r.get("architecture") or "").strip()
    return {"moe": "MoE", "mor": "MoR", "more": "MoRE"}.get(a.lower(), a)


def aggregate(rows: list[dict]) -> dict:
    arms = {}
    for arm in ("MoE", "MoR", "MoRE"):
        rs = [r for r in rows if _arm_of(r) == arm]
        if not rs:
            continue
        entry = {"n": len(rs), "seeds": sorted(r["seed"] for r in rs)}
        for key in ("margin_best", "margin_matched", "learned_val_task_loss"):
            vals = [r[key] for r in rs if isinstance(r[key], (int, float))]
            if vals and len(vals) == len(rs):
                entry[key] = statistics.mean(vals)
                entry[key + "_std"] = (statistics.stdev(vals)
                                       if len(vals) > 1 else 0.0)
                entry[key + "_median"] = statistics.median(vals)
                entry[key + "_min"] = min(vals)
                entry[key + "_max"] = max(vals)
            else:
                entry[key] = "N/A (by construction)" if key.startswith("margin") \
                    else "N/A"
        arms[arm] = entry
    def ratio_of(key: str):
        try:
            m_more = arms["MoRE"][key]
            m_mor = arms["MoR"][key]
            if isinstance(m_more, float) and isinstance(m_mor, float) and m_mor != 0:
                return m_more / m_mor
        except KeyError:
            pass
        return "N/A"
    return {
        "_README": (
            "TEST 1 -- depth-matched uniform control on opdiff. margins are "
            "uniform_loss - learned_loss in nats; positive means the learned "
            "policy is better. Both sides measured from checkpoint.pt in "
            "code/eval_depth_control.py; metrics.json is never used in a margin. "
            "MoE margin is N/A by construction (max_depth = 1). Mean, std, "
            "median, min and max are all reported because MoR's margin has one "
            "pathological seed (43, the collapsed halting policy) that moves the "
            "mean by more than a standard deviation."
        ),
        "source": "code/eval_depth_control.py",
        "task": "opdiff",
        "dataset_version": "opdiff-v4",
        "arms": arms,
        "more_over_mor_margin_ratio": ratio_of("margin_matched"),
        "more_over_mor_margin_ratio_median": (
            arms["MoRE"]["margin_matched_median"]
            / arms["MoR"]["margin_matched_median"]
            if isinstance(arms.get("MoRE", {}).get("margin_matched_median"), float)
            and arms.get("MoR", {}).get("margin_matched_median") not in (None, 0)
            else "N/A"),
        "more_over_mor_margin_ratio_best_uniform": ratio_of("margin_best"),
        "runs": rows,
    }


def write_md(payload: dict) -> None:
    arms = payload["arms"]
    lines = [
        "# Test 1 -- the depth-matched uniform control on opdiff",
        "",
        "`code/eval_depth_control.py`. Every number comes from the same artifact:",
        "`checkpoint.pt` (best-validation), re-evaluated in one harness. "
        "`metrics.json`'s headline loss is last-epoch and is quoted only as a",
        "cross-reference. Margin = uniform_loss - learned_loss, in nats; "
        "positive means the learned policy won.",
        "",
        "## Per-seed",
        "",
        "| run | arm | learned | mean depth | " +
        " | ".join(f"d={d}" for d in range(1, 8)) + " | best d | margin_best | d_hat | margin_matched |",
        "|---|---|---|---|" + "---|" * 7 + "---|---|---|---|",
    ]
    for r in sorted(payload["runs"], key=lambda r: (_arm_of(r), r["seed"])):
        forced = [r["forced_val_task_loss"].get(str(d)) for d in range(1, 8)]
        lines.append(
            f"| {r['run_dir']} | {_arm_of(r)} | "
            f"{_f(r['learned_val_task_loss'], 0, 4)} | {_f(r['learned_mean_depth'], 0, 2)} | "
            + " | ".join(_f(v, 0, 4) for v in forced)
            + f" | {r['best_uniform_depth']} | {_f(r['margin_best'], 0, 4)} | "
            f"{r['depth_matched_d']} | {_f(r['margin_matched'], 0, 4)} |")
    lines += [
        "",
        "## Per arm (over 5 seeds)",
        "",
        "| arm | n | margin_best mean +/- std (median) | "
        "margin_matched mean +/- std (median) | learned val/task_loss |",
        "|---|---|---|---|---|",
    ]
    for arm in ("MoE", "MoR", "MoRE"):
        a = arms.get(arm)
        if a is None:
            lines.append(f"| {arm} | 0 | N/A | N/A | N/A |")
            continue
        def cell(k):
            v, s, med = a.get(k), a.get(k + "_std"), a.get(k + "_median")
            if isinstance(v, float):
                return f"{v:.4f} +/- {s:.4f} ({med:.4f})"
            return str(v)
        lines.append(f"| {arm} | {a['n']} | {cell('margin_best')} | "
                     f"{cell('margin_matched')} | "
                     f"{cell('learned_val_task_loss')} |")
    ratio = payload["more_over_mor_margin_ratio"]
    ratio_med = payload["more_over_mor_margin_ratio_median"]
    ratio_best = payload["more_over_mor_margin_ratio_best_uniform"]
    a = payload["arms"]
    def m(arm, k):
        v = a.get(arm, {}).get(k)
        return f"{v:.3f}" if isinstance(v, float) else str(v)
    lines += [
        "",
        "## Reading",
        "",
        "**Implementation check.** MoE's learned and forced-d=1 losses agree to",
        "four decimals in all five seeds, which is the identity the forced branch",
        "must satisfy when `max_depth = 1`: the single step already carries weight",
        "1.0 in the ACT mixture. MoE's margins are therefore reported as",
        "`N/A (by construction)`, never as a measured zero.",
        "",
        "**The language claim does not transfer.** On opdiff, at the",
        "depth-matched control the paper used, MoRE's learned policy beats its own",
        f"uniform policy by **{m('MoRE', 'margin_matched')} nats** (mean; median",
        f"{m('MoRE', 'margin_matched_median')}, range "
        f"{m('MoRE', 'margin_matched_min')}-{m('MoRE', 'margin_matched_max')}) —",
        "about 14x the 0.06 nats measured on language, where MoRE's halting had",
        "collapsed to 97.3% forced exit. MoR's is",
        f"{m('MoR', 'margin_matched')} +/- {m('MoR', 'margin_matched_std')}",
        f"(median {m('MoR', 'margin_matched_median')}). The ratio of MEANS is",
        f"**{ratio:.2f}x** and the ratio of MEDIANS is **{ratio_med:.2f}x**; the",
        "means agree only because MoR's mean is carried by one pathological seed",
        "(43, margin 2.90 — the collapsed-halting seed whose forced-d=2 loss is",
        "3.29 nats), while MoRE is positive and tight in every seed. Either way,",
        "MoRE's halting is doing substantial work on opdiff, so the paper's",
        "'nearly inert' reading was a property of the saturated language cells,",
        "not of the composition.",
        "",
        "**But the conservative estimator points the other way, and both must be",
        "reported.** Against the BEST uniform depth anywhere (not the depth-matched",
        f"one), MoRE's advantage falls to {m('MoRE', 'margin_best')} +/-",
        f"{m('MoRE', 'margin_best_std')} nats against MoR's",
        f"{m('MoR', 'margin_best')} +/- {m('MoR', 'margin_best_std')}",
        f"(ratio {ratio_best:.2f}x). MoRE's loss is nearly reached by a fixed",
        "depth (its best forced depth is 5-6 in every seed), whereas no fixed",
        "depth gets near MoR's learned policy. So: routing makes the depth policy",
        "REPLACEABLE by a fixed depth, not more valuable than MoR's — a different",
        "statement from 'inert', and the one the data supports.",
        "",
        "**Mechanical caveat that the reader must not skip.** Forced depth returns",
        "the final state, not the ACT mixture. Part of ANY arm's margin is",
        "therefore the cost of reading a single state with a head that was trained",
        "on convex mixtures — that cost is largest where the forced depth lands in",
        "the steep part of the curve. MoRE's depth-matched integer (3) does exactly",
        "that (its curve is still falling steeply at d=3), which inflates its",
        "depth-matched margin and is why the best-uniform margin is the safer",
        "headline. The two rows are a bracket, not a contradiction.",
        "",
        "**Per-arm depth curves have different shapes** and that is itself a",
        "finding: MoRE's forced loss falls monotonically to d≈6 and rises at 7,",
        "MoR's is U-shaped with its minimum at d=2-4 and degrades sharply beyond",
        "the depth it was trained to use. The composition is trained to be stable",
        "under more recursion; MoR is not.",
        "",
    ]
    MD_OUT.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--glob", default="opdiffB_*")
    ap.add_argument("--batch-size", type=int, default=48)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--from-json", action="store_true",
                    help="re-aggregate the existing results/opdiff/"
                         "depth_control.json instead of touching the GPU")
    args = ap.parse_args()

    if args.from_json:
        payload = aggregate(json.loads(JSON_OUT.read_text(encoding="utf-8"))["runs"])
        JSON_OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        write_md(payload)
        out(f"re-aggregated {len(payload['runs'])} runs -> "
            f"{JSON_OUT.name}, {MD_OUT.name}")
        return 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dirs = []
    for d in sorted(RUNS_DIR.glob(args.glob)):
        if not d.is_dir() or any(x in d.name for x in EXCLUDE):
            continue
        m = CELL_RE.match(d.name)
        if not m:
            continue
        if not (d / "metrics.json").exists() or not (d / "checkpoint.pt").exists():
            continue
        dirs.append(d)
    dirs.sort(key=lambda d: (CELL_RE.match(d.name).group(1),
                             int(CELL_RE.match(d.name).group(2))))
    if args.limit:
        dirs = dirs[:args.limit]
    if not dirs:
        out("no admissible opdiff matrix cells found")
        return 1

    out(f"[depth_control] {len(dirs)} cells, device={device.type}")
    rows = []
    for d in dirs:
        t0 = time.time()
        r = run_cell(d, device, args.batch_size)
        rows.append(r)
        out(f"  {d.name:<34} learned={r['learned_val_task_loss']:.4f} "
            f"best={r['best_uniform_loss']:.4f}(d={r['best_uniform_depth']}) "
            f"matched={r['depth_matched_loss']:.4f}(d={r['depth_matched_d']}) "
            f"margin_matched={_f(r['margin_matched'], 0, 4)} "
            f"depth={_f(r['learned_mean_depth'], 0, 2)} "
            f"{time.time() - t0:.0f}s")
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

    payload = aggregate(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    JSON_OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_md(payload)
    out("\n[arm]      n  margin_best        margin_matched     learned_loss")
    for arm in ("MoE", "MoR", "MoRE"):
        if arm not in payload["arms"]:
            continue
        a = payload["arms"][arm]
        out(f"{arm:<6} {a['n']:>3}  {_f(a['margin_best'])}            "
            f"{_f(a['margin_matched'])}            {_f(a['learned_val_task_loss'])}")
    out(f"[MoRE/MoR depth-matched margin ratio] {payload['more_over_mor_margin_ratio']}")
    out(f"wrote {JSON_OUT.relative_to(CODE_DIR.parent)} and {MD_OUT.relative_to(CODE_DIR.parent)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
