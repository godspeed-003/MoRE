"""eval_lang_depth_control.py -- T-OD.13 TEST 2, the forced-depth control on
the language subset cells.

    <interpreter> code/eval_lang_depth_control.py --device cpu --limit 1
    <interpreter> code/eval_lang_depth_control.py --all
    <interpreter> code/eval_lang_depth_control.py --from-json

WHAT IT MEASURES. For each cell's best-val checkpoint, the SAME validation pass
is run nine times: once with the learned halting policy, then once for every
fixed depth 1..max_depth. All forced passes set `force_depth`, which (model.py)
puts step weight 1.0 on the forced step and 0.0 earlier, so the returned state is
the state AT that depth, not the ACT mixture. Then

    margin_matched = loss(forced at round(learned mean depth)) - loss(learned)
    margin_best    = loss(best forced depth)                  - loss(learned)

Both are positive when the learned policy beats a uniform one. `margin_matched`
is the estimator the language write-up used for its 24x claim; `margin_best` is
the stricter one that lets the uniform policy pick its own depth. TEST 1 on
opdiff showed they can point in OPPOSITE directions, so both are written and
neither is a footnote.

WHY IT REFUSES TO TRUST ITSELF. The first pass (force_depth=None) must reproduce
the cell's own `metrics.json` `val/task_loss`, because that number was produced
by engine.py's validation loop and this script reimplements the loop. If
|delta| > 1e-3 the cell is recorded with `learned_matches_metrics: false` and its
margins are `N/A` -- a margin computed against a loss this script cannot
reproduce is a number from a different measurement, and the whole point of the
control is that the two losses are comparable.

MoE margin is `N/A by construction` (max_depth = 1, so the depth-matched control
is the identity); it is emitted as a string, never as 0.0.

CPU is supported (`--device cpu`) because the Fortran-order attribute of running
this WHILE a training sweep owns the GPU is OOM on an 8 GB card, which would
destroy the sweep rather than just slow it down.
"""

import argparse
import glob
import json
import os
import re
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from more.config import TASK_LANGUAGE
from more.engine import _task_loss
from more.lang_data import MoRELanguageDataset
from more.model import MoREModel

CODE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(CODE)
RUNS = os.path.join(REPO, "runs")
OUT_JSON = os.path.join(REPO, "results", "language",
                        "depth_control_langsub.json")
OUT_MD = os.path.join(REPO, "results", "language",
                      "depth_control_langsub.md")

SUB_RE = re.compile(
    r"^langSub_(MoE|MoR|MoRE)_(f(?:025|100|400))_seed(\d\d)"
    r"(?:__[0-9a-f]{8})(?:__r\d+)?$")
CANON_RE = re.compile(r"^langB_(MoE|MoR|MoRE)_seed(\d\d)__[0-9a-f]{8}$")
FRAC_OF_TAG = {"f025": 0.025, "f100": 0.10, "f400": 0.40}


def out(msg):
    print(msg, flush=True)


def build_model(cfg, device):
    mc, dc = cfg["model"], cfg["data"]
    return MoREModel(
        step_feat_dim=mc.get("step_feat_dim", 8),
        d_model=mc["d_model"],
        num_experts=mc["num_experts"],
        max_depth=mc["max_depth"],
        num_blocks=mc["num_blocks"],
        dropout=mc.get("dropout", 0.0),
        fixed_depth=mc.get("fixed_depth", False),
        routing_mode=mc.get("routing_mode", "top1"),
        routing_persistence=mc.get("routing_persistence", "step"),
        router_noise=mc.get("router_noise", "none"),
        ffn_mult=mc.get("ffn_mult", 4),
        num_families=6,
        attention=bool(mc.get("attention", False)),
        n_heads=int(mc.get("n_heads", 4)),
        max_seq_len=int(dc["seq_len"]),
        task=TASK_LANGUAGE,
        vocab_size=int(dc["vocab_size"]),
    ).to(device)


class Forced:
    """Wrap a model so engine.py's call shape can carry a forced depth."""

    def __init__(self, model, force_depth):
        self.model = model
        self.force_depth = force_depth

    def __call__(self, *a):
        return self.model(*a, force_depth=self.force_depth)


def val_loss_of(model, cfg, device, batch_size, force_depth):
    ds = MoRELanguageDataset(cfg["data"].get("corpus", "wikitext-103"), "val")
    loader = torch.utils.data.DataLoader(
        ds, batch_size=min(batch_size, len(ds)), shuffle=False, num_workers=0)
    call = Forced(model, force_depth) if force_depth is not None else model
    total = 0.0
    n = 0
    depth_sum = 0.0
    depth_n = 0
    forced_exit = 0.0
    early_exit = 0.0
    with torch.no_grad():
        for x_v, sm_v, se_v, so_v, *_rest in loader:
            x_v, sm_v, se_v, so_v = (x_v.to(device), sm_v.to(device),
                                     se_v.to(device), so_v.to(device))
            res = call(x_v, sm_v, se_v, so_v)
            reg_v = res[0]
            depth_exits = res[5]
            halt = res[12] if len(res) > 12 else None
            l_v = _task_loss(reg_v, x_v, None, TASK_LANGUAGE)
            total += float(l_v) * x_v.shape[0]
            n += x_v.shape[0]
            if depth_exits is not None:
                # mean recursion depth over real tokens, exactly as model.py
                # computes avg_depth: sum over steps of d * P(exit at d)
                steps = torch.arange(1, depth_exits.shape[-1] + 1,
                                     dtype=torch.float32, device=device)
                depth_sum += float(
                    (depth_exits * steps).sum(dim=-1).sum()) * 1.0
                depth_n += int(depth_exits.shape[0])
            if isinstance(halt, dict):
                forced_exit += float(halt.get("forced_exits", 0.0))
                early_exit += float(halt.get("early_exits", 0.0))
    loss = total / max(n, 1)
    depth_mean = depth_sum / max(depth_n, 1)
    denom = forced_exit + early_exit
    return {
        "val_task_loss": loss,
        "depth_mean": depth_mean,
        "forced_exit_rate": (forced_exit / denom) if denom else None,
        "early_exit_rate": (early_exit / denom) if denom else None,
    }


def cell_dirs(pattern, include_canonical):
    dirs = []
    for d in sorted(glob.glob(os.path.join(RUNS, "langSub_*"))):
        name = os.path.basename(d)
        m = SUB_RE.match(name)
        if not m:
            continue
        mp = os.path.join(d, "metrics.json")
        ck = os.path.join(d, "checkpoint.pt")
        if not (os.path.exists(mp) and os.path.exists(ck)):
            continue
        dirs.append((d, m.group(1), FRAC_OF_TAG[m.group(2)], int(m.group(3))))
    if include_canonical:
        for d in sorted(glob.glob(os.path.join(RUNS, "langB_*"))):
            name = os.path.basename(d)
            m = CANON_RE.match(name)
            if not m:
                continue
            mp = os.path.join(d, "metrics.json")
            ck = os.path.join(d, "checkpoint.pt")
            if not (os.path.exists(mp) and os.path.exists(ck)):
                continue
            dirs.append((d, m.group(1), 1.0, int(m.group(2))))
    return dirs


def evaluate_cell(d, arm, frac, seed, device, verbose=True):
    cfg = json.load(open(os.path.join(d, "resolved_config.json"),
                         encoding="utf-8"))
    met = json.load(open(os.path.join(d, "metrics.json"), encoding="utf-8"))
    model = build_model(cfg, device)
    state = torch.load(os.path.join(d, "checkpoint.pt"), map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()
    bs = int(cfg["training"]["batch_size"])
    max_depth = int(cfg["model"]["max_depth"])

    t0 = time.time()
    learned = val_loss_of(model, cfg, device, bs, None)
    reported = met.get("val/task_loss")
    delta = (abs(learned["val_task_loss"] - reported)
             if isinstance(reported, (int, float)) else None)
    matches = delta is not None and delta <= 1e-3

    forced = {}
    if arm != "MoE" and max_depth > 1:
        for dep in range(1, max_depth + 1):
            forced[dep] = val_loss_of(model, cfg, device, bs, dep)["val_task_loss"]
    d_hat = max(1, min(max_depth, int(round(learned["depth_mean"]))))
    margin_matched = (forced[d_hat] - learned["val_task_loss"]
                      if forced else "N/A by construction")
    if forced:
        bd = min(forced, key=lambda k: forced[k])
        margin_best = forced[bd] - learned["val_task_loss"]
    else:
        bd, margin_best = 1, "N/A by construction"

    row = {
        "run_dir": os.path.basename(d),
        "arm": arm,
        "frac": frac,
        "seed": seed,
        "max_depth": max_depth,
        "learned_val_task_loss": learned["val_task_loss"],
        "learned_mean_depth": learned["depth_mean"],
        "learned_forced_exit_rate": learned["forced_exit_rate"],
        "metrics_val_task_loss": reported,
        "learned_vs_metrics_delta": delta,
        "learned_matches_metrics": bool(matches) if delta is not None else None,
        "forced_val_task_loss": {str(k): v for k, v in forced.items()},
        "depth_matched_d": d_hat,
        "margin_matched": margin_matched if matches else "N/A (loss mismatch)",
        "best_uniform_depth": bd if forced else "N/A",
        "margin_best": margin_best if matches else "N/A (loss mismatch)",
        "seconds": round(time.time() - t0, 1),
    }
    if verbose:
        out(f"  {row['run_dir']}: learned {row['learned_val_task_loss']:.4f} "
            f"(metrics {reported}) delta {delta:.2e} d={row['learned_mean_depth']:.2f} "
            f"matched {row['margin_matched'] if isinstance(row['margin_matched'], str) else round(row['margin_matched'], 4)} "
            f"best {row['margin_best'] if isinstance(row['margin_best'], str) else round(row['margin_best'], 4)} "
            f"({row['seconds']}s)")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return row


def aggregate(rows):
    arms = {}
    for arm in ("MoE", "MoR", "MoRE"):
        for frac in (0.025, 0.10, 0.40):
            rs = [r for r in rows if r["arm"] == arm and r["frac"] == frac]
            if not rs:
                continue
            entry = {"n": len(rs), "seeds": sorted(r["seed"] for r in rs)}
            for key in ("margin_matched", "margin_best"):
                vals = [r[key] for r in rs if isinstance(r[key], (int, float))]
                if vals and len(vals) == len(rs):
                    entry[key] = statistics.mean(vals)
                    entry[key + "_std"] = (statistics.stdev(vals)
                                           if len(vals) > 1 else 0.0)
                    entry[key + "_median"] = statistics.median(vals)
                else:
                    entry[key] = "N/A (by construction)" if arm == "MoE" \
                        else "N/A"
            arms.setdefault(arm, {})[f"{frac:.3f}"] = entry
    ratios = {}
    for frac in ("0.025", "0.100", "0.400"):
        try:
            mo = arms["MoRE"][frac]["margin_matched"]
            mr = arms["MoR"][frac]["margin_matched"]
            ratios[frac] = (mo / mr) if isinstance(mo, float) and \
                isinstance(mr, float) and mr else "N/A"
        except KeyError:
            ratios[frac] = "N/A"
    return {
        "_README": (
            "T-OD.13 TEST 2 forced-depth control on the language subset cells. "
            "margin = uniform_loss - learned_loss (nats/token); positive means "
            "the learned policy is better. `margin_matched` is evaluated at "
            "round(learned mean depth), the estimator the language write-up "
            "used; `margin_best` lets the uniform policy choose its own depth "
            "and is the stricter control. A cell whose learned loss does not "
            "reproduce its metrics.json within 1e-3 gets N/A margins. MoE is "
            "N/A by construction (max_depth 1)."),
        "source": "code/eval_lang_depth_control.py",
        "dataset_version": "lang-wikitext-103-bpe8192-len256-800d6154",
        "arms": arms,
        "more_over_mor_margin_ratio_by_scale": ratios,
        "runs": rows,
    }


def write_md(payload):
    rows = payload["runs"]
    arms = payload["arms"]
    lines = [
        "# Language subset sweep -- forced-depth control (TEST 2b)",
        "",
        f"{len(rows)} cells evaluated from `checkpoint.pt`. "
        "Margins in nats/token; positive = learned halting beats a uniform",
        "policy. `matched` is at round(learned mean depth); `best` lets the",
        "uniform policy pick its own depth.",
        "",
        "| arm | scale | n | margin matched | margin best | learned loss |",
        "|---|---|---|---|---|---|",
    ]
    for arm in ("MoE", "MoR", "MoRE"):
        for frac in ("0.025", "0.100", "0.400"):
            e = arms.get(arm, {}).get(frac)
            if not e:
                continue
            def c(k):
                v = e.get(k)
                if isinstance(v, float):
                    return f"{v:.4f} +/- {e.get(k + '_std', 0.0):.4f}"
                return str(v)
            learned = statistics.mean(
                [r["learned_val_task_loss"] for r in rows
                 if r["arm"] == arm and abs(r["frac"] - float(frac)) < 1e-9])
            lines.append(f"| {arm} | {float(frac):.1%} | {e['n']} | "
                         f"{c('margin_matched')} | {c('margin_best')} | "
                         f"{learned:.4f} |")
    lines += ["", "## MoRE / MoR margin ratio by scale", "",
              "| scale | ratio (matched) | ratio (best) |",
              "|---|---|---|"]
    for frac, ratio in payload["more_over_mor_margin_ratio_by_scale"].items():
        try:
            mo_b = arms["MoRE"][frac]["margin_best"]
            mr_b = arms["MoR"][frac]["margin_best"]
            rb = f"{mo_b / mr_b:.2f}x" if isinstance(mo_b, float) and \
                isinstance(mr_b, float) and mr_b else "N/A"
        except KeyError:
            rb = "N/A"
        rs = f"{ratio:.2f}x" if isinstance(ratio, float) else str(ratio)
        lines.append(f"| {float(frac):.1%} | {rs} | {rb} |")
    lines += ["", "## Cells whose learned loss did not reproduce metrics.json", ""]
    bad = [r for r in rows if r.get("learned_matches_metrics") is False]
    if not bad:
        lines.append("None -- every learned pass is within 1e-3 of the cell's "
                     "recorded val/task_loss.")
    else:
        for r in bad:
            lines.append(f"- `{r['run_dir']}`: this script "
                         f"{r['learned_val_task_loss']:.6f} vs metrics "
                         f"{r['metrics_val_task_loss']!r} "
                         f"(delta {r['learned_vs_metrics_delta']:.2e})")
    lines += ["", "## Per-cell forced curves", "",
              "| run | arm | scale | seed | learned | " +
              " | ".join(f"d={k}" for k in range(1, 8)) + " | d_hat | matched | best |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: (r["arm"], r["frac"], r["seed"])):
        fr = [r["forced_val_task_loss"].get(str(k)) for k in range(1, 8)]
        def f(v):
            return f"{v:.4f}" if isinstance(v, (int, float)) else "N/A"
        lines.append(
            f"| `{r['run_dir']}` | {r['arm']} | {r['frac']:.1%} | {r['seed']} | "
            f"{f(r['learned_val_task_loss'])} | " + " | ".join(f(v) for v in fr)
            + f" | {r['depth_matched_d']} | {f(r['margin_matched'])} | "
            f"{f(r['margin_best'])} |")
    with open(OUT_MD, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--include-canonical", action="store_true",
                    help="also evaluate runs/langB_* (cross-check path)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    ap.add_argument("--from-json", action="store_true")
    args = ap.parse_args(argv)

    if args.from_json:
        payload = aggregate(json.load(open(OUT_JSON, encoding="utf-8"))["runs"])
        json.dump(payload, open(OUT_JSON, "w", encoding="utf-8"), indent=2)
        write_md(payload)
        out(f"re-aggregated {len(payload['runs'])} cells -> {OUT_MD.name}")
        return 0

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        out("cuda requested but unavailable")
        return 2
    cells = cell_dirs(None, args.include_canonical)
    if args.limit:
        cells = cells[:args.limit]
    out(f"{len(cells)} cells on {device}")
    rows = []
    for d, arm, frac, seed in cells:
        try:
            rows.append(evaluate_cell(d, arm, frac, seed, device))
        except Exception as e:
            out(f"  [FAIL] {os.path.basename(d)}: {type(e).__name__}: {e}")
            raise
        payload = aggregate(rows)
        os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
        json.dump(payload, open(OUT_JSON, "w", encoding="utf-8"), indent=2)
        write_md(payload)
    out(f"-> {OUT_JSON.name}, {OUT_MD.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
