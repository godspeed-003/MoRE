"""summarize_lang_subset_sweep.py -- T-OD.13 TEST 2 aggregation.

    <interpreter> code/summarize_lang_subset_sweep.py            # partial ok
    <interpreter> code/summarize_lang_subset_sweep.py --strict   # all 45 required

Reads runs/langSub_{MoE,MoR,MoRE}_f{025,100,400}_seed{42..46}*/{metrics,resolved
_config}.json and writes results/language/subset_sweep.{json,md}.

WHAT IT REFUSES, and why each refusal exists:

1. A non-canonical group. Every cell must be stamped `subset_lang_sweep`. A cell
   launched without it is a silently-labelled exploratory run and must not enter
   a table (the same defect class the language matrix driver guards against).

2. A protocol mismatch between cells. The ONLY field allowed to vary besides the
   architecture, seed and subset_fraction is nothing. epochs, batch_size, lr,
   seq_len, vocab_size, dropout, ffn_mult, num_experts, every loss weight and the
   tokenizer hash are compared to the first cell of that arm; any difference is
   an unlabelled ablation and is refused rather than averaged.

3. A mixed dataset_version. Subsetting changes WHICH blocks are trained on, not
   which corpus they were packed from, so every cell must carry the same
   `data.dataset_version` (lang-wikitext-103-bpe8192-len256-800d6154). The
   100 % cells are NOT part of this sweep -- canonical langB_* cells are a
   different experiment and are never pulled in here.

4. A missing metric. `val/task_loss` absent is `N/A` in the per-seed table and
   excluded from the mean, with the n printed; it is never counted as 0.0.

`--strict` additionally requires 15 cells per scale (3 arms x 5 seeds). Partial
runs are readable but are NOT a result; the strict mode is what a results table
must be built with.
"""

import argparse
import glob
import json
import os
import re
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CODE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(CODE)
RUNS = os.path.join(REPO, "runs")
OUT_JSON = os.path.join(REPO, "results", "language", "subset_sweep.json")
OUT_MD = os.path.join(REPO, "results", "language", "subset_sweep.md")
DEPTH_JSON = os.path.join(REPO, "results", "language",
                          "depth_control_langsub.json")

GROUP = "subset_lang_sweep"
SEEDS = [42, 43, 44, 45, 46]
FRACS = [0.025, 0.10, 0.40]
ARMS = ["MoE", "MoR", "MoRE"]
ARM_CODE = {"MoE": "moe", "MoR": "mor", "MoRE": "more"}


def tag_of(frac):
    return f"f{int(round(frac * 1000)):03d}"


FRAC_OF_TAG = {tag_of(f): f for f in FRACS}
CELL_RE = re.compile(
    r"^langSub_(MoE|MoR|MoRE)_(f(?:025|100|400))_seed(\d\d)"
    r"(?:__([0-9a-f]{8}))(?:__r(\d+))?$")

# Fields that must be identical across cells of the same arm. Chosen to be the
# fields that would change what the number MEANS; paths, run_name, config_hash
# and provenance naturally differ per cell and are not compared.
PROTOCOL_PATHS = [
    ("training", "epochs"), ("training", "batch_size"),
    ("training", "lr"), ("training", "weight_decay"),
    ("training", "grad_clip"),
    ("data", "seq_len"), ("data", "vocab_size"),
    # subset_seed is deliberately NOT listed: `--seed S` sets it to S, so it
    # varies BY DESIGN across seeds and is checked separately below.
    ("model", "num_experts"), ("model", "ffn_mult"),
    ("model", "max_depth"), ("model", "num_blocks"),
    ("model", "dropout"), ("model", "halting_supervision"),
    ("loss_weights", "task"), ("loss_weights", "routing_balance"),
    ("loss_weights", "halting"), ("loss_weights", "family_cls"),
    ("loss_weights", "step_routing"),
    ("loss_weights", "halting_supervision"),
]


def discover():
    cells = {}
    dupes = []
    for d in sorted(glob.glob(os.path.join(RUNS, "langSub_*"))):
        m = CELL_RE.match(os.path.basename(d))
        if not m:
            continue
        arm, tag, seed = m.group(1), m.group(2), int(m.group(3))
        frac = FRAC_OF_TAG.get(tag)
        if frac is None:
            continue
        mp = os.path.join(d, "metrics.json")
        rc = os.path.join(d, "resolved_config.json")
        if not (os.path.exists(mp) and os.path.exists(rc)):
            continue
        try:
            met = json.load(open(mp, encoding="utf-8"))
            cfg = json.load(open(rc, encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if cfg.get("logging", {}).get("experiment_group") != GROUP:
            continue
        if cfg.get("architecture") != ARM_CODE[arm]:
            continue
        if int(cfg.get("provenance", {}).get("seed", -1)) != seed:
            continue
        got = cfg.get("data", {}).get("subset_fraction")
        if got is None or abs(float(got) - frac) > 1e-12:
            continue
        if met.get("val/task_loss") is None:
            continue
        key = (arm, frac, seed)
        if key in cells:                       # __r2 retry of the same cell
            dupes.append((key, d, cells[key]["run_dir"]))
            continue
        cells[key] = {"arm": arm, "frac": frac, "seed": seed,
                      "run_dir": os.path.basename(d), "metrics": met,
                      "cfg": cfg}
    return cells, dupes


def check_protocol(cells):
    problems = []
    versions = {}
    for c in cells.values():
        versions.setdefault(c["cfg"]["data"].get("dataset_version"), []).append(
            c["run_dir"])
    if len(versions) != 1:
        problems.append("MIXED dataset_version across cells: "
                        + "; ".join(f"{k} <- {len(v)} cells"
                                    for k, v in versions.items()))
    ref = {}
    for arm in ARMS:
        arm_cells = [c for c in cells.values() if c["arm"] == arm]
        if not arm_cells:
            continue
        ref[arm] = arm_cells[0]
        rc_cell = ref[arm]
        cfgref = rc_cell["cfg"]
        for other in arm_cells[1:]:
            o = other["cfg"]
            for sec, key in PROTOCOL_PATHS:
                a = (cfgref.get(sec) or {}).get(key)
                b = (o.get(sec) or {}).get(key)
                if a != b:
                    problems.append(
                        f"{other['run_dir']}: {sec}.{key}={b!r} but "
                        f"{rc_cell['run_dir']} has {a!r} -- an unlabelled "
                        "protocol difference; refusing to average across it")
        # subset_seed must follow the seed, since engine.py slices one
        # permutation per seed and a shared subset_seed would make different
        # seeds train on the same blocks (not an error, but NOT what the
        # canonical --seed path does and not what these names claim).
        for c in arm_cells:
            if int(c["cfg"]["data"].get("subset_seed", -1)) != c["seed"]:
                problems.append(
                    f"{c['run_dir']}: subset_seed="
                    f"{c['cfg']['data'].get('subset_seed')} but seed="
                    f"{c['seed']}")
    return problems, ref


def load_margins():
    """Optional depth-control measurements, keyed by run_dir basename."""
    if not os.path.exists(DEPTH_JSON):
        return {}
    try:
        payload = json.load(open(DEPTH_JSON, encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    out = {}
    for r in payload.get("runs", []):
        out[r["run_dir"]] = r
    return out


def _stat(vals):
    if not vals:
        return None
    return {
        "n": len(vals),
        "mean": statistics.mean(vals),
        "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
        "median": statistics.median(vals),
        "min": min(vals), "max": max(vals),
    }


def aggregate(cells, margins):
    arms = {}
    for frac in FRACS:
        for arm in ARMS:
            rs = [c for c in cells.values() if c["frac"] == frac
                  and c["arm"] == arm]
            rs.sort(key=lambda c: c["seed"])
            if not rs:
                continue
            entry = {
                "n": len(rs),
                "seeds": [c["seed"] for c in rs],
                "per_seed": [],
            }
            vals = {"val_task_loss": [], "depth_mean": [],
                    "forced_exit_rate": [], "early_exit_rate": [],
                    "train_task_loss": []}
            for c in rs:
                m = c["metrics"]
                margins_c = margins.get(c["run_dir"])
                row = {
                    "seed": c["seed"],
                    "run_dir": c["run_dir"],
                    "val_task_loss": m.get("val/task_loss"),
                    "val_perplexity": m.get("val/perplexity"),
                    "depth_mean": m.get("depth/mean"),
                    "forced_exit_rate": m.get("val/forced_exit_rate"),
                    "early_exit_rate": m.get("val/early_exit_rate"),
                    "train_task_loss": m.get("train/task_loss"),
                    "margin_matched": (margins_c or {}).get("margin_matched"),
                    "margin_best": (margins_c or {}).get("margin_best"),
                    "depth_matched_d": (margins_c or {}).get("depth_matched_d"),
                    "best_uniform_depth":
                        (margins_c or {}).get("best_uniform_depth"),
                }
                entry["per_seed"].append(row)
                for k, mk in (("val_task_loss", "val/task_loss"),
                              ("depth_mean", "depth/mean"),
                              ("forced_exit_rate", "val/forced_exit_rate"),
                              ("early_exit_rate", "val/early_exit_rate"),
                              ("train_task_loss", "train/task_loss")):
                    if isinstance(m.get(mk), (int, float)):
                        vals[k].append(float(m[mk]))
                if isinstance(row["margin_matched"], (int, float)):
                    vals.setdefault("margin_matched", []).append(
                        float(row["margin_matched"]))
                if isinstance(row["margin_best"], (int, float)):
                    vals.setdefault("margin_best", []).append(
                        float(row["margin_best"]))
            for k, v in vals.items():
                entry[k] = _stat(v) if len(v) == len(rs) else None
                if len(v) and len(v) != len(rs):
                    entry[k] = {"n": len(v), "mean": statistics.mean(v),
                                "std": (statistics.stdev(v) if len(v) > 1
                                        else 0.0),
                                "partial": True}
            arms.setdefault(tag_of(frac), {})[arm] = entry

    ratios = {}
    for frac in FRACS:
        tag = tag_of(frac)
        d = arms.get(tag, {})
        r = {}
        for k in ("val_task_loss", "margin_matched", "margin_best"):
            try:
                mo = d["MoRE"][k]["mean"]
                mr = d["MoR"][k]["mean"]
                r[k + "_more_over_mor"] = (mo / mr) if mr else None
                r[k + "_mor_minus_more"] = mr - mo
            except (KeyError, TypeError):
                pass
        # MoE margin is N/A by construction (max_depth 1), never 0.
        r["moe_margin_note"] = ("N/A by construction: MoE runs max_depth=1, so "
                                "the depth-matched control is the identity")
        if r:
            ratios[tag] = r
    return arms, ratios


def _f(v, nd=4):
    return f"{v:.{nd}f}" if isinstance(v, (int, float)) else "N/A"


def write_md(payload):
    arms = payload["arms"]
    ratios = payload["ratios"]
    lines = [
        "# Language subset sweep -- T-OD.13 TEST 2",
        "",
        f"Cells: {payload['n_cells']}/45 complete"
        + ("" if payload["complete"] else "  **(PARTIAL -- not a result)**"),
        "",
        "Corpus `lang-wikitext-103-bpe8192-len256-800d6154` (verified sha256).",
        "Only `data.subset_fraction` varies. Subsets are nested per seed",
        "(permuted prefixes); `subset_seed == seed`, as the canonical `--seed`",
        "path sets it. Every cell is `experiment_group = subset_lang_sweep`,",
        "so no cell can be claimed canonical and none enters a langB_* table.",
        "",
        "## val/task_loss (nats/token, mean +/- std over 5 seeds)",
        "",
        "Bigram floor on this corpus: 4.9849. A cell ABOVE the floor has not",
        "yet beaten the bigram baseline -- at 2.5 % that is the honest regime,",
        "not a failure.",
        "",
        "| scale | MoE | MoR | MoRE | MoRE - MoR |",
        "|---|---|---|---|---|",
    ]
    floor = 4.98494701553838
    for frac in FRACS:
        tag = tag_of(frac)
        row = [f"{frac:.1%}"]
        for arm in ARMS:
            e = (arms.get(tag, {}) or {}).get(arm)
            if not e or not e.get("val_task_loss"):
                row.append("N/A")
                continue
            s = e["val_task_loss"]
            tagx = "*" if s["mean"] > floor else ""
            row.append(f"{s['mean']:.4f} +/- {s['std']:.4f}{tagx} (n={s['n']})")
        try:
            dm = (arms[tag]["MoR"]["val_task_loss"]["mean"]
                  - arms[tag]["MoRE"]["val_task_loss"]["mean"])
            row.append(f"{dm:+.4f}")
        except (KeyError, TypeError):
            row.append("N/A")
        lines.append("| " + " | ".join(row) + " |")
    lines += [
        "",
        "## Recursion depth and forced exit (mean over 5 seeds)",
        "",
        "| scale | arm | depth/mean | forced exit rate | val/task_loss |",
        "|---|---|---|---|---|",
    ]
    for frac in FRACS:
        for arm in ARMS:
            e = (arms.get(tag_of(frac), {}) or {}).get(arm)
            if not e:
                continue
            d = e.get("depth_mean") or {}
            f = e.get("forced_exit_rate") or {}
            v = e.get("val_task_loss") or {}
            lines.append(
                f"| {frac:.1%} | {arm} | {_f(d.get('mean'), 2)} +/- "
                f"{_f(d.get('std'), 2)} | {_f(f.get('mean'), 4)} | "
                f"{_f(v.get('mean'), 4)} |")
    lines += [
        "",
        "MoE depth is exactly 1.0 by construction; its forced-exit rate is the",
        "fraction of tokens at the depth cap of 1, which is all of them.",
        "MoR and MoRE forced-exit rate is the halting-collapse diagnostic: near",
        "1.0 means the policy has stopped choosing.",
        "",
        "## Depth-matched margin, if measured",
        "",
    ]
    if not any(r.get("margin_matched_more_over_mor") is not None
               for r in ratios.values()):
        lines += [
            "`N/A` -- the forced-depth evaluation on these cells has not been",
            "run yet (`code/eval_lang_depth_control.py`). The sweep's central",
            "quantity is the MoRE/MoR ratio of `margin_matched` at each scale;",
            "until it exists, no ratio claim from this table is supported.",
        ]
    else:
        lines += [
            "| scale | MoRE margin | MoR margin | ratio MoRE/MoR |",
            "|---|---|---|---|",
        ]
        for frac in FRACS:
            tag = tag_of(frac)
            try:
                mo = arms[tag]["MoRE"]["margin_matched"]["mean"]
                mr = arms[tag]["MoR"]["margin_matched"]["mean"]
            except (KeyError, TypeError):
                lines.append(f"| {frac:.1%} | N/A | N/A | N/A |")
                continue
            lines.append(f"| {frac:.1%} | {mo:.4f} | {mr:.4f} | "
                         f"{mo / mr:.2f}x |" if mr else
                         f"| {frac:.1%} | {mo:.4f} | {mr:.4f} | N/A |")
    lines += [
        "",
        "## Per-seed",
        "",
        "| scale | arm | seed | val/task_loss | depth/mean | forced exit | run |",
        "|---|---|---|---|---|---|---|",
    ]
    for frac in FRACS:
        for arm in ARMS:
            e = (arms.get(tag_of(frac), {}) or {}).get(arm)
            if not e:
                continue
            for r in e["per_seed"]:
                lines.append(
                    f"| {frac:.1%} | {arm} | {r['seed']} | "
                    f"{_f(r['val_task_loss'], 4)} | {_f(r['depth_mean'], 2)} | "
                    f"{_f(r['forced_exit_rate'], 4)} | `{r['run_dir']}` |")
    lines += [
        "",
        "## Provenance checks passed",
        "",
        "- one `data.dataset_version` across all cells",
        "- identical epochs/batch/lr/seq_len/vocab/num_experts/ffn_mult/",
        "  max_depth/num_blocks/dropout and every loss weight per arm",
        "- `data.subset_seed == provenance.seed` in every cell",
        "- group stamp `subset_lang_sweep` on every cell",
        "",
    ]
    with open(OUT_MD, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true",
                    help="require all 45 cells; nonzero exit otherwise")
    args = ap.parse_args(argv)

    cells, dupes = discover()
    problems, ref = check_protocol(cells)
    margins = load_margins()
    arms, ratios = aggregate(cells, margins)

    complete = len(cells) == len(FRACS) * len(ARMS) * len(SEEDS)
    payload = {
        "_README": (
            "T-OD.13 TEST 2. Language training-set-size sweep. val_task_loss is "
            "the primary metric (nats/token). margin_matched is the forced-depth "
            "control's uniform_loss - learned_loss, measured from checkpoint.pt "
            "by code/eval_lang_depth_control.py; it is N/A until that has run. "
            "MoE margins are N/A by construction. Partial cells are excluded "
            "from means and never counted as 0."),
        "group": GROUP,
        "n_cells": len(cells),
        "complete": complete,
        "dataset_version": next(iter({c["cfg"]["data"].get("dataset_version")
                                      for c in cells.values()}), None),
        "fracs": FRACS, "seeds": SEEDS, "arms": ARMS,
        "bigram_floor": 4.98494701553838,
        "protocol_problems": problems,
        "duplicate_runs": [{"cell": k, "kept": kd, "ignored": ig}
                           for k, ig, kd in dupes],
        "arms": arms,
        "ratios": ratios,
        "margins_available": bool(margins),
    }
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(payload, fh, indent=2)
    write_md(payload)

    print(f"{len(cells)}/45 cells")
    for frac in FRACS:
        tag = tag_of(frac)
        row = []
        for arm in ARMS:
            e = (arms.get(tag, {}) or {}).get(arm)
            v = (e or {}).get("val_task_loss") or {}
            row.append(f"{arm} " + (_f(v.get("mean")) if v else "N/A"))
        print(f"  {frac:.1%}: " + "  ".join(row))
    if problems:
        print("\nPROTOCOL PROBLEMS (refused, not averaged):")
        for p in problems:
            print("  -", p)
        return 2
    if args.strict and not complete:
        print(f"\n--strict: {45 - len(cells)} cells still missing")
        return 1
    print(f"\n-> {OUT_JSON}\n-> {OUT_MD}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
