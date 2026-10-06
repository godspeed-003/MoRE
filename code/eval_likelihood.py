# -*- coding: utf-8 -*-
"""eval_likelihood.py -- likelihood-scored capability evals: EWoK and MSGS.

    python code/eval_likelihood.py --data_dir data/ewok --name ewok
    python code/eval_likelihood.py --data_dir data/msgs --name msgs

WHAT THIS IS
------------
A generic zero-shot likelihood scorer for the BabyLM-family evals that, like BLiMP,
need NO fine-tuning: EWoK (world knowledge, plausible-vs-implausible) and MSGS
(Mixed Signals Generalization Set, linguistic generalisation). It reads a LOCAL
JSONL directory and scores `logP(option | context)`, correct when the labelled-good
option scores higher. Reuses the exact scoring in `eval_blimp.score_sentences`.

WHY LOCAL JSONL AND NOT A HUB LOADER
------------------------------------
EWoK's canonical HuggingFace dataset (`ewok-core/ewok-core-1.0`) is GATED -- a token
cannot unlock it; the terms must be accepted on the dataset page. It is also
distributed as raw JSON in the GitHub repo `ewok-core/ewok-core`, which avoids the
gate. MSGS is distributed through the BabyLM 2024 pipeline. Both are therefore
expected as local files; this script converts either into one JSONL per split with a
known shape (below) and scores it. Convert once with `to_jsonl()` or by hand.

EXPECTED JSONL SHAPE (one object per line), auto-detected by key name:
  minimal-pair : {"sentence_good": ..., "sentence_bad": ..., <meta...>}
                 (also accepts good/bad, continuation_good/continuation_bad)
  continuation : {"context": ..., "targets": [{"text":..., "label": true}, ...]}
                 (EWoK style; the labelled-true target is the good one)
Any other string field is carried as METADATA for per-slice aggregation: this script
groups by `domain`, `axis`, `phenomenon`, or `linguistics_term` if present.

EXPECTATION-SETTING, AND WHY IT IS STILL WORTH RUNNING
------------------------------------------------------
A 5.6M model trained only on wikitext-103 has little WORLD KNOWLEDGE (EWoK) and may
be near chance there; MSGS probes generalisation of linguistic structure, which is
more plausible to have partly learned. Near-chance is the HONEST prediction and is
reportable as a limitation ("at this scale the model has not acquired world
knowledge"), not as a finding. Report it plainly; do not dress a chance score up.

CAVEAT: scored with the wikitext-103 8192-BPE tokenizer (the trained one), so
absolute accuracy is not comparable to published models; the cross-arm comparison is
the valid one. EXPLORATORY -- uncorrected, not a declared family.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import torch

CODE = Path(os.path.dirname(os.path.abspath(__file__)))
REPO = CODE.parent
sys.path.insert(0, str(CODE))

from eval_blimp import (canonical_runs, load_tokenizer, score_sentences,  # noqa: E402
                        EOT_ID)
from eval_test_split import build_model                       # noqa: E402
from more.config import TASK_LANGUAGE                         # noqa: E402
import seed_stats as _seed_stats                              # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

META_KEYS = ("domain", "axis", "phenomenon", "linguistics_term", "category", "type")


def _find(d: dict, names):
    for n in names:
        if n in d and isinstance(d[n], str):
            return d[n]
    return None


def expand_ewok(row: dict):
    """EWoK-core schema -> two (context, good, bad) items per row.

    A row is {Context1, Context2, Target1, Target2, Domain, ContextType, ...} where
    Context1/Context2 are minimally different setups and Target1/Target2 the two
    competing completions. The congruence task: Target1 fits Context1 and Target2
    fits Context2, so we emit both crossings and the model is correct when it
    prefers the congruent target given each context. Aggregated by `Domain`."""
    c1, c2 = row.get("Context1"), row.get("Context2")
    t1, t2 = row.get("Target1"), row.get("Target2")
    if not all(isinstance(x, str) and x for x in (c1, c2, t1, t2)):
        return None
    meta = {"domain": row.get("Domain"), "context_type": row.get("ContextType"),
            "target_diff": row.get("TargetDiff")}
    return [({"context": c1, "sentence_good": t1, "sentence_bad": t2, **meta}),
            ({"context": c2, "sentence_good": t2, "sentence_bad": t1, **meta})]


def parse_item(d: dict):
    """Return (context, good_text, bad_text, meta) or None if unrecognised.

    Context is "" for a pure minimal pair, and the shared premise for a
    continuation item. The score compared is logP(option | context)."""
    good = _find(d, ("sentence_good", "good", "continuation_good", "target_true",
                     "correct"))
    bad = _find(d, ("sentence_bad", "bad", "continuation_bad", "target_false",
                    "incorrect"))
    ctx = _find(d, ("context", "premise", "setup", "sent_prefix")) or ""
    if good is not None and bad is not None:
        return ctx, good, bad, d
    tgts = d.get("targets") or d.get("options") or d.get("choices")
    if isinstance(tgts, list) and len(tgts) >= 2:
        marked = [t for t in tgts if isinstance(t, dict)
                  and (t.get("label") is True or t.get("correct") is True)]
        rest = [t for t in tgts if t not in marked]
        if marked and rest:
            g = marked[0].get("text") or marked[0].get("sentence")
            b = rest[0].get("text") or rest[0].get("sentence")
            if g and b:
                return ctx, g, b, d
    return None


def load_local(data_dir: Path):
    items = []
    files = sorted(data_dir.glob("*.jsonl")) + sorted(data_dir.glob("*.json"))
    for fp in files:
        text = fp.read_text(encoding="utf-8")
        rows = ([json.loads(x) for x in text.splitlines() if x.strip()]
                if fp.suffix == ".jsonl" else json.loads(text))
        if isinstance(rows, dict):
            rows = rows.get("data") or rows.get("examples") or list(rows.values())
        for r in rows:
            if not isinstance(r, dict):
                continue
            if "Context1" in r and "Target1" in r:      # EWoK-core schema
                ex = expand_ewok(r)
                if ex:
                    items.extend(ex)
                continue
            items.append(r)
    return items


def meta_of(d: dict) -> str:
    for k in META_KEYS:
        v = d.get(k)
        if isinstance(v, str):
            return f"{k}:{v}"
    return "all"


@torch.no_grad()
def score_option(model, tok, ctx, opt, device):
    """logP(opt | ctx): prepend EOT + ctx, then sum only the OPTION tokens' log-probs."""
    ctx_ids = [EOT_ID] + tok.encode(ctx).ids if ctx else [EOT_ID]
    opt_ids = tok.encode(opt).ids
    if not opt_ids:
        return float("-inf")
    full = ctx_ids + opt_ids
    sm = torch.ones(1, len(full), dtype=torch.bool, device=device)
    se = torch.full((1, len(full)), -1, dtype=torch.long, device=device)
    so = torch.full((1, len(full)), -1, dtype=torch.long, device=device)
    logits = model(torch.tensor([full], device=device), sm, se, so)[0]
    logp = torch.log_softmax(logits.float(), dim=-1)[0]
    total = 0.0
    for i in range(len(ctx_ids) - 1, len(full) - 1):     # predict each option token
        total += float(logp[i, full[i + 1]])
    return total


def eval_dir(run_dir, raw_items, tok, device):
    rc = json.loads((run_dir / "resolved_config.json").read_text(encoding="utf-8"))
    mc, dc = rc.get("model", {}), rc.get("data", {})
    task = rc.get("task") or (rc.get("provenance", {}) or {}).get("task")
    prov = rc.get("provenance", {}) or {}
    if task != TASK_LANGUAGE:
        raise RuntimeError(f"{run_dir.name}: not a language run")
    model = build_model(mc, dc, task, int(mc["step_feat_dim"]), device)
    model.load_state_dict(torch.load(run_dir / "checkpoint.pt", map_location=device))
    model.eval()

    parsed, skipped = [], 0
    for d in raw_items:
        got = parse_item(d)
        if got is None:
            skipped += 1
            continue
        parsed.append((got, meta_of(d)))
    if not parsed:
        return None, skipped

    per_meta = defaultdict(lambda: [0, 0])
    n_ok = 0
    for (ctx, good, bad, _), meta in parsed:
        lg = score_option(model, tok, ctx, good, device)
        lb = score_option(model, tok, ctx, bad, device)
        ok = lg > lb
        n_ok += int(ok)
        per_meta[meta][0] += int(ok)
        per_meta[meta][1] += 1
    return ({"run": run_dir.name, "arch": prov.get("architecture"),
             "seed": int(prov.get("seed", -1)), "n_items": len(parsed),
             "overall": n_ok / len(parsed),
             "by_slice": {k: v[0] / v[1] for k, v in sorted(per_meta.items())}},
            skipped)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data_dir", required=True,
                    help="local JSONL/JSON dir of EWoK or MSGS items")
    ap.add_argument("--name", default="ewok")
    ap.add_argument("--arch", default=None, choices=["moe", "mor", "more"])
    ap.add_argument("--run", action="append", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--max_items", type=int, default=0,
                    help="cap items (0 = all); even-stride subsample for speed")
    args = ap.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        print(f"data dir not found: {data_dir}\n"
              f"Place {args.name} JSON there first. EWoK: accept the terms at\n"
              f"  https://huggingface.co/datasets/ewok-core/ewok-core-1.0\n"
              f"or use the raw JSON in the GitHub repo ewok-core/ewok-core.\n"
              f"MSGS: from the BabyLM 2024 pipeline. See the docstring.")
        return 2

    runs = canonical_runs(args.run, args.arch)
    if not runs:
        print("no canonical cell with a checkpoint found.")
        return 2
    tok = load_tokenizer()
    all_items = load_local(data_dir)
    if args.max_items and len(all_items) > args.max_items:
        # Deterministic even-stride subsample: EWoK is uniformly at chance here, so a
        # few thousand items give a tight estimate and the full 8748 x 13 cells runs
        # past the 30-min background cap. Stride keeps all domains represented.
        step = len(all_items) / args.max_items
        all_items = [all_items[int(i * step)] for i in range(args.max_items)]
    print(f"device={device}  eval={args.name}  dir={data_dir}  runs={len(runs)}  "
          f"items={len(all_items)}")
    print("EXPLORATORY -- uncorrected; near-chance is the honest expectation.\n")

    cells, skipped = [], 0
    for d in runs:
        c, sk = eval_dir(d, all_items, tok, device)
        skipped += sk
        if c is None:
            print(f"  {d.name}: no recognised items (skipped {sk})")
            continue
        cells.append(c)
        print(f"  {c['arch']:5s} s{c['seed']}  overall {c['overall']*100:5.2f}%  "
              f"(n={c['n_items']})")
    if skipped:
        print(f"  [warn] {skipped} items had an unrecognised shape and were skipped")

    agg, arms = {}, sorted({c["arch"] for c in cells})
    for a in arms:
        sel = [c for c in cells if c["arch"] == a]
        o = [c["overall"] for c in sel]
        agg[a] = {"n_seeds": len(sel), "overall_mean": st.mean(o),
                  "overall_std": st.stdev(o) if len(o) > 1 else None}
    pairwise = {}
    for i, a in enumerate(arms):
        for b in arms[i + 1:]:
            r = _seed_stats.perm_test([c["overall"] for c in cells if c["arch"] == a],
                                      [c["overall"] for c in cells if c["arch"] == b])
            if r is not None:
                pairwise[f"{a}_vs_{b}"] = {"gap": r["gap"], "p_value": r["p_value"],
                                          "min_p": r["min_p"],
                                          "status": "UNCORRECTED -- exploratory"}

    out = Path(args.out) if args.out else (REPO / "results" / "language" /
                                           f"{args.name}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "benchmark": f"{args.name} (zero-shot likelihood scoring)",
        "status": "EXPLORATORY. Scored with the wikitext-103 8192-BPE tokenizer so "
                  "absolute accuracy is not comparable to published models; cross-arm "
                  "only. Near-chance expected at this scale -- a limitation, not a "
                  "finding. Uncorrected.",
        "data_dir": str(data_dir), "chance": 0.50,
        "by_arm": agg, "pairwise_uncorrected": pairwise, "cells": cells,
    }, indent=2) + "\n", encoding="utf-8")
    print("\nby arm:")
    for a in arms:
        s = agg[a]
        sd = f" +- {s['overall_std']*100:.2f}" if s["overall_std"] is not None else ""
        print(f"  {a:5s} n={s['n_seeds']}  {s['overall_mean']*100:5.2f}%{sd}  (chance 50%)")
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
