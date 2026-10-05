# -*- coding: utf-8 -*-
"""eval_ood.py -- out-of-domain perplexity transfer for the language checkpoints.

    python code/eval_ood.py                      # ptb test, all arms with a checkpoint
    python code/eval_ood.py --corpus ptb --split test
    python code/eval_ood.py --arch moe

WHAT THIS IS
------------
Zero-shot perplexity on a corpus the models never trained on. Valid at ANY scale --
unlike MMLU/HellaSwag -- and standard in LM papers. The only cross-arm-valid number
is the comparison under ONE tokenizer, so this uses the SAME wikitext-103 8192-BPE
tokenizer the models trained on (`data/lang/wikitext-103/tokenizer.json`). Absolute
perplexities are therefore not comparable to published models on their own vocab;
the MoE-vs-MoRE-vs-MoR comparison is what this measures.

NOT wikitext-2: T-L2.0 established its validation split is byte-identical to
wikitext-103's, so it is not held out. PTB (`ptb_text_only`) is genuinely unseen.

HOW IT IS PACKED AND SCORED
---------------------------
The OOD corpus is tokenized with the wikitext-103 tokenizer, joined with the
`<|endoftext|>` boundary (id 0) between source lines exactly as the training corpus
was packed, and chunked into non-overlapping seq_len blocks with the trailing
partial block dropped (so `step_mask` is all-True and the model's validated path is
used -- the same reason the training packer dropped it, plan_language.md §3).
Per-token cross-entropy is `engine.py`'s formula with `reduction="mean"` over every
scored position; perplexity is its exp. No sliding window: a single clean forward
per block, which slightly overestimates perplexity near block starts equally for
all arms, so the comparison is unaffected.

EXPLORATORY. Not a declared confirmatory family, so reported with the exact
randomization test but UNCORRECTED and with no abstract-level verdict.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

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
TOKENIZER_PATH = REPO / "data" / "lang" / "wikitext-103" / "tokenizer.json"
EOT_ID = 0
NO_FAMILY = -1


def load_tokenizer():
    from tokenizers import Tokenizer
    return Tokenizer.from_file(str(TOKENIZER_PATH))


def fetch_ood_lines(corpus: str, split: str):
    """Raw text lines of the OOD corpus, from a PARQUET-format HF dataset.

    `datasets` >= 3 refuses script-based datasets (the old `ptb_text_only` loader is
    dead), so each corpus must resolve to parquet/arrow. The models trained on
    wikitext-103 (curated Wikipedia), so a genuine OOD set must be a DIFFERENT
    domain. `pile` is a 10k-document sample of The Pile (web/code/books/news) and is
    strongly OOD; `text8` is weaker (also Wikipedia-derived) but always loads.
    """
    from datasets import load_dataset
    if corpus == "pile":
        ds = load_dataset("NeelNanda/pile-10k", split="train")
        return [r["text"] for r in ds if r.get("text", "").strip()][:4000]
    if corpus == "text8":
        ds = load_dataset("afmck/text8", split="train")
        return [ds[0]["text"][:5_000_000]]            # text8 is one long string
    raise ValueError(f"unknown OOD corpus {corpus!r}; add a fetch rule")


def pack(tok, lines, seq_len):
    """Tokenize, join with the EOT boundary, chunk into full seq_len blocks."""
    ids = []
    for ln in lines:
        ids.append(EOT_ID)
        ids.extend(tok.encode(ln).ids)
    n_blocks = len(ids) // seq_len
    arr = np.asarray(ids[: n_blocks * seq_len], dtype=np.int64).reshape(n_blocks, seq_len)
    return arr


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


@torch.no_grad()
def perplexity(run_dir, blocks, device, batch=32):
    rc = json.loads((run_dir / "resolved_config.json").read_text(encoding="utf-8"))
    mc, dc = rc.get("model", {}), rc.get("data", {})
    task = rc.get("task") or (rc.get("provenance", {}) or {}).get("task")
    if "step_feat_dim" not in mc:
        raise KeyError(f"{run_dir.name}: no model.step_feat_dim")
    model = build_model(mc, dc, task, int(mc["step_feat_dim"]), device)
    model.load_state_dict(torch.load(run_dir / "checkpoint.pt", map_location=device))
    model.eval()

    L = blocks.shape[1]
    total_loss, total_tok = 0.0, 0
    for i in range(0, blocks.shape[0], batch):
        x = torch.from_numpy(blocks[i:i + batch]).to(device)
        B = x.shape[0]
        sm = torch.ones(B, L, dtype=torch.bool, device=device)
        se = torch.full((B, L), NO_FAMILY, dtype=torch.long, device=device)
        so = torch.full((B, L), -1, dtype=torch.long, device=device)
        logits = model(x, sm, se, so)[0]
        loss = F.cross_entropy(logits[:, :-1, :].reshape(-1, logits.shape[-1]),
                               x[:, 1:].reshape(-1), reduction="sum")
        total_loss += float(loss)
        total_tok += B * (L - 1)
    mean_loss = total_loss / total_tok
    prov = rc.get("provenance", {}) or {}
    return {"run": run_dir.name, "arch": prov.get("architecture"),
            "seed": int(prov.get("seed", -1)),
            "mean_loss": mean_loss, "perplexity": float(np.exp(mean_loss)),
            "n_tokens": total_tok}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corpus", default="pile", choices=["pile", "text8"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--arch", default=None, choices=["moe", "mor", "more"])
    ap.add_argument("--run", action="append", default=None)
    ap.add_argument("--seq_len", type=int, default=256)
    ap.add_argument("--max_blocks", type=int, default=2000,
                    help="cap scored blocks (0 = all); a few thousand give stable ppl")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runs = canonical_runs(args.run, args.arch)
    if not runs:
        print("no canonical cell with a checkpoint found (MoE and MoRE have them; "
              "MoR not until retrained).")
        return 2

    tok = load_tokenizer()
    lines = fetch_ood_lines(args.corpus, args.split)
    blocks = pack(tok, lines, args.seq_len)
    if args.max_blocks and blocks.shape[0] > args.max_blocks:
        # A few thousand blocks give a stable perplexity; the Pile's long documents
        # otherwise pack into tens of thousands and the scoring never finishes.
        blocks = blocks[: args.max_blocks]
    print(f"device={device}  OOD={args.corpus}:{args.split}  "
          f"{blocks.shape[0]} blocks x {args.seq_len}  runs={len(runs)}", flush=True)
    print("EXPLORATORY -- uncorrected, not a declared family.\n")

    cells = []
    for d in runs:
        c = perplexity(d, blocks, device)
        cells.append(c)
        print(f"  {c['arch']:5s} s{c['seed']}  loss {c['mean_loss']:.4f}  "
              f"ppl {c['perplexity']:.2f}")

    agg, arms = {}, sorted({c["arch"] for c in cells})
    for a in arms:
        sel = [c for c in cells if c["arch"] == a]
        losses = [c["mean_loss"] for c in sel]
        agg[a] = {"n_seeds": len(sel), "mean_loss": st.mean(losses),
                  "std_loss": st.stdev(losses) if len(losses) > 1 else None,
                  "mean_perplexity": float(np.exp(st.mean(losses)))}

    pairwise = {}
    for i, a in enumerate(arms):
        for b in arms[i + 1:]:
            r = _seed_stats.perm_test([c["mean_loss"] for c in cells if c["arch"] == a],
                                      [c["mean_loss"] for c in cells if c["arch"] == b])
            if r is not None:
                pairwise[f"{a}_vs_{b}"] = {"gap": r["gap"], "p_value": r["p_value"],
                                          "min_p": r["min_p"],
                                          "status": "UNCORRECTED -- exploratory"}

    out = Path(args.out) if args.out else (REPO / "results" / "language" /
                                           f"ood_{args.corpus}_{args.split}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "benchmark": f"out-of-domain perplexity ({args.corpus}:{args.split})",
        "status": "EXPLORATORY. Scored with the wikitext-103 8192-BPE tokenizer, so "
                  "absolute perplexity is not comparable to published models; the "
                  "cross-arm comparison is the valid one. Uncorrected.",
        "tokenizer": "wikitext-103 bpe8192", "seq_len": args.seq_len,
        "n_blocks": int(blocks.shape[0]),
        "by_arm": agg, "pairwise_uncorrected": pairwise, "cells": cells,
    }, indent=2) + "\n", encoding="utf-8")

    print(f"\nby arm ({args.corpus}:{args.split}):")
    for a in arms:
        s = agg[a]
        sd = f" +- {s['std_loss']:.4f}" if s["std_loss"] is not None else ""
        print(f"  {a:5s} n={s['n_seeds']}  loss {s['mean_loss']:.4f}{sd}  "
              f"ppl {s['mean_perplexity']:.2f}")
    for k, v in pairwise.items():
        print(f"  {k}: gap {v['gap']:+.4f} nats, p={v['p_value']:.4f} "
              f"(floor {v['min_p']:.4f}, uncorrected)")
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
