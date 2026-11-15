# -*- coding: utf-8 -*-
"""eval_blimp.py -- BLiMP minimal-pair accuracy for the language checkpoints.

    python code/eval_blimp.py                         # all arms with a checkpoint
    python code/eval_blimp.py --max_per_paradigm 200  # fast subsample
    python code/eval_blimp.py --arch moe              # one arm

WHAT THIS IS
------------
BLiMP (Warstadt et al., TACL 2020) is 67 paradigms x 1000 minimal pairs of a
grammatical and an ungrammatical sentence. The model is CORRECT on a pair when it
assigns the grammatical sentence higher probability. No fine-tuning, no prompting,
no generation -- pure likelihood scoring, which is the BabyLM-standard way to
evaluate a sub-10M-parameter LM where MMLU/HellaSwag/ARC sit at chance. Chance here
is 50%.

HOW A SENTENCE IS SCORED
------------------------
`logP(sentence) = sum_t logP(token_{t+1} | tokens_{<=t})`, teacher-forced, with the
corpus boundary token `<|endoftext|>` (id 0) prepended so the first real token has
left context -- the same token that separates documents in the packed corpus. Raw
sum, not length-normalised: this is the BLiMP convention and the two sentences in a
pair are near-equal length, so a length term would mostly cancel. A per-token-mean
accuracy is reported alongside as a secondary cross-check.

THE TOKENIZER IS THE TRAINED ONE, AND THAT IS A CAVEAT FOR THE PAPER. BLiMP is
scored with `data/lang/wikitext-103/tokenizer.json` -- the 8192-BPE vocab the models
trained on. A BLiMP sentence therefore fragments into more sub-word tokens than it
would under a 32k-50k vocab, so ABSOLUTE accuracies are NOT comparable to published
models. The CROSS-ARM comparison (MoE vs MoRE vs MoR under one tokenizer) is valid
and is the only comparison this file makes.

WHY LENGTH-GROUPED BATCHING, NOT PADDING. The language model is validated for an
all-True `step_mask` (the corpus packs to exactly seq_len, so there is no pad path;
`lang_data.py` and `model.py:786` both rely on it). Padding a batch to a common
length would set `step_mask=False` on pad positions and route zeroed vectors into
attention as keys, which is the one path the language model was never trained for.
So sentences are grouped by EXACT token length and each group is batched with no
padding -- a real speed-up with zero change to the scored distribution.

EXPLORATORY. Not a declared confirmatory family (`code/confirmatory_tests.json`
prohibits adding one after results exist), so results are reported with the exact
randomization test but UNCORRECTED, and carry no significance verdict in an abstract.
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
EOT_ID = 0                        # <|endoftext|>, the corpus boundary token
NO_FAMILY = -1                    # placeholder oracle; routing is learned from x, not this
OUT = REPO / "results" / "language" / "blimp.json"


def load_tokenizer():
    from tokenizers import Tokenizer
    return Tokenizer.from_file(str(TOKENIZER_PATH))


def load_blimp(local_dir: Path | None):
    """List of {uid, phenomenon, good, bad}. Local JSONL dir first, else HF."""
    pairs = []
    if local_dir and local_dir.is_dir():
        for fp in sorted(local_dir.glob("*.jsonl")):
            for line in fp.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                pairs.append({"uid": r.get("UID", fp.stem),
                              "phenomenon": r.get("linguistics_term", "?"),
                              "good": r["sentence_good"], "bad": r["sentence_bad"]})
        if pairs:
            print(f"[blimp] loaded {len(pairs)} pairs from {local_dir}")
            return pairs

    from datasets import get_dataset_config_names, load_dataset
    name = "nyu-mll/blimp"
    try:
        configs = get_dataset_config_names(name)
    except Exception:
        name = "blimp"
        configs = get_dataset_config_names(name)
    print(f"[blimp] {len(configs)} paradigms from {name}; downloading/caching ...")
    for cfg in configs:
        ds = load_dataset(name, cfg, split="train")
        for r in ds:
            pairs.append({"uid": r.get("UID", cfg),
                          "phenomenon": r.get("linguistics_term", "?"),
                          "good": r["sentence_good"], "bad": r["sentence_bad"]})
    print(f"[blimp] loaded {len(pairs)} pairs across {len(configs)} paradigms")
    return pairs


def canonical_runs(explicit, arch_filter):
    out = []
    runs = REPO / "runs"
    for d in sorted(runs.iterdir()) if runs.is_dir() else []:
        if explicit and d.name not in explicit:
            continue
        rc_p, ck_p = d / "resolved_config.json", d / "checkpoint.pt"
        if not (rc_p.exists() and ck_p.exists()):
            continue
        try:
            rc = json.loads(rc_p.read_text(encoding="utf-8"))
        except ValueError:
            continue
        prov = rc.get("provenance", {}) or {}
        if prov.get("experiment_group") != CANONICAL_GROUP:
            continue
        if arch_filter and prov.get("architecture") != arch_filter:
            continue
        out.append(d)
    return out


def load_model(run_dir: Path, device):
    rc = json.loads((run_dir / "resolved_config.json").read_text(encoding="utf-8"))
    mc, dc = rc.get("model", {}), rc.get("data", {})
    task = rc.get("task") or (rc.get("provenance", {}) or {}).get("task")
    if task != TASK_LANGUAGE:
        raise RuntimeError(f"{run_dir.name}: not a language run")
    if "step_feat_dim" not in mc:
        raise KeyError(f"{run_dir.name}: no model.step_feat_dim; refusing to guess")
    model = build_model(mc, dc, task, int(mc["step_feat_dim"]), device)
    model.load_state_dict(torch.load(run_dir / "checkpoint.pt", map_location=device))
    model.eval()
    return model, (rc.get("provenance", {}) or {})


@torch.no_grad()
def score_sentences(model, tok_id_lists, device, batch_cap=192):
    """Sentence log-probs for a list of token-id lists, grouped by exact length so
    no padding is needed. Returns a list of floats aligned with the input order."""
    order = sorted(range(len(tok_id_lists)), key=lambda i: len(tok_id_lists[i]))
    out = [0.0] * len(tok_id_lists)
    i = 0
    while i < len(order):
        L = len(tok_id_lists[order[i]])
        group = [order[i]]
        i += 1
        while i < len(order) and len(tok_id_lists[order[i]]) == L and len(group) < batch_cap:
            group.append(order[i])
            i += 1
        if L < 2:
            # A single-token sentence has no next-token prediction; score 0 so the
            # pair falls back to a tie rather than crashing. Vanishingly rare.
            continue
        x = torch.tensor([tok_id_lists[j] for j in group], dtype=torch.long,
                         device=device)                              # (B, L)
        B = x.shape[0]
        sm = torch.ones(B, L, dtype=torch.bool, device=device)
        se = torch.full((B, L), NO_FAMILY, dtype=torch.long, device=device)
        so = torch.full((B, L), -1, dtype=torch.long, device=device)
        logits = model(x, sm, se, so)[0]                             # (B, L, V)
        logp = F.log_softmax(logits.float(), dim=-1)
        tgt = x[:, 1:]                                               # (B, L-1)
        tok_lp = logp[:, :-1, :].gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        sent_lp = tok_lp.sum(dim=1)                                  # (B,)
        for k, j in enumerate(group):
            out[j] = float(sent_lp[k])
    return out


def eval_checkpoint(run_dir, pairs, tok, device, batch_cap):
    model, prov = load_model(run_dir, device)
    # Tokenize every good and bad sentence once, with the EOT boundary prepended.
    def enc(s):
        return [EOT_ID] + tok.encode(s).ids
    good_ids = [enc(p["good"]) for p in pairs]
    bad_ids = [enc(p["bad"]) for p in pairs]
    lp_good = score_sentences(model, good_ids, device, batch_cap)
    lp_bad = score_sentences(model, bad_ids, device, batch_cap)

    by_uid, by_phen = defaultdict(lambda: [0, 0]), defaultdict(lambda: [0, 0])
    n_correct = 0
    for p, g, b in zip(pairs, lp_good, lp_bad):
        correct = g > b
        n_correct += int(correct)
        for d in (by_uid[p["uid"]], by_phen[p["phenomenon"]]):
            d[0] += int(correct)
            d[1] += 1
    return {
        "run": run_dir.name, "arch": prov.get("architecture"),
        "seed": int(prov.get("seed", -1)),
        "overall": n_correct / len(pairs),
        "by_paradigm": {k: v[0] / v[1] for k, v in sorted(by_uid.items())},
        "by_phenomenon": {k: v[0] / v[1] for k, v in sorted(by_phen.items())},
        "n_pairs": len(pairs),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--arch", default=None, choices=["moe", "mor", "more"])
    ap.add_argument("--run", action="append", default=None)
    ap.add_argument("--max_per_paradigm", type=int, default=0,
                    help="0 = all 1000; a smaller N subsamples each paradigm for speed")
    ap.add_argument("--data_dir", default=str(REPO / "data" / "blimp"),
                    help="local BLiMP JSONL dir; falls back to HuggingFace")
    ap.add_argument("--batch_cap", type=int, default=192)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runs = canonical_runs(args.run, args.arch)
    if not runs:
        print("no canonical cell with a checkpoint found. MoE and MoRE have "
              "checkpoints; MoR does not until it is retrained.")
        return 2

    pairs = load_blimp(Path(args.data_dir))
    if args.max_per_paradigm > 0:
        kept, seen = [], defaultdict(int)
        for p in pairs:
            if seen[p["uid"]] < args.max_per_paradigm:
                kept.append(p)
                seen[p["uid"]] += 1
        pairs = kept
        print(f"[blimp] subsampled to {len(pairs)} pairs "
              f"(<= {args.max_per_paradigm}/paradigm)")

    tok = load_tokenizer()
    print(f"device={device}  runs={len(runs)}  pairs={len(pairs)}")
    print("EXPLORATORY -- uncorrected, not a declared family.\n")

    cells = []
    for d in runs:
        c = eval_checkpoint(d, pairs, tok, device, args.batch_cap)
        cells.append(c)
        print(f"  {c['arch']:5s} s{c['seed']}  overall {c['overall']*100:5.2f}%")

    # Aggregate per arm across seeds.
    agg = {}
    arms = sorted({c["arch"] for c in cells})
    for a in arms:
        sel = [c for c in cells if c["arch"] == a]
        overalls = [c["overall"] for c in sel]
        phens = sorted({k for c in sel for k in c["by_phenomenon"]})
        agg[a] = {
            "n_seeds": len(sel),
            "overall_mean": st.mean(overalls),
            "overall_std": st.stdev(overalls) if len(overalls) > 1 else None,
            "by_phenomenon_mean": {
                ph: st.mean([c["by_phenomenon"][ph] for c in sel
                             if ph in c["by_phenomenon"]]) for ph in phens},
        }

    pairwise = {}
    for i, a in enumerate(arms):
        for b in arms[i + 1:]:
            xa = [c["overall"] for c in cells if c["arch"] == a]
            xb = [c["overall"] for c in cells if c["arch"] == b]
            r = _seed_stats.perm_test(xa, xb)
            if r is not None:
                pairwise[f"{a}_vs_{b}"] = {
                    "gap": r["gap"], "p_value": r["p_value"], "min_p": r["min_p"],
                    "status": "UNCORRECTED -- exploratory"}

    out = Path(args.out) if args.out else OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "benchmark": "BLiMP (Warstadt et al. 2020), minimal-pair likelihood scoring",
        "status": "EXPLORATORY. Scored with the wikitext-103 8192-BPE tokenizer, so "
                  "absolute accuracy is NOT comparable to published 32k-50k-vocab "
                  "models; the cross-arm comparison is the only valid one here. "
                  "Uncorrected, not a declared family.",
        "scoring": "sum of teacher-forced token log-probs, <|endoftext|> prepended; "
                   "chance = 0.50",
        "max_per_paradigm": args.max_per_paradigm or "all",
        "n_pairs": cells[0]["n_pairs"] if cells else 0,
        "by_arm": agg, "pairwise_uncorrected": pairwise,
        "cells": cells,
    }, indent=2) + "\n", encoding="utf-8")

    print("\nby arm:")
    for a in arms:
        s = agg[a]
        sd = f" +- {s['overall_std']*100:.2f}" if s["overall_std"] is not None else ""
        print(f"  {a:5s} n={s['n_seeds']}  {s['overall_mean']*100:5.2f}%{sd}")
    for k, v in pairwise.items():
        print(f"  {k}: gap {v['gap']*100:+.2f} pts, p={v['p_value']:.4f} "
              f"(floor {v['min_p']:.4f}, uncorrected)")
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
