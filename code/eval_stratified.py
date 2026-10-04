# -*- coding: utf-8 -*-
"""eval_stratified.py -- per-token loss from a checkpoint, grouped three ways.

    python code/eval_stratified.py                        # val split, all arms
    python code/eval_stratified.py --split test           # the held-out test split
    python code/eval_stratified.py --fixed-depth 3        # budgeted-inference point
    python code/eval_stratified.py --run runs/langB_MoRE_seed42__9cfd3f1c

WHY ONE SCRIPT AND NOT THREE
----------------------------
Difficulty-stratified loss, the held-out test evaluation and the budgeted-inference
curve are the SAME operation with a different input split, a different grouping, or a
forced depth. The operation is: load a checkpoint, run one forward pass, emit
**per-token** loss rather than the mean, and group it. Writing three scripts would mean
three copies of the token-alignment logic, and that alignment is the part that is easy
to get silently wrong (see THE ALIGNMENT TRAP below).

Out-of-domain perplexity is the same operation again, on a different corpus, and is NOT
implemented here because it needs a corpus packed with THIS tokenizer first -- see the
note at the bottom of this docstring.

WHAT THIS ANSWERS, AND WHY IT IS THE RIGHT QUESTION FOR THIS PAPER
------------------------------------------------------------------
MoRE's claim is that it spends more compute where a token is harder. The canonical arm's
mean loss is WORSE than MoR's (3.6370 vs 3.5202), but a mean can hide a mechanism: a
model may be worse on average and better exactly where its mechanism is supposed to
help. So the question is not "is the mean lower" -- that is already answered and the
answer is no -- but "does the deficit concentrate or invert on the hardest tokens".

Three outcomes, all publishable, and the table below decides which:
  * the deficit is flat across deciles          -> the mechanism does nothing
  * it shrinks on rare tokens                   -> the mechanism works, outweighed
  * it inverts on the hardest decile            -> the mechanism works and the mean hid it

READ THIS BEFORE BELIEVING A DEPTH-STRATIFIED RESULT. The canonical MoRE arm forced-exits
97.3% of tokens at step 7 and `depth/std` is 0.43. Depth is very nearly a CONSTANT, so a
per-decile depth column will be flat almost by construction, and "depth did not vary by
difficulty" is a statement about the halting collapse (T-LX.19/T-LX.20), not about the
routing architecture. Do not report it as the latter.

THE ALIGNMENT TRAP -- the reason this file exists rather than an inline loop
---------------------------------------------------------------------------
`engine.py:993` computes the language loss as

    F.cross_entropy(reg[:, :-1, :].reshape(-1, V), x[:, 1:].reshape(-1))

so the prediction at position t is scored against the token at t+1, and **each block's
last position is dropped** because it has no next-token target. Three consequences, and
all three have to hold or the numbers are not comparable to anything else in the paper:

1. The population is "every position except each block's last". That is exactly the
   population `depth/hist` and `depth/spearman_*` use, documented in each run's
   `depth_key_provenance`. Deviating gives a table that cannot be read beside any depth
   key.
2. **Stratify by the TARGET token id, not the input token id.** Difficulty is a property
   of what the model must PREDICT. `token_decile.npy` and `token_family.npy` are
   per-TYPE arrays indexed by token id, so they are gathered with `x[:, 1:]`, never with
   `x[:, :-1]`. Getting this backwards produces a plausible-looking table that answers a
   different question, and nothing in the output would look wrong.
3. Reduction must be `none`. A mean per batch cannot be regrouped afterwards.

WHAT IS NOT HERE, AND WHY
-------------------------
* **Out-of-domain perplexity.** It needs a second corpus packed with
  `data/lang/wikitext-103/tokenizer.json` -- the SAME tokenizer, or the perplexities are
  not comparable to each other or to anything published. Point
  `data/lang/build_language_dataset.py` at PTB or text8 with the existing tokenizer
  passed in, then run this script with `--corpus <that corpus>`. Do NOT train a new BPE,
  and do NOT use wikitext-2: T-L2.0 established its validation split is byte-identical
  to wikitext-103's, so it is not held out.
* **Length extrapolation** (train 256, score 512/1024). The corpus is pre-packed at
  seq_len 256 and `MoRELanguageDataset` reads that packing, so this needs a re-pack at
  the longer length rather than a flag. Worth doing -- recursive models often hold
  perplexity better than fixed-depth ones on extended context, which would be a genuine
  MoRE/MoR advantage -- but it is a data-layer change, not an eval-layer one.
* **BLiMP.** Separate harness: it scores minimal PAIRS, comparing summed log-prob between
  two sentences, which is a different operation from grouping one corpus's per-token loss.

EVERYTHING THIS WRITES IS EXPLORATORY. `code/confirmatory_tests.json` declares two
families and prohibits adding a third after results exist, so these numbers are reported
with effect sizes, UNCORRECTED and labelled, and they carry no significance verdict in an
abstract or conclusion. They do not touch `results/language/results_tables.md`.
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
from torch.utils.data import DataLoader

CODE = Path(os.path.dirname(os.path.abspath(__file__)))
REPO = CODE.parent
sys.path.insert(0, str(CODE))

from eval_test_split import build_model                        # noqa: E402
from more.config import TASK_LANGUAGE                          # noqa: E402
from more.metrics import compute_token_exit_depths             # noqa: E402
import seed_stats as _seed_stats                               # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

CANONICAL_GROUP = "canonical_lang_b"
N_DECILES = 10
# Position buckets within a 256-token block. Recursion should matter most where context
# is shortest, so the first bucket is deliberately narrow.
POS_EDGES = [0, 8, 32, 64, 128, 256]


def arm_of(rc: dict) -> str:
    return (rc.get("provenance", {}) or {}).get("architecture", "?")


def seed_of(rc: dict) -> int:
    return int((rc.get("provenance", {}) or {}).get("seed", -1))


def canonical_runs(explicit: list[str] | None) -> list[Path]:
    """Canonical cells that actually have a checkpoint, newest-first per (arm, seed)."""
    if explicit:
        return [Path(p) for p in explicit]
    out = []
    runs = REPO / "runs"
    for d in sorted(runs.iterdir()) if runs.is_dir() else []:
        rc_p, ck_p = d / "resolved_config.json", d / "checkpoint.pt"
        if not (rc_p.exists() and ck_p.exists()):
            continue
        try:
            rc = json.loads(rc_p.read_text(encoding="utf-8"))
        except ValueError:
            continue
        if (rc.get("provenance", {}) or {}).get("experiment_group") != CANONICAL_GROUP:
            continue
        out.append(d)
    return out


def load_type_arrays(corpus: str) -> dict:
    """Per-TYPE arrays indexed by token id. Absent ones become None rather than zeros --
    a zero would stratify every token into decile 0 and look like a result."""
    base = REPO / "data" / "lang" / corpus
    got = {}
    for name, fn in (("decile", "token_decile.npy"),
                     ("family", "token_family.npy"),
                     ("train_count", "token_train_count.npy")):
        p = base / fn
        got[name] = np.load(p) if p.exists() else None
        if got[name] is None:
            print(f"  [warn] {fn} absent under {base.name}; that grouping -> N/A")
    return got


def per_token_losses(run_dir: Path, split: str, device: torch.device,
                     fixed_depth: int | None, corpus_override: str | None):
    """One forward pass. Returns (loss, target_id, exit_depth, position) flat arrays.

    `loss` is per TARGET token under engine.py's exact formulation, so summing it and
    dividing by its length reproduces that run's `val/task_loss` on the val split. That
    identity is the correctness check -- see `verify_matches_metrics`.
    """
    rc = json.loads((run_dir / "resolved_config.json").read_text(encoding="utf-8"))
    mc, dc = dict(rc.get("model", {})), dict(rc.get("data", {}))
    task = rc.get("task") or (rc.get("provenance", {}) or {}).get("task")
    if task != TASK_LANGUAGE:
        raise RuntimeError(f"{run_dir.name}: task={task!r}, this script is language-only")

    if "step_feat_dim" not in mc:
        # Same refusal as eval_test_split: a guessed width either fails the load or,
        # worse, loads a model that never trained.
        raise KeyError(f"{run_dir.name}: no model.step_feat_dim; refusing to guess")
    step_feat_dim = int(mc["step_feat_dim"])

    # A forced depth is an INFERENCE-TIME override and must not be written back into the
    # run's config. Copy, mutate the copy, and record it in the output.
    if fixed_depth is not None:
        if not (1 <= fixed_depth <= int(mc["max_depth"])):
            raise ValueError(f"--fixed-depth {fixed_depth} outside 1..{mc['max_depth']}")
        mc["fixed_depth"] = True
        mc["max_depth"] = int(fixed_depth)

    corpus = corpus_override or dc.get("corpus") or dc.get("dataset_version")
    from more.lang_data import MoRELanguageDataset
    ds = MoRELanguageDataset(corpus, split)

    model = build_model(mc, dc, task, step_feat_dim, device)
    model.load_state_dict(torch.load(run_dir / "checkpoint.pt", map_location=device))
    model.eval()

    md = int(mc["max_depth"])
    bs = int(rc.get("training", {}).get("batch_size", 48))
    loader = DataLoader(ds, batch_size=bs, shuffle=False)

    L, T, D, P = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            x, sm, se, so = [b.to(device) if torch.is_tensor(b) else b
                             for b in batch[:4]]
            out = model(x, sm, se, so)
            reg, depth_exits = out[0], out[5]

            # engine.py:993 verbatim, except reduction="none" so it can be regrouped.
            V = reg.shape[-1]
            tgt = x[:, 1:]                                   # THE TARGET, see trap 2
            tok_loss = F.cross_entropy(reg[:, :-1, :].reshape(-1, V),
                                       tgt.reshape(-1), reduction="none")
            n_pos = tgt.shape[1]
            L.append(tok_loss.float().cpu().numpy())
            T.append(tgt.reshape(-1).cpu().numpy())
            P.append(np.tile(np.arange(n_pos, dtype=np.int32), tgt.shape[0]))

            if depth_exits is not None:
                # compute_token_exit_depths returns a FLAT (B*T,) tensor, not (B, T) --
                # eval_test_split.py:1072 consumes it against an equally flat sm mask.
                # Reshape before slicing, or the drop-last-position below silently
                # removes the wrong elements.
                ed = compute_token_exit_depths(depth_exits, md).reshape(x.shape[0], -1)
                # Drop each block's last position so depth aligns with the loss
                # population exactly. Misaligning by one would correlate a token's
                # depth with the NEXT token's loss.
                D.append(ed[:, :n_pos].reshape(-1).float().cpu().numpy())
            else:
                D.append(np.full(tok_loss.shape[0], np.nan, dtype=np.float32))

    return (np.concatenate(L), np.concatenate(T),
            np.concatenate(D), np.concatenate(P))


def verify_matches_metrics(run_dir: Path, mean_loss: float, split: str) -> str:
    """The correctness check that makes this script trustworthy: on the val split the
    mean of the per-token losses must reproduce the run's published val/task_loss. If it
    does not, the alignment is wrong and every stratified number below is wrong too."""
    if split != "val":
        return "n/a (only the val split has a published number to check against)"
    mp = run_dir / "metrics.json"
    if not mp.exists():
        return "no metrics.json"
    m = json.loads(mp.read_text(encoding="utf-8"))
    flat = {}

    def w(o, p=""):
        if isinstance(o, dict):
            for k, v in o.items():
                w(v, f"{p}/{k}" if p else k)
        else:
            flat[p] = o

    w(m)
    pub = flat.get("val/task_loss")
    if not isinstance(pub, (int, float)):
        return "val/task_loss absent"
    d = abs(mean_loss - float(pub))
    return (f"{'OK' if d < 2e-3 else 'MISMATCH'} recomputed {mean_loss:.6f} vs "
            f"published {float(pub):.6f} (delta {d:.2e})")


def group_mean(loss: np.ndarray, key: np.ndarray, n_groups: int) -> list:
    """Mean loss per group. An empty group is None -> N/A, never 0.0 (CLAUDE.md 4)."""
    out = []
    for g in range(n_groups):
        sel = key == g
        out.append(float(loss[sel].mean()) if bool(sel.any()) else None)
    return out


def bucket_positions(pos: np.ndarray) -> np.ndarray:
    b = np.zeros_like(pos)
    for i, lo in enumerate(POS_EDGES[:-1]):
        b[(pos >= lo) & (pos < POS_EDGES[i + 1])] = i
    return b


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--run", action="append", default=None,
                    help="explicit run dir; repeatable. Default: every canonical cell "
                         "that has a checkpoint.")
    ap.add_argument("--fixed-depth", type=int, default=None,
                    help="force this inference depth (budgeted-inference point). The "
                         "recursive arms accept any 1..max_depth; MoE has max_depth 1 "
                         "and is a single point by construction.")
    ap.add_argument("--corpus", default=None,
                    help="override the corpus, for out-of-domain scoring. MUST have "
                         "been packed with the same tokenizer.")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runs = canonical_runs(args.run)
    if not runs:
        print("no canonical cell with a checkpoint was found under runs/.\n"
              "Only the five route-once MoRE cells carry one; the MoE and MoR language\n"
              "weights were lost with the first rented V100. Retrain those two arms\n"
              "(RUNBOOK_ACL_4060.md) before a three-arm stratified table is possible.")
        return 2

    print(f"device={device}  split={args.split}  runs={len(runs)}"
          + (f"  fixed_depth={args.fixed_depth}" if args.fixed_depth else "")
          + (f"  corpus={args.corpus}" if args.corpus else ""))
    print("EXPLORATORY -- uncorrected, and not part of either declared family.\n")

    cells, types_cache = [], {}
    for d in runs:
        rc = json.loads((d / "resolved_config.json").read_text(encoding="utf-8"))
        arm, seed = arm_of(rc), seed_of(rc)
        corpus = args.corpus or rc.get("data", {}).get("corpus") \
            or rc.get("data", {}).get("dataset_version")
        if corpus not in types_cache:
            types_cache[corpus] = load_type_arrays(corpus)
        ta = types_cache[corpus]

        loss, tgt, depth, pos = (None,) * 4
        try:
            loss, tgt, depth, pos = per_token_losses(
                d, args.split, device, args.fixed_depth, args.corpus)
        except ValueError as e:
            # A forced depth beyond this arm's max_depth: MoE has max_depth 1, so the
            # budgeted-depth curve does not apply to it and the cell is SKIPPED, not a
            # crash. This keeps an overnight `--fixed-depth` sweep alive once the MoE
            # checkpoints are present -- MoE is a single compute point by construction.
            if "fixed-depth" in str(e):
                print(f"  skip  {arm_of(rc):5s} s{seed_of(rc)}  "
                      f"({str(e).split('--')[1].strip() if '--' in str(e) else e})")
                continue
            raise
        mean_loss = float(loss.mean())
        check = verify_matches_metrics(d, mean_loss, args.split)

        row = {"run": d.name, "arch": arm, "seed": seed, "corpus": corpus,
               "split": args.split, "fixed_depth": args.fixed_depth,
               "n_tokens": int(loss.size), "mean_loss": mean_loss,
               "perplexity": float(np.exp(mean_loss)),
               "reproduces_published_val_task_loss": check,
               "mean_exit_depth": (None if np.all(np.isnan(depth))
                                   else float(np.nanmean(depth)))}

        if ta["decile"] is not None:
            dec = ta["decile"][tgt]                    # TARGET id, see trap 2
            row["loss_by_decile"] = group_mean(loss, dec, N_DECILES)
            row["depth_by_decile"] = (None if np.all(np.isnan(depth)) else
                                      group_mean(depth, dec, N_DECILES))
            row["n_by_decile"] = [int((dec == g).sum()) for g in range(N_DECILES)]
        else:
            row["loss_by_decile"] = row["depth_by_decile"] = None

        if ta["family"] is not None:
            fam = ta["family"][tgt]
            nf = int(fam.max()) + 1 if fam.size else 0
            # -1 marks an unmapped token; exclude rather than bucket it into family 0.
            keep = fam >= 0
            row["loss_by_family"] = group_mean(loss[keep], fam[keep], nf)
            row["n_by_family"] = [int((fam == g).sum()) for g in range(nf)]
        else:
            row["loss_by_family"] = None

        pb = bucket_positions(pos)
        row["position_edges"] = POS_EDGES
        row["loss_by_position"] = group_mean(loss, pb, len(POS_EDGES) - 1)

        cells.append(row)
        print(f"  {arm:5s} s{seed}  mean={mean_loss:.4f}  ppl={np.exp(mean_loss):7.2f}  "
              f"depth={row['mean_exit_depth'] if row['mean_exit_depth'] is None else round(row['mean_exit_depth'],3)}  "
              f"{check}")

    # Aggregate per arm, and run the same exact test the canonical tables use so the
    # floor and the uncorrected status are explicit rather than implied.
    agg, arms = {}, sorted({c["arch"] for c in cells})
    for a in arms:
        sel = [c for c in cells if c["arch"] == a]
        means = [c["mean_loss"] for c in sel]
        agg[a] = {
            "n_seeds": len(sel),
            "mean_loss": st.mean(means),
            "std_loss": st.stdev(means) if len(means) > 1 else None,
            "loss_by_decile_mean": [
                (None if any(c["loss_by_decile"] is None for c in sel) else
                 st.mean([c["loss_by_decile"][g] for c in sel
                          if c["loss_by_decile"][g] is not None])
                 if any(c["loss_by_decile"] and c["loss_by_decile"][g] is not None
                        for c in sel) else None)
                for g in range(N_DECILES)],
        }

    pairwise = {}
    for i, a in enumerate(arms):
        for b in arms[i + 1:]:
            xa = [c["mean_loss"] for c in cells if c["arch"] == a]
            xb = [c["mean_loss"] for c in cells if c["arch"] == b]
            r = _seed_stats.perm_test(xa, xb)
            if r is not None:
                pairwise[f"{a}_vs_{b}"] = {
                    "gap": r["gap"], "p_value": r["p_value"], "min_p": r["min_p"],
                    "cohens_d": r["cohens_d"],
                    "status": "UNCORRECTED -- exploratory, not in a declared family"}

    out = Path(args.out) if args.out else (
        REPO / "results" / "language" /
        f"stratified_{args.split}"
        f"{'_d' + str(args.fixed_depth) if args.fixed_depth else ''}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "status": "EXPLORATORY. Uncorrected and outside both declared families in "
                  "code/confirmatory_tests.json, which prohibits adding a family after "
                  "results exist. No significance verdict from this file may appear in "
                  "an abstract or conclusion.",
        "split": args.split, "fixed_depth": args.fixed_depth,
        "corpus_override": args.corpus,
        "stratified_by": "TARGET token id (the token being predicted), not the input "
                         "token -- difficulty is a property of what must be predicted",
        "population": "every position except each block's last, which has no next-token "
                      "target; identical to the depth/hist population",
        "depth_caveat": "The canonical MoRE arm forced-exits 97.3% of tokens at step 7 "
                        "with depth/std 0.43, so a flat depth_by_decile is expected and "
                        "is a statement about the halting collapse (T-LX.19/T-LX.20), "
                        "NOT about the routing architecture.",
        "n_deciles": N_DECILES, "position_edges": POS_EDGES,
        "cells": cells, "by_arm": agg, "pairwise_uncorrected": pairwise,
    }, indent=2) + "\n", encoding="utf-8")

    print(f"\nby arm ({args.split}):")
    for a in arms:
        s = agg[a]
        sd = f" +- {s['std_loss']:.4f}" if s["std_loss"] is not None else ""
        print(f"  {a:5s} n={s['n_seeds']}  {s['mean_loss']:.4f}{sd}")
    if len(arms) < 2:
        print("  (one arm only -- a stratified COMPARISON needs the MoE and MoR "
              "checkpoints, which do not exist yet)")
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
