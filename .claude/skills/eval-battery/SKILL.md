---
name: eval-battery
description: Run the language-model evaluation battery on MoRE/MoE/MoR checkpoints — stratified loss, test split, budgeted-depth curve, BLiMP, OOD perplexity, subspace and dynamic probes — and read the results. Use whenever a trained language checkpoint must be scored, when a new arm lands, or when the suite is re-run for the paper.
---

# Running the language evaluation battery

Five scripts under `code/`, one per question. They all share the same three moves:
load a canonical checkpoint (`eval_test_split.build_model` + `checkpoint.pt`), run a
forward pass, and emit something groupable. Copy that pattern for any new eval.

## Which script answers which question

| script | evaluates | writes |
|---|---|---|
| `eval_stratified.py` | per-token loss grouped by frequency decile, position bucket, and POS family; `--split test` for the held-out split; `--fixed-depth k` for the budgeted-depth curve | `results/language/stratified_{val,test}.json`, `stratified_val_d{k}.json` |
| `eval_blimp.py` | BLiMP minimal-pair accuracy (67 paradigms, chance 50%) | `results/language/blimp.json` |
| `eval_ood.py` | zero-shot perplexity on a corpus the models never saw | `results/language/ood_{corpus}_{split}.json` |
| `eval_subspace.py` | subspace-trapping probe: participation ratio + inter-expert separation | `results/language/subspace.json` |
| `eval_dynamic.py` | across-depth velocity + update collinearity (MoRE only) | `results/language/dynamic.json` |

## The command

```bash
cd "D:/res/git/MoRE/.claude/worktrees/english-language-dataset-migration-92946e"
export PYTHONIOENCODING=utf-8
PY="D:/res/git/MoRE/.venv_cuda/Scripts/python.exe"
"$PY" code/eval_stratified.py --split val
"$PY" code/eval_stratified.py --split test
for d in 1 2 3 4 5 6 7; do "$PY" code/eval_stratified.py --fixed-depth $d; done
"$PY" code/eval_blimp.py
"$PY" code/eval_subspace.py
"$PY" code/eval_dynamic.py
```

`PYTHONIOENCODING=utf-8` is mandatory (the exporter and these scripts print ± and →).
Use the **CUDA** interpreter. No GPU is needed for BLiMP/stratified/OOD/subspace/dynamic
beyond inference; all are minutes to ~40 min.

Every script auto-discovers canonical cells under `runs/` that have BOTH
`resolved_config.json` and `checkpoint.pt`, filtered by `experiment_group ==
canonical_lang_b`. Add `--arch more` / `--run <dir>` to restrict.

## Getting the checkpoints in (the #1 blocker)

`runs/**/checkpoint.pt` is **gitignored** (`.gitignore:21`), so a `git push` of a run
directory carries metrics but NOT weights. This is how the first MoE/MoR weights were
lost. To get weights onto a machine: Ayan zips them and they are extracted to a local
`moe-chkpt/chkpt/<run_id>/` folder, then copied into the matching `runs/<run_id>/`:

```bash
for h in seed42__51000018 seed43__7d85fe81 seed44__c348b39a seed45__64800f08 seed46__3d9d12a9; do
  cp "moe-chkpt/chkpt/langB_MoE_$h/checkpoint.pt" "runs/langB_MoE_$h/checkpoint.pt"
done
```

Match by the run-dir hash against the zip's directory names, and verify the
`config_hash` in `resolved_config.json` matches before trusting the copy.

## Traps that cost real time here — read before a fresh run

1. **`--fixed-depth` prints `MISMATCH` — that is a FALSE ALARM.** The verify line
   compares the recomputed mean loss against the run's *published* `val/task_loss`,
   which was measured at full adaptive depth. A depth-crippled model legitimately does
   not reproduce it. The numbers written to JSON are correct. The check is only
   meaningful on `--split val` with no `--fixed-depth`.
2. **`--fixed-depth > 1` used to crash on MoE** (`max_depth 1`). It now SKIPS such arms
   with a note — an overnight `for d in 1..7` loop stays alive once MoE is present.
   Don't remove that skip.
3. **Stratify by the TARGET token id `x[:, 1:]`, never the input.** Difficulty is a
   property of what must be predicted. `token_decile.npy` / `token_family.npy` are
   per-TYPE arrays gathered with the target ids. Getting this backwards yields a
   plausible-looking table answering the wrong question.
4. **The population is every position except each block's last** (no next-token
   target) — identical to `depth/hist`, so the columns align with the depth keys.
5. **BLiMP uses length-grouped batching, not padding.** The language model is validated
   for an all-True `step_mask`; padding would route zeroed vectors into attention as
   keys. Sentences of equal token length batch together with no padding.
6. **The tokenizer is the wikitext-103 one, always.** A different BPE makes the numbers
   incomparable to each other and to nothing published. Caveat for the paper: absolute
   BLiMP accuracy is not comparable to 32k–50k-vocab models; only cross-arm is.
7. **OOD corpora must be parquet** — `datasets >= 3` refuses script-based datasets
   (`ptb_text_only` is dead). `eval_ood.py` uses `NeelNanda/pile-10k` (strongly OOD) with
   a `text8` fallback (weak OOD — also Wikipedia-derived). NOT wikitext-2 (its val split
   is byte-identical to wikitext-103's, T-L2.0).
8. **OOD needs the HF token and a block cap.** Load it from `.env` (key
   `hf_access_token`), map to `HF_TOKEN`/`HUGGING_FACE_HUB_TOKEN`, **never echo or commit
   it**. The Pile's long documents pack into tens of thousands of blocks — always pass
   `--max_blocks 2000` (a few thousand give a stable estimate). Re-runs: `HF_HUB_OFFLINE=1`
   reuses the cache.
9. **The dynamic probe forces `fixed_depth`.** Adaptive halting shrinks the active set
   each step, so `layer_norm` output changes length and row-order and consecutive states
   cannot be differenced. Forcing full depth fixes the token order. The hook is on the
   single `*.layer_norm` `nn.LayerNorm` inside `MoREWrapper` (model.py:878).

## Reading the results, and reporting them honestly

- **Every one of these is EXPLORATORY.** `code/confirmatory_tests.json` declares two
  families and FORBIDS adding a third after results exist. Report with effect sizes and
  the exact randomization test (`code/seed_stats.py`), labelled **uncorrected**, and keep
  them out of any abstract/conclusion significance claim. The 5-vs-5 resolution floor is
  0.00794 and every headline comparison sits on it.
- **`exp` is convex:** report perplexity as `exp(mean nats)`, never as the mean of the
  per-seed perplexities. A gap of Δ nats is a **factor `exp(Δ)`** in perplexity — e.g.
  0.656 nats is ×1.93, which is why a "2k" perplexity difference and "0.66 nats" are the
  SAME result reported two ways.
- Mine per-decile and per-phenomenon breakdowns from the JSON (`by_arm`,
  `loss_by_decile`, `by_phenomenon`); they carry the mechanism signal (MoRE's advantage
  grows with difficulty; BLiMP gains concentrate on long-distance agreement).
- The canonical **`results/language/results_tables.md`** is generated separately by
  `code/export_results.py --task language`; these eval JSONs are NOT part of it and must
  not be merged into it.

## After MoR lands (or any new arm)

The scripts auto-include every arm with a checkpoint, so once MoR weights are present
just re-run the battery — no edits. Then re-export the canonical table and append a
`changelog.md` entry naming the new numbers. MoE has `max_depth 1`, so the depth curve
and dynamic probe correctly report N/A for it by construction: the budgeted-depth curve
is a capability the single-pass baseline structurally lacks.
