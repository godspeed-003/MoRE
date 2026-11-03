# Language subset sweep — STATE AT STOP (T-OD.13 TEST 2)

Stopped on request ("abort everything, resume tomorrow"). Both processes killed;
no `python.exe` left running and the GPU is idle (213/8188 MiB). All run
directories are on disk and untouched; nothing was deleted or edited.

## Progress: 35 of 45 cells complete (10 remaining)

Correction: an earlier draft of this file said 30; the progress TSV lists 35
rows (15 + 15 + the five MoE 40 % cells). Remaining = MoR 40 % × 5 and
MoRE 40 % × 5.

`results/language/subset_sweep_progress.tsv` is the authoritative list (one line
per completed cell, written only after `metrics.json` exists).

| scale | MoE | MoR | MoRE |
|---|---|---|---|
| 2.5 % | 5/5 | 5/5 | 5/5 |
| 10 % | 5/5 | 5/5 | 5/5 |
| 40 % | 5/5 | 0/5 | 0/5 |

Killed mid-cell: `MoR f=0.40 seed42` — directory
`runs/langSub_MoR_f400_seed42__e813baf0` has `checkpoint.pt` and `results.tsv`
but **no `metrics.json`**. It is an interrupted run at an unknown epoch and must
NOT be trusted; the driver's resume rule re-runs it and `_allocate_id` gives the
retry a `__r2` suffix. Do not hand-promote that checkpoint.

## Early signal (do not write up yet — 40 % arms are missing)

val/task_loss, mean over 5 seeds (nats/token, bigram floor 4.9849):

| scale | MoE | MoR | MoRE |
|---|---|---|---|
| 2.5 % | 5.154 | 5.067 | 4.968 |
| 10 % | 4.427 | 4.335 | 4.150 |

MoRE < MoR < MoE at both completed scales, by margins that do NOT shrink with
data (MoRE−MoR: −0.099 at 2.5 %, −0.185 at 10 %). Also: at 2.5 % MoRE's learned
depth is 3.83–5.31 and at 10 % it saturates at 6.74–6.82, i.e. the halting
collapse the language matrix saw returns as soon as the budget allows it — the
sweep is doing what it was designed to test.

## Resume (exactly this)

```powershell
cd C:\Users\Hp\Desktop\MoRE
$env:WANDB_MODE="offline"
& "C:\Users\Hp\anaconda3\envs\more_env\python.exe" code/run_lang_subset_sweep.py --all
```

`--all` skips the 35 completed cells (identity checked on architecture, seed,
`experiment_group`, and `data.subset_fraction`, not on directory name) and starts
at MoR f=0.40 seed42. Remaining cost ≈ 2 h (MoR seed42, restarted from scratch)
+ 4 × ≈ 2.2 h (MoR) + 5 × ≈ 2.8 h (MoRE) ≈ **24–25 h**.

## Still owed before any of this is citable

1. `results/language/subset_sweep.{json,md}` aggregator (per-arm mean ± std per
   scale, and the MoRE/MoR margin ratio per scale — the number the sweep exists
   to measure).
2. Correct the `expandable_segments` claim in `changelog.md` and
   `results/opdiff/results.md`: it is a NO-OP on this platform, so it did not
   convert the MoR-45 OOM into paging; the 12.84 GiB reserved reading is WDDM
   oversubscription without it.
3. TEST 1 write-up is complete in `results/opdiff/depth_control.md`; its numbers
   go into the paper with BOTH estimators (depth-matched ratio 1.01× of means /
   3.61× of medians; best-uniform 0.43×), never a single ratio.
4. `code/test_depth_control.py`: TD.1's identity-order clause is wrong (a subset
   is a permuted prefix, not a prefix of `range(n)`); fix before quoting 15/15.
5. Uncommitted: `code/more/model.py` (force_depth), `code/eval_depth_control.py`,
   `code/more/cli.py` + `code/more/run_context.py` (--subset_fraction),
   `code/test_depth_control.py`, `code/run_lang_subset_sweep.py`, and
   `data/lang/wikitext-103/dataset_meta.json` (decile rebuild: every recorded
   value reproduced bit-identically; only `decile_generator_commit` changed).
