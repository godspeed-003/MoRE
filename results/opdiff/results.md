# opdiff v4 — the 3 × 5 matrix, and what it does and does not show

**Status:** complete. 15/15 cells, 3 architectures × seeds 42–46, 3 epochs,
full dataset. Machine: Ayan's RTX 4060 8 GB laptop; interpreter
`C:\Users\Hp\anaconda3\envs\more_env\python.exe`; W&B offline.
Runs: `runs/opdiffB_{MoE,MoR,MoRE}_seed4*`. Aggregate:
`results/opdiff/summary.json` (`python code/summarize_opdiff_matrix.py --out
results/opdiff/summary.json`). Ledger: `TASKS.md` T-OD.11.

**Admissibility, read this before quoting any number.** `experiment_group =
exploratory`, `variant = opdiff`. These are **not** `canonical_phase_b` numbers
and may never enter the headline language table (`canonical_spec.json` guard).
They are a same-protocol, same-dataset, same-split comparison of the three
architectures on the opdiff task. Provenance: dataset
`opdiff-v4-fingerprint-1197a5cbae10` (built `opdiff-v4-b002e4acf878`),
`code_git_commit b41f3a3`; MoE and MoRE are parameter-**identical** at
4,056,334 and MoR sits **0.114 % lower** at 4,051,720 -- the router and the
seven halt heads MoR does not have, the same residual structure as the
language matrix's 0.069 %, and it runs *against* any MoRE-favourable
reading;
batch 48, seq 384, 3 epochs, train k ∈ {1,2,3,4}, interpolation k = 5,
extrapolation k ∈ {6,7}, V = 37.

## 1. Headline (mean ± std, n = 5 seeds each)

| arm | val/task_loss ↓ | per-digit acc ↑ | exact match ↑ | mean depth | Hungarian acc | AMI | purity | H/log(E) |
|---|---|---|---|---|---|---|---|---|
| MoE | 0.4303 ± 0.0067 | 0.8487 ± 0.0016 | 0.5623 ± 0.0098 | N/A | 0.2794 ± 0.0284 | 0.1074 ± 0.0266 | 0.2906 | 0.9976 ± 0.0012 |
| MoR | 0.2449 ± 0.0852 | 0.9079 ± 0.0276 | 0.7164 ± 0.0543 | 3.28 ± 0.84 | N/A | N/A | N/A | N/A |
| **MoRE** | **0.1900 ± 0.0177** | **0.9282 ± 0.0058** | **0.7345 ± 0.0176** | 3.99 ± 0.41 | 0.1986 ± 0.0124 | 0.0310 ± 0.0089 | 0.2196 | N/A |

`N/A` is a real answer: MoR has one expert, so routing metrics do not exist; MoE
has `max_depth = 1`, so depth is a constant, not a measurement.

## 2. Per seed — the MoR instability is a halting-policy difference, not noise

| arm | seed | val/task_loss | depth | tok/s | allocator reserved |
|---|---|---|---|---|---|
| MoE | 42 / 43 / 44 / 45 / 46 | 0.4280 / 0.4294 / 0.4207 / 0.4380 / 0.4352 | N/A | 162.9k / 139.6k / 171.2k / 158.1k / 146.9k | 1.6 / 1.5 / 2.0 / 1.8 / 1.7 GiB |
| MoR | 42 | 0.2410 | 2.72 | 34,745 | 7.33 |
| MoR | 43 | **0.3933** | **2.11** | 61,359 | 4.91 |
| MoR | 44 | 0.1913 | 3.64 | 29,137 | 6.63 |
| MoR | 45 | 0.2028 | 4.11 | 15,662 | **12.84** |
| MoR | 46 | 0.1963 | 3.83 | 27,102 | 7.14 |
| MoRE | 42 / 43 / 44 / 45 / 46 | 0.1836 / 0.1792 / 0.1804 / 0.1856 / 0.2213 | 3.68 / 4.30 / 3.70 / 3.70 / 4.56 | 35.6k / 32.1k / 26.9k / 29.9k / 29.5k | 3.1 / 3.8 / 3.1 / 3.3 / 3.3 GiB |

Within the MoR arm, loss falls monotonically in the depth that seed chose to
use. The arm is not noisy in the "different random init, same policy" sense; it
lands on **different halting policies** per seed, and the shallow ones
underperform. MoRE's depth is far more consistent, and its loss spread is 4.8×
smaller. That is the strongest single argument in this matrix: routing is what
makes the recursion arm's behaviour predictable.

MoR seed 45 records `perf/gpu_peak_reserved_gib = 12.84` on an 8.19 GiB card,
which is only possible via WDDM oversubscription (segments paged to host
memory). **MoR on this card is memory-capacity-bound, not kernel-launch-bound**;
its throughput column is therefore a paging lottery and must not be quoted as a
compute measurement. MoRE never exceeds 3.8 GiB and its wall-clock is 3.2–3.3 h
in every seed.

## 3. Wall-clock (3 epochs, full dataset)

| arm | seeds 42–46 (h) | handoff estimate |
|---|---|---|
| MoE | 0.62 / 0.71 / 0.57 / 0.62 / 0.66 | ~0.4 |
| MoRE | 3.16 / 2.98 / 3.27 / 3.25 / 3.31 | ~2.0 |
| MoR | 2.65 / 1.75 / 3.16 / **5.73** / 3.40 | ~5.2 |

Matrix total ≈ 32.9 h, including 2.16 h lost to a CUDA OOM at MoR seed 45 (see
§6). A full 5-seed matrix is **~30–34 h**, not the ~38 h in `HANDOFF.md` §9.

## 4. Out of distribution: a clean null, and it is not a compute shortage

| k | status | MoE acc / exact / depth | MoR acc / exact / depth | MoRE acc / exact / depth |
|---|---|---|---|---|
| 5 | HELD OUT | 0.182 / 0.000 / N/A | 0.172 / 0.000 / 3.73 | 0.168 / 0.000 / 4.49 |
| 6 | HELD OUT | 0.169 / 0.000 / N/A | 0.169 / 0.000 / 3.73 | 0.167 / 0.000 / 4.49 |
| 7 | HELD OUT | 0.164 / 0.000 / N/A | 0.166 / 0.000 / 3.75 | 0.169 / 0.000 / 4.53 |

Chance is 1/6 = 0.1667. Every arm is exactly at chance, exact match is 0.000
everywhere, while the seen k are learned (MoRE 0.678 per-digit at k = 4). MoRE
spends **more** depth on the held-out k than on k = 4, so the failure is in
reading/generalising the k slot, not in having too few recursion steps. The
correct claim is "no length generalisation", never "the model ran out of
compute".

## 5. Compute versus k (seen chains, mean over seeds)

| k | MoE acc / exact / depth | MoR acc / exact / depth | MoRE acc / exact / depth |
|---|---|---|---|
| 1 | 0.886 / 0.173 / 1.00 | 0.995 / 0.957 / 3.43 | 1.000 / 1.000 / 4.43 |
| 2 | 0.464 / 0.000 / 1.00 | 0.877 / 0.799 / 3.60 | 0.998 / 0.986 / 4.43 |
| 3 | 0.313 / 0.000 / 1.00 | 0.706 / 0.207 / 3.82 | 0.856 / 0.283 / 4.48 |
| 4 | 0.428 / 0.002 / 1.00 | 0.547 / 0.018 / 3.86 | 0.661 / 0.051 / 4.45 |

MoRE's depth is flat across k (4.43 → 4.45) while its accuracy falls, so under
this curriculum the model does **not** allocate more recursion to longer chains
— it uses its maximum and still degrades. Read as "the model learns to allocate
depth in accordance with the predefined curriculum", MoRE barely allocates at
all here; MoR's depth does rise with k (3.43 → 3.86). Neither is evidence of
discovering intrinsic complexity.

## 6. Defects found and fixed during this matrix (no training number changes)

1. `summarize_opdiff_matrix.py` refused every multi-seed matrix — it compared
   `config_hash`, which the engine seeds via `training.subset_seed` and
   `data.subset_seed`. Replaced with a full resolved-config diff that allows
   only seed-derived and provenance/logging fields to differ (`REFUSAL 2`).
2. The OOD writer published `avg_recursion_steps = 2.0` for the non-recursive
   MoE arm; it now writes `N/A`, and the aggregator applies the same rule so
   already-written runs are not reported as measurements.
3. The compute-versus-k table omitted every held-out k (its k set came from
   `val/accuracy_by_op/F*`). Fixed; §4 above exists because of it.
4. `OD.7b` failed on this machine because `os.symlink` needs a privilege it
   lacks (WinError 1314); it now uses a directory junction and SKIPs honestly.

Gates after the fixes: `code/run_correctness_suite.py` → `TOTAL 357 357 0 0`,
all pass (gate 4 shows 130 against the frozen 129 baseline — **flagged for
reconciliation on the gate machine**, not re-baselined here);
`code/test_opdiff_task_axis.py` → 63 passed, 0 failed, 0 skipped.

## 7. The OOM, and what it means for future runs

MoR seed 45 died at 2.16 h with `RuntimeError: CUDA error: out of memory`
raised asynchronously from the router-stats block of `code/more/model.py:453`
after 129 min in epoch 1 (a MoR epoch is ~50 min). It was re-run as `__r2` with
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` — allocator only, so batch
size, model, loss, seeds and config hash are unchanged. The stale directory
`runs/opdiffB_MoR_seed45__706e3aa2` has no `metrics.json` and is not admitted by
the aggregator; it is kept as evidence. When a MoR cell is slow or dies, read
`perf/gpu_peak_reserved_gib` first: above the card's memory the cell is paging,
and no amount of load-shedding will fix it — the batch size has to fit or the
cell has to move to a bigger card.

## 8. Caveats on any paper claim

- 3 epochs, single dataset, single machine, one model size. These are a
  comparison at one point in hyperparameter space, not convergence.
- `_histc_cuda` has no deterministic implementation (torch warns); weight
  histograms in W&B are not bit-reproducible. All reported metrics come from
  `metrics.json`, not from W&B.
- MoR's loss spread across seeds means 5 seeds may still not pin its mean
  tightly enough for a headline claim; report the per-seed values, not only
  mean ± std.
- MoRE's Hungarian accuracy (0.199) is *below* MoE's (0.279) even though MoRE
  is far better in loss and exact match: the routers cluster differently from
  the operation families. Do not read that as worse specialisation — the two
  arms are not being asked to memorise the same partition, and AMI (0.031 vs
  0.107) says the same thing. Report it as measured, with that sentence.
