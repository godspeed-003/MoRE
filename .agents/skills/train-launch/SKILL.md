---
name: train-launch
description: Launch a MoRE training run correctly and read its per-run outputs — the CUDA interpreter, the flags that change scientific identity, the proxy guard, and which files carry the answer. Use for any diagnostic run, timing run, or canonical matrix arm.
---

# Launching a run and reading it back

## The command

```bash
cd "D:/res/git/MoRE/.Codex/worktrees/english-language-dataset-migration-92946e/code" && "D:/res/git/MoRE/.venv_cuda/Scripts/python.exe" train.py --architecture more --seed 42 --run_name phaseB_more
```

Use the **CUDA** interpreter for anything trained or timed
(`D:/res/git/MoRE/.venv_cuda/Scripts/python.exe`). The CPU interpreter runs the
gates. RTX 3050 **6 GB** — see [ENVIRONMENT.md](../../../ENVIRONMENT.md).

Add `WANDB_MODE=disabled` for smoke and diagnostic runs so they never reach the
dashboard:

```bash
cd "D:/res/git/MoRE/.Codex/worktrees/english-language-dataset-migration-92946e/code" && WANDB_MODE=disabled "D:/res/git/MoRE/.venv_cuda/Scripts/python.exe" train.py --architecture mor --epochs 1 --seed 44 --run_name t_diag
```

Run it in the **background** and read the output file — a canonical arm is tens of
minutes and the session should not block on it.

`train_moe.py` / `train_mor.py` / `train_more.py` are thin launchers over the same
`engine.py`. There is one training system; `--architecture` is a runtime mode.

## Flags, and which ones change what the run *means*

Shape/protocol overrides — all reach `resolved_config.json` and the config hash:
`--config --epochs --blocks --batch_size --seed --run_name --ffn_mult`

Flags that make the run a **labelled ablation**, not canonical. `resolve_variant`
derives the label from the config; you never declare it:

| flag | what it turns on |
|---|---|
| `--halting_supervision` (+ `--halting_supervision_weight`) | the operation-complexity curriculum as a supervised halting target. Default OFF = pure unsupervised ACT. |
| `--step_routing <w>` | **routing supervision against the oracle expert index.** Non-zero = oracle-routing ablation (T10.E). Never in a canonical table. |
| `--routing_balance <w>` | balance-loss coefficient. Choose from *measured* magnitudes, then freeze in `config.json`. |
| `--family_cls <w>` | the 6-way family cross-entropy. Canonical **0.5**; `0.0` is the T10.H no-family-supervision ablation. |
| `--experiment_group <g>` | run group. Default `exploratory`. |

`--halting_supervision_weight 0` with `--halting_supervision` set is refused: a
curriculum flag with zero weight contributes nothing and must not look like a
variant.

## The proxy guard — expect it to refuse you

`--experiment_group canonical_phase_b` is checked against
`code/canonical_spec.json` and the run is **aborted** (`ProxyGuardError`) unless it
matches that spec exactly, uses the full dataset, and names a seed from the frozen
set `42–46`. While any spec field is `null` the guard refuses *every* canonical
claim. **That is deliberate, not a bug** — do not edit the spec to get a run
through. Anything with a non-canonical epoch count, subset fraction, dataset
version, batch protocol, loss configuration or architecture is a **proxy** and no
proxy enters the headline table (`AGENTS.md` §6).

## Reading the run back

Output is `runs/<experiment_id>/`, the name ending in the first 8 hex digits of the
config hash. Change any hashed field and the identity changes with it. There is
deliberately **no timestamp** — the same config re-run is recognisably the same
run. No global `best_model.pt`, `results.tsv` or `*_results.csv` is ever written;
`code/more/run_context.py` refuses.

| file | what it answers |
|---|---|
| `resolved_config.json` | what actually ran, after every override and architecture default. **`provenance` here is the record** — architecture, variant, seed, dataset/split version, git commit, config hash, shapes, resolved epochs and subset fraction. |
| `metrics.json` | the measured series, including the permutation-invariant routing block |
| `results.tsv` | per-epoch table |
| `stdout.log` | the run's own log |
| `checkpoint.pt` | git-ignored; regenerable from `resolved_config.json` |

Checkpoint selection is the **final epoch**, not best-val (T11.0b): best-val is an
order statistic whose downward bias scales with unmatched per-arm validation noise,
so it flatters whichever arm is noisier.

## Reading the numbers honestly

- **Primary metric is validation task loss.** Never `total_loss`. A negative
  weighted total is not automatically a bug and is never evidence of quality.
- Routing accuracy alone is uninformative under an unsupervised router — quote
  **Hungarian-matched accuracy, AMI, purity**. Expert indices carry no semantic
  identity.
- Entropy is normalized `H/log(E)` and is **`N/A` at E=1**, never `0` used as a
  score. Every routing and diversity key on a MoR run is the *string* `"N/A"`,
  **present, not absent**.
- Low cosine similarity is "consistent with differentiated parameterizations",
  never proof of orthogonality.
- "Depth allocation error", never "compute efficiency".
- Seeds `42, 43, 44, 45, 46`, report **mean ± std**. One favourable seed is never
  evidence. The randomization-test resolution floor at 5v5 is **0.0040** — a
  smaller difference is not resolvable, whatever the point estimate looks like.
