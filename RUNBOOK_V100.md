# RUNBOOK_V100.md — renting a V100 and running the canonical language matrix on it

**Audience.** Whoever provisions the rented box (Vedant, or an agent driving it).
`HANDOFF.md` is the equivalent document for Ayan's local 4060 and is still correct for
that machine; this file replaces it when the GPU is rented. `SETUP.md` remains the
authority on the environment and the corpus build — this file only names which of its
steps apply on a fresh cloud box and in what order.

**Why a V100 and not the cheapest available card.** The 15 published `canonical_lang_b`
cells were all trained on a V100. Ten of them (MoE ×5, MoR ×5) are being **kept** —
T-LX.12 changed only the MoRE arm's architecture, so re-running MoE and MoR would burn
~16 GPU-hours to reproduce numbers that are already valid. But that means the
re-run MoRE cells will be compared against MoE/MoR cells trained on different silicon
unless the card matches. Per-epoch time, `perf/throughput_items_sec` and peak memory are
only comparable within one card, and a per-machine confound inside the arm comparison is
exactly what `CLAUDE.md` §9 refuses for the seed matrix. **Rent a V100.**

---

## What is actually being run, and what is not

| arm | cells on disk | status after T-LX.12 | action |
|---|---|---|---|
| MoE | 5 | canonical, untouched — no protocol field moved | **keep. Do not re-run.** |
| MoR | 5 | canonical, untouched — `num_experts = 1`, the axis cannot reach it | **keep. Do not re-run.** |
| MoRE | 5 | now a *labelled per-step ablation*: resolving that config today yields `variant = "language+route_per_step"` | **re-run all 5 seeds** |

So this is a **5-run job, not a 15-run job**. `--all` would skip the 10 completed cells
anyway (a cell counts as complete on arm + seed + canonical group + finished
`metrics.json`), but say what you mean: use `--arch more`.

The 5 existing MoRE directories are **evidence, not garbage** (`CLAUDE.md` §6). They are
the per-step ablation the fix is measured against. Do not delete them and do not let the
runner overwrite them — it appends `__r2`, `__r3`, … rather than overwriting, which is
why the tree already carries several such suffixes.

---

## Budget, derived from the V100 cells themselves

One epoch is 526,320 blocks of 256 tokens. Training throughput as published
(`perf/throughput_items_sec`, mean over 5 seeds, on the V100):

| arm | items/s | h / epoch | × 3 epochs | × 5 seeds |
|---|---|---|---|---|
| MoE | 718.8 ± 7.2 | 0.20 | 0.61 | 3.1 |
| MoR | 173.8 ± 6.3 | 0.84 | 2.52 | 12.6 |
| **MoRE** | **168.5 ± 2.5** | **0.87** | **2.60** | **13.0** |

Those are **training-loop hours only** — validation, the depth/routing metric pass and
checkpoint writes are on top, and the runner prints real elapsed hours as it goes. Budget
**~15 h of V100 time for the MoRE arm**, and rent for 18 h so a restart does not strand you
mid-arm.

MoE's row needs one footnote: its raw `perf/throughput_tokens_sec` reads `718.81`, which is
items/s under a pre-T-LX.8 key name. The exporter repairs it at admission
(`normalize_throughput_units`); never read that key straight out of a pre-fix
`metrics.json`.

---

## Phase 1 — provision

Ubuntu 22.04 LTS, a PyTorch 2.5.x / CUDA 12.4+ image, **60–100 GB disk**. On Vast.ai the
default `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel` is fine. Add your SSH key, then:

```bash
ssh -p <PORT> root@<IP>
```

A V100 is `sm_70` and is supported by every official wheel this project uses. The model is
pure FP32 with stock `nn.MultiheadAttention` and no `torch.compile`, Triton, flash-attention
or explicit SDPA backend selection, so nothing in the code needs an Ampere-or-newer feature.

**Disk.** 60 GB is genuinely the floor, not padding: the HF Arrow cache is ~537 MB, the
packed corpus ~269 MB, and each MoRE checkpoint is ~64 MB — but the image itself plus a
second torch install (see Phase 2) is most of it.

**Pick the host on reliability, not on price.** At this scale the whole arm is ~15 GPU-hours,
so the spread between the cheapest V100 offers is single-digit dollars — $0.06/hr and
$0.085/hr differ by about $0.40 over the job. Host *reliability* is not a rounding error on
the same axis: an interruption partway through a 2.6 h cell loses that cell's wall-clock,
and the runner's resume logic deliberately re-runs any directory holding a checkpoint
without a `metrics.json`, because a checkpoint at an unknown epoch is exactly the artifact
that ends up in a table by accident. A 93.9%-reliability host is a worse buy than a
98.2%-reliability host at any price difference this job can generate. Sort by reliability,
then take the cheapest offer above ~98%.

**Record the card in the artifacts, not in your memory.** `engine.py` writes
`provenance.gpu_name`, `gpu_total_mem_gib`, `gpu_capability`, `torch_version` and
`torch_cuda_version` into each `resolved_config.json` (this arm reads
`Tesla V100-SXM2-16GB`, 15.766 GiB, `sm_70`, torch `2.6.0+cu124`). The MoE and MoR cells
predate those fields and carry `None`, so for those ten the V100 16 GB attribution rests on
the operator record in this file and nothing else — which is precisely why the fields exist
now. Do not retrofit them into the older `resolved_config.json` files; a provenance field
invented after the fact is worse than an absent one.

---

## Phase 2 — environment

```bash
apt update && apt install -y git tmux htop nvtop rsync
```

```bash
git clone https://github.com/godspeed-003/MoRE.git && cd MoRE
```

```bash
git checkout claude/english-language-dataset-migration-92946e
```

**That branch name matters.** `rtx4060-english` — the branch the older runbook named — is
not where this work lives. On the wrong branch you would get a codebase without the
T-LX.12 fix, the runs would be per-step MoRE, and the guard would happily stamp them
`canonical_lang_b` under the *old* spec. That failure is silent.

```bash
pip install -r requirements.txt
```

```bash
pip install wandb nltk scipy scikit-learn datasets tokenizers
```

**Read what the first command does to the image's torch.** `requirements.txt` pins
`torch==2.6.0`, so in a 2.5.1 image pip replaces torch with a ~2.5 GB download from PyPI.
On Linux that wheel is CUDA-enabled, so this works — but it is not a no-op and it is not
what the image advertised. Verify afterwards, and **write down what you got**:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

`True` and `Tesla V100-...` or stop. As of T-LX.13 the run itself records `torch_version`,
`torch_cuda_version`, `gpu_name`, `gpu_total_mem_gib` and `gpu_capability` into
`resolved_config.json`, so the answer ends up in the artifact rather than only in your
scrollback — but the 10 kept MoE/MoR cells predate that, and their torch version is
**not** recoverable from the repo. If the re-run MoRE cells land on a different torch than
those did, that is a real limitation of the comparison and belongs in the paper's
threats-to-validity, not in a silent footnote.

```bash
python -c "import nltk; nltk.download('averaged_perceptron_tagger_eng')"
```

nltk 3.9 renamed this resource; the older `averaged_perceptron_tagger` does not satisfy it.
`token_family.npy` is tracked in git, so you should not need to rebuild the POS lookup —
this download is insurance.

---

## Phase 3 — build the corpus, then **prove** it is the right corpus

Canonical corpus (~540 MB download, ~10 min total):

```bash
python data/lang/build_language_dataset.py --corpus wikitext-103 --stage all --cache_dir data/lang/_hf_cache
```

```bash
python data/lang/build_baseline_floors.py --corpus wikitext-103
```

```bash
python data/lang/build_depth_deciles.py --corpus wikitext-103
```

```bash
python data/lang/build_shuffled_control.py --corpus wikitext-103
```

**Do not run two of those concurrently on the same corpus.** They read-modify-write one
`dataset_meta.json`; running two at once once erased all four floor keys with no error.

Also build the dev corpus, because `--smoke` (Phase 5) needs it and ten minutes here buys
you a real end-to-end training check before the 15-hour arm:

```bash
python data/lang/build_language_dataset.py --corpus wikitext-2 --stage all --cache_dir data/lang/_hf_cache && python data/lang/build_baseline_floors.py --corpus wikitext-2 && python data/lang/build_depth_deciles.py --corpus wikitext-2 && python data/lang/build_shuffled_control.py --corpus wikitext-2
```

Then the step the older runbook omits entirely, and the one that makes every number after
it citable:

```bash
python code/test_language_data.py
```

`0 failed` required. It grades the `.npy` files you just built against SHA-256 values in the
tracked manifest, so a pass means your corpus is byte-identical to the one all 15 published
cells were measured on. A hash mismatch is a **STOP**: report the corpus name and both
hashes. Without this step a subtly different corpus produces a full, plausible, uncomparable
matrix.

---

## Phase 4 — gates, then preflight

The gates are not optional on a rented box, because you are on a **different torch build
than the `TOTAL 356` baseline was established on** (that baseline is CPU torch 2.6.0 on
Vedant's machine). Run them and record the count:

```bash
python code/run_correctness_suite.py
```

```bash
python code/test_route_persistence.py && python code/test_language_spec_freeze.py && python code/test_language_gate_l6_l7.py
```

Expect `TOTAL 356 356 0 0` / `ALL GATES PASS`, then `37`, `64`, `28` passed with `0 failed`.
`test_language_gate_l6_l7.py` prints the parameter table — **MoRE must be 5,584,908**; a
different number means the arms are not budget-matched and nothing downstream means
anything. A *different* 356 on a different torch is a version difference to diagnose, not
automatically a code defect (`SETUP.md` §0) — but it is a STOP either way until you know
which it is.

W&B, before the first run and not after it. An unconfigured W&B kills the run *inside*
`wandb.init`, after the run directory exists, leaving a `config.json` with no
`metrics.json`:

```bash
export WANDB_MODE=offline
export WANDB_DIR=/root/wandb_offline
```

`WANDB_DIR` outside the repo, or a `wandb/` directory lands in the tree.

```bash
export WANDB_MODE=offline && python code/run_language_matrix.py --preflight
```

This runs a real matmul rather than trusting `is_available()`, checks the corpus and POS
lookup, checks the spec is fully frozen, and checks the frozen config is **accepted by the
proxy guard on all three arms**. It refuses rather than warns. ~15 s.

What the banner must say on this box:

```text
[ok ] CUDA device usable  -- Tesla V100-SXM2-16GB, 16384 MiB
[ok ] corpus wikitext-103 built  -- 526,320 train blocks x 256, V=8192, lang-wikitext-103-bpe8192-len256-800d6154
[ok ] canonical_spec_language.json is fully frozen  -- L7.2-TLX.12-route-persistence
[ok ] more: the frozen config is accepted as canonical_lang_b  -- group=canonical_lang_b
preflight clean.
```

Two fields in the older runbook's expected output are now wrong and one of them is
load-bearing. `NVIDIA H100 80GB HBM3, 81559 MiB` is what that document was written against
— it is an **H100** runbook, which is also why its ~5.0 h / ~4.5 h per-arm timings are
H100 timings and not the V100 figures in the budget table above. And the spec line now
reads `L7.2-TLX.12-route-persistence`; if it still says `L7.1-language-frozen`, you are on
the wrong branch or the checkout is stale — go back to Phase 2.

---

## Phase 5 — the runs

```bash
tmux new -s more_matrix
```

Inside tmux, re-export the environment (a new shell does not inherit it):

```bash
cd /workspace/MoRE && export WANDB_MODE=offline && export WANDB_DIR=/root/wandb_offline
```

Prove the box can train, on the dev corpus, in ~10 min:

```bash
python code/run_language_matrix.py --smoke
```

That is MoRE seed 44, 1 epoch, wikitext-2, group `langB_smoke`. It **cannot** contaminate
the matrix: wikitext-2 carries its own `dataset_version`, so the guard refuses it as
canonical even if the label were wrong. `[Model] Total parameters: 5,584,908` must appear.

Then the arm:

```bash
python code/run_language_matrix.py --arch more
```

Detach `Ctrl+B, D`; reattach `tmux attach -t more_matrix`. The runner is
**interruption-safe** — re-running it skips completed cells and re-runs a directory that
has a checkpoint but no `metrics.json`, because a checkpoint at an unknown epoch is exactly
the artifact that ends up in a table by accident.

Do **not** add `--arch moe` or `--arch mor`. Nothing about those arms changed, and a
re-run would produce a second set of cells for the same `(arch, seed)` — which the
exporter's consistency check rejects as duplicate cells, so you would have spent 16 GPU-hours
to break the export.

`nvtop` in a second tmux window is worth having: a V100 at 16 GB has headroom the 6 GB 3050
did not, so if utilisation sits low the bottleneck is the loader, not the card.

---

## Phase 6 — get the artifacts OFF the box before you terminate it

**This is the phase the older runbook gets wrong, and it has already cost this project all
15 canonical checkpoints.**

`.gitignore` excludes `runs/**/checkpoint.pt` (line 21), `runs/**/*.pt` (22), `*.pt` (47)
**and** `runs/**/stdout.log` (23). So `git add runs/langB_MoRE_*` commits exactly four files
per cell — `config.json`, `resolved_config.json`, `metrics.json`, `results.tsv` — and
carries neither the weights nor the log. Terminating the instance after a
commit-and-push therefore destroys both, permanently. That is precisely what happened to the
first matrix: the checkpoints existed, were never copied off, and the box was released.

Copy them first, from your **local** machine:

```bash
rsync -avP -e "ssh -p <PORT>" root@<IP>:/workspace/MoRE/runs/ ./runs_v100_backup/
```

Then verify locally that you actually have 5 checkpoints before going any further:

```bash
find ./runs_v100_backup -name checkpoint.pt | wc -l
```

Only now aggregate. Run it **read-only first** — `export_results.py` has no `--help` and no
argparse; any invocation without `--check` performs a real export and rewrites files under
`results/`:

```bash
python code/export_results.py --task language --check
```

### Retiring the superseded cells: use `git mv`, not `mv`

The new arm and the arm it supersedes are both `canonical_lang_b` cells for the same five
seeds, so while both sit under `runs/` the exporter refuses the whole export with
`duplicate cell ('more', 42)` and writes nothing. That refusal is the guard working — it is
the check that stops two different architectures being averaged into one row.

Retire the old five into `archive/`, and **use `git mv`**:

```bash
mkdir -p archive/pre_finalization/more_l7_1_per_step
```

```bash
git mv runs/langB_MoRE_seed42__5c34c9dd archive/pre_finalization/more_l7_1_per_step/
```

Plain `mv` is the trap, and it was hit on this run. `mv` removes the directory from the
working tree but leaves it in the index at the old path; `git add <newpath>` then stages the
copy without staging the deletion. The commit therefore carries the cell at **both** paths,
the box exports fine because its own working tree is clean, and the duplicate only appears
on the next machine to pull — where the exporter refuses. Confirm before committing that the
old paths are staged as deletions:

```bash
git status --short | grep '^D' | head
```

Archive, never delete (`CLAUDE.md` §6): the retired cells are the labelled ablation the new
arm is measured against, and `depth_routeonce_vs_perstep` in `code/confirmatory_tests.json`
is a declared comparison against exactly those five directories.

Then export for real:

```bash
python code/export_results.py --task language
```

`--task language` is not optional: without it the exporter exports the *arithmetic* matrix
and writes 15 rows of MSE into `results/`. Language output goes to `results/language/`.
Never hand-copy a number out of the console into a table.

Commit only the paths you mean — **never `git add .`**, which on this box would stage the
HF Arrow cache and the packed corpus:

```bash
git add runs/langB_MoRE_* results/language/ && git status --short
```

Read that `git status --short` before committing. Then:

```bash
git commit -m "feat(matrix): MoRE canonical language arm re-run on V100 with per-token routing persistence (T-LX.12)"
```

```bash
git push origin claude/english-language-dataset-migration-92946e
```

Terminate the instance **only after** the push has succeeded *and* the local
`runs_v100_backup` holds 5 checkpoints and 5 `stdout.log` files.

---

## What must be true before any of this counts as canonical

- `preflight` printed `L7.2-TLX.12-route-persistence` and accepted `more` as
  `canonical_lang_b`.
- `test_language_data.py` passed, so the corpus is the published one.
- Each new cell's `resolved_config.json` records `variant = "language"` (not
  `language+route_per_step`) and `routing_persistence = "per_token"`.
- `dispatch/router_calls_per_dispatch` is **1/7**, not 1 — that metric is the artifact-level
  proof the fix was in effect, and it is the first thing to read in the new `metrics.json`.
- The 5 old MoRE directories are still present and unedited.
