# SETUP.md — building the environment for a MoRE **language** run

Written for the machine that will actually run the canonical matrix (Ayan's RTX 4060
Laptop, 8 GB). Everything here has been executed on a 3050 6 GB; the only expected
difference is more VRAM headroom, which changes the numbers Gate L5 picks, not the
steps.

If a command below fails, **stop and report it** rather than working around it. Most of
these steps exist because a silent success would produce numbers that look fine and
aren't — see `changelog.md` for the specific cases.

---

## 0. What you are setting up

Two Python interpreters, deliberately:

| | used for | why separate |
|---|---|---|
| **CPU** — a plain Conda/system Python with `torch` CPU | every `test_*.py`, `run_correctness_suite.py` | the correctness gates must not depend on a GPU being present or on which GPU it is |
| **CUDA** — a venv with `torch==2.6.0+cu126` | training, timing, the seed matrix | the only place a run's numbers come from |

`ENVIRONMENT.md` records the exact versions this was developed against.

---

## 1. Clone the branch

```bash
git clone https://github.com/godspeed-003/MoRE.git
```

```bash
cd MoRE && git checkout claude/english-language-dataset-migration-92946e
```

## 2. CPU interpreter (the gates)

Any Python 3.12 with `torch` (CPU build), `numpy`, `scipy`, `datasets`, `tokenizers`,
`nltk`, `wandb`, `matplotlib`, `scikit-learn`. If you have Anaconda, its base
environment plus `pip install datasets tokenizers nltk wandb` is enough.

Confirm it, and note the path you used — call it `PY_CPU`:

```bash
python -c "import torch, numpy, scipy, datasets, tokenizers, nltk, wandb; print(torch.__version__, torch.cuda.is_available())"
```

`torch.cuda.is_available()` printing `False` here is **correct** — this interpreter is
the CPU one.

## 3. CUDA interpreter (the runs)

Built with `--system-site-packages` on purpose, so `numpy`/`scipy`/`datasets`/`nltk`
come from the base install and **only torch differs**. Two independent copies of numpy
is how a metric ends up computed by two slightly different libraries.

```bash
python -m venv --system-site-packages .venv_cuda
```

Then install CUDA torch **over** the inherited CPU torch. The
`--ignore-installed torch` is the step most often got wrong — without it pip sees the
inherited CPU torch, decides the requirement is satisfied, and you get a venv that
silently trains on CPU at ~1/40 the speed:

```bash
.venv_cuda/Scripts/python.exe -m pip install --ignore-installed torch==2.6.0 --index-url https://download.pytorch.org/whl/cu126
```

(On Linux/macOS the path is `.venv_cuda/bin/python`.)

Verify — this must print `True` and name your GPU:

```bash
.venv_cuda/Scripts/python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## 4. The nltk tagger (needed only to rebuild the POS lookup)

```bash
python -c "import nltk; nltk.download('averaged_perceptron_tagger_eng')"
```

nltk 3.9 renamed this resource; the older `averaged_perceptron_tagger` will **not**
satisfy it, and `PerceptronTagger()` raises `LookupError` if it is missing.

## 5. Rebuild the dataset

`data/lang/**/*.npy` is **not** in git — 269 MB for the canonical corpus, and it is
fully reproducible from the tracked `tokenizer.json` plus the HF revision pinned in
`dataset_meta.json`. `token_family.npy` **is** tracked (8 KB) because it needs a
specific nltk model to reproduce.

Dev corpus first (~2 minutes) — it is the fast path for checking the pipeline works:

```bash
python data/lang/build_language_dataset.py --corpus wikitext-2 --stage all --cache_dir data/lang/_hf_cache
```

Then the canonical corpus. This downloads ~540 MB and takes ~10 minutes total (fetch,
then a 171 s tokenizer fit, then a 195 s encode):

```bash
python data/lang/build_language_dataset.py --corpus wikitext-103 --stage all --cache_dir data/lang/_hf_cache
```

The trivial floors and the frequency-decile/count artifacts, both corpora:

```bash
python data/lang/build_baseline_floors.py --corpus wikitext-2 && python data/lang/build_baseline_floors.py --corpus wikitext-103
```

```bash
python data/lang/build_depth_deciles.py --corpus wikitext-2 && python data/lang/build_depth_deciles.py --corpus wikitext-103
```

```bash
python data/lang/build_shuffled_control.py --corpus wikitext-2 && python data/lang/build_shuffled_control.py --corpus wikitext-103
```

**Do not run two of these concurrently on the same corpus.** They read-modify-write one
`dataset_meta.json`; running the floors and the family lookup at the same time once
erased all four floor keys with no error at all.

`token_family.npy` is tracked, so you should not need `build_family_lookup.py`. If you
do rebuild it (~19 min on 6 cores), use `--workers` and expect the hash to match only
if your nltk model version matches.

## 6. Verify the rebuild reproduced the tracked dataset

This is the step that makes everything after it trustworthy:

```bash
python code/test_language_data.py
```

Expect **`0 failed`**. The suite checks the `.npy` files you just built against the
SHA-256 values recorded in the tracked manifest, so a pass means your corpus is
byte-identical to the one every number in `TASKS_LANGUAGE.md` was measured on. A hash
mismatch is a **STOP**: report it with the corpus name and the two hashes.

## 7. Run every gate

```bash
python code/run_correctness_suite.py
```

Must print `TOTAL 356 356 0 0` and `CORRECTNESS SUITE: ALL GATES PASS`. **Any other
count is a STOP** — do not proceed, do not "fix" the count, report the actual output.
Takes ~4 minutes and launches two real 1-epoch arithmetic training runs.

```bash
python code/test_lang_causality.py && python code/test_lang_heads.py && python code/test_lang_recursion.py && python code/test_language_families.py && python code/test_language_task_axis.py
```

Expected: `22`, `25`, `11`, `89`, `72` passed, `0 failed` throughout.

## 8. W&B — settle this BEFORE the first canonical run

An unconfigured W&B kills the run in `wandb.init` **after** the run directory has been
created, leaving a `config.json` and `resolved_config.json` with no `metrics.json`.
That already happened once here. Pick one:

```bash
wandb login
```

or, to skip W&B entirely (fine for calibration, and what was used for every smoke run):

```bash
export WANDB_MODE=offline
```

If you go offline, also set `WANDB_DIR` to somewhere **outside** the repo so no
`wandb/` directory lands in the tree.

## 9. Your first language run

From inside `code/` (the config path is relative):

```bash
cd code && python train.py --architecture more --task language --corpus wikitext-2 --epochs 1 --batch_size 8 --seed 42 --experiment_group lang_smoke
```

Use the **CUDA** interpreter for anything you intend to quote. On the CPU one this takes
tens of minutes; on a 4060 it should be a few minutes.

What a correct start looks like:

```text
[Dataset] MoRELanguageDataset(wikitext-2/train: 10,527 blocks x 256 tokens, V=8192, lang-wikitext-2-bpe8192-len256-9f870794)
[Dataset] floor = 5.398304166059712 nats/token (bigram, T-L2.5)
[Model] Total parameters: 5,584,908
[Train] task=language: the confusion diagonal is published as val/routing_agreement_with_pos
[Train] halting: ponder_weight=0.001, supervision=OFF (pure ACT)
```

`5,584,908` should match exactly for MoRE. If your parameter count differs, something in
the config resolved differently and the arms are no longer budget-matched — stop and
report it.

---

## Things that will bite, and what they mean

| symptom | cause |
|---|---|
| `config.json not found` | run `train.py` from inside `code/`, not the repo root |
| `torch.cuda.is_available()` is `False` in `.venv_cuda` | the `--ignore-installed torch` step was skipped |
| `LookupError: averaged_perceptron_tagger_eng` | step 4 not done |
| `FileNotFoundError: .../train.npy` | step 5 not done for that corpus |
| `data.seq_len = N but ... was packed at seq_len = 256` | the corpus on disk was built at a different length; repack or drop the override |
| a hash mismatch in `test_language_data.py` | **STOP** — your corpus is not the one the numbers were measured on |
| `UsageError: No API key configured` | step 8 |
| `ValueError: path is on mount 'C:', start on mount 'D:'` | a path argument crossed drives; report it, this class of bug has bitten twice |
