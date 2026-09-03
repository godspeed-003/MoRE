# ENVIRONMENT.md — the machine this repository actually runs on

**Why this file exists.** `CLAUDE.md` §9 and `ARCHITECTURE.md` §9 named an
interpreter (`C:\Users\Hp\anaconda3\envs\more_env\python.exe`, torch 2.5.1) and a
GPU (RTX 4060 Laptop 8 GB) that do not exist on this machine. A future agent
following those instructions gets `No such file or directory` on its first
command, and — worse — cannot tell whether a stale path means *"wrong path"* or
*"you are on the wrong machine, stop"*. Written under **T-L0.2**; the correction
of the two documents is **T-L0.0**.

The arithmetic study's runs in `runs/` and the numbers in `README.md` /
`results/results.md` **were** produced on the RTX 4060 machine. Those references
are historically accurate and are deliberately left alone (`CLAUDE.md` §6:
archive, never delete, evidence). They are not re-run — see `TASKS_LANGUAGE.md`
T-LX.2.

---

## 1. Two interpreters, and which to use

| | Path | Use for |
|---|---|---|
| **CPU** | `C:\Users\vedan\anaconda3\python.exe` | the correctness suite and every gate; anything that only needs to be *correct* |
| **CUDA** | `D:\res\git\MoRE\.venv_cuda\Scripts\python.exe` | training runs, the seed matrix, anything timed |

Both are **Python 3.12.7, AMD64**. They differ in exactly one package — torch —
and in nothing else, because the venv was built with `--system-site-packages` and
inherits the Anaconda base site-packages for everything torch does not touch.

Anaconda **base** has `torch 2.6.0+cpu` (`cuda False`). `C:\Python314` has no
torch at all. The gate counts in `plan_language.md` and `TASKS_LANGUAGE.md`
(**356**) were measured on the CPU interpreter.

```bash
C:/Users/vedan/anaconda3/python.exe code/run_correctness_suite.py
```

```bash
D:/res/git/MoRE/.venv_cuda/Scripts/python.exe code/train.py --architecture more --seed 42
```

Git Bash accepts the forward-slash form above. In Python string literals inside
the repo, the raw-string backslash form is used instead.

---

## 2. Measured, not remembered

Produced by the probe in §6 on 2026-09-03. Differences between the two columns
are the whole reason there are two interpreters.

| | CPU (`anaconda3`) | CUDA (`.venv_cuda`) |
|---|---|---|
| python | 3.12.7 AMD64 | 3.12.7 AMD64 |
| **torch** | **2.6.0+cpu** | **2.6.0+cu126** |
| `torch.cuda.is_available()` | `False` | `True` (CUDA 12.6) |
| numpy | 1.26.4 | 1.26.4 |
| scipy | 1.14.1 | 1.14.1 |
| scikit-learn | 1.3.2 | 1.3.2 |
| pandas | 2.2.1 | 2.2.1 |
| matplotlib | 3.10.0 | 3.10.0 |
| wandb | 0.25.0 | 0.25.0 |
| datasets | 4.6.1 | 4.6.1 |
| transformers | 4.51.3 | 4.51.3 |
| tokenizers | 0.21.0 | 0.21.0 |
| huggingface_hub | 0.30.2 | 0.30.2 |
| pyarrow | 23.0.1 | 23.0.1 |
| nltk | 3.9.1 | 3.9.1 |
| spacy / stanza / flair | absent | absent |

**GPU:** `NVIDIA GeForce RTX 3050 6GB Laptop GPU`, 6143 MiB reported by torch /
6144 MiB by `nvidia-smi`, compute capability **8.6**, driver **596.08**.

6 GB is the binding constraint on the language phase and is why `T-L7.0` measures
throughput and VRAM headroom *before* the protocol is frozen rather than after a
matrix run dies at hour three. The arithmetic POC never came close to the limit;
a `seq_len × vocab` logit tensor will.

**Disk:** `D:` 35 GB free of 200 G (the repo, `runs/`, tokenized `.npy`).
`C:` 74 GB free of 430 G (the HuggingFace cache, which is *not* on `D:`).

### scipy and sklearn are present

This matters because three places in the repo justified hand-rolled algorithms
with "no scipy/sklearn in this environment (checked)". That justification was
false here and is corrected (T-L0.0) in `code/more/metrics.py`,
`ARCHITECTURE.md` §T6.3 and `code/test_phase6_routing_metrics.py`. The exact
Hungarian assignment and the AMI chance correction **stay local** — they are exact
at `E <= 15`, checked against brute force, and keep the metric layer's imports to
torch/numpy. Only the stated reason changed.

---

## 3. The HuggingFace stack is already installed

`datasets 4.6.1`, `transformers 4.51.3`, `tokenizers 0.21.0`,
`huggingface_hub 0.30.2` and `pyarrow 23.0.1` are importable from **both**
interpreters. Phases L-2 (fetch WikiText, fit the byte-level BPE) and L-3 (POS
tagging is *not* from this stack — see below) therefore need **no installation**
and no new pin. Do not add a requirements entry for them as if they were new.

Only `tokenizers` is on the critical path: `T-L2.1` fits a byte-level BPE at
`V = 8192` on the **training split only**. `transformers` is present but the
migration does **not** use a pretrained tokenizer — GPT-2's `V = 50257` would put
a 12.9 M-parameter embedding table in front of a 3.2 M-parameter expert stack and
the study would be measuring the embedding, not the architecture.

The HF cache is at `C:\Users\vedan\.cache\huggingface` (no `HF_HOME` or
`HF_DATASETS_CACHE` is set) and already holds unrelated datasets and models, so
the hub has been reachable from this machine. **WikiText is not cached yet** —
`T-L2.0` is a real download.

### The POS tagger for T-L3.0: nltk is present, `universal_tagset` is not

`nltk 3.9.1` imports from **both** interpreters, and two of its data packages are
already downloaded to `C:\Users\vedan\AppData\Roaming\nltk_data`:

| resource | state |
|---|---|
| `taggers/averaged_perceptron_tagger` | **present** |
| `tokenizers/punkt` | **present** |
| `corpora/universal_tagset` | **MISSING** |

`spacy`, `stanza` and `flair` are all absent, and none should be added.

The consequence for `T-L3.0`: use `nltk.pos_tag`, which returns **Penn Treebank**
tags, together with a **PTB → six-family mapping checked into
`code/more/lang_families.py`**. Do *not* call `pos_tag(tagset="universal")` — that
needs `universal_tagset`, which is missing, so it would trigger a silent download
at dataset-build time and make the family definition depend on network state. An
explicit PTB map is also the better artifact: it is reviewable, it is the thing a
reader disagrees with, and an unmapped tag can be made to *raise* rather than fall
back to a catch-all (`CLAUDE.md` §2).

`averaged_perceptron_tagger` is a fixed pickled model, so it is deterministic given
the same nltk version and data. Pin **both** `nltk 3.9.1` and the tagger package
name in the dataset manifest (`T-L2.4`), because the family assignment for all 8192
vocabulary types is a *function of the tagger*, and a manifest that records the
family counts without recording what produced them cannot be reproduced.

Tagging is type-level and runs **once** over the 8192-entry vocabulary, cached in
`token_family[V]` — not per token, per epoch.

<!-- APPEND-MARKER-1 -->


---

## 4. How the CUDA venv was built (T-L0.1)

```bash
C:/Users/vedan/anaconda3/python.exe -m venv --system-site-packages D:/res/git/MoRE/.venv_cuda
D:/res/git/MoRE/.venv_cuda/Scripts/python.exe -m pip install \
    torch==2.6.0 --index-url https://download.pytorch.org/whl/cu126
```

`--system-site-packages` is deliberate: it keeps one copy of numpy/scipy/pandas/
wandb and guarantees the two interpreters cannot drift apart in anything except
torch. The venv's own `site-packages` shadows base only for torch and its CUDA
wheels.

**The install printed four dependency-conflict warnings** — `google-genai`,
`pymilvus`, `s3fs`, `streamlit`. All four are unrelated base packages that pin
loose ranges; none is imported anywhere in this repository. They were verified
harmless rather than assumed harmless.

**Base was not modified.** Confirmed after the install:
`base torch 2.6.0+cpu cuda False`. The venv is fully reversible — delete
`D:\res\git\MoRE\.venv_cuda` and nothing else changes. It is git-ignored.

CUDA was verified by running a real device matmul, not by trusting
`torch.cuda.is_available()`, which returns `True` on a driver that then fails to
launch a kernel.

---

## 5. Environment facts that changed a result

Two defects fixed in `T-L0.3` were **not** code bugs in the ordinary sense — they
were assumptions about the filesystem that happened to hold on the development
machine and not here. Both are recorded because the class recurs:

1. **`os.path.relpath` raises across Windows drives.** `run_context.py` recorded
   `run_dir` relative to the repo root. The repo is on `D:`;
   `tempfile.mkdtemp()` returns a `C:` path; gate 4 died with
   `ValueError: path is on mount 'C:', start on mount 'D:'` and no `[FAIL]`
   marker, so the runner could only report "output format not recognised". Fixed
   with a fallback to the absolute path.

2. **`git worktree add` rewrites every mtime.** `runs/` is tracked, so creating
   this worktree stamped all 132 run directories with checkout order (a 2125.8 s
   spread). `test_gate5.py` selected its single-expert run by newest `mtime` and
   therefore graded a run built **before** the fix it was testing, reporting a
   defect the current code does not have. Fixed by selecting on
   `provenance.code_git_commit == git_commit()`.

**The principle, worth carrying into the language phase: filesystem metadata is
not experiment metadata.** An mtime, a directory ordering, or a drive letter is a
property of the substrate. Using one as evidence is the same error class as
reporting a sentinel as a measurement — a value that looks like data but is an
artifact of where it was stored.

The suite total is **356** on this machine (350 at the close of the arithmetic
study, +6 from the two fixes above: +1 gate 4, +5 gate 5's new T6.7d producer).
`plan_language.md` §0 records the derivation.

---

## 6. Re-measuring this file

Nothing above is remembered; all of it is printed by this probe. Re-run it on any
new machine and update the table in §2 before trusting a gate count or a timing.

```python
import sys, platform
print("exe        ", sys.executable)
print("python     ", sys.version.split()[0], platform.machine())
for m in ("torch", "numpy", "scipy", "sklearn", "pandas", "matplotlib", "wandb",
          "datasets", "transformers", "tokenizers", "huggingface_hub", "pyarrow",
          "nltk", "spacy", "stanza"):
    try:
        print(f"{m:<16}", getattr(__import__(m), "__version__", "?"))
    except Exception as e:
        print(f"{m:<16} ABSENT ({type(e).__name__})")
import torch
print("torch.cuda ", torch.cuda.is_available(), torch.version.cuda)
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print("gpu        ", p.name, p.total_memory // 1048576, "MiB",
          f"cc {p.major}.{p.minor}")
```

`nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader`
supplies the driver version, which torch does not expose.

---

## 7. Shell

Git Bash on Windows 11 Home Single Language 10.0.26200. The working directory
persists between tool calls, so **absolute paths** are used everywhere
(`CLAUDE.md` §9).

Work happens in a git **worktree** at
`D:\res\git\MoRE\.claude\worktrees\english-language-dataset-migration-92946e` on
branch `claude/english-language-dataset-migration-92946e`. The stash stack is
shared with the main checkout: never bare `git stash` / `git stash pop`. Use a
temporary WIP commit to set work aside.





