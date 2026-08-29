# anon.md — anonymization and reproducibility work order

**Audience: the agent that executes this.** Every row below is a concrete edit with
a file, a line range, the replacement, and why. Do the rows in section order; §4
lists things that must NOT be touched and is the most important section in the file.

All line numbers verified against commit `08e9042` (`main`, clean tree). If a line
number does not match what you find, locate the string instead of trusting the
number, and correct this file.

## Scope and delivery mechanism

The submission is delivered through <https://anonymous.4open.science/>, which mirrors
the **file snapshot of one branch** through a read-only GitHub integration, serves it
as a web tree plus a `.zip`, and applies a caller-supplied list of terms as text
redaction. Three consequences drive everything below:

1. **Redaction is text-only.** Binary blobs are passed through verbatim. This is why
   `archive/` cannot be redacted — see below.
2. **Redaction breaks hashes.** Any file whose text 4open rewrites no longer hashes to
   its recorded value. This repo has two hash-verification stories
   (`data/dataset_meta.json` split SHA-256, and `config_hash` in `runs/`), so the
   redaction list must stay narrow. §6.
3. **Reviewers get a zip, not a git remote.** The README may not say `git clone`.

## Already decided — do not re-litigate

- **`archive/` is removed wholesale from the submission branch.** It is 216 MB and
  3,945 of 4,597 tracked files, and its `run-*.wandb` payloads carry
  `vedantkorade@gmail.com` and the W&B entity `vedantkorade-mumbai-university` inside
  **binary** files that neither `sed` nor 4open redaction can clean. It stays on
  `main`, so `CLAUDE.md` §6 "archive, never delete, evidence" is satisfied: nothing is
  destroyed, one branch simply does not carry it.
- **Delivery is an orphan `submission` branch** built in a throwaway clone, committed
  as `Anonymous Author <anonymous@anonymous.invalid>`, pushed alone (never `--all`,
  never `--mirror`, never `--tags`) to a private temp repo that 4open reads. `main`
  keeps the real 23-commit history, which leaks four author identities and six merge
  messages containing a GitHub handle — that history is never mirrored.
- **`runs/**/config.json` data paths stay exactly as they are.** §4, row DN-1.

Everything below is what remains **after** those decisions.

> **This file must not reach the submission branch.** §6 collects every real name,
> email, GitHub handle and institution slug into one list, which is exactly the
> artifact a reviewer must not be able to open. Delete `anon.md` in the throwaway
> clone before the orphan commit (step 9 of §7), and re-run §5's identity sweep
> afterwards — that sweep will fail while this file is present, which is the
> intended behaviour, not a false positive.


---

## 1. Identity removal

After `archive/` is gone, **exactly five tracked files** in the repository still match
any identity term. Verified:

```bash
git grep -l -iE "ayan|farooque|vedant|korade|godspeed|rshisode|aurangabad|mumbai-university|165833401|125894887" -- . ':!archive'
```
→ the five `automated/*.log` files, and nothing else. Rows ID-1 and ID-2 close them.
Everything from ID-3 on is the weaker `C:\Users\Hp` path fingerprint.

| # | Change | File | Lines | Replace / remove | Reasoning |
|---|---|---|---|---|---|
| ID-1 | Strip W&B banner lines | `automated/phase10_launch.log` (1448 L), `phase10_launch_resume.log` (723 L), `phase10_ponder_launch.log` (464 L), `phase10_seeds4546.log` (595 L), `phase10_fixed_depth_4546.log` (292 L) | every line matching `^wandb:` | Delete those lines **in place**. Do **not** delete the files. | These lines carry the W&B username `rshisode607`, the entity `rshisode607-government-college-of-engineering-aurangabad` (a **named institution**), the credential-file location `C:\Users\Hp\_netrc`, and **147 clickable public dashboard URLs** (`https://wandb.ai/rshisode607-.../micro-MoRE-poc/runs/<id>`) pointing at the actual Phase 10 ablation runs reported in README §4. A reviewer clicking one lands on the authors' account. The files must survive because `automated/phase10_code_state_proof.json:76` cites `phase10_fixed_depth_4546.log` as the provenance for the fixed-depth seeds 45–46 extension. Nothing traceable is lost: each run's `wandb_run_id` is still in `runs/<id>/resolved_config.json` under `provenance`. |
| ID-2 | Rewrite machine paths in the same five logs | same five files | 93 / 45 / 27 / 36 / 16 hits respectively (217 total) | `C:\Users\Hp\Desktop\Waste\MoRE` → `<REPO_ROOT>`; `C:\Users\Hp\anaconda3\envs\more_env` → `<PYTHON_ENV>` | Verified that after ID-1 removes the `wandb:` lines, `C:\Users` is the **only** remaining identity-adjacent string in these files — no identity term survives outside the banner lines. Do both passes, then re-run the ID verification grep. |
| ID-3 | Interpreter path in prose | `ARCHITECTURE.md` :1098, :1109, :1113, :1117, :1126 · `CLAUDE.md` :218 · `TASKS.md` :461 | 7 hits | Replace `C:\Users\Hp\anaconda3\envs\more_env\python.exe` with plain `python`; replace the `ARCHITECTURE.md` §9 code fence with a prose line naming Python 3.11+ and torch 2.5.1 from `requirements.txt`. Drop `C:\Python314` and the `Anaconda base` sentence — both are machine trivia. | `Hp` is an OEM default username, so alone it is weakly identifying — but the five logs put `C:\Users\Hp\Desktop\Waste\MoRE` and the `rshisode607` login on the *same machine*, which links the fingerprint to a named account. Even after ID-1, a reviewer reading a documented command that names someone's home directory reads it as a repo that was never prepared for blind review. |
| ID-4 | Interpreter path in test docstrings | `code/test_phase2_routing.py:6` · `test_phase3_halting.py:6` · `test_phase4_balance.py:6` · `test_phase5_dimensions.py:7` | 1 hit each | Replace the usage line with `python code/test_phaseN_<name>.py` | Docstring-only, so **no behaviour changes**: `run_correctness_suite.py:89` already invokes each suite as `[sys.executable, path]`, and `test_phase6_provenance.py:30` sets `PY = sys.executable`. The 350-check suite is portable today; only its documentation is not. |
| ID-5 | Dead absolute paths in exploratory drivers | `automated/rerun_mor.py:7` (5 hits) · `run_remaining_tests.py:10` (9) · `run_sweeps.py:9` (6) · `clean_csv.py:3` (1) | Replace each hardcoded constant with a repo-relative computation from `__file__` (snippet in §3). Keep all four files. | Deleting them is tempting but creates dangling references: `TASKS.md:71-73` cites all four as the evidence that exploratory drivers live in `automated/`, and `TASKS.md:1836` / `changelog.md:3646` specifically record that `run_sweeps.py` wrote the abolished `sweep_results.csv`. The repo runs its own stale-reference sweeps; breaking six citations to remove a path string is a bad trade. |
| ID-6 | Absolute data paths in generated ablation configs | `automated/phase10_configs/` — `_base_more.json`, `dense_routing.json`, `fixed_depth.json`, `ponder_cost_low.json`, `router_noise.json`, `routing_supervision.json`, `two_blocks.json` | :38, :39, :40, :46 in each (28 hits) | `C:\\Users\\Hp\\Desktop\\Waste\\MoRE\\data\\<split>.jsonl` → `../../data/<split>.jsonl` (also `jsonl_path` at :46 → `../../data/train.jsonl`) | Two problems in one: the path fingerprint, and the fact that **all seven ablations are unrunnable as committed**. `code/more/config.py:238-242` resolves relative paths against the config file's own directory, so `../../data/` from `automated/phase10_configs/` lands correctly. This is a stopgap — see RP-6, these files are generated. |
| ID-7 | `.claude/` is not ignored | `.gitignore` (append) | new lines | Add `.claude/` and `*.local.json` | `git check-ignore -v .claude` reports **NOT ignored**, and `.claude/settings.local.json` (untracked, 554 B) contains `Read(//c/Users/vedan/anaconda3/envs/**)`. The branch build runs `git add -A`, which would commit a username straight into the anonymized artifact. |
| ID-8 | W&B project slug | `code/config.json:50` · `automated/phase10_configs/*.json:50` · `runs/*/config.json` + `runs/*/resolved_config.json` (`logging.wandb_project`) | 1 hit per file | `"micro-MoRE-poc"` → `"more-anon"` | **Hash-safe by construction**: `run_context.config_hash` (`code/more/run_context.py:82-99`) strips exactly `provenance` and `logging` before hashing, so editing anything under `logging` cannot change a recorded `config_hash`. Worth doing because `micro-MoRE-poc` is a distinctive enough slug to be searchable on W&B and resolves to the authors' account. Pair it with setting the real W&B project to **private** for the review window — that is the actual fix; this is defence in depth. |
| ID-9 | Audit the provenance blocks | `runs/*/resolved_config.json` → `provenance` | n/a | Enumerate the key set (command in §5); scrub any hostname, OS username, or W&B URL found | Also free: `provenance` is stripped from the hash, so anything in it can be edited with zero effect on run identity. I enumerated the seeding/determinism keys but did not audit the full key set across all 129 directories — do that before the branch build. |

---

## 2. Reproducibility

The blocking facts: no dependency manifest exists anywhere in the live tree, and the
primary entrypoint cannot print its own `--help`.

| # | Change | File | Lines | Replace / remove | Reasoning |
|---|---|---|---|---|---|
| RP-1 | Add a pinned dependency manifest | `requirements.txt` (**new**, repo root) | whole file | Template + extraction command in §3 | There is no `requirements.txt`, `environment.yml`, `pyproject.toml` or `setup.py` anywhere outside `archive/`. `README.md:338` tells the reader `conda activate more_env` — an environment nobody else has, with no recipe to build it. **Do not invent version pins.** The real pins are in `archive/pre_finalization/wandb_runs_micropoc/*/files/requirements.txt` on `main`; extract them before the branch build. |
| RP-2 | Make the wandb import lazy | `code/more/engine.py` | remove :20 · insert helper after :35 · bind at :356 | Diff in §3 | `python train.py --help` dies with `ModuleNotFoundError: No module named 'wandb'`. The chain is `code/train.py:16` → `code/more/__init__.py:27` (`from .engine import train`, eager) → `code/more/engine.py:20` (`import wandb`, top level). It breaks `train.py`, the three launchers, `depth_null_model.py`, `eval_val_depth.py`, `smoke_test.py` and `audit_leakage.py` — and **five of those never log anything**. A reviewer without a W&B account cannot read the argument list. `train()` at `engine.py:51` is the only top-level function in the file, so one local binding covers `watch` (:363), `log` (:1038) and `finish` (:1275). |
| RP-3 | Add `--wandb_mode` | `code/more/cli.py` | :14 (`import os`) · after :113 (new argument) · after :118 (set env) | Diff in §3 | `wandb.init(...)` at `engine.py:356` is unconditional and there is no offline flag among cli.py's 14 arguments. As shipped, an independent researcher must create a W&B account before they can train at all. Setting `WANDB_MODE` via the environment rather than a config field means **no config schema change and no hash impact**. |
| RP-4 | Record the effective mode | `code/more/engine.py` | :370 area (the `provenance` update block) | Add `"wandb_mode": os.environ.get("WANDB_MODE", "online"),` | An offline run must never be mistakable for an online one — same discipline as `halting_mode` and `routing_supervision_enabled`. `os` is already imported at `engine.py:7`, and `provenance` is stripped from the hash, so this is free. |
| RP-5 | Rewrite "Getting started" | `README.md` | :333–:382 | Full replacement in §3 | Four defects. (a) It begins at `conda activate more_env` — no download step, no Python version, no `pip install`. (b) It must say *download and unpack the archive*, not `git clone`: 4open serves a zip and a web tree, not a git remote. (c) It must offer `--wandb_mode offline`. (d) It must disclose the `config_hash` limitation in DN-1/RP-7 — a reviewer who discovers undisclosed non-reproducibility marks you down; one who reads a precise statement of a known limitation does not. |
| RP-6 | Regenerate the ablation configs | `automated/phase10_ablations.py:223` (`materialized_base()`) | n/a | After ID-6, regenerate and confirm the seven files come back with relative paths | `_base_more.json:55` says `GENERATED by automated/phase10_ablations.py -- do not hand-edit`, and `materialized_base()` builds them by calling `load_config(CODE/"config.json")` — which absolutizes paths at `config.py:238`. So ID-6's hand edit is **lost on the next regeneration**, and regenerating on a different machine substitutes *that* machine's absolute path. ID-6 is the stopgap for the submission; RP-7 is the durable fix. Note the trade-off: a re-launched ablation now hashes differently from the recorded one. Record that in `changelog.md` (CQ-4) rather than leaving it implicit. |
| RP-7 | **DEFERRED** — machine-independent `config_hash` | `code/more/config.py` :237–:243, plus every `cfg["data"][...path]` read site in `engine.py` / `data.py`, plus a `provenance.config_hash_scheme` stamp in `run_context.py` | see §4 note | Stop writing absolutized paths into the hashed config; resolve at the point of use | **Do not apply before submission.** `config_hash` covers the `data` block, and `config.py:238-242` absolutizes the four path keys against the config file's directory — so the hash depends on where the clone lives. A reviewer running the documented command in a fresh clone gets a different hash, a different run-directory suffix, and `export_results.py` refuses to pool their run with the recorded ones. Fixing it changes every future hash, and the historical hashes cannot be recomputed without falsifying the record, so the fix inherently needs a versioned scheme. It also changes what `run_correctness_suite.py`'s provenance test (which performs a real CLI run) observes — the 350-check suite must be re-run and re-read. Disclose now (RP-5), migrate after. |
| RP-8 | Refresh the stale results artifacts | `results/results.json` · `results/results_tables.md` | whole files | Run `python export_results.py` from `code/` once and commit the output | The committed artifacts predate two tracked run directories (`runs/t67_provenance_check_seed44__59000a77__r12` and `__r13`), so regenerating moves the refusal ledger 114 → 116. The only other drift is float repr (e.g. `0.0002138247110035728` → `…57282`). **Headline numbers do not change.** This must land before RP-5, because the new README states that the exporter leaves `git status` clean — which is currently false. |
| RP-9 | Add a licence | `LICENSE` (**new**, repo root) | whole file | MIT or Apache-2.0 with `Copyright (c) 2026 Anonymous Author` | No licence exists. An artifact with no licence is legally ambiguous and some venues reject it; a licence naming a real copyright holder defeats the anonymisation. |

---

## 3. Code quality and presentation

| # | Change | File | Lines | Replace / remove | Reasoning |
|---|---|---|---|---|---|
| CQ-1 | Delete the 7-expert scratch configs | `code/temp_config_mor.json` · `code/temp_config_sweep.json` | whole files | **Delete both** | Both declare `"num_experts": 7` — the E7 catch-all that `CLAUDE.md` §2 says is *removed* and that an unmapped operation must *raise* on. `temp_config_sweep.json` also carries `d_model: 512`. A reviewer who greps `num_experts` finds a `7` in a tracked file and concludes the stated architectural constraint is not actually enforced. They are runtime scratch written by the three drivers' `TEMP_CONFIG_PATH` and are cited **nowhere** in `TASKS.md` or `changelog.md`, so deletion creates no dangling reference and the drivers regenerate them. |
| CQ-2 | Defuse the Ollama probe | `code/verify_pipeline.py` :2, :4, :19, :110, :233–:323, :641–:642, :645, :670 | Retitle the module docstring; drop the `autoresearch_runner.py` framing at :4 and :89; default the probe to skipped. **Do not delete the file.** | A "MoRE Pipeline Pre-flight Checker" that probes an **Ollama LLM server at `http://localhost:11434` for VRAM offload** — in a repository with no LLM component — is the single most confusing artifact a reviewer will open. But the file is cited as gate evidence (`verify_pipeline.py 35/35`) at `TASKS.md` :273, :381, :548, :817, :839, :845, :1730 and `changelog.md` :54, :241, :440, :682, :827, :1078, :3391, :3624 — deleting it orphans ~15 evidence references. **Verify the check count is still 35 afterwards**; if the count moves, revert to a docstring-only edit, because every one of those citations quotes the number. |
| CQ-3 | Label the autoresearch driver | `automated/autoresearch_runner.py` :1–:10 (header), :52 | Add two header lines: retired exploratory tooling, used for no reported result, never launches canonical runs | `OLLAMA_URL = "http://localhost:11434/api/generate"` is localhost and not identifying, and `TASKS.md:72` cites the file — so keep it. The issue is purely that an LLM-driven research supervisor sitting at the top of `automated/` reads like part of the method unless labelled. |
| CQ-4 | Append a changelog entry | `changelog.md` (append) | new entry | One entry for this pass | `CLAUDE.md`: append an entry whenever a task completes, naming the code location to open when the area misbehaves. It must record three things a future reader will otherwise re-derive painfully: that `logging` and `provenance` are outside `config_hash` (which is what made ID-8/ID-9/RP-4 free), that RP-7 was **deliberately deferred** and why, and that ID-6's relative paths mean a re-launched Phase 10 arm no longer hashes to its recorded value. |
| CQ-5 | Ignore the submission clone | `.gitignore` (append) | new line | `/MoRE-submission/` | The throwaway clone is built beside the repo; without this it can be committed to `main`. |

### Snippet for ID-5 (all four drivers)

```python
# Repo-relative: the old constant named one machine's Desktop and was dead on
# every other. __file__ is in automated/, so parents[1] is the repo root.
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
TEMP_CONFIG_PATH = str(REPO / "code" / "temp_config_mor.json")   # or _sweep
```

`automated/clean_csv.py:3` gets the same treatment (`REPO / "sweep_results.csv"`), plus a
one-line header noting it is a retired utility for an output filename that
`.gitignore:47` now abolishes.

### Diff for RP-2 (`code/more/engine.py`)

```diff
@@ -17,7 +17,6 @@
 from torch.utils.data import Dataset, DataLoader, random_split
 import numpy as np
-import wandb
 import matplotlib
```

Insert after the `from .metrics import (...)` block that ends at line 35:

```python
def _wandb():
    """
    Import wandb only when a run is actually about to log.

    It used to be a module-level import, which made wandb a hard dependency of
    the WHOLE package via `from .engine import train` in __init__.py:27. The
    consequence: `python train.py --help`, depth_null_model.py, eval_val_depth.py,
    smoke_test.py and audit_leakage.py all died on ModuleNotFoundError -- and
    four of those five never log anything. A reviewer with no W&B account could
    not so much as read the argument list.
    """
    try:
        import wandb
        return wandb
    except ImportError as e:
        raise ImportError(
            "wandb is required for training runs: it carries the provenance "
            "record required by updated_rules.md 9. Install it with "
            "`pip install wandb`, or pass --wandb_mode offline to write that "
            "record to disk with no account and no network. Analysis-only "
            "entrypoints (export_results, depth_null_model, eval_val_depth, "
            "run_correctness_suite) do not need it at all."
        ) from e
```

```diff
@@ -353,6 +353,9 @@
         "total_params": total_params,
     })
+    # Bound here, not at module scope, so the package imports without wandb
+    # installed. train() is the only top-level function in this file, so this
+    # one name covers .watch (363), .log (1038) and .finish (1275).
+    wandb = _wandb()
     wandb.init(
         project=log["wandb_project"],
```

RP-4, in the same file:

```diff
@@ -368,6 +371,7 @@
         "resolved_batch_size":      batch_sz,
         "wandb_run_id":             getattr(wandb.run, "id", None),
+        "wandb_mode":               os.environ.get("WANDB_MODE", "online"),
         "device":                   str(device),
```

### Diff for RP-3 (`code/more/cli.py`)

```diff
@@ -12,6 +12,7 @@
 from __future__ import annotations
 
 import argparse
+import os
 import sys
```

```diff
@@ -111,6 +112,16 @@
              "(default: 'exploratory') marks the run non-canonical so the "
              "exporter excludes it.",
     )
+    p.add_argument(
+        "--wandb_mode", choices=("online", "offline", "disabled"), default=None,
+        help="Sets WANDB_MODE for this run. 'offline' writes the full "
+             "provenance record to disk with no account and no network, which "
+             "is what an independent reproducer needs; 'disabled' skips "
+             "telemetry entirely. Default follows the ambient WANDB_MODE. The "
+             "effective value is recorded as provenance.wandb_mode -- which "
+             "sits in the block config_hash strips, so choosing offline does "
+             "NOT change the run's config identity.",
+    )
     return p
```

```diff
@@ -117,6 +128,11 @@
 def main(argv=None, default_architecture: str | None = None) -> int:
     args = build_parser(default_architecture).parse_args(argv)
 
+    # Before anything imports wandb. wandb reads WANDB_MODE at init time, so
+    # setting it here is sufficient: no config field is introduced, no schema
+    # changes, and config_hash is untouched.
+    if args.wandb_mode is not None:
+        os.environ["WANDB_MODE"] = args.wandb_mode
+
     ctx = None
```

### Template for RP-1 (`requirements.txt`)

Fill the pins from `main` **before** the branch build removes `archive/`:

```bash
cd /d/res/git/MoRE && cat archive/pre_finalization/wandb_runs_micropoc/*/files/requirements.txt 2>/dev/null | sort -u | grep -iE "^(torch|numpy|scipy|pandas|scikit|matplotlib|wandb|tqdm)"
```

```
# requirements.txt
# The versions that produced the runs in results/ and runs/, taken from the
# W&B environment snapshot of those runs.
#
# torch: install the CUDA build matching your driver from pytorch.org first if
# you want GPU training. CPU-only torch is sufficient for
# run_correctness_suite.py, export_results.py, depth_null_model.py and
# eval_val_depth.py.
torch==2.5.1
numpy==<fill>
scipy==<fill>
pandas==<fill>
scikit-learn==<fill>
matplotlib==<fill>
wandb==<fill>
```

Drop `requests` unless something still imports it after CQ-2 and CQ-3 — `verify_pipeline.py:110`
and `autoresearch_runner.py` were its only consumers.

### Replacement for RP-5 — `README.md` lines 333–382

`````markdown
## 6. Getting started

Python 3.11 or newer. A CUDA GPU is needed for training; everything else — the
correctness suite, the exporter, the depth null model, the ablation report — runs
on CPU.

Download the repository archive from the anonymous link and unpack it, then:

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**No data download is required.** `more6-v1` ships as three frozen files under
`data/`. `data/dataset_meta.json` records their SHA-256 and line counts, so you can
confirm you have the exact splits the reported runs used:

```bash
python -c "import hashlib,json,pathlib; m=json.load(open('data/dataset_meta.json'))['sha256']; print({k:(hashlib.sha256(pathlib.Path(f'data/{k}.jsonl').read_bytes()).hexdigest()==v) for k,v in m.items()})"
```

All three must print `True`. To regenerate the dataset from scratch instead:
`python data/script.py --out_dir data`.

Verify the implementation against the five architectural gates — 350 checks, no GPU
and no W&B account needed. This is the one check worth quoting:

```bash
cd code && python run_correctness_suite.py
```

Expected: `TOTAL 350 350 0 0` and `CORRECTNESS SUITE: ALL GATES PASS`, in about
2½ minutes.

Train one architecture at one seed. `--wandb_mode offline` writes the full
provenance record to disk with no account and no network:

```bash
cd code && python train.py --architecture more --seed 42 --run_name phaseB_more --wandb_mode offline
```

`--config` defaults to `config.json`. Output goes to `runs/<experiment_id>/`. No
global `best_model.pt`, `results.tsv` or `*_results.csv` is ever written;
`code/more/run_context.py` enforces that.
`````

…continued:

`````markdown
**A known limitation of run identity.** The directory name ends in the first 8 hex
digits of `config_hash`, and `config_hash` covers the `data` block — including the
dataset paths, which `load_config` resolves to absolute against the config file's
directory. Your clone therefore produces a **different hash** from the one recorded
here, and `export_results.py` will refuse to pool your run with ours: it admits runs
on provenance identity, and a differing hash is exactly what it is built to catch.
Every scientific field is unaffected — compare `runs/<yours>/metrics.json` against
the recorded run directly. Making the hash machine-independent requires a versioned
hash scheme and is deliberately deferred rather than retrofitted onto the existing
record; see `changelog.md`.

Regenerate every derived results artifact:

```bash
cd code && python export_results.py
```

The exporter admits runs on **provenance fields only**, never on directory names. It
currently admits 15 of 129 directories and records a reason for each refusal — a
refusal is the tool working, not an error to route around. On an unmodified tree this
command leaves `git status` clean.

Reproduce the depth null model, the offline held-out depth metrics, and the ablation
matrix:

```bash
cd code && python depth_null_model.py
```

```bash
cd code && python eval_val_depth.py --glob 'phaseB_*'
```

```bash
cd code && python ../automated/phase10_ablations.py --report-only
```
`````

The `git status` sentence is **only true after RP-8 lands**. Do RP-8 first or delete
the sentence.

---

## 4. DO NOT CHANGE — read before touching anything

| # | Do not touch | Why |
|---|---|---|
| DN-1 | The four `data.*_path` values inside the **262** `runs/*/config.json` and `runs/*/resolved_config.json` files | `config_hash` (`code/more/run_context.py:82-99`) hashes the config with only `provenance` and `logging` stripped, so the dataset path strings are **inside** the hash. Verified empirically on `runs/phaseB_more_seed42__989e89cc/resolved_config.json`: the recorded hash `989e89cc9efc…ee18004` recomputes to itself today (chain intact), and rewriting the paths to `../data/` yields `40aa75c0b807…23a4d66`. `export_results.py` refuses to mix config hashes, so a blanket `sed` over `runs/` drops **all 15 admitted canonical runs** and empties the headline table. The `C:\Users\Hp` strings there are disclosed in prose (RP-5), not edited. |
| DN-2 | `data/train.jsonl`, `data/val.jsonl`, `data/test.jsonl` | Their SHA-256 in `data/dataset_meta.json` must keep verifying — `d9ecb9a6…0324e5` / `189dd7a8…fa0a0a` / `2a88633f…3e41d1`, at 59500 / 5250 / 5250 lines. I recomputed all three: they match. This is the repo's strongest reproducibility asset; do not let a redaction pass or a line-ending conversion near it. |
| DN-3 | Any hashed config field in `code/config.json`, `code/canonical_spec.json`, or the `model` / `training` / `loss_weights` / `data` blocks anywhere | Changing one invalidates the Gate 0 proxy guard for every canonical run. `config.json`'s own `_README` says it: change a value there and `canonical_spec.json` must change in the same commit. Nothing in this work order requires it. |
| DN-4 | `archive/` **on `main`** | `CLAUDE.md` §6: archive, never delete, evidence. It is omitted from the submission branch only. Deleting it from `main` destroys the pre-audit evidence and the only surviving record of the environment pins RP-1 needs. |
| DN-5 | `code/more/seeding.py` | Audited and passing. It seeds `random`, `numpy`, torch CPU + `cuda.manual_seed_all`, a purpose-offset DataLoader generator, a picklable module-level `seed_worker`, `PYTHONHASHSEED`, `CUBLAS_WORKSPACE_CONFIG`, both cuDNN flags and `use_deterministic_algorithms`; every field in its report is read back from torch rather than echoing the request, and runtime non-determinism warnings are recorded into `provenance.determinism_exceptions` instead of being suppressed. It is better than most published artifacts. Leave it alone. |
| DN-6 | `code/more/config.py:237-243` (**for this submission**) | This is RP-7. It is the right fix and it must not ship in the submission branch: it changes every future `config_hash`, needs a versioned scheme so historical runs are not falsified, and alters what `run_correctness_suite.py`'s provenance test observes. File it as a follow-up task in `TASKS.md` with the reasoning, and disclose the limitation in the README instead. |
| DN-7 | Git history on `main` | Do not run `filter-branch` / `filter-repo`. It would have to rewrite four author identities, both committer fields, and six merge-commit messages containing `godspeed-003` (`231d9cc`, `2642a92`, `325197f`, `8a0d643`, `1c6b3c8`, `4240394`) across two branches — and would still leave the binary W&B blobs in the object store. The orphan-branch approach sidesteps all of it and leaves the real history intact for your own use. |
| DN-8 | Anything under `results/` other than RP-8's two regenerated files | `results/bench_capacity.log` names the GPU model (`NVIDIA GeForce RTX 4060 Laptop GPU`) — that is hardware, not identity, and it is load-bearing for the capacity analysis. `results/results.csv` and `results_aggregate.csv` are clean of paths; I checked. |

**Secrets: verified clean, no action needed.** `git log --all -p -S` finds no
`WANDB_API_KEY`, `HF_TOKEN`, `OPENAI_API_KEY`, `AWS_*`, `ghp_`, `sk-` or `hf_` string
on any ref, and `--diff-filter=A` across `--all` shows no `_netrc`, `.env`,
`credentials`, `*.key` or `*.pem` file was ever added. The 40-hex strings in the tree
are git commit SHAs and dataset digests, not API keys. Do not "fix" them.

---

## 5. Verification gauntlet

Run all of these on the `submission` branch before generating the 4open link. Each has
a stated expected result; a deviation is a stop condition, not a note.

Identity sweep — must print `CLEAN`:

```bash
git grep -n -iE "ayan|farooque|vedant|korade|godspeed|rshisode|aurangabad|mumbai-university|165833401|125894887|wandb\.ai/[a-z0-9]|@(gmail|outlook|yahoo)\." || echo "CLEAN"
```

Binary sweep — `git grep` without `-I` reports `Binary file … matches`; must print
nothing:

```bash
git grep -l -iE "vedant|ayan|rshisode|godspeed" | grep -i "^Binary" || echo "NO BINARY HITS"
```

Remaining machine paths — after ID-2..ID-6 the only hits may be under `runs/`
(DN-1). Anything else is unfinished work:

```bash
git grep -l -iE "C:[\\\\/]+Users|/home/[a-z]|/Users/[a-z]" -- . ':!runs'
```

History — exactly one commit, `Anonymous Author`, and exactly one branch:

```bash
git log --format='%an|%ae|%cn|%ce|%s' && git branch -a && git tag
```

Entrypoints — all must print `OK`:

```bash
cd code && for f in train.py train_moe.py train_mor.py train_more.py depth_null_model.py eval_val_depth.py smoke_test.py audit_leakage.py run_correctness_suite.py; do printf '%-26s ' "$f"; python "$f" --help >/dev/null 2>&1 && echo OK || echo FAIL; done
```

Gates — expect `TOTAL 350 350 0 0` and `CORRECTNESS SUITE: ALL GATES PASS`:

```bash
cd code && python run_correctness_suite.py
```

Provenance key audit for ID-9:

```bash
python -c "import json,glob,collections; k=collections.Counter(); [k.update(json.load(open(f)).get('provenance',{}).keys()) for f in glob.glob('runs/*/resolved_config.json')]; print(sorted(k))"
```

Clean tree after the documented regeneration (RP-8) — must print nothing:

```bash
cd code && python export_results.py >/dev/null && cd .. && git status --porcelain
```

**Then verify the artifact the reviewer actually receives.** This is the step that
catches redaction collateral damage and is the easiest one to skip:

```bash
cd /tmp && rm -rf anon && mkdir anon && cd anon && unzip -q ~/Downloads/*.zip && grep -rn -iE "vedant|ayan|farooque|korade|rshisode|godspeed|aurangabad|mumbai-university" . || echo "ZIP CLEAN"
```

```bash
cd /tmp/anon/* && python -c "import hashlib,json,pathlib; m=json.load(open('data/dataset_meta.json'))['sha256']; print({k:(hashlib.sha256(pathlib.Path(f'data/{k}.jsonl').read_bytes()).hexdigest()==v) for k,v in m.items()})"
```

All three splits must still print `True` in the downloaded zip. If any is `False`,
4open's redaction touched a dataset file and the term list in §6 must be narrowed.

---

## 6. The 4open.science redaction list

Paste in this order — longest first, so entity slugs are consumed before the bare
names inside them. Treat it as a safety net: after §1 and §5, nothing on this list
should still exist in the branch.

```
rshisode607-government-college-of-engineering-aurangabad
vedantkorade-mumbai-university
165833401+ayanfarooque@users.noreply.github.com
125894887+godspeed-003@users.noreply.github.com
vedantkorade@gmail.com
ayfarooque@gmail.com
Ayan Ahmad Farooque
ayanfarooque
rshisode607
godspeed-003
godspeed
Farooque
Korade
Vedant
Ayan
```

**Must NOT go on the list:**

| Term | Why not |
|---|---|
| `Hp` | An OEM default username, two characters, and it appears inside ordinary prose. Redacting it mangles the documentation. ID-2..ID-6 remove the *paths* instead. |
| `C:\Users\Hp\...` | Redaction rewrites served text, so the `runs/*/config.json` a reviewer downloads would no longer recompute to its recorded `config_hash` — breaking the verification on exactly the files that prove provenance. DN-1. |
| `MoRE`, `more`, `micro-MoRE-poc` | `MoRE`/`more` are the method and the package name. `micro-MoRE-poc` is handled properly by ID-8 (renamed in-file, hash-free) plus setting the real W&B project private for the review window — redacting it would leave visible masks across 262 run configs. |

---

## 7. Execution order

1. **RP-1** — extract the pins from `archive/` on `main` while it is still there.
2. **RP-2, RP-3, RP-4** — lazy wandb, `--wandb_mode`, mode in provenance. Then run the
   entrypoint loop and the 350-check suite from §5.
3. **CQ-1, CQ-2, CQ-3, ID-7, CQ-5** — deletions, defusing, gitignore.
4. **ID-3, ID-4, ID-5, ID-6** — path scrubs. Then **RP-6**: regenerate the ablation
   configs and confirm the relative paths survive.
5. **ID-1, ID-2** — the five logs. Re-run the identity sweep; it must print `CLEAN`.
6. **RP-8** — regenerate `results/`, commit.
7. **ID-8, ID-9** — hash-free `logging` / `provenance` scrubs.
8. **RP-5, ID-3 (docs), RP-9, CQ-4** — README, remaining prose, LICENSE, changelog.
9. Build the orphan `submission` branch in a throwaway clone; drop `archive/`, `.claude/`
   **and `anon.md` itself**; commit as `Anonymous Author <anonymous@anonymous.invalid>`;
   push **that branch only** to a private temp repo.
10. Full §5 gauntlet on the branch, then generate the link, then §5's zip checks on the
    **downloaded artifact**.
11. File **RP-7** in `TASKS.md` as a post-submission task with DN-6's reasoning.

Steps 1–8 are ordinary edits. Step 2 is the only one that can break the correctness
suite; if it does, stop and report the actual failing output rather than adjusting the
gate — `CLAUDE.md` §7.

