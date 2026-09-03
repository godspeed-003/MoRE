# TASKS_LANGUAGE.md — MoRE on English text

Derived from [plan_language.md](plan_language.md), constrained by
[updated_rules.md](updated_rules.md) and [updated_objective.md](updated_objective.md).
Operating rules: [CLAUDE.md](CLAUDE.md). The arithmetic ledger is
[TASKS.md](TASKS.md) and is **closed** — do not add language tasks there.

**Ticking rule.** A box may only be checked when its **Verify** command/criterion
has actually been executed and passed. Record the observed evidence inline. If a
gate fails: STOP, report, identify cause, fix, re-run the gate (`plan.md` §20).

**Legend** — `[x]` verified complete · `[~]` in progress · `[ ]` not started ·
`[!]` blocked · `[✗]` closed WITHOUT running it, with the narrowing decision
recorded inline. Never convert a `[✗]` to `[x]`.

**This file is the resume point.** A session that picks the work up with no
conversational context reads the first `[~]` or `[ ]` in phase order and continues
there. Anything a resumed session needs to know that is not in the code belongs in
the Evidence line of the task that produced it.

**Interpreter.** `C:\Users\vedan\anaconda3\python.exe` for CPU work.
`D:\res\git\MoRE\.venv_cuda\Scripts\python.exe` for CUDA work (see T-L0.1).
The `C:\Users\Hp\anaconda3\envs\more_env\python.exe` in CLAUDE.md §9 and
ARCHITECTURE.md §9 is from the machine the arithmetic POC was developed on and
does not exist here (T-L0.0).

---

## Phase L-0 — Environment  (`plan_language.md` §12)

- [x] **T-L0.0 Correct the documented environment.** CLAUDE.md §9 and
  ARCHITECTURE.md §9 name an interpreter path and a GPU that do not exist on this
  machine (`C:\Users\Hp\...\more_env`, RTX 4060 Laptop 8 GB). Actual: user
  `vedan`, `C:\Users\vedan\anaconda3\python.exe` (Python 3.12.7), **RTX 3050
  6 GB Laptop GPU**, driver 596.08. Also correct the stale in-source claim in
  `code/more/metrics.py` that there is "no scipy/sklearn in this environment" —
  sklearn 1.3.2 and scipy are both present. The hand-rolled exact Hungarian and
  AMI stay: they are correct, dependency-free, and replacing working exact code
  to use a library is churn, not a fix. Only the justifying comment is wrong.
  **Verify:** both docs name the real interpreter and GPU; the metrics.py comment
  states the real reason (exactness + no new dependency), not a false absence.
  **Evidence:** `CLAUDE.md` §9 and `ARCHITECTURE.md` §9 now carry both
  interpreters and the RTX 3050 (6143 MiB, cc 8.6, driver 596.08), and link
  `ENVIRONMENT.md`. The false-absence claim was corrected in three places —
  `code/more/metrics.py` (the comment above `_to_contingency`),
  `ARCHITECTURE.md` §T6.3, and `code/test_phase6_routing_metrics.py`'s docstring —
  each now stating the real reason: exact at `E <= 15`, checked against brute
  force, and the metric layer imports nothing beyond torch/numpy. Usage lines in
  `test_phase2_routing.py`, `test_phase3_halting.py`, `test_phase4_balance.py`,
  `test_phase5_dimensions.py` and `README.md` §6 repointed at the real
  interpreters; `README.md`'s two `350 checks` references corrected to 356 with
  the derivation. Gate L0 re-run after the edits: **`TOTAL 356 356 0 0`, 5 gates,
  ALL GATES PASS** (gate 1 19, gate 2 31, gate 3 27, gate 4 129, gate 5 150).
  Deliberately **not** changed: the three `automated/*.py` `PYTHON_EXE`
  constants — those drivers also write a global `results.tsv` /
  `final_run_metrics.json`, which `CLAUDE.md` §5 prohibits and `run_context.py`
  refuses, so each got a NOT-RUNNABLE banner instead of a working interpreter
  path. `changelog.md`, `TASKS.md`, `results/results.md` and
  `ARCHITECTURE.md`'s scale caveat keep their `more_env` / RTX 4060 references:
  those record what was true when the arithmetic runs were made, and that is
  accurate (`CLAUDE.md` §6). The scale caveat gained one clause saying so.

- [x] **T-L0.1 CUDA-capable torch in a dedicated venv.** Base anaconda has
  torch 2.6.0+**cpu**, so `torch.cuda.is_available()` is False and the whole
  matrix would run on CPU. `D:\res\git\MoRE\.venv_cuda` was created with
  `python -m venv --system-site-packages` so `datasets`/`transformers`/`wandb`/
  `nltk`/`sklearn` are inherited from base and only torch is reinstalled; then
  `pip install --ignore-installed torch==2.6.0 --index-url
  https://download.pytorch.org/whl/cu126`. `--ignore-installed` is required or
  pip treats base's `2.6.0+cpu` as already satisfying `torch==2.6.0`. cu126
  matches the 2.6.0 wheel line. A separate venv, not a mutation of base, so the
  change is reversible and cannot break the working CPU environment.
  **Verify:** `.venv_cuda\Scripts\python.exe -c "import torch;
  print(torch.__version__, torch.cuda.is_available(),
  torch.cuda.get_device_name(0))"` prints a `+cu126` version, `True`, and the
  RTX 3050. Also assert `import datasets, transformers, tokenizers, nltk` still
  resolve inside the venv (the `--system-site-packages` inheritance held).
  **Evidence:** `torch 2.6.0+cu126`, `cuda_available True`,
  `NVIDIA GeForce RTX 3050 6GB Laptop GPU`, 6143 MiB, compute capability (8, 6);
  a 512×512 `cuda` matmul executed (so kernels really load, not just
  `is_available()`). Inheritance held: datasets 4.6.1, transformers 4.51.3,
  tokenizers 0.21.0, nltk 3.9.1, sklearn 1.3.2, scipy 1.14.1, wandb 0.25.0,
  numpy 1.26.4, matplotlib 3.10.0. Base interpreter confirmed untouched:
  `base torch 2.6.0+cpu cuda False`. pip printed conflict warnings for
  `google-genai`/`pymilvus`/`s3fs`/`streamlit` — those are unrelated base
  packages the venv shadows, not repo dependencies.

- [x] **T-L0.2 Record the environment in the repo, not just in chat.** Add an
  `ENVIRONMENT.md` (or a section of ARCHITECTURE.md) giving both interpreters,
  their torch builds, the GPU and VRAM, and the one-line rule *CPU interpreter
  for correctness gates, CUDA interpreter for training*. A resumed session must
  not have to rediscover this. **Verify:** the file exists and a fresh reader can
  pick the right interpreter for a given task from it alone.
  **Evidence:** [ENVIRONMENT.md](ENVIRONMENT.md), seven sections. §1 is the
  which-interpreter rule with a runnable command for each. §2 is the measured
  version table for both, side by side, from the probe reproduced in §6 — nothing
  in it is remembered. §3 records two findings that change later phases: the
  **HuggingFace stack is already installed** in both (`datasets 4.6.1`,
  `transformers 4.51.3`, `tokenizers 0.21.0`, `huggingface_hub 0.30.2`,
  `pyarrow 23.0.1`), so L-2 needs no installation and no new pin; and **`nltk
  3.9.1` is present with `averaged_perceptron_tagger` and `punkt` already
  downloaded but `universal_tagset` MISSING**, which decides T-L3.0 in favour of
  Penn Treebank tags plus an explicit PTB→family map in the repo rather than
  `pos_tag(tagset="universal")`, whose missing corpus would trigger a network
  download at dataset-build time. §4 is how the venv was built and why base is
  untouched; §5 records the two T-L0.3 defects as one class — *filesystem
  metadata is not experiment metadata*; §7 the shell and worktree rules.
  Disk recorded because L-2 needs it: `D:` 35 GB free, `C:` 74 GB free (the HF
  cache is on `C:`, and WikiText is **not** cached yet — T-L2.0 is a real
  download). Gate L0 unaffected: `TOTAL 356 356 0 0`, ALL GATES PASS.

- [x] **T-L0.3 Baseline the arithmetic gate before touching anything.** Run the
  correctness suite on the untouched worktree and record the exact totals. This
  is the reference every later Gate L0 comparison is made against; without it
  "unchanged" is unfalsifiable. **Verify:** `run_correctness_suite.py` prints
  `CORRECTNESS SUITE: ALL GATES PASS` and its totals line is copied into Evidence
  verbatim.
  **Evidence:** the first run did **not** pass — `TOTAL 340 339 1 0`, with Gate 4
  crashed mid-suite and one Gate 5 failure. Both were environment-dependent
  defects in the suite itself, fixed in T-L0.3a and T-L0.3b. After those fixes:

  ```text
  gate   checks   pass   fail   skip     time  verdict
  1          19     19      0      0    24.2s  PASS
  2          31     31      0      0    13.7s  PASS
  3          27     27      0      0    12.9s  PASS
  4         129    129      0      0    31.1s  PASS
  5         150    150      0      0   170.8s  PASS
  TOTAL     356    356      0      0
  CORRECTNESS SUITE: ALL GATES PASS
  ```

  **The Gate L0 reference is therefore 356, not the 350 written in the
  arithmetic-era notes** (+9 in Gate 4 where the seeding suite now runs to
  completion instead of dying at check 51, +5 new provenance checks from
  T-L0.3b). `plan_language.md` §0/§9/§11 were corrected to 356. Interpreter:
  `C:\Users\vedan\anaconda3\python.exe`, ~4.2 min wall-clock.

- [x] **T-L0.3a Cross-drive `relpath` killed Gate 4.**
  `RunContext.create` built `provenance.run_dir` with
  `os.path.relpath(directory, _REPO_DIR)`, and on Windows `relpath` **raises**
  when the two paths are on different drives:
  `ValueError: path is on mount 'C:', start on mount 'D:'`.
  `test_phase6_seeding.py` passes a `tempfile.mkdtemp()` runs root, which lands on
  `C:\Users\...\Temp` while this repo is on `D:`, so `create` raised before
  writing anything. Gate 4 died at check 51 having printed no `[FAIL]` marker, so
  the runner could only report "output format not recognised". It passed on the
  machine the POC was developed on solely because repo and temp shared a drive
  there — a latent defect, not a new one. Fixed with a `_repo_relative()` helper
  ([code/more/run_context.py:130](code/more/run_context.py:130)) that keeps the
  relative form when one exists and falls back to the absolute path otherwise. A
  run directory outside the repo is legitimate (a temp-root test, a `runs/` on a
  data disk) and provenance should record where the run actually is; both
  consumers (`export_results.py`, `depth_null_model.py`) only echo the string.
  The two `pathlib.relative_to` sites were checked and cannot go cross-drive —
  both write into `ROOT`. **Verify:** Gate 4 passes.
  **Evidence:** `gate 4  129 checks  129 pass  0 fail  0 skip  21.1s  PASS`;
  the seeding suite went 51 pass + crash → **60 pass, rc=0**.

- [x] **T-L0.3b Gate 5 was grading pre-fix history, and reported a defect the
  code does not have.** G5.2b failed with `val/routing_accuracy`,
  `val/routing_hungarian_accuracy`, `val/routing_ami`, `val/routing_purity`,
  `val/routing_macro_recall` **absent** instead of the string `"N/A"` on a
  single-expert run. The writers are correct: a MoR run built at HEAD emits all
  eleven routing/diversity keys as `"N/A"` (verified directly on
  `runs/t_l03_mor_timing_seed44__fa9339bc`). The fault was the selection key.
  `test_gate5.py` picked `RUN_E1` as the newest single-expert directory **by
  `os.path.getmtime`** — but `runs/` is tracked (555 files), so `git worktree
  add` rewrote all 132 directory mtimes into checkout order (measured spread:
  01:54:39–02:30:05 on 2026-09-03, i.e. 35 min of checkout, unrelated to when the
  experiments ran). That put `t83_mor_headfix_seed42__dd11e6dc__r2` last — commit
  `e5b0f6df`, i.e. from **before** the T8.1 fix that made those keys
  present-and-`"N/A"`. An audit of all 18 single-expert runs showed 9 satisfying
  the contract (all five `phaseB_mor_seed4*` plus `t81b`/`t81c` at `30831ff6`)
  and 9 pre-fix; `t81_mor` (0 N/A) and `t81b_mor_seed42` (5 N/A) at the *same*
  commit are the before/after pair T8.1 was verified with, so the archive was
  behaving exactly as intended. **mtime is a property of the filesystem, not of
  the experiment** — any clone, worktree, archive restore or copy destroys it.
  Two fixes: (1) selection is now by `provenance.code_git_commit == git_commit()`
  with a `_pick()` helper that reports whether the chosen run came from HEAD, and
  the G5.2b contract block **SKIPs with the commit named** rather than grading a
  historical directory; (2) `test_phase6_provenance.py` now produces the
  single-expert run it needs (new **T6.7d** section), so the slot has a producer
  ahead of it in the same gate instead of falling back to history. T6.7d also
  closes a real blind spot: provenance had only ever been asserted on a `more`
  run, and E=1 / max_depth=1 is exactly where architecture-specific gaps hide
  (T8.3). Its five checks cover the full `updated_rules.md` §9 field list at E=1
  (extracted into `_required_fields()` so one list serves both architectures),
  MoR's `ffn_mult == 24`, and that a legitimately-zero `routing_balance` survives
  as `0.0` rather than being read as missing — the falsy-zero form of reporting a
  sentinel as a measurement. **Verify:** Gate 5 passes with the contract block
  actually running, not skipping. **Evidence:**
  `gate 5  150 checks  150 pass  0 fail  0 skip  PASS`; provenance 38 → **43
  pass**, `test_gate5.py` 38 pass/1 fail → **39 pass**; gate 5's header now
  prints which runs came from HEAD. Cost: one extra 1-epoch MoR run in the suite,
  ~49 s on CPU.

---

## Phase L-1 — Task axis  (`plan_language.md` §1)

- [x] **T-L1.0 `task` in the config layer.** Add `task = arithmetic | language`
  to `load_config_defaults` with default `"arithmetic"`, a `TASKS` tuple beside
  `ARCHITECTURES`, and an `apply_task(cfg, task)` that stamps the
  language-specific shape fields (`seq_len`, `vocab_size`, `n_heads`,
  `attention`) exactly as `apply_architecture` stamps the architecture fields.
  `apply_task` **raises** for `task == "language"` with
  `loss_weights.family_cls != 0.0` (§7.3: the family head is a detached probe on
  language and must not be weighted into the trunk). Default-arithmetic is the
  whole point: no existing config, test, or archived resolved config changes
  meaning. **Verify:** a config with no `task` key resolves byte-identically to
  today's resolved config (diff the two resolved dicts, not just spot fields).

  **THE BOX'S LITERAL WORDING IS NOT IMPLEMENTABLE, AND THAT IS THE FINDING.**
  "default `arithmetic`" read naturally means
  `cfg.setdefault("task", TASK_ARITHMETIC)` in `load_config_defaults` — and that
  single line fails this box's own Verify criterion. `config_hash` is SHA-256 over
  the resolved config with only `provenance` and `logging` removed
  ([code/more/run_context.py:82](code/more/run_context.py:82)), so injecting a new
  key — however inert at runtime — moves the identity of **every** run that
  resolves through it, including the fifteen published canonical Phase-B ones.
  The default is therefore the **ABSENCE of the key**: nothing writes `task`
  unless the operator asks for a non-default task, and
  [config.py:resolve_task](code/more/config.py:84) maps absence → `arithmetic`.
  `apply_task(cfg, "arithmetic")` **pops** the key rather than writing it, so an
  explicit `--task arithmetic` and no flag at all produce the same bytes.
  **Evidence:** `code/test_language_task_axis.py`, 72 checks, 0 failed.
  TL1.0c/d: the two resolutions are byte-identical (809 canonical-JSON bytes both,
  hash `846c6e36…`) and 2077 bytes both through the CLI (TL1.2f). TL1.0g: all
  **15** seeded `runs/phaseB_*` directories re-hash from their own stored
  `resolved_config.json` to the `config_hash` they recorded before this axis
  existed — 0 mismatched. TL1.0h: all 18 still match the hash8 suffix in their
  directory name. **A second finding, recorded because it looked like a
  regression and is not:** the 3 *unseeded* `phaseB_*__seedNA__*` runs do **not**
  re-derive under today's hash — they reproduce exactly under a `provenance`-only
  exclusion, i.e. they predate `logging` joining the exclusion set. TL1.0g2
  asserts that explanation rather than leaving it as a story; those three are
  non-citable under T6.1 anyway (no declared seed).

- [x] **T-L1.1 `task` in the variant string and run naming.** `resolve_variant`
  must not silently label a language run `canonical`. Extend it so the task
  appears in the variant when it is not arithmetic, and add
  `CANONICAL_RUN_PREFIX_LANGUAGE = "langB"` so run names are
  `langB_MoE_seed42` (§10) and can never be confused with `phaseB_*` in W&B, in
  `runs/`, or in an exporter table. **Verify:** an unmodified language config
  resolves to a variant naming the task; `canonical_run_name` for
  `(language, moe, 42)` returns `langB_MoE_seed42`.
  **A REAL PRE-EXISTING DEFECT, found by writing the test rather than by
  inspection.** `resolve_variant` compared `loss_weights.family_cls` against the
  arithmetic canonical `0.5` only. On language `0.0` is *required* (§7.3), so an
  unmodified canonical language config resolved to
  `variant = "no_family_supervision"` — a canonical run wearing an ablation label,
  in the exact field that exists to stop an ablation being read as canonical.
  Fixed with `CANONICAL_FAMILY_CLS_WEIGHT_BY_TASK`
  ([code/more/config.py:436](code/more/config.py:436)); the arithmetic behaviour is
  unchanged, and TL1.1f pins it by asserting arithmetic `family_cls = 0.0` still
  tags `no_family_supervision`.
  **Evidence:** TL1.1a–f — clean language → `"language"`, language + fixed depth →
  `"language+fixed_depth"`, clean arithmetic → `"canonical"` (unchanged for the 15
  published runs), and the word `canonical` is never emitted for a non-default
  task. TL1.1g–o — `langB_{MoE,MoR,MoRE}_seed42`, `phaseB_*` unchanged, the two
  prefixes cannot collide under a glob, and a `run_name` the operator chose (e.g. a
  `dense_routing_ablation` label that `enforce_routing_mode` reads) survives
  `apply_task` untouched: the repair replaces only the exact stamped
  `phaseB_<Arch>` default, character for character.

- [x] **T-L1.2 `--task`, `--seq_len`, `--vocab_size` on the CLI.** Add the flags
  in `cli.py` and place `apply_task` immediately after `apply_architecture` and
  before the override blocks, so an explicit `--family_cls` on a language run
  hits the same refuse-don't-ignore path as every other override rather than
  being quietly overwritten. **Verify:** `--task language --family_cls 0.5` exits
  non-zero with a refusal naming §7.3; `--task language` alone succeeds.
  **Evidence:** TL1.2a–q, driving `cli.main()` end to end with only
  `RunContext.create` and `train()` stubbed (the argument contract is fully
  resolved before either runs, and a real call would need the L-2 corpus and the
  L-4 model, and would deposit a `runs/` directory that a later reader could
  mistake for evidence). TL1.2a/b: `rc = 1`, `train()` never reached, `ValueError`
  naming §7.3 and the detached probe. TL1.2c–e: `--task language` alone succeeds
  and resolves to a complete language config named `langB_MoRE`. TL1.2f/g: no
  `--task` and `--task arithmetic` are byte-identical through the CLI (2077 bytes
  both) and neither carries a `task` key or a language shape field. TL1.2i–n: the
  shape flags are **refused on arithmetic rather than ignored** (accepting them
  would write a field into the resolved config, the `config_hash` and the
  provenance record while describing nothing the run did), `--seq_len` overrides
  the provisional 256 that Gate L5 owns, `--seq_len 0` is refused, and
  `--vocab_size` is bounded at the uint16 ceiling — 65536 accepted, 70000 refused,
  because the tokenized arrays are `uint16` and id 65536 would wrap to id 0, a
  valid id for a different token, corrupting the corpus with no error anywhere.
  TL1.2p/q: all three arms resolve with attention on, MoR included.

- [x] **T-L1.3 Task-selected canonical spec.** `run_context.load_canonical_spec`
  takes a default `path=CANONICAL_SPEC_PATH`; make the selection task-based so
  `language` reads `code/canonical_spec_language.json`. `code/canonical_spec.json`
  and its three other readers (`export_results.py`,
  `test_phase6_provenance.py`, `test_phase6_seeding.py`) are untouched.
  **Verify:** an arithmetic run still loads the arithmetic spec (assert on the
  loaded dict's identity fields); a language run loads the language spec; a
  language run declaring `experiment_group = canonical_phase_b` is **refused**
  (wrong group for the task).
  **THE REFUSAL WAS ALREADY IMPLICIT, AND THAT WAS NOT GOOD ENOUGH.** Because
  T-L1.1 makes `resolve_variant` return `"language"` and never `"canonical"`,
  check 2c of the existing guard already refused a language `canonical_phase_b`
  claim — but with the message *"variant='language' … this run has at least one
  ablation active"*, which is false and would send the reader hunting for an
  ablation that does not exist. The cross-task claim is now its own first-class
  check with its own message
  ([code/more/run_context.py:370](code/more/run_context.py:370)), and it fires
  **before** the field loop so it cannot be buried under a list of null-field
  complaints. Each task also gets its own group string —
  `canonical_phase_b` / `canonical_lang_b` — because `canonical_phase_b` is the
  exact string `export_results.py` filters the headline arithmetic table on, and a
  language row there would put a per-token cross-entropy in nats in a column of
  arithmetic MSEs. The symmetric case (an arithmetic run claiming
  `canonical_lang_b`) is refused too.
  **`_effective()` had to grow five keys** — `seq_len`, `vocab_size`, `n_heads`,
  `attention`, `tie_lm_head`. The guard iterates the *spec's* `enforced_fields`,
  so a field the spec lists but `_effective` does not produce reports as
  *"requires 8192, run has None"*: a real check failing for a fake reason. All
  five are read with `.get`, so on arithmetic they are `None` and nothing compares
  them — `canonical_spec.json` does not list them.
  **THE NEW SPEC HAS MORE NULLS THAN THIS BOX ASKED FOR, deliberately.** The box
  named eight (`epochs`, `batch_size`, `lr`, `weight_decay`, `seq_len`,
  `max_depth`, `primary_metric_floor`, `ffn_mult.mor`). Three more are null and the
  reasons are in `canonical_spec_language.json:_NULLS_NOTE`: `dropout`
  (0.1 was *inherited* on 70 k records; ~103 M tokens against ~3.2 M parameters
  underfits, so the inherited value points the wrong way), and
  `routing_balance_weight` + `halting_weight` — T4.2 chose `0.001` because it put
  the weighted balance term at `0.017×` the task loss, and a per-token
  cross-entropy near `ln(8192) = 9.0` nats at init is two orders of magnitude
  larger than arithmetic's `~0.06` MSE, so the same coefficient would put the
  balance term at `~1e-4 ×` the task loss — effectively **off**, in the arm whose
  central failure mode is router collapse; and an equally invisible ponder cost
  would silently turn adaptive depth into always-max depth. Every non-null value
  in the file is definitional or protocol, and `frozen_by` says which for each one.
  **Evidence:** TL1.3a–p. TL1.3a/b: `load_canonical_spec()` with no arguments
  still reads the arithmetic spec, whose identity fields (`run_name_template`,
  `dataset_version`, `family_cls_weight = 0.5`) are unchanged and which gained no
  key. TL1.3f: an unregistered task **raises** rather than falling back to another
  task's numbers. TL1.3h: every field the language spec enforces is visible to the
  guard (`unreadable = []`). TL1.3i–k: the cross-task claim is refused in both
  directions, naming the task and the right group. TL1.3l–n: a language run
  claiming its *own* group is refused for **unfrozen fields**, not for its group,
  and the message names `canonical_spec_language.json`. TL1.3o: an exploratory
  label is still free. TL1.3p: the arithmetic guard still refuses router noise —
  `test_phase5_dimensions.py:G4.37`'s exact case, re-checked here because that
  gate's message assertions are what the rewrite could most easily have broken.

- [x] **T-L1.4 GATE L0 after the task axis.** The task axis is the change most
  likely to silently alter arithmetic behaviour, so the gate runs here before any
  language code exists. **Verify:** `run_correctness_suite.py` still prints
  `TOTAL 356 356 0 0` (the T-L0.3 baseline); arithmetic parameter counts for all
  three arms are
  unchanged from T-L0.3; a 1-epoch arithmetic run's `metrics.json` matches the
  pre-change run on `task_loss` to full printed precision at the same seed.
  **Evidence — all three criteria, in order.**

  *(1) The suite.* `C:\Users\vedan\anaconda3\python.exe code/run_correctness_suite.py`:

  ```text
  gate   checks   pass   fail   skip     time  verdict
  1          19     19      0      0     5.1s  PASS
  2          31     31      0      0     6.7s  PASS
  3          27     27      0      0     8.2s  PASS
  4         129    129      0      0    17.7s  PASS
  5         150    150      0      0   109.1s  PASS
  TOTAL     356    356      0      0
  CORRECTNESS SUITE: ALL GATES PASS
  ```

  Plus `test_language_task_axis.py`: **72 passed, 0 failed** (39 → 72 this
  phase). The language suite is deliberately *not* in `run_correctness_suite.py`
  yet — Gate L0's job is to prove the arithmetic number has not moved, and adding
  checks to it would make "356" stop meaning what T-L0.3 froze. It joins the
  runner at T-L8.x when the language gates are the thing being defended.

  *(2) Parameter counts.* Compared by extracting the T-L0.3 tree with
  `git archive 112809a code` into a scratch dir and building all three arms from a
  resolved config in both trees — same interpreter, same `config.json`, model
  constructed exactly as `engine.py:202-229` does. Compared as JSON including a
  **per-tensor `numel` map**, not just the total, because a compensating pair of
  shape changes would leave the total intact:

  | arm | total params | tensors | E | max_depth | ffn_mult |
  |---|---|---|---|---|---|
  | moe  | 3,201,555 | 54 | 6 | 1 | 4 |
  | mor  | 3,197,710 | 24 | 1 | 7 | 24 |
  | more | 3,201,555 | 54 | 6 | 7 | 4 |

  `diff` of the two JSON dumps: **empty**. (The MoR/MoRE gap is 3,845 params =
  0.12%, the T4.x parameter-matching result, unchanged.) Note `engine.py:220`'s
  comment cites `6,358,553` — that is the *2-block* figure from before T8.2 set
  `num_blocks = 1` as the default, and is not the T-L0.3 baseline; the baseline is
  whatever the T-L0.3 tree produces, which is what was measured here.

  *(3) The 1-epoch run.* `--architecture mor --epochs 1 --seed 44` against the
  pre-change reference `runs/t_l03_mor_timing_seed44__fa9339bc` (built at
  `08e90427`, i.e. before the task axis), same CPU interpreter, W&B offline.
  Result stronger than the box asks for:

  - **`config_hash` identical to all 64 hex digits** —
    `fa9339bc279495e8b5bb8c604833526d90adf3a6a0808319d0cf927ffed90f73`, which is
    why the re-run auto-named itself `…__fa9339bc__r3`. The resolved config with
    `provenance` removed is byte-identical, **2014 bytes both**, and `task` is
    absent from both. This is the absence-means-arithmetic decision of T-L1.0
    demonstrated on a real run rather than on a unit test.
  - **All 93 comparable `metrics.json` keys bit-identical**, not just
    `task_loss`: `train/task_loss = 0.09949329204093187`,
    `val/task_loss = best_val_loss = 0.07614141606433052`,
    `train/halting_loss = 0.27312992666012204`,
    `train/step_routing_loss = 0.5629466978403238`,
    `train/family_cls_loss = 0.05077054923906242`,
    `train/total_loss = 0.12512240410806277`. Excluded from the comparison:
    `experiment_id` (contains the `__r3` suffix by construction),
    `perf/throughput_tokens_sec` (wall-clock, 1940 → 1599 tok/s, a machine-load
    measurement and not a property of the model), and
    `_non_scalar_keys_omitted`. `nondeterministic_ops_observed = ['none']` in
    both.
  - `[Model] Total parameters: 3,197,710` printed by the run itself, agreeing
    with criterion (2)'s offline count.

  **`wandb.init` aborted the first attempt** (`UsageError: No API key
  configured`) after `RunContext.create` had already made the directory, leaving
  `runs/t_l03_mor_timing_seed44__fa9339bc__r2` holding a `config.json`,
  `resolved_config.json` and a `stdout.log` with the traceback, but no
  `metrics.json`. **It is kept, not deleted** — §T-LX.2 and CLAUDE.md §6 say
  archive rather than delete, and a directory with no `metrics.json` cannot be
  mistaken for a result. Its `resolved_config.json` is in fact the artifact that
  proves the hash claim above; the `stdout.log` is on-disk only, excluded by the
  standing `runs/**/stdout.log` rule, which is why the one line that matters is
  quoted here instead of cited. Re-run with `WANDB_MODE=offline` and
  `WANDB_DIR` pointed outside the repo, so no `wandb/` directory landed in the
  tree. Worth knowing before Phase L-7: **a canonical language run will hit this
  same wall**, and the run directory it leaves behind will look like a started
  run. W&B credentials need to be settled before the matrix, not during it.

---

## Phase L-2 — Data pipeline  (`plan_language.md` §3)

- [x] **T-L2.0 Fetch WikiText.** `data/lang/build_language_dataset.py` downloads
  `wikitext-103-raw-v1` (canonical) and `wikitext-2-raw-v1` (dev loop) via
  `datasets`. Raw variants only: the non-raw ones are pre-tokenized with `<unk>`
  substituted, which would put an artificial high-frequency type in the middle of
  the frequency distribution the depth correlations are measured against. The
  author-provided splits are **document-disjoint**, which is what makes the
  leakage audit meaningful. **Verify:** the three splits load, their line counts
  and byte sizes are recorded in the manifest, and no split contains the literal
  token `<unk>`.
  **Evidence:** both corpora fetched at HF revision `b08601e04326…` of
  `Salesforce/wikitext`, pinned in each manifest. Verified by
  `code/test_language_data.py` (TL2.0a–m on both corpora):

  | corpus | split | lines | nonempty | bytes | `<unk>` |
  |---|---|---|---|---|---|
  | wikitext-2 | train | 36,718 | 23,767 | 10,914,845 | **0** |
  | wikitext-2 | val | 3,760 | 2,461 | 1,144,248 | **0** |
  | wikitext-2 | test | 4,358 | 2,891 | 1,287,656 | **0** |
  | wikitext-103 | train | 1,801,350 | 1,165,029 | 539,295,549 | **0** |
  | wikitext-103 | val | 3,760 | 2,461 | 1,144,248 | **0** |
  | wikitext-103 | test | 4,358 | 2,891 | 1,287,656 | **0** |

  `<unk>` is **counted**, not asserted away: `raw_stats()` accumulates
  `text.count("<unk>")` per split so the number reaches the manifest whether it is
  zero or not. Zero everywhere confirms the `-raw-` choice empirically rather than
  on the strength of the dataset card. Train/val byte ratio 9.5× (wikitext-2),
  471.3× (wikitext-103).

  **FINDING — the two corpora share byte-identical val and test splits.** Not
  merely equal in size: hashing the raw row text of both corpora in memory gives
  `val sha256 = 8ef749789ca0693435d20b3f81d5638c19edcebc5a68586dcf09bdf47ef9542f`
  and `test sha256 = bbf94c53a05abe9ee670d3b6343608095822c85e26de37c70b24fc571964574a`
  for **both** wikitext-2 and wikitext-103 (3,760 / 4,358 rows each). Verified with
  a scratch script that reads through `fetch_splits()` and writes nothing into
  `data/lang/`, deliberately: materializing a `val_text.txt` there would break
  TL2.1e, which proves the tokenizer never saw held-out text *by the absence of
  such a file*.

  Consequence, and it is a methodological constraint on every later phase:
  **wikitext-2 is a correctness corpus, never a model-selection corpus.** Any
  hyperparameter, early-stopping point or architecture choice made by looking at
  wikitext-2's val loss has been made by looking at wikitext-103's val set,
  because they are the same 1,144,248 bytes. So dev-loop runs on wikitext-2 may
  answer "does it run, are the shapes right, is the loss finite", and may never be
  cited as an independent replication of a canonical result nor used to pick
  anything that is then reported on the canonical corpus. `plan_language.md` §3
  called wikitext-2 the "dev loop" without stating this; it is stated here.

- [x] **T-L2.1 Byte-level BPE, V=8192, fitted on train only.** Two independent
  reasons, both recorded in the script's docstring: (a) fitting a tokenizer on
  val/test is tokenizer-level leakage — the merge table would encode the held-out
  text's statistics; (b) a stock GPT-2 vocabulary would put 50257×256 = 12.9 M
  parameters in the embedding against a 3.2 M expert stack, so the §6 parameter
  budget would be satisfied by the embedding alone and the MoR/MoRE match would
  be measuring nothing. Byte-level so there is no OOV and no `<unk>`.
  **Verify:** the tokenizer round-trips a held-out paragraph exactly; the fitted
  vocab size is exactly 8192; the training corpus passed to the trainer is the
  train split alone (assert on the file list, not on intent).
  **Evidence:** V = **8192** exactly on both corpora, `eot_id = 0`, held-out
  round-trip exact on an 8,700-byte **val** paragraph (the builder *raises* if it
  is not, so a build cannot complete with an inexact tokenizer). Fit time 18.4 s
  (wikitext-2, 630 documents) and 170.7 s (wikitext-103, 29,445 documents).
  Tokenizer hashes `2a7147479382…` and `6a1f1477f4d9…`, both re-verified against
  the files by TL2.1b.

  The train-only property is checked three ways, none of which trusts the
  builder's intent:
  1. `tokenizer_training_files` has exactly one entry and it is
     `data/lang/<corpus>/train_text.txt` (TL2.1c/d).
  2. **No val or test text file exists in the corpus directory at all** (TL2.1e
     globs `*_text.txt` and requires the set to be `{train_text.txt}`), so a
     held-out file cannot have been passed by a path the manifest omits.
     `write_train_text` materializes only the train split for exactly this reason.
  3. The round-trip probe is drawn from val — held-out text the tokenizer must
     handle without having been fitted on it — and TL2.1g re-runs an independent
     round-trip through the saved file, since the build's own check is the builder
     grading itself.

  Byte-level is verified as *no-OOV-is-representable* rather than as
  *no-`<unk>`-was-emitted*: TL2.1h encodes every printable ASCII byte
  individually and requires a non-empty id list for each, confirming the 256-byte
  initial alphabet survived training (TL2.1i).

  **Document segmentation is lossless, measured:** `train_text_bytes` equals the
  independent raw byte scan exactly — 10,914,845 and 539,295,549 — so joining HF
  rows with `""` reproduces the corpus byte-for-byte and `iter_documents` drops
  nothing. This matters because `_is_doc_start` is a heuristic on ` = title = `
  headings; if it mis-split, the EOT positions would be wrong and this equality is
  what says they are not.

- [x] **T-L2.2 Pack to fixed-length blocks, drop the trailing partial.** Encode
  each split into one stream, cut into `seq_len`-length blocks, and **discard the
  final short block**. This is not laziness: an all-True `step_mask` means no
  `key_padding_mask`, which means the NaN path in attention over a fully-masked
  row is unreachable and the ACT denominators are exact rather than
  mask-conditional. Store as `.npy` `uint16` (V=8192 fits) and memory-map at load
  — not JSONL, which would re-parse ~100 M tokens every epoch.
  **Verify:** every stored block has length exactly `seq_len`; the dropped tail is
  `< seq_len` tokens and its size is recorded in the manifest;
  `np.load(mmap_mode="r")` opens without materializing the array.
  **Evidence:** `seq_len = 256`, `uint16`, all three assertions checked on the
  **stored arrays**, not on the builder's arithmetic — block length from
  `a.shape[1]`, block count from `a.shape[0]`, and `isinstance(a, np.memmap)`.

  | corpus | split | tokens | blocks | dropped tail |
  |---|---|---|---|---|
  | wikitext-2 | train | 2,695,045 | 10,527 | 133 |
  | wikitext-2 | val | 284,328 | 1,110 | 168 |
  | wikitext-2 | test | 325,675 | 1,272 | 43 |
  | wikitext-103 | train | **134,737,951** | **526,320** | 31 |
  | wikitext-103 | val | 281,335 | 1,098 | 247 |
  | wikitext-103 | test | 321,583 | 1,256 | 47 |

  Every tail `< 256`, and the accounting identity
  `tokens == blocks × seq_len + dropped_tail` holds **exactly** for all six
  split/corpus pairs (TL2.2g) — so nothing is lost except the tail that is
  recorded. Total dropped across the canonical corpus: 325 tokens of 135,340,869,
  i.e. 2.4 × 10⁻⁶. `dataset_version = lang-wikitext-2-bpe8192-len256-9f870794`
  and `lang-wikitext-103-bpe8192-len256-800d6154`; per-split SHA-256 recorded and
  re-verified against the files.

  Memory behaviour is the reason this task is not cosmetic: the canonical
  `train.npy` is 269 MB and opens as a `numpy.memmap`, and the encode path streams
  uint16 to a raw `.bin` before reshaping — a Python list of 134.7 M ints would
  have been ~3.6 GB. Peak build RSS stayed well inside the 6 GB card's host
  budget. Encode wall-clock 194.6 s for the canonical train split.

  **FINDING — per-token loss is not comparable across the two corpora.** The val
  and test *text* is byte-identical (see T-L2.0), but each corpus has its own BPE,
  so the same 1,144,248 bytes of val become 284,328 tokens under wikitext-2's
  tokenizer and 281,335 under wikitext-103's: 4.024 vs 4.067 bytes/token, a 1.05%
  difference in the **denominator** of any nats/token figure. The canonical
  corpus's tokenizer is the more efficient one (fitted on 50× more text), so a
  naive cross-corpus comparison flatters wikitext-103 by ~1% for free. Any
  cross-corpus statement must therefore be in **bits per byte**, not nats per
  token. Noted now because it is invisible once the numbers are in a table.

- [x] **T-L2.3 `MoRELanguageDataset` in `code/more/lang_data.py`.** Returns the
  same-arity tuple shape the engine already consumes, with the language meanings:
  `input_ids [seq_len] long`, `step_mask [seq_len] bool` (all True),
  `step_experts [seq_len] long` from the POS lookup (`-1` = unmapped → ignore),
  `step_ops` = `N/A` for language, plus the LM target derived by shift inside the
  loss rather than stored twice. All shuffling is the DataLoader's; the stored
  block order is fixed so the manifest hash is stable. **Verify:** a batch has the
  documented dtypes and shapes; two DataLoader epochs at the same seed produce the
  same permutation, and different seeds produce different ones.
  **Evidence:** checks TL2.3a–m, both corpora. The arity is **7**, and TL2.3a reads
  that number out of `MoREDataset.__getitem__`'s source rather than hard-coding it,
  so a change on the arithmetic side fails here instead of drifting.

  | slot | arithmetic | language | canonical value |
  |---|---|---|---|
  | 0 | `x [S,F]` float | `input_ids [256]` | int64, max id 8129 < 8192 |
  | 1 | `step_mask [S]` bool | same | **all True** |
  | 2 | `step_experts [S]` long | same | in `[-1, 5]`, `== token_family[input_ids]` |
  | 3 | `step_ops [S]` long | same | **all −1** |
  | 4 | `family` scalar long | same | **−1** |
  | 5 | `depth` scalar long | same | 256 = `seq_len` |
  | 6 | `target` scalar float | same | **NaN** |

  The arity is preserved rather than a new signature introduced because
  `engine.py:438` and the validation loop at 686 both unpack positionally, and a
  second arity would force a second training loop — which `plan.md` §9 forbids.
  Slots with no language meaning carry the documented ignore label instead of being
  dropped.

  **Slot 6 is NaN on purpose: it is a poison value, not a placeholder.** The LM
  target is `input_ids[1:]` predicted from `input_ids[:-1]`, derived by shifting
  inside the loss so the ids are not stored twice. The slot still has to hold
  something collatable, and `0.0` or `-1.0` would let a mis-wired language loss
  train silently against a constant and produce a plausible-looking curve. NaN makes
  that defect surface on the first optimizer step. It is never logged and never
  reported, so this is not a sentinel presented as a measurement (CLAUDE.md §4) — it
  is a value that cannot be mistaken for one.

  **Determinism is checked on the emitted block CONTENT, not on indices.** An index
  permutation that were reproducible while `__getitem__` was not would pass an
  index-level check and still be non-deterministic. Two epochs at seed 42 give
  byte-identical first batches; seed 43 differs; with `shuffle=False`, block 0 of the
  loader is block 0 of the file, so the manifest's per-split hash describes exactly
  what the model reads. Reproducibility comes from
  `seeding.make_generator(seed, "dataloader_shuffle")` unchanged — no new generator
  purpose was added.

  **The memory-map is opened per process, keyed on pid.** A `np.memmap` held as an
  attribute is pickled by materializing it, so with `num_workers > 0` under Windows
  spawn every worker would receive its own **269 MB** copy of the canonical train
  split. Opened lazily in `_blocks()` instead. The lookup is validated on load rather
  than trusted: a wrong length means it was built against a different tokenizer, and
  a family index ≥ 6 would silently widen the oracle label space — the defect T8.3
  fixed on the arithmetic side.

  `more/__init__.py` exports `MoRELanguageDataset` and `TRAIN_SPLIT_VERSION`, and
  **deliberately re-exports nothing from `.lang_families`**: the eight names above it
  come from `.families` unconditionally, and putting a second manifest's labels in
  the same namespace is how `EXPERT_FAMILY_LABELS` ends up meaning whichever module
  imported last. `train_split_version = "wikitext-author-splits-v1"` is defined here,
  which is what T-L2.4's `frozen_by` note said would supply it.


- [x] **T-L2.4 Manifest with token- and type-level statistics.** Mirror
  `data/dataset_meta.json`'s schema: dataset version, tokenizer hash, per-split
  block counts and token counts, per-split SHA-256, `family_counts` at **both**
  the type level (how many vocabulary entries per family) and the token level (how
  many corpus occurrences per family), the unmapped share, and the frequency-decile
  table. Both levels are needed because a family can be 2% of types and 40% of
  tokens; a load-balance number read against the wrong denominator is
  uninterpretable. **Verify:** the manifest validates against the documented
  schema and its split hashes reproduce on a second build from the same seed.
  **Evidence:** checks TL2.4a–d, both corpora. **50 schema keys across 6 build
  stages**, all present with the declared type on both corpora; 8 per-split fields
  each carrying exactly `{train, val, test}`; type-level counts over all six family
  labels and token-level counts over all six labels for each of the three splits.

  **"The documented schema" is now an object, not prose.**
  `build_language_dataset.MANIFEST_SCHEMA` maps `key → (stage, type, required)`, so a
  missing key names *which build stage* failed to write it. `validate_manifest`
  returns a list of problems rather than raising, because the caller wants all of
  them. The `bool`-is-a-subclass-of-`int` trap is handled in both directions: an
  `int` field holding `True` is a problem, and JSON's habit of writing `1.0` as `1`
  in a float slot is not.

  The correspondence with the arithmetic manifest is written out in that constant's
  comment rather than left as "mirrors", including the four fields language
  legitimately lacks — `operation_counts` (no operation axis), `seed`, `train_frac`,
  `test_frac` (nothing is sampled: the splits are author-provided). That last point
  reinterprets the Verify clause: there is **no seed** to rebuild "from the same
  seed" with, so the check is plain determinism, which is the stronger requirement.

  **The rebuild clause is executed, not asserted.** TL2.4d builds wikitext-2 from
  scratch through all three stages into a scratch out-root and compares nine fields.
  All identical: `dataset_version = lang-wikitext-2-bpe8192-len256-9f870794`, all
  three per-split SHA-256s, all token and block counts, the dropped tails, and —
  the one that was not guaranteed — **`tokenizer.json` byte-identical**. So on a
  fixed `tokenizers` version the BPE fit is deterministic; the `.gitignore` note
  about merge-order tie-breaking is a cross-version caution and now says so. Dev
  corpus only, because the code path is identical and rebuilding wikitext-103 would
  cost ~6 minutes of encode for no extra information. `SKIP_REBUILD=1` skips the
  ~20 s.

  **A latent cross-drive defect fired on the first run of this check, and it is the
  same one as T-L0.3a.** `fit_tokenizer` recorded its training-file list with
  `os.path.relpath(train_text, _REPO)`, and on Windows `relpath` **raises** when the
  two paths are on different drives: `ValueError: path is on mount 'C:', start on
  mount 'D:'`. The repository is on `D:`; the scratch out-root is under
  `C:\Users\...\Temp`. It was invisible while every build wrote under `data/lang/`.
  Fixed with a local `_repo_relative()` helper mirroring
  `more/run_context.py:_repo_relative` — relative form when one exists, absolute
  otherwise. An off-drive rebuild therefore records absolute paths in *its* manifest,
  which is correct, and is why TL2.1d grades only the canonical location.

  **A number in this ledger was wrong and is corrected.** T-L2.0's Evidence table
  gave wikitext-2's non-empty line counts as 31,175 / 3,213 / 3,760. The manifest and
  TL2.0l's own output say **23,767 / 2,461 / 2,891**. The test was reading the right
  values throughout; only the hand-written table was wrong — which is the case for
  `updated_rules.md`'s "never hand-copy numbers into a results table".

- [x] **T-L2.5 Trivial-baseline floors.** Compute, from the train split only and
  evaluated on val: `uniform_ce = ln(8192) ≈ 9.0109`, `unigram_ce`, and a
  Katz-backoff `bigram_ce`. The lowest of these is `primary_metric_floor` and is
  written into the manifest. Without it a val loss of 5.4 nats/token is a number
  with no meaning. **Verify:** all three are finite, ordered
  `uniform ≥ unigram ≥ bigram`, and the unigram/bigram models are fitted on train
  counts alone (assert the fit never touches the val arrays).
  **Evidence:** `data/lang/build_baseline_floors.py`, checks TL2.5a–g. All three
  finite and correctly ordered on both corpora:

  | corpus | uniform | unigram | bigram | floor | bits/token | perplexity |
  |---|---|---|---|---|---|---|
  | wikitext-2 | 9.0109 | 7.1865 | **5.3983** | 5.3983 | 7.7881 | 221.0 |
  | wikitext-103 | 9.0109 | 7.1982 | **4.9849** | 4.9849 | 7.1918 | 146.2 |

  **The number the whole language study is measured against is 4.9849 nats/token.**
  A canonical language run that does not beat it has learned nothing a two-column
  count table could not, and on a 10–20 M-parameter budget that is a real
  possibility rather than a formality — which is why the floor is computed before
  the runs.

  `uniform_ce` is checked against the closed form rather than against itself:
  `ln(8192) = 9.010913` nats = **13.0000 bits exactly**, so a wrong `vocab_size`
  would fail TL2.5c even though it would pass every other check here.

  **The train-only property is proved by interception, not asserted.** TL2.5g wraps
  `numpy.load` for the duration of `fit_unigram` *and* `fit_bigram` and records
  every path handed to it: `numpy.load was called 2x, all on ['train.npy']`. So a
  fit that read val — through any code path, including one the manifest does not
  mention — shows up. TL2.5f additionally re-derives `unigram_ce` from the arrays
  inside the test file and requires agreement to 1e-9 (`7.186473059` vs
  `7.186473059`), which catches a writer that computed one number and recorded
  another. The bigram is deliberately not re-derived: re-implementing its
  discounting in the test would only check that the same author wrote the same
  formula twice.

  **Naming, corrected rather than glossed.** The plan says "Katz-backoff". What is
  implemented is the **absolute-discounting** variant of the same backoff structure
  — `max(c(v,w) − D, 0) / c(v)` for seen bigrams with `λ(v) = D·types_after(v)/c(v)`
  of leftover mass on the unigram, `D = 0.75` — not Katz's original Good–Turing
  discounting. The manifest records
  `bigram_smoothing = "backoff, absolute discounting D=0.75"` so nobody reads
  "Katz" and assumes Good–Turing. For a floor the difference is far below the
  precision anyone quotes, and TL2.5e requires the smoothing string to be present.
  Unigram smoothing is chosen by measurement, not by default: 80 of 8192 types are
  unseen in wikitext-103's train split (151 in wikitext-2), so add-1 applies and is
  recorded; with no zeros the builder would have used MLE and said so.

  **Block boundaries are not sequence boundaries**, and the bigram model depends on
  it. `pack_split` cuts one contiguous stream, so row *i*'s last token really was
  adjacent to row *i+1*'s first — the fit and the evaluation both read each split as
  one flat stream. Treating rows as independent would discard one bigram per block
  and would not match how a trained model sees the data. Canonical train: 134,737,919
  bigrams over 5,033,639 distinct context–continuation pairs, i.e. **7.5% of the
  8192² possible pairs are observed**, which is why the backoff mass matters.

  Also recorded in `code/canonical_spec_language.json`:
  `protocol.primary_metric_floor` is now **4.98494701553838** rather than null, and
  its note is corrected — it previously named the *unigram* entropy as the language
  floor, which the plan's own §3 contradicts. It also had `ln(8192) = 9.0106`, which
  is wrong in the fourth decimal.

- [x] **T-L2.6 Frequency-decile depth table — built, and deliberately unused.**
  Produce a `token_target_depth[V]` table from train frequency deciles so
  ablation F (`supervised_curriculum`) is runnable, and write into the module
  docstring that it is **not** part of the canonical configuration and is a design
  choice rather than a measurement. `plan_language.md` §5 is explicit: canonical
  language runs invent no depth target. **Verify:** the table exists with one entry
  per vocabulary id; the canonical language config's
  `loss_weights.halting_supervision` is `0.0` and `halt_target_mode` is the pure
  ACT mode.
  **Evidence:** `data/lang/build_depth_deciles.py`, checks TL2.6a–j. `token_decile.npy`
  is `int8`, length 8192, values exactly `{0..9}` on both corpora, hash recorded and
  re-verified. `canonical_spec_language.json` has `halting_mode = "pure_act"` and
  `family_cls_weight = 0.0`, and its `frozen_by` note states the stronger fact:
  supervised_curriculum is **not available** on language at all, because a packed LM
  sequence has no per-token complexity label to build a per-item oracle depth from.

  **Measurement and choice are in different files, on purpose.** The builder
  produces `token_decile[V]` — a measured property of the train split. The
  decile→depth mapping is a design choice and lives in
  `lang_families.decile_target_depth(decile, max_depth)`, which takes `max_depth` as
  an argument so the artifact does not hard-code a recursion budget it cannot know
  at dataset-build time. At `max_depth = 7` the map is
  `[1, 2, 2, 3, 4, 4, 5, 6, 6, 7]`; an out-of-range decile or `max_depth < 1`
  raises, with no fallback.

  **Deciles are cut at equal TOKEN mass, not equal type count**, and the canonical
  numbers show why that was not a stylistic choice:

  | decile | types | tokens | share | count range |
  |---|---|---|---|---|
  | 0 | **3** | 14,202,737 | 10.54% | 3,630,874–5,600,357 |
  | 4 | 127 | 13,387,574 | 9.94% | 72,364–155,817 |
  | 9 | **4,265** | 13,471,820 | 10.00% | 0–5,248 |

  Three types carry 10.5% of the corpus; 4,265 types — 52% of the vocabulary — carry
  the last 10%. Equal-type-count deciles would have put ~90% of corpus occurrences
  in the single most-frequent decile, so the "curriculum" would have assigned one
  depth to nearly every token the loss actually weights.

  **The confound is measured, and it is milder than feared.**
  `families.OP_TARGET_DEPTH` was deliberately built so target depth is *not* a
  function of the expert index, so a model could not score well on depth by routing
  alone. A frequency decile has no such protection — function words are frequent,
  subword pieces are rare. The token-weighted normalized mutual information between
  decile and POS family is **0.2724** on wikitext-103 (0.2811 on wikitext-2):
  `I = 0.4364` nats against `H(decile) = 2.3016`, `H(family) = 1.6021`. So the two
  are correlated but far from equivalent, and ablation F's depth numbers can be read
  as carrying real information provided they are quoted against 0.27 rather than
  against 0.

- [x] **T-L2.7 GATE L2 — dataset integrity.** **Verify:** no train/val/test block
  overlap by exact-content hash; the tokenizer was fitted on train only; every
  block is full-length; the unmapped-token share is ≤ 2% (§4.3 budget) and its
  measured value is published in the manifest; the trivial floors are present and
  finite; the manifest hash is reproducible.
  **Evidence: GATE L2 PASSES on both corpora.** `code/test_language_data.py` →
  **133 passed, 0 failed, 0 skipped**; `code/test_language_families.py` →
  **57 passed, 0 failed, 2 skipped** (the skips are T-L3.2/T-L3.3, not yet built);
  `code/test_language_task_axis.py` → **72 passed, 0 failed**. Checks TL2.7a–d.

  **Overlap is checked by exact content, not by hash.** Raw 512-byte block bytes are
  used as dict keys — 2,354 held-out canonical blocks is 1.2 MB, so exactness costs
  nothing and there is no collision argument to make. All **526,320** canonical
  train blocks were streamed against that set (the 269 MB array stays memory-mapped):
  **zero overlap**, and val/test share no block with each other either
  (2,354 distinct blocks from 2,354 stored, so no duplicates within the held-out
  data at all). This is the check that makes the author-provided splits' claimed
  document-disjointness a measured fact for this build.

  **"The manifest hash is reproducible"** is interpreted as the one hash the
  manifest actually publishes: `dataset_version`. TL2.7c re-derives it from the
  recorded per-split SHA-256 values and requires equality —
  `lang-wikitext-103-bpe8192-len256-800d6154` and
  `lang-wikitext-2-bpe8192-len256-9f870794`. So the version names the exact bytes,
  and a silently rebuilt split changes it.

  The five aggregate clauses (TL2.7d): tokenizer fitted on train only; every block
  full-length; unmapped ≤ 2% (**1.2344%** canonical, 1.3011% dev); the measured
  unmapped share published as a float; all four floor keys present and finite.

  **A DEFECT FOUND AND FIXED WHILE RUNNING THIS GATE — a silent manifest clobber.**
  `build_baseline_floors.py --corpus wikitext-103` finished while
  `build_family_lookup.py --corpus wikitext-103` was still tagging. The lookup then
  called `save_manifest` with the manifest snapshot it had loaded 25 minutes
  earlier, **erasing `uniform_ce`, `unigram_ce`, `bigram_ce` and
  `primary_metric_floor`.** Nothing raised; the JSON stayed valid. It surfaced only
  because TL2.5 reports a missing floor as a `[SKIP]` naming the reason, so the
  absence was visible in the output rather than merely absent — an argument for
  skipping loudly instead of conditionally not checking. Fixed by
  `bld.update_manifest(corpus, new_fields)`, which re-reads the on-disk manifest
  immediately before merging, and all three stage builders now use it. The residual
  race is stated in that function's docstring rather than papered over: two builders
  whose read-and-write windows overlap can still lose keys, the window is now
  milliseconds instead of minutes, and the operational rule is that two stage
  builders must not run concurrently on the same corpus.


---

## Phase L-3 — Family lookup and routing semantics  (`plan_language.md` §4)

- [x] **T-L3.0 Six POS families in `code/more/lang_families.py`.** `L1 FUNCTION,
  L2 NOUN, L3 VERB, L4 MODIFIER, L5 PUNCT_SYM, L6 NUM_SUBWORD`, mirroring
  `families.py`'s structure so `metrics.py` needs no new label plumbing. Six to
  keep the MoE axis width identical to arithmetic, so any difference between the
  two studies is not a difference in expert count. **Verify:** labels contain no
  `/` (W&B nesting rule); `NUM_FAMILIES == 6`; the module exposes the same
  accessor names `metrics.py` already imports from `families.py`.
  **Evidence:** `code/test_language_families.py` → **24 passed, 0 failed, 3
  skipped** (the 3 are T-L3.1/3.2/3.3, not yet built). All three Verify clauses
  hold: no label contains `/`, `NUM_FAMILIES == 6`, and every one of the four
  names `metrics.py` imports from `families.py` is present.

  Two checks are deliberately not restatements of the module. **TL3.0e/f parse the
  import statements out of `metrics.py` and `__init__.py` with a regex** and
  require this module to satisfy whatever those files actually ask for today — so
  adding an import there fails the check until someone classifies the new name,
  instead of the check silently going out of date. **TL3.0w loads the file by path
  in a clean subprocess** and inspects `sys.modules`, because "the manifest is
  dependency-free" is a claim about import side effects that cannot be tested from
  inside a process that already imported torch.

  **Both of those checks failed on first run, and both failures were real.**
  1. `__init__.py` re-exports **eight** names from `families.py`, and
     `OP_TO_EXPERT` has no language meaning — it maps the 16 arithmetic op codes to
     experts. Defining `OP_TO_EXPERT = {}` for parity was rejected: an empty dict
     makes `OP_TO_EXPERT[op]` raise a bare `KeyError` at whatever line indexes it,
     which reads as a missing operation rather than as a caller that reached for
     the arithmetic axis while running language. Fixed with a PEP 562 module
     `__getattr__` that **raises `AttributeError` naming the reason**, plus an
     `_ARITHMETIC_ONLY` registry (`OP_TO_EXPERT`, `OP_TARGET_DEPTH`), and TL3.0f
     rewritten to require every re-exported name to be *either* defined *or*
     explicitly declared arithmetic-only. **A leak this does not close, recorded
     because it will matter in Phase L-5:** `more/__init__.py` imports from
     `.families` unconditionally, so `from more import OP_TO_EXPERT` still yields
     the arithmetic mapping regardless. Only `lang_families.OP_TO_EXPERT` is
     guarded; making the package export task-conditional is a Phase L-5 change.
  2. Importing `more.lang_families` pulls in **torch and numpy** — not from this
     file, but because `more/__init__.py` runs first and imports `engine`/`model`.
     The docstring's "deliberately free of torch" was true of the file and false of
     the import, and a reader would have taken it as "reading this is cheap". Fixed
     both ways: the probe now loads by path via `importlib.util`, and the docstring
     states the package-level truth explicitly.

  Design decisions made here, all recorded in the module rather than left implicit:

  - **`ALL_OP_NAMES = []`, `NUM_OP_TYPES = 0`, `op_target_depth_table() → []`.**
    Language has no operation axis, and the emptiness is load-bearing:
    `engine.py:803` iterates `enumerate(ALL_OP_NAMES)`, so an empty list yields an
    empty `op_avg_depth` and the metric layer reports *absent* rather than `0.0`
    (CLAUDE.md §4: no sentinel reported as a measurement). An empty depth table is
    also stronger than a zero-filled one — it cannot supervise depth whatever
    weight a later config sets, which is `plan_language.md` §5 made mechanical
    instead of merely written down.
  - **The auxiliary override is the BE paradigm only.** §4.1 puts auxiliary and
    copular verbs in L1, but Penn tags them `VB*` like main verbs, so an override
    is unavoidable. It covers `be/am/is/are/was/were/been/being` plus the clitics
    `'s/'re/'m` — BE has no main-verb use in English other than the copula, so the
    override adds no judgement the manifest has not already made. **`have` and
    `do` are deliberately excluded and land in L3 VERB**, because both have
    genuine main-verb uses and deciding between them is exactly the job of
    T-L3.1's type-level majority vote; hard-coding it would bypass the mechanism
    whose purpose is to make that call from the corpus. TL3.0r pins the
    consequence so it cannot be "fixed" later without a decision.
  - **`not` / `n't` → L1.** Penn tags them `RB`, which would put negation among
    open-class adverbs; it is a closed-class particle and §4.1's L1 list names
    particles.
  - **`WRB` (how/where/when/why) → L1, `UH` → L1, `LS` → L5, `FW` → −1.** The
    L1/L4 boundary is drawn at "closed grammatical class vs open modifier class",
    which puts wh-adverbs with the other wh-words. `FW` is the one tag with no
    English POS at all, so it spends ignore budget rather than assert a class the
    tagger did not find.
  - **The L6-vs-−1 boundary**, which §4.1 and §4.3 leave overlapping.
    `surface_class_family` is consulted only for types the corpus never shows as a
    standalone word: digits → L6, punctuation-only → L5, contains a letter → L6
    (the "subword continuation piece" case), everything else → −1 (whitespace-only
    pieces, lone continuation bytes, unused vocabulary entries). Stated as a design
    choice: sending every no-evidence type to −1 would make the 2% budget check
    vacuous, and sending everything to L6 would make L6 a catch-all and reproduce
    E7. T-L3.1 writes the share each branch produces into the manifest so the split
    is auditable.
  - **Five import-time guards**, mirroring `families.py`'s raise-at-import
    discipline: no `/` in a label, `NUM_FAMILIES == 6`, the mapped and ignored tag
    sets disjoint, every mapped index in `0..5`, every family reachable from at
    least one tag (a family nothing can reach gives an empty confusion-matrix row
    that reads as a routing failure rather than a manifest defect), and full
    coverage of the 45-tag standard Penn set. A bad edit fails on first import, not
    after a six-minute dataset build.

  Gate L0 after the Phase L-2 commit `0c3a883`: `TOTAL 356 356 0 0`, ALL GATES
  PASS (T-LX.0 discharged; 173.8 s on the CPU interpreter).


- [x] **T-L3.1 Type-level `token_family[V]` from majority POS.** Tag the train
  split with nltk's averaged-perceptron tagger, take each vocabulary type's
  majority POS across its train occurrences, and freeze an `int8` lookup of length
  V. Type-level, not token-level: a per-occurrence oracle would make the "correct"
  expert depend on context the router cannot see at its own input, so disagreement
  would be measuring the tagger's context sensitivity rather than the router's
  behaviour. Unmapped → `-1` → **ignored** in the metric, not raised and not
  routed to a fallback (the arithmetic raise-don't-fallback rule protects a closed
  16-op set; a byte-level BPE vocabulary has no such closure, and a fallback expert
  is exactly the E7 catch-all the rules removed). **Verify:** the lookup has length
  V and dtype int8; the `-1` share matches the manifest; a spot sample of 40 types
  per family is linguistically sane on inspection.
  **Evidence:** `data/lang/build_family_lookup.py`, checks TL3.1a–q. `int8`,
  length 8192, values exactly `{-1, 0..5}`, hash recorded and re-verified, EOT
  mapped to `-1` (a document separator is not a lexical class). Both the type-level
  and token-level counts are **recounted from the array and the packed `.npy`
  files** rather than read from the manifest, so a writer that recorded different
  numbers than it computed fails. 100,285,152 words tagged on the canonical corpus
  in 19.1 min with 6 workers.

  | | wikitext-2 | wikitext-103 |
  |---|---|---|
  | types decided by corpus vote | 4,689 | **5,559** |
  | types by surface-class fallback | 3,502 | 2,632 |
  | types left `-1` | 180 (2.20% of V) | 176 (2.15% of V) |
  | voted types that were contested | 45.1% | **63.1%** |
  | half-vs-half majority disagreement | 2.40% | **1.61%** |
  | **unmapped TRAIN-TOKEN share** | 1.3011% | **1.2344%** |

  The two columns move in the directions more data should produce: more types earn a
  standalone vote, more of them turn out to be genuinely ambiguous, and the majority
  becomes *more* stable rather than less — 1.61% of 5,266 types flip between the
  first and second half of the corpus. Halves rather than an interleaved split
  deliberately: interleaving samples the same distribution twice and would converge
  trivially, hiding any drift across the corpus.

  **Canonical token-level family shares:** L1_FUNCTION 27.8%, L2_NOUN 29.6%,
  L3_VERB 5.7%, L4_MODIFIER 6.5%, L5_PUNCT_SYM 8.7%, L6_NUM_SUBWORD 20.4%.

  **FINDING — the oracle partition is not balanced, and the balance loss is
  therefore in tension with the POS metric in a way it never was on arithmetic.**
  The partition's own normalized load entropy is **H/log(6) = 0.8884** (wikitext-2
  measurement; largest/smallest family token ratio **5.29×**). A router that
  perfectly reproduced POS would score 0.8884, not 1.0 — so a router driven to 1.0
  by the balance term must *disagree* with POS by construction. Every language
  load-entropy number has to be read against the partition's own entropy, never
  against 1.0. On arithmetic the oracle families were near-uniform by construction,
  so this question did not arise. TL3.1l pins it.

  **The type/token divergence T-L2.4 was written to guard against is extreme here:**
  L1_FUNCTION is **2.6% of types and 27.3% of tokens** — a 10.7× ratio — and
  L5_PUNCT_SYM is 0.57% of types and 8.9% of tokens. TL3.1k requires at least one
  family to diverge by more than 10 percentage points, so the manifest's two
  denominators are justified by measurement rather than by hypothesis.

  **Whitespace tokenization, chosen by measurement.** `TreebankWordTokenizer` yields
  1.0251× more tokens on wikitext-2, and the excess is entirely units that **do not
  exist in the byte stream the BPE was fitted on**: it splits WikiText's own `@-@` /
  `@.@` / `@,@` escapes into `@`+`-`+`@` (2,830 occurrences per ~330 k tokens),
  rewrites `"` as Penn's ` `` `, splits `cannot` into `can`+`not`, and strips the
  final period off `U.S.`. Tagging a unit that never appears in the corpus would
  break the word-to-id alignment the vote depends on. WikiText is already
  Moses-tokenized with spaces around punctuation, so whitespace *is* its
  tokenization. Those `@-@` markers are ~0.86% of tokens and got an explicit
  `CORPUS_MARKER_FORMS → L5` override, because the tagger has no opinion worth
  having about them: `@-@` is not English, so the perceptron falls back on shape
  features and emits whatever the context suggests.

  **How a type earns a vote, and what earns nothing.** Each word is encoded *with
  its leading space* (byte-level BPE distinguishes `" the"` from `"the"`); if it
  encodes to exactly one id, that id gets one vote for the word's family. A word
  that splits into several pieces votes for **nothing** — its part of speech is a
  property of the word, and attributing it to an arbitrary piece would invent
  information. On the canonical corpus 80.2% of word occurrences were single-id and
  19.8% were multi-piece; 0.04% were tag-ignored (`FW`). All three are counted in
  the manifest, because "how many occurrences were unusable" is the number that says
  whether the vote had enough evidence.

  **Determinism, verified rather than argued.** Vote counts merge associatively and
  `majority()` breaks ties by lowest family index, never by dict order, so the
  parallel and serial paths must agree. `--workers 1` and `--workers 6` produced
  **byte-identical** lookups on wikitext-2 (both `sha256 731fc370ecd08d87`), and the
  same hash survived the byte-range refactor below. TL3.1q checks the tie rule
  directly; the end-to-end identity is recorded here rather than re-run in the suite,
  because it costs two full corpus passes.

  **Spot sample of 40 types per family (the Verify clause that needs a human).**
  Linguistically sane throughout. L1: ` on ' for be it which who she than where
  some This against may could though And We without my should himself upon above
  must themselves Nor inside Both below Of onto Among Unlike Your unlike`. L3: ` including
  elect want developed meet fight know born taking try constructed proposed`.
  L5: `! # $ % & ' ( ) + , . / : ; < > @ ' .. ... @-@ @.@ @,@ £`. L6 is exactly
  what it should be — `re ion ow se ose uc ians ator ilities` alongside ` 1998 1942
  59 166` and word-initial fragments ` Par Mag Cath Tw Bor Gra Hug Shakespe Celt
  tele prohib`. The 180 `-1` types are **replacement characters (lone UTF-8
  continuation bytes) and C0 control characters** — precisely §4.3's "fragments that
  have no POS", and nothing else.

  Two observations from that inspection, recorded because they look like defects and
  are not:
  1. `'is'` **without** a leading space → L6_NUM_SUBWORD (2,662 train tokens), while
     `' is'` → L1_FUNCTION (11,750). Correct: the space-less type is the word-internal
     fragment of `basis`/`crisis`/`this`, and the leading-space distinction is doing
     exactly the work it was introduced for.
  2. Negated contraction stems (` wasn`, ` doesn`, ` didn`, ` don`, ` cannot`) land in
     L3_VERB rather than L1, because the BE override is a closed list of full forms.
     Total ~600 tokens of 2.69 M = **0.02%**. Left alone deliberately: adding only the
     BE stems while excluding the DO/HAVE stems would trade a stated rule for a
     0.002% effect.

  **A memory defect found and fixed before the canonical build.** The first version
  did `list(iter_chunks(...))` and materialized 1.8 M lines of the 539 MB corpus as
  Python strings — **measured at 1,013 MB resident before a single word was tagged**,
  and feeding a lazy generator to `imap_unordered` would not have helped, because the
  pool drains its input as fast as it can into an unbounded queue. Workers were a
  second 358 MB each, because importing `more.lang_families` runs `more/__init__.py`
  and therefore torch — the exact leak TL3.0w documents. Both fixed: the parent now
  passes `(index, start_byte, end_byte)` triples and each worker reads its own slice,
  and the manifest is loaded by path. Measured after: parent **45 MB**, workers
  **161 MB** each, **1,025 MB total against ~3,100 MB before**, and throughput went
  up (104 k words/s). The refactor was gated on reproducing the pre-refactor lookup
  byte-for-byte, which it did.


- [x] **T-L3.2 Demote routing accuracy to `routing_agreement_with_pos`.**
  STRICTER THAN ARITHMETIC (§4.4). On arithmetic, `OP_TO_EXPERT` is ground truth:
  the family genuinely is the right answer. A POS partition is a linguistic prior,
  not a functional one — nothing says the optimal expert split for next-token
  prediction is noun-vs-verb. So the key is renamed to state what it measures, and
  the permutation-invariant metrics (Hungarian, AMI, purity) become **primary**
  for language rather than supplementary. **Verify:** no language metric key is
  named `routing_accuracy`; `results.md` for language reports Hungarian/AMI/purity
  in the specialization section before any POS-agreement number; the agreement
  number's caption states it is agreement with a prior, not accuracy.
  **Evidence:** checks TL3.2a–m (TL3.2n skipped, see below).
  `code/test_language_families.py` → **89 passed, 0 failed, 1 skipped**. Gate L0
  after the change: `TOTAL 356 356 0 0`, ALL GATES PASS — the arithmetic path is
  untouched.

  **One helper, five call sites.** `metrics.routing_agreement_key(task)` returns
  `val/routing_accuracy` for arithmetic and `val/routing_agreement_with_pos` for
  language, and it is the *only* place either string appears: the W&B log dict, the
  console line, the `results.tsv` header and the `metrics.json` `"N/A"` contract all
  resolve through it. TL3.2m asserts `engine.py` no longer hard-codes the key
  anywhere. The number is unchanged — TL3.2f pins the language value equal to the
  arithmetic one and to the confusion diagonal, to 1e-15. Only the name moved.

  **The default is arithmetic, and that is load-bearing.** Gate L0's G5.2b checks
  the literal string `val/routing_accuracy` on a single-expert run, and
  `resolve_task` makes absence mean arithmetic so no existing config's hash moves.
  TL3.2g calls the log-dict builder with no `task` argument and requires all 46 keys
  to match the explicit-arithmetic call. **An unknown task raises** rather than
  defaulting (TL3.2d): a default would publish a language agreement figure under
  `val/routing_accuracy`, silently, in the key the exporter joins on.

  **The two spellings are deliberately different strings** (TL3.2c) so no exporter
  can join a column of accuracies to a column of agreements and average them.

  **"Primary" is now an object, not a convention.**
  `metrics.LANGUAGE_SPECIALIZATION_ORDER` is
  `hungarian_accuracy → ami → purity → matched_macro_recall → agreement_with_pos`,
  with the agreement key **last**; TL3.2h pins the order and TL3.2i checks every
  name in it is a key a real language log dict actually emits, so it is consumable
  rather than aspirational. TL3.2n is the one skip: the language `results.md` writer
  is Phase L-9, and this constant is the contract it must consume.

  **The caption travels with the number, in the artifact.** `metrics.json` now
  carries `routing_agreement_metric_key` and `routing_agreement_caption`, because a
  caveat that lives only in a prose file is a caveat that gets omitted from a table
  and a caveat that lives only on stdout is one nothing keeps. The language caption
  opens `AGREEMENT WITH A PRIOR, NOT ACCURACY` and names the POS partition a
  linguistic hypothesis rather than a functional ground truth; the arithmetic caption
  still claims a functional ground truth, and TL3.2j/k require the two to make
  different claims. The language caption also carries the T-L3.1 finding forward:
  read load entropy against the partition's own **0.8884**, not against 1.0.

- [x] **T-L3.3 Mandatory shuffled-control partition (ablation G).** Build a second
  `token_family_shuffled[V]` by permuting the family assignment across types while
  holding the per-family type counts fixed, and evaluate the same
  agreement/Hungarian/AMI/purity metrics against it. This is evaluation-time only
  and costs no extra training runs. Its purpose: if the learned partition scores as
  well against a meaningless partition as against POS, then the POS number was
  measuring the metric's floor, not specialization. **Verify:** the shuffled
  control has identical per-family type counts to the real partition; both metric
  sets appear side by side for every language run; the difference is reported with
  its uncertainty, not just the raw pair.
  **Evidence:** `data/lang/build_shuffled_control.py`, checks TL3.3a–l, both
  corpora. `token_family_shuffled.npy` is `int8` `(10, 8192)` — **ten** independent
  draws, not one, because T-L3.3 requires the difference with its uncertainty and a
  single control gives a point estimate with no spread. Seed **20260903**, kept
  separate from the frozen run seeds so a data artifact is never tied to a training
  seed.

  **THE VERIFY CLAUSE WAS NOT IMPLEMENTED AS WRITTEN, AND THE DEPARTURE IS THE POINT.**
  T-L3.3 says "holding the per-family **type** counts fixed". `plan_language.md` §4.4
  says "a random type-level 6-way partition with the observed family **token**
  proportions". Those are different partitions, and on this corpus they are far
  apart: L1_FUNCTION is 2.6% of types and 27.3% of tokens. **§4.4 is the correct
  specification and this artifact implements it**, because every entry of the routing
  confusion matrix is a *token*, so AMI's and Hungarian accuracy's chance levels
  depend on the token marginals. A type-count-matched control would carry a roughly
  token-uniform profile against the real partition's 5.29× imbalance, and the
  comparison would then confound "meaningless" with "differently balanced" — exactly
  the confusion the control exists to remove. The realised type counts are recorded
  anyway (L1: 243 real vs ~2,506 control on wikitext-103) so the departure is
  visible, and the manifest's
  `shuffled_control_matched_note` states it. Marked `[x]` rather than `[!]` because
  the *scientific* requirement is met and improved on; the ledger's wording is what
  was wrong.

  **Matching, measured:** worst per-family token-share error **9.74e-05**
  (wikitext-2) and **3.77e-03** (wikitext-103) across all ten draws, recomputed in
  the test from `train.npy` rather than read from the manifest. The quantity that
  actually sets AMI's chance level is the marginal **entropy**, and that agrees to
  1e-4: 0.894156 real vs 0.894249 ± 2.9e-04 control on the canonical corpus. Unmapped
  types are **exactly** preserved in every draw (TL3.3c), so both metric sets are
  computed over the same token population — otherwise the pair is not comparable.

  **The control DISCRIMINATES, and that is verified now rather than assumed.** Two
  synthetic routers, scored with no trained model in sight:

  | router | AMI vs POS | AMI vs control | Δ |
  |---|---|---|---|
  | random 6-way | 0.0000 | −0.0000 ± 0.0000 | **+0.0000** |
  | POS-perfect | 1.0000 | 0.0527 | **+0.9473** (z = 47.5) |

  The control behaves as a floor for one and not the other, so it separates a real
  partition from the metric's chance level — which is its entire job. **And it gives
  the number the paper needs: the AMI floor for a 6-way partition with these
  marginals is 0.0527, not 0.** An observed AMI of 0.06 against POS would be *at the
  floor*, and without this control it would have read as weak specialization.

  **How a draw is made.** Types are visited in seeded random order and each goes to
  whichever family is furthest below its token-mass target; a repair pass then removes
  the tail. Randomising the visit order is load-bearing — a frequency-ordered pass
  would rebuild part of the real partition through the frequency-family correlation
  (normalized MI 0.27, T-L2.6). The repair needed a **swap** operation, not just a
  move: canonical draw 0 stalled at 1.4e-02 with one-way moves because the over-family
  held no type small enough to transfer without overshooting, while the other nine
  reached ~5e-05. A swap transfers an arbitrarily small *net* mass out of two large
  types, so the granularity floor disappears; every selection is by mass alone and
  never consults the real partition, so the repair cannot reintroduce POS structure.

  **Side by side, with uncertainty.** `metrics.specialization_vs_control` scores the
  same predicted assignments against POS once and against each draw, returning
  `real`, `control_mean`, `control_std`, `delta` and `delta_z` for each of
  `raw_accuracy`, `hungarian_accuracy`, `ami`, `purity`.
  `control_comparison_to_wandb` flattens it and **omits** undefined entries rather
  than writing 0.0 — a 0.0 in a "real minus control" column reads as "the POS
  partition is no better than noise", which is a finding, not a missing value.
  `delta_z` is `None` when the control has no spread, because a z with a zero
  denominator is undefined, not large.


---

## Phase L-4 — Attention  (`plan_language.md` §6)

- [x] **T-L4.0 One shared causal-attention sublayer per `MoREWrapper`.** The model
  as it stands has **no attention at all** — it is a per-token FFN mixture with
  masked mean-pooling and a scalar regression head. Without attention the best
  achievable language model is a unigram model, and the entire MoE/MoR/MoRE matrix
  would be comparing three ways of failing to use context. Add exactly one
  `nn.MultiheadAttention(d_model, n_heads=4, batch_first=True)` per wrapper,
  **reused at every recursion depth** — a per-depth attention module would break
  the §2.1 weight-sharing invariant that makes depth "computation through time".
  Learned absolute position embeddings, frozen and identical across all three arms.
  **Verify:** `sum(p.numel() for p in wrapper.attn.parameters())` appears once per
  wrapper regardless of `max_depth`; the same parameter tensors are used at depth 1
  and depth 7 (assert by identity, `is`, not by shape).
  **Evidence:** `code/test_lang_causality.py` → **22 passed, 0 failed, 0 skipped**
  (checks TLC.0a–e, TLC.2a–e, TLC.3a–g, TLC.4a–e). Gate L0 after the change:
  `TOTAL 356 356 0 0`, ALL GATES PASS.

  One `nn.MultiheadAttention(d_model, n_heads=4, batch_first=True)` plus its own
  `attn_norm`, constructed once in `MoREWrapper.__init__`. At `d_model = 32` the
  sublayer is **4,224 parameters = 4·d² + 4·d exactly**, and it is the same 4,224 at
  `max_depth = 4` and `max_depth = 7` — the count does not grow with depth.

  **The weight-sharing invariant is asserted by storage identity, not by shape.**
  TLC.0d hooks `attn.forward` and records `data_ptr()` for `in_proj_weight`,
  `out_proj.weight` and `attn_norm.weight` at every depth step: one distinct set of
  pointers across four steps. Two distinct modules with equal shapes would pass a
  shape check and silently break `updated_rules.md` §2.1, which is the invariant that
  makes depth "computation through time" rather than a stack of depth-specific
  networks — i.e. the invariant that makes the MoR and MoRE arms mean what they claim.

  Learned absolute position embeddings live in `MoREModel`, not the wrapper, under the
  same flag: `nn.Embedding(max_seq_len, d_model)`, added **once before the depth
  loop**. Not per step, because position is a property of a token's place in the
  sequence rather than of how much computation it has received, and re-adding it every
  step would make the positional signal grow with depth. `attention=True` **requires**
  `max_seq_len` and raises without it — a default would silently truncate or oversize
  the table. Learned-absolute over RoPE/ALiBi per §6.6: both of those modify attention
  scores, and any positional signal that varied with step count would confound the
  depth analysis, which is the one measurement this study exists to make.

- [x] **T-L4.1 `attention=False` constructs no module.** Follow the
  `router_noise_scale` precedent exactly: when the flag is off, the attribute is
  absent rather than present-and-unused, so the arithmetic `state_dict`, the
  arithmetic parameter counts, and Gate 4 are provably untouched rather than
  merely believed to be. **Verify:** an arithmetic `MoREWrapper`'s `state_dict()`
  key set is identical to the pre-change key set; `hasattr(wrapper, "attn")` is
  False; arithmetic parameter totals match T-L0.3.
  **Evidence:** all three clauses hold. `hasattr(w, "attn")` and
  `hasattr(w, "attn_norm")` are both **False** at `attention=False` (TLC.0a);
  the state_dict key set gains exactly the six attention keys when the flag is on and
  **loses nothing** (TLC.0b), so Gate 4's "no unexpected parameter in the canonical
  checkpoint" check still means what it meant.

  **Arithmetic parameter counts are byte-identical to the T-L0.3 baseline**, compared
  per-tensor rather than by total — a compensating pair of shape changes would leave
  the total intact. `diff` of the two JSON dumps is empty:

  | arm | total_params | tensors |
  |---|---|---|
  | moe | 3,201,555 | 54 |
  | mor | 3,197,710 | 24 |
  | more | 3,201,555 | 54 |

  So `README.md`'s published counts and the 0.120% MoR/MoRE budget residual stay
  correct without any test being edited.

- [x] **T-L4.2 Explicit upper-triangular bool mask; `is_causal=True` forbidden.**
  Build the mask explicitly and pass it as `attn_mask`. `is_causal` is a *hint* in
  PyTorch's API: whether it is honoured depends on the backend kernel selected at
  runtime, so a shape/backend change could silently make the model
  non-causal — i.e. silently leak the target — with no error. An explicit mask is
  checked by the same code path on every backend. **Verify:** the mask is
  materialized as a bool tensor with the documented triangle orientation; a grep
  shows no `is_causal=True` anywhere in the language path.
  **Evidence:** `MoREWrapper.causal_mask(S, device)` returns
  `torch.triu(ones(S, S, bool), diagonal=1)` — entry `[i, j]` is `True`
  (= disallowed) exactly when `j > i`, verified against an independently constructed
  reference (TLC.2a). The **diagonal is allowed** (TLC.2b): `diagonal=1`, not
  `diagonal=0`, so a token may attend to itself — excluding the self-position would be
  a different model, not a stricter one. Cached per `(S, device)` in a **plain dict,
  not a registered buffer** (TLC.2c): a buffer would enter `state_dict()` and make a
  language checkpoint structurally dependent on the sequence length it last ran at.

  **The grep clause is implemented with `ast`, not string search, and the naive
  version failed on its own documentation.** `"is_causal=True" in src` matches the
  `causal_mask` docstring that explains why the hint is not used. TLC.2d walks the
  parse tree of `model.py`, `engine.py` and `lang_data.py` for a call keyword actually
  named `is_causal` with a literal `True`: none. TLC.2e separately confirms the mask
  is *passed* (`attn_mask=self.causal_mask(`), so it cannot be built and discarded.

- [x] **T-L4.3 Halted tokens stay attendable but frozen.** A halted position must
  remain visible as a *key/value* to still-active later positions (removing it
  would change the context of tokens that have not halted, coupling their
  predictions to unrelated tokens' halting decisions), while its own state is
  frozen and never recomputed (§2.2). Attention outputs are therefore computed at
  halted query positions and discarded; that waste is real and must be
  **reported**, not hidden, as `route_stats.attn_query_waste_fraction`.
  **Verify:** freezing holds bitwise — a halted position's state at depth `d+1`
  equals its state at depth `d`; the waste fraction is in `metrics.json` and is a
  measured value, never a sentinel.
  **Evidence:** checks TLC.3a–g. Freezing is asserted **bitwise with
  `torch.equal`** on the state the attention sublayer is handed at each depth step
  (captured with a forward pre-hook): 24 positions, exit depths `{2, 3, 4}` over 4
  captures, **zero violations**. TLC.3b makes the check non-vacuous — tokens really do
  halt at three different depths, so there is something frozen to observe; if every
  token halted at `max_depth` the assertion would be empty.

  **"Attendable" is verified as behaviour, not as code.** TLC.3c perturbs an
  early-halting position (position 0 of row 0, halting at depth 2) and requires later
  positions in the same sequence to move: they move by **0.720**, six orders above the
  1e-06 tiling floor. So halting early cannot blind position 40 to the word at
  position 4 — which would couple one token's prediction to another token's halting
  decision. Compared within the position's own batch row deliberately: the other row
  shifts by ~2 ULP through the dispatch path of TLC.4e, which is not the effect under
  test.

  **The waste is measured, and it moves.** `route_stats.attn_query_waste_fraction`
  reads **39.58%** (38 of 96 query slots) at `max_depth = 6`, and **exactly 0.0** at
  `max_depth = 1` where nothing has halted yet (TLC.3f) — so the number is tracking the
  halted fraction rather than reporting a constant. Deeper recursion wastes strictly
  more (TLC.3g). Aggregated across blocks as one fraction over summed slots, not as a
  mean of per-block fractions, so a block that ran seven depth steps contributes seven
  steps' worth rather than one block's worth.

  **On the arithmetic path the key is ABSENT, not 0.0** (TLC.3e). There is no attention
  sublayer, so there is no waste; a `0.0` would be a sentinel standing in for "not
  applicable", which CLAUDE.md §4 forbids. Consumers must read absence as N/A, exactly
  as they already do for the routing keys at `E == 1`.

- [x] **T-L4.4 GATE L1 — causality and leakage.** The single most important gate
  in the language migration: it is the direct analogue of the arithmetic
  target-in-input leakage audit. Perturb `input_ids[b, t+k]` for `k ≥ 1` and assert
  the logits at position `t` are **bit-identical**. Run it at multiple `t` and `k`,
  for all three architectures, and at both `max_depth = 1` and canonical depth,
  because the ACT loop is the place a future-position value could reach the past
  through the halting statistics. **Verify:** `code/test_lang_causality.py` passes;
  a deliberately broken variant (mask removed) **fails** it — a leakage test that
  has never failed has not been shown to be able to.
  **Evidence: GATE L1 PASSES — 22 passed, 0 failed, 0 skipped.** Both clauses hold,
  and the deliberately-broken variant fails by six orders of magnitude.

  **THE VERIFY WORDING NEEDED AMENDING, AND THE AMENDMENT IS THE FINDING.**
  `plan_language.md` §6.4 specifies **bit-identical** logits at positions `≤ t`.
  Measured, that holds *exactly* only at `num_experts = 1, max_depth = 1`. Above it
  there is a residual — and **it is not a causality defect**:

  | configuration | masked | mask removed |
  |---|---|---|
  | E=1, depth 1 | **0.0 exactly** (`torch.equal` True) | 0.37 |
  | E=1, depth 4 / 7 | 1.2e-07 (1 ULP) | 0.93 / 0.95 |
  | E=6, depth 1 / 4 / 7 | 2.4e-07 (2 ULP) | 0.37 / 0.51 / 0.51 |

  The cause is **grouped Top-1 dispatch**. When a perturbation flips the perturbed
  token's expert, the per-expert row counts change — measured
  `[9,6,4,3,2,0] → [8,7,5,2,2,0]` — so two `nn.Linear` GEMMs get different shapes and
  tile differently, moving the shared rows in their last one or two bits. The same
  happens for `E = 1` at depth > 1, where a changed halt decision changes `N_active`.

  **The mechanism is proved, not asserted (TLC.4e):** `E=6, depth 1` **with attention
  entirely absent** still shifts the past by 2.4e-07, while `E=1, depth 1` is exactly
  0.0. That isolates the cause to the grouping and rules out the mask.

  So the gate is three parts, which together assert *more* than a bare `torch.equal`
  could: **TLC.4a** exact bit-identity where the confound is absent; **TLC.4b** an
  8-ULP bound (9.5e-07) for every architecture, worst observed 2.4e-07; **TLC.4c** the
  mask-removed variant exceeding that bound by ≥ 1e5 in all five configurations
  (smallest ratio **2.1e+06**). A bare `torch.equal` would have been either
  unachievable or — if the tolerance had been loosened without this explanation —
  unfalsifiable. Marked `[x]` rather than `[!]` because the scientific requirement is
  met and strengthened; the ledger's and plan's wording is what was imprecise, and
  §6.4 now carries the amendment.

  All three architectures are covered as required: MoE (E=6, depth 1), MoR (E=1, depth
  7), MoRE (E=6, depth 7), plus MoRE at depth 4 and an attention-only E=1 depth-4 arm.
  **MoR keeps attention** — `updated_objective.md` §4 defines it as recursion without
  multiple experts, not as no context — and is just as causal. TLC.4d records that
  `attention=False` is causal for a *different* reason worth separating: absence of any
  cross-position path, so it could not leak even with a broken mask.


---

## Phase L-5 — Heads and the language loss  (`plan_language.md` §7)

- [x] **T-L5.0 Token embedding + weight-tied LM head.** `nn.Embedding(V, d_model)`
  on input, and an output projection whose weight **is** the embedding weight.
  Tying halves the embedding parameter cost and keeps the §6 parameter budget a
  statement about the expert stack rather than about two independent V×d_model
  tables. **Verify:** the two weights are the same tensor object (`is`); the tied
  head's gradient accumulates from both the input and output paths.
  **Evidence:** `code/test_lang_heads.py` → **25 passed, 0 failed, 0 skipped**
  (TLH.0a–f, TLH.1a–d, TLH.2a–f, TLH.3a–d, TL5.4a–e). Gate L0 after the change:
  `TOTAL 356 356 0 0`, ALL GATES PASS.

  `lm_head.weight is tok_embed.weight` → **True** (identity, not equality — a copy
  kept in sync would drift the moment either was updated in place), and the head has
  **no bias**: a bias would be a per-token logit offset with no counterpart in the
  embedding, so the head would stop being the transpose of the input map.

  **The saving is measured against an untied copy of the same model**, not against a
  formula: **5,523,468 tied vs 7,620,620 untied = exactly 2,097,152 saved**. The
  expert stack plus attention is the other 3,426,316, which is what makes a MoR/MoRE
  budget comparison a statement about *them* rather than about lookup tables.

  **The tie is real in the backward direction too**, which is what "keeps the
  embedding under gradient from both directions" actually names: after one backward
  pass **8,192 of 8,192** embedding rows carry gradient, against only **64** distinct
  ids in the batch. The excess is the output path, which touches every row of the
  vocabulary.

  `task='language'` without `vocab_size` **raises** (TLH.0e), and a float input to the
  language path **raises** rather than being indexed by truncation (TLH.0f).

  **A structural note recorded because it forced a small refactor.** `MoREModel` has
  to branch on the task to decide which heads exist, and `config.py` imports
  `model.py` and never the reverse (`run_context.py:52`). So `TASK_ARITHMETIC` and
  `TASK_LANGUAGE` moved *down* into `model.py` and `config.py` re-exports them
  unchanged — the alternative was duplicating two string literals across two modules,
  which is against the one-manifest principle. No call site moved, and Gate L0 covers
  `config.py`.

- [x] **T-L5.1 Causal-LM cross-entropy as `task_loss`, perplexity alongside.**
  `task_loss = F.cross_entropy(logits[:, :-1].reshape(-1, V),
  input_ids[:, 1:].reshape(-1))` in **nats/token**; `perplexity = exp(task_loss)`
  reported and never optimized. Nats because every other loss term in the assembly
  is in nats and a bits/nats mix in one weighted sum is a silent scaling bug.
  **Verify:** at initialization `task_loss ≈ ln(8192) ≈ 9.011` (the uniform floor —
  a strong check that the shift, the reshape and the vocabulary agree); perplexity
  equals `exp(task_loss)` to floating-point tolerance.
  **Evidence:** at initialization, over the five frozen seeds,
  `task_loss = 9.0277 … 9.0898` against `ln(8192) = 9.0109` — **worst relative error
  0.88%**. Perplexity 8,331 … 8,864 against `V = 8,192`, and equals `exp(task_loss)`
  to 1e-9. The unit is checked against the closed form rather than against a comment:
  9.0567 is near `ln(V) = 9.0109`, not near `log2(V) = 13.0000`.

  **THAT CHECK EARNED ITS PLACE IMMEDIATELY. With `nn.Embedding`'s default `N(0, 1)`
  the measured initial loss was 167.6 nats** — an 18× overshoot of the uniform floor,
  perplexity ≈ 1e73. Weight tying makes the *embedding's* init scale an *output-logit*
  scale: `logits = h @ W.T`, so with `h` at unit RMS the logit spread is
  `std(W) · sqrt(d_model) = 1 × 16`, and a 16-nat spread over 8,192 classes is a
  confidently wrong distribution rather than a uniform one. A model starting there
  spends its first epochs undoing its own initialization, which appears as a
  suspiciously steep early loss curve and never as an error. Fixed with
  `std = 0.02` (the GPT-2 convention) on both `tok_embed` and `pos_embed` — the same
  scale on both because they are *added*, so an `N(0,1)` positional table beside an
  `N(0,0.02)` token table would make position 50× louder than identity at init.

  **A second measured number worth having (TLH.1d): a mis-shifted language run would
  report ~5.6 nats at epoch 0.** Scoring the *unshifted* target gives **5.5552** —
  3.46 nats below the uniform floor, before any training — because a tied head over a
  residual trunk already peaks `logits_i` at `x_i`. That is the copy shortcut tying
  gives for free, and it is what a wrongly-shifted run would show while looking like
  it had learned instantly. Gate L1 detects the general case; this pins the magnitude,
  and 5.6 is close enough to the 4.98 bigram floor to be genuinely deceiving.

- [x] **T-L5.2 Retire the regression head on the language path.** `regression_head`
  and the masked mean-pool exist for the scalar arithmetic answer and have no
  meaning for next-token prediction. They must be absent on the language path, not
  merely unused with a zero weight — an unused head still contributes parameters to
  the budget comparison and still invites a future reader to weight it.
  **Verify:** a language model's `state_dict` contains no `regression_head.*` key;
  an arithmetic model's still does.
  **Evidence:** both clauses hold (TLH.2a/b). A language `state_dict` has **zero**
  `regression_head.*` keys; an arithmetic one still has all six. Going further than
  the box asked, **five** arithmetic-only modules are absent as attributes too —
  `regression_head`, `cls_head`, `step_cls_head`, `step_proj`, `op_embed` (TLH.2c) —
  and the four language-only ones (`tok_embed`, `lm_head`, `family_probe`,
  `pos_embed`) are absent on arithmetic (TLH.2d).

  `op_embed` is the one that mattered most to drop: on language it would have been
  `nn.Embedding(0, d_model)`, because `lang_families.NUM_OP_TYPES` is 0 — legal to
  build, and raising only when indexed. That is the worst kind of latent trap.

  **The tuple arity stays 13** so `engine.py` needs no second training loop
  (`plan.md` §9). Slot 0 carries `[B, S, V]` logits, slot 1 is **None** (a packed LM
  block has no whole-sequence family, so the head is not constructed and the engine
  emits a structural zero), slot 2 is the probe at `[B, S, 6]`. **No pooling**: every
  position is a prediction, so collapsing the sequence would throw the task away
  (TLH.2f).

- [x] **T-L5.3 `family_probe` reads `h.detach()`.** STRICTER THAN ARITHMETIC
  (§7.3). On arithmetic, `family_cls` at weight 0.5 shapes the trunk, which
  permanently qualifies the specialization claim there: some of the observed
  expert structure could be the family head's gradient rather than the router's own
  behaviour. On language the head becomes a **read-only probe** on a detached
  hidden state, so it can report how linearly decodable the POS family is without
  being able to create that decodability. Its loss weight is documented as
  scientifically inert. **Verify:** `assert not h_probe.requires_grad` at the probe
  input; zeroing vs. setting the probe weight produces **bit-identical** trunk
  gradients (the real test — a detached tensor that still routes gradient through a
  second path would pass a naive `requires_grad` check).
  **Evidence:** both clauses hold, and the second is the one that matters.
  `requires_grad` at the probe input is **False** (TLH.3a — kept because it is cheap,
  not because it is sufficient). **The real test: trunk gradients are BITWISE
  identical at probe weight 0.0, 0.5 and 1.0 — 47 tensors compared with
  `torch.equal`, zero differences** (TLH.3b). So no value of the probe's loss weight
  can move a single trunk gradient, which makes "scientifically inert" a measured fact
  rather than an intention.

  **And the check is not passing because the probe is inert** (TLH.3c): its own
  gradients are exactly zero at weight 0, non-zero at weight 1, and **exactly half**
  the weight-1 values at weight 0.5. The probe learns; the trunk cannot feel it.

  What this buys, stated plainly: on arithmetic `family_cls` sits in the objective at
  weight 0.5 and shapes the trunk, so "MoRE's representation separates operation
  families" is permanently weaker there — some of the observed structure could be that
  head's gradient rather than the router's behaviour, which is why the arithmetic study
  needs the labelled `no_family_supervision` ablation. **Language does not inherit the
  confound at all.** Every routing / Hungarian / AMI / purity number on the language
  arm measures emergent structure with no family signal anywhere in the trunk's
  objective. Two attribute names for two scientific roles, so the distinction cannot be
  lost in a refactor (TLH.3d).

- [x] **T-L5.4 Engine metric namespace and `results.tsv` extension.** Add
  `val/task_loss` in nats, `val_perplexity`, and `depth_rho_model_loss` by
  **appending** columns to `results.tsv` — never reordering, because the existing
  arithmetic readers and the archived files index by position.
  **Verify:** an arithmetic `results.tsv` parses identically before and after; the
  new language columns are populated for language runs and absent-or-`N/A`, never
  `0.0`, for arithmetic runs.
  **Evidence:** checks TL5.4a–e, verified against a `results.tsv` that Gate 5 wrote
  during this session's suite run
  (`runs/t67_provenance_check_seed44__6b711a59__r19`). The **ten pre-L5 columns are
  unchanged in position** — `epoch`, `train_task_loss`, `val_loss`,
  `expert_entropy_normalized`, `avg_depth`, `mean_cos_sim`, `max_cos_sim`,
  `routing_accuracy`, `routing_hungarian_acc`, `routing_ami` — and the three new ones
  are **appended**: `val_perplexity`, `nats_below_bigram_floor`,
  `depth_rho_model_loss`. On the arithmetic row all three read **`N/A`**, never `0.0`:
  `exp()` of an MSE is not a perplexity, and a `0.0` in a perplexity column is a
  fabricated measurement.

  Appended rather than inserted because the archived arithmetic files and every
  existing reader index by **position**, so an insertion would silently reinterpret
  published columns.

  `depth_rho_model_loss` is Phase L-6's measurement. The column is reserved now so the
  header never has to be reordered later, and it reads `N/A` until L-6 computes it —
  the honest value for "not measured yet".

  New W&B keys on the language path only: `val/perplexity`, `val/bits_per_token`, and
  `val/nats_below_bigram_floor`. That last one is the form in which a language loss
  number means anything (T-L2.5), and the floor is read from the **dataset manifest**
  via `MoRELanguageDataset.primary_metric_floor` rather than from a config literal, so
  a run cannot quote a floor measured on a different corpus. Absent, not `0.0`, when
  the dataset supplies none. The probe CE is published as **`probe/family_ce`** on
  language and `train/step_cls_ce` on arithmetic — the same tensor under two names,
  because on one task it is a read-only measurement and on the other it trains the
  trunk.


---

## Phase L-6 — Depth measurement without an invented target  (`plan_language.md` §5)

- [x] **T-L6.0 `depth/allocation_error_*` is `N/A` on language.** There is no
  per-token ground-truth depth for English, and inventing one and then reporting
  agreement with it would be circular — the metric would measure the curriculum,
  not the model. The arithmetic keys must therefore emit `N/A`, not `0.0` and not
  `-1` (§8 sentinel rule). **Verify:** the language `metrics.json` carries `N/A`
  for every allocation-error key; the exporter renders it as `N/A`.

- [x] **T-L6.1 Substitute correlational depth metrics.** `depth/spearman_vs_logfreq`
  (train log-frequency), `depth/spearman_vs_unigram_surprisal` (frozen, from train
  counts), `depth/spearman_vs_model_loss` (the model's own per-token loss), plus
  `depth/mean_by_family` and `depth/hist`. These are non-circular in a way an
  invented curriculum is not: the first two are properties of the training corpus
  fixed before any model runs, and the third is a genuine question — does the model
  spend more computation where it is less certain? **Verify:** each correlation is
  computed over the identical token set, ties handled explicitly, with the token
  count and a permutation-based null band reported next to each rho.

- [x] **T-L6.2 Constant-depth null model is mandatory.** A model that halts at a
  near-constant depth can still produce a nonzero Spearman rho through nothing but
  tie-breaking noise. Reuse `code/depth_null_model.py`'s logic: resample depths
  from the observed marginal while destroying the per-token pairing, and report the
  null band. A depth correlation published without it is uninterpretable.
  **Verify:** the null band is present for every reported rho; a synthetic
  constant-depth run's rho falls inside its own null band.

- [x] **T-L6.3 Re-derive `ffn_mult.mor` for the language arms.** Attention adds
  ~263 K parameters to **all three** arms equally, which shifts the
  parameter-matched MoR solution. `CANONICAL_FFN_MULT` for arithmetic is
  `{moe: 4, mor: 24, more: 4}`; the language value must be recomputed, not copied.
  **Verify:** `abs(P_MoR - P_MoRE) / P_MoRE < 0.05` on **both** the total count and
  the count excluding embeddings — the second is the one that is actually about the
  expert stack, and with a tied 8192×256 embedding the first alone is too easy to
  satisfy.

- [x] **T-L6.4 GATE L3 / GATE L4 — recursion and halting under attention.**
  Attention is the change most likely to break the two invariants that took the
  longest to get right in the arithmetic POC. **Verify (L3):** identical block
  parameters at every depth (by object identity); halted states frozen bitwise;
  forced exit at max depth; balance loss depth-invariant after normalization
  (measure at depth 1 vs canonical and compare). **Verify (L4):** a real gradient
  reaches every halt head under the language loss — `halt_head.weight.grad` is
  non-`None` with nonzero norm for every expert, from `task_loss` alone with the
  ponder weight set to zero, which is the only version of this test that proves the
  task path (not the ponder regularizer) supplies the gradient.

  **Evidence for T-L6.0 – T-L6.4** (one block, because the five share their
  measurements). `code/test_lang_recursion.py` → **11 passed, 0 failed** (GATE L3 +
  GATE L4); Gate L0 unchanged at `TOTAL 356 356 0 0`.

  **T-L6.0.** `val/depth_allocation_error_abs` and `_rel` are written as the string
  `"N/A"` on language rather than left absent. They are *structurally* undefined —
  `lang_families.op_target_depth_table()` is empty by construction, so
  `val_depth_err_n` is 0 and the arithmetic branch never fires — and writing `"N/A"`
  makes the absence a statement. A 0.0 or −1 there would read as perfect allocation
  against a curriculum that does not exist, which §5.1 calls the single most misleading
  number this migration could produce.

  **T-L6.1.** `metrics.spearman_rho` is hand-rolled with **averaged ties** and
  **matches `scipy.stats.spearmanr` to 1.7e-16** over 200 trials, one third of them
  heavily tied. Not a formality: exit depth is an integer, so on a collapsed-depth model
  almost every value is tied, and `argsort().argsort()` would break ties by array
  position and manufacture a confident rho out of batch arrival order — the arithmetic
  POC put 93–97% of tokens at exactly 2 steps. It returns **`None`, not 0.0**, when
  either side is constant, because a 0.0 would read as "measured no relationship". Both
  frozen difficulty vectors derive from a new hash-recorded artifact
  `token_train_count.npy`, so `log_freq` and `unigram_surprisal` cannot disagree about
  smoothing; `log1p` rather than a masked log, because 80 of 8192 canonical types are
  unseen and `log(0)` in a rank vector is a NaN generator rather than an extreme value.

  **T-L6.2 — the null band works, measured on synthetic depth vectors:**

  | depth vector | rho | null band | exceeds? |
  |---|---|---|---|
  | 100% at depth 2 | **None** | — | undefined, correctly |
  | 97% at depth 2 (the POC's regime) | −0.0277 | [−0.0448, +0.0381] | **No** |
  | genuinely correlated | +0.8577 | [−0.0374, +0.0342] | **Yes** |

  So a near-constant allocation's rho is correctly identified as indistinguishable from
  its own marginal paired at random. The null permutes only the pairing, so the tie
  structure that causes the problem is preserved exactly.

  **T-L6.3 — re-derived, and it lands on 24, which is also the arithmetic value.**
  `code/derive_lang_ffn_mult.py`, table in `code/lang_ffn_mult_derivation.json`. MoRE
  reference at E=6, ffn_mult 4: **5,584,908** total / **3,422,220** non-embedding.

  | ffn_mult | MoR total | rel(total) | MoR non-emb | rel(non-emb) | both < 5% |
  |---|---|---|---|---|---|
  | 23 | 5,449,735 | 2.420% | 3,287,047 | 3.950% | yes |
  | **24** | **5,581,063** | **0.069%** | **3,418,375** | **0.112%** | **yes** |
  | 25 | 5,712,391 | 2.283% | 3,549,703 | 3.725% | yes |

  Only 23/24/25 satisfy both criteria and 24 is best on the non-embedding gap. **The
  coincidence with arithmetic is structural, not a copy:** attention (~263 K) and the
  tied 2,097,152-element embedding are added *equally* to every arm and cancel in
  `P_MoR − P_MoRE`, leaving the expert-stack condition 6 experts at mult 4 against 1 at
  mult 24 — unchanged. Note `rel(non-emb) > rel(total)` at every row, which confirms
  this task's premise that the shared embedding makes the total-count criterion too
  easy. Also measured: **completely insensitive to `seq_len`** (128/256/512/1024 all
  give 24), so this field and Gate L5's `seq_len` are independent. Frozen into
  `canonical_spec_language.json` and `config.CANONICAL_FFN_MULT_LANGUAGE`.

  **T-L6.4 — GATE L3 PASSES.** One block object, 5 recursion calls, **1 distinct object
  id** (identity, not tensor equality — a stack of independently-initialised blocks
  could satisfy equality for a single step). Halting suppressed → all tokens force-exit
  at `max_depth` with `early_exits = 0`; halting saturated → all tokens exit at depth 1
  with `forced_exits = 0`. The two extremes bracket the mechanism, so a depth that never
  varied would fail one of them. Balance term across a **9× depth range** (1/3/5/9):
  −0.7400 / −0.7056 / −0.6082 / −0.6082, relative spread **19.8%** — it does not scale
  with the budget, which is the T4.1 property re-measured under attention. Two eval
  passes give bit-identical logits and exit markers, so no halted state is being
  rewritten by a still-active neighbour's attention output.

  **T-L6.4 — GATE L4 PASSES, in the strong form.** With **`ponder_weight = 0`**, so the
  backward pass carries no explicit function of the halt probabilities, every one of the
  six halt heads has a non-`None` weight gradient with nonzero norm:
  `E1 8.42e-03, E2 1.52e-02, E3 1.12e-02, E4 1.04e-02, E5 1.15e-02, E6 5.40e-03`. The
  bias gradients are nonzero too, so the halt threshold is learnable. Separately the
  ponder cost alone also reaches every head (`3.14e-02 … 7.26e-02`), so the two are
  independent paths rather than one wearing two names, and
  `expected_depth.requires_grad` is True — a boolean threshold drives dispatch but the
  objective keeps a differentiable route to the halt head. **The naive version of this
  test — ponder weight left at its canonical value — passes on a model whose task path
  is entirely disconnected from halting**, which is why it is run this way.

- [ ] **T-L6.5 Induced per-block TOPIC partition, as a SECOND reference axis.**
  `plan_language.md` §4.1a. Build `block_topic[n_blocks]` from the
  ` = Title = `-delimited documents `iter_documents` already segments: TF-IDF over
  documents, k-means at **k = 6** with a recorded seed, then each packed block takes
  the topic of the document that supplies most of its tokens. Six so the topic axis is
  directly comparable with the POS axis and with the arithmetic router width.
  **Labelled induced, not ground truth** — WikiText ships no topic labels, so this is
  our construction and is reported as such, exactly like the shuffled control.
  **Verify:** the label array has one entry per stored block; the six clusters are
  non-degenerate (no cluster below 2% of blocks); the top TF-IDF terms per cluster are
  printed and are humanly recognisable as topics; a second run at the same seed
  reproduces the assignment byte-identically.

- [ ] **T-L6.6 Article-level routing agreement, beside POS and against its own null.**
  Score the learned partition against the topic axis with the same machinery T-L3.3
  built: `specialization_vs_control` takes any partition, so this needs a per-block
  variant (`agreement = did this article's tokens concentrate on one expert`) plus a
  marginal-matched shuffled control for the topic labels. Reported BESIDE
  `routing_agreement_with_pos`, never instead of it, so the paper answers "what does a
  small recursive MoE organize by" rather than presupposing POS. **Verify:** both
  agreement numbers and both null bands appear for every language run; a synthetic
  router that routes by topic scores above the topic null and near the POS null, and
  vice versa — i.e. the two axes are shown to be separable before either is quoted.

- [ ] **T-L6.7 Span-level depth reporting.** `plan_language.md` §4.1b. Per-token
  recursion is kept, but "does the model spend more computation on harder *passages*"
  is answered as a measurement: `depth/mean_by_document`, and the within-document
  versus between-document variance of exit depth. If between-document variance is a
  negligible share of the total, the model is not allocating at the passage level and
  that is the honest finding. **Verify:** the variance decomposition sums to the total
  depth variance; document boundaries come from the stored EOT positions, not from a
  re-segmentation.

---

## Phase L-7 — Calibration and protocol freeze  (`plan_language.md` §10, Gate L5)
- [ ] **T-L7.0 Measure throughput and memory on the RTX 3050 (6 GB).** The
  arithmetic protocol's `batch_size = 768` is meaningless for `[B, S, V]` logits;
  6 GB is the binding constraint and must be measured, not estimated. Sweep batch
  size and `seq_len` on wikitext-2 for a fixed small step count, recording
  tokens/s, peak VRAM, and whether the step OOMs. **Verify:** a recorded table of
  (batch, seq_len) → tokens/s, peak MiB, OOM yes/no, taken with the CUDA
  interpreter on the real GPU.

- [ ] **T-L7.1 Freeze the language protocol in one commit.** `seq_len`,
  `vocab_size`, `batch_size`, `lr`, `weight_decay`, `epochs`, `max_depth`,
  `num_blocks`, `d_model`, `n_heads`, `ffn_mult` per arm, and the loss weights are
  written into `code/canonical_spec_language.json` **together**, in a single
  commit, from the T-L7.0 measurements. They start `null` precisely so the existing
  proxy guard refuses every canonical claim until this task runs — that refusal is
  the mechanism working, not a bug to route around. One resolved protocol for all
  three arms (§ canonical hyperparameter principle); the only per-arm difference is
  what is needed to instantiate MoE/MoR/MoRE plus the T-L6.3 `ffn_mult`.
  **Verify:** no `null` remains in the spec; a language run with the frozen config
  is accepted as canonical; the same run with any single protocol field perturbed
  is **refused**.

- [ ] **T-L7.2 GATE L6 / GATE L7 — parameter budget and floor.**
  **Verify (L6):** the MoR/MoRE parameter difference is < 5% on both the total and
  the non-embedding count, and the three arms' counts are published as a table.
  **Verify (L7):** a single short language run beats `primary_metric_floor` from
  T-L2.5. If it does not, the architecture matrix is not yet worth running and the
  gate-failure protocol applies — that is the whole point of having a floor.

- [ ] **T-L7.3 Input-parity test across the three arms.** §4's bit-identical-input
  invariant is trivially checkable on language because all three arms consume the
  same `input_ids`: assert the tensors are equal for the same record index under
  all three configs. **Verify:** `torch.equal` on the input batch across MoE/MoR/
  MoRE at the same seed and index.

- [ ] **T-L7.4 One-command runner for the 4060 8 GB co-author machine.** The best
  GPU available to the project is Ayan's RTX 4060 Laptop (8 GB), not this machine's
  3050 (6 GB), so the canonical matrix should run there. What must exist before that
  hand-off: a `SETUP.md` naming the exact environment build (the `--system-site-packages`
  + `--ignore-installed torch==2.6.0 --index-url .../cu126` recipe from T-L0.1, which is
  the step most likely to be got wrong), a **dataset rebuild command** because
  `data/lang/**/*.npy` is not tracked, and a single entry point that runs the whole arm
  and writes only into `runs/<experiment_id>/`. W&B needs settling first — a canonical
  run will die in `wandb.init` with `UsageError: No API key configured` exactly as
  T-L1.4's re-run did, and it will leave a directory-shaped artifact behind when it does.
  **Verify:** on a machine that has never seen this repo, `git clone` + the documented
  setup + **one** command reproduces the wikitext-2 dataset (hashes matching the tracked
  manifest) and completes a short language run; the 8 GB VRAM figure replaces the 6 GB
  one in the T-L7.0 table for the arms actually run there.

- [ ] **T-L7.5 Push after every gate that a REAL RUN validated.** Standing
  instruction, recorded here because it is a protocol rule rather than a one-off: the
  branch is pushed to `origin` after each gate whose pass depends on an executed run,
  not on a pre-written assertion alone. Gate L0 qualifies (Gate 5 launches two real
  1-epoch training runs); Gates L1/L2/L3 are static and do not by themselves trigger a
  push. **Verify:** `git log origin/<branch>..HEAD` is empty immediately after each
  run-validated gate passes.

---

## Phase L-8 — Canonical language matrix  (`plan_language.md` §9)

- [ ] **T-L8.0 15 canonical runs: {MoE, MoR, MoRE} × seeds {42,43,44,45,46}.**
  Attention in all three arms; one frozen protocol; `langB_*` run names; per-run
  directories. Checkpoint selection is the **final epoch**, per T11.0b — best-val
  is an order statistic whose downward bias scales with each arm's validation
  noise, so with unmatched noise across arms it silently favours the noisiest one.
  **Verify:** 15 `runs/langB_*` directories each with `config.json`,
  `resolved_config.json`, `metrics.json`, `results.tsv`, `checkpoint.pt`,
  `stdout.log`; every one stamped with the full §9 provenance list; all 15 sharing
  one `config_hash` modulo architecture and seed.

- [ ] **T-L8.1 Seed statistics via `seed_stats.py`.** mean ± std per arm for
  val task loss, perplexity, average depth, normalized entropy, Hungarian/AMI/
  purity, the depth correlations, and throughput. Then the exact two-sided
  randomization test between arms with Bonferroni correction, and the resolution
  floor stated: at 5-vs-5 the smallest attainable two-sided p is 0.0040, so no
  pairwise claim can be made below it no matter how large the effect looks.
  **Verify:** the statistics are computed by the existing script from the run
  directories — never hand-copied — and the floor is printed alongside every p.

- [ ] **T-L8.2 GATE L8 — provenance.** **Verify:** every figure and table in the
  language results traces to explicit run ids; the exporter refuses a mixed
  `dataset_version` or `config_hash`; no arithmetic run appears in a language table
  and vice versa; nothing invalidated is silently reused.

---

## Phase L-9 — Ablations  (`plan_language.md` §9, A–G)

Each is a labelled variant, never a canonical claim. Run at a reduced seed set
where compute forces it, and say so in the caption rather than presenting a
1-seed ablation as if it carried the canonical error bars.

- [ ] **T-L9.A Fixed depth vs adaptive depth.** Isolates whether adaptivity does
  anything, holding total capacity fixed. **Verify:** `fixed_depth` in the variant
  string; matched protocol otherwise.
- [ ] **T-L9.B Learned routing vs POS-oracle routing.** Upper-bound/capacity
  diagnostic only; must never replace the learned-router result in the headline
  (§ required ablations). **Verify:** `oracle_routing` in the variant string; the
  headline table contains only the learned-router run.
- [ ] **T-L9.C One recursive block vs two.** **Verify:** `num_blocks_2` in the
  variant string; balance loss still depth- and block-invariant at 2 blocks.
- [ ] **T-L9.D Parameter-matched MoR.** Uses the T-L6.3 `ffn_mult`. **Verify:** the
  < 5% bound holds on both counts and is restated in the caption.
- [ ] **T-L9.E Dense routing vs canonical top-1.** **Verify:**
  `dense_routing_ablation` appears in the run label — `enforce_routing_mode`
  refuses `dense_blend` without it, so this is enforced, not remembered.
- [ ] **T-L9.F Supervised frequency-decile curriculum.** Uses T-L2.6's table.
  **Verify:** `supervised_curriculum` in the variant string; the caption states the
  curriculum is a design choice, and no canonical run has it enabled.
- [ ] **T-L9.G POS vs shuffled control.** Evaluation-time only (T-L3.3), zero extra
  training runs. **Verify:** both metric sets reported for every language run.

- [ ] **T-L9.H Router noise — deferred by default.** `[✗]` unless a measured
  collapse motivates it. Canonical is `router_noise = none` (§7) and the rules
  forbid adding several routing regularizers at once; adding noise without an
  observed pathology to fix would be tuning. **Narrowing decision to record here
  if closed:** the measured normalized entropy and `max_load_fraction` that show no
  collapse, so a future reader sees the evidence rather than the omission.

---

## Phase L-10 — Reporting  (`plan_language.md` §14)

- [ ] **T-L10.0 `results/language/{results.csv,results.json,results.md}`.** All 17
  sections from `updated_objective.md`, with the three language-specific foldings:
  allocation error replaced by the correlational depth section; routing accuracy
  replaced by agreement-plus-permutation-invariant metrics with the shuffled
  control; the trivial-baseline floor stated before any loss number.
  **Verify:** all 17 sections present; every number traceable to a run id; measured
  facts and interpretation visibly separated.

- [ ] **T-L10.1 Paper-safe claims and explicitly unsupported claims.** Write both
  lists. Specifically forbidden (§13): calling the depth correlations "complexity
  discovery"; calling POS agreement "routing accuracy"; comparing this perplexity
  to published WikiText numbers (different vocabulary, different tokenizer,
  different parameter scale — the comparison is meaningless in both directions);
  claiming orthogonality from cosine similarity. **Verify:** both lists exist and
  every headline sentence in `results.md` appears in the paper-safe list.

- [ ] **T-L10.2 Cross-task comparison, carefully.** The arithmetic study concluded
  **Outcome C** with MoR marginally best (MoR 0.062706 ± 0.000112 vs MoRE
  0.063186 ± 0.000249 vs MoE 0.063260 ± 0.000153). Whether language reproduces that
  ordering is the interesting question, and it must be reported as two separate
  studies on one engine — not merged, not averaged, not presented as a trend across
  two points. **Verify:** the section names both dataset versions and both config
  hashes and makes no claim that requires the two to be commensurable.

- [ ] **T-L10.3 Update README and ARCHITECTURE for two tasks.** README gains a
  language quickstart; ARCHITECTURE gains the `task` axis, the attention sublayer,
  and the language data contract. **Verify:** a reader following README alone can
  build the dataset and launch one language run.

- [ ] **T-L10.4 Final go/no-go.** State which of Outcome A/B/C the language study
  supports, on the evidence, with the resolution floor and the seed variance in
  view. Outcome C is a valid result and is reported without hedging if that is what
  the numbers say. **Verify:** the assessment names the specific measurements it
  rests on.

---

## Standing obligations (not phase-ordered)

- [ ] **T-LX.0 Gate L0 after every language commit.** `run_correctness_suite.py`
  must still print `TOTAL 356 356 0 0` (T-L0.3). The arithmetic result is
  published; a
  language change that breaks it has broken a finished study.
- [ ] **T-LX.1 One `changelog.md` entry per completed task**, each naming the code
  location to open when that area misbehaves (CLAUDE.md).
- [ ] **T-LX.2 Never re-run or delete arithmetic artifacts.** `runs/phaseB_*`,
  `archive/`, `data/dataset_meta.json` and the arithmetic results are evidence.
- [ ] **T-LX.3 Keep this ledger current in the same commit as the code.** A ticked
  box with no evidence line, or evidence in chat only, is the failure mode this
  file exists to prevent.

- [ ] **T-LX.4 Oracle routing is ABLATION-ONLY and must never enter a canonical
  language run.** It existed in the micro-POC to prove MoRE could route at all, by
  hard-coding each token's expert; in a real run the router learns from the loss alone.
  Already enforced rather than merely intended: `canonical_spec_language.json` pins
  `step_routing_weight = 0.0` in **all three** architecture blocks, and
  `config.resolve_variant` tags any non-zero value `oracle_routing`, which is not equal
  to `language` and is therefore refused the canonical group by the proxy guard. §7.4
  notes the ablation is *more* interesting on language than on arithmetic — it measures
  how much achievable loss is given up by forcing a POS partition — and it still may
  never share a table with canonical runs. **Verify:** all three arms show
  `step_routing_weight = 0.0`; a run with a non-zero value is refused
  `canonical_lang_b`.

- [ ] **T-LX.5 `N/A` stays in the machine record; the PAPER renderer drops
  all-`N/A` columns.** These are different requests and conflating them would undo a
  fix. `metrics.json` and `results.tsv` must keep `"N/A"`, because an exporter joining
  MoE/MoR/MoRE rows on a common column set has to decide what a *missing* column means —
  that is the defect the `"N/A"` contract closed (see the changelog entry on the eleven
  flat routing keys), and CLAUDE.md §4 forbids the 0.0 that would otherwise fill the
  gap. What should not carry dead columns is the human-facing table: the language
  `results.md` writer drops any column that is `N/A` for **every** row in the table it is
  rendering, and says in a footnote which ones it dropped and why. **Verify:** a
  language `results.tsv` still contains `depth_allocation_error_*` as `N/A`; the
  rendered language results table does not show them, and names them in a footnote.

