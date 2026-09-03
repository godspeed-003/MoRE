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
  | wikitext-2 | train | 36,718 | 31,175 | 10,914,845 | **0** |
  | wikitext-2 | val | 3,760 | 3,213 | 1,144,248 | **0** |
  | wikitext-2 | test | 4,358 | 3,760 | 1,287,656 | **0** |
  | wikitext-103 | train | 1,801,350 | 1,165,029 | 539,295,549 | **0** |
  | wikitext-103 | val | 3,760 | 3,213 | 1,144,248 | **0** |
  | wikitext-103 | test | 4,358 | 3,760 | 1,287,656 | **0** |

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

- [ ] **T-L2.3 `MoRELanguageDataset` in `code/more/lang_data.py`.** Returns the
  same-arity tuple shape the engine already consumes, with the language meanings:
  `input_ids [seq_len] long`, `step_mask [seq_len] bool` (all True),
  `step_experts [seq_len] long` from the POS lookup (`-1` = unmapped → ignore),
  `step_ops` = `N/A` for language, plus the LM target derived by shift inside the
  loss rather than stored twice. All shuffling is the DataLoader's; the stored
  block order is fixed so the manifest hash is stable. **Verify:** a batch has the
  documented dtypes and shapes; two DataLoader epochs at the same seed produce the
  same permutation, and different seeds produce different ones.

- [ ] **T-L2.4 Manifest with token- and type-level statistics.** Mirror
  `data/dataset_meta.json`'s schema: dataset version, tokenizer hash, per-split
  block counts and token counts, per-split SHA-256, `family_counts` at **both**
  the type level (how many vocabulary entries per family) and the token level (how
  many corpus occurrences per family), the unmapped share, and the frequency-decile
  table. Both levels are needed because a family can be 2% of types and 40% of
  tokens; a load-balance number read against the wrong denominator is
  uninterpretable. **Verify:** the manifest validates against the documented
  schema and its split hashes reproduce on a second build from the same seed.

- [ ] **T-L2.5 Trivial-baseline floors.** Compute, from the train split only and
  evaluated on val: `uniform_ce = ln(8192) ≈ 9.0109`, `unigram_ce`, and a
  Katz-backoff `bigram_ce`. The lowest of these is `primary_metric_floor` and is
  written into the manifest. Without it a val loss of 5.4 nats/token is a number
  with no meaning. **Verify:** all three are finite, ordered
  `uniform ≥ unigram ≥ bigram`, and the unigram/bigram models are fitted on train
  counts alone (assert the fit never touches the val arrays).

- [ ] **T-L2.6 Frequency-decile depth table — built, and deliberately unused.**
  Produce a `token_target_depth[V]` table from train frequency deciles so
  ablation F (`supervised_curriculum`) is runnable, and write into the module
  docstring that it is **not** part of the canonical configuration and is a design
  choice rather than a measurement. `plan_language.md` §5 is explicit: canonical
  language runs invent no depth target. **Verify:** the table exists with one entry
  per vocabulary id; the canonical language config's
  `loss_weights.halting_supervision` is `0.0` and `halt_target_mode` is the pure
  ACT mode.

- [ ] **T-L2.7 GATE L2 — dataset integrity.** **Verify:** no train/val/test block
  overlap by exact-content hash; the tokenizer was fitted on train only; every
  block is full-length; the unmapped-token share is ≤ 2% (§4.3 budget) and its
  measured value is published in the manifest; the trivial floors are present and
  finite; the manifest hash is reproducible.

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


- [ ] **T-L3.1 Type-level `token_family[V]` from majority POS.** Tag the train
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

- [ ] **T-L3.2 Demote routing accuracy to `routing_agreement_with_pos`.**
  STRICTER THAN ARITHMETIC (§4.4). On arithmetic, `OP_TO_EXPERT` is ground truth:
  the family genuinely is the right answer. A POS partition is a linguistic prior,
  not a functional one — nothing says the optimal expert split for next-token
  prediction is noun-vs-verb. So the key is renamed to state what it measures, and
  the permutation-invariant metrics (Hungarian, AMI, purity) become **primary**
  for language rather than supplementary. **Verify:** no language metric key is
  named `routing_accuracy`; `results.md` for language reports Hungarian/AMI/purity
  in the specialization section before any POS-agreement number; the agreement
  number's caption states it is agreement with a prior, not accuracy.

- [ ] **T-L3.3 Mandatory shuffled-control partition (ablation G).** Build a second
  `token_family_shuffled[V]` by permuting the family assignment across types while
  holding the per-family type counts fixed, and evaluate the same
  agreement/Hungarian/AMI/purity metrics against it. This is evaluation-time only
  and costs no extra training runs. Its purpose: if the learned partition scores as
  well against a meaningless partition as against POS, then the POS number was
  measuring the metric's floor, not specialization. **Verify:** the shuffled
  control has identical per-family type counts to the real partition; both metric
  sets appear side by side for every language run; the difference is reported with
  its uncertainty, not just the raw pair.

---

## Phase L-4 — Attention  (`plan_language.md` §6)

- [ ] **T-L4.0 One shared causal-attention sublayer per `MoREWrapper`.** The model
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

- [ ] **T-L4.1 `attention=False` constructs no module.** Follow the
  `router_noise_scale` precedent exactly: when the flag is off, the attribute is
  absent rather than present-and-unused, so the arithmetic `state_dict`, the
  arithmetic parameter counts, and Gate 4 are provably untouched rather than
  merely believed to be. **Verify:** an arithmetic `MoREWrapper`'s `state_dict()`
  key set is identical to the pre-change key set; `hasattr(wrapper, "attn")` is
  False; arithmetic parameter totals match T-L0.3.

- [ ] **T-L4.2 Explicit upper-triangular bool mask; `is_causal=True` forbidden.**
  Build the mask explicitly and pass it as `attn_mask`. `is_causal` is a *hint* in
  PyTorch's API: whether it is honoured depends on the backend kernel selected at
  runtime, so a shape/backend change could silently make the model
  non-causal — i.e. silently leak the target — with no error. An explicit mask is
  checked by the same code path on every backend. **Verify:** the mask is
  materialized as a bool tensor with the documented triangle orientation; a grep
  shows no `is_causal=True` anywhere in the language path.

- [ ] **T-L4.3 Halted tokens stay attendable but frozen.** A halted position must
  remain visible as a *key/value* to still-active later positions (removing it
  would change the context of tokens that have not halted, coupling their
  predictions to unrelated tokens' halting decisions), while its own state is
  frozen and never recomputed (§2.2). Attention outputs are therefore computed at
  halted query positions and discarded; that waste is real and must be
  **reported**, not hidden, as `route_stats.attn_query_waste_fraction`.
  **Verify:** freezing holds bitwise — a halted position's state at depth `d+1`
  equals its state at depth `d`; the waste fraction is in `metrics.json` and is a
  measured value, never a sentinel.

- [ ] **T-L4.4 GATE L1 — causality and leakage.** The single most important gate
  in the language migration: it is the direct analogue of the arithmetic
  target-in-input leakage audit. Perturb `input_ids[b, t+k]` for `k ≥ 1` and assert
  the logits at position `t` are **bit-identical**. Run it at multiple `t` and `k`,
  for all three architectures, and at both `max_depth = 1` and canonical depth,
  because the ACT loop is the place a future-position value could reach the past
  through the halting statistics. **Verify:** `code/test_lang_causality.py` passes;
  a deliberately broken variant (mask removed) **fails** it — a leakage test that
  has never failed has not been shown to be able to.

---

## Phase L-5 — Heads and the language loss  (`plan_language.md` §7)

- [ ] **T-L5.0 Token embedding + weight-tied LM head.** `nn.Embedding(V, d_model)`
  on input, and an output projection whose weight **is** the embedding weight.
  Tying halves the embedding parameter cost and keeps the §6 parameter budget a
  statement about the expert stack rather than about two independent V×d_model
  tables. **Verify:** the two weights are the same tensor object (`is`); the tied
  head's gradient accumulates from both the input and output paths.

- [ ] **T-L5.1 Causal-LM cross-entropy as `task_loss`, perplexity alongside.**
  `task_loss = F.cross_entropy(logits[:, :-1].reshape(-1, V),
  input_ids[:, 1:].reshape(-1))` in **nats/token**; `perplexity = exp(task_loss)`
  reported and never optimized. Nats because every other loss term in the assembly
  is in nats and a bits/nats mix in one weighted sum is a silent scaling bug.
  **Verify:** at initialization `task_loss ≈ ln(8192) ≈ 9.011` (the uniform floor —
  a strong check that the shift, the reshape and the vocabulary agree); perplexity
  equals `exp(task_loss)` to floating-point tolerance.

- [ ] **T-L5.2 Retire the regression head on the language path.** `regression_head`
  and the masked mean-pool exist for the scalar arithmetic answer and have no
  meaning for next-token prediction. They must be absent on the language path, not
  merely unused with a zero weight — an unused head still contributes parameters to
  the budget comparison and still invites a future reader to weight it.
  **Verify:** a language model's `state_dict` contains no `regression_head.*` key;
  an arithmetic model's still does.

- [ ] **T-L5.3 `family_probe` reads `h.detach()`.** STRICTER THAN ARITHMETIC
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

- [ ] **T-L5.4 Engine metric namespace and `results.tsv` extension.** Add
  `val/task_loss` in nats, `val_perplexity`, and `depth_rho_model_loss` by
  **appending** columns to `results.tsv` — never reordering, because the existing
  arithmetic readers and the archived files index by position.
  **Verify:** an arithmetic `results.tsv` parses identically before and after; the
  new language columns are populated for language runs and absent-or-`N/A`, never
  `0.0`, for arithmetic runs.

---

## Phase L-6 — Depth measurement without an invented target  (`plan_language.md` §5)

- [ ] **T-L6.0 `depth/allocation_error_*` is `N/A` on language.** There is no
  per-token ground-truth depth for English, and inventing one and then reporting
  agreement with it would be circular — the metric would measure the curriculum,
  not the model. The arithmetic keys must therefore emit `N/A`, not `0.0` and not
  `-1` (§8 sentinel rule). **Verify:** the language `metrics.json` carries `N/A`
  for every allocation-error key; the exporter renders it as `N/A`.

- [ ] **T-L6.1 Substitute correlational depth metrics.** `depth/spearman_vs_logfreq`
  (train log-frequency), `depth/spearman_vs_unigram_surprisal` (frozen, from train
  counts), `depth/spearman_vs_model_loss` (the model's own per-token loss), plus
  `depth/mean_by_family` and `depth/hist`. These are non-circular in a way an
  invented curriculum is not: the first two are properties of the training corpus
  fixed before any model runs, and the third is a genuine question — does the model
  spend more computation where it is less certain? **Verify:** each correlation is
  computed over the identical token set, ties handled explicitly, with the token
  count and a permutation-based null band reported next to each rho.

- [ ] **T-L6.2 Constant-depth null model is mandatory.** A model that halts at a
  near-constant depth can still produce a nonzero Spearman rho through nothing but
  tie-breaking noise. Reuse `code/depth_null_model.py`'s logic: resample depths
  from the observed marginal while destroying the per-token pairing, and report the
  null band. A depth correlation published without it is uninterpretable.
  **Verify:** the null band is present for every reported rho; a synthetic
  constant-depth run's rho falls inside its own null band.

- [ ] **T-L6.3 Re-derive `ffn_mult.mor` for the language arms.** Attention adds
  ~263 K parameters to **all three** arms equally, which shifts the
  parameter-matched MoR solution. `CANONICAL_FFN_MULT` for arithmetic is
  `{moe: 4, mor: 24, more: 4}`; the language value must be recomputed, not copied.
  **Verify:** `abs(P_MoR - P_MoRE) / P_MoRE < 0.05` on **both** the total count and
  the count excluding embeddings — the second is the one that is actually about the
  expert stack, and with a tied 8192×256 embedding the first alone is too easy to
  satisfy.

- [ ] **T-L6.4 GATE L3 / GATE L4 — recursion and halting under attention.**
  Attention is the change most likely to break the two invariants that took the
  longest to get right in the arithmetic POC. **Verify (L3):** identical block
  parameters at every depth (by object identity); halted states frozen bitwise;
  forced exit at max depth; balance loss depth-invariant after normalization
  (measure at depth 1 vs canonical and compare). **Verify (L4):** a real gradient
  reaches every halt head under the language loss — `halt_head.weight.grad` is
  non-`None` with nonzero norm for every expert, from `task_loss` alone with the
  ponder weight set to zero, which is the only version of this test that proves the
  task path (not the ponder regularizer) supplies the gradient.

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
