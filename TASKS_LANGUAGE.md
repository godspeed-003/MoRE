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

- [ ] **T-L1.0 `task` in the config layer.** Add `task = arithmetic | language`
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

- [ ] **T-L1.1 `task` in the variant string and run naming.** `resolve_variant`
  must not silently label a language run `canonical`. Extend it so the task
  appears in the variant when it is not arithmetic, and add
  `CANONICAL_RUN_PREFIX_LANGUAGE = "langB"` so run names are
  `langB_MoE_seed42` (§10) and can never be confused with `phaseB_*` in W&B, in
  `runs/`, or in an exporter table. **Verify:** an unmodified language config
  resolves to a variant naming the task; `canonical_run_name` for
  `(language, moe, 42)` returns `langB_MoE_seed42`.

- [ ] **T-L1.2 `--task`, `--seq_len`, `--vocab_size` on the CLI.** Add the flags
  in `cli.py` and place `apply_task` immediately after `apply_architecture` and
  before the override blocks, so an explicit `--family_cls` on a language run
  hits the same refuse-don't-ignore path as every other override rather than
  being quietly overwritten. **Verify:** `--task language --family_cls 0.5` exits
  non-zero with a refusal naming §7.3; `--task language` alone succeeds.

- [ ] **T-L1.3 Task-selected canonical spec.** `run_context.load_canonical_spec`
  takes a default `path=CANONICAL_SPEC_PATH`; make the selection task-based so
  `language` reads `code/canonical_spec_language.json`. `code/canonical_spec.json`
  and its three other readers (`export_results.py`,
  `test_phase6_provenance.py`, `test_phase6_seeding.py`) are untouched.
  **Verify:** an arithmetic run still loads the arithmetic spec (assert on the
  loaded dict's identity fields); a language run loads the language spec; a
  language run declaring `experiment_group = canonical_phase_b` is **refused**
  (wrong group for the task).

- [ ] **T-L1.4 GATE L0 after the task axis.** The task axis is the change most
  likely to silently alter arithmetic behaviour, so the gate runs here before any
  language code exists. **Verify:** `run_correctness_suite.py` still prints
  `TOTAL 356 356 0 0` (the T-L0.3 baseline); arithmetic parameter counts for all
  three arms are
  unchanged from T-L0.3; a 1-epoch arithmetic run's `metrics.json` matches the
  pre-change run on `task_loss` to full printed precision at the same seed.

---

## Phase L-2 — Data pipeline  (`plan_language.md` §3)

- [ ] **T-L2.0 Fetch WikiText.** `data/lang/build_language_dataset.py` downloads
  `wikitext-103-raw-v1` (canonical) and `wikitext-2-raw-v1` (dev loop) via
  `datasets`. Raw variants only: the non-raw ones are pre-tokenized with `<unk>`
  substituted, which would put an artificial high-frequency type in the middle of
  the frequency distribution the depth correlations are measured against. The
  author-provided splits are **document-disjoint**, which is what makes the
  leakage audit meaningful. **Verify:** the three splits load, their line counts
  and byte sizes are recorded in the manifest, and no split contains the literal
  token `<unk>`.

- [ ] **T-L2.1 Byte-level BPE, V=8192, fitted on train only.** Two independent
  reasons, both recorded in the script's docstring: (a) fitting a tokenizer on
  val/test is tokenizer-level leakage — the merge table would encode the held-out
  text's statistics; (b) a stock GPT-2 vocabulary would put 50257×256 = 12.9 M
  parameters in the embedding against a 3.2 M expert stack, so the §6 parameter
  budget would be satisfied by the embedding alone and the MoR/MoRE match would
  be measuring nothing. Byte-level so there is no OOV and no `<unk>`.
  **Verify:** the tokenizer round-trips a held-out paragraph exactly; the fitted
  vocab size is exactly 8192; the training corpus passed to the trainer is the
  train split alone (assert on the file list, not on intent).

- [ ] **T-L2.2 Pack to fixed-length blocks, drop the trailing partial.** Encode
  each split into one stream, cut into `seq_len`-length blocks, and **discard the
  final short block**. This is not laziness: an all-True `step_mask` means no
  `key_padding_mask`, which means the NaN path in attention over a fully-masked
  row is unreachable and the ACT denominators are exact rather than
  mask-conditional. Store as `.npy` `uint16` (V=8192 fits) and memory-map at load
  — not JSONL, which would re-parse ~100 M tokens every epoch.
  **Verify:** every stored block has length exactly `seq_len`; the dropped tail is
  `< seq_len` tokens and its size is recorded in the manifest;
  `np.load(mmap_mode="r")` opens without materializing the array.

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

- [ ] **T-L3.0 Six POS families in `code/more/lang_families.py`.** `L1 FUNCTION,
  L2 NOUN, L3 VERB, L4 MODIFIER, L5 PUNCT_SYM, L6 NUM_SUBWORD`, mirroring
  `families.py`'s structure so `metrics.py` needs no new label plumbing. Six to
  keep the MoE axis width identical to arithmetic, so any difference between the
  two studies is not a difference in expert count. **Verify:** labels contain no
  `/` (W&B nesting rule); `NUM_FAMILIES == 6`; the module exposes the same
  accessor names `metrics.py` already imports from `families.py`.

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
