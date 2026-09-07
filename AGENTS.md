# AGENTS.md — MoRE Repository Operating Rules

**Authority.** This file is the working contract for all agent work in this
repository. It condenses three governing documents, which remain authoritative
in case of any conflict:

| Document | Role |
|---|---|
| [plan.md](plan.md) | The execution plan. Phases 0–11, gates, immediate step order. |
| [updated_rules.md](updated_rules.md) | Architectural and scientific constraints. Non-negotiable. |
| [updated_objective.md](updated_objective.md) | What is being tested and what counts as success. |

Read [ARCHITECTURE.md](ARCHITECTURE.md) first for codebase orientation — repo
layout, the one-engine/three-modes design (and why there is no `moe/` or `mor/`
folder), the data contract, gate status, and the current known-defect list. It is
descriptive; the three documents above remain authoritative.

Superseded and **not** to be used: `archive/pre_finalization/sweep_results/rules.md`,
`archive/pre_finalization/sweep_results/objective.md`, and every result in
`archive/`. In particular the old primary criterion `expert_entropy > baseline`
is void — high entropy is a load-balance diagnostic only.

Task tracking: [TASKS.md](TASKS.md). Do not tick a box without the stated
technical verification actually passing.

Change history: [changelog.md](changelog.md) — one entry per completed task, each
naming the code location to open when that area misbehaves. **Append an entry
whenever you complete a task**, and consult it before changing routing, halting,
the loss assembly or the metric layer: it records the traps already hit there
(guards landing in dead code, label checks running before override resolution,
sentinels reported as measurements).

---

## 1. Terminology — never conflate these

- **MoE** — multiple independent expert FFNs, learned **Top-1 sparse** token
  routing. Answers *which computation module?*
- **MoR** — one shared block reused across recursion steps with adaptive
  halting. Answers *how much computation?*
- **MoRE** — a token is routed to a specialized expert, and that expert's
  weights are recursively reused for an adaptive number of steps. Answers both.

Do not collapse "which expert" and "how much depth" in variable names,
comments, plots, or claims.

---

## 2. Hard architectural constraints

**Routing.** Canonical MoE/MoRE path is `Linear router -> softmax -> argmax ->
dispatch to that expert only -> optional multiply by the selected gate prob`.
Dense evaluate-all-and-blend may exist **only** as an explicitly labelled
"MoRE – Dense Routing Ablation" and must never become the canonical path.

**Expert independence.** Experts are independent FFNs. No cross-expert
communication or expert-specific skips except as a labelled ablation.

**Weight sharing.** The *same* block parameters at every recursion depth. Depth
is computation through time, not a stack of depth-specific networks.

**Halting.** Differentiable ACT / Universal-Transformer style. Required:
learned per-step halt probability; early exit for active tokens; halted states
frozen and not recomputed; forced exit at max depth; a differentiable
ponder/recursion cost; **a real gradient reaching the halt parameters.** A
boolean threshold may drive dispatch, but the objective must keep a
differentiable path to the halt head.

**Balance loss.** Required to discourage collapse; must never become the
primary objective. Must be **normalized so its magnitude does not grow with
recursion depth or block count**. Report the entropy term and the Switch-style
auxiliary term separately. Never tune blindly toward maximal entropy.

**Expert count.** Canonical is **six**: `E1 ADD/SUB`, `E2 MULT/DIV`,
`E3 MOD/POW`, `E4 LOGIC`, `E5 SHIFT`, `E6 SORT/STAT`. The E7 catch-all is
removed. No hard-coded 7-class heads. All tensor dims derive from
`num_experts`; all labels derive from one family manifest; an unmapped
operation must **raise**, never fall back.

**Router noise.** Canonical default `router_noise = none`. Noise variants are
ablations only. Do not add router z-loss to canonical. The old L2-on-trainable-
noise-scale is not a proven fix and must not be described as one.

---

## 3. Input integrity — prohibited in the canonical representation

```text
expert_id / family_id as a numerical input feature
oracle routing class encoded into an input slot
the regression target (final answer / step result) embedded in input features
```

Operation *identity* is legitimate task information — encode it as an operation
embedding, not as the oracle expert index. Oracle labels stay available for
diagnostics and labelled oracle-routing experiments only.

**Input features must be bit-identical across MoE, MoR and MoRE for the same
record.** This is a testable invariant, not an aspiration.

---

## 4. Metric rules

- **Never** report `total_loss` as the primary quality metric. Track `task_loss`,
  `classification_loss`, `routing_balance_loss`, `entropy_term`,
  `switch_aux_term`, `halting_loss`/`ponder_cost`, `routing_supervision_loss`,
  and `weighted_total_loss` separately.
- **Primary predictive metric: validation task loss.** A negative weighted total
  is not automatically a bug but is never evidence of quality.
- **Routing accuracy** is authoritatively `mean(predicted_expert == oracle_expert)`
  over exactly the tokens the confusion matrix uses. The confusion diagonal
  fraction must equal it within tolerance. Also report per-family
  precision/recall, macro accuracy, **Hungarian-matched accuracy**, AMI, and
  cluster purity — expert indices carry no semantic identity, so
  permutation-invariant metrics are mandatory.
- **Entropy** is reported normalized: `H / log(E)`. For `E = 1` entropy is
  `N/A`, never `0` used as a comparative score.
- **Cosine similarity**: report mean *and* max pairwise when `E > 1`. Phrase as
  "consistent with differentiated parameterizations" — **not** "proves
  orthogonality".
- **Depth**: report average depth, per-operation average depth, absolute and
  relative **depth allocation error**, early-exit rate, forced-exit rate. Call
  it *depth allocation error*, never "compute efficiency", unless real FLOPs are
  in the metric.
- **No sentinel value may be reported as a measurement.** `-1`, `0.0`
  placeholders, etc. must be `N/A`.
- Do not claim the model "discovers intrinsic mathematical complexity". Correct
  phrasing: *the model learns to allocate recursion depth in accordance with the
  predefined operation-complexity curriculum.*

---

## 5. Reproducibility, provenance and output layout

- Seeds `42, 43, 44, 45, 46`; report **mean ± std**. One favourable seed is
  never evidence.
- Global seeding is required (`torch`, `numpy`, `random`, CUDA), not just the
  `random_split` generator.
- **Per-run output directories only** — `runs/<experiment_id>/` with
  `config.json`, `resolved_config.json`, `metrics.json`, `results.tsv`,
  `checkpoint.pt`, `stdout.log`. Never write global `best_model.pt`,
  `results.tsv`, `final_run_metrics.json`, `*_results.csv`. Enforced by
  [code/run_context.py](code/run_context.py).
- Every canonical W&B run records the full provenance list in
  `updated_rules.md` §9 (`experiment_id`, `experiment_group`, `architecture`,
  `variant`, `seed`, `dataset_version`, `train_split_version`,
  `code_git_commit`, `config_hash`, all shape/optimizer fields, `resolved
  epochs`, `resolved subset_fraction`).
- W&B metric keys must not contain `/` inside a label (it creates spurious
  nesting). Use `E1_ADD_SUB`, not `E1 ADD/SUB`.
- Run names: `phaseB_MoE_seed42`, `phaseB_MoR_seed42`, `phaseB_MoRE_seed42`.
- Never hand-copy numbers into a results table. The exporter must refuse to mix
  dataset versions or config hashes and must emit `N/A` rather than fabricate.

**The proxy guard.** `code/canonical_spec.json` is the single frozen definition
of "canonical". A run may only declare `experiment_group = canonical_phase_b` if
it matches that spec exactly, uses the full dataset, and names a seed from the
frozen set. While any spec field is `null` the guard refuses every canonical
claim — that is deliberate, not a bug.

---

## 6. Experimental integrity

- **No proxy run enters the headline table.** A proxy is anything with a
  non-canonical epoch count, subset fraction, dataset version, batch protocol,
  loss configuration, or architecture.
- No table may mix configurations unless explicitly labelled an ablation with
  the varying field named.
- No causal claim without an experiment that isolates the variable.
- Do not alter the architecture to obtain a favourable result. If an
  improvement is found, preserve the original run and add the change as an
  explicit variant.
- **Archive, never delete, evidence.**
- Prohibited: post-hoc favourable subsets; changing depth labels after the fact;
  putting oracle routing in the main model; hiding parameter mismatches;
  treating high entropy as proof of specialization; treating low cosine
  similarity as proof of orthogonality; merging proxy and canonical runs; tuning
  until the desired conclusion appears.

---

## 7. When a test or gate fails

From `plan.md` §20, in order:

1. **STOP.**
2. **Report the failure** — plainly, with the actual output.
3. **Identify the cause.**
4. **Fix it.**
5. **Re-run the gate.**

Never revert a correctness fix because the old number looked better. Never
proceed past a failing gate.

---

## 8. Outcome honesty

Three outcomes are all scientifically valid and none may be optimized toward:

- **A** — MoRE retains/approaches baseline quality with stable specialization,
  adaptive depth, and low seed variance.
- **B** — MoRE is not lowest-loss but shows a stable, interpretable combination
  of specialization and adaptive recursion that neither component alone gives.
- **C** — once leakage, halting, routing and objective bugs are fixed, MoRE
  gives no meaningful advantage or fails to learn adaptive computation.

**Outcome C must be reported honestly.** The objective is to learn what the
architecture does, not what we want it to do. Distinguish measured facts from
interpretation everywhere.

---

## 9. Repository conventions

- **Two machines, two authors.** Interpreter paths are per-machine and are NOT
  interchangeable; check which machine you are on before quoting a path or a
  timing. Full record in [ENVIRONMENT.md](ENVIRONMENT.md), handoff procedure in
  [HANDOFF.md](HANDOFF.md).

  **Vedant's machine — RTX 3050 6 GB Laptop.** The gate, calibration and
  development machine. Windows user `vedan` (Windows spells it that way; the
  person is Vedant).
  - *CPU / correctness gates:* `C:\Users\vedan\anaconda3\python.exe`
    (Python 3.12.7, torch 2.6.0+**cpu**). Runs every `test_*.py` and
    `run_correctness_suite.py`; the **356** baseline was measured here.
  - *CUDA / training and timing:* `D:\res\git\MoRE\.venv_cuda\Scripts\python.exe`
    (torch 2.6.0+cu126). Built with `--system-site-packages` so
    `datasets`/`transformers`/`wandb`/`nltk`/`sklearn` come from base and only
    torch differs.

  **Ayan's machine — RTX 4060 8 GB Laptop.** Co-author on MoRE, and the machine
  the arithmetic POC was developed on: every run in `runs/` and every number in
  `README.md` / `results/results.md` was produced there, which is why those
  references are historically correct and are left alone (`CLAUDE.md` §6).
  - *Interpreter:* `C:\Users\Hp\anaconda3\envs\more_env\python.exe`
    (torch 2.5.1 + CUDA). That machine's Anaconda **base** and `C:\Python314`
    lack `torch`, so the env path is required, not optional.
  - **This is the canonical-matrix machine for the language phase.** 8 GB is the
    largest VRAM budget available to the project continuously, and the frozen
    `batch_size` was tuned for it (T-L7.0/T-L7.1), not for 6 GB.
  - A path under `C:\Users\Hp\` in a traceback means the log came from Ayan's
    machine. A path under `C:\Users\vedan\` means this one. Do not "fix" one into
    the other.

  **The college-lab RTX 4060 8 GB** is deliberately **excluded from the canonical
  matrix**: office-hours-only access means interrupted epochs, and splitting one
  matrix across two cards with different thermal behaviour puts a per-machine
  confound inside the seed variance the study reports. Use it only for
  independently labelled ablations, never for a subset of `canonical_lang_b` seeds.
- Shell is Git Bash on Windows; the working directory persists between calls —
  **use absolute paths.**
- One training system with an explicit `architecture = moe | mor | more` mode.
  Do **not** create parallel `run_moe.py` / `run_mor.py` implementations.
- Exploratory sweep drivers live in `automated/` and must never launch canonical
  runs. `archive/` holds invalidated history; re-stamp it with
  `python archive/mark_invalidated.py`.

---

## 10. Two audiences, two registers

**The rule: files are for the next agent, chat is for the researcher.** Both must
be technical. They must be technical about *different things*.

**This section is mandatory on every response, not advisory.** Its operational
form — the message shape, the four-question self-check before sending, the
progress-report template for autonomous loops, and worked ✗/✓ contrasts drawn from
this project — is
[`.claude/skills/research-writing/SKILL.md`](.claude/skills/research-writing/SKILL.md).
Load that skill at the start of any research session and apply it to every chat
message that reports work, run results, a defect, or a decision. The common
failure is not excessive technicality; it is technicality on the wrong axis —
plumbing detail crowding out the architectural choice the reader has to make.

### Files in the repo — keep the deep engineering detail

`changelog.md`, `TASKS.md`, code comments, docstrings, `ARCHITECTURE.md`,
`results_exp.md`, `canonical_spec.json` notes, and the paper draft. Write these as
densely as they already are. Their purpose is that six months from now a person or
an agent debugging a bad number can find the exact line, the exact field, and the
reason a past decision went the way it did. Keep:

- file:line references, exact metric keys, exact config field paths
- the failure mode in full, including the wrong output verbatim
- why an alternative fix was rejected
- provenance details, hashes, counts, seeds, run-directory names

Nothing in this section reduces what goes into files. Under-documenting a trap is
the more expensive error.

### Chat messages to the user — write for an AI researcher, not a build engineer

The reader is a competent AI researcher who is making architectural and scientific
decisions and does not want their working memory filled with build details. They
want to reason about things like *"should `cls_loss` exist at all"* and *"is MoR
actually the better architecture"* — not about deterministic-kernel warnings or
`stdout.log` line counts.

**Lead with the decision or the result.** State what was learned, what it means
for the architecture or the paper, and what the open choice is. Then, only if it
matters to that decision, one line of mechanism.

**Include, in chat:**
- the scientific finding and the number that supports it, with its uncertainty
- what it implies for the architecture, the claims, or the next experiment
- the decision you made and the one you need from the user
- cost in wall-clock or money when it affects a choice
- a defect's *consequence* ("those runs can't go in the paper, ~1.6 h lost")

**Leave out of chat** (put it in the files instead):
- verbatim log lines, stack traces, warning text, line counts
- file:line citations unless the user is about to open that line
- exact metric-key strings, config paths, hashes, directory names
- how a fix was implemented, unless the user must choose between fixes
- restating what a test asserts; report that it passed and what it covers

**A worked contrast.**

> ✗ *"`log_interval` gates validation at `engine.py:629` via `epoch %
> log["log_interval"] == 0 or epoch == epochs`, so `results.tsv` carried `nan` in
> 18 of 20 `val_loss` rows; changed `logging.log_interval` 10 → 2, which mutates
> `config_hash`."*
>
> ✓ *"We were only measuring validation loss twice per run, so 'best checkpoint'
> was really just 'last checkpoint'. Fixed — we now validate every 2 epochs. This
> had to land before the matrix, because it changes the run's config identity."*

Tables of results are welcome in chat — they are the finding. Tables of
implementation detail are not.

**When the user asks a mechanism question, answer it at full depth.** This section
sets the default, not a ceiling. "Why is the GPU idle", "how does the halting
gradient reach the halt head" — those get the real answer.

