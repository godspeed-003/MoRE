# PAPER_REVIEW.md — adversarial review of the language paper, 2026-10-15

**What this is.** A reviewer-perspective audit of the TMLR/ACL paper as it stands
at `edc268d`, cross-checked against the two prior review sources this repository
already holds:

| source | what it is |
|---|---|
| **NeurIPS-workshop reviewer 652J** | a real external review of the *arithmetic* submission. Four numbered concerns, referenced in `code/diag_halt_gradient.py` (1), `code/flops_accounting.py` (3), `code/eval_test_split.py` (4). Concern 2 is not recorded anywhere in the repo. |
| **`PHASE11_AUDIT.md`** | the internal blocking audit, 2026-08-29. 12 required actions, arithmetic-era. |
| **`PAPER_GUIDE.md` §7, §12** | the honesty payload and the forbidden-sentence checklist. |

Scope: `paper/common/*.tex` (the shared body), both venue `main.tex` files, and
`results/language/results.json` as the authoritative artifact.

**Headline.** The paper's *claims* are well-disciplined — I could not find a
forbidden sentence, an unlabelled hypothesis, a sentinel reported as a
measurement, or a number that disagrees with the artifact. The exposure is
almost entirely **under-documentation**: several of the paper's load-bearing
quantities are derived by code that is rigorous and unreported, and the released
artifact contains a metric family the paper neither uses nor explains, one value
of which is **impossible on its face**.

---

## 1. Where the old reviews stand

### 652J concern 1 — "the halt head has no real gradient path"

**Answered, and answered well.** The reviewer's mechanism claim was wrong (the
model does accumulate an ACT-weighted output, so a gradient path exists) but
their *prediction* — a near-constant policy — was right. The paper now reports
the dominance ratio rather than merely asserting the path exists: the halt
gradient is **~285× below the task term** (§3). That is the honest answer and it
is in the body.

**Residual exposure:** the 285× is stated without a method. A reviewer cannot
tell whether it is an L2 norm of per-term gradients at one checkpoint, averaged
over seeds, at what point in training. `code/diag_halt_gradient.py` knows; the
paper does not say.

### 652J concern 3 — "MoRE vs MoR needs an iso-FLOP or iso-latency control"

**Answered numerically, undocumented methodologically. This is the single
largest gap in the paper.**

FLOPs appear **17 times** across the body and carry the paper's entire positive
claim — "2.19× vs 9.07×", "72% of the gain for 24% of the compute", "3× quality
per unit compute", and the Pareto-dominance result that the abstract leads on.
The only pointer the reader gets is "(§\ref{sec:metrics})" — and **§Metrics does
not define FLOPs**. It defines the primary metric, the routing measures and the
depth measures. The cross-reference resolves to a section that does not contain
the thing it promises.

What is actually behind the number, from `code/flops_accounting.py`, and none of
it is in the paper:

- counted with `torch.utils.flop_counter.FlopCounterMode` on a **real forward
  pass** of the arm as its own `resolved_config.json` built it — no typed-in
  formula, so the count cannot drift from the model;
- depth swept with `fixed_depth=True`, `max_depth=k` for k=1..7, giving
  `FLOPs(d) = base + (d-1)·per_step`;
- **linearity verified against the measured k=7 count, >0.1% residual is a hard
  failure** — a non-linearity there would mean the recursion is not doing what
  the architecture claims;
- evaluated at each arm's **measured mean depth**, not at `max_depth`, which
  would overstate both recursive arms;
- includes attention, embeddings and the 8192-way LM head, which the file notes
  are *not* negligible on a 256-token block — earlier hand derivations of the
  FFN term alone were wrong for exactly that reason.

That is a better FLOPs accounting than most papers ship, and the paper gets no
credit for any of it. **A reviewer's most natural attack — "your efficiency
claim is a back-of-envelope FFN ratio" — is already answered in the repo and not
in the paper.**

### 652J concern 4 — "headline on validation despite a declared test split"

**Answered.** Table 2 carries a Test loss row (3.947 / 3.513 / 3.621) alongside
validation, and the ordering is identical. 652J called this "the most serious
methodological objection in either review, because unlike the others it can
invalidate a number rather than merely under-describe it." It is closed.

### 652J concern 2

**Not recorded anywhere in this repository.** Three of four concerns have a
dedicated script with the quotation in its header; concern 2 has none. Before
resubmitting anywhere, find the original review and check whether concern 2 was
answered or silently dropped. An unanswered prior concern resurfacing is
avoidable damage.

### `PHASE11_AUDIT.md` required actions, as they bear on the language paper

| audit item | status in the language paper |
|---|---|
| 2 — replace `k×std` with an exact test | **done**; §Statistics, with the floor derived |
| 5 — absent → `N/A`, never `0.0` | **done**; `N/A` is used and the depth-allocation-error `N/A` is explicitly defended in the disclosures |
| 6 — carry a scale floor, numbers uninterpretable without it | **done**, in the language analogue: the backoff-bigram floor at 4.9849 nats with per-arm margins |
| 7 — **label depth claims "training-time" or add a validation-pass metric** | **this is the one that is still open — see §3.1** |
| 9 — provenance / dirty trees | **done**; four commits and `code_git_dirty` are disclosed |

---

## 2. Measured but not written down

Each of these exists in the repo and is absent from the paper. All are cheap to
fix; none requires a run.

1. **The FLOPs method.** §1 above. Highest priority of anything in this document.
2. **The OOD evaluation protocol.** The Pareto-dominance claim — the paper's one
   "something neither does alone" result — rests on `pile:test` scored at
   `n_blocks = 1500`, `seq_len = 256`, i.e. **384,000 tokens**, with the
   WikiText-fitted 8192-BPE tokenizer. The paper states the tokenizer caveat but
   **never states how much data the claim rests on**. 1500 blocks is a small
   evaluation for a headline inversion.
3. **Per-seed values.** The paper reports mean ± std for 15 cells and never the
   cells. With n = 5, every headline p pinned at the floor, and a variance-ratio
   claim in the text, the per-seed numbers *are* the evidence. They exist in
   `results.json`. An appendix table costs a quarter page in a venue with no
   page limit.
4. **Per-family depth.** `depth/mean_by_family` shows MoR's depth spanning
   6.572–6.887 across six linguistic families (spread 0.31) against MoRE's
   6.953–6.962 (spread **0.009**). That is a direct, clean demonstration of the
   paper's central claim — MoR differentiates depth, MoRE does not — on a
   linguistic partition rather than via a correlation against a near-constant.
   **It is measured and unused.** It is stronger evidence than the ρ it would
   sit beside, because it does not suffer the near-constant problem the paper
   has to spend a paragraph defending.
5. **The halt-gradient measurement protocol** behind "285×" (§1 above).
6. **No learning curves, though the data exists.** Every admitted run records
   per-epoch validation — 3 points × 15 runs. The paper asserts an underfitting
   regime at 3 epochs and concedes the behaviour at 10 epochs is unknown, while
   the curves that make both statements auditable sit unused in the run
   directories. See `PAPER_MITIGATION.md` §4: they show MoRE still improving
   fastest at the budget cut, which is double-edged and should still be
   published.

   *(An earlier draft of this file said no curves existed. That was read off a
   non-admitted run directory — 62 exist, 15 are admitted — whose `results.tsv`
   carries a single row. The admitted runs carry three.)*

---

## 3. Internal inconsistencies a reviewer can catch

### 3.1 Two halting metric families disagree, and one value is impossible

`results/language/results.json` carries two sets of halting statistics:

| quantity | `halt/*` and `depth*` family | `val/*` family |
|---|---|---|
| MoRE forced-exit | 0.9734 ± 0.0075 | 0.9809 ± 0.0042 |
| MoR early-exit | 0.1877 ± 0.1507 | 0.1569 ± 0.1119 |
| MoRE mean depth | 6.9552 ± 0.0044 | **7.6152 ± 0.0142** |
| MoR mean depth | 6.7297 ± 0.2410 | 6.8985 ± 0.2835 |

The paper consistently uses the first family, and those numbers are correct and
mutually consistent: `halt/forced_exit_rate` 0.97340 equals
`depth_dist/step_7_pct` 97.3405 exactly, on the same 279,990-token pass.
The paper's 6.955, 6.730, 97.3% ± 0.8%, 18.8% ± 15.1%, depth std 0.43, ρ
+0.2985 and −0.0230 all check out against the artifact.

**The problem is `val/avg_recursion_steps` = 7.6152 for MoRE, with `max_depth =
7`.** A mean recursion depth cannot exceed the architectural ceiling. It is
almost certainly a differently-defined quantity (plausibly an `N + R`
ponder-style figure, where the remainder pushes it above the step count), but
the paper does not use it, does not mention it, and does not explain it.

**Why this matters more than it looks.** The paper promises artifacts, and the
forced-depth control — the paper's own proposed general diagnostic — is matched
to "the arm's own measured mean depth of 6.955". A reviewer who opens
`results.json` to check that matching will find a *second* mean-depth metric for
the same arm that is both larger and above the stated maximum. The honest
reading is a naming collision between a training-pass accumulator and an
evaluation-pass metric; the reviewer's first reading will be that the depth
number was chosen.

**Recommended:** either delete the `val/*` halting keys from the released
artifact, or document in one line what each family measures and why they differ.
This is also the live remnant of `PHASE11_AUDIT` item 7, which asked precisely
for depth claims to be unambiguous about which pass they come from.

### 3.2 The paper contradicts itself on hardware

- §Training protocol: trained "**on a single laptop GPU**".
- Limitations: "**The matrix spans two GPUs** and four code revisions."
- TMLR Reproducibility Statement: two arms on an **RTX 4060 (8 GB)**, "the third
  remains on its original **V100**".

A V100 is not a laptop GPU, and "a single laptop GPU" is contradicted twice. The
disclosure version is the correct and creditable one; the setup sentence is
stale. A reviewer reading §Setup and then the Limitations will catch this, and
it devalues a disclosure the paper is otherwise right to be proud of.

### 3.3 The frozen spec disagrees with the paper's mechanism table

`PAPER_GUIDE.md` flagged this and told the writer to "fix the loser in the same
commit". **It was not fixed.**

| the (0.001, 0.001) dev calibration point | val loss | depth | AMI |
|---|---|---|---|
| `code/canonical_spec_language.json` prose | 5.2406 | 2.63 | 0.2907 |
| `code/lang_calibration_weights.json` | 5.2352 | 2.698 | 0.2851 |
| **the paper, §6.3 competition table** | 5.2352 | 2.698 | 0.2851 |

The paper took the machine-written record, which is the right call. But the
**frozen specification** — the document whose entire purpose is to define
canonical — still carries three numbers that contradict the paper. A reviewer
who requests the spec finds the paper disagreeing with its own frozen reference
on the table that carries the novelty claim.

---

## 4. What reviewers will attack that is not a documentation problem

These are real limits. The paper discloses all of them; the question is whether
disclosure is enough.

1. **One scale, and it is tiny.** 5.58M parameters. Sparse-MoE benefits are
   known to emerge with scale, so "the negative result is an artefact of
   operating below the regime where sparsity pays" is the obvious objection. The
   paper converts it into a falsifiable crossover prediction, which is the best
   available move without a second model size — but a second size remains the
   single most valuable missing experiment, and the paper says so.
2. **n = 5, every headline pinned at the resolution floor.** The paper is
   scrupulous here ("as extreme as a five-seed design admits", weaker than
   significance). A reviewer may still say that a design in which *no* effect
   size can reach significance under the declared correction is a design
   problem, not just a limitation. Going to n = 7 moves the floor to
   2/3432 = 0.00058 and would cost 6 more cells.
3. **E = 6 is never justified or varied.** Why six experts? No sweep, and the
   limitations do not name expert count as an unexplored axis (they name model
   size, epochs and the unrun controls). An MoE paper with one expert count will
   be asked.
4. **The mechanism story rests on a dev-corpus grid at one seed.** §6.3 — the
   competition between depth and specialisation, explicitly "the novelty" — is
   measured on a smaller corpus at a single seed, and labelled as such. A
   reviewer may reasonably decline to accept the paper's central mechanism on
   that evidence.
5. **A declared Family-A comparison was never run** and consumes its Holm slot.
   The paper handles this with unusual integrity. A hostile reviewer still gets
   to say the pre-registered design was not executed.
6. **Throughput is identical for MoR and MoRE (170 items/s each).** The paper
   explains this as latency-bound dispatch, which is credible. Two arms with 4×
   different arithmetic landing on exactly the same three-significant-figure
   throughput will read as a copy error to a skim-reader; a decimal place or a
   measurement protocol would defuse it.

---

## 5. Priority

| # | action | cost | why |
|---|---|---|---|
| 1 | Document the FLOPs method in the appendix | ~half a page, no run | The efficiency claim is the paper's positive result and currently has no stated derivation; the real method is strong and hidden |
| 2 | Resolve the `val/*` halting metrics in the released artifact | one line of docs, or delete the keys | A depth metric above its own ceiling, in the file the paper tells reviewers to read |
| 3 | Fix "a single laptop GPU" | one sentence | Self-contradiction against the paper's own disclosure |
| 4 | Reconcile `canonical_spec_language.json` with the paper | three numbers | The frozen spec contradicts the novelty table |
| 5 | State the OOD protocol (1500 blocks / 384k tokens, pile:test) | one sentence | The Pareto claim's evidence base is unstated |
| 6 | Add per-seed appendix table | quarter page | With n=5 at the floor, the cells are the evidence |
| 7 | Add the per-family depth table | quarter page | Free strengthening of the central claim, already measured |
| 8 | Recover 652J concern 2 | lookup | Unknown unanswered prior objection |

Items 1–5 are all documentation of work already done. None requires a GPU.
