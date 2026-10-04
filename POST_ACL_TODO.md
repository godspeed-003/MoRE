# POST_ACL_TODO.md — work deferred until after the ACL submission (12th)

Everything here is **out of scope for the 8-day ACL window** and is parked so it is not
lost. None of it blocks the submission. Order is rough priority. When you pick one up, move
it into `changelog.md` on completion and strike it here.

Authority is unchanged (`CLAUDE.md` → the three governing docs). This is a parking lot, not
a spec.

---

## 1. PonderNet-style halting — the principled fix for depth saturation (deferred by decision, 2026-10-04)

**Why deferred, not dropped.** T-LX.20 established that MoRE's depth saturation (97.3%
forced-exit at depth 6.96 on the full corpus) is **not architectural** — the same model
allocates depth adaptively (forced-exit ≈ 0, depth ~2) on less data. It is a calibration
failure: the fixed ACT ponder penalty is outgrown by the task gradient as optimizer steps
accumulate, and once the halting sigmoid saturates its gradient vanishes (Graves 2016). The
linear-penalty ACT scheme cannot recover from that on its own.

**The fix.** Replace the linear ponder cost with PonderNet's formulation (Banino et al.,
NeurIPS 2021): a probabilistic halting distribution penalized by a **KL divergence against a
geometric prior**, which keeps a gradient on the halt head even near saturation and makes the
expected-depth target explicit rather than emergent. A dynamically annealed `lambda` (scale
the ponder cost up as the task loss falls) is the cheaper half-measure to try first.

**Why it is not in the ACL window.** It is a new loss term and a changed halting head, so it
needs: a new differentiable-path test in Gate 2 (`CLAUDE.md` §2 requires a real gradient to
the halt parameters), re-validation that it does not regress the 356-check suite, and its own
labelled-ablation runs. That is a research sub-project, not a week's work, and rushing a loss
change before a deadline is exactly how a silent bug reaches a table.

**What exists to build on.** The ACT forward path and the halt head are in
`code/more/model.py`; the ponder/halting loss assembly is in `code/more/losses.py` (grep
`ponder`/`halting`). The halt-gradient attribution is already measured — T-LX.14 found the
halt gradient is 285× task-dominated, which is the quantitative statement of the
gradient-overpower problem PonderNet addresses. Keep it a **labelled variant**, never a change
to canonical (`CLAUDE.md` §6).

**What to measure.** Whether PonderNet recovers adaptive depth on the FULL corpus WITHOUT the
specialization/quality cost that the plain ponder-weight ladder incurs (T-LX.20 Cause 3: on
the dev grid, forcing depth down via the ponder weight drove AMI 0.29 → 0.003). If it
decouples depth from specialization, that is a genuine positive result and a second paper. If
it does not, the competition finding stands and is strengthened.

---

## 2. The ponder-weight ladder at full corpus (in flight / may be partial at ACL time)

A labelled ladder of halting weights {0.001, 0.01, 0.03, 0.1} at `subset_fraction = 1.0`,
one seed, via `automated/sweep_tlx20_weights.py --axes halting --subset 1.0`. Tests whether a
higher *fixed* ponder weight recovers adaptive depth at full-corpus scale, and at what cost to
val loss and AMI — i.e. the Pareto frontier of the depth↔quality↔specialization competition
on the real corpus rather than the dev corpus. If it completes before the deadline the result
goes in the paper as the full-corpus confirmation of T-LX.20 Cause 3; if not, the dev-corpus
grid carries that claim with its stated caveat. Output: `code/lang_sweep_tlx20.json`
(the committed version is the subset-0.1 sweep; a full-corpus run overwrites it — rename
first). EXPLORATORY: single seed, never a canonical result.

---

## 3. The scale study — the single most valuable upgrade past ACL

Run the three arms at a second `d_model` (128 and 256) to turn the scale limitation into a
measured trend. The capacity-dilution account (`PAPER_GUIDE.md` §6 Cause 1) predicts the
MoRE−MoR gap shrinks as each expert gets more tokens per parameter; showing that crossover,
or its absence, is what moves the paper from "negative result at one scale" to "scaling trend
with a falsifiable prediction." ~15 GPU-hours. This is the item most likely to lift a venue
tier.

---

## 4. The remaining BabyLM-family benchmarks (BLiMP is the ACL one; these come after)

BLiMP is being built for the ACL submission (minimal-pair likelihood scoring, valid at this
scale). The rest of the BabyLM 2023–2025 zero-shot likelihood suite is post-ACL:
- **MSGS** — memorization vs linguistic-hierarchy generalization.
- **EWoK** — physical/social reasoning via plausible-vs-implausible likelihood scoring.
- **LongTail-Swap (2025)** — rare-word generalization, directly relevant to a 100M-word budget.
All are scoring-based (no fine-tuning/prompting), so they fit the harness pattern in
`code/eval_stratified.py`. The 8192-BPE-on-wikitext-103 vocab caveat applies: absolute scores
are not comparable to published 32k–50k-vocab models, cross-arm comparison is.

---

## 5. Length extrapolation

Train at seq_len 256, evaluate likelihood at 512 and 1024. Recursive, weight-shared models
often hold perplexity better than fixed-depth ones on extended context, which would be a
genuine MoR/MoRE advantage. Needs a **re-pack** of the corpus at the longer length (it is
pre-packed at 256 and `MoRELanguageDataset` reads that packing), so it is a data-layer change,
not an eval flag — point `data/lang/build_language_dataset.py` at the longer length with the
existing tokenizer.

---

## 6. Psycholinguistic surprisal (speculative, higher effort)

Correlate ACT depth with human reading-time corpora (Provo, Dundee): a genuinely adaptive
model should spend more steps on garden-path and morphologically anomalous tokens. Only
meaningful once halting is fixed (item 1) — on the current saturated halting there is no depth
variance to correlate, so this is downstream of PonderNet.

---

## 7. Synthetic sanity check (a reviewer may ask for this; cheap)

Show MoRE converges on a toy task (character-level parity, or Dyck-language closing) to prove
the full-corpus failure is tied to the representation-shifting natural language demands and
not a code bug. Small and fast; worth having ready as a rebuttal artifact even if not in the
main paper. Note the arithmetic engine already exists and could host a parity task.
