---
name: benchmark-selection
description: Choose the right benchmark or specialized test for a model/architecture so the evaluation actually shows what you aim to show. Use when picking how to evaluate a model, when a general benchmark returns chance-level noise, when you need a test that isolates an architectural claim, or when writing an experiments/results section for a paper.
---

# Choosing the evaluation that shows what you aim to show

The failure mode this skill prevents: reaching for a famous leaderboard (MMLU,
GLUE, HellaSwag) because it is recognizable, and reporting three chance-level
numbers that show nothing about the model you built. The eval must be chosen to
**isolate the specific claim**, and it must be **valid at your scale**.

## The decision procedure

Work these in order. Each one can eliminate a whole class of eval.

### 1. What does the model actually claim to do? Evaluate THAT, not general ability.

General benchmarks measure "is this a good model". Your paper's claim is usually
narrower and architectural: *this mechanism does X better*. Design the eval to make
X visible. If MoRE claims "spend more compute where it's hard", the eval is
per-token loss **stratified by difficulty**, not an average that drowns the signal.
A model can be worse on average and better exactly where its mechanism is supposed
to help — only a stratified eval shows that.

### 2. Is the eval valid at this scale / this training data?

- **Chance-level check first.** At <100M params trained on one corpus, MMLU /
  HellaSwag / ARC / PIQA are at chance (GPT-2 small, 124M, is near-chance on MMLU).
  Three chance scores are noise, and a reviewer reads them as padding. Do not run
  them; say once in limitations that scale precludes zero-shot task eval.
- **What could the model have learned?** A model trained on Wikipedia has no
  world-knowledge (EWoK), no goal-state/action priors (agent benchmarks), no code
  (HumanEval). Near-chance is the honest prediction — report it as a limitation,
  or skip it, but never claim it as a finding.
- **Prefer evals valid at ANY scale:** zero-shot likelihood (perplexity, BLiMP,
  minimal pairs, cloze) and intrinsic probes (representation geometry, depth use).
  These need no fine-tuning and no scale floor.

### 3. Zero-shot likelihood > fine-tuning, when you can choose.

Fine-tuning evals (GLUE, SuperGLUE) add a confound: a worse score could be the
architecture OR the fine-tuning recipe (LR, epochs, head, early stopping), and
fairness demands tuning the recipe per arm — a researcher degree of freedom. At a
deadline that is a cost and a confound for no clean architectural signal. Likelihood
scoring (compare log-probs of two options) has neither. **If the eval needs
fine-tuning and you can find a likelihood-based alternative, take the alternative.**

### 4. Separate CAPABILITY benchmarks from MECHANISM probes — a paper wants both.

- **Capability** — does it do the job better than baselines? (perplexity in-domain,
  held-out test split, out-of-domain transfer, BLiMP). These give the headline
  "it works / transfers" numbers.
- **Mechanism** — WHY. Probe the claim directly as a falsifiable hypothesis
  (does adaptive depth track difficulty? are representations confined? do experts
  differentiate?). Design the probe so a clear negative is as informative as a
  positive. Report which the data supports and which it refutes.
- A capability win with no mechanism is weak; a mechanism probe that refutes the
  elegant story is a *result* (see "reporting" below).

### 5. Enumerate the tests that isolate the variable, then cost them.

For an architectural claim, ask "what would distinguish this from the obvious
alternative?":
- The **control that isolates the variable**: fixed-depth-matched control (is it
  adaptivity or just average compute?); matched-parameter baselines; a
  mechanism-disabled variant.
- The **cost-adjusted framing**: any quality win trades against compute. Report the
  *Pareto* view — quality vs FLOPs and vs wall-clock (they can disagree: a model can
  be FLOP-efficient and latency-bound). A win that costs 10× compute is a different
  claim from a free one.
- The **capability the baseline structurally lacks**: a recursive model exposes a
  test-time compute/quality curve a single-pass model cannot enter. Showing the
  curve IS a result, even before comparing to the baseline.
- The **held-out / OOD** axis: a within-distribution win may not transfer.

### 6. Check the assets exist before committing to an eval design.

Stratification by difficulty needs per-token annotations (frequency deciles,
POS); minimal-pair scoring needs a paradigm set; OOD needs a corpus packed with the
SAME tokenizer. Verify the arrays/datasets are on disk or downloadable (and that
`datasets>=3` supports them — script-based datasets are dead; use parquet). An eval
you cannot compute is not a plan.

## Where to look for candidate benchmarks

- **Literature in the subfield**: the paper a model is defined in (Universal
  Transformer → bAbI, LAMBADA, WMT; PonderNet → algorithmic tasks), and what its
  baselines reported *beyond perplexity* (Pareto FLOPs-vs-accuracy, sample
  efficiency, halting/routing heatmaps). Reviewers of that venue expect those.
- **Small-scale shared tasks**: BabyLM (for ~10M–100M-word budgets) → BLiMP,
  MSGS, EWoK, LongTail-Swap; GenBench for generalization-specific sets. Its
  pipeline is all zero-shot likelihood — a good match for the constraint in §3.
- **Build-the-baseline**: often no ready benchmark fits the exact claim, and the
  honest contribution is a purpose-built probe (the stratified curve, the subspace
  probe, the dynamic velocity probe). Document it so it is reproducible.

## Reporting — the honesty rules that make the eval count

- **Label measured fact vs interpretation everywhere.** "The ordering is measured;
  the capacity-dilution account is a hypothesis consistent with it."
- **A refuted hypothesis is a result.** If the elegant explanation (subspace
  trapping) fails every probe, say so plainly and give the mechanism the data does
  support. Testing an appealing story and dropping it beats asserting it.
- **Uncorrected = labelled uncorrected.** If the eval is not in a pre-registered
  family, report effect sizes and the exact test, mark it exploratory, and keep any
  significance verdict out of the abstract and conclusions.
- **No sentinel as a measurement.** `N/A` for a quantity that does not exist (a
  single-step model has no depth curve), never `0.0` that reads as a value.
- **Do not optimize toward a desired outcome.** If the data says the architecture
  does not deliver, report Outcome C; a diagnostic negative result is publishable
  and an overclaim is not.
- **Mind the units.** Perplexity is `exp(nats)` — a Δ-nat gap is a factor
  `exp(Δ)`; quote the nats (the metric the test operates on), not a mean of
  per-seed perplexities.

## A worked example (this project)

Goal: show whether MoRE (route-once + recurse) beats MoE (single-pass sparse) and
MoR (recurse). Rejected: MMLU/HellaSwag/GLUE (chance at 5.6M params; GLUE also
needs fine-tuning). Chosen:
1. **Capability** — val + held-out test loss; OOD perplexity on a different domain
   (parquet corpus, same tokenizer) → MoRE beats MoE and the gap WIDENS out of domain.
2. **Mechanism** — per-token loss by difficulty decile → advantage grows with
   difficulty; budgeted-depth curve → the recursive arms expose a test-time-compute
   trade-off the baseline cannot; BLiMP → gains concentrate on long-distance
   agreement.
3. **Claims isolated** — the fixed-depth control (adaptivity vs average compute);
   FLOPs-vs-quality Pareto (FLOP-efficient but latency-bound).
4. **Hypothesis probes** — subspace (trapping) and dynamic (velocity) probes, which
   *refuted* the elegant trapping story and located the real cause.
Every probe was designed so a clean negative would still be reportable.
