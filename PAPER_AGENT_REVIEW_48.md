# External agent review of the MoRE language paper — 48 claims, under verification

**Received.** 2026-10-18, from the co-author's agent.
**Status.** UNDER VERIFICATION. Every row below is a CLAIM BY A THIRD PARTY, not a
finding of this repo. Do not act on any of it until the Verdict column says so.

**How to read the verdicts.**

| Verdict | Meaning |
|---|---|
| `CONFIRMED` | The repo's own canonical data or paper text shows the claim is right. |
| `PARTLY` | Right in substance, wrong or overstated in some part. |
| `REFUTED` | The canonical data shows the claim is wrong. |
| `UNVERIFIABLE` | Cannot be settled from this repo (e.g. needs a literature lookup). |
| `ALREADY-HANDLED` | The paper already discloses or addresses this. |

**Canonical sources used for adjudication** (do not adjudicate from the paper's own
prose when a stored number exists):
`results/language/results.json` (spec L7.2-TLX.12-route-persistence, 15 admitted
runs), `results/flops_language.json` (derived by `code/flops_accounting.py` on a
real forward pass), `code/eval_val_depth.py`, `paper/common/*.tex`.

---

## Blocking issues

| # | Claim | Verdict | Evidence |
|---|---|---|---|
| 1 | Forced-d7 changes the readout (mixture → last state), so the 1.47-nat / 24× gap is not attributable to input-dependent depth | **CONFIRMED** | See `code/eval_val_depth.py`; forced-depth sweep re-runs the trained weights at fixed depth, and the mixture readout of Eq. `halt-w` is replaced by the final state. The paper itself says this is "the same trained weights with depth forced at inference" (04_results.tex:55-57) — inference-time intervention, so the gap conflates policy with readout. |
| 2 | The "factor of 24" divides by a near-zero denominator (0.060), no CI | **PARTLY** | The 24 is `ffn_mult=24` for MoR vs 4 for MoE/MoRE — a *width* ratio, not a measured quotient with a 0.060 denominator. No 0.060 denominator exists in `flops_language.json`. But the underlying worry (an unstable ratio reported without a CI) is fair for the *per-FLOP* ratios; they are means over 5 seeds with no interval reported. |
| 3 | "MoE" is a depth-1 single block, so beating it is mostly "7 passes beat 1 layer" | **CONFIRMED** | `flops_language.json`: moe `max_depth=1`, mor/more `max_depth=7`. MoE is literally one pass. |
| 4 | No matched-FLOPs MoR (4× FFN at depth 7), so composition is untested beyond sparse-activation savings | **CONFIRMED** | Only three arms admitted; no ffn_mult=16 MoR, no depth-1 MoRE. `updated_objective.md` REQUIRED ABLATIONS lists "Parameter-matched MoR" as required; it is unrun. |
| 5 | Pre-registered fixed-depth-trained control never run | **CONFIRMED** | Paper states it directly: "that control is unrun" (04_results.tex:60), and Limitations §"What was declared and not run" keeps its Holm slot consumed and unfilled. Correctly disclosed, but the gap is real. |
| 6 | Equal epochs ≠ equal compute; MoE gets ~2–9× less training FLOPs | **PARTLY / CONCEDED** | True that MoE is 1 pass vs ~7. But the paper reports inference FLOPs from a real forward pass and states per-arm training cost separately. Still, no training-FLOPs parity was enforced, so MoE's deficit is partly a compute artefact. Not disclosed as a confound. |
| 7 | "Frontier" from three points is vacuous; any monotone triple is trivially non-dominated | **CONFIRMED (methodological)** | Three points, one axis ordering. The Pareto language is weak. The paper partially concedes this ("the three trace a frontier and MoRE's claim is the slope from MoE, not its position") but the "frontier" wording survives in the abstract/conclusion. |
| 8 | MoRE may beat MoR in-domain at larger T (still improving at d=7) | **PLAUSIBLE, UNTESTED** | `depth_dist/step_7_pct` = 97.3% for MoRE, 81.2% for MoR — both pinned at the ceiling. The forced-depth sweep does not extend past 7, so the claim is untestable with stored runs and is a legitimate forward-looking caveat. |
| 9 | "Per unit compute" is arbitrary; unembedding dominates the FLOP count | **CONFIRMED — strongest single finding** | `flops_language.json` gives 5,249,536 FLOPs/token for MoE. The shared output projection is 8192×256×2 = 4,194,304 FLOPs/token = **79.9% of MoE's total**. Excluding it: MoE 1,055,232; MoRE 7,302,777 (**6.92×**, not the reported 2.19×); MoR 42,349,459 (**40.1×**, not 8.87×). The 2.19× and 8.87× figures are deflated by a constant shared by all three arms. |
| 10 | At equal total params MoR activates 6× more FFN params/token, which alone predicts the in-domain gap | **CONFIRMED** | Total params identical (5,584,908 all three). Active FFN/token: MoR 3,145,728 (one 24-wide block) vs MoE/MoRE 524,288 (one of six 4-wide experts) = exactly **6.00×**. MoR is lowest-loss. This is a live, unaddressed alternative explanation for the headline MoRE–MoR gap of +0.1216. |
| 11 | §6.1 "arithmetically false" overreaches | **CONFIRMED (rhetorical)** | The dilution account is refuted by matched tokens-per-parameter, which rules out one narrow mechanism, not dilution in general. Overreach in tone. |
| 12 | §6.3 evidence too thin: 3 grid points, 1 seed, different dev corpus, loss ~5.3, billed as central | **CONFIRMED** | The ponder-weight calibration is a 3-point, single-seed sweep on a different corpus at much higher loss. It cannot bear the weight of a central claim. |
| 13 | §6.3 judges adaptivity by mean depth, which §5.2 says cannot establish adaptivity | **CONFIRMED (internal inconsistency)** | §5.2 states mean depth alone does not establish adaptivity; §6.3 then argues from mean depth. Self-contradiction. |
| 14 | §6.3 confounds: lower depth = less training compute for experts; frozen depth 2.7 not saturated | **CONFIRMED** | Untestable separation from stored runs; the frozen setting does not saturate, so the "competition" reading is not identified. |
| 15 | Trapping logic flawed: monotone settling is what fixed-point trapping predicts, used as refutation | **CONFIRMED (logic)** | If updates settle monotonically toward a fixed point, that is consistent with, not evidence against, trapping. Direction of inference is wrong. |
| 16 | Cosine criterion sign error: negative update-state cosine = shrinkage, contradicting a "positive collinearity" test | **NEEDS CHECK** | Must inspect the diagnostic definition and sign convention in `code/`. Canonical `diag/mean_pairwise_cosine_sim` is +0.079 (MoRE) / +0.030 (MoE) — both positive, both low. |
| 17 | Participation-ratio test weak: comparing to MoE (no recursion) does not test trapping within recursion | **CONFIRMED (methodological)** | A no-recursion arm cannot serve as the control for a within-recursion phenomenon. |
| 18 | Inter-expert cosine "MoRE 0.916 vs MoE 0.630" suggests redundant experts | **REFUTED as stated** | Canonical values are **0.0791 (MoRE) and 0.0304 (MoE)** mean pairwise cosine, max 0.0968 / 0.0572. The quoted 0.916 / 0.630 do not appear in `results.json`. Even taken at face value the inference is backwards: *lower* cosine means *more* differentiated experts, and both canonical values are low, i.e. differentiated — consistent with the specialisation claim, not against it. Needs the metric source identified before anything is conceded. |
| 19 | Ponder weight not tuned on full corpus; MoRE may be mistuned rather than failing | **CONFIRMED** | The paper concedes non-transfer. This is a legitimate alternative to "composition fails". |
| 20 | Hardware/revision confound: MoRE on V100, others retrained on RTX 4060 across four revisions | **CONFIRMED — disclosed, still live** | Paper's Disclosures state the matrix spans two GPUs and four revisions, and that retrained arms reproduced to −0.117σ and −0.119σ (common-mode, cancels in between-arm gaps). The ACT-threshold numerical-sensitivity part is not addressed. |
| 21 | "No arm tuned for itself" contradicts "one configuration per arm frozen from a grid" | **CONFIRMED (contradiction)** | `updated_objective.md` CANONICAL HYPERPARAMETER PRINCIPLE forbids per-arm tuning; freezing one config *per arm* from a grid is exactly per-arm selection. Real tension with the stated principle. |
| 22 | Missing training details: LR schedule, warmup, precision, document-boundary packing | **CONFIRMED** | Not in the main text. Recoverable from the frozen config spec and run `resolved_config.json` files; a writing gap, not a data gap. |
| 23 | Adaptive depth barely exists: MoR 6.73/7, MoRE at ceiling | **CONFIRMED — strongest finding against the paper's own title** | `depth/mean`: MoR **6.7297**, MoRE **6.9552** (T=7). `depth_dist/step_7_pct`: MoR **81.2%**, MoRE **97.3%**. MoRE halts at the maximum budget for 97% of tokens. Both arms are near-saturated at the depth ceiling. |
| 24 | OOD headlined despite §7.5 saying such results carry no verdict | **CONFIRMED (inconsistency)** | Abstract/conclusion assert OOD dominance at p=0.024 uncorrected while the body disclaims uncorrected results. Direct self-conflict. |
| 25 | OOD losses near uniform (7.2–7.9 vs ln 8192 = 9.01); ordering may be calibration | **CONFIRMED (weak evidence)** | All arms sit well short of the uniform-loss ceiling, i.e. the task is barely learned out of domain; small differences there are not robust. |
| 26 | OOD set redefined after first look; only one non-Wikipedia domain | **CONFIRMED (disclosed forking path)** | Paper discloses the capped re-scoring and reports both. Real but disclosed. |
| 27 | n=5 is the binding limit; every result sits at the p-floor | **CONFIRMED** | `min_p = 0.00794` from 252 permutations; all three declared comparisons return exactly `p = 0.00794`, `at_floor=True`. No comparison can be more significant; only more seeds help. |
| 28 | Family B (pre-registered depth–difficulty correlation) not reported in main text | **CONFIRMED** | `depth/spearman_vs_logfreq` ≈ 0.028 and `depth/spearman_vs_model_loss` ≈ −0.023 — both far above their own null bands (±0.0037) yet near-zero in magnitude. Reported only in appendix. |

## Errors and inconsistencies

| # | Claim | Verdict | Evidence |
|---|---|---|---|
| 1 | Intro OOD numbers stale (7.573/7.802) vs Table 2 (7.212/7.423) | **CONFIRMED** | 04_results.tex quotes the superseded capped set alongside the Table 2 values. Both sets are reported and the substitution is explained, so this is a clarity defect, not a false number. |
| 2 | Fig. 5 caption says MoRE is worst in domain, but MoE is worst | **CHECK** | Compare `val/task_loss`: MoE 3.9483 > MoRE 3.6370. MoE is worst. Verify the caption text. |
| 3 | "Ordering inverts" misleading — only MoR/MoRE swap | **CONFIRMED** | In-domain MoR < MoRE < MoE; OOD MoRE < MoR < MoE. Only the top two swap. |
| 4 | "Significant at the floor" (abstract) vs "weaker than significance" (§7.3) | **CONFIRMED (conflict)** | Both phrases present. §7.3 is the more careful statement; the abstract should match it. |
| 5 | §7.6 truncated: "Three facts…" followed by nothing | **FIXED at T-LA.24** | This was real: the replacement paragraph existed but the disclosures prose had been moved to the appendix. Body now carries a self-contained paragraph. Verify in the current build. |
| 6 | "Closing" claim unsupported: MoRE–MoR gap by epoch 0.121 → 0.166 → 0.122 | **CONFIRMED** | Per-seed MoRE−MoR gaps are 0.0958, 0.1285, 0.0831, 0.1795, 0.1208 — variable, not monotone closing. The Limitations section concedes budget-dependence; the "closing" phrasing overreaches. |
| 7 | Table 1 POS ambiguity (input vs target tokens; tag→BPE mapping) | **CONFIRMED** | Not stated. |
| 8 | BLiMP per-phenomenon claim (anaphor/determiner-noun concentration) has no data | **CONFIRMED** | Aggregate BLiMP only; per-phenomenon breakdown absent. |
| 9 | App. F gives no accounting for the 0.069% gap | **CONFIRMED** | Derived without shown numbers. |
| 10 | Eq. 1 indexes t from 0 while text says c₁=0 | **CHECK** | Cosmetic consistency. |
| 11 | Eq. 6 clutter: zero-weight terms | **CHECK** | `train/family_cls_loss` is exactly 0.0 across all 5 seeds — a structurally-zero term carried in the loss display. |
| 12 | Fig. 4 x-label clipped, Fig. 2 labels overlap lines | **CHECK** | Visual; verifiable in the built PDF. |

## Related work

| # | Claim | Verdict | Evidence |
|---|---|---|---|
| 1 | Sparse Universal Transformer and MoEUT already combine MoE with recursion, undermining "not aware of a matched comparison" | **UNVERIFIABLE HERE — must be looked up** | Not resolvable from this repo. Material to the novelty claim; needs a literature check before conceding or denying. |
| 2 | Naming collision: this "MoR" is not Bae et al.'s Mixture-of-Recursions | **CONFIRMED (collision real)** | `updated_objective.md` defines MoR as "Mixture of Recursion". The name is taken in the literature. Rename or qualify. |
| 3 | Mixture-of-Depths mischaracterised — it skips layers rather than reusing weight-tied recursion | **CONFIRMED** | MoD is layer-skipping; MoR is weight-tied block reuse. Distinct mechanisms; the paper's related work conflates them. |
| 4 | "This literature almost always reports mean and histogram only" needs citations or softening | **CONFIRMED** | Uncited comparative claim about the literature. |

## Presentation

| # | Claim | Verdict |
|---|---|---|
| 1 | Repetition and defensive tone; claims recur 4–5 times; 15–20% cuttable | **CONFIRMED** — consistent with the independent T-LA.24 work, where the same claims were compressed. |
| 2 | Abstract lacks the evidence-status hedging the body carries | **CONFIRMED** — same conflict as Error 4 / Blocking 24. |
| 3 | Redundant appendices N, M, I duplicate main-text sections | **CHECK** |
| 4 | No anonymised code repo despite reproducibility statement | **CONFIRMED** — submission-blocking for both venues. |

---

## Bottom line (to be completed after verification)

See the audit conversation and `changelog.md` for the resolved conclusions.

---

# FINAL VERDICTS

## Corrections to the provisional verdicts above

Three provisional rows were wrong or under-argued and are corrected here. The
provisional text is left in place rather than deleted, because the reasoning path is
part of the record.

**Blocking 18 (inter-expert cosine) — PARTLY, and I was wrong to call the numbers
unsourced.** `results/language/subspace.json` contains exactly the quoted values:
MoE `between_expert_mean_cosine` 0.6296 +/- 0.0398, MoRE 0.9161 +/- 0.0136, global
participation ratio 41.52 / 70.25 / 57.27. They are real and correctly sourced from
the subspace probe. The reviewer's *inference direction is also right*: higher
between-expert cosine means MORE similar experts, and MoRE is more similar than MoE,
which sits badly with the specialisation claim. `between_over_total_variance`
reinforces it — MoE 0.164 vs MoRE 0.030, between-expert variance ~5x lower in the
composed arm. **So the real defect is definitional, not numerical.** The paper reports
two quantities both called "cosine" that differ by 10-30x:
`diag/mean_pairwise_cosine_sim` = 0.079 (MoRE) / 0.030 (MoE), and the subspace
`between_expert_mean_cosine` = 0.916 / 0.630. One is expert-weight cosine, the other
is final-state (lm_head input) cosine grouped by expert. Neither is defined at the
point of use. A reader who takes 0.916 as the specialisation metric concludes the
experts are redundant, and the paper never says otherwise.

**Blocking 2 (ratio instability) — REFUTED as stated.** There is no "factor of 24"
quotient and no 0.060 denominator anywhere. 24 is MoR's `ffn_mult` (block width)
against 4 for MoE/MoRE — a width ratio, not a measured quotient. 0.060 is MoRE's
*learned-policy-minus-forced-d7* loss gap in nats (0.060 nats), a subtraction, not a
denominator. The review appears to have fused two unrelated numbers. The underlying
concern — ratios reported over 5 seeds with no interval — remains fair and is
subsumed by Blocking 27.

**Blocking 22 (missing training details) — PARTLY.** Already specified: AdamW,
lr 1e-3, weight decay 1e-4, gradient clipping 1.0, dropout 0.1, d_model 256, one
block, 4 heads, T=7, batch size 48, 3 epochs. Genuinely absent: the LR *schedule* and
warmup, numerical precision, and document-boundary packing. A writing gap, and
recoverable from the frozen spec and the per-run `resolved_config.json` files.

## Claims already disclosed by the paper

Disclosed is not the same as resolved. These are real limitations the paper states
about itself — a mark against the reviewer's implication that they were concealed,
not against the science.

- **Blocking 5** (fixed-depth-trained control unrun): stated verbatim, and the declared
  Holm slot is deliberately left consumed and unfilled.
- **Blocking 20** (hardware/revision confound): stated in Disclosures with the measured
  -0.117 / -0.119 sigma reproduction and the 0.000185-nat gap movement. The
  unaddressed part is ACT-threshold numerical sensitivity across revisions.
- **Blocking 26** (OOD set redefined): the capped re-scoring rule was fixed before
  scoring and BOTH sets are reported.
- **Errors 5** (section 7.6 truncated): already fixed at T-LA.24; the body now carries a
  self-contained disclosures paragraph.

## Damage assessment, ordered by how much it threatens the paper's claims

### Tier 1 — invalidates a headline number or the title claim

| Item | What breaks |
|---|---|
| **Blocking 9** (unembedding dominates) | The shared output projection is 4,194,304 of MoE's 5,249,536 FLOPs/token = **79.9%**. Remove it and the ratios go MoE 1.00x, MoRE **6.92x** (not 2.19x), MoR **40.1x** (not 8.87x). Every "2.19x", "8.87x" and "25% of MoR's compute" statement is deflated by a constant common to all arms. The most serious single finding in the review, and disclosed nowhere. |
| **Blocking 10** (active-parameter confound) | Total params identical at 5,584,908 across all three arms, but MoR activates **3,145,728** FFN params/token (one 24-wide block) against **524,288** for MoE and MoRE (one of six 4-wide experts) — exactly **6x**. MoR is the lowest-loss arm. "MoR wins because it does 6x the FFN work per token at matched total parameters" is an unexcluded explanation for the entire +0.1216-nat MoRE-MoR gap, and the paper never raises it. |
| **Blocking 23** (adaptive depth barely exists) | `depth/mean`: MoR **6.7297**, MoRE **6.9552** against T=7. `depth_dist/step_7_pct`: MoR **81.2%**, MoRE **97.3%**. MoRE halts at the maximum budget for 97% of tokens. "Adaptive Compute" in the title is close to unsupported by the canonical runs; consistent with the pre-registered Family B depth-difficulty correlation being ~0.028 and near-zero in magnitude. |
| **Blocking 1** (forced-depth readout) | **Confirmed in code, disclosed in one clause, still load-bearing.** `code/more/model.py` under `fixed_depth=True` sets `step_weight = stop_local`, which is 1 only at the final step and 0 at every earlier step, so the ACT-weighted accumulator **collapses to the last hidden state**. The paper states this ("the output is the final state, in place of the halting-weighted mixture") but then still reads the 1.471-nat MoR gap as measuring reliance on the learned policy. Two variables move at once; the gap is not attributable to input-dependent depth alone. |

### Tier 2 — real logical or factual errors in the paper's own reasoning

- **Blocking 16 (collinearity sign) — CONFIRMED.** `dynamic.json`
  `update_collinearity_mean` is **negative at every step**: MoR -0.262 rising through
  -0.147; MoRE -0.267 through -0.051. An anti-parallel update relative to the state is
  a contraction signature. The paper tests only for **+1** ("positive collinearity
  approaching +1" would indicate rescaling) and calls the observed values
  "near-orthogonal". `code/eval_dynamic.py`'s own docstring says "near +-1 = pure
  rescale" — the code's criterion includes the negative branch, the paper's does not.
- **Blocking 15 (trapping logic) — CONFIRMED.** Velocity decays monotonically
  (MoR 11.06 -> 5.71, MoRE 9.09 -> 3.49) and is read as evidence the recursion is
  "dynamically healthy". Monotone settling is equally what fixed-point trapping
  predicts. The observation does not discriminate between the two hypotheses, so it
  cannot refute either.
- **Blocking 13 (internal contradiction) — CONFIRMED.** Section 5.2 states that mean
  depth alone cannot establish adaptivity; section 6.3 then argues from mean depth.
- **Errors 2 (Fig. 5 caption) — CONFIRMED, factually wrong.** The caption reads "MoRE
  is worst in domain (a) and best on the Pile (b)". In-domain `val/task_loss` is MoE
  3.9483, MoRE 3.6370, MoR 3.5154 — **MoE is worst**, not MoRE.
- **Errors 1 (stale intro OOD) — CONFIRMED.** The introduction quotes "MoRE reaches
  7.573 nats against MoR's 7.802", the superseded earlier Pile scoring; Table 2
  reports 7.212 / 7.423 from the capped set. Both sets appear in the results section,
  but the headline in the intro is the older pair.
- **Blocking 27 (p-floor) — CONFIRMED.** `min_p = 0.00794` over 252 permutations; all
  three declared comparisons return exactly that, `at_floor = true`. Effect sizes are
  enormous (Cohen's d 4.7 to 15.9), so the ORDERING is not in doubt — but no
  comparison can ever be more significant, and "significant" should not be the word.
- **Blocking 21 (hyperparameter contradiction) — CONFIRMED.**
  `updated_objective.md`'s CANONICAL HYPERPARAMETER PRINCIPLE forbids per-arm tuning
  until an arm wins; freezing one configuration *per arm* from a grid is per-arm
  selection. Real tension with the project's own stated rule.
- **Blocking 4 / Errors 11 (compliance with the project's own plan) — CONFIRMED.**
  `updated_objective.md` REQUIRED ABLATIONS lists "Parameter-matched MoR" as required;
  it is unrun. `train/family_cls_loss` is exactly 0.0 across all 5 seeds in all 3 arms,
  i.e. a structurally-zero term still carried in the loss display (Eq. 6).
- **Blocking 24 / Errors 4 (hedging conflict) — CONFIRMED.** The abstract says
  "significant at our resolution floor" and asserts OOD dominance; the body says
  "weaker than significance" and disclaims uncorrected results. The abstract should
  carry the body's more careful wording.
- **Blocking 28 (Family B missing) — CONFIRMED.** The pre-registered depth-difficulty
  correlation (`depth/spearman_vs_logfreq` 0.028, `depth/spearman_vs_model_loss`
  -0.023) is appendix-only.

### Tier 3 — valid methodological criticism, unfixable without new runs

Blocking 3 (MoE is depth-1 by construction), 6 (no training-FLOPs parity), 7 (three
points cannot establish a frontier), 8 (MoRE wants more depth than T=7), 12 and 14
(the section 6.3 ponder-weight grid is 3 points, 1 seed, different corpus), 17 (a
no-recursion arm cannot control a within-recursion test), 19 (ponder weight admitted
non-transferring), 25 (OOD losses 7.2-7.9 against a uniform-loss ceiling of 9.01, so
the task is barely learned out of domain and the ordering may be calibration).

## Claims that are wrong or overstated

- **Blocking 2** — refuted above; a garbled reading, not a defect.
- **Blocking 18** — the numbers are real; only the *definition* is missing.
- **Blocking 22** — mostly already specified.
- **Errors 9** (App. F accounting) and **Errors 10** (Eq. 1 indexing) — cosmetic.
- **Related work 2 and 3** — correct. "MoR" collides with Bae et al.'s Mixture-of-
  Recursions, and Mixture-of-Depths is layer-skipping rather than weight-tied block
  reuse, so the related-work paragraph conflates two different mechanisms.
- **Related work 1** (Sparse Universal Transformer, MoEUT) — **UNVERIFIABLE from this
  repository.** If either does combine MoE with recursion, the "not aware of a matched
  comparison" novelty claim is weakened. Needs a literature lookup before it is
  conceded or denied; do not treat it as settled.
- **Presentation 1-4** — all four are fair. Repetition, abstract/body hedging
  mismatch, duplicated appendices, and the absence of an anonymised repository are all
  real and all cheap to fix.

## What the review does NOT overturn

Stated plainly, because a review this large should not be allowed to imply that
everything is now in doubt:

- The **in-domain ordering** MoR (3.5154) < MoRE (3.6370) < MoE (3.9483) is robust.
  Every gap sits far outside seed spread (std 0.013-0.034) with Cohen's d from 4.7 to
  15.9. The Tier-1 findings attack the *interpretation* of that ordering, not the
  ordering itself.
- The **equal-tokens-per-parameter arithmetic** refuting the narrow dilution account is
  correct as arithmetic. Only Blocking 11's charge of rhetorical overreach stands.
- **n=5, five frozen seeds, the exact randomization test, Holm correction, and full
  per-run provenance** are all correctly implemented. The p-floor is a design limit that
  was found and reported, not a defect introduced by analysis.

## The honest summary

The paper's *measurements* are in good order. Its *interpretations* carry three
specific overreaches a reviewer will find: the FLOPs comparison is confounded by a
shared constant the paper never mentions; the MoR-MoR gap has an unexcluded
active-parameter explanation; and the two mechanism refutations (trapping, collinearity)
use tests that do not discriminate between the hypotheses they claim to separate.
Underneath all of it, the depth result says something more uncomfortable than the
title claims: both recursive arms sit at their ceiling, so "adaptive compute" is the
weakest part of the story, not the strongest.
