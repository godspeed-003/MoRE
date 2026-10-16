# Reply to PAPER_REVIEW / PAPER_MITIGATION (T-LA.19)

Written 2026-10-15 on Vedant's machine, against `d114337`. Every claim in the
brief that is checkable from the repo was re-derived here before being accepted.
**Two claims did not survive**; both are corrected below with the actual numbers,
because the remedies the brief proposes for them are aimed at the wrong target.

The latest paper is `paper/tmlr_submission/` (assembled from `paper/common/` —
edit `common/`, never `*_submission/sections/`, which `assemble_submissions.py`
regenerates and silently overwrites).

---

## UPDATE 2026-10-16 — READ FIRST. §3.1, §3.4, §3.5 and §6 below are SUPERSEDED

Vedant's decisions after reading this reply, plus **two errors in the reply
itself**, found while re-checking the points Vedant pushed back on. Sections
below are left as written so the reasoning trail survives; where they conflict
with this block, this block wins.

### U1. TMLR goes UNDER 12 pages, with headroom (reverses §3.1)

Vedant's reason: the ≤12-page track (~2-week review) is what makes a
reject-and-resubmit cycle fast. That only works with **headroom** — a body at
11.95 goes over on the first reviewer-requested paragraph. **Target ≈ 11.5–11.7.**

Baseline, measured on the `d114337` build: **13.43** (appendix-first landmark,
per T-LA.18). **Every pre-built lever is already pulled**: all 11 ACLMOVE blocks
are in `TMLR_APPENDIX`. So the savings need NEW moves. Body page map (pp):

| block | pp | note |
|---|---|---|
| Reproducibility Statement | 0.73 | restates §4 Setup + §7.6; TMLR does not require it in the body |
| §7.6 Disclosures | 0.62 | Vedant already questioned this as "history of the study" |
| §5.3 (incl. Table 1 + Fig 3) | 1.86 | Fig 3 (Pareto) renders 6.50 in wide |
| §3.3 (incl. Fig 1) | 1.61 | **Fig 1 alone is 6.50 × 4.70 in = 0.43 of a page** |
| Broader Impact | 0.23 | optional for TMLR at this risk level |

Proposed moves — all to the appendix, nothing deleted:

| move | est. saving |
|---|---|
| Reproducibility Statement → appendix (first appendix section) | −0.73 |
| §7.6 Disclosures → 3-line body summary, full text → appendix. The summary MUST keep the integrity-critical facts: two arms retrained on a different GPU after checkpoint loss (−0.117σ / −0.119σ), four code revisions, and that `mor_vs_moe` was declared after its data existed (`confirmatory_tests.json` `_README` requires the paper to say so) | ≈ −0.50 |
| Broader Impact → 3 sentences | ≈ −0.10 |
| Figs 1 and 3 at ≈0.7 `\linewidth` in TMLR only (they were drawn for a 3.35-in ACL column; at full 6.5-in TMLR width their labels render at 14–15 pt against 10 pt body text). Needs a per-venue width macro, like `\dispsize`. **Vedant's call — he has expressed a preference for large visuals** | ≈ −0.25 |
| De-duplicate §6.1's closing paragraph against §7.1 (both state the scale prediction) | ≈ −0.10 |
| New body text from the work order (learning curves, OOD disclosure, §6.2↔§6.3 bridge, E=1 reframe, FLOPs pointer) | ≈ +0.25 |
| **net** | **≈ 11.7** |

Estimates come from the page map. **Measure each move on a real build** — float
placement in this paper is non-monotone (shrinking a float once added 0.74 pp).

### U2. NO new seeds — n=6 and n=7 both withdrawn (reverses §3.5). The "7 comparisons" premise is a MISCOUNT, and this reply endorsed it

`code/confirmatory_tests.json` declares **six** tests: family A `m=4`, family B
`m=2`. The 7 originates in `changelog.md` ~L7393, a hypothetical about a FUTURE
revision ("the fixed-depth sweep alone is up to 7 comparisons"); the same entry
says "The paper sits at 6 tests". It then propagated into: the
`confirmatory_tests.json` `correction_note`, `code/seed_stats.py` docstrings
(L63, L214), `paper/common/02_setup.tex` (holm_detail — body short form AND the
appendix long form), `PAPER_GUIDE.md:564`, `PAPER_MITIGATION.md:227`, and §1
row 5 / §3.5 of this file.

Arithmetic with the true count:

| threshold | value | floor 0.00794 clears it? |
|---|---|---|
| Holm step 1, family A (m=4) | 0.0125 | yes |
| Holm step 1, family B (m=2) | 0.025 | yes |
| plain Bonferroni, ALL SIX declared tests pooled | 0.00833 | **yes** (margin 0.0004) |
| plain Bonferroni at m=7 (no 7th test exists) | 0.00714 | no |

Recomputed here by exhaustive enumeration over all 252 regroupings:
`more_vs_moe`, `mor_vs_moe`, `more_vs_mor` (val/task_loss) and
`depth_more_routeonce_vs_mor` (depth/spearman_vs_model_loss) — **all p =
0.00794**. The other two (`more_vs_depth_matched_fixed`,
`depth_routeonce_vs_perstep`) need the forced-depth and per-step arms; the paper
reports both at the floor; not re-derived here.

**A second error in the same paper paragraph, independent of the count.** It says
Holm is "load-bearing" by comparing plain Bonferroni at m=7 *pooled* against Holm
at m=4 *per family*. Holm's first-step threshold at a given m EQUALS Bonferroni's
at that m, so with every p at the floor the two give identical verdicts. The
quantity that sets the threshold is the **family size**, not the correction
method. A statistician reviewer will mark this; it is exactly TMLR's
technical-correctness axis. Replacement substance for `holm_detail`:

> Holm's first-step threshold equals plain Bonferroni's at the same m, so at our
> resolution floor the two give identical verdicts; what sets the threshold is
> the family size. Within the declared families it is 0.0125 (A) and 0.025 (B).
> Pooling all six declared comparisons into one family gives 0.05/6 = 0.00833,
> which the floor of 0.00794 still clears — by 0.0004. The margin is thin: a
> seventh confirmatory comparison would put the floor above the threshold, which
> is one more reason no comparison may be added to a family after results exist.

**Do NOT edit the frozen pre-registration text.** Add a dated
`_ERRATUM_2026_10_16` key to `confirmatory_tests.json` stating that the
`correction_note`'s "7 comparisons" is a miscount of the declared six, with the
table above. `families`, `family_size` and `correction` are correct and unchanged.
Code docstrings in `seed_stats.py` are not frozen — fix them directly.

### U3. OOD: §3.4's reason for re-scoring was WRONG; there is a different, stronger one

§3.4 argued from sample size. Wrong: all 15 cells score the identical blocks, the
reported ± is seed variance, and more blocks shrink only evaluation-sample noise,
which is common to all arms and cancels in the between-arm gap. Vedant's
objection is right on that axis — **the scoring itself was done correctly.**

But the COMPOSITION of the scored set, reproduced exactly (`eval_ood.py` path:
first 4000 non-empty documents of `NeelNanda/pile-10k`, EOT-packed, first
1500 × 256 blocks):

| Pile subset | docs | tokens | share |
|---|---|---|---|
| **Books3** | **1** | **223,739** | **58.3%** |
| PubMed Central | 2 | 37,465 | 9.8% |
| Pile-CC | 21 | 30,064 | 7.8% |
| StackExchange | 20 | 21,308 | 5.5% |
| FreeLaw | 2 | 20,413 | 5.3% |
| OpenWebText2 | 23 | 19,847 | 5.2% |
| USPTO Backgrounds | 5 | 12,160 | 3.2% |
| Github | 7 | 10,167 | 2.6% |
| PubMed Abstracts | 13 | 4,249 | 1.1% |
| Wikipedia (en) | 5 | 2,263 | 0.6% |
| NIH ExPorter / HackerNews / Enron | 3 | 2,325 | 0.6% |
| **total** | **102** | **384,000** | |

**58.3% of the OOD evaluation tokens are one book.** The abstract's "Out of
domain it is stronger still---it dominates recursion alone" is therefore mostly a
statement about one Books3 document, and the paper describes it as "held-out
Pile blocks". Two more provenance defects in the same artifact:

* `pile-10k` is a sample of the Pile's **train** split, and `eval_ood.py` ignores
  `--split` for `pile` — yet `results/language/ood_pile_test.json` is labelled
  `pile:test`. ("Held-out" is still true in the sense that matters: our models
  never trained on any Pile text.)
* 5 documents (0.6%) are Pile's **Wikipedia (en)** subset — in-domain by
  construction for a WikiText-103 model.

**Proposal (pending Vedant): a document-capped re-score.** ≤ 2 blocks per
document, Wikipedia (en) excluded, 1500 blocks from ≥ 750 documents, all 15
checkpoints, reported BESIDE the original with per-subset shares. Inference only,
local, minutes. If the inversion holds, the paper's one "neither mechanism alone"
claim becomes its best-supported OOD number; if it does not, the abstract's second
paragraph changes, and Outcome-C honesty (CLAUDE.md §8) applies. Disclosing the
composition is required either way — the dataset is public and a reviewer can
reproduce this table in five minutes.

### U4. Another artifact landmine

`results/language/results.json` → `admitted_runs[*].metrics_nonscalar.
routing_permutation_invariant.per_family` carries **arithmetic** labels
(`E1_ADD_SUB` … `E6_SORT_STAT`) on language runs. The supports (77,954 / 83,944 /
16,320 / 17,985 / 24,269 / 57,044; sum 277,516 ≈ the validation token count)
are consistent with the language POS families, so this looks like a naming defect
in the metric layer's manifest lookup rather than wrong numbers — **but the
index → family mapping is not yet verified.** CLAUDE.md §2 requires every label to
derive from one family manifest. Verify and fix (or document) before the artifact
ships; it belongs in the same README note as the 7.615 (§2.2).

### U5. Work order — replaces §6

**Correctness** (a reviewer would mark these wrong):
1. `holm_detail`: six not seven; family size, not Holm, sets the threshold (U2).
   Plus the `_ERRATUM` key and the `seed_stats.py` docstrings.
2. GPU contradiction (§1 row 1a).
3. FLOPs definition in the appendix; §4.3 points at it (§1 row 1c).
4. OOD: disclose source, n, document count and composition (U3); re-score if
   approved.
5. §6.2 ↔ §6.3 bridging sentence (§1 row 3b).

**Free wins** (already measured, no compute):
6. Per-family depth table, corrected MoR range (§2.3); shorten the ρ defence.
7. Learning curves in §7.2, as corroboration of §6.1 (§3.3).
8. E=1 reframe in §6.1 **with the router confound stated** (old §6 item 9).
9. N+R footnote (§2.2) and the U4 landmine, in the artifact README.
10. Freeze-note one-line correction (§2.1) — repo only.

**Page count** (U1): the moves table, measured build by build.

**Compute:** the U3 capped re-score only, if approved. No training runs.

---

## 1. Verification table

| # | Claim | Verdict | Evidence |
|---|---|---|---|
| 1a | `02_setup` "single laptop GPU" contradicts Limitations "two GPUs" and the TMLR repro statement's V100 | **CONFIRMED** | `02_setup.tex:48` vs `06_limitations.tex:107` vs `tmlr_submission/main.tex:165-168` |
| 1b | `canonical_spec_language.json` says 5.2406 / 2.63 / 0.2907 for the (0.001, 0.001) dev point | **WRONG AS STATED** — see §2.1 | those three numbers co-occur nowhere |
| 1c | FLOPs carry the efficiency result but `\ref{sec:metrics}` resolves to a section that never defines them | **CONFIRMED** | `02_setup.tex:62-92` covers loss, routing, depth — zero occurrences of "FLOP" |
| 2 | `val/avg_recursion_steps` = 7.615 exceeds `max_depth` 7; it is expected depth N+R, not a bug | **CONFIRMED in substance, "agree exactly" overstated** | see §2.2 |
| 3a | MoR *is* the matched-width E=1 point of an expert-count sweep | **CONFIRMED** | resolved configs: MoE 6x4, MoR 1x24, MoRE 6x4 — total FFN width 24 in all three, both recursive arms at `max_depth` 7 |
| 3b | §6.2 (ponder weight does not transfer) and §6.3 (explains full-corpus behaviour with a small-corpus grid over that weight) are in tension and the paper never says so | **CONFIRMED** | `05_mechanism.tex:70-112` vs `115-147` |
| iii | All 15 admitted runs carry 3-point val curves; MoRE is improving fastest at cutoff | **CONFIRMED, numbers exact** | see §3.3 |
| iv | OOD rests on 1500 Pile blocks and the paper never states it | **CONFIRMED — and 58.3% of those tokens are ONE Books3 document**, see U3 | `results/language/ood_pile_test.json: n_blocks 1500`; x256 = 384,000 tokens |
| free | Per-family depth is the central claim shown directly | **CONFIRMED, MoR range wrong** — see §2.3 | real spreads: MoR 0.3168, MoRE 0.0090 |
| 5 | n=6 drops the resolution floor to 0.00216 (3 cells, not 6) | **CONFIRMED but MOOT** — see U2 | 2/C(12,6) = 2/924 = 0.0021645; n=7 → 2/3432 = 0.00058 |
| 5 | Plain Bonferroni at m=7 is 0.00714, below the n=5 floor of 0.00794 | **WITHDRAWN 2026-10-16 — arithmetic true, premise false: there are SIX declared tests, not seven** | see UPDATE U2 |
| 7 | All 15 checkpoints exist, so re-scoring is inference-only | **CONFIRMED** | 15/15 `checkpoint.pt`, 0.34 GB total |

Also confirmed: nothing hard-wires 6x4 (answer to Ayan's engine question, §4.1).

---

## 2. The two corrections

### 2.1 The spec mismatch is real but much narrower, and the guard is untouched

The brief says the *frozen spec* "still says 5.2406 / 2.63 / 0.2907" and asks for
a dated amendment "since you froze it". That overstates it in a way that matters,
because it implies a frozen **field** disagrees with the paper — which would put
the proxy guard and every canonical admission in question. It does not.

What is actually true:

* `0.2907` appears **once**, in prose, at `canonical_spec_language.json`
  `_T_L7_1_FREEZE_NOTE[11]`: *"AMI falls 0.2907 -> 0.0025"*. It is a
  leading-underscore note key. `enforced_fields` and `architecture_variants` —
  the only blocks `assert_not_silent_proxy()` consults — are **unaffected**. No
  canonical claim, no admitted cell, and no guard behaviour changes.
* `0.2907` appears in **no data file at all**. The weights grid says 0.2851.
  So this is an *unsourced number in a prose note*, not a competing measurement.
* `5.2406` and `2.63` are real, but they come from a **different grid**:
  `lang_calibration_reg.json` (stage `reg`, 6 rows, a regularisation sweep that
  records **no AMI key whatsoever**). The paper's §6.3 table correctly uses
  `lang_calibration_weights.json` (stage `weights`, 3 rows) → 5.2352 / 2.6976 /
  0.2851.
* The note's `0.0025` endpoint **agrees exactly** with the paper's table.
* The note's depth "1.49" is the grid's 1.4761, i.e. rounded wrong (1.48).

So the defect is: **one prose line in a freeze note cites an AMI that no file
supports, and rounds a depth wrong.** The fix is a one-line correction of the
note to `0.2851 -> 0.0025` and `1.48`, with a dated `_AMENDED` entry recording
that the original line quoted an unsourced figure. Not a spec amendment
ceremony — there is no frozen field to amend. "Fix the loser" was never the
right instruction either, because the two files are not competing measurements
of the same thing; they are two different sweeps and both are correct.

**Do not** reconcile `lang_calibration_reg.json` to the paper. It measures a
different axis and has no AMI column.

### 2.2 The N+R identity holds to ~0.007, not exactly

The diagnosis is right and the key is not a bug: `avg_recursion_steps` is the
ACT ponder cost `N + R`, which can exceed `max_depth` by up to 1 by construction.
But the brief's "those two agree exactly" is not what the data says:

| arm | `val/avg_recursion_steps` | `halt/mean_remainder` | difference | paper's mean depth |
|---|---|---|---|---|
| MoR | 6.8985 | 0.1750 | 6.7234 | 6.7297 |
| MoRE | 7.6152 | 0.6549 | 6.9603 | 6.9552 |

Residuals 0.0063 and 0.0051. They disagree at the third decimal because the two
quantities are measured on **different passes over different populations** —
`val/*` is the online validation pass; `depth/*` is the offline pass that excludes
each block's last position (no next-token target). `depth_key_provenance` in every
run's `metrics.json` already documents exactly this, and says the two "will not
agree". A footnote claiming exact agreement would be falsified by a reviewer who
subtracts, so phrase it as *consistent to 0.006, the two passes differing in
population as `depth_key_provenance` records*.

### 2.3 Per-family depth: MoRE's range is right, MoR's is not

The brief's MoRE figures are exact. Its MoR range is not — and `6.572` is
suspicious because that literal value is
`depth/by_document/per_document_mean_max` in MoR seed 42, a different key.

Re-derived from `depth/mean_by_family/*`, mean over all 5 seeds per arm:

| family | MoR | MoRE |
|---|---|---|
| L1_FUNCTION | 6.9069 | 6.9601 |
| L2_NOUN | 6.5901 | 6.9531 |
| L3_VERB | 6.8798 | 6.9572 |
| L4_MODIFIER | 6.7557 | 6.9551 |
| L5_PUNCT_SYM | 6.8594 | 6.9619 |
| L6_NUM_SUBWORD | 6.6103 | 6.9529 |
| **spread** | **0.3168** | **0.0090** |

Sanity check passed: `depth/mean` aggregates to MoR 6.7297 ± 0.2410 and MoRE
6.9552 ± 0.0044, matching the paper's 6.730 and 6.955.

**The conclusion is unchanged and stronger than the brief claims: MoR's depth
varies 35x more across linguistic families than MoRE's.** Use 6.590–6.907 and
0.3168, not 6.572–6.887.

This is the single best free win in the brief. It converts the paper's weakest
passage — a paragraph defending why ρ = −0.023 is *uninformative* rather than
negative evidence — into a positive demonstration: MoRE assigns every one of six
families a depth inside a 0.009 band. That is non-allocation shown, not an
absence of evidence explained away. **Add the table and keep the ρ paragraph
short**, pointing at the table instead of carrying the whole defence.

---

## 3. Judgement calls — answers

### 3.1 (i) TMLR page count: ~~stay over 12~~ — SUPERSEDED by U1: go under 12, with headroom

Measured now: **13.43 pages to References** (`code/measure_counted_pages.py`,
which as of `d17b035` knows TMLR's rule — single-column, Limitations counts,
count to References — rather than silently applying ACL's).

Stay over, for three reasons. TMLR sets **no hard limit**; the only cost is
review latency, already priced in and accepted (no target venue for ~2 months).
The items this review adds that are worth adding — the FLOPs appendix above all —
make the body *longer*, and cutting to 12 would mean cutting the material that
makes the paper defensible on the axis TMLR actually judges. And float packing
here is non-monotone: shrinking a float has already *increased* the page count
once in this paper (8.09 → 8.83, reverted), so "get it to 12" is not a cheap
mechanical operation.

Decision: **do not spend effort on the page count. Put the new material in the
appendix, which is page-exempt.**

### 3.2 (ii) ACL: moot, not deferred — check the dual-submission line first

Before anyone touches 8.29 or the 2 overfull hboxes: **TMLR requires that
submitted work not be under review elsewhere.** If that reading holds, ACL is not
a parallel track at all and the 8-page problem cannot be worked on while TMLR is
live. That makes item (ii) zero work rather than deferred work.

Verify the exact policy line before acting. If it holds — which I expect — the
answer to "fix it or drop ACL" is **neither: it is blocked.** Revisit only if
TMLR rejects, at which point the paper will have changed anyway and today's 0.29
page overage is not worth predicting.

This kills the one item in the brief that carried desk-reject risk.

### 3.3 (iii) Learning curves: publish, and publish them as a prediction

Re-derived exactly as the brief reports:

| arm | epoch 1 | epoch 2 | epoch 3 | final-epoch drop |
|---|---|---|---|---|
| MoE | 4.1453 ± 0.0056 | 4.0392 ± 0.0324 | 3.9483 ± 0.0242 | +0.0909 |
| MoR | 3.8202 ± 0.0389 | 3.6472 ± 0.0328 | 3.5154 ± 0.0340 | +0.1318 |
| MoRE | 3.9416 ± 0.0338 | 3.8131 ± 0.0211 | 3.6370 ± 0.0134 | **+0.1761** |

Publish. Three reasons, and the third is why this is an asset rather than a
concession.

1. A reviewer derives it from the artifact in ten minutes. Volunteering it costs
   nothing we still hold.
2. Limitations already concedes the budget point in words; this is the number
   that was missing from a concession we were already making.
3. **It independently corroborates the §6.1 mechanism.** The diversity-dilution
   hypothesis predicts MoRE's deficit shrinks as tokens-per-expert grows. The
   curves say MoRE is furthest from convergence at cutoff. Two independent
   measurements, same direction. That turns §6.1 from "the only account left
   after we excluded the arithmetic one" into an account with a positive
   prediction that has now been met twice.

State the per-epoch differential (MoRE gains 0.044/epoch more than MoR at
cutoff, against a 0.1216 gap) and stop there. **Do not name a crossing epoch** —
curves decelerate and that number would be invented. The defensible sentence is
*the in-domain ordering is budget-dependent and we did not run to convergence*,
which is a statement of fact about our own design.

One more thing the curves give free: MoRE's seed spread **tightens** across
epochs (0.0338 → 0.0211 → 0.0134) while MoR's is flat (~0.033). That is direct
support for the parsimonious reading the paper already prefers for the
variance-ratio result — a collapsing policy has less left to vary — so cite it
there rather than leaving the hedge unsupported.

### 3.4 (iv) OOD re-score — SUPERSEDED by U3: the sample-size reason below is wrong; the composition reason in U3 replaces it

All 15 checkpoints are present (0.34 GB, verified), so this is inference-only and
local — no Vast.ai, no training.

Do it, and state `n` in the paper either way. Two caveats to carry:

* The comparison is **paired** — all three arms score the identical 1500 blocks —
  which makes 384k tokens considerably less fragile than an unpaired n of that
  size, and is worth one clause in the text.
* The inversion may not survive. That is the point of running it now. The gap is
  0.229 nats (MoRE 7.573 vs MoR 7.802) against stds of 0.122 and 0.158, so it is
  not a knife-edge, but "MoR is Pareto-dominated" is the paper's one
  *neither-mechanism-alone* claim and it should be the best-supported number in
  the paper rather than the thinnest.

`--max_blocks` already defaults to 2000, so raising it is a flag, not a code
change.

### 3.5 (5) GPU spend — SUPERSEDED by U2: no runs; the 3-run argument below rests on a miscount

Agreed that documentation plus free appendix tables is the right scope, with one
exception, and the brief has already found the argument for it — it is just
stronger than the brief states it.

**n=6 is not about reporting a smaller p-value. It is about surviving a reviewer
who uses plain Bonferroni.** At n=5 the floor is 2/252 = 0.00794. Plain
Bonferroni over our 7 comparisons is 0.05/7 = 0.00714. **The floor sits above the
threshold**, so at n=5 *no comparison in this paper can reach plain-Bonferroni
significance no matter how large the effect* — a 10-nat gap would still give
0.00794. We escape only via Holm's first step at 0.0125. That is a single point
of failure on our entire statistical claim, and it is a methodological
preference, not a bug, so we cannot argue our way out of it.

Three runs (one extra seed x three arms) take the floor to 0.00216 and remove
that failure mode permanently. That is the cheapest risk reduction available
anywhere in this project.

On a second model scale: agreed, it is the wrong experiment, and §3.3 above
sharpens why. Our mechanism now has two independent lines of support pointing at
tokens-per-expert, and the subset-fraction machinery already exists from the
saturation result. MoR+MoRE at 3 fractions x 3 seeds tests **our own claim** on
existing machinery; a second parameter scale answers a generic objection we can
instead answer in prose as a falsifiable prediction, which §7 already does.

---

## 4. Ayan's two direct questions

### 4.1 Does the engine accept arbitrary (num_experts, ffn_mult)?

**Yes — nothing hard-wires 6x4.** `code/derive_lang_ffn_mult.py` already
parametrises `build(num_experts, ffn_mult, ...)` and sweeps the width axis; it is
the script that derived MoR's 24 in the first place. The only 4s and 6s in the
engine path are config defaults, and `CLAUDE.md` §2 requires every tensor dim to
derive from `num_experts`, which Gate L0 suite 4 enforces (129 checks).

Integer-matched points on the width-24 budget: **E=1 → 24, E=2 → 12, E=3 → 8,
E=4 → 6, E=6 → 4**. The brief's E=2 → 12 and E=3 → 8 are right.

### 4.2 Footnote the 7.615, or rename the exporter keys?

**Footnote now; do not rename.** Renaming is cleaner in the abstract and wrong
here, for a reason specific to this repo: `depth_key_provenance` is already
written into all 15 admitted `metrics.json` files and documents these key names
and their populations verbatim. Renaming the live keys makes every archived
`metrics.json` disagree with the exporter — we would be buying a clean name at
the cost of a provenance mismatch across the whole matrix, which is the more
expensive kind of confusion and exactly what `CLAUDE.md` §5 exists to prevent.

So: footnote in the paper, one line in the artifact README, **and** add
`val/avg_recursion_steps` to `depth_key_provenance`'s own note set so the
explanation ships inside the data rather than only in the paper. Deleting the
keys was correctly ruled out.

If the name is to change, it changes once, at the next matrix — not under a
submission.

---

## 5. Only Vedant can answer

**652J concern 2 is recorded nowhere in the repo.** Concerns 1, 3 and 4 each have
a script quoting the reviewer verbatim in its header (`diag_halt_gradient`,
`flops_accounting`, `eval_test_split`); concern 2 has nothing. If the original
review still exists, paste concern 2 in and it gets the same treatment. An
unanswered prior objection resurfacing in review is avoidable damage, and this is
the one item in the whole brief that no amount of repo archaeology can resolve.

---

## 6. Work order — SUPERSEDED by U5

Unambiguous, verified, no judgement — do these:

1. **1a** — one GPU sentence. `02_setup.tex:48` is the wrong one; make it defer
   to Limitations rather than asserting a single GPU.
2. **1c** — the FLOPs appendix. Biggest gap in the paper relative to the work
   actually done: `flops_accounting.py` measures a real forward pass rather than
   evaluating a typed formula, verifies linearity in depth to 0.1% as a hard
   failure, evaluates at each arm's measured mean depth, and includes attention,
   embeddings and the LM head. None of that is in the paper while FLOPs appear 17
   times and carry the entire efficiency headline. Half an appendix page.
   Appendices are page-exempt in both venues.
3. **Per-family depth table** (§2.3) — with the corrected MoR range, and shorten
   the ρ defence to point at it.
4. **Learning curves** (§3.3) — in Limitations or §6.1, with the differential and
   no crossing-point extrapolation.
5. **OOD n** — state 1500 blocks / 384k tokens and that the comparison is paired,
   independently of whether the re-score happens.
6. **2.2** — footnote the N+R identity at ~0.006, not "exactly"; extend
   `depth_key_provenance`.
7. **3b** — one sentence distinguishing non-transfer of a *level* from transfer of
   a *trade-off*, placed at the top of §6.3 so the reader meets the resolution
   before the tension.
8. **2.1** — correct the freeze note's `0.2907` → `0.2851` and `1.49` → `1.48`,
   with a dated `_AMENDED` line. No spec field changes; the guard is untouched.
9. **3a reframe** — present MoR-vs-MoRE as two points (E=1, E=6) on a
   matched-width expert sweep. **Carry the confound explicitly**: the E=1 endpoint
   also removes the router entirely (its parameters, the gate multiply, the
   balance term), so this is two points on the axis with one endpoint differing in
   a second way — not a clean sweep. An E=2 or E=3 point would have a router and
   would close that gap. Stating the confound is what makes the reframe survive
   review; omitting it is what would sink it.

Then, if approved: the OOD re-score (§3.4, local, inference-only) and the 3 runs
for n=6 (§3.5).
