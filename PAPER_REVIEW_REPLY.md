# Reply to PAPER_REVIEW / PAPER_MITIGATION (T-LA.19)

Written 2026-10-15 on Vedant's machine, against `d114337`. Every claim in the
brief that is checkable from the repo was re-derived here before being accepted.
**Two claims did not survive**; both are corrected below with the actual numbers,
because the remedies the brief proposes for them are aimed at the wrong target.

The latest paper is `paper/tmlr_submission/` (assembled from `paper/common/` —
edit `common/`, never `*_submission/sections/`, which `assemble_submissions.py`
regenerates and silently overwrites).

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
| iv | OOD rests on 1500 Pile blocks and the paper never states it | **CONFIRMED** | `results/language/ood_pile_test.json: n_blocks 1500`; x256 = 384,000 tokens |
| free | Per-family depth is the central claim shown directly | **CONFIRMED, MoR range wrong** — see §2.3 | real spreads: MoR 0.3168, MoRE 0.0090 |
| 5 | n=6 drops the resolution floor to 0.00216 (3 cells, not 6) | **CONFIRMED** | 2/C(12,6) = 2/924 = 0.0021645; n=7 → 2/3432 = 0.00058 |
| 5 | Plain Bonferroni at m=7 is 0.00714, below the n=5 floor of 0.00794 | **CONFIRMED and it is the sharpest argument in the brief** | 0.05/7 = 0.0071429 < 2/252 = 0.0079365 |
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

### 3.1 (i) TMLR page count: stay over 12

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

### 3.4 (iv) OOD re-score: yes, do it

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

### 3.5 (5) GPU spend: 3 runs, and the reason is narrower than "more power"

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

## 6. Work order

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
