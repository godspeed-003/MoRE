# PAPER_MITIGATION.md — problem register and mitigation plan, 2026-10-15

Companion to [PAPER_REVIEW.md](PAPER_REVIEW.md), which establishes the findings.
This file is the **decision document**: every problem, what it costs to fix, the
options, and a recommended strategy.

**Framing.** Nothing here is a defect in the science. The measurements hold —
every number in the paper reconciles with `results/language/results.json`. The
register below is dominated by work that was *done well and not written down*,
plus three internal contradictions and four structural limits that no amount of
writing can remove.

---

## 1. Problem register

Severity: **H** = a reviewer can use it to reject; **M** = costs credibility;
**L** = tidiness. "Catcher" = who finds it.

| # | problem | sev | catcher | cost to fix |
|---|---|---|---|---|
| P1 | FLOPs method undocumented; the §Metrics cross-reference does not contain the definition | **H** | any reviewer checking the efficiency claim | ½ page, no compute |
| P2 | `val/*` halting keys disagree with `halt/*`; `val/avg_recursion_steps` = 7.615 with `max_depth` 7 | **H** | a reviewer who opens the artifact | 2 lines, no compute — **diagnosed, see §2** |
| P3 | §Setup says "a single laptop GPU"; Limitations say two GPUs; Repro statement names a V100 | **M** | any careful reader | 1 sentence |
| P4 | `canonical_spec_language.json` contradicts the §6.3 table on all three numbers | **M** | a reviewer who requests the frozen spec | 3 numbers + a dated amendment note |
| P5 | OOD protocol unstated: the Pareto claim rests on 1500 blocks ≈ 384k tokens | **H** | a reviewer weighing the headline inversion | 1 sentence, or an inference run |
| P6 | No per-seed table; n=5 at the floor, so the cells *are* the evidence | **M** | a reproducibility-minded reviewer | ¼ page |
| P7 | Per-family depth measured and unused — the cleanest evidence for the central claim | **M** | nobody; this is a missed opportunity | ¼ page |
| P8 | 652J **concern 2** is recorded nowhere | **M** | the same reviewer, if reassigned | lookup only |
| P9 | The "285×" halt-gradient ratio is stated without a method | **L** | a mechanism-focused reviewer | 1 sentence |
| P10 | No learning curves, though 3-point curves exist for all 15 runs | **M** | a reviewer asking "how do you know you're underfitting" | ¼ page, no compute |
| P11 | n = 5; every headline pinned at the resolution floor 0.00794 | **H** | every reviewer | 6 more cells ≈ 40 GPU-h |
| P12 | One model scale (5.58M); sparsity benefits are known to need scale | **H** | every reviewer | a second size, days |
| P13 | E = 6 never justified or varied, and not named as an unexplored axis | **M** | any MoE reviewer | 1 sentence, or a sweep |
| P14 | The mechanism section — the stated novelty — rests on a dev-corpus grid at one seed | **H** | a reviewer who reads §6 carefully | canonical-corpus grid, days |
| P15 | 62 run directories, 15 admitted; admission rests on the exporter's logic | **L** | an artifact reviewer | already documented; mention it |

---

## 2. P2 is diagnosed — it is a naming collision, not a bug

`val/avg_recursion_steps` is `val_exp_depth_sum / val_exp_depth_n`: **expected**
depth, the ACT ponder quantity `N + R`, not the realised exit step. Confirmed
arithmetically on both arms against numbers the paper already publishes:

| arm | mean depth (paper) | halt remainder (paper) | sum | `val/avg_recursion_steps` |
|---|---|---|---|---|
| MoRE | 6.955 | 0.655 | 7.610 | **7.615** |
| MoR | 6.730 | 0.175 | 6.905 | **6.899** |

So the paper already reports both components of the "impossible" number, in the
same paragraph, and simply never says that their sum appears in the artifact
under a name that reads like realised depth. The `val/forced_exit_rate` gap
(0.9809 vs 0.9734) has the same shape: a soft/expected quantity beside a
realised one.

**This is now cheap and fully honest to fix**, and it converts the single most
dangerous artifact finding into a footnote.

---

## 3. Options, per problem

### P1 — FLOPs method

| option | what | cost | verdict |
|---|---|---|---|
| **A** | Appendix subsection: `FlopCounterMode` on a real forward pass, the `base + (d−1)·per_step` fit, the **0.1 % linearity check as a hard failure**, evaluation at *measured* mean depth not `max_depth`, and that attention/embeddings/the 8192-way head are included | ½ page | **recommended** |
| B | One paragraph in §Metrics + a pointer to `code/flops_accounting.py` | 1 para | weaker; a pointer to code is not a method |
| C | A, plus a 7-row table of measured FLOPs(d) for d = 1..7 per arm so the reader can check the fit | ¾ page | best if space allows — the sweep data already exists |

Appendices are page-exempt in **both** venues, so even ACL at 8.29 pages can take
C. Fix the dangling §Metrics cross-reference either way.

### P2 — the `val/*` halting family

| option | what | verdict |
|---|---|---|
| **A** | Keep the keys, add one line to the artifact README and a footnote in §Metrics: these are *expected* depth `N+R` and the soft exit rate; the realised quantities are `depth/*` | **recommended** — honest, loses nothing |
| B | Rename to `val/expected_depth_N_plus_R` and regenerate | cleaner long-term, touches the exporter and every downstream consumer |
| C | Delete the keys from the release | **rejected** — removing data to avoid a question is the opposite of this project's posture, and violates archive-never-delete |

### P3 — hardware contradiction

A: change §Setup to "on single consumer GPUs (see the Limitations for the
two-card split)". B: drop the hardware clause from §Setup entirely. **A** — the
reader wants the hardware scale early, and the disclosure is creditable.

### P4 — frozen spec vs the paper

| option | what | verdict |
|---|---|---|
| **A** | Amend the spec's prose to the machine-written values, with a **dated amendment note** saying the prose was wrong and `lang_calibration_weights.json` is authoritative | **recommended** |
| B | Footnote the discrepancy in the paper | rejected — invites exactly the question it answers |
| C | Re-measure the calibration point | the point is a single-seed dev-corpus proxy; re-measuring buys nothing and may produce a third number |

Note this edits a **frozen** document. Do it as a recorded amendment, never
silently.

### P5 — OOD evidence base

| option | what | cost | verdict |
|---|---|---|---|
| A | State the protocol: `pile:test`, 1500 blocks, 256 tokens, ≈384k tokens, WikiText-fitted tokenizer | 1 sentence | minimum |
| **B** | A, **plus re-score at 10–20× more Pile blocks** — inference only, all 15 checkpoints present, no training | hours of inference | **recommended** |

The Pareto-dominance inversion is the paper's one "something neither does alone"
result and currently rests on the smallest evidence base of any headline claim.
This is the single place where cheap compute buys real credibility. If the
inversion survives at 4M tokens it is a much harder result to dismiss; if it does
not, you need to know that before a reviewer does.

### P6, P7, P10 — three free appendix tables

All measured, all absent, none needs compute.

- **P6** per-seed table, 15 cells.
- **P7** per-family depth: MoR spans **6.572–6.887** across six linguistic
  families; MoRE spans **6.953–6.962**, a spread of **0.009**. This demonstrates
  the central claim on a linguistic partition and does **not** suffer the
  near-constant problem that forces the paper to spend a paragraph defending
  ρ = −0.023. Strictly stronger than the statistic it would sit beside.
- **P10** validation curves, 3 points × 15 runs. See §4 — this one has a catch.

### P11–P14 — the structural limits

| problem | mitigation | cost | realistic verdict |
|---|---|---|---|
| P11 n=5 | add seeds 47, 48 → n=7, floor drops 0.00794 → 0.00058 | ~40 GPU-h, 6 cells | only worth it if a venue rejects on the floor; the paper's phrasing is already honest |
| P12 one scale | a second model size, even at 3 seeds, labelled exploratory | days | the single most valuable missing experiment, and the paper says so |
| P13 E=6 | **name expert count as an unexplored axis in the Limitations** — it currently is not | 1 sentence | do the sentence now; the sweep only if targeting an MoE-focused venue |
| P14 mechanism on dev corpus | run the competition grid on the canonical corpus | days | the honest alternative is to keep the label and accept the weaker claim |

P13's one-sentence fix should happen regardless — the Limitations name model
size, epochs and the unrun controls, but **not** expert count, which is a
conspicuous omission in a paper about expert routing.

---

## 4. P10 is double-edged, and should still go in

Mean validation loss per epoch, all 15 admitted runs:

| arm | epoch 1 | epoch 2 | epoch 3 | final-epoch drop |
|---|---|---|---|---|
| MoE | 4.145 | 4.039 | 3.948 | +0.091 |
| MoR | 3.820 | 3.647 | 3.515 | +0.132 |
| MoRE | 3.942 | 3.813 | 3.637 | **+0.176** |

**MoRE is improving fastest at the moment the budget is cut** — 1.33× MoR's
final-epoch rate — so the arm that loses in domain is the one furthest from
convergence.

This hands a reviewer a number they would otherwise have to guess at: the
headline ordering is budget-dependent, and the dependence runs *against* the
paper's own in-domain finding. The Limitations already concede that the
behaviour at 10 epochs is unknown; the curve makes that concession auditable
instead of rhetorical.

**Include it.** The paper's entire posture is that disclosed weaknesses add
credibility, and a reviewer who derives this independently from the artifact
will treat its absence as concealment. **Do not extrapolate** to where the curves
cross — that would be inventing a number, and the curves are decelerating. The
defensible sentence is that the ordering is budget-dependent in a direction the
runs cannot resolve.

---

## 5. Strategy

### Tier 0 — before any submission. No compute. ~1½ pages.

P1, P2, P3, P5(A), P9, P13-sentence. Every one is documentation of work already
done. Together they remove **every** "this is undocumented" criticism in the
register and convert the most dangerous artifact finding into a footnote.

### Tier 1 — cheap strengthening. Inference only; all 15 checkpoints exist.

P5(B) enlarged OOD, P6 per-seed table, P7 per-family depth, P10 curves. No
training, no new runs, and P7 and P10 are already measured — only P5(B) needs
GPU time at all, and only for forward passes.

### Tier 2 — governance.

P4 spec amendment, P8 recover concern 2, P15 one line on run admission.

### Tier 3 — structural. Training compute. Venue-dependent.

P11 (n=7), P12 (second scale), P14 (canonical-corpus mechanism grid).

### The venue argument

**TMLR's review criteria are "are the claims supported by evidence" and "would
some audience be interested" — not novelty or impact.** A rigorous negative
result with disclosed limits is squarely in scope, and reviewers read appendices.
For TMLR, **Tier 0 + 1 + 2 is very likely sufficient**, and Tier 3 is not a good
use of GPU time.

**ACL is the opposite.** It weighs novelty and impact, enforces a hard 8-page
limit the paper is already over at 8.29, and its reviewers are more likely to
treat one scale and n = 5 as disqualifying regardless of how well they are
disclosed. Documentation cannot fix P11/P12 for ACL.

**Recommendation: execute Tier 0 + 1 + 2 and send to TMLR.** Treat Tier 3 as
conditional on a TMLR reviewer explicitly demanding it — which is what TMLR's
revision cycle is for, and is far cheaper than speculatively burning 40+ GPU-hours
now.

---

## 6. What makes P11–P14 *structural*

A problem is structural here when **no amount of writing changes what the
evidence can support**. P1–P10 are all cases where the work exists and the prose
does not. These four are cases where the work does not exist, and the paper's
phrasing — which is already honest — is the ceiling of what can be claimed.

### P11 — the resolution floor is a property of the design, not the data

With five seeds per arm the exact two-sided test enumerates
$\binom{10}{5} = 252$ regroupings, and the two-sided statistic is invariant under
swapping the groups, so extreme regroupings arrive in mirror pairs and the
smallest attainable $p$ is $2/252 = 0.00794$. **Every headline comparison returns
exactly that number.** If MoR beat MoRE by 10 nats instead of 0.1216, the test
would still return 0.00794. The design cannot express a stronger result.

The margin is thinner than it looks. Plain Bonferroni over the seven comparisons
gives $0.05/7 = 0.00714$, **below the floor** — a correction under which no
effect of any size could ever be declared significant. The paper escapes only
because Holm's easiest threshold at $m = 4$ is $0.0125$, a margin of $0.0046$.
That is a live dependency on the choice of correction, and the choice is
defensible but load-bearing.

| seeds per arm | regroupings | floor |
|---|---|---|
| 5 (current) | 252 | 0.00794 |
| 6 | 924 | 0.00216 |
| 7 | 3,432 | 0.00058 |

**One seed pair (n → 6, three cells) already buys a 3.7× lower floor.** That is a
much cheaper intervention than the n = 7 figure quoted in §3 and should be the
default if any seeds are added at all.

### P12 — one point cannot constrain a curve, and a data sweep is the cheaper test

The paper's negative finding is exactly the kind known to be scale-dependent,
and the usual sparse-MoE defence — more parameters at constant FLOPs — is
*unavailable* here by construction, because all three arms are pinned to one
parameter count. The paper converts the objection into a falsifiable crossover
prediction, which is the strongest available rhetorical move, but a prediction
is not a measurement: the paper has one point and is making a claim about the
shape of a curve.

**The cheaper and better-targeted experiment is a data sweep, not a second model
size.** The stated mechanism is that routing divides the *diversity* of data each
parameter block sees, which predicts the MoR−MoRE gap closes as tokens-per-expert
grows. The paper already runs subset fractions — 2.5 %, 10 %, 40 %, 100 % — for
the saturation finding, so the machinery exists. Running MoR and MoRE at three
fractions × three seeds puts three points on the curve the mechanism predicts,
using infrastructure already built, and tests the paper's *own* account rather
than a generic scale objection. A second parameter scale tests neither.

### P13 — the decisive axis already has two measured points, and the paper does not say so

$E$ is not an incidental hyperparameter. The dilution account says each expert
sees $1/E$ of the token stream, so **$E$ is the knob that controls the very
mechanism the paper proposes**, and it is the one axis never varied.

Except that it is. The paper states in the architecture caption that **setting
$E = 1$ recovers MoR**. So the MoR-versus-MoRE contrast *is* an $E$-sweep with
two points, $E = 1$ and $E = 6$, run at matched parameters and matched budget —
and the paper presents it only as an architecture comparison. Reframing it costs
nothing and turns the headline contrast into the first two points of the
mechanism's own test. Adding $E \in \{2, 3\}$ would populate the curve; the
dilution account predicts the gap to MoR shrinks monotonically as $E$ falls.

At minimum, name expert count as an unexplored axis: the Limitations currently
name model size, epochs and the unrun controls, and omit the one parameter the
paper's mechanism is about.

### P14 — the novelty rests on the weakest evidence, and contradicts §6.2

The headline rests on 15 fully-controlled cells. The *mechanism* — §6.3, the
competition between depth and specialisation, explicitly "the paper's central
claim" — rests on three single-seed points on a ~60× smaller dev corpus. The
evidential standard drops sharply exactly where the novelty is, and the paper
labels this but does not resolve it.

**There is also a direct internal tension.** §6.2 establishes, as a measured
finding, that the ponder weight calibrated on the small corpus **does not
transfer** to the full corpus — that is the whole content of the depth-saturation
result. §6.3 then uses a small-corpus grid over **that same ponder weight** to
explain full-corpus behaviour. The paper is simultaneously claiming that
small-corpus calibration of this hyperparameter does not transfer, and using
small-corpus calibration of this hyperparameter as its mechanism.

Both can be true — non-transfer of a *level* does not preclude transfer of a
*trade-off* — but the paper never says so, and a reviewer who reads §6.2 and §6.3
together will find the tension before the resolution. **This needs a sentence
even if the grid is never re-run**, and it is the strongest argument for running
the halting-weight grid on the canonical corpus.

