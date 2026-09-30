# PAPER_REVISION.md — revision agenda for resubmission

**Status.** The MoRE arithmetic paper was rejected from the NeurIPS 2026 LIGHT
workshop on 2026-09-30. Two reviews:

| reviewer | rating | confidence | thrust |
|---|---|---|---|
| Mqc5 | 4 | 3 | exposition: undefined acronyms, abstract/intro/related-work weakness |
| 652J | 5 | 4 | technical: ACT formulation, depth analysis, compute-matching, statistics, novelty |

**There is no rebuttal and no author response.** The decision is final and the
paper goes to a different venue. The reviews are therefore being used as what
they actually are — **a free, expert defect list** — and this file is the work
queue derived from it, not correspondence. Nothing here is written to persuade a
reviewer; every item is either a repair the paper needs on its own merits or a
measurement that was missing.

That reframing matters for prioritization. Items are ranked by *how much they
improve the paper*, not by how loudly a reviewer raised them:

- Items that change a **number or a claim** are mandatory (§1.2, §2.3, §2.5).
- Items that add a **missing experiment** are high value because they answer
  questions the paper itself raises and leaves open (§2.1, §2.2).
- Items that are **presentation** are cheap and get done last (§3, §4).

652J is the load-bearing review and is explicit that neither the scale nor the
null result is grounds for rejection — they state that LIGHT should not reject
the paper for its scale or for finding no gain, and that acceptance does not
require main-track scale. Read that as evidence the *finding* is publishable and
the *execution* was not yet.

This file is the single source of truth for the revision. The paper draft does
not live in this repository (no `.tex` under version control), so text written
here must be carried across by hand; keep this file authoritative and let the
draft follow it, never the reverse.

**Scope discipline.** Nothing below changes an architecture to obtain a
favourable result (CLAUDE.md §6). Two of the reviewers' technical objections are
**refuted by measurement** and the paper should state the measured fact rather
than quietly adopting the reviewer's premise; the rest are real defects and get
fixed. Keep those two categories distinct — silently conceding a point that
measurement contradicts would put a false statement in the next draft.

---

## 1. The blocking item: write the ACT forward path down

Both reviewers land here, and 652J's concern 1 is the single most important item
in either review. It has two halves, and **they resolve differently.**

### 1.1 The "counts p_t twice" reading — notation, not a defect

The submitted paper defines cumulative halt mass as `c_t = Σ_{τ<t} p_τ` and then
gives the halt condition `c_t + p_t > 1 − ε`. 652J read this as double-counting
`p_t`.

It is not. `c_t` as written sums over `τ < t` — strictly *before* step t — so
`c_t + p_t` is the cumulative mass *including* step t, which is exactly the
Graves (2016) / Universal Transformer condition. The implementation agrees:
`code/more/model.py`, the `# ---- 4. ACT halting` block, computes
`would_cross = (cum_active + halt_probs) > self.halt_threshold` where
`cum_active` is the buffer *before* this step's `p` is added.

**Revision action.** The ambiguity is real and cost us a reviewer's confidence,
so fix the notation rather than argue it: either subscript the cumulative as
`c_{t-1}` throughout, or define `c_t` inclusively and write the condition as
`c_{t-1} + p_t`. Add one sentence naming the formulation as standard ACT and
citing Graves 2016 and Dehghani et al., so a reader knows no new halting
equation is being proposed (`plan.md` §5.2 forbids inventing one).

### 1.2 The undefined output aggregation — a genuine and serious omission

652J is correct that the paper never defines the probability-weighted output
aggregation, and their inference from that gap is reasonable: if the task loss
sees only a discrete halted state, the halt head would be trained mainly to stop
early and a near-constant policy would be expected.

**The aggregation exists in the code and must be in the paper.** It is
`code/more/model.py`, section `# ---- 5. Accumulate the ACT-weighted output`:

```
accumulator[active_positions] += updated * step_weight.unsqueeze(-1)
```

with per-token step weights

```
w_t = p_t                while the token continues
w_t = R = 1 − c_{t−1}    at the step where it halts
```

which sum to **exactly 1** over the steps a token takes. The output is therefore
a convex combination of the per-step states, the regression/LM head reads that
combination, and the task loss consequently has a differentiable path to every
`p_t` and hence to the halt head. The boolean `> threshold` gates *dispatch
only*, which `updated_rules.md` §2.2 permits precisely because the objective
retains this path.

**Revision action.** Add to §3.3, as displayed equations: the per-step halt
probability, the cumulative mass with unambiguous subscripts, the halt
condition, the remainder, the step weights, the statement that they sum to 1,
and the accumulator. Then state in one sentence that this is the only gradient
path required to reach the halt parameters, and that the ponder term is an
additional pressure rather than the sole path — which is what the submitted
version implied and what triggered the concern.

### 1.3 The measurement that settles the mechanism

652J's *prediction* from the wrong mechanism — a near-constant policy — is
**empirically correct**, which is why "the aggregation exists, so the premise is
wrong" would be an inadequate thing to put in the paper. The live question is
not whether the task gradient exists but whether it **dominates**.

`code/diag_halt_gradient.py` (T-LX.14) backpropagates each objective term
separately into `blocks[*].expert_halt_heads.*` on a converged checkpoint.
Language MoRE, 30 epochs, wikitext-2, `runs/langB_MoRE_seed44__68d032ba`,
12 tensors / 1,542 params, eval mode, 4 validation batches:

| objective term | ‖∂term/∂halt-head‖₂ |
|---|---|
| task (LM cross-entropy) | **7.786e-02** ± 9.3e-03 |
| ponder (weight 0.001) | **2.734e-04** ± 1.3e-05 |
| routing balance | exact structural 0 |
| step routing | exact structural 0 |

**task / ponder = 285×.** The halt heads are trained overwhelmingly by the task
loss. This refutes *both* available explanations for constant depth: the
gradient is not missing, and the ponder cost is not suppressing depth — it is
285× too weak to do so.

The two exact zeros are measurements, not sentinels (CLAUDE.md §4): the balance
and step-routing terms are functions of `router_logits` alone and no autograd
edge runs from them to a halt head, so the derivative is structurally zero
rather than numerically small. The script labels them as such and raises rather
than printing `0.0` if the parameter-name match comes back empty, because an
empty match and a genuinely dead gradient would otherwise be indistinguishable.

### 1.4 The positive claim this licenses

Depth rises **because the task loss pulls it up**, not because nothing holds it
down. Over the 30-epoch language trend, mean depth and validation loss improve
together:

| epoch | 1 | 3 | 5 | 8 | 10 | 12 | 15 | 20 | 30 |
|---|---|---|---|---|---|---|---|---|---|
| mean depth | 1.96 | 3.20 | 4.53 | 5.70 | 6.35 | 6.49 | 6.66 | 6.77 | **6.82** |
| val loss | 5.68 | 5.08 | 4.82 | 4.68 | **4.63** | 4.63 | 4.65 | 4.72 | 4.81 |

with 88.9% of tokens force-exiting at step 7 by the end, per-family mean depth
spanning only 6.80–6.94 across six linguistic families, and
`depth/by_document/between_share` = 0.0008.

**The claim to make:** *at this scale, no token has a task-optimal early exit,
so the halting policy has nothing to be adaptive about.* This is Outcome C
(CLAUDE.md §8) with a measured mechanism attached, which is a materially
stronger contribution than the same null with no mechanism — and it is the
honest form, because it concedes the reviewer's symptom while correcting their
cause. Do **not** soften it toward "MoRE shows partial adaptivity."

**Do not** respond to this by raising the ponder weight until the depth
histogram spreads out. That is tuning toward a conclusion (CLAUDE.md §6). The
legitimate experiment is a **labelled ponder-weight ablation** that reports the
depth–quality tradeoff curve honestly, including the case where forcing depth
below the cap simply costs loss.

---

## 2. Reviewer 652J, remaining concerns

### 2.1 Concern 2 — the depth analysis needs controls

Conceded. Three controls, all cheap, none yet run:

1. **Shuffled-depth control.** Re-evaluate with the same depth histogram but
   depths permuted across tokens. If quality is unchanged, the *allocation* is
   not doing work even when the *distribution* looks adaptive. The permutation
   machinery already exists — `depth/spearman_vs_*_null_*` in `metrics.json`
   builds null bands this way — so this is an extension, not new infrastructure.
2. **Separately trained fixed depths 1..7** (652J's question 3). **Config-only:**
   `fixed_depth` is already plumbed (`config_moe_50e.json:10`,
   `config_mor_50e.json:10`, `config_depth5_50e.json:10`, and honoured at
   `model.py:952`), so this needs no model code. It is the direct test of
   whether adaptive depth beats the best constant depth, which is the question
   the paper should have answered and did not.
3. **Depth–input mutual information.** Report I(depth; token identity) and
   I(depth; log-frequency) alongside the existing Spearman correlations. The
   correlations are already known to exceed their permutation nulls while being
   tiny (language: ρ = 0.155 vs null band ±0.0036), and an MI figure states the
   magnitude of the dependence rather than only its sign.

### 2.2 Concern 3 — parameter-matched is not compute-matched

**Conceded, and this was independently the top finding of the internal audit.**
The three arms are parameter-matched by construction (MoE 6×4, MoR 1×24,
MoRE 6×4 FFN units — 24 each) and that had been allowed to stand in for
compute-matching. It cannot: MoR runs one 24×-wide FFN fully active, MoRE
activates one of six 4× experts, and both then multiply by mean depth.

Before T-LX.15 the repository had **no FLOPs accounting anywhere** — only two
comments marking the hole (`engine.py:1458`, `metrics.py:1647`). It now has
`code/flops_accounting.py`, which measures a real forward with
`FlopCounterMode` from each arm's own `resolved_config.json` (so attention, the
positional table and the 8192-way LM head are counted — the terms every hand
derivation dropped) and verifies the `base + (d−1)·per_step` depth fit against a
third measured point rather than assuming linearity.

Language arms, each at its own measured `depth/mean`:

| arm | E | ffn | d_mean | GFLOPs @1 | GF/extra step | GFLOPs @ d_mean | items/s |
|---|---|---|---|---|---|---|---|
| MoE | 6 | 4 | 1.000 | 10.751 | N/A | 10.751 | 718.8 |
| MoR | 1 | 24 | 6.899 | 21.483 | 12.887 | **97.506** | 173.8 |
| MoRE | 6 | 4 | 6.947 | 10.751 | 2.155 | **23.565** | 168.5 |

MoR/MoRE **4.14×**, MoR/MoE **9.07×**, MoRE/MoE **2.19×**.

**What the revised paper must report, in three parts — two of which cost
nothing.**

- **iso-parameter** holds by construction, already reported.
- **iso-latency already holds, and the paper failed to notice.** MoR 173.8 vs
  MoRE 168.5 items/s on the same card is a 3% gap. The published matrix
  therefore *already contains* an iso-latency control; the failure was in the
  reporting, not in the experiments. Report it as a control.
- **iso-FLOP does not hold**, and is now quantified rather than hand-waved.
  Critically, iso-FLOP and iso-parameter are **mutually exclusive in this
  architecture family**: matching MoR's active width to MoRE's means
  `ffn_mult 24 → 4` at `num_experts = 1`, which drops MoR's FFN parameters to a
  sixth. An iso-FLOP MoR is therefore a *second* control, never a replacement
  for the parameter-matched one. State the impossibility explicitly — it is a
  structural property of the comparison and belongs in the methods section.

**The trap that must travel with every one of these numbers.** The 4.14× FLOP
advantage buys **nothing** in wall-clock; MoRE is 3% *slower*. Top-1 dispatch
spends the entire theoretical saving on gather/scatter and never reaches a fused
sparse kernel. FLOPs and latency point in **opposite directions** here, so every
claim must name the axis it rests on, and no FLOP ratio may be phrased as a
speedup. `flops_accounting.py` prints this sentence under every table for that
reason.

**Known gap.** `flops_accounting.py` currently profiles the *language* arms
only. The arithmetic paper needs the same table, which means giving the module
an arithmetic mode (it must build `MoREDataset` shapes rather than the packed
LM block). 652J reports the arithmetic paper's own figure as ~5.8×; do not reuse
that, and do not reuse the ~3.7× / ~25× multipliers quoted in session chat
before this module existed — those were FFN-only hand derivations and are wrong.
Measure it.

### 2.3 Concern 4 — the fixed-`T = 7` claim is overstated, and the split is wrong

Both conceded, and the second is the more serious.

- **Overstatement.** Fixed `T = 7` improves mean validation loss at `p = 0.0476`,
  which does **not** clear the paper's own Bonferroni threshold of 0.0083. The
  abstract states it too strongly. Rewrite it as a non-significant trend and
  move the number to the results table with its threshold beside it.
- **Validation used as test.** Headline results are reported on validation
  despite a declared test split, and the paper does not say whether the same
  held-out data both selected and evaluated the constant `c = 2`. If it did,
  that is a model-selection leak and it is the only objection in either review
  that can invalidate a published number rather than merely under-describe it.

  **Verified 2026-09-30 — the test split is clean, and the fix needs no
  retraining.** `data/test.jsonl` holds 5,250 records and is genuinely
  untouched, on two independent grounds:

  1. **No code path could have read it.** `engine.py` opens
     `dc["train_path"]`/`dc["jsonl_path"]` (engine.py:179) and `dc["val_path"]`
     (engine.py:201), and nothing else. There is no test-split loader in the
     arithmetic path at all — `SPLITS = ("train","val","test")` appears only at
     `lang_data.py:68`. A run could not have touched it even by accident.
  2. **Byte-level disjointness.** Whole-line SHA-1 over all three files:
     train 59,500 / val 5,250 / test 5,250 records, every record unique within
     its split, and **train∩test = 0, val∩test = 0, train∩val = 0**.

  So the repair is a forward pass, not a training run. `code/eval_test_split.py`
  (T-LX.16) does it: it rebuilds the model from the run's own
  `resolved_config.json`, loads `checkpoint.pt`, and scores `test.jsonl` with
  the *identical* reduction the engine uses for validation (`F.mse_loss` on the
  regression head, sample-weighted exactly as engine.py:999-1002), so
  `test/task_loss` is directly comparable to the published `val/task_loss`
  instead of being a differently-reduced near-miss. It also reports test-split
  routing accuracy, mean depth and exit histogram, since the depth story and the
  specialization story both have to survive the split change too.

  It re-verifies disjointness on every invocation, records the test file's
  SHA-256 in its sidecar, and cross-checks routing accuracy against the
  confusion diagonal (CLAUDE.md §4) — raising rather than reporting the more
  flattering of the two.

  **Blocker, and it is not compute.** The 15 canonical `phaseB_*` cells have
  **no `checkpoint.pt` in this worktree**: `.gitignore:21` excludes
  `runs/**/checkpoint.pt`, so they were never committed. They were produced on
  Ayan's machine and the checkpoints should still be there. **Action: fetch the
  15 checkpoints from Ayan's machine**, then run
  `python eval_test_split.py --group canonical_phase_b`. Do *not* retrain to
  recover them — a retrained model is a different model, and the paper's numbers
  describe the originals (CLAUDE.md §6: preserve the original run).

  Validated by execution on the one arithmetic checkpoint that *does* exist in
  this worktree (`t67_provenance_check_seed44__6b711a59`, an exploratory smoke
  run, not canonical): test 0.069772 vs published val 0.073807, delta +0.004035
  — i.e. test came out slightly *better* than validation. One exploratory cell
  proves only that the harness runs; it is not evidence about the canonical
  matrix.

- **Carry the lesson to the language matrix.** The language runs report
  `best_val_loss`, selected on validation. Before the language paper makes any
  headline claim, confirm a test split exists, is untouched, and is what the
  table reports.

### 2.4 Concern 5 — novelty must narrow around LoopMoE

Conceded. LoopMoE (June 2026) predates the submission and compares looped MoE
against vanilla MoE at 3B and 9B under matched total parameters, matched
per-token FLOPs and matched active ratios. 652J is precise about the damage: it
does **not** study the route-once ACT design, so a distinct diagnostic claim
survives, but it **does invalidate the broad statement that matched-budget
composition had not been isolated.**

**Action.** Add LoopMoE to Related Work with that framing. Delete the
matched-budget-novelty sentence and replace the contribution claim with the
narrower one that actually survives: a *route-once-then-recurse* design with ACT
halting, with per-token routing persistence isolated as a controlled variable
(T-LX.12) and the halt-gradient attribution measured (T-LX.14). Note that LoopMoE
operating at 3B/9B while this work is at 3.2M is a scale gap to disclose, not a
defence.

### 2.5 Concern 6 — the permutation-test resolution is wrong, and it is wrong for us too

**Conceded; the reviewer's arithmetic is right and I verified it against our own
data.** For two groups of five, a two-sided exhaustive permutation test on the
absolute mean difference has minimum attainable

```
p_min = 2 / C(10,5) = 2 / 252 = 0.00794
```

not 0.00397 (= 1/252, the one-sided count). The smallest value the paper reports,
0.0079, is consistent with the correct floor — so the *reported* p-values are
fine and the stated resolution is the error.

**This is not a cosmetic fix, because we live at that floor.** Running the same
test on the five-seed language matrix:

| comparison | |Δ mean| | p | count |
|---|---|---|---|
| MoRE vs MoE | 0.4499 | **0.00794** | 2/252 |
| MoR vs MoE | 0.4327 | **0.00794** | 2/252 |
| MoRE vs MoR | 0.0172 | 0.40476 | 102/252 |

Both significant comparisons sit **exactly at the resolution floor**, and clear a
Bonferroni threshold of 0.05/6 = 0.00833 by 0.0004. One additional test in the
family breaks them. And MoRE vs MoR at p = 0.405 is not resolvable at all: the
two arms are statistically indistinguishable on quality, which is precisely why
the MoRE claim must be a *compute* claim (§2.2) and not a quality claim.

**Actions.** (a) Correct the resolution statement to 0.00794 and say whether the
test is one- or two-sided. (b) Report the attainable floor next to every p-value
so a reader can see when a result is resolution-limited. (c) Decide the
multiplicity family *before* adding tests, since the margin is 0.0004.

**The 7-seed question, and what it actually buys.** No reviewer asked for more
seeds. 652J only pointed out that the *stated* resolution was wrong. Seven seeds
is a consequence of that correction, and the reason is not "a better p-value":

| | 5 seeds/arm | 7 seeds/arm |
|---|---|---|
| partitions, C(2n,n) | 252 | 3,432 |
| min attainable two-sided p | 0.00794 | **0.00058** |
| Bonferroni α at 6 tests | 0.00833 | 0.00833 |
| margin | **0.0004** | 0.0078 |
| max tests before the floor exceeds α | **6** | ~85 |

**The binding constraint is the size of the test family, not the effect size.**
At 5 seeds the floor is 0.00794 and Bonferroni at 7 tests is 0.05/7 = 0.00714.
Cross that line and **no result can be significant at any effect size** — the
smallest p the design can emit is larger than the threshold it must clear. The
paper is currently at 6 tests with a 0.0004 margin, i.e. exactly one test away
from a mathematically unreachable bar.

And the revision *adds* tests. The fixed-depth sweep (§2.1, 652J's own question)
is up to 7 new comparisons on its own; the shuffled-depth control and an
iso-FLOP MoR arm are more. **Running the controls the paper needs would, at 5
seeds, destroy the significance of the main results it already has.** That is
the whole argument for 7.

What 7 seeds does **not** buy: it will not rescue MoRE vs MoR. That is
diff 0.0172 against a pooled sd near 0.03 — a genuine null, not a resolution
artifact. More seeds would tighten the interval around ~zero, not move it. Say
so explicitly rather than letting a reader infer that more seeds were a fishing
expedition.

**Cost.** Cheap on arithmetic and essentially free in scheduling terms: the
canonical arithmetic cells ran at 2,216 items/s over 59,500 records × 50 epochs
≈ **22 min per run**, so 2 extra seeds × 3 arms ≈ **2.2 h**, no rental. Language
is the expensive side, and there the extension should be folded into the MoRE
re-run that is already mandatory (§6) rather than scheduled as separate work.

**Integrity precondition.** Seeds 42–46 are the frozen set in
`canonical_spec.json` / `canonical_spec_language.json` and CLAUDE.md §5.
Extending to 42–48 is a **spec amendment that must be recorded before the new
runs are executed**, with the amendment timestamped in `changelog.md`. Amending
after seeing results is seed-shopping regardless of intent; amending because the
planned control experiments require the resolution is legitimate, and that
reason must be the one written down.

---

## 3. Reviewer Mqc5 — exposition

All conceded; none require an experiment.

- **Acronyms undefined before first use**, in both the abstract and the main
  paper. Sweep for MoE, MoR, MoRE, ACT, AMI, FFN, MI at minimum.
- **Abstract is too implementation-heavy and too light on the research
  question.** Rewrite around the question — *does routing specialization compose
  with adaptive depth under a matched budget?* — and state the answer, including
  the null, in the abstract.
- **Introduction too brief for the claims it makes.** Expand to motivate the
  composition question and to state what a negative result would mean.
- **Related Work underdeveloped.** Needs ACT/Universal Transformer lineage,
  Switch/top-1 routing, Mixture-of-Recursions, and LoopMoE (§2.4).

---

## 4. Mechanical repairs

| item | detail |
|---|---|
| `"VERIFY FIRST"` in the reference list | The Mixture-of-Recursions citation still contains the literal placeholder. Resolve the real reference and remove it. |
| Reproducibility checklist contradiction | The checklist makes conflicting statements about whether code and data are released, and no anonymous artifact is linked. Reconcile, and link one. |
| Undefined constructs | The PDF does not define program-token communication, pooling to a scalar, target normalization, or the ACT output aggregation (§1.2). Define all four. |
| Invalid hardware sweep | Already disclosed in the submission; keep the disclosure and point at the T-LX.13 hardware-provenance fields as the fix for future runs. |

---

## 5. Strengths to preserve

Do not lose these in the rewrite — both reviewers credited them explicitly, and
they are the reason 652J rates the work a 5:

- MoE and MoRE share parameter count, input features, data split and training
  engine — one engine, one `architecture` switch.
- Five seeds with effect sizes, exact randomization tests and multiplicity
  correction.
- Routing analysis uses Hungarian matching, AMI, purity and load balance —
  permutation-invariant, as CLAUDE.md §4 requires.
- The paper *discloses* low R², external family supervision, incomplete
  ablations, an invalid hardware sweep, and the absence of optimized sparse
  kernels. That candour is an asset; increase it rather than trimming it.

---

## 6. What still needs a GPU, and what does not

**No GPU (do these first):** §1 ACT equations; §1.3 gradient-attribution table
(already measured); §2.2 FLOPs table for language (already measured) plus the
arithmetic mode for `flops_accounting.py`; §2.5 statistical corrections;
§2.4 Related Work; §3 exposition; §4 mechanical repairs.

**No GPU, but blocked on a file transfer:** §2.3's test-split evaluation. The
harness exists and is validated (`code/eval_test_split.py`); the test split is
provably untouched; the only missing input is the 15 `phaseB_*/checkpoint.pt`
files, which are gitignored and live on Ayan's machine. Fetching them costs a
copy. Retraining to recover them would cost ~5.6 h **and produce different
models than the paper describes** — so fetch, do not retrain.

**Needs a GPU:** the three §2.1 depth controls (the fixed-depth arm is
config-only but still a training run); the five MoRE-per-token language cells,
which are required regardless because the published "MoRE" cells re-route every
recursion step and are therefore not MoRE as CLAUDE.md §1 defines it; and the
seed extension (§2.5) — ~2.2 h for arithmetic on local hardware, and folded
into the mandatory MoRE re-run on the language side.

**Predicted in advance, so the rental is not buying it:** the MoRE-per-token
re-run will land near the depth cap (§1.4). It buys architectural correctness,
not a depth finding. Recording that prediction here *before* the runs so it
cannot be retrofitted afterwards.

---

## 7. Venue

The paper goes to a different venue; there is no author response to write. Two
consequences for how the revision is written:

1. **Nothing may assume the next reviewer has seen these reviews.** Every fix
   has to stand as ordinary good practice in the text itself — the ACT
   equations, the FLOPs table, the corrected resolution and the test-split
   numbers all belong in methods and results, not in a response letter.
2. **The extra time is an asset, not slack.** A resubmission cycle is long
   enough to land the §2.1 controls and the corrected MoRE arm, which converts
   the paper from "null result with an unexplained constant-depth policy" into
   "null result with a measured mechanism and the controls that isolate it."
   That is the version worth submitting.
