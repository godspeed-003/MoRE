# PAPER_GUIDE.md — onboarding for writing the TMLR language paper

**Who this is for.** A fresh agent session (or person) on **Ayan's machine** that has
never seen this repository and has to produce the updated research paper. It assumes no
prior context. Read this file top to bottom before opening anything else; it tells you
which of the ~20 other Markdown files actually bear on the paper and which are traps.

**What this file is not.** It is not the authority on the architecture or the rules. The
authority order is unchanged and is stated in [CLAUDE.md](CLAUDE.md):
[plan.md](plan.md) → [updated_rules.md](updated_rules.md) →
[updated_objective.md](updated_objective.md) → `CLAUDE.md`. This file is a **map plus a
claim ledger**: where every number lives, which claims the evidence supports, and which
sentences are forbidden. Where it disagrees with those four documents, they win and this
file is the bug.

---

## 0. The three things that will go wrong if you skip them

1. **Never hand-copy a number into the paper.** Every figure in the paper comes from
   `results/language/results_tables.md` or `results/language/results.json`, both generated
   by `code/export_results.py`. If a number you need is not in there, the answer is to
   make the exporter emit it, not to read it out of a run directory. `CLAUDE.md` §5 makes
   this a hard rule, and the reason is that a hand-copied number cannot be regenerated or
   audited. If you catch yourself opening `runs/*/metrics.json` to get a paper number,
   stop.
2. **The arithmetic paper is dead. Do not quote a single arithmetic number.** The
   arithmetic study was withdrawn because the dataset choice was flawed from the start.
   `results/results_tables.md` (no `language/`) is the arithmetic matrix and is **mean
   squared error**; the language matrix is **per-token cross-entropy in nats**. They may
   never appear in one table (`CLAUDE.md` §6) and mixing them is the single most damaging
   error available in this repo. Many `.md` files in the root still describe the
   arithmetic study as live — see §8 for the list.
3. **This is a negative result and it must be framed as a claim, not an apology.** The
   finding is *"adaptive depth and sparse routing do not compose at fixed budget, and here
   is the mechanism"* — not *"we tried MoRE and it did not work"*. `CLAUDE.md` §8 requires
   Outcome C be reported honestly; it does not require it be reported timidly. See §5.

---

## 1. Machine setup

**You are on Ayan's machine (RTX 4060 8 GB, Windows user `Hp`).** Full record in
[ENVIRONMENT.md](ENVIRONMENT.md).

- **Interpreter:** `C:\Users\Hp\anaconda3\envs\more_env\python.exe` (torch 2.5.1 + CUDA).
  That machine's Anaconda **base** and `C:\Python314` have no torch, so the env path is
  required, not optional.
- Prefix every command with `PYTHONIOENCODING=utf-8`. Without it the exporter and the
  test suites raise `UnicodeEncodeError` under cp1252 on the arrow and ± characters.
- Shell is Git Bash; the working directory persists between calls, so **use absolute
  paths**.
- A traceback path under `C:\Users\vedan\` came from Vedant's 3050 machine, not yours. Do
  not "fix" one into the other.

**You do not need a GPU for the paper.** Every number the paper reports already exists in
`results/language/`. The only GPU-dependent items are listed in §9 as unrun, and the
budget for them is spent — write the paper without them.

Two commands are worth running once before you write, to prove the tree is sane:

```bash
PYTHONIOENCODING=utf-8 C:/Users/Hp/anaconda3/envs/more_env/python.exe code/export_results.py --task language --check
```

Expect `admitted: 15   refused: 299` and a line saying consistency passed. A refusal is
the admission filter working, not an error.

```bash
PYTHONIOENCODING=utf-8 C:/Users/Hp/anaconda3/envs/more_env/python.exe code/run_correctness_suite.py
```

Expect `TOTAL 357 357 0 0`. If the pass column does not equal the checks column, or either
the fail or skip column is non-zero, something is broken and the paper numbers are not
trustworthy — stop and fix it before writing (`CLAUDE.md` §7).

> This line read `356 356` until T-LA.1 (2026-10-04); one check was added after this file
> was written. Read the **fail and skip columns**, not the absolute total, which drifts
> upward as gates are added.

---

## 2. Repo map — only these files matter for the paper

### Read these

| file | what you get from it |
|---|---|
| [CLAUDE.md](CLAUDE.md) | The operating contract. §1 terminology, §4 metric rules, §6 integrity, §8 outcome honesty. **§4 and §6 are the ones that constrain paper sentences.** |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Codebase orientation: the one-engine/three-modes design, the data contract, why there is no `moe/` or `mor/` folder. Read first for orientation. |
| `results/language/results_tables.md` | **The numbers. All of them.** Generated, never edited. |
| `results/language/results.json` | The same numbers machine-readable, plus non-scalar metrics (confusion matrices, Hungarian assignment vectors) that are not table cells. |
| `code/confirmatory_tests.json` | The **pre-registered** test families, the correction method, and an explicit honest-disclosure block about which comparisons were specified after their data existed. Methods section depends on this. |
| `code/canonical_spec_language.json` | The frozen definition of "canonical" for the language matrix, plus a `frozen_by` note per field recording whether each value was **measured** or merely **carried**. Hyperparameter table in the paper comes from here. |
| [changelog.md](changelog.md) | Newest-last. The last four entries (T-LX.16 through T-LX.20) are the current state. Each has a *Failure tracing* line naming the code location to open. |
| [PAPER_REVISION.md](PAPER_REVISION.md) | The revision agenda. **Partly arithmetic-era** — see §8. Its §1 (ACT equations) and §3 (exposition) are still fully valid and are real work items. |
| `code/seed_stats.py` | Module docstring is the authority on the resolution floor and the multiple-comparisons layer. Quote its reasoning in the methods section. |
| `code/lang_calibration_weights.json` | The hyperparameter calibration grid. Load-bearing for §6's mechanism argument. |

### Do not use these

- Everything in `archive/` is invalidated history, **except** `archive/pre_finalization/more_l7_1_per_step/`,
  which holds the five retired per-step MoRE cells. Those are a **labelled ablation the
  paper reports** (§7), not garbage.
- `archive/pre_finalization/sweep_results/rules.md` and `.../objective.md` are superseded.
  The old primary criterion `expert_entropy > baseline` is **void** — high entropy is a
  load-balance diagnostic only, never a quality score (§6, and the router-entropy note).
- `results/results_tables.md`, `results/results.json`, `moe_batch1_results.md`,
  `results_exp.md` — arithmetic. Not for this paper.
- `README.md` describes the arithmetic POC. Historically correct, left alone on purpose
  (`CLAUDE.md` §6), irrelevant here.

---

## 3. What the study is

Three architectures, one training engine, one `--architecture moe | mor | more` switch.
`CLAUDE.md` §1 defines them and the paper must not blur them:

- **MoE** — six independent expert FFNs, learned **Top-1 sparse** token routing. Answers
  *which computation module?*
- **MoR** — one shared block reused across recursion steps with adaptive ACT halting.
  Answers *how much computation?*
- **MoRE** — a token is routed **once** to a specialized expert, and that expert's weights
  are recursively reused for an adaptive number of steps. Answers both.

The canonical language matrix is `canonical_lang_b`: **15 cells = 3 arms × seeds
42,43,44,45,46**, wikitext-103, BPE vocab 8192, seq_len 256, d_model 256, 1 block,
max_depth 7, 4 attention heads, batch_size 48, 3 epochs, lr 1e-3, weight decay 1e-4,
grad_clip 1.0.

- `dataset_version = lang-wikitext-103-bpe8192-len256-800d6154`
- `train_split_version = wikitext-author-splits-v1` (the corpus's **own** author-provided
  validation split — no random carve-out of train, which is what makes the leakage audit
  meaningful)
- **Parameters are matched to 0.069%:** MoE and MoRE both 5,584,908; MoR 5,581,063. The
  3,845-parameter difference is the router MoR does not have. State this number in the
  paper — matched-parameter comparison is one of the study's strongest claims.
- **Primary metric: `val/task_loss`**, final-epoch per-token cross-entropy in nats.
  `best == last` on all 15 cells because `log_interval = 1` and the curves decrease
  monotonically, so **no model selection occurred**. Say so; it pre-empts a reviewer
  question.
- **Baseline floor: 4.984947 nats** — a backoff bigram on the same corpus (7.1918 bits,
  perplexity 146.2). A model that does not beat it has learned nothing a two-column count
  table could not. All three arms beat it.

---

## 4. The results, and where each number lives

Everything in this section is in `results/language/results_tables.md`. It is reproduced
here **only so you can tell at a glance whether a draft sentence is consistent with the
data** — the paper must cite the generated table, not this file.

### Headline (`val/task_loss`, nats/token, lower is better, n = 5)

| arm | val/task_loss | perplexity | margin vs bigram floor | params |
|---|---|---|---|---|
| MoE | 3.9483 ± 0.0242 | 51.85 | +1.0366 | 5,584,908 |
| **MoR** | **3.5154 ± 0.0340** | **33.63** | **+1.4695** | 5,581,063 |
| MoRE (route-once, canonical) | 3.6370 ± 0.0134 | 37.98 | +1.3480 | 5,584,908 |

> **Both MoE and MoR were retrained on Ayan's 4060** to recover the checkpoints lost with the
> rented V100 (T-LA.1 2026-10-04, T-LA.2 2026-10-06). MoE 3.9529 ± 0.0396 → 3.9483 ± 0.0242;
> MoR 3.5202 ± 0.0406 → 3.5154 ± 0.0340. Both V100 sets are in
> `archive/pre_finalization/lang_v100_no_checkpoints/`; the 4060 cells are the admitted ones.
> **MoRE is still the V100 arm**, so the matrix is two cards — justified, not merely caveated,
> by the next paragraph.
>
> **The reproduction is the strongest hardware-independence evidence the study has, and it is
> a two-arm result.** MoE moved −0.117 σ and MoR −0.119 σ; in absolute terms −0.0046 and
> −0.0048 nats. Agreement to the third decimal across arms sharing no parameters is not
> five-seed sampling noise — it is a small **common-mode** offset of the `sm_70 → sm_89` move
> (kernel selection and reduction order; the seed-blind config check passes). A common-mode
> shift **cancels in the between-arm gaps that carry every verdict**: `MoE − MoR` went
> +0.432711 → +0.432896, a move of 0.000185 nats. Say it that way.
>
> **Do not claim per-seed reproduction.** Same-seed drift is +0.055, −0.068, −0.021, −0.009,
> +0.019 for MoR — an order of magnitude above the shift in the mean. The seed fixes
> initialization, data order and dropout, not non-deterministic CUDA reduction order. The
> reproducible quantity is the **arm mean**, not the cell.

Perplexity is `exp(mean nats)` and is **not** the mean of the five per-seed perplexities —
`exp` is convex. The verdict is taken on nats either way. Do not report a mean perplexity
with a ± derived from the nats std.

### Pairwise tests

| pair | gap | Cohen d | p (exact) | floor | Holm threshold | verdict |
|---|---|---|---|---|---|---|
| MoRE − MoE | −0.311346 | −15.92 | 0.00794 | 0.00794 | 0.01250 | significant, **at the resolution floor** |
| MoRE − MoR | +0.121551 | +4.71 | 0.00794 | 0.00794 | 0.01667 | significant, **at the resolution floor** |
| MoE − MoR | +0.432896 | +14.68 | 0.00794 | 0.00794 | 0.02500 | significant, **at the resolution floor** |

> Updated at T-LA.2. **Every effect size in this table has now moved twice while the gaps
> moved by ≤ 0.005 nats**: across the two retrains `MoRE − MoE` went d −10.68 → −15.92,
> `MoRE − MoR` +3.87 → +4.71, `MoE − MoR` +10.79 → +14.68 — driven entirely by both retrained
> arms having tighter stds, not by any gap widening. This is the `CLAUDE.md` §4 trap and it has
> now fired three times in three exports: **a large d is not a large difference.** Quote the
> gap in nats first and the d second, and never carry a d forward by hand.

### Depth and halting

| arm | depth mean | depth std | early exit | forced exit | leftover halt mass | ρ(depth, token loss) |
|---|---|---|---|---|---|---|
| MoR | 6.730 ± 0.241 | 0.647 | 18.8% ± 15.1% | 81.2% | 0.175 | **+0.2985 ± 0.0654** |
| MoRE | 6.955 ± 0.004 | 0.427 | 2.66% ± 0.75% | 97.3% | 0.655 | −0.0230 ± 0.0115 |
| *MoRE per-step (retired)* | *6.965 ± 0.012* | *0.378* | *1.94% ± 0.30%* | *98.1%* | *0.610* | *−0.0154 ± 0.0186* |

Per-seed, which matters for how strongly you may state it:

- **MoR ρ is positive on 5/5 seeds** (+0.3573, +0.2736, +0.3600, +0.2023, +0.2991) — retrained
  values, T-LA.2. The V100 arm gave +0.2577, +0.3512, +0.3508, +0.2275, +0.2896, i.e. the
  **mean reproduced to +0.2953 → +0.2985 and the sign held 5/5 on both cards.** This is the
  paper's main positive claim and it is now a cross-hardware replication, not a single run.
- **MoRE ρ is negative on 5/5 seeds** (−0.0056, −0.0303, −0.0352, −0.0183, −0.0255).
- **MoR's early-exit rate ranges 7%–76% across seeds** (8.26, 75.75, 33.86, 6.95, 8.79).
  The two seeds that exit most are the two with the **worst** loss (3.5693, 3.5572) and
  the three that exit least are the best (3.4768, 3.4965, 3.5014). So MoR's
  depth–difficulty *alignment* is robust while the *amount* of compute it skips is
  seed-unstable and buys loss. That is an honest adaptive-computation result in its own
  right and belongs in the paper.

### Routing and specialization

| arm | AMI | AMI control mean | AMI delta z | Hungarian | Hungarian z | purity | collapsed experts | max load |
|---|---|---|---|---|---|---|---|---|
| MoE | 0.1615 ± 0.0231 | — | 13.60 ± 5.99 | 0.3944 ± 0.0185 | 9.96 | 0.4495 | none | 0.686 |
| MoRE | 0.1803 ± 0.0246 | ≈0.030 | 13.43 ± 3.87 | 0.3592 ± 0.0271 | 5.44 | 0.4712 | none | 0.448 |

Every one of these is measured against a **per-run permutation control** (shuffled POS
labels, 10 draws), which is why the `delta_z` columns exist. Report the z, not just the
raw value — `CLAUDE.md` §4 requires permutation-invariant metrics, and an uncontrolled AMI
is the exact thing reviewers of MoE papers distrust.

**Important caveat you must carry.** The POS partition is a *linguistic hypothesis about a
useful expert split*, not a functional ground truth. The run artifacts say this themselves
in `routing_agreement_caption`. A router can be better for next-token prediction and score
low here. Never call it "routing accuracy" without that qualifier; the metric key is
`val/routing_agreement_with_pos` for exactly that reason.

**A labelling defect to be aware of, not to fix in the artifacts.** The per-family routing
keys on language runs carry **arithmetic** family names (`E1_ADD_SUB`, `E2_MULT_DIV`, …)
even though the underlying partition is the POS one (`L1_FUNCTION` … `L6_NUM_SUBWORD`,
which do appear correctly under `depth/mean_by_family`). The numbers are right; the key
names are stale. **Relabel in the paper's table; do not rewrite the run artifacts.**

### Router entropy — read this before writing anything about specialization

| arm | per-token router entropy (H/log E) | expert load entropy | routing balance loss |
|---|---|---|---|
| MoE | 0.9024 ± 0.0342 | 0.6515 ± 0.0852 | +0.1235 ± 0.2556 |
| MoRE | **0.9999 ± 0.0000** | 0.9582 ± 0.0061 | −0.7822 ± 0.0012 |

MoRE's router softmax sits at 99.99% of maximum PER-TOKEN entropy — near-uniform over six
experts, so Top-1 argmax selects among six near-tied logits. **But do not write that this is
a balance-loss pathology** — T-LX.19 did, and T-LX.21 withdrew it. Two distinct quantities
are at play and the table above keeps them separate: per-token router-output entropy (0.9999)
is NOT the aggregate load entropy (0.9582), and MoRE's AMI (0.1803) is a healthy ~0.15 above
its permutation control. A near-maximal per-token entropy coexisting with balanced load and
real, above-chance specialization is not evidence of a broken router. The T-LX.20 sweep
confirms the direction: setting the balance weight to 0 **collapses** load (entropy 0.0001,
one expert takes everything) and drives AMI to zero, so the balance term is load-bearing and
"lower the balance weight" is the wrong fix. `CLAUDE.md` §2 still forbids *tuning toward*
maximal entropy and §6 forbids reading high entropy *as* specialization — both hold — but the
honest sentence is that the high per-token entropy is a benign property here, not the cause
of MoRE's deficit. The cause is the depth/specialization competition (§6) and the budget
dilution, not the router being miscalibrated.

Compare against the corpus's own POS-partition load entropy, read from the manifest and
never pasted: **0.894156 for wikitext-103** (wikitext-2 is 0.888442, and the difference
flips the sign of the comparison — this error reached five documents once already).
MoRE's 0.9582 is **above** the partition the data itself has, i.e. it is buying uniformity
the corpus does not exhibit.

### Cost

| arm | items/s | relative |
|---|---|---|
| MoE | 718.8 ± 7.2 | 1.0× |
| MoR | 173.8 ± 6.3 | 4.1× slower |
| MoRE | 97.0 ± 1.0 | 7.4× slower |
| *MoRE per-step (retired)* | *168.5 ± 2.5* | *4.3× slower* |

Route-once makes **7× fewer router calls** and exits early **more** often than per-step,
so it performs strictly less arithmetic — yet throughput fell 168.5 → 97.0. That 1.74×
regression is **overhead in the persistent per-token dispatch path, not extra compute**.
Say it that way; calling it a compute cost would be wrong. And per `CLAUDE.md` §4, a FLOP
ratio is **not** a speedup — the workload is latency-bound and measured throughput is the
only honest cost number.

---

## 5. The claim structure for TMLR

**Target: TMLR.** It accepts well-executed diagnostic and negative results and judges on
correctness rather than novelty-of-win, which is exactly this paper's profile. The top-tier
conference objection that cannot currently be answered is **scale** (see §10), and TMLR is
where that is a stated limitation rather than a rejection.

Frame the paper as a **controlled decomposition**. The unit of contribution is the ablation
lattice, not a win.

1. **Claim.** MoE and MoR answer orthogonal questions — *which* computation and *how
   much*. The natural composition should inherit both. We test that directly, at matched
   parameters from one codebase, and it does not.
2. **Setup.** One engine, three modes, bit-identical inputs per record, parameters matched
   to 0.069%, 5 seeds, exact randomization tests, Holm correction over a pre-registered
   family.
3. **Result 1 — the ordering.** MoR > MoRE > MoE. Recursion helps, sparse routing hurts,
   the composition lands between them.
4. **Result 2 — depth saturation is a budget effect, not an architectural limit.** §6.
5. **Result 3 — the contribution: the two mechanisms compete.** §6.
6. **Result 4 — the honesty payload.** §7.
7. **Limitations and the forward prediction.** §10. State the scale threshold as a
   falsifiable claim, not as a hedge.

---

## 6. Why the result came out this way — three causes, with their evidential status

Label these correctly in the paper. `CLAUDE.md` §4 requires measured facts be
distinguished from interpretation everywhere.

### Cause 1 — recursion multiplies capacity; sparse routing divides it. (MEASURED ordering, HYPOTHESIS mechanism)

MoR's single block sees 100% of tokens. MoE and MoRE split the same FFN parameter budget
across six experts, each seeing a fraction of the stream (`max_load_fraction` 0.448, so
unevenly). Under a frozen 3-epoch budget each expert receives roughly 1/E of the updates
per parameter. The loss ordering tracks this exactly:

- MoE — divides only — 3.9529 (worst)
- MoRE — divides **and** multiplies — 3.6370
- MoR — multiplies only — 3.5202 (best)

**The ordering is measured. The capacity-dilution explanation is a hypothesis** consistent
with it, and the paper must say so. It is the hypothesis that generates the scale
prediction in §10, which is why it is worth stating despite not being isolated.

### Cause 2 — the ponder weight does not transfer across corpus scale. (MEASURED, three points)

Same halting weight 0.001 throughout; only the number of optimizer steps differs:

| setting | avg depth | source |
|---|---|---|
| dev corpus (wikitext-2), 3 epochs | **2.70** of 7 | `code/lang_calibration_weights.json` |
| dev corpus, 8 epochs | **5.16** of 7 | `code/calibrate_lang_weights.py` docstring |
| wikitext-103, 3 epochs | **6.96** of 7 | the canonical arm |

Depth is monotone in training steps at a fixed ponder weight. The halting weight was
frozen on a corpus ~60× smaller than the one the paper reports — a corpus where it
produced genuinely adaptive depth — and **it does not transfer**.

**This is the most important correction in the whole study, and it changes a claim from
fatal to fixable.** The honest statement is *"this ponder weight cannot hold depth down at
this corpus scale"* (a calibration finding), **not** *"MoRE cannot learn adaptive
computation"* (an architectural one).

**Corollary you must not get wrong:** ρ(depth, token loss) = −0.023 is **not** evidence
about allocation. With 97.9% of validation tokens exiting at step 7 and depth std 0.43,
that correlation is computed against a near-constant. A near-zero correlation with a
constant is not a finding. Writing it up as "MoRE fails to allocate depth by difficulty"
would be the paper's most likely fatal reviewer catch.

**One discrepancy to resolve before you cite it.** The `frozen_by` prose in
`code/canonical_spec_language.json` quotes the `(0.001, 0.001)` dev point as
`val 5.2406, depth 2.63, AMI 0.2907`, while `code/lang_calibration_weights.json` records
`val_loss 5.2352, avg_depth 2.698, ami 0.2851` for what appears to be the same point. The
JSON is the machine-written record and should win, but **check which is right before
putting either in the paper**, and fix the loser in the same commit. Do not average them.

### Cause 3 — the two mechanisms compete for the same budget. (MEASURED on the dev corpus; this is the novelty)

From the same calibration grid, raising the halting weight to recover adaptive depth also
destroys expert differentiation:

| halting weight | avg depth | AMI | load entropy | val loss |
|---|---|---|---|---|
| 0.001 (frozen) | 2.698 | **0.2851** | 0.853 | 5.2352 |
| 0.107 | 1.476 | **0.0025** | 0.549 | 5.3346 |
| 0.5 | 1.072 | 0.0207 | 0.285 | 5.3306 |

AMI 0.0025 is statistically indistinguishable from the shuffled control. Mechanism: fewer
recursion steps means fewer expert applications per token, means fewer opportunities for
experts to differentiate.

**So depth and specialization are not independent knobs in MoRE the way they are in MoR
and MoE separately.** Tune toward adaptive depth and you lose specialization; tune toward
specialization and depth saturates. **That is the paper's actual contribution**: the
composition has a structural conflict because both mechanisms spend the same per-token
compute budget, and at this parameter and data budget neither gets enough.

Caveat to state: this grid is on the **dev corpus at one seed**, so it is a mechanism
demonstration, not a canonical result. `code/lang_calibration_weights.json` carries its own
caveat field — quote it.

### Cause 4 — why fixing the architecture lowered the number. (MEASURED gap, HYPOTHESIS mechanism)

The retired per-step arm scored 3.5031 ± 0.0133 and was statistically indistinguishable
from MoR (+0.0172, p = 0.4048, a genuine null). The architecturally-correct route-once arm
scores 3.6370 ± 0.0134, **+0.1167 worse than MoR** at d = +3.87.

Hypothesis: per-step re-routing let a token traverse up to seven *different* expert FFNs —
a heterogeneous seven-layer network, not one block iterated — so it was strictly more
expressive than the spec allows. The old arm's apparent parity with MoR was an artifact of
a spec violation. Report this. It is the single most credibility-building paragraph
available to you.

---

## 7. The honesty payload — disclose all of these

Each of these is a thing a reviewer could otherwise discover and hold against the paper.
Disclosed, each one *adds* credibility.

1. **The per-step arm was not MoRE as defined.** It re-routed at every recursion step,
   violating `CLAUDE.md` §1. It is kept unedited as a labelled ablation in
   `archive/pre_finalization/more_l7_1_per_step/`. Its `resolved_config.json` files must
   never be rewritten to add `routing_persistence` — that is explicitly prohibited in
   `code/confirmatory_tests.json`.
2. **Fixing it made the headline number worse**, by 0.134 nats. Say so plainly.
3. **The pre-registration is partial, and the file says so itself.** Read the
   `HONEST DISCLOSURE` block in `code/confirmatory_tests.json`: MoE and MoR were already
   trained when the families were declared, so only the MoRE-involving comparisons are
   genuinely pre-result. The `mor_vs_moe` test carries `"pre_result": "known"` and the
   paper **must** state that this one comparison was specified after its data existed. Do
   not describe the file as fully pre-registered.
4. **A prediction was recorded before the run and it was confirmed.**
   `PAPER_REVISION.md` §6 predicted in advance that the route-once re-run would land near
   the depth cap, and recorded that prediction specifically so it could not be retrofitted.
   Depth came in at 6.955 of 7. That is a pre-registered confirmed prediction — use it.
5. **Every p-value is identical at 0.00794 because that is the floor.** See §11.
6. **Provenance gaps.** The five MoRE cells record `gpu_name = Tesla V100-SXM2-16GB`,
   `sm_70`, `torch 2.6.0+cu124`. The ten MoE/MoR cells predate those fields and carry
   `None`; their V100 16 GB attribution rests on the operator record in
   `RUNBOOK_V100.md` §Phase 1. **Do not retrofit the older files** — a provenance field
   invented after the fact is worse than an absent one. All 15 cells also have
   `code_git_dirty = True`, and the matrix spans **4 distinct git commits**; the matrix is
   frozen by `canonical_spec_language.json`, not by one revision, and the seed-blind config
   equality check is what holds the arms comparable. The exporter lists all four commits in
   the generated table header — reproduce that in the reproducibility statement.
7. **Three loss terms are structurally zero on language and the reason is not laziness.**
   `family_cls_weight = 0.0` because the family head reads `h.detach()` so its gradient
   cannot reach the trunk — which **strengthens** every specialization claim relative to
   the arithmetic study, where a 6-way family cross-entropy was the second-largest term in
   the objective. `step_routing_weight = 0.0` and `halting_supervision_weight = 0.0`
   because language has no per-token oracle expert and no per-token oracle depth. Say all
   three; the first is a genuine advantage of this task.
8. **`depth/allocation_error_*` is `N/A`, not 0.0.** There is no per-token ground-truth
   depth for English. A 0.0 would read as perfect allocation against a curriculum that
   does not exist. `CLAUDE.md` §4: no sentinel may be reported as a measurement.

---

## 8. Files that will mislead you

These are not wrong about their own subject; they are stale with respect to *this* paper.

- **`PAPER_REVISION.md`** — written when the arithmetic paper was still live. Its §1 (write
  the ACT forward path down) and §3 (exposition) are **fully valid and are real work**. But
  strike the arithmetic-only items, and the §2.3 test-split item is **not** blocked on
  fetching checkpoints any more — the test split is provably untouched and the honest
  action is to relabel and state the protocol. Treat the file as an agenda to prune, not a
  spec.
- **`README.md`** — the arithmetic POC. Historically correct, deliberately left alone.
- **`TASKS.md`** vs **`TASKS_LANGUAGE.md`** — the second is the language one.
- **`plan.md`** vs **`plan_language.md`** — likewise. `plan_language.md` §5.1 is where
  "there is no per-token ground-truth depth on language" is established.
- **`HANDOFF.md`** — the procedure for Ayan's local 4060. Still correct for that machine.
  `RUNBOOK_V100.md` supersedes it only when the GPU is rented.
- Any file still quoting the resolution floor as `1/C(n_a+n_b, n_a)` — that was corrected
  at T-LX.18 and every known site was fixed, but if you find a survivor, it is wrong. See
  §11.

---

## 9. What was never run, and how to describe it

Do **not** describe these as future work in a way that implies they were optional. State
them as limitations.

- **The depth-matched fixed-depth control.** `more_vs_depth_matched_fixed` is a declared
  Family A comparison and it is **unrun**. It still consumes its Holm slot — that is the
  whole point of declaring family size in advance, and `code/confirmatory_tests.json`
  explicitly prohibits deleting it to raise the other three thresholds. The paper reports
  it as N/A with its slot consumed. This is the control that would have tested whether the
  benefit is adaptivity rather than merely the average compute spent, so its absence is a
  real limitation and should be named as one.
- **The fixed-depth sweep d = 1..6**, the shuffled-depth control, and the depth–input
  mutual information analysis — all listed as exploratory in
  `code/confirmatory_tests.json`. Unrun.
- **The T-LX.20 weight sweep** (`automated/sweep_tlx20_weights.py`) was launched on
  Vedant's 3050 and writes to `code/lang_sweep_tlx20.json` after every point. **Check
  whether that file exists and is complete before you write §6.** If it has rows, they
  test Cause 2 and Cause 3 on the canonical corpus rather than the dev corpus, which
  materially strengthens both. If it is absent or partial, write §6 from the dev-corpus
  grid and say the canonical-corpus confirmation was not run. Every row in that file is a
  `subset_fraction < 1`, single-seed proxy and **may never be quoted as a canonical
  result** — it picks a hypothesis, it does not test one.
- **A second model size.** Not run, and this is the one that would have made the paper
  main-track viable at a top conference. See §10.

---

## 10. Limitations, written as claims

The scale objection is the one a strong reviewer will lead with, and it is **correct**.
Do not bury it.

The study operates at 5.6M parameters, d_model 256, one block, vocab 8192, three epochs on
wikitext-103. Sparse-MoE benefits are known to emerge with scale, so a reviewer can
reasonably say the negative result may be an artifact of operating below the regime where
sparsity pays. **That cannot be answered with the runs that exist.**

The productive move is to convert it into the paper's forward claim. The capacity-dilution
account in §6 *predicts* a scale threshold: each expert needs enough tokens per parameter
for the division to stop dominating the multiplication. Write that prediction down with
the quantity it depends on (tokens per expert per parameter), show that all three arms sit
below it, and state what measurement would falsify it. A limitation with a falsifiable
prediction attached reads as a contribution; the same limitation unaddressed reads as a
confound.

Secondary limitations to state plainly: one task and one corpus; three epochs, with a
mechanism that is demonstrably budget-dependent, so the 10-epoch behaviour is genuinely
unknown; one hyperparameter configuration per arm, frozen from a dev-corpus grid that
§6 Cause 2 shows mis-transferred at least one weight; and n = 5 seeds, which puts every
headline comparison exactly on the design's resolution floor (§11).

---

## 11. The statistics, and the exact sentences to use

This layer was wrong until T-LX.18 and a reviewer caught it. Get it right.

**The test.** Exact two-sided randomization test on the difference of means
(`code/seed_stats.py:perm_test`). It enumerates every way of splitting the pooled values
into groups of the original sizes — C(10,5) = 252 at five-vs-five — and counts the
fraction whose `|mean difference|` is at least the observed one. Exhaustive, therefore
deterministic: **no sampling error, no seed, no normality assumption, no threshold to
choose.** The superseded `k × std` heuristic is not used anywhere and must not appear.

**The resolution floor is `2 / C(n_a+n_b, n_a)` for equal arms, `1 / C` otherwise.** The
statistic is `|mean(a) − mean(b)|`. When the arms are the same size, the complement of any
group-a subset is itself a valid group-a subset and swapping the groups negates the
difference while preserving its absolute value — so extremes come in **mirror pairs**, the
count at the extreme can never be 1, and the floor is 2/C. When the arms differ in size
the complement has the wrong size, is not an admissible regrouping, no mirror is forced,
and 1/C is attainable. At five-vs-five the floor is **2/252 = 0.00794**.

**This is why all three p-values are identical.** Every headline comparison in the paper
sits exactly on that floor. Phrase it as *"as extreme as a five-seed design admits"* —
which is a **weaker** statement than significance and must not be dressed up as a stronger
one. Only more seeds can strengthen these claims; no larger effect can.

**The correction.** Holm–Bonferroni over the families declared in
`code/confirmatory_tests.json` (Family A: predictive quality, m = 4; Family B: adaptive
depth mechanism, m = 2). Holm is a step-down procedure — sort p ascending, compare the
i-th against `alpha/(m−i)`, reject while that holds, stop at the first failure and retain
every larger p. It controls the same familywise error rate as plain Bonferroni and is
uniformly more powerful, so the plain version is strictly worse here with no compensating
property.

**Explain why that choice is load-bearing, because it is:** plain Bonferroni over seven
comparisons gives 0.05/7 = 0.00714, which is **below** the five-vs-five floor of 0.00794 —
a design in which no effect of any size could be declared significant. Holm's easiest
threshold at m = 4 is 0.0125, a margin of 0.0046 above the floor. A reviewer who checks
this will find the reasoning already done.

**Two phrasings that are forbidden.**
- A comparison whose floor exceeds its Holm threshold is **undecidable** and reports
  **N/A**, never "not significant". Calling it not-significant blames the data for a limit
  of the design.
- Never write `1/C` as the two-sided floor for equal arms. It is 2/C.

**Also report, and never instead of the p-value:** Cohen's d on the pooled SD, and the raw
gap. `|d|` reaches 4.1 on differences of 0.0005 elsewhere in this project — a large d on a
tiny gap is exactly what a reader must not be allowed to misread.

---

## 12. Forbidden sentences — a checklist to run over the draft

From `CLAUDE.md` §4, §6 and §8. Grep the draft for each.

- ✗ `total_loss` or `weighted_total_loss` as a quality metric. The primary predictive
  metric is **validation task loss**. A negative weighted total is not automatically a bug
  and is never evidence of quality.
- ✗ "compute efficiency" for a depth number. It is **depth allocation**, unless real FLOPs
  are in the metric. A FLOP ratio is not a speedup — this workload is latency-bound.
- ✗ "proves orthogonality" from cosine similarity. Correct phrasing: *"consistent with
  differentiated parameterizations."* Report mean **and** max pairwise.
- ✗ High entropy as evidence of specialization — but equally, ✗ high per-token router
  entropy as evidence of a *broken* router (it coexists with balanced load and real AMI
  here; see the router-entropy note).
- ✗ Low cosine similarity as proof of orthogonality.
- ✗ "discovers intrinsic mathematical complexity" — and no language analogue of it either.
- ✗ Any sentinel (`-1`, `0.0` placeholder) reported as a measurement. Use `N/A`.
- ✗ Entropy for `E = 1` reported as 0 and compared. It is `N/A` for MoR.
- ✗ A metric key with `/` inside a label (creates spurious nesting): `E1_ADD_SUB`, not
  `E1 ADD/SUB`.
- ✗ Mixing the arithmetic MSE table and the language nats table.
- ✗ Describing `confirmatory_tests.json` as fully pre-registered.
- ✗ Any claim of causality without an experiment that isolates the variable. Causes 1 and
  4 in §6 are **hypotheses** — label them.
- ✗ Reporting a per-test α = 0.05 verdict when several comparisons are read off one table.

---

## 13. If a number you need disagrees with this file

This file was written on 2026-10-04 from the export generated at commit `84d585f`. It can
go stale; the generated tables cannot.

**Order of trust:** `results/language/results.json` → `results/language/results_tables.md`
→ `changelog.md` → this file. If this file disagrees with the generated output,
**regenerate and trust the output**, then fix this file in the same commit.

If the exporter refuses with `duplicate cell`, two `canonical_lang_b` cells exist for one
`(arch, seed)` under `runs/` — retire one with **`git mv`** (not `mv`, which leaves it in
the index at the old path and is what caused exactly this failure once already) and
confirm the deletion is staged. The full trap is written up in `RUNBOOK_V100.md` §Phase 6
and in the T-LX.19 changelog entry.

Append a `changelog.md` entry when you finish a section of the paper, naming the file and
the claim it settled. That is how the next session finds out what you decided and why.
