# ARCHITECTURE.md — MoRE codebase orientation

**Audience:** an agent or human opening this repository for the first time.
**Purpose:** what the code *is* and *why it is shaped this way*, so you don't
"fix" a deliberate decision.

This file describes **structure**. It does not override:

| Document | Authority |
|---|---|
| [plan.md](plan.md) | execution plan, phases 0–11, gates |
| [updated_rules.md](updated_rules.md) | architectural + scientific constraints |
| [updated_objective.md](updated_objective.md) | what is being tested, what counts as success |
| [CLAUDE.md](CLAUDE.md) | condensed working contract |
| [TASKS.md](TASKS.md) | task checklist + the evidence behind every tick |

On conflict, those win. `archive/` is invalidated history — never cite it.

---

## 1. The one decision that explains the layout

> **There is one training system. Architecture is a runtime *mode*, not a
> codebase.**

`plan.md` §9 and `CLAUDE.md` §9 forbid parallel `run_moe.py` / `run_mor.py`
implementations. The reason is scientific, not aesthetic: the paper's central
claim is a *comparison* between MoE, MoR and MoRE. If each had its own data
loader, its own loss assembly, or its own metric code, any measured difference
would be confounded with an implementation difference, and no amount of care
would let you separate the two.

So there is exactly one package, `code/more/`, and three thin launchers that
differ only in which string they pass in.

### Why there is no `moe/` and no `mor/` folder

This is the question every new reader asks. **It is intentional and verified,
not unfinished work.**

`code/more/` is the ***project*** package — MoRE is the name of the repository
and the research. It is **not** the "more" architecture mode. All three
architectures are implemented inside it and selected by
`architecture = "moe" | "mor" | "more"`.

Adding `code/moe/` and `code/mor/` would mean three copies of `engine.py`,
`data.py`, `metrics.py` and `model.py`. That is precisely the confound the rule
exists to prevent, and it is how the invalidated results in `archive/` were
produced.

**Evidence that the three modes share one pipeline.** All three launchers were
run for one epoch on the full dataset and their `resolved_config.json` files
diffed field by field:

```
TOTAL fields = 47   IDENTICAL across all three = 35   DIFFERING = 12
```

Of the 12 differing fields, **6 are run identity** (`run_name`, `config_hash`,
`experiment_id`, `run_dir`, `wandb_run_id`) and cannot be equal by definition.
The other **6 are the architecture definition itself**:

| field | MoE | MoR | MoRE |
|---|---|---|---|
| `architecture` | `moe` | `mor` | `more` |
| `model.num_experts` | 6 | **1** | 6 |
| `model.max_depth` | **1** | 7 | 7 |
| `model.adaptive_halting` | **False** | True | True |
| `loss_weights.halting` | **0.0** | 0.001 | 0.001 |
| `loss_weights.routing_balance` | 0.05 | **0.0** | 0.05 |
| `loss_weights.step_routing` | 0.5 | **0.0** | 0.5 |

Everything else — `data.dataset_version`, all four split paths,
`data.subset_fraction`, `data.max_val`, `model.d_model`, `model.max_steps`,
`model.step_feat_dim`, the optimizer block — is **bit-identical**. That is the
comparability guarantee, and it is a re-runnable check, not a claim.

Read the table as the definition of the ablation:

- **MoE** = routing without depth (6 experts, depth 1, halting off)
- **MoR** = depth without routing (1 expert, depth 7, halting on)
- **MoRE** = both

To reproduce the diff, run the three launchers with `--epochs 1` and compare
`runs/<experiment_id>/resolved_config.json`.

### Why the package is not called `train/`

A directory named `code/train/` would sit beside `code/train.py`. On an
`import train`, Python resolves the **package** first, so `train.py` would
become unimportable and the entry point would silently shadow itself. `more/`
avoids this. Do not rename it.

---

## 2. Repository map

```
MoRE/
├── plan.md, updated_rules.md, updated_objective.md   governing documents
├── CLAUDE.md            agent working contract
├── TASKS.md             checklist; every [x] carries its verification evidence
├── ARCHITECTURE.md      this file
│
├── code/
│   ├── train.py         (19 lines)  generic entry point, --architecture required
│   ├── train_moe.py     (16 lines)  launcher, default_architecture="moe"
│   ├── train_mor.py     (16 lines)  launcher, default_architecture="mor"
│   ├── train_more.py    (16 lines)  launcher, default_architecture="more"
│   │
│   ├── more/            THE training system (one implementation, three modes)
│   │   ├── __init__.py      (32)   public surface
│   │   ├── families.py      (58)   six-expert manifest, op→expert map, op codes
│   │   ├── data.py         (219)   MoREDataset, per-step tokenisation
│   │   ├── model.py        (501)   MoREModel, MoREBlock, router, halting head
│   │   ├── engine.py       (594)   THE train/val loop, loss assembly, logging
│   │   ├── metrics.py      (369)   entropy, cosine, depth, routing, figures
│   │   ├── run_context.py  (488)   per-run dirs, provenance, Gate-0 proxy guard
│   │   ├── seeding.py      (315)   T6.1 global seeding: random/numpy/torch/CUDA
│   │   │                           + DataLoader generator & worker, determinism,
│   │   │                           runtime non-determinism recorder
│   │   ├── config.py       (139)   config load + resolution, ARCH_DISPLAY,
│   │   │                           canonical_run_name, resolve_variant
│   │   └── cli.py           (83)   argparse → RunContext → engine.train
│   │
│   ├── canonical_spec.json  the frozen definition of "canonical"
│   ├── config.json          the canonical config
│   ├── audit_leakage.py     Gate 1 — 8 leakage / parity / floor checks
│   ├── test_phase6_seeding.py         T6.1 — 61 seeding / determinism checks
│   ├── test_phase6_routing_metrics.py T6.2/T6.3 — 42 permutation-invariance checks
│   ├── test_phase6_provenance.py      T6.6/T6.7 — 32 naming / provenance checks
│   ├── test_gate5.py                  GATE 5 — 34 closure checks (run LAST: it
│   │                                  reads the run dir the provenance test makes)
│   ├── smoke_test.py        fast import + shape sanity
│   └── verify_pipeline.py   35 static + runtime pipeline checks
│
├── data/
│   ├── script.py            generator (with verify())
│   ├── train/val/test.jsonl the dataset
│   └── dataset_meta.json    SHA-256s, trivial baselines, Gate-1 audit record
│
├── runs/<experiment_id>/    ALL run output lands here (repo root, not code/)
│   ├── config.json  resolved_config.json  metrics.json
│   ├── results.tsv  checkpoint.pt  stdout.log
│
├── automated/               exploratory sweep drivers — never canonical runs
└── archive/pre_finalization/  INVALIDATED history. Do not cite.
```

The largest module is 594 lines. The pre-refactor `train.py` was 1615 lines in
one file; it was split so an agent can edit the loss assembly without loading the
data loader into context.

---

## 3. Call graph

```
train_more.py
  └─ more.cli.main(default_architecture="more")
       ├─ more.config      load config.json, apply CLI overrides
       ├─ more.run_context.RunContext.create(raw, resolved)
       │     ├─ Gate-0 proxy guard (may raise ProxyGuardError)
       │     ├─ mkdir runs/<experiment_id>/
       │     ├─ write config.json + resolved_config.json + provenance
       │     └─ tee stdout → stdout.log
       └─ more.engine.train(cfg, ctx=ctx)
             ├─ more.seeding.apply_seeding(ctx.resolved_cfg)   ← T6.1, FIRST
             │     random / numpy / torch / CUDA + determinism flags, then
             │     provenance.resolved_seed / seed_source / determinism_* and a
             │     re-flush of resolved_config.json. Anything that draws random
             │     numbers must be constructed BELOW this line.
             ├─ more.data.MoREDataset  ×3  (train / val / test files)
             ├─ more.model.MoREModel(num_experts, max_depth, adaptive_halting)
             ├─ per-epoch: forward → loss assembly → step → validate
             ├─ more.metrics.*  (entropy, cosine, confusion, depth, figures)
             └─ ctx.write_metrics(...) → metrics.json ; results.tsv ; checkpoint.pt
```

`more/engine.py` is the only place a training step exists. If you change the
objective, you change it once and all three architectures move together — which
is the point.

---

## 4. Data contract

Per record, `MoREDataset.__getitem__` returns a **7-tuple**:

| name | shape | dtype | meaning |
|---|---|---|---|
| `x` | `[max_steps, step_feat_dim]` = `[7, 12]` | float32 | per-step token matrix; **numeric arguments only** |
| `step_mask` | `[7]` | bool | True = real step, False = pad |
| `step_experts` | `[7]` | long | oracle expert per step, `-1` = pad |
| `step_ops` | `[7]` | long | operation code per step, `-1` = pad |
| `family` | scalar | long | whole-program family 0–5, `-1` = MIXED |
| `depth` | scalar | long | number of real steps |
| `target` | scalar | float32 | normalised final answer |

### What must never enter `x` (`updated_rules.md` §3)

```
expert_id / family_id as a numerical input feature
oracle routing class in an input slot
the regression target (final answer / step result) in input features
```

All three were present before Phase 1 and all three are now proven absent. See
7.

**Operation identity is legitimate task information** — it tells the model
*which function to compute*. It travels in `step_ops` and is consumed by
`nn.Embedding(NUM_OP_TYPES=16, d_model)`. That is an embedding over
**operations**, never over experts. Do not "simplify" it into `x`.

### No catch-all

Six experts: `E1 ADD/SUB`, `E2 MULT/DIV`, `E3 MOD/POW`, `E4 LOGIC`, `E5 SHIFT`,
`E6 SORT/STAT`. The old E7 catch-all is removed. An unmapped operation or family
**raises** in `data.py` — it never falls back to a bucket. Multi-operation CHAIN
programs get family `-1` (`MIXED`) and are excluded from the classification loss
via `ignore_index=-1`; they still train the task and routing losses.

All tensor dimensions derive from `num_experts` (**T5.1**, verified at E = 1, 6
and 7 — Gate 4). Labels derive from `expert_labels(num_experts)` in
`families.py`; call that rather than indexing `EXPERT_FAMILY_LABELS`, which is
only correct at E = 6. The two remaining `7`s in the config —
`model.max_depth` and `model.max_steps` — are the recursion budget and the number
of steps per program, **not** the expert count, and must not be "fixed" to 6.

---

## 5. Routing: the canonical path and the one permitted ablation

Set by `model.routing_mode`, validated by `config.enforce_routing_mode()`:

| mode | what it does | status |
|---|---|---|
| `top1_sparse` | `logits → softmax → argmax → **that expert only** → × selected gate prob` | **canonical default** |
| `dense_blend` | evaluates every expert on every token, blends by router prob | labelled ablation **only** |

Dense is not MoE — no token is routed, every expert sees every token, and expert
independence is gone. `updated_rules.md` §1.1 (ablation F) permits it solely as
"MoRE – Dense Routing Ablation", so the gate is on the **run label**: the run name
must contain `dense_routing_ablation` or the run refuses to start. A quiet
boolean flag would satisfy the rule while leaving the output
indistinguishable from canonical MoE in a table — which is the actual risk, since
**both paths emit `[N, d_model]` and both report an argmax.** The loss curve,
entropy and confusion matrix look normal either way.

`enforce_routing_mode()` is called from `cli.py` **after** override resolution,
not from `apply_architecture()`. Called earlier it tests the default run name and
refuses even correctly-labelled ablations. Any label check must run after
overrides.

**How to tell which path a finished run used:** `dispatch/evals_per_token` in its
`metrics.json`. `1.0` = every token seen by exactly one expert (true sparse);
`num_experts` = dense. `dispatched` counts tokens and is identical on both paths,
so it cannot answer the question.

**Capacity policy: `no_capacity_limit`** (`plan.md` §4.3 — prefer the simple
implementation when the distribution does not require capacity limiting). Nothing
is dropped. Consequences for reading the logs:

- `overflow = 0` is a **measured** zero, not a placeholder for "not implemented".
- Real capacity pressure is `max_load_fraction`.
- **Reading caveat:** MoRE's `max_load_fraction = 1.0` is the worst single step
  across blocks × depths. At deep recursion steps few tokens remain active, so one
  expert legitimately takes 100% of a 1-token step. It is **not** router collapse.
  MoR's `1.0` is `1.0` by definition (one expert).

The router's only gradient path to the task loss is the **gate multiply** at the
end of the sparse branch (`argmax` has no gradient). Remove or detach it and the
router silently trains on the auxiliary and oracle terms alone; nothing else in
the pipeline will complain. Measured non-zero: router grad `30.010744`.

### Router exploration noise (`model.router_noise`) — off in canonical

| mode | behaviour | `state_dict` |
|---|---|---|
| `none` | logits used as-is; **canonical default** | no noise key at all |
| `fixed_annealed` | Gaussian noise scaled by `router_noise_init × (1 − step/anneal_steps)`, training mode only | no noise key (the counter is a plain `int`, deliberately not a buffer) |
| `trainable` | noise scaled by a learned `nn.Parameter`, training mode only | one `blocks.*.moe_block.router_noise_scale` per block |

Before **T5.4** the trainable variant was the *only* behaviour: a
`nn.Parameter(0.1)` was always created and always injected, while appearing in no
config file, no `resolved_config.json` and no W&B record. Three reasons it cannot
be canonical: it is an unreported architectural difference; a *trainable* scale
flattens the softmax and therefore moves the balance objective's entropy term
directly; and the L2-on-noise-scale penalty is explicitly **not** a proven fix
(`CLAUDE.md` §2).

Canonical creates no parameter rather than pinning one to `0.0`, so its absence is
verifiable — a pinned parameter would still sit in the optimizer and still have to
be reported. Two consequences worth knowing: canonical dispatch is deterministic
even in training mode (measured 64/64 tokens identical across two passes, versus
9/64 moved with noise on), and the noise-regularization term in `engine.py` is
gated on the parameter's existence, so it is an exact `0.0` tensor in canonical
rather than a term over a parameter that is not there. `assert_not_silent_proxy`
refuses any canonical claim with noise on or with `dense_blend`, and unlike the
`enforced_fields` loop that check fires **even while `canonical_spec.json` is
unfrozen**, because these are architectural choices rather than tuning.

### One config value, one source (`subset_fraction`, `subset_seed`)

`subset_fraction` and `subset_seed` appear in both `training` and `data`.
Before **T5.3**, `load_config` applied `setdefault` to `training` and *then* asked
`if "subset_fraction" in cfg["training"]`, which was therefore always true — so
`data.subset_fraction` was overwritten on every load. All consumers read `data.*`
(the `Subset` in `engine.py`, the proxy guard, `resolved_subset_fraction`), so a
config asking for half the data trained on all of it and *reported* `1.0`:
self-consistent and wrong. Now both sections are snapshotted **before** any
default is applied, a value given in either section wins and is written to both,
and a genuine disagreement raises instead of letting `training` win silently.

---

## 5a. Halting: canonical ACT, and what the depth numbers mean

`MoREWrapper.forward` in [code/more/model.py](code/more/model.py) implements ACT
exactly as Graves 2016 / Universal Transformer define it — `plan.md` §5.2 forbids
inventing a variant:

```text
p_t   = sigmoid(expert_halt_heads[e](state_t))    # per-expert head
c_t   = running cumulative halt mass
halt  when c_t + p_t > halt_threshold = 1 - halt_eps   (eps = 0.01)
R     = 1 - c_t                                   # remainder at the halt step
w_t   = p_t while continuing, R at the halt step   ->  Σ w_t == 1 exactly
out   = Σ_t state_t * w_t                          # convex combination
```

**Why the convex combination, and not "write the state at exit time".** Copying
the exit-time state is a discrete choice: the halt probability decides *which*
state is copied and then leaves the graph. That is what the pre-Phase-3 code did,
and the consequence was measured, not inferred — **all 24 halt-head parameters had
`grad is None` after a full backward pass through the real objective.** Halting
was never trained; the depth spread in those runs came from random initialisation.
The `state_t * w_t` multiply is the only path from the **task** loss to the halt
heads. Treat it the way you treat the router's gate multiply (§5): if it is
removed or detached, training continues and nothing complains.

`halt_eps > 0` is load-bearing. At a threshold of exactly 1.0 a token with
`p_1 = 0.999` is still forced into a second step, so a depth-1 exit is unreachable
and E1 / E4 / E5 could never reach their curriculum target of 1.

**Two halting terms, never summed.** They pull in opposite directions, so a single
combined number can sit flat while both components move a lot (`plan.md` §5.4):

| term | key | what it is | gradient path |
|---|---|---|---|
| ponder cost | `train/ponder_cost` | `mean(N + R) / (max_depth + 1)` | through `R`; `N` is a discrete count |
| curriculum supervision | `train/halting_supervision_loss` | MSE of `N + R` against `families.OP_TARGET_DEPTH` | through `N + R` |

Both normalizations are deliberate. Dividing the ponder cost by `max_depth + 1`
keeps it in (0, 1] so the term does not grow with `max_depth`; averaging it over
blocks rather than summing keeps it from growing with `num_blocks`. Without both,
a deeper or wider model is penalised for its *shape* instead of its *behaviour* —
the same magnitude defect `CLAUDE.md` §2 forbids for the balance loss.

**The operation-complexity curriculum is a design choice, not data.** The dataset
has no per-operation depth label — records carry `depth`, which is program chain
length. `families.OP_TARGET_DEPTH` was written by hand and says so in-source. Two
consequences:

- The only defensible claim is *the model learns to allocate recursion depth in
  accordance with the predefined operation-complexity curriculum.* Never
  "discovers intrinsic mathematical complexity".
- The table is deliberately **correlated with but not identical to** the expert
  families: E4 LOGIC and E5 SHIFT both target 1, and E6 SORT/STAT spans 2
  (MAX/MIN) and 4 (SORT/MEDIAN). If target depth were a pure function of the
  expert index, perfect routing would imply a perfect depth-allocation error and
  the depth metric would carry no information independent of routing accuracy.

**Supervised or unsupervised is a config choice recorded in provenance.** Default
is `halting_mode = pure_act`. `--halting_supervision` (with
`--halting_supervision_weight`) switches to `supervised_curriculum`, and that
string lands in `resolved_config.json`, `metrics.json` and the W&B config.
**Read `halting_mode` before reading any depth number:** a supervised run was told
the answer, so its allocation error is not evidence the architecture found
anything. Two guards raise rather than mislabel — enabling the flag with a zero
weight (would record "supervised" while training unsupervised), and enabling it on
MoE (`max_depth = 1` makes targets of 2–4 unreachable, so the term is a permanent
irreducible loss with an unsatisfiable gradient).

**The decision: canonical halting is `pure_act`.** Settled by a controlled
experiment, [automated/act_decision.py](automated/act_decision.py), whose decision
rule is written into the top of the driver *before* the runs. Matched protocol —
`architecture=more`, 20 epochs, `blocks=1`, `batch_size=768`, seeds 42/43/44,
identical lr / d_model / weight decay / balance weights; the **only** difference is
`--halting_supervision` (weight 0.1). Proxy runs (reduced epochs, hyperparameters
not yet frozen), so no number here may enter the headline table — only the mode
decision survives.

| MoRE, mean ± std over 3 seeds | val task loss | avg depth | depth alloc err (abs) | per-op depth spread | corr. with `OP_TARGET_DEPTH` |
|---|---|---|---|---|---|
| `pure_act` | **0.063676 ± 0.000283** | 2.292 | 1.025 ± 0.059 | 0.452 ± 0.044 | **+0.204 ± 0.554** |
| `supervised_curriculum` | 0.063953 ± 0.000190 | 2.057 | **0.445 ± 0.012** | 0.983 ± 0.084 | +0.957 ± 0.027 |

The gap on the primary criterion is `−0.000277` against a largest seed std of
`0.000283` — inside noise. `pure_act` therefore wins by the pre-registered tie
rule: only an unsupervised run supports the adaptive-computation claim, so a tie
is a reason to keep the honest variant, not the flattering one. Allocation error
and depth spread were **reported but explicitly not used to select**, because the
supervised arm is handed `OP_TARGET_DEPTH` and wins those columns by construction.

**And the finding that matters more than the decision.** Under `pure_act` the
per-operation depths genuinely differ (spread 0.452), but their correlation with
the curriculum is `+0.608 / −0.428 / +0.431` across seeds 42/43/44 — **the sign
flips.** Unsupervised ACT learns a non-trivial but essentially *arbitrary* depth
partition, not the operation-complexity curriculum. This is **Outcome C for the
adaptive-computation claim** (`CLAUDE.md` §8) and must be reported as such. It is
not to be fixed by switching supervision on; if supervision appears in the paper
it is a labelled ablation whose purpose is to demonstrate exactly this gap
(`r = +0.957` when the answer is supplied, `+0.204` when it is not).

Note the diagnostic gap this exposed: the pre-registered rule 4 tested depth
*spread*, which is necessary but not sufficient — a model can differentiate depth
strongly and still point it at the wrong operations. The correlation was therefore
added to the driver **after** the runs and is labelled as such in-source. It does
not change the decision, which the primary criterion and the tie rule had already
settled.

Earlier uncontrolled 1-epoch figures (`pure_act` 0.072163 / depth 4.756 / error
2.718 vs `supervised` 0.074127 / 2.206 / 0.729) are superseded by the table above
and were never publishable: one epoch, one unseeded run each.

**MoE's halt and depth columns are `N/A` by construction.** At `max_depth = 1`
every token is a forced exit at step 1 and the ponder cost is `(1 + 1) / 2 = 1.0`
for every token, always. Tabled beside MoR's `0.274614` that constant would read
as "MoE ponders hardest" — the same sentinel-as-measurement failure as T6.4. The
gate is in `engine.py` where `metrics.json` is assembled.

**Diagnostic to check first when depth looks wrong:** `halt/mean_halt_mass` must
be `1.0` to ~1e-7. Below 1 means the step weights stopped summing to one, the
output is a shrunken state, and depth is silently rescaling activations into the
task loss.

---

## 5b. The balance objective: what the number means

Read this before interpreting, tuning, or tabling any `routing_balance` number.

**The formula, per router call** (`MoEBlock.forward`, `code/more/model.py`):

```text
avg_probs   = mean over tokens of softmax(router logits)     # soft
f_e         = fraction of tokens whose ARGMAX is expert e    # hard, detached
P_e         = avg_probs[e]

entropy_term = -Σ_e avg_probs[e] · log avg_probs[e]     # in [0, log E]
switch_aux   = E · Σ_e f_e.detach() · P_e               # in [1, E]

balance_loss = -entropy_term + switch_aux                # minimised
```

Phase 4 did **not** change this formula. It changed only how the per-call values
are combined, how they are reported, and the coefficient.

**Aggregation — the T4.1 fix.** The per-call value is a **mean**, twice: mean over
the depth calls a wrapper actually made, then mean over blocks.
`metrics.json` stamps `routing_balance_normalization = "mean over depth calls per
block, then mean over blocks"` so no reader has to guess. The divisor is the
*realised* call count, not `max_depth`, because under ACT the loop breaks early
once every token has halted. Summed — the pre-Phase-4 behaviour — a 2-block
7-depth model produced up to 14 copies of a term a 1-block 1-depth model produced
once, **for identical routing behaviour**, so every depth and block-count
comparison was confounded and the auxiliary silently outweighed the task loss.
Measured invariance after the fix: `max_depth` 3 vs 7 differ by **0.180%**; the
unnormalized sums differ by **132.9%**.

**The number has a large constant offset, and the offset carries no gradient.**
At its own optimum — perfectly uniform routing — the objective is not 0 but

```text
-log(E) + 1  =  -1.791759 + 1  =  -0.791759      (E = 6)
```

so a *good* run reports a balance loss near −0.79, and full collapse reports
`+E = +6`. Two consequences that have already caused confusion:

- **A negative balance loss is not a bug.** It is the expected sign.
- **Most of the reported magnitude is that constant.** At the measured
  `-0.6534` the informative part — the distance from the optimum — is `0.1384`,
  and the other `0.7918` is a fixed offset whose gradient is ~0 (measured
  `|grad| ≈ 1e-8` at uniform routing). So a raw value-to-task ratio *overstates*
  the term's actual influence on learning. Weight selection below uses the raw
  ratio anyway, because it is the conservative bound.

**Reporting is split, and must stay split** (`CLAUDE.md` §4). `train/entropy_term`
and `train/switch_aux_term` are logged separately, plus
`train/entropy_term_normalized` = `H / log E`. The two halves pull in opposite
directions — entropy up is good, switch-aux up is bad — so the fused
`routing_balance_loss` can sit flat while both components drift in tandem. If a
balance number looks static, read the two components before concluding nothing
is happening.

**`E = 1` makes the whole objective a constant.** Entropy of a one-element
distribution is 0 and the Switch term is exactly `1 · 1 · 1 = 1`, so MoR reports
`bal = 1.0000` on every batch forever. MoR's weight is already `0.0`, so it never
enters the objective. `engine.py` therefore writes `"N/A"` — not `1.0` — for all
six balance keys when `num_experts <= 1`, because tabled next to MoRE's `-0.65`
that constant reads as "MoR is maximally imbalanced". Same sentinel-as-measurement
rule as the halting columns in §5a.

**Two measured limitations, recorded in §10's defect list:** normalized entropy
near 1.0 does not mean balanced load (entropy is computed on the mean softmax and
is blind to the argmax that actually dispatches), and a fully saturated router has
numerically zero balance gradient.

**Which load diagnostic to read beside entropy.** There are three different
"entropy-ish" numbers and they are not interchangeable:

| Key | Computed on | Use |
|---|---|---|
| `train/entropy_term_normalized` | mean **softmax**, `H / log E` | the loss term itself; what the gradient sees |
| `train/expert_load_entropy_normalized` | **hard argmax** counts over the epoch | whether real dispatch was balanced |
| `dispatch/max_load_fraction` | hard load, **peak over every (batch, block, depth) step** | capacity/overflow pressure only |

`max_load_fraction` is a deliberate worst-case peak, so in any ACT run it
saturates at **1.0** and stays there — with `halt/early_exit_rate ≈ 0.999`, the
last depth steps have a handful of active tokens and one expert trivially takes
all of them. Measured `1.0` in both Phase 4 runs. It is therefore **not** the
epoch-level balance companion; use `expert_load/expert_*_pct` and
`train/expert_load_entropy_normalized` for that. `max_load_fraction` *is*
meaningful in a single synthetic forward pass with a fixed active set, which is
how the Gate 3 check uses it (`0.8083` against a uniform `0.1667` while soft
entropy read `0.9988`).

**Where to change things.** Formula and per-call `route_stats`: `MoEBlock.forward`.
Depth-axis normalization: end of `MoREWrapper.forward`. Block-axis normalization:
end of `MoREModel.forward`. Reporting, ratio and `N/A` gating: `code/more/engine.py`.
Coefficient: `loss_weights.routing_balance` in `code/config.json`, overridable per
run with `--routing_balance` (which **refuses** `num_experts <= 1` and refuses
negative values, since a negative coefficient would minimise entropy and drive
the very collapse the term exists to prevent). Gate: `code/test_phase4_balance.py`.

---

## 6. Metric rules that the code enforces

`CLAUDE.md` §4 in full; the mechanics you will trip over:

**No sentinel may be reported as a measurement.** Not `-1`, not `0.0`, not a
`nan`. When a quantity does not exist, the pipeline emits the string `N/A` — in
the console line, in `results.tsv`, and in `metrics.json`.

Three quantities do not exist for MoR (`E = 1`), and each one previously
published a sentinel that would have ranked MoR as *bad at routing* when it
performs no routing at all:

| quantity | pre-fix MoR output | now | why it does not exist |
|---|---|---|---|
| normalized entropy | `-0.000` | `N/A` | `H/log(E)`, and `log(1) = 0` |
| max/mean pairwise cosine | `-1.000` | `N/A` | there is no *pair* |
| routing accuracy | `0.1005` | `N/A` | argmax is 0 for every token, so the diagonal fraction measures "what fraction of tokens happen to be E1" |
| per-family recall | `E1=1.0, E2..E6=0.0` | omitted | artifact of the degenerate 1×1 confusion shape |

Depth metrics are **not** gated for MoR. MoR does allocate depth, and that is
exactly its contribution.

**Routing accuracy has one implementation.**
`metrics.routing_accuracy_from_confusion(confusion, num_experts)`. It is the
confusion diagonal fraction *by construction*, so the reported scalar and the
reported matrix cannot disagree. Before this existed there were two independent
copies — one in `metrics.evaluate_paper_metrics`, one inlined in `engine.py` —
and **only the engine's ran**, so a guard added to the other had no effect on any
run. `evaluate_paper_metrics` is retained but marked UNUSED in-source.
**If you are fixing a routing metric, fix the engine's validation loop.**

**Other standing rules.** Never report `total_loss` as the quality metric —
primary is **validation task loss**. Entropy is a load-balance diagnostic, never
evidence of specialization; the void old criterion `expert_entropy > baseline`
must not return. Low cosine similarity is "consistent with differentiated
parameterizations", not proof of orthogonality. Call depth error *depth
allocation error*, never "compute efficiency". Never claim the model "discovers
intrinsic mathematical complexity" — it *allocates depth in accordance with the
predefined operation-complexity curriculum*.

---

## 7. Gate status

### Gate 5 — CLOSED: **`test_gate5.py` 34/34 PASS**

`python code/test_gate5.py`, run after `code/test_phase6_provenance.py` (which
produces the run directory it reads). The six criteria are re-asserted in one
file against a live model **and** a real run directory, so "Gate 5 passes" is a
statement about the code as it stands now rather than about six suites that
passed at different times.

| criterion | how it is asserted | evidence |
|---|---|---|
| valid provenance | 21 fields present and non-None; declared seed == resolved seed; determinism as two distinct fields; `variant` non-empty | `missing: []` |
| valid metrics | 10 loss components separate; `val/task_loss` primary and ≠ `train/total_loss`; entropy normalized to `[0,1]`; cosine mean **and** max; `depth/allocation_error_abs`+`_rel` and **no key containing "efficiency"**; early **and** forced exit rate; full permutation-invariant key set; `pim.raw_accuracy == val/routing_accuracy` < 1e-6 | all pass |
| correct dimensions | `cls_head`, `step_cls_head`, `router`, expert list, halt-head list all measured `== num_experts == 6`; op embedding `== NUM_OP_TYPES == 16 ≠ 6`; label width == reported families; `ffn_mult` agrees with provenance | all pass |
| routing assertion | `routing_mode == top1_sparse` in model *and* metrics; measured `dispatch/evals_per_token == 1.0`; noise `none` / scale `0.0`; all 64×7 indices in `[0,6)` | all pass |
| halt-gradient assertion | per-expert gradient from the task loss, **scoped to experts that received tokens**, plus a coverage assertion `len(used) == 6`, plus `ponder_cost.backward()` alone reaching every active halt head | all pass |
| no sentinel metrics | no metric `== -1`; none NaN; undefined entries are `"N/A"`; a single-expert confusion returns `None` for **every** routing quantity | all pass |

**Why the halt-gradient check is scoped.** Under Top-1 dispatch an expert that
no token selected is never called, so its halt head having `grad is None` is
arithmetic, not a defect — the first version of this check failed 2/12 because
the *test* was wrong. Hence: assert only on used experts, and assert coverage
separately so the check cannot go silently weak. Both halves are required; drop
the coverage assertion and it is vacuous, drop the scoping and it fails on
correct code.

#### T6.1 reproducibility: **61/61 PASS**

`python code/test_phase6_seeding.py`.

| claim | checks | evidence |
|---|---|---|
| every stream in `plan.md` §8.1 is reseeded | T6.1a | same seed reproduces `random`, `numpy`, torch CPU, torch CUDA and the DataLoader generator; **all five differ** at seed 43 |
| weight init is covered, not just data order | T6.1b | parameter digest `15a7bd18…` at seed 42 twice, `57ff38c9…` at 43 |
| a full train-mode step is bitwise reproducible | T6.1c | loss `1.1138908863067627`, gradient sum `1991.3999771339586`, 137 routing slots identical; different at 43 |
| determinism flags actually applied | T6.1d | every field **read back out of torch**, not echoed; `warn_only` mode; `notes` never an empty list |
| non-determinism seen **at runtime** is recorded, not just at seed time | T6.1d2 | five identical `does not have a deterministic implementation` warnings are recorded by op name and print **once**; `nondeterministic_ops_observed` never an empty list |
| DataLoader generator + worker | T6.1e | shuffle order reproducible, seed-sensitive, and **unchanged after 1000 global RNG draws are burned in between**; `seed_worker` picklable for Windows spawn |
| declared ≠ resolved seed, Gate 0 still armed | T6.1f | default 42 recorded as `resolved_seed` only; `provenance.seed` stays `null`; `ProxyGuardError` still raised on a canonical claim |
| one frozen seed set | T6.1g | `CANONICAL_SEED_SET` == `canonical_spec.json:seed_set` == 42–46 |
| **end-to-end** two runs, same seed | T6.1h | first-epoch row byte-identical: `1  0.134741  0.062896  0.7943  2.21  0.0030  0.0309  0.0241`; seed 43 gives `1  0.137417  0.049722  0.8968  2.48  -0.0024  0.0231  0.0964` |

**Why every check has a paired "seed 43 differs".** A seeding function that seeds
nothing at all also produces identical reruns on a deterministic machine. The
non-vacuity half is what proves the seed is doing the work.

**Why the test sets `log_interval = 1`.** The engine validates only when
`epoch % log_interval == 0 or epoch == epochs`. At the config default of 10, the
first-epoch row carries `val_loss = nan` and `routing_accuracy = N/A`, and
"identical first-epoch losses" would be comparing two placeholders.

**Two determinism fields, not one — they answer different questions.**
`determinism_exceptions` is what could not be *configured*, known at seed time.
`nondeterministic_ops_observed` is what actually *ran* non-deterministically,
knowable only after the run. The first real CLI run recorded `["none"]` in the
former while `stdout.log` held 281 `_histc_cuda` warnings; that gap is now closed
and the warnings are deduplicated to one line per op. Measured on
`runs/t61_nondet_check__seed42__42b858b3`: 1 line instead of 281,
`nondeterministic_ops_observed = ["_histc_cuda"]`, epoch row unchanged. The op
comes from `wandb.watch`'s gradient histograms — logging, not the loss path.

#### T6.2 / T6.3 routing metrics: **42/42 PASS**

`python code/test_phase6_routing_metrics.py`. An expert index carries no semantic
identity, so a model that discovered all six families but numbered them
differently would score `raw_accuracy = 0.0`. Permutation-invariant metrics are
therefore mandatory, not decorative.

| claim | evidence |
|---|---|
| the `plan.md` §8.3 criterion | a cyclic permutation of six experts gives `hungarian_accuracy = 1.0` against `raw_accuracy = 0.0`, recovered assignment `[1,2,3,4,5,0]`; AMI and purity bit-identical to the identity control |
| the exact assignment is exact | subset DP (`O(E²·2^E)`) matches exhaustive enumeration over all `n!` permutations on 40 random matrices, max \|Δ\| < 1e-9; **refuses** `n > 15` rather than degrading to greedy |
| the AMI chance correction is right | closed-form EMI matches literal enumeration of the permutation model over 720 permutations, agreement 1e-10 |
| chance-level input is not flattered | independent (product) table: MI exactly 0, AMI ≤ 0, while purity stays ≈ 1/E — which is why AMI and purity are always reported together |
| collapse is not flattered | all tokens to one expert: AMI 0, `hungarian = purity = 1/E` |
| per-family figures | hand-built imbalanced table: recall 8/10, precision 8/14, F1 harmonic mean, support, `N/A` for zero support, and `macro_recall = 0.75 ≠ raw = 22/30` (deliberately unequal, so the macro average cannot pass by coincidence) |
| **T6.2** authoritative accuracy | over 200 random token streams the confusion diagonal fraction, the engine's `mean(pred == oracle)` and `pim.raw_accuracy` differ by **max 0.000e+00** — one number, not three that agree |
| label hygiene | no expert label contains `/` |

**How to read raw vs. matched — this decides what the paper may claim.** The pair
is stored in `metrics.json` with a `note` saying the same thing:

| observed | means |
|---|---|
| raw ≈ matched | the **index itself is supervised** (`loss_weights.step_routing = 0.5`, the canonical setting). Expected, not a finding. |
| raw ≪ matched | a real partition exists under relabelling. The only case supporting an unsupervised specialization claim. |
| both low, AMI ≈ 0 | no partition. Purity may still look respectable at ≈ 1/E because purity is **not** chance-corrected. |

Measured on `runs/t63_pim_check__seed42__b19853cd`:
`route_acc=1.0000 route_hung=1.0000 route_ami=1.0000`, 24 clean `val/routing_*`
W&B keys, and train/val loss **byte-identical** to the pre-T6.3 run
(0.081314 / 0.073893) — the metric layer observes without perturbing training.

`more_env` has neither scipy nor sklearn, so both algorithms live in
[code/more/metrics.py](code/more/metrics.py) and are checked against brute force
rather than against a library.

#### T6.6 / T6.7 naming and provenance: **32/32 PASS**

`python code/test_phase6_provenance.py`.

| claim | evidence |
|---|---|
| canonical names are generated, not typed | `canonical_run_name` gives exactly `phaseB_MoE_seed42` / `phaseB_MoR_seed42` / `phaseB_MoRE_seed42`, and matches `canonical_spec.json:run_name_template` formatted from the same `ARCH_DISPLAY` map |
| a bad architecture cannot produce a name | unknown arch **raises** |
| an undeclared seed is not claimed | `canonical_run_name("more", None)` → `phaseB_MoRE` |
| stamping is safe | idempotent (no `_seed43_seed43`); **extends** a custom `--run_name` (`t63_pim_check` → `t63_pim_check_seed42`); no-ops when no seed is declared |
| `variant` cannot be omitted | derived by `resolve_variant`: canonical / dense / noise / fixed-depth / supervised; two deviations compose to `fixed_depth+router_noise_gaussian`; a curriculum flag at **zero weight** is correctly not a variant |
| `config_hash` identifies the configuration | two seeds of one experiment **share** a hash (so the exporter can average them); an `lr` or `ffn_mult` change does not |
| a real run carries every §9 field | `--epochs 1 --seed 44 --run_name t67_provenance_check` exits 0; all 27 fields resolved from `resolved_config.json` → `missing/None: []` |
| the refactor is behaviour-identical | `total_params == 6,358,553`, unchanged by threading `ffn_mult` |
| the seed reached the RNGs | `seed == resolved_seed == 44` |

**Two latent defects fixed here, both of which would have surfaced only in the
final matrix.** (a) `config_hash` included the `logging` block, and the run name
lives there — so two seeds of one experiment hashed differently and the exporter,
which by rule refuses to mix hashes, would have refused to average them.
`config_hash` now strips `logging` as well as `provenance`. (b) `_allocate_id`
appended the seed a second time once T6.6 put it in the run name
(`..._seed44__seed44__c3571d3a`); now suppressed when the base already carries it.
The test picks the run directory by **mtime**, not alphabetically — alphabetical
ordering selected the stale doubled directory and would have kept passing.

### Gate 4 — dimensions and config precedence: **62/62 PASS**

`python code/test_phase5_dimensions.py`. Guards a **silent** defect class: a
7-wide head over 6 families raises nothing, it just normalises softmax over a
logit no family can occupy. Every width-sensitive check therefore runs at
**E = 1, 6 and 7** — 7 is tested *alongside* 6 because anything hard-coded to 7 is
accidentally correct there, and only the sweep tells "derives its width" apart from
"happens to match". E = 1 doubles as the MoR baseline, where routing does not
exist.

| Criterion (`plan.md` §7) | Checks | Measured |
|---|---|---|
| no hard-coded 7 (**T5.1**) | G4.1–G4.7b, G4.39 | heads, router, expert list and halt heads all width `E` at E = 1/6/7; `step_cls_out` reshapes `(4,3,E) → (12,E)`, so a 7-wide *view* can no longer fold step tokens into each other's logits; `apply_architecture` stamps E = 6/1/6 for moe/mor/more and the built model agrees |
| matrix and scalar cannot disagree | G4.13–G4.16 | diagonal fraction = `mean(pred == oracle)` = reported scalar to 1e-9 in float64; per-expert recall keys number 0 at E = 1 and E at E = 6/7; `E = 1` entropy returns `None` → `N/A`, never `0.0` |
| no silent fallback (**T5.2**) | G4.17–G4.21 | 16 ops → experts {0..5}, no catch-all; `_EXPERT_FALLBACK is None`; unknown op and unknown family label both **raise**, naming the offender; `MIXED` and legacy `E7` map to −1 |
| labels from one manifest (**T5.5**) | G4.8–G4.12 | `expert_labels(E)` returns exactly E labels, raises at E = 0, equals `EXPERT_FAMILY_LABELS` at 6; no label up to E = 9 contains `/` or a space; confusion figures carry exactly E ticks per axis |
| config precedence explicit (**T5.3**) | G4.22–G4.28 | `data`-only 0.5 → 0.5/0.5; `training`-only 0.25 → 0.25/0.25; neither → 1.0/1.0; equal duplication accepted; **0.3 vs 0.7 raises** instead of picking a winner; CLI `--epochs/--batch_size/--seed` reach the resolved config and `--seed` lands in both `subset_seed` slots |
| router noise off in canonical (**T5.4**) | G4.29–G4.38 | canonical `state_dict` has **zero** noise keys against exactly one per block in the trainable ablation; canonical dispatch identical for 64/64 tokens across two training-mode passes, noisy variant moved 9/64; unknown mode raises; the Gate 0 guard refuses a canonical claim for `router_noise="trainable"` or `routing_mode="dense_blend"` |

### Gate 3 — balance objective: **27/27 PASS**

`python code/test_phase4_balance.py`. Guards a **magnitude** defect, which is why
every check is a numeric comparison rather than a "does it run" test — the
pre-Phase-4 code ran fine and simply optimised the wrong thing 146× too hard.

| Criterion (`plan.md` §6.3) | Checks | Measured |
|---|---|---|
| balance loss depth-invariant | T4.1a–g | `max_depth` 3 vs 7 differ **0.180 %**; the unnormalized sum differs 132.9 % on identical routing; sum ratio 2.329 ≈ 7/3 confirms the fix is load-bearing; aggregate lies inside `[min, max]` of its 14 per-call values, which a sum cannot |
| gradient direction correct | T4.3a–g | collapsed router: entropy `0.820 → 1.790` (ceiling `log 6 = 1.792`), Switch-aux `4.804 → 1.059` (floor 1.0), hard peak load `1.0000 → 0.8083`; uniform router sits exactly on `−log(6)+1 = −0.791759` with `\|grad\| ≈ 1e-8` |
| task loss dominates | G3.1, T4.2a–h | `train/balance_to_task_ratio = 0.017323` read out of a real run's `metrics.json`; the two halves are logged separately; the coefficient is frozen identically in `config.json` and the `config.py` default |
| artifact honesty | G3.2–G3.5 | decomposition keys and `routing_balance_normalization` present in `metrics.json`; `H / log E` inside `[0, 1]`; a single-expert run writes `"N/A"` for all six balance keys |

Two limitations are asserted as *passing* checks rather than hidden: T4.3d (soft
entropy `0.9988` beside hard `max_load_fraction 0.8083`) and T4.3g (a saturated
router has numerically zero balance gradient). See §5b.

### Gate 2 — real ACT halting: **31/31 PASS**

`python code/test_phase3_halting.py`. Every check is a measurement of gradient
flow or of tensor values in the running model, never an inspection of source —
because the defect it guards against is invisible from the loss curve, the depth
histogram and the exit-rate metric alike.

| check | result |
|---|---|
| T3.1a step weights sum to 1 per token | **PASS** — mean halt mass `1.000000000` |
| T3.1b halted tokens evicted, never recomputed | **PASS** — active rows per depth `[28, 28, 16, 5, 1]`, measured with a forward pre-hook |
| T3.1c every real token exits exactly once | **PASS** — min = max = 1.0 over 28 tokens |
| T3.1d forced exit reachable | **PASS** — halt bias −20 → `forced=28 early=0` |
| T3.1e early exit reachable | **PASS** — halt bias +20 → `forced=0 early=28` |
| T3.1f remainder is `1 − c_t` | **PASS** — exactly `1.000000000` at a depth-1 exit |
| T3.1g pads never enter the depth loop | **PASS** — 12 pads, **0** exits |
| T3.2a–c ponder cost + `N + R` differentiable | **PASS** — `grad_fn = DivBackward0` / `AddBackward0` |
| T3.2d–e normalized, depth-invariant magnitude | **PASS** — `0.480394` at depth 5; depth 3 → `0.720968`, depth 9 → `0.303979`, all in (0, 1] |
| T3.3b target depth not a function of expert index | **PASS** — E6 spans `[2, 4]`; depths 1 and 2 shared across experts |
| T3.3e depth error is absolute | **PASS** — +3 and −3 give `3.0`, not `0.0` |
| T3.3f undefined → `None` → `"N/A"` | **PASS** |
| T3.3g–i mode recorded; MoE refuses supervision | **PASS** — `pure_act` vs `supervised_curriculum` in provenance |
| **T3.4a every halt param gets task-loss gradient** | **PASS** — **24/24** on a batch where all six experts are dispatched (288 tokens), total \|grad\| `0.331801` (pre-fix: `None` on all 24) |
| T3.4a2 a dispatched expert's halt head is on the graph | **PASS** — halt heads are **per expert**, so under top-1 routing an expert no token selected has no graph node at all; that is sparsity, not a severed path |
| T3.4a3 no gradient for an undispatched expert | **PASS** — live sets are subsets of the dispatch sets, read from `MoEBlock`'s own `expert_idx` via a forward hook |
| T3.4b non-vacuity: frozen heads get none | **PASS** — 0/24 |
| T3.4c ponder cost alone also reaches them | **PASS** — 24/24 on the same fully-dispatched batch |
| GATE2a–c known-target synthetic case, both directions | **PASS** — SORT (target 4) `2.0474 → 2.0618` up, loss `0.105904 → 0.104354`; ADD (target 1) `6.7629 → 2.2913` down |
| GATE2d–e ponder and supervision logged separately; `N/A` where undefined | **PASS** |

**The measurement worth remembering:** before this gate,
`halt_loss.requires_grad = False`, `grad_fn = None`, and **`grad is None` on all
24 `expert_halt_heads` parameters** after backward through the full objective. No
result predating Phase 3 may be described as learned adaptive depth.

**Why T3.4 asks for a fully-dispatched batch.** The halt heads are per expert and
canonical routing is top-1 sparse, so on a small batch some expert simply receives
no token and its head correctly has no gradient. The original criterion demanded
all 24 be live on an arbitrary batch, which is an assertion that dispatch is
*dense*; it passed only because the pre-**T5.1** 7-wide `cls_head` consumed a
different slice of the init RNG stream and that router init happened to reach
every expert. Measured after the fix at B = 4, S = 3, the number of expert slots
left unused across the two blocks is 4, 3, 3, 1, 3, 1 for seeds 0–5. If T3.4a ever
fails again, check the dispatch sets first: a starved expert is sparsity, an
all-`None` gradient set is the Phase-3 defect returning.

### Gate 1 — input integrity: **8/8 PASS**

`python code/audit_leakage.py`. Mutation-based, with non-vacuity assertions so a
check cannot pass by testing nothing:

| check | result |
|---|---|
| A. target absent from input (perturb results → features unchanged) | **PASS** — 368/400 targets changed (non-vacuous), **0/400** feature changes |
| B. oracle expert absent (permute op→expert → features unchanged) | **PASS** — 400/400 `step_experts` changed, **0/400** feature changes |
| B2. no low-cardinality slot maps 1:1 onto the expert label | **PASS** — offending slots `[]`, majority rate 0.3026 |
| C. features bit-identical across E = 1 / 5 / 6 | **PASS** — 200 records × 3, slot-0 range `[-1,1]` for all, 0 mismatches |
| D. no program overlap across splits | **PASS** — `test|train 0, test|val 0, train|val 0` |
| E. copy baseline no longer free | **PASS** — copy `0.081853` ≥ predict-mean `0.080914` |

**The MSE floor.** predict-zero `0.081853`, predict-train-mean `0.080914`,
best-single-feature-copy `0.081853`, val target variance `0.080899`. Any
val loss must be read against `0.080914`, not against zero.

Honest residual diagnostic: the target coincides with one of the arguments in
33.45% of val records — inherent to MAX/MIN/MEDIAN/SORT, where the answer *is*
an input. Worst single slot 16.06%. Pre-fix the worst slot was 74.15% and the
copy baseline was exactly `0.000000`.

### Gate 0 — the proxy guard

`code/canonical_spec.json` is the single frozen definition of canonical. A run
may declare `experiment_group = canonical_phase_b` only if it matches that spec
exactly, uses the full dataset, and names a seed from `{42,43,44,45,46}`.

The guard reads **`enforced_fields`**, not the top-level mirror. While any
`enforced_fields` entry is `null` the guard **refuses every canonical claim**.
Still null by design: `d_model`, `lr`, `weight_decay`, `routing_balance`, and
`architecture_variants.{mor,more}.max_depth`. **That refusal is the intended
behaviour, not a bug** — those values are not yet chosen.

---

## 8. Reproducibility

Per-run output only: `runs/<experiment_id>/` with `config.json`,
`resolved_config.json`, `metrics.json`, `results.tsv`, `checkpoint.pt`,
`stdout.log`. Enforced by `code/more/run_context.py`. **Never** write global
`best_model.pt`, `results.tsv`, `final_run_metrics.json`, or `*_results.csv` —
that is how the `archive/` results became unattributable.

Note the location: `runs/` is at the **repo root**, not under `code/`.

Seeds `42, 43, 44, 45, 46`; report **mean ± std**. One favourable seed is never
evidence.

**Global seeding: done (T6.1), in `code/more/seeding.py`.** `engine.train()` calls
`apply_seeding(ctx.resolved_cfg)` as its first action — before the dataset, the
model or the optimiser exist, because anything constructed above that line draws
from an unseeded RNG. Covered: `random`, `numpy`, torch CPU, all CUDA devices, the
DataLoader shuffle generator (its own sub-stream, so batch order does not move
when the number of init draws changes) and `worker_init_fn`. Measured: two real
runs at seed 42 give a byte-identical first-epoch results row; seed 43 differs.

Three fields, three different questions, all in `provenance`:

| field | meaning |
|---|---|
| `seed` | the **declared** seed (`--seed`). `null` when none was passed. This is what Gate 0 tests. |
| `resolved_seed` | the integer the RNGs actually received. Never null. |
| `seed_source` | `cli` \| `config` \| `default` |

A run with no `--seed` is still bit-reproducible under the documented default 42,
but `seed` stays `null` on purpose: back-filling it would silently let such a run
pass the canonical check. Run IDs read `seedNA` for exactly that case, and
`__seed42__` once a seed is declared.

Determinism is **`warn_only`**: `cudnn.deterministic` on, `cudnn.benchmark` off,
`torch.use_deterministic_algorithms(True, warn_only=True)`,
`CUBLAS_WORKSPACE_CONFIG=:4096:8` exported before CUDA init. The top-1 dispatch
path uses scatter/index writes with no deterministic CUDA kernel in torch 2.5, so
`plan.md` §8.1's instruction applies: document the exception instead of pretending
exact determinism. Two provenance fields carry that documentation, and they are
not interchangeable:

| field | means | known when |
|---|---|---|
| `determinism_exceptions` | what could not be **configured** (e.g. cuBLAS workspace unset after CUDA init, CUDA absent) | at seed time |
| `nondeterministic_ops_observed` | ops that actually **ran** without a deterministic implementation, by name | only at end of run |

Both read `["none"]` rather than `[]` when empty. Warnings also appear in
`stdout.log`, deduplicated to one line per op — before that dedupe a 1-epoch run
emitted 281 identical `_histc_cuda` lines while provenance still claimed `["none"]`.
Set `training.deterministic_strict = true` to make them fatal instead.

W&B metric keys must not contain `/` inside a label — use `E1_ADD_SUB`, not
`E1 ADD/SUB`. Run names: `phaseB_MoE_seed42`, `phaseB_MoR_seed42`,
`phaseB_MoRE_seed42`.

---

## 9. Environment

```
C:\Users\Hp\anaconda3\envs\more_env\python.exe     torch 2.5.1 + CUDA
```

Anaconda base and `C:\Python314` have **no torch**. GPU is an RTX 4060 Laptop,
8 GB. Shell is Git Bash on Windows; the working directory persists between calls,
so **use absolute paths**. The console is cp1252 — Unicode output needs a stream
reconfigure or it raises `UnicodeEncodeError`.

Quick checks, cheapest first:

```bash
"C:/Users/Hp/anaconda3/envs/more_env/python.exe" code/smoke_test.py
```

```bash
"C:/Users/Hp/anaconda3/envs/more_env/python.exe" code/verify_pipeline.py
```

```bash
"C:/Users/Hp/anaconda3/envs/more_env/python.exe" code/audit_leakage.py
```

Expected: `ALL TESTS PASSED`, `35/35 checks passed`, `Gate 1: 8/8 checks passed`.

**Known editing hazard.** Do not put `\t` or `\n` literals inside a Python patch
delivered through a Bash heredoc — the backslashes are collapsed before Python
sees them, so `assert old in s` fails on exactly those patterns while others
match. Use the Edit tool for anything containing a backslash escape.

---

## 10. Where the work stands

**Refactor: complete.** 1615-line monolith → 9 modules, largest 594 lines. All
three launchers exit 0; `smoke_test.py` ALL PASSED; `verify_pipeline.py` 35/35.
The launchers are pure — `grep -nE "class |def |nn\.|torch\."` across all four
returns nothing.

**Phase 0: 11/11. Phase 1 / Gate 1: 6/6 + 8/8. Phase 2: 19/19 (T2.1–T2.4 done —
canonical routing is now true Top-1 sparse). Phase 3 / Gate 2: 31/31 (T3.1–T3.4
done — halting is now trained ACT, not a bool-tensor scalar). Phase 4 / Gate 3:
27/27 (T4.1–T4.3 done — the balance term is depth- and block-invariant, its two
halves are reported separately, and `routing_balance` is frozen at **0.001** from
measurement). Phase 5 / Gate 4: 62/62 (T5.1–T5.5 done — every tensor dimension
derives from `num_experts`, verified at E = 1/6/7; no catch-all expert; config
precedence explicit and conflicts raise; router noise off in canonical and absent
from its `state_dict`). **Phase 6 / Gate 5: CLOSED** — T6.1 61/61 (global
seeding, determinism recorded, reruns bitwise identical), T6.2 + T6.3 42/42
(routing accuracy is one computation shared with the confusion matrix, and
Hungarian / AMI / purity / per-family figures make it permutation-invariant),
T6.4 + T6.5 (no sentinel reported as a measurement), T6.6 + T6.7 32/32 (generated
run names, derived `variant`, and all 27 provenance fields non-None in a real CLI
run), Gate 5 itself 34/34.** Next: the controlled ACT decision experiment, then
freezing `canonical_spec.json`, then the final matrix. See TASKS.md for the
evidence behind each tick, and
[changelog.md](changelog.md) for how each was done and where to look when it
breaks.

**1-epoch reference numbers** (full dataset, unseeded, `WANDB_MODE=offline`,
post-Phase-2 sparse routing). These are sanity checks, **not results** — 1 epoch,
no seed control, so they may not enter any table:

| | params | val loss | route acc | entropy_norm | cos_max |
|---|---|---|---|---|---|
| MoE | 6,359,069 | 0.0752 | 0.8532 | 0.9891 | 0.0058 |
| MoR | 1,098,259 | 0.0778 | N/A | N/A | N/A |
| MoRE | 6,359,069 | 0.0752 | 0.5090 | 0.9994 | 0.0342 |
| MoRE dense ablation | 6,359,069 | 0.0779 | 0.2660 | — | — |

Floor: predict-train-mean `0.080914`.

**Do not read the route-acc column as a finding.** Unseeded 1-epoch MoRE runs gave
`0.5090`, `0.1689` and `0.2445`. That spread is larger than any difference the
paper would claim, which is exactly why seeds `42–46` with mean ± std are
mandatory. T6.1 (global seeding) is now done, so a seed-controlled rerun is
possible — but these particular numbers predate it and remain uncontrolled.

**Both tables predate Phase 5** and were produced with 7-wide `cls_head` /
`step_cls_head` over 6 families and with router exploration noise silently on in
training. Neither is the canonical architecture any more, so these numbers are now
only useful as "the code ran end to end" evidence; the route-acc and entropy
columns in particular were computed against a phantom seventh logit.

**Post-Phase-3 halting numbers** (same caveat — 1 epoch, unseeded, not results):

| | avg depth | early-exit rate | forced-exit rate | halt mass | ponder cost | depth alloc err |
|---|---|---|---|---|---|---|
| MoE | 1.000 | N/A | N/A | N/A | N/A | N/A |
| MoR | 2.000 | 0.99998 | 0.000025 | 1.0000000 | 0.274614 | 0.930909 |
| MoRE | 4.756 | 0.87586 | 0.124137 | 1.0000000 | 0.474371 | 2.718111 |

MoE's `N/A`s are by construction (see §5a), not omissions.

**Primary metric key.** Validation task loss is logged as **`val/task_loss`**
(alias `val/loss`, kept for existing readers). It is `F.mse_loss` on the
regression head only — no auxiliary, balance, halting or routing-supervision term
is inside it. Build tables from that key, never from `train/total_loss`.

### Known defects — deliberately not papered over

- **Normalized entropy near 1.0 does not mean balanced load.** `entropy_term` is
  computed on the *mean softmax*; the argmax that actually dispatches is invisible
  to it. Measured together in one Gate 3 synthetic check:
  `entropy_normalized = 0.9988` while hard `max_load_fraction = 0.8083` against a
  uniform `0.1667` — one expert taking 81% of the tokens, reported as
  near-perfect entropy. For epoch-level runs read `expert_load/expert_*_pct` and
  `train/expert_load_entropy_normalized`, **not** `dispatch/max_load_fraction`,
  which is a peak-over-all-steps capacity metric and pins to `1.0` in any ACT run
  (see the table in §5b). Neither entropy is evidence of specialization; §6.
- **The balance term cannot rescue a saturated router.** At full float32 softmax
  saturation the gradient is numerically zero (measured: entropy `1.879e-07`,
  `|grad| 2.352e-07`). It discourages collapse from a *soft* state; recovering
  from a hard one is an initialization or noise problem, not a loss-weight
  problem. The relevant knob is `model.router_noise`, which since **T5.4** is
  `none` in canonical and available as ablation D (`fixed_annealed`, `trainable`)
  — so this limitation may only be addressed by a *labelled* variant, never by
  quietly turning noise back on.
- **The balance coefficient trades directly against routing accuracy.** Measured
  at 1 epoch, full dataset, MoRE, everything else identical: `w = 0.05` gave soft
  entropy `0.9861` but `val/routing_accuracy` **0.9229**; `w = 0.001` gave entropy
  `0.6946` and routing accuracy **0.9979**. Pushing routing toward uniform makes
  the router *disagree with the oracle families more*. This is the concrete reason
  `CLAUDE.md` §2 forbids tuning toward maximal entropy and §4 voids the old
  `expert_entropy > baseline` criterion.
- **Entropy ≈ 1.0 means routing is near-uniform.** That is load balance, not
  specialization. Do not read it as a win.
- **Parameter mismatch** MoE/MoRE 6,359,069 vs MoR 1,098,259 — the comparison is
  not parameter-matched. **T9.2.**
- **Top-1 routing starves experts on small batches, and that is not a bug.** With
  12 tokens over 6 experts, some expert wins no argmax in some block, so its
  per-expert halt head has no gradient that step. Two consequences to remember
  when writing a test: never assert that all E halt heads are live on an arbitrary
  batch (that asserts *dense* dispatch), and read the dispatch decision from
  `MoEBlock`'s third output rather than inferring it. See the T3.4 note in §7.
- **Unsupervised ACT does not learn the operation-complexity curriculum.**
  `pure_act` MoRE shows depth allocation error **2.718** against a 1–4 target
  range, versus **0.729** with supervision on. This is a measured finding, not a
  bug, but it means whether the canonical matrix uses `pure_act` or
  `supervised_curriculum` is a live `canonical_spec.json` decision that has **not
  been made**. Both modes are implemented and provenance-labelled.
- **The `MoREModel.forward` tuple is 13 elements and has grown twice.** Index it
  positionally by slot number, never from the end — `res[-1]` broke two Phase 2
  checks when Phase 3 appended two fields.

### When a gate fails (`plan.md` §20)

1. **STOP.** 2. Report the failure with the actual output. 3. Identify the
cause. 4. Fix it. 5. Re-run the gate.

Never revert a correctness fix because the old number looked better. Never
proceed past a failing gate. Never tick a TASKS.md box without the stated
verification actually passing.

### Outcome honesty

Outcomes **A** (MoRE approaches baseline with stable specialization and adaptive
depth), **B** (not lowest-loss but a stable interpretable combination neither
component gives alone), and **C** (no meaningful advantage once the bugs are
fixed) are all scientifically valid, and none may be optimized toward.
**Outcome C must be reported honestly.** The objective is to learn what the
architecture does, not what we want it to do.
