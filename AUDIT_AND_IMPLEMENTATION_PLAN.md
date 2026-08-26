# MoRE — Repository Audit & Implementation Plan

**Scope:** measures the repository at commit `136da4e` against the finalization spec in `changes.md`.
**Method:** full read of `code/train.py`, `data/script.py`, all 9 configs, all runner scripts, `README.md`,
`rules.md`, `objective.md`, `cannonic.md`, plus 42 W&B run directories, plus six executed diagnostics
under `C:\Users\Hp\anaconda3\envs\more_env\python.exe` (torch 2.5.1 + CUDA).
**Every claim below is backed by a file:line citation or a reproducible measurement.** Where I could not
verify something, it says so.

---

## 0. Verdict

`changes.md` is written on the assumption that the codebase is *scientifically sound but sloppily
instrumented* — that the job is to fix metric names, add provenance, freeze a dataset, and re-run a clean
matrix. That assumption does not hold.

Six defects were confirmed empirically. Four of them mean **the current numbers do not measure what the
paper claims they measure**, so the canonical matrix in §26 cannot be started yet:

| # | Defect | Status | Consequence |
|---|--------|--------|-------------|
| **B1** | The regression target is present verbatim as an input feature in 100% of validation records | measured | The task is a copy task, not arithmetic. Every val-loss number in `cannonic.md` and `README.md` is void. |
| **B2** | `halt_loss.requires_grad == False`; every halting-head parameter receives `grad=None` | measured | Halting is **never trained**. "Adaptive computation" is untrained random initialization. |
| **B3** | Routing is a dense weighted blend of all experts, not top-1 | code read | No sparsity, no compute saving. Violates the repo's own `rules.md` §1. |
| **B4** | The load-balance term is accumulated per depth per block and is ~146× the task loss | measured | The training objective is dominated by "maximize entropy" ⇒ near-uniform router ⇒ below-chance routing. Also makes `total_loss` non-comparable across depths. |
| **B5** | `num_experts=6` (the canonical setting `changes.md` §1 mandates) is unsupported | measured | Hard-coded `7` in three places. A 6-expert run silently builds 7-class heads and a 7×7 confusion matrix. |
| **B6** | Input features differ between MoR (E=1) and MoRE (E=6/7) | measured | The headline MoR-vs-MoRE comparison is confounded at the data level, on top of a 6.6× parameter gap. |

Plus five instrumentation/provenance defects (**I1–I5**, §3) and three reporting violations already
committed in `cannonic.md`/`README.md` (**R1–R3**, §4).

**One item in `changes.md` is itself a misdiagnosis.** §2.1 asks you to reconcile `val/routing_accuracy`
with `val/step_routing_accuracy` and the confusion matrix. `val/step_routing_accuracy` does not exist
anywhere in the repository (`grep` returns nothing), and in the current code the logged accuracy and the
plotted matrix are computed from the *same* tensor in the *same* loop, so they agree by construction —
the assertion §2.1 demands is already trivially satisfied. The real routing problem is different and
worse; see **B4** and §5.2.

---

## 1. Blocking defects, with receipts

### B1 — 100% label leakage: the task is solved by copying one input feature

**Mechanism.** `data/script.py`'s `verify()` asserts that `example["output"] == steps[-1]["result"]`,
i.e. the generator *guarantees* the final step's result equals the regression target. The tokenizer at
[train.py:270-276](code/train.py:270) then writes each step's result into that step's feature row:

```python
numeric_vals = flat_args + [scalar_result]          # result appended to the features
numeric_vals = [max(-max_val, min(max_val, v)) / max_val for v in numeric_vals]
row[0] = expert_id / max(num_experts - 1, 1)
for i, v in enumerate(numeric_vals[: step_feat_dim - 1]):
    row[i + 1] = v                                  # ← target lands in row[1..11]
```

and the target itself is normalized identically at [train.py:283](code/train.py:283).

**Measurement (val split, n=5250).** Some feature slot at the last real step equals the target to
floating-point exactness in **5250 / 5250 = 100.0%** of records. The leak is *concentrated*, not diffuse:

```
slot  3: exact match with target in 3893/5250 = 74.15%    ← binary-op result lands here
slot  1: 17.18%   slot  4: 15.31%   slot  5: 13.73%   slot  6: 13.01%
slot  2: 11.37%   slot  7: 11.87%   slot 0/8/9/10/11: 5.4–5.9%
```

So a *linear readout of feature slot 3 at the last unmasked step* solves 74% of records exactly, and
some fixed-slot combination solves all of them.

**Why this voids the numbers.** Trivial predictors on the same normalized targets:

```
val target variance             = 0.079380
MSE(predict 0)                  = 0.080291
MSE(predict train mean)         = 0.079389
MSE(copy the leaked feature)    = 0.000000   ← achievable in closed form
```

Against those, the published 50-epoch results are:

| Model | reported val MSE | R² vs predict-mean | vs achievable floor |
|---|---|---|---|
| MoE | 0.001957 | 0.9753 | 0 |
| MoR | 0.003160 | 0.9602 | 0 |
| MoRE 1-block | 0.002320 | 0.9708 | 0 |
| MoRE 2-block | 0.002285 | 0.9712 | 0 |
| FixedDepth-5 | 0.001778 | 0.9776 | 0 |

Every model is *worse* than the closed-form copy. The 0.00178–0.00316 spread is not "arithmetic ability"
— it is how imperfectly each architecture approximates a copy, and it spans a factor of 1.8 on a task
whose floor is exactly zero. `README.md`'s headline "**87.26% reduction in validation loss**" is a
comparison between two copy-approximators across two different code versions.

**Fix.** Two options; pick (a).

- **(a) Withhold the answer.** In the tokenizer, drop the final element of `numeric_vals` (the step
  result) from the feature row — feed only `flat_args`. Additionally, for the *last* real step, mask the
  entire feature row's numeric slots, keeping only the op identity. Then the model must actually compute.
- **(b) Predict the next step's result** from the prefix (a genuine multi-step reasoning objective).
  Larger change; better science; more work.

**Acceptance test (must be added and must pass):** for every split, assert
`max over records of ( min over (step, slot) of |feature − target| ) > tol`, i.e. *no* record contains
its own target. This must be a hard `assert` in the dataset constructor, not a warning.

---

### B2 — Halting is never trained

**Code.** [train.py:591-596](code/train.py:591):

```python
stop_local = halt_probs > 0.5          # non-differentiable threshold
```

[train.py:609](code/train.py:609):

```python
total_halt_loss = total_halt_loss + active_mask.float().mean() * 0.05
```

`active_mask` is a boolean tensor; `.float().mean()` carries no gradient. No loss term anywhere touches
`halt_probs`.

**Measurement.** Under `more_env`, on a live model with all loss terms assembled:

```
halt_loss.requires_grad = False
halt_loss.grad_fn       = None
torch.autograd.grad(halt_loss, halt_params) → RuntimeError:
    element 0 of tensors does not require grad and does not have a grad_fn
```

and after a full `total_loss.backward()`, **every** parameter matching
`blocks.0.expert_halt_heads.*` has `grad is None`.

**Consequence.** The halting heads are random-init linear projections that receive no learning signal and
only decay under `weight_decay=1e-4` → their outputs drift toward 0 → `sigmoid(0)=0.5` → the `>0.5`
threshold becomes a coin flip that never reliably fires. Therefore:

- "Adaptive recursion depth" is **untrained noise**, not a learned policy.
- `README.md`'s "212% throughput acceleration due to healthy adaptive early exiting" attributes a speedup
  to a mechanism that does not learn.
- The op→depth curriculum story in `cannonic.md` §3.5 is unsupported.

**Corroborating evidence — two runs report opposite op→depth patterns:**

| Operation | run `20260823_153736` | `cannonic.md` §3.5 / `1block_results.csv` | generator's intent (`data/script.py`) |
|---|---|---|---|
| ADD/SUB (E1) | 5.00 (deep) | 1.036 (shallow) | 1–2 (shallow) |
| MULT/DIV (E2) | 1.17–1.21 (shallow) | 1.029 (shallow) | 2–3 |
| LOGIC (E4) | 4.80–4.87 (deep) | 6.677 (deepest) | 1–2 (shallow) |
| SHIFT (E5) | 5.00 (deep) | 6.563 (deep) | 1–2 (shallow) |
| SORT/STAT (E6) | 2.02 | 1.294 (shallow) | 3–5 (deep) |

Both runs are *anti-correlated with the intended curriculum*, and they contradict each other on E1. That
is the signature of an unlearned, initialization-dependent quantity.

**Fix.** Restore a real ACT objective — this is what `rules.md` §2 already requires ("standard Adaptive
Computation Time (ACT) or Universal Transformer halting logic… you must calculate and return a ponder
cost"). Concretely:

1. Accumulate a differentiable ponder cost: `ponder += halt_probs.sum()` (or the standard
   remainder-corrected ACT cost) over active tokens, so gradients reach the halt heads.
2. Keep the discrete `>0.5` for the *dispatch* mask, but weight each depth's contribution to the output
   by the (differentiable) cumulative continue-probability so the task loss also reaches the halt heads.
3. Only if `changes.md` §1's per-operation halting curriculum is desired as *supervision*, add a
   separate, explicitly-labeled `halting_supervision_loss` against the manifest's target depths — and log
   it as its own term per §5, never folded into `task`.

**Acceptance test:** after one backward pass, assert every parameter in `expert_halt_heads` has
`grad is not None` and `grad.abs().sum() > 0`. This is exactly the check
[verify_pipeline.py](code/verify_pipeline.py) currently *fails to make* — its "gradient flow verified"
step tests a standalone `nn.Linear`, not the MoRE model, which is why B2 survived 42 runs.

---

### B3 — Routing is dense, not top-1

[train.py:404-405](code/train.py:404):

```python
expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)   # ALL experts run
out = (router_probs.unsqueeze(-1) * expert_outputs).sum(dim=1)                # soft blend
```

`argmax` is computed at [train.py:400](code/train.py:400) but used only for logging, the balance loss,
and halt-head selection. So:

- There is **no sparsity** and **no compute saving** — the headline efficiency claim of a MoE has no
  mechanism behind it.
- The confusion matrix and `routing_accuracy` describe an `argmax` that **does not determine the forward
  computation**. With near-uniform `router_probs` (see B4), the argmax is nearly arbitrary while the
  actual output barely changes — which is precisely why routing accuracy can sit at 0.0119 while val loss
  improves monotonically.
- It violates `rules.md` §1, which declares Top-1 routing *immutable* and instructs refusal of requests
  to alter it.

`README.md` §1.1 documents this as an intentional fix ("hard boolean masking blocked gradients"). The
diagnosis was right; the remedy overshot. The standard remedy is top-1 dispatch **multiplied by its own
gate probability** — `out[mask] = p_top1[mask] * expert_k(x[mask])` — which is differentiable w.r.t. the
router *and* sparse. That is the Switch-Transformer formulation the repo's own `rules.md` names.

**Fix.** Implement top-1-with-gate-multiplier. Keep the dense path behind a config flag
`"routing_mode": "dense" | "top1"` so the dense variant can be reported as a *labeled ablation* per §29.

---

### B4 — The balance loss dominates the objective and is depth-coupled

**Code.** [train.py:428](code/train.py:428): `balance_loss = -entropy_term + switch_aux`.
[train.py:563](code/train.py:563): `total_bal_loss = total_bal_loss + b_loss` — **inside the recursion
depth loop**. [train.py:749](code/train.py:749): summed again across blocks.

So `total_bal_loss ≈ num_blocks × depth × (−entropy + switch_aux)`, and it is **unbounded below in the
number of recursion steps**.

**Measurement** — `code/final_run_metrics.json` (the FixedDepth-5 run, depth 5, 1 block):

```json
"train/total_loss":       -0.2371584673722585,
"train/task_loss":         0.001666332254759394,
"train/aux_routing_loss": -4.861128214078072,
"train/expert_load_entropy": 1.9205399751663208
```

Check: `0.05 × (−4.861) = −0.2431`, and `−0.2431 + 0.00167 + (halting/step terms) ≈ −0.2372` = the logged
total. So the balance term contributes **−0.243 against a task loss of 0.00167 — a magnitude ratio of
≈146×**. `5 × (−0.95) ≈ −4.86` confirms the per-depth accumulation exactly.

**Three consequences:**

1. **The optimizer's primary job is to maximize router entropy.** It succeeds: entropy is pinned at
   1.92–1.945 out of `log 7 = 1.9459` in *every* run. A near-maximal-entropy router is a *near-uniform*
   router — i.e. one that is barely routing. `README.md` §3 and `objective.md` read high entropy as
   success ("Prevent expert collapse: target `expert_entropy > baseline`"); combined with below-chance
   routing accuracy it is the opposite — evidence the router is uninformative.
2. **`total_loss` is not comparable across depth or block settings**, because deeper/wider configs
   accumulate a larger negative reward. This directly contaminates the fixed-depth ablation
   (`changes.md` §10) and the 1-block vs 2-block ablation (§11) if `total_loss` is ever compared.
   `changes.md` §5 is right that validation *task* loss must be the primary metric — this is why.
3. **A negative training loss is being reported as a loss.** `train/total_loss = −0.237` in the shipped
   metrics file.

**Fix.**
- Divide the accumulated balance loss by the number of `(block, depth)` calls so it is a **mean, not a
  sum** — making it depth-invariant.
- Re-tune `routing_balance` so `|weight × balance| ≲ 0.1 × task_loss` at convergence. At the current
  scale the setting `0.05` is not a regularizer.
- Report the entropy term and the Switch-aux term separately (`changes.md` §5 asks for exactly this).
- Report **normalized entropy `H / log E`** per `changes.md` §16 — mandatory, since existing tables
  compare `E=1` (H=0 by definition), `E=6`, and `E=7` on the same axis.

**Acceptance test (`changes.md` §6, extended).** Test *gradient direction*, not just the scalar:
- collapsed router (all mass on one expert): `entropy ≈ 0`, `switch_aux ≈ E`, and
  `∂balance/∂logits` must point *away* from collapse.
- uniform router: `entropy ≈ log E`, `switch_aux ≈ 1`, `balance` at its minimum.
- **new:** `balance_loss(depth=1) ≈ balance_loss(depth=5)` after the mean-normalization fix.

---

### B5 — `num_experts=6` (the canonical setting) is unsupported

`changes.md` §1 mandates 6 expert families with the E7 catch-all removed. The code cannot express that:

| Site | Code | Effect at `num_experts=6` |
|---|---|---|
| [train.py:698](code/train.py:698) | `self.cls_head = nn.Linear(d_model, 7)` | 7-way head, class 6 unreachable |
| [train.py:703](code/train.py:703) | `self.step_cls_head = nn.Linear(d_model, 7)` | same |
| [train.py:1267](code/train.py:1267) | `step_cls_out.reshape(-1, 7)[valid_mask]` | same |
| [train.py:1323](code/train.py:1323) | `confusion = torch.zeros(7, 7, ...)` | permanently empty row/col 6 |
| [train.py:163](code/train.py:163) | `EXPERT_FAMILY_LABELS` — 7 entries | label/index mismatch |
| [train.py:160](code/train.py:160) | `_EXPERT_FALLBACK = 6` | ops not in `OP_TO_EXPERT` map to a nonexistent expert |

**Measured:** constructing `MoREModel` with `num_experts=6` yields `cls_head.out_features == 7` and
`step_cls_head.out_features == 7`, silently, with no error.

Note the standalone evaluator at [train.py:852](code/train.py:852) *is* dynamic
(`torch.zeros(num_experts, num_experts)`) — so the two confusion matrices in the codebase disagree in
shape. This is a real inconsistency, and it is the *actual* version of the problem `changes.md` §2.1 was
groping toward.

**Also (B5b):** expert 6 has **zero oracle support** — no operation in `OP_TO_EXPERT` maps to it — while
the balance loss pushes toward `1/7` uniform load. The balance loss and the oracle CE are in direct
opposition on that expert. Removing E7 per §1 fixes this, but only once B5 is fixed.

**Fix.** Replace every literal `7` with `num_experts`; derive `EXPERT_FAMILY_LABELS` from a single
manifest; delete `_EXPERT_FALLBACK` and instead `raise` on an unmapped op.

---

### B6 — MoR and MoRE do not receive the same inputs

[train.py:274](code/train.py:274):

```python
row[0] = expert_id / max(num_experts - 1, 1)
```

**Measured slot-0 ranges:**

| Config | `num_experts` | slot-0 range |
|---|---|---|
| MoR | 1 | **[0, 5.0]** |
| MoRE canonical | 6 | [0, 1.0] |
| MoRE as-run | 7 | [0, 0.833] |

Two separate problems:

1. **The primary MoR-vs-MoRE comparison is confounded at the data level.** MoR sees an input feature on a
   5× wider scale than MoRE. `README.md` §3 attributes MoR's worse loss to "capacity constraints"; the
   input distribution also differs, and the parameter count differs 6.6× (see §2, I-params). Three
   confounds, one conclusion.
2. **Slot 0 *is* the oracle routing label.** The router receives the correct expert identity as an input
   feature. Any routing-accuracy result is therefore not a discovery — the answer is in the input. This
   is a stronger version of the "hidden oracle supervision" risk `changes.md` §30 names, and it persists
   even when `routing_supervision.enabled = false`.

**Fix.** Remove `expert_id` from the feature row entirely. Represent the operation by an embedding of the
*op token* (which is legitimate task input) and never the *family label* (which is the routing answer).
If op identity is also considered too informative, hold it out and let the model infer it. Either way the
encoding must be **identical across all architectures** — assert that the feature tensor for a fixed
record is bit-identical under `num_experts ∈ {1, 6, 7}`.

---

## 2. Instrumentation & provenance defects

### I1 — Config precedence silently discards `subset_fraction`

[train.py:101](code/train.py:101) and [train.py:126-127](code/train.py:126):

```python
cfg["training"].setdefault("subset_fraction", 1.0)     # line 101 — always creates the key
...
if "subset_fraction" in cfg["training"]:                # line 126 — therefore always True
    cfg["data"]["subset_fraction"] = cfg["training"]["subset_fraction"]
```

The `setdefault` guarantees the key exists, so the backward-compat shim **unconditionally overwrites**
`data.subset_fraction`. Every 50-epoch config on disk specifies `"subset_fraction": 0.5` under `data`;
every W&B run logs `subset_fraction: 1`. `run_remaining_tests.py:229` also sets
`cfg["data"]["subset_fraction"] = 0.5` and is likewise ignored.

**Net effect on the science:** *favourable but accidental.* The five 50-epoch runs used the **full**
training set (n=59,500), so they are mutually comparable in data volume — the `changes.md` §25 Rule-B
concern about proxy data does **not** apply to them. But the intent expressed in six config files was
silently discarded with no warning, and any future subset experiment configured this way will silently
run at full scale.

**Fix.** Delete the shim; make `data.subset_fraction` the single source of truth; `raise` on a
`training.subset_fraction` key with a message telling the user to move it.

### I2 — W&B config misreports epochs

`wandb.init(config={**mc, **tc, **lw, **dc})` at [train.py:1185](code/train.py:1185) logs `tc["epochs"]`,
but the actual loop uses `run_epochs` from `--epochs` ([train.py:1078](code/train.py:1078)).

**Measured:** run `20260823_165250` was launched with `['--config', 'config_2block_ablation.json',
'--epochs', '1']` and its `config.yaml` records `epochs: value: 50`.

This is exactly the "wrong/mixed W&B run" failure mode `changes.md` §30 lists. Any query filtering on
`epochs == 50` will pull 1-epoch runs.

**Fix.** Write `run_epochs` back into `cfg["training"]["epochs"]` *before* `wandb.init`. Better: build the
W&B config from the fully-resolved `cfg` after all overrides, and refuse to start if any CLI override is
not reflected.

### I3 — No global seeding; `changes.md` §21 is currently unexecutable

`grep -c "torch.manual_seed("` on `train.py` returns **0**. The only seeded generators are the
`random_split` calls at [train.py:1112](code/train.py:1112) and [train.py:1145](code/train.py:1145).
`--seed` at [train.py:1559](code/train.py:1559) sets only `subset_seed`. Model init, dropout, and the
router noise draw are all unseeded.

The §21 protocol (seeds 42–46, report mean ± std) cannot be run as specified. Given B2, this is not
cosmetic: with untrained halting heads, **seed variance is the dominant source of variation in every
depth metric** — which is the most plausible explanation for the contradictory op→depth tables in B2.

**Fix.** A `set_seed(seed)` covering `random`, `numpy`, `torch`, `torch.cuda`, plus
`torch.use_deterministic_algorithms(True)` and a seeded `DataLoader` `worker_init_fn` / `generator`. Log
the resolved seed.

### I4 — No provenance in `wandb.init`

The config passed to W&B is `{**mc, **tc, **lw, **dc}` — no `experiment_id`, `experiment_group`,
`architecture`, `variant`, `seed`, `dataset_version`, `train_split_version`, `code_git_commit`, or
`config_hash`. Every field in `changes.md` §22 is absent, and §23's `export_results.py` filter
(`experiment_group == "canonical_phase_b"`) has nothing to filter on.

### I5 — Two metric-corruption bugs

- **Sentinel published as a measurement.** [train.py:814](code/train.py:814) initializes
  `max_sim = -1.0` and returns it unchanged when `num_experts == 1` (no pairs to compare).
  `cannonic.md` §4 publishes `Max Cos Sim = -1.0000` for Nano/Micro/Mini MoR — three times — as if it
  were a measured cosine similarity. Cosine similarity of −1 would mean perfectly anti-parallel experts.
  Per `changes.md` §25 Rule D this must be `N/A`.
- **W&B key namespace collision.** `EXPERT_FAMILY_LABELS` contain `/`
  ([train.py:163](code/train.py:163)), so keys like `val/routing_recall/E1 ADD/SUB` are parsed by W&B as
  a nested path and the leaf collapses to `SUB` / `DIV` / `POW` / `STAT`. Verified present in
  `code/final_run_metrics.json`. Per-family identity is destroyed in the UI and in any programmatic
  export. **Fix:** rename to `E1_ADD_SUB` etc.

Also: MoR reports `val/routing_accuracy = 0.1046` in every run. With `num_experts=1` the router index is
always 0, so this is just "the fraction of val tokens whose oracle family is E1" — a data statistic, not
a routing result. `changes.md` §21 correctly requires `N/A` for MoR routing metrics; the code emits a
number instead, and `cannonic.md` §4 publishes it three times.

---

## 3. Which published results survive

Cross-referencing `cannonic.md` against the 42 W&B run directories:

| `cannonic.md` row | Source run | `step_routing` | blocks | `max_depth` | verdict |
|---|---|---|---|---|---|
| MoE Baseline 0.001957 | `20260825_141936` | 0.0 | 1 | 1 (fixed) | void (B1) |
| MoR Baseline 0.003160 | `20260825_142544` | 0.0 | 1 | 7 | void (B1, B6) |
| MoRE **1-Block** 0.002320 | `20260824_005230` | **0.0** | **1** | **7** | void (B1) |
| MoRE **2-Block** 0.002285 | `20260823_153736` | **0.5** | **2** | **5** | void (B1) **+ not comparable** |
| Fixed Depth 5 0.001778 | `20260825_143330` | 0.0 | 1 | 5 (fixed) | void (B1) |

### R1 — The headline 1-block vs 2-block comparison changes three variables at once

blocks (1→2), `max_depth` (7→5), **and** `step_routing` (0.0→0.5, i.e. oracle routing supervision
**off → on**). `changes.md` §25 Rule A ("no mixed configurations in one table") is violated by the
repository's own flagship table, and §11's ablation is unrecoverable from these runs.

The entropy difference the paper interprets architecturally — 1.9347 (1-block) vs 1.6837 (2-block) — is
fully explained by the third variable: the 2-block run had oracle CE active, pushing load away from
uniform. Its `expert_6_pct = 0.1328%` (vs 11.79% in the unsupervised run) is the fingerprint of oracle
supervision on an expert with zero oracle support (B5b), not of "weaker routing balance."

### R2 — The 2-block run's perfect routing is circular

Run `20260823_153736` reports `val/routing_accuracy = 1.0000` with **every** per-family recall at 1.0.
It was trained with `step_routing = 0.5`, i.e. cross-entropy against the very oracle labels the metric
scores. That is the "hidden oracle supervision" risk of `changes.md` §30, realized. It cannot appear
anywhere except as an explicitly labeled *"MoRE – Oracle Routing"* ablation (§20).

### R3 — Proxy data in headline-adjacent tables

`cannonic.md` §4 is 5-epoch / 10%-subset proxy data (`run_sweeps.py:29,45`), correctly labeled as such.
But it is the *only* source for the scaling-laws and batch-size conclusions restated without qualification
in `README.md` §3. Per §25 Rule B those conclusions must move to supplementary or be re-run at full scale.
`cannonic.md` §3.2's "Val Loss (1-Epoch)" column is honestly labeled but sourced from
`run_remaining_tests.py:139-140` runs that intended 50% data and got 100% (I1) — a labeling error, not a
confound.

### What does survive

- **Parameter counts** (`cannonic.md` §3.1), computed programmatically by
  `run_remaining_tests.py:20`. These are trustworthy and they confirm `changes.md` §7:
  `|567,569 − 3,724,055| / 3,724,055 = **84.8%**` — vastly outside the §8 threshold of 5%. MoE and MoRE
  are genuinely parameter-identical (both 7 experts), so *that* comparison is matched.
- **Throughput and memory profiling** (§3.2) — mechanical measurements, unaffected by B1. Note they
  measure the *dense* forward pass (B3), so they will change once top-1 lands.
- **The epoch-by-epoch training curve shape** (§5) — as evidence of optimizer stability, not of task
  performance.

---

## 4. `objective.md` and `rules.md` vs `changes.md`

Three governing documents currently conflict. This must be resolved explicitly or the next round of work
will reintroduce the same defects.

| Conflict | Resolution |
|---|---|
| `objective.md`: *"Avoid changing … logging, metric definitions"* | **`changes.md` supersedes.** §2, §5, §15–§18 all require metric redefinition. Mark `objective.md` as superseded for the finalization phase. |
| `objective.md`: primary objective is `expert_entropy > baseline`; "never accept routing collapse" | **Superseded.** Per B4, near-maximal entropy in this codebase means a *near-uniform, uninformative* router. Optimizing entropy as the primary objective is what produced below-chance routing accuracy. The primary metric is validation **task** loss (§5), with normalized entropy reported as diagnostic only (§16). |
| `objective.md`: allows tuning "router noise" as a free knob | **Constrain per §4.** Router noise becomes ablation A (none) / B (fixed annealed) / C (trainable), default **A or B**, not a search dimension. |
| `rules.md` §1: Top-1 routing is *immutable*; §5 mandates refusing changes to it | **`rules.md` is correct and the code violates it (B3).** |
| `rules.md` §2: mandates ACT halting + ponder cost | **`rules.md` is correct and the code violates it (B2).** |

### This resolves the §29 tension

`changes.md` §29 forbids architecture changes during the canonical matrix. B2 and B3 look like
architecture changes — but they are **restorations of `rules.md`, the repo's own frozen specification**.
Fixing them brings the implementation into compliance with the design it claims to implement. That is a
correctness fix, not an invention. Classify as follows:

| Defect | Classification under §29 | Allowed before the canonical matrix? |
|---|---|---|
| B1 leakage | Correctness (task definition was broken) | **Required** |
| B2 halting gradient | Correctness — restores `rules.md` §2 | **Required** |
| B3 top-1 routing | Correctness — restores `rules.md` §1 | **Required** |
| B4 balance normalization | Correctness (metric/objective scaling) | **Required** |
| B5 `num_experts` generality | Correctness | **Required** |
| B6 identical inputs | Correctness (fair comparison) | **Required** |
| I1–I5 | Logging / reproducibility | **Required** |
| Dense-routing variant | New ablation | Labeled ablation only |
| Router noise B/C | New ablation | Labeled ablation only |
| Routing supervision on | New ablation | Labeled ablation only, `"MoRE – Oracle Routing"` |

**Consequence: the canonical matrix (§26) cannot begin until Phase 1 and Phase 2 below are complete and
their acceptance tests pass.** Everything currently in `cannonic.md` §3–§5 must be re-run from scratch.
There is no partial-salvage path, because B1 alone invalidates every loss number.

---

## 5. Implementation plan

Phases are strictly ordered. Each phase has a **gate**: an automated test that must pass before the next
phase starts. Do not run experiments between phases.

### Phase 0 — Freeze and quarantine (½ day)

1. `git checkout -b finalization` from `136da4e`. Tag `136da4e` as `pre-finalization-archive`.
2. Move `cannonic.md` → `archive/cannonic_PRE_FIX_DO_NOT_CITE.md`; prepend a header stating that all
   loss-based results are invalidated by 100% label leakage, with a pointer to this document.
3. Do the same for `README.md` §2–§3 (the results sections). Keep §1 as a changelog, annotated with which
   "fixes" are now known to be wrong (§1.1 overshot → B3; §1.4 is inverted → see below).
4. Delete or `.gitignore` the CWD-written artifacts (`best_model.pt`, `results.tsv`,
   `1block_results.csv`, `2block_results.csv`, `sweep_results.csv`, `final_run_metrics.json`) — they are
   unversioned outputs from invalidated runs, written to fixed filenames at
   [train.py:1467](code/train.py:1467), [1488](code/train.py:1488), [1512](code/train.py:1512), so every
   run silently overwrites the last. Replace with `runs/{experiment_id}/` output directories.
5. Retire `code/autoresearch_runner.py` from the finalization path. Its loop is proxy-only (5 epochs /
   10% data, `autoresearch_runner.py:482`) and it lets an LLM patch configs and commit them — the exact
   generator of un-provenanced mixed runs that `changes.md` §22/§25 exists to prevent. Keep it for
   exploratory work on a separate branch.

**Note on `README.md` §1.4.** It states the trainable `router_noise_scale` "converged to 0, leaving the
router vulnerable to collapse" and that the fix was to add `0.001 × noise_scale²`
([train.py:1283](code/train.py:1283)). An L2 penalty **drives the parameter toward zero** — the stated
remedy accelerates the stated problem. Under `changes.md` §4 the whole mechanism should be deleted from
the canonical config anyway (ablation A), which resolves it. Flagging it because the same reasoning error
could recur.

**Gate 0:** `git log` shows the archive tag; no invalidated result file is reachable from `README.md`.

---

### Phase 1 — Fix the task (2–3 days)

This is the only phase that touches `data/`.

1. **Remove the leak** (B1). In `data/script.py`, keep `verify()`'s correctness assertion but stop the
   tokenizer from consuming it: in [train.py:270-276](code/train.py:270), feed only `flat_args`, and
   fully mask the numeric slots of the *final* real step.
2. **Remove the oracle label from the features** (B6). Delete `row[0] = expert_id / ...`. Replace with a
   learned embedding of the *operation token*, sized independently of `num_experts`.
3. **Assert input identity across architectures** (B6). A test that tokenizes a fixed record under
   `num_experts ∈ {1, 6, 7}` and asserts the resulting tensors are bit-identical.
4. **Regenerate the canonical dataset** with `num_experts = 6`, E7 removed (`changes.md` §1). Redistribute
   `FAMILY_WEIGHTS` (currently E7 holds 0.40 in `data/script.py`) across E1–E6 and document the choice in
   the manifest.
5. **Write the dataset manifest** (§13, §14): `dataset_version`, generator commit, seed, `TOTAL`, split
   fractions, per-family counts, `INT_RANGE`, `LIST_LEN`, nesting ratio, per-operation target-depth
   curriculum, and SHA-256 of each `.jsonl`.
6. **Leakage checks, mandatory and blocking** (§14): record IDs, input expression strings, and
   `(op, args, result)` tuples — pairwise across train/val/test. Plus the B1 target-in-input check. Any
   failure aborts.
7. **Re-derive and log the trivial baselines** for the new dataset: predict-zero MSE, predict-train-mean
   MSE, and best-single-feature-copy MSE. These become a permanent row in every results table. Without
   them a val loss is uninterpretable — the entire B1 finding follows from their absence.

**Gate 1 (all must pass):**
- No record in any split contains its own target in any feature slot (tolerance-checked).
- Zero overlap on record ID, expression string, and `(op, args, result)` across splits.
- Feature tensors bit-identical across `num_experts ∈ {1, 6, 7}`.
- Manifest hashes match the files on disk.
- `MSE(best single-feature copy) > 0.5 × MSE(predict train mean)` — i.e. the task is no longer solvable by copying.

---

### Phase 2 — Fix the model and the objective (3–4 days)

No dataset changes in this phase.

1. **Generalize `num_experts`** (B5). Replace literal `7` at
   [698](code/train.py:698), [703](code/train.py:703), [1267](code/train.py:1267),
   [1323](code/train.py:1323); derive `EXPERT_FAMILY_LABELS` from the manifest; delete
   `_EXPERT_FALLBACK` ([160](code/train.py:160)) and `raise` on an unmapped op.
2. **Rename family labels** to `E1_ADD_SUB` … `E6_SORT_STAT` (no `/`) — fixes I5.
3. **Top-1 routing with gate multiplier** (B3), behind `"routing_mode": "top1" | "dense"`, default
   `top1`.
4. **Differentiable halting** (B2): ponder cost + probability-weighted output mixing. Add
   `"halting": {"mode": "act" | "fixed", "ponder_weight": …}`.
5. **Normalize the balance loss** (B4): mean over `(block, depth)` calls, not sum. Re-tune
   `routing_balance` so `|weight × balance| ≲ 0.1 × task_loss`. Log the entropy term and Switch-aux term
   separately (§5).
6. **Remove router noise from canonical** (§4): `"router_noise": {"mode": "none"}` default; delete
   `noise_reg_loss` ([1283](code/train.py:1283)); keep `fixed_annealed` and `trainable` as ablations
   B and C. Do **not** add a router z-loss in the same change (§4).
7. **Gate routing supervision** (§3): `"routing_supervision": {"enabled": false, "weight": 0.0}` in
   canonical. Split the two currently-fused paths — `oracle_routing_ce` (router logits) and
   `step_cls_loss` (`step_cls_head`) — into separately weighted, separately logged terms, replacing the
   hard-coded `0.01 * oracle_routing_ce + 0.3 * step_cls_loss` at [train.py:1284](code/train.py:1284).
   Assert both are exactly `0.0` when disabled.
8. **Separate loss logging** (§5): `task_loss`, `classification_loss`, `routing_balance_loss`
   (+ entropy / switch components), `halting_loss`, `routing_supervision_loss`, `total_loss`.
9. **Global seeding** (I3): `set_seed()` + deterministic algorithms + seeded DataLoader.

**Gate 2 — a real test suite (`code/tests/`), all must pass:**

| Test | Assertion |
|---|---|
| `test_halt_gradients` | after one backward, every `expert_halt_heads` param has `grad is not None` and `grad.abs().sum() > 0` |
| `test_topk_sparsity` | with `routing_mode="top1"`, exactly one expert's FLOPs are consumed per token (count via forward hooks) |
| `test_balance_sign` | collapsed → `H≈0`, `switch_aux≈E`, gradient points away from collapse; uniform → `H≈log E`, `switch_aux≈1`, balance at minimum (§6) |
| `test_balance_depth_invariance` | `balance_loss(depth=1) ≈ balance_loss(depth=5)` within 1e-6 |
| `test_num_experts_generality` | for `E ∈ {1,2,6,7}`: `cls_head.out_features == E`, `step_cls_head.out_features == E`, confusion is `E×E` |
| `test_routing_accuracy_consistency` | `abs(confusion_accuracy − direct_accuracy) < 1e-6` (§2.1), asserted in **both** the train-loop and standalone evaluators — which currently disagree in shape |
| `test_supervision_off` | with `routing_supervision.enabled=false`, both supervision terms are exactly `0.0` and no gradient reaches the router from them |
| `test_determinism` | two runs with the same seed produce bit-identical loss for 10 steps |
| `test_no_sentinel_leak` | with `E=1`, `max_pairwise_cosine` is `None`/`nan`, never `-1.0` (I5) |
| `test_param_match` | asserts the recorded param counts for each variant against a frozen expected value |

---

### Phase 3 — Metrics, provenance, and export (2 days)

1. **Authoritative routing accuracy** (§2.1). One function, one definition:
   `(predicted_expert == oracle_expert).float().mean()` over tokens where
   `real ∧ oracle_valid ∧ pred_valid`. Both the confusion matrix and the scalar derive from it. Assert
   agreement to `1e-6` and **stop the run on failure**. Name it *"first-step expert routing accuracy"*
   everywhere.
2. **Add permutation-aligned metrics.** This is the substantive addition beyond `changes.md`, and it is
   necessary. In run `20260824_005230` the router achieved near-maximal entropy (1.9451) with
   **below-chance** accuracy (0.0119 vs chance `1/7 = 0.1429`), and per-family recall showing one family
   near 1.0 and the rest near 0. Below-chance-with-high-entropy is the signature of a **consistent but
   permuted** assignment: the router may be partitioning tokens systematically while its expert *indices*
   don't match the oracle's labeling. Raw accuracy cannot see this. Report all three:
   - raw first-step routing accuracy (the §2.1 metric, for the confusion matrix);
   - **Hungarian-matched accuracy** (optimal permutation of predicted → oracle labels);
   - **AMI** and **cluster purity** (permutation-invariant by construction).
   If matched accuracy ≫ raw accuracy, the honest finding is *"the router learns a consistent but
   unaligned partition"* — a genuinely publishable negative-to-neutral result. If both are at chance,
   the router learned nothing, and that must be stated plainly.
3. **Normalized entropy** `H / log E` (§16), reported alongside raw `H`. Never compare raw `H` across
   different `E`. Emit `N/A` for `E=1`.
4. **Cosine similarity** (§17): mean **and** max pairwise. Phrase as *"consistent with differentiated
   parameterizations"* — never "proves orthogonality." `N/A` for `E=1`, never `-1.0`.
5. **Depth allocation error** (§18): absolute and relative, against the manifest's curriculum.
6. **Per-family precision and recall + macro accuracy** (§15), with underscore-safe keys.
7. **Full W&B provenance** (§22): every field in the §22 list, plus `experiment_group`, resolved
   `epochs` (fixing I2), resolved `subset_fraction` (fixing I1), and a `config_hash` over the fully
   resolved config. Run names `phaseB_{ARCH}_seed{N}`.
8. **`code/export_results.py`** (§23): filter `experiment_group == "canonical_phase_b" ∧
   dataset_version == CANONICAL` → `results.csv` / `results.json` → `results.md`. It must emit `N/A`
   (never a fabricated or sentinel value) for any metric a variant cannot produce (§25 Rule D), and it
   must **refuse to emit a table** if any run in the filter set has a mismatched `config_hash` or
   `dataset_version`.

**Gate 3:** a 2-epoch smoke run writes a complete provenance block; `export_results.py` produces a valid
`results.md` skeleton from it; the routing-accuracy assertion is live and fires when deliberately broken;
MoR emits `N/A` for entropy, cosine, and routing accuracy.

---

### Phase 4 — Canonical run matrix (§26)

Only after Gates 1–3. One frozen config family, one frozen dataset version, `batch_size = 768` fixed for
all architecture comparisons (§12).

**Primary controlled comparison** — identical hyperparameters, `num_experts` and `max_depth` varying only
as the architecture definition requires (§9):

| Variant | experts | max_depth | halting | routing |
|---|---|---|---|---|
| MoE | 6 | 1 | fixed | learned top-1 |
| MoR | 1 | canonical | adaptive | N/A |
| MoRE | 6 | canonical | adaptive | learned top-1 |

× seeds {42, 43, 44, 45, 46} = **15 runs**. Report mean ± std (§21).

**Secondary parameter-matched comparison** (§7/§8): widen MoR's `d_model`/FFN until
`|params_MoR − params_MoRE| / params_MoRE < 0.05`, verified programmatically and logged. × 5 seeds =
**5 runs**. Report *both* comparisons; neither alone is honest.

**Ablations**, each 5 seeds, each explicitly labeled and never in the headline table:
fixed-depth {1, 3, 5} on the canonical dataset (§10) · 1-block vs 2-block with *only* the block count
varying (§11, fixing R1) · router noise A/B/C (§4) · routing supervision on = **"MoRE – Oracle Routing"**
(§3, §20) · dense vs top-1 routing (B3).

Batch-size sweep → supplementary only (§12).

**Reporting rules to enforce mechanically in `export_results.py`** (§25):
no mixed configurations in one table (assert identical `config_hash` modulo the declared varying field) ·
no proxy values in headline tables (assert `epochs == canonical ∧ subset_fraction == 1.0`) · no run
lacking seed + config hash + dataset version · **`N/A` rather than any fabricated or sentinel value** ·
no causal language without an ablation supporting it · every figure carries its source run list.

Include the **trivial-baseline row** (predict-zero / predict-mean / best-copy) in every loss table.

---

### Phase 5 — `results.md` and framing

Write `results.md` to the exact 17-section structure of §24. Apply §27's Outcome A/B/C decision rules
honestly — including the real possibility, given B2 and B4, that the answer is **Outcome C**: adaptive
depth confers no measurable benefit on this benchmark once halting is actually trained and the leak is
removed. That is a legitimate result and the framing (§28: exploratory proof-of-concept, "controlled
synthetic benchmark," not "beating MoE") is built to accommodate it.

Language bans, carried from §16/§17/§19/§20/§28:
- ✗ "proves experts learn orthogonal representations" → ✓ "consistent with differentiated parameterizations"
- ✗ "the model discovered that logic ops are harder" → ✓ "target depth is a property of our supervision design, not measured intrinsic complexity"
- ✗ "adaptive computation entirely without supervision" (§3) → state exactly which supervision was enabled
- ✗ comparing raw entropy across different `E` → always `H / log E`
- ✗ any oracle-routing number in a headline → labeled `"MoRE – Oracle Routing"` ablation only

---

## 6. Effort and sequencing

| Phase | Work | Compute |
|---|---|---|
| 0 Freeze & quarantine | ½ day | — |
| 1 Fix task + dataset | 2–3 days | dataset regen (minutes) |
| 2 Fix model + objective | 3–4 days | test suite only |
| 3 Metrics + provenance + export | 2 days | 2-epoch smoke runs |
| 4 Canonical matrix | 20 primary + ~40 ablation runs | the bulk of GPU time |
| 5 `results.md` | 2 days | — |

Phase 4 GPU cost: the 50-epoch full-data runs took roughly 30–75 min each on the RTX 4060 Laptop 8 GB
(per W&B durations), but this will **increase** once top-1 dispatch replaces the dense blend and the
recursion becomes probability-weighted. Budget ≈ 60 runs × ~1 h ≈ **60 GPU-hours**, i.e. 3–5 days wall
clock locally. `cannonic.md` §2's rental guidance ($0.15–0.25/hr spot) puts a cloud alternative at
**$10–15**, which is likely worth it for the 5-seed protocol.

**Phases 1 and 2 are parallelizable** (disjoint files: `data/script.py` + tokenizer vs model + loss), but
Gate 1 must pass before Phase 4 either way.

---

## 7. Risk register

`changes.md` §30's named risks, mapped to what was actually found:

| §30 risk | Actual status |
|---|---|
| Metric inconsistency | **Partly misdiagnosed.** `val/step_routing_accuracy` does not exist; the two live accuracy computations agree by construction. The real inconsistency is that the train-loop confusion is hard-coded `7×7` ([1323](code/train.py:1323)) while the standalone evaluator is `E×E` ([852](code/train.py:852)). |
| Unfair MoR comparison | **Confirmed and worse than described.** 84.8% parameter gap *and* a 5× different slot-0 input scale (B6). |
| Hidden oracle supervision | **Confirmed twice.** `step_routing=0.5` in the 2-block headline run (R2), *and* the oracle family label fed as input feature slot 0 in every run (B6). |
| Unexplained router noise | **Confirmed**, plus `README.md` §1.4's L2 "fix" is inverted. |
| Mixed / proxy W&B runs | **Confirmed.** R1 (three variables at once), I2 (epochs misreported), I1 (subset silently overridden). |
| Ambiguous depth labels | **Confirmed and moot** — depth is untrained (B2), so the labels describe noise. |
| Overclaiming adaptive discovery | **Confirmed.** `README.md`'s "212% throughput acceleration due to healthy adaptive early exiting" and `cannonic.md` §3.5's op→depth curriculum both rest on B2. |

**New risks not in §30:**

| Risk | Mitigation |
|---|---|
| **Task admits a closed-form copy solution** (B1) — the single most damaging defect | Gate 1's target-in-input assertion + trivial-baseline row in every table |
| **Halting has no gradient path** (B2) | `test_halt_gradients` in Gate 2 |
| **Objective dominated by an unbounded depth-coupled entropy reward** (B4) | mean-normalization + `test_balance_depth_invariance` + re-tuned weight |
| **Below-chance accuracy at maximal entropy is read as "balanced routing"** | Hungarian / AMI / purity metrics (Phase 3.2) |
| **`verify_pipeline.py` gives false assurance** — its "gradient flow verified" step tests a bare `nn.Linear`, not MoRE, which is why B2 survived 42 runs | replace with the Gate-2 suite; make `verify_pipeline.py` invoke it |
| **CWD-fixed output filenames** ([1467](code/train.py:1467), [1488](code/train.py:1488), [1512](code/train.py:1512)) silently overwrite prior runs | per-run `runs/{experiment_id}/` directories |
| **`autoresearch_runner.py` can commit LLM-proposed config patches** on a proxy budget | quarantined in Phase 0 |

---

## 8. The one-paragraph summary

The repository's engineering is competent — the W&B instrumentation, the confusion-matrix figures, the
config system, and the profiling harness are all real work. But four defects mean the experiments measure
something other than what they claim. The regression target is present verbatim in the input features of
100% of records, so validation loss measures feature-copying, and every published loss number is void.
The halting loss has `requires_grad=False` and every halting-head parameter receives `grad=None`, so
"adaptive computation" is untrained random initialization — which is why two runs report opposite,
mutually contradictory op→depth curricula, both anti-correlated with the generator's intent. Routing
executes a dense blend of all experts rather than top-1, so there is no sparsity and the argmax being
scored does not determine the forward pass. And the load-balance term is accumulated per depth per block
at ~146× the task loss, making "maximize router entropy" the de facto training objective — which is why
entropy sits at 1.945/1.946 while routing accuracy sits *below chance*, and why the shipped metrics file
records a negative total loss. Separately, the flagship 1-block vs 2-block comparison varies three
things at once, including oracle routing supervision on versus off. `changes.md` §29's prohibition on
architecture changes cannot be honored as literally written — but it does not need to be: the two changes
that look architectural (top-1 routing, ACT halting with a ponder cost) are restorations of the
repository's own `rules.md`, which declares both immutable. Fixing them is compliance, not invention.
Phases 1–3 above take roughly two weeks and are prerequisites; the canonical matrix cannot start before
Gate 3, and everything in `cannonic.md` §3–§5 must be regenerated.
