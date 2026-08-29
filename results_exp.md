# results_exp.md — how to read every number this project produces

**What this file is.** One entry per metric that reaches `metrics.json` or the
results table: what it measures, what value is good, whether higher or lower is
better, and when it is legitimately `N/A`. It is a reading guide, not a results
file — no measured result is a target, and nothing here may be optimized toward
(`CLAUDE.md` §8).

**Two rules that override any "ideal" column below.**

1. **`"N/A"` is a real answer, not a missing one.** Every `N/A` in this
   repository means *this quantity does not exist for this architecture*. MoR has
   one expert, so it has no routing accuracy, no load entropy, no pairwise
   cosine. MoE has `max_depth = 1`, so its depth metrics are constants, not
   measurements. A `0.0` in place of an `N/A` is a fabricated data point — that is
   why the code writes the string `"N/A"` and omits the key from W&B rather than
   charting a zero.
2. **A metric being "better" is not the same as the architecture being better.**
   Several metrics here are *diagnostics* (load entropy, cosine similarity,
   per-token router entropy). High entropy is not specialization. Low cosine is
   not orthogonality. Those readings are explicitly prohibited
   (`CLAUDE.md` §4, §6) and the wording in each entry below is the wording to use.

---

## 0. Before trusting any number: check the provenance block

`resolved_config.json:provenance`. If any of these disagrees with what you think
you ran, stop and read no further.

| field | why it decides whether the numbers mean anything |
|---|---|
| `experiment_group` | `canonical_phase_b` = admissible in the headline table. Anything else (`exploratory`, `t81_smoke`, …) is a **proxy** and may never be tabled as a result. |
| `variant` | `canonical`, or the name of the single field that differs (`no_family_supervision`, `ffn_mult_4`, `oracle_routing`, …). A non-`canonical` variant belongs in an ablation table whose caption names that field. |
| `seed_declared` / `resolved_seed` | `false` means the run used the default 42 without declaring it; Gate 0 refuses a canonical claim from it. |
| `routing_supervision_weight` | Non-zero means the router was **trained on the oracle expert index**. Every routing metric below then measures the supervision signal, not the architecture. Canonical is `0.0`. |
| `family_cls_weight` | Non-zero (canonical `0.5`) means a 6-way cross-entropy on the **external** whole-program family label shaped the shared trunk. See §3's honesty note. |
| `halt_target_mode` | `pure_act` = halting is learned with no depth targets. `supervised_curriculum` = depth was supervised; depth metrics are then not a discovery. |
| `total_params` | The parameter-matching claim rests on this, not on `ffn_mult`. |
| `dataset_version`, `train_split_version` | Two runs with different values may never appear in one table. |
| `config_hash` | Canonical runs must differ only by architecture and seed. |

---

## 1. Primary predictive metric

| key | meaning | ideal | direction |
|---|---|---|---|
| **`val/task_loss`** | MSE of the numeric prediction on the held-out val split. **This is the primary quality metric for the whole project.** | as low as the task allows | **lower is better** |
| `val/loss`, `best_val_loss` | Same quantity; `best_val_loss` is the minimum over epochs and is what checkpoint selection uses. | — | lower |
| `train/task_loss` | Same MSE on train. Read only against `val/task_loss` to judge overfitting. | close to val | lower, but a large train↔val gap is the finding |

**The reference floors — read `val/task_loss` against these, never against 0.**
Computed on the real split (val n = 5250, target mean 0.030892, std 0.284455,
targets in [−1, 1]):

| trivial predictor | val MSE |
|---|---|
| predict the train mean | **0.080914** |
| predict the train median | 0.081853 |
| predict zero | 0.081853 |
| *copy one input slot* (leakage canary) | *0.081853* |

Measured for orientation: ~0.0713–0.0721 at 2 epochs, ~0.0635 at 20 epochs
→ **R² ≈ 0.215** against predict-the-mean. That is a modest fraction of target
variance and must be reported as such. It also means the three architectures sit
in a narrow band, so a task-loss difference is only a difference if it exceeds the
seed std (report mean ± std over seeds 42–46, never one seed).

**The leakage canary.** If `val/task_loss` ever approaches 0, suspect input
leakage before celebrating: the copy-one-input-slot baseline used to score exactly
`0.000000`. Run `python code/audit_leakage.py` (Gate 1, 8/8). A residual honest
artefact: the target coincides with an argument in 33.45% of val records, inherent
to MAX/MIN/MEDIAN/SORT where the answer *is* an input.

---

## 2. Loss decomposition — never report `train/total_loss` as quality

`CLAUDE.md` §4 forbids using the weighted total as the headline number. It can go
**negative**, because the balance term contains a negative entropy contribution,
and a negative total is not evidence of anything.

| key | what it is | ideal | direction |
|---|---|---|---|
| `train/total_loss` | The weighted sum actually optimized. Diagnostic only. | — | **not a quality metric** |
| `train/task_loss` | The MSE component. Dominates a healthy run. | — | lower |
| `train/classification_loss` (= `train/family_cls_loss`) | 6-way CE on `cls_head(pooled)` vs the **external** family label, weight `family_cls = 0.5`. | low | lower, but see §3 note |
| `train/routing_balance_loss` (= `train/aux_routing_loss`) | `−entropy_term + switch_aux_term`. **Legitimately negative.** | small in magnitude | neither — it is a regularizer |
| `train/entropy_term` | Mean per-token router entropy (nats). | — | see §4 |
| `train/entropy_term_normalized` | The same, ÷ `log(E)`. Comparable across expert counts. | — | see §4 |
| `train/switch_aux_term` | Switch-style load-balancing auxiliary. Pulls opposite to the entropy term. | small | lower |
| `train/balance_to_task_ratio` | \|weight × balance\| ÷ \|task loss\|. The T4.2 acceptance measure. | **≪ 1** (~0.008 measured) | lower; > 1 means the balancer became the objective |
| `train/step_routing_loss` | `0.01·oracle_routing_CE + 0.3·step_cls_loss`. **Weight is 0.0 in canonical runs**, so this is reported but not optimized. | — | ignore unless the weight is non-zero |
| `train/halting_loss` (= `train/ponder_cost`) | Differentiable recursion cost; the term that puts gradient on the halt heads. Must be > 0 whenever depth is adaptive. | > 0 | lower = cheaper, but 0 means halting is untrained |
| `train/halting_supervision_loss` | CE against the hand-written depth curriculum. **0.0 in canonical** (`pure_act`). | 0.0 | non-zero ⇒ depth was supervised |

Never sum `ponder_cost` and `halting_supervision_loss` into one "halting loss".

---

## 3. Routing metrics — the index carries no meaning

Expert indices are arbitrary. A router can partition the operation space perfectly
and number the groups differently on every seed. **Permutation-invariant metrics
are mandatory**, and the *gap* between raw and matched is the actual result.

| key | meaning | ideal | direction | `N/A` when |
|---|---|---|---|---|
| `val/routing_accuracy` (`raw_accuracy`) | `mean(argmax == oracle_expert)` under the **identity** mapping. | — | **do not read alone** | `num_experts ≤ 1` |
| **`val/routing_hungarian_accuracy`** | Accuracy after the optimal 1-to-1 family↔expert matching. **The headline specialization number.** | ≫ chance **0.167** | higher | `num_experts ≤ 1` |
| `val/routing_ami` | Adjusted mutual information, cluster-vs-family. Chance-corrected. | > 0; ≫ 0 is a real partition | higher | `num_experts ≤ 1` |
| `val/routing_purity` | Fraction in each expert's majority family. | ≫ 0.167 | higher | `num_experts ≤ 1` |
| `val/routing_macro_recall` | Unweighted mean per-family recall, **identity** mapping. | — | higher, but see below | `num_experts ≤ 1` |
| **`val/routing_matched_macro_recall`** | The same **after** the Hungarian permutation. **This is the one to report for an unsupervised router.** | high | higher | `num_experts ≤ 1` |
| `val/routing_precision/…`, `…_recall/…`, `…_f1/…` | Per-family, **identity** mapping. Interpretable only when `step_routing` supervised the index. | — | higher | pre-T6.8 runs lack the matched pair |
| `val/routing_{precision,recall,f1}_matched/<FAMILY>` | Per-family **after** matching, plus `matched_expert`. | high and even across families | higher | `num_experts ≤ 1` |
| `val/routing_hungarian_assignment` | The recovered matching, e.g. `4->0->5->1->3->2`. | — | see below | `num_experts ≤ 1` |
| `val/routing_collapsed_experts` | Experts receiving ~no tokens. | `none` | fewer | `num_experts ≤ 1` |

**How to read the raw↔Hungarian gap (the T6.3 contract).**

| pattern | reading |
|---|---|
| raw ≪ Hungarian, AMI > 0 | **A real partition, arbitrarily numbered.** The publishable specialization result. Measured: raw 0.0800 vs Hungarian **0.5147 ± 0.0047**, AMI 0.4940, chance 0.167, with a *different* assignment on every seed — that stability under relabelling is the evidence. |
| Hungarian ≈ chance, AMI ≈ 0 | **No partition.** Outcome C for the specialization claim. Report it. |
| raw ≈ Hungarian ≈ 1.0 | The index itself was supervised. Legitimate **only** with `routing_supervision_weight > 0`; on a canonical run treat it as a suspected label leak, not a result. |
| identical assignment on every seed at high accuracy | Same — supervision or leak, not discovery. |

**Chance is 0.167** (1/6), not 0. Always state it beside the number.

**The honesty note that must accompany every specialization claim.** Canonical
runs set `step_routing = 0.0`, so there is no *direct* routing supervision — but
`family_cls = 0.5` still trains `cls_head` on the same oracle family label, read
off disk from a hand-written manifest. The claim therefore reads *"specialization
emerges without direct routing supervision, under whole-program family
supervision"*, never "unsupervised expert discovery". T10.H measures how much of
the partition that term accounts for: **~24%**. With `family_cls = 0.0` as well —
no oracle-derived label anywhere in the objective — Hungarian accuracy is still
**0.4300 ± 0.0293** against chance 0.167 (AMI 0.3568 ± 0.0431), so 76% of the
above-chance partition is retained, and removing the term costs nothing
measurable on task loss (−0.000100 against a seed std of 0.000428). That is a
labelled ablation result, not a canonical one; the sentence above still governs
every canonical row.

---

## 4. Load-balance and differentiation diagnostics — none of these is specialization

| key | meaning | ideal | direction | `N/A` when |
|---|---|---|---|---|
| `train/expert_load_entropy_normalized` | Entropy of the **aggregate** expert-load distribution ÷ `log(E)`. 1.0 = every expert gets an equal share, 0.0 = one takes everything. | high, but **not a target** | higher = balanced | `num_experts ≤ 1` |
| `train/entropy_term_normalized` | Mean **per-token** router entropy ÷ `log(E)`. Near 1.0 means the router is *undecided per token*, which is the opposite of confident routing. | — | **neither** | `num_experts ≤ 1` |
| `expert_load/expert_<i>_pct` | Share of tokens per expert. | no expert ≈ 0% | — | `num_experts ≤ 1` |
| `dispatch/max_load_fraction` | Largest single-expert share. | ≪ 1.0 | lower | — |
| `diag/mean_pairwise_cosine_sim`, `diag/max_pairwise_cosine_sim` | Similarity between expert parameter vectors. | low | lower | `num_experts ≤ 1` |

**Do not write** "high entropy proves specialization" or "low cosine proves
orthogonality". The permitted phrasing for cosine is *"consistent with
differentiated parameterizations"*. The two entropy metrics above are **different
quantities** and are easy to confuse: a run can have balanced *load* (0.914) and
an almost-uniform *per-token* distribution (0.975) at the same time — which
together mean tokens are spread evenly because the router is barely committing,
not because it has learned six crisp specialists. Read them with the Hungarian
number, never instead of it.

---

## 5. Depth / adaptive computation

| key | meaning | ideal | direction | `N/A` when |
|---|---|---|---|---|
| `train/avg_recursion_steps` | Mean recursion steps per token. | between 1 and `max_depth` | neither — it is the mechanism | `max_depth = 1` |
| `depth_dist/step_<k>_pct` | Share of tokens exiting at step *k*. | spread, not a single spike | — | `max_depth = 1` |
| `recursion/avg_depth_by_family/<FAMILY>`, `…_by_op/<OP>` | Mean depth per family / operation. | ordered like the complexity curriculum | — | `max_depth = 1` |
| **`depth/allocation_error_abs`** | Mean \|actual depth − curriculum target depth\|, in steps. | low | **lower** | `max_depth = 1` |
| `depth/allocation_error_rel` | The same, relative to the target. | low | lower | `max_depth = 1` |
| `halt/early_exit_rate` | Fraction halting before `max_depth`. | high | higher = adaptive | `max_depth = 1` |
| `halt/forced_exit_rate` | Fraction hitting the hard cap. | ≈ 0 | lower | `max_depth = 1` |
| `halt/mean_halt_mass` | Total halting probability mass accumulated. ≈ 1.0 means ACT closed properly. | ≈ 1.0 | — | `max_depth = 1` |
| `halt/mean_remainder` | Leftover mass assigned at exit. | small | lower | `max_depth = 1` |

**Terminology is enforced.** Call it **depth allocation error**, never "compute
efficiency" — no FLOPs are in the metric. Never write that the model "discovers
intrinsic mathematical complexity"; the honest sentence is *the model allocates
recursion depth in accordance with the predefined operation-complexity
curriculum* — **and only if the measurement supports it.** It currently does not:
the per-seed correlation between allocated and target depth is
**r = +0.204 ± 0.554**, so depth is adaptive in the mechanical sense (tokens exit
at different steps, the halt heads receive gradient) while the allocation it
settles on is seed-dependent and not the designed curriculum. Say that.

---

## 6. Sparsity and cost

| key | meaning | ideal | direction |
|---|---|---|---|
| **`dispatch/evals_per_token`** | Expert FFN evaluations per token. **`1.0` = true Top-1 sparse dispatch; `E` (6.0) = dense evaluate-all-and-blend.** The single number that distinguishes the canonical path from the dense ablation. | **exactly 1.0** for canonical | must be 1.0 |
| `dispatch/routing_mode` | `top1_sparse` (canonical) or the dense ablation tag. | `top1_sparse` | — |
| `dispatch/expert_evaluations`, `…/tokens_dispatched` | The raw counts `evals_per_token` is built from. | equal to each other | — |
| `dispatch/experts_called_max` | Distinct experts used across the epoch. | `E` | higher |
| `dispatch/overflow_rate`, `…/overflow_tokens` | Tokens dropped by a capacity limit. | **0.0** | lower |
| `dispatch/capacity_policy` | `no_capacity_limit` in canonical runs. | — | — |
| `dispatch/router_noise`, `…/router_noise_scale` | Canonical `none` / `0.0`. Noise is an ablation only. | `none` / 0.0 | — |
| `perf/throughput_tokens_sec` | Tokens/second, this machine, this batch size. **Not comparable across machines or batch sizes**, and MoRE's sparse gather makes it much slower than MoE at equal parameters (measured 2411 vs 7919 tok/s). Report as an engineering observation, never as an architectural efficiency claim. | — | higher |

---

## 7. Fields that are *not* metrics but decide admissibility

`architecture`, `num_experts`, `halting_mode`, `routing_mode`, `router_noise`,
`experiment_group`, `experiment_id`, `config_hash`, `epoch`,
`routing_balance_normalization`, `nondeterministic_ops_observed`
(must read `['none']`), and `_non_scalar_keys_omitted` — the list of plots and
tables deliberately **not** serialized into `metrics.json`, so that an object repr
with a memory address can never be mistaken for a measurement.

---

## 8. The five misreadings this project has already made once

1. **A sentinel reported as a measurement.** MoR once published *routing accuracy
   0.1005* with one expert — that number is "the fraction of tokens that happen to
   belong to E1", and in a table it reads as *MoR is bad at routing*. Anything
   architecture-inapplicable must be `N/A`.
2. **Absence read as a pass.** A gate that counts zero assertions is not a
   passing gate, and a missing key is not `"N/A"`. Both were live defects.
3. **A supervised number read as a discovery.** Routing accuracy ~1.0 with
   `step_routing > 0` measures the label, not the model.
4. **A proxy quoted as a result.** Anything with a non-canonical epoch count,
   subset fraction, loss configuration, dataset version or batch protocol is a
   proxy. `experiment_group` is the field that says so.
5. **One favourable seed.** Report mean ± std over seeds 42–46. Differences
   smaller than the seed std are not differences.
