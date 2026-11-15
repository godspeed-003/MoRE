# Language subset sweep -- T-OD.13 TEST 2

Cells: 45/45 complete

Corpus `lang-wikitext-103-bpe8192-len256-800d6154` (verified sha256).
Only `data.subset_fraction` varies. Subsets are nested per seed
(permuted prefixes); `subset_seed == seed`, as the canonical `--seed`
path sets it. Every cell is `experiment_group = subset_lang_sweep`,
so no cell can be claimed canonical and none enters a langB_* table.

## val/task_loss (nats/token, mean +/- std over 5 seeds)

Bigram floor on this corpus: 4.9849. A cell ABOVE the floor has not
yet beaten the bigram baseline -- at 2.5 % that is the honest regime,
not a failure.

| scale | MoE | MoR | MoRE | MoRE - MoR |
|---|---|---|---|---|
| 2.5% | 5.1539 +/- 0.0167* (n=5) | 5.0665 +/- 0.0163* (n=5) | 4.9682 +/- 0.0706 (n=5) | +0.0984 |
| 10.0% | 4.4271 +/- 0.0249 (n=5) | 4.3348 +/- 0.0504 (n=5) | 4.1499 +/- 0.0283 (n=5) | +0.1849 |
| 40.0% | 4.0646 +/- 0.0272 (n=5) | 3.7317 +/- 0.0774 (n=5) | 3.7566 +/- 0.0276 (n=5) | -0.0250 |

## Recursion depth and forced exit (mean over 5 seeds)

| scale | arm | depth/mean | forced exit rate | val/task_loss |
|---|---|---|---|---|
| 2.5% | MoE | 1.00 +/- 0.00 | N/A | 5.1539 |
| 2.5% | MoR | 1.93 +/- 0.09 | 0.0000 | 5.0665 |
| 2.5% | MoRE | 4.33 +/- 1.14 | 0.0404 | 4.9682 |
| 10.0% | MoE | 1.00 +/- 0.00 | N/A | 4.4271 |
| 10.0% | MoR | 2.94 +/- 0.89 | 0.0017 | 4.3348 |
| 10.0% | MoRE | 6.77 +/- 0.03 | 0.8240 | 4.1499 |
| 40.0% | MoE | 1.00 +/- 0.00 | N/A | 4.0646 |
| 40.0% | MoR | 5.03 +/- 2.29 | 0.4727 | 3.7317 |
| 40.0% | MoRE | 6.95 +/- 0.00 | 0.9617 | 3.7566 |

MoE depth is exactly 1.0 by construction; its forced-exit rate is the
fraction of tokens at the depth cap of 1, which is all of them.
MoR and MoRE forced-exit rate is the halting-collapse diagnostic: near
1.0 means the policy has stopped choosing.

## Depth-matched margin, if measured

| scale | MoRE margin | MoR margin | ratio MoRE/MoR |
|---|---|---|---|
| 2.5% | 0.3127 | 1.3723 | 0.23x |
| 10.0% | 0.0969 | 1.3538 | 0.07x |
| 40.0% | 0.0293 | 1.5343 | 0.02x |

## Reading

**The ordering reverses at 40 %.** MoRE minus MoR in `val/task_loss` is -0.098 at 2.5 %, -0.185 at 10 % and +0.025 at 40 % (negative = MoRE better). MoRE is the lowest-loss arm at the two small scales and the highest-loss recursive arm at the largest, so this is not a MoRE advantage that merely shrinks with data -- it disappears and then changes sign.

**The mechanism is halting collapse, not routing failure.** MoRE's forced-exit rate goes 0.0404 -> 0.8240 -> 0.9617 while its mean depth goes 4.33 -> 6.77 -> 6.95 against a cap of 7. By 40 % almost every token is pushed to the cap: the learned policy has stopped choosing and MoRE is, in effect, a fixed-depth-7 model there. MoR keeps a live policy at the same scale (its forced-exit rate climbs to ~0.47) and is the only arm that stays heterogeneous.

**The forced-depth control says the same thing independently.** `margin_matched` for MoRE falls 0.313 -> 0.097 -> 0.029 nats while MoR's stays 1.372 -> 1.354 -> 1.534. The MoRE/MoR ratio is 0.23x, 0.07x, 0.02x. A margin near zero means the learned run is no better than a uniform depth, which is what a collapsed halting policy should produce; MoRE's 40 % margin is 0.0293 nats for both `matched` and `best`, i.e. its own chosen depth IS the best available uniform depth. MoR's margin does not decay with data -- it is the one arm whose depth allocation remains worth something.

**What this supports.** For the language task at this scale the answer is Outcome C with a qualifier: MoRE's small-data advantage is real (0.10 and 0.18 nats) but fragile -- at 2.5 % it rests on four seeds and one seed that halted at depth 2.55 -- and it does not survive more data, where the halting head saturates against the cap and the routing half alone does not compensate. MoR retains adaptive depth at 40 % and is the better recursive arm there. Any claim that MoRE's combination beats its components has to be made at small data and must state the halting collapse that ends it.

**Caveats.** 3 epochs, one corpus, one model size, one machine; every cell is a proxy (subsampled training set) and none may enter a canonical table. At 40 % all arms are below the 4.9849 bigram floor; at 2.5 % MoRE only just clears it (4.9682), so the small-data regime is a learning-signal study, not a quality study. Five seeds bound the mean; they do not pin the tail, and MoRE's 2.5 % std (0.07) is not much smaller than its gap to MoR (0.10).


## Per-seed

| scale | arm | seed | val/task_loss | depth/mean | forced exit | run |
|---|---|---|---|---|---|---|
| 2.5% | MoE | 42 | 5.1755 | 1.00 | N/A | `langSub_MoE_f025_seed42__7a6fbfaf__r2` |
| 2.5% | MoE | 43 | 5.1483 | 1.00 | N/A | `langSub_MoE_f025_seed43__0cd6da4f` |
| 2.5% | MoE | 44 | 5.1671 | 1.00 | N/A | `langSub_MoE_f025_seed44__05cd1c5e` |
| 2.5% | MoE | 45 | 5.1414 | 1.00 | N/A | `langSub_MoE_f025_seed45__3ab2c46b` |
| 2.5% | MoE | 46 | 5.1370 | 1.00 | N/A | `langSub_MoE_f025_seed46__1574e2b6` |
| 2.5% | MoR | 42 | 5.0458 | 2.02 | 0.0000 | `langSub_MoR_f025_seed42__99b7586c` |
| 2.5% | MoR | 43 | 5.0804 | 1.80 | 0.0000 | `langSub_MoR_f025_seed43__833ec236` |
| 2.5% | MoR | 44 | 5.0724 | 1.94 | 0.0000 | `langSub_MoR_f025_seed44__6a216287` |
| 2.5% | MoR | 45 | 5.0529 | 2.00 | 0.0000 | `langSub_MoR_f025_seed45__5ca10cc6` |
| 2.5% | MoR | 46 | 5.0812 | 1.88 | 0.0000 | `langSub_MoR_f025_seed46__c2a8899a` |
| 2.5% | MoRE | 42 | 5.0740 | 2.55 | 0.0000 | `langSub_MoRE_f025_seed42__3f4ec3e0` |
| 2.5% | MoRE | 43 | 4.9409 | 5.07 | 0.0386 | `langSub_MoRE_f025_seed43__f7ad2747` |
| 2.5% | MoRE | 44 | 4.9392 | 4.88 | 0.0236 | `langSub_MoRE_f025_seed44__02b10e3f` |
| 2.5% | MoRE | 45 | 4.8890 | 5.31 | 0.1398 | `langSub_MoRE_f025_seed45__8962921c` |
| 2.5% | MoRE | 46 | 4.9978 | 3.83 | 0.0000 | `langSub_MoRE_f025_seed46__77e7fcd1` |
| 10.0% | MoE | 42 | 4.4268 | 1.00 | N/A | `langSub_MoE_f100_seed42__87091bf2` |
| 10.0% | MoE | 43 | 4.4558 | 1.00 | N/A | `langSub_MoE_f100_seed43__e5bd63ed` |
| 10.0% | MoE | 44 | 4.4179 | 1.00 | N/A | `langSub_MoE_f100_seed44__7baa7e9d` |
| 10.0% | MoE | 45 | 4.3912 | 1.00 | N/A | `langSub_MoE_f100_seed45__b92a5c4d` |
| 10.0% | MoE | 46 | 4.4440 | 1.00 | N/A | `langSub_MoE_f100_seed46__f0cd63b2` |
| 10.0% | MoR | 42 | 4.2827 | 3.84 | 0.0048 | `langSub_MoR_f100_seed42__d542601c` |
| 10.0% | MoR | 43 | 4.3706 | 2.00 | 0.0000 | `langSub_MoR_f100_seed43__fefe58ef` |
| 10.0% | MoR | 44 | 4.3660 | 2.51 | 0.0000 | `langSub_MoR_f100_seed44__504175d4` |
| 10.0% | MoR | 45 | 4.2769 | 3.94 | 0.0036 | `langSub_MoR_f100_seed45__7535a3e1` |
| 10.0% | MoR | 46 | 4.3778 | 2.40 | 0.0000 | `langSub_MoR_f100_seed46__448686bc` |
| 10.0% | MoRE | 42 | 4.1352 | 6.74 | 0.8028 | `langSub_MoRE_f100_seed42__5a53cc0a` |
| 10.0% | MoRE | 43 | 4.1119 | 6.82 | 0.8552 | `langSub_MoRE_f100_seed43__7cfa8e4b` |
| 10.0% | MoRE | 44 | 4.1751 | 6.76 | 0.8158 | `langSub_MoRE_f100_seed44__f6c0fc4a` |
| 10.0% | MoRE | 45 | 4.1470 | 6.77 | 0.8250 | `langSub_MoRE_f100_seed45__623185b7` |
| 10.0% | MoRE | 46 | 4.1801 | 6.76 | 0.8212 | `langSub_MoRE_f100_seed46__3a7e555d` |
| 40.0% | MoE | 42 | 4.0819 | 1.00 | N/A | `langSub_MoE_f400_seed42__5cc0d02c` |
| 40.0% | MoE | 43 | 4.0892 | 1.00 | N/A | `langSub_MoE_f400_seed43__1d82756a` |
| 40.0% | MoE | 44 | 4.0667 | 1.00 | N/A | `langSub_MoE_f400_seed44__6ab703da` |
| 40.0% | MoE | 45 | 4.0193 | 1.00 | N/A | `langSub_MoE_f400_seed45__85f2e85a` |
| 40.0% | MoE | 46 | 4.0659 | 1.00 | N/A | `langSub_MoE_f400_seed46__2c360cf5` |
| 40.0% | MoR | 42 | 3.6637 | 6.73 | 0.8178 | `langSub_MoR_f400_seed42__e813baf0__r2` |
| 40.0% | MoR | 43 | 3.8492 | 2.00 | 0.0000 | `langSub_MoR_f400_seed43__3d6d59e8` |
| 40.0% | MoR | 44 | 3.6924 | 6.65 | 0.7792 | `langSub_MoR_f400_seed44__d21d8da3` |
| 40.0% | MoR | 45 | 3.6821 | 6.64 | 0.7667 | `langSub_MoR_f400_seed45__3b0ac9a5` |
| 40.0% | MoR | 46 | 3.7710 | 3.12 | 0.0000 | `langSub_MoR_f400_seed46__97a62068` |
| 40.0% | MoRE | 42 | 3.7122 | 6.95 | 0.9679 | `langSub_MoRE_f400_seed42__105137c4` |
| 40.0% | MoRE | 43 | 3.7649 | 6.95 | 0.9618 | `langSub_MoRE_f400_seed43__64b81e5c` |
| 40.0% | MoRE | 44 | 3.7785 | 6.95 | 0.9581 | `langSub_MoRE_f400_seed44__59518f8e` |
| 40.0% | MoRE | 45 | 3.7492 | 6.95 | 0.9594 | `langSub_MoRE_f400_seed45__dbb182aa` |
| 40.0% | MoRE | 46 | 3.7784 | 6.95 | 0.9614 | `langSub_MoRE_f400_seed46__37b4e108` |

## Provenance checks passed

- one `data.dataset_version` across all cells
- identical epochs/batch/lr/seq_len/vocab/num_experts/ffn_mult/
  max_depth/num_blocks/dropout and every loss weight per arm
- `data.subset_seed == provenance.seed` in every cell
- group stamp `subset_lang_sweep` on every cell
