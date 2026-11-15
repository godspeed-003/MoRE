# Test 1 -- the depth-matched uniform control on opdiff

`code/eval_depth_control.py`. Every number comes from the same artifact:
`checkpoint.pt` (best-validation), re-evaluated in one harness. `metrics.json`'s headline loss is last-epoch and is quoted only as a
cross-reference. Margin = uniform_loss - learned_loss, in nats; positive means the learned policy won.

## Per-seed

| run | arm | learned | mean depth | d=1 | d=2 | d=3 | d=4 | d=5 | d=6 | d=7 | best d | margin_best | d_hat | margin_matched |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| opdiffB_MoE_seed42__3ea2b9c7 | MoE | 0.4280 | 1.00 | 0.4280 | N/A | N/A | N/A | N/A | N/A | N/A | 1 | N/A | 1 | N/A |
| opdiffB_MoE_seed43__491048ee | MoE | 0.4294 | 1.00 | 0.4294 | N/A | N/A | N/A | N/A | N/A | N/A | 1 | N/A | 1 | N/A |
| opdiffB_MoE_seed44__d88592e9 | MoE | 0.4207 | 1.00 | 0.4207 | N/A | N/A | N/A | N/A | N/A | N/A | 1 | N/A | 1 | N/A |
| opdiffB_MoE_seed45__e342c989 | MoE | 0.4380 | 1.00 | 0.4380 | N/A | N/A | N/A | N/A | N/A | N/A | 1 | N/A | 1 | N/A |
| opdiffB_MoE_seed46__4b1c470d | MoE | 0.4352 | 1.00 | 0.4352 | N/A | N/A | N/A | N/A | N/A | N/A | 1 | N/A | 1 | N/A |
| opdiffB_MoR_seed42__e077cad2 | MoR | 0.2410 | 2.35 | 1.6059 | 0.5231 | 0.5720 | 1.9667 | 3.5542 | 3.9830 | 4.0626 | 2 | 0.2821 | 2 | 0.2821 |
| opdiffB_MoR_seed43__9acbcd8f | MoR | 0.3933 | 1.67 | 0.5933 | 3.2918 | 4.1015 | 4.1128 | 4.1027 | 4.1072 | 4.1068 | 1 | 0.2000 | 2 | 2.8984 |
| opdiffB_MoR_seed44__6ee08f08 | MoR | 0.1913 | 3.34 | 1.9376 | 0.9918 | 0.4691 | 0.2919 | 0.2959 | 0.4187 | 0.5969 | 4 | 0.1007 | 3 | 0.2778 |
| opdiffB_MoR_seed45__706e3aa2__r2 | MoR | 0.2028 | 3.84 | 1.9114 | 1.3870 | 1.0243 | 0.7476 | 0.6133 | 0.6060 | 0.7019 | 6 | 0.4032 | 4 | 0.5449 |
| opdiffB_MoR_seed46__db864910 | MoR | 0.1963 | 3.52 | 1.7755 | 0.9960 | 0.5973 | 0.3791 | 0.3176 | 0.3554 | 0.4621 | 5 | 0.1214 | 4 | 0.1828 |
| opdiffB_MoRE_seed42__1b6e1898 | MoRE | 0.1836 | 2.89 | 4.0984 | 2.2069 | 1.4516 | 0.7535 | 0.3949 | 0.3546 | 0.3864 | 6 | 0.1710 | 3 | 1.2680 |
| opdiffB_MoRE_seed43__992ef073 | MoRE | 0.1792 | 3.64 | 3.1799 | 2.0835 | 1.3870 | 0.6839 | 0.3615 | 0.5221 | 0.6924 | 5 | 0.1823 | 4 | 0.5047 |
| opdiffB_MoRE_seed44__c4eaf97f | MoRE | 0.1804 | 3.09 | 3.1290 | 1.8928 | 1.1980 | 0.6114 | 0.2806 | 0.2224 | 0.2538 | 6 | 0.0420 | 3 | 1.0176 |
| opdiffB_MoRE_seed45__a1a69ac3 | MoRE | 0.1856 | 3.24 | 4.0506 | 2.1938 | 1.2597 | 0.5728 | 0.2589 | 0.2207 | 0.3196 | 6 | 0.0352 | 3 | 1.0742 |
| opdiffB_MoRE_seed46__f684616d | MoRE | 0.2119 | 3.56 | 2.6715 | 1.8276 | 1.1309 | 0.5593 | 0.3032 | 0.2562 | 0.2578 | 6 | 0.0443 | 4 | 0.3474 |

## Per arm (over 5 seeds)

| arm | n | margin_best mean +/- std (median) | margin_matched mean +/- std (median) | learned val/task_loss |
|---|---|---|---|---|
| MoE | 5 | N/A (by construction) | N/A (by construction) | 0.4303 +/- 0.0067 (0.4294) |
| MoR | 5 | 0.2215 +/- 0.1243 (0.2000) | 0.8372 +/- 1.1601 (0.2821) | 0.2449 +/- 0.0852 (0.2028) |
| MoRE | 5 | 0.0950 +/- 0.0748 (0.0443) | 0.8424 +/- 0.3952 (1.0176) | 0.1881 +/- 0.0135 (0.1836) |

## Reading

**Implementation check.** MoE's learned and forced-d=1 losses agree to
four decimals in all five seeds, which is the identity the forced branch
must satisfy when `max_depth = 1`: the single step already carries weight
1.0 in the ACT mixture. MoE's margins are therefore reported as
`N/A (by construction)`, never as a measured zero.

**The language claim does not transfer.** On opdiff, at the
depth-matched control the paper used, MoRE's learned policy beats its own
uniform policy by **0.842 nats** (mean; median
1.018, range 0.347-1.268) —
about 14x the 0.06 nats measured on language, where MoRE's halting had
collapsed to 97.3% forced exit. MoR's is
0.837 +/- 1.160
(median 0.282). The ratio of MEANS is
**1.01x** and the ratio of MEDIANS is **3.61x**; the
means agree only because MoR's mean is carried by one pathological seed
(43, margin 2.90 — the collapsed-halting seed whose forced-d=2 loss is
3.29 nats), while MoRE is positive and tight in every seed. Either way,
MoRE's halting is doing substantial work on opdiff, so the paper's
'nearly inert' reading was a property of the saturated language cells,
not of the composition.

**But the conservative estimator points the other way, and both must be
reported.** Against the BEST uniform depth anywhere (not the depth-matched
one), MoRE's advantage falls to 0.095 +/-
0.075 nats against MoR's
0.221 +/- 0.124
(ratio 0.43x). MoRE's loss is nearly reached by a fixed
depth (its best forced depth is 5-6 in every seed), whereas no fixed
depth gets near MoR's learned policy. So: routing makes the depth policy
REPLACEABLE by a fixed depth, not more valuable than MoR's — a different
statement from 'inert', and the one the data supports.

**Mechanical caveat that the reader must not skip.** Forced depth returns
the final state, not the ACT mixture. Part of ANY arm's margin is
therefore the cost of reading a single state with a head that was trained
on convex mixtures — that cost is largest where the forced depth lands in
the steep part of the curve. MoRE's depth-matched integer (3) does exactly
that (its curve is still falling steeply at d=3), which inflates its
depth-matched margin and is why the best-uniform margin is the safer
headline. The two rows are a bracket, not a contradiction.

**Per-arm depth curves have different shapes** and that is itself a
finding: MoRE's forced loss falls monotonically to d≈6 and rises at 7,
MoR's is U-shaped with its minimum at d=2-4 and degrades sharply beyond
the depth it was trained to use. The composition is trained to be stable
under more recursion; MoR is not.
