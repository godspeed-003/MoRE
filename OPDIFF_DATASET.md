# The opdiff dataset — what it is, and why it is built this way

The generator is [`make_opdiff_dataset_v3.py`](make_opdiff_dataset_v3.py). This
file explains the data contract it writes, and it is the reference for anyone
reading a metric that came out of it.

```bash
python make_opdiff_dataset_v3.py --out opdiff_v3
python make_opdiff_dataset_v3.py --out opdiff_v3_smoke --train_rows_per_group 512 --eval_rows_per_group 128 --ood_rows_per_k 128
```

---

## Row design

**Row:** 16 packed examples of 24 tokens each, so **384 tokens per row**. The
examples are independent of each other — there is no cross-example structure
to exploit, and the model cannot read one example's answer out of another's.

**One example (24 tokens, the same layout for every operation):**

```
[OP] k4 k3 k2 k1 k0  a3 a2 a1 a0  b3 b2 b1 b0  [SEP]  r7 r6 r5 r4 r3 r2 r1 r0  [EOS]
 0    1..5            6..9         10..13         14     15..22                  23
```

- `OP` — the operation token (19…36).
- `k4…k0` — the **iteration count of the F operation, in binary**, five tokens
  drawn from the two hex-digit tokens the model has already learned (`3` = '0',
  `4` = '1'). For every non-F operation the slot holds five NULL tokens (2),
  so the layout is identical for all operations.
- `a…`, `b…` — the two 16-bit operands, four hex digits each.
- `SEP` — the separator (0).
- `r7…r0` — the answer, eight hex digits.
- `EOS` — end of example (1).

**Group purity:** all 16 examples in a row use operations from the same group,
and rows never mix groups. A row is therefore a single well-defined difficulty.

**No group token in the input:** the router has to infer the difficulty from
the operation token (and, for F, from the k slot). The group is saved
separately, for analysis and for the reward/label side only — never as an
input feature.

**Loss mask:** it covers the answer digits and the EOS in each example
(`position % 24 ≥ 15`), so 9 of the 24 tokens of each example are scored and
**144 of the 384 per row**. `meta.json` stores the full mask; the engine
consumes `mask[1:]`, because the targets are `x[:, 1:]` and the mask is
indexed by the position whose value is predicted.

| group | operations in a row |
|---|---|
| E1 | AND, OR, XOR, NOT |
| E2 | SHL, SHR, REV, MIN, MAX |
| E3 | ADD, SUB |
| E4 | MUL |
| E5 | DIV, MOD |
| E6 | SORT, MEDIAN |
| E7 | F(k) for k in TRAIN_K, plus GCD |

For E7, each row uses a single sub-op **and a single k**, because mixing F1 and
F16 in one row would blur the per-row difficulty.

---

## Per-row labels, for your loss and your reward

Each split has these files:

| file | contents |
|---|---|
| `*_x.npy` | the tokens, shape `(rows, 384)`, uint8 |
| `*_group.npy` | the expert group, 1 to 7 |
| `*_diff.npy` | a **nominal** difficulty. 1–6 for E1–E6. `7 + log2(k)` for F_k, and 9 for GCD. The OOD rows get `7 + log2(k)` as well: 8.58 (k=3), 9.58 (k=6), 10.17 (k=9), 10.46 (k=11), 10.81 (k=14), 11.09 (k=17), 11.32 (k=20). **Overwrite these with your calibrated ranking once your dense baselines are done.** |
| `*_k.npy` | the exact iteration count `k` of the row, 0 when the row is not an F row. This is the label the compute-versus-k analysis is read against. |
| `*_sub.npy` | the op token for E7 and OOD rows, −1 otherwise |

The token vocabulary is **37**: SEP 0, EOS 1, NULL 2, the 16 hex digits
3…18, and the 18 operation tokens 19…36.

---

## F1 to F16: the difficulty dial

Each F_k takes one 32-bit number, `x = A‖B` (the eight input hex digits read
as one number), and applies the same scrambling step k times:

```
f(x) = (x² + 12345) mod 4294967291
```

For example, F1 is one step, F2 is f(f(x)), and F16 is sixteen steps in a
row. The answer is the final x written as 8 hex digits.

**Why it works as a dial:** the input format and the output length are
identical for every k. Only the operation token changes (F1, F2, F4, F8 or
F16) and the number of sequential steps changes with it.

**No shortcut:** squaring plus a modulus by a prime is nonlinear, so the model
can't skip steps. A model without chain of thought has to chain k steps inside
its layers. A 1-step op might fit in a couple of layers; 16 steps might need
far more depth, or might not be learnable at all at this parameter count.

**Small example:** f(5) = 25 + 12345 = 12370. f(12370) = 153,029,245, which is
F2(5). The modulus only starts to matter once x² passes about 4.29 billion.

---

## The OOD part — and the problem with it

OOD means out-of-distribution. The training set only has F1, F2, F4, F5, F7,
F8, F10, F12, F13, F15 and F16.

**The flaw in the first version of this split.** It held F3, F6 and F12 —
*separate operation tokens* that never appear in training. Each of those
tokens' embeddings is therefore never trained, so the model has no way to know
that "F3" means three steps. Its accuracy on those rows would be near chance
for every architecture, and the earlier claim that this tests whether the
router extrapolates was too strong: it could not distinguish "cannot infer
the difficulty" from "cannot do the arithmetic".

**The fix (v3).** There is now ONE F operation token, and **k is written out
in the input**, in five binary tokens built from tokens the model has already
trained on. The model never sees the bit pattern for k = 3, but it can work
out what it means: the tokens `0` and `1` appear in every slot position
during training, and the bit patterns overlap (k = 3 combines bits the model
has seen together in k = 7 and k = 15).

So a held-out k is a real test of a real capability — reading a composition
out of a slot and spending the matching amount of computation — rather than a
test of whether an untrained embedding can be guessed.

| split | k values | what it asks |
|---|---|---|
| train / val / test | 1, 2, 4, 5, 7, 8, 10, 12, 13, 15, 16 | the seen k values |
| **ood_interp** | 3, 6, 9, 11, 14 | inside the trained range, never seen — the realistic target and the stronger claim |
| **ood_extrap** | 17, 20 | beyond the trained range — harder, because bit 4 only ever appeared alone (in k = 16) during training. Treat it as a stretch result. The baselines will probably fail here too, so it shows a gap even if you do not fully succeed. |

**The claim this supports:** on unseen k, the model gives the right answers and
spends compute that scales with k.

**Caveats, and what to do about them.**

- *Learning k is part of the task.* The model has to learn what the binary
  slot means from only 11 k values. **If it fails on ood_interp, check whether
  it also fails on the SEEN k values before blaming your architecture.** The
  run publishes the per-k accuracy and the per-k exact match on the validation
  split (`val/accuracy_by_op/F<k>`, `val/exact_match_by_op/F<k>`) precisely so
  that comparison is one lookup: if the seen F rows are also unsolved, the
  architecture is not what is failing. If it is, add more training k values.
- *High k may be too hard for any model.* F with k ≥ 13 without chain of
  thought may stay near chance for your model and for the baselines alike. The
  dense-baseline calibration will show where that point is. If it is too low,
  shrink TRAIN_K and the held-out lists to a lower range — they are constants
  at the top of the generator.

---

## Measured properties of the generated dataset

Full generation (`--train_rows_per_group 42000 --eval_rows_per_group 1000
--ood_rows_per_k 1000`):

| split | rows | what it is |
|---|---|---|
| train | 294,000 | 42,000 rows per group × 7 groups |
| val | 7,000 | 1,000 per group, disjoint inputs from train |
| test | 7,000 | 1,000 per group, disjoint from train and val |
| ood_interp | 5,000 | 1,000 per held-out k (5 k values), all group E7 |
| ood_extrap | 2,000 | 1,000 per held-out k (2 k values), all group E7 |

- **112,896,000 training tokens** (294,000 × 384) — a little above WikiText-103.
- **vocabulary 37**, row length 384, example length 24, 16 examples per row.
- Generation takes about 10 seconds.
- 144 scored positions per row (9 per example × 16 examples).
- Verified over a 104k-example sample: **0 wrong answers and 0 row-purity
  violations**, and the decoded k always matched the label. Val and test do not
  overlap with train.
