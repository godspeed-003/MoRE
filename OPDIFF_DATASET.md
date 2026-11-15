# The opdiff dataset — what it is, and why it is built this way

The generator is `make_opdiff_dataset_v4.py` for the current design;
`make_opdiff_dataset_v3.py` is kept alongside it because it is the record of
the ORIGINAL F operation, and the measurement on it is what justifies the
change (see "The F operation, and why it changed" at the end). This file
explains the data contract, and it is the reference for anyone reading a
metric that came out of a run.

```bash
python make_opdiff_dataset_v4.py --out opdiff_v4
python make_opdiff_dataset_v4.py --out opdiff_v4_smoke --train_rows_per_group 64 --eval_rows_per_group 64 --ood_rows_per_k 64
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
F4 in one row would blur the per-row difficulty.

---

## Per-row labels, for your loss and your reward

Each split has these files:

| file | contents |
|---|---|
| `*_x.npy` | the tokens, shape `(rows, 384)`, uint8 |
| `*_group.npy` | the expert group, 1 to 7 |
| `*_diff.npy` | a **nominal** difficulty. 1–6 for E1–E6. `7 + log2(k)` for F_k, and 9 for GCD. The OOD rows get `7 + log2(k)` as well: 8.58 (k=3), 9.58 (k=6), 10.17 (k=9), 10.46 (k=11), 10.81 (k=14) in the v3 design. **Overwrite these with your calibrated ranking once your dense baselines are done.** |
| `*_k.npy` | the exact iteration count `k` of the row, 0 when the row is not an F row. This is the label the compute-versus-k analysis is read against. |
| `*_sub.npy` | the op token for E7 and OOD rows, −1 otherwise |

The token vocabulary is **37**: SEP 0, EOS 1, NULL 2, the 16 hex digits
3…18, and the 18 operation tokens 19…36.

---

## The F operation, and why it changed

**The v3 design (kept for the record).** `f(x) = (x² + 12345) mod 4294967291`
on a 32-bit `x = A‖B`, applied k times. The answer is the final x as eight hex
digits. It was a clean dial: the input format and the output length are
identical for every k, only the op token and the number of sequential steps
change, and squaring plus a modulus by a prime is nonlinear so the model
cannot skip steps.

**Why it had to change.** The first full-scale run (MoR at ffn_mult 4, the
full 294,000-row dataset, 3 epochs = 18,375 optimizer steps) measured the F
rows at **0.161–0.172 per-digit accuracy and 0.0 per-example exact match for
EVERY k — including k = 1** — while the other six groups reached 0.850
per-digit accuracy. So the F operation was not "hard at high k": it was not
learned AT ALL, at any k. The reason is the data, not the architecture: one
iteration of a 32-bit modular squaring is a 32×32-bit multiplication, and
learning a 32-bit multiplication from data is a known-hard problem for a
transformer without chain of thought. The failure is flat in k because there
is no gradient signal in those rows at all. **A difficulty dial whose dial is
stuck at zero cannot be measured**, so the operation had to change. A dense
reference (one block, no routing, no recursion) is what confirms that this is
the data and not the architecture.

**The v4 design.** The F operation is a per-nibble nonlinear recurrence over
the eight nibbles of the state `x = A‖B`:

```
d'_i = ((d_i XOR d_{(i+1) mod 8}) + 3) mod 16        (one iteration, i = 0..7)
```

applied k times, with the answer being the state after k iterations. It has the
three properties the study needs:

1. **One iteration is LEARNABLE.** Each output digit is a function of two
   *input* digits and a constant — a 256-case table, the size of "learn the
   addition of two 4-bit numbers", which transformers do learn. The state space
   is 16⁸ = 4.3 × 10⁹, so the operation cannot be memorised.
2. **The k-fold composition is NOT SHORTCUT-ABLE.** XOR is nonlinear over
   Z₁₆ and the digit additions carry, so there is no closed form and no small
   algebraic form; the composed map is a fixed function on 4.3 × 10⁹ states,
   far too large to memorise. A model that wants the answer for a given k has
   to UNROLL.
3. **One iteration costs roughly ONE LAYER**, because the eight digits update
   in PARALLEL (every `d'_i` depends only on the previous state). So k
   iterations need about k layers, and **the dial spans the model's depth
   budget instead of overshooting it** — which is why the k lists are set from
   the MEASURED frontier rather than from a guess.

Everything else survives the change: the token layout, the loss mask, the
vocabulary, the operation tokens, the seven groups, the E7 structure, the
split definitions, the per-row label files, and the operation-instance axis
the code publishes (one entry per (operation, k), so the per-k depth table
and the compute-versus-k analysis are unchanged).

---

## The OOD part — and the problem with it

OOD means out-of-distribution. The training set has only the trained k values;
the held-out ones are split into two parts, and they are TWO DIFFERENT
QUESTIONS, which is why they are two splits:

| split | what it asks |
|---|---|
| `ood_interp` | a k INSIDE the trained range that was never seen. The model has to read a composition out of the k slot and apply it — the bit patterns overlap with trained ones, so the answer is inferable. This is the realistic target and the stronger claim. |
| `ood_extrap` | a k BEYOND the trained range. Harder, because the high bit appeared only alone (in k = 16) during training. Treat it as a stretch result. The baselines will probably fail here too, so it shows a gap even if you do not fully succeed. |

**The flaw in the FIRST version of this split, and why it is not repeated
here.** v2 gave F3, F6 and F12 their own operation tokens. A token that never
appears in training has an embedding that is never trained, so no model could
know that "F3" means three iterations, and the accuracy on those rows would be
near chance for every architecture. The claim it was meant to test ("does the
router extrapolate to unseen iteration counts?") could not have been told apart
from "the model cannot do F at all". **In v3 and v4 there is ONE F operation
token and k is written out in the input**, so a held-out k is inferable from
trained tokens and the test is of a real capability.

**The claim this supports:** on unseen k, the model gives the right answers and
spends compute that scales with k.

**Caveats, and what to do about them.**

- *Learning k is part of the task.* The model has to learn what the binary slot
  means from the trained k values. **If it fails on ood_interp, check whether
  it also fails on the SEEN k values before blaming your architecture.** The
  run publishes the per-k accuracy and the per-k exact match on the
  validation split (`val/accuracy_by_op/F<k>`,
  `val/exact_match_by_op/F<k>`) precisely so that comparison is one lookup:
  if the seen F rows are also unsolved, the architecture is not what is
  failing.
- *High k may be too hard for any model.* The dense calibration measures
  where that point is. If the frontier sits below the trained k values, the
  k lists have to shrink — they are constants at the top of the generator.

---

## Measured properties of the generated dataset

Full generation (`--train_rows_per_group 42000 --eval_rows_per_group 1000
--ood_rows_per_k 1000`):

| split | rows | what it is |
|---|---|---|
| train | 294,000 | 42,000 rows per group × 7 groups |
| val | 7,000 | 1,000 per group, disjoint inputs from train |
| test | 7,000 | 1,000 per group, disjoint from train and val |
| ood_interp | 1,000 per held-out k | all group E7 |
| ood_extrap | 1,000 per held-out k | all group E7 |

- **112,896,000 training tokens** (294,000 × 384) — a little above WikiText-103.
- **vocabulary 37**, row length 384, example length 24, 16 examples per row.
- Generation takes about 10 seconds.
- 144 scored positions per row (9 per example × 16 examples).
- Verified over a 300-row sample per split: **0 wrong answers, 0 row-purity
  violations, 0 group violations**, and the decoded k always matched the
  label. Val and test do not overlap with train.
