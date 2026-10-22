"""
Op-difficulty benchmark v4: the F operation is REDESIGNED so that one iteration
is LEARNABLE, and the difficulty dial is the number of iterations.

WHY v4 EXISTS, and it is a measurement rather than a guess. The v3 F operation
was f(x) = (x^2 + 12345) mod 4294967291 on a 32-bit x = A||B. The first
full-scale opdiff run (MoR at ffn_mult 4, the full 294,000-row dataset, 3
epochs = 18,375 optimizer steps) measured the F rows at 0.161-0.172 per-digit
accuracy and 0.0 per-example exact match for EVERY k -- including k = 1, the
easiest one -- while the other six groups reached 0.850 per-digit accuracy.

So the F operation was not "hard at high k": it was not LEARNED AT ALL, at any
k, by a 4 M-parameter model in 3 epochs. The reason is the DATA, not the
architecture: one iteration of a 32-bit modular squaring is a 32x32-bit
multiplication, and learning a 32-bit multiplication from data is a known-hard
problem for a transformer without chain of thought. No amount of extra
training at this budget fixes it, which is why the failure is FLAT in k
instead of worsening with it. A difficulty dial whose dial is stuck at zero
cannot be measured, so the operation itself has to change.

WHAT CHANGED, and the three properties the new operation has to have.

  1. ONE ITERATION MUST BE LEARNABLE. The new F is a per-nibble nonlinear
     recurrence over the EIGHT nibbles of the state x = A||B:
         d'_i = ((d_i XOR d_{(i+1) mod 8}) + 3) mod 16
     Each output digit is a function of TWO INPUT DIGITS and a constant, so
     the per-digit rule is a 256-case table -- the same size as "learn the
     addition of two 4-bit numbers", which transformers do learn. The state
     space is 16^8 = 4.3e9, so the operation cannot be memorised.

  2. THE k-FOLD COMPOSITION MUST NOT BE SHORTCUT-ABLE. XOR is nonlinear over
     Z_16 and the digit additions carry, so the map has no closed form and no
     small algebraic form; the only way to produce d^(k) is to compose the
     step k times. The k-fold map is a fixed function on 4.3e9 states, which
     is far too large to memorise, so a model that wants the answer for a
     given k has to UNROLL.

  3. ONE ITERATION MUST COST ROUGHLY ONE LAYER, so that the dial spans the
     model's depth budget instead of overshooting it. The eight digits update
     in PARALLEL (every d'_i depends only on the PREVIOUS state), so one
     iteration is one layer of the recurrence and k iterations need k layers.
     With max_depth = 7 the frontier is therefore around k = 5..7, and the
     k lists are chosen from the MEASURED frontier (see dense calibration)
     rather than from a guess.

EVERYTHING ELSE IS UNCHANGED, and that is the point: the token layout (24
tokens per example, 16 examples per row, 384 tokens per row), the loss mask
(position % 24 >= 15, i.e. the answer digits and the EOS), the vocabulary
(37), the operation tokens, the seven expert groups, the E7 structure (one
sub-op and one k per row), the split definitions, and the three k lists all
survive. The port's manifest, the dataset class and the 35-wide
operation-instance axis change only where the k values change.

Row = 16 packed examples x 24 tokens = 384 tokens. Every row belongs to exactly one expert group.
Example block (24 tokens, identical layout for every op):
  [OP] k4 k3 k2 k1 k0  a3 a2 a1 a0  b3 b2 b1 b0  [SEP]  r7..r0  [EOS]
   0    1..5            6..9         10..13         14     15..22  23
  - k-slot: 5 binary tokens (tokens '0'/'1') for the F op; 5 NULL tokens for every other op.
  - F op:  the state is the eight nibbles of x = A||B; one iteration is
           d'_i = ((d_i XOR d_{(i+1) mod 8}) + 3) mod 16, applied k times.
  - loss only where (position % 24) >= 15 (answer digits + EOS)

Expert groups (a row never mixes groups; E7 rows use ONE sub-op and ONE k):
  E1 AND OR XOR NOT | E2 SHL SHR REV MIN MAX | E3 ADD SUB | E4 MUL | E5 DIV MOD | E6 SORT MEDIAN
  E7 F(k) for k in TRAIN_K, plus GCD

Splits: train / val / test (seen k) ; ood_interp (held-out k inside the trained range)
        ood_extrap (k beyond the trained range).

Files (.npy, mmap-friendly) per split: x (rows,384) uint8, group (rows,), diff (rows,) float32,
  k (rows,) int16 [0 if not F], sub (rows,) int16 [op token for E7 rows, -1 otherwise]. Plus meta.json.
"""
import argparse, hashlib, json, os
import numpy as np

SEP, EOS, NULL, HEX0, OP0 = 0, 1, 2, 3, 19
K_BITS = 5
EX_LEN = 1 + K_BITS + 4 + 4 + 1 + 8 + 1      # 24
EX_PER_ROW = 16
ROW_LEN = EX_LEN * EX_PER_ROW                # 384
ANS_START = 1 + K_BITS + 8 + 1               # 15
M, C = 4294967291, 12345
U = np.uint64
MASK = U(0xFFFF)

# T-OD.10: the k lists. THE FRONTIER IS MEASURED, NOT ASSUMED -- the dense
# calibration on this generator says at which k a plain transformer still
# learns the operation, and the lists are chosen so that (a) the trained k
# values are all inside the frontier, (b) the interpolation k values are
# inside the trained RANGE but never seen, and (c) the extrapolation k
# values are beyond the trained range AND beyond the depth budget, which is
# what makes them a test of reading the slot rather than of doing the work.
#
# The defaults below are the MEASURED frontier (see the changelog): a plain
# one-block transformer learns k = 1..4, fails at k = 5, and is hopeless past
# it. So the trained values are the four learnable ones, the interpolation
# value is the first unlearnable-but-in-range one, and the extrapolation
# values are past the trained range entirely.
TRAIN_K = [1, 2, 3, 4]
INTERP_K = [5]
EXTRAP_K = [6, 7]

# The F operation's per-iteration twist. Fixed, so the k-fold map is a fixed
# function of the state -- and the nonlinearity (XOR plus a carry) is what
# makes that function have no small closed form.
F_TWIST = 3


def to_digits(v, n):
    sh = (4 * np.arange(n - 1, -1, -1)).astype(U)
    return ((v[:, None] >> sh) & U(0xF)).astype(np.uint8)


def num(f):
    return lambda A, B, k: to_digits(f(A, B), 8)


def _sorted8(A, B):
    return np.sort(np.concatenate([to_digits(A, 4), to_digits(B, 4)], 1), axis=1)


def _median(A, B, k):
    out = np.zeros((len(A), 8), np.uint8)
    out[:, 7] = _sorted8(A, B)[:, 3]
    return out


def _rev(A, B, k):
    out = np.zeros((len(A), 8), np.uint8)
    out[:, 4:] = to_digits(A, 4)[:, ::-1]
    return out


def _F(A, B, k):
    """The redesigned F: a per-nibble nonlinear recurrence, iterated k times.

    The state is the EIGHT nibbles of x = A||B, in the same order the operands
    are written (A's most significant nibble first, then B's). One iteration
    maps every nibble from the PREVIOUS state:
        d'_i = ((d_i XOR d_{(i+1) mod 8}) + F_TWIST) mod 16
    so the eight digits update in PARALLEL (one layer of the recurrence) and
    each output digit is a function of two INPUT digits -- a 256-case table.

    The answer is the state after k iterations, written as eight hex digits.
    """
    st = np.concatenate([to_digits(A, 4), to_digits(B, 4)], axis=1).astype(np.int64)
    for _ in range(k):
        nxt = np.empty_like(st)
        for i in range(8):
            nb = np.roll(st, -1, axis=1)[:, i]      # d_{(i+1) mod 8}
            nxt[:, i] = ((st[:, i] ^ nb) + F_TWIST) & 0xF
        st = nxt
    return st.astype(np.uint8)


OPS = {
    "AND": num(lambda A, B: A & B), "OR": num(lambda A, B: A | B), "XOR": num(lambda A, B: A ^ B),
    "NOT": num(lambda A, B: (~A) & MASK),
    "SHL": num(lambda A, B: (A << (B & U(15))) & MASK), "SHR": num(lambda A, B: A >> (B & U(15))),
    "REV": _rev, "MIN": num(np.minimum), "MAX": num(np.maximum),
    "ADD": num(lambda A, B: A + B), "SUB": num(lambda A, B: (A - B) & MASK),
    "MUL": num(lambda A, B: A * B),
    "DIV": num(lambda A, B: A // B), "MOD": num(lambda A, B: A % B),
    "SORT": lambda A, B, k: _sorted8(A, B), "MEDIAN": _median,
    "GCD": num(np.gcd),
    "F": _F,
}
TOK = {n: OP0 + i for i, n in enumerate(OPS)}

GROUPS = {1: ["AND", "OR", "XOR", "NOT"], 2: ["SHL", "SHR", "REV", "MIN", "MAX"], 3: ["ADD", "SUB"],
          4: ["MUL"], 5: ["DIV", "MOD"], 6: ["SORT", "MEDIAN"]}   # E7 built from TRAIN_K + GCD


def nominal(name, k, g):
    if name == "F":
        return 7 + float(np.log2(k))
    if name == "GCD":
        return 9.0
    return float(g)


def k_tokens(k, n):
    t = np.full((n, K_BITS), NULL, np.uint8)
    if k:
        t[:] = [HEX0 + ((k >> i) & 1) for i in reversed(range(K_BITS))]
    return t


def sample_ab(rng, n):
    A = rng.integers(0, 1 << 16, n, dtype=np.int64)
    bits = rng.integers(1, 17, n)
    lo = np.left_shift(1, bits - 1).astype(np.int64)
    B = lo + (rng.random(n) * lo).astype(np.int64)
    return A, B


def sample_unique(rng, n, forbid):
    A, B = sample_ab(rng, n)
    if forbid is not None and len(forbid):
        for _ in range(30):
            bad = np.isin((A << 16) | B, forbid)
            if not bad.any():
                break
            a2, b2 = sample_ab(rng, int(bad.sum()))
            A[bad], B[bad] = a2, b2
    return A, B


def gen_ops(rng, name, k, n, forbid, used):
    label = f"F{k}" if name == "F" else name
    A, B = sample_unique(rng, n, forbid.get(label))
    used.setdefault(label, []).append((A << 16) | B)
    with np.errstate(over="ignore"):
        ans = OPS[name](A.astype(U), B.astype(U), k)
    b = np.empty((n, EX_LEN), np.uint8)
    b[:, 0] = TOK[name]
    b[:, 1:1 + K_BITS] = k_tokens(k, n)
    b[:, 6:10] = to_digits(A.astype(U), 4) + HEX0
    b[:, 10:14] = to_digits(B.astype(U), 4) + HEX0
    b[:, 14] = SEP
    b[:, ANS_START:ANS_START + 8] = ans + HEX0
    b[:, EX_LEN - 1] = EOS
    return b


def gen_homog(rng, subs, n_rows, forbid, used, g=7):
    X = np.empty((n_rows, ROW_LEN), np.uint8)
    diff = np.empty(n_rows, np.float32); K = np.zeros(n_rows, np.int16); sub = np.full(n_rows, -1, np.int16)
    assign = rng.permutation(np.tile(np.arange(len(subs)), n_rows // len(subs) + 1)[:n_rows])
    for i, (name, k) in enumerate(subs):
        idx = np.where(assign == i)[0]
        if not len(idx):
            continue
        X[idx] = gen_ops(rng, name, k, len(idx) * EX_PER_ROW, forbid, used).reshape(len(idx), ROW_LEN)
        diff[idx] = nominal(name, k, g); K[idx] = k; sub[idx] = TOK[name]
    return X, np.full(n_rows, g, np.uint8), diff, K, sub


def gen_mixed(rng, g, n_rows, forbid, used):
    ops = GROUPS[g]
    n_ex = n_rows * EX_PER_ROW
    assign = rng.permutation(np.tile(np.arange(len(ops)), n_ex // len(ops) + 1)[:n_ex])
    E = np.empty((n_ex, EX_LEN), np.uint8)
    for i, name in enumerate(ops):
        idx = np.where(assign == i)[0]
        E[idx] = gen_ops(rng, name, 0, len(idx), forbid, used)
    return (E.reshape(n_rows, ROW_LEN), np.full(n_rows, g, np.uint8), np.full(n_rows, g, np.float32),
            np.zeros(n_rows, np.int16), np.full(n_rows, -1, np.int16))


def finish(rng, parts):
    X, G, D, K, S = (np.concatenate(z) for z in zip(*parts))
    p = rng.permutation(len(X))
    return (X[p], G[p], D[p], K[p], S[p])


def make_main(rng, rows_per_group, forbid):
    used, parts = {}, []
    for g in GROUPS:
        parts.append(gen_mixed(rng, g, rows_per_group, forbid, used))
    subs = [("F", k) for k in TRAIN_K] + [("GCD", 0)]
    parts.append(gen_homog(rng, subs, rows_per_group, forbid, used))
    return finish(rng, parts), {k: np.concatenate(v) for k, v in used.items()}


def make_ood(rng, ks, rows_per_k):
    subs = [("F", k) for k in ks]
    return finish(rng, [gen_homog(rng, subs, rows_per_k * len(ks), {}, {})])


def _sha(a):
    """SHA-256 of an array's bytes, in the order it is written to disk."""
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def save(out, name, arrs):
    """Write the five arrays of one split and return their content hashes.

    The hashes are what make `dataset_version` a statement about the BYTES
    rather than about the generator call: `meta.json` on its own is entirely
    parameter-derived, so two generations that sampled different rows would
    carry the same id.
    """
    digests = {}
    for suffix, a in zip(["x", "group", "diff", "k", "sub"], arrs):
        np.save(os.path.join(out, f"{name}_{suffix}.npy"), a)
        digests[suffix] = _sha(a)
    return digests


def dataset_version(all_digests, rows, train_rows_per_group, eval_rows_per_group, seed):
    """One string identifying THIS generation: content hashes + the call that made it."""
    h = hashlib.sha256()
    for split in sorted(all_digests):
        for suffix in sorted(all_digests[split]):
            h.update(f"{split}:{suffix}:{all_digests[split][suffix]}".encode())
    h.update(f"rows={rows}".encode())
    h.update(f"train_rows_per_group={train_rows_per_group}".encode())
    h.update(f"eval_rows_per_group={eval_rows_per_group}".encode())
    h.update(f"seed={seed}".encode())
    return f"opdiff-v4-{h.hexdigest()[:12]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="opdiff_v4")
    ap.add_argument("--train_rows_per_group", type=int, default=42_000)  # 294k rows ~ 112.9M tokens
    ap.add_argument("--eval_rows_per_group", type=int, default=1_000)
    ap.add_argument("--ood_rows_per_k", type=int, default=1_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train_k", type=int, nargs="*", default=None,
                    help="override the TRAINED k values. Used to build the "
                         "CALIBRATION dataset, which trains on every k so that "
                         "a model's per-k accuracy measures whether the "
                         "OPERATION is learnable at that k rather than whether "
                         "the model has seen that k. The final train/interp/"
                         "extrap split is chosen FROM the measured frontier.")
    ap.add_argument("--interp_k", type=int, nargs="*", default=None)
    ap.add_argument("--extrap_k", type=int, nargs="*", default=None)
    a = ap.parse_args()
    if a.train_k is not None:
        TRAIN_K[:] = list(a.train_k)
    if a.interp_k is not None:
        INTERP_K[:] = list(a.interp_k)
    if a.extrap_k is not None:
        EXTRAP_K[:] = list(a.extrap_k)
    rng = np.random.default_rng(a.seed)
    os.makedirs(a.out, exist_ok=True)

    tr, used_tr = make_main(rng, a.train_rows_per_group, {})
    va, used_va = make_main(rng, a.eval_rows_per_group, used_tr)
    forbid = {k: np.concatenate([used_tr[k], used_va[k]]) for k in used_tr}
    te, _ = make_main(rng, a.eval_rows_per_group, forbid)
    oi = make_ood(rng, INTERP_K, a.ood_rows_per_k)
    oe = make_ood(rng, EXTRAP_K, a.ood_rows_per_k)
    digests = {}
    for n, s in [("train", tr), ("val", va), ("test", te), ("ood_interp", oi), ("ood_extrap", oe)]:
        digests[n] = save(a.out, n, s)

    mask = ((np.arange(ROW_LEN) % EX_LEN) >= ANS_START).astype(np.uint8)
    rows = {n: len(s[0]) for n, s in [("train", tr), ("val", va), ("test", te), ("ood_interp", oi), ("ood_extrap", oe)]}
    meta = dict(
        vocab_size=OP0 + len(OPS), row_len=ROW_LEN, example_len=EX_LEN, examples_per_row=EX_PER_ROW,
        tokens=dict(SEP=SEP, EOS=EOS, NULL=NULL, hex0=HEX0, op0=OP0, k_bits=K_BITS), op_tokens=TOK,
        groups={**{f"E{g}": v for g, v in GROUPS.items()}, "E7": [f"F(k) k in TRAIN_K", "GCD"]},
        train_k=TRAIN_K, ood_interp_k=INTERP_K, ood_extrap_k=EXTRAP_K,
        f_operation="per-nibble nonlinear recurrence: d'_i = ((d_i XOR d_{(i+1) mod 8}) + 3) mod 16, "
                    "iterated k times over the eight nibbles of x = A||B",
        f_twist=F_TWIST,
        loss_mask_note="mask[t]=1 where t%24>=15 (answer + EOS); targets are x[:,1:], use mask[1:]",
        loss_mask=mask.tolist(),
        rows=rows,
        train_tokens=int(len(tr[0]) * ROW_LEN),
        content_sha256=digests,
        dataset_version=dataset_version(digests, rows, a.train_rows_per_group,
                                       a.eval_rows_per_group, a.seed),
        generator_args=dict(train_rows_per_group=a.train_rows_per_group,
                            eval_rows_per_group=a.eval_rows_per_group, seed=a.seed))
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=1)
    print(meta["rows"], "vocab", meta["vocab_size"], "train tokens", meta["train_tokens"],
          "train_k", TRAIN_K, "interp", INTERP_K, "extrap", EXTRAP_K)


if __name__ == "__main__":
    main()
