"""
Op-difficulty benchmark v2: every ROW belongs to exactly one expert group (E1..E7).

Row = 16 packed examples x 19 tokens = 304 tokens, no BOS/PAD, no group token in the input
(the router must infer difficulty from the op token itself).

Example block (19 tokens):
  [OP] a3 a2 a1 a0  b3 b2 b1 b0  [SEP]  r7 r6 r5 r4 r3 r2 r1 r0  [EOS]
   0    1..4         5..8          9       10..17                  18
 - A, B: 16-bit, 4 hex digits (identical format and distribution for every op)
 - answer: always 8 hex digits, zero-padded, no chain of thought
 - loss only where (position % 19) >= 10  (answer + EOS)

Expert groups (rows never mix groups):
  E1 AND OR XOR NOT          E2 SHL SHR REV MIN MAX     E3 ADD SUB      E4 MUL
  E5 DIV MOD                 E6 SORT MEDIAN
  E7 F1 F2 F4 F8 F16 GCD     (each E7 row uses ONE sub-op, so row difficulty is exact)
F_k: x -> (x*x + 12345) mod 4294967291 applied k times, x = A||B (32-bit).
OOD split: F3 F6 F12 rows (iteration counts never seen in training).

Files written to --out (all .npy, mmap-friendly):
  {train,val,test,ood}_x.npy      (rows, 304) uint8 tokens
  {split}_group.npy               (rows,) uint8   expert group 1..7  (for analysis/reward, NOT an input)
  {split}_diff.npy                (rows,) float32 nominal difficulty (placeholder - calibrate with dense baselines)
  {split}_sub.npy                 (rows,) int16   op token for E7/OOD rows, -1 otherwise
  meta.json
"""
import argparse, json, os
import numpy as np

SEP, EOS, HEX0, OP0 = 0, 1, 2, 18
EX_LEN, EX_PER_ROW = 19, 16
ROW_LEN = EX_LEN * EX_PER_ROW  # 304
M, C = 4294967291, 12345
U = np.uint64
MASK = U(0xFFFF)

def to_digits(v, n):
    sh = (4 * np.arange(n - 1, -1, -1)).astype(U)
    return ((v[:, None] >> sh) & U(0xF)).astype(np.uint8)

def num(f):
    return lambda A, B: to_digits(f(A, B), 8)

def _sorted8(A, B):
    return np.sort(np.concatenate([to_digits(A, 4), to_digits(B, 4)], 1), axis=1)

def _median(A, B):
    out = np.zeros((len(A), 8), np.uint8)
    out[:, 7] = _sorted8(A, B)[:, 3]
    return out

def _rev(A, B):
    out = np.zeros((len(A), 8), np.uint8)
    out[:, 4:] = to_digits(A, 4)[:, ::-1]
    return out

def fk(k):
    def f(A, B):
        x = (A << U(16)) | B
        for _ in range(k):
            x = (x * x + U(C)) % U(M)
        return to_digits(x, 8)
    return f

OPS = {
    "AND": num(lambda A, B: A & B), "OR": num(lambda A, B: A | B), "XOR": num(lambda A, B: A ^ B),
    "NOT": num(lambda A, B: (~A) & MASK),
    "SHL": num(lambda A, B: (A << (B & U(15))) & MASK), "SHR": num(lambda A, B: A >> (B & U(15))),
    "REV": _rev, "MIN": num(np.minimum), "MAX": num(np.maximum),
    "ADD": num(lambda A, B: A + B), "SUB": num(lambda A, B: (A - B) & MASK),
    "MUL": num(lambda A, B: A * B),
    "DIV": num(lambda A, B: A // B), "MOD": num(lambda A, B: A % B),
    "SORT": lambda A, B: _sorted8(A, B), "MEDIAN": _median,
    "GCD": num(np.gcd),
    **{f"F{k}": fk(k) for k in (1, 2, 4, 8, 16)},
}
OOD_OPS = {f"F{k}": fk(k) for k in (3, 6, 12)}
ALL_OPS = {**OPS, **OOD_OPS}
TOK = {n: OP0 + i for i, n in enumerate(ALL_OPS)}

GROUPS = {1: ["AND", "OR", "XOR", "NOT"], 2: ["SHL", "SHR", "REV", "MIN", "MAX"], 3: ["ADD", "SUB"],
          4: ["MUL"], 5: ["DIV", "MOD"], 6: ["SORT", "MEDIAN"],
          7: ["F1", "F2", "F4", "F8", "F16", "GCD"]}
HOMOGENEOUS = {7}                       # groups whose rows use a single sub-op
def nominal(name, g):
    if name.startswith("F"):
        return 7 + float(np.log2(int(name[1:])))
    if name == "GCD":
        return 9.0
    return float(g)

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

def block(op_tok, A, B, ans):
    b = np.empty((len(A), EX_LEN), np.uint8)
    b[:, 0] = op_tok
    b[:, 1:5] = to_digits(A.astype(U), 4) + HEX0
    b[:, 5:9] = to_digits(B.astype(U), 4) + HEX0
    b[:, 9] = SEP
    b[:, 10:18] = ans + HEX0
    b[:, 18] = EOS
    return b

def gen_ops(rng, name, n, forbid, used):
    A, B = sample_unique(rng, n, forbid.get(name))
    used.setdefault(name, []).append((A << 16) | B)
    with np.errstate(over="ignore"):
        ans = ALL_OPS[name](A.astype(U), B.astype(U))
    return block(TOK[name], A, B, ans)

def gen_group(rng, g, n_rows, forbid, used, ops=None):
    ops = ops or GROUPS[g]
    X = np.empty((n_rows, ROW_LEN), np.uint8)
    diff = np.empty(n_rows, np.float32)
    sub = np.full(n_rows, -1, np.int16)
    if g in HOMOGENEOUS:
        assign = rng.permutation(np.tile(np.arange(len(ops)), n_rows // len(ops) + 1)[:n_rows])
        for i, name in enumerate(ops):
            idx = np.where(assign == i)[0]
            if not len(idx):
                continue
            X[idx] = gen_ops(rng, name, len(idx) * EX_PER_ROW, forbid, used).reshape(len(idx), ROW_LEN)
            diff[idx] = nominal(name, g); sub[idx] = TOK[name]
    else:
        n_ex = n_rows * EX_PER_ROW
        assign = rng.permutation(np.tile(np.arange(len(ops)), n_ex // len(ops) + 1)[:n_ex])
        E = np.empty((n_ex, EX_LEN), np.uint8)
        for i, name in enumerate(ops):
            idx = np.where(assign == i)[0]
            E[idx] = gen_ops(rng, name, len(idx), forbid, used)
        X[:] = E.reshape(n_rows, ROW_LEN)
        diff[:] = g
    return X, np.full(n_rows, g, np.uint8), diff, sub

def make_split(rng, rows_per_group, forbid, ood=False):
    used, parts = {}, []
    if ood:
        parts.append(gen_group(rng, 7, rows_per_group * 3, {}, used, ops=list(OOD_OPS)))
    else:
        for g in GROUPS:
            parts.append(gen_group(rng, g, rows_per_group, forbid, used))
    X, G, D, S = (np.concatenate(z) for z in zip(*parts))
    p = rng.permutation(len(X))
    return (X[p], G[p], D[p], S[p]), {k: np.concatenate(v) for k, v in used.items()}

def save(out, name, arrs):
    for suffix, a in zip(["x", "group", "diff", "sub"], arrs):
        np.save(os.path.join(out, f"{name}_{suffix}.npy"), a)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="opdiff_v2")
    ap.add_argument("--train_rows_per_group", type=int, default=52_000)  # 364k rows ~ 110.7M tokens
    ap.add_argument("--eval_rows_per_group", type=int, default=1_000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    os.makedirs(a.out, exist_ok=True)

    tr, used_tr = make_split(rng, a.train_rows_per_group, {})
    va, used_va = make_split(rng, a.eval_rows_per_group, used_tr)
    forbid = {k: np.concatenate([used_tr[k], used_va[k]]) for k in used_tr}
    te, _ = make_split(rng, a.eval_rows_per_group, forbid)
    oo, _ = make_split(rng, a.eval_rows_per_group, {}, ood=True)
    for n, s in [("train", tr), ("val", va), ("test", te), ("ood", oo)]:
        save(a.out, n, s)

    mask = ((np.arange(ROW_LEN) % EX_LEN) >= 10).astype(np.uint8)   # token positions to compute loss on
    meta = dict(
        vocab_size=OP0 + len(ALL_OPS), row_len=ROW_LEN, example_len=EX_LEN, examples_per_row=EX_PER_ROW,
        tokens=dict(SEP=SEP, EOS=EOS, hex0=HEX0, op0=OP0), op_tokens=TOK,
        groups={f"E{g}": v for g, v in GROUPS.items()}, ood_ops=list(OOD_OPS),
        loss_mask_note="mask[t]=1 where t%19>=10 (answer digits + EOS); targets are x[:,1:], use mask[1:]",
        loss_mask=mask.tolist(),
        rows=dict(train=len(tr[0]), val=len(va[0]), test=len(te[0]), ood=len(oo[0])),
        train_tokens=int(len(tr[0]) * ROW_LEN))
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=1)
    print(meta["rows"], "vocab", meta["vocab_size"], "train tokens", meta["train_tokens"])

if __name__ == "__main__":
    main()
