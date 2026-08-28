"""
MoE Dataset Generator — 70K Multi-Operation Symbolic Reasoning Examples
=======================================================================
Expert families and depth-based weights:
  E1: ADD, SUB          depth 1-2  → weight  6%
  E2: MULT, DIV         depth 2-3  → weight 10%
  E3: MOD, POW          depth 3-4  → weight 14%
  E4: AND, OR, XOR, NOT depth 1-2  → weight  6%
  E5: SHIFT_L, SHIFT_R  depth 1-2  → weight  6%
  E6: SORT, MIN, MAX,
      MEDIAN            depth 3-5  → weight 18%
  E7: CHAIN (multi-op)  depth 4-7  → weight 40%

Depth-weighted rationale:
  Shallow families (depth 1-2) get the smallest slice — the router
  needs very few such examples to stabilise single-expert paths.
  Deeper families get proportionally larger slices so the model
  learns multi-hop routing and recursive depth scaling.
  E7 (CHAIN) dominates because it forces the router to traverse
  ALL expert classes in a single sequence.

Within each family, 20 % of samples are simple (1 op) and
80 % are nested / chained (2+ ops) — matching the Dataset Balance Rule.
"""

import random
import json
import hashlib
import subprocess
from collections import Counter
import math
import statistics
import argparse
from pathlib import Path
from typing import Any
import sys

# The Windows console defaults to cp1252 and cannot encode the report glyphs
# below; without this, generation completes and then dies while printing.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
TOTAL = 70_000
SEED  = 42
TRAIN_FRAC = 0.85
VAL_FRAC = 0.075
TEST_FRAC = 0.075

# Depth-proportional weights (must sum to 1.0)
FAMILY_WEIGHTS = {
    "E1": 0.06,   # ADD, SUB          — depth 1-2
    "E2": 0.10,   # MULT, DIV         — depth 2-3
    "E3": 0.14,   # MOD, POW          — depth 3-4
    "E4": 0.06,   # AND, OR, XOR, NOT — depth 1-2
    "E5": 0.06,   # SHIFT_L, SHIFT_R  — depth 1-2
    "E6": 0.18,   # SORT, MIN, MAX, MEDIAN — depth 3-5
    "E7": 0.40,   # CHAIN (multi-op)  — depth 4-7
}
assert abs(sum(FAMILY_WEIGHTS.values()) - 1.0) < 1e-9, "Weights must sum to 1"

# Emitted family label per generator slot. The generator's "E7" slot produces
# multi-operation CHAIN programs; canonical MoRE has six experts and routes per
# step, so a chain has no single expert family. Its mass is redefined as MIXED
# rather than dropped -- dropping it would remove every depth 4-7 program and
# make adaptive-depth allocation unmeasurable (plan.md 3.4).
FAMILY_OUTPUT_LABEL = {
    "E1": "E1", "E2": "E2", "E3": "E3",
    "E4": "E4", "E5": "E5", "E6": "E6",
    "E7": "MIXED",
}

INT_RANGE   = (-1000, 1000)   # operand sampling range
LIST_LEN    = (2, 8)          # list length for E6
CHAIN_DEPTH = (4, 7)          # ops per E7 chain

# ──────────────────────────────────────────────
# SAFE MATH HELPERS
# ──────────────────────────────────────────────

def safe_div(a: int, b: int) -> int:
    """Integer division; returns 0 on divide-by-zero."""
    return a // b if b != 0 else 0

def safe_mod(a: int, b: int) -> int:
    """Modulo; returns 0 on divide-by-zero."""
    return a % b if b != 0 else 0

def safe_pow(base: int, exp: int) -> int:
    """
    Clamp exponent to [0, 6] and base to [-50, 50]
    to avoid integer explosion while keeping interesting values.
    """
    base = max(-50, min(50, base))
    exp  = max(0,   min(6,  abs(exp)))
    return int(base ** exp)

def safe_shift(val: int, bits: int, direction: str) -> int:
    """Bit-shift on the 32-bit signed integer range."""
    bits = abs(bits) % 32
    if direction == "SHIFT_L":
        result = (val << bits) & 0xFFFFFFFF
        # sign-extend
        if result >= 0x80000000:
            result -= 0x100000000
    else:
        result = val >> bits
    return result

def safe_median(lst: list[int]) -> float:
    return statistics.median(lst)

# ──────────────────────────────────────────────
# OPERAND GENERATORS
# ──────────────────────────────────────────────

def rand_int(lo: int = INT_RANGE[0], hi: int = INT_RANGE[1]) -> int:
    return random.randint(lo, hi)

def rand_nonzero() -> int:
    v = 0
    while v == 0:
        v = rand_int()
    return v

def rand_list() -> list[int]:
    length = random.randint(*LIST_LEN)
    return [rand_int() for _ in range(length)]

def rand_small_positive() -> int:
    """Used for bit counts and exponents."""
    return random.randint(0, 6)

# ──────────────────────────────────────────────
# PER-FAMILY GENERATORS
# ──────────────────────────────────────────────

def gen_E1(nested: bool) -> dict:
    """ADD, SUB — depth 1-2."""
    if not nested:
        op   = random.choice(["ADD", "SUB"])
        a, b = rand_int(), rand_int()
        res  = a + b if op == "ADD" else a - b
        seq  = [{"op": op, "args": [a, b], "result": res}]
        inp  = f"{op}({a},{b})"
        return {"input": inp, "steps": seq, "output": res, "family": "E1", "depth": 1}
    # nested: result of step1 feeds step2
    ops = random.choices(["ADD", "SUB"], k=2)
    a, b, c = rand_int(), rand_int(), rand_int()
    r1 = a + b if ops[0] == "ADD" else a - b
    r2 = r1 + c if ops[1] == "ADD" else r1 - c
    seq = [
        {"op": ops[0], "args": [a, b],  "result": r1},
        {"op": ops[1], "args": [r1, c], "result": r2},
    ]
    inp = f"{ops[0]}({a},{b}) → {ops[1]}(result,{c})"
    return {"input": inp, "steps": seq, "output": r2, "family": "E1", "depth": 2}


def gen_E2(nested: bool) -> dict:
    """MULT, DIV — depth 2-3."""
    if not nested:
        op   = random.choice(["MULT", "DIV"])
        a, b = rand_int(), rand_nonzero()
        res  = a * b if op == "MULT" else safe_div(a, b)
        seq  = [{"op": op, "args": [a, b], "result": res}]
        inp  = f"{op}({a},{b})"
        return {"input": inp, "steps": seq, "output": res, "family": "E2", "depth": 1}
    depth = random.choice([2, 3])
    ops   = random.choices(["MULT", "DIV"], k=depth)
    vals  = [rand_int()] + [rand_nonzero() for _ in range(depth)]
    seq   = []
    cur   = vals[0]
    for i, op in enumerate(ops):
        nxt = cur * vals[i+1] if op == "MULT" else safe_div(cur, vals[i+1])
        seq.append({"op": op, "args": [cur, vals[i+1]], "result": nxt})
        cur = nxt
    parts = [f"{ops[0]}({vals[0]},{vals[1]})"]
    for i in range(1, depth):
        parts.append(f"{ops[i]}(result,{vals[i+1]})")
    return {"input": " → ".join(parts), "steps": seq, "output": cur,
            "family": "E2", "depth": depth}


def gen_E3(nested: bool) -> dict:
    """MOD, POW — depth 3-4."""
    if not nested:
        op = random.choice(["MOD", "POW"])
        a  = rand_int()
        b  = rand_nonzero() if op == "MOD" else rand_small_positive()
        res = safe_mod(a, b) if op == "MOD" else safe_pow(a, b)
        seq = [{"op": op, "args": [a, b], "result": res}]
        return {"input": f"{op}({a},{b})", "steps": seq, "output": res,
                "family": "E3", "depth": 1}
    depth = random.choice([3, 4])
    seq, parts = [], []
    cur = rand_int()
    parts.append(str(cur))
    for i in range(depth):
        op = random.choice(["MOD", "POW"])
        b  = rand_nonzero() if op == "MOD" else rand_small_positive()
        nxt = safe_mod(cur, b) if op == "MOD" else safe_pow(cur, b)
        if i == 0:
            parts = [f"{op}({cur},{b})"]
        else:
            parts.append(f"{op}(result,{b})")
        seq.append({"op": op, "args": [cur, b], "result": nxt})
        cur = nxt
    return {"input": " → ".join(parts), "steps": seq, "output": cur,
            "family": "E3", "depth": depth}


def gen_E4(nested: bool) -> dict:
    """AND, OR, XOR, NOT — depth 1-2."""
    unary_ops  = ["NOT"]
    binary_ops = ["AND", "OR", "XOR"]

    def apply(op, a, b=None):
        if op == "NOT":  return ~a & 0xFFFF          # 16-bit NOT for readability
        if op == "AND":  return a & b
        if op == "OR":   return a | b
        if op == "XOR":  return a ^ b

    if not nested:
        op = random.choice(binary_ops + unary_ops)
        a  = rand_int(0, 255)
        if op == "NOT":
            res = apply("NOT", a)
            seq = [{"op": op, "args": [a], "result": res}]
            inp = f"NOT({a})"
        else:
            b   = rand_int(0, 255)
            res = apply(op, a, b)
            seq = [{"op": op, "args": [a, b], "result": res}]
            inp = f"{op}({a},{b})"
        return {"input": inp, "steps": seq, "output": res, "family": "E4", "depth": 1}

    # Two-step chain
    op1 = random.choice(binary_ops)
    op2 = random.choice(binary_ops + unary_ops)
    a, b, c = rand_int(0, 255), rand_int(0, 255), rand_int(0, 255)
    r1 = apply(op1, a, b)
    if op2 == "NOT":
        r2  = apply("NOT", r1)
        inp = f"{op1}({a},{b}) → NOT(result)"
        seq = [
            {"op": op1, "args": [a, b],  "result": r1},
            {"op": op2, "args": [r1],    "result": r2},
        ]
    else:
        r2  = apply(op2, r1, c)
        inp = f"{op1}({a},{b}) → {op2}(result,{c})"
        seq = [
            {"op": op1, "args": [a, b],  "result": r1},
            {"op": op2, "args": [r1, c], "result": r2},
        ]
    return {"input": inp, "steps": seq, "output": r2, "family": "E4", "depth": 2}


def gen_E5(nested: bool) -> dict:
    """SHIFT_L, SHIFT_R — depth 1-2."""
    if not nested:
        op   = random.choice(["SHIFT_L", "SHIFT_R"])
        a, b = rand_int(), random.randint(0, 10)
        res  = safe_shift(a, b, op)
        seq  = [{"op": op, "args": [a, b], "result": res}]
        return {"input": f"{op}({a},{b})", "steps": seq, "output": res,
                "family": "E5", "depth": 1}
    ops = [random.choice(["SHIFT_L", "SHIFT_R"]) for _ in range(2)]
    a   = rand_int()
    b1, b2 = random.randint(0, 10), random.randint(0, 10)
    r1  = safe_shift(a,  b1, ops[0])
    r2  = safe_shift(r1, b2, ops[1])
    seq = [
        {"op": ops[0], "args": [a,  b1], "result": r1},
        {"op": ops[1], "args": [r1, b2], "result": r2},
    ]
    inp = f"{ops[0]}({a},{b1}) → {ops[1]}(result,{b2})"
    return {"input": inp, "steps": seq, "output": r2, "family": "E5", "depth": 2}


def gen_E6(nested: bool) -> dict:
    """SORT, MIN, MAX, MEDIAN — depth 3-5."""
    base_op = random.choice(["SORT", "MIN", "MAX", "MEDIAN"])
    lst = rand_list()

    def apply_list_op(op, arr):
        if op == "SORT":   return sorted(arr)
        if op == "MIN":    return min(arr)
        if op == "MAX":    return max(arr)
        if op == "MEDIAN": return safe_median(arr)

    if not nested:
        res = apply_list_op(base_op, lst)
        seq = [{"op": base_op, "args": lst, "result": res}]
        return {"input": f"{base_op}({lst})", "steps": seq, "output": res,
                "family": "E6", "depth": 1}

    depth = random.randint(3, 5)
    seq, parts = [], []
    cur_list = lst
    cur_val  = None
    for i in range(depth):
        op = random.choice(["SORT", "MIN", "MAX", "MEDIAN"])
        if i == 0:
            res = apply_list_op(op, cur_list)
            parts.append(f"{op}({cur_list})")
            seq.append({"op": op, "args": cur_list, "result": res})
            if isinstance(res, list):
                cur_list = res
                cur_val  = None
            else:
                cur_val  = res
                # inject a new list incorporating cur_val for the next step
                cur_list = [int(cur_val)] + [rand_int() for _ in range(random.randint(2, 5))]
        else:
            res = apply_list_op(op, cur_list)
            parts.append(f"{op}([result,...,{cur_list[-1]}])")
            seq.append({"op": op, "args": cur_list, "result": res})
            if isinstance(res, list):
                cur_list = res
                cur_val  = None
            else:
                cur_val  = res
                cur_list = [int(cur_val)] + [rand_int() for _ in range(random.randint(2, 5))]
    final = seq[-1]["result"]
    if isinstance(final, list):
        final_out = final
    else:
        final_out = float(final) if isinstance(final, float) else int(final)
    return {"input": " → ".join(parts), "steps": seq, "output": final_out,
            "family": "E6", "depth": depth}


# Operation table for E7 chains
E7_OPS = {
    "ADD":     lambda a, b, _: a + b,
    "SUB":     lambda a, b, _: a - b,
    "MULT":    lambda a, b, _: a * b,
    "DIV":     lambda a, b, _: safe_div(a, b),
    "MOD":     lambda a, b, _: safe_mod(a, b),
    "POW":     lambda a, b, _: safe_pow(a, b),
    "AND":     lambda a, b, _: a & b,
    "OR":      lambda a, b, _: a | b,
    "XOR":     lambda a, b, _: a ^ b,
    "NOT":     lambda a, b, _: ~a & 0xFFFF,
    "SHIFT_L": lambda a, b, _: safe_shift(a, b, "SHIFT_L"),
    "SHIFT_R": lambda a, b, _: safe_shift(a, b, "SHIFT_R"),
    "MIN":     lambda a, b, lst: min(lst),
    "MAX":     lambda a, b, lst: max(lst),
    "SORT":    lambda a, b, lst: sorted(lst),
    "MEDIAN":  lambda a, b, lst: safe_median(lst),
}

ALL_OPS = list(E7_OPS.keys())

def gen_E7(_nested: bool) -> dict:  # nested is always true for E7
    """CHAIN — depth 4-7, spans all expert families."""
    depth = random.randint(*CHAIN_DEPTH)
    ops   = random.choices(ALL_OPS, k=depth)

    seq, parts = [], []
    # Initial scalar value
    cur = rand_int()
    start_val = cur

    for i, op in enumerate(ops):
        if op in ("MIN", "MAX", "SORT", "MEDIAN"):
            # Build a list from current val + randoms
            lst = [cur] + [rand_int() for _ in range(random.randint(2, 5))]
            random.shuffle(lst)
            res = E7_OPS[op](cur, None, lst)
            arg_repr = str(lst)
            if i == 0:
                parts.append(f"{op}({arg_repr})")
            else:
                parts.append(f"{op}([result,...])")
            seq.append({"op": op, "args": lst, "result": res})
            # next cur: extract scalar from sorted or keep numeric
            if isinstance(res, list):
                cur = res[0]
            elif isinstance(res, float):
                cur = int(res)
            else:
                cur = int(res)
        elif op == "NOT":
            res = E7_OPS["NOT"](cur, None, None)
            if i == 0:
                parts.append(f"NOT({cur})")
            else:
                parts.append("NOT(result)")
            seq.append({"op": op, "args": [cur], "result": res})
            cur = res
        elif op in ("POW", "SHIFT_L", "SHIFT_R"):
            b = rand_small_positive()
            res = E7_OPS[op](cur, b, None)
            if i == 0:
                parts.append(f"{op}({cur},{b})")
            else:
                parts.append(f"{op}(result,{b})")
            seq.append({"op": op, "args": [cur, b], "result": res})
            cur = res
        elif op == "DIV":
            b = rand_nonzero()
            res = safe_div(cur, b)
            if i == 0:
                parts.append(f"DIV({cur},{b})")
            else:
                parts.append(f"DIV(result,{b})")
            seq.append({"op": op, "args": [cur, b], "result": res})
            cur = res
        elif op == "MOD":
            b = rand_nonzero()
            res = safe_mod(cur, b)
            if i == 0:
                parts.append(f"MOD({cur},{b})")
            else:
                parts.append(f"MOD(result,{b})")
            seq.append({"op": op, "args": [cur, b], "result": res})
            cur = res
        else:  # ADD, SUB, MULT, AND, OR, XOR
            b = rand_int(0, 255) if op in ("AND", "OR", "XOR") else rand_int()
            res = E7_OPS[op](cur, b, None)
            if i == 0:
                parts.append(f"{op}({cur},{b})")
            else:
                parts.append(f"{op}(result,{b})")
            seq.append({"op": op, "args": [cur, b], "result": res})
            cur = res

    final = seq[-1]["result"]
    if isinstance(final, float):
        final = float(final)
    elif isinstance(final, list):
        final = final
    else:
        final = int(final)

    return {
        "input": " → ".join(parts),
        "steps": seq,
        "output": final,
        "family": "E7",
        "depth": depth,
        "ops_used": ops,
    }


# ──────────────────────────────────────────────
# CORRECTNESS VERIFIER
# ──────────────────────────────────────────────

def verify(example: dict) -> bool:
    """
    Re-executes every step from scratch and checks that each step's
    stored result matches the recomputed result, and that the final
    output matches the last step.
    """
    steps = example["steps"]
    if not steps:
        return False

    for step in steps:
        op   = step["op"]
        args = step["args"]
        stored = step["result"]

        try:
            if op == "ADD":     recomputed = args[0] + args[1]
            elif op == "SUB":   recomputed = args[0] - args[1]
            elif op == "MULT":  recomputed = args[0] * args[1]
            elif op == "DIV":   recomputed = safe_div(args[0], args[1])
            elif op == "MOD":   recomputed = safe_mod(args[0], args[1])
            elif op == "POW":   recomputed = safe_pow(args[0], args[1])
            elif op == "AND":   recomputed = args[0] & args[1]
            elif op == "OR":    recomputed = args[0] | args[1]
            elif op == "XOR":   recomputed = args[0] ^ args[1]
            elif op == "NOT":   recomputed = ~args[0] & 0xFFFF
            elif op == "SHIFT_L": recomputed = safe_shift(args[0], args[1], "SHIFT_L")
            elif op == "SHIFT_R": recomputed = safe_shift(args[0], args[1], "SHIFT_R")
            elif op == "SORT":  recomputed = sorted(args)
            elif op == "MIN":   recomputed = min(args)
            elif op == "MAX":   recomputed = max(args)
            elif op == "MEDIAN":recomputed = safe_median(args)
            else:
                return False  # unknown op

            # float comparison for MEDIAN
            if isinstance(recomputed, float) or isinstance(stored, float):
                if abs(float(recomputed) - float(stored)) > 1e-9:
                    return False
            elif recomputed != stored:
                return False
        except Exception:
            return False

    # Final output check
    last_result = steps[-1]["result"]
    out         = example["output"]
    if isinstance(last_result, float) or isinstance(out, float):
        return abs(float(last_result) - float(out)) < 1e-9
    return last_result == out


# ──────────────────────────────────────────────
# MAIN GENERATION LOOP
# ──────────────────────────────────────────────

GENERATORS = {
    "E1": gen_E1,
    "E2": gen_E2,
    "E3": gen_E3,
    "E4": gen_E4,
    "E5": gen_E5,
    "E6": gen_E6,
    "E7": gen_E7,
}

def build_family_schedule(total: int, weights: dict) -> dict[str, int]:
    """Convert weight fractions to exact integer counts (with rounding correction)."""
    raw    = {f: total * w for f, w in weights.items()}
    counts = {f: int(v) for f, v in raw.items()}
    deficit = total - sum(counts.values())
    # Distribute remainder to families with largest fractional parts
    fractions = sorted(raw.keys(), key=lambda f: raw[f] - counts[f], reverse=True)
    for i in range(deficit):
        counts[fractions[i % len(fractions)]] += 1
    return counts


def _program_key(example: dict) -> str:
    """
    Identity of a program: its exact (op, args, result) sequence.

    Used to guarantee global uniqueness, which in turn guarantees zero
    cross-split overlap. The pre-Phase-1 dataset had 22 programs shared between
    test and train and 14 between train and val, which inflates any reported
    validation score (plan.md 3.5).
    """
    return json.dumps(
        [[s.get("op"), s.get("args"), s.get("result")]
         for s in example.get("steps", [])],
        sort_keys=True, separators=(",", ":"),
    )


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _generator_identifier() -> str:
    """Git commit of the repository, so a dataset is traceable to its generator."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(Path(__file__).resolve().parent),
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return "unknown"


def split_and_save(
    examples: list[dict],
    out_dir: str,
    fmt: str = "jsonl",
    seed: int = SEED,
) -> None:
    """Save train/val/test splits and metadata to a distinct output folder."""
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_total = len(examples)
    n_train = int(n_total * TRAIN_FRAC)
    n_val = int(n_total * VAL_FRAC)

    splits = {
        "train": examples[:n_train],
        "val": examples[n_train:n_train + n_val],
        "test": examples[n_train + n_val:],
    }

    split_paths = {}
    for split_name, rows in splits.items():
        if fmt == "jsonl":
            split_path = output_dir / f"{split_name}.jsonl"
            with split_path.open("w") as f:
                for row in rows:
                    f.write(json.dumps(row) + "\n")
        else:
            split_path = output_dir / f"{split_name}.json"
            with split_path.open("w") as f:
                json.dump(rows, f, indent=2)
        split_paths[split_name] = split_path
        print(f"  Saved {split_name:5s} -> {split_path} ({len(rows):,} rows)")

    # ---- Manifest (plan.md 3.4) ------------------------------------------
    hashes = {k: _sha256(p) for k, p in split_paths.items()}
    combined = hashlib.sha256(
        "".join(hashes[k] for k in sorted(hashes)).encode()
    ).hexdigest()
    dataset_version = f"more6-v1-seed{seed}-n{n_total}-{combined[:12]}"

    # Overlap must be zero by construction; assert it rather than trust it.
    key_sets = {k: {_program_key(e) for e in v} for k, v in splits.items()}
    overlaps = {
        "train|val":  len(key_sets["train"] & key_sets["val"]),
        "train|test": len(key_sets["train"] & key_sets["test"]),
        "val|test":   len(key_sets["val"] & key_sets["test"]),
    }
    if any(overlaps.values()):
        raise RuntimeError(
            f"Cross-split program overlap detected: {overlaps}. Generation is "
            "supposed to reject duplicate programs globally."
        )

    def _counts(rows, key):
        c = Counter()
        for r in rows:
            c[r.get(key)] += 1
        return dict(sorted(c.items(), key=lambda kv: str(kv[0])))

    def _op_counts(rows):
        c = Counter()
        for r in rows:
            for s in r.get("steps", []):
                c[s.get("op")] += 1
        return dict(sorted(c.items()))

    # Raw value range actually present in the data (the tokenizer's own
    # normalisation constant lives in the model config, not here).
    lo, hi = math.inf, -math.inf
    for r in examples:
        for s in r.get("steps", []):
            vals = []
            for a in s.get("args", []):
                vals.extend(a if isinstance(a, list) else [a])
            res = s.get("result")
            vals.extend(res if isinstance(res, list) else [res])
            for v in vals:
                if isinstance(v, (int, float)):
                    lo, hi = min(lo, float(v)), max(hi, float(v))

    meta = {
        "dataset_version": dataset_version,
        "generator_script": Path(__file__).name,
        "generator_commit": _generator_identifier(),
        "seed": seed,
        "total": n_total,
        "splits": {k: len(v) for k, v in splits.items()},
        "sha256": hashes,
        "train_frac": TRAIN_FRAC,
        "val_frac": VAL_FRAC,
        "test_frac": TEST_FRAC,
        "family_weights": FAMILY_WEIGHTS,
        "family_label_redefinition": {
            "rule": (
                "The generator's E7 slot is the multi-operation CHAIN family, "
                "not a seventh expert. Canonical MoRE has exactly six experts "
                "(E1-E6) and routes PER STEP, so a chain's steps are routed to "
                "E1-E6 individually. Its 0.40 mass is therefore REDEFINED, not "
                "dropped: these records are emitted with family='MIXED' and no "
                "whole-program expert label. Dropping them instead would delete "
                "every program of depth 4-7 and make adaptive-depth allocation "
                "unmeasurable, which is the phenomenon under study."
            ),
            "generator_slot": "E7",
            "emitted_family_label": FAMILY_OUTPUT_LABEL["E7"],
            "mass": FAMILY_WEIGHTS["E7"],
            "whole_program_family_index": -1,
        },
        "family_counts": {k: _counts(v, "family") for k, v in splits.items()},
        "operation_counts": {k: _op_counts(v) for k, v in splits.items()},
        "depth_counts": {k: _counts(v, "depth") for k, v in splits.items()},
        "raw_value_range": [lo if lo != math.inf else None,
                            hi if hi != -math.inf else None],
        "split_overlap_program_keys": overlaps,
        "unique_programs": {k: len(v) for k, v in key_sets.items()},
        "input_feature_definition": (
            "Per step, one row of `step_feat_dim` floats holding ONLY that step's "
            "numeric arguments, each clamped to +/-max_val and divided by max_val. "
            "The step result is excluded (it equals the regression target for the "
            "final step). The oracle expert index is excluded. Operation identity "
            "is supplied separately as `step_ops` and consumed by an embedding "
            "over operations. Consequently the feature row is independent of "
            "num_experts and is bit-identical for MoE / MoR / MoRE."
        ),
        "target_definition": (
            "record['output'], mean-reduced if it is a list, clamped to "
            "+/-max_val and divided by max_val."
        ),
    }
    meta_path = output_dir / "dataset_meta.json"
    with meta_path.open("w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Metadata -> {meta_path}")
    print(f"  dataset_version = {dataset_version}")
    print(f"  split overlap   = {overlaps}")


def generate_dataset(
    total: int = TOTAL,
    seed: int = SEED,
    out_dir: str = "moe_dataset2",
    fmt: str = "jsonl",
    verify_all: bool = True,
) -> None:
    random.seed(seed)

    schedule = build_family_schedule(total, FAMILY_WEIGHTS)
    print("=" * 60)
    print(f"{'Family':<8} {'Count':>8}  {'Weight':>8}  {'Depth'}")
    print("-" * 60)
    depth_labels = {
        "E1": "1-2", "E2": "2-3", "E3": "3-4",
        "E4": "1-2", "E5": "1-2", "E6": "3-5", "E7": "4-7"
    }
    for fam, cnt in schedule.items():
        print(f"  {fam:<6} {cnt:>8}  {FAMILY_WEIGHTS[fam]:>7.0%}  depth {depth_labels[fam]}")
    print(f"  {'TOTAL':<6} {sum(schedule.values()):>8}")
    print("=" * 60)

    examples     = []
    verify_fails = 0
    duplicates   = 0
    retry_limit  = 10
    seen_keys    = set()

    for family, count in schedule.items():
        gen = GENERATORS[family]
        produced = 0
        attempts = 0
        while produced < count:
            attempts += 1
            if attempts > count * retry_limit:
                print(f"  WARNING: {family} hit retry limit. Produced {produced}/{count}.")
                break
            # 20% simple, 80% nested
            nested = random.random() < 0.80
            try:
                ex = gen(nested)
            except Exception as e:
                continue  # skip malformed generations
            if verify_all:
                if not verify(ex):
                    verify_fails += 1
                    continue
            # Global uniqueness: a program may appear at most once in the whole
            # dataset, which makes cross-split overlap impossible by
            # construction rather than by luck (plan.md 3.5).
            key = _program_key(ex)
            if key in seen_keys:
                duplicates += 1
                continue
            seen_keys.add(key)
            # The generator's E7 slot is the multi-op CHAIN family, not a
            # seventh expert; it is emitted under its redefined label. See the
            # family_label_redefinition block in dataset_meta.json.
            ex["family"] = FAMILY_OUTPUT_LABEL.get(family, family)
            ex["generator_family_slot"] = family
            ex["id"] = f"{family}_{produced:06d}"
            examples.append(ex)
            produced += 1

    # Shuffle before split to avoid family clustering in one split
    random.shuffle(examples)

    print(f"\n✓  Generated {len(examples):,} verified examples")
    if verify_fails:
        print(f"   (Discarded {verify_fails:,} examples that failed correctness check)")
    if duplicates:
        print(f"   (Rejected {duplicates:,} duplicate programs to guarantee no cross-split overlap)")

    print(f"\nSaving train/val/test splits to: {Path(out_dir)}")
    split_and_save(examples, out_dir=out_dir, fmt=fmt, seed=seed)

    # Quick distribution sanity check
    dist = Counter(e["family"] for e in examples)
    label = FAMILY_OUTPUT_LABEL
    avg_depth = {}
    for fam in FAMILY_WEIGHTS:
        depths = [e["depth"] for e in examples
                  if e["family"] == label.get(fam, fam)]
        avg_depth[fam] = sum(depths) / len(depths) if depths else 0

    print("\nFinal distribution:")
    print(f"  {'Family':<8} {'Count':>8}  {'Avg Depth':>10}")
    for fam in FAMILY_WEIGHTS:
        emitted = label.get(fam, fam)
        print(f"  {emitted:<8} {dist[emitted]:>8}  {avg_depth[fam]:>10.2f}")


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate 70K MoE symbolic reasoning examples"
    )
    parser.add_argument("--total",   type=int,   default=TOTAL,
                        help="Total number of examples (default 70000)")
    parser.add_argument("--seed",    type=int,   default=SEED,
                        help="Random seed (default 42)")
    parser.add_argument("--out-dir", type=str,   default="moe_dataset2",
                        help="Output directory for train/val/test files")
    parser.add_argument("--format",  choices=["jsonl", "json"], default="jsonl",
                        help="Split file format (jsonl or json)")
    parser.add_argument("--no-verify", action="store_true",
                        help="Skip correctness verification (faster, not recommended)")
    args = parser.parse_args()

    generate_dataset(
        total      = args.total,
        seed       = args.seed,
        out_dir    = args.out_dir,
        fmt        = args.format,
        verify_all = not args.no_verify,
    )