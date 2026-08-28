"""
audit_leakage.py -- Gate 1 acceptance test (plan.md 3.5 / TASKS.md T1.1-T1.5).

Answers five questions with measurements, not opinions:

  A. Is the regression target present in any input slot?          (T1.1)
  B. Can the oracle expert label be read off any single slot?     (T1.2)
  C. Are input features bit-identical across MoE / MoR / MoRE?    (T1.3)
  D. Do train / val / test splits overlap?                        (T1.5)
  E. What do trivial predictors score -- the MSE floor below which
     a "good" task loss means nothing?                            (T1.5)

Exit code 0 = Gate 1 passes. Non-zero = a specific named failure.

    python audit_leakage.py
    python audit_leakage.py --json runs/gate1_audit.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

from more.config import load_config
from more.data import MoREDataset
from more.families import OP_TO_EXPERT, ALL_OP_NAMES

_CODE_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_DIR = os.path.dirname(_CODE_DIR)

TOL = 1e-9


class Report:
    def __init__(self) -> None:
        self.rows: list[tuple[str, bool, str]] = []
        self.facts: dict = {}

    def check(self, name: str, ok: bool, detail: str = "") -> bool:
        self.rows.append((name, bool(ok), detail))
        glyph = "PASS" if ok else "FAIL"
        print(f"  [{glyph}] {name}" + (f"  -- {detail}" if detail else ""))
        return ok

    def note(self, key: str, value) -> None:
        self.facts[key] = value

    @property
    def failed(self) -> list[str]:
        return [n for n, ok, _ in self.rows if not ok]


def load_split(path: str, cfg: dict, num_experts: int) -> MoREDataset:
    mc, dc = cfg["model"], cfg["data"]
    return MoREDataset(
        jsonl_path=path,
        max_steps=mc["max_steps"],
        step_feat_dim=mc["step_feat_dim"],
        max_val=dc["max_val"],
        pad_value=dc["pad_value"],
        num_experts=num_experts,
    )


# ---------------------------------------------------------------------------
# A. target-in-input
# ---------------------------------------------------------------------------

def audit_target_leakage(ds: MoREDataset, rep: Report, label: str) -> None:
    """
    Reports the rate at which the target value coincides with some input slot.

    This is a DIAGNOSTIC, not the gate. For MAX / MIN / MEDIAN / SORT the answer
    IS one of the arguments -- that is the task, not a leak. The decisive tests
    are `audit_result_mutation` (proves no step result reaches the input) and
    baseline E (proves no copy channel is exploitable).
    """
    n = len(ds)
    per_slot_hits = None
    records_with_any_hit = 0

    for i in range(n):
        x, step_mask, _, _, _, _, target = ds[i]
        real = x[step_mask]                       # [n_real_steps, F]
        if real.numel() == 0:
            continue
        eq = (real - target).abs() <= TOL         # [n_real_steps, F]
        if per_slot_hits is None:
            per_slot_hits = torch.zeros(x.shape[1], dtype=torch.long)
        per_slot_hits += eq.any(dim=0).long()
        if bool(eq.any()):
            records_with_any_hit += 1

    frac = records_with_any_hit / max(n, 1)
    worst_slot = int(torch.argmax(per_slot_hits)) if per_slot_hits is not None else -1
    worst_cnt = int(per_slot_hits[worst_slot]) if per_slot_hits is not None else 0
    worst_frac = worst_cnt / max(n, 1)

    rep.note(f"{label}_records", n)
    rep.note(f"{label}_records_with_target_in_input", records_with_any_hit)
    rep.note(f"{label}_target_in_input_fraction", round(frac, 6))
    rep.note(f"{label}_worst_slot", worst_slot)
    rep.note(f"{label}_worst_slot_fraction", round(worst_frac, 6))
    rep.note(f"{label}_per_slot_target_hits",
             per_slot_hits.tolist() if per_slot_hits is not None else [])

    print(f"  [ .. ] {label}: target coincides with some arg in "
          f"{records_with_any_hit}/{n} = {frac:.2%} of records "
          f"(inherent for MAX/MIN/MEDIAN/SORT); worst single slot {worst_slot} "
          f"at {worst_frac:.2%}. Pre-fix: 100.00% of records, worst slot 74.15%.")


def audit_result_mutation(path: str, cfg: dict, rep: Report,
                          n_check: int = 400) -> None:
    """
    DECISIVE test for T1.1. If no step `result` reaches the input, then changing
    every result in the file cannot change a single input feature. Bit-identical
    features under result mutation is proof by construction -- it cannot be
    satisfied by a leak that merely looks small.
    """
    import tempfile

    orig_lines = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                orig_lines.append(line)
            if len(orig_lines) >= n_check:
                break

    mutated = []
    for line in orig_lines:
        rec = json.loads(line)
        for s in rec.get("steps", []):
            r = s.get("result", 0)
            # A distinctive, order-of-magnitude change: any leak shows up loudly.
            s["result"] = ([float(v) * 3.0 + 137.0 for v in r]
                           if isinstance(r, list) else float(r) * 3.0 + 137.0)
        out = rec.get("output", 0)
        rec["output"] = ([float(v) * 3.0 + 137.0 for v in out]
                         if isinstance(out, list) else float(out) * 3.0 + 137.0)
        mutated.append(json.dumps(rec))

    tmp = tempfile.mkdtemp(prefix="more_gate1_")
    p_a = os.path.join(tmp, "orig.jsonl")
    p_b = os.path.join(tmp, "mutated.jsonl")
    with open(p_a, "w") as f:
        f.write("\n".join(orig_lines) + "\n")
    with open(p_b, "w") as f:
        f.write("\n".join(mutated) + "\n")

    ds_a = load_split(p_a, cfg, 6)
    ds_b = load_split(p_b, cfg, 6)

    bad = []
    target_changed = 0
    for i in range(len(ds_a)):
        xa, ma, ea, oa, _, _, ta = ds_a[i]
        xb, mb, eb, ob, _, _, tb = ds_b[i]
        if not torch.equal(xa, xb):
            bad.append(i)
        if not torch.equal(ma, mb) or not torch.equal(oa, ob) or not torch.equal(ea, eb):
            bad.append(i)
        if not torch.equal(ta, tb):
            target_changed += 1

    import shutil
    shutil.rmtree(tmp, ignore_errors=True)

    rep.note("result_mutation_records", len(ds_a))
    rep.note("result_mutation_feature_changes", len(bad))
    rep.note("result_mutation_targets_changed", target_changed)

    # Sanity: the mutation must actually have done something, else the test is
    # vacuous and would "pass" on a broken harness.
    effective = target_changed > 0
    rep.check(
        "A0. result mutation actually changed the targets (test is not vacuous)",
        effective,
        f"{target_changed}/{len(ds_a)} targets changed",
    )
    rep.check(
        "A. no step result reaches the input (mutation leaves features identical)",
        not bad,
        f"{len(ds_a)} records mutated; {len(bad)} feature change(s). "
        f"Pre-fix every record's features changed.",
    )


# ---------------------------------------------------------------------------
# B. oracle expert label readable from a slot
# ---------------------------------------------------------------------------

def audit_expert_permutation(path: str, cfg: dict, rep: Report,
                             n_check: int = 400) -> None:
    """
    DECISIVE test for T1.2. Expert indices are arbitrary labels. If the input
    encodes no oracle expert index, then relabelling the experts (permuting
    op -> expert) cannot change any input feature, while the supervision target
    `step_experts` must change. Pre-fix, slot 0 literally held
    expert_id / (num_experts - 1), so this test would have failed on every row.
    """
    import tempfile
    import more.data as data_mod

    lines = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                lines.append(line)
            if len(lines) >= n_check:
                break
    tmp = tempfile.mkdtemp(prefix="more_gate1_perm_")
    p = os.path.join(tmp, "subset.jsonl")
    with open(p, "w") as f:
        f.write("\n".join(lines) + "\n")

    original_map = dict(data_mod.OP_TO_EXPERT)
    n_exp = max(original_map.values()) + 1
    # A cyclic relabelling: expert e becomes (e + 1) % n_exp.
    permuted = {op: (e + 1) % n_exp for op, e in original_map.items()}

    ds_a = load_split(p, cfg, 6)
    try:
        data_mod.OP_TO_EXPERT = permuted
        ds_b = load_split(p, cfg, 6)
    finally:
        data_mod.OP_TO_EXPERT = original_map

    feature_changes = 0
    label_changes = 0
    for i in range(len(ds_a)):
        xa, ma, ea, oa, _, _, _ = ds_a[i]
        xb, mb, eb, ob, _, _, _ = ds_b[i]
        if not (torch.equal(xa, xb) and torch.equal(ma, mb) and torch.equal(oa, ob)):
            feature_changes += 1
        if not torch.equal(ea, eb):
            label_changes += 1

    import shutil
    shutil.rmtree(tmp, ignore_errors=True)

    rep.note("expert_permutation_records", len(ds_a))
    rep.note("expert_permutation_feature_changes", feature_changes)
    rep.note("expert_permutation_label_changes", label_changes)

    rep.check(
        "B0. expert permutation actually relabelled the experts (not vacuous)",
        label_changes > 0,
        f"{label_changes}/{len(ds_a)} records changed step_experts",
    )
    rep.check(
        "B. no oracle expert label in the input (permutation leaves features identical)",
        feature_changes == 0,
        f"{feature_changes}/{len(ds_a)} records changed features. "
        f"Pre-fix slot 0 was expert_id/(E-1), so all would change.",
    )


def audit_low_cardinality_slots(ds: MoREDataset, rep: Report,
                                num_experts: int) -> None:
    """
    Secondary check: a planted categorical label shows up as a slot with very few
    distinct values that map near-perfectly onto the expert label. Continuous
    argument slots have many distinct values and are exempt -- grouping continuous
    floats by exact value would score ~1.0 for any slot and prove nothing.
    """
    import collections

    slot_dim = ds[0][0].shape[1]
    value_to_expert = [collections.defaultdict(collections.Counter)
                       for _ in range(slot_dim)]
    expert_counts = collections.Counter()
    total = 0

    for i in range(len(ds)):
        x, m, se, _, _, _, _ = ds[i]
        rows, exps = x[m], se[m]
        for r in range(rows.shape[0]):
            e = int(exps[r])
            expert_counts[e] += 1
            total += 1
            for s in range(slot_dim):
                value_to_expert[s][round(float(rows[r, s]), 9)][e] += 1

    majority = (max(expert_counts.values()) / total) if total else 0.0
    offenders, table = [], []
    for s in range(slot_dim):
        card = len(value_to_expert[s])
        purity = sum(c.most_common(1)[0][1]
                     for c in value_to_expert[s].values()) / max(total, 1)
        table.append({"slot": s, "distinct_values": card, "purity": round(purity, 6)})
        if card <= num_experts + 1 and purity >= 0.99:
            offenders.append(s)

    rep.note("low_cardinality_slot_table", table)
    rep.note("oracle_majority_class_rate", round(majority, 6))
    rep.note("expert_step_counts", dict(sorted(expert_counts.items())))
    rep.check(
        "B2. no low-cardinality slot maps 1:1 onto the expert label",
        not offenders,
        f"offending slots {offenders}; majority-class rate {majority:.4f}",
    )


# ---------------------------------------------------------------------------
# C. input parity across architectures
# ---------------------------------------------------------------------------

def audit_input_parity(path: str, cfg: dict, rep: Report, n_check: int = 200) -> None:
    """
    The same record must produce bit-identical features whatever num_experts is,
    otherwise MoE / MoR / MoRE are not solving the same task and the comparison
    is void. E=1 is MoR, E=6 is canonical MoE/MoRE, E=5 is an arbitrary third
    value that would expose any residual /(E-1) scaling.
    """
    dss = {e: load_split(path, cfg, e) for e in (1, 5, 6)}
    n = min(n_check, *(len(d) for d in dss.values()))

    mismatches = []
    for i in range(n):
        x1, m1, e1, o1, f1, d1, t1 = dss[1][i]
        for e in (5, 6):
            x2, m2, e2_, o2, f2, d2, t2 = dss[e][i]
            if not torch.equal(x1, x2):
                mismatches.append((i, e, "x"))
            if not torch.equal(m1, m2):
                mismatches.append((i, e, "step_mask"))
            if not torch.equal(o1, o2):
                mismatches.append((i, e, "step_ops"))
            if not torch.equal(t1, t2):
                mismatches.append((i, e, "target"))

    # Slot-0 range was the pre-fix tell: [0, 5.0] at E=1 vs [0, 1.0] at E=6.
    ranges = {}
    for e, d in dss.items():
        cols = [d[i][0][d[i][1]][:, 0] for i in range(min(n, len(d)))]
        cols = [c for c in cols if c.numel()]
        vals = torch.cat(cols)
        ranges[e] = [round(float(vals.min()), 6), round(float(vals.max()), 6)]

    rep.note("parity_records_checked", n)
    rep.note("parity_slot0_range_by_num_experts", ranges)
    rep.note("parity_mismatches", mismatches[:20])
    rep.check(
        "C. input features bit-identical across E = 1 / 5 / 6",
        not mismatches,
        f"{n} records x 3 expert counts; slot-0 ranges {ranges}; "
        f"{len(mismatches)} mismatch(es)",
    )


# ---------------------------------------------------------------------------
# D. split overlap
# ---------------------------------------------------------------------------

def _record_key(rec: dict) -> str:
    steps = rec.get("steps", [])
    return json.dumps(
        [[s.get("op"), s.get("args"), s.get("result")] for s in steps],
        sort_keys=True, separators=(",", ":"),
    )


def audit_split_overlap(paths: dict, rep: Report) -> None:
    keys = {}
    for name, p in paths.items():
        if not (p and os.path.exists(p)):
            continue
        seen = set()
        with open(p, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        seen.add(_record_key(json.loads(line)))
                    except json.JSONDecodeError:
                        continue
        keys[name] = seen
        rep.note(f"{name}_unique_programs", len(seen))

    overlaps = {}
    names = sorted(keys)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            inter = len(keys[a] & keys[b])
            overlaps[f"{a}|{b}"] = inter
    rep.note("split_overlap_counts", overlaps)
    rep.check(
        "D. no (op, args, result) program overlap across splits",
        all(v == 0 for v in overlaps.values()),
        f"{overlaps}",
    )


# ---------------------------------------------------------------------------
# E. trivial baselines
# ---------------------------------------------------------------------------

def audit_trivial_baselines(train_ds, val_ds, rep: Report) -> None:
    """
    The MSE floor. A model is only interesting if it beats all three.
    best-single-feature-copy = min over input slots of MSE(slot value, target);
    pre-fix it was exactly 0.0, i.e. the task was solvable by copying.
    """
    def targets(ds):
        return torch.stack([ds[i][6] for i in range(len(ds))]).double()

    t_train = targets(train_ds)
    t_val = targets(val_ds)
    train_mean = float(t_train.mean())

    predict_zero = float((t_val ** 2).mean())
    predict_mean = float(((t_val - train_mean) ** 2).mean())
    val_variance = float(t_val.var(unbiased=False))

    # Copy baseline: for each (slot, step-position) use the first real step's
    # value at that slot as the prediction.
    slot_dim = val_ds[0][0].shape[1]
    sums = torch.zeros(slot_dim, dtype=torch.float64)
    cnt = 0
    for i in range(len(val_ds)):
        x, m, _, _, _, _, t = val_ds[i]
        real = x[m]
        if real.numel() == 0:
            continue
        last = real[-1].double()          # last real step: closest to the answer
        sums += (last - float(t)) ** 2
        cnt += 1
    copy_mse_per_slot = (sums / max(cnt, 1)).tolist()
    best_copy = float(min(copy_mse_per_slot))
    best_slot = int(copy_mse_per_slot.index(best_copy))

    rep.note("n_train", len(train_ds))
    rep.note("n_val", len(val_ds))
    rep.note("train_target_mean", train_mean)
    rep.note("val_target_variance", val_variance)
    rep.note("baseline_predict_zero_mse", predict_zero)
    rep.note("baseline_predict_train_mean_mse", predict_mean)
    rep.note("baseline_best_single_feature_copy_mse", best_copy)
    rep.note("baseline_copy_best_slot", best_slot)
    rep.note("baseline_copy_mse_per_slot", [round(v, 9) for v in copy_mse_per_slot])

    print(f"  [ .. ] predict-zero MSE            = {predict_zero:.6f}")
    print(f"  [ .. ] predict-train-mean MSE      = {predict_mean:.6f}")
    print(f"  [ .. ] best-single-feature-copy MSE= {best_copy:.6f} (slot {best_slot})")
    print(f"  [ .. ] val target variance         = {val_variance:.6f}")

    rep.check(
        "E. copy baseline is no longer free (>= predict-train-mean)",
        best_copy >= predict_mean * 0.999,
        f"copy {best_copy:.6f} vs predict-mean {predict_mean:.6f}; "
        f"pre-fix copy was 0.000000",
    )


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=os.path.join(_CODE_DIR, "config.json"))
    ap.add_argument("--json", default=None, help="write measurements here")
    ap.add_argument("--num_experts", type=int, default=6)
    a = ap.parse_args()

    cfg = load_config(a.config)
    dc = cfg["data"]
    train_path, val_path = dc.get("train_path"), dc.get("val_path")

    print("=" * 70)
    print("  Gate 1 audit -- leakage, oracle labels, input parity, baselines")
    print(f"  config     : {a.config}")
    print(f"  train      : {train_path}")
    print(f"  val        : {val_path}")
    print("=" * 70)

    rep = Report()
    train_ds = load_split(train_path, cfg, a.num_experts)
    val_ds = load_split(val_path, cfg, a.num_experts)

    print("\n[A] Target leakage")
    audit_result_mutation(val_path, cfg, rep)
    audit_target_leakage(val_ds, rep, "val")
    audit_target_leakage(train_ds, rep, "train")

    print("\n[B] Oracle expert label in input")
    audit_expert_permutation(val_path, cfg, rep)
    audit_low_cardinality_slots(val_ds, rep, a.num_experts)

    print("\n[C] Input parity across architectures")
    audit_input_parity(val_path, cfg, rep)

    print("\n[D] Split overlap")
    audit_split_overlap(
        {"train": train_path, "val": val_path, "test": dc.get("test_path")}, rep
    )

    print("\n[E] Trivial baselines (the MSE floor)")
    audit_trivial_baselines(train_ds, val_ds, rep)

    print("\n" + "-" * 70)
    passed = len(rep.rows) - len(rep.failed)
    print(f"  Gate 1: {passed}/{len(rep.rows)} checks passed")
    if rep.failed:
        print("  FAILED: " + ", ".join(rep.failed))
    print("-" * 70)

    if a.json:
        out = {"checks": [{"name": n, "passed": ok, "detail": d}
                          for n, ok, d in rep.rows],
               "measurements": rep.facts}
        os.makedirs(os.path.dirname(os.path.abspath(a.json)), exist_ok=True)
        with open(a.json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"  measurements written to {a.json}")

    return 1 if rep.failed else 0


if __name__ == "__main__":
    sys.exit(main())
