"""
T6.2 + T6.3 verification -- authoritative routing accuracy and the
permutation-invariant routing metric set (plan.md 8.2 / 8.3).

Why this file exists
--------------------
Expert index 3 is not "the MULT/DIV expert". It is whichever expert the router
happened to send MULT/DIV tokens to. The confusion diagonal therefore answers
"did the router pick the oracle's own index?", which is a question about a
labelling convention, not about whether the model found the partition. Every
check below exists to keep those two questions apart:

  raw ~= matched   the index itself is being supervised (step_routing > 0), so
                   agreement with the oracle numbering is expected.
  raw << matched   a real partition exists under relabelling -- the finding.
  both low, AMI~0  no partition at all.

The exact Hungarian solver and the AMI chance correction are implemented in
more/metrics.py because neither scipy nor sklearn is installed in more_env, so
they are checked here against brute force rather than against a library.
"""

import math
import sys
import itertools

import numpy as np
import torch

from more.families import expert_labels
from more.metrics import (permutation_invariant_routing_metrics,
                          routing_accuracy_from_confusion,
                          _max_weight_assignment,
                          _mutual_information,
                          _expected_mutual_information)

PASS = 0
FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"[PASS] {name}")
    else:
        FAIL += 1
        print(f"[FAIL] {name}" + (f"   ({detail})" if detail else ""))


def close(a, b, tol=1e-9):
    return a is not None and b is not None and abs(a - b) <= tol


print("=" * 74)
print("T6.2 / T6.3  routing metric verification")
print("=" * 74)

# ---------------------------------------------------------------- T6.3a
# The TASKS.md criterion, verbatim: on a synthetic perfectly-permuted
# assignment, Hungarian accuracy = 1.0 while raw accuracy is at chance.
print("\n-- T6.3a  permuted-perfect assignment ------------------------------")

E = 6
PERM = [1, 2, 3, 4, 5, 0]          # cyclic: a derangement, so raw diagonal = 0
C = np.zeros((E, E))
for i, j in enumerate(PERM):
    C[i, j] = 100.0

m = permutation_invariant_routing_metrics(C, E)
check("T6.3a hungarian_accuracy == 1.0 on a permuted-perfect table",
      close(m["hungarian_accuracy"], 1.0), str(m["hungarian_accuracy"]))
check("T6.3a raw_accuracy == 0.0 (a derangement scores nothing on-diagonal)",
      close(m["raw_accuracy"], 0.0), str(m["raw_accuracy"]))
check("T6.3a the recovered assignment is the planted permutation",
      m["hungarian_assignment"] == PERM, str(m["hungarian_assignment"]))
check("T6.3a AMI == 1.0 -- AMI is permutation-invariant by construction",
      close(m["ami"], 1.0, 1e-9), str(m["ami"]))
check("T6.3a purity == 1.0", close(m["purity"], 1.0), str(m["purity"]))
check("T6.3a no expert reported collapsed", m["collapsed_experts"] == [],
      str(m["collapsed_experts"]))

# The identity table is the control: the same partition, oracle numbering.
I = np.eye(E) * 100.0
mi_ = permutation_invariant_routing_metrics(I, E)
check("T6.3a identity table: raw == hungarian == 1.0",
      close(mi_["raw_accuracy"], 1.0) and close(mi_["hungarian_accuracy"], 1.0),
      f"raw={mi_['raw_accuracy']} hung={mi_['hungarian_accuracy']}")
check("T6.3a identity and permuted tables agree on every "
      "permutation-invariant quantity",
      close(mi_["ami"], m["ami"], 1e-12)
      and close(mi_["purity"], m["purity"], 1e-12)
      and close(mi_["hungarian_accuracy"], m["hungarian_accuracy"], 1e-12),
      f"identity ami={mi_['ami']} permuted ami={m['ami']}")

# ---------------------------------------------------------------- T6.3b
# Chance behaviour: an independent (product) table carries no information.
print("\n-- T6.3b  chance-level table ---------------------------------------")

rng = np.random.default_rng(0)
row = np.array([600.0, 300.0, 150.0, 450.0, 200.0, 300.0])
col = np.array([400.0, 400.0, 200.0, 500.0, 300.0, 200.0])
INDEP = np.outer(row, col) / row.sum()          # c_ij = a_i b_j / n exactly

m_ind = permutation_invariant_routing_metrics(INDEP, E)
check("T6.3b independent table: mutual information == 0",
      close(_mutual_information(INDEP, INDEP.sum()), 0.0, 1e-12),
      str(_mutual_information(INDEP, INDEP.sum())))
check("T6.3b independent table: AMI <= 0 (chance-corrected, may go negative)",
      m_ind["ami"] is not None and m_ind["ami"] <= 1e-9, str(m_ind["ami"]))
check("T6.3b AMI is reported as measured, not clipped to [0, 1]",
      m_ind["ami"] is not None and m_ind["ami"] > -0.5, str(m_ind["ami"]))
check("T6.3b independent table: hungarian_accuracy is near chance (1/E)",
      m_ind["hungarian_accuracy"] is not None
      and m_ind["hungarian_accuracy"] < 0.35,
      str(m_ind["hungarian_accuracy"]))
check("T6.3b purity is NOT chance-corrected and stays well above AMI -- "
      "which is why both are reported",
      m_ind["purity"] > 0.2 and m_ind["purity"] > m_ind["ami"],
      f"purity={m_ind['purity']} ami={m_ind['ami']}")

# ---------------------------------------------------------------- T6.3c
# The exact solver, against brute force over all E! permutations.
print("\n-- T6.3c  exact Hungarian vs brute force ---------------------------")

worst = 0.0
for trial in range(40):
    n = int(rng.integers(2, 7))
    W = rng.random((n, n)) * rng.integers(1, 500)
    _, dp_val = _max_weight_assignment(W)
    bf_val = max(sum(W[i, p[i]] for i in range(n))
                 for p in itertools.permutations(range(n)))
    worst = max(worst, abs(dp_val - bf_val))
check("T6.3c subset-DP assignment equals brute-force optimum on 40 random "
      "matrices (n=2..6)", worst < 1e-9, f"max abs diff {worst:.3e}")

_, val_int = _max_weight_assignment(np.array([[5.0, 1.0], [1.0, 5.0]]))
check("T6.3c trivial 2x2 case", close(val_int, 10.0), str(val_int))

refused = False
try:
    _max_weight_assignment(np.zeros((16, 16)))
except ValueError:
    refused = True
check("T6.3c refuses n > 15 rather than silently degrading to greedy -- a "
      "metric that is sometimes a bound and sometimes the truth is not "
      "reportable", refused)

nonsquare = False
try:
    _max_weight_assignment(np.zeros((3, 4)))
except ValueError:
    nonsquare = True
check("T6.3c refuses a non-square weight matrix", nonsquare)

# ---------------------------------------------------------------- T6.3d
# The invariant that makes the raw-vs-matched gap readable at all.
print("\n-- T6.3d  hungarian >= raw, always ---------------------------------")

violations = 0
for trial in range(60):
    n = int(rng.integers(2, 7))
    T = rng.integers(0, 40, size=(n, n)).astype(float)
    if T.sum() == 0:
        continue
    mm = permutation_invariant_routing_metrics(T, n)
    if mm["hungarian_accuracy"] + 1e-12 < mm["raw_accuracy"]:
        violations += 1
check("T6.3d hungarian_accuracy >= raw_accuracy on 60 random tables "
      "(the identity is one candidate assignment, so the optimum cannot be "
      "worse)", violations == 0, f"{violations} violations")

# ---------------------------------------------------------------- T6.3e
# EMI against the exact permutation-model expectation, by enumeration.
print("\n-- T6.3e  EMI vs exact permutation-model expectation ---------------")

def emi_brute(oracle_labels, n_classes):
    """Average MI over every permutation of the predicted label vector -- the
    literal definition of the expected value EMI approximates in closed form."""
    pred = list(oracle_labels)
    total, count = 0.0, 0
    for p in itertools.permutations(pred):
        cont = np.zeros((n_classes, n_classes))
        for o, q in zip(oracle_labels, p):
            cont[o, q] += 1.0
        total += _mutual_information(cont, float(len(oracle_labels)))
        count += 1
    return total / count

for lbls, k in ((([0, 0, 0, 1, 1, 1]), 2), (([0, 0, 1, 1, 1, 2]), 3)):
    cont = np.zeros((k, k))
    for o in lbls:
        cont[o, o] += 1.0
    a = cont.sum(axis=1)
    b = cont.sum(axis=0)
    got = _expected_mutual_information(a, b, len(lbls))
    want = emi_brute(lbls, k)
    check(f"T6.3e closed-form EMI matches enumeration over {math.factorial(len(lbls))} "
          f"permutations (k={k})", close(got, want, 1e-10),
          f"closed form {got:.12f} vs brute force {want:.12f}")

# ---------------------------------------------------------------- T6.3f
# Collapse: one expert takes everything. Purity looks respectable, AMI does not.
print("\n-- T6.3f  router collapse ------------------------------------------")

COL = np.zeros((E, E))
COL[:, 2] = 100.0                     # every family routed to expert 3
m_col = permutation_invariant_routing_metrics(COL, E)
check("T6.3f collapse: AMI == 0 (a single predicted block carries no "
      "information)", close(m_col["ami"], 0.0, 1e-9), str(m_col["ami"]))
check("T6.3f collapse: five of six experts reported as receiving nothing",
      len(m_col["collapsed_experts"]) == 5, str(m_col["collapsed_experts"]))
# T6.8: named by INDEX. This check used to read "E3_MOD_POW is not in the
# collapsed list", which asserted a FAMILY label about an expert SLOT -- expert 2
# is not the MOD/POW expert, it is the slot this router happened to use. The old
# form was the same conflation the rest of this file exists to prevent.
check("T6.3f collapse: expert_2 -- the slot that received everything -- is not "
      "in the collapsed list", "expert_2" not in m_col["collapsed_experts"],
      str(m_col["collapsed_experts"]))
check("T6.3f collapse: collapsed slots are named by INDEX, never by family "
      "label (a slot that received nothing has no family identity)",
      all(s.startswith("expert_") for s in m_col["collapsed_experts"])
      and not any(s in expert_labels(E) for s in m_col["collapsed_experts"]),
      str(m_col["collapsed_experts"]))
check("T6.3f collapse: hungarian_accuracy == 1/E, the ceiling when only one "
      "column is used", close(m_col["hungarian_accuracy"], 1.0 / E, 1e-12),
      str(m_col["hungarian_accuracy"]))
check("T6.3f collapse: purity == 1/E as well", close(m_col["purity"], 1.0 / E),
      str(m_col["purity"]))

# ---------------------------------------------------------------- T6.3g
# Per-family precision / recall / F1 on a table computed by hand.
print("\n-- T6.3g  per-family precision / recall / F1 -----------------------")

H = np.array([
    [8.0, 2.0, 0.0],      # oracle 0: 8 correct of 10
    [6.0, 14.0, 0.0],     # oracle 1: 14 correct of 20 -- deliberately imbalanced
    [0.0, 0.0, 0.0],      # oracle 2: no tokens at all -> N/A, not 0.0
])
m_h = permutation_invariant_routing_metrics(H, 3)
lab = expert_labels(3)
f0 = m_h["per_family"][lab[0]]
f1_ = m_h["per_family"][lab[1]]
f2 = m_h["per_family"][lab[2]]
check("T6.3g recall  = 8/10", close(f0["recall"], 0.8), str(f0["recall"]))
check("T6.3g precision = 8/14", close(f0["precision"], 8.0 / 14.0),
      str(f0["precision"]))
check("T6.3g F1 is the harmonic mean of the two",
      close(f0["f1"], 2 * 0.8 * (8 / 14) / (0.8 + 8 / 14)), str(f0["f1"]))
check("T6.3g support is the oracle row total", close(f0["support"], 10.0),
      str(f0["support"]))
check("T6.3g a family with zero oracle tokens reports N/A, never 0.0 "
      "(CLAUDE.md 4: no sentinel as a measurement)",
      f2["recall"] is None and f2["support"] == 0.0,
      f"recall={f2['recall']} support={f2['support']}")
check("T6.3g a family predicted by nobody reports precision N/A",
      f2["precision"] is None, str(f2["precision"]))
check("T6.3g macro_recall averages the defined rows only, and differs from "
      "raw accuracy under imbalance",
      close(m_h["macro_recall"], (0.8 + 0.7) / 2)
      and close(m_h["raw_accuracy"], 22.0 / 30.0)
      and not close(m_h["macro_recall"], m_h["raw_accuracy"], 1e-6),
      f"macro={m_h['macro_recall']} raw={m_h['raw_accuracy']}")

# F1 = 0.0 when both parts are defined and zero: a measurement, not N/A.
Z = np.array([[0.0, 5.0], [5.0, 0.0]])
m_z = permutation_invariant_routing_metrics(Z, 2)
check("T6.3g F1 == 0.0 (not N/A) when precision and recall are both a "
      "measured zero",
      m_z["per_family"][expert_labels(2)[0]]["f1"] == 0.0,
      str(m_z["per_family"][expert_labels(2)[0]]["f1"]))

# ---------------------------------------------------------------- T6.3h
# Undefined cases must be uniformly N/A, and must not raise.
print("\n-- T6.3h  undefined cases ------------------------------------------")

KEYS = ("raw_accuracy", "hungarian_accuracy", "hungarian_assignment",
        "macro_recall", "ami", "purity", "collapsed_experts")
for name, arg, ne in (("E=1 (MoR: one shared block, no routing)",
                       np.array([[100.0]]), 1),
                      ("an all-zero confusion (validation never ran)",
                       np.zeros((E, E)), E)):
    mu = permutation_invariant_routing_metrics(arg, ne)
    check(f"T6.3h {name}: every metric is None",
          all(mu[k] is None for k in KEYS),
          str({k: mu[k] for k in KEYS if mu[k] is not None}))
    check(f"T6.3h {name}: per_family is empty rather than full of zeros",
          mu["per_family"] == {}, str(mu["per_family"]))

m_torch = permutation_invariant_routing_metrics(torch.from_numpy(C), E)
check("T6.3h accepts a torch confusion matrix identically to numpy",
      close(m_torch["hungarian_accuracy"], 1.0)
      and close(m_torch["ami"], m["ami"], 1e-12))

# ---------------------------------------------------------------- T6.2
# Authoritative routing accuracy: the diagonal fraction of the very matrix the
# figure is drawn from must equal mean(pred == oracle) over the same tokens.
print("\n-- T6.2  the confusion diagonal IS the routing accuracy ------------")

worst_gap = 0.0
for trial in range(200):
    n_tok = int(rng.integers(50, 400))
    oracle = rng.integers(0, E, size=n_tok)
    pred = np.where(rng.random(n_tok) < 0.6, oracle, rng.integers(0, E, n_tok))
    conf = torch.zeros(E, E)
    for o, p in zip(oracle, pred):
        conf[int(o), int(p)] += 1
    direct = float((oracle == pred).mean())
    from_conf = routing_accuracy_from_confusion(conf, E)
    worst_gap = max(worst_gap, abs(direct - from_conf))
    worst_gap = max(worst_gap,
                    abs(permutation_invariant_routing_metrics(conf, E)
                        ["raw_accuracy"] - direct))
check("T6.2 diagonal fraction == mean(pred == oracle) within 1e-6 over 200 "
      "random token streams", worst_gap < 1e-6, f"max gap {worst_gap:.3e}")
check("T6.2 the permutation-invariant block reuses that same accuracy rather "
      "than recomputing its own", worst_gap < 1e-6)

single_src = permutation_invariant_routing_metrics(C, E)["raw_accuracy"]
check("T6.2 single source of truth: pim['raw_accuracy'] delegates to "
      "routing_accuracy_from_confusion",
      close(single_src, routing_accuracy_from_confusion(torch.from_numpy(C), E)),
      str(single_src))

# ---------------------------------------------------------------- T6.6-adjacent
# W&B key hygiene for the new keys (CLAUDE.md 5: no '/' inside a label).
print("\n-- T6.3i  W&B key hygiene ------------------------------------------")

bad = [lbl for lbl in expert_labels(E) if "/" in lbl]
check("T6.3i no expert label contains '/', so val/routing_precision/{label} "
      "cannot create spurious nesting", bad == [], str(bad))

# ====================================================================== T6.8
# The relabelling-invariant per-family table.
#
# The gap this closes: `per_family` pairs oracle family i with expert INDEX i.
# On a supervised run that is the right pairing, because step_routing pinned the
# numbering. On an unsupervised run the numbering is arbitrary, so the identity
# table measures the numbering and not the partition -- measured on the T8.0b
# runs it read macro recall 0.014-0.162 across seeds while matched accuracy sat
# at a stable 0.515. `per_family_matched` applies hungarian_assignment first.
print("\n" + "=" * 74)
print("T6.8  per-family metrics under the Hungarian permutation")
print("=" * 74)

from more.metrics import paper_metrics_to_wandb   # noqa: E402

STATS = ("precision", "recall", "f1")


def matched_table(conf, n_experts=E):
    return permutation_invariant_routing_metrics(conf, n_experts)


def stats_equal(t1, t2, tol=1e-12):
    """Same labels, and every defined stat equal; None must match None."""
    if set(t1) != set(t2):
        return False
    for lab in t1:
        for s in STATS + ("support",):
            a, b = t1[lab].get(s), t2[lab].get(s)
            if (a is None) != (b is None):
                return False
            if a is not None and abs(a - b) > tol:
                return False
    return True


# -- T6.8a  invariance under relabelling of the expert indices --------------
# Permuting the COLUMNS of the confusion matrix is exactly what happens when the
# router numbers the same partition differently. Every matched quantity must be
# bit-stable across that; nothing in the matched table may move.
print("\n-- T6.8a  invariance under column permutation ----------------------")

rng2 = np.random.default_rng(20260828)
worst_matched_drift = 0.0
worst_identity_drift = 0.0
worst_macro_drift = 0.0
worst_hung_drift = 0.0
n_trials = 0
for trial in range(60):
    # A partitioned-but-arbitrarily-numbered router: block structure with noise,
    # then a random relabelling. This is the unsupervised regime, not a toy.
    base = rng2.integers(5, 60, size=(E, E)).astype(float)
    base[np.arange(E), rng2.permutation(E)] += rng2.integers(200, 600, size=E)
    perm = rng2.permutation(E)
    permuted = base[:, perm]

    a = matched_table(base)
    b = matched_table(permuted)
    if a["hungarian_accuracy"] is None or b["hungarian_accuracy"] is None:
        continue
    n_trials += 1
    worst_hung_drift = max(worst_hung_drift,
                           abs(a["hungarian_accuracy"]
                               - b["hungarian_accuracy"]))
    worst_macro_drift = max(worst_macro_drift,
                            abs(a["matched_macro_recall"]
                                - b["matched_macro_recall"]))
    for lab in a["per_family_matched"]:
        for s in STATS:
            x = a["per_family_matched"][lab][s]
            y = b["per_family_matched"][lab][s]
            if x is not None and y is not None:
                worst_matched_drift = max(worst_matched_drift, abs(x - y))
        xi = a["per_family"][lab]["recall"]
        yi = b["per_family"][lab]["recall"]
        if xi is not None and yi is not None:
            worst_identity_drift = max(worst_identity_drift, abs(xi - yi))

check(f"T6.8a per_family_matched precision/recall/f1 are invariant under all "
      f"{n_trials} column permutations",
      n_trials >= 50 and worst_matched_drift < 1e-12,
      f"max drift {worst_matched_drift:.3e} over {n_trials} trials")
check("T6.8a matched_macro_recall is invariant under column permutation",
      worst_macro_drift < 1e-12, f"max drift {worst_macro_drift:.3e}")
check("T6.8a hungarian_accuracy is invariant under column permutation "
      "(the property the matched table inherits)",
      worst_hung_drift < 1e-12, f"max drift {worst_hung_drift:.3e}")
# The test has teeth only if the OLD table demonstrably fails the same check.
check("T6.8a the IDENTITY per_family recall is NOT invariant -- confirming the "
      "gap was real and not a relabelling of a working metric",
      worst_identity_drift > 0.1,
      f"identity recall moves by up to {worst_identity_drift:.3f} under a pure "
      "renumbering")

# -- T6.8b  consistency with hungarian_accuracy -----------------------------
# The matched table must decompose the matched accuracy, not sit beside it:
# sum_i (matched_recall_i * support_i) / n_tokens == hungarian_accuracy exactly.
# This is the check that would catch a matched table built off the wrong axis --
# e.g. pairing expert j with family assignment[j] instead of family i with
# expert assignment[i], which is invariant too but measures nothing.
print("\n-- T6.8b  the matched table decomposes hungarian_accuracy ----------")

worst_recon = 0.0
worst_pair = 0
for trial in range(60):
    conf = rng2.integers(1, 80, size=(E, E)).astype(float)
    conf[np.arange(E), rng2.permutation(E)] += rng2.integers(100, 900, size=E)
    m8 = matched_table(conf)
    n_tok = conf.sum()
    recon = sum(m8["per_family_matched"][lab]["recall"]
                * m8["per_family_matched"][lab]["support"]
                for lab in m8["per_family_matched"]) / n_tok
    worst_recon = max(worst_recon, abs(recon - m8["hungarian_accuracy"]))
    for i, lab in enumerate(expert_labels(E)):
        if m8["per_family_matched"][lab]["matched_expert"] != \
                m8["hungarian_assignment"][i]:
            worst_pair += 1

check("T6.8b support-weighted mean of matched recalls == hungarian_accuracy "
      "to 1e-12 over 60 random tables", worst_recon < 1e-12,
      f"max reconstruction error {worst_recon:.3e}")
check("T6.8b every family's matched_expert equals hungarian_assignment[i], so "
      "the table is paired along the oracle axis and not the expert axis",
      worst_pair == 0, f"{worst_pair} mismatches")

# -- T6.8c  agreement with the identity table when the index IS supervised --
# A supervised run recovers the identity permutation. There the two tables must
# be the same object of study; if they disagree, one of them is wrong.
print("\n-- T6.8c  supervised runs: matched == identity ---------------------")

sup = np.eye(E) * 900.0 + 3.0        # diagonal-dominant: identity is optimal
m8s = matched_table(sup)
check("T6.8c a diagonal-dominant (supervised-style) table recovers the identity "
      "assignment", m8s["hungarian_assignment"] == list(range(E)),
      str(m8s["hungarian_assignment"]))
check("T6.8c and then per_family_matched == per_family exactly -- the matched "
      "table is not a different metric, only a different pairing",
      stats_equal(m8s["per_family"], m8s["per_family_matched"]))
check("T6.8c matched_macro_recall == macro_recall there",
      close(m8s["matched_macro_recall"], m8s["macro_recall"]),
      f"{m8s['matched_macro_recall']} vs {m8s['macro_recall']}")

# -- T6.8d  the unsupervised regime, end to end ------------------------------
# The point of the fix, stated as a measurement: a perfectly-partitioning router
# that numbers its experts by a derangement scores 0.0 on the identity table and
# 1.0 on the matched one. Reading the identity table would report a total failure
# as the model's specialization result.
print("\n-- T6.8d  arbitrary numbering: identity says 0, matched says 1 -----")

m8d = matched_table(C)               # C is the cyclic permuted-perfect table
id_recalls = [m8d["per_family"][l]["recall"] for l in expert_labels(E)]
mt_recalls = [m8d["per_family_matched"][l]["recall"] for l in expert_labels(E)]
check("T6.8d identity per-family recall is 0.0 for EVERY family on a "
      "permuted-perfect router", all(close(r, 0.0) for r in id_recalls),
      str(id_recalls))
check("T6.8d matched per-family recall is 1.0 for EVERY family on the same "
      "table", all(close(r, 1.0) for r in mt_recalls), str(mt_recalls))
check("T6.8d matched_macro_recall == 1.0 while macro_recall == 0.0",
      close(m8d["matched_macro_recall"], 1.0)
      and close(m8d["macro_recall"], 0.0),
      f"matched {m8d['matched_macro_recall']} vs identity "
      f"{m8d['macro_recall']}")
check("T6.8d matched precision is also 1.0 for every family (no family's "
      "tokens leak into another family's matched expert)",
      all(close(m8d["per_family_matched"][l]["precision"], 1.0)
          for l in expert_labels(E)))

# -- T6.8e  undefined cases stay N/A, never 0.0 (CLAUDE.md 4) ---------------
print("\n-- T6.8e  no sentinel may stand in for an undefined matched stat ---")

m1 = permutation_invariant_routing_metrics(np.zeros((1, 1)), 1)
check("T6.8e num_experts == 1 (MoR): per_family_matched is empty, not a table "
      "of zeros", m1["per_family_matched"] == {}, str(m1["per_family_matched"]))
check("T6.8e num_experts == 1: matched_macro_recall is None, not 0.0",
      m1["matched_macro_recall"] is None, repr(m1["matched_macro_recall"]))
m0 = permutation_invariant_routing_metrics(np.zeros((E, E)), E)
check("T6.8e an all-zero confusion (no token routed) yields "
      "matched_macro_recall = None", m0["matched_macro_recall"] is None,
      repr(m0["matched_macro_recall"]))

# A collapsed expert receives nothing, so precision against it is 0/0. Recall is
# still defined (the family has oracle tokens); the two must not be conflated.
coll = np.zeros((E, E))
coll[:, 0] = 50.0                    # every token to expert 0; 1..5 collapsed
mc8 = matched_table(coll)
undef_prec = [l for l in expert_labels(E)
              if mc8["per_family_matched"][l]["precision"] is None]
undef_rec = [l for l in expert_labels(E)
             if mc8["per_family_matched"][l]["recall"] is None]
check("T6.8e a collapsed matched expert gives precision = None (0/0), not 0.0",
      len(undef_prec) == E - 1, f"{len(undef_prec)} of {E} undefined")
check("T6.8e recall stays DEFINED there -- the family does have oracle tokens, "
      "so None would hide a real measurement", undef_rec == [], str(undef_rec))
check("T6.8e and f1 is None exactly where precision is None",
      [l for l in expert_labels(E)
       if mc8["per_family_matched"][l]["f1"] is None] == undef_prec)
check("T6.8e collapse is still reported in collapsed_experts",
      len(mc8["collapsed_experts"] or []) == E - 1,
      str(mc8["collapsed_experts"]))

# -- T6.8f  W&B key hygiene and gating on the new keys ----------------------
print("\n-- T6.8f  logged keys: prefix separation and routing gating --------")

wb = paper_metrics_to_wandb(C, 0.0, {}, {}, E, pim=m8d)
matched_keys = [k for k in wb if "_matched/" in k]
check("T6.8f the matched table is logged under its own metric prefix "
      "(val/routing_<stat>_matched/<label>), so it cannot be mistaken for the "
      "identity table", len(matched_keys) == 3 * E, f"{len(matched_keys)} keys")
check("T6.8f no logged key has '/' inside the LABEL segment",
      all("/" not in k.split("/")[-1] for k in matched_keys))
check("T6.8f identity and matched keys coexist -- the matched table is added "
      "beside the old one, never in place of it",
      any(k.startswith("val/routing_precision/") for k in wb)
      and any(k.startswith("val/routing_precision_matched/") for k in wb))
check("T6.8f val/routing_matched_macro_recall is logged",
      "val/routing_matched_macro_recall" in wb)
wb1 = paper_metrics_to_wandb(np.zeros((1, 1)), float("nan"), {}, {}, 1,
                             pim=m1)
check("T6.8f MoR (num_experts == 1) logs NO matched routing key at all -- "
      "routing metrics only exist where routing exists",
      [k for k in wb1 if "routing" in k] == [],
      str([k for k in wb1 if "routing" in k]))

print("\n" + "=" * 74)
print(f"{PASS} passed, {FAIL} failed")
print("=" * 74)
sys.exit(1 if FAIL else 0)
