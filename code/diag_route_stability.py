# -*- coding: utf-8 -*-
"""Route-stability diagnostic for the MoRE language architecture.

QUESTION. `MoREWrapper.forward` calls `self.moe_block(...)` INSIDE the depth loop
(code/more/model.py:605 loop, :646 call), so a token's expert is re-chosen at every
recursion step, and the per-expert halt head at :723-728 is swapped along with it.
CLAUDE.md 1 defines MoRE as "a token is routed to a specialized expert, and THAT
EXPERT'S weights are recursively reused". Those are different architectures. This
measures how different they are in practice: what fraction of tokens keep their
depth-1 expert at depth d?

PREDICTION under the interference hypothesis. If assignments are near-random across
depth, agreement with depth 1 falls toward 1/E = 0.1667 and each expert receives an
incoherent token population -- which would explain (a) the 98.4% forced-exit rate,
since the halt head is re-selected every step and therefore never accumulates a
consistent halting policy for any token, (b) MoRE's routing AMI (0.141) sitting
BELOW MoE's (0.162) despite MoRE having strictly more routing opportunities, and
(c) MoRE being statistically indistinguishable from MoR (p=0.40), six experts
gradient-averaged toward one function.

If agreement is near 1.0 the hypothesis is dead and the depth loop is incidental.

PROVENANCE -- READ THIS BEFORE QUOTING ANY NUMBER FROM HERE. None of the 15
canonical cells retains weights: `*.pt` is gitignored (.gitignore:47) and the rented
V100 that produced the canonical matrix was terminated, so `checkpoint.pt` is MISSING
for every canonical run directory. Every checkpoint used below is therefore a PROXY
under CLAUDE.md 6 -- non-canonical epoch count and/or non-canonical corpus -- and may
never appear in a headline table or be substituted for a canonical cell. What makes
the measurement still valid: route stability is a property of the forward graph, which
is bit-identical across these runs and the canonical ones (same code path, E=6,
max_depth=7, top1_sparse, router_noise=none).

METHOD. Two passes per checkpoint.
  adaptive  -- the model exactly as trained. Tokens halt, so the active set shrinks
               and the per-depth expert vectors stop being row-aligned with depth 1.
               Comparison is truncated at the first depth whose width differs, which
               is honest but caps coverage on runs that halt early.
  fixed     -- `fixed_depth=True` at inference only. No weight or shape changes; it
               only stops tokens from halting, so all N tokens are present at all 7
               depths and agreement is exactly aligned. Cost: a token that would have
               halted keeps updating, so its late-depth state is off-distribution.
               On the canonical-corpus proxy 96.3% of tokens force-exit anyway, so
               this touches under 4% of the population there.
Reporting both means neither artifact can hide the answer.

NO GPU TRAINING RUN IS STARTED. Forward-only over the val split.
"""
import json
import os
import statistics
import sys
from pathlib import Path

import torch

# Resolved from this file so the diagnostic runs from any checkout or worktree.
# Hard-coding one machine's path is what made the first version of this script
# unrunnable on Ayan's machine (CLAUDE.md 9: interpreter and repo paths are
# per-machine and are NOT interchangeable).
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "code"))

from more.model import MoREModel, MoEBlock                       # noqa: E402
from more.lang_data import MoRELanguageDataset                   # noqa: E402
from more.config import (CANONICAL_ROUTER_NOISE,                 # noqa: E402
                         ROUTER_NOISE_INIT_SCALE_DEFAULT,
                         ROUTER_NOISE_ANNEAL_STEPS_DEFAULT,
                         ROUTING_PERSISTENCE_LEGACY,
                         NUM_EXPERTS_CANONICAL, TASK_LANGUAGE)
from torch.utils.data import DataLoader                          # noqa: E402

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")


def discover():
    """Every language MoRE checkpoint whose forward graph matches the canonical arm.

    Filtering on (num_experts, max_depth, routing_mode) rather than on run name is
    deliberate: the question is about the graph, and run names in `runs/` encode
    seed and config hash, not architecture.
    """
    out = []
    for cfg in sorted((REPO / "runs").glob("*/resolved_config.json")):
        d = cfg.parent
        if not (d / "checkpoint.pt").exists():
            continue
        try:
            rc = json.loads(cfg.read_text(encoding="utf-8"))
            mt = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
        except Exception:
            continue
        if (rc.get("task") or "arithmetic") != TASK_LANGUAGE:
            continue
        mc = rc.get("model", {})
        if mc.get("num_experts") != 6 or mc.get("max_depth") != 7:
            continue
        if mc.get("routing_mode", "top1_sparse") != "top1_sparse":
            continue
        corpus = rc.get("data", {}).get("corpus")
        if not (REPO / "data" / "lang" / str(corpus) / "val.npy").exists():
            continue
        out.append((d, rc, mt))
    return out


def build(mc, dc, force_fixed_depth):
    return MoREModel(
        step_feat_dim=mc["step_feat_dim"], d_model=mc["d_model"],
        num_experts=mc["num_experts"], max_depth=mc["max_depth"],
        num_blocks=mc["num_blocks"], dropout=mc["dropout"],
        fixed_depth=bool(force_fixed_depth or mc.get("fixed_depth", False)),
        routing_mode=mc.get("routing_mode", "top1_sparse"),
        router_noise=mc.get("router_noise", CANONICAL_ROUTER_NOISE),
        router_noise_init=mc.get("router_noise_init", ROUTER_NOISE_INIT_SCALE_DEFAULT),
        router_noise_anneal_steps=mc.get("router_noise_anneal_steps",
                                         ROUTER_NOISE_ANNEAL_STEPS_DEFAULT),
        ffn_mult=mc.get("ffn_mult", 4), num_families=NUM_EXPERTS_CANONICAL,
        attention=bool(mc.get("attention", False)), n_heads=int(mc.get("n_heads", 4)),
        max_seq_len=int(dc["seq_len"]), task=TASK_LANGUAGE,
        vocab_size=int(dc["vocab_size"]),
        # T-LX.12. Read off the run's OWN config with the legacy default, so this
        # script measures the architecture that run actually had. Hard-coding either
        # value would make the diagnostic answer a different question than the one
        # its docstring asks -- per_token would report a flat 1.0000 for every
        # checkpoint including the pre-fix ones, and per_step would keep reporting a
        # decay curve for runs that no longer re-route.
        routing_persistence=mc.get("routing_persistence",
                                   ROUTING_PERSISTENCE_LEGACY),
    ).to(device)


# ---- capture every per-depth argmax by wrapping MoEBlock.forward -------------
# Wrapping rather than editing the model keeps the diagnostic strictly read-only
# with respect to the canonical code path: nothing in `runs/` or `code/` changes.
_CAPTURE: list = []
_orig_forward = MoEBlock.forward


def _capturing_forward(self, x, route=None):
    # T-LX.12 added the `route` parameter to MoEBlock.forward -- it is how the depth
    # loop hands back the persisted (expert_idx, gate) pair at depths >= 2. The
    # wrapper must FORWARD it, not swallow it: a signature of (self, x) raises
    # TypeError on a per_token model, and a signature that accepted it and dropped it
    # would silently make every model look like the legacy per-step one, i.e. this
    # diagnostic would report the very defect it is measuring even after the fix.
    out = _orig_forward(self, x, route=route)
    _CAPTURE.append(out[2].detach().to("cpu"))      # expert_idx, [N_active]
    return out


def measure(model, loader, E, MAXD, max_batches=None):
    """Return (agree_per_depth, switch_rate, expert_load, depth_coverage)."""
    agree, seen = [0] * MAXD, [0] * MAXD
    switches = transitions = 0
    hist = torch.zeros(E, dtype=torch.long)
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            if max_batches is not None and bi >= max_batches:
                break
            batch = [b.to(device) if torch.is_tensor(b) else b for b in batch]
            _CAPTURE.clear()
            model(*batch[:4])
            if not _CAPTURE:
                continue
            base = _CAPTURE[0]
            N = base.numel()
            prev = base
            for step, idx in enumerate(_CAPTURE[:MAXD]):
                if idx.numel() != N:
                    break                       # active set shrank: rows no longer align
                agree[step] += int((idx == base).sum())
                seen[step] += N
                if step > 0:
                    switches += int((idx != prev).sum())
                    transitions += N
                prev = idx
                hist += torch.bincount(idx, minlength=E)
    a = [(agree[i] / seen[i]) if seen[i] else None for i in range(MAXD)]
    sw = (switches / transitions) if transitions else None
    load = (hist.float() / max(1, int(hist.sum()))).tolist()
    cov = sum(1 for v in a if v is not None)
    return a, sw, load, cov


rows = []
found = discover()
print(f"same-architecture language MoRE checkpoints found: {len(found)}\n")

MoEBlock.forward = _capturing_forward
try:
    for d, rc, mt in found:
        mc, dc, tc = rc["model"], rc["data"], rc["training"]
        E, MAXD = mc["num_experts"], mc["max_depth"]
        canon_corpus = dc.get("dataset_version", "").startswith("lang-wikitext-103-bpe8192-len256-800d6154")

        ds = MoRELanguageDataset(dc["corpus"], "val")
        loader = DataLoader(ds, batch_size=min(int(tc["batch_size"]), len(ds)),
                            shuffle=False, num_workers=0)
        state = torch.load(d / "checkpoint.pt", map_location=device, weights_only=True)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]

        res = {}
        for mode in ("adaptive", "fixed"):
            model = build(mc, dc, force_fixed_depth=(mode == "fixed"))
            model.load_state_dict(state)
            model.eval()
            res[mode] = measure(model, loader, E, MAXD)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

        rows.append({
            "run": d.name, "canonical_corpus": canon_corpus,
            "epochs": tc["epochs"], "corpus": dc["corpus"],
            "val_task_loss": mt.get("val/task_loss"),
            "forced_exit": mt.get("val/forced_exit_rate"),
            "avg_depth": mt.get("val/avg_depth"),
            "ami": mt.get("val/routing_ami"),
            "adaptive": {"agree": res["adaptive"][0], "switch": res["adaptive"][1],
                         "load": res["adaptive"][2], "depths": res["adaptive"][3]},
            "fixed": {"agree": res["fixed"][0], "switch": res["fixed"][1],
                      "load": res["fixed"][2], "depths": res["fixed"][3]},
        })
        f = res["fixed"]
        print("  %-38s %s  switch/step=%s  agree@d7=%s  depths_aligned(adaptive)=%d"
              % (d.name, "CANON-CORPUS" if canon_corpus else "proxy-corpus",
                 "--" if f[1] is None else "%.4f" % f[1],
                 "--" if f[0][MAXD - 1] is None else "%.4f" % f[0][MAXD - 1],
                 res["adaptive"][3]))
finally:
    MoEBlock.forward = _orig_forward

if not rows:
    print("no runs evaluated")
    raise SystemExit(1)

E, MAXD = 6, 7
print()
print("=" * 86)
print("ROUTE STABILITY ACROSS RECURSION DEPTH  (PROXY checkpoints -- see module docstring)")
print("=" * 86)
print("  chance level for E=%d is %.4f ; perfect persistence is 1.0000" % (E, 1.0 / E))
print()
hdr = "  %-36s %6s %7s %7s %7s %7s" % ("run", "epochs", "forced", "AMI", "sw/step", "agree@7")
print(hdr)
print("  " + "-" * (len(hdr) - 2))
for r in rows:
    f = r["fixed"]
    print("  %-36s %6s %7s %7s %7s %7s%s" % (
        r["run"][:36], r["epochs"],
        "%.3f" % r["forced_exit"] if isinstance(r["forced_exit"], float) else "--",
        "%.3f" % r["ami"] if isinstance(r["ami"], float) else "--",
        "%.4f" % f["switch"] if f["switch"] is not None else "--",
        "%.4f" % f["agree"][MAXD - 1] if f["agree"][MAXD - 1] is not None else "--",
        "  <- canonical corpus" if r["canonical_corpus"] else ""))

print()
print("  agreement with the depth-1 expert, halting disabled so all tokens are present:")
print("  %-36s %s" % ("run", " ".join("d%d" % (i + 1) for i in range(MAXD))))
for r in rows:
    a = r["fixed"]["agree"]
    print("  %-36s %s" % (r["run"][:36],
                          " ".join("--   " if v is None else "%.3f" % v for v in a)))

print()
print("  per-expert load (fixed-depth pass), share of all depth-level dispatches:")
print("  %-36s %s" % ("run", " ".join("E%d" % (i + 1) for i in range(E))))
for r in rows:
    print("  %-36s %s" % (r["run"][:36],
                          " ".join("%.3f" % v for v in r["fixed"]["load"][:E])))

canon = [r for r in rows if r["canonical_corpus"]]
if canon:
    print()
    print("  CANONICAL-CORPUS PROXY (architecture identical to the frozen MoRE arm):")
    for r in canon:
        f = r["fixed"]
        print("    %s  epochs=%s (canonical is 3)" % (r["run"], r["epochs"]))
        print("    switch rate per recursion step : %.4f" % f["switch"])
        print("    P(expert@d7 == expert@d1)      : %.4f  (chance %.4f)"
              % (f["agree"][MAXD - 1], 1.0 / E))
        print("    recorded forced-exit rate      : %.4f" % r["forced_exit"])
        print("    recorded routing AMI           : %.4f" % r["ami"])

out = str(REPO / "results" / "language" / "diagnostics" / "route_stability.json")
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w", encoding="utf-8") as fh:
    json.dump(rows, fh, indent=1)
print("\n  written: %s" % out)


# ---------------------------------------------------------------------------
# PART 2 -- the consequence, measured on the CANONICAL matrix.
#
# Part 1 establishes that the expert is re-drawn during recursion. Part 2 asks
# what that costs, and unlike Part 1 it needs no checkpoints: every number below
# is already in the canonical `metrics.json` files, so this half IS canonical
# evidence and may be quoted as such.
#
# The comparison is MoR (E=1, so the router is a no-op and the single halt head
# governs every token at every step) against MoRE (E=6, so both the expert and
# the halt head are re-drawn per step). MoE is excluded because `max_depth = 1`
# leaves it with no depth to allocate.
#
# WHAT IS AND IS NOT CONTROLLED. MoR and MoRE differ in three things at once:
# expert count (1 vs 6), halt-head count (1 vs 6), and `ffn_mult` (24 vs 4, set
# to match total parameters at 5.58M). So this pair CANNOT by itself attribute
# the loss of adaptive depth to re-routing -- it establishes that adaptive depth
# is present in one arm and absent in the other. The attributing experiment is a
# MoRE variant that freezes the depth-1 assignment, which holds E, halt-head
# count and ffn_mult fixed and moves only the re-routing.
# ---------------------------------------------------------------------------
import glob                                                      # noqa: E402
import itertools                                                 # noqa: E402

DEPTH_KEYS = [
    ("depth/spearman_vs_model_loss", "depth vs per-token model loss"),
    ("depth/spearman_vs_logfreq", "depth vs token log-frequency"),
    ("depth/embed_norm_logfreq_partial_norm", "  same, embed-norm corrected"),
    ("depth/by_document/between_share", "between-document share of var"),
    ("val/forced_exit_rate", "forced-exit rate"),
    ("depth/mean", "mean exit depth"),
    ("val/task_loss", "val task loss"),
]


def permutation_p(a, b):
    """Exact two-sided permutation test. With n=5 vs n=5 the floor is 2/252."""
    obs = abs(statistics.fmean(a) - statistics.fmean(b))
    pool = list(a) + list(b)
    ge = tot = 0
    for combo in itertools.combinations(range(len(pool)), len(a)):
        s = set(combo)
        x = [pool[i] for i in combo]
        y = [pool[i] for i in range(len(pool)) if i not in s]
        tot += 1
        if abs(statistics.fmean(x) - statistics.fmean(y)) >= obs - 1e-12:
            ge += 1
    return obs, ge / tot, tot


canon = {}
for cfg in sorted(glob.glob(str(REPO / "runs" / "*" / "resolved_config.json"))):
    d = os.path.dirname(cfg)
    try:
        rc = json.loads(open(cfg, encoding="utf-8").read())
        mt = json.loads(open(os.path.join(d, "metrics.json"), encoding="utf-8").read())
    except Exception:
        continue
    if rc.get("provenance", {}).get("experiment_group") != "canonical_lang_b":
        continue
    canon.setdefault(rc["architecture"], []).append(mt)

print()
print("=" * 86)
print("DEPTH ALLOCATION QUALITY -- CANONICAL MATRIX (no checkpoints needed)")
print("=" * 86)
if not all(len(canon.get(a, [])) == 5 for a in ("mor", "more")):
    print("  canonical MoR/MoRE cells not both at n=5 -- refusing to test")
else:
    print("  %-32s %12s %12s %10s" % ("quantity", "MoR (n=5)", "MoRE (n=5)", "perm p"))
    print("  " + "-" * 70)
    for key, label in DEPTH_KEYS:
        a = [m[key] for m in canon["mor"] if isinstance(m.get(key), (int, float))]
        b = [m[key] for m in canon["more"] if isinstance(m.get(key), (int, float))]
        if len(a) != 5 or len(b) != 5:
            print("  %-32s %12s %12s %10s" % (label, "N/A", "N/A", "--"))
            continue
        _, p, _ = permutation_p(a, b)
        print("  %-32s %+7.4f+-%.4f %+7.4f+-%.4f %10.4f"
              % (label, statistics.fmean(a), statistics.stdev(a),
                 statistics.fmean(b), statistics.stdev(b), p))
    print()
    print("  READING. A positive depth-vs-model-loss correlation means the model")
    print("  spends more recursion on the tokens it finds hard, which is the whole")
    print("  point of adaptive computation. Near zero means depth is being spent")
    print("  without regard to difficulty -- the halting head is running, but it is")
    print("  not allocating.")

