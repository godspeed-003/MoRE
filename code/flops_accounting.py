"""
T-LX.15 -- measured FLOPs per arm, so "compute-matched" stops being an assumption.

WHY THIS EXISTS
---------------
Two independent sources demanded it in the same week:

  * The internal audit of the language matrix found that the three arms are
    parameter-matched BY CONSTRUCTION (MoE 6x4, MoR 1x24, MoRE 6x4 FFN units,
    all 24) but say nothing about ACTIVE compute: MoR runs one 24x-wide FFN with
    every unit active, MoRE runs one of six 4x experts, so 1/6 of its stored FFN
    is active per token, and both then multiply by mean recursion depth.

  * NeurIPS-workshop reviewer 652J, concern 3: "The clean comparison is MoRE
    versus parameter-identical MoE; MoRE versus MoR needs an iso-FLOP or
    iso-latency control."

Until this file existed the repository had NO FLOPs accounting anywhere -- only
two comments flagging its absence (engine.py:1458, metrics.py:1647), both saying
the depth-allocation-error metric must not be called "compute efficiency"
because no FLOPs are in it. Any multiplier quoted before this file was a hand
derivation of the FFN term alone and silently ignored attention, embeddings and
the 8192-way LM head, which on a 256-token block are NOT negligible.

METHOD
------
`torch.utils.flop_counter.FlopCounterMode` counts what actually executes, on a
real forward pass, for the arm as its own resolved_config.json built it. No
formula is typed in, so the count cannot drift from the model.

Depth is swept with `fixed_depth=True` and `max_depth = k` for k = 1..7, which
makes every token take exactly k steps. The recursive part of the graph is exact-
ly linear in k, so a two-point fit gives

    FLOPs(d) = base + (d - 1) * per_step

where `base` is the k=1 count (embeddings, positional table, LM head, one block
pass) and `per_step` is the marginal cost of one extra recursion step. The fit is
VERIFIED against the measured k=7 count rather than assumed; a residual above
0.1% is a hard failure, because a non-linearity there would mean the recursion is
not doing what the architecture claims.

Reporting then evaluates that line at each arm's MEASURED mean depth from its
canonical metrics.json -- not at max_depth, which would overstate MoR and MoRE
by however much early exit actually buys.

CAVEAT THAT MUST TRAVEL WITH THESE NUMBERS
------------------------------------------
FLOPs are an analytic cost, not a runtime. The measured throughputs on the same
V100 are MoE 718.8, MoR 173.8, MoRE 168.5 items/s -- MoR and MoRE within 3% of
each other despite the FLOP gap this file measures, because MoRE spends its
theoretical advantage on gather/scatter in the top-1 dispatch and never runs a
fused sparse kernel. Quoting the FLOP ratio as though it were a speedup would
invert the truth (CLAUDE.md 4: do not call an allocation metric "compute
efficiency"). Report both, always, and say which one a claim rests on.

USAGE
    python flops_accounting.py                    # all three canonical arms
    python flops_accounting.py --run <run_dir>    # one specific run
    python flops_accounting.py --json <path>      # also write a machine artifact
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from torch.utils.flop_counter import FlopCounterMode

sys.path.insert(0, str(Path(__file__).resolve().parent))

from more.families import NUM_EXPERTS_CANONICAL                       # noqa: E402
from more.model import (MoREModel, CANONICAL_ROUTING_MODE,            # noqa: E402
                        CANONICAL_ROUTER_NOISE,
                        ROUTER_NOISE_INIT_SCALE_DEFAULT,
                        ROUTER_NOISE_ANNEAL_STEPS_DEFAULT,
                        ROUTING_PERSISTENCE_LEGACY)

CODE_DIR = Path(__file__).resolve().parent
RUNS = CODE_DIR.parent / "runs"
TASK_LANGUAGE = "language"
LINEARITY_TOL = 1e-3          # 0.1% -- see METHOD above


def _out(s: str) -> None:
    sys.stdout.buffer.write(s.encode("utf-8"))
    sys.stdout.buffer.write(b"\n")
    sys.stdout.flush()


def build(mc: dict, dc: dict, max_depth: int, fixed: bool) -> MoREModel:
    """engine.py:265 with max_depth/fixed_depth overridden for the depth sweep."""
    return MoREModel(
        step_feat_dim=int(dc.get("step_feat_dim", 8)),
        d_model=mc["d_model"],
        num_experts=mc["num_experts"],
        max_depth=max_depth,
        num_blocks=mc["num_blocks"],
        dropout=0.0,
        fixed_depth=fixed,
        routing_mode=mc.get("routing_mode", CANONICAL_ROUTING_MODE),
        routing_persistence=mc.get("routing_persistence",
                                   ROUTING_PERSISTENCE_LEGACY),
        router_noise=mc.get("router_noise", CANONICAL_ROUTER_NOISE),
        router_noise_init=mc.get("router_noise_init",
                                 ROUTER_NOISE_INIT_SCALE_DEFAULT),
        router_noise_anneal_steps=mc.get("router_noise_anneal_steps",
                                         ROUTER_NOISE_ANNEAL_STEPS_DEFAULT),
        ffn_mult=mc.get("ffn_mult", 4),
        num_families=NUM_EXPERTS_CANONICAL,
        attention=bool(mc.get("attention", False)),
        n_heads=int(mc.get("n_heads", 4)),
        max_seq_len=int(dc["seq_len"]),
        task=TASK_LANGUAGE,
        vocab_size=int(dc["vocab_size"]),
    ).eval()


def count(model: MoREModel, batch: int, seq: int, vocab: int) -> int:
    """
    FLOPs for ONE forward of [batch, seq]. `display=False` because the per-module
    table is 200 lines and the caller wants one number.

    torch.no_grad(): this is the inference cost. A training step is conventionally
    ~3x (forward + backward), and that factor is applied in the report rather than
    measured, because backward FLOPs depend on what is checkpointed.
    """
    x = torch.randint(0, vocab, (batch, seq), dtype=torch.long)
    mask = torch.ones(batch, seq, dtype=torch.bool)
    ops = torch.zeros(batch, seq, dtype=torch.long)
    fc = FlopCounterMode(display=False, depth=None)
    with fc, torch.no_grad():
        model(x, mask, None, ops)
    return fc.get_total_flops()


def profile_arm(name: str, rc: dict, batch: int, mean_depth: float) -> dict:
    mc, dc = rc["model"], rc["data"]
    seq, vocab = int(dc["seq_len"]), int(dc["vocab_size"])
    md = int(mc["max_depth"])

    per_tok = {}
    # k=1 always; k=max_depth when the arm actually recurses. MoE has max_depth=1
    # and therefore no marginal step to measure -- reported as N/A, never as 0.
    ks = [1] if md == 1 else [1, md]
    for k in ks:
        per_tok[k] = count(build(mc, dc, k, fixed=True), batch, seq, vocab)

    base = per_tok[1]
    if md == 1:
        per_step = None
        at_mean = base
        resid = None
    else:
        per_step = (per_tok[md] - base) / (md - 1)
        # Verify linearity against a third point instead of trusting the fit.
        probe_k = max(2, md // 2)
        probe = count(build(mc, dc, probe_k, fixed=True), batch, seq, vocab)
        pred = base + (probe_k - 1) * per_step
        resid = abs(probe - pred) / probe
        if resid > LINEARITY_TOL:
            raise RuntimeError(
                f"{name}: FLOPs are not linear in depth (k={probe_k}: measured "
                f"{probe}, predicted {pred:.0f}, residual {resid:.2%}). The "
                f"two-point fit is invalid; do not report a composed number."
            )
        at_mean = base + (mean_depth - 1) * per_step

    tokens = batch * seq
    return {
        "arm": name,
        "num_experts": mc["num_experts"],
        "ffn_mult": mc.get("ffn_mult", 4),
        "max_depth": md,
        "mean_depth_measured": mean_depth,
        "flops_depth1": base,
        "flops_per_extra_step": per_step,
        "flops_at_mean_depth": at_mean,
        "flops_per_token_at_mean_depth": at_mean / tokens,
        "linearity_residual": resid,
        "batch": batch, "seq_len": seq, "vocab_size": vocab,
    }


def main() -> int:
    argv = sys.argv[1:]
    batch = 8
    if "--batch" in argv:
        batch = int(argv[argv.index("--batch") + 1])

    # T-LA.22. THE MEASURED DEPTH IS THE MEAN OVER THE ADMITTED RUNS.
    #
    # This used to take ONE run per arm -- the first directory, in sort order,
    # whose resolved_config claims canonical_lang_b -- and evaluate FLOPs at that
    # single run's depth/mean. Two defects compounded. (a) Claiming the group is
    # not being admitted: the exporter decides admission, and the MoR run picked
    # (langB_MoR_seed42__547e435b__r4) was later superseded and no longer exists.
    # (b) One seed's depth is not the arm's depth; MoR's depth/mean spreads
    # 6.37..7.0 across seeds. MoR was costed at 6.899 where the admitted 5-seed
    # mean is 6.730, overstating its multiplier 9.07x -> 8.87x -- on the axis the
    # paper's efficiency claim rests on.
    #
    # FLOPs are linear in depth (verified below to 0.1%), so the mean of the
    # per-seed FLOPs equals the FLOPs at the mean depth; the config is identical
    # across an arm's seeds, so any admitted run supplies it. The admitted list
    # comes from results/language/results.json, the exporter's own output.
    admitted = None
    res_p = CODE_DIR.parent / "results" / "language" / "results.json"
    if res_p.exists():
        res = json.loads(res_p.read_text(encoding="utf-8"))
        admitted = {r["run_dir"] for r in res.get("admitted_runs", [])}
    depths = {}
    picks = {}
    for d in sorted(RUNS.glob("langB_*")):
        rc_p, m_p = d / "resolved_config.json", d / "metrics.json"
        if not (rc_p.exists() and m_p.exists()):
            continue
        rc = json.loads(rc_p.read_text(encoding="utf-8"))
        prov = rc.get("provenance", {})
        if (prov.get("experiment_group") or rc.get("experiment_group")) != "canonical_lang_b":
            continue
        if admitted is not None and d.name not in admitted:
            continue
        arm = rc.get("architecture") or rc["model"].get("architecture")
        m = json.loads(m_p.read_text(encoding="utf-8"))
        depths.setdefault(arm, []).append(float(m.get("depth/mean") or 1.0))
        if arm in picks:
            continue
        # Two throughput keys exist across the matrix: the language runs log
        # `perf/throughput_items_sec`, older cells `perf/throughput_tokens_sec`.
        # Falling back rather than reporting None keeps the comparison column
        # populated; a missing value still prints N/A rather than 0.
        _thr = (m.get("perf/throughput_items_sec")
                or m.get("perf/throughput_tokens_sec"))
        picks[arm] = (d.name, rc, m.get("depth/mean") or 1.0, _thr)

    if not picks:
        _out("[FAIL] no canonical_lang_b runs found under runs/")
        return 1
    # Replace the single-run depth with the admitted-run mean.
    for arm, (run_name, rc, _single, thr) in list(picks.items()):
        ds = depths[arm]
        picks[arm] = (f"{run_name} (config); depth = mean of {len(ds)} admitted runs",
                      rc, sum(ds) / len(ds), thr)

    rows = []
    for arm in ("moe", "mor", "more"):
        if arm not in picks:
            continue
        run_name, rc, mean_depth, thr = picks[arm]
        r = profile_arm(arm, rc, batch, mean_depth)
        r["source_run"] = run_name
        # T-LA.22: NOT a cross-arm comparison. The admitted runs trained on two
        # GPUs (MoE/MoR on an RTX 4060, MoRE on a V100), so this logged TRAINING
        # throughput compares cards as much as architectures. The same-card
        # number is eval_ood.py's blocks_per_sec (every checkpoint, one GPU).
        r["training_throughput_items_sec_UNCOMPARABLE"] = thr
        rows.append(r)

    _out("")
    _out(f"Measured forward FLOPs -- batch={batch}, seq={rows[0]['seq_len']}, "
         f"vocab={rows[0]['vocab_size']}  (FlopCounterMode, torch {torch.__version__})")
    _out("")
    hdr = (f"  {'arm':<6} {'E':>2} {'ffn':>4} {'d_mean':>7} "
           f"{'GFLOPs @1':>10} {'GF/step':>9} {'GFLOPs @d_mean':>15} {'MFLOPs/token':>13}")
    _out(hdr)
    for r in rows:
        ps = (f"{r['flops_per_extra_step']/1e9:9.3f}"
              if r["flops_per_extra_step"] is not None else f"{'N/A':>9}")
        _out(f"  {r['arm']:<6} {r['num_experts']:>2} {r['ffn_mult']:>4} "
             f"{r['mean_depth_measured']:>7.3f} {r['flops_depth1']/1e9:>10.3f} {ps} "
             f"{r['flops_at_mean_depth']/1e9:>15.3f} "
             f"{r['flops_per_token_at_mean_depth']/1e6:>13.3f}")
    _out("")

    by = {r["arm"]: r["flops_at_mean_depth"] for r in rows}
    if {"mor", "more"} <= by.keys():
        _out(f"  MoR / MoRE  FLOPs ratio = {by['mor']/by['more']:.2f}x")
    if {"moe", "more"} <= by.keys():
        _out(f"  MoRE / MoE  FLOPs ratio = {by['more']/by['moe']:.2f}x")
    if {"mor", "moe"} <= by.keys():
        _out(f"  MoR  / MoE  FLOPs ratio = {by['mor']/by['moe']:.2f}x")
    _out("")
    _out("  FLOPs are analytic inference cost. They are NOT a speedup. Same-card")
    _out("  throughput for every admitted checkpoint is eval_ood.py's blocks_per_sec;")
    _out("  the admitted runs trained on two different GPUs, so their logged")
    _out("  training throughputs are not comparable across arms.")
    _out("")

    if "--json" in argv:
        p = Path(argv[argv.index("--json") + 1])
        if not p.is_absolute():
            p = (CODE_DIR.parent / p).resolve()
        p.write_text(json.dumps(
            {"torch": torch.__version__, "batch": batch, "arms": rows},
            indent=2), encoding="utf-8")
        _out(f"  wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
