"""
T-LX.14 -- attribute the gradient arriving at the ACT halt heads to each term of
the objective, separately, on a TRAINED checkpoint.

WHY THIS EXISTS
---------------
A NeurIPS-workshop reviewer (652J, concern 1) read the paper's ACT section and
concluded:

    "The paper never defines the probability-weighted output aggregation ... If
     the task loss sees only a discrete halted state, the halt head learns mainly
     to stop early, and a near-constant policy is expected."

The mechanism half of that is wrong: `model.py` DOES accumulate
`accumulator += state_t * w_t` with per-token weights summing to exactly 1
(model.py:995, "5. Accumulate the ACT-weighted output"), so the task loss sees a
convex combination of every step's state and a real gradient path to the halt
heads exists. `test_phase3_halting.py` already asserts that the path is non-zero.

The PREDICTION half is empirically correct, and that is the uncomfortable part:
depth still collapses to the cap (30-epoch dev trend 1.96 -> 6.82, 88.9% of
tokens exiting at step 7; the five canonical MoRE cells sit at
val/forced_exit_rate 0.980-0.987). So "the gradient is missing" is refuted and
"the policy is near-constant" is confirmed at the same time, which means the
interesting question is not EXISTENCE but DOMINANCE:

    of the gradient that reaches the halt heads, what fraction comes from the
    task loss (which can reward more depth) versus the ponder cost (which only
    ever pushes depth down)?

A non-zero but heavily dominated task gradient explains a near-constant policy
without any bug, and is the honest answer to the reviewer. That ratio is what
this script measures. It is a DIAGNOSTIC: it reports magnitudes, it does not
change any objective, and nothing here is tuned toward a nicer number
(CLAUDE.md 6).

WHAT IT REPORTS
---------------
Per objective term, the L2 norm of d(term)/d(halt-head params), where the
halt-head parameter set is exactly `blocks[*].expert_halt_heads.*` -- the
parameters updated_rules.md 2.2 requires a real gradient to reach. Terms are
taken one at a time with `retain_graph=True` and a zeroed grad buffer between
them, so the numbers are per-term and not cumulative.

The terms mirror engine.py:830-838 exactly, including the weights actually
applied, because an unweighted comparison is not the comparison that trains:
    lw["task"] * (task_loss + lw["family_cls"] * cls_loss)
    lw["step_routing"]        * step_routing_loss
    lw["routing_balance"]     * bal_loss
    current_halt_weight       * ponder_cost
    lw["halting_supervision"] * halt_sup_loss

`current_halt_weight` is the WARMED-UP target weight, not the step-1 value: this
runs on a converged checkpoint, where warmup is long over (engine.py:801).

Terms that are structural zeros on language are reported as such rather than as
0.0000 measurements (CLAUDE.md 4: no sentinel may be reported as a measurement) --
`cls_loss` has no head on a packed LM block (engine.py:740) and
`halting_supervision` is 0.0 in the language config.

USAGE
    python diag_halt_gradient.py <run_dir> [--batches N] [--device cpu|cuda]

Writes `halt_gradient_attribution.json` into the run directory as a sidecar and
prints a table. Read-only with respect to every canonical artifact: it never
touches metrics.json, results.tsv or the checkpoint.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

from more.families import NUM_EXPERTS_CANONICAL                       # noqa: E402
from more.lang_data import MoRELanguageDataset                        # noqa: E402
from more.model import (MoREModel, CANONICAL_ROUTING_MODE,            # noqa: E402
                        CANONICAL_ROUTER_NOISE,
                        ROUTER_NOISE_INIT_SCALE_DEFAULT,
                        ROUTER_NOISE_ANNEAL_STEPS_DEFAULT,
                        ROUTING_PERSISTENCE_LEGACY)

CODE_DIR = Path(__file__).resolve().parent
SIDECAR = "halt_gradient_attribution.json"
TASK_LANGUAGE = "language"


def _out(s: str) -> None:
    """cp1252-safe stdout -- this file prints arrows and en-dashes."""
    sys.stdout.buffer.write(s.encode("utf-8"))
    sys.stdout.buffer.write(b"\n")
    sys.stdout.flush()


def build_model(mc: dict, dc: dict, task: str, step_feat_dim: int,
                device: torch.device) -> MoREModel:
    """
    Mirror of engine.py:265. Every default here is the engine's default, including
    the T-LX.12 legacy `routing_persistence` fallback -- a checkpoint written
    before that axis existed must rebuild as the per-step architecture it was, or
    the state_dict load would succeed while the FORWARD differed.
    """
    return MoREModel(
        step_feat_dim=step_feat_dim,
        d_model=mc["d_model"],
        num_experts=mc["num_experts"],
        max_depth=mc["max_depth"],
        num_blocks=mc["num_blocks"],
        dropout=mc["dropout"],
        fixed_depth=mc.get("fixed_depth", False),
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
        max_seq_len=(int(dc["seq_len"]) if task == TASK_LANGUAGE else None),
        task=task,
        vocab_size=(int(dc["vocab_size"]) if task == TASK_LANGUAGE else None),
    ).to(device)


def halt_params(model: MoREModel) -> list[tuple[str, torch.nn.Parameter]]:
    """
    The exact parameter set updated_rules.md 2.2 is about. Selected by name so a
    future rename shows up as an empty list (and an explicit raise) rather than
    as a silent 0.0 that would read as "no gradient reaches the halt head".
    """
    named = [(n, p) for n, p in model.named_parameters()
             if "expert_halt_heads" in n and p.requires_grad]
    if not named:
        raise RuntimeError(
            "no parameters matched 'expert_halt_heads' -- the halt head was "
            "renamed or is not trainable. Refusing to report 0.0, which would "
            "be indistinguishable from a real absent gradient."
        )
    return named


def grad_norm(term: torch.Tensor, params: list[torch.nn.Parameter],
              model: MoREModel) -> float:
    """
    L2 norm of d(term)/d(halt params), with the grad buffer zeroed first so the
    result is this term alone and not the running sum of every term before it.
    `retain_graph=True` because the caller walks several terms over one forward.
    """
    model.zero_grad(set_to_none=True)
    if not term.requires_grad:
        # A structural zero (no head constructed, or a detached constant). Report
        # it as such upstream; returning 0.0 here would be a sentinel.
        return float("nan")
    term.backward(retain_graph=True)
    sq = 0.0
    for p in params:
        if p.grad is not None:
            sq += float(p.grad.detach().pow(2).sum())
    return sq ** 0.5


def main() -> int:
    argv = sys.argv[1:]
    if not argv:
        _out(__doc__ or "")
        return 2
    run_dir = Path(argv[0])
    if not run_dir.is_absolute():
        run_dir = (CODE_DIR.parent / run_dir).resolve()

    n_batches = 4
    if "--batches" in argv:
        n_batches = int(argv[argv.index("--batches") + 1])
    device = torch.device(
        argv[argv.index("--device") + 1] if "--device" in argv
        else ("cuda" if torch.cuda.is_available() else "cpu"))

    rc_path = run_dir / "resolved_config.json"
    ck_path = run_dir / "checkpoint.pt"
    for p in (rc_path, ck_path):
        if not p.exists():
            _out(f"[FAIL] missing {p.name} in {run_dir}")
            return 1

    rc = json.loads(rc_path.read_text(encoding="utf-8"))
    mc, dc, lw = rc.get("model", {}), rc.get("data", {}), rc.get("loss_weights", {})
    task = rc.get("task") or rc.get("provenance", {}).get("task") or TASK_LANGUAGE
    if task != TASK_LANGUAGE:
        _out(f"[FAIL] this diagnostic is language-only, run declares task={task!r}")
        return 1

    corpus = dc.get("corpus") or dc.get("dataset_version")
    ds = MoRELanguageDataset(corpus, "val")
    bs = int(rc.get("training", {}).get("batch_size", 8))
    loader = DataLoader(ds, batch_size=bs, shuffle=False)

    step_feat_dim = int(dc.get("step_feat_dim", 8))
    model = build_model(mc, dc, task, step_feat_dim, device)
    state = torch.load(ck_path, map_location=device)
    model.load_state_dict(state)
    # EVAL mode: dropout off. The question is what the converged policy's gradient
    # looks like, not what one stochastic training step happened to see.
    model.eval()

    hp = halt_params(model)
    params = [p for _, p in hp]
    halt_w = float(rc.get("training", {}).get("halt_weight",
                   lw.get("halting", lw.get("ponder", 0.0))))

    acc: dict[str, list[float]] = {}
    depth_seen: list[float] = []
    for bi, batch in enumerate(loader):
        if bi >= n_batches:
            break
        x, step_mask, step_experts, step_ops, family, depth, target = \
            [b.to(device) if torch.is_tensor(b) else b for b in batch]

        (reg_out, cls_out, step_cls_out, bal_loss, ponder_cost,
         depth_exits, avg_depth, batch_expert_idx, oracle_routing_ce,
         _first_route, route_stats, expected_depth, halt_stats) = \
            model(x, step_mask, step_experts, step_ops)

        V = reg_out.shape[-1]
        task_loss = F.cross_entropy(
            reg_out[:, :-1, :].reshape(-1, V), x[:, 1:].reshape(-1))

        terms: dict[str, torch.Tensor | None] = {
            "task": lw.get("task", 1.0) * task_loss,
            "ponder": halt_w * ponder_cost,
            "routing_balance": (lw.get("routing_balance", 0.0) * bal_loss
                                if bal_loss is not None else None),
            "step_routing": (lw.get("step_routing", 0.0) * oracle_routing_ce
                             if torch.is_tensor(oracle_routing_ce) else None),
        }
        for name, t in terms.items():
            if t is None or not torch.is_tensor(t):
                acc.setdefault(name, []).append(float("nan"))
                continue
            acc.setdefault(name, []).append(grad_norm(t, params, model))
        depth_seen.append(float(avg_depth) if avg_depth is not None else float("nan"))

    def summarize(v: list[float]) -> tuple[float | None, float | None]:
        ok = [z for z in v if z == z]                       # drop NaN
        if not ok:
            return None, None
        m = sum(ok) / len(ok)
        sd = (sum((z - m) ** 2 for z in ok) / len(ok)) ** 0.5 if len(ok) > 1 else 0.0
        return m, sd

    means = {k: summarize(v) for k, v in acc.items()}
    t_mean = (means.get("task") or (None, None))[0]
    p_mean = (means.get("ponder") or (None, None))[0]
    ratio = (t_mean / p_mean) if (t_mean and p_mean) else None

    _out("")
    _out(f"Halt-head gradient attribution -- {run_dir.name}")
    _out(f"  corpus={corpus}  batches={n_batches}  bs={bs}  device={device.type}")
    _out(f"  routing_persistence={mc.get('routing_persistence', ROUTING_PERSISTENCE_LEGACY)}"
         f"  max_depth={mc.get('max_depth')}  ponder_weight={halt_w}")
    _out(f"  halt-head tensors={len(hp)}  params={sum(p.numel() for p in params)}")
    _out(f"  mean avg_depth over these batches={sum(depth_seen)/len(depth_seen):.4f}")
    _out("")
    _out(f"  {'objective term':<20} {'||grad|| at halt heads':>24}   note")
    for name in ("task", "ponder", "routing_balance", "step_routing"):
        m, sd = means.get(name, (None, None))
        if m is None:
            note = ("structural zero on language (no head / weight 0.0)"
                    if name in ("routing_balance", "step_routing")
                    else "no differentiable path")
            _out(f"  {name:<20} {'N/A':>24}   {note}")
        else:
            # An EXACT 0.0 here is a measurement, not a sentinel, and the
            # distinction matters (CLAUDE.md 4). The balance and step-routing
            # terms are functions of `router_logits` alone; no edge of the graph
            # runs from them to `expert_halt_heads`, so their derivative w.r.t.
            # the halt parameters is structurally zero rather than numerically
            # small. Labelled so a reader cannot mistake it for a dead gradient
            # of the kind updated_rules.md 2.2 forbids.
            note = ("exact structural 0 -- no graph path to the halt heads"
                    if m == 0.0 else "")
            _out(f"  {name:<20} {m:>16.6e} +- {sd:.1e}   {note}")
    _out("")
    if ratio is not None:
        _out(f"  task/ponder gradient ratio = {ratio:.4e}"
             f"   ({'task-dominated' if ratio > 1 else 'PONDER-DOMINATED'})")
    _out("")

    payload = {
        "run": run_dir.name,
        "corpus": corpus,
        "batches": n_batches,
        "batch_size": bs,
        "device": device.type,
        "routing_persistence": mc.get("routing_persistence",
                                      ROUTING_PERSISTENCE_LEGACY),
        "ponder_weight": halt_w,
        "halt_head_tensors": len(hp),
        "halt_head_params": sum(p.numel() for p in params),
        "avg_depth_mean": sum(depth_seen) / len(depth_seen),
        "grad_norm": {k: (v[0] if v[0] is not None else "N/A")
                      for k, v in means.items()},
        "grad_norm_std": {k: (v[1] if v[1] is not None else "N/A")
                          for k, v in means.items()},
        "task_over_ponder_ratio": ratio if ratio is not None else "N/A",
    }
    (run_dir / SIDECAR).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _out(f"  wrote {run_dir / SIDECAR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
