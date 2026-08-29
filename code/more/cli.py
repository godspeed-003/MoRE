"""
more/cli.py -- the ONE command-line entry point for every architecture.

plan.md 9 / updated_rules.md 6: there is a single training system with an
explicit `architecture = moe | mor | more` mode. `train_moe.py`,
`train_mor.py` and `train_more.py` are three-line launchers that call
`main(default_architecture=...)` here; they contain no model, data or training
code of their own, so the three baselines are guaranteed to share one
tokenizer, one optimizer setup and one evaluation path.
"""

from __future__ import annotations

import argparse
import sys

from .config import (load_config, apply_architecture, enforce_routing_mode,
                     stamp_seed_into_run_name, ARCHITECTURES)
from .engine import train
from .run_context import RunContext, resolve_overrides, ProxyGuardError


def build_parser(default_architecture: str | None = None) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train MoE / MoR / MoRE from one shared pipeline"
    )
    p.add_argument(
        "--architecture", choices=ARCHITECTURES, default=default_architecture,
        required=default_architecture is None,
        help="moe = experts, depth 1 | mor = 1 expert, adaptive depth | "
             "more = experts + adaptive depth",
    )
    p.add_argument("--config", default="config.json",
                   help="Path to config.json (default: config.json)")
    p.add_argument("--epochs", type=int, default=None,
                   help="Override epochs from config (diagnostic runs)")
    p.add_argument("--blocks", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument(
        "--halting_supervision", dest="halting_supervision",
        action="store_true", default=None,
        help="Enable the operation-complexity curriculum as a supervised "
             "halting target (T3.3, updated_rules.md 2.3). Default OFF = pure "
             "unsupervised ACT. Ignored for --architecture moe, which runs at "
             "max_depth 1 and has no depth to allocate. The mode is written to "
             "provenance as halting_mode, so a supervised run is never "
             "mistakable for an unsupervised one.",
    )
    p.add_argument(
        "--halting_supervision_weight", type=float, default=None,
        help="Loss weight for the curriculum term. Kept separate from the "
             "ponder-cost weight on purpose: the two pull in opposite "
             "directions (plan.md 5.4). Required non-zero when "
             "--halting_supervision is set.",
    )
    p.add_argument(
        "--routing_balance", type=float, default=None,
        help="Override loss_weights.routing_balance (T4.2 / plan.md 6.2). The "
             "coefficient must be chosen from MEASURED magnitudes -- balance "
             "loss vs task loss after warmup -- not inherited. Provided so the "
             "selection can be done with runs instead of by editing config.json "
             "between them; the chosen value is then frozen in config.json. "
             "Ignored for --architecture mor, which has one expert and nothing "
             "to balance.",
    )
    p.add_argument(
        "--step_routing", type=float, default=None,
        help="Override loss_weights.step_routing -- the ROUTING-SUPERVISION "
             "weight. engine.py assembles this term as 0.01*oracle_routing_ce + "
             "0.3*step_cls_loss, so any non-zero value trains the router against "
             "the ORACLE EXPERT INDEX. plan.md E: supervision OFF is canonical, "
             "supervision ON is an explicitly labelled oracle-routing ablation "
             "(T10.E), and the two may not share a headline table. Provided so "
             "that choice can be settled with matched runs instead of by editing "
             "config.json between them. The value reaches provenance as "
             "routing_supervision_enabled / routing_supervision_weight, so a "
             "supervised run is never mistakable for an unsupervised one. "
             "Ignored for --architecture mor, which has one expert and therefore "
             "no routing target.",
    )
    p.add_argument(
        "--family_cls", type=float, default=None,
        help="Override loss_weights.family_cls -- the weight on the 6-way "
             "whole-program FAMILY cross-entropy (cls_head). This was a bare "
             "0.5 literal inside engine.py's total_loss until T8.3, i.e. the "
             "second-largest term in the objective appeared in no config, no "
             "provenance block and no results table. It is EXTERNAL "
             "annotation, not self-supervision: data/script.py stamps a "
             "\"family\" string into every record and more/data.py reads it off "
             "disk. Canonical is 0.5; 0.0 is the T10.H "
             "no-family-supervision ablation, labelled as such by "
             "resolve_variant.",
    )
    p.add_argument(
        "--ffn_mult", type=int, default=None,
        help="Override model.ffn_mult (FFN hidden width = ffn_mult * d_model). "
             "apply_architecture stamps the canonical, budget-matched value per "
             "architecture (MoE/MoRE 4, MoR 24 = 4x6, giving MoR's single FFN "
             "the total hidden width of MoRE's six experts, T9.2 gap 0.12%%). An "
             "explicit value overrides that and is labelled ffn_mult_<n>, so the "
             "under-budgeted MoR-small baseline (ffn_mult 4, 82.2%% short) can "
             "never be tabled as canonical.",
    )
    p.add_argument(
        "--experiment_group", type=str, default=None,
        help="Run group. Use 'canonical_phase_b' ONLY for the frozen canonical "
             "matrix; the Gate 0 guard in run_context.py refuses the run if it "
             "does not match code/canonical_spec.json exactly. Any other value "
             "(default: 'exploratory') marks the run non-canonical so the "
             "exporter excludes it.",
    )
    return p


def main(argv=None, default_architecture: str | None = None) -> int:
    args = build_parser(default_architecture).parse_args(argv)

    ctx = None
    try:
        raw_cfg = load_config(args.config)

        # Architecture is stamped BEFORE resolve_overrides so the resolved
        # config -- the single source of truth that trains, is hashed, and is
        # logged -- already carries num_experts / max_depth / halting.
        apply_architecture(raw_cfg, args.architecture)

        # T3.3 halting-supervision override. Applied AFTER apply_architecture so
        # the architecture's own decision still wins where it is a hard
        # constraint: MoE runs at max_depth 1, where a curriculum target of 2-4
        # is unreachable by construction, so supervising it would inject a
        # permanent irreducible loss. Refuse rather than silently ignore.
        if args.halting_supervision:
            if raw_cfg["model"]["max_depth"] <= 1:
                raise ValueError(
                    "--halting_supervision requires max_depth > 1. "
                    f"architecture={args.architecture!r} runs at max_depth="
                    f"{raw_cfg['model']['max_depth']}, so every token exits at "
                    "step 1 and curriculum targets of 2-4 can never be reached. "
                    "The term would contribute a constant, irreducible loss and "
                    "a gradient that cannot be satisfied."
                )
            raw_cfg["model"]["halting_supervision"] = True
            raw_cfg["loss_weights"]["halting_supervision"] = (
                args.halting_supervision_weight
                if args.halting_supervision_weight is not None else 0.1
            )
        elif args.halting_supervision_weight is not None:
            raise ValueError(
                "--halting_supervision_weight was given without "
                "--halting_supervision, so the curriculum term would be "
                "weighted but never enabled. Pass both, or neither."
            )

        # T4.2 routing-balance override. Also applied AFTER apply_architecture,
        # for the same reason: `mor` sets routing_balance = 0.0 because a single
        # expert has nothing to balance, and a CLI value must not resurrect a
        # term that is definitionally absent for that architecture. Refuse
        # rather than silently ignore, so a sweep cannot record a weight the run
        # did not actually use.
        if args.routing_balance is not None:
            if raw_cfg["model"]["num_experts"] <= 1:
                raise ValueError(
                    "--routing_balance requires num_experts > 1. "
                    f"architecture={args.architecture!r} runs with "
                    f"num_experts={raw_cfg['model']['num_experts']}, so the "
                    "balance objective is a constant (entropy of a one-element "
                    "distribution is 0, the Switch term is exactly 1) with no "
                    "gradient. Weighting it would record a coefficient that "
                    "changes nothing."
                )
            if args.routing_balance < 0.0:
                raise ValueError(
                    "--routing_balance must be >= 0. A negative coefficient "
                    "flips the sign of the objective, which would MINIMISE "
                    "routing entropy and actively drive the expert collapse the "
                    "term exists to prevent (CLAUDE.md 2)."
                )
            raw_cfg["loss_weights"]["routing_balance"] = args.routing_balance

        # T8.0b / T10.E routing-supervision override. Same placement and the
        # same refuse-don't-ignore discipline as the two above. `mor` sets
        # step_routing = 0.0 in apply_architecture because a single expert has no
        # routing target at all; a CLI value must not resurrect an oracle CE
        # against a label space of size one.
        if args.step_routing is not None:
            if args.step_routing < 0.0:
                raise ValueError(
                    "--step_routing must be >= 0. A negative coefficient would "
                    "MAXIMISE the oracle routing cross-entropy, i.e. train the "
                    "router to avoid the correct expert."
                )
            if raw_cfg["model"]["num_experts"] <= 1 and args.step_routing != 0.0:
                raise ValueError(
                    "--step_routing requires num_experts > 1. architecture="
                    f"{args.architecture!r} runs with num_experts="
                    f"{raw_cfg['model']['num_experts']}, so there is no routing "
                    "decision to supervise and the oracle CE is a constant. "
                    "Weighting it would record a supervision weight the run did "
                    "not actually use."
                )
            raw_cfg["loss_weights"]["step_routing"] = args.step_routing

        # T8.3 / T10.H family-supervision override. Same placement and the same
        # refuse-don't-ignore discipline. This is the term that was a bare 0.5
        # literal in engine.py, so before this flag existed the only way to ablate
        # it was to edit source -- which produces an unlabelled variant.
        if args.family_cls is not None:
            if args.family_cls < 0.0:
                raise ValueError(
                    "--family_cls must be >= 0. A negative coefficient would "
                    "MAXIMISE the family cross-entropy, i.e. train the pooled "
                    "representation to be family-INdistinguishable."
                )
            raw_cfg["loss_weights"]["family_cls"] = args.family_cls

        # T8.3 / T9.2 parameter-budget override. apply_architecture has just
        # stamped the canonical value for this architecture (MoE/MoRE 4, MoR 24);
        # an explicit value overrides it and is labelled `ffn_mult_<n>` by
        # resolve_variant, so the under-budgeted MoR small baseline can never be
        # tabled as canonical.
        if args.ffn_mult is not None:
            if args.ffn_mult < 1:
                raise ValueError(
                    f"--ffn_mult must be >= 1, got {args.ffn_mult}. The FFN "
                    "hidden width is ffn_mult * d_model."
                )
            raw_cfg["model"]["ffn_mult"] = args.ffn_mult

        resolved = resolve_overrides(
            raw_cfg,
            epochs=args.epochs,
            blocks=args.blocks,
            batch_size=args.batch_size,
            seed=args.seed,
            run_name=args.run_name,
            experiment_group=args.experiment_group,
        )
        # T2.4: gate the dense-routing ablation on the FINAL run label, after
        # --run_name / --experiment_group have been applied. Checking earlier
        # would test the default label instead of the real one.
        enforce_routing_mode(resolved)

        # T6.6: the seed belongs in the run label (updated_rules.md 9:
        # phaseB_MoRE_seed42). Done here, after resolve_overrides, because only
        # now are both the seed and the run name final. It runs BEFORE
        # RunContext.create so the label reaches the run directory, the W&B run
        # name and provenance as one consistent string.
        stamp_seed_into_run_name(resolved)

        ctx = RunContext.create(raw_cfg, resolved)
        print(f"[Run] architecture   = {resolved.get('architecture')}")
        train(resolved, ctx=ctx)
        return 0

    except ProxyGuardError as e:
        print("[REFUSED] Gate 0 (proxy guard) refused this run.", file=sys.stderr)
        print(str(e), file=sys.stderr)
        return 2
    finally:
        if ctx is not None:
            ctx.close()
