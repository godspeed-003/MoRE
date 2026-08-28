"""config.py - Config loading and precedence.

Extracted from the former monolithic train.py (plan.md 9: ONE training
system, split into readable modules; not parallel implementations).
"""

import os
import json

# ---------------------------------------------------------------------------
# 0. Configuration loader
# ---------------------------------------------------------------------------

from .model import (CANONICAL_ROUTING_MODE, DENSE_ABLATION_ROUTING_MODE,
                    ROUTING_MODES, CANONICAL_ROUTER_NOISE, ROUTER_NOISE_MODES,
                    ROUTER_NOISE_INIT_SCALE_DEFAULT,
                    ROUTER_NOISE_ANNEAL_STEPS_DEFAULT)
from .families import NUM_EXPERTS_CANONICAL

# The canonical expert count, E1..E6; the E7 catch-all is removed
# (updated_rules.md 2, plan.md 7.2). Defined HERE, above the defaults that use
# it, and taken from the one family manifest rather than restated as a literal
# -- T5.1: a second copy of the number is how `num_experts` and the family
# label list drift apart in the first place.
CANONICAL_NUM_EXPERTS = NUM_EXPERTS_CANONICAL


def load_config_defaults(cfg: dict | None = None) -> dict:
    """
    Fill in every default key, in place, and return cfg.

    Split out of load_config so the defaults can be exercised without a file on
    disk. This matters for the gates: a test that hand-builds a dict would be
    asserting against its own copy of the defaults rather than against the ones
    the training run actually uses, and the two could drift apart silently.
    """
    cfg = {} if cfg is None else cfg

    # T5.3 (plan.md 7.3): snapshot what the CALLER actually wrote for the
    # duplicated subset keys, BEFORE any setdefault below runs. Once the
    # defaults are in, `"subset_fraction" in cfg["training"]` is true
    # unconditionally and "the user asked for this" is indistinguishable from
    # "we filled it in" -- which is precisely how the removed
    # backward-compatibility shim came to overwrite a configured
    # `data.subset_fraction` with the default 1.0 on every single load.
    _user_subset = {
        key: {
            section: cfg[section][key]
            for section in ("training", "data")
            if isinstance(cfg.get(section), dict) and key in cfg[section]
        }
        for key in ("subset_fraction", "subset_seed")
    }

    # Provide safe defaults for any key the agent may have deleted
    cfg.setdefault("model", {})
    cfg["model"].setdefault("d_model", 256)
    # T5.1 (plan.md 7.1): the canonical expert count, from the one manifest.
    # This default was the literal 7 while CANONICAL_NUM_EXPERTS was 6, so any
    # config that omitted the key -- or any caller of load_config_defaults that
    # never went through apply_architecture, which includes several tests --
    # silently built a 7-expert model with a permanently unreachable 7th class.
    cfg["model"].setdefault("num_experts", CANONICAL_NUM_EXPERTS)
    cfg["model"].setdefault("routing_mode", CANONICAL_ROUTING_MODE)
    # T5.4 (plan.md 7.4): router exploration noise, canonical OFF. Previously
    # there was no key at all and a trainable noise scale of 0.1 was always
    # active, so no config, resolved config or W&B record mentioned that every
    # training-time routing decision was made on perturbed logits.
    cfg["model"].setdefault("router_noise", CANONICAL_ROUTER_NOISE)
    cfg["model"].setdefault("router_noise_init", ROUTER_NOISE_INIT_SCALE_DEFAULT)
    cfg["model"].setdefault("router_noise_anneal_steps",
                            ROUTER_NOISE_ANNEAL_STEPS_DEFAULT)
    # NOTE: these two 7s are NOT the expert count and must not be "fixed" to 6.
    # max_depth is the recursion budget (how many times the shared block may be
    # re-applied); max_steps is the number of step-tokens the dataset packs per
    # program. They coincide with the old expert count only by accident, and
    # that coincidence is what made the T5.1 landmines hard to spot.
    cfg["model"].setdefault("max_depth", 7)
    cfg["model"].setdefault("max_steps", 7)        # number of step-tokens per program
    cfg["model"].setdefault("step_feat_dim", 12)   # features per step-token
    # T8.2: 1, not 2. canonical_spec.json freezes num_blocks=1 (plan.md Phase 8:
    # "one recursive MoRE block initially"), and code/config.json says 1. This
    # default said 2, making load_config_defaults a THIRD source of truth that
    # disagreed with the other two. It only bit callers that build a config
    # without reading config.json -- which is every gate test and every
    # hand-built dict -- so the tests were exercising a 2-block model while the
    # canonical run is 1-block. Same reasoning for batch_size below.
    cfg["model"].setdefault("num_blocks", 1)
    # Halting supervision (T3.3). DEFAULT OFF = pure unsupervised ACT.
    # When True the halt heads additionally receive an MSE target from the
    # predefined operation-complexity curriculum (families.OP_TARGET_DEPTH).
    # updated_rules.md 2.3 permits either, but requires the choice be reported
    # explicitly -- so this flag is recorded in provenance, never inferred.
    cfg["model"].setdefault("halting_supervision", False)
    cfg["model"].setdefault("dropout", 0.1)

    cfg.setdefault("training", {})
    cfg["training"].setdefault("lr", 1e-3)
    cfg["training"].setdefault("weight_decay", 1e-4)
    cfg["training"].setdefault("batch_size", 768)   # T8.2: was 128; see num_blocks
    cfg["training"].setdefault("epochs", 50)
    cfg["training"].setdefault("val_split", 0.1)
    cfg["training"].setdefault("grad_clip", 1.0)
    cfg["training"].setdefault("subset_fraction", 1.0)
    cfg["training"].setdefault("subset_seed", 42)

    cfg.setdefault("loss_weights", {})
    cfg["loss_weights"].setdefault("task", 1.0)
    # T4.2 (plan.md 6.2): FROZEN at 0.001 from measurement, not inherited. The
    # historical 0.05 put the weighted balance term at 0.41x the task loss after
    # warmup and, projected onto the converged task loss this repo has actually
    # reached (0.001666), at ~20x it -- i.e. it re-dominates by the end of a
    # 50-epoch run, which is the defect Phase 4 exists to remove. At 0.001 the
    # ratio is 0.017 after warmup and ~0.82 at that converged task loss, and the
    # hard expert load stays spread (9.5-29.3%, no expert starved). This default
    # must agree with code/config.json; see changelog.md "Phase 4" and
    # ARCHITECTURE.md 5b for the full measurement table.
    cfg["loss_weights"].setdefault("routing_balance", 0.001)
    cfg["loss_weights"].setdefault("halting", 0.001)
    # T8.0b: 0.0, NOT 0.5. engine.py builds step_routing_loss as
    # 0.01*oracle_routing_ce + 0.3*step_cls_loss, so any non-zero weight trains
    # the router against the ORACLE EXPERT INDEX. plan.md E: supervision OFF is
    # canonical, supervision ON is an explicitly labelled oracle-routing
    # ablation (T10.E). The old 0.5 default made the canonical config an
    # ablation, and drove routing_accuracy = hungarian = AMI = purity = 1.0 with
    # the identity assignment recovered -- a measurement of the supervision
    # signal, not of learned specialization. A default that silently turns
    # supervision on is the more dangerous direction of error, so the default
    # matches the canonical value. Must agree with code/config.json.
    cfg["loss_weights"].setdefault("step_routing", 0.0)
    # Kept SEPARATE from "halting" (the ponder cost). plan.md 5.4: the two pull
    # in opposite directions -- ponder cost pushes depth down, curriculum
    # supervision pushes it toward the target -- so they must never be summed
    # into one opaque number. 0.0 by default because supervision is off.
    cfg["loss_weights"].setdefault("halting_supervision", 0.0)

    cfg.setdefault("data", {})
    # An explicitly configured single-file dataset (`jsonl_path`) must win over
    # the split defaults. Without this, setdefault below injects
    # ../data/train.jsonl and the engine's `train_path or jsonl_path`
    # precedence silently trains on a file the config never named.
    if "jsonl_path" in cfg["data"] and "train_path" not in cfg["data"]:
        cfg["data"]["train_path"] = cfg["data"]["jsonl_path"]
    cfg["data"].setdefault("train_path", "../data/train.jsonl")
    cfg["data"].setdefault("val_path", "../data/val.jsonl")
    cfg["data"].setdefault("test_path", "../data/test.jsonl")
    cfg["data"].setdefault("jsonl_path", cfg["data"]["train_path"])
    cfg["data"].setdefault("subset_fraction", 1.0)
    cfg["data"].setdefault("subset_seed", 42)
    cfg["data"].setdefault("max_val", 1e6)
    cfg["data"].setdefault("pad_value", 0.0)

    cfg.setdefault("logging", {})
    cfg["logging"].setdefault("wandb_project", "micro-MoRE-poc")
    cfg["logging"].setdefault("log_interval", 10)

    # T5.3 (plan.md 7.3): ONE source of truth for the subset keys.
    #
    # `subset_fraction` / `subset_seed` are readable in two sections for
    # historical reasons: older fast_config files wrote them under `training`,
    # while every CONSUMER reads `data` -- engine.py builds the Subset from
    # `data.subset_fraction`, run_context.py's proxy guard refuses a canonical
    # claim from it, and provenance publishes it as `resolved_subset_fraction`.
    #
    # What used to be here was `if "subset_fraction" in cfg["training"]:
    # cfg["data"][...] = cfg["training"][...]`, whose test is ALWAYS true
    # because of the setdefault above. So a config that said
    # `data.subset_fraction: 0.5` trained on the full dataset and reported 1.0,
    # with nothing in the output disagreeing with itself. The two sections are
    # now reconciled instead:
    #   * given in neither          -> the default, mirrored into both
    #   * given in exactly one      -> that value, mirrored into both
    #   * given in both and equal   -> that value (harmless duplication)
    #   * given in both and DIFFER  -> raise
    # The last case refuses rather than picking a winner: either choice trains
    # on a dataset size the config does not state, and `resolved_subset_fraction`
    # then records that size as though it had been asked for.
    for _key in ("subset_fraction", "subset_seed"):
        _given  = _user_subset[_key]
        _values = set(_given.values())
        if len(_values) > 1:
            raise ValueError(
                f"Conflicting {_key}: training.{_key}={_given['training']!r} "
                f"but data.{_key}={_given['data']!r}. These are the same "
                "quantity read from two sections; the old shim silently let "
                "`training` win, so a configured `data` value never took effect "
                "and the run reported a subset size it did not use "
                "(plan.md 7.3). Set exactly one of them."
            )
        # No user value anywhere -> fall through to the default already applied
        # by setdefault. `training` and `data` carry the same default, so which
        # one is read here does not matter.
        _resolved = _values.pop() if _values else cfg["data"][_key]
        cfg["training"][_key] = _resolved
        cfg["data"][_key]     = _resolved

    return cfg


def load_config(path: str = "config.json") -> dict:
    """Load the agent-modifiable config file, apply defaults, resolve paths."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"config.json not found at '{path}'. "
            "Create it before running train.py."
        )
    with open(path, "r") as f:
        cfg = json.load(f)

    cfg = load_config_defaults(cfg)

    # Resolve relative dataset paths against the config file directory.
    cfg_dir = os.path.dirname(os.path.abspath(path))
    for key in ("train_path", "val_path", "test_path", "jsonl_path"):
        p = cfg["data"].get(key)
        if isinstance(p, str) and p and not os.path.isabs(p):
            cfg["data"][key] = os.path.normpath(os.path.join(cfg_dir, p))

    return cfg




# ---------------------------------------------------------------------------
# Architecture selection  (plan.md 9 / updated_rules.md 6)
# ---------------------------------------------------------------------------
# ONE training system. `architecture` derives only the settings that are
# *definitionally* different between the three baselines; everything else --
# dataset, split, optimizer, lr, weight decay, batch size, epochs, seed set,
# input representation -- stays identical so the comparison is fair
# (updated_objective.md "CANONICAL HYPERPARAMETER PRINCIPLE").

ARCHITECTURES = ("moe", "mor", "more")

# T6.6 (updated_rules.md 9 / canonical_spec.json run_name_template). Run names
# must be STABLE and readable: phaseB_MoE_seed42, not phaseB_moe or
# more_run_final_v3. The display spelling is separate from the config value
# because the config value is a lowercase enum used for dispatch, while the run
# label is what appears in W&B, the run directory and every figure caption.
ARCH_DISPLAY = {"moe": "MoE", "mor": "MoR", "more": "MoRE"}
CANONICAL_RUN_PREFIX = "phaseB"

# T6.7 (updated_rules.md 9). `variant` is a required provenance field and must
# never be hand-written, because the whole point of it is to make an ablation
# impossible to mistake for the canonical setting in a results table. It is
# DERIVED from the resolved config: "canonical" only when every deviation-
# carrying field sits at its canonical value, otherwise a sorted, joined list of
# the deviations that are actually active. A new ablation must be added here or
# it will silently be reported as canonical.
CANONICAL_VARIANT = "canonical"


def resolve_variant(cfg: dict) -> str:
    """
    Derive the `variant` label from the resolved config.

    Returns "canonical", or a '+'-joined sorted list of active deviations, e.g.
    "dense_routing+supervised_curriculum". Never raises: an unknown value is
    reported as a deviation rather than dropped, since silence is the failure
    mode this field exists to prevent.
    """
    mc = cfg.get("model", {}) or {}
    lw = cfg.get("loss_weights", {}) or {}
    tags: list[str] = []

    routing_mode = mc.get("routing_mode", CANONICAL_ROUTING_MODE)
    if routing_mode != CANONICAL_ROUTING_MODE:
        tags.append(f"routing_{routing_mode}")

    router_noise = mc.get("router_noise", CANONICAL_ROUTER_NOISE)
    if router_noise != CANONICAL_ROUTER_NOISE:
        tags.append(f"router_noise_{router_noise}")

    if mc.get("fixed_depth", False):
        tags.append("fixed_depth")

    # Supervised depth curriculum vs pure ACT. MoE is excluded: it runs at
    # max_depth 1, where there is no depth to allocate, so "pure ACT" would be a
    # claim about a mechanism the architecture does not have.
    if mc.get("halting_supervision", False) and float(
            lw.get("halting_supervision", 0.0)) != 0.0:
        tags.append("supervised_curriculum")

    # T8.2 BUGFIX. This line read `lw.get("routing_oracle", 0.0)` -- a key that
    # exists NOWHERE in this repository: not in config.json, not in any of the
    # config_*.json variants, not in load_config_defaults, not in cli.py, and not
    # in engine.py's loss assembly. So the test was `0.0 != 0.0`, permanently
    # False, and the `oracle_routing` tag could never be emitted.
    #
    # The real routing-supervision weight is `loss_weights.step_routing`
    # (engine.py: step_routing_loss = 0.01*oracle_routing_ce + 0.3*step_cls_loss,
    # then `+ lw["step_routing"] * step_routing_loss`). Any non-zero value trains
    # the router against the ORACLE EXPERT INDEX.
    #
    # Consequence while the typo stood: every oracle-supervised run was stamped
    # variant="canonical". Verified on runs/routedec_oracle_supervised_router_
    # seed42__4bd62d20, whose provenance says routing_supervision_enabled=true /
    # routing_supervision_weight=0.5 and variant="canonical" in the same block.
    # No MEASUREMENT was wrong -- the provenance pair carried the truth, which is
    # how T8.0b validated its arms -- but `variant` is precisely the field that
    # exists to stop an ablation being read as canonical in a results table
    # (plan.md E / T10.E: supervision OFF is canonical, ON is a labelled
    # ablation, never mixed in one table), and it was inoperative for the one
    # ablation the routing half of the paper turns on.
    if float(lw.get("step_routing", 0.0)) != 0.0:
        tags.append("oracle_routing")

    return "+".join(sorted(tags)) if tags else CANONICAL_VARIANT


# T8.2. ONE definition of "which halting objective ran", imported by both
# engine.py (which logs it as provenance.halting_mode / halt_target_mode) and
# run_context.py (whose Gate 0 guard enforces the canonical value frozen by
# T8.0a). The guard runs BEFORE engine.train() builds its provenance block, so it
# cannot read the value off provenance and must derive it -- and a second copy of
# the rule living in the guard is how a guard comes to pass a run whose objective
# no longer matches (the T5.4 trap).
HALTING_MODE_PURE_ACT   = "pure_act"
HALTING_MODE_SUPERVISED = "supervised_curriculum"


def resolve_halting_mode(cfg: dict) -> str:
    """
    "pure_act" or "supervised_curriculum", from the resolved config.

    Requires BOTH the model flag and a non-zero loss weight, matching engine.py:
    a flag set with a zero weight computes the curriculum term and multiplies it
    away, which engine.py refuses outright rather than silently recording as
    supervised.
    """
    mc = cfg.get("model", {}) or {}
    lw = cfg.get("loss_weights", {}) or {}
    supervised = bool(mc.get("halting_supervision", False)) and float(
        lw.get("halting_supervision", 0.0)) != 0.0
    return HALTING_MODE_SUPERVISED if supervised else HALTING_MODE_PURE_ACT



def canonical_run_name(architecture: str, seed: int | None) -> str:
    """
    The stable run name required by updated_rules.md 9: phaseB_<Arch>_seed<S>.

    Seed omitted when no seed is declared -- an undeclared-seed run must not be
    given a name that claims a seed it never set (T6.1).
    """
    arch = str(architecture).lower()
    if arch not in ARCHITECTURES:
        raise ValueError(
            f"architecture must be one of {ARCHITECTURES}, got {architecture!r}"
        )
    stem = f"{CANONICAL_RUN_PREFIX}_{ARCH_DISPLAY[arch]}"
    return stem if seed is None else f"{stem}_seed{int(seed)}"


def stamp_seed_into_run_name(cfg: dict) -> str:
    """
    Append `_seed<S>` to logging.run_name when a seed is declared and the name
    does not already carry one, and return the final name.

    Why this is a separate step from apply_architecture: the seed is not known
    until resolve_overrides has applied `--seed`, and the name is not final
    until `--run_name` has been applied. Stamping earlier would label every run
    with the config-file seed. Custom names are kept and extended rather than
    replaced, so a diagnostic run stays identifiable while still recording which
    seed produced it.
    """
    log = cfg.setdefault("logging", {})
    seed = cfg.get("provenance", {}).get("seed")
    name = str(log.get("run_name") or "run")
    if seed is not None and "_seed" not in name:
        name = f"{name}_seed{int(seed)}"
        log["run_name"] = name
    return name


# CANONICAL_NUM_EXPERTS is defined at the top of this module (T5.1) so the
# defaults in load_config_defaults can use it instead of a second literal.


def apply_architecture(cfg: dict, architecture: str) -> dict:
    """
    Stamp the architecture-specific fields into cfg, in place, and return it.

    moe  : 6 experts, depth 1, no adaptive halting, learned top-1 routing
    mor  : 1 expert,  canonical depth, adaptive halting, routing N/A
    more : 6 experts, canonical depth, adaptive halting, learned top-1 routing
    """
    arch = str(architecture).lower()
    if arch not in ARCHITECTURES:
        raise ValueError(
            f"architecture must be one of {ARCHITECTURES}, got {architecture!r}"
        )

    mc = cfg.setdefault("model", {})
    lw = cfg.setdefault("loss_weights", {})
    canonical_depth = mc.get("max_depth", mc.get("max_steps", 7))

    if arch == "moe":
        mc["num_experts"] = CANONICAL_NUM_EXPERTS
        mc["max_depth"] = 1
        mc["adaptive_halting"] = False
        lw["halting"] = 0.0          # no ponder cost when there is no recursion
        # max_depth == 1 means every token forced-exits at step 1. A depth
        # curriculum target of 2, 3 or 4 is then unreachable by construction, so
        # supervising it would inject a constant, irreducible loss and a
        # gradient that can never be satisfied. MoE has no depth to allocate.
        mc["halting_supervision"] = False
        lw["halting_supervision"] = 0.0
    elif arch == "mor":
        mc["num_experts"] = 1
        mc["max_depth"] = canonical_depth
        mc["adaptive_halting"] = True
        lw["routing_balance"] = 0.0  # nothing to balance with a single expert
        lw["step_routing"] = 0.0     # no routing target with a single expert
    else:  # more
        mc["num_experts"] = CANONICAL_NUM_EXPERTS
        mc["max_depth"] = canonical_depth
        mc["adaptive_halting"] = True

    cfg["architecture"] = arch
    cfg.setdefault("logging", {}).setdefault(
        "run_name", canonical_run_name(arch, None))
    # NOTE: enforce_routing_mode is deliberately NOT called here. At this point
    # the run_name is still the default stamped one line above -- the CLI's
    # --run_name override has not been applied yet -- so gating on the label
    # here refuses even a correctly labelled dense ablation. The gate runs in
    # cli.py AFTER resolve_overrides, against the label the run will really use.
    return cfg


# ---------------------------------------------------------------------------
# Routing-mode gate  (T2.4 / updated_rules.md 1.1 ablation F / plan.md 4.1)
# ---------------------------------------------------------------------------

DENSE_ABLATION_LABEL = "dense_routing_ablation"


def enforce_routing_mode(cfg: dict) -> str:
    """
    Resolve model.routing_mode and refuse dense routing on any unlabelled run.

    plan.md 4.1 states the dense evaluate-all-and-blend path "may remain only
    as: MoRE - Dense Routing Ablation". The danger is not that the dense path
    exists -- it is that a dense run gets written up as if it were MoE, because
    nothing in the output distinguishes them. So the gate is on the RUN LABEL,
    not on a quiet boolean: to use dense routing you must name the run
    something containing "dense_routing_ablation", which then appears in the
    run directory, the W&B run name, and every provenance record downstream.

    Raises ValueError when routing_mode is dense but the label does not say so.
    """
    mc   = cfg.setdefault("model", {})
    mode = str(mc.setdefault("routing_mode", CANONICAL_ROUTING_MODE)).lower()

    if mode not in ROUTING_MODES:
        raise ValueError(
            f"model.routing_mode must be one of {ROUTING_MODES}, got {mode!r}."
        )

    if mode == DENSE_ABLATION_ROUTING_MODE:
        label = " ".join(str(v) for v in (
            cfg.get("logging", {}).get("run_name", ""),
            cfg.get("experiment_group", ""),
            cfg.get("variant", ""),
        )).lower()
        if DENSE_ABLATION_LABEL not in label:
            raise ValueError(
                f"model.routing_mode = {DENSE_ABLATION_ROUTING_MODE!r} evaluates "
                "every expert on every token and blends by router probability. "
                "That is not MoE: no token is routed, every expert sees every "
                "token, and expert independence is destroyed. plan.md 4.1 and "
                "updated_rules.md 1.1 permit it ONLY as an explicitly labelled "
                f"ablation, so the run label must contain "
                f"{DENSE_ABLATION_LABEL!r}. Got run_name="
                f"{cfg.get('logging', {}).get('run_name', '')!r}, "
                f"experiment_group={cfg.get('experiment_group', '')!r}, "
                f"variant={cfg.get('variant', '')!r}. Set the run name (e.g. "
                f"--run_name phaseB_more_{DENSE_ABLATION_LABEL}) so the dense "
                "path cannot be mistaken for canonical MoE in any table."
            )
        mc["variant"] = cfg.setdefault("variant", DENSE_ABLATION_LABEL)

    mc["routing_mode"] = mode
    return mode
