"""
more - Mixture of Recursive Experts.

ONE training system (plan.md 9). `architecture` selects MoE / MoR / MoRE; the
dataset, tokenizer, engine and metrics are shared so the three are directly
comparable. train_moe.py / train_mor.py / train_more.py are thin launchers over
this same engine, not parallel implementations.

Modules:
    families      operation -> expert manifest (single source of labels)
    config        config loading and precedence
    data          dataset + per-step tokenisation
    model         router, experts, recursive block, halting
    metrics       entropy, cosine, routing, depth metrics + paper figures
    engine        the training loop
    run_context   per-run output dirs, provenance, Gate 0 proxy guard
    seeding       global deterministic seeding (T6.1): random / numpy / torch /
                  CUDA / DataLoader generator + workers
"""

from .config import load_config
from .families import (OP_TO_EXPERT, EXPERT_FAMILY_LABELS, FAMILY_TO_IDX,
                       ALL_OP_NAMES, OP_NAME_TO_IDX, NUM_OP_TYPES,
                       NUM_EXPERTS_CANONICAL, expert_labels)
from .data import MoREDataset
# T-L2.3. Deliberately NOT re-exporting anything from `.lang_families` here: the
# eight names above come from `.families` unconditionally, and adding a second
# manifest's labels to the same namespace is how `EXPERT_FAMILY_LABELS` ends up
# meaning whichever module imported last. Language code imports
# `more.lang_families` by name; see that module's `_ARITHMETIC_ONLY`.
from .lang_data import MoRELanguageDataset, TRAIN_SPLIT_VERSION
from .model import MoEBlock, MoREWrapper, MoREModel
from .engine import train
from .run_context import RunContext, resolve_overrides, ProxyGuardError
from .seeding import (seed_everything, apply_seeding, resolve_effective_seed,
                      make_generator, seed_worker, CANONICAL_SEED_SET)

__all__ = [
    "load_config", "MoREDataset", "MoRELanguageDataset", "TRAIN_SPLIT_VERSION",
    "MoEBlock", "MoREWrapper", "MoREModel",
    "train", "RunContext", "resolve_overrides", "ProxyGuardError",
    "OP_TO_EXPERT", "EXPERT_FAMILY_LABELS", "FAMILY_TO_IDX",
    "ALL_OP_NAMES", "OP_NAME_TO_IDX", "NUM_OP_TYPES",
    # T5.1: the width-aware label helper. Exported because every consumer of a
    # confusion matrix or per-expert metric key needs it, and reaching for
    # EXPERT_FAMILY_LABELS instead is the defect Gate 4 exists to catch.
    "NUM_EXPERTS_CANONICAL", "expert_labels",
    # T6.1: one seeding entry point. Anything that builds a model or a loader
    # outside engine.train() (probes, tests, ablation drivers) must call
    # seed_everything, not torch.manual_seed alone -- the latter leaves
    # random/numpy/cuDNN untouched.
    "seed_everything", "apply_seeding", "resolve_effective_seed",
    "make_generator", "seed_worker", "CANONICAL_SEED_SET",
]
