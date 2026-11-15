"""
train_moe.py -- MoE baseline: 6 independent experts, learned Top-1 routing, depth 1 (no recursion).

A launcher, not an implementation. It sets architecture="moe" and delegates to
the single shared pipeline in more/ (plan.md 9 forbids parallel per-architecture
training code), so MoE / MoR / MoRE runs stay directly comparable.

    python train_moe.py --config config.json --seed 42
"""

import sys

from more.cli import main

if __name__ == "__main__":
    sys.exit(main(default_architecture="moe"))
