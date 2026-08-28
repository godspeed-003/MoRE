"""
train_mor.py -- MoR baseline: ONE shared block, adaptive recursion depth, no expert routing.

A launcher, not an implementation. It sets architecture="mor" and delegates to
the single shared pipeline in more/ (plan.md 9 forbids parallel per-architecture
training code), so MoE / MoR / MoRE runs stay directly comparable.

    python train_mor.py --config config.json --seed 42
"""

import sys

from more.cli import main

if __name__ == "__main__":
    sys.exit(main(default_architecture="mor"))
