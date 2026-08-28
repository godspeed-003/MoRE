"""
train_more.py -- MoRE: per-token Top-1 expert routing AND weight-shared adaptive recursion.

A launcher, not an implementation. It sets architecture="more" and delegates to
the single shared pipeline in more/ (plan.md 9 forbids parallel per-architecture
training code), so MoE / MoR / MoRE runs stay directly comparable.

    python train_more.py --config config.json --seed 42
"""

import sys

from more.cli import main

if __name__ == "__main__":
    sys.exit(main(default_architecture="more"))
