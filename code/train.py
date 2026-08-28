"""
train.py -- generic entry point. Runs any architecture from the one pipeline.

    python train.py --architecture more --config config.json

The implementation lives in the `more/` package, split into readable modules
(config, families, data, model, metrics, engine, run_context). This file is
deliberately thin: plan.md 9 mandates ONE training system, and a 1.6k-line
monolith was neither reviewable nor cheap to edit.

Architecture-specific shorthands: train_moe.py / train_mor.py / train_more.py.
"""

import sys

from more.cli import main

if __name__ == "__main__":
    sys.exit(main())
