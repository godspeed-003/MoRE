"""seeding.py -- T6.1 / plan.md §8.1: the ONE global seeding function.

Before this module, the only thing seeded anywhere in the training path was the
`random_split` generator (`torch.Generator().manual_seed(subset_seed)`). Python's
`random`, NumPy, the global torch CPU generator, the CUDA generators, cuDNN
algorithm selection and the DataLoader shuffle order were all unseeded, so two
runs with the same `--seed` differed in weight initialisation, dropout masks and
batch order. Seed variance (`updated_rules.md` §5: report mean ± std over seeds
42-46) is only interpretable if the seed is the *only* thing that varies.

What is covered, in the order plan.md §8.1 lists it:

    random                      random.seed
    numpy                       np.random.seed
    torch                       torch.manual_seed (CPU + all CUDA devices)
    torch.cuda                  torch.cuda.manual_seed_all
    DataLoader generator        make_generator("dataloader_shuffle")
    DataLoader worker           seed_worker

Honesty rule (plan.md §8.1, last paragraph): "If strict deterministic execution
conflicts with a necessary kernel, document the exception instead of pretending
exact determinism." So `seed_everything` returns a report of what it actually
managed to set, every failure to set something is appended to `report["notes"]`,
and that list is written into provenance as `determinism_exceptions`. It is never
silently empty: the no-exception case records the string "none".
"""

from __future__ import annotations

import os
import random
import re
import warnings

import numpy as np
import torch

# updated_rules.md §5 / plan.md §8.1. The frozen seed set. canonical_spec.json
# carries the same list for the Gate 0 guard; test_phase6_seeding.py asserts the
# two agree, because a seed set that exists in two places is a seed set that can
# disagree with itself.
CANONICAL_SEED_SET = (42, 43, 44, 45, 46)

# Used only when no seed is declared (no --seed, no training.seed). Such a run is
# still bit-reproducible, but it is NOT canonical: provenance.seed stays None, so
# the Gate 0 guard still refuses to let it claim canonical_phase_b.
DEFAULT_SEED = 42

# cuBLAS needs a fixed workspace to make GEMM reductions deterministic on CUDA
# >= 10.2. It is read when the CUDA context is created, so setting it after that
# point has no effect -- hence the "already initialised" note below.
CUBLAS_WORKSPACE_CONFIG = ":4096:8"

# Sub-stream offsets. The DataLoader gets its own generator rather than drawing
# from the global torch RNG, so batch order does not silently change whenever the
# number of initialiser draws changes. That coupling is not hypothetical: in
# Phase 5, narrowing cls_head from 7 to 6 outputs shifted every later parameter
# draw and flipped which experts the router picked.
_PURPOSE_OFFSETS = {"dataloader_shuffle": 10_007}


def resolve_effective_seed(cfg: dict) -> tuple[int, str]:
    """
    The seed the RNGs will actually get, plus where it came from.

    Precedence: --seed (recorded as provenance.seed) > training.seed > default.

    Deliberately does NOT write provenance.seed. That field means "a seed was
    declared" and is what the Gate 0 proxy guard checks; defaulting it would let
    an undeclared run pass the canonical check by accident.
    """
    prov = cfg.get("provenance") or {}
    tc = cfg.get("training") or {}
    if prov.get("seed") is not None:
        return int(prov["seed"]), "cli"
    if tc.get("seed") is not None:
        return int(tc["seed"]), "config"
    return DEFAULT_SEED, "default"


def make_generator(seed: int, purpose: str) -> torch.Generator:
    """A dedicated torch.Generator for one consumer (e.g. DataLoader shuffling)."""
    if purpose not in _PURPOSE_OFFSETS:
        raise KeyError(
            f"unknown generator purpose {purpose!r}. Add it to _PURPOSE_OFFSETS "
            "with its own offset -- two consumers sharing one sub-stream would "
            "make each one's draws depend on how often the other was called."
        )
    g = torch.Generator()
    g.manual_seed((int(seed) + _PURPOSE_OFFSETS[purpose]) % (2 ** 63))
    return g


def seed_worker(worker_id: int) -> None:
    """
    DataLoader `worker_init_fn`. Module-level on purpose: a closure or lambda is
    not picklable, so with num_workers > 0 on Windows (spawn start method) it
    would fail at worker launch.

    torch already gives worker *w* the seed `base_seed + w`, where base_seed is
    drawn from the loader's generator, so `torch.initial_seed()` is both
    per-worker distinct and reproducible. `worker_id` is therefore not folded in
    again; it is kept in the signature because DataLoader passes it positionally.
    """
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# --------------------------------------------------------------------------- #
# Non-determinism observed AT RUNTIME
# --------------------------------------------------------------------------- #
# `seed_everything`'s `notes` can only report what went wrong while seeding. A
# kernel with no deterministic implementation only announces itself when it is
# first CALLED, which is somewhere in the middle of training. Recording "none"
# in provenance while 281 such warnings sat in stdout.log would be exactly the
# kind of clean-looking-but-false record CLAUDE.md §4 forbids, so the warnings
# are collected here and written to provenance at the end of the run.
#
# The dedupe is deliberate: torch re-warns on every call, and `wandb.watch`'s
# gradient histograms alone produced 281 identical lines in a 1-epoch run. One
# line per distinct op is documentation; 281 copies is noise that buries the rest
# of the log.

_NONDET_OPS: set[str] = set()
_NONDET_RE = re.compile(r"([\w:.]+) does not have a deterministic implementation")
_RECORDER_INSTALLED = False


def _install_nondeterminism_recorder() -> None:
    global _RECORDER_INSTALLED
    if _RECORDER_INSTALLED:
        return
    previous = warnings.showwarning

    def _showwarning(message, category, filename, lineno, file=None, line=None):
        m = _NONDET_RE.search(str(message))
        if m:
            op = m.group(1)
            if op in _NONDET_OPS:
                return                      # already documented once
            _NONDET_OPS.add(op)
        return previous(message, category, filename, lineno, file, line)

    warnings.showwarning = _showwarning
    _RECORDER_INSTALLED = True


def nondeterministic_ops_observed() -> list[str]:
    """Ops that warned about missing determinism so far. Never an empty list."""
    return sorted(_NONDET_OPS) or ["none"]


def seed_everything(seed: int, *, deterministic: bool = True,
                    strict: bool = False, verbose: bool = True) -> dict:
    """
    Seed every RNG stream in the process and request deterministic kernels.

    Args:
        deterministic : set cudnn.deterministic, clear cudnn.benchmark, and turn
                        on torch.use_deterministic_algorithms.
        strict        : with deterministic, make a non-deterministic op RAISE
                        instead of warn. Off by default: the top-1 dispatch path
                        uses scatter/index writes whose CUDA kernels have no
                        deterministic implementation in torch 2.5, and aborting
                        the run is worse than recording the exception.

    Returns:
        A report dict. Every field is a real measurement read back from torch
        after the fact -- not the value that was requested -- so a setting that
        did not take effect shows up as False rather than as a claim.
    """
    seed = int(seed)
    if not (0 <= seed < 2 ** 32):
        raise ValueError(
            f"seed={seed} is outside [0, 2**32). numpy.random.seed rejects "
            "anything else, so a run would die after the torch RNGs were "
            "already reseeded -- half-seeded and unreproducible."
        )

    notes: list[str] = []
    cuda_available = torch.cuda.is_available()

    # -- cuBLAS workspace: must be set BEFORE the CUDA context exists ------
    cublas_cfg = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if deterministic and cublas_cfg is None:
        if cuda_available and torch.cuda.is_initialized():
            notes.append(
                "CUBLAS_WORKSPACE_CONFIG was unset and the CUDA context was "
                "already initialised, so cuBLAS GEMM reductions in this process "
                "may be non-deterministic. Export "
                f"CUBLAS_WORKSPACE_CONFIG={CUBLAS_WORKSPACE_CONFIG} before "
                "launching to close this gap."
            )
        else:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_WORKSPACE_CONFIG
            cublas_cfg = CUBLAS_WORKSPACE_CONFIG

    # Only affects CHILD processes (hash randomisation is fixed at interpreter
    # start), which is exactly what DataLoader workers are.
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)          # CPU + all CUDA devices
    if cuda_available:
        torch.cuda.manual_seed_all(seed)
    else:
        notes.append("CUDA not available: CUDA generators were not seeded.")

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=not strict)
        except Exception as e:                                  # pragma: no cover
            notes.append(
                f"torch.use_deterministic_algorithms(True) failed: {e}. cuDNN "
                "flags are still set, but op-level determinism is not enforced."
            )
        # warn_only warnings are per-location and would be printed once and then
        # deduplicated away. Force every occurrence through so an op that first
        # fires late in training is still seen, and let the recorder below
        # collapse the repeats by op name.
        warnings.filterwarnings(
            "always", message=".*does not have a deterministic implementation.*"
        )
        _install_nondeterminism_recorder()
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        notes.append(
            "deterministic=False: cuDNN autotuning is on and kernel selection "
            "may vary between runs. Losses are NOT expected to match bitwise."
        )

    report = {
        "seed": seed,
        "determinism_requested": bool(deterministic),
        "determinism_mode": ("strict" if (deterministic and strict)
                             else "warn_only" if deterministic else "off"),
        "torch_deterministic_algorithms":
            bool(torch.are_deterministic_algorithms_enabled()),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cublas_workspace_config": cublas_cfg or "N/A",
        "cuda_available": bool(cuda_available),
        "cuda_device_count": torch.cuda.device_count() if cuda_available else 0,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda or "N/A",
        "streams_seeded": (
            ["random", "numpy", "torch_cpu"]
            + (["torch_cuda"] if cuda_available else [])
            + ["dataloader_generator", "dataloader_worker"]
        ),
        # CLAUDE.md §4: no sentinel may be reported as a measurement. An empty
        # list here would read as "nothing to report" only if you knew the field
        # existed; "none" says it explicitly.
        "notes": notes or ["none"],
    }

    if verbose:
        print(f"[Seed] seed={seed}  determinism={report['determinism_mode']}  "
              f"streams={'+'.join(report['streams_seeded'])}")
        print(f"[Seed] cudnn.deterministic={report['cudnn_deterministic']}  "
              f"cudnn.benchmark={report['cudnn_benchmark']}  "
              f"deterministic_algorithms="
              f"{report['torch_deterministic_algorithms']}  "
              f"CUBLAS_WORKSPACE_CONFIG={report['cublas_workspace_config']}")
        for n in notes:
            print(f"[Seed] exception: {n}")

    return report


def apply_seeding(cfg: dict, *, verbose: bool = True) -> tuple[int, dict]:
    """
    Resolve the seed from a resolved config, seed the process, and record what
    happened in `cfg["provenance"]`.

    Config knobs (both optional, both defaulting to the reproducible choice):
        training.seed               fallback seed when --seed is not passed
        training.deterministic      default True
        training.deterministic_strict  default False (warn instead of raise)
    """
    seed, source = resolve_effective_seed(cfg)
    tc = cfg.get("training") or {}
    report = seed_everything(
        seed,
        deterministic=bool(tc.get("deterministic", True)),
        strict=bool(tc.get("deterministic_strict", False)),
        verbose=verbose,
    )
    report["seed_source"] = source

    prov = cfg.setdefault("provenance", {})
    # provenance.seed (the DECLARED seed) is left exactly as it was: it is the
    # field Gate 0 tests, and an undeclared run must keep failing that test.
    prov["resolved_seed"] = seed
    prov["seed_source"] = source
    prov["seed_declared"] = prov.get("seed") is not None
    prov["seed_set_canonical"] = list(CANONICAL_SEED_SET)
    prov["determinism_mode"] = report["determinism_mode"]
    prov["torch_deterministic_algorithms"] = report["torch_deterministic_algorithms"]
    prov["cudnn_deterministic"] = report["cudnn_deterministic"]
    prov["cudnn_benchmark"] = report["cudnn_benchmark"]
    prov["cublas_workspace_config"] = report["cublas_workspace_config"]
    prov["determinism_exceptions"] = report["notes"]

    if verbose and source != "cli":
        print(f"[Seed] no --seed given; using the {source} seed {seed}. This run "
              "is reproducible but declares no seed, so Gate 0 will refuse any "
              "canonical claim (provenance.seed is null).")
    return seed, report
