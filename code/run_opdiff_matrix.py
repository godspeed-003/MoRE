"""run_opdiff_matrix.py - T-OD.5/T-OD.6. ONE command that runs the opdiff matrix.

    <interpreter> code/run_opdiff_matrix.py --preflight
    <interpreter> code/run_opdiff_matrix.py --pilot
    <interpreter> code/run_opdiff_matrix.py --all
    <interpreter> code/run_opdiff_matrix.py --arch more --seed 42
    <interpreter> code/run_opdiff_matrix.py --smoke

On the co-author's machine the interpreter is the CUDA one recorded in
ENVIRONMENT.md (`C:/Users/Hp/anaconda3/envs/more_env/python.exe`). That
machine's Anaconda base and `C:\\Python314` have no torch, so a bare `python`
fails at `import torch` rather than falling back to CPU.

WHY A DRIVER SCRIPT AND NOT FIFTEEN DOCUMENTED COMMANDS. The launch line is
already long enough to get wrong in a way that does not fail loudly:

    train.py --config code/config_opdiff.json --architecture more --task opdiff
             --seed 42

Drop `--config code/config_opdiff.json` and it resolves against the ARITHMETIC
defaults -- a different dataset, a different input adapter, a different output
head -- and the run completes, writes a full run directory, and reports an
arithmetic MSE as if it were the opdiff task. Drop `--opdiff_root` and it reads
whatever dataset the config names, which on a fresh clone is nothing at all.
Fifteen hand-typed commands is fifteen chances at that. This file makes the
whole matrix one invocation and puts every check before the first gradient
step.

THE ONE DIFFERENCE FROM THE LANGUAGE DRIVER, and it is the important one:
EVERY opdiff RUN IS EXPLORATORY. `canonical_spec_opdiff.json` is deliberately
UNFROZEN -- every enforced field is null -- so the Gate 0 proxy guard refuses
an opdiff run that claims `canonical_opdiff_b`. That is the intended direction
of failure until the protocol is frozen FROM the pilot's measurements
(T-OD.5), and this driver therefore stamps `exploratory` explicitly rather
than relying on the default. A matrix whose runs are all exploratory can still
be tabled, as long as the table says so; a matrix whose runs CLAIM canonical
status while the spec is unfrozen would be a table nobody could audit.

WHAT THIS SCRIPT DOES NOT DO, deliberately:
  * It does not touch `runs/`. `RunContext` owns the output layout; this driver
    only launches and reads.
  * It does not retry a failed run. A crash mid-matrix is a STOP under
    plan.md 20 -- report, diagnose, fix, re-run -- not something to paper over
    with a loop. It stops at the first failure and says which run died.
  * It does not choose hyperparameters. Every number comes from
    `config_opdiff.json`.
  * It does not average or export. `code/summarize_opdiff_matrix.py` owns
    that, because it refuses to mix dataset versions and config hashes and this
    script must not become a second, weaker path to a results table.

RESUMABILITY IS BY SKIPPING COMPLETED RUNS, not by checkpoint resume. A
multi-hour matrix on a laptop will be interrupted. A run directory is skipped
only when it is (a) for this arm and seed, (b) carrying a finished
`metrics.json`, and (c) for the same dataset. Anything less is re-run rather
than trusted: a directory with a checkpoint but no metrics is an interrupted
run, and its checkpoint is a partially-trained model at an unknown epoch.
"""

import argparse
import glob
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from more.config import (load_config, apply_architecture, apply_task,
                         stamp_opdiff_dataset_versions, resolve_task,
                         TASK_OPDIFF, ARCHITECTURES)
from more.opdiff_data import resolve_opdiff_root, opdiff_versions
from more.run_context import load_canonical_spec, assert_not_silent_proxy

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

CODE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(CODE)
RUNS = os.path.join(REPO, "runs")
OPDIFF_CONFIG = os.path.join(CODE, "config_opdiff.json")
TRAIN = os.path.join(CODE, "train.py")

SPEC = load_canonical_spec(task=TASK_OPDIFF)
SEEDS = list(SPEC["seed_set"])
# The group every opdiff run carries. NOT `canonical_opdiff_b`: the spec is
# unfrozen, the guard refuses that claim, and a run that cannot be admitted to
# a table must not be labelled as if it could.
GROUP = "exploratory"
# MoE first, then MoR, then MoRE: the cheapest arm first, so a protocol
# mistake surfaces after one short run rather than after a long one. The ORDER
# OF RUNS CANNOT AFFECT A RESULT -- every run is independently seeded through
# seeding.py's global seeding -- so this is free.
ARMS = ["moe", "mor", "more"]


# ---------------------------------------------------------------------------
# preflight
# ---------------------------------------------------------------------------

def _dataset_root():
    return resolve_opdiff_root(load_config(OPDIFF_CONFIG)["data"]["opdiff_root"])


def resolved(arch, seed, root=None):
    """The config an opdiff run of (arch, seed) will actually resolve to.

    Built through cli.py's own call order so this is the real thing and not a
    parallel reimplementation: load the file, apply the architecture, apply the
    task, re-stamp the dataset version strings.
    """
    cfg = load_config(OPDIFF_CONFIG)
    apply_architecture(cfg, arch)
    apply_task(cfg, TASK_OPDIFF)
    stamp_opdiff_dataset_versions(cfg)
    cfg.setdefault("provenance", {})["seed"] = seed
    cfg.setdefault("logging", {})["experiment_group"] = GROUP
    return cfg


def preflight(verbose=True):
    """Every check that can fail BEFORE a GPU-hour is spent.

    Returns `(problems, warnings)`. A PROBLEM refuses the matrix -- it will
    cost GPU-hours or invalidate the run. A WARNING is a measured fact the
    operator should know and then proceed anyway.
    """
    problems, warnings = [], []

    def say(ok, what, detail=""):
        if verbose:
            print(f"  [{'ok ' if ok else 'XX '}] {what}" +
                  (f"  -- {detail}" if detail else ""))

    # 1. torch, and a real device. `torch.cuda.is_available()` returning True on
    #    a driver that then cannot launch a kernel is a real failure mode, so a
    #    matmul is run rather than a flag read.
    try:
        import torch
        dev_ok = torch.cuda.is_available()
        name, total = "cpu", 0
        if dev_ok:
            p = torch.cuda.get_device_properties(0)
            name, total = p.name, p.total_memory // 1048576
            (torch.randn(64, 64, device="cuda") @
             torch.randn(64, 64, device="cuda")).sum().item()
        say(dev_ok, "CUDA device usable", f"{name}, {total} MiB"
            if dev_ok else "no CUDA device")
        if not dev_ok:
            problems.append(
                "No usable CUDA device. On the co-author's machine the interpreter "
                "must be C:/Users/Hp/anaconda3/envs/more_env/python.exe -- base "
                "and C:\\Python314 have no torch at all (ENVIRONMENT.md).")
        elif total < 5800:
            problems.append(
                f"This card reports only {total} MiB. The MEASURED peak of the "
                "heaviest opdiff arm (MoR, one dense FFN at ffn_mult 28 over "
                "304-token rows at batch 48) is 5,491 MiB, measured on a 6,143 "
                "MiB card at T-OD.4 -- a 91% occupancy, which is the soft-cliff "
                "regime the language calibration found. On a card below ~5.8 GB "
                "this will OOM partway through a multi-hour matrix, and "
                "lowering batch_size is NOT the fix: it is a protocol field, "
                "and a run at a different batch is a different experiment that "
                "must be labelled as one.")
        elif total < 7000:
            say(True, "VRAM headroom is thin but sufficient",
                f"worst arm peaks at 5,491 MiB of {total} MiB "
                f"({5491 / total * 100:.0f}%)")
            warnings.append(
                f"{total} MiB card: the opdiff MoR arm peaks at a MEASURED "
                "5,491 MiB -- 91% of this card, and the language calibration "
                "found that above ~87% the epoch time goes UP rather than the "
                "run failing. Prefer the co-author's 8 GB card for the matrix "
                "(the same arm is 69% there). Nothing else may use the GPU "
                "during a run.")
    except Exception as e:
        say(False, "torch imports", f"{type(e).__name__}: {e}")
        problems.append(f"torch is not importable: {e}")

    # 2. THE DATASET. Not tracked in git (113 MB), so this is the check a fresh
    #    clone fails, and the one that has to name the exact build command.
    root = _dataset_root()
    man_path = os.path.join(root, "meta.json")
    ok = os.path.exists(man_path)
    detail = ""
    if ok:
        m = json.load(open(man_path, "r", encoding="utf-8"))
        detail = (f"train {m['rows']['train']:,} rows x {m['row_len']}, "
                  f"V={m['vocab_size']}, {m.get('dataset_version', 'no version')}")
    say(ok, f"opdiff dataset built at {os.path.basename(root)}",
        detail or f"{man_path} missing")
    if not ok:
        problems.append(
            f"The opdiff dataset is not built ({man_path} missing). It is NOT "
            "tracked in git. Build it with:\n"
            f"    python make_opdiff_dataset_v2.py --out {os.path.basename(root)}\n"
            "  (about 15 s; 113 MB; deterministic at seed 0). Then re-run the "
            "preflight. Record the dataset_version it prints -- it is the "
            "identity of the bytes and it goes into every run's provenance.")
    else:
        for split in ("train", "val", "test", "ood_interp", "ood_extrap"):
            for suffix in ("x", "group", "diff", "k", "sub"):
                arr = os.path.join(root, f"{split}_{suffix}.npy")
                if not os.path.exists(arr):
                    problems.append(f"{arr} is missing though meta.json lists the split.")
        m = json.load(open(man_path, "r", encoding="utf-8"))
        # THE OOD SPLITS ARE THE POINT of the v3 dataset, so their absence is a
        # problem and not a detail: without them the held-out iteration counts
        # are not evaluated at all.
        for split in ("ood_interp", "ood_extrap"):
            if m.get("rows", {}).get(split, 0) <= 0:
                problems.append(
                    f"the {split} split is empty or missing from meta.json. The "
                    "held-out iteration counts ARE the OOD evaluation; without "
                    "them the run would report an in-distribution number and "
                    "call it the OOD result.")
        _train_k = set(m.get("train_k", []))
        _interp_k = set(m.get("ood_interp_k", []))
        _extrap_k = set(m.get("ood_extrap_k", []))
        if _train_k & (_interp_k | _extrap_k):
            problems.append(
                f"a trained k value {_train_k & (_interp_k | _extrap_k)} is also "
                "in a held-out list. The held-out lists are the whole point: a k "
                "the model has trained on is not an OOD k.")
        if m.get("dataset_version") is None:
            warnings.append(
                "meta.json carries no dataset_version: the dataset was generated "
                "by a generator that does not write content hashes, so the "
                "identity recorded in provenance is the weaker fallback "
                "fingerprint (meta.json plus a 4 MiB edge of the train split). "
                "It is still content-sensitive, but a full content hash would be "
                "better.")

    # 3. THE SPEC IS DELIBERATELY UNFROZEN. This is a NOTE, not a problem:
    #    every opdiff run is exploratory, and the guard refusing a canonical
    #    claim is the designed behaviour.
    nulls = sorted(k for k, v in SPEC["enforced_fields"].items() if v is None)
    say(True, "the opdiff spec is unfrozen BY DESIGN (all runs are exploratory)",
        f"{len(nulls)} null enforced fields, group={SPEC['canonical_group']} "
        f"is refused by the guard")

    # 4. The real guard, on the real resolved config, for every arm. A run that
    #    claims canonical status must be refused -- and it must be refused for
    #    the RIGHT reason (unfrozen spec), not by accident.
    for arch in ARMS:
        try:
            got = assert_not_silent_proxy(resolved(arch, SEEDS[0]))
            ok = got == GROUP
            say(ok, f"{arch}: the run is stamped {GROUP!r}", f"group={got!r}")
            if not ok:
                problems.append(f"{arch} was stamped {got!r}, not {GROUP!r}.")
        except Exception as e:
            say(False, f"{arch}: the proxy guard REFUSED the config",
                f"{type(e).__name__}: {str(e)[:90]}")
            problems.append(
                f"{arch}: the proxy guard refuses the opdiff config: {e}")

    # 5. W&B. A run dies in `wandb.init` with `UsageError: No API key
    #    configured` and leaves a directory-shaped artifact behind when it does.
    #    Offline is a legitimate answer; unset is not.
    mode = os.environ.get("WANDB_MODE", "")
    have_key = bool(os.environ.get("WANDB_API_KEY")) or os.path.exists(
        os.path.expanduser("~/.netrc"))
    ok = mode in ("offline", "disabled") or have_key
    say(ok, "W&B is settled", f"WANDB_MODE={mode!r}" if mode else
        ("api key present" if have_key else "no key and no WANDB_MODE"))
    if not ok:
        problems.append(
            "W&B is neither logged in nor explicitly offline, so the first run "
            "will die in wandb.init after the guard has passed. Pick one:\n"
            "    export WANDB_MODE=offline      # metrics still land in runs/\n"
            "    wandb login                    # if the project should stream\n"
            "  runs/<id>/metrics.json is written either way; W&B is convenience, "
            "not the record.")

    # 6. Disk. One checkpoint per run, measured from an existing opdiff
    #    checkpoint rather than guessed.
    ckpts = glob.glob(os.path.join(RUNS, "opdiffB_*", "checkpoint.pt"))
    per = os.path.getsize(ckpts[0]) if ckpts else 16 * 1024 ** 2
    need = per * len(ARMS) * len(SEEDS)
    try:
        import shutil
        free = shutil.disk_usage(REPO).free
        ok = free > need * 3
        say(ok, "disk headroom for 15 checkpoints",
            f"need ~{need / 1024**3:.1f} GiB, free {free / 1024**3:.1f} GiB")
        if not ok:
            problems.append(
                f"Only {free / 1024**3:.1f} GiB free; 15 checkpoints need "
                f"~{need / 1024**3:.1f} GiB plus room for logs.")
    except Exception:
        pass

    return problems, warnings


# ---------------------------------------------------------------------------
# launching
# ---------------------------------------------------------------------------

def finished_run(arch, seed, root=None):
    """An existing COMPLETE run directory for (arch, seed), or None.

    All FOUR of arm, seed, a finished metrics.json and the CONFIGURED dataset
    version are required. A directory with a checkpoint but no metrics is an
    INTERRUPTED run and is re-run, not skipped: its checkpoint is a partially
    trained model at an unknown epoch, and treating it as done is how such a
    thing reaches a results table.

    THE DATASET VERSION IS PART OF THE IDENTITY, and it is the check that keeps
    a smoke run out of the matrix. The opdiff dataset is not tracked in git,
    so it is regenerated per machine, and a smoke run over `opdiff_v2_smoke`
    lands in a directory named exactly like a matrix run (`opdiffB_MoRE_seed42`
    -- the name comes from the config, not from the data). Without this filter
    a smoke run would be skipped as "already complete" and the matrix would
    silently contain a run trained on 1% of the data.
    """
    want = None
    try:
        dsv, _ = opdiff_versions(root or _dataset_root())
        want = dsv
    except Exception:
        want = None
    for d in sorted(glob.glob(os.path.join(RUNS, f"opdiffB_{arch}_seed{seed}*"))):
        mp, rc = (os.path.join(d, "metrics.json"),
                  os.path.join(d, "resolved_config.json"))
        if not (os.path.exists(mp) and os.path.exists(rc)):
            continue
        try:
            cfg = json.load(open(rc, "r", encoding="utf-8"))
            met = json.load(open(mp, "r", encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if cfg.get("architecture") != arch:
            continue
        if int(cfg.get("provenance", {}).get("seed", -1)) != int(seed):
            continue
        if want is not None and (cfg.get("data") or {}).get("dataset_version") != want:
            continue
        if met.get("val/task_loss") is None:
            continue
        return d
    return None


def launch(arch, seed, group=GROUP, extra=(), dry_run=False):
    """Run one arm/seed as a child process. Returns (returncode, seconds)."""
    cmd = [sys.executable, TRAIN,
           "--config", OPDIFF_CONFIG,
           "--task", TASK_OPDIFF,
           "--architecture", arch,
           "--seed", str(seed),
           "--experiment_group", group, *extra]
    print("\n" + "=" * 78)
    print(f"  {arch.upper()}  seed {seed}   group={group}")
    print("  " + " ".join(os.path.basename(c) if c.endswith(".py") or
                          c.endswith(".json") else c for c in cmd))
    print("=" * 78, flush=True)
    if dry_run:
        return 0, 0.0
    t0 = time.perf_counter()
    # cwd=CODE and NOT capture_output: a multi-hour run with a swallowed stdout
    # is unobservable, and RunContext already tees the same stream into
    # runs/<id>/stdout.log.
    proc = subprocess.run(cmd, cwd=CODE)
    return proc.returncode, time.perf_counter() - t0


def summarise(pairs):
    """Read back what actually landed. Never a number this script computed."""
    print("\n" + "=" * 78)
    print("  RESULTS AS RECORDED (read from runs/<id>/metrics.json)")
    print("=" * 78)
    print("  primary metric = val/task_loss, nats/token over the generator's")
    print("  scored positions (answer digits + EOS).")
    print(f"  {'arch':<6} {'seed':>4}  {'val_loss':>9} {'depth':>6} "
          f"{'agree':>7} {'hungar':>7}  run")
    for arch, seed in pairs:
        d = finished_run(arch, seed)
        if d is None:
            print(f"  {arch:<6} {seed:>4}  {'MISSING':>9} {'N/A':>6} "
                  f"{'N/A':>7} {'N/A':>7}")
            continue
        m = json.load(open(os.path.join(d, "metrics.json"), "r",
                           encoding="utf-8"))

        def _f(k):
            v = m.get(k)
            return f"{v:.4f}" if isinstance(v, (int, float)) else "N/A"

        print(f"  {arch:<6} {seed:>4}  {_f('val/task_loss'):>9} "
              f"{_f('val/avg_recursion_steps'):>6} "
              f"{_f('val/routing_agreement_with_group'):>7} "
              f"{_f('val/routing_hungarian_accuracy'):>7}  "
              f"{os.path.basename(d)}")
    print("\n  'agree' is AGREEMENT WITH THE GENERATOR'S GROUP PRIOR, not")
    print("  accuracy; 'hungar' is the permutation-invariant matched accuracy,")
    print("  which is the primary routing number. Do not hand-copy these into a")
    print("  table. Aggregate with:")
    print("      python code/summarize_opdiff_matrix.py")
    print("  (that is the ONLY aggregation command for this task.)")


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Run the opdiff matrix (T-OD.5/T-OD.6).")
    ap.add_argument("--all", action="store_true",
                    help="the full 3 arms x 5 seeds matrix, skipping completed runs")
    ap.add_argument("--pilot", action="store_true",
                    help="3 arms x seed 42 only -- the pilot the budget is measured from")
    ap.add_argument("--arch", choices=ARCHITECTURES, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--preflight", action="store_true",
                    help="run every check and exit without training")
    ap.add_argument("--smoke", action="store_true",
                    help="one short run on the SMOKE dataset, to prove the "
                         "machine works without spending a matrix hour")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the commands that would run")
    ap.add_argument("--force", action="store_true",
                    help="re-run arms that already have a complete run directory")
    args = ap.parse_args(argv)

    print("\n=== PREFLIGHT " + "=" * 63)
    problems, warns = preflight()
    if problems:
        print(f"\n{len(problems)} problem(s) must be fixed before the matrix "
              f"starts:\n")
        for i, p in enumerate(problems, 1):
            print(f"  {i}. {p}\n")
        return 1
    for w in warns:
        print(f"\n  NOTE: {w}")
    print("\n  preflight clean.")
    if args.preflight:
        return 0

    if args.smoke:
        # The SMOKE dataset, 1 epoch, exploratory. Proves the machine can train
        # and write a run directory without spending a matrix hour. It CANNOT
        # contaminate the matrix: the smoke dataset has its own dataset_version,
        # so a table built on the full dataset can never admit it.
        #
        # THE RUN NAME IS EXPLICIT, and that is the point. The default name
        # comes from the config and is `opdiffB_<Arch>_seed<S>__<hash>` -- the
        # same shape as a matrix cell, differing only in an 8-hex suffix that
        # nobody can read. A smoke run whose directory is indistinguishable
        # from a matrix cell is a smoke run that ends up in a table. The name
        # says SMOKE, and it uses the DISPLAY spelling of the arm (MoE, MoR,
        # MoRE) that every other run name in this repository uses.
        arch = args.arch or "more"
        disp = {"moe": "MoE", "mor": "MoR", "more": "MoRE"}[arch]
        name = f"opdiffB_smoke_{disp}_seed44"
        rc, secs = launch(arch, 44, group="opdiffB_smoke",
                          extra=["--epochs", "1",
                                 "--opdiff_root", "opdiff_v3_smoke",
                                 "--run_name", name],
                          dry_run=args.dry_run)
        print(f"\n  smoke run {'OK' if rc == 0 else f'FAILED rc={rc}'} "
              f"in {secs / 60:.1f} min")
        return rc

    if args.pilot:
        pairs = [(a, SEEDS[0]) for a in ARMS]
    elif args.all:
        pairs = [(a, s) for a in ARMS for s in SEEDS]
    elif args.arch and args.seed is not None:
        pairs = [(args.arch, args.seed)]
    elif args.arch:
        pairs = [(args.arch, s) for s in SEEDS]
    else:
        ap.error("give --all, --pilot, --arch [--seed], --preflight or --smoke")

    todo = []
    for arch, seed in pairs:
        done = finished_run(arch, seed)
        if done and not args.force:
            print(f"  skip  {arch} seed {seed}  -- already complete: "
                  f"{os.path.basename(done)}")
        else:
            todo.append((arch, seed))

    if not todo:
        print("\n  nothing to do; every requested run is already complete.")
        summarise(pairs)
        return 0

    epochs = int(load_config(OPDIFF_CONFIG)["training"]["epochs"])
    print(f"\n  {len(todo)} run(s) to go at {epochs} epoch(s) each.")
    print("  Wall-clock is not estimated here: the per-arm epoch time is a")
    print("  MEASUREMENT (T-OD.5), and an estimate built on a guessed rate is")
    print("  how the language matrix was once costed at 15 days instead of 39")
    print("  hours. Measure it on the pilot and put it in the handoff.")

    t_start = time.perf_counter()
    for i, (arch, seed) in enumerate(todo, 1):
        print(f"\n  [{i}/{len(todo)}]  elapsed "
              f"{(time.perf_counter() - t_start) / 3600:.2f} h")
        rc, secs = launch(arch, seed, dry_run=args.dry_run)
        if rc != 0:
            # STOP, per plan.md 20. Do not continue the matrix around a failure.
            print(f"\n  {arch} seed {seed} FAILED (rc={rc}) after "
                  f"{secs / 60:.1f} min.")
            print("  STOPPING. plan.md 20: report the failure with its actual "
                  "output, find the cause, fix it, re-run. The completed runs are "
                  "intact and will be skipped when you resume.")
            summarise(pairs)
            return rc
        print(f"  done in {secs / 60:.1f} min")

    print(f"\n  matrix complete in {(time.perf_counter() - t_start) / 3600:.2f} h")
    summarise(pairs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
