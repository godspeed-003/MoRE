"""run_language_matrix.py - T-L7.4. ONE command that runs the canonical language matrix.

    <interpreter> code/run_language_matrix.py --preflight
    <interpreter> code/run_language_matrix.py --all
    <interpreter> code/run_language_matrix.py --arch more --seed 42
    <interpreter> code/run_language_matrix.py --smoke

On Ayan's machine the interpreter is
`C:/Users/Hp/anaconda3/envs/more_env/python.exe` -- see ENVIRONMENT.md 8. That
machine's Anaconda base and C:\\Python314 have no torch, so a bare `python` fails
at `import torch` rather than falling back to CPU.

WHY A DRIVER SCRIPT AND NOT FIFTEEN DOCUMENTED COMMANDS. The launch line is
already long enough to get wrong in a way that does not fail loudly:

    train.py --config code/config_language.json --architecture more --seed 42
             --experiment_group canonical_lang_b

Drop `--experiment_group` and the run completes, writes a full run directory, and
is stamped `exploratory` -- so it is silently absent from the headline table and
nobody notices until the export is short. Drop `--config code/config_language.json`
and it resolves against the ARITHMETIC defaults, which the proxy guard does refuse,
but only after the operator has waited for `wandb.init`. Fifteen hand-typed
commands is fifteen chances at that. This file makes the whole matrix one
invocation and puts every check before the first gradient step.

WHAT THIS SCRIPT DOES NOT DO, deliberately:
  * It does not touch `runs/`. `RunContext` owns the output layout
    (CLAUDE.md 5, per-run directories only); this driver only launches and reads.
  * It does not retry a failed run. A crash mid-matrix is a STOP under plan.md 20
    -- report, diagnose, fix, re-run -- not something to paper over with a loop.
    It stops at the first failure and says which run died.
  * It does not choose hyperparameters. Every number comes from
    `config_language.json`, which is checked against `canonical_spec_language.json`
    field by field before anything launches. A flag that could change a frozen
    protocol field is not offered.
  * It does not average or export. `code/seed_stats.py` and
    `code/export_results.py` own that, because they refuse to mix dataset versions
    and config hashes and this script must not become a second, weaker path to a
    results table.

RESUMABILITY IS BY SKIPPING COMPLETED RUNS, not by checkpoint resume. A 39-hour
matrix on a laptop will be interrupted. `--all` looks for an existing run
directory that is (a) for this arm and seed, (b) stamped with the canonical group,
(c) carrying a finished `metrics.json`, and skips it. Anything less than all three
is re-run rather than trusted: a directory with a checkpoint but no metrics is an
interrupted run, and its checkpoint is a partially-trained model at an unknown
epoch, which is exactly the kind of artifact that ends up in a table by accident.
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
                         stamp_language_dataset_versions, resolve_task,
                         TASK_LANGUAGE, ARCHITECTURES)
from more.lang_data import LANG_ROOT
from more.run_context import (load_canonical_spec, assert_not_silent_proxy,
                              ProxyGuardError, _effective)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

CODE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(CODE)
RUNS = os.path.join(REPO, "runs")
LANG_CONFIG = os.path.join(CODE, "config_language.json")
TRAIN = os.path.join(CODE, "train.py")

SPEC = load_canonical_spec(task=TASK_LANGUAGE)
GROUP = SPEC["canonical_group"]
SEEDS = list(SPEC["seed_set"])
# MoE first, then MoR, then MoRE: cheapest arm first, so a protocol mistake
# surfaces after ~0.4 h rather than ~1.2 h. The ORDER OF RUNS CANNOT AFFECT A
# RESULT -- each run is independently seeded through seeding.py's global seeding --
# so this is free.
ARMS = ["moe", "mor", "more"]


# ---------------------------------------------------------------------------
# preflight
# ---------------------------------------------------------------------------

def _corpus_dir(corpus):
    return os.path.join(LANG_ROOT, corpus)


def resolved(arch, seed):
    """The config a canonical run of (arch, seed) will actually resolve to.

    Built through cli.py's own call order so this is the real thing and not a
    parallel reimplementation: load the file, apply the architecture, apply the
    task, re-stamp the corpus version strings.
    """
    cfg = load_config(LANG_CONFIG)
    apply_architecture(cfg, arch)
    apply_task(cfg, TASK_LANGUAGE)
    stamp_language_dataset_versions(cfg)
    cfg.setdefault("provenance", {})["seed"] = seed
    cfg.setdefault("logging", {})["experiment_group"] = GROUP
    return cfg


def preflight(verbose=True):
    """Every check that can fail BEFORE a GPU-hour is spent.

    Returns `(problems, warnings)`. A PROBLEM refuses the matrix -- it will cost
    GPU-hours or invalidate the run. A WARNING is a measured fact the operator
    should know and then proceed anyway. The distinction matters: the 6 GB card
    warns (the whole calibration phase ran on it) while a 4 GB card refuses,
    because the difference is not a judgement call, it is the 5,342 MiB measured
    peak of the MoR arm.
    """
    problems, warnings = [], []

    def say(ok, what, detail=""):
        if verbose:
            print(f"  [{'ok ' if ok else 'XX '}] {what}" +
                  (f"  -- {detail}" if detail else ""))

    # 1. torch, and a real device. `torch.cuda.is_available()` returning True on a
    #    driver that then cannot launch a kernel is a real failure mode
    #    (ENVIRONMENT.md 4), so a matmul is run rather than a flag read.
    try:
        import torch
        dev_ok = torch.cuda.is_available()
        name, total = "cpu", 0
        if dev_ok:
            p = torch.cuda.get_device_properties(0)
            name, total = p.name, p.total_memory // 1048576
            (torch.randn(64, 64, device="cuda") @
             torch.randn(64, 64, device="cuda")).sum().item()
        say(dev_ok, f"CUDA device usable", f"{name}, {total} MiB"
            if dev_ok else "no CUDA device")
        if not dev_ok:
            problems.append(
                "No usable CUDA device. On Ayan's machine the interpreter must be "
                "C:/Users/Hp/anaconda3/envs/more_env/python.exe -- base and "
                "C:\\Python314 have no torch at all (ENVIRONMENT.md 8).")
        elif total < 5800:
            # A hard refusal. 5,342 MiB is the MEASURED peak of the worst arm
            # (T-L7.0, MoR at ffn_mult 24, batch 48); below ~5.8 GB total it will
            # OOM partway through, which wastes more time than refusing does.
            problems.append(
                f"This card reports only {total} MiB. The memory-heavy MoR arm peaks "
                f"at a measured 5,342 MiB at batch_size 48, so this will OOM. Do NOT "
                f"lower batch_size to fit -- it is a frozen protocol field and "
                f"changing it makes the run non-canonical. Use a larger card.")
        elif total < 7000:
            # NOT a problem: this is Vedant's 3050, on which the whole calibration
            # phase ran. 5,342 of 6,143 MiB is 87% and it does complete. Say so
            # rather than blocking, but name the two consequences.
            say(True, "VRAM headroom is thin but sufficient",
                f"worst arm peaks 5,342 of {total} MiB (87%)")
            warnings.append(
                f"{total} MiB card: the MoR arm runs at 87% of VRAM. It completes "
                f"(the entire calibration phase ran here) but (a) nothing else may "
                f"use the GPU during the matrix, and (b) the T-L7.0 timings say this "
                f"is ~1.5x slower per epoch than the 8 GB 4060. Prefer Ayan's "
                f"machine for the matrix; see ENVIRONMENT.md 8.")
    except Exception as e:
        say(False, "torch imports", f"{type(e).__name__}: {e}")
        problems.append(f"torch is not importable: {e}")

    # 2. The corpus. Not tracked in git, so this is the check a fresh clone fails.
    corpus = load_config(LANG_CONFIG)["data"]["corpus"]
    man = os.path.join(_corpus_dir(corpus), "dataset_meta.json")
    ok = os.path.exists(man)
    detail = ""
    if ok:
        m = json.load(open(man, "r", encoding="utf-8"))
        detail = (f"{m['splits']['train']:,} train blocks x {m['seq_len']}, "
                  f"V={m['vocab_size']}, {m['dataset_version']}")
    say(ok, f"corpus {corpus} built", detail or f"{man} missing")
    if not ok:
        problems.append(
            f"data/lang/{corpus}/ is not built (*.npy is not tracked). Run:\n"
            f"    python data/lang/build_language_dataset.py --corpus {corpus} "
            f"--stage all\n"
            f"  then  python data/lang/build_family_lookup.py --corpus {corpus}\n"
            f"  See SETUP.md 5. This is a real download plus tokenizer fit, "
            f"budget ~40 min.")
    else:
        for split in ("train", "val", "test"):
            arr = os.path.join(_corpus_dir(corpus), f"{split}.npy")
            if not os.path.exists(arr):
                problems.append(f"{arr} is missing though the manifest lists it.")
        fam = os.path.join(_corpus_dir(corpus), "token_family.npy")
        say(os.path.exists(fam), "POS family lookup built",
            "token_family.npy" if os.path.exists(fam) else "missing")
        if not os.path.exists(fam):
            problems.append(
                f"token_family.npy is missing. The routing metrics (AMI, purity, "
                f"Hungarian accuracy) are computed against it, so without it the "
                f"run trains but reports N/A for everything the study measures. "
                f"Run: python data/lang/build_family_lookup.py --corpus {corpus}")

    # 3. The spec is frozen. A null here refuses every canonical claim, and the
    #    refusal happens after wandb.init, so catch it now.
    nulls = sorted(k for k, v in SPEC["enforced_fields"].items() if v is None)
    nulls += sorted(f"{a}.{k}" for a, b in SPEC["architecture_variants"].items()
                    if not a.startswith("_") for k, v in b.items() if v is None)
    say(not nulls, "canonical_spec_language.json is fully frozen",
        f"nulls = {nulls}" if nulls else SPEC["spec_version"])
    if nulls:
        problems.append(
            f"The spec has unfrozen fields: {nulls}. T-L7.1 froze all of them, so "
            f"a null here means the file was edited or reverted. Do NOT fill it in "
            f"to make the matrix start -- find out what changed.")

    # 4. The real guard, on the real resolved config, for every arm. This is the
    #    check that catches a config_language.json / spec disagreement, which is
    #    the most likely thing to be wrong after any protocol edit.
    for arch in ARMS:
        try:
            got = assert_not_silent_proxy(resolved(arch, SEEDS[0]))
            ok = got == GROUP
            say(ok, f"{arch}: the frozen config is accepted as {GROUP}",
                f"group={got}")
            if not ok:
                problems.append(f"{arch} was stamped {got!r}, not {GROUP!r}.")
        except ProxyGuardError as e:
            first = [l for l in str(e).splitlines() if l.strip().startswith("-")]
            say(False, f"{arch}: proxy guard REFUSED the frozen config",
                first[0].strip() if first else "see below")
            problems.append(
                f"{arch}: the proxy guard refuses config_language.json. Either the "
                f"defaults file or the spec was edited without the other "
                f"(config_language.json's _README states the rule). Full message:\n"
                + "\n".join("      " + l for l in str(e).splitlines()))

    # 5. W&B. A canonical run dies in `wandb.init` with `UsageError: No API key
    #    configured` and leaves a directory-shaped artifact behind when it does --
    #    T-L1.4 hit exactly this. Offline is a legitimate answer; unset is not.
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
            "not the record (SETUP.md 8).")

    # 6. Disk. 15 runs x one checkpoint. Measured from an existing language
    #    checkpoint rather than guessed, because the tied embedding dominates it.
    ckpts = glob.glob(os.path.join(RUNS, "langB_*", "checkpoint.pt"))
    per = os.path.getsize(ckpts[0]) if ckpts else 23 * 1024 ** 2
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

def finished_run(arch, seed):
    """An existing COMPLETE canonical run directory for (arch, seed), or None.

    All three of arm, seed and a finished metrics.json are required. A directory
    with a checkpoint but no metrics is an INTERRUPTED run and is re-run, not
    skipped: its checkpoint is a partially trained model at an unknown epoch, and
    treating it as done is how such a thing reaches a results table.
    """
    for d in sorted(glob.glob(os.path.join(RUNS, f"langB_{arch}_seed{seed}*"))):
        mp, rc = (os.path.join(d, "metrics.json"),
                  os.path.join(d, "resolved_config.json"))
        if not (os.path.exists(mp) and os.path.exists(rc)):
            continue
        try:
            cfg = json.load(open(rc, "r", encoding="utf-8"))
            met = json.load(open(mp, "r", encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if cfg.get("logging", {}).get("experiment_group") != GROUP:
            continue
        if cfg.get("architecture") != arch:
            continue
        if int(cfg.get("provenance", {}).get("seed", -1)) != int(seed):
            continue
        if met.get("val/task_loss") is None and met.get("best_val_loss") is None:
            continue
        return d
    return None


def launch(arch, seed, group=GROUP, extra=(), dry_run=False):
    """Run one arm/seed as a child process. Returns (returncode, seconds)."""
    cmd = [sys.executable, TRAIN,
           "--config", LANG_CONFIG,
           "--task", TASK_LANGUAGE,
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
    # cwd=CODE and NOT capture_output: a 1.2 h run with a swallowed stdout is
    # unobservable, and RunContext already tees the same stream into
    # runs/<id>/stdout.log.
    proc = subprocess.run(cmd, cwd=CODE)
    return proc.returncode, time.perf_counter() - t0


def summarise(pairs):
    """Read back what actually landed. Never a number this script computed."""
    print("\n" + "=" * 78)
    print("  RESULTS AS RECORDED (read from runs/<id>/metrics.json)")
    print("=" * 78)
    floor = SPEC["protocol"]["primary_metric_floor"]
    print(f"  primary metric = val/task_loss, nats/token. "
          f"Corpus bigram floor = {floor:.4f}.")
    print(f"  {'arch':<6} {'seed':>4}  {'val_loss':>9} {'vs floor':>9} "
          f"{'depth':>6}  run")
    for arch, seed in pairs:
        d = finished_run(arch, seed)
        if d is None:
            print(f"  {arch:<6} {seed:>4}  {'MISSING':>9} {'N/A':>9} {'N/A':>6}")
            continue
        m = json.load(open(os.path.join(d, "metrics.json"), "r",
                           encoding="utf-8"))
        vl = m.get("val/task_loss")
        # `depth/mean` and NOT `depth/avg`: verified against a real language
        # metrics.json. `depth/avg` does not exist, and .get() on it would have
        # printed N/A in every row of every matrix -- a missing-key bug that looks
        # exactly like a legitimately unmeasured metric.
        dep = m.get("depth/mean")
        # N/A, never a sentinel dressed as a measurement (CLAUDE.md 4).
        vls = f"{vl:.4f}" if isinstance(vl, (int, float)) else "N/A"
        rel = f"{floor - vl:+.4f}" if isinstance(vl, (int, float)) else "N/A"
        deps = f"{dep:.2f}" if isinstance(dep, (int, float)) else "N/A"
        print(f"  {arch:<6} {seed:>4}  {vls:>9} {rel:>9} {deps:>6}  "
              f"{os.path.basename(d)}")
    print("\n  'vs floor' is positive when the run BEATS the bigram baseline.")
    print("  Do not hand-copy these into a table. Aggregate with:")
    print("      python code/seed_stats.py --group " + GROUP)
    print("      python code/export_results.py --group " + GROUP)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Run the canonical MoRE language matrix (T-L7.4).")
    ap.add_argument("--all", action="store_true",
                    help="the full 3 arms x 5 seeds matrix, skipping completed runs")
    ap.add_argument("--arch", choices=ARCHITECTURES, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--preflight", action="store_true",
                    help="run every check and exit without training")
    ap.add_argument("--smoke", action="store_true",
                    help="one short NON-canonical run on the dev corpus, to prove "
                         "the machine works. Stamped langB_smoke, never canonical.")
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
        # Refuse rather than warn. Every one of these costs GPU-hours if it is
        # discovered at hour three instead of minute zero.
        return 1
    for w in warns:
        print(f"\n  NOTE: {w}")
    print("\n  preflight clean.")
    if args.preflight:
        return 0

    if args.smoke:
        # The dev corpus, 1 epoch, an exploratory group. Proves the machine can
        # train and write a run directory without spending a canonical hour, and
        # CANNOT contaminate the matrix: wikitext-2 has its own dataset_version,
        # so the guard would refuse it as canonical even if the label were wrong.
        rc, secs = launch("more", 44, group="langB_smoke",
                          extra=["--epochs", "1", "--corpus", "wikitext-2"],
                          dry_run=args.dry_run)
        print(f"\n  smoke run {'OK' if rc == 0 else f'FAILED rc={rc}'} "
              f"in {secs / 60:.1f} min")
        return rc

    if args.all:
        pairs = [(a, s) for a in ARMS for s in SEEDS]
    elif args.arch and args.seed is not None:
        pairs = [(args.arch, args.seed)]
    elif args.arch:
        pairs = [(args.arch, s) for s in SEEDS]
    else:
        ap.error("give --all, or --arch [--seed], or --preflight, or --smoke")

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

    # Estimate from the T-L7.0 per-arm measurements, labelled as an estimate.
    HOURS = {"moe": 0.41, "mor": 1.03, "more": 1.16}   # per epoch, batch 48, 3050
    epochs = int(SPEC["enforced_fields"]["epochs"])
    est = sum(HOURS[a] for a, _ in todo) * epochs
    print(f"\n  {len(todo)} run(s) to go. Estimated {est:.1f} h from the T-L7.0 "
          f"timings, which were measured on a 6 GB 3050 -- an 8 GB 4060 should be "
          f"faster, so treat this as an upper bound.")

    t_start = time.perf_counter()
    for i, (arch, seed) in enumerate(todo, 1):
        print(f"\n  [{i}/{len(todo)}]  elapsed {(time.perf_counter() - t_start) / 3600:.2f} h")
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
