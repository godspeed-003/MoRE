"""run_lang_subset_sweep.py -- T-OD.13 TEST 2. The language-data-size sweep.

    <interpreter> code/run_lang_subset_sweep.py --preflight
    <interpreter> code/run_lang_subset_sweep.py --all
    <interpreter> code/run_lang_subset_sweep.py --all --dry-run

WHY THIS SWEEP. The language matrix found MoRE's halting saturated (96.2 % of
tokens forced to exit at max depth 7) and the depth-matched uniform control
therefore read MoRE's halting as "nearly inert" relative to MoR's (a 24x margin
ratio). The opdiff forced-depth control showed that the same composition DOES
pay when halting is live. That leaves one unexcluded explanation: the language
result may be a property of the corpus SIZE / training budget, not of the
architecture. Three training-set sizes (2.5 %, 10 %, 40 % of the 526,320
canonical train blocks), three arms, five seeds = 45 cells. If the margin ratio
moves monotonically with data size, the language number is a budget artefact; if
it sits at the 24x across all three, it is not.

WHAT THIS SCRIPT DELIBERATELY DOES NOT DO:
  * It does not claim canonical status. Every cell is launched with
    `--experiment_group subset_lang_sweep` and named `langSub_<Arch>_f<NNN>`,
    so neither the Gate 0 proxy guard nor `export_results.py` can mistake one
    for a `langB_*` canonical run. The guard will refuse a canonical claim from
    a subset run -- that refusal is the design, not an obstacle.
  * It does not change a protocol field. epochs, lr, batch, seq_len, vocab,
    tokenizer, loss weights and the val/test splits come from
    `code/config_language.json` untouched. The ONLY varying field is
    `data.subset_fraction`, which is written into both config sections and
    published as `resolved_subset_fraction`.
  * It does not retry. A crash stops the sweep (plan.md 20). Re-running `--all`
    skips finished cells and resumes from the first unfinished one -- resumable
    by SKIP, not by checkpoint resume, because a directory with a checkpoint but
    no metrics is an interrupted run at an unknown epoch.
  * It does not average. `code/summarize_lang_subset_sweep.py` owns that.

DATA PRECONDITION. `data/lang/wikitext-103/{train,val,test}.npy` are not tracked
(large binaries). They are rebuilt deterministically with
`data/lang/build_language_dataset.py --corpus wikitext-103 --stage pack`, and
`--preflight` refuses to launch unless all three exist with the shapes and
sha256 the manifest records. On this machine the rebuild reproduced
dataset_version lang-wikitext-103-bpe8192-len256-800d6154 and all three sha256s
exactly.
"""

import argparse
import glob
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from more.config import (ARCH_DISPLAY, TASK_LANGUAGE, load_config,
                         apply_architecture, apply_task,
                         stamp_language_dataset_versions)
from more.lang_data import LANG_ROOT

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

CODE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(CODE)
RUNS = os.path.join(REPO, "runs")
LANG_CONFIG = os.path.join(CODE, "config_language.json")
TRAIN = os.path.join(CODE, "train.py")
CORPUS = "wikitext-103"
MANIFEST = os.path.join(LANG_ROOT, CORPUS, "dataset_meta.json")

GROUP = "subset_lang_sweep"
SEEDS = [42, 43, 44, 45, 46]
# Cheapest first: a protocol mistake then surfaces 40x sooner than it would on
# the 40 % cells, and an interrupted night still leaves complete scales rather
# than a third of every scale.
FRACTIONS = [0.025, 0.10, 0.40]
ARMS = ["moe", "mor", "more"]
LOG_DIR = os.path.join(REPO, "results", "language", "subset_sweep_logs")
PROGRESS = os.path.join(REPO, "results", "language", "subset_sweep_progress.tsv")


def tag(frac):
    return f"f{int(round(frac * 1000)):03d}"


def run_name(arch, frac):
    return f"langSub_{ARCH_DISPLAY[arch]}_{tag(frac)}"


def cells():
    for frac in FRACTIONS:
        for arch in ARMS:
            for seed in SEEDS:
                yield arch, frac, seed


# ---------------------------------------------------------------------------
# preflight
# ---------------------------------------------------------------------------

def _say(ok, what, detail=""):
    print(f"  [{'ok  ' if ok else 'FAIL'}] {what}"
          + (f"  -- {detail}" if detail else ""))
    return ok


def preflight():
    problems = []
    print("\n=== preflight ===")

    # 1. The arrays exist, have the recorded shapes, and hash to the recorded
    #    value. A rebuild that silently tokenized a different corpus would still
    #    produce plausible arrays and a plausible loss curve.
    if not os.path.exists(MANIFEST):
        problems.append(f"no manifest at {MANIFEST}")
        man = {}
    else:
        with open(MANIFEST, "r", encoding="utf-8") as f:
            man = json.load(f)
    import hashlib
    for split in ("train", "val", "test"):
        p = os.path.join(LANG_ROOT, CORPUS, f"{split}.npy")
        if not os.path.exists(p):
            problems.append(f"missing {p} -- rebuild with build_language_dataset.py "
                            f"--stage pack")
            continue
        import numpy as np
        a = np.load(p, mmap_mode="r")
        want_blocks = man.get("splits", {}).get(split)
        ok = (a.shape == (want_blocks, man.get("seq_len"))
              and a.dtype == np.uint16)
        _say(ok, f"{split}.npy shape dtype", f"{a.shape} {a.dtype}, "
             f"want ({want_blocks}, {man.get('seq_len')})")
        if not ok:
            problems.append(f"{split}.npy has shape/dtype {a.shape}/{a.dtype}, "
                            f"manifest says ({want_blocks}, "
                            f"{man.get('seq_len')}) uint16")
        if man.get("sha256", {}).get(split):
            h = hashlib.sha256()
            with open(p, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 22), b""):
                    h.update(chunk)
            ok = h.hexdigest() == man["sha256"][split]
            _say(ok, f"{split}.npy sha256", h.hexdigest()[:12])
            if not ok:
                problems.append(f"{split}.npy sha256 {h.hexdigest()[:12]} != "
                                f"manifest {man['sha256'][split][:12]} -- the "
                                "corpus on disk is NOT the one the canonical "
                                "matrix was measured on")

    # 2. Nesting, at the REAL block count, for every seed. This is the property
    #    that makes the three scales comparable; test_depth_control.py proves it
    #    at n=20000, this proves it on the data that will actually be read.
    if man.get("splits", {}).get("train"):
        from torch.utils.data import random_split
        n = int(man["splits"]["train"])
        import torch
        for seed in SEEDS:
            per = {}
            for frac in FRACTIONS + [1.0]:
                k = max(1, int(n * frac))
                ds = torch.arange(n)
                sub, _ = random_split(
                    ds, [k, n - k],
                    generator=torch.Generator().manual_seed(seed))
                per[frac] = sub.indices if hasattr(sub, "indices") else list(sub)
            ok = (per[0.025] == per[0.10][:len(per[0.025])]
                  and per[0.10] == per[0.40][:len(per[0.10])])
            _say(ok, f"nesting seed{seed} at n={n}",
                 f"{len(per[0.025])}/{len(per[0.10])}/{len(per[0.40])} blocks")
            if not ok:
                problems.append(f"subsets are not nested for seed {seed}")

    # 3. W&B settled, or the first cell dies in wandb.init after the guard.
    mode = os.environ.get("WANDB_MODE", "")
    ok = mode in ("offline", "disabled")
    _say(ok, "WANDB_MODE offline/disabled", mode or "unset")
    if not ok:
        problems.append("set WANDB_MODE=offline (metrics land in runs/ either way)")

    # 4. Disk. 45 checkpoints, ~22 MiB each; the 40 % cells are the same model.
    import shutil
    need = 45 * 23 * 1024 ** 2
    free = shutil.disk_usage(REPO).free
    ok = free > need * 2
    _say(ok, "disk headroom", f"need ~{need / 1024**3:.1f} GiB, "
                              f"free {free / 1024**3:.1f} GiB")
    if not ok:
        problems.append("not enough free disk for 45 checkpoints")

    # 5. The three names must NOT match the canonical glob.
    globs = glob.glob(os.path.join(RUNS, run_name(ARMS[0], FRACTIONS[0]) + "*"))
    ok = not globs
    _say(ok, "langSub_* does not collide with existing dirs",
         f"{len(globs)} pre-existing")

    return problems


# ---------------------------------------------------------------------------
# launch / resume
# ---------------------------------------------------------------------------

def resolved(arch, frac, seed):
    cfg = load_config(LANG_CONFIG)
    apply_architecture(cfg, arch)
    apply_task(cfg, TASK_LANGUAGE)
    stamp_language_dataset_versions(cfg)
    cfg["data"]["subset_fraction"] = frac
    cfg["training"]["subset_fraction"] = frac
    cfg["data"]["subset_seed"] = seed
    cfg["training"]["subset_seed"] = seed
    cfg.setdefault("provenance", {})["seed"] = seed
    cfg.setdefault("logging", {})["experiment_group"] = GROUP
    return cfg


def finished_run(arch, frac, seed):
    """A COMPLETE cell for (arch, frac, seed), or None.

    Requires metrics.json, a resolved config that agrees on all four identity
    fields, and a real val loss. A directory with a checkpoint but no metrics is
    an interrupted run and is re-run, never trusted.
    """
    pat = os.path.join(RUNS, run_name(arch, frac) + f"_seed{seed}*")
    for d in sorted(glob.glob(pat)):
        mp = os.path.join(d, "metrics.json")
        rc = os.path.join(d, "resolved_config.json")
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
        got = cfg.get("data", {}).get("subset_fraction")
        if got is None or abs(float(got) - frac) > 1e-12:
            continue
        if met.get("val/task_loss") is None:
            continue
        return d
    return None


def launch(arch, frac, seed, dry_run=False):
    name = run_name(arch, frac)
    cmd = [sys.executable, TRAIN,
           "--config", LANG_CONFIG,
           "--task", TASK_LANGUAGE,
           "--architecture", arch,
           "--seed", str(seed),
           "--subset_fraction", repr(frac),
           "--run_name", name,
           "--experiment_group", GROUP]
    print("\n" + "=" * 78)
    print(f"  {ARCH_DISPLAY[arch]}  f={frac:.3f}  seed {seed}")
    print("  " + " ".join(cmd[1:]))
    print("=" * 78, flush=True)
    if dry_run:
        return 0, 0.0
    os.makedirs(LOG_DIR, exist_ok=True)
    log = os.path.join(LOG_DIR, f"{name}_seed{seed}.log")
    t0 = time.perf_counter()
    with open(log, "w", encoding="utf-8", errors="replace") as fh:
        proc = subprocess.run(cmd, cwd=CODE, stdout=fh, stderr=subprocess.STDOUT)
    dt = time.perf_counter() - t0
    if proc.returncode == 0:
        d = finished_run(arch, frac, seed)
        if d is None:
            print("  [FAIL] child exited 0 but no complete run directory exists")
            return 1, dt
        met = json.load(open(os.path.join(d, "metrics.json"), encoding="utf-8"))
        vl = met.get("val/task_loss")
        dep = met.get("depth/mean")
        os.makedirs(os.path.dirname(PROGRESS), exist_ok=True)
        new = not os.path.exists(PROGRESS)
        with open(PROGRESS, "a", encoding="utf-8") as fh:
            if new:
                fh.write("arch\tfrac\tseed\tval_task_loss\tdepth_mean\t"
                         "seconds\trun_dir\n")
            fh.write(f"{arch}\t{frac:.3f}\t{seed}\t"
                     f"{vl if vl is not None else 'N/A'}\t"
                     f"{dep if dep is not None else 'N/A'}\t{dt:.0f}\t"
                     f"{os.path.basename(d)}\n")
        print(f"  val/task_loss = {vl if vl is not None else 'N/A'}   "
              f"depth = {dep if dep is not None else 'N/A'}   "
              f"{dt / 3600:.2f} h   {os.path.basename(d)}", flush=True)
    else:
        print(f"  [FAIL] exit {proc.returncode}; log = {log}", flush=True)
    return proc.returncode, dt


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--preflight", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--arch", choices=ARMS, default=None)
    ap.add_argument("--frac", type=float, default=None)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args(argv)

    if not (args.all or args.preflight or args.arch or args.frac or args.seed):
        ap.error("choose --preflight or --all (or one explicit cell)")

    problems = preflight()
    if problems:
        print("\nREFUSING to launch:")
        for p in problems:
            print("  -", p)
        return 2
    if args.preflight and not (args.all or args.arch or args.frac):
        print("\npreflight passed")
        return 0

    todo = list(cells())
    if args.arch:
        todo = [c for c in todo if c[0] == args.arch]
    if args.frac:
        todo = [c for c in todo if abs(c[1] - args.frac) < 1e-12]
    if args.seed:
        todo = [c for c in todo if c[2] == args.seed]

    print(f"\n{len(todo)} cells queued, "
          f"{'dry run' if args.dry_run else 'launching'}")
    started = time.perf_counter()
    done = skipped = 0
    for arch, frac, seed in todo:
        d = finished_run(arch, frac, seed)
        if d is not None:
            skipped += 1
            print(f"  [skip] {os.path.basename(d)} already complete")
            continue
        rc, _ = launch(arch, frac, seed, dry_run=args.dry_run)
        if rc != 0:
            print(f"\nSTOPPED at {ARCH_DISPLAY[arch]} f={frac} seed{seed} "
                  f"(exit {rc}). Nothing was retried; re-run --all to resume.")
            return rc
        done += 1

    print(f"\nfinished: {done} new, {skipped} skipped, "
          f"{(time.perf_counter() - started) / 3600:.2f} h wall")
    if not args.dry_run and done:
        print(f"progress -> {PROGRESS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
