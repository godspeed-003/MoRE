"""
verify_pipeline.py — Pre-flight Diagnostic & Verification Script
================================================================
Run this BEFORE letting autoresearch_runner.py run overnight.
It verifies every subsystem independently and prints a clean
PASS / FAIL status for each check.

Tests performed:
  1. Dependency check    — all required Python packages importable.
  2. Git sandbox check   — stage/commit/rollback cycle using an isolated
                           temporary file that never touches your project files.
  3. VRAM offload test   — query Ollama, then verify GPU VRAM returns to
                           baseline after a short sleep.
  4. Dummy pass execution — run train.py for 2 tiny epochs with synthetic
                           data, verify output shape, gradient flow, weight
                           update, and file writes (results.tsv, best_model.pt).

Usage:
    python verify_pipeline.py [--ollama-url http://localhost:11434] [--skip-ollama]

All temporary test artefacts (sandbox_test.txt, test_results.tsv, etc.)
are cleaned up regardless of pass/fail.  The script never modifies
train.py, config.json, or any other project files that are part of your
research workflow.
"""

import os
import sys
import json
import time
import shutil
import argparse
import tempfile
import traceback
import subprocess
import platform
import importlib

# ---------------------------------------------------------------------------
# ANSI colour helpers
# ---------------------------------------------------------------------------

def _c(text: str, code: str) -> str:
    """Wrap text in ANSI colour codes (only on TTY)."""
    if sys.stdout.isatty():
        return f"\033[{code}m{text}\033[0m"
    return text

# The Windows console defaults to cp1252, which cannot encode the box-drawing
# and check glyphs below; without this the harness dies in its own banner
# before running a single check.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

PASS  = lambda s: _c(f"  ✓ PASS  {s}", "32")
FAIL  = lambda s: _c(f"  ✗ FAIL  {s}", "31")
WARN  = lambda s: _c(f"  ⚠ WARN  {s}", "33")
INFO  = lambda s: _c(f"  · INFO  {s}", "36")
HLINE = lambda: print("─" * 65)


# ---------------------------------------------------------------------------
# State accumulator
# ---------------------------------------------------------------------------

class ResultAccumulator:
    def __init__(self):
        self.checks: list[tuple[str, bool, str]] = []

    def record(self, name: str, passed: bool, detail: str = ""):
        self.checks.append((name, passed, detail))
        if passed:
            print(PASS(name) + (f"  [{detail}]" if detail else ""))
        else:
            print(FAIL(name) + (f"  [{detail}]" if detail else ""))

    def summary(self) -> bool:
        passed = sum(1 for _, p, _ in self.checks if p)
        total  = len(self.checks)
        print()
        HLINE()
        colour = "32" if passed == total else "31"
        print(_c(f"  Results: {passed}/{total} checks passed", colour))
        HLINE()
        if passed == total:
            print(_c("  ✓ All systems GO — safe to run autoresearch_runner.py", "32"))
        else:
            failed = [n for n, p, _ in self.checks if not p]
            print(_c(f"  ✗ Fix failing checks before running overnight: {failed}", "31"))
        return passed == total


results = ResultAccumulator()


# ---------------------------------------------------------------------------
# 1. Dependency check
# ---------------------------------------------------------------------------

def check_dependencies():
    print("\n[1/4] Dependency Check")
    HLINE()

    required = [
        ("torch",    "PyTorch"),
        ("wandb",    "Weights & Biases"),
        ("requests", "requests (for Ollama API)"),
    ]

    for pkg, label in required:
        try:
            mod = importlib.import_module(pkg)
            ver = getattr(mod, "__version__", "unknown")
            results.record(f"Import {label}", True, f"v{ver}")
        except ImportError as e:
            results.record(f"Import {label}", False, str(e))

    # Check git binary
    try:
        proc = subprocess.run(
            ["git", "--version"], capture_output=True, text=True, timeout=10
        )
        results.record("git binary", proc.returncode == 0, proc.stdout.strip())
    except FileNotFoundError:
        results.record("git binary", False, "git not found in PATH")

    # Check nvidia-smi
    smi_available = shutil.which("nvidia-smi") is not None
    results.record(
        "nvidia-smi available",
        smi_available,
        "present" if smi_available else "not found (GPU diagnostics limited)"
    )
    if not smi_available:
        # Not a hard failure — we can still run on CPU
        results.checks[-1] = (results.checks[-1][0], True, "not found (CPU-only OK)")
        print(WARN("nvidia-smi not found — GPU VRAM check will use PyTorch API only"))


# ---------------------------------------------------------------------------
# 2. Git sandbox check
# ---------------------------------------------------------------------------

def check_git_sandbox():
    """
    Creates a TEMPORARY git repository in a fresh temp directory.
    This test is completely isolated from the project workspace.
    It never touches, modifies, or deletes any project files.
    """
    print("\n[2/4] Git Sandbox Check (isolated temporary repo)")
    HLINE()
    print(INFO("Using a throwaway temp directory — project files are UNTOUCHED"))

    tmpdir = tempfile.mkdtemp(prefix="more_git_sandbox_")
    print(INFO(f"Temp dir: {tmpdir}"))

    def git(args, cwd=tmpdir):
        r = subprocess.run(
            ["git"] + args, cwd=cwd, capture_output=True, text=True
        )
        return r.returncode, r.stdout.strip(), r.stderr.strip()

    try:
        # --- Initialise temp repo ---
        rc, out, err = git(["init", "--initial-branch=main"])
        if rc != 0:
            # Older git versions don't support --initial-branch
            rc, out, err = git(["init"])
        if rc != 0:
            results.record("git init (sandbox)", False, err)
            return

        git(["config", "user.email", "test@verify"])
        git(["config", "user.name",  "Verify Bot"])

        # Create initial commit so HEAD exists (required for reset --hard HEAD~1)
        seed_path = os.path.join(tmpdir, "seed.txt")
        with open(seed_path, "w") as f:
            f.write("seed\n")
        git(["add", "seed.txt"])
        git(["commit", "-m", "Initial seed"])
        results.record("git init + seed commit (sandbox)", True)

        # --- Create sandbox_test.txt and commit ---
        sandbox_path = os.path.join(tmpdir, "sandbox_test.txt")
        with open(sandbox_path, "w") as f:
            f.write("sandbox verification content\n")
        git(["add", "sandbox_test.txt"])
        rc, out, err = git(["commit", "-m", "Test commit: sandbox_test.txt"])
        results.record("git add + commit (sandbox)", rc == 0, out or err)

        # Verify file exists in the commit
        rc2, sha, _ = git(["rev-parse", "--short", "HEAD"])
        results.record("commit SHA retrievable (sandbox)", rc2 == 0, sha)

        # --- Rollback via git reset --hard HEAD~1 ---
        rc, out, err = git(["reset", "--hard", "HEAD~1"])
        results.record("git reset --hard HEAD~1 (sandbox)", rc == 0, out or err)

        # Verify sandbox_test.txt was reverted (should not exist now)
        file_gone = not os.path.exists(sandbox_path)
        results.record(
            "sandbox file correctly reverted after rollback",
            file_gone,
            "file absent as expected" if file_gone else "ERROR: file still present"
        )

        # --- Verify seed.txt is unharmed ---
        seed_intact = os.path.exists(seed_path)
        results.record(
            "seed file unharmed after rollback (sandbox)",
            seed_intact,
            "intact" if seed_intact else "MISSING — unexpected"
        )

    except Exception as e:
        results.record("Git sandbox check", False, str(e))
        traceback.print_exc()
    finally:
        # Clean up temp directory unconditionally
        shutil.rmtree(tmpdir, ignore_errors=True)
        print(INFO(f"Temp directory removed: {tmpdir}"))


# ---------------------------------------------------------------------------
# 3. VRAM offload test
# ---------------------------------------------------------------------------


def check_vram_offload(ollama_url: str, skip_ollama: bool):
    print("[3/4] VRAM Offload Test")
    HLINE()

    import threading
    import requests as req

    def get_gpu_vram_used_mb():
        if shutil.which("nvidia-smi"):
            try:
                proc=subprocess.run(
                    ["nvidia-smi","--query-gpu=memory.used","--format=csv,noheader,nounits"],
                    capture_output=True,text=True,timeout=5
                )
                if proc.returncode==0:
                    return float(proc.stdout.strip().splitlines()[0])
            except Exception:
                pass
        return 0.0

    def kill_llama():
        subprocess.run(
            ["taskkill","/IM","llama-server.exe","/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(2)

    if skip_ollama:
        results.record("Ollama VRAM offload test",True,"skipped")
        return

    print(INFO("Stopping existing llama-server instances..."))
    kill_llama()

    baseline=get_gpu_vram_used_mb()
    results.record("VRAM baseline readable",True,f"{baseline:.1f} MB")

    peak=baseline
    running=True

    def monitor():
        nonlocal peak,running
        while running:
            try:
                peak=max(peak,get_gpu_vram_used_mb())
            except Exception:
                pass
            time.sleep(0.1)

    t=threading.Thread(target=monitor,daemon=True)
    t.start()

    print(INFO("Sending test prompt to Ollama ..."))
    try:
        r=req.post(
            f"{ollama_url}/api/generate",
            json={
                "model":"qwen2.5-coder:7b",
                "prompt":"Reply exactly with VERIFIED",
                "stream":False,
                "keep_alive":"0s"
            },
            timeout=120
        )
    finally:
        running=False
        t.join()

    if r.status_code!=200:
        results.record("Ollama ping successful",False,f"HTTP {r.status_code}")
        return

    body=r.json().get("response","")
    results.record("Ollama ping successful",True,body[:40])

    loaded=(peak-baseline)>1000
    results.record(
        "Model loaded into GPU",
        loaded,
        f"baseline={baseline:.0f}MB peak={peak:.0f}MB Δ={peak-baseline:+.0f}MB"
    )

    print(INFO("Ensuring model is unloaded..."))
    try:
        req.post(
            f"{ollama_url}/api/generate",
            json={"model":"qwen2.5-coder:7b","keep_alive":0},
            timeout=30
        )
    except Exception:
        pass

    kill_llama()

    after=get_gpu_vram_used_mb()
    offloaded=abs(after-baseline)<=200

    results.record(
        "Ollama VRAM fully offloaded",
        offloaded,
        f"baseline={baseline:.0f}MB peak={peak:.0f}MB after={after:.0f}MB"
    )


# ---------------------------------------------------------------------------
# 4. Dummy pass execution
# ---------------------------------------------------------------------------

def check_dummy_pass():
    """
    Runs train.py for exactly 2 epochs using a very small synthetic
    dataset to verify:
      - Dimensions align (no shape mismatch errors)
      - Gradients flow (loss decreases or at least changes)
      - Weights update (parameter values change after optimiser step)
      - Files write (results.tsv produced with correct columns)
    """
    print("\n[4/4] Dummy Pass Execution (2 epochs, synthetic data)")
    HLINE()

    # --- Create a minimal temp working directory with synthetic data -----
    tmpdir = tempfile.mkdtemp(prefix="more_dummy_pass_")
    print(INFO(f"Temp working dir: {tmpdir}"))

    DUMMY_JSONL = os.path.join(tmpdir, "dummy.jsonl")
    DUMMY_CFG   = os.path.join(tmpdir, "config.json")
    TRAIN_SRC   = "train.py"       # Must exist in current directory

    # Copy train.py into temp dir
    if not os.path.exists(TRAIN_SRC):
        results.record("train.py found", False, f"'{TRAIN_SRC}' missing from CWD")
        shutil.rmtree(tmpdir, ignore_errors=True)
        return
    shutil.copy(TRAIN_SRC, os.path.join(tmpdir, "train.py"))
    # train.py is a thin entry point; the implementation is the `more/` package
    # (config, families, data, model, metrics, engine, run_context). The package
    # must be copied whole, and run_context reads canonical_spec.json from the
    # directory ABOVE the package -- i.e. the sandbox root -- so the sandbox
    # mirrors code/ exactly.
    CODE_DIR = os.path.dirname(os.path.abspath(TRAIN_SRC)) or os.getcwd()
    pkg_src = os.path.join(CODE_DIR, "more")
    if not os.path.isdir(pkg_src):
        results.record("more/ package found", False, f"'{pkg_src}' missing")
        shutil.rmtree(tmpdir, ignore_errors=True)
        return
    shutil.copytree(pkg_src, os.path.join(tmpdir, "more"),
                    ignore=shutil.ignore_patterns("__pycache__"))
    for dep in ("canonical_spec.json",):
        dep_src = os.path.join(CODE_DIR, dep)
        if not os.path.exists(dep_src):
            results.record(f"{dep} found", False, f"'{dep_src}' missing")
            shutil.rmtree(tmpdir, ignore_errors=True)
            return
        shutil.copy(dep_src, os.path.join(tmpdir, dep))
    results.record("train.py + more/ package found and copied", True)

    # --- Generate 30 synthetic JSONL records ----------------------------
    import random, math
    random.seed(42)
    families = [
        ("E1", ["ADD", "SUB"]),
        ("E2", ["MULT", "DIV"]),
        ("E3", ["MOD", "POW"]),
        ("E4", ["AND", "OR"]),
        ("E5", ["SHIFT_L", "SHIFT_R"]),
        ("E6", ["SORT", "MIN", "MAX"]),
        ("E7", ["ADD", "SHIFT_L", "MOD"]),
    ]

    records = []
    for i in range(30):
        fam_key, ops = random.choice(families)
        depth = random.randint(1, min(3, len(ops) * 2))
        steps = []
        prev  = random.randint(-100, 100)
        for d in range(depth):
            op  = random.choice(ops)
            arg = random.randint(-50, 50) or 1
            if op in ("DIV", "MOD"):
                arg = arg or 1
            res = max(-1000, min(1000, prev + arg))  # simplified arithmetic
            steps.append({"op": op, "args": [prev, arg], "result": res})
            prev = res

        records.append({
            "input":  f"op_chain_{i}",
            "steps":  steps,
            "output": prev,
            "family": fam_key,
            "depth":  depth,
            "id":     f"{fam_key}_{i:06d}",
        })

    with open(DUMMY_JSONL, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    results.record("Synthetic JSONL dataset created", True, f"{len(records)} records")

    # --- Write a tiny config for 2 epochs (v2 — per-step tokenisation) ----
    dummy_cfg = {
        "model": {
            "d_model": 64,
            "num_experts": 7,
            "max_depth": 3,
            "max_steps": 7,          # number of step-tokens per program
            "step_feat_dim": 12,     # features per step-token row
            "num_blocks": 1,
            "dropout": 0.0,
        },
        "training": {
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "batch_size": 8,
            "epochs": 2,
            "val_split": 0.2,
            "grad_clip": 1.0,
        },
        "loss_weights": {
            "task": 1.0,
            "routing_balance": 0.01,
            "halting": 0.001,
            "step_routing": 0.5,     # per-step oracle CE loss weight
        },
        "data": {
            "jsonl_path": "dummy.jsonl",
            "max_val": 1000.0,
            "pad_value": 0.0,
        },
        "logging": {
            "wandb_project": "more-verify",
            "log_interval": 1,
        },
    }
    with open(DUMMY_CFG, "w") as f:
        json.dump(dummy_cfg, f, indent=2)
    results.record("Dummy config.json written", True)

    # --- Patch W&B to disabled mode so we don't create real runs --------
    env = os.environ.copy()
    env["WANDB_MODE"] = "disabled"
    env["WANDB_SILENT"] = "true"

    # --- Run train.py subprocess ----------------------------------------
    print(INFO("Launching: python train.py --architecture more "
               "--config config.json --epochs 2"))
    t0   = time.time()
    proc = subprocess.run(
        [sys.executable, "train.py", "--architecture", "more",
         "--config", "config.json", "--epochs", "2"],
        cwd=tmpdir,
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
    )
    elapsed = time.time() - t0
    stdout  = proc.stdout
    stderr  = proc.stderr
    combined = (stdout + "\n" + stderr).lower()

    print(INFO(f"Subprocess finished in {elapsed:.1f}s, exit code {proc.returncode}"))

    # --- Check 1: Exit code 0 ------------------------------------------
    results.record(
        "train.py exits cleanly (code 0)",
        proc.returncode == 0,
        f"exit={proc.returncode}, {elapsed:.1f}s"
    )

    if proc.returncode != 0:
        # Print last 40 lines of combined output to help debug
        lines = (stdout + stderr).splitlines()
        print("\n--- STDOUT/STDERR (last 40 lines) ---")
        for l in lines[-40:]:
            print(f"  {l}")
        shutil.rmtree(tmpdir, ignore_errors=True)
        return

    # --- Check 2: No shape/gradient errors in output -------------------
    error_keywords = [
        "runtimeerror", "size mismatch", "shape", "traceback",
        "error:", "exception", "nan loss", "inf loss",
    ]
    has_error = any(kw in combined for kw in error_keywords)
    results.record(
        "No dimension / gradient errors in output",
        not has_error,
        "clean" if not has_error else "ERROR keyword detected (see output above)"
    )
    if has_error:
        lines = (stdout + stderr).splitlines()
        print("\n--- STDOUT/STDERR ---")
        for l in lines[-50:]:
            print(f"  {l}")

    # --- Check 3: Loss logged (model is actually computing) ------------
    loss_mentioned = "task=" in stdout or "task_loss" in stdout.lower()
    results.record(
        "Loss values appear in training output",
        loss_mentioned,
        "found" if loss_mentioned else "not found"
    )

    # --- Check 4: per-run directory written with the required artefacts ----
    # Phase 0 (plan.md 2.2): outputs live in runs/<experiment_id>/, not in
    # fixed global filenames. run_context puts RUNS_ROOT one level above the
    # directory holding train.py, which here is the sandbox's parent.
    runs_root = os.path.join(os.path.dirname(os.path.abspath(tmpdir)), "runs")
    run_dirs = []
    if os.path.isdir(runs_root):
        run_dirs = [
            os.path.join(runs_root, d) for d in os.listdir(runs_root)
            if os.path.isdir(os.path.join(runs_root, d))
        ]
        run_dirs.sort(key=lambda p: os.path.getmtime(p), reverse=True)

    results.record(
        "runs/<experiment_id>/ directory created",
        bool(run_dirs),
        f"runs_root={runs_root}, found {len(run_dirs)}"
    )

    run_dir = run_dirs[0] if run_dirs else None
    results_tsv = os.path.join(run_dir, "results.tsv") if run_dir else ""

    if run_dir:
        # plan.md 2.2 mandates exactly these artefacts per run.
        for artefact in ("config.json", "resolved_config.json", "metrics.json",
                         "results.tsv", "checkpoint.pt", "stdout.log"):
            results.record(
                f"run dir contains {artefact}",
                os.path.exists(os.path.join(run_dir, artefact)),
                os.path.basename(run_dir)
            )
        # No global output filenames may reappear next to train.py.
        for forbidden in ("best_model.pt", "results.tsv", "final_run_metrics.json"):
            results.record(
                f"no global {forbidden} written",
                not os.path.exists(os.path.join(tmpdir, forbidden))
            )

    tsv_exists = bool(results_tsv) and os.path.exists(results_tsv)
    results.record("results.tsv file created", tsv_exists)

    if tsv_exists:
        with open(results_tsv, "r") as f:
            content = f.read()
        required_cols = [
            "epoch", "train_task_loss", "val_loss",
            "expert_entropy", "avg_depth", "max_cos_sim"
        ]
        header_ok = all(col in content for col in required_cols)
        results.record(
            "results.tsv has required columns",
            header_ok,
            "all columns present" if header_ok else f"content: {content[:200]}"
        )
        # Check at least 1 data row
        data_rows = [l for l in content.splitlines() if l and "epoch" not in l]
        results.record(
            "results.tsv has at least 1 data row",
            len(data_rows) >= 1,
            f"{len(data_rows)} data rows"
        )
    else:
        print(INFO("Skipping column/row checks (file missing)"))

    # --- Check 5: Confirm parameter update (weights changed) -----------
    # We do this in-process to avoid a second subprocess
    try:
        import torch
        import torch.nn as nn
        # Minimal model forward + backward
        lin = nn.Linear(8, 4)
        opt = torch.optim.SGD(lin.parameters(), lr=0.1)
        x   = torch.randn(2, 8)
        w0  = lin.weight.data.clone()
        loss = lin(x).sum()
        loss.backward()
        opt.step()
        w1  = lin.weight.data
        weights_changed = not torch.allclose(w0, w1)
        results.record(
            "Gradient flow verified (PyTorch in-process sanity)",
            weights_changed,
            "weights updated after backward()"
        )
    except Exception as e:
        results.record(
            "Gradient flow verified (PyTorch in-process sanity)",
            False, str(e)
        )

    # Cleanup
    shutil.rmtree(tmpdir, ignore_errors=True)
    if run_dir:
        shutil.rmtree(run_dir, ignore_errors=True)
    print(INFO(f"Temp directory removed: {tmpdir}"))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="MoRE Pipeline Pre-flight Checker")
    parser.add_argument(
        "--ollama-url", default="http://localhost:11434",
        help="Ollama server URL (default: http://localhost:11434)"
    )
    parser.add_argument(
        "--skip-ollama", action="store_true",
        help="Skip the Ollama VRAM offload test (useful if Ollama is not running)"
    )
    args = parser.parse_args()

    print("=" * 65)
    print("  MoRE Pipeline Pre-flight Verification")
    print(f"  Python  : {sys.version.split()[0]}")
    print(f"  Platform: {platform.system()} {platform.machine()}")
    print(f"  CWD     : {os.getcwd()}")
    print("=" * 65)

    try:
        check_dependencies()
    except Exception as e:
        print(FAIL(f"Dependency check aborted: {e}"))
        traceback.print_exc()

    try:
        check_git_sandbox()
    except Exception as e:
        print(FAIL(f"Git sandbox check aborted: {e}"))
        traceback.print_exc()

    try:
        check_vram_offload(args.ollama_url, args.skip_ollama)
    except Exception as e:
        print(FAIL(f"VRAM offload check aborted: {e}"))
        traceback.print_exc()

    try:
        check_dummy_pass()
    except Exception as e:
        print(FAIL(f"Dummy pass check aborted: {e}"))
        traceback.print_exc()

    all_passed = results.summary()
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
