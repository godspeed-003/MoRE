"""
autoresearch_runner.py — CPU-bound Automated Research Supervisor
================================================================
This script orchestrates an overnight automated research loop.
It NEVER loads Ollama and PyTorch into VRAM simultaneously.

Loop per iteration:
  1. Query Ollama (Qwen2.5-Coder:7b) for a config / code tweak.
  2. Apply the tweak to config.json or train.py.
  3. git commit the change.
  4. Spawn `python train.py` as a subprocess (max 5 minutes).
  5. Read results.tsv to evaluate the validation loss.
  6. If improved  → keep the commit, update the baseline.
     If worse/crash → `git reset --hard HEAD~1` to restore baseline.

Zero-Collision VRAM Strategy
-----------------------------
  • Ollama must be started externally with OLLAMA_KEEP_ALIVE=0m so it
    auto-unloads after each generation call.
  • This script inserts a configurable sleep (default 10 s) after the
    Ollama API call returns, giving the GPU driver time to fully
    reclaim VRAM before we launch PyTorch.
  • The subprocess poll loop watches for OOM errors in stdout/stderr
    and aborts the run early, triggering a rollback.

Usage:
    # Start Ollama in a separate terminal FIRST:
    #   OLLAMA_KEEP_ALIVE=0m ollama serve
    #
    # Then run:
    python autoresearch_runner.py [--iterations N] [--duration-minutes M]
"""

import os
import sys
import json
import time
import math
import shutil
import argparse
import datetime
import subprocess
import traceback
import csv
import re
import textwrap

import requests   # pip install requests


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OLLAMA_URL    = "http://localhost:11434/api/generate"
OLLAMA_MODEL  = "qwen2.5-coder:7b"
TRAIN_SCRIPT  = "train.py"
CONFIG_FILE   = "config.json"
RESULTS_FILE  = "results.tsv"

# How long to let each train.py subprocess run (seconds)
TRAIN_TIMEOUT_SECS = 300   # 5 minutes

# Buffer sleep after Ollama finishes (let VRAM drain)
VRAM_DRAIN_SLEEP_SECS = 10

# Maximum consecutive failed (crash/OOM) runs before we abort
MAX_CONSECUTIVE_FAILS = 3


# ---------------------------------------------------------------------------
# Helper: Git utilities
# ---------------------------------------------------------------------------

def _run_git(args: list[str], cwd: str = ".") -> tuple[int, str, str]:
    """Run a git command, return (returncode, stdout, stderr)."""
    result = subprocess.run(
        ["git"] + args,
        cwd=cwd, capture_output=True, text=True
    )
    return result.returncode, result.stdout.strip(), result.stderr.strip()


def git_stage_and_commit(message: str) -> bool:
    """Stage all modified tracked files and create a commit."""
    rc, out, err = _run_git(["add", CONFIG_FILE, TRAIN_SCRIPT])
    if rc != 0:
        print(f"[Git] WARNING: git add failed: {err}")

    rc, out, err = _run_git(["commit", "-m", message])
    if rc != 0:
        # Nothing to commit is okay (no changes were actually made)
        if "nothing to commit" in out.lower() or "nothing to commit" in err.lower():
            print("[Git] Nothing to commit — skipping.")
            return False
        print(f"[Git] ERROR: commit failed: {err}")
        return False

    rc2, sha, _ = _run_git(["rev-parse", "--short", "HEAD"])
    print(f"[Git] Committed: {sha}  '{message}'")
    return True


def git_rollback() -> bool:
    """Hard-reset to the previous commit, discarding the last change."""
    rc, out, err = _run_git(["reset", "--hard", "HEAD~1"])
    if rc != 0:
        print(f"[Git] ERROR: rollback failed: {err}")
        return False
    rc2, sha, _ = _run_git(["rev-parse", "--short", "HEAD"])
    print(f"[Git] Rolled back to: {sha}")
    return True


def git_current_sha() -> str:
    _, sha, _ = _run_git(["rev-parse", "--short", "HEAD"])
    return sha


# ---------------------------------------------------------------------------
# Helper: Ollama query
# ---------------------------------------------------------------------------

def query_ollama(prompt: str, timeout: int = 120) -> str:
    """
    Send a prompt to the local Ollama server and return the response text.
    Ollama must be running with OLLAMA_KEEP_ALIVE=0m so it unloads after
    this call completes.
    """
    payload = {
        "model":  OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.3,
            "num_predict": 512,
        },
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        return data.get("response", "").strip()
    except requests.exceptions.ConnectionError:
        print("[Ollama] ERROR: Cannot connect to Ollama. Is 'ollama serve' running?")
        return ""
    except requests.exceptions.Timeout:
        print("[Ollama] ERROR: Request timed out.")
        return ""
    except Exception as e:
        print(f"[Ollama] ERROR: {e}")
        return ""


# ---------------------------------------------------------------------------
# Helper: Parse Ollama response into actionable tweaks
# ---------------------------------------------------------------------------

def parse_llm_tweak(response: str, current_cfg: dict) -> dict | None:
    """
    Try to extract a JSON block from the LLM response.  We ask the model
    to respond with a JSON object containing optional keys:
        config_patch  : dict of dot-path → new value for config.json
        commit_message: string

    Returns the parsed dict, or None if parsing failed.

    Example LLM output:
      {
        "config_patch": {"training.lr": 5e-4, "loss_weights.routing_balance": 0.05},
        "commit_message": "Lower LR and increase routing penalty"
      }
    """
    # Extract the first JSON block from the response
    json_pattern = re.compile(r'\{[^{}]*\}', re.DOTALL)
    matches = json_pattern.findall(response)

    for match in matches:
        try:
            parsed = json.loads(match)
            if "config_patch" in parsed or "commit_message" in parsed:
                return parsed
        except json.JSONDecodeError:
            continue

    # Fallback: try the whole response as JSON
    try:
        parsed = json.loads(response)
        return parsed
    except json.JSONDecodeError:
        return None


def apply_config_patch(patch: dict, cfg_path: str = CONFIG_FILE) -> bool:
    """
    Apply a dot-path patch to config.json.
    e.g. {"training.lr": 0.0005, "model.max_depth": 5}
    """
    if not patch:
        return False

    with open(cfg_path, "r") as f:
        cfg = json.load(f)

    changed = False
    for dotpath, value in patch.items():
        keys = dotpath.split(".")
        node = cfg
        try:
            for k in keys[:-1]:
                node = node[k]
            old_val = node.get(keys[-1])
            if old_val != value:
                node[keys[-1]] = value
                print(f"[Config] {dotpath}: {old_val} → {value}")
                changed = True
        except (KeyError, TypeError) as e:
            print(f"[Config] WARNING: Could not apply patch '{dotpath}': {e}")

    if changed:
        with open(cfg_path, "w") as f:
            json.dump(cfg, f, indent=2)

    return changed


# ---------------------------------------------------------------------------
# Helper: Run training subprocess
# ---------------------------------------------------------------------------

def run_training(timeout_secs: int = TRAIN_TIMEOUT_SECS) -> tuple[bool, str]:
    """
    Spawn `python train.py` as a subprocess.
    Returns (success: bool, log_excerpt: str).
      success=True  if the process exited with code 0 within timeout.
      success=False if it crashed, timed out, or produced OOM output.
    """
    print(f"\n[Runner] Launching training (max {timeout_secs}s) ...")
    proc = subprocess.Popen(
        [sys.executable, TRAIN_SCRIPT, "--config", CONFIG_FILE],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    log_lines  = []
    start_time = time.time()
    oom_detected = False

    while True:
        elapsed = time.time() - start_time

        # Non-blocking readline
        try:
            line = proc.stdout.readline()
        except Exception:
            line = ""

        if line:
            stripped = line.rstrip()
            log_lines.append(stripped)
            # Print live output
            print(f"  [train] {stripped}")

            # Detect OOM or fatal errors early
            low = stripped.lower()
            if any(kw in low for kw in ["out of memory", "cuda oom", "killed", "sigkill"]):
                print("[Runner] OOM detected — killing subprocess.")
                oom_detected = True
                proc.kill()
                break

        # Check if the process exited
        if proc.poll() is not None:
            # Drain remaining output
            remaining = proc.stdout.read()
            if remaining:
                for l in remaining.splitlines():
                    log_lines.append(l)
                    print(f"  [train] {l}")
            break

        # Enforce timeout
        if elapsed >= timeout_secs:
            print(f"[Runner] Timeout ({timeout_secs}s) reached — terminating.")
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
            break

    exit_code  = proc.returncode
    log_excerpt = "\n".join(log_lines[-30:])   # last 30 lines for diagnostics

    if oom_detected:
        return False, f"OOM detected.\n{log_excerpt}"
    if exit_code is None or exit_code != 0:
        return False, f"Exit code {exit_code}.\n{log_excerpt}"

    return True, log_excerpt


# ---------------------------------------------------------------------------
# Helper: Read validation loss from results.tsv
# ---------------------------------------------------------------------------

def read_best_val_loss(results_path: str = RESULTS_FILE) -> float:
    """
    Parse results.tsv and return the minimum val_loss seen so far.
    Returns inf if the file does not exist or has no valid rows.
    """
    if not os.path.exists(results_path):
        return float("inf")

    best = float("inf")
    try:
        with open(results_path, "r", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                try:
                    v = float(row.get("val_loss", "nan"))
                    if not math.isnan(v) and v < best:
                        best = v
                except ValueError:
                    continue
    except Exception as e:
        print(f"[Runner] WARNING: Could not read {results_path}: {e}")

    return best


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def build_llm_prompt(
    iteration: int,
    best_val_loss: float,
    last_val_loss: float,
    current_cfg: dict,
) -> str:
    """
    Build a concise prompt for Qwen2.5-Coder to suggest a config tweak.
    """
    cfg_summary = json.dumps(
        {
            "training.lr":                   current_cfg["training"]["lr"],
            "training.batch_size":           current_cfg["training"]["batch_size"],
            "loss_weights.routing_balance":  current_cfg["loss_weights"]["routing_balance"],
            "loss_weights.halting":          current_cfg["loss_weights"]["halting"],
            "model.dropout":                 current_cfg["model"]["dropout"],
        },
        indent=2,
    )

    prompt = textwrap.dedent(f"""
    You are an ML research assistant optimising a Mixture of Recursive Experts
    (MoRE) model for sequence regression on synthetic math/logic traces.

    Current hyperparameters:
    {cfg_summary}

    Training history:
      - Iteration:     {iteration}
      - Best val_loss so far: {best_val_loss:.6f}
      - Last val_loss:        {last_val_loss:.6f}

    Your task: suggest ONE small hyperparameter change that may improve val_loss.
    Valid dot-paths to modify:
      training.lr, training.batch_size, training.grad_clip,
      loss_weights.routing_balance, loss_weights.halting, model.dropout

    Constraints:
      - Do NOT change model architecture keys (d_model, num_experts, max_depth).
      - Keep lr between 1e-5 and 5e-3.
      - Keep routing_balance between 0.001 and 0.1.
      - Keep halting between 0.0001 and 0.01.
      - Keep dropout between 0.0 and 0.3.

    Respond ONLY with a valid JSON object, nothing else, no markdown:
    {{
      "config_patch": {{"training.lr": <new_value>}},
      "commit_message": "<short description of the change>"
    }}
    """).strip()
    return prompt


# ---------------------------------------------------------------------------
# Main runner loop
# ---------------------------------------------------------------------------

def run_autoresearch(max_iterations: int = 20, duration_minutes: float = 0):
    """
    Main automated research loop.

    Args:
        max_iterations  : stop after this many iterations regardless
        duration_minutes: if > 0, also stop when this many minutes have elapsed
    """
    start_wall = time.time()
    deadline   = start_wall + duration_minutes * 60 if duration_minutes > 0 else None

    print("=" * 70)
    print("  MoRE Autoresearch Runner")
    print(f"  Model   : {OLLAMA_MODEL}")
    print(f"  Timeout : {TRAIN_TIMEOUT_SECS}s per run")
    print(f"  Max iters: {max_iterations}")
    if deadline:
        print(f"  Wall deadline: {duration_minutes:.1f} min")
    print("=" * 70)

    # Verify git repo
    rc, out, err = _run_git(["rev-parse", "--is-inside-work-tree"])
    if rc != 0:
        print("[Runner] FATAL: Not inside a git repository.")
        print("  Run:  git init && git add . && git commit -m 'Initial baseline'")
        sys.exit(1)

    # Verify Ollama is reachable
    print("[Runner] Pinging Ollama ...")
    test_resp = query_ollama("Reply with the single word: READY", timeout=30)
    if not test_resp:
        print("[Runner] FATAL: Ollama not responding. Start it with:")
        print("  OLLAMA_KEEP_ALIVE=0m ollama serve")
        sys.exit(1)
    print(f"[Runner] Ollama responded: '{test_resp[:60]}'")

    # Read baseline
    baseline_val_loss = read_best_val_loss()
    if math.isinf(baseline_val_loss):
        print("[Runner] No results.tsv found — running initial baseline training ...")
        success, log = run_training()
        if not success:
            print(f"[Runner] FATAL: Baseline training failed:\n{log}")
            sys.exit(1)
        baseline_val_loss = read_best_val_loss()
        print(f"[Runner] Baseline val_loss = {baseline_val_loss:.6f}")
        git_stage_and_commit("Baseline run")

    print(f"[Runner] Starting from baseline val_loss = {baseline_val_loss:.6f}\n")

    consecutive_fails = 0
    last_val_loss     = baseline_val_loss

    for iteration in range(1, max_iterations + 1):
        iter_start = time.time()
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        print(f"\n{'─'*60}")
        print(f"  ITERATION {iteration}/{max_iterations}  [{ts}]")
        print(f"  Baseline val_loss = {baseline_val_loss:.6f}")
        print(f"{'─'*60}")

        # --- Wall clock deadline check ----------------------------------
        if deadline and time.time() >= deadline:
            print("[Runner] Wall-clock deadline reached. Stopping.")
            break

        # --- Load current config ----------------------------------------
        try:
            with open(CONFIG_FILE, "r") as f:
                current_cfg = json.load(f)
        except Exception as e:
            print(f"[Runner] ERROR loading config: {e}")
            break

        # --- Step 1: Query Ollama ---------------------------------------
        print(f"\n[Step 1] Querying {OLLAMA_MODEL} for suggestions ...")
        prompt = build_llm_prompt(iteration, baseline_val_loss, last_val_loss, current_cfg)
        llm_response = query_ollama(prompt)

        if not llm_response:
            print("[Step 1] Empty LLM response — skipping iteration.")
            consecutive_fails += 1
            if consecutive_fails >= MAX_CONSECUTIVE_FAILS:
                print("[Runner] Too many consecutive failures. Aborting.")
                break
            continue

        print(f"[Step 1] LLM response:\n  {llm_response[:300]}")

        # Wait for VRAM to drain after Ollama finishes
        print(f"[Step 1] Sleeping {VRAM_DRAIN_SLEEP_SECS}s for VRAM drain ...")
        time.sleep(VRAM_DRAIN_SLEEP_SECS)

        # --- Step 2: Parse & apply tweak --------------------------------
        print("\n[Step 2] Parsing LLM suggestion ...")
        parsed = parse_llm_tweak(llm_response, current_cfg)

        if parsed is None:
            print("[Step 2] Could not parse a valid JSON tweak. Skipping.")
            consecutive_fails += 1
            if consecutive_fails >= MAX_CONSECUTIVE_FAILS:
                break
            continue

        config_patch   = parsed.get("config_patch", {})
        commit_message = parsed.get(
            "commit_message", f"Auto-tweak iteration {iteration}"
        )

        changed = apply_config_patch(config_patch)
        if not changed:
            print("[Step 2] No effective config change — skipping commit & run.")
            continue

        # --- Step 3: Git commit -----------------------------------------
        print(f"\n[Step 3] Committing: '{commit_message}' ...")
        committed = git_stage_and_commit(f"[iter-{iteration:03d}] {commit_message}")
        pre_run_sha = git_current_sha()

        # --- Step 4: Run training ---------------------------------------
        print(f"\n[Step 4] Running train.py (up to {TRAIN_TIMEOUT_SECS}s) ...")

        # Remove stale results so we don't read a previous run's metrics
        if os.path.exists(RESULTS_FILE):
            os.remove(RESULTS_FILE)

        success, log_excerpt = run_training(TRAIN_TIMEOUT_SECS)

        # --- Step 5: Evaluate ------------------------------------------
        print("\n[Step 5] Evaluating results ...")
        new_val_loss = read_best_val_loss()
        last_val_loss = new_val_loss

        if not success:
            print(f"[Step 5] Training FAILED. Log tail:\n{log_excerpt[-500:]}")
            print("[Step 5] Rolling back ...")
            git_rollback()
            consecutive_fails += 1
            if consecutive_fails >= MAX_CONSECUTIVE_FAILS:
                print("[Runner] Too many consecutive fails. Aborting.")
                break
            continue

        consecutive_fails = 0

        if math.isinf(new_val_loss):
            print("[Step 5] results.tsv missing or empty after training — rolling back.")
            git_rollback()
            continue

        delta = new_val_loss - baseline_val_loss
        improved = new_val_loss < baseline_val_loss

        print(
            f"[Step 5] val_loss: {new_val_loss:.6f}  "
            f"(Δ = {delta:+.6f}  baseline = {baseline_val_loss:.6f})"
        )

        if improved:
            print(f"[Step 5] ✓ IMPROVED — keeping commit {pre_run_sha}")
            baseline_val_loss = new_val_loss
        else:
            print("[Step 5] ✗ NOT IMPROVED — rolling back.")
            git_rollback()

        iter_elapsed = time.time() - iter_start
        print(f"\n[Iter {iteration}] Done in {iter_elapsed:.1f}s")

    total_elapsed = time.time() - start_wall
    print("\n" + "=" * 70)
    print(f"  Autoresearch complete after {total_elapsed/60:.1f} min")
    print(f"  Final baseline val_loss = {baseline_val_loss:.6f}")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MoRE Autoresearch Runner")
    parser.add_argument(
        "--iterations", type=int, default=20,
        help="Maximum number of LLM→train iterations (default: 20)"
    )
    parser.add_argument(
        "--duration-minutes", type=float, default=0,
        help="Optional wall-clock limit in minutes (0 = no limit)"
    )
    args = parser.parse_args()

    try:
        run_autoresearch(
            max_iterations=args.iterations,
            duration_minutes=args.duration_minutes,
        )
    except KeyboardInterrupt:
        print("\n[Runner] Interrupted by user. Exiting cleanly.")
    except Exception as e:
        print(f"\n[Runner] FATAL: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
