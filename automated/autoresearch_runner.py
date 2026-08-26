"""
autoresearch_runner.py — Score-Driven Automated Research Supervisor
===================================================================
This script orchestrates an overnight automated research loop.
It NEVER loads Ollama and PyTorch into VRAM simultaneously.

*** PROXY-ONLY STRATEGY ***
  All trials run exclusively against the 5-epoch / 10%-data PROXY config
  (fast_config.json).  The full config.json (50 epochs, 100% data) is
  NEVER modified or executed by this runner.

  In the morning:
    1. Run: git log --oneline  (on the epoch5 branch)
    2. Identify the commit tagged [BEST] in the log or in the summary
       printed at the end of this script.
    3. Manually copy the winning hyperparameters into config.json.
    4. Run: python train.py --config config.json   (full 50-epoch run)

Loop per iteration:
  1. Query Ollama (Qwen2.5-Coder:7b) with detailed structural scores.
  2. Apply hyperparameter patch to fast_config.json only.
  3. Git commit the patch on the epoch5 branch.
  4. Spawn `python train.py --config fast_config.json` (5-epoch proxy).
  5. Compute comprehensive Architecture Score from results.tsv.
  6. If score improves -> keep commit, update structural baseline.
     If score degrades -> git rollback to restore working baseline.

Zero-Collision VRAM Strategy
-----------------------------
  • Ollama must be started externally with OLLAMA_KEEP_ALIVE=0m so it
    auto-unloads after each text generation call.
  • Subprocess execution unloads and isolates VRAM spaces cleanly.
"""

import os
import sys
import json
import time
import math
import argparse
import datetime
import subprocess
import traceback
import csv
import re
import textwrap
import requests

# ---------------------------------------------------------------------------
# Configuration Constants
# ---------------------------------------------------------------------------
OLLAMA_URL            = "http://localhost:11434/api/generate"
OLLAMA_MODEL          = "qwen2.5-coder:7b"
TRAIN_SCRIPT          = "train.py"

# PROXY config — the ONLY config this runner ever touches.
# fast_config.json: 5 epochs, 10% data subset, ~5-min wall time per trial.
CONFIG_FILE           = "fast_config.json"

# Full-run config — NEVER modified or executed by this runner.
# Use it manually in the morning for the final 50-epoch paper run.
FULL_CONFIG_FILE      = "config.json"

RESULTS_FILE          = "results.tsv"

# Branch that all overnight commits must land on.
PROXY_BRANCH          = "epoch5"

TRAIN_TIMEOUT_SECS    = 360   # 6-minute ceiling per proxy trial (5 epochs)
VRAM_DRAIN_SLEEP_SECS = 10    # Cooling window to drop LLM from GPU VRAM
MAX_CONSECUTIVE_FAILS = 3     # Crash safety threshold

# Pre-computed baseline metrics from the already-finished full 50-epoch run.
# This file lives in the wandb/ folder at the project root.
WANDB_BASELINE_CSV    = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "wandb", "baseline_values.csv"
)

# ---------------------------------------------------------------------------
# Git Framework Utilities
# ---------------------------------------------------------------------------
def _run_git(args: list[str], cwd: str = ".") -> tuple[int, str, str]:
    """Run a local git command execution line safely."""
    result = subprocess.run(
        ["git"] + args,
        cwd=cwd, capture_output=True, text=True
    )
    return result.returncode, result.stdout.strip(), result.stderr.strip()


def git_assert_proxy_branch() -> None:
    """
    Hard-abort if we are not on the designated proxy branch (epoch5).
    Prevents accidentally polluting main/other branches with overnight commits.
    """
    rc, current_branch, _ = _run_git(["rev-parse", "--abbrev-ref", "HEAD"])
    if rc != 0:
        print("[Git Guard] ERROR: Could not determine current branch. Aborting.")
        sys.exit(1)
    if current_branch != PROXY_BRANCH:
        print(
            f"[Git Guard] ABORT: Current branch is '{current_branch}', "
            f"but overnight proxy commits must land on '{PROXY_BRANCH}'.\n"
            f"  Run:  git checkout {PROXY_BRANCH}   then restart this script."
        )
        sys.exit(1)
    print(f"[Git Guard] ✓ Confirmed on branch '{PROXY_BRANCH}' — safe to proceed.")


def git_stage_and_commit(message: str) -> bool:
    """Stage proxy config adjustments and commit to the epoch5 branch."""
    # Only stage the proxy config — never touch config.json or train.py here.
    _run_git(["add", CONFIG_FILE])
    rc, out, err = _run_git(["commit", "-m", message])
    if rc != 0:
        if "nothing to commit" in out.lower() or "nothing to commit" in err.lower():
            print("[Git] No parameter adjustments detected—skipping loop commit.")
            return False
        print(f"[Git] ERROR: Commit state failed: {err}")
        return False
    _, sha, _ = _run_git(["rev-parse", "--short", "HEAD"])
    print(f"[Git] Active Commit Ratchet: {sha} | Info: '{message}'")
    return True


def git_rollback() -> bool:
    """Hard-reset workspace back to previous state to purge degraded trials."""
    rc, out, err = _run_git(["reset", "--hard", "HEAD~1"])
    if rc != 0:
        print(f"[Git] CRITICAL ERROR: Workspace rollback failed: {err}")
        return False
    _, sha, _ = _run_git(["rev-parse", "--short", "HEAD"])
    print(f"[Git] Successfully restored baseline state at commit: {sha}")
    return True


def git_current_sha() -> str:
    """Retrieve tracking signature of current working commit branch."""
    _, sha, _ = _run_git(["rev-parse", "--short", "HEAD"])
    return sha


# ---------------------------------------------------------------------------
# Local LLM Communication Engine
# ---------------------------------------------------------------------------
def query_ollama(prompt: str, timeout: int = 120) -> str:
    """Query local Ollama instance. Relies on short keep-alive to drop VRAM footprints."""
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.4,
            "num_predict": 512,
        },
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except requests.exceptions.ConnectionError:
        print("[Ollama] ERROR: Connection refused. Verify 'ollama serve' is running.")
        return ""
    except Exception as e:
        print(f"[Ollama] Query exception encountered: {e}")
        return ""


# ---------------------------------------------------------------------------
# Configuration & Response Parsing Engines
# ---------------------------------------------------------------------------
def parse_llm_tweak(response: str) -> dict | None:
    """Extract and validate the proposed patch blocks from LLM generation streams."""
    json_pattern = re.compile(r'\{[^{}]*\}', re.DOTALL)
    matches = json_pattern.findall(response)
    for match in matches:
        try:
            parsed = json.loads(match)
            if "config_patch" in parsed:
                return parsed
        except json.JSONDecodeError:
            continue
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        return None


def apply_config_patch(patch: dict, cfg_path: str = CONFIG_FILE) -> bool:
    """Apply a dot-notation JSON parameter patch to the primary config file."""
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
                print(f"[Config Optimization] Layer '{dotpath}': {old_val} -> {value}")
                changed = True
        except (KeyError, TypeError) as e:
            print(f"[Config] Out-of-bounds parameter path '{dotpath}': {e}")
    if changed:
        with open(cfg_path, "w") as f:
            json.dump(cfg, f, indent=2)
    return changed


# ---------------------------------------------------------------------------
# Wandb Baseline Loader
# ---------------------------------------------------------------------------
def load_wandb_baseline(csv_path: str = WANDB_BASELINE_CSV) -> tuple[float, dict]:
    """
    Reads the pre-computed baseline metrics from wandb/baseline_values.csv
    and converts them into the same (score, metrics) format used by
    calculate_architecture_score().

    Column layout expected:
      metric,value
      train/expert_load_entropy,<float>
      train/avg_recursion_steps,<float>
      val/loss,<float>
      diag/max_pairwise_cosine_sim,<float>
      train/task_loss,<float>
    """
    failed = {"entropy": 0.0, "slope": 0.0, "depth": 1.0, "cossim": 1.0, "val_loss": 1.0}
    csv_path = os.path.normpath(csv_path)
    if not os.path.exists(csv_path):
        print(f"[Baseline] ERROR: Baseline CSV not found at: {csv_path}")
        return -9999.0, failed

    lookup: dict[str, str] = {}
    try:
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                metric = row.get("metric", "").strip()
                value  = row.get("value",  "").strip().replace(",", "")  # handle "1,164.1" style
                if metric:
                    lookup[metric] = value
    except Exception as e:
        print(f"[Baseline] ERROR reading CSV: {e}")
        return -9999.0, failed

    def _f(key: str, default: float = 0.0) -> float:
        try:
            return float(lookup.get(key, default))
        except (ValueError, TypeError):
            return default

    entropy     = _f("train/expert_load_entropy",    0.0)
    avg_depth   = _f("train/avg_recursion_steps",    1.0)
    val_loss    = _f("val/loss",                     1.0)
    max_cos_sim = _f("diag/max_pairwise_cosine_sim", 1.0)
    task_loss   = _f("train/task_loss",              1.0)
    # For a static baseline the loss-slope cannot be computed from a single
    # snapshot, so we set it to 0 (conservative — proxy trials will earn
    # slope credit on top of this).
    loss_slope  = 0.0

    score = 0.0
    score += entropy    * 15.0
    score += loss_slope * 10.0
    if 1.5 <= avg_depth <= 3.5:
        score += 5.0
    elif avg_depth <= 1.05 or avg_depth >= 6.95:
        score -= 20.0
    score -= max_cos_sim * 8.0
    if entropy < 0.1:
        score -= 30.0          # collapsed routing penalty (matches score calc)
    score += (1.0 - min(val_loss, 1.0)) * 2.0

    metrics = {
        "entropy": entropy,
        "slope":   loss_slope,
        "depth":   avg_depth,
        "cossim":  max_cos_sim,
        "val_loss": val_loss,
    }
    print(f"[Baseline] Loaded from {os.path.basename(csv_path)}:")
    print(f"  Entropy={entropy:.4f}  Depth={avg_depth:.3f}  "
          f"ValLoss={val_loss:.5f}  CosSim={max_cos_sim:.5f}")
    print(f"  => Baseline Architecture Score: {score:.2f}")
    return score, metrics


# ---------------------------------------------------------------------------
# Training Subprocess Runtime Engine
# ---------------------------------------------------------------------------
def run_training(timeout_secs: int = TRAIN_TIMEOUT_SECS) -> tuple[bool, str]:
    """Execute proxy training (fast_config.json) as an isolated subprocess."""
    print(f"[Subprocess] Running 5-epoch PROXY trial via '{CONFIG_FILE}' (Max window: {timeout_secs}s)...")
    proc = subprocess.Popen(
        [sys.executable, TRAIN_SCRIPT, "--config", CONFIG_FILE],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )
    log_lines = []
    start_time = time.time()
    oom_detected = False

    while True:
        elapsed = time.time() - start_time
        try:
            line = proc.stdout.readline()
        except Exception:
            line = ""
        if line:
            stripped = line.rstrip()
            log_lines.append(stripped)
            print(f"  [train.py] {stripped}")
            low = stripped.lower()
            if any(kw in low for kw in ["out of memory", "cuda oom", "killed", "sigkill"]):
                print("[Subprocess] Out-of-Memory event confirmed. Aborting process thread.")
                oom_detected = True
                proc.kill()
                break
        if proc.poll() is not None:
            remaining = proc.stdout.read()
            if remaining:
                log_lines.extend(remaining.splitlines())
            break
        if elapsed >= timeout_secs:
            print(f"[Subprocess] Trial deadline reached ({timeout_secs}s)—terminating process.")
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
            break

    exit_code = proc.returncode
    log_excerpt = "\n".join(log_lines[-25:])
    if oom_detected or (exit_code is not None and exit_code != 0):
        return False, log_excerpt
    return True, log_excerpt


# ---------------------------------------------------------------------------
# Core Architecture Scoring Mechanics
# ---------------------------------------------------------------------------
def calculate_architecture_score(results_path: str = RESULTS_FILE) -> tuple[float, dict]:
    """
    Parses logs and computes multi-metric fitness scores.
    Heavily targets routing exploration while guarding against metric degradation.
    """
    failed_metrics = {"entropy": 0.0, "slope": 0.0, "depth": 1.0, "cossim": 1.0, "val_loss": 1.0}
    if not os.path.exists(results_path):
        return -9999.0, failed_metrics

    rows = []
    try:
        with open(results_path, "r") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                rows.append(row)
    except Exception as e:
        print(f"[Score Calculation] File read error: {e}")
        return -9999.0, failed_metrics

    if not rows:
        return -9999.0, failed_metrics

    final_epoch = rows[-1]
    try:
        val_loss    = float(final_epoch.get("val_loss", 1.0))
        entropy     = float(final_epoch.get("expert_entropy", 0.0))
        avg_depth   = float(final_epoch.get("avg_depth", 1.0))
        max_cos_sim = float(final_epoch.get("max_cos_sim", 0.0))
        
        init_loss  = float(rows[0].get("train_task_loss", 1.0))
        end_loss   = float(final_epoch.get("train_task_loss", 1.0))
        loss_slope = max(0.0, init_loss - end_loss)
    except (ValueError, TypeError):
        return -9999.0, failed_metrics

    # --- Structural Scaling Calculation Equations ---
    score = 0.0
    score += entropy * 15.0       # Priority 1: High expert routing distribution weight
    score += loss_slope * 10.0     # Priority 2: Accelerated convergence velocity
    
    if 1.5 <= avg_depth <= 3.5:
        score += 5.0              # Priority 3: Stable early-halting window reward
    elif avg_depth <= 1.05 or avg_depth >= 6.95:
        score -= 20.0             # Severe penalty: Locked depth routing loops

    score -= max_cos_sim * 8.0     # Priority 4: Enforce representation divergence
    
    if entropy < 0.1:
        score -= 30.0             # Strict Guardrail: Prevent softargmax routing collapse
    score += (1.0 - min(val_loss, 1.0)) * 2.0

    metrics_pack = {
        "entropy": entropy, "slope": loss_slope, 
        "depth": avg_depth, "cossim": max_cos_sim, "val_loss": val_loss
    }
    return score, metrics_pack


# ---------------------------------------------------------------------------
# Structured LLM Prompt Engineer
# ---------------------------------------------------------------------------
def build_llm_prompt(
    iteration: int,
    base_score: float,
    last_score: float,
    base_m: dict,
    last_m: dict,
    current_cfg: dict
) -> str:
    """Constructs explicit prompt briefs tracking structural score velocities."""
    cfg_summary = json.dumps({
        "training.lr":                  current_cfg["training"]["lr"],
        "training.batch_size":          current_cfg["training"]["batch_size"],
        "training.grad_clip":           current_cfg["training"].get("grad_clip", 1.0),
        "loss_weights.routing_balance": current_cfg["loss_weights"]["routing_balance"],
        "loss_weights.halting":         current_cfg["loss_weights"]["halting"],
        "loss_weights.step_routing":    current_cfg["loss_weights"].get("step_routing", 0.5),
        "model.dropout":                current_cfg["model"]["dropout"],
        # Read-only context (do not modify these via config_patch)
        "[proxy] training.epochs":      current_cfg["training"].get("epochs", 5),
        "[proxy] subset_fraction":      current_cfg["training"].get("subset_fraction", 0.10),
        "[arch] model.max_steps":       current_cfg["model"].get("max_steps", 7),
        "[arch] model.step_feat_dim":   current_cfg["model"].get("step_feat_dim", 12),
    }, indent=2)

    return textwrap.dedent(f"""
    You are an automated ML optimization agent engineering a Mixture of Recursive Experts (MoRE) network.
    Your absolute objective is maximizing the composite 'Architecture Health Score'. Do NOT focus on validation loss alone.

    CONTEXT: This is a PROXY environment — 5 epochs on 10% of training data.
    Scores are relative indicators, not final values. The winning config will
    be transferred to the full 50-epoch run manually in the morning.
    DO NOT suggest changes to 'training.epochs' or 'subset_fraction'.

    Current Proxy Hyperparameter Layout (fast_config.json):
    {cfg_summary}

    Empirical Optimization History:
      - Current Trial Iteration: {iteration}
      - Benchmark Baseline Score: {base_score:.2f} (Entropy: {base_m['entropy']:.3f}, Depth: {base_m['depth']:.2f}, Loss: {base_m['val_loss']:.4f})
      - Immediate Prior Trial Score: {last_score:.2f} (Entropy: {last_m['entropy']:.3f}, Depth: {last_m['depth']:.2f}, Loss: {last_m['val_loss']:.4f})

    Optimization Metric Hierarchy Priority:
      1. Routing Load Entropy (Must avoid routing collapse 0.0; target floor > 1.2)
      2. Learning Rate & Velocity (Maximize initial training convergence trajectory slope)
      3. Dynamic Exiting Mechanics (Average token recursion depth must hover between 1.5 and 3.5 steps)
      4. Weight Diversity (Prevent representation collapse by keeping max pairwise expert cosine similarity low)

    Valid parameter dot-paths open for modification:
      training.lr, training.batch_size, training.grad_clip,
      loss_weights.routing_balance, loss_weights.halting, loss_weights.step_routing,
      model.dropout

    Operational Boundary Constraints:
      - LR: [1e-5 to 5e-3] | Dropout: [0.0 to 0.3]
      - routing_balance loss weight: [0.001 to 0.1]
      - halting loss weight: [0.0001 to 0.01]
      - step_routing loss weight: [0.1 to 2.0]  (per-step oracle CE supervision strength)

    DO NOT modify: training.epochs, subset_fraction, model.max_steps, model.step_feat_dim
    (these are architecture constants — changing them breaks the input pipeline).

    Respond ONLY with a valid, parsable raw JSON structural patch string. Do not include markdown wraps or explanations:
    {{
      "config_patch": {{"loss_weights.routing_balance": 0.05, "training.lr": 0.0008}},
      "commit_message": "Elevate routing balancing penalty weight to wake up latent expert pathways"
    }}
    """).strip()


# ---------------------------------------------------------------------------
# Core Supervisor Loop Execution Space
# ---------------------------------------------------------------------------
def run_autoresearch(max_iterations: int = 20, duration_minutes: float = 0):  # noqa: C901
    """
    Orchestrates proxy-only (5-epoch / 10% data) hyperparameter search.

    IMPORTANT: This function NEVER modifies config.json or triggers a
    full 50-epoch training run.  All work is confined to fast_config.json
    and the epoch5 git branch.  At the end it prints the best commit SHA
    so you can cherry-pick the winner and run train.py manually.
    """
    start_wall = time.time()
    deadline = start_wall + duration_minutes * 60 if duration_minutes > 0 else None

    print("=" * 75)
    print("  MoRE Proxy Hyperparameter Search — epoch5 branch")
    print(f"  Proxy Config : {CONFIG_FILE}  (5 epochs, 10% data)")
    print(f"  Full Config  : {FULL_CONFIG_FILE}  ← NOT touched by this script")
    print(f"  LLM Engine   : {OLLAMA_MODEL} | Trial ceiling: {TRAIN_TIMEOUT_SECS}s")
    print("=" * 75)

    # ── Safety Guards ────────────────────────────────────────────────────────
    # 1. Confirm we are inside a git repo.
    if _run_git(["rev-parse", "--is-inside-work-tree"])[0] != 0:
        print("[System ERROR] Workspace is not inside a git repository. Initialize git tracking first.")
        sys.exit(1)

    # 2. Confirm we are on the epoch5 branch — hard abort otherwise.
    git_assert_proxy_branch()

    # 3. Confirm the proxy config exists.
    if not os.path.exists(CONFIG_FILE):
        print(f"[System ERROR] Proxy config '{CONFIG_FILE}' not found. Cannot proceed.")
        sys.exit(1)

    # 4. Paranoia check: ensure full config is never the active proxy.
    if CONFIG_FILE == FULL_CONFIG_FILE:
        print("[System ERROR] CONFIG_FILE and FULL_CONFIG_FILE must differ. Check constants.")
        sys.exit(1)

    # 5. Confirm Ollama is alive.
    if not query_ollama("Reply ONLY with the single word READY"):
        print("[System ERROR] Ollama engine offline. Run: OLLAMA_KEEP_ALIVE=0m ollama serve")
        sys.exit(1)

    # ── Baseline (loaded from wandb/baseline_values.csv — no training run) ───
    print("\n[Baseline] Loading pre-computed baseline from wandb folder...")
    base_score, base_metrics = load_wandb_baseline()
    if base_score == -9999.0:
        print("[Baseline] CRITICAL: Could not parse baseline CSV. Check wandb/baseline_values.csv.")
        sys.exit(1)

    # The baseline comes from the already-committed full run — just record
    # which HEAD we are sitting on so rollbacks work correctly.
    baseline_sha = git_current_sha()
    print(f"\n[Baseline Confirmed] Score floor: {base_score:.2f} | HEAD: {baseline_sha}\n")

    # ── Best-Trial Tracker ───────────────────────────────────────────────────
    best_score = base_score
    best_metrics = base_metrics
    best_sha = baseline_sha
    best_iteration = 0  # 0 = baseline
    best_commit_msg = "[Proxy Baseline] Reference score established on fast_config.json"

    last_score = base_score
    last_metrics = base_metrics
    consecutive_fails = 0

    for iteration in range(1, max_iterations + 1):  # proxy-only loop
        iter_start = time.time()
        if deadline and time.time() >= deadline:
            print("[System Loop] Assigned wall-clock duration window completed. Exiting.")
            break

        print(f"\n{'━'*70}\n  OPTIMIZATION TRIAL ITERATION {iteration}/{max_iterations}\n  Current Benchmark Baseline Score Target: {base_score:.2f}\n{'━'*70}")

        try:
            with open(CONFIG_FILE, "r") as f:
                current_cfg = json.load(f)
        except Exception as e:
            print(f"[System Loop] Failed to load configuration layer: {e}")
            break

        # Step 1: Request parameter adjustments from the local agent
        prompt = build_llm_prompt(iteration, base_score, last_score, base_metrics, last_metrics, current_cfg)
        llm_response = query_ollama(prompt)
        if not llm_response:
            print("[Step 1] Ollama output stream returned empty tracking logs—skipping trial.")
            consecutive_fails += 1
            if consecutive_fails >= MAX_CONSECUTIVE_FAILS:
                break
            continue

        print(f"[Step 1] Received Agent Modification Plan:\n{llm_response[:320]}\n...")
        print(f"[Step 1] Pausing {VRAM_DRAIN_SLEEP_SECS}s to clean LLM memory states from GPU VRAM...")
        time.sleep(VRAM_DRAIN_SLEEP_SECS)

        # Step 2: Parse and process proposed configurations
        parsed = parse_llm_tweak(llm_response)
        if parsed is None or "config_patch" not in parsed:
            print("[Step 2] Parameter structure parsing failed or holds incomplete parameters—skipping loop.")
            consecutive_fails += 1
            if consecutive_fails >= MAX_CONSECUTIVE_FAILS:
                break
            continue

        config_patch = parsed.get("config_patch", {})
        commit_msg = parsed.get("commit_message", f"Auto-patch trial run - iteration {iteration}")

        if not apply_config_patch(config_patch):
            print("[Step 2] Modifications matched existing baseline parameters. Skipping trial run.")
            continue

        # Step 3: Branch adjustment history state preservation
        git_stage_and_commit(f"[Trial-{iteration:03d}] {commit_msg}")
        pre_run_sha = git_current_sha()

        # Step 4: Subprocess execution space setup
        if os.path.exists(RESULTS_FILE):
            os.remove(RESULTS_FILE)

        success, log_excerpt = run_training(TRAIN_TIMEOUT_SECS)

        # Step 5: Architecture scoring matrix processing
        print("\n[Step 5] Processing Trial Metrics Verification...")
        trial_score, trial_metrics = calculate_architecture_score()

        if not success or trial_score == -9999.0:
            print(f"[Step 5] Run failed or encountered memory fault bounds. Processing immediate workspace rollback.")
            print(f"[Diagnostics Trace] Log Tail:\n{log_excerpt[-400:]}")
            git_rollback()
            consecutive_fails += 1
            last_score = -50.0  # Penalize historical index for failure trajectories
            if consecutive_fails >= MAX_CONSECUTIVE_FAILS:
                print("[System Loop] Safety boundary exception triggered: too many sequential crash failures. Aborting.")
                break
            continue

        consecutive_fails = 0
        last_score = trial_score
        last_metrics = trial_metrics

        # Evaluate performance parameters against structural benchmarks
        improved = trial_score > base_score
        current_sha = git_current_sha()
        print(f"[Step 5] Verified Trial Score: {trial_score:.2f} | Benchmark Target Floor: {base_score:.2f} (Δ = {trial_score - base_score:+.2f})")

        if improved:
            print(f"[Step 5] ✓ STRUCTURAL RATINGS ELEVATED — keeping commit {current_sha}")
            base_score = trial_score
            base_metrics = trial_metrics
            # Track the all-time best result for the morning summary.
            if trial_score > best_score:
                best_score = trial_score
                best_metrics = trial_metrics
                best_sha = current_sha
                best_iteration = iteration
                best_commit_msg = commit_msg
                print(f"[Step 5] ★ NEW ALL-TIME BEST: {best_score:.2f} at {best_sha}")
        else:
            print("[Step 5] ✗ METRIC LOSS OR STRUCTURAL DEGRADATION DETECTED — Executing branch recovery rollback.")
            git_rollback()

        print(f"[Iteration Complete] Processed runtime window loop in {time.time() - iter_start:.1f}s")

    elapsed_min = (time.time() - start_wall) / 60.0
    print("\n" + "═" * 75)
    print("  PROXY SEARCH COMPLETE — Morning Action Summary")
    print("═" * 75)
    print(f"  Total runtime      : {elapsed_min:.1f} minutes")
    print(f"  Iterations run     : {iteration}")
    print(f"  Best proxy score   : {best_score:.2f}  (baseline was {base_score:.2f})")
    print(f"  Best found at      : iteration {best_iteration}")
    print(f"  Best commit SHA    : {best_sha}")
    print(f"  Best commit msg    : {best_commit_msg}")
    print(f"  Best metrics       : Entropy={best_metrics['entropy']:.3f}  "
          f"Depth={best_metrics['depth']:.2f}  "
          f"CosSim={best_metrics['cossim']:.3f}  "
          f"ValLoss={best_metrics['val_loss']:.4f}")
    print("─" * 75)
    print("  ► NEXT STEPS (do manually):")
    print(f"    1. git show {best_sha}:{CONFIG_FILE}")
    print(f"       Copy the winning hyperparameters into {FULL_CONFIG_FILE}")
    print(f"       (keep epochs=50, remove subset_fraction / subset_seed)")
    print(f"    2. python {TRAIN_SCRIPT} --config {FULL_CONFIG_FILE}")
    print( "       This is your full 50-epoch paper run.")
    print("═" * 75)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score-Driven MoRE Optimization Loop")
    parser.add_argument("--iterations", type=int, default=30, help="Maximum search loop parameters")
    parser.add_argument("--duration-minutes", type=float, default=0, help="Wall-clock execution limit")
    args = parser.parse_args()

    try:
        run_autoresearch(max_iterations=args.iterations, duration_minutes=args.duration_minutes)
    except KeyboardInterrupt:
        print("\n[System Execution] Optimization sequence halted via terminal command. Exiting.")
    except Exception as e:
        print(f"\n[System Fatal Exception] Script crashed: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)