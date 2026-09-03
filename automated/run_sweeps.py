"""
PRE-FINALIZATION SWEEP DRIVER -- NOT RUNNABLE, RETAINED AS HISTORY.
See the banner in automated/rerun_mor.py: absolute paths into
`C:\\Users\\Hp\\Desktop\\Waste\\MoRE`, and a global `results.tsv` / sweep CSV that
CLAUDE.md §5 prohibits and run_context.py refuses. T-L0.0 deliberately left the
interpreter constant stale rather than making this one edit from runnable.
"""

import json
import subprocess
import os
import sys
import time

PYTHON_EXE = r"C:\Users\Hp\anaconda3\envs\more_env\python.exe"
BASE_CONFIG_PATH = r"c:\Users\Hp\Desktop\Waste\MoRE\code\config_1block_fixed.json"
TEMP_CONFIG_PATH = r"c:\Users\Hp\Desktop\Waste\MoRE\code\temp_config_sweep.json"
RESULTS_FILE = r"c:\Users\Hp\Desktop\Waste\MoRE\code\results.tsv"
SWEETS_CSV = r"c:\Users\Hp\Desktop\Waste\MoRE\sweep_results.csv"

# Pre-defined base config
base_config = {
    "model": {
        "d_model": 256,
        "num_experts": 7,
        "max_depth": 7,
        "max_steps": 7,
        "step_feat_dim": 12,
        "num_blocks": 1,
        "dropout": 0.1,
        "fixed_depth": False
    },
    "training": {
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "batch_size": 768,
        "epochs": 5,           # Proxy sweep limit (5 epochs)
        "val_split": 0.5,
        "grad_clip": 1.0
    },
    "loss_weights": {
        "task": 1.0,
        "routing_balance": 0.05,
        "step_routing": 0.5,
        "halting": 0.001
    },
    "data": {
        "train_path": "../data/train.jsonl",
        "val_path": "../data/val.jsonl",
        "test_path": "../data/test.jsonl",
        "max_val": 1e6,
        "pad_value": 0.0,
        "subset_fraction": 0.1,  # 10% proxy subset
        "subset_seed": 42
    },
    "logging": {
        "wandb_project": "micro-MoRE-poc",
        "log_interval": 1,
        "run_name": "sweep_run"
    }
}

runs = []

# 1. Structural Ablation
runs.append({
    "group": "structural",
    "name": "MoRE_1block",
    "tweaks": {
        "model.num_blocks": 1,
        "model.num_experts": 7,
        "model.max_depth": 7,
        "model.fixed_depth": False
    }
})
runs.append({
    "group": "structural",
    "name": "MoRE_2blocks",
    "tweaks": {
        "model.num_blocks": 2,
        "model.num_experts": 7,
        "model.max_depth": 7,
        "model.fixed_depth": False
    }
})

# 2. Fixed Depth Ablation
runs.append({
    "group": "fixed_depth",
    "name": "FixedDepth_1",
    "tweaks": {
        "model.num_blocks": 1,
        "model.num_experts": 7,
        "model.max_depth": 1,
        "model.fixed_depth": True
    }
})
runs.append({
    "group": "fixed_depth",
    "name": "FixedDepth_5",
    "tweaks": {
        "model.num_blocks": 1,
        "model.num_experts": 7,
        "model.max_depth": 5,
        "model.fixed_depth": True
    }
})

# 3. Batch Size Sweep
for bs in [128, 256, 512, 1024, 1536, 2048, 2560]:
    runs.append({
        "group": "batch_size",
        "name": f"BatchSize_{bs}",
        "tweaks": {
            "training.batch_size": bs,
            "model.num_blocks": 1,
            "model.num_experts": 7,
            "model.max_depth": 7,
            "model.fixed_depth": False
        }
    })

# 4. Scaling Laws
scaling_combos = [
    ("Nano", 128),
    ("Micro", 256),
    ("Mini", 512)
]
for size_name, d_model in scaling_combos:
    # MoE: max_depth=1, num_experts=7, fixed_depth=True
    runs.append({
        "group": "scaling_laws",
        "name": f"{size_name}_MoE",
        "tweaks": {
            "model.d_model": d_model,
            "model.num_blocks": 1,
            "model.num_experts": 7,
            "model.max_depth": 1,
            "model.fixed_depth": True
        }
    })
    # MoR: max_depth=7, num_experts=1, fixed_depth=False
    runs.append({
        "group": "scaling_laws",
        "name": f"{size_name}_MoR",
        "tweaks": {
            "model.d_model": d_model,
            "model.num_blocks": 1,
            "model.num_experts": 1,
            "model.max_depth": 7,
            "model.fixed_depth": False
        }
    })
    # MoRE: max_depth=7, num_experts=7, fixed_depth=False
    # Only append if it's not the micro-baseline we already cover in structural or other runs
    runs.append({
        "group": "scaling_laws",
        "name": f"{size_name}_MoRE",
        "tweaks": {
            "model.d_model": d_model,
            "model.num_blocks": 1,
            "model.num_experts": 7,
            "model.max_depth": 7,
            "model.fixed_depth": False
        }
    })

def apply_tweak(config, dotpath, val):
    keys = dotpath.split(".")
    node = config
    for k in keys[:-1]:
        node = node[k]
    node[keys[-1]] = val

# Initialize CSV
with open(SWEETS_CSV, "w") as f:
    f.write("group,run_name,epochs,batch_size,d_model,num_blocks,num_experts,max_depth,fixed_depth,train_task_loss,val_loss,expert_entropy,avg_depth,max_cos_sim,routing_accuracy,elapsed_sec\n")

print(f"[Sweep Runner] Starting sequential execution of {len(runs)} configurations...")

for i, run in enumerate(runs):
    print(f"\n[Run {i+1}/{len(runs)}] Group: {run['group']} | Name: {run['name']}")
    
    # Reset base config
    run_cfg = json.loads(json.dumps(base_config))
    run_cfg["logging"]["run_name"] = f"sweep_{run['group']}_{run['name']}"
    
    # Apply tweaks
    for path, val in run["tweaks"].items():
        apply_tweak(run_cfg, path, val)
        
    # Write temp config
    with open(TEMP_CONFIG_PATH, "w") as f:
        json.dump(run_cfg, f, indent=2)
        
    # Clear results.tsv to capture only this run
    if os.path.exists(RESULTS_FILE):
        os.remove(RESULTS_FILE)
        
    # Spawn subprocess
    t0 = time.perf_counter()
    p = subprocess.Popen(
        [PYTHON_EXE, "train.py", "--config", TEMP_CONFIG_PATH],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        cwd=r"c:\Users\Hp\Desktop\Waste\MoRE\code"
    )
    
    # Stream stdout to keep logs updated
    for line in p.stdout:
        print(f"  {line.strip()}")
        
    p.wait()
    t1 = time.perf_counter()
    elapsed = t1 - t0
    
    # Extract last metrics
    train_task_loss = float("nan")
    val_loss = float("nan")
    expert_entropy = float("nan")
    avg_depth = float("nan")
    max_cos_sim = float("nan")
    routing_accuracy = float("nan")
    
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "r") as f:
            lines = f.readlines()
            if len(lines) > 1:
                # epoch, train_task_loss, val_loss, expert_entropy, avg_depth, max_cos_sim, routing_accuracy
                last_row = lines[-1].strip().split("\t")
                if len(last_row) >= 7:
                    try:
                        train_task_loss = float(last_row[1])
                        val_loss = float(last_row[2])
                        expert_entropy = float(last_row[3])
                        avg_depth = float(last_row[4])
                        max_cos_sim = float(last_row[5])
                        routing_accuracy = float(last_row[6])
                    except ValueError:
                        pass
                        
    # Append to sweep results
    mc = run_cfg["model"]
    tc = run_cfg["training"]
    with open(SWEETS_CSV, "a") as f:
        f.write(
            f"{run['group']},{run['name']},{tc['epochs']},{tc['batch_size']},"
            f"{mc['d_model']},{mc['num_blocks']},{mc['num_experts']},{mc['max_depth']},{mc['fixed_depth']},"
            f"{train_task_loss},{val_loss},{expert_entropy},{avg_depth},{max_cos_sim},{routing_accuracy},{elapsed:.2f}\n"
        )
        
    print(f"Completed {run['name']}. Elapsed: {elapsed:.2f}s | Val Loss: {val_loss} | Entropy: {expert_entropy}")
    
    # Cool down VRAM
    time.sleep(5)

print("\n[Sweep Runner] ALL SWEEPS COMPLETE!")
