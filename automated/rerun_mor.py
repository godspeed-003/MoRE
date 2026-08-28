import json
import subprocess
import os
import time

PYTHON_EXE = r"C:\Users\Hp\anaconda3\envs\more_env\python.exe"
TEMP_CONFIG_PATH = r"c:\Users\Hp\Desktop\Waste\MoRE\code\temp_config_mor.json"
RESULTS_FILE = r"c:\Users\Hp\Desktop\Waste\MoRE\code\results.tsv"
SWEETS_CSV = r"c:\Users\Hp\Desktop\Waste\MoRE\sweep_results.csv"

base_config = {
    "model": {
        "d_model": 256,
        "num_experts": 1,      # MoR has 1 expert
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
        "epochs": 5,           # 5 epochs proxy
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
        "subset_fraction": 0.1,
        "subset_seed": 42
    },
    "logging": {
        "wandb_project": "micro-MoRE-poc",
        "log_interval": 1,
        "run_name": "mor_run"
    }
}

runs = [
    {"name": "Nano_MoR", "d_model": 128},
    {"name": "Micro_MoR", "d_model": 256},
    {"name": "Mini_MoR", "d_model": 512}
]

print("[Rerun MoR] Starting rerun of the 3 MoR configurations...")

for run in runs:
    print(f"\nRunning {run['name']} (d_model={run['d_model']})...")
    
    run_cfg = json.loads(json.dumps(base_config))
    run_cfg["model"]["d_model"] = run["d_model"]
    run_cfg["logging"]["run_name"] = f"sweep_scaling_laws_{run['name']}"
    
    with open(TEMP_CONFIG_PATH, "w") as f:
        json.dump(run_cfg, f, indent=2)
        
    if os.path.exists(RESULTS_FILE):
        os.remove(RESULTS_FILE)
        
    t0 = time.perf_counter()
    p = subprocess.Popen(
        [PYTHON_EXE, "train.py", "--config", TEMP_CONFIG_PATH],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        cwd=r"c:\Users\Hp\Desktop\Waste\MoRE\code"
    )
    
    for line in p.stdout:
        print(f"  {line.strip()}")
        
    p.wait()
    t1 = time.perf_counter()
    elapsed = t1 - t0
    
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
            f"scaling_laws,{run['name']},{tc['epochs']},{tc['batch_size']},"
            f"{mc['d_model']},{mc['num_blocks']},{mc['num_experts']},{mc['max_depth']},{mc['fixed_depth']},"
            f"{train_task_loss},{val_loss},{expert_entropy},{avg_depth},{max_cos_sim},{routing_accuracy},{elapsed:.2f}\n"
        )
        
    print(f"Completed {run['name']}. Elapsed: {elapsed:.2f}s | Val Loss: {val_loss} | Entropy: {expert_entropy}")
    time.sleep(2)

print("\n[Rerun MoR] COMPLETE!")
