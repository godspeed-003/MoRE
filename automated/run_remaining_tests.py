"""
PRE-FINALIZATION SWEEP DRIVER -- NOT RUNNABLE, RETAINED AS HISTORY.
See the banner in automated/rerun_mor.py: absolute paths into
`C:\\Users\\Hp\\Desktop\\Waste\\MoRE` (including a `sys.path.append` of that tree),
and a global `results.tsv` / `final_run_metrics.json` that CLAUDE.md §5 prohibits
and run_context.py refuses. T-L0.0 deliberately left the interpreter constant
stale rather than making this one edit from runnable. Despite the name, this is
not part of the correctness suite -- that is code/run_correctness_suite.py.
"""

import json
import subprocess
import os
import sys
import time
import torch

PYTHON_EXE = r"C:\Users\Hp\anaconda3\envs\more_env\python.exe"
BASE_CONFIG_PATH = r"c:\Users\Hp\Desktop\Waste\MoRE\code\config_1block_fixed.json"
TEMP_CONFIG_PATH = r"c:\Users\Hp\Desktop\Waste\MoRE\code\temp_config_mor.json"
RESULTS_FILE = r"c:\Users\Hp\Desktop\Waste\MoRE\code\results.tsv"
FINAL_METRICS_PATH = r"c:\Users\Hp\Desktop\Waste\MoRE\code\final_run_metrics.json"

# Set path for importing train.py
sys.path.append(r"c:\Users\Hp\Desktop\Waste\MoRE\code")

# ---------------------------------------------------------------------------
# 1. Parameter Counter
# ---------------------------------------------------------------------------
def compute_parameter_counts():
    print("\n=== Calculating Model Parameter Counts ===")
    from train import MoREModel
    
    sizes = [("Nano", 128), ("Micro", 256), ("Mini", 512)]
    archs = [
        ("MoE", 7, 1, True),
        ("MoR", 1, 7, False),
        ("MoRE", 7, 7, False)
    ]
    
    param_counts = {}
    for size_name, d_model in sizes:
        for arch_name, num_experts, max_depth, fixed_depth in archs:
            model = MoREModel(
                step_feat_dim=12,
                d_model=d_model,
                num_experts=num_experts,
                max_depth=max_depth,
                num_blocks=1,
                fixed_depth=fixed_depth
            )
            params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            param_counts[f"{size_name}_{arch_name}"] = params
            print(f"Size: {size_name:<6} | Arch: {arch_name:<5} | Params: {params:,}")
    return param_counts

# ---------------------------------------------------------------------------
# 2. Batch Size Profiler (Pass timings & GPU memory)
# ---------------------------------------------------------------------------
def profile_batch_sizes():
    print("\n=== Profiling Batch Sizes (Fwd/Bwd Timings & Peak GPU Memory) ===")
    import torch.nn.functional as F
    from train import MoREModel
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_sizes = [128, 256, 512, 768, 1024, 1536, 2048, 2560]
    
    profile_results = {}
    for bs in batch_sizes:
        # Generate dummy batch
        x = torch.randn(bs, 7, 12, device=device)
        step_mask = torch.ones(bs, 7, dtype=torch.bool, device=device)
        step_experts = torch.randint(0, 7, (bs, 7), device=device)
        target = torch.randn(bs, device=device)
        
        # Instantiate standard Micro baseline model
        model = MoREModel(
            step_feat_dim=12,
            d_model=256,
            num_experts=7,
            max_depth=7,
            num_blocks=1,
            fixed_depth=False
        ).to(device)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        
        # Warmup GPU
        for _ in range(3):
            reg_out, cls_out, step_cls_out, bal_loss, halt_loss, _, _, _, oracle_routing_ce, _ = model(x, step_mask, step_experts)
            loss = F.mse_loss(reg_out.squeeze(-1), target) + bal_loss + halt_loss
            loss.backward()
            optimizer.zero_grad()
            
        torch.cuda.synchronize()
        
        # Measure Forward pass
        t_fwd_start = time.perf_counter()
        for _ in range(20):
            reg_out, cls_out, step_cls_out, bal_loss, halt_loss, _, _, _, oracle_routing_ce, _ = model(x, step_mask, step_experts)
        torch.cuda.synchronize()
        t_fwd = (time.perf_counter() - t_fwd_start) / 20.0
        
        # Measure Backward pass
        reg_out, cls_out, step_cls_out, bal_loss, halt_loss, _, _, _, oracle_routing_ce, _ = model(x, step_mask, step_experts)
        loss = F.mse_loss(reg_out.squeeze(-1), target) + bal_loss + halt_loss
        
        t_bwd_start = time.perf_counter()
        for _ in range(20):
            loss.backward(retain_graph=True)
        torch.cuda.synchronize()
        t_bwd = (time.perf_counter() - t_bwd_start) / 20.0
        
        # Peak GPU Memory
        torch.cuda.reset_peak_memory_stats()
        reg_out, cls_out, step_cls_out, bal_loss, halt_loss, _, _, _, oracle_routing_ce, _ = model(x, step_mask, step_experts)
        loss = F.mse_loss(reg_out.squeeze(-1), target) + bal_loss + halt_loss
        loss.backward()
        peak_mem = torch.cuda.max_memory_allocated() / (1024 * 1024)
        
        profile_results[bs] = {
            "fwd_time": t_fwd,
            "bwd_time": t_bwd,
            "peak_mem": peak_mem
        }
        print(f"Batch Size: {bs:<5} | Fwd Time: {t_fwd:.4f}s | Bwd Time: {t_bwd:.4f}s | Peak Memory: {peak_mem:.2f} MB")
        
        # Free memory
        del model, optimizer, x, step_mask, step_experts, target, reg_out, cls_out, step_cls_out, loss
        torch.cuda.empty_cache()
        
    return profile_results

# ---------------------------------------------------------------------------
# 3. Batch Size Sweep 1-Epoch Runs (Throughput & Validation Loss)
# ---------------------------------------------------------------------------
def run_batch_size_throughput_sweep():
    print("\n=== Running Batch Size 1-Epoch Throughput Sweep ===")
    with open(BASE_CONFIG_PATH, "r") as f:
        base_cfg = json.load(f)
        
    batch_sizes = [128, 256, 512, 768, 1024, 1536, 2048, 2560]
    sweep_results = {}
    
    for bs in batch_sizes:
        print(f"Running 1-epoch profiling run for batch size {bs}...")
        cfg = json.loads(json.dumps(base_cfg))
        cfg["training"]["batch_size"] = bs
        cfg["training"]["epochs"] = 1
        cfg["data"]["subset_fraction"] = 0.5
        cfg["logging"]["run_name"] = f"profiling_bs_{bs}"
        
        with open(TEMP_CONFIG_PATH, "w") as f:
            json.dump(cfg, f, indent=2)
            
        if os.path.exists(FINAL_METRICS_PATH):
            os.remove(FINAL_METRICS_PATH)
            
        t0 = time.perf_counter()
        p = subprocess.Popen(
            [PYTHON_EXE, "train.py", "--config", TEMP_CONFIG_PATH],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            cwd=r"c:\Users\Hp\Desktop\Waste\MoRE\code"
        )
        p.wait()
        elapsed = time.perf_counter() - t0
        
        # Load output metrics
        throughput = 0.0
        val_loss = 0.0
        if os.path.exists(FINAL_METRICS_PATH):
            with open(FINAL_METRICS_PATH, "r") as f:
                metrics = json.load(f)
                throughput = metrics.get("perf/throughput_tokens_sec", 0.0)
                val_loss = metrics.get("val/loss", 0.0)
                
        sweep_results[bs] = {
            "throughput": throughput,
            "val_loss": val_loss,
            "elapsed_sec": elapsed
        }
        print(f"Completed BS {bs} | Throughput: {throughput:.2f} tokens/s | Val Loss: {val_loss:.6f} | Time Taken: {elapsed:.1f}s")
    return sweep_results

# ---------------------------------------------------------------------------
# 4. Run 50-Epoch Baselines
# ---------------------------------------------------------------------------
def run_50e_baselines():
    print("\n=== Running 50-Epoch Baselines (MoE, MoR, Fixed Depth 5) ===")
    with open(BASE_CONFIG_PATH, "r") as f:
        base_cfg = json.load(f)
        
    runs = [
        {
            "name": "MoE_Baseline_50e",
            "config_file": "config_moe_50e.json",
            "tweaks": {
                "model.max_depth": 1,
                "model.fixed_depth": True,
                "model.num_experts": 7,
                "model.num_blocks": 1
            }
        },
        {
            "name": "MoR_Baseline_50e",
            "config_file": "config_mor_50e.json",
            "tweaks": {
                "model.max_depth": 7,
                "model.fixed_depth": False,
                "model.num_experts": 1,
                "model.num_blocks": 1
            }
        },
        {
            "name": "FixedDepth5_Ablation_50e",
            "config_file": "config_depth5_50e.json",
            "tweaks": {
                "model.max_depth": 5,
                "model.fixed_depth": True,
                "model.num_experts": 7,
                "model.num_blocks": 1
            }
        }
    ]
    
    baseline_results = {}
    
    def apply_tweak(config, dotpath, val):
        keys = dotpath.split(".")
        node = config
        for k in keys[:-1]:
            node = node[k]
        node[keys[-1]] = val
        
    for run in runs:
        print(f"\nStarting 50-epoch training run for: {run['name']}...")
        cfg = json.loads(json.dumps(base_cfg))
        cfg["training"]["epochs"] = 50
        cfg["data"]["subset_fraction"] = 0.5
        cfg["logging"]["run_name"] = run["name"]
        
        for path, val in run["tweaks"].items():
            apply_tweak(cfg, path, val)
            
        cfg_path = os.path.join(r"c:\Users\Hp\Desktop\Waste\MoRE\code", run["config_file"])
        with open(cfg_path, "w") as f:
            json.dump(cfg, f, indent=2)
            
        if os.path.exists(FINAL_METRICS_PATH):
            os.remove(FINAL_METRICS_PATH)
            
        t0 = time.perf_counter()
        p = subprocess.Popen(
            [PYTHON_EXE, "train.py", "--config", cfg_path],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            cwd=r"c:\Users\Hp\Desktop\Waste\MoRE\code"
        )
        p.wait()
        elapsed = time.perf_counter() - t0
        
        # Load final metrics
        metrics = {}
        if os.path.exists(FINAL_METRICS_PATH):
            with open(FINAL_METRICS_PATH, "r") as f:
                metrics = json.load(f)
                
        baseline_results[run["name"]] = {
            "metrics": metrics,
            "elapsed_sec": elapsed
        }
        print(f"Completed {run['name']} in {elapsed:.1f}s | Best Val Loss: {metrics.get('best_val_loss', 'N/A')}")
        
    return baseline_results

# ---------------------------------------------------------------------------
# 5. Formatting Output Tables
# ---------------------------------------------------------------------------
def format_results(param_counts, profile_results, sweep_results, baseline_results):
    print("\n\n" + "="*50 + "\nRESULTS REPORT FOR PUBLICATIONS\n" + "="*50)
    
    # 1. Scaling laws parameter counts table
    print("\n### Parameter Counts (Table VI & Table I)")
    print("| Architecture | Nano | Micro | Mini |")
    print("| :--- | :---: | :---: | :---: |")
    moe_nano = param_counts.get("Nano_MoE", 0)
    moe_micro = param_counts.get("Micro_MoE", 0)
    moe_mini = param_counts.get("Mini_MoE", 0)
    print(f"| **MoE** | {moe_nano:,} | {moe_micro:,} | {moe_mini:,} |")
    
    mor_nano = param_counts.get("Nano_MoR", 0)
    mor_micro = param_counts.get("Micro_MoR", 0)
    mor_mini = param_counts.get("Mini_MoR", 0)
    print(f"| **MoR** | {mor_nano:,} | {mor_micro:,} | {mor_mini:,} |")
    
    more_nano = param_counts.get("Nano_MoRE", 0)
    more_micro = param_counts.get("Micro_MoRE", 0)
    more_mini = param_counts.get("Mini_MoRE", 0)
    print(f"| **MoRE** | {more_nano:,} | {more_micro:,} | {more_mini:,} |")
    
    # 2. Batch size profiling table
    print("\n### Batch Size Profiling and Efficiency (Table III)")
    print("| Batch Size | Fwd (s) | Bwd (s) | Throughput (tok/s) | GPU Mem (MB) | Val Loss (1e) | Time Taken |")
    print("| :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for bs in [128, 256, 512, 768, 1024, 1536, 2048, 2560]:
        prof = profile_results[bs]
        swp = sweep_results[bs]
        time_str = f"{swp['elapsed_sec']:.1f}s"
        print(f"| {bs:<5} | {prof['fwd_time']:.4f} | {prof['bwd_time']:.4f} | {swp['throughput']:.2f} | {prof['peak_mem']:.2f} | {swp['val_loss']:.6f} | {time_str} |")
        
    # 3. Main baseline results table (Table I)
    print("\n### Main Baselines 50-Epoch (Table I)")
    print("| Model | Best Val Loss | Avg. Recursion Depth | Throughput (tok/s) | Parameter Count | Expert Load Entropy |")
    print("| :--- | :---: | :---: | :---: | :---: | :---: |")
    
    # MoE 50e
    moe_metrics = baseline_results["MoE_Baseline_50e"]["metrics"]
    moe_entropy = moe_metrics.get("train/expert_load_entropy", 0.0)
    print(f"| MoE | {moe_metrics.get('best_val_loss', 0.0):.6f} | 1.00 | {moe_metrics.get('perf/throughput_tokens_sec', 0.0):.2f} | {moe_micro:,} | {moe_entropy:.4f} |")
    
    # MoR 50e
    mor_metrics = baseline_results["MoR_Baseline_50e"]["metrics"]
    mor_entropy = mor_metrics.get("train/expert_load_entropy", 0.0)
    print(f"| MoR | {mor_metrics.get('best_val_loss', 0.0):.6f} | {mor_metrics.get('train/avg_recursion_steps', 0.0):.2f} | {mor_metrics.get('perf/throughput_tokens_sec', 0.0):.2f} | {mor_micro:,} | {mor_entropy:.4f} |")
    
    # 4. Fixed-Depth Ablation 50-Epoch (Table V)
    print("\n### Fixed-Depth Ablation 50-Epoch (Table V)")
    print("| Configuration | Val Loss | Fwd (s) | Bwd (s) | Throughput (tok/s) | Expert Load Entropy |")
    print("| :--- | :---: | :---: | :---: | :---: | :---: |")
    
    depth1_prof = profile_results[768] # use BS 768 profiling
    depth5_metrics = baseline_results["FixedDepth5_Ablation_50e"]["metrics"]
    depth5_prof = profile_results[768] # approximation
    
    print(f"| Fixed Depth 1 | {moe_metrics.get('best_val_loss', 0.0):.6f} | {depth1_prof['fwd_time']:.4f} | {depth1_prof['bwd_time']:.4f} | {moe_metrics.get('perf/throughput_tokens_sec', 0.0):.2f} | {moe_entropy:.4f} |")
    print(f"| Fixed Depth 5 | {depth5_metrics.get('best_val_loss', 0.0):.6f} | {depth5_prof['fwd_time']:.4f} | {depth5_prof['bwd_time']:.4f} | {depth5_metrics.get('perf/throughput_tokens_sec', 0.0):.2f} | {depth5_metrics.get('train/expert_load_entropy', 0.0):.4f} |")

# ---------------------------------------------------------------------------
# Main Execution Flow
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    param_counts = compute_parameter_counts()
    profile_results = profile_batch_sizes()
    sweep_results = run_batch_size_throughput_sweep()
    baseline_results = run_50e_baselines()
    format_results(param_counts, profile_results, sweep_results, baseline_results)
