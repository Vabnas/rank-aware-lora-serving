"""
rerun_noisy.py — Re-run specific (model, tenant) configs that showed system
hiccups, with extra runs, and patch them into exp6_stats.json.
"""
import sys, os, json, argparse, random
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
from config import MODELS, RESULTS_DIR, gpu_name
from core.lora_model import MultiTenantLoRA
from utils.benchmark import time_adapter
from experiments.exp6_repeated_stats import measure_c1, measure_c2, mixed_ranks
random.seed(42)

NOISY = [("gemma", 128), ("llama", 64), ("llama", 128)]

def redo(model_key, T, n_runs):
    cfg = MODELS[model_key]
    ranks = mixed_ranks(T)
    seq_runs, fused_runs, pad_runs, split_runs = [], [], [], []
    for i in range(n_runs):
        seq, fused = measure_c1(cfg, T)
        pad, split = measure_c2(cfg, T, ranks)
        seq_runs.append(seq); fused_runs.append(fused)
        pad_runs.append(pad); split_runs.append(split)
        print(f"  {model_key} T={T} run{i+1}: seq={seq:.3f} fused={fused:.3f} pad={pad:.3f} split={split:.3f}")
    def stat(a): return round(float(np.mean(a)),4), round(float(np.std(a)),4)
    seq_m, seq_s = stat(seq_runs); fused_m, fused_s = stat(fused_runs)
    pad_m, pad_s = stat(pad_runs); split_m, split_s = stat(split_runs)
    return {
        "model_key": model_key, "model": cfg["name"], "num_tenants": T, "n_runs": n_runs,
        "seq_mean": seq_m, "seq_std": seq_s, "fused_mean": fused_m, "fused_std": fused_s,
        "pad_mean": pad_m, "pad_std": pad_s, "split_mean": split_m, "split_std": split_s,
        "c1_speedup": round(seq_m/fused_m,3), "c2_speedup": round(pad_m/split_m,3),
        "seq_runs": [round(x,4) for x in seq_runs],
        "fused_runs": [round(x,4) for x in fused_runs],
        "pad_runs": [round(x,4) for x in pad_runs],
        "split_runs": [round(x,4) for x in split_runs],
    }

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--runs", type=int, default=5)
    args = ap.parse_args(); print(f"GPU: {gpu_name()}  re-running noisy configs, {args.runs} runs each\n")

    path = os.path.join(RESULTS_DIR, "exp6_stats.json")
    with open(path) as f: data = json.load(f)

    for mk, T in NOISY:
        print(f"Re-running {mk} T={T}...")
        new_row = redo(mk, T, args.runs)
        for i, r in enumerate(data):
            if r["model_key"]==mk and r["num_tenants"]==T:
                data[i] = new_row; break
        print(f"  -> C1 {new_row['c1_speedup']}x (std {new_row['fused_std']})  "
              f"C2 {new_row['c2_speedup']}x (std {new_row['split_std']})\n")

    with open(path, "w") as f: json.dump(data, f, indent=2)
    print(f"Patched {path} with clean re-runs.")

if __name__ == "__main__": main()