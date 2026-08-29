"""
exp6_repeated_stats.py — Repeated-run statistics for latency experiments.

Runs each latency configuration N times (default 3) and reports mean +/- std,
giving real variance for statistical tests and error-bar figures. Covers:
    - C1 fused batching (sequential vs fused)
    - C2 rank-aware splitting (pad vs split)
    - baseline (sequential)

Run:  python -m experiments.exp6_repeated_stats --model all --runs 3
Output: results/exp6_stats.json with per-config mean, std, and all raw runs.
"""
import sys, os, json, argparse, random
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from config import MODELS, TENANT_COUNTS, RESULTS_DIR, gpu_name
from core.lora_model import MultiTenantLoRA
from utils.benchmark import time_adapter, vram_gb
random.seed(42)

FIXED_RANK = 8
HIGH_RANK = 128
LOW_RANK = 8
SPLIT_KNEE = 32

def mixed_ranks(T, high_fraction=0.25):
    n_high = max(1, int(T * high_fraction))
    ranks = [HIGH_RANK]*n_high + [LOW_RANK]*(T - n_high)
    random.shuffle(ranks)
    return ranks

def measure_c1(cfg, T):
    m = MultiTenantLoRA(cfg["path"], T, FIXED_RANK, cfg["dtype"])
    h = m.compute_hidden_state()
    seq = time_adapter(m, h, "sequential")
    fused = time_adapter(m, h, "fused")
    m.cleanup()
    return seq, fused

def measure_c2(cfg, T, ranks):
    m = MultiTenantLoRA(cfg["path"], T, max(ranks), cfg["dtype"])
    h = m.compute_hidden_state()
    pad = time_adapter(m, h, "fused")
    m.cleanup()
    lows = [r for r in ranks if r <= SPLIT_KNEE]
    highs = [r for r in ranks if r > SPLIT_KNEE]
    split = 0.0
    for group, gr in [(lows, LOW_RANK), (highs, HIGH_RANK)]:
        if not group: continue
        m = MultiTenantLoRA(cfg["path"], len(group), gr, cfg["dtype"])
        h = m.compute_hidden_state()
        split += time_adapter(m, h, "fused")
        m.cleanup()
    return pad, split

def run(model_key, cfg, n_runs):
    print(f"\n{'='*65}\n  REPEATED STATS — {cfg['name']} ({cfg['company']})  [{n_runs} runs]\n{'='*65}")
    if not os.path.exists(cfg["path"]):
        print(f"  X Path not found: {cfg['path']}"); return []
    rows = []
    for T in TENANT_COUNTS:
        ranks = mixed_ranks(T)
        seq_runs, fused_runs, pad_runs, split_runs = [], [], [], []
        for run_i in range(n_runs):
            try:
                seq, fused = measure_c1(cfg, T)
                pad, split = measure_c2(cfg, T, ranks)
                seq_runs.append(seq); fused_runs.append(fused)
                pad_runs.append(pad); split_runs.append(split)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"  OOM at T={T}"); torch.cuda.empty_cache(); break
                raise
        if not seq_runs: continue
        def stat(a): return round(float(np.mean(a)),4), round(float(np.std(a)),4)
        seq_m, seq_s = stat(seq_runs); fused_m, fused_s = stat(fused_runs)
        pad_m, pad_s = stat(pad_runs); split_m, split_s = stat(split_runs)
        c1_speedup = seq_m / fused_m
        c2_speedup = pad_m / split_m
        print(f"  T={T:<4} C1: {seq_m:.3f}->{fused_m:.3f} ({c1_speedup:.2f}x +/-{fused_s:.3f})  "
              f"C2: {pad_m:.3f}->{split_m:.3f} ({c2_speedup:.2f}x)")
        rows.append({
            "model_key": model_key, "model": cfg["name"], "num_tenants": T, "n_runs": n_runs,
            "seq_mean": seq_m, "seq_std": seq_s, "fused_mean": fused_m, "fused_std": fused_s,
            "pad_mean": pad_m, "pad_std": pad_s, "split_mean": split_m, "split_std": split_s,
            "c1_speedup": round(c1_speedup,3), "c2_speedup": round(c2_speedup,3),
            "seq_runs": [round(x,4) for x in seq_runs],
            "fused_runs": [round(x,4) for x in fused_runs],
            "pad_runs": [round(x,4) for x in pad_runs],
            "split_runs": [round(x,4) for x in split_runs],
        })
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS)+["all"], default="all")
    ap.add_argument("--runs", type=int, default=3)
    args = ap.parse_args()
    print(f"GPU: {gpu_name()}  |  runs per config: {args.runs}")
    keys = list(MODELS) if args.model=="all" else [args.model]
    allr = []
    for k in keys:
        allr += run(k, MODELS[k], args.runs)
        with open(os.path.join(RESULTS_DIR, f"exp6_stats_{k}.json"), "w") as f:
            json.dump([r for r in allr if r["model_key"]==k], f, indent=2)
    combined = []
    for mk in MODELS:
        p = os.path.join(RESULTS_DIR, f"exp6_stats_{mk}.json")
        if os.path.exists(p):
            with open(p) as f: combined += json.load(f)
    with open(os.path.join(RESULTS_DIR, "exp6_stats.json"), "w") as f:
        json.dump(combined, f, indent=2)
    print(f"\nSaved exp6_stats.json ({len(combined)} configs)")

if __name__ == "__main__": main()