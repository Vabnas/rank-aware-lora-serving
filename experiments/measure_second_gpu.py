"""
measure_second_gpu.py -- Run C1, C2, and roofline/bandwidth measurements on a
SECOND GPU (e.g. RTX 4060) for cross-hardware validation, using the same
methodology as the main RTX 4080 SUPER results.

This produces a directly comparable results file: same rank (8), same tenant
sweep, same mixed-rank composition (25% high-rank) as the main experiments.

Run from the MAIN folder on the machine/environment with the RTX 4060:
    python measure_second_gpu.py

If you have both GPUs in the SAME machine, select the 4060 explicitly with:
    CUDA_VISIBLE_DEVICES=<index_of_4060> python measure_second_gpu.py
(use `nvidia-smi` to find the correct index if you have more than one GPU)

Output: results/second_gpu_results.json
"""
import sys, os, json, time
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
from config import MODELS, RESULTS_DIR, TENANT_COUNTS, gpu_name, WORKLOAD_BATCH, WORKLOAD_SEQ_LEN
from core.lora_model import MultiTenantLoRA

RANK = 8
DTYPE = torch.float16
HIGH_RANK = 32
LOW_RANK = 8
HIGH_FRACTION = 0.25   # matches the main paper's mixed-rank composition

PEAK_BW_TABLE = {
    # Fill in / adjust if your card differs from the reference spec.
    "RTX 4060": 272.0,        # GB/s, 8GB variant
    "RTX 4060 Ti": 288.0,     # GB/s, 8GB variant (or 16GB variant same BW)
    "RTX 4080 SUPER": 736.0,
}
PEAK_FP16_TFLOPS = {
    "RTX 4060": 15.1,
    "RTX 4060 Ti": 22.1,
    "RTX 4080 SUPER": 52.0,
}

def detect_peak_bw(name):
    for key, val in PEAK_BW_TABLE.items():
        if key in name:
            return val
    print(f"  ! GPU '{name}' not in PEAK_BW_TABLE -- edit the script to add it.")
    return None

def detect_peak_flops(name):
    for key, val in PEAK_FP16_TFLOPS.items():
        if key in name:
            return val
    return None

def time_fn(fn, iters=50, warmup=10):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters): fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / iters * 1000.0  # ms

def mixed_ranks(T):
    n_high = max(1, round(T * HIGH_FRACTION))
    n_low = T - n_high
    return [LOW_RANK] * n_low + [HIGH_RANK] * n_high

def adapter_bytes(T, rank, d, tokens, dtype_bytes=2):
    weights = (T*d*rank + T*rank*d) * dtype_bytes
    h_read  = T*tokens*d * dtype_bytes
    ax_rw   = 2 * T*tokens*rank * dtype_bytes
    out_wr  = T*tokens*d * dtype_bytes
    return weights + h_read + ax_rw + out_wr

def run_model(model_key, peak_bw):
    cfg = MODELS[model_key]
    if not os.path.exists(cfg["path"]):
        print(f"  skip {model_key}: path not found"); return []
    print(f"\n{'='*64}\n  {cfg['name']}\n{'='*64}")
    rows = []
    tokens = WORKLOAD_BATCH * WORKLOAD_SEQ_LEN
    for T in TENANT_COUNTS:
        # --- C1: fused vs sequential, uniform rank ---
        mt = MultiTenantLoRA(cfg["path"], num_tenants=T, lora_rank=RANK, dtype=DTYPE)
        h = mt.compute_hidden_state()
        seq_ms = time_fn(lambda: mt.adapter_sequential(h))
        fused_ms = time_fn(lambda: mt.adapter_fused(h))
        c1_speedup = seq_ms / fused_ms
        d = mt.d_model
        mt.cleanup(); del mt; torch.cuda.empty_cache()

        # --- C2: pad-and-fuse vs rank-aware split, mixed rank ---
        ranks = mixed_ranks(T)
        max_rank = max(ranks)
        mt_pad = MultiTenantLoRA(cfg["path"], num_tenants=T, lora_rank=max_rank, dtype=DTYPE)
        h_pad = mt_pad.compute_hidden_state()
        pad_ms = time_fn(lambda: mt_pad.adapter_fused(h_pad))
        mt_pad.cleanup(); del mt_pad; torch.cuda.empty_cache()

        n_high = sum(1 for r in ranks if r == HIGH_RANK)
        n_low = T - n_high
        mt_low = MultiTenantLoRA(cfg["path"], num_tenants=n_low, lora_rank=LOW_RANK, dtype=DTYPE)
        h_low = mt_low.compute_hidden_state()
        low_ms = time_fn(lambda: mt_low.adapter_fused(h_low))
        mt_low.cleanup(); del mt_low; torch.cuda.empty_cache()

        if n_high > 0:
            mt_high = MultiTenantLoRA(cfg["path"], num_tenants=n_high, lora_rank=HIGH_RANK, dtype=DTYPE)
            h_high = mt_high.compute_hidden_state()
            high_ms = time_fn(lambda: mt_high.adapter_fused(h_high))
            mt_high.cleanup(); del mt_high; torch.cuda.empty_cache()
        else:
            high_ms = 0.0
        split_ms = low_ms + high_ms
        c2_speedup = pad_ms / split_ms if split_ms > 0 else float("nan")

        # --- roofline / bandwidth (fused, uniform rank, same as main study) ---
        nbytes = adapter_bytes(T, RANK, d, tokens)
        achieved_bw = nbytes / (fused_ms/1000.0) / 1e9
        bw_util = 100.0 * achieved_bw / peak_bw if peak_bw else None

        row = {
            "model_key": model_key, "model": cfg["name"], "num_tenants": T,
            "seq_ms": round(seq_ms, 4), "fused_ms": round(fused_ms, 4),
            "c1_speedup": round(c1_speedup, 3),
            "pad_ms": round(pad_ms, 4), "split_ms": round(split_ms, 4),
            "c2_speedup": round(c2_speedup, 3),
            "achieved_bw_GBs": round(achieved_bw, 1),
            "bw_utilization_pct": round(bw_util, 1) if bw_util else None,
        }
        rows.append(row)
        print(f"  T={T:<4} C1={c1_speedup:.2f}x  C2={c2_speedup:.2f}x  "
              f"BW={achieved_bw:.1f}GB/s"
              + (f" ({bw_util:.1f}% peak)" if bw_util else ""))
    return rows

def main():
    name = gpu_name()
    peak_bw = detect_peak_bw(name)
    peak_flops = detect_peak_flops(name)
    print(f"GPU: {name}")
    if peak_bw:
        ridge = peak_flops*1e12/(peak_bw*1e9) if peak_flops else None
        print(f"Peak BW={peak_bw} GB/s" + (f", ridge point={ridge:.1f} FLOP/byte" if ridge else ""))
    else:
        print("Peak BW unknown for this GPU -- edit PEAK_BW_TABLE in this script.")

    all_rows = []
    for mk in ["llama", "deepseek", "gemma"]:
        all_rows += run_model(mk, peak_bw)

    out = {"gpu": name, "peak_bw_GBs": peak_bw, "peak_fp16_tflops": peak_flops,
           "rank": RANK, "high_rank": HIGH_RANK, "low_rank": LOW_RANK,
           "high_fraction": HIGH_FRACTION, "results": all_rows}
    path = os.path.join(RESULTS_DIR, "second_gpu_results.json")
    with open(path, "w") as f: json.dump(out, f, indent=2)
    print(f"\nSaved {path} ({len(all_rows)} rows).")

if __name__ == "__main__": main()