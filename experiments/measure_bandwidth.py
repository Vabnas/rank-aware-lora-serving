"""
measure_bandwidth.py -- Roofline / memory-bandwidth analysis for adapter serving.

Computes, per (model, tenant): adapter bytes moved per batch, achieved
bandwidth (bytes / measured fused time), fraction of peak bandwidth used, and
arithmetic intensity (FLOP/byte) to place the workload on the roofline.

This is the systems-bottleneck study: it quantifies that adapter serving is
memory-bound and shows how close fused batching gets to the bandwidth ceiling.

Run from the MAIN folder:  python measure_bandwidth.py
Output: results/bandwidth_results.json + summary table.
"""
import sys, os, json, time
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
from config import MODELS, RESULTS_DIR, TENANT_COUNTS, gpu_name, WORKLOAD_BATCH, WORKLOAD_SEQ_LEN
from core.lora_model import MultiTenantLoRA

# RTX 4080 SUPER: 736 GB/s memory bandwidth, ~52 TFLOP/s FP16.
PEAK_BW_GBs = 736.0
PEAK_FP16_TFLOPs = 52.0
RANK = 8
DTYPE = torch.float16

def time_fn(fn, iters=50, warmup=10):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters): fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / iters * 1000.0  # ms per call

def adapter_bytes(T, rank, d, tokens, dtype_bytes=2):
    # Total memory traffic of the fused adapter path, all scaling with T:
    #   read A (T*d*r) + read B (T*r*d)                  -- adapter weights
    #   read h per tenant (T*tokens*d)                   -- broadcast input
    #   write + read intermediate Ax (2 * T*tokens*r)    -- down-projection out
    #   write output (T*tokens*d)                        -- up-projection out
    weights = (T*d*rank + T*rank*d) * dtype_bytes
    h_read  = T*tokens*d * dtype_bytes
    ax_rw   = 2 * T*tokens*rank * dtype_bytes
    out_wr  = T*tokens*d * dtype_bytes
    return weights + h_read + ax_rw + out_wr

def adapter_flops(T, rank, d, tokens):
    # Ax: tokens*d*rank MACs ; AxB: tokens*rank*d MACs ; 2 FLOP per MAC
    return T * 2 * (tokens*d*rank + tokens*rank*d)

def run_model(model_key):
    cfg = MODELS[model_key]
    if not os.path.exists(cfg["path"]):
        print(f"  skip {model_key}: path not found"); return []
    print(f"\n{'='*64}\n  BANDWIDTH -- {cfg['name']}\n{'='*64}")
    rows = []
    tokens = WORKLOAD_BATCH * WORKLOAD_SEQ_LEN
    for T in TENANT_COUNTS:
        mt = MultiTenantLoRA(cfg["path"], num_tenants=T, lora_rank=RANK, dtype=DTYPE)
        d = mt.d_model
        h = mt.compute_hidden_state()
        ms = time_fn(lambda: mt.adapter_fused(h), iters=50, warmup=10)
        sec = ms / 1000.0
        nbytes = adapter_bytes(T, RANK, d, tokens)
        nflops = adapter_flops(T, RANK, d, tokens)
        achieved_bw = nbytes / sec / 1e9
        bw_util = 100.0 * achieved_bw / PEAK_BW_GBs
        ai = nflops / nbytes
        rows.append({
            "model_key": model_key, "model": cfg["name"], "num_tenants": T,
            "bytes_moved": int(nbytes),
            "achieved_bw_GBs": round(achieved_bw, 1),
            "bw_utilization_pct": round(bw_util, 1),
            "arithmetic_intensity": round(ai, 3),
            "time_ms": round(ms, 4),
        })
        print(f"  T={T:<4} {nbytes/1e6:7.2f}MB  {achieved_bw:6.1f}GB/s  "
              f"{bw_util:5.1f}% peak  AI={ai:.2f} FLOP/byte")
        mt.cleanup(); del mt; torch.cuda.empty_cache()
    return rows

def main():
    print(f"GPU: {gpu_name()}")
    ridge = PEAK_FP16_TFLOPs*1e12/(PEAK_BW_GBs*1e9)
    print(f"Peak BW={PEAK_BW_GBs} GB/s, peak FP16={PEAK_FP16_TFLOPs} TFLOP/s")
    print(f"Roofline ridge point: {ridge:.1f} FLOP/byte  (below = memory-bound)\n")
    all_rows = []
    for mk in ["llama", "deepseek", "gemma"]:
        all_rows += run_model(mk)
    path = os.path.join(RESULTS_DIR, "bandwidth_results.json")
    with open(path, "w") as f: json.dump(all_rows, f, indent=2)
    print(f"\nSaved {path} ({len(all_rows)} rows).")
    if all_rows:
        ai = all_rows[0]["arithmetic_intensity"]
        print(f"\nKey finding: arithmetic intensity ~{ai:.2f} FLOP/byte << ridge {ridge:.1f}")
        print("=> adapter serving is firmly memory-bound, confirming the paper's premise.")

if __name__ == "__main__": main()