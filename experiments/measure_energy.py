"""
measure_energy.py -- Energy per request for sequential vs fused adapter serving,
using nvidia-smi power sampling.

Energy = average power over the timed window x elapsed time. We sample GPU
power in a background thread while running each strategy many times, then
divide total energy by the number of requests served.

Run from the MAIN folder:  python measure_energy.py
Output: results/energy_results.json + summary table.
Requires nvidia-smi on PATH (ships with the NVIDIA driver).
"""
import sys, os, json, time, threading, subprocess
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
from config import MODELS, RESULTS_DIR, TENANT_COUNTS, gpu_name
from core.lora_model import MultiTenantLoRA

RANK = 8
DTYPE = torch.float16
ITERS = 200   # many iters so the power sampler captures a stable window

class PowerSampler(threading.Thread):
    def __init__(self, interval=0.01):
        super().__init__(daemon=True)
        self.interval = interval; self.samples = []
        self._stop = threading.Event()
    def run(self):
        while not self._stop.is_set():
            try:
                out = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=power.draw",
                     "--format=csv,noheader,nounits"], timeout=1.0).decode().strip()
                self.samples.append(float(out.splitlines()[0]))
            except Exception:
                pass
            time.sleep(self.interval)
    def stop(self): self._stop.set()

def measure(fn, n_requests):
    torch.cuda.synchronize()
    sampler = PowerSampler(0.01); sampler.start()
    t0 = time.time()
    for _ in range(ITERS): fn()
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    sampler.stop(); sampler.join(timeout=1.0)
    if not sampler.samples: return None, None
    avg_power = float(np.mean(sampler.samples))     # watts
    energy_per_request = (avg_power * elapsed) / (ITERS * n_requests)  # joules
    return energy_per_request, avg_power

def run_model(model_key):
    cfg = MODELS[model_key]
    if not os.path.exists(cfg["path"]):
        print(f"  skip {model_key}: path not found"); return []
    print(f"\n{'='*60}\n  ENERGY -- {cfg['name']}\n{'='*60}")
    rows = []
    for T in TENANT_COUNTS:
        mt = MultiTenantLoRA(cfg["path"], num_tenants=T, lora_rank=RANK, dtype=DTYPE)
        h = mt.compute_hidden_state()
        for _ in range(10): mt.adapter_fused(h)  # warmup
        e_seq, p_seq = measure(lambda: mt.adapter_sequential(h), T)
        e_fused, p_fused = measure(lambda: mt.adapter_fused(h), T)
        if e_seq is None:
            print("  ! no power samples -- is nvidia-smi available?")
            mt.cleanup(); del mt; torch.cuda.empty_cache(); continue
        saving = 100.0 * (1 - e_fused / e_seq)
        rows.append({
            "model_key": model_key, "model": cfg["name"], "num_tenants": T,
            "energy_seq_mJ": round(e_seq*1000, 4),
            "energy_fused_mJ": round(e_fused*1000, 4),
            "energy_saving_pct": round(saving, 1),
            "avg_power_seq_W": round(p_seq, 1),
            "avg_power_fused_W": round(p_fused, 1),
        })
        print(f"  T={T:<4} seq={e_seq*1000:.3f}mJ  fused={e_fused*1000:.3f}mJ  "
              f"saving={saving:.1f}%  (P~{p_fused:.0f}W)")
        mt.cleanup(); del mt; torch.cuda.empty_cache()
    return rows

def main():
    print(f"GPU: {gpu_name()}   energy measurement, {ITERS} iters/config")
    all_rows = []
    for mk in ["llama", "deepseek", "gemma"]:
        all_rows += run_model(mk)
    path = os.path.join(RESULTS_DIR, "energy_results.json")
    with open(path, "w") as f: json.dump(all_rows, f, indent=2)
    print(f"\nSaved {path} ({len(all_rows)} rows).")

if __name__ == "__main__": main()