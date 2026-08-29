"""
exp5_baseline.py — Baseline multi-tenant performance characterization.

Reports the cost of NAIVE multi-tenant LoRA serving (sequential application,
uniform rank) before any optimization. This is the reference point the paper's
contributions improve on. Measures, as tenants scale:
    - total adapter latency (ms)
    - per-tenant latency (ms)
    - VRAM (GB)

Run:  python -m experiments.exp5_baseline --model llama
      python -m experiments.exp5_baseline --model all
"""
import sys, os, json, argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from config import MODELS, TENANT_COUNTS, RESULTS_DIR, gpu_name
from core.lora_model import MultiTenantLoRA
from utils.benchmark import time_adapter, vram_gb

BASELINE_RANK = 8   # fixed rank for the baseline reference

def run(model_key, cfg):
    print(f"\n{'='*65}\n  BASELINE — {cfg['name']} ({cfg['company']})\n{'='*65}")
    if not os.path.exists(cfg["path"]):
        print(f"  X Path not found: {cfg['path']}"); return []
    rows = []
    for T in TENANT_COUNTS:
        try:
            m = MultiTenantLoRA(cfg["path"], T, BASELINE_RANK, cfg["dtype"])
            h = m.compute_hidden_state()
            # Baseline = sequential application (naive multi-tenant serving)
            total = time_adapter(m, h, "sequential")
            v = vram_gb(); m.cleanup()
            per_tenant = total / T
            print(f"  T={T:<4} total={total:8.3f}ms  per-tenant={per_tenant:.4f}ms  VRAM={v:.2f}GB")
            rows.append({"contribution":"baseline","model_key":model_key,"model":cfg["name"],
                "company":cfg["company"],"num_tenants":T,"lora_rank":BASELINE_RANK,
                "total_ms":round(total,3),"per_tenant_ms":round(per_tenant,4),
                "vram_gb":round(v,2)})
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  OOM at T={T}"); torch.cuda.empty_cache(); break
            raise
    return rows

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", choices=list(MODELS)+["all"], default="all")
    args = ap.parse_args(); print(f"GPU: {gpu_name()}")
    keys = list(MODELS) if args.model=="all" else [args.model]
    allr = []
    for k in keys: allr += run(k, MODELS[k])

    out = os.path.join(RESULTS_DIR, f"exp5_baseline_{args.model}.json")
    with open(out,"w") as f: json.dump(allr, f, indent=2)
    print(f"\nSaved {out}")

    # Merge per-model files
    combined = []
    for mk in MODELS:
        p = os.path.join(RESULTS_DIR, f"exp5_baseline_{mk}.json")
        if os.path.exists(p):
            with open(p) as f: combined += json.load(f)
    with open(os.path.join(RESULTS_DIR, "exp5_baseline.json"), "w") as f:
        json.dump(combined, f, indent=2)
    n = sum(1 for mk in MODELS if os.path.exists(os.path.join(RESULTS_DIR, f"exp5_baseline_{mk}.json")))
    print(f"Merged {n} model(s) into exp5_baseline.json")

if __name__ == "__main__": main()