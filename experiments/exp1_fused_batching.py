"""exp1_fused_batching.py — Contribution 1. Sequential vs fused, ADAPTER ONLY."""
import sys, os, json, argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from config import MODELS, TENANT_COUNTS, RESULTS_DIR, gpu_name
from core.lora_model import MultiTenantLoRA
from utils.benchmark import time_adapter, vram_gb

def run(model_key, cfg, fixed_rank=8):
    print(f"\n{'='*65}\n  C1: FUSED BATCHING — {cfg['name']} ({cfg['company']})\n{'='*65}")
    if not os.path.exists(cfg["path"]):
        print(f"  X Path not found: {cfg['path']}"); return []
    rows = []
    for T in TENANT_COUNTS:
        try:
            model = MultiTenantLoRA(cfg["path"], T, fixed_rank, cfg["dtype"])
            h = model.compute_hidden_state()
            seq   = time_adapter(model, h, "sequential")
            fused = time_adapter(model, h, "fused")
            v = vram_gb(); model.cleanup()
            speedup = seq/fused; reduction = (1-fused/seq)*100
            print(f"  T={T:<3} seq={seq:.5f}ms  fused={fused:.5f}ms  speedup={speedup:.2f}x  reduction={reduction:.1f}%")
            rows.append({"contribution":"C1_fused_batching","model_key":model_key,"model":cfg["name"],
                "company":cfg["company"],"num_tenants":T,"lora_rank":fixed_rank,
                "sequential_ms":round(seq,5),"fused_ms":round(fused,5),
                "speedup":round(speedup,2),"reduction_pct":round(reduction,1),"vram_gb":round(v,2)})
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
    out = os.path.join(RESULTS_DIR, "exp1_fused_batching.json")
    with open(out,"w") as f: json.dump(allr, f, indent=2)
    print(f"\nSaved {out}")

if __name__ == "__main__": main()
