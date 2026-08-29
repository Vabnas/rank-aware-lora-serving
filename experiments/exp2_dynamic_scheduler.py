"""exp2_dynamic_scheduler.py — Contribution 2. Static vs dynamic, ADAPTER ONLY."""
import sys, os, json, argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from config import MODELS, TENANT_COUNTS, SCHEDULER_THRESHOLD, RESULTS_DIR, gpu_name
from core.lora_model import MultiTenantLoRA
from utils.benchmark import time_adapter, vram_gb

def run(model_key, cfg, fixed_rank=8, threshold=SCHEDULER_THRESHOLD):
    print(f"\n{'='*65}\n  C2: DYNAMIC SCHEDULER (theta={threshold}) — {cfg['name']}\n{'='*65}")
    if not os.path.exists(cfg["path"]):
        print(f"  X Path not found: {cfg['path']}"); return []
    rows = []
    for T in TENANT_COUNTS:
        try:
            model = MultiTenantLoRA(cfg["path"], T, fixed_rank, cfg["dtype"])
            h = model.compute_hidden_state()
            seq     = time_adapter(model, h, "sequential")
            fused   = time_adapter(model, h, "fused")
            dynamic = time_adapter(model, h, "dynamic", threshold=threshold)
            v = vram_gb(); model.cleanup()
            chosen = "fused" if T>=threshold else "sequential"
            improvement = (1-dynamic/fused)*100
            print(f"  T={T:<3} seq={seq:.5f} fused={fused:.5f} dynamic={dynamic:.5f}ms chose={chosen} vs_fused={improvement:+.1f}%")
            rows.append({"contribution":"C2_dynamic_scheduler","model_key":model_key,"model":cfg["name"],
                "company":cfg["company"],"num_tenants":T,"lora_rank":fixed_rank,"threshold":threshold,
                "sequential_ms":round(seq,5),"fused_ms":round(fused,5),"dynamic_ms":round(dynamic,5),
                "chosen":chosen,"improvement_vs_fused_pct":round(improvement,1),"vram_gb":round(v,2)})
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  OOM at T={T}"); torch.cuda.empty_cache(); break
            raise
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS)+["all"], default="all")
    ap.add_argument("--threshold", type=int, default=SCHEDULER_THRESHOLD)
    args = ap.parse_args(); print(f"GPU: {gpu_name()}")
    keys = list(MODELS) if args.model=="all" else [args.model]
    allr = []
    for k in keys: allr += run(k, MODELS[k], threshold=args.threshold)
    out = os.path.join(RESULTS_DIR, "exp2_dynamic_scheduler.json")
    with open(out,"w") as f: json.dump(allr, f, indent=2)
    print(f"\nSaved {out}")

if __name__ == "__main__": main()
