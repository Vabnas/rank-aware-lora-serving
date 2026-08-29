"""exp0_rank_sweep.py — Motivation data for C3. Times ADAPTER ONLY."""
import sys, os, json, argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from config import MODELS, TENANT_COUNTS, RANK_COUNTS, RESULTS_DIR, gpu_name
from core.lora_model import MultiTenantLoRA
from utils.benchmark import time_adapter, vram_gb

def run(model_key, cfg):
    print(f"\n{'='*65}\n  RANK SWEEP — {cfg['name']} ({cfg['company']})\n{'='*65}")
    if not os.path.exists(cfg["path"]):
        print(f"  X Path not found: {cfg['path']}"); return []
    rows = []
    for T in TENANT_COUNTS:
        for r in RANK_COUNTS:
            try:
                model = MultiTenantLoRA(cfg["path"], T, r, cfg["dtype"])
                h = model.compute_hidden_state()
                fused = time_adapter(model, h, "fused")
                v = vram_gb(); model.cleanup()
                print(f"  T={T:<3} r={r:<4} adapter={fused:.5f}ms  VRAM={v:.1f}GB")
                rows.append({"contribution":"motivation_rank_sweep","model_key":model_key,
                    "model":cfg["name"],"company":cfg["company"],"num_tenants":T,"lora_rank":r,
                    "adapter_ms":round(fused,5),"per_tenant_ms":round(fused/T,6),"vram_gb":round(v,2)})
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"  OOM at T={T} r={r}"); torch.cuda.empty_cache(); continue
                raise
    return rows

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", choices=list(MODELS)+["all"], default="all")
    args = ap.parse_args(); print(f"GPU: {gpu_name()}")
    keys = list(MODELS) if args.model=="all" else [args.model]
    allr = []
    for k in keys: allr += run(k, MODELS[k])
    out = os.path.join(RESULTS_DIR, "exp0_rank_sweep.json")
    with open(out,"w") as f: json.dump(allr, f, indent=2)
    print(f"\nSaved {out}")

if __name__ == "__main__": main()
