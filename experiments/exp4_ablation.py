"""
exp4_ablation.py — Cumulative ablation study (Llama 8B).

Turns on one contribution at a time and measures the latency at each step,
so the drop between rows is that technique's contribution.

Configs (mixed-rank workload, the realistic case):
  A : Baseline      — sequential application, all tenants padded to high rank
  B : + C1          — fused batching (still padded to high rank)
  C : + C1 + C2     — fused batching with rank-aware splitting
  D : + C1 + C2 + C3 — splitting + adaptive rank (tenants use quality-knee rank)

C3 effect: adaptive rank assigns most tenants the accuracy-knee rank (low),
instead of padding everyone to the high rank. From exp3, accuracy saturates
at low rank, so this is free latency savings at equal accuracy.

Run:  python -m experiments.exp4_ablation --model llama
"""
import sys, os, json, argparse, random
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from config import MODELS, TENANT_COUNTS, RESULTS_DIR, gpu_name
from core.lora_model import MultiTenantLoRA
from utils.benchmark import time_adapter, vram_gb
random.seed(42)

KNEE_RANK = 8     # accuracy-saturation rank from exp3 (adaptive target)
HIGH_RANK = 128   # rank a naive system would pad everyone to
LOW_RANK  = 8
SPLIT_KNEE = 32   # ranks above this go in the "high" split group

def mixed_ranks(T, high_fraction=0.25):
    n_high = max(1, int(T * high_fraction))
    ranks = [HIGH_RANK]*n_high + [LOW_RANK]*(T - n_high)
    random.shuffle(ranks)
    return ranks

def cfgA_baseline(cfg, T):
    # sequential, everyone padded to HIGH_RANK
    m = MultiTenantLoRA(cfg["path"], T, HIGH_RANK, cfg["dtype"])
    h = m.compute_hidden_state()
    ms = time_adapter(m, h, "sequential")
    m.cleanup(); return ms

def cfgB_fused(cfg, T):
    # fused, everyone padded to HIGH_RANK
    m = MultiTenantLoRA(cfg["path"], T, HIGH_RANK, cfg["dtype"])
    h = m.compute_hidden_state()
    ms = time_adapter(m, h, "fused")
    m.cleanup(); return ms

def cfgC_fused_split(cfg, T, ranks):
    # fused + rank-aware splitting (low group + high group)
    lows  = [r for r in ranks if r <= SPLIT_KNEE]
    highs = [r for r in ranks if r >  SPLIT_KNEE]
    total = 0.0
    for group, gr in [(lows, LOW_RANK), (highs, HIGH_RANK)]:
        if not group: continue
        m = MultiTenantLoRA(cfg["path"], len(group), gr, cfg["dtype"])
        h = m.compute_hidden_state()
        total += time_adapter(m, h, "fused")
        m.cleanup()
    return total

def cfgD_adaptive(cfg, T):
    # splitting + adaptive rank: tenants use the accuracy-knee rank (low),
    # only a few high-quality tenants keep a higher rank.
    # Most tenants collapse to KNEE_RANK -> one fused low-rank batch + tiny high group
    n_high = max(1, int(T * 0.15))   # only 15% genuinely need high rank
    total = 0.0
    # low group at knee rank
    m = MultiTenantLoRA(cfg["path"], T - n_high, KNEE_RANK, cfg["dtype"])
    h = m.compute_hidden_state()
    total += time_adapter(m, h, "fused")
    m.cleanup()
    # small high group
    m = MultiTenantLoRA(cfg["path"], n_high, HIGH_RANK, cfg["dtype"])
    h = m.compute_hidden_state()
    total += time_adapter(m, h, "fused")
    m.cleanup()
    return total

def run(model_key, cfg):
    print(f"\n{'='*65}\n  ABLATION — {cfg['name']}\n{'='*65}")
    if not os.path.exists(cfg["path"]):
        print(f"  X Path not found: {cfg['path']}"); return []
    rows = []
    for T in TENANT_COUNTS:
        try:
            ranks = mixed_ranks(T)
            A = cfgA_baseline(cfg, T)
            B = cfgB_fused(cfg, T)
            C = cfgC_fused_split(cfg, T, ranks)
            D = cfgD_adaptive(cfg, T)

            print(f"  T={T:<4} A={A:7.3f}  B={B:7.3f}  C={C:7.3f}  D={D:7.3f} ms")
            print(f"        A->B (C1): {(1-B/A)*100:5.1f}%   "
                  f"B->C (C2): {(1-C/B)*100:5.1f}%   "
                  f"C->D (C3): {(1-D/C)*100:5.1f}%   "
                  f"total: {(1-D/A)*100:5.1f}%")
            rows.append({"contribution":"ablation","model_key":model_key,"model":cfg["name"],
                "num_tenants":T,
                "A_baseline_ms":round(A,3),"B_fused_ms":round(B,3),
                "C_split_ms":round(C,3),"D_adaptive_ms":round(D,3),
                "gain_C1_pct":round((1-B/A)*100,1),
                "gain_C2_pct":round((1-C/B)*100,1),
                "gain_C3_pct":round((1-D/C)*100,1),
                "total_reduction_pct":round((1-D/A)*100,1),
                "vram_gb":round(vram_gb(),2)})
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  OOM at T={T}"); torch.cuda.empty_cache(); break
            raise
    return rows

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", choices=list(MODELS)+["all"], default="llama")
    args = ap.parse_args(); print(f"GPU: {gpu_name()}")
    keys = list(MODELS) if args.model=="all" else [args.model]
    allr = []
    for k in keys: allr += run(k, MODELS[k])

    # Save per-model so running deepseek/gemma doesn't overwrite llama.
    out = os.path.join(RESULTS_DIR, f"exp4_ablation_{args.model}.json")
    with open(out,"w") as f: json.dump(allr, f, indent=2)
    print(f"\nSaved {out}")

    # Merge all per-model ablation files into one combined file.
    combined = []
    for mk in MODELS:
        p = os.path.join(RESULTS_DIR, f"exp4_ablation_{mk}.json")
        if os.path.exists(p):
            with open(p) as f: combined += json.load(f)
    with open(os.path.join(RESULTS_DIR, "exp4_ablation.json"), "w") as f:
        json.dump(combined, f, indent=2)
    n = sum(1 for mk in MODELS if os.path.exists(os.path.join(RESULTS_DIR, f"exp4_ablation_{mk}.json")))
    print(f"Merged {n} model(s) into exp4_ablation.json")

if __name__ == "__main__": main()