"""exp3_adaptive_rank.py — Contribution 3. Adaptive vs uniform ranks, ADAPTER ONLY."""
import sys, os, json, argparse, random
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from config import MODELS, TENANT_COUNTS, QUALITY_TIERS, RESULTS_DIR, gpu_name
from core.lora_model import MultiTenantLoRA
from utils.benchmark import time_adapter, vram_gb
random.seed(42)

def assign_tiers(num_tenants):
    weights = {"low":0.50, "medium":0.35, "high":0.15}
    tiers = random.choices(list(weights), weights=list(weights.values()), k=num_tenants)
    ranks = [QUALITY_TIERS[t] for t in tiers]
    counts = {t: tiers.count(t) for t in weights}
    return ranks, counts

def run(model_key, cfg):
    print(f"\n{'='*65}\n  C3: ADAPTIVE RANK ALLOCATION — {cfg['name']}\n{'='*65}")
    if not os.path.exists(cfg["path"]):
        print(f"  X Path not found: {cfg['path']}"); return []
    rows = []
    for T in TENANT_COUNTS:
        try:
            m_high = MultiTenantLoRA(cfg["path"], T, 32, cfg["dtype"])
            h = m_high.compute_hidden_state()
            high_ms = time_adapter(m_high, h, "sequential"); m_high.cleanup()

            m_low = MultiTenantLoRA(cfg["path"], T, 4, cfg["dtype"])
            h = m_low.compute_hidden_state()
            low_ms = time_adapter(m_low, h, "sequential"); m_low.cleanup()

            ranks, counts = assign_tiers(T); avg_rank = sum(ranks)/len(ranks)
            m_ad = MultiTenantLoRA(cfg["path"], T, None, cfg["dtype"], ranks_per_tenant=ranks)
            h = m_ad.compute_hidden_state()
            adapt_ms = time_adapter(m_ad, h, "adaptive")
            v = vram_gb(); m_ad.cleanup()

            speedup = high_ms/adapt_ms
            print(f"  T={T:<3} high(r32)={high_ms:.5f} low(r4)={low_ms:.5f} adaptive={adapt_ms:.5f}ms avg_rank={avg_rank:.1f} speedup_vs_high={speedup:.2f}x")
            print(f"        tier mix: {counts}")
            rows.append({"contribution":"C3_adaptive_rank","model_key":model_key,"model":cfg["name"],
                "company":cfg["company"],"num_tenants":T,"uniform_high_r32_ms":round(high_ms,5),
                "uniform_low_r4_ms":round(low_ms,5),"adaptive_ms":round(adapt_ms,5),
                "avg_rank":round(avg_rank,2),"tier_counts":counts,
                "speedup_vs_high":round(speedup,2),"vram_gb":round(v,2)})
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
    out = os.path.join(RESULTS_DIR, "exp3_adaptive_rank.json")
    with open(out,"w") as f: json.dump(allr, f, indent=2)
    print(f"\nSaved {out}")

if __name__ == "__main__": main()
