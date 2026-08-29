"""
exp2_rank_split.py — Contribution 2 (REFRAMED): Rank-Aware Batch Splitting.

The dead idea: "sequential vs fused with theta=12" (fused always wins, so no threshold).
The real idea: when a batch MIXES cheap (low-rank) and expensive (high-rank) tenants,
padding everyone to the max rank wastes the high-rank penalty on tenants that don't need it.
Splitting into a low-rank group and a high-rank group — each fused — avoids that waste.

Compares, on a mixed-rank batch:
  (a) PAD-AND-FUSE : pad all tenants to max rank, one fused batch
  (b) RANK-SPLIT   : group by rank (low vs high), one fused batch per group

Run:  python -m experiments.exp2_rank_split --model llama
"""
import sys, os, json, argparse, random
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from config import MODELS, TENANT_COUNTS, RESULTS_DIR, gpu_name
from core.lora_model import MultiTenantLoRA
from utils.benchmark import time_adapter, vram_gb
random.seed(42)

# The "knee": ranks at/below this are cheap; above are expensive (from rank sweep).
KNEE = 32
LOW_RANK  = 8     # cheap tenants
HIGH_RANK = 128   # expensive tenants

def make_mixed_ranks(T, high_fraction=0.25):
    """A realistic mixed batch: most tenants low-rank, some high-rank."""
    n_high = max(1, int(T * high_fraction))
    ranks = [HIGH_RANK]*n_high + [LOW_RANK]*(T - n_high)
    random.shuffle(ranks)
    return ranks

def time_pad_and_fuse(cfg, T, ranks):
    """Pad ALL tenants up to the max rank, run one fused batch."""
    max_r = max(ranks)
    m = MultiTenantLoRA(cfg["path"], T, max_r, cfg["dtype"])  # uniform = max rank
    h = m.compute_hidden_state()
    ms = time_adapter(m, h, "fused")
    m.cleanup()
    return ms

def time_rank_split(cfg, T, ranks):
    """Split into low-rank group and high-rank group; fuse each; sum times."""
    lows  = [r for r in ranks if r <= KNEE]
    highs = [r for r in ranks if r >  KNEE]
    total = 0.0
    for group, gr in [(lows, LOW_RANK), (highs, HIGH_RANK)]:
        if not group:
            continue
        m = MultiTenantLoRA(cfg["path"], len(group), gr, cfg["dtype"])
        h = m.compute_hidden_state()
        total += time_adapter(m, h, "fused")
        m.cleanup()
    return total

def run(model_key, cfg):
    print(f"\n{'='*65}\n  C2: RANK-AWARE SPLITTING — {cfg['name']}\n{'='*65}")
    if not os.path.exists(cfg["path"]):
        print(f"  X Path not found: {cfg['path']}"); return []
    rows = []
    for T in TENANT_COUNTS:
        try:
            ranks = make_mixed_ranks(T)
            n_high = sum(1 for r in ranks if r > KNEE)
            pad = time_pad_and_fuse(cfg, T, ranks)
            split = time_rank_split(cfg, T, ranks)
            speedup = pad / split
            print(f"  T={T:<4} pad_and_fuse={pad:7.3f}ms  rank_split={split:7.3f}ms  "
                  f"speedup={speedup:.2f}x  (high={n_high}/{T})")
            rows.append({"contribution":"C2_rank_split","model_key":model_key,"model":cfg["name"],
                "company":cfg["company"],"num_tenants":T,"n_high_rank":n_high,
                "pad_and_fuse_ms":round(pad,3),"rank_split_ms":round(split,3),
                "speedup":round(speedup,2),"vram_gb":round(vram_gb(),2)})
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
    out = os.path.join(RESULTS_DIR, "exp2_rank_split.json")
    with open(out,"w") as f: json.dump(allr, f, indent=2)
    print(f"\nSaved {out}")

if __name__ == "__main__": main()
