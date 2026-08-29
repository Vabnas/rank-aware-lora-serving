"""
cross_model_eval.py — Consolidated cross-model evaluation.

Reads from whichever result files exist and builds:
  1. A consolidated TABLE (printed + CSV) of C1 speedup across models.
  2. Cross-model FIGURES for C1 and C2 speedup.

Robust to the per-model vs combined file situation: it prefers per-model
files (results_<model>.json), falls back to all_results.json, and also
reads exp2_rank_split.json (which already has all 3 models) for C2.

Run:  python cross_model_eval.py
"""
import os, json, csv
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # silence the OpenMP warning on Windows
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import RESULTS_DIR, TENANT_COUNTS, MODELS

matplotlib.rcParams.update({"font.family": "serif", "font.size": 11})
COLORS = {"llama": "#4C72B0", "deepseek": "#DD8452", "gemma": "#55A868"}
LABELS = {"llama": "Llama 3 8B (Meta)",
          "deepseek": "DeepSeek-R1 7B (DeepSeek)",
          "gemma": "Gemma 2 9B (Google)"}

def load(name):
    p = os.path.join(RESULTS_DIR, name)
    if not os.path.exists(p): return None
    with open(p) as f: return json.load(f)

# ---- Gather C1 from every available source ----
c1 = []
# 1) per-model results_<model>.json
for mk in MODELS:
    d = load(f"results_{mk}.json")
    if d and "c1" in d:
        c1 += d["c1"]
# 2) fallback: all_results.json (may only have last model run)
if not c1:
    d = load("all_results.json")
    if d and "c1" in d: c1 = d["c1"]
# de-dup by (model_key, tenants)
seen = set(); c1u = []
for r in c1:
    k = (r["model_key"], r["num_tenants"])
    if k not in seen:
        seen.add(k); c1u.append(r)
c1 = c1u

split = load("exp2_rank_split.json") or []

def mkeys(rows):
    out = []
    for r in rows:
        if r["model_key"] not in out: out.append(r["model_key"])
    return out

# ---- 1. Table ----
print("\n" + "="*70)
print("CROSS-MODEL EVALUATION — C1 Fused Batching Speedup")
print("="*70)
keys = mkeys(c1)
if not keys:
    print("No C1 data found. Re-run: python run_all.py --model <each model>")
else:
    header = f"{'Tenants':<10}" + "".join(f"{LABELS.get(k,k):<28}" for k in keys)
    print(header); print("-"*len(header))
    rows_csv = []
    for T in TENANT_COUNTS:
        line = f"{T:<10}"; row = {"tenants": T}
        for k in keys:
            m = next((r for r in c1 if r["model_key"]==k and r["num_tenants"]==T), None)
            if m:
                line += f"{m['speedup']:.2f}x ({m['reduction_pct']:.0f}%)".ljust(28)
                row[k] = m["speedup"]
        print(line); rows_csv.append(row)
    with open(os.path.join(RESULTS_DIR, "cross_model_c1.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["tenants"]+keys)
        for r in rows_csv: w.writerow([r["tenants"]]+[r.get(k,"") for k in keys])
    print(f"\nSaved table -> cross_model_c1.csv")

# ---- 2. C1 figure ----
if keys:
    plt.figure(figsize=(7,5))
    for k in keys:
        rows = sorted([r for r in c1 if r["model_key"]==k], key=lambda x:x["num_tenants"])
        plt.plot([r["num_tenants"] for r in rows], [r["speedup"] for r in rows],
                 marker="o", color=COLORS.get(k,"#444"), linewidth=2, label=LABELS.get(k,k))
    plt.title("C1 Fused Batching Speedup Across Models")
    plt.xlabel("Number of Tenants"); plt.ylabel("Speedup vs Sequential (x)")
    plt.xticks(TENANT_COUNTS); plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
    f1 = os.path.join(RESULTS_DIR, "fig_crossmodel_c1.pdf")
    plt.savefig(f1, dpi=300, bbox_inches="tight")
    plt.savefig(f1.replace(".pdf",".png"), dpi=300, bbox_inches="tight"); plt.close()
    print("Saved fig_crossmodel_c1.pdf")

# ---- 3. C2 figure ----
if split:
    sk = mkeys(split)
    plt.figure(figsize=(7,5))
    for k in sk:
        rows = sorted([r for r in split if r["model_key"]==k], key=lambda x:x["num_tenants"])
        plt.plot([r["num_tenants"] for r in rows], [r["speedup"] for r in rows],
                 marker="s", color=COLORS.get(k,"#444"), linewidth=2, label=LABELS.get(k,k))
    plt.axhline(1.0, color="gray", linestyle="--", alpha=0.5)
    plt.title("C2 Rank-Aware Splitting Speedup Across Models")
    plt.xlabel("Number of Tenants"); plt.ylabel("Speedup vs Pad-and-Fuse (x)")
    plt.xticks(TENANT_COUNTS); plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
    f2 = os.path.join(RESULTS_DIR, "fig_crossmodel_c2.pdf")
    plt.savefig(f2, dpi=300, bbox_inches="tight")
    plt.savefig(f2.replace(".pdf",".png"), dpi=300, bbox_inches="tight"); plt.close()
    print("Saved fig_crossmodel_c2.pdf")

print("\nDone.")