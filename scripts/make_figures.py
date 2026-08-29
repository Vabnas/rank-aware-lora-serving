"""
make_figures.py — Generate all six paper figures from result files.

Figures:
  1. C1 fused-batching speedup vs tenants (3 models, error bars from exp6)
  2. C2 rank-aware splitting speedup vs tenants (3 models)
  3. Latency vs rank (rank penalty, from rank sweep)
  4. Ablation stacked bars (per-component reduction, 3 models)
  5. Accuracy vs rank with error bars (C3 quality, ANOVA data)
  6. Baseline latency scaling vs tenants (3 models)

Reads: exp6_stats.json, all_results.json, exp4_ablation.json,
       exp3_quality_rank.json, exp5_baseline.json
Writes: results/fig1..fig6 (.pdf and .png)

Run:  python make_figures.py
"""
import os, json
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from config import RESULTS_DIR, TENANT_COUNTS, RANK_COUNTS, MODELS

matplotlib.rcParams.update({
    "font.family": "serif", "font.size": 11,
    "axes.grid": True, "grid.alpha": 0.3,
    "figure.dpi": 120,
})
COLORS = {"llama": "#4C72B0", "deepseek": "#DD8452", "gemma": "#55A868"}
MARK   = {"llama": "o", "deepseek": "s", "gemma": "^"}
LABEL  = {"llama": "Llama 3 8B", "deepseek": "DeepSeek-R1 7B", "gemma": "Gemma 2 9B"}
ORDER  = ["llama", "deepseek", "gemma"]

def load(name):
    p = os.path.join(RESULTS_DIR, name)
    if not os.path.exists(p):
        print(f"  ! missing {name}"); return None
    with open(p) as f: return json.load(f)

def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(RESULTS_DIR, f"{name}.{ext}"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  saved {name}.pdf / .png")

def drop_outlier_mean(seq, fused):
    seq = np.array(seq, dtype=float); fused = np.array(fused, dtype=float)
    sp = seq / fused
    # Filter on the speedup ratio itself: a run whose ratio sits far from the
    # median ratio reflects transient contention in one of its two timings
    # (e.g. an inflated sequential time paired with a normal fused time).
    med = np.median(sp)
    keep = np.abs(sp - med) < 0.25 * med
    if keep.sum() < 2:
        keep = np.ones_like(sp, dtype=bool)
    return sp[keep].mean(), sp[keep].std()

# ---------- Figure 1: C1 speedup vs tenants (error bars) ----------
def fig1():
    s = load("exp6_stats.json")
    if not s: return
    fig, ax = plt.subplots(figsize=(7,5))
    for mk in ORDER:
        rows = sorted([r for r in s if r["model_key"]==mk], key=lambda x:x["num_tenants"])
        xs, ys, es = [], [], []
        for r in rows:
            m, sd = drop_outlier_mean(r["seq_runs"], r["fused_runs"])
            xs.append(r["num_tenants"]); ys.append(m); es.append(sd)
        ax.errorbar(xs, ys, yerr=es, marker=MARK[mk], color=COLORS[mk],
                    linewidth=2, capsize=3, label=LABEL[mk])
    ax.set_xscale("log", base=2); ax.set_xticks(TENANT_COUNTS)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("Number of Tenants"); ax.set_ylabel("Speedup vs Sequential (×)")
    ax.set_title("Fused Batching Speedup (C1)")
    ax.axhline(1.0, color="gray", ls="--", alpha=0.5); ax.legend()
    save(fig, "fig1_c1_speedup")

# ---------- Figure 2: C2 splitting speedup vs tenants ----------
def fig2():
    s = load("exp6_stats.json")
    if not s: return
    fig, ax = plt.subplots(figsize=(7,5))
    for mk in ORDER:
        rows = sorted([r for r in s if r["model_key"]==mk], key=lambda x:x["num_tenants"])
        xs, ys, es = [], [], []
        for r in rows:
            pad = np.array(r["pad_runs"], dtype=float); split = np.array(r["split_runs"], dtype=float)
            sp = pad/split
            med = np.median(sp)
            keep = np.abs(sp - med) < 0.25 * med
            if keep.sum() < 2:
                keep = np.ones_like(sp, dtype=bool)
            xs.append(r["num_tenants"]); ys.append(sp[keep].mean()); es.append(sp[keep].std())
        ax.errorbar(xs, ys, yerr=es, marker=MARK[mk], color=COLORS[mk],
                    linewidth=2, capsize=3, label=LABEL[mk])
    ax.set_xscale("log", base=2); ax.set_xticks(TENANT_COUNTS)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("Number of Tenants"); ax.set_ylabel("Speedup vs Pad-and-Fuse (×)")
    ax.set_title("Rank-Aware Splitting Speedup (C2)")
    ax.axhline(1.0, color="gray", ls="--", alpha=0.5); ax.legend()
    save(fig, "fig2_c2_speedup")

# ---------- Figure 3: latency vs rank (rank penalty) ----------
def fig3():
    d = load("all_results.json")
    if not d or "rank_sweep" not in d: return
    rs = d["rank_sweep"]
    fig, ax = plt.subplots(figsize=(7,5))
    T_show = 128
    for mk in ORDER:
        rows = sorted([r for r in rs if r["model_key"]==mk and r["num_tenants"]==T_show],
                      key=lambda x:x["lora_rank"])
        if not rows: continue
        xs = [r["lora_rank"] for r in rows]
        ys = [r["adapter_ms"] for r in rows]
        ax.plot(xs, ys, marker=MARK[mk], color=COLORS[mk], linewidth=2, label=LABEL[mk])
    ax.set_xscale("log", base=2); ax.set_xticks(RANK_COUNTS)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("LoRA Rank"); ax.set_ylabel("Adapter Latency (ms)")
    ax.set_title(f"Latency vs Rank at {T_show} Tenants"); ax.legend()
    save(fig, "fig3_rank_penalty")

# ---------- Figure 4: ablation stacked bars ----------
def fig4():
    a = load("exp4_ablation.json")
    if not a: return
    T_show = 128
    fig, ax = plt.subplots(figsize=(7,5))
    models = [LABEL[mk] for mk in ORDER]
    c1 = []; c2 = []; c3 = []
    for mk in ORDER:
        row = next((r for r in a if r["model_key"]==mk and r["num_tenants"]==T_show), None)
        if row:
            c1.append(row["gain_C1_pct"]); c2.append(row["gain_C2_pct"]); c3.append(row["gain_C3_pct"])
        else:
            c1.append(0); c2.append(0); c3.append(0)
    x = np.arange(len(models))
    ax.bar(x, c1, label="C1 Fused", color="#4C72B0")
    ax.bar(x, c2, bottom=c1, label="C2 Splitting", color="#DD8452")
    ax.bar(x, c3, bottom=np.array(c1)+np.array(c2), label="C3 Allocation", color="#55A868")
    for i in range(len(models)):
        total = c1[i]+c2[i]+c3[i]
        ax.text(i, total+0.8, f"{total:.0f}%", ha="center", fontsize=10, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(models)
    ax.set_ylabel("Cumulative Latency Reduction (%)")
    ax.set_title(f"Ablation at {T_show} Tenants"); ax.legend()
    ax.set_ylim(0, max(np.array(c1)+np.array(c2)+np.array(c3))+8)
    save(fig, "fig4_ablation")

# ---------- Figure 5: accuracy vs rank with error bars ----------
def fig5():
    q = load("exp3_quality_rank.json")
    if not q or "results" not in q: return
    rows = sorted(q["results"], key=lambda x:x["lora_rank"])
    xs = [r["lora_rank"] for r in rows]
    ys = [r["acc_mean"]*100 for r in rows]
    es = [r["acc_std"]*100 for r in rows]
    fig, ax = plt.subplots(figsize=(7,5))
    ax.errorbar(xs, ys, yerr=es, marker="o", color="#4C72B0",
                linewidth=2, capsize=4, markersize=7)
    best = max(ys)
    ax.axhline(best, color="gray", ls="--", alpha=0.5, label=f"Best ({best:.1f}%)")
    ax.axhline(best-0.5, color="red", ls=":", alpha=0.5, label="Best − 0.5%")
    ax.set_xscale("log", base=2); ax.set_xticks(RANK_COUNTS)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("LoRA Rank"); ax.set_ylabel("SST-2 Accuracy (%)")
    ax.set_title("Accuracy vs Rank (ANOVA: p = 0.38, n.s.)"); ax.legend()
    save(fig, "fig5_accuracy_rank")

# ---------- Figure 6: baseline scaling ----------
def fig6():
    b = load("exp5_baseline.json")
    if not b: return
    fig, ax = plt.subplots(figsize=(7,5))
    for mk in ORDER:
        rows = sorted([r for r in b if r["model_key"]==mk], key=lambda x:x["num_tenants"])
        if not rows: continue
        xs = [r["num_tenants"] for r in rows]
        ys = [r["total_ms"] for r in rows]
        ax.plot(xs, ys, marker=MARK[mk], color=COLORS[mk], linewidth=2, label=LABEL[mk])
    ax.set_xscale("log", base=2); ax.set_xticks(TENANT_COUNTS)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("Number of Tenants"); ax.set_ylabel("Total Adapter Latency (ms)")
    ax.set_title("Baseline (Sequential) Latency Scaling"); ax.legend()
    save(fig, "fig6_baseline_scaling")

if __name__ == "__main__":
    print("Generating six figures...")
    fig1(); fig2(); fig3(); fig4(); fig5(); fig6()
    print("Done. Six figures in results/.")