"""
stats_tests.py — Statistical tests on the experimental results.
Drops clear system-hiccup outliers (runs > 2x median latency) before testing.
Run:  python stats_tests.py
Requires: pip install scipy
"""
import os, json
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
from scipy import stats
from config import RESULTS_DIR

def load(name):
    p = os.path.join(RESULTS_DIR, name)
    if not os.path.exists(p): return None
    with open(p) as f: return json.load(f)

def drop_outliers(seq, fused, pad, split):
    seq = np.array(seq); fused = np.array(fused)
    pad = np.array(pad); split = np.array(split)
    med = np.median(seq)
    keep = seq < 2.0 * med
    n_drop = int((~keep).sum())
    return seq[keep], fused[keep], pad[keep], split[keep], n_drop

print("="*70)
print("STATISTICAL TESTS")
print("="*70)

print("\n[1] One-way ANOVA: does LoRA rank affect accuracy? (C3 quality)")
print("-"*70)
q = load("exp3_quality_rank.json")
if q and "results" in q:
    groups, ranks = [], []
    for r in q["results"]:
        runs = r.get("acc_runs", [])
        if runs:
            groups.append(runs); ranks.append(r["lora_rank"])
    if len(groups) >= 2 and all(len(g) >= 2 for g in groups):
        F, pval = stats.f_oneway(*groups)
        print(f"  Ranks tested: {ranks}")
        print(f"  F = {F:.3f},  p = {pval:.4f}")
        if pval > 0.05:
            print(f"  => p > 0.05: NO significant accuracy difference across ranks.")
            print(f"     Supports low-rank allocation (no quality loss).")
        else:
            print(f"  => p < 0.05: accuracy differs across ranks.")
    else:
        print("  Not enough per-rank seeds for ANOVA.")
else:
    print("  exp3_quality_rank.json not found.")

s = load("exp6_stats.json")
if s:
    print("\n[2] Paired t-test: fused vs sequential (C1)")
    print("-"*70)
    for r in s:
        seq, fused, pad, split, nd = drop_outliers(r["seq_runs"], r["fused_runs"], r["pad_runs"], r["split_runs"])
        if len(seq) >= 2:
            t, p = stats.ttest_rel(seq, fused)
            speedup = seq.mean()/fused.mean()
            sig = "significant" if p < 0.05 else "n.s."
            note = f"  (dropped {nd} outlier)" if nd else ""
            print(f"  {r['model']:<28} T={r['num_tenants']:<4} {speedup:.2f}x  t={t:.2f}  p={p:.4f}  [{sig}]{note}")

    print("\n[3] Paired t-test: split vs pad-and-fuse (C2)")
    print("-"*70)
    for r in s:
        seq, fused, pad, split, nd = drop_outliers(r["seq_runs"], r["fused_runs"], r["pad_runs"], r["split_runs"])
        if len(pad) >= 2:
            t, p = stats.ttest_rel(pad, split)
            speedup = pad.mean()/split.mean()
            sig = "significant" if p < 0.05 else "n.s."
            print(f"  {r['model']:<28} T={r['num_tenants']:<4} {speedup:.2f}x  t={t:.2f}  p={p:.4f}  [{sig}]")

    print("\n[4] 95% confidence interval for C1 speedup")
    print("-"*70)
    for r in s:
        seq, fused, pad, split, nd = drop_outliers(r["seq_runs"], r["fused_runs"], r["pad_runs"], r["split_runs"])
        if len(seq) >= 2:
            sp = seq/fused
            m, sem = sp.mean(), stats.sem(sp)
            if sem > 0:
                ci = stats.t.interval(0.95, len(sp)-1, loc=m, scale=sem)
                print(f"  {r['model']:<28} T={r['num_tenants']:<4} {m:.2f}x  95% CI [{ci[0]:.2f}, {ci[1]:.2f}]")
            else:
                print(f"  {r['model']:<28} T={r['num_tenants']:<4} {m:.2f}x  (near-zero variance)")
else:
    print("\n[2-4] exp6_stats.json not found.")

print("\n" + "="*70)
print("Done.")