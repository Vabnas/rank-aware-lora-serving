"""
run_all.py — Run every experiment end to end.

Usage:
    python run_all.py                # all experiments, all models
    python run_all.py --model llama  # all experiments, one model

This runs:
    exp0_rank_sweep        (motivation for C3)
    exp1_fused_batching    (Contribution 1)
    exp2_dynamic_scheduler (Contribution 2)
    exp3_adaptive_rank     (Contribution 3)
"""

import argparse
from config import MODELS, gpu_name
from experiments import exp0_rank_sweep, exp1_fused_batching
from experiments import exp2_dynamic_scheduler, exp3_adaptive_rank
import json, os
from config import RESULTS_DIR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS) + ["all"], default="all")
    args = ap.parse_args()

    keys = list(MODELS) if args.model == "all" else [args.model]
    print("=" * 65)
    print(f"  RUNNING ALL EXPERIMENTS — GPU: {gpu_name()}")
    print(f"  Models: {keys}")
    print("=" * 65)

    bundle = {}

    print("\n##### EXP 0: RANK SWEEP (motivation) #####")
    bundle["rank_sweep"] = []
    for k in keys:
        bundle["rank_sweep"] += exp0_rank_sweep.run(k, MODELS[k])

    print("\n##### EXP 1: FUSED BATCHING (C1) #####")
    bundle["c1"] = []
    for k in keys:
        bundle["c1"] += exp1_fused_batching.run(k, MODELS[k])

    print("\n##### EXP 2: DYNAMIC SCHEDULER (C2) #####")
    bundle["c2"] = []
    for k in keys:
        bundle["c2"] += exp2_dynamic_scheduler.run(k, MODELS[k])

    print("\n##### EXP 3: ADAPTIVE RANK (C3) #####")
    bundle["c3"] = []
    for k in keys:
        bundle["c3"] += exp3_adaptive_rank.run(k, MODELS[k])

    # Save a SEPARATE file per model so nothing gets overwritten when you
    # run models one at a time (e.g. --model gemma after --model llama).
    tag = args.model  # "llama", "gemma", "deepseek", or "all"
    out = os.path.join(RESULTS_DIR, f"results_{tag}.json")
    with open(out, "w") as f:
        json.dump(bundle, f, indent=2)
    print(f"\n✓ Saved {out}")

    # Also merge ALL per-model files into one combined all_results.json,
    # so the combined file accumulates every model you've run so far.
    combined = {"rank_sweep": [], "c1": [], "c2": [], "c3": []}
    for mk in MODELS:
        path = os.path.join(RESULTS_DIR, f"results_{mk}.json")
        if os.path.exists(path):
            with open(path) as f:
                d = json.load(f)
            for section in combined:
                combined[section] += d.get(section, [])
    all_path = os.path.join(RESULTS_DIR, "all_results.json")
    with open(all_path, "w") as f:
        json.dump(combined, f, indent=2)
    n_models = sum(1 for mk in MODELS if os.path.exists(os.path.join(RESULTS_DIR, f"results_{mk}.json")))
    print(f"✓ Merged {n_models} model(s) into {all_path}")
    print("  Now run:  python plot_results.py")


if __name__ == "__main__":
    main()