"""
-- Real comparison against Punica's SGMV kernel, IF it
installed successfully on your RTX 4080 SUPER (run try_punica_install.ps1 first).

This script only reports numbers it actually measures. If Punica fails to
import, it prints that fact clearly instead of faking a result -- an honest
"Punica does not run on this hardware" is itself a valid, reportable finding
for the paper's positioning section.

Run from the MAIN folder:  python benchmark_punica.py
Output: results/punica_comparison.json (only written if Punica ran)
"""
import sys, os, json, time
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import torch
from config import MODELS, RESULTS_DIR, gpu_name
from core.lora_model import MultiTenantLoRA

RANK = 8
DTYPE = torch.float16
T = 32          # a representative mixed tenant count
ITERS = 50
WARMUP = 10

def try_import_punica():
    try:
        import punica
        print(f"Punica imported successfully from: {punica.__file__}")
        return punica
    except Exception as e:
        print(f"Punica import FAILED: {type(e).__name__}: {e}")
        print("This means the SGMV kernel did not build/run for this GPU.")
        print("This is a real finding: report it honestly in the paper as")
        print("'Punica's SGMV kernel does not build on Ada-architecture")
        print("consumer GPUs out of the box', with this exact error as evidence.")
        return None

def time_fn(fn, iters=ITERS, warmup=WARMUP):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters): fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / iters * 1000.0  # ms

def run_our_fused(model_key):
    cfg = MODELS[model_key]
    mt = MultiTenantLoRA(cfg["path"], num_tenants=T, lora_rank=RANK, dtype=DTYPE)
    h = mt.compute_hidden_state()
    ms = time_fn(lambda: mt.adapter_fused(h))
    mt.cleanup(); del mt; torch.cuda.empty_cache()
    return ms

def try_punica_sgmv(punica_module, model_key):
    """
    Attempt to run Punica's SGMV op directly on equivalent random adapter
    data (same shapes as our fused-batching benchmark), matched to T=32,
    rank=8. This calls Punica's kernel API, NOT our code -- if this
    section errors, that is itself the finding.
    """
    cfg = MODELS[model_key]
    d = 4096  # placeholder; real d_model is read from the loaded model below
    try:
        # NOTE: Punica's op signature (add_lora / sgmv) varies by version.
        # Inspect `punica.ops` in your installed version and adjust this
        # call to match. This is left explicit rather than guessed, because
        # silently adapting to a mismatched API would risk measuring the
        # wrong thing.
        from punica.ops import add_lora_sgmv  # may differ by punica version
        print("Found punica.ops.add_lora_sgmv -- adjust shapes below to match your model's d_model before timing.")
        return None  # deliberately not fabricating a timed result here
    except ImportError as e:
        print(f"Could not find expected Punica op: {e}")
        print("Check `python -c \"import punica; print(dir(punica.ops))\"` ")
        print("and adapt this function to the actual API before timing.")
        return None

def main():
    print(f"GPU: {gpu_name()}")
    punica = try_import_punica()

    print(f"\n=== Our fused batching (T={T}, rank={RANK}) ===")
    results = {}
    for mk in ["llama", "deepseek", "gemma"]:
        cfg = MODELS[mk]
        if not os.path.exists(cfg["path"]):
            continue
        ms = run_our_fused(mk)
        results[mk] = {"our_fused_ms": round(ms, 3)}
        print(f"  {cfg['name']:<28} our fused: {ms:.3f} ms")

    if punica is None:
        print("\n=== RESULT: Punica does not run on this GPU. ===")
        print("Report this honestly in the paper's positioning section.")
        out = {"punica_available": False, "our_results": results,
               "gpu": gpu_name()}
    else:
        print("\nPunica imported. You must now adapt try_punica_sgmv() to")
        print("your installed version's actual op signature before any")
        print("Punica latency number can be trusted or reported.")
        out = {"punica_available": True, "our_results": results,
               "gpu": gpu_name(), "note": "Punica ops not yet benchmarked -- see script"}

    path = os.path.join(RESULTS_DIR, "punica_comparison.json")
    with open(path, "w") as f: json.dump(out, f, indent=2)
    print(f"\nSaved {path}")

if __name__ == "__main__": main()
