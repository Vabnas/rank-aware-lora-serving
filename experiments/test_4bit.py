"""
test_4bit.py — Confirm whether bitsandbytes 4-bit actually works on this machine.
Run:  python test_4bit.py
"""
import torch

print("="*60)
print("STEP 1: Is bitsandbytes importable and CUDA-enabled?")
print("="*60)
try:
    import bitsandbytes as bnb
    print(f"  bitsandbytes version: {bnb.__version__}")
    # Try a real 4-bit op
    try:
        x = torch.randn(64, 64, device="cuda", dtype=torch.float16)
        q, state = bnb.functional.quantize_4bit(x)
        print("  ✓ 4-bit quantize works — bitsandbytes CUDA backend OK")
        bnb_ok = True
    except Exception as e:
        print(f"  ✗ 4-bit op failed: {e}")
        bnb_ok = False
except Exception as e:
    print(f"  ✗ Cannot import bitsandbytes: {e}")
    bnb_ok = False

print()
print("="*60)
print("STEP 2: Does USE_4BIT get read from config?")
print("="*60)
try:
    from config import USE_4BIT
    print(f"  USE_4BIT = {USE_4BIT}")
except Exception as e:
    print(f"  ✗ {e}")

print()
print("="*60)
print("STEP 3: Load Llama and check actual memory")
print("="*60)
from config import MODELS, USE_4BIT
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import os

cfg = MODELS["llama"]
if not os.path.exists(cfg["path"]):
    print(f"  ✗ Llama not found at {cfg['path']}")
else:
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    qc = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    try:
        m = AutoModelForCausalLM.from_pretrained(
            cfg["path"], quantization_config=qc,
            device_map="cuda:0", local_files_only=True)
        vram = torch.cuda.memory_allocated()/1e9
        # Check if layers are actually quantized
        is_quant = any("4bit" in type(mod).__name__.lower() or "Params4bit" in type(p).__name__
                       for mod in m.modules() for p in mod.parameters(recurse=False)) \
                   if False else False
        from bitsandbytes.nn import Linear4bit
        n4 = sum(1 for mod in m.modules() if isinstance(mod, Linear4bit))
        print(f"  VRAM after 4-bit load: {vram:.1f} GB")
        print(f"  Number of Linear4bit layers: {n4}")
        if n4 > 0 and vram < 8:
            print("  ✓ 4-bit IS working")
        else:
            print("  ✗ 4-bit NOT applied — model loaded in full precision")
        del m; torch.cuda.empty_cache()
    except Exception as e:
        print(f"  ✗ Load failed: {e}")
