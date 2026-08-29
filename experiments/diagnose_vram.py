"""
diagnose_vram.py — Check how each model loads and whether it offloads to CPU.
Respects USE_4BIT in config. Run:  python diagnose_vram.py
"""
import torch, os
from config import MODELS, USE_4BIT

def check(model_key, cfg):
    print(f"\n{'='*60}\n{cfg['name']} ({cfg['company']})  [4-bit={USE_4BIT}]\n{'='*60}")
    if not os.path.exists(cfg["path"]):
        print(f"  X Not found: {cfg['path']}"); return
    from transformers import AutoModelForCausalLM
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()

    kw = dict(device_map="cuda:0", local_files_only=True)
    if USE_4BIT:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=cfg["dtype"], bnb_4bit_use_double_quant=True)
    else:
        kw["dtype"] = cfg["dtype"]

    model = AutoModelForCausalLM.from_pretrained(cfg["path"], **kw)
    devices = {str(p.device) for p in model.parameters()}
    vram = torch.cuda.memory_allocated()/1e9
    peak = torch.cuda.max_memory_allocated()/1e9
    print(f"  Param devices: {devices}")
    print(f"  VRAM allocated: {vram:.1f} GB   Peak: {peak:.1f} GB")
    print("  ! SOME LAYERS ON CPU" if any("cpu" in d for d in devices) else "  OK Fully on GPU")
    del model; torch.cuda.empty_cache()

if __name__ == "__main__":
    total = torch.cuda.get_device_properties(0).total_memory/1e9
    print(f"GPU: {torch.cuda.get_device_name(0)}  |  Total VRAM: {total:.1f} GB  |  4-bit: {USE_4BIT}")
    for k, cfg in MODELS.items():
        try: check(k, cfg)
        except RuntimeError as e: print(f"  X {e}"); torch.cuda.empty_cache()
