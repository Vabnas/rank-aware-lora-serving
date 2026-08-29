"""
config.py — Central configuration for the multi-tenant LoRA project.
Edit BASE_DIR and model folder names to match your machine.
"""

import os
import torch

# ============================================================
# PATHS — models live in the "model/" subfolder next to this file
# ============================================================
# This resolves relative to config.py, so it works no matter where you
# launch python from. Models are in multi_tenant_lora/model/.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(PROJECT_ROOT, "model")

# ============================================================
# QUANTIZATION
# ============================================================
# 4-bit (NF4) quantization for the BASE model. LoRA adapters stay in float16,
# so all three contributions (which operate on the adapters) are unaffected.
# This reflects real consumer-GPU deployment and keeps all 3 models within 16 GB.
# Requires: pip install bitsandbytes
USE_4BIT = True

# Model registry. Folder names must match what's on disk.
MODELS = {
    "llama": {
        "name":    "Llama 3 8B",
        "company": "Meta",
        "path":    os.path.join(BASE_DIR, "Meta-Llama-3-8B"),
        "dtype":   torch.bfloat16,
    },
    "gemma": {
        "name":    "Gemma 2 9B",
        "company": "Google",
        "path":    os.path.join(BASE_DIR, "gemma-2-9B"),
        "dtype":   torch.float16,   # float16 keeps 9B within 16 GB
    },
    "deepseek": {
        "name":    "DeepSeek-R1-Distill-Qwen-7B",
        "company": "DeepSeek",
        "path":    os.path.join(BASE_DIR, "DeepSeek-R1-Distill-Qwen-7B"),
        "dtype":   torch.bfloat16,
    },
}

# ============================================================
# SWEEP PARAMETERS
# ============================================================
TENANT_COUNTS = [4, 8, 16, 32, 64, 128]
RANK_COUNTS   = [4, 8, 16, 32, 64, 128]

# Workload size — how many tokens the adapters process per forward.
# A realistic batch*seq makes rank and batching effects measurable.
# (A single pooled vector is too small and hides all the effects.)
WORKLOAD_BATCH   = 8
WORKLOAD_SEQ_LEN = 512

# Benchmark settings (fewer iters now — each does much more work)
WARMUP_RUNS    = 10
NUM_ITERATIONS = 50

# Contribution 2: dynamic scheduler threshold
SCHEDULER_THRESHOLD = 12   # theta — re-evaluate against the new workload data

# Contribution 3: adaptive rank allocation
# Map a tenant "quality tier" to a LoRA rank.
QUALITY_TIERS = {
    "low":    4,    # simple tasks — smallest rank
    "medium": 16,   # default
    "high":   32,   # demanding tasks — larger rank
}

# Output locations
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def gpu_name():
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return "CPU"
