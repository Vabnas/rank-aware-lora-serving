"""
utils/benchmark.py — Timing helpers.

CRITICAL: time_adapter() times ONLY the adapter application, using a pre-computed
hidden state. The expensive base-model forward runs once via compute_hidden_state()
and is NOT included in the timing loop. This isolates the LoRA cost that the three
contributions actually optimize.
"""

import time
import torch


def time_adapter(model, h, strategy, threshold=12, warmup=20, iters=200):
    """Time ONLY adapter application on a cached hidden state h. Returns ms (avg)."""
    # Warmup
    for _ in range(warmup):
        _ = model.apply_adapters(h, strategy=strategy, threshold=threshold)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Measure
    start = time.perf_counter()
    for _ in range(iters):
        _ = model.apply_adapters(h, strategy=strategy, threshold=threshold)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    return (time.perf_counter() - start) / iters * 1000


def vram_gb():
    return torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
