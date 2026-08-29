"""
core/lora_model.py — Multi-tenant LoRA model + the three serving strategies.

KEY DESIGN: the base-model forward pass and the LoRA adapter application are
SEPARATE. Benchmarks time ONLY the adapter application, because that is what
all three contributions optimize. The base forward (~400ms for an 8B model)
would otherwise swamp the adapter cost (sub-millisecond) and hide every effect.

Strategies:
    sequential  — baseline, one tenant at a time
    fused       — Contribution 1, all tenants in one bmm pass
    dynamic     — Contribution 2, pick sequential vs fused by threshold
    adaptive    — Contribution 3, per-tenant rank from quality tier
"""

import torch
import torch.nn as nn
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from config import USE_4BIT, WORKLOAD_BATCH, WORKLOAD_SEQ_LEN
except Exception:
    USE_4BIT = False
    WORKLOAD_BATCH = 8
    WORKLOAD_SEQ_LEN = 512


class MultiTenantLoRA(nn.Module):
    def __init__(self, model_path, num_tenants, lora_rank, dtype,
                 ranks_per_tenant=None, force_gpu=True):
        super().__init__()
        self.num_tenants = num_tenants
        self.lora_rank   = lora_rank

        # Load frozen base model. force_gpu=True puts everything on one GPU
        # (no CPU offload) so timing reflects GPU compute only.
        device_map = "cuda:0" if force_gpu else "auto"

        load_kwargs = dict(device_map=device_map, local_files_only=True)
        if USE_4BIT:
            # 4-bit NF4 quantization for the base model. Adapters stay fp16.
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True,
            )
        else:
            load_kwargs["dtype"] = dtype

        self.base_model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
        # Adapters always use a real float dtype (never the 4-bit base dtype).
        self.adapter_dtype = dtype if dtype in (torch.float16, torch.bfloat16) else torch.float16
        for p in self.base_model.parameters():
            p.requires_grad = False

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.d_model     = self.base_model.config.hidden_size
        self.model_dtype = next(self.base_model.parameters()).dtype
        # With 4-bit base, params report as uint8; pin adapters to a real float.
        if self.model_dtype not in (torch.float16, torch.bfloat16, torch.float32):
            self.model_dtype = self.adapter_dtype
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        # LoRA adapters — always float16/bfloat16 (never the 4-bit base dtype)
        if ranks_per_tenant is None:
            ranks_per_tenant = [lora_rank] * num_tenants
        self.ranks_per_tenant = ranks_per_tenant
        self.uniform = len(set(ranks_per_tenant)) == 1

        ad = self.adapter_dtype
        if self.uniform:
            r = ranks_per_tenant[0]
            self.A = (torch.randn(num_tenants, self.d_model, r, dtype=ad, device=self.device) * 0.01)
            self.B = (torch.randn(num_tenants, r, self.d_model, dtype=ad, device=self.device) * 0.01)
        else:
            self.A_list = [(torch.randn(self.d_model, r, dtype=ad, device=self.device) * 0.01)
                           for r in ranks_per_tenant]
            self.B_list = [(torch.randn(r, self.d_model, dtype=ad, device=self.device) * 0.01)
                           for r in ranks_per_tenant]

    # Build a REALISTIC adapter input: many tokens per request, batched.
    # Real LoRA serving processes a batch of requests, each with many tokens.
    # A (batch x seq, d_model) workload makes rank and batching effects real;
    # a single pooled (1, d_model) vector is too small to show either.
    @torch.no_grad()
    def compute_hidden_state(self, batch=WORKLOAD_BATCH, seq_len=WORKLOAD_SEQ_LEN):
        # One real base forward to get a representative hidden vector...
        inputs = self.tokenizer("Hello, how are you today?", return_tensors="pt").to(self.device)
        out = self.base_model(**inputs, output_hidden_states=True)
        base_vec = out.hidden_states[-1].mean(dim=1)            # (1, d_model)
        # ...then tile it into a realistic (batch*seq_len, d_model) workload.
        n_tokens = batch * seq_len
        h = base_vec.expand(n_tokens, -1).contiguous()
        return h.detach().to(self.adapter_dtype)

    # Strategy 1: Sequential (baseline) — ADAPTER ONLY
    def adapter_sequential(self, h):
        outputs = []
        if self.uniform:
            for i in range(self.num_tenants):
                outputs.append((h @ self.A[i]) @ self.B[i])
        else:
            for i in range(self.num_tenants):
                outputs.append((h @ self.A_list[i]) @ self.B_list[i])
        return torch.stack(outputs).sum(dim=0)

    # Strategy 2: Fused batching (C1) — ADAPTER ONLY
    def adapter_fused(self, h):
        x  = h.unsqueeze(0).expand(self.num_tenants, -1, -1)
        Ax = torch.bmm(x, self.A)
        d  = torch.bmm(Ax, self.B)
        return d.sum(dim=0)

    # Strategy 3: Dynamic schedule (C2) — ADAPTER ONLY
    def adapter_dynamic(self, h, threshold):
        if self.num_tenants >= threshold and self.uniform:
            return self.adapter_fused(h)
        return self.adapter_sequential(h)

    # Dispatch (adapter stage only)
    def apply_adapters(self, h, strategy="fused", threshold=12):
        if strategy == "sequential":
            return self.adapter_sequential(h)
        elif strategy == "fused":
            return self.adapter_fused(h)
        elif strategy == "dynamic":
            return self.adapter_dynamic(h, threshold)
        elif strategy == "adaptive":
            return self.adapter_sequential(h)
        raise ValueError(f"Unknown strategy: {strategy}")

    def cleanup(self):
        del self.base_model
        if self.uniform:
            del self.A, self.B
        else:
            del self.A_list, self.B_list
        torch.cuda.empty_cache()
        gc.collect()
