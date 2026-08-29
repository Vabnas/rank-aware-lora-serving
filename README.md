# rank-aware-lora-serving
Memory-aware multi-tenant LoRA serving with fused batching, rank-aware splitting, and quality-constrained rank allocation for efficient LLM inference.
# Intelligent Rank-Aware Multi-Tenant LoRA Serving

Implementation and experimental artifacts for the research study:

**Intelligent Rank-Aware Multi-Tenant LoRA Serving for Efficient LLM Inference on Consumer GPUs**

## Overview

This repository contains the implementation and experimental artifacts
for a memory-aware multi-tenant serving framework for Large Language
Models (LLMs) using Low-Rank Adaptation (LoRA).

The framework investigates efficient execution of multiple task-specific
LoRA adapters sharing a common base model on a consumer GPU.

The proposed approach combines three complementary techniques:

- **Fused batching**
- **Rank-aware splitting**
- **Quality-constrained rank allocation**

The framework is designed to reduce redundant adapter computation and
memory movement while maintaining the required task quality.

## Main Components

### Fused Batching

Fused batching consolidates adapter operations from multiple tenants
into batched computations to reduce redundant memory accesses and
execution overhead.

### Rank-Aware Splitting

Rank-aware splitting separates heterogeneous LoRA adapters into
rank-homogeneous groups to reduce unnecessary processing caused by
rank padding.

### Quality-Constrained Rank Allocation

Quality-constrained rank allocation selects the lowest evaluated LoRA
rank that satisfies a specified task-quality requirement.

### Memory-Traffic Cost Model

A memory-aware cost model is used to characterize the trade-off between
memory traffic and the execution overhead introduced by additional
rank groups.

## Evaluated Models

The experimental evaluation includes:

- Llama 3 8B
- Gemma 2 9B
- DeepSeek-R1-Distill-Qwen-7B

## LoRA Rank Configurations

The evaluated LoRA ranks are:

```text
4, 8, 16, 32, 64, 128
Experimental Platform

The experiments reported in the manuscript were conducted on:

GPU: NVIDIA RTX 4080 SUPER
GPU Memory: 16 GB
Theoretical Memory Bandwidth: approximately 736 GB/s
Base Model Quantization: 4-bit
Repository Structure
rank-aware-lora-serving/
│
├── src/
│   ├── fused_batching.py
│   ├── rank_aware_splitting.py
│   ├── rank_allocation.py
│   ├── cost_model.py
│   └── utils.py
│
├── experiments/
│   ├── baseline.py
│   ├── fused_batching.py
│   ├── rank_aware_splitting.py
│   ├── ablation.py
│   ├── quality_vs_rank.py
│   └── roofline.py
│
├── configs/
│   ├── llama3_8b.yaml
│   ├── gemma2_9b.yaml
│   └── deepseek_r1_7b.yaml
│
├── results/
│
├── figures/
│
├── scripts/
│
├── requirements.txt
├── README.md
└── .gitignore
Experimental Evaluation

The evaluation considers:

Sequential multi-tenant adapter execution
Fused batching
Rank-aware splitting
Quality-constrained rank allocation
Combined framework performance
LoRA rank versus task quality
Roofline and memory-bandwidth analysis
Main Results

At 128 tenants, the complete framework achieved the following latency
reductions relative to sequential adapter execution:

Model	Latency Reduction
Llama 3 8B	58.2%
DeepSeek-R1-Distill-Qwen-7B	58.8%
Gemma 2 9B	67.5%

Fused batching achieved approximately 49–56% latency reduction across
the evaluated configurations.

Rank-aware splitting provided additional improvements for
heterogeneous-rank workloads.

The detailed experimental results, methodology, and analysis are
provided in the associated manuscript.

Reproducibility

The repository is intended to provide the code and experimental
artifacts required to reproduce the reported results.

Reproducibility instructions will include:

Software dependencies
Model configuration
LoRA configuration
Quantization settings
Workload configuration
Tenant and rank distributions
Experiment execution commands
Result-processing scripts
Figure-generation scripts

Exact software versions will be documented based on the experimental
environment used for the study.

Models and Weights

Model weights are not included in this repository.

Users should obtain the corresponding model checkpoints from their
official distribution sources and comply with the licenses and usage
conditions associated with each model.

Data

The repository does not redistribute third-party datasets or model
weights unless their respective licenses permit redistribution.

Dataset preparation and acquisition instructions will be provided
where applicable.

Citation

If you use this repository or the associated experimental artifacts,
please cite the corresponding research article:

Hasi Akter Vabna and Prof. Wang.
Intelligent Rank-Aware Multi-Tenant LoRA Serving for Efficient LLM
Inference on Consumer GPUs.

The final bibliographic information and DOI will be added after
publication.

Code Availability

The source code and experimental artifacts associated with this study
will be made publicly available through this repository.

License

License information will be added after the ownership and
redistribution terms for the research code have been confirmed.
