---
title: "Quantized Flagship Checkpoints Catalog"
description: "Technical specifications, precision formats, and hardware matrix for Solstice open-weights flagship suites: Qwopus 3.8 Flash, DeepSeek-V4 Vision, GLM-5.3 Flash, and NEO-CODER."
category: "models"
order: 2
lastUpdated: 2026-09-09
githubUrl: "https://github.com/Solstice-Labs"
specs:
  "Formats": "GGUF Dynamic / AWQ INT4 / NVFP4 / MXFP4 / MLX"
  "Compatible Runtimes": "vLLM, SGLang, llama.cpp, Apple MLX, TensorRT-LLM"
  "Hugging Face Org": "Solstice-AI"
  "Context Capacity": "1M Tokens Native YaRN / Variable Native"
supportedFormats:
  - "AWQ INT4 (W4A16 GEMM for vLLM & SGLang)"
  - "llama.cpp GGUF with mmproj-BF16 Multimodal Projectors"
  - "Apple MLX Mixed-Precision (oQ4e, oQ6e, oQ8e)"
  - "NVIDIA Blackwell NVFP4 & OCP MXFP4 Microscaling"
---

## Available Flagship Checkpoints

All Solstice-distilled model checkpoints and quantizations are published under the official [`Solstice-AI`](https://huggingface.co/Solstice-AI) organization on Hugging Face:

### 1. Qwopus3.8-27B-Flash Suite (1M Native Context)

| Model Identifier | Format / Precision | Context | Key Features | Target Hardware | Compatible Engine |
| :--- | :--- | :---: | :--- | :--- | :--- |
| **[`Qwopus3.8-27B-Flash-AWQ-1M`](https://huggingface.co/Solstice-AI/Qwopus3.8-27B-Flash-AWQ-1M)** | AWQ INT4 (W4A16 GEMM) | **1M Tokens** | AutoRound W4A16, 1M YaRN scaling | RTX 3090 / 4090, A100, L40S | vLLM / SGLang |
| **[`Qwopus3.8-27B-Flash-GGUF-1M`](https://huggingface.co/Solstice-AI/Qwopus3.8-27B-Flash-GGUF-1M)** | GGUF Dynamic v3.0 | **1M Tokens** | UD-IQ4_XS to UD-Q8_XL + mmproj-BF16 | CPU, CUDA, Apple Metal | llama.cpp |
| **[`Qwopus3.8-27B-Flash-mlx-oQ4e-1M`](https://huggingface.co/Solstice-AI/Qwopus3.8-27B-Flash-mlx-oQ4e-1M)** | MLX 4-Bit Mixed | **1M Tokens** | Apple Metal MTP, 2.2x speed | Apple Silicon (16GB–24GB RAM) | Apple MLX |
| **[`Qwopus3.8-27B-Flash-mlx-oQ8e-1M`](https://huggingface.co/Solstice-AI/Qwopus3.8-27B-Flash-mlx-oQ8e-1M)** | MLX 8-Bit Mixed | **1M Tokens** | Apple Metal MTP, reference grade | Apple Silicon (36GB+ RAM) | Apple MLX |
| **[`Qwopus3.8-27B-Flash-NVFP4-1M`](https://huggingface.co/Solstice-AI/Qwopus3.8-27B-Flash-NVFP4-1M)** | Blackwell NVFP4 | **1M Tokens** | Native NVFP4 Tensor Cores + MTP | RTX 4090 / 5090, L40S | TensorRT-LLM / vLLM |
| **[`Qwopus3.8-27B-Flash-MXFP4-1M`](https://huggingface.co/Solstice-AI/Qwopus3.8-27B-Flash-MXFP4-1M)** | OCP MXFP4 | **1M Tokens** | Microscaling FP4 + MTP | Universal 24GB+ GPU | vLLM / SGLang |

### 2. DeepSeek-V4-Flash-Vision Suite (Multimodal & Uncensored)

| Model Identifier | Format / Precision | Context | Key Features | Target Hardware | Compatible Engine |
| :--- | :--- | :---: | :--- | :--- | :--- |
| **[`DeepSeek-V4-Flash-Vision-UNCENSORED-AWQ-DSpark`](https://huggingface.co/Solstice-AI/DeepSeek-V4-Flash-Vision-UNCENSORED-AWQ-DSpark)** | AWQ INT4 (W4A16 GEMM) | Native | High-speed spatial reasoning + DSpark | RTX 3090 / 4090, A100 | vLLM / SGLang |
| **[`DeepSeek-V4-Flash-Vision-UNCENSORED-GGUF-DSpark`](https://huggingface.co/Solstice-AI/DeepSeek-V4-Flash-Vision-UNCENSORED-GGUF-DSpark)** | GGUF Matrix + mmproj | Native | Multimodal vision with DSpark drafter | CPU, CUDA, Metal | llama.cpp |
| **[`DeepSeek-V4-Flash-Vision-UNCENSORED-mlx-oQ4e-DSpark`](https://huggingface.co/Solstice-AI/DeepSeek-V4-Flash-Vision-UNCENSORED-mlx-oQ4e-DSpark)** | MLX 4-Bit Mixed | Native | High-efficiency Apple Metal vision | Apple Silicon (16GB–24GB RAM) | Apple MLX |
| **[`DeepSeek-V4-Flash-Vision-UNCENSORED-mlx-oQ8e-DSpark`](https://huggingface.co/Solstice-AI/DeepSeek-V4-Flash-Vision-UNCENSORED-mlx-oQ8e-DSpark)** | MLX 8-Bit Mixed | Native | High-precision Apple Metal vision | Apple Silicon (36GB+ RAM) | Apple MLX |
| **[`DeepSeek-V4-Flash-Vision-UNCENSORED-NVFP4-DSpark`](https://huggingface.co/Solstice-AI/DeepSeek-V4-Flash-Vision-UNCENSORED-NVFP4-DSpark)** | Blackwell NVFP4 | Native | Native Tensor Core FP4 acceleration | RTX 4090 / 5090, L40S | TensorRT-LLM / vLLM |
| **[`DeepSeek-V4-Flash-Vision-UNCENSORED-MXFP4-DSpark`](https://huggingface.co/Solstice-AI/DeepSeek-V4-Flash-Vision-UNCENSORED-MXFP4-DSpark)** | OCP MXFP4 | Native | Microscaling FP4 with DSpark drafter | Universal 24GB+ GPU | vLLM / SGLang |

### 3. GLM-5.3-Flash Suite (High-Throughput Uncensored)

| Model Identifier | Format / Precision | Context | Key Features | Target Hardware | Compatible Engine |
| :--- | :--- | :---: | :--- | :--- | :--- |
| **[`GLM-5.3-Flash-UNCENSORED-AWQ`](https://huggingface.co/Solstice-AI/GLM-5.3-Flash-UNCENSORED-AWQ)** | AWQ INT4 (W4A16 GEMM) | Native | Low-latency agent rollouts | RTX 3090 / 4090, A100 | vLLM / SGLang |
| **[`GLM-5.3-Flash-UNCENSORED-GGUF`](https://huggingface.co/Solstice-AI/GLM-5.3-Flash-UNCENSORED-GGUF)** | GGUF Quantized Matrix | Native | Comprehensive desktop deployment | CPU, CUDA, Metal | llama.cpp |
| **[`GLM-5.3-Flash-UNCENSORED-NVFP4-DSpark`](https://huggingface.co/Solstice-AI/GLM-5.3-Flash-UNCENSORED-NVFP4-DSpark)** | Blackwell NVFP4 | Native | High-throughput batch reasoning | RTX 4090 / 5090, L40S | TensorRT-LLM / vLLM |
| **[`GLM-5.3-Flash-UNCENSORED-mlx-oQ4e-DSpark`](https://huggingface.co/Solstice-AI/GLM-5.3-Flash-UNCENSORED-mlx-oQ4e-DSpark)** | MLX 4-Bit Mixed | Native | Compact mobile workstation inference | Apple Silicon (16GB RAM) | Apple MLX |
| **[`GLM-5.3-Flash-UNCENSORED-mlx-oQ8e-DSpark`](https://huggingface.co/Solstice-AI/GLM-5.3-Flash-UNCENSORED-mlx-oQ8e-DSpark)** | MLX 8-Bit Mixed | Native | Lossless reference agent trace grade | Apple Silicon (32GB+ RAM) | Apple MLX |

### 4. Qwen3.8-27B-TURBO NEO-CODER Suite (Agentic SWE)

| Model Identifier | Format / Precision | Context | Key Features | Target Hardware | Compatible Engine |
| :--- | :--- | :---: | :--- | :--- | :--- |
| **[`NEO-CODER-AWQ-1M`](https://huggingface.co/Solstice-AI/Qwen3.8-27B-TURBO-Fable-Cold-Fusion-NEO-CODER-AWQ-1M)** | AWQ INT4 (W4A16 GEMM) | **1M Tokens** | Full repository SWE-bench verification | RTX 3090 / 4090, A100 | vLLM / SGLang |
| **[`NEO-CODER-NVFP4-1M`](https://huggingface.co/Solstice-AI/Qwen3.8-27B-TURBO-Fable-Cold-Fusion-NEO-CODER-NVFP4-1M)** | Blackwell NVFP4 | **1M Tokens** | Microscaling Tensor Cores + MTP | RTX 4090 / 5090, L40S | TensorRT-LLM / vLLM |
| **[`NEO-CODER-MXFP4-1M`](https://huggingface.co/Solstice-AI/Qwen3.8-27B-TURBO-Fable-Cold-Fusion-NEO-CODER-MXFP4-1M)** | OCP MXFP4 | **1M Tokens** | Universal enterprise cluster serving | Universal 24GB+ GPU | vLLM / SGLang |
| **[`NEO-CODER-GGUF`](https://huggingface.co/Solstice-AI/Qwen3.8-27B-TURBO-Fable-Cold-Fusion-NEO-CODER-GGUF)** | GGUF Quantized Matrix | 262K Tokens | Integrated speculative drafting | CPU, CUDA, Metal | llama.cpp |
| **[`NEO-CODER-mlx-oQ4e-1M`](https://huggingface.co/Solstice-AI/Qwen3.8-27B-TURBO-Fable-Cold-Fusion-NEO-CODER-mlx-oQ4e-1M)** | MLX 4-Bit Mixed | **1M Tokens** | Local agentic coding on macOS | Apple Silicon (16GB–24GB RAM) | Apple MLX |
| **[`NEO-CODER-mlx-oQ8e-1M`](https://huggingface.co/Solstice-AI/Qwen3.8-27B-TURBO-Fable-Cold-Fusion-NEO-CODER-mlx-oQ8e-1M)** | MLX 8-Bit Mixed | **1M Tokens** | Reference competitive coding grade | Apple Silicon (36GB+ RAM) | Apple MLX |

---

## Quantization Formats & Specifications

### AWQ (Activation-aware Weight Quantization)
- **Format**: INT4 weights, FP16 activations (W4A16 GEMM).
- **Footprint**: ~18–20 GB total weights for 27B parameter models.
- **Design Target**: Production GPU serving engines including vLLM and SGLang. Preserves salient weight channels by analyzing activation magnitude distributions, preventing outlier clipping degradation in reasoning tasks.

### GGUF Dynamic Matrix (Unsloth Dynamic v3.0)
- **Format**: Mixed per-tensor quantization ranging from `UD-IQ4_XS` (compact desktop footprint) to `UD-Q8_XL` (lossless reference quality).
- **Vision Projection**: Packaged with companion `mmproj-BF16.gguf` for visual spatial-temporal comprehension on vision-capable architectures.
- **Design Target**: Local deployment across CPU, CUDA, and Apple Silicon unified memory via standard llama.cpp bindings.

### Apple MLX Mixed Precision
- **Format**: 4-bit (`mlx-oQ4e`), 6-bit (`mlx-oQ6e`), and 8-bit (`mlx-oQ8e`) mixed-precision configurations.
- **Footprint**: ~17.0 GB for `oQ4e` (operational on 16GB–24GB unified memory MacBooks) and ~30.0 GB for `oQ8e`.
- **Design Target**: Native Apple Silicon Metal acceleration with integrated Multi-Token Prediction (MTP) drafting support.

### NVIDIA NVFP4 & OCP MXFP4
- **Format**: 4-bit floating point microscaling format designed for Blackwell and Ada Lovelace Tensor Cores.
- **Footprint**: ~18.8 GB for NVFP4, ~18.0 GB for OCP MXFP4.
- **Design Target**: Next-generation datacenter and workstation silicon optimizing FP4 arithmetic throughput while maintaining activation fidelity.
