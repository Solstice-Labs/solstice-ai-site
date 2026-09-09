---
title: "Quantized Models Quickstart"
description: "Reference guide for Solstice quantized checkpoints across GGUF, AWQ, NVFP4, MXFP4, and Apple MLX formats."
category: "models"
order: 2
lastUpdated: 2026-09-09
githubUrl: "https://github.com/Solstice-Labs"
specs:
  "Formats": "GGUF / AWQ INT4 / NVFP4 / MXFP4 / MLX"
  "Inference Runtimes": "vLLM, SGLang, llama.cpp, Apple MLX"
  "Hugging Face Org": "Solstice-AI"
  "Context Capacity": "1M Tokens (Native YaRN)"
supportedFormats:
  - "AWQ INT4 (W4A16 GEMM for vLLM & SGLang)"
  - "llama.cpp GGUF with mmproj-BF16"
  - "Apple MLX Mixed-Precision (oQ4e, oQ8e)"
  - "NVIDIA Blackwell NVFP4 & OCP MXFP4"
---

## Available Model Hub Checkpoints

All Solstice models and quantizations are published directly under the official [`Solstice-AI`](https://huggingface.co/Solstice-AI) organization on Hugging Face:

### Flagship Suite: Qwopus3.8-27B-Flash (1M Native Context)

| Model Identifier | Format / Precision | Context | Key Features | Target Hardware | Recommended Engine |
| :--- | :--- | :---: | :--- | :--- | :--- |
| **[`Qwopus3.8-27B-Flash-AWQ-1M`](https://huggingface.co/Solstice-AI/Qwopus3.8-27B-Flash-AWQ-1M)** | AWQ INT4 (W4A16 GEMM) | **1M Tokens** | AutoRound W4A16, 1M YaRN | RTX 3090 / 4090, A100, L40S | vLLM / SGLang |
| **[`Qwopus3.8-27B-Flash-GGUF-1M`](https://huggingface.co/Solstice-AI/Qwopus3.8-27B-Flash-GGUF-1M)** | GGUF Dynamic v3.0 | **1M Tokens** | UD-IQ4_XS to UD-Q8_XL + mmproj-BF16 | CPU, CUDA, Apple Metal | llama.cpp |
| **[`Qwopus3.8-27B-Flash-mlx-oQ4e-1M`](https://huggingface.co/Solstice-AI/Qwopus3.8-27B-Flash-mlx-oQ4e-1M)** | MLX 4-Bit Mixed | **1M Tokens** | Apple Metal MTP, 2.2x speed | Apple Silicon (16GB–24GB RAM) | Apple MLX |
| **[`Qwopus3.8-27B-Flash-mlx-oQ8e-1M`](https://huggingface.co/Solstice-AI/Qwopus3.8-27B-Flash-mlx-oQ8e-1M)** | MLX 8-Bit Mixed | **1M Tokens** | Apple Metal MTP, reference grade | Apple Silicon (36GB+ RAM) | Apple MLX |
| **[`Qwopus3.8-27B-Flash-NVFP4-1M`](https://huggingface.co/Solstice-AI/Qwopus3.8-27B-Flash-NVFP4-1M)** | Blackwell NVFP4 | **1M Tokens** | Native NVFP4 Tensor Cores + MTP | RTX 4090 / 5090, L40S | TensorRT-LLM / vLLM |
| **[`Qwopus3.8-27B-Flash-MXFP4-1M`](https://huggingface.co/Solstice-AI/Qwopus3.8-27B-Flash-MXFP4-1M)** | OCP MXFP4 | **1M Tokens** | Microscaling FP4 + MTP | Universal 24GB+ GPU | vLLM / SGLang |

### Additional Checkpoint Suites

| Model Suite | Available Formats | Context | Focus Area | Recommended Engine |
| :--- | :--- | :---: | :--- | :--- |
| **[`Qwen3.8-27B-TURBO NEO-CODER`](https://huggingface.co/Solstice-AI)** | GGUF, AWQ, NVFP4, MXFP4, MLX | 1M / 262K | Code synthesis & SWE benchmarks | vLLM, llama.cpp, MLX |
| **[`DeepSeek-V4-Flash-Vision-UNCENSORED`](https://huggingface.co/Solstice-AI)** | GGUF, AWQ, NVFP4, MXFP4, MLX | Native | Multimodal vision & reasoning | vLLM, llama.cpp, MLX |
| **[`GLM-5.3-Flash-UNCENSORED`](https://huggingface.co/Solstice-AI)** | GGUF, AWQ, NVFP4, MLX | Native | High-throughput agent rollouts | vLLM, llama.cpp, MLX |

---

## 1. Enterprise Serving with vLLM & SGLang (AWQ INT4)

The **`Qwopus3.8-27B-Flash-AWQ-1M`** release packs weights into high-performance `W4A16 GEMM` format, enabling standard 24GB GPUs (RTX 3090, 4090, A10G, L40S) to run 1M context with Tensor Core acceleration.

### Launch with vLLM

```bash
# 1. Install vLLM
pip install vllm

# 2. Serve 1M YaRN context on a single 24GB GPU
vllm serve Solstice-AI/Qwopus3.8-27B-Flash-AWQ-1M \
  --quantization awq \
  --max-model-len 1048576 \
  --gpu-memory-utilization 0.95 \
  --port 8000
```

### Launch with SGLang

```bash
# High-throughput RadixAttention prefix caching server
python -m sglang.launch_server \
  --model-path Solstice-AI/Qwopus3.8-27B-Flash-AWQ-1M \
  --quantization awq \
  --port 30000
```

---

## 2. Running on Apple Silicon (MLX)

For M-series Macs (M1/M2/M3/M4/M5), run quantized mixed-precision checkpoints natively on Metal:

```bash
# Install MLX LM runtime
pip install mlx-lm

# Run 4-bit mixed precision (fits in 17 GB unified memory)
python -m mlx_lm.generate \
  --model Solstice-AI/Qwopus3.8-27B-Flash-mlx-oQ4e-1M \
  --prompt "Analyze this codebase architecture and identify concurrency bottlenecks." \
  --max-tokens 1024
```

---

## 3. Local Inference with llama.cpp (GGUF)

The **`Qwopus3.8-27B-Flash-GGUF-1M`** repository contains Unsloth Dynamic v3.0 quants from `UD-IQ4_XS` to `UD-Q8_XL`, as well as the multimodal vision projector `mmproj-BF16.gguf`:

```bash
# Run 1M context session with llama-cli
./llama-cli \
  -hf Solstice-AI/Qwopus3.8-27B-Flash-GGUF-1M \
  -m UD-Q4_K_XL.gguf \
  --ctx-size 1048576 \
  -p "Your prompt here"
```

To process image and video frames, attach the included vision projector:

```bash
./llama-cli \
  -hf Solstice-AI/Qwopus3.8-27B-Flash-GGUF-1M \
  -m UD-Q4_K_XL.gguf \
  --mmproj mmproj-BF16.gguf \
  --image /path/to/diagram.png \
  -p "Describe the architecture depicted in this diagram."
```
