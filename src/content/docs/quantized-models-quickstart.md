---
title: "Quantized Models & Anvil Quickstart"
description: "Step-by-step recipes for pulling and running Solstice quantized checkpoints with the Anvil CLI, 1,048,576 YaRN scaling, Multi-Token Prediction (MTP), and Google TurboQuant KV cache compression."
category: "models"
order: 2
lastUpdated: 2026-09-02
githubUrl: "https://github.com/Solstice-Labs/anvil"
specs:
  "Default Precision": "NVFP4 / MXFP4 / oMLX Mixed (oQ8e, oQ6e, oQ4e) / Q8_0 GGUF"
  "Inference Engine": "Anvil CLI (anvil 0.8.6+)"
  "Hugging Face Org": "Solstice-AI"
  "Context Capacity": "1M Tokens Native YaRN"
  "KV Cache Compression": "Google TurboQuant (~12–18 GB for 1M Tokens)"
supportedFormats:
  - "Anvil CLI (anvil run / anvil pull / anvil serve)"
  - "TurboQuant Presets (turbo4 / turbo3 / turbo2)"
  - "NVIDIA Blackwell NVFP4 & OCP MXFP4"
  - "Apple MLX Mixed-Precision (oQ8e, oQ6e, oQ4e)"
  - "llama.cpp GGUF with mmproj-BF16 Multimodal Projectors"
---

## Available Model Hub Checkpoints

All Solstice-distilled models and quantizations are published directly under the official [`Solstice-AI`](https://huggingface.co/Solstice-AI) organization on Hugging Face:

| Model Identifier | Format / Precision | Native Context | Key Features | Target Hardware |
| :--- | :--- | :---: | :--- | :--- |
| **[`...-AWQ-1M`](https://huggingface.co/Solstice-AI/Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-Heretic-Uncensored-NM-DAU-AWQ-1M)** | AWQ INT4 (W4A16 GEMM) | **1M Tokens** | vLLM / SGLang + MTP + Vision | RTX 3090 / 4090 / A10G / A100 |
| **[`...-AWQ`](https://huggingface.co/Solstice-AI/Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-Heretic-Uncensored-NM-DAU-AWQ)** | AWQ INT4 (W4A16 GEMM) | 262K Tokens | vLLM / SGLang + MTP + Vision | RTX 3090 / 4090 / A10G / A100 |
| **[`...-NVFP4-1M`](https://huggingface.co/Solstice-AI/Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-Heretic-Uncensored-NM-DAU-NVFP4-1M)** | Blackwell NVFP4 | **1M Tokens** | MTP Speculative Heads + mmproj-BF16 | RTX 4090 / 5090 / L40S |
| **[`...-NVFP4`](https://huggingface.co/Solstice-AI/Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-Heretic-Uncensored-NM-DAU-NVFP4)** | Blackwell NVFP4 | 262K Tokens | MTP Speculative Heads + mmproj-BF16 | RTX 4090 / A10G |
| **[`...-MXFP4-1M`](https://huggingface.co/Solstice-AI/Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-Heretic-Uncensored-NM-DAU-MXFP4-1M)** | OCP MXFP4 | **1M Tokens** | MTP Speculative Heads + mmproj-BF16 | Universal 24GB+ GPU |
| **[`...-MXFP4`](https://huggingface.co/Solstice-AI/Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-Heretic-Uncensored-NM-DAU-MXFP4)** | OCP MXFP4 | 262K Tokens | MTP Speculative Heads + mmproj-BF16 | Universal 24GB+ GPU |
| **[`...-mlx-oQ8e-1M`](https://huggingface.co/Solstice-AI/Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-Heretic-Uncensored-NM-DAU-mlx-oQ8e-1M)** | MLX 8-Bit Mixed | **1M Tokens** | Metal MTP + Vision | Apple Silicon (36GB+ RAM) |
| **[`...-mlx-oQ8e`](https://huggingface.co/Solstice-AI/Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-Heretic-Uncensored-NM-DAU-mlx-oQ8e)** | MLX 8-Bit Mixed | 262K Tokens | Metal MTP + Vision | Apple Silicon (36GB+ RAM) |
| **[`...-mlx-oQ6e-1M`](https://huggingface.co/Solstice-AI/Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-Heretic-Uncensored-NM-DAU-mlx-oQ6e-1M)** | MLX 6-Bit Mixed | **1M Tokens** | Metal MTP + Vision | Apple Silicon (32GB+ RAM) |
| **[`...-mlx-oQ6e`](https://huggingface.co/Solstice-AI/Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-Heretic-Uncensored-NM-DAU-mlx-oQ6e)** | MLX 6-Bit Mixed | 262K Tokens | Metal MTP + Vision | Apple Silicon (32GB+ RAM) |
| **[`...-mlx-oQ4e-1M`](https://huggingface.co/Solstice-AI/Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-Heretic-Uncensored-NM-DAU-mlx-oQ4e-1M)** | MLX 4-Bit Mixed | **1M Tokens** | 2.20x Speed + Vision | Apple Silicon (16GB–24GB RAM) |
| **[`...-mlx-oQ4e`](https://huggingface.co/Solstice-AI/Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-Heretic-Uncensored-NM-DAU-mlx-oQ4e)** | MLX 4-Bit Mixed | 262K Tokens | 2.20x Speed + Vision | Apple Silicon (16GB–24GB RAM) |
| **[`...-GGUF-UltraOptimised`](https://huggingface.co/Solstice-AI/Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-Heretic-Uncensored-GGUF-UltraOptimised)** | GGUF Matrix (424GB) | 262K Tokens | mmproj-BF16 + DiMatrix MTP | CPU / CUDA / Metal |

---

## 1. Running Anvil (`anvil`) with TurboQuant & 1M Context

[**Anvil**](https://github.com/Solstice-Labs/anvil) is Solstice Labs' terminal-first in-process runtime. It integrates **Google TurboQuant (FWHT rotation)** and native **YaRN 1 Million (1M) token context**:

```
1 Million (1M) Token KV Cache Footprint (Qwen 3.8):
Standard FP16 KV Cache:       88.4 GB VRAM  (Requires 2x A100 80GB)
Anvil TurboQuant (turbo4):     18.2 GB VRAM  (4.8x compression)
Anvil TurboQuant (turbo3):     12.4 GB VRAM  (7.1x compression, <1% delta)
Anvil TurboQuant (turbo2):     10.2 GB VRAM  (8.6x compression)
```

### Install Anvil

```bash
curl -fsSL https://anvil-llm.github.io/anvil/install.sh | sh
```

### Pull and Execute Instant 1M Context Session

```bash
# Pull model into local registry
anvil pull hf:Solstice-AI/Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-Heretic-Uncensored-NM-DAU-NVFP4-1M

# Run interactive 1,048,576 session with TurboQuant 3-bit KV cache compression
anvil run hf:Solstice-AI/Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-Heretic-Uncensored-NM-DAU-NVFP4-1M \
  --ctx 1048576 \
  --type-k turbo4 \
  --type-v turbo3

# Serve OpenAI-compatible API on port 8080
anvil serve --port 8080
```

---

## 2. Running Apple Silicon Checkpoints (MLX)

For M-series Macs (M1/M2/M3/M4/M5), run quantized mixed-precision checkpoints with hardware-accelerated **Multi-Token Prediction (MTP)**:

```bash
# Install MLX LM runtime
pip install mlx-lm

# Run 4-bit mixed precision (fits in 17 GB RAM, runs at 2.20x speed)
mlx_lm.generate \
  --model Solstice-AI/Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-Heretic-Uncensored-NM-DAU-mlx-oQ4e-1M \
  --prompt "Analyze this full repository architecture and verify concurrent safety." \
  --max-tokens 1024
```

---

## 3. Multimodal Image & Video Execution

Load the included **`mmproj-BF16.gguf`** projector to process high-resolution images and spatial-temporal video frames:

```bash
anvil run hf:Solstice-AI/Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-Heretic-Uncensored-GGUF-UltraOptimised:Q4_K_M \
  --mmproj mmproj-BF16.gguf \
  --image /path/to/system_diagram.png
```

---

## 4. Enterprise Serving with vLLM & SGLang (AWQ INT4)

The **`...-AWQ-1M`** and **`...-AWQ`** releases pack weights into high-performance `W4A16 GEMM` format, enabling standard 24GB GPUs (RTX 3090, 4090, A10G, L40S) to achieve maximum Tensor Core throughput in production orchestrators.

### Launch with vLLM

```bash
# 1. Install vLLM
pip install vllm

# 2. Serve 1,048,576 YaRN context on a single 24GB GPU
vllm serve Solstice-AI/Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-Heretic-Uncensored-NM-DAU-AWQ-1M \
  --quantization awq \
  --max-model-len 1048576 \
  --gpu-memory-utilization 0.95 \
  --port 8000
```

### Launch with SGLang

```bash
# High-throughput RadixAttention prefix caching server
python -m sglang.launch_server \
  --model-path Solstice-AI/Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-Heretic-Uncensored-NM-DAU-AWQ-1M \
  --quantization awq \
  --port 30000
```

