---
title: "Quantized Models & Anvil Quickstart"
description: "Step-by-step recipes for pulling and running Solstice quantized checkpoints with the Anvil CLI and Google TurboQuant KV cache compression."
category: "models"
order: 2
lastUpdated: 2026-08-31
githubUrl: "https://github.com/gondaliyashreyan1/anvil-llama-turbo"
specs:
  "Default Precision": "6-bit MLX / Q8_0 GGUF"
  "Inference Engine": "Anvil CLI (anvil 0.8.6+)"
  "Hugging Face Org": "Solstice-AI"
  "Context Compression": "262k Context in ~4 GB VRAM"
supportedFormats:
  - "Anvil CLI (anvil run / anvil pull)"
  - "TurboQuant Presets (turbo4 / turbo3 / turbo2)"
  - "Apple MLX (Fused Metal)"
  - "vLLM / PyTorch"
---

## Available Model Hub Checkpoints

All Solstice-distilled models and quantizations are published directly under the official [`Solstice-AI`](https://huggingface.co/Solstice-AI) organization on Hugging Face:

| Model Identifier | Base Architecture | Format / Precision | Target Hardware | Downloads |
| :--- | :--- | :--- | :--- | :--- |
| **`SolsticeAI/Qwen3.8-27B-Uncensored-mlx-6Bit`** | Qwen 3.8 27B | 6-bit MLX | Apple Silicon (M-Series) | 790 |
| **`Solstice-AI/Qwen3.8-27B-Cold-Fusion-GAIN-V1.1-mlx-6Bit`** | Qwen 3.8 27B | 6-bit MLX | Apple Silicon (M-Series) | 324 |
| **`Solstice-AI/Huihui-Qwen3.6-35B-A3B-Claude-4.7-Opus-abliterated-Q8_0-GGUF`** | Qwen 3.6 35B | Q8_0 GGUF | Anvil (CUDA / Metal) | 100 |
| **`Solstice-AI/ThinkingCap-Qwen3.6-27B-mlx-6Bit`** | Qwen 3.6 27B | 6-bit MLX | Apple Silicon (M-Series) | 79 |
| **`Solstice-AI/Qwopus3.6-27B-Coder-mlx-6Bit`** | Qwen 3.6 27B | 6-bit MLX | Apple Silicon (M-Series) | 48 |
| **`Solstice-AI/Athena-27B-UltraEfficient`** | Athena 27B | Native PyTorch | vLLM / SGLang | 23 |

---

## 1. Running Apple Silicon Checkpoints (MLX)

For M-series Macs (M1/M2/M3/M4 Pro/Max/Ultra), run quantized 6-bit checkpoints with native unified memory bandwidth:

### Quick CLI Generation

```bash
# Install MLX LM runtime
pip install mlx-lm

# Run Qwen 3.8 Cold-Fusion GAIN checkpoint
mlx_lm.generate \
  --model Solstice-AI/Qwen3.8-27B-Cold-Fusion-GAIN-V1.1-mlx-6Bit \
  --prompt "Write an optimized concurrent task scheduler in Rust." \
  --max-tokens 1024 \
  --temp 0.2
```

### Python API Integration

```python
from mlx_lm import load, generate

model, tokenizer = load("Solstice-AI/Qwen3.8-27B-Cold-Fusion-GAIN-V1.1-mlx-6Bit")

response = generate(
    model,
    tokenizer,
    prompt="Explain the core mechanism of multi-teacher distillation in LLMs.",
    max_tokens=1024,
    verbose=True
)
```

---

## 2. Running Anvil (`anvil`) with TurboQuant

**Anvil (`anvil`)** is Solstice Labs' native inference CLI. It integrates **Google TurboQuant (FWHT rotation)** to compress large context windows (up to 262,144 tokens) into consumer GPU VRAM.

```
262,144 Token Context (Qwen 3.8 / Huihui 35B)
Standard FP16 KV Cache:     18.6 GB VRAM  (OOM on 24GB GPUs)
Anvil TurboQuant (turbo3):   4.1 GB VRAM  (Fits comfortably in single GPU)
Anvil TurboQuant (turbo2):   2.8 GB VRAM  (Extreme long-context density)
```

### Pulling Models Directly from Hugging Face

```bash
# Pull Solstice GGUF checkpoint into local Anvil registry
anvil pull hf:Solstice-AI/Huihui-Qwen3.6-35B-A3B-Claude-4.7-Opus-abliterated-Q8_0-GGUF
```

### Single-Word Execution with Anvil

```bash
# Interactive chat REPL with 262k context and TurboQuant 3-bit KV cache
anvil run huihui-qwen --ctx 262144 --type-k turbo4 --type-v turbo3

# Save hardware flags into the model registry for instant one-word launch
anvil run huihui-qwen --ctx 262144 --type-k turbo4 --type-v turbo3 --save

# Now launch with a single word
anvil huihui-qwen

# Single-shot prompt generation
anvil run huihui-qwen -p "Analyze this 260k token codebase and verify concurrency safety."
```

### Serving OpenAI-Compatible API with Anvil

```bash
# Launch server with multi-slot KV prefix caching
anvil serve --port 8080 --slots 4
```
