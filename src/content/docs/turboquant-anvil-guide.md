---
title: "Anvil: TurboQuant KV Cache & Speculative Decoding Engine"
description: "Deploying Google TurboQuant FWHT-rotated KV cache compression, single-word model launches, and high-throughput inference with the Anvil CLI."
category: "tooling"
order: 2
lastUpdated: 2026-08-31
githubUrl: "https://github.com/Solstice-Labs/anvil"
specs:
  "Upstream Algorithm": "Google TurboQuant (FWHT + QJL)"
  "Compression Factor": "4.3x–6.4x KV Cache Reduction"
  "Context Capacity": "262k Context in ~4 GB VRAM"
  "CLI Interface": "anvil run <model> / anvil <model>"
  "Official Repository": "Solstice-Labs/anvil"
supportedFormats:
  - "Anvil CLI (anvil run / anvil pull)"
  - "TurboQuant Presets (turbo4 / turbo3 / turbo2)"
  - "Metal TurboFlash (Apple Silicon)"
  - "CUDA / Vulkan / HIP"
---

## Overview

As frontier language models scale to massive context windows (32k–262k+ tokens), the Key-Value (KV) cache becomes the primary memory bottleneck. **Google TurboQuant (TQ)** solves this by rotating high-dimensional activation vectors using Fast Walsh-Hadamard Transforms (FWHT), eliminating outlier activation spikes without requiring expensive retraining.

**Anvil ([`Solstice-Labs/anvil`](https://github.com/Solstice-Labs/anvil))** is the dedicated inference engine and model management CLI:
1. **TurboQuant WHT Rotation:** Fast Walsh-Hadamard Transform ($O(d \log d)$) for **~4.3× to 6.4× KV cache compression** with backend-native kernels (`TurboFlash` on Metal, CUDA, Vulkan, HIP).
2. **Single-Word Launch:** Run any registered checkpoint instantly (`anvil <model>` or `anvil run <model>`).
3. **Speculative Decoding:** Integrated NextN draft reuse and Gemma 4 Multi-Token Prediction (MTP).

```
Input Tokens ──► FWHT Rotation ──► Lloyd-Max Quantizer ──► turbo3 / turbo2 KV Cache
                      │                                            │
                      ▼                                            ▼
               QJL Error Correction ◄────────────────────── Attention Compute (Flash-Attn)
```

---

## 1. How Google TurboQuant Works

### The Outlier Bottleneck
Standard INT8/INT4 quantization fails on raw KV tensors because a few outlier channels carry disproportionately high amplitudes. Clamping these outliers degrades reasoning, while expanding quantization bounds wastes bit-depth on near-zero dimensions.

### The TurboQuant Solution
1. **Decorrelation via Fast Walsh-Hadamard Transform (FWHT):**
   TurboQuant rotates key-value vectors into a coordinate frame where energy is uniformly distributed:
   $$\mathbf{v}' = \frac{1}{\sqrt{d}} \mathbf{H}_d \mathbf{v}$$
   Computed in $O(d \log d)$ operations via butterfly network kernels.
2. **Scalar Quantization:** The rotated distribution approaches a zero-mean Gaussian, enabling optimal **Lloyd-Max quantization** at 2-bit, 3-bit, or 4-bit precision.
3. **Quantized Johnson-Lindenstrauss (QJL) Correction:** Preserves inner-product similarity for exact attention score calculation.

---

## 2. Using the Anvil CLI (`anvil`)

### Pulling Checkpoints

```bash
# Pull directly from Hugging Face Hub into Anvil registry
anvil pull hf:Solstice-AI/Huihui-Qwen3.6-35B-A3B-Claude-4.7-Opus-abliterated-Q8_0-GGUF
```

### Running 262k Long-Context Inference

To compress the full **262k context window on Qwen 3.8 and derivative models into ~4 GB of VRAM**, pass the TurboQuant cache presets:

```bash
# Interactive chat REPL with 262k context and TurboQuant 3-bit KV cache
anvil run huihui-qwen --ctx 262144 --type-k turbo4 --type-v turbo3

# Save flags so future runs launch instantly with a single word
anvil run huihui-qwen --ctx 262144 --type-k turbo4 --type-v turbo3 --save

# Now run instantly with one word
anvil huihui-qwen

# Single-shot prompt generation
anvil run huihui-qwen -p "Analyze this 260,000 token code repository and verify all concurrency safety invariants..."
```

### TurboQuant KV Presets

| Preset | Command Flags | Compression vs FP16 | Recommended Use Case |
| :--- | :--- | :--- | :--- |
| **Recommended** | `--type-k turbo4 --type-v turbo3` | **~4.3×** | **Sweet spot** (262k in ~4.1 GB VRAM, ~1% loss) |
| **Quality+** | `--type-k q8_0 --type-v turbo3` | **~3.0×** | Near-lossless precision (<1% quality loss) |
| **High Compression** | `--type-k turbo4 --type-v turbo2` | **~6.1×** | Maximum context density (262k in ~2.8 GB VRAM) |

---

## 3. Serving OpenAI-Compatible API Server

```bash
# Launch multi-slot server with cross-request prefix caching
anvil serve --port 8080 --slots 4
```

---

## 4. Empirical Performance Benchmarks

*Evaluated on Qwen 3.8 27B and Huihui 35B across 262k context lengths on 1x RTX 4090 / M4 Max:*

| KV Cache Configuration | 32k Context | 128k Context | 262k Context | Degradation vs FP16 |
| :--- | :--- | :--- | :--- | :--- |
| **Standard FP16** | 2.3 GB | 9.2 GB | **18.6 GB (OOM risk)** | Baseline (0.00) |
| **Standard INT8** | 1.2 GB | 4.8 GB | 9.6 GB | +0.18 PPL |
| **Anvil TurboQuant 4-bit (`turbo4`)** | 0.6 GB | 2.4 GB | **4.9 GB** | **+0.03 PPL** |
| **Anvil TurboQuant 3-bit (`turbo3`)** | 0.5 GB | 2.0 GB | **4.1 GB** | **+0.05 PPL** |
| **Anvil TurboQuant 2-bit (`turbo2`)** | 0.35 GB | 1.4 GB | **2.8 GB** | **+0.12 PPL** |
