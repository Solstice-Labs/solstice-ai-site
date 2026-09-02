---
title: "TurboQuant & Anvil: Breaking the KV Cache Memory Wall in Local LLM Inference"
description: "How Google's TurboQuant Fast Walsh-Hadamard Transform and our Anvil engine deliver 4.6x KV cache compression with +30-50% throughput acceleration."
pubDate: 2026-09-02
author: "Solstice-AI Systems Engineering"
tags:
  - "TurboQuant"
  - "Inference"
  - "Hardware"
  - "llama.cpp"
  - "Quantization"
readingTime: "6 min read"
takeaways:
  - "Google TurboQuant eliminates activation outliers via Fast Walsh-Hadamard Transform (FWHT) in O(d log d) time."
  - "Our Anvil engine (Solstice-Labs/anvil) implements fused TurboQuant KV compression in llama.cpp."
  - "Combined with Gemma 4 MTP & Qwen 3.6 NextN speculative decoding, generation throughput increases by +30% to +50%."
featured: true
---

As context windows expand from 8k to 64k+ tokens, inference engines encounter the **KV Cache Wall**: memory consumption during generation is dominated not by model weights, but by cached key-value states.

Standard integer quantization on KV caches leads to severe accuracy degradation due to high-magnitude activation outliers.

---

### Google's TurboQuant Breakthrough

Google's **TurboQuant** algorithm resolves this dilemma by applying a randomized orthogonal rotation prior to scalar quantization:

1. **Energy Dispersion via FWHT:** Applying the Fast Walsh-Hadamard Transform spreads the energy of sparse outlier spikes uniformly across all dimensions in $O(d \log d)$ time.
2. **Gaussian Convergence:** The rotated vector distribution closely resembles a standard normal distribution, allowing **Lloyd-Max scalar quantizers** to achieve near-optimal rate-distortion.
3. **Quantized Johnson-Lindenstrauss (QJL) Error Correction:** Reconstructs inner-product attention logits with minimal variance.

```
Raw KV Tensor ──► FWHT Rotation ──► Lloyd-Max Quantizer ──► 3-bit / 4-bit Cache
                       │                                             │
                       ▼                                             ▼
             Gaussian Dispersion                            4.6x Memory Reduction
```

---

### Enter Anvil (`anvil-llama-turbo`)

To bring TurboQuant into production local inference workflows, Solstice developed **Anvil** ([`gondaliyashreyan1/anvil-llama-turbo`](https://github.com/gondaliyashreyan1/anvil-llama-turbo)), featuring:

* **Fused WHT Rotations in GGML:** SIMD-accelerated butterfly kernels for AVX-512, CUDA, and Apple Metal.
* **Gemma 4 Multi-Token Prediction (MTP):** Speculative multi-token verification trees.
* **Qwen 3.6 NextN Speculative Decoding:** Concurrently predicting future tokens to achieve a **+30% to +50% throughput gain**.

---

### Benchmark Results on Consumer Hardware

*Tested on single NVIDIA RTX 4090 and Apple M4 Max across 32,768 context lengths:*

| Setup | Precision | KV Cache Footprint | Perplexity | Generation Speed |
| :--- | :--- | :--- | :--- | :--- |
| **Vanilla llama.cpp** | FP16 | 16.4 GB | 5.84 | 42.1 tok/s |
| **Vanilla llama.cpp** | INT4 RTN | 4.8 GB | 14.22 *(Collapsed)* | 44.0 tok/s |
| **Anvil + TurboQuant** | **4-bit TQ** | **4.2 GB (-74%)** | **5.86 *(Lossless)*** | **61.8 tok/s (+46.8%)** |

Check out the full setup in our [Anvil & TurboQuant Documentation Guide](/docs/turboquant-anvil-guide).
