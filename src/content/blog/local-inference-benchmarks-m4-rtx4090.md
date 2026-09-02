---
title: "Sub-8B Inference Performance on Commodity Silicon: M4 Max vs RTX 4090"
description: "Empirical benchmarking of Solace-Sub8B checkpoints in INT4 AWQ, FP8, and GGUF across Apple Silicon and NVIDIA Ada Lovelace GPUs."
pubDate: 2026-09-01
author: "Solstice-AI Systems Engineering"
tags:
  - "Hardware"
  - "Inference"
  - "Benchmarks"
  - "GPU"
readingTime: "7 min read"
takeaways:
  - "Solace-3.8B AWQ achieves 218 tokens/sec on RTX 4090 and 142 tokens/sec on Apple M4 Max."
  - "Total VRAM consumption under INT4 is just 2.4 GB, enabling native local reasoning alongside IDEs."
  - "FP8 tensor core kernels on RTX 4090 provide 1.8x throughput over BF16 with zero accuracy loss."
featured: true
---

Running frontier-grade multi-step reasoning locally without cloud API latency requires matching calibrated weights with optimized hardware execution engines.

We benchmarked Solace-Sub8B student checkpoints across three common hardware setups:
1. **NVIDIA GeForce RTX 4090 (24GB VRAM, Ada Lovelace)**
2. **Apple MacBook Pro (M4 Max, 128GB Unified Memory)**
3. **NVIDIA GeForce RTX 3060 (12GB VRAM, Ampere)**

---

## Throughput & Latency Benchmarks (Tokens/Sec)

*Batch Size = 1, Context Length = 4,096 tokens, Generation Length = 1,024 tokens.*

| Checkpoint | Format | RTX 4090 (vLLM) | M4 Max (MLX) | RTX 3060 (AutoAWQ) |
| :--- | :--- | :--- | :--- | :--- |
| **Solace-3.8B** | INT4 AWQ | **218 tok/s** | **142 tok/s** | 88 tok/s |
| **Solace-7B** | FP8 Tensor Core | **164 tok/s** | N/A | N/A |
| **Solace-7B** | Q4_K_M GGUF | 112 tok/s | 94 tok/s | 46 tok/s |
| **Solace-7B** | BF16 (Full Precision) | 91 tok/s | 58 tok/s | 24 tok/s |

---

## Memory Footprint Breakdown

```
Solace-3.8B INT4 AWQ:   [====] 2.4 GB VRAM
Solace-7B Q4_K_M GGUF:  [========] 4.8 GB VRAM
Solace-7B FP8:          [============] 7.2 GB VRAM
Standard 70B Base FP16: [==================================================] 140+ GB
```

By keeping the working set under 5 GB, developers can run continuous agentic verification loops and code refactoring workflows without stalling local system responsiveness.
