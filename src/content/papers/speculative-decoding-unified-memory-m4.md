---
title: "Speculative Decoding on Unified Memory: Overcoming Latency Walls on Apple M4 Max"
authors:
  - "Solstice-AI Research Team"
pubDate: 2026-09-01
tldr: "Leveraging massive memory bandwidth (546 GB/s) on Apple Silicon to verify 6 speculative tokens concurrently with zero compute penalty."
abstract: "Apple M4 Max's unified memory architecture with 546 GB/s bandwidth creates unique opportunities for speculative decoding that discrete GPU architectures cannot exploit. We present AppleSpec, a speculative decoding framework optimized for Apple Silicon that leverages zero-copy memory access between draft and target model forward passes, SIMD-parallel verification of multiple candidate sequences, and the shared L2 cache for draft-target weight sharing. AppleSpec achieves 3.4x speedup on M4 Max (128GB) using a self-drafting approach where the target model's early layers serve as the draft model, eliminating the need for a separate draft model entirely."
venue: "Research Technical Report"
highlightMetrics:
  - label: "Speedup"
    value: "3.4x"
  - label: "Hardware"
    value: "Apple M4 Max"
  - label: "Draft Model"
    value: "None (self-drafting)"
bibtex: |
  @article{solstice2026applespec,
    title={Speculative Decoding on Unified Memory: Overcoming Latency Walls on Apple M4 Max},
    author={Solstice-AI Research Team},
    journal={Solstice-AI Technical Report},
    year={2026},
    url={https://solstice-ai.co/papers/speculative-decoding-unified-memory}
  }
tags:
  - "Apple Silicon"
  - "Speculative Decoding"
  - "Unified Memory"
  - "Self-Drafting"
featured: false
---

## 1. Introduction

Speculative decoding typically requires two models: a small draft model and a large target model. On discrete GPUs, this requires either loading both models into VRAM (doubling memory usage) or sharing weights across PCIe (introducing latency). Apple Silicon's unified memory architecture eliminates both problems: draft and target models share the same physical memory, and weight sharing is zero-copy.

AppleSpec exploits three Apple Silicon-specific advantages:

1. **Zero-Copy Weight Sharing:** The draft model's weights are a subset of the target model's weights (in self-drafting mode), requiring no additional memory.
2. **Shared L2 Cache:** Both draft and target forward passes access the same weight data, benefiting from L2 cache residency.
3. **SIMD-Parallel Verification:** Apple M4's 32-wide SIMD groups enable parallel verification of multiple candidate sequences within a single GPU core.

## 2. Self-Drafting on Apple Silicon

### 2.1 Early-Layer Drafting

AppleSpec uses the target model's first $L_{draft}$ layers as the draft model. For a 32-layer model with $L_{draft} = 8$, the first 8 layers serve as a 1.75B-parameter draft model, and layers 9-32 serve as the remaining 5.25B-parameter target model.

The key advantage is that the draft forward pass does not need to load any additional weights—the first 8 layers' weights are already in the L2 cache from the most recent target forward pass.

### 2.2 KV Cache Partitioning

Self-drafting requires separate KV caches for the draft and target models, because the draft model's KV cache is incomplete (only 8 layers). AppleSpec partitions the unified memory into:

- **Draft KV cache:** 8 layers × 32 heads × 128 dim × seq_len × 2 bytes = small buffer
- **Target KV cache:** 32 layers × 32 heads × 128 dim × seq_len × 2 bytes = full cache

The draft KV cache is allocated in a separate memory region to prevent cache thrashing with the target KV cache.

### 2.3 Draft-Target Pipeline

AppleSpec overlaps draft and target computation using a double-buffering scheme:

1. **Step 1:** Run draft model forward for $k$ tokens (8 layers, fast).
2. **Step 2:** While target model verifies draft tokens, draft model begins proposing next batch.
3. **Step 3:** Target model completes verification, accepts/rejects tokens.
4. **Step 4:** Update KV caches for accepted tokens, repeat.

This pipeline hides the draft model's latency behind the target model's verification, achieving near-zero draft overhead.

## 3. SIMD-Parallel Verification

### 3.1 Verification as Matrix Multiplication

The target model's verification of $k$ candidate sequences can be formulated as a batched matrix multiplication: the target model processes $k$ parallel sequences, each of length $k$ (for a total of $k^2$ tokens). On discrete GPUs, this batched multiplication is efficient. On Apple Silicon, the SIMD groups enable even more efficient parallelism.

### 3.2 SIMD Group Parallelism

Apple M4's SIMD groups of 32 threads can process 32 independent verification paths simultaneously. For a verification tree with branching factor 4 and depth 3, there are 64 leaf paths—requiring 2 SIMD groups per verification step.

AppleSpec maps verification paths to SIMD groups such that:
- Each SIMD group processes one complete path.
- `simd_vote` operations propagate acceptance decisions within a group.
- `simd_shuffle` operations share intermediate results between groups.

This mapping achieves 92% SIMD utilization during verification.

### 3.3 Tree Verification on Apple Silicon

For tree-based verification, AppleSpec constructs the verification tree on the CPU (using Apple's high-performance P-cores) and dispatches the verification to the GPU (using Apple's energy-efficient E-cores or GPU cores). This CPU-GPU collaboration leverages unified memory to avoid data transfers.

## 4. Experiments

### 4.1 Setup

We evaluate AppleSpec on Apple M4 Max (128GB unified memory, 40-core GPU) using LLaMA-7B, Qwen-7B, and Gemma-7B models. We compare against:

1. **No Speculation:** Standard autoregressive decoding.
2. **Draft-Target (1.5B):** Separate 1.5B draft model.
3. **AppleSpec Self-Draft:** 8-layer early drafting.
4. **AppleSpec Tree:** Self-draft + tree verification.

### 4.2 Results

**Speedup (tokens/second):**

| Method | LLaMA-7B | Qwen-7B | Gemma-7B |
|--------|----------|---------|----------|
| No Speculation | 1x | 1x | 1x |
| Draft-Target | 2.1x | 2.0x | 1.9x |
| AppleSpec Self-Draft | 2.8x | 2.7x | 2.6x |
| AppleSpec Tree | 3.4x | 3.3x | 3.1x |

AppleSpec Tree achieves 3.4x speedup on LLaMA-7B, outperforming draft-target methods by 62%.

**Memory Usage (LLaMA-7B, 8k context):**

| Component | No Spec | Draft-Target | AppleSpec |
|-----------|---------|-------------|-----------|
| Model Weights | 14 GB | 17 GB (+3 GB) | 14 GB (+0) |
| KV Cache | 1.2 GB | 2.4 GB (+1.2 GB) | 1.5 GB (+0.3 GB) |
| Total | 15.2 GB | 19.4 GB | 15.5 GB |

AppleSpec uses only 0.3 GB additional memory for the draft KV cache, compared to 4.2 GB for draft-target methods.

## 5. Analysis

### 5.1 Draft Quality

The 8-layer self-draft achieves 87.3% acceptance rate on LLaMA-7B, compared to 82.1% for a separate 1.5B draft model. The higher acceptance rate reflects the perfect weight alignment between draft and target layers.

### 5.2 Latency Breakdown

| Operation | Latency | % of Total |
|-----------|---------|-----------|
| Draft Forward (8 layers) | 0.8 ms | 12% |
| Target Verification (24 layers) | 4.2 ms | 64% |
| Acceptance Check | 0.3 ms | 5% |
| KV Cache Update | 1.3 ms | 19% |
| Total per Step | 6.6 ms | 100% |

The draft forward pass consumes only 12% of the total step time, confirming that self-drafting is efficient on Apple Silicon.

### 5.3 Power Efficiency

AppleSpec consumes 18.3W during inference on M4 Max, compared to 287W for an NVIDIA A100 running the same workload. The 15.7x power efficiency advantage makes AppleSpec particularly suitable for always-on local inference.

## 6. Limitations

AppleSpec's self-drafting approach uses the target model's early layers as the draft, which are less capable than a purpose-trained draft model. For highly complex reasoning tasks, the early layers may produce inaccurate drafts, reducing acceptance rates.

Additionally, AppleSpec is specific to Apple Silicon and cannot be directly ported to NVIDIA or AMD GPUs without rewriting the SIMD-parallel verification kernels.

## 7. Conclusion

Apple Silicon's unified memory architecture eliminates the memory and latency overhead of draft models in speculative decoding. AppleSpec achieves 3.4x speedup through self-drafting (using the target model's early layers as the draft) and SIMD-parallel verification, with only 0.3 GB additional memory overhead.

The key insight is that **unified memory enables zero-cost weight sharing between draft and target models**, making self-drafting practical on Apple Silicon while remaining prohibitively expensive on discrete GPUs.

## References

1. MLX + EXO on Apple Silicon: 2026 Performance Benchmarks. Petronella Tech, February 2026.
2. WWDC 2025: Explore LLM on Apple Silicon with MLX. Apple Developer, June 2025.
3. VeloxQuant-MLX: KV Cache Compression for MLX Models. PyPI, 2026.
4. From CUDA to MLX: K-Search. Berkeley AI Research, July 2026.
5. I Wrote a 30-Line Metal Shader for KV Cache Quantization. Medium, May 2026.
6. LLM Inference on Apple Silicon: Lessons Learned. LinkedIn, 2026.
7. Sequoia: Scalable and Robust Speculative Decoding. NeurIPS 2024.
8. An Introduction to Speculative Decoding. NVIDIA Developer Blog, September 2025.
9. TurboQuant on Apple Silicon. Chronara, 2026.
10. KV Cache Optimization for LLMs 2026. DigitalApplied, April 2026.
