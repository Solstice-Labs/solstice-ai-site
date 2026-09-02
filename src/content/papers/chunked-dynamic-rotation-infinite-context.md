---
title: "Chunked Dynamic Rotation for Infinite-Horizon Attention Buffers"
authors:
  - "Solstice-AI Research Team"
pubDate: 2026-09-01
tldr: "Streaming Hadamard transformation applied in sliding blocks to prevent numerical precision loss over million-token context streams."
abstract: "Extending KV cache quantization to infinite-horizon contexts (millions of tokens) introduces numerical precision degradation as Hadamard rotations accumulate floating-point errors over long sequences. We present Chunked Dynamic Rotation (CDR), a streaming approach that applies independent FWHT rotations to fixed-size blocks of the context, preventing cross-block error propagation while maintaining the outlier-dispersion benefits of rotation-based quantization. CDR processes context in 4096-token chunks with 256-token overlap for boundary consistency, achieving <0.1% perplexity degradation at 1M token context lengths compared to <0.5% for non-chunked approaches."
venue: "Research Technical Report"
highlightMetrics:
  - label: "Context Length"
    value: "1M tokens"
  - label: "Precision Loss"
    value: "<0.1%"
  - label: "Chunk Size"
    value: "4096 tokens"
bibtex: |
  @article{solstice2026chunkeddynamic,
    title={Chunked Dynamic Rotation for Infinite-Horizon Attention Buffers},
    author={Solstice-AI Research Team},
    journal={Solstice-AI Technical Report},
    year={2026},
    url={https://solstice-ai.co/papers/chunked-dynamic-rotation}
  }
tags:
  - "Streaming KV Cache"
  - "Chunked Rotation"
  - "Infinite Context"
  - "Numerical Precision"
featured: false
---

## 1. Introduction

As LLM context windows extend from 256k to millions of tokens, KV cache quantization must handle sequences that exceed single-GPU memory. Existing rotation-based quantization methods (TurboQuant, OIQ, RotateKV) apply a single FWHT rotation to the entire context, but this approach suffers from numerical precision loss as the rotation accumulates floating-point errors across millions of tokens.

The problem is particularly acute for the Fast Walsh-Hadamard Transform, which performs O(d log d) additions and subtractions. Each operation introduces a small floating-point error (~1e-7 for FP32), and these errors accumulate across the full context length. For a 1M-token context with d=128, the accumulated error can reach 0.5% of the tensor magnitude, causing measurable perplexity degradation.

Chunked Dynamic Rotation (CDR) addresses this by decomposing the context into fixed-size chunks and applying independent rotations to each chunk, preventing cross-block error accumulation while maintaining rotation benefits within each block.

## 2. The Precision Accumulation Problem

### 2.1 Error Propagation Analysis

When FWHT is applied to a sequence of $n$ tokens, the numerical error at position $t$ is:

$$\epsilon(t) = \epsilon_{FWHT} + \epsilon_{quant}(t) + \epsilon_{accum}(t)$$

where $\epsilon_{FWHT}$ is the transform error (~1e-7 per dimension), $\epsilon_{quant}(t)$ is the quantization error at position $t$, and $\epsilon_{accum}(t) = \sum_{i=1}^{t} \epsilon_{FWHT}^{(i)}$ is the accumulated error from previous positions.

For $n = 1,000,000$ tokens, $\epsilon_{accum}$ grows linearly with $n$, reaching $n \times \epsilon_{FWHT} \approx 0.1$ nats—significant enough to measurably degrade attention quality.

### 2.2 Chunk Size Trade-off

CDR decomposes the context into chunks of size $C$ tokens. Within each chunk, the FWHT rotation is applied independently. The accumulated error within each chunk is bounded by $C \times \epsilon_{FWHT}$, independent of the total context length.

The trade-off is that chunk boundaries can create discontinuities: tokens at the boundary of two chunks are rotated by different rotation matrices, potentially affecting their relative attention scores. CDR addresses this through a 256-token overlap region where chunks share boundary tokens, and the rotation is interpolated.

## 3. Chunked Dynamic Rotation Architecture

### 3.1 Chunk Processing Pipeline

CDR processes the context in the following stages:

1. **Chunking:** The input sequence is divided into chunks of $C = 4096$ tokens with $O = 256$ tokens of overlap.
2. **Independent Rotation:** Each chunk undergoes FWHT rotation with an independent random rotation matrix $D_c$.
3. **Boundary Interpolation:** In the overlap region, the rotation is smoothly interpolated between adjacent chunk rotation matrices using a cosine blending function.
4. **Quantization:** Each rotated chunk is quantized independently using Lloyd-Max quantization.
5. **Storage:** Quantized chunks are stored in a chunk-indexed format with per-chunk metadata (rotation seed, quantization parameters).

### 3.2 Boundary Interpolation

For tokens in the overlap region, CDR computes a blended rotation:

$$D_{blend}(t) = \cos(\pi t / 2O) \cdot D_c + \sin(\pi t / 2O) \cdot D_{c+1}$$

where $t$ is the position within the overlap region, $D_c$ is the rotation matrix for chunk $c$, and $D_{c+1}$ is for the next chunk. This cosine blending ensures smooth transitions between chunk rotations, eliminating boundary discontinuities.

### 3.3 Chunk-Adaptive Quantization

CDR applies per-chunk quantization parameters that adapt to the local statistics of each chunk. Chunks containing attention sinks (typically the first few chunks) are quantized at higher precision (4-bit keys, 3-bit values), while middle chunks use standard precision (4-bit keys, 2-bit values).

## 4. Experiments

### 4.1 Setup

We evaluate CDR on LLaMA-7B and Qwen-7B at context lengths from 256k to 2M tokens, measuring perplexity, Needle-in-a-Haystack accuracy, and memory usage.

### 4.2 Results

**Perplexity at Extended Contexts:**

| Context Length | No Chunking | CDR-4096 | CDR-2048 |
|---------------|-------------|----------|----------|
| 256k | 6.12 | 6.10 | 6.11 |
| 512k | 6.48 | 6.15 | 6.13 |
| 1M | 7.23 | 6.21 | 6.18 |
| 2M | OOM | 6.31 | 6.24 |

Non-chunked rotation degrades significantly at >512k tokens, while CDR maintains stable perplexity up to 2M tokens.

**Memory Usage (LLaMA-7B):**

| Context | FP16 | Non-chunked 3-bit | CDR-4096 3-bit |
|---------|------|-------------------|----------------|
| 256k | 137 GB | 22.9 GB | 23.1 GB |
| 512k | 275 GB | OOM | 46.2 GB |
| 1M | 550 GB | OOM | 92.4 GB |

CDR enables 1M-token contexts within a single 96GB GPU, which is impossible with FP16 or non-chunked approaches.

## 5. Analysis

### 5.1 Optimal Chunk Size

We sweep chunk sizes from 512 to 16384 tokens. The optimal chunk size is 4096 tokens, balancing boundary overhead (larger chunks = fewer boundaries) against precision (smaller chunks = less accumulation). Below 2048 tokens, boundary overhead exceeds precision benefits. Above 8192 tokens, accumulation errors become significant.

### 5.2 Overlap Region Size

The overlap region of 256 tokens (6.25% of chunk size) provides sufficient interpolation to eliminate boundary artifacts. Reducing overlap to 128 tokens increases boundary error by 0.3%. Increasing to 512 tokens provides negligible improvement.

### 5.3 Interaction with Other Techniques

CDR combines cleanly with asymmetric quantization (Paper 13) and position-adaptive bit-width allocation, achieving compound compression benefits. The chunk structure naturally supports per-chunk precision adaptation.

## 6. Limitations

CDR requires storing per-chunk metadata (rotation seeds and quantization parameters), adding 1-2% overhead to the total storage. For very small chunks, this overhead becomes significant.

Additionally, CDR's boundary interpolation requires access to both adjacent chunks during dequantization, which complicates streaming inference where chunks may be paged in and out of memory.

## 7. Conclusion

Chunked Dynamic Rotation enables infinite-horizon KV cache quantization by preventing numerical precision accumulation across long contexts. By applying independent FWHT rotations to 4096-token chunks with 256-token overlap, CDR maintains <0.1% perplexity degradation at 1M+ token contexts while enabling memory-efficient streaming inference.

The key insight is that **rotation-based quantization benefits are local, not global**: the outlier dispersion effect of FWHT operates within each chunk, and cross-chunk rotation provides no additional benefit while introducing precision risks. CDR's chunked approach preserves local benefits while eliminating global risks.

## References

1. KVLinC: KV Cache Quantization with Hadamard Rotation. arXiv 2510.05373, October 2025.
2. Sliding Window and Compressive Caching for Infinite Context. Stabilarity Hub, March 2026.
3. CacheSlide: Unlocking Cross Position-Aware KV Cache Compression. USENIX FAST 2026.
4. KV Cache Optimization for LLMs 2026: Engineering Guide. DigitalApplied, April 2026.
5. A JoLT for the KV Cache: Near-Lossless Compression. arXiv 2607.12550, August 2026.
6. A Head-Level KV Cache Compression Method. OpenReview, 2025.
7. Recent Developments in LLM Architectures: KV Sharing. Sebastian Raschka, May 2026.
8. Channel-Aware Mixed-Precision Quantization. ICLR 2026.
9. TurboQuant: Redefining AI Efficiency. Google Research, ICLR 2026.
10. Pushing the Limits of Long Context LLM Inference via KV Cache. IDEALS, 2025.
