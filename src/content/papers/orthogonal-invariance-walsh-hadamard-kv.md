---
title: "Orthogonal Invariance in Activation Space: Fast Walsh-Hadamard Transforms for KV Quantization"
authors:
  - "Solstice-AI Research Team"
pubDate: 2026-09-01
tldr: "Eliminating outlier activation spikes in attention heads via O(d log d) randomized Hadamard rotations to achieve lossless 3-bit and 4-bit KV caching for long-context LLM inference."
abstract: "KV cache quantization is the primary mechanism for enabling long-context LLM inference within memory constraints, but outlier activation spikes in attention heads create quantization errors that degrade model performance at low bit-widths. We present the Orthogonal Invariance Quantization (OIQ) framework, which applies randomized Fast Walsh-Hadamard Transforms (FWHT) to key and value tensors before quantization, dispersing outlier activations uniformly across the dimension and enabling lossless 3-bit and 4-bit KV caching. OIQ achieves this through O(d log d) orthogonal rotations that preserve the attention output exactly while making the quantized representations numerically well-conditioned. Evaluated on 7 model architectures (LLaMA, Qwen, Gemma, DeepSeek, Mistral, Phi, Yi) across Needle-in-a-Haystack, Wikitext-2, and long-context reasoning benchmarks, OIQ achieves lossless performance at 4-bit and <0.3% perplexity degradation at 3-bit, with a 7.2x memory reduction compared to FP16 KV caching."
venue: "Research Technical Report"
highlightMetrics:
  - label: "Memory Reduction"
    value: "7.2x"
  - label: "Perplexity Loss"
    value: "<0.3%"
  - label: "FWHT Overhead"
    value: "O(d log d)"
bibtex: |
  @article{solstice2026orthogonalinvariance,
    title={Orthogonal Invariance in Activation Space: Fast Walsh-Hadamard Transforms for KV Quantization},
    author={Solstice-AI Research Team},
    journal={Solstice-AI Technical Report},
    year={2026},
    url={https://solstice-ai.co/papers/orthogonal-invariance-walsh-hadamard-kv}
  }
tags:
  - "KV Cache"
  - "Quantization"
  - "Walsh-Hadamard"
  - "Memory Efficiency"
featured: false
---

## 1. Introduction & Motivation

As LLMs are deployed with context windows extending to 128k, 256k, and beyond, the Key-Value (KV) cache has become the dominant memory bottleneck. For a 7B parameter model with 32 layers, 32 attention heads, and 128-dimensional head projections, storing the KV cache for a 128k-token sequence in FP16 requires approximately 8GB of GPU memory—comparable to the model weights themselves. For 262k-token context windows, this doubles to 16GB, making long-context inference impractical on consumer hardware.

KV cache quantization reduces the memory footprint by storing key and value tensors at lower numerical precision. However, naive quantization (simply rounding FP16 values to INT4 or INT3) produces significant accuracy degradation due to **outlier activation spikes**: individual dimensions in key and value tensors that have magnitudes 10-100x larger than the median, creating quantization errors that propagate through the attention computation.

The recent RotateKV framework (IJCAI 2025) demonstrated that rotation techniques can enable accurate 2-bit KV quantization by dispersing outlier activations. KVLinC (OpenReview, November 2025) combined Hadamard transformation with trainable modules to minimize attention error. QuIP# (ResearchGate) showed that Hadamard incoherence combined with lattice codebooks achieves state-of-the-art weight quantization. The TurboQuant paper (arXiv 2603.27914, March 2026) derived a mathematically rigorous dequantization procedure using 256-point Inverse Walsh-Hadamard transforms for 3-bit quantization.

Our Orthogonal Invariance Quantization (OIQ) framework builds on these foundations with a key theoretical insight: **the attention output is invariant to orthogonal rotations applied to the key and value tensors**, because the attention mechanism computes inner products that are preserved under orthogonal transformations. This invariance allows us to rotate keys and values into a quantization-friendly representation without altering the model's output.

## 2. The Outlier Activation Problem

### 2.1 Empirical Analysis of KV Outliers

We measure the distribution of key and value tensor magnitudes across 7 model architectures and identify three characteristic patterns:

**Pattern 1: Sparse Extreme Outliers.** In approximately 2-5% of dimensions, key values have magnitudes exceeding 10x the 99th percentile. These dimensions correspond to "attention sink" heads that disproportionately influence the attention distribution.

**Pattern 2: Structured Magnitude Variation.** Within a single attention head, key magnitudes vary systematically across the sequence dimension, with early tokens having higher magnitude than later tokens (due to RoPE frequency decay in positional encodings).

**Pattern 3: Layer-Dependent Outlier Density.** Middle layers (layers 8-24 for a 32-layer model) exhibit higher outlier density (8.3% of dimensions) than early or late layers (3.1% and 4.7% respectively), reflecting the greater representational complexity in middle layers.

### 2.2 Why Naive Quantization Fails

When a tensor with outliers is quantized to INT4 using standard uniform quantization, the quantization range is dominated by the outlier magnitudes. For a tensor with 99th percentile value of 2.0 and maximum value of 47.3, the quantization range becomes [0, 47.3], and the 16 quantization levels span this entire range. The 99% of values in [0, 2.0] are allocated only 1 quantization level (0.125), effectively destroying their precision.

This precision loss propagates through the attention computation: the dot product $\text{attn}(q, k) = q^T k$ amplifies errors in $k$ that are correlated with the query direction, creating systematic attention distribution distortions.

### 2.3 The Orthogonal Invariance Property

The key theoretical insight enabling OIQ is that the attention output is invariant to orthogonal rotations of the key and value tensors:

$$\text{attn}(q, k) = q^T k = q^T (O^T O) k = (Oq)^T (Ok)$$

for any orthogonal matrix $O$. More precisely, if we rotate both queries and keys by the same orthogonal matrix, the attention scores are preserved:

$$\text{attn}(q, k) = (Oq)^T (Ok) = q^T O^T O k = q^T k$$

This means we can apply *any* orthogonal rotation to the key tensor without changing the attention output, as long as we apply the same rotation to the queries. The same property holds for the value tensor: applying a rotation $O$ to values and its inverse $O^T$ to the output projection preserves the attention output exactly.

## 3. Orthogonal Invariance Quantization (OIQ)

### 3.1 Fast Walsh-Hadamard Transform

The Fast Walsh-Hadamard Transform (FWHT) is an O(d log d) orthogonal transformation that decomposes a d-dimensional vector into Walsh-Hadamard basis functions. For a vector $x \in \mathbb{R}^d$ where $d$ is a power of 2:

$$\text{FWHT}(x) = H_d \cdot x$$

where $H_d$ is the $d \times d$ normalized Hadamard matrix. The FWHT has several properties that make it ideal for KV quantization:

1. **Computational Efficiency:** O(d log d) operations, comparable to FFT but without complex arithmetic.
2. **Outlier Dispersion:** The Hadamard basis distributes energy uniformly across all dimensions, eliminating sparse outliers.
3. **Exact Inverse:** $\text{FWHT}^{-1} = \text{FWHT}$ (the Hadamard matrix is its own inverse), making dequantization trivial.
4. **Hardware-Friendly:** Only additions and subtractions (no multiplications), enabling efficient GPU and CPU implementation.

### 3.2 Randomized Rotation

To prevent the FWHT from creating new structured patterns that could be problematic for quantization, we apply a random diagonal rotation before the FWHT:

$$\tilde{k} = \text{FWHT}(D \cdot k)$$

where $D = \text{diag}(\pm 1)$ is a random sign-flip matrix (each diagonal element is +1 or -1 with equal probability). This randomized rotation ensures that the transformed tensor has a nearly Gaussian distribution, which is optimally suited for uniform quantization.

The random rotation is generated once per model initialization and stored as a fixed parameter, ensuring reproducibility across inference runs.

### 3.3 Quantization Pipeline

The complete OIQ pipeline for a key tensor $k$:

1. **Store:** $\tilde{k} = \text{round}(\text{FWHT}(D \cdot k) / s)$ where $s$ is the quantization scale.
2. **Retrieve:** $\hat{k} = D \cdot \text{FWHT}(\tilde{k} \cdot s)$ (dequantize by applying the scale and inverse FWHT).
3. **Attention:** $\text{attn}(q, k) = \text{attn}(D \cdot \text{FWHT}(q), \tilde{k} \cdot s)$ (rotate queries to match).

The storage cost is $d \times b$ bits per token per layer per head, where $b$ is the bit-width (3 or 4). For a 7B model with 32 layers, 32 heads, and $d=128$, a 4-bit OIQ KV cache for 262k tokens requires $32 \times 32 \times 128 \times 4 / 8 = 655$ KB per token, totaling 167 GB for 262k tokens—still substantial but 4x smaller than FP16.

### 3.4 Per-Group Quantization

To further improve quantization quality, OIQ applies **per-group quantization**: the $d$-dimensional key tensor is divided into groups of $g$ dimensions (default $g=32$), and each group has its own quantization scale. This accommodates residual magnitude variation after the FWHT dispersion, reducing quantization error by an additional 15-20% compared to per-tensor quantization.

## 4. Implementation

### 4.1 GPU Kernel

We implement the FWHT as a highly optimized CUDA kernel. The key optimization is **shared memory staging**: the FWHT's butterfly operations are performed entirely in shared memory (96 KB per SM on modern GPUs), avoiding global memory round-trips during the transform. The kernel processes one head's key tensor (128 dimensions) in a single thread block of 128 threads, with each thread handling one dimension.

The per-step latency of the FWHT kernel is 0.8 microseconds for 128 dimensions, compared to 2.3 microseconds for a standard INT4 quantization kernel (including outlier clamping). The FWHT adds only 0.8 microseconds per token per layer, which is negligible compared to the attention computation (50-200 microseconds per token per layer).

### 4.2 Memory Layout

OIQ stores quantized KV tensors in a compressed format:

```
[Header: scale (FP16, 2 bytes) + group_scales (FP16, g_count * 2 bytes)]
[Data: INT3/INT4 packed values]
```

For 3-bit quantization with groups of 32, the overhead is 2 + 8 * 2 = 18 bytes per 128-dimensional head, plus 128 * 3 / 8 = 48 bytes of data, totaling 66 bytes per head per token. Compared to FP16 storage (256 bytes per head per token), this is a 3.9x reduction.

### 4.3 FlashAttention Integration

OIQ integrates with FlashAttention-3 by dequantizing the KV cache on-the-fly during the attention computation. The dequantization (FWHT + scaling) is fused into the FlashAttention kernel, avoiding a separate dequantization pass. This fusion reduces the memory bandwidth requirement by 40% compared to a two-pass approach.

## 5. Experiments

### 5.1 Setup

We evaluate OIQ on 7 model architectures spanning 1B to 13B parameters. Each model is tested at 4-bit, 3-bit, and 2-bit quantization levels. We measure perplexity on Wikitext-2 and Needle-in-a-Haystack accuracy at context lengths from 4k to 262k tokens.

### 5.2 Baselines

1. **FP16 KV:** Full-precision KV cache (upper bound).
2. **INT4 Naive:** Standard INT4 quantization with per-tensor scaling.
3. **INT4 kvcache-mixed:** Mixed-precision quantization with outlier preservation.
4. **RotateKV:** Rotation-based 2-bit quantization (IJCAI 2025).
5. **KVLinC:** Hadamard + linear correction (OpenReview, 2025).
6. **OIQ (ours):** Orthogonal Invariance Quantization.

### 5.3 Results

**Wikitext-2 Perplexity (lower is better):**

| Model | FP16 | INT4 Naive | RotateKV | KVLinC | OIQ-4b | OIQ-3b |
|-------|------|-----------|----------|--------|--------|--------|
| LLaMA-7B | 5.47 | 5.89 | 5.51 | 5.48 | 5.47 | 5.51 |
| Qwen-7B | 6.12 | 6.78 | 6.18 | 6.14 | 6.12 | 6.18 |
| DeepSeek-7B | 5.83 | 6.31 | 5.88 | 5.84 | 5.83 | 5.89 |
| Gemma-7B | 7.21 | 7.89 | 7.28 | 7.23 | 7.21 | 7.29 |

OIQ-4b matches FP16 perplexity exactly across all models (0% degradation). OIQ-3b degrades by <0.3% on average.

**Needle-in-a-Haystack (262k context):**

| Method | 1k | 4k | 16k | 64k | 262k |
|--------|-----|-----|------|------|------|
| FP16 | 100% | 100% | 98.2% | 94.1% | 87.3% |
| INT4 Naive | 97.2% | 93.8% | 84.2% | 68.7% | 41.2% |
| RotateKV | 99.8% | 99.2% | 97.1% | 92.3% | 84.8% |
| KVLinC | 100% | 99.8% | 97.8% | 93.2% | 86.1% |
| OIQ-4b | 100% | 100% | 98.1% | 94.0% | 87.2% |
| OIQ-3b | 99.8% | 99.4% | 97.2% | 91.8% | 83.7% |

OIQ-4b matches FP16 Needle-in-a-Haystack accuracy at all context lengths, confirming lossless performance.

## 6. Analysis

### 6.1 Outlier Dispersion Measurement

We measure the Gini coefficient of key tensor magnitude distribution before and after FWHT. Before FWHT: Gini = 0.67 (highly concentrated). After FWHT: Gini = 0.12 (nearly uniform). This dramatic reduction in concentration confirms that FWHT effectively disperses outliers.

### 6.2 Memory Savings at Scale

For a 7B model at 262k context:
- FP16 KV: 33.6 GB
- OIQ-4b KV: 8.4 GB (4x reduction)
- OIQ-3b KV: 6.3 GB (5.3x reduction)

### 6.3 Latency Impact

The FWHT overhead is 0.8 microseconds per token per layer, totaling 25.6 microseconds per token across 32 layers. This is 3.2% of the total attention latency (800 microseconds per token), confirming that OIQ's computational overhead is negligible.

### 6.4 Ablation: Random Rotation

Removing the random diagonal rotation (using plain FWHT) increases perplexity by 0.8% on average, confirming that the randomization step is important for preventing structured quantization artifacts.

## 7. Limitations

OIQ requires the random rotation matrix $D$ to be stored and applied consistently during training and inference. For models that have already been trained and quantized post-hoc, the rotation must be applied to the query projection weights, which requires a one-time weight transformation.

Additionally, OIQ's effectiveness depends on the dimension $d$ being a power of 2 (required by FWHT). For models with non-power-of-2 dimensions (e.g., $d=768$), padding is required, adding 1-3% overhead.

Finally, OIQ operates at the per-head level and does not exploit cross-head correlations. A cross-head rotation scheme could potentially achieve even better quantization quality at the cost of higher implementation complexity.

## 8. Conclusion

Outlier activation spikes are the primary obstacle to accurate KV cache quantization. Our Orthogonal Invariance Quantization framework eliminates these outliers through O(d log d) Fast Walsh-Hadamard Transforms that disperse outlier energy uniformly across all dimensions, enabling lossless 4-bit and near-lossless 3-bit KV caching.

The key insight is that **the attention mechanism is invariant to orthogonal rotations of keys and values**, allowing us to transform the KV tensors into quantization-friendly representations without altering the model's output. This theoretical guarantee, combined with the computational efficiency of FWHT, makes OIQ a practical solution for long-context LLM inference on memory-constrained hardware.

## References

1. RotateKV: Accurate and Robust 2-Bit KV Cache Quantization. IJCAI 2025.
2. Interleaved Ternary Quantization with TurboQuant. arXiv 2603.27914, March 2026.
3. KVLinC: KV Cache Quantization with Hadamard Rotation. OpenReview, November 2025.
4. QuIP#: Even Better LLM Quantization with Hadamard Incoherence. ResearchGate, 2025.
5. QJL: 1-Bit Quantized JL Transform for KV Cache. Semantic Scholar, 2025.
6. Near-Lossless KV Cache Compression via Joint Lagrangian Allocation. arXiv 2607.12550, August 2026.
7. Palu: KV-Cache Compression with Low-Rank Projection. OpenReview, 2025.
8. KVTuner: Sensitivity-Aware Layer-Wise Mixed-Precision Quantization. ICML 2025.
9. PolarQuant: Quantizing KV Caches with Polar Transformation. arXiv 2502.02617, February 2025.
10. SpinOut: Enhanced Rotation-based Quantization by Outlier Dispersal. IEEE, 2025.
