---
title: "Entropy-Adaptive Speculative Lengths: Halting Draft Generation on Ambiguous Logical Forks"
authors:
  - "Solstice-AI Research Team"
pubDate: 2026-09-01
tldr: "Terminating speculative draft chains immediately when branch entropy exceeds a dynamic threshold to avoid wasted verification FLOPs."
abstract: "Fixed-length speculative decoding wastes verification compute when the draft model encounters ambiguous token positions where multiple continuations are equally likely. We present Entropy-Adaptive Speculation (EAS), a framework that dynamically adjusts the draft chain length based on the draft model's token-level entropy, halting generation when entropy exceeds a threshold that indicates low acceptance probability. EAS saves 38% of wasted verification FLOPs while maintaining 95.1% of the throughput of fixed-length speculation, achieving 2.6x speedup on 7B models."
venue: "Research Technical Report"
highlightMetrics:
  - label: "FLOPs Saved"
    value: "38%"
  - label: "Throughput Retention"
    value: "95.1%"
  - label: "Speedup"
    value: "2.6x"
bibtex: |
  @article{solstice2026entropyadaptive,
    title={Entropy-Adaptive Speculative Lengths: Halting Draft Generation on Ambiguous Logical Forks},
    author={Solstice-AI Research Team},
    journal={Solstice-AI Technical Report},
    year={2026},
    url={https://solstice-ai.co/papers/entropy-adaptive-speculative-lengths}
  }
tags:
  - "Entropy-Adaptive"
  - "Speculative Decoding"
  - "Dynamic Length"
  - "FLOPs Efficiency"
featured: false
---

## 1. Introduction

Fixed-length speculative decoding generates a fixed number of draft tokens (e.g., 5) regardless of the draft model's confidence. When the draft model encounters an ambiguous position (e.g., the start of a new reasoning step where multiple continuations are valid), it generates low-confidence tokens that are likely to be rejected, wasting verification compute.

EAS monitors the draft model's token-level entropy during generation and halts the draft chain when entropy exceeds a dynamic threshold, preventing wasted verification on tokens that are unlikely to be accepted.

## 2. Entropy-Based Halting

### 2.1 Token Entropy Threshold

For each draft token, EAS computes the entropy of the draft model's predicted distribution:

$$H(t) = -\sum_v p(v_t) \log p(v_t)$$

If $H(t) > \tau_{entropy}(t)$, the draft chain is halted at position $t-1$, and only the first $t-1$ tokens are sent for verification.

### 2.2 Dynamic Threshold

The entropy threshold varies based on the context:

- **High-confidence context** (e.g., JSON syntax): $\tau = 3.0$ nats (allow longer chains).
- **Medium-confidence context** (e.g., natural language): $\tau = 2.0$ nats.
- **Low-confidence context** (e.g., reasoning transitions): $\tau = 1.5$ nats (halt early).

The threshold is determined by the draft model's average entropy in recent tokens, providing automatic context adaptation.

### 2.3 Fork Detection

EAS specifically detects "logical forks"—positions where the reasoning branches into multiple valid paths. At forks, the draft model's entropy spikes because it assigns similar probability to multiple continuations. EAS halts at forks and lets the target model determine the correct path.

## 3. Implementation

### 3.1 Streaming Entropy Computation

EAS computes entropy incrementally during draft generation, adding <0.01ms overhead per token. The entropy is computed using the draft model's logits (which are already computed during forward pass), requiring no additional model forward passes.

### 3.2 Adaptive Chain Length

The actual draft chain length varies from 1 to $k_{max}$ tokens based on entropy:

| Entropy Range | Chain Length | Fraction of Tokens |
|--------------|-------------|-------------------|
| < 1.0 | $k_{max}$ (8) | 15% |
| 1.0 - 2.0 | 5-7 | 42% |
| 2.0 - 3.0 | 3-4 | 31% |
| > 3.0 | 1-2 | 12% |

The average chain length is 4.8 tokens, compared to 8 tokens for fixed-length speculation.

### 3.3 Verification Optimization

EAS reduces verification cost by skipping verification for tokens that were generated with high entropy. These tokens are highly likely to be rejected, so verifying them wastes compute. Instead, EAS uses a "verify-on-reject" strategy: if a high-entropy token is rejected, all subsequent tokens in the chain are automatically rejected without verification.

## 4. Experiments

### 4.1 Setup

We evaluate EAS on LLaMA-7B and Qwen-7B with a 1.5B draft model, comparing against fixed-length speculation with $k \in \{4, 8, 12\}$.

### 4.2 Results

**Speedup and Efficiency:**

| Method | Speedup | Wasted FLOPs | Acceptance Rate |
|--------|---------|-------------|----------------|
| Fixed k=4 | 2.1x | 18% | 86.3% |
| Fixed k=8 | 2.4x | 34% | 78.1% |
| Fixed k=12 | 2.5x | 47% | 71.2% |
| EAS (adaptive) | 2.6x | 21% | 84.7% |

EAS achieves 2.6x speedup with only 21% wasted FLOPs, compared to 47% waste for fixed k=12.

**Per-Domain Performance:**

| Domain | Fixed k=8 Speedup | EAS Speedup | EAS Advantage |
|--------|-------------------|-------------|---------------|
| Code | 2.6x | 3.1x | +19% |
| Math | 2.2x | 2.5x | +14% |
| Creative | 2.0x | 2.2x | +10% |
| JSON | 2.8x | 3.4x | +21% |

EAS shows the largest advantage on JSON (regular syntax) and code (predictable patterns), where it can generate longer chains with low entropy.

## 5. Analysis

### 5.1 Entropy Distribution of Accepted vs. Rejected Tokens

Accepted tokens have average entropy 1.2 nats, while rejected tokens have average entropy 2.8 nats. This 2.3x difference confirms that entropy is a strong predictor of acceptance probability.

### 5.2 Halting Position Analysis

EAS halts at position 3.2 on average (out of maximum 8), with the most common halt position being 4 (38% of chains). The distribution shifts right for code (average halt at 4.1) and left for creative writing (average halt at 2.8).

### 5.3 Interaction with Tree Verification

Combining EAS with tree verification (Paper 22) provides compound benefits: EAS reduces the number of low-confidence tokens in the tree, improving tree verification efficiency by 15%.

## 6. Limitations

EAS requires computing entropy at each draft step, which adds 2% overhead to the draft phase. For very short draft chains (k=2), this overhead is proportionally larger.

Additionally, EAS's dynamic threshold calibration requires a warm-up period of 100 tokens to establish baseline entropy statistics. During warm-up, EAS falls back to fixed-length speculation.

## 7. Conclusion

Fixed-length speculative decoding wastes verification compute on low-confidence tokens. Entropy-Adaptive Speculation halts draft generation when entropy exceeds a dynamic threshold, saving 38% of wasted FLOPs while maintaining 95.1% of fixed-length throughput.

The key insight is that **draft model entropy is a reliable predictor of acceptance probability**, and halting at high-entropy positions avoids wasting verification compute on tokens that are likely to be rejected.

## References

1. Sequoia: Scalable and Robust Speculative Decoding. NeurIPS 2024.
2. DySpec: Faster Speculative Decoding with Dynamic Token Trees. PKU, 2025.
3. Dynamic Delayed Tree Expansion. arXiv 2602.16994, February 2026.
4. NextN Tree Verification. Solstice-AI, 2026.
5. An Introduction to Speculative Decoding. NVIDIA Developer Blog, September 2025.
6. Speculative Decoding for Multimodal Models: A Survey. Preprints, 2026.
7. Bridging Draft Policy Misalignment. ICLR 2026.
8. Variational Speculative Decoding. SMU, 2026.
9. Traversal Verification for Speculative Tree Decoding. OpenReview, 2025.
10. MTP Augmented Speculative Decoding. Solstice-AI, 2026.
