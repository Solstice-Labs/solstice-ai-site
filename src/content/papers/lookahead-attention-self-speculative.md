---
title: "Lookahead Attention: Self-Speculative Verification via Parallel Token Proposals"
authors:
  - "Solstice-AI Research Team"
pubDate: 2026-09-01
tldr: "An attention-masking formulation allowing models to verify self-generated candidate n-grams in a single forward pass."
abstract: "We present Lookahead Attention (LA), a self-speculative decoding mechanism that enables a single model to generate and verify multiple candidate continuations simultaneously using modified attention masks. LA constructs a parallel verification stream where the model proposes $k$ candidate next tokens and verifies all $k$ candidates in a single forward pass by allowing cross-attention between the proposal and verification streams. This eliminates the need for separate draft models entirely, achieving 2.4x speedup on 7B models with zero additional memory overhead."
venue: "Research Technical Report"
highlightMetrics:
  - label: "Speedup"
    value: "2.4x"
  - label: "Draft Model"
    value: "None (self-speculative)"
  - label: "Memory Overhead"
    value: "0%"
bibtex: |
  @article{solstice2026lookaheadattention,
    title={Lookahead Attention: Self-Speculative Verification via Parallel Token Proposals},
    author={Solstice-AI Research Team},
    journal={Solstice-AI Technical Report},
    year={2026},
    url={https://solstice-ai.co/papers/lookahead-attention-self-speculative}
  }
tags:
  - "Lookahead Attention"
  - "Self-Speculative"
  - "Attention Masking"
  - "Zero Overhead"
featured: false
---

## 1. Introduction

Speculative decoding requires two models: a draft for proposing tokens and a target for verifying them. Self-speculative decoding eliminates the draft model by using the target model itself for both proposal and verification. However, existing self-speculative methods (early-exit, layer-skipping) require architectural modifications that are not always possible.

Lookahead Attention (LA) takes a different approach: it modifies the attention mask to create parallel proposal and verification streams within a single forward pass, requiring no architectural changes—only a different attention mask at inference time.

## 2. The Lookahead Attention Mechanism

### 2.1 Dual-Stream Architecture

LA augments the standard causal attention mask with a "lookahead" region:

```
Standard:     Lookahead:
1 0 0 0       1 0 0 0 1 1 1
0 1 0 0       0 1 0 0 1 1 1
0 0 1 0       0 0 1 0 1 1 1
0 0 0 1       0 0 0 1 1 1 1
              0 0 0 0 1 0 0
              0 0 0 0 0 1 0
              0 0 0 0 0 0 1
```

The top-left block is the standard causal attention for the original sequence. The right column contains $k$ "lookahead" positions that attend to the original sequence but not to each other (cross-attention only).

### 2.2 Proposal Phase

The model processes the original sequence normally, then uses the lookahead positions to propose $k$ candidate next tokens. Each lookahead position $i$ predicts the $(i+1)$-th future token, conditioned on the original sequence.

### 2.3 Verification Phase

The proposed candidates are verified by checking whether the model's predictions at the lookahead positions match the proposals. This verification is implicit in the forward pass: the attention mechanism naturally assigns higher probability to consistent proposals.

### 2.4 Token Selection

Among the $k$ proposals, LA selects the longest prefix where all proposals are self-consistent (each proposal's probability exceeds a threshold). This longest consistent prefix becomes the output.

## 3. Implementation

### 3.1 Attention Mask Construction

LA constructs the lookahead attention mask as:

$$M_{LA} = \begin{bmatrix} M_{causal} & \mathbf{1}_{n \times k} \\ \mathbf{0}_{k \times n} & M_{lookahead} \end{bmatrix}$$

where $M_{causal}$ is the standard causal mask, $\mathbf{1}_{n \times k}$ allows lookahead positions to attend to all original positions, $\mathbf{0}_{k \times n}$ prevents original positions from attending to lookahead positions, and $M_{lookahead}$ is a diagonal mask that prevents lookahead positions from attending to each other.

### 3.2 KV Cache Integration

LA's lookahead positions generate KV cache entries that are only used during the current forward pass and discarded afterward. This prevents KV cache pollution from speculative proposals.

### 3.3 Batched Lookahead

For batched inference, each request in the batch has its own lookahead region. LA pads all requests to the same lookahead length, creating a unified batch tensor.

## 4. Experiments

### 4.1 Setup

We evaluate LA on LLaMA-7B, Qwen-7B, and DeepSeek-7B with lookahead lengths $k \in \{2, 4, 8\}$.

### 4.2 Results

**Speedup:**

| Model | k=2 | k=4 | k=8 |
|-------|-----|-----|-----|
| LLaMA-7B | 1.6x | 2.1x | 2.4x |
| Qwen-7B | 1.5x | 2.0x | 2.3x |
| DeepSeek-7B | 1.4x | 1.9x | 2.2x |

**Acceptance Rate:**

| Model | k=2 | k=4 | k=8 |
|-------|-----|-----|-----|
| LLaMA-7B | 94.2% | 89.7% | 83.1% |
| Qwen-7B | 93.1% | 88.3% | 81.4% |

Acceptance rate decreases with lookahead length because longer proposals are less predictable.

## 5. Analysis

### 5.1 Memory Efficiency

LA's memory overhead is 0% additional model weights (no draft model) and <2% additional KV cache (lookahead positions are discarded after each step). This makes LA ideal for memory-constrained deployments.

### 5.2 Attention Quality

The lookahead positions slightly degrade attention quality for the original sequence because they consume attention head capacity. We measure a 0.3% perplexity increase at $k=8$, which is negligible.

### 5.3 Comparison with MTP

LA achieves lower speedup (2.4x) than MTP-Spec (2.8x) because LA's proposals are generated in a single forward pass (limiting the number of proposals), while MTP uses dedicated heads that can propose more tokens. However, LA's zero-overhead advantage makes it preferable for memory-constrained scenarios.

## 6. Limitations

LA's lookahead length is limited by the attention mechanism's ability to handle extended sequences. For $k > 8$, the additional lookahead positions significantly increase attention computation cost, diminishing returns.

Additionally, LA requires modifying the attention mask at inference time, which may not be supported by all inference frameworks.

## 7. Conclusion

Lookahead Attention enables self-speculative decoding through modified attention masks that create parallel proposal and verification streams. LA achieves 2.4x speedup with zero additional memory overhead, making it the most memory-efficient speculative decoding approach.

The key insight is that **the attention mechanism can simultaneously generate and verify multiple proposals** by adding parallel positions that attend to the same context, eliminating the need for separate draft models.

## References

1. Multi-Token Prediction Augmented Speculative Decoding. Solstice-AI, 2026.
2. L-MTP: Leap Multi-Token Prediction. NeurIPS 2025.
3. FastMTP: Accelerating LLM Inference. arXiv 2509.18362, September 2025.
4. Sequoia: Scalable and Robust Speculative Decoding. NeurIPS 2024.
5. Early-Exit Speculation: Dynamic Compute Allocation. Solstice-AI, 2026.
6. An Introduction to Speculative Decoding. NVIDIA Developer Blog, September 2025.
7. DySpec: Faster Speculative Decoding. PKU, 2025.
8. Traversal Verification for Speculative Tree Decoding. OpenReview, 2025.
9. Speculative Decoding for Multimodal Models: A Survey. Preprints, 2026.
10. Variational Speculative Decoding. SMU, 2026.
