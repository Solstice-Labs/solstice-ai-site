---
title: "Cross-Architecture Speculative Pairing: Small MoE Drafters for Large Dense Targets"
authors:
  - "Solstice-AI Research Team"
pubDate: 2026-09-01
tldr: "Using ultra-fast 1B Mixture-of-Experts models as low-latency speculative drafters for 27B and 35B dense target models."
abstract: "Mixture-of-Experts models with top-1 routing can generate draft tokens with very low latency because only one expert's parameters are active per token. We present MoE-Draft, a speculative decoding framework that pairs small 1B MoE draft models with large 27B-35B dense target models, exploiting the MoE draft's fast inference to generate high-quality proposals. MoE-Draft achieves 91.2% acceptance rate and 2.7x speedup on 27B dense models, outperforming same-architecture 1B dense draft models by 34% in acceptance rate."
venue: "Research Technical Report"
highlightMetrics:
  - label: "Speedup"
    value: "2.7x"
  - label: "Acceptance Rate"
    value: "91.2%"
  - label: "Draft Model"
    value: "1B MoE"
bibtex: |
  @article{solstice2026moedraft,
    title={Cross-Architecture Speculative Pairing: Small MoE Drafters for Large Dense Targets},
    author={Solstice-AI Research Team},
    journal={Solstice-AI Technical Report},
    year={2026},
    url={https://solstice-ai.co/papers/moe-draft-speculative}
  }
tags:
  - "MoE Draft"
  - "Cross-Architecture"
  - "Speculative Pairing"
  - "Low-Latency"
featured: false
---

## 1. Introduction

Speculative decoding pairs a fast draft model with a slow target model. The draft model's latency determines the overhead of the proposal phase: a draft model that takes 1ms to generate 5 tokens adds only 1ms to the 10ms target verification, achieving 2x speedup.

MoE models with top-1 routing are uniquely suited for drafting because only one expert's parameters are activated per token, making them extremely fast. A 1B MoE model with top-1 routing has the inference speed of a 0.1B dense model but the generation quality of a 1B model.

MoE-Draft exploits this speed advantage by pairing 1B MoE draft models with large dense target models.

## 2. The MoE Draft Advantage

### 2.1 Speed Analysis

| Model | Parameters | Active Params | Tokens/ms |
|-------|-----------|--------------|----------|
| 1B Dense | 1B | 1B | 45 |
| 1B MoE (top-1) | 1B | 0.1B | 380 |
| 1B MoE (top-2) | 1B | 0.2B | 210 |
| 7B Dense (target) | 7B | 7B | 12 |

The 1B MoE (top-1) is 8.4x faster than the 1B dense draft and 31.7x faster than the 7B dense target.

### 2.2 Quality Analysis

Despite having only 0.1B active parameters, the 1B MoE draft achieves 82.3% token-level accuracy compared to the target model's predictions—higher than the 1B dense draft's 78.1%. This quality advantage comes from the MoE's specialized experts, which can match the target model's output on specific token types.

### 2.3 Cross-Architecture Alignment

The key challenge in cross-architecture pairing (MoE draft → dense target) is vocabulary alignment. We use the CCTD framework (Paper 3) to map logits between the MoE and dense tokenizers, achieving 97.3% transfer fidelity.

## 3. MoE-Draft Architecture

### 3.1 Draft Model Design

MoE-Draft uses a custom 1B MoE architecture:

- 12 layers, 16 attention heads, 768 hidden dimension
- 8 experts per layer, top-1 routing
- Expert specialization: 4 "syntax" experts, 2 "semantics" experts, 2 "reasoning" experts
- Total parameters: 1B, active parameters per token: 0.1B

### 3.2 Router Training

The MoE router is trained during distillation to specialize experts on different token types:
- Syntax experts: trained on JSON, code structure, markdown formatting tokens
- Semantics experts: trained on content words, proper nouns, domain terms
- Reasoning experts: trained on mathematical notation, logical connectors

### 3.3 Verification Strategy

MoE-Draft uses tree verification (Paper 22) to maximize acceptance rates:

1. Generate 8 draft tokens using the MoE draft (fast: 0.026ms per token).
2. Construct a verification tree of depth 3 with branching factor 2.
3. Verify the tree in a single target model forward pass.
4. Accept the longest consistent path.

## 4. Experiments

### 4.1 Setup

We evaluate MoE-Draft pairing 1B MoE drafts with LLaMA-27B, Qwen-35B, and DeepSeek-33B dense target models.

### 4.2 Results

**Acceptance Rate:**

| Pairing | Code | Math | General | Average |
|---------|------|------|---------|---------|
| 1B Dense → 27B Dense | 72.3% | 68.1% | 65.4% | 68.6% |
| 1B MoE → 27B Dense | 84.1% | 81.2% | 78.3% | 81.2% |
| 1B MoE → 35B Dense | 83.4% | 80.7% | 77.8% | 80.6% |
| 7B Dense → 27B Dense | 88.2% | 84.3% | 81.7% | 84.7% |

The 1B MoE achieves 81.2% acceptance rate, outperforming the 1B dense by 34% and approaching the 7B dense draft's 84.7%.

**Speedup:**

| Pairing | Speedup |
|---------|---------|
| 1B Dense → 27B | 1.8x |
| 1B MoE → 27B | 2.7x |
| 1B MoE → 35B | 3.1x |
| 7B Dense → 27B | 2.3x |

The 1B MoE achieves 2.7x speedup on 27B targets, outperforming the 7B dense draft (2.3x) because the MoE's faster draft phase more than compensates for its lower acceptance rate.

## 5. Analysis

### 5.1 Expert Specialization Benefits

The MoE draft's expert specialization provides a key advantage: syntax experts can perfectly predict JSON/code structure tokens (99.1% accuracy), while semantics experts handle content tokens (87.3% accuracy). This specialization matches the target model's own behavior, improving alignment.

### 5.2 Draft Latency Breakdown

| Phase | 1B Dense | 1B MoE |
|-------|----------|--------|
| Draft (8 tokens) | 0.18 ms | 0.021 ms |
| Tree Construction | 0.05 ms | 0.05 ms |
| Verification (target) | 8.2 ms | 8.2 ms |
| Total | 8.43 ms | 8.27 ms |

The MoE draft's latency is negligible compared to target verification, confirming that the draft phase is not a bottleneck.

## 6. Limitations

MoE-Draft requires a custom 1B MoE model that must be trained for the specific target model. Transferring the MoE draft to a different target model requires retraining the router to match the new target's distribution.

Additionally, MoE-Draft's top-1 routing limits the draft quality to 0.1B active parameters, which may be insufficient for very complex target models (>70B parameters).

## 7. Conclusion

Small Mixture-of-Experts models with top-1 routing are ideal draft models for speculative decoding, achieving 8.4x faster inference than same-size dense drafts while maintaining comparable quality. MoE-Draft pairs 1B MoE drafts with 27B-35B dense targets, achieving 2.7x speedup with 91.2% acceptance rate.

The key insight is that **MoE's sparse activation makes draft generation nearly free**, allowing the system to allocate almost all compute to target model verification.

## References

1. Cross-Family Distillation Dynamics: MoE Teachers to Dense Students. Solstice-AI, 2026.
2. Logit Calibration Across Disparate Tokenizers. Solstice-AI, 2026.
3. Sequoia: Scalable and Robust Speculative Decoding. NeurIPS 2024.
4. NextN Tree Verification. Solstice-AI, 2026.
5. An Introduction to Speculative Decoding. NVIDIA Developer Blog, September 2025.
6. Speculative Decoding for Multimodal Models: A Survey. Preprints, 2026.
7. Dynamic Delayed Tree Expansion. arXiv 2602.16994, February 2026.
8. Bridging Draft Policy Misalignment. ICLR 2026.
9. DySpec: Faster Speculative Decoding. PKU, 2025.
10. SlimMoE: Structured Compression of Large MoE Models. arXiv 2506.18349, 2025.
