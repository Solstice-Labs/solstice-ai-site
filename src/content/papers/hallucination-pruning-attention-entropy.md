---
title: "Hallucination Pruning: Attention-Entropy Signatures of Ungrounded Token Generation"
authors:
  - "Solstice-AI Research Team"
pubDate: 2026-09-01
tldr: "Identifying the exact attention-layer entropy spikes that precede hallucinated tool calls and synthetic citations in frontier model outputs."
abstract: "LLM hallucinations—fabricated citations, non-existent tool calls, and invented facts—exhibit characteristic attention-layer entropy signatures that precede their generation. We present HalluPrune, a detection framework that identifies hallucination-prone tokens by monitoring attention entropy spikes across transformer layers, pruning suspected hallucinations before they enter distillation datasets. HalluPrune achieves 91.3% hallucination detection accuracy with 4.2% false positive rate, removing 11.8% of hallucinated content from synthetic datasets and improving downstream student factual accuracy by 7.4%."
venue: "Research Technical Report"
highlightMetrics:
  - label: "Detection Accuracy"
    value: "91.3%"
  - label: "False Positive Rate"
    value: "4.2%"
  - label: "Factual Accuracy Gain"
    value: "+7.4%"
bibtex: |
  @article{solstice2026halluprune,
    title={Hallucination Pruning: Attention-Entropy Signatures of Ungrounded Token Generation},
    author={Solstice-AI Research Team},
    journal={Solstice-AI Technical Report},
    year={2026},
    url={https://solstice-ai.co/papers/hallucination-pruning-attention-entropy}
  }
tags:
  - "Hallucination Detection"
  - "Attention Entropy"
  - "Token Pruning"
  - "Factual Accuracy"
featured: false
---

## 1. Introduction

LLM hallucinations are generated when the model produces tokens that are not grounded in its training data or the input context. Research on hallucination detection (Farquhar et al., Nature 2024; NeurIPS 2025 HaMI) has shown that entropy-based uncertainty estimators can identify hallucinated content. HalluPrune extends this to multi-teacher distillation, using attention-layer entropy signatures to detect and prune hallucinations before they enter the training dataset.

## 2. Attention-Entropy Hallucination Signatures

### 2.1 Pre-Hallucination Entropy Spikes

We discover that hallucinated tokens are preceded by characteristic entropy spikes in specific attention layers:

- **Layers 12-16** (mid-network): Show 2.3x higher entropy variance before hallucinated tokens compared to grounded tokens.
- **Layers 24-28** (late-network): Show 1.8x higher entropy before hallucinated tokens.
- **Overall pattern:** A "entropy cascade" where early layers show uncertainty that propagates to later layers before the hallucinated token is generated.

### 2.2 Hallucination Type Signatures

Different hallucination types have distinct entropy profiles:

| Hallucination Type | Signature | Detection Method |
|-------------------|-----------|-----------------|
| Fabricated Citations | Spike in layer 16, sustained through layer 28 | Layer-specific threshold |
| Non-existent Tool Calls | Sharp spike in layers 12-14 | Transient spike detection |
| Invented Facts | Gradual entropy increase over 5-10 tokens | Trend analysis |
| Circular Reasoning | Periodic entropy oscillation | Frequency analysis |

## 3. HalluPrune Framework

### 3.1 Multi-Layer Monitoring

HalluPrune monitors attention entropy across all transformer layers, maintaining a running statistics model for each layer:

$$\mu_l(t) = \alpha \cdot \mu_l(t-1) + (1-\alpha) \cdot H_l(t)$$
$$\sigma_l^2(t) = \alpha \cdot \sigma_l^2(t-1) + (1-\alpha) \cdot (H_l(t) - \mu_l(t))^2$$

### 3.2 Spike Detection

A token is flagged as potential hallucination if:
1. **Layer-specific spike:** $H_l(t) > \mu_l(t) + k \cdot \sigma_l(t)$ for the critical layers.
2. **Cascade detection:** Multiple consecutive layers show spikes.
3. **Sustained elevation:** The spike persists for >3 tokens.

### 3.3 False Positive Mitigation

To reduce false positives:
- **Context-aware thresholds:** Adjust thresholds based on the input type (factual question vs. creative writing).
- **Teacher ensemble voting:** Require hallucination detection from multiple teachers.
- **Recovery verification:** Re-generate suspected hallucinations and check for consistency.

## 4. Experiments

### 4.1 Setup

We monitor 500,000 reasoning traces from 7 teachers, using manual annotation of 10,000 traces for ground truth.

### 4.2 Results

| Metric | Value |
|--------|-------|
| Detection Accuracy | 91.3% |
| False Positive Rate | 4.2% |
| Hallucinations Removed | 11.8% |
| Factual Accuracy Gain | +7.4% |
| Citation Accuracy Gain | +12.1% |

## 5. Conclusion

Attention-layer entropy signatures reliably predict hallucinated token generation, enabling proactive pruning of hallucinations from distillation datasets.

The key insight is that **hallucinations are preceded by detectable uncertainty signatures** in specific attention layers, enabling proactive detection rather than post-hoc correction.

## References

1. Detecting Hallucinations in LLMs Using Semantic Entropy. Nature, 2024.
2. Robust Hallucination Detection via Adaptive Token Selection. NeurIPS 2025.
3. HalLoc: Token-level Localization of Hallucinations. CVPR 2025.
4. Entropy-Weighted Consensus Filtering. Solstice-AI, 2026.
5. Token-Level Hallucination Detection. Emergent Mind, 2025.
6. Hallucination Detection in Black-Box LLMs. arXiv 2509.04492, 2025.
7. Attention-Entropy Analysis for LLMs. 2025.
8. Hallucination Pruning for Training Data. 2025.
9. From Illusion to Insight: Hallucination Mitigation Survey. MDPI, 2025.
10. Semantic Entropy for Hallucination Detection. 2025.
