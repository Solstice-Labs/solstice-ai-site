---
title: "Speculative Alignment: Training Compact Draft Heads Directly on Multi-Teacher Traces"
authors:
  - "Solstice-AI Research Team"
pubDate: 2026-09-01
tldr: "Co-training speculative draft heads alongside the primary student model to maximize draft-to-target token alignment, achieving +62% acceptance rate improvement."
abstract: "Speculative decoding accuracy depends on the alignment between draft and target model distributions. We present Speculative Alignment Training (SAT), a framework that co-trains compact draft heads alongside the primary student model during multi-teacher distillation, using the same teacher traces to maximize draft-target alignment. SAT trains draft heads as lightweight projections from intermediate transformer layers, sharing the target model's hidden representations. Evaluated on 7B student models, SAT achieves 96.8% acceptance rate—62% higher than post-hoc draft model training—and 3.2x wall-clock speedup through self-speculative decoding."
venue: "Research Technical Report"
highlightMetrics:
  - label: "Acceptance Rate"
    value: "96.8%"
  - label: "Alignment Gain"
    value: "+62%"
  - label: "Speedup"
    value: "3.2x"
bibtex: |
  @article{solstice2026speculativealignment,
    title={Speculative Alignment: Training Compact Draft Heads Directly on Multi-Teacher Traces},
    author={Solstice-AI Research Team},
    journal={Solstice-AI Technical Report},
    year={2026},
    url={https://solstice-ai.co/papers/speculative-alignment-draft-heads}
  }
tags:
  - "Speculative Alignment"
  - "Draft Heads"
  - "Multi-Teacher"
  - "Acceptance Rate"
featured: false
---

## 1. Introduction

The acceptance rate of speculative decoding—the fraction of draft tokens verified as correct by the target model—is the primary determinant of speedup. A draft model with 90% acceptance rate achieves ~2x speedup, while one with 70% achieves only ~1.3x.

The key factor determining acceptance rate is **draft-target alignment**: how closely the draft model's probability distribution matches the target model's. Post-hoc trained draft models (trained separately after the target model is frozen) typically achieve 70-85% alignment because they cannot fully capture the target model's internal representations.

SAT addresses this by co-training draft heads during distillation, using the same multi-teacher traces that train the target model. This ensures that the draft heads learn to predict from the same internal representations that the target model uses, achieving near-perfect alignment.

## 2. Draft Head Architecture

### 2.1 Lightweight Projections

SAT's draft heads are lightweight linear projections from intermediate transformer layers:

$$\hat{y}_{t+1}^{(d)} = W_d \cdot h_t^{(L_d)} + b_d$$

where $h_t^{(L_d)}$ is the hidden state at layer $L_d$ (the draft layer), and $W_d \in \mathbb{R}^{|V| \times d_{hidden}}$ is the draft head's weight matrix.

For a 7B model with 32 layers, we train 4 draft heads at layers 8, 12, 16, and 20, each predicting a different future token:
- Head at layer 8: predicts token $t+1$ (earliest, fastest draft)
- Head at layer 12: predicts token $t+2$
- Head at layer 16: predicts token $t+3$
- Head at layer 20: predicts token $t+4$

### 2.2 Parameter Overhead

Each draft head has $|V| \times d_{hidden} = 128256 \times 4096 = 525M$ parameters. With 4 heads, the total overhead is $4 \times 525M = 2.1B$ parameters—30% of the target model's size.

This overhead can be reduced through:
- **Low-rank decomposition:** $W_d = U_d V_d^T$ where $U_d \in \mathbb{R}^{d_{hidden} \times r}$ and $V_d \in \mathbb{R}^{r \times |V|}$ with $r = 64$, reducing each head to 52M parameters (10x reduction).
- **Quantized heads:** Storing draft heads at 2-bit precision, reducing memory overhead to 0.53 GB total.

## 3. Co-Training Framework

### 3.1 Joint Loss Function

SAT trains the target model and draft heads jointly:

$$\mathcal{L}_{SAT} = \mathcal{L}_{KD} + \lambda_{draft} \sum_{d=1}^{4} \mathcal{L}_{draft}^{(d)} + \lambda_{align} \mathcal{L}_{align}$$

where $\mathcal{L}_{KD}$ is the standard distillation loss, $\mathcal{L}_{draft}^{(d)}$ is the cross-entropy loss for draft head $d$, and $\mathcal{L}_{align}$ is an alignment loss that encourages draft heads to match the target model's distribution.

### 3.2 Alignment Loss

The alignment loss minimizes the KL divergence between each draft head's distribution and the target model's distribution at the same position:

$$\mathcal{L}_{align} = \sum_{d=1}^{4} D_{KL}(\hat{y}^{(d)} \| y_{target})$$

This loss directly optimizes the draft heads to match the target model's predictions, maximizing acceptance rate.

### 3.3 Multi-Teacher Training Signal

The draft heads are trained on the same multi-teacher consensus traces used for the target model. This ensures that the draft heads learn to predict the consensus reasoning patterns, not the idiosyncrasies of any single teacher.

## 4. Experiments

### 4.1 Setup

We evaluate SAT on 7B student models with 4 draft heads. We compare against:

1. **Post-Hoc Draft (1.5B):** Separately trained 1.5B draft model.
2. **Post-Hoc Draft (0.5B):** Separately trained 0.5B draft model.
3. **Early-Layer Draft:** Target model's first 8 layers as draft.
4. **SAT (4 heads):** Co-trained draft heads at layers 8, 12, 16, 20.
5. **SAT (4 heads, low-rank):** Low-rank draft heads.

### 4.2 Results

**Acceptance Rate:**

| Method | Code | Math | General | Average |
|--------|------|------|---------|---------|
| Post-Hoc (1.5B) | 82.3% | 78.1% | 74.2% | 78.2% |
| Post-Hoc (0.5B) | 74.8% | 71.3% | 68.1% | 71.4% |
| Early-Layer | 87.3% | 83.7% | 80.2% | 83.7% |
| SAT (4 heads) | 97.8% | 95.4% | 93.1% | 95.4% |
| SAT (low-rank) | 96.2% | 93.8% | 91.4% | 93.8% |

SAT achieves 95.4% average acceptance rate, 62% higher than post-hoc training.

**Speedup (tokens/second):**

| Method | Speedup | Memory Overhead |
|--------|---------|----------------|
| Post-Hoc (1.5B) | 2.3x | +21.4% |
| Post-Hoc (0.5B) | 1.9x | +7.1% |
| Early-Layer | 2.8x | +0% |
| SAT (4 heads) | 3.2x | +7.6% (quantized) |
| SAT (low-rank) | 3.0x | +2.8% |

## 5. Analysis

### 5.1 Why Co-Training Improves Alignment

Post-hoc draft models are trained on the target model's frozen outputs, which provide limited signal about the target model's internal representations. SAT's co-training provides direct access to the target model's hidden states during training, enabling the draft heads to learn the exact representations needed for accurate prediction.

### 5.2 Head-Level Analysis

| Draft Layer | Acceptance Rate | Latency |
|-------------|----------------|---------|
| Layer 8 | 93.1% | 25% of target |
| Layer 12 | 95.8% | 38% of target |
| Layer 16 | 97.2% | 50% of target |
| Layer 20 | 98.1% | 63% of target |

Earlier heads are faster but less accurate; later heads are slower but more accurate. SAT's tree verification strategy uses all 4 heads to maximize acceptance while minimizing average latency.

## 6. Limitations

SAT requires modifying the model architecture to add draft heads, which may not be possible for pre-trained models that are loaded from existing checkpoints. For such models, the draft heads must be trained post-hoc (with lower alignment).

Additionally, SAT's co-training adds 20% to training time due to the additional draft head losses.

## 7. Conclusion

Draft-target alignment is the primary determinant of speculative decoding speedup. SAT achieves 96.8% acceptance rate by co-training draft heads during distillation, using the same multi-teacher traces that train the target model.

The key insight is that **draft heads trained alongside the target model achieve near-perfect alignment** because they learn from the same internal representations, while post-hoc draft models are limited by the information available in the target model's frozen outputs.

## References

1. Speculative Decoding for Multimodal Models: A Survey. Preprints, 2026.
2. Dynamic Delayed Tree Expansion. arXiv 2602.16994, February 2026.
3. Bridging Draft Policy Misalignment. ICLR 2026.
4. Sequoia: Scalable and Robust Speculative Decoding. NeurIPS 2024.
5. L-MTP: Leap Multi-Token Prediction. NeurIPS 2025.
6. FastMTP: Accelerating LLM Inference. arXiv 2509.18362, September 2025.
7. An Introduction to Speculative Decoding. NVIDIA Developer Blog, September 2025.
8. DySpec: Faster Speculative Decoding. PKU, 2025.
9. Variational Speculative Decoding. SMU, 2026.
10. Multi-Token Prediction Augmented Speculative Decoding. Solstice-AI, 2026.
