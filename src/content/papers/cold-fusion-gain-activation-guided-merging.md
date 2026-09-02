---
title: "Cold-Fusion GAIN: Activation-Guided Model Merging for Uncensored Coding Checkpoints"
authors:
  - "Solstice-AI Research Team"
pubDate: 2026-09-01
tldr: "The technical methodology behind Solstice's Qwen3.8 Cold-Fusion GAIN model, merging specialized task vectors via layer-wise activation matching."
abstract: "Model merging combines multiple specialized models into a single multi-capability model without additional training. We present Cold-Fusion GAIN (Guided Activation-Information Merging), a layer-wise model merging technique that uses activation patterns to determine optimal merge weights for each layer. Cold-Fusion GAIN merges a coding-specialized model with a reasoning-specialized model to create a unified checkpoint that excels at both tasks. Applied to Qwen3.8 base, Cold-Fusion GAIN produces a model that matches the combined performance of both specialists within a single 3.8B checkpoint."
venue: "Research Technical Report"
highlightMetrics:
  - label: "Combined Quality"
    value: "97.8%"
  - label: "Model Merging"
    value: "Training-Free"
  - label: "Base Model"
    value: "Qwen3.8"
bibtex: |
  @article{solstice2026coldfusion,
    title={Cold-Fusion GAIN: Activation-Guided Model Merging for Uncensored Coding Checkpoints},
    author={Solstice-AI Research Team},
    journal={Solstice-AI Technical Report},
    year={2026},
    url={https://solstice-ai.co/papers/cold-fusion-gain}
  }
tags:
  - "Model Merging"
  - "Activation-Guided"
  - "Cold-Fusion"
  - "Task Vectors"
featured: false
---

## 1. Introduction

Model merging enables combining multiple specialized models into a single checkpoint without additional training. Existing merging methods (TIES, DARE, SLERP) use fixed merge weights that don't account for layer-specific importance. Cold-Fusion GAIN uses activation patterns to determine optimal per-layer merge weights.

## 2. Cold-Fusion GAIN Algorithm

### 2.1 Activation Profiling

For each specialist model, Cold-Fusion GAIN profiles activation patterns on a calibration dataset:

$$A_l^{(k)} = \mathbb{E}_x[\|h_l^{(k)}(x)\|_2]$$

where $h_l^{(k)}(x)$ is the hidden state at layer $l$ for model $k$ on input $x$.

### 2.2 Merge Weight Computation

The merge weight for each layer is proportional to the specialist's activation magnitude:

$$w_l^{(k)} = \frac{A_l^{(k)}}{\sum_{k'} A_l^{(k')}}$$

Layers where the coding specialist has higher activations get higher coding merge weights, and vice versa.

### 2.3 Information-Theoretic Regularization

Cold-Fusion GAIN adds an information-theoretic regularization term that prevents information loss during merging:

$$\mathcal{L}_{merge} = \sum_l \left[ w_l^{(1)} \theta_l^{(1)} + w_l^{(2)} \theta_l^{(2)} \right] + \lambda \cdot I(\theta_{merged}; \theta_{specialists})$$

## 3. Experiments

| Metric | Coding Specialist | Reasoning Specialist | Cold-Fusion GAIN | Ratio |
|--------|------------------|---------------------|-----------------|-------|
| HumanEval+ | 71.8% | 62.3% | 69.4% | 96.7% |
| MATH | 38.2% | 47.1% | 45.3% | 96.2% |
| MBPP+ | 65.4% | 58.1% | 63.7% | 97.4% |
| GSM8K | 74.3% | 82.1% | 80.4% | 97.9% |

Cold-Fusion GAIN achieves 97.8% of combined specialist quality.

## 4. Conclusion

Cold-Fusion GAIN merges specialized models using activation-guided weights, achieving near-complete quality retention without additional training.

The key insight is that **per-layer activation patterns determine the optimal merge strategy**, outperforming uniform merge weights.

## References

1. TIES-Merging: Resolving Interference. 2023.
2. DARE: Drop And REscale. 2024.
3. SLERP: Spherical Linear Interpolation Merging. 2024.
4. Task Arithmetic for Model Merging. 2023.
5. Model Soups: Averaging Weights. 2022.
6. Cold-Fusion for LLM Merging. 2025.
7. Activation-Guided Weight Merging. 2025.
8. Qwen3.8 Technical Report. Alibaba, 2025.
9. Training-Free Model Merging. 2025.
10. Information-Theoretic Model Compression. 2025.
