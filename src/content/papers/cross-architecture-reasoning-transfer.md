---
title: "Empirical Limits of Cross-Architecture Reasoning Transfer in Sub-8B Student LLMs"
authors:
  - "Solstice-AI Research Team"
pubDate: 2026-08-30
tldr: "A comprehensive investigation into attention map divergence, reasoning token entropy, and layer-to-layer distillation across disparate transformer architectures."
abstract: "Distilling long chain-of-thought reasoning from frontier teacher architectures into dense sub-8B students poses unique representational challenges. When student architectures differ fundamentally in layer depth, attention head count, or rotary embedding configurations, direct logit matching degrades generalization. In this report, we present an empirical study of cross-architecture reasoning transfer across 50,000 algorithmic benchmarks, demonstrating that intermediate scratchpad supervision with consensus rejection sampling outperforms traditional token-level KL divergence minimization."
venue: "Research Technical Report"
huggingfaceUrl: "https://huggingface.co/Solstice-AI"
highlightMetrics:
  - label: "Math-500 Accuracy"
    value: "86.4%"
  - label: "Distillation Efficiency"
    value: "+14.8% vs KL Div"
  - label: "Evaluation Benchmarks"
    value: "50k Tasks"
bibtex: |
  @article{solstice2026crossarchitecture,
    title={Empirical Limits of Cross-Architecture Reasoning Transfer in Sub-8B Student LLMs},
    author={Solstice-AI Research Team},
    journal={Solstice-AI Technical Report},
    year={2026},
    url={https://solstice-ai.co/papers/cross-architecture-reasoning-transfer}
  }
tags:
  - "Distillation"
  - "Attention Alignment"
  - "Reasoning"
  - "Benchmarks"
featured: false
---

## 1. Abstract & Motivation

When distilling dense reasoning models, a fundamental question arises: *Can a compact 3.8B–7B parameter model genuinely internalize the non-linear deduction capabilities of a 100B+ MoE teacher, or is it merely mimicking surface-level phrasing?*

We evaluate three distinct distillation methodologies:
1. **Token-Level KL Divergence (Standard KD):** Minimizing logit divergence over intermediate tokens.
2. **Hidden State Alignment:** Forcing intermediate student representations to match projected teacher layer outputs.
3. **Step-Level Consensus SFT:** Supervising the student on multi-teacher verified chain-of-thought scratchpads with automated verification filters.

---

## 2. Key Findings

* **Outlier Attention Saturation:** In single-teacher KD, the student model overfits to high-magnitude attention spikes unique to the teacher's RoPE frequency base.
* **Multi-Teacher Regularization:** Aggregating thought chains from diverse models (dense, MoE, and long-context architectures) acts as a natural regularizer, preventing student attention collapse on edge cases.
* **Inference Latency Decoupling:** Step-level distilled students generate compact, efficient reasoning paths that are 22% shorter than raw teacher outputs without sacrificing proof accuracy.
