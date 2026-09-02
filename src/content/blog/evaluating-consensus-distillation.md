---
title: "Why Multi-Teacher Consensus Eliminates Mode Collapse in Reasoning SFT"
description: "How training on multi-teacher verified consensus scratchpads prevents student models from memorizing single-model hallucination patterns."
pubDate: 2026-08-31
author: "Solstice-AI Core Team"
tags:
  - "Research"
  - "Distillation"
  - "Reasoning"
readingTime: "5 min read"
takeaways:
  - "Single-teacher distillation creates stylistic overfitting and catastrophic blind spots."
  - "Multi-teacher consensus forces the student model to learn invariant underlying logical structures."
  - "Solace 1.0 achieves 91.7% frontier reasoning recovery across 8 architectures."
featured: false
---

When fine-tuning student language models on synthetic reasoning data, engineers frequently observe a frustrating phenomenon: the student model reproduces the teacher's exact phrasing habits, transition phrases ("Wait, let me double check that..."), and distinctive reasoning errors.

We call this **Single-Teacher Representational Entanglement**.

### The Multi-Teacher Solution

Instead of treating one proprietary model as the absolute oracle, Solstice-AI constructs a consensus graph across 8 frontier architectures.

```
Problem Input
     │
     ├──► GLM 5.2 ────────┐
     ├──► DeepSeek V4 ────┼──► Graph Alignment & Verification ──► High-Density Student SFT
     ├──► GPT-5.6 Sol ────┤
     └──► Qwen 3.8 Max ───┘
```

When 8 disparate models converge on the identical mathematical deduction using different algebraic representations, the distilled student model learns the underlying invariant truth rather than superficial tokens.
