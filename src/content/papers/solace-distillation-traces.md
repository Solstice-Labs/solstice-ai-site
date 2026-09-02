---
title: "Project Solace: Distilling Multi-Teacher Reasoning Traces from 7 Frontier Model Families"
authors:
  - "Solstice-AI Research Team"
pubDate: 2026-08-19
tldr: "A standardized methodology and open dataset for distilling cross-architecture reasoning, code generation, and agentic traces into efficient sub-8B parameter models."
abstract: "Frontier large language models exhibit remarkable reasoning and multi-step problem-solving capabilities, but their proprietary weights and massive parameter counts hinder open scientific inquiry and cost-effective local deployment. In this report, we introduce Project Solace, an open-access distillation repository comprising 12,586,893 unique conversations (137.7 GB uncompressed / 34.3 GB compressed) curated from 60 audited datasets across seven diverse frontier architectures (GLM-5.2, Claude Fable 5 & Mythos 5, GPT-5.6 Sol & GPT-5.5 Codex, DeepSeek V4 Pro 0813, Qwen 3.8-Max, Kimi K3, and Manus Agents). We describe our multi-stage quality verification pipeline, exact SHA256 deduplication (purging 3.44M redundant instances), and demonstrate that fine-tuning compact student base models on Solace recovers frontier reasoning performance across MMLU-Pro, Math-500, and ARC-AGI3."
venue: "Technical Specification & Dataset Report"
huggingfaceUrl: "https://huggingface.co/datasets/Solstice-AI/Solace-1.0-Omni"
highlightMetrics:
  - label: "Unique Conversations"
    value: "12.59M"
  - label: "Corpus Volume"
    value: "137.7 GB"
  - label: "Audited Sources"
    value: "60 Datasets"
  - label: "License"
    value: "AGPL-3.0"
bibtex: |
  @article{solstice2026solace,
    title={Project Solace: Distilling Multi-Teacher Reasoning Traces from 7 Frontier Model Families},
    author={Solstice-AI Research Team},
    journal={Solstice-AI Technical Report},
    year={2026},
    url={https://solstice-ai.co/papers/solace-distillation-traces}
  }
tags:
  - "Distillation"
  - "Reasoning Traces"
  - "SFT"
  - "Open Datasets"
featured: true
---

## 1. Introduction & Motivation

Knowledge distillation has emerged as the primary mechanism for transferring the dense problem-solving capabilities of closed frontier models into accessible, transparent, and deployable sub-8B parameter models.

However, existing open distillation corpora suffer from three fundamental limitations:
1. **Single-Teacher Bias:** Models trained on single-source traces inherit specific stylistic and hallucination quirks of the teacher.
2. **Shallow Chain-of-Thought:** Many datasets truncate intermediate scratchpads, losing non-linear backtracking steps.
3. **Inconsistent Quality Filters & Massive Duplication:** Low-quality automated labeling frequently pollutes reasoning paths with circular logic and up to 25% verbatim duplicated prompts across scraped repositories.

Project Solace directly resolves these failure modes through a 60-dataset aggregation matrix combined with exact SHA256 deduplication and weighted round-robin shuffling.

```
Frontier Teachers (GLM 5.2, DeepSeek V4 Pro, GPT-5.6 Sol, Qwen 3.8 Max, Kimi K3, Fable 5, Codex, Manus)
                                      │
                                      ▼
             Exact SHA256 Deduplication (Purged 3,445,534 Overlapping Rows)
                                      │
                                      ▼
                 Project Solace 1.0 Omni (12,586,893 Unique Conversations)
                                      │
                                      ▼
             Target Student Models (Sub-8B Checkpoints: FP8, AWQ, GGUF, MLX)
```

---

## 2. Multi-Teacher Corpus Composition

The Solace 1.0 Omni corpus covers four primary cognitive domains:
- **Mathematical & Algorithmic Proofs:** Formal step-by-step proofs, Olympiad-level number theory, and Lean 4 verification.
- **Complex Code Synthesis & Refactoring:** Multi-file edits, compiler error resolution, SWE-bench replays, and unit test generation.
- **Agentic Tool Orchestration:** Real-world tool call transcripts, OpenHands rollouts, sandbox execution feedback, and browser traces.
- **Abstract Logic & ARC-AGI3:** Grid transformations and multi-step non-linear deduction trees.
