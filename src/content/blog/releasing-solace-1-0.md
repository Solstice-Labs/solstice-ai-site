---
title: "Releasing Project Solace 1.0 Omni: 12.59M Conversations Across 60 Datasets"
description: "We are releasing Project Solace 1.0 Omni, a 137.7 GB open dataset of verified frontier-model distillation conversations across 7 model families."
pubDate: 2026-08-19
author: "Solstice-AI Core Team"
tags:
  - "Releases"
  - "Datasets"
  - "Distillation"
readingTime: "5 min read"
takeaways:
  - "Solace 1.0 Omni aggregates 12,586,893 unique conversations across 60 audited source datasets."
  - "Exact SHA256 deduplication removed 3,445,534 redundant copies (21.5% overlap)."
  - "Released in drop-in OpenAI messages format under AGPL-3.0 on Hugging Face (34.3 GB compressed)."
featured: true
---

Today we are releasing **Project Solace 1.0 Omni**, the largest verified frontier-model distillation corpus ever released — comprising **12,586,893 unique conversations** (137.7 GB uncompressed / 34.3 GB compressed) in drop-in OpenAI messages format.

### Why Multi-Teacher Distillation?

Most distillation datasets rely on a single teacher model. This creates an echo-chamber effect where the student model inherits the teacher's idiosyncratic weaknesses, phrasing tropes, and blind spots.

Solace 1.0 Omni samples problem-solving paths from 7 verified frontier model families:
- **GLM 5.2 (15 datasets):** OpenHands agent rollouts, on-policy regeneration, and ARC traces.
- **Claude Fable 5 & Mythos 5:** Anthropic's reasoning engines and agentic coding sessions.
- **GPT-5.6 Sol & GPT-5.5 Codex:** Coding, debugging, and ARC-AGI3 challenges.
- **DeepSeek V4 Pro 0813:** 200K math/STEM and SWE-bench replays.
- **Qwen 3.8-Max:** Cross-architecture distillation data.
- **Kimi K3 / Opus 4.7 Multi:** Long-context synthesis and reasoning specialists.
- **Manus Agents:** Tool use and browser execution transcripts.

---

### Key Corpus Metrics (Measured, Not Estimated)

| Metric | Value |
| :--- | :--- |
| **Unique Conversations** | **12,586,893** |
| **Source Rows Ingested** | 16,032,427 |
| **Exact Duplicates Purged** | 3,445,534 (21.5% cross-dataset overlap) |
| **Uncompressed Size** | 137.7 GB |
| **Compressed File** | `solace_final.jsonl.gz` — 34.3 GB |
| **Source Datasets** | 60 Audited Datasets |
| **Ordering** | Weighted round-robin interleaved |

---

### How to Access the Data

The complete dataset is hosted on Hugging Face:

```python
from datasets import load_dataset

dataset = load_dataset(
    "Solstice-AI/Solace-1.0-Omni",
    data_files="solace_final.jsonl.gz",
    split="train"
)
print(f"Total samples: {len(dataset):,}")
```

For technical methodology and dedup details, read the [Solace 1.0 Technical Report](/papers/solace-distillation-traces).
