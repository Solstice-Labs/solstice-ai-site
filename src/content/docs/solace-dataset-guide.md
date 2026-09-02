---
title: "Project Solace 1.0 Omni Dataset Guide"
description: "How to inspect, download, filter, and train models on the 12.59M verified frontier conversation corpus (137.7 GB)."
category: "datasets"
order: 1
lastUpdated: 2026-08-30
hfRepoId: "Solstice-AI/Solace-1.0-Omni"
specs:
  "Unique Conversations": "12,586,893"
  "Source Datasets": "60 Datasets (7 Frontier Model Families)"
  "Uncompressed Size": "137.7 GB (JSONL)"
  "Compressed Size": "34.3 GB (solace_final.jsonl.gz)"
  "Duplicates Purged": "3,445,534 Exact SHA256 Copies (21.5% overlap)"
  "License": "AGPL-3.0"
supportedFormats:
  - "JSONL (OpenAI messages format)"
  - "Hugging Face Datasets"
  - "Gzip Stream (.jsonl.gz)"
---

## Overview

**Project Solace 1.0 Omni** is the largest verified frontier-model distillation corpus ever released. It aggregates **12,586,893 unique conversations** across **60 audited source datasets** from 7 premier 2026 frontier model families:

* **GLM-5.2 (15 datasets):** OpenHands agent rollouts, on-policy regeneration, and ARC traces.
* **Claude Fable 5 & Mythos 5:** Anthropic's reasoning engines and full agentic coding sessions.
* **GPT-5.6 Sol & GPT-5.5 Codex:** OpenAI coding traces, debugging trajectories, and ARC-AGI3 challenges.
* **DeepSeek V4 Pro 0813:** 200K-distilled math/STEM, ResearchMath, and SWE-bench replays.
* **Qwen 3.8-Max:** Cross-architecture multi-teacher distillation traces.
* **Kimi K3 & Opus 4.7 Multi:** Long-context informational synthesis and reasoning specialists.
* **Manus Agents:** Real-world tool call transcripts and browser execution traces.

---

## 1. Quickstart: Load & Stream with Datasets

```python
from datasets import load_dataset

# Load full training set directly from Hugging Face
dataset = load_dataset(
    "Solstice-AI/Solace-1.0-Omni",
    data_files="solace_final.jsonl.gz",
    split="train"
)

print(f"Total verified examples: {len(dataset):,}")
```

---

## 2. Decompression & Direct CLI Usage

```bash
# Download compressed artifact (34.3 GB) via modern hf CLI
hf download Solstice-AI/Solace-1.0-Omni solace_final.jsonl.gz --repo-type dataset --local-dir ./data

# Decompress to drop-in 137.7 GB JSONL
zcat ./data/solace_final.jsonl.gz > ./data/solace_final.jsonl
```

---

## 3. Schema & Row Format

Every row is formatted in native OpenAI `messages` format with intact `<think>` reasoning blocks and tool-call metadata:

```json
{
  "messages": [
    {"role": "system", "content": "You are a multi-step reasoning assistant."},
    {"role": "user", "content": "Find all solutions to f(x + y) = f(x) + f(y) + 2xy."},
    {"role": "assistant", "content": "<think>\n1. Set y = 0...\n2. Let g(x) = f(x) - x^2...\n</think>\nAll solutions are of the form f(x) = x^2 + cx."}
  ],
  "source": "mgoin/open-perfectblend-glm5.2-regen",
  "model": "glm-5.2"
}
```

---

## 4. Filtering by Model or Capability

Filter specific model distributions in one line:

```python
import json

with open("./data/solace_final.jsonl") as f:
    # Filter GLM-5.2 agent rollouts
    glm_examples = [json.loads(line) for line in f if '"model":"glm-5.2"' in line]
```

---

## 5. Companion Datasets on Hugging Face Hub

* **[`Solstice-AI/Axiom-1.0`](https://huggingface.co/datasets/Solstice-AI/Axiom-1.0-Opus4.7-Kimi2.6-GLM5.2-Deepseek4-Mythos5-Fable5-Qwen3.7):** 102 GB pure text reasoning corpus, 4.64M samples, 27B+ tokens.
* **[`Solstice-AI/Complete-FABLE.5-traces-2M`](https://huggingface.co/datasets/Solstice-AI/Complete-FABLE.5-traces-2M):** 2,006,487 deduplicated agentic coding and CoT traces in Parquet & JSONL format.
