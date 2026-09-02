---
title: "Dataset Schema & Parquet Specification"
description: "Detailed Apache Parquet field definitions, reasoning token tags, and multi-teacher provenance metadata."
category: "datasets"
order: 2
lastUpdated: 2026-08-22
hfRepoId: "Solstice-AI/Solace-1.0-GLM5.2-Fable5-GPT5.6Sol-DeepSeekV4Pro0813-Qwen3.8Max-KimiK3-Manus"
specs:
  "Parquet Columns": "11 Primary Fields"
  "Compression": "Snappy"
  "Row Count": "482,910 Traces"
  "Total Size": "4.82 GB (Compressed)"
supportedFormats:
  - "Apache Parquet"
  - "HuggingFace Datasets"
  - "JSON Lines (.jsonl)"
---

## Overview

Solace 1.0 is distributed as Apache Parquet files partitioned by task domain. Each sample includes the raw problem statement, the multi-turn verified scratchpad, final grounded solution, and comprehensive provenance metadata attributing teacher models and validation metrics.

---

## Parquet Schema Reference

```json
{
  "id": "solace_v1_math_0049182",
  "domain": "mathematics",
  "subdomain": "olympiad_number_theory",
  "teacher_models": ["gpt-5.6-sol", "deepseek-v4-pro", "glm-5.2"],
  "consensus_score": 0.942,
  "verification_passed": true,
  "compiler_feedback": null,
  "prompt": "Find all positive integers n such that 2^n + 12^n + 2026^n is a perfect square.",
  "thought_chain": "1. Analyze modulo arithmetic...\n2. Test base cases n=1, 2, 3...\n3. For n > 2, examine mod 7 and mod 13...",
  "solution": "The only positive integer solution is n = 1.",
  "token_count": 1420
}
```

---

## Field Descriptions

| Field Name | Type | Description |
| :--- | :--- | :--- |
| `id` | `String` | Unique cryptographic identifier based on prompt hash. |
| `domain` | `String` | High-level domain: `mathematics`, `code_synthesis`, `agentic_tool_use`, `strategic_reasoning`. |
| `subdomain` | `String` | Granular specialization (e.g. `dynamic_programming`, `kernel_optimization`, `proof_verification`). |
| `teacher_models` | `List[String]` | List of teacher architectures that reached consensus on this solution path. |
| `consensus_score` | `Float` | Semantic reasoning graph overlap score (ranges from $0.0$ to $1.0$). |
| `verification_passed` | `Boolean` | Flag indicating successful pass through compiler sandboxes and proof engines. |
| `compiler_feedback` | `Nullable[String]` | Output logs from unit tests, linter assertions, or Lean 4 proof logs if applicable. |
| `prompt` | `String` | Unmodified task instruction or user problem statement. |
| `thought_chain` | `String` | Full step-by-step thinking process without truncation or artificial summaries. |
| `solution` | `String` | Final grounded answer and code artifacts. |
| `token_count` | `Integer` | Total token count across prompt, thought chain, and solution. |
