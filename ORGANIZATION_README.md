<div align="center">

```
  ███████╗ ██████╗ ██╗     ███████╗████████╗██╗ ██████╗███████╗    ██╗      █████╗ ██████╗ ███████╗
  ██╔════╝██╔═══██╗██║     ██╔════╝╚══██╔══╝██║██╔════╝██╔════╝    ██║     ██╔══██╗██╔══██╗██╔════╝
  ███████╗██║   ██║██║     ███████╗   ██║   ██║██║     █████╗      ██║     ███████║██████╔╝███████╗
  ╚════██║██║   ██║██║     ╚════██║   ██║   ██║██║     ██╔══╝      ██║     ██╔══██║██╔══██╗╚════██║
  ███████║╚██████╔╝███████╗███████║   ██║   ██║╚██████╗███████╗    ███████╗██║  ██║██████╔╝███████║
  ╚══════╝ ╚═════╝ ╚══════╝╚══════╝   ╚═╝   ╚═╝ ╚═════╝╚══════╝    ╚══════╝╚═╝  ╚═╝╚═════╝ ╚══════╝
```

### **Distilling Frontier Intelligence into Open Silicon**

[![Website](https://img.shields.io/badge/Website-solstice--ai.co-1a1a1a?style=for-the-badge&logo=cloudflare&logoColor=F38020)](https://solstice-ai.co)
[![Hugging Face](https://img.shields.io/badge/Hugging_Face-Solstice--AI-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/Solstice-AI)
[![Organization](https://img.shields.io/badge/GitHub-Solstice--Labs-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Solstice-Labs)
[![License](https://img.shields.io/badge/License-AGPL--3.0-blue?style=for-the-badge)](LICENSE)

<p align="center">
  <a href="#the-thesis"><b>The Thesis</b></a> •
  <a href="#verified-benchmark-metrics"><b>Verified Metrics</b></a> •
  <a href="#core-repositories--tooling"><b>Core Repositories</b></a> •
  <a href="#released-weights--quantized-checkpoints"><b>Models</b></a> •
  <a href="#quickstart"><b>Quickstart</b></a> •
  <a href="#technical-reports--research"><b>Papers</b></a>
</p>

---

</div>

## The Thesis

Frontier intelligence is increasingly locked behind centralized APIs and massive parameter barriers that commodity datacenters cannot sustain. 

**Solstice Labs** engineers the open post-training infrastructure required to break this bottleneck:
* **Multi-Teacher Distillation:** Synthesizing cross-architecture reasoning, agentic tool transcripts, and Olympiad proof traces from top frontier models without single-teacher bias.
* **Radical Compression:** Deploying **Google TurboQuant** (Fast Walsh-Hadamard Transform rotations) and sub-8B integer quantization to run long-context reasoning locally on consumer GPUs and Apple Silicon.

---

## Verified Benchmark Metrics

```
+---------------------------+------------------------------------------+------------------------+
| Corpus / Engine           | Scale / Measurement                      | Provenance / Lab       |
+---------------------------+------------------------------------------+------------------------+
| Project Solace 1.0 Omni   | 12,586,893 Unique Conversations          | 60 Sources · 7 Models  |
| Uncompressed Volume       | 137.7 GB (34.3 GB compressed jsonl.gz)   | Verified SHA256 Dedup  |
| Project Axiom             | 102 GB Pure Text · 27B+ Reasoning Tokens | 8 Frontier Labs        |
| Complete FABLE.5 Traces   | 2,006,487 Deduplicated Agentic Traces    | Parquet + JSONL        |
| Anvil Engine (TurboQuant) | 4.6x KV Cache Reduction (+46.8% Speed)   | FWHT + Gemma MTP       |
+---------------------------+------------------------------------------+------------------------+
```

---

## Ecosystem Architecture

```
                    +--------------------------------------------------------+
                    |            Frontier Multi-Teacher Ensemble             |
                    |   GLM-5.2 • Fable 5 • GPT-5.6 Sol • DeepSeek V4 Pro    |
                    |      Qwen 3.8-Max • Kimi K3 • Mythos 5 • Manus         |
                    +---------------------------+----------------------------+
                                                |
                                                v
                    +--------------------------------------------------------+
                    |               Automated Consensus Harness              |
                    |  • Exact SHA256 Deduplication (Purged 3.45M copies)   |
                    |  • Trace Inversion & Multi-Step Backtracking Checks    |
                    |  • Weighted Round-Robin Interleaved Shuffling          |
                    +---------------------------+----------------------------+
                                                |
                                                v
                    +--------------------------------------------------------+
                    |              Project Solace 1.0 Omni (137.7 GB)        |
                    |              12,586,893 Verified JSONL Conversations   |
                    +---------------------------+----------------------------+
                                                |
                                                v
                    +--------------------------------------------------------+
                    |             Anvil High-Throughput Engine               |
                    |  • Google TurboQuant (FWHT O(d log d) Rotation)        |
                    |  • Qwen 3.6 NextN Speculative Decoding (+30-50% Tok/s) |
                    |  • Fused Metal & CUDA 4-bit / 8-bit Execution          |
                    +--------------------------------------------------------+
```

---

## Core Repositories & Tooling

### Inference & Hardware Engines
* **[`Solstice-Labs/anvil`](https://github.com/Solstice-Labs/anvil)**  
  High-performance inference engine integrating **Google TurboQuant** (Fast Walsh-Hadamard Transform KV cache compression), Gemma 4 Multi-Token Prediction (MTP), and Qwen 3.6 NextN speculative decoding.
* **[`Solstice-Labs/turboquant-mlx`](https://github.com/Solstice-Labs/turboquant-mlx)**  
  Fused Apple Metal kernels for TurboQuant KV cache compression on Apple Silicon Macs. 4.6x memory reduction at 98% FP16 speed.

### Open Distillation Datasets (Hugging Face)
* **[`Solstice-AI/Solace-1.0-Omni`](https://huggingface.co/datasets/Solstice-AI/Solace-1.0-Omni)**  
  *Flagship Dataset.* 12.59M verified multi-turn conversations across 60 audited sources and 7 frontier model architectures.
* **[`Solstice-AI/Axiom-1.0`](https://huggingface.co/datasets/Solstice-AI/Axiom-1.0-Opus4.7-Kimi2.6-GLM5.2-Deepseek4-Mythos5-Fable5-Qwen3.7)**  
  102 GB pure-text reasoning corpus with 4.64 million samples and ~27.3 billion tokens of long-horizon CoT traces.
* **[`Solstice-AI/Complete-FABLE.5-traces-2M`](https://huggingface.co/datasets/Solstice-AI/Complete-FABLE.5-traces-2M)**  
  2,006,487 deduplicated agentic coding and reasoning trajectories in Parquet & raw JSONL format.

---

## Released Weights & Quantized Checkpoints

All models are published under [`Solstice-AI`](https://huggingface.co/Solstice-AI) and [`SolsticeAI`](https://huggingface.co/SolsticeAI):

| Checkpoint Name | Arch / Base | Precision | Target Platform | Link |
| :--- | :--- | :--- | :--- | :--- |
| **Qwen3.8-27B-Uncensored** | Qwen 3.8 27B | 6-bit MLX | Apple Silicon | [Hub Checkpoint](https://huggingface.co/SolsticeAI/Qwen3.8-27B-Uncensored-mlx-6Bit) |
| **Qwen3.8-27B-Cold-Fusion-GAIN** | Qwen 3.8 27B | 6-bit MLX | Apple Silicon | [Hub Checkpoint](https://huggingface.co/Solstice-AI/Qwen3.8-27B-Cold-Fusion-GAIN-V1.1-mlx-6Bit) |
| **Huihui-Qwen3.6-35B-Opus-Abliterated** | Qwen 3.6 35B | Q8_0 GGUF | llama.cpp / CUDA | [Hub Checkpoint](https://huggingface.co/Solstice-AI/Huihui-Qwen3.6-35B-A3B-Claude-4.7-Opus-abliterated-Q8_0-GGUF) |
| **Qwen3.6-35B-Claude-Opus-Distill** | Qwen 3.6 35B | 8-bit MLX | Apple Silicon | [Hub Checkpoint](https://huggingface.co/Solstice-AI/qwen3.6-35b-a3b-claude-opus-4.7-distill-abliterated-mlx-8bit) |
| **ThinkingCap-Qwen3.6-27B** | Qwen 3.6 27B | 6-bit MLX | Apple Silicon | [Hub Checkpoint](https://huggingface.co/Solstice-AI/ThinkingCap-Qwen3.6-27B-mlx-6Bit) |
| **Athena-27B-UltraEfficient** | Athena 27B | Native | vLLM / PyTorch | [Hub Checkpoint](https://huggingface.co/Solstice-AI/Athena-27B-UltraEfficient) |

---

## Quickstart

### 1. Download Datasets with modern `hf` CLI

```bash
# Download Solace 1.0 Omni compressed archive (34.3 GB)
hf download Solstice-AI/Solace-1.0-Omni solace_final.jsonl.gz --repo-type dataset --local-dir ./data

# Stream directly with Hugging Face Datasets
python3 -c "
from datasets import load_dataset
ds = load_dataset('Solstice-AI/Solace-1.0-Omni', data_files='solace_final.jsonl.gz', split='train')
print(f'Loaded {len(ds):,} verified instances.')
"
```

### 2. Run with Anvil CLI (`anvil`)

```bash
# Pull Solstice GGUF checkpoint into local Anvil registry
anvil pull hf:Solstice-AI/Huihui-Qwen3.6-35B-A3B-Claude-4.7-Opus-abliterated-Q8_0-GGUF

# Run with 262k context compressed to ~4 GB via TurboQuant (and save flags)
anvil run huihui-qwen --ctx 262144 --type-k turbo4 --type-v turbo3 --save

# Launch instantly with one word
anvil huihui-qwen
```

### 3. Apple Silicon Inference (MLX)

```bash
pip install mlx-lm
mlx_lm.generate \
  --model Solstice-AI/Qwen3.8-27B-Cold-Fusion-GAIN-V1.1-mlx-6Bit \
  --prompt "Write an optimized concurrent task scheduler in Rust." \
  --max-tokens 1024
```

---

## Technical Reports & Research

* **[Project Solace: Distilling Multi-Teacher Reasoning Traces from 7 Frontier Architectures](https://solstice-ai.co/papers/solace-distillation-traces)**  
  *Gondaliya et al., Solstice Labs Technical Report, 2026.*
* **[Cross-Architecture Reasoning Transfer in Sub-8B Parameter Regimes](https://solstice-ai.co/papers/cross-architecture-reasoning-transfer)**  
  *Solstice Labs Research, 2026.*
* **[TurboQuant & Anvil: Breaking the KV Cache Memory Wall](https://solstice-ai.co/blog/turboquant-anvil-llama-cpp)**  
  *Solstice Labs Systems Engineering, 2026.*

```bibtex
@article{solstice2026solace,
  title={Project Solace: Distilling Multi-Teacher Reasoning Traces from 7 Frontier Model Families},
  author={Solstice-AI Research Team},
  journal={Solstice-AI Technical Report},
  year={2026},
  url={https://solstice-ai.co/papers/solace-distillation-traces}
}
```

---

## Community & Ecosystem

* **Website & Docs:** [https://solstice-ai.co](https://solstice-ai.co)
* **GitHub Organization:** [https://github.com/Solstice-Labs](https://github.com/Solstice-Labs)
* **Hugging Face Organization:** [https://huggingface.co/Solstice-AI](https://huggingface.co/Solstice-AI)

<div align="center">
<sub>Built with precision by Solstice Labs. Open weights, open data, open science.</sub>
</div>
