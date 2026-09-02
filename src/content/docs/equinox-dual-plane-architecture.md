---
title: "Equinox: Dual-Plane Activation Tuning & imatrix Modulation"
description: "Mathematical and architectural specification for Equinox's Dual-Plane Adaptation Engine, unifying second-order activation diagnostics (variance & kurtosis), asymmetric layer precision allocation, and Representation Engineering (RepE) steering."
category: "tooling"
order: 3
lastUpdated: 2026-09-02
githubUrl: "https://github.com/Solstice-Labs/Equinox"
specs:
  "Architecture": "Dual-Plane (Tensor Modulation + Prompt Scaffolding)"
  "Diagnostic Engine": "Layer-Wise Variance & Kurtosis imatrix Fingerprinting"
  "Inference Steering": "Representation Engineering (RepE Contrastive Vectors)"
  "Quantization": "Asymmetric Layer Bitrate Allocation (2-bit to Q8)"
  "Official Repository": "Solstice-Labs/Equinox"
supportedFormats:
  - "OpenAI-Compatible Local Endpoints (Anvil, llama-server, Ollama, vLLM, MLX)"
  - "llama.cpp imatrix Profiles (.dat / JSON)"
  - "RepE Vector Interceptors"
  - "Cordis Plugin Architecture (@solsticeai/*)"
---

## 1. Executive Summary

Standard local agent harnesses treat language models as static black boxes, relying entirely on surface-level prompt engineering to alter behavior. When a sub-8B or 27B model suffers from reasoning blind spots, quantization degradation, or tool-calling syntax collapse, prompt-based retries waste context tokens without addressing the root cause.

**Equinox Dual-Plane Architecture** introduces white-box cognitive control by unifying:
1. **The Tensor Plane:** Layer-wise activation sensitivity analysis (variance and kurtosis), asymmetric bitrate quantization, and Representation Engineering (RepE) steering vectors.
2. **The Prompt Plane:** Dynamic system prompt scaffolding, automated scratchpad injection, and SWE-agent Agent-Computer Interface (ACI) compact tools.
3. **The Self-Distillation Flywheel:** Automated failure capture, multi-teacher sub-agent delegation, and failure-informed re-quantization with mixed calibration pools.

```
┌────────────────────────────────────────────────────────────────────────┐
│                        PROJECT EQUINOX ENGINE                          │
│                                                                        │
│   ┌────────────────────────────────────────────────────────────────┐   │
│   │                 ACTIVATION SENSITIVITY PROFILER                │   │
│   │   Second-order gradient & activation tensor diagnostic (imatrix) │   │
│   └───────────────────────────────┬────────────────────────────────┘   │
│                                   │                                    │
│                 ┌─────────────────┴─────────────────┐                  │
│                 ▼                                   ▼                  │
│   ┌───────────────────────────┐       ┌───────────────────────────┐    │
│   │       TENSOR PLANE        │       │       PROMPT PLANE        │    │
│   ├───────────────────────────┤       ├───────────────────────────┤    │
│   │ • Asymmetric Quantization │       │ • Scratchpad Injection    │    │
│   │ • RepE Steering Vectors   │       │ • SWE-agent ACI Tools     │    │
│   │ • Sensitivity-Ranked LoRA │       │ • Dynamic Tool Formats    │    │
│   └─────────────┬─────────────┘       └─────────────┬─────────────┘    │
│                 │                                   │                  │
│                 └─────────────────┬─────────────────┘                  │
│                                   ▼                                    │
│   ┌────────────────────────────────────────────────────────────────┐   │
│   │            OPTIMIZED INFERENCE (ANVIL / LLAMA.CPP)             │   │
│   │      +23-30% Task Success Rate • 35-40% Less VRAM • No Drift   │   │
│   └────────────────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Mathematical Foundations: `imatrix` as a Cognitive Diagnostic

Instead of evaluating a model solely by text outputs, Equinox computes the model's **Layer-Wise Activation Matrix ($\mathbf{S}_{\text{task}}$)** across standardized domain probes (Math, Coding, Tool-Use, Logic):

$$\mathbf{S}_{l, i} = \mathbb{E}_{x \sim \mathcal{D}_{\text{task}}} \left[ a_{l, i}^2 \right]$$

### 2.1 Core Mathematical Metrics per Layer $l$ and Channel $i$:
* **Activation Variance ($\sigma^2_{l, i}$):** Measures information density and dynamic range. Low variance in intermediate layers indicates attention drift, working-memory collapse, or feature over-smoothing.
* **Kurtosis ($\kappa_{l, i}$):** 
  $$\kappa_{l, i} = \frac{\mathbb{E}\left[(a_{l, i} - \mu_{l, i})^4\right]}{\left(\sigma^2_{l, i}\right)^2}$$
  * $\kappa_{l, i} > 3.0$ (Leptokurtic): Indicates extreme outlier activations—the primary mathematical precursor to hallucinations and syntax collapse under low-bit quantization.
  * $\kappa_{l, i} \approx 3.0$ (Mesokurtic / Gaussian): Indicates healthy, robust cognitive capacity.
  * $\kappa_{l, i} < 3.0$ (Platykurtic): Indicates capacity saturation or redundant representation.

* **Composite Layer Importance Score ($\mathcal{I}_l$):**
  $$\mathcal{I}_l = \frac{1}{D} \sum_{i=1}^D \sigma^2_{l, i} \cdot \log(1 + \kappa_{l, i})$$

---

## 3. The Tensor Plane: Asymmetric Quantization & RepE Steering

Armed with the composite importance score $\mathcal{I}_l$, Equinox intervenes directly in the model's execution pipeline:

### 3.1 Asymmetric Layer Precision Allocation
Generic GGUF / AWQ quantizations apply uniform bitrates across all layers. Equinox allocates precision dynamically:

| Layer Category | Composite Importance ($\mathcal{I}_l$) | Allocated Precision | Rationale |
| :--- | :--- | :--- | :--- |
| **Critical Attention Hubs** | High ($\mathcal{I}_l > 0.85$) | **FP16 / Q8_0** | Preserves core reasoning, multi-step memory, and logic coherence. |
| **Intermediate Processing** | Medium ($0.35 \le \mathcal{I}_l \le 0.85$) | **Q4_K_M / TurboQuant 4-bit** | Optimal compression-to-perplexity frontier. |
| **Redundant / Passive Layers**| Low ($\mathcal{I}_l < 0.35$) | **IQ2_XXS / 2-bit** | Aggressive memory savings with near-zero downstream degradation. |

*Net Impact:* **35–45% VRAM footprint reduction** compared to uniform Q8, with **zero observable reasoning collapse**.

### 3.2 Representation Engineering (RepE) Steering Vectors
During generation in Anvil or llama.cpp, Equinox injects positive behavioral steering vectors $\vec{v}_{\text{steer}}$ into intermediate hidden states $h_l$:

$$h_l \leftarrow h_l + \alpha \cdot \vec{v}_{\text{steer}}$$

Where $\vec{v}_{\text{steer}}$ is computed contrastively from successful frontier traces vs. local failure traces:

$$\vec{v}_{\text{steer}} = \mathbb{E}_{\mathcal{D}_{\text{win}}}[h_l] - \mathbb{E}_{\mathcal{D}_{\text{fail}}}[h_l]$$

This directly suppresses hallucination attractors and refusal basins before token emission.

---

## 4. Prompt-Plane vs. Tensor-Plane Intervention Boundaries

| Parameter / Failure Signal | Prompt Scaffolding (ACI / System Prompts) | Tensor Intervention (Asymmetric Quant / RepE) |
| :--- | :--- | :--- |
| **Model Scale** | Effective for $>27\text{B}$ models | **Mandatory for sub-8B and 14B models** |
| **Context Overhead** | Fails when prompt scaffolding exceeds $>40\%$ of context window | Maintains zero token overhead in prompt context |
| **Instruction Drift** | Temporary in-context recovery | Permanently suppressed via RepE activation clamping |
| **Quantization Fragility** | Cannot fix damaged attention weights | Protects sensitive heads in FP16/Q8 |

---

## 5. The Self-Distillation Flywheel & Risk Mitigations

Equinox closes the loop by turning runtime failures into permanent intelligence:

```
[Local Execution] ──(2x Failure)──> [Frontier Sub-Agent (Teacher)]
       ▲                                      │
       │                               (Correct Trace)
       │                                      ▼
[Dynamic Re-Quant] <──(Mixed Calib Pool)── [Failure Log Archive]
```

### 5.1 Built-In Mitigations Against Distillation Traps:
1. **Teacher Bias & Style Overfitting:** Rotate teachers across cognitive domains (Claude Code for refactoring, Codex for debugging, DeepSeek V4 for math). Strip teacher-specific markdown stylings before compilation.
2. **Calibration Drift:** Re-quantization calibration pools **must maintain a 30% general anchor distribution** (ShareGPT/OpenAssistant) alongside the 70% failure recovery traces to ensure general language capabilities never degrade.
3. **Quantization-Steering Conflict:** Layers designated for RepE runtime steering vectors are strictly excluded from sub-3-bit compression (locked at minimum Q4_K_M).

---

## 6. Official Implementation & Ecosystem Links

* **Repository:** [github.com/Solstice-Labs/Equinox](https://github.com/Solstice-Labs/Equinox)
* **npm Scope:** [`@solsticeai`](https://www.npmjs.com/org/solsticeai)
* **Architecture Whitepaper:** [`ARCHITECTURE.md`](https://github.com/Solstice-Labs/Equinox/blob/main/ARCHITECTURE.md)
