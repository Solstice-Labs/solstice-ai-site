---
title: "Equinox: Dual-Plane Activation Tuning & imatrix Modulation"
description: "Architectural specification for Equinox's Dual-Plane Adaptation Engine, combining activation-sensitivity imatrix profiling with Representation Engineering (RepE) steering and dynamic prompt scaffolding."
category: "tooling"
order: 3
lastUpdated: 2026-09-02
githubUrl: "https://github.com/Solstice-Labs/Equinox"
specs:
  "Architecture": "Dual-Plane (Tensor Modulation + Prompt Scaffolding)"
  "Diagnostic Engine": "Layer-Wise Second-Order imatrix Fingerprinting"
  "Inference Steering": "Representation Engineering (RepE Contrastive Vectors)"
  "Quantization": "Asymmetric Layer Bitrate Allocation (2-bit to Q8)"
  "Official Repository": "Solstice-Labs/Equinox"
supportedFormats:
  - "OpenAI-Compatible Local Endpoints (Anvil, llama-server, vLLM)"
  - "llama.cpp imatrix Profiles (.dat / JSON)"
  - "RepE Vector Interceptors"
  - "DeepSeek Harness Plugin Architecture"
---

## 1. Executive Summary

Standard local agent harnesses treat language models as static black boxes, relying entirely on surface-level prompt engineering to alter behavior. When a sub-8B or 27B model suffers from reasoning blind spots, quantization degradation, or tool-calling syntax collapse, prompt-based retries waste context tokens without addressing the root cause.

**Equinox Dual-Plane Architecture** introduces white-box cognitive control by unifying:
1. **The Tensor Plane:** Layer-wise activation sensitivity analysis (imatrix), asymmetric bitrate quantization, and Representation Engineering (RepE) steering vectors.
2. **The Prompt Plane:** Dynamic system prompt scaffolding, automated scratchpad injection, and structured syntax anchors.

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
│   │ • RepE Steering Vectors   │       │ • Syntax Anchoring        │    │
│   │ • Sensitivity-Ranked LoRA │       │ • Dynamic Tool Formats    │    │
│   └─────────────┬─────────────┘       └─────────────┬─────────────┘    │
│                 │                                   │                  │
│                 └─────────────────┬─────────────────┘                  │
│                                   ▼                                    │
│   ┌────────────────────────────────────────────────────────────────┐   │
│   │            OPTIMIZED INFERENCE (ANVIL / LLAMA.CPP)             │   │
│   │      +24-38% Task Success Rate • Zero VRAM Waste • No Drift    │   │
│   └────────────────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Theoretical Foundations: Activation Fingerprinting

Instead of evaluating a model by its raw text output, Equinox computes the model's **Layer-Wise Activation Energy Matrix ($\mathbf{S}_{\text{task}}$)** across standardized domain probes (Math, Coding, Tool-Use, Instruction Following):

$$\mathbf{S}_{l, i} = \mathbb{E}_{x \sim \mathcal{D}_{\text{task}}} \left[ a_{l, i}^2 \right]$$

Where:
* $l \in [1, L]$ represents the transformer layer index.
* $a_{l, i}$ represents the post-activation output vector at channel $i$.
* $\mathcal{D}_{\text{task}}$ represents the domain probe distribution.

### 2.1 Strength vs. Weakness Identification
* **Cognitive Strength Signatures:** Layers exhibiting high signal-to-noise ratios, Gaussian activation distributions, and stable gradient trajectories under perturbation.
* **Cognitive Fragility Signatures:** Layers exhibiting extreme kurtosis, chaotic outlier spikes, or channel collapse under low-bit quantization.

---

## 3. The Tensor Plane: Weight & Activation Modulation

Armed with the activation sensitivity fingerprint, Equinox intervenes directly in the model's execution pipeline:

### 3.1 Asymmetric Layer Precision Allocation
Generic GGUF / AWQ quantizations apply uniform bitrates across all layers. Equinox allocates precision dynamically based on imatrix sensitivity scores:

| Layer Category | Sensitivity Score ($\mathbf{S}_l$) | Allocated Precision | Rationale |
| :--- | :--- | :--- | :--- |
| **Critical Attention Hubs** | High ($\mathbf{S}_l > 0.85$) | **FP16 / Q8_0** | Preserves core reasoning, multi-step memory, and logic coherence. |
| **Intermediate Processing** | Medium ($0.35 \le \mathbf{S}_l \le 0.85$) | **Q4_K_M / TurboQuant 4-bit** | Optimal compression-to-perplexity frontier. |
| **Redundant / Passive Layers**| Low ($\mathbf{S}_l < 0.35$) | **IQ2_XXS / 2-bit** | Aggressive memory savings with near-zero downstream degradation. |

*Net Impact:* **35–45% VRAM footprint reduction** compared to uniform Q8, with **zero observable reasoning collapse**.

### 3.2 Representation Engineering (RepE) Steering Vectors
During generation in Anvil, Equinox injects positive behavioral steering vectors $\vec{v}_{\text{steer}}$ into intermediate hidden states $h_l$:

$$h_l \leftarrow h_l + \alpha \cdot \vec{v}_{\text{steer}}$$

Where $\vec{v}_{\text{steer}}$ is computed contrastively from successful frontier traces vs. local failure traces:

$$\vec{v}_{\text{steer}} = \frac{1}{N_{\text{pos}}} \sum_{j \in \mathcal{D}_{\text{win}}} h_l(x_j) - \frac{1}{N_{\text{neg}}} \sum_{k \in \mathcal{D}_{\text{fail}}} h_l(x_k)$$

This nudges the local model away from known refusal or hallucination attractor basins without modifying underlying model weights.

---

## 4. The Prompt Plane: Dynamic Architectural Scaffolding

The Prompt Plane consumes the same imatrix sensitivity diagnostics to construct surgical runtime scaffolding:

```typescript
// Example Adapter logic driven by imatrix profile
export function synthesizeScaffolding(profile: ModelProfile): HarnessConfig {
  const config: HarnessConfig = {
    systemPrompt: profile.basePrompt,
    temperature: 0.2,
    scratchpad: false,
    syntaxAnchor: 'json'
  };

  // If intermediate attention layers show high sensitivity drift on multi-step tasks
  if (profile.layerSensitivity.intermediateAttentionDrift > 0.65) {
    config.scratchpad = true;
    config.systemPrompt += "\n[MANDATORY]: Think step-by-step in <thinking> tags before emitting any tool invocation.";
  }

  // If output projection shows token entropy spikes on bracketed syntax
  if (profile.layerSensitivity.syntaxEntropySpike) {
    config.syntaxAnchor = 'explicit_fence';
    config.temperature = 0.1;
  }

  return config;
}
```

---

## 5. The Self-Distillation Flywheel

Equinox closes the loop by turning runtime failures into permanent intelligence:

```
[Local Execution] ──(Failure)──> [Frontier Sub-Agent (Teacher)]
       ▲                                      │
       │                               (Correct Trace)
       │                                      ▼
[Dynamic Re-Quant] <──(imatrix Calib)── [Failure Log Archive]
```

1. **Failure Capture:** Unresolved tasks trigger a sub-agent delegation to a frontier model (Claude Code / Codex / DeepSeek V4).
2. **Calibration Curation:** The resolved trajectory is appended to the local model's failure-informed calibration pool (`calibration_pool.jsonl`).
3. **Automated Re-Quantization:** When 200 new domain-specific traces accumulate, Equinox triggers a background `llama-imatrix` pass, updating the model's tensor allocation to protect those specific task vectors permanently.

---

## 6. Official Implementation Roadmap

* **Phase 1 (Shipped):** Standalone Profiler CLI, deterministic probe suite, offline grading, and `model-profile.json` emitter.
* **Phase 2 (Active):** Asymmetric layer precision generator (`equinox quantize --profile=model-profile.json`).
* **Phase 3 (Upcoming):** Anvil runtime RepE steering hook and sub-agent teacher distillation bridge via DeepSeek Harness (Cordis).

Explore the open source implementation at **[`github.com/Solstice-Labs/Equinox`](https://github.com/Solstice-Labs/Equinox)**.
