---
title: "Zero-Degradation Knowledge Distillation: Preserving Out-of-Distribution Math and Code Capabilities"
authors:
  - "Solstice-AI Research Team"
pubDate: 2026-09-01
tldr: "Empirical study measuring reasoning decay across STEM benchmarks when optimizing student models for general conversational fluencies, with a capability-preserving distillation framework that maintains 99.1% of teacher STEM performance."
abstract: "Knowledge distillation pipelines optimized for general conversational fluency often inadvertently degrade the student model's out-of-distribution capabilities in STEM domains, particularly mathematical reasoning and code generation. We present Capability-Preserving Distillation (CPD), a framework that identifies and protects critical capability regions during training through gradient surgery, capability-aware data balancing, and elastic weight consolidation on STEM-critical parameters. Evaluated across 15 STEM benchmarks and 5 general conversational metrics, CPD achieves 99.1% preservation of teacher STEM performance while matching or exceeding general conversational quality—a zero-degradation outcome that standard distillation fails to achieve, with typical STEM accuracy losses of 5-12%."
venue: "Research Technical Report"
highlightMetrics:
  - label: "STEM Preservation"
    value: "99.1%"
  - label: "Conversational Match"
    value: "101.3%"
  - label: "Degradation"
    value: "0.9%"
bibtex: |
  @article{solstice2026zerodegradation,
    title={Zero-Degradation Knowledge Distillation: Preserving Out-of-Distribution Math and Code Capabilities},
    author={Solstice-AI Research Team},
    journal={Solstice-AI Technical Report},
    year={2026},
    url={https://solstice-ai.co/papers/zero-degradation-distillation}
  }
tags:
  - "Zero Degradation"
  - "STEM Preservation"
  - "Capability Conservation"
  - "Gradient Surgery"
featured: false
---

## 1. Introduction & Motivation

Knowledge distillation from frontier LLMs into compact student models is typically optimized for overall performance across a mixed training distribution. However, this optimization strategy has a well-documented failure mode: the student model improves on in-distribution tasks at the expense of out-of-distribution (OOD) capabilities, particularly in specialized STEM domains.

The "Strong Teacher Not Needed? On Distillation in LLM Pretraining" paper (arXiv 2605.23857, May 2026) explored the assumption that stronger teachers always yield better students, finding that this is not always the case when the student's training objective differs from the teacher's capabilities. The on-policy distillation survey (arXiv 2604.00626, May 2026) noted that distillation minimizes KL divergence over states drawn from the dataset distribution, which "preserves out-of-domain capabilities" only when the dataset distribution is sufficiently broad. The DA-KD framework (ICML 2025) demonstrated difficulty-aware weighting can improve text generation quality but did not explicitly address STEM capability preservation.

We call this phenomenon **reasoning decay**: the progressive loss of specialized reasoning capabilities as the student model is optimized for general conversational quality. Our measurements reveal that standard distillation reduces Math-500 accuracy by 7.3%, HumanEval+ by 8.1%, and MATH by 9.4% in 7B student models trained on general conversational corpora.

## 2. The Anatomy of Reasoning Decay

### 2.1 Capability Interference

Reasoning decay arises from **capability interference**: the gradient updates that improve conversational fluency conflict with the gradient updates that maintain STEM reasoning capabilities. We measure this interference by computing the gradient conflict ratio between conversational and STEM training examples:

$$\rho_{conv,STEM} = 1 - \frac{g_{conv} \cdot g_{STEM}}{\|g_{conv}\| \cdot \|g_{STEM}\|}$$

Across 50,000 training steps, the average conflict ratio is 0.67, indicating that 67% of conversational gradient updates point in directions that are detrimental to STEM performance. This interference is not uniform: mathematical reasoning shows higher conflict (0.74) than code generation (0.58), likely because mathematical proofs require more rigid logical structure that conflicts with the flexible, conversational tone optimized in general training.

### 2.2 Parameter-Level Analysis

We identify which parameters are most critical for STEM capabilities by computing the Fisher Information Matrix diagonal for STEM-specific examples. The top 5% of parameters by Fisher Information contribute 73% of STEM task accuracy. These parameters are disproportionately concentrated in middle layers (layers 16-28 for a 32-layer model) and in attention heads that track mathematical notation and code syntax.

When standard distillation updates these STEM-critical parameters to improve conversational quality, STEM accuracy drops. The rate of decay is proportional to the gradient conflict ratio: STEM-critical parameters with higher conflict to conversational gradients degrade faster.

### 2.3 Catastrophic vs. Gradual Decay

We observe two distinct decay regimes. **Gradual decay** occurs during early training (steps 0-10k), where STEM accuracy declines steadily at 0.3% per 1,000 steps as the student adjusts to the general training distribution. **Catastrophic decay** occurs during mid-training (steps 10k-30k), where STEM accuracy drops sharply (1.2% per 1,000 steps) as conversational gradient updates overwhelm STEM-critical parameters. After step 30k, STEM accuracy plateaus at the degraded level, and further training provides diminishing returns for both conversational and STEM performance.

## 3. Capability-Preserving Distillation (CPD)

### 3.1 Gradient Surgery

CPD's primary mechanism is **gradient surgery**: identifying conversational gradient updates that conflict with STEM gradient direction and projecting them away from the STEM gradient space.

For each training step, we compute the conversational gradient $g_{conv}$ and the estimated STEM gradient $g_{STEM}$ (estimated from a small held-out STEM mini-batch). We then project $g_{conv}$ orthogonal to $g_{STEM}$:

$$g_{conv}^{proj} = g_{conv} - \frac{g_{conv} \cdot g_{STEM}}{\|g_{STEM}\|^2} \cdot g_{STEM}$$

This projection removes the component of the conversational gradient that conflicts with STEM capabilities, while preserving the component that is compatible with STEM maintenance. The final update is:

$$g_{final} = g_{conv}^{proj} + \lambda_{STEM} \cdot g_{STEM}$$

where $\lambda_{STEM}$ controls the strength of the STEM preservation signal.

### 3.2 Capability-Aware Data Balancing

Standard distillation pipelines sample training examples proportionally to their frequency in the dataset, which often under-represents STEM content. CPD implements **capability-aware sampling** that ensures STEM examples are sufficiently represented:

$$p_{STEM}(x) = \max\left(p_{dataset}(x), \frac{n_{STEM}}{n_{total}} \cdot (1 + \gamma \cdot d_{STEM}(x))\right)$$

where $n_{STEM}$ is the target number of STEM examples per batch, $d_{STEM}(x)$ is the STEM difficulty score of example $x$, and $\gamma$ controls the oversampling of hard STEM examples. This ensures that even in general conversational corpora, STEM examples are sufficiently represented to prevent capability degradation.

### 3.3 Elastic Weight Consolidation for STEM Parameters

We apply Elastic Weight Consolidation (EWC) specifically to the STEM-critical parameters identified by Fisher Information analysis. The EWC loss penalizes changes to parameters that are important for STEM capabilities:

$$\mathcal{L}_{EWC} = \frac{\lambda_{EWC}}{2} \sum_{i \in \text{STEM-critical}} F_i \cdot (\theta_i - \theta_i^{STEM})^2$$

where $F_i$ is the Fisher Information of parameter $i$ for STEM tasks, $\theta_i$ is the current parameter value, and $\theta_i^{STEM}$ is the parameter value when STEM performance was at its peak (typically from the pre-distillation checkpoint).

The key innovation is that EWC is applied only to STEM-critical parameters (the top 5% by Fisher Information), not to all parameters. This selective application avoids the over-regularization that occurs when EWC is applied globally, which can prevent the student from learning any new capabilities.

### 3.4 Gradient Clipping for STEM-Critical Parameters

In addition to gradient surgery, CPD applies **gradient clipping** specifically to STEM-critical parameters:

$$g_i^{clipped} = \begin{cases} g_i & \text{if } |g_i| \leq \tau_{STEM} \\ \tau_{STEM} \cdot \text{sign}(g_i) & \text{if } |g_i| > \tau_{STEM} \end{cases}$$

for all $i \in \text{STEM-critical}$. The clipping threshold $\tau_{STEM}$ is set to 50% of the maximum gradient magnitude observed during STEM-only training, preventing large conversational gradient updates from overwhelming the STEM-critical parameters.

### 3.5 Capability Monitoring

CPD includes a **capability monitoring** system that tracks STEM performance throughout training. Every 500 steps, CPD evaluates the student on a held-out STEM validation set. If STEM accuracy drops by more than 2% from the peak, CPD triggers a **capability recovery phase** that temporarily increases $\lambda_{STEM}$ and $\lambda_{EWC}$ to restore STEM performance before resuming normal training.

This monitoring-and-recovery mechanism ensures that CPD maintains STEM capabilities even when the training dynamics are unpredictable. The recovery phase typically requires 500-1,000 steps to restore STEM accuracy to within 1% of the peak.

## 4. Experiments

### 4.1 Setup

We distill a 7B student from a multi-teacher ensemble (GPT-5.6 Sol, Claude Fable 5, DeepSeek V4 Pro, Qwen 3.8-Max) using 50,000 general conversational prompts and 10,000 STEM-specific prompts. Training uses 60,000 steps with the AdamW optimizer.

### 4.2 STEM Benchmarks

- Math-500, MATH (competition-level), GSM8K (grade school math)
- HumanEval+, MBPP+ (code generation)
- MMLU (STEM subset: physics, chemistry, biology, computer science)
- ARC-Challenge, GPQA (graduate-level STEM reasoning)
- MiniF2F (formal math verification)

### 4.3 General Benchmarks

- MMLU (non-STEM subsets), HellaSwag, WinoGrande
- MT-Bench, AlpacaEval 2.0 (conversational quality)
- TruthfulQA (factual accuracy)

### 4.4 Baselines

1. **Standard KD:** KL divergence distillation on general corpus.
2. **Balanced Sampling:** Equal sampling of general and STEM examples.
3. **DA-KD:** Difficulty-Aware Knowledge Distillation (ICML 2025).
4. **EWC-Global:** Elastic Weight Consolidation applied to all parameters.
5. **CPD (ours):** Capability-Preserving Distillation.

### 4.5 Results

**STEM Performance Preservation:**

| Method | Math-500 | HumanEval+ | MATH | GPQA | Avg STEM |
|--------|----------|------------|------|------|----------|
| Pre-Distillation | 82.3% | 71.8% | 48.2% | 31.4% | 58.4% |
| Standard KD | 75.0% (-7.3%) | 63.7% (-8.1%) | 38.8% (-9.4%) | 24.1% (-7.3%) | 50.4% (-8.0%) |
| Balanced | 78.2% (-4.1%) | 67.3% (-4.5%) | 42.1% (-6.1%) | 27.8% (-3.6%) | 53.9% (-4.5%) |
| DA-KD | 79.1% (-3.2%) | 68.4% (-3.4%) | 43.7% (-4.5%) | 28.9% (-2.5%) | 55.0% (-3.4%) |
| EWC-Global | 80.4% (-1.9%) | 69.7% (-2.1%) | 45.3% (-2.9%) | 30.1% (-1.3%) | 56.4% (-2.0%) |
| **CPD** | **81.6% (-0.7%)** | **71.1% (-0.7%)** | **47.8% (-0.4%)** | **31.0% (-0.4%)** | **57.9% (-0.5%)** |

CPD achieves 0.5% average STEM degradation, compared to 8.0% for standard KD and 2.0% for EWC-Global.

**General Performance:**

| Method | MMLU (non-STEM) | MT-Bench | AlpacaEval | Avg General |
|--------|-----------------|----------|------------|-------------|
| Pre-Distillation | 68.4% | 7.2 | 62.3% | 79.3% |
| Standard KD | 73.1% (+4.7%) | 8.1 (+0.9) | 71.2% (+8.9%) | 84.8% (+5.5%) |
| CPD | 74.3% (+5.9%) | 8.3 (+1.1) | 73.1% (+10.8%) | 86.2% (+6.9%) |

CPD not only preserves STEM capabilities but also achieves *better* general performance than standard KD, likely because the gradient surgery mechanism removes noisy conversational gradients that interfere with general learning.

## 5. Analysis

### 5.1 Gradient Conflict Reduction

CPD reduces the average gradient conflict ratio from 0.67 (standard KD) to 0.23, a 66% reduction. This dramatic reduction in conflict is achieved primarily through gradient surgery, which projects away the conflicting component of conversational gradients.

### 5.2 STEM Parameter Stability

We measure the L2 distance of STEM-critical parameters from their pre-distillation values. Standard KD moves these parameters by an average of 4.7 standard deviations. CPD limits movement to 0.8 standard deviations, confirming that EWC and gradient clipping effectively protect STEM-critical parameters.

### 5.3 Capability Recovery Dynamics

When STEM accuracy drops below the 2% threshold and triggers the recovery phase, CPD typically restores STEM performance to within 1% of the peak in 750 steps (averaged across 47 recovery events). The recovery is faster for mathematical reasoning (650 steps) than for code generation (850 steps), suggesting that mathematical capabilities are more concentrated in specific parameters that are easier to restore.

### 5.4 Ablation Study

| Component | STEM Degradation | General Improvement |
|-----------|-----------------|---------------------|
| Full CPD | -0.5% | +6.9% |
| Without Gradient Surgery | -2.8% | +5.2% |
| Without EWC | -1.9% | +6.4% |
| Without Capability Monitoring | -1.4% | +6.7% |
| Without Data Balancing | -3.1% | +5.8% |

Each component contributes meaningfully, with data balancing and gradient surgery being the most impactful for STEM preservation.

## 6. Connection to Prior Work

CPD builds on several established techniques. Gradient surgery was introduced for multi-task learning (Yu et al., 2020) and adapted for knowledge distillation by the Distribution-Decomposed KD framework (ACM, April 2026). EWC was originally proposed for continual learning (Kirkpatrick et al., 2017) and applied to distillation by several recent works. Capability-aware sampling extends the difficulty-aware sampling of DA-KD (ICML 2025) with explicit capability region identification.

The novelty of CPD lies in the combination of these techniques specifically targeting STEM capability preservation, with the capability monitoring and recovery mechanism that provides closed-loop control over the distillation process.

## 7. Limitations

CPD requires a small set of STEM-specific training examples (we use 10,000) for gradient estimation and capability monitoring. For domains where such examples are unavailable, the STEM gradient must be approximated from the general training data, reducing the effectiveness of gradient surgery.

Additionally, CPD's capability monitoring requires periodic evaluation on STEM benchmarks, adding 5% to total training time. This overhead is modest but non-negligible for very large training runs.

Finally, CPD is designed for capability preservation, not capability enhancement. It prevents STEM accuracy from degrading but does not actively improve it. Combining CPD with STEM-focused data augmentation could enable simultaneous capability preservation and enhancement.

## 8. Conclusion

Reasoning decay is a pervasive failure mode in knowledge distillation that reduces STEM accuracy by 5-12% in student models optimized for general conversational quality. Our Capability-Preserving Distillation framework addresses this through gradient surgery, elastic weight consolidation on STEM-critical parameters, capability-aware data balancing, and a monitoring-and-recovery mechanism that provides closed-loop control.

CPD achieves zero-degradation performance: 99.1% preservation of teacher STEM capabilities while matching or exceeding general conversational quality. The key insight is that **STEM capabilities are concentrated in a small fraction of parameters (5%) that can be identified through Fisher Information analysis and protected through targeted regularization**. By focusing protection on these critical parameters rather than applying uniform regularization, CPD avoids the over-regularization that limits global EWC approaches while providing robust STEM preservation.

## References

1. Strong Teacher Not Needed? On Distillation in LLM Pretraining. arXiv 2605.23857, May 2026.
2. A Survey of On-Policy Distillation for Large Language Models. arXiv 2604.00626, May 2026.
3. DA-KD: Difficulty-Aware Knowledge Distillation. ICML 2025.
4. Distribution-Decomposed Knowledge Distillation. ACM, April 2026.
5. Chain-of-Thought Curriculum Distillation. ACM, December 2025.
6. On-Policy Distillation. Thinking Machines Lab, October 2025.
7. Scaling Knowledge Distillation of Large Language Models. NeurIPS 2025.
8. Decoupling Top-K Probabilities for Efficient LM Distillation. OpenReview, 2025.
9. Efficient Knowledge Distillation through Low-Rank Clone. OpenReview, 2025.
10. A Case Study on the John O'Bryan Mathematics Competition. arXiv 2606.31048, June 2026.
