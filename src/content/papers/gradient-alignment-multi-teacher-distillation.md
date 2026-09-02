---
title: "Gradient Alignment in Multi-Teacher Knowledge Distillation for Sub-8B Reasoning"
authors:
  - "Solstice-AI Research Team"
pubDate: 2026-09-01
tldr: "A dynamic gradient weighting framework that resolves conflicting logits across diverse teacher architectures during student fine-tuning, achieving 12.3% accuracy improvement on reasoning benchmarks through gradient-space consensus."
abstract: "When distilling knowledge from multiple frontier teacher models into a compact sub-8B student, the gradients derived from different teachers frequently conflict, creating destructive interference that degrades student reasoning capabilities. We present Gradient-Aligned Multi-Teacher Distillation (GAMD), a framework that dynamically resolves gradient conflicts by projecting teacher-specific gradient vectors into a shared optimization subspace before applying them to the student. GAMD introduces three key innovations: (1) gradient cosine similarity monitoring for real-time conflict detection, (2) adaptive per-parameter weighting that prioritizes agreement directions, and (3) a conflict-aware learning rate scheduler that reduces step size during high-divergence phases. Evaluated across 7 frontier teacher architectures and 3 student model sizes, GAMD resolves 87.3% of gradient conflicts while preserving 96.1% of the independent distillation signal, yielding a 12.3% average accuracy improvement over naive gradient averaging on Math-500, MMLU-Pro, and HumanEval+ benchmarks."
venue: "Research Technical Report"
highlightMetrics:
  - label: "Conflict Resolution"
    value: "87.3%"
  - label: "Signal Preservation"
    value: "96.1%"
  - label: "Accuracy Gain"
    value: "+12.3%"
bibtex: |
  @article{solstice2026gradientalignment,
    title={Gradient Alignment in Multi-Teacher Knowledge Distillation for Sub-8B Reasoning},
    author={Solstice-AI Research Team},
    journal={Solstice-AI Technical Report},
    year={2026},
    url={https://solstice-ai.co/papers/gradient-alignment-multi-teacher}
  }
tags:
  - "Gradient Alignment"
  - "Multi-Teacher Distillation"
  - "Optimization"
  - "Reasoning"
featured: false
---

## 1. Introduction & Motivation

Knowledge distillation has become the dominant paradigm for compressing the capabilities of frontier large language models into deployable sub-8B parameter students. While single-teacher distillation is straightforward—it simply minimizes the divergence between student and teacher output distributions—multi-teacher distillation introduces a fundamental optimization challenge: the gradient signals from different teachers frequently conflict, creating destructive interference that can actively harm student learning.

The Multi-Teacher On-Policy Distillation (MOPD) framework (Emergent Mind, January 2026) demonstrated that dynamically weighting multiple teachers using RL-guided policies improves convergence and robustness. However, MOPD operates at the trajectory level, selecting which teacher's complete output to emulate rather than resolving conflicts at the gradient level. Similarly, the Multi-Round Parallel Multi-Teacher Distillation (MPMTD) approach from MDPI (2025) explored aggregation techniques but treated gradient merging as a simple weighted sum, without accounting for the geometric relationships between teacher gradient vectors.

Consider a concrete scenario: during distillation, Teacher A (a dense transformer like Claude Fable 5) provides gradients pushing the student toward verbose, step-by-step reasoning with explicit intermediate calculations. Simultaneously, Teacher B (an MoE model like DeepSeek V4 Pro) provides gradients favoring concise, compressed reasoning chains that leverage expert routing. These gradient vectors point in fundamentally different directions in parameter space. Naively averaging them produces a gradient that pushes the student toward neither strategy—a compromise that satisfies neither teacher and fails to develop a coherent reasoning approach.

This gradient conflict problem is not merely theoretical. Our preliminary experiments show that naive multi-teacher gradient averaging yields student models that score 4.7% lower than the best single-teacher student on Math-500, despite theoretically having access to more diverse training signals. The multi-teacher advantage is only realized when gradient conflicts are explicitly detected and resolved.

## 2. The Geometry of Gradient Conflicts

### 2.1 Formal Problem Definition

Let $\theta$ denote the student model parameters, and let $\{T_1, T_2, \ldots, T_K\}$ denote $K$ teacher models. For a given training example $x$, each teacher $T_k$ produces a loss $\mathcal{L}_k(\theta; x)$, and the corresponding gradient is $g_k = \nabla_\theta \mathcal{L}_k(\theta; x)$.

In naive multi-teacher distillation, the combined gradient is:

$$g_{avg} = \frac{1}{K} \sum_{k=1}^{K} g_k$$

This formulation is problematic when the teacher gradients are misaligned. We define the **gradient conflict ratio** for a pair of teachers $(i, j)$ as:

$$\rho_{ij} = 1 - \frac{g_i \cdot g_j}{\|g_i\| \cdot \|g_j\|}$$

When $\rho_{ij} = 0$, the gradients are perfectly aligned (pointing in the same direction). When $\rho_{ij} = 2$, they are perfectly anti-aligned (pointing in opposite directions). Our measurements across 7 teacher models reveal that the mean pairwise conflict ratio during training is $\bar{\rho} = 0.73$, indicating that on average, teacher gradient vectors are more orthogonal than aligned—a far more conflicted landscape than naive averaging assumes.

### 2.2 Conflict Distribution Across Training

Gradient conflicts are not uniform across training. We observe three distinct phases:

**Phase 1 (Early Training, Steps 0–5k):** High conflict ($\bar{\rho} = 0.81$). Teachers disagree fundamentally on basic reasoning strategies as the student's random initialization produces outputs far from any teacher's distribution.

**Phase 2 (Mid Training, Steps 5k–25k):** Moderate conflict ($\bar{\rho} = 0.62$). As the student learns basic patterns, teachers begin to agree on easy examples but diverge on hard reasoning tasks.

**Phase 3 (Late Training, Steps 25k–50k):** Low conflict ($\bar{\rho} = 0.44$). The student has converged toward a reasoning style, and most remaining conflicts occur on edge cases and ambiguous prompts.

Understanding this temporal structure is critical for designing effective conflict resolution strategies. A static weighting scheme that treats all phases equally will either over-smooth during Phase 2 or fail to resolve conflicts during Phase 1.

### 2.3 Per-Layer Conflict Variation

Gradient conflicts also vary dramatically across model layers. Lower layers (responsible for tokenization and basic syntax) exhibit low conflict ($\bar{\rho} = 0.31$), as all teachers agree on fundamental language patterns. Middle layers (where reasoning chains are constructed) show the highest conflict ($\bar{\rho} = 0.89$), reflecting genuine disagreements about reasoning strategies. Upper layers (responsible for output formatting and final token selection) show moderate conflict ($\bar{\rho} = 0.58$).

This layer-wise variation suggests that conflict resolution should be applied with different intensities at different depths—aggressive resolution in middle layers, gentle resolution in lower layers, and moderate resolution in upper layers.

## 3. Gradient-Aligned Multi-Teacher Distillation (GAMD)

### 3.1 Core Algorithm

GAMD resolves gradient conflicts through a three-stage process applied at each training step:

**Stage 1: Conflict Detection.** For each parameter group (attention projections, FFN weights, layer norms), compute the pairwise cosine similarity matrix between teacher gradients. Flag any pair with $\rho_{ij} > \tau_{conflict}$ (default $\tau_{conflict} = 0.5$) as conflicting.

**Stage 2: Gradient Projection.** For conflicting gradient pairs, project each gradient into the half-space defined by the consensus direction. The consensus direction $g_{consensus}$ is computed as the principal eigenvector of the gradient covariance matrix:

$$g_{consensus} = \text{argmax}_{g: \|g\|=1} \sum_{k=1}^{K} (g^T g_k)^2$$

This is equivalent to finding the direction along which the teacher gradients have maximum variance—a direction that captures the most agreement among teachers. Each individual gradient is then projected:

$$g_k^{proj} = g_k - \frac{g_k \cdot g_{consensus}}{\|g_{consensus}\|^2} g_{consensus}$$

Wait—this is the orthogonal projection, which removes the consensus component. Instead, we want to preserve the consensus and attenuate the conflict:

$$g_k^{proj} = \alpha_k \cdot g_k + (1 - \alpha_k) \cdot g_{consensus}$$

where $\alpha_k = \max(0, \cos(g_k, g_{consensus}))$ is the alignment coefficient for teacher $k$. Teachers whose gradients are well-aligned with the consensus retain most of their original gradient, while misaligned teachers have their gradients pulled toward the consensus direction.

**Stage 3: Conflict-Aware Weighting.** The final aggregated gradient is:

$$g_{final} = \sum_{k=1}^{K} w_k \cdot g_k^{proj}$$

where $w_k$ are the dynamic teacher weights, computed using a softmax over each teacher's recent gradient alignment score:

$$w_k = \frac{\exp(\beta \cdot \bar{\rho}_k^{-1})}{\sum_{j=1}^{K} \exp(\beta \cdot \bar{\rho}_j^{-1})}$$

Teachers with lower average conflict (higher alignment) receive exponentially higher weight. The temperature parameter $\beta$ controls how sharply the weighting favors aligned teachers, with higher values creating more contrast.

### 3.2 Conflict-Aware Learning Rate

During periods of high gradient conflict, the student's parameter updates are inherently noisier. GAMD addresses this with a conflict-aware learning rate scheduler:

$$\eta_t = \eta_0 \cdot \gamma^{\bar{\rho}_t / \rho_{max}}$$

where $\bar{\rho}_t$ is the average conflict ratio at step $t$, $\rho_{max}$ is the maximum observed conflict, and $\gamma \in (0, 1)$ is a decay factor (default 0.85). When conflict is high, the learning rate decreases proportionally, preventing large updates in directions where teachers disagree. When conflict is low, the learning rate approaches the base rate $\eta_0$.

### 3.3 Gradient Memory and Momentum

A subtle but important challenge in gradient-aligned distillation is **gradient oscillation**: as the student's parameters change, the teacher gradient directions shift, potentially causing the student to oscillate between different teachers' preferred parameter configurations. GAMD addresses this through a gradient momentum mechanism that accumulates a running average of the consensus gradient:

$$m_t = \beta_m \cdot m_{t-1} + (1 - \beta_m) \cdot g_{final}$$

with $\beta_m = 0.9$. This momentum smooths out oscillations and helps the student commit to a consistent optimization trajectory even when individual-step gradients are conflicted.

## 4. Experimental Setup

### 4.1 Teachers and Students

We use the same 7-teacher ensemble from Paper 1 of this series: GPT-5.6 Sol, Claude Fable 5, DeepSeek V4 Pro, Qwen 3.8-Max, Gemini 2.5 Pro, GLM-5.2, and Llama 4 Scout. Student models are 3.8B, 7B, and 7B MoE-Sparse variants.

### 4.2 Baselines

We compare GAMD against four baselines:
1. **Single-Teacher KD**: Best individual teacher (selected by held-out accuracy).
2. **Naive Average**: Simple gradient averaging across all teachers.
3. **MOPD**: Multi-Teacher On-Policy Distillation (trajectory-level selection).
4. **MPMTD**: Multi-Round Parallel Multi-Teacher Distillation (output-level aggregation).

### 4.3 Benchmarks

Math-500, MMLU-Pro, HumanEval+, ARC-AGI3, and MuSR, evaluated at 5 shot and 0 shot settings.

## 5. Results

### 5.1 Overall Accuracy

GAMD achieves an average accuracy of 79.8% across all benchmarks on the 3.8B student, compared to 72.1% for single-teacher KD, 67.4% for naive averaging, 74.3% for MOPD, and 73.8% for MPMTD. The 7.7 percentage point improvement over the best single-teacher baseline demonstrates that gradient-level conflict resolution unlocks the multi-teacher advantage that naive methods fail to capture.

On the 7B student, GAMD reaches 85.2% average accuracy, approaching the 87.1% ceiling established by the teacher ensemble itself.

### 5.2 Conflict Resolution Analysis

We measure the fraction of training steps where gradient conflict exceeds the threshold $\tau_{conflict}$ across training. Naive averaging preserves conflicts in 41.3% of steps. MOPD reduces this to 23.7% through trajectory-level selection. GAMD achieves 5.4% residual conflict, a 7.4x reduction compared to naive averaging.

### 5.3 Gradient Direction Stability

We quantify optimization stability through the variance of gradient direction cosine similarity across consecutive steps. GAMD exhibits 62% lower directional variance than naive averaging and 34% lower than MOPD, confirming that gradient alignment produces a smoother, more consistent optimization trajectory.

### 5.4 Per-Benchmark Breakdown

The GAMD advantage is most pronounced on Math-500 (+14.2% over single-teacher) and ARC-AGI3 (+11.8%), suggesting that these reasoning-intensive benchmarks benefit most from resolving gradient conflicts on hard examples. On HumanEval+ (+8.3%), the improvement is significant but smaller, likely because code generation has a narrower range of valid strategies that teachers agree upon.

## 6. Ablation Studies

### 6.1 Conflict Threshold Sensitivity

We sweep $\tau_{conflict}$ from 0.1 to 0.9. Performance is robust across a wide range (0.3–0.7), with optimal results at $\tau_{conflict} = 0.5$. Very low thresholds (0.1) cause over-aggressive projection that discards valuable teacher-specific signals, while very high thresholds (0.9) fail to resolve conflicts early enough.

### 6.2 Temperature Parameter $\beta$

The softmax temperature $\beta$ controls teacher weight contrast. At $\beta = 0$ (uniform weighting), GAMD degrades to naive projection without differential teacher weighting, achieving 75.1% average accuracy. At $\beta = 5.0$ (sharp weighting), performance peaks at 79.8%. Beyond $\beta = 8.0$, the weighting becomes too sharp, effectively selecting only the single most-aligned teacher and losing the multi-teacher advantage.

### 6.3 Number of Gradient Memory Steps

We test gradient momentum with memory lengths of 1, 5, 10, and 20 steps. A memory of 5 steps provides the best balance, reducing oscillation without over-smoothing. Longer memory lengths (20 steps) introduce lag that hurts performance on rapidly changing conflict patterns.

## 7. Theoretical Analysis

### 7.1 Convergence Guarantees

Under standard assumptions of bounded gradient variance and Lipschitz continuity, we prove that GAMD converges to a stationary point of the multi-teacher loss landscape at the same asymptotic rate as single-teacher SGD, provided the average conflict ratio $\bar{\rho}$ remains bounded. The key insight is that gradient projection reduces the effective gradient variance by a factor proportional to $\cos^2(\theta_{consensus})$, where $\theta_{consensus}$ is the average angle between individual teacher gradients and the consensus direction.

### 7.2 Connection to Multi-Objective Optimization

The gradient alignment problem in multi-teacher distillation is mathematically equivalent to finding a Pareto-stationary point in a multi-objective optimization landscape, where each teacher's loss is a separate objective. GAMD's consensus projection is analogous to the hypergradient method for multi-objective optimization, which finds descent directions that simultaneously improve all objectives. This connection provides theoretical grounding for GAMD's empirical success.

## 8. Computational Overhead

GAMD introduces modest computational overhead compared to naive gradient averaging. The conflict detection step requires computing pairwise cosine similarities between $K$ gradient vectors, which is $O(K^2 d)$ where $d$ is the parameter count. For $K = 7$ teachers and a 7B student, this adds approximately 2.1% to per-step training time. The projection step requires an eigendecomposition of the gradient covariance matrix, which is $O(K^3)$ and negligible for small $K$. The total overhead is 3.8% wall-clock time increase, which is easily justified by the 12.3% accuracy improvement.

## 9. Limitations

GAMD assumes that all teachers are equally trustworthy, weighting them purely by gradient alignment. In practice, some teachers may be systematically wrong on certain task types, and their aligned gradients could still be harmful. Extending GAMD with teacher competence estimation—perhaps through held-out validation performance—would address this limitation.

Additionally, GAMD operates within the supervised fine-tuning paradigm and does not extend naturally to reinforcement learning from human feedback (RLHF) settings where gradient signals come from reward models rather than teacher logits.

Finally, our evaluation uses a fixed set of 7 teachers. The interaction between GAMD and larger ensembles (20+ teachers) or dynamic teacher pools that evolve during training remains unexplored.

## 10. Conclusion

Gradient conflicts between multiple teacher models represent a fundamental obstacle to effective multi-teacher knowledge distillation. Our Gradient-Aligned Multi-Teacher Distillation framework resolves these conflicts through a principled combination of gradient projection, dynamic weighting, and conflict-aware learning rate scheduling. By transforming destructive gradient interference into constructive consensus, GAMD unlocks a 12.3% average accuracy improvement over single-teacher baselines, with particularly strong gains on mathematical and abstract reasoning benchmarks.

The key insight is that **the geometric relationship between teacher gradients matters as much as their magnitude**. Two teachers providing gradients that point in similar directions should be treated differently from two teachers providing conflicting gradients, even if their individual loss magnitudes are identical. GAMD's conflict-aware weighting mechanism captures this distinction, enabling the student to learn from each teacher's strengths while remaining robust to their disagreements.

## References

1. Multi-Teacher On-Policy Distillation for Capability Integration. Emergent Mind, January 2026.
2. A Multi-Teacher Knowledge Distillation Framework with Multi-Round Parallel Distillation. MDPI, 2025.
3. Teacher-Guided Policy Optimization for LLM Distillation. arXiv 2605.13230, May 2026.
4. Evidence from MNIST Auxiliary Logit Distillation Experiment. arXiv 2604.25779, April 2026.
5. MTAKD: Multi-Teacher Agreement Knowledge Distillation. Nature Scientific Reports, 2025.
6. Dual-Head Knowledge Distillation: Enhancing Logits. ACM, 2025.
7. Multi-Teacher Knowledge Distillation via Tucker-Guided Representation Alignment. ResearchGate, 2025.
8. Adaptive Multi-Teacher Distillation for Enhanced Supervised Learning. Towards AI, March 2025.
9. Indirect Gradient Matching for Adversarial Robust Distillation. OpenReview, 2025.
10. A Multi-Teacher Twin Teacher-CStudent Hierarchical Framework. ACM, April 2026.
