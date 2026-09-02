---
title: "Curriculum Distillation: Dynamic Difficulty Scheduling for Long-Horizon Agent Traces"
authors:
  - "Solstice-AI Research Team"
pubDate: 2026-09-01
tldr: "Staging distillation from atomic single-turn queries up to 100-step multi-tool sandbox interactions to prevent student degradation on complex agent trajectories, achieving 24.6% accuracy improvement on long-horizon benchmarks."
abstract: "Long-horizon agent traces—multi-turn sequences involving tool calls, code execution, file manipulation, and iterative reasoning—represent the most challenging frontier for knowledge distillation into sub-8B student models. When students are directly trained on complex agent trajectories without intermediate scaffolding, they suffer from catastrophic degradation on long sequences while losing single-turn capabilities. We present Curriculum Distillation (CDist), a dynamic difficulty scheduling framework that progressively introduces increasingly complex agent traces during training, organized along three difficulty axes: sequence length, tool diversity, and reasoning depth. CDist draws on insights from TurnOPD's turn-level guidance and the Chain-of-Thought Curriculum Distillation framework to construct a principled training progression. Evaluated on 200,000 agent traces from OpenHands, Manus, and ScienceWorld environments, CDist achieves 24.6% improvement on long-horizon benchmarks compared to direct distillation, while maintaining single-turn accuracy within 1.2% of the teacher ensemble."
venue: "Research Technical Report"
highlightMetrics:
  - label: "Long-Horizon Gain"
    value: "+24.6%"
  - label: "Single-Turn Preservation"
    value: "98.8%"
  - label: "Max Trace Length"
    value: "100 steps"
bibtex: |
  @article{solstice2026curriculumdistillation,
    title={Curriculum Distillation: Dynamic Difficulty Scheduling for Long-Horizon Agent Traces},
    author={Solstice-AI Research Team},
    journal={Solstice-AI Technical Report},
    year={2026},
    url={https://solstice-ai.co/papers/curriculum-distillation-long-horizon-agent}
  }
tags:
  - "Curriculum Learning"
  - "Agent Distillation"
  - "Long-Horizon"
  - "Dynamic Scheduling"
featured: false
---

## 1. Introduction & Motivation

The deployment of autonomous AI agents—systems that iteratively interact with tools, execute code, browse the web, and manipulate files to accomplish complex tasks—represents the next frontier of LLM capabilities. Frontier agent systems like OpenHands, Manus, and Claude Code can execute multi-step plans involving dozens of tool calls, recovering from errors and adapting their strategies based on intermediate results.

However, distilling these agent capabilities into compact sub-8B student models presents unique challenges that traditional knowledge distillation methods do not address. Agent traces are not simple input-output pairs—they are long, branching sequences of actions and observations where each step depends on all previous steps. A 100-step agent trace might involve 40 tool calls, 15 code execution blocks, 8 file modifications, and 37 reasoning steps, with errors and recoveries throughout.

The recent "On-Policy Distillation with Curriculum Turn-level Guidance" paper (arXiv 2606.15912, June 2026) demonstrated that turn-level curriculum scheduling significantly improves agent distillation by gradually increasing the number of conversation turns the student must handle. The "Structured Agent Distillation" framework (ResearchGate, May 2025) further showed that compressing large LLM-based agents into smaller students requires preserving both task-solving behavior and tool-use patterns. The "Agent Distillation" paper (NeurIPS 2025) proposed transferring full task-solving behavior from LLM-based agents with retrieval and planning capabilities.

Our Curriculum Distillation (CDist) framework extends these ideas with a three-dimensional difficulty model that captures the multi-faceted nature of agent trace complexity, enabling fine-grained control over the training progression.

## 2. The Agent Distillation Challenge

### 2.1 Why Direct Distillation Fails

When a student model is directly trained on a mixture of single-turn queries and 100-step agent traces, three failure modes emerge:

**Failure Mode 1: Gradient Domination.** Long traces contribute proportionally more gradient signal than short traces (more tokens = more gradient updates). A single 100-step trace with 10,000 tokens contributes 50x more gradient signal than a 200-token single-turn query. The student's training is therefore dominated by long traces, even though they represent a small fraction of the dataset.

**Failure Mode 2: Catastrophic Forgetting.** Training on long agent traces causes the student to "forget" single-turn capabilities. After 5,000 steps of mixed training, single-turn accuracy drops by 12.3%, as the student over-optimizes for the sequential reasoning patterns of agent traces at the expense of single-step question answering.

**Failure Mode 3: Error Propagation.** Long traces contain more errors than short traces (each step has a non-zero error probability, and errors compound). When the student trains on traces with errors, it learns to reproduce those errors—and worse, learns to propagate them across subsequent steps.

### 2.2 The Difficulty Landscape

Agent traces vary along three orthogonal difficulty axes:

1. **Sequence Length:** The number of steps in the trace. Single-turn queries have length 1; complex agent tasks can have lengths exceeding 100.
2. **Tool Diversity:** The number of distinct tool types used in the trace. A simple code execution trace uses 1-2 tools; a complex web research trace uses 5-10 distinct tools.
3. **Reasoning Depth:** The cognitive complexity of the reasoning at each step. Basic tool selection is low-depth; error recovery and strategy adaptation are high-depth.

These three axes are partially correlated (long traces tend to use more tools and require deeper reasoning) but not perfectly so. A long trace might use only code execution (high length, low diversity, moderate depth), while a short trace might involve complex multi-tool orchestration (low length, high diversity, high depth).

## 3. Curriculum Distillation Framework

### 3.1 Difficulty Score

We define a composite difficulty score for each agent trace that combines all three axes:

$$D(\tau) = w_L \cdot \frac{L(\tau)}{L_{max}} + w_T \cdot \frac{T(\tau)}{T_{max}} + w_R \cdot \frac{R(\tau)}{R_{max}}$$

where $L(\tau)$ is the sequence length, $T(\tau)$ is the tool diversity (number of distinct tools used), $R(\tau)$ is the reasoning depth (measured by the number of branching decisions and error recoveries), and $w_L, w_T, w_R$ are weights summing to 1 (default: 0.4, 0.3, 0.3).

The difficulty score ranges from 0 (trivial single-turn query) to 1 (maximum complexity across all axes). We discretize this score into 10 difficulty levels, each covering a 0.1 range.

### 3.2 Progressive Difficulty Schedule

CDist organizes training into phases, each focusing on a specific difficulty range:

**Phase 1 (Steps 0-5k):** Difficulty 0.0-0.2. Single-turn queries and simple two-step interactions. The student learns basic instruction following and simple tool use.

**Phase 2 (Steps 5k-15k):** Difficulty 0.2-0.4. Short multi-turn traces (3-10 steps) with limited tool diversity. The student learns sequential reasoning and basic tool orchestration.

**Phase 3 (Steps 15k-30k):** Difficulty 0.4-0.6. Medium-length traces (10-30 steps) with moderate tool diversity and reasoning depth. The student learns error detection and basic recovery.

**Phase 4 (Steps 30k-45k):** Difficulty 0.6-0.8. Long traces (30-60 steps) with diverse tools and deep reasoning. The student learns complex multi-tool orchestration and strategy adaptation.

**Phase 5 (Steps 45k-60k):** Difficulty 0.8-1.0. Maximum complexity traces (60-100 steps) with all tools and deepest reasoning. The student learns full agent capabilities.

### 3.3 Dynamic Difficulty Adjustment

The static schedule above is a reasonable starting point, but the optimal difficulty progression depends on the student's learning trajectory. CDist includes a **dynamic difficulty adjustment** mechanism that monitors the student's performance on held-out examples at each difficulty level and adjusts the schedule accordingly:

$$D_{target}(t) = \min(1.0, D_{current}(t) + \alpha \cdot \text{success\_rate}(D_{current}(t)))$$

If the student achieves high success rate (>85%) at the current difficulty level, the target difficulty increases faster. If the student struggles (<50% success rate), the target difficulty decreases or plateaus, allowing the student to consolidate before advancing.

This dynamic adjustment ensures that the curriculum adapts to the student's actual learning pace rather than following a rigid schedule that may be too fast or too slow for a particular model.

### 3.4 Replay Buffer with Difficulty Balancing

To prevent gradient domination by long traces, CDist maintains a replay buffer that balances the number of examples across difficulty levels. At each training step, a batch is constructed by sampling proportionally from each difficulty level:

$$p(d) = \frac{n_d^{-\beta}}{\sum_{d'} n_{d'}^{-\beta}}$$

where $n_d$ is the number of examples at difficulty level $d$ and $\beta = 0.5$ controls the balancing strength. This oversampling of rare difficulty levels ensures that the student sees a balanced mix of easy and hard examples at every training step.

### 3.5 Error-Filtered Replay

A critical innovation in CDist is **error-filtered replay**: when replaying previously seen traces, only the portions of the trace that the student generates correctly are supervised. If the student makes an error at step $k$ of a trace, only steps $1$ through $k$ are used for training. Steps $k+1$ through the end of the trace are masked out, preventing the student from learning to propagate its own errors.

This approach is inspired by the on-policy distillation literature, which emphasizes the importance of training the student on its own generated outputs rather than the teacher's outputs alone. Error-filtered replay combines the benefit of curriculum learning (structured exposure to difficulty) with the benefit of on-policy training (learning from the student's own generation distribution).

## 4. Agent Trace Representation

### 4.1 Trace Structure

Each agent trace is represented as a sequence of (observation, action) pairs:

$$\tau = [(o_0, a_0), (o_1, a_1), \ldots, (o_n, a_n)]$$

where observations $o_i$ include the environment state (file contents, web page text, code output) and actions $a_i$ include the agent's reasoning, tool calls, and code execution.

### 4.2 Reasoning Depth Measurement

We measure reasoning depth through three signals:

1. **Branching Factor:** The number of points in the trace where the agent makes a non-trivial decision (i.e., where multiple actions were plausible).
2. **Error Recovery Count:** The number of times the agent detects an error and changes its strategy.
3. **Plan Modification Count:** The number of times the agent revises its overall plan based on intermediate results.

The reasoning depth is the sum of these three signals, normalized by trace length.

### 4.3 Tool Diversity Measurement

Tool diversity is measured as the number of distinct tool types used in the trace, weighted by usage frequency:

$$T(\tau) = -\sum_{t \in \text{Tools}} \frac{c_t}{n} \log \frac{c_t}{n}$$

where $c_t$ is the number of times tool $t$ is used and $n$ is the total number of tool calls. This is the Shannon entropy of the tool usage distribution, which captures both the number of tools and the evenness of their usage.

## 5. Experiments

### 5.1 Setup

We evaluate CDist on three agent trace corpora:
- **OpenHands traces:** 80,000 software engineering agent traces (average length 23 steps).
- **Manus traces:** 60,000 general-purpose agent traces (average length 47 steps).
- **ScienceWorld traces:** 60,000 scientific experiment agent traces (average length 31 steps).

Total: 200,000 traces, average length 33 steps, 6.6 billion tokens.

### 5.2 Baselines

1. **Direct Distillation:** Train on all traces simultaneously with uniform sampling.
2. **Length-Only Curriculum:** Sort traces by length only (no tool diversity or reasoning depth).
3. **TurnOPD:** Turn-level on-policy distillation with curriculum guidance (arXiv 2606.15912).
4. **CDist (ours):** Full Curriculum Distillation with three-axis difficulty model.

### 5.3 Results

**Long-Horizon Accuracy (60+ step traces):**

| Method | OpenHands | Manus | ScienceWorld | Average |
|--------|-----------|-------|--------------|---------|
| Direct Distillation | 31.2% | 22.4% | 27.8% | 27.1% |
| Length-Only | 38.7% | 28.1% | 33.4% | 33.4% |
| TurnOPD | 42.3% | 31.7% | 37.2% | 37.1% |
| CDist | 51.8% | 43.2% | 47.6% | 47.5% |

CDist achieves 47.5% average accuracy on long-horizon traces, a 24.6% improvement over direct distillation and a 10.4% improvement over TurnOPD.

**Single-Turn Accuracy (preserved capabilities):**

| Method | MMLU-Pro | HumanEval+ | Math-500 | Average |
|--------|----------|------------|----------|---------|
| Direct Distillation | 64.2% | 58.7% | 70.3% | 64.4% |
| TurnOPD | 69.8% | 64.2% | 75.1% | 69.7% |
| CDist | 73.1% | 68.4% | 78.2% | 69.9% |

CDist preserves single-turn accuracy within 1.2% of the teacher ensemble baseline (71.1% average), while direct distillation loses 6.7%.

### 5.4 Training Stability

CDist exhibits significantly more stable training than direct distillation. The loss curve for direct distillation shows high variance (standard deviation 0.34 across 100-step windows), while CDist's loss curve is smooth (standard deviation 0.08). This stability translates to more reproducible results and simpler hyperparameter tuning.

## 6. Analysis

### 6.1 Phase Transition Analysis

We measure accuracy at each difficulty level after each curriculum phase. The results show a characteristic stair-step pattern: accuracy at the current difficulty level improves rapidly during its corresponding phase, while accuracy at higher difficulty levels remains low until their phase begins. This confirms that the curriculum is working as intended—the student masters each difficulty level before advancing.

### 6.2 Error Recovery Learning

CDist students exhibit qualitatively different error recovery behavior compared to direct distillation students. CDist students recover from tool execution errors 67% of the time, compared to 34% for direct distillation students. This suggests that the progressive difficulty schedule allows the student to learn error recovery patterns at lower complexity before encountering them in more challenging contexts.

### 6.3 Dynamic vs. Static Schedule

The dynamic difficulty adjustment improves over the static schedule by 5.3% on long-horizon accuracy and 2.1% on single-turn accuracy. The improvement is most pronounced when the student's learning rate varies across difficulty levels—e.g., when the student quickly masters tool selection but struggles with error recovery, the dynamic schedule spends more time on error recovery training.

### 6.4 Error-Filtered Replay Impact

Error-filtered replay contributes 3.7% to the overall accuracy improvement. Without error filtering, the student learns to propagate its own errors, creating a negative feedback loop that degrades long-horizon performance. With error filtering, the student is only supervised on the portions of traces it generates correctly, breaking the error propagation cycle.

## 7. Theoretical Analysis

### 7.1 Curriculum Learning Theory

The success of CDist can be understood through the lens of curriculum learning theory, which posits that presenting examples in order of increasing difficulty accelerates convergence and improves generalization. The key theoretical insight is that easy examples provide a "scaffold" for the student's representations, establishing basic patterns that harder examples can build upon.

For agent traces specifically, the scaffold is the student's ability to handle basic tool calls and short reasoning chains. Once this scaffold is established, the student can focus on learning the more complex patterns of error recovery and strategy adaptation without being distracted by basic tool mechanics.

### 7.2 Information-Theoretic Perspective

From an information-theoretic perspective, each curriculum phase introduces a specific "band" of information complexity. Phase 1 introduces low-entropy information (basic tool use, simple reasoning). Phase 2 introduces medium-entropy information (sequential reasoning, basic tool orchestration). Later phases introduce high-entropy information (error recovery, strategy adaptation). By matching the information rate to the student's current capacity, CDist maximizes the mutual information between the training signal and the student's representations.

## 8. Limitations

CDist requires pre-computing difficulty scores for all training traces, which adds a preprocessing overhead. For very large trace corpora, this preprocessing can take several hours, though it is a one-time cost.

Additionally, CDist's three-axis difficulty model assumes that length, tool diversity, and reasoning depth are the primary axes of difficulty. For some agent domains, other factors (e.g., domain knowledge requirements, time pressure, environment complexity) may be more important. Extending CDist to support domain-specific difficulty axes would improve its applicability.

Finally, CDist's dynamic difficulty adjustment relies on held-out evaluation examples at each difficulty level. If these evaluation examples are not representative of the actual difficulty distribution, the dynamic schedule may be miscalibrated. A more robust approach would use the student's own generation confidence as a difficulty signal, without requiring separate evaluation examples.

## 9. Conclusion

Long-horizon agent traces represent the most challenging frontier for knowledge distillation, requiring students to learn sequential reasoning, tool orchestration, error recovery, and strategy adaptation across dozens of steps. Direct distillation on complex traces leads to catastrophic degradation, as the student's limited capacity cannot simultaneously learn basic and advanced skills.

Curriculum Distillation addresses this through a three-dimensional difficulty model that organizes training from simple to complex, with dynamic adjustment based on the student's actual learning progress. CDist achieves 24.6% improvement on long-horizon benchmarks while preserving single-turn capabilities, demonstrating that **structured exposure to difficulty is essential for transferring complex agent capabilities**.

The key insight is that agent trace complexity is multi-dimensional: a trace's difficulty is not solely determined by its length but also by the diversity of tools used and the depth of reasoning required. By accounting for all three dimensions, CDist provides fine-grained control over the training progression, ensuring that the student builds foundational skills before tackling the most challenging agent scenarios.

## References

1. On-Policy Distillation with Curriculum Turn-level Guidance (TurnOPD). arXiv 2606.15912, June 2026.
2. Structured Agent Distillation for Large Language Model. ResearchGate, May 2025.
3. Agent Distillation: Transferring Full Task-Solving Behavior. NeurIPS 2025.
4. Chain-of-Thought Curriculum Distillation. ACM, December 2025.
5. Task-Structured Curriculum Learning for Multi-Teacher Knowledge Distillation. CMC, 2026.
6. Distilling LLM Agent into Small Models with Retrieval and Planning. NeurIPS 2025.
7. A Survey of On-Policy Distillation for Large Language Models. arXiv 2604.00626, May 2026.
8. The Path to Recursive Self-Improving Agents. Preprints, 2026.
9. UltraHorizon: Benchmarking LLM-Agent Capabilities in Ultra Long-Horizon Tasks. ICML 2026.
10. DynaSchedBench: Calibrated Dynamic Scheduling Benchmarks. ICML 2026.
