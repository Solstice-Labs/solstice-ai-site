---
title: "Reinforcement Learning from Compiler Feedback (RLCF): Direct Policy Optimization on Verified Traces"
authors:
  - "Solstice-AI Research Team"
pubDate: 2026-09-01
tldr: "Replacing human reward models with deterministic binary compiler return codes and runtime memory profiles for student model alignment."
abstract: "Reinforcement Learning from Human Feedback (RLHF) requires expensive human annotation for reward model training. We present RLCF, a framework that replaces human reward models with deterministic compiler feedback: binary pass/fail signals from code compilation, test execution, and formal proof verification. RLCF trains student models using Direct Policy Optimization (DPO) with compiler-verified rewards, achieving 8.7% higher code accuracy and 5.3% higher math accuracy compared to standard SFT, without any human annotation."
venue: "Research Technical Report"
highlightMetrics:
  - label: "Code Accuracy Gain"
    value: "+8.7%"
  - label: "Math Accuracy Gain"
    value: "+5.3%"
  - label: "Human Annotation"
    value: "0%"
bibtex: |
  @article{solstice2026rlcf,
    title={Reinforcement Learning from Compiler Feedback: Direct Policy Optimization on Verified Traces},
    author={Solstice-AI Research Team},
    journal={Solstice-AI Technical Report},
    year={2026},
    url={https://solstice-ai.co/papers/rlcf-compiler-feedback}
  }
tags:
  - "RLCF"
  - "Compiler Feedback"
  - "Direct Policy Optimization"
  - "Reward Model"
featured: false
---

## 1. Introduction

RLHF requires training a reward model on human preferences, which is expensive, slow, and inconsistent. RLCF replaces the human reward model with deterministic compiler feedback—binary signals that indicate whether code compiles, tests pass, or proofs verify.

The key advantage of compiler feedback is that it is:
- **Deterministic:** The same code always produces the same result.
- **Scalable:** Millions of code examples can be verified automatically.
- **Noiseless:** No human subjectivity or annotation errors.

## 2. RLCF Framework

### 2.1 Reward Signal Design

RLCF defines rewards based on compiler/test outcomes:

| Outcome | Reward |
|---------|--------|
| Code compiles + tests pass | +1.0 |
| Code compiles + some tests pass | +0.5 |
| Code compiles + no tests pass | 0.0 |
| Code fails to compile | -0.5 |
| Proof verified (Lean 4) | +1.0 |
| Proof rejected | -0.5 |

### 2.2 DPO Training

RLCF uses Direct Policy Optimization (DPO) to train the student model on compiler-verified rewards:

1. **Generate:** Sample multiple completions for each prompt.
2. **Verify:** Score each completion using compiler/test feedback.
3. **Pair:** Construct preference pairs (preferred = passes tests, rejected = fails tests).
4. **Optimize:** Train the student model using DPO loss on preference pairs.

### 2.3 Multi-Signal Reward

RLCF combines multiple feedback signals into a composite reward:

$$R = w_1 \cdot R_{compile} + w_2 \cdot R_{test} + w_3 \cdot R_{memory} + w_4 \cdot R_{proof}$$

where $R_{compile}$ is compilation success, $R_{test}$ is test pass rate, $R_{memory}$ is runtime memory efficiency, and $R_{proof}$ is formal verification success.

## 3. Experiments

### 3.1 Setup

We train 7B student models using RLCF on 100,000 code examples and 50,000 math proofs, comparing against standard SFT and RLHF.

### 3.2 Results

| Method | HumanEval+ | MATH | Training Cost |
|--------|-----------|------|--------------|
| SFT | 67.4% | 42.1% | 1x |
| RLHF | 72.1% | 45.3% | 10x (human annotation) |
| RLCF | 76.1% | 47.4% | 0.5x (automated) |

RLCF achieves 8.7% higher code accuracy and 5.3% higher math accuracy than SFT, with 20x lower training cost than RLHF.

## 4. Analysis

### 4.1 Reward Signal Quality

RLCF's binary compiler rewards correlate 0.91 with human quality ratings, confirming that compiler feedback is a reliable proxy for code quality.

### 4.2 DPO vs PPO

DPO outperforms PPO for RLCF because the binary reward signal is better suited to pairwise preference learning than value-based reinforcement learning.

## 5. Conclusion

RLCF replaces expensive human annotation with deterministic compiler feedback, achieving superior alignment at 20x lower cost.

The key insight is that **compiler feedback is a scalable, noiseless alternative to human preferences** for code and math alignment.

## References

1. Training Language Models to Follow Instructions with Human Feedback. 2022.
2. Direct Preference Optimization. 2023.
3. Step-Level AST Validation. Solstice-AI, 2026.
4. Formal Verification Sandboxes. Solstice-AI, 2026.
5. Unit-Test Driven Synthesis. Solstice-AI, 2026.
6. Compiler Feedback for Code Generation. 2025.
7. DPO for Code Models. 2025.
8. Automated Reward Models for LLM Alignment. 2025.
9. RLHF vs RLAIF: A Comparison. 2025.
10. Deterministic Rewards for LLM Training. 2025.
