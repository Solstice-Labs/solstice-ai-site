---
title: "Backtracking and Recovery Traces: Training Students to Self-Correct via Execution Failures"
authors:
  - "Solstice-AI Research Team"
pubDate: 2026-09-01
tldr: "Injecting intentional compiler error logs and successful self-correction loops into synthetic datasets to teach student models autonomous debugging."
abstract: "Human programmers rarely write correct code on the first attempt—they iterate, debug, and recover from errors. Current distillation datasets present only polished, error-free reasoning chains, depriving students of the opportunity to learn self-correction. We present RecoveryTrace, a framework that injects intentional execution failures (compiler errors, runtime exceptions, failed test cases) into synthetic reasoning traces, followed by successful self-correction sequences. RecoveryTrace generates 150,000 recovery traces across Python, Rust, and Go, teaching student models to detect errors, diagnose root causes, and generate corrections. Students trained on RecoveryTrace data achieve 8.3% higher bug-fix accuracy on SWE-Bench and 12.1% higher self-correction success rate on a custom debugging benchmark."
venue: "Research Technical Report"
highlightMetrics:
  - label: "Bug-Fix Accuracy"
    value: "+8.3%"
  - label: "Self-Correction Rate"
    value: "+12.1%"
  - label: "Recovery Traces"
    value: "150k"
bibtex: |
  @article{solstice2026backtrackingrecovery,
    title={Backtracking and Recovery Traces: Training Students to Self-Correct via Execution Failures},
    author={Solstice-AI Research Team},
    journal={Solstice-AI Technical Report},
    year={2026},
    url={https://solstice-ai.co/papers/backtracking-recovery-traces}
  }
tags:
  - "Self-Correction"
  - "Recovery Traces"
  - "Debugging"
  - "Execution Failures"
featured: false
---

## 1. Introduction

Expert programmers spend 35-50% of their time debugging—reading error messages, hypothesizing root causes, and generating fixes. This debugging skill is a critical component of programming expertise that is absent from standard distillation datasets, which present only the final, correct reasoning chain.

When student models encounter errors during inference, they lack the experience to diagnose and recover from failures. They either repeat the same error or abandon the task entirely. By training on recovery traces—reasoning chains that include intentional failures and successful corrections—students learn to treat errors as normal parts of the reasoning process rather than catastrophic failures.

## 2. The Recovery Trace Structure

### 2.1 Trace Anatomy

A recovery trace follows this structure:

```
1. Initial attempt: Generate code/solution
2. Execution: Run the code
3. Failure: Observe error (compiler/runtime/test failure)
4. Diagnosis: Analyze the error message
5. Hypothesis: Identify the root cause
6. Correction: Generate a fix
7. Re-execution: Run the corrected code
8. Success: Verify the fix works
```

### 2.2 Failure Types

RecoveryTrace injects three types of failures:

1. **Compiler Errors:** Syntax errors, type errors, undefined variables.
2. **Runtime Errors:** Division by zero, index out of bounds, null pointer dereference.
3. **Test Failures:** Code runs but produces incorrect output (detected by unit tests).

### 2.3 Correction Quality

RecoveryTrace ensures that all corrections are verified by re-executing the corrected code. If the correction does not fix the error, the trace is discarded. This guarantees that students learn from successful recovery patterns, not from failed correction attempts.

## 3. Trace Generation Pipeline

### 3.1 Error Injection

RecoveryTrace injects errors into existing correct reasoning traces using:

1. **Mutation:** Modify correct code to introduce specific error types (e.g., change `==` to `=` for comparison errors).
2. **Truncation:** Remove critical lines of code (e.g., missing import statements).
3. **Type Confusion:** Replace variables with incorrectly typed values.

### 3.2 Error Message Generation

For each injected error, RecoveryTrace generates a realistic error message by:
1. Running the mutated code through the actual compiler/interpreter.
2. Capturing the compiler's error message verbatim.
3. Formatting the error message to match the style of the reasoning trace.

### 3.3 Self-Correction Generation

RecoveryTrace generates self-correction sequences using the teacher model:

1. Present the error message to the teacher model.
2. Ask the teacher to diagnose the root cause.
3. Ask the teacher to generate a fix.
4. Verify the fix by re-executing the code.

## 4. Experiments

### 4.1 Setup

We generate 150,000 recovery traces (50k Python, 50k Rust, 50k Go) and train 7B student models on both standard and recovery-augmented datasets.

### 4.2 Results

| Benchmark | Standard KD | + RecoveryTrace | Gain |
|-----------|------------|----------------|------|
| SWE-Bench | 28.3% | 36.6% | +8.3% |
| HumanEval+ | 67.4% | 71.2% | +3.8% |
| DebugBench (custom) | 41.2% | 53.3% | +12.1% |
| MBPP+ | 62.1% | 64.8% | +2.7% |

### 4.3 Recovery Quality Analysis

| Metric | Student Trained on RecoveryTrace |
|--------|--------------------------------|
| Error Detection Rate | 89.3% |
| Root Cause Accuracy | 78.1% |
| First-Attempt Fix Rate | 72.4% |
| Multi-Try Recovery Rate | 91.7% |

## 5. Analysis

### 5.1 Failure Type Impact

| Failure Type | DebugBench Gain | Most Beneficial For |
|-------------|----------------|-------------------|
| Compiler Errors | +5.2% | Syntax-level bugs |
| Runtime Errors | +4.1% | Logic bugs |
| Test Failures | +2.8% | Algorithmic bugs |

Compiler error recovery provides the largest improvement because compiler errors are the most common and most structured failure type.

### 5.2 Trace Length Effect

Longer recovery traces (3+ correction attempts) provide 40% more debugging benefit than shorter traces (1 attempt), because they teach students to persist through multiple failure-correction cycles.

## 6. Limitations

RecoveryTrace requires executing code to generate error messages, which adds computational overhead. For code that cannot be executed (e.g., incomplete programs, pseudocode), RecoveryTrace falls back to LLM-predicted error messages, which are less accurate.

## 7. Conclusion

Standard distillation datasets present only error-free reasoning chains, depriving students of debugging experience. RecoveryTrace injects intentional failures and successful corrections into synthetic traces, teaching students to detect, diagnose, and recover from errors.

The key insight is that **debugging is a learnable skill that can be distilled from teacher models**, just like reasoning and code generation. By exposing students to recovery patterns, we improve their ability to handle real-world programming challenges.

## References

1. How Rust's Compiler Catches What Coding Agents Get Wrong. Marc Love Blog, December 2025.
2. Detecting and Correcting Hallucinations in LLM-Generated Code. arXiv 2601.19106, January 2026.
3. SWE-Bench: Can Language Models Resolve Real-World GitHub Issues? 2024.
4. DebugBench: A Benchmark for LLM Debugging. 2025.
5. Step-Level AST Validation. Solstice-AI, 2026.
6. Formal Verification Sandboxes. Solstice-AI, 2026.
7. Curriculum Distillation for Long-Horizon Agent Traces. Solstice-AI, 2026.
8. Unit-Test Driven Synthesis for SWE-Bench Distillation. Solstice-AI, 2026.
9. Code Distillation with Compiler Feedback. 2025.
10. Self-Correction in Large Language Models. 2025.
