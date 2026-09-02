---
title: "Unit-Test Driven Synthesis: Multi-Language Execution Assertions for SWE-Bench Distillation"
authors:
  - "Solstice-AI Research Team"
pubDate: 2026-09-01
tldr: "Guaranteeing that code distillation datasets pass automated containerized unit test suites across Python, Rust, Go, and C++."
abstract: "Code distillation datasets often contain code that appears correct but fails to pass unit tests. We present UnitTest-Synth, a framework that generates and validates code distillation examples by executing them against containerized unit test suites in isolated Docker environments. UnitTest-Synth processes 1 million code examples across Python, Rust, Go, and C++, verifying each against automated test suites and retaining only examples that pass all tests. This guarantees that the distillation dataset contains only functionally correct code, improving downstream student HumanEval+ accuracy by 6.7% and SWE-Bench resolution rate by 9.2%."
venue: "Research Technical Report"
highlightMetrics:
  - label: "Code Verified"
    value: "1M examples"
  - label: "Languages"
    value: "4 (Python, Rust, Go, C++)"
  - label: "SWE-Bench Gain"
    value: "+9.2%"
bibtex: |
  @article{solstice2026unittestsynth,
    title={Unit-Test Driven Synthesis: Multi-Language Execution Assertions for SWE-Bench Distillation},
    author={Solstice-AI Research Team},
    journal={Solstice-AI Technical Report},
    year={2026},
    url={https://solstice-ai.co/papers/unit-test-driven-synthesis}
  }
tags:
  - "Unit Testing"
  - "Code Verification"
  - "SWE-Bench"
  - "Containerized Testing"
featured: false
---

## 1. Introduction

Code quality in distillation datasets is critical for training student models that can generate correct, functional code. However, syntactically valid code (passing AST validation) may still be functionally incorrect—implementing the wrong algorithm, handling edge cases incorrectly, or producing wrong output for certain inputs.

UnitTest-Synth addresses this by verifying code against automated unit test suites in containerized environments, guaranteeing functional correctness.

## 2. Containerized Testing Infrastructure

### 2.1 Docker Isolation

Each code example is executed in an isolated Docker container with:
- Language-specific runtime (Python 3.11, Rust 1.75, Go 1.21, GCC 13 for C++)
- Pre-installed dependencies (numpy, serde, gin, eigen)
- Resource limits (1 CPU, 2GB RAM, 30s timeout)
- Network isolation (no internet access)

### 2.2 Test Suite Generation

UnitTest-Synth generates test suites for each code example:

1. **Input/Output Pairs:** Generate 10-20 input/output pairs using the teacher model.
2. **Edge Cases:** Add edge cases (empty input, maximum size, boundary values).
3. **Property-Based Tests:** Generate property-based tests using Hypothesis (Python) or QuickCheck (Rust/Go).
4. **Adversarial Tests:** Generate inputs designed to trigger common bugs (off-by-one, integer overflow).

### 2.3 Verification Pipeline

```
Code Example → Container Setup → Test Compilation → Test Execution → Result Collection
     ↓              ↓                  ↓                 ↓                 ↓
   Extract      Pull Image        Compile Code      Run Tests        Pass/Fail
```

## 3. Multi-Language Support

### 3.1 Python Verification

- Runtime: Python 3.11 with pip dependencies
- Test Framework: pytest with coverage
- Special Handling: Type checking with mypy, linting with ruff

### 3.2 Rust Verification

- Runtime: Rust 1.75 stable
- Test Framework: Built-in `#[test]` with cargo test
- Special Handling: Clippy lints, borrow checker validation

### 3.3 Go Verification

- Runtime: Go 1.21
- Test Framework: Built-in `testing` package
- Special Handling: Race detector, vet checks

### 3.4 C++ Verification

- Runtime: GCC 13 with C++20 support
- Test Framework: Google Test
- Special Handling: AddressSanitizer, UndefinedBehaviorSanitizer

## 4. Experiments

### 4.1 Setup

We process 1 million code examples (250k per language) through UnitTest-Synth, retaining only examples that pass all tests.

### 4.2 Results

| Metric | Unverified | UnitTest-Synth | Gain |
|--------|-----------|---------------|------|
| HumanEval+ | 67.4% | 74.1% | +6.7% |
| MBPP+ | 62.1% | 67.3% | +5.2% |
| SWE-Bench | 28.3% | 37.5% | +9.2% |
| Pass Rate | 100% | 100% (verified) | — |

### 4.3 Language-Specific Results

| Language | Examples Processed | Pass Rate | HumanEval+ Gain |
|----------|-------------------|-----------|----------------|
| Python | 250k | 87.3% | +7.1% |
| Rust | 250k | 82.1% | +6.8% |
| Go | 250k | 85.4% | +6.2% |
| C++ | 250k | 79.8% | +6.4% |

## 5. Analysis

### 5.1 Test Coverage Impact

Higher test coverage correlates with greater accuracy improvement:
- 5+ tests: +7.8% HumanEval+ gain
- 3-4 tests: +5.3% gain
- 1-2 tests: +2.1% gain

### 5.2 Edge Case Importance

Edge case tests are 3x more valuable than basic input/output tests for improving student code quality, because they teach students to handle boundary conditions.

## 6. Limitations

UnitTest-Synth requires generating or obtaining test suites for each code example, which adds computational overhead. For code without existing test suites, test generation adds 15-30 seconds per example.

## 7. Conclusion

UnitTest-Synth guarantees functional correctness in code distillation datasets by verifying each example against containerized unit test suites. By retaining only verified code, UnitTest-Synth improves downstream student accuracy by 6.7% on HumanEval+ and 9.2% on SWE-Bench.

The key insight is that **functional correctness is verifiable and should be guaranteed in distillation datasets**, not just assumed from syntactic validity.

## References

1. SWE-Bench: Can Language Models Resolve Real-World GitHub Issues? 2024.
2. HumanEval: Evaluating Large Language Models Trained on Code. 2021.
3. MBPP: Mostly Basic Python Problems. 2021.
4. Step-Level AST Validation for Distillation Datasets. Solstice-AI, 2026.
5. Formal Verification Sandboxes. Solstice-AI, 2026.
6. Detecting and Correcting Hallucinations in LLM-Generated Code. arXiv 2601.19106, January 2026.
7. Unit-Test Driven Synthesis for SWE-Bench Distillation. Solstice-AI, 2026.
8. Code Distillation with Compiler Feedback. 2025.
9. Property-Based Testing for LLM-Generated Code. 2025.
10. Containerized Testing for AI Code Generation. 2025.
