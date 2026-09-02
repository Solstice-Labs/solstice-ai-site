---
title: "Formal Verification Sandboxes: Integrating Lean 4 and Isabelle into Synthetic CoT Generation"
authors:
  - "Solstice-AI Research Team"
pubDate: 2026-09-01
tldr: "Validating step-by-step mathematical theorems with formal interactive proof assistants to eliminate reasoning hallucinations in STEM corpora."
abstract: "Mathematical reasoning traces from LLMs often contain logical errors that are invisible to informal checking but detectable by formal proof assistants. We present FormalVerify, a sandboxed framework that integrates Lean 4 and Isabelle into the synthetic chain-of-thought generation pipeline, validating each mathematical step against a formal proof assistant before inclusion in the distillation dataset. FormalVerify processes 200,000 mathematical reasoning traces, detecting and removing 18.3% of hallucinated proofs (including 7.2% that pass informal verification), improving downstream student math accuracy by 5.8% on competition-level benchmarks."
venue: "Research Technical Report"
highlightMetrics:
  - label: "Hallucinations Detected"
    value: "18.3%"
  - label: "Invisible to Informal Check"
    value: "7.2%"
  - label: "Math Accuracy Gain"
    value: "+5.8%"
bibtex: |
  @article{solstice2026formalverification,
    title={Formal Verification Sandboxes: Integrating Lean 4 and Isabelle into Synthetic CoT Generation},
    author={Solstice-AI Research Team},
    journal={Solstice-AI Technical Report},
    year={2026},
    url={https://solstice-ai.co/papers/formal-verification-sandboxes}
  }
tags:
  - "Formal Verification"
  - "Lean 4"
  - "Isabelle"
  - "Mathematical Reasoning"
featured: false
---

## 1. Introduction

Mathematical reasoning traces are among the most valuable content in distillation datasets, but they are also among the most error-prone. Frontier LLMs generate plausible-looking mathematical proofs that contain subtle logical errors—wrong algebraic manipulations, invalid theorem applications, and circular reasoning—that are difficult to detect through informal checking.

Formal proof assistants like Lean 4 and Isabelle provide a deterministic mechanism for validating mathematical proofs: if a proof compiles in Lean 4, it is guaranteed to be correct. By integrating these proof assistants into the distillation pipeline, we can automatically validate mathematical reasoning steps and prune hallucinated proofs.

## 2. The Formal Verification Challenge

### 2.1 Informal vs. Formal Verification

Informal verification (checking by LLM or human) catches ~82% of mathematical errors. Formal verification catches ~99.7%. The 17.7% gap includes:

- **Subtle sign errors:** $-x^2$ vs. $(-x)^2$ (caught by formal verification, often missed informally).
- **Invalid theorem applications:** Using a theorem with incorrect preconditions.
- **Dimensional inconsistencies:** Mixing units or types in physics equations.
- **Division by zero:** Dividing by expressions that may be zero.

### 2.2 The Translation Problem

Mathematical reasoning in natural language must be translated to formal proof syntax before verification. This translation is non-trivial because natural language proofs use informal notation, implicit steps, and domain-specific conventions that don't map directly to formal syntax.

## 3. FormalVerify Framework

### 3.1 Lean 4 Integration

FormalVerify uses Lean 4 for algebra and number theory proofs:

1. **Step Extraction:** Parse the natural language proof into discrete mathematical steps.
2. **Translation:** Use a fine-tuned translation model to convert each step to Lean 4 syntax.
3. **Verification:** Submit the Lean 4 code to the Lean 4 type checker.
4. **Feedback Loop:** If verification fails, use the error message to identify and correct the problematic step.

### 3.2 Isabelle Integration

FormalVerify uses Isabelle/HOL for analysis and discrete mathematics proofs:

1. **Theory File Generation:** Generate an Isabelle theory file from the proof steps.
2. **Sledgehammer:** Use Isabelle's Sledgehammer tool to automatically find relevant lemmas.
3. **Proof Checking:** Submit the theory file for proof checking.
4. **Counterexample Detection:** Use Nitpick to find counterexamples to false claims.

### 3.3 Hybrid Verification Strategy

FormalVerify uses a hybrid strategy:

- **Lean 4:** For proofs involving algebra, number theory, and combinatorics (faster verification, broader coverage).
- **Isabelle:** For proofs involving real analysis, topology, and formal logic (deeper verification, narrower coverage).
- **Informal Fallback:** For proofs that cannot be translated to formal syntax (flagged for manual review).

## 4. Experiments

### 4.1 Setup

We process 200,000 mathematical reasoning traces from the Solace dataset through FormalVerify, using a 32-node cluster with 256 CPU cores for parallel verification.

### 4.2 Results

| Metric | Value |
|--------|-------|
| Traces Processed | 200,000 |
| Steps Verified | 1.4M |
| Hallucinations Detected | 18.3% |
| Invisible to Informal Check | 7.2% |
| Auto-Correctable | 41% |
| Pruned | 59% |
| Verification Time | 8.2 hours |

### 4.3 Downstream Impact

| Benchmark | Unvalidated | FormalVerify | Gain |
|-----------|------------|-------------|------|
| MATH | 42.1% | 47.9% | +5.8% |
| GSM8K | 78.3% | 82.1% | +3.8% |
| AMC | 31.2% | 36.4% | +5.2% |
| Olympiad | 18.7% | 23.1% | +4.4% |

## 5. Analysis

### 5.1 Verification Coverage

Lean 4 covers 73% of mathematical steps (algebra, number theory, combinatorics). Isabelle covers an additional 14% (analysis, logic). The remaining 13% require manual review.

### 5.2 Error Type Detection

| Error Type | Informal Detection | Formal Detection |
|-----------|-------------------|-----------------|
| Sign Errors | 62% | 99.8% |
| Invalid Theorem Use | 71% | 99.9% |
| Division by Zero | 54% | 100% |
| Circular Reasoning | 78% | 95.2% |

Formal verification is particularly effective at detecting sign errors and division by zero, which are often missed by informal checking.

## 6. Limitations

Formal verification requires translating natural language proofs to formal syntax, which introduces translation errors. We estimate that 3.1% of valid proofs are incorrectly rejected due to translation failures.

Additionally, FormalVerify cannot verify proofs that use domain-specific notation not supported by Lean 4 or Isabelle (e.g., advanced physics notation, specialized mathematical conventions).

## 7. Conclusion

Formal proof assistants provide a deterministic, scalable mechanism for validating mathematical reasoning in distillation datasets. FormalVerify integrates Lean 4 and Isabelle into the synthetic CoT generation pipeline, detecting 18.3% of hallucinated proofs (including 7.2% invisible to informal checking) and improving downstream math accuracy by 5.8%.

The key insight is that **formal verification is not just more thorough than informal checking—it catches qualitatively different errors** that informal methods cannot detect, particularly subtle sign errors and invalid theorem applications.

## References

1. Detecting and Correcting Hallucinations in LLM-Generated Code. arXiv 2601.19106, January 2026.
2. Lean 4: A Theorem Proving Framework. Lean Prover Community, 2025.
3. Isabelle/HOL: A Proof Assistant. TU Munich, 2025.
4. Sledgehammer for Isabelle. 2025.
5. Project Solace: Distilling Multi-Teacher Reasoning Traces. Solstice-AI, 2026.
6. Formal Mathematical Reasoning in LLMs. 2025.
7. Lean 4 for Mathematical Reasoning. 2025.
8. Automated Theorem Proving with Large Language Models. 2025.
9. Step-Level AST Validation for Distillation Datasets. Solstice-AI, 2026.
10. Unit-Test Driven Synthesis for SWE-Bench Distillation. Solstice-AI, 2026.
