---
title: "SymPy-Assisted Symbolic Assertions for Deterministic Mathematical Distillation"
authors:
  - "Solstice-AI Research Team"
pubDate: 2026-09-01
tldr: "Using symbolic computer algebra systems (CAS) to assert exact equality on intermediate algebraic steps in high-school and Olympiad math traces."
abstract: "Mathematical reasoning traces from LLMs contain intermediate algebraic steps that may introduce subtle errors (wrong sign, dropped term, incorrect simplification) invisible to numerical checking. We present SymPyAssert, a framework that uses SymPy's symbolic computer algebra system to verify exact algebraic equality between consecutive reasoning steps. SymPyAssert processes 300,000 mathematical reasoning traces, detecting 9.7% of hallucinated intermediate steps (including 4.3% that pass numerical verification at 1000 random test points), improving downstream student MATH accuracy by 4.1%."
venue: "Research Technical Report"
highlightMetrics:
  - label: "Hallucinations Detected"
    value: "9.7%"
  - label: "Invisible to Numerical Check"
    value: "4.3%"
  - label: "MATH Accuracy Gain"
    value: "+4.1%"
bibtex: |
  @article{solstice2026sympyassert,
    title={SymPy-Assisted Symbolic Assertions for Deterministic Mathematical Distillation},
    author={Solstice-AI Research Team},
    journal={Solstice-AI Technical Report},
    year={2026},
    url={https://solstice-ai.co/papers/sympy-symbolic-assertions}
  }
tags:
  - "SymPy"
  - "Symbolic Verification"
  - "Mathematical Reasoning"
  - "Computer Algebra"
featured: false
---

## 1. Introduction

Mathematical reasoning traces contain intermediate algebraic steps that must be exactly equal to the previous step for the proof to be valid. However, LLMs often introduce subtle algebraic errors—dropping a term, changing a sign, or simplifying incorrectly—that are invisible to numerical checking.

Numerical verification (evaluating at 1000 random points) catches most algebraic errors, but misses errors that happen to be zero at the test points (e.g., errors in higher-order terms that are small at most points). SymPy's symbolic CAS provides exact algebraic verification, catching 100% of algebraic errors.

## 2. The Symbolic Verification Challenge

### 2.1 Why Numerical Checking Fails

Consider the identity $\sin^2(x) + \cos^2(x) = 1$. A student might "prove" this by writing $\sin^2(x) + \cos^2(x) = \sin^2(x) + (1 - \sin^2(x)) = 1$. This is correct. But if the student writes $\sin^2(x) + \cos^2(x) = \sin^2(x) + (1 + \sin^2(x)) = 1 + 2\sin^2(x)$, this is incorrect—but numerical checking at $x = 0$ gives $0 + 1 = 1$ and $0 + 1 = 1$, which matches!

The error only manifests at $x \neq 0$, and even then, numerical checking at 1000 random points might miss the exact values where the error is detectable.

### 2.2 SymPy's Exact Verification

SymPy can verify exact algebraic equality by simplifying both sides of an equation to canonical form and comparing:

```python
from sympy import symbols, sin, cos, simplify
x = symbols('x')
lhs = sin(x)**2 + cos(x)**2
rhs = 1
assert simplify(lhs - rhs) == 0  # Exact verification
```

This verification is deterministic and catches all algebraic errors, regardless of how subtle they are.

## 3. SymPyAssert Framework

### 3.1 Step Extraction

SymPyAssert extracts consecutive mathematical expressions from the reasoning trace and parses them into SymPy expressions using a custom parser that handles:

- Standard mathematical notation (fractions, exponents, roots)
- Greek letters and mathematical symbols
- Summation and product notation
- Integral and derivative notation

### 3.2 Equality Verification

For each pair of consecutive steps, SymPyAssert verifies:

$$\text{simplify}(\text{step}_{i+1} - \text{step}_i) = 0$$

If the simplification returns 0, the steps are exactly equal. If not, the step is flagged as a potential hallucination.

### 3.3 Fallback Strategies

Some expressions are too complex for direct SymPy simplification. SymPyAssert uses three fallback strategies:

1. **Numerical Substitution:** Evaluate at 10,000 random points (more thorough than standard 1,000).
2. **Factored Form:** Factor both sides and compare factor structures.
3. **Taylor Expansion:** Expand both sides around a reference point and compare coefficients.

### 3.4 Translation Pipeline

Natural language math expressions are translated to SymPy syntax using:

1. **Rule-based parsing:** Handle common patterns (fractions, exponents, trig functions).
2. **LLM-assisted parsing:** Use a fine-tuned model for complex expressions.
3. **Human-in-the-loop:** Flag ambiguous expressions for manual review.

## 4. Experiments

### 4.1 Setup

We process 300,000 mathematical reasoning traces through SymPyAssert, verifying 2.1M intermediate algebraic steps.

### 4.2 Results

| Metric | Value |
|--------|-------|
| Steps Verified | 2.1M |
| Hallucinations Detected | 9.7% |
| Invisible to Numerical Check | 4.3% |
| Translation Success Rate | 91.2% |
| Verification Time | 4.7 hours |

### 4.3 Downstream Impact

| Benchmark | Numerical Check Only | + SymPyAssert | Gain |
|-----------|---------------------|--------------|------|
| MATH | 43.8% | 47.9% | +4.1% |
| GSM8K | 79.1% | 82.1% | +3.0% |
| AMC | 32.4% | 36.4% | +4.0% |

## 5. Analysis

### 5.1 Error Type Detection

| Error Type | Numerical Detection | SymPy Detection |
|-----------|-------------------|----------------|
| Sign Errors | 94.3% | 100% |
| Dropped Terms | 87.1% | 100% |
| Incorrect Simplification | 78.4% | 100% |
| Wrong Identity | 91.2% | 100% |

### 5.2 Verification Speed

SymPy verification averages 2.2ms per step, compared to 0.8ms for numerical checking. The 2.75x overhead is justified by the 4.3% additional hallucination detection.

## 6. Limitations

SymPyAssert requires translating natural language math to SymPy syntax, which succeeds for 91.2% of expressions. The remaining 8.8% require manual review or fall back to numerical checking.

## 7. Conclusion

Numerical verification misses algebraic errors that are zero at test points. SymPyAssert uses symbolic CAS to verify exact algebraic equality, catching 9.7% of hallucinated intermediate steps (including 4.3% invisible to numerical checking).

The key insight is that **exact symbolic verification is strictly more powerful than numerical verification** for algebraic correctness, and the 2.75x overhead is justified by the significant improvement in dataset quality.

## References

1. SymPy: Python Library for Symbolic Mathematics. 2025.
2. Project Solace: Distilling Multi-Teacher Reasoning Traces. Solstice-AI, 2026.
3. Formal Verification Sandboxes with Lean 4 and Isabelle. Solstice-AI, 2026.
4. Step-Level AST Validation. Solstice-AI, 2026.
5. Unit-Test Driven Synthesis. Solstice-AI, 2026.
6. Automated Mathematical Reasoning with CAS. 2025.
7. Symbolic Verification of LLM Mathematical Proofs. 2025.
8. MATH: A Dataset for Measuring Mathematical Reasoning. 2021.
9. GSM8K: Grade School Math Problems. 2021.
10. Hallucination Detection in Mathematical Reasoning. 2025.
