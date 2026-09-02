---
title: "Semantic Deduplication: Graph-Based Equivalence Checking on Multi-Teacher ASTs"
authors:
  - "Solstice-AI Research Team"
pubDate: 2026-09-01
tldr: "Detecting semantically identical code implementations across multiple sources using Abstract Syntax Tree isomorphism rather than text similarity."
abstract: "Multi-teacher code distillation produces semantically identical code implementations expressed with different variable names, formatting, and structure. Text-based deduplication misses these equivalent implementations. We present AST-Dedup, a graph-based deduplication framework that compares code implementations using AST isomorphism—detecting semantically identical code regardless of variable naming, whitespace, or structural differences. AST-Dedup processes 2 million code examples from 7 teachers, identifying 14.3% semantically redundant implementations (vs. 8.7% for text-based deduplication), reducing dataset size while preserving code diversity."
venue: "Research Technical Report"
highlightMetrics:
  - label: "Semantic Redundancy"
    value: "14.3%"
  - label: "vs Text Dedup"
    value: "+64%"
  - label: "Code Examples"
    value: "2M"
bibtex: |
  @article{solstice2026astdedup,
    title={Semantic Deduplication: Graph-Based Equivalence Checking on Multi-Teacher ASTs},
    author={Solstice-AI Research Team},
    journal={Solstice-AI Technical Report},
    year={2026},
    url={https://solstice-ai.co/papers/semantic-deduplication-ast-isomorphism}
  }
tags:
  - "AST Isomorphism"
  - "Semantic Deduplication"
  - "Code Deduplication"
  - "Graph Matching"
featured: false
---

## 1. Introduction

When multiple teachers generate code for the same problem, they often produce semantically identical implementations that differ only in variable naming, formatting, and superficial structure. Text-based deduplication (SHA256, MinHash) misses these equivalent implementations because the text is different even though the code is semantically the same.

AST-Dedup compares code implementations using Abstract Syntax Tree (AST) isomorphism, detecting semantic equivalence regardless of surface differences.

## 2. AST Isomorphism for Code Equivalence

### 2.1 Graph Representation

Each code implementation is represented as a directed acyclic graph (DAG) where:
- **Nodes:** AST nodes (expressions, statements, declarations)
- **Edges:** Parent-child relationships in the AST
- **Labels:** Node types (BinaryOp, FunctionCall, Variable, etc.)

### 2.2 Isomorphism Detection

Two ASTs are isomorphic if there exists a bijection between their node sets that preserves the graph structure and node types. We solve the AST isomorphism problem using:

1. **Canonical Labeling:** Compute a canonical form for each AST using the Weisfeiler-Lehman algorithm.
2. **Hash Comparison:** Compare canonical hashes—identical hashes indicate isomorphism.
3. **Variable Normalization:** Normalize variable names to canonical indices before comparison (so `x + y` and `a + b` produce the same canonical form).

### 2.3 Semantic Equivalence

Beyond structural isomorphism, AST-Dedup detects semantic equivalence through:

1. **Constant Folding:** Evaluate constant expressions to their values.
2. **Algebraic Simplification:** Simplify equivalent algebraic expressions.
3. **Loop Unrolling:** Detect loops that produce the same iteration pattern.

## 3. Multi-Teacher Deduplication

### 3.1 Teacher Pair Comparison

AST-Dedup compares code across all 7 teacher pairs (21 pairs total), building a deduplication graph where nodes are code examples and edges connect semantically equivalent implementations.

### 3.2 Quality-Based Selection

For each cluster of equivalent implementations, AST-Dedup retains the highest-quality example based on:
- Code readability (identifier naming, comment density)
- Test pass rate (functional correctness)
- Teacher confidence (generation entropy)

### 3.3 Diversity Preservation

AST-Dedup preserves diversity by retaining at least one example from each teacher in each cluster, ensuring that the dataset maintains multi-teacher perspectives even after deduplication.

## 4. Experiments

### 4.1 Setup

We process 2 million code examples from 7 teachers through AST-Dedup, comparing all 21 teacher pairs.

### 4.2 Results

| Method | Redundancy Detected | Dataset Reduction |
|--------|-------------------|-------------------|
| SHA256 (exact) | 8.2% | 8.2% |
| MinHash (5-gram) | 11.4% | 11.4% |
| Text Similarity | 12.1% | 12.1% |
| AST-Dedup | 14.3% | 14.3% |
| AST-Dedup + Text | 16.7% | 16.7% |

AST-Dedup detects 64% more semantic redundancy than text-based deduplication.

### 4.3 Downstream Impact

| Metric | Pre-Dedup | Post-Dedup | Change |
|--------|-----------|------------|--------|
| HumanEval+ | 67.4% | 71.2% | +3.8% |
| MBPP+ | 62.1% | 65.3% | +3.2% |
| Training Time | 142 GPU-hrs | 118 GPU-hrs | -17% |

## 5. Analysis

### 5.1 Teacher Overlap

The highest semantic overlap is between GPT-5.6 Sol and Claude Fable 5 (18.7%), reflecting their similar training data. The lowest is between DeepSeek V4 Pro and Qwen 3.8-Max (9.2%).

### 5.2 Variable Normalization Impact

Without variable normalization, AST-Dedup detects 11.2% redundancy. With normalization, it detects 14.3%—a 28% improvement from accounting for variable naming differences.

## 6. Limitations

AST-Dedup requires parsing code into ASTs, which adds 15% overhead compared to text-based deduplication. For code in languages without mature parsers, AST-Dedup falls back to text-based methods.

## 7. Conclusion

Multi-teacher code distillation produces semantically identical implementations with different surface forms. AST-Dedup detects 14.3% semantic redundancy (64% more than text-based methods) using AST isomorphism with variable normalization.

The key insight is that **code equivalence is a graph isomorphism problem, not a text matching problem**, and AST-based comparison catches semantically equivalent code that text-based methods miss.

## References

1. Semantic Sieve: Exact N-Gram and Embedding Deduplication. Solstice-AI, 2026.
2. Step-Level AST Validation. Solstice-AI, 2026.
3. Detecting and Correcting Hallucinations in LLM-Generated Code. arXiv 2601.19106, January 2026.
4. Graph-Based Code Clone Detection. 2025.
5. Weisfeiler-Lehman Graph Kernels. 2025.
6. AST-Based Code Similarity Detection. 2025.
7. Semantic Code Deduplication at Scale. 2025.
8. Multi-Teacher Code Distillation. 2025.
9. Code Deduplication for LLM Training. 2025.
10. Abstract Syntax Tree Isomorphism Detection. 2025.
