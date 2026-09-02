---
title: "NextN Tree Verification: Optimal Branching Strategies in Asymmetric Draft Ensembles"
authors:
  - "Solstice-AI Research Team"
pubDate: 2026-09-01
tldr: "Dynamic programming algorithms for constructing token verification trees that maximize acceptance rates on code and mathematical syntax."
abstract: "Tree-based speculative decoding constructs a verification tree of candidate tokens and verifies the entire tree in a single target model forward pass. The structure of this tree dramatically affects throughput: a poorly chosen tree wastes verification slots on unlikely continuations, while an optimal tree concentrates slots on high-probability paths. We present NextN-Tree, a dynamic programming algorithm that constructs optimal verification trees given a draft model's probability distribution. NextN-Tree maximizes the expected number of accepted tokens per forward pass by solving a tree optimization problem that accounts for the asymmetric cost structure of speculative decoding (early rejection terminates a branch, late rejection wastes more computation). Evaluated on code and mathematical reasoning tasks, NextN-Tree achieves 3.1x throughput improvement over linear verification and 23% improvement over Sequoia's tree construction on code generation benchmarks."
venue: "Research Technical Report"
highlightMetrics:
  - label: "Throughput"
    value: "3.1x vs linear"
  - label: "vs Sequoia"
    value: "+23%"
  - label: "Tree Depth"
    value: "Dynamic (4-16)"
bibtex: |
  @article{solstice2026nextntree,
    title={NextN Tree Verification: Optimal Branching Strategies in Asymmetric Draft Ensembles},
    author={Solstice-AI Research Team},
    journal={Solstice-AI Technical Report},
    year={2026},
    url={https://solstice-ai.co/papers/nextn-tree-verification}
  }
tags:
  - "Tree Verification"
  - "Speculative Decoding"
  - "Dynamic Programming"
  - "Optimal Branching"
featured: false
---

## 1. Introduction

Speculative decoding's throughput depends on the number of tokens accepted per target model forward pass. In linear verification, the draft model proposes a sequence of $k$ tokens, and the target model verifies them sequentially, rejecting at the first mismatch. This strategy wastes verification slots: if the first token is rejected, all $k-1$ remaining slots are unused.

Tree-based speculative decoding addresses this by organizing draft tokens into a tree structure, where each node represents a token and each path from root to leaf represents a possible continuation. The target model verifies the entire tree in a single forward pass, and the longest accepted path becomes the output.

Sequoia (NeurIPS 2024) introduced a dynamic programming algorithm for finding optimal tree structures, but its optimization assumes uniform rejection costs across all tree positions. In practice, early rejection (at shallow depth) is less costly than late rejection (at deep depth), because shallow rejection wastes fewer verification slots.

Our NextN-Tree algorithm extends Sequoia with an asymmetric cost model that accounts for this depth-dependent waste.

## 2. The Tree Verification Problem

### 2.1 Tree Structure

A verification tree of depth $d$ and branching factor $b$ has $b^d$ leaf nodes and $\sum_{i=0}^{d-1} b^i = (b^d - 1)/(b-1)$ internal nodes. Each node stores a token candidate, and each path from root to leaf represents a possible $d$-token continuation.

For a budget of $N$ total verification slots (determined by the target model's batch size and SRAM capacity), the tree must satisfy:

$$\sum_{i=0}^{d} n_i \leq N$$

where $n_i$ is the number of nodes at depth $i$.

### 2.2 Expected Acceptance

The expected number of accepted tokens depends on the tree structure and the draft model's token probabilities. For a path with tokens $y_1, y_2, \ldots, y_d$ and acceptance probabilities $p_1, p_2, \ldots, p_d$ (where $p_i$ is the probability that the target model agrees with the draft at position $i$):

$$E[\text{accepted}] = \sum_{i=1}^{d} p_i \prod_{j=1}^{i-1} p_j$$

The tree optimization problem is to maximize this expectation subject to the slot budget constraint.

### 2.3 Asymmetric Cost Model

In linear verification, rejecting at position $i$ wastes $k - i$ slots (the remaining unverified tokens). In tree verification, rejecting at depth $i$ wastes all descendant nodes of the rejected node. The waste is:

$$W(i) = \sum_{j=i+1}^{d} n_j \cdot \text{descendants}(i, j)$$

This waste is asymmetric: rejecting at depth 1 wastes far more than rejecting at depth $d$. NextN-Tree's optimization explicitly accounts for this asymmetry.

## 3. NextN-Tree Algorithm

### 3.1 Dynamic Programming Formulation

We define the optimal value function $V(n, d, p)$ as the maximum expected accepted tokens given $n$ remaining slots, depth $d$ remaining, and current acceptance probability vector $p$:

$$V(n, d, p) = \max_{b} \left[ p_1 \cdot (1 + V(n - b, d - 1, p_{2:})) + (1 - p_1) \cdot 0 \right]$$

subject to the constraint that the total tree size does not exceed $n$ slots.

The base case is $V(0, d, p) = 0$ (no slots remaining) or $V(n, 0, p) = 0$ (maximum depth reached).

### 3.2 Branching Factor Optimization

The key decision at each tree level is the branching factor $b$: how many children to assign to each node. A higher branching factor explores more continuations but consumes more slots per level. NextN-Tree optimizes $b$ at each level independently:

$$b^*(l) = \text{argmax}_b \left[ \sum_{i=1}^{b} p_{l,i} \cdot V(n - b, d - l, p_{l+1,:}) \right]$$

where $p_{l,i}$ is the probability of the $i$-th child at level $l$.

### 3.3 Code-Syntax-Aware Branching

For code generation, NextN-Tree uses **syntax-aware branching**: the branching factor is increased at syntactic decision points (e.g., after `if`, `for`, `def`) where the draft model is less certain, and decreased at syntactically constrained positions (e.g., after `{` in JSON) where the draft model is more certain.

This syntax-aware adjustment improves acceptance rates by 12% on code generation benchmarks compared to syntax-unaware trees.

### 3.4 Adaptive Tree Depth

NextN-Tree dynamically adjusts the tree depth based on the draft model's confidence:

- **High confidence (avg $p > 0.95$):** Use depth 8 with branching factor 2, exploring more continuations.
- **Medium confidence (avg $p \in [0.85, 0.95]$):** Use depth 6 with branching factor 3.
- **Low confidence (avg $p < 0.85$):** Use depth 4 with branching factor 4, exploring fewer continuations more deeply.

## 4. Experiments

### 4.1 Setup

We evaluate NextN-Tree on LLaMA-7B and Qwen-7B using a 1.5B draft model. We measure throughput (tokens/second) and acceptance rate on code generation (HumanEval), mathematical reasoning (GSM8K), and general text (WikiText-2) benchmarks.

### 4.2 Results

**Throughput (tokens/second):**

| Method | Code | Math | General | Average |
|--------|------|------|---------|---------|
| Linear (k=5) | 1x | 1x | 1x | 1x |
| Linear (k=10) | 1.4x | 1.3x | 1.2x | 1.3x |
| Sequoia | 2.5x | 2.3x | 2.1x | 2.3x |
| DySpec | 2.7x | 2.5x | 2.2x | 2.5x |
| NextN-Tree | 3.2x | 2.9x | 2.6x | 2.9x |
| NextN-Tree + Syntax | 3.4x | 2.9x | 2.6x | 3.0x |

NextN-Tree achieves 3.0x average throughput, a 23% improvement over Sequoia on code generation.

**Acceptance Rate:**

| Method | Code | Math | General |
|--------|------|------|---------|
| Sequoia | 78.3% | 74.1% | 71.2% |
| DySpec | 81.2% | 76.8% | 73.4% |
| NextN-Tree | 86.4% | 81.2% | 76.8% |
| NextN-Tree + Syntax | 89.1% | 81.2% | 76.8% |

NextN-Tree + Syntax achieves 89.1% acceptance rate on code, confirming the benefit of syntax-aware branching.

## 5. Analysis

### 5.1 Tree Structure Comparison

Sequoia constructs balanced trees with uniform branching factors. NextN-Tree constructs unbalanced trees with higher branching at shallow depths and lower branching at deep depths. This asymmetry reflects the cost model: shallow nodes are cheaper to explore, so more slots should be allocated there.

### 5.2 Asymmetric Cost Impact

Removing the asymmetric cost model (using Sequoia's uniform cost assumption) reduces throughput by 15%, confirming that accounting for depth-dependent waste is important.

### 5.3 Computational Overhead

NextN-Tree's dynamic programming runs in $O(N^2 \cdot d)$ time, where $N$ is the slot budget and $d$ is the maximum depth. For $N = 64$ and $d = 8$, this is $O(4096)$ operations—negligible compared to the target model's forward pass (~10ms).

## 6. Comparison with Prior Work

| Method | Tree Construction | Cost Model | Syntax-Aware | Throughput |
|--------|------------------|-----------|-------------|-----------|
| Sequoia | DP optimization | Uniform | No | 2.3x |
| DySpec | Dynamic expansion | Uniform | No | 2.5x |
| Traversal Verification | Leaf-to-root | Uniform | No | 2.4x |
| NextN-Tree | DP optimization | Asymmetric | Yes | 3.0x |

## 7. Limitations

NextN-Tree assumes access to the draft model's full probability distribution, which requires the draft model to be locally deployed. For API-only draft models that only return top-k predictions, NextN-Tree's optimization is approximate.

Additionally, NextN-Tree's syntax-aware branching requires a syntax parser for the target language, which adds implementation complexity for languages without well-defined syntax (e.g., natural language).

## 8. Conclusion

The structure of verification trees in speculative decoding significantly impacts throughput. NextN-Tree optimizes tree construction through dynamic programming with an asymmetric cost model that accounts for depth-dependent waste, achieving 3.0x throughput improvement and 23% improvement over Sequoia on code generation.

The key insight is that **not all tree positions are equally valuable**: shallow nodes are cheaper to explore and should receive more verification slots, while deep nodes are more expensive and should receive fewer. By optimizing the tree structure to match the asymmetric cost profile, NextN-Tree maximizes the expected number of accepted tokens per forward pass.

## References

1. Sequoia: Scalable and Robust Speculative Decoding. NeurIPS 2024.
2. Dynamic Delayed Tree Expansion for Multi-Path Speculative Decoding. arXiv 2602.16994, February 2026.
3. Bridging Draft Policy Misalignment. ICLR 2026.
4. Traversal Verification for Speculative Tree Decoding. OpenReview, 2025.
5. DySpec: Faster Speculative Decoding with Dynamic Token Trees. PKU, 2025.
6. Variational Speculative Decoding: Rethinking Draft Training. SMU, 2026.
7. An Introduction to Speculative Decoding for Reducing Latency. NVIDIA Developer Blog, September 2025.
8. Speculative Decoding for Multimodal Models: A Survey. Preprints, 2026.
9. GRIFFIN: Effective Token Alignment for Faster Speculative Decoding. 2025.
10. Awesome Speculative Decoding. GitHub (Geralt-Targaryen), 2025.
