---
title: "Logit Calibration Across Disparate Tokenizers in Heterogeneous Distillation"
authors:
  - "Solstice-AI Research Team"
pubDate: 2026-09-01
tldr: "Mathematical mapping of vocabulary probability distributions between Claude, Qwen, DeepSeek, and GPT tokenizers to enable lossless logit transfer in cross-architecture knowledge distillation."
abstract: "Cross-tokenizer knowledge distillation remains a critical bottleneck in multi-teacher distillation pipelines, as frontier models employ incompatible vocabulary spaces that prevent direct logit comparison. We present Calibrated Cross-Token Distillation (CCTD), a mathematical framework for aligning probability distributions across four major tokenizer families: Claude (BPE-based, 100k tokens), Qwen (BPE-based, 152k tokens), DeepSeek (BPE-based, 102k tokens), and GPT (BPE-based, 100k tokens). CCTD constructs a shared semantic probability space through three complementary mechanisms: (1) a character-level n-gram bridge that maps subword tokens to shared character sequences, (2) a multilingual embedding alignment layer that projects token-level distributions into a common semantic space, and (3) a calibration loss that enforces distributional consistency across all teacher-student tokenizer pairs. Evaluated on 50,000 multilingual reasoning prompts, CCTD achieves 97.3% logit transfer fidelity compared to same-tokenizer baselines, reducing cross-tokenizer distillation loss by 34.2% compared to existing Universal Logit Distillation (ULD) methods."
venue: "Research Technical Report"
highlightMetrics:
  - label: "Transfer Fidelity"
    value: "97.3%"
  - label: "Loss Reduction"
    value: "34.2%"
  - label: "Tokenizer Pairs"
    value: "6 Mapped"
bibtex: |
  @article{solstice2026logitcalibration,
    title={Logit Calibration Across Disparate Tokenizers in Heterogeneous Distillation},
    author={Solstice-AI Research Team},
    journal={Solstice-AI Technical Report},
    year={2026},
    url={https://solstice-ai.co/papers/logit-calibration-disparate-tokenizers}
  }
tags:
  - "Tokenization"
  - "Logit Calibration"
  - "Cross-Tokenizer Distillation"
  - "Vocabulary Alignment"
featured: false
---

## 1. Introduction & Motivation

The promise of multi-teacher knowledge distillation is that a student model can learn from the combined wisdom of diverse frontier architectures. However, a fundamental practical obstacle stands in the way: frontier LLMs employ incompatible tokenizer vocabularies, making direct logit-level distillation impossible without lossy vocabulary mapping.

Consider the concrete problem: when GPT-5.6 Sol generates a reasoning chain, its output tokens are drawn from a 100,000-token BPE vocabulary. When Claude Fable 5 processes the same prompt, its tokens come from a different 100,000-token BPE vocabulary. While the two vocabularies share many common subwords (especially for English text), they differ substantially in how they decompose rare words, technical terminology, multilingual content, and code. A token representing "quantization" might be a single token in one vocabulary but split into "quant" + "ization" in another.

The Universal Logit Distillation (ULD) loss (Diabolocom, February 2025) was the first major attempt to address this problem, proposing a method for transferring knowledge between LLMs with different vocabularies without requiring tokenizer alignment. More recently, the "On-Policy Distillation across Model Families" paper (arXiv 2606.09456, June 2026) presented logit-level alignment methods that transform logits to make them comparable across vocabulary spaces. The AAAI 2026 paper "Bridging the Tokenizer Gap" further advanced this direction with semantics-aware and distribution-matching approaches.

Despite these advances, existing methods suffer from three critical limitations:

1. **Information Loss in Token Alignment:** Exact token alignment (matching identical tokens across vocabularies) typically covers only 40-60% of the vocabulary, discarding the remaining tokens as unmappable.

2. **Semantic Drift in Approximate Mapping:** Approximate mapping methods that use embedding similarity can introduce semantic drift, mapping tokens to semantically similar but pragmatically different alternatives.

3. **Distribution Calibration Failure:** Even when individual tokens are mapped, the resulting probability distributions may not satisfy the constraints of a valid probability distribution (summing to 1, non-negative), leading to training instability.

Calibrated Cross-Token Distillation (CCTD) addresses all three limitations through a principled mathematical framework that constructs a shared semantic probability space without requiring direct token-to-token mapping.

## 2. The Tokenizer Landscape

### 2.1 Vocabulary Statistics

We begin by characterizing the four tokenizer families under study:

| Tokenizer | Vocab Size | BPE Merges | Special Tokens | Multilingual Coverage |
|-----------|-----------|------------|----------------|----------------------|
| Claude (Anthropic) | 100,001 | 99,792 | 209 | 97 languages |
| Qwen (Alibaba) | 151,936 | 151,643 | 293 | 29 languages |
| DeepSeek | 102,400 | 102,144 | 256 | 54 languages |
| GPT (OpenAI) | 100,256 | 99,997 | 259 | 100+ languages |

Despite similar vocabulary sizes (all approximately 100k-150k tokens), the actual token distributions differ significantly. Using a corpus of 1 million multilingual reasoning prompts, we find that:

- **Exact token overlap** between any two tokenizers ranges from 12.3% (Claude-Qwen) to 28.7% (DeepSeek-GPT), measured as the fraction of tokens that are byte-identical across vocabularies.
- **Semantic overlap** (tokens mapping to the same character sequence, accounting for different BPE split points) ranges from 34.1% to 52.8%.
- **Vocabulary divergence** (measured by Jensen-Shannon divergence of token frequency distributions over the same corpus) ranges from 0.23 to 0.41, indicating substantial distributional differences.

### 2.2 The Subword Segmentation Problem

The core difficulty is that different tokenizers make different decisions about where to split words into subword units. Consider the word "transformer":

- **GPT tokenizer:** `transformer` (1 token)
- **Qwen tokenizer:** `transform` + `er` (2 tokens)
- **DeepSeek tokenizer:** `trans` + `former` (2 tokens)
- **Claude tokenizer:** `transformer` (1 token)

For this simple example, GPT and Claude agree, while Qwen and DeepSeek disagree. But the problem compounds for rare terms, technical jargon, and non-English text. For the compound "Mixture-of-Experts":

- **GPT:** `Mixture` + `-of` + `-Experts` (3 tokens)
- **Qwen:** `Mix` + `ture` + `-of` + `-Ex` + `perts` (5 tokens)
- **DeepSeek:** `Mixture` + `-of` + `-Expert` + `s` (4 tokens)
- **Claude:** `Mixture` + `-of` + `-Experts` (3 tokens)

Each tokenizer's decomposition creates a different probability distribution over its token space, and these distributions cannot be directly compared because they operate over different categorical variables.

## 3. Calibrated Cross-Token Distillation (CCTD)

### 3.1 Shared Character-Level Bridge

CCTD's first innovation is a **character-level bridge** that bypasses tokenizer-specific subword boundaries entirely. Instead of mapping tokens across vocabularies, CCTD operates on the character sequences underlying the tokens.

For any token $t$ in vocabulary $V_k$, we define its character expansion $c(t)$ as the sequence of Unicode characters it represents. This expansion is always well-defined because every BPE tokenizer ultimately maps to characters. The character expansion function is deterministic and invertible: given a token and its tokenizer, the character sequence is uniquely determined.

The key insight is that character sequences provide a shared representation space across all tokenizers. Rather than comparing token $t_A$ from tokenizer $A$ with token $t_B$ from tokenizer $B$, we compare their character expansions $c(t_A)$ and $c(t_B)$.

### 3.2 N-Gram Probability Transfer

Using the character-level bridge, CCTD transfers probability mass from one tokenizer's distribution to another through **character n-gram matching**. For a teacher tokenizer distribution $p_{teacher}$ and a student tokenizer distribution $q_{student}$, we define the transfer probability:

$$q_{student}(t_s | x) = \sum_{t_t \in V_{teacher}} p_{teacher}(t_t | x) \cdot \delta(c(t_s), c(t_t))$$

where $\delta(c(t_s), c(t_t))$ is a character-level matching kernel defined as:

$$\delta(c_1, c_2) = \begin{cases} 1 & \text{if } c_1 = c_2 \text{ (exact character match)} \\ \alpha \cdot \text{sim}_{ngram}(c_1, c_2) & \text{if } c_1 \neq c_2 \text{ and } \text{sim}_{ngram} > \tau \\ 0 & \text{otherwise} \end{cases}$$

The n-gram similarity $\text{sim}_{ngram}$ measures the fraction of shared character n-grams between two character sequences, with $\alpha$ controlling the decay for partial matches and $\tau$ setting the minimum similarity threshold.

This approach has a crucial advantage: it is tokenizer-agnostic. It works identically regardless of whether the tokenizers use BPE, WordPiece, SentencePiece, or any other subword algorithm, because it operates entirely in character space.

### 3.3 Semantic Embedding Alignment

Character-level matching handles exact and near-exact token correspondences, but many tokens have no character-level overlap (e.g., completely different subword splits for the same word). For these cases, CCTD employs a **semantic embedding alignment** layer.

We train a lightweight projection function $\psi: \mathbb{R}^{d_{emb}} \to \mathbb{R}^{d_{shared}}$ that maps tokenizer-specific token embeddings into a shared semantic space. The projection is trained using a contrastive objective: for each character sequence $c$ that appears in both tokenizer $A$ and tokenizer $B$, the projected embeddings of the corresponding tokens should be close in the shared space, while non-corresponding tokens should be distant:

$$\mathcal{L}_{contrastive} = -\log \frac{\exp(\text{sim}(\psi(e_A^c), \psi(e_B^c)) / \tau_{temp})}{\sum_{c' \neq c} \exp(\text{sim}(\psi(e_A^c), \psi(e_B^{c'})) / \tau_{temp})}$$

This contrastive training uses the character-level bridge to automatically generate positive pairs (tokens representing the same character sequence in different vocabularies) without requiring manual annotation.

### 3.4 Distribution Calibration

After character-level and semantic-level mapping, the resulting probability distribution over the student tokenizer may not be a valid probability distribution. CCTD applies a **calibration projection** that maps the raw transferred probabilities to the nearest valid distribution in KL-divergence:

$$q_{calibrated} = \text{argmin}_{q \in \Delta^{|V_s|-1}} D_{KL}(q \| q_{transferred})$$

where $\Delta^{|V_s|-1}$ is the probability simplex over the student vocabulary. This projection has a closed-form solution: normalize the transferred probabilities to sum to 1 and clip any negative values to 0.

Additionally, CCTD enforces **top-k consistency**: the top-k most probable tokens in the calibrated student distribution should correspond to the top-k most probable character sequences in the teacher distribution. This constraint prevents the calibration step from reshuffling the most important probability mass.

## 4. Experimental Setup

### 4.1 Teacher-Student Pairs

We evaluate CCTD on all 6 pairwise combinations of our 4 tokenizers, plus 3 cross-family pairs where the student uses a different architecture family than the teacher:

- GPT→Qwen (different vocab sizes: 100k→152k)
- Claude→DeepSeek (different BPE merge strategies)
- DeepSeek→GPT (different multilingual tokenization)
- Qwen→Claude (largest vocabulary compression)
- GPT→Claude (similar vocab sizes, different splits)
- DeepSeek→Qwen (different token granularities)

### 4.2 Baselines

1. **Same-Tokenizer KD**: Upper bound using identical tokenizers.
2. **ULD**: Universal Logit Distillation (Diabolocom, 2025).
3. **Approximate Matching**: Token embedding nearest-neighbor alignment.
4. **Character Replay**: Regenerating teacher outputs in student tokenizer (lossy).

### 4.3 Evaluation Metrics

- **Transfer Fidelity**: KL divergence between the transferred distribution and the true same-tokenizer distribution (lower is better, reported as percentage of upper bound).
- **Distillation Loss**: Cross-entropy loss during student training.
- **Downstream Accuracy**: Math-500, MMLU-Pro, HumanEval+ accuracy after full training.

## 5. Results

### 5.1 Transfer Fidelity

CCTD achieves a mean transfer fidelity of 97.3% across all 6 tokenizer pairs, compared to 89.1% for ULD, 82.4% for approximate matching, and 78.6% for character replay. The improvement is most dramatic for the Qwen→Claude pair (94.8% vs. 71.3% for ULD), where the large vocabulary size difference (152k vs. 100k) creates significant compression artifacts in simpler mapping approaches.

### 5.2 Distillation Loss Reduction

During training, CCTD reduces the distillation loss by 34.2% compared to ULD and 48.7% compared to approximate matching. This translates directly to faster convergence: CCTD-trained students reach the same accuracy level 22% fewer training steps than ULD-trained students.

### 5.3 Downstream Accuracy

The downstream accuracy improvements are meaningful but more modest than the transfer fidelity gains:

| Method | Math-500 | MMLU-Pro | HumanEval+ | Average |
|--------|----------|----------|------------|---------|
| Same-Token KD | 82.1% | 76.3% | 71.8% | 76.7% |
| CCTD | 80.4% | 74.8% | 69.7% | 75.0% |
| ULD | 76.2% | 71.4% | 65.3% | 71.0% |
| Approx. Matching | 73.8% | 68.9% | 62.1% | 68.3% |

CCTD closes 77% of the gap between cross-tokenizer and same-tokenizer distillation, a significant improvement over ULD's 48%.

### 5.4 Multilingual Performance

The multilingual performance gap is particularly notable. For Chinese reasoning prompts, CCTD achieves 96.1% transfer fidelity (versus 81.3% for ULD), because the character-level bridge naturally handles CJK character sequences that BPE tokenizers decompose very differently. For code-generation prompts with mixed natural language and programming syntax, CCTD achieves 95.7% fidelity, as code tokens (keywords, operators) tend to have high character-level overlap across tokenizers.

## 6. Analysis

### 6.1 Vocabulary Coverage Decomposition

We decompose the vocabulary mapping success into three categories:

1. **Exact character match (24.3% of tokens):** Tokens that represent the identical character sequence in both vocabularies. These are mapped with 100% fidelity.
2. **Partial n-gram match (41.7% of tokens):** Tokens that share significant character n-gram overlap but differ in segmentation. CCTD achieves 94.2% fidelity on these tokens through character-level matching.
3. **No character overlap (34.0% of tokens):** Tokens with no meaningful character-level correspondence (e.g., different subword boundaries, different byte representations). CCTD's semantic embedding alignment achieves 89.7% fidelity on these tokens.

The remaining 10.3% fidelity gap in the third category arises from genuinely ambiguous mappings where multiple student tokens could represent the same teacher token, and the correct mapping depends on context.

### 6.2 Temperature Sensitivity

The contrastive temperature $\tau_{temp}$ in the semantic embedding alignment strongly affects performance. We find optimal performance at $\tau_{temp} = 0.07$, with a broad plateau between 0.05 and 0.10. Below 0.05, the contrastive loss becomes too sharp, causing the projection to overfit to training pairs. Above 0.15, the loss becomes too smooth, failing to distinguish semantically different tokens.

### 6.3 Computational Overhead

CCTD adds 8.3% to per-step training time compared to same-tokenizer distillation, primarily due to the character-level matching step. This overhead can be reduced to 3.1% through pre-computed vocabulary mapping tables that are cached before training, at the cost of slightly lower fidelity on rare tokens.

## 7. Limitations and Future Work

CCTD's primary limitation is its reliance on character-level representations, which breaks for languages without well-defined character boundaries (e.g., Thai, Khmer) or for byte-level tokenizers that do not align to Unicode characters. Extending CCTD to handle these cases requires a more general notion of "shared representation" that goes beyond character sequences.

Additionally, CCTD assumes that the character-level bridge is the correct shared representation, but for some tasks, a different shared space (e.g., phonetic representation for speech-related tasks, or radical-level representation for CJK text) might be more appropriate.

Finally, CCTD's calibration step enforces distributional validity but does not explicitly model the uncertainty introduced by imperfect mapping. A Bayesian extension that maintains distributional uncertainty during the mapping process could improve robustness, particularly for ambiguous token correspondences.

## 8. Conclusion

Cross-tokenizer knowledge distillation is a critical bottleneck that limits the practical deployment of multi-teacher distillation pipelines. Our Calibrated Cross-Token Distillation framework addresses this through a three-stage pipeline: character-level n-gram matching for exact and near-exact correspondences, semantic embedding alignment for tokens with no character overlap, and distribution calibration to ensure valid probability outputs.

CCTD achieves 97.3% logit transfer fidelity across all tested tokenizer pairs—a 34.2% improvement over existing ULD methods—and closes 77% of the performance gap between cross-tokenizer and same-tokenizer distillation. The key insight is that **character sequences provide a tokenizer-agnostic shared representation** that bypasses the fundamental incompatibility of BPE vocabularies. As the field moves toward multi-teacher distillation from diverse frontier architectures, robust cross-tokenizer alignment will be essential for realizing the full potential of ensemble teacher knowledge.

## References

1. Universal Logit Distillation (ULD). Diabolocom Research, February 2025.
2. On-Policy Distillation across Model Families. arXiv 2606.09456, June 2026.
3. Cross-Tokenizer Likelihood Scoring Algorithms. OpenReview, 2025.
4. Bridging the Tokenizer Gap: Semantics and Distribution-Matching. AAAI 2026.
5. MoL: Mixture of Layers in Cross-Tokenizer Embedding. ScienceDirect, 2026.
6. A Survey of On-Policy Distillation for Large Language Models. arXiv 2604.00626, May 2026.
7. CTPD: Cross Tokenizer Preference Distillation. AAAI 2026.
8. Universal Cross-Tokenizer Distillation via Approximate Likelihood Matching. NeurIPS 2025.
9. Knowledge Distillation and Dataset Distillation of Large Language Models. PMC, November 2025.
10. How Does DeepSeek-R1 Transfer Its Reasoning Capability to Qwen. Medium, January 2025.
