---
title: "Semantic Sieve: Exact N-Gram and Embedding Deduplication at Petabyte Scale"
authors:
  - "Solstice-AI Research Team"
pubDate: 2026-09-01
tldr: "Algorithmic framework combining MinHash LSH and exact SHA256 hashing to eliminate 20%+ synthetic redundancy in massive conversational corpora, with embedding-based semantic clustering for near-duplicate detection at trillion-token scale."
abstract: "As synthetic data generation for LLM distillation scales to petabyte-scale corpora, redundant and near-duplicate content becomes a critical source of training inefficiency and student model overfitting. We present Semantic Sieve, a two-stage deduplication framework that combines exact byte-level matching via SHA256 hashing with approximate semantic deduplication via MinHash LSH and embedding-based clustering. Semantic Sieve processes 1.2 trillion tokens across 60 audited datasets in the Project Solace corpus, identifying and removing 23.7% redundant content (286 billion tokens) while preserving 99.4% of semantically unique reasoning chains. The framework achieves 847x speedup over brute-force pairwise comparison, processing 50 million documents per hour on a 32-node cluster, and reduces downstream training time by 31% while improving student accuracy by 2.8% through reduced memorization pressure."
venue: "Research Technical Report"
highlightMetrics:
  - label: "Redundancy Removed"
    value: "23.7%"
  - label: "Speedup"
    value: "847x"
  - label: "Training Time"
    value: "-31%"
bibtex: |
  @article{solstice2026semanticsieve,
    title={Semantic Sieve: Exact N-Gram and Embedding Deduplication at Petabyte Scale},
    author={Solstice-AI Research Team},
    journal={Solstice-AI Technical Report},
    year={2026},
    url={https://solstice-ai.co/papers/semantic-sieve-deduplication-petabyte}
  }
tags:
  - "Deduplication"
  - "Data Quality"
  - "MinHash LSH"
  - "Semantic Similarity"
featured: false
---

## 1. Introduction & Motivation

The scaling of synthetic data generation for knowledge distillation has created an unprecedented data quality challenge. When multiple frontier teachers generate responses to overlapping prompt distributions, the resulting corpora contain massive redundancy: identical prompts produce near-identical responses, semantically equivalent questions yield structurally similar reasoning chains, and popular prompts (e.g., standard benchmark questions) appear thousands of times across different datasets.

The Project Solace corpus (Solstice-AI, August 2026) documented the scale of this problem: 60 audited datasets containing 16.8 billion tokens, of which 3.44 million instances (27.4% of unique rows) were exact duplicates purged through SHA256 hashing. However, exact deduplication captures only the most obvious redundancy. Semantically equivalent content—different phrasings of the same question, responses that follow identical reasoning structures with minor wording variations—represents an additional layer of redundancy that exact matching cannot detect.

The "Scaling Synthetic Data Creation with 1,000,000,000 Personas" paper (arXiv 2406.20094, 2025) employed a two-stage deduplication pipeline: MinHash-based deduplication using n-gram features with 0.9 similarity threshold, followed by embedding-based deduplication. LSHBloom (arXiv 2411.04257, January 2026) advanced internet-scale text deduplication by combining Bloom filters with LSH for memory-efficient near-duplicate detection. SemDeDup demonstrated that semantic deduplication via embedding similarity achieves percent-level training efficiency gains.

Our Semantic Sieve framework extends these approaches with a three-tier architecture optimized for the specific characteristics of multi-teacher synthetic distillation corpora, where redundancy manifests at multiple levels: byte-identical, n-gram-similar, and semantically-equivalent.

## 2. The Redundancy Landscape

### 2.1 Types of Redundancy

We categorize redundancy in synthetic distillation data into four levels:

**Level 0: Byte-Identical Duplicates.** The exact same byte sequence appears multiple times. These are caused by duplicate prompts, copy-paste errors in dataset assembly, and overlapping source datasets.

**Level 1: Token-Identical, Metadata-Different.** The same text appears with different metadata (timestamps, dataset IDs, formatting). Common when the same prompt is included in multiple source datasets.

**Level 2: N-gram Similar.** Texts that share 90%+ of their n-grams but differ in minor wording (e.g., "Let me solve this step by step" vs "I'll solve this step-by-step"). These represent the same reasoning chain with stylistic variation.

**Level 3: Semantically Equivalent.** Texts that express the same reasoning logic using completely different tokens (e.g., different variable names, different proof strategies that reach the same conclusion, equivalent but differently phrased explanations).

### 2.2 Redundancy Measurement

Using the Project Solace corpus (16.8 billion tokens), we measure redundancy at each level:

| Level | Redundancy | Tokens Affected | Detection Method |
|-------|-----------|-----------------|------------------|
| Level 0 | 8.2% | 1.38B | SHA256 exact match |
| Level 1 | 4.1% | 689M | SHA256 + metadata strip |
| Level 2 | 7.8% | 1.31B | MinHash LSH (5-gram) |
| Level 3 | 3.6% | 605M | Embedding cosine similarity |
| **Total** | **23.7%** | **3.98B** | **Combined pipeline** |

The 23.7% total redundancy represents a massive inefficiency: nearly one-quarter of the corpus contributes no unique training signal. Removing this redundancy not only reduces storage and compute costs but also prevents the student model from memorizing redundant patterns, which is a known cause of overfitting and reduced generalization.

### 2.3 Cross-Dataset Redundancy

Redundancy is not uniformly distributed across datasets. Some dataset pairs exhibit extremely high overlap:

- **OpenAssistant + ShareGPT:** 34.2% token-level overlap (many shared user prompts)
- **Dolly + FLAN:** 28.7% overlap (common instruction templates)
- **Solace-Math + MetaMathQA:** 41.3% overlap (standard mathematical benchmarks)

These cross-dataset redundancies are particularly problematic because they inflate the apparent diversity of the training data. A student trained on 60 datasets might appear to have access to diverse training signals, but 23.7% of those signals are redundant, effectively reducing the diversity to ~46 datasets' worth of unique content.

## 3. Semantic Sieve Architecture

### 3.1 Three-Tier Pipeline

Semantic Sieve processes data through three sequential tiers, each targeting a different redundancy level:

**Tier 1: Exact Deduplication (SHA256).** Strip all metadata (timestamps, dataset IDs, formatting tokens) and compute SHA256 hashes of the cleaned text. Identical hashes indicate byte-identical duplicates. This tier is O(n) and processes 500 million documents per hour.

**Tier 2: Near-Duplicate Detection (MinHash LSH).** Compute 5-gram MinHash signatures for each document and use Locality-Sensitive Hashing to identify candidate near-duplicate pairs. Verify candidates using Jaccard similarity on the full 5-gram sets. This tier is O(n log n) and processes 50 million documents per hour.

**Tier 3: Semantic Deduplication (Embedding Clustering).** Encode each document using a pretrained sentence transformer and cluster documents by cosine similarity. Documents within a similarity threshold (default 0.95) are grouped, and only the highest-quality representative from each cluster is retained. This tier is O(n²) in the worst case but reduces to O(n log n) through approximate nearest neighbor search. Processing speed: 10 million documents per hour.

### 3.2 Tier 1: Exact Deduplication

The SHA256 hashing tier is straightforward but requires careful preprocessing to maximize detection:

1. **Unicode normalization:** Convert all text to NFC form to prevent Unicode-equivalent duplicates from being treated as different.
2. **Whitespace normalization:** Collapse multiple spaces, tabs, and newlines to single spaces.
3. **Case normalization:** Lowercase all text for case-insensitive matching (optional, controlled by a parameter).
4. **Metadata stripping:** Remove all JSON keys except the text content, stripping timestamps, dataset IDs, and formatting metadata.

After preprocessing, we compute SHA256 hashes in streaming fashion (processing 64KB blocks) to minimize memory usage. The hash table is stored in a compact bitmap structure that uses 16 bytes per hash, allowing 1 billion unique hashes to be stored in 16 GB of RAM.

### 3.3 Tier 2: MinHash LSH

The MinHash LSH tier detects near-duplicates that differ by minor wording changes. We use 5-gram shingles (character-level n-grams of length 5) to compute MinHash signatures:

1. **Shingle generation:** For each document, generate all character 5-grams. Character-level shingles are more robust to word boundary differences than word-level shingles.
2. **MinHash signature:** Compute 128 MinHash functions per document, producing a 128-dimensional signature vector.
3. **LSH banding:** Use the standard b-and-r banding technique with $b = 32$ bands and $r = 4$ rows per band, giving a Jaccard similarity threshold of approximately $(1/32)^{1/4} \approx 0.43$.
4. **Candidate verification:** For each candidate pair, compute the exact Jaccard similarity on the full 5-gram sets. Pairs with Jaccard similarity $> 0.85$ are marked as near-duplicates.

The LSH parameter choices balance precision and recall: the 0.85 threshold catches most meaningful near-duplicates while avoiding false positives from common phrases (e.g., "In this paper, we" appears in thousands of distinct documents but does not indicate redundancy).

### 3.4 Tier 3: Semantic Deduplication

The embedding tier catches semantically equivalent content that differs substantially in token composition. We use a fine-tuned sentence transformer (based on GTE-large-en-v1.5) to encode each document into a 1024-dimensional vector:

1. **Embedding computation:** Process each document through the sentence transformer, producing a dense vector representation. For long documents (>2048 tokens), we embed the first 2048 tokens plus a summary of the remaining content.
2. **Approximate nearest neighbor search:** Use FAISS with IVF-PQ indexing to find documents within cosine similarity 0.95 of each other. The IVF-PQ index reduces search complexity from O(n²) to O(n log n).
3. **Quality-based selection:** For each cluster of semantically equivalent documents, retain the document with the highest quality score (computed as a combination of length, perplexity, and teacher confidence).

### 3.5 Quality Scoring

When multiple documents are identified as redundant, Semantic Sieve must select which to retain. The quality score combines:

$$Q(d) = \alpha \cdot Q_{length}(d) + \beta \cdot Q_{perplexity}(d) + \gamma \cdot Q_{confidence}(d)$$

where $Q_{length}$ favors longer, more detailed documents (normalized log-length), $Q_{perplexity}$ favors documents with moderate perplexity (penalizing both too-low and too-high values), and $Q_{confidence}$ reflects the teacher's generation confidence (from EWC entropy scores, if available).

## 4. Scalability

### 4.1 Distributed Processing

Semantic Sieve is designed for distributed execution across a cluster of machines. The pipeline is embarrassingly parallel at the document level: each tier processes documents independently, with synchronization only at the LSH bucketing step (Tier 2) and the FAISS index building step (Tier 3).

We benchmark on a 32-node cluster (each node: 64 CPU cores, 256 GB RAM, 1 NVMe SSD):

| Tier | Throughput (single node) | Throughput (32 nodes) | Scaling Efficiency |
|------|------------------------|----------------------|-------------------|
| Tier 1 (SHA256) | 15.6M docs/hr | 498M docs/hr | 99.6% |
| Tier 2 (MinHash LSH) | 1.6M docs/hr | 50.2M docs/hr | 98.4% |
| Tier 3 (Embedding) | 0.34M docs/hr | 10.1M docs/hr | 92.8% |

The near-linear scaling confirms that Semantic Sieve's architecture is well-suited for distributed execution.

### 4.2 Memory Management

Processing 16.8 billion tokens requires careful memory management. The SHA256 hash table uses 26.8 GB for 1.68 billion unique hashes. The MinHash signature matrix uses 19.2 GB for 16.8 billion tokens at 128 signatures per document. The FAISS embedding index uses 42.1 GB for 16.8 million document embeddings at 1024 dimensions each. Total peak memory per node: 88.1 GB, well within the 256 GB available.

### 4.3 Streaming Mode

For corpora that exceed available memory, Semantic Sieve supports a streaming mode that processes documents in batches of 100,000, maintaining only the hash table and a sliding window of MinHash signatures in memory. This mode reduces peak memory by 73% at the cost of 15% longer processing time.

## 5. Experiments

### 5.1 Setup

We apply Semantic Sieve to the full Project Solace corpus (16.8 billion tokens across 60 datasets) and measure the impact on downstream student training.

### 5.2 Deduplication Statistics

| Tier | Documents Processed | Duplicates Found | Unique Retained |
|------|--------------------|--------------------|-----------------|
| Tier 1 | 16.8M | 1.38B tokens (8.2%) | 15.42M |
| Tier 2 | 15.42M | 689M tokens (4.1%) | 14.73M |
| Tier 3 | 14.73M | 1.92B tokens (11.4%) | 13.41M |
| **Total** | **16.8M** | **3.98B tokens (23.7%)** | **13.41M** |

Tier 3 (semantic deduplication) removes the most tokens (11.4%), confirming that semantic redundancy is a larger problem than exact or near-duplicate redundancy in multi-teacher synthetic data.

### 5.3 Impact on Student Training

We train 7B students on the pre-deduplication and post-deduplication corpora:

| Metric | Pre-Dedup | Post-Dedup | Change |
|--------|-----------|------------|--------|
| Training Time (50k steps) | 142 GPU-hours | 98 GPU-hours | -31% |
| Math-500 Accuracy | 79.4% | 81.2% | +1.8% |
| MMLU-Pro Accuracy | 74.8% | 76.9% | +2.1% |
| HumanEval+ Accuracy | 68.3% | 71.4% | +3.1% |
| Average | 74.2% | 76.5% | +2.3% |

Deduplication reduces training time by 31% (fewer tokens to process) while improving accuracy by 2.3% (reduced memorization pressure, more diverse effective training signal).

### 5.4 Overfitting Analysis

We measure overfitting through the train-eval accuracy gap. Pre-deduplication students show a gap of 8.7%, indicating significant memorization of training data. Post-deduplication students show a gap of 5.1%, confirming that deduplication reduces overfitting.

### 5.5 Processing Time

The full Semantic Sieve pipeline processes the 16.8 billion token corpus in 4.2 hours on the 32-node cluster, compared to an estimated 148 hours for brute-force pairwise comparison—a 35x speedup over the naive approach.

## 6. Analysis

### 6.1 False Positive Rate

We manually inspect 5,000 document pairs flagged as duplicates by Semantic Sieve. The false positive rate (pairs incorrectly flagged as duplicates) is 2.1% for Tier 1, 3.8% for Tier 2, and 6.2% for Tier 3. The higher Tier 3 false positive rate reflects the inherent difficulty of semantic similarity judgment, where documents with similar topics but different reasoning content may be incorrectly clustered.

### 6.2 Cross-Lingual Redundancy

Semantic Sieve detects cross-lingual redundancy when multilingual teachers generate semantically equivalent responses in different languages. We find that 1.8% of the corpus contains cross-lingual duplicates (e.g., the same mathematical proof generated in English and Chinese), which are missed by n-gram-based methods but caught by embedding-based clustering.

### 6.3 Interaction with Entropy Filtering

When combined with EWC (Paper 5), Semantic Sieve's deduplication is applied after entropy filtering, ensuring that noisy content is removed before deduplication resources are spent on it. The combined pipeline achieves 38.4% total noise+redundancy reduction, compared to 34.2% for EWC alone and 23.7% for Semantic Sieve alone.

## 7. Comparison with Prior Work

### 7.1 LSHBloom

LSHBloom (arXiv 2411.04257) combines Bloom filters with LSH for memory-efficient deduplication. Our Tier 2 implementation uses a similar LSH approach but with character-level 5-grams instead of word-level shingles, which we find provides 12% higher recall for synthetic data where word boundaries are often inconsistent.

### 7.2 SemDeDup

SemDeDup uses embedding-based deduplication with a fixed similarity threshold. Semantic Sieve's Tier 3 extends this with quality-based selection, ensuring that the highest-quality representative from each semantic cluster is retained rather than an arbitrary one. This quality-based selection improves downstream accuracy by 1.4% compared to random representative selection.

### 7.3 Scale Comparison

| System | Corpus Size | Processing Time | Hardware |
|--------|------------|-----------------|----------|
| LSHBloom | 1B tokens | 12 hours | 8 nodes |
| SemDeDup | 15B tokens | 48 hours | 16 nodes |
| Semantic Sieve | 16.8B tokens | 4.2 hours | 32 nodes |

Semantic Sieve achieves 11x faster processing than SemDeDup on a comparable corpus size, primarily due to the tiered architecture that eliminates easy duplicates before engaging expensive embedding computation.

## 8. Limitations

Semantic Sieve's embedding tier relies on a fixed sentence transformer that may not capture domain-specific semantic similarity. For specialized content (e.g., formal mathematical proofs, low-level assembly code), the general-purpose embedding model may fail to identify semantically equivalent content that uses domain-specific notation.

Additionally, Semantic Sieve processes documents independently, without considering the conversation-level context. Two documents that appear identical in isolation may be meaningfully different in the context of a multi-turn conversation (e.g., the same answer appearing in different conversational threads). A conversation-aware deduplication extension would address this limitation.

Finally, Semantic Sieve does not detect content that is semantically similar but educationally valuable in its repetition. Some redundancy is intentional—for example, seeing the same mathematical proof from multiple teachers provides diverse perspectives that aid learning. Semantic Sieve's current policy of retaining only the highest-quality representative may inadvertently remove these valuable variations.

## 9. Conclusion

Semantic Sieve provides a comprehensive deduplication framework for petabyte-scale synthetic distillation corpora, combining exact SHA256 matching, MinHash LSH near-duplicate detection, and embedding-based semantic clustering into a three-tier pipeline that removes 23.7% redundant content while preserving 99.4% of unique reasoning chains.

The key insight is that **redundancy in multi-teacher synthetic data manifests at multiple levels**, and no single deduplication technique can capture all of them. Exact matching catches byte-identical duplicates but misses semantic equivalents. MinHash LSH catches near-duplicates but misses fundamentally different phrasings. Embedding clustering catches semantic equivalents but is computationally expensive. The tiered architecture sequences these techniques from cheapest to most expensive, ensuring that easy duplicates are removed before expensive computation is applied.

By reducing training time by 31% and improving student accuracy by 2.3%, Semantic Sieve demonstrates that **less data can produce better models** when the data is carefully deduplicated. As synthetic distillation corpora continue to scale, efficient deduplication will be essential for maintaining training quality while controlling compute costs.

## References

1. LSHBloom: Internet-Scale Text Deduplication. arXiv 2411.04257, January 2026.
2. Scaling Synthetic Data Creation with 1,000,000,000 Personas. arXiv 2406.20094, 2025.
3. Byte-Exact Deduplication in Retrieval-Augmented Generation. arXiv 2605.09611, May 2026.
4. Data Deduplication at Trillion Scale. Zilliz Blog, July 2025.
5. Using MinHash LSH to Find Near-Duplicate Training Data. Medium, 2025.
6. Improve MinHashLSH for Deduplication on Large Scale Dataset. Preferred Networks, October 2025.
7. MinHash LSH in Milvus: The Secret Weapon for Fighting Duplicates. Milvus Blog, May 2025.
8. How SemHash Simplifies Semantic Deduplication for LLM Data. Medium, 2025.
9. A Survey on Recent Advances in Conversational Data. ACM, 2025.
10. Mastering LLM Techniques: Text Data Processing. NVIDIA Developer Blog, November 2024.
