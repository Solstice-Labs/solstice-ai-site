---
title: "Quantization Without Degradation: Preserving Multi-Step Reasoning at INT4 & FP8"
description: "A practical study on preserving chain-of-thought integrity when compressing distilled models to sub-4-bit and 8-bit precision."
pubDate: 2026-08-28
author: "Solstice-AI Research"
tags:
  - "Quantization"
  - "Hardware"
  - "Inference"
readingTime: "6 min read"
takeaways:
  - "Standard post-training quantization often destroys fragile activation outliers in deep reasoning attention heads."
  - "Activation-aware calibration with Solace 1.0 traces retains 98.6% of FP16 accuracy on Math-500 in INT4 AWQ."
  - "GGUF and FP8 checkpoints are now available on Hugging Face."
featured: false
---

Quantizing reasoning models has notoriously high loss margins compared to conversational chatbots. Chain-of-thought tokens require high dynamic range to avoid error cascades.

In this technical note, we detail our activation calibration pipeline for producing zero-degradation FP8 and INT4 AWQ checkpoints.

```
FP16 Baseline Accuracy (Math-500):    84.2%
Standard RTN INT4:                    61.8%  (-22.4% drop)
Solace Calibrated AWQ INT4:           83.1%  (-1.1% delta)
```

Check out the full quantization recipes in our [Docs Guide](/docs/quantized-models-quickstart).
