---
title: "Fine-Tuning Student Models with Solace 1.0"
description: "Step-by-step recipe for fine-tuning sub-8B base models (Llama 3.1 8B, Qwen 2.5 7B) using Axolotl, Unsloth, and TRL."
category: "guides"
order: 1
lastUpdated: 2026-08-27
specs:
  "Recommended Base Models": "Qwen 2.5 7B Base, Llama 3.1 8B Base"
  "Hardware Minimum": "1x NVIDIA RTX 3090 / 4090 (24GB)"
  "Training Duration": "~14 hours (1 epoch on 630M tokens with LoRA)"
supportedFormats:
  - "Axolotl"
  - "Unsloth"
  - "Hugging Face TRL"
---

## Overview

Fine-tuning a base model on Solace 1.0 imparts structured reasoning capabilities without requiring pre-training scale compute. This guide demonstrates how to fine-tune `Qwen/Qwen2.5-7B` using Unsloth on a single 24GB consumer GPU.

---

## 1. Quickstart with Unsloth (Single 24GB GPU)

Install dependencies:

```bash
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps trl peft accelerate bitsandbytes
```

Training script (`train_solace.py`):

```python
import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

max_seq_length = 8192
load_in_4bit = True

# 1. Load base model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen2.5-7B",
    max_seq_length=max_seq_length,
    load_in_4bit=load_in_4bit,
)

# 2. Add LoRA adapters targeting all linear layers
model = FastLanguageModel.get_peft_model(
    model,
    r=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=64,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

# 3. Load Solace 1.0 Parquet dataset
dataset = load_dataset(
    "Solstice-AI/Solace-1.0-GLM5.2-Fable5-GPT5.6Sol-DeepSeekV4Pro0813-Qwen3.8Max-KimiK3-Manus",
    split="train[:50000]"  # Use subset or full split
)

def format_prompts(examples):
    texts = []
    for p, c, s in zip(examples['prompt'], examples['thought_chain'], examples['solution']):
        text = f"<|im_start|>system\nYou are Solace, a verified reasoning model.<|im_end|>\n<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n<think>\n{c}\n</think>\n{s}<|im_end|>"
        texts.append(text)
    return {"text": texts}

dataset = dataset.map(format_prompts, batched=True)

# 4. Trainer Configuration
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=4,
    packing=True,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_ratio=0.03,
        num_train_epochs=1,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=42,
        output_dir="solace-student-checkpoints",
    ),
)

trainer.train()

# 5. Export quantized GGUF and FP16 weights
model.save_pretrained_merged("solace-qwen-7b-finetuned", tokenizer, save_method="merged_16bit")
model.save_pretrained_gguf("solace-qwen-7b-gguf", tokenizer, quantization_method="q4_k_m")
```

---

## 2. Recommended Hyperparameters

| Parameter | LoRA (Consumer GPU) | Full Parameter (8x H100) |
| :--- | :--- | :--- |
| **Learning Rate** | $2.0 \times 10^{-4}$ | $2.5 \times 10^{-5}$ |
| **LR Scheduler** | Cosine with 3% Warmup | Cosine with 2% Warmup |
| **Effective Batch Size** | 64 sequences | 256 sequences |
| **Sequence Length** | 8,192 tokens | 32,768 tokens |
| **Weight Decay** | 0.01 | 0.1 |
| **Precision** | BF16 / 4-bit Base | Native FP8 / BF16 |
