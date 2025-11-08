# PlanckGPT

PlanckGPT is my attempt on making a tiny language model (planck length refence :D) from scratch mostly for fun and educational purposes, but also to see how far a consumer-level computer can go. It has about 150m parameters and is trained on roughly 3 billion tokens of the Fineweb dataset. This is small compared to modern LLMs' standards, which also explains why it is goofy when you use it (lol), but you can definitely train this on a mid-range card for just 1-2 days, and it can still generate proper English and data that should be related to the user's prompt (its performance roughly equals to that of GPT2).

## Setup

Setup venv and install necessary packages:

```sh
# Create and activate venv
python -m venv venv
# Run this every time you start
source venv/scripts/activate
# or "./venv/scripts/activate" if you are on windows

# Install packages (once)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129
pip install tiktoken datasets bitsandbytes
```

Of course, you should already install compatible CUDA and Python versions, I currently use Python 3.13 and CUDA 13 (which is compatible with CUDA 12.9 mentioned above).

## Running PlanckGPT

1. Download the latest model (`chatbot.pth`) in the releases page.
2. Simply run:
```sh
python inference.py
```

A prompt will appear for you to chat with the model.

If you want to run the pretrained model only with no finetuning at all, then simply download `chatbot_pretrained.pth` from the releases page and move it to this directory. You have to do this because the pretrained model does not have user or assistant distinctions and have to be treated differently.

## Pretraining

To pretrain the model from scratch, run:
```sh
python train.py
```

The model will train with 3b+ tokens with 20 150m-token segments (estimated 45 hours on my Laptop RTX 5070), and after each epoch it will save the current model to `./chatbot.pth`.

## Finetuning

PlanckGPT is finetuned with the Smol-smoltalk set, which has roughly 430m tokens.

To finetune, simply rename your model file into `chatbot_continue.pth` and run:
```sh
python finetune.py
```

It will save the finetuned model to `./chatbot.pth` just like pretraining does.

## Architecture

Currently it uses:

* Tokenizer: Tiktoken with GPT-2 encoding (50,257 vocab size).
* Embedding: 768-dimensional token embedding.
* Rotary positional embedding.
* Transformer: 12 decoder layers, 6 heads, 3072 d_ffn, 768 d_model.
* Multi-Query Attention with flash attention support (sdpa).
* Squared ReLU for activation.
* RMSNorm without learnable params for normalization, applied how you would expect, but also used on QK, embedding, and before output projection.
* Output: Linear layer to vocabulary.

and is trained with:

* Dataset: Fineweb (~3b tokens) with no overlapping.
* Context Window: 1024 tokens.
* Batch Size: 4 (effective batch size: 512 with gradient accumulation).
* Muon optimizer for transformer weights, 8-bit AdamW optimizer for embedding and output projection.
* Stable LR for the first 55% of the steps, LinearLR decay to 0.1x for the rest.
* BF16 mixed precision training and other Blackwell-specific features.
* Training with torch.compile on "max-autotune" mode.
* Gradient checkpointing in 2/3 of the transformer layers.

and is finetuned with:

* Dataset: Smol-smoltalk (~430m tokens) with no overlapping.
* Same configuration as pretraining.

and generates text with:

* Sampling: Top-k sampling (k=50).
* Temperature: 0.7.
* Context Window: 1024 tokens.
* Stopping: EOS token for fixed limit (10240 by default).
* Simple repetition penalty with 64 latest tokens.

The current configuration is designed to squeeze out the best possible performance out of an 8gb 5070, you can change the configs to match your card.

## Copyrights and License

Copyrights © 2025 Nguyen Phu Minh.

This project is licensed under the Apache 2.0 License.
