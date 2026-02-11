# PlanckGPT

PlanckGPT (planck length reference :D) is my attempt to make a tiny language model from scratch mostly for fun and educational purposes, but also to see how far a consumer-level computer can go in AI development. It has about 150m parameters and is pretrained on roughly 3 billion tokens of the Fineweb-edu dataset. This is small compared to modern LLMs' standards, and it only does next token prediction, but you can definitely train this on a mid-range card for just 1-2 days. Its performance should match that of a GPT2-small, with ~3.15 val loss on Fineweb-edu.

In previous versions, there are also finetuning code and a chat model, but I have decided to remove them for now due to low quality. There is on-going development for that in [this branch](https://github.com/nguyenphuminh/planckgpt/tree/big-fix-1) though.

## Setup

Setup venv and install necessary packages:
```sh
# Create and activate venv
python -m venv venv
# Run this every time you start
source venv/scripts/activate
# or "./venv/scripts/activate" if you are on windows

# Install packages (once)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install tiktoken datasets bitsandbytes
```

Of course, you should already install compatible CUDA and Python versions, I currently use Python 3.13 and CUDA 13.

## Running PlanckGPT

1. Download the latest model (`chatbot.pth`) in the releases page.
2. Simply run:
```sh
python inference.py
```

A prompt will appear for you to chat with the model.

## Pretraining

To pretrain the model from scratch, run:
```sh
python train.py
```

The model will train with 3b+ tokens with 20 150m-token segments (estimated 40 hours on my Laptop RTX 5070 Mobile), and after each epoch it will save the current model to `./chatbot.pth`.

## Architecture

Currently it uses:

* Tokenizer: Tiktoken with GPT-2 encoding (50,257 vocab size).
* Embedding: 768-dimensional token embedding.
* Rotary positional embedding.
* Transformer: 12 decoder layers, 6 heads, 3072 d_ffn, 768 d_model.
* Multi-Query Attention with merged qkv.
* Squared ReLU for activation.
* RMSNorm without learnable params for normalization, applied how you would expect, but also used on QK, embedding, and before output projection.
* Output: Linear layer to vocabulary.

and is pretrained with:

* Dataset: Fineweb-edu (~3b tokens).
* Context Window: 1024 tokens.
* Batch Size: 4 (effective batch size: 512 with gradient accumulation).
* NorMuon optimizer for transformer weights, 8-bit Adam optimizer for embedding and output projection.
* Stable LR for the first 55% of the steps, LinearLR decay to 0.1x for the rest.
* BF16 mixed precision training and other Blackwell-specific features.
* Training with torch.compile on "max-autotune" mode and `dynamic=False`.
* Gradient checkpointing in 1/3 of the transformer layers.

and generates text with:

* Sampling: Top-k sampling (k=50).
* Temperature: 0.8.
* Context Window: 1024 tokens.
* Stopping: EOS token for fixed limit (10240 by default).
* KV cache for faster inference.

The current configuration is designed to squeeze out the best possible performance out of an 8gb 5070 Mobile, you can change the configs to match your card.

## Acknowledgements

PlanckGPT is inspired by [`modded-nanogpt`](https://github.com/KellerJordan/modded-nanogpt) and [`nanochat`](https://github.com/karpathy/nanochat).

## Cite PlanckGPT

```bibtex
@misc{planckgpt,
  author = {Phu Minh Nguyen},
  title = {PlanckGPT: Train a GPT from scratch on your laptop},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/nguyenphuminh/planckgpt}
}
```

## Copyrights and License

Copyrights © 2025 Nguyen Phu Minh.

This project is licensed under the Apache 2.0 License.
