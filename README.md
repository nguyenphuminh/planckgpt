# PlanckGPT

PlanckGPT (planck length reference :D) is my attempt to make a tiny language model from scratch mostly for fun and educational purposes, but also to see how far a consumer-level computer can go in AI development. It has about 150m parameters and is pretrained on roughly 3 billion tokens of the Fineweb-edu dataset and finetuned with specifications I will mention below. This is small compared to modern LLMs' standards, which also explains why it is goofy when you use it (lol), but you can definitely train this on a mid-range card for just 1-2 days, and it can still generate proper English and data that should be related to the user's prompt (its pretrain performance roughly matches that of GPT2 just so you know). To really squeeze out every little performance gain, I'm using JAX rather than Pytorch.

## Setup

Setup venv and install necessary packages:
```sh
# Create and activate venv
python -m venv venv
# Run this every time you start
source venv/bin/activate
# or "./venv/scripts/activate" if you are on windows

# Install packages (once)
pip install jax-ai-stack "jax[cuda]" tiktoken datasets
```

Of course, you should already install compatible CUDA and Python versions, I currently use Python 3.13 and CUDA 13.

## Running PlanckGPT

1. Download the latest model (`jaxgpt.npz`) in the releases page.
2. Simply run:
```sh
python inference.py
```

A prompt will appear for you to chat with the model.

If you want to run the pretrained model only with no finetuning at all, then simply download `jaxgpt_pretrained.npz` from the releases page and move it to this directory. You have to do this because the pretrained model does not have user or assistant distinctions and have to be treated differently.

## Pretraining

To pretrain the model from scratch, run:
```sh
python train.py
```

The model will train with 3b+ tokens with 20 150m-token segments (estimated 40 hours on my Laptop RTX 5070 Mobile), and after each epoch it will save the current model to `./jaxgpt.npz`.

## Finetuning

To finetune, simply rename your model file into `jaxgpt_continue.npz` and run:
```sh
python midtrain.py
```

This will run the mid-training process, and it will save the finetuned model to `./jaxgpt.npz` just like pretraining does. When finished, rename it back to `jaxgpt_continue.npz` and run:
```sh
python sft.py
```

The output model is also in `./jaxgpt.npz`.

## Architecture

Currently it uses:

* Tokenizer: Tiktoken with GPT-2 encoding (50,257 vocab size).
* Embedding: 768-dimensional token embedding.
* Rotary positional embedding.
* Transformer: 12 decoder layers, 6 heads, 3072 d_ffn, 768 d_model.
* Multi-Query Attention with JAX's dot_product_attention.
* Squared ReLU for activation.
* RMSNorm without learnable params for normalization, applied how you would expect, but also used on QK, embedding, and before output projection.
* Output: Linear layer to vocabulary.
* JAX.

and is trained with:

* Dataset: Fineweb-edu (~3b tokens).
* Context Window: 1024 tokens.
* Batch Size: 4 (effective batch size: 512 with gradient accumulation).
* Muon optimizer for transformer weights, Adam optimizer for embedding and output projection.
* Stable LR for the first 55% of the steps, LinearLR decay to 0.1x for the rest.
* Full BF16 training and other Blackwell-specific features.
* No gradient checkpointing (but you can configure it if you want).

and is finetuned with:

* Midtrain:
    * Smol-smoltalk (460k rows).
    * MMLU (100k rows).
    * GSM8K (8k rows).
    * Custom identity json (1000 rows, repeated 4 times, generated from Claude Sonnet 4.5).
    * Same configuration as pretraining, but with 80% stable LR range.
* SFT:
    * Smol-smoltalk (10k rows).
    * Arc-Easy (2300 rows).
    * Arc-Challenge (1100 rows).
    * Custom identity json (similar to above, but repeated 2 times).
    * Same configuration as pretraining, but LR decays right from the start of training and decays to 0.

and generates text with:

* Sampling: Top-k sampling (k=50).
* Temperature: 0.7.
* Context Window: 1024 tokens.
* Stopping: EOS token for fixed limit (4096 by default).
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
