# PlanckGPT

PlanckGPT (planck length reference :D) is my attempt to make a tiny language model from scratch mostly for fun and educational purposes, but also to see how far a consumer-level computer can go in AI development **from scratch**. It has about 206m parameters and is pretrained on roughly 2 billion tokens of the Fineweb-edu dataset. This is small compared to modern LLMs' standards, and it only does next token prediction, but you can definitely train this on a mid-range card for just 1-2 days. Its performance should match that of a GPT2-small, with ~3.0793 average val loss and ~0.9538 bpb val loss on Fineweb-edu.

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

Of course, you should already install compatible CUDA and Python versions, I currently use Python 3.14 and CUDA 13.

## Running PlanckGPT

1. Download the latest model (`planckgpt.pth`) in the releases page and place it in `./artifacts/`.
2. Simply run:
```sh
python -m scripts.inference
```

A prompt will appear for you to chat with the model.

## Pretraining

To pretrain the model from scratch, run:
```sh
python -m scripts.pretrain
```

The model will train with ~2b tokens/20 100m-token segments (estimated 28 hours on my Laptop RTX 5070 Mobile), and after each epoch it will save the current model to `./artifacts/planckgpt.pth` and a checkpoint containing the model and training state in `.artifacts/checkpoints/`.

You can then evaluate the model by running:
```sh
python -m scripts.evaluate
```

For more control, you can modify the scripts in `./scripts/`, and you shall see a marked section for configuration in each file.

## Architecture

Currently it uses:

* Tokenizer: Tiktoken with GPT-2 encoding (50,257 vocab size).
* Embedding: 896-dimensional token embedding.
* Rotary positional embedding.
* Transformer: 14 decoder layers, 7 query heads, 3584 ffn dim, 896 embedding dim.
* Multi-Query Attention.
* Squared ReLU for activation.
* RMSNorm without learnable params, notably used on QK, transformer (how you would expect), embedding, and output logits.
* Output: Linear projection with softcap logits (-15, 15).

and is pretrained with:

* Dataset: Fineweb-edu (~2b tokens).
* Context Window: 1024 tokens.
* Batch Size: 4 (effective batch size: 512 with gradient accumulation).
* NorMuon optimizer for transformer weights, 8-bit Adam optimizer for embedding and output projection.
* Stable LR for the first 65% of the steps (40 first steps have warmup), LinearLR decay to 5% of base LR for the rest.
* Cautious weight decay inspired by nanochat.
* BF16 mixed precision training and other Blackwell-specific features.
* Training with torch.compile.
* Gradient checkpointing.

and generates text with:

* Top-k sampling (k=50) and top-p sampling (p=0.95) right after.
* Temperature: 1.0.
* Context Window: 1024 tokens.
* Repetition penalty: 1.2 on 64-token window.
* Stopping: EOS token or fixed limit (1024 by default).
* KV cache for faster inference.

The current configuration is designed to squeeze out the best possible performance out of an 8gb 5070 Mobile, you can change the configs to match your card.

## Potential todos

These are things I might implement in the future:

* Training improvements:
  * Try out different pretraining datasets, e.g. ClimbMix.
  * Try Gram Newton-Schulz to improve Muon's speed.
  * Use up-to-date Flash Attention implementation.
  * Support FP8 and potentially NVFP4 training.
  * Tune hyperparameters further.
* Architecture improvements:
  * Interesting idea to try out: Overwhelmingly large vocab like Gemma-3-270m which might help with small models.
  * Custom tokenizer.
  * Value embeddings.
  * Dynamic scales for some layers.
  * Mamba? RWKV? MoE?
  * Sliding window attention.
  * Smear.
  * Backout.
* Potential issues to look after:
  * The current data to params ratio still needs more tuning.
  * Current stable range might be too high.
  * Some data segments might be noisy.
  * Embedding might be unstable currently due to AdamW8bit.
  * Weight decay might need more tuning.
* Finetuning for multiple purposes.
* Try different datasets for both pretraining and finetuning.
* Export to multiple formats for inference.
* Code refactoring.

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
