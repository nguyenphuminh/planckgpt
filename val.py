import torch
from model import GPT
from data import load_val_data

if __name__ == "__main__":
    torch.cuda.empty_cache()

    # Initialize model
    gpt = GPT()
    gpt.load()
    gpt = torch.compile(gpt, mode="max-autotune", dynamic=False)

    print(f"Using device: {gpt.device}")
    print(f"Model parameters: {sum(p.numel() for p in gpt.parameters()):,}")

    # Evaluate
    avg_val_loss, val_perplexity, val_bpb = gpt.evaluate(
        load_val_data()
    )

    print(f", Val Loss: {avg_val_loss:.4f}, Val BPB: {val_bpb:.4f}, Val Perplexity: {val_perplexity:.2f}")
