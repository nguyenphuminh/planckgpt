import torch
from model import ChatBot
from data import load_data, load_val_data

if __name__ == "__main__":
    torch.cuda.empty_cache()

    # Initialize model
    chatbot = ChatBot()
    chatbot.load()
    chatbot = torch.compile(chatbot, mode="max-autotune", dynamic=False)

    print(f"Using device: {chatbot.device}")
    print(f"Model parameters: {sum(p.numel() for p in chatbot.parameters()):,}")

    # Pretrain
    avg_val_loss, val_perplexity = chatbot.evaluate(
        load_val_data()
    )

    print(f", Val Loss: {avg_val_loss:.4f}, Val Perplexity: {val_perplexity:.2f}")
