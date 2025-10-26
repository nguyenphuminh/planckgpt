import torch
from model import ChatBot
from data import load_data, load_val_data

if __name__ == "__main__":
    torch.cuda.empty_cache()

    # Initialize model
    chatbot = ChatBot()
    chatbot = torch.compile(chatbot, mode="reduce-overhead")

    print(f"Using device: {chatbot.device}")
    print(f"Model parameters: {sum(p.numel() for p in chatbot.parameters()):,}")

    # Load
    print("Loaded from chatbot.pth")
    chatbot.load()

    # Validate
    chatbot.validate_model(load_val_data())