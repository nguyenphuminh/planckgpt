import torch
from model import ChatBot
from data import load_data

if __name__ == "__main__":
    torch.cuda.empty_cache()

    # Initialize model
    chatbot = ChatBot()
    chatbot = torch.compile(chatbot, mode="max-autotune", dynamic=False)

    print(f"Using device: {chatbot.device}")
    print(f"Model parameters: {sum(p.numel() for p in chatbot.parameters()):,}")

    # Pretrain
    chatbot.train_model(load_data())
    
    # Final save
    print("Final save to chatbot.pth")
    chatbot.save()
