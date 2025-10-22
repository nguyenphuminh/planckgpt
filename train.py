import torch
import os
from model import ChatBot
from data import load_data

if __name__ == "__main__":
    torch.cuda.empty_cache()

    # Initialize model
    chatbot = ChatBot()
    chatbot = torch.compile(chatbot, mode="max-autotune")

    print(f"Using device: {chatbot.device}")
    print(f"Model parameters: {sum(p.numel() for p in chatbot.parameters()):,}")
    
    # Load existing model to continue training if exists
    if os.path.exists("./chatbot_continue.pth"):
        print("Found model to continue training from")
        chatbot.load("./chatbot_continue.pth")

    # Pretrain
    chatbot.train_model(load_data())
    
    # Final save
    print("Final save to chatbot.pth")
    chatbot.save()
