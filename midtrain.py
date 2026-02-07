import torch
import os
from model import ChatBot
from data import load_midtrain_data

if __name__ == "__main__":
    torch.cuda.empty_cache()

    # Initialize model
    chatbot = ChatBot()
    chatbot = torch.compile(chatbot, mode="max-autotune", dynamic=False)

    print(f"Using device: {chatbot.device}")
    print(f"Model parameters: {sum(p.numel() for p in chatbot.parameters()):,}")
    
    # Load existing model to train
    if os.path.exists("./chatbot_continue.pth"):
        print("Found model to continue training from")
        chatbot.load("./chatbot_continue.pth")
    else:
        raise RuntimeError("No existing model found to train")

    # Train
    chatbot.train_model(load_midtrain_data(), stable_range=0.8, total_steps=797)
    
    # Final save
    print("Final save to chatbot.pth")
    chatbot.save()
