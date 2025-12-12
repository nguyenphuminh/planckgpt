import torch
import os
from model import ChatBot
from data import load_sft_data

if __name__ == "__main__":
    torch.cuda.empty_cache()

    # Initialize model
    chatbot = ChatBot()
    chatbot = torch.compile(chatbot, mode="max-autotune")

    print(f"Using device: {chatbot.device}")
    print(f"Model parameters: {sum(p.numel() for p in chatbot.parameters()):,}")
    
    # Load existing model to finetune
    if os.path.exists("./chatbot_continue.pth"):
        print("Found model to continue training from")
        chatbot.load("./chatbot_continue.pth")
    else:
        raise RuntimeError("No existing model found to finetune")

    # Finetune
    chatbot.train_model(
        load_sft_data(),
        adam_lr=0.00016,
        muon_lr=0.0012,
        stable_range=0.0,
        max_decay=0.0,
        total_steps=20
    )
    
    # Final save
    print("Final save to chatbot.pth")
    chatbot.save()
