import torch
from model import GPT
from data import load_data, load_val_data

if __name__ == "__main__":
    torch.cuda.empty_cache()

    # Initialize model
    gpt = GPT()
    gpt = torch.compile(gpt, mode="max-autotune", dynamic=False)

    print(f"Using device: {gpt.device}")
    print(f"Model parameters: {sum(p.numel() for p in gpt.parameters()):,}")

    # Pretrain
    gpt.train_model(
        load_data(),
        [*load_val_data()] # Loads one time into memory
    )
    
    # Final save
    print("Final save to planckgpt.pth")
    gpt.save()
