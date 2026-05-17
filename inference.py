import torch
from model import GPT

if __name__ == "__main__":
    torch.cuda.empty_cache()

    # Initialize model
    gpt = GPT()
    gpt = torch.compile(gpt, mode="reduce-overhead")

    print(f"Using device: {gpt.device}")
    print(f"Model parameters: {sum(p.numel() for p in gpt.parameters()):,}")

    # Load model
    gpt.load()
    print("Loaded from planckgpt.pth\n")

    # Prompt
    while True:
        prompt = input("Prompt: ")
        stream = gpt.generate(prompt)

        try:
            while True:
                print(next(stream), end="")
        except StopIteration:
            print("\n")
