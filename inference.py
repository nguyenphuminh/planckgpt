import torch
from model import ChatBot

if __name__ == "__main__":
    torch.cuda.empty_cache()

    # Initialize model
    chatbot = ChatBot()
    chatbot = torch.compile(chatbot, mode="reduce-overhead")

    print(f"Using device: {chatbot.device}")
    print(f"Model parameters: {sum(p.numel() for p in chatbot.parameters()):,}")

    # Load model
    chatbot.load()
    print("Loaded from chatbot.pth\n")

    # Prompt
    while True:
        prompt = input("Prompt: ")
        stream = chatbot.generate(prompt)

        try:
            while True:
                print(next(stream), end="")
        except StopIteration:
            print("\n")
