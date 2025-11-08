import torch
import os
from model import ChatBot

if __name__ == "__main__":
    torch.cuda.empty_cache()

    # Initialize model
    chatbot = ChatBot()
    chatbot = torch.compile(chatbot, mode="reduce-overhead")

    print(f"Using device: {chatbot.device}")
    print(f"Model parameters: {sum(p.numel() for p in chatbot.parameters()):,}")

    # Load
    if os.path.exists("chatbot_pretrained.pth"):
        print("Loaded from chatbot_pretrained.pth")
        chatbot.load(path="chatbot_pretrained.pth")
    else:
        print("Loaded from chatbot.pth")
        chatbot.load()

    # Prompt
    while True:
        prompt = input("\n\n\033[32mPrompt (type \"/help\" to see a list of commands): ")

        print("\033[33m")

        if prompt == "/help":
            print("All commands:")
            print("/help - View a list of commands")
            print("/clear - Clear the console")
        elif prompt == "/clear":
            print("\033c", end="")
        else:
            print("Assistant:", end="")
            chatbot.generate(f"User: {prompt}\nAssistant: ")
            print("\033[0m", end="")
