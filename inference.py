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
    is_pretrained = False

    if os.path.exists("chatbot_pretrained.pth"):
        print("Loaded from chatbot_pretrained.pth")
        chatbot.load(path="chatbot_pretrained.pth")

        is_pretrained = True
    else:
        print("Loaded from chatbot.pth")
        chatbot.load()

    # Prompt
    memory = []
    is_mem_enabled = False

    while True:
        prompt = input("\n\n\033[32mPrompt (type \"/help\" to see a list of commands): ")

        print("\033[33m")

        if prompt == "/help":
            print("All commands:")
            print("/help       - View a list of commands")
            print("/clear      - Clear the console")
            print("/enablemem  - Enable chat memory, not recommended for now")
            print("/disablemem - Disable chat memory")
            print("/clearmem   - Clear chat memory")
        elif prompt == "/clear":
            print("\033c", end="")
        elif prompt == "/enablemem":
            is_mem_enabled = True
        elif prompt == "/disablemem":
            is_mem_enabled = False
        elif prompt == "/clearmem":
            memory = []
        else:
            memory = memory if is_mem_enabled else []

            print("Assistant:", end="")
            if is_pretrained:
                memory = chatbot.generate(prompt, memory=memory)
            else:
                memory = chatbot.generate(f"User: {prompt}\nAssistant: ", memory=memory)
            print("\033[0m", end="")
