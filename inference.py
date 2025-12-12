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
            print("/enablemem  - Enable chat memory (disabled by default)")
            print("/disablemem - Disable chat memory")
            print("/clearmem   - Clear chat memory (use this if it hallucinates)")
            print("/showmem    - Log out chat memory")
        elif prompt == "/clear":
            print("\033c", end="")
        elif prompt == "/enablemem":
            is_mem_enabled = True
            print("Memory enabled")
        elif prompt == "/disablemem":
            is_mem_enabled = False
            print("Memory disabled")
        elif prompt == "/clearmem":
            print("Memory cleared")
            memory = []
        elif prompt == "/showmem":
            print(chatbot.tokens_to_text(memory))
        else:
            current_memory = memory if is_mem_enabled else []

            print("Assistant:", end="")
            if is_pretrained:
                new_memory = chatbot.generate(prompt, memory=current_memory)
            else:
                new_memory = chatbot.generate(f"User: {prompt}\nAssistant: ", memory=current_memory)

            if is_mem_enabled:
                memory = new_memory

            print("\033[0m", end="")
