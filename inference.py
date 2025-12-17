import os
import jax
import flax.nnx as nnx
from model import JAXGPT
from data import load_sft_data

if __name__ == "__main__":
    # Initialize model
    is_pretrained = False

    # Load pretrained model and set flag if found
    if os.path.exists("jaxgpt_pretrained.npz"):
        print("Loaded from jaxgpt_pretrained.npz")
        gpt = JAXGPT.load(path="jaxgpt_pretrained.npz")

        is_pretrained = True
    else:
        print("Loaded from jaxgpt.npz")
        gpt = JAXGPT.load()

    print(f"Model parameters: {sum(x.size for x in jax.tree_util.tree_leaves(nnx.state(gpt))):,}")

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
            print(gpt.tokens_to_text(memory))
        else:
            current_memory = memory if is_mem_enabled else []

            print("Assistant:", end="")
            if is_pretrained:
                new_memory = gpt.generate(prompt, memory=current_memory)
            else:
                new_memory = gpt.generate(f"User: {prompt}\nAssistant: ", memory=current_memory)

            if is_mem_enabled:
                memory = new_memory

            print("\033[0m", end="")
