import jax
import flax.nnx as nnx
from model import JAXGPT
from data import load_midtrain_data

if __name__ == "__main__":
    # Initialize model
    gpt = JAXGPT.load(path="./jaxgpt_continue.npz")

    print(f"Model parameters: {sum(x.size for x in jax.tree_util.tree_leaves(nnx.state(gpt))):,}")

    # Finetune
    gpt.train_model(load_midtrain_data(), stable_range=0.8, total_steps=797)
    
    # Final save
    print("Final save to jaxgpt.npz")
    gpt.save()
