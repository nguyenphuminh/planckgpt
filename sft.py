import jax
import flax.nnx as nnx
from model import JAXGPT
from data import load_sft_data

if __name__ == "__main__":
    # Initialize model
    gpt = JAXGPT.load(path="./jaxgpt_continue.npz")

    print(f"Model parameters: {sum(x.size for x in jax.tree_util.tree_leaves(nnx.state(gpt))):,}")

    # Finetune
    gpt.train_model(
        load_sft_data(),
        adam_lr=0.00016,
        muon_lr=0.0012,
        stable_range=0.0,
        max_decay=0.0,
        total_steps=20
    )
    
    # Final save
    print("Final save to jaxgpt.npz")
    gpt.save()
