import jax
import flax.nnx as nnx
from model import JAXGPT
from data import load_data

if __name__ == "__main__":
    # Initialize model
    gpt = JAXGPT()

    print(f"Model parameters: {sum(x.size for x in jax.tree_util.tree_leaves(nnx.state(gpt))):,}")

    # Pretrain
    gpt.train_model(load_data())

    # Final save
    print("Final save to jaxgpt.npz")
    gpt.save()
