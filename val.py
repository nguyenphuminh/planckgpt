import jax
import flax.nnx as nnx
from model import JAXGPT
from data import load_val_data

if __name__ == "__main__":
    # Initialize model
    gpt = JAXGPT.load()

    print(f"Model parameters: {sum(x.size for x in jax.tree_util.tree_leaves(nnx.state(gpt))):,}")

    # Validate
    gpt.validate_model(load_val_data())
