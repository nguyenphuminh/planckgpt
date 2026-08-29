import math
import torch
from torch.amp import autocast
from model.core import GPT
from data.finetune_mix import load_val_data


# Cache token->bytes lookups so repeated calls don't rebuild them
_token_bytes_cache = {}


def evaluate(gpt, val_data_loader, eval_batches=40):
    val_loss = 0
    val_tokens = 0
    val_bytes = 0

    # Build token->bytes lookup for BPB
    cache_key = (gpt.vocab_size, str(gpt.device))
    token_bytes = _token_bytes_cache.get(cache_key)
    if token_bytes is None:
        token_bytes = torch.zeros(gpt.vocab_size, dtype=torch.int64, device=gpt.device)
        for token_id in range(gpt.vocab_size):
            token_bytes[token_id] = len(gpt.encoding.decode([token_id]).encode("utf-8"))
        _token_bytes_cache[cache_key] = token_bytes

    # Accept either an iterable of batches or a loader factory. The packing loader
    # is a generator, so it is single use and has to be rebuilt for every call.
    val_batches = val_data_loader() if callable(val_data_loader) else val_data_loader

    with torch.no_grad():
        for i, (rows, masks, _) in enumerate(val_batches):
            # Only score a fixed number of batches, the val set is large
            if i >= eval_batches:
                break

            batch_sequences = torch.tensor(rows, dtype=torch.long, device=gpt.device)
            batch_mask = torch.tensor(masks, dtype=torch.bool, device=gpt.device)

            input_tokens = batch_sequences[:, :-1]
            # Positions the assistant is not responsible for become -1, which is
            # the ignore_index core.py passes to cross_entropy
            target_tokens = batch_sequences[:, 1:].clone()
            target_tokens[~batch_mask[:, 1:]] = -1
            target_tokens = target_tokens.reshape(-1)

            # Loss, bytes and token counts are over assistant tokens only
            supervised = target_tokens[target_tokens != -1]
            if supervised.numel() == 0:
                continue

            # Enable mixed precision
            with autocast(device_type=gpt.device.type, dtype=torch.bfloat16):
                loss = gpt.forward(input_tokens, target_tokens)

            val_loss += loss.item() * supervised.numel()
            val_tokens += supervised.numel()
            val_bytes += token_bytes[supervised].sum().item()

    avg_val_loss = val_loss / val_tokens if val_tokens > 0 else 0
    val_perplexity = math.exp(avg_val_loss) if avg_val_loss < 20 else float("inf")
    avg_bytes_per_token = val_bytes / val_tokens if val_tokens > 0 else 1
    val_bpb = avg_val_loss * math.log2(math.e) / avg_bytes_per_token
    return avg_val_loss, val_perplexity, val_bpb


if __name__ == "__main__":
    # -----------------------------------------------------------------------------
    sequence_length = 1024
    batch_size = 8
    eval_batches = 100
    model_path = "./artifacts/planckgpt-chat.pth"
    device = torch.device("cuda")
    # -----------------------------------------------------------------------------


    # Initialize model
    gpt = GPT({
        "device": device
    })
    gpt.load(model_path)
    gpt = torch.compile(gpt)

    print(f"Using device: {gpt.device}")
    print(f"Model parameters: {sum(p.numel() for p in gpt.parameters()):,}")

    encode = gpt.encoding.encode_ordinary
    val_data_loader = lambda: load_val_data(encode, batch_size, sequence_length)

    avg_val_loss, val_perplexity, val_bpb = evaluate(gpt, val_data_loader, eval_batches)

    print(f"Val BPB: {val_bpb:.4f}, Val Loss: {avg_val_loss:.4f}, Val Perplexity: {val_perplexity:.2f}")
