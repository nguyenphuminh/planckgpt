import math
import torch
import torch.nn as nn
from torch.amp import autocast
from model.core import GPT
import numpy as np
from data.fwedu import load_val_data


# Cache token->bytes lookups so repeated calls don't rebuild them
_token_bytes_cache = {}


def evaluate(gpt, val_data_loader, sequence_length=1024, batch_size=4):
    val_loss = 0
    val_tokens = 0
    val_bytes = 0

    # Loss
    criterion = nn.CrossEntropyLoss()

    # Build token->bytes lookup for BPB
    cache_key = (gpt.vocab_size, str(gpt.device))
    token_bytes = _token_bytes_cache.get(cache_key)
    if token_bytes is None:
        token_bytes = torch.zeros(gpt.vocab_size, dtype=torch.int64, device=gpt.device)
        for token_id in range(gpt.vocab_size):
            token_bytes[token_id] = len(gpt.encoding.decode([token_id]).encode("utf-8"))
        _token_bytes_cache[cache_key] = token_bytes

    # Accept either an iterable of segments or a loader factory (fresh generator per call)
    val_segments = val_data_loader() if callable(val_data_loader) else val_data_loader

    # Switch to eval for this pass, restore the previous mode afterwards
    was_training = gpt.training
    gpt.eval()

    with torch.no_grad():
        for val_segment in val_segments:
            # Encode segment to tokens
            val_tokens_array = np.array(gpt.encode(val_segment))

            # Truncate to fit exact number of sequences
            val_num_sequences = len(val_tokens_array) // sequence_length
            val_truncated = val_tokens_array[:val_num_sequences * sequence_length]
            # Reshape into 2D array
            val_sequences = val_truncated.reshape(val_num_sequences, sequence_length)

            for batch_start in range(0, len(val_sequences), batch_size):
                # Skip incomplete batches to avoid recompilation
                if batch_start + batch_size > len(val_sequences):
                    continue

                batch_sequences = torch.tensor(val_sequences[batch_start:batch_start + batch_size], dtype=torch.long, device=gpt.device)
                input_tokens = batch_sequences[:, :-1]
                target_tokens = batch_sequences[:, 1:]

                # Enable mixed precision
                with autocast(device_type=gpt.device.type, dtype=torch.bfloat16):
                    output = gpt.forward(input_tokens)  # [batch_size, seq_len-1, vocab_size]
                    output = output.reshape(-1, gpt.vocab_size)  # [batch_size * seq_len-1, vocab_size]
                    target_tokens = target_tokens.reshape(-1)  # [batch_size * seq_len-1]
                    loss = criterion(output, target_tokens)

                val_loss += loss.item() * target_tokens.size(0)
                val_tokens += target_tokens.size(0)
                val_bytes += token_bytes[target_tokens].sum().item()

    # Restore training mode for the caller
    if was_training:
        gpt.train()

    avg_val_loss = val_loss / val_tokens if val_tokens > 0 else 0
    val_perplexity = math.exp(avg_val_loss) if avg_val_loss < 20 else float("inf")
    avg_bytes_per_token = val_bytes / val_tokens if val_tokens > 0 else 1
    val_bpb = avg_val_loss * math.log2(math.e) / avg_bytes_per_token

    return avg_val_loss, val_perplexity, val_bpb


if __name__ == "__main__":
    # -----------------------------------------------------------------------------
    val_data_loader = load_val_data()
    sequence_length=1024
    batch_size=8
    model_path = "./artifacts/planckgpt.pth"
    device = torch.device("cuda")
    # -----------------------------------------------------------------------------


    # Initialize model
    gpt = GPT({
        "device": device
    })
    gpt.load(model_path)
    gpt = torch.compile(gpt)
    gpt.eval()

    print(f"Using device: {gpt.device}")
    print(f"Model parameters: {sum(p.numel() for p in gpt.parameters()):,}")

    avg_val_loss, val_perplexity, val_bpb = evaluate(gpt, val_data_loader, sequence_length, batch_size)

    print(f"Val BPB: {val_bpb:.4f}, Val Loss: {avg_val_loss:.4f}, Val Perplexity: {val_perplexity:.2f}")
