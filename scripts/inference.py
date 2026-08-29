import torch
from torch.amp import autocast
from model.core import GPT

# -----------------------------------------------------------------------------
context_window=1024
max_length=1024
temperature=1.0
topk=50
topp=0.95
repetition_penalty=1.2
repetition_window=64
model_path="./artifacts/planckgpt.pth"
# -----------------------------------------------------------------------------

# Initialize model
gpt = GPT({
    "device": torch.device("cuda")
})

print(f"Using device: {gpt.device}")
print(f"Model parameters: {sum(p.numel() for p in gpt.parameters()):,}")

# Load model
gpt.load(model_path)
print(f"Loaded from {model_path}")

# Prompt
while True:
    prompt = input("Prompt: ")

    # Enable kv cache and reset previous kv caches
    gpt.kv_caches = []
    gpt.use_kv_cache = True

    with torch.inference_mode():
        current_tokens = gpt.encode(prompt)

        # Stack in case a char is made up of multiple tokens
        word_stack = []

        for i in range(max_length):
            if i == 0 or len(gpt.kv_caches) == 0:
                # First iteration: process full context
                input_tokens = current_tokens[-context_window:]
            else:
                # Subsequent: only process new token
                input_tokens = [current_tokens[-1]]

            input_tensor = torch.tensor(input_tokens, device=gpt.device).unsqueeze(0)

            # Forward pass
            with autocast(device_type=gpt.device.type, dtype=torch.bfloat16):
                output = gpt.forward(input_tensor)
            logits = output[0, -1, :]

            # Apply temperature scaling
            scaled_logits = logits / temperature

            # Penalize tokens in recent window only
            if repetition_penalty != 1.0:
                window = current_tokens[-repetition_window:] if repetition_window > 0 else current_tokens
                window_ids = torch.tensor(list(set(window)), device=gpt.device)
                window_logits = scaled_logits[window_ids]
                scaled_logits[window_ids] = torch.where(
                    window_logits < 0,
                    window_logits * repetition_penalty,
                    window_logits / repetition_penalty
                )

            # Top-k filtering
            top_k_values, top_k_indices = torch.topk(scaled_logits, k=topk)

            # Top-p filtering within top-k candidates
            if topp < 1.0:
                top_k_probs = torch.softmax(top_k_values, dim=0)
                cumulative_probs = torch.cumsum(top_k_probs, dim=0)
                # Remove tokens that push cumulative prob over topp
                nucleus_mask = cumulative_probs - top_k_probs < topp
                top_k_values = top_k_values[nucleus_mask]
                top_k_indices = top_k_indices[nucleus_mask]

            top_k_probs = torch.softmax(top_k_values, dim=0)

            # Sample from top-k
            sampled_index = torch.multinomial(top_k_probs, 1).item()
            next_token_id = top_k_indices[sampled_index].item()

            # Stop on eos token and conversation overlap
            if next_token_id == gpt.eos_token_id:
                current_tokens.pop()
                break

            # Push newest token
            current_tokens.append(next_token_id)

            # Stream output
            word_stack.append(next_token_id)
            decoded_word = gpt.decode(word_stack)

            if "\ufffd" not in decoded_word:
                print(decoded_word, end="")
                word_stack = []

            # Reset kv cache if too long
            if len(gpt.kv_caches) > 0 and gpt.kv_caches[0][0].size(2) >= context_window:
                gpt.kv_caches = []
                current_tokens = current_tokens[-context_window:]

    # Disable kv cache when done generating and clear kv cache
    gpt.use_kv_cache = False
    gpt.kv_caches = []

    print("\n")
