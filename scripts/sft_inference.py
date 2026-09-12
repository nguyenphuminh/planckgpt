import torch
from torch.amp import autocast
from model.core import GPT
from data.finetune_mix import USER_PREFIX, ASSISTANT_PREFIX, STOP_STRING, EOS

# -----------------------------------------------------------------------------
context_window=1024
max_length=400
temperature=0.8
topk=50
topp=0.9
repetition_penalty=1.05   # Gentler than the base model, chat repeats itself less
repetition_window=64
model_path="./artifacts/planckgpt-chat.pth"
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
print("/reset resets context, /clear clears console, /exit quits\n")

# Never treat what the user types as special tokens
encode = gpt.encoding.encode_ordinary

def build_prompt(history, message):
    """
    Render the conversation the same way finetune_mix.render does, but stopping
    at "Assistant:" so the model continues from there. Training rows are packed
    back to back, so every conversation but the first in a row is preceded by
    the previous one's end of text token: the leading EOS here reproduces that.
    """

    ids = [EOS]

    for past_message, past_reply in history:
        ids += encode(USER_PREFIX + past_message + ASSISTANT_PREFIX)
        ids += encode(" " + past_reply)
        ids.append(EOS)

    ids += encode(USER_PREFIX + message + ASSISTANT_PREFIX)

    return ids

# Prompt
history = []

while True:
    prompt = input("User: ")

    if not prompt.strip():
        continue
    if prompt == "/exit":
        break
    if prompt == "/reset":
        history = []
        print("(history cleared)\n")
        continue
    if prompt == "/clear":
        print("\033c", end="")
        continue

    # Drop the oldest exchanges until there is room to answer
    current_tokens = build_prompt(history, prompt)

    while len(current_tokens) > context_window - max_length and len(history) > 0:
        history = history[1:]
        current_tokens = build_prompt(history, prompt)

    current_tokens = current_tokens[-(context_window - 1):]
    prompt_length = len(current_tokens)

    # Enable kv cache and reset previous kv caches
    gpt.kv_caches = []
    gpt.use_kv_cache = True

    print("Assistant:", end="")

    with torch.inference_mode():
        # Stack in case a char is made up of multiple tokens
        word_stack = []
        # The reply so far, and how much of it has already been printed
        reply = ""
        printed = 0

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

            # Penalize tokens in recent window only, and only ones this reply
            # generated: the prompt is the user's words, not the model's
            generated = current_tokens[prompt_length:]

            if repetition_penalty != 1.0 and generated:
                window = generated[-repetition_window:] if repetition_window > 0 else generated
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

            # Stop on eos token, this is what the assistant turn was trained to end with
            if next_token_id == gpt.eos_token_id:
                break

            # Push newest token
            current_tokens.append(next_token_id)

            # Stream output
            word_stack.append(next_token_id)
            decoded_word = gpt.decode(word_stack)

            if "\ufffd" not in decoded_word:
                word_stack = []
                reply += decoded_word

                # Fallback stop: the model forgot the eos and started writing the
                # user's next turn instead. Cut the reply off there.
                if STOP_STRING in reply:
                    reply = reply.split(STOP_STRING)[0]
                    print(reply[printed:], end="", flush=True)
                    printed = len(reply)
                    break

                # Hold back a trailing partial stop string so it never reaches the
                # screen if the next token turns out to complete it
                limit = len(reply)
                for k in range(min(len(STOP_STRING) - 1, len(reply)), 0, -1):
                    if reply.endswith(STOP_STRING[:k]):
                        limit = len(reply) - k
                        break

                print(reply[printed:limit], end="", flush=True)
                printed = limit

            # Out of positions, stop rather than corrupt the rotary embeddings
            if len(current_tokens) >= context_window:
                break

        # Flush whatever was held back
        print(reply[printed:], end="", flush=True)

    # Disable kv cache when done generating and clear kv cache
    gpt.use_kv_cache = False
    gpt.kv_caches = []

    history.append((prompt, reply.strip()))

    print("\n")
