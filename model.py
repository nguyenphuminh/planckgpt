import math
import tiktoken
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LinearLR
from torch.utils.checkpoint import checkpoint
from torch.amp import autocast
from muon import Muon, get_muon_momentum
from bitsandbytes.optim import Adam8bit
import numpy as np

# RMS norm with no learnable params
def rms_norm(x):
    return F.rms_norm(x, (x.size(-1),))

# Rotate embeddings for RoPE
def apply_rotary_emb(x, cos, sin):
    x1, x2 = torch.chunk(x, 2, dim=-1)
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat((y1, y2), dim=-1)

# MQA for less memory use
class MultiQueryAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        assert dim % num_heads == 0
        
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.head_dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x, cos, sin, kv_cache=None):
        B, L, _ = x.shape

        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, 1, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, 1, self.head_dim).transpose(1, 2)

        # RoPE
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        # QK norm
        q = rms_norm(q)
        k = rms_norm(k)

        # Handle KV cache
        if kv_cache is not None:
            # Concatenate with cached K, V
            k = torch.cat([kv_cache[0], k], dim=2)
            v = torch.cat([kv_cache[1], v], dim=2)
        # Update cache with current K, V
        new_kv_cache = (k, v)

        # Expand KV to match Q heads
        k = k.expand(B, self.num_heads, k.size(2), self.head_dim)
        v = v.expand(B, self.num_heads, v.size(2), self.head_dim)

        # Pytorch's scaled dot product attention, should use flash attention behind the hood
        Tq = q.size(2)
        Tk = k.size(2)

        # Full causal mask in training with no kv cache
        if kv_cache is None or Tq == Tk:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        # No causal mask in inference when generating with single tokens
        elif Tq == 1:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        # Custom causal mask in inference when generating with chunks
        else:
            attn_mask = torch.zeros((Tq, Tk), dtype=torch.bool, device=q.device)
            prefix_len = Tk - Tq
            if prefix_len > 0:
                attn_mask[:, :prefix_len] = True
            attn_mask[:, prefix_len:] = torch.tril(
                torch.ones((Tq, Tq), dtype=torch.bool, device=q.device)
            )
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)

        return self.out_proj(out.transpose(1, 2).reshape(B, L, -1)), new_kv_cache

# Transformer block
class Transformer(nn.Module):
    def __init__(self, dim, num_heads, dim_ff):
        super().__init__()

        self.attn = MultiQueryAttention(dim, num_heads)
        self.ffn1 = nn.Linear(dim, dim_ff, bias=False)
        self.ffn2 = nn.Linear(dim_ff, dim, bias=False)

    def forward(self, x, cos, sin, kv_cache=None):
        # Attention with kv cache
        attn, new_kv_cache = self.attn(rms_norm(x), cos, sin, kv_cache)
        x = x + attn
        # Uses squared relu for activation
        x = x + self.ffn2(F.relu(self.ffn1(rms_norm(x))).square())
        return x, new_kv_cache

# Main chatbot model
class ChatBot(nn.Module):
    def __init__(self, options={}):
        super().__init__()

        # Vocab setup - tiktoken BPE from GPT2
        self.encoding = tiktoken.get_encoding("gpt2")
        self.vocab_size = options.get("vocab_size", 50257)
        self.eos_token_id = self.encoding.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]

        # Config
        self.d_model = options.get("d_model", 768)
        self.num_layers = options.get("num_layers", 12)
        self.num_heads = options.get("num_heads", 6)
        self.rotary_seq_len = options.get("rotary_seq_len", 1024)

        # Embedding
        self.embedding = nn.Embedding(self.vocab_size, self.d_model, dtype=torch.bfloat16)

        # Transformer decoder layers
        self.transformer = nn.ModuleList([
            Transformer(
                self.d_model,
                self.num_heads,
                self.d_model * 4
            ) for _ in range(self.num_layers)
        ])

        # One-hot output
        self.output = nn.Linear(self.d_model, self.vocab_size, bias=False)

        # Apply weight init
        self.apply(self._init_weights)
        # Zero out specific output projections for residual paths
        for layer in self.transformer:
            torch.nn.init.zeros_(layer.attn.out_proj.weight)
            torch.nn.init.zeros_(layer.ffn2.weight)
        torch.nn.init.zeros_(self.output.weight)

        # Device
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.fp32_precision = "tf32"
            torch.backends.cudnn.fp32_precision = "tf32"
            self.device = torch.device("cuda")
        else:
            self.device = options.get("device", torch.device("cpu"))
        self.to(self.device)

        # Precompute cos and sin
        self.cos, self.sin = self._precompute_rotary_embeddings(self.rotary_seq_len, self.d_model // self.num_heads)

        # Init kv cache
        self.kv_caches = []
        self.use_kv_cache = False

    def _init_weights(self, module):
        # Yoinked from nanochat basically, depth-aware weight init
        if isinstance(module, nn.Linear):
            fan_out = module.weight.size(0)
            fan_in = module.weight.size(1)
            std = 1.0 / math.sqrt(fan_in) * min(1.0, math.sqrt(fan_out / fan_in))

            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=1.0)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000):
        # Stride the channels
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=self.device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))

        # Stride the time steps
        t = torch.arange(seq_len, dtype=torch.float32, device=self.device)

        # Calculate the rotation frequencies at each (time, channel) pair
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()

        # After we have used float32 for more accurate cos and sin, we keep bfloat16
        cos, sin = cos.bfloat16(), sin.bfloat16()
        cos, sin = cos[None, None, :, :], sin[None, None, :, :]
        return cos, sin

    def forward(self, token_ids):
        _, seq_len = token_ids.shape

        # Token embedding
        embedding = self.embedding(token_ids)

        # Embedding norm
        embedding = rms_norm(embedding)

        # Get position for RoPE
        if self.use_kv_cache and len(self.kv_caches) > 0 and self.kv_caches[0] is not None:
            # When using cache, position starts from cache length
            cache_len = self.kv_caches[0][0].size(2)
            cos = self.cos[:, :, cache_len:cache_len + seq_len, :]
            sin = self.sin[:, :, cache_len:cache_len + seq_len, :]
        else:
            # No cache or not inference, use positions from 0
            cos = self.cos[:, :, :seq_len, :]
            sin = self.sin[:, :, :seq_len, :]

        # Initialize cache list
        new_kv_caches = []

        # Transformer forward pass
        for i, layer in enumerate(self.transformer):
            if self.use_kv_cache:
                layer_cache = self.kv_caches[i] if i < len(self.kv_caches) else None
                embedding, new_kv_cache = layer(embedding, cos, sin, layer_cache)
                new_kv_caches.append(new_kv_cache)
            else:
                if i % 3 == 0:
                    embedding, _ = checkpoint(layer, embedding, cos, sin, None, use_reentrant=False)
                else:
                    embedding, _ = layer(embedding, cos, sin, None)

        # Update cache list
        self.kv_caches = new_kv_caches

        # Final norm
        embedding = rms_norm(embedding)

        # Linear output projection
        output = self.output(embedding)

        # Logits softcapping
        softcap = 15.0
        output = softcap * torch.tanh(output / softcap)
        
        return output

    def train_model(
        self,
        data_loader,
        sequence_length=1024,
        batch_size=4,
        gradient_accumulation_steps=128,
        adam_lr=0.008,
        adam_betas=(0.65, 0.95),
        muon_lr=0.06,
        stable_range=0.55,
        total_steps=5277,
        max_decay=0.1
    ):
        print(f"Training with batch_size={batch_size}, gradient_accumulation_steps={gradient_accumulation_steps}")
        print(f"Effective batch size: {batch_size * gradient_accumulation_steps}")

        # Cap context window
        sequence_length = min(sequence_length, self.rotary_seq_len)

        # Loss
        criterion = nn.CrossEntropyLoss()

        # (45%) cooldown to min (0.1x)
        stable_steps = int(stable_range * total_steps)
        cooldown_steps = int((1 - stable_range) * total_steps)

        # AdamW for embedding/linear weights
        linear_params = [self.embedding.weight, self.output.weight]
        adam_opt = Adam8bit(linear_params, lr=adam_lr, betas=adam_betas)
        adam_cooldown_scheduler = LinearLR(adam_opt, start_factor=1.0, end_factor=max_decay, total_iters=cooldown_steps)

        # Muon for transformer params
        transformer_params = [p for n, p in self.named_parameters() if "embedding" not in n and "output" not in n]
        muon_opt = Muon(transformer_params, lr=muon_lr, momentum=0.95)
        muon_cooldown_scheduler = LinearLR(muon_opt, start_factor=1.0, end_factor=max_decay, total_iters=cooldown_steps)

        # Track optimizer step for Muon momentum update
        optimizer_step = 0

        def opt_step():
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)

            # AdamW step
            if optimizer_step % 2 != 0:
                adam_opt.step()
                adam_opt.zero_grad(set_to_none=True)

            # Muon step
            for group in muon_opt.param_groups:
                group["momentum"] = get_muon_momentum(optimizer_step)
            muon_opt.step()
            muon_opt.zero_grad(set_to_none=True)

            # LR scheduler steps
            if optimizer_step > stable_steps:
                adam_cooldown_scheduler.step()
                muon_cooldown_scheduler.step()

        for segment_index, segment in enumerate(data_loader):
            # Encode segment to tokens
            tokens = np.array(self.text_to_tokens(segment))
            print(f"Segment {segment_index + 1}: {len(segment)} chars -> {len(tokens)} tokens")
            
            # Truncate to fit exact number of sequences
            num_sequences = len(tokens) // sequence_length
            truncated = tokens[:num_sequences * sequence_length]
            # Reshape into 2D array
            sequences = truncated.reshape(num_sequences, sequence_length)
            
            print(f"Segment {segment_index + 1}: Pre-computed {len(sequences)} sequences in memory")

            # Training loop for this segment
            self.train()

            total_loss = 0
            num_batches = 0
            
            adam_opt.zero_grad(set_to_none=True)
            muon_opt.zero_grad(set_to_none=True)

            for batch_start in range(0, len(sequences), batch_size):
                batch_sequences = torch.tensor(sequences[batch_start:batch_start + batch_size], dtype=torch.long, device=self.device)
                input_tokens = batch_sequences[:, :-1]
                target_tokens = batch_sequences[:, 1:]

                # Enable mixed precision
                with autocast(device_type=self.device.type, dtype=torch.bfloat16):
                    output = self.forward(input_tokens)  # [batch_size, seq_len-1, vocab_size]
                    output = output.reshape(-1, self.vocab_size)  # [batch_size * seq_len-1, vocab_size]
                    target_tokens = target_tokens.reshape(-1)  # [batch_size * seq_len-1]
                    loss = criterion(output, target_tokens)
                    loss = loss / gradient_accumulation_steps

                # Propagate grad
                loss.backward()
                total_loss += loss.item() * gradient_accumulation_steps
                num_batches += 1
                
                # Update weights every gradient_accumulation_steps
                if num_batches % gradient_accumulation_steps == 0:
                    opt_step()
                    optimizer_step += 1

            # Final update if needed
            if num_batches % gradient_accumulation_steps != 0:
                opt_step()
                optimizer_step += 1

            # Get log info
            avg_loss = total_loss / num_batches if num_batches > 0 else 0
            adamw_current_lr = adam_opt.param_groups[0]["lr"]
            muon_current_lr = muon_opt.param_groups[0]["lr"]

            # Log and save
            print(f"Segment {segment_index + 1}: Loss: {avg_loss:.4f}, AdamW LR: {adamw_current_lr:.6f}, Muon LR: {muon_current_lr:.6f}, Batches: {num_batches}")
            self.save()
            print(f"Segment {segment_index + 1}: Saved to chatbot.pth")

    def generate(
        self,
        prompt,
        context_window=1024,
        max_length=10240,
        temperature=0.7,
        topk=50,
        memory=[]
    ):
        self.eval()
        # Enable kv cache and reset previous kv caches
        self.kv_caches = []
        self.use_kv_cache = True

        with torch.inference_mode():
            current_tokens = memory + self.text_to_tokens(prompt)

            # Stack in case a char is made up of multiple tokens
            word_stack = []

            for i in range(max_length):
                if i == 0 or len(self.kv_caches) == 0:
                    # First iteration: process full context
                    input_tokens = current_tokens[-context_window:]
                else:
                    # Subsequent: only process new token
                    input_tokens = [current_tokens[-1]]

                input_tensor = torch.tensor(input_tokens, device=self.device).unsqueeze(0)

                # Forward pass
                with autocast(device_type=self.device.type, dtype=torch.bfloat16):
                    output = self.forward(input_tensor)
                logits = output[0, -1, :]

                # Apply temperature scaling
                scaled_logits = logits / temperature

                # Top-k scaling
                top_k_values, top_k_indices = torch.topk(scaled_logits, k=topk)
                top_k_probs = torch.softmax(top_k_values, dim=0)

                # Sample from top-k
                sampled_index = torch.multinomial(top_k_probs, 1).item()
                next_token_id = top_k_indices[sampled_index].item()

                torch.cuda.empty_cache()

                if (
                    # Stop on eos token and conversation overlap
                    next_token_id == self.eos_token_id or
                    # Stop on "User: or Assistant:"
                    (next_token_id == 25 and current_tokens[-1] in [12982, 48902])
                ):
                    break

                # Push newest token
                current_tokens.append(next_token_id)

                # Stream output
                word_stack.append(next_token_id)
                decoded_word = self.tokens_to_text(word_stack)

                if (
                    "\ufffd" not in decoded_word and
                    "User" not in decoded_word and
                    "Assistant" not in decoded_word
                ):
                    print(decoded_word, end="")
                    word_stack = []

                # Reset kv cache if too long
                if len(self.kv_caches) > 0 and self.kv_caches[0][0].size(2) >= context_window:
                    self.kv_caches = []
                    current_tokens = current_tokens[-context_window:]

        # Disable kv cache when done generating and clear kv cache
        self.use_kv_cache = False
        self.kv_caches = []

        return current_tokens[-context_window:]

    def validate_model(self, data_loader, sequence_length=1024, batch_size=4):
        print(f"Running validation with batch_size={batch_size}, sequence_length={sequence_length}")

        # Cap context window
        sequence_length = min(sequence_length, self.rotary_seq_len)

        # Set model to eval mode
        self.eval()

        criterion = nn.CrossEntropyLoss()

        total_loss = 0
        total_tokens = 0
        num_segments = 0

        # No gradient computation during validation
        with torch.no_grad():
            for segment_index, segment in enumerate(data_loader):
                # Encode segment to tokens (same as training)
                tokens = self.text_to_tokens(segment)
                print(f"Val Segment {segment_index + 1}: {len(segment)} chars -> {len(tokens)} tokens")

                # Pre-create all sequences (same as training)
                sequences = []
                for start_idx in range(0, len(tokens) - sequence_length, sequence_length):
                    sequence = tokens[start_idx:start_idx + sequence_length]
                    if len(sequence) == sequence_length:
                        sequences.append(sequence)

                print(f"Val Segment {segment_index + 1}: Pre-computed {len(sequences)} sequences in memory")

                segment_loss = 0
                segment_tokens = 0

                # Process batches (same as training)
                for batch_start in range(0, len(sequences), batch_size):
                    batch_sequences = torch.tensor(sequences[batch_start:batch_start + batch_size], device=self.device)
                    input_tokens = batch_sequences[:, :-1]
                    target_tokens = batch_sequences[:, 1:]

                    # Enable mixed precision (same as training)
                    with autocast(device_type=self.device.type, dtype=torch.bfloat16):
                        output = self.forward(input_tokens)  # [batch_size, seq_len-1, vocab_size]
                        output = output.reshape(-1, self.vocab_size)  # [batch_size * seq_len-1, vocab_size]
                        target_tokens = target_tokens.reshape(-1)  # [batch_size * seq_len-1]
                        loss = criterion(output, target_tokens)

                    segment_loss += loss.item() * target_tokens.size(0)
                    segment_tokens += target_tokens.size(0)

                total_loss += segment_loss
                total_tokens += segment_tokens
                num_segments += 1

                # Log segment validation results
                avg_segment_loss = segment_loss / segment_tokens if segment_tokens > 0 else 0
                avg_segment_perplexity = math.exp(avg_segment_loss) if avg_segment_loss < 20 else float("inf")
                print(f"Val Segment {segment_index + 1}: Loss: {avg_segment_loss:.4f}, Perplexity: {avg_segment_perplexity:.2f}")

        # Calculate overall validation metrics
        avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
        perplexity = math.exp(avg_loss) if avg_loss < 20 else float("inf")

        print(f"Segments: {num_segments}")
        print(f"Total tokens: {total_tokens}")
        print(f"Average Loss: {avg_loss:.4f}")
        print(f"Perplexity: {perplexity:.2f}")

    def save(self, path="./chatbot.pth"):
        torch.save({
            "model_state_dict": self.state_dict(),
            "d_model": self.d_model,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "vocab_size": self.vocab_size,
            "rotary_seq_len": self.rotary_seq_len,
            "eos_token_id": self.eos_token_id
        }, path)

    def load(self, path="./chatbot.pth"):
        checkpoint = torch.load(path, map_location=self.device)
        self.load_state_dict(checkpoint["model_state_dict"])

    def text_to_tokens(self, text):
        return self.encoding.encode(text, allowed_special={"<|endoftext|>"})

    def tokens_to_text(self, tokens):
        return self.encoding.decode(tokens)
