import math
import tiktoken
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim.lr_scheduler import LinearLR
from torch.utils.checkpoint import checkpoint
from muon import Muon, get_muon_momentum
from bitsandbytes.optim import Adam8bit
import numpy as np

# -----------------------------------------------------------------------------
# Custom operators: FP8 matmul (from modded-nanogpt by @YouJiacheng)

@torch.library.custom_op("nanogpt::mm", mutates_args=())
def mm_op(x: Tensor, w: Tensor, x_s: float, w_s: float, grad_s: float) -> tuple[Tensor, Tensor, Tensor]:
    @torch.compile
    def impl(x: Tensor, w: Tensor):
        assert x.is_contiguous() and w.is_contiguous()
        x_f8 = x.div(x_s).to(torch.float8_e4m3fn)
        w_f8 = w.div(w_s).to(torch.float8_e4m3fn)
        out = torch._scaled_mm(
            x_f8,
            w_f8.T,
            out_dtype=torch.bfloat16,
            scale_a=x.new_tensor(x_s, dtype=torch.float32),
            scale_b=x.new_tensor(w_s, dtype=torch.float32),
            use_fast_accum=True,
        )
        return out, x_f8, w_f8
    return impl(x, w)

@mm_op.register_fake
def _(x: Tensor, w: Tensor, *_):
    assert x.ndim == w.ndim == 2
    assert x.shape[1] == w.shape[1]
    assert x.device == w.device
    assert x.is_contiguous() and w.is_contiguous()
    return x @ w.T, x.to(torch.float8_e4m3fn), w.to(torch.float8_e4m3fn)

@torch.library.custom_op("nanogpt::mm_backward", mutates_args=())
def mm_backward_op(g: Tensor, x_f8: Tensor, w_f8: Tensor, x_s: float, w_s: float, grad_s: float) -> tuple[Tensor, Tensor]:
    @torch.compile
    def impl(grad: Tensor, x_f8: Tensor, w_f8: Tensor):
        assert grad.is_contiguous()
        x_inv_s = grad.new_tensor(x_s, dtype=torch.float32)
        w_inv_s = grad.new_tensor(w_s, dtype=torch.float32)
        grad_inv_s = grad.new_tensor(grad_s, dtype=torch.float32)
        grad_f8 = grad.div(grad_s).to(torch.float8_e5m2)
        grad_x = torch._scaled_mm(
            grad_f8,
            w_f8.T.contiguous().T,
            out_dtype=torch.bfloat16,
            scale_a=grad_inv_s,
            scale_b=w_inv_s,
            use_fast_accum=False,
        )
        grad_w = torch._scaled_mm(
            x_f8.T.contiguous(),
            grad_f8.T.contiguous().T,
            out_dtype=torch.float32,
            scale_a=x_inv_s,
            scale_b=grad_inv_s,
            use_fast_accum=False,
        ).T
        return grad_x, grad_w
    return impl(g, x_f8, w_f8)

@mm_backward_op.register_fake
def _(g: Tensor, x_f8: Tensor, w_f8: Tensor, *_):
    return x_f8.to(torch.bfloat16), w_f8.T.contiguous().T.to(torch.float32)

def _mm_backward(ctx, grad_out: Tensor, *_):
    x_f8, w_f8 = ctx.saved_tensors
    x_s, w_s, grad_s = ctx.scales
    grad_x, grad_w = torch.ops.nanogpt.mm_backward(
        grad_out, x_f8, w_f8, x_s, w_s, grad_s
    )
    return grad_x, grad_w, None, None, None

def _mm_setup_context(ctx: torch.autograd.function.FunctionCtx, inputs, output):
    *_, x_s, w_s, grad_s = inputs
    _, x_f8, w_f8 = output
    ctx.save_for_backward(x_f8, w_f8)
    ctx.scales = x_s, w_s, grad_s
    ctx.set_materialize_grads(False)

mm_op.register_autograd(_mm_backward, setup_context=_mm_setup_context)

# -----------------------------------------------------------------------------
# RMS norm with no learnable params

def rms_norm(x):
    return F.rms_norm(x, (x.size(-1),))

# -----------------------------------------------------------------------------
# CastedLinear with optional FP8 matmul

class CastedLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = False, 
                 use_fp8: bool = False, x_s: float = 1.0, w_s: float = 1.0, grad_s: float = 1.0):
        super().__init__(in_features, out_features, bias=bias)
        self.use_fp8 = use_fp8
        self.x_s = x_s
        self.w_s = w_s
        self.grad_s = grad_s

    def forward(self, x: Tensor) -> Tensor:
        if self.use_fp8 and self.training:
            _x = x.flatten(0, -2)
            out: Tensor = torch.ops.nanogpt.mm(_x, self.weight, x_s=self.x_s, w_s=self.w_s, grad_s=self.grad_s)[0]
            return out.reshape(*x.shape[:-1], -1)
        else:
            return F.linear(x, self.weight.type_as(x), self.bias)

# -----------------------------------------------------------------------------
# Rotate embeddings for RoPE

def apply_rotary_emb(x, cos, sin):
    x1, x2 = torch.chunk(x, 2, dim=-1)
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat((y1, y2), dim=-1)

# -----------------------------------------------------------------------------
# MQA for less memory use

class MultiQueryAttention(nn.Module):
    def __init__(self, dim, num_heads, use_fp8=False):
        super().__init__()
        assert dim % num_heads == 0
        
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.use_fp8 = use_fp8

        # Compute FP8 scaling factors based on dimension
        x_s = (dim ** 0.5) / 448
        w_s = 2 ** -9
        grad_s = 1 / 448

        self.q_proj = CastedLinear(dim, dim, bias=False, use_fp8=use_fp8, x_s=x_s, w_s=w_s, grad_s=grad_s)
        self.k_proj = CastedLinear(dim, self.head_dim, bias=False, use_fp8=use_fp8, x_s=x_s, w_s=w_s, grad_s=grad_s)
        self.v_proj = CastedLinear(dim, self.head_dim, bias=False, use_fp8=use_fp8, x_s=x_s, w_s=w_s, grad_s=grad_s)
        self.out_proj = CastedLinear(dim, dim, bias=False, use_fp8=use_fp8, x_s=x_s, w_s=w_s, grad_s=grad_s)

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
            k = torch.cat([kv_cache[0], k], dim=2)
            v = torch.cat([kv_cache[1], v], dim=2)
        new_kv_cache = (k, v)

        # Expand KV to match Q heads
        k = k.expand(B, self.num_heads, k.size(2), self.head_dim)
        v = v.expand(B, self.num_heads, v.size(2), self.head_dim)

        # Pytorch's scaled dot product attention
        Tq = q.size(2)
        Tk = k.size(2)

        if kv_cache is None or Tq == Tk:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        elif Tq == 1:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
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

# -----------------------------------------------------------------------------
# Transformer block

class Transformer(nn.Module):
    def __init__(self, dim, num_heads, dim_ff, use_fp8=False):
        super().__init__()
        
        # Compute FP8 scaling factors
        x_s = (dim ** 0.5) / 448
        w_s = 2 ** -9
        grad_s = 1 / 448

        self.attn = MultiQueryAttention(dim, num_heads, use_fp8=use_fp8)
        self.ffn1 = CastedLinear(dim, dim_ff, bias=False, use_fp8=use_fp8, x_s=x_s, w_s=w_s, grad_s=grad_s)
        self.ffn2 = CastedLinear(dim_ff, dim, bias=False, use_fp8=use_fp8, x_s=x_s, w_s=w_s, grad_s=grad_s)

    def forward(self, x, cos, sin, kv_cache=None):
        attn, new_kv_cache = self.attn(rms_norm(x), cos, sin, kv_cache)
        x = x + attn
        x = x + self.ffn2(F.relu(self.ffn1(rms_norm(x))).square())
        return x, new_kv_cache

# -----------------------------------------------------------------------------
# Main chatbot model

class ChatBot(nn.Module):
    def __init__(self, options={}):
        super().__init__()

        # Vocab setup - tiktoken BPE from GPT2
        self.encoding = tiktoken.get_encoding("gpt2")
        self.vocab_size = options.get("vocab_size", 50304)
        self.eos_token_id = self.encoding.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]

        # Config
        self.d_model = options.get("d_model", 768)
        self.num_layers = options.get("num_layers", 12)
        self.num_heads = options.get("num_heads", 6)
        self.rotary_seq_len = options.get("rotary_seq_len", 1025)
        
        # Whether to use FP8 matmuls (requires compatible GPU)
        self.use_fp8 = options.get("use_fp8", True)

        # Embedding (bfloat16)
        self.embedding = nn.Embedding(self.vocab_size, self.d_model)

        # Transformer decoder layers
        self.transformer = nn.ModuleList([
            Transformer(
                self.d_model,
                self.num_heads,
                self.d_model * 4,
                use_fp8=self.use_fp8
            ) for _ in range(self.num_layers)
        ])

        # Output projection with FP8
        x_s = (self.d_model ** 0.5) / 448
        w_s = 2 ** -9
        grad_s = 1 / 448
        self.output = CastedLinear(self.d_model, self.vocab_size, bias=False, 
                                    use_fp8=self.use_fp8, x_s=x_s, w_s=w_s, grad_s=grad_s)

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
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            self.device = torch.device("cuda")
        else:
            self.device = options.get("device", torch.device("cpu"))
        self.to(self.device)
        
        # Convert all parameters to bfloat16
        self.bfloat16()

        # Precompute cos and sin (in bfloat16)
        self.cos, self.sin = self._precompute_rotary_embeddings(self.rotary_seq_len, self.d_model // self.num_heads)

        # Init kv cache
        self.kv_caches = []
        self.use_kv_cache = False

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, CastedLinear)):
            fan_out = module.weight.size(0)
            fan_in = module.weight.size(1)
            std = 1.0 / math.sqrt(fan_in) * min(1.0, math.sqrt(fan_out / fan_in))
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=1.0)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000):
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=self.device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=self.device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        # Convert to bfloat16 after computing in float32
        cos, sin = cos.bfloat16(), sin.bfloat16()
        cos, sin = cos[None, None, :, :], sin[None, None, :, :]
        return cos, sin

    def forward(self, token_ids):
        _, seq_len = token_ids.shape

        # Token embedding (already in bfloat16)
        embedding = self.embedding(token_ids)

        # Embedding norm
        embedding = rms_norm(embedding)

        # Get position for RoPE
        if self.use_kv_cache and len(self.kv_caches) > 0 and self.kv_caches[0] is not None:
            cache_len = self.kv_caches[0][0].size(2)
            cos = self.cos[:, :, cache_len:cache_len + seq_len, :]
            sin = self.sin[:, :, cache_len:cache_len + seq_len, :]
        else:
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
        sequence_length=1025,
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
            nonlocal optimizer_step
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

                # Forward pass
                output = self.forward(input_tokens)  # [batch_size, seq_len-1, vocab_size]
                # Cast to float32 for loss computation
                output = output.float().reshape(-1, self.vocab_size)  # [batch_size * seq_len-1, vocab_size]
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
        context_window=1025,
        max_length=10240,
        repetition_penalty=1.1,
        repetition_penalty_range=64,
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

                input_tensor = torch.tensor(input_tokens, dtype=torch.long, device=self.device).unsqueeze(0)

                # Forward pass
                output = self.forward(input_tensor)
                logits = output[0, -1, :].float()  # Cast to float for sampling

                # Apply temperature scaling
                scaled_logits = logits / temperature

                # Repetition penalty
                if len(current_tokens) > 0:
                    recent_tokens = current_tokens[-repetition_penalty_range:]
                    token_counts = {}
                    for token in recent_tokens:
                        token_counts[token] = token_counts.get(token, 0) + 1
                    
                    for token_id, count in token_counts.items():
                        if token_id < len(scaled_logits):
                            penalty = repetition_penalty ** count
                            if scaled_logits[token_id] > 0:
                                scaled_logits[token_id] /= penalty
                            else:
                                scaled_logits[token_id] *= penalty

                # Top-k scaling
                top_k_values, top_k_indices = torch.topk(scaled_logits, k=topk)
                top_k_probs = torch.softmax(top_k_values, dim=0)

                # Sample from top-k
                sampled_index = torch.multinomial(top_k_probs, 1).item()
                next_token_id = top_k_indices[sampled_index].item()

                torch.cuda.empty_cache()

                if (
                    next_token_id == self.eos_token_id or
                    (next_token_id == 25 and current_tokens[-1] in [12982, 48902])
                ):
                    break

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

    def validate_model(self, data_loader, sequence_length=1025, batch_size=4):
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
                tokens = self.text_to_tokens(segment)
                print(f"Val Segment {segment_index + 1}: {len(segment)} chars -> {len(tokens)} tokens")

                sequences = []
                for start_idx in range(0, len(tokens) - sequence_length, sequence_length):
                    sequence = tokens[start_idx:start_idx + sequence_length]
                    if len(sequence) == sequence_length:
                        sequences.append(sequence)

                print(f"Val Segment {segment_index + 1}: Pre-computed {len(sequences)} sequences in memory")

                segment_loss = 0
                segment_tokens = 0

                for batch_start in range(0, len(sequences), batch_size):
                    batch_sequences = torch.tensor(sequences[batch_start:batch_start + batch_size], dtype=torch.long, device=self.device)
                    input_tokens = batch_sequences[:, :-1]
                    target_tokens = batch_sequences[:, 1:]

                    # Forward pass
                    output = self.forward(input_tokens)
                    output = output.float().reshape(-1, self.vocab_size)
                    target_tokens = target_tokens.reshape(-1)
                    loss = criterion(output, target_tokens)

                    segment_loss += loss.item() * target_tokens.size(0)
                    segment_tokens += target_tokens.size(0)

                total_loss += segment_loss
                total_tokens += segment_tokens
                num_segments += 1

                avg_segment_loss = segment_loss / segment_tokens if segment_tokens > 0 else 0
                avg_segment_perplexity = math.exp(avg_segment_loss) if avg_segment_loss < 20 else float("inf")
                print(f"Val Segment {segment_index + 1}: Loss: {avg_segment_loss:.4f}, Perplexity: {avg_segment_perplexity:.2f}")

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
