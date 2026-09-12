import tiktoken
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

def rms_norm(x):
    """RMS norm with no learnable params"""
    return F.rms_norm(x, (x.size(-1),))

def apply_rotary_emb(x, cos, sin):
    """Utility to rotate embeddings for RoPE"""
    x1, x2 = torch.chunk(x, 2, dim=-1)
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat((y1, y2), dim=-1)

def has_ve(index, num_layers):
    """Returns True if a transformer layer should have Value Embedding (alternating, last layer always included)."""
    return index % 2 == (num_layers - 1) % 2

class MultiHeadAttention(nn.Module):
    """MHA with kv cache support"""
    def __init__(self, dim, num_heads, has_ve):
        super().__init__()
        assert dim % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

        self.ve_gate_channels = 12
        self.ve_gate = nn.Linear(self.ve_gate_channels, self.num_heads, bias=False) if has_ve else None

    def forward(self, x, ve, cos, sin, kv_cache=None):
        B, L, _ = x.shape

        # QKV projection
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        # Value embedding
        if ve is not None:
            ve = ve.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
            gate = 3 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
            v = v + gate.transpose(1, 2).unsqueeze(-1) * ve

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

class Transformer(nn.Module):
    """Transformer block with MQA and Squared Relu activation"""
    def __init__(self, dim, num_heads, dim_ff, has_ve):
        super().__init__()

        self.attn = MultiHeadAttention(dim, num_heads, has_ve)
        self.ffn1 = nn.Linear(dim, dim_ff, bias=False)
        self.ffn2 = nn.Linear(dim_ff, dim, bias=False)

    def forward(self, x, ve, cos, sin, kv_cache=None):
        # Attention with kv cache
        attn, new_kv_cache = self.attn(rms_norm(x), ve, cos, sin, kv_cache)
        x = x + attn
        # Uses squared relu for activation
        x = x + self.ffn2(F.relu(self.ffn1(rms_norm(x))).square())
        return x, new_kv_cache

class GPT(nn.Module):
    """GPT class containing the model, training loop, and other utilities"""
    def __init__(self, options=None):
        super().__init__()
        
        if options is None:
            options = {}

        # Vocab setup - tiktoken BPE from GPT2
        self.encoding = tiktoken.get_encoding("gpt2")
        self.vocab_size = self.encoding.n_vocab
        self.eos_token_id = self.encoding.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]

        # Config
        self.d_model = options.get("d_model", 896)
        self.num_layers = options.get("num_layers", 14)
        self.num_heads = options.get("num_heads", 7)
        self.rotary_seq_len = options.get("rotary_seq_len", 1024)

        # Embedding
        self.embedding = nn.Embedding(self.vocab_size, self.d_model)

        # Per-layer learnable scalars, fake init, modified in-place later
        self.resid_lambdas = nn.Parameter(torch.ones(self.num_layers))
        self.x0_lambdas = nn.Parameter(torch.zeros(self.num_layers))

        # Value embeddings
        self.value_embeds = nn.ModuleDict({
            str(i): nn.Embedding(self.vocab_size, self.d_model) for i in range(self.num_layers) if has_ve(i, self.num_layers)
        })

        # Transformer decoder layers
        self.transformer = nn.ModuleList([
            Transformer(
                self.d_model,
                self.num_heads,
                self.d_model * 4,
                has_ve(i, self.num_layers)
            ) for i in range(self.num_layers)
        ])

        # One-hot output
        self.output = nn.Linear(self.d_model, self.vocab_size, bias=False)

        # Device
        self.device = options.get("device", torch.device("cpu"))

        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.fp32_precision = "tf32"
            torch.backends.cudnn.fp32_precision = "tf32"

        self.to(self.device)

        # Weight init
        self.init_weights()

        # Precompute cos and sin
        self.cos, self.sin = self._precompute_rotary_embeddings(self.rotary_seq_len, self.d_model // self.num_heads)

        # Init kv cache
        self.kv_caches = []
        self.use_kv_cache = False

    def init_weights(self):
        """
        Initialize weights following nanochat approach:
        - Embedding: normal, std=1.0
        - Output: normal, std=0.001
        - Attention Q,K,V & FFN: uniform, std=1/sqrt(d_model)
        - Output projections (attn.out_proj, ffn2): zeros
        """
        # Embedding
        torch.nn.init.normal_(self.embedding.weight, mean=0.0, std=0.8)
        
        # Output head - small init instead of zeros
        torch.nn.init.normal_(self.output.weight, mean=0.0, std=0.001)

        # Per-layer resid init: stronger residual at early layers, weaker at deep layers
        for i in range(self.num_layers):
            self.resid_lambdas.data[i] = 1.15 - (0.10 * i / max(self.num_layers - 1, 1))
        # Decaying x0 init: earlier layers get more input embedding blending
        for i in range(self.num_layers):
            self.x0_lambdas.data[i] = 0.20 - (0.15 * i / max(self.num_layers - 1, 1))
        
        # Transformer blocks: uniform init with bound = sqrt(3) * std
        s = 3**0.5 * self.d_model**-0.5  # sqrt(3)/sqrt(d_model)
        
        for layer in self.transformer:
            # Attention projections (Q, K, V)
            torch.nn.init.uniform_(layer.attn.q_proj.weight, -s, s)
            torch.nn.init.uniform_(layer.attn.k_proj.weight, -s, s)
            torch.nn.init.uniform_(layer.attn.v_proj.weight, -s, s)
            
            # Attention output projection - zero
            torch.nn.init.zeros_(layer.attn.out_proj.weight)
            
            # FFN first layer - uniform
            if hasattr(layer, "ffn1"):
                torch.nn.init.uniform_(layer.ffn1.weight, -s * 0.4, s * 0.4)
            
            # FFN output projection - zero
            torch.nn.init.zeros_(layer.ffn2.weight)

            # VE gate
            if layer.attn.ve_gate is not None:
                torch.nn.init.uniform_(layer.attn.ve_gate.weight, 0.0, 0.02)

        # Value embeddings
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)
        
        # Cast embeddings to bfloat16 if on CUDA
        if self.embedding.weight.device.type == "cuda":
            self.embedding.to(dtype=torch.bfloat16)

            for ve in self.value_embeds.values():
                ve.to(dtype=torch.bfloat16)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000):
        """Utility to precompute rotary embeddings for RoPE"""

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

    def forward(self, token_ids, targets=None, loss_reduction="mean"):
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

        # Store initial embedding to blend in later
        initial_embedding = embedding

        # Transformer forward pass
        for i, layer in enumerate(self.transformer):
            embedding = self.resid_lambdas[i] * embedding + self.x0_lambdas[i] * initial_embedding
            ve = self.value_embeds[str(i)](token_ids) if str(i) in self.value_embeds else None

            if self.use_kv_cache:
                layer_cache = self.kv_caches[i] if i < len(self.kv_caches) else None
                embedding, new_kv_cache = layer(embedding, ve, cos, sin, layer_cache)
                new_kv_caches.append(new_kv_cache)
            elif i % 3 == 0:
                embedding, _ = checkpoint(layer, embedding, ve, cos, sin, None, use_reentrant=False)
            else:
                embedding, _ = layer(embedding, ve, cos, sin, None)

        # Update cache list
        self.kv_caches = new_kv_caches

        # Final norm
        embedding = rms_norm(embedding)

        # Linear output projection
        output = self.output(embedding)

        # Logits softcapping
        softcap = 15.0
        output = softcap * torch.tanh(output.float() / softcap)
        
        if targets is not None:
            return F.cross_entropy(
                output.view(-1, output.size(-1)),
                targets.view(-1),
                ignore_index=-1,
                reduction=loss_reduction
            )
        else:
            return output

    def save(self, path):
        """Utility to save model"""
        torch.save({
            "model_state_dict": self.state_dict()
        }, path)

    def load(self, path):
        """Utility to load saved model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.load_state_dict(checkpoint["model_state_dict"])
        del checkpoint
        torch.cuda.empty_cache()

    def encode(self, text):
        """Utility to convert text string to list of tokens"""
        return self.encoding.encode(text, allowed_special={"<|endoftext|>"})

    def decode(self, tokens):
        """Utility to convert list of tokens to text string"""
        return self.encoding.decode(tokens)
