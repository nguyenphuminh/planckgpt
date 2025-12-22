import math
import tiktoken
import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax
import numpy as np
import functools

# RMS norm with no learnable params
def rms_norm(x):
    return x * jax.lax.rsqrt(jnp.mean(x ** 2, axis=-1, keepdims=True) + 1e-6)

# Rotate embeddings for RoPE
def apply_rotary_emb(x, cos, sin):
    x1, x2 = jnp.split(x, 2, axis=-1)
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return jnp.concat([y1, y2], axis=-1)

# Calculate cos and sin for RoPE
def compute_rotary_embeddings(seq_len, head_dim, base=10000):
    channel_range = jnp.arange(0, head_dim, 2, dtype=jnp.float32)
    inv_freq = 1.0 / (base ** (channel_range / head_dim))

    t = jnp.arange(seq_len, dtype=jnp.float32)
    freqs = jnp.outer(t, inv_freq)
    cos, sin = jnp.cos(freqs), jnp.sin(freqs)

    # After we have used float32 for more accurate cos and sin, convert to bfloat16
    cos, sin = cos.astype(jnp.bfloat16), sin.astype(jnp.bfloat16)
    cos, sin = cos[None, :, None, :], sin[None, :, None, :]
    return cos, sin

# Depth aware init function, inspired by nanochat
def depth_aware_init(key, shape, dtype):
    if len(shape) == 2:
        fan_in, fan_out = shape
        std = 1.0 / math.sqrt(fan_in) * min(1.0, math.sqrt(fan_out / fan_in))
    else:
        # Fallback for non-2D
        fan_in = shape[0]
        std = 1.0 / math.sqrt(fan_in)
    return jax.random.normal(key, shape, dtype=dtype) * std

# MQA for less memory use
class MultiQueryAttention(nnx.Module):
    def __init__(self, dim, num_heads, rngs):
        assert dim % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q_proj = nnx.Linear(
            dim,
            dim,
            kernel_init=depth_aware_init,
            use_bias=False,
            dtype=jnp.bfloat16,
            param_dtype=jnp.float32,
            rngs=rngs
        )

        self.k_proj = nnx.Linear(
            dim,
            self.head_dim,
            kernel_init=depth_aware_init,
            use_bias=False,
            dtype=jnp.bfloat16,
            param_dtype=jnp.float32,
            rngs=rngs
        )

        self.v_proj = nnx.Linear(
            dim,
            self.head_dim,
            kernel_init=depth_aware_init,
            use_bias=False,
            dtype=jnp.bfloat16,
            param_dtype=jnp.float32,
            rngs=rngs
        )

        self.out_proj = nnx.Linear(
            dim,
            dim,
            kernel_init=nnx.initializers.zeros,
            use_bias=False,
            dtype=jnp.bfloat16,
            param_dtype=jnp.float32,
            rngs=rngs
        )

    def __call__(self, x, cos, sin, kv_cache=None):
        # Init utils
        B, L, _ = x.shape

        # Get q, k, v
        q = self.q_proj(x).reshape(B, L, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(B, L, 1, self.head_dim)
        v = self.v_proj(x).reshape(B, L, 1, self.head_dim)

        # RoPE
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        # QK norm
        q = rms_norm(q)
        k = rms_norm(k)

        # Handle KV cache
        if kv_cache is not None:
            k = jnp.concat([kv_cache[0], k], axis=1)
            v = jnp.concat([kv_cache[1], v], axis=1)
        new_kv_cache = (k, v)

        # Expand KV to match Q heads
        k = jnp.broadcast_to(k, (B, k.shape[1], self.num_heads, self.head_dim))
        v = jnp.broadcast_to(v, (B, v.shape[1], self.num_heads, self.head_dim))

        # Scaled dot product attention
        Tq = q.shape[1]
        Tk = k.shape[1]

        # Create causal mask
        if kv_cache is None or Tq == Tk:
            # Full causal mask for training
            mask = jnp.tril(jnp.ones((Tq, Tk), dtype=jnp.bool_))
        elif Tq == 1:
            # No masking for single token
            mask = None
        else:
            # Custom causal mask for chunked inference
            mask = jnp.zeros((Tq, Tk), dtype=jnp.bool_)
            prefix_len = Tk - Tq
            if prefix_len > 0:
                mask = mask.at[:, :prefix_len].set(True)
            causal_part = jnp.tril(jnp.ones((Tq, Tq), dtype=jnp.bool_))
            mask = mask.at[:, prefix_len:].set(causal_part)

        # Use JAX's scaled dot product attention implementation
        out = jax.nn.dot_product_attention(
            q, k, v, 
            mask=mask,
            scale=1.0 / jnp.sqrt(self.head_dim)
        )

        out = out.reshape(B, L, -1)
        return self.out_proj(out), new_kv_cache

# Transformer block
class TransformerBlock(nnx.Module):
    def __init__(self, dim, num_heads, dim_ff, rngs):
        self.attn = MultiQueryAttention(dim, num_heads, rngs)
        
        self.ffn1 = nnx.Linear(
            dim,
            dim_ff,
            kernel_init=depth_aware_init,
            use_bias=False,
            dtype=jnp.bfloat16,
            param_dtype=jnp.float32,
            rngs=rngs
        )
        
        self.ffn2 = nnx.Linear(
            dim_ff,
            dim,
            kernel_init=nnx.initializers.zeros,
            use_bias=False,
            dtype=jnp.bfloat16,
            param_dtype=jnp.float32,
            rngs=rngs
        )

    def __call__(self, x, cos, sin, kv_cache=None):
        # Attention with kv cache
        attn, new_kv_cache = self.attn(rms_norm(x), cos, sin, kv_cache)
        x = x + attn
        # Uses squared relu for activation
        ffn_out = self.ffn1(rms_norm(x))
        ffn_out = jax.nn.relu(ffn_out) ** 2
        x = x + self.ffn2(ffn_out)
        return x, new_kv_cache

# GPT model in JAX
class JAXGPT(nnx.Module):
    def __init__(self, options={}, rngs=nnx.Rngs(0)):
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
        self.embedding = nnx.Embed(
            self.vocab_size,
            self.d_model,
            embedding_init=nnx.initializers.normal(stddev=1.0),
            dtype=jnp.bfloat16,
            param_dtype=jnp.float32,
            rngs=rngs
        )

        # Transformer decoder layers
        self.transformer = nnx.List([
            TransformerBlock(self.d_model, self.num_heads, self.d_model * 4, rngs)
            for _ in range(self.num_layers)
        ])

        # Output projection
        self.output = nnx.Linear(
            self.d_model,
            self.vocab_size,
            kernel_init=nnx.initializers.zeros,
            use_bias=False,
            dtype=jnp.bfloat16,
            param_dtype=jnp.float32,
            rngs=rngs
        )

        # Precompute cos and sin
        self.cos, self.sin = compute_rotary_embeddings(
            self.rotary_seq_len,
            self.d_model // self.num_heads
        )

    def __call__(self, token_ids, kv_caches=None, checkpoint_cond=None):
        # Init util vars
        _, seq_len = token_ids.shape
        kv_cache_not_empty = kv_caches is not None and len(kv_caches) > 0 and kv_caches[0] is not None

        # Token embedding
        embedding = self.embedding(token_ids)

        # Embedding norm
        embedding = rms_norm(embedding)

        # Get position for RoPE
        if kv_cache_not_empty:
            cache_len = kv_caches[0][0].shape[1]
            cos = self.cos[:, cache_len:cache_len + seq_len, :, :]
            sin = self.sin[:, cache_len:cache_len + seq_len, :, :]
        else:
            cos = self.cos[:, :seq_len, :, :]
            sin = self.sin[:, :seq_len, :, :]

        # Transformer forward pass
        new_kv_caches = []

        for i, layer in enumerate(self.transformer):
            if kv_caches is not None:
                embedding, new_kv_cache = layer(embedding, cos, sin, kv_caches[i] if kv_cache_not_empty else None)
                new_kv_caches.append(new_kv_cache)
            else:
                # checkpoint_cond determines whether a layer should be checkpointed
                if checkpoint_cond is not None and checkpoint_cond(i):
                    embedding, _ = jax.checkpoint(layer)(embedding, cos, sin, None)
                # No gradient checkpointing
                else:
                    embedding, _ = layer(embedding, cos, sin, None)

        # Final norm
        embedding = rms_norm(embedding)

        # Linear output projection
        output = self.output(embedding)

        # Logits softcapping
        softcap = 15.0
        output = softcap * jnp.tanh(output / softcap)
        
        return output, new_kv_caches

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

        # Calculate LR schedule parameters
        stable_steps = int(stable_range * total_steps)
        cooldown_steps = int((1 - stable_range) * total_steps)

        # ==================== ADAM OPTIMIZER ====================
        # For embedding and output layers
        adam_schedule = optax.linear_schedule(
            init_value=adam_lr,
            end_value=adam_lr * max_decay,
            transition_steps=cooldown_steps,
            transition_begin=stable_steps
        )
        
        # Build Adam transformation chain
        adam_tx = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.scale_by_adam(b1=adam_betas[0], b2=adam_betas[1]),
            optax.scale_by_schedule(adam_schedule),
            optax.scale(-1.0)
        )
        
        # Wrap with MultiSteps for gradient accumulation
        # This accumulates gradients over K steps before updating weights
        adam_tx = optax.MultiSteps(adam_tx, every_k_schedule=gradient_accumulation_steps)
        
        adam_opt = nnx.Optimizer(
            model=self,
            wrt=lambda path, _: any("embedding" in str(k) or "output" in str(k) for k in path),
            tx=adam_tx
        )

        # ==================== MUON OPTIMIZER ====================
        # For transformer layers (uses Muon for 2D params, Adam for others)
        muon_schedule = optax.linear_schedule(
            init_value=muon_lr,
            end_value=muon_lr * max_decay,
            transition_steps=cooldown_steps,
            transition_begin=stable_steps
        )
        
        # Build Muon transformation chain
        muon_tx = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.contrib.muon(
                learning_rate=1.0,  # Set to 1.0, we apply schedule separately
                beta=0.95,  # Muon momentum parameter
                adam_b1=adam_betas[0],  # Adam beta1 for non-2D params
                adam_b2=adam_betas[1]   # Adam beta2 for non-2D params
            ),
            optax.scale_by_schedule(muon_schedule),
        )
        
        # Wrap with MultiSteps for gradient accumulation
        muon_tx = optax.MultiSteps(muon_tx, every_k_schedule=gradient_accumulation_steps)
        
        muon_opt = nnx.Optimizer(
            model=self,
            wrt=nnx.PathContains("transformer"),
            tx=muon_tx
        )

        # ==================== TRAINING STEP ====================
        @nnx.jit
        def train_step(model, batch, adam_opt, muon_opt):
            input_tokens = batch[:, :-1]
            target_tokens = batch[:, 1:]

            # Loss function
            def loss_fn(mdl):
                output, _ = mdl(input_tokens, kv_caches=None)
                output = output.reshape(-1, mdl.vocab_size)
                target_flat = target_tokens.reshape(-1)
                
                # Cross entropy in fp32 for numerical stability
                return optax.softmax_cross_entropy_with_integer_labels(
                    output.astype(jnp.float32),
                    target_flat
                ).mean()

            loss, grads = nnx.value_and_grad(loss_fn)(model)

            # Get grads for just embedding and output
            adam_grads = nnx.State({
                k: v for k, v in grads.items() 
                if k in ["embedding", "output"]
            })

            # Get grads for just transformer
            muon_grads = nnx.State({
                k: v for k, v in grads.items() 
                if k == "transformer"
            })

            # Update weights through optimizers
            adam_opt.update(model, grads=adam_grads)
            muon_opt.update(model, grads=muon_grads)

            return loss

        # ==================== TRAINING LOOP ====================
        for segment_index, segment in enumerate(data_loader):
            # Encode segment to tokens
            tokens = np.array(self.text_to_tokens(segment))
            print(f"Segment {segment_index + 1}: {len(segment)} chars -> {len(tokens)} tokens")

            # Prepare sequences
            num_sequences = len(tokens) // sequence_length
            truncated = tokens[:num_sequences * sequence_length]
            sequences = truncated.reshape(num_sequences, sequence_length)
            print(f"Segment {segment_index + 1}: Pre-computed {len(sequences)} sequences in memory")

            total_loss = 0.0
            num_batches = 0

            # Process in small batches
            # MultiSteps will accumulate gradients automatically
            for batch_start in range(0, len(sequences), batch_size):
                batch_end = min(batch_start + batch_size, len(sequences))

                # Skip incomplete batches to prevent recompilation
                if batch_end - batch_start < batch_size:
                    continue

                # Load one small batch at a time
                batch = jnp.array(sequences[batch_start:batch_end], dtype=jnp.int32)

                # Single training step
                loss = train_step(self, batch, adam_opt, muon_opt)

                total_loss += float(loss)
                num_batches += 1

            # Log segment statistics
            avg_loss = total_loss / num_batches if num_batches > 0 else 0
            # Get the actual gradient step from MultiSteps wrapper
            adam_step = adam_opt.opt_state.gradient_step
            muon_step = muon_opt.opt_state.gradient_step
            # Query the schedules with the actual step count
            adam_lr_current = float(adam_schedule(adam_step))
            muon_lr_current = float(muon_schedule(muon_step))

            print(
                f"Segment {segment_index + 1}: Loss: {avg_loss:.4f}, "
                f"Adam LR: {adam_lr_current:.6f}, Muon LR: {muon_lr_current:.6f}, "
                f"Batches: {num_batches}"
            )

            # Save model after each segment
            self.save()
            print(f"Segment {segment_index + 1}: Saved to jaxgpt.npz")

    def generate(
        self,
        prompt,
        context_window=1024,
        max_length=4096,
        temperature=0.7,
        topk=50,
        memory=[],
        rng_key=jax.random.PRNGKey(0),
        kv_caches=[]
    ):
        current_tokens = memory + self.text_to_tokens(prompt)
        word_stack = []

        for i in range(max_length):
            if i == 0 or len(kv_caches) == 0:
                input_tokens = current_tokens[-context_window:]
            else:
                input_tokens = [current_tokens[-1]]

            input_tensor = jnp.array(input_tokens, dtype=jnp.int32)[None, :]

            # Forward pass (model uses bfloat16 internally)
            output, kv_caches = self(input_tensor, kv_caches)
            logits = output[0, -1, :].astype(jnp.float32)  # Cast to float32 for sampling

            # Apply temperature scaling
            scaled_logits = logits / temperature

            # Top-k sampling
            top_k_values, top_k_indices = jax.lax.top_k(scaled_logits, k=topk)
            top_k_probs = jax.nn.softmax(top_k_values, axis=0)

            # Sample from top-k
            rng_key, sample_key = jax.random.split(rng_key)
            sampled_index = jax.random.categorical(sample_key, jnp.log(top_k_probs))
            next_token_id = int(top_k_indices[sampled_index])

            if (
                next_token_id == self.eos_token_id or
                ((next_token_id == 25 or next_token_id in [12982, 48902]) and 
                 current_tokens[-1] in [12982, 48902])
            ):
                current_tokens.pop()
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
            if len(kv_caches) > 0 and kv_caches[0][0].shape[1] >= context_window:
                kv_caches = []
                current_tokens = current_tokens[-context_window:]

        return current_tokens[-context_window:]

    def validate_model(self, data_loader, sequence_length=1024, batch_size=4):
        print(f"Running validation with batch_size={batch_size}, sequence_length={sequence_length}")

        sequence_length = min(sequence_length, self.rotary_seq_len)

        total_loss = 0.0
        total_tokens = 0
        num_segments = 0

        @jax.jit
        def compute_loss(model, input_tokens, target_tokens):
            output, _ = model(input_tokens, kv_caches=None)
            output = output.reshape(-1, model.vocab_size)
            target_flat = target_tokens.reshape(-1)
            
            # Cross entropy in fp32 for numerical stability
            losses = optax.softmax_cross_entropy_with_integer_labels(
                output.astype(jnp.float32),
                target_flat
            )
            return jnp.sum(losses), target_flat.shape[0]

        for segment_index, segment in enumerate(data_loader):
            # Encode segment to tokens
            tokens = np.array(self.text_to_tokens(segment))
            print(f"Val Segment {segment_index + 1}: {len(segment)} chars -> {len(tokens)} tokens")

            # Prepare sequences
            num_sequences = len(tokens) // sequence_length
            truncated = tokens[:num_sequences * sequence_length]
            sequences = truncated.reshape(num_sequences, sequence_length)
            print(f"Val Segment {segment_index + 1}: Pre-computed {len(sequences)} sequences")

            segment_loss = 0.0
            segment_tokens = 0

            for batch_start in range(0, len(sequences), batch_size):
                batch_end = min(batch_start + batch_size, len(sequences))
                
                # Skip incomplete batches for consistency
                if batch_end - batch_start < batch_size:
                    continue
                
                # Load one small batch at a time
                batch = jnp.array(sequences[batch_start:batch_end], dtype=jnp.int32)
                input_tokens = batch[:, :-1]
                target_tokens = batch[:, 1:]

                loss, num_tokens = compute_loss(self, input_tokens, target_tokens)
                segment_loss += float(loss)
                segment_tokens += int(num_tokens)

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

    def save(self, path="./jaxgpt.npz"):
        state = nnx.state(self)
        with open(path, "wb") as f:
            np.savez(f, **jax.tree.map(np.array, state))

    @staticmethod
    def load(path="./jaxgpt.npz", options=None):
        if options is None:
            options = {}
        
        # Create new model
        model = JAXGPT(options=options, rngs=nnx.Rngs(0))
        
        # Load state
        with open(path, "rb") as f:
            data = np.load(f)
            state_dict = {k: jnp.array(v) for k, v in data.items()}
        
        nnx.update(model, state_dict)
        return model

    def text_to_tokens(self, text):
        return self.encoding.encode(text, allowed_special={"<|endoftext|>"})

    def tokens_to_text(self, tokens):
        return self.encoding.decode(tokens)
