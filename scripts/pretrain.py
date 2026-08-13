import math
import torch
import torch.nn as nn
from torch.amp import autocast
from optim.muon import Muon
from bitsandbytes.optim import AdamW8bit
import numpy as np
from model.core import GPT
from data.fwedu import load_data, load_val_data
from scripts.evaluate import evaluate


# -----------------------------------------------------------------------------
#  Model config
d_model = 896
num_layers = 14
num_heads = 7
rotary_seq_len = 1024
device = torch.device("cuda")

# Decay config
stable_range = 0.65
total_steps = 3815
warmup_steps = 40
max_decay = 0.05

# Batch config
sequence_length = 1024
batch_size = 4
gradient_accumulation_steps = 128

# Data config
data_loader = load_data()
val_data_loader = [*load_val_data()]
num_segments=20

# Hyperparams config
scale = 1 / (d_model / 768) ** 0.5 # Scale for different d_model
adam_config = {
    "output":    { "lr": 0.008 * scale, "betas": (0.8, 0.96),  "eps": 1e-10, "weight_decay": 0.01  },
    "embedding": { "lr": 0.3 * scale,   "betas": (0.8, 0.995), "eps": 1e-10, "weight_decay": 0.001 },
    "resid_lambdas": { "lr": 0.5 * 0.01, "betas": (0.8, 0.95), "eps": 1e-10, "weight_decay": 0.05 },
    "x0_lambdas": { "lr": 0.5, "betas": (0.96, 0.95), "eps": 1e-10, "weight_decay": 0.0 },
}
muon_config = {
    "matrix": { "lr": 0.02, "weight_decay": 0.185 }
}

# Paths
checkpoint_path = "./artifacts/checkpoints"
model_path = "./artifacts/planckgpt.pth"
# -----------------------------------------------------------------------------


# Initialize model
raw_gpt = GPT({
    "d_model": d_model,
    "num_layers": num_layers,
    "num_heads": num_heads,
    "rotary_seq_len": rotary_seq_len,
    "device": device
})
gpt = torch.compile(raw_gpt, mode="max-autotune-no-cudagraphs", dynamic=False)
gpt.train()

print(f"Using device: {gpt.device}")
print(f"Model parameters: {sum(p.numel() for p in gpt.parameters()):,}")

# Cap context window
sequence_length = min(sequence_length, gpt.rotary_seq_len)

# Loss
criterion = nn.CrossEntropyLoss()

# Warmup steps and base weight decay to prepare for warmdown
base_wd = muon_config["matrix"]["weight_decay"]

# AdamW for embedding/linear weights
adam_params = [
    { "params": [gpt.output.weight],    **adam_config["output"],    "lr": adam_config["output"]["lr"]    },
    { "params": [gpt.embedding.weight], **adam_config["embedding"], "lr": adam_config["embedding"]["lr"] },
    { "params": [gpt.resid_lambdas], **adam_config["resid_lambdas"], "lr": adam_config["resid_lambdas"]["lr"] },
    { "params": [gpt.x0_lambdas], **adam_config["x0_lambdas"], "lr": adam_config["x0_lambdas"]["lr"] },
]
adam_opt = AdamW8bit(adam_params)
adam_initial_lrs = [group["lr"] for group in adam_opt.param_groups]

# Muon for transformer params
muon_params = [p for n, p in gpt.named_parameters() if all(key not in n for key in adam_config.keys())]
muon_opt = Muon(muon_params, lr = muon_config["matrix"]["lr"])
muon_initial_lrs = [group["lr"] for group in muon_opt.param_groups]

# Track optimizer step for Muon momentum update
optimizer_step = 0

# Resume from latest checkpoint if one exists
resume_segment = 0
if checkpoint_path:
    import os, glob
    os.makedirs(checkpoint_path, exist_ok=True)
    ckpts = sorted(glob.glob(os.path.join(checkpoint_path, "checkpoint_seg*.pt")))
    if ckpts:
        latest = ckpts[-1]
        print(f"Resuming from checkpoint: {latest}")
        ckpt = torch.load(latest, map_location=gpt.device)
        raw_gpt.load_state_dict(ckpt["model_state_dict"])
        adam_opt.load_state_dict(ckpt["adam_opt_state_dict"])
        muon_opt.load_state_dict(ckpt["muon_opt_state_dict"])
        optimizer_step = ckpt["optimizer_step"]
        resume_segment = ckpt["segment_index"] + 1

        # Deallocate redundant mem
        del ckpt
        torch.cuda.empty_cache()
        
        print(f"Resumed at segment {resume_segment}, optimizer_step={optimizer_step}")

def get_lr_multiplier(step):
    warmdown_steps = int((1 - stable_range) * total_steps)
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    elif step <= total_steps - warmdown_steps:
        return 1.0
    else:
        progress = max((total_steps - step) / warmdown_steps, 0.0)
        return progress * 1.0 + (1 - progress) * max_decay

def opt_step():
    # LR schedule
    lrm = get_lr_multiplier(optimizer_step)
    for i, group in enumerate(adam_opt.param_groups):
        group["lr"] = adam_initial_lrs[i] * lrm
    for i, group in enumerate(muon_opt.param_groups):
        group["lr"] = muon_initial_lrs[i] * lrm

    # Weight decay cosine schedule to zero
    wd = base_wd * 0.5 * (1 + math.cos(math.pi * min(optimizer_step, total_steps) / total_steps))
    for group in muon_opt.param_groups:
        group["weight_decay"] = wd

    # Step both optimizers
    adam_opt.step()
    adam_opt.zero_grad(set_to_none=True)
    muon_opt.step()
    muon_opt.zero_grad(set_to_none=True)

for segment_index, segment in enumerate(data_loader):
    # Skip already-completed segments when resuming
    if segment_index < resume_segment:
        print(f"Skipping segment {segment_index + 1} (already checkpointed)")
        continue

    # Encode segment to tokens
    tokens = np.array(gpt.encode(segment))
    print(f"Segment {segment_index + 1}: {len(segment)} chars -> {len(tokens)} tokens")
    # Truncate to fit exact number of sequences
    num_sequences = len(tokens) // sequence_length
    truncated = tokens[:num_sequences * sequence_length]
    # Reshape into 2D array
    sequences = truncated.reshape(num_sequences, sequence_length)
    
    print(f"Segment {segment_index + 1}: Pre-computed {len(sequences)} sequences in memory")

    total_loss = 0
    num_batches = 0
    
    adam_opt.zero_grad(set_to_none=True)
    muon_opt.zero_grad(set_to_none=True)

    for batch_start in range(0, len(sequences), batch_size):
        # Skip incomplete batches to avoid recompilation
        if batch_start + batch_size > len(sequences):
            continue

        # Get batch input and target
        batch_sequences = torch.tensor(sequences[batch_start:batch_start + batch_size], dtype=torch.long, device=gpt.device)
        input_tokens = batch_sequences[:, :-1]
        target_tokens = batch_sequences[:, 1:]

        # Enable mixed precision
        with autocast(device_type=gpt.device.type, dtype=torch.bfloat16):
            output = gpt.forward(input_tokens)  # [batch_size, seq_len-1, vocab_size]
            output = output.reshape(-1, gpt.vocab_size)  # [batch_size * seq_len-1, vocab_size]
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
    if num_batches % gradient_accumulation_steps != 0 and segment_index == num_segments - 1:
        opt_step()
        optimizer_step += 1

    # Get log info
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    output_lr = adam_opt.param_groups[0]["lr"]
    embedding_lr = adam_opt.param_groups[1]["lr"]
    muon_current_lr = muon_opt.param_groups[0]["lr"]

    # Validation if val_data_loader provided
    val_info = ""
    if val_data_loader is not None:
        avg_val_loss, val_perplexity, val_bpb = evaluate(gpt, val_data_loader, sequence_length, batch_size)
        val_info = f", Val Loss: {avg_val_loss:.4f}, Val BPB: {val_bpb:.4f}, Val Perplexity: {val_perplexity:.2f}"

    # Log and save model
    print(
        f"Segment {segment_index + 1}: "
        f"Loss: {avg_loss:.4f}{val_info}, "
        f"Embedding LR: {embedding_lr:.6f}, "
        f"Output LR: {output_lr:.6f}, "
        f"Matrix LR: {muon_current_lr:.6f}, "
        f"Batches: {num_batches}"
    )
    raw_gpt.save(model_path)
    print(f"Segment {segment_index + 1}: Saved to planckgpt.pth")

    # Save training checkpoint
    if checkpoint_path:
        import os
        ckpt_path = os.path.join(checkpoint_path, f"checkpoint_seg{segment_index:04d}.pt")
        torch.save({
            "segment_index": segment_index,
            "optimizer_step": optimizer_step,
            "model_state_dict": raw_gpt.state_dict(),
            "adam_opt_state_dict": adam_opt.state_dict(),
            "muon_opt_state_dict": muon_opt.state_dict(),
        }, ckpt_path)
        print(f"Segment {segment_index + 1}: Checkpoint saved to {ckpt_path}")
