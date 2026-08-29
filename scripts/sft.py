import os
import time
import torch
from torch.amp import autocast
from optim.muon import Muon
from bitsandbytes.optim import AdamW8bit
from model.core import GPT
from data.finetune_mix import load_data, load_val_data
from scripts.sft_evaluate import evaluate


# -----------------------------------------------------------------------------
#  Model config
d_model = 896
num_layers = 14
num_heads = 7
rotary_seq_len = 1024
device = torch.device("cuda")

# Decay config
init_lr_frac = 0.8
warmup_ratio = 0.0
warmdown_ratio = 0.5
final_lr_frac = 0.0

# Batch config
sequence_length = 1024
batch_size = 4
gradient_accumulation_steps = 128

# Hyperparams config
scale = 1 / (d_model / 768) ** 0.5 # Scale for different d_model
adam_config = {
    "output":    { "lr": 0.008 * scale, "betas": (0.8, 0.96),  "eps": 1e-10, "weight_decay": 0.0 },
    "embedding": { "lr": 0.3 * scale,   "betas": (0.8, 0.995), "eps": 1e-10, "weight_decay": 0.0 },
    "resid_lambdas": { "lr": 0.5 * 0.01, "betas": (0.8, 0.95), "eps": 1e-10, "weight_decay": 0.0 },
    "x0_lambdas": { "lr": 0.5, "betas": (0.96, 0.95), "eps": 1e-10, "weight_decay": 0.0 },
}
muon_config = {
    "matrix": { "lr": 0.02, "weight_decay": 0.0 }
}
muon_momentum_warmup_steps = 300   # 0.85 -> 0.95

# Eval / logging config
eval_every = 200
eval_batches = 40
log_every = 10
save_every = 200

# Paths
base_model_path = "./artifacts/planckgpt.pth"
base_checkpoint_path = "./artifacts/checkpoints"   # to warm-start the optimizers
model_path = "./artifacts/planckgpt-chat.pth"
# -----------------------------------------------------------------------------


# Initialize model from the pretrained base
raw_gpt = GPT({
    "d_model": d_model,
    "num_layers": num_layers,
    "num_heads": num_heads,
    "rotary_seq_len": rotary_seq_len,
    "device": device
})
raw_gpt.load(base_model_path)
gpt = torch.compile(raw_gpt, mode="max-autotune-no-cudagraphs", dynamic=False)

print(f"Loaded base model from {base_model_path}")
print(f"Using device: {gpt.device}")
print(f"Model parameters: {sum(p.numel() for p in gpt.parameters()):,}")

# Cap context window
sequence_length = min(sequence_length, raw_gpt.rotary_seq_len)

# AdamW for embedding/linear weights
adam_params = [
    { "params": [raw_gpt.output.weight],    **adam_config["output"]    },
    { "params": [raw_gpt.embedding.weight], **adam_config["embedding"] },
    { "params": [raw_gpt.resid_lambdas],    **adam_config["resid_lambdas"] },
    { "params": [raw_gpt.x0_lambdas],       **adam_config["x0_lambdas"] },
]
adam_opt = AdamW8bit(adam_params)

# Muon for transformer params
muon_params = [p for n, p in raw_gpt.named_parameters() if all(key not in n for key in adam_config.keys())]
muon_opt = Muon(muon_params, lr=muon_config["matrix"]["lr"], weight_decay=muon_config["matrix"]["weight_decay"])

# Warm-start the optimizers from the end of pretraining, so Muon's momentum
# buffers and Adam's second moments do not have to be rebuilt from scratch
if base_checkpoint_path:
    import glob
    ckpts = sorted(glob.glob(os.path.join(base_checkpoint_path, "checkpoint_seg*.pt")))
    if ckpts:
        print(f"Warm-starting optimizers from {ckpts[-1]}")
        ckpt = torch.load(ckpts[-1], map_location=raw_gpt.device)

        for opt, key in ((adam_opt, "adam_opt_state_dict"), (muon_opt, "muon_opt_state_dict")):
            incoming = dict(ckpt[key])
            current = opt.state_dict()

            # State is keyed by flattened param index, so the groups must line up
            if len(incoming["param_groups"]) == len(current["param_groups"]):
                incoming["param_groups"] = current["param_groups"]
                opt.load_state_dict(incoming)
            else:
                print(f"Skipping {key}: param group count changed")

        # Deallocate redundant mem
        del ckpt
        torch.cuda.empty_cache()

# Scale down from the pretraining LRs and remember the starting point.
# This happens after the warm-start, which would otherwise restore the old LRs.
for group in adam_opt.param_groups:
    group["lr"] = group["lr"] * init_lr_frac
for group in muon_opt.param_groups:
    group["lr"] = group["lr"] * init_lr_frac
    group["weight_decay"] = muon_config["matrix"]["weight_decay"]
adam_initial_lrs = [group["lr"] for group in adam_opt.param_groups]
muon_initial_lrs = [group["lr"] for group in muon_opt.param_groups]

# Data
print("Building the chat mixture...")
encode = raw_gpt.encoding.encode_ordinary   # never treat user text as special tokens
train_loader = load_data(encode, batch_size, sequence_length)
val_data_loader = lambda: load_val_data(encode, batch_size, sequence_length)

def to_tensors(rows, masks):
    batch_sequences = torch.tensor(rows, dtype=torch.long, device=raw_gpt.device)
    batch_mask = torch.tensor(masks, dtype=torch.bool, device=raw_gpt.device)

    input_tokens = batch_sequences[:, :-1]
    # Positions the assistant is not responsible for become -1, which is the
    # ignore_index core.py passes to cross_entropy
    target_tokens = batch_sequences[:, 1:].clone()
    target_tokens[~batch_mask[:, 1:]] = -1

    return input_tokens, target_tokens

def get_lr_multiplier(progress):
    if progress < warmup_ratio:
        return (progress + 1e-8) / warmup_ratio
    elif progress <= 1.0 - warmdown_ratio:
        return 1.0
    else:
        decay = (progress - (1.0 - warmdown_ratio)) / warmdown_ratio
        return (1 - decay) * 1.0 + decay * final_lr_frac

def opt_step(progress, step):
    # LR schedule
    lrm = get_lr_multiplier(progress)
    for i, group in enumerate(adam_opt.param_groups):
        group["lr"] = adam_initial_lrs[i] * lrm

    # Muon momentum warmup, 0.85 -> 0.95
    frac = min(step / muon_momentum_warmup_steps, 1.0)
    momentum = (1 - frac) * 0.85 + frac * 0.95
    for i, group in enumerate(muon_opt.param_groups):
        group["lr"] = muon_initial_lrs[i] * lrm
        group["momentum"] = momentum

    # Step both optimizers
    adam_opt.step()
    adam_opt.zero_grad(set_to_none=True)
    muon_opt.step()
    muon_opt.zero_grad(set_to_none=True)

    return lrm


# -----------------------------------------------------------------------------
os.makedirs(os.path.dirname(model_path), exist_ok=True)

adam_opt.zero_grad(set_to_none=True)
muon_opt.zero_grad(set_to_none=True)

step = 0
micro_step = 0
progress = 0.0
# Accumulated on the GPU: reading it every micro-step would force a sync and
# serialise the tokenizing and packing against the backward pass
running_loss = torch.zeros((), device=raw_gpt.device)
logged_micro_steps = 0
t0 = time.time()

def prefetch():
    """Pack and upload the next micro-batch. Returns None when the pass ends"""

    item = next(train_loader, None)

    if item is None:
        return None

    rows, masks, batch_progress = item

    return to_tensors(rows, masks), batch_progress

batch = prefetch()

while batch is not None:
    (input_tokens, target_tokens), batch_progress = batch

    # Enable mixed precision
    with autocast(device_type=raw_gpt.device.type, dtype=torch.bfloat16):
        loss = gpt.forward(input_tokens, target_tokens) / gradient_accumulation_steps

    # Propagate grad
    loss.backward()
    running_loss += loss.detach()
    micro_step += 1
    logged_micro_steps += 1
    progress = max(progress, batch_progress)   # Only ever move forwards

    # backward() is queued asynchronously, so the packing for the next
    # micro-batch overlaps with it
    del input_tokens, target_tokens, loss
    batch = prefetch()

    # Update weights every gradient_accumulation_steps
    if micro_step % gradient_accumulation_steps != 0:
        continue

    lrm = opt_step(progress, step)
    step += 1

    # Get log info
    if step % log_every == 0:
        dt = time.time() - t0
        tokens = log_every * gradient_accumulation_steps * batch_size * sequence_length
        avg_loss = (running_loss / logged_micro_steps).item() * gradient_accumulation_steps

        print(
            f"Step: {step:05d} ({100 * progress:.2f}%), "
            f"Loss: {avg_loss:.4f}, LR multiplier: {lrm:.3f}, "
            f"dt: {dt / log_every * 1000:.0f}ms/step, tok/s: {int(tokens / dt):,}, "
            f"vram: {torch.cuda.max_memory_allocated() / 1024 ** 3:.2f}G alloc / "
            f"{torch.cuda.max_memory_reserved() / 1024 ** 3:.2f}G reserved"
        )

        running_loss = torch.zeros((), device=raw_gpt.device)
        logged_micro_steps = 0
        t0 = time.time()

    # Validation
    if eval_every > 0 and step % eval_every == 0:
        torch.cuda.empty_cache()
        avg_val_loss, val_perplexity, val_bpb = evaluate(gpt, val_data_loader, eval_batches)

        print(
            f"Step: {step:05d}, Val Loss: {avg_val_loss:.4f}, "
            f"Val BPB: {val_bpb:.4f}, Val Perplexity: {val_perplexity:.2f}"
        )

        torch.cuda.empty_cache()
        t0 = time.time()

    # Save model
    if save_every > 0 and step % save_every == 0:
        raw_gpt.save(model_path)
        print(f"Step {step:05d}: Saved to {model_path}")
        t0 = time.time()

# Final update if needed
if micro_step % gradient_accumulation_steps != 0:
    opt_step(1.0, step)
    step += 1

torch.cuda.empty_cache()
avg_val_loss, val_perplexity, val_bpb = evaluate(gpt, val_data_loader, eval_batches)

print(f"Final | Val Loss: {avg_val_loss:.4f}, Val BPB: {val_bpb:.4f}, Val Perplexity: {val_perplexity:.2f}")
print(f"Peak VRAM: {torch.cuda.max_memory_allocated() / 1024 ** 3:.2f}G allocated, "
      f"{torch.cuda.max_memory_reserved() / 1024 ** 3:.2f}G reserved")

raw_gpt.save(model_path)
print(f"Done after {step} optimizer steps. Saved chat model to {model_path}")
