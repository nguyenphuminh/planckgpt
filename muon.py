"""Effectively NorMuon from modded-nanogpt but with a more Pytorch optimizer ish interface.

Changes from original:
- Momentum buffer and variance stored in BF16 (half the memory vs FP32)
- All optimizer state math runs in BF16
- _polar_express and _variance_reduction compiled with torch.compile
- clamp floor raised to 1e-6 (1e-10 underflows in BF16 rsqrt)
"""

import torch
from torch.optim.optimizer import Optimizer

# Coefficients for Polar Express (num_iters=5)
POLAR_EXPRESS_COEFFS = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]

# ── Compiled compute functions ────────────────────────────────────────────────
# Extracted at module level so torch.compile can cache traces across optimizer
# steps. torch.compile on instance methods re-traces every call.

@torch.compile
def _polar_express(
    g: torch.Tensor,
    ns_steps: int,
) -> torch.Tensor:
    """
    Orthogonalize gradient via Polar Express iteration.
    Computes Q from the polar decomposition G = Q·P.
    Already in BF16 on entry; returns BF16.
    """
    X = g.bfloat16()
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-6)

    # torch.compile will specialize on the g.size(-2) > g.size(-1) branch,
    # caching one trace per shape orientation (tall vs wide).
    if g.size(-2) > g.size(-1):
        for a, b, c in POLAR_EXPRESS_COEFFS[:ns_steps]:
            A = X.mT @ X
            B = b * A + c * (A @ A)
            X = a * X + X @ B
    else:
        for a, b, c in POLAR_EXPRESS_COEFFS[:ns_steps]:
            A = X @ X.mT
            B = b * A + c * (A @ A)
            X = a * X + B @ X

    return X  # already BF16


@torch.compile
def _variance_reduction(
    g: torch.Tensor,
    variance: torch.Tensor,
    beta2: float,
    reduction_dim: int,
) -> torch.Tensor:
    """
    Normalize per-neuron update scale using variance tracking.
    All tensors are BF16.

    Note: clamp floor is 1e-6 rather than 1e-10 — BF16 rsqrt of values
    below ~1e-19 produces inf due to limited precision.
    """
    # Per-neuron squared mean — stay in BF16
    v_mean = g.square().mean(dim=reduction_dim, keepdim=True)

    # EMA update in-place (both BF16)
    variance.lerp_(v_mean, 1 - beta2)

    # Adaptive step size
    step_size = variance.clamp_min(1e-6).rsqrt()

    # Preserve total update magnitude across neurons
    reduction_dim_size = g.size(reduction_dim)
    v_scaled = v_mean * reduction_dim_size
    v_norm = v_scaled.sum(dim=(-2, -1), keepdim=True).sqrt()

    v_norm_new = (v_scaled * step_size.square()).sum(
        dim=(-2, -1), keepdim=True
    ).sqrt()

    final_scale = step_size * (v_norm / v_norm_new.clamp_min(1e-6))

    return g * final_scale


# ── Optimizer ─────────────────────────────────────────────────────────────────

class Muon(Optimizer):
    """
    Muon optimizer — orthogonalized gradients with per-neuron variance
    normalization. Applies only to ≥2D parameters (weight matrices).
    Use AdamW / AdamW8bit for embeddings, biases, and the LM head.

    State is stored in BF16, halving optimizer memory vs the FP32 original.

    Args:
        params:       Iterable of parameters to optimize
        lr:           Learning rate (default: 0.02)
        momentum:     Nesterov momentum coefficient (default: 0.95)
        weight_decay: Cautious weight decay coefficient (default: 0.0)
        beta2:        Second-moment EMA decay for variance reduction (default: 0.95)
        ns_steps:     Polar Express iterations (default: 5, max: 5)
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        weight_decay: float = 0.0,
        beta2: float = 0.95,
        ns_steps: int = 5,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Invalid momentum: {momentum}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        if not 0.0 <= beta2 < 1.0:
            raise ValueError(f"Invalid beta2: {beta2}")
        if not 1 <= ns_steps <= 5:
            raise ValueError(f"Invalid ns_steps: {ns_steps} (must be 1–5)")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            beta2=beta2,
            ns_steps=ns_steps,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr           = group["lr"]
            momentum     = group["momentum"]
            weight_decay = group["weight_decay"]
            beta2        = group["beta2"]
            ns_steps     = group["ns_steps"]

            params_2d = [
                p for p in group["params"]
                if p.grad is not None and p.dim() >= 2
            ]
            if not params_2d:
                continue

            for param in params_2d:
                # Always work in BF16; cast grad once up-front
                grad  = param.grad.bfloat16()
                state = self.state[param]

                # ── State initialisation ──────────────────────────────────
                if len(state) == 0:
                    state["step"] = 0

                    # Momentum buffer in BF16 (was: zeros_like(grad) → FP32/BF16)
                    state["momentum_buffer"] = torch.zeros_like(
                        grad, dtype=torch.bfloat16
                    )

                    # Factored variance in BF16 (was: float32)
                    # Tall/square → reduce cols, store per-row (shape [..., 1])
                    # Wide        → reduce rows, store per-col (shape [..., 1, n])
                    if grad.size(-2) >= grad.size(-1):
                        state["variance"] = torch.zeros(
                            grad.shape[:-1] + (1,),
                            dtype=torch.bfloat16,
                            device=grad.device,
                        )
                        state["reduction_dim"] = -1
                    else:
                        state["variance"] = torch.zeros(
                            grad.shape[:-2] + (1, grad.shape[-1]),
                            dtype=torch.bfloat16,
                            device=grad.device,
                        )
                        state["reduction_dim"] = -2

                state["step"] += 1

                momentum_buffer = state["momentum_buffer"]
                variance        = state["variance"]
                reduction_dim   = state["reduction_dim"]

                # ── Step 1: Nesterov momentum (BF16 throughout) ───────────
                momentum_buffer.lerp_(grad, 1 - momentum)
                g = (1 - momentum) * grad + momentum * momentum_buffer

                # ── Step 2: Polar Express orthogonalisation ───────────────
                g = _polar_express(g, ns_steps)

                # ── Step 3: Variance reduction ────────────────────────────
                g = _variance_reduction(g, variance, beta2, reduction_dim)

                # ── Step 4: Cautious weight decay + update ────────────────
                if weight_decay > 0:
                    # Apply decay only where gradient and param share sign
                    mask = (g * param.bfloat16()) >= 0
                    param.sub_(
                        (lr * g + lr * weight_decay * param.bfloat16() * mask).to(param.dtype)
                    )
                else:
                    param.sub_(g.to(param.dtype) * lr)

        return loss
