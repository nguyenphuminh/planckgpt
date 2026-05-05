"""Effectively NorMuon from nanochat but with a more Pytorch optimizer ish interface"""

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer
from collections import defaultdict

# Coefficients for Polar Express (num_iters=5)
POLAR_EXPRESS_COEFFS = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]

@torch.compile(dynamic=False, fullgraph=True)
def muon_step_fused(
    stacked_grads: Tensor,          # (N, H, W) - stacked gradients
    stacked_params: Tensor,         # (N, H, W) - stacked parameters
    momentum_buffer: Tensor,        # (N, H, W) - first moment buffer
    second_momentum_buffer: Tensor, # (N, H, 1) or (N, 1, W) - factored second moment
    momentum_t: Tensor,             # () - 0-D CPU tensor
    lr_t: Tensor,                   # () - 0-D CPU tensor
    wd_t: Tensor,                   # () - 0-D CPU tensor
    beta2_t: Tensor,                # () - 0-D CPU tensor
    ns_steps: int,
    red_dim: int,                   # -1 or -2
) -> None:
    # Nesterov momentum
    momentum = momentum_t.to(stacked_grads.dtype)
    momentum_buffer.lerp_(stacked_grads, 1 - momentum)
    g = stacked_grads.lerp_(momentum_buffer, momentum)

    # Polar Express orthogonalization
    X = g.bfloat16()
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-6)
    if g.size(-2) > g.size(-1):  # Tall matrix
        for a, b, c in POLAR_EXPRESS_COEFFS[:ns_steps]:
            A = X.mT @ X
            B = b * A + c * (A @ A)
            X = a * X + X @ B
    else:  # Wide matrix
        for a, b, c in POLAR_EXPRESS_COEFFS[:ns_steps]:
            A = X @ X.mT
            B = b * A + c * (A @ A)
            X = a * X + B @ X
    g = X.to(stacked_grads.dtype)

    # Variance reduction
    v_mean = g.float().square().mean(dim=red_dim, keepdim=True)
    v_scaled = v_mean * g.size(red_dim)
    v_norm = v_scaled.sum(dim=(-2, -1), keepdim=True).sqrt()
    second_momentum_buffer.lerp_(v_mean.to(second_momentum_buffer.dtype), 1 - beta2_t.to(second_momentum_buffer.dtype))
    step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt()
    v_norm_new = (v_scaled * step_size.float().square()).sum(dim=(-2, -1), keepdim=True).sqrt()
    g = g * (step_size * (v_norm / v_norm_new.clamp_min(1e-10))).to(g.dtype)

    # Cautious weight decay + parameter update
    lr = lr_t.to(g.dtype)
    wd = wd_t.to(g.dtype)
    mask = (g * stacked_params) >= 0
    stacked_params.sub_(lr * g + lr * wd * stacked_params * mask)


class Muon(Optimizer):
    """
    Muon optimizer - Orthogonalized gradients with per-neuron variance normalization.

    Args:
        params: Iterable of parameters to optimize
        lr: Learning rate (default: 0.02)
        momentum: Nesterov momentum coefficient (default: 0.95)
        weight_decay: Cautious weight decay coefficient (default: 0.0)
        beta2: Second moment decay rate for variance reduction (default: 0.95)
        ns_steps: Number of Polar Express iterations (default: 5, max: 5)
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        weight_decay: float = 0.0,
        beta2: float = 0.9,
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
            raise ValueError(f"Invalid ns_steps: {ns_steps} (must be 1-5)")

        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, beta2=beta2, ns_steps=ns_steps)
        super().__init__(params, defaults)

        # 0-D CPU tensors to avoid torch.compile recompilation when hyperparams change
        self._momentum_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._lr_t       = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._wd_t       = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._beta2_t    = torch.tensor(0.0, dtype=torch.float32, device="cpu")

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_2d = [p for p in group["params"] if p.grad is not None and p.dim() >= 2]
            if not params_2d:
                continue

            # Group params by shape so each batch can be stacked into one kernel call
            by_shape = defaultdict(list)
            for p in params_2d:
                by_shape[p.shape].append(p)

            for shape, params in by_shape.items():
                p0 = params[0]
                state = self.state[p0]
                n = len(params)
                H, W = shape[-2], shape[-1]
                red_dim = -1 if H >= W else -2

                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros(n, *shape, dtype=p0.dtype, device=p0.device)
                    buf_shape = (n, H, 1) if H >= W else (n, 1, W)
                    state["second_momentum_buffer"] = torch.zeros(buf_shape, dtype=p0.dtype, device=p0.device)

                stacked_grads  = torch.stack([p.grad for p in params])
                stacked_params = torch.stack(params)

                self._momentum_t.fill_(group["momentum"])
                self._beta2_t.fill_(group["beta2"])
                # Scale lr by sqrt(max(1, H/W)) to equalize update magnitude across shapes
                self._lr_t.fill_(group["lr"] * max(1.0, H / W) ** 0.5)
                self._wd_t.fill_(group["weight_decay"])

                muon_step_fused(
                    stacked_grads,
                    stacked_params,
                    state["momentum_buffer"],
                    state["second_momentum_buffer"],
                    self._momentum_t,
                    self._lr_t,
                    self._wd_t,
                    self._beta2_t,
                    group["ns_steps"],
                    red_dim,
                )

                # Copy updated values back to original params
                torch._foreach_copy_(params, list(stacked_params.unbind(0)))

        return loss
