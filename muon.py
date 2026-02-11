"""Effectively NorMuon from modded-nanogpt but with a more Pytorch optimizer ish interface"""

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
            raise ValueError(f"Invalid ns_steps: {ns_steps} (must be 1-5)")
        
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
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            weight_decay = group["weight_decay"]
            beta2 = group["beta2"]
            ns_steps = group["ns_steps"]
            
            # Collect all 2D parameters (weight matrices) in this group
            params_2d = [p for p in group["params"] if p.grad is not None and p.dim() >= 2]
            
            if not params_2d:
                continue
            
            # Process each parameter
            for param in params_2d:
                grad = param.grad
                state = self.state[param]
                
                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # First moment (momentum buffer)
                    state["momentum_buffer"] = torch.zeros_like(grad)
                    # Second moment (per-neuron variance) - factored representation
                    # Store as column vector or row vector to save memory
                    if grad.size(-2) <= grad.size(-1):
                        # Store per-output-neuron variance (column)
                        state["variance"] = torch.zeros(
                            grad.shape[:-1] + (1,),
                            dtype=torch.float32,
                            device=grad.device
                        )
                        state["reduction_dim"] = -1
                    else:
                        # Store per-input-neuron variance (row)
                        state["variance"] = torch.zeros(
                            grad.shape[:-2] + (1, grad.shape[-1]),
                            dtype=torch.float32,
                            device=grad.device
                        )
                        state["reduction_dim"] = -2
                
                state["step"] += 1
                
                # Get state
                momentum_buffer = state["momentum_buffer"]
                variance = state["variance"]
                reduction_dim = state["reduction_dim"]
                
                # Step 1: Nesterov momentum
                momentum_buffer.lerp_(grad, 1 - momentum)
                g = (1 - momentum) * grad + momentum * momentum_buffer
                
                # Step 2: Polar Express orthogonalization
                g = self._polar_express(g, ns_steps)
                
                # Step 3: Variance reduction (per-neuron adaptive learning rate)
                g = self._variance_reduction(g, variance, beta2, reduction_dim)
                
                # Step 4: Cautious weight decay + parameter update
                if weight_decay > 0:
                    # Only apply weight decay when gradient and param have same sign
                    mask = (g * param) >= 0
                    param.sub_(lr * g + lr * weight_decay * param * mask)
                else:
                    param.sub_(lr * g)
        
        return loss
    
    def _polar_express(
        self, 
        g: torch.Tensor, 
        ns_steps: int
    ) -> torch.Tensor:
        """
        Orthogonalize gradient via Polar Express iteration.
        Computes Q from the polar decomposition G = Q·P.
        """
        # Convert to bfloat16 for speed, normalize
        X = g.bfloat16()
        X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.02 + 1e-6)
        
        # Choose iteration based on matrix shape
        if g.size(-2) > g.size(-1):
            # Tall matrix - use X.T @ X formulation
            for a, b, c in POLAR_EXPRESS_COEFFS[:ns_steps]:
                A = X.mT @ X
                B = b * A + c * (A @ A)
                X = a * X + X @ B
        else:
            # Wide matrix - use X @ X.T formulation  
            for a, b, c in POLAR_EXPRESS_COEFFS[:ns_steps]:
                A = X @ X.mT
                B = b * A + c * (A @ A)
                X = a * X + B @ X
        
        return X.to(g.dtype)
    
    def _variance_reduction(
        self,
        g: torch.Tensor,
        variance: torch.Tensor,
        beta2: float,
        reduction_dim: int,
    ) -> torch.Tensor:
        """
        Normalize per-neuron update scale using variance tracking.
        Similar to Adam's second moment, but factored per-neuron.
        """
        # Compute per-neuron variance
        v_mean = g.float().square().mean(dim=reduction_dim, keepdim=True)
        
        # Update EMA of variance
        variance.lerp_(v_mean.to(variance.dtype), 1 - beta2)
        
        # Compute adaptive step size (like Adam's 1/sqrt(v))
        step_size = variance.clamp_min(1e-10).rsqrt()
        
        # Normalize to preserve total update magnitude
        # This ensures we don't change the overall learning rate scale
        reduction_dim_size = g.size(reduction_dim)
        v_norm_sq = (v_mean * reduction_dim_size).sum(dim=(-2, -1), keepdim=True)
        v_norm = v_norm_sq.sqrt()
        
        scaled_sq_sum = (v_mean * reduction_dim_size) * step_size.float().square()
        v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt()
        
        final_scale = step_size * (v_norm / v_norm_new.clamp_min(1e-10))
        
        return g * final_scale.to(g.dtype)
