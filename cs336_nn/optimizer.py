from collections.abc import Callable, Iterable
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import math

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None: continue
                state = self.state[p]
                step = state.get("step", 0)
                grad = p.grad.data
                p.data -= lr / math.sqrt(step + 1) * grad
                state["step"] = step + 1
        return loss

class AdamW(torch.optim.Optimizer):
    def __init__(
            self,
            params,
            lr:float=1e-3,
            weight_decay:float=0.0,
            betas:Tuple[float, float]=(0.99, 0.999),
            eps:float=1e-8,
            param_names:Dict[int, str]={}):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {
            "lr": lr,
            "weight_decay": weight_decay,
            "betas": betas,
            "eps": eps,
        }
        self._param_names = param_names
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None, return_log=False):
        loss = None if closure is None else closure()
        log_dict = {} if return_log else None
        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            beta_1, beta_2 = group["betas"]
            eps = group["eps"]
            for p in group["params"]:
                if p.grad is None: continue
                state = self.state[p]
                step = state.get("step", 1)
                m = state.get("m", torch.zeros_like(p.data))
                v = state.get("v", torch.zeros_like(p.data))
                grad = p.grad.data
                m = beta_1 * m + (1 - beta_1) * grad
                v = beta_2 * v + (1 - beta_2) * grad**2
                curr_lr = lr * ((1 - beta_2**step)**0.5) / (1 - beta_1**step)
                # Main update
                update_matrix = curr_lr * m / (v**0.5 + eps)
                # Weight decay
                update_matrix += lr * weight_decay * p.data
                if log_dict is not None:
                    param_name = self._param_names.get(id(p), "<unknown>")
                    p_norm = torch.linalg.vector_norm(p.data.detach())
                    log_dict[f"param_norm/{param_name}"] = p_norm
                    update_norm = torch.linalg.vector_norm(update_matrix.detach())
                    log_dict[f"param_update_norm/{param_name}"] = update_norm
                    log_dict[f"param_update_ratio/{param_name}"] = update_norm / p_norm
                p.data -= update_matrix

                # Update state
                state["step"] = step + 1
                state["m"] = m
                state["v"] = v
        return loss, log_dict

def cosine_lr_schedule(t: int, lr_max: float, lr_min: float, warmup_t: int, t_c: int) -> float:
    if t < warmup_t:
        return lr_max * t / warmup_t
    if t > t_c:
        return lr_min
    lr = 0.5 * (1 + np.cos((t - warmup_t) / (t_c - warmup_t) * np.pi)) * (lr_max - lr_min)
    return lr_min + lr

@torch.no_grad()
def clip_grad(params: Iterable[torch.nn.Parameter], max_l2_norm: float, eps: float = 1e-6):
    grads = [p.grad for p in params if p.grad is not None]
    if not grads:
        return
    # vector (elementwise) L2 norms per tensor — no SVD on MPS
    per_tensor = [torch.linalg.vector_norm(g) for g in grads]
    total_norm = torch.linalg.vector_norm(torch.stack(per_tensor))
    if not torch.isfinite(total_norm) or total_norm <= 0.0:
        return
    clip_coef = max_l2_norm / (total_norm + eps)
    if clip_coef < 1.0:
        for g in grads:
            g.mul_(clip_coef)
