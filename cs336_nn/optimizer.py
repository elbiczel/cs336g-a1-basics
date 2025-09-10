from collections.abc import Callable, Iterable
from typing import Optional

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
    def __init__(self, params, lr=1e-3, weight_decay=0.0, betas=(0.99, 0.999), eps=1e-8):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {
            "lr": lr,
            "weight_decay": weight_decay,
            "betas": betas,
            "eps": eps,
        }
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
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
                # Update
                p.data -= curr_lr * m / (v**0.5 + eps)
                # Weight decay
                p.data -= lr * weight_decay * p.data

                # Update state
                state["step"] = step + 1
                state["m"] = m
                state["v"] = v
        return loss

def cosine_lr_schedule(t: int, lr_max: float, lr_min: float, warmup_t: int, t_c: int) -> float:
    if t < warmup_t:
        return lr_max * t / warmup_t
    if t > t_c:
        return lr_min
    lr = 0.5 * (1 + np.cos((t - warmup_t) / (t_c - warmup_t) * np.pi)) * (lr_max - lr_min)
    return lr_min + lr

def clip_grad(params: Iterable[torch.nn.Parameter], max_l2_norm: float, eps: float = 10e-6):
    grads = [p.grad for p in params if p.grad is not None]
    if not grads:
        return
    total_norm = torch.linalg.norm(torch.stack([torch.linalg.norm(g.detach(), ord=2) for g in grads]), ord=2)
    clip_coef = max_l2_norm / (total_norm + eps)
    if clip_coef < 1.0:
        for g in grads:
            g.mul_(clip_coef)
