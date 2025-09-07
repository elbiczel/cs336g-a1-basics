from typing import Optional

import torch
from torch import nn
from einops import einsum

from cs336_nn import linear

def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)

class SwiGLU(nn.Module):
    # Note: d_ff is usually 8/3 * d_model. But needs to be a multiplication of 64 for memory alignment.
    def __init__(self, d_model: int, d_ff: int, device: Optional[torch.device]=None, dtype: Optional[torch.dtype]=torch.float32):
        super().__init__()
        self.w1 = linear.Linear(d_model, d_ff, device, dtype)
        self.w2 = linear.Linear(d_ff, d_model, device, dtype)
        self.w3 = linear.Linear(d_model, d_ff, device, dtype)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        act = silu(self.w1(x))
        res = self.w3(x)
        out = einsum(act, res, "... d_ff, ... d_ff -> ... d_ff")
        return self.w2(out)
