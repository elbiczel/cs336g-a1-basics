from typing import Optional

import torch
from torch import nn
from einops import einsum, reduce

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device: Optional[torch.device]=None, dtype: Optional[torch.dtype]=torch.float32):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model, dtype=dtype, device=device))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_type = x.dtype
        out = x.to(torch.float32)
        rms = torch.sqrt(reduce(out**2 + self.eps, "... d_model -> ... 1", "mean"))
        out = out / rms
        return einsum(out, self.weight, "... d_model, d_model -> ... d_model").to(x_type)
