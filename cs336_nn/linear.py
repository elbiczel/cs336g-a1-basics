from typing import Optional

import torch
from torch import nn
from einops import einsum


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device: Optional[torch.device]=None, dtype: Optional[torch.dtype]=torch.float32):
        super().__init__()
        self.weight = nn.Parameter(torch.empty((out_features, in_features), dtype=dtype, device=device))
        std=2/(in_features+out_features)
        nn.init.trunc_normal_(self.weight, std=std, a=-3.0*std, b=3*std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")
