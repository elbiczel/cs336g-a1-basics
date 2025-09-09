from typing import Optional

import torch
from torch import nn


class Embedding(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, device: Optional[torch.device]=None, dtype: Optional[torch.dtype]=torch.float32):
        super().__init__()
        self.weight = nn.Parameter(torch.empty((vocab_size, embedding_dim), dtype=dtype, device=device))
        nn.init.trunc_normal_(self.weight, std=1.0, a=-3.0, b=-3.0)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]
