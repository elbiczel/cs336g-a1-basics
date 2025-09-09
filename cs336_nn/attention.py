import torch
from torch import nn
from einops import einsum, rearrange

from typing import Optional

from cs336_nn._utils import softmax
from cs336_nn import linear

def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Returns Scaled Dot-product attention.
    Args:
        q (Float[Tensor, " ... queries d_k"]): Query tensor
        k (Float[Tensor, " ... keys d_k"]): Key tensor
        v (Float[Tensor, " ... values d_v"]): Values tensor
        mask (Bool[Tensor, " ... queries keys"] | None): Mask tensor
    Returns:
        Float[Tensor, " ... queries d_v"]: Output of SDPA
    """
    scale = q.shape[-1] ** -0.5
    scores = einsum(q, k, "... q d_k, ... k d_k -> ... q k")
    scores = scores * scale
    if mask is not None:
        scores = scores.masked_fill(~mask, -float('inf'))
    attn_weights = softmax(scores, -1)
    return einsum(attn_weights, v, "... q k, ... k d_v -> ... q d_v")

class MultiHeadSelfAttention(nn.Module):
    def __init__(self,
                 d_model: int,
                 num_heads: int,
                 rope: Optional[nn.Module]=None,
                 device: Optional[torch.device]=None,
                 dtype: Optional[torch.dtype]=torch.float32):
        """
        q_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_v"]): Weights for the output projection
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.qkv = linear.Linear(d_model, 3 * d_model, device, dtype)
        self.o_proj = linear.Linear(d_model, d_model, device, dtype)
        self.rope = rope

    def forward(self, x: torch.Tensor, token_positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        h_q = rearrange(q, "... seq_len (h d_k) -> ... h seq_len d_k", h=self.num_heads)
        h_k = rearrange(k, "... seq_len (h d_k) -> ... h seq_len d_k", h=self.num_heads)
        h_v = rearrange(v, "... seq_len (h d_k) -> ... h seq_len d_k", h=self.num_heads)
        seq_len = x.size(-2)
        causal_mask = ~torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device).triu(1)
        if self.rope is not None:
            h_q = self.rope(h_q, token_positions)
            h_k = self.rope(h_k, token_positions)
        self_attention = attention(h_q, h_k, h_v, causal_mask)
        self_attention = rearrange(self_attention, "... h seq_len d_k -> ... seq_len (h d_k)", h=self.num_heads)
        return self.o_proj(self_attention)
