from typing import Optional

import torch
from torch import nn
from einops import einsum

def _compute_freqs_cis(
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: Optional[torch.device]=None,
        dtype: Optional[torch.dtype]=torch.float32) -> torch.Tensor:
    inv_freq = (theta ** (torch.arange(0, d_k, 2, device=device, dtype=dtype) / d_k)).reciprocal()
    # Precompute rotation angles for all possible sequence lengths.
    t = torch.arange(max_seq_len, device=device, dtype=dtype)
    freqs = einsum(t, inv_freq, "i, j -> i j")
    # e^(1j * freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis.to(torch.complex64 if dtype in (torch.float16, torch.bfloat16, torch.float32) else torch.complex128)


class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: Optional[torch.device]=None, dtype: Optional[torch.dtype]=torch.float32):
        super().__init__()
        if d_k % 2 != 0:
            raise ValueError(f"d_k must be even for RoPE, got {d_k}.")
        self.register_buffer(
            "freqs_cis",
            _compute_freqs_cis(theta, d_k, max_seq_len, device, dtype),
            persistent=False,
        )
        

    def forward(self, x: torch.Tensor, token_positions: Optional[torch.Tensor]) -> torch.Tensor:
        orig_dtype = x.dtype
        *prefix, seq, d_k = x.shape
        if d_k % 2 != 0:
            raise ValueError(f"Last dim must be even, got {d_k}.")
        x_pairs = x.view(*prefix, seq, d_k // 2, 2)
        # Upcast inputs to types supported by complex.
        float_type = torch.float64 if orig_dtype == torch.float64 else torch.float32
        x_complex = torch.view_as_complex(x_pairs.to(float_type))
        # The code above is for PyTorch 2.6.0.
        # Since 2.7.0 a simple `x.view(complex_type)` would suffice.

        # Select rotation factors
        if token_positions is None:
            token_positions = torch.arange(seq, device=x.device)
        # Broadcast positions to shape [..., seq] then index
        if token_positions.ndim == 1:
            freqs_cis = self.freqs_cis[token_positions]
            # expand to match prefix dims
            while freqs_cis.ndim < x_complex.ndim:
                freqs_cis = freqs_cis.unsqueeze(0)
        else:
            #flat_pos = token_positions.reshape(-1)
            freqs_cis = self.freqs_cis[token_positions]
        
        x_rot = x_complex * freqs_cis
        x_real = torch.view_as_real(x_rot).reshape(*prefix, seq, d_k)
        return x_real.to(orig_dtype)
