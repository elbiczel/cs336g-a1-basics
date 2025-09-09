from typing import Optional

import torch
from torch import nn
from einops import einsum


class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, device: Optional[torch.device]=None):
        super().__init__()
        self.register_buffer("thetas",  torch.pow(theta, -2 * torch.arange(d_k // 2, device=device) / d_k))

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # Upcast inputs to float32 if needed, complex32 is not supported for float16s.
        float_type, complex_type = torch.float32, torch.complex64
        if x.dtype == torch.float64:
            float_type, complex_type = torch.float64, torch.complex128
        x_pairs = x.view(*x.shape[:-1], -1, 2)
        x_complex = x_pairs.contiguous().view(float_type).view(complex_type).squeeze(-1)
        # The code above is for PyTorch 2.6.0.
        # Since 2.7.0 a simple `x.view(complex_type)` would suffice.
        rotation_angle = einsum(token_positions, self.thetas, "... seq_len, d_k_half -> ... seq_len d_k_half")
        rotations = torch.exp(1j * rotation_angle)
        x_rotated_complex = x_complex * rotations
        # The above is from Euler's fomrula:
        # e^(i*rotation_angle) = cos(rotation_angle) + i*sin(rotation_angle)
        # which after expansion is equivalent to the rotation matrices.
        return x_rotated_complex.view(x.dtype)
