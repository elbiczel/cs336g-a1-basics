import torch
from torch import nn

from typing import Optional

from cs336_nn import attention, activation, embedding, linear, rope, layer_norm
from cs336_nn._utils import softmax     

class TransformerBlock(nn.Module):
    def __init__(self,
                 d_model: int,
                 num_heads: int,
                 d_ff: int,
                 theta: float,
                 max_seq_len: int,
                 device: Optional[torch.device]=None,
                 dtype: Optional[torch.dtype]=torch.float32):
        super().__init__()
        self.attn = attention.MultiHeadSelfAttention(d_model, num_heads, rope.RoPE(theta, d_model // num_heads, max_seq_len, device), device, dtype)
        self.ffn = activation.SwiGLU(d_model, d_ff, device, dtype)
        self.ln1 = layer_norm.RMSNorm(d_model, device=device, dtype=dtype)
        self.ln2 = layer_norm.RMSNorm(d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = x + self.attn(self.ln1(x))
        h2 = h1 + self.ffn(self.ln2(h1))
        return h2


class TransformerLM(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 context_length: int,
                 num_layers: int,
                 d_model: int,
                 num_heads: int,
                 d_ff: int,
                 theta: float,
                 device: Optional[torch.device]=None,
                 dtype: Optional[torch.dtype]=torch.float32):
        super().__init__()
        self.token_embeddings = embedding.Embedding(vocab_size, d_model, device, dtype)
        self.layers = nn.ModuleList([TransformerBlock(d_model, num_heads, d_ff, theta, context_length, device, dtype) for _ in range(num_layers)])
        self.ln_final = layer_norm.RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = linear.Linear(d_model, vocab_size, device, dtype)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.token_embeddings(x)
        for block in self.layers:
            h = block(h)
        logits = self.lm_head(self.ln_final(h))
        return logits
