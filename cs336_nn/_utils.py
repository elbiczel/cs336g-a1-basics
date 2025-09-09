import torch


def softmax(x: torch.Tensor, dim: int):
    x = x - x.amax(dim, keepdim=True)
    x = torch.exp(x)
    x = x / x.sum(dim, keepdim=True)
    return x
