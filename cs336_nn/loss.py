import torch


def cross_entropy(x: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    x = x - x.amax(-1, keepdim=True)
    log_probs = x - torch.log(torch.exp(x).sum(-1, keepdim=True))
    tgt = targets.unsqueeze(-1)
    nll = -log_probs.gather(dim=-1, index=tgt).squeeze(-1)
    return nll.mean()

def perplexity(x: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return torch.exp(cross_entropy(x, targets))
