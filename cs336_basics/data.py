import numpy as np
import numpy.typing as npt
import torch
import os
from typing import BinaryIO, IO, Optional

def get_batch(
        data: npt.NDArray[np.uint16],
        batch_size: int,
        context_length: int,
        device: str,
        start: Optional[int] = None) -> tuple[torch.Tensor, torch.Tensor]:
    N = data.shape[0]
    elements = context_length + 1
    if elements > N:
        raise ValueError(f"context_length+1 ({elements}) exceeds data length ({N}).")

    if start is None:
        starts = np.random.randint(0, N - elements + 1, size=batch_size)
    else:
        starts = start + np.arange(batch_size)
    idx = starts[:, None] + np.arange(elements)
    seqs = data[idx].astype(np.int64)

    x_cpu = torch.from_numpy(seqs[:, :-1])
    y_cpu = torch.from_numpy(seqs[:, 1:])
    batch = x_cpu.to(device, non_blocking=True)
    targets = y_cpu.to(device, non_blocking=True)
    return batch, targets

def save_checkpoint(
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        iteration: int,
        out: str | os.PathLike | BinaryIO | IO[bytes]):
    data = {}
    data["model"] = model.state_dict()
    data["opt"] = optimizer.state_dict()
    data["iter"] = iteration
    torch.save(data, out)

def load_checkpoint(
        src: str | os.PathLike | BinaryIO | IO[bytes],
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer]) -> int:
    data = torch.load(src)
    model.load_state_dict(data["model"])
    if optimizer is not None:
        optimizer.load_state_dict(data["opt"])
    return data["iter"]
