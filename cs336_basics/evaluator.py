import argparse
import math
import random
from types import SimpleNamespace
from typing import Iterator, Tuple, Optional

import numpy as np
import numpy.typing as npt
import torch

import wandb

from cs336_basics import data, io
import cs336_nn as nn


def parse_params():
    parser = argparse.ArgumentParser(description="Training configuration")
    # Basics
    parser.add_argument("--project", type=str, default="", help="Project name.")
    parser.add_argument("--trainer_run", type=str, default="", help="The run id to extract the model config and artefact from.")
    parser.add_argument("--models_base_path", type=str, default="", help="Base path to store models.")
    parser.add_argument("--device", type=str, default="", help="Device to use (cpu, cuda, etc).")
    parser.add_argument("--data_override", type=str, default="", help="If set uses this dataset. Otherwise uses val_path from the model.")

     # Model params
    parser.add_argument("--compile", type=bool, default=True, help="Whether the model should be compiled before generation.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--resume", type=bool, default=False, help="If true, reas the resume checkpoint from the model, not the best one.")

    # Evaluator options
    parser.add_argument(
        "--eval_steps", type=int, default=100, help="Number of batches to evaluate validation loss on. If 0.0 runs of the full dataset."
    )
    parser.add_argument(
        "--log_freq", type=int, default=10, help="Batch metrics logging frequency."
    )

    return parser.parse_args()


def dataset_iterator(
        cfg,
        d: npt.NDArray[np.uint16],
        shuffle: bool = True,
        max_steps: Optional[int] = None,
) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    num_iterations = len(d) // cfg.batch_size
    if max_steps:
        num_iterations = min(num_iterations, max_steps)

    i = 0
    start = 0 if not shuffle else None
    while i < num_iterations:
        yield data.get_batch(d, cfg.batch_size, cfg.context_length, cfg.device, start)
        if start is not None:
            start += cfg.batch_size
        i += 1

def print_metrics(metrics):
    for k, v in metrics.items():
        print(f"{k} => {v}")


@torch.no_grad()
def eval(cfg):
    # Prep data
    test_data = io.read_tokens(cfg.data_override or cfg.val_path)

    # Prep model.
    model_name = "resume" if cfg.resume else "best"
    model_path = f"{cfg.models_base_path}/{cfg.project}/{cfg.group}/{cfg.run_name}/{model_name}.pt"
    model = nn.transformer.TransformerLM(
        cfg.vocab_size,
        cfg.context_length,
        cfg.num_layers,
        cfg.d_model,
        cfg.num_heads,
        cfg.d_ff,
        cfg.theta,
    ).to(cfg.device)
    if cfg.compile:
        if cfg.device == "cpu":
            model = torch.compile(model)
        elif cfg.device == "mps":
            model = torch.compile(model, backend="aot_eager")
        else:
            model = torch.compile(model, mode="max-autotune")
    if cfg.resume:
        param_dict = {}
        decay_params = set()
        no_decay_params = set()
        for module_name, module in model.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                full_name = f"{module_name}.{param_name}" if module_name else param_name
                assert full_name not in param_dict
                param_dict[full_name] = param
                if isinstance(module, (nn.layer_norm.RMSNorm, nn.embedding.Embedding)):
                    no_decay_params.add(full_name)
                else:
                    decay_params.add(full_name)
        optimizer_grouped_parameters = [
            # Default group.
            {"params": [param_dict[n] for n in sorted(decay_params)]},
            # Params without weight decay
            {
                "params": [param_dict[n] for n in sorted(no_decay_params)],
                "weight_decay": 0.0
            },
        ]
        opt = nn.optimizer.AdamW(
            optimizer_grouped_parameters,
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            betas=(cfg.adam_beta_1, cfg.adam_beta_2),
            eps=cfg.adam_eps
        )
    else:
        opt= None
    data.load_checkpoint(model_path, model, opt)
    model.eval()
    
    step = total_correct = total_count = 0; total_loss = 0.0
    for (xb, yb) in dataset_iterator(cfg, test_data, shuffle=cfg.eval_steps > 0, max_steps=cfg.eval_steps):
        step += 1
        logits = model(xb)
        batch_loss = nn.loss.cross_entropy(logits, yb)
        total_loss += batch_loss.item() * yb.numel()
        pred = logits.argmax(dim=-1)
        batch_correct = (pred == yb).sum().item() 
        total_correct += batch_correct
        total_count += yb.numel()
        if step % cfg.log_freq == 0:
            metrics = {
                "step": step,
                "batch/loss": batch_loss.item(),
                "batch/acc": batch_correct / yb.numel(),
                "batch/perplexity": math.exp(batch_loss.item())
            }
            print_metrics(metrics)
    metrics = {
        "total/loss": total_loss / total_count,
        "total/acc": total_correct / total_count,
        "total/perplexity": math.exp(total_loss / total_count)
    }
    print_metrics(metrics)


def main():
    params = parse_params()
    api = wandb.Api()
    trainer_run = api.run(f"elbiczel-personal/{params.project}/{params.trainer_run}")
    trainer_config = trainer_run.config
    cfg = trainer_config | vars(params)
    cfg = SimpleNamespace(cfg)
    device = cfg.device
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("Tried running on CUDA, but it's not available")
    if device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("Tried running on Apple Silicon, but it's not available")
    if device == "mps" and not torch.backends.mps.is_built():
        raise ValueError("Tried running on Apple Silicon, but it's not built")
    # Would be better to use the generator APIs for both torch and np.
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    eval(cfg)


if __name__ == "__main__":
    main()
