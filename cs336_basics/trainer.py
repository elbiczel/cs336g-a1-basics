import argparse
import math
import os
import random
import time
from typing import Iterator, Tuple, Optional

import numpy as np
import numpy.typing as npt
import torch

import wandb

from cs336_basics import data, io
import cs336_nn as nn


def parse_params():
    parser = argparse.ArgumentParser(description="Training configuration")

    # Project Data
    parser.add_argument("--project", type=str, default="", help="Project name.")
    parser.add_argument("--train_path", type=str, default="", help="Path to training data.")
    parser.add_argument("--val_path", type=str, default="", help="Path to validation data.")
    parser.add_argument("--models_base_path", type=str, default="", help="Base path to store models.")
    parser.add_argument("--device", type=str, default="", help="Device to use (cpu, cuda, etc).")
    parser.add_argument("--run_name", type=str, default="", help="Run name.")
    parser.add_argument("--group", type=str, default="", help="Experiment group,")
    parser.add_argument(
        "--tags", type=str, nargs="*", default=(), help="List of tags for the run."
    )

    # Model Params
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--vocab_size", type=int, default=10_000, help="Vocabulary size.")
    parser.add_argument("--context_length", type=int, default=256, help="Context length.")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of layers.")
    parser.add_argument("--d_model", type=int, default=512, help="Attention dimension.")
    parser.add_argument("--num_heads", type=int, default=16, help="Number of attention heads.")
    parser.add_argument("--d_ff", type=int, default=1344, help="Feedforward hidden dimension.")
    parser.add_argument("--theta", type=float, default=10_000.0, help="Theta value for RoPE.")

    # Training Options
    parser.add_argument("--max_steps", type=int, default=5000, help="Max steps to do over the dataset during training.")
    parser.add_argument("--compile", type=bool, default=True, help="Whether the model should be compiled before training.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Peak learning rate.")
    parser.add_argument("--final_lr", type=float, default=1e-5, help="Final learning rate.")
    parser.add_argument("--warmup_t", type=int, default=100, help="Number of warm-up steps for the optimizer.")
    parser.add_argument("--t_c", type=int, default=5000, help="Cosine cycle duration. After it's reached uses final_lr.")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay.")
    parser.add_argument("--max_grad_l2_norm", type=float, default=2.0, help="max l2 norm applied for gradient clipping. If <= 0.0 no clipping is applied.")
    parser.add_argument("--z_loss_weight", type=float, default=1e-4, help="Weight for the Z-loss.")

    # Logging Options
    parser.add_argument(
        "--log_freq", type=int, default=20, help="Log fast train/val metrics from a single batch."
    )
    parser.add_argument(
        "--val_steps", type=int, default=5, help="Number of batches to evaluate validation loss on."
    )
    parser.add_argument("--checkpoint_step_freq", type=int, default=100, help="Checkpoint model every N steps. If 0 only saves the final checkpoint.")

    return parser.parse_args()


def dataset_iterator(
        d: npt.NDArray[np.uint16],
        shuffle: bool = True,
        max_steps: Optional[int] = None,
) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    cfg = wandb.config
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

@torch.no_grad()
def get_validation_metrics(model, val_data) -> dict:
    cfg = wandb.config
    model.eval()
    val_correct = val_count = 0; val_loss = 0.0
    for xb, yb in dataset_iterator(val_data, shuffle=True, max_steps=cfg.val_steps):
        logits = model(xb)
        loss = nn.loss.cross_entropy(logits, yb)
        val_loss += loss.item() * yb.numel()
        pred = logits.argmax(dim=-1)
        val_correct += (pred == yb).sum().item()
        val_count += yb.numel()
    model.train()
    return {
        "val/loss": val_loss / val_count,
        "val/acc": val_correct / val_count,
        "val/perplexity": math.exp(val_loss / val_count)
    }

def train(models_dir: str):
    cfg = wandb.config
    train_data = io.read_tokens(cfg.train_path)
    val_data = io.read_tokens(cfg.val_path)

    # IMPORTANT:
    # TODO: Adjust HParams: lr, final_lr, warmup_t, beta_1, beta_2, adam_eps, weight_decay
    # Nice to have:
    # TODO: Try adding post-Norm (inside residual).
    # TODO: Add QK LayerNorms in Attention - for softmax stability there.
    # TODO: Try weight tying: embeddings == lm_head.
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
    opt = nn.optimizer.AdamW(optimizer_grouped_parameters, cfg.lr, cfg.weight_decay)
    wandb.watch(model, log="all", log_freq=cfg.log_freq)

    global_step, t0 = 0, time.perf_counter()
    model.train()
    step_time_accum = 0.0
    step_t0 = time.perf_counter()
    
    for xb, yb in dataset_iterator(train_data, shuffle=True, max_steps=cfg.max_steps):
        logits = model(xb)
        ce_loss = nn.loss.cross_entropy(logits, yb)        
        logZ = torch.logsumexp(logits, dim=-1)
        z_loss = (logZ ** 2).mean()
        loss = ce_loss
        if cfg.z_loss_weight > 0.0:
            loss += (z_loss * cfg.z_loss_weight)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.max_grad_l2_norm > 0.0:
            nn.optimizer.clip_grad(model.parameters(), cfg.max_grad_l2_norm)
        for grp in opt.param_groups:
            grp["lr"] = max(nn.optimizer.cosine_lr_schedule(global_step, cfg.lr, cfg.final_lr, cfg.warmup_t, cfg.t_c), 1e-8)
        opt.step()

        # stats
        now = time.perf_counter()
        step_time = now - step_t0
        step_time_accum += step_time
        step_t0 = now
        with torch.no_grad():
            pred = logits.argmax(dim=-1)
            correct = (pred == yb).sum().item()

        if (global_step > 0 and global_step % cfg.log_freq == 0) or global_step == cfg.max_steps-1:
            val_metrics = get_validation_metrics(model, val_data)
            per_param_stats = {
                f"grad_norm/{name}": torch.linalg.vector_norm(p.grad.detach())
                for name, p in model.named_parameters()
                if p.grad is not None
            }
            total_grad_norm = torch.linalg.vector_norm(torch.stack(list(per_param_stats.values())))
            per_param_stats.update({
                f"grad_rms/{name}": p.grad.detach().pow(2).mean().sqrt()
                for name, p in model.named_parameters()
                if p.grad is not None
            })
            per_param_stats.update({
                f"param_numel/{name}": torch.tensor(p.numel())
                for name, p in model.named_parameters()
                if p.grad is not None
            })
            opt_lr = opt.param_groups[0]["lr"]
            sec_per_step = step_time_accum / cfg.log_freq
            step_time_accum = 0.0
            wandb.log({
                "global_step": global_step,
                "train/loss": loss.item(),
                "train/ce_loss": ce_loss.item(),
                "train/z_loss": z_loss.item(),
                "train/acc": correct / yb.numel(),
                "train/perplexity": math.exp(loss.item()),
                "time/sec_per_step": sec_per_step,
                "time/minutes": (time.perf_counter() - t0) / 60.0,
                "opt/lr": opt_lr,
                "opt/grad_norm_total": total_grad_norm.item(),
            } | {k: v.item()  for k, v in per_param_stats.items()}| val_metrics)
        if global_step > 0 and cfg.checkpoint_step_freq > 0 and global_step % cfg.checkpoint_step_freq == 0:
            chkpt_path = f"{models_dir}/step_{global_step}.pt"
            data.save_checkpoint(model, opt, global_step, chkpt_path)
        global_step += 1

    # TODO: Track and save the best checkpoint.
    chkpt_path = f"{models_dir}/final.pt"
    data.save_checkpoint(model, opt, global_step, chkpt_path)
    wandb.save(chkpt_path)


def main():
    params = parse_params()

    run = wandb.init(
        project=params.project,
        name=params.run_name,
        group=params.group,
        job_type='train',
        tags=list(params.tags),
        config=vars(params),
    )
    wandb.define_metric("global_step")
    wandb.define_metric("train/*", step_metric="global_step")
    wandb.define_metric("val/*", step_metric="global_step")
    wandb.define_metric("opt/*", step_metric="global_step")
    wandb.define_metric("grad_norm/*", step_metric="global_step")
    wandb.define_metric("grad_rms/*", step_metric="global_step")
    wandb.define_metric("param_numel/*", step_metric="global_step")
    wandb.define_metric("time/*", step_metric="global_step")
    cfg = wandb.config
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

    models_dir = f"{cfg.models_base_path}/{cfg.project}/{cfg.group}/{cfg.run_name}"
    os.makedirs(models_dir, exist_ok=True)
    train(models_dir)

    run.finish()


if __name__ == "__main__":
    main()
