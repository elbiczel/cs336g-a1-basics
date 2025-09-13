import argparse
import os
import random
import time
from typing import Iterator, Tuple

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
    parser.add_argument("--compile", type=bool, default=False, help="Whether the model should be compiled before training.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size.")
    parser.add_argument("--lr", type=float, default=3e-3, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay.")

    # Logging Options
    parser.add_argument("--max_epochs", type=int, default=5, help="Maximum epochs.")
    parser.add_argument(
        "--max_steps_per_epoch", type=int, default=0, help="Max steps per epoch (default: 0 means use all steps)."
    )
    parser.add_argument(
        "--log_freq", type=int, default=50, help="Log fast train/val metrics from a single batch."
    )
    parser.add_argument("--checkpoint_step_freq", type=int, default=0, help="Checkpoint model every N steps. If 0 only checkpoint_epoch_freq is used.")
    parser.add_argument("--checkpoint_epoch_freq", type=int, default=1, help="Checkpoint model every N steps.")

    return parser.parse_args()


def epoch_iterator(
        d: npt.NDArray[np.uint16],
        shuffle: bool = True,
) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    # Statistically we should iterate over the whole dataset, but we sample with returns.
    cfg = wandb.config
    num_iterations = len(d) // cfg.batch_size
    if cfg.max_steps_per_epoch:
        num_iterations = min(num_iterations, cfg.max_steps_per_epoch)

    i = 0
    start = 0 if not shuffle else None
    while i < num_iterations:
        yield data.get_batch(d, cfg.batch_size, cfg.context_length, cfg.device, start)
        if start is not None:
            start += cfg.batch_size
        i += 1


def train(models_dir: str):
    cfg = wandb.config
    train_data = io.read_tokens(cfg.train_path)
    val_data = io.read_tokens(cfg.val_path)

    # IMPORTANT:
    # TODO: Add per-token perplexity loss in reporting.
    # TODO: Fix loss logging to consistently log per-token loss.
    # TODO: Clip Gradients.
    # TODO: Adjust learning rate schedule, e.g. cosine_lr_schedule?
    # TODO: Adjust HParams, e.g. betas are lower for LLMs.
    # Nice to have:
    # TODO: Change loss, e.g. to Z-loss for numerical stability.
    # TODO: Try adding post-Norm (inside residual).
    # TODO: Add QK LayerNorms in Attention - for softmax stability there.
    # TODO: Add wandb watch, especially to observe gradient norms.
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
    opt = nn.optimizer.AdamW(model.parameters(), cfg.lr, cfg.weight_decay)

    global_step, t0 = 0, time.perf_counter()
    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0; epoch_count = 0
        step_time_accum = 0.0

        step_t0 = time.perf_counter()

        for xb, yb in epoch_iterator(train_data, shuffle=True):
            logits = model(xb)
            loss = nn.loss.cross_entropy(logits, yb)

            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()

            global_step += 1

            # stats
            with torch.no_grad():
                pred = logits.argmax(dim=-1)
                correct = (pred == yb).sum().item()
            batch_size = yb.numel()
            epoch_loss += loss.item() * batch_size
            epoch_correct += correct
            epoch_count += batch_size

            now = time.perf_counter()
            step_time = now - step_t0
            step_time_accum += step_time
            step_t0 = now

            if global_step % cfg.log_freq == 0:
                with torch.no_grad():
                    xb, yb = data.get_batch(val_data, cfg.batch_size, cfg.context_length, cfg.device)
                    logits = model(xb)
                    val_loss = nn.loss.cross_entropy(logits, yb).item()
                    val_pred = logits.argmax(dim=-1)
                    val_correct = (val_pred == yb).sum().item()
                    val_size = yb.numel()
                    total_norm = torch.norm(torch.stack([p.grad.detach().data.norm(2) 
                                        for p in model.parameters() if p.grad is not None]), 2).item()
                opt_lr = opt.param_groups[0]["lr"]
                sec_per_step = step_time_accum / cfg.log_freq
                step_time_accum = 0.0
                wandb.log({
                    "global_step": global_step,
                    "train/loss": loss.item(),
                    "train/acc_batch": correct / batch_size,
                    "val/loss": val_loss,
                    "val/acc_batch": val_correct / val_size,
                    "time/sec_per_step": sec_per_step,
                    "time/minutes": (time.perf_counter() - t0) / 60.0,
                    "opt/grad_norm": total_norm,
                    "opt/lr": opt_lr,
                })
            if cfg.checkpoint_step_freq > 0 and global_step % cfg.checkpoint_step_freq == 0:
                chkpt_path = f"{models_dir}/step_{global_step}.pt"
                data.save_checkpoint(model, opt, global_step, chkpt_path)
        # end of epoch: aggregate train metrics
        train_loss = epoch_loss / epoch_count
        train_acc = epoch_correct / epoch_count
        wandb.log({
            "global_step": global_step,
            "epoch": epoch,
            "train/loss_epoch": train_loss,
            "train/acc_epoch": train_acc,
            "time/epoch_minutes": (time.perf_counter() - t0) / 60.0
        })
        if epoch % cfg.checkpoint_epoch_freq == 0:
            chkpt_path = f"{models_dir}/epoch_{epoch}.pt"
            data.save_checkpoint(model, opt, global_step, chkpt_path)

        # Validation
        model.eval()
        val_correct = val_count = 0
        val_loss = 0.0
        with torch.no_grad():
            # If max_steps_per_epoch is set, we will shuffle the data from the val dataset.
            for xb, yb in epoch_iterator(val_data, shuffle=cfg.max_steps_per_epoch == 0):
                pred = model(xb).argmax(-1)
                loss = nn.loss.cross_entropy(logits, yb)
                val_loss += loss.item() * yb.size(0)
                val_correct += (pred == yb).sum().item(); val_count += yb.numel()
        val_loss /= val_count
        wandb.log({
            "global_step": global_step,
            "val/acc_epoch": val_correct / val_count,
            "val/loss_epoch": val_loss
        })

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
