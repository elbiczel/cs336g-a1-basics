import argparse
import wandb
import random
import time
from types import SimpleNamespace
from typing import Iterator

import numpy as np
import torch

from cs336_basics import data, utils
from cs336_tokenizer.bpe import TrieTokenizer
import cs336_nn as nn

def parse_params():
    parser = argparse.ArgumentParser(description="LLM Text generation configuration")
    # Basics
    parser.add_argument("--project", type=str, default="", help="Project name.")
    parser.add_argument("--trainer_run", type=str, default="", help="The run id to extract the model config and artefact from.")
    parser.add_argument("--models_base_path", type=str, default="", help="Base path to store models.")
    parser.add_argument("--device", type=str, default="", help="Device to use (cpu, cuda, etc).")
    # Tokenizer
    parser.add_argument("--tokenizer_vocab", type=str, default="", help="Tokenizer vocab file.")
    parser.add_argument("--tokenizer_merges", type=str, default="", help="Tokenizer merges file.")

    parser.add_argument("--prompt", type=str, default="", help="Prompt to feed the generation.")
    # Generation params
    parser.add_argument("--temp", type=float, default=1.0, help="Temperature for sampling.")
    parser.add_argument("--nucelus_top_p", type=float, default=1.0, help="Top-P probability for nucleus sampling.")
    parser.add_argument("--max_tokens", type=int, default=256, help="Max number of tokens to generate. Generation will finish earlier if end of text token is generated.")
    # Model params
    parser.add_argument("--compile", type=bool, default=True, help="Whether the model should be compiled before generation.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--resume", type=bool, default=False, help="If true, reas the resume checkpoint from the model, not the best one.")
    return parser.parse_args()

eof_token_text = b"<|endoftext|>"
eof_token_str = eof_token_text.decode("utf-8")

def top_p_indices(probs: torch.Tensor, p: float, dim: int = -1) -> torch.Tensor:
    "Returns a boolean mask (same shape as `probs`) marking the kept entries."
    if not (0.0 < p <= 1.0):
        raise ValueError("p must be in (0, 1].")
    dim = dim % probs.ndim
    sorted_probs, sorted_idx = torch.sort(probs, dim=dim, descending=True)
    cumsum = sorted_probs.cumsum(dim=dim)

    target = torch.full(
        probs.shape[:dim] + probs.shape[dim+1:],
        fill_value=p,
        dtype=probs.dtype,
        device=probs.device,
    ).unsqueeze(-1)
    cutoff = torch.searchsorted(cumsum,  target).squeeze(dim)
    
    arange = torch.arange(probs.size(dim), device=probs.device)
    cmp_shape = [1]*probs.ndim
    cmp_shape[dim] = -1
    keep_sorted = (arange.view(cmp_shape) <= cutoff.unsqueeze(dim))

    keep_mask = torch.zeros_like(probs, dtype=torch.bool)
    keep_mask.scatter_(dim, sorted_idx, keep_sorted)
    return keep_mask


@utils.stopwatch
def generate_tokens(cfg: SimpleNamespace, model: torch.nn.Module, tokens: torch.Tensor, tokenizer) -> list[int]:
    def _it(tokens) -> Iterator[int]:
        i = 0
        while i < cfg.max_tokens:
            i += 1
            logits = model(tokens)[cfg.next_token_i, :]
            if cfg.temp > 0.0:
                logits = logits / cfg.temp
            probs = nn.softmax(logits, dim=-1)
            if cfg.nucelus_top_p < 1.0:
                kept_mask = top_p_indices(probs, cfg.nucelus_top_p)
                probs[~kept_mask] = 0.0
                # Ensure the probabilities sum to 1.0
                probs = probs / probs.sum()
            if cfg.temp <= 0.0:
                ind = torch.argmax(probs)
            else:
                ind = torch.multinomial(probs, num_samples=1)
            next_token = ind.item()
            if next_token == cfg.eof_token:
                break
            yield next_token
            if cfg.next_token_i == cfg.context_length - 1:
                tokens = torch.roll(tokens, shifts=-1, dims=0)
                tokens[-1] = ind
            else:
                cfg.next_token_i += 1
                tokens[cfg.next_token_i] = ind
    return list(_it(tokens))

@torch.no_grad
def generate(cfg) -> str:
    # Prep tokenizer.
    tokenizer = TrieTokenizer.from_files(cfg.tokenizer_vocab, cfg.tokenizer_merges, special_tokens=[eof_token_text])
    eof_tokens = tokenizer.encode(eof_token_str)
    assert len(eof_tokens) == 1
    cfg.eof_token = eof_tokens[0]

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

    # Generate from model:
    prompt_tokens = tokenizer.encode(cfg.prompt)
    cfg.next_token_i = cfg.context_length - 1
    if len(prompt_tokens) > cfg.context_length:
        prompt_tokens = prompt_tokens[-cfg.context_length:]
    elif len(prompt_tokens) < cfg.context_length:
        cfg.next_token_i = len(prompt_tokens) - 1
        num_to_pad = cfg.context_length - len(prompt_tokens)
        prompt_tokens += [cfg.eof_token] * num_to_pad
    assert len(prompt_tokens) == cfg.context_length
    prompt_tokens = torch.tensor(prompt_tokens, dtype=torch.long, device=cfg.device)
    tokens = generate_tokens(cfg, model, prompt_tokens, tokenizer)

    return tokenizer.decode(tokens)

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
    out = generate(cfg)
    print("> ", params.prompt)
    print(out)
    

if __name__ == "__main__":
    main()
