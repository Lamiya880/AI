import torch
import torch.nn as nn


def get_optimizer(
    model: nn.Module,
    name: str = "adamw8bit",
    lr: float = 3e-4,
    weight_decay: float = 0.1,
    betas: tuple[float, float] = (0.9, 0.95),
) -> torch.optim.Optimizer:
    decay_params = []
    no_decay_params = []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim >= 2 and "embedding" not in n and "A_log" not in n:
            decay_params.append(p)
        else:
            no_decay_params.append(p)

    param_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    if name == "adamw8bit":
        import bitsandbytes as bnb
        return bnb.optim.AdamW8bit(param_groups, lr=lr, betas=betas)
    elif name == "adamw":
        return torch.optim.AdamW(param_groups, lr=lr, betas=betas)
    elif name == "lion":
        try:
            from lion_pytorch import Lion
            return Lion(param_groups, lr=lr, betas=betas)
        except ImportError:
            return torch.optim.AdamW(param_groups, lr=lr, betas=betas)
    else:
        return torch.optim.AdamW(param_groups, lr=lr, betas=betas)
