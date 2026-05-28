from pathlib import Path

import torch
import torch.nn as nn


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    step: int,
    loss: float,
    path: str,
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else None,
        "step": step,
        "loss": loss,
        "rng_state": torch.random.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
    }, path)


def load_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler=None,
    path: str = "checkpoint.pt",
) -> dict:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    if optimizer and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler and ckpt.get("scheduler"):
        scheduler.load_state_dict(ckpt["scheduler"])
    if "rng_state" in ckpt:
        torch.random.set_rng_state(ckpt["rng_state"])
    if ckpt.get("cuda_rng_state") and torch.cuda.is_available():
        torch.cuda.set_rng_state(ckpt["cuda_rng_state"])
    return ckpt
