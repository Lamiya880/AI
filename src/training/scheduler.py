import math

from torch.optim.lr_scheduler import LambdaLR


def get_wsd_scheduler(
    optimizer,
    warmup_steps: int,
    stable_steps: int,
    decay_steps: int,
) -> LambdaLR:
    total = warmup_steps + stable_steps + decay_steps

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        elif step < warmup_steps + stable_steps:
            return 1.0
        else:
            progress = (step - warmup_steps - stable_steps) / max(1, decay_steps)
            return max(0.0, 1.0 - progress)

    return LambdaLR(optimizer, lr_lambda)


def get_cosine_scheduler(
    optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.1,
) -> LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)
