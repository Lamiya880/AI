import math

import torch


def precompute_rope_freqs(
    dim: int,
    max_len: int,
    base: float = 10000.0,
    scaling_factor: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if scaling_factor is not None:
        base = base * (scaling_factor ** (dim / (dim - 2)))

    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_len).float()
    freqs = torch.outer(t, inv_freq)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    d = x.shape[-1]
    x1 = x[..., : d // 2]
    x2 = x[..., d // 2 :]

    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]

    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


def yarn_rope(
    dim: int,
    base: float = 10000.0,
    scale: float = 8.0,
    beta_fast: float = 32.0,
    beta_slow: float = 1.0,
) -> tuple[torch.Tensor, float]:
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    wavelength = 2 * math.pi / inv_freq

    lo = wavelength / (2 * math.pi * beta_fast)
    hi = wavelength / (2 * math.pi * beta_slow)
    ramp = torch.clamp((1.0 / scale - lo) / (hi - lo), 0, 1)

    new_base = base * (scale ** (dim / (dim - 2)))
    scaled_inv_freq = 1.0 / (new_base ** (torch.arange(0, dim, 2).float() / dim))
    inv_freq = inv_freq * (1 - ramp) + scaled_inv_freq * ramp

    attn_scale = 0.1 * math.log(scale) + 1.0
    return inv_freq, attn_scale
