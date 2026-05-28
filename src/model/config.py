from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml


@dataclass
class ModelConfig:
    # Core dimensions
    d_model: int = 768
    n_layers: int = 12
    vocab_size: int = 32000
    max_seq_len: int = 4096

    # Layer type distribution (indices of attention layers, rest are Mamba)
    attention_layer_indices: list[int] = field(default_factory=lambda: [7, 11])

    # Mamba-2 SSM parameters
    d_state: int = 64
    d_conv: int = 4
    expand: int = 2
    mamba_ngroups: int = 8

    # Attention parameters
    n_heads: int = 12
    n_kv_heads: int = 4
    head_dim: int = 64

    # MoE parameters
    num_experts: int = 8
    num_shared_experts: int = 1
    moe_top_k: int = 2
    expert_dim: int = 2048

    # Normalization
    norm_eps: float = 1e-6

    # RoPE
    rope_base: float = 10000.0
    rope_scaling: float | None = None

    # Architecture choices
    use_mamba: bool = True

    @property
    def d_inner(self) -> int:
        return self.d_model * self.expand

    @property
    def n_mamba_layers(self) -> int:
        return self.n_layers - len(self.attention_layer_indices)

    @property
    def n_attention_layers(self) -> int:
        return len(self.attention_layer_indices)

    def is_attention_layer(self, layer_idx: int) -> bool:
        return layer_idx in self.attention_layer_indices

    def to_dict(self) -> dict:
        return {
            k: (v if not isinstance(v, Path) else str(v))
            for k, v in self.__dict__.items()
        }

    @classmethod
    def from_dict(cls, d: dict) -> ModelConfig:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def save(self, path: str | Path) -> None:
        Path(path).write_text(yaml.dump(self.to_dict(), default_flow_style=False))

    @classmethod
    def load(cls, path: str | Path) -> ModelConfig:
        d = yaml.safe_load(Path(path).read_text())
        return cls.from_dict(d)
