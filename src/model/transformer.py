import torch
import torch.nn as nn

from .config import ModelConfig
from .embedding import TokenEmbedding
from .attention import AttentionBlock
from .mamba_block import MambaBlock
from .moe import MoELayer
from .norm import RMSNorm
from .rope import precompute_rope_freqs


class HybridModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.embedding = TokenEmbedding(config.vocab_size, config.d_model)

        self.layers = nn.ModuleList()
        for i in range(config.n_layers):
            if config.is_attention_layer(i):
                self.layers.append(
                    AttentionBlock(
                        config.d_model, config.n_heads, config.n_kv_heads,
                        config.head_dim, config.norm_eps,
                    )
                )
            else:
                self.layers.append(
                    MambaBlock(
                        config.d_model, config.d_state, config.d_conv,
                        config.expand, config.norm_eps,
                    )
                )

        self.moe_layers = nn.ModuleList([
            MoELayer(
                config.d_model, config.expert_dim, config.num_experts,
                config.num_shared_experts, config.moe_top_k,
            )
            for _ in range(config.n_layers)
        ])

        self.norm = RMSNorm(config.d_model, eps=config.norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self._precompute_rope()
        self._init_weights()

    def _precompute_rope(self) -> None:
        cos, sin = precompute_rope_freqs(
            self.config.head_dim, self.config.max_seq_len, self.config.rope_base
        )
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def _init_weights(self) -> None:
        std = self.config.d_model ** -0.5
        for name, p in self.named_parameters():
            if p.dim() <= 1 or "A_log" in name:
                continue
            if "embedding" in name:
                nn.init.normal_(p, mean=0.0, std=std)
            elif "out_proj" in name or "wo" in name or "w2" in name:
                nn.init.normal_(p, mean=0.0, std=std / (2 * self.config.n_layers) ** 0.5)
            else:
                nn.init.normal_(p, mean=0.0, std=std)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        x = self.embedding(input_ids)

        cos = self.rope_cos[: x.shape[1]]
        sin = self.rope_sin[: x.shape[1]]

        for i, (layer, moe) in enumerate(zip(self.layers, self.moe_layers)):
            if self.config.is_attention_layer(i):
                x = layer(x, cos, sin)
            else:
                x = layer(x)
            x = x + moe(x)

        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                labels.view(-1),
                ignore_index=-100,
            )

        return {"logits": logits, "loss": loss}

    def count_parameters(self) -> dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}
