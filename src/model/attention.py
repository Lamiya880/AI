import torch
import torch.nn as nn
import torch.nn.functional as F

from .norm import RMSNorm
from .rope import apply_rope


class GroupedQueryAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int = 64,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_groups = n_heads // n_kv_heads
        self.head_dim = head_dim

        self.wq = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.wk = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        self.wv = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        self.wo = nn.Linear(n_heads * head_dim, d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor | None = None,
        sin: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, seqlen, _ = x.shape

        q = self.wq(x).view(batch, seqlen, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(batch, seqlen, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(batch, seqlen, self.n_kv_heads, self.head_dim).transpose(1, 2)

        if cos is not None and sin is not None:
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)

        k = k.repeat_interleave(self.n_groups, dim=1)
        v = v.repeat_interleave(self.n_groups, dim=1)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(batch, seqlen, -1)
        return self.wo(out)


class AttentionBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int = 64,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.norm = RMSNorm(d_model, eps=norm_eps)
        self.attn = GroupedQueryAttention(d_model, n_heads, n_kv_heads, head_dim, norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor | None = None,
        sin: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return x + self.attn(self.norm(x), cos, sin)
