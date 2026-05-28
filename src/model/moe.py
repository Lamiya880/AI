import torch
import torch.nn as nn
import torch.nn.functional as F

from .ffn import SwiGLU


class TopKRouter(nn.Module):
    def __init__(self, hidden_dim: int, num_experts: int, top_k: int = 2):
        super().__init__()
        self.top_k = top_k
        self.gate = nn.Linear(hidden_dim, num_experts, bias=False)
        self.register_buffer("expert_bias", torch.zeros(num_experts))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.gate(x)
        scores = F.softmax(logits, dim=-1)

        routing_scores = scores + self.expert_bias
        _, top_k_indices = torch.topk(routing_scores, self.top_k, dim=-1)

        top_k_scores = torch.gather(scores, -1, top_k_indices)
        top_k_scores = top_k_scores / top_k_scores.sum(dim=-1, keepdim=True)

        return top_k_scores, top_k_indices

    @torch.no_grad()
    def update_bias(self, expert_load_counts: torch.Tensor, target_load: float) -> None:
        load_diff = expert_load_counts - target_load
        self.expert_bias -= 0.001 * load_diff.sign()


class MoELayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        expert_dim: int = 2048,
        num_experts: int = 8,
        num_shared_experts: int = 1,
        top_k: int = 2,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = TopKRouter(d_model, num_experts, top_k)

        self.routed_experts = nn.ModuleList([
            SwiGLU(d_model, expert_dim) for _ in range(num_experts)
        ])
        self.shared_experts = nn.ModuleList([
            SwiGLU(d_model, expert_dim) for _ in range(num_shared_experts)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, hidden = x.shape
        scores, indices = self.router(x)

        shared_out = sum(expert(x) for expert in self.shared_experts)

        flat_x = x.reshape(-1, hidden)
        flat_scores = scores.reshape(-1, self.top_k)
        flat_indices = indices.reshape(-1, self.top_k)
        routed_out = torch.zeros_like(flat_x)

        for k in range(self.top_k):
            expert_idx = flat_indices[:, k]
            expert_wt = flat_scores[:, k].unsqueeze(-1)
            for e in range(self.num_experts):
                mask = expert_idx == e
                if mask.any():
                    routed_out[mask] += expert_wt[mask] * self.routed_experts[e](flat_x[mask])

        return routed_out.reshape(batch, seq_len, hidden) + shared_out
