import torch
import torch.nn as nn
import torch.nn.functional as F

from .norm import RMSNorm


class SelectiveSSM(nn.Module):
    def __init__(self, d_model: int, d_state: int = 64, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_inner = d_model * expand
        self.d_state = d_state
        self.d_conv = d_conv
        self.dt_rank = d_state

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner, d_conv,
            padding=d_conv - 1, groups=self.d_inner,
        )

        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        A = torch.arange(1, d_state + 1).float().unsqueeze(0).expand(self.d_inner, -1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.dt_min = 0.001
        self.dt_max = 0.1

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seqlen, _ = x.shape
        xz = self.in_proj(x)
        x_ssm, z = xz.chunk(2, dim=-1)

        x_conv = self.conv1d(x_ssm.transpose(1, 2))[:, :, :seqlen].transpose(1, 2)
        x_conv = F.silu(x_conv)

        x_proj_out = self.x_proj(x_conv)
        dt_param = x_proj_out[..., : self.dt_rank]
        B_param = x_proj_out[..., self.dt_rank : self.dt_rank + self.d_state]
        C_param = x_proj_out[..., self.dt_rank + self.d_state :]
        delta = F.softplus(self.dt_proj(dt_param))
        delta = torch.clamp(delta, self.dt_min, self.dt_max)

        A = -torch.exp(self.A_log.clamp(max=4.0))
        deltaA = torch.exp((delta.unsqueeze(-1) * A).clamp(-5.0, 5.0))
        deltaB = delta.unsqueeze(-1) * B_param.unsqueeze(2)

        y = self._selective_scan(x_conv, deltaA, deltaB, C_param)
        y = y + self.D.unsqueeze(0).unsqueeze(0) * x_conv
        y = y * F.silu(z)
        return self.out_proj(y)

    def _selective_scan(
        self,
        u: torch.Tensor,
        deltaA: torch.Tensor,
        deltaB: torch.Tensor,
        C: torch.Tensor,
    ) -> torch.Tensor:
        batch, seqlen, d_inner, d_state = deltaA.shape
        h = torch.zeros(batch, d_inner, d_state, device=u.device, dtype=u.dtype)
        outputs = []
        for t in range(seqlen):
            h = deltaA[:, t] * h + deltaB[:, t] * u[:, t, :, None]
            y_t = (h * C[:, t, None, :]).sum(dim=-1)
            outputs.append(y_t)
        return torch.stack(outputs, dim=1)


class MambaBlock(nn.Module):
    def __init__(self, d_model: int, d_state: int = 64, d_conv: int = 4, expand: int = 2, norm_eps: float = 1e-6):
        super().__init__()
        self.norm = RMSNorm(d_model, eps=norm_eps)
        self.ssm = SelectiveSSM(d_model, d_state, d_conv, expand)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ssm(self.norm(x))
