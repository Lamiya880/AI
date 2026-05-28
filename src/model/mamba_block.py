import torch
import torch.nn as nn
import torch.nn.functional as F

from .norm import RMSNorm

# Try to import CUDA-accelerated Mamba
try:
    from mamba_ssm import Mamba as CudaMamba
    HAS_CUDA_MAMBA = True
except ImportError:
    HAS_CUDA_MAMBA = False


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

        # Chunked parallel scan: process in chunks, propagate state between chunks
        chunk_size = 64
        n_chunks = (seqlen + chunk_size - 1) // chunk_size
        h = torch.zeros(batch, d_inner, d_state, device=u.device, dtype=u.dtype)
        outputs = []

        for c in range(n_chunks):
            start = c * chunk_size
            end = min(start + chunk_size, seqlen)
            chunk_len = end - start

            # Process chunk - vectorized within chunk
            chunk_dA = deltaA[:, start:end]  # [B, chunk, D, N]
            chunk_dB = deltaB[:, start:end]  # [B, chunk, D, N]
            chunk_u = u[:, start:end]        # [B, chunk, D]
            chunk_C = C[:, start:end]        # [B, chunk, N]

            chunk_out = []
            for t in range(chunk_len):
                h = chunk_dA[:, t] * h + chunk_dB[:, t] * chunk_u[:, t, :, None]
                y_t = (h * chunk_C[:, t, None, :]).sum(dim=-1)
                chunk_out.append(y_t)

            outputs.extend(chunk_out)

        return torch.stack(outputs, dim=1)


class MambaBlock(nn.Module):
    def __init__(self, d_model: int, d_state: int = 64, d_conv: int = 4, expand: int = 2, norm_eps: float = 1e-6):
        super().__init__()
        self.norm = RMSNorm(d_model, eps=norm_eps)
        if HAS_CUDA_MAMBA:
            self.ssm = CudaMamba(d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            self.use_cuda = True
        else:
            self.ssm = SelectiveSSM(d_model, d_state, d_conv, expand)
            self.use_cuda = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ssm(self.norm(x))
