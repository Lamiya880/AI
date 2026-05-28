import torch


class KVCache:
    def __init__(
        self,
        max_len: int,
        n_layers: int,
        n_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        self.max_len = max_len
        self.cache_k = torch.zeros(
            n_layers, 1, n_kv_heads, max_len, head_dim, dtype=dtype, device=device
        )
        self.cache_v = torch.zeros_like(self.cache_k)
        self.cur_len = 0

    def update(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor) -> None:
        seq_len = k.shape[2]
        self.cache_k[layer_idx, :, :, self.cur_len : self.cur_len + seq_len] = k
        self.cache_v[layer_idx, :, :, self.cur_len : self.cur_len + seq_len] = v

    def get(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self.cache_k[layer_idx, :, :, : self.cur_len],
            self.cache_v[layer_idx, :, :, : self.cur_len],
        )

    def advance(self, n: int) -> None:
        self.cur_len += n


class SSMStateCache:
    def __init__(
        self,
        n_layers: int,
        batch_size: int,
        d_inner: int,
        d_state: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        self.states = torch.zeros(
            n_layers, batch_size, d_inner, d_state, dtype=dtype, device=device
        )

    def get(self, layer_idx: int) -> torch.Tensor:
        return self.states[layer_idx]

    def update(self, layer_idx: int, state: torch.Tensor) -> None:
        self.states[layer_idx] = state
