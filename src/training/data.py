import torch
from torch.utils.data import Dataset, IterableDataset


class PreTokenizedDataset(Dataset):
    def __init__(self, data_path: str, seq_len: int = 4096):
        self.tokens = torch.load(data_path, weights_only=True)
        self.seq_len = seq_len

    def __len__(self) -> int:
        return max(0, len(self.tokens) - self.seq_len - 1)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        chunk = self.tokens[idx : idx + self.seq_len + 1]
        return {
            "input_ids": chunk[:-1],
            "labels": chunk[1:],
        }


class StreamingTokenDataset(IterableDataset):
    def __init__(self, token_iterator, seq_len: int = 4096, buffer_size: int = 100000):
        self.token_iterator = token_iterator
        self.seq_len = seq_len
        self.buffer_size = buffer_size

    def __iter__(self):
        buffer = []
        for tokens in self.token_iterator:
            buffer.extend(tokens)
            while len(buffer) >= self.seq_len + 1:
                chunk = torch.tensor(buffer[: self.seq_len + 1], dtype=torch.long)
                buffer = buffer[self.seq_len :]
                yield {
                    "input_ids": chunk[:-1],
                    "labels": chunk[1:],
                }


def pack_sequences(
    token_lists: list[list[int]],
    seq_len: int,
) -> torch.Tensor:
    all_tokens = []
    for tokens in token_lists:
        all_tokens.extend(tokens)

    n_full = len(all_tokens) // (seq_len + 1)
    trimmed = all_tokens[: n_full * (seq_len + 1)]
    return torch.tensor(trimmed, dtype=torch.long)
