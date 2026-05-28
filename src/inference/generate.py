import torch
import torch.nn.functional as F


@torch.no_grad()
def generate(
    model,
    input_ids: torch.Tensor,
    max_new_tokens: int = 256,
    temperature: float = 0.8,
    top_p: float = 0.9,
    top_k: int = 50,
    eos_token_id: int = 2,
) -> torch.Tensor:
    model.eval()
    generated = input_ids

    for _ in range(max_new_tokens):
        if generated.shape[1] > model.config.max_seq_len:
            input_slice = generated[:, -model.config.max_seq_len :]
        else:
            input_slice = generated

        outputs = model(input_slice)
        logits = outputs["logits"][:, -1, :]

        logits = logits / max(temperature, 1e-8)

        if top_k > 0:
            top_k_vals, _ = torch.topk(logits, min(top_k, logits.shape[-1]))
            logits[logits < top_k_vals[:, -1:]] = float("-inf")

        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
            sorted_logits[mask] = float("-inf")
            logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        generated = torch.cat([generated, next_token], dim=1)

        if next_token.item() == eos_token_id:
            break

    return generated


@torch.no_grad()
def generate_stream(
    model,
    input_ids: torch.Tensor,
    max_new_tokens: int = 256,
    temperature: float = 0.8,
    top_p: float = 0.9,
    top_k: int = 50,
    eos_token_id: int = 2,
):
    model.eval()
    generated = input_ids

    for _ in range(max_new_tokens):
        if generated.shape[1] > model.config.max_seq_len:
            input_slice = generated[:, -model.config.max_seq_len :]
        else:
            input_slice = generated

        outputs = model(input_slice)
        logits = outputs["logits"][:, -1, :]

        logits = logits / max(temperature, 1e-8)

        if top_k > 0:
            top_k_vals, _ = torch.topk(logits, min(top_k, logits.shape[-1]))
            logits[logits < top_k_vals[:, -1:]] = float("-inf")

        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
            sorted_logits[mask] = float("-inf")
            logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        generated = torch.cat([generated, next_token], dim=1)

        yield next_token

        if next_token.item() == eos_token_id:
            break
