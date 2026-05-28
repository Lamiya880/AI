import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from model.config import ModelConfig
from model.transformer import HybridModel


def test_model_creation():
    config = ModelConfig(
        d_model=128,
        n_layers=4,
        vocab_size=1000,
        max_seq_len=64,
        attention_layer_indices=[2],
        d_state=16,
        n_heads=4,
        n_kv_heads=2,
        head_dim=32,
        num_experts=4,
        num_shared_experts=1,
        moe_top_k=2,
        expert_dim=256,
    )
    model = HybridModel(config)
    params = model.count_parameters()
    print(f"Model parameters: {params['total']:,} total, {params['trainable']:,} trainable")
    assert params["total"] > 0
    assert params["trainable"] > 0


def test_model_forward():
    config = ModelConfig(
        d_model=128,
        n_layers=4,
        vocab_size=1000,
        max_seq_len=64,
        attention_layer_indices=[2],
        d_state=16,
        n_heads=4,
        n_kv_heads=2,
        head_dim=32,
        num_experts=4,
        num_shared_experts=1,
        moe_top_k=2,
        expert_dim=256,
    )
    model = HybridModel(config)
    model.eval()

    input_ids = torch.randint(0, 1000, (2, 32))
    with torch.no_grad():
        outputs = model(input_ids)

    assert outputs["logits"].shape == (2, 32, 1000)
    assert outputs["loss"] is None


def test_model_forward_with_labels():
    config = ModelConfig(
        d_model=128,
        n_layers=4,
        vocab_size=1000,
        max_seq_len=64,
        attention_layer_indices=[2],
        d_state=16,
        n_heads=4,
        n_kv_heads=2,
        head_dim=32,
        num_experts=4,
        num_shared_experts=1,
        moe_top_k=2,
        expert_dim=256,
    )
    model = HybridModel(config)

    input_ids = torch.randint(0, 1000, (2, 32))
    labels = torch.randint(0, 1000, (2, 32))
    outputs = model(input_ids, labels=labels)

    assert outputs["loss"] is not None
    assert outputs["loss"].requires_grad
    assert torch.isfinite(outputs["loss"]), f"Loss is not finite: {outputs['loss'].item()}"


def test_model_backward():
    config = ModelConfig(
        d_model=128,
        n_layers=4,
        vocab_size=1000,
        max_seq_len=64,
        attention_layer_indices=[2],
        d_state=16,
        n_heads=4,
        n_kv_heads=2,
        head_dim=32,
        num_experts=4,
        num_shared_experts=1,
        moe_top_k=2,
        expert_dim=256,
    )
    model = HybridModel(config)

    input_ids = torch.randint(0, 1000, (2, 32))
    labels = torch.randint(0, 1000, (2, 32))
    outputs = model(input_ids, labels=labels)
    outputs["loss"].backward()

    grads = {n: p.grad for n, p in model.named_parameters() if p.grad is not None}
    assert len(grads) > 0
    print(f"Gradients computed for {len(grads)} parameters")


def test_config_serialization():
    config = ModelConfig(d_model=256, n_layers=6, vocab_size=16000)
    d = config.to_dict()
    restored = ModelConfig.from_dict(d)
    assert restored.d_model == 256
    assert restored.n_layers == 6
    assert restored.vocab_size == 16000


if __name__ == "__main__":
    test_model_creation()
    test_model_forward()
    test_model_forward_with_labels()
    test_model_backward()
    test_config_serialization()
    print("All model tests passed!")
