import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from model.config import ModelConfig
from model.transformer import HybridModel
from inference.generate import generate
from inference.export import export_safetensors


def make_test_model():
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
    return HybridModel(config)


def test_generate():
    model = make_test_model()
    model.eval()

    input_ids = torch.randint(0, 1000, (1, 10))
    output_ids = generate(model, input_ids, max_new_tokens=20, temperature=1.0)

    assert output_ids.shape[0] == 1
    assert output_ids.shape[1] > 10
    assert output_ids.shape[1] <= 30
    print(f"Generated {output_ids.shape[1] - 10} tokens")


def test_generate_deterministic():
    model = make_test_model()
    model.eval()

    input_ids = torch.randint(0, 1000, (1, 10))
    torch.manual_seed(42)
    out1 = generate(model, input_ids, max_new_tokens=10, temperature=0.0)
    torch.manual_seed(42)
    out2 = generate(model, input_ids, max_new_tokens=10, temperature=0.0)

    assert torch.equal(out1, out2), "Deterministic generation should produce same output"


def test_export_safetensors(tmp_path):
    model = make_test_model()
    output_dir = str(tmp_path / "export")
    path = export_safetensors(model, output_dir)

    assert Path(path).exists()
    assert (Path(output_dir) / "config.json").exists()
    print(f"Exported to {path}")


if __name__ == "__main__":
    import tempfile
    test_generate()
    test_generate_deterministic()
    with tempfile.TemporaryDirectory() as tmp:
        from pathlib import Path
        test_export_safetensors(Path(tmp))
    print("All inference tests passed!")
