from pathlib import Path

import torch
from safetensors.torch import save_file


def export_safetensors(model, output_dir: str) -> str:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    state_dict = model.state_dict()
    state_dict = {k: v.contiguous() for k, v in state_dict.items()}

    path = output_dir / "model.safetensors"
    save_file(state_dict, str(path))

    config_path = output_dir / "config.json"
    import json
    config_path.write_text(json.dumps(model.config.to_dict(), indent=2))

    return str(path)


def export_pytorch(model, output_dir: str) -> str:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    path = output_dir / "model.pt"
    torch.save(model.state_dict(), str(path))
    return str(path)
