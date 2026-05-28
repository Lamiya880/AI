import json
from pathlib import Path

import torch


def quantize_gptq(
    model_path: str,
    output_path: str,
    bits: int = 4,
    group_size: int = 128,
    calibration_dataset=None,
) -> str:
    from gptqmodel import GPTQModel, QuantizeConfig

    quantize_config = QuantizeConfig(bits=bits, group_size=group_size)
    model = GPTQModel.load(model_path, quantize_config)

    if calibration_dataset is None:
        raise ValueError("calibration_dataset is required for GPTQ quantization")

    model.quantize(calibration_dataset)
    model.save(output_path)
    return output_path


def export_gguf(model_path: str, output_path: str, quant_type: str = "Q4_K_M") -> str:
    import subprocess
    import sys

    script = Path(__file__).parent.parent.parent / "scripts" / "convert_to_gguf.py"
    cmd = [
        sys.executable, str(script),
        str(model_path),
        "--outfile", str(output_path),
        "--outtype", quant_type,
    ]
    subprocess.run(cmd, check=True)
    return output_path


def quantize_int8_dynamic(model):
    from torchao.quantization import int8_dynamic_activation_int8_weight, quantize_
    quantize_(model, int8_dynamic_activation_int8_weight())
    return model


def quantize_int4_weight_only(model):
    from torchao.quantization import int4_weight_only, quantize_
    quantize_(model, int4_weight_only())
    return model
