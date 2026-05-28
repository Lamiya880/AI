import torch


def get_gpu_memory() -> dict[str, float]:
    if not torch.cuda.is_available():
        return {"total": 0, "used": 0, "free": 0}

    return {
        "total": torch.cuda.get_device_properties(0).total_mem / 1e9,
        "used": torch.cuda.memory_allocated() / 1e9,
        "reserved": torch.cuda.memory_reserved() / 1e9,
        "free": (torch.cuda.get_device_properties(0).total_mem - torch.cuda.memory_reserved()) / 1e9,
    }


def print_memory_summary() -> None:
    if not torch.cuda.is_available():
        print("CUDA not available")
        return

    mem = get_gpu_memory()
    print(f"GPU Memory: {mem['used']:.2f} / {mem['total']:.2f} GB used, {mem['free']:.2f} GB free")
