import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from model.config import ModelConfig
from model.transformer import HybridModel
from training.data import PreTokenizedDataset
from training.trainer import Trainer
from torch.utils.data import DataLoader


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/train_300m.yaml"
    with open(config_path) as f:
        train_config = yaml.safe_load(f)

    model_config = ModelConfig.load(train_config["model"]["config_path"])
    model = HybridModel(model_config)

    params = model.count_parameters()
    print(f"Model: {params['total']:,} total params, {params['trainable']:,} trainable")

    data_cfg = train_config["data"]
    if Path(data_cfg.get("tokenizer_path", "")).exists():
        print(f"Tokenizer found at {data_cfg['tokenizer_path']}")

    train_cfg = train_config["training"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    dataloader = None
    data_path = Path("data") / "train_tokens.pt"
    if data_path.exists():
        dataset = PreTokenizedDataset(str(data_path), train_cfg["seq_len"])
        dataloader = DataLoader(
            dataset,
            batch_size=train_cfg["batch_size"],
            shuffle=True,
            num_workers=2,
            pin_memory=True,
        )
        print(f"Dataset loaded: {len(dataset)} samples")
    else:
        print(f"No pre-tokenized data found at {data_path}")
        print("Run scripts/download_data.py first, or train on dummy data for testing")

        dummy_tokens = torch.randint(0, model_config.vocab_size, (10000,))
        torch.save(dummy_tokens, "data/dummy_tokens.pt")
        dataset = PreTokenizedDataset("data/dummy_tokens.pt", train_cfg["seq_len"])
        dataloader = DataLoader(dataset, batch_size=train_cfg["batch_size"], shuffle=True)
        print(f"Using dummy dataset: {len(dataset)} samples")

    trainer = Trainer(
        model=model,
        train_dataloader=dataloader,
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
        warmup_steps=train_cfg["warmup_steps"],
        stable_steps=train_cfg["stable_steps"],
        decay_steps=train_cfg["decay_steps"],
        grad_accum_steps=train_cfg["grad_accum_steps"],
        max_grad_norm=train_cfg["max_grad_norm"],
        checkpoint_dir=train_cfg["checkpoint_dir"],
        log_interval=train_cfg["log_interval"],
        save_interval=train_cfg["save_interval"],
        device=device,
        use_amp=train_cfg["use_amp"],
        optimizer_name=train_cfg["optimizer"],
    )

    print(f"Starting training for {train_cfg['max_steps']} steps...")
    history = trainer.train(max_steps=train_cfg["max_steps"])

    print(f"Training complete. Final loss: {history['train_loss'][-1]:.4f}")


if __name__ == "__main__":
    main()
