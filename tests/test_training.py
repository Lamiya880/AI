import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from model.config import ModelConfig
from model.transformer import HybridModel
from training.scheduler import get_wsd_scheduler, get_cosine_scheduler
from training.checkpoint import save_checkpoint, load_checkpoint


def make_dummy_model_and_data():
    config = ModelConfig(
        d_model=128,
        n_layers=4,
        vocab_size=1000,
        max_seq_len=32,
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

    n_samples = 20
    seq_len = 32
    input_ids = torch.randint(0, 1000, (n_samples, seq_len))
    labels = torch.randint(0, 1000, (n_samples, seq_len))
    dataset = TensorDataset(input_ids, labels)

    def collate(batch):
        ids, lbls = zip(*batch)
        return {"input_ids": torch.stack(ids), "labels": torch.stack(lbls)}

    dataloader = DataLoader(dataset, batch_size=4, collate_fn=collate)
    return model, dataloader


def test_training_step():
    model, dataloader = make_dummy_model_and_data()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    model.train()
    batch = next(iter(dataloader))
    outputs = model(batch["input_ids"], labels=batch["labels"])
    loss = outputs["loss"]

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    assert loss.item() > 0
    print(f"Training step loss: {loss.item():.4f}")


def test_loss_decreases():
    model, dataloader = make_dummy_model_and_data()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    model.train()
    losses = []
    for _ in range(5):
        batch = next(iter(dataloader))
        outputs = model(batch["input_ids"], labels=batch["labels"])
        loss = outputs["loss"]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    print(f"Losses: {[f'{l:.4f}' for l in losses]}")
    assert losses[-1] < losses[0], f"Loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"


def test_wsd_scheduler():
    model, _ = make_dummy_model_and_data()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = get_wsd_scheduler(optimizer, warmup_steps=5, stable_steps=10, decay_steps=5)

    lrs = []
    for _ in range(20):
        lrs.append(scheduler.get_last_lr()[0])
        scheduler.step()

    assert lrs[0] < lrs[4], "LR should increase during warmup"
    assert abs(lrs[6] - lrs[14]) < 1e-6, "LR should be stable"
    assert lrs[-1] < lrs[14], "LR should decrease during decay"
    print(f"WSD LR schedule: {[f'{lr:.6f}' for lr in lrs]}")


def test_cosine_scheduler():
    model, _ = make_dummy_model_and_data()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = get_cosine_scheduler(optimizer, warmup_steps=5, total_steps=20)

    lrs = []
    for _ in range(20):
        lrs.append(scheduler.get_last_lr()[0])
        scheduler.step()

    assert lrs[0] < lrs[4], "LR should increase during warmup"
    assert lrs[-1] < lrs[5], "LR should decrease after warmup"
    print(f"Cosine LR schedule: {[f'{lr:.6f}' for lr in lrs]}")


def test_checkpoint_save_load(tmp_path):
    model, _ = make_dummy_model_and_data()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = get_wsd_scheduler(optimizer, 5, 10, 5)

    path = str(tmp_path / "test_ckpt.pt")
    save_checkpoint(model, optimizer, scheduler, step=42, loss=2.5, path=path)

    model2, _ = make_dummy_model_and_data()
    optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
    scheduler2 = get_wsd_scheduler(optimizer2, 5, 10, 5)

    ckpt = load_checkpoint(model2, optimizer2, scheduler2, path)
    assert ckpt["step"] == 42
    assert ckpt["loss"] == 2.5

    for p1, p2 in zip(model.parameters(), model2.parameters()):
        assert torch.allclose(p1, p2), "Model parameters should match after load"


if __name__ == "__main__":
    import tempfile
    test_training_step()
    test_loss_decreases()
    test_wsd_scheduler()
    test_cosine_scheduler()
    with tempfile.TemporaryDirectory() as tmp:
        from pathlib import Path
        test_checkpoint_save_load(Path(tmp))
    print("All training tests passed!")
