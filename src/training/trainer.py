import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .checkpoint import save_checkpoint
from .optim import get_optimizer
from .scheduler import get_wsd_scheduler


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader | None = None,
        lr: float = 3e-4,
        weight_decay: float = 0.1,
        warmup_steps: int = 1000,
        stable_steps: int = 80000,
        decay_steps: int = 500,
        grad_accum_steps: int = 8,
        max_grad_norm: float = 1.0,
        checkpoint_dir: str = "checkpoints",
        log_interval: int = 10,
        save_interval: int = 1000,
        device: str = "cuda",
        use_amp: bool = True,
        optimizer_name: str = "adamw8bit",
    ):
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        self.grad_accum_steps = grad_accum_steps
        self.max_grad_norm = max_grad_norm
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.use_amp = use_amp

        self.optimizer = get_optimizer(model, optimizer_name, lr, weight_decay)
        self.scheduler = get_wsd_scheduler(self.optimizer, warmup_steps, stable_steps, decay_steps)

        # GradScaler only needed for fp16, not bf16
        self.use_scaler = use_amp  # will be overridden to False for bf16
        self.scaler = torch.amp.GradScaler("cuda", enabled=False)
        self.amp_dtype = torch.bfloat16
        self.global_step = 0
        self.best_loss = float("inf")

        # Enable gradient checkpointing to save memory
        if hasattr(model, "use_gradient_checkpointing"):
            model.use_gradient_checkpointing = True
            print("Gradient checkpointing enabled")

    def train(self, max_steps: int | None = None) -> dict[str, list[float]]:
        self.model.train()
        self.optimizer.zero_grad()

        history = {"train_loss": [], "lr": []}
        running_loss = 0.0
        step_start = time.time()

        epoch = 0
        while True:
            epoch += 1
            for batch in self.train_dataloader:
                if max_steps and self.global_step >= max_steps:
                    return history

                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)

                with torch.amp.autocast("cuda", dtype=self.amp_dtype, enabled=self.use_amp):
                    outputs = self.model(input_ids, labels=labels)
                    loss = outputs["loss"] / self.grad_accum_steps

                loss.backward()
                running_loss += loss.item()

                if (self.global_step + 1) % self.grad_accum_steps == 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.scheduler.step()
                    if self.device == "cuda":
                        torch.cuda.empty_cache()
                    if self.device == "cuda":
                        torch.cuda.empty_cache()

                if (self.global_step + 1) % self.log_interval == 0:
                    avg_loss = running_loss / self.log_interval
                    lr = self.scheduler.get_last_lr()[0]
                    elapsed = time.time() - step_start
                    tokens_per_sec = (
                        self.train_dataloader.batch_size
                        * input_ids.shape[1]
                        * self.log_interval
                        / elapsed
                    )

                    history["train_loss"].append(avg_loss)
                    history["lr"].append(lr)

                    print(
                        f"Step {self.global_step + 1} | "
                        f"Loss: {avg_loss:.4f} | "
                        f"LR: {lr:.2e} | "
                        f"Tok/s: {tokens_per_sec:.0f} | "
                        f"Time: {elapsed:.1f}s"
                    )

                    running_loss = 0.0
                    step_start = time.time()

                if (self.global_step + 1) % self.save_interval == 0:
                    avg_loss = running_loss / max(1, (self.global_step + 1) % self.save_interval)
                    self._save(avg_loss)

                self.global_step += 1

        return history

    def _save(self, loss: float) -> None:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = self.checkpoint_dir / f"checkpoint_{self.global_step}.pt"
        save_checkpoint(
            self.model, self.optimizer, self.scheduler,
            self.global_step, loss, str(path),
        )
        if loss < self.best_loss:
            self.best_loss = loss
            best_path = self.checkpoint_dir / "best_model.pt"
            save_checkpoint(
                self.model, self.optimizer, self.scheduler,
                self.global_step, loss, str(best_path),
            )
        print(f"Checkpoint saved: {path}")

    def evaluate(self) -> float:
        if not self.val_dataloader:
            return 0.0

        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for batch in self.val_dataloader:
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)

                with torch.amp.autocast("cuda", dtype=self.amp_dtype, enabled=self.use_amp):
                    outputs = self.model(input_ids, labels=labels)

                total_loss += outputs["loss"].item()
                n_batches += 1

        self.model.train()
        return total_loss / max(1, n_batches)
