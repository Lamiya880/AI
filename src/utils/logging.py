import time
from pathlib import Path


class TrainingLogger:
    def __init__(self, log_dir: str = "logs", use_wandb: bool = False, project: str = "advanced-llm"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.use_wandb = use_wandb

        if use_wandb:
            try:
                import wandb
                wandb.init(project=project)
            except ImportError:
                print("wandb not installed, falling back to file logging")
                self.use_wandb = False

    def log(self, metrics: dict, step: int) -> None:
        if self.use_wandb:
            import wandb
            wandb.log(metrics, step=step)

        line = f"Step {step} | " + " | ".join(f"{k}: {v}" for k, v in metrics.items())
        log_file = self.log_dir / "training.log"
        with open(log_file, "a") as f:
            f.write(line + "\n")

    def log_config(self, config: dict) -> None:
        if self.use_wandb:
            import wandb
            wandb.config.update(config)

        config_file = self.log_dir / "config.yaml"
        import yaml
        config_file.write_text(yaml.dump(config, default_flow_style=False))
