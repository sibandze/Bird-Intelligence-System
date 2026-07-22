# src/training/callbacks.py

import json
import csv
from pathlib import Path
from typing import Dict, Any, Optional
import torch

try:
    import wandb
except ImportError:
    wandb = None


class Callback:
    """Base class for training callbacks."""
    def on_train_begin(self, trainer: Any): pass
    def on_train_end(self, trainer: Any): pass
    def on_epoch_begin(self, trainer: Any, epoch: int): pass
    def on_epoch_end(self, trainer: Any, epoch: int, logs: Dict[str, Any]): pass
    def on_batch_end(self, trainer: Any, batch: int, logs: Dict[str, Any]): pass


class CallbackRunner:
    """Executes a list of callbacks in sequence."""
    def __init__(self, callbacks: list[Callback]):
        self.callbacks = callbacks or []

    def on_train_begin(self, trainer):
        for cb in self.callbacks: cb.on_train_begin(trainer)

    def on_train_end(self, trainer):
        for cb in self.callbacks: cb.on_train_end(trainer)

    def on_epoch_begin(self, trainer, epoch: int):
        for cb in self.callbacks: cb.on_epoch_begin(trainer, epoch)

    def on_epoch_end(self, trainer, epoch: int, logs: Dict[str, Any]):
        for cb in self.callbacks: cb.on_epoch_end(trainer, epoch, logs)


# =====================================================================
# 1. Early Stopping Callback
# =====================================================================
class EarlyStoppingCallback(Callback):
    def __init__(self, monitor: str = "val_acc", mode: str = "max", patience: int = 15):
        self.monitor = monitor
        self.mode = mode
        self.patience = patience
        self.best_score = float("-inf") if mode == "max" else float("inf")
        self.patience_counter = 0

    def on_epoch_end(self, trainer, epoch: int, logs: Dict[str, Any]):
        score = logs.get(self.monitor)
        if score is None:
            return

        improved = (score > self.best_score) if self.mode == "max" else (score < self.best_score)
        
        if improved:
            self.best_score = score
            self.patience_counter = 0
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.patience:
                print(f"\n ⏹ Early stopping triggered! No improvement in '{self.monitor}' for {self.patience} epochs.")
                trainer.stop_training = True


# =====================================================================
# 2. Checkpoint Callback
# =====================================================================
class CheckpointCallback(Callback):
    def __init__(self, run_dir: Path, monitor: str = "val_acc", mode: str = "max"):
        self.run_dir = Path(run_dir)
        self.monitor = monitor
        self.mode = mode
        self.best_score = float("-inf") if mode == "max" else float("inf")

    def on_epoch_end(self, trainer, epoch: int, logs: Dict[str, Any]):
        current_score = logs.get(self.monitor, 0.0)
        
        checkpoint = {
            "epoch": epoch + 1,
            "val_acc": logs.get("val_acc", 0.0),
            "val_loss": logs.get("val_loss", float("inf")),
            "model_state_dict": trainer.model.state_dict(),
            "optimizer_state_dict": trainer.optimizer.state_dict(),
            "scheduler_state_dict": trainer.scheduler.state_dict() if trainer.scheduler else None,
            "precision_state_dict": trainer.precision.state_dict(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "git_commit": getattr(trainer, "git_hash", None),
        }

        # 1. Always save latest state for seamless resuming
        torch.save(checkpoint, self.run_dir / "checkpoint_last.pth")
        torch.save(trainer.model.state_dict(), self.run_dir / "last_model.pth")

        # 2. Save best checkpoint based on monitored metric
        improved = (current_score > self.best_score) if self.mode == "max" else (current_score < self.best_score)
        if improved:
            self.best_score = current_score
            trainer.best_val_acc = logs.get("val_acc", 0.0)
            trainer.best_epoch = epoch + 1
            
            torch.save(checkpoint, self.run_dir / "checkpoint_best.pth")
            torch.save(trainer.model.state_dict(), self.run_dir / "best_model.pth")
            print(f"    ✓ Saved new best model ({self.monitor}: {current_score:.4f})")


# =====================================================================
# 3. JSON Logger Callback
# =====================================================================
class JSONLoggerCallback(Callback):
    def __init__(self, run_dir: Path):
        self.json_path = Path(run_dir) / "training_metrics.json"
        self.history = []
        if self.json_path.exists():
            with open(self.json_path, "r") as f:
                self.history = json.load(f)

    def on_epoch_end(self, trainer, epoch: int, logs: Dict[str, Any]):
        self.history.append(logs)
        with open(self.json_path, "w") as f:
            json.dump(self.history, f, indent=2)


# =====================================================================
# 4. CSV Logger Callback
# =====================================================================
class CSVLoggerCallback(Callback):
    def __init__(self, run_dir: Path):
        self.csv_path = Path(run_dir) / "training_log.csv"

    def on_epoch_end(self, trainer, epoch: int, logs: Dict[str, Any]):
        file_exists = self.csv_path.exists()
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=logs.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(logs)


# =====================================================================
# 5. WandB Logger Callback
# =====================================================================
class WandBLoggerCallback(Callback):
    def __init__(self, config: Dict[str, Any], run_dir: Path):
        self.config = config
        self.run_dir = Path(run_dir)
        self.enabled = config.get("logging", {}).get("use_wandb", False) and wandb is not None

    def on_train_begin(self, trainer):
        if not self.enabled: return
        log_cfg = self.config.get("logging", {})
        wandb.init(
            project=log_cfg.get("wandb_project", "bird-song-classifier"),
            name=log_cfg.get("wandb_run_name", self.run_dir.name),
            config=self.config,
            dir=str(self.run_dir),
            resume="allow",
            id=log_cfg.get("wandb_run_id", self.run_dir.name),
        )

    def on_epoch_end(self, trainer, epoch: int, logs: Dict[str, Any]):
        if not self.enabled: return
        wandb.log(logs, step=epoch)

    def on_train_end(self, trainer):
        if not self.enabled: return
        wandb.summary["best_val_acc"] = trainer.best_val_acc
        wandb.summary["best_epoch"] = trainer.best_epoch
        
        for file in ["confusion_matrix.png", "per_class_metrics.png", "evaluation_metrics.json"]:
            path = self.run_dir / file
            if path.exists():
                wandb.save(str(path), base_path=str(self.run_dir))
        wandb.finish()
