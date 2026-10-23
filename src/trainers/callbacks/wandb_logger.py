# src/trainers/callbacks/wandb_logger.py
import logging
from pathlib import Path
from typing import Any, Dict

from .base import Callback, _EPOCH_META_KEYS

logger = logging.getLogger(__name__)

try:
    import wandb
except ImportError:
    wandb = None


class WandBLoggerCallback(Callback):
    def __init__(self, config: Dict[str, Any], run_dir: Path, max_failures: int = 3):
        self.config = config
        self.run_dir = Path(run_dir)
        self.enabled = (
            config.get("logging", {}).get("use_wandb", False) and wandb is not None
        )
        self._failure_count = 0
        self._max_failures = max_failures
        self._wandb_initialized = False

    def _disable(self, reason: str):
        if self.enabled:
            logger.warning(f"WandB logging disabled: {reason}")
            self.enabled = False

    def _handle_error(self, context: str, exc: Exception):
        self._failure_count += 1
        logger.warning(
            f"WandB error during {context}: {exc}. "
            f"Failure count: {self._failure_count}/{self._max_failures}"
        )
        if self._failure_count >= self._max_failures:
            self._disable(f"too many consecutive failures ({self._failure_count})")

    def on_train_begin(self, trainer):
        if not self.enabled:
            return
        log_cfg = self.config.get("logging", {})
        try:
            wandb.init(
                project=log_cfg.get("wandb_project", "bird-song-classifier"),
                name=log_cfg.get("wandb_run_name", self.run_dir.name),
                config=self.config,
                dir=str(self.run_dir),
                resume="allow",
                id=log_cfg.get("wandb_run_id", self.run_dir.name),
            )
            wandb.watch(trainer.model, log="gradients", log_freq=100)
            self._wandb_initialized = True
            self._failure_count = 0
        except Exception as e:
            self._handle_error("initialization", e)
            self._wandb_initialized = False

    def on_epoch_end(self, trainer, epoch: int, logs: Dict[str, Any]):
        if not self.enabled or not self._wandb_initialized:
            return
        # `wandb.log(..., step=epoch)` already carries the epoch on the
        # step axis; `attempt` / `run_id` are metadata, not metrics.
        strip = _EPOCH_META_KEYS - {"global_epoch"}
        metrics = {k: v for k, v in logs.items() if k not in strip}
        try:
            wandb.log(metrics, step=epoch)
            self._failure_count = 0
        except Exception as e:
            self._handle_error("log", e)

    def on_train_end(self, trainer):
        if not self.enabled:
            return
        try:
            if self._wandb_initialized:
                state = getattr(trainer, "trainer_state", {})
                monitor = getattr(trainer, "best_monitor", None)
                best_value = state.get("best_metric")

                if monitor == "val_acc":
                    wandb.summary["best_val_acc"] = best_value
                elif monitor == "val_loss":
                    wandb.summary["best_val_loss"] = best_value

                if "best_epoch" in state:
                    wandb.summary["best_epoch"] = state["best_epoch"]

                for file in [
                    "confusion_matrix.png",
                    "per_class_metrics.png",
                    "evaluation_metrics.json",
                ]:
                    path = self.run_dir / file
                    if path.exists():
                        wandb.save(str(path), base_path=str(self.run_dir))
            wandb.finish()
        except Exception as e:
            self._handle_error("finalization", e)
            try:
                wandb.finish(exit_code=1)
            except Exception:
                pass
