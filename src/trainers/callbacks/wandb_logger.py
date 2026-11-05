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
        log_cfg = trainer.config.get("logging", {})

        # Only resume wandb if we are resuming a checkpoint for THIS run_dir.
        # New run_dir -> fresh run, even if id collides.
        ckpt_exists = (Path(trainer.run_dir) / "checkpoint_last.pth").exists()
        # Also check attempt - trainer sets this after _resume_checkpoint,
        # but on_train_begin runs AFTER resume, so attempt > 0 means resumed
        is_resume = ckpt_exists or trainer.trainer_state.get("attempt", 0) > 0

        try:
            if wandb.run is not None:
                # Orphaned run from previous experiment that crashed
                # before on_train_end. Finish it, don't resume it.
                # This is what caused step 0 < 305.
                logger.warning("wandb.run already active, finishing orphaned run")
                try:
                    wandb.finish(exit_code=1)
                except Exception:
                    pass

            wandb.init(
                project=log_cfg.get("wandb_project", "bird-song-classifier"),
                name=log_cfg.get("wandb_run_name", trainer.run_dir.name),
                config=trainer.config,
                dir=str(trainer.run_dir),
                id=log_cfg.get("wandb_run_id", trainer.run_dir.name),
                # KEY: allow only when resuming same run_dir
                resume="allow" if is_resume else "never",
                reinit=True,
            )
            wandb.watch(trainer.model, log="gradients", log_freq=100)
            self._wandb_initialized = True
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
