# src/trainers/callbacks/wandb_logger.py
import logging
from pathlib import Path
from typing import Any, Dict

from .base import Callback

logger = logging.getLogger(__name__)

try:
    import wandb
except ImportError:
    wandb = None


class WandBLoggerCallback(Callback):
    def __init__(self, config: Dict[str, Any], run_dir: Path, max_failures: int = 3):
        self.config = config
        self.run_dir = Path(run_dir)
        # `enabled` = config-level capability: use_wandb set and package importable.
        self.enabled = (
            config.get("logging", {}).get("use_wandb", False) and wandb is not None
        )
        self._failure_count = 0
        self._max_failures = max_failures
        self._wandb_initialized = False
        # After too many consecutive failures we stop trying to LOG, but
        # we must still finalize any run we successfully initialized.
        self._logging_disabled = False
        self._wandb_run = None

    def _disable(self, reason: str):
        if not self._logging_disabled:
            logger.warning(f"WandB logging disabled: {reason}")
            self._logging_disabled = True

    def _handle_error(self, context: str, exc: Exception):
        self._failure_count += 1
        logger.warning(
            f"WandB error during {context}: {exc}. "
            f"Failure count: {self._failure_count}/{self._max_failures}"
        )
        if self._failure_count >= self._max_failures:
            self._disable(f"too many consecutive failures ({self._failure_count})")

    def on_train_begin(self, trainer):
        if not self.enabled or self._logging_disabled:
            return
        log_cfg = trainer.config.get("logging", {})

        # Resume only when reusing a run_dir that already has a checkpoint.
        ckpt_exists = (Path(trainer.run_dir) / "checkpoint_last.pth").exists()
        is_resume = ckpt_exists or trainer.trainer_state.get("attempt", 0) > 0

        # Finish any orphaned global run before we call init. This is what
        # caused the "step 0 < current step 305" warnings. Uses the module-
        # level wandb.finish() deliberately - the orphan is not *our* run.
        if wandb.run is not None:
            logger.warning("wandb.run already active, finishing orphaned run")
            try:
                wandb.finish(exit_code=1)
            except Exception:
                pass

        try:
            self._wandb_run = wandb.init(
                project=log_cfg.get("wandb_project", "bird-song-classifier"),
                name=log_cfg.get("wandb_run_name", trainer.run_dir.name),
                config=trainer.config,
                dir=str(trainer.run_dir),
                id=log_cfg.get("wandb_run_id", trainer.run_dir.name),
                resume="allow" if is_resume else "never",
                reinit=True,
            )
            self._wandb_initialized = True
            self._failure_count = 0
            self._logging_disabled = (
                False  # callback may be reused across train() calls
            )
        except Exception as e:
            self._handle_error("initialization", e)
            self._wandb_initialized = False
            self._wandb_run = None
            return

        # Tell W&B to use the `epoch` field as the x-axis for every metric
        # that carries it. This replaces what `log(step=epoch)` was trying
        # to do, without fighting the run's internal _step counter (which
        # watch() advances on its own schedule). Gradient histograms from
        # watch() don't carry an `epoch` field, so they continue to key
        # off _step — correct, because they are batch-level events.
        try:
            wandb.define_metric("epoch")
            wandb.define_metric("*", step_metric="epoch")
        except Exception as e:
            self._handle_error("define_metric", e)

        # watch() failures are non-fatal - keep the run alive for logging.
        # log_freq is bumped to at least steps_per_epoch so gradient
        # histogram events fire at most once per epoch; with smaller
        # values, watch can burn many _step values inside a single epoch,
        # polluting the _step axis for readers who chart batch-level
        # histograms alongside epoch metrics.
        try:
            log_freq = max(100, len(trainer.train_loader))
            self._wandb_run.watch(trainer.model, log="gradients", log_freq=log_freq)
        except Exception as e:
            self._handle_error("watch", e)

    def on_epoch_end(self, trainer, epoch: int, logs: Dict[str, Any]):
        if (
            not self.enabled
            or self._logging_disabled
            or not self._wandb_initialized
            or self._wandb_run is None
        ):
            return

        # W&B is observability, not a hard dependency. Any exception here
        # — a ValueError from the trainer's filter, a KeyError from a
        # subclass that excluded "epoch" by mistake, a wandb SDK failure —
        # must be caught, counted toward the three-strike disable, and
        # dropped. It must never propagate into BaseTrainer.train() and
        # kill a multi-hour run over a logging bug.
        try:
            if hasattr(trainer, "get_wandb_metrics"):
                metrics = trainer.get_wandb_metrics(logs)
            else:
                # Fallback for trainers without the hook. Mirror the same
                # non-None + exclude-keys filtering so a hand-built ``logs``
                # dict can't sneak a None step into wandb.log().
                metrics = {
                    k: v
                    for k, v in logs.items()
                    if k not in {"attempt", "run_id", "objective", "global_epoch"}
                    and v is not None
                }
                if "epoch" not in metrics:
                    metrics["epoch"] = epoch + 1

            if "epoch" not in metrics:
                raise ValueError("Epoch missing from W&B metrics")

            # No `step=` argument: W&B auto-increments its internal _step,
            # so watch()'s gradient events and this epoch log can never
            # collide on the same step number. Chart x-axis alignment with
            # console "Epoch N/M" comes from define_metric("*",
            # step_metric="epoch") in on_train_begin — not from asserting
            # a step here.
            self._wandb_run.log(metrics)
            self._failure_count = 0
            logger.debug(
                "W&B epoch log | epoch=%s | metrics=%s",
                metrics["epoch"],
                sorted(metrics.keys()),
            )
        except Exception as e:
            self._handle_error("log", e)

    def on_train_end(self, trainer):
        # Finalize whenever we successfully initialized - even if
        # _logging_disabled was set during training. Skipping this would
        # leak the W&B run object into the next experiment.
        if not self.enabled or not self._wandb_initialized or self._wandb_run is None:
            return

        try:
            state = getattr(trainer, "trainer_state", {})
            monitor = getattr(trainer, "best_monitor", None)
            best_value = state.get("best_metric")

            if monitor == "val_acc":
                self._wandb_run.summary["best_val_acc"] = best_value
            elif monitor == "val_loss":
                self._wandb_run.summary["best_val_loss"] = best_value

            if "best_epoch" in state:
                self._wandb_run.summary["best_epoch"] = state["best_epoch"]

            for file in [
                "confusion_matrix.png",
                "per_class_metrics.png",
                "evaluation_metrics.json",
            ]:
                path = self.run_dir / file
                if path.exists():
                    # FIXED: was watch() in buggy version, must be save()
                    self._wandb_run.save(str(path), base_path=str(self.run_dir))
        except Exception as e:
            self._handle_error("finalization (summary)", e)

        try:
            self._wandb_run.finish()
        except Exception as e:
            logger.warning(f"wandb.finish failed: {e}")
        finally:
            self._wandb_initialized = False
            self._wandb_run = None
