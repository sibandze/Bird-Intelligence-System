# src/trainers/callbacks/checkpoint.py
import logging
import random
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from .base import Callback, _atomic_torch_save, _unwrap_compile

logger = logging.getLogger(__name__)


class CheckpointCallback(Callback):
    """
    Manages all checkpoint I/O. See original docstring for semantics.

    Lifecycle contract per epoch:
      1. Read/validate the monitored score.
      2. Decide whether it improved; update `self.best_score` *before*
         building the checkpoint so the embedded `callbacks_state_dict`
         reflects the same epoch.
      3. Build the checkpoint once.
      4. Always write checkpoint_last.pth.
      5. On improvement only, write checkpoint_best.pth + best_model.pth.
    """

    def __init__(
        self,
        run_dir: Path,
        monitor: str = "val_acc",
        mode: str = "max",
        state_dict_fn=None,
    ):
        if mode not in {"min", "max"}:
            raise ValueError(f"mode must be 'min' or 'max', got {mode!r}")
        if state_dict_fn is not None and not callable(state_dict_fn):
            raise TypeError(
                f"state_dict_fn must be callable, got {type(state_dict_fn).__name__}"
            )

        self.best_score: Optional[float] = None
        self.run_dir = Path(run_dir)
        self.monitor = monitor
        self.mode = mode
        self.state_dict_fn = state_dict_fn

    def _build_checkpoint(self, trainer, epoch: int, logs: Dict[str, Any]) -> dict:
        raw_model = _unwrap_compile(trainer.model)
        return {
            "epoch": epoch + 1,
            "logs": logs,
            "model_state_dict": raw_model.state_dict(),
            "trainer_state": trainer.trainer_state,
            "optimizer_state_dict": trainer.optimizer.state_dict(),
            "scheduler_state_dict": (
                trainer.scheduler.state_dict() if trainer.scheduler else None
            ),
            "precision_state_dict": trainer.precision.state_dict(),
            "callbacks_state_dict": trainer.cb_runner.state_dict(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            ),
            "numpy_rng_state": np.random.get_state(),
            "python_rng_state": random.getstate(),
            "git_commit": getattr(trainer, "git_hash", None),
            "torch_version": torch.__version__,
            "cuda_version": (torch.version.cuda if torch.cuda.is_available() else None),
        }

    def on_train_begin(self, trainer: Any):
        # Ensure the directory exists before any atomic write.
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Monitor/mode agreement (unchanged — still critical).
        trainer_monitor = getattr(trainer, "best_monitor", None)
        trainer_mode = getattr(trainer, "best_mode", None)
        if trainer_monitor is not None and self.monitor != trainer_monitor:
            raise ValueError(
                f"Checkpoint monitor {self.monitor!r} != trainer monitor {trainer_monitor!r}"
            )
        if trainer_mode is not None and self.mode != trainer_mode:
            raise ValueError(
                f"Checkpoint mode {self.mode!r} != trainer mode {trainer_mode!r}"
            )

    def on_epoch_end(self, trainer, epoch: int, logs: Dict[str, Any]):
        # (1) Validate the monitored score.
        current_score = logs.get(self.monitor)
        improved = False
        if current_score is not None:
            try:
                fv = float(current_score)
            except (TypeError, ValueError):
                logger.warning(
                    "CheckpointCallback: invalid %s=%r; skipping best checkpoint",
                    self.monitor,
                    current_score,
                )
                fv = None
            if fv is not None:
                if not np.isfinite(fv):
                    logger.warning(
                        "CheckpointCallback: non-finite %s=%r; skipping best checkpoint",
                        self.monitor,
                        fv,
                    )
                else:
                    # (2) Determine improvement and update best_score BEFORE
                    # building the checkpoint.
                    improved = self.best_score is None or (
                        fv > self.best_score
                        if self.mode == "max"
                        else fv < self.best_score
                    )
                    if improved:
                        self.best_score = fv
                        current_score = fv

        # (3) Build once — even when validation is unavailable.
        checkpoint = self._build_checkpoint(trainer, epoch, logs)

        # (4) Always persist last.
        _atomic_torch_save(checkpoint, self.run_dir / "checkpoint_last.pth")

        if not improved:
            return

        # (5) Best artifacts only on improvement; reuse the same dict.
        _atomic_torch_save(checkpoint, self.run_dir / "checkpoint_best.pth")

        model = _unwrap_compile(trainer.model)
        weights = (
            self.state_dict_fn(model) if self.state_dict_fn else model.state_dict()
        )
        if not isinstance(weights, Mapping):
            raise TypeError(
                "CheckpointCallback.state_dict_fn must return a Mapping "
                f"(state_dict), got {type(weights).__name__}. "
                "Did you forget to call `.state_dict()`?"
            )
        _atomic_torch_save(weights, self.run_dir / "best_model.pth")
        print(f"    ✓ Saved new best model ({self.monitor}: {current_score:.4f})")

    def state_dict(self) -> Dict[str, Any]:
        return {"best_score": self.best_score}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        if not isinstance(state_dict, dict):
            logger.warning(
                "CheckpointCallback: malformed state (expected dict, got %s); ignoring",
                type(state_dict).__name__,
            )
            return
        if "best_score" not in state_dict:
            return
        val = state_dict["best_score"]
        if val is None:
            self.best_score = None
            return
        try:
            fv = float(val)
            if not np.isfinite(fv):
                logger.warning(
                    "CheckpointCallback: loaded non-finite best_score=%r, resetting to None",
                    val,
                )
                self.best_score = None
                return
            self.best_score = fv
        except (TypeError, ValueError):
            logger.warning(
                "CheckpointCallback: invalid loaded best_score=%r, resetting to None",
                val,
            )
            self.best_score = None
