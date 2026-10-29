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

    def __init__(self, run_dir, monitor="val_acc", mode="max", state_dict_fn=None):
        if mode not in {"min", "max"}:
            raise ValueError(f"mode must be 'min' or 'max', got {mode!r}")
        if state_dict_fn is not None and not callable(state_dict_fn):
            raise TypeError(
                f"state_dict_fn must be callable, got {type(state_dict_fn).__name__}"
            )
        self.best_score: Optional[float] = None
        self.best_epoch: Optional[int] = None  # 1-indexed, matches checkpoint["epoch"]
        self.run_dir = Path(run_dir)
        self.monitor = monitor
        self.mode = mode
        self.state_dict_fn = state_dict_fn

    def on_train_begin(self, trainer: Any):
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Monitor/mode agreement
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

        # Adopt trainer_state as authoritative on resume.
        #
        # BaseTrainer._update_best() runs *before* cb_runner.on_epoch_end(),
        # so trainer_state is always at least as current as the callback's
        # own best_epoch/best_score when a checkpoint is written. If the two
        # disagree here (ambiguous CallbackRunner restore,
        # or user-edited state), the callback's tracking is stale and must be
        # realigned to prevent it from saving a worse model as "best".
        ts = getattr(trainer, "trainer_state", None) or {}
        ts_best = ts.get("best_metric")
        ts_best_epoch = ts.get("best_epoch")

        if self.best_score != ts_best or self.best_epoch != ts_best_epoch:
            logger.warning(
                "CheckpointCallback: best-state mismatch with trainer_state "
                "(callback=(score=%r, epoch=%r), trainer=(score=%r, epoch=%r)); "
                "adopting trainer_state as the source of truth.",
                self.best_score,
                self.best_epoch,
                ts_best,
                ts_best_epoch,
            )
            self.best_score = ts_best
            self.best_epoch = ts_best_epoch

    def on_epoch_end(self, trainer, epoch: int, logs: Dict[str, Any]):
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
                    improved = self.best_score is None or (
                        fv > self.best_score
                        if self.mode == "max"
                        else fv < self.best_score
                    )
                    if improved:
                        self.best_score = fv
                        self.best_epoch = epoch + 1
                        current_score = fv

        checkpoint = self._build_checkpoint(trainer, epoch, logs)
        _atomic_torch_save(checkpoint, self.run_dir / "checkpoint_last.pth")

        if not improved:
            return

        # see §3 — validate inference weights BEFORE writing best checkpoint
        model = _unwrap_compile(trainer.model)
        weights = (
            self.state_dict_fn(model) if self.state_dict_fn else model.state_dict()
        )
        _validate_state_dict_like(
            weights,
            context="state_dict_fn" if self.state_dict_fn else "model.state_dict()",
        )

        _atomic_torch_save(checkpoint, self.run_dir / "checkpoint_best.pth")
        _atomic_torch_save(weights, self.run_dir / "best_model.pth")
        print(f"    ✓ Saved new best model ({self.monitor}: {current_score:.4f})")

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

    def state_dict(self) -> Dict[str, Any]:
        return {"best_score": self.best_score, "best_epoch": self.best_epoch}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        if not isinstance(state_dict, dict):
            logger.warning(
                "CheckpointCallback: malformed state (expected dict, got %s); ignoring",
                type(state_dict).__name__,
            )
            return
        if "best_score" in state_dict:
            val = state_dict["best_score"]
            if val is None:
                self.best_score = None
            else:
                try:
                    fv = float(val)
                    self.best_score = fv if np.isfinite(fv) else None
                    if self.best_score is None:
                        logger.warning(
                            "CheckpointCallback: non-finite loaded best_score=%r, resetting to None",
                            val,
                        )
                except (TypeError, ValueError):
                    logger.warning(
                        "CheckpointCallback: invalid loaded best_score=%r, resetting to None",
                        val,
                    )
                    self.best_score = None
        if "best_epoch" in state_dict:
            be = state_dict["best_epoch"]
            if be is None:
                self.best_epoch = None
            else:
                try:
                    be_int = int(be)
                    if be_int < 1:
                        raise ValueError
                    self.best_epoch = be_int
                except (TypeError, ValueError):
                    logger.warning(
                        "CheckpointCallback: invalid loaded best_epoch=%r, resetting to None",
                        be,
                    )
                    self.best_epoch = None

    def _validate_state_dict_like(self, weights: Any, context: str) -> None:
        """Reject obviously-wrong state_dict_fn output before it becomes an artifact."""
        if not isinstance(weights, Mapping):
            raise TypeError(
                f"CheckpointCallback: {context} must return a Mapping, "
                f"got {type(weights).__name__}. Did you forget `.state_dict()`?"
            )
        if not weights:
            raise ValueError(
                f"CheckpointCallback: {context} returned an empty Mapping."
            )
        non_str = next((k for k in weights if not isinstance(k, str)), None)
        if non_str is not None:
            raise TypeError(
                f"CheckpointCallback: {context} has non-string keys "
                f"(first: {non_str!r}); expected model state_dict keys."
            )
        if not any(isinstance(v, torch.Tensor) for v in weights.values()):
            raise TypeError(
                f"CheckpointCallback: {context} contains no tensor values; "
                "not a model state_dict."
            )
        non_tensor = [k for k, v in weights.items() if not isinstance(v, torch.Tensor)]
        if non_tensor:
            logger.warning(
                "CheckpointCallback: %s contains %d non-tensor value(s) "
                "(first key: %r); best_model.pth may not load cleanly via "
                "model.load_state_dict().",
                context,
                len(non_tensor),
                non_tensor[0],
            )
