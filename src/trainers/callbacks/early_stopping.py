# src/trainers/callbacks/early_stopping.py
import logging
from typing import Any, Dict, Optional

import numpy as np

from .base import Callback

logger = logging.getLogger(__name__)


class EarlyStoppingCallback(Callback):
    def __init__(self, monitor: str = "val_acc", mode: str = "max", patience: int = 15):
        if mode not in {"min", "max"}:
            raise ValueError(f"mode must be 'min' or 'max', got {mode!r}")
        # `bool` is a subclass of `int`; reject it explicitly.
        if isinstance(patience, bool) or not isinstance(patience, int):
            raise ValueError(f"patience must be a positive integer, got {patience!r}")
        if patience <= 0:
            raise ValueError(f"patience must be positive, got {patience}")

        self.monitor = monitor
        self.mode = mode
        self.patience = patience
        self.best_score: Optional[float] = None
        self.patience_counter = 0

    def on_epoch_end(self, trainer, epoch: int, logs: Dict[str, Any]):
        score = logs.get(self.monitor)
        if score is None:
            return
        try:
            fv = float(score)
            if not np.isfinite(fv):
                return
        except (TypeError, ValueError):
            logger.warning("EarlyStopping: invalid %s=%r", self.monitor, score)
            return

        improved = self.best_score is None or (
            fv > self.best_score if self.mode == "max" else fv < self.best_score
        )
        if improved:
            self.best_score = fv
            self.patience_counter = 0
            return

        self.patience_counter += 1
        if self.patience_counter >= self.patience:
            print(
                f"\n ⏹ Early stopping triggered! "
                f"No improvement in '{self.monitor}' for {self.patience} epochs."
            )
            trainer.request_stop()

    def state_dict(self) -> Dict[str, Any]:
        return {
            "best_score": self.best_score,
            "patience_counter": self.patience_counter,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        if not isinstance(state_dict, dict) or not state_dict:
            return
        if "best_score" in state_dict:
            val = state_dict["best_score"]
            if val is None:
                self.best_score = None
            else:
                try:
                    fv = float(val)
                    if not np.isfinite(fv):
                        logger.warning(
                            "EarlyStopping: loaded non-finite best_score=%r, resetting to None",
                            val,
                        )
                        self.best_score = None
                    else:
                        self.best_score = fv
                except (TypeError, ValueError):
                    logger.warning(
                        "EarlyStopping: invalid loaded best_score=%r, resetting to None",
                        val,
                    )
                    self.best_score = None
        if "patience_counter" in state_dict:
            pc = state_dict["patience_counter"]
            try:
                pc_int = int(pc)
                if pc_int < 0:
                    raise ValueError
                self.patience_counter = pc_int
            except (TypeError, ValueError):
                logger.warning(
                    "EarlyStopping: invalid loaded patience_counter=%r, resetting to 0",
                    pc,
                )
                self.patience_counter = 0
