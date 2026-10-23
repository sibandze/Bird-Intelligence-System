# src/trainers/callbacks/base.py
import logging
import os
from pathlib import Path
from typing import Any, Dict

import torch

logger = logging.getLogger(__name__)

# BaseTrainer sets epoch/attempt/run_id; JSONLoggerCallback owns global_epoch.
_EPOCH_META_KEYS = frozenset({"epoch", "global_epoch", "attempt", "run_id"})


def _unwrap_compile(model):
    if hasattr(model, "_orig_mod"):
        return model._orig_mod
    return model


def _atomic_torch_save(obj: Any, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


class Callback:
    def on_train_begin(self, trainer: Any):
        pass

    def on_train_end(self, trainer: Any):
        pass

    def on_epoch_begin(self, trainer: Any, epoch: int):
        pass

    def on_epoch_end(self, trainer: Any, epoch: int, logs: Dict[str, Any]):
        pass

    def on_batch_begin(self, trainer: Any, batch: int, logs: Dict[str, Any]):
        pass

    def on_batch_end(self, trainer: Any, batch: int, logs: Dict[str, Any]):
        pass

    def on_validation_begin(self, trainer: Any):
        pass

    def on_validation_end(self, trainer: Any, logs: Dict[str, Any]):
        pass

    def state_dict(self) -> Dict[str, Any]:
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        pass


class CallbackRunner:
    """Executes callbacks in order; manages collective state."""

    def __init__(self, callbacks: list):
        self.callbacks = callbacks or []

    def on_train_begin(self, trainer):
        for cb in self.callbacks:
            cb.on_train_begin(trainer)

    def on_train_end(self, trainer):
        for cb in self.callbacks:
            cb.on_train_end(trainer)

    def on_epoch_begin(self, trainer, epoch):
        for cb in self.callbacks:
            cb.on_epoch_begin(trainer, epoch)

    def on_epoch_end(self, trainer, epoch, logs):
        for cb in self.callbacks:
            cb.on_epoch_end(trainer, epoch, logs)

    def on_batch_begin(self, trainer, batch, logs):
        for cb in self.callbacks:
            cb.on_batch_begin(trainer, batch, logs)

    def on_batch_end(self, trainer, batch, logs):
        for cb in self.callbacks:
            cb.on_batch_end(trainer, batch, logs)

    def on_validation_begin(self, trainer):
        for cb in self.callbacks:
            cb.on_validation_begin(trainer)

    def on_validation_end(self, trainer, logs):
        for cb in self.callbacks:
            cb.on_validation_end(trainer, logs)

    # --- state (unique per-instance keys, 0-indexed) ---
    def _iter_keyed(self):
        """Yield (cb, key) where key is 'ClassName#N' (N = occurrence index)."""
        counts: Dict[str, int] = {}
        for cb in self.callbacks:
            name = cb.__class__.__name__
            idx = counts.get(name, 0)
            counts[name] = idx + 1
            yield cb, f"{name}#{idx}"

    def state_dict(self) -> Dict[str, Any]:
        return {key: cb.state_dict() for cb, key in self._iter_keyed()}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        if not state_dict:
            return
        for cb, key in self._iter_keyed():
            if key in state_dict:
                cb.load_state_dict(state_dict[key])
