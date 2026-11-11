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


def _cleanup_tmp(tmp: Path) -> None:
    """Best-effort removal of a leftover .tmp file. Never raises."""
    try:
        tmp.unlink(missing_ok=True)
    except Exception:
        pass


def _fsync_file(path: Path) -> None:
    """Best-effort fsync. Durability hint only - never raises."""
    try:
        with open(path, "rb") as f:
            os.fsync(f.fileno())
    except Exception:
        pass


def _fsync_dir(path: Path) -> None:
    """Best-effort fsync of the parent dir so the rename itself is durable."""
    try:
        fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except Exception:
        pass


def _atomic_torch_save(obj: Any, path: Path) -> bool:
    """Save with atomic rename, with fallbacks. Never raises.

    1. Try: write to ``<path>.tmp`` in the same directory, fsync, ``os.replace``.
       Same directory = same filesystem, so replace is atomic and cheap.
    2. Fallback: direct ``torch.save(obj, path)``. Non-atomic - a crash
       mid-write can leave a corrupt checkpoint - but a possibly-corrupt
       checkpoint is better than killing the run, and the next epoch will
       overwrite it anyway.
    3. Last resort: log and return ``False``. The run continues; the
       checkpoint for this epoch is simply missing.

    Returns ``True`` if the file was written by either path.
    Callers that need to know (e.g. checkpoint callback logging
    "saved best") should check the return value. Callers that ignore it
    get the old behaviour: no exception, run continues.
    """
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(obj, tmp)
        _fsync_file(tmp)
        os.replace(tmp, path)
        _fsync_dir(path)
        return True
    except Exception as e:
        logger.warning(
            "Atomic torch save failed for %s (%s: %s). " "Falling back to direct save.",
            path,
            type(e).__name__,
            e,
        )
        _cleanup_tmp(tmp)

    # Fallback: direct write. Still guarded - disk-full / permission
    # errors here must also not kill training.
    try:
        torch.save(obj, path)
        _fsync_file(path)
        return True
    except Exception as e:
        logger.warning(
            "Direct torch save also failed for %s (%s: %s). "
            "Continuing without saving this checkpoint.",
            path,
            type(e).__name__,
            e,
        )
        return False


def _atomic_write_text(path: Path, text: str) -> bool:
    """Write text with atomic rename, with fallbacks. Never raises.

    Same contract as ``_atomic_torch_save``: atomic first, direct write
    second, log-and-continue third. Returns ``True`` on success.
    """
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        tmp.write_text(text)
        _fsync_file(tmp)
        os.replace(tmp, path)
        _fsync_dir(path)
        return True
    except Exception as e:
        logger.warning(
            "Atomic text write failed for %s (%s: %s). "
            "Falling back to direct write.",
            path,
            type(e).__name__,
            e,
        )
        _cleanup_tmp(tmp)

    try:
        path.write_text(text)
        _fsync_file(path)
        return True
    except Exception as e:
        logger.warning(
            "Direct text write also failed for %s (%s: %s). "
            "Continuing without writing this file.",
            path,
            type(e).__name__,
            e,
        )
        return False


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
