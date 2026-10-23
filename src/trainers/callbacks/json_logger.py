# src/trainers/callbacks/json_logger.py
import csv
import json
import logging
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .base import Callback, _atomic_write_text

logger = logging.getLogger(__name__)


def _json_safe(value):
    """Recursively coerce non-finite floats to None so json.dumps(allow_nan=False) succeeds."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.floating):
        fv = float(value)
        return fv if math.isfinite(fv) else None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


class JSONLoggerCallback(Callback):
    """
    Source of truth: training_metrics.json.
    (See original docstring for full semantics.)
    """

    def __init__(self, run_dir: Path):
        self.json_path = Path(run_dir) / "training_metrics.json"
        self.archive_path = Path(run_dir) / "training_metrics.archive.jsonl"
        self.history: List[Dict[str, Any]] = []
        self._epoch_index: Dict[int, int] = {}
        self._load_from_disk()

    def _load_from_disk(self):
        if self.json_path.exists():
            try:
                self.history = json.loads(self.json_path.read_text())
                self._rebuild_index()
                return
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(
                    f"Failed to parse {self.json_path}: {e}; "
                    f"rebuilding from {self.archive_path}"
                )
        if self.archive_path.exists():
            by_epoch: Dict[int, Dict[str, Any]] = {}
            for line in self.archive_path.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "epoch" in rec:
                    by_epoch[rec["epoch"]] = rec
            self.history = sorted(by_epoch.values(), key=lambda r: r["epoch"])
            self._rebuild_index()

    def _rebuild_index(self):
        self._epoch_index = {
            h["epoch"]: i
            for i, h in enumerate(self.history)
            if h.get("epoch") is not None
        }

    def on_epoch_end(self, trainer, epoch: int, logs: Dict[str, Any]):
        epoch_key = epoch + 1

        if epoch_key in self._epoch_index:
            idx = self._epoch_index[epoch_key]
            global_epoch = self.history[idx].get("global_epoch", epoch_key)
            record = {**logs, "epoch": epoch_key, "global_epoch": global_epoch}
            self.history[idx] = record
        else:
            global_epoch = (
                max((h.get("global_epoch", 0) for h in self.history), default=0) + 1
            )
            record = {**logs, "epoch": epoch_key, "global_epoch": global_epoch}
            self.history.append(record)
            self._epoch_index[epoch_key] = len(self.history) - 1

        safe_history = _json_safe(self.history)
        _atomic_write_text(
            self.json_path,
            json.dumps(safe_history, indent=2, allow_nan=False),
        )
        with open(self.archive_path, "a") as f:
            f.write(json.dumps(_json_safe(record), allow_nan=False) + "\n")

    def export_csv(self, csv_path: Optional[Path] = None) -> None:
        csv_path = (
            Path(csv_path) if csv_path else self.json_path.parent / "training_log.csv"
        )
        if not self.history:
            logger.info("export_csv: empty history, nothing to export")
            return
        fixed = ["epoch", "global_epoch", "attempt", "run_id"]
        other_keys = sorted(
            {k for h in self.history for k in h.keys() if k not in fixed}
        )
        fieldnames = [
            f for f in fixed if any(f in h for h in self.history)
        ] + other_keys
        tmp = csv_path.with_suffix(csv_path.suffix + ".tmp")
        with open(tmp, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.history)
        os.replace(tmp, csv_path)
        logger.info(f"CSV exported to {csv_path}")
