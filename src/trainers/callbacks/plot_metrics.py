# src/trainers/callbacks/plot_metrics.py
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .base import Callback, _EPOCH_META_KEYS

logger = logging.getLogger(__name__)

# Keys BaseTrainer emits that are not plot-worthy metrics.
_NON_METRIC_EXACT = frozenset(
    {
        "learning_rate",
        "lr",
        "grad_norm",
        "epoch_time_sec",
        "samples_per_sec",
        "loss_scale",
        "precision",
    }
)


def _is_non_metric_key(k: str) -> bool:
    if k in _EPOCH_META_KEYS or k in _NON_METRIC_EXACT:
        return True
    # Per-parameter-group learning rates: lr_encoder, encoder_lr, lr/group, ...
    if k.startswith("lr_") or k.endswith("_lr") or k.startswith("lr/"):
        return True
    return False


class PlotMetricsCallback(Callback):
    def __init__(
        self,
        run_dir: Path,
        metrics: Optional[List[str]] = None,
        figsize: tuple = (12, 5),
    ):
        self.run_dir = Path(run_dir)
        self.history: List[Dict[str, Any]] = []
        self.metrics = metrics
        self.figsize = figsize

    def on_epoch_end(self, trainer, epoch: int, logs: Dict[str, Any]):
        record = logs.copy()
        record["epoch"] = epoch + 1
        for i, h in enumerate(self.history):
            if h.get("epoch") == record["epoch"]:
                self.history[i] = record
                break
        else:
            self.history.append(record)
        self._plot()

    def on_train_end(self, trainer):
        self._plot()

    def _plot(self):
        if not self.history:
            return

        if self.metrics is None:
            metrics = [
                k
                for k in self.history[0].keys()
                if not _is_non_metric_key(k)
                and isinstance(self.history[0][k], (int, float))
            ]
        else:
            metrics = self.metrics

        if not metrics:
            return

        num_metrics = len(metrics)
        fig, axes = plt.subplots(
            1, num_metrics, figsize=(self.figsize[0] * num_metrics, self.figsize[1])
        )
        if num_metrics == 1:
            axes = [axes]

        epochs = [h.get("epoch", i + 1) for i, h in enumerate(self.history)]

        for ax, metric in zip(axes, metrics):
            values = [h.get(metric) for h in self.history]
            # Filter both None *and* non-finite floats.
            pairs = [
                (e, v)
                for e, v in zip(epochs, values)
                if isinstance(v, (int, float)) and math.isfinite(v)
            ]
            if not pairs:
                continue
            clean_epochs, clean_values = zip(*pairs)
            ax.plot(
                clean_epochs,
                clean_values,
                marker="o",
                linestyle="-",
                linewidth=2,
                markersize=4,
            )
            ax.set_title(metric)
            ax.set_xlabel("Epoch")
            ax.set_ylabel(metric)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.run_dir / "training_plots.png", dpi=100, bbox_inches="tight")
        plt.close(fig)

    def state_dict(self) -> Dict[str, Any]:
        return {"history": self.history}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        if not isinstance(state_dict, dict):
            return
        self.history = state_dict.get("history", []) or []
        self._plot()
