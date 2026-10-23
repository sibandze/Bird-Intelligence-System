# src/trainers/callbacks.py

# TODO:
# Add LearningRateMonitorCallback.
# Add TensorBoardLoggerCallback.
# Add RichProgressBarCallback.
# Add ModelSummaryCallback.
# Add GradientNormMonitorCallback.
# Add ExponentialMovingAverageCallback (EMA).
# Add StochasticWeightAveragingCallback (SWA).
# Add ProfilerCallback for PyTorch Profiler.
# Add LRFinderCallback.
# Add ConfusionMatrixCallback during validation

import csv
import json
import logging
import os
import random
from pathlib import Path
from typing import Dict, Any, Optional, List

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

try:
    import wandb
except ImportError:
    wandb = None


# ── Epoch-meta contract ────────────────────────────────────────────────
# BaseTrainer is responsible for setting these fields in `logs` before
# dispatching to callbacks:
#   "epoch"    : 1-indexed local epoch (int)
#   "attempt"  : resume attempt counter (0 = fresh run)
#   "run_id"   : stable run identifier (usually run_dir.name)
# JSONLoggerCallback internally derives and persists a fourth field:
#   "global_epoch" : monotonic across resumes, derived from persisted history
# All other callbacks must NOT treat these as metrics to plot or log.
_EPOCH_META_KEYS = frozenset({"epoch", "global_epoch", "attempt", "run_id"})


def _unwrap_compile(model):
    """Return the underlying module if model was wrapped by torch.compile."""
    if hasattr(model, "_orig_mod"):
        return model._orig_mod
    return model


def _atomic_write_text(path: Path, text: str) -> None:
    """Write `text` to `path` atomically (tmp file + rename)."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


class Callback:
    """Base class for all training callbacks."""

    # --- Lifecycle Hooks ---
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

    # --- State Serialization for Seamless Resume ---
    def state_dict(self) -> Dict[str, Any]:
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        pass


class CallbackRunner:
    """Executes callbacks in ordered sequence and manages collective states."""

    def __init__(self, callbacks: list):
        self.callbacks = callbacks or []

    def on_train_begin(self, trainer):
        for cb in self.callbacks:
            cb.on_train_begin(trainer)

    def on_train_end(self, trainer):
        for cb in self.callbacks:
            cb.on_train_end(trainer)

    def on_epoch_begin(self, trainer, epoch: int):
        for cb in self.callbacks:
            cb.on_epoch_begin(trainer, epoch)

    def on_epoch_end(self, trainer, epoch: int, logs: Dict[str, Any]):
        for cb in self.callbacks:
            cb.on_epoch_end(trainer, epoch, logs)

    def on_batch_begin(self, trainer, batch: int, logs: Dict[str, Any]):
        for cb in self.callbacks:
            cb.on_batch_begin(trainer, batch, logs)

    def on_batch_end(self, trainer, batch: int, logs: Dict[str, Any]):
        for cb in self.callbacks:
            cb.on_batch_end(trainer, batch, logs)

    def on_validation_begin(self, trainer):
        for cb in self.callbacks:
            cb.on_validation_begin(trainer)

    def on_validation_end(self, trainer, logs: Dict[str, Any]):
        for cb in self.callbacks:
            cb.on_validation_end(trainer, logs)

    def state_dict(self) -> Dict[str, Any]:
        return {cb.__class__.__name__: cb.state_dict() for cb in self.callbacks}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        if not state_dict:
            return
        for cb in self.callbacks:
            name = cb.__class__.__name__
            if name in state_dict:
                cb.load_state_dict(state_dict[name])


# =====================================================================
# 1. Early Stopping Callback
# =====================================================================
class EarlyStoppingCallback(Callback):
    def __init__(self, monitor: str = "val_acc", mode: str = "max", patience: int = 15):
        self.monitor = monitor
        self.mode = mode
        self.patience = patience
        self.best_score = float("-inf") if mode == "max" else float("inf")
        self.patience_counter = 0

    def on_epoch_end(self, trainer, epoch: int, logs: Dict[str, Any]):
        score = logs.get(self.monitor)
        if score is None:
            return

        improved = (
            (score > self.best_score)
            if self.mode == "max"
            else (score < self.best_score)
        )

        if improved:
            self.best_score = score
            self.patience_counter = 0
        else:
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
        self.best_score = state_dict.get("best_score", self.best_score)
        self.patience_counter = state_dict.get(
            "patience_counter", self.patience_counter
        )


# =====================================================================
# 2. Checkpoint Callback
# =====================================================================
class CheckpointCallback(Callback):
    """
    Manages all checkpoint I/O. Trainers never call torch.save directly.

    Every epoch:
      - checkpoint_last.pth  : full resumable checkpoint (complete model state
        dict, optimizer, scheduler, precision, callbacks, RNG). Always contains
        the full model including any projection head.

    On improvement:
      - checkpoint_best.pth  : same full resumable checkpoint at best metric.
      - best_model.pth       : weights-only file for inference/linear probing.
        Uses state_dict_fn if provided (e.g. lambda m: m.encoder.state_dict()
        for SSL to strip the projection head).

    Args:
        run_dir: Directory to save checkpoints.
        monitor: Metric name to watch for improvement.
        mode: 'max' (higher is better) or 'min' (lower is better).
        state_dict_fn: Optional callable(model) -> state_dict.
    """

    def __init__(
        self,
        run_dir: Path,
        monitor: str = "val_acc",
        mode: str = "max",
        state_dict_fn=None,
    ):
        self.run_dir = Path(run_dir)
        self.monitor = monitor
        self.mode = mode
        self.best_score = float("-inf") if mode == "max" else float("inf")
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

    def on_epoch_end(self, trainer, epoch: int, logs: Dict[str, Any]):
        current_score = logs.get(self.monitor)
        if current_score is None:
            return

        improved = (
            current_score > self.best_score
            if self.mode == "max"
            else current_score < self.best_score
        )
        if improved:
            self.best_score = current_score

        # Always save last checkpoint for resumption
        checkpoint = self._build_checkpoint(trainer, epoch, logs)
        torch.save(checkpoint, self.run_dir / "checkpoint_last.pth")

        if improved:
            torch.save(checkpoint, self.run_dir / "checkpoint_best.pth")
            if self.state_dict_fn is not None:
                weights = self.state_dict_fn(_unwrap_compile(trainer.model))
            else:
                weights = _unwrap_compile(trainer.model).state_dict()
            torch.save(weights, self.run_dir / "best_model.pth")
            print(f"    ✓ Saved new best model ({self.monitor}: {current_score:.4f})")

    def state_dict(self) -> Dict[str, Any]:
        return {"best_score": self.best_score}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.best_score = state_dict.get("best_score", self.best_score)


# =====================================================================
# 3. JSON Logger - Single source of truth
# =====================================================================
class JSONLoggerCallback(Callback):
    """
    Source of truth: training_metrics.json

    - Persists one record per epoch, deduped by `epoch`.
    - Enriches each record with `global_epoch` (monotonic across resumes).
    - Appends every write to training_metrics.archive.jsonl for audit.
    - Atomic JSON write (tmp + rename); on corruption, rebuilds from archive.
    - Intentionally does NOT implement state_dict/load_state_dict: history
      lives on disk, so it survives resume regardless of checkpoint state.
    - Does NOT mutate the shared `logs` dict passed to other callbacks.
      BaseTrainer owns `epoch`, `attempt`, `run_id`; this callback owns
      `global_epoch` (derived from persisted history).
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
                    by_epoch[rec["epoch"]] = rec  # last-write-wins
            self.history = sorted(by_epoch.values(), key=lambda r: r["epoch"])
            self._rebuild_index()

    def _rebuild_index(self):
        self._epoch_index = {
            h["epoch"]: i
            for i, h in enumerate(self.history)
            if h.get("epoch") is not None
        }

    def on_epoch_end(self, trainer, epoch: int, logs: Dict[str, Any]):
        epoch_key = epoch + 1  # authoritative; do not trust logs["epoch"]

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

        _atomic_write_text(self.json_path, json.dumps(self.history, indent=2))

        with open(self.archive_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def export_csv(self, csv_path: Optional[Path] = None) -> None:
        """On-demand CSV export. Source of truth remains the JSON file."""
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


# =====================================================================
# 4. WandB Logger Callback
# =====================================================================
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


# =====================================================================
# 5. Plot Metrics Callback
# =====================================================================
class PlotMetricsCallback(Callback):
    """
    Plots numeric training metrics across epochs.

    - Dedupes history by `epoch` so resumed runs don't double-plot.
    - Auto-selects numeric metrics from logs, excluding epoch-meta keys and
      known non-metric fields (learning rate, timings, etc.).
    - Uses `epoch` (local) for the x-axis. For a cross-resume view, use
      `global_epoch` from training_metrics.json directly.
    """

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
        record["epoch"] = epoch + 1  # authoritative
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
            excluded = _EPOCH_META_KEYS | {
                "learning_rate",
                "grad_norm",
                "epoch_time_sec",
                "samples_per_sec",
                "loss_scale",
                "precision",
            }
            metrics = [
                k
                for k in self.history[0].keys()
                if k not in excluded and isinstance(self.history[0][k], (int, float))
            ]
        else:
            metrics = self.metrics

        if not metrics:
            return

        num_metrics = len(metrics)
        fig, axes = plt.subplots(
            1,
            num_metrics,
            figsize=(self.figsize[0] * num_metrics, self.figsize[1]),
        )
        if num_metrics == 1:
            axes = [axes]

        epochs = [h.get("epoch", i + 1) for i, h in enumerate(self.history)]

        for ax, metric in zip(axes, metrics):
            values = [h.get(metric) for h in self.history]
            clean_epochs = [e for e, v in zip(epochs, values) if v is not None]
            clean_values = [v for v in values if v is not None]
            if not clean_values:
                continue
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
        plot_path = self.run_dir / "training_plots.png"
        plt.savefig(plot_path, dpi=100, bbox_inches="tight")
        plt.close(fig)

    def state_dict(self) -> Dict[str, Any]:
        return {"history": self.history}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.history = state_dict.get("history", [])
        self._plot()


# =====================================================================
# 6. Auto-Report Callback
# =====================================================================
class AutoReportCallback(Callback):
    """
    Runs the deterministic training-log analyst at end of training.

    Failures here must not fail an otherwise-successful training run, so
    any exception is logged and swallowed. Imports are lazy to keep
    callbacks.py independent of tools.mlops at module load time.
    """

    def on_train_end(self, trainer):
        try:
            from tools.mlops.log_analyst import TrainingLogAnalyst

            report = TrainingLogAnalyst(trainer.run_dir).analyze()

            if trainer.config.get("reporting", {}).get("llm", {}).get("enabled", False):
                from tools.mlops.llm_analyst import enrich_report

                report = enrich_report(
                    report, trainer.config, trainer.run_dir / ".llm_cache"
                )
                (trainer.run_dir / "report.json").write_text(
                    json.dumps(report, indent=2)
                )

                # Re-render md with llm banner
                md = TrainingLogAnalyst(trainer.run_dir)._render_markdown(report)
                (trainer.run_dir / "report.md").write_text(md)

        except Exception as e:
            logger.warning(f"Auto-report failed: {e}", exc_info=True)
