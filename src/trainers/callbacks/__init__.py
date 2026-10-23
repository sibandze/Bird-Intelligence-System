# src/trainers/callbacks/__init__.py
from .base import (
    Callback,
    CallbackRunner,
    _EPOCH_META_KEYS,
    _atomic_torch_save,
    _atomic_write_text,
    _unwrap_compile,
)
from .early_stopping import EarlyStoppingCallback
from .checkpoint import CheckpointCallback
from .json_logger import JSONLoggerCallback
from .wandb_logger import WandBLoggerCallback
from .plot_metrics import PlotMetricsCallback
from .auto_report import AutoReportCallback

__all__ = [
    "Callback",
    "CallbackRunner",
    "EarlyStoppingCallback",
    "CheckpointCallback",
    "JSONLoggerCallback",
    "WandBLoggerCallback",
    "PlotMetricsCallback",
    "AutoReportCallback",
    "_EPOCH_META_KEYS",
    "_atomic_torch_save",
    "_atomic_write_text",
    "_unwrap_compile",
]


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
