# experiments/sweep_configs.py
"""Hyperparameter sweep configurations for experiments."""

from typing import List, Dict, Any
from dataclasses import dataclass
import itertools


@dataclass
class HyperparameterSweep:
    """Defines a sweep over hyperparameter space."""
    name: str
    params: Dict[str, List[Any]]
    description: str = ""

    def generate_configs(self) -> List[Dict[str, Any]]:
        """Generate all combinations of hyperparameters."""
        keys = self.params.keys()
        values = self.params.values()
        configs = []
        for combo in itertools.product(*values):
            configs.append(dict(zip(keys, combo)))
        return configs


# ===== BASELINE SWEEPS =====

BASELINE_LEARNING_RATE_SWEEP = HyperparameterSweep(
    name="baseline_lr_sweep",
    description="Sweep over learning rates for baseline model",
    params={
        "learning_rate": [1e-5, 5e-5, 1e-4, 5e-4, 1e-3],
    }
)

BASELINE_BATCH_SIZE_SWEEP = HyperparameterSweep(
    name="baseline_batch_sweep",
    description="Sweep over batch sizes for baseline model",
    params={
        "batch_size": [16, 32, 64],
    }
)

BASELINE_ARCHITECTURE_SWEEP = HyperparameterSweep(
    name="baseline_arch_sweep",
    description="Sweep over model architecture parameters",
    params={
        "embed_dim": [128, 256, 512],
        "num_layers": [3, 6, 12],
        "heads": [4, 8, 16],
    }
)

BASELINE_DROPOUT_SWEEP = HyperparameterSweep(
    name="baseline_dropout_sweep",
    description="Sweep over dropout rates for regularization",
    params={
        "dropout": [0.0, 0.1, 0.2, 0.3],
    }
)

BASELINE_AUGMENTATION_SWEEP = HyperparameterSweep(
    name="baseline_augmentation_sweep",
    description="Sweep over SpecAugment configurations",
    params={
        "spec_aug_prob": [0.0, 0.3, 0.5, 0.7],
        "freq_mask_param": [3, 6, 10],
        "time_mask_param": [5, 10, 20],
    }
)

# ===== TARGETED EXPLORATION SWEEPS =====

FOCUSED_LR_MOMENTUM_SWEEP = HyperparameterSweep(
    name="focused_lr_momentum",
    description="Fine-tune learning rate with momentum/weight decay",
    params={
        "learning_rate": [1e-4, 3e-4, 5e-4],
        "weight_decay": [0.0, 1e-5, 1e-4],
    }
)

WARMUP_SCHEDULER_SWEEP = HyperparameterSweep(
    name="warmup_scheduler",
    description="Compare different warmup and scheduling strategies",
    params={
        "scheduler_type": ["constant", "cosine", "linear_decay", "reduce_on_plateau", "cosine_warm_restarts"],
        "warmup_steps": [0, 500, 1000],
    }
)

SCHEDULER_FINETUNE_SWEEP = HyperparameterSweep(
    name="scheduler_finetune",
    description="Fine-tune cosine scheduler parameters",
    params={
        "scheduler_type": ["cosine"],
        "warmup_steps": [250, 500, 1000, 2000],
        "min_lr": [1e-6, 1e-5, 1e-4],
    }
)

MIXED_PRECISION_SWEEP = HyperparameterSweep(
    name="mixed_precision_test",
    description="Test impact of mixed precision training",
    params={
        "use_mixed_precision": [False, True],
        "learning_rate": [1e-4, 5e-4],
    }
)

# ===== SWEEP SUITES =====

SWEEP_SUITES = {
    "quick_baseline": [
        BASELINE_LEARNING_RATE_SWEEP,
    ],
    "standard_baseline": [
        BASELINE_LEARNING_RATE_SWEEP,
        BASELINE_BATCH_SIZE_SWEEP,
        BASELINE_DROPOUT_SWEEP,
    ],
    "comprehensive": [
        BASELINE_LEARNING_RATE_SWEEP,
        BASELINE_BATCH_SIZE_SWEEP,
        BASELINE_ARCHITECTURE_SWEEP,
        BASELINE_DROPOUT_SWEEP,
        BASELINE_AUGMENTATION_SWEEP,
    ],
    "optimization_focus": [
        FOCUSED_LR_MOMENTUM_SWEEP,
        WARMUP_SCHEDULER_SWEEP,
        MIXED_PRECISION_SWEEP,
    ],
    "scheduler_ablation": [
        SCHEDULER_FINETUNE_SWEEP,
        WARMUP_SCHEDULER_SWEEP,
    ],
}
