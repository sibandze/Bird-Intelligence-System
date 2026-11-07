# experiments/sweep_configs.py
"""Hyperparameter sweep configurations for experiments."""

from typing import List, Dict, Any
from dataclasses import dataclass
import itertools


@dataclass
class HyperparameterSweep:
    """
    Defines a hyperparameter sweep over a configuration space.

    Each key in `params` should be a dotted config path matching the
    experiment runner's config structure.

    Values are applied directly via set_nested_config() without a
    separate mapping dict.

    Each sweep generates the Cartesian product of its parameter values.
    """

    name: str
    params: Dict[str, List[Any]]
    description: str = ""

    def generate_configs(self) -> List[Dict[str, Any]]:
        """Generate all combinations of hyperparameters."""
        keys = list(self.params.keys())
        values = list(self.params.values())

        configs = []

        for combo in itertools.product(*values):
            configs.append(dict(zip(keys, combo)))

        return configs


# =====================================================================
# SUPERVISED BASELINE SWEEPS
# =====================================================================

BASELINE_LEARNING_RATE_SWEEP = HyperparameterSweep(
    name="baseline_lr_sweep",
    description="Sweep learning rates for the supervised baseline",
    params={
        "training.learning_rate": [
            3e-4,
            5e-4,
            1e-3,
        ],
    },
)


BASELINE_BATCH_SIZE_SWEEP = HyperparameterSweep(
    name="baseline_batch_sweep",
    description="Sweep supervised training batch sizes",
    params={
        "training.batch_size": [16, 32, 64],
    },
)


BASELINE_ARCHITECTURE_SWEEP = HyperparameterSweep(
    name="baseline_arch_sweep",
    description="Joint sweep over supervised Transformer architecture",
    params={
        "model.embed_dim": [128, 256, 512],
        "model.num_layers": [3, 6, 12],
        "model.heads": [4, 8, 16],
    },
)


BASELINE_DROPOUT_SWEEP = HyperparameterSweep(
    name="baseline_dropout_sweep",
    description="Sweep model dropout rates",
    params={
        "model.dropout": [0.0, 0.1, 0.2, 0.3],
    },
)


# ---------------------------------------------------------------------
# Supervised augmentation sweeps
# These are separated to improve interpretability.
# ---------------------------------------------------------------------

BASELINE_AUGMENTATION_PROB_SWEEP = HyperparameterSweep(
    name="baseline_augmentation_prob",
    description="Sweep SpecAugment application probability",
    params={
        "augmentation.prob": [0.0, 0.3, 0.5, 0.7],
    },
)


BASELINE_FREQ_MASK_SWEEP = HyperparameterSweep(
    name="baseline_freq_mask",
    description="Sweep SpecAugment frequency masking strength",
    params={
        "augmentation.freq_mask_param": [3, 6, 10],
    },
)


BASELINE_TIME_MASK_SWEEP = HyperparameterSweep(
    name="baseline_time_mask",
    description="Sweep SpecAugment time masking strength",
    params={
        "augmentation.time_mask_param": [5, 10, 20],
    },
)


# =====================================================================
# TARGETED SUPERVISED SWEEPS
# =====================================================================

FOCUSED_LR_WEIGHT_DECAY_SWEEP = HyperparameterSweep(
    name="focused_lr_weight_decay",
    description="Jointly fine-tune learning rate and AdamW weight decay",
    params={
        "training.learning_rate": [1e-4, 3e-4, 5e-4],
        "training.weight_decay": [0.0, 1e-5, 1e-4],
    },
)


# ---------------------------------------------------------------------
# Scheduler selection
# ---------------------------------------------------------------------
SCHEDULER_TYPE_SWEEP = HyperparameterSweep(
    name="scheduler_type",
    description="Compare scheduler policies without warmup",
    params={
        "training.scheduler_type": [
            "constant",
            "cosine",
            "linear_decay",
            "reduce_on_plateau",
            "cosine_warm_restarts",
        ],
        "training.warmup_epochs": [0],
    },
)

# ---------------------------------------------------------------------
# Warmup ablation
#
# Keep the scheduler fixed so this experiment isolates the effect
# of warmup duration.
# ---------------------------------------------------------------------
WARMUP_EPOCHS_SWEEP = HyperparameterSweep(
    name="warmup_epochs",
    description="Ablate warmup duration with cosine scheduling",
    params={
        "training.scheduler_type": ["cosine"],
        "training.warmup_epochs": [0, 2, 5, 10],
    },
)


# ---------------------------------------------------------------------
# Cosine scheduler fine-tuning
# ---------------------------------------------------------------------
SCHEDULER_FINETUNE_SWEEP = HyperparameterSweep(
    name="scheduler_finetune",
    description="Fine-tune cosine scheduler warmup (epochs) and minimum LR",
    params={
        "training.scheduler_type": ["cosine"],
        "training.warmup_epochs": [2, 5, 10, 20],
        "training.min_lr": [1e-6, 1e-5, 1e-4],
    },
)

MIXED_PRECISION_SWEEP = HyperparameterSweep(
    name="mixed_precision_test",
    description="Compare mixed precision enabled and disabled",
    params={
        "training.mixed_precision.enabled": [False, True],
        "training.learning_rate": [1e-4, 5e-4],
    },
)


# =====================================================================
# SELF-SUPERVISED LEARNING (SSL) SWEEPS
# =====================================================================

SSL_LEARNING_RATE_SWEEP = HyperparameterSweep(
    name="ssl_lr_sweep",
    description="Sweep learning rates for SimCLR pretraining",
    params={
        "training.learning_rate": [1e-4, 3e-4, 5e-4, 1e-3],
    },
)


SSL_TEMPERATURE_SWEEP = HyperparameterSweep(
    name="ssl_temperature_sweep",
    description="Sweep SimCLR NT-Xent contrastive temperature",
    params={
        # Correct path from configs/config.yaml
        "ssl.temperature": [0.03, 0.05, 0.07, 0.1, 0.2],
    },
)


SSL_PROJECTION_SWEEP = HyperparameterSweep(
    name="ssl_projection_sweep",
    description="Jointly sweep SimCLR projection head dimensions",
    params={
        # Correct paths from configs/config.yaml
        "ssl.projection_hidden_dim": [128, 256, 512],
        "ssl.projection_dim": [64, 128, 256],
    },
)


SSL_BATCH_SIZE_SWEEP = HyperparameterSweep(
    name="ssl_batch_sweep",
    description="Sweep batch sizes for SimCLR pretraining",
    params={
        "training.batch_size": [16, 32, 64, 128],
    },
)


# ---------------------------------------------------------------------
# SSL augmentation sweeps
# These are separated to avoid confounding probability and mask sizes.
# ---------------------------------------------------------------------

SSL_AUGMENTATION_PROB_SWEEP = HyperparameterSweep(
    name="ssl_augmentation_prob",
    description="Sweep SpecAugment application probability for SSL",
    params={
        "augmentation.prob": [0.3, 0.5, 0.7, 0.9],
    },
)


SSL_FREQ_MASK_SWEEP = HyperparameterSweep(
    name="ssl_freq_mask",
    description="Sweep SSL frequency masking strength",
    params={
        "augmentation.freq_mask_param": [4, 6, 10],
    },
)


SSL_TIME_MASK_SWEEP = HyperparameterSweep(
    name="ssl_time_mask",
    description="Sweep SSL time masking strength",
    params={
        "augmentation.time_mask_param": [5, 10, 20],
    },
)


SSL_WEIGHT_DECAY_SWEEP = HyperparameterSweep(
    name="ssl_weight_decay_sweep",
    description="Sweep AdamW weight decay for SimCLR pretraining",
    params={
        "training.weight_decay": [0.0, 1e-6, 1e-5, 1e-4, 1e-3],
    },
)


SSL_QUICK_SANITY = HyperparameterSweep(
    name="ssl_quick_sanity",
    description="Minimal configuration to verify the SSL pipeline",
    params={
        "training.learning_rate": [3e-4],
        "training.batch_size": [16],
        # Correct path from configs/config.yaml
        "ssl.temperature": [0.07],
    },
)


# =====================================================================
# SWEEP SUITES
# =====================================================================

SWEEP_SUITES = {
    # -----------------------------------------------------------------
    # Supervised
    # -----------------------------------------------------------------

    "quick_baseline": [
        BASELINE_LEARNING_RATE_SWEEP,
    ],

    "standard_baseline": [
        BASELINE_LEARNING_RATE_SWEEP,
        BASELINE_BATCH_SIZE_SWEEP,
        BASELINE_DROPOUT_SWEEP,
        BASELINE_AUGMENTATION_PROB_SWEEP,
    ],

    "comprehensive": [
        BASELINE_LEARNING_RATE_SWEEP,
        BASELINE_BATCH_SIZE_SWEEP,
        BASELINE_ARCHITECTURE_SWEEP,
        BASELINE_DROPOUT_SWEEP,
        BASELINE_AUGMENTATION_PROB_SWEEP,
        BASELINE_FREQ_MASK_SWEEP,
        BASELINE_TIME_MASK_SWEEP,
    ],

    "optimization_focus": [
        FOCUSED_LR_WEIGHT_DECAY_SWEEP,
        SCHEDULER_TYPE_SWEEP,
        WARMUP_EPOCHS_SWEEP,
        MIXED_PRECISION_SWEEP,
    ],

    "scheduler_ablation": [
        SCHEDULER_FINETUNE_SWEEP,
        SCHEDULER_TYPE_SWEEP,
        WARMUP_EPOCHS_SWEEP,
    ],

    # -----------------------------------------------------------------
    # SSL
    # -----------------------------------------------------------------

    "ssl_sanity": [
        SSL_QUICK_SANITY,
    ],

    "ssl_lr": [
        SSL_LEARNING_RATE_SWEEP,
    ],

    "ssl_standard": [
        SSL_LEARNING_RATE_SWEEP,
        SSL_TEMPERATURE_SWEEP,
        SSL_WEIGHT_DECAY_SWEEP,
    ],

    "ssl_comprehensive": [
        SSL_LEARNING_RATE_SWEEP,
        SSL_TEMPERATURE_SWEEP,
        SSL_PROJECTION_SWEEP,
        SSL_BATCH_SIZE_SWEEP,
        SSL_AUGMENTATION_PROB_SWEEP,
        SSL_FREQ_MASK_SWEEP,
        SSL_TIME_MASK_SWEEP,
        SSL_WEIGHT_DECAY_SWEEP,
    ],
}
