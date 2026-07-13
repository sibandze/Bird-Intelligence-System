# Experiments & Evaluation Pipeline

This document describes the comprehensive experiments framework for the Multimodal Bird Intelligence System. The goal is to establish a strong supervised baseline before implementing contrastive learning.

## Overview

The experiments pipeline enables:

1. **Systematic hyperparameter sweeps** with reproducible configurations
2. **Comprehensive evaluation metrics** at the per-class and overall level
3. **Detailed logging** of all runs for later comparison
4. **Baseline establishment** for comparing against contrastive learning

## Directory Structure

```
experiments/
├── sweep_configs.py          # Hyperparameter sweep definitions
├── experiment_runner.py       # Main experiment orchestration
├── EXPERIMENTS.md            # This file
└── results/                  # Output directory (created at runtime)
    └── exp_YYYYMMDD_HHMMSS/  # Timestamped experiment
        ├── EXPERIMENT_SUMMARY.md
        ├── results.csv              # Aggregate results across all runs
        └── run_0000_sweep_name/    # Individual run directories
            ├── config.yaml          # Exact config used
            ├── best_model.pth       # Trained model checkpoint
            ├── training_metrics.json # Epoch-by-epoch logs
            ├── evaluation_metrics.json # Final test metrics (JSON)
            ├── EVALUATION_REPORT.md # Final test metrics (Markdown)
            ├── confusion_matrix.png
            └── per_class_metrics.png
```

## Quick Start

### 1. Run a Quick Baseline Test

For testing the pipeline without long waits:

```bash
python -m experiments.experiment_runner \
  --suite quick_baseline \
  --config configs/config.yaml \
  --dry-run
```

This shows what would run without actually training.

### 2. Run Standard Baseline Experiments

A moderate-sized sweep covering learning rate, batch size, and dropout:

```bash
python -m experiments.experiment_runner \
  --suite standard_baseline \
  --config configs/config.yaml \
  --seed 42
```

### 3. Run Comprehensive Experiments

Full sweep including architecture exploration:

```bash
python -m experiments.experiment_runner \
  --suite comprehensive \
  --config configs/config.yaml \
  --seed 42
```

### 4. Run Optimization-Focused Experiments

Focus on training dynamics (warmup, scheduling, mixed precision):

```bash
python -m experiments.experiment_runner \
  --suite optimization_focus \
  --config configs/config.yaml \
  --seed 42
```

## Available Sweep Suites

### `quick_baseline` (5 runs)
- Tests learning rate sensitivity: `[1e-5, 5e-5, 1e-4, 5e-4, 1e-3]`
- **Use this** to verify the pipeline works
- **Estimated time:** ~10 minutes per run (50 total)

### `standard_baseline` (60 runs)
- Learning rates: `[1e-5, 5e-5, 1e-4, 5e-4, 1e-3]`
- Batch sizes: `[16, 32, 64]`
- Dropout rates: `[0.0, 0.1, 0.2, 0.3]`
- **Use this** for robust baseline establishment
- **Estimated time:** ~600 minutes total (~10 hours)

### `comprehensive` (3,600 runs)
- All of `standard_baseline` PLUS:
- Embedding dimensions: `[128, 256, 512]`
- Num layers: `[3, 6, 12]`
- Num heads: `[4, 8, 16]`
- **Use this** for thorough exploration (compute-intensive)
- **Estimated time:** Several days

### `optimization_focus` (36 runs)
- Learning rate + weight decay combinations
- Warmup strategies (0, 500, 1000 steps)
- Scheduler types: constant, linear_decay, cosine
- Mixed precision: on/off
- **Use this** to optimize training dynamics
- **Estimated time:** ~360 minutes total (~6 hours)

## Results Analysis

After running experiments, results are organized in:

```
results/exp_YYYYMMDD_HHMMSS/
```

### 1. Quick Overview: `results.csv`

Aggregate metrics across all runs:

```csv
run_id,sweep_name,timestamp,learning_rate,batch_size,dropout,...,accuracy,macro_f1,weighted_f1
0,baseline_lr_sweep,2025-01-15T...,0.00001,...,0.82,0.79,0.81
1,baseline_lr_sweep,2025-01-15T...,0.00005,...,0.85,0.82,0.84
...
```

**Analysis tips:**
- Sort by `accuracy`, `macro_f1`, or `weighted_f1` to find best configs
- Group by hyperparameter to see sensitivity
- Use this to select the best baseline config

### 2. Per-Run Analysis

For each run in `run_XXXX_*/`, inspect:

#### `config.yaml`
The exact hyperparameters and configuration used. **Always check this when reviewing a run.**

#### `training_metrics.json`
Epoch-by-epoch metrics:
```json
[
  {"epoch": 1, "train_loss": 4.23, "train_acc": 0.45, "val_loss": 3.87, "val_acc": 0.52},
  {"epoch": 2, "train_loss": 3.45, "train_acc": 0.58, "val_loss": 3.21, "val_acc": 0.64},
  ...
]
```

**Signals to look for:**
- Overfitting: train_acc >> val_acc → increase dropout/regularization
- Underfitting: both accuracies low → increase model capacity
- Unstable training: loss oscillates → reduce learning rate

#### `evaluation_metrics.json`
Final test set metrics with per-class breakdowns:
```json
{
  "accuracy": 0.87,
  "macro_f1": 0.84,
  "weighted_f1": 0.86,
  "per_class_precision": {"species_1": 0.89, "species_2": 0.82, ...},
  "per_class_recall": {...},
  "per_class_f1": {...},
  "per_class_support": {...},
  "top_confusions": [
    {"true_class": "species_1", "pred_class": "species_2", "count": 15},
    ...
  ]
}
```

#### `EVALUATION_REPORT.md`
Markdown-formatted evaluation report with visualizations.

#### `confusion_matrix.png` & `per_class_metrics.png`
Visualizations for quick assessment of performance.

## Workflow: From Baseline to Contrastive Learning

### Phase 1: Establish Baseline (Current)
1. Run `standard_baseline` or `comprehensive` suite
2. Identify best hyperparameters from `results.csv`
3. Document the best config in `BASELINE_CONFIG.md`
4. Save the best model and metrics

### Phase 2: Implement Contrastive Learning
1. Create new sweep suite for contrastive configs
2. Run experiments with contrastive loss
3. Keep comparable metrics format

### Phase 3: Compare Results
1. Run this command to generate a comparison:
   ```bash
   python scripts/compare_experiments.py \
     --baseline results/baseline_exp/ \
     --contrastive results/contrastive_exp/
   ```
2. Generate plots showing improvement
3. Document findings in a research note

## Configuration Files

### Base Config: `configs/config.yaml`
Defines default hyperparameters:
```yaml
training:
  batch_size: 32
  epochs: 100
  learning_rate: 0.0001
  device: "cuda"
  weight_decay: 0.0              # New: regularization
  warmup_steps: 0                # New: learning rate warmup
  scheduler_type: "constant"     # New: LR scheduling
  use_mixed_precision: false     # New: enable mixed precision

augmentation:                    # New: SpecAugment config
  enabled: true
  prob: 0.5
  freq_mask_param: 6
  time_mask_param: 10
```

### Per-Run Config: `run_XXXX_*/config.yaml`
Auto-generated override configs for each run. This is the authoritative record of what was trained.

## Extending the Framework

### Adding a New Sweep

Edit `experiments/sweep_configs.py`:

```python
NEW_SWEEP = HyperparameterSweep(
    name="my_new_sweep",
    description="Tests something new",
    params={
        "param_1": [value_a, value_b, value_c],
        "param_2": [value_x, value_y],
    }
)

SWEEP_SUITES["my_suite"] = [NEW_SWEEP]
```

Then run:
```bash
python -m experiments.experiment_runner --suite my_suite
```

### Adding New Metrics

Edit `src/evaluation/metrics_collector.py` to add custom metrics in `compute_metrics()`:

```python
def compute_metrics(self) -> Dict[str, Any]:
    # ... existing code ...
    
    # Add your metric
    my_metric = compute_my_metric(all_preds, all_labels)
    self.metrics["my_metric"] = my_metric
    
    return self.metrics
```

## Reproducibility Notes

- All runs use the same random seed (default: 42) for reproducibility
- Random seed is set in `experiment_runner.py` before each sweep
- Config is saved per-run in `config.yaml`
- To reproduce a specific run, use the saved `config.yaml` directly

## Troubleshooting

### Out of Memory (OOM)
- Reduce `batch_size`
- Reduce `embed_dim` or `num_layers`
- Enable mixed precision: `use_mixed_precision: true`

### Slow Training
- Increase `batch_size`
- Use mixed precision for ~2× speedup
- Reduce `num_layers` for faster feedback loops

### Results Not Improving
- Check learning rate: too high (loss oscillates) or too low (no progress)
- Check data augmentation: `spec_aug_prob` might be too high
- Increase model capacity: `embed_dim`, `num_layers`

## Next Steps

1. ✅ Run `quick_baseline` to verify setup works
2. ✅ Run `standard_baseline` to establish baseline metrics
3. ✅ Review `results/exp_YYYYMMDD_HHMMSS/results.csv` to find best config
4. ✅ Document baseline metrics in `BASELINE_RESULTS.md`
5. ⏭️ Proceed to contrastive learning experiments with comparable setup
