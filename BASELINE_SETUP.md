# Baseline Supervised Setup

This document describes the baseline supervised classification system established for the Multimodal Bird Intelligence System.

## What is the Baseline?

The baseline is a **supervised audio transformer** trained on bird species classification without any contrastive learning:

```
Mel Spectrogram → Audio Transformer → CLS Token → Classification Head → Species Logits
```

## Why Establish a Baseline?

1. **Measurement**: Provides quantitative metrics (accuracy, F1, precision/recall) for comparison
2. **Reference**: Future contrastive learning experiments will be compared against this
3. **Reproducibility**: Ensures we can recreate results and test hypotheses
4. **Documentation**: Records what works well with current data and architecture

## Baseline Snapshot

**Branch:** `baseline-supervised-v1`
- Frozen at commit: `ed6543f081ed8a89ac9c52f875ac5a4e64cd4868`
- Architecture: 6-layer transformer, 256-dim embeddings, 8 heads
- Training: 100 epochs, CrossEntropyLoss, AdamW optimizer
- Data: Stratified 80/20 train/val split

Use this branch anytime you need the original baseline.

## How to Run Baseline Experiments

### Step 1: Verify Pipeline Works

```bash
cd experiments
python experiment_runner.py --suite quick_baseline --dry-run
```

This shows the configurations that will be tested without running training.

### Step 2: Run Standard Baseline Experiments

```bash
cd experiments
python experiment_runner.py --suite standard_baseline --seed 42
```

Expected output structure:
```
results/
└── exp_20250115_143022/
    ├── EXPERIMENT_SUMMARY.md
    ├── results.csv              # <-- Main results table
    └── run_0000_baseline_lr_sweep/
        ├── config.yaml
        ├── best_model.pth
        ├── training_metrics.json
        ├── evaluation_metrics.json
        ├── EVALUATION_REPORT.md
        ├── confusion_matrix.png
        └── per_class_metrics.png
```

### Step 3: Analyze Results

```bash
# View top results
cat results/exp_20250115_143022/results.csv | head -20

# Find best by accuracy
cat results/exp_20250115_143022/results.csv | sort -t',' -k11 -nr | head -5
```

Key columns in `results.csv`:
- `accuracy` - overall test accuracy
- `macro_f1` - unweighted F1 (best for imbalanced data)
- `weighted_f1` - weighted by class support

## Metrics Interpretation

For each run, you'll see:

### Training Curves
- Should show decreasing train loss and increasing accuracy
- Val accuracy should follow train accuracy (slight lag expected)
- Divergence indicates overfitting

### Per-Class Metrics
From `evaluation_metrics.json`:
- **Precision**: Of items predicted as class X, how many are actually X?
- **Recall**: Of items actually class X, how many did we predict correctly?
- **F1**: Harmonic mean of precision and recall

### Confusion Matrix
Shows which species are confused with each other. Large off-diagonal values indicate systematic misclassifications.

## Configuration Space Tested

### `standard_baseline` Suite (60 configurations)

| Hyperparameter | Values | Notes |
|----------------|--------|-------|
| Learning Rate | [1e-5, 5e-5, 1e-4, 5e-4, 1e-3] | Primary sensitivity lever |
| Batch Size | [16, 32, 64] | Memory/convergence tradeoff |
| Dropout | [0.0, 0.1, 0.2, 0.3] | Regularization strength |
| Model | Fixed (6L, 256d, 8h) | Same architecture for all |

**Total combinations:** 5 × 3 × 4 = 60 runs

### Expected Results

Based on typical transformer training:
- **Best learning rate**: Usually 1e-4 to 5e-4
- **Best batch size**: Usually 32 (balance of stability and speed)
- **Best dropout**: Usually 0.1-0.2 (prevents overfitting without hurting learning)
- **Expected accuracy**: 70-85% depending on class balance and data quality

## After Baselines: Next Steps

1. **Document the best config**: Save the top performing configuration
2. **Create contrastive sweep**: Use similar parameter ranges for contrastive experiments
3. **Run comparison**: Use `scripts/compare_experiments.py` to benchmark improvements
4. **Write findings**: Document insights about what works and why

## Troubleshooting

### All metrics are near random (25% accuracy for 4 classes)
- Data pipeline issue: Check that labels are being loaded correctly
- Model issue: Verify model is learning (loss should decrease)
- Learning rate: Try 1e-3 for faster initial progress

### Memory errors on GPU
- Reduce batch_size: `[8, 16]`
- Reduce embed_dim: `[64, 128]`
- Reduce num_layers: `[2, 4]`

### Training stuck at same loss
- Increase learning rate: try [1e-3, 5e-3]
- Check data: ensure labels and features align
- Reduce dropout if it's very high

## Files Reference

- **Baseline code**: `src/training/experiment_train.py` (adapted from `src/training/train.py`)
- **Metrics collection**: `src/evaluation/metrics_collector.py`
- **Sweep configuration**: `experiments/sweep_configs.py`
- **Experiment runner**: `experiments/experiment_runner.py`
- **Results directory**: `results/exp_YYYYMMDD_HHMMSS/`

## Reproducing a Specific Baseline Run

If you want to re-train using the same exact configuration:

```bash
# Copy the saved config
cp results/exp_20250115_143022/run_0015_baseline_lr_sweep/config.yaml custom_config.yaml

# Train using that config
python main.py --train --config custom_config.yaml
```

The model will save to `checkpoints/best_model.pth` and can be evaluated with `--evaluate`.
