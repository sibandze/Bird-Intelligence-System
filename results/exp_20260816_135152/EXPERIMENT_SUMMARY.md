# Experiment Summary

**Experiment ID:** exp_20260816_135152
**Date:** 2026-08-16 20:19:29
**Total Runs:** 5

## Results Location
- Detailed results: `results/exp_20260816_135152/results.csv`
- Run directories: `results/exp_20260816_135152/run_XXXX_*/`

## Best Results

### Top 5 Runs by Accuracy

| Run ID | Accuracy | Macro F1 | Learning Rate | Batch Size |
|--------|----------|----------|---------------|------------|
| 3 | 0.6818 | 0.6868 | 0.0005 | N/A |
| 2 | 0.6136 | 0.6207 | 0.0001 | N/A |
| 0 | 0.5455 | 0.5486 | 1e-05 | N/A |
| 1 | 0.5455 | 0.5401 | 5e-05 | N/A |
| 4 | 0.4773 | 0.4530 | 0.001 | N/A |


## Instructions

1. Review `results.csv` for aggregate metrics across all runs
2. Inspect individual `run_XXXX_*/` directories for:
   - `config.yaml` - exact hyperparameters used
   - `best_model.pth` - trained model checkpoint
   - `training_metrics.json` - epoch-by-epoch training logs
   - `evaluation_metrics.json` - final test metrics
   - `confusion_matrix.png` - confusion matrix visualization

## Next Steps

After reviewing results:
- Identify best-performing configurations
- Use those as baseline for contrastive learning experiments
- Compare final contrastive model against this baseline using: `python scripts/compare_experiments.py`
