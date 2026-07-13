"""Orchestrate multiple training experiments with different configurations."""

import os
import sys
import json
import yaml
import argparse
from pathlib import Path
from datetime import datetime
import random
import numpy as np
import torch
from typing import Dict, Any, List
import csv

from sweep_configs import SWEEP_SUITES


class ExperimentManager:
    """Manages experiment runs and result collection."""

    def __init__(self, base_config_path: str, results_dir: str = "results"):
        self.base_config_path = Path(base_config_path)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # Load base config
        with open(self.base_config_path, "r") as f:
            self.base_config = yaml.safe_load(f)
        
        self.experiment_name = None
        self.experiment_dir = None
        self.results_csv = None
        self.run_counter = 0

    def set_seed(self, seed: int = 42):
        """Set random seeds for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def create_experiment_run(self, sweep_name: str, run_index: int, hyperparams: Dict[str, Any]) -> tuple[Path, Dict]:
        """Create a unique directory and config for this experiment run."""
        # Create experiment directory structure
        if not self.experiment_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.experiment_name = f"exp_{timestamp}"
            self.experiment_dir = self.results_dir / self.experiment_name
            self.experiment_dir.mkdir(exist_ok=True)
            
            # Initialize results CSV
            self.results_csv = self.experiment_dir / "results.csv"
        
        # Create run-specific directory
        run_dir = self.experiment_dir / f"run_{run_index:04d}_{sweep_name}"
        run_dir.mkdir(exist_ok=True)
        
        # Create run-specific config
        run_config = self._merge_config_with_hyperparams(hyperparams, run_dir)
        
        # Save config to run directory
        config_path = run_dir / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(run_config, f, default_flow_style=False)
        
        return run_dir, run_config

    def _merge_config_with_hyperparams(self, hyperparams: Dict[str, Any], run_dir: Path) -> Dict:
        """Merge base config with hyperparameter overrides."""
        config = yaml.safe_load(yaml.dump(self.base_config))  # Deep copy
        
        # Map hyperparams to config sections
        param_mapping = {
            "learning_rate": ("training", "learning_rate"),
            "batch_size": ("training", "batch_size"),
            "embed_dim": ("model", "embed_dim"),
            "num_layers": ("model", "num_layers"),
            "heads": ("model", "heads"),
            "dropout": ("model", "dropout"),
            "weight_decay": ("training", "weight_decay"),
            "warmup_steps": ("training", "warmup_steps"),
            "scheduler_type": ("training", "scheduler_type"),
            "use_mixed_precision": ("training", "use_mixed_precision"),
            "spec_aug_prob": ("augmentation", "prob"),
            "freq_mask_param": ("augmentation", "freq_mask_param"),
            "time_mask_param": ("augmentation", "time_mask_param"),
        }
        
        for param_name, param_value in hyperparams.items():
            if param_name in param_mapping:
                section, key = param_mapping[param_name]
                if section not in config:
                    config[section] = {}
                config[section][key] = param_value
        
        # Add experiment metadata
        config["experiment"] = {
            "run_dir": str(run_dir),
            "timestamp": datetime.now().isoformat(),
            "hyperparams": hyperparams,
        }
        
        return config

    def log_run_result(self, run_index: int, sweep_name: str, hyperparams: Dict, metrics: Dict):
        """Append run results to the results CSV."""
        # Prepare row
        row = {
            "run_id": run_index,
            "sweep_name": sweep_name,
            "timestamp": datetime.now().isoformat(),
        }
        row.update(hyperparams)
        row.update(metrics)
        
        # Write to CSV
        if not self.results_csv.exists():
            with open(self.results_csv, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=row.keys())
                writer.writeheader()
                writer.writerow(row)
        else:
            with open(self.results_csv, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=row.keys())
                writer.writerow(row)
        
        self.run_counter += 1

    def save_experiment_summary(self):
        """Generate a summary of the experiment."""
        summary_path = self.experiment_dir / "EXPERIMENT_SUMMARY.md"
        
        summary = f"""# Experiment Summary

**Experiment ID:** {self.experiment_name}
**Date:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Total Runs:** {self.run_counter}

## Results Location
- Detailed results: `{self.results_csv}`
- Run directories: `{self.experiment_dir}/run_XXXX_*/`

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
- Compare final contrastive model against this baseline
"""
        
        with open(summary_path, "w") as f:
            f.write(summary)


def main():
    parser = argparse.ArgumentParser(description="Run hyperparameter sweep experiments")
    parser.add_argument(
        "--suite",
        type=str,
        default="quick_baseline",
        choices=list(SWEEP_SUITES.keys()),
        help="Which sweep suite to run"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to base config file"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory to save experiment results"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print configurations without running"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    # Initialize experiment manager
    manager = ExperimentManager(args.config, args.results_dir)
    manager.set_seed(args.seed)
    
    # Get sweep suite
    sweep_suite = SWEEP_SUITES[args.suite]
    
    print(f"\n{'='*70}")
    print(f"Running Experiment Suite: {args.suite}")
    print(f"{'='*70}\n")
    
    total_runs = sum(len(sweep.generate_configs()) for sweep in sweep_suite)
    print(f"Total configurations to run: {total_runs}\n")
    
    run_index = 0
    
    # Iterate through sweeps
    for sweep in sweep_suite:
        print(f"\n>>> Sweep: {sweep.name}")
        print(f"    Description: {sweep.description}")
        print(f"    Configurations: {len(sweep.generate_configs())}")
        
        configs = sweep.generate_configs()
        
        for config_idx, hyperparams in enumerate(configs, 1):
            run_dir, run_config = manager.create_experiment_run(sweep.name, run_index, hyperparams)
            
            print(f"\n  [{config_idx}/{len(configs)}] Run {run_index}: {hyperparams}")
            
            if args.dry_run:
                print(f"      [DRY RUN] Would save config to {run_dir}/config.yaml")
            else:
                # TODO: Here is where you would call the actual training function
                # from src.training.train import train_model
                # metrics = train_model(run_config)
                # manager.log_run_result(run_index, sweep.name, hyperparams, metrics)
                print(f"      [PLACEHOLDER] Training would run here")
                print(f"      Config saved to: {run_dir}/config.yaml")
            
            run_index += 1
    
    manager.save_experiment_summary()
    print(f"\n{'='*70}")
    print(f"Experiment complete! Results saved to: {manager.experiment_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
