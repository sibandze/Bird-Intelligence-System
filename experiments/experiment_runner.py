# experiments/experiment_runner.py
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
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.sweep_configs import SWEEP_SUITES
from src.training.experiment_train import ExperimentTrainer
from src.utils.configs import load_and_resolve_config


class ExperimentManager:
    """Manages experiment runs and result collection."""
    
    # Define root_dir at class level
    ROOT_DIR = Path(__file__).parent.parent

    def __init__(self, base_config_path: str, results_dir: str = "results"):
        self.base_config_path = Path(base_config_path)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True, parents=True)
        
        # Load base config using utility function
        config_rel_path = str(self.base_config_path.relative_to(self.ROOT_DIR)) if self.base_config_path.is_absolute() else str(self.base_config_path)
        self.base_config = load_and_resolve_config(self.ROOT_DIR, config_rel_path)
        
        self.experiment_name = None
        self.experiment_dir = None
        self.results_csv = None
        self.run_counter = 0
        
        # For loading data once
        self.df = None

    def set_seed(self, seed: int = 42):
        """Set random seeds for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def load_data(self):
        """Load dataset once and cache it."""
        if self.df is None:
            import pandas as pd
            from src.utils.configs import resolve_metadata_csv_path
            
            csv_path = resolve_metadata_csv_path(self.base_config)
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"Processed CSV not found at {csv_path}. Run data pipeline first with: python main.py --pipeline")
            
            self.df = pd.read_csv(csv_path)
            print(f"✓ Loaded {len(self.df)} samples from {csv_path}")
        
        return self.df

    def create_experiment_run(self, sweep_name: str, run_index: int, hyperparams: Dict[str, Any]) -> tuple:
        """Create a unique directory and config for this experiment run."""
        # Create experiment directory structure
        if not self.experiment_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.experiment_name = f"exp_{timestamp}"
            self.experiment_dir = self.results_dir / self.experiment_name
            self.experiment_dir.mkdir(exist_ok=True, parents=True)
            
            # Initialize results CSV
            self.results_csv = self.experiment_dir / "results.csv"
        
        # 1. Generate run_name matching directory naming convention
        run_name = f"run_{run_index:04d}_{sweep_name}"
        run_dir = self.experiment_dir / run_name
        run_dir.mkdir(exist_ok=True, parents=True)
        
        # 2. Pass run_name into config generator
        run_config = self._merge_config_with_hyperparams(hyperparams, run_dir, run_name)
        
        # Save config to run directory
        config_path = run_dir / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(run_config, f, default_flow_style=False)
        
        return run_dir, run_config

    def _merge_config_with_hyperparams(self, hyperparams: Dict[str, Any], run_dir: Path, run_name: str) -> Dict:
        """Merge base config with hyperparameter overrides."""
        config = yaml.safe_load(yaml.dump(self.base_config))  # Deep copy
        
        # Ensure sections exist
        if "training" not in config:
            config["training"] = {}
        if "model" not in config:
            config["model"] = {}
        if "augmentation" not in config:
            config["augmentation"] = {}
        if "logging" not in config:
            config["logging"] = {}
        
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
        
        # Configure logging and W&B settings dynamically
        config["logging"]["wandb_run_name"] = run_name
        config["logging"]["wandb_run_id"] = f"{self.experiment_name}_{run_name}"
        if "wandb_project" not in config["logging"]:
            config["logging"]["wandb_project"] = "bird-song-classifier"
        
        # Add experiment metadata
        config["experiment"] = {
            "name": run_name,
            "experiment_group": self.experiment_name,
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

    def run_experiment(self, run_index: int, sweep_name: str, hyperparams: Dict, dry_run: bool = False):
        """Run a single training experiment."""
        run_dir, run_config = self.create_experiment_run(sweep_name, run_index, hyperparams)
        
        if dry_run:
            print(f"  [{run_index}] [DRY RUN] Would train with: {hyperparams}")
            print(f"      Config: {run_dir}/config.yaml")
            return None
        
        try:
            # Create trainer and run experiment
            trainer = ExperimentTrainer(run_config, run_dir)
            
            print(f"\n  [{run_index}] Training: {hyperparams}")
            metrics = trainer.train(self.df)
            
            # Log results
            self.log_run_result(run_index, sweep_name, hyperparams, metrics)
            
            print(f"      ✓ Accuracy: {metrics.get('accuracy', 0):.4f} | Macro F1: {metrics.get('macro_f1', 0):.4f}")
            
            return metrics
        
        except Exception as e:
            print(f"      ✗ Error: {str(e)}")
            # Log failure to CSV
            metrics = {"accuracy": 0.0, "macro_f1": 0.0, "weighted_f1": 0.0, "error": str(e)}
            self.log_run_result(run_index, sweep_name, hyperparams, metrics)
            return None

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

## Best Results

"""
        
        # Try to read CSV and show top results
        if self.results_csv.exists():
            import pandas as pd
            try:
                df_results = pd.read_csv(self.results_csv)
                df_results_sorted = df_results.sort_values("accuracy", ascending=False)
                
                summary += "### Top 5 Runs by Accuracy\n\n"
                summary += "| Run ID | Accuracy | Macro F1 | Learning Rate | Batch Size |\n"
                summary += "|--------|----------|----------|---------------|------------|\n"
                
                for idx, row in df_results_sorted.head(5).iterrows():
                    summary += f"| {int(row['run_id'])} | {row['accuracy']:.4f} | {row['macro_f1']:.4f} | {row.get('learning_rate', 'N/A')} | {row.get('batch_size', 'N/A')} |\n"
            except Exception as e:
                summary += f"(Error reading results: {e})\n"
        
        summary += f"""

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
"""
        
        with open(summary_path, "w") as f:
            f.write(summary)
        
        print(f"\n✓ Experiment summary saved to {summary_path}")


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
        help="Print configurations without running training"
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
    
    print(f"\n{'='*80}")
    print(f"🚀 Running Experiment Suite: {args.suite}")
    print(f"{'='*80}\n")
    
    total_runs = sum(len(sweep.generate_configs()) for sweep in sweep_suite)
    print(f"📊 Total configurations to run: {total_runs}")
    
    if not args.dry_run:
        print(f"⚠️  This will take approximately {total_runs * 10} minutes (assuming ~10 min/run)")
        print(f"💾 Results will be saved to: {manager.results_dir}\n")
        
        # Load data once
        try:
            manager.load_data()
        except FileNotFoundError as e:
            print(f"❌ Error: {e}")
            sys.exit(1)
    
    run_index = 0
    
    # Iterate through sweeps
    for sweep in sweep_suite:
        print(f"\n{'─'*80}")
        print(f"📋 Sweep: {sweep.name}")
        print(f"   Description: {sweep.description}")
        print(f"{'─'*80}")
        
        configs = sweep.generate_configs()
        print(f"   Configurations: {len(configs)}\n")
        
        for config_idx, hyperparams in enumerate(configs, 1):
            manager.run_experiment(run_index, sweep.name, hyperparams, dry_run=args.dry_run)
            run_index += 1
    
    # Save summary
    manager.save_experiment_summary()
    
    print(f"\n{'='*80}")
    print(f"✅ Experiment complete!")
    print(f"📁 Results saved to: {manager.experiment_dir}")
    print(f"{'='*80}\n")
    
    # Print next steps
    print("📖 Next Steps:")
    print(f"   1. cd results/{manager.experiment_name}")
    print(f"   2. cat results.csv | head -20  # View top results")
    print(f"   3. Review EXPERIMENT_SUMMARY.md for overview")
    print(f"   4. Inspect individual run_XXXX_*/ directories for details")
    print()


if __name__ == "__main__":
    main()
