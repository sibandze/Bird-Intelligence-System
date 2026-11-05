# experiments/experiment_runner.py
"""Orchestrate multiple training experiments with different configurations."""

import os
import sys
import json
import traceback
import yaml
import argparse
from pathlib import Path
from datetime import datetime
import random
import warnings

import numpy as np
import pandas as pd
import torch
from typing import Dict, Any
import csv

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.utils.configs import (
    load_and_resolve_config,
    resolve_metadata_csv_path,
    set_nested_config,
)
from experiments.sweep_configs import SWEEP_SUITES
from src.trainers import SupervisedTransformerExperimentTrainer
from src.trainers import SimCLRExperimentTrainer

# Which suites are SSL vs supervised (by naming convention)
SSL_SUITE_PREFIXES = ("ssl_",)


def is_ssl_suite(sweep_name: str) -> bool:
    """Check if a sweep suite name indicates SSL training."""
    return any(sweep_name.startswith(prefix) for prefix in SSL_SUITE_PREFIXES)


class ExperimentManager:
    """Manages experiment runs and result collection."""

    ROOT_DIR = Path(__file__).parent.parent

    def __init__(
        self,
        base_config_path: str,
        results_dir: str = "results",
        mode: str = "auto",
        seed: int = 42,
        quick: bool = False,
    ):
        """
        Args:
            mode: 'supervised', 'ssl', or 'auto'.
                  'auto' infers from suite name prefix.
            quick: If True, force 2 epochs + 5% stratified data sample
                   (smoke-test mode). Overrides any sweep `training.epochs`.
            seed: Base seed. Per-run seeds are base_seed + run_index - 1.
        """
        self.base_config_path = Path(base_config_path)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True, parents=True)
        self.mode = mode
        self.quick = quick
        if quick:
            print("⚡ QUICK MODE: forcing 2 epochs + 5% per-class data sample")

        # base_seed is IMMUTABLE — never reassigned after __init__.
        # self.seed tracks the currently-active seed (updated per run) so
        # load_data() and other early methods have a sensible value.
        self.base_seed = seed
        self.seed = self.base_seed
        self.set_seed(seed)

        # Load base config
        config_rel_path = (
            str(self.base_config_path.relative_to(self.ROOT_DIR))
            if self.base_config_path.is_absolute()
            else str(self.base_config_path)
        )
        self.base_config = load_and_resolve_config(self.ROOT_DIR, config_rel_path)

        self.experiment_name = None
        self.experiment_dir = None
        self.results_csv = None
        self.run_counter = 0

        self.df = None

    def set_seed(self, seed: int = 42):
        """Set RNG seeds for reproducibility. Updates self.seed (the
        current run's seed) but never self.base_seed."""
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def load_data(self):
        """Load and distill dataset to num_classes and num_samples_per_class.

        In quick mode, skip distillation and take 5% per class (min 1)
        to exercise every code path on the full class set.
        """
        if self.df is not None:
            return self.df

        # Reset global RNGs to base seed before any sampling. load_data()
        # may be called after run_experiment() has already mutated self.seed
        # to a per-run seed, so without this the distillation / quick-sample
        # would depend on call order.
        self.set_seed(self.base_seed)

        csv_path = resolve_metadata_csv_path(self.base_config)
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"Processed CSV not found at {csv_path}.",
                f"Run data pipeline first: python main.py --pipeline",
            )
        full_df = pd.read_csv(csv_path)
        print(f"✓ Loaded {len(full_df)} samples from {csv_path}")

        data_cfg = self.base_config.get("data", {})
        label_col = data_cfg.get("label_column") or "scientific_name_id"

        # ── Quick mode: 5% per class, min 1, no distillation ────────────
        if self.quick:
            n_classes = full_df[label_col].nunique()
            print(
                f"⚡ Quick mode: sampling 5% per class (min 1) "
                f"across {n_classes} classes"
            )
            sampled_dfs = [
                g.sample(n=max(1, int(len(g) * 0.05)), random_state=self.base_seed)
                for _, g in full_df.groupby(label_col)
            ]
            self.df = pd.concat(sampled_dfs, ignore_index=True)
            self.df = self.df.sample(frac=1, random_state=self.base_seed).reset_index(
                drop=True
            )
            print(f"✓ Quick dataset: {len(self.df)} samples")
            return self.df

        # ── Standard distillation ───────────────────────────────────────
        use_full_dataset = data_cfg.get("full_dataset", False)

        if not use_full_dataset:
            requested_classes = data_cfg.get("num_classes")
            requested_samples = data_cfg.get("num_samples_per_class")
            print(
                f"🔬 Distilling dataset to {requested_classes} classes x "
                f"{requested_samples} samples each"
            )

            # Group by label
            grouped = full_df.groupby(label_col)
            class_counts = grouped.size().sort_values(ascending=False)

            # Check how many classes have enough samples
            eligible_classes = class_counts[class_counts >= requested_samples]

            if len(eligible_classes) < requested_classes:
                # Auto-reduce to the highest common number
                top_class_counts = class_counts.head(requested_classes)
                max_possible = int(top_class_counts.min())

                warnings.warn(
                    f"Only {len(eligible_classes)} classes have at least "
                    f"{requested_samples} samples. "
                    f"Auto-reducing samples per class to {max_possible} "
                    f"(max common across top {requested_classes} classes)."
                )
                requested_samples = max_possible

                # Re-check eligibility with reduced samples
                eligible_classes = class_counts[class_counts >= requested_samples]

                if len(eligible_classes) < requested_classes:
                    # If still not enough, reduce number of classes too
                    new_num_classes = len(eligible_classes)
                    warnings.warn(
                        f"Still only {len(eligible_classes)} classes available "
                        f"with {requested_samples} samples. "
                        f"Auto-reducing num_classes to {new_num_classes}."
                    )
                    requested_classes = new_num_classes

            selected_classes = sorted(
                eligible_classes.index[:requested_classes].tolist()
            )
            print(
                f"   Selected {len(selected_classes)} classes with "
                f"{requested_samples} samples each"
            )
            print(f"   Classes: {selected_classes}")

            # Sample per class
            sampled_dfs = []
            rng = np.random.RandomState(self.base_seed)
            for cls in selected_classes:
                cls_df = full_df[full_df[label_col] == cls]
                sampled = cls_df.sample(n=requested_samples, random_state=rng)
                sampled_dfs.append(sampled)

            self.df = pd.concat(sampled_dfs, ignore_index=True)
            print(f"✓ Final distilled dataset: {len(self.df)} samples")

            # Optional: shuffle the final dataset
            self.df = self.df.sample(frac=1, random_state=rng).reset_index(drop=True)

            # Update config with actual values used
            self.base_config["data"]["num_classes"] = requested_classes
            self.base_config["data"]["num_samples_per_class"] = requested_samples
        else:
            self.df = full_df

        return self.df

    def _resolve_mode(self, sweep_name: str) -> str:
        """Determine training mode for a given sweep."""
        if self.mode == "auto":
            return "ssl" if is_ssl_suite(sweep_name) else "supervised"
        return self.mode

    def _write_run_status(
        self,
        run_dir: Path,
        status: str,
        metrics: Dict = None,
        error: str = None,
        start_time: str = None,
        seed: int = None,
    ):
        """Write per-run run_status.json - local source of truth for Colab
        recovery. `seed` should be passed explicitly (run_seed) so the
        artifact is correct regardless of call ordering; falls back to the
        current self.seed when omitted."""
        payload = {
            "status": status,
            "run_dir": str(run_dir),
            "start_time": start_time,
            "end_time": datetime.now().isoformat(),
            "seed": seed if seed is not None else self.seed,
            "error": error,
            "metrics": metrics or {},
        }
        with open(Path(run_dir) / "run_status.json", "w") as f:
            json.dump(payload, f, indent=2, default=str)

    def create_experiment_run(
        self,
        sweep_name: str,
        run_index: int,
        hyperparams: Dict[str, Any],
        run_seed: int,
    ) -> tuple:
        """Create a unique directory and config for this experiment run."""
        if not self.experiment_dir:
            # Microseconds + exist_ok=False: two managers started in the
            # same second get distinct dirs. Refuses accidental reuse if
            # the same experiment id already exists on disk.
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            self.experiment_name = f"exp_{timestamp}"
            self.experiment_dir = self.results_dir / self.experiment_name
            self.experiment_dir.mkdir(parents=True, exist_ok=False)
            self.results_csv = self.experiment_dir / "results.csv"

        run_name = f"run_{run_index:04d}_{sweep_name}"
        run_dir = self.experiment_dir / run_name
        # exist_ok=False: never resume a stale run silently.
        run_dir.mkdir(parents=True, exist_ok=False)

        run_config = self._merge_config_with_hyperparams(
            hyperparams, run_dir, run_name, run_seed
        )

        config_path = run_dir / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(run_config, f, default_flow_style=False)

        return run_dir, run_config

    def _merge_config_with_hyperparams(
        self,
        hyperparams: Dict[str, Any],
        run_dir: Path,
        run_name: str,
        run_seed: int,
    ) -> Dict:
        """
        Merge base config with hyperparameter overrides.

        Sweep params use dotted paths (e.g. "training.learning_rate") which
        are applied directly via set_nested_config(). No mapping dict needed.
        """
        config = yaml.safe_load(yaml.dump(self.base_config))  # Deep copy

        # Ensure sections exist
        for section in (
            "training",
            "model",
            "augmentation",
            "logging",
            "projection",
            "experiment",
        ):
            if section not in config:
                config[section] = {}

        # Apply sweep hyperparameters via dotted-path keys
        for dotted_path, param_value in hyperparams.items():
            set_nested_config(config, dotted_path, param_value)

        # ── Quick mode is authoritative - must win over sweep values ────
        if self.quick:
            config["training"]["epochs"] = 2
            config["training"]["patience"] = 5
            config["training"]["warmup_steps"] = 2

        # Per-run seed for reproducible, distinct runs
        config["experiment"]["seed"] = run_seed

        # Configure logging
        config["logging"]["wandb_run_name"] = run_name
        config["logging"]["wandb_run_id"] = f"{self.experiment_name}_{run_name}"
        if "wandb_project" not in config["logging"]:
            config["logging"]["wandb_project"] = "bird-song-classifier"

        # Experiment metadata
        config["experiment"]["name"] = run_name
        config["experiment"]["experiment_group"] = self.experiment_name
        config["experiment"]["run_dir"] = str(run_dir)
        config["experiment"]["timestamp"] = datetime.now().isoformat()
        config["experiment"]["hyperparams"] = hyperparams
        if self.quick:
            config["experiment"]["quick_mode"] = True

        return config

    def log_run_result(
        self, run_index: int, sweep_name: str, hyperparams: Dict, metrics: Dict
    ):
        """Append run results to the results CSV."""
        row = {
            "run_id": run_index,
            "sweep_name": sweep_name,
            "timestamp": datetime.now().isoformat(),
        }
        row.update(hyperparams)
        row.update(metrics)

        if not self.results_csv.exists():
            with open(self.results_csv, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=row.keys())
                writer.writeheader()
                writer.writerow(row)
        else:
            # Handle case where new columns appear in later runs
            existing_rows = []
            with open(self.results_csv, "r", newline="") as f:
                reader = csv.DictReader(f)
                existing_fieldnames = reader.fieldnames or []
                for r in reader:
                    existing_rows.append(r)

            all_fieldnames = list(dict.fromkeys(existing_fieldnames + list(row.keys())))

            with open(self.results_csv, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=all_fieldnames)
                writer.writeheader()
                for r in existing_rows:
                    writer.writerow(r)
                writer.writerow(row)

        self.run_counter += 1

    def run_experiment(
        self,
        run_index: int,
        sweep_name: str,
        hyperparams: Dict,
        dry_run: bool = False,
    ):
        """Run a single training experiment."""
        # First run uses the declared base seed; subsequent runs march
        # forward deterministically (42, 43, 44, ...).
        run_seed = self.base_seed + run_index - 1
        self.set_seed(run_seed)

        run_dir, run_config = self.create_experiment_run(
            sweep_name, run_index, hyperparams, run_seed
        )

        mode = self._resolve_mode(sweep_name)
        start_time = datetime.now().isoformat()

        if dry_run:
            print(f"  [{run_index}] [DRY RUN] [{mode}] Would train with: {hyperparams}")
            print(f"      Seed: {run_seed}")
            print(f"      Epochs: {run_config['training']['epochs']}")
            print(f"      Config: {run_dir}/config.yaml")
            self._write_run_status(
                run_dir, "dry_run", start_time=start_time, seed=run_seed
            )
            return None

        try:
            if mode == "ssl":
                trainer = SimCLRExperimentTrainer(run_config, run_dir)
            else:
                trainer = SupervisedTransformerExperimentTrainer(run_config, run_dir)

            print(f"\n  [{run_index}] [{mode}] Training: {hyperparams}")
            print(f"      Seed: {run_seed}")
            metrics = trainer.train(self.df)

            # Explicit success marker so CSV ranking can distinguish
            # completed runs from crashed ones. experiment.seed is
            # recorded so the summary table can display the actual
            # per-run seed (it lives in run_config but is not otherwise
            # flattened into results.csv).
            metrics = dict(metrics) if metrics else {}
            metrics["status"] = "success"
            metrics["experiment.seed"] = run_seed
            self.log_run_result(run_index, sweep_name, hyperparams, metrics)
            self._write_run_status(
                run_dir, "success", metrics, start_time=start_time, seed=run_seed
            )

            if mode == "ssl":
                print(
                    f"      ✓ Val Loss: {metrics.get('val_loss', 0):.4f} | "
                    f"Contrastive Acc: {metrics.get('val_contrastive_acc', 0):.4f}"
                )
            else:
                print(
                    f"      ✓ Accuracy: {metrics.get('accuracy', 0):.4f} | "
                    f"Macro F1: {metrics.get('macro_f1', 0):.4f}"
                )

            return metrics

        except Exception as e:
            tb = traceback.format_exc()
            print(f"      ✗ Error during training: {str(e)}\n{tb}")

            # Belt-and-suspenders: finish orphaned W&B run if training
            # crashed before on_train_end could fire.
            try:
                import wandb as _wandb

                if _wandb.run is not None:
                    _wandb.finish(exit_code=1)
            except Exception:
                pass

            # NO zero-valued metrics - those pollute sweep ranking.
            # The `status=failed` marker lets save_experiment_summary
            # filter failed rows out of the leaderboard.
            error_metrics = {
                "status": "failed",
                "error": str(e),
                "error_traceback": tb,
                "experiment.seed": run_seed,
            }
            self.log_run_result(run_index, sweep_name, hyperparams, error_metrics)
            self._write_run_status(
                run_dir,
                "failed",
                error_metrics,
                error=str(e),
                start_time=start_time,
                seed=run_seed,
            )
            return None

    def save_experiment_summary(self, mode: str = "supervised"):
        """Generate a summary of the experiment."""
        summary_path = self.experiment_dir / "EXPERIMENT_SUMMARY.md"
        quick_note = " (⚡ quick)" if self.quick else ""
        summary = (
            f"# Experiment Summary\n"
            f"\n"
            f"**Experiment ID:** {self.experiment_name}\n"
            f"**Mode:** {mode}{quick_note}\n"
            f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"**Total Runs:** {self.run_counter}\n"
            f"**Base Seed:** {self.base_seed}\n"
            f"\n"
            f"## Results Location\n"
            f"- Detailed results: `{self.results_csv}`\n"
            f"- Run directories: `{self.experiment_dir}/run_XXXX_*/`\n"
            f"\n"
        )

        if self.results_csv.exists():
            try:
                df_results = pd.read_csv(self.results_csv)

                # Failed runs must not pollute the leaderboard. Older
                # CSVs predating the status column are assumed successful.
                if "status" in df_results.columns:
                    df_results = df_results[df_results["status"] == "success"]

                if len(df_results) == 0:
                    summary += "### No successful runs to rank\n\n"
                elif mode == "ssl":
                    sort_col = "val_loss"
                    ascending = True
                    summary += "### Top 5 Runs by Val Loss (lowest first)\n\n"
                    summary += "| Run ID | Val Loss | Contrastive Acc | LR | Batch Size | Seed |\n"
                    summary += "|--------|----------|-----------------|-----|------------|------|\n"

                    df_sorted = df_results.sort_values(sort_col, ascending=ascending)
                    for _, row in df_sorted.head(5).iterrows():
                        summary += (
                            f"| {int(row['run_id'])} "
                            f"| {row.get('val_loss', 0):.4f} "
                            f"| {row.get('val_contrastive_acc', 0):.4f} "
                            f"| {row.get('training.learning_rate', 'N/A')} "
                            f"| {row.get('training.batch_size', 'N/A')} "
                            f"| {row.get('experiment.seed', 'N/A')} |\n"
                        )
                else:
                    sort_col = "accuracy"
                    ascending = False
                    summary += "### Top 5 Runs by Accuracy\n\n"
                    summary += "| Run ID | Accuracy | Macro F1 | Learning Rate | Batch Size | Seed |\n"
                    summary += "|--------|----------|----------|---------------|------------|------|\n"

                    df_sorted = df_results.sort_values(sort_col, ascending=ascending)
                    for _, row in df_sorted.head(5).iterrows():
                        summary += (
                            f"| {int(row['run_id'])} "
                            f"| {row.get('accuracy', 0):.4f} "
                            f"| {row.get('macro_f1', 0):.4f} "
                            f"| {row.get('training.learning_rate', 'N/A')} "
                            f"| {row.get('training.batch_size', 'N/A')} "
                            f"| {row.get('experiment.seed', 'N/A')} |\n"
                        )
            except Exception as e:
                summary += f"(Error reading results: {e})\n"

        summary += (
            f"\n"
            f"## Instructions\n"
            f"\n"
            f"1. Review `results.csv` for aggregate metrics across all runs\n"
            f"2. Inspect individual `run_XXXX_*/` directories for:\n"
            f"   - `config.yaml` - exact hyperparameters used\n"
            f"   - `label_mappings.json` - class name to index mapping\n"
            f"   - `checkpoint_best.pth` - trained model checkpoint\n"
            f"   - `training_metrics.json` - epoch-by-epoch training logs\n"
        )

        if mode == "ssl":
            summary += (
                f"   - `checkpoint_best.pth` - encoder-only weights (projection head excluded)\n"
                f"\n"
                f"## Next Steps\n"
                f"\n"
                f"After reviewing SSL results:\n"
                f"- Use best encoder checkpoint for linear probing evaluation\n"
                f"- Compare against supervised baseline\n"
            )
        else:
            summary += (
                f"   - `evaluation_metrics.json` - final test metrics\n"
                f"   - `confusion_matrix.png` - confusion matrix visualization\n"
                f"\n"
                f"## Next Steps\n"
                f"\n"
                f"After reviewing results:\n"
                f"- Identify best-performing configurations\n"
                f"- Use those as baseline for contrastive learning experiments\n"
            )

        if self.quick:
            summary += (
                f"\n"
                f"## ⚡ Quick Mode Notice\n"
                f"\n"
                f"This run was executed in **quick mode**:\n"
                f"- Epochs forced to 2\n"
                f"- Data reduced to 5% per class (min 1 sample)\n"
                f"- Patience forced to 5\n"
                f"\n"
                f"These results are for **smoke-testing** only — do not\n"
                f"compare them against full-data baselines.\n"
            )

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
        help="Which sweep suite to run",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="auto",
        choices=["supervised", "ssl", "auto"],
        help="Training mode: supervised, ssl, or auto (inferred from suite name)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to base config file",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory to save experiment results",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print configurations without running training",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Smoke test: force 2 epochs + 5%% per-class data sample",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    # Initialize experiment manager. Base seed is set once here and stays
    # immutable for the lifetime of the sweep.
    manager = ExperimentManager(
        args.config,
        args.results_dir,
        mode=args.mode,
        quick=args.quick,
        seed=args.seed,
    )

    # Get sweep suite
    sweep_suite = SWEEP_SUITES[args.suite]

    # Resolve effective mode for display
    effective_mode = manager._resolve_mode(args.suite)

    print(f"\n{'='*80}")
    print(f"🚀 Running Experiment Suite: {args.suite}")
    print(f"   Mode: {effective_mode}")
    print(f"   Base Seed: {args.seed}")
    if args.quick:
        print(f"   ⚡ QUICK MODE: 2 epochs, 5% per-class data")
    print(f"{'='*80}\n")

    total_runs = sum(len(sweep.generate_configs()) for sweep in sweep_suite)
    print(f"📊 Total configurations to run: {total_runs}")

    if not args.dry_run:
        est_min = 1 if args.quick else 10
        print(
            f"⚠️  This will take approximately {total_runs * est_min} minutes "
            f"(assuming ~{est_min} min/run)"
        )
        print(f"💾 Results will be saved to: {manager.results_dir}\n")

        try:
            manager.load_data()
        except FileNotFoundError as e:
            print(f"❌ Error: {e}")
            sys.exit(1)

    run_index = 1

    for sweep in sweep_suite:
        print(f"\n{'─'*80}")
        print(f"📋 Sweep: {sweep.name}")
        print(f"   Description: {sweep.description}")
        print(f"{'─'*80}")

        configs = sweep.generate_configs()
        print(f"   Configurations: {len(configs)}\n")

        for config_idx, hyperparams in enumerate(configs, 1):
            manager.run_experiment(
                run_index, sweep.name, hyperparams, dry_run=args.dry_run
            )
            run_index += 1

    # Save summary with mode context
    manager.save_experiment_summary(mode=effective_mode)

    print(f"\n{'='*80}")
    print(f"✅ Experiment complete!")
    print(f"📁 Results saved to: {manager.experiment_dir}")
    print(f"{'='*80}\n")

    print("📖 Next Steps:")
    print(f"   1. cd results/{manager.experiment_name}")
    print(f"   2. cat results.csv | head -20  # View top results")
    print(f"   3. Review EXPERIMENT_SUMMARY.md for overview")
    print(f"   4. Inspect individual run_XXXX_*/ directories for details")
    print()


if __name__ == "__main__":
    main()
