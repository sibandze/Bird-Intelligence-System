#!/usr/bin/env python3
"""Compare results between baseline and contrastive experiments."""

import argparse
import json
import csv
from pathlib import Path
from typing import Dict, List
import numpy as np
import matplotlib.pyplot as plt


class ExperimentComparator:
    """Compares two experiment suites."""
    
    def __init__(self, baseline_dir: Path, contrastive_dir: Path):
        self.baseline_dir = Path(baseline_dir)
        self.contrastive_dir = Path(contrastive_dir)
        
        self.baseline_results = self._load_results(self.baseline_dir)
        self.contrastive_results = self._load_results(self.contrastive_dir)
    
    def _load_results(self, exp_dir: Path) -> Dict:
        """Load all results from an experiment directory."""
        results_csv = exp_dir / "results.csv"
        
        if not results_csv.exists():
            raise FileNotFoundError(f"No results.csv found in {exp_dir}")
        
        results = []
        with open(results_csv, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                results.append(row)
        
        return results
    
    def get_best_run(self, results: List[Dict], metric: str = "accuracy") -> Dict:
        """Get the best run by metric."""
        best = max(results, key=lambda x: float(x.get(metric, 0)))
        return best
    
    def get_average_metrics(self, results: List[Dict]) -> Dict[str, float]:
        """Get average metrics across all runs."""
        metrics = {}
        metric_keys = ["accuracy", "macro_f1", "weighted_f1"]
        
        for key in metric_keys:
            values = [float(r.get(key, 0)) for r in results if key in r]
            if values:
                metrics[f"avg_{key}"] = np.mean(values)
                metrics[f"std_{key}"] = np.std(values)
                metrics[f"max_{key}"] = np.max(values)
                metrics[f"min_{key}"] = np.min(values)
        
        return metrics
    
    def generate_comparison_report(self, output_file: Path = None) -> str:
        """Generate a comparison report."""
        if output_file is None:
            output_file = Path("COMPARISON_REPORT.md")
        
        baseline_avg = self.get_average_metrics(self.baseline_results)
        contrastive_avg = self.get_average_metrics(self.contrastive_results)
        
        baseline_best = self.get_best_run(self.baseline_results)
        contrastive_best = self.get_best_run(self.contrastive_results)
        
        # Calculate improvements
        improvements = {}
        for key in baseline_avg:
            if key.startswith("avg_"):
                baseline_val = baseline_avg[key]
                contrastive_val = contrastive_avg[key]
                improvement = ((contrastive_val - baseline_val) / baseline_val) * 100
                improvements[key] = improvement
        
        report = f"""# Experiment Comparison Report

## Summary

| Metric | Baseline | Contrastive | Improvement |
|--------|----------|-------------|-------------|
"""
        
        for key in ["avg_accuracy", "avg_macro_f1", "avg_weighted_f1"]:
            metric_name = key.replace("avg_", "").replace("_", " ").title()
            baseline_val = baseline_avg[key]
            contrastive_val = contrastive_avg[key]
            improvement = improvements[key]
            improvement_str = f"+{improvement:.2f}%" if improvement >= 0 else f"{improvement:.2f}%"
            report += f"| {metric_name} | {baseline_val:.4f} | {contrastive_val:.4f} | {improvement_str} |\n"
        
        report += f"\n## Baseline Results\n\n"
        report += f"**Average Metrics:**\n\n"
        for key, value in baseline_avg.items():
            report += f"- {key}: {value:.4f}\n"
        
        report += f"\n**Best Run:**\n\n"
        report += f"```json\n{json.dumps(baseline_best, indent=2)}\n```\n"
        
        report += f"\n## Contrastive Results\n\n"
        report += f"**Average Metrics:**\n\n"
        for key, value in contrastive_avg.items():
            report += f"- {key}: {value:.4f}\n"
        
        report += f"\n**Best Run:**\n\n"
        report += f"```json\n{json.dumps(contrastive_best, indent=2)}\n```\n"
        
        with open(output_file, "w") as f:
            f.write(report)
        
        print(f"Comparison report saved to {output_file}")
        return report


def main():
    parser = argparse.ArgumentParser(description="Compare baseline and contrastive experiments")
    parser.add_argument(
        "--baseline",
        type=str,
        required=True,
        help="Path to baseline experiment directory"
    )
    parser.add_argument(
        "--contrastive",
        type=str,
        required=True,
        help="Path to contrastive experiment directory"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="COMPARISON_REPORT.md",
        help="Output file for comparison report"
    )
    
    args = parser.parse_args()
    
    comparator = ExperimentComparator(args.baseline, args.contrastive)
    comparator.generate_comparison_report(Path(args.output))


if __name__ == "__main__":
    main()
