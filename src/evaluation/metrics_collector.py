"""Comprehensive metrics collection for model evaluation."""

import json
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Any, List, Tuple
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
import matplotlib.pyplot as plt
import seaborn as sns


class MetricsCollector:
    """Collects and aggregates evaluation metrics."""
    
    def __init__(self, output_dir: Path, class_names: List[str]):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.class_names = class_names
        
        self.all_preds = []
        self.all_labels = []
        self.all_probs = []
        self.metrics = {}
    
    def add_batch(self, preds: np.ndarray, labels: np.ndarray, probs: np.ndarray = None):
        """Add predictions and labels from a batch."""
        self.all_preds.extend(preds.flatten())
        self.all_labels.extend(labels.flatten())
        if probs is not None:
            self.all_probs.extend(probs)
    
    def compute_metrics(self) -> Dict[str, Any]:
        """Compute all metrics."""
        all_preds = np.array(self.all_preds)
        all_labels = np.array(self.all_labels)
        
        # Basic accuracy
        accuracy = accuracy_score(all_labels, all_preds)
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            all_labels, all_preds, average=None, zero_division=0
        )
        macro_f1 = precision_recall_fscore_support(
            all_labels, all_preds, average="macro", zero_division=0
        )[2]
        weighted_f1 = precision_recall_fscore_support(
            all_labels, all_preds, average="weighted", zero_division=0
        )[2]
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds, labels=range(len(self.class_names)))
        
        self.metrics = {
            "accuracy": float(accuracy),
            "macro_f1": float(macro_f1),
            "weighted_f1": float(weighted_f1),
            "per_class_precision": {self.class_names[i]: float(precision[i]) for i in range(len(self.class_names))},
            "per_class_recall": {self.class_names[i]: float(recall[i]) for i in range(len(self.class_names))},
            "per_class_f1": {self.class_names[i]: float(f1[i]) for i in range(len(self.class_names))},
            "per_class_support": {self.class_names[i]: int(support[i]) for i in range(len(self.class_names))},
            "confusion_matrix": cm.tolist(),
        }
        
        # Compute top-N confusions
        self._compute_top_confusions(cm)
        
        return self.metrics
    
    def _compute_top_confusions(self, cm: np.ndarray, top_n: int = 5):
        """Find top N most common misclassifications."""
        confusions = []
        for i in range(len(self.class_names)):
            for j in range(len(self.class_names)):
                if i != j:
                    confusions.append({
                        "true_class": self.class_names[i],
                        "pred_class": self.class_names[j],
                        "count": int(cm[i, j])
                    })
        
        # Sort by count
        confusions = sorted(confusions, key=lambda x: x["count"], reverse=True)[:top_n]
        self.metrics["top_confusions"] = confusions
    
    def save_metrics_json(self, filepath: Path = None):
        """Save metrics to JSON."""
        if filepath is None:
            filepath = self.output_dir / "evaluation_metrics.json"
        
        with open(filepath, "w") as f:
            json.dump(self.metrics, f, indent=2)
        
        print(f"Saved metrics to {filepath}")
    
    def plot_confusion_matrix(self, figsize: Tuple[int, int] = (12, 10), filepath: Path = None):
        """Plot and save confusion matrix."""
        if filepath is None:
            filepath = self.output_dir / "confusion_matrix.png"
        
        cm = np.array(self.metrics["confusion_matrix"])
        
        plt.figure(figsize=figsize)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cbar_kws={"label": "Count"}
        )
        plt.title("Confusion Matrix", fontsize=16, fontweight="bold")
        plt.ylabel("True Label", fontsize=12)
        plt.xlabel("Predicted Label", fontsize=12)
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close()
        
        print(f"Saved confusion matrix to {filepath}")
    
    def plot_per_class_metrics(self, filepath: Path = None):
        """Plot per-class precision, recall, F1 as bar chart."""
        if filepath is None:
            filepath = self.output_dir / "per_class_metrics.png"
        
        precisions = [self.metrics["per_class_precision"][c] for c in self.class_names]
        recalls = [self.metrics["per_class_recall"][c] for c in self.class_names]
        f1s = [self.metrics["per_class_f1"][c] for c in self.class_names]
        
        x = np.arange(len(self.class_names))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.bar(x - width, precisions, width, label="Precision", alpha=0.8)
        ax.bar(x, recalls, width, label="Recall", alpha=0.8)
        ax.bar(x + width, f1s, width, label="F1-Score", alpha=0.8)
        
        ax.set_xlabel("Class", fontsize=12)
        ax.set_ylabel("Score", fontsize=12)
        ax.set_title("Per-Class Metrics", fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(self.class_names, rotation=45, ha="right")
        ax.legend()
        ax.set_ylim([0, 1.05])
        ax.grid(axis="y", alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close()
        
        print(f"Saved per-class metrics plot to {filepath}")
    
    def generate_markdown_report(self, filepath: Path = None):
        """Generate a markdown report of all metrics."""
        if filepath is None:
            filepath = self.output_dir / "EVALUATION_REPORT.md"
        
        report = f"""# Evaluation Report

## Overall Metrics

| Metric | Value |
|--------|-------|
| Accuracy | {self.metrics['accuracy']:.4f} |
| Macro F1 | {self.metrics['macro_f1']:.4f} |
| Weighted F1 | {self.metrics['weighted_f1']:.4f} |

## Per-Class Metrics

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|----------|
"""
        
        for class_name in self.class_names:
            precision = self.metrics["per_class_precision"][class_name]
            recall = self.metrics["per_class_recall"][class_name]
            f1 = self.metrics["per_class_f1"][class_name]
            support = self.metrics["per_class_support"][class_name]
            report += f"| {class_name} | {precision:.4f} | {recall:.4f} | {f1:.4f} | {support} |\n"
        
        report += f"\n## Top Misclassifications\n\n"
        for i, confusion in enumerate(self.metrics.get("top_confusions", []), 1):
            report += f"{i}. {confusion['true_class']} → {confusion['pred_class']} ({confusion['count']} times)\n"
        
        report += f"""\n## Confusion Matrix

![Confusion Matrix](confusion_matrix.png)

## Per-Class Metrics Visualization

![Per-Class Metrics](per_class_metrics.png)
"""
        
        with open(filepath, "w") as f:
            f.write(report)
        
        print(f"Saved evaluation report to {filepath}")
