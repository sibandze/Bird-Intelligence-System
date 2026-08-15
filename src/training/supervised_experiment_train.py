# src/training/supervised_experiment_train.py

import subprocess
import time
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.datasets import SupervisedBirdSongDataset
from src.evaluation.metrics_collector import MetricsCollector
from src.models import SupervisedTransformer
from src.training.precision import PrecisionManager
from src.training.scheduler import create_scheduler, get_scheduler_step_frequency
from src.utils.memory_utils import get_gpu_memory_info, log_memory_usage
from src.training.callbacks import (
    Callback, CallbackRunner, CheckpointCallback, EarlyStoppingCallback,
    JSONLoggerCallback, CSVLoggerCallback, WandBLoggerCallback
)

def supervised_val_collate_fn(batch):
    """
    Collate for validation/test batches of (x, y, recording_id).
    Stacks tensors but keeps recording_ids as a list of strings/ints.
    """
    mel_segments, labels, recording_ids = zip(*batch)
    mel_segments = torch.stack(mel_segments, dim=0)
    labels = torch.stack(labels, dim=0)
    return mel_segments, labels, list(recording_ids)


def aggregate_recordings(logits_all, labels_all, recording_ids_all):
    """
    Aggregate window-level outputs to recording-level predictions.

    Args:
        logits_all: tensor of shape [total_windows, num_classes]
        labels_all: tensor of shape [total_windows]
        recording_ids_all: list of recording IDs (same length as total_windows)

    Returns:
        rec_ids: list of unique recording IDs
        rec_targets: tensor of true labels [num_recordings]
        rec_logits: tensor of averaged logits [num_recordings, num_classes]
    """
    rec_ids = []
    rec_targets = []
    rec_logits = []

    # Convert recording_ids to strings for consistent grouping
    rec_ids_str = [str(r) for r in recording_ids_all]

    unique_ids = []
    for r in rec_ids_str:
        if r not in unique_ids:
            unique_ids.append(r)

    for rec_id in unique_ids:
        mask = np.array([r == rec_id for r in rec_ids_str])
        avg_logits = logits_all[mask].mean(dim=0)  # [num_classes]
        true_label = labels_all[mask][0]           # all same for a recording

        rec_ids.append(rec_id)
        rec_targets.append(true_label.item())
        rec_logits.append(avg_logits)

    rec_targets = torch.tensor(rec_targets, dtype=torch.long)
    rec_logits = torch.stack(rec_logits, dim=0)
    return rec_ids, rec_targets, rec_logits

class SupervisedExperimentTrainer:
    """Supervised learning training engine with callback-driven architecture."""

    def __init__(self, config: Dict[str, Any], run_dir: Path, callbacks: Optional[List[Callback]] = None):
        self.config = config
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(exist_ok=True, parents=True)

        self.device = torch.device(config['training'].get('device', 'cuda') if torch.cuda.is_available() else 'cpu')
        self.precision = PrecisionManager(
            enabled=config["training"].get("mixed_precision", {}).get("enabled", True),
            device=self.device.type,
            use_bfloat16=config["training"].get("mixed_precision", {}).get("use_bfloat16", False),
        )

        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.stop_training = False

        # Callback order matters! EarlyStopping and Checkpoint come before Loggers
        if callbacks is None:
            callbacks = [
                CheckpointCallback(self.run_dir, monitor="val_acc", mode="max"),
                EarlyStoppingCallback(monitor="val_acc", mode="max", patience=config["training"].get("patience", 15)),
                JSONLoggerCallback(self.run_dir),
                CSVLoggerCallback(self.run_dir),
                WandBLoggerCallback(config, self.run_dir)
            ]
        self.cb_runner = CallbackRunner(callbacks)

        # Store environment state
        try:
            self.git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
        except Exception:
            self.git_hash = None

    def request_stop(self):
        """Clean external API for callbacks to trigger early stopping."""
        self.stop_training = True

    def get_dataloaders(self, df: pd.DataFrame):
        batch_size = self.config['training']['batch_size']
        num_workers = self.config['training']['num_workers']
        segment_size = self.config["audio"]["segment_size"]

        # Recording-level train/val/test split (70/15/15 by default)
        test_size = self.config['training'].get('test_size', 0.15)
        val_size = self.config['training'].get('val_size', 0.15)

        # First split: train + temp
        train_df, temp_df = train_test_split(
            df,
            test_size=(val_size + test_size),
            random_state=42,
            stratify=df['scientific_name_id']
        )
        # Second split: val + test from temp
        val_df, test_df = train_test_split(
            temp_df,
            test_size=test_size / (val_size + test_size),
            random_state=42,
            stratify=temp_df['scientific_name_id']
        )

        window_config = self.config['window']

        train_dataset = SupervisedBirdSongDataset(
            df=train_df,
            segment_size=segment_size,
            train=True,
            spec_aug_config=self._get_augmentation_config(),
            min_db=self.config['audio']['min_db'],
            max_db=self.config['audio']['max_db'],
            window_config=window_config,
        )

        val_dataset = SupervisedBirdSongDataset(
            df=val_df,
            segment_size=segment_size,
            train=False,
            label_to_idx=train_dataset.label_to_idx,
            min_db=self.config['audio']['min_db'],
            max_db=self.config['audio']['max_db'],
            window_config={
                "strategy": "sliding",
                "stride": segment_size,   # non-overlapping windows for validation
            },
            return_recording_id=True,
        )

        test_dataset = SupervisedBirdSongDataset(
            df=test_df,
            segment_size=segment_size,
            train=False,
            label_to_idx=train_dataset.label_to_idx,
            min_db=self.config['audio']['min_db'],
            max_db=self.config['audio']['max_db'],
            window_config={
                "strategy": "sliding",
                "stride": segment_size,
            },
            return_recording_id=True,
        )

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=(self.device.type == "cuda"),
            persistent_workers=num_workers > 0
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=(self.device.type == "cuda"),
            persistent_workers=num_workers > 0,
            collate_fn=supervised_val_collate_fn
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=(self.device.type == "cuda"),
            persistent_workers=num_workers > 0,
            collate_fn=supervised_val_collate_fn
        )

        return train_loader, val_loader, test_loader, train_dataset.label_to_idx, train_dataset.idx_to_label

    def _get_augmentation_config(self) -> Dict:
        """Extract spec augmentation configuration."""
        aug_cfg = self.config['augmentation']
        return {
            'enabled': aug_cfg.get('enabled', True),
            'prob': aug_cfg.get('prob', 0.5),
            'num_freq_masks': aug_cfg.get('num_freq_masks', 2),
            'freq_mask_param': aug_cfg.get('freq_mask_param', 6),
            'num_time_masks': aug_cfg.get('num_time_masks', 2),
            'time_mask_param': aug_cfg.get('time_mask_param', 10),
        }

    def train(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Run supervised training loop."""
        train_loader, val_loader, test_loader, label_to_idx, idx_to_label = self.get_dataloaders(df)
        class_names = [idx_to_label[i] for i in range(len(idx_to_label))]

        segment_size = self.config['audio']['segment_size']

        # Initialize Model
        self.model = SupervisedTransformer(
            config=self.config,
            device=str(self.device),
            num_classes=len(class_names),
        ).to(self.device)

        # Print Model and Environment Summary
        num_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        compiled = self.config["training"].get("compile_model", False)

        print(f"\n>>> Initializing Supervised Training Run:")
        print(f"    Device:    {self.device}")
        print(f"    Precision: {self.precision.precision_name()}")
        print(f"    Compiled:  {compiled}")
        print(f"    Params:    {num_params:,} (Trainable: {trainable_params:,})")
        print(f"    Classes:   {len(class_names)}")
        print(f"    Train samples: {len(train_loader.dataset)}")
        print(f"    Test samples:  {len(val_loader.dataset)}")

        if compiled:
            self.model = torch.compile(self.model)

        criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training'].get('weight_decay', 0.01)
        )

        epochs = self.config['training']['epochs']
        scheduler_type = self.config['training'].get('scheduler_type', 'cosine')
        warmup_steps = self.config['training'].get('warmup_steps', 0)
        total_steps = max(len(train_loader) * epochs, warmup_steps*2)

        self.scheduler = create_scheduler(
            optimizer=self.optimizer, scheduler_type=scheduler_type,
            warmup_steps=warmup_steps, total_steps=total_steps,
            min_lr=self.config['training'].get('min_lr', 1e-6)
        )
        step_frequency = get_scheduler_step_frequency(scheduler_type)

        # Checkpoint Resumption Logic
        resume_epoch = 0
        checkpoint_path = self.run_dir / "checkpoint_last.pth"
        if checkpoint_path.exists():
            print(f"    ↻ Found existing checkpoint. Resuming from {checkpoint_path.name}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if self.scheduler and checkpoint.get("scheduler_state_dict"):
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            if checkpoint.get("precision_state_dict"):
                self.precision.load_state_dict(checkpoint["precision_state_dict"])

            if "callbacks_state_dict" in checkpoint:
                self.cb_runner.load_state_dict(checkpoint["callbacks_state_dict"])

            if "torch_rng_state" in checkpoint:
                torch.set_rng_state(checkpoint["torch_rng_state"])
            if checkpoint.get("cuda_rng_state") and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])

            resume_epoch = checkpoint["epoch"]
            print(f"    ✓ Resumed successfully from epoch {resume_epoch + 1}")

        # Trigger Train Begin Callbacks
        self.cb_runner.on_train_begin(self)

        for epoch in range(resume_epoch, epochs):
            if self.stop_training:
                break

            # Update window indices for sliding window strategy
            train_loader.dataset.set_epoch(epoch)

            self.cb_runner.on_epoch_begin(self, epoch)
            epoch_start_time = time.time()

            # --- Training Phase ---
            self.model.train()
            train_loss, train_correct, train_total, epoch_grad_norm, num_batches = 0.0, 0, 0, 0.0, 0

            for batch_idx, (mel_segments, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False)):
                #if batch_idx == 0:
                #    print(
                #        f"supervised_experiment_train.py train loop "
                #        f"mel_segments: {tuple(mel_segments.shape)}"
                #    )
                #    print(
                #        f"supervised_experiment_train.py train loop "
                #        f"labels: {tuple(labels.shape)}"
                #    )
                batch_start_logs = {"batch": batch_idx}
                self.cb_runner.on_batch_begin(self, batch_idx, batch_start_logs)

                mel_segments, labels = mel_segments.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad(set_to_none=True)

                with self.precision.autocast():
                    logits = self.model(mel_segments)
                    loss = criterion(logits, labels)

                self.precision.scale_loss(loss).backward()
                self.precision.unscale_gradients(self.optimizer)

                grad_clip = self.config["training"].get("gradient_clip")
                batch_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=grad_clip if grad_clip is not None else float('inf')
                ).item()
                epoch_grad_norm += batch_norm
                num_batches += 1

                self.precision.step(self.optimizer)
                self.precision.update()

                if self.scheduler and step_frequency == 'batch':
                    self.scheduler.step()

                batch_loss = loss.item()
                train_loss += batch_loss * labels.size(0)
                preds = torch.argmax(logits, dim=1)
                train_correct += (preds == labels).sum().item()
                train_total += labels.size(0)

                batch_end_logs = {
                    "loss": batch_loss,
                    "grad_norm": batch_norm,
                    "lr": self.optimizer.param_groups[0]["lr"],
                }
                self.cb_runner.on_batch_end(self, batch_idx, batch_end_logs)

                # --- Validation Phase ---
                self.cb_runner.on_validation_begin(self)
                self.model.eval()
                val_loss = 0.0
                val_correct_window = 0
                val_total_windows = 0

                all_logits = []
                all_labels = []
                all_rec_ids = []

                with torch.no_grad():
                    for batch_idx, (mel_segments, labels, recording_ids) in enumerate(
                        tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", leave=False)
                    ):
                        # Debug prints commented out
                        # if batch_idx == 0:
                        #     print(f"validation mel_segments: {tuple(mel_segments.shape)}")
                        #     print(f"validation labels: {tuple(labels.shape)}")
                        #     print(f"validation recording_ids: {recording_ids[:5]}")

                        mel_segments = mel_segments.to(self.device)
                        labels = labels.to(self.device)

                        with self.precision.autocast():
                            logits = self.model(mel_segments)

                        # Window-level metrics
                        loss = criterion(logits, labels)
                        val_loss += loss.item() * labels.size(0)
                        preds = torch.argmax(logits, dim=1)
                        val_correct_window += (preds == labels).sum().item()
                        val_total_windows += labels.size(0)

                        all_logits.append(logits.detach().cpu())
                        all_labels.append(labels.cpu())
                        all_rec_ids.extend(recording_ids)   # recording_ids is a list

                # Concatenate
                all_logits = torch.cat(all_logits, dim=0)
                all_labels = torch.cat(all_labels, dim=0)

                # Aggregate to recording level
                rec_ids, rec_targets, rec_logits = aggregate_recordings(
                    all_logits, all_labels, all_rec_ids
                )

                # Recording-level loss and accuracy
                if len(rec_targets) > 0:
                    rec_loss = criterion(rec_logits.to(self.device), rec_targets.to(self.device))
                    avg_val_loss = rec_loss.item()
                    rec_preds = rec_logits.argmax(dim=1).cpu()
                    val_acc = (rec_preds == rec_targets).float().mean().item()
                else:
                    avg_val_loss = 0.0
                    val_acc = 0.0

            # Window-level accuracy for diagnostics
            val_window_acc = val_correct_window / val_total_windows if val_total_windows > 0 else 0.0

            val_logs = {
                "val_loss": avg_val_loss,           # recording-level loss
                "val_acc": val_acc,                 # recording-level accuracy
                "val_window_acc": val_window_acc,   # window-level accuracy (for reference)
            }
            self.cb_runner.on_validation_end(self, val_logs)

            if self.scheduler and step_frequency == 'epoch':
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(avg_val_loss)
                else:
                    self.scheduler.step()

            # Construct Epoch Metrics Dictionary
            epoch_duration = time.time() - epoch_start_time
            logs = {
                'epoch': epoch + 1,
                'train_loss': train_loss / train_total,
                'train_acc': train_correct / train_total,
                'val_loss': avg_val_loss,
                'val_acc': val_acc,
                'learning_rate': self.optimizer.param_groups[0]["lr"],
                "precision": self.precision.precision_name(),
                "loss_scale": self.precision.current_scale(),
                "grad_norm": epoch_grad_norm / num_batches,
                "epoch_time_sec": epoch_duration,
                "samples_per_sec": train_total / epoch_duration,
            }

            # Memory usage logging
            logs.update(get_gpu_memory_info(self.device))

            print(f"Epoch {epoch+1}/{epochs} | {epoch_duration:.1f}s | "
                  f"Train Loss: {logs['train_loss']:.4f} | Train Acc: {logs['train_acc']:.4f} | "
                  f"Val Loss: {logs['val_loss']:.4f} | Val Acc: {logs['val_acc']:.4f}")

            # Notify Epoch End Callbacks
            self.cb_runner.on_epoch_end(self, epoch, logs)

        # Run Test Evaluation on Best Weights
        best_ckpt = torch.load(self.run_dir / "checkpoint_best.pth", weights_only=False)
        self.model.load_state_dict(best_ckpt["model_state_dict"])
        metrics = self._evaluate(self.model, val_loader, class_names)

        # Trigger Train End Callbacks
        self.cb_runner.on_train_end(self)
        return metrics

    def _evaluate(self, model: nn.Module, test_loader: DataLoader, class_names: list) -> Dict:
        model.eval()
        collector = MetricsCollector(self.run_dir, class_names)

        all_logits = []
        all_labels = []
        all_rec_ids = []

        with torch.no_grad():
            for batch_idx, (mel_segments, labels, recording_ids) in enumerate(
                tqdm(test_loader, desc="Evaluating", leave=False)
            ):
                # Debug prints commented out
                # if batch_idx == 0:
                #     print(f"eval mel_segments: {tuple(mel_segments.shape)}")
                #     print(f"eval labels: {tuple(labels.shape)}")
                #     print(f"eval recording_ids: {recording_ids[:5]}")
                mel_segments = mel_segments.to(self.device)
                with self.precision.autocast():
                    logits = model(mel_segments)

                all_logits.append(logits.detach().cpu())
                all_labels.append(labels.cpu())
                all_rec_ids.extend(recording_ids)

        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        rec_ids, rec_targets, rec_logits = aggregate_recordings(
            all_logits, all_labels, all_rec_ids
        )

        probs = torch.softmax(rec_logits, dim=1).numpy()
        preds = rec_logits.argmax(dim=1).numpy()
        targets = rec_targets.numpy()

        collector.add_batch(preds, targets, probs)
        metrics = collector.compute_metrics()
        collector.save_metrics_json()
        collector.plot_confusion_matrix()
        collector.plot_per_class_metrics()
        collector.generate_markdown_report()
        return metrics
