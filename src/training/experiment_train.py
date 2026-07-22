# src/training/experiment_train.py

import subprocess
import time
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional  # 1. Added Optional

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import BirdSongDataset
from src.evaluation.metrics_collector import MetricsCollector
from src.models.bird_classifier import BirdClassifier
from src.training.precision import PrecisionManager
from src.training.scheduler import create_scheduler, get_scheduler_step_frequency
from src.utils.memory_utils import get_gpu_memory_info, log_memory_usage  # 10. Memory helper import
from src.training.callbacks import (
    Callback, CallbackRunner, CheckpointCallback, EarlyStoppingCallback,
    JSONLoggerCallback, CSVLoggerCallback, WandBLoggerCallback
)


class ExperimentTrainer:
    """Fully modular, callback-driven training engine."""
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
        self.stop_training = False  # Controlled via request_stop()
        
        # 8. Note: Callback order matters! EarlyStopping and Checkpoint come before Loggers.
        if callbacks is None:
            callbacks = [
                CheckpointCallback(self.run_dir, monitor="val_acc", mode="max"),
                EarlyStoppingCallback(monitor="val_acc", mode="max", patience=config["training"].get("patience", 15)),
                JSONLoggerCallback(self.run_dir),
                CSVLoggerCallback(self.run_dir),
                WandBLoggerCallback(config, self.run_dir)
            ]
        self.cb_runner = CallbackRunner(callbacks)

        # 6. Store environment state
        try:
            self.git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
        except Exception:
            self.git_hash = None

    def request_stop(self):
        """5. Clean external API for callbacks to trigger early stopping."""
        self.stop_training = True

    def get_dataloaders(self, df: pd.DataFrame) -> Tuple[DataLoader, DataLoader, Dict[str, int], Dict[int, str]]:
        batch_size = self.config['training']['batch_size']
        num_workers = self.config['training']['num_workers']
        segment_size = (self.config['audio']['sr'] * self.config['audio']['segment_seconds']) // self.config['audio']['hop_length']
        
        train_df, test_df = train_test_split(
            df, test_size=0.2, random_state=42, stratify=df['scientific_name_id']
        )
        
        train_dataset = BirdSongDataset(
            df=train_df, segment_size=segment_size, train=True,
            spec_aug_config=self._get_augmentation_config(),
            min_db=self.config['audio']['min_db'], max_db=self.config['audio']['max_db']
        )
        test_dataset = BirdSongDataset(
            df=test_df, segment_size=segment_size, train=False,
            label_to_idx=train_dataset.label_to_idx,
            min_db=self.config['audio']['min_db'], max_db=self.config['audio']['max_db']
        )
        
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=(self.device.type == "cuda"), persistent_workers=num_workers > 0
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=(self.device.type == "cuda"), persistent_workers=num_workers > 0
        )
        return train_loader, test_loader, train_dataset.label_to_idx, train_dataset.idx_to_label

    def _get_augmentation_config(self) -> Dict:
        aug_cfg = self.config.get('augmentation', {})
        return {
            'enabled': aug_cfg.get('enabled', True), 'prob': aug_cfg.get('prob', 0.5),
            'num_freq_masks': aug_cfg.get('num_freq_masks', 2), 'freq_mask_param': aug_cfg.get('freq_mask_param', 6),
            'num_time_masks': aug_cfg.get('num_time_masks', 2), 'time_mask_param': aug_cfg.get('time_mask_param', 10),
        }

    def train(self, df: pd.DataFrame) -> Dict[str, Any]:
        train_loader, test_loader, label_to_idx, idx_to_label = self.get_dataloaders(df)
        class_names = [idx_to_label[i] for i in range(len(idx_to_label))]
        
        segment_size = (self.config['audio']['sr'] * self.config['audio']['segment_seconds']) // self.config['audio']['hop_length']

        # Initialize Model
        self.model = BirdClassifier(
            n_mels=self.config["audio"]["n_mels"],
            patch_size=self.config["model"]["patch_size"],
            embed_dim=self.config["model"]["embed_dim"],
            num_layers=self.config["model"]["num_layers"],
            heads=self.config["model"]["heads"],
            forward_expansion=self.config["model"]["forward_expansion"],
            dropout=self.config["model"]["dropout"],
            num_classes=self.config["data"]["num_classes"],
            time_steps=segment_size,
        ).to(self.device)

        # 7. Print Model and Environment Summary once
        num_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        compiled = self.config["training"].get("compile_model", False)

        print(f"\n>>> Initializing Training Run:")
        print(f"    Device:    {self.device}")
        print(f"    Precision: {self.precision.precision_name()}")
        print(f"    Compiled:  {compiled}")
        print(f"    Params:    {num_params:,} (Trainable: {trainable_params:,})")

        if compiled:
            self.model = torch.compile(self.model)
            # TODO:
            # Benchmark torch.compile() separately.
            # Compilation introduces startup overhead and is beneficial
            # primarily for long training runs.
        
        criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training'].get('weight_decay', 0.01)
        )

        epochs = self.config['training']['epochs']
        scheduler_type = self.config['training'].get('scheduler_type', 'cosine')
        warmup_steps = self.config['training'].get('warmup_steps', 0)
        
        self.scheduler = create_scheduler(
            optimizer=self.optimizer, scheduler_type=scheduler_type,
            warmup_steps=warmup_steps, total_steps=len(train_loader) * epochs,
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
            
            # 4. Restore Callback States
            if "callbacks_state_dict" in checkpoint:
                self.cb_runner.load_state_dict(checkpoint["callbacks_state_dict"])
                
            if "torch_rng_state" in checkpoint: torch.set_rng_state(checkpoint["torch_rng_state"])
            if checkpoint.get("cuda_rng_state") and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])
                
            resume_epoch = checkpoint["epoch"]
            print(f"    ✓ Resumed successfully from epoch {resume_epoch + 1}")

        # Trigger Train Begin Callbacks
        self.cb_runner.on_train_begin(self)

        for epoch in range(resume_epoch, epochs):
            if self.stop_training:
                break
                
            self.cb_runner.on_epoch_begin(self, epoch)
            epoch_start_time = time.time()

            # --- Training Phase ---
            self.model.train()
            train_loss, train_correct, train_total, epoch_grad_norm, num_batches = 0.0, 0, 0, 0.0, 0

            for batch_idx, (mel_segments, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False)):
                # 2. Fire on_batch_begin Hook
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

                # 3. Fire on_batch_end Hook
                batch_end_logs = {
                    "loss": batch_loss,
                    "grad_norm": batch_norm,
                    "lr": self.optimizer.param_groups[0]["lr"],
                }
                self.cb_runner.on_batch_end(self, batch_idx, batch_end_logs)

            # --- Validation Phase ---
            # 9. Trigger on_validation_begin Hook
            self.cb_runner.on_validation_begin(self)
            self.model.eval()
            val_loss, val_correct, val_total = 0.0, 0, 0
            
            with torch.no_grad():
                for mel_segments, labels in tqdm(test_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", leave=False):
                    mel_segments, labels = mel_segments.to(self.device), labels.to(self.device)
                    with self.precision.autocast():
                        logits = self.model(mel_segments)
                        loss = criterion(logits, labels)

                    val_loss += loss.item() * labels.size(0)
                    preds = torch.argmax(logits, dim=1)
                    val_correct += (preds == labels).sum().item()
                    val_total += labels.size(0)

            avg_val_loss = val_loss / val_total
            avg_val_acc = val_correct / val_total

            # 9. Trigger on_validation_end Hook
            val_logs = {"val_loss": avg_val_loss, "val_acc": avg_val_acc}
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
                'val_acc': avg_val_acc,
                'learning_rate': self.optimizer.param_groups[0]["lr"],
                "precision": self.precision.precision_name(),
                "loss_scale": self.precision.current_scale(),
                "grad_norm": epoch_grad_norm / num_batches,
                "epoch_time_sec": epoch_duration,
                "samples_per_sec": train_total / epoch_duration,
            }
            
            # 10. Memory helper integration
            logs.update(get_gpu_memory_info(self.device))

            print(f"Epoch {epoch+1}/{epochs} | {epoch_duration:.1f}s | "
                  f"Train Loss: {logs['train_loss']:.4f} | Train Acc: {logs['train_acc']:.4f} | "
                  f"Val Loss: {logs['val_loss']:.4f} | Val Acc: {logs['val_acc']:.4f}")

            # Notify Epoch End Callbacks
            self.cb_runner.on_epoch_end(self, epoch, logs)

        # Run Test Evaluation on Best Weights
        best_ckpt = torch.load(self.run_dir / "checkpoint_best.pth", weights_only=False)
        self.model.load_state_dict(best_ckpt["model_state_dict"])
        metrics = self._evaluate(self.model, test_loader, class_names)

        # Trigger Train End Callbacks
        self.cb_runner.on_train_end(self)
        return metrics

    def _evaluate(self, model: nn.Module, test_loader: DataLoader, class_names: list) -> Dict:
        model.eval()
        collector = MetricsCollector(self.run_dir, class_names)
        with torch.no_grad():
            for mel_segments, labels in tqdm(test_loader, desc="Evaluating", leave=False):
                mel_segments = mel_segments.to(self.device)
                with self.precision.autocast():
                    logits = model(mel_segments)
                    probs = torch.softmax(logits, dim=1)
                    preds = torch.argmax(logits, dim=1)
                collector.add_batch(preds.cpu().numpy(), labels.numpy(), probs.cpu().numpy())
        
        metrics = collector.compute_metrics()
        collector.save_metrics_json()
        collector.plot_confusion_matrix()
        collector.plot_per_class_metrics()
        collector.generate_markdown_report()
        return metrics
