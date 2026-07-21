# src/training/experiment_train.py

"""Training function adapted for experiments with enhanced logging."""

import json
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import time
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Any, Tuple

from src.data.dataset import BirdSongDataset
from src.models.bird_classifier import BirdClassifier
from src.evaluation.metrics_collector import MetricsCollector
from src.training.scheduler import create_scheduler, get_scheduler_step_frequency
from src.training.precision import PrecisionManager


class ExperimentTrainer:
    """Training orchestrator for experiments with comprehensive logging."""
    def __init__(self, config: Dict[str, Any], run_dir: Path):
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
        
        # Load existing history if resuming, otherwise start fresh
        history_path = self.run_dir / "training_metrics.json"
        if history_path.exists():
            with open(history_path, "r") as f:
                self.training_history = json.load(f)
        else:
            self.training_history = []
    
    
    def get_dataloaders(self, df: pd.DataFrame) -> Tuple[DataLoader, DataLoader, Dict[str, int], Dict[int, str]]:
        """Create train/val dataloaders and label mappings."""
        batch_size = self.config['training']['batch_size']
        num_workers = self.config['training']['num_workers']
        segment_size = self.config['audio']['segment_size']
        min_db = self.config['audio']['min_db']
        max_db = self.config['audio']['max_db']
        
        train_df, test_df = train_test_split(
            df,
            test_size=0.2,
            random_state=42,
            stratify=df['scientific_name_id']
        )
        
        # Create datasets
        train_dataset = BirdSongDataset(
            df=train_df,
            segment_size=segment_size,
            train=True,
            spec_aug_config=self._get_augmentation_config(),
            min_db=min_db,
            max_db=max_db,
        )
        
        test_dataset = BirdSongDataset(
            df=test_df,
            segment_size=segment_size,
            train=False,
            label_to_idx=train_dataset.label_to_idx,            
            min_db=min_db,
            max_db=max_db,
        )
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=self.device.type == "cuda",
            persistent_workers=num_workers > 0,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=self.device.type == "cuda",
            persistent_workers=num_workers > 0,
        )
        
        label_to_idx = train_dataset.label_to_idx
        idx_to_label = train_dataset.idx_to_label
        
        return train_loader, test_loader, label_to_idx, idx_to_label
    
    def _get_augmentation_config(self) -> Dict:
        """Get augmentation config from experiment config."""
        aug_cfg = self.config.get('augmentation', {})
        return {
            'enabled': aug_cfg.get('enabled', True),
            'prob': aug_cfg.get('prob', 0.5),
            'num_freq_masks': aug_cfg.get('num_freq_masks', 2),
            'freq_mask_param': aug_cfg.get('freq_mask_param', 6),
            'num_time_masks': aug_cfg.get('num_time_masks', 2),
            'time_mask_param': aug_cfg.get('time_mask_param', 10),
        }

    def train(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Run full training loop with experiment logging."""
        print(f"\n>>> Training with config:")
        print(f"    Device: {self.device}")
        print(f"    Run dir: {self.run_dir}")

        # Setup dataloaders
        train_loader, test_loader, label_to_idx, idx_to_label = self.get_dataloaders(df)
        class_names = [idx_to_label[i] for i in range(len(idx_to_label))]

        # Initialize model
        model = BirdClassifier(
            n_mels=self.config["audio"]["n_mels"],
            patch_size=self.config["model"]["patch_size"],
            embed_dim=self.config["model"]["embed_dim"],
            num_layers=self.config["model"]["num_layers"],
            heads=self.config["model"]["heads"],
            forward_expansion=self.config["model"]["forward_expansion"],
            dropout=self.config["model"]["dropout"],
            num_classes=self.config["data"]["num_classes"],
            time_steps=self.config["audio"]["segment_size"],
        ).to(self.device)

        # Setup training
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training'].get('weight_decay', 0.01)
        )

        epochs = self.config['training']['epochs']

        # Setup Scheduler
        scheduler_type = self.config['training'].get('scheduler_type', 'cosine')
        warmup_steps = self.config['training'].get('warmup_steps', 0)
        total_steps = len(train_loader) * epochs
    
        scheduler = create_scheduler(
            optimizer=optimizer,
            scheduler_type=scheduler_type,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            min_lr=self.config['training'].get('min_lr', 1e-6)
        )
    
        step_frequency = get_scheduler_step_frequency(scheduler_type)
    
        if scheduler is not None:
            print(f"    Scheduler: {scheduler_type} (warmup: {warmup_steps}, step: {step_frequency})")
        else:
            print(f"    Scheduler: constant LR")

        # Checkpoint Resumption Logic
        start_epoch = 0
        checkpoint_path = self.run_dir / "checkpoint_last.pth"
        
        if checkpoint_path.exists():
            print(f"    ↻ Found existing checkpoint. Resuming from {checkpoint_path.name}")
            # Use map_location to ensure safe loading across different hardware (e.g., CPU to GPU)
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            
            # Restore states
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            
            if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                
            if checkpoint.get("precision_state_dict") is not None:
                self.precision.load_state_dict(checkpoint["precision_state_dict"])
                
            # Restore epoch and best metrics
            start_epoch = checkpoint["epoch"]
            self.best_val_acc = checkpoint.get("val_acc", 0.0)
            self.best_epoch = start_epoch
            
            print(f"    ✓ Fast-forwarding to epoch {start_epoch + 1}. Current best val_acc: {self.best_val_acc:.4f}")

        # Training loop
        for epoch in range(start_epoch, epochs):
            epoch_start_time = time.time()  # START TIMING
            
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            epoch_grad_norm = 0.0  # TRACK GRADIENT NORM
            num_batches = 0

            for mel_segments, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False):
                mel_segments, labels = mel_segments.to(self.device), labels.to(self.device)

                optimizer.zero_grad(set_to_none=True)

                with self.precision.autocast():
                   logits = model(mel_segments)
                   loss = criterion(logits, labels)

                self.precision.scale_loss(loss).backward()
                
                # Unscale explicitly so we can accurately measure (and clip) the gradient norm
                self.precision.unscale_gradients(optimizer)
                
                gradient_clip_val = self.config["training"].get("gradient_clip")
                
                # PyTorch's clip_grad_norm_ returns the total norm, which is perfect for logging
                batch_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 
                    max_norm=gradient_clip_val if gradient_clip_val is not None else float('inf')
                ).item()
                
                epoch_grad_norm += batch_norm
                num_batches += 1

                self.precision.step(optimizer)
                self.precision.update()

                # Step scheduler per batch
                if scheduler is not None and step_frequency == 'batch':
                    scheduler.step()

                train_loss += loss.item() * labels.size(0)
                preds = torch.argmax(logits, dim=1)
                train_correct += (preds == labels).sum().item()
                train_total += labels.size(0)

            avg_train_loss = train_loss / train_total
            train_acc = train_correct / train_total
            avg_grad_norm = epoch_grad_norm / num_batches

            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for mel_segments, labels in tqdm(test_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", leave=False):
                    mel_segments, labels = mel_segments.to(self.device), labels.to(self.device)

                    with self.precision.autocast():
                       logits = model(mel_segments)
                       loss = criterion(logits, labels)

                    val_loss += loss.item() * labels.size(0)
                    preds = torch.argmax(logits, dim=1)
                    val_correct += (preds == labels).sum().item()
                    val_total += labels.size(0)

            avg_val_loss = val_loss / val_total
            val_acc = val_correct / val_total

            # Step scheduler per epoch
            if scheduler is not None and step_frequency == 'epoch':
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(avg_val_loss)
                else:
                    scheduler.step()

            # Calculate Throughput
            epoch_duration = time.time() - epoch_start_time
            throughput = train_total / epoch_duration

            # Log epoch
            epoch_log = {
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'train_acc': train_acc,
                'val_loss': avg_val_loss,
                'val_acc': val_acc,
                'learning_rate': optimizer.param_groups[0]['lr'],  
                "precision": self.precision.precision_name(),
                "loss_scale": self.precision.current_scale(),
                "grad_norm": avg_grad_norm,
                "epoch_time_sec": epoch_duration,
                "samples_per_sec": throughput,
            }
            self.training_history.append(epoch_log)

            print(f"Epoch {epoch+1}/{epochs} | {epoch_duration:.1f}s ({throughput:.1f} seq/s) | "
                  f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f} | "
                  f"Grad Norm: {avg_grad_norm:.2f}")
            
            # TODO: memory logging every few epoch
            if epoch%10==0:
                from src.utils.memory_utils import log_memory_usage
                log_memory_usage(
                    prefix=f"Epoch {epoch+1}",
                    device=self.device,
                )
            
            # Prepare checkpoint package
            checkpoint = {
                "epoch": epoch + 1,
                "val_acc": val_acc,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": (
                    scheduler.state_dict() if scheduler is not None else None
                ),
                "precision_state_dict": self.precision.state_dict(),
            }
            
            # 1. Always save the latest epoch to allow resuming
            torch.save(checkpoint, self.run_dir / "checkpoint_last.pth")
            
            # 2. Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_epoch = epoch + 1
                torch.save(checkpoint, self.run_dir / "checkpoint_best.pth")
                print(f"    ✓ Saved new best model (val_acc: {val_acc:.4f})")

        # Save training history
        self._save_training_history()
        
        # Evaluate on test set with best model 
        best_checkpoint = torch.load(self.run_dir / "checkpoint_best.pth", weights_only=False)
        model.load_state_dict(best_checkpoint["model_state_dict"])
        
        metrics = self._evaluate(model, test_loader, class_names)
        
        # Save metrics
        metrics_path = self.run_dir / "evaluation_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        
        return metrics
    
    def _save_training_history(self):
        """Save epoch-by-epoch training metrics."""
        history_path = self.run_dir / "training_metrics.json"
        with open(history_path, "w") as f:
            json.dump(self.training_history, f, indent=2)
    
    def _evaluate(self, model: nn.Module, test_loader: DataLoader, class_names: list) -> Dict:
        """Evaluate model on test set and collect metrics."""
        model.eval()
        
        metrics_collector = MetricsCollector(self.run_dir, class_names)
        
        with torch.no_grad():
            for mel_segments, labels in tqdm(test_loader, desc="Evaluating", leave=False):
                mel_segments = mel_segments.to(self.device)
                with self.precision.autocast():
                    logits = model(mel_segments)
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)
                
                metrics_collector.add_batch(
                    preds.cpu().numpy(),
                    labels.numpy(),
                    probs.cpu().numpy()
                )
        
        metrics = metrics_collector.compute_metrics()
        metrics_collector.save_metrics_json()
        metrics_collector.plot_confusion_matrix()
        metrics_collector.plot_per_class_metrics()
        metrics_collector.generate_markdown_report()
        
        return metrics
