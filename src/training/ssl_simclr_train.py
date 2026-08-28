# src/training/ssl_simclr_train.py

import time
from pathlib import Path
from typing import Dict, Any, Optional, List

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from src.data.datasets.ssl import (
    SimCLRDataset,
    simclr_collate_fn,
)
from src.models.ssl.simclr import SimCLR
from src.models.encoders import CNNEncoder
from src.models.heads import ProjectionHead

from src.training.precision import PrecisionManager
from src.training.scheduler import (
    create_scheduler,
    get_scheduler_step_frequency,
)

from src.utils.memory_utils import get_gpu_memory_info

from src.training.callbacks import (
    Callback,
    CallbackRunner,
    CheckpointCallback,
    EarlyStoppingCallback,
    JSONLoggerCallback,
    CSVLoggerCallback,
    WandBLoggerCallback,
)


class SimCLRExperimentTrainer:
    """
    Trainer for self-supervised contrastive learning with SimCLR.

    Pipeline:

        spectrogram
            ↓
        two independently augmented views
            ↓
        shared CNN encoder
            ↓
        projection head
            ↓
        normalized embeddings
            ↓
        NT-Xent / InfoNCE loss

    The encoder can later be extracted using:

        trainer.model.encode(x)

    for downstream linear probing or fine-tuning.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        run_dir: Path,
        callbacks: Optional[List[Callback]] = None,
    ):
        self.config = config
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(exist_ok=True, parents=True)

        # -------------------------------------------------------------
        # Device
        # -------------------------------------------------------------

        requested_device = config["training"].get("device", "cuda")

        if requested_device == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # -------------------------------------------------------------
        # Precision
        # -------------------------------------------------------------

        mixed_precision_cfg = config["training"].get(
            "mixed_precision",
            {},
        )

        self.precision = PrecisionManager(
            enabled=mixed_precision_cfg.get("enabled", True),
            device=self.device.type,
            use_bfloat16=mixed_precision_cfg.get(
                "use_bfloat16",
                False,
            ),
        )

        # -------------------------------------------------------------
        # Trainer state
        # -------------------------------------------------------------

        self.model = None
        self.optimizer = None
        self.scheduler = None

        self.best_loss = float("inf")
        self.best_epoch = 0
        self.best_val_acc = 0.0

        self.stop_training = False

        # -------------------------------------------------------------
        # Callbacks
        # -------------------------------------------------------------

        if callbacks is None:
            callbacks = [
                CheckpointCallback(
                    self.run_dir,
                    monitor="val_loss",
                    mode="min",
                ),
                EarlyStoppingCallback(
                    monitor="val_loss",
                    mode="min",
                    patience=config["training"].get(
                        "patience",
                        15,
                    ),
                ),
                JSONLoggerCallback(self.run_dir),
                CSVLoggerCallback(self.run_dir),
                WandBLoggerCallback(
                    config,
                    self.run_dir,
                ),
            ]

        self.cb_runner = CallbackRunner(callbacks)

    # =================================================================
    # Control
    # =================================================================

    def request_stop(self):
        """Request training to stop after the current epoch."""
        self.stop_training = True

    # =================================================================
    # Data
    # =================================================================

    def get_dataloaders(self, df):
        """
        Build train and validation dataloaders.

        SSL does not require labels.

        Important:
            The train/validation split is performed BEFORE temporal
            windowing, so recordings do not get split into both sets.
        """

        batch_size = self.config["training"]["batch_size"]
        num_workers = self.config["training"]["num_workers"]

        segment_size = self.config["audio"]["segment_size"]

        window_config = self.config.get(
            "window",
            {},
        )

        # -------------------------------------------------------------
        # Recording-level split
        # -------------------------------------------------------------

        train_df, val_df = train_test_split(
            df,
            test_size=0.05,
            random_state=42,
            shuffle=True,
        )

        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)

        # -------------------------------------------------------------
        # Training dataset
        # -------------------------------------------------------------

        train_dataset = SimCLRDataset(
            df=train_df,
            segment_size=segment_size,
            min_db=self.config["audio"]["min_db"],
            max_db=self.config["audio"]["max_db"],
            train=True,

            # SSL augmentation is controlled by apply_augmentation.
            apply_augmentation=True,

            window_config=window_config,

            acoustic_aug_config=self.config.get(
                "acoustic_augmentation",
                {},
            ),

            spec_aug_config=self._get_augmentation_config(),
        )

        # -------------------------------------------------------------
        # Validation dataset
        # -------------------------------------------------------------

        val_dataset = SimCLRDataset(
            df=val_df,
            segment_size=segment_size,
            min_db=self.config["audio"]["min_db"],
            max_db=self.config["audio"]["max_db"],
            train=False,

            # IMPORTANT:
            # Validation should not receive stochastic augmentations.
            apply_augmentation=False,

            # Evaluate every deterministic window.
            window_config={
                "strategy": "sliding",
                "stride": segment_size,
            },

            acoustic_aug_config=self.config.get(
                "acoustic_augmentation",
                {},
            ),

            spec_aug_config=self._get_augmentation_config(),
        )

        # -------------------------------------------------------------
        # DataLoaders
        # -------------------------------------------------------------

        loader_kwargs = {
            "batch_size": batch_size,
            "num_workers": num_workers,
            "pin_memory": self.device.type == "cuda",
            "persistent_workers": num_workers > 0,
            "collate_fn": simclr_collate_fn,
        }

        train_loader = DataLoader(
            train_dataset,
            shuffle=True,
            **loader_kwargs,
        )

        val_loader = DataLoader(
            val_dataset,
            shuffle=False,
            **loader_kwargs,
        )

        return train_loader, val_loader

    # =================================================================
    # Augmentation
    # =================================================================

    def _get_augmentation_config(self):
        """
        Extract spectrogram augmentation configuration.
        """

        aug_cfg = self.config.get(
            "augmentation",
            {},
        )

        return {
            "enabled": aug_cfg.get(
                "enabled",
                True,
            ),
            "prob": aug_cfg.get(
                "prob",
                0.5,
            ),
            "num_freq_masks": aug_cfg.get(
                "num_freq_masks",
                2,
            ),
            "freq_mask_param": aug_cfg.get(
                "freq_mask_param",
                6,
            ),
            "num_time_masks": aug_cfg.get(
                "num_time_masks",
                2,
            ),
            "time_mask_param": aug_cfg.get(
                "time_mask_param",
                10,
            ),
        }

    # =================================================================
    # Model
    # =================================================================

    def _build_model(self):
        """
        Build CNN encoder + projection head + SimCLR.
        """

        encoder_cfg = self.config["model"]
        projection_cfg = self.config.get(
            "projection",
            {},
        )

        encoder = CNNEncoder(
            n_mels=self.config["audio"]["n_mels"],
            embed_dim=encoder_cfg.get(
                "embed_dim",
                512,
            ),
            base_channels=encoder_cfg.get(
                "base_channels",
                64,
            ),
            dropout=encoder_cfg.get(
                "dropout",
                0.1,
            ),
        )

        projection = ProjectionHead(
            input_dim=encoder.get_output_dim(),
            hidden_dim=projection_cfg.get(
                "hidden_dim",
                256,
            ),
            output_dim=projection_cfg.get(
                "output_dim",
                128,
            ),
        )

        model = SimCLR(
            encoder=encoder,
            projection=projection,
            temperature=self.config.get(
                "temperature",
                0.07,
            ),
        )

        return model.to(self.device)

    # =================================================================
    # Checkpoint resume
    # =================================================================

    def _resume_from_checkpoint(
        self,
        checkpoint_path: Path,
    ) -> int:
        """
        Restore model, optimizer, scheduler, precision,
        callbacks and RNG state.

        Returns:
            Epoch from which training should continue.
        """

        print(
            f"    ↻ Resuming from "
            f"{checkpoint_path.name}"
        )

        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

        if (
            self.scheduler is not None
            and checkpoint.get("scheduler_state_dict") is not None
        ):
            self.scheduler.load_state_dict(
                checkpoint["scheduler_state_dict"]
            )

        if checkpoint.get("precision_state_dict"):
            self.precision.load_state_dict(
                checkpoint["precision_state_dict"]
            )

        if checkpoint.get("callbacks_state_dict"):
            self.cb_runner.load_state_dict(
                checkpoint["callbacks_state_dict"]
            )

        # -------------------------------------------------------------
        # Restore RNG state
        # -------------------------------------------------------------

        if checkpoint.get("torch_rng_state") is not None:
            torch.set_rng_state(
                checkpoint["torch_rng_state"]
            )

        if (
            torch.cuda.is_available()
            and checkpoint.get("cuda_rng_state") is not None
        ):
            torch.cuda.set_rng_state_all(
                checkpoint["cuda_rng_state"]
            )

        resume_epoch = checkpoint.get(
            "epoch",
            0,
        )

        print(
            f"    ↻ Continuing from epoch "
            f"{resume_epoch + 1}"
        )

        return resume_epoch

    # =================================================================
    # Training
    # =================================================================

    def train(self, df):
        """
        Run complete SimCLR training experiment.
        """

        # -------------------------------------------------------------
        # Data
        # -------------------------------------------------------------

        train_loader, val_loader = self.get_dataloaders(df)

        # -------------------------------------------------------------
        # Model
        # -------------------------------------------------------------

        self.model = self._build_model()

        # -------------------------------------------------------------
        # Parameter count
        # -------------------------------------------------------------

        num_params = sum(
            p.numel()
            for p in self.model.parameters()
            if p.requires_grad
        )

        print("\n>>> Initializing SimCLR Training:")
        print(f"    Device: {self.device}")
        print(
            f"    Precision: "
            f"{self.precision.precision_name()}"
        )
        print(
            f"    Trainable params: "
            f"{num_params:,}"
        )
        print(
            f"    Train windows: "
            f"{len(train_loader.dataset):,}"
        )
        print(
            f"    Val windows: "
            f"{len(val_loader.dataset):,}"
        )

        # -------------------------------------------------------------
        # Optimizer
        # -------------------------------------------------------------

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config["training"]["learning_rate"],
            weight_decay=self.config["training"].get(
                "weight_decay",
                1e-4,
            ),
        )

        # -------------------------------------------------------------
        # Scheduler
        # -------------------------------------------------------------

        epochs = self.config["training"]["epochs"]

        scheduler_type = self.config["training"].get(
            "scheduler_type",
            "cosine",
        )

        warmup_steps = self.config["training"].get(
            "warmup_steps",
            0,
        )

        total_steps = len(train_loader) * epochs

        self.scheduler = create_scheduler(
            optimizer=self.optimizer,
            scheduler_type=scheduler_type,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            min_lr=self.config["training"].get(
                "min_lr",
                1e-6,
            ),
        )

        step_frequency = get_scheduler_step_frequency(
            scheduler_type
        )

        # -------------------------------------------------------------
        # Resume
        # -------------------------------------------------------------

        resume_epoch = 0

        checkpoint_path = (
            self.run_dir /
            "checkpoint_last.pth"
        )

        if checkpoint_path.exists():
            resume_epoch = self._resume_from_checkpoint(
                checkpoint_path
            )

        # -------------------------------------------------------------
        # Training start
        # -------------------------------------------------------------

        self.cb_runner.on_train_begin(self)

        # Protect against empty datasets.
        if len(train_loader.dataset) == 0:
            raise ValueError(
                "Training dataset contains zero windows."
            )

        if len(val_loader.dataset) == 0:
            raise ValueError(
                "Validation dataset contains zero windows."
            )

        # -------------------------------------------------------------
        # Epoch loop
        # -------------------------------------------------------------

        for epoch in range(
            resume_epoch,
            epochs,
        ):

            if self.stop_training:
                break

            # ---------------------------------------------------------
            # Update temporal window sampling
            # ---------------------------------------------------------

            train_loader.dataset.set_epoch(epoch)

            self.cb_runner.on_epoch_begin(
                self,
                epoch,
            )

            epoch_start = time.time()

            # =========================================================
            # TRAIN
            # =========================================================

            self.model.train()

            train_loss_total = 0.0
            train_acc_total = 0.0
            train_samples = 0
            grad_norm_total = 0.0
            num_batches = 0

            train_pbar = tqdm(
                train_loader,
                desc=(
                    f"Epoch {epoch + 1}/{epochs} "
                    "[Train]"
                ),
                leave=False,
            )

            for batch_idx, (x1, x2) in enumerate(
                train_pbar
            ):

                self.cb_runner.on_batch_begin(
                    self,
                    batch_idx,
                    {},
                )

                x1 = x1.to(
                    self.device,
                    non_blocking=True,
                )

                x2 = x2.to(
                    self.device,
                    non_blocking=True,
                )

                self.optimizer.zero_grad(
                    set_to_none=True
                )

                # -----------------------------------------------------
                # Forward + contrastive loss
                # -----------------------------------------------------

                with self.precision.autocast():

                    loss, acc = (
                        self.model.training_step(
                            x1,
                            x2,
                        )
                    )

                # -----------------------------------------------------
                # Backward
                # -----------------------------------------------------

                self.precision.scale_loss(
                    loss
                ).backward()

                self.precision.unscale_gradients(
                    self.optimizer
                )

                # -----------------------------------------------------
                # Gradient clipping
                # -----------------------------------------------------

                grad_clip = self.config["training"].get(
                    "gradient_clip"
                )

                grad_norm = (
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=(
                            grad_clip
                            if grad_clip is not None
                            else float("inf")
                        ),
                    ).item()
                )

                # -----------------------------------------------------
                # Optimizer
                # -----------------------------------------------------

                self.precision.step(
                    self.optimizer
                )

                self.precision.update()

                # -----------------------------------------------------
                # Scheduler
                # -----------------------------------------------------

                if (
                    self.scheduler is not None
                    and step_frequency == "batch"
                ):
                    self.scheduler.step()

                # -----------------------------------------------------
                # Metrics
                # -----------------------------------------------------

                batch_size = x1.size(0)

                train_loss_total += (
                    loss.item() * batch_size
                )

                train_acc_total += (
                    acc.item() * batch_size
                )

                train_samples += batch_size

                grad_norm_total += grad_norm
                num_batches += 1

                batch_logs = {
                    "loss": loss.item(),
                    "contrastive_acc": acc.item(),
                    "grad_norm": grad_norm,
                }

                self.cb_runner.on_batch_end(
                    self,
                    batch_idx,
                    batch_logs,
                )

                train_pbar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    acc=f"{acc.item():.4f}",
                )

            # ---------------------------------------------------------
            # Train averages
            # ---------------------------------------------------------

            avg_train_loss = (
                train_loss_total /
                train_samples
            )

            avg_train_acc = (
                train_acc_total /
                train_samples
            )

            # =========================================================
            # VALIDATION
            # =========================================================

            self.cb_runner.on_validation_begin(
                self
            )

            self.model.eval()

            val_loss_total = 0.0
            val_acc_total = 0.0
            val_samples = 0

            val_pbar = tqdm(
                val_loader,
                desc=(
                    f"Epoch {epoch + 1}/{epochs} "
                    "[Val]"
                ),
                leave=False,
            )

            with torch.no_grad():

                for x1, x2 in val_pbar:

                    x1 = x1.to(
                        self.device,
                        non_blocking=True,
                    )

                    x2 = x2.to(
                        self.device,
                        non_blocking=True,
                    )

                    with self.precision.autocast():

                        loss, acc = (
                            self.model.training_step(
                                x1,
                                x2,
                            )
                        )

                    batch_size = x1.size(0)

                    val_loss_total += (
                        loss.item() * batch_size
                    )

                    val_acc_total += (
                        acc.item() * batch_size
                    )

                    val_samples += batch_size

                    val_pbar.set_postfix(
                        loss=f"{loss.item():.4f}",
                        acc=f"{acc.item():.4f}",
                    )

            # ---------------------------------------------------------
            # Validation averages
            # ---------------------------------------------------------

            avg_val_loss = (
                val_loss_total /
                val_samples
            )

            avg_val_acc = (
                val_acc_total /
                val_samples
            )

            validation_logs = {
                "val_loss": avg_val_loss,
                "val_contrastive_acc": avg_val_acc,
            }

            self.cb_runner.on_validation_end(
                self,
                validation_logs,
            )

            # ---------------------------------------------------------
            # Epoch scheduler
            # ---------------------------------------------------------

            if (
                self.scheduler is not None
                and step_frequency == "epoch"
            ):
                self.scheduler.step(
                    avg_val_loss
                )

            # =========================================================
            # LOGGING
            # =========================================================

            epoch_duration = (
                time.time() -
                epoch_start
            )

            current_lr = (
                self.optimizer.param_groups[0]["lr"]
            )

            logs = {
                "epoch": epoch + 1,

                "train_loss": avg_train_loss,
                "train_contrastive_acc": avg_train_acc,

                "val_loss": avg_val_loss,
                "val_contrastive_acc": avg_val_acc,

                "learning_rate": current_lr,

                "grad_norm": (
                    grad_norm_total /
                    num_batches
                    if num_batches > 0
                    else 0.0
                ),

                "epoch_time_sec": epoch_duration,
            }

            logs.update(
                get_gpu_memory_info(
                    self.device
                )
            )

            # Keep trainer-level best metrics available
            # to callbacks.
            if avg_val_loss < self.best_loss:
                self.best_loss = avg_val_loss
                self.best_epoch = epoch + 1

            if avg_val_acc > self.best_val_acc:
                self.best_val_acc = avg_val_acc

            # ---------------------------------------------------------
            # Console output
            # ---------------------------------------------------------

            print(
                f"Epoch {epoch + 1}/{epochs} | "
                f"{epoch_duration:.1f}s | "
                f"Train Loss: "
                f"{avg_train_loss:.4f} | "
                f"Train Contrastive Acc: "
                f"{avg_train_acc:.4f} | "
                f"Val Loss: "
                f"{avg_val_loss:.4f} | "
                f"Val Contrastive Acc: "
                f"{avg_val_acc:.4f} | "
                f"LR: "
                f"{current_lr:.2e}"
            )

            # ---------------------------------------------------------
            # Callbacks
            # ---------------------------------------------------------

            self.cb_runner.on_epoch_end(
                self,
                epoch,
                logs,
            )

        # =============================================================
        # END
        # =============================================================

        self.cb_runner.on_train_end(
            self
        )

        print(
            "\n>>> SimCLR training complete."
        )

        print(
            f"    Best validation loss: "
            f"{self.best_loss:.6f}"
        )

        print(
            f"    Best epoch: "
            f"{self.best_epoch}"
        )

        print(
            f"    Best contrastive accuracy: "
            f"{self.best_val_acc:.4f}"
        )

        # -------------------------------------------------------------
        # Experiment runner-compatible return
        # -------------------------------------------------------------

        return {
            "val_loss": avg_val_loss,
            "val_acc": avg_val_acc,
            "train_loss": avg_train_loss,
            "train_acc": avg_train_acc,
        }
