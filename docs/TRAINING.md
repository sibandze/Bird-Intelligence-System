# Bird Intelligence System

## Training Guide

**Version:** 1.0
**Project:** Bird Intelligence System

---

# Overview

This document describes the complete training workflow used by the Bird Intelligence System. It covers dataset preparation, model training, validation, experiment management, checkpointing, and future extensions.

The training framework has been designed to support both rapid experimentation on small development datasets and large-scale training on datasets such as BirdCLEF.

---

# Training Philosophy

The training system follows several guiding principles:

* Configuration-driven experiments
* Reproducible training
* Modular components
* Clear experiment tracking
* Separation between training and evaluation
* Easy transition from research to production

Every experiment should be reproducible from a single configuration file.

---

# Training Pipeline

```text
Configuration
      │
      ▼
Dataset
      │
      ▼
DataLoader
      │
      ▼
Model Initialization
      │
      ▼
Optimizer
      │
      ▼
Learning Rate Scheduler
      │
      ▼
Training Loop
      │
      ▼
Validation
      │
      ▼
Checkpointing
      │
      ▼
Evaluation
```

---

# Training Workflow

The complete workflow consists of seven stages.

1. Load configuration
2. Build datasets
3. Construct dataloaders
4. Initialize the model
5. Train the model
6. Validate after every epoch
7. Save checkpoints and metrics

---

# Dataset Preparation

Before training begins:

* Audio has already been downloaded.
* Spectrograms have already been generated.
* Metadata has already been created.
* Dataset splits already exist.

Training never performs expensive preprocessing.

---

# Data Loading

Training uses three dataloaders.

```text
Training Loader

Validation Loader

Testing Loader
```

Typical responsibilities include:

* batching
* shuffling
* worker management
* tensor conversion

Only the training loader applies data augmentation.

---

# Model Initialization

The classifier is constructed using the configuration file.

Typical initialization includes:

* embedding dimension
* transformer depth
* attention heads
* dropout
* patch size
* number of output classes

The architecture itself is documented in **ARCHITECTURE.md**.

---

# Optimizer

The baseline model uses **AdamW**.

Reasons for this choice:

* Stable convergence
* Well suited for transformers
* Proper weight decay implementation
* Widely adopted in transformer literature

Future optimizers may include:

* Lion
* SGD
* AdaFactor
* RMSProp

---

# Learning Rate Scheduling

The scheduler controls learning rate throughout training.

Planned schedulers include:

* Cosine Annealing
* Cosine Annealing with Warm Restarts
* Linear Warmup
* Reduce-on-Plateau
* OneCycleLR

The scheduler is fully configurable.

---

# Training Loop

Each epoch follows the same sequence.

```text
Training Epoch

↓

Forward Pass

↓

Loss Computation

↓

Backpropagation

↓

Gradient Update

↓

Validation

↓

Checkpoint
```

The implementation keeps the training loop independent from the model architecture.

---

# Forward Pass

For every batch:

1. Load spectrograms.
2. Transfer tensors to the training device.
3. Execute the forward pass.
4. Compute predictions.
5. Compute loss.

---

# Loss Function

The supervised baseline minimizes categorical cross-entropy.

Future experiments will compare:

* Cross Entropy
* Label Smoothing
* Focal Loss
* Class-Balanced Loss

The loss function is configurable.

---

# Backpropagation

The optimization step follows the standard sequence.

```text
Zero Gradients

↓

Forward Pass

↓

Loss

↓

Backward Pass

↓

Optimizer Step
```

Future versions may include gradient accumulation.

---

# Mixed Precision Training

Automatic Mixed Precision (AMP) is planned for large-scale training.

Advantages include:

* Faster training
* Reduced GPU memory usage
* Larger batch sizes
* Improved throughput

Mixed precision can be enabled or disabled through configuration.

---

# Gradient Clipping

Gradient clipping will be supported to improve stability during transformer training.

Benefits include:

* Prevent exploding gradients
* Improve convergence
* Stabilize deeper architectures

---

# Validation

Validation is performed after every epoch.

Validation never:

* updates gradients
* performs optimization
* applies augmentation

Validation computes:

* loss
* accuracy
* precision
* recall
* macro F1

---

# Early Stopping

Future versions will support early stopping.

Typical workflow:

```text
Validation

↓

Metric Improvement?

↓

Yes → Save Model

↓

No

↓

Increase Patience Counter

↓

Stop if Patience Exceeded
```

This prevents unnecessary overfitting.

---

# Checkpointing

Training periodically saves checkpoints.

Each checkpoint contains:

* model weights
* optimizer state
* scheduler state
* epoch number
* configuration
* best validation score

Training can resume directly from a checkpoint.

---

# Experiment Tracking

Every experiment generates its own directory.

Example:

```text
results/

└── experiment_001/

    ├── config.yaml

    ├── best_model.pth

    ├── last_checkpoint.pth

    ├── training_metrics.json

    ├── evaluation_metrics.json

    ├── confusion_matrix.png

    └── training_curves.png
```

This structure ensures experiments remain isolated and reproducible.

---

# Metrics

Training records metrics after every epoch.

Examples include:

Training

* loss
* accuracy

Validation

* loss
* accuracy
* precision
* recall
* macro F1

Timing

* epoch duration
* total training time

Resource usage

* GPU memory
* learning rate

---

# Hyperparameter Sweeps

The experiment framework supports automated sweeps.

Examples include:

Optimization

* learning rate
* batch size
* optimizer
* scheduler

Architecture

* embedding dimension
* transformer depth
* attention heads
* dropout

Regularization

* weight decay
* SpecAugment strength

The sweep framework executes multiple experiments independently.

---

# Random Seed Management

Every experiment should use fixed random seeds.

Randomness should be controlled for:

* Python
* NumPy
* PyTorch
* CUDA

This improves reproducibility across repeated experiments.

---

# Device Management

Training supports:

* CPU
* Single GPU

Future versions will support:

* Multi-GPU
* Distributed Data Parallel (DDP)
* Multi-node training

---

# Logging

Training should use structured logging rather than print statements.

Logs should include:

* epoch progress
* learning rate
* losses
* validation metrics
* checkpoint events
* training duration

Future integration with MLflow or Weights & Biases is planned.

---

# Failure Recovery

Training should recover gracefully from interruptions.

Supported recovery features include:

* resume from checkpoint
* restore optimizer state
* restore scheduler state
* continue epoch numbering

This is especially important for long-running BirdCLEF experiments.

---

# Configuration

Training behavior is controlled entirely through the configuration file.

Examples include:

```yaml
training:

    batch_size: 32

    epochs: 100

    learning_rate: 0.0003

    weight_decay: 0.01

    mixed_precision: true

    gradient_clip: 1.0
```

Adding new training options should not require changes to the training loop.

---

# Future Training Features

Planned improvements include:

## Optimization

* Gradient accumulation
* Exponential Moving Average (EMA)
* Sharpness-Aware Minimization (SAM)
* Stochastic Weight Averaging (SWA)

---

## Efficiency

* FlashAttention
* Fused optimizers
* Torch Compile
* Asynchronous data loading
* Memory-mapped datasets

---

## Experiment Management

* MLflow integration
* Weights & Biases integration
* Automatic artifact versioning
* Experiment comparison dashboard

---

## Distributed Training

* Multi-GPU support
* Distributed checkpointing
* Gradient synchronization
* Elastic training

---

# Best Practices

Recommended workflow:

1. Verify dataset integrity.
2. Run development experiments.
3. Tune hyperparameters on the development dataset.
4. Train on the full dataset.
5. Evaluate using the test set only once.
6. Archive configuration and checkpoints.
7. Record experimental observations before starting the next run.

---

# Summary

The Bird Intelligence System training framework is designed to be modular, reproducible, and scalable. By separating preprocessing, model definition, training, evaluation, and experiment management, the framework supports rapid research iteration while remaining suitable for large-scale transformer training on bioacoustic datasets.

As the project evolves, the same training infrastructure will support supervised learning, self-supervised pre-training, transfer learning, and future production deployments with minimal architectural changes.
