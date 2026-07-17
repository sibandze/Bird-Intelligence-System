# DATASET.md

# Bird Intelligence System

## Dataset Documentation

**Version:** 1.0
**Project:** Bird Intelligence System

---

# Overview

The Bird Intelligence System is designed to work with multiple bird audio datasets through a unified preprocessing pipeline. Regardless of the original source, all datasets are converted into a common internal representation before training.

The primary objective is to ensure that the model is completely independent of any particular dataset format. Once data has been processed, every training sample follows the same structure, allowing experiments to scale from small development datasets to large collections such as BirdCLEF without changing the training code.

---

# Design Philosophy

The dataset pipeline follows several guiding principles:

* **Dataset agnostic:** Any bird audio dataset can be integrated after conversion.
* **Offline preprocessing:** Audio is converted to spectrograms once rather than during every training epoch.
* **Reproducibility:** Identical inputs always produce identical processed outputs.
* **Scalability:** Support datasets ranging from a few hundred recordings to millions of audio clips.
* **Extensibility:** New datasets should only require a parser, not changes to the learning pipeline.

---

# Supported Datasets

## Current

* Development Dataset (Birds Voices Dataset)

## Planned

* BirdCLEF 2023
* BirdCLEF 2024
* Xeno-canto
* Macaulay Library
* Custom user datasets

Future datasets should be converted into the unified metadata format described below.

---

# Data Pipeline

```text
Raw Audio
    │
    ▼
Metadata Parsing
    │
    ▼
Audio Validation
    │
    ▼
Resampling
    │
    ▼
Mel Spectrogram Generation
    │
    ▼
NumPy Storage
    │
    ▼
Metadata Generation
    │
    ▼
PyTorch Dataset
```

---

# Directory Structure

```text
data/

├── raw_audio/
│
├── processed_spectrograms/
│
├── metadata/
│
└── birds_voices.csv
```

---

# Raw Audio

The `raw_audio/` directory stores downloaded recordings before preprocessing.

Supported formats include

* WAV
* MP3
* OGG
* FLAC
* M4A (planned)

The preprocessing pipeline converts every recording into a common format before feature extraction.

---

# Audio Standardization

Every recording is standardized before feature extraction.

Typical preprocessing steps include

1. Audio loading
2. Mono conversion
3. Resampling
4. Normalization
5. Silence trimming (optional)
6. Duration adjustment
7. Spectrogram generation

This guarantees consistent inputs regardless of recording source.

---

# Feature Extraction

Each audio recording is converted into a mel spectrogram.

Typical configuration

```yaml
audio:

    sr: 32000

    n_fft: 2048

    hop_length: 512

    n_mels: 128

    segment_seconds: 6.0
```

The configuration loader computes

```text
segment_size

segment_seconds_actual

n_frames_raw
```

automatically to ensure compatibility with the transformer patch size.

---

# Spectrogram Storage

Instead of computing spectrograms during every epoch, they are generated once and stored as NumPy arrays.

Advantages

* Faster training
* Reduced CPU usage
* Deterministic preprocessing
* Easier debugging
* Reproducible experiments

Each processed file contains

```text
shape = (128, T)
dtype = float32
```

where

* 128 is the number of mel bins
* T is the patch-aligned number of time frames

---

# Metadata

Every processed spectrogram has a corresponding metadata entry.

Example

| Column         | Description                   |
| -------------- | ----------------------------- |
| sample_id      | Unique sample identifier      |
| species        | Bird species label            |
| file_path      | Path to processed spectrogram |
| original_audio | Original recording            |
| duration       | Audio duration (seconds)      |
| split          | train / validation / test     |

Additional fields may be added for future datasets.

Examples include

* latitude
* longitude
* country
* elevation
* recording quality
* recorder
* source dataset
* background species

---

# Dataset Splits

The pipeline produces three datasets.

```text
Training

Validation

Testing
```

Default split

```text
70%

15%

15%
```

The split is performed once during preprocessing and saved in the metadata to ensure reproducibility.

---

# Development Dataset

The default configuration uses a small subset for rapid experimentation.

Typical configuration

```yaml
num_classes: 5

num_samples_per_class: 50
```

Advantages

* Fast preprocessing
* Fast training
* Rapid debugging
* Hyperparameter tuning

---

# Production Dataset

The production pipeline removes development restrictions.

Characteristics

* Hundreds of species
* Thousands of recordings
* Multiple datasets
* Class imbalance
* No artificial sampling limits

---

# Data Loading

The PyTorch Dataset is intentionally lightweight.

Responsibilities

* Load NumPy arrays
* Apply augmentations
* Encode labels
* Return tensors

Heavy computation is intentionally performed offline.

---

# Data Augmentation

Training-only augmentations currently include

* SpecAugment
* Time masking
* Frequency masking

Planned augmentations

* Pitch shifting
* Time stretching
* Gaussian noise
* Background noise mixing
* Reverberation
* Random gain

Validation and testing datasets are never augmented.

---

# Label Encoding

Species names are converted into integer labels.

Example

```text
Robin      → 0

Sparrow    → 1

Canary     → 2

...
```

The mapping is stored to ensure consistent predictions across experiments.

---

# Class Imbalance

Bird datasets naturally exhibit long-tail distributions.

Planned mitigation strategies include

* Weighted sampling
* Class-balanced loss
* Minimum sample thresholds
* Oversampling
* Focal loss experiments

The baseline implementation uses standard cross-entropy to establish a reference point before introducing imbalance-aware methods.

---

# Dataset Validation

Before preprocessing, every recording is validated.

Validation checks include

* File exists
* Audio can be decoded
* Correct sample rate after resampling
* Valid duration
* No corrupted files
* Valid metadata

Invalid recordings are skipped and logged.

---

# BirdCLEF Support

Future BirdCLEF integration will support

* Official metadata parsing
* Primary species labels
* Background species labels
* Recording quality
* Geographic metadata
* Competition evaluation format

The preprocessing pipeline converts BirdCLEF metadata into the project's unified metadata format.

---

# Future Dataset Support

The architecture allows additional datasets through dataset-specific parsers.

Future integrations include

* Xeno-canto
* Macaulay Library
* eBird recordings
* Citizen science datasets
* Private research collections

Once converted into the unified metadata format, no further modifications are required by the training pipeline.

---

# Dataset Statistics

Future preprocessing runs will automatically generate summary statistics.

Examples

* Number of species
* Number of recordings
* Class distribution
* Recording duration distribution
* Source distribution
* Missing metadata
* Storage requirements

These reports assist in dataset quality assessment before training.

---

# Reproducibility

The preprocessing pipeline is deterministic.

Given

* identical configuration,
* identical datasets,
* identical software versions,

the generated spectrograms and metadata will be identical.

This guarantees that experiments remain reproducible over time.

---

# Future Improvements

Planned enhancements include

* Parallel preprocessing
* Incremental preprocessing
* Audio checksum verification
* Corrupted file recovery
* Memory-mapped datasets
* Distributed preprocessing
* Data versioning (DVC)
* Automatic dataset reports
* Dataset caching
* Support for streaming datasets

---

# Summary

The dataset subsystem provides a unified interface between raw bird recordings and machine learning models.

By separating dataset-specific parsing from the training pipeline and performing deterministic offline preprocessing, the Bird Intelligence System ensures scalability, reproducibility, and ease of extension. This design allows new bird audio datasets to be integrated with minimal effort while keeping the learning pipeline unchanged.
