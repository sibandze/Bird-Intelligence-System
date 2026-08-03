# src/data/__init__.py

"""
Data module for Bird-Intelligence-System.

This module handles data downloading, preprocessing, and dataset creation
for bird song classification.
"""

from .datasets import (
    BaseSpectrogramDataset,
    SSLBirdSongDataset,
    SimCLRDataset,
    BYOLDataset,
    MoCoDataset,
    simclr_collate_fn,
    byol_collate_fn,
    moco_collate_fn,
    SupervisedBirdSongDataset,
)
from .process_audio import (
    generate_mel_spectrogram_data,
    save_spectrogram_npy,
    preprocess_and_save,
    load_local_spectrogram,
    visualize_mel_spectrogram,
    save_spectrogram_image,
)
from .download import download_audio
from .run_pipeline import run_data_pipeline

__all__ = [
    # Datasets
    'BaseSpectrogramDataset',
    'SSLBirdSongDataset',
    'SimCLRDataset',
    'BYOLDataset',
    'MoCoDataset',
    'simclr_collate_fn',
    'byol_collate_fn',
    'moco_collate_fn',
    'SupervisedBirdSongDataset',
    # Audio processing
    'generate_mel_spectrogram_data',
    'save_spectrogram_npy',
    'preprocess_and_save',
    'load_local_spectrogram',
    'visualize_mel_spectrogram',
    'save_spectrogram_image',
    # Utilities
    'download_audio',
    'run_data_pipeline',
]
