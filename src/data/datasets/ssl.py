# src/data/datasets/ssl.py

import random
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
from .base import BaseSpectrogramDataset

# =====================================================================
# Self-Supervised Learning (SSL) Pipeline -> Returns (x1, x2)
# =====================================================================

class SSLBirdSongDataset(BaseSpectrogramDataset):
    """
    Base SSL dataset producing dual augmented views (x1, x2) for representation learning.
    """
    def __init__(self,
                 df: pd.DataFrame,
                 segment_size: int,
                 min_db: float,
                 max_db: float,
                 train: bool = True,
                 acoustic_aug_config: dict = None,
                 spec_aug_config: dict = None):
        super().__init__(df, segment_size, min_db, max_db, train, spec_aug_config)
        self.acoustic_aug_config = acoustic_aug_config or {
            "enabled": True, "time_shift_max_frac": 0.1, "noise_level": 0.05, "mix_prob": 0.5,
        }

    def _apply_acoustic_augmentations(self, mel_tensor: torch.Tensor) -> torch.Tensor:
        # Placeholder for acoustic augmentations (e.g., Gaussian noise, time shifts, pitch shift)
        if self.acoustic_aug_config.get("noise_level", 0) > 0:
            noise = torch.randn_like(mel_tensor) * self.acoustic_aug_config["noise_level"]
            mel_tensor = torch.clamp(mel_tensor + noise, 0.0, 1.0)
        return mel_tensor

    def _generate_view(self, path: str) -> torch.Tensor:
        view = self._load_and_preprocess(path)
        if self.train and self.acoustic_aug_config.get("enabled", True):
            view = self._apply_acoustic_augmentations(view)
        view = self._apply_spec_augment(view.clone())
        return view

    def __getitem__(self, idx: int):
        path = self.df.iloc[idx]['local_spectrogram_path']
        x1 = self._generate_view(path)
        x2 = self._generate_view(path)
        return x1, x2


# ---------------------------------------------------------------------
# Algorithm-Specific Subclasses & Collate Functions
# ---------------------------------------------------------------------

class SimCLRDataset(SSLBirdSongDataset):
    """SimCLR Specific Dataset Adapter."""
    pass

def simclr_collate_fn(batch):
    """
    Collates [(x1, x2), ...] into a concatenated batch [2*B, 1, F, T]
    typically expected by SimCLR implementations.
    """
    view1_list, view2_list = zip(*batch)
    view1 = torch.stack(view1_list, dim=0)  # [B, 1, F, T]
    view2 = torch.stack(view2_list, dim=0)  # [B, 1, F, T]
    return torch.cat([view1, view2], dim=0)  # [2*B, 1, F, T]


class BYOLDataset(SSLBirdSongDataset):
    """BYOL Specific Dataset Adapter."""
    pass

def byol_collate_fn(batch):
    """
    Collates [(x1, x2), ...] into two separate batched tensors:
    (view1 [B, 1, F, T], view2 [B, 1, F, T]) for online & target networks.
    """
    view1_list, view2_list = zip(*batch)
    return torch.stack(view1_list, dim=0), torch.stack(view2_list, dim=0)


class MoCoDataset(SSLBirdSongDataset):
    """MoCo Specific Dataset Adapter."""
    pass

def moco_collate_fn(batch):
    """
    Collates [(x1, x2), ...] into query and key batches:
    (im_q [B, 1, F, T], im_k [B, 1, F, T]).
    """
    view1_list, view2_list = zip(*batch)
    im_q = torch.stack(view1_list, dim=0)
    im_k = torch.stack(view2_list, dim=0)
    return im_q, im_k
