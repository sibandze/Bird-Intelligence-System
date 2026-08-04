# src/data/datasets/ssl.py

import pandas as pd
import torch
from .base import BaseSpectrogramDataset
from ..augmentations import AcousticAugmentation, SpecAugmentation, AugmentationPipeline


class SSLBirdSongDataset(BaseSpectrogramDataset):
    """
    Base SSL dataset producing dual augmented views (x1, x2)
    from the SAME WindowIndex entry.
    """
    def __init__(
        self,
        df: pd.DataFrame,
        segment_size: int,
        min_db: float,
        max_db: float,
        acoustic_aug_config: dict = None,
        spec_aug_config: dict = None,
        **kwargs,
    ):
        super().__init__(
            df=df,
            segment_size=segment_size,
            min_db=min_db,
            max_db=max_db,
            **kwargs,
        )
        
        # Initialize augmentation pipeline
        self.aug_pipeline = AugmentationPipeline()
        
        # Add acoustic augmentations
        acoustic_cfg = acoustic_aug_config or {"enabled": True, "noise_level": 0.05}
        self.aug_pipeline.add(
            AcousticAugmentation(
                enabled=acoustic_cfg.get("enabled", True),
                noise_level=acoustic_cfg.get("noise_level", 0.05),
                noise_prob=acoustic_cfg.get("noise_prob", 1.0),
            )
        )
        
        # Add spectrogram augmentations
        spec_cfg = spec_aug_config or {"enabled": True}
        self.aug_pipeline.add(
            SpecAugmentation(
                enabled=spec_cfg.get("enabled", False),
                prob=spec_cfg.get("prob", 0.5),
                num_freq_masks=spec_cfg.get("num_freq_masks", 2),
                freq_mask_param=spec_cfg.get("freq_mask_param", 6),
                num_time_masks=spec_cfg.get("num_time_masks", 2),
                time_mask_param=spec_cfg.get("time_mask_param", 10),
            )
        )

    def _generate_view(self, window) -> torch.Tensor:
        """Generate a single augmented view from a window."""
        view = self._extract_window_tensor(window)
        if self.train:
            view = self.aug_pipeline(view)
        return view

    def __getitem__(self, idx: int):
        window = self.windows[idx]
        x1 = self._generate_view(window)
        x2 = self._generate_view(window)
        return x1, x2


# ---------------------------------------------------------------------
# Framework Adapters & Collate Functions
# ---------------------------------------------------------------------

class SimCLRDataset(SSLBirdSongDataset):
    """SimCLR Specific Dataset Adapter."""
    pass

def simclr_collate_fn(batch):
    """Collates [(x1, x2), ...] into batch [2*B, 1, F, T]."""
    view1_list, view2_list = zip(*batch)
    view1 = torch.stack(view1_list, dim=0)  # [B, 1, F, T]
    view2 = torch.stack(view2_list, dim=0)  # [B, 1, F, T]
    return torch.cat([view1, view2], dim=0)


class BYOLDataset(SSLBirdSongDataset):
    """BYOL Specific Dataset Adapter."""
    pass

def byol_collate_fn(batch):
    """Collates [(x1, x2), ...] into separate tensors (view1 [B, 1, F, T], view2 [B, 1, F, T])."""
    view1_list, view2_list = zip(*batch)
    return torch.stack(view1_list, dim=0), torch.stack(view2_list, dim=0)


class MoCoDataset(SSLBirdSongDataset):
    """MoCo Specific Dataset Adapter."""
    pass

def moco_collate_fn(batch):
    """Collates [(x1, x2), ...] into query (im_q) and key (im_k) batches."""
    view1_list, view2_list = zip(*batch)
    im_q = torch.stack(view1_list, dim=0)
    im_k = torch.stack(view2_list, dim=0)
    return im_q, im_k
