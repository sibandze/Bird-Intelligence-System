import pandas as pd
import torch
from .base import BaseSpectrogramDataset


class SSLBirdSongDataset(BaseSpectrogramDataset):
    """
    Base SSL dataset producing dual augmented views (x1, x2) 
    from the SAME WindowIndex entry.
    """
    def __init__(self, 
                 df: pd.DataFrame, 
                 segment_size: int, 
                 min_db: float, 
                 max_db: float,
                 acoustic_aug_config: dict = None, 
                 **kwargs):
        super().__init__(df=df, segment_size=segment_size, min_db=min_db, max_db=max_db, **kwargs)
        self.acoustic_aug_config = acoustic_aug_config or {"enabled": True, "noise_level": 0.05}

    def _apply_acoustic_augmentations(self, mel_tensor: torch.Tensor) -> torch.Tensor:
        if self.acoustic_aug_config.get("noise_level", 0) > 0:
            noise = torch.randn_like(mel_tensor) * self.acoustic_aug_config["noise_level"]
            mel_tensor = torch.clamp(mel_tensor + noise, 0.0, 1.0)
        return mel_tensor

    def _generate_view(self, window) -> torch.Tensor:
        view = self._extract_window_tensor(window)
        if self.train and self.acoustic_aug_config.get("enabled", True):
            view = self._apply_acoustic_augmentations(view)
        view = self._apply_spec_augment(view.clone())
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
