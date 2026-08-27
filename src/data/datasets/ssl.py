# src/data/datasets/ssl.py

import torch
from .base import BaseSpectrogramDataset
from ..augmentations import AcousticAugmentation, SpecAugmentation, AugmentationPipeline


class SSLBirdSongDataset(BaseSpectrogramDataset):
    """
    Base SSL dataset producing dual augmented views (x1, x2)
    from the SAME WindowIndex entry.

    Additional parameters (beyond those of BaseSpectrogramDataset):
        acoustic_aug_config: dict or None – configuration for acoustic augmentation.
        spec_aug_config: dict or None – configuration for spectrogram augmentation
                           (used in the SSL pipeline, independent of the base's spec_aug).
        apply_augmentation: bool – whether to apply the augmentation pipeline.
    All other arguments are passed directly to BaseSpectrogramDataset via **kwargs.
    """
    def __init__(
        self,
        acoustic_aug_config: dict = None,
        spec_aug_config: dict = None,
        apply_augmentation: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.apply_augmentation = apply_augmentation

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
        if self.apply_augmentation:
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

class BYOLDataset(SSLBirdSongDataset):
    """BYOL Specific Dataset Adapter."""
    pass

class MoCoDataset(SSLBirdSongDataset):
    """MoCo Specific Dataset Adapter."""
    pass

def _ssl_dual_view_collate(batch):
    """
    Base collate for all dual-view SSL methods.

    Input:  batch = [(x1, x2), (x1, x2), ...] where x is [1, F, T]
    Output: (x1_batch, x2_batch) where each is [B, 1, F, T]
    """
    view1_list, view2_list = zip(*batch)
    x1 = torch.stack(view1_list, dim=0)  # [B, 1, F, T]
    x2 = torch.stack(view2_list, dim=0)
    return x1, x2

# Framework-specific aliases
# Keep separate names so we can extend them later without breaking API
simclr_collate_fn = _ssl_dual_view_collate
byol_collate_fn = _ssl_dual_view_collate
moco_collate_fn = _ssl_dual_view_collate
