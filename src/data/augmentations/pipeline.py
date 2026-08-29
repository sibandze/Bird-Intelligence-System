# src/data/augmentations/pipeline.py

from typing import List, Optional
import torch
from .base import BaseAugmentation


class AugmentationPipeline(BaseAugmentation):
    """
    Composable augmentation pipeline.

    Applies a sequence of augmentations in order.
    Can be configured globally or per-augmentation.
    """

    def __init__(
        self,
        augmentations: Optional[List[BaseAugmentation]] = None,
        enabled: bool = True,
    ):
        """
        Args:
            augmentations: List of augmentation instances to apply sequentially
            enabled: Global enable/disable switch
        """
        self.augmentations = augmentations or []
        self.enabled = enabled

    def add(self, augmentation: BaseAugmentation):
        """Add an augmentation to the pipeline."""
        self.augmentations.append(augmentation)
        return self

    def __call__(self, mel_tensor: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return mel_tensor

        for aug in self.augmentations:
            mel_tensor = aug(mel_tensor)

        return mel_tensor

    def get_params(self) -> dict:
        return {
            "type": "pipeline",
            "enabled": self.enabled,
            "num_augmentations": len(self.augmentations),
            "augmentations": [aug.get_params() for aug in self.augmentations],
        }

    def __len__(self) -> int:
        return len(self.augmentations)

    def __repr__(self) -> str:
        aug_names = [type(a).__name__ for a in self.augmentations]
        return f"AugmentationPipeline({', '.join(aug_names)})"
