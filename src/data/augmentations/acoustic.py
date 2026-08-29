# src/data/augmentations/acoustic.py

import torch
from .base import BaseAugmentation


class AcousticAugmentation(BaseAugmentation):
    """
    Acoustic augmentations that simulate recording variations.

    Currently supports:
    - Gaussian noise injection
    - (Future: room impulse response, background noise mixing, etc.)
    """

    def __init__(
        self,
        enabled: bool = True,
        noise_level: float = 0.05,
        noise_prob: float = 1.0,
    ):
        """
        Args:
            enabled: Whether augmentation is active
            noise_level: Standard deviation of Gaussian noise (relative to normalized [0,1] range)
            noise_prob: Probability of applying noise when enabled
        """
        self.enabled = enabled
        self.noise_level = noise_level
        self.noise_prob = noise_prob

    def __call__(self, mel_tensor: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return mel_tensor

        # Add Gaussian noise
        if self.noise_level > 0 and torch.rand(1).item() < self.noise_prob:
            noise = torch.randn_like(mel_tensor) * self.noise_level
            mel_tensor = torch.clamp(mel_tensor + noise, 0.0, 1.0)

        return mel_tensor

    def get_params(self) -> dict:
        return {
            "type": "acoustic",
            "enabled": self.enabled,
            "noise_level": self.noise_level,
            "noise_prob": self.noise_prob,
        }
