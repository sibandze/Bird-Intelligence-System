# src/data/augmentations/base.py

from abc import ABC, abstractmethod
import torch


class BaseAugmentation(ABC):
    """
    Abstract base class for audio/spectrogram augmentations.
    
    All augmentations operate on tensors of shape [C, F, T] or [B, C, F, T].
    """
    
    @abstractmethod
    def __call__(self, mel_tensor: torch.Tensor) -> torch.Tensor:
        """
        Apply augmentation to a spectrogram tensor.
        
        Args:
            mel_tensor: Input tensor of shape [C, F, T] or [B, C, F, T]
            
        Returns:
            Augmented tensor of same shape
        """
        pass
    
    @abstractmethod
    def get_params(self) -> dict:
        """Return current augmentation parameters for logging."""
        pass
