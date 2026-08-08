# src/data/augmentations/spectrogram.py

import random
import torch
from .base import BaseAugmentation


class SpecAugmentation(BaseAugmentation):
    """
    SpecAugment: Frequency and time masking augmentation.
    
    Based on the paper:
    "SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition"
    Park et al., 2019
    """
    
    def __init__(
        self,
        enabled: bool = True,
        prob: float = 0.5,
        num_freq_masks: int = 2,
        freq_mask_param: int = 6,
        num_time_masks: int = 2,
        time_mask_param: int = 10,
    ):
        """
        Args:
            enabled: Whether augmentation is active
            prob: Probability of applying augmentation to a sample
            num_freq_masks: Number of frequency masks to apply
            freq_mask_param: Maximum width of frequency masks
            num_time_masks: Number of time masks to apply
            time_mask_param: Maximum width of time masks
        """
        self.enabled = enabled
        self.prob = prob
        self.num_freq_masks = num_freq_masks
        self.freq_mask_param = freq_mask_param
        self.num_time_masks = num_time_masks
        self.time_mask_param = time_mask_param
    
    def __call__(self, mel_tensor: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return mel_tensor
        # Handle both [C, F, T] and [B, C, F, T] shapes
        if mel_tensor.dim() == 3:
            mel_tensor = mel_tensor.unsqueeze(0)
            was_3d = True
        else:
            was_3d = False
    
        B, C, n_mels, n_frames = mel_tensor.shape
       
        # Apply per-sample masks
        for b in range(B):
            if random.random() >= self.prob:
                continue      
            # Frequency masking
            for _ in range(self.num_freq_masks):
                f = random.randint(0, self.freq_mask_param)
                if f > 0:
                    f0 = random.randint(0, n_mels - f)
                    mel_tensor[b, :, f0:f0 + f, :] = 0.0
        
            # Time masking
            for _ in range(self.num_time_masks):
                t = random.randint(0, self.time_mask_param)
                if t > 0:
                    t0 = random.randint(0, n_frames - t)
                    mel_tensor[b, :, :, t0:t0 + t] = 0.0
    
        if was_3d:
            mel_tensor = mel_tensor.squeeze(0)
    
        return mel_tensor
    
    def get_params(self) -> dict:
        return {
            'type': 'specaugment',
            'enabled': self.enabled,
            'prob': self.prob,
            'num_freq_masks': self.num_freq_masks,
            'freq_mask_param': self.freq_mask_param,
            'num_time_masks': self.num_time_masks,
            'time_mask_param': self.time_mask_param,
        }
