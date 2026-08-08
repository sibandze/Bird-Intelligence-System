# src/data/augmentations/spectrogram.py

import random
import torch
from .base import BaseAugmentation


class SpecAugmentation(BaseAugmentation):
    """SpecAugment: Frequency and time masking augmentation."""

    def __init__(
        self,
        enabled: bool = True,
        prob: float = 0.5,
        num_freq_masks: int = 2,
        freq_mask_param: int = 6,
        num_time_masks: int = 2,
        time_mask_param: int = 10,
    ):
        super().__init__()
        self.enabled = enabled
        self.prob = prob
        self.num_freq_masks = num_freq_masks
        self.freq_mask_param = freq_mask_param
        self.num_time_masks = num_time_masks
        self.time_mask_param = time_mask_param

    def __call__(self, mel_tensor: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return mel_tensor

        mel_tensor = mel_tensor.clone()

        # Handle shape normalization
        if mel_tensor.dim() == 3:
            # If shape is [B, F, T] (e.g. 1-channel batch), unsqueeze channel dim -> [B, 1, F, T]
            # If shape is [C, F, T] where C is 1, treat as unbatched -> [1, C, F, T]
            if mel_tensor.shape[0] > 1 and mel_tensor.shape[0] != 1:  # [B, F, T]
                mel_tensor = mel_tensor.unsqueeze(1)  # [B, 1, F, T]
                was_3d_batch = True
                was_3d_single = False
            else:  # [1, F, T] or [C, F, T] single sample
                mel_tensor = mel_tensor.unsqueeze(0)  # [1, C, F, T]
                was_3d_batch = False
                was_3d_single = True
        elif mel_tensor.dim() == 2:  # [F, T]
            mel_tensor = mel_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, F, T]
            was_3d_batch = False
            was_3d_single = True
        else:  # Already 4D [B, C, F, T]
            was_3d_batch = False
            was_3d_single = False

        B, C, n_mels, n_frames = mel_tensor.shape

        # Apply per-sample masks across batch dimension B
        for b in range(B):
            if random.random() >= self.prob:
                continue

            # Frequency masking
            if self.freq_mask_param > 0:
                max_f = min(self.freq_mask_param, n_mels)
                for _ in range(self.num_freq_masks):
                    f = random.randint(1, max_f)
                    f0 = random.randint(0, n_mels - f)
                    mel_tensor[b, :, f0:f0 + f, :] = 0.0

            # Time masking
            if self.time_mask_param > 0:
                max_t = min(self.time_mask_param, n_frames)
                for _ in range(self.num_time_masks):
                    t = random.randint(1, max_t)
                    t0 = random.randint(0, n_frames - t)
                    mel_tensor[b, :, :, t0:t0 + t] = 0.0

        # Restore original tensor shape layout
        if was_3d_batch:
            mel_tensor = mel_tensor.squeeze(1)  # Return to [B, F, T]
        elif was_3d_single:
            mel_tensor = mel_tensor.squeeze(0)  # Return to original 3D or 2D

        return mel_tensor

    def get_params(self) -> dict:
        """Return current augmentation parameters for logging."""
        return {
            'type': 'spectrogram',
            'enabled': self.enabled,
            'prob': self.prob,
            'num_freq_masks': self.num_freq_masks,
            'freq_mask_param': self.freq_mask_param,
            'num_time_masks': self.num_time_masks,
            'time_mask_param': self.time_mask_param,
        }
