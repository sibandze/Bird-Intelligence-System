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
        # print(f"SpecAugmentation input: {tuple(mel_tensor.shape)}")

        if not self.enabled:
            return mel_tensor

        mel_tensor = mel_tensor.clone()

        # Track original shape and normalization
        original_dim = mel_tensor.dim()
        was_2d = False
        was_3d_batch = False
        was_3d_single = False

        # Normalize to 4D [B, C, F, T]
        if original_dim == 2:
            mel_tensor = mel_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, F, T]
            was_2d = True
        elif original_dim == 3:
            # If first dim > 1, assume batch of 1‑channel spectrograms [B, F, T]
            if mel_tensor.shape[0] > 1:
                mel_tensor = mel_tensor.unsqueeze(1)  # [B, 1, F, T]
                was_3d_batch = True
            else:
                # Single sample with channel dim [C, F, T] (C likely 1)
                mel_tensor = mel_tensor.unsqueeze(0)  # [1, C, F, T]
                was_3d_single = True
        # else: already 4D, no flags needed

        B, C, n_mels, n_frames = mel_tensor.shape

        # Apply masking (unchanged from original)
        for b in range(B):
            if random.random() >= self.prob:
                continue

            # Frequency masking
            if self.freq_mask_param > 0:
                max_f = min(self.freq_mask_param, n_mels)
                for _ in range(self.num_freq_masks):
                    f = random.randint(1, max_f)
                    f0 = random.randint(0, n_mels - f)
                    mel_tensor[b, :, f0 : f0 + f, :] = 0.0

            # Time masking
            if self.time_mask_param > 0:
                max_t = min(self.time_mask_param, n_frames)
                for _ in range(self.num_time_masks):
                    t = random.randint(1, max_t)
                    t0 = random.randint(0, n_frames - t)
                    mel_tensor[b, :, :, t0 : t0 + t] = 0.0

        # print(f"SpecAugmentation before_restore: {tuple(mel_tensor.shape)}")

        # Restore original shape
        if was_2d:
            mel_tensor = mel_tensor.squeeze(0).squeeze(0)  # [F, T]
        elif was_3d_batch:
            mel_tensor = mel_tensor.squeeze(1)  # [B, F, T]
        elif was_3d_single:
            mel_tensor = mel_tensor.squeeze(0)  # [C, F, T]

        # print(f"SpecAugmentation output: {tuple(mel_tensor.shape)}")

        return mel_tensor

    def get_params(self) -> dict:
        """Return current augmentation parameters for logging."""
        return {
            "type": "specaugment",
            "enabled": self.enabled,
            "prob": self.prob,
            "num_freq_masks": self.num_freq_masks,
            "freq_mask_param": self.freq_mask_param,
            "num_time_masks": self.num_time_masks,
            "time_mask_param": self.time_mask_param,
        }
