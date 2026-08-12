# src/data/datasets/base.py

import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ..process_audio import load_local_spectrogram
from ..windowing import (
    build_window_strategy,
    BaseWindowStrategy,
    WindowIndex,
)
from ..augmentations import SpecAugmentation


class BaseSpectrogramDataset(Dataset):
    """
    Base dataset for spectrogram-based learning.

    The dataset exposes one temporal window per recording per epoch.
    The selected windows are controlled by the configured
    BaseWindowStrategy.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        segment_size: int,
        min_db: float,
        max_db: float,
        train: bool = True,
        window_config: dict = None,
        spec_aug_config: dict = None,
    ):
        self.df = df.reset_index(drop=True)
        self.segment_size = segment_size
        self.min_db = min_db
        self.max_db = max_db
        self.train = train
        self.epoch = 0

        # Initialize spec augmentation
        spec_cfg = spec_aug_config or {"enabled": False}
        self.spec_aug = SpecAugmentation(
            enabled=spec_cfg.get("enabled", False),
            prob=spec_cfg.get("prob", 0.5),
            num_freq_masks=spec_cfg.get("num_freq_masks", 1),
            freq_mask_param=spec_cfg.get("freq_mask_param", 0),
            num_time_masks=spec_cfg.get("num_time_masks", 1),
            time_mask_param=spec_cfg.get("time_mask_param", 0),
        )

        self.window_strategy: BaseWindowStrategy = (
            build_window_strategy(window_config)
        )

        self._rebuild_window_index()

    def _get_total_frames(
        self,
        row: pd.Series,
        idx: int,
    ) -> int:
        """
        Get total spectrogram frames for a recording.
        
        Uses pre-computed 'total_frames' column from metadata when available.
        Falls back to loading spectrogram header for legacy data.
        """
        # Fast path: use pre-computed frame count from metadata
        if "total_frames" in row and not pd.isna(row["total_frames"]):
            return int(row["total_frames"])
        
        # Legacy fallback: load spectrogram to get shape
        # This should rarely happen if pipeline properly computes total_frames
        mel = load_local_spectrogram(row["local_spectrogram_path"])
        return mel.shape[1]

    def _rebuild_window_index(self):
        """
        Rebuild the temporal sample index for the current epoch.
        """

        self.windows: list[WindowIndex] = (
            self.window_strategy.build_window_index(
                df=self.df,
                segment_size=self.segment_size,
                get_frames_fn=self._get_total_frames,
                epoch=self.epoch,
                is_train=self.train,
            )
        )

    def set_epoch(self, epoch: int):
        """
        Update the epoch and rebuild temporal sampling.

        Training loops should call this once before each epoch.
        
        Note for DDP: Call dataset.set_epoch(epoch) alongside 
        sampler.set_epoch(epoch) to keep window indices synchronized 
        across distributed processes.
        """

        self.epoch = epoch
        self._rebuild_window_index()

    def __len__(self) -> int:
        return len(self.windows)
        
    def _extract_window_tensor(self, window: WindowIndex) -> torch.Tensor:
        row = self.df.iloc[window.recording_idx]
        mel = load_local_spectrogram(row["local_spectrogram_path"])
        
        print(  
            f"base.py _extract_window_tensor mel_loaded: {tuple(mel.shape)}"
        )
        mel = self._normalize(mel)
        T = mel.shape[1]
        
        if T <= self.segment_size:
            pad = self.segment_size - T
            mel_crop = np.pad(mel, ((0, 0), (0, pad)), mode="constant", constant_values=0.0).astype(mel.dtype)
            print(
                f"base.py _extract_window_tensor mel_crop 1: {tuple(mel_crop.shape)}"
            )
        else:
            mel_crop = mel[:, window.start_frame:window.end_frame]
             print(
                f"base.py _extract_window_tensor mel_crop 2: {tuple(mel_crop.shape)}"
            )
        
        tensor = torch.from_numpy(mel_crop).float() 
        print(
            f"base.py _extract_window_tensor output: {tuple(tensor.shape)}"
        )
        return tensor

    def _normalize(
        self,
        mel: np.ndarray,
    ) -> np.ndarray:

        mel = np.clip(
            mel,
            self.min_db,
            self.max_db,
        )

        return (
            (mel - self.min_db)
            / (self.max_db - self.min_db)
        )

    def _apply_spec_augment(
        self,
        mel_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """Apply spec augmentation (delegates to SpecAugmentation instance)."""
        if self.train:
            return self.spec_aug(mel_tensor)
        return mel_tensor
