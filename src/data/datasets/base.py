import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from ..process_audio import load_local_spectrogram
from ..windowing import build_window_strategy, BaseWindowStrategy, WindowIndex


class BaseSpectrogramDataset(Dataset):
    """
    Base Dataset mapping dataset index space to precomputed WindowIndex entries.
    `len(dataset)` equals total temporal windows across all audio recordings.
    """
    def __init__(self,
                 df: pd.DataFrame,
                 segment_size: int,
                 min_db: float,
                 max_db: float,
                 train: bool = True,
                 window_config: dict = None,
                 spec_aug_config: dict = None):
        self.df = df.reset_index(drop=True)
        self.segment_size = segment_size
        self.min_db = min_db
        self.max_db = max_db
        self.train = train
        self.spec_aug_config = spec_aug_config or {"enabled": False}

        # Build window index space using specified strategy (default: sliding)
        self.window_strategy: BaseWindowStrategy = build_window_strategy(window_config)
        self.windows: list[WindowIndex] = self.window_strategy.build_window_index(
            df=self.df,
            segment_size=self.segment_size,
            get_frames_fn=self._get_total_frames
        )

    def _get_total_frames(self, row: pd.Series, idx: int) -> int:
        """Helper to extract total frames from metadata or spectrogram on disk."""
        if 'total_frames' in row and not pd.isna(row['total_frames']):
            return int(row['total_frames'])
        mel = load_local_spectrogram(row['local_spectrogram_path'])
        return mel.shape[1]

    def __len__(self) -> int:
        return len(self.windows)

    def _extract_window_tensor(self, window: WindowIndex) -> torch.Tensor:
        row = self.df.iloc[window.recording_idx]
        mel = load_local_spectrogram(row['local_spectrogram_path'])  # Shape: (n_mels, T)
        mel = self._normalize(mel)

        T = mel.shape[1]
        if T <= self.segment_size:
            pad = self.segment_size - T
            mel_crop = np.pad(mel, ((0, 0), (0, pad)), mode='constant')
        else:
            mel_crop = mel[:, window.start_frame:window.end_frame]

        return torch.from_numpy(mel_crop).float().unsqueeze(0)

    def _normalize(self, mel: np.ndarray) -> np.ndarray:
        mel = np.clip(mel, self.min_db, self.max_db)
        return (mel - self.min_db) / (self.max_db - self.min_db)

    def _apply_spec_augment(self, mel_tensor: torch.Tensor) -> torch.Tensor:
        cfg = self.spec_aug_config
        if not (self.train and cfg.get("enabled", False) and random.random() < cfg.get("prob", 0.5)):
            return mel_tensor

        _, n_mels, n_frames = mel_tensor.shape
        for _ in range(cfg.get("num_freq_masks", 1)):
            f = random.randint(0, cfg.get("freq_mask_param", 0))
            if f > 0:
                f0 = random.randint(0, n_mels - f)
                mel_tensor[:, f0:f0 + f, :] = 0.0

        for _ in range(cfg.get("num_time_masks", 1)):
            t = random.randint(0, cfg.get("time_mask_param", 0))
            if t > 0:
                t0 = random.randint(0, n_frames - t)
                mel_tensor[:, :, t0:t0 + t] = 0.0

        return mel_tensor
