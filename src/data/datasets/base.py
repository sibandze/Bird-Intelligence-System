import random
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
from .process_audio import load_local_spectrogram


class BaseSpectrogramDataset(Dataset):
    """
    Base Dataset handling loading, normalization, cropping/padding, 
    and SpecAugment transformation steps for spectrograms.
    """
    def __init__(self,
                 df: pd.DataFrame,
                 segment_size: int,
                 min_db: float,
                 max_db: float,
                 train: bool = True,
                 spec_aug_config: dict = None):
        self.df = df.reset_index(drop=True)
        self.segment_size = segment_size
        self.min_db = min_db
        self.max_db = max_db
        self.train = train
        self.spec_aug_config = spec_aug_config or {
            "enabled": True,
            "prob": 0.5,
            "num_freq_masks": 2,
            "freq_mask_param": 6,
            "num_time_masks": 2,
            "time_mask_param": 10
        }

    def __len__(self) -> int:
        return len(self.df)

    def _load_and_preprocess(self, path: str) -> torch.Tensor:
        """Loads, normalizes, crops/pads a spectrogram into a 3D Tensor [1, F, T]."""
        mel = load_local_spectrogram(path)  # Shape: (n_mels, T)
        mel = self._normalize(mel)
        mel = self._crop_or_pad(mel)
        return torch.from_numpy(mel).float().unsqueeze(0)

    def _normalize(self, mel: np.ndarray) -> np.ndarray:
        mel = np.clip(mel, self.min_db, self.max_db)
        return (mel - self.min_db) / (self.max_db - self.min_db)

    def _crop_or_pad(self, mel: np.ndarray) -> np.ndarray:
        T = mel.shape[1]
        if T > self.segment_size:
            start = random.randint(0, T - self.segment_size) if self.train else (T - self.segment_size) // 2
            return mel[:, start:start + self.segment_size]
        else:
            pad = self.segment_size - T
            return np.pad(mel, ((0, 0), (0, pad)), mode='constant')

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

class MoCoDataset(SSLBirdSongDataset):
    """MoCo Specific Dataset Adapter."""
    pass

def moco_collate_fn(batch):
    """
    Collates [(x1, x2), ...] into query and key batches:
    (im_q [B, 1, F, T], im_k [B, 1, F, T]).
    """
    view1_list, view2_list = zip(*batch)
    im_q = torch.stack(view1_list, dim=0)
    im_k = torch.stack(view2_list, dim=0)
    return im_q, im_k
