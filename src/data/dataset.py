import random
import torch
import pandas as pd
import numpy as np
from .process_audio import load_local_spectrogram

class BirdSongDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 df: pd.DataFrame, 
                 segment_size: int, 
                 min_db: int,
                 max_db: int,
                 train=True, 
                 label_to_idx=None, 
                 spec_aug_config=None):
        self.df = df.reset_index(drop=True)
        self.segment_size = segment_size
        self.train = train
        
        # Fixed dB Normalization boundaries
        self.min_db = min_db
        self.max_db = max_db

        # Config fallback dictionary
        self.spec_aug_config = spec_aug_config or {
            "enabled": True,
            "prob": 0.5,
            "num_freq_masks": 2,
            "freq_mask_param": 6,
            "num_time_masks": 2,
            "time_mask_param": 10
        }

        species_df = df[['scientific_name_id', 'scientific_name']].drop_duplicates().sort_values('scientific_name_id')

        if label_to_idx is None:
            self.label_to_idx = {row.scientific_name: int(row.scientific_name_id) for _, row in species_df.iterrows()}
        else:
            self.label_to_idx = label_to_idx

        self.idx_to_label = {v: k for k, v in self.label_to_idx.items()}
        self.num_classes = len(self.label_to_idx)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        mel = load_local_spectrogram(row['local_spectrogram_path']) # (n_mels, T)

        # Execution Pipeline
        mel = self._normalize(mel)
        mel = self._crop_or_pad(mel)
        
        mel_tensor = torch.from_numpy(mel).float()
        mel_tensor = self._apply_spec_augment(mel_tensor)

        label = torch.tensor(int(row['scientific_name_id'])).long()

        return mel_tensor, label

    def _normalize(self, mel: np.ndarray) -> np.ndarray:
        """Clips and scales raw dB features to a clean [0.0, 1.0] range."""
        mel = np.clip(mel, self.min_db, self.max_db)
        return (mel - self.min_db) / (self.max_db - self.min_db)

    def _crop_or_pad(self, mel: np.ndarray) -> np.ndarray:
        """Ensures the time axis strictly matches the targeted segment size."""
        T = mel.shape[1]
        if T > self.segment_size:
            start = random.randint(0, T - self.segment_size) if self.train else (T - self.segment_size) // 2
            return mel[:, start:start+self.segment_size]
        else:
            pad = self.segment_size - T
            return np.pad(mel, ((0,0),(0,pad)), mode='constant')

    def _apply_spec_augment(self, mel_tensor: torch.Tensor) -> torch.Tensor:
        """Applies multi-mask frequency and time erasure directly to the tensor."""
        cfg = self.spec_aug_config
        if not (self.train and cfg.get("enabled", False) and random.random() < cfg.get("prob", 0.5)):
            return mel_tensor

        n_mels, n_frames = mel_tensor.shape
        
        # Apply multiple Frequency Masks
        for _ in range(cfg.get("num_freq_masks", 1)):
            f = random.randint(0, cfg.get("freq_mask_param", 0))
            if f > 0:
                f0 = random.randint(0, n_mels - f)
                mel_tensor[f0:f0+f, :] = 0.0
        
        # Apply multiple Time Masks
        for _ in range(cfg.get("num_time_masks", 1)):
            t = random.randint(0, cfg.get("time_mask_param", 0))
            if t > 0:
                t0 = random.randint(0, n_frames - t)
                mel_tensor[:, t0:t0+t] = 0.0

        return mel_tensor
