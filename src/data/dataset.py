# src/data/dataset.py

import random
import torch
import numpy as np
from torch.utils.data import Dataset
from.process_audio import load_local_spectrogram

class BirdSongDataset(Dataset):
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
        self.min_db = min_db
        self.max_db = max_db
        self.spec_aug_config = spec_aug_config or {
            "enabled": True, "prob": 0.5, "num_freq_masks": 2, "freq_mask_param": 6,
            "num_time_masks": 2, "time_mask_param": 10
        }
        species_df = df[['scientific_name_id', 'scientific_name']].drop_duplicates().sort_values('scientific_name_id')
        if label_to_idx is None:
            self.label_to_idx = {row.scientific_name: int(row.scientific_name_id) for _, row in species_df.iterrows()}
        else:
            self.label_to_idx = label_to_idx
        self.idx_to_label = {v: k for k, v in self.label_to_idx.items()}
        self.num_classes = len(self.label_to_idx)

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        mel = load_local_spectrogram(row['local_spectrogram_path']) # (n_mels, T)
        mel = self._normalize(mel)
        mel = self._crop_or_pad(mel)
        mel_tensor = torch.from_numpy(mel).float().unsqueeze(0) # add channel dim: [1, F, T]
        mel_tensor = self._apply_spec_augment(mel_tensor)
        label = torch.tensor(int(row['scientific_name_id'])).long()
        return mel_tensor, label

    def _normalize(self, mel: np.ndarray) -> np.ndarray:
        mel = np.clip(mel, self.min_db, self.max_db)
        return (mel - self.min_db) / (self.max_db - self.min_db)

    def _crop_or_pad(self, mel: np.ndarray) -> np.ndarray:
        T = mel.shape[1]
        if T > self.segment_size:
            start = random.randint(0, T - self.segment_size) if self.train else (T - self.segment_size) // 2
            return mel[:, start:start+self.segment_size]
        else:
            pad = self.segment_size - T
            return np.pad(mel, ((0,0),(0,pad)), mode='constant')

    def _apply_spec_augment(self, mel_tensor: torch.Tensor) -> torch.Tensor:
        cfg = self.spec_aug_config
        if not (self.train and cfg.get("enabled", False) and random.random() < cfg.get("prob", 0.5)):
            return mel_tensor
        _, n_mels, n_frames = mel_tensor.shape
        for _ in range(cfg.get("num_freq_masks", 1)):
            f = random.randint(0, cfg.get("freq_mask_param", 0))
            if f > 0: f0 = random.randint(0, n_mels - f); mel_tensor[:, f0:f0+f, :] = 0.0
        for _ in range(cfg.get("num_time_masks", 1)):
            t = random.randint(0, cfg.get("time_mask_param", 0))
            if t > 0: t0 = random.randint(0, n_frames - t); mel_tensor[:, :, t0:t0+t] = 0.0
        return mel_tensor

class ContrastiveBirdSongDataset(BirdSongDataset):
    def __init__(self, *args, background_noises=None, acoustic_aug_config=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.background_noises = background_noises
        self.acoustic_aug_config = acoustic_aug_config or {
            "enabled": True, "time_shift_max_frac": 0.1, "noise_level": 0.05, "mix_prob": 0.5,
        }

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        mel = load_local_spectrogram(row['local_spectrogram_path'])
        mel = self._normalize(mel)
        mel = self._crop_or_pad(mel)

        view1 = torch.from_numpy(mel).float().unsqueeze(0) # [1, F, T]
        view2 = torch.from_numpy(mel).float().unsqueeze(0)

        if self.train and self.acoustic_aug_config.get("enabled", True):
            view1 = self._apply_acoustic_augmentations(view1)
            view2 = self._apply_acoustic_augmentations(view2)

        view1 = self._apply_spec_augment(view1.clone())
        view2 = self._apply_spec_augment(view2.clone())

        return view1, view2 # no labels

def contrastive_collate_fn(batch):
    """
    Collate for ContrastiveBirdSongDataset.
    Input: list of tuples [(v1, v2), (v1, v2),...] where v.shape = [1, F, T]
    Output: tensor [2*B, 1, F, T]
    """
    view1_list, view2_list = zip(*batch)
    view1 = torch.stack(view1_list, dim=0) # [B, 1, F, T]
    view2 = torch.stack(view2_list, dim=0) # [B, 1, F, T]
    combined = torch.cat([view1, view2], dim=0) # [2*B, 1, F, T]
    return combined
