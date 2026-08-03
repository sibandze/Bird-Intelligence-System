# src/data/datasets/supervised.py

import random
import numpy as np
import torch
import pandas as pd
from .process_audio import load_local_spectrogram
from .base import BaseSpectrogramDataset

# =====================================================================
# Supervised Dataset Pipeline -> Returns (x, y)
# =====================================================================

class SupervisedBirdSongDataset(BaseSpectrogramDataset):
    """
    Supervised learning dataset returning pairs of (x, y).
    """
    def __init__(self,
                 df: pd.DataFrame,
                 segment_size: int,
                 min_db: float,
                 max_db: float,
                 train: bool = True,
                 label_to_idx: dict = None,
                 spec_aug_config: dict = None):
        super().__init__(df, segment_size, min_db, max_db, train, spec_aug_config)

        species_df = df[['scientific_name_id', 'scientific_name']].drop_duplicates().sort_values('scientific_name_id')
        if label_to_idx is None:
            self.label_to_idx = {row.scientific_name: int(row.scientific_name_id) for _, row in species_df.iterrows()}
        else:
            self.label_to_idx = label_to_idx

        self.idx_to_label = {v: k for k, v in self.label_to_idx.items()}
        self.num_classes = len(self.label_to_idx)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        x = self._load_and_preprocess(row['local_spectrogram_path'])
        x = self._apply_spec_augment(x)
        y = torch.tensor(int(row['scientific_name_id'])).long()
        return x, y
