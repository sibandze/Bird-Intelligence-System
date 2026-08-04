# src/data/datasets/supervised.py
import pandas as pd
import torch
from .base import BaseSpectrogramDataset


class SupervisedBirdSongDataset(BaseSpectrogramDataset):
    """
    Supervised learning dataset returning (x, y) for each WindowIndex entry.
    """
    def __init__(self, 
                 df: pd.DataFrame, 
                 segment_size: int, 
                 min_db: float, 
                 max_db: float,
                 label_to_idx: dict = None, 
                 **kwargs):
        super().__init__(df=df, segment_size=segment_size, min_db=min_db, max_db=max_db, **kwargs)

        species_df = df[['scientific_name_id', 'scientific_name']].drop_duplicates().sort_values('scientific_name_id')
        self.label_to_idx = label_to_idx or {row.scientific_name: int(row.scientific_name_id) for _, row in species_df.iterrows()}
        self.idx_to_label = {v: k for k, v in self.label_to_idx.items()}
        self.num_classes = len(self.label_to_idx)

    def __getitem__(self, idx):  
        window = self.windows[idx]
        
        x = self._extract_window_tensor(window)
        x = self._apply_spec_augment(x)
        
        row = self.df.iloc[window.recording_idx]
        
        y = torch.tensor(
            int(row["scientific_name_id"])
        ).long()
        
        return x, y
