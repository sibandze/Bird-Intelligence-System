# src/data/dataset.py

import random
import torch
import pandas as pd
import numpy as np
from.process_audio import load_local_spectrogram

class BirdSongDataset(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame, segment_size=187, train=True, label_to_idx=None):
        self.df = df.reset_index(drop=True)
        self.segment_size = segment_size
        self.train = train

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

        T = mel.shape[1]
        if T > self.segment_size:
            start = random.randint(0, T - self.segment_size) if self.train else (T - self.segment_size) // 2
            mel_segment = mel[:, start:start+self.segment_size]
        else:
            pad = self.segment_size - T
            mel_segment = np.pad(mel, ((0,0),(0,pad)), mode='constant')

        mel_segment = torch.from_numpy(mel_segment).float()
        label = torch.tensor(int(row['scientific_name_id'])).long()

        return mel_segment, label
