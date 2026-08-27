# src/data/datasets/supervised.py

import pandas as pd
import torch
from .base import BaseSpectrogramDataset
from collections import Counter

class SupervisedBirdSongDataset(BaseSpectrogramDataset):
    """
    Supervised learning dataset returning (x, y) for each WindowIndex entry.

    Handles non-contiguous label IDs by mapping them to contiguous indices
    [0, num_classes-1] expected by CrossEntropyLoss.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        label_to_idx: dict = None,
        return_recording_id: bool = False,
        **kwargs,
    ):
        super().__init__(
            df=df,
            **kwargs,
        )

        if label_to_idx is None:
            # Build mapping from unique species in the dataframe
            species_df = (
                df[['scientific_name_id', 'scientific_name']]
                .drop_duplicates()
                .sort_values('scientific_name_id')
            )
            # Map scientific_name → contiguous 0-indexed label
            self.label_to_idx = {
                row.scientific_name: idx
                for idx, (_, row) in enumerate(species_df.iterrows())
            }
        else:
            self.label_to_idx = label_to_idx

        # Create reverse mapping
        self.idx_to_label = {v: k for k, v in self.label_to_idx.items()}
        self.num_classes = len(self.label_to_idx)

        # Validate that all labels in df exist in mapping
        df_labels = set(df['scientific_name'].unique())
        mapped_labels = set(self.label_to_idx.keys())
        missing_labels = df_labels - mapped_labels
        if missing_labels:
            raise ValueError(
                f"DataFrame contains labels not in label_to_idx mapping: {missing_labels}"
            )
        self.return_recording_id = return_recording_id

    def __getitem__(self, idx: int):
        window = self.windows[idx]

        x = self._extract_window_tensor(window)
        x = self._apply_spec_augment(x)

        row = self.df.iloc[window.recording_idx]

        # Map scientific_name to contiguous index
        # This ensures CrossEntropyLoss receives valid [0, num_classes-1] indices
        y = torch.tensor(
            self.label_to_idx[row["scientific_name"]],
            dtype=torch.long,
        )
        if self.return_recording_id:
            return x, y, row['rc_id']
        return x, y

    def get_label_distribution(self) -> dict:
        """
        Get the distribution of labels in the dataset.
        Useful for class imbalance analysis.

        Returns:
            dict mapping idx → count
        """
        label_counts = Counter()
        for window in self.windows:
            row = self.df.iloc[window.recording_idx]
            label_idx = self.label_to_idx[row["scientific_name"]]
            label_counts[label_idx] += 1
        return dict(label_counts)
