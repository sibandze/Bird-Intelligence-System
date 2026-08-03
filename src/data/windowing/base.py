# src/data/windowing/base.py
from abc import ABC, abstractmethod
import pandas as pd
from .index import WindowIndex


class BaseWindowStrategy(ABC):
    """
    Abstract Base Class for windowing strategies that build an index space 
    of discrete temporal windows from a dataframe of audio recordings.
    """

    @abstractmethod
    def build_window_index(
        self, 
        df: pd.DataFrame, 
        segment_size: int, 
        get_frames_fn: callable
    ) -> list[WindowIndex]:
        """
        Builds the complete list of WindowIndex items across all recordings.

        Args:
            df (pd.DataFrame): Audio recording metadata.
            segment_size (int): Temporal window length in frames.
            get_frames_fn (callable): Callback `fn(row, idx) -> int` to obtain 
                                       the total frame count for a recording.

        Returns:
            list[WindowIndex]: Complete index space mapping.
        """
        pass
