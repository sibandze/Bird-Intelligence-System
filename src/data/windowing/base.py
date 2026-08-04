# src/data/windowing/base.py

from abc import ABC, abstractmethod
import pandas as pd
from .index import WindowIndex


class BaseWindowStrategy(ABC):
    """
    Abstract base class for temporal window selection.

    A strategy determines which temporal window from each recording
    should be exposed during a particular epoch.
    """

    @abstractmethod
    def build_window_index(
        self,
        df: pd.DataFrame,
        segment_size: int,
        get_frames_fn: callable,
        epoch: int = 0,
        is_train: bool = True,
    ) -> list[WindowIndex]:
        """
        Build the window index exposed for one epoch.

        Args:
            df:
                Recording metadata.

            segment_size:
                Number of spectrogram frames per training example.

            get_frames_fn:
                Callback: fn(row, idx) -> total number of frames.

            epoch:
                Current training epoch.

            is_train:
                Whether the dataset is being used for training.

        Returns:
            One or more WindowIndex objects per recording.
        """
        pass
