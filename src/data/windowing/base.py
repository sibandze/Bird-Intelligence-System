# src/data/windowing/base.py
from abc import ABC, abstractmethod

class BaseSamplingStrategy(ABC):
    """Abstract base class for audio spectrogram sampling strategies."""

    @abstractmethod
    def get_start_frame(
        self,
        total_frames: int,
        segment_size: int,
        epoch: int = 0,
        is_train: bool = True
    ) -> int:
        """
        Calculates the start frame index for cropping a spectrogram.

        Args:
            total_frames (int): Total number of time frames in the spectrogram (T).
            segment_size (int): Expected window size in frames.
            epoch (int): Current epoch number. Defaults to 0.
            is_train (bool): Whether in training mode.

        Returns:
            int: The starting frame index for the crop.
        """
        pass
