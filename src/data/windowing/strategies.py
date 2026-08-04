# src/data/windowing/strategies.py

import math
import random
import pandas as pd

from .base import BaseWindowStrategy
from .index import WindowIndex


class SlidingWindowStrategy(BaseWindowStrategy):
    """
    Exposes one temporal window per recording per epoch.

    The selected window advances by `stride` frames each epoch.

    Example:

        Epoch 0 -> window 0
        Epoch 1 -> window 1
        Epoch 2 -> window 2
        ...

    Once all valid windows have been visited, the strategy cycles
    back through the recording.

    This allows long recordings to be progressively exposed to the
    model across epochs without increasing dataset size.
    """

    def __init__(self, stride: int = 256):
        if stride <= 0:
            raise ValueError(f"Stride must be positive, got {stride}")

        self.stride = stride

    def _build_starts(
        self,
        total_frames: int,
        segment_size: int,
    ) -> list[int]:
        """
        Build all valid temporal window start positions.

        The final window is explicitly included so that the tail of
        the recording is covered.
        """

        if total_frames <= segment_size:
            return [0]

        max_start = total_frames - segment_size

        starts = list(range(0, max_start + 1, self.stride))

        # Guarantee final temporal region is covered.
        if starts[-1] != max_start:
            starts.append(max_start)

        return starts

    def build_window_index(
        self,
        df: pd.DataFrame,
        segment_size: int,
        get_frames_fn: callable,
        epoch: int = 0,
        is_train: bool = True,
    ) -> list[WindowIndex]:

        windows = []

        for idx, row in df.iterrows():

            total_frames = get_frames_fn(row, idx)

            starts = self._build_starts(
                total_frames=total_frames,
                segment_size=segment_size,
            )

            if not is_train:
                # Deterministic center window for validation/test.
                start = starts[len(starts) // 2]

            else:
                # One temporal window per recording per epoch.
                window_idx = epoch % len(starts)
                start = starts[window_idx]

            end = min(
                start + segment_size,
                total_frames,
            )

            windows.append(
                WindowIndex(
                    recording_idx=idx,
                    start_frame=start,
                    end_frame=end,
                )
            )

        return windows


class RandomWindowStrategy(BaseWindowStrategy):
    """
    Selects one random temporal window per recording.

    A new window is selected whenever the epoch index is rebuilt.
    """

    def build_window_index(
        self,
        df: pd.DataFrame,
        segment_size: int,
        get_frames_fn: callable,
        epoch: int = 0,
        is_train: bool = True,
    ) -> list[WindowIndex]:

        windows = []

        for idx, row in df.iterrows():

            total_frames = get_frames_fn(row, idx)

            max_start = max(0, total_frames - segment_size)

            if is_train and max_start > 0:
                start = random.randint(0, max_start)
            else:
                start = max_start // 2

            end = min(
                start + segment_size,
                total_frames,
            )

            windows.append(
                WindowIndex(
                    recording_idx=idx,
                    start_frame=start,
                    end_frame=end,
                )
            )

        return windows


class CenterWindowStrategy(BaseWindowStrategy):
    """
    Selects one deterministic center window per recording.
    """

    def build_window_index(
        self,
        df: pd.DataFrame,
        segment_size: int,
        get_frames_fn: callable,
        epoch: int = 0,
        is_train: bool = False,
    ) -> list[WindowIndex]:

        windows = []

        for idx, row in df.iterrows():

            total_frames = get_frames_fn(row, idx)

            max_start = max(0, total_frames - segment_size)

            start = max_start // 2

            end = min(
                start + segment_size,
                total_frames,
            )

            windows.append(
                WindowIndex(
                    recording_idx=idx,
                    start_frame=start,
                    end_frame=end,
                )
            )

        return windows
