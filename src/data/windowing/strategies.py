import math
import random
import pandas as pd
from .base import BaseWindowStrategy
from .index import WindowIndex


class SlidingWindowStrategy(BaseWindowStrategy):
    """
    Expands the dataset index space across all valid temporal windows.
    
    Guarantees every recording is fully covered every epoch using the given stride.
    Clamps the final window to ensure trailing audio coverage without exceeding bounds.
    """

    def __init__(self, stride: int = 256):
        if stride <= 0:
            raise ValueError(f"Stride must be positive, got {stride}")
        self.stride = stride

    def build_window_index(
        self, 
        df: pd.DataFrame, 
        segment_size: int, 
        get_frames_fn: callable
    ) -> list[WindowIndex]:
        windows = []

        for idx, row in df.iterrows():
            total_frames = get_frames_fn(row, idx)

            if total_frames <= segment_size:
                # Audio shorter or equal to segment size -> single padded window [0, segment_size]
                windows.append(WindowIndex(
                    recording_idx=idx, 
                    start_frame=0, 
                    end_frame=segment_size
                ))
            else:
                max_start = total_frames - segment_size
                num_windows = math.ceil(max_start / self.stride) + 1

                for w in range(num_windows):
                    start = min(w * self.stride, max_start)
                    windows.append(WindowIndex(
                        recording_idx=idx,
                        start_frame=start,
                        end_frame=start + segment_size
                    ))

        return windows


class RandomWindowStrategy(BaseWindowStrategy):
    """
    Legacy 1-crop per recording mode. Randomly generates a single window index
    for each recording in the dataframe during index construction.
    """

    def build_window_index(
        self, 
        df: pd.DataFrame, 
        segment_size: int, 
        get_frames_fn: callable
    ) -> list[WindowIndex]:
        windows = []

        for idx, row in df.iterrows():
            total_frames = get_frames_fn(row, idx)
            max_start = max(0, total_frames - segment_size)
            start = random.randint(0, max_start) if max_start > 0 else 0

            windows.append(WindowIndex(
                recording_idx=idx,
                start_frame=start,
                end_frame=start + segment_size
            ))

        return windows


class CenterWindowStrategy(BaseWindowStrategy):
    """
    Generates exactly one centered window index per recording.
    """

    def build_window_index(
        self, 
        df: pd.DataFrame, 
        segment_size: int, 
        get_frames_fn: callable
    ) -> list[WindowIndex]:
        windows = []

        for idx, row in df.iterrows():
            total_frames = get_frames_fn(row, idx)
            max_start = max(0, total_frames - segment_size)
            start = max_start // 2

            windows.append(WindowIndex(
                recording_idx=idx,
                start_frame=start,
                end_frame=start + segment_size
            ))

        return windows
