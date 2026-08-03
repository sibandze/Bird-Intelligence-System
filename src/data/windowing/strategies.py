# src/data/windowing/strategies.py

import random
from .base import BaseSamplingStrategy


class RandomSamplingStrategy(BaseSamplingStrategy):
    """
    Baseline sampling strategy.
    Randomly crops during training; center crops during evaluation.
    """

    def get_start_frame(
        self,
        total_frames: int,
        segment_size: int,
        epoch: int = 0,
        is_train: bool = True
    ) -> int:
        max_start = total_frames - segment_size
        if max_start <= 0:
            return 0

        if is_train:
            return random.randint(0, max_start)
        else:
            return max_start // 2


class CenterSamplingStrategy(BaseSamplingStrategy):
    """Always crops the deterministic center of the spectrogram."""

    def get_start_frame(
        self,
        total_frames: int,
        segment_size: int,
        epoch: int = 0,
        is_train: bool = True
    ) -> int:
        max_start = total_frames - segment_size
        if max_start <= 0:
            return 0
        return max_start // 2


class SlidingSamplingStrategy(BaseSamplingStrategy):
    """
    Deterministic sliding window sampling across epochs with clamping.
    
    Advances window start by `stride` frames every epoch.
    Clamps to `max_start` when the window exceeds audio bounds.
    """

    def __init__(self, stride: int = 256):
        self.stride = stride

    def get_start_frame(
        self,
        total_frames: int,
        segment_size: int,
        epoch: int = 0,
        is_train: bool = True
    ) -> int:
        max_start = total_frames - segment_size
        if max_start <= 0:
            return 0

        # Center crop during evaluation to ensure deterministic evaluation
        if not is_train:
            return max_start // 2

        # Compute sliding start based on current epoch
        start = epoch * self.stride

        # Handle End-of-Recording via Clamping
        return min(start, max_start)


class SlidingJitterSamplingStrategy(BaseSamplingStrategy):
    """
    Sliding window with stochastic frame jitter.
    
    Calculates sliding base position from epoch and adds a random frame offset 
    within [-jitter_max, +jitter_max].
    """

    def __init__(self, stride: int = 256, jitter_max: int = 16):
        self.stride = stride
        self.jitter_max = jitter_max

    def get_start_frame(
        self,
        total_frames: int,
        segment_size: int,
        epoch: int = 0,
        is_train: bool = True
    ) -> int:
        max_start = total_frames - segment_size
        if max_start <= 0:
            return 0

        if not is_train:
            return max_start // 2

        # Compute deterministic sliding base position
        base_start = epoch * self.stride

        # Add random jitter offset
        jitter = random.randint(-self.jitter_max, self.jitter_max)
        start = base_start + jitter

        # Clamp within valid recording boundaries [0, max_start]
        return max(0, min(start, max_start))
