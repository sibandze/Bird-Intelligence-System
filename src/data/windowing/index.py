# src/data/windowing/index.py

from dataclasses import dataclass

@dataclass(frozen=True)
class WindowIndex:
    """Represents a discrete temporal slice within a specific recording."""
    recording_idx: int
    start_frame: int
    end_frame: int

    @property
    def duration_frames(self) -> int:
        return self.end_frame - self.start_frame
