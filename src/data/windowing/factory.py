# src/data/windowing/factory.py

from typing import Dict, Any
from .base import BaseSamplingStrategy
from .strategies import (
    RandomSamplingStrategy,
    CenterSamplingStrategy,
    SlidingSamplingStrategy,
    SlidingJitterSamplingStrategy,
)


def build_sampling_strategy(config: Dict[str, Any] = None) -> BaseSamplingStrategy:
    """
    Factory function to build a sampling strategy from a config dictionary.

    Args:
        config (dict): Configuration options, e.g.:
                       {"strategy": "sliding", "stride": 256}

    Returns:
        BaseSamplingStrategy: An instantiated sampling strategy.
    """
    config = config or {"strategy": "random"}
    strategy_type = config.get("strategy", "random").lower()

    if strategy_type == "random":
        return RandomSamplingStrategy()

    elif strategy_type == "center":
        return CenterSamplingStrategy()

    elif strategy_type == "sliding":
        stride = config.get("stride", 256)
        return SlidingSamplingStrategy(stride=stride)

    elif strategy_type == "sliding_jitter":
        stride = config.get("stride", 256)
        jitter_max = config.get("jitter_max", 16)
        return SlidingJitterSamplingStrategy(stride=stride, jitter_max=jitter_max)

    else:
        raise ValueError(f"Unknown sampling strategy: '{strategy_type}'")
