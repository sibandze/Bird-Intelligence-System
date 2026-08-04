# src/data/windowing/factory.py

from typing import Dict, Any

from .base import BaseWindowStrategy
from .strategies import (
    SlidingWindowStrategy,
    RandomWindowStrategy,
    CenterWindowStrategy,
)


def build_window_strategy(
    config: Dict[str, Any] = None,
) -> BaseWindowStrategy:

    config = config or {
        "strategy": "sliding",
        "stride": 256,
    }

    strategy_type = config.get(
        "strategy",
        "sliding",
    ).lower()

    if strategy_type == "sliding":

        return SlidingWindowStrategy(
            stride=config.get("stride", 256)
        )

    if strategy_type == "random":

        return RandomWindowStrategy()

    if strategy_type == "center":

        return CenterWindowStrategy()

    raise ValueError(
        f"Unknown windowing strategy: '{strategy_type}'"
    )
