# src/data/windowing/factory.py
from typing import Dict, Any
from .base import BaseWindowStrategy
from .strategies import (
    SlidingWindowStrategy,
    RandomWindowStrategy,
    CenterWindowStrategy,
)


def build_window_strategy(config: Dict[str, Any] = None) -> BaseWindowStrategy:
    config = config or {"strategy": "sliding", "stride": 256}
    strategy_type = config.get("strategy", "sliding").lower()

    if strategy_type == "sliding":
        stride = config.get("stride", 256)
        return SlidingWindowStrategy(stride=stride)
    elif strategy_type == "random":
        return RandomWindowStrategy()
    elif strategy_type == "center":
        return CenterWindowStrategy()
    else:
        raise ValueError(f"Unknown windowing strategy: '{strategy_type}'")# src/data/windowing/factory.py
from typing import Dict, Any
from .base import BaseWindowStrategy
from .strategies import (
    SlidingWindowStrategy,
    RandomWindowStrategy,
    CenterWindowStrategy,
)


def build_window_strategy(config: Dict[str, Any] = None) -> BaseWindowStrategy:
    config = config or {"strategy": "sliding", "stride": 256}
    strategy_type = config.get("strategy", "sliding").lower()

    if strategy_type == "sliding":
        stride = config.get("stride", 256)
        return SlidingWindowStrategy(stride=stride)
    elif strategy_type == "random":
        return RandomWindowStrategy()
    elif strategy_type == "center":
        return CenterWindowStrategy()
    else:
        raise ValueError(f"Unknown windowing strategy: '{strategy_type}'")
