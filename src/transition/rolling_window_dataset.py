# src/transition/rolling_window_dataset.py

from src.transition.path_targets import PathTargetEngineer
from src.transition.state_sequence_builder import StateSequenceBuilder
from src.utils.logger import system_logger
from typing import Any
from dataclasses import dataclass
import numpy as np
import pandas as pd

FEATURE_POOL_COLUMNS , [
    "log_returns",
    "volatility_20",
    "volume_ratio",
    "rsi",
    "macd",
    "macd_signal",
    "macd_histogram",
    "bb_position",
    "bb_width",
    "atr",
    "volatility_regime",
    "volatility_acceleration",
]


@dataclass
class RollingWindowConfig:
    pre_window: int
    post_window: int
    onset_horizon_bars: int
    end_horizon_bars: int
    include_direction_horizons: list[int]
    max_samples: int | None


