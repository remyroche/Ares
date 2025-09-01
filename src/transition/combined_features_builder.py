# src/transition/combined_features_builder.py

from src.utils.logger import system_logger
from typing import Any
from dataclasses import dataclass
import numpy as np
import pandas as pd

REQUIRED_FEATURES = [
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
class CombinedFeaturesConfig:
    volatility_threshold: float = 0.02


