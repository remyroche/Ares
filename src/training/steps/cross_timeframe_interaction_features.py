from __future__ import annotations
'\nRefactored cross-timeframe and interaction feature generation with reduced complexity.\nThis module breaks down the high-complexity feature generation methods into smaller,\nfocused functions with proper type annotations.\n'
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from enum import Enum
import numpy as np
import pandas as pd


@dataclass
class CrossTimeframeConfig:
    """Configuration for cross-timeframe feature generation"""
    momentum_timeframes: list[int] = None
    volatility_timeframes: list[int] = None
    volume_timeframes: list[int] = None
    rsi_periods: list[int] = None
    macd_fast_periods: list[int] = None
    macd_slow_periods: list[int] = None
    bb_windows: list[int] = None
    bb_stds: list[float] = None
    min_data_points: int = 100
    variance_threshold: float = 1e-12
    parallel_processing: bool = True
    max_workers: int = 4

    def __post_init__(self) -> None:
        """Initialize default values"""
        if self.momentum_timeframes is None:
            self.momentum_timeframes = [1, 3, 5, 10, 15, 20]
        if self.volatility_timeframes is None:
            self.volatility_timeframes = [3, 5, 10, 15, 20, 30]
        if self.volume_timeframes is None:
            self.volume_timeframes = [5, 10, 15, 30]
        if self.rsi_periods is None:
            self.rsi_periods = [3, 5, 10, 14, 21]
        if self.macd_fast_periods is None:
            self.macd_fast_periods = [3, 5, 8, 12]
        if self.macd_slow_periods is None:
            self.macd_slow_periods = [10, 15, 20, 26]
        if self.bb_windows is None:
            self.bb_windows = [10, 15, 20]
        if self.bb_stds is None:
            self.bb_stds = [1.0, 1.5, 2.0]

@dataclass
class InteractionConfig:
    """Configuration for interaction feature generation"""
    max_interaction_depth: int = 2
    top_k_features: int = 50
    correlation_threshold: float = 0.95
    variance_threshold: float = 1e-12
    polynomial_degree: int = 2
    include_ratios: bool = True
    include_differences: bool = True
    include_products: bool = True
    parallel_processing: bool = True
    max_workers: int = 4

