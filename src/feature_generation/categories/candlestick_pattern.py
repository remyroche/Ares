"""
Candlestick Pattern Feature Generator

This module provides feature generators for candlestick pattern recognition,
including doji, hammer, engulfing patterns, and other candlestick formations.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional

from ..core.feature_generator import (
    FeatureGenerator, 
    FeatureConfig, 
    FeatureCategory,
    VectorizedFeatureGenerator
)

class CandlestickPatternFeatureGenerator(VectorizedFeatureGenerator):
    """Feature generator for candlestick pattern-based features."""
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        if config is None:
            config = self._create_default_config()
        super().__init__(config, enable_matrix_ops=True)
    
    @classmethod
    def _create_default_config(cls) -> FeatureConfig:
        return FeatureConfig(
            name="candlestick_pattern_features",
            category=FeatureCategory.CANDLESTICK_PATTERN,
            description="Comprehensive candlestick pattern features including doji, hammer, and engulfing patterns",
            required_columns=["open", "high", "low", "close"],
            default_lookback=1,
            min_lookback=1,
            max_lookback=1,
            parameters={
                "patterns": ["doji", "hammer", "engulfing"],
                "body_threshold": 0.1
            },
            matrix_optimized=True,
            gpu_accelerated=False
        )
    
    @classmethod
    def create_default(cls) -> 'CandlestickPatternFeatureGenerator':
        return cls()
    
    @staticmethod
    def _compute_pattern_signals(data: pd.DataFrame, body_threshold: float) -> Dict[str, pd.Series]:
        open_prices = data['open'].astype(float)
        close_prices = data['close'].astype(float)
        high_prices = data['high'].astype(float)
        low_prices = data['low'].astype(float)

        body = close_prices - open_prices
        body_size = body.abs()
        full_range = (high_prices - low_prices).replace(0.0, np.nan)
        upper_shadow = high_prices - pd.concat([open_prices, close_prices], axis=1).max(axis=1)
        lower_shadow = pd.concat([open_prices, close_prices], axis=1).min(axis=1) - low_prices

        body_ratio = (body_size / full_range).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        direction = np.sign(body).replace(0.0, np.sign(close_prices.diff().fillna(0.0)))
        min_body_threshold = max(body_threshold, 1e-6)

        doji_strength = (1.0 - (body_ratio / min_body_threshold)).clip(0.0, 1.0)
        doji_signal = (direction * doji_strength).fillna(0.0)

        small_body = body_ratio <= 0.5
        long_lower = lower_shadow >= (body_size * 2.0)
        short_upper = upper_shadow <= (body_size * 0.5)
        hammer_strength = ((lower_shadow / (full_range.replace(0.0, np.nan))).clip(0.0, 1.0)).fillna(0.0)
        hammer_signal = hammer_strength.where(small_body & long_lower & short_upper & (body > 0.0), 0.0)

        long_upper = upper_shadow >= (body_size * 2.0)
        short_lower = lower_shadow <= (body_size * 0.5)
        shooting_strength = ((upper_shadow / full_range.replace(0.0, np.nan)).clip(0.0, 1.0)).fillna(0.0)
        shooting_signal = -shooting_strength.where(small_body & long_upper & short_lower & (body < 0.0), 0.0)

        prev_open = open_prices.shift(1)
        prev_close = close_prices.shift(1)
        prev_body = prev_close - prev_open
        prev_body_size = prev_body.abs().replace(0.0, np.nan)
        magnitude = (body_size / prev_body_size).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        magnitude = np.tanh(magnitude)
        bullish_engulf = (body > 0.0) & (prev_body < 0.0) & (close_prices >= prev_open) & (open_prices <= prev_close)
        bearish_engulf = (body < 0.0) & (prev_body > 0.0) & (close_prices <= prev_open) & (open_prices >= prev_close)
        engulfing_signal = pd.Series(0.0, index=data.index, dtype=float)
        engulfing_signal = engulfing_signal.where(~bullish_engulf, magnitude)
        engulfing_signal = engulfing_signal.where(~bearish_engulf, -magnitude)

        return {
            'doji': doji_signal.rename('candlestick_doji'),
            'hammer': hammer_signal.rename('candlestick_hammer'),
            'shooting_star': shooting_signal.rename('candlestick_shooting_star'),
            'engulfing': engulfing_signal.rename('candlestick_engulfing'),
        }

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        if data.empty:
            return pd.Series(dtype=float, index=data.index, name='candlestick_pattern_signal')

        params = self.config.parameters or {}
        patterns = [pattern.lower() for pattern in params.get('patterns', ['doji', 'hammer', 'engulfing'])]
        body_threshold = float(params.get('body_threshold', 0.1))

        signals = self._compute_pattern_signals(data, body_threshold)
        aggregated = pd.Series(0.0, index=data.index, dtype=float)
        contributions = 0

        for pattern in patterns:
            series = signals.get(pattern)
            if series is None:
                continue
            if (series != 0).any():
                contributions += 1
            aggregated = aggregated.add(series, fill_value=0.0)

        if not contributions:
            return aggregated.rename('candlestick_pattern_signal')

        signal = aggregated / float(contributions)
        signal = signal.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return signal.clip(-1.0, 1.0).rename('candlestick_pattern_signal')

class CandlestickPatternSignalGenerator(FeatureGenerator):
    """Single candlestick pattern signal generator."""

    def __init__(self, pattern: str, body_threshold: float = 0.1):
        pattern_lower = pattern.lower()
        config = FeatureConfig(
            name=f"candlestick_{pattern_lower}",
            category=FeatureCategory.CANDLESTICK_PATTERN,
            description=f"Signal strength for the {pattern_lower} candlestick pattern",
            required_columns=["open", "high", "low", "close"],
            default_lookback=2,
            min_lookback=1,
            max_lookback=5,
            parameters={'pattern': pattern_lower, 'body_threshold': body_threshold},
        )
        super().__init__(config)
        self.pattern = pattern_lower
        self.body_threshold = body_threshold

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        if data.empty:
            return pd.Series(dtype=float, index=data.index, name=self.config.name)
        signals = CandlestickPatternFeatureGenerator._compute_pattern_signals(data, self.body_threshold)
        series = signals.get(self.pattern)
        if series is None:
            return pd.Series(0.0, index=data.index, name=self.config.name)
        return series.rename(self.config.name)


def create_candlestick_pattern_generators(patterns: List[str] = None) -> List[FeatureGenerator]:
    """Create a set of candlestick pattern feature generators."""
    if patterns is None:
        patterns = ["doji", "hammer", "engulfing", "shooting_star"]

    generators: List[FeatureGenerator] = [CandlestickPatternFeatureGenerator()]
    for pattern in patterns:
        generators.append(CandlestickPatternSignalGenerator(pattern))
    return generators

def create_default_candlestick_pattern_generators() -> List[FeatureGenerator]:
    return create_candlestick_pattern_generators()