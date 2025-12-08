"""
Momentum and Statistical Feature Generators

This file provides implementations for:
1. Kaufman's Efficiency Ratio (ER) - for Momentum
2. Averaged Serial Correlation (ACF) - for Statistical analysis
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Any

from ..core.feature_generator import VectorizedFeatureGenerator, FeatureConfig, FeatureCategory

class KaufmanEfficiencyRatioGenerator(VectorizedFeatureGenerator):
    """
    Generator for Kaufman's Efficiency Ratio (ER).

    ER = Absolute Price Change / Sum of Absolute Price Changes

    Signal:
    - Near 1.0: Clean, efficient trend (Signal)
    - Near 0.0: Chaotic path that went nowhere (Noise)
    """

    def __init__(self, period: int = 10):
        config = FeatureConfig(
            name=f"kaufman_er_{period}",
            category=FeatureCategory.MOMENTUM,
            description=f"Kaufman's Efficiency Ratio over {period} periods",
            required_columns=["close"],
            default_lookback=period,
            min_lookback=period,
            max_lookback=period * 2,
            parameters={"period": period},
            matrix_optimized=True
        )
        super().__init__(config, enable_matrix_ops=True)
        self.period = period

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        close = data['close']

        # Calculate price changes (1-period diff)
        diffs = close.diff().abs()

        # Calculate signal (net direction)
        change = close.diff(self.period).abs()

        # Calculate noise (sum of individual moves)
        volatility = diffs.rolling(window=self.period).sum()

        # Calculate ER
        # Handle division by zero
        er = change / (volatility + 1e-8)

        return er

class AveragedACFGenerator(VectorizedFeatureGenerator):
    """
    Generator for Averaged Autocorrelation Function (ACF) at lags 1, 2, and 5.

    Combines serial correlation signals into a single feature.
    Positive ACF means "buying strength works" (trend persistence).
    Negative ACF means "buying strength fails" (mean reversion).
    """

    def __init__(self, window: int = 20):
        config = FeatureConfig(
            name=f"averaged_acf_{window}",
            category=FeatureCategory.ADVANCED_STATISTICAL,
            description=f"Averaged Autocorrelation (lags 1,2,5) over {window} periods",
            required_columns=["close"],
            default_lookback=window + 5,
            min_lookback=window + 5,
            max_lookback=window * 2,
            parameters={"window": window},
            matrix_optimized=True
        )
        super().__init__(config, enable_matrix_ops=True)
        self.window = window
        self.lags = [1, 2, 5]

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        close = data['close']
        returns = close.pct_change()

        # We need to calculate rolling autocorrelation for each lag
        # Pandas rolling().corr() can do this if we provide the shifted series

        acf_sum = pd.Series(0.0, index=data.index)
        valid_lags = 0

        for lag in self.lags:
            # Shifted returns for correlation
            shifted_returns = returns.shift(lag)

            # Rolling correlation
            # We align the calculation so that the result at time t uses returns up to t
            # corr(returns[t-w:t], returns[t-w-lag:t-lag])

            # Note: rolling().corr() between two series aligns by index
            # returns.rolling(window).corr(shifted_returns) computes corr of aligned windows
            # This is effectively autocorrelation at lag k computed over rolling window

            acf = returns.rolling(window=self.window).corr(shifted_returns)

            # Accumulate
            acf_sum = acf_sum.add(acf, fill_value=0)
            valid_lags += 1

        # Average
        avg_acf = acf_sum / valid_lags

        return avg_acf
