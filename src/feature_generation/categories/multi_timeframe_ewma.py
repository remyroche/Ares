"""
Multi-Timeframe EWMA Feature Generator

This module provides comprehensive EWMA-based features across multiple timeframes,
inspired by rolling_hmm_clustering. Includes returns, volatility, trend, and volume
features with EWMA smoothing for better regime detection.

Features include:
- EWMA returns and cumulative returns
- EWMA volatility (realized, log-transformed)
- EWMA trend (SMA, slopes, Sharpe ratios)
- EWMA volume (ratios, z-scores, weighted returns)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from src.feature_generation.core.feature_generator import (
    FeatureGenerator,
    VectorizedFeatureGenerator,
    FeatureResult
)


class MultiTimeframeEWMAReturnsGenerator(VectorizedFeatureGenerator):
    """
    Generate EWMA-based returns features across multiple timeframes.

    Features:
    - ewma_returns_{window}: EWMA of returns
    - cum_returns_{window}: Cumulative returns
    - ewma_returns_diff_{short}_{long}: EWMA spread
    """

    def __init__(
        self,
        windows: Optional[List[int]] = None,
        use_log_returns: bool = True,
        name: Optional[str] = None
    ):
        """
        Initialize Multi-Timeframe EWMA Returns Generator.

        Args:
            windows: List of EWMA windows (default: [8, 12, 16, 20, 24])
            use_log_returns: Use log returns instead of simple returns
            name: Optional custom name
        """
        super().__init__(name=name or "multi_timeframe_ewma_returns")
        self.windows = windows or [8, 12, 16, 20, 24]
        self.use_log_returns = use_log_returns

    def generate_features(self, data: pd.DataFrame, **kwargs) -> Dict[str, np.ndarray]:
        """Generate EWMA returns features."""
        features = {}

        close = data['close']

        # Calculate returns
        if self.use_log_returns:
            returns = np.log(close / close.shift(1))
        else:
            returns = close.pct_change()

        # EWMA returns for each window
        for window in self.windows:
            ewma_ret = returns.ewm(span=window, adjust=False).mean()
            features[f'ewma_returns_{window}'] = ewma_ret.values

            # Cumulative returns
            cum_ret = returns.rolling(window).sum()
            features[f'cum_returns_{window}'] = cum_ret.values

        # EWMA spreads (short - long)
        for i, short_window in enumerate(self.windows[:-1]):
            long_window = self.windows[i + 1]
            ewma_short = returns.ewm(span=short_window, adjust=False).mean()
            ewma_long = returns.ewm(span=long_window, adjust=False).mean()
            features[f'ewma_returns_diff_{short_window}_{long_window}'] = (ewma_short - ewma_long).values

        return features


class MultiTimeframeEWMAVolatilityGenerator(VectorizedFeatureGenerator):
    """
    Generate EWMA-based volatility features across multiple timeframes.

    Features:
    - ewma_volatility_{window}: EWMA of rolling volatility
    - volatility_ratio_{short}_{long}: Short/long volatility ratio
    - realized_volatility_{window}: Parkinson high-low estimator
    - log_volatility_{window}: Log-transformed volatility for stability
    """

    def __init__(
        self,
        windows: Optional[List[int]] = None,
        use_log_returns: bool = True,
        name: Optional[str] = None
    ):
        """
        Initialize Multi-Timeframe EWMA Volatility Generator.

        Args:
            windows: List of EWMA windows (default: [8, 12, 16, 20, 24])
            use_log_returns: Use log returns for volatility calculation
            name: Optional custom name
        """
        super().__init__(name=name or "multi_timeframe_ewma_volatility")
        self.windows = windows or [8, 12, 16, 20, 24]
        self.use_log_returns = use_log_returns

    def generate_features(self, data: pd.DataFrame, **kwargs) -> Dict[str, np.ndarray]:
        """Generate EWMA volatility features."""
        features = {}

        close = data['close']
        high = data['high']
        low = data['low']

        # Calculate returns with safe division
        if self.use_log_returns:
            # Safe log returns calculation with division by zero protection
            close_shifted = close.shift(1)
            returns = np.where(close_shifted > 0, np.log(close / close_shifted), 0.0)
        else:
            # Safe percentage change with division by zero protection
            close_shifted = close.shift(1)
            returns = np.where(close_shifted > 0, (close - close_shifted) / close_shifted, 0.0)

        # EWMA volatility for each window
        for window in self.windows:
            # EWMA volatility
            ewma_vol = returns.ewm(span=window, adjust=False).std()
            features[f'ewma_volatility_{window}'] = ewma_vol.values

            # Realized volatility (Parkinson estimator)
            hl_ratio = (high / low).apply(np.log)
            realized_vol = hl_ratio.rolling(window).std() * np.sqrt(252)
            features[f'realized_volatility_{window}'] = realized_vol.values

            # Log volatility (stabilizes scale)
            rolling_vol = returns.rolling(window).std()
            log_vol = np.log(rolling_vol + 1e-8)
            features[f'log_volatility_{window}'] = log_vol.values

        # Volatility ratios (short / long)
        for i, short_window in enumerate(self.windows[:-1]):
            long_window = self.windows[i + 1]
            vol_short = returns.rolling(short_window).std()
            vol_long = returns.rolling(long_window).std()
            vol_ratio = vol_short / (vol_long + 1e-8)
            features[f'volatility_ratio_{short_window}_{long_window}'] = vol_ratio.values

        return features


class MultiTimeframeEWMATrendGenerator(VectorizedFeatureGenerator):
    """
    Generate EWMA-based trend features across multiple timeframes.

    Features:
    - ewma_price_{window}: EWMA of price
    - ewma_diff_{short}_{long}: EWMA price spread
    - price_to_ewma_{window}: Price relative to EWMA
    - sma_slope_{window}: SMA slope for trend strength
    - rolling_sharpe_{window}: Rolling Sharpe ratio
    - rolling_zscore_{window}: Mean-reversion indicator
    """

    def __init__(
        self,
        windows: Optional[List[int]] = None,
        use_log_returns: bool = True,
        name: Optional[str] = None
    ):
        """
        Initialize Multi-Timeframe EWMA Trend Generator.

        Args:
            windows: List of EWMA windows (default: [8, 12, 16, 20, 24])
            use_log_returns: Use log returns for Sharpe calculation
            name: Optional custom name
        """
        super().__init__(name=name or "multi_timeframe_ewma_trend")
        self.windows = windows or [8, 12, 16, 20, 24]
        self.use_log_returns = use_log_returns

    def generate_features(self, data: pd.DataFrame, **kwargs) -> Dict[str, np.ndarray]:
        """Generate EWMA trend features."""
        features = {}

        close = data['close']

        # Calculate returns
        if self.use_log_returns:
            returns = np.log(close / close.shift(1))
        else:
            returns = close.pct_change()

        # EWMA price for each window
        for window in self.windows:
            # EWMA price
            ewma_price = close.ewm(span=window, adjust=False).mean()
            features[f'ewma_price_{window}'] = ewma_price.values

            # Price to EWMA ratio
            price_to_ewma = close / (ewma_price + 1e-8)
            features[f'price_to_ewma_{window}'] = price_to_ewma.values

            # SMA and slope
            sma = close.rolling(window).mean()
            sma_slope = sma.diff(window // 2)
            features[f'sma_{window}'] = sma.values
            features[f'sma_slope_{window}'] = sma_slope.values

            # Price to SMA ratio
            price_to_sma = close / (sma + 1e-8)
            features[f'price_to_sma_{window}'] = price_to_sma.values

            # Rolling Sharpe ratio (annualized)
            rolling_mean = returns.rolling(window).mean()
            rolling_std = returns.rolling(window).std()
            rolling_sharpe = (rolling_mean / (rolling_std + 1e-8)) * np.sqrt(252)
            features[f'rolling_sharpe_{window}'] = rolling_sharpe.values

            # Rolling Z-score (mean-reversion indicator)
            rolling_zscore = (close - sma) / (rolling_std * close + 1e-8)
            features[f'rolling_zscore_{window}'] = rolling_zscore.values

        # EWMA spreads (short - long)
        for i, short_window in enumerate(self.windows[:-1]):
            long_window = self.windows[i + 1]
            ewma_short = close.ewm(span=short_window, adjust=False).mean()
            ewma_long = close.ewm(span=long_window, adjust=False).mean()
            features[f'ewma_diff_{short_window}_{long_window}'] = (ewma_short - ewma_long).values

            # SMA spreads
            sma_short = close.rolling(short_window).mean()
            sma_long = close.rolling(long_window).mean()
            features[f'sma_diff_{short_window}_{long_window}'] = (sma_short - sma_long).values

        return features


class MultiTimeframeEWMAVolumeGenerator(VectorizedFeatureGenerator):
    """
    Generate EWMA-based volume features across multiple timeframes.

    Features:
    - ewma_volume_{window}: EWMA of volume
    - volume_ratio_{window}: Current volume / EWMA volume
    - volume_zscore_{window}: Volume Z-score
    - ewma_volume_change_{window}: EWMA of volume changes
    - volume_weighted_returns_{window}: Returns weighted by volume
    """

    def __init__(
        self,
        windows: Optional[List[int]] = None,
        use_log_returns: bool = True,
        name: Optional[str] = None
    ):
        """
        Initialize Multi-Timeframe EWMA Volume Generator.

        Args:
            windows: List of EWMA windows (default: [8, 12, 16, 20, 24])
            use_log_returns: Use log returns for volume-weighted returns
            name: Optional custom name
        """
        super().__init__(name=name or "multi_timeframe_ewma_volume")
        self.windows = windows or [8, 12, 16, 20, 24]
        self.use_log_returns = use_log_returns

    def generate_features(self, data: pd.DataFrame, **kwargs) -> Dict[str, np.ndarray]:
        """Generate EWMA volume features."""
        features = {}

        volume = data['volume']
        close = data['close']

        # Log volume (stabilizes scale)
        log_volume = np.log(volume + 1)
        features['log_volume'] = log_volume.values

        # Calculate returns
        if self.use_log_returns:
            returns = np.log(close / close.shift(1))
        else:
            returns = close.pct_change()

        # Volume changes
        volume_change = volume.pct_change()
        features['volume_change'] = volume_change.values

        # EWMA volume for each window
        for window in self.windows:
            # EWMA volume
            ewma_vol = volume.ewm(span=window, adjust=False).mean()
            features[f'ewma_volume_{window}'] = ewma_vol.values

            # Volume ratio (current / EWMA)
            vol_ratio = volume / (ewma_vol + 1e-8)
            features[f'volume_ratio_{window}'] = vol_ratio.values

            # Volume Z-score
            vol_mean = volume.rolling(window).mean()
            vol_std = volume.rolling(window).std()
            vol_zscore = (volume - vol_mean) / (vol_std + 1e-8)
            features[f'volume_zscore_{window}'] = vol_zscore.values

            # EWMA of volume changes
            ewma_vol_change = volume_change.ewm(span=window, adjust=False).mean()
            features[f'ewma_volume_change_{window}'] = ewma_vol_change.values

            # Volume-weighted returns
            avg_vol = volume.rolling(window).mean()
            vol_weighted_ret = returns * (volume / (avg_vol + 1e-8))
            features[f'volume_weighted_returns_{window}'] = vol_weighted_ret.values

        return features


# Export all generators
__all__ = [
    'MultiTimeframeEWMAReturnsGenerator',
    'MultiTimeframeEWMAVolatilityGenerator',
    'MultiTimeframeEWMATrendGenerator',
    'MultiTimeframeEWMAVolumeGenerator',
]
