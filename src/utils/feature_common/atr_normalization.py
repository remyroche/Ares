"""
ATR Normalization for Spatial Distance Features

This module provides ATR-based normalization for features that measure spatial
distances or levels (e.g., trend distance, breakout levels, Bollinger Band width,
candle size, stop loss, wick size, FVG size).

ATR normalization is more appropriate for these features than winsorized z-score
because it normalizes by the actual price volatility, making the features comparable
across different volatility regimes and price levels.

For momentum/speed features, continue using winsorized_zscore_normalize.
"""

import pandas as pd
import numpy as np
from typing import Union, Optional, Tuple


def calculate_atr(
    high: Union[pd.Series, np.ndarray],
    low: Union[pd.Series, np.ndarray],
    close: Union[pd.Series, np.ndarray],
    window: int = 14,
    min_periods: int = 1
) -> Union[pd.Series, np.ndarray]:
    """
    Calculate Average True Range (ATR).

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        window: ATR window size (default: 14)
        min_periods: Minimum periods for rolling window

    Returns:
        ATR values
    """
    # Convert to Series if needed
    if not isinstance(high, pd.Series):
        high = pd.Series(high)
    if not isinstance(low, pd.Series):
        low = pd.Series(low)
    if not isinstance(close, pd.Series):
        close = pd.Series(close)

    # Calculate True Range components
    prev_close = close.shift(1)

    # TR = max(high - low, abs(high - prev_close), abs(low - prev_close))
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    # Take maximum of the three
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Fill first value with high - low
    tr = tr.fillna(high - low)

    # Calculate ATR using EMA for smoothness
    atr = tr.ewm(span=window, min_periods=min_periods, adjust=False).mean()

    return atr


def atr_normalize(
    data: Union[pd.DataFrame, pd.Series],
    high: Union[pd.Series, np.ndarray],
    low: Union[pd.Series, np.ndarray],
    close: Union[pd.Series, np.ndarray],
    window: int = 14,
    min_atr: float = 1e-8,
    return_atr: bool = False
) -> Union[pd.DataFrame, pd.Series, Tuple]:
    """
    Normalize features by ATR for spatial distance/level features.

    This normalization is appropriate for features measuring:
    - Trend distance
    - Breakout levels
    - Bollinger Band width
    - Candle size
    - Stop loss distance
    - Wick size
    - Fair Value Gap (FVG) size
    - Support/Resistance distance

    Formula: normalized = feature / ATR

    Args:
        data: Feature data to normalize (Series or DataFrame)
        high: High prices for ATR calculation
        low: Low prices for ATR calculation
        close: Close prices for ATR calculation
        window: ATR window size (default: 14)
        min_atr: Minimum ATR value to avoid division by zero
        return_atr: If True, also return the ATR values

    Returns:
        Normalized data (same type as input)
        If return_atr=True, returns (normalized_data, atr)

    Examples:
        >>> # Normalize Bollinger Band width by ATR
        >>> bb_width_normalized = atr_normalize(bb_width, high, low, close)
        >>>
        >>> # Normalize multiple spatial features
        >>> spatial_features = df[['trend_distance', 'bb_width', 'candle_size']]
        >>> normalized = atr_normalize(spatial_features, high, low, close)
    """
    # Calculate ATR
    atr = calculate_atr(high, low, close, window=window)

    # Ensure ATR is not too small to avoid division issues
    atr_safe = atr.clip(lower=min_atr)

    # Normalize data
    if isinstance(data, pd.DataFrame):
        # Normalize each column by ATR
        normalized = data.div(atr_safe, axis=0)
        normalized = normalized.fillna(0.0)
    elif isinstance(data, pd.Series):
        # Normalize series by ATR
        normalized = data / atr_safe
        normalized = normalized.fillna(0.0)
    else:
        # Convert to Series, normalize, then convert back
        data_series = pd.Series(data)
        normalized_series = data_series / atr_safe
        normalized = normalized_series.fillna(0.0).values

    if return_atr:
        return normalized, atr
    else:
        return normalized


def atr_percent_normalize(
    data: Union[pd.DataFrame, pd.Series],
    close: Union[pd.Series, np.ndarray],
    high: Union[pd.Series, np.ndarray],
    low: Union[pd.Series, np.ndarray],
    window: int = 14,
    min_atr: float = 1e-8,
    min_close: float = 1e-8
) -> Union[pd.DataFrame, pd.Series]:
    """
    Normalize features by ATR as percentage of close price.

    This is useful for comparing features across different price levels.

    Formula: normalized = (feature / close) / (ATR / close) = feature / ATR

    Args:
        data: Feature data to normalize
        close: Close prices
        high: High prices for ATR calculation
        low: Low prices for ATR calculation
        window: ATR window size
        min_atr: Minimum ATR value to avoid division by zero
        min_close: Minimum close price to avoid division by zero

    Returns:
        Normalized data (same type as input)
    """
    # Calculate ATR
    atr = calculate_atr(high, low, close, window=window)

    # Convert close to Series if needed
    if not isinstance(close, pd.Series):
        close = pd.Series(close)

    # Calculate ATR as percentage of close
    close_safe = close.clip(lower=min_close)
    atr_percent = atr / close_safe
    atr_percent_safe = atr_percent.clip(lower=min_atr / close_safe.mean())

    # Normalize data
    if isinstance(data, pd.DataFrame):
        # Convert data to percentage of close first
        data_percent = data.div(close_safe, axis=0)
        # Then normalize by ATR percent
        normalized = data_percent.div(atr_percent_safe, axis=0)
        normalized = normalized.fillna(0.0)
    elif isinstance(data, pd.Series):
        # Convert to percentage of close
        data_percent = data / close_safe
        # Normalize by ATR percent
        normalized = data_percent / atr_percent_safe
        normalized = normalized.fillna(0.0)
    else:
        # Convert to Series, normalize, then convert back
        data_series = pd.Series(data)
        close_safe_series = pd.Series(close_safe)
        data_percent = data_series / close_safe_series
        normalized_series = data_percent / atr_percent_safe
        normalized = normalized_series.fillna(0.0).values

    return normalized


# List of features that should use ATR normalization instead of winsorized z-score
# These are features measuring spatial distance or levels
ATR_NORMALIZED_FEATURES = [
    # Trend features (distance/levels)
    'trend_distance',
    'trend_deviation',
    'distance_to_ema',
    'distance_to_sma',

    # Breakout features (levels)
    'breakout_level',
    'breakout_distance',
    'resistance_distance',
    'support_distance',

    # Bollinger Bands (width/distance)
    'bb_width',
    'bb_pctb',
    'bb_upper_distance',
    'bb_lower_distance',

    # Candle features (size)
    'candle_size',
    'candle_body',
    'candle_range',
    'body_size',
    'true_range',

    # Stop loss features (distance)
    'stop_loss',
    'stop_distance',
    'stop_level',

    # Wick features (size)
    'wick_size',
    'upper_wick',
    'lower_wick',
    'upper_shadow',
    'lower_shadow',

    # Fair Value Gap (FVG) features (size)
    'fvg_size',
    'fvg_width',
    'gap_size',

    # Volatility bands (width)
    'keltner_width',
    'donchian_width',
    'atr_bands_width',

    # Support/Resistance (distance/levels)
    'sr_distance',
    'sr_level',
    'pivot_distance',
]


def should_use_atr_normalization(feature_name: str) -> bool:
    """
    Check if a feature should use ATR normalization.

    Args:
        feature_name: Name of the feature

    Returns:
        True if feature should use ATR normalization, False otherwise
    """
    feature_lower = feature_name.lower()

    # Check against known ATR-normalized features
    for atr_feature in ATR_NORMALIZED_FEATURES:
        if atr_feature.lower() in feature_lower:
            return True

    # Check for common patterns
    atr_patterns = [
        'distance', 'level', 'width', 'size', 'range',
        'wick', 'shadow', 'gap', 'band', 'breakout'
    ]

    for pattern in atr_patterns:
        if pattern in feature_lower:
            return True

    return False


__all__ = [
    'calculate_atr',
    'atr_normalize',
    'atr_percent_normalize',
    'should_use_atr_normalization',
    'ATR_NORMALIZED_FEATURES',
]
