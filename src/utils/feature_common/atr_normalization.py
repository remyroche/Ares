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

try:
    import polars as pl  # type: ignore[import]
except Exception:  # pragma: no cover - optional dependency
    pl = None


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
    if pl is not None and isinstance(high, pl.Series):
        high_pd = high.to_pandas()
        low_pd = low.to_pandas() if isinstance(low, pl.Series) else pd.Series(low)
        close_pd = close.to_pandas() if isinstance(close, pl.Series) else pd.Series(close)
        atr_pd = calculate_atr(high_pd, low_pd, close_pd, window=window, min_periods=min_periods)
        if hasattr(atr_pd, "to_numpy"):
            values = atr_pd.to_numpy()
        else:
            values = np.asarray(atr_pd)
        return pl.Series(name=high.name, values=values)

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
        >>> # Normalize multiple spatial features
        >>> spatial_features = df[['trend_distance', 'bb_width', 'candle_size']]
        >>> normalized = atr_normalize(spatial_features, high, low, close)
    """
    # Polars-native path: when data and price inputs are Polars objects, compute
    # TR/ATR without converting to pandas and normalize using Polars operations.
    if (
        pl is not None
        and (isinstance(data, pl.DataFrame) or isinstance(data, pl.Series))  # type: ignore[arg-type]
        and isinstance(high, pl.Series)  # type: ignore[arg-type]
        and isinstance(low, pl.Series)   # type: ignore[arg-type]
        and isinstance(close, pl.Series) # type: ignore[arg-type]
    ):
        # Build True Range (TR) components in Polars
        df_prices = pl.DataFrame({
            "high": high,
            "low": low,
            "close": close,
        })

        df_prices = df_prices.with_columns([
            pl.col("close").shift(1).alias("prev_close"),
        ])

        df_prices = df_prices.with_columns([
            (pl.col("high") - pl.col("low")).alias("tr1"),
            (pl.col("high") - pl.col("prev_close")).abs().alias("tr2"),
            (pl.col("low") - pl.col("prev_close")).abs().alias("tr3"),
        ])

        # Row-wise maximum of TR components
        df_prices = df_prices.with_columns(
            pl.max_horizontal("tr1", "tr2", "tr3").alias("tr")
        )

        tr_series = df_prices["tr"]
        tr_values = tr_series.to_numpy()

        # Ensure first TR value is valid by falling back to high-low if needed
        if tr_values.size > 0 and not np.isfinite(tr_values[0]):
            high_values = df_prices["high"].to_numpy()
            low_values = df_prices["low"].to_numpy()
            tr_values[0] = float(high_values[0] - low_values[0])

        # Compute ATR via exponential moving average (EMA) in NumPy
        alpha = 2.0 / float(window + 1)
        atr_values = np.empty_like(tr_values, dtype=float)

        if tr_values.size > 0:
            atr_values[0] = tr_values[0]
            for i in range(1, tr_values.size):
                if not np.isfinite(tr_values[i]):
                    atr_values[i] = atr_values[i - 1]
                else:
                    atr_values[i] = alpha * tr_values[i] + (1.0 - alpha) * atr_values[i - 1]

        # Clip ATR to avoid division issues
        atr_values = np.clip(atr_values, min_atr, None)
        atr_series_pl = pl.Series("atr", atr_values)

        # Normalize data using Polars operations
        if isinstance(data, pl.DataFrame):  # type: ignore[arg-type]
            normalized_df = data.with_columns([
                (pl.col(col) / atr_series_pl)
                .fill_nan(0.0)
                .fill_null(0.0)
                .alias(col)
                for col in data.columns
            ])

            if return_atr:
                return normalized_df, atr_series_pl
            return normalized_df

        # data is a Polars Series
        data_series_pl = data  # type: ignore[assignment]
        normalized_series_pl = (
            (data_series_pl / atr_series_pl)
            .fill_nan(0.0)
            .fill_null(0.0)
        )
        normalized_series_pl = normalized_series_pl.rename(getattr(data_series_pl, "name", None))

        if return_atr:
            return normalized_series_pl, atr_series_pl
        return normalized_series_pl

    # Polars compatibility for mixed inputs: track original types and convert to
    # pandas for computation when full Polars path is not available.
    data_is_pl_df = pl is not None and isinstance(data, pl.DataFrame)
    data_is_pl_series = pl is not None and hasattr(pl, "Series") and isinstance(data, pl.Series)  # type: ignore[arg-type]

    if data_is_pl_df:
        data_pd = data.to_pandas()
    elif data_is_pl_series:
        data_pd = data.to_pandas()
    else:
        data_pd = data

    if pl is not None and hasattr(pl, "Series") and isinstance(high, pl.Series):  # type: ignore[arg-type]
        high_pd = high.to_pandas()
    else:
        high_pd = high

    if pl is not None and hasattr(pl, "Series") and isinstance(low, pl.Series):  # type: ignore[arg-type]
        low_pd = low.to_pandas()
    else:
        low_pd = low

    if pl is not None and hasattr(pl, "Series") and isinstance(close, pl.Series):  # type: ignore[arg-type]
        close_pd = close.to_pandas()
    else:
        close_pd = close

    # Calculate ATR
    atr = calculate_atr(high_pd, low_pd, close_pd, window=window)

    # Ensure ATR is not too small to avoid division issues
    atr_safe = atr.clip(lower=min_atr)

    # Normalize data using pandas/NumPy
    if isinstance(data_pd, pd.DataFrame):
        normalized = data_pd.div(atr_safe, axis=0).fillna(0.0)
    elif isinstance(data_pd, pd.Series):
        normalized = (data_pd / atr_safe).fillna(0.0)
    else:
        data_series = pd.Series(data_pd)
        normalized_series = data_series / atr_safe
        normalized = normalized_series.fillna(0.0).values

    # Convert back to Polars if needed
    if data_is_pl_df:
        if isinstance(normalized, pd.DataFrame):
            normalized_out = pl.from_pandas(normalized)
        else:
            normalized_out = pl.from_pandas(pd.DataFrame(normalized))
    elif data_is_pl_series:
        if isinstance(normalized, pd.Series):
            normalized_out = pl.Series(name=getattr(data, "name", None), values=normalized.values)
        else:
            normalized_out = pl.Series(normalized)
    else:
        normalized_out = normalized

    if return_atr:
        return normalized_out, atr
    else:
        return normalized_out


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
    # Polars compatibility: track original types and convert to pandas for computation
    data_is_pl_df = pl is not None and isinstance(data, pl.DataFrame)
    data_is_pl_series = pl is not None and hasattr(pl, "Series") and isinstance(data, pl.Series)  # type: ignore[arg-type]

    if data_is_pl_df:
        data_pd = data.to_pandas()
    elif data_is_pl_series:
        data_pd = data.to_pandas()
    else:
        data_pd = data

    if pl is not None and hasattr(pl, "Series") and isinstance(close, pl.Series):  # type: ignore[arg-type]
        close_pd = close.to_pandas()
    else:
        close_pd = close

    if pl is not None and hasattr(pl, "Series") and isinstance(high, pl.Series):  # type: ignore[arg-type]
        high_pd = high.to_pandas()
    else:
        high_pd = high

    if pl is not None and hasattr(pl, "Series") and isinstance(low, pl.Series):  # type: ignore[arg-type]
        low_pd = low.to_pandas()
    else:
        low_pd = low

    # Calculate ATR
    atr = calculate_atr(high_pd, low_pd, close_pd, window=window)

    # Convert close to Series if needed
    if not isinstance(close_pd, pd.Series):
        close_pd = pd.Series(close_pd)

    # Calculate ATR as percentage of close
    close_safe = close_pd.clip(lower=min_close)
    atr_percent = atr / close_safe
    atr_percent_safe = atr_percent.clip(lower=min_atr / close_safe.mean())

    # Normalize data using pandas/NumPy
    if isinstance(data_pd, pd.DataFrame):
        data_percent = data_pd.div(close_safe, axis=0)
        normalized = data_percent.div(atr_percent_safe, axis=0).fillna(0.0)
    elif isinstance(data_pd, pd.Series):
        data_percent = data_pd / close_safe
        normalized = (data_percent / atr_percent_safe).fillna(0.0)
    else:
        data_series = pd.Series(data_pd)
        close_safe_series = pd.Series(close_safe)
        data_percent = data_series / close_safe_series
        normalized_series = data_percent / atr_percent_safe
        normalized = normalized_series.fillna(0.0).values

    # Convert back to Polars if needed
    if data_is_pl_df:
        if isinstance(normalized, pd.DataFrame):
            return pl.from_pandas(normalized)
        return pl.from_pandas(pd.DataFrame(normalized))
    if data_is_pl_series:
        if isinstance(normalized, pd.Series):
            return pl.Series(name=getattr(data, "name", None), values=normalized.values)
        return pl.Series(normalized)

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

    # Volume profile (range)
    'volume_profile_range',
]


# Explicit prefixes for SR level-style features that should use ATR normalization.
# Using prefixes avoids accidental matches on unrelated features that merely
# contain the same substrings.
ATR_SR_LEVEL_PREFIXES = [
    'support_level_',
    'resistance_level_',
    'pivot_point_',
    'fibonacci_',
    'sr_volume_weighted_',
    'sr_dynamic_',
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

    # Check explicit SR level-style prefixes first (pattern-safe)
    for prefix in ATR_SR_LEVEL_PREFIXES:
        if feature_lower.startswith(prefix):
            return True

    # Check against known ATR-normalized features (substring match)
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
