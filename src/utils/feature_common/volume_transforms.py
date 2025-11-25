"""
Robust Volume Transformations

This module provides robust volume transformations for feature engineering,
including log transformations, median-based normalizations, and MAD-based z-scores.

Key transformations:
- log_volume: Logarithmic transformation to stabilize volume
- median_log_volume: Rolling median of log volume
- robust_z: Robust z-score using median and MAD
- vol_norm: Volume normalized by true range (TR) or ATR
"""

import numpy as np
import pandas as pd
from typing import Optional, Union
import warnings


def stabilize_volume(volume: Union[pd.Series, np.ndarray],
                     min_volume: float = 1.0) -> Union[pd.Series, np.ndarray]:
    """
    Stabilize volume by ensuring all values are positive and non-zero.

    Args:
        volume: Volume data
        min_volume: Minimum volume value to use for stabilization

    Returns:
        Stabilized volume
    """
    if isinstance(volume, pd.Series):
        return volume.clip(lower=min_volume)
    else:
        return np.clip(volume, a_min=min_volume, a_max=None)


def log_volume(volume: Union[pd.Series, np.ndarray],
               stabilize_first: bool = True,
               min_volume: float = 1.0) -> Union[pd.Series, np.ndarray]:
    """
    Calculate log-transformed volume to stabilize variance.

    Args:
        volume: Volume data
        stabilize_first: Whether to stabilize volume first (ensure positive)
        min_volume: Minimum volume value to use for stabilization

    Returns:
        Log-transformed volume
    """
    if stabilize_first:
        volume = stabilize_volume(volume, min_volume)

    if isinstance(volume, pd.Series):
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            result = np.log(volume)
            return pd.Series(result, index=volume.index, name=f'log_{volume.name}' if volume.name else 'log_volume')
    else:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            return np.log(volume)


def rolling_median_log_volume(volume: Union[pd.Series, np.ndarray],
                               window: int = 20,
                               min_volume: float = 1.0) -> Union[pd.Series, np.ndarray]:
    """
    Calculate rolling median of log-transformed volume.

    Args:
        volume: Volume data
        window: Rolling window size
        min_volume: Minimum volume value to use for stabilization

    Returns:
        Rolling median of log volume
    """
    log_vol = log_volume(volume, stabilize_first=True, min_volume=min_volume)

    if isinstance(log_vol, pd.Series):
        return log_vol.rolling(window=window, min_periods=1).median()
    else:
        return pd.Series(log_vol).rolling(window=window, min_periods=1).median().values


def calculate_mad(data: Union[pd.Series, np.ndarray],
                  center: Optional[Union[pd.Series, np.ndarray]] = None,
                  window: Optional[int] = None) -> Union[pd.Series, np.ndarray]:
    """
    Calculate Median Absolute Deviation (MAD).

    Args:
        data: Input data
        center: Center values (median). If None, calculated from data
        window: Rolling window size. If None, calculates global MAD

    Returns:
        MAD values
    """
    if isinstance(data, pd.Series):
        if window is not None:
            # Rolling MAD
            if center is None:
                center = data.rolling(window=window, min_periods=1).median()

            # Calculate absolute deviations
            abs_dev = (data - center).abs()

            # Rolling median of absolute deviations
            mad = abs_dev.rolling(window=window, min_periods=1).median()

            # Scale factor to make MAD comparable to standard deviation
            # MAD * 1.4826 ≈ std for normal distribution
            return mad * 1.4826
        else:
            # Global MAD
            if center is None:
                center = data.median()
            abs_dev = (data - center).abs()
            mad = abs_dev.median()
            return mad * 1.4826
    else:
        data_series = pd.Series(data)
        if window is not None:
            if center is None:
                center = data_series.rolling(window=window, min_periods=1).median()
            elif not isinstance(center, pd.Series):
                center = pd.Series(center)

            abs_dev = (data_series - center).abs()
            mad = abs_dev.rolling(window=window, min_periods=1).median()
            return (mad * 1.4826).values
        else:
            if center is None:
                center = data_series.median()
            abs_dev = (data_series - center).abs()
            mad = abs_dev.median()
            return mad * 1.4826


def robust_z_score(volume: Union[pd.Series, np.ndarray],
                   window: int = 20,
                   min_volume: float = 1.0,
                   min_mad: float = 1e-8) -> Union[pd.Series, np.ndarray]:
    """
    Calculate robust z-score for volume using median and MAD.

    robust_z = (log_volume - median_log_volume) / MAD(log_volume)

    This is more robust to outliers than standard z-score.

    Args:
        volume: Volume data
        window: Rolling window size
        min_volume: Minimum volume value to use for stabilization
        min_mad: Minimum MAD value to avoid division by zero

    Returns:
        Robust z-scores
    """
    # Calculate log volume
    log_vol = log_volume(volume, stabilize_first=True, min_volume=min_volume)

    # Calculate rolling median
    if isinstance(log_vol, pd.Series):
        median_log_vol = log_vol.rolling(window=window, min_periods=1).median()
    else:
        log_vol = pd.Series(log_vol)
        median_log_vol = log_vol.rolling(window=window, min_periods=1).median()

    # Calculate MAD
    mad = calculate_mad(log_vol, center=median_log_vol, window=window)

    # Ensure MAD is not too small to avoid division issues
    if isinstance(mad, pd.Series):
        mad = mad.clip(lower=min_mad)
    else:
        mad = np.clip(mad, a_min=min_mad, a_max=None)

    # Calculate robust z-score
    robust_z = (log_vol - median_log_vol) / mad

    if isinstance(volume, pd.Series):
        if isinstance(robust_z, pd.Series):
            return robust_z.rename(f'robust_z_{volume.name}' if volume.name else 'robust_z_volume')
        else:
            return pd.Series(robust_z, index=volume.index, name=f'robust_z_{volume.name}' if volume.name else 'robust_z_volume')
    else:
        if isinstance(robust_z, pd.Series):
            return robust_z.values
        else:
            return robust_z


def calculate_true_range(high: Union[pd.Series, np.ndarray],
                         low: Union[pd.Series, np.ndarray],
                         close: Optional[Union[pd.Series, np.ndarray]] = None) -> Union[pd.Series, np.ndarray]:
    """
    Calculate True Range (TR).

    If close is provided:
        TR = max(high - low, abs(high - prev_close), abs(low - prev_close))
    Otherwise:
        TR = high - low

    Args:
        high: High prices
        low: Low prices
        close: Close prices (optional, for full TR calculation)

    Returns:
        True Range
    """
    if isinstance(high, pd.Series):
        if close is not None:
            # Full TR calculation
            if isinstance(close, pd.Series):
                prev_close = close.shift(1)
            else:
                prev_close = pd.Series(close).shift(1)

            tr1 = high - low
            tr2 = (high - prev_close).abs()
            tr3 = (low - prev_close).abs()

            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            return tr.fillna(high - low)  # Fill first value with high - low
        else:
            # Simple range
            return high - low
    else:
        high = np.asarray(high)
        low = np.asarray(low)

        if close is not None:
            # Full TR calculation
            close = np.asarray(close)
            prev_close = np.roll(close, 1)
            prev_close[0] = close[0]  # Handle first value

            tr1 = high - low
            tr2 = np.abs(high - prev_close)
            tr3 = np.abs(low - prev_close)

            return np.maximum(tr1, np.maximum(tr2, tr3))
        else:
            # Simple range
            return high - low


def calculate_atr(high: Union[pd.Series, np.ndarray],
                  low: Union[pd.Series, np.ndarray],
                  close: Union[pd.Series, np.ndarray],
                  window: int = 14) -> Union[pd.Series, np.ndarray]:
    """
    Calculate Average True Range (ATR).

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        window: Rolling window size

    Returns:
        ATR
    """
    tr = calculate_true_range(high, low, close)

    if isinstance(tr, pd.Series):
        # Use EMA for smoother ATR
        atr = tr.ewm(span=window, min_periods=1).mean()
        return atr
    else:
        tr_series = pd.Series(tr)
        atr = tr_series.ewm(span=window, min_periods=1).mean()
        return atr.values


def volume_normalized_by_tr(volume: Union[pd.Series, np.ndarray],
                             high: Union[pd.Series, np.ndarray],
                             low: Union[pd.Series, np.ndarray],
                             close: Optional[Union[pd.Series, np.ndarray]] = None,
                             use_atr: bool = False,
                             atr_window: int = 14,
                             min_tr: float = 1e-8) -> Union[pd.Series, np.ndarray]:
    """
    Normalize volume by True Range or ATR.

    vol_norm = volume / TR  (or volume / ATR if use_atr=True)

    This normalizes volume by price movement, making it comparable across different
    price levels and volatility regimes.

    Args:
        volume: Volume data
        high: High prices
        low: Low prices
        close: Close prices (required if use_atr=True or for full TR calculation)
        use_atr: Whether to use ATR instead of TR for smoothing
        atr_window: ATR window size (only used if use_atr=True)
        min_tr: Minimum TR/ATR value to avoid division by zero

    Returns:
        Volume normalized by TR or ATR
    """
    if use_atr:
        if close is None:
            raise ValueError("close prices are required for ATR calculation")
        normalizer = calculate_atr(high, low, close, window=atr_window)
    else:
        normalizer = calculate_true_range(high, low, close)

    # Ensure normalizer is not too small
    if isinstance(normalizer, pd.Series):
        normalizer = normalizer.clip(lower=min_tr)
    else:
        normalizer = np.clip(normalizer, a_min=min_tr, a_max=None)

    # Normalize volume
    vol_norm = volume / normalizer

    if isinstance(volume, pd.Series):
        if isinstance(vol_norm, pd.Series):
            return vol_norm.rename(f'vol_norm_{volume.name}' if volume.name else 'vol_norm')
        else:
            return pd.Series(vol_norm, index=volume.index, name=f'vol_norm_{volume.name}' if volume.name else 'vol_norm')
    else:
        if isinstance(vol_norm, pd.Series):
            return vol_norm.values
        else:
            return vol_norm


# Convenience function for getting all robust volume features at once
def calculate_robust_volume_features(data: pd.DataFrame,
                                      volume_col: str = 'volume',
                                      high_col: str = 'high',
                                      low_col: str = 'low',
                                      close_col: str = 'close',
                                      window: int = 20,
                                      atr_window: int = 14,
                                      include_log_volume: bool = True,
                                      include_robust_z: bool = True,
                                      include_vol_norm: bool = True,
                                      use_atr: bool = False) -> pd.DataFrame:
    """
    Calculate all robust volume features at once.

    Args:
        data: DataFrame with OHLCV data
        volume_col: Name of volume column
        high_col: Name of high column
        low_col: Name of low column
        close_col: Name of close column
        window: Rolling window size for median/MAD calculations
        atr_window: ATR window size
        include_log_volume: Whether to include log volume
        include_robust_z: Whether to include robust z-score
        include_vol_norm: Whether to include volume normalized by TR/ATR
        use_atr: Whether to use ATR instead of TR for volume normalization

    Returns:
        DataFrame with robust volume features
    """
    results = {}

    volume = data[volume_col]

    if include_log_volume:
        results['log_volume'] = log_volume(volume, stabilize_first=True)
        results['median_log_volume'] = rolling_median_log_volume(volume, window=window)

    if include_robust_z:
        results['robust_z_volume'] = robust_z_score(volume, window=window)

    if include_vol_norm:
        high = data[high_col]
        low = data[low_col]
        close = data[close_col] if close_col in data.columns else None

        results['vol_norm'] = volume_normalized_by_tr(
            volume, high, low, close,
            use_atr=use_atr,
            atr_window=atr_window
        )

    return pd.DataFrame(results, index=data.index)


def log1p_zscore_normalize(
    volume: Union[pd.Series, np.ndarray],
    window: int = 500,
    min_periods: int = 1,
    ddof: int = 1,
) -> Union[pd.Series, np.ndarray]:
    """
    Apply log1p transformation followed by rolling z-score normalization.

    This is the recommended approach for volume features:
    1. log1p(volume) - stabilizes variance and handles zero volumes
    2. Rolling z-score with growing window up to 500 - normalizes across time

    Args:
        volume: Volume data
        window: Rolling window size (default: 500 for ~500 bars)
        min_periods: Minimum periods for rolling window (default: 1 for growing window)
        ddof: Degrees of freedom for std calculation (default: 1)

    Returns:
        Log1p + z-score normalized volume

    Examples:
        >>> # Normalize volume for ML model
        >>> volume_normalized = log1p_zscore_normalize(df['volume'])
        >>>
        >>> # With custom window
        >>> volume_normalized = log1p_zscore_normalize(df['volume'], window=300)
    """
    # Apply log1p transformation
    if isinstance(volume, pd.Series):
        log_vol = np.log1p(volume)
        log_vol.name = f'log1p_{volume.name}' if volume.name else 'log1p_volume'
    else:
        log_vol = np.log1p(volume)

    # Apply rolling z-score with growing window
    if isinstance(log_vol, pd.Series):
        # Calculate rolling statistics using only past data
        rolling_mean = log_vol.rolling(window=window, min_periods=min_periods).mean()
        rolling_std = log_vol.rolling(window=window, min_periods=min_periods).std(ddof=ddof)

        # Normalize using rolling statistics
        rolling_std_safe = rolling_std.replace(0, np.nan)
        normalized = (log_vol - rolling_mean) / rolling_std_safe
        normalized = normalized.fillna(0.0)

        if volume.name:
            normalized.name = f'log1p_zscore_{volume.name}'

        return normalized
    else:
        # Convert to Series for rolling operations
        log_vol_series = pd.Series(log_vol)

        rolling_mean = log_vol_series.rolling(window=window, min_periods=min_periods).mean()
        rolling_std = log_vol_series.rolling(window=window, min_periods=min_periods).std(ddof=ddof)

        rolling_std_safe = rolling_std.replace(0, np.nan)
        normalized = (log_vol_series - rolling_mean) / rolling_std_safe
        normalized = normalized.fillna(0.0)

        return normalized.values


__all__ = [
    'stabilize_volume',
    'log_volume',
    'rolling_median_log_volume',
    'calculate_mad',
    'robust_z_score',
    'calculate_true_range',
    'calculate_atr',
    'volume_normalized_by_tr',
    'calculate_robust_volume_features',
    'log1p_zscore_normalize',
]
