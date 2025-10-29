"""
Time series utilities for trading data manipulation and validation.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Union, Tuple, Any
from datetime import datetime, timedelta

from .error_handling import ValidationError, TradingErrorSeverity
from .constants import MIN_MARKET_DATA_ROWS

def align_time_series(
    series_list: List[pd.Series],
    method: str = 'forward_fill',
    tolerance: Optional[pd.Timedelta] = None
) -> List[pd.Series]:
    """
    Align multiple time series to a common index.

    Args:
        series_list: List of time series to align
        method: Alignment method ('forward_fill', 'backward_fill', 'interpolate')
        tolerance: Maximum time difference for alignment

    Returns:
        List of aligned series
    """
    if not series_list:
        return []

    # Find common index
    common_index = series_list[0].index
    for series in series_list[1:]:
        common_index = common_index.union(series.index)

    aligned_series = []
    for series in series_list:
        aligned = series.reindex(common_index, method=method, tolerance=tolerance)
        aligned_series.append(aligned)

    return aligned_series

def fill_time_series_gaps(
    series: pd.Series,
    method: str = 'forward_fill',
    limit: Optional[int] = None
) -> pd.Series:
    """
    Fill gaps in time series data.

    Args:
        series: Time series to fill
        method: Filling method ('forward_fill', 'backward_fill', 'interpolate', 'zero')
        limit: Maximum consecutive periods to fill

    Returns:
        Filled time series
    """
    if method == 'forward_fill':
        return series.fillna(method='ffill', limit=limit)
    elif method == 'backward_fill':
        return series.fillna(method='bfill', limit=limit)
    elif method == 'interpolate':
        return series.interpolate(method='linear', limit=limit)
    elif method == 'zero':
        return series.fillna(0)
    else:
        raise ValueError(f"Unknown fill method: {method}")

def resample_time_series(
    data: pd.DataFrame,
    target_frequency: str,
    aggregation_method: str = 'ohlc'
) -> pd.DataFrame:
    """
    Resample time series to a different frequency.

    Args:
        data: DataFrame with datetime index
        target_frequency: Target frequency (e.g., '1H', '1D')
        aggregation_method: Aggregation method ('ohlc', 'mean', 'last')

    Returns:
        Resampled DataFrame
    """
    if data.empty:
        return data

    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValidationError(
            "DataFrame must have DatetimeIndex for resampling",
            severity=TradingErrorSeverity.HIGH
        )

    if aggregation_method == 'ohlc':
        if all(col in data.columns for col in ['open', 'high', 'low', 'close']):
            agg_dict = {
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last'
            }
            if 'volume' in data.columns:
                agg_dict['volume'] = 'sum'
            return data.resample(target_frequency).agg(agg_dict)
        else:
            return data.resample(target_frequency).agg('last')
    elif aggregation_method == 'mean':
        return data.resample(target_frequency).mean()
    elif aggregation_method == 'last':
        return data.resample(target_frequency).last()
    else:
        raise ValueError(f"Unknown aggregation method: {aggregation_method}")

def validate_time_series_continuity(
    series: pd.Series,
    expected_frequency: Optional[str] = None,
    max_gap_multiplier: float = 3.0
) -> Dict[str, Any]:
    """
    Validate time series continuity.

    Args:
        series: Time series to validate
        expected_frequency: Expected frequency
        max_gap_multiplier: Multiplier for expected frequency

    Returns:
        Dictionary with validation results
    """
    if not isinstance(series.index, pd.DatetimeIndex):
        return {
            'valid': False,
            'error': 'Index is not DatetimeIndex'
        }

    if len(series) < 2:
        return {
            'valid': True,
            'warning': 'Series too short for validation'
        }

    timestamps = series.index
    diffs = timestamps.to_series().diff()

    if expected_frequency:
        expected_timedelta = pd.Timedelta(expected_frequency)
        threshold = expected_timedelta * max_gap_multiplier
    else:
        median_diff = diffs.median()
        threshold = median_diff * max_gap_multiplier

    gaps = (diffs > threshold).sum()

    return {
        'valid': gaps == 0,
        'gap_count': int(gaps),
        'max_gap': str(diffs.max()) if len(diffs) > 0 else None,
        'threshold': str(threshold)
    }

def merge_time_series(
    series_list: List[pd.Series],
    how: str = 'outer',
    fill_method: Optional[str] = None
) -> pd.DataFrame:
    """
    Merge multiple time series into a DataFrame.

    Args:
        series_list: List of time series to merge
        how: Merge method ('outer', 'inner', 'left', 'right')
        fill_method: Fill method for NaN values

    Returns:
        Merged DataFrame
    """
    if not series_list:
        return pd.DataFrame()

    # Create DataFrame from series
    df = pd.DataFrame()
    for i, series in enumerate(series_list):
        df[f'series_{i}'] = series

    if fill_method:
        df = df.fillna(method=fill_method)

    return df

def detect_time_series_anomalies(
    series: pd.Series,
    method: str = 'zscore',
    threshold: float = 3.0
) -> Dict[str, Any]:
    """
    Detect anomalies in time series.

    Args:
        series: Time series to analyze
        method: Detection method ('zscore', 'iqr')
        threshold: Threshold for detection

    Returns:
        Dictionary with anomaly information
    """
    if len(series) < 3:
        return {'anomalies': [], 'anomaly_count': 0}

    anomalies = []

    if method == 'zscore':
        mean = series.mean()
        std = series.std()
        if std > 0:
            z_scores = (series - mean) / std
            anomaly_indices = series.index[abs(z_scores) > threshold]
            for idx in anomaly_indices:
                anomalies.append({
                    'timestamp': idx,
                    'value': series[idx],
                    'z_score': z_scores[idx]
                })
    elif method == 'iqr':
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        anomaly_indices = series.index[(series < lower_bound) | (series > upper_bound)]
        for idx in anomaly_indices:
            anomalies.append({
                'timestamp': idx,
                'value': series[idx],
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            })
    else:
        raise ValueError(f"Unknown detection method: {method}")

    return {
        'anomalies': anomalies,
        'anomaly_count': len(anomalies),
        'method': method,
        'threshold': threshold
    }

def aggregate_time_series_features(
    data: pd.DataFrame,
    windows: List[int] = [5, 10, 20, 50],
    functions: List[str] = ['mean', 'std', 'min', 'max']
) -> pd.DataFrame:
    """
    Create rolling window features from time series.

    Args:
        data: Time series DataFrame
        windows: List of window sizes
        functions: List of aggregation functions

    Returns:
        DataFrame with aggregated features
    """
    result = data.copy()

    numeric_columns = data.select_dtypes(include=[np.number]).columns

    for col in numeric_columns:
        for window in windows:
            for func in functions:
                feature_name = f"{col}_{func}_{window}"
                if func == 'mean':
                    result[feature_name] = data[col].rolling(window=window).mean()
                elif func == 'std':
                    result[feature_name] = data[col].rolling(window=window).std()
                elif func == 'min':
                    result[feature_name] = data[col].rolling(window=window).min()
                elif func == 'max':
                    result[feature_name] = data[col].rolling(window=window).max()

    return result
