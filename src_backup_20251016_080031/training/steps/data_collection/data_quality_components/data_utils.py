"""Data Utilities Component

Common utility functions for data processing and analysis.
Extracted from raw_data_quality_checker.py
"""

from datetime import datetime, timedelta
from typing import Any, Optional, List
import pandas as pd
import logging
import numpy as np

from src.utils.logger import system_logger
from src.training.steps.standardized_parquet_handler import standardized_parquet_handler

def determine_timeframe_from_data(data: pd.DataFrame) -> str:
    """Determine the timeframe from the data intervals.
    
    Args:
        data: Market data with datetime index
        
    Returns:
        Timeframe string (e.g., '1m', '5m', '15m', '1h')
    """
    if len(data) < 2:
        return '1m'
        
    time_diffs = data.index.to_series().diff().dropna()
    if len(time_diffs) == 0:
        return '1m'
        
    most_common_interval = time_diffs.mode().iloc[0] if len(time_diffs.mode()) > 0 else time_diffs.median()
    interval_seconds = most_common_interval.total_seconds()
    
    if interval_seconds <= 60:
        return '1m'
    elif interval_seconds <= 300:
        return '5m'
    elif interval_seconds <= 900:
        return '15m'
    elif interval_seconds <= 1800:
        return '30m'
    elif interval_seconds <= 3600:
        return '1h'
    elif interval_seconds <= 14400:
        return '4h'
    elif interval_seconds <= 86400:
        return '1d'
    else:
        return '1d'

def estimate_timeframe_from_data(data: pd.DataFrame) -> str:
    """Estimate the timeframe from data characteristics.
    
    Args:
        data: DataFrame to analyze
        
    Returns:
        Estimated timeframe string
    """
    try:
        column_names = ' '.join(data.columns).lower()
        if any(tf in column_names for tf in ['1m', '1min', 'minute']):
            return '1m'
        elif any(tf in column_names for tf in ['5m', '5min']):
            return '5m'
        elif any(tf in column_names for tf in ['15m', '15min']):
            return '15m'
        elif any(tf in column_names for tf in ['30m', '30min']):
            return '30m'
        elif any(tf in column_names for tf in ['1h', 'hour']):
            return '1h'
        elif any(tf in column_names for tf in ['4h', '4hour']):
            return '4h'
        elif any(tf in column_names for tf in ['1d', 'day', 'daily']):
            return '1d'
        elif len(data) > 10000:
            return '1m'
        elif len(data) > 1000:
            return '5m'
        elif len(data) > 100:
            return '15m'
        else:
            return '1h'
    except Exception as e:
        logger = system_logger.getChild("DataUtils")
        logger.debug(f'⚠️ Error estimating timeframe: {e}')
        return '1m'

def fix_datetime_index(data: pd.DataFrame) -> pd.DataFrame | None:
    """Fix missing datetime index by creating one from available data.
    
    Args:
        data: DataFrame with missing datetime index
        
    Returns:
        DataFrame with datetime index or None if failed
    """
    logger = system_logger.getChild("DataUtils")
    
    try:
        logger.info('🔧 Attempting to create datetime index...')
        
        # Check for timestamp columns
        timestamp_columns = ['timestamp', 'time', 'date', 'datetime', 'index']
        for col in timestamp_columns:
            if col in data.columns:
                logger.info(f'🔧 Found timestamp column: {col}')
                try:
                    if data[col].dtype == 'object':
                        # Try different datetime formats
                        for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d', '%Y-%m-%d %H:%M:%S.%f', 
                                  '%Y-%m-%dT%H:%M:%S', '%Y-%m-%dT%H:%M:%S.%f']:
                            try:
                                timestamps = pd.to_datetime(data[col], format = fmt)
                                if not timestamps.isna().all():
                                    fixed_data = data.copy()
                                    fixed_data.index = timestamps
                                    fixed_data = fixed_data.drop(columns=[col])
                                    logger.info(f'✅ Created datetime index from {col} using format {fmt}')
                                    return fixed_data
                            except Exception:
                                continue
                    else:
                        timestamps = pd.to_datetime(data[col])
                        if not timestamps.isna().all():
                            fixed_data = data.copy()
                            fixed_data.index = timestamps
                            fixed_data = fixed_data.drop(columns=[col])
                            logger.info(f'✅ Created datetime index from {col}')
                            return fixed_data
                except Exception as e:
                    logger.debug(f'⚠️ Failed to parse {col}: {e}')
                    continue
                    
        # Try to parse existing index
        try:
            if data.index.dtype == 'object':
                timestamps = pd.to_datetime(data.index)
                if not timestamps.isna().all():
                    fixed_data = data.copy()
                    fixed_data.index = timestamps
                    logger.info('✅ Created datetime index from existing index')
                    return fixed_data
        except Exception as e:
            logger.debug(f'⚠️ Failed to parse existing index: {e}')
            
        # Create synthetic datetime index as last resort
        logger.info('🔧 Creating synthetic datetime index...')
        timeframe = estimate_timeframe_from_data(data)
        logger.info(f'🔧 Estimated timeframe: {timeframe}')
        
        interval_map = {
            '1m': pd.Timedelta(minutes = 1),
            '5m': pd.Timedelta(minutes = 5),
            '15m': pd.Timedelta(minutes = 15),
            '30m': pd.Timedelta(minutes = 30),
            '1h': pd.Timedelta(hours = 1),
            '4h': pd.Timedelta(hours = 4),
            '1d': pd.Timedelta(days = 1)
        }
        interval = interval_map.get(timeframe, pd.Timedelta(minutes = 1))
        
        start_time = pd.Timestamp('2024-01-01 00:00:00')
        timestamps = [start_time + i * interval for i in range(len(data))]
        fixed_data = data.copy()
        fixed_data.index = timestamps
        logger.info(f'✅ Created synthetic datetime index with {timeframe} intervals')
        return fixed_data
        
    except Exception as e:
        logger.exception(f'❌ Failed to create datetime index: {e}')
        return None

def calculate_interval_statistics(data: pd.DataFrame) -> dict[str, Any]:
    """Calculate statistics about time intervals in the data.
    
    Args:
        data: DataFrame with datetime index
        
    Returns:
        Dictionary with interval statistics
    """
    if not isinstance(data.index, pd.DatetimeIndex) or len(data) < 2:
        return {
            'total_intervals': 0,
            'expected_interval': None,
            'irregular_intervals': 0,
            'irregular_ratio': 0.0,
            'mean_interval_seconds': 0.0,
            'std_interval_seconds': 0.0,
            'coefficient_of_variation': 0.0
        }
        
    time_diffs = data.index.to_series().diff().dropna()
    if len(time_diffs) == 0:
        return {
            'total_intervals': 0,
            'expected_interval': None,
            'irregular_intervals': 0,
            'irregular_ratio': 0.0,
            'mean_interval_seconds': 0.0,
            'std_interval_seconds': 0.0,
            'coefficient_of_variation': 0.0
        }
        
    expected_interval = time_diffs.mode().iloc[0] if len(time_diffs.mode()) > 0 else time_diffs.median()
    tolerance_percentage = 0.15
    tolerance_seconds = expected_interval.total_seconds() * tolerance_percentage
    irregular_intervals = time_diffs[abs(time_diffs - expected_interval) > pd.Timedelta(seconds = tolerance_seconds)]
    
    time_diffs_seconds = time_diffs.dt.total_seconds()
    mean_interval = time_diffs_seconds.mean()
    std_interval = time_diffs_seconds.std()
    cv = std_interval / mean_interval if mean_interval > 0 else 0
    
    return {
        'total_intervals': len(time_diffs),
        'expected_interval': str(expected_interval),
        'irregular_intervals': len(irregular_intervals),
        'irregular_ratio': len(irregular_intervals) / len(time_diffs),
        'mean_interval_seconds': mean_interval,
        'std_interval_seconds': std_interval,
        'coefficient_of_variation': cv
    }

def detect_data_gaps(data: pd.DataFrame, max_gap_hours: float = 1.0) -> dict[str, Any]:
    """Detect gaps in the data.
    
    Args:
        data: DataFrame with datetime index
        max_gap_hours: Maximum acceptable gap in hours
        
    Returns:
        Dictionary with gap information
    """
    if not isinstance(data.index, pd.DatetimeIndex) or len(data) < 2:
        return {
            'total_gaps': 0,
            'large_gaps': 0,
            'max_gap_hours': 0.0,
            'gap_positions': []
        }
        
    time_diffs = data.index.to_series().diff().dropna()
    if len(time_diffs) == 0:
        return {
            'total_gaps': 0,
            'large_gaps': 0,
            'max_gap_hours': 0.0,
            'gap_positions': []
        }
        
    expected_interval = time_diffs.mode().iloc[0] if len(time_diffs.mode()) > 0 else time_diffs.median()
    max_gap_threshold = pd.Timedelta(hours = max_gap_hours)
    
    gaps = time_diffs[time_diffs > expected_interval]
    large_gaps = gaps[gaps > max_gap_threshold]
    
    gap_positions = []
    for gap_time, gap_duration in gaps.items():
        gap_positions.append({
            'timestamp': gap_time,
            'duration_hours': gap_duration.total_seconds() / 3600,
            'is_large': gap_duration > max_gap_threshold
        })
        
    return {
        'total_gaps': len(gaps),
        'large_gaps': len(large_gaps),
        'max_gap_hours': gaps.max().total_seconds() / 3600 if len(gaps) > 0 else 0.0,
        'gap_positions': gap_positions
    }

def calculate_data_span(data: pd.DataFrame) -> dict[str, Any]:
    """Calculate data span information.
    
    Args:
        data: DataFrame with datetime index
        
    Returns:
        Dictionary with span information
    """
    if not isinstance(data.index, pd.DatetimeIndex) or len(data) == 0:
        return {
            'span_days': 0,
            'span_hours': 0,
            'start_time': None,
            'end_time': None,
            'is_single_timestamp': True
        }
        
    start_time = data.index.min()
    end_time = data.index.max()
    
    if start_time == end_time:
        return {
            'span_days': 0,
            'span_hours': 0,
            'start_time': start_time,
            'end_time': end_time,
            'is_single_timestamp': True
        }
        
    span = end_time - start_time
    span_days = span.days
    span_hours = span.total_seconds() / 3600
    
    return {
        'span_days': span_days,
        'span_hours': span_hours,
        'start_time': start_time,
        'end_time': end_time,
        'is_single_timestamp': False
    }

def validate_ohlc_consistency(data: pd.DataFrame) -> dict[str, Any]:
    """Validate OHLC data consistency.
    
    Args:
        data: DataFrame with OHLC columns
        
    Returns:
        Dictionary with consistency validation results
    """
    ohlc_columns = ['open', 'high', 'low', 'close']
    missing_columns = [col for col in ohlc_columns if col not in data.columns]
    
    if missing_columns:
        return {
            'is_consistent': False,
            'missing_columns': missing_columns,
            'inconsistent_records': 0,
            'inconsistency_ratio': 0.0,
            'issues': [f'Missing required columns: {missing_columns}']
        }
        
    # Check for OHLC consistency
    ohlc_inconsistent = (
        (data['high'] < data['low']) |
        (data['open'] > data['high']) |
        (data['close'] > data['high']) |
        (data['open'] < data['low']) |
        (data['close'] < data['low'])
    )
    
    inconsistent_records = ohlc_inconsistent.sum()
    inconsistency_ratio = inconsistent_records / len(data)
    
    issues = []
    if inconsistent_records > 0:
        issues.append(f'OHLC inconsistency found in {inconsistent_records} records ({inconsistency_ratio:.3f})')
        
    # Check for negative prices
    negative_prices = (data[ohlc_columns] < 0).any(axis = 1)
    negative_count = negative_prices.sum()
    if negative_count > 0:
        issues.append(f'Negative prices found in {negative_count} records')
        
    return {
        'is_consistent': inconsistent_records == 0 and negative_count == 0,
        'missing_columns': missing_columns,
        'inconsistent_records': inconsistent_records,
        'inconsistency_ratio': inconsistency_ratio,
        'negative_price_records': negative_count,
        'issues': issues
    }

def calculate_volume_statistics(data: pd.DataFrame) -> dict[str, Any]:
    """Calculate volume statistics.
    
    Args:
        data: DataFrame with volume column
        
    Returns:
        Dictionary with volume statistics
    """
    if 'volume' not in data.columns:
        return {
            'mean_volume': 0.0,
            'std_volume': 0.0,
            'min_volume': 0.0,
            'max_volume': 0.0,
            'zero_volume_ratio': 0.0,
            'negative_volume_ratio': 0.0,
            'volume_spikes': 0,
            'volume_drops': 0
        }
        
    volume = data['volume']
    mean_volume = volume.mean()
    std_volume = volume.std()
    
    zero_volume_ratio = (volume <= 0).sum() / len(volume)
    negative_volume_ratio = (volume < 0).sum() / len(volume)
    
    # Detect volume spikes (5x mean) and drops (0.1x mean)
    volume_spikes = (volume > mean_volume + 5 * std_volume).sum()
    volume_drops = (volume < mean_volume - 5 * std_volume).sum()
    
    return {
        'mean_volume': float(mean_volume),
        'std_volume': float(std_volume),
        'min_volume': float(volume.min()),
        'max_volume': float(volume.max()),
        'zero_volume_ratio': float(zero_volume_ratio),
        'negative_volume_ratio': float(negative_volume_ratio),
        'volume_spikes': int(volume_spikes),
        'volume_drops': int(volume_drops)
    }

def generate_data_summary(data: pd.DataFrame) -> dict[str, Any]:
    """Generate a comprehensive data summary.
    
    Args:
        data: DataFrame to summarize
        
    Returns:
        Dictionary with comprehensive data summary
    """
    summary = {
        'basic_info': {
            'shape': data.shape,
            'columns': list(data.columns),
            'dtypes': data.dtypes.to_dict(),
            'memory_usage': data.memory_usage(deep = True).sum()
        },
        'data_span': calculate_data_span(data),
        'interval_statistics': calculate_interval_statistics(data),
        'gap_analysis': detect_data_gaps(data),
        'ohlc_consistency': validate_ohlc_consistency(data),
        'volume_statistics': calculate_volume_statistics(data)
    }
    
    # Add missing value analysis
    missing_analysis = {}
    for col in data.columns:
        missing_count = data[col].isna().sum()
        missing_ratio = missing_count / len(data) if len(data) > 0 else 0
        missing_analysis[col] = {
            'missing_count': int(missing_count),
            'missing_ratio': float(missing_ratio)
        }
    summary['missing_values'] = missing_analysis
    
    return summary