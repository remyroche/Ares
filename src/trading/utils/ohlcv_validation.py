"""
Enhanced OHLCV validation utilities.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, Tuple
from datetime import datetime, timedelta

from src.printing import tprint
from .error_handling import ValidationError, TradingErrorSeverity
from .validation import validate_market_data
from .constants import EXTREME_PRICE_CHANGE_THRESHOLD

def detect_timestamp_gaps(
    data: pd.DataFrame,
    expected_frequency: Optional[str] = None,
    max_gap_multiplier: float = 3.0
) -> Dict[str, Any]:
    """
    Detect gaps in timestamp data.

    Args:
        data: DataFrame with datetime index
        expected_frequency: Expected frequency (e.g., '1T', '1H')
        max_gap_multiplier: Multiplier for expected frequency to consider a gap

    Returns:
        Dictionary with gap information
    """
    tprint(f"[OHLCV_VALIDATION] detect_timestamp_gaps: data_shape={data.shape if not data.empty else 'empty'}, expected_frequency={expected_frequency}")
    if data.empty or len(data) < 2:
        tprint(f"[OHLCV_VALIDATION] detect_timestamp_gaps -> no gaps (insufficient data)")
        return {'gaps': [], 'gap_count': 0, 'max_gap': None}

    if not isinstance(data.index, pd.DatetimeIndex):
        return {'gaps': [], 'gap_count': 0, 'max_gap': None, 'error': 'Index is not DatetimeIndex'}

    gaps = []
    timestamps = data.index
    diffs = timestamps.to_series().diff()

    if expected_frequency:
        expected_timedelta = pd.Timedelta(expected_frequency)
        threshold = expected_timedelta * max_gap_multiplier
    else:
        # Use median of diffs as baseline
        median_diff = diffs.median()
        threshold = median_diff * max_gap_multiplier

    for i, diff in enumerate(diffs):
        if pd.notna(diff) and diff > threshold:
            gaps.append({
                'index': i,
                'timestamp': timestamps[i],
                'gap_duration': str(diff),
                'gap_seconds': diff.total_seconds()
            })

    max_gap = max([g['gap_seconds'] for g in gaps]) if gaps else None

    result = {
        'gaps': gaps,
        'gap_count': len(gaps),
        'max_gap': max_gap,
        'threshold_seconds': threshold.total_seconds() if expected_frequency else threshold
    }
    tprint(f"[OHLCV_VALIDATION] detect_timestamp_gaps -> {len(gaps)} gaps found, max_gap={max_gap}")
    return result

def detect_price_jumps(
    data: pd.DataFrame,
    threshold: float = EXTREME_PRICE_CHANGE_THRESHOLD,
    column: str = 'close'
) -> Dict[str, Any]:
    """
    Detect unusual price jumps that might indicate data errors or flash crashes.

    Args:
        data: DataFrame with price data
        threshold: Percentage change threshold for jumps
        column: Column to analyze

    Returns:
        Dictionary with jump information
    """
    tprint(f"[OHLCV_VALIDATION] detect_price_jumps: data_shape={data.shape if not data.empty else 'empty'}, column={column}, threshold={threshold}")
    if data.empty or column not in data.columns:
        tprint(f"[OHLCV_VALIDATION] detect_price_jumps -> no jumps (empty data or missing column)")
        return {'jumps': [], 'jump_count': 0}

    prices = data[column]
    pct_changes = prices.pct_change().abs()

    jumps = []
    for i, (idx, change) in enumerate(pct_changes.items()):
        if pd.notna(change) and change > threshold:
            jumps.append({
                'index': i,
                'timestamp': idx if isinstance(idx, (datetime, pd.Timestamp)) else None,
                'price': prices.iloc[i],
                'previous_price': prices.iloc[i-1] if i > 0 else None,
                'change_pct': change * 100
            })

    result = {
        'jumps': jumps,
        'jump_count': len(jumps),
        'threshold': threshold,
        'max_jump': max([j['change_pct'] for j in jumps]) if jumps else None
    }
    tprint(f"[OHLCV_VALIDATION] detect_price_jumps -> {len(jumps)} jumps found")
    return result

def detect_volume_spikes(
    data: pd.DataFrame,
    threshold_multiplier: float = 5.0,
    window: int = 20
) -> Dict[str, Any]:
    """
    Detect unusual volume spikes.

    Args:
        data: DataFrame with volume data
        threshold_multiplier: Multiplier of rolling mean to consider a spike
        window: Rolling window for mean calculation

    Returns:
        Dictionary with spike information
    """
    if data.empty or 'volume' not in data.columns:
        return {'spikes': [], 'spike_count': 0}

    volume = data['volume']
    rolling_mean = volume.rolling(window=window, min_periods=window//2).mean()
    threshold = rolling_mean * threshold_multiplier

    spikes = []
    for i, (idx, vol) in enumerate(volume.items()):
        if pd.notna(vol) and pd.notna(rolling_mean.iloc[i]):
            if vol > threshold.iloc[i]:
                spikes.append({
                    'index': i,
                    'timestamp': idx if isinstance(idx, (datetime, pd.Timestamp)) else None,
                    'volume': vol,
                    'mean_volume': rolling_mean.iloc[i],
                    'multiplier': vol / rolling_mean.iloc[i] if rolling_mean.iloc[i] > 0 else None
                })

    return {
        'spikes': spikes,
        'spike_count': len(spikes),
        'threshold_multiplier': threshold_multiplier,
        'max_spike_multiplier': max([s['multiplier'] for s in spikes if s['multiplier']]) if spikes else None
    }

def validate_ohlcv_enhanced(
    data: pd.DataFrame,
    check_gaps: bool = True,
    check_jumps: bool = True,
    check_volume_spikes: bool = True,
    expected_frequency: Optional[str] = None
) -> Dict[str, Any]:
    """
    Comprehensive OHLCV validation with gap, jump, and spike detection.

    Args:
        data: OHLCV DataFrame
        check_gaps: Whether to check for timestamp gaps
        check_jumps: Whether to check for price jumps
        check_volume_spikes: Whether to check for volume spikes
        expected_frequency: Expected data frequency

    Returns:
        Dictionary with validation results

    Raises:
        ValidationError: If critical issues found
    """
    tprint(f"[OHLCV_VALIDATION] validate_ohlcv_enhanced: data_shape={data.shape}, check_gaps={check_gaps}, check_jumps={check_jumps}, check_volume_spikes={check_volume_spikes}")
    # First run standard validation
    validate_market_data(data)

    results = {
        'standard_validation': True,
        'gaps': {},
        'jumps': {},
        'volume_spikes': {}
    }

    errors = []
    warnings = []

    # Check for gaps
    if check_gaps:
        gap_info = detect_timestamp_gaps(data, expected_frequency)
        results['gaps'] = gap_info
        if gap_info['gap_count'] > 0:
            warnings.append(f"Found {gap_info['gap_count']} timestamp gaps")

    # Check for price jumps
    if check_jumps:
        jump_info = detect_price_jumps(data)
        results['jumps'] = jump_info
        if jump_info['jump_count'] > 0:
            warnings.append(f"Found {jump_info['jump_count']} price jumps")

    # Check for volume spikes
    if check_volume_spikes:
        spike_info = detect_volume_spikes(data)
        results['volume_spikes'] = spike_info
        if spike_info['spike_count'] > 0:
            warnings.append(f"Found {spike_info['spike_count']} volume spikes")

    # Log warnings
    if warnings:
        from .validation import tprint_warning
        for warning in warnings:
            tprint_warning(f"⚠️ OHLCV Validation Warning: {warning}")

    tprint(f"[OHLCV_VALIDATION] validate_ohlcv_enhanced -> validation complete with {len(warnings)} warnings")
    return results

def validate_multi_timeframe_consistency(
    dataframes: Dict[str, pd.DataFrame],
    base_timeframe: str = '1T'
) -> bool:
    """
    Validate consistency across multiple timeframes.

    Args:
        dataframes: Dictionary of {timeframe: DataFrame}
        base_timeframe: Base timeframe for comparison

    Returns:
        bool: True if consistent

    Raises:
        ValidationError: If inconsistencies found
    """
    errors = []

    if base_timeframe not in dataframes:
        errors.append(f"Base timeframe {base_timeframe} not found in dataframes")

    if not errors:
        base_df = dataframes[base_timeframe]
        base_prices = base_df['close']

        for tf, df in dataframes.items():
            if tf == base_timeframe:
                continue

            # Resample base to match timeframe
            try:
                resampled = base_prices.resample(tf).last()
                tf_prices = df['close']

                # Check if prices match at common timestamps
                common_timestamps = resampled.index.intersection(tf_prices.index)
                if len(common_timestamps) > 0:
                    resampled_aligned = resampled.loc[common_timestamps]
                    tf_aligned = tf_prices.loc[common_timestamps]

                    # Allow small tolerance for floating point
                    mismatches = (resampled_aligned - tf_aligned).abs() > 0.01
                    mismatch_count = mismatches.sum()

                    if mismatch_count > 0:
                        errors.append(
                            f"Timeframe {tf} has {mismatch_count} price mismatches with base"
                        )
            except Exception as e:
                errors.append(f"Error validating timeframe {tf}: {str(e)}")

    if errors:
        raise ValidationError(
            f"Multi-timeframe consistency validation failed: {'; '.join(errors)}",
            severity=TradingErrorSeverity.HIGH,
            context={'dataframes': list(dataframes.keys()), 'errors': errors}
        )

    return True
