"""
Panel Data Validation Layer

Validates OHLCV data quality and consistency to prevent silent failures.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import warnings


class ValidationError(Exception):
    """Custom exception for validation failures."""
    pass


class ValidationWarning(UserWarning):
    """Custom warning for non-critical validation issues."""
    pass


def validate_ohlc_relationships(
    open_df: pd.DataFrame,
    high_df: pd.DataFrame,
    low_df: pd.DataFrame,
    close_df: pd.DataFrame,
    raise_on_error: bool = False
) -> Dict[str, int]:
    """
    Validate OHLC price relationships: High >= Open/Close, Low <= Open/Close.
    
    Args:
        open_df, high_df, low_df, close_df: Price DataFrames
        raise_on_error: If True, raise exception on violation
        
    Returns:
        Dict with violation counts
    """
    violations = {
        'high_below_open': 0,
        'high_below_close': 0,
        'low_above_open': 0,
        'low_above_close': 0,
        'high_below_low': 0
    }
    
    # Check High >= Open
    high_below_open = (high_df < open_df).sum().sum()
    violations['high_below_open'] = int(high_below_open)
    
    # Check High >= Close
    high_below_close = (high_df < close_df).sum().sum()
    violations['high_below_close'] = int(high_below_close)
    
    # Check Low <= Open
    low_above_open = (low_df > open_df).sum().sum()
    violations['low_above_open'] = int(low_above_open)
    
    # Check Low <= Close
    low_above_close = (low_df > close_df).sum().sum()
    violations['low_above_close'] = int(low_above_close)
    
    # Check High >= Low
    high_below_low = (high_df < low_df).sum().sum()
    violations['high_below_low'] = int(high_below_low)
    
    total_violations = sum(violations.values())
    
    if total_violations > 0:
        msg = f"OHLC relationship violations: {violations}"
        if raise_on_error:
            raise ValidationError(msg)
        else:
            warnings.warn(msg, ValidationWarning)
    
    return violations


def check_missing_timestamps(
    df: pd.DataFrame,
    expected_freq: str = '1H',
    max_gap_hours: int = 24
) -> Dict[str, any]:
    """
    Check for missing timestamps in time series.
    
    Args:
        df: DataFrame with DatetimeIndex
        expected_freq: Expected frequency (e.g., '1H', '15T')
        max_gap_hours: Maximum acceptable gap in hours
        
    Returns:
        Dict with gap statistics
    """
    if df.empty or len(df) < 2:
        return {'gaps': 0, 'max_gap_hours': 0, 'gap_timestamps': []}
    
    # Expected time range
    full_range = pd.date_range(df.index.min(), df.index.max(), freq=expected_freq)
    
    # Missing timestamps
    missing = full_range.difference(df.index)
    
    # Find consecutive gaps
    if len(missing) > 0:
        gaps = []
        current_gap_start = missing[0]
        prev_ts = missing[0]
        
        for i in range(1, len(missing)):
            ts = missing[i]
            expected_next = prev_ts + pd.Timedelta(expected_freq)
            
            if ts != expected_next:
                # Gap ended, record it
                gap_duration = (prev_ts - current_gap_start).total_seconds() / 3600
                gaps.append((current_gap_start, prev_ts, gap_duration))
                current_gap_start = ts
            
            prev_ts = ts
        
        # Record last gap
        gap_duration = (prev_ts - current_gap_start).total_seconds() / 3600
        gaps.append((current_gap_start, prev_ts, gap_duration))
        
        max_gap = max(g[2] for g in gaps)
        
        if max_gap > max_gap_hours:
            warnings.warn(
                f"Large data gap detected: {max_gap:.1f} hours exceeds threshold of {max_gap_hours}",
                ValidationWarning
            )
        
        return {
            'gaps': len(gaps),
            'max_gap_hours': max_gap,
            'gap_timestamps': [(g[0].isoformat(), g[1].isoformat(), g[2]) for g in gaps[:5]]  # Return first 5
        }
    
    return {'gaps': 0, 'max_gap_hours': 0, 'gap_timestamps': []}


def detect_volume_outliers(
    volume_df: pd.DataFrame,
    threshold_std: float = 5.0
) -> Tuple[int, List[Tuple[str, str, float]]]:
    """
    Detect volume outliers using z-score method.
    
    Args:
        volume_df: Volume DataFrame
        threshold_std: Z-score threshold for outliers
        
    Returns:
        (outlier_count, list of (timestamp, symbol, z_score) tuples)
    """
    if volume_df.empty:
        return 0, []
    
    # Log-transform volume to handle skewness
    log_vol = np.log1p(volume_df.fillna(0))
    
    # Compute z-scores
    mean_vol = log_vol.mean()
    std_vol = log_vol.std()
    z_scores = (log_vol - mean_vol) / (std_vol + 1e-12)
    
    # Find outliers
    outliers_mask = np.abs(z_scores) > threshold_std
    outlier_count = outliers_mask.sum().sum()
    
    # Get top outliers
    outlier_list = []
    if outlier_count > 0:
        for col in z_scores.columns:
            col_outliers = z_scores[col][outliers_mask[col]]
            for idx in col_outliers.index:
                outlier_list.append((idx.isoformat(), col, float(col_outliers.loc[idx])))
    
    # Sort by absolute z-score and return top 10
    outlier_list.sort(key=lambda x: abs(x[2]), reverse=True)
    
    if outlier_count > 0:
        warnings.warn(
            f"Volume outliers detected: {outlier_count} points exceed {threshold_std} std",
            ValidationWarning
        )
    
    return outlier_count, outlier_list[:10]


def validate_timezone_consistency(
    *dfs: pd.DataFrame,
    expected_tz: str = 'UTC'
) -> bool:
    """
    Validate that all DataFrames have consistent timezone.
    
    Args:
        *dfs: Variable number of DataFrames to check
        expected_tz: Expected timezone string
        
    Returns:
        True if all consistent, False otherwise
    """
    for i, df in enumerate(dfs):
        if df.empty:
            continue
        
        if not isinstance(df.index, pd.DatetimeIndex):
            warnings.warn(f"DataFrame {i} does not have DatetimeIndex", ValidationWarning)
            return False
        
        tz = df.index.tz
        if tz is None:
            warnings.warn(f"DataFrame {i} has no timezone information", ValidationWarning)
            return False
        
        if str(tz) != expected_tz:
            warnings.warn(
                f"DataFrame {i} has timezone {tz}, expected {expected_tz}",
                ValidationWarning
            )
            return False
    
    return True


def validate_panel(
    panel: Dict[str, pd.DataFrame],
    raise_on_error: bool = False,
    verbose: bool = True
) -> Dict[str, any]:
    """
    Comprehensive validation of OHLCV panel data.
    
    Args:
        panel: Dict with keys ['open', 'high', 'low', 'close', 'volume']
        raise_on_error: If True, raise exception on critical errors
        verbose: If True, print validation summary
        
    Returns:
        Dict with validation results
    """
    results = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'ohlc_violations': {},
        'missing_timestamps': {},
        'volume_outliers': 0,
        'timezone_consistent': True
    }
    
    required_keys = ['open', 'high', 'low', 'close', 'volume']
    for key in required_keys:
        if key not in panel:
            results['valid'] = False
            results['errors'].append(f"Missing required key: {key}")
    
    if not results['valid']:
        if raise_on_error:
            raise ValidationError(f"Panel validation failed: {results['errors']}")
        return results
    
    o, h, l, c, v = panel['open'], panel['high'], panel['low'], panel['close'], panel['volume']
    
    # 1. Check OHLC relationships
    try:
        ohlc_violations = validate_ohlc_relationships(o, h, l, c, raise_on_error=False)
        results['ohlc_violations'] = ohlc_violations
        if sum(ohlc_violations.values()) > 0:
            results['warnings'].append(f"OHLC violations: {sum(ohlc_violations.values())}")
    except Exception as e:
        results['errors'].append(f"OHLC validation error: {e}")
    
    # 2. Check missing timestamps
    try:
        gap_stats = check_missing_timestamps(c, expected_freq='1H', max_gap_hours=24)
        results['missing_timestamps'] = gap_stats
        if gap_stats['gaps'] > 0:
            results['warnings'].append(f"Data gaps: {gap_stats['gaps']}")
    except Exception as e:
        results['errors'].append(f"Timestamp validation error: {e}")
    
    # 3. Check volume outliers
    try:
        outlier_count, outlier_list = detect_volume_outliers(v, threshold_std=5.0)
        results['volume_outliers'] = outlier_count
        if outlier_count > 0:
            results['warnings'].append(f"Volume outliers: {outlier_count}")
    except Exception as e:
        results['errors'].append(f"Volume validation error: {e}")
    
    # 4. Check timezone consistency
    try:
        tz_consistent = validate_timezone_consistency(o, h, l, c, v, expected_tz='UTC')
        results['timezone_consistent'] = tz_consistent
        if not tz_consistent:
            results['warnings'].append("Timezone inconsistency detected")
    except Exception as e:
        results['errors'].append(f"Timezone validation error: {e}")
    
    # Summary
    if len(results['errors']) > 0:
        results['valid'] = False
    
    if verbose:
        print("=" * 50)
        print("PANEL VALIDATION SUMMARY")
        print("=" * 50)
        print(f"Valid: {results['valid']}")
        print(f"Errors: {len(results['errors'])}")
        print(f"Warnings: {len(results['warnings'])}")
        if results['ohlc_violations']:
            print(f"OHLC Violations: {sum(results['ohlc_violations'].values())}")
        if results['missing_timestamps']:
            print(f"Data Gaps: {results['missing_timestamps']['gaps']}")
        print(f"Volume Outliers: {results['volume_outliers']}")
        print(f"Timezone Consistent: {results['timezone_consistent']}")
        print("=" * 50)
    
    if raise_on_error and not results['valid']:
        raise ValidationError(f"Panel validation failed: {results}")
    
    return results
