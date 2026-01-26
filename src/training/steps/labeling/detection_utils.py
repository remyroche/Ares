"""
Detection Utils

Shared utilities for adaptive event detection, replacing fixed threshold logic
with rolling quantile barriers (De Prado style adaptation).
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Union

def detect_rolling_quantile_surprises(
    series: pd.Series,
    window: int = 500,
    quantiles: Tuple[float, float] = (0.96, 0.98),
    min_periods: Optional[int] = None,
    fallback_quantiles: Tuple[float, float] = (0.96, 0.98),
    return_thresholds: bool = False,
    return_details: bool = False,
    min_coverage: Optional[float] = None
) -> Union[pd.Series, pd.DataFrame]:
    """
    Detect anomalies using adaptive rolling quantile thresholds.
    
    Adherence to De Prado:
    - Adaptive: Thresholds evolve with local regime (non-stationarity).
    - Robust: Uses quantiles instead of standard deviation (fat tails).
    
    Args:
        series: Input time series (e.g., probability errors or signal magnitude)
        window: Rolling window size
        quantiles: Tuple of (Zone 2, Zone 3) percentiles (e.g., Top 4%, Top 2%)
        min_periods: Minimum observations for rolling window
        fallback_quantiles: Global quantiles to use if rolling fails
        return_thresholds: Deprecated, use return_details.
        return_details: If True, returns DataFrame with ['level', 'weight', 'q1_threshold', 'q2_threshold']
                        Weight = Series Value / Q1 Threshold (Relative Severity)
                           
    Returns:
        pd.Series of Zone Levels (1.0=Normal, 2.0=High, 3.0=Extreme)
        OR pd.DataFrame with details if requested.
    """
    if series.empty:
        return pd.Series(index=series.index)
        
    if min_periods is None:
        min_periods = min(window, 100)
        
    try:
        q1_raw, q2_raw = quantiles
        q1 = float(np.clip(q1_raw, 0.0, 1.0))
        q2 = float(np.clip(q2_raw, 0.0, 1.0))
        if q2 <= q1:
            q2 = min(1.0, q1 + 1e-6)
        # FIX for MultiIndex (Asset Boundaries)
        if isinstance(series.index, pd.MultiIndex):
            ticker_level = 'ticker' if 'ticker' in series.index.names else 1
            q1_rolling = series.groupby(level=ticker_level, group_keys=False).rolling(window=window, min_periods=min_periods).quantile(q1)
            q2_rolling = series.groupby(level=ticker_level, group_keys=False).rolling(window=window, min_periods=min_periods).quantile(q2)
            
            # Align back - rolling on groupby returns (ticker, timestamp, ticker) or similar
            # We need to ensure it matches the original series index
            q1_rolling = q1_rolling.reset_index(level=0, drop=True).reindex(series.index)
            q2_rolling = q2_rolling.reset_index(level=0, drop=True).reindex(series.index)
        else:
            q1_rolling = series.rolling(window=window, min_periods=min_periods).quantile(q1)
            q2_rolling = series.rolling(window=window, min_periods=min_periods).quantile(q2)
        
        # Backfill/Forwardfill edges
        q1_rolling = q1_rolling.bfill().ffill()
        q2_rolling = q2_rolling.bfill().ffill()
        
        # Apply safety floors
        q1_rolling = np.maximum(q1_rolling, 1e-6)
        
        # Coverage Assurance Logic
        # If min_coverage is requested (e.g. 0.04), ensure we don't under-select due to strict rolling quantiles
        if min_coverage is not None and min_coverage > 0:
            # Check current coverage
            mask = series >= q1_rolling
            current_coverage = mask.mean()
            
            if current_coverage < min_coverage * 0.8: # Allow 20% tolerance
                # Relax threshold using global quantile as clamp
                # This ensures we pick at least the global top N% if local rolling is too strict
                q1_global = series.quantile(1.0 - min_coverage)
                q1_rolling = np.minimum(q1_rolling, q1_global)

        q2_rolling = np.maximum(q2_rolling, q1_rolling + 1e-6)
        
        # Vectorized Zone Classification
        levels = pd.Series(1.0, index=series.index)
        levels[series >= q1_rolling] = 2.0
        levels[series >= q2_rolling] = 3.0
        
        if return_thresholds or return_details:
            # Calculate weight (Severity relative to Zone 2 threshold)
            # If q1_rolling is approx 2.0 sigma, and value is 3.0 sigma, weight = 1.5
            weights = series.abs() / q1_rolling
            
            return pd.DataFrame({
                'level': levels,
                'weight': weights,
                'q1_threshold': q1_rolling,
                'q2_threshold': q2_rolling
            })
        return levels
        
    except Exception:
        # Global Fallback
        fb_q1 = float(np.clip(fallback_quantiles[0], 0.0, 1.0))
        fb_q2 = float(np.clip(fallback_quantiles[1], 0.0, 1.0))
        if fb_q2 <= fb_q1:
            fb_q2 = min(1.0, fb_q1 + 1e-6)
        q1 = series.quantile(fb_q1)
        q2 = series.quantile(fb_q2)
        if q2 <= q1:
            q2 = q1 + 1e-6
        levels = pd.cut(
            series,
            bins=[-np.inf, q1, q2, np.inf],
            labels=[1.0, 2.0, 3.0],
            duplicates='drop'
        ).astype(float)
        levels = levels.fillna(1.0)

        if return_thresholds or return_details:
            q1_series = pd.Series(q1, index=series.index)
            q2_series = pd.Series(q2, index=series.index)
            weights = series.abs() / np.maximum(q1_series, 1e-6)
            return pd.DataFrame({
                'level': levels,
                'weight': weights,
                'q1_threshold': q1_series,
                'q2_threshold': q2_series
            })

        return levels
