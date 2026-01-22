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
    return_details: bool = False
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
        # FIX for MultiIndex (Asset Boundaries)
        if isinstance(series.index, pd.MultiIndex):
            ticker_level = 'ticker' if 'ticker' in series.index.names else 1
            q1_rolling = series.groupby(level=ticker_level, group_keys=False).rolling(window=window, min_periods=min_periods).quantile(quantiles[0])
            q2_rolling = series.groupby(level=ticker_level, group_keys=False).rolling(window=window, min_periods=min_periods).quantile(quantiles[1])
            
            # Align back - rolling on groupby returns (ticker, timestamp, ticker) or similar
            # We need to ensure it matches the original series index
            q1_rolling = q1_rolling.reset_index(level=0, drop=True).reindex(series.index)
            q2_rolling = q2_rolling.reset_index(level=0, drop=True).reindex(series.index)
        else:
            q1_rolling = series.rolling(window=window, min_periods=min_periods).quantile(quantiles[0])
            q2_rolling = series.rolling(window=window, min_periods=min_periods).quantile(quantiles[1])
        
        # Backfill/Forwardfill edges
        q1_rolling = q1_rolling.bfill().ffill()
        q2_rolling = q2_rolling.bfill().ffill()
        
        # Apply safety floors (prevent thresholds from collapsing to near-zero in flat markets)
        # Using 1.0/1.5 relative floors assuming inputs are roughly z-score scaled or similar
        # If raw inputs are small, these floors might be too aggressive, but for "Surprise", 1.0 is standard base.
        # Ideally caller handles scaling, but we add a small epsilon protection.
        q1_rolling = np.maximum(q1_rolling, 1e-6)
        q2_rolling = np.maximum(q2_rolling, q1_rolling + 1e-6)
        
        # Vectorized Zone Classification
        levels = pd.Series(1.0, index=series.index)
        levels[series >= q1_rolling] = 2.0
        levels[series >= q2_rolling] = 3.0
        
        if return_thresholds or return_details:
            # Calculate weight (Severity relative to Zone 2 threshold)
            # If q1_rolling is approx 2.0 sigma, and value is 3.0 sigma, weight = 1.5
            weights = series / q1_rolling
            
            return pd.DataFrame({
                'level': levels,
                'weight': weights,
                'q1_threshold': q1_rolling,
                'q2_threshold': q2_rolling
            })
        return levels
        
    except Exception:
        # Global Fallback
        q1 = series.quantile(fallback_quantiles[0])
        q2 = series.quantile(fallback_quantiles[1])
        levels = pd.cut(
            series, 
            bins=[-np.inf, q1, q2, np.inf], 
            labels=[1.0, 2.0, 3.0]
        ).astype(float)
        return levels.fillna(1.0)
