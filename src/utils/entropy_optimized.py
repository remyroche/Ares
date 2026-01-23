"""
Optimized Entropy Feature Calculations with Numba JIT

This module provides highly optimized entropy-based feature calculations
for financial time series analysis using Numba JIT compilation.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple
from numba import jit, njit, prange, types
from numba.typed import Dict as NumbaDict
import warnings

try:
    from src.utils.tprint import tprint_info, tprint_warning
except ImportError:
    def tprint_info(msg): print(f"[INFO] {msg}")
    def tprint_warning(msg): print(f"[WARNING] {msg}")


@njit(fastmath=True, parallel=True)
def rolling_entropy_numba(
    values: np.ndarray, 
    window: int, 
    n_bins: int = 10
) -> np.ndarray:
    """
    Calculate rolling Shannon entropy using Numba.
    
    Args:
        values: Input array of values
        window: Rolling window size
        n_bins: Number of bins for discretization
        
    Returns:
        Array of rolling entropy values
    """
    n = len(values)
    result = np.full(n, np.nan)
    
    if window >= n:
        return result
    
    for i in prange(window - 1, n):
        window_data = values[i - window + 1:i + 1]
        entropy_val = shannon_entropy_numba(window_data, n_bins)
        result[i] = entropy_val
    
    return result


@njit(fastmath=True)
def shannon_entropy_numba(values: np.ndarray, n_bins: int = 10) -> float:
    """
    Calculate Shannon entropy using Numba.
    
    Args:
        values: Input array of values
        n_bins: Number of bins for discretization
        
    Returns:
        Shannon entropy value
    """
    if len(values) == 0:
        return 0.0
    
    # Filter non-finite values (NaNs, Infs)
    valid_mask = np.isfinite(values)
    clean_values = values[valid_mask]

    if len(clean_values) == 0:
        return 0.0

    # Discretize values
    min_val = np.min(clean_values)
    max_val = np.max(clean_values)
    
    if max_val == min_val:
        return 0.0
    
    bin_width = (max_val - min_val) / n_bins
    if bin_width == 0:
        return 0.0
    
    bins = np.floor((clean_values - min_val) / bin_width).astype(np.int32)
    bins = np.clip(bins, 0, n_bins - 1)
    
    # Calculate entropy
    counts = np.zeros(n_bins, dtype=np.int32)
    for bin_val in bins:
        counts[bin_val] += 1
    
    entropy = 0.0
    total = len(clean_values)
    
    for count in counts:
        if count > 0:
            p = count / total
            entropy -= p * np.log(p)
    
    return entropy / np.log(2)  # Convert to base-2


@njit(fastmath=True)
def lempel_ziv_complexity_numba(
    values: np.ndarray,
    normalize: bool = True,
    max_lookback: int = 5000
) -> np.ndarray:
    """
    Calculate Lempel-Ziv (LZ76) complexity for each position in the series.

    Implements Kaspar-Schuster algorithm (1987) with lookback limit optimization.
    The complexity is reduced from O(N^2) to O(N * max_lookback).
    
    Args:
        values: Input time series (converted to binary internally)
        normalize: Whether to normalize by sequence length
        max_lookback: Maximum history to search for pattern matching (default: 5000)
        
    Returns:
        Array of LZ complexity values (one for each prefix)
    """
    n = len(values)
    complexity_values = np.zeros(n, dtype=np.float64)
    
    if n == 0:
        return complexity_values
    
    # Convert to binary sequence based on median
    median_val = np.median(values)
    binary_seq = (values > median_val).astype(np.int32)
    
    # Kaspar-Schuster Algorithm (1987)
    # c: complexity counter (number of phrases)
    # i: current index (start of new component)
    # l: current component length

    c = 1
    i = 0
    l = 1

    # First element has complexity 1
    complexity_values[0] = 1.0

    while i + l <= n:
        # Check if binary_seq[i : i+l] (target) appears in binary_seq[0 : i+l-1] (history)
        # The history includes previous phrases and the current partial phrase minus one char.
        # We search backwards for efficiency as matches are often recent.
        
        found = False
        target = binary_seq[i : i+l]
        
        # Search backwards from i-1 down to start_search
        # Optimization: Limit lookback to avoid O(N^2) behavior
        start_search = max(0, i - max_lookback)

        for p in range(i - 1, start_search - 1, -1):
            match = True
            for k in range(l):
                if binary_seq[p + k] != target[k]:
                    match = False
                    break
            if match:
                found = True
                break

        if found:
            # Found a match, can extend the current phrase.
            # Complexity remains 'c' for the extended prefix.
            if i + l - 1 < n:
                complexity_values[i + l - 1] = c
            l += 1
        else:
            # No match, this phrase is new.
            # Complexity is 'c' at the end of this new phrase.
            if i + l - 1 < n:
                complexity_values[i + l - 1] = c

            c += 1
            i += l
            l = 1

    # Fill remaining gaps (if any) and normalize
    # The loop sets complexity at specific points (end of phrases).
    # We should fill the complexity for intermediate points.
    # Logic: if we are building phrase 'c', the complexity is 'c'.
    
    current_c = 1.0
    for k in range(n):
        if complexity_values[k] > 0:
            current_c = complexity_values[k]
        else:
            complexity_values[k] = current_c

        if normalize:
            complexity_values[k] = complexity_values[k] / (k + 1)

    return complexity_values


@njit(fastmath=True, parallel=True)
def calculate_trend_conviction_index_numba(
    entropy_values: np.ndarray,
    timestamps: np.ndarray
) -> np.ndarray:
    """
    Calculate Trend Conviction Index (TCI): Delta Entropy / Delta Time.
    
    Args:
        entropy_values: Array of entropy values
        timestamps: Array of timestamps (in seconds)
        
    Returns:
        Array of TCI values
    """
    n = len(entropy_values)
    tci_values = np.zeros(n)
    
    for i in prange(1, n):
        delta_entropy = entropy_values[i] - entropy_values[i-1]
        delta_time = timestamps[i] - timestamps[i-1]
        
        if delta_time > 0:
            tci_values[i] = delta_entropy / delta_time
        else:
            tci_values[i] = 0.0
    
    return tci_values


@njit(fastmath=True, parallel=True)
def calculate_staleness_features_numba(
    current_timestamps: np.ndarray,
    last_update_timestamp: float,
    volatility_values: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate staleness and staleness-adjusted features.
    
    Args:
        current_timestamps: Array of current timestamps
        last_update_timestamp: Timestamp of last update
        volatility_values: Array of volatility values
        
    Returns:
        Tuple of (staleness_seconds, staleness_adjusted_drift)
    """
    n = len(current_timestamps)
    staleness_seconds = np.zeros(n)
    staleness_adjusted_drift = np.zeros(n)
    
    for i in prange(n):
        staleness_seconds[i] = current_timestamps[i] - last_update_timestamp
        
        # Staleness-adjusted drift (simplified version)
        if volatility_values[i] > 0:
            staleness_adjusted_drift[i] = staleness_seconds[i] / volatility_values[i]
        else:
            staleness_adjusted_drift[i] = 0.0
    
    return staleness_seconds, staleness_adjusted_drift


@njit(fastmath=True, parallel=True)
def calculate_drift_proxy_numba(
    current_prices: np.ndarray,
    specialist_prices: np.ndarray
) -> np.ndarray:
    """
    Calculate drift proxy between current and specialist prices.
    
    Args:
        current_prices: Array of current prices
        specialist_prices: Array of specialist prices
        
    Returns:
        Array of drift proxy values
    """
    n = len(current_prices)
    drift_proxy = np.zeros(n)
    
    for i in prange(n):
        if specialist_prices[i] > 0:
            drift_proxy[i] = (current_prices[i] - specialist_prices[i]) / specialist_prices[i]
        else:
            drift_proxy[i] = 0.0
    
    return drift_proxy


@njit(fastmath=True, parallel=True)
def calculate_entropy_statistics_numba(
    entropy_values: np.ndarray,
    window: int = 10
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate rolling entropy statistics.
    
    Args:
        entropy_values: Array of entropy values
        window: Rolling window size
        
    Returns:
        Tuple of (entropy_ma, entropy_std, entropy_zscore)
    """
    n = len(entropy_values)
    entropy_ma = np.full(n, np.nan)
    entropy_std = np.full(n, np.nan)
    entropy_zscore = np.full(n, np.nan)
    
    if window >= n:
        return entropy_ma, entropy_std, entropy_zscore
    
    for i in prange(window - 1, n):
        window_data = entropy_values[i - window + 1:i + 1]
        ma_val = np.mean(window_data)
        std_val = np.std(window_data)
        
        entropy_ma[i] = ma_val
        entropy_std[i] = std_val
        
        if std_val > 0:
            entropy_zscore[i] = (entropy_values[i] - ma_val) / std_val
        else:
            entropy_zscore[i] = 0.0
    
    return entropy_ma, entropy_std, entropy_zscore


def vectorized_entropy_features(
    df: pd.DataFrame,
    entropy_bars: Optional[pd.DataFrame] = None,
    config: Optional[Dict] = None,
    use_numba: bool = True
) -> pd.DataFrame:
    """
    Calculate all entropy features with optional Numba acceleration.
    
    Args:
        df: DataFrame with market data
        entropy_bars: DataFrame with entropy bar data
        config: Configuration dictionary
        use_numba: Whether to use Numba acceleration
        
    Returns:
        DataFrame with entropy features
    """
    if config is None:
        config = {}
    
    features = pd.DataFrame(index=df.index)
    
    # Configuration
    n_bins = config.get('entropy_bins', 10)
    window_size = config.get('entropy_window', 100)
    volatility_window = config.get('volatility_window', 20)
    
    # Use entropy bars if available, otherwise use close prices
    if entropy_bars is not None and not entropy_bars.empty:
        source_data = entropy_bars
        tprint_info("🎯 Using entropy bars for feature calculation")
    else:
        source_data = df
        tprint_info("📊 Using market data for entropy feature calculation")
    
    # Extract required columns
    if 'close' not in source_data.columns:
        raise ValueError("Missing 'close' column for entropy calculations")
    
    close_prices = source_data['close'].values
    timestamps = source_data.index.astype(np.int64) // 10**9  # Convert to seconds
    
    if use_numba:
        tprint_info("🚀 Using Numba-optimized entropy feature calculation")
        
        # Calculate simple returns to match Pandas fallback behavior (pct_change)
        # Pad with 0 at start to maintain length
        # Simple returns: (p[t] / p[t-1]) - 1
        returns = np.concatenate((np.array([0.0]), close_prices[1:] / close_prices[:-1] - 1.0))

        # 1. Rolling entropy features
        for window in [20, 40, 60, 100]:
            if window < len(close_prices):
                # Pass returns instead of close_prices
                entropy_values = rolling_entropy_numba(returns, window, n_bins)
                features[f'entropy_rolling_{window}'] = entropy_values
        
        # 2. Lempel-Ziv complexity
        lz_complexity = lempel_ziv_complexity_numba(close_prices, normalize=True)
        features['lz_complexity'] = lz_complexity
        
        # 3. Trend Conviction Index (if entropy contribution available)
        if 'entropy_contribution' in source_data.columns:
            entropy_contribution = source_data['entropy_contribution'].values
            tci_values = calculate_trend_conviction_index_numba(entropy_contribution, timestamps)
            features['trend_conviction_index'] = tci_values
            
            # Entropy statistics
            entropy_ma, entropy_std, entropy_zscore = calculate_entropy_statistics_numba(entropy_contribution)
            features['entropy_ma'] = entropy_ma
            features['entropy_std'] = entropy_std
            features['entropy_zscore'] = entropy_zscore
        
        # 4. Staleness features
        last_update_time = timestamps[-1] if len(timestamps) > 0 else 0
        
        # Calculate volatility for staleness adjustment
        # Use simple returns already calculated above
        volatility = np.full(len(close_prices), np.std(returns))
        
        if len(returns) > volatility_window:
            for i in range(volatility_window, len(returns)):
                volatility[i] = np.std(returns[i-volatility_window:i])
        
        staleness_seconds, staleness_adjusted_drift = calculate_staleness_features_numba(
            timestamps, last_update_time, volatility
        )
        features['staleness_seconds'] = staleness_seconds
        features['staleness_minutes'] = staleness_seconds / 60.0
        features['staleness_adjusted_drift'] = staleness_adjusted_drift
        
        # 5. Drift proxy (if specialist prices available)
        if 'specialist_close' in source_data.columns:
            specialist_prices = source_data['specialist_close'].values
            drift_proxy = calculate_drift_proxy_numba(close_prices, specialist_prices)
            features['drift_proxy'] = drift_proxy
        else:
            # Use self-referential drift as fallback
            drift_proxy = calculate_drift_proxy_numba(close_prices, close_prices)
            features['drift_proxy'] = drift_proxy
    
    else:
        # Fall back to pandas implementation
        tprint_info("📊 Using pandas implementation for entropy features")
        
        returns = source_data['close'].pct_change()
        
        # Rolling entropy
        for window in [20, 40, 60, 100]:
            if window < len(source_data):
                entropy_values = returns.rolling(window).apply(
                    lambda x: shannon_entropy_numba(x.values, n_bins) if len(x.dropna()) > 0 else 0
                )
                features[f'entropy_rolling_{window}'] = entropy_values
        
        # Lempel-Ziv complexity (simplified)
        features['lz_complexity'] = 0.5  # Placeholder
        
        # Staleness features
        current_time = source_data.index.max().timestamp()
        features['staleness_seconds'] = (source_data.index - pd.Timestamp(current_time, unit='s')).total_seconds()
        features['staleness_minutes'] = features['staleness_seconds'] / 60.0
        
        # Drift proxy
        features['drift_proxy'] = 0.0  # Placeholder
        
        # Trend conviction index
        if 'entropy_contribution' in source_data.columns:
            entropy_diff = source_data['entropy_contribution'].diff().fillna(0)
            time_diff = source_data.index.to_series().diff().dt.total_seconds().fillna(60)
            features['trend_conviction_index'] = entropy_diff / (time_diff + 1e-9)
    
    tprint_info(f"✅ Generated {len(features.columns)} entropy features")
    return features


# Export main functions
__all__ = [
    'vectorized_entropy_features',
    'rolling_entropy_numba',
    'shannon_entropy_numba',
    'lempel_ziv_complexity_numba',
    'calculate_trend_conviction_index_numba',
    'calculate_staleness_features_numba',
    'calculate_drift_proxy_numba',
    'calculate_entropy_statistics_numba'
]
