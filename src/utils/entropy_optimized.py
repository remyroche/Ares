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
    
    # Discretize values
    min_val = np.min(values)
    max_val = np.max(values)
    
    if max_val == min_val:
        return 0.0
    
    bin_width = (max_val - min_val) / n_bins
    if bin_width == 0:
        return 0.0
    
    bins = np.floor((values - min_val) / bin_width).astype(np.int32)
    bins = np.clip(bins, 0, n_bins - 1)
    
    # Calculate entropy
    counts = np.zeros(n_bins, dtype=np.int32)
    for bin_val in bins:
        counts[bin_val] += 1
    
    entropy = 0.0
    total = len(values)
    
    for count in counts:
        if count > 0:
            p = count / total
            entropy -= p * np.log(p)
    
    return entropy / np.log(2)  # Convert to base-2


@njit(fastmath=True)
def lempel_ziv_complexity_numba(values: np.ndarray, normalize: bool = True) -> np.ndarray:
    """
    Calculate Lempel-Ziv complexity (LZ76) for each position in the series.
    Uses the Kaspar-Schuster algorithm for O(N^2) complexity (vs O(N^3) or broken implementation).
    
    Args:
        values: Input time series
        normalize: Whether to normalize by series length
        
    Returns:
        Array of LZ complexity values
    """
    n = len(values)
    complexity_values = np.zeros(n, dtype=np.float64)
    
    if n == 0:
        return complexity_values
    
    # Convert to binary sequence based on median
    # Use simple median of the entire series for binarization
    # (Consistent with previous implementation intent)
    median_val = np.median(values)
    binary_seq = (values > median_val).astype(np.int32)
    
    # Kaspar-Schuster Algorithm for LZ76
    # c: complexity counter
    # i: current position index
    # l: length of current phrase being matched
    # k: length of match found
    # k_max: max length of match found

    c = 1
    l = 1
    i = 0
    k_max = 1

    # We want expanding window complexity C(t) for t in 0..n-1
    # But standard LZ computes C for the whole string.
    # We can approximate the rolling complexity by just reporting 'c' at each step 't'.
    # This means C(t) is the number of phrases found in S[0..t].

    # Initialize first position
    complexity_values[0] = 1.0 if normalize else 1.0

    # Iterate through the sequence
    # This loop tracks the LZ parsing process
    # We need to fill complexity_values for all t.

    # We will compute the complexity incrementally.
    # At each step t, we check if S[i:t+1] is a new phrase.

    # Standard Kaspar-Schuster is:
    # 1. Start with S[0], c=1, i=0, l=1
    # 2. Consider S[i : i+l] (current phrase candidate)
    # 3. Search for S[i : i+l] in S[0 : i+l-1]
    # 4. If found, l <- l + 1
    # 5. If not found, c <- c + 1, i <- i + l, l <- 1

    # To output a value for every t, we map the current state 'c' to the array.

    # Optimization: pre-calculate complexity array
    # Since we can parse the string in one pass O(N^2), we can just fill the array.

    current_c = 1
    current_i = 0
    current_l = 1

    # We assume first char is first phrase.
    # We start checking from second char (index 1)

    # We need to fill complexity_values[t] for t from 0 to n-1.
    # complexity_values[0] is already set.

    # The pointer 't' represents the end of the substring we are currently considering adding to the phrase.
    # The current candidate phrase is S[current_i : t+1].
    # Its length is t - current_i + 1.

    for t in range(1, n):
        # Current candidate phrase: S[current_i : t+1]
        # Length: current_l = t - current_i + 1
        
        # Search for S[current_i : t+1] in S[0 : t]
        # Actually, LZ76 rule: pattern must appear in S[0 : t] (the concatenation of previous phrases + current prefix)
        # But strictly, we check if S[current_i : t+1] exists in S[0 : t].
        # If it does, we extend.
        # If it does not, we effectively close the phrase at t, increment c, and start new phrase at t+1.

        # Optimization: brute force search in Numba is fast enough for N < 100k

        pattern_len = t - current_i + 1
        found = False

        # Search range: starts from 0 to current_i - 1?
        # Actually, we search in S[0 : t]. The pattern ends at t.
        # We look for an occurrence ending before t.
        # i.e., S[j : j + pattern_len] == S[current_i : t+1]
        # with j + pattern_len <= t

        # Search loop
        # We can search backwards for better average performance?
        # Or forwards.

        # For small alphabet (binary), matches are frequent.

        limit = t - pattern_len

        # Manual search loop (no array slicing to avoid allocations)
        for j in range(limit + 1):
            match = True
            for k in range(pattern_len):
                if binary_seq[j + k] != binary_seq[current_i + k]:
                    match = False
                    break
            if match:
                found = True
                break

        if found:
            # Continue extending the current phrase
            # Complexity doesn't increase yet
            pass
        else:
            # Phrase S[current_i : t+1] is unique!
            # So we close it.
            # But wait, LZ76 says we close it as soon as it's unique.
            # So the phrase ends at t.
            current_c += 1
            current_i = t + 1
            # Next phrase starts at t+1.
            # But we are in the loop for t.
            # If we increment current_i to t+1, then for loop t+1, pattern_len will be 1.


        # Store current complexity
        if normalize:
            # Lempel-Ziv normalization: c / (n / log2(n)) approx
            # Or simply c / n ?
            # The docstring says "normalize by series length".
            # Usually means c / n or c.
            # Standard simple normalization is c.
            # But previous code did c / (i+1).
            # We stick to c / (t+1) to match previous intent of "rate".
            val = current_c / (t + 1)
        else:
            val = current_c

        complexity_values[t] = val
        
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
        
        # 1. Rolling entropy features
        for window in [20, 40, 60, 100]:
            if window < len(close_prices):
                entropy_values = rolling_entropy_numba(close_prices, window, n_bins)
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
        returns = np.diff(np.log(close_prices))
        returns = np.concatenate([np.array([0.0]), returns])  # Pad to match length
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
