"""
Numba-optimized functions for risk regime calculations.

This module provides JIT-compiled versions of computationally intensive
risk regime feature calculations for maximum performance.
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple
from numba import jit, njit, prange
import warnings

try:
    from src.utils.tprint import tprint_info, tprint_warning
except ImportError:
    def tprint_info(msg): print(f"[INFO] {msg}")
    def tprint_warning(msg): print(f"[WARNING] {msg}")


@njit(parallel=True, fastmath=True)
def rolling_std_numba(values: np.ndarray, window: int) -> np.ndarray:
    """
    Fast rolling standard deviation calculation using Numba.
    
    Args:
        values: Input array of values
        window: Rolling window size
        
    Returns:
        Array of rolling standard deviations
    """
    n = len(values)
    result = np.full(n, np.nan)
    
    if window >= n:
        return result
    
    for i in prange(window - 1, n):
        window_data = values[i - window + 1:i + 1]
        mean_val = np.mean(window_data)
        std_val = np.sqrt(np.mean((window_data - mean_val) ** 2))
        result[i] = std_val
    
    return result


@njit(parallel=True, fastmath=True)
def rolling_mean_numba(values: np.ndarray, window: int) -> np.ndarray:
    """
    Fast rolling mean calculation using Numba.
    
    Args:
        values: Input array of values
        window: Rolling window size
        
    Returns:
        Array of rolling means
    """
    n = len(values)
    result = np.full(n, np.nan)
    
    if window >= n:
        return result
    
    for i in prange(window - 1, n):
        window_data = values[i - window + 1:i + 1]
        result[i] = np.mean(window_data)
    
    return result


@njit(fastmath=True)
def calculate_returns_numba(prices: np.ndarray) -> np.ndarray:
    """
    Fast returns calculation using Numba.
    
    Args:
        prices: Array of prices
        
    Returns:
        Array of returns (first element is 0)
    """
    n = len(prices)
    returns = np.zeros(n)
    
    for i in range(1, n):
        if prices[i-1] != 0:
            returns[i] = (prices[i] - prices[i-1]) / prices[i-1]
    
    return returns


@njit(parallel=True, fastmath=True)
def calculate_drawdown_numba(returns: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fast drawdown calculations using Numba.
    
    Args:
        returns: Array of returns
        
    Returns:
        Tuple of (cum_returns, drawdown, max_drawdown)
    """
    n = len(returns)
    cum_returns = np.ones(n)
    drawdown = np.zeros(n)
    max_drawdown = np.zeros(n)
    
    # Calculate cumulative returns
    for i in range(1, n):
        cum_returns[i] = cum_returns[i-1] * (1 + returns[i])
    
    # Calculate drawdown
    peak = cum_returns[0]
    for i in range(n):
        if cum_returns[i] > peak:
            peak = cum_returns[i]
        
        drawdown[i] = (cum_returns[i] - peak) / peak
        max_drawdown[i] = np.min(drawdown[:i+1]) if i > 0 else 0
    
    return cum_returns, drawdown, max_drawdown


@njit(parallel=True, fastmath=True)
def calculate_risk_features_numba(
    prices: np.ndarray,
    volumes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    windows: np.ndarray
) -> dict:
    """
    Calculate all risk features using Numba for maximum performance.
    
    Args:
        prices: Array of close prices
        volumes: Array of volumes
        highs: Array of high prices
        lows: Array of low prices
        windows: Array of window sizes for calculations
        
    Returns:
        Dictionary with calculated features
    """
    n = len(prices)
    returns = calculate_returns_numba(prices)
    
    features = {}
    
    # Calculate features for each window
    for window in windows:
        if window >= n:
            continue
            
        # Volatility
        vol = rolling_std_numba(returns, window)
        features[f'volatility_{window}'] = vol
        
        # Mean returns
        mean_ret = rolling_mean_numba(returns, window)
        features[f'mean_return_{window}'] = mean_ret
        
        # Downside volatility
        downside_returns = np.where(returns < 0, returns, 0)
        downside_vol = rolling_std_numba(downside_returns, window)
        features[f'downside_volatility_{window}'] = downside_vol
        
        # Upside volatility
        upside_returns = np.where(returns > 0, returns, 0)
        upside_vol = rolling_std_numba(upside_returns, window)
        features[f'upside_volatility_{window}'] = upside_vol
        
        # Volume features
        vol_ma = rolling_mean_numba(volumes, window)
        features[f'volume_ma_{window}'] = vol_ma
        
        # Price range features
        price_range = (highs - lows) / prices
        range_ma = rolling_mean_numba(price_range, window)
        features[f'range_ma_{window}'] = range_ma
        
        # Volume-price correlation (simplified)
        vol_norm = (volumes - vol_ma) / (rolling_std_numba(volumes, window) + 1e-8)
        price_norm = (prices - rolling_mean_numba(prices, window)) / (rolling_std_numba(prices, window) + 1e-8)
        
        # Simple rolling correlation approximation
        vol_price_corr = np.zeros(n)
        for i in prange(window - 1, n):
            vol_window = vol_norm[i - window + 1:i + 1]
            price_window = price_norm[i - window + 1:i + 1]
            if len(vol_window) > 1 and len(price_window) > 1:
                corr = np.corrcoef(vol_window, price_window)[0, 1]
                vol_price_corr[i] = corr if not np.isnan(corr) else 0
        
        features[f'vol_price_corr_{window}'] = vol_price_corr
    
    # Drawdown features
    cum_returns, drawdown, max_drawdown = calculate_drawdown_numba(returns)
    features['cum_returns'] = cum_returns
    features['drawdown'] = drawdown
    features['max_drawdown'] = max_drawdown
    
    # Additional risk metrics
    features['returns'] = returns
    
    return features


@njit(fastmath=True)
def calculate_entropy_numba(values: np.ndarray, n_bins: int = 10) -> float:
    """
    Calculate Shannon entropy of a value array using Numba.
    
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


def vectorized_risk_features(
    df: pd.DataFrame,
    windows: Optional[list] = None,
    use_numba: bool = True
) -> pd.DataFrame:
    """
    Calculate risk features with optional Numba acceleration.
    
    Args:
        df: DataFrame with OHLCV data
        windows: List of window sizes (default: [20, 40, 60, 80, 100])
        use_numba: Whether to use Numba acceleration
        
    Returns:
        DataFrame with calculated risk features
    """
    if windows is None:
        windows = [20, 40, 60, 80, 100]
    
    required_cols = ['close', 'high', 'low', 'volume']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Extract numpy arrays
    prices = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    volumes = df['volume'].values
    
    if use_numba:
        # Use Numba-optimized calculation
        features_dict = calculate_risk_features_numba(
            prices, volumes, highs, lows, np.array(windows)
        )
        
        # Convert back to DataFrame
        result = pd.DataFrame(index=df.index)
        
        for key, values in features_dict.items():
            if key != 'returns':  # Skip raw returns, already handled
                result[key] = values
        
        # Add additional calculated features
        returns = features_dict['returns']
        result['returns'] = returns
        
        # Risk stress index (vectorized)
        vol_20 = features_dict.get('volatility_20', np.zeros_like(returns))
        vol_zscore_20 = (vol_20 - rolling_mean_numba(vol_20, 100)) / (rolling_std_numba(vol_20, 100) + 1e-8)
        
        drawdown = features_dict.get('drawdown', np.zeros_like(returns))
        max_drawdown = features_dict.get('max_drawdown', np.zeros_like(returns))
        
        risk_stress_index = (
            0.3 * (vol_zscore_20 > 1).astype(np.float64) +
            0.3 * (max_drawdown < -0.05).astype(np.float64) +
            0.2 * (returns * volumes > 0).astype(np.float64) +  # Simplified volume divergence
            0.2 * (np.gradient(drawdown) < -0.01).astype(np.float64)
        )
        
        result['risk_stress_index'] = risk_stress_index
        result['risk_appetite'] = 1 - risk_stress_index
        
    else:
        # Fall back to pandas implementation
        result = pd.DataFrame(index=df.index)
        returns = df['close'].pct_change()
        result['returns'] = returns
        
        for window in windows:
            # Basic rolling calculations
            result[f'volatility_{window}'] = returns.rolling(window).std()
            result[f'mean_return_{window}'] = returns.rolling(window).mean()
            
            # Downside/upside volatility
            downside_returns = returns.where(returns < 0, 0)
            upside_returns = returns.where(returns > 0, 0)
            result[f'downside_volatility_{window}'] = downside_returns.rolling(window).std()
            result[f'upside_volatility_{window}'] = upside_returns.rolling(window).std()
            
            # Volume features
            result[f'volume_ma_{window}'] = df['volume'].rolling(window).mean()
            
            # Price range
            price_range = (df['high'] - df['low']) / df['close']
            result[f'range_ma_{window}'] = price_range.rolling(window).mean()
        
        # Drawdown calculations
        cum_returns = (1 + returns).cumprod()
        running_max = cum_returns.expanding().max()
        drawdown = (cum_returns - running_max) / running_max
        max_drawdown = drawdown.expanding().min()
        
        result['cum_returns'] = cum_returns
        result['drawdown'] = drawdown
        result['max_drawdown'] = max_drawdown
    
    return result


# Export main functions
__all__ = [
    'vectorized_risk_features',
    'rolling_std_numba',
    'rolling_mean_numba', 
    'calculate_returns_numba',
    'calculate_drawdown_numba',
    'calculate_risk_features_numba',
    'calculate_entropy_numba'
]
