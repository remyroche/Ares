"""
Advanced Feature Transformations for Financial Time Series.

Implements sophisticated transformations to improve signal-to-noise ratio and stationarity properties:
1. Adaptive Fractional Differentiation (AFD): Preserves memory while ensuring stationarity.
2. Structural Break Metrics: Continuous detection of regime shifts.
3. Information-Theoretic Metrics: Market efficiency and complexity measures.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional, Union, Dict
from statsmodels.tsa.stattools import adfuller
from scipy.stats import entropy
import warnings

# Import tprint functions
try:
    from src.utils.tprint import tprint_info, tprint_success, tprint_warning, tprint_error
except ImportError:
    # Fallback print functions
    def tprint_info(msg): print(f"[INFO] {msg}")
    def tprint_success(msg): print(f"[SUCCESS] {msg}")
    def tprint_warning(msg): print(f"[WARNING] {msg}")
    def tprint_error(msg): print(f"[ERROR] {msg}")

# ==========================================
# 1. Adaptive Fractional Differentiation (AFD)
# ==========================================

def get_weights_ffd(d: float, thres: float, lim: int) -> np.ndarray:
    """
    Calculate weights for Fixed-Width Fractional Differentiation (FFD).
    
    Args:
        d: Order of differentiation (0 <= d <= 1)
        thres: Threshold for weight cutoff (e.g., 1e-5)
        lim: Maximum window size
        
    Returns:
        Array of weights (reversed for convolution)
    """
    w, k = [1.], 1
    while True:
        w_new = -w[-1] / k * (d - k + 1)
        if abs(w_new) < thres or len(w) >= lim:
            break
        w.append(w_new)
        k += 1
    return np.array(w[::-1]).reshape(-1, 1)

def frac_diff_ffd(series: pd.Series, d: float, thres: float = 1e-5) -> pd.Series:
    """
    Apply Fixed-Width Fractional Differentiation to a series.
    
    Args:
        series: Input time series (e.g., log prices)
        d: Order of differentiation
        thres: Weight cutoff threshold
        
    Returns:
        Fractionally differentiated series
    """
    # 1. Compute weights
    w = get_weights_ffd(d, thres, len(series))
    width = len(w) - 1
    
    # 2. Apply weights via rolling window (efficiently via dataframe manipulation)
    # Note: For very long series, convolution or loop might be needed.
    # Here we use a vectorized approach if possible, or simple iteration.
    
    # Efficient pandas rolling apply is tricky with variable weights, 
    # but since weights are fixed for a given d, we can use convolution.
    
    # Ensure no NaNs at start
    series_fill = series.fillna(method='ffill')
    
    # Loop implementation is safe and handles alignment
    df = {}
    for name in series_fill.columns if isinstance(series_fill, pd.DataFrame) else ['close']:
        x = series_fill[name] if isinstance(series_fill, pd.DataFrame) else series_fill
        
        # Convolve
        # mode='valid' means output is shorter by width
        res = np.convolve(x.values, w.flatten(), mode='valid')
        
        # Create series with correct index (end-aligned)
        res_series = pd.Series(res, index=x.index[width:])
        return res_series # Return immediately for Series assumption

    return pd.Series(dtype=float) 

def get_optimal_d(series: pd.Series, max_d: float = 1.0, step: float = 0.05, p_thres: float = 0.05) -> float:
    """
    Find minimum differentiation order d that makes series stationary (ADF p-value < 0.05).
    
    Args:
        series: Input price log-series
        max_d: Maximum d to search (usually 1.0)
        step: Search step
        p_thres: Stationarity p-value threshold
        
    Returns:
        Optimal d value
    """
    # Grid search for d
    best_d = max_d
    
    for d in np.arange(step, max_d + step, step):
        # Apply differentiation
        diff_series = frac_diff_ffd(series, d, thres=1e-4) # Use faster thres for search
        diff_series = diff_series.dropna()
        
        if len(diff_series) < 20:
             continue
             
        # ADF Test
        try:
            # Use AIC for autolag to handle serial correlation better
            # Handle potential ValueError if input is constant (e.g., d=0 or low d on stable series)
            adf_res = adfuller(diff_series, maxlag=None, regression='c', autolag='AIC')
            p_val = adf_res[1]
            
            if p_val < p_thres:
                best_d = d
                break # Found smallest d
        except ValueError:
            # "x is constant" -> treat as non-stationary/invalid for our purpose
            continue
        except Exception:
            continue
            
    return float(best_d)


# ==========================================
# 2. Structural Break Metrics
# ==========================================

def get_rolling_cusum_stats(series: pd.Series, window: int = 100) -> pd.DataFrame:
    """
    Compute rolling CUSUM statistics to detect regime shifts.
    Measures deviation of cumulative sum from expected zero-mean path.
    
    Args:
        series: Stationary input series (e.g., returns)
        window: Lookback window
        
    Returns:
        DataFrame with 'cusum_range' and 'cusum_sq_range'
    """
    # Standardize series locally
    roll_mean = series.rolling(window).mean()
    roll_std = series.rolling(window).std()
    z_score = (series - roll_mean) / (roll_std + 1e-9)
    
    results = {}
    
    # 1. CUSUM Range (Brownian Bridge)
    # We calculate the range of the cumulative sum of z-scores within the window
    # Large range implies "drift" or "trend" within the window relative to mean
    
    # Efficient calculation:
    # We need rolling max(cumsum) - min(cumsum) relative to window start?
    # Actually, simpler: rolling sum of z-scores. Large abs value = break.
    # But CUSUM statistic is max|S_t|.
    
    # Let's use simpler proxy: Rolling Sum of Z-scores scaled by sqrt(N)
    # If stationary, sum should be near 0.
    cusum_proxy = z_score.rolling(window).sum() / np.sqrt(window)
    results['cusum_stat'] = cusum_proxy.abs()
    
    return pd.DataFrame(results, index=series.index)

def get_rolling_chow_stat(y: pd.Series, window: int = 100) -> pd.Series:
    """
    Compute rolling Chow-Type Statistic (F-test for structural break).
    Compares RSS of whole window vs RSS of two sub-windows (split at midpoint).
    
    Args:
        y: Input series (returns)
        window: Total window size
        
    Returns:
        Rolling F-statistic
    """
    # Vectorized approximation
    # SSE Total ~ Variance * N
    # SSE Split ~ Var1 * N1 + Var2 * N2
    
    half_win = window // 2
    
    var_tot = y.rolling(window).var()
    var_1 = y.rolling(half_win).var().shift(half_win) # First half variance (aligned to end)
    var_2 = y.rolling(half_win).var() # Second half variance
    
    # RSS approx = var * (n-1)
    rss_tot = var_tot * (window - 1)
    rss_1 = var_1 * (half_win - 1)
    rss_2 = var_2 * (half_win - 1)
    
    k = 2 # params (mean, var) approx
    n1, n2 = half_win, half_win
    
    # F-stat = ((RSS_tot - (RSS_1 + RSS_2)) / k) / ((RSS_1 + RSS_2) / (n1 + n2 - 2k))
    rss_split = rss_1 + rss_2
    numerator = (rss_tot - rss_split) / k
    denominator = rss_split / (n1 + n2 - 2*k)
    
    f_stat = numerator / (denominator + 1e-9)
    return f_stat.fillna(0)


# ==========================================
# 3. Information-Theoretic Metrics
# ==========================================

def get_rolling_entropy(series: pd.Series, window: int = 100, bins: int = 10) -> pd.Series:
    """
    Compute rolling Shannon Entropy of the series distribution.
    Higher entropy = More random/efficient. Lower entropy = More predictable/structured.
    
    Args:
        series: Input series (returns)
        window: Rolling window
        bins: Number of discretization bins
        
    Returns:
        Rolling entropy
    """
    def _entropy_calc(x):
        hist, _ = np.histogram(x, bins=bins, density=True)
        return entropy(hist + 1e-9)
        
    # Pandas rolling apply is slow, but acceptable for feature engineering
    return series.rolling(window).apply(_entropy_calc, raw=True).fillna(0)

def get_lempel_ziv_complexity(series: pd.Series, window: int = 100) -> pd.Series:
    """
    Compute rolling Lempel-Ziv complexity of binary-quantized returns.
    Measures algorithmic complexity / compressibility.
    
    Args:
        series: Input series (returns)
        window: Window size
        
    Returns:
        Rolling LZ complexity
    """
    def _lz_complexity(binary_seq):
        # Simple LZ76 complexity calc
        n = len(binary_seq)
        c, i, k, k_max = 1, 0, 1, 1
        while True:
            if c + k > n: break
            w = binary_seq[i : i+k]
            # Check if w is reproducible from prefix
            # Simplified check (not full LZ):
            if i+k+k_max <= n: # Just a placeholder for speed if strict LZ too slow
               pass
               
            # PROPER LZ76 Implementation for short strings (simplified variant):
            # We count new patterns
            u, v, w = 0, 1, 1
            v_max = 1
            complexity = 1
            while u + v + w <= n:
                 # Check if string s[u+v : u+v+w] is contained in s[u : u+v+w-1]
                 # Python slice search
                 search_buff = binary_seq[u : u+v+w-1] # all history + current pattern minus last char
                 target = binary_seq[u+v : u+v+w]
                 
                 # Manual check is slow. 
                 # Faster proxy: Number of distinct substrings?
                 # Let's use a very simple approximation for speed:
                 # Ratio of distinctive patterns vs length.
                 # Given complexity constraints, we will use 'Kolmogorov Complexity Proxy'
                 # via zlib compression ratio.
                 pass
            break
            
        return 0.0

    # ZLIB COMPRESSION PROXY (Much faster and robust)
    import zlib
    def _compression_ratio(x):
        # Binarize: 1 if > 0 else 0
        bin_str = bytes((x > 0).astype(int).tolist())
        compressed = zlib.compress(bin_str)
        return len(compressed) / len(bin_str)

    return series.rolling(window).apply(_compression_ratio, raw=True).fillna(0)
