"""
Adaptive Fractional Differentiation (AFML Chapter 5)

Finds minimal differentiation order 'd' that achieves stationarity
while preserving maximum memory.
"""
import numpy as np
import pandas as pd
from typing import Tuple, Optional
from statsmodels.tsa.stattools import adfuller
import warnings


def get_weights_ffd(d: float, thres: float = 1e-5) -> np.ndarray:
    """
    Compute weights for Fixed-Width Window Fractional Differentiation.
    
    Args:
        d: Differentiation order
        thres: Threshold for weight truncation
        
    Returns:
        Array of weights
    """
    w = [1.0]
    k = 1
    while True:
        w_ = -w[-1] / k * (d - k + 1)
        if abs(w_) < thres:
            break
        w.append(w_)
        k += 1
    w = np.array(w[::-1])
    return w


def frac_diff_ffd(series: pd.Series, d: float, thres: float = 1e-5) -> pd.Series:
    """
    Apply fractional differentiation with fixed-width window.
    
    Args:
        series: Input time series
        d: Differentiation order
        thres: Weight truncation threshold
        
    Returns:
        Fractionally differentiated series
    """
    w = get_weights_ffd(d, thres)
    width = len(w) - 1
    
    # Convolve
    output = pd.Series(index=series.index, dtype=float)
    for i in range(width, len(series)):
        output.iloc[i] = np.dot(w, series.iloc[i-width:i+1])
    
    return output


def adf_test(series: pd.Series, max_lag: Optional[int] = None) -> Tuple[float, float, bool]:
    """
    Perform Augmented Dickey-Fuller test for stationarity.
    
    Args:
        series: Time series to test
        max_lag: Maximum lag for ADF test
        
    Returns:
        (adf_stat, p_value, is_stationary)
    """
    series_clean = series.dropna()
    
    if len(series_clean) < 50:
        return np.nan, 1.0, False
    
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            adf_result = adfuller(series_clean, maxlag=max_lag, regression='c', autolag='AIC')
        
        adf_stat = adf_result[0]
        p_value = adf_result[1]
        is_stationary = p_value < 0.05
        
        return adf_stat, p_value, is_stationary
        
    except Exception as e:
        return np.nan, 1.0, False


def find_min_ffd(
    series: pd.Series,
    d_range: Tuple[float, float] = (0.0, 1.0),
    step: float = 0.05,
    thres: float = 1e-5,
    target_pvalue: float = 0.05
) -> Tuple[float, pd.Series, dict]:
    """
    Find minimum d that achieves stationarity (AFML 5.4.1).
    
    Binary search for optimal d:
    - Start from d=0 (no differencing)
    - Increase d until ADF test passes
    - Return minimal d that makes series stationary
    
    Args:
        series: Input time series
        d_range: Range to search (min_d, max_d)
        step: Initial step size for search
        thres: Weight truncation threshold
        target_pvalue: P-value threshold for stationarity
        
    Returns:
        (optimal_d, transformed_series, diagnostics)
    """
    min_d, max_d = d_range
    
    # Test if already stationary
    _, p_val, is_stat = adf_test(series)
    if is_stat:
        return 0.0, series, {'p_value': p_val, 'iterations': 0, 'already_stationary': True}
    
    # Binary search
    best_d = max_d
    best_series = None
    best_p_value = 1.0
    iterations = 0
    
    d_values = np.arange(min_d + step, max_d + step, step)
    
    for d in d_values:
        iterations += 1
        
        # Apply fractional differentiation
        series_ffd = frac_diff_ffd(series, d, thres)
        
        # Test stationarity
        _, p_val, is_stat = adf_test(series_ffd)
        
        if is_stat and p_val < best_p_value:
            best_d = d
            best_series = series_ffd
            best_p_value = p_val
            
            # Found minimal d, can exit early
            if p_val < target_pvalue:
                break
    
    if best_series is None:
        # Fallback to max_d if nothing worked
        best_series = frac_diff_ffd(series, max_d, thres)
        best_d = max_d
    
    diagnostics = {
        'p_value': best_p_value,
        'iterations': iterations,
        'already_stationary': False,
        'd_tested': list(d_values[:iterations])
    }
    
    return best_d, best_series, diagnostics


def apply_adaptive_ffd_panel(
    panel_df: pd.DataFrame,
    d_range: Tuple[float, float] = (0.0, 1.0),
    step: float = 0.1,
    thres: float = 1e-5,
    fallback_d: float = 0.4
) -> Tuple[pd.DataFrame, dict]:
    """
    Apply adaptive FFD to a panel of time series (e.g., multi-asset prices).
    
    Args:
        panel_df: DataFrame with time series in columns
        d_range: Range to search for each series
        step: Step size for d search
        thres: Weight truncation threshold
        fallback_d: Fallback d if search fails
        
    Returns:
        (transformed_panel, diagnostics_dict)
    """
    output = pd.DataFrame(index=panel_df.index, columns=panel_df.columns)
    diagnostics = {}
    
    for col in panel_df.columns:
        series = panel_df[col].dropna()
        
        if len(series) < 100:
            # Not enough data, use fallback
            output[col] = frac_diff_ffd(series, fallback_d, thres).reindex(panel_df.index)
            diagnostics[col] = {'d': fallback_d, 'fallback': True}
            continue
        
        try:
            d_opt, series_ffd, diag = find_min_ffd(series, d_range, step, thres)
            output[col] = series_ffd.reindex(panel_df.index)
            diagnostics[col] = {'d': d_opt, **diag}
        except Exception as e:
            # Fallback on error
            output[col] = frac_diff_ffd(series, fallback_d, thres).reindex(panel_df.index)
            diagnostics[col] = {'d': fallback_d, 'fallback': True, 'error': str(e)}
    
    return output, diagnostics
