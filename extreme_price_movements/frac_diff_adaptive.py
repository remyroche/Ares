"""
Adaptive Fractional Differentiation (AFML Chapter 5)

Finds minimal differentiation order 'd' that achieves stationarity
while preserving maximum memory.
"""
import numpy as np
import pandas as pd
from typing import Tuple, Optional, Union
from statsmodels.tsa.stattools import adfuller
import warnings
from functools import lru_cache
from numba import jit

@lru_cache(maxsize=256)
def get_weights_ffd(d: float, thres: float = 1e-5) -> np.ndarray:
    """
    Compute weights for Fixed-Width Window Fractional Differentiation.
    
    Args:
        d: Differentiation order
        thres: Threshold for weight truncation
        
    Returns:
        Array of weights
    """
    d = float(d)
    w = [1.0]
    k = 1
    while True:
        w_ = -w[-1] / k * (d - k + 1)
        if abs(w_) < thres:
            break
        w.append(w_)
        k += 1
    w = np.array(w[::-1], dtype=np.float64)
    return w

@jit(nopython=True, nogil=True, cache=True)
def _numba_apply_weights(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Apply convolution with weights, handling NaNs.
    """
    n = len(x)
    window = len(weights)
    out = np.full(n, np.nan, dtype=np.float64)

    # If window is larger than data, return all NaNs
    if window > n:
        return out

    # Convolution
    # x_tilde_t = sum(w_k * x_{t-k}) for k=0..window-1
    # weights are [w_{window-1}, ..., w_0] based on get_weights_ffd returning w[::-1]
    # Wait, get_weights_ffd returns w[::-1].
    # Original code: w = np.array(w[::-1])
    # loop: np.dot(w, series.iloc[i-width:i+1])
    # series.iloc[i-width:i+1] has length width+1.
    # Wait, original code:
    # w = get_weights_ffd(d, thres) -> returns array of length K.
    # width = len(w) - 1.
    # range(width, len(series)):
    #   dot(w, series[i-width : i+1]) -> length width+1.
    # So w has length width+1.
    # Yes. w includes w_0, w_1, ... w_m.
    # Original get_weights_ffd logic:
    # w builds [w_0, w_1, ... w_m].
    # Then returns w[::-1] -> [w_m, ..., w_1, w_0].
    # dot product with series[i-width : i+1] (which is [x_{t-m}, ..., x_t])
    # aligns [w_m, ..., w_0] with [x_{t-m}, ..., x_t].
    # So w_m * x_{t-m} + ... + w_0 * x_t.
    # This is correct.

    for i in range(window - 1, n):
        val = 0.0
        valid = True
        for k in range(window):
            # i-k goes from i down to i-(window-1)
            # x index: i-window+1+k for forward iteration?
            # Let's align carefully.
            # weights[k] corresponds to x at some index.
            # We want dot product of weights and x window.
            # x window is x[i-window+1 : i+1] -> length window.
            # weights is length window.
            # weights[0] is w_m (oldest), weights[-1] is w_0 (current).
            # x[i-window+1] is oldest. x[i] is current.
            # So weights[k] multiplies x[i - window + 1 + k].

            curr_x = x[i - window + 1 + k]
            if np.isnan(curr_x):
                valid = False
                break
            val += weights[k] * curr_x

        if valid:
            out[i] = val

    return out

def _frac_diff_ffd_numpy(x: np.ndarray, d: float, thres: float = 1e-5) -> np.ndarray:
    w = get_weights_ffd(d, thres)
    return _numba_apply_weights(x, w)

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
    x = series.to_numpy(dtype=np.float64)
    out = _frac_diff_ffd_numpy(x, d, thres)
    return pd.Series(out, index=series.index, name=series.name)

def adf_test(
    series: Union[pd.Series, np.ndarray],
    max_lag: int = 5,
    max_len: int = 5000
) -> Tuple[float, float, bool]:
    """
    Perform Augmented Dickey-Fuller test for stationarity.
    
    Args:
        series: Time series to test
        max_lag: Maximum lag for ADF test (fixed to speed up)
        max_len: Maximum length of tail to test (speed up)
        
    Returns:
        (adf_stat, p_value, is_stationary)
    """
    if hasattr(series, 'values'):
        series = series.values

    # Remove NaNs
    mask = ~np.isnan(series)
    series_clean = series[mask]

    # Use tail
    if len(series_clean) > max_len:
        series_clean = series_clean[-max_len:]
    
    if len(series_clean) < 20:
        return np.nan, 1.0, False
    
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Use autolag=None and fixed maxlag for speed
            adf_result = adfuller(series_clean, maxlag=max_lag, regression='c', autolag=None)
        
        adf_stat = adf_result[0]
        p_value = adf_result[1]
        is_stationary = p_value < 0.05
        
        return adf_stat, p_value, is_stationary
        
    except Exception:
        return np.nan, 1.0, False


def find_min_ffd(
    series: pd.Series,
    d_range: Tuple[float, float] = (0.0, 1.0),
    step: float = 0.05,  # Kept for signature compatibility, but used as initial guide
    thres: float = 1e-5,
    target_pvalue: float = 0.05
) -> Tuple[float, pd.Series, dict]:
    """
    Find minimum d that achieves stationarity (AFML 5.4.1).
    Optimized with coarse-to-fine search.
    
    Args:
        series: Input time series
        d_range: Range to search (min_d, max_d)
        step: Step size (ignored in optimized version, uses adaptive steps)
        thres: Weight truncation threshold
        target_pvalue: P-value threshold for stationarity
        
    Returns:
        (optimal_d, transformed_series, diagnostics)
    """
    min_d, max_d = d_range
    series_arr = series.to_numpy(dtype=np.float64)
    
    # 1. Test original
    _, p_val, is_stat = adf_test(series_arr)
    if is_stat and p_val < target_pvalue:
        return 0.0, series, {'p_value': p_val, 'iterations': 0, 'already_stationary': True}
    
    # 2. Coarse Search
    coarse_step = 0.1
    start_d = min_d
    if start_d == 0.0:
        start_d += coarse_step

    d_coarse = np.arange(start_d, max_d + 1e-9, coarse_step)
    
    found_coarse = False
    coarse_pass_d = max_d
    iterations = 0
    d_tested = []
    
    for d in d_coarse:
        iterations += 1
        d_tested.append(d)
        out = _frac_diff_ffd_numpy(series_arr, d, thres)
        _, p_val, is_stat = adf_test(out)
        
        if is_stat and p_val < target_pvalue:
            coarse_pass_d = d
            found_coarse = True
            break
            
    # 3. Fine Search
    # Search in (previous_coarse, coarse_pass_d]
    fine_step = 0.01
    start_fine = max(min_d, coarse_pass_d - coarse_step + fine_step)
    end_fine = coarse_pass_d

    # If coarse step didn't find anything, we might be at max_d.
    # If we found something at 0.1, we search 0.01..0.1.
    
    d_fine = np.arange(start_fine, end_fine - fine_step/2, fine_step)
    # Ensure we include end_fine if not already covered (arange excludes end usually)
    # Actually, we want to stop BEFORE end_fine, because end_fine is already known to pass (or is max_d).
    # We want to see if a smaller d also passes.
    
    best_d = coarse_pass_d
    best_out = None
    best_p = 1.0

    # Scan fine range
    for d in d_fine:
        # Avoid re-testing coarse_pass_d
        if abs(d - coarse_pass_d) < 1e-9:
            continue

        iterations += 1
        d_tested.append(d)
        out = _frac_diff_ffd_numpy(series_arr, d, thres)
        _, p_val, is_stat = adf_test(out)

        if is_stat and p_val < target_pvalue:
            best_d = d
            best_out = out
            best_p = p_val
            break # Found smaller d that passes

    # If fine search didn't yield better d, verify coarse_pass_d again (compute output)
    if best_out is None:
        best_d = coarse_pass_d
        best_out = _frac_diff_ffd_numpy(series_arr, best_d, thres)
        _, best_p, _ = adf_test(best_out)

    diagnostics = {
        'p_value': best_p,
        'iterations': iterations,
        'already_stationary': False,
        'd_tested': d_tested
    }
    
    return best_d, pd.Series(best_out, index=series.index, name=series.name), diagnostics


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
    # Pre-allocate output numpy array
    n_rows, n_cols = panel_df.shape
    out_data = np.full((n_rows, n_cols), np.nan, dtype=np.float32)
    diagnostics = {}
    
    cols = panel_df.columns
    idx = panel_df.index

    for i, col in enumerate(cols):
        series = panel_df[col]
        # Dropna logic: we extract clean part for finding d, but apply to full (aligned)
        # To match original behavior which did `series = panel_df[col].dropna()`,
        # we need to be careful. The original code returned reindexed series.
        # Here we will compute on full array (with NaNs) using our NaN-safe convolution.

        # Extract valid part for find_min_ffd
        series_valid = series.dropna()
        
        if len(series_valid) < 100:
            # Fallback
            d_use = fallback_d
            out_vec = _frac_diff_ffd_numpy(series.to_numpy(dtype=np.float64), d_use, thres)
            out_data[:, i] = out_vec
            diagnostics[col] = {'d': d_use, 'fallback': True}
            continue
        
        try:
            # find_min_ffd expects Series to preserve index for return, but we optimized it to work with arrays internally.
            # We pass series_valid to finding logic.
            d_opt, _, diag = find_min_ffd(series_valid, d_range, step, thres)

            # Apply to FULL series (including leading NaNs if any)
            out_vec = _frac_diff_ffd_numpy(series.to_numpy(dtype=np.float64), d_opt, thres)
            out_data[:, i] = out_vec
            diagnostics[col] = {'d': d_opt, **diag}

        except Exception as e:
            d_use = fallback_d
            out_vec = _frac_diff_ffd_numpy(series.to_numpy(dtype=np.float64), d_use, thres)
            out_data[:, i] = out_vec
            diagnostics[col] = {'d': d_use, 'fallback': True, 'error': str(e)}
    
    output = pd.DataFrame(out_data, index=idx, columns=cols)
    return output, diagnostics


def compute_weight_window_sizes(d_values, thres: float = 1e-5) -> dict:
    """Return effective fixed-width window sizes for each FFD d value.

    The effective memory is determined by weight truncation (thres) and should
    be measured empirically from get_weights_ffd instead of assumed.
    """
    out = {}
    for d in d_values:
        d_f = float(d)
        weights = get_weights_ffd(d_f, thres)
        k = int(len(weights))
        out[d_f] = {
            "K": k,
            "warmup_bars": max(0, k - 1),
            "compute_cost": f"O(N x {k})",
        }
    return out
