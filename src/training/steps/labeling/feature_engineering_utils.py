import numpy as np
import pandas as pd
import logging
from src.utils.tprint import tprint_info, tprint_warning, tprint_success
from src.utils.orthogonal_numba import _numba_apply_fracdiff
from src.utils.numba_funcs import _numba_ewma, _numba_ewm_std, _numba_rolling_sum, _numba_rolling_skew, _numba_rolling_kurt, _numba_rolling_mean, _numba_rolling_std

def _causal_denoise(signal: np.ndarray, halflife: float = 4.0) -> np.ndarray:
    """
    Apply causal denoising using Exponential Weighted Moving Average (EWMA).
    Replaces non-causal wavelet denoising to prevent lookahead bias.
    Uses Numba-optimized implementation.

    Args:
        signal: Input signal array.
        halflife: Half-life for EWMA decay (in bars).
    """
    if len(signal) == 0:
        return signal

    # Convert halflife to alpha
    # alpha = 1 - exp(log(0.5)/halflife)
    if halflife <= 0:
        return signal

    alpha = 1.0 - np.exp(np.log(0.5) / halflife)
    return _numba_ewma(signal.astype(np.float64), alpha=alpha, adjust=False)

def _apply_fracdiff(series: pd.Series, d: float = 0.4, threshold: float = 1e-5) -> pd.Series:
    """
    Apply fractional differentiation using fixed-width window.
    """
    # Calculate weights
    def _get_weights(d: float, size: int, threshold: float) -> np.ndarray:
        w = [1.0]
        for k in range(1, size):
            w_k = -w[-1] * (d - k + 1) / k
            if abs(w_k) < threshold:
                break
            w.append(w_k)
        return np.array(w)

    # Get weights
    w = _get_weights(d, len(series), threshold)

    # Apply convolution (Numba)
    result = _numba_apply_fracdiff(series.values, w)

    return pd.Series(result, index=series.index)

def apply_layer2_price_processing(df: pd.DataFrame,
                                   price_col: str = 'close',
                                   vol_window: int = 20, # Deprecated
                                   fracdiff_d: float = 0.4,
                                   wavelet: str = 'db4', # Deprecated
                                   wavelet_level: int = 2, # Deprecated
                                   enable_price_features: bool = True) -> pd.DataFrame:
    """
    Apply de Prado-compliant price processing and "Anti-Explosion" feature generation.
    Optimized with Numba for EWMA and Volatility calculations.

    Args:
        df: DataFrame with price data.
        price_col: Column name for price.
        vol_window: (Deprecated usage) Window for volatility estimation.
        fracdiff_d: Fractional differentiation order (0.3-0.5 typical).
        enable_price_features: Flag to enable/disable processing.

    Returns:
        DataFrame with processed price features added.
    """
    if not enable_price_features:
        return df

    if price_col not in df.columns:
        return df

    result = df.copy()
    price = df[price_col]

    # 1. Log-Returns
    # Use 1e-9 to prevent log(0)
    log_price = np.log(price.replace(0, np.nan).ffill())
    # Leave NaNs where they naturally occur (start of series)
    log_returns = log_price.diff()
    result['log_returns'] = log_returns.fillna(0) # Fill initial NaN with 0 for downstream safety

    # Prepare data for Numba
    log_ret_vals = log_returns.values.astype(np.float64)

    # 2. Vol-Adjusted Returns
    # Using strictly causal EWMA volatility with Half-Life = 16 bars
    # alpha = 1 - exp(log(0.5)/16) approx 0.042
    alpha_vol = 1.0 - np.exp(np.log(0.5) / 16.0)

    # Calculate EWMA Std using Numba
    vol_vals = _numba_ewm_std(log_ret_vals, alpha=alpha_vol, adjust=False)

    # Handle initial NaNs/Warmup (min_periods behavior simulation)
    # Pandas min_periods=16 means first 15 are NaN.
    # _numba_ewm_std returns NaNs for first point, then valid.
    # We should enforce NaN for first 15 points to match "min_periods" safety if desired,
    # or just accept early noisy estimates.
    # To match previous logic (min_periods=16):
    vol_vals[:16] = np.nan

    # Backfill warmup
    # Find first valid
    mask = ~np.isnan(vol_vals)
    if mask.any():
        first_valid_idx = np.argmax(mask)
        vol_vals[:first_valid_idx] = vol_vals[first_valid_idx]
    else:
        vol_vals[:] = 0.01 # Fallback

    vol = pd.Series(vol_vals, index=df.index)

    vol_adjusted_returns = log_returns / (vol + 1e-9)
    result['vol_adjusted_returns'] = vol_adjusted_returns.clip(-10, 10)

    # 3. Fractional Differentiation (FracDiff)
    try:
        fracdiff_series = _apply_fracdiff(log_price.ffill(), d=fracdiff_d)
        result['fracdiff_log_price'] = fracdiff_series
    except Exception as e:
        tprint_warning(f"   ⚠️ FracDiff failed: {e}. Skipping.")
        result['fracdiff_log_price'] = np.nan

    # 4. Causal Denoising
    try:
        # Robust EWMA smoother on vol-adjusted returns with Half-Life = 4 bars
        denoised = _causal_denoise(vol_adjusted_returns.fillna(0).values, halflife=4.0)
        result['causal_denoised_returns'] = pd.Series(denoised, index=df.index)
    except Exception as e:
        tprint_warning(f"   ⚠️ Causal denoising failed: {e}. Skipping.")
        result['causal_denoised_returns'] = vol_adjusted_returns

    # --- Anti-Explosion Feature Set ---

    # A. Primary Set
    # Rolling Volatility
    result['rolling_volatility_20'] = vol

    # rolling_volatility_50 with HL=40
    alpha_vol_50 = 1.0 - np.exp(np.log(0.5) / 40.0)
    vol_50_vals = _numba_ewm_std(log_ret_vals, alpha=alpha_vol_50, adjust=False)
    vol_50_vals[:40] = np.nan # Simulate min_periods

    # Forward fill NaNs for this one (as per previous logic .ffill())
    # Use pandas ffill for simplicity or numpy logic
    result['rolling_volatility_50'] = pd.Series(vol_50_vals, index=df.index).ffill()

    # Rolling Momentum (using sum of log returns)
    for w in [10, 20, 50]:
        # Using Numba Rolling Sum
        mom_vals = _numba_rolling_sum(log_ret_vals, w)
        mom_vals[:w-1] = np.nan
        result[f'rolling_momentum_{w}'] = pd.Series(mom_vals, index=df.index)

    # Skew/Kurtosis (Optimized with Numba)
    skew_vals = _numba_rolling_skew(log_ret_vals, 50)
    result['rolling_skew_50'] = pd.Series(skew_vals, index=df.index)

    kurt_vals = _numba_rolling_kurt(log_ret_vals, 50)
    result['rolling_kurtosis_50'] = pd.Series(kurt_vals, index=df.index)

    # Drawdown
    rolling_max = price.rolling(100, min_periods=1).max()
    result['drawdown_100'] = (price / (rolling_max + 1e-9)) - 1.0

    # B. Augmentations

    # From vol_adjusted_returns: Tail/exceedance
    result['vol_adj_tail_20'] = vol_adjusted_returns.abs().rolling(20, min_periods=20).max()

    # From denoised_*: Trend/persistence (Divergence from raw)
    result['denoised_divergence'] = result['causal_denoised_returns'] - vol_adjusted_returns

    # From fracdiff_log_price: State/slow features
    fd = result['fracdiff_log_price']

    fd_mean = fd.rolling(50, min_periods=50).mean()
    fd_std = fd.rolling(50, min_periods=50).std()
    result['fracdiff_zscore_50'] = (fd - fd_mean) / (fd_std + 1e-9)

    return result
