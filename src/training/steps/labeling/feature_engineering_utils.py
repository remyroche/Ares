import numpy as np
import pandas as pd
import logging
from src.utils.tprint import tprint_info, tprint_warning, tprint_success
from src.utils.orthogonal_numba import _numba_apply_fracdiff

def _causal_denoise(signal: np.ndarray, span: int = 10) -> np.ndarray:
    """
    Apply causal denoising using Exponential Weighted Moving Average (EWMA).
    Replaces non-causal wavelet denoising to prevent lookahead bias.
    """
    return pd.Series(signal).ewm(span=span, adjust=False).mean().values

def _apply_fracdiff(series: pd.Series, d: float = 0.4, threshold: float = 1e-5) -> pd.Series:
    """
    Apply fractional differentiation using fixed-width window.

    Uses the approach from AFML Ch. 5:
    (1-B)^d = sum_{k=0}^{inf} C(d,k) * (-B)^k
    where C(d,k) = d*(d-1)*...*(d-k+1) / k!
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
                                   vol_window: int = 20,
                                   fracdiff_d: float = 0.4,
                                   wavelet: str = 'db4', # Deprecated, kept for signature compatibility
                                   wavelet_level: int = 2, # Deprecated
                                   enable_price_features: bool = True) -> pd.DataFrame:
    """
    Apply de Prado-compliant price processing and "Anti-Explosion" feature generation.

    Pipeline:
    1. Log-Returns (eliminates price level non-stationarity)
    2. Vol-Adjusted (GARCH-style normalization for regime invariance)
    3. FracDiff (fractional differentiation to preserve memory while ensuring stationarity)
    4. Causal Denoising (EWMA-based trend extraction)

    Anti-Explosion Features:
    - Primary set: log returns, rolling volatility, rolling momentum (10, 20, 50), skew, kurtosis, drawdown
    - Augmentations: vol-adjusted tail, denoised trend, fracdiff state

    Args:
        df: DataFrame with price data.
        price_col: Column name for price.
        vol_window: Window for volatility estimation.
        fracdiff_d: Fractional differentiation order (0.3-0.5 typical).
        wavelet: (Deprecated) Wavelet family.
        wavelet_level: (Deprecated) Decomposition level.
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
    result['log_returns'] = log_returns.fillna(0) # Fill initial NaN with 0 for downstream safety, but keep awareness

    # 2. Vol-Adjusted Returns
    # Use min_periods to avoid early unstable estimates
    # ffill() propagates the last valid volatility forward to avoid lookahead median filling
    vol = log_returns.rolling(vol_window, min_periods=vol_window).std()

    # Fill initial NaNs with an expanding mean or just forward fill the first valid
    if vol.first_valid_index() is not None:
        first_valid_vol = vol.loc[vol.first_valid_index()]
        vol = vol.fillna(first_valid_vol) # Backfill warmup period with first valid estimate (mild leakage only at very start)
    else:
        vol = vol.fillna(0.01) # Fallback if series is too short

    vol_adjusted_returns = log_returns / (vol + 1e-9)
    result['vol_adjusted_returns'] = vol_adjusted_returns.clip(-10, 10)

    # 3. Fractional Differentiation (FracDiff)
    try:
        # FracDiff on log prices (standard practice)
        fracdiff_series = _apply_fracdiff(log_price.ffill(), d=fracdiff_d)
        result['fracdiff_log_price'] = fracdiff_series
    except Exception as e:
        tprint_warning(f"   ⚠️ FracDiff failed: {e}. Skipping.")
        result['fracdiff_log_price'] = np.nan  # Explicit failure signal

    # 4. Causal Denoising (formerly Wavelet)
    try:
        # Use a robust EWMA smoother on vol-adjusted returns
        denoised = _causal_denoise(vol_adjusted_returns.fillna(0).values, span=10)
        result['causal_denoised_returns'] = pd.Series(denoised, index=df.index)
    except Exception as e:
        tprint_warning(f"   ⚠️ Causal denoising failed: {e}. Skipping.")
        result['causal_denoised_returns'] = vol_adjusted_returns

    # --- Anti-Explosion Feature Set ---

    # A. Primary Set
    # Rolling Volatility
    result['rolling_volatility_20'] = vol
    result['rolling_volatility_50'] = log_returns.rolling(50, min_periods=50).std().ffill()

    # Rolling Momentum (using sum of log returns = log return over window)
    for w in [10, 20, 50]:
        result[f'rolling_momentum_{w}'] = log_returns.rolling(w, min_periods=w).sum()

    # Skew/Kurtosis
    result['rolling_skew_50'] = log_returns.rolling(50, min_periods=50).skew()
    result['rolling_kurtosis_50'] = log_returns.rolling(50, min_periods=50).kurt()

    # Drawdown
    # Use min_periods=1 to allow calculation from start
    rolling_max = price.rolling(100, min_periods=1).max()
    # Avoid zero division
    result['drawdown_100'] = (price / (rolling_max + 1e-9)) - 1.0

    # B. Augmentations

    # From vol_adjusted_returns: Tail/exceedance (Rolling max of abs)
    result['vol_adj_tail_20'] = vol_adjusted_returns.abs().rolling(20, min_periods=20).max()

    # From denoised_*: Trend/persistence (Divergence from raw)
    result['denoised_divergence'] = result['causal_denoised_returns'] - vol_adjusted_returns

    # From fracdiff_log_price: State/slow features (Rolling mean/Z-score)
    fd = result['fracdiff_log_price']
    fd_mean = fd.rolling(50, min_periods=50).mean()
    fd_std = fd.rolling(50, min_periods=50).std()
    result['fracdiff_zscore_50'] = (fd - fd_mean) / (fd_std + 1e-9)

    return result
