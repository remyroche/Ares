import numpy as np
import pandas as pd
import logging
from numba import jit
from src.utils.tprint import tprint_info, tprint_warning, tprint_success
from src.utils.orthogonal_numba import _numba_apply_fracdiff, _numba_rolling_hurst
from src.utils.entropy_optimized import lempel_ziv_complexity_numba
from src.utils.numba_funcs import _numba_ewma, _numba_ewm_std, _numba_rolling_skew, _numba_rolling_kurt

def _calculate_rolling_vwap(price: pd.Series, volume: pd.Series, window: int = 20) -> pd.Series:
    """
    Calculate Rolling VWAP.
    VWAP = Sum(Price * Volume) / Sum(Volume)
    """
    pv = price * volume
    sum_pv = pv.rolling(window, min_periods=1).sum()
    sum_v = volume.rolling(window, min_periods=1).sum()
    return sum_pv / (sum_v + 1e-9)

def _causal_vwap_residual(price: pd.Series, volume: pd.Series, vwap_window: int = 100, vol_span: int = 50, clip_z: float = 5.0) -> pd.Series:
    """
    Universal, causal price residualisation that explicitly uses VWAP:
        vwap_t   = rolling_vwap(P, V, window=vwap_window)
        e_t      = log(P_t) - log(vwap_t)          (dimensionless deviation from VWAP)
        sigma_t  = sqrt(EWMA( (Δlog(P))^2, span=vol_span ))
        z_t      = e_t / sigma_t                   (vol-normalized residual)
    """
    # Basic input hygiene
    p = price.replace([0.0, -0.0], np.nan).astype(float)
    v = volume.clip(lower=0.0).astype(float)

    # Causal rolling VWAP
    pv = p * v
    vwap = (
        pv.rolling(window=vwap_window, min_periods=1).sum()
        / v.rolling(window=vwap_window, min_periods=1).sum().replace(0.0, np.nan)
    )

    # Log-domain residual versus VWAP
    log_p = np.log(p)
    log_vwap = np.log(vwap)
    e = log_p - log_vwap

    # Causal EWMA volatility of log returns
    log_ret = log_p.diff()

    # EWMA of squared returns
    sigma = np.sqrt((log_ret ** 2).ewm(span=vol_span, adjust=False, min_periods=2).mean())

    # Vol-normalized residual (z-score style)
    z = e / sigma
    if clip_z is not None:
        z = z.clip(-clip_z, clip_z)

    return z

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

@jit(nopython=True)
def _numba_efficiency_ratio(log_returns: np.ndarray, window: int) -> np.ndarray:
    """
    Calculate Kaufman Efficiency Ratio: Abs(Net Change) / Sum(Abs(Change))
    Optimized to O(N) using sliding window updates.
    """
    n = len(log_returns)
    out = np.full(n, np.nan)

    if n < window:
        return out

    # Needs absolute returns
    abs_rets = np.abs(log_returns)

    # Initialize sums for the first window
    current_net = 0.0
    current_vol = 0.0
    nan_count = 0

    # Sum for indices [0, window-1]
    for k in range(window):
        val = log_returns[k]
        if not np.isnan(val):
            current_net += val
            current_vol += np.abs(val)
        else:
            nan_count += 1

    # Set the first result at index window-1
    if nan_count == 0 and current_vol > 1e-12:
        out[window - 1] = np.abs(current_net) / current_vol
    else:
        out[window - 1] = 0.0

    # Sliding window loop
    # We produce outputs for i from window to n-1
    # i is the index of the entering element
    for i in range(window, n):
        leaving = log_returns[i - window]
        if np.isnan(leaving):
            nan_count -= 1
            leaving = 0.0

        entering = log_returns[i]
        if np.isnan(entering):
            nan_count += 1
            entering = 0.0

        current_net += entering - leaving
        current_vol += np.abs(entering) - np.abs(leaving)

        # Numerical stability fix: current_vol should not be negative due to float errors
        if current_vol < 0:
            current_vol = 0.0

        if nan_count == 0 and current_vol > 1e-12:
            out[i] = np.abs(current_net) / current_vol
        else:
            out[i] = 0.0

    return out

def apply_layer2_price_processing(df: pd.DataFrame,
                                  price_col: str = 'close',
                                  volume_col: str = None,
                                  vol_window: int = 20, # Deprecated
                                  fracdiff_d: float = 0.4,
                                  wavelet: str = 'db4', # Deprecated
                                  wavelet_level: int = 2, # Deprecated
                                  enable_price_features: bool = True,
                                  vwap_window: int = 5) -> pd.DataFrame:
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

    asset_col = None
    if 'asset_id' in df.columns:
        asset_col = 'asset_id'
    elif 'ticker' in df.columns:
        asset_col = 'ticker'
    elif isinstance(df.index, pd.MultiIndex) and 'ticker' in df.index.names:
        asset_col = 'ticker'

    def _apply_single_asset(asset_df: pd.DataFrame) -> pd.DataFrame:
        result = asset_df.copy()
        price_raw = asset_df[price_col]

        # Determine Effective Price (VWAP vs Raw)
        # Shift price returns-based features to VWAP if volume is available
        use_vwap = False
        if volume_col and volume_col in asset_df.columns:
            volume = asset_df[volume_col]
            # Calculate Base VWAP (Rolling)
            # This shifts Trend, Momentum, Volatility to VWAP-based
            effective_price = _calculate_rolling_vwap(price_raw, volume, window=vwap_window)
            use_vwap = True
        else:
            effective_price = price_raw

        # 1. Log-Returns
        # Use 1e-9 to prevent log(0)
        log_price = np.log(effective_price.replace(0, np.nan).ffill())
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

        vol = pd.Series(vol_vals, index=asset_df.index)

        vol_adjusted_returns = log_returns / (vol + 1e-9)
        result['vol_adjusted_returns'] = vol_adjusted_returns.clip(-10, 10)

        # 3. Fractional Differentiation (FracDiff)
        try:
            if use_vwap:
                # "If using FracDiff, VWAP is used after FracDiff"
                # Logic: FracDiff(Price) -> VWAP(FracDiff)
                # Note: Usually FracDiff is on log prices
                log_price_raw = np.log(price_raw.replace(0, np.nan).ffill())
                fd_raw = _apply_fracdiff(log_price_raw.ffill(), d=fracdiff_d)

                # Apply VWAP AFTER FracDiff
                # Calculate VWAP of the stationary series
                fracdiff_series = _calculate_rolling_vwap(fd_raw, volume, window=vwap_window)
            else:
                fracdiff_series = _apply_fracdiff(log_price.ffill(), d=fracdiff_d)

            result['fracdiff_log_price'] = fracdiff_series
        except Exception as e:
            tprint_warning(f"   ⚠️ FracDiff failed: {e}. Skipping.")
            result['fracdiff_log_price'] = np.nan

        # 4. Causal Denoising
        try:
            # Robust EWMA smoother on vol-adjusted returns with Half-Life = 4 bars
            denoised = _causal_denoise(vol_adjusted_returns.fillna(0).values, halflife=4.0)
            result['causal_denoised_returns'] = pd.Series(denoised, index=asset_df.index)
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
        result['rolling_volatility_50'] = pd.Series(vol_50_vals, index=asset_df.index).ffill()

        # Rolling Momentum (using sum of log returns)
        # Optimized: sum(log_returns) over window w is equivalent to log_price.diff(w).
        # This vectorizes the operation (O(1) overhead vs O(W) rolling).
        for w in [10, 20, 50]:
            result[f'rolling_momentum_{w}'] = log_price.diff(w)

        # Skew/Kurtosis (Optimized with Numba)
        # Use clean log_returns (0-filled) to prevent NaN propagation in online algorithm
        clean_log_ret = result['log_returns'].fillna(0).values.astype(np.float64)
        skew_vals = _numba_rolling_skew(clean_log_ret, 50)
        kurt_vals = _numba_rolling_kurt(clean_log_ret, 50)

        result['rolling_skew_50'] = pd.Series(skew_vals, index=asset_df.index)
        result['rolling_kurtosis_50'] = pd.Series(kurt_vals, index=asset_df.index)

        # Drawdown (Excluding VWAP: Always use raw price)
        rolling_max = price_raw.rolling(100, min_periods=1).max()
        result['drawdown_100'] = (price_raw / (rolling_max + 1e-9)) - 1.0

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

        # --- New Features (Audit Request) ---

        # 1. Hurst Exponent (Proxy or Rolling)
        # Using Numba optimized rolling Hurst on log prices
        try:
            hurst_100 = _numba_rolling_hurst(log_price.ffill().values, window=100)
            result['hurst_100'] = pd.Series(hurst_100, index=asset_df.index).ffill()
        except Exception as e:
            tprint_warning(f"   ⚠️ Hurst calculation failed: {e}")

        # 2. LZ Complexity
        try:
            # On log returns (discretized implicitly by algo) or prices?
            # LZ on raw prices captures structure.
            # Normalize=True divides by n/log(n), making it comparable.
            lz_vals = lempel_ziv_complexity_numba(log_price.ffill().values, normalize=True)
            result['lz_complexity'] = pd.Series(lz_vals, index=asset_df.index).ffill()
        except Exception as e:
            tprint_warning(f"   ⚠️ LZ Complexity failed: {e}")

        # 3. Efficiency Ratio
        try:
            er_50 = _numba_efficiency_ratio(log_returns.fillna(0).values, window=50)
            result['efficiency_ratio_50'] = pd.Series(er_50, index=asset_df.index).ffill()
        except Exception as e:
            tprint_warning(f"   ⚠️ Efficiency Ratio failed: {e}")

        # 4. Bar Tightness
        # (High - Low) / (High + Low) or similar.
        # We check if High/Low exist
        cols_map = {c.lower(): c for c in asset_df.columns}
        if 'high' in cols_map and 'low' in cols_map:
            h = asset_df[cols_map['high']]
            l = asset_df[cols_map['low']]

            # Normalized Range: (H - L) / (H + L)
            # Or relative to close: (H - L) / Close
            # Using (H - L) / (H + L) is scale invariant
            tightness = (h - l) / (h + l + 1e-9)

            # Invert so higher = tighter?
            # User said "Bar Tightness".
            # Usually "Tightness" means small range.
            # So maybe 1 - normalized_range?
            # Or just the metric itself and let model decide.
            # "Tightness" often refers to "Spread Tightness" (Bid-Ask).
            # But we don't have bid/ask.
            # I'll compute Range Ratio and let tree decide.
            # Actually, let's call it 'bar_tightness' = 1 / (range_pct + epsilon) to match "tightness" (high = tight)
            range_pct = (h - l) / (price_raw + 1e-9) # Keep raw price for range ratio
            result['bar_tightness'] = 1.0 / (range_pct + 1e-4) # Cap at 10000

        # VWAP Residualisation
        # "if using price residualisation against itself, it is applied before the residualisation"
        # Logic: VWAP -> Residualisation (Detrending)
        # We perform Causal VWAP Residualisation with Volatility Normalization
        try:
            # We use effective_price (which might be VWAP(5)) as the price input 'p'
            # But strictly, the snippet expects raw 'P' and 'V' to calculate its own VWAP(100).
            # If we pass VWAP(5) as P, we are doing VWAP(100) of VWAP(5), which is fine/robust.
            # Using volume from input if available.
            if volume_col and volume_col in asset_df.columns:
                volume = asset_df[volume_col]
                result['vwap_residual'] = _causal_vwap_residual(
                    effective_price,
                    volume,
                    vwap_window=100,
                    vol_span=50,
                    clip_z=5.0
                )
            else:
                # Fallback if no volume: simple EMA detrending (normalized?)
                # Or just skip. Since this is "vwap_residual", skipping is safer.
                # But to maintain previous behavior (fallback exists):
                pass
        except Exception as e:
            # Fallback or silent fail
            pass

        return result

    if asset_col is None:
        return _apply_single_asset(df)

    tprint_info(f"   🔧 Per-asset price processing enabled (col={asset_col})")
    processed_chunks = []
    if asset_col in df.columns:
        for _asset, asset_df in df.groupby(asset_col, sort=False):
            processed_chunks.append(_apply_single_asset(asset_df))
    else:
        for _asset, asset_df in df.groupby(level=asset_col, sort=False):
            processed_chunks.append(_apply_single_asset(asset_df))

    combined = pd.concat(processed_chunks).sort_index()
    return combined
