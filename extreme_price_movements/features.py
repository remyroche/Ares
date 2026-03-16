import warnings

import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import os
import pickle
import re
from joblib import Memory
from extreme_price_movements.intraday_crypto_library import (
    INTRADAY_TRIGGER_COLUMNS,
    LOCATION_FILTER_COLUMNS,
    build_intraday_crypto_library,
)
from extreme_price_movements.utils import tprint, check_inf_nan
from extreme_price_movements.feature_transforms import CausalFeatureTransformer
from extreme_price_movements.time_utils import ensure_utc
from extreme_price_movements.frac_diff_adaptive import find_min_ffd, frac_diff_ffd, get_weights_ffd
from extreme_price_movements.validation import validate_panel
from extreme_price_movements.gated_features import add_accept_gate_features, add_gate_features, add_gate_interaction_panel
from extreme_price_movements.perp_features import compute_features as compute_perp_features
import extreme_price_movements.fast_funcs as ff

# Suppress expected RuntimeWarnings from nanmin/nanmean/nanmax on all-NaN slices
# These are handled gracefully by replacing with 0 later in the pipeline
warnings.filterwarnings('ignore', message='.*All-NaN slice.*')
warnings.filterwarnings('ignore', message='.*Mean of empty slice.*')
# Suppress divide warnings from correlation calculations when stddev is 0
warnings.filterwarnings('ignore', message='.*invalid value encountered.*')

# Initialize joblib cache
_cache = Memory("./cache/features", verbose=0)

# --- Per-column FFD incremental cache ---
_FFD_COL_CACHE_DIR = "./cache/ffd_columns"
EPS = 1e-12
_PERP_FEATURE_COLLISION_RENAMES = {
    "ret1h": "ret1h_perp",
}

_INTRADAY_PERSISTED_KEY_SET = set(LOCATION_FILTER_COLUMNS) | set(INTRADAY_TRIGGER_COLUMNS)


def _compute_intraday_library_features_wide(
    open_df: pd.DataFrame,
    high_df: pd.DataFrame,
    low_df: pd.DataFrame,
    close_df: pd.DataFrame,
    volume_df: pd.DataFrame,
    requested_feature_set: set[str],
) -> dict[str, pd.DataFrame]:
    selected_keys = (
        sorted(_INTRADAY_PERSISTED_KEY_SET)
        if not requested_feature_set
        else sorted(_INTRADAY_PERSISTED_KEY_SET.intersection(requested_feature_set))
    )
    if not selected_keys:
        return {}

    session_ids = pd.Series(
        pd.factorize(open_df.index.normalize())[0].astype(np.int32),
        index=open_df.index,
        dtype="int32",
    )
    out: dict[str, pd.DataFrame] = {
        key: pd.DataFrame(index=open_df.index, columns=open_df.columns, dtype=np.float32)
        for key in selected_keys
    }
    total_symbols = len(open_df.columns)
    tprint(
        f"Features: computing intraday location/trigger library "
        f"({len(selected_keys)} keys x {total_symbols} symbols)"
    )
    for i, sym in enumerate(open_df.columns, start=1):
        local_df = pd.DataFrame(
            {
                "open": open_df[sym].astype(np.float32, copy=False),
                "high": high_df[sym].astype(np.float32, copy=False),
                "low": low_df[sym].astype(np.float32, copy=False),
                "close": close_df[sym].astype(np.float32, copy=False),
                "volume": volume_df[sym].astype(np.float32, copy=False),
                "session_id": session_ids,
            },
            index=open_df.index,
        )
        local_lib = build_intraday_crypto_library(local_df)
        for key in selected_keys:
            col = local_lib.get(key)
            if col is None:
                continue
            out[key][sym] = np.asarray(col, dtype=np.float32)
        if i % 25 == 0 or i == total_symbols:
            tprint(f"  Intraday library progress: {i}/{total_symbols}")
    return out

def _sanitize_col_name(name):
    """Make column name filesystem-safe."""
    return re.sub(r'[^\w\-.]', '_', str(name))

def _col_data_hash(arr):
    """Fast hash of column data for cache key."""
    return hashlib.md5(np.ascontiguousarray(arr).tobytes()).hexdigest()[:16]

def _rolling_winsorize_causal(x: pd.DataFrame, window: int, q_lo: float, q_hi: float) -> pd.DataFrame:
    """Winsorize using causal rolling quantile bands (shifted by 1)."""
    if window <= 1:
        return x.astype(np.float32)
    lo = ff.numba_rolling_quantile(x, window, float(q_lo)).shift(1)
    hi = ff.numba_rolling_quantile(x, window, float(q_hi)).shift(1)
    return x.clip(lower=lo, upper=hi).astype(np.float32)


def zscore_rolling(
    x: pd.DataFrame,
    n: int,
    *,
    winsorize: bool = True,
    q_lo: float = 0.01,
    q_hi: float = 0.99,
    std_floor: float = 1e-6,
    use_ewma: bool = False,
    ewma_span: int | None = None,
):
    """
    Guarded rolling z-score.
    - Optional causal winsorization before mean/std
    - Std floor guard (flatline windows -> 0)
    - Optional EWMA mean/std mode for faster adaptation
    """
    x_in = x.astype(np.float32)
    x_proc = _rolling_winsorize_causal(x_in, max(2, int(n)), q_lo, q_hi) if winsorize else x_in

    if use_ewma:
        span = max(2, int(ewma_span or n))
        alpha = 2.0 / (span + 1.0)
        mu = ff.numba_ewma(x_proc, alpha, False).shift(1)
        dev2 = (x_proc - mu) ** 2
        var = ff.numba_ewma(dev2, alpha, False).shift(1)
        sd = np.sqrt(var.clip(lower=0))
    else:
        mu = ff.numba_rolling_mean(x_proc, max(2, int(n)))
        sd = ff.numba_rolling_std(x_proc, max(2, int(n)))

    z = (x_proc - mu) / (sd + 1e-12)
    z = z.where(sd >= float(std_floor), 0.0)
    return z.astype(np.float32)


def robust_zscore_rolling(
    x: pd.DataFrame,
    n: int,
    *,
    quantile: float = 0.50,
    eps: float = 1e-6,
):
    """Robust rolling z-score: anchor=Q(quantile), scale=MAD."""
    return ff.numba_rolling_robust_zscore(x.astype(np.float32), int(n), float(quantile), float(eps)).astype(np.float32)

def rsi(close: pd.DataFrame, n: int):
    return ff.numba_rsi(close, n)

def ema(x: pd.DataFrame, span: int):
    alpha = 2.0 / (span + 1.0)
    return ff.numba_ewma(x, alpha, False)


def _safe_log_df(df: pd.DataFrame, eps: float = 1e-9) -> pd.DataFrame:
    """Causal-safe log transform for strictly positive inputs."""
    return np.log(np.maximum(df, eps)).astype(np.float32)


def _transform_close_fixed_ffd(
    df: pd.DataFrame,
    d: float = 0.4,
    _label: str = "close",
    already_logged: bool = False,
    thres: float = 1e-5,
) -> pd.DataFrame:
    """Transform close only with fixed d to avoid adaptive ADF leakage."""
    tprint(f"Transforming Close ({_label}): Log -> EWMA(5) -> FFD(d={d:.2f}) [{df.shape[1]} cols]")
    df_log = df.astype(np.float32) if already_logged else _safe_log_df(df)
    df_den = ff.numba_ewma(df_log, 2.0 / 6.0, False)

    out = pd.DataFrame(index=df.index, columns=df.columns, dtype=np.float32)
    fallback_d_values = [float(x) for x in (0.6, 0.5, 0.4)]
    d_candidates = []
    for cand in [float(d)] + fallback_d_values:
        if cand not in d_candidates:
            d_candidates.append(cand)
    win_by_d = {cand: int(len(get_weights_ffd(cand, float(thres)))) for cand in d_candidates}
    fallback_used = 0
    direct_used = 0
    total_cols = len(df_den.columns)
    for i, col in enumerate(df_den.columns):
        ser = df_den[col]
        valid_n = int(ser.notna().sum())

        d_use = None
        for cand in d_candidates:
            if win_by_d[cand] <= valid_n:
                d_use = cand
                break

        if d_use is None:
            # Series is too short for all configured FFD windows: keep denoised log-close
            # rather than generating all-NaN features.
            out[col] = ser.astype(np.float32)
            direct_used += 1
        else:
            if d_use != float(d):
                fallback_used += 1
            out[col] = frac_diff_ffd(ser, d=float(d_use), thres=float(thres)).astype(np.float32)

        if (i + 1) % 10 == 0 or (i + 1) == total_cols:
            tprint(f"Fixed FFD ({_label}, d={d:.2f}): {i+1}/{total_cols}")
    if fallback_used > 0 or direct_used > 0:
        tprint(
            f"Fixed FFD ({_label}, d={d:.2f}) short-history fallback: "
            f"d-fallback={fallback_used}, direct-denoised={direct_used}, total={total_cols}"
        )
    return out

def atr_percent(high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame, n: int):
    return ff.numba_atr_no_norm(high, low, close, n)

def rolling_mad(df: pd.DataFrame, window: int):
    """
    Calculate rolling MAD (Median Absolute Deviation) using Numba.
    Uses standard definition: Median(|x - Median(Window)|).
    """
    return ff.numba_rolling_mad(df, window).astype(np.float32)


def _rolling_shannon_entropy_df(df: pd.DataFrame, window: int, bins: int = 16) -> pd.DataFrame:
    """Fast Shannon entropy proxy using quantile-based spread measure.
    
    Improved proxy that better approximates true entropy by measuring:
    1. Inter-quartile range (IQR) spread - captures distribution width
    2. Quantile dispersion - captures how spread out the distribution is
    
    This is more accurate than CV alone because it:
    - Is scale-invariant (unlike CV which fails for zero-mean data)
    - Captures tail behavior via IQR
    - Correlates better with true histogram-based entropy
    """
    # Optimized Numba implementation (fused kernel)
    # Replaces 7 separate rolling passes with 1 pass.
    # Note: bins arg is unused in this proxy method
    
    # Compute aligned rolling entropy
    entropy_proxy = ff.numba_rolling_entropy_proxy(df, window)
    
    # Shift(1) to match original predictive behavior (value at t uses info up to t-1)
    return entropy_proxy.shift(1).fillna(0.5).astype(np.float32)


def _rolling_permutation_entropy_df(df: pd.DataFrame, window: int, order: int = 3, delay: int = 1) -> pd.DataFrame:
    """Fast permutation entropy proxy using rank correlation structure.
    
    Improved proxy that captures ordinal pattern information by measuring:
    1. Autocorrelation at multiple lags - captures trend vs mean-reversion
    2. Run length statistics - captures clustering behavior
    
    This is more accurate than sign-change alone because it:
    - Distinguishes between trending and mean-reverting regimes
    - Captures the ordinal structure of patterns
    """
    # Multi-lag autocorrelation structure
    # High PE = low autocorrelation = random = high entropy
    # Low PE = high autocorrelation = predictable = low entropy
    
    # Use returns for autocorrelation calculation
    rets = df.diff(delay)
    
    # Autocorrelation at lag 1 (primary)
    roll = rets.rolling(window, min_periods=max(4, window // 2))
    mean = roll.mean().shift(1)
    var = roll.var(ddof=0).shift(1).clip(lower=1e-12)
    
    # Covariance with lagged self
    rets_lag = rets.shift(delay)
    cov = (rets * rets_lag).rolling(window, min_periods=max(4, window // 2)).mean().shift(1) - mean * mean.shift(delay)
    
    # Autocorrelation
    autocorr = (cov / var).clip(-1, 1)
    
    # Also measure run length (consecutive same-sign periods)
    sign = (rets > 0).astype(np.float32)
    sign_change = (sign != sign.shift(delay)).astype(np.float32)
    run_freq = sign_change.rolling(window, min_periods=max(4, window // 2)).mean().shift(1)
    
    # Combine: 
    # - High run_freq = frequent sign changes = mean-reverting = medium entropy
    # - Low run_freq = trending = low entropy
    # - autocorr near 0 = random = high entropy
    # - autocorr near ±1 = predictable = low entropy
    
    # Map autocorr to entropy: |autocorr| = 0 -> 1.0, |autocorr| = 1 -> 0.0
    autocorr_entropy = 1.0 - autocorr.abs()
    
    # Run frequency: 0.5 = random (max entropy), 0 or 1 = trending (low entropy)
    run_entropy = 1.0 - 2.0 * (run_freq - 0.5).abs()
    
    # Weighted combination
    entropy_proxy = 0.6 * autocorr_entropy + 0.4 * run_entropy
    
    return entropy_proxy.clip(0, 1).fillna(0.5).astype(np.float32)


def _rolling_spectral_entropy_df(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Fast spectral entropy proxy using multi-scale variance decomposition.
    
    Improved proxy that better approximates spectral flatness by measuring:
    1. Variance ratio across multiple time scales (not just short/long)
    2. Hurst exponent proxy - captures long-range dependence
    
    This is more accurate than single variance ratio because it:
    - Captures power law decay in spectrum
    - Distinguishes between 1/f noise, white noise, and trending
    """
    # Multi-scale variance decomposition
    # White noise: variance scales linearly with window
    # Trend: variance scales super-linearly
    # Mean-reversion: variance scales sub-linearly
    
    # Compute variance at multiple scales
    scales = [max(2, window // 8), max(4, window // 4), max(8, window // 2), window]
    variances = []
    for s in scales:
        v = df.rolling(s, min_periods=max(2, s // 2)).var(ddof=0).shift(1)
        variances.append(v)
    
    # Variance ratio matrix: how variance scales with window
    # For white noise, var(s) / var(s/2) ≈ 2
    # For trend, var(s) / var(s/2) > 2
    # For MR, var(s) / var(s/2) < 2
    
    ratios = []
    for i in range(1, len(variances)):
        r = (variances[i] / (variances[i-1] + 1e-12)).clip(0.1, 10)
        ratios.append(r)
    
    # Stack and compute flatness
    # Flat spectrum = all ratios near expected value (white noise behavior)
    # Concentrated spectrum = ratios deviate from expected
    
    # Expected ratio for white noise: scale_factor = scales[i] / scales[i-1]
    expected_ratios = [scales[i] / scales[i-1] for i in range(1, len(scales))]
    
    # Measure deviation from white noise behavior
    deviations = []
    for i, r in enumerate(ratios):
        dev = ((r - expected_ratios[i]) / expected_ratios[i]).abs()
        deviations.append(dev)
    
    # Average deviation: 0 = white noise = high entropy, high = structured = low entropy
    avg_deviation = sum(deviations) / len(deviations)
    
    # Map to entropy: low deviation = high entropy
    entropy_proxy = (1.0 / (1.0 + avg_deviation)).clip(0, 1)
    
    return entropy_proxy.fillna(0.5).astype(np.float32)


def _transform_price(df, _label=""):
    """Transform raw prices: Log -> EWMA(5) -> Adaptive FracDiff.

    Two-level per-column incremental caching:
      L1: Raw column data unchanged  -> load cached FFD result  (0 cost)
      L2: Data changed, d_opt cached  -> skip find_min_ffd      (~80% faster)
    """
    tprint(f"Transforming Prices ({_label}): Log -> EWMA(5) -> Adaptive FracDiff [{df.shape[1]} cols]")
    # Safe Log: Clip input to be at least 1e-9 to avoid log(0) or log(neg)
    df_log = np.log(np.maximum(df, 1e-9))
    df_den = ff.numba_ewma(df_log, 2.0/6.0, False)

    # Per-column incremental FFD cache
    cache_dir = os.path.join(_FFD_COL_CACHE_DIR, _sanitize_col_name(_label or "default"))
    os.makedirs(cache_dir, exist_ok=True)

    df_fd = pd.DataFrame(index=df.index, columns=df.columns, dtype=np.float32)
    total_cols = len(df_den.columns)
    stats = {"cached": 0, "cached_d": 0, "computed": 0}

    for i, col in enumerate(df_den.columns):
        safe_col = _sanitize_col_name(col)
        # Hash RAW input — deterministic key for the full pipeline
        col_raw = df[col].to_numpy(dtype=np.float64)
        data_hash = _col_data_hash(col_raw)

        col_dir = os.path.join(cache_dir, safe_col)
        os.makedirs(col_dir, exist_ok=True)
        result_path = os.path.join(col_dir, f"ffd_{data_hash}.npy")
        d_opt_path = os.path.join(col_dir, "d_opt.pkl")

        # --- Level 1: exact raw-data match -> instant load ---
        if os.path.exists(result_path):
            try:
                cached_vals = np.load(result_path, allow_pickle=False)
                if len(cached_vals) == len(df_fd):
                    df_fd[col] = cached_vals
                    stats["cached"] += 1
                    continue
            except Exception:
                pass

        # --- Level 2: reuse cached d_opt (skip expensive ADF search) ---
        d_opt = None
        if os.path.exists(d_opt_path):
            try:
                with open(d_opt_path, 'rb') as f:
                    d_info = pickle.load(f)
                d_opt = d_info.get('d_opt')
                if d_opt is not None:
                    stats["cached_d"] += 1
            except Exception:
                d_opt = None

        # --- Full compute: find optimal d ---
        if d_opt is None:
            series = df_den[col].dropna()
            if len(series) < 100:
                d_opt = 0.4
            else:
                d_opt, _, _ = find_min_ffd(series, d_range=(0.0, 1.0), step=0.1)
            stats["computed"] += 1

        # Apply FFD with (cached or computed) d_opt
        result = frac_diff_ffd(df_den[col], d_opt, thres=1e-5)
        df_fd[col] = result

        # Persist caches
        try:
            # Clean stale result files for this column
            for fname in os.listdir(col_dir):
                if fname.startswith("ffd_") and fname.endswith(".npy") and fname != os.path.basename(result_path):
                    os.remove(os.path.join(col_dir, fname))
            np.save(result_path, result.values.astype(np.float32))
            with open(d_opt_path, 'wb') as f:
                pickle.dump({'d_opt': d_opt, 'n_rows': len(df)}, f)
        except Exception as e:
            tprint(f"Warning: FFD cache write failed for {col}: {e}")

        if (i + 1) % 5 == 0 or (i + 1) == total_cols:
            tprint(f"Adaptive FFD ({_label}): {i+1}/{total_cols} - {col}")

    tprint(f"Adaptive FFD ({_label}): cache_hit={stats['cached']}, "
           f"reused_d={stats['cached_d']}, full_compute={stats['computed']} "
           f"(total {total_cols})")
    tprint(f"Adaptive FFD ({_label}): d range [{df_fd.min().min():.3f}, {df_fd.max().max():.3f}]")
    return df_fd

@_cache.cache
def _transform_volume(df):
    tprint("Transforming Volume: Log -> EWMA(5)")
    df_log = np.log(df + 1.0)
    df_den = ff.numba_ewma(df_log, 2.0/6.0, False)
    return df_den

def time_sin_cos(index: pd.DatetimeIndex):
    hod = index.hour.to_numpy()
    dow = index.dayofweek.to_numpy()
    sin_hod = np.sin(2*np.pi*hod/24.0)
    cos_hod = np.cos(2*np.pi*hod/24.0)
    sin_dow = np.sin(2*np.pi*dow/7.0)
    cos_dow = np.cos(2*np.pi*dow/7.0)
    return sin_hod, cos_hod, sin_dow, cos_dow

def compute_market_features(panel, basket_syms, trend_sma_hours=24*14):
    tprint(f"Entering function: compute_market_features in features.py")
    c = panel["close"]
    h = panel["high"]
    l = panel["low"]
    v = panel["volume"]

    basket = [s for s in basket_syms if s in c.columns]
    if not basket:
        basket = list(c.columns)

    mkt_close_raw = c[basket].mean(axis=1)
    mkt_high_raw  = h[basket].mean(axis=1)
    mkt_low_raw   = l[basket].mean(axis=1)
    mkt_vol_raw   = v[basket].mean(axis=1)

    mkt_close = ff.numba_ewma(_safe_log_df(mkt_close_raw.to_frame(name="c")), 2.0 / 6.0, False)["c"]
    mkt_high  = ff.numba_ewma(_safe_log_df(mkt_high_raw.to_frame(name="h")), 2.0 / 6.0, False)["h"]
    mkt_low   = ff.numba_ewma(_safe_log_df(mkt_low_raw.to_frame(name="l")), 2.0 / 6.0, False)["l"]
    mkt_vol   = _transform_volume(mkt_vol_raw.to_frame(name="v"))["v"]

    mkt_ret24h_df = ff.numba_rolling_sum(mkt_close.to_frame(), 24)
    mkt_ret24h = mkt_ret24h_df[mkt_ret24h_df.columns[0]]

    mkt_ret6h_df  = ff.numba_rolling_sum(mkt_close.to_frame(), 6)
    mkt_ret6h = mkt_ret6h_df[mkt_ret6h_df.columns[0]]

    sma_df = ff.numba_rolling_mean(mkt_close.to_frame(), trend_sma_hours)
    sma = sma_df[sma_df.columns[0]]

    mkt_trend = (mkt_close - sma)
    mkt_ret1h = mkt_close

    mkt_rv_df = ff.numba_rolling_std(mkt_ret1h.to_frame(), 24)
    mkt_rv = mkt_rv_df[mkt_rv_df.columns[0]]

    mkt_df = pd.DataFrame({
        "mkt_close": mkt_close,
        "mkt_high":  mkt_high,
        "mkt_low":   mkt_low,
        "mkt_volume": mkt_vol,
        "mkt_ret24h": mkt_ret24h,
        "mkt_ret6h":  mkt_ret6h,
        "mkt_trend":  mkt_trend,
        "mkt_rv":     mkt_rv
    })
    return mkt_df.astype(np.float32)

def add_regime_gates(mkt_df: pd.DataFrame, gate_vol_lookback_hours: int, gate_trend_thr: float):
    tprint(f"Entering function: add_regime_gates in features.py")
    df = mkt_df.copy()
    rv_med_df = ff.numba_rolling_median(df[["mkt_rv"]], gate_vol_lookback_hours)
    df["mkt_rv_med"] = rv_med_df["mkt_rv"]

    df["G_VOL"] = (df["mkt_rv"] > df["mkt_rv_med"]).astype(np.int32)
    
    # Dynamic Trend Threshold (Vol-Adjusted) to ensure variation
    # Fixed 0.02 is too high for low-vol regimes.
    # Use 1.5 * Daily Volatility (approx 1.5 sigma move)
    daily_vol = df["mkt_rv"] * np.sqrt(24)
    # Use dynamic threshold but floor it at small value to avoid noise in 0 vol
    dyn_thr = np.maximum(daily_vol * 1.5, 0.005) 
    
    df["G_TREND"] = (df["mkt_ret24h"].abs() > dyn_thr).astype(np.int32)
    df["mkt_rv_ratio"] = df["mkt_rv"] / (df["mkt_rv_med"] + 1e-12)

    rv_mean = ff.numba_rolling_mean(df[["mkt_rv"]], gate_vol_lookback_hours)["mkt_rv"].shift(1)
    rv_std = ff.numba_rolling_std(df[["mkt_rv"]], gate_vol_lookback_hours)["mkt_rv"].shift(1).clip(lower=1e-6)
    df["mkt_rv_pct"] = ((df["mkt_rv"] - rv_mean) / rv_std).clip(-6, 6).fillna(0.0).astype(np.float32)
    df["mkt_rv_pct"] = (0.5 * (1.0 + np.vectorize(np.math.erf)(df["mkt_rv_pct"] / np.sqrt(2.0)))).astype(np.float32)

    abs_ret = df["mkt_ret24h"].abs()
    abs_ret_mean = ff.numba_rolling_mean(abs_ret.to_frame("x"), gate_vol_lookback_hours)["x"].shift(1)
    abs_ret_std = ff.numba_rolling_std(abs_ret.to_frame("x"), gate_vol_lookback_hours)["x"].shift(1).clip(lower=1e-6)
    df["abs_mkt_ret24h_z"] = ((abs_ret - abs_ret_mean) / abs_ret_std).clip(-6, 6).fillna(0.0).astype(np.float32)
    df["trend_bin3"] = np.digitize(df["abs_mkt_ret24h_z"].to_numpy(), bins=[-0.5, 0.5]).astype(np.int8)

    float_cols = ["mkt_rv_med", "mkt_rv_ratio", "mkt_rv_pct", "abs_mkt_ret24h_z", "trend_bin3"]
    for c in float_cols:
        df[c] = df[c].astype(np.float32)

    return df


def compute_vol_regime_features(close_df: pd.DataFrame, vol_window: int = 24, pct_window: int = 252):
    """Compute volatility-regime features from close prices."""
    # Use np.maximum to avoid log(0) or negative values in the ratio
    ratio = close_df / close_df.shift(1)
    ratio_safe = np.maximum(ratio, 1e-9)
    ret = np.log(ratio_safe)
    rv = ret.rolling(vol_window).std().shift(1)

    vol_pct = rv.rolling(pct_window).rank(pct=True).clip(0.0, 1.0)
    vol_high = (vol_pct - 0.8).clip(lower=0.0)
    vol_low = (0.2 - vol_pct).clip(lower=0.0)

    return vol_pct.astype(np.float32), vol_high.astype(np.float32), vol_low.astype(np.float32)


def compute_cusum_regime_features(cusum_strength_df: pd.DataFrame, h: float):
    """Normalize cusum strength and expose a high-regime hinge."""
    cusum_strength_norm = (cusum_strength_df / (h + EPS)).clip(lower=0.0)
    cusum_high = (cusum_strength_norm - 1.0).clip(lower=0.0)
    return cusum_strength_norm.astype(np.float32), cusum_high.astype(np.float32)


def compute_liquidity_features(volume_df: pd.DataFrame, avg_window: int = 720):
    """Compute liquidity ratios from volume relative to lagged rolling baseline."""
    vol_avg = volume_df.rolling(avg_window).mean().shift(1)
    liq_ratio = volume_df / (vol_avg + EPS)
    liq_low = (1.0 - liq_ratio).clip(lower=0.0)
    return liq_ratio.astype(np.float32), liq_low.astype(np.float32)


def add_interactions(p_success_df: pd.DataFrame, vol_high: pd.DataFrame, cusum_high: pd.DataFrame, liq_low: pd.DataFrame):
    """Interaction terms between success probability signal and regime shocks."""
    return {
        "p_vol_high": (p_success_df * vol_high).astype(np.float32),
        "p_cusum_high": (p_success_df * cusum_high).astype(np.float32),
        "p_liq_low": (p_success_df * liq_low).astype(np.float32),
    }


def _robust_obs_var_per_col(df: pd.DataFrame) -> np.ndarray:
    """Robust baseline observation variance estimate per column from first differences."""
    d = df.diff().to_numpy(dtype=np.float64)
    med = np.nanmedian(d, axis=0)
    mad = np.nanmedian(np.abs(d - med), axis=0)
    sigma = (1.4826 * mad) / np.sqrt(2.0)
    var = np.square(np.clip(sigma, 1e-6, None))
    var[~np.isfinite(var)] = 1.0
    return var.astype(np.float64)


def _kalman_local_level_df(y_df: pd.DataFrame, lambda_qr: float, r_base: np.ndarray | None = None):
    """Local-level Kalman filter: y_t = x_t + eps_t, x_t = x_{t-1} + eta_t."""
    y = y_df.to_numpy(dtype=np.float64)
    t_len, n_cols = y.shape
    r = _robust_obs_var_per_col(y_df) if r_base is None else np.asarray(r_base, dtype=np.float64)
    r = np.clip(r, 1e-8, None)
    q = np.clip(lambda_qr, 1e-8, None) * r

    x = np.full_like(y, np.nan, dtype=np.float64)
    innov_var = np.full_like(y, np.nan, dtype=np.float64)
    p_state = np.full_like(y, np.nan, dtype=np.float64)

    # initialize from first finite observation or zero fallback
    first_obs = np.where(np.isfinite(y[0]), y[0], 0.0)
    x_prev = first_obs.copy()
    p_prev = r.copy()

    for t in range(t_len):
        y_t = y[t]
        x_pred = x_prev
        p_pred = p_prev + q

        s_t = p_pred + r
        k_t = p_pred / np.clip(s_t, 1e-12, None)
        innov_t = y_t - x_pred

        valid = np.isfinite(y_t)
        x_new = np.where(valid, x_pred + k_t * innov_t, x_pred)
        p_new = np.where(valid, (1.0 - k_t) * p_pred, p_pred)

        x[t] = x_new
        innov_var[t] = s_t
        p_state[t] = p_new

        x_prev = x_new
        p_prev = p_new

    return (
        pd.DataFrame(x, index=y_df.index, columns=y_df.columns).astype(np.float32),
        pd.DataFrame(innov_var, index=y_df.index, columns=y_df.columns).astype(np.float32),
        pd.DataFrame(p_state, index=y_df.index, columns=y_df.columns).astype(np.float32),
        pd.Series(r.astype(np.float32), index=y_df.columns),
    )


def _decile_monotonicity_score(signal_df: pd.DataFrame, ret_df: pd.DataFrame) -> float:
    """Cross-sectional decile monotonicity score using mean return per decile."""
    s = signal_df.to_numpy(dtype=np.float64)
    r = ret_df.to_numpy(dtype=np.float64)
    sums = np.zeros(10, dtype=np.float64)
    counts = np.zeros(10, dtype=np.float64)

    for t in range(s.shape[0]):
        s_t = s[t]
        r_t = r[t]
        valid = np.isfinite(s_t) & np.isfinite(r_t)
        if valid.sum() < 20:
            continue
        s_v = s_t[valid]
        r_v = r_t[valid]
        q = np.nanpercentile(s_v, [10,20,30,40,50,60,70,80,90])
        dec = np.searchsorted(q, s_v, side='right')
        for d in range(10):
            m = dec == d
            if m.any():
                sums[d] += r_v[m].sum()
                counts[d] += m.sum()

    means = sums / np.clip(counts, 1.0, None)
    if np.all(~np.isfinite(means)):
        return 0.0
    x = np.arange(10, dtype=np.float64)
    if np.nanstd(means) < 1e-12:
        return 0.0
    corr = np.corrcoef(x, np.nan_to_num(means, nan=np.nanmean(means)))[0, 1]
    return float(np.nan_to_num(corr, nan=0.0))


def _turnover_penalty(signal_df: pd.DataFrame) -> float:
    arr = signal_df.to_numpy(dtype=np.float64)
    sd = np.nanstd(arr, axis=0)
    z = arr / np.clip(sd, 1e-6, None)
    pos = np.tanh(z)
    dpos = np.abs(np.diff(pos, axis=0))
    return float(np.nanmean(dpos)) if dpos.size else 0.0


def tune_global_kalman_lambda(score_df: pd.DataFrame, net_ret_df: pd.DataFrame, grid_size: int = 15) -> float:
    """Tune global lambda=Q/R via decile monotonicity with mild turnover penalty on subsample."""
    n_t, n_c = score_df.shape
    row_step = max(1, n_t // 1500)
    col_step = max(1, n_c // 64)
    score_sub = score_df.iloc[::row_step, ::col_step]
    ret_sub = net_ret_df.reindex(score_sub.index).iloc[:, ::col_step]

    r_base = _robust_obs_var_per_col(score_sub)
    lam_grid = np.logspace(-3, 1, int(np.clip(grid_size, 10, 20)))

    best_lam = float(lam_grid[len(lam_grid)//2])
    best_obj = -1e18
    for lam in lam_grid:
        state_df, _, _, _ = _kalman_local_level_df(score_sub, lambda_qr=float(lam), r_base=r_base)
        mono = _decile_monotonicity_score(state_df, ret_sub)
        turn = _turnover_penalty(state_df)
        obj = mono - 0.05 * turn
        if obj > best_obj:
            best_obj = obj
            best_lam = float(lam)

    return float(best_lam)

def compute_regime_features(c, h, l, v, atr_base, mkt_gates):
    """
    Compute regime conditioning features (cusum, vol, etc.).
    Returns a dict of new features.
    """
    feats = {}

    # 1. CUSUM Strength (Trend Persistence)
    # Detects if price is persistently drifting away from mean
    # Normalized by volatility
    ret1h = c.diff(1).fillna(0)
    rv_24 = ff.numba_rolling_std(ret1h, 24)
    std_ret = (rv_24 + 1e-12)

    # Vectorized approximation: Rolling Sum of (Ret - Mean) / Vol
    # This captures local trend strength
    # Shift by 1 to ensure decision-time causality (no current-bar leakage).
    roll_z = (ff.numba_rolling_mean(ret1h / std_ret, 24) * np.sqrt(24)).shift(1)
    feats["cusum_strength"] = roll_z.astype(np.float32)

    # 2. Standardized Move Magnitude |z| (over 24h)
    ret_24 = ff.numba_rolling_sum(ret1h, 24)
    feats["move_magnitude_z"] = (ret_24 / (rv_24 * np.sqrt(24) + 1e-12)).shift(1).astype(np.float32)

    # 3. Time Since CUSUM Trigger (Trend Age Proxy)
    # Trigger when |cusum| > 5 based on lagged signal only.
    is_trigger = (feats["cusum_strength"].abs() > 5.0).astype(np.float32)
    # Count bars since last trigger (decay proxy)
    feats["cusum_decay"] = ff.numba_ewma(is_trigger, 2.0/25.0, False).shift(1).astype(np.float32)

    # 4. Volatility percentile and hinges
    vol_pct, vol_high, vol_low = compute_vol_regime_features(c, vol_window=24, pct_window=252)
    feats["vol_percentile"] = vol_pct
    feats["vol_high"] = vol_high
    feats["vol_low"] = vol_low

    # 5. Vol of Vol (Rolling Std of Sigma)
    # Coefficient of variation of volatility
    vv = ff.numba_rolling_std(rv_24, 24)
    feats["vol_of_vol"] = (vv / (rv_24 + 1e-12)).shift(1).astype(np.float32)

    # 6. ATR Percentile (similar to vol percentile but using ATR)
    atr_min = ff.numba_rolling_min(atr_base, 24*30)
    atr_max = ff.numba_rolling_max(atr_base, 24*30)
    feats["atr_percentile"] = ((atr_base - atr_min) / (atr_max - atr_min + 1e-12)).clip(0, 1).shift(1).astype(np.float32)

    # 7. Liquidity ratio and low-liquidity hinge
    liq_ratio, liq_low = compute_liquidity_features(v, avg_window=24 * 30)
    feats["liquidity_ratio"] = liq_ratio
    feats["liq_low"] = liq_low

    # 8. CUSUM normalization and high-regime hinge
    cusum_strength_norm, cusum_high = compute_cusum_regime_features(feats["cusum_strength"].abs(), h=6.0)
    feats["cusum_strength_norm"] = cusum_strength_norm
    feats["cusum_high"] = cusum_high

    assert float(feats["vol_percentile"].max().max()) <= 1.0 + 1e-6
    assert float(feats["vol_percentile"].min().min()) >= -1e-6
    assert float(feats["vol_high"].min().min()) >= -1e-6
    assert float(feats["vol_low"].min().min()) >= -1e-6
    assert float(feats["cusum_high"].min().min()) >= -1e-6
    assert float(feats["liq_low"].min().min()) >= -1e-6

    return feats


def compute_funding_proxy(c, h, l, v, mkt_df):
    c_ma = ff.numba_rolling_mean(c, 24)
    dist = (c - c_ma)

    mkt_close_df = mkt_df[["mkt_close"]]
    mkt_ma_df = ff.numba_rolling_mean(mkt_close_df, 24)
    mkt_dist = (mkt_df["mkt_close"] - mkt_ma_df["mkt_close"])

    relative_premium = dist.sub(mkt_dist, axis=0)

    candle_pos = (c - l) / ((h - l) + 1e-9)
    vol_z = zscore_rolling(v, 24)
    intensity = (candle_pos - 0.5) * vol_z

    return (relative_premium + (0.5 * intensity)).astype(np.float32)

def compute_features_hourly(panel, mkt_gates, cfg, requested_feature_keys=None):
    """
    Compute features. Joblib caching removed — features are persisted to parquet
    by save_features, and the joblib serialization doubled peak memory.
    """
    return _compute_features_impl(panel, mkt_gates, cfg, requested_feature_keys=requested_feature_keys)


def _compute_hvn_col(col, o_col, h_col, l_col, c_col, v_col):
    from .volume_node_features import hvn_lvn_features_ohlcv
    df_col = pd.DataFrame({
        "open": o_col,
        "high": h_col,
        "low": l_col,
        "close": c_col,
        "volume": v_col
    })
    return col, hvn_lvn_features_ohlcv(df_col)


def _compute_hvn_feature_frames(
    o: pd.DataFrame,
    h: pd.DataFrame,
    l: pd.DataFrame,
    c_log: pd.DataFrame,
    v: pd.DataFrame,
    hvn_keys: list[str],
    compute_col_fn=None,
) -> dict[str, pd.DataFrame]:
    if compute_col_fn is None:
        compute_col_fn = _compute_hvn_col

    hvn_results = {
        k: pd.DataFrame(index=c_log.index, columns=c_log.columns, dtype=np.float32)
        for k in hvn_keys
    }
    total_cols = len(c_log.columns)

    def _assign_hvn_result(col_name, res_df):
        for k in hvn_keys:
            hvn_results[k][col_name] = res_df[k].values.astype(np.float32)

    import multiprocessing
    max_workers = min(8, multiprocessing.cpu_count())

    try:
        os.sysconf("SC_SEM_NSEMS_MAX")
        can_use_process_pool = True
    except (AttributeError, ValueError, OSError, PermissionError):
        can_use_process_pool = False

    completed = 0
    if can_use_process_pool and total_cols > 1:
        try:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                for col in c_log.columns:
                    futures.append(
                        executor.submit(compute_col_fn, col, o[col], h[col], l[col], c_log[col], v[col])
                    )

                for future in as_completed(futures):
                    col, res_df = future.result()
                    _assign_hvn_result(col, res_df)
                    completed += 1
                    if completed % 50 == 0:
                        tprint(f"HVN/LVN: {completed}/{total_cols}")
        except (OSError, PermissionError) as e:
            tprint(f"HVN/LVN: process pool unavailable ({e}); falling back to single-process.")
            can_use_process_pool = False

    if not can_use_process_pool or total_cols <= 1:
        if total_cols > 1:
            tprint("HVN/LVN: using single-process fallback.")
        for col in c_log.columns:
            _, res_df = compute_col_fn(col, o[col], h[col], l[col], c_log[col], v[col])
            _assign_hvn_result(col, res_df)
            completed += 1
            if completed % 50 == 0:
                tprint(f"HVN/LVN: {completed}/{total_cols}")

    return hvn_results

def _compute_features_impl(panel, mkt_gates, cfg, requested_feature_keys=None):
    tprint("Features: compute base matrices")
    requested_feature_set = set(requested_feature_keys or [])

    def _needs_feature(*keys: str) -> bool:
        return (not requested_feature_set) or any(k in requested_feature_set for k in keys)
    
    # Check inputs
    # Check inputs (removing debug checks to reduce spam)
    # for k, v in panel.items():
    #     check_inf_nan(v, f"input_panel_{k}")
    
    # Validate panel data quality
    validation_results = validate_panel(panel, raise_on_error=False, verbose=False)
    if not validation_results['valid']:
        tprint(f"WARNING: Panel validation failed with {len(validation_results['errors'])} errors")
        for error in validation_results['errors'][:3]:  # Show first 3 errors
            tprint(f"  - {error}")
    
    # Memory Optim: Process sequentially and clear panel/raw data aggressively
    import gc

    # 1. Setup Index
    # We need a reference index. Use Close logic from original code.
    c_ref = panel["close"]
    new_idx = ensure_utc(pd.DataFrame(index=c_ref.index)).index

    funding_panel = panel.get("funding_rate")
    oi_panel = panel.get("open_interest")
    spot_close_panel = panel.get("spot_close")
    
    if len(mkt_gates) == len(new_idx):
        mkt_gates.index = new_idx
    else:
        mkt_gates = mkt_gates.reindex(new_idx)

    safe_log_eps = float(cfg.get("safe_log_eps", 1e-9))

    # 2. Transform Open (log-space only; no adaptive FFD on OHLC geometry)
    o_raw = panel.pop("open").astype(np.float32)
    o_raw.index = new_idx
    o = ff.numba_ewma(_safe_log_df(o_raw, eps=safe_log_eps), 2.0 / 6.0, False)
    # 3. Transform High (log-space)
    h_raw = panel.pop("high").astype(np.float32)
    h_raw.index = new_idx
    h = ff.numba_ewma(_safe_log_df(h_raw, eps=safe_log_eps), 2.0 / 6.0, False)
    # keep h_raw alive for raw ATR% computation below

    # 4. Transform Low (log-space)
    l_raw = panel.pop("low").astype(np.float32)
    l_raw.index = new_idx
    l = ff.numba_ewma(_safe_log_df(l_raw, eps=safe_log_eps), 2.0 / 6.0, False)
    # keep l_raw alive for raw ATR% computation below

    # 5. Transform Close
    c_raw = panel.pop("close").astype(np.float32)
    c_raw.index = new_idx

    # Compute Proxy Target for gate-feature selection:
    # use average of 2h and 8h forward returns to reduce horizon mismatch.
    fwd_ret_2h = (c_raw.shift(-2) / c_raw - 1.0).fillna(0.0).astype(np.float32)
    fwd_ret_4h = (c_raw.shift(-4) / c_raw - 1.0).fillna(0.0).astype(np.float32)
    fwd_ret_8h = (c_raw.shift(-8) / c_raw - 1.0).fillna(0.0).astype(np.float32)
    target_proxy = (0.3 * fwd_ret_2h + 0.4 * fwd_ret_4h + 0.3 * fwd_ret_8h).astype(np.float32)
    del fwd_ret_2h, fwd_ret_4h, fwd_ret_8h
    gc.collect()

    # --- Raw-scale asset identity features (computed before FFD transform deletes raw data) ---
    # Raw ATR% = ATR(h_raw, l_raw, c_raw, 14) / c_raw  (fraction, not log-differenced)
    _raw_atr = ff.numba_atr_no_norm(h_raw, l_raw, c_raw, n=cfg["atr_n"])
    _raw_atr_pct = (_raw_atr / (c_raw + 1e-12)).astype(np.float32)
    del _raw_atr

    # --- Liquidity Features (User Request) ---
    # Must compute before deleting h_raw, l_raw, c_raw, v_raw
    # Volume is in panel still, so we can access it
    _v_raw = panel["volume"].astype(np.float32)
    _v_raw.index = new_idx
    _rng = np.log(h_raw / np.maximum(l_raw, 1e-12)).astype(np.float32)
    _dollar_vol = (c_raw * _v_raw).astype(np.float32)
    _rng_sum_48 = ff.numba_rolling_sum(_rng, 48)
    _dv_sum_48 = ff.numba_rolling_sum(_dollar_vol, 48)
    _impact = (_rng_sum_48 / np.maximum(_dv_sum_48, 1e-12)).astype(np.float32)
    _dv_log = np.log(np.maximum(_dollar_vol, 1e-12)).astype(np.float32)
    def _zscore(x: pd.DataFrame) -> pd.DataFrame:
        return robust_zscore_rolling(x, 24 * 30, quantile=0.50)

    _liq_feats_temp = {}
    _liq_feats_temp["dv_z"] = _zscore(_dv_log).astype(np.float32)
    _liq_feats_temp["rng_z"] = _zscore(_rng).astype(np.float32)
    _liq_feats_temp["impact_z"] = _zscore(_impact).astype(np.float32)
    _liq_score = (_liq_feats_temp["dv_z"] - _liq_feats_temp["rng_z"] - _liq_feats_temp["impact_z"]).astype(np.float32)
    _liq_feats_temp["liq_score"] = _zscore(_liq_score).astype(np.float32)

    # Causal rank proxy to simulate qcut
    _liq_pct = ff.numba_rolling_rank_pct(_liq_feats_temp["liq_score"], window=24 * 30).fillna(0.5)
    _liq_feats_temp["liq_state"] = np.floor(_liq_pct.clip(0, 0.9999) * 5).astype(np.float32)

    del _v_raw, _rng, _dollar_vol, _rng_sum_48, _dv_sum_48, _impact, _dv_log, _liq_score, _liq_pct
    gc.collect()

    c_log = _safe_log_df(c_raw, eps=safe_log_eps)
    ffd_thres = float(cfg.get("ffd_thres", 1e-5))

    c = _transform_close_fixed_ffd(
        c_raw,
        d=float(cfg.get("ffd_d_base", 0.4)),
        _label="close",
        already_logged=False,
        thres=ffd_thres,
    )
    # 6. Transform Volume
    v_raw = panel.pop("volume").astype(np.float32)
    v_raw.index = new_idx
    # Raw log(volume_usd) — unnormalized, for asset identity
    _raw_log_vol = np.log1p(v_raw).astype(np.float32)
    v = _transform_volume(v_raw)
    intraday_library_feats = _compute_intraday_library_features_wide(
        open_df=o_raw,
        high_df=h_raw,
        low_df=l_raw,
        close_df=c_raw,
        volume_df=v_raw,
        requested_feature_set=requested_feature_set,
    )
    del o_raw
    gc.collect()
    
    # Clear panel rest
    panel.clear()

    feats = {}
    feats.update(_liq_feats_temp)
    feats.update(intraday_library_feats)

    # Correct return naming: log returns from log close
    feats["lr_1h"] = c_log.diff(1).astype(np.float32)
    feats["lr_2h"] = c_log.diff(2).astype(np.float32)
    feats["lr_4h"] = c_log.diff(4).astype(np.float32)
    feats["lr_6h"] = c_log.diff(6).astype(np.float32)
    feats["lr_12h"] = c_log.diff(12).astype(np.float32)
    feats["lr_24h"] = c_log.diff(24).astype(np.float32)

    # Compatibility aliases (one-release bridge)
    feats["ret1h"] = feats["lr_1h"]
    feats["ret6h"] = feats["lr_6h"]

    for H in [2, 3, 4, 5, 8, 10, 12, 16, 20, 24, 28, 48, 72, 120]:
        feats[f"ret{H}h"] = ff.numba_rolling_sum(c, H)

    # Geometry in log-space with ATR_ln normalization
    prev_c_log = c_log.shift(1)
    tr_ln_1 = (h - l)
    tr_ln_2 = (h - prev_c_log).abs()
    tr_ln_3 = (l - prev_c_log).abs()
    tr_ln = np.maximum(tr_ln_1, np.maximum(tr_ln_2, tr_ln_3))
    atr_ln = ff.numba_ewma(tr_ln, 1.0 / cfg["atr_n"], False).clip(lower=float(cfg.get("atr_ln_floor", 1e-6)))

    feats["atr_ln"] = atr_ln.astype(np.float32)
    feats["range_ln"] = (h - l).astype(np.float32)
    feats["gap_ln"] = (o - prev_c_log).astype(np.float32)
    feats["body_ln"] = (c_log - o).astype(np.float32)
    feats["upper_wick_ln"] = (h - np.maximum(o, c_log)).clip(lower=0).astype(np.float32)
    feats["lower_wick_ln"] = (np.minimum(o, c_log) - l).clip(lower=0).astype(np.float32)

    feats["range_pct"] = (feats["range_ln"] / (feats["atr_ln"] + 1e-12)).astype(np.float32)
    feats["gap_pct"] = (feats["gap_ln"] / (feats["atr_ln"] + 1e-12)).astype(np.float32)

    atr_base = atr_percent(h, l, c, n=cfg["atr_n"])
    feats["atr_pct_base"] = atr_base

    # Fixed multi-d close-only FFD block
    d_values = [float(d) for d in cfg.get("ffd_d_values", [0.4, 0.5, 0.6])]
    d_values = sorted(set(d_values))

    impulse_d_values = [float(d) for d in cfg.get("ffd_impulse_d_values", [0.6, 0.5])]
    carry_d_values = [float(d) for d in cfg.get("ffd_carry_d_values", [0.5, 0.4])]
    context_d_values = [float(d) for d in cfg.get("ffd_context_d_values", [0.4])]

    ffd_close = {}
    for d in d_values:
        d_tag = f"{int(round(d * 10)):02d}"
        ffd_c_d = _transform_close_fixed_ffd(
            c_log,
            d=d,
            _label=f"close_d{d_tag}",
            already_logged=True,
            thres=ffd_thres,
        )
        ffd_close[d] = ffd_c_d

        # Core per-d template block (ensures broad, symmetric d coverage)
        feats[f"ffd_diff_1_{d_tag}"] = ffd_c_d.diff(1).astype(np.float32)
        feats[f"ffd_diff_2_{d_tag}"] = ffd_c_d.diff(2).astype(np.float32)
        feats[f"ffd_diff_4_{d_tag}"] = ffd_c_d.diff(4).astype(np.float32)
        feats[f"ffd_diff_8_{d_tag}"] = ffd_c_d.diff(8).astype(np.float32)

        ffd_ema_6 = ff.numba_ewma(ffd_c_d, 2.0 / 7.0, False)
        ffd_ema_24 = ff.numba_ewma(ffd_c_d, 2.0 / 25.0, False)
        feats[f"ffd_ema_spread_{d_tag}"] = (ffd_ema_6 - ffd_ema_24).astype(np.float32)

        ffd_rv_12 = ff.apply_to_frame(feats[f"ffd_diff_1_{d_tag}"], ff._numba_rolling_std_nan_safe, 12)
        ffd_rv_24 = ff.apply_to_frame(feats[f"ffd_diff_1_{d_tag}"], ff._numba_rolling_std_nan_safe, 24)
        feats[f"ffd_rv_12_{d_tag}"] = ffd_rv_12.astype(np.float32)
        feats[f"ffd_rv_24_{d_tag}"] = ffd_rv_24.astype(np.float32)

        ffd_mu_24 = ff.numba_rolling_mean(ffd_c_d, 24)
        ffd_sd_24 = ff.numba_rolling_std(ffd_c_d, 24)
        feats[f"ffd_z_24_{d_tag}"] = ((ffd_c_d - ffd_mu_24) / (ffd_sd_24 + 1e-12)).astype(np.float32)

        ffd_max_24 = ff.numba_rolling_max(ffd_c_d, 24)
        ffd_min_24 = ff.numba_rolling_min(ffd_c_d, 24)
        feats[f"ffd_range_24_{d_tag}"] = (ffd_max_24 - ffd_min_24).astype(np.float32)

    # Carry layer (mid-speed continuation): d=0.5 primary, d=0.4 secondary
    for d in carry_d_values:
        if d in ffd_close:
            d_tag = f"{int(round(d * 10)):02d}"
            d_series = ffd_close[d]
            for w in cfg.get("ffd_slope_windows", [12, 24]):
                feats[f"ffd_slope_{d_tag}_{int(w)}"] = ff.apply_to_frame(d_series, ff._numba_rolling_slope, int(w)).astype(np.float32)
            mr_w = int(cfg.get("ffd_mr_window", 24))
            mu = ff.numba_rolling_mean(d_series, mr_w)
            sd = ff.numba_rolling_std(d_series, mr_w)
            feats[f"ffd_mr_z_{d_tag}"] = ((d_series - mu) / (sd + 1e-12)).astype(np.float32)

    # Impulse/event momentum diffs (fastest): d=0.6 primary, d=0.5 backup
    for d in impulse_d_values:
        if d in ffd_close:
            d_tag = f"{int(round(d * 10)):02d}"
            d_series = ffd_close[d]
            feats[f"ffd_d1_{d_tag}"] = d_series.diff(1).astype(np.float32)
            feats[f"ffd_d4_{d_tag}"] = d_series.diff(4).astype(np.float32)

    # Context/trend under noise (slowest in this triad): d=0.4 primary
    for d in context_d_values:
        if d in ffd_close:
            d_tag = f"{int(round(d * 10)):02d}"
            d_series = ffd_close[d]
            for w in cfg.get("ffd_slope_windows", [12, 24]):
                feats[f"ffd_ctx_slope_{d_tag}_{int(w)}"] = ff.apply_to_frame(d_series, ff._numba_rolling_slope, int(w)).astype(np.float32)

    def _pick_primary_d(preferred_d_values):
        for d in preferred_d_values:
            if d in ffd_close:
                return d
        return d_values[0] if d_values else float(cfg.get("ffd_d_base", 0.4))

    impulse_primary_d = _pick_primary_d(impulse_d_values)
    carry_primary_d = _pick_primary_d(carry_d_values)
    context_primary_d = _pick_primary_d(context_d_values)

    c_impulse = ffd_close.get(impulse_primary_d, c)
    c_carry = ffd_close.get(carry_primary_d, c)
    c_context = ffd_close.get(context_primary_d, c)

    # Remap key legacy return family to configured d-policy:
    # - ret1h: impulse-primary (fast shock reaction)
    # - ret2h..ret6h: carry-primary (move continuation)
    # - ret>=8h: context-primary (regime/trend under noise)
    feats["ret1h"] = c_impulse.diff(1).astype(np.float32)
    for H in [2, 3, 4, 5, 6]:
        feats[f"ret{H}h"] = ff.numba_rolling_sum(c_carry, H).astype(np.float32)
    for H in [8, 10, 12, 16, 20, 24, 28, 48, 72, 120]:
        feats[f"ret{H}h"] = ff.numba_rolling_sum(c_context, H).astype(np.float32)

    # Carry price becomes default base for many pre-existing features.
    c = c_carry

    # --- Asset Identity Features (raw-scale, NOT cross-sectionally normalized) ---
    # These provide "who is this asset" context without one-hot encoding.
    # asset_atr_level: smooth baseline of raw ATR% over 60 days — stable volatility fingerprint
    # asset_vol_level: smooth baseline of raw log(volume_usd) over 60 days — stable liquidity fingerprint
    # Use EWMA (alpha=2/(1440+1)) as fast O(T*S) proxy for rolling median.
    _ALPHA_IDENTITY = 2.0 / (24 * 60 + 1)  # EWMA alpha matching 60-day span
    feats["asset_atr_level"] = ff.numba_ewma(_raw_atr_pct, _ALPHA_IDENTITY, False).astype(np.float32)
    feats["asset_vol_level"] = ff.numba_ewma(_raw_log_vol, _ALPHA_IDENTITY, False).astype(np.float32)
    # atr_state: current ATR% / long-run level — >1 means elevated vol vs own baseline
    feats["atr_state"] = (_raw_atr_pct / (feats["asset_atr_level"] + 1e-9)).astype(np.float32)
    # vol_state: current log_vol / long-run level — >1 means elevated activity vs own baseline
    feats["vol_state"] = (_raw_log_vol / (feats["asset_vol_level"] + 1e-9)).astype(np.float32)
    del _raw_atr_pct, _raw_log_vol

    # --- D-Specific Feature Families ---
    # Realized volatility family (d=0.4,0.6)
    for d in [0.4, 0.6]:
        d_tag = f"{int(round(d * 10)):02d}"
        base_diff = feats[f"ffd_diff_1_{d_tag}"]
        feats[f"ffd_rv_2h_{d_tag}"] = ff.apply_to_frame(base_diff, ff._numba_rolling_std_nan_safe, 2).astype(np.float32)
        feats[f"ffd_rv_6h_{d_tag}"] = ff.apply_to_frame(base_diff, ff._numba_rolling_std_nan_safe, 6).astype(np.float32)
        feats[f"ffd_rv_24h_{d_tag}"] = ff.apply_to_frame(base_diff, ff._numba_rolling_std_nan_safe, 24).astype(np.float32)

    # Momentum acceleration features (d=0.6)
    for d in [0.6]:
        d_tag = f"{int(round(d * 10)):02d}"
        diff = feats[f"ffd_diff_1_{d_tag}"]
        feats[f"ffd_accel_{d_tag}"] = diff.diff().astype(np.float32)
        vol = ff.apply_to_frame(diff, ff._numba_rolling_std_nan_safe, 24)
        feats[f"ffd_z_{d_tag}"] = (diff / (vol + 1e-12)).fillna(0).clip(-50, 50).astype(np.float32)

    # Volume-price correlation features (d=0.4,0.6)
    for d in [0.4, 0.6]:
        d_tag = f"{int(round(d * 10)):02d}"
        diff = feats[f"ffd_diff_1_{d_tag}"]
        feats[f"ffd_vol_price_corr_10h_{d_tag}"] = ff.numba_rolling_corr(diff.abs(), v, 10).fillna(0).astype(np.float32)

    # Donchian channel features (d=0.4,0.6)
    for d in [0.4, 0.6]:
        d_tag = f"{int(round(d * 10)):02d}"
        d_series = ffd_close[d]
        for k in [12, 24, 48]:
            rmax = ff.numba_rolling_max(d_series, k)
            rmin = ff.numba_rolling_min(d_series, k)
            rmax_s = rmax.shift(1)
            rmin_s = rmin.shift(1)
            dir_s = np.sign(ff.numba_rolling_sum(d_series, 24))
            donch = dir_s * (d_series - rmax_s)
            donch = donch.where(dir_s > 0, -1 * (d_series - rmin_s))
            feats[f"ffd_donch_dist_{d_tag}_{k}"] = (donch / atr_base).clip(lower=0).astype(np.float32)

    # ATR expansion and tail risk (d=0.6)
    for d in [0.6]:
        d_tag = f"{int(round(d * 10)):02d}"
        d_series = ffd_close[d]
        tr_d = np.maximum(h - l, np.maximum((h - d_series.shift(1)).abs(), (l - d_series.shift(1)).abs()))
        atr_tr_d = ff.numba_ewma(tr_d, 1.0/cfg["atr_n"], False)
        feats[f"ffd_atr_expansion_{d_tag}"] = (tr_d / (atr_tr_d + 1e-12)).astype(np.float32)
        diff = d_series.diff(1)
        feats[f"ffd_cvar_5pct_{d_tag}"] = ff.numba_rolling_quantile(diff, 48, 0.05).fillna(0).astype(np.float32)

    # Liquidity shock features (d=0.4,0.6)
    for d in [0.4, 0.6]:
        d_tag = f"{int(round(d * 10)):02d}"
        diff = feats[f"ffd_diff_1_{d_tag}"]
        illiq_raw = (diff.abs() / ((v * ffd_close[d]) + 1e-12)).replace([np.inf, -np.inf], np.nan)
        feats[f"ffd_amihud_{d_tag}"] = ff.numba_rolling_mean(illiq_raw, 24).fillna(0).astype(np.float32)
        vr = v * diff.abs()
        ema_vr = ema(vr, 24)
        ratio_floor = float(cfg.get("ratio_denom_floor", 1e-6))
        vr_ratio = vr / ema_vr.abs().clip(lower=ratio_floor)
        if bool(cfg.get("ratio_use_log", True)):
            vr_ratio = np.log1p(vr_ratio.clip(lower=0))
        feats[f"ffd_vol_range_shock_{d_tag}"] = vr_ratio.astype(np.float32)

    # --- Technical Regime (Ridge) Features (User Request) ---
    tprint("Features: technical regime (ridge) indicators")
    ema20 = ff.numba_ewma(c_log, 2.0 / 21.0, False)
    ema50 = ff.numba_ewma(c_log, 2.0 / 51.0, False)
    ema200 = ff.numba_ewma(c_log, 2.0 / 201.0, False)
    
    _safe_range_ln = (h - l).clip(lower=1e-12)
    feats["range_atr"] = (feats["range_ln"] / (feats["atr_ln"] + 1e-12)).astype(np.float32)
    feats["body_ratio"] = ((c_log - o).abs() / _safe_range_ln).astype(np.float32)
    feats["upper_wick_ratio"] = ((h - np.maximum(o, c_log)) / _safe_range_ln).astype(np.float32)
    feats["lower_wick_ratio"] = ((np.minimum(o, c_log) - l) / _safe_range_ln).astype(np.float32)
    
    # Aliases for exact user-requested names
    feats["upper_wick"] = feats["upper_wick_ratio"]
    feats["lower_wick"] = feats["lower_wick_ratio"]
    
    feats["ema20_slope_5h"] = ((ema20 - ema20.shift(5)) / (feats["atr_ln"] + 1e-12)).astype(np.float32)
    feats["ema_slope_norm"] = feats["ema20_slope_5h"]
    feats["ema_slope"] = (ema20 - ema20.shift(5)).astype(np.float32)
    
    feats["pullback_depth"] = ((ema20 - l) / (feats["atr_ln"] + 1e-12)).astype(np.float32)
    
    atr_long = ff.numba_ewma(tr_ln, 1.0 / (24 * 7), False).clip(lower=1e-9)
    feats["atr_compression_ratio"] = (feats["atr_ln"] / atr_long).astype(np.float32)
    feats["compression_ratio"] = feats["atr_compression_ratio"]
    
    _accel_raw = (c_log - 2 * c_log.shift(1) + c_log.shift(2))
    feats["acceleration"] = _accel_raw.astype(np.float32)
    feats["acceleration_norm"] = (_accel_raw / (feats["atr_ln"] + 1e-12)).astype(np.float32)
    
    # Volume Spike: volume / volume_ma(24)
    vol_ma24 = ff.numba_rolling_mean(v, 24)
    feats["volume_spike"] = (v / (vol_ma24 + 1e-12)).astype(np.float32)
    
    feats["ema20_gt_ema50"] = (ema20 > ema50).astype(np.float32)
    feats["ema50_gt_ema200"] = (ema50 > ema200).astype(np.float32)
    feats["dist_ema20_atr"] = ((c_log - ema20) / (feats["atr_ln"] + 1e-12)).astype(np.float32)
    feats["distance_to_ema"] = feats["dist_ema20_atr"]
    
    # --- End Technical Regime ---

    # Distance-from-mean-reversion features (d=0.4)
    for d in [0.4]:
        d_tag = f"{int(round(d * 10)):02d}"
        d_series = ffd_close[d]
        ema_fast = ema(d_series, max(4, int(cfg["ema_fast"] * 0.5)))
        ema_slow = ema(d_series, int(cfg["ema_fast"] * 2))
        feats[f"ffd_dist_ema_fast_{d_tag}"] = ((d_series - ema_fast) / (atr_base + 1e-12)).astype(np.float32)
        feats[f"ffd_dist_ema_slow_{d_tag}"] = ((d_series - ema_slow) / (atr_base + 1e-12)).astype(np.float32)

    # D-family strength indicators
    abs_04 = feats["ffd_diff_1_04"].abs()
    abs_05 = feats["ffd_diff_1_05"].abs()
    abs_06 = feats["ffd_diff_1_06"].abs()
    total = abs_04 + abs_05 + abs_06 + 1e-12
    feats["ffd_strength_04"] = (abs_04 / total).astype(np.float32)
    feats["ffd_strength_05"] = (abs_05 / total).astype(np.float32)
    feats["ffd_strength_06"] = (abs_06 / total).astype(np.float32)

    rsi_base = rsi(c, n=cfg["rsi_n"])
    feats["rsi_base"] = rsi_base
    feats["rsi_slope_base"] = rsi_base.diff(cfg["rsi_slope_n"]).astype(np.float32)


    feats["rv_24h"] = ff.apply_to_frame(feats["ret1h"], ff._numba_rolling_std_nan_safe, 24)
    feats["rv_2h"] = ff.apply_to_frame(feats["ret1h"], ff._numba_rolling_std_nan_safe, 2)
    feats["rv_4h"] = ff.apply_to_frame(feats["ret1h"], ff._numba_rolling_std_nan_safe, 4)
    feats["rv_6h"] = ff.apply_to_frame(feats["ret1h"], ff._numba_rolling_std_nan_safe, 6)
    feats["rv_8h"] = ff.apply_to_frame(feats["ret1h"], ff._numba_rolling_std_nan_safe, 8)
    feats["rv_12h"] = ff.apply_to_frame(feats["ret1h"], ff._numba_rolling_std_nan_safe, 12)

    # New Filter Features (Range & Vol Z-score)
    h_24 = ff.numba_rolling_max(h, 24)
    l_24 = ff.numba_rolling_min(l, 24)
    h_12 = ff.numba_rolling_max(h, 12)
    l_12 = ff.numba_rolling_min(l, 12)
    h_16 = ff.numba_rolling_max(h, 16)
    l_16 = ff.numba_rolling_min(l, 16)

    # range_XXh_pct is max_h - min_l. inputs are log-FFD, so diff is %-ish.
    # Do NOT divide by c (FFD) as it crosses 0.
    # Use np.where to handle cases where rolling windows produce NaN
    feats["range_24h_pct"] = np.where(np.isfinite(h_24) & np.isfinite(l_24), (h_24 - l_24), 0.0).astype(np.float32)
    feats["range_12h_pct"] = np.where(np.isfinite(h_12) & np.isfinite(l_12), (h_12 - l_12), 0.0).astype(np.float32)
    feats["range_16h_pct"] = np.where(np.isfinite(h_16) & np.isfinite(l_16), (h_16 - l_16), 0.0).astype(np.float32)
    del h_24, l_24, h_12, l_12, h_16, l_16

    # Volatility Z-score (using Log-ATR robust z-score)
    # Baseline: 90 days. x = log(ATR/Close).
    # Z = (x - Q(0.50)) / (1.4826 * MAD)
    # atr_base is raw ATR (price units), so we normalize by C
    vol_proxy = (atr_base / (c + 1e-12))
    log_vol = np.log(vol_proxy + 1e-9).astype(np.float32)
    vol_z = robust_zscore_rolling(log_vol, 24 * 90, quantile=0.50)
    feats["volatility_zscore"] = np.where(np.isfinite(vol_z), vol_z, 0.0).astype(np.float32)
    del vol_proxy, log_vol

    feats["qv"] = (c * v).astype(np.float32)
    feats["vol_z24_base"] = zscore_rolling(v, 24)
    feats["vol_z_base"]   = zscore_rolling(v, cfg["volz_n"])

    ema_fast_base = ema(c, cfg["ema_fast"])
    ema_slow_base = ema(c, cfg["ema_slow"])
    feats["dist_ema_fast_base"] = ((c - ema_fast_base) / (atr_base + 1e-12)).astype(np.float32)
    feats["dist_ema_slow_base"] = ((c - ema_slow_base) / (atr_base + 1e-12)).astype(np.float32)

    feats["roc_div"] = (feats["ret1h"] - feats["ret6h"]).astype(np.float32)
    # ret1h_z: if rv_24h is 0 (constant trend), this explodes. Cap it.
    z_raw = feats["ret1h"] / (feats["rv_24h"] + 1e-9)
    feats["ret1h_z"] = z_raw.fillna(0).clip(-50, 50).astype(np.float32)

    body = (c - o).abs()
    upper_wick = (h - c.where(c >= o, o)).clip(lower=0)
    lower_wick = (c.where(c <= o, o) - l).clip(lower=0)
    feats["body_pct"] = body.astype(np.float32)
    feats["wick_body_ratio"] = ((upper_wick + lower_wick) / (body + 1e-12)).astype(np.float32)

    # New Spike Features
    max_oc = np.maximum(o, c)
    feats["wick_ratio"] = ((h - max_oc) / ((h - l) + 1e-12)).astype(np.float32)
    del body, upper_wick, lower_wick, max_oc

    # --- New Exhaustion & Risk Features (Report 2026-02-10) ---

    # 1. Wick Ratio Max (Exhaustion for short_mr)
    feats["wick_ratio_4h_max"] = ff.numba_rolling_max(feats["wick_ratio"], 4).astype(np.float32)

    # 2. Volume/Price Divergence (Exhaustion for short_mr)
    # Correlation between price changes and volume changes over 12 hours.
    v_chg = ff.numba_pct_change(v, 1).fillna(0).astype(np.float32)
    # Using numba rolling corr (O(N) vs Pandas O(N^2) or O(N log N))
    feats["vol_price_div"] = ff.numba_rolling_corr(feats["ret1h"], v_chg, 12).fillna(0).astype(np.float32)
    del v_chg

    # 3. RSI Lagged (for divergence check)
    # Use base RSI here (adaptive RSI is created later).
    feats["rsi_lag1"] = rsi_base.shift(1).astype(np.float32)
    # RSI Slope 1h (Momentum Turn for long_mr)
    feats["rsi_1h_slope"] = rsi_base.diff(1).fillna(0).astype(np.float32)

    # 4. Tail Risk (CVaR Proxy for long_tf)
    # 5th percentile return over 48 hours (2 days)
    # Use Numba-optimized rolling quantile (O(N) vs Pandas O(N log W))
    feats["cvar_5pct"] = ff.numba_rolling_quantile(feats["ret1h"], 48, 0.05).fillna(0).astype(np.float32)

    # 5. Liquidity Shock (Amihud Proxy for long_tf)
    # |Ret| / (Volume * Price). Spikes indicate price moving on thin liquidity.
    illiq_raw = (feats["ret1h"].abs() / ((v * c) + 1e-12)).replace([np.inf, -np.inf], np.nan)
    feats["amihud_illiq"] = ff.numba_rolling_mean(illiq_raw, 24).fillna(0).astype(np.float32)

    # 6. Skew Proxy (Close Location Value Mean)
    clv_raw_early = ((2 * c - h - l) / ((h - l) + 1e-9)).fillna(0)
    feats["clv_mean_24"] = ff.apply_to_frame(clv_raw_early, ff._numba_rolling_mean_nan_safe, 24).fillna(0).astype(np.float32)


    # 7. Stabilization / Falling Knife Features (for long_mr)
    # Climax Volume
    feats["vol_z_4h"] = zscore_rolling(v, 4).fillna(0).astype(np.float32)

    # ATR pct change (Volatility Cooling)
    feats["atr_pct_change"] = atr_base.pct_change().fillna(0).astype(np.float32)

    # --- End New Features ---

    feats["vol_price_spread"] = (v / ((h - l) + 1e-12)).astype(np.float32)

    prev_close = c.shift(1)
    tr_1 = (h - l)
    tr_2 = (h - prev_close).abs()
    tr_3 = (l - prev_close).abs()
    tr = np.maximum(tr_1, np.maximum(tr_2, tr_3))
    atr_tr = ff.numba_ewma(tr, 1.0/cfg["atr_n"], False)
    feats["atr_expansion"] = (tr / (atr_tr + 1e-12)).astype(np.float32)
    del prev_close, tr_1, tr_2, tr_3, tr, atr_tr

    sma_base = ff.numba_rolling_mean(c_context, cfg["trend_sma_n"])
    feats["trend_pct_base"] = (c_context - sma_base).astype(np.float32)

    hod = pd.Series(v.index.hour, index=v.index)
    rvol_denom = ff.numba_grouped_rolling_mean(v, hod, int(cfg["rvol_days"]*24))
    feats["rvol_hod_base"] = (v / (rvol_denom + 1e-12)).astype(np.float32)

    feats["funding_proxy"] = compute_funding_proxy(c, h, l, v, mkt_gates)

    sin_hod, cos_hod, sin_dow, cos_dow = time_sin_cos(c.index)
    feats["sin_hod"] = pd.DataFrame(np.repeat(sin_hod[:,None], c.shape[1], axis=1), index=c.index, columns=c.columns).astype(np.float32)
    feats["cos_hod"] = pd.DataFrame(np.repeat(cos_hod[:,None], c.shape[1], axis=1), index=c.index, columns=c.columns).astype(np.float32)
    feats["sin_dow"] = pd.DataFrame(np.repeat(sin_dow[:,None], c.shape[1], axis=1), index=c.index, columns=c.columns).astype(np.float32)
    feats["cos_dow"] = pd.DataFrame(np.repeat(cos_dow[:,None], c.shape[1], axis=1), index=c.index, columns=c.columns).astype(np.float32)

    signed_vol = v * np.sign(c - o)
    sv_abs = signed_vol.abs()
    ewma_sv_fast = ema(signed_vol, 6)
    ewma_sv_slow = ema(sv_abs, 24)

    feats["flow_persistence"] = (ewma_sv_fast / (ewma_sv_slow + 1e-12)).astype(np.float32)
    feats["flow_ratio"] = feats["flow_persistence"]

    eff = (c - o).abs() / ((h - l) + 1e-9)
    feats["efficiency"] = ff.numba_rolling_mean(eff, 12)

    # Use Pearson Mode Skewness Proxy: 3 * (Mean - Median) / Std
    # More stable for small N (works for N>=2) and cheaper.
    r1 = feats["ret1h"]
    cs_mean = r1.mean(axis=1)
    cs_median = r1.median(axis=1)
    cs_std = r1.std(axis=1)

    skew_ser = 3.0 * (cs_mean - cs_median) / (cs_std + 1e-6)
    feats["skew"] = pd.DataFrame(np.repeat(skew_ser.values[:,None], c.shape[1], axis=1), index=c.index, columns=c.columns).astype(np.float32)

    r = feats["ret1h"]
    r2 = r**2
    up_sq = r2.where(r > 0, 0.0)
    dn_sq = r2.where(r < 0, 0.0)
    up_vol = ema(up_sq, 24)
    dn_vol = ema(dn_sq, 24)
    feats["up_vol"] = up_vol
    feats["dn_vol"] = dn_vol
    feats["vol_asym"] = (up_vol - dn_vol).astype(np.float32)

    up_vol_6 = ema(up_sq, 6)
    dn_vol_6 = ema(dn_sq, 6)
    feats["up_vol_6"] = up_vol_6
    feats["dn_vol_6"] = dn_vol_6
    feats["vol_asym_6"] = (up_vol_6 - dn_vol_6).astype(np.float32)

    l_prev2 = l.shift(2)
    h_prev2 = h.shift(2)
    # FVG uses log-FFD prices, so diff is already relative. Do not divide by c.
    fvg_bull = (l_prev2 - h).clip(lower=0) 
    fvg_bear = (l - h_prev2).clip(lower=0)
    feats["fvg"] = (fvg_bull - fvg_bear).astype(np.float32)

    feats["churn"] = (v / ((c - o).abs() + 1e-12)).astype(np.float32)
    feats["slope"] = ((ema_fast_base - ema_slow_base) / (atr_base + 1e-12)).astype(np.float32)

    t_snr_num = ema(feats["ret1h"], 6).abs()
    t_snr_den = ff.numba_rolling_std(feats["ret1h"], 24)
    feats["trend_snr"] = (t_snr_num / (t_snr_den + 1e-12)).astype(np.float32)

    # v_power: Volume / Abs Price Change? Normalizing by c.abs() (FFD) is unstable if c~0.
    # Normalize by ATR base instead.
    feats["v_power"] = (v / (atr_base + 1e-9)).astype(np.float32)
    feats["signed_vol"] = signed_vol.astype(np.float32)

    atr_ema_f = ema(atr_base, 6)
    atr_ema_s = ema(atr_base, 24)
    feats["atr_slope"] = ((atr_ema_f - atr_ema_s) / (atr_ema_s + 1e-12)).astype(np.float32)

    vwap_24 = ff.numba_rolling_vwap(c, v, 24)

    feats["dist_vwap_norm"] = ((c - vwap_24) / (atr_base + 1e-12)).astype(np.float32)

    feats["momentum_accel"] = feats["ret1h"].diff().astype(np.float32)

    log_v = v
    feats["rvol_z"] = zscore_rolling(
        log_v,
        cfg["volz_n"],
        winsorize=bool(cfg.get("zscore_winsorize", True)),
        q_lo=float(cfg.get("zscore_winsor_q_lo", 0.01)),
        q_hi=float(cfg.get("zscore_winsor_q_hi", 0.99)),
        std_floor=float(cfg.get("zscore_std_floor", 1e-6)),
        use_ewma=bool(cfg.get("zscore_use_ewma", False)),
        ewma_span=int(cfg.get("zscore_ewma_span", cfg["volz_n"])),
    ).astype(np.float32)

    vr = v * feats["ret1h"].abs()
    ema_vr = ema(vr, 24)
    ratio_floor = float(cfg.get("ratio_denom_floor", 1e-6))
    ema_vr_floor = ema_vr.abs().clip(lower=ratio_floor)
    vol_range_ratio = vr / ema_vr_floor
    if bool(cfg.get("ratio_use_log", True)):
        vol_range_ratio = np.log1p(vol_range_ratio.clip(lower=0))
    feats["vol_range_shock"] = vol_range_ratio.astype(np.float32)

    v_max = ff.numba_rolling_max(v, 24)
    v_floor = v.abs().clip(lower=ratio_floor)
    climax_ratio = v_max / v_floor
    if bool(cfg.get("ratio_use_log", True)):
        climax_ratio = np.log1p(climax_ratio.clip(lower=0))
    feats["climax_decay"] = climax_ratio.astype(np.float32)

    cum_sv = ff.numba_rolling_sum(signed_vol, 24)
    # Correlation uses internal robust logic, but fillna(0) is good
    feats["cumulative_delta_stall"] = ff.numba_rolling_corr(c, cum_sv, 24).fillna(0).astype(np.float32)
    cum_sv_6 = ff.numba_rolling_sum(signed_vol, 6)
    feats["delta_stall_6"] = ff.numba_rolling_corr(c, cum_sv_6, 6).fillna(0).astype(np.float32)

    feats["vol_expansion_ratio"] = (atr_ema_f / (atr_ema_s + 1e-12)).astype(np.float32)

    sig_s = ff.numba_rolling_std(feats["ret1h"], 6)
    sig_m = ff.numba_rolling_std(feats["ret1h"], 18)
    sig_m_floor = sig_m.abs().clip(lower=ratio_floor)
    vol_comp = sig_s / sig_m_floor
    if bool(cfg.get("ratio_use_log", False)):
        vol_comp = np.log1p(vol_comp.clip(lower=0))
    feats["vol_compression"] = vol_comp.astype(np.float32)

    rv_ratio_s = mkt_gates["mkt_rv_ratio"].reindex(c.index).astype(np.float32)
    rv_ratio = pd.DataFrame(np.repeat(rv_ratio_s.to_numpy()[:,None], c.shape[1], axis=1),
                            index=c.index, columns=c.columns).astype(np.float32)
    feats["mkt_rv_ratio"] = rv_ratio

    mkt_rv_pct_s = mkt_gates["mkt_rv_pct"].reindex(c.index).astype(np.float32)
    mkt_rv_pct = pd.DataFrame(np.repeat(mkt_rv_pct_s.to_numpy()[:, None], c.shape[1], axis=1),
                              index=c.index, columns=c.columns).astype(np.float32)
    feats["mkt_rv_pct"] = mkt_rv_pct

    abs_mkt_ret24h_z_s = mkt_gates["abs_mkt_ret24h_z"].reindex(c.index).astype(np.float32)
    abs_mkt_ret24h_z = pd.DataFrame(np.repeat(abs_mkt_ret24h_z_s.to_numpy()[:, None], c.shape[1], axis=1),
                                    index=c.index, columns=c.columns).astype(np.float32)
    feats["abs_mkt_ret24h_z"] = abs_mkt_ret24h_z

    trend_bin3_s = mkt_gates["trend_bin3"].reindex(c.index).astype(np.float32)
    trend_bin3 = pd.DataFrame(np.repeat(trend_bin3_s.to_numpy()[:, None], c.shape[1], axis=1),
                              index=c.index, columns=c.columns).astype(np.float32)
    feats["trend_bin3"] = trend_bin3

    def pick_by_rv(fast_df, base_df, slow_df):
        rr = rv_ratio
        smooth_span = max(1, int(cfg.get("rv_selector_smooth_span", 6)))
        if smooth_span > 1:
            rr = ff.numba_ewma(rr, 2.0 / (smooth_span + 1.0), False)
        fast_thr = float(cfg["rv_ratio_fast_thr"])
        slow_thr = float(cfg["rv_ratio_slow_thr"])
        mode = str(cfg.get("rv_selector_mode", "blend")).lower()

        if mode == "blend" and fast_thr > slow_thr:
            mid = 0.5 * (fast_thr + slow_thr)
            half = max(0.5 * (fast_thr - slow_thr), 1e-6)
            dist = ((rr - mid).abs() / half).clip(upper=1.0)
            w_base = (1.0 - dist).clip(lower=0.0, upper=1.0)
            rem = 1.0 - w_base
            w_fast_side = ((rr - mid) / half).clip(lower=0.0, upper=1.0)
            w_slow_side = ((mid - rr) / half).clip(lower=0.0, upper=1.0)
            w_fast = rem * w_fast_side
            w_slow = rem * w_slow_side
            out = w_fast * fast_df + w_base * base_df + w_slow * slow_df
            return out.astype(np.float32)

        hyst = max(0.0, float(cfg.get("rv_selector_hysteresis", 0.02)))
        out = base_df.copy()
        out = out.where(~(rr > (fast_thr + hyst)), fast_df)
        out = out.where(~(rr < (slow_thr - hyst)), slow_df)
        return out.astype(np.float32)

    rsi_fast = rsi(c, max(2, int(cfg["rsi_n"] * 0.5)))
    rsi_slow = rsi(c, int(cfg["rsi_n"] * 2))
    feats["rsi"] = pick_by_rv(rsi_fast, rsi_base, rsi_slow)
    del rsi_fast, rsi_slow

    atr_fast = atr_percent(h, l, c, max(2, int(cfg["atr_n"] * 0.5)))
    atr_slow = atr_percent(h, l, c, int(cfg["atr_n"] * 2))
    feats["atr_pct"] = pick_by_rv(atr_fast, atr_base, atr_slow)
    del atr_fast, atr_slow

    volz_fast = zscore_rolling(v, max(24, int(cfg["volz_n"] * 0.5)))
    volz_slow = zscore_rolling(v, int(cfg["volz_n"] * 2))
    feats["vol_z"] = pick_by_rv(volz_fast, feats["vol_z_base"], volz_slow)
    del volz_fast, volz_slow

    # --- New Volume & Liquidity Gates (Z-score based) ---
    feats["G_VOL_LIQ_GT1"] = (feats["vol_z"] > 1.0).astype(np.int8)
    feats["G_VOL_LIQ_GT2"] = (feats["vol_z"] > 2.0).astype(np.int8)
    feats["G_VOL_LIQ_GT3"] = (feats["vol_z"] > 3.0).astype(np.int8)

    # Amihud Z-score (Illiquidity Z-score, lower is better)
    # Use robust Z-score over long window (30d)
    amihud_log = np.log(feats["amihud_illiq"] + 1e-12)
    feats["amihud_z"] = robust_zscore_rolling(amihud_log, 24 * 30, quantile=0.50).astype(np.float32)
    del amihud_log

    # Liquidity Gates (0 = average, -1 = good liquidity, -2 = excellent)
    # Since amihud is illiquidity, lower Z is better.
    feats["G_LIQ_GOOD"] = (feats["amihud_z"] < 0.0).astype(np.int8)
    feats["G_LIQ_GREAT"] = (feats["amihud_z"] < -1.0).astype(np.int8)
    feats["G_LIQ_EXCEL"] = (feats["amihud_z"] < -2.0).astype(np.int8)

    # Earlier trend detection / volatility-of-volatility composites
    vov_fast = ff.numba_rolling_std(feats["ret1h"], 20)
    vov_slow = ff.numba_rolling_std(feats["ret1h"], 60)
    q25_20, q75_20 = ff.numba_rolling_quantile_dual(vov_fast, 20, 0.25, 0.75)
    feats["vov_iqr_20"] = (q75_20 - q25_20).astype(np.float32)
    feats["vov_mad_20"] = rolling_mad(vov_fast, 20)
    feats["vov_mad_60"] = rolling_mad(vov_fast, 60)
    feats["vov_ratio"] = (feats["vov_mad_20"] / (feats["vov_mad_60"] + 1e-12)).astype(np.float32)
    feats["vov_fast_slow_ratio"] = (vov_fast / (vov_slow + 1e-12)).astype(np.float32)
    relu_vov_z = feats["vol_z"].clip(lower=0)
    feats["vov_interaction"] = (feats["vol_z"] * relu_vov_z).astype(np.float32)
    del vov_fast, vov_slow, q25_20, q75_20, relu_vov_z

    feats["accel_5h"] = (feats["ret5h"] - (feats["ret10h"] / 2.0)).astype(np.float32)
    feats["dlog_vol_5h"] = (v - v.shift(5)).astype(np.float32)
    max_bar = ff.numba_rolling_max(feats["ret1h"].abs(), 5)
    sign_max_bar = np.sign(ff.numba_rolling_sum(feats["ret1h"], 5))
    feats["signed_max_bar_ret_5h"] = (sign_max_bar * max_bar).astype(np.float32)
    q90_dx = ff.numba_rolling_quantile(feats["ret1h"].abs(), 24 * 30, 0.90)
    feats["jump_rate_10h"] = ff.numba_rolling_mean((feats["ret1h"].abs() > q90_dx).astype(np.float32), 10).astype(np.float32)
    vol_mu_30d = ff.numba_rolling_mean(v, 24 * 30)
    vol_sd_30d = ff.numba_rolling_std(v, 24 * 30)
    feats["volu_z"] = ((v - vol_mu_30d) / (vol_sd_30d + 1e-12)).astype(np.float32)
    del max_bar, sign_max_bar, q90_dx, vol_mu_30d, vol_sd_30d
    feats["vol_z_30_calm"] = robust_zscore_rolling(np.log(feats["atr_pct_base"] + 1e-9), 24 * 30, quantile=0.50).astype(np.float32)
    feats["volume_price_corr_10h"] = ff.numba_rolling_corr(feats["ret1h"].abs(), v, 10).fillna(0).astype(np.float32)

    sma_fast = ff.numba_rolling_mean(c_context, max(24, int(cfg["trend_sma_n"] * 0.5)))
    sma_slow = ff.numba_rolling_mean(c_context, int(cfg["trend_sma_n"] * 2))
    trend_fast = (c_context - sma_fast)
    trend_slow = (c_context - sma_slow)
    feats["trend_pct"] = pick_by_rv(trend_fast, feats["trend_pct_base"], trend_slow)
    del sma_fast, sma_slow, trend_fast, trend_slow

    ema_fast_f = ema(c, max(4, int(cfg["ema_fast"] * 0.5)))
    ema_fast_s = ema(c, int(cfg["ema_fast"] * 2))
    dist_fast_f = (c - ema_fast_f) / (feats["atr_pct"] + 1e-12)
    dist_fast_s = (c - ema_fast_s) / (feats["atr_pct"] + 1e-12)
    feats["dist_ema_fast"] = pick_by_rv(dist_fast_f, feats["dist_ema_fast_base"], dist_fast_s)
    del ema_fast_f, ema_fast_s, dist_fast_f, dist_fast_s

    feats["vol_z24"] = feats["vol_z24_base"]
    feats["rsi_slope"] = feats["rsi"].diff(cfg["rsi_slope_n"]).astype(np.float32)
    feats["a_funding_proxy"] = feats["funding_proxy"]

    if bool(cfg.get("use_perps", False)):
        if isinstance(funding_panel, pd.DataFrame) and isinstance(oi_panel, pd.DataFrame):
            tprint("Computing perp derivative features...")
            perp_price_panel = np.exp(c_log).astype(np.float32)
            volume_panel = np.exp(v).astype(np.float32)
            if isinstance(spot_close_panel, pd.DataFrame):
                spot_price_panel = spot_close_panel.reindex(
                    index=perp_price_panel.index,
                    columns=perp_price_panel.columns,
                ).astype(np.float32)
                spot_price_panel = spot_price_panel.where(spot_price_panel > 0, perp_price_panel)
            else:
                spot_price_panel = perp_price_panel

            funding_aligned = funding_panel.reindex(
                index=perp_price_panel.index,
                columns=perp_price_panel.columns,
            )
            oi_aligned = oi_panel.reindex(
                index=perp_price_panel.index,
                columns=perp_price_panel.columns,
            )

            perp_buffers: dict[str, dict[str, pd.Series]] = {}
            for sym in perp_price_panel.columns:
                df_sym = pd.DataFrame(
                    {
                        "funding_rate": funding_aligned[sym],
                        "open_interest": oi_aligned[sym],
                        "perp_price": perp_price_panel[sym],
                        "spot_price": spot_price_panel[sym],
                        "volume": volume_panel[sym],
                        "close": perp_price_panel[sym],
                    },
                    index=perp_price_panel.index,
                )
                try:
                    sym_feats = compute_perp_features(df_sym)
                except Exception as exc:
                    tprint(f"WARN perp feature compute failed for {sym}: {exc}")
                    continue

                for raw_name, ser in sym_feats.items():
                    feat_name = _PERP_FEATURE_COLLISION_RENAMES.get(raw_name, raw_name)
                    if feat_name not in perp_buffers:
                        perp_buffers[feat_name] = {}
                    perp_buffers[feat_name][sym] = pd.to_numeric(ser, errors="coerce").astype(np.float32)

            for feat_name, by_sym in perp_buffers.items():
                feats[feat_name] = (
                    pd.DataFrame(by_sym)
                    .reindex(index=perp_price_panel.index, columns=perp_price_panel.columns)
                    .astype(np.float32)
                )
            tprint(f"Perp derivative features added: {len(perp_buffers)}")
        else:
            tprint("Perps mode enabled but funding/open_interest data missing; skipping perp derivatives block.")

    # --- Regime Conditioning Features ---
    if bool(cfg.get("use_regime_features", True)):
        feats.update(compute_regime_features(c, h, l, v, atr_base, mkt_gates))

    # --- New Helper Features for Models ---
    dir_s = np.sign(feats["ret24h"])
    dir_s[dir_s == 0] = 1 # fallback

    atr = feats["atr_pct"] + 1e-12
    rv6 = feats["rv_6h"] + 1e-12
    rv8 = feats["rv_8h"] + 1e-12
    rv12 = feats["rv_12h"] + 1e-12

    for k in [2, 4, 6, 8, 12, 24, 48, 72, 120]:
        rmax = ff.numba_rolling_max(c_context, k)
        rmin = ff.numba_rolling_min(c_context, k)

        rmax_s = rmax.shift(1)
        rmin_s = rmin.shift(1)

        donch = dir_s * (c_context - rmax_s)
        donch = donch.where(dir_s > 0, -1 * (c_context - rmin_s))
        feats[f"donch_dist_{k}"] = (donch / atr).clip(lower=0).astype(np.float32)

        pb_raw = dir_s * (c_context - rmax)
        pb_raw = pb_raw.where(dir_s > 0, -1 * (c_context - rmin))
        feats[f"pullback_{k}"] = (pb_raw / atr).astype(np.float32)

        # Distance-from-high (always negative or zero for longs, measures drawdown from peak)
        if k >= 48:
            dist_high = (c_context - rmax) / (atr + 1e-12)
            dist_low = (c_context - rmin) / (atr + 1e-12)
            feats[f"dist_from_high_{k}h"] = dist_high.astype(np.float32)
            feats[f"dist_from_low_{k}h"] = dist_low.astype(np.float32)

    # Multi-day trend slopes (SMA-based, captures macro regime)
    for k_trend in [48, 72, 120]:
        sma_k = ff.numba_rolling_mean(c_context, k_trend)
        feats[f"trend_slope_{k_trend}h"] = ((c_context - sma_k) / (atr + 1e-12)).astype(np.float32)
        # Trend acceleration: is the trend strengthening or weakening?
        feats[f"trend_accel_{k_trend}h"] = feats[f"trend_slope_{k_trend}h"].diff(12).fillna(0).astype(np.float32)
        del sma_k

    # --- Event timing + policy-normalized stage difficulty (entry-time, past-only) ---
    eps = 1e-12
    c_prev = c_context.shift(1)
    h_prev = h.shift(1)
    l_prev = l.shift(1)

    def _rolling_bars_since_extreme(df: pd.DataFrame, window: int, mode: str) -> pd.DataFrame:
        return ff.numba_rolling_bars_since_extreme(df, window, mode)

    # Time since local peak/trough in the last 12h window (all windows end at t-1)
    time_since_peak_12h = _rolling_bars_since_extreme(h_prev, 12, "max")
    time_since_trough_12h = _rolling_bars_since_extreme(l_prev, 12, "min")
    # Event-direction proxy: if 12h return into t-1 is up, use peak timing; else trough timing.
    up_dir = (c_prev / c_prev.shift(12) - 1.0) >= 0.0
    feats["time_since_peak_12h"] = time_since_peak_12h.fillna(0.0).astype(np.float32)
    feats["time_since_trough_12h"] = time_since_trough_12h.fillna(0.0).astype(np.float32)
    feats["time_since_event_extreme_12h"] = np.where(
        up_dir,
        feats["time_since_peak_12h"],
        feats["time_since_trough_12h"],
    ).astype(np.float32)

    # "Second-leg" acceleration indicators (with/without volume confirmation), past-only.
    d1 = c_prev.pct_change(1).fillna(0.0)
    d2 = c_prev.pct_change(2).fillna(0.0)
    accel_1 = (d1 - d1.shift(1)).fillna(0.0)
    accel_2 = (d2 - d2.shift(1)).fillna(0.0)
    v_prev = v.shift(1)
    vol_ratio = (v_prev / (v_prev.rolling(24, min_periods=1).median() + eps)).fillna(1.0)
    feats["second_leg_accel_1h"] = accel_1.astype(np.float32)
    feats["second_leg_accel_2h"] = accel_2.astype(np.float32)
    feats["second_leg_accel_vol_1h"] = (accel_1 * vol_ratio).astype(np.float32)
    feats["second_leg_accel_vol_2h"] = (accel_2 * vol_ratio).astype(np.float32)

    # Policy-normalized stage difficulty / timing proxies (entry-time only).
    vol_scale = feats["atr_pct"].shift(1).fillna(feats["atr_pct"]).clip(lower=eps)
    hr_48 = feats["ret1h"].abs().shift(1).rolling(48, min_periods=1).median().clip(lower=eps)
    be_threshold_pct = float(cfg.get("be_threshold_pct", 0.0035))
    profit_lock_pct = float(cfg.get("profit_lock_pct", 0.0050))
    tp_mult = float(cfg.get("tp_mult", 0.50))
    giveback_pct = float(cfg.get("giveback_pct", 0.35))
    trail_act_pct = tp_mult * vol_scale

    feats["vol_scale"] = vol_scale.astype(np.float32)
    feats["be_vol_units"] = (be_threshold_pct / (vol_scale + eps)).astype(np.float32)
    feats["pl_vol_units"] = (profit_lock_pct / (vol_scale + eps)).astype(np.float32)
    feats["trail_act_pct"] = trail_act_pct.astype(np.float32)
    feats["trail_act_vol_units"] = (trail_act_pct / (vol_scale + eps)).astype(np.float32)
    feats["giveback_vol_units"] = (giveback_pct / (vol_scale + eps)).astype(np.float32)

    feats["t_be_proxy"] = (be_threshold_pct / (hr_48 + eps)).astype(np.float32)
    feats["t_pl_proxy"] = (profit_lock_pct / (hr_48 + eps)).astype(np.float32)
    feats["t_trail_proxy"] = (trail_act_pct / (hr_48 + eps)).astype(np.float32)

    shock_12h = (c_prev / (c_prev.shift(12) + eps) - 1.0).abs().fillna(0.0)
    hh_12 = h_prev.rolling(12, min_periods=1).max()
    ll_12 = l_prev.rolling(12, min_periods=1).min()
    dist_from_low = (c_prev / (ll_12 + eps) - 1.0).fillna(0.0)
    dist_from_high = (hh_12 / (c_prev + eps) - 1.0).fillna(0.0)

    feats["shock_12h"] = shock_12h.astype(np.float32)
    feats["shock_vol_ratio"] = (shock_12h / ((vol_scale * np.sqrt(12.0)) + eps)).astype(np.float32)
    feats["dist_from_low_event_12h"] = dist_from_low.astype(np.float32)
    feats["dist_from_high_event_12h"] = dist_from_high.astype(np.float32)
    feats["dist_from_low_vol"] = (dist_from_low / (vol_scale + eps)).astype(np.float32)
    feats["dist_from_high_vol"] = (dist_from_high / (vol_scale + eps)).astype(np.float32)

    # Realized volatility at longer horizons
    feats["rv_48h"] = ff.apply_to_frame(feats["ret1h"], ff._numba_rolling_std_nan_safe, 48).astype(np.float32)
    feats["rv_120h"] = ff.apply_to_frame(feats["ret1h"], ff._numba_rolling_std_nan_safe, 120).astype(np.float32)
    # Vol regime ratio: short-term vs multi-day vol
    feats["rv_ratio_24_120"] = (feats["rv_24h"] / (feats["rv_120h"] + 1e-12)).astype(np.float32)

    # --- Multi-Horizon Aggregated Features (Report 2026-02-10) ---
    # Vectorized aggregate statistics across multiple return windows
    # feats["ret1h"] etc. are DataFrames (T, S); stack along axis=2 → (T, S, N)
    _ret_ref = feats["ret1h"]
    ret_stack = np.stack([
        feats["ret1h"].to_numpy(),
        feats["ret2h"].to_numpy(),
        feats["ret4h"].to_numpy(),
        feats["ret6h"].to_numpy(),
        feats["ret8h"].to_numpy()
    ], axis=2)
    feats["ret_mean"] = pd.DataFrame(np.nanmean(ret_stack, axis=2), index=_ret_ref.index, columns=_ret_ref.columns).astype(np.float32)
    feats["ret_max"] = pd.DataFrame(np.nanmax(ret_stack, axis=2), index=_ret_ref.index, columns=_ret_ref.columns).astype(np.float32)
    feats["ret_min"] = pd.DataFrame(np.nanmin(ret_stack, axis=2), index=_ret_ref.index, columns=_ret_ref.columns).astype(np.float32)
    del ret_stack
    
    # Vectorized aggregate statistics across multiple volatility windows
    _rv_ref = feats["rv_2h"]
    rv_stack = np.stack([
        feats["rv_2h"].to_numpy(),
        feats["rv_4h"].to_numpy(),
        feats["rv_6h"].to_numpy(),
        feats["rv_8h"].to_numpy(),
        feats["rv_12h"].to_numpy(),
        feats["rv_24h"].to_numpy()
    ], axis=2)
    feats["rv_mean"] = pd.DataFrame(np.nanmean(rv_stack, axis=2), index=_rv_ref.index, columns=_rv_ref.columns).astype(np.float32)
    feats["rv_max"] = pd.DataFrame(np.nanmax(rv_stack, axis=2), index=_rv_ref.index, columns=_rv_ref.columns).astype(np.float32)
    feats["rv_min"] = pd.DataFrame(np.nanmin(rv_stack, axis=2), index=_rv_ref.index, columns=_rv_ref.columns).astype(np.float32)
    del rv_stack
    
    # --- Tail-Risk Features (Report 2026-02-10) ---
    # Use optimized Numba quantile kernel for rolling percentiles
    # ret_pct5_24h: 5th percentile of returns over 24h rolling window
    # ret_pct95_24h: 95th percentile of returns over 24h rolling window
    # Note: feats["ret1h"] is a DataFrame (rows × symbols), pass 2D directly
    _ret1h_ref = feats["ret1h"]
    ret_1h_arr = _ret1h_ref.to_numpy(dtype=np.float32)  # (T, S) 2D
    ret_pct5, ret_pct95 = ff._numba_rolling_quantile_dual_parallel(
        ret_1h_arr, 24, 0.05, 0.95
    )
    feats["ret_pct5_24h"] = pd.DataFrame(ret_pct5, index=_ret1h_ref.index, columns=_ret1h_ref.columns).shift(1).astype(np.float32)
    feats["ret_pct95_24h"] = pd.DataFrame(ret_pct95, index=_ret1h_ref.index, columns=_ret1h_ref.columns).shift(1).astype(np.float32)
    
    # gap_zscore: Overnight gap z-score relative to recent gaps
    # Vectorized gap calculation with Numba rolling stats
    gap_df = (o - c.shift(1))
    gap = gap_df.to_numpy(dtype=np.float32)  # (T, S) 2D
    gap_mean = ff._numba_rolling_mean_parallel(gap, 24)
    gap_std = ff._numba_rolling_std_parallel(gap, 24)
    gap_std = np.maximum(gap_std, 1e-12)
    # Shift for causality (roll along axis=0)
    gap_mean_shifted = np.roll(gap_mean, 1, axis=0)
    gap_std_shifted = np.roll(gap_std, 1, axis=0)
    gap_mean_shifted[0, :] = np.nan
    gap_std_shifted[0, :] = np.nan
    feats["gap_zscore"] = pd.DataFrame(
        np.nan_to_num((gap - gap_mean_shifted) / gap_std_shifted, nan=0.0),
        index=_ret1h_ref.index, columns=_ret1h_ref.columns
    ).astype(np.float32)
    
    # vol_shock_z: Volatility shock z-score (rv spike detection)
    _rv6_ref = feats["rv_6h"]
    rv_6_arr = _rv6_ref.to_numpy(dtype=np.float32)  # (T, S) 2D
    rv_6_mean = ff._numba_rolling_mean_parallel(rv_6_arr, 24)
    rv_6_std = ff._numba_rolling_std_parallel(rv_6_arr, 24)
    rv_6_std = np.maximum(rv_6_std, 1e-12)
    rv_6_mean_shifted = np.roll(rv_6_mean, 1, axis=0)
    rv_6_std_shifted = np.roll(rv_6_std, 1, axis=0)
    rv_6_mean_shifted[0, :] = np.nan
    rv_6_std_shifted[0, :] = np.nan
    feats["vol_shock_z"] = pd.DataFrame(
        np.nan_to_num((rv_6_arr - rv_6_mean_shifted) / rv_6_std_shifted, nan=0.0),
        index=_rv6_ref.index, columns=_rv6_ref.columns
    ).astype(np.float32)
    
    # range_zscore: Range (high-low) z-score
    range_hl_df = (h - l)
    range_hl_arr = range_hl_df.to_numpy(dtype=np.float32)  # (T, S) 2D
    range_mean = ff._numba_rolling_mean_parallel(range_hl_arr, 24)
    range_std = ff._numba_rolling_std_parallel(range_hl_arr, 24)
    range_std = np.maximum(range_std, 1e-12)
    range_mean_shifted = np.roll(range_mean, 1, axis=0)
    range_std_shifted = np.roll(range_std, 1, axis=0)
    range_mean_shifted[0, :] = np.nan
    range_std_shifted[0, :] = np.nan
    feats["range_zscore"] = pd.DataFrame(
        np.nan_to_num((range_hl_arr - range_mean_shifted) / range_std_shifted, nan=0.0),
        index=range_hl_df.index, columns=range_hl_df.columns
    ).astype(np.float32)
    
    # tail_risk_score: Combined tail risk metric (vectorized)
    # High when: negative tail returns, high vol shock, large gaps
    ret_pct5_arr = feats["ret_pct5_24h"].to_numpy()
    vol_shock_arr = feats["vol_shock_z"].to_numpy()
    gap_zscore_arr = feats["gap_zscore"].to_numpy()
    feats["tail_risk_score"] = pd.DataFrame(
        np.clip(-ret_pct5_arr, 0, None) * 0.4 +  # Negative tail returns
        np.clip(vol_shock_arr, 0, None) * 0.3 +   # Vol spikes
        np.abs(gap_zscore_arr) * 0.3,              # Large gaps
        index=_ret1h_ref.index, columns=_ret1h_ref.columns
    ).astype(np.float32)

    feats["excess_6h"] = (feats["ret1h"].abs() / rv6).astype(np.float32)
    feats["excess_12h"] = (feats["ret1h"].abs() / rv12).astype(np.float32)

    for k in [2, 4, 8]:
        feats[f"ft_{k}"] = (feats[f"ret{k}h"] / (feats["ret1h"].abs() + 1e-12)).astype(np.float32)
        feats[f"failure_{k}"] = (-1 * feats[f"ft_{k}"]).clip(lower=0).astype(np.float32)

    # clv: (2c - h - l) / (h - l). h-l can be 0.
    clv_raw = ((2 * c - h - l) / ((h - l) + 1e-9)).fillna(0)
    feats["clv"] = clv_raw.astype(np.float32)
    feats["clv_mean_2"] = ff.numba_rolling_mean(feats["clv"], 2).fillna(0).astype(np.float32)
    feats["clv_mean_4"] = ff.numba_rolling_mean(feats["clv"], 4).fillna(0).astype(np.float32)

    for k in [3, 6]:
        v_sum = ff.numba_rolling_sum(v, k)
        ret_k_abs = feats[f"ret{k if k in [6] else 1}h"].abs()
        if k == 3:
            ret_k_abs = ff.numba_rolling_sum(c, 3).abs()

        feats[f"evr_{k}"] = (v_sum / (ret_k_abs + 1e-12)).astype(np.float32)

    feats["progress"] = (feats["ret1h"].abs() / (v + 1e-12)).astype(np.float32)
    feats["speed"] = (feats["ret1h"].abs() / atr).astype(np.float32)

    tail_denom = feats["up_vol_6"] + feats["dn_vol_6"] + 1e-12
    tail_ratio = feats["dn_vol_6"] / tail_denom
    tail_ratio = tail_ratio.where(dir_s > 0, feats["up_vol_6"] / tail_denom)
    feats["tail_against"] = tail_ratio.astype(np.float32)

    feats["asym_ratio"] = (feats["vol_asym_6"] / tail_denom).astype(np.float32)

    o_entry = o.shift(3)
    h_max_4 = ff.numba_rolling_max(h, 4)
    l_min_4 = ff.numba_rolling_min(l, 4)

    mfe_long = h_max_4 - o_entry
    mae_long = o_entry - l_min_4

    mfe = mfe_long.where(dir_s > 0, o_entry - l_min_4)
    mae = mae_long.where(dir_s > 0, h_max_4 - o_entry)

    feats["mfe_4h"] = (mfe / atr).shift(1).astype(np.float32)
    feats["mae_4h"] = (mae / atr).shift(1).astype(np.float32)

    # 8h MFE/MAE
    o_entry_8 = o.shift(7)
    h_max_8 = ff.numba_rolling_max(h, 8)
    l_min_8 = ff.numba_rolling_min(l, 8)

    mfe_long_8 = h_max_8 - o_entry_8
    mae_long_8 = o_entry_8 - l_min_8

    mfe_8 = mfe_long_8.where(dir_s > 0, o_entry_8 - l_min_8)
    mae_8 = mae_long_8.where(dir_s > 0, h_max_8 - o_entry_8)

    feats["mfe_8h"] = (mfe_8 / atr).shift(1).astype(np.float32)
    feats["mae_8h"] = (mae_8 / atr).shift(1).astype(np.float32)

    cur_pnl = (c - o_entry) * dir_s
    gb = (mfe - cur_pnl) / (mfe + 1e-12)
    feats["giveback"] = gb.clip(0, 1).shift(1).astype(np.float32)

    # 2h MFE/MAE and directional path-risk block
    o_entry_2 = o.shift(1)
    h_max_2 = ff.numba_rolling_max(h, 2)
    l_min_2 = ff.numba_rolling_min(l, 2)

    mfe_long_2 = h_max_2 - o_entry_2
    mae_long_2 = o_entry_2 - l_min_2

    mfe_2 = mfe_long_2.where(dir_s > 0, o_entry_2 - l_min_2)
    mae_2 = mae_long_2.where(dir_s > 0, h_max_2 - o_entry_2)

    feats["mfe_2h"] = (mfe_2 / atr).shift(1).astype(np.float32)
    feats["mae_2h"] = (mae_2 / atr).shift(1).astype(np.float32)

    d_long = (h_max_2 - o_entry_2) / atr
    d_short = (o_entry_2 - l_min_2) / atr
    risk_long = (mae_long_2 / (mfe_long_2 + 1e-12)).clip(0, 10)
    risk_short = ((h_max_2 - o_entry_2) / ((o_entry_2 - l_min_2) + 1e-12)).clip(0, 10)

    feats["dir_path_long_2h"] = d_long.shift(1).astype(np.float32)
    feats["dir_path_short_2h"] = d_short.shift(1).astype(np.float32)
    feats["dir_path_risk_long_2h"] = risk_long.shift(1).astype(np.float32)
    feats["dir_path_risk_short_2h"] = risk_short.shift(1).astype(np.float32)
    feats["dir_path_edge_2h"] = (feats["dir_path_long_2h"] - feats["dir_path_short_2h"]).astype(np.float32)
    feats["dir_path_risk_skew_2h"] = (feats["dir_path_risk_long_2h"] - feats["dir_path_risk_short_2h"]).astype(np.float32)
    del o_entry, h_max_4, l_min_4, mfe_long, mae_long, mfe, mae, cur_pnl, gb

    # --- Memory checkpoint: free GC before composite features ---
    tprint(f"Features: {len(feats)} base features computed. Running GC before composites...")
    gc.collect()

    # --- COMPOSITE / INTERACTION FEATURES ---

    # 1/ Exhaustion
    feats["overext"] = (feats["donch_dist_12"] * feats["excess_6h"]).fillna(0).astype(np.float32)
    feats["overext_weak"] = (feats["donch_dist_12"] * (1.0 - feats["clv_mean_4"].clip(lower=0))).fillna(0).astype(np.float32)
    feats["effort_gate"] = (feats["evr_6"] * (feats["vol_z24"] + 1.0) / (feats["progress"] + 1e-12)).fillna(0).astype(np.float32)
    feats["stall_ext"] = (feats["donch_dist_12"] * (1.0 - feats["delta_stall_6"])).fillna(0).astype(np.float32)
    feats["tail_fail"] = (feats["tail_against"] * (feats["ft_2"] - feats["ft_4"]).clip(lower=0)).fillna(0).astype(np.float32)

    pb_avg = (feats["pullback_2"] + feats["pullback_4"]) / 2.0
    fail_term = (feats["failure_2"] + 0.5 * feats["failure_4"])
    feats["reject_score"] = ((1.0 - feats["clv_mean_4"].clip(lower=0)) * pb_avg * fail_term).fillna(0).astype(np.float32)

    feats["impulse_ratio_24"] = (feats["ret1h"].abs() / (feats["ret24h"].abs() + 1e-12)).fillna(0).astype(np.float32)
    feats["impulse_ratio_12"] = (feats["ret1h"].abs() / (feats["ret12h"].abs() + 1e-12)).fillna(0).astype(np.float32)
    feats["accel"] = (feats["ret1h"] - feats["ret1h"].shift(1)).abs() / (feats["rv_6h"] + 1e-12)
    feats["blowoff_risk"] = (feats["impulse_ratio_24"] * feats["accel"] * feats["donch_dist_12"]).fillna(0).astype(np.float32)

    # 2/ Spike Anatomy / Regime
    s_max = feats["ret16h"].abs()
    for k in [20, 24, 28]:
        s_max = np.maximum(s_max, feats[f"ret{k}h"].abs())
    feats["S"] = (dir_s * s_max).astype(np.float32)

    # --- Global Kalman Q/R tuner + Kalman features ---
    # Tune lambda(Q/R) on score monotonicity across deciles with mild turnover penalty.
    # Exclude OOS period from tuning to prevent leakage
    train_n_kalman = max(0, len(c.index) - int(24 * cfg.get("oos_holdout_days", 730)))
    if train_n_kalman > 0:
        feats_S_train = feats["S"].iloc[:train_n_kalman]
        feats_ret1h_train = feats["ret1h"].iloc[:train_n_kalman]
    else:
        feats_S_train = feats["S"]
        feats_ret1h_train = feats["ret1h"]
    kalman_lambda = tune_global_kalman_lambda(feats_S_train, feats_ret1h_train, grid_size=15)

    score_rm24 = ff.numba_rolling_mean(feats["S"], 24).shift(1).astype(np.float32)
    vol_ratio_input = feats.get("liquidity_ratio", (v / (ff.numba_rolling_mean(v, 24 * 30).shift(1) + EPS)).astype(np.float32))

    kf_score_mean, kf_innov_var, kf_state_unc, r_score = _kalman_local_level_df(feats["S"], kalman_lambda)
    kf_rm24_mean, _, _, _ = _kalman_local_level_df(score_rm24, kalman_lambda)
    kf_atr_mean, _, _, _ = _kalman_local_level_df(feats["atr_pct"], kalman_lambda)
    kf_vol_ratio_mean, _, _, _ = _kalman_local_level_df(vol_ratio_input, kalman_lambda)
    kf_ret1h_mean, _, _, _ = _kalman_local_level_df(feats["ret1h"], kalman_lambda)

    feats["kf_score_mean"] = kf_score_mean.astype(np.float32)
    feats["kf_score_rm24_mean"] = kf_rm24_mean.astype(np.float32)
    feats["kf_atr_mean"] = kf_atr_mean.astype(np.float32)
    feats["kf_vol_ratio_mean"] = kf_vol_ratio_mean.astype(np.float32)
    feats["kf_ret1h_mean"] = kf_ret1h_mean.astype(np.float32)

    # Meta diagnostics: innovation variance, SNR estimate, and state uncertainty.
    feats["kf_innov_var"] = kf_innov_var.astype(np.float32)
    feats["kf_state_uncertainty"] = kf_state_unc.astype(np.float32)
    r_score_df = pd.DataFrame(np.repeat(r_score.values.reshape(1, -1), len(c.index), axis=0), index=c.index, columns=c.columns).astype(np.float32)
    q_score_df = (kalman_lambda * r_score_df).astype(np.float32)
    feats["kf_snr_est"] = (q_score_df / (r_score_df + EPS)).astype(np.float32)

    feats["coherence_24"] = (dir_s * (feats["ret6h"] + feats["ret12h"] + feats["ret24h"]) / (feats["rv_24h"] + 1e-12)).astype(np.float32)

    turb = rv_ratio # Already broadcasted

    mkt_ret6h_raw = mkt_gates["mkt_ret6h"].reindex(c.index).astype(np.float32)
    mkt_ret6h_s = pd.DataFrame(np.repeat(mkt_ret6h_raw.to_numpy()[:,None], c.shape[1], axis=1),
                               index=c.index, columns=c.columns).astype(np.float32)

    tape_align = (dir_s * mkt_ret6h_s)
    feats["tf_tape"] = (tape_align.clip(lower=0) / (1.0 + turb)).astype(np.float32)
    feats["mr_tape"] = ((-tape_align).clip(lower=0) / (1.0 + turb)).astype(np.float32)

    feats["tf_minus_mr"] = (feats["tf_tape"] - feats["mr_tape"]).astype(np.float32)
    feats["body_ratio"] = feats["efficiency"]

    # Define vars explicitly used in gates and other features
    ft2_pos = feats["ft_2"].clip(lower=0)
    ft4_pos = feats["ft_4"].clip(lower=0)
    clv4_pos = feats["clv_mean_4"].clip(lower=0)
    pb2_mag = feats["pullback_2"].abs().clip(0, 1)
    pb2_inv = (1.0 - pb2_mag)
    pb4_mag = feats["pullback_4"].abs().clip(0, 1)
    pb4_inv = (1.0 - pb4_mag)

    fail_sum = (feats["failure_2"] + feats["failure_4"])
    clv_inv = (1.0 - feats["clv_mean_4"])
    pb_avg_abs = (feats["pullback_2"].abs() + feats["pullback_4"].abs()) / 2.0
    ret_rat = (feats["ret4h"].abs() / (feats["ret1h"].abs() + 1e-12))

    # 3/ TF vs MR
    feats["accept_score"] = (ft2_pos * clv4_pos * pb2_inv).astype(np.float32)
    feats["retest_accept_score"] = (ft4_pos * clv4_pos * pb4_inv).astype(np.float32)

    feats["tf_qual_score"] = (feats["accept_score"] * feats["tf_tape"]).astype(np.float32)

    feats["mr_qual_score"] = (feats["reject_score"] * feats["mr_tape"]).astype(np.float32)
    feats["retrace_12"] = (-feats["pullback_12"]).astype(np.float32)

    # --- Gate Generation & Selection (Updated 2026-02-10) ---
    from .gated_features import add_gate_features_panel, select_gated_features

    # Gate windows: 16 bars = 4 hours, 24 bars = 6 hours at 15m timeframe
    # These capture intraday patterns without excessive lag
    gate_window = int(cfg.get("accept_gate_window", 24))
    gate_windows = sorted(set([16, gate_window]))
    percentile_mode = cfg.get("accept_gate_percentile_mode", "approx")

    # Define Gate Sources (Panel Data directly from feats)
    # Mapping: Source Name -> (Panel Data, Output Prefix)
    # Note: accept_score maps to prefix 's' for legacy reasons
    gate_configs = {
        "accept_score":        (feats["accept_score"], "s"),
        "reject_score":        (feats["reject_score"], "reject"),
        "retest_accept_score": (feats["retest_accept_score"], "retest_accept"),
        "tf_qual_score":       (feats["tf_qual_score"], "tf_qual"),
        "mr_qual_score":       (feats["mr_qual_score"], "mr_qual"),
        "vol_z":               (feats["vol_z"], "vol_z"),
        # Liquidity Score: Higher is better (more liquid). Amihud is Illiq (lower is better).
        "liquidity_score":     (-feats["amihud_z"], "liquidity"),
    }

    tprint(f"Generating Gated Features for windows {gate_windows} with selection...")

    # Skill metric: Monthly time blocks for robust evaluation
    periods = c.index.to_period("M")
    unique_periods = periods.unique()
    time_blocks = [(periods == p) for p in unique_periods]
    # Train mask: Exclude OOS holdout period to prevent leakage
    train_mask_proxy = pd.Series(True, index=c.index)
    oos_holdout_hours = int(24 * cfg.get("oos_holdout_days", 730))

    # Always drop at least 8 hours for the forward target shift buffer
    # If the holdout period is larger, drop that instead, but bound by array length
    if len(train_mask_proxy) > 8:
        train_mask_proxy.iloc[-8:] = False

    if oos_holdout_hours > 8 and len(train_mask_proxy) > oos_holdout_hours:
        train_mask_proxy.iloc[-oos_holdout_hours:] = False

    for w in gate_windows:
        for source_name, (source_panel, prefix) in gate_configs.items():
            # 1. Generate ALL candidates for this family (mean, std, z, pct, bin3, gt25..gt90)
            # Returns dict: feature_name -> Panel DataFrame
            family_features = add_gate_features_panel(
                source_panel,
                prefix=prefix,
                n=w,
                add_strict=True,
                percentile_mode=percentile_mode
            )

            # 2. Extract BASE features (Always keep mean, std, z, pct, bin3)
            base_suffixes = ["mean", "std", "z", "pct", "bin3"]
            for suffix in base_suffixes:
                feat_name = f"{prefix}_{suffix}_{w}"
                if feat_name in family_features:
                    feats[feat_name] = family_features[feat_name]

            # 3. SELECT best threshold features (from gt25, gt50, ..., gt90)
            # Construct mini-table for selection function
            # Only include the 'gt' threshold candidates
            candidates_table = {k: v for k, v in family_features.items() if "_gt" in k}
            
            # If no candidates produced, skip selection
            if not candidates_table:
                continue

            # Run selection: Selects globally best thresholds based on prevalence/skill
            selected_names = select_gated_features(
                gate_feature_table=candidates_table,
                families=[(prefix, w)],
                target=target_proxy,
                time_blocks=time_blocks,
                train_mask=train_mask_proxy
            )

            # 4. Store SELECTED features
            for name in selected_names:
                if name in candidates_table:
                    feats[name] = candidates_table[name]
            
            # Explicitly clear intermediate dict to free memory
            del family_features
            del candidates_table
            # import gc; gc.collect() # Optional frequent GC

    # Re-bind standardized names for downstream dependencies
    # These rely on the standard `gate_window` (e.g. 64) features being present
    # Warning: If `select_gated_features` didn't select gt66/gt85, these might fall back or error?
    # Actually, `select_gated_features` has fallback logic to ensure *some* gates are selected.
    # But `s_gt66` specifically is used below.
    # We should ensure s_gt66_64 exists if needed, or update this logic to use selected gates.
    
    # Safe getters since selection is dynamic
    def get_feat(name, fallback_zeros=True):
        if name in feats:
            return feats[name]
        if fallback_zeros:
            return pd.DataFrame(0, index=c.index, columns=c.columns, dtype=np.float32)
        raise KeyError(name)

    s_pct = get_feat(f"s_pct_{gate_window}")
    s_bin3 = get_feat(f"s_bin3_{gate_window}")
    
    # Dynamic selection might explicitly select gt66/gt85 or might select gt50/gt90.
    # For backward compatibility variables, we ideally want specific thresholds if they exist,
    # or the "best" available proxy?
    # Let's check what was selected for 's' (accept_score) at gate_window.
    # If gt66 not selected, try to find nearest? Or just use zeros?
    # User code implies selection is for "feature table".
    # But `feats["accept_gt66"]` might be used by Meta model expecting exactly that?
    # If Meta model is retrained, it will use whatever is available.
    # But hardcoded `accept_gt66` reference suggests we might want to force potential "standard" gates into feats?
    # Compromise: `select_gated_features` picks the *best*.
    # If we need specific ones for legacy logic, we might need to update legacy logic.
    # For now, let's map `accept_gt66` to `s_gt66_{w}` ONLY if it exists.
    
    feats["accept"] = s_pct
    feats["accept_bin3"] = s_bin3.astype(np.float32)

    # reject_like: reject gate percentile (MR counterpart to accept)
    reject_like = get_feat(f"reject_pct_{gate_window}")
    
    # Map strict gates if they exist
    if f"s_gt66_{gate_window}" in feats:
        feats["accept_gt66"] = feats[f"s_gt66_{gate_window}"]
        feats["retest_accept"] = feats[f"s_gt66_{gate_window}"] # Legacy alias
    else:
        # Fallback to whatever was selected as "broad" or "rare"?
        pass

    if f"s_gt85_{gate_window}" in feats:
        feats["accept_gt85"] = feats[f"s_gt85_{gate_window}"]
    else:
        # Keep stable key availability even when dynamic gate selection skips gt85.
        feats["accept_gt85"] = (s_pct >= 0.85).astype(np.float32)

    feats["tf_qual"] = (s_pct * feats["tf_tape"]).astype(np.float32)
    feats["mr_qual"] = (reject_like * feats["mr_tape"]).astype(np.float32)

    # Gate interactions with directional 2h path-risk block
    dir_edge = feats.get("dir_path_edge_2h", pd.DataFrame(0, index=c.index, columns=c.columns, dtype=np.float32))
    feats["accept_x_dir_edge_2h"] = (s_pct * dir_edge).astype(np.float32)
    feats["reject_x_dir_edge_2h"] = (reject_like * dir_edge).astype(np.float32)
    feats["tfq_x_dir_edge_2h"] = (feats["tf_qual"] * dir_edge).astype(np.float32)
    feats["mrq_x_dir_edge_2h"] = (feats["mr_qual"] * dir_edge).astype(np.float32)

    gate_interactions = {}
    gate_interactions.update(add_gate_interaction_panel(s_pct, dir_edge, prefix="accept_dir2h"))
    gate_interactions.update(add_gate_interaction_panel(reject_like, dir_edge, prefix="reject_dir2h"))
    gate_interactions.update(add_gate_interaction_panel(feats["tf_qual"], dir_edge, prefix="tfq_dir2h"))
    gate_interactions.update(add_gate_interaction_panel(feats["mr_qual"], dir_edge, prefix="mrq_dir2h"))
    for gk, gv in gate_interactions.items():
        feats[gk] = gv.astype(np.float32)

    # 4/ Meta
    feats["rv_ratio_6_24"] = (feats["rv_6h"] / (feats["rv_24h"] + 1e-12)).astype(np.float32)

    # Define gates helpers for Meta

    feats["G_EXH_EFFORT"] = (feats["evr_6"] * (feats["vol_z24"] + 1.0) / (feats["progress"] + 1e-12)).fillna(0).astype(np.float32)
    feats["G_EXH_GIVEBACK"] = (feats["giveback"] * (1.0 + feats["donch_dist_12"])).fillna(0).astype(np.float32)
    feats["G_EXH_TAIL_FAIL"] = (feats["tail_against"] * (feats["ft_2"] - feats["ft_4"]).clip(lower=0)).fillna(0).astype(np.float32)

    feats["G_MR_SPIKE"] = (feats["speed"] * feats["excess_6h"] * clv_inv).fillna(0).astype(np.float32)
    feats["G_TF_GRIND"] = (ret_rat * feats["clv_mean_4"] * pb2_inv).astype(np.float32)
    feats["G_TF_TREND"] = (feats["speed"] * feats["coherence_24"] * clv4_pos).fillna(0).astype(np.float32)
    feats["G_MR_TAIL"] = (feats["tail_against"] * (1.0 + feats["donch_dist_6"])).astype(np.float32)

    # Meta Features using Gates
    ambig_term = (1.0 - np.maximum(feats["accept"], reject_like))
    feats["ambig"] = (ambig_term * feats["rv_ratio_6_24"]).astype(np.float32)

    feats["stage_tf"] = (feats["accept"] * feats["coherence_24"]).astype(np.float32)
    feats["stage_blowoff"] = (feats["blowoff_risk"] + feats["effort_gate"] + feats["stall_ext"]).astype(np.float32)
    feats["stage_mr"] = (reject_like * (1.0 + feats["overext"])).astype(np.float32)
    feats["exh_qual"] = (feats["effort_gate"] + feats["stall_ext"] + feats["tail_fail"] + feats["overext_weak"]).astype(np.float32)

    feats["thrust_decay_4"] = (feats["ret1h"].abs() / (feats["ret4h"].abs() + 1e-12)).astype(np.float32)
    feats["decel_4"] = (feats["momentum_accel"].abs() / rv6).astype(np.float32)
    feats["ft_drop"] = (feats["ft_2"] - feats["ft_4"]).astype(np.float32)

    feats["thrust_decay_8"] = (feats["ret1h"].abs() / (feats["ret8h"].abs() + 1e-12)).astype(np.float32)
    feats["decel_8"] = (feats["momentum_accel"].abs() / rv12).astype(np.float32)
    feats["ft_drop_8"] = (feats["ft_4"] - feats["ft_8"]).astype(np.float32)
    feats["ext_excess"] = (feats["donch_dist_12"] * feats["excess_6h"]).astype(np.float32)
    feats["ext_atrExp"] = (feats["donch_dist_12"] * np.log(feats["atr_expansion"] + 1e-12)).astype(np.float32)
    feats["comp_to_exp"] = ((1.0 / (feats["vol_compression"] + 1e-12)) * feats["atr_expansion"]).astype(np.float32)
    feats["evr6_x_volz"] = (feats["evr_6"] * (feats["vol_z24"] + 1.0)).astype(np.float32)
    feats["stall_x_flow"] = (feats["delta_stall_6"] * feats["flow_persistence"]).astype(np.float32)
    feats["prog_def"] = (feats["excess_6h"] / (feats["progress"] + 1e-12)).astype(np.float32)
    feats["clv_collapse"] = (feats["clv_mean_2"] - feats["clv_mean_4"]).astype(np.float32)
    feats["clv_pullback"] = ((1.0 - feats["clv_mean_4"]) * feats["pullback_4"].abs()).astype(np.float32)
    feats["coh"] = (dir_s * (feats["ret1h"] + feats["ret2h"] + feats["ret4h"])) / rv6
    feats["align"] = (dir_s * np.sign(feats["slope"])).astype(np.float32)
    feats["retest_quality"] = ((1.0 - feats["pullback_2"].abs()) * feats["clv_mean_2"]).astype(np.float32)
    feats["pb_accel"] = ((feats["pullback_2"] - feats["pullback_4"]) / atr).astype(np.float32)
    feats["excess_coh"] = (feats["excess_6h"] * feats["coh"]).astype(np.float32)
    feats["asym_ft"] = (feats["ft_2"] * feats["asym_ratio"] * dir_s).astype(np.float32)
    feats["dist_stack"] = (feats["dist_ema_fast"] + feats["dist_vwap_norm"] + feats["trend_pct"]).astype(np.float32)
    feats["tf_bias"] = (feats["coh"] * (1.0 / (1.0 + feats["donch_dist_12"]))).astype(np.float32)
    feats["shock_rel"] = feats["excess_6h"]
    feats["resid_strength"] = feats["excess_6h"]
    feats["evr_slope"] = (feats["evr_3"] - feats["evr_6"]).astype(np.float32)
    
    # Base components for interactions
    ema_6 = ema(c, 6)
    ema_24 = ema(c, 24)
    feats["trend_t"] = ema_6.diff(1).astype(np.float32)

    # Volatility Interaction Context (New)
    feats["dist_ext_x_vol"] = (feats["donch_dist_12"] * feats["vol_z"]).fillna(0).astype(np.float32)
    feats["regime_x_vol"] = (feats["rv_ratio_6_24"] * feats["vol_z"]).fillna(0).astype(np.float32)
    feats["rsi_x_vol"] = ((feats["rsi"] - 50.0) * feats["vol_z"]).fillna(0).astype(np.float32)
    feats["vol_z_x_trend_t"] = (feats["vol_z"] * feats["trend_t"]).fillna(0).astype(np.float32)

    feats["stall_ext_corr"] = (feats["delta_stall_6"] * feats["donch_dist_12"]).astype(np.float32)

    feats["G_META_EXH"] = (feats["overext"] + feats["G_EXH_EFFORT"] + feats["stall_ext"] + feats["G_EXH_GIVEBACK"]).astype(np.float32)
    feats["G_META_TF_QUAL"] = (feats["accept"] * (1.0 - feats["G_META_EXH"].clip(0,1))).astype(np.float32)
    feats["G_META_MR_QUAL"] = (reject_like * (1.0 - feats["overext"].clip(0,1))).astype(np.float32)
    feats["G_META_AMBIG"] = (ambig_term * feats["rv_ratio_6_24"]).astype(np.float32)

    ret_w = feats["ret10h"]
    local_low = ff.numba_rolling_min(l, 10)
    local_high = ff.numba_rolling_max(h, 10)
    draw_num = np.where((ret_w > 0).to_numpy(), (c - local_low).to_numpy(), (c - local_high).to_numpy())
    # Use safe division with proper handling of non-finite values
    draw_sym = np.where(
        np.isfinite(draw_num) & np.isfinite(c) & (c.abs() > 1e-12),
        np.sign(ret_w) * draw_num / (c + 1e-12),
        0.0
    )
    feats["draw_sym_10h"] = draw_sym.astype(np.float32)
    feats["draw_extreme_10h"] = np.abs(draw_sym).astype(np.float32)

    hi_24_prev = ff.numba_rolling_max(h.shift(1), 24)
    lo_24_prev = ff.numba_rolling_min(l.shift(1), 24)
    up_break = c - hi_24_prev
    dn_break = c - lo_24_prev
    choose_up = (np.abs(up_break) >= np.abs(dn_break))
    # Use safe division with proper handling of non-finite values
    breakout_raw = np.where(choose_up, up_break, dn_break).astype(np.float32)
    feats["breakout_24h"] = np.where(
        np.isfinite(breakout_raw) & np.isfinite(c) & (np.abs(c) > 1e-12),
        breakout_raw / (c + 1e-12),
        0.0
    ).astype(np.float32)

    abs_net_score = feats["accept"] + reject_like
    feats["meta_abs_net_x_breakout"] = (abs_net_score * np.abs(feats["breakout_24h"])).astype(np.float32)
    feats["meta_abs_net_x_drawext"] = (abs_net_score * np.abs(feats["draw_extreme_10h"])).astype(np.float32)
    feats["meta_abs_net_x_vov_ratio"] = (abs_net_score * (feats["vov_ratio"] - 1.0).clip(lower=0)).astype(np.float32)
    # Safe meta_alignment computation
    accept_diff = feats["accept"] - reject_like
    ret5h_safe = np.where(np.isfinite(feats["ret5h"]), feats["ret5h"], 0.0)
    feats["meta_alignment"] = (np.sign(accept_diff) * np.sign(ret5h_safe)).astype(np.float32)
    feats["meta_signal_x_accel"] = ((feats["accept"] - reject_like) * feats["accel_5h"]).astype(np.float32)

    # Regime interactions using base-model agreement-weighted success signal.
    base_agreement = (1.0 - np.abs(feats["accept"] - reject_like)).clip(0.0, 1.0)
    p_success_df = (((feats["accept"] + reject_like) * 0.5) * base_agreement).astype(np.float32)
    vol_high = feats.get("vol_high", pd.DataFrame(0.0, index=c.index, columns=c.columns, dtype=np.float32))
    cusum_high = feats.get("cusum_high", pd.DataFrame(0.0, index=c.index, columns=c.columns, dtype=np.float32))
    liq_low = feats.get("liq_low", pd.DataFrame(0.0, index=c.index, columns=c.columns, dtype=np.float32))
    interaction_dict = add_interactions(p_success_df, vol_high, cusum_high, liq_low)
    for ik, iv in interaction_dict.items():
        feats[ik] = iv.astype(np.float32)

    # Robust Score Calculation with clipping to prevent Inf/Overflow
    # We clip components to avoid exploding values when denominators are near zero
    feats["spike_score"] = (feats["speed"].clip(0, 100) * feats["excess_6h"].clip(0, 100)).fillna(0).astype(np.float32)
    feats["grind_score"] = (ret_rat.clip(0, 100) * feats["clv_mean_4"]).fillna(0).astype(np.float32)
    coh_norm = feats["coh"].clip(0,1).fillna(0)
    feats["chop_score"] = (feats["rv_ratio_6_24"].clip(0, 100) * (1.0 - coh_norm)).fillna(0).astype(np.float32)

    # =====================================================================
    # ORTHOGONAL FEATURES — structurally independent from existing clusters
    # =====================================================================

    # --- Cross-asset features (temporarily disabled) ---
    # feats["xs_rank_ret6h"] = feats["ret6h"].rank(axis=1, pct=True).astype(np.float32)
    # feats["xs_rank_vol_z"] = feats["vol_z"].rank(axis=1, pct=True).astype(np.float32)
    # feats["xs_rank_rv24"] = feats["rv_24h"].rank(axis=1, pct=True).astype(np.float32)
    # feats["beta_24h"] = ...
    # feats["resid_ret_6h"] = ...

    # 1. Multi-timeframe momentum divergence: short vs long disagreement
    #    Sign disagreement between 2h and 24h returns — captures regime transitions
    sign_2h = np.sign(np.where(np.isfinite(feats["ret2h"]), feats["ret2h"], 0.0))
    sign_24h = np.sign(np.where(np.isfinite(feats["ret24h"]), feats["ret24h"], 0.0))
    feats["mtf_divergence"] = (sign_2h * sign_24h * -1.0).astype(np.float32)  # +1 = diverging
    #    Magnitude-weighted divergence
    feats["mtf_div_mag"] = ((feats["ret2h"] - feats["ret24h"] / 12.0) / (feats["rv_6h"] + 1e-12)).clip(-10, 10).astype(np.float32)

    # 2. Mean-reversion speed proxy: rolling autocorrelation of returns
    #    Negative autocorr = fast mean-reversion, positive = trending
    feats["autocorr_6h"] = ff.numba_rolling_corr(
        feats["ret1h"], feats["ret1h"].shift(1), 6
    ).fillna(0).astype(np.float32)
    feats["autocorr_24h"] = ff.numba_rolling_corr(
        feats["ret1h"], feats["ret1h"].shift(1), 24
    ).fillna(0).astype(np.float32)

    # 3. Price path entropy proxy: ratio of actual path length to displacement
    #    High = choppy/random, Low = directional/clean
    abs_ret_sum_12 = ff.numba_rolling_sum(feats["ret1h"].abs(), 12)
    displacement_12 = feats["ret12h"].abs()
    feats["path_efficiency_12"] = (displacement_12 / (abs_ret_sum_12 + 1e-12)).clip(0, 1).astype(np.float32)
    abs_ret_sum_24 = ff.numba_rolling_sum(feats["ret1h"].abs(), 24)
    displacement_24 = feats["ret24h"].abs()
    feats["path_efficiency_24"] = (displacement_24 / (abs_ret_sum_24 + 1e-12)).clip(0, 1).astype(np.float32)

    # 6. Hurst exponent proxy: R/S ratio over rolling window
    #    H > 0.5 = trending, H < 0.5 = mean-reverting
    range_24 = ff.numba_rolling_max(c, 24) - ff.numba_rolling_min(c, 24)
    std_24 = ff.numba_rolling_std(feats["ret1h"], 24)
    feats["hurst_proxy_24"] = (np.log(range_24 / (std_24 * np.sqrt(24) + 1e-12) + 1e-12) / np.log(24)).clip(0, 1).fillna(0.5).astype(np.float32)

    # 7. Volume concentration: rolling Gini-like measure (max_vol / sum_vol over 12h)
    #    High = volume clustered in few bars, Low = evenly distributed
    v_max_12 = ff.numba_rolling_max(v, 12)
    v_sum_12 = ff.numba_rolling_sum(v, 12)
    feats["vol_concentration_12"] = (v_max_12 / (v_sum_12 + 1e-12)).astype(np.float32)

    # 4. Signed volume divergence: volume trend vs price trend disagreement
    vol_trend = ff.numba_rolling_sum(v, 6) - ff.numba_rolling_sum(v, 24) / 4.0
    price_trend = np.where(np.isfinite(feats["ret6h"]), feats["ret6h"], 0.0)
    feats["vol_price_diverge"] = (np.sign(vol_trend) * np.sign(price_trend) * -1.0).astype(np.float32)

    # 5. Alpha asymmetry-volatility features (MR/TF, long/short)
    neg_ret = feats["ret1h"].clip(upper=0)
    pos_ret = feats["ret1h"].clip(lower=0)
    neg_sq = neg_ret * neg_ret
    pos_sq = pos_ret * pos_ret

    # Downside / Upside semivariance
    feats["downside_semivariance_8"] = ff.apply_to_frame(neg_sq, ff._numba_rolling_mean_nan_safe, 8).astype(np.float32)
    feats["downside_semivariance_24"] = ff.apply_to_frame(neg_sq, ff._numba_rolling_mean_nan_safe, 24).astype(np.float32)
    feats["upside_semivariance_8"] = ff.apply_to_frame(pos_sq, ff._numba_rolling_mean_nan_safe, 8).astype(np.float32)
    feats["upside_semivariance_24"] = ff.apply_to_frame(pos_sq, ff._numba_rolling_mean_nan_safe, 24).astype(np.float32)

    # Downside / Upside volatility ratio (std ratio, not variance ratio)
    down_vol_8 = np.sqrt(feats["downside_semivariance_8"].clip(lower=0))
    up_vol_8 = np.sqrt(feats["upside_semivariance_8"].clip(lower=0))
    down_vol_24 = np.sqrt(feats["downside_semivariance_24"].clip(lower=0))
    up_vol_24 = np.sqrt(feats["upside_semivariance_24"].clip(lower=0))
    feats["down_up_vol_ratio_8"] = (down_vol_8 / (up_vol_8 + 1e-12)).astype(np.float32)
    feats["down_up_vol_ratio_24"] = (down_vol_24 / (up_vol_24 + 1e-12)).astype(np.float32)

    # Volatility shock asymmetry
    feats["vol_shock_asym_8_24"] = (feats["rv_8h"] - feats["rv_24h"]).astype(np.float32)
    feats["vol_shock_asym_4_12"] = (feats["rv_4h"] - feats["rv_12h"]).astype(np.float32)
    # Backward-compatible alias for requested notation "σ4 - σ212" (interpreted as 4 vs 12)
    feats["vol_shock_asym_4_212"] = feats["vol_shock_asym_4_12"].astype(np.float32)

    # 6. Alpha entropy features (MR/TF, long/short)
    # Shannon entropy of returns
    feats["shannon_entropy_ret_8"] = _rolling_shannon_entropy_df(feats["ret1h"], window=8, bins=8)
    feats["shannon_entropy_ret_16"] = _rolling_shannon_entropy_df(feats["ret1h"], window=16, bins=12)

    # Permutation entropy of returns
    feats["perm_entropy_ret_12"] = _rolling_permutation_entropy_df(feats["ret1h"], window=12, order=3, delay=1)
    feats["perm_entropy_ret_24"] = _rolling_permutation_entropy_df(feats["ret1h"], window=24, order=3, delay=1)

    # Spectral entropy of returns
    feats["spectral_entropy_ret_24"] = _rolling_spectral_entropy_df(feats["ret1h"], window=24)
    feats["spectral_entropy_ret_48"] = _rolling_spectral_entropy_df(feats["ret1h"], window=48)

    # Volume entropy
    feats["volume_entropy_12"] = _rolling_shannon_entropy_df(v, window=12, bins=10)
    feats["volume_entropy_24"] = _rolling_shannon_entropy_df(v, window=24, bins=12)

    # =====================================================================
    # RESIDUALISED FEATURES — relative surprise, not absolute magnitude
    # =====================================================================
    # Rationale: low-conviction trades outperform high-conviction ones,
    # meaning relative surprise matters, not absolute score.

    # (a) Z-scored surprise signals: s_z = (s_t - rolling_mean) / rolling_std
    #     Window = 48h (~2x max hold) to capture "unusual for recent regime"
    RESID_WINDOW = 48

    for feat_name in ["rsi", "dist_ema_fast", "dist_vwap_norm", "flow_persistence",
                      "excess_6h", "vol_z", "atr_expansion", "coherence_24"]:
        if feat_name in feats:
            raw = feats[feat_name]
            roll_mu = ff.numba_rolling_mean(raw, RESID_WINDOW)
            roll_sd = ff.numba_rolling_std(raw, RESID_WINDOW)
            feats[f"{feat_name}_z"] = ((raw - roll_mu) / (roll_sd + 1e-12)).clip(-5, 5).fillna(0).astype(np.float32)

    # (b) Rolling edge residual: how much is the model's current signal
    #     deviating from its recent realised performance?
    #     Proxy: z-score of composite scores (accept, reject, overext)
    for comp_name in ["accept", "overext", "blowoff_risk", "exh_qual"]:
        if comp_name in feats:
            raw = feats[comp_name]
            roll_mu = ff.numba_rolling_mean(raw, RESID_WINDOW)
            roll_sd = ff.numba_rolling_std(raw, RESID_WINDOW)
            feats[f"{comp_name}_surprise"] = ((raw - roll_mu) / (roll_sd + 1e-12)).clip(-5, 5).fillna(0).astype(np.float32)

    # (c) Residual distance from value vs market trend
    #     dist_resid = dist_to_vwap - k * market_trend_strength
    #     Stops MR entries that are "cheap" only because market is trending hard
    mkt_trend_s = mkt_gates["mkt_trend"].reindex(c.index).astype(np.float32)
    mkt_trend_bc = pd.DataFrame(
        np.repeat(np.asarray(mkt_trend_s)[:, None], c.shape[1], axis=1),
        index=c.index, columns=c.columns
    ).astype(np.float32)
    mkt_rv_s = mkt_gates["mkt_rv"].reindex(c.index).astype(np.float32)
    mkt_rv_bc = pd.DataFrame(
        np.repeat(np.asarray(mkt_rv_s)[:, None], c.shape[1], axis=1),
        index=c.index, columns=c.columns
    ).astype(np.float32)
    # Normalised market trend strength (in vol units)
    mkt_trend_z = mkt_trend_bc / (mkt_rv_bc * np.sqrt(24) + 1e-12)
    # Use safe operations for residual features
    mkt_trend_z_safe = np.where(np.isfinite(mkt_trend_z), mkt_trend_z, 0.0)
    feats["dist_vwap_resid"] = np.where(
        np.isfinite(feats["dist_vwap_norm"]) & np.isfinite(mkt_trend_z_safe),
        (feats["dist_vwap_norm"] - 0.5 * mkt_trend_z_safe),
        0.0
    ).astype(np.float32)
    feats["dist_ema_fast_resid"] = np.where(
        np.isfinite(feats["dist_ema_fast"]) & np.isfinite(mkt_trend_z_safe),
        (feats["dist_ema_fast"] - 0.5 * mkt_trend_z_safe),
        0.0
    ).astype(np.float32)
    feats["trend_pct_resid"] = np.where(
        np.isfinite(feats["trend_pct"]) & np.isfinite(mkt_trend_z_safe),
        (feats["trend_pct"] - 0.5 * mkt_trend_z_safe),
        0.0
    ).astype(np.float32)

    # =====================================================================
    # OHLCV-Based Trend Quality Features (Report 2026-02-12)
    # =====================================================================

    # --- TF Features: Trend Quality & Regime Context ---

    # trend_age_hours: How long has the current trend been in place?
    # Count consecutive bars where price is on same side of EMA
    ema_fast = ema(c, cfg["ema_fast"])
    above_ema = (c > ema_fast).astype(np.float32)
    # Count consecutive bars in same direction using rolling sum of sign changes
    trend_sign = (2 * above_ema - 1).astype(np.float32)  # +1 above, -1 below
    # trend_age: count bars since last sign flip (simplified: use run-length encoding proxy)
    trend_sign_change = (trend_sign != trend_sign.shift(1)).astype(np.float32)
    trend_age_cumsum = trend_sign_change.cumsum()
    # Within each trend regime, count bars (per-column run-length encoding)
    # For each column: age = row_number - first_row_of_current_regime + 1
    # Vectorized computation instead of groupby loop
    _v = trend_sign_change.values
    _idx = np.arange(len(_v))[:, None]
    _last_change = np.where(_v == 1, _idx, 0)
    _last_change = np.maximum.accumulate(_last_change, axis=0)
    _rank = _idx - _last_change + 1
    feats["trend_age_hours"] = pd.DataFrame(
        _rank, index=trend_age_cumsum.index, columns=trend_age_cumsum.columns
    ).astype(np.float32).fillna(1)

    # higher_highs_count_48h: Count of higher highs in last 48 hours (trend quality)
    # A higher high is when current high > previous high
    higher_high = (h > h.shift(1)).astype(np.float32)
    feats["higher_highs_count_48h"] = ff.numba_rolling_sum(higher_high, 48).astype(np.float32)

    # trend_retest_success_rate: How often do retests hold?
    # Proxy: when price pulls back to EMA, does it bounce?
    near_ema = (feats["dist_ema_fast"].abs() < 0.5).astype(np.float32)  # Within 0.5 ATR of EMA
    ret_after_near = feats["ret4h"].shift(-4).fillna(0)  # Return 4h later
    retest_success = (near_ema * (ret_after_near * trend_sign > 0)).astype(np.float32)
    retest_attempts = near_ema.rolling(48, min_periods=1).sum()
    retest_successes = retest_success.rolling(48, min_periods=1).sum()
    feats["trend_retest_success_rate"] = (retest_successes / (retest_attempts + 1e-12)).clip(0, 1).astype(np.float32)

    # trend_overextension_z: Z-scored distance from EMA (overextension detection)
    dist_ema_rolling_mean = ff.numba_rolling_mean(feats["dist_ema_fast"], 48)
    dist_ema_rolling_std = ff.numba_rolling_std(feats["dist_ema_fast"], 48)
    feats["trend_overextension_z"] = ((feats["dist_ema_fast"] - dist_ema_rolling_mean) / (dist_ema_rolling_std + 1e-12)).clip(-5, 5).astype(np.float32)

    # volume_trend_alignment: Is volume rising with the trend?
    # Correlation between volume and price direction over 24h
    vol_change = v.diff(1).fillna(0).astype(np.float32)
    price_dir = np.sign(feats["ret1h"]).astype(np.float32)
    feats["volume_trend_alignment"] = ff.numba_rolling_corr(vol_change, price_dir, 24).fillna(0).clip(-1, 1).astype(np.float32)

    # trend_regime_stability: How stable is the current trend regime?
    # Low value = regime transition risk, high value = stable trend
    trend_sign_flips = (trend_sign != trend_sign.shift(1)).rolling(48, min_periods=1).sum()
    feats["trend_regime_stability"] = (1.0 / (1.0 + trend_sign_flips)).astype(np.float32)

    # --- MR Features: Dip Quality & Support Context ---

    # trend_strength_vs_reversion: Ratio of trend force to mean-reversion force
    # High = trending (avoid MR), Low = ranging (good for MR)
    trend_force = feats["ret24h"].abs()
    mr_force = feats["autocorr_6h"].abs().clip(0, 1)  # Negative autocorr = MR force
    feats["trend_strength_vs_reversion"] = (trend_force / (mr_force + 1e-12)).astype(np.float32)

    # support_quality_score: How strong is nearby support?
    # Based on: volume at nearby price levels, number of touches, recency
    # Proxy: count how often price bounced from current level in last 120h
    lo_24 = ff.numba_rolling_min(l, 24)
    dist_to_low = ((c - lo_24) / (atr_base + 1e-12)).astype(np.float32)
    # Support quality is high when: close to recent low, high volume there
    near_support = (dist_to_low.abs() < 1.0).astype(np.float32)  # Within 1 ATR of 24h low
    vol_at_support = (near_support * v).astype(np.float32)
    vol_total = v.rolling(24, min_periods=1).sum()
    support_vol_ratio = vol_at_support.rolling(24, min_periods=1).sum() / (vol_total + 1e-12)
    feats["support_quality_score"] = (near_support * support_vol_ratio).astype(np.float32)

    # dip_velocity: How fast did we dip? (Sharp dips = better MR)
    # Rate of change of distance from high
    hi_12 = ff.numba_rolling_max(h, 12)
    dist_from_high_12 = ((c - hi_12) / (atr_base + 1e-12)).astype(np.float32)
    feats["dip_velocity"] = (dist_from_high_12.diff(1).fillna(0) * -1).astype(np.float32)  # Positive = dipping fast

    # dip_volume_profile: Volume characteristics during the dip
    # High volume on dip = capitulation (good MR), low volume = orderly decline (bad MR)
    is_dipping = (feats["ret4h"] < 0).astype(np.float32)
    vol_on_dip = (is_dipping * v).astype(np.float32)
    vol_avg = v.rolling(24, min_periods=1).mean()
    feats["dip_volume_profile"] = ((vol_on_dip / (vol_avg + 1e-12)) * is_dipping).fillna(0).astype(np.float32)

    # reversion_target_distance: Distance to mean (upside potential for MR)
    # Using VWAP as mean proxy
    vwap_proxy = (c * v).rolling(24, min_periods=1).sum() / (v.rolling(24, min_periods=1).sum() + 1e-12)
    feats["reversion_target_distance"] = ((vwap_proxy - c) / (atr_base + 1e-12)).astype(np.float32)

    # ---------------------------------------------------------------------
    # Regime-transition / complexity features (2h/4h/8h trade-horizon focus)
    # ---------------------------------------------------------------------
    # Volatility regime in rolling z-space
    feats["vol_regime_z"] = zscore_rolling(feats["rv_24h"], 48).fillna(0).astype(np.float32)
    feats["is_high_vol_regime"] = (feats["vol_regime_z"] > 0.75).astype(np.float32)
    feats["is_low_vol_regime"] = (feats["vol_regime_z"] < -0.75).astype(np.float32)

    # Trend regime score from 24h return in local-vol units
    feats["trend_regime"] = (
        feats["ret24h"] / (feats["rv_24h"] * np.sqrt(24.0) + 1e-12)
    ).clip(-3, 3).astype(np.float32)
    feats["is_trending"] = (feats["trend_regime"].abs() >= 0.75).astype(np.float32)
    feats["is_ranging"] = (1.0 - feats["is_trending"]).astype(np.float32)

    # Liquidity regime: high value means better-than-usual liquidity
    feats["liq_regime"] = (-feats["amihud_z"]).clip(-5, 5).astype(np.float32)

    # Regime switching intensity (12h) and stability (24h)
    trend_state = np.sign(feats["trend_regime"]).replace(0, np.nan).ffill().fillna(0)
    vol_state = np.sign(feats["vol_regime_z"]).replace(0, np.nan).ffill().fillna(0)
    trend_switch_evt = (trend_state != trend_state.shift(1)).astype(np.float32)
    vol_switch_evt = (vol_state != vol_state.shift(1)).astype(np.float32)
    feats["trend_regime_switch_12h"] = ff.numba_rolling_sum(trend_switch_evt, 12).astype(np.float32)
    feats["vol_regime_switch_12h"] = ff.numba_rolling_sum(vol_switch_evt, 12).astype(np.float32)
    feats["regime_stability_24h"] = (
        1.0 / (1.0 + ff.numba_rolling_sum((trend_switch_evt + vol_switch_evt) > 0, 24))
    ).astype(np.float32)

    # Entropy of switching process (binary entropy of switch-rate over horizon)
    def _binary_entropy(p):
        p = p.clip(1e-6, 1 - 1e-6)
        return -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))

    sw12 = (ff.numba_rolling_sum((trend_switch_evt + vol_switch_evt) > 0, 12) / 12.0).clip(0, 1)
    sw48 = (ff.numba_rolling_sum((trend_switch_evt + vol_switch_evt) > 0, 48) / 48.0).clip(0, 1)
    feats["regime_transition_entropy_12h"] = _binary_entropy(sw12).astype(np.float32)
    feats["regime_transition_entropy_48h"] = _binary_entropy(sw48).astype(np.float32)
    feats["entropy_jump_24h"] = (
        feats["regime_transition_entropy_12h"] - feats["regime_transition_entropy_48h"]
    ).astype(np.float32)
    feats["complexity_regime_24h"] = (
        0.5 * feats["regime_transition_entropy_12h"]
        + 0.5 * feats["regime_transition_entropy_48h"]
    ).astype(np.float32)

    # ---------------------------------------------------------------------
    # Extended 4-day regime features for medium-term market structure
    # ---------------------------------------------------------------------
    # 4-day volatility regime (96-hour rolling z-score)
    feats["vol_regime_z_4d"] = zscore_rolling(feats["rv_24h"], 96).fillna(0).astype(np.float32)
    
    # 4-day trend strength (normalized by local volatility)
    feats["trend_strength_4d"] = (
        ff.numba_rolling_mean(feats["ret24h"], 96) / 
        (ff.numba_rolling_std(feats["ret24h"], 96) * np.sqrt(96.0) + 1e-12)
    ).clip(-3, 3).astype(np.float32)
    
    # 4-day regime stability (inverse of regime changes over 96 hours)
    trend_switch_4d = ff.numba_rolling_sum(trend_switch_evt, 96)
    vol_switch_4d = ff.numba_rolling_sum(vol_switch_evt, 96)
    feats["regime_stability_4d"] = (
        1.0 / (1.0 + trend_switch_4d + vol_switch_4d)
    ).astype(np.float32)
    
    # 4-day volatility persistence (autocorrelation of volatility)
    vol_persistence_4d = ff.numba_rolling_corr(
        feats["rv_24h"], feats["rv_24h"].shift(96), 96
    ).fillna(0).clip(-1, 1).astype(np.float32)
    feats["vol_persistence_4d"] = vol_persistence_4d
    
    # 4-day average trend regime duration (vectorized)
    # Average duration = window / (number of trend changes + 1)
    trend_changes_4d = ff.numba_rolling_sum(trend_sign_change, 96)
    feats["trend_regime_duration_4d"] = (
        96.0 / (trend_changes_4d + 1.0)
    ).astype(np.float32)

    # Regime interaction terms requested in config
    feats["rsi_z_x_regime_vol"] = (feats.get("rsi_z", 0.0) * feats["vol_regime_z"]).astype(np.float32)
    feats["vol_z_x_regime_trend"] = (feats["vol_z"] * feats["trend_regime"]).astype(np.float32)
    feats["mtf_divergence_x_regime_vol_12h"] = (
        feats["mtf_div_mag"] * ff.numba_rolling_mean(feats["vol_regime_z"], 12)
    ).astype(np.float32)
    feats["hurst_proxy_x_regime_trend_48h"] = (
        feats["hurst_proxy_24"] * ff.numba_rolling_mean(feats["trend_regime"], 48)
    ).astype(np.float32)
    feats["rsi_x_high_vol"] = (
        ((feats["rsi"] - 50.0) / 50.0) * feats["is_high_vol_regime"]
    ).astype(np.float32)
    feats["trend_x_trending"] = (feats["trend_regime"] * feats["is_trending"]).astype(np.float32)
    feats["vol_z_x_low_vol"] = (feats["vol_z"] * feats["is_low_vol_regime"]).astype(np.float32)

    # ---------------------------------------------------------------------
    # Entry/trap quality features for 2h/4h/8h opportunity framing
    # ---------------------------------------------------------------------
    # Bounce signal: short-horizon reversal after stress in the opposite direction.
    down_stress = ((feats["ret2h"] < 0) | (feats["ret4h"] < 0)).astype(np.float32)
    up_bounce = (feats["ret2h"] > 0).astype(np.float32)
    feats["bounce_signal"] = (
        down_stress * up_bounce * (1.0 + feats["vol_z"].clip(lower=0))
    ).astype(np.float32)

    # Volume capitulation: sharp adverse move + high abnormal volume (causal proxy).
    adverse_4h = (-feats["ret4h"]).clip(lower=0)
    feats["volume_capitulation"] = (
        adverse_4h * feats["vol_z"].clip(lower=0)
    ).astype(np.float32)

    # Trap strength: exhaustion + capitulation + failed continuation context.
    feats["trap_strength"] = (
        feats["volume_capitulation"] * (1.0 + feats["overext"].clip(lower=0)) * (1.0 - feats["accept"])
    ).astype(np.float32)

    # Composite entry quality across 2h/4h/8h context.
    feats["entry_quality_composite"] = (
        0.40 * feats["accept"]
        + 0.25 * feats["bounce_signal"]
        + 0.20 * feats["retest_accept_score"]
        + 0.15 * (1.0 - feats["ambig"].clip(0, 1))
    ).astype(np.float32)

    # Specialist proxies at feature stage (actual specialist outputs are added in engine.py).
    feats["predicted_vol_6h"] = (
        (feats["rv_6h"] / (feats["rv_24h"] + 1e-12)).clip(0, 10)
    ).astype(np.float32)
    feats["trap_quality"] = (
        (1.0 / (1.0 + feats["trap_strength"])) * (1.0 - feats["ambig"].clip(0, 1))
    ).astype(np.float32)

    # =====================================================================
    # User Requested Features (Report 2026-02-10) - TF/MR/Alpha
    # =====================================================================

    # Base Components (already moved earlier for interactions)
    # ema_6 = ema(c, 6)
    # trend_t = ema_6.diff(1).astype(np.float32)
    # feats["trend_t"] = trend_t
    trend_t = feats["trend_t"]

    # trend_z_t = trend_t / std(price, 24)
    std_c_24 = ff.numba_rolling_std(c, 24)
    feats["trend_z_t"] = (trend_t / (std_c_24 + 1e-12)).astype(np.float32)

    # convexity_t
    convexity_t = trend_t.diff(1).astype(np.float32)
    feats["convexity_t"] = convexity_t

    # convexity_bis_t
    feats["convexity_bis_t"] = (ema_6 - ema_24).diff(1).astype(np.float32)

    # convexity_z_t
    convexity_z_t = zscore_rolling(convexity_t, 24).fillna(0).astype(np.float32)
    # feats["convexity_z_t"] = convexity_z_t # Not requested but needed for intermediates

    # breakout_t / breakout_z
    feats["breakout_t"] = ((c - ema_24) / (std_c_24 + 1e-12)).astype(np.float32)
    breakout_z = feats["breakout_t"]

    # rvol
    # v is log-transformed volume (Log -> EWMA(5)) from _transform_volume
    # ema_v_24 is EMA of log-volume
    # rvol_ratio = exp(log(vol) - log(avg_vol)) = vol / avg_vol
    ema_v_24 = ema(v, 24)
    rvol_ratio = np.exp(v - ema_v_24)
    log_1_rvol = np.log(1.0 + rvol_ratio).astype(np.float32)

    # impulse
    feats["impulse"] = (feats["ret1h"] / (feats["rv_6h"] + 1e-12)).astype(np.float32)
    impulse = feats["impulse"]

    # pct_pos
    min_24 = ff.numba_rolling_min(c, 24)
    max_24 = ff.numba_rolling_max(c, 24)
    pct_pos = ((c - min_24) / (max_24 - min_24 + 1e-12)).clip(0, 1)

    # squeeze
    squeeze = feats["vol_compression"]

    # --- TF Meta Features ---
    feats["vw_breakout"] = (breakout_z * log_1_rvol).astype(np.float32)

    sigmoid_rvol = (1.0 / (1.0 + np.exp(-(v - ema_v_24)))).astype(np.float32)
    feats["breakout_soft"] = (breakout_z * sigmoid_rvol).astype(np.float32)

    feats["tail_score"] = (feats["trend_z_t"] *
                           np.maximum(0, convexity_z_t) *
                           np.maximum(0, breakout_z)).astype(np.float32)

    # --- MR Meta Features ---
    sigmoid_neg_conv_z = (1.0 / (1.0 + np.exp(convexity_z_t))).astype(np.float32) # sigmoid(-x)
    feats["mr_soft"] = (breakout_z.abs() * sigmoid_neg_conv_z).astype(np.float32)

    feats["mr_potential"] = ((c - ema_24).abs() / (feats["atr_pct_base"] * c + 1e-12)).astype(np.float32)

    feats["mr_potential_exhaust"] = (feats["mr_potential"] * np.maximum(0, -convexity_z_t)).astype(np.float32)

    feats["climax"] = (breakout_z.abs() * log_1_rvol).astype(np.float32)

    sigmoid_conv_z = (1.0 / (1.0 + np.exp(-convexity_z_t))).astype(np.float32)
    feats["vol_exhaust"] = (log_1_rvol * sigmoid_conv_z).astype(np.float32)

    feats["mr_climax"] = (breakout_z.abs() * log_1_rvol * sigmoid_neg_conv_z).astype(np.float32)

    imp_abs = impulse.abs()
    imp_abs_lag = imp_abs.shift(1).fillna(0)
    feats["shock_decay"] = (imp_abs_lag * np.maximum(0, imp_abs_lag - imp_abs)).astype(np.float32)

    feats["pct_extreme"] = (pct_pos - 0.5).abs().astype(np.float32)

    feats["mr_pct"] = (feats["pct_extreme"] * sigmoid_conv_z).astype(np.float32)

    tz_abs = feats["trend_z_t"].abs()
    feats["stall"] = np.maximum(0, tz_abs.shift(1).fillna(0) - tz_abs).astype(np.float32)

    feats["mr_failure"] = (squeeze * breakout_z.abs() * feats["stall"]).astype(np.float32)

    # --- Alpha Features ---
    feats["breakout_min"] = np.minimum(np.maximum(0, breakout_z), log_1_rvol).astype(np.float32)

    imp_lag = impulse.shift(1).fillna(0)
    feats["impulse_reversal"] = (np.maximum(0, -imp_lag) * np.maximum(0, impulse)).astype(np.float32)

    feats["impulse_reversal_short"] = (np.maximum(0, imp_lag) * np.maximum(0, -impulse)).astype(np.float32)

    feats["breakout_confirmed"] = (breakout_z * (rvol_ratio > 1.2).astype(np.float32)).astype(np.float32)

    feats["pct_breakout_t"] = np.maximum(0, pct_pos - 0.9).astype(np.float32)

    # --- New User Requested Features (KER, Vortex, ADX, VWAP, HVN/LVN) ---

    # 1. Kaufman Efficiency Ratio (KER)
    # Using c_log (log-prices) for calculation
    for n in [10, 16, 24]:
        feats[f"ker_{n}"] = ff.numba_ker(c_log, n).astype(np.float32)

    # 2. Vortex Indicator
    for n in [14, 21, 34]:
        # Using smoothed log-prices (o, h, l, c_log)
        feats[f"vortex_diff_{n}"] = ff.numba_vortex(h, l, c_log, n).astype(np.float32)

    # 3. ADX & Gated Features
    for n in [7, 10, 14]:
        adx, dip, dim = ff.numba_adx(h, l, c_log, n)
        feats[f"adx_{n}"] = adx.astype(np.float32)
        feats[f"adx_di_plus_{n}"] = dip.astype(np.float32)
        feats[f"adx_di_minus_{n}"] = dim.astype(np.float32)

        # Gated features
        feats[f"adx_{n}_gt25"] = (feats[f"adx_{n}"] > 25).astype(np.float32)
        # Slope of ADX (is trend strengthening?)
        feats[f"adx_{n}_slope"] = feats[f"adx_{n}"].diff(1).astype(np.float32)

    # 4. Trapped Longs / VWAP Distance
    # "Distance from the average entry price of the last N hours"
    # We use c_log and v (log-vol) for VWAP proxy in log-space
    for n in [12, 24, 96]:
        vwap_n = ff.numba_rolling_vwap(c_log, v, n)

        # Distance normalized by ATR (atr_base is raw ATR, atr_ln is log-ATR)
        # Since we are in log-space, (c_log - vwap_log) is a percentage diff.
        # We normalize by atr_ln (volatility in log-space).
        feats[f"dist_vwap_{n}_atr"] = ((c_log - vwap_n) / (feats["atr_ln"] + 1e-12)).astype(np.float32)

        # Trapped Longs: Price < VWAP. Magnitude of trapped signal.
        # Positive value = Longs are trapped (Price below VWAP)
        feats[f"trapped_longs_{n}"] = ((vwap_n - c_log) / (feats["atr_ln"] + 1e-12)).clip(lower=0).astype(np.float32)

    feats["clv_t"] = (((c_log - l) - (h - c_log)) / ((h - l) + 1e-9)).astype(np.float32)

    tr_15m = np.maximum(h - l, np.maximum((h - c_log.shift(1)).abs(), (l - c_log.shift(1)).abs()))
    body_ratio_15m = (c_log - o).abs() / ((h - l) + 1e-9)
    feats["body_ratio_15m"] = body_ratio_15m.astype(np.float32)

    upper_wick = (h - np.maximum(o, c_log)).clip(lower=0)
    lower_wick = (np.minimum(o, c_log) - l).clip(lower=0)
    feats["rejection_proxy"] = ((lower_wick - upper_wick) / ((h - l) + 1e-9)).astype(np.float32)

    sv = v * np.sign(c_log - c_log.shift(1))
    c_log_diff1 = c_log.diff(1)
    c_log_diff1_abs = c_log_diff1.abs()
    c_log_diff1_sign = np.sign(c_log_diff1)
    press_base = ((c_log - o) / ((h - l) + 1e-9)) * v
    h_minus_l = h - l

    for n in [12, 24]:
        atr_15m_n = ff.numba_ewma(tr_15m, 2.0 / (n + 1.0), False)
        feats[f"range_norm_{n}"] = (h_minus_l / (atr_15m_n + 1e-12)).astype(np.float32)

        sv_sum_n = ff.numba_rolling_sum(sv, n)
        v_sum_n = ff.numba_rolling_sum(v, n)
        feats[f"sv_imb_{n}"] = (sv_sum_n / (v_sum_n + 1e-12)).astype(np.float32)

        feats[f"press_{n}"] = ff.numba_rolling_mean(press_base, n).astype(np.float32)

        feats[f"impact_{n}"] = ff.numba_rolling_mean(c_log_diff1_abs / (v + 1e-9), n).astype(np.float32)

        ts_mean_n = ff.numba_rolling_mean(c_log_diff1, n)
        ts_std_n = ff.numba_rolling_std(c_log_diff1, n)
        feats[f"ts_{n}"] = (ts_mean_n / (ts_std_n + 1e-12)).astype(np.float32)

        prog_n = (c_log - c_log.shift(n)).abs()
        feats[f"prog_eff_{n}"] = (prog_n / (v_sum_n + 1e-12)).astype(np.float32)

        feats[f"pers_{n}"] = ff.numba_rolling_mean(c_log_diff1_sign, n).astype(np.float32)

        hh_count_n = ff.numba_rolling_sum((h > h.shift(1)).astype(np.float32), n)
        feats[f"hh_count_{n}"] = hh_count_n.astype(np.float32)

        ll_count_n = ff.numba_rolling_sum((l < l.shift(1)).astype(np.float32), n)
        feats[f"ll_count_{n}"] = ll_count_n.astype(np.float32)

        feats[f"skew_{n}"] = ff.apply_to_frame(c_log_diff1, ff._numba_rolling_skew, n).astype(np.float32)

        climax_range_med_n = ff.apply_to_frame(h_minus_l, ff._numba_rolling_median, n)
        feats[f"climax_range_{n}"] = (h_minus_l / (climax_range_med_n + 1e-12)).astype(np.float32)

        climax_vol_med_n = ff.apply_to_frame(v, ff._numba_rolling_median, n)
        feats[f"climax_vol_{n}"] = (v / (climax_vol_med_n + 1e-12)).astype(np.float32)

        vwap_z_n = ff.numba_rolling_vwap(c_log, v, n)

        diff_vwap = c_log - vwap_z_n
        std_vwap = ff.numba_rolling_std(diff_vwap, n)
        feats[f"z_vwap_{n}"] = (diff_vwap / (std_vwap + 1e-12)).astype(np.float32)

        feats[f"z_r_{n}"] = ((c_log_diff1 - ts_mean_n) / (ts_std_n + 1e-12)).astype(np.float32)

        c_log_mean_n = ff.numba_rolling_mean(c_log, n)
        c_log_std_n = ff.numba_rolling_std(c_log, n)
        feats[f"bb_pos_{n}"] = ((c_log - c_log_mean_n) / (c_log_std_n + 1e-12)).astype(np.float32)

    # 5. Volume Node Features (HVN/LVN)
    # Loop over columns, construct DF, call function, stack results
    need_vp_features = (
        not requested_feature_set
        or any(str(k).startswith("vp_") for k in requested_feature_set)
    )
    if need_vp_features:
        tprint("Computing HVN/LVN features...")
        try:
            from .volume_node_features import hvn_lvn_features_ohlcv

            # Get feature names from a sample run
            first_col = c_log.columns[0]
            df_first = pd.DataFrame({
                "open": o[first_col],
                "high": h[first_col],
                "low": l[first_col],
                "close": c_log[first_col], # Use c_log, not FFD c
                "volume": v[first_col]
            })
            sample_res = hvn_lvn_features_ohlcv(df_first)
            hvn_keys = list(sample_res.columns)

            hvn_results = _compute_hvn_feature_frames(o, h, l, c_log, v, hvn_keys)

            # Add to main feats with prefix
            for k, df_res in hvn_results.items():
                feats[f"vp_{k}"] = df_res

        except Exception as e:
            tprint(f"WARNING: HVN/LVN calculation failed: {e}")
    else:
        tprint("Skipping HVN/LVN features: no vp_* keys requested")

    # Free target_proxy — no longer needed after gated feature selection
    del target_proxy, time_blocks, train_mask_proxy
    # Free base price/volume DataFrames and all remaining intermediates before CausalTransform
    del o, h, l, c, v
    del atr_base, atr, dir_s, rv6, rv12, rv_ratio, mkt_gates
    gc.collect()

    # --- explicit peer-context and ts-percentile features ---
    if (
        not requested_feature_set
        or any(str(k).startswith("cs_rank_") or str(k).startswith("cs_rz_") for k in requested_feature_set)
    ):
        cs_feats = add_cross_sectional_peer_context_features(feats, min_group_size=5)
        feats.update(cs_feats)

    if not requested_feature_set or any(str(k).startswith("ts_pct_") for k in requested_feature_set):
        ts_feats = add_time_series_percentile_features(feats, lookback=720, min_history_fraction=0.25)
        feats.update(ts_feats)

    if requested_feature_set:
        feats = {k: v for k, v in feats.items() if k in requested_feature_set}
    tprint(f"Features: {len(feats)} features before CausalTransform. Applying transforms...")
    # Transform cache can be enabled for incremental/tail-only runs to persist parquet transforms.
    transform_cache_enabled = bool(cfg.get("feature_transform_cache_enabled", False))
    transform_cache_dir = cfg.get("feature_transform_cache_dir", "./cache/feature_transforms")
    transformer = CausalFeatureTransformer(
        winsor_qt=0.02,
        roll_window=24 * 30,
        cache_dir=transform_cache_dir,
        enable_cache=transform_cache_enabled,
    )

    skip_transform_set = {
        "liq_state", "sin_hod", "cos_hod", "sin_dow", "cos_dow", "range_24h_pct", "range_12h_pct",
        "volatility_zscore", "breakout_24h", "draw_sym_10h", "draw_extreme_10h",
        "G_VOL_LIQ_GT1", "G_VOL_LIQ_GT2", "G_VOL_LIQ_GT3", "G_LIQ_GOOD", "G_LIQ_GREAT", "G_LIQ_EXCEL",
        "mtf_divergence", "vol_price_diverge", "meta_alignment",
        # Residualised features — already z-scored, don't double-transform
        "rsi_z", "dist_ema_fast_z", "dist_vwap_norm_z", "flow_persistence_z",
        "excess_6h_z", "vol_z_z", "atr_expansion_z", "coherence_24_z",
        "accept_surprise", "overext_surprise",
        "blowoff_risk_surprise", "exh_qual_surprise",
        "dist_vwap_resid", "dist_ema_fast_resid", "trend_pct_resid",
    }

    position_sizer_keys = {
        "ATR_decay_rate", "ATR_spike_ratio", "ATR_ratio_short_long",
        "bar_direction_entropy", "realized_vol_15m_realized_vol_2h",
        "micro_range_decay", "range_decay", "vol_regime_transition",
        "close_position_in_range", "distance_to_local_high", "distance_to_local_low",
        "distance_to_vwap", "bidirectional_range_ratio", "bollinger_band_width",
        "choppiness_index_20", "climax_volume_ratio", "dist_ema50_atr",
        "dist_ema100_atr", "dist_ema200_atr", "dist_rolling_7d_high",
        "dist_prior_day_high", "dist_prior_day_low", "dist_range_mid_atr",
        "dist_weekly_vwap", "cost_to_atr", "direction_entropy_20",
        "atr_change_rate", "acceleration_of_move", "accept_gt66",
        "bars_since_trend_flip", "MACD_histogram", "RSI", "dist_local_swing",
        "dist_ma100_atr", "dist_vwap_atr", "abs_edge_pred",
    }
    if not requested_feature_set or position_sizer_keys.intersection(requested_feature_set):
        tprint("Features: adding missing position sizer features")

        atr_base = feats.get("atr_pct_base", pd.DataFrame(index=c_log.index, columns=c_log.columns, dtype=np.float32))
        if atr_base.empty:
            atr_base = _raw_atr_pct

        ret_1 = None
        bar_range = None
        atr_mean_24 = None
        high_12 = None
        low_12 = None
        high_24 = None
        low_24 = None

        if _needs_feature("bar_direction_entropy", "realized_vol_15m_realized_vol_2h", "direction_entropy_20", "acceleration_of_move", "vol_regime_transition"):
            ret_1 = (c_log.shift(-1) / c_log - 1.0).fillna(0.0).astype(np.float32)

        if _needs_feature("micro_range_decay", "range_decay", "choppiness_index_20", "accept_gt66"):
            bar_range = h_raw - l_raw

        if _needs_feature("ATR_decay_rate"):
            feats["ATR_decay_rate"] = atr_base.apply(lambda x: pd.Series(slope_nb(x.values, 6), index=x.index), axis=0).astype(np.float32)

        if _needs_feature("ATR_spike_ratio", "ATR_ratio_short_long"):
            atr_mean_24 = atr_base.rolling(24, min_periods=1).mean()
        if _needs_feature("ATR_spike_ratio"):
            feats["ATR_spike_ratio"] = (atr_base / (atr_mean_24 + 1e-9)).fillna(1.0).astype(np.float32)
        if _needs_feature("ATR_ratio_short_long"):
            atr_mean_3 = atr_base.rolling(3, min_periods=1).mean()
            feats["ATR_ratio_short_long"] = (atr_mean_3 / (atr_mean_24 + 1e-9)).fillna(1.0).astype(np.float32)
            del atr_mean_3

        if _needs_feature("bar_direction_entropy"):
            feats["bar_direction_entropy"] = ret_1.apply(lambda x: pd.Series(binary_entropy_nb(x.values, 12), index=x.index), axis=0).astype(np.float32)

        rv_1 = rv_2 = rv_24 = None
        if _needs_feature("realized_vol_15m_realized_vol_2h", "vol_regime_transition"):
            rv_1 = ret_1.apply(lambda x: pd.Series(realized_vol_nb(x.values, 1), index=x.index), axis=0)
            rv_2 = ret_1.apply(lambda x: pd.Series(realized_vol_nb(x.values, 2), index=x.index), axis=0)
        if _needs_feature("realized_vol_15m_realized_vol_2h"):
            feats["realized_vol_15m_realized_vol_2h"] = (rv_1 / (rv_2 + 1e-9)).fillna(1.0).astype(np.float32)
        if _needs_feature("vol_regime_transition"):
            rv_24 = ret_1.apply(lambda x: pd.Series(realized_vol_nb(x.values, 24), index=x.index), axis=0)
            rv24_mean_48 = rv_24.rolling(48, min_periods=1).mean()
            feats["vol_regime_transition"] = (rv_24 / (rv24_mean_48 + 1e-9)).fillna(1.0).astype(np.float32)

        if _needs_feature("micro_range_decay"):
            feats["micro_range_decay"] = bar_range.apply(lambda x: pd.Series(slope_nb(x.values, 3), index=x.index), axis=0).astype(np.float32)

        if _needs_feature("range_decay"):
            range_mean_3 = bar_range.rolling(3, min_periods=1).mean()
            range_mean_6 = bar_range.rolling(6, min_periods=1).mean()
            feats["range_decay"] = (range_mean_3 / (range_mean_6 + 1e-9)).fillna(1.0).astype(np.float32)

        if _needs_feature("close_position_in_range"):
            close_pos_in_range = pd.DataFrame(index=h_raw.index, columns=h_raw.columns, dtype=np.float32)
            for col in h_raw.columns:
                close_pos_in_range[col] = close_location_in_bar_nb(h_raw[col].values, l_raw[col].values, c_raw[col].values)
            feats["close_position_in_range"] = close_pos_in_range.astype(np.float32)

        if _needs_feature("distance_to_local_high", "distance_to_local_low", "bidirectional_range_ratio", "dist_local_swing"):
            high_12 = h_raw.rolling(12, min_periods=1).max()
            low_12 = l_raw.rolling(12, min_periods=1).min()
        if _needs_feature("distance_to_local_high"):
            feats["distance_to_local_high"] = ((high_12 - c_raw) / (c_raw + 1e-9)).fillna(0.0).astype(np.float32)
        if _needs_feature("distance_to_local_low"):
            feats["distance_to_local_low"] = ((c_raw - low_12) / (c_raw + 1e-9)).fillna(0.0).astype(np.float32)

        if _needs_feature("distance_to_vwap", "dist_vwap_atr"):
            vwap_24 = pd.DataFrame(index=c_raw.index, columns=c_raw.columns, dtype=np.float32)
            for col in c_raw.columns:
                vwap_24[col] = vwap_nb(c_raw[col].values, v_raw[col].values, 24)
            feats["distance_to_vwap"] = ((c_raw - vwap_24) / (vwap_24 + 1e-9)).fillna(0.0).astype(np.float32)
            if _needs_feature("dist_vwap_atr"):
                feats["dist_vwap_atr"] = feats["distance_to_vwap"]

        if _needs_feature("bidirectional_range_ratio"):
            range_12 = high_12 - low_12
            high_3 = h_raw.rolling(3, min_periods=1).max()
            low_3 = l_raw.rolling(3, min_periods=1).min()
            range_3 = high_3 - low_3
            feats["bidirectional_range_ratio"] = (range_3 / (range_12 + 1e-9)).fillna(1.0).astype(np.float32)

        if _needs_feature("bollinger_band_width"):
            sma_20 = c_log.rolling(20, min_periods=1).mean()
            std_20 = c_log.rolling(20, min_periods=1).std()
            feats["bollinger_band_width"] = (2 * std_20 / (sma_20 + 1e-9)).fillna(0.0).astype(np.float32)

        if _needs_feature("choppiness_index_20"):
            tr = pd.concat([h_raw - l_raw, (h_raw - c_raw.shift(1)).abs(), (l_raw - c_raw.shift(1)).abs()], axis=1).max(axis=1)
            tr_20 = tr.rolling(20, min_periods=1).sum()
            high_max_20 = h_raw.rolling(20, min_periods=1).max()
            low_min_20 = l_raw.rolling(20, min_periods=1).min()
            range_20 = high_max_20 - low_min_20
            feats["choppiness_index_20"] = (100.0 * np.log(tr_20 / (range_20 + 1e-9) + 1e-9) / np.log(20.0)).clip(0, 100).astype(np.float32)

        if _needs_feature("climax_volume_ratio"):
            vol_mean_24 = v_raw.rolling(24, min_periods=1).mean()
            vol_max_6 = v_raw.rolling(6, min_periods=1).max()
            feats["climax_volume_ratio"] = (vol_max_6 / (vol_mean_24 + 1e-9)).fillna(1.0).astype(np.float32)

        if _needs_feature("dist_ema50_atr"):
            ema_50 = c_log.apply(lambda x: pd.Series(ema_nb(x.values, 50), index=x.index), axis=0)
            feats["dist_ema50_atr"] = ((c_log - ema_50) / (atr_base + 1e-9)).astype(np.float32)
        if _needs_feature("dist_ema100_atr"):
            ema_100 = c_log.apply(lambda x: pd.Series(ema_nb(x.values, 100), index=x.index), axis=0)
            feats["dist_ema100_atr"] = ((c_log - ema_100) / (atr_base + 1e-9)).astype(np.float32)
        if _needs_feature("dist_ema200_atr"):
            ema_200 = c_log.apply(lambda x: pd.Series(ema_nb(x.values, 200), index=x.index), axis=0)
            feats["dist_ema200_atr"] = ((c_log - ema_200) / (atr_base + 1e-9)).astype(np.float32)

        if _needs_feature("dist_rolling_7d_high"):
            high_168 = h_raw.rolling(168, min_periods=1).max()
            feats["dist_rolling_7d_high"] = ((high_168 - c_raw) / (c_raw + 1e-9)).fillna(0.0).astype(np.float32)

        if _needs_feature("dist_prior_day_high", "dist_prior_day_low", "dist_range_mid_atr"):
            high_24 = h_raw.rolling(24, min_periods=1).max()
            low_24 = l_raw.rolling(24, min_periods=1).min()
        if _needs_feature("dist_prior_day_high"):
            feats["dist_prior_day_high"] = ((high_24.shift(1) - c_raw) / (c_raw + 1e-9)).fillna(0.0).astype(np.float32)
        if _needs_feature("dist_prior_day_low"):
            feats["dist_prior_day_low"] = ((c_raw - low_24.shift(1)) / (c_raw + 1e-9)).fillna(0.0).astype(np.float32)
        if _needs_feature("dist_range_mid_atr"):
            range_mid = (high_24 + low_24) / 2.0
            feats["dist_range_mid_atr"] = ((c_raw - range_mid) / (atr_base + 1e-9)).astype(np.float32)

        if _needs_feature("dist_weekly_vwap"):
            vwap_168 = pd.DataFrame(index=c_raw.index, columns=c_raw.columns, dtype=np.float32)
            for col in c_raw.columns:
                vwap_168[col] = vwap_nb(c_raw[col].values, v_raw[col].values, 168)
            feats["dist_weekly_vwap"] = ((c_raw - vwap_168) / (vwap_168 + 1e-9)).fillna(0.0).astype(np.float32)

        if _needs_feature("cost_to_atr"):
            spread_pct = pd.DataFrame(index=atr_base.index, columns=atr_base.columns, dtype=np.float32).fillna(0.0)
            feats["cost_to_atr"] = (spread_pct / (atr_base + 1e-9)).fillna(0.0).astype(np.float32)

        if _needs_feature("direction_entropy_20"):
            feats["direction_entropy_20"] = ret_1.apply(lambda x: pd.Series(binary_entropy_nb(x.values, 20), index=x.index), axis=0).astype(np.float32)

        if _needs_feature("atr_change_rate"):
            feats["atr_change_rate"] = atr_base.pct_change().fillna(0.0).astype(np.float32)

        if _needs_feature("acceleration_of_move"):
            feats["acceleration_of_move"] = ret_1.apply(lambda x: pd.Series(slope_nb(x.values, 6), index=x.index), axis=0).astype(np.float32)

        if _needs_feature("accept_gt66"):
            close_in_range = (c_raw - l_raw) / (h_raw - l_raw + 1e-9)
            feats["accept_gt66"] = (close_in_range > 0.66).astype(np.float32).rolling(6, min_periods=1).mean().astype(np.float32)

        if _needs_feature("bars_since_trend_flip"):
            trend_slope = c_log.apply(lambda x: pd.Series(slope_nb(x.values, 6), index=x.index), axis=0)
            trend_sign = (trend_slope > 0).astype(np.float32)
            bars_since_flip = pd.DataFrame(index=trend_sign.index, columns=trend_sign.columns, dtype=np.float32)
            for col in trend_sign.columns:
                sign = trend_sign[col].values
                count = np.zeros_like(sign, dtype=np.float32)
                for i in range(1, len(sign)):
                    count[i] = count[i - 1] + 1 if sign[i] == sign[i - 1] else 0
                bars_since_flip[col] = count
            feats["bars_since_trend_flip"] = bars_since_flip.astype(np.float32)

        if _needs_feature("MACD_histogram"):
            ema_12 = c_log.apply(lambda x: pd.Series(ema_nb(x.values, 12), index=x.index), axis=0)
            ema_26 = c_log.apply(lambda x: pd.Series(ema_nb(x.values, 26), index=x.index), axis=0)
            macd = ema_12 - ema_26
            signal = macd.apply(lambda x: pd.Series(ema_nb(x.values, 9), index=x.index), axis=0)
            feats["MACD_histogram"] = (macd - signal).astype(np.float32)

        if _needs_feature("RSI") and "rsi" not in feats:
            rsi_14 = c_log.apply(lambda x: pd.Series(rsi(x.to_frame(), 14)[x.name], index=x.index), axis=0)
            feats["RSI"] = rsi_14.astype(np.float32)

        if _needs_feature("dist_local_swing"):
            dist_to_high = (high_12 - c_raw).abs()
            dist_to_low = (c_raw - low_12).abs()
            feats["dist_local_swing"] = np.minimum(dist_to_high, dist_to_low).astype(np.float32)

        if _needs_feature("dist_ma100_atr"):
            ma_100 = c_log.rolling(100, min_periods=1).mean()
            feats["dist_ma100_atr"] = ((c_log - ma_100) / (atr_base + 1e-9)).astype(np.float32)

        if _needs_feature("abs_edge_pred") and "abs_edge_pred" not in feats:
            feats["abs_edge_pred"] = pd.DataFrame(index=c_log.index, columns=c_log.columns, dtype=np.float32).fillna(0.0)

        del ret_1, bar_range, atr_mean_24, rv_1, rv_2, rv_24, high_12, low_12, high_24, low_24
        gc.collect()

    del h_raw, l_raw, c_raw, v_raw
    gc.collect()

    tprint(f"Features: done ({len(feats)} keys)")

    # Add dynamically generated peer context and TS pct to skip set
    for k in feats.keys():
        if k.startswith("cs_rank_") or k.startswith("cs_rz_") or k.startswith("ts_pct_"):
            skip_transform_set.add(k)


    for w in gate_windows:
        for prefix in ["s", "reject", "retest_accept", "tf_qual", "mr_qual", "vol_z", "liquidity"]:
            for suffix in ["mean", "std", "z", "pct", "bin3", "gt25", "gt50", "gt66", "gt75", "gt85", "gt90"]:
                skip_transform_set.add(f"{prefix}_{suffix}_{w}")

    def _is_boolean_like_feature(arr_like) -> bool:
        arr = np.asarray(arr_like, dtype=np.float32)
        if arr.size == 0:
            return False
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return False
        if finite.min() < 0.0 or finite.max() > 1.0:
            return False
        rounded = np.round(finite)
        return bool(np.all(np.abs(finite - rounded) <= 1e-6))

    # Capture shared index/columns ONCE before converting to numpy
    _sample_df = feats[list(feats.keys())[0]]
    _feat_index = _sample_df.index
    _feat_columns = list(_sample_df.columns)
    del _sample_df

    feat_keys_list = list(feats.keys())
    transform_keys: list[str] = []
    n_skipped = 0
    for k in feat_keys_list:
        arr = np.asarray(feats[k], dtype=np.float32)
        if (
            k in skip_transform_set
            or k.startswith("cs_rank_")
            or k.startswith("cs_rz_")
            or k.startswith("ts_pct_")
            or _is_boolean_like_feature(arr)
        ):
            feats[k] = arr
            n_skipped += 1
        else:
            feats[k] = arr
            transform_keys.append(k)

    tprint(
        f"CausalTransform workset: {len(transform_keys)} transform, {n_skipped} skipped"
    )

    n_transformed = 0
    for i, k in enumerate(transform_keys):
        arr = feats[k]
        try:
            feats[k] = np.asarray(transformer.transform(arr, name=k), dtype=np.float32)
            n_transformed += 1
        except Exception as e:
            tprint(f"Warning: Transform failed for {k}: {e}")
            import traceback
            traceback.print_exc()
            feats[k] = arr
        if (i + 1) % 50 == 0:
            gc.collect()
            tprint(f"  CausalTransform progress: {i+1}/{len(transform_keys)}")
    del transformer
    gc.collect()
    tprint(f"CausalTransform complete: {n_transformed} transformed, {n_skipped} skipped")

    # Final check for Inf/NaN (numpy arrays now)
    tprint("Features: performing final Inf/NaN check")
    for k in feats:
        arr = feats[k]
        if not np.isfinite(arr).all():
            n_bad = (~np.isfinite(arr)).sum()
            tprint(f"  WARNING: {k} has {n_bad} non-finite values, replacing with 0")
            feats[k] = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    tprint(f"Features: done ({len(feats)} keys)")
    return feats, _feat_index, _feat_columns

# ============================================================
# Position Sizer V2 Numba/Numpy Feature Builders
# ============================================================
import numpy as np
from typing import Dict, List, Tuple
from numba import njit

@njit(cache=True)
def rolling_mean_nb(x: np.ndarray, window: int) -> np.ndarray:
    out = np.full_like(x, np.nan)
    n = len(x)
    if n == 0 or window <= 0:
        return out

    run_sum = 0.0
    count = 0
    for i in range(n):
        val = x[i]
        if not np.isnan(val):
            run_sum += val
            count += 1

        if i >= window:
            old_val = x[i - window]
            if not np.isnan(old_val):
                run_sum -= old_val
                count -= 1

        if count > 0:
            out[i] = run_sum / count
    return out

@njit(cache=True)
def ema_nb(x: np.ndarray, window: int) -> np.ndarray:
    out = np.full_like(x, np.nan)
    n = len(x)
    if n == 0 or window <= 0:
        return out

    alpha = 2.0 / (window + 1.0)
    ema = np.nan

    for i in range(n):
        val = x[i]
        if not np.isnan(val):
            if np.isnan(ema):
                ema = val
            else:
                ema = val * alpha + ema * (1.0 - alpha)
            out[i] = ema
        else:
            out[i] = ema
    return out

@njit(cache=True)
def rolling_std_nb(x: np.ndarray, window: int) -> np.ndarray:
    out = np.full_like(x, np.nan)
    n = len(x)
    if n == 0 or window <= 0:
        return out

    for i in range(n):
        start = max(0, i - window + 1)
        slice_x = x[start:i+1]

        valid_count = 0
        mean = 0.0
        for val in slice_x:
            if not np.isnan(val):
                mean += val
                valid_count += 1

        if valid_count > 1:
            mean /= valid_count
            var = 0.0
            for val in slice_x:
                if not np.isnan(val):
                    var += (val - mean)**2
            out[i] = np.sqrt(var / (valid_count - 1))
        elif valid_count == 1:
            out[i] = 0.0

    return out

@njit(cache=True)
def rolling_zscore_nb(x: np.ndarray, window: int) -> np.ndarray:
    out = np.full_like(x, np.nan)
    mean_arr = rolling_mean_nb(x, window)
    std_arr = rolling_std_nb(x, window)

    for i in range(len(x)):
        std = std_arr[i]
        if not np.isnan(std) and std > 1e-9:
            out[i] = (x[i] - mean_arr[i]) / std
        elif not np.isnan(x[i]) and not np.isnan(mean_arr[i]):
            out[i] = 0.0
    return out

@njit(cache=True)
def realized_vol_nb(ret: np.ndarray, window: int) -> np.ndarray:
    return rolling_std_nb(ret, window)

@njit(cache=True)
def downside_semivol_nb(ret: np.ndarray, window: int) -> np.ndarray:
    out = np.full_like(ret, np.nan)
    n = len(ret)
    if n == 0 or window <= 0:
        return out

    for i in range(n):
        start = max(0, i - window + 1)
        slice_ret = ret[start:i+1]

        valid_count = 0
        var = 0.0
        for val in slice_ret:
            if not np.isnan(val):
                valid_count += 1
                if val < 0:
                    var += val**2

        if valid_count > 1:
            out[i] = np.sqrt(var / (valid_count - 1))
        elif valid_count == 1:
            out[i] = 0.0
    return out

@njit(cache=True)
def close_location_in_bar_nb(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    out = np.full_like(close, np.nan)
    for i in range(len(close)):
        rng = high[i] - low[i]
        if not np.isnan(rng) and rng > 1e-9:
            out[i] = (close[i] - low[i]) / rng
        else:
            out[i] = 0.5
    return out

@njit(cache=True)
def range_over_atr_nb(high: np.ndarray, low: np.ndarray, atr: np.ndarray) -> np.ndarray:
    out = np.full_like(high, np.nan)
    for i in range(len(high)):
        a = atr[i]
        if not np.isnan(a) and a > 1e-9:
            out[i] = (high[i] - low[i]) / a
        else:
            out[i] = 1.0
    return out

@njit(cache=True)
def base_pred_summary_nb(base_pred_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n, k = base_pred_matrix.shape

    b_mean = np.zeros(n, dtype=np.float32)
    b_std = np.zeros(n, dtype=np.float32)
    b_min = np.zeros(n, dtype=np.float32)
    b_max = np.zeros(n, dtype=np.float32)
    b_range = np.zeros(n, dtype=np.float32)
    sign_agree = np.zeros(n, dtype=np.float32)
    top2_gap = np.zeros(n, dtype=np.float32)

    for i in range(n):
        row = base_pred_matrix[i, :]

        valid_count = 0
        r_mean = 0.0
        for val in row:
            if not np.isnan(val):
                r_mean += val
                valid_count += 1

        if valid_count == 0:
            continue

        r_mean /= valid_count
        b_mean[i] = r_mean

        var = 0.0
        r_min = 1e9
        r_max = -1e9
        pos_count = 0
        neg_count = 0

        for val in row:
            if not np.isnan(val):
                var += (val - r_mean)**2
                if val < r_min: r_min = val
                if val > r_max: r_max = val
                if val > 0: pos_count += 1
                elif val < 0: neg_count += 1

        b_std[i] = np.sqrt(var / valid_count) if valid_count > 0 else 0.0
        b_min[i] = r_min
        b_max[i] = r_max
        b_range[i] = r_max - r_min
        sign_agree[i] = max(pos_count, neg_count) / valid_count

        if valid_count >= 2:
            sorted_vals = np.sort(row[~np.isnan(row)])
            top2_gap[i] = sorted_vals[-1] - sorted_vals[-2]

    return b_mean, b_std, b_min, b_max, b_range, sign_agree, top2_gap

@njit(cache=True)
def liquidity_shock_nb(short_vol: np.ndarray, long_vol: np.ndarray) -> np.ndarray:
    out = np.zeros_like(short_vol)
    for i in range(len(out)):
        lv = long_vol[i]
        sv = short_vol[i]
        if not np.isnan(lv) and not np.isnan(sv) and lv > 1e-9:
            out[i] = (sv - lv) / lv
    return out

def compute_returns_nb(close: np.ndarray, periods: int) -> np.ndarray:
    out = np.full_like(close, np.nan)
    n = len(close)
    for i in range(periods, n):
        old = close[i - periods]
        if not np.isnan(old) and old > 1e-9:
            out[i] = (close[i] - old) / old
    return out


@njit(cache=True)
def rolling_sum_nb(x: np.ndarray, window: int) -> np.ndarray:
    out = np.full_like(x, np.nan)
    n = len(x)
    if n == 0 or window <= 0:
        return out
    run_sum = 0.0
    count = 0
    for i in range(n):
        val = x[i]
        if not np.isnan(val):
            run_sum += val
            count += 1
        if i >= window:
            old_val = x[i - window]
            if not np.isnan(old_val):
                run_sum -= old_val
                count -= 1
        if count > 0:
            out[i] = run_sum
    return out

@njit(cache=True)
def rolling_max_nb(x: np.ndarray, window: int) -> np.ndarray:
    out = np.full_like(x, np.nan)
    n = len(x)
    for i in range(n):
        start = max(0, i - window + 1)
        mx = -np.inf
        valid = False
        for j in range(start, i + 1):
            if not np.isnan(x[j]):
                valid = True
                if x[j] > mx: mx = x[j]
        if valid: out[i] = mx
    return out

@njit(cache=True)
def rolling_min_nb(x: np.ndarray, window: int) -> np.ndarray:
    out = np.full_like(x, np.nan)
    n = len(x)
    for i in range(n):
        start = max(0, i - window + 1)
        mn = np.inf
        valid = False
        for j in range(start, i + 1):
            if not np.isnan(x[j]):
                valid = True
                if x[j] < mn: mn = x[j]
        if valid: out[i] = mn
    return out

@njit(cache=True)
def slope_nb(x: np.ndarray, window: int) -> np.ndarray:
    out = np.full_like(x, np.nan)
    n = len(x)
    if n < 2 or window < 2: return out
    for i in range(window - 1, n):
        sum_x = 0.0; sum_y = 0.0; sum_xy = 0.0; sum_x2 = 0.0
        count = 0
        for j in range(window):
            idx = i - window + 1 + j
            y = x[idx]
            if not np.isnan(y):
                sum_x += j
                sum_y += y
                sum_xy += j * y
                sum_x2 += j * j
                count += 1
        if count > 1:
            mean_x = sum_x / count
            mean_y = sum_y / count
            num = sum_xy - count * mean_x * mean_y
            den = sum_x2 - count * mean_x**2
            if den > 1e-9:
                out[i] = num / den
            else:
                out[i] = 0.0
    return out

@njit(cache=True)
def vwap_nb(close: np.ndarray, volume: np.ndarray, window: int) -> np.ndarray:
    out = np.full_like(close, np.nan)
    n = len(close)
    for i in range(n):
        start = max(0, i - window + 1)
        sum_cv = 0.0
        sum_v = 0.0
        for j in range(start, i + 1):
            c = close[j]
            v = volume[j]
            if not np.isnan(c) and not np.isnan(v):
                sum_cv += c * v
                sum_v += v
        if sum_v > 1e-9:
            out[i] = sum_cv / sum_v
        else:
            out[i] = close[i]
    return out

@njit(cache=True)
def entropy_nb(x: np.ndarray, window: int, n_bins: int = 5) -> np.ndarray:
    out = np.full_like(x, np.nan)
    n = len(x)
    for i in range(window - 1, n):
        start = i - window + 1
        slice_x = x[start:i+1]
        valid_x = []
        for val in slice_x:
            if not np.isnan(val): valid_x.append(val)

        if len(valid_x) > 1:
            mn = min(valid_x); mx = max(valid_x)
            if mx > mn:
                counts = np.zeros(n_bins)
                step = (mx - mn) / n_bins
                for val in valid_x:
                    b = int((val - mn) / step)
                    if b == n_bins: b -= 1
                    counts[b] += 1
                probs = counts / len(valid_x)
                ent = 0.0
                for p in probs:
                    if p > 0: ent -= p * np.log2(p)
                out[i] = ent
            else:
                out[i] = 0.0
        elif len(valid_x) == 1:
            out[i] = 0.0
    return out

@njit(cache=True)
def binary_entropy_nb(x: np.ndarray, window: int) -> np.ndarray:
    out = np.full_like(x, np.nan)
    n = len(x)
    for i in range(window - 1, n):
        start = i - window + 1
        pos_c = 0; neg_c = 0; tot = 0
        for j in range(start, i + 1):
            val = x[j]
            if not np.isnan(val):
                tot += 1
                if val > 0: pos_c += 1
                elif val < 0: neg_c += 1
        if tot > 0:
            p_pos = pos_c / tot
            p_neg = neg_c / tot
            ent = 0.0
            if p_pos > 0: ent -= p_pos * np.log2(p_pos)
            if p_neg > 0: ent -= p_neg * np.log2(p_neg)
            out[i] = ent
    return out


def build_position_sizer_feature_frame(raw_inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    close = np.ascontiguousarray(raw_inputs.get("close", np.empty(0)), dtype=np.float32)
    high = np.ascontiguousarray(raw_inputs.get("high", np.empty(0)), dtype=np.float32)
    low = np.ascontiguousarray(raw_inputs.get("low", np.empty(0)), dtype=np.float32)
    volume = np.ascontiguousarray(raw_inputs.get("volume", np.empty(0)), dtype=np.float32)
    atr = np.ascontiguousarray(raw_inputs.get("atr", np.empty(0)), dtype=np.float32)
    spread = np.ascontiguousarray(raw_inputs.get("spread_pct", np.empty(0)), dtype=np.float32)
    meta_pred = np.ascontiguousarray(raw_inputs.get("meta_oof_pred", np.empty(0)), dtype=np.float32)
    base_pred_matrix = np.ascontiguousarray(raw_inputs.get("base_oof_pred_matrix", np.empty((len(close), 0))), dtype=np.float32)

    n = len(close)
    if n == 0:
        return {}

    # Basic state / Ensembles
    b_mean, b_std, b_min, b_max, b_range, sign_agree, top2_gap = base_pred_summary_nb(base_pred_matrix)
    ret_1 = compute_returns_nb(close, 1)
    ret_3 = compute_returns_nb(close, 3)
    ret_6 = compute_returns_nb(close, 6)
    ret_12 = compute_returns_nb(close, 12)
    ret_24 = compute_returns_nb(close, 24)

    # Ranges & ATR
    atr_pct = np.where(close > 1e-9, atr / close, 0.0).astype(np.float32)
    bar_range = high - low
    range_1 = range_over_atr_nb(high, low, atr)

    high_3 = rolling_max_nb(high, 3)
    low_3 = rolling_min_nb(low, 3)
    range_3 = range_over_atr_nb(high_3, low_3, atr)
    range_3_abs = high_3 - low_3

    high_6 = rolling_max_nb(high, 6)
    low_6 = rolling_min_nb(low, 6)
    range_6_abs = high_6 - low_6

    # Model 1 additions
    impulse_range = range_6_abs # Proxy impulse window as 6 bars (1h=6h)
    range_last_3bars_impulse_range = np.where(impulse_range > 1e-9, range_3_abs / impulse_range, 1.0).astype(np.float32)
    volatility_contraction_ratio = np.where(rolling_mean_nb(bar_range, 24) > 1e-9, rolling_mean_nb(bar_range, 4) / rolling_mean_nb(bar_range, 24), 1.0).astype(np.float32)
    atr_decay_rate = slope_nb(atr, 6).astype(np.float32) # Using short slope for decay rate

    rv_1 = realized_vol_nb(ret_1, 1) # ~15m assuming 15m underlying or 1 bar
    rv_2 = realized_vol_nb(ret_1, 2)
    rv_4 = realized_vol_nb(ret_1, 4)
    rv_6 = realized_vol_nb(ret_1, 6)
    rv_12 = realized_vol_nb(ret_1, 12)
    rv_24 = realized_vol_nb(ret_1, 24)
    rv_48 = realized_vol_nb(ret_1, 48)

    realized_vol_15m_2h = np.where(rv_2 > 1e-9, rv_1 / rv_2, 1.0).astype(np.float32) # Ratio approximation
    micro_range_decay = slope_nb(bar_range, 3).astype(np.float32)

    wick_ratio_last_bar = np.where(bar_range > 1e-9, np.minimum(high - np.maximum(close, opens:=np.roll(close, 1)), np.minimum(close, opens) - low) / bar_range, 0.0).astype(np.float32)
    close_position_in_range = close_location_in_bar_nb(high, low, close).astype(np.float32)

    # Simple rejection logic (wick > 50% bar)
    rejection_ratio = np.where(bar_range > 1e-9, (np.maximum(high - np.maximum(close, opens), np.minimum(close, opens) - low)) / bar_range, 0.0)
    rejection_ratio = rolling_mean_nb(rejection_ratio, 6).astype(np.float32)

    vol_sum_3 = rolling_sum_nb(volume, 3)
    vol_sum_4 = rolling_sum_nb(volume, 4)
    vol_sum_6 = rolling_sum_nb(volume, 6)
    vol_sum_12 = rolling_sum_nb(volume, 12)
    vol_sum_24 = rolling_sum_nb(volume, 24)

    mean_vol_24 = rolling_mean_nb(volume, 24)

    impulse_participation_volume = np.where(mean_vol_24 > 1e-9, vol_sum_6 / (6 * mean_vol_24), 1.0).astype(np.float32)
    terminal_climax_volume = np.where(mean_vol_24 > 1e-9, volume / mean_vol_24, 1.0).astype(np.float32)
    post_impulse_persistence = np.where(vol_sum_6 > 1e-9, vol_sum_4 / vol_sum_6, 1.0).astype(np.float32)

    reversal_bar_strength = np.where(bar_range > 1e-9, (close - opens) / bar_range, 0.0).astype(np.float32)
    bidirectional_range_ratio = np.where(rolling_max_nb(high, 12) - rolling_min_nb(low, 12) > 1e-9, range_3_abs / (rolling_max_nb(high, 12) - rolling_min_nb(low, 12)), 1.0).astype(np.float32)

    momentum_last_3bars_impulse_return = np.where(np.abs(ret_6) > 1e-9, ret_3 / ret_6, 0.0).astype(np.float32)
    drift_after_impulse = slope_nb(close, 4).astype(np.float32)
    slope_last_n_bars = slope_nb(close, 6).astype(np.float32)

    impulse_volume_ratio = np.where(mean_vol_24 > 1e-9, vol_sum_12 / (12 * mean_vol_24), 1.0).astype(np.float32)
    terminal_volume_ratio = np.where(vol_sum_6 > 1e-9, vol_sum_3 / (vol_sum_6/2), 1.0).astype(np.float32)
    post_impulse_volume_persistence2 = np.where(vol_sum_6 > 1e-9, vol_sum_4 / (vol_sum_6 * 0.66), 1.0).astype(np.float32)
    impulse_volume_slope = slope_nb(volume, 6).astype(np.float32)

    impulse_vol_ratio = np.where(rv_48 > 1e-9, rv_12 / rv_48, 1.0).astype(np.float32)
    impulse_range_atr_ratio = np.where(atr > 1e-9, range_6_abs / rolling_mean_nb(atr, 24), 1.0).astype(np.float32)
    vol_compression_ratio = np.where(rv_6 > 1e-9, rv_4 / rv_6, 1.0).astype(np.float32)
    range_decay = np.where(rolling_mean_nb(bar_range, 6) > 1e-9, rolling_mean_nb(bar_range, 3) / rolling_mean_nb(bar_range, 6), 1.0).astype(np.float32)

    # Model 2 additions
    impulse_speed = np.where(range_6_abs > 1e-9, ret_6 / range_6_abs, 0.0).astype(np.float32)
    impulse_acceleration = slope_nb(ret_1, 6).astype(np.float32)
    wick_cluster_ratio = rolling_mean_nb(wick_ratio_last_bar, 3).astype(np.float32)
    rejection_bar_count = rolling_sum_nb(np.where(wick_ratio_last_bar > 0.4, 1.0, 0.0), 6).astype(np.float32)
    atr_spike_ratio = np.where(rolling_mean_nb(atr, 24) > 1e-9, atr / rolling_mean_nb(atr, 24), 1.0).astype(np.float32)

    high_12 = rolling_max_nb(high, 12)
    low_12 = rolling_min_nb(low, 12)
    distance_to_local_high = np.where(close > 1e-9, (high_12 - close) / close, 0.0).astype(np.float32)
    distance_to_local_low = np.where(close > 1e-9, (close - low_12) / close, 0.0).astype(np.float32)

    vwap_val = vwap_nb(close, volume, 24)
    distance_to_vwap = np.where(vwap_val > 1e-9, (close - vwap_val) / vwap_val, 0.0).astype(np.float32)

    climax_volume_ratio = np.where(mean_vol_24 > 1e-9, rolling_max_nb(volume, 6) / mean_vol_24, 1.0).astype(np.float32)

    vol_countertrend = np.where(np.sign(ret_1) != np.sign(ret_6), volume, 0.0)
    reversal_volume_ratio = np.where(vol_sum_6 > 1e-9, rolling_sum_nb(vol_countertrend, 6) / vol_sum_6, 0.0).astype(np.float32)

    vol_wicks = np.where(wick_ratio_last_bar > 0.4, volume, 0.0)
    mean_vol_12 = rolling_mean_nb(volume, 12)
    rejection_volume_ratio = np.where(mean_vol_12 > 1e-9, rolling_sum_nb(vol_wicks, 6) / mean_vol_12, 0.0).astype(np.float32)

    terminal_vol_ratio = np.where(rv_6 > 1e-9, rv_3:=realized_vol_nb(ret_1, 3) / rv_6, 1.0).astype(np.float32)

    vol_up = rolling_sum_nb(np.where(ret_1 > 0, volume, 0.0), 12)
    vol_down = rolling_sum_nb(np.where(ret_1 < 0, volume, 0.0), 12)
    volatility_asymmetry = np.where(vol_up + vol_down > 1e-9, vol_up / (vol_up + vol_down), 0.5).astype(np.float32)

    # Model 3 additions
    vol_regime_transition = np.where(rolling_mean_nb(rv_24, 48) > 1e-9, rv_24 / rolling_mean_nb(rv_24, 48), 1.0).astype(np.float32)
    atr_ratio_short_long = np.where(rolling_mean_nb(atr, 24) > 1e-9, rolling_mean_nb(atr, 3) / rolling_mean_nb(atr, 24), 1.0).astype(np.float32)

    bar_direction_entropy = binary_entropy_nb(ret_1, 12).astype(np.float32)
    wick_entropy = entropy_nb(wick_ratio_last_bar, 12).astype(np.float32)
    impulse_breakdown_score = np.where(ret_6 > 1e-9, ret_3 / ret_6, 0.0).astype(np.float32) # Same proxy as momentum ratio

    volume_volatility = np.where(mean_vol_12 > 1e-9, rolling_std_nb(volume, 12) / mean_vol_12, 0.0).astype(np.float32)
    volume_regime_shift = np.where(mean_vol_24 > 1e-9, rolling_mean_nb(volume, 6) / mean_vol_24, 1.0).astype(np.float32)
    volume_entropy = entropy_nb(volume, 12).astype(np.float32)

    return_per_volume = np.where(volume > 1e-9, np.abs(ret_1) / volume, 0.0).astype(np.float32)

    mean_rv_12 = rolling_mean_nb(rv_12, 12)
    vol_of_vol = np.where(mean_rv_12 > 1e-9, rolling_std_nb(rv_12, 12) / mean_rv_12, 0.0).astype(np.float32)
    vol_regime_shift_4_16 = np.where(rolling_mean_nb(rv_12, 16) > 1e-9, rolling_mean_nb(rv_12, 4) / rolling_mean_nb(rv_12, 16), 1.0).astype(np.float32)

    range_cv = np.where(rolling_mean_nb(bar_range, 12) > 1e-9, rolling_std_nb(bar_range, 12) / rolling_mean_nb(bar_range, 12), 0.0).astype(np.float32)
    return_vol_ratio = np.where(rv_12 > 1e-9, np.abs(ret_1) / rv_12, 0.0).astype(np.float32)

    # Pre-existing standard features
    ema_12 = ema_nb(close, 12)
    ema_24 = ema_nb(close, 24)
    price_vs_ema_12_z = np.where(ema_12 > 1e-9, (close - ema_12) / ema_12, 0.0).astype(np.float32)
    price_vs_ema_24_z = np.where(ema_24 > 1e-9, (close - ema_24) / ema_24, 0.0).astype(np.float32)
    ema_12_minus_ema_24_z = np.where(ema_24 > 1e-9, (ema_12 - ema_24) / ema_24, 0.0).astype(np.float32)
    rv_ratio_6_24 = np.where(rv_24 > 1e-9, rv_6 / rv_24, 1.0).astype(np.float32)
    dsv_12 = downside_semivol_nb(ret_1, 12)

    vol_std_24 = rolling_std_nb(volume, 24)
    volume_z_24 = np.where(vol_std_24 > 1e-9, (volume - mean_vol_24) / vol_std_24, 0.0).astype(np.float32)
    volume_z_12 = np.where(rolling_std_nb(volume, 12) > 1e-9, (volume - mean_vol_12) / rolling_std_nb(volume, 12), 0.0).astype(np.float32)
    dollar_vol_z_24 = volume_z_24 # Proxy

    spread_to_atr = np.where(atr_pct > 1e-9, spread / atr_pct, 0.0).astype(np.float32)

    regime_trend = np.ascontiguousarray(raw_inputs.get("regime_trend_score", np.zeros(n)), dtype=np.float32)
    regime_vol = np.ascontiguousarray(raw_inputs.get("regime_vol_score", np.zeros(n)), dtype=np.float32)
    regime_liq = np.ascontiguousarray(raw_inputs.get("regime_liquidity_score", np.zeros(n)), dtype=np.float32)
    hod = raw_inputs.get("hour_of_day", np.zeros(n))
    dow = raw_inputs.get("day_of_week", np.zeros(n))

    hour_sin = np.sin(2 * np.pi * hod / 24.0).astype(np.float32)
    hour_cos = np.cos(2 * np.pi * hod / 24.0).astype(np.float32)
    dow_sin = np.sin(2 * np.pi * dow / 7.0).astype(np.float32)
    dow_cos = np.cos(2 * np.pi * dow / 7.0).astype(np.float32)

    feature_dict = {
        "oof_base_mean": b_mean,
        "oof_base_std": b_std,
        "oof_base_min": b_min,
        "oof_base_max": b_max,
        "oof_base_range": b_range,
        "oof_sign_agreement_frac": sign_agree,
        "oof_top2_gap": top2_gap,
        "oof_meta_pred": meta_pred,
        "oof_meta_minus_base_mean": meta_pred - b_mean,
        "oof_rank_among_candidates": np.zeros(n, dtype=np.float32), # Replaced dynamically cross-sectionally

        "ret_1": ret_1, "ret_3": ret_3, "ret_6": ret_6, "ret_12": ret_12, "ret_24": ret_24,
        "price_vs_ema_12_z": price_vs_ema_12_z, "price_vs_ema_24_z": price_vs_ema_24_z,
        "ema_12_minus_ema_24_z": ema_12_minus_ema_24_z, "trend_slope_12_z": slope_last_n_bars,
        "trend_slope_24_z": slope_nb(close, 24).astype(np.float32),

        "atr_pct": atr_pct, "range_1_atr": range_1, "range_3_atr": range_3,
        "rv_6": rv_6, "rv_12": rv_12, "rv_24": rv_24, "rv_ratio_6_24": rv_ratio_6_24,
        "close_location_in_bar": close_position_in_range,
        "downside_semivol_12": dsv_12,

        "volume_z_12": volume_z_12, "volume_z_24": volume_z_24, "dollar_vol_z_24": dollar_vol_z_24,
        "spread_pct": spread, "spread_to_atr": spread_to_atr, "cost_to_atr": spread_to_atr,
        "slippage_proxy": spread_to_atr * 1.5, "liquidity_shock_z": liquidity_shock_nb(vol_sum_6, vol_sum_24).astype(np.float32),

        "regime_trend_score": regime_trend, "regime_vol_score": regime_vol, "regime_liquidity_score": regime_liq,
        "hour_sin": hour_sin, "hour_cos": hour_cos, "dow_sin": dow_sin, "dow_cos": dow_cos,
        "session_progress": (hod / 24.0).astype(np.float32),

        # --- NEW MODEL 1 ---
        "range_last_3bars_impulse_range": range_last_3bars_impulse_range,
        "volatility_contraction_ratio": volatility_contraction_ratio,
        "ATR_decay_rate": atr_decay_rate,
        "realized_vol_15m_realized_vol_2h": realized_vol_15m_2h,
        "micro_range_decay": micro_range_decay,
        "wick_ratio_last_bar": wick_ratio_last_bar,
        "close_position_in_range": close_position_in_range,
        "rejection_ratio": rejection_ratio,
        "impulse_participation_volume": impulse_participation_volume,
        "terminal_climax_volume": terminal_climax_volume,
        "post_impulse_persistence": post_impulse_persistence,
        "reversal_bar_strength": reversal_bar_strength,
        "bidirectional_range_ratio": bidirectional_range_ratio,
        "momentum_last_3bars_impulse_return": momentum_last_3bars_impulse_return,
        "drift_after_impulse": drift_after_impulse,
        "slope_last_n_bars": slope_last_n_bars,
        "impulse_volume_ratio": impulse_volume_ratio,
        "terminal_volume_ratio": terminal_volume_ratio,
        "post_impulse_volume_persistence": post_impulse_volume_persistence2,
        "impulse_volume_slope": impulse_volume_slope,
        "impulse_vol_ratio": impulse_vol_ratio,
        "impulse_range_atr_ratio": impulse_range_atr_ratio,
        "vol_compression_ratio": vol_compression_ratio,
        "range_decay": range_decay,

        # --- NEW MODEL 2 ---
        "impulse_speed": impulse_speed,
        "impulse_acceleration": impulse_acceleration,
        "wick_cluster_ratio": wick_cluster_ratio,
        "rejection_bar_count": rejection_bar_count,
        "ATR_spike_ratio": atr_spike_ratio,
        "distance_to_local_high": distance_to_local_high,
        "distance_to_local_low": distance_to_local_low,
        "distance_to_vwap": distance_to_vwap,
        "climax_volume_ratio": climax_volume_ratio,
        "reversal_volume_ratio": reversal_volume_ratio,
        "rejection_volume_ratio": rejection_volume_ratio,
        "terminal_vol_ratio": terminal_vol_ratio,
        "volatility_asymmetry": volatility_asymmetry,

        # --- NEW MODEL 3 ---
        "vol_regime_transition": vol_regime_transition,
        "ATR_ratio_short_long": atr_ratio_short_long,
        "bar_direction_entropy": bar_direction_entropy,
        "wick_entropy": wick_entropy,
        "impulse_breakdown_score": impulse_breakdown_score,
        "volume_volatility": volume_volatility,
        "volume_regime_shift": volume_regime_shift,
        "volume_entropy": volume_entropy,
        "return_per_volume": return_per_volume,
        "vol_of_vol": vol_of_vol,
        "vol_regime_shift": vol_regime_shift_4_16,
        "range_cv": range_cv,
        "return_vol_ratio": return_vol_ratio,
    }

    from extreme_price_movements.config import POSITION_SIZER_V2_FEATURE_CONFIG
    for key in POSITION_SIZER_V2_FEATURE_CONFIG["shared_feature_keys"]:
        if key not in feature_dict:
            feature_dict[key] = np.zeros(n, dtype=np.float32)

    for k in POSITION_SIZER_V2_FEATURE_CONFIG["model1_edge_feature_keys"] + POSITION_SIZER_V2_FEATURE_CONFIG["model2_downside_feature_keys"] + POSITION_SIZER_V2_FEATURE_CONFIG["model3_uncertainty_feature_keys"]:
        if k not in feature_dict and k not in ["edge_pred", "downside_pred", "edge_minus_downside", "abs_edge_pred"]:
            feature_dict[k] = np.zeros(n, dtype=np.float32)

    return feature_dict

def assemble_feature_matrix(feature_dict: Dict[str, np.ndarray], keys: List[str]) -> np.ndarray:
    """
    Returns a contiguous float32 matrix given a dictionary and a list of keys.
    Missing keys are filled with 0.0.
    """
    if not keys:
        return np.empty((0, 0), dtype=np.float32)

    n = 0
    for v in feature_dict.values():
        if isinstance(v, np.ndarray):
            n = len(v)
            break

    if n == 0:
        return np.empty((0, len(keys)), dtype=np.float32)

    out = np.zeros((n, len(keys)), dtype=np.float32)

    for i, k in enumerate(keys):
        if k in feature_dict:
            arr = feature_dict[k]
            if len(arr) == n:
                out[:, i] = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    return np.ascontiguousarray(out)

def add_cross_sectional_peer_context_features(feats: dict[str, pd.DataFrame], min_group_size: int = 5) -> dict[str, pd.DataFrame]:
    """
    Computes cross-sectional percentile ranks for select features to provide peer context.
    Ranks are strictly causal (computed per-timestamp).
    """
    from .utils import tprint
    tprint("Computing explicit cross-sectional peer-context features...")

    # Candidates specifically requested for cross-sectional ranking
    cs_candidates = {
        # Momentum relative
        "ret1h", "ret6h", "impulse", "trend_strength_4d", "trend_regime",
        # Volatility relative
        "atr_pct", "rv_6h", "rv_24h", "volatility_zscore",
        # Activity relative
        "vol_shock_z", "dollar_vol_z_24", "amihud_z", "volume_z_24", "vol_z",
        # Geometry relative
        "breakout_24h", "range_24h_pct", "range_zscore", "dist_ema_fast", "dist_vwap_norm"
    }

    added_feats = {}
    total_added = 0

    for candidate in cs_candidates:
        if candidate in feats:
            df = feats[candidate]
            if not isinstance(df, pd.DataFrame):
                continue

            # rank(axis=1, pct=True) computes cross-sectional percentile
            # We want to handle small cross sections safely.
            valid_counts = df.notna().sum(axis=1)
            mask = valid_counts < min_group_size

            # 1) Compute Cross-Sectional Rank
            cs_rank = df.rank(axis=1, pct=True)
            if mask.any():
                cs_rank.loc[mask, :] = 0.5
            cs_rank = cs_rank.fillna(0.5).astype(np.float32)
            added_feats[f"cs_rank_{candidate}"] = cs_rank

            # 2) Compute Cross-Sectional Robust Z-score (cs_rz)
            med = df.median(axis=1)
            mad = (df.sub(med, axis=0)).abs().median(axis=1)

            # MAD * 1.4826 for normal std proxy, bound by eps
            eps = 1e-6
            scale = (mad * 1.4826).clip(lower=eps)

            cs_rz = df.sub(med, axis=0).div(scale, axis=0)

            # Mask out insufficient group size with neutral 0.0
            if mask.any():
                cs_rz.loc[mask, :] = 0.0

            # Fill NaNs with 0.0 (neutral)
            cs_rz = cs_rz.fillna(0.0).astype(np.float32)
            added_feats[f"cs_rz_{candidate}"] = cs_rz

            total_added += 2

    tprint(f"Added {total_added} cross-sectional peer-context features.")
    return added_feats

def add_time_series_percentile_features(feats: dict[str, pd.DataFrame], lookback: int = 720, min_history_fraction: float = 0.25) -> dict[str, pd.DataFrame]:
    """
    Computes rolling causal time-series percentile ranks for select features.
    """
    from .utils import tprint
    import extreme_price_movements.fast_funcs as ff
    tprint("Computing rolling time-series percentile companion features...")

    # Candidates specifically requested for ts percentiles
    ts_pct_candidates = {
        # Price dynamics
        "ret1h", "ret6h", "impulse", "trend_strength_4d", "trend_regime",
        # Volatility & range
        "atr_pct", "rv_6h", "rv_24h", "vol_compression_ratio",
        # Activity
        "vol_shock_z", "dollar_vol_z_24", "amihud_z",
        # Geometry
        "breakout_24h", "dist_ema_fast", "wick_ratio"
    }

    added_feats = {}
    total_added = 0

    for candidate in ts_pct_candidates:
        if candidate in feats:
            df = feats[candidate]
            if not isinstance(df, pd.DataFrame):
                continue

            # Compute rolling rank percentile using fast_funcs
            ts_pct = ff.numba_rolling_rank_pct(df, window=lookback)

            # Mask out periods with insufficient history (e.g., < 25% of lookback)
            valid_counts = df.notna().rolling(lookback, min_periods=1).sum()
            min_required = int(lookback * min_history_fraction)
            mask = valid_counts < min_required
            if mask.any().any():
                # Emit neutral 0.5 where history is too short
                ts_pct = ts_pct.where(~mask, 0.5)

            ts_pct = ts_pct.fillna(0.5).astype(np.float32)

            added_feats[f"ts_pct_{candidate}"] = ts_pct
            total_added += 1

    tprint(f"Added {total_added} time-series percentile features.")
    return added_feats
