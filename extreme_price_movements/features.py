import hashlib
import os
import pickle
import re
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.special
from joblib import Memory
from numba import njit, prange

import extreme_price_movements.fast_funcs as ff
from extreme_price_movements.feature_transforms import CausalFeatureTransformer
from extreme_price_movements.frac_diff_adaptive import (
    find_min_ffd,
    frac_diff_ffd,
    get_weights_ffd,
)
from extreme_price_movements.intraday_crypto_library import (
    PERSISTED_INTRADAY_LIBRARY_COLUMNS,
    build_intraday_crypto_library,
)
from extreme_price_movements.perp_features import (
    compute_features as compute_perp_features,
)
from extreme_price_movements.time_utils import ensure_utc
from extreme_price_movements.utils import tprint
from extreme_price_movements.validation import validate_panel

# Suppress expected RuntimeWarnings from nanmin/nanmean/nanmax on all-NaN slices
# These are handled gracefully by replacing with 0 later in the pipeline
warnings.filterwarnings("ignore", message=".*All-NaN slice.*")
warnings.filterwarnings("ignore", message=".*Mean of empty slice.*")
# Suppress divide warnings from correlation calculations when stddev is 0
warnings.filterwarnings("ignore", message=".*invalid value encountered.*")

# Initialize joblib cache (use /tmp for writability)
_CACHE_DIR = os.environ.get("EPM_CACHE_DIR", "/tmp/epm_cache")
_cache = Memory(os.path.join(_CACHE_DIR, "features"), verbose=0)

# --- Per-column FFD incremental cache ---
_FFD_COL_CACHE_DIR = os.path.join(_CACHE_DIR, "ffd_columns")
EPS = 1e-12
_PERP_FEATURE_COLLISION_RENAMES = {
    "ret1h": "ret1h_perp",
}

_INTRADAY_PERSISTED_KEY_SET = set(PERSISTED_INTRADAY_LIBRARY_COLUMNS)


def _broadcast_series_to_frame(
    series: pd.Series, index: pd.Index, columns: pd.Index
) -> pd.DataFrame:
    arr = np.ascontiguousarray(series.to_numpy(dtype=np.float32))
    view = np.broadcast_to(arr[:, None], (len(index), len(columns)))
    return pd.DataFrame(view, index=index, columns=columns, copy=False).astype(
        np.float32, copy=False
    )


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
    tprint(
        f"Features: computing intraday location/trigger library "
        f"({len(selected_keys)} keys x {len(open_df.columns)} symbols) [vectorized]"
    )

    wide_lib = build_intraday_crypto_library(
        {
            "open": open_df.astype(np.float32, copy=False),
            "high": high_df.astype(np.float32, copy=False),
            "low": low_df.astype(np.float32, copy=False),
            "close": close_df.astype(np.float32, copy=False),
            "volume": volume_df.astype(np.float32, copy=False),
            "session_id": session_ids,
        }
    )
    if not isinstance(wide_lib, dict):
        raise TypeError("Expected dict output for dict input in intraday library")

    out: dict[str, pd.DataFrame] = {}
    for key in selected_keys:
        value = wide_lib.get(key)
        if isinstance(value, pd.DataFrame):
            out[key] = value.astype(np.float32, copy=False)
        elif isinstance(value, pd.Series):
            out[key] = _broadcast_series_to_frame(
                value, index=open_df.index, columns=open_df.columns
            )
    return out


def _sanitize_col_name(name):
    """Make column name filesystem-safe."""
    return re.sub(r"[^\w\-.]", "_", str(name))


def _col_data_hash(arr):
    """Fast hash of column data for cache key."""
    return hashlib.md5(np.ascontiguousarray(arr).tobytes()).hexdigest()[:16]


def _rolling_winsorize_causal(
    x: pd.DataFrame, window: int, q_lo: float, q_hi: float
) -> pd.DataFrame:
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
    x_proc = (
        _rolling_winsorize_causal(x_in, max(2, int(n)), q_lo, q_hi)
        if winsorize
        else x_in
    )

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
    return ff.numba_rolling_robust_zscore(
        x.astype(np.float32), int(n), float(quantile), float(eps)
    ).astype(np.float32)


def rsi(close: pd.DataFrame, n: int):
    return ff.numba_rsi(close, n)


def ema(x: pd.DataFrame, span: int):
    alpha = 2.0 / (span + 1.0)
    return ff.numba_ewma(x, alpha, False)


def _safe_log_df(df: pd.DataFrame, eps: float = 1e-9) -> pd.DataFrame:
    """Causal-safe log transform for strictly positive inputs."""
    return np.log(np.maximum(df, np.float32(eps))).astype(np.float32)


def _transform_close_fixed_ffd(
    df: pd.DataFrame,
    d: float = 0.4,
    _label: str = "close",
    already_logged: bool = False,
    thres: float = 1e-5,
) -> pd.DataFrame:
    """Transform close only with fixed d to avoid adaptive ADF leakage."""
    tprint(
        f"Transforming Close ({_label}): Log -> EWMA(5) -> FFD(d={d:.2f}) [{df.shape[1]} cols]"
    )
    df_log = df.astype(np.float32) if already_logged else _safe_log_df(df)
    df_den = ff.numba_ewma(df_log, 2.0 / 6.0, False)

    out = pd.DataFrame(index=df.index, columns=df.columns, dtype=np.float32)
    fallback_d_values = [float(x) for x in (0.6, 0.5, 0.4)]
    d_candidates = []
    for cand in [float(d)] + fallback_d_values:
        if cand not in d_candidates:
            d_candidates.append(cand)
    win_by_d = {
        cand: int(len(get_weights_ffd(cand, float(thres)))) for cand in d_candidates
    }
    fallback_used = 0
    direct_used = 0
    total_cols = len(df_den.columns)
    groups: dict[float | None, list[str]] = {}
    valid_counts = df_den.notna().sum(axis=0).to_numpy(dtype=np.int32, copy=False)
    for col, valid_n in zip(df_den.columns, valid_counts):
        valid_n = int(valid_n)

        d_use = None
        for cand in d_candidates:
            if win_by_d[cand] <= valid_n:
                d_use = cand
                break

        if d_use not in groups:
            groups[d_use] = []
        groups[d_use].append(col)

    for d_use, group_cols in groups.items():
        if d_use is None:
            out[group_cols] = df_den[group_cols].astype(np.float32)
            direct_used += len(group_cols)
        else:
            if d_use != float(d):
                fallback_used += len(group_cols)
            # Use parallel matrix-based FFD
            out[group_cols] = frac_diff_ffd(
                df_den[group_cols], d=float(d_use), thres=float(thres)
            ).astype(np.float32)
            tprint(
                f"Fixed FFD ({_label}): Applied d={d_use:.2f} to {len(group_cols)} columns"
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


def _rolling_shannon_entropy_df(
    df: pd.DataFrame, window: int, bins: int = 16
) -> pd.DataFrame:
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


def _rolling_permutation_entropy_df(
    df: pd.DataFrame, window: int, order: int = 3, delay: int = 1
) -> pd.DataFrame:
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
    # Using numba-accelerated rolling kernels instead of Pandas .rolling()
    mean = ff.numba_rolling_mean(rets, window).shift(1)
    var = (ff.numba_rolling_std(rets, window) ** 2).shift(1).clip(lower=1e-12)

    # Covariance with lagged self
    rets_lag = rets.shift(delay)
    rets_prod = rets * rets_lag
    cov = ff.numba_rolling_mean(rets_prod, window).shift(1) - mean * mean.shift(delay)

    # Autocorrelation
    autocorr = (cov / var).clip(-1, 1)

    # Also measure run length (consecutive same-sign periods)
    sign = (rets > 0).astype(np.float32)
    sign_change = (sign != sign.shift(delay)).astype(np.float32)
    run_freq = ff.numba_rolling_mean(sign_change, window).shift(1)

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

    # Compute variance at multiple scales using Numba-accelerated std function
    scales = [max(2, window // 8), max(4, window // 4), max(8, window // 2), window]
    variances = []
    for s in scales:
        v = (ff.numba_rolling_std(df, s) ** 2).shift(1)
        variances.append(v)

    # Variance ratio matrix: how variance scales with window
    # For white noise, var(s) / var(s/2) ≈ 2
    # For trend, var(s) / var(s/2) > 2
    # For MR, var(s) / var(s/2) < 2

    ratios = []
    for i in range(1, len(variances)):
        r = (variances[i] / (variances[i - 1] + 1e-12)).clip(0.1, 10)
        ratios.append(r)

    # Stack and compute flatness
    # Flat spectrum = all ratios near expected value (white noise behavior)
    # Concentrated spectrum = ratios deviate from expected

    # Expected ratio for white noise: scale_factor = scales[i] / scales[i-1]
    expected_ratios = [scales[i] / scales[i - 1] for i in range(1, len(scales))]

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
    tprint(
        f"Transforming Prices ({_label}): Log -> EWMA(5) -> Adaptive FracDiff [{df.shape[1]} cols]"
    )
    # Safe Log: Clip input to be at least 1e-9 to avoid log(0) or log(neg)
    df_log = np.log(np.maximum(df, 1e-9))
    df_den = ff.numba_ewma(df_log, 2.0 / 6.0, False)

    # Per-column incremental FFD cache
    cache_dir = os.path.join(
        _FFD_COL_CACHE_DIR, _sanitize_col_name(_label or "default")
    )
    os.makedirs(cache_dir, exist_ok=True)
    manifest_path = os.path.join(cache_dir, "manifest.pkl")
    try:
        with open(manifest_path, "rb") as f:
            manifest = pickle.load(f)
        if not isinstance(manifest, dict):
            manifest = {}
    except Exception:
        manifest = {}

    df_fd = pd.DataFrame(index=df.index, columns=df.columns, dtype=np.float32)
    total_cols = len(df_den.columns)
    stats = {"cached": 0, "cached_d": 0, "computed": 0}
    updated_manifest = {}

    for i, col in enumerate(df_den.columns):
        safe_col = _sanitize_col_name(col)
        # Hash RAW input — deterministic key for the full pipeline
        col_raw = df[col].to_numpy(dtype=np.float64)
        data_hash = _col_data_hash(col_raw)

        col_dir = os.path.join(cache_dir, safe_col)
        os.makedirs(col_dir, exist_ok=True)
        result_path = os.path.join(col_dir, f"ffd_{data_hash}.npy")
        col_manifest = manifest.get(col, {})

        # --- Level 1: exact raw-data match -> instant load ---
        if col_manifest.get("data_hash") == data_hash and os.path.exists(result_path):
            try:
                cached_vals = np.load(result_path, allow_pickle=False)
                if len(cached_vals) == len(df_fd):
                    df_fd[col] = cached_vals
                    updated_manifest[col] = col_manifest
                    stats["cached"] += 1
                    continue
            except Exception:
                pass

        # --- Level 2: reuse cached d_opt (skip expensive ADF search) ---
        d_opt = col_manifest.get("d_opt")
        if d_opt is not None:
            stats["cached_d"] += 1

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
            prev_hash = col_manifest.get("data_hash")
            if prev_hash and prev_hash != data_hash:
                stale_path = os.path.join(col_dir, f"ffd_{prev_hash}.npy")
                if os.path.exists(stale_path):
                    os.remove(stale_path)
            np.save(result_path, result.values.astype(np.float32))
            updated_manifest[col] = {
                "d_opt": d_opt,
                "n_rows": len(df),
                "data_hash": data_hash,
            }
        except Exception as e:
            tprint(f"Warning: FFD cache write failed for {col}: {e}")
            updated_manifest[col] = {
                "d_opt": d_opt,
                "n_rows": len(df),
                "data_hash": data_hash,
            }

        if (i + 1) % 5 == 0 or (i + 1) == total_cols:
            tprint(f"Adaptive FFD ({_label}): {i+1}/{total_cols} - {col}")

    tprint(
        f"Adaptive FFD ({_label}): cache_hit={stats['cached']}, "
        f"reused_d={stats['cached_d']}, full_compute={stats['computed']} "
        f"(total {total_cols})"
    )
    tprint(
        f"Adaptive FFD ({_label}): d range [{df_fd.min().min():.3f}, {df_fd.max().max():.3f}]"
    )
    try:
        with open(manifest_path, "wb") as f:
            pickle.dump(updated_manifest, f)
    except Exception as e:
        tprint(f"Warning: FFD manifest write failed for {_label}: {e}")
    return df_fd


@_cache.cache
def _transform_volume(df):
    tprint("Transforming Volume: Log -> EWMA(5)")
    df_log = np.log(df + 1.0)
    df_den = ff.numba_ewma(df_log, 2.0 / 6.0, False)
    return df_den


def time_sin_cos(index: pd.DatetimeIndex):
    hod = index.hour.to_numpy()
    dow = index.dayofweek.to_numpy()
    sin_hod = np.sin(2 * np.pi * hod / 24.0)
    cos_hod = np.cos(2 * np.pi * hod / 24.0)
    sin_dow = np.sin(2 * np.pi * dow / 7.0)
    cos_dow = np.cos(2 * np.pi * dow / 7.0)
    return sin_hod, cos_hod, sin_dow, cos_dow


def compute_orderbook_wall_primitives(
    orderbook_panel: Optional[pd.DataFrame],
    close_panel: pd.DataFrame,
    volume_panel: pd.DataFrame,
    atr_panel: Optional[pd.DataFrame] = None,
    shift_bars: int = 1,
) -> dict[str, pd.DataFrame]:
    """Compute side-agnostic order-book wall primitives and meta features.

    Gracefully returns zero-filled features when depth snapshots are unavailable.
    """
    eps = 1e-12
    idx, cols = close_panel.index, close_panel.columns
    out: dict[str, pd.DataFrame] = {}

    def z() -> pd.DataFrame:
        return pd.DataFrame(0.0, index=idx, columns=cols, dtype=np.float32)

    qv = (close_panel * volume_panel).replace([np.inf, -np.inf], np.nan)
    qv24 = qv.rolling(24, min_periods=1).mean().shift(1).fillna(0.0)

    bands = {"r005": 0.005, "r010": 0.01, "r020": 0.02, "r030": 0.03}
    for b in ("a05", "a10", "a20", "a30"):
        bands[b] = np.nan

    for b in bands:
        out[f"obw_wall_skew_book_{b}"] = z()
        out[f"obw_wall_skew_vol_{b}"] = z()
        out[f"obw_wall_pressure_skew_{b}"] = z()
        out[f"obw_band_depth_skew_vol_{b}"] = z()
        out[f"_obw_bid_wall_to_vol_{b}"] = z()
        out[f"_obw_ask_wall_to_vol_{b}"] = z()
        out[f"_obw_bid_wall_pressure_{b}"] = z()
        out[f"_obw_ask_wall_pressure_{b}"] = z()
        out[f"_obw_bid_wall_distance_{b}"] = z()
        out[f"_obw_ask_wall_distance_{b}"] = z()
        out[f"_obw_bid_path_depth_to_target_{b}"] = z()
        out[f"_obw_ask_path_depth_to_target_{b}"] = z()
        if b.startswith("r"):
            out[f"obw_wall_concentration_skew_{b}"] = z()

    out["obw_nearest_bid_wall_to_vol"] = z()
    out["obw_nearest_ask_wall_to_vol"] = z()
    out["obw_nearest_wall_skew_vol"] = z()
    out["obw_nearest_wall_distance_skew"] = z()

    # If snapshot panel is not available in the expected wide format, return neutral zeros.
    if not isinstance(orderbook_panel, pd.DataFrame) or orderbook_panel.empty:
        return out

    # Minimal fast path using available top-of-book cumulative proxies if present.
    req = {"cum_bid_qty_l20", "cum_ask_qty_l20", "best_bid", "best_ask"}
    if not req.issubset(set(orderbook_panel.columns)):
        return out

    ob = orderbook_panel.copy()
    ob.index = pd.to_datetime(ob.index, utc=True)
    ob = ob.reindex(idx).ffill()
    mid = (
        pd.to_numeric(ob["best_bid"], errors="coerce")
        + pd.to_numeric(ob["best_ask"], errors="coerce")
    ) * 0.5
    bid_depth = (mid * pd.to_numeric(ob["cum_bid_qty_l20"], errors="coerce")).fillna(
        0.0
    )
    ask_depth = (mid * pd.to_numeric(ob["cum_ask_qty_l20"], errors="coerce")).fillna(
        0.0
    )
    total_book = (bid_depth + ask_depth).replace(0.0, np.nan)

    # Broadcast symbolic single-series proxies across columns for now.
    def bc(ser: pd.Series) -> pd.DataFrame:
        arr = np.asarray(ser.fillna(0.0), dtype=np.float32)
        return pd.DataFrame(
            np.repeat(arr[:, None], len(cols), axis=1), index=idx, columns=cols
        )

    for b in bands:
        bid_w = 0.2 * bid_depth
        ask_w = 0.2 * ask_depth
        bid_to_vol = bid_w / (qv24.mean(axis=1) + eps)
        ask_to_vol = ask_w / (qv24.mean(axis=1) + eps)
        skew_book = (bid_w / (total_book + eps)) - (ask_w / (total_book + eps))
        skew_book = (
            skew_book
            / ((bid_w / (total_book + eps)) + (ask_w / (total_book + eps)) + eps)
        ).clip(-1, 1)
        skew_vol = ((bid_to_vol - ask_to_vol) / (bid_to_vol + ask_to_vol + eps)).clip(
            -1, 1
        )
        depth_skew = ((bid_depth - ask_depth) / (bid_depth + ask_depth + eps)).clip(
            -1, 1
        )

        out[f"obw_wall_skew_book_{b}"] = bc(skew_book).shift(shift_bars).fillna(0.0)
        out[f"obw_wall_skew_vol_{b}"] = bc(skew_vol).shift(shift_bars).fillna(0.0)
        out[f"obw_wall_pressure_skew_{b}"] = bc(skew_vol).shift(shift_bars).fillna(0.0)
        out[f"obw_band_depth_skew_vol_{b}"] = (
            bc(depth_skew).shift(shift_bars).fillna(0.0)
        )
        out[f"_obw_bid_wall_to_vol_{b}"] = (
            bc(np.log1p(bid_to_vol)).shift(shift_bars).fillna(0.0)
        )
        out[f"_obw_ask_wall_to_vol_{b}"] = (
            bc(np.log1p(ask_to_vol)).shift(shift_bars).fillna(0.0)
        )
        out[f"_obw_bid_wall_pressure_{b}"] = out[f"_obw_bid_wall_to_vol_{b}"]
        out[f"_obw_ask_wall_pressure_{b}"] = out[f"_obw_ask_wall_to_vol_{b}"]
        out[f"_obw_bid_wall_distance_{b}"] = z()
        out[f"_obw_ask_wall_distance_{b}"] = z()
        out[f"_obw_bid_path_depth_to_target_{b}"] = (
            bc(np.log1p(bid_depth / (qv24.mean(axis=1) + eps)))
            .shift(shift_bars)
            .fillna(0.0)
        )
        out[f"_obw_ask_path_depth_to_target_{b}"] = (
            bc(np.log1p(ask_depth / (qv24.mean(axis=1) + eps)))
            .shift(shift_bars)
            .fillna(0.0)
        )
        if b.startswith("r"):
            out[f"obw_wall_concentration_skew_{b}"] = (
                bc(
                    ((bid_w / (bid_depth + eps)) - (ask_w / (ask_depth + eps))).clip(
                        -1, 1
                    )
                )
                .shift(shift_bars)
                .fillna(0.0)
            )

    out["obw_nearest_bid_wall_to_vol"] = out["_obw_bid_wall_to_vol_r030"].copy()
    out["obw_nearest_ask_wall_to_vol"] = out["_obw_ask_wall_to_vol_r030"].copy()
    out["obw_nearest_wall_skew_vol"] = (
        (
            (out["obw_nearest_bid_wall_to_vol"] - out["obw_nearest_ask_wall_to_vol"])
            / (
                out["obw_nearest_bid_wall_to_vol"]
                + out["obw_nearest_ask_wall_to_vol"]
                + eps
            )
        )
        .clip(-1, 1)
        .astype(np.float32)
    )
    out["obw_nearest_wall_distance_skew"] = z()
    return out




def compute_orderbook_snapshot_features(orderbook_panel: Optional[pd.DataFrame], close_panel: pd.DataFrame, volume_panel: pd.DataFrame, atr_panel: Optional[pd.DataFrame], cfg: dict, shift_bars: int = 1) -> dict[str, pd.DataFrame]:
    """Compute causal per-symbol orderbook features from long-format L2 snapshots.

    Supported schema: timestamp, symbol, side, level, price, qty.
    """
    idx = pd.to_datetime(close_panel.index, utc=True)
    close_panel = close_panel.copy()
    close_panel.index = idx
    volume_panel = volume_panel.copy()
    volume_panel.index = pd.to_datetime(volume_panel.index, utc=True)
    volume_panel = volume_panel.reindex(index=idx, columns=close_panel.columns)
    if isinstance(atr_panel, pd.DataFrame):
        atr_panel = atr_panel.copy()
        atr_panel.index = pd.to_datetime(atr_panel.index, utc=True)
        atr_panel = atr_panel.reindex(index=idx, columns=close_panel.columns)
    idx, cols = close_panel.index, close_panel.columns
    depth_bps = [int(x) for x in cfg.get("orderbook_depth_bps", [5, 10, 25, 50, 100])]
    stale_hours = float(cfg.get("orderbook_stale_hours", 2))
    max_levels = int(cfg.get("orderbook_levels", 20))
    atr_panel = atr_panel if isinstance(atr_panel, pd.DataFrame) else pd.DataFrame(np.nan, index=idx, columns=cols)
    qv24 = (close_panel * volume_panel).rolling(24, min_periods=1).mean().shift(1)
    out: dict[str, pd.DataFrame] = {}

    def blank(v: float = 0.0) -> pd.DataFrame:
        return pd.DataFrame(v, index=idx, columns=cols, dtype=np.float32)

    keys = ["ob_available", "ob_snapshot_age_min", "ob_stale_flag", "ob_spread_bps", "ob_mid_vs_close_bps", "ob_l1_imbalance", "ob_l5_imbalance", "ob_l10_imbalance", "ob_l20_imbalance", "ob_microprice_premium_bps", "ob_bid_slope_20", "ob_ask_slope_20", "ob_bid_wall_found", "ob_ask_wall_found", "ob_nearest_bid_wall_dist_bps", "ob_nearest_ask_wall_dist_bps", "ob_nearest_bid_wall_dist_atr", "ob_nearest_ask_wall_dist_atr", "ob_nearest_bid_wall_to_qv24", "ob_nearest_ask_wall_to_qv24", "ob_liquidity_void_up_bps", "ob_liquidity_void_down_bps", "ob_max_gap_up_bps", "ob_max_gap_down_bps", "ob_imbalance_delta_1h", "ob_spread_delta_1h", "ob_depth_skew_delta_1h"]
    for bps in depth_bps:
        keys += [f"ob_bid_depth_{bps}bps", f"ob_ask_depth_{bps}bps", f"ob_depth_skew_{bps}bps"]
    for k in keys:
        out[k] = blank(0.0)
    out["ob_stale_flag"] = blank(1.0)
    out["ob_snapshot_age_min"] = blank(float(cfg.get("orderbook_missing_age_sentinel_min", np.nan)))

    if not isinstance(orderbook_panel, pd.DataFrame) or orderbook_panel.empty:
        return out
    if not {"timestamp", "symbol", "side", "level", "price", "qty"}.issubset(orderbook_panel.columns):
        raise ValueError("orderbook_hourly must be long format: timestamp,symbol,side,level,price,qty")

    ob = orderbook_panel.copy()
    ob["timestamp"] = pd.to_datetime(ob["timestamp"], utc=True, errors="coerce")
    ob["level"] = pd.to_numeric(ob["level"], errors="coerce").astype("Int64")
    ob["price"] = pd.to_numeric(ob["price"], errors="coerce")
    ob["qty"] = pd.to_numeric(ob["qty"], errors="coerce")
    ob["side"] = ob["side"].astype(str).str.lower()
    ob = ob.dropna(subset=["timestamp", "symbol", "side", "level", "price", "qty"])
    ob = ob[(ob["level"] > 0) & (ob["level"] <= max_levels)]

    for sym in cols:
        ss = ob[ob["symbol"] == sym].sort_values(["timestamp", "side", "level"])
        if ss.empty:
            continue
        recs: list[dict[str, float | pd.Timestamp]] = []
        for ts, g in ss.groupby("timestamp"):
            bids = g[g["side"].str.startswith("b")].sort_values("price", ascending=False).head(max_levels)
            asks = g[g["side"].str.startswith("a")].sort_values("price", ascending=True).head(max_levels)
            if bids.empty or asks.empty:
                continue
            bb, ba = float(bids.iloc[0]["price"]), float(asks.iloc[0]["price"])
            mid = (bb + ba) * 0.5
            rec: dict[str, float | pd.Timestamp] = {"timestamp": ts, "_snapshot_ts": ts, "bb": bb, "ba": ba, "mid": mid, "bq1": float(bids.iloc[0]["qty"]), "aq1": float(asks.iloc[0]["qty"])}
            for lvl in (1, 5, 10, 20):
                rec[f"bi{lvl}"] = float(bids.head(lvl)["qty"].sum())
                rec[f"ai{lvl}"] = float(asks.head(lvl)["qty"].sum())
            for bps in depth_bps:
                bthr = mid * (1.0 - bps / 1e4)
                athr = mid * (1.0 + bps / 1e4)
                rec[f"bd{bps}"] = float((bids.loc[bids["price"] >= bthr, "price"] * bids.loc[bids["price"] >= bthr, "qty"]).sum())
                rec[f"ad{bps}"] = float((asks.loc[asks["price"] <= athr, "price"] * asks.loc[asks["price"] <= athr, "qty"]).sum())
            wall_qty_mult = float(cfg.get("orderbook_wall_qty_mult", 3.0))
            bid_thresh = float(bids["qty"].median()) * wall_qty_mult
            ask_thresh = float(asks["qty"].median()) * wall_qty_mult
            bid_cands = bids[bids["qty"] >= bid_thresh]
            ask_cands = asks[asks["qty"] >= ask_thresh]
            wall_bid_found = float(not bid_cands.empty)
            wall_ask_found = float(not ask_cands.empty)
            wall_bid = bid_cands.iloc[0] if wall_bid_found > 0 else bids.iloc[0]
            wall_ask = ask_cands.iloc[0] if wall_ask_found > 0 else asks.iloc[0]
            rec["nwb_bps"] = max((mid - float(wall_bid["price"])) / mid * 1e4, 0.0)
            rec["nwa_bps"] = max((float(wall_ask["price"]) - mid) / mid * 1e4, 0.0)
            gaps_ask = np.abs(np.diff(np.sort(asks["price"].to_numpy())))
            gaps_bid = np.abs(np.diff(np.sort(bids["price"].to_numpy())[::-1]))
            rec["void_up_bps"] = float(np.nanmax(gaps_ask) / mid * 1e4) if len(gaps_ask) else 0.0
            rec["void_dn_bps"] = float(np.nanmax(gaps_bid) / mid * 1e4) if len(gaps_bid) else 0.0
            rec["bid_slope"] = float(np.polyfit(np.arange(1, len(bids) + 1), (bids["price"].to_numpy() / (mid + 1e-9)) * 1e4, 1)[0]) if len(bids) > 1 else 0.0
            rec["ask_slope"] = float(np.polyfit(np.arange(1, len(asks) + 1), (asks["price"].to_numpy() / (mid + 1e-9)) * 1e4, 1)[0]) if len(asks) > 1 else 0.0
            rec["wall_bid_notional"] = float(wall_bid["price"] * wall_bid["qty"])
            rec["wall_ask_notional"] = float(wall_ask["price"] * wall_ask["qty"])
            rec["wall_bid_found"] = wall_bid_found
            rec["wall_ask_found"] = wall_ask_found
            recs.append(rec)
        if not recs:
            continue
        sdf = pd.DataFrame(recs).set_index("timestamp").sort_index()
        mapped = sdf.reindex(idx, method="ffill")
        snap_ts = pd.to_datetime(mapped["_snapshot_ts"]).shift(shift_bars)
        age_min = (pd.Series(idx, index=idx) - snap_ts).dt.total_seconds() / 60.0
        stale = ((age_min > stale_hours * 60) | age_min.isna()).astype(np.float32)
        avail = ((~stale.astype(bool)) & mapped["bb"].notna() & mapped["ba"].notna()).astype(np.float32)
        m = mapped.shift(shift_bars)
        out["ob_available"][sym] = avail.values
        out["ob_snapshot_age_min"][sym] = age_min.values
        out["ob_stale_flag"][sym] = stale.values
        out["ob_spread_bps"][sym] = (((m["ba"] - m["bb"]) / (m["mid"] + 1e-9)) * 1e4).fillna(0).values
        out["ob_mid_vs_close_bps"][sym] = (((m["mid"] / (close_panel[sym] + 1e-9)) - 1.0) * 1e4).fillna(0).values
        out["ob_l1_imbalance"][sym] = ((m["bi1"] - m["ai1"]) / (m["bi1"] + m["ai1"] + 1e-9)).fillna(0).values
        out["ob_l5_imbalance"][sym] = ((m["bi5"] - m["ai5"]) / (m["bi5"] + m["ai5"] + 1e-9)).fillna(0).values
        out["ob_l10_imbalance"][sym] = ((m["bi10"] - m["ai10"]) / (m["bi10"] + m["ai10"] + 1e-9)).fillna(0).values
        out["ob_l20_imbalance"][sym] = ((m["bi20"] - m["ai20"]) / (m["bi20"] + m["ai20"] + 1e-9)).fillna(0).values
        out["ob_microprice_premium_bps"][sym] = ((((m["ba"] * m["bq1"] + m["bb"] * m["aq1"]) / (m["bq1"] + m["aq1"] + 1e-9)) / (m["mid"] + 1e-9) - 1.0) * 1e4).fillna(0).values
        for bps in depth_bps:
            out[f"ob_bid_depth_{bps}bps"][sym] = m[f"bd{bps}"].fillna(0).values
            out[f"ob_ask_depth_{bps}bps"][sym] = m[f"ad{bps}"].fillna(0).values
            bd = out[f"ob_bid_depth_{bps}bps"][sym]
            ad = out[f"ob_ask_depth_{bps}bps"][sym]
            out[f"ob_depth_skew_{bps}bps"][sym] = ((bd - ad) / (bd + ad + 1e-9)).values
        out["ob_bid_slope_20"][sym] = m["bid_slope"].fillna(0).values
        out["ob_ask_slope_20"][sym] = m["ask_slope"].fillna(0).values
        out["ob_bid_wall_found"][sym] = m["wall_bid_found"].fillna(0).values
        out["ob_ask_wall_found"][sym] = m["wall_ask_found"].fillna(0).values
        out["ob_nearest_bid_wall_dist_bps"][sym] = m["nwb_bps"].fillna(0).values
        out["ob_nearest_ask_wall_dist_bps"][sym] = m["nwa_bps"].fillna(0).values
        # atr_panel is expected to be ATR as fraction-of-price (atr_pct); convert to bps.
        atr_bps = (atr_panel[sym] * 1e4)
        out["ob_nearest_bid_wall_dist_atr"][sym] = (m["nwb_bps"] / (atr_bps + 1e-9)).fillna(0).values
        out["ob_nearest_ask_wall_dist_atr"][sym] = (m["nwa_bps"] / (atr_bps + 1e-9)).fillna(0).values
        out["ob_nearest_bid_wall_to_qv24"][sym] = (m["wall_bid_notional"] / (qv24[sym] + 1e-9)).fillna(0).values
        out["ob_nearest_ask_wall_to_qv24"][sym] = (m["wall_ask_notional"] / (qv24[sym] + 1e-9)).fillna(0).values
        out["ob_liquidity_void_up_bps"][sym] = m["void_up_bps"].fillna(0).values
        out["ob_liquidity_void_down_bps"][sym] = m["void_dn_bps"].fillna(0).values
        out["ob_max_gap_up_bps"][sym] = out["ob_liquidity_void_up_bps"][sym].values
        out["ob_max_gap_down_bps"][sym] = out["ob_liquidity_void_down_bps"][sym].values
        valid = out["ob_available"][sym].astype(bool)
        dependent_ob_features = [k for k in out if k not in {"ob_available", "ob_snapshot_age_min", "ob_stale_flag"}]
        for feat_name in dependent_ob_features:
            out[feat_name].loc[~valid, sym] = 0.0
    avail = out["ob_available"].astype(bool)
    out["ob_imbalance_delta_1h"] = out["ob_l10_imbalance"].diff(1).where(avail & avail.shift(1), 0.0).astype(np.float32)
    out["ob_spread_delta_1h"] = out["ob_spread_bps"].diff(1).where(avail & avail.shift(1), 0.0).astype(np.float32)
    ds_key = f"ob_depth_skew_{depth_bps[min(len(depth_bps) - 1, 2)]}bps"
    out["ob_depth_skew_delta_1h"] = out[ds_key].diff(1).where(avail & avail.shift(1), 0.0).astype(np.float32)
    return {k: v.astype(np.float32) for k, v in out.items()}


def compute_market_features(panel, basket_syms, trend_sma_hours=24 * 14):
    tprint("Entering function: compute_market_features in features.py")
    c = panel["close"]
    h = panel["high"]
    l = panel["low"]
    v = panel["volume"]

    basket = [s for s in basket_syms if s in c.columns]
    if not basket:
        basket = list(c.columns)

    mkt_close_raw = c[basket].mean(axis=1)
    mkt_high_raw = h[basket].mean(axis=1)
    mkt_low_raw = l[basket].mean(axis=1)
    mkt_vol_raw = v[basket].mean(axis=1)

    mkt_close = ff.numba_ewma(
        _safe_log_df(mkt_close_raw.to_frame(name="c")), 2.0 / 6.0, False
    )["c"]
    mkt_high = ff.numba_ewma(
        _safe_log_df(mkt_high_raw.to_frame(name="h")), 2.0 / 6.0, False
    )["h"]
    mkt_low = ff.numba_ewma(
        _safe_log_df(mkt_low_raw.to_frame(name="l")), 2.0 / 6.0, False
    )["l"]
    mkt_vol = _transform_volume(mkt_vol_raw.to_frame(name="v"))["v"]

    mkt_ret24h_df = ff.numba_rolling_sum(mkt_close.to_frame(), 24)
    mkt_ret24h = mkt_ret24h_df[mkt_ret24h_df.columns[0]]

    mkt_ret6h_df = ff.numba_rolling_sum(mkt_close.to_frame(), 6)
    mkt_ret6h = mkt_ret6h_df[mkt_ret6h_df.columns[0]]

    sma_df = ff.numba_rolling_mean(mkt_close.to_frame(), trend_sma_hours)
    sma = sma_df[sma_df.columns[0]]

    mkt_trend = mkt_close - sma
    mkt_ret1h = mkt_close

    mkt_rv_df = ff.numba_rolling_std(mkt_ret1h.to_frame(), 24)
    mkt_rv = mkt_rv_df[mkt_rv_df.columns[0]]

    mkt_df = pd.DataFrame(
        {
            "mkt_close": mkt_close,
            "mkt_high": mkt_high,
            "mkt_low": mkt_low,
            "mkt_volume": mkt_vol,
            "mkt_ret24h": mkt_ret24h,
            "mkt_ret6h": mkt_ret6h,
            "mkt_trend": mkt_trend,
            "mkt_rv": mkt_rv,
        }
    )
    return mkt_df.astype(np.float32)


def add_regime_gates(
    mkt_df: pd.DataFrame, gate_vol_lookback_hours: int, gate_trend_thr: float
):
    tprint("Entering function: add_regime_gates in features.py")
    df = mkt_df.copy()
    rv_med_df = ff.numba_rolling_median(df[["mkt_rv"]], gate_vol_lookback_hours)
    df["mkt_rv_med"] = rv_med_df["mkt_rv"]

    df["G_VOL"] = (df["mkt_rv"] > df["mkt_rv_med"]).astype(np.int32)

    # Dynamic Trend Threshold (Vol-Adjusted) to ensure variation
    # Fixed 0.02 is too high for low-vol regimes.
    # Use 1.5 * Daily Volatility (approx 1.5 sigma move)
    daily_vol = df["mkt_rv"] * np.float32(np.sqrt(24))
    # Use dynamic threshold but floor it at small value to avoid noise in 0 vol
    dyn_thr = np.maximum(daily_vol * 1.5, 0.005)

    df["G_TREND"] = (df["mkt_ret24h"].abs() > dyn_thr).astype(np.int32)
    safe_mkt_rv_med = np.where(df["mkt_rv_med"] > 1e-12, df["mkt_rv_med"], 1e-12)
    df["mkt_rv_ratio"] = df["mkt_rv"] / safe_mkt_rv_med

    rv_mean = ff.numba_rolling_mean(df[["mkt_rv"]], gate_vol_lookback_hours)[
        "mkt_rv"
    ].shift(1)
    rv_std = (
        ff.numba_rolling_std(df[["mkt_rv"]], gate_vol_lookback_hours)["mkt_rv"]
        .shift(1)
        .clip(lower=1e-6)
    )
    df["mkt_rv_pct"] = (
        ((df["mkt_rv"] - rv_mean) / rv_std).clip(-6, 6).fillna(0.0).astype(np.float32)
    )

    df["mkt_rv_pct"] = (
        0.5 * (1.0 + scipy.special.erf(df["mkt_rv_pct"] / np.sqrt(2.0)))
    ).astype(np.float32)

    abs_ret = df["mkt_ret24h"].abs()
    abs_ret_mean = ff.numba_rolling_mean(
        abs_ret.to_frame("x"), gate_vol_lookback_hours
    )["x"].shift(1)
    abs_ret_std = (
        ff.numba_rolling_std(abs_ret.to_frame("x"), gate_vol_lookback_hours)["x"]
        .shift(1)
        .clip(lower=1e-6)
    )
    df["abs_mkt_ret24h_z"] = (
        ((abs_ret - abs_ret_mean) / abs_ret_std)
        .clip(-6, 6)
        .fillna(0.0)
        .astype(np.float32)
    )
    mkt_vol_mean = ff.numba_rolling_mean(df[["mkt_volume"]], gate_vol_lookback_hours)[
        "mkt_volume"
    ].shift(1)
    mkt_vol_std = (
        ff.numba_rolling_std(df[["mkt_volume"]], gate_vol_lookback_hours)["mkt_volume"]
        .shift(1)
        .clip(lower=1e-6)
    )
    df["mkt_volume_z_24"] = (
        ((df["mkt_volume"] - mkt_vol_mean) / mkt_vol_std)
        .clip(-6, 6)
        .fillna(0.0)
        .astype(np.float32)
    )
    df["regime_trend_score"] = df["abs_mkt_ret24h_z"].astype(np.float32)
    df["regime_vol_score"] = df["mkt_rv_pct"].astype(np.float32)
    df["regime_liquidity_score"] = df["mkt_volume_z_24"].astype(np.float32)

    float_cols = [
        "mkt_rv_med",
        "mkt_rv_ratio",
        "mkt_rv_pct",
        "abs_mkt_ret24h_z",
        "mkt_volume_z_24",
        "regime_trend_score",
        "regime_vol_score",
        "regime_liquidity_score",
    ]
    for c in float_cols:
        df[c] = df[c].astype(np.float32)

    return df


def compute_vol_regime_features(
    close_df: pd.DataFrame,
    vol_window: int = 24,
    pct_window: int = 252,
    rv_cache: pd.DataFrame = None,
):
    """Compute volatility-regime features from close prices."""
    if rv_cache is not None:
        rv = rv_cache
    else:
        # Use np.maximum to avoid log(0) or negative values in the ratio
        ratio = close_df / close_df.shift(1)
        ratio_safe = np.maximum(ratio, 1e-9)
        ret = np.log(ratio_safe)
        rv = (ff.numba_rolling_std(ret, vol_window)).shift(1)

    vol_pct = ff.numba_rolling_rank_pct(rv, pct_window).clip(0.0, 1.0)
    vol_high = (vol_pct - 0.8).clip(lower=0.0)
    vol_low = (0.2 - vol_pct).clip(lower=0.0)

    return (
        vol_pct.astype(np.float32),
        vol_high.astype(np.float32),
        vol_low.astype(np.float32),
    )


def compute_cusum_regime_features(cusum_strength_df: pd.DataFrame, h: float):
    """Normalize cusum strength and expose a high-regime hinge."""
    cusum_strength_norm = (cusum_strength_df / (h + EPS)).clip(lower=0.0)
    cusum_high = (cusum_strength_norm - 1.0).clip(lower=0.0)
    return cusum_strength_norm.astype(np.float32), cusum_high.astype(np.float32)


def compute_liquidity_features(volume_df: pd.DataFrame, avg_window: int = 720):
    """Compute liquidity ratios from volume relative to lagged rolling baseline."""
    vol_avg = volume_df.rolling(avg_window).mean().shift(1)
    liq_ratio = volume_df / (vol_avg + EPS)
    liq_low = (1.0 - liq_ratio).clip(lower=0.0).astype(np.float32)
    liq_low = liq_low.clip(lower=0.0)
    return liq_ratio.astype(np.float32), liq_low


def add_interactions(
    p_success_df: pd.DataFrame,
    vol_high: pd.DataFrame,
    cusum_high: pd.DataFrame,
    liq_low: pd.DataFrame,
):
    """Interaction terms between success probability signal and regime shocks."""
    return {
        "p_vol_high": (p_success_df * vol_high).astype(np.float32),
        "p_cusum_high": (p_success_df * cusum_high).astype(np.float32),
        "p_liq_low": (p_success_df * liq_low).astype(np.float32),
    }


def _robust_obs_var_per_col(df: pd.DataFrame) -> np.ndarray:
    """Robust baseline observation variance estimate per column from first differences."""
    arr = np.ascontiguousarray(df.to_numpy(dtype=np.float64))
    return _robust_obs_var_per_col_nb(arr)


def _kalman_local_level_df(
    y_df: pd.DataFrame, lambda_qr: float, r_base: np.ndarray | None = None
):
    """Local-level Kalman filter: y_t = x_t + eps_t, x_t = x_{t-1} + eta_t."""
    y = np.ascontiguousarray(y_df.to_numpy(dtype=np.float64))
    r = (
        _robust_obs_var_per_col(y_df)
        if r_base is None
        else np.asarray(r_base, dtype=np.float64)
    )
    r = np.clip(r, 1e-8, None)
    x, innov_var, p_state = _kalman_local_level_nb(
        y, np.clip(lambda_qr, 1e-8, None) * r, r
    )

    return (
        pd.DataFrame(x, index=y_df.index, columns=y_df.columns).astype(np.float32),
        pd.DataFrame(innov_var, index=y_df.index, columns=y_df.columns).astype(
            np.float32
        ),
        pd.DataFrame(p_state, index=y_df.index, columns=y_df.columns).astype(
            np.float32
        ),
        pd.Series(r.astype(np.float32), index=y_df.columns),
    )


def _decile_monotonicity_score(signal_df: pd.DataFrame, ret_df: pd.DataFrame) -> float:
    """Cross-sectional decile monotonicity score using mean return per decile."""
    s = np.ascontiguousarray(signal_df.to_numpy(dtype=np.float64))
    r = np.ascontiguousarray(ret_df.to_numpy(dtype=np.float64))
    return float(_decile_monotonicity_score_nb(s, r))


def _rolling_autocorr_df(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Fast rolling lag-1 autocorrelation using the shared Numba correlation kernel."""
    return (
        ff.numba_rolling_corr(df, df.shift(1), int(window))
        .fillna(0.0)
        .astype(np.float32)
    )


def _turnover_penalty(signal_df: pd.DataFrame) -> float:
    arr = signal_df.to_numpy(dtype=np.float64)
    sd = np.nanstd(arr, axis=0)
    z = arr / np.clip(sd, 1e-6, None)
    pos = np.tanh(z)
    dpos = np.abs(np.diff(pos, axis=0))
    return float(np.nanmean(dpos)) if dpos.size else 0.0


def tune_global_kalman_lambda(
    score_df: pd.DataFrame, net_ret_df: pd.DataFrame, grid_size: int = 15
) -> float:
    """Tune global lambda=Q/R via decile monotonicity with mild turnover penalty on subsample."""
    n_t, n_c = score_df.shape
    row_step = max(1, n_t // 1500)
    col_step = max(1, n_c // 64)
    score_sub = score_df.iloc[::row_step, ::col_step]
    ret_sub = net_ret_df.reindex(score_sub.index).iloc[:, ::col_step]

    r_base = _robust_obs_var_per_col(score_sub)
    lam_grid = np.logspace(-3, 1, int(np.clip(grid_size, 10, 20)))

    best_lam = float(lam_grid[len(lam_grid) // 2])
    best_obj = -1e18
    for lam in lam_grid:
        state_df, _, _, _ = _kalman_local_level_df(
            score_sub, lambda_qr=float(lam), r_base=r_base
        )
        mono = _decile_monotonicity_score(state_df, ret_sub)
        turn = _turnover_penalty(state_df)
        obj = mono - 0.05 * turn
        if obj > best_obj:
            best_obj = obj
            best_lam = float(lam)

    return float(best_lam)


def compute_regime_features(
    c, h, l, v, atr_base, mkt_gates, rv_24_cache=None, input_feats=None
):
    """
    Compute regime conditioning features (cusum, vol, etc.).
    Returns a dict of new features.
    """
    feats = {}

    # 1. CUSUM Strength (Trend Persistence)
    # Detects if price is persistently drifting away from mean
    # Normalized by volatility
    # --- Pipeline Hardening: Resolve return features from input_feats or recompute ---
    ret1h = (
        input_feats["ret1h"]
        if (input_feats is not None and "ret1h" in input_feats)
        else c.diff(1).fillna(0.0)
    )
    ret12h = (
        input_feats["ret12h"]
        if (input_feats is not None and "ret12h" in input_feats)
        else ff.numba_rolling_sum(c, 12)
    )

    if rv_24_cache is not None:
        rv_24 = rv_24_cache
    else:
        rv_24 = ff.numba_rolling_std(ret1h, 24)
    std_ret = rv_24 + np.float32(1e-12)

    # Vectorized approximation: Rolling Sum of (Ret - Mean) / Vol
    # This captures local trend strength
    # Shift by 1 to ensure decision-time causality (no current-bar leakage).
    roll_z = (
        ff.numba_rolling_mean(ret1h / std_ret, 24) * np.float32(np.sqrt(24))
    ).shift(1)
    feats["cusum_strength"] = roll_z.astype(np.float32)

    # 2. Standardized Move Magnitude |z| (over 24h)
    ret_24 = ff.numba_rolling_sum(ret1h, 24)
    feats["move_magnitude_z"] = (
        (ret_24 / (rv_24 * np.float32(np.sqrt(24)) + 1e-12)).shift(1).astype(np.float32)
    )

    adx_raw, _, _ = ff.numba_adx(h, l, c, 14)
    adx_mean = ff.numba_rolling_mean(adx_raw, 24 * 14).shift(1)
    adx_std = ff.numba_rolling_std(adx_raw, 24 * 14).shift(1).clip(lower=1e-6)
    feats["adx_zscore"] = (
        ((adx_raw - adx_mean) / adx_std).clip(-6, 6).fillna(0.0).astype(np.float32)
    )

    # 3. Time Since CUSUM Trigger (Trend Age Proxy)
    # Trigger when |cusum| > 5 based on lagged signal only.
    is_trigger = (feats["cusum_strength"].abs() > 5.0).astype(np.float32)
    # Count bars since last trigger (decay proxy)
    feats["cusum_decay"] = (
        ff.numba_ewma(is_trigger, 2.0 / 25.0, False).shift(1).astype(np.float32)
    )

    # 4. Volatility percentile and hinges
    vol_pct, vol_high, vol_low = compute_vol_regime_features(
        c,
        vol_window=24,
        pct_window=252,
        rv_cache=rv_24.shift(1) if rv_24_cache is not None else None,
    )
    feats["vol_percentile"] = vol_pct
    feats["vol_high"] = vol_high
    feats["vol_low"] = vol_low

    # 5. Vol of Vol (Rolling Std of Sigma)
    # Coefficient of variation of volatility
    vv = ff.numba_rolling_std(rv_24, 16)
    mean_rv_24_16 = ff.numba_rolling_mean(rv_24, 16)
    safe_mean_rv_24_16 = np.where(mean_rv_24_16 > 1e-12, mean_rv_24_16, 1e-12)
    feats["vol_regime_shift"] = (
        (ff.numba_rolling_mean(rv_24, 4) / safe_mean_rv_24_16)
        .shift(1)
        .astype(np.float32)
    )
    feats["vol_of_vol"] = (vv / (rv_24 + 1e-12)).shift(1).astype(np.float32)

    # 6. ATR Percentile (similar to vol percentile but using ATR)
    atr_min = ff.numba_rolling_min(atr_base, 24 * 30)
    atr_max = ff.numba_rolling_max(atr_base, 24 * 30)
    atr_range = atr_max - atr_min
    safe_atr_range = np.where(atr_range > 1e-12, atr_range, 1e-12)
    feats["atr_percentile"] = (
        ((atr_base - atr_min) / safe_atr_range).clip(0, 1).shift(1).astype(np.float32)
    )

    # 7. Liquidity ratio and low-liquidity hinge
    liq_ratio, liq_low = compute_liquidity_features(v, avg_window=24 * 30)
    feats["liquidity_ratio"] = liq_ratio
    feats["liq_low"] = (
        liq_low.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0)
    )
    # 8. CUSUM normalization and high-regime hinge
    cusum_strength_norm, cusum_high = compute_cusum_regime_features(
        feats["cusum_strength"].abs(), h=6.0
    )
    feats["cusum_strength_norm"] = cusum_strength_norm
    feats["cusum_high"] = cusum_high
    feats["regime_trend_score"] = (
        mkt_gates["regime_trend_score"].reindex(c.index).astype(np.float32)
    )
    feats["regime_vol_score"] = (
        mkt_gates["regime_vol_score"].reindex(c.index).astype(np.float32)
    )
    feats["regime_liquidity_score"] = (
        mkt_gates["regime_liquidity_score"].reindex(c.index).astype(np.float32)
    )

    assert float(feats["vol_percentile"].max().max()) <= 1.0 + np.float32(1e-6)
    assert float(feats["vol_percentile"].min().min()) >= -1e-6
    assert float(feats["vol_high"].min().min()) >= -1e-6
    assert float(feats["vol_low"].min().min()) >= -1e-6
    assert float(feats["cusum_high"].min().min()) >= -1e-6
    assert float(feats["liq_low"].min().min()) >= -1e-6

    # --- New Features ---
    # Meta
    # trendiness = variance(return_12h) / (12 * variance(return_1h))
    # Pipeline Hardening: Use locally resolved ret1h/ret12h variables
    ret12h_var = ff.numba_rolling_std(ret12h.to_numpy(), 48).astype(np.float32) ** 2
    ret1h_var = ff.numba_rolling_std(ret1h.to_numpy(), 48).astype(np.float32) ** 2
    feats["trendiness"] = (ret12h_var / (12 * ret1h_var + 1e-12)).astype(np.float32)

    # seasonality_strength = |return_this_hour − hourly_avg_return|
    # hourly avg return can be rolling mean of ret1h
    hourly_avg_ret = ff.apply_to_frame(
        ret1h, ff._numba_rolling_mean_nan_safe, 24
    ).astype(np.float32)
    feats["seasonality_strength"] = (ret1h - hourly_avg_ret).abs().astype(np.float32)

    # hour_vol / rolling_daily_vol
    # rolling_daily_vol can be ff.numba_rolling_std(ret1h, 96)
    hour_vol = ff.apply_to_frame(ret1h, ff._numba_rolling_std_nan_safe, 4).astype(
        np.float32
    )
    feats["hour_vol_ratio"] = (hour_vol / (rv_24 + 1e-12)).astype(np.float32)

    # jump_intensity = rolling_mean(jump_t, window=48)
    jump_t = (ret1h.abs() > 3 * rv_24).astype(np.float32)
    feats["jump_intensity"] = ff.numba_rolling_mean(jump_t.to_numpy(), 48).astype(
        np.float32
    )

    # vol_regime = short_vol / long_vol
    short_vol = ff.apply_to_frame(ret1h, ff._numba_rolling_std_nan_safe, 12).astype(
        np.float32
    )
    long_vol = rv_24
    feats["vol_regime_ratio"] = (short_vol / (long_vol + 1e-12)).astype(np.float32)

    # Base
    # trend_strength = |EMA_fast − EMA_slow| / rolling_vol
    # we already have ema(c, 12) and ema(c, 24) perhaps? Let's compute them locally.
    ema_fast = ff.numba_ema_nan_safe(c.to_numpy(), 12)
    ema_fast = pd.DataFrame(ema_fast, index=c.index, columns=c.columns)
    ema_slow = ff.numba_ema_nan_safe(c.to_numpy(), 50)
    ema_slow = pd.DataFrame(ema_slow, index=c.index, columns=c.columns)

    feats["trend_strength"] = ((ema_fast - ema_slow).abs() / (rv_24 + 1e-12)).astype(
        np.float32
    )

    # log(volume / rolling_volume)
    rolling_volume = ff.apply_to_frame(v, ff._numba_rolling_mean_nan_safe, 24).astype(
        np.float32
    )
    feats["volume_surge"] = np.log(
        (v / (rolling_volume + 1e-12)).clip(lower=1e-5)
    ).astype(np.float32)
    return feats


def compute_funding_proxy(c, h, l, v, mkt_df):
    c_ma = ff.numba_rolling_mean(c, 24)
    dist = c - c_ma

    mkt_close_df = mkt_df[["mkt_close"]]
    mkt_ma_df = ff.numba_rolling_mean(mkt_close_df, 24)
    mkt_dist = mkt_df["mkt_close"] - mkt_ma_df["mkt_close"]

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
    if requested_feature_keys is None:
        import extreme_price_movements.config as cfg_mod
        import extreme_price_movements.training_utils as tu

        all_keys = set()

        # Base features
        all_keys.update(tu.get_base_feature_keys("long", cfg))
        all_keys.update(tu.get_base_feature_keys("short", cfg))

        # Meta features
        for head in ["reg", "clf", "mfe", "mae", "asym"]:
            all_keys.update(tu.get_meta_feature_keys(head, cfg))

        # Other runtimes
        for group in [
            "RIDGE_FEATURE_COLS",
            "CONTINUOUS_LOCATION_COLS",
            "TEST_FEATURE_KEYS",
            "FEATURE_SELECTION_KEYS",
            "TRAINING_RESIDUALIZATION_FEATURE_KEYS",
            "MODEL_FEATURES",
        ]:
            if group in cfg_mod.CFG:
                all_keys.update(tu.expand_feature_group_refs(cfg_mod.CFG[group], cfg))
            elif hasattr(cfg_mod, group):
                all_keys.update(
                    tu.expand_feature_group_refs(getattr(cfg_mod, group), cfg)
                )

        requested_feature_keys = list(all_keys)

    return _compute_features_impl(
        panel, mkt_gates, cfg, requested_feature_keys=requested_feature_keys
    )


def _compute_hvn_col(col, o_col, h_col, l_col, c_col, v_col):
    from .volume_node_features import hvn_lvn_features_ohlcv

    df_col = pd.DataFrame(
        {"open": o_col, "high": h_col, "low": l_col, "close": c_col, "volume": v_col}
    )
    return col, hvn_lvn_features_ohlcv(df_col)


def _compute_hvn_batch(
    compute_col_fn, cols, o_batch, h_batch, l_batch, c_batch, v_batch
):
    return [
        compute_col_fn(
            col, o_batch[col], h_batch[col], l_batch[col], c_batch[col], v_batch[col]
        )
        for col in cols
    ]


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

    # Create a dict of dicts instead of incrementally updating pandas DataFrames
    hvn_raw_dict: dict[str, dict[str, np.ndarray]] = {k: {} for k in hvn_keys}
    total_cols = len(c_log.columns)

    def _assign_hvn_result(col_name, res_df):
        for k in hvn_keys:
            hvn_raw_dict[k][col_name] = res_df[k].to_numpy(dtype=np.float32)

    import multiprocessing

    max_workers = min(8, multiprocessing.cpu_count())

    main_path = getattr(__import__("__main__"), "__file__", "") or ""
    if not main_path or main_path == "<stdin>" or not os.path.exists(main_path):
        tprint(
            "HVN/LVN: process pool disabled because the current Python entrypoint "
            "is not importable by spawn workers."
        )
        can_use_process_pool = False
    else:
        can_use_process_pool = True

    try:
        os.sysconf("SC_SEM_NSEMS_MAX")
    except (AttributeError, ValueError, OSError, PermissionError):
        can_use_process_pool = False

    completed = 0
    batch_size = max(1, min(16, total_cols // max(max_workers, 1) or 1))
    col_batches = [
        list(c_log.columns[i : i + batch_size])
        for i in range(0, total_cols, batch_size)
    ]
    if can_use_process_pool and total_cols > 1:
        try:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                for cols in col_batches:
                    futures.append(
                        executor.submit(
                            _compute_hvn_batch,
                            compute_col_fn,
                            cols,
                            o[cols],
                            h[cols],
                            l[cols],
                            c_log[cols],
                            v[cols],
                        )
                    )

                for future in as_completed(futures):
                    for col, res_df in future.result():
                        _assign_hvn_result(col, res_df)
                        completed += 1
                        if completed % 50 == 0:
                            tprint(f"HVN/LVN: {completed}/{total_cols}")
        except Exception as e:
            tprint(
                f"HVN/LVN: process pool unavailable ({e}); falling back to single-process."
            )
            can_use_process_pool = False

    if not can_use_process_pool or total_cols <= 1:
        if total_cols > 1:
            tprint("HVN/LVN: using single-process fallback.")
        for cols in col_batches:
            for col, res_df in _compute_hvn_batch(
                compute_col_fn, cols, o[cols], h[cols], l[cols], c_log[cols], v[cols]
            ):
                _assign_hvn_result(col, res_df)
                completed += 1
                if completed % 50 == 0:
                    tprint(f"HVN/LVN: {completed}/{total_cols}")

    # Build final DataFrames efficiently from complete dictionaries
    hvn_results = {}
    for k in hvn_keys:
        hvn_results[k] = pd.DataFrame(hvn_raw_dict[k], index=c_log.index).reindex(
            columns=c_log.columns
        )

    return hvn_results


def _safe_div(a, b, eps=1e-12):
    return a / (b + eps)


def _safe_log_ratio(a, b, eps=1e-12):
    return np.log((a + eps) / (b + eps))


def _signed_log1p(x):
    return np.sign(x) * np.log1p(np.abs(x))


def _compute_features_impl(panel, mkt_gates, cfg, requested_feature_keys=None):
    tprint("Features: compute base matrices")
    requested_feature_set = set(requested_feature_keys or [])

    def _needs_feature(*keys: str) -> bool:
        return (not requested_feature_set) or any(
            k in requested_feature_set for k in keys
        )

    primitive_cache: dict[tuple[str, str, int], pd.DataFrame] = {}

    def _roll_std(name: str, src: pd.DataFrame, window: int) -> pd.DataFrame:
        key = ("roll_std", name, int(window))
        if key not in primitive_cache:
            primitive_cache[key] = ff.apply_to_frame(
                src, ff._numba_rolling_std_nan_safe, int(window)
            ).astype(np.float32)
        return primitive_cache[key]

    def _roll_mean(name: str, src: pd.DataFrame, window: int) -> pd.DataFrame:
        key = ("roll_mean", name, int(window))
        if key not in primitive_cache:
            primitive_cache[key] = ff.numba_rolling_mean(src, int(window)).astype(
                np.float32
            )
        return primitive_cache[key]

    def _roll_sum(name: str, src: pd.DataFrame, window: int) -> pd.DataFrame:
        key = ("roll_sum", name, int(window))
        if key not in primitive_cache:
            primitive_cache[key] = ff.numba_rolling_sum(src, int(window)).astype(
                np.float32
            )
        return primitive_cache[key]

    def _roll_max(name: str, src: pd.DataFrame, window: int) -> pd.DataFrame:
        key = ("roll_max", name, int(window))
        if key not in primitive_cache:
            primitive_cache[key] = ff.numba_rolling_max(src, int(window)).astype(
                np.float32
            )
        return primitive_cache[key]

    def _roll_min(name: str, src: pd.DataFrame, window: int) -> pd.DataFrame:
        key = ("roll_min", name, int(window))
        if key not in primitive_cache:
            primitive_cache[key] = ff.numba_rolling_min(src, int(window)).astype(
                np.float32
            )
        return primitive_cache[key]

    def _roll_robust_zscore(name: str, src: pd.DataFrame, window: int) -> pd.DataFrame:
        key = ("roll_robust_zscore", name, int(window))
        if key not in primitive_cache:
            # We must pass numpy array to numba_rolling_robust_zscore in fast_funcs ?
            # Wait, `numba_rolling_robust_zscore` takes a df but the caller code passed `.to_numpy()`.
            primitive_cache[key] = pd.DataFrame(
                ff.numba_rolling_robust_zscore(
                    src.to_numpy() if hasattr(src, "to_numpy") else src, int(window)
                ),
                index=src.index if hasattr(src, "index") else None,
                columns=src.columns if hasattr(src, "columns") else None,
            ).astype(np.float32)
        return primitive_cache[key]

    def _batch_roll_robust_zscore(
        items: list[tuple[str, pd.DataFrame]], window: int
    ) -> dict[str, pd.DataFrame]:
        pending: list[
            tuple[str, pd.Index, pd.Index, np.ndarray, tuple[str, str, int]]
        ] = []
        out: dict[str, pd.DataFrame] = {}

        for name, src in items:
            key = ("roll_robust_zscore", name, int(window))
            cached = primitive_cache.get(key)
            if cached is not None:
                out[name] = cached
                continue

            arr = src.to_numpy(dtype=np.float32, copy=False)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            pending.append(
                (name, src.index, src.columns, np.ascontiguousarray(arr), key)
            )

        if not pending:
            return out

        stacked = np.concatenate([item[3] for item in pending], axis=1)
        stacked_out = ff.numba_rolling_robust_zscore(stacked, int(window))

        widths = [item[3].shape[1] for item in pending]
        offsets = np.cumsum([0, *widths], dtype=np.int32)
        for i, (name, idx, cols, _arr, key) in enumerate(pending):
            start = int(offsets[i])
            end = int(offsets[i + 1])
            res = pd.DataFrame(
                stacked_out[:, start:end],
                index=idx,
                columns=cols,
                copy=False,
            ).astype(np.float32, copy=False)
            primitive_cache[key] = res
            out[name] = res

        return out

    def _frame_like(src: pd.DataFrame, values: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame(values, index=src.index, columns=src.columns)

    def _roll_rank_pct(name: str, src: pd.DataFrame, window: int) -> pd.DataFrame:
        key = ("roll_rank_pct", name, int(window))
        if key not in primitive_cache:
            primitive_cache[key] = pd.DataFrame(
                ff.numba_rolling_rank_pct(
                    src.to_numpy() if hasattr(src, "to_numpy") else src, int(window)
                ),
                index=src.index if hasattr(src, "index") else None,
                columns=src.columns if hasattr(src, "columns") else None,
            ).astype(np.float32)
        return primitive_cache[key]

    def _ewma(
        name: str, src: pd.DataFrame, alpha: float, adjust: bool = False
    ) -> pd.DataFrame:
        key = ("ewma", name, int(alpha * 1_000_000))
        if key not in primitive_cache:
            primitive_cache[key] = ff.numba_ewma(src, float(alpha), adjust).astype(
                np.float32
            )
        return primitive_cache[key]

    # Check inputs
    # Check inputs (removing debug checks to reduce spam)
    # for k, v in panel.items():
    #     check_inf_nan(v, f"input_panel_{k}")

    # Validate panel data quality
    validation_results = validate_panel(panel, raise_on_error=False, verbose=False)
    if not validation_results["valid"]:
        tprint(
            f"WARNING: Panel validation failed with {len(validation_results['errors'])} errors"
        )
        for error in validation_results["errors"][:3]:  # Show first 3 errors
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
    o_raw = panel["open"].astype(np.float32)
    o_raw.index = new_idx
    o = ff.numba_ewma(_safe_log_df(o_raw, eps=safe_log_eps), 2.0 / 6.0, False)
    # 3. Transform High (log-space)
    h_raw = panel["high"].astype(np.float32)
    h_raw.index = new_idx
    h = ff.numba_ewma(_safe_log_df(h_raw, eps=safe_log_eps), 2.0 / 6.0, False)
    # keep h_raw alive for raw ATR% computation below

    # 4. Transform Low (log-space)
    l_raw = panel["low"].astype(np.float32)
    l_raw.index = new_idx
    l = ff.numba_ewma(_safe_log_df(l_raw, eps=safe_log_eps), 2.0 / 6.0, False)
    # keep l_raw alive for raw ATR% computation below

    # 5. Transform Close
    c_raw = panel["close"].astype(np.float32)
    c_raw.index = new_idx

    # Compute Proxy Target for gate-feature selection:
    # strictly past-only to avoid any lookahead leakage.
    # Emphasize:
    #   - trailing realized volatility
    #   - entropy regime change
    #   - regime transition frequency
    #   - directional persistence
    ret_1 = c_raw.pct_change().fillna(0.0).astype(np.float32)
    ret_sign = np.sign(ret_1).astype(np.float32)

    past_rv_2h = (
        ret_1.rolling(2, min_periods=1).std().shift(1).fillna(0.0).astype(np.float32)
    )
    past_rv_4h = (
        ret_1.rolling(4, min_periods=1).std().shift(1).fillna(0.0).astype(np.float32)
    )
    past_rv_8h = (
        ret_1.rolling(8, min_periods=1).std().shift(1).fillna(0.0).astype(np.float32)
    )
    past_entropy_4h = ff.apply_to_frame(ret_sign, ff.binary_entropy_nb, 4).shift(1)
    past_entropy_8h = ff.apply_to_frame(ret_sign, ff.binary_entropy_nb, 8).shift(1)
    past_entropy_4h = past_entropy_4h.fillna(0.5).astype(np.float32)
    past_entropy_8h = past_entropy_8h.fillna(0.5).astype(np.float32)
    entropy_change = (past_entropy_4h - past_entropy_8h).astype(np.float32)

    regime_transitions_8h = ff.apply_to_frame(ret_sign, ff.binary_entropy_nb, 8)
    regime_transitions_8h = (
        regime_transitions_8h.shift(1).fillna(0.5).astype(np.float32)
    )
    directional_persistence_8h = (
        _rolling_autocorr_df(ret_sign, 8).shift(1).fillna(0.0).astype(np.float32)
    )
    directional_persistence_8h = ((directional_persistence_8h + 1.0) * 0.5).astype(
        np.float32
    )
    rv_proxy = np.log1p(past_rv_2h + 0.5 * past_rv_4h + 0.25 * past_rv_8h).astype(
        np.float32
    )
    target_proxy = (
        0.40 * rv_proxy
        + 0.20 * entropy_change
        + 0.20 * regime_transitions_8h
        + 0.20 * directional_persistence_8h
    ).astype(np.float32)
    del ret_1, ret_sign, past_rv_2h, past_rv_4h, past_rv_8h
    del past_entropy_4h, past_entropy_8h, entropy_change
    del regime_transitions_8h, directional_persistence_8h, rv_proxy
    gc.collect()

    # --- Raw-scale asset identity features (computed before FFD transform deletes raw data) ---
    # Raw ATR% = ATR(h_raw, l_raw, c_raw, 14) / c_raw  (fraction, not log-differenced)
    _raw_atr = ff.numba_atr_no_norm(h_raw, l_raw, c_raw, n=cfg["atr_n"])
    _raw_atr_pct = (_raw_atr / (c_raw + 1e-12)).astype(np.float32)
    raw_atr_pct = _raw_atr_pct
    del _raw_atr

    # --- Liquidity Features (User Request) ---
    # Must compute before deleting h_raw, l_raw, c_raw, v_raw
    # Volume is in panel still, so we can access it
    _v_raw = panel["volume"].astype(np.float32)
    _v_raw.index = new_idx
    _rng = np.log(h_raw / np.maximum(l_raw, 1e-12)).astype(np.float32)
    _dollar_vol = (c_raw * _v_raw).astype(np.float32)
    _rng_sum_48 = _roll_sum("rng", _rng, 48)
    _dv_sum_48 = _roll_sum("dollar_vol", _dollar_vol, 48)
    _impact = (_rng_sum_48 / np.maximum(_dv_sum_48, 1e-12)).astype(np.float32)
    _dv_log = np.log(np.maximum(_dollar_vol, 1e-12)).astype(np.float32)

    def _zscore(x: pd.DataFrame) -> pd.DataFrame:
        return robust_zscore_rolling(x, 24 * 30, quantile=0.50)

    _liq_feats_temp = {}
    _liq_feats_temp["dv_z"] = _zscore(_dv_log).astype(np.float32)
    _liq_feats_temp["rng_z"] = _zscore(_rng).astype(np.float32)
    _liq_feats_temp["impact_z"] = _zscore(_impact).astype(np.float32)
    _liq_score = (
        _liq_feats_temp["dv_z"] - _liq_feats_temp["rng_z"] - _liq_feats_temp["impact_z"]
    ).astype(np.float32)
    _liq_feats_temp["liq_score"] = _zscore(_liq_score).astype(np.float32)

    # Causal rank proxy to simulate qcut
    _liq_pct = ff.numba_rolling_rank_pct(
        _liq_feats_temp["liq_score"], window=24 * 30
    ).fillna(0.5)
    _liq_feats_temp["liq_state"] = np.floor(_liq_pct.clip(0, 0.9999) * 5).astype(
        np.float32
    )

    del (
        _v_raw,
        _rng,
        _dollar_vol,
        _rng_sum_48,
        _dv_sum_48,
        _impact,
        _dv_log,
        _liq_score,
        _liq_pct,
    )
    gc.collect()

    c_log = _safe_log_df(c_raw, eps=safe_log_eps)

    c_log_diff1 = c_log.diff(1).astype(np.float32)
    c_log_diff1_abs = c_log_diff1.abs().astype(np.float32)
    c_log_diff1_sign = np.sign(c_log_diff1).astype(np.float32)
    h_minus_l = (h - l).astype(np.float32)
    c_log_minus_o = (c_log - o).astype(np.float32)

    ffd_thres = float(cfg.get("ffd_thres", 1e-5))

    c = _transform_close_fixed_ffd(
        c_raw,
        d=float(cfg.get("ffd_d_base", 0.4)),
        _label="close",
        already_logged=False,
        thres=ffd_thres,
    )
    feature_index = c.index
    feature_columns = c.columns
    feature_shape = c.shape
    # 6. Transform Volume
    v_raw = panel["volume"].astype(np.float32)
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

    feats = {}
    feats.update(_liq_feats_temp)
    feats.update(intraday_library_feats)

    # Correct return naming: log returns from log close
    feats["lr_1h"] = c_log_diff1
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
    tr_ln_1 = h_minus_l
    tr_ln_2 = (h - prev_c_log).abs()
    tr_ln_3 = (l - prev_c_log).abs()
    tr_ln = np.maximum(tr_ln_1, np.maximum(tr_ln_2, tr_ln_3))
    atr_ln = ff.numba_ewma(tr_ln, 1.0 / cfg["atr_n"], False).clip(
        lower=float(cfg.get("atr_ln_floor", 1e-6))
    )

    feats["atr_ln"] = atr_ln.astype(np.float32)
    feats["range_ln"] = h_minus_l
    feats["gap_ln"] = (o - prev_c_log).astype(np.float32)
    feats["body_ln"] = c_log_minus_o
    feats["upper_wick_ln"] = (h - np.maximum(o, c_log)).clip(lower=0).astype(np.float32)
    feats["lower_wick_ln"] = (np.minimum(o, c_log) - l).clip(lower=0).astype(np.float32)

    feats["range_pct"] = (feats["range_ln"] / (feats["atr_ln"] + 1e-12)).astype(
        np.float32
    )
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

        ffd_rv_12 = ff.apply_to_frame(
            feats[f"ffd_diff_1_{d_tag}"], ff._numba_rolling_std_nan_safe, 12
        )
        ffd_rv_24 = ff.apply_to_frame(
            feats[f"ffd_diff_1_{d_tag}"], ff._numba_rolling_std_nan_safe, 24
        )
        feats[f"ffd_rv_12_{d_tag}"] = ffd_rv_12.astype(np.float32)
        feats[f"ffd_rv_24_{d_tag}"] = ffd_rv_24.astype(np.float32)

        ffd_mu_24 = ff.numba_rolling_mean(ffd_c_d, 24)
        ffd_sd_24 = ff.numba_rolling_std(ffd_c_d, 24)
        feats[f"ffd_z_24_{d_tag}"] = (
            (ffd_c_d - ffd_mu_24) / (ffd_sd_24 + 1e-12)
        ).astype(np.float32)

        ffd_max_24 = ff.numba_rolling_max(ffd_c_d, 24)
        ffd_min_24 = ff.numba_rolling_min(ffd_c_d, 24)
        feats[f"ffd_range_24_{d_tag}"] = (ffd_max_24 - ffd_min_24).astype(np.float32)

    # Carry layer (mid-speed continuation): d=0.5 primary, d=0.4 secondary
    for d in carry_d_values:
        if d in ffd_close:
            d_tag = f"{int(round(d * 10)):02d}"
            d_series = ffd_close[d]
            for w in cfg.get("ffd_slope_windows", [12, 24]):
                feats[f"ffd_slope_{d_tag}_{int(w)}"] = ff.apply_to_frame(
                    d_series, ff._numba_rolling_slope, int(w)
                ).astype(np.float32)
            mr_w = int(cfg.get("ffd_mr_window", 24))
            mu = ff.numba_rolling_mean(d_series, mr_w)
            sd = ff.numba_rolling_std(d_series, mr_w)
            feats[f"ffd_mr_z_{d_tag}"] = ((d_series - mu) / (sd + 1e-12)).astype(
                np.float32
            )

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
                feats[f"ffd_ctx_slope_{d_tag}_{int(w)}"] = ff.apply_to_frame(
                    d_series, ff._numba_rolling_slope, int(w)
                ).astype(np.float32)

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

    # We can optimize these multiple horizons by using `diff` on cumulative sum,
    # but `numba_rolling_sum` handles NaNs and edge cases. We just use them directly.
    carry_windows = [2, 3, 4, 5, 6]
    for H in carry_windows:
        feats[f"ret{H}h"] = ff.numba_rolling_sum(c_carry, H).astype(np.float32)

    context_windows = [8, 10, 12, 16, 20, 24, 28, 48, 72, 120]
    for H in context_windows:
        feats[f"ret{H}h"] = ff.numba_rolling_sum(c_context, H).astype(np.float32)

    # Carry price becomes default base for many pre-existing features.
    c = c_carry

    # --- Asset Identity Features (raw-scale, NOT cross-sectionally normalized) ---
    # These provide "who is this asset" context without one-hot encoding.
    # asset_atr_level: smooth baseline of raw ATR% over 60 days — stable volatility fingerprint
    # asset_vol_level: smooth baseline of raw log(volume_usd) over 60 days — stable liquidity fingerprint
    # Use EWMA (alpha=2/(1440+1)) as fast O(T*S) proxy for rolling median.
    _ALPHA_IDENTITY = 2.0 / (24 * 60 + 1)  # EWMA alpha matching 60-day span
    feats["asset_atr_level"] = ff.numba_ewma(
        _raw_atr_pct, _ALPHA_IDENTITY, False
    ).astype(np.float32)
    feats["asset_vol_level"] = ff.numba_ewma(
        _raw_log_vol, _ALPHA_IDENTITY, False
    ).astype(np.float32)
    # vol_state: current log_vol / long-run level — >1 means elevated activity vs own baseline
    feats["vol_state"] = (_raw_log_vol / (feats["asset_vol_level"] + 1e-9)).astype(
        np.float32
    )
    del _raw_atr_pct, _raw_log_vol

    # --- D-Specific Feature Families ---
    # Realized volatility family (d=0.4,0.6)
    for d in [0.4, 0.6]:
        d_tag = f"{int(round(d * 10)):02d}"
        base_diff = feats[f"ffd_diff_1_{d_tag}"]
        feats[f"ffd_rv_2h_{d_tag}"] = ff.apply_to_frame(
            base_diff, ff._numba_rolling_std_nan_safe, 2
        ).astype(np.float32)
        feats[f"ffd_rv_6h_{d_tag}"] = ff.apply_to_frame(
            base_diff, ff._numba_rolling_std_nan_safe, 6
        ).astype(np.float32)
        feats[f"ffd_rv_24h_{d_tag}"] = ff.apply_to_frame(
            base_diff, ff._numba_rolling_std_nan_safe, 24
        ).astype(np.float32)

    # Momentum acceleration features (d=0.6)
    for d in [0.6]:
        d_tag = f"{int(round(d * 10)):02d}"
        diff = feats[f"ffd_diff_1_{d_tag}"]
        feats[f"ffd_accel_{d_tag}"] = diff.diff().astype(np.float32)
        vol = ff.apply_to_frame(diff, ff._numba_rolling_std_nan_safe, 24)
        feats[f"ffd_z_{d_tag}"] = (
            (diff / (vol + 1e-12)).fillna(0.0).clip(-50, 50).astype(np.float32)
        )

    # Volume-price correlation features (d=0.4,0.6)
    for d in [0.4, 0.6]:
        d_tag = f"{int(round(d * 10)):02d}"
        diff = feats[f"ffd_diff_1_{d_tag}"]
        feats[f"ffd_vol_price_corr_10h_{d_tag}"] = (
            ff.numba_rolling_corr(diff.abs(), v, 10).fillna(0.0).astype(np.float32)
        )

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
            feats[f"ffd_donch_dist_{d_tag}_{k}"] = (
                (donch / atr_base).clip(lower=0).astype(np.float32)
            )

    # ATR expansion and tail risk (d=0.6)
    for d in [0.6]:
        d_tag = f"{int(round(d * 10)):02d}"
        d_series = ffd_close[d]
        tr_d = np.maximum(
            h - l,
            np.maximum((h - d_series.shift(1)).abs(), (l - d_series.shift(1)).abs()),
        )
        atr_tr_d = ff.numba_ewma(tr_d, 1.0 / cfg["atr_n"], False)
        feats[f"ffd_atr_expansion_{d_tag}"] = (tr_d / (atr_tr_d + 1e-12)).astype(
            np.float32
        )
        diff = d_series.diff(1)
        feats[f"ffd_cvar_5pct_{d_tag}"] = (
            ff.numba_rolling_quantile(diff, 48, 0.05).fillna(0.0).astype(np.float32)
        )

    # Liquidity shock features (d=0.4,0.6)
    for d in [0.4, 0.6]:
        d_tag = f"{int(round(d * 10)):02d}"
        diff = feats[f"ffd_diff_1_{d_tag}"]
        illiq_raw = (diff.abs() / ((v * ffd_close[d]) + 1e-12)).replace(
            [np.inf, -np.inf], np.nan
        )
        feats[f"ffd_amihud_{d_tag}"] = (
            ff.numba_rolling_mean(illiq_raw, 24).fillna(0.0).astype(np.float32)
        )
        vr = v * diff.abs()
        ema_vr = ema(vr, 24)
        ratio_floor = float(cfg.get("ratio_denom_floor", 1e-6))
        vr_ratio = vr / ema_vr.abs().clip(lower=ratio_floor)
        if bool(cfg.get("ratio_use_log", True)):
            vr_ratio = np.log1p(vr_ratio.clip(lower=0))
        feats[f"ffd_vol_range_shock_{d_tag}"] = vr_ratio.astype(np.float32)

    # --- Technical Regime (Ridge) Features (User Request) ---
    tprint("Features: technical regime (ridge) indicators")
    ema20 = _ewma("c_log", c_log, 2.0 / 21.0, False)
    ema50 = _ewma("c_log", c_log, 2.0 / 51.0, False)
    ema200 = _ewma("c_log", c_log, 2.0 / 201.0, False)

    _safe_range_ln = h_minus_l.clip(lower=1e-12)
    feats["range_atr"] = (feats["range_ln"] / (feats["atr_ln"] + 1e-12)).astype(
        np.float32
    )
    _c_log_minus_o_val = c_log_minus_o.to_numpy(dtype=np.float32)
    _safe_range_ln_val = _safe_range_ln.to_numpy(dtype=np.float32)
    feats["body_ratio"] = pd.DataFrame(
        np.abs(_c_log_minus_o_val) / _safe_range_ln_val,
        index=c_log.index,
        columns=c_log.columns,
    ).astype(np.float32)

    _h_val = h.to_numpy(dtype=np.float32)
    _c_log_val = c_log.to_numpy(dtype=np.float32)
    _o_val = o.to_numpy(dtype=np.float32)
    _l_val = l.to_numpy(dtype=np.float32)

    feats["upper_wick_ratio"] = pd.DataFrame(
        (_h_val - np.maximum(_o_val, _c_log_val)) / _safe_range_ln_val,
        index=c_log.index,
        columns=c_log.columns,
    ).astype(np.float32)

    feats["lower_wick_ratio"] = pd.DataFrame(
        (np.minimum(_o_val, _c_log_val) - _l_val) / _safe_range_ln_val,
        index=c_log.index,
        columns=c_log.columns,
    ).astype(np.float32)

    # Missing Trigger/Location features from ridge_regime_event_assessment
    _upper_wick_arr = feats["upper_wick_ratio"].to_numpy(dtype=np.float32)
    _lower_wick_arr = feats["lower_wick_ratio"].to_numpy(dtype=np.float32)

    feats["wick_to_range"] = pd.DataFrame(
        _upper_wick_arr + _lower_wick_arr, index=c_log.index, columns=c_log.columns
    ).astype(np.float32)

    feats["orderflow_imbalance"] = pd.DataFrame(
        (_c_log_val - _o_val) / _safe_range_ln_val,
        index=c_log.index,
        columns=c_log.columns,
    ).astype(np.float32)

    # Aliases for exact user-requested names
    feats["upper_wick"] = feats["upper_wick_ratio"]
    feats["lower_wick"] = feats["lower_wick_ratio"]

    _ema20_val = ema20.to_numpy(dtype=np.float32)
    _ema20_shift_val = ema20.shift(5).to_numpy(dtype=np.float32)
    _atr_ln_val = feats["atr_ln"].to_numpy(dtype=np.float32)

    feats["ema20_slope_5h"] = pd.DataFrame(
        (_ema20_val - _ema20_shift_val) / (_atr_ln_val + 1e-12),
        index=c_log.index,
        columns=c_log.columns,
    ).astype(np.float32)
    feats["ema_slope_norm"] = feats["ema20_slope_5h"]
    feats["ema_slope"] = pd.DataFrame(
        (_ema20_val - _ema20_shift_val), index=c_log.index, columns=c_log.columns
    ).astype(np.float32)

    _l_val = l.to_numpy(dtype=np.float32)
    feats["pullback_depth"] = pd.DataFrame(
        (_ema20_val - _l_val) / (_atr_ln_val + 1e-12),
        index=c_log.index,
        columns=c_log.columns,
    ).astype(np.float32)

    atr_long = ff.numba_ewma(tr_ln, 1.0 / (24 * 7), False).clip(lower=1e-9)
    _atr_long_val = atr_long.to_numpy(dtype=np.float32)
    feats["atr_compression_ratio"] = pd.DataFrame(
        _atr_ln_val / _atr_long_val, index=c_log.index, columns=c_log.columns
    ).astype(np.float32)
    feats["compression_ratio"] = feats["atr_compression_ratio"]

    # Missing from Technical Regime (Ridge) Features
    _tr_ln_val = tr_ln.to_numpy(dtype=np.float32)
    feats["range_expansion_ratio"] = pd.DataFrame(
        _tr_ln_val / (_atr_ln_val + 1e-12), index=c_log.index, columns=c_log.columns
    ).astype(np.float32)

    _accel_raw = c_log - 2 * c_log.shift(1) + c_log.shift(2)
    feats["acceleration"] = _accel_raw.astype(np.float32)
    feats["acceleration_norm"] = (_accel_raw / (feats["atr_ln"] + 1e-12)).astype(
        np.float32
    )

    # Volume Spike: volume / volume_ma(24)
    vol_ma24 = _roll_mean("v", v, 24)
    feats["volume_spike"] = (v / (vol_ma24 + 1e-12)).astype(np.float32)

    feats["ema20_gt_ema50"] = (ema20 > ema50).astype(np.float32)
    feats["ema50_gt_ema200"] = (ema50 > ema200).astype(np.float32)
    feats["price_lt_ema200"] = (c_log < ema200).astype(np.float32)
    feats["ema50_ema200_spread_atr"] = (
        (ema(c, 50) - ema(c, 200)) / (atr_base + 1e-12)
    ).astype(np.float32)
    feats["ema50_slope"] = (ema50 - ema50.shift(1)).astype(np.float32)
    feats["trend_strength_percentile"] = ff.numba_rolling_rank_pct(
        feats["ema50_slope"].abs(), 1000
    ).astype(np.float32)
    feats["ema50_ema200_spread_continuous"] = feats["ema50_ema200_spread_atr"]

    ret1h_std_16 = _roll_std("ret1h", feats["ret1h"], 16)

    # Calculate rv_24h and use it for realized_volatility_24h too
    feats["rv_24h"] = _roll_std("ret1h", feats["ret1h"], 24)
    # realized_volatility_24h is 24h (96 15m bars). Note rv_24h is standard 24 bars.
    ret1h_std_96_temp = _roll_std("ret1h", feats["ret1h"], 96)
    feats["realized_volatility_24h"] = ret1h_std_96_temp  # 24h = 96 * 15m
    _atr_change = (
        feats["atr_ln"]
        .pct_change()
        .replace([np.inf, -np.inf], 0.0)
        .fillna(0.0)
        .astype(np.float32)
    )
    feats["atr_change_rate"] = _atr_change
    feats["atr_change_rate_ts_continuous"] = _atr_change
    feats["true_range_percentile"] = ff.numba_rolling_rank_pct(tr_ln, 1000).astype(
        np.float32
    )

    # Bollinger Band Width
    bb_mean = _roll_mean("c_log", c_log, 20)
    bb_std = _roll_std("c_log", c_log, 20)
    feats["bollinger_band_width"] = (2 * 2 * bb_std / (bb_mean + 1e-12)).astype(
        np.float32
    )

    h_max_20 = _roll_max("h", h, 20)
    l_min_20 = _roll_min("l", l, 20)
    feats["rolling_range_20"] = (h_max_20 - l_min_20).astype(np.float32)
    feats["atr_percentile"] = ff.numba_rolling_rank_pct(feats["atr_ln"], 1000).astype(
        np.float32
    )

    feats["prior_range"] = feats["range_ln"].shift(1).astype(np.float32)
    feats["prior_volatility"] = feats["atr_ln"].shift(1).astype(np.float32)

    # Efficiency ratio over 20
    direction = (c_log - c_log.shift(20)).abs()
    volatility = _roll_sum("c_log_abs_diff", (c_log - c_log.shift(1)).abs(), 20)
    feats["efficiency_ratio_20"] = (direction / (volatility + 1e-12)).astype(np.float32)

    # Choppiness index over 20
    atr_sum = _roll_sum("tr_ln", tr_ln, 20)
    high_20 = h_max_20
    low_20 = l_min_20
    range_20 = high_20 - low_20

    range_safe = np.where(range_20 > 1e-12, range_20, 1e-12)
    ratio = atr_sum / range_safe
    ratio_clean = np.where(np.isfinite(ratio), ratio, 1e-12)

    feats["choppiness_index_20"] = (
        100 * np.log10(np.clip(ratio_clean, 1e-12, None)) / np.float32(np.log10(20))
    ).astype(np.float32)

    # Direction Entropy 20
    ret_sign = np.sign(feats["ret1h"])
    feats["direction_entropy_20"] = ff.apply_to_frame(
        ret_sign, ff.binary_entropy_nb, 20
    )

    if _needs_feature("bars_since_trend_flip"):
        trend_slope = ff.apply_to_frame(c_log, ff.slope_nb, 6)
        trend_sign = (trend_slope > 0).astype(np.float32)
        feats["bars_since_trend_flip"] = ff.apply_to_frame(
            trend_sign, ff.bars_since_flip_nb
        ).astype(np.float32)

    if _needs_feature("bars_since_ema20_ema50_cross_log_norm"):
        ema_20 = ff.apply_to_frame(c_log, ff.ema_nb, 20)
        ema_50 = ff.apply_to_frame(c_log, ff.ema_nb, 50)
        ema_diff_sign = ((ema_20 - ema_50) > 0).astype(np.float32)
        raw = ff.apply_to_frame(ema_diff_sign, ff.bars_since_flip_nb)
        feats["bars_since_ema20_ema50_cross_log_norm"] = (
            np.log1p(np.minimum(raw, 100)) / np.log1p(100)
        ).astype(np.float32)

    if _needs_feature("bars_in_high_vol_state_log_norm"):
        # Depends on atr_percentile (line 1683)
        high_vol_state = (feats["atr_percentile"] >= 0.8).astype(np.float32)
        raw = ff.apply_to_frame(high_vol_state, ff.consecutive_bars_nb)
        feats["bars_in_high_vol_state_log_norm"] = (
            np.log1p(np.minimum(raw, 50)) / np.log1p(50)
        ).astype(np.float32)

    if _needs_feature("bars_outside_ema20_atr_band_log_norm"):
        ema_20 = ff.apply_to_frame(c_log, ff.ema_nb, 20)
        dist = np.abs(c_raw - np.exp(ema_20)) / np.maximum(atr_base, 1e-8)
        outside_state = (dist >= 1.0).astype(np.float32)
        raw = ff.apply_to_frame(outside_state, ff.consecutive_bars_nb)
        feats["bars_outside_ema20_atr_band_log_norm"] = (
            np.log1p(np.minimum(raw, 50)) / np.log1p(50)
        ).astype(np.float32)

    if _needs_feature("up_down_semivol_ratio_tanh"):
        feats["up_down_semivol_ratio_tanh"] = ff.apply_to_frame(
            feats["ret1h"], ff.up_down_semivol_ratio_nb, 20
        ).astype(np.float32)

    if _needs_feature("up_down_return_mass_ratio_tanh"):
        feats["up_down_return_mass_ratio_tanh"] = ff.apply_to_frame(
            feats["ret1h"], ff.up_down_return_mass_ratio_nb, 20
        ).astype(np.float32)

    if _needs_feature("tail_asymmetry_q90_q10_atr_norm"):
        q90 = ff.numba_rolling_quantile(feats["ret1h"], 50, 0.90)
        q10 = np.abs(ff.numba_rolling_quantile(feats["ret1h"], 50, 0.10))
        raw = np.log((q90 + 1e-8) / (q10 + 1e-8))
        feats["tail_asymmetry_q90_q10_atr_norm"] = np.tanh(raw).astype(np.float32)

    # Volatility Ratio Short/Long (e.g., 2h vs 24h)
    ret1h_std_8 = _roll_std("ret1h", feats["ret1h"], 8)
    feats["volatility_ratio_short_long"] = (
        ret1h_std_8 / (feats["realized_volatility_24h"] + 1e-12)
    ).astype(np.float32)
    feats["volume_percentile"] = ff.numba_rolling_rank_pct(v, 1000).astype(np.float32)

    feats["volume_zscore_48h"] = ff.apply_to_frame(
        v, ff._numba_rolling_zscore_nan_safe_1d, 192
    ).astype(
        np.float32
    )  # 48h = 192 * 15m
    feats["compression_score"] = (
        feats["atr_compression_ratio"] * feats["bollinger_band_width"]
    ).astype(np.float32)

    # Fast func vectorization where appropriate (avoiding Series apply loop)
    # Autocorrelation 48
    ret_48 = feats["ret1h"]
    ret_48_mean = _roll_mean("ret1h", ret_48, 48)
    ret_48_std = _roll_std("ret1h", ret_48, 48)
    ret_48_var = ret_48_std**2
    ret_cov_48 = ff.numba_rolling_mean(
        (ret_48 - ret_48_mean) * (ret_48.shift(1) - ret_48_mean.shift(1)), 48
    )
    feats["return_autocorr_48"] = (ret_cov_48 / (ret_48_var + 1e-12)).astype(np.float32)

    ret1h_std_10 = _roll_std("ret1h", feats["ret1h"], 10)
    feats["variance_ratio_10_48"] = ((ret1h_std_10**2) / (ret_48_var + 1e-12)).astype(
        np.float32
    )

    feats["volume_trend_48"] = (
        _ewma("v", v, 2.0 / 49.0, False) - _ewma("v", v, 2.0 / 193.0, False)
    ).astype(np.float32)

    v_48_mean = _roll_mean("v", v, 48)
    v_48_std = _roll_std("v", v, 48)
    v_48_var = v_48_std**2
    v_cov_48 = ff.numba_rolling_mean(
        (v - v_48_mean) * (v.shift(1) - v_48_mean.shift(1)), 48
    )
    feats["volume_autocorr_48"] = (v_cov_48 / (v_48_var + 1e-12)).astype(np.float32)

    # Volatility of volatility 48
    vol_48 = ret_48_std
    feats["volatility_of_volatility_48"] = ff.apply_to_frame(
        vol_48, ff._numba_rolling_std_nan_safe, 48
    ).astype(np.float32)

    feats["trend_acceleration"] = (
        feats["ema50_slope"] - feats["ema50_slope"].shift(1)
    ).astype(np.float32)

    vol_48_mean = ff.numba_rolling_mean(vol_48, 48)
    vol_48_var = feats["volatility_of_volatility_48"] ** 2
    vol_cov_48 = ff.numba_rolling_mean(
        (vol_48 - vol_48_mean) * (vol_48.shift(1) - vol_48_mean.shift(1)), 48
    )
    feats["volatility_autocorr_48"] = (vol_cov_48 / (vol_48_var + 1e-12)).astype(
        np.float32
    )

    feats["dist_ema20_atr"] = ((c_log - ema20) / (feats["atr_ln"] + 1e-12)).astype(
        np.float32
    )
    feats["distance_to_ema"] = feats["dist_ema20_atr"]

    # Missing Technical Regime Location Features
    # Note: 50 * 2 = 100 bars for 15m conversion match logic in ridge
    _c_log_arr = c_log.to_numpy(dtype=np.float32)

    _mean_100 = ff._numba_rolling_mean_parallel(_c_log_arr, 100)
    _std_100 = np.maximum(ff._numba_rolling_std_parallel(_c_log_arr, 100), 1e-12)
    feats["zscore_price_50"] = pd.DataFrame(
        (_c_log_arr - _mean_100) / _std_100, index=c_log.index, columns=c_log.columns
    ).astype(np.float32)

    # Note: 200 * 2 = 400 bars for 15m conversion match logic in ridge
    _mean_400 = ff._numba_rolling_mean_parallel(_c_log_arr, 400)
    _std_400 = np.maximum(ff._numba_rolling_std_parallel(_c_log_arr, 400), 1e-12)
    feats["zscore_price_200"] = pd.DataFrame(
        (_c_log_arr - _mean_400) / _std_400, index=c_log.index, columns=c_log.columns
    ).astype(np.float32)

    # --- End Technical Regime ---

    # Distance-from-mean-reversion features (d=0.4)
    for d in [0.4]:
        d_tag = f"{int(round(d * 10)):02d}"
        d_series = ffd_close[d]
        ema_fast = ema(d_series, max(4, int(cfg["ema_fast"] * 0.5)))
        ema_slow = ema(d_series, int(cfg["ema_fast"] * 2))
        feats[f"ffd_dist_ema_fast_{d_tag}"] = (
            (d_series - ema_fast) / (atr_base + 1e-12)
        ).astype(np.float32)
        feats[f"ffd_dist_ema_slow_{d_tag}"] = (
            (d_series - ema_slow) / (atr_base + 1e-12)
        ).astype(np.float32)

    # D-family strength indicators
    abs_04 = feats["ffd_diff_1_04"].abs()
    abs_05 = feats["ffd_diff_1_05"].abs()
    abs_06 = feats["ffd_diff_1_06"].abs()
    total = abs_04 + abs_05 + abs_06 + np.float32(1e-12)
    feats["ffd_strength_04"] = (abs_04 / total).astype(np.float32)
    feats["ffd_strength_05"] = (abs_05 / total).astype(np.float32)
    feats["ffd_strength_06"] = (abs_06 / total).astype(np.float32)

    if 0.6 in ffd_close:
        d_series = ffd_close[0.6]
        mr_w = int(cfg.get("ffd_mr_window", 24))
        mu = ff.numba_rolling_mean(d_series, mr_w)
        sd = ff.numba_rolling_std(d_series, mr_w)
        feats["ffd_mr_z_06"] = ((d_series - mu) / (sd + 1e-12)).astype(np.float32)

    rsi_base = rsi(c, n=cfg["rsi_n"])
    feats["rsi_base"] = rsi_base
    feats["rsi_slope_base"] = rsi_base.diff(cfg["rsi_slope_n"]).astype(np.float32)

    # rv_24h calculated earlier
    feats["rv_2h"] = _roll_std("ret1h", feats["ret1h"], 2)
    feats["rv_4h"] = ff.apply_to_frame(
        feats["ret1h"], ff._numba_rolling_std_nan_safe, 4
    ).astype(np.float32)
    feats["rv_6h"] = _roll_std("ret1h", feats["ret1h"], 6)
    feats["rv_8h"] = _roll_std("ret1h", feats["ret1h"], 8)
    feats["rv_12h"] = ff.apply_to_frame(
        feats["ret1h"], ff._numba_rolling_std_nan_safe, 12
    ).astype(np.float32)

    # New Filter Features (Range & Vol Z-score)
    h_24 = _roll_max("h", h, 24)
    l_24 = _roll_min("l", l, 24)
    h_12 = _roll_max("h", h, 12)
    l_12 = _roll_min("l", l, 12)
    h_16 = _roll_max("h", h, 16)
    l_16 = _roll_min("l", l, 16)

    # range_XXh_pct is max_h - min_l. inputs are log-FFD, so diff is %-ish.
    # Do NOT divide by c (FFD) as it crosses 0.
    # Use np.where to handle cases where rolling windows produce NaN
    feats["range_24h_pct"] = np.where(
        np.isfinite(h_24) & np.isfinite(l_24), (h_24 - l_24), 0.0
    ).astype(np.float32)
    feats["range_12h_pct"] = np.where(
        np.isfinite(h_12) & np.isfinite(l_12), (h_12 - l_12), 0.0
    ).astype(np.float32)
    feats["range_16h_pct"] = np.where(
        np.isfinite(h_16) & np.isfinite(l_16), (h_16 - l_16), 0.0
    ).astype(np.float32)
    del h_24, l_24, h_12, l_12, h_16, l_16

    # Volatility Z-score (using Log-ATR robust z-score)
    # Baseline: 90 days. x = log(ATR/Close).
    # Z = (x - Q(0.50)) / (1.4826 * MAD)
    # atr_base is raw ATR (price units), so we normalize by C
    vol_proxy = atr_base / (c + 1e-12)
    log_vol = np.log(vol_proxy + 1e-9).astype(np.float32)
    vol_z = robust_zscore_rolling(log_vol, 24 * 90, quantile=0.50)
    feats["volatility_zscore"] = np.where(np.isfinite(vol_z), vol_z, 0.0).astype(
        np.float32
    )
    del vol_proxy, log_vol

    feats["qv"] = (c * v).astype(np.float32)
    feats["vol_z24_base"] = zscore_rolling(v, 24)
    feats["vol_z_base"] = zscore_rolling(v, cfg["volz_n"])

    ema_fast_base = ema(c, cfg["ema_fast"])
    ema_slow_base = ema(c, cfg["ema_slow"])
    feats["dist_ema_fast_base"] = ((c - ema_fast_base) / (atr_base + 1e-12)).astype(
        np.float32
    )
    feats["dist_ema_slow_base"] = ((c - ema_slow_base) / (atr_base + 1e-12)).astype(
        np.float32
    )

    feats["roc_div"] = (feats["ret1h"] - feats["ret6h"]).astype(np.float32)
    # ret1h_z: if rv_24h is 0 (constant trend), this explodes. Cap it.
    z_raw = feats["ret1h"] / (feats["rv_24h"] + 1e-9)
    feats["ret1h_z"] = z_raw.fillna(0.0).clip(-50, 50).astype(np.float32)

    body = (c - o).abs()
    upper_wick = (h - c.where(c >= o, o)).clip(lower=0)
    lower_wick = (c.where(c <= o, o) - l).clip(lower=0)
    feats["body_pct"] = body.astype(np.float32)
    feats["wick_body_ratio"] = ((upper_wick + lower_wick) / (body + 1e-12)).astype(
        np.float32
    )

    # New Spike Features
    max_oc = np.maximum(o, c)
    feats["wick_ratio"] = ((h - max_oc) / ((h - l) + 1e-12)).astype(np.float32)
    del body, upper_wick, lower_wick, max_oc

    # --- New Exhaustion & Risk Features (Report 2026-02-10) ---

    # 1. Wick Ratio Max (Exhaustion for short_mr)
    feats["wick_ratio_4h_max"] = ff.numba_rolling_max(feats["wick_ratio"], 4).astype(
        np.float32
    )

    # 2. Volume/Price Divergence (Exhaustion for short_mr)
    # Correlation between price changes and volume changes over 12 hours.
    v_chg = ff.numba_pct_change(v, 1).fillna(0.0).astype(np.float32)
    # Using numba rolling corr (O(N) vs Pandas O(N^2) or O(N log N))
    feats["vol_price_div"] = (
        ff.numba_rolling_corr(feats["ret1h"], v_chg, 12).fillna(0.0).astype(np.float32)
    )
    del v_chg

    # 3. RSI Lagged (for divergence check)
    # Use base RSI here (adaptive RSI is created later).
    feats["rsi_lag1"] = rsi_base.shift(1).astype(np.float32)
    # RSI Slope 1h (Momentum Turn for long_mr)
    feats["rsi_1h_slope"] = rsi_base.diff(1).fillna(0.0).astype(np.float32)

    # 4. Tail Risk (CVaR Proxy for long_tf)
    # 5th percentile return over 48 hours (2 days)
    # Use Numba-optimized rolling quantile (O(N) vs Pandas O(N log W))
    feats["cvar_5pct"] = (
        ff.numba_rolling_quantile(feats["ret1h"], 48, 0.05)
        .fillna(0.0)
        .astype(np.float32)
    )

    # 5. Liquidity Shock (Amihud Proxy for long_tf)
    # |Ret| / (Volume * Price). Spikes indicate price moving on thin liquidity.
    illiq_raw = (feats["ret1h"].abs() / ((v * c) + 1e-12)).replace(
        [np.inf, -np.inf], np.nan
    )
    feats["amihud_illiq"] = (
        ff.numba_rolling_mean(illiq_raw, 24).fillna(0.0).astype(np.float32)
    )

    # 6. Skew Proxy (Close Location Value Mean)
    clv_raw_early = ((2 * c - h - l) / ((h - l) + 1e-9)).fillna(0.0)
    feats["clv_mean_24"] = (
        ff.apply_to_frame(clv_raw_early, ff._numba_rolling_mean_nan_safe, 24)
        .fillna(0.0)
        .astype(np.float32)
    )

    # 7. Stabilization / Falling Knife Features (for long_mr)
    # Climax Volume
    feats["vol_z_4h"] = zscore_rolling(v, 4).fillna(0.0).astype(np.float32)

    # ATR pct change (Volatility Cooling)
    feats["atr_pct_change"] = atr_base.pct_change().fillna(0.0).astype(np.float32)

    # --- End New Features ---

    feats["vol_price_spread"] = (v / ((h - l) + 1e-12)).astype(np.float32)

    prev_close = c.shift(1)
    tr_1 = h - l
    tr_2 = (h - prev_close).abs()
    tr_3 = (l - prev_close).abs()
    tr = np.maximum(tr_1, np.maximum(tr_2, tr_3))
    atr_tr = ff.numba_ewma(tr, 1.0 / cfg["atr_n"], False)
    feats["atr_expansion"] = (tr / (atr_tr + 1e-12)).astype(np.float32)
    del prev_close, tr_1, tr_2, tr_3, tr, atr_tr

    sma_base = ff.numba_rolling_mean(c_context, cfg["trend_sma_n"])
    feats["trend_pct_base"] = (c_context - sma_base).astype(np.float32)

    hod = pd.Series(v.index.hour, index=v.index)
    rvol_denom = ff.numba_grouped_rolling_mean(v, hod, int(cfg["rvol_days"] * 24))
    feats["rvol_hod_base"] = (v / (rvol_denom + 1e-12)).astype(np.float32)

    feats["funding_proxy"] = compute_funding_proxy(c, h, l, v, mkt_gates)

    sin_hod, cos_hod, sin_dow, cos_dow = time_sin_cos(c.index)
    feats["sin_hod"] = pd.DataFrame(
        np.broadcast_to(sin_hod[:, None], c.shape),
        index=c.index,
        columns=c.columns,
    ).astype(np.float32)
    feats["cos_hod"] = pd.DataFrame(
        np.broadcast_to(cos_hod[:, None], c.shape),
        index=c.index,
        columns=c.columns,
    ).astype(np.float32)
    feats["sin_dow"] = pd.DataFrame(
        np.broadcast_to(sin_dow[:, None], c.shape),
        index=c.index,
        columns=c.columns,
    ).astype(np.float32)
    feats["cos_dow"] = pd.DataFrame(
        np.broadcast_to(cos_dow[:, None], c.shape),
        index=c.index,
        columns=c.columns,
    ).astype(np.float32)

    signed_vol = v * np.sign(c - o)
    sv_abs = signed_vol.abs()
    ewma_sv_fast = ema(signed_vol, 6)
    ewma_sv_slow = ema(sv_abs, 24)

    feats["flow_persistence"] = (ewma_sv_fast / (ewma_sv_slow + 1e-12)).astype(
        np.float32
    )
    feats["flow_ratio"] = feats["flow_persistence"]

    eff = (c - o).abs() / ((h - l) + 1e-9)
    feats["efficiency"] = ff.numba_rolling_mean(eff, 12)

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
    feats["slope"] = ((ema_fast_base - ema_slow_base) / (atr_base + 1e-12)).astype(
        np.float32
    )

    t_snr_num = ema(feats["ret1h"], 6).abs()
    t_snr_den = _roll_std("ret1h", feats["ret1h"], 24)
    feats["trend_snr"] = (t_snr_num / (t_snr_den + 1e-12)).astype(np.float32)

    # v_power: Volume / Abs Price Change? Normalizing by c.abs() (FFD) is unstable if c~0.
    # Normalize by ATR base instead.
    feats["v_power"] = (v / (atr_base + 1e-9)).astype(np.float32)
    feats["signed_vol"] = signed_vol.astype(np.float32)

    atr_ema_f = ema(atr_base, 6)
    atr_ema_s = ema(atr_base, 24)
    feats["atr_slope"] = ((atr_ema_f - atr_ema_s) / (atr_ema_s + 1e-12)).astype(
        np.float32
    )

    vwap_24 = ff.numba_rolling_vwap(c, v, 24).astype(np.float32)

    feats["dist_vwap_norm"] = ((c - vwap_24) / (atr_base + np.float32(1e-12))).astype(
        np.float32
    )

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
    feats["cumulative_delta_stall"] = (
        ff.numba_rolling_corr(c, cum_sv, 24).fillna(0.0).astype(np.float32)
    )
    cum_sv_6 = ff.numba_rolling_sum(signed_vol, 6)
    feats["delta_stall_6"] = (
        ff.numba_rolling_corr(c, cum_sv_6, 6).fillna(0.0).astype(np.float32)
    )

    feats["vol_expansion_ratio"] = (atr_ema_f / (atr_ema_s + 1e-12)).astype(np.float32)

    sig_s = _roll_std("ret1h", feats["ret1h"], 6)
    sig_m = ff.numba_rolling_std(feats["ret1h"], 18)
    sig_m_floor = sig_m.abs().clip(lower=ratio_floor)
    vol_comp = sig_s / sig_m_floor
    if bool(cfg.get("ratio_use_log", False)):
        vol_comp = np.log1p(vol_comp.clip(lower=0))
    feats["vol_compression"] = vol_comp.astype(np.float32)

    rv_ratio = mkt_gates["mkt_rv_ratio"].reindex(c.index).astype(np.float32)
    feats["mkt_rv_ratio"] = rv_ratio

    mkt_rv_pct = mkt_gates["mkt_rv_pct"].reindex(c.index).astype(np.float32)
    feats["mkt_rv_pct"] = mkt_rv_pct

    abs_mkt_ret24h_z = mkt_gates["abs_mkt_ret24h_z"].reindex(c.index).astype(np.float32)
    feats["abs_mkt_ret24h_z"] = abs_mkt_ret24h_z

    _rv_ratio_smooth = rv_ratio
    _smooth_span = max(1, int(cfg.get("rv_selector_smooth_span", 6)))
    if _smooth_span > 1:
        # Avoid creating DataFrame back and forth here if `rv_ratio` is already broadcasted properly
        # Wait, `rv_ratio` is now just a 1D Series, `numba_ewma` can handle it natively
        _rv_ratio_smooth = ff.numba_ewma(
            _rv_ratio_smooth.to_frame(), 2.0 / (_smooth_span + 1.0), False
        ).iloc[:, 0]

    def pick_by_rv(fast_df, base_df, slow_df):
        # We process this mostly in numpy array space
        fast_arr = fast_df.to_numpy(dtype=np.float32)
        base_arr = base_df.to_numpy(dtype=np.float32)
        slow_arr = slow_df.to_numpy(dtype=np.float32)
        rr = _rv_ratio_smooth.to_numpy(dtype=np.float32)[:, None]

        fast_thr = float(cfg["rv_ratio_fast_thr"])
        slow_thr = float(cfg["rv_ratio_slow_thr"])
        mode = str(cfg.get("rv_selector_mode", "blend")).lower()

        if mode == "blend" and fast_thr > slow_thr:
            mid = 0.5 * (fast_thr + slow_thr)
            half = max(0.5 * (fast_thr - slow_thr), 1e-6)

            # Using NumPy avoids many temporary large (T, S) DataFrames
            dist = np.clip(np.abs(rr - mid) / half, None, 1.0)
            w_base = np.clip(1.0 - dist, 0.0, 1.0)
            rem = 1.0 - w_base
            w_fast_side = np.clip((rr - mid) / half, 0.0, 1.0)
            w_slow_side = np.clip((mid - rr) / half, 0.0, 1.0)

            w_fast = rem * w_fast_side
            w_slow = rem * w_slow_side

            out_arr = w_fast * fast_arr + w_base * base_arr + w_slow * slow_arr
            return pd.DataFrame(
                out_arr, index=base_df.index, columns=base_df.columns
            ).astype(np.float32)

        hyst = max(0.0, float(cfg.get("rv_selector_hysteresis", 0.02)))

        # Start with base
        out_arr = np.copy(base_arr)

        # Where > fast + hyst, use fast
        out_arr = np.where(rr > (fast_thr + hyst), fast_arr, out_arr)
        # Where < slow - hyst, use slow
        out_arr = np.where(rr < (slow_thr - hyst), slow_arr, out_arr)

        return pd.DataFrame(
            out_arr, index=base_df.index, columns=base_df.columns
        ).astype(np.float32)

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
    rz30_items: list[tuple[str, pd.DataFrame]] = []
    amihud_log = np.log(feats["amihud_illiq"] + 1e-12)
    rz30_items.append(("amihud_z", amihud_log))

    # Liquidity Gates (0 = average, -1 = good liquidity, -2 = excellent)
    # Since amihud is illiquidity, lower Z is better.
    vol_z_30_calm_src = np.log(feats["atr_pct_base"] + 1e-9)
    rz30_items.append(("vol_z_30_calm", vol_z_30_calm_src))

    rz30_out = _batch_roll_robust_zscore(rz30_items, 24 * 30)
    feats["amihud_z"] = rz30_out["amihud_z"].astype(np.float32)
    feats["vol_z_30_calm"] = rz30_out["vol_z_30_calm"].astype(np.float32)
    del amihud_log, vol_z_30_calm_src, rz30_out

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
    feats["vov_ratio"] = (feats["vov_mad_20"] / (feats["vov_mad_60"] + 1e-12)).astype(
        np.float32
    )
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
    feats["jump_rate_10h"] = ff.numba_rolling_mean(
        (feats["ret1h"].abs() > q90_dx).astype(np.float32), 10
    ).astype(np.float32)
    vol_mu_30d = _roll_mean("v", v, 24 * 30)
    vol_sd_30d = _roll_std("v", v, 24 * 30)
    feats["volu_z"] = ((v - vol_mu_30d) / (vol_sd_30d + 1e-12)).astype(np.float32)
    del max_bar, sign_max_bar, q90_dx, vol_mu_30d, vol_sd_30d
    feats["volume_price_corr_10h"] = (
        ff.numba_rolling_corr(feats["ret1h"].abs(), v, 10)
        .fillna(0.0)
        .astype(np.float32)
    )

    sma_fast = ff.numba_rolling_mean(c_context, max(24, int(cfg["trend_sma_n"] * 0.5)))
    sma_slow = ff.numba_rolling_mean(c_context, int(cfg["trend_sma_n"] * 2))
    trend_fast = c_context - sma_fast
    trend_slow = c_context - sma_slow
    feats["trend_pct"] = pick_by_rv(trend_fast, feats["trend_pct_base"], trend_slow)
    del sma_fast, sma_slow, trend_fast, trend_slow

    ema_fast_f = ema(c, max(4, int(cfg["ema_fast"] * 0.5)))
    ema_fast_s = ema(c, int(cfg["ema_fast"] * 2))
    dist_fast_f = (c - ema_fast_f) / (feats["atr_pct"] + 1e-12)
    dist_fast_s = (c - ema_fast_s) / (feats["atr_pct"] + 1e-12)
    feats["dist_ema_fast"] = pick_by_rv(
        dist_fast_f, feats["dist_ema_fast_base"], dist_fast_s
    )
    del ema_fast_f, ema_fast_s, dist_fast_f, dist_fast_s

    feats["vol_z24"] = feats["vol_z24_base"]
    feats["rsi_slope"] = feats["rsi"].diff(cfg["rsi_slope_n"]).astype(np.float32)
    feats["a_funding_proxy"] = feats["funding_proxy"]

    if bool(cfg.get("use_perps", False)):
        if isinstance(funding_panel, pd.DataFrame) and isinstance(
            oi_panel, pd.DataFrame
        ):
            tprint("Computing perp derivative features...")
            perp_price_panel = np.exp(c_log).astype(np.float32)
            volume_panel = np.exp(v).astype(np.float32)
            if isinstance(spot_close_panel, pd.DataFrame):
                spot_price_panel = spot_close_panel.reindex(
                    index=perp_price_panel.index,
                    columns=perp_price_panel.columns,
                ).astype(np.float32)
                spot_price_panel = spot_price_panel.where(
                    spot_price_panel > 0, perp_price_panel
                )
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
                    perp_buffers[feat_name][sym] = pd.to_numeric(
                        ser, errors="coerce"
                    ).astype(np.float32)

            for feat_name, by_sym in perp_buffers.items():
                feats[feat_name] = (
                    pd.DataFrame(by_sym)
                    .reindex(
                        index=perp_price_panel.index, columns=perp_price_panel.columns
                    )
                    .astype(np.float32)
                )
            tprint(f"Perp derivative features added: {len(perp_buffers)}")
        else:
            tprint(
                "Perps mode enabled but funding/open_interest data missing; skipping perp derivatives block."
            )

    # --- Orderbook/Funding/Cross-Asset extensions (graceful with missing panels) ---
    eps = 1e-12
    idx = c_log.index
    cols = c_log.columns
    close_panel = np.exp(c_log).astype(np.float32)

    def _zero_panel() -> pd.DataFrame:
        return pd.DataFrame(0.0, index=idx, columns=cols, dtype=np.float32)

    def _broadcast_series(ser: pd.Series) -> pd.DataFrame:
        arr = np.asarray(
            pd.to_numeric(ser, errors="coerce").fillna(0.0), dtype=np.float32
        )
        return pd.DataFrame(
            np.repeat(arr[:, None], len(cols), axis=1), index=idx, columns=cols
        )

    # funding-derived base building blocks
    fund_aligned = _zero_panel()
    if isinstance(funding_panel, pd.DataFrame):
        fund_aligned = (
            funding_panel.reindex(index=idx, columns=cols)
            .ffill()
            .fillna(0.0)
            .astype(np.float32)
        )

    feats["fund_rate"] = fund_aligned.clip(lower=-0.01, upper=0.01).astype(np.float32)
    feats["fund_rate_z_14d"] = _batch_roll_zscore(feats["fund_rate"], 14 * 24).clip(
        -6, 6
    )
    feats["fund_rate_mom_8h"] = _batch_roll_zscore(
        feats["fund_rate"].diff(8), 14 * 24
    ).clip(-6, 6)
    feats["fund_rate_mom_24h"] = _batch_roll_zscore(
        feats["fund_rate"].diff(24), 14 * 24
    ).clip(-6, 6)
    feats["fund_sign_persistence_3"] = (
        np.sign(feats["fund_rate"])
        .rolling(3, min_periods=1)
        .mean()
        .clip(-1, 1)
        .astype(np.float32)
    )
    feats["fund_abs_z"] = feats["fund_rate_z_14d"].abs().clip(0, 6).astype(np.float32)

    spot_panel = panel.get("spot_close") if isinstance(panel, dict) else None
    if not isinstance(spot_panel, pd.DataFrame):
        spot_panel = close_panel
    spot_panel = (
        spot_panel.reindex(index=idx, columns=cols).ffill().bfill().astype(np.float32)
    )
    basis_pct = ((close_panel - spot_panel) / (spot_panel + eps)).clip(-0.05, 0.05)
    feats["basis_pct"] = basis_pct.astype(np.float32)
    feats["basis_pct_z"] = _batch_roll_zscore(basis_pct, 14 * 24).clip(-6, 6)
    feats["basis_mom_4h"] = _batch_roll_zscore(basis_pct.diff(4), 14 * 24).clip(-6, 6)
    feats["basis_fund_div_z"] = (
        (feats["basis_pct_z"] - feats["fund_rate_z_14d"])
        .clip(-10, 10)
        .astype(np.float32)
    )

    # default orderbook features fallback to neutral zeros when snapshots are missing
    ob_default = _zero_panel()
    ob_names = [
        "ob_microprice_dev_bps",
        "ob_microprice_ret_1",
        "ob_imb_l1",
        "ob_imb_l5",
        "ob_imb_l10",
        "ob_imb_l20",
        "ob_wimb_l10",
        "ob_wimb_l20",
        "ob_slope_diff_l10",
        "ob_bid_depth_decay_l20",
        "ob_ask_depth_decay_l20",
        "ob_wall_imb_l20",
        "ob_gap_up_bps_l10",
        "ob_gap_dn_bps_l10",
        "ob_gap_skew_l10",
        "ob_book_pressure_l10",
        "ob_imb_chg_1",
        "ob_imb_accel_4",
        "ob_spread_bps",
        "ob_spread_z_24h",
        "ob_top_liquidity_usd",
        "ob_depth_usd_l10",
        "ob_depth_usd_l20",
        "ob_depth_usd_z_24h",
        "ob_depth_asym_stability_24h",
        "ob_snapshot_age_sec",
        "ob_update_gap_flag",
        "ob_stale_flag",
        "ob_mid_close_dislocation_bps",
        "ob_liquidity_shock_z",
    ]
    for name in ob_names:
        if name not in feats:
            feats[name] = ob_default.copy()

    feats["ob_gap_skew_l10"] = (
        feats["ob_gap_up_bps_l10"] - feats["ob_gap_dn_bps_l10"]
    ).astype(np.float32)
    feats["ob_book_pressure_l10"] = (
        feats["ob_wimb_l10"] * feats["ob_microprice_dev_bps"]
    ).astype(np.float32)
    feats["ob_imb_chg_1"] = _batch_roll_zscore(feats["ob_imb_l10"].diff(1), 24)
    feats["ob_imb_accel_4"] = _batch_roll_zscore(
        feats["ob_imb_l10"].diff(1).diff(4), 24 * 7
    )

    # cross-asset basket extensions (available-symbol mean only)
    basket = [s for s in list(cfg.get("market_basket", [])) if s in cols]
    if basket:
        mkt_ob_pressure = feats["ob_book_pressure_l10"][basket].mean(axis=1)
        mkt_funding = feats["fund_rate_z_14d"][basket].mean(axis=1)
    else:
        mkt_ob_pressure = pd.Series(0.0, index=idx, dtype=np.float32)
        mkt_funding = pd.Series(0.0, index=idx, dtype=np.float32)
    feats["xasset_asset_minus_mkt_ob_pressure"] = (
        feats["ob_book_pressure_l10"].sub(mkt_ob_pressure, axis=0).astype(np.float32)
    )
    feats["xasset_asset_minus_mkt_funding"] = (
        feats["fund_rate_z_14d"]
        .sub(mkt_funding, axis=0)
        .clip(-10, 10)
        .astype(np.float32)
    )
    feats["xasset_btc_ob_pressure"] = _broadcast_series(
        feats["ob_book_pressure_l10"].get("BTC/USDT", mkt_ob_pressure)
    )
    feats["xasset_eth_ob_pressure"] = _broadcast_series(
        feats["ob_book_pressure_l10"].get("ETH/USDT", mkt_ob_pressure)
    )
    feats["xasset_btc_funding_z"] = _broadcast_series(
        feats["fund_rate_z_14d"].get("BTC/USDT", mkt_funding)
    )
    lev_build = (
        (feats["fund_rate_z_14d"] > 1).astype(np.float32)
        + (feats["basis_pct_z"] > 1).astype(np.float32)
        + (feats["ob_imb_l10"] > 0).astype(np.float32)
    ) / 3.0
    feats["xasset_leverage_build_score"] = lev_build.astype(np.float32)

    feats["fund_time_to_next"] = _zero_panel()
    feats["fund_countdown_sin"] = _zero_panel()
    feats["fund_countdown_cos"] = _zero_panel()
    feats["fund_dispersion_basket"] = _zero_panel()
    feats["fund_extreme_share_basket"] = _zero_panel()
    feats["basis_dispersion_basket"] = _zero_panel()
    feats["xasset_mkt_spread_bps"] = _zero_panel()
    feats["xasset_mkt_depth_z"] = _zero_panel()
    feats["xasset_mkt_ob_stress"] = _zero_panel()
    feats["xasset_unwind_pressure"] = _zero_panel()

    # extended meta funding/orderbook feature set
    feats["fund_rate_ffill"] = feats["fund_rate"].astype(np.float32)
    feats["fund_abs_z_14d"] = (
        feats["fund_rate_z_14d"].abs().clip(0, 6).astype(np.float32)
    )
    feats["fund_carry_24h"] = _batch_roll_zscore(
        feats["fund_rate"].rolling(24, min_periods=1).sum(), 14 * 24
    )
    feats["fund_mom_8h"] = feats["fund_rate_mom_8h"].astype(np.float32)
    feats["fund_mom_24h"] = feats["fund_rate_mom_24h"].astype(np.float32)
    feats["fund_sign_persistence_24h"] = (
        np.sign(feats["fund_rate"])
        .rolling(24, min_periods=1)
        .mean()
        .clip(-1, 1)
        .astype(np.float32)
    )
    feats["fund_extreme_duration_24h"] = (
        (feats["fund_rate_z_14d"].abs() > 2)
        .astype(np.float32)
        .rolling(24, min_periods=1)
        .mean()
        .astype(np.float32)
    )
    feats["fund_rank_30d"] = (
        feats["fund_rate"]
        .rolling(24 * 30, min_periods=24)
        .rank(pct=True)
        .clip(0.01, 0.99)
        .astype(np.float32)
    )
    hours_to_next = _broadcast_series(
        pd.Series(np.mod(8 - (np.arange(len(idx)) % 8), 8), index=idx)
    )
    feats["fund_countdown_pressure"] = (
        (feats["fund_abs_z_14d"] * (1.0 - hours_to_next / 8.0))
        .clip(0, 6)
        .astype(np.float32)
    )

    feats["ob_top_liquidity_usd_z"] = _batch_roll_zscore(
        np.log1p(close_panel * (feats["ob_imb_l1"].abs() + 1.0)), 24 * 7
    )
    feats["ob_depth_usd_l10_z"] = _batch_roll_zscore(
        np.log1p(close_panel * (feats["ob_imb_l10"].abs() + 1.0)), 24 * 7
    )
    feats["ob_depth_usd_l20_z"] = _batch_roll_zscore(
        np.log1p(close_panel * (feats["ob_imb_l20"].abs() + 1.0)), 24 * 7
    )
    feats["ob_depth_ratio_l1_l20"] = (
        (
            (feats["ob_top_liquidity_usd_z"] + 10)
            / (feats["ob_depth_usd_l20_z"] + 10 + eps)
        )
        .clip(0, 1)
        .astype(np.float32)
    )
    feats["ob_imb_near_far_delta"] = (
        (feats["ob_imb_l1"] - feats["ob_imb_l20"]).clip(-2, 2).astype(np.float32)
    )
    feats["ob_depth_decay_asym_l20"] = _batch_roll_zscore(
        feats["ob_bid_depth_decay_l20"] - feats["ob_ask_depth_decay_l20"], 24 * 7
    )
    feats["ob_wall_skew_l20"] = feats["ob_wall_imb_l20"].astype(np.float32)

    basket_fund_std = (
        feats["fund_rate_z_14d"][basket].std(axis=1)
        if basket
        else pd.Series(0.0, index=idx)
    )
    feats["xasset_fund_dispersion_basket"] = _batch_roll_zscore(
        _broadcast_series(basket_fund_std), 14 * 24
    )
    basket_ext_share = (
        (feats["fund_rate_z_14d"][basket].abs() > 2).mean(axis=1)
        if basket
        else pd.Series(0.0, index=idx)
    ).astype(np.float32)
    feats["xasset_fund_extreme_share_basket"] = _broadcast_series(basket_ext_share)
    feats["xasset_asset_minus_basket_fund_z"] = feats[
        "xasset_asset_minus_mkt_funding"
    ].astype(np.float32)
    feats["xasset_btc_fund_z"] = (
        feats["xasset_btc_funding_z"].clip(-6, 6).astype(np.float32)
    )
    feats["xasset_ob_stress_basket"] = _broadcast_series(
        (
            feats["ob_spread_z_24h"].mean(axis=1)
            - feats["ob_depth_usd_l20_z"].mean(axis=1)
        ).clip(-10, 10)
    )
    feats["xasset_asset_minus_basket_ob_pressure"] = _batch_roll_zscore(
        feats["xasset_asset_minus_mkt_ob_pressure"], 24 * 7
    )
    feats["xasset_ob_liquidity_divergence"] = (
        feats["ob_depth_usd_l20_z"].sub(
            feats["ob_depth_usd_l20_z"].mean(axis=1), axis=0
        )
    ).astype(np.float32)

    rv24z = _batch_roll_zscore(feats["rv_24h"], 14 * 24)
    trend_strength = feats.get("trend_strength_percentile", _zero_panel())
    feats["fund_abs_z_x_ret24h_sign"] = (
        (feats["fund_abs_z_14d"] * np.sign(feats["ret24h"]).astype(np.float32))
        .clip(-6, 6)
        .astype(np.float32)
    )
    feats["fund_abs_z_x_rv_24h"] = (
        (feats["fund_abs_z_14d"] * rv24z).clip(-12, 12).astype(np.float32)
    )
    feats["fund_z_x_trend_strength"] = (
        (feats["fund_rate_z_14d"] * trend_strength).clip(-6, 6).astype(np.float32)
    )
    feats["ob_pressure_x_ret4h_sign"] = _batch_roll_zscore(
        feats["ob_book_pressure_l10"] * np.sign(feats["ret4h"]).astype(np.float32),
        24 * 7,
    )
    feats["ob_spread_z_x_rv_24h"] = (
        (feats["ob_spread_z_24h"] * rv24z).clip(-12, 12).astype(np.float32)
    )
    feats["ob_depth_z_x_rvol_z"] = (
        (feats["ob_depth_usd_l20_z"] * feats["rvol_z"]).clip(-12, 12).astype(np.float32)
    )

    orderbook_feature_keys = set(cfg.get("ORDERBOOK_FEATURE_KEYS", []))
    if not orderbook_feature_keys and requested_feature_set:
        orderbook_feature_keys = {k for k in requested_feature_set if k.startswith("ob_")}
    should_compute_orderbook = bool(cfg.get("enable_orderbook_features", False)) and (
        not requested_feature_set
        or bool(orderbook_feature_keys.intersection(requested_feature_set))
    )
    if should_compute_orderbook:
        ob_snapshot_feats = compute_orderbook_snapshot_features(
            panel.get("orderbook_hourly") if isinstance(panel, dict) else None,
            close_panel=close_panel,
            volume_panel=panel.get("volume", np.exp(v)).astype(np.float32),
            atr_panel=feats.get("atr_pct"),
            cfg=cfg,
            shift_bars=int(cfg.get("microstructure_shift_bars", 1)),
        )
        feats.update(ob_snapshot_feats)

    requested_obw = any(k.startswith("obw_") or k.startswith("_obw_") for k in requested_feature_set)
    if bool(cfg.get("enable_orderbook_wall_features", True)) and (
        not requested_feature_set or requested_obw
    ):
        wall_primitives = compute_orderbook_wall_primitives(
            panel.get("orderbook_hourly") if isinstance(panel, dict) else None,
            close_panel=close_panel,
            volume_panel=np.exp(v).astype(np.float32),
            atr_panel=feats.get("atr_pct"),
            shift_bars=int(cfg.get("microstructure_shift_bars", 1)),
        )
        feats.update(wall_primitives)
    inferred_side_long = (feats.get("ret4h", _zero_panel()) >= 0).astype(np.float32)
    for band in ("r005", "r010", "r020", "r030", "a05", "a10", "a20", "a30"):
        bid_to = feats.get(f"_obw_bid_wall_to_vol_{band}", _zero_panel())
        ask_to = feats.get(f"_obw_ask_wall_to_vol_{band}", _zero_panel())
        bid_p = feats.get(f"_obw_bid_wall_pressure_{band}", _zero_panel())
        ask_p = feats.get(f"_obw_ask_wall_pressure_{band}", _zero_panel())
        bid_d = feats.get(f"_obw_bid_wall_distance_{band}", _zero_panel())
        ask_d = feats.get(f"_obw_ask_wall_distance_{band}", _zero_panel())
        bid_path = feats.get(f"_obw_bid_path_depth_to_target_{band}", _zero_panel())
        ask_path = feats.get(f"_obw_ask_path_depth_to_target_{band}", _zero_panel())
        blocking_to = np.where(inferred_side_long > 0, ask_to, bid_to)
        support_to = np.where(inferred_side_long > 0, bid_to, ask_to)
        blocking_p = np.where(inferred_side_long > 0, ask_p, bid_p)
        blocking_d = np.where(inferred_side_long > 0, ask_d, bid_d)
        path_to = np.where(inferred_side_long > 0, ask_path, bid_path)
        feats[f"obw_blocking_wall_to_vol_{band}"] = pd.DataFrame(
            blocking_to, index=idx, columns=cols
        ).astype(np.float32)
        feats[f"obw_support_wall_to_vol_{band}"] = pd.DataFrame(
            support_to, index=idx, columns=cols
        ).astype(np.float32)
        feats[f"obw_blocking_minus_support_wall_{band}"] = (
            (
                (
                    feats[f"obw_blocking_wall_to_vol_{band}"]
                    - feats[f"obw_support_wall_to_vol_{band}"]
                )
                / (
                    feats[f"obw_blocking_wall_to_vol_{band}"]
                    + feats[f"obw_support_wall_to_vol_{band}"]
                    + eps
                )
            )
            .clip(-1, 1)
            .astype(np.float32)
        )
        feats[f"obw_blocking_wall_pressure_{band}"] = pd.DataFrame(
            blocking_p, index=idx, columns=cols
        ).astype(np.float32)
        feats[f"obw_blocking_wall_distance_{band}"] = (
            pd.DataFrame(blocking_d, index=idx, columns=cols)
            .clip(0, 1)
            .astype(np.float32)
        )
        feats[f"obw_path_depth_to_target_{band}"] = pd.DataFrame(
            path_to, index=idx, columns=cols
        ).astype(np.float32)

    # Cross-sectional regime / relative-value meta features
    basket_cols = [c for c in cols if c in cfg.get("market_basket", [])] or list(cols)
    ret1 = feats.get("ret1h", _zero_panel()).astype(np.float32)
    ret4 = feats.get("ret4h", _zero_panel()).astype(np.float32)
    ret24 = feats.get("ret24h", _zero_panel()).astype(np.float32)
    ret48 = feats.get("ret48h", _zero_panel()).astype(np.float32)
    rv24 = feats.get("rv_24h", _zero_panel()).astype(np.float32)
    rvol = feats.get("rvol_z", _zero_panel()).astype(np.float32)
    volz = feats.get("vol_z", _zero_panel()).astype(np.float32)

    med4 = ret4.median(axis=1)
    med24 = ret24.median(axis=1)
    med48 = ret48.median(axis=1)
    feats["asset_minus_universe_median_ret_4h"] = ret4.sub(med4, axis=0).astype(
        np.float32
    )
    feats["asset_minus_universe_median_ret_24h"] = ret24.sub(med24, axis=0).astype(
        np.float32
    )
    feats["asset_minus_universe_median_ret_48h"] = ret48.sub(med48, axis=0).astype(
        np.float32
    )
    feats["asset_mom_minus_basket_mom_4h"] = ret4.sub(
        ret4[basket_cols].median(axis=1), axis=0
    ).astype(np.float32)
    feats["asset_mom_minus_basket_mom_24h"] = ret24.sub(
        ret24[basket_cols].median(axis=1), axis=0
    ).astype(np.float32)

    btc4 = ret4.get("BTC/USDT", med4)
    eth4 = ret4.get("ETH/USDT", med4)
    mix4 = 0.5 * (btc4 + eth4)
    feats["resid_ret_vs_btc_4h"] = ret4.sub(btc4, axis=0).astype(np.float32)
    feats["resid_ret_vs_eth_4h"] = ret4.sub(eth4, axis=0).astype(np.float32)
    feats["resid_ret_vs_btceth_4h"] = ret4.sub(mix4, axis=0).astype(np.float32)
    beta = (
        ret24.rolling(24, min_periods=8)
        .corr(pd.concat([btc4] * len(cols), axis=1).set_axis(cols, axis=1))
        .fillna(0)
    )
    feats["beta_adj_resid_ret_24h"] = (
        ret24 - beta * pd.concat([btc4] * len(cols), axis=1).set_axis(cols, axis=1)
    ).astype(np.float32)

    feats["rv_rel_universe"] = (rv24 / (rv24.median(axis=1) + 1e-12)).astype(np.float32)
    feats["vol_surprise_rel_peers"] = (volz.sub(volz.median(axis=1), axis=0)).astype(
        np.float32
    )
    feats["ret_rank_universe"] = ret4.rank(axis=1, pct=True).astype(np.float32)
    feats["vol_surprise_rank"] = volz.rank(axis=1, pct=True).astype(np.float32)
    feats["volatility_rank"] = rv24.rank(axis=1, pct=True).astype(np.float32)
    feats["momentum_percentile"] = ret24.rank(axis=1, pct=True).astype(np.float32)

    disp4 = ret4.std(axis=1).astype(np.float32)
    disp24 = ret24.std(axis=1).astype(np.float32)
    feats["cs_dispersion_ret_4h"] = _broadcast_series(disp4)
    feats["cs_dispersion_ret_24h"] = _broadcast_series(disp24)
    feats["pct_assets_up_1h"] = _broadcast_series(
        (ret1 > 0).mean(axis=1).astype(np.float32)
    )
    feats["pct_assets_up_4h"] = _broadcast_series(
        (ret4 > 0).mean(axis=1).astype(np.float32)
    )
    feats["pct_assets_up_24h"] = _broadcast_series(
        (ret24 > 0).mean(axis=1).astype(np.float32)
    )
    feats["pct_assets_above_ema_fast"] = _broadcast_series(
        (feats.get("dist_ema_fast", _zero_panel()) > 0).mean(axis=1).astype(np.float32)
    )
    feats["pct_assets_above_vwap"] = _broadcast_series(
        (feats.get("dist_weekly_vwap", _zero_panel()) > 0)
        .mean(axis=1)
        .astype(np.float32)
    )

    corr_proxy = ret1.rolling(24, min_periods=8).corr(ret1.median(axis=1)).fillna(0.0)
    feats["avg_pair_corr_24h"] = _broadcast_series(
        corr_proxy.mean(axis=1).astype(np.float32)
    )
    feats["corr_concentration_24h"] = _broadcast_series(
        corr_proxy.std(axis=1).astype(np.float32)
    )

    def _pct_rank_series(ser):
        return ser.rolling(24 * 14, min_periods=24).rank(pct=True).clip(0.01, 0.99)

    btc24 = ret24.get("BTC/USDT", med24)
    btc48 = ret48.get("BTC/USDT", med48)
    feats["btc_ret_4h_pct"] = _broadcast_series(_pct_rank_series(btc4))
    feats["btc_ret_24h_pct"] = _broadcast_series(_pct_rank_series(btc24))
    feats["btc_ret_48h_pct"] = _broadcast_series(_pct_rank_series(btc48))

    btc_rv = rv24.get("BTC/USDT", rv24.median(axis=1))
    eth_rv = rv24.get("ETH/USDT", rv24.median(axis=1))
    feats["btc_rv_ratio_1h24h_pct"] = _broadcast_series(
        _pct_rank_series((btc_rv / (btc_rv.rolling(24, min_periods=1).mean() + 1e-12)))
    )
    feats["btc_rv_ratio_4h24h_pct"] = _broadcast_series(
        _pct_rank_series(
            (
                btc_rv.rolling(4, min_periods=1).mean()
                / (btc_rv.rolling(24, min_periods=1).mean() + 1e-12)
            )
        )
    )
    feats["eth_rv_ratio_1h24h_pct"] = _broadcast_series(
        _pct_rank_series((eth_rv / (eth_rv.rolling(24, min_periods=1).mean() + 1e-12)))
    )
    feats["eth_rv_ratio_4h24h_pct"] = _broadcast_series(
        _pct_rank_series(
            (
                eth_rv.rolling(4, min_periods=1).mean()
                / (eth_rv.rolling(24, min_periods=1).mean() + 1e-12)
            )
        )
    )

    feats["cs_ret_dispersion_4h_pct"] = _broadcast_series(_pct_rank_series(disp4))
    feats["cs_ret_dispersion_24h_pct"] = _broadcast_series(_pct_rank_series(disp24))
    feats["asset_ret_vs_btc_4h"] = ret4.sub(btc4, axis=0).astype(np.float32)
    feats["asset_ret_vs_btc_24h"] = ret24.sub(btc24, axis=0).astype(np.float32)
    feats["asset_ret_vs_btc_48h"] = ret48.sub(btc48, axis=0).astype(np.float32)
    feats["asset_ret_vs_universe_4h"] = feats["asset_minus_universe_median_ret_4h"]
    feats["asset_ret_vs_universe_24h"] = feats["asset_minus_universe_median_ret_24h"]
    feats["asset_ret_vs_universe_48h"] = feats["asset_minus_universe_median_ret_48h"]

    feats["median_rvol_z"] = _broadcast_series(rvol.median(axis=1).astype(np.float32))
    feats["pct_assets_high_rvol"] = _broadcast_series(
        (rvol > 1.0).mean(axis=1).astype(np.float32)
    )
    feats["median_spread_bps"] = _broadcast_series(
        feats.get("ob_spread_bps", _zero_panel()).median(axis=1).astype(np.float32)
    )
    feats["pct_assets_wide_spread"] = _broadcast_series(
        (feats.get("ob_spread_bps", _zero_panel()) > 10.0)
        .mean(axis=1)
        .astype(np.float32)
    )
    feats["median_volume_z"] = _broadcast_series(volz.median(axis=1).astype(np.float32))

    # --- Regime Conditioning Features ---
    if bool(cfg.get("use_regime_features", True)):
        feats.update(
            compute_regime_features(
                c,
                h,
                l,
                v,
                atr_base,
                mkt_gates,
                rv_24_cache=feats["rv_24h"],
                input_feats=feats,
            )
        )

    # --- New Helper Features for Models ---
    dir_s = np.sign(feats["ret24h"])
    dir_s[dir_s == 0] = 1  # fallback

    atr = feats["atr_pct"] + np.float32(1e-12)
    rv6 = feats["rv_6h"] + np.float32(1e-12)
    rv8 = feats["rv_8h"] + np.float32(1e-12)
    rv12 = feats["rv_12h"] + np.float32(1e-12)

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
        feats[f"trend_slope_{k_trend}h"] = ((c_context - sma_k) / (atr + 1e-12)).astype(
            np.float32
        )
        # Trend acceleration: is the trend strengthening or weakening?
        feats[f"trend_accel_{k_trend}h"] = (
            feats[f"trend_slope_{k_trend}h"].diff(12).fillna(0.0).astype(np.float32)
        )
        del sma_k

    # --- Event timing + policy-normalized stage difficulty (entry-time, past-only) ---
    eps = 1e-12
    c_prev = c_context.shift(1)
    h_prev = h.shift(1)
    l_prev = l.shift(1)

    def _rolling_bars_since_extreme(
        df: pd.DataFrame, window: int, mode: str
    ) -> pd.DataFrame:
        return ff.numba_rolling_bars_since_extreme(df, window, mode)

    # Time since local peak/trough in the last 12h window (all windows end at t-1)
    time_since_peak_12h = _rolling_bars_since_extreme(h_prev, 12, "max")
    time_since_trough_12h = _rolling_bars_since_extreme(l_prev, 12, "min")
    # Event-direction proxy: if 12h return into t-1 is up, use peak timing; else trough timing.
    up_dir = (c_prev / c_prev.shift(12) - 1.0) >= 0.0
    feats["time_since_peak_12h"] = time_since_peak_12h.fillna(0.0).astype(np.float32)
    feats["time_since_trough_12h"] = time_since_trough_12h.fillna(0.0).astype(
        np.float32
    )
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
    vol_ratio = (v_prev / (v_prev.rolling(24, min_periods=1).median() + eps)).fillna(
        1.0
    )
    feats["second_leg_accel_1h"] = accel_1.astype(np.float32)
    feats["second_leg_accel_2h"] = accel_2.astype(np.float32)
    feats["second_leg_accel_vol_1h"] = (accel_1 * vol_ratio).astype(np.float32)
    feats["second_leg_accel_vol_2h"] = (accel_2 * vol_ratio).astype(np.float32)

    # Policy-normalized stage difficulty / timing proxies (entry-time only).
    vol_scale = feats["atr_pct"].shift(1).fillna(feats["atr_pct"]).clip(lower=eps)
    hr_48 = (
        feats["ret1h"]
        .abs()
        .shift(1)
        .rolling(48, min_periods=1)
        .median()
        .clip(lower=eps)
    )
    be_threshold_pct = float(cfg.get("be_threshold_pct", 0.0035))
    profit_lock_pct = float(cfg.get("profit_lock_pct", 0.0050))
    tp_mult = float(cfg.get("tp_mult", 0.50))
    giveback_pct = float(cfg.get("giveback_pct", 0.35))
    trail_act_pct = tp_mult * vol_scale

    feats["vol_scale"] = vol_scale.astype(np.float32)
    feats["be_vol_units"] = (be_threshold_pct / (vol_scale + eps)).astype(np.float32)
    feats["pl_vol_units"] = (profit_lock_pct / (vol_scale + eps)).astype(np.float32)
    feats["trail_act_pct"] = trail_act_pct.astype(np.float32)
    feats["trail_act_vol_units"] = (trail_act_pct / (vol_scale + eps)).astype(
        np.float32
    )
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
    feats["shock_vol_ratio"] = (shock_12h / ((vol_scale * np.sqrt(12.0)) + eps)).astype(
        np.float32
    )
    feats["dist_from_low_event_12h"] = dist_from_low.astype(np.float32)
    feats["dist_from_high_event_12h"] = dist_from_high.astype(np.float32)
    feats["dist_from_low_vol"] = (dist_from_low / (vol_scale + eps)).astype(np.float32)
    feats["dist_from_high_vol"] = (dist_from_high / (vol_scale + eps)).astype(
        np.float32
    )

    # Realized volatility at longer horizons
    feats["rv_48h"] = ff.apply_to_frame(
        feats["ret1h"], ff._numba_rolling_std_nan_safe, 48
    ).astype(np.float32)
    feats["rv_120h"] = ff.apply_to_frame(
        feats["ret1h"], ff._numba_rolling_std_nan_safe, 120
    ).astype(np.float32)
    # Vol regime ratio: short-term vs multi-day vol
    feats["rv_ratio_24_120"] = (feats["rv_24h"] / (feats["rv_120h"] + 1e-12)).astype(
        np.float32
    )

    # --- Multi-Horizon Aggregated Features (Report 2026-02-10) ---
    # Vectorized aggregate statistics across multiple return windows
    # feats["ret1h"] etc. are DataFrames (T, S); stack along axis=2 → (T, S, N)
    _ret_ref = feats["ret1h"]
    ret_stack = np.stack(
        [
            feats["ret1h"].to_numpy(),
            feats["ret2h"].to_numpy(),
            feats["ret4h"].to_numpy(),
            feats["ret6h"].to_numpy(),
            feats["ret8h"].to_numpy(),
        ],
        axis=2,
    )
    feats["ret_mean"] = pd.DataFrame(
        np.nanmean(ret_stack, axis=2), index=_ret_ref.index, columns=_ret_ref.columns
    ).astype(np.float32)
    feats["ret_max"] = pd.DataFrame(
        np.nanmax(ret_stack, axis=2), index=_ret_ref.index, columns=_ret_ref.columns
    ).astype(np.float32)
    feats["ret_min"] = pd.DataFrame(
        np.nanmin(ret_stack, axis=2), index=_ret_ref.index, columns=_ret_ref.columns
    ).astype(np.float32)
    del ret_stack

    # Vectorized aggregate statistics across multiple volatility windows
    _rv_ref = feats["rv_2h"]
    rv_stack = np.stack(
        [
            feats["rv_2h"].to_numpy(),
            feats["rv_4h"].to_numpy(),
            feats["rv_6h"].to_numpy(),
            feats["rv_8h"].to_numpy(),
            feats["rv_12h"].to_numpy(),
            feats["rv_24h"].to_numpy(),
        ],
        axis=2,
    )
    feats["rv_mean"] = pd.DataFrame(
        np.nanmean(rv_stack, axis=2), index=_rv_ref.index, columns=_rv_ref.columns
    ).astype(np.float32)
    feats["rv_max"] = pd.DataFrame(
        np.nanmax(rv_stack, axis=2), index=_rv_ref.index, columns=_rv_ref.columns
    ).astype(np.float32)
    feats["rv_min"] = pd.DataFrame(
        np.nanmin(rv_stack, axis=2), index=_rv_ref.index, columns=_rv_ref.columns
    ).astype(np.float32)
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
    feats["ret_pct5_24h"] = (
        pd.DataFrame(ret_pct5, index=_ret1h_ref.index, columns=_ret1h_ref.columns)
        .shift(1)
        .astype(np.float32)
    )
    feats["ret_pct95_24h"] = (
        pd.DataFrame(ret_pct95, index=_ret1h_ref.index, columns=_ret1h_ref.columns)
        .shift(1)
        .astype(np.float32)
    )

    # gap_zscore: Overnight gap z-score relative to recent gaps
    # Vectorized gap calculation with Numba rolling stats
    gap_df = o - c.shift(1)
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
        index=_ret1h_ref.index,
        columns=_ret1h_ref.columns,
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
        index=_rv6_ref.index,
        columns=_rv6_ref.columns,
    ).astype(np.float32)

    # range_zscore: Range (high-low) z-score
    range_hl_df = h - l
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
        index=range_hl_df.index,
        columns=range_hl_df.columns,
    ).astype(np.float32)

    # tail_risk_score: Combined tail risk metric (vectorized)
    # High when: negative tail returns, high vol shock, large gaps
    ret_pct5_arr = feats["ret_pct5_24h"].to_numpy()
    vol_shock_arr = feats["vol_shock_z"].to_numpy()
    gap_zscore_arr = feats["gap_zscore"].to_numpy()
    feats["tail_risk_score"] = pd.DataFrame(
        np.clip(-ret_pct5_arr, 0, None) * 0.4
        + np.clip(vol_shock_arr, 0, None) * 0.3  # Negative tail returns
        + np.abs(gap_zscore_arr) * 0.3,  # Vol spikes  # Large gaps
        index=_ret1h_ref.index,
        columns=_ret1h_ref.columns,
    ).astype(np.float32)

    feats["excess_6h"] = (feats["ret1h"].abs() / rv6).astype(np.float32)
    feats["excess_12h"] = (feats["ret1h"].abs() / rv12).astype(np.float32)

    for k in [2, 4, 8]:
        feats[f"ft_{k}"] = (feats[f"ret{k}h"] / (feats["ret1h"].abs() + 1e-12)).astype(
            np.float32
        )
        feats[f"failure_{k}"] = (-1 * feats[f"ft_{k}"]).clip(lower=0).astype(np.float32)

    # clv: (2c - h - l) / (h - l). h-l can be 0.
    clv_raw = ((2 * c - h - l) / ((h - l) + 1e-9)).fillna(0.0)
    feats["clv"] = clv_raw.astype(np.float32)
    feats["clv_mean_2"] = (
        ff.numba_rolling_mean(feats["clv"], 2).fillna(0.0).astype(np.float32)
    )
    feats["clv_mean_4"] = (
        ff.numba_rolling_mean(feats["clv"], 4).fillna(0.0).astype(np.float32)
    )

    for k in [3, 6]:
        v_sum = ff.numba_rolling_sum(v, k)
        ret_k_abs = feats[f"ret{k if k in [6] else 1}h"].abs()
        if k == 3:
            ret_k_abs = ff.numba_rolling_sum(c, 3).abs()

        feats[f"evr_{k}"] = (v_sum / (ret_k_abs + 1e-12)).astype(np.float32)

    feats["progress"] = (feats["ret1h"].abs() / (v + 1e-12)).astype(np.float32)
    feats["speed"] = (feats["ret1h"].abs() / atr).astype(np.float32)

    tail_denom = feats["up_vol_6"] + feats["dn_vol_6"] + np.float32(1e-12)
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
    feats["dir_path_edge_2h"] = (
        feats["dir_path_long_2h"] - feats["dir_path_short_2h"]
    ).astype(np.float32)
    feats["dir_path_risk_skew_2h"] = (
        feats["dir_path_risk_long_2h"] - feats["dir_path_risk_short_2h"]
    ).astype(np.float32)
    del o_entry, h_max_4, l_min_4, mfe_long, mae_long, mfe, mae, cur_pnl, gb

    # --- Memory checkpoint: free GC before composite features ---
    tprint(
        f"Features: {len(feats)} base features computed. Running GC before composites..."
    )
    gc.collect()

    # --- COMPOSITE / INTERACTION FEATURES ---

    # 1/ Exhaustion
    feats["overext"] = (
        (feats["donch_dist_12"] * feats["excess_6h"]).fillna(0.0).astype(np.float32)
    )
    feats["overext_weak"] = (
        (feats["donch_dist_12"] * (1.0 - feats["clv_mean_4"].clip(lower=0)))
        .fillna(0.0)
        .astype(np.float32)
    )
    feats["effort_gate"] = (
        (feats["evr_6"] * (feats["vol_z24"] + 1.0) / (feats["progress"] + 1e-12))
        .fillna(0.0)
        .astype(np.float32)
    )
    feats["stall_ext"] = (
        (feats["donch_dist_12"] * (1.0 - feats["delta_stall_6"]))
        .fillna(0.0)
        .astype(np.float32)
    )
    feats["tail_fail"] = (
        (feats["tail_against"] * (feats["ft_2"] - feats["ft_4"]).clip(lower=0))
        .fillna(0.0)
        .astype(np.float32)
    )

    pb_avg = (feats["pullback_2"] + feats["pullback_4"]) / 2.0
    fail_term = feats["failure_2"] + 0.5 * feats["failure_4"]
    feats["reject_score"] = (
        ((1.0 - feats["clv_mean_4"].clip(lower=0)) * pb_avg * fail_term)
        .fillna(0.0)
        .astype(np.float32)
    )

    feats["impulse_ratio_24"] = (
        (feats["ret1h"].abs() / (feats["ret24h"].abs() + 1e-12))
        .fillna(0.0)
        .astype(np.float32)
    )
    feats["impulse_ratio_12"] = (
        (feats["ret1h"].abs() / (feats["ret12h"].abs() + 1e-12))
        .fillna(0.0)
        .astype(np.float32)
    )
    feats["accel"] = (feats["ret1h"] - feats["ret1h"].shift(1)).abs() / (
        feats["rv_6h"] + np.float32(1e-12)
    )
    feats["blowoff_risk"] = (
        (feats["impulse_ratio_24"] * feats["accel"] * feats["donch_dist_12"])
        .fillna(0.0)
        .astype(np.float32)
    )

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
    kalman_lambda = tune_global_kalman_lambda(
        feats_S_train, feats_ret1h_train, grid_size=15
    )

    score_rm24 = ff.numba_rolling_mean(feats["S"], 24).shift(1).astype(np.float32)
    vol_ratio_input = feats.get(
        "liquidity_ratio",
        (v / (ff.numba_rolling_mean(v, 24 * 30).shift(1) + EPS)).astype(np.float32),
    )

    kf_score_mean, kf_innov_var, kf_state_unc, r_score = _kalman_local_level_df(
        feats["S"], kalman_lambda
    )
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
    r_score_df = pd.DataFrame(
        np.repeat(r_score.values.reshape(1, -1), len(c.index), axis=0),
        index=c.index,
        columns=c.columns,
    ).astype(np.float32)
    q_score_df = (kalman_lambda * r_score_df).astype(np.float32)
    feats["kf_snr_est"] = (q_score_df / (r_score_df + EPS)).astype(np.float32)

    # --- New Kalman Trends on Price, Realized Volatility, and Log Volume ---
    log_price_df = np.log(c)
    rv_input = feats.get("rv_24h", ff.numba_rolling_std(feats["ret1h"], 24)).astype(
        np.float32
    )
    log_vol_df = np.log1p(v).astype(np.float32)

    kf_price_state, kf_price_innov_var, kf_price_unc, r_price = _kalman_local_level_df(
        log_price_df, kalman_lambda
    )
    kf_vol_state, kf_vol_innov_var, kf_vol_unc, r_vol = _kalman_local_level_df(
        rv_input, kalman_lambda
    )
    (
        kf_log_vol_state,
        kf_log_vol_innov_var,
        kf_log_vol_unc,
        r_log_vol,
    ) = _kalman_local_level_df(log_vol_df, kalman_lambda)

    # Base model keys
    feats["kalman_price"] = kf_price_state.astype(np.float32)
    feats["price_state_slope_1h"] = (kf_price_state - kf_price_state.shift(1)).astype(
        np.float32
    )
    feats["price_state_slope_6h"] = (kf_price_state - kf_price_state.shift(6)).astype(
        np.float32
    )
    feats["price_state_slope_ratio_1h_6h"] = (
        feats["price_state_slope_1h"] / (feats["price_state_slope_6h"].abs() + EPS)
    ).astype(np.float32)

    # Calculate price minus state standardized by innovation standard deviation
    price_minus_state = log_price_df - kf_price_state
    price_innovation_std = np.sqrt(kf_price_innov_var + EPS)
    feats["price_minus_state_z"] = (price_minus_state / price_innovation_std).astype(
        np.float32
    )

    # Meta model keys
    feats["price_innovation_z"] = feats["price_minus_state_z"]
    feats["rolling_std(price_innovation)"] = _roll_std(
        "price_innovation_z", feats["price_innovation_z"], 24
    )
    feats["state_uncertainty_1h"] = kf_price_unc.astype(np.float32)

    # Kalman gain estimate: P / (P + R) (Simplified scalar or approx using variance)
    # Using the uncertainty and observation variance
    r_price_df = pd.DataFrame(
        np.repeat(r_price.values.reshape(1, -1), len(c.index), axis=0),
        index=c.index,
        columns=c.columns,
    ).astype(np.float32)
    feats["kalman_gain_1h"] = (kf_price_unc / (kf_price_unc + r_price_df + EPS)).astype(
        np.float32
    )

    feats["vol_state_slope_1h"] = (kf_vol_state - kf_vol_state.shift(1)).astype(
        np.float32
    )
    feats["realized_vol_minus_vol_state"] = (rv_input - kf_vol_state).astype(np.float32)
    feats["log_volume_state_1h"] = kf_log_vol_state.astype(np.float32)
    feats["volume_state_slope_1h"] = (
        kf_log_vol_state - kf_log_vol_state.shift(1)
    ).astype(np.float32)

    # volume surprise is typically log_volume - state
    volume_surprise = log_vol_df - kf_log_vol_state
    feats["price_slope_x_volume_surprise"] = (
        feats["price_state_slope_1h"] * volume_surprise
    ).astype(np.float32)
    feats["vol_state_x_volume_state"] = (kf_vol_state * kf_log_vol_state).astype(
        np.float32
    )

    # Position sizer keys
    feats["vol_state_1h"] = kf_vol_state.astype(np.float32)

    # Need short and long vol state for short_vol_state_over_long_vol_state
    # Let's say long is 24h vol state and short is 4h vol state, or kalman vs simple.
    # The requirement is short_vol_state_over_long_vol_state. We can create a longer window kalman
    kf_vol_state_long, _, _, _ = _kalman_local_level_df(
        rv_input, kalman_lambda * 0.1
    )  # Slower adaptation
    feats["short_vol_state_over_long_vol_state"] = (
        kf_vol_state / (kf_vol_state_long + EPS)
    ).astype(np.float32)

    volume_state_std = np.sqrt(kf_log_vol_innov_var + EPS)
    feats["volume_surprise_vs_state"] = (volume_surprise / volume_state_std).astype(
        np.float32
    )

    # --- New Trend Stack / Alignment features for base models ---
    nATR_36h = ff.numba_rolling_mean(h - l, 36) / c
    nATR_36h_eps = nATR_36h + EPS

    zr_1h = feats.get("ret1h", c.pct_change(1)) / nATR_36h_eps
    zr_3h = feats.get("ret3h", c.pct_change(3)) / nATR_36h_eps
    zr_6h = feats.get("ret6h", c.pct_change(6)) / nATR_36h_eps
    zr_12h = feats.get("ret12h", c.pct_change(12)) / nATR_36h_eps
    zr_24h = feats.get("ret24h", c.pct_change(24)) / nATR_36h_eps

    feats["trend_stack_3_6_12"] = (zr_3h + zr_6h + zr_12h).astype(np.float32)
    feats["trend_stack_6_12_24"] = (zr_6h + zr_12h + zr_24h).astype(np.float32)

    feats["zr_1h_minus_zr_6h"] = (zr_1h - zr_6h).astype(np.float32)
    feats["zr_3h_minus_zr_12h"] = (zr_3h - zr_12h).astype(np.float32)
    feats["zr_6h_minus_zr_24h"] = (zr_6h - zr_24h).astype(np.float32)

    # Trend dispersion
    # std across features per row. pd.DataFrame.std does this efficiently
    feats["trend_dispersion_1_3_6"] = pd.DataFrame(
        np.std([zr_1h.values, zr_3h.values, zr_6h.values], axis=0),
        index=c.index,
        columns=c.columns,
    ).astype(np.float32)
    feats["trend_dispersion_3_6_12"] = pd.DataFrame(
        np.std([zr_3h.values, zr_6h.values, zr_12h.values], axis=0),
        index=c.index,
        columns=c.columns,
    ).astype(np.float32)

    # Note: price_innovation_z may be a df depending on context
    innovation_z = feats.get("price_innovation_z")
    if innovation_z is not None:
        feats["innovation_z_x_zr_1h"] = (innovation_z * zr_1h).astype(np.float32)
        feats["innovation_z_x_zr_3h"] = (innovation_z * zr_3h).astype(np.float32)

    # Volume and range Z
    # We'll calculate simple Z scores if not existing.
    vol_24h_mean = _roll_mean("vol_24", v, 24)
    vol_24h_std = _roll_std("vol_24", v, 24)
    vol_z_24h = (v - vol_24h_mean) / (vol_24h_std + EPS)

    vol_48h_mean = _roll_mean("vol_48", v, 48)
    vol_48h_std = _roll_std("vol_48", v, 48)
    vol_z_48h = (v - vol_48h_mean) / (vol_48h_std + EPS)

    range_hl = h - l
    range_24h_mean = _roll_mean("range_hl_24", range_hl, 24)
    range_24h_std = _roll_std("range_hl_24", range_hl, 24)
    range_z_24h = (range_hl - range_24h_mean) / (range_24h_std + EPS)

    range_48h_mean = _roll_mean("range_hl_48", range_hl, 48)
    range_48h_std = _roll_std("range_hl_48", range_hl, 48)
    range_z_48h = (range_hl - range_48h_mean) / (range_48h_std + EPS)

    feats["zr_1h_x_volume_z_24h"] = (zr_1h * vol_z_24h).astype(np.float32)
    feats["zr_3h_x_volume_z_24h"] = (zr_3h * vol_z_24h).astype(np.float32)
    feats["zr_6h_x_volume_z_48h"] = (zr_6h * vol_z_48h).astype(np.float32)

    feats["zr_6h_x_range_z_24h"] = (zr_6h * range_z_24h).astype(np.float32)
    feats["zr_12h_x_range_z_48h"] = (zr_12h * range_z_48h).astype(np.float32)

    feats["zr_3h"] = zr_3h.astype(np.float32)
    feats["zr_6h"] = zr_6h.astype(np.float32)
    feats["zr_12h"] = zr_12h.astype(np.float32)

    _zr_1h_sign = np.sign(zr_1h.fillna(0.0)).astype(np.float32)
    _zr_3h_sign = np.sign(zr_3h.fillna(0.0)).astype(np.float32)
    _zr_6h_sign = np.sign(zr_6h.fillna(0.0)).astype(np.float32)
    _zr_12h_sign = np.sign(zr_12h.fillna(0.0)).astype(np.float32)
    _zr_24h_sign = np.sign(zr_24h.fillna(0.0)).astype(np.float32)
    feats["trend_alignment_1_3_6"] = (_zr_1h_sign + _zr_3h_sign + _zr_6h_sign).astype(
        np.float32
    )
    feats["trend_alignment_3_6_12"] = (_zr_3h_sign + _zr_6h_sign + _zr_12h_sign).astype(
        np.float32
    )
    feats["trend_alignment_6_12_24"] = (
        _zr_6h_sign + _zr_12h_sign + _zr_24h_sign
    ).astype(np.float32)

    feats["coherence_24"] = (
        dir_s
        * (feats["ret6h"] + feats["ret12h"] + feats["ret24h"])
        / (feats["rv_24h"] + 1e-12)
    ).astype(np.float32)

    turb = rv_ratio  # Already broadcasted

    mkt_ret6h_raw = mkt_gates["mkt_ret6h"].reindex(c.index).astype(np.float32)

    # Multiply DataFrame `dir_s` with Series `mkt_ret6h_raw` over axis 0
    tape_align = dir_s.multiply(mkt_ret6h_raw, axis=0)
    # turb is `rv_ratio` which is a Series
    feats["tf_tape"] = (tape_align.clip(lower=0).div(1.0 + turb, axis=0)).astype(
        np.float32
    )
    feats["mr_tape"] = ((-tape_align).clip(lower=0).div(1.0 + turb, axis=0)).astype(
        np.float32
    )

    feats["tf_minus_mr"] = (feats["tf_tape"] - feats["mr_tape"]).astype(np.float32)
    feats["body_ratio"] = feats["efficiency"]

    # Define vars explicitly used in gates and other features
    ft2_pos = feats["ft_2"].clip(lower=0)
    ft4_pos = feats["ft_4"].clip(lower=0)
    clv4_pos = feats["clv_mean_4"].clip(lower=0)
    pb2_mag = feats["pullback_2"].abs().clip(0, 1)
    pb2_inv = 1.0 - pb2_mag
    pb4_mag = feats["pullback_4"].abs().clip(0, 1)
    pb4_inv = 1.0 - pb4_mag

    fail_sum = feats["failure_2"] + feats["failure_4"]
    clv_inv = 1.0 - feats["clv_mean_4"]
    pb_avg_abs = (feats["pullback_2"].abs() + feats["pullback_4"].abs()) / 2.0
    ret_rat = feats["ret4h"].abs() / (feats["ret1h"].abs() + 1e-12)

    # 3/ TF vs MR
    feats["accept_score"] = (ft2_pos * clv4_pos * pb2_inv).astype(np.float32)
    feats["retest_accept_score"] = (ft4_pos * clv4_pos * pb4_inv).astype(np.float32)

    feats["tf_qual_score"] = (feats["accept_score"] * feats["tf_tape"]).astype(
        np.float32
    )

    feats["mr_qual_score"] = (feats["reject_score"] * feats["mr_tape"]).astype(
        np.float32
    )
    feats["retrace_12"] = (-feats["pullback_12"]).astype(np.float32)

    # --- Gate Generation & Selection (Updated 2026-02-10) ---
    if cfg.get("enable_gated_features", False):
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
            "accept_score": (feats["accept_score"], "s"),
            "reject_score": (feats["reject_score"], "reject"),
            "retest_accept_score": (feats["retest_accept_score"], "retest_accept"),
            "tf_qual_score": (feats["tf_qual_score"], "tf_qual"),
            "mr_qual_score": (feats["mr_qual_score"], "mr_qual"),
            "vol_z": (feats["vol_z"], "vol_z"),
            # Liquidity Score: Higher is better (more liquid). Amihud is Illiq (lower is better).
            "liquidity_score": (-feats["amihud_z"], "liquidity"),
        }

        tprint(
            f"Generating Gated Features for windows {gate_windows} with selection..."
        )

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
                # 1. Generate ALL candidates for this family (mean, std, z, pct, bin3, gt25..gt75)
                # Returns dict: feature_name -> Panel DataFrame
                family_features = add_gate_features_panel(
                    source_panel,
                    prefix=prefix,
                    n=w,
                    add_strict=True,
                    percentile_mode=percentile_mode,
                )

                # 2. Extract BASE features (Always keep mean, std, z, pct, bin3)
                base_suffixes = ["mean", "std", "z", "pct", "bin3"]
                for suffix in base_suffixes:
                    feat_name = f"{prefix}_{suffix}_{w}"
                    if feat_name in family_features:
                        feats[feat_name] = family_features[feat_name]

                # 3. SELECT best threshold features (from gt25, gt50, ..., gt75)
                # Construct mini-table for selection function
                # Only include the 'gt' threshold candidates
                candidates_table = {
                    k: v for k, v in family_features.items() if "_gt" in k
                }

                # If no candidates produced, skip selection
                if not candidates_table:
                    continue

                # Run selection: Selects globally best thresholds based on prevalence/skill
                selected_names = select_gated_features(
                    gate_feature_table=candidates_table,
                    families=[(prefix, w)],
                    target=target_proxy,
                    time_blocks=time_blocks,
                    train_mask=train_mask_proxy,
                )

                # 4. Store SELECTED features
                for name in selected_names:
                    if name in candidates_table:
                        feats[name] = candidates_table[name]

                # Explicitly clear intermediate dict to free memory
                del family_features
                del candidates_table
                # import gc; gc.collect() # Optional frequent GC
    else:
        # Gated features disabled - provide zero fallbacks
        tprint("Gated features disabled - providing zero fallbacks")
        gate_window = int(cfg.get("accept_gate_window", 24))
        # Zero fallbacks for core gated features
        zero_panel = pd.DataFrame(0, index=c.index, columns=c.columns, dtype=np.float32)
        feats["accept_gt66"] = zero_panel.copy()
        feats["retest_accept"] = zero_panel.copy()
        feats["reject_like"] = zero_panel.copy()
        feats["tf_qual"] = zero_panel.copy()
        feats["mr_qual"] = zero_panel.copy()
        s_pct = zero_panel.copy()
        s_bin3 = zero_panel.copy()
        reject_like = zero_panel.copy()

    # Re-bind standardized names for downstream dependencies
    # These rely on the standard `gate_window` (e.g. 64) features being present
    # Warning: If `select_gated_features` didn't select gt66/gt75, these might fall back or error?
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

    # Dynamic selection might explicitly select gt66/gt75 or might select gt50/gt75.
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

    # reject_like: reject gate percentile (MR counterpart to the trend gate score)
    reject_like = get_feat(f"reject_pct_{gate_window}")

    # Map strict gates if they exist
    if f"s_gt66_{gate_window}" in feats:
        feats["accept_gt66"] = feats[f"s_gt66_{gate_window}"]
        feats["retest_accept"] = feats[f"s_gt66_{gate_window}"]  # Legacy alias
    else:
        # Fallback to whatever was selected as "broad" or "rare"?
        pass

    feats["tf_qual"] = (s_pct * feats["tf_tape"]).astype(np.float32)
    feats["mr_qual"] = (reject_like * feats["mr_tape"]).astype(np.float32)

    # Gate interactions with directional 2h path-risk block
    dir_edge = feats.get(
        "dir_path_edge_2h",
        pd.DataFrame(0, index=c.index, columns=c.columns, dtype=np.float32),
    )
    # 4/ Meta
    feats["rv_ratio_6_24"] = (feats["rv_6h"] / (feats["rv_24h"] + 1e-12)).astype(
        np.float32
    )

    # Define gates helpers for Meta

    feats["G_EXH_EFFORT"] = (
        (feats["evr_6"] * (feats["vol_z24"] + 1.0) / (feats["progress"] + 1e-12))
        .fillna(0.0)
        .astype(np.float32)
    )
    feats["G_EXH_GIVEBACK"] = (
        (feats["giveback"] * (1.0 + feats["donch_dist_12"]))
        .fillna(0.0)
        .astype(np.float32)
    )
    feats["G_EXH_TAIL_FAIL"] = (
        (feats["tail_against"] * (feats["ft_2"] - feats["ft_4"]).clip(lower=0))
        .fillna(0.0)
        .astype(np.float32)
    )

    feats["G_MR_SPIKE"] = (
        (feats["speed"] * feats["excess_6h"] * clv_inv).fillna(0.0).astype(np.float32)
    )
    feats["G_TF_TREND"] = (
        (feats["speed"] * feats["coherence_24"] * clv4_pos)
        .fillna(0.0)
        .astype(np.float32)
    )

    # Meta Features using Gates
    ambig_term = 1.0 - np.maximum(s_pct, reject_like)
    feats["ambig"] = (ambig_term * feats["rv_ratio_6_24"]).astype(np.float32)

    feats["stage_tf"] = (s_pct * feats["coherence_24"]).astype(np.float32)
    feats["stage_blowoff"] = (
        feats["blowoff_risk"] + feats["effort_gate"] + feats["stall_ext"]
    ).astype(np.float32)
    feats["stage_mr"] = (reject_like * (1.0 + feats["overext"])).astype(np.float32)
    feats["exh_qual"] = (
        feats["effort_gate"]
        + feats["stall_ext"]
        + feats["tail_fail"]
        + feats["overext_weak"]
    ).astype(np.float32)

    feats["thrust_decay_4"] = (
        feats["ret1h"].abs() / (feats["ret4h"].abs() + 1e-12)
    ).astype(np.float32)
    feats["decel_4"] = (feats["momentum_accel"].abs() / rv6).astype(np.float32)
    feats["ft_drop"] = (feats["ft_2"] - feats["ft_4"]).astype(np.float32)

    feats["thrust_decay_8"] = (
        feats["ret1h"].abs() / (feats["ret8h"].abs() + 1e-12)
    ).astype(np.float32)
    feats["decel_8"] = (feats["momentum_accel"].abs() / rv12).astype(np.float32)
    feats["ft_drop_8"] = (feats["ft_4"] - feats["ft_8"]).astype(np.float32)
    feats["ext_excess"] = (feats["donch_dist_12"] * feats["excess_6h"]).astype(
        np.float32
    )
    feats["ext_atrExp"] = (
        feats["donch_dist_12"] * np.log(feats["atr_expansion"] + 1e-12)
    ).astype(np.float32)
    feats["comp_to_exp"] = (
        (1.0 / (feats["vol_compression"] + 1e-12)) * feats["atr_expansion"]
    ).astype(np.float32)
    feats["evr6_x_volz"] = (feats["evr_6"] * (feats["vol_z24"] + 1.0)).astype(
        np.float32
    )
    feats["stall_x_flow"] = (feats["delta_stall_6"] * feats["flow_persistence"]).astype(
        np.float32
    )
    feats["prog_def"] = (feats["excess_6h"] / (feats["progress"] + 1e-12)).astype(
        np.float32
    )
    feats["clv_collapse"] = (feats["clv_mean_2"] - feats["clv_mean_4"]).astype(
        np.float32
    )
    feats["clv_pullback"] = (
        (1.0 - feats["clv_mean_4"]) * feats["pullback_4"].abs()
    ).astype(np.float32)
    feats["coh"] = (dir_s * (feats["ret1h"] + feats["ret2h"] + feats["ret4h"])) / rv6
    feats["align"] = (dir_s * np.sign(feats["slope"])).astype(np.float32)
    feats["retest_quality"] = (
        (1.0 - feats["pullback_2"].abs()) * feats["clv_mean_2"]
    ).astype(np.float32)
    feats["pb_accel"] = ((feats["pullback_2"] - feats["pullback_4"]) / atr).astype(
        np.float32
    )
    feats["excess_coh"] = (feats["excess_6h"] * feats["coh"]).astype(np.float32)
    feats["asym_ft"] = (feats["ft_2"] * feats["asym_ratio"] * dir_s).astype(np.float32)
    feats["dist_stack"] = (
        feats["dist_ema_fast"] + feats["dist_vwap_norm"] + feats["trend_pct"]
    ).astype(np.float32)
    feats["tf_bias"] = (feats["coh"] * (1.0 / (1.0 + feats["donch_dist_12"]))).astype(
        np.float32
    )
    feats["shock_rel"] = feats["excess_6h"]
    feats["resid_strength"] = feats["excess_6h"]
    feats["evr_slope"] = (feats["evr_3"] - feats["evr_6"]).astype(np.float32)

    # Base components for interactions
    ema_6 = ema(c, 6)
    ema_24 = ema(c, 24)
    feats["trend_t"] = ema_6.diff(1).astype(np.float32)

    # Volatility Interaction Context (New)
    feats["dist_ext_x_vol"] = (
        (feats["donch_dist_12"] * feats["vol_z"]).fillna(0.0).astype(np.float32)
    )
    feats["regime_x_vol"] = (
        (feats["rv_ratio_6_24"] * feats["vol_z"]).fillna(0.0).astype(np.float32)
    )
    feats["rsi_x_vol"] = (
        ((feats["rsi"] - 50.0) * feats["vol_z"]).fillna(0.0).astype(np.float32)
    )
    feats["vol_z_x_trend_t"] = (
        (feats["vol_z"] * feats["trend_t"]).fillna(0.0).astype(np.float32)
    )

    feats["stall_ext_corr"] = (feats["delta_stall_6"] * feats["donch_dist_12"]).astype(
        np.float32
    )

    feats["G_META_EXH"] = (
        feats["overext"]
        + feats["G_EXH_EFFORT"]
        + feats["stall_ext"]
        + feats["G_EXH_GIVEBACK"]
    ).astype(np.float32)
    ret_w = feats["ret10h"]
    local_low = ff.numba_rolling_min(l, 10)
    local_high = ff.numba_rolling_max(h, 10)
    draw_num = np.where(
        (ret_w > 0).to_numpy(), (c - local_low).to_numpy(), (c - local_high).to_numpy()
    )
    # Use safe division with proper handling of non-finite values
    c_safe = np.where(np.isfinite(c) & (np.abs(c) > 1e-12), c, 1.0)
    draw_sym = np.where(
        np.isfinite(draw_num) & np.isfinite(c) & (np.abs(c) > 1e-12),
        np.sign(ret_w) * draw_num / c_safe,
        0.0,
    )
    feats["draw_sym_10h"] = draw_sym.astype(np.float32)
    feats["draw_extreme_10h"] = np.abs(draw_sym).astype(np.float32)

    hi_24_prev = ff.numba_rolling_max(h.shift(1), 24)
    lo_24_prev = ff.numba_rolling_min(l.shift(1), 24)
    up_break = c - hi_24_prev
    dn_break = c - lo_24_prev
    choose_up = np.abs(up_break) >= np.abs(dn_break)
    # Use safe division with proper handling of non-finite values
    breakout_raw = np.where(choose_up, up_break, dn_break).astype(np.float32)
    feats["breakout_24h"] = np.where(
        np.isfinite(breakout_raw) & np.isfinite(c) & (np.abs(c) > 1e-12),
        breakout_raw / (c + 1e-12),
        0.0,
    ).astype(np.float32)

    abs_net_score = s_pct + reject_like
    feats["meta_abs_net_x_breakout"] = (
        abs_net_score * np.abs(feats["breakout_24h"])
    ).astype(np.float32)
    feats["meta_abs_net_x_drawext"] = (
        abs_net_score * np.abs(feats["draw_extreme_10h"])
    ).astype(np.float32)
    feats["meta_abs_net_x_vov_ratio"] = (
        abs_net_score * (feats["vov_ratio"] - 1.0).clip(lower=0)
    ).astype(np.float32)
    # Safe meta_alignment computation
    accept_diff = s_pct - reject_like
    ret5h_safe = np.where(np.isfinite(feats["ret5h"]), feats["ret5h"], 0.0)
    feats["meta_alignment"] = (np.sign(accept_diff) * np.sign(ret5h_safe)).astype(
        np.float32
    )
    feats["meta_signal_x_accel"] = ((s_pct - reject_like) * feats["accel_5h"]).astype(
        np.float32
    )

    # Regime interactions using base-model agreement-weighted success signal.
    base_agreement = (1.0 - np.abs(s_pct - reject_like)).clip(0.0, 1.0)
    p_success_df = (((s_pct + reject_like) * 0.5) * base_agreement).astype(np.float32)
    vol_high = feats.get(
        "vol_high",
        pd.DataFrame(0.0, index=c.index, columns=c.columns, dtype=np.float32),
    )
    cusum_high = feats.get(
        "cusum_high",
        pd.DataFrame(0.0, index=c.index, columns=c.columns, dtype=np.float32),
    )
    liq_low = feats.get(
        "liq_low", pd.DataFrame(0.0, index=c.index, columns=c.columns, dtype=np.float32)
    )
    interaction_dict = add_interactions(p_success_df, vol_high, cusum_high, liq_low)
    for ik, iv in interaction_dict.items():
        feats[ik] = iv.astype(np.float32)

    # Robust Score Calculation with clipping to prevent Inf/Overflow
    # We clip components to avoid exploding values when denominators are near zero
    feats["spike_score"] = (
        (feats["speed"].clip(0, 100) * feats["excess_6h"].clip(0, 100))
        .fillna(0.0)
        .astype(np.float32)
    )
    feats["grind_score"] = (
        (ret_rat.clip(0, 100) * feats["clv_mean_4"]).fillna(0.0).astype(np.float32)
    )
    coh_norm = feats["coh"].clip(0, 1).fillna(0.0)
    feats["chop_score"] = (
        (feats["rv_ratio_6_24"].clip(0, 100) * (1.0 - coh_norm))
        .fillna(0.0)
        .astype(np.float32)
    )

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
    feats["mtf_divergence"] = (sign_2h * sign_24h * -1.0).astype(
        np.float32
    )  # +1 = diverging
    #    Magnitude-weighted divergence
    feats["mtf_div_mag"] = (
        ((feats["ret2h"] - feats["ret24h"] / 12.0) / (feats["rv_6h"] + 1e-12))
        .clip(-10, 10)
        .astype(np.float32)
    )

    # 2. Mean-reversion speed proxy: rolling autocorrelation of returns
    #    Negative autocorr = fast mean-reversion, positive = trending
    feats["autocorr_6h"] = (
        ff.numba_rolling_corr(feats["ret1h"], feats["ret1h"].shift(1), 6)
        .fillna(0.0)
        .astype(np.float32)
    )
    feats["autocorr_24h"] = (
        ff.numba_rolling_corr(feats["ret1h"], feats["ret1h"].shift(1), 24)
        .fillna(0.0)
        .astype(np.float32)
    )

    # 3. Price path entropy proxy: ratio of actual path length to displacement
    #    High = choppy/random, Low = directional/clean
    abs_ret_sum_12 = ff.numba_rolling_sum(feats["ret1h"].abs(), 12)
    displacement_12 = feats["ret12h"].abs()
    feats["path_efficiency_12"] = (
        (displacement_12 / (abs_ret_sum_12 + 1e-12)).clip(0, 1).astype(np.float32)
    )
    abs_ret_sum_24 = ff.numba_rolling_sum(feats["ret1h"].abs(), 24)
    displacement_24 = feats["ret24h"].abs()
    feats["path_efficiency_24"] = (
        (displacement_24 / (abs_ret_sum_24 + 1e-12)).clip(0, 1).astype(np.float32)
    )

    # 6. Hurst exponent proxy: R/S ratio over rolling window
    #    H > 0.5 = trending, H < 0.5 = mean-reverting
    range_24 = _roll_max("c", c, 24) - _roll_min("c", c, 24)
    std_24 = _roll_std("ret1h", feats["ret1h"], 24)
    feats["hurst_proxy_24"] = (
        (
            np.log(range_24 / (std_24 * np.float32(np.sqrt(24)) + 1e-12) + 1e-12)
            / np.log(24)
        )
        .clip(0, 1)
        .fillna(0.5)
        .astype(np.float32)
    )

    # 7. Volume concentration: rolling Gini-like measure (max_vol / sum_vol over 12h)
    #    High = volume clustered in few bars, Low = evenly distributed
    v_max_12 = ff.numba_rolling_max(v, 12)
    v_sum_12 = ff.numba_rolling_sum(v, 12)
    feats["vol_concentration_12"] = (v_max_12 / (v_sum_12 + 1e-12)).astype(np.float32)

    # 4. Signed volume divergence: volume trend vs price trend disagreement
    vol_trend = ff.numba_rolling_sum(v, 6) - _roll_sum("v", v, 24) / 4.0
    price_trend = np.where(np.isfinite(feats["ret6h"]), feats["ret6h"], 0.0)
    feats["vol_price_diverge"] = (
        np.sign(vol_trend) * np.sign(price_trend) * -1.0
    ).astype(np.float32)

    # 5. Alpha asymmetry-volatility features (MR/TF, long/short)
    neg_ret = feats["ret1h"].clip(upper=0)
    pos_ret = feats["ret1h"].clip(lower=0)
    neg_sq = neg_ret * neg_ret
    pos_sq = pos_ret * pos_ret

    # Downside / Upside semivariance
    feats["downside_semivariance_24"] = ff.apply_to_frame(
        neg_sq, ff._numba_rolling_mean_nan_safe, 24
    ).astype(np.float32)
    feats["upside_semivariance_8"] = ff.apply_to_frame(
        pos_sq, ff._numba_rolling_mean_nan_safe, 8
    ).astype(np.float32)
    feats["upside_semivariance_24"] = ff.apply_to_frame(
        pos_sq, ff._numba_rolling_mean_nan_safe, 24
    ).astype(np.float32)

    # Downside / Upside volatility ratio (std ratio, not variance ratio)
    up_vol_8 = np.sqrt(feats["upside_semivariance_8"].clip(lower=0))
    down_vol_24 = np.sqrt(feats["downside_semivariance_24"].clip(lower=0))
    up_vol_24 = np.sqrt(feats["upside_semivariance_24"].clip(lower=0))
    feats["down_up_vol_ratio_24"] = (down_vol_24 / (up_vol_24 + 1e-12)).astype(
        np.float32
    )

    # Volatility shock asymmetry
    feats["vol_shock_asym_8_24"] = (feats["rv_8h"] - feats["rv_24h"]).astype(np.float32)
    feats["vol_shock_asym_4_12"] = (feats["rv_4h"] - feats["rv_12h"]).astype(np.float32)
    # Backward-compatible alias for requested notation "σ4 - σ212" (interpreted as 4 vs 12)
    feats["vol_shock_asym_4_212"] = feats["vol_shock_asym_4_12"].astype(np.float32)

    # 6. Alpha entropy features (MR/TF, long/short)
    # Shannon entropy of returns
    feats["shannon_entropy_ret_8"] = _rolling_shannon_entropy_df(
        feats["ret1h"], window=8, bins=8
    )
    feats["shannon_entropy_ret_16"] = _rolling_shannon_entropy_df(
        feats["ret1h"], window=16, bins=12
    )

    # Permutation entropy of returns
    feats["perm_entropy_ret_12"] = _rolling_permutation_entropy_df(
        feats["ret1h"], window=12, order=3, delay=1
    )
    feats["perm_entropy_ret_24"] = _rolling_permutation_entropy_df(
        feats["ret1h"], window=24, order=3, delay=1
    )

    # Spectral entropy of returns
    feats["spectral_entropy_ret_24"] = _rolling_spectral_entropy_df(
        feats["ret1h"], window=24
    )
    feats["spectral_entropy_ret_48"] = _rolling_spectral_entropy_df(
        feats["ret1h"], window=48
    )

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

    for feat_name in [
        "rsi",
        "dist_ema_fast",
        "dist_vwap_norm",
        "flow_persistence",
        "excess_6h",
        "vol_z",
        "atr_expansion",
        "coherence_24",
    ]:
        if feat_name in feats:
            raw = feats[feat_name]
            roll_mu = ff.numba_rolling_mean(raw, RESID_WINDOW)
            roll_sd = ff.numba_rolling_std(raw, RESID_WINDOW)
            feats[f"{feat_name}_z"] = (
                ((raw - roll_mu) / (roll_sd + 1e-12))
                .clip(-5, 5)
                .fillna(0.0)
                .astype(np.float32)
            )

    # (b) Rolling edge residual: how much is the model's current signal
    #     deviating from its recent realised performance?
    #     Proxy: z-score of composite scores (trend gate, reject, overext)
    for comp_name in ["overext", "blowoff_risk", "exh_qual"]:
        if comp_name in feats:
            raw = feats[comp_name]
            roll_mu = ff.numba_rolling_mean(raw, RESID_WINDOW)
            roll_sd = ff.numba_rolling_std(raw, RESID_WINDOW)
            feats[f"{comp_name}_surprise"] = (
                ((raw - roll_mu) / (roll_sd + 1e-12))
                .clip(-5, 5)
                .fillna(0.0)
                .astype(np.float32)
            )

    # (c) Residual distance from value vs market trend
    #     dist_resid = dist_to_vwap - k * market_trend_strength
    #     Stops MR entries that are "cheap" only because market is trending hard
    mkt_trend_s = mkt_gates["mkt_trend"].reindex(c.index).astype(np.float32)
    mkt_rv_s = mkt_gates["mkt_rv"].reindex(c.index).astype(np.float32)
    # Normalised market trend strength (in vol units) as a Series
    mkt_trend_z = mkt_trend_s / (mkt_rv_s * np.float32(np.sqrt(24)) + 1e-12)
    mkt_trend_z_safe = np.where(np.isfinite(mkt_trend_z), mkt_trend_z, 0.0)

    # Broadcast subtraction by aligning Series across rows using .sub(..., axis=0)
    dist_vwap_norm_values = feats["dist_vwap_norm"].to_numpy()
    dist_ema_fast_values = feats["dist_ema_fast"].to_numpy()
    trend_pct_values = feats["trend_pct"].to_numpy()
    mkt_trend_z_values = mkt_trend_z_safe[:, None]

    feats["dist_vwap_resid"] = pd.DataFrame(
        np.where(
            np.isfinite(dist_vwap_norm_values) & np.isfinite(mkt_trend_z_values),
            dist_vwap_norm_values - 0.5 * mkt_trend_z_values,
            0.0,
        ),
        index=c.index,
        columns=c.columns,
    ).astype(np.float32)

    feats["dist_ema_fast_resid"] = pd.DataFrame(
        np.where(
            np.isfinite(dist_ema_fast_values) & np.isfinite(mkt_trend_z_values),
            dist_ema_fast_values - 0.5 * mkt_trend_z_values,
            0.0,
        ),
        index=c.index,
        columns=c.columns,
    ).astype(np.float32)

    feats["trend_pct_resid"] = pd.DataFrame(
        np.where(
            np.isfinite(trend_pct_values) & np.isfinite(mkt_trend_z_values),
            trend_pct_values - 0.5 * mkt_trend_z_values,
            0.0,
        ),
        index=c.index,
        columns=c.columns,
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
    feats["trend_age_hours"] = (
        pd.DataFrame(
            _rank, index=trend_age_cumsum.index, columns=trend_age_cumsum.columns
        )
        .astype(np.float32)
        .fillna(1)
    )

    # higher_highs_count_48h: Count of higher highs in last 48 hours (trend quality)
    # A higher high is when current high > previous high
    higher_high = (h > h.shift(1)).astype(np.float32)
    feats["higher_highs_count_48h"] = ff.numba_rolling_sum(higher_high, 48).astype(
        np.float32
    )

    # trend_retest_success_rate: How often do retests hold?
    # Proxy: when price pulls back to EMA, does it bounce?
    near_ema = (feats["dist_ema_fast"].abs() < 0.5).astype(
        np.float32
    )  # Within 0.5 ATR of EMA
    ret_after_near = feats["ret4h"].shift(-4).fillna(0.0)  # Return 4h later
    retest_success = (near_ema * (ret_after_near * trend_sign > 0)).astype(np.float32)
    retest_attempts = near_ema.rolling(48, min_periods=1).sum()
    retest_successes = retest_success.rolling(48, min_periods=1).sum()
    feats["trend_retest_success_rate"] = (
        (retest_successes / (retest_attempts + 1e-12)).clip(0, 1).astype(np.float32)
    )

    # trend_overextension_z: Z-scored distance from EMA (overextension detection)
    dist_ema_rolling_mean = ff.numba_rolling_mean(feats["dist_ema_fast"], 48)
    dist_ema_rolling_std = ff.numba_rolling_std(feats["dist_ema_fast"], 48)
    feats["trend_overextension_z"] = (
        (
            (feats["dist_ema_fast"] - dist_ema_rolling_mean)
            / (dist_ema_rolling_std + 1e-12)
        )
        .clip(-5, 5)
        .astype(np.float32)
    )

    # volume_trend_alignment: Is volume rising with the trend?
    # Correlation between volume and price direction over 24h
    vol_change = v.diff(1).fillna(0.0).astype(np.float32)
    price_dir = np.sign(feats["ret1h"]).astype(np.float32)
    feats["volume_trend_alignment"] = (
        ff.numba_rolling_corr(vol_change, price_dir, 24)
        .fillna(0.0)
        .clip(-1, 1)
        .astype(np.float32)
    )

    # trend_regime_stability: How stable is the current trend regime?
    # Low value = regime transition risk, high value = stable trend
    trend_sign_flips = (
        (trend_sign != trend_sign.shift(1)).rolling(48, min_periods=1).sum()
    )
    feats["trend_regime_stability"] = (1.0 / (1.0 + trend_sign_flips)).astype(
        np.float32
    )

    # --- MR Features: Dip Quality & Support Context ---

    # trend_strength_vs_reversion: Ratio of trend force to mean-reversion force
    # High = trending (avoid MR), Low = ranging (good for MR)
    trend_force = feats["ret24h"].abs()
    mr_force = feats["autocorr_6h"].abs().clip(0, 1)  # Negative autocorr = MR force
    feats["trend_strength_vs_reversion"] = (trend_force / (mr_force + 1e-12)).astype(
        np.float32
    )

    # support_quality_score: How strong is nearby support?
    # Based on: volume at nearby price levels, number of touches, recency
    # Proxy: count how often price bounced from current level in last 120h
    lo_24 = ff.numba_rolling_min(l, 24)
    dist_to_low = ((c - lo_24) / (atr_base + 1e-12)).astype(np.float32)
    # Support quality is high when: close to recent low, high volume there
    near_support = (dist_to_low.abs() < 1.0).astype(
        np.float32
    )  # Within 1 ATR of 24h low
    vol_at_support = (near_support * v).astype(np.float32)
    vol_total = v.rolling(24, min_periods=1).sum()
    support_vol_ratio = vol_at_support.rolling(24, min_periods=1).sum() / (
        vol_total + np.float32(1e-12)
    )
    feats["support_quality_score"] = (near_support * support_vol_ratio).astype(
        np.float32
    )

    # dip_velocity: How fast did we dip? (Sharp dips = better MR)
    # Rate of change of distance from high
    hi_12 = ff.numba_rolling_max(h, 12)
    dist_from_high_12 = ((c - hi_12) / (atr_base + 1e-12)).astype(np.float32)
    feats["dip_velocity"] = (dist_from_high_12.diff(1).fillna(0.0) * -1).astype(
        np.float32
    )  # Positive = dipping fast

    # dip_volume_profile: Volume characteristics during the dip
    # High volume on dip = capitulation (good MR), low volume = orderly decline (bad MR)
    is_dipping = (feats["ret4h"] < 0).astype(np.float32)
    vol_on_dip = (is_dipping * v).astype(np.float32)
    vol_avg = v.rolling(24, min_periods=1).mean()
    feats["dip_volume_profile"] = (
        ((vol_on_dip / (vol_avg + 1e-12)) * is_dipping).fillna(0.0).astype(np.float32)
    )

    # reversion_target_distance: Distance to mean (upside potential for MR)
    # Using VWAP as mean proxy
    vwap_proxy = (c * v).rolling(24, min_periods=1).sum() / (
        v.rolling(24, min_periods=1).sum() + np.float32(1e-12)
    )
    feats["reversion_target_distance"] = ((vwap_proxy - c) / (atr_base + 1e-12)).astype(
        np.float32
    )

    # ---------------------------------------------------------------------
    # Regime-transition / complexity features (2h/4h/8h trade-horizon focus)
    # ---------------------------------------------------------------------
    # Volatility regime in rolling z-space
    feats["vol_regime_z"] = (
        zscore_rolling(feats["rv_24h"], 48).fillna(0.0).astype(np.float32)
    )
    feats["is_high_vol_regime"] = (feats["vol_regime_z"] > 0.75).astype(np.float32)
    feats["is_low_vol_regime"] = (feats["vol_regime_z"] < -0.75).astype(np.float32)

    # Trend regime score from 24h return in local-vol units
    feats["trend_regime"] = (
        (feats["ret24h"] / (feats["rv_24h"] * np.sqrt(24.0) + 1e-12))
        .clip(-3, 3)
        .astype(np.float32)
    )
    feats["is_trending"] = (feats["trend_regime"].abs() >= 0.75).astype(np.float32)
    feats["is_ranging"] = (1.0 - feats["is_trending"]).astype(np.float32)

    # Liquidity regime: high value means better-than-usual liquidity
    feats["liq_regime"] = (-feats["amihud_z"]).clip(-5, 5).astype(np.float32)

    # Regime switching intensity (12h) and stability (24h)
    trend_state = np.sign(feats["trend_regime"]).replace(0, np.nan).ffill().fillna(0.0)
    vol_state = np.sign(feats["vol_regime_z"]).replace(0, np.nan).ffill().fillna(0.0)
    trend_switch_evt = (trend_state != trend_state.shift(1)).astype(np.float32)
    vol_switch_evt = (vol_state != vol_state.shift(1)).astype(np.float32)
    feats["trend_regime_switch_12h"] = ff.numba_rolling_sum(
        trend_switch_evt, 12
    ).astype(np.float32)
    feats["vol_regime_switch_12h"] = ff.numba_rolling_sum(vol_switch_evt, 12).astype(
        np.float32
    )
    feats["regime_stability_24h"] = (
        1.0 / (1.0 + ff.numba_rolling_sum((trend_switch_evt + vol_switch_evt) > 0, 24))
    ).astype(np.float32)

    # Entropy of switching process (binary entropy of switch-rate over horizon)
    def _binary_entropy(p):
        p = p.clip(1e-6, 1 - 1e-6)
        return -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))

    sw12 = (
        ff.numba_rolling_sum((trend_switch_evt + vol_switch_evt) > 0, 12) / 12.0
    ).clip(0, 1)
    sw48 = (
        ff.numba_rolling_sum((trend_switch_evt + vol_switch_evt) > 0, 48) / 48.0
    ).clip(0, 1)
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
    feats["vol_regime_z_4d"] = (
        zscore_rolling(feats["rv_24h"], 96).fillna(0.0).astype(np.float32)
    )

    # 4-day trend strength (normalized by local volatility)
    feats["trend_strength_4d"] = (
        (
            ff.numba_rolling_mean(feats["ret24h"], 96)
            / (ff.numba_rolling_std(feats["ret24h"], 96) * np.sqrt(96.0) + 1e-12)
        )
        .clip(-3, 3)
        .astype(np.float32)
    )

    # 4-day regime stability (inverse of regime changes over 96 hours)
    trend_switch_4d = ff.numba_rolling_sum(trend_switch_evt, 96)
    vol_switch_4d = ff.numba_rolling_sum(vol_switch_evt, 96)
    feats["regime_stability_4d"] = (
        1.0 / (1.0 + trend_switch_4d + vol_switch_4d)
    ).astype(np.float32)

    # 4-day volatility persistence (autocorrelation of volatility)
    vol_persistence_4d = (
        ff.numba_rolling_corr(feats["rv_24h"], feats["rv_24h"].shift(96), 96)
        .fillna(0.0)
        .clip(-1, 1)
        .astype(np.float32)
    )
    feats["vol_persistence_4d"] = vol_persistence_4d

    # 4-day average trend regime duration (vectorized)
    # Average duration = window / (number of trend changes + 1)
    trend_changes_4d = ff.numba_rolling_sum(trend_sign_change, 96)
    feats["trend_regime_duration_4d"] = (96.0 / (trend_changes_4d + 1.0)).astype(
        np.float32
    )

    # Regime interaction terms requested in config
    feats["rsi_z_x_regime_vol"] = (
        feats.get("rsi_z", 0.0) * feats["vol_regime_z"]
    ).astype(np.float32)
    feats["vol_z_x_regime_trend"] = (feats["vol_z"] * feats["trend_regime"]).astype(
        np.float32
    )
    feats["mtf_divergence_x_regime_vol_12h"] = (
        feats["mtf_div_mag"] * ff.numba_rolling_mean(feats["vol_regime_z"], 12)
    ).astype(np.float32)
    feats["hurst_proxy_x_regime_trend_48h"] = (
        feats["hurst_proxy_24"] * ff.numba_rolling_mean(feats["trend_regime"], 48)
    ).astype(np.float32)
    feats["rsi_x_high_vol"] = (
        ((feats["rsi"] - 50.0) / 50.0) * feats["is_high_vol_regime"]
    ).astype(np.float32)
    feats["trend_x_trending"] = (feats["trend_regime"] * feats["is_trending"]).astype(
        np.float32
    )
    feats["vol_z_x_low_vol"] = (feats["vol_z"] * feats["is_low_vol_regime"]).astype(
        np.float32
    )

    # -----------------------------------------------------------------
    # Position-sizer interaction features
    # Normalisation: _signed_log1p then robust z-score (30-day window)
    # to tame heavy right-skew inherent to product terms.
    # -----------------------------------------------------------------
    _ps_interact_items: list[tuple[str, pd.DataFrame]] = []

    if _needs_feature(
        "atr_pct_x_amihud_z",
        "rvol_z_x_range_expansion",
        "close_loc_x_hurst",
        "dist_vwap_x_vov",
        "abs_ret6h_sigma_x_vov",
        "amihud_z_x_close_loc",
    ):
        _clb = feats.get(
            "close_location_in_bar",
            feats.get(
                "close_position_in_range",
                pd.DataFrame(0, index=c.index, columns=c.columns, dtype=np.float32),
            ),
        )
        _vov48 = feats.get(
            "volatility_of_volatility_48",
            pd.DataFrame(0, index=c.index, columns=c.columns, dtype=np.float32),
        )
        _az = feats.get(
            "amihud_z",
            pd.DataFrame(0, index=c.index, columns=c.columns, dtype=np.float32),
        )
        _ap = feats.get(
            "atr_pct",
            pd.DataFrame(1, index=c.index, columns=c.columns, dtype=np.float32),
        )
        _dvn = feats.get(
            "dist_vwap_norm",
            pd.DataFrame(0, index=c.index, columns=c.columns, dtype=np.float32),
        )
        _rz = feats.get(
            "rvol_z",
            pd.DataFrame(0, index=c.index, columns=c.columns, dtype=np.float32),
        )
        _rer = feats.get(
            "range_expansion_ratio",
            pd.DataFrame(1, index=c.index, columns=c.columns, dtype=np.float32),
        )
        _hp = feats.get(
            "hurst_proxy_24",
            pd.DataFrame(0.5, index=c.index, columns=c.columns, dtype=np.float32),
        )
        _r6 = feats.get(
            "ret6h", pd.DataFrame(0, index=c.index, columns=c.columns, dtype=np.float32)
        )
        _rv6 = feats.get(
            "rv_6h", pd.DataFrame(1, index=c.index, columns=c.columns, dtype=np.float32)
        )

        if _needs_feature("atr_pct_x_amihud_z"):
            _raw = (_ap * _az).astype(np.float32)
            _ps_interact_items.append(("atr_pct_x_amihud_z", _signed_log1p(_raw)))

        if _needs_feature("rvol_z_x_range_expansion"):
            _raw = (_rz * _rer).astype(np.float32)
            _ps_interact_items.append(("rvol_z_x_range_expansion", _signed_log1p(_raw)))

        if _needs_feature("close_loc_x_hurst"):
            _raw = (_clb * _hp).astype(np.float32)
            _ps_interact_items.append(("close_loc_x_hurst", _signed_log1p(_raw)))

        if _needs_feature("dist_vwap_x_vov"):
            _raw = (_dvn * _vov48).astype(np.float32)
            _ps_interact_items.append(("dist_vwap_x_vov", _signed_log1p(_raw)))

        if _needs_feature("abs_ret6h_sigma_x_vov"):
            _raw = ((np.abs(_r6) / (_rv6 + np.float32(1e-12))) * _vov48).astype(
                np.float32
            )
            _ps_interact_items.append(("abs_ret6h_sigma_x_vov", _signed_log1p(_raw)))

        if _needs_feature("amihud_z_x_close_loc"):
            _raw = (_az * _clb).astype(np.float32)
            _ps_interact_items.append(("amihud_z_x_close_loc", _signed_log1p(_raw)))

        if _ps_interact_items:
            _ps_interact_rz = _batch_roll_robust_zscore(_ps_interact_items, 24 * 30)
            for _name, _rz_df in _ps_interact_rz.items():
                feats[_name] = _rz_df.astype(np.float32)

    mkt_ret24h = mkt_gates["mkt_ret24h"].reindex(c.index).astype(np.float32)
    mkt_ret6h = mkt_gates["mkt_ret6h"].reindex(c.index).astype(np.float32)
    ret24h = feats["ret24h"].astype(np.float32)
    ret24h_mean = ff.numba_rolling_mean(ret24h, 24)
    mkt_ret24h_df = mkt_ret24h.to_frame("mkt_ret24h")
    mkt_ret24h_mean = ff.numba_rolling_mean(mkt_ret24h_df, 24)["mkt_ret24h"]
    mkt_ret24h_sq_mean = ff.numba_rolling_mean(
        (mkt_ret24h * mkt_ret24h).to_frame("mkt_ret24h"), 24
    )["mkt_ret24h"]
    cov_24h = ff.numba_rolling_mean(
        ret24h.multiply(mkt_ret24h, axis=0), 24
    ) - ret24h_mean.multiply(mkt_ret24h_mean, axis=0)
    beta_24h = cov_24h.divide(
        mkt_ret24h_sq_mean - mkt_ret24h_mean.pow(2) + 1e-12,
        axis=0,
    )
    feats["beta_24h"] = beta_24h.replace([np.inf, -np.inf], np.nan).astype(np.float32)
    feats["resid_ret_6h"] = (
        feats["ret6h"] - beta_24h.multiply(mkt_ret6h, axis=0)
    ).astype(np.float32)
    feats["xs_rank_vol_z"] = feats["vol_z"].rank(axis=1, pct=True).astype(np.float32)

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
    feats["volume_capitulation"] = (adverse_4h * feats["vol_z"].clip(lower=0)).astype(
        np.float32
    )

    # Trap strength: exhaustion + capitulation + failed continuation context.
    feats["trap_strength"] = (
        feats["volume_capitulation"]
        * (1.0 + feats["overext"].clip(lower=0))
        * (1.0 - s_pct)
    ).astype(np.float32)

    # Composite entry quality across 2h/4h/8h context.
    feats["entry_quality_composite"] = (
        0.40 * s_pct
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
    std_c_24 = _roll_std("c", c, 24)
    feats["trend_z_t"] = (trend_t / (std_c_24 + 1e-12)).astype(np.float32)

    # convexity_t
    convexity_t = trend_t.diff(1).astype(np.float32)
    feats["convexity_t"] = convexity_t

    # convexity_bis_t
    feats["convexity_bis_t"] = (ema_6 - ema_24).diff(1).astype(np.float32)

    # convexity_z_t
    convexity_z_t = zscore_rolling(convexity_t, 24).fillna(0.0).astype(np.float32)
    feats["convexity_z_t"] = convexity_z_t

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
    min_24 = _roll_min("c", c, 24)
    max_24 = _roll_max("c", c, 24)
    pct_pos = ((c - min_24) / (max_24 - min_24 + 1e-12)).clip(0, 1)

    # squeeze
    squeeze = feats["vol_compression"]

    # --- TF Meta Features ---
    feats["vw_breakout"] = (breakout_z * log_1_rvol).astype(np.float32)

    sigmoid_rvol = (1.0 / (1.0 + np.exp(-(v - ema_v_24)))).astype(np.float32)
    feats["breakout_soft"] = (breakout_z * sigmoid_rvol).astype(np.float32)

    feats["tail_score"] = (
        feats["trend_z_t"] * np.maximum(0, convexity_z_t) * np.maximum(0, breakout_z)
    ).astype(np.float32)

    # --- MR Meta Features ---
    sigmoid_neg_conv_z = (1.0 / (1.0 + np.exp(convexity_z_t))).astype(
        np.float32
    )  # sigmoid(-x)
    feats["mr_soft"] = (breakout_z.abs() * sigmoid_neg_conv_z).astype(np.float32)

    feats["mr_potential"] = (
        (c - ema_24).abs() / (feats["atr_pct_base"] * c + 1e-12)
    ).astype(np.float32)

    feats["mr_potential_exhaust"] = (
        feats["mr_potential"] * np.maximum(0, -convexity_z_t)
    ).astype(np.float32)

    feats["climax"] = (breakout_z.abs() * log_1_rvol).astype(np.float32)

    sigmoid_conv_z = (1.0 / (1.0 + np.exp(-convexity_z_t))).astype(np.float32)
    feats["vol_exhaust"] = (log_1_rvol * sigmoid_conv_z).astype(np.float32)

    feats["mr_climax"] = (breakout_z.abs() * log_1_rvol * sigmoid_neg_conv_z).astype(
        np.float32
    )

    imp_abs = impulse.abs()
    imp_abs_lag = imp_abs.shift(1).fillna(0.0)
    feats["shock_decay"] = (imp_abs_lag * np.maximum(0, imp_abs_lag - imp_abs)).astype(
        np.float32
    )

    feats["pct_extreme"] = (pct_pos - 0.5).abs().astype(np.float32)

    feats["mr_pct"] = (feats["pct_extreme"] * sigmoid_conv_z).astype(np.float32)

    tz_abs = feats["trend_z_t"].abs()
    feats["stall"] = np.maximum(0, tz_abs.shift(1).fillna(0.0) - tz_abs).astype(
        np.float32
    )

    feats["mr_failure"] = (squeeze * breakout_z.abs() * feats["stall"]).astype(
        np.float32
    )

    # --- Alpha Features ---
    feats["breakout_min"] = np.minimum(np.maximum(0, breakout_z), log_1_rvol).astype(
        np.float32
    )

    imp_lag = impulse.shift(1).fillna(0.0)
    feats["impulse_reversal"] = (
        np.maximum(0, -imp_lag) * np.maximum(0, impulse)
    ).astype(np.float32)

    feats["impulse_reversal_short"] = (
        np.maximum(0, imp_lag) * np.maximum(0, -impulse)
    ).astype(np.float32)

    feats["breakout_confirmed"] = (
        breakout_z * (rvol_ratio > 1.2).astype(np.float32)
    ).astype(np.float32)

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
        feats[f"dist_vwap_{n}_atr"] = (
            (c_log - vwap_n) / (feats["atr_ln"] + 1e-12)
        ).astype(np.float32)

        # Trapped Longs: Price < VWAP. Magnitude of trapped signal.
        # Positive value = Longs are trapped (Price below VWAP)
        feats[f"trapped_longs_{n}"] = (
            ((vwap_n - c_log) / (feats["atr_ln"] + 1e-12))
            .clip(lower=0)
            .astype(np.float32)
        )

    feats["clv_t"] = (((c_log - l) - (h - c_log)) / ((h - l) + 1e-9)).astype(np.float32)

    tr_15m = np.maximum(
        h - l, np.maximum((h - c_log.shift(1)).abs(), (l - c_log.shift(1)).abs())
    )
    body_ratio_15m = (c_log - o).abs() / ((h - l) + 1e-9)
    feats["body_ratio_15m"] = body_ratio_15m.astype(np.float32)

    upper_wick = (h - np.maximum(o, c_log)).clip(lower=0)
    lower_wick = (np.minimum(o, c_log) - l).clip(lower=0)
    feats["rejection_proxy"] = ((lower_wick - upper_wick) / ((h - l) + 1e-9)).astype(
        np.float32
    )

    sv = v * np.sign(c_log - c_log.shift(1))
    press_base = ((c_log - o) / ((h - l) + 1e-9)) * v

    for n in [12, 24]:
        atr_15m_n = ff.numba_ewma(tr_15m, 2.0 / (n + 1.0), False)
        feats[f"range_norm_{n}"] = (h_minus_l / (atr_15m_n + 1e-12)).astype(np.float32)

        sv_sum_n = ff.numba_rolling_sum(sv, n)
        v_sum_n = ff.numba_rolling_sum(v, n)
        feats[f"sv_imb_{n}"] = (sv_sum_n / (v_sum_n + 1e-12)).astype(np.float32)

        feats[f"press_{n}"] = ff.numba_rolling_mean(press_base, n).astype(np.float32)

        feats[f"impact_{n}"] = ff.numba_rolling_mean(
            c_log_diff1_abs / (v + 1e-9), n
        ).astype(np.float32)

        ts_mean_n = ff.numba_rolling_mean(c_log_diff1, n)
        ts_std_n = ff.numba_rolling_std(c_log_diff1, n)
        feats[f"ts_{n}"] = (ts_mean_n / (ts_std_n + 1e-12)).astype(np.float32)

        prog_n = (c_log - c_log.shift(n)).abs()
        feats[f"prog_eff_{n}"] = (prog_n / (v_sum_n + 1e-12)).astype(np.float32)

        feats[f"pers_{n}"] = ff.numba_rolling_mean(c_log_diff1_sign, n).astype(
            np.float32
        )

        hh_count_n = ff.numba_rolling_sum((h > h.shift(1)).astype(np.float32), n)
        feats[f"hh_count_{n}"] = hh_count_n.astype(np.float32)

        ll_count_n = ff.numba_rolling_sum((l < l.shift(1)).astype(np.float32), n)
        feats[f"ll_count_{n}"] = ll_count_n.astype(np.float32)

        feats[f"skew_{n}"] = ff.apply_to_frame(
            c_log_diff1, ff._numba_rolling_skew, n
        ).astype(np.float32)

        climax_range_med_n = ff.apply_to_frame(h_minus_l, ff._numba_rolling_median, n)
        feats[f"climax_range_{n}"] = (h_minus_l / (climax_range_med_n + 1e-12)).astype(
            np.float32
        )

        climax_vol_med_n = ff.apply_to_frame(v, ff._numba_rolling_median, n)
        feats[f"climax_vol_{n}"] = (v / (climax_vol_med_n + 1e-12)).astype(np.float32)

        vwap_z_n = ff.numba_rolling_vwap(c_log, v, n)

        diff_vwap = c_log - vwap_z_n
        std_vwap = ff.numba_rolling_std(diff_vwap, n)
        feats[f"z_vwap_{n}"] = (diff_vwap / (std_vwap + 1e-12)).astype(np.float32)

        feats[f"z_r_{n}"] = ((c_log_diff1 - ts_mean_n) / (ts_std_n + 1e-12)).astype(
            np.float32
        )

        c_log_mean_n = ff.numba_rolling_mean(c_log, n)
        c_log_std_n = ff.numba_rolling_std(c_log, n)
        feats[f"bb_pos_{n}"] = ((c_log - c_log_mean_n) / (c_log_std_n + 1e-12)).astype(
            np.float32
        )

    # 5. Volume Node Features (HVN/LVN)
    # Loop over columns, construct DF, call function, stack results
    need_vp_features = not requested_feature_set or any(
        str(k).startswith("vp_") for k in requested_feature_set
    )
    if need_vp_features:
        tprint("Computing HVN/LVN features...")
        try:
            from .volume_node_features import hvn_lvn_features_ohlcv

            # Get feature names from a sample run
            first_col = c_log.columns[0]
            df_first = pd.DataFrame(
                {
                    "open": o[first_col],
                    "high": h[first_col],
                    "low": l[first_col],
                    "close": c_log[first_col],  # Use c_log, not FFD c
                    "volume": v[first_col],
                }
            )
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
    del target_proxy
    # Free time_blocks and train_mask_proxy if they were defined (gated features enabled)
    try:
        del time_blocks, train_mask_proxy
    except NameError:
        pass
    # Note: o, h, l, c, v are deleted later after all features that use them are computed

    # --- explicit peer-context and ts-percentile features ---
    if not requested_feature_set or any(
        str(k).startswith("cs_rank_") or str(k).startswith("cs_rz_")
        for k in requested_feature_set
    ):
        cs_feats = add_cross_sectional_peer_context_features(feats, min_group_size=5)
        feats.update(cs_feats)

    if not requested_feature_set or any(
        str(k).startswith("ts_pct_") for k in requested_feature_set
    ):
        ts_feats = add_time_series_percentile_features(
            feats, lookback=720, min_history_fraction=0.25
        )
        feats.update(ts_feats)

    if requested_feature_set:
        feats = {k: v for k, v in feats.items() if k in requested_feature_set}
    tprint(
        f"Features: {len(feats)} features before CausalTransform. Applying transforms..."
    )
    # =========================================================================
    # PORTABILITY HARDENING (IN-PLACE)
    # =========================================================================

    # 1) Excursion / path-risk family

    # We use event-time RV (rv_24h) for scaling.
    scale_rv = feats.get("rv_24h")
    rz96_items: list[tuple[str, pd.DataFrame]] = []
    if scale_rv is not None:
        for f in [
            "mfe_2h",
            "mae_2h",
            "mfe_4h",
            "mae_4h",
            "mfe_8h",
            "mae_8h",
            "dir_path_long_2h",
            "dir_path_short_2h",
        ]:
            if f in feats:
                feats[f] = _safe_div(feats[f], scale_rv).astype(np.float32)

        for f in ["dir_path_risk_long_2h", "dir_path_risk_short_2h"]:
            if f in feats:
                tmp = _safe_div(feats[f], scale_rv)
                rz96_items.append((f, tmp))

        if "dir_path_edge_2h" in feats:
            # mfe_scaled / (mae_scaled + eps) -- assume we can just use raw mfe_2h/mae_2h since scales cancel
            if "mfe_2h" in feats and "mae_2h" in feats:
                feats["dir_path_edge_2h"] = _safe_div(
                    feats["mfe_2h"], feats["mae_2h"]
                ).astype(np.float32)

        if "dir_path_risk_skew_2h" in feats:
            feats["dir_path_risk_skew_2h"] = np.tanh(
                feats["dir_path_risk_skew_2h"]
            ).astype(np.float32)

        if "giveback" in feats:
            feats["giveback"] = (
                _safe_div(feats["giveback"], scale_rv).clip(0, 5.0).astype(np.float32)
            )

        if "tail_against" in feats:
            feats["tail_against"] = _safe_div(feats["tail_against"], scale_rv).astype(
                np.float32
            )

        if "dip_velocity" in feats:
            feats["dip_velocity"] = _safe_div(feats["dip_velocity"], scale_rv).astype(
                np.float32
            )

        if "reversion_target_distance" in feats:
            feats["reversion_target_distance"] = _safe_div(
                feats["reversion_target_distance"], scale_rv
            ).astype(np.float32)

    # 2) Medium / long horizon returns
    if scale_rv is not None:
        for f in ["ret2h", "ret4h", "ret6h", "ret8h"]:
            if f in feats:
                feats[f] = _safe_div(feats[f], scale_rv).astype(np.float32)

    scale_rv_long = feats.get("rv_120h")
    if scale_rv_long is not None:
        for f in ["ret48h", "ret72h"]:
            if f in feats:
                feats[f] = _safe_div(feats[f], scale_rv_long).astype(np.float32)
    rz480_items: list[tuple[str, pd.DataFrame]] = []
    if scale_rv_long is None:
        for f in ["ret48h", "ret72h"]:
            if f in feats:
                rz480_items.append((f, feats[f]))
    if "ret120h" in feats:
        rz480_items.append(("ret120h", feats["ret120h"]))
    if rz480_items:
        feats.update(_batch_roll_robust_zscore(rz480_items, 480))

    # 3) RV / ATR level features
    if scale_rv is not None:
        for f in ["rv_2h", "rv_4h", "rv_6h", "rv_8h"]:
            if f in feats:
                feats[f] = _safe_log_ratio(feats[f], scale_rv).astype(np.float32)

    scale_rv_48 = feats.get("rv_48h")
    if scale_rv_48 is not None and "rv_12h" in feats:
        feats["rv_12h"] = _safe_log_ratio(feats["rv_12h"], scale_rv_48).astype(
            np.float32
        )

    if scale_rv_long is not None and "rv_48h" in feats:
        feats["rv_48h"] = _safe_log_ratio(feats["rv_48h"], scale_rv_long).astype(
            np.float32
        )

    if "rv_120h" in feats:
        feats["rv_120h"] = _roll_rank_pct("rv_120h", feats["rv_120h"], 480)

    if "rv_24h" in feats:
        rz96_items.append(
            (
                "rv_24h",
                _frame_like(
                    feats["rv_24h"],
                    np.log1p(feats["rv_24h"]).to_numpy(dtype=np.float32, copy=False),
                ),
            )
        )
    if "atr_pct_change" in feats:
        rz96_items.append(("atr_pct_change", feats["atr_pct_change"]))

    if "atr_expansion" in feats:
        rz96_items.append(("atr_expansion", feats["atr_expansion"]))

    # 4) Heavy-tailed flow / liquidity / divergence
    # Need to robustify if still raw
    flow_rz = [
        "flow_persistence",
        "cumulative_delta_stall",
        "delta_stall_6",
        "vol_price_div",
        "vol_price_diverge",
        "return_per_volume",
        "volume_trend_48",
    ]
    for f in flow_rz:
        if f in feats:
            rz96_items.append((f, feats[f]))

    if "signed_vol" in feats:
        baseline_vol = ff.numba_rolling_mean(
            (
                feats["volume"].to_numpy()
                if "volume" in feats
                else np.abs(feats["signed_vol"].to_numpy())
            ),
            96,
        )
        feats["signed_vol"] = _safe_div(
            feats["signed_vol"],
            pd.DataFrame(
                baseline_vol,
                index=feats["signed_vol"].index,
                columns=feats["signed_vol"].columns,
            ),
        ).astype(np.float32)

    for f in ["up_vol", "dn_vol", "up_vol_6", "dn_vol_6"]:
        if f in feats:
            baseline = ff.numba_rolling_quantile(feats[f].to_numpy(), 96, 0.5)
            feats[f] = _safe_log_ratio(
                feats[f],
                pd.DataFrame(baseline, index=feats[f].index, columns=feats[f].columns),
            ).astype(np.float32)

    if "churn" in feats:
        baseline = ff.numba_rolling_quantile(feats["churn"].to_numpy(), 96, 0.5)
        feats["churn"] = _safe_log_ratio(
            feats["churn"],
            pd.DataFrame(
                baseline, index=feats["churn"].index, columns=feats["churn"].columns
            ),
        ).astype(np.float32)

    if "v_power" in feats:
        rz96_items.append(
            (
                "v_power",
                _frame_like(
                    feats["v_power"],
                    _signed_log1p(feats["v_power"]).to_numpy(
                        dtype=np.float32, copy=False
                    ),
                ),
            )
        )

    if "amihud_illiq" in feats:
        rz96_items.append(
            (
                "amihud_illiq",
                _frame_like(
                    feats["amihud_illiq"],
                    np.log1p(feats["amihud_illiq"]).to_numpy(
                        dtype=np.float32, copy=False
                    ),
                ),
            )
        )

    for f in ["impact_12", "impact_24", "impact_12_perp", "impact_24_perp"]:
        if f in feats:
            rz96_items.append((f, feats[f]))

    if "vol_price_spread" in feats:
        rz96_items.append(
            (
                "vol_price_spread",
                _frame_like(
                    feats["vol_price_spread"],
                    _signed_log1p(feats["vol_price_spread"]).to_numpy(
                        dtype=np.float32, copy=False
                    ),
                ),
            )
        )

    if rz96_items:
        feats.update(_batch_roll_robust_zscore(rz96_items, 96))

    if "volume_price_corr_10h" in feats:
        feats["volume_price_corr_10h"] = np.tanh(feats["volume_price_corr_10h"]).astype(
            np.float32
        )

    # 5) Duration / age
    if "trend_age_hours" in feats:
        tmp = np.log1p(feats["trend_age_hours"])
        feats["trend_age_hours"] = ff.numba_rolling_rank_pct(
            tmp.to_numpy(), 480
        ).astype(np.float32)

    # =========================================================================

    # Transform cache can be enabled for incremental/tail-only runs to persist parquet transforms.
    transform_cache_enabled = bool(cfg.get("feature_transform_cache_enabled", False))
    transform_cache_dir = cfg.get(
        "feature_transform_cache_dir", "./cache/feature_transforms"
    )
    transformer = CausalFeatureTransformer(
        winsor_qt=0.02,
        roll_window=24 * 30,
        cache_dir=transform_cache_dir,
        enable_cache=transform_cache_enabled,
    )

    skip_transform_set = {
        "liq_state",
        "sin_hod",
        "cos_hod",
        "sin_dow",
        "cos_dow",
        "range_24h_pct",
        "range_12h_pct",
        "volatility_zscore",
        "breakout_24h",
        "draw_sym_10h",
        "draw_extreme_10h",
        "G_VOL_LIQ_GT1",
        "G_VOL_LIQ_GT2",
        "G_VOL_LIQ_GT3",
        "G_LIQ_GOOD",
        "G_LIQ_GREAT",
        "G_LIQ_EXCEL",
        "mtf_divergence",
        "vol_price_diverge",
        "meta_alignment",
        # Residualised features — already z-scored, don't double-transform
        "rsi_z",
        "dist_ema_fast_z",
        "dist_vwap_norm_z",
        "flow_persistence_z",
        "excess_6h_z",
        "vol_z_z",
        "atr_expansion_z",
        "coherence_24_z",
        "overext_surprise",
        "blowoff_risk_surprise",
        "exh_qual_surprise",
        "dist_vwap_resid",
        "dist_ema_fast_resid",
        "trend_pct_resid",
    }

    position_sizer_keys = {
        "ATR_spike_ratio",
        "ATR_ratio_short_long",
        "bar_direction_entropy",
        "realized_vol_15m_realized_vol_2h",
        "micro_range_decay",
        "range_decay",
        "vol_regime_transition",
        "close_position_in_range",
        "distance_to_local_high",
        "distance_to_local_low",
        "distance_to_vwap",
        "bidirectional_range_ratio",
        "bollinger_band_width",
        "choppiness_index_20",
        "climax_volume_ratio",
        "volume_z_12",
        "volume_z_24",
        "dist_ema50_atr",
        "dist_ema100_atr",
        "dist_ema200_atr",
        "dist_rolling_7d_high",
        "dist_prior_day_high",
        "dist_prior_day_low",
        "dist_range_mid_atr",
        "dist_weekly_vwap",
        "direction_entropy_20",
        "atr_change_rate",
        "acceleration_of_move",
        "accept_gt66",
        "bars_since_trend_flip",
        "MACD_histogram",
        "RSI",
        "dist_local_swing",
        "dist_ma100_atr",
        "dist_vwap_atr",
        "abs_edge_pred",
    }
    if _needs_feature(
        "close_location_in_bar",
        "session_progress",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
    ):
        idx_utc = c_log.index
        if isinstance(idx_utc, pd.DatetimeIndex):
            if idx_utc.tz is None:
                idx_utc = idx_utc.tz_localize("UTC")
            else:
                idx_utc = idx_utc.tz_convert("UTC")

            hours = idx_utc.hour.to_numpy(dtype=np.float32, copy=False)
            dows = idx_utc.dayofweek.to_numpy(dtype=np.float32, copy=False)

            if _needs_feature("session_progress"):
                feats["session_progress"] = pd.DataFrame(
                    np.tile((hours / 24.0)[:, None], (1, len(c_log.columns))),
                    index=c_log.index,
                    columns=c_log.columns,
                    dtype=np.float32,
                )
            if _needs_feature("hour_sin"):
                feats["hour_sin"] = pd.DataFrame(
                    np.tile(
                        np.sin(2.0 * np.pi * hours / 24.0)[:, None],
                        (1, len(c_log.columns)),
                    ),
                    index=c_log.index,
                    columns=c_log.columns,
                    dtype=np.float32,
                )
            if _needs_feature("hour_cos"):
                feats["hour_cos"] = pd.DataFrame(
                    np.tile(
                        np.cos(2.0 * np.pi * hours / 24.0)[:, None],
                        (1, len(c_log.columns)),
                    ),
                    index=c_log.index,
                    columns=c_log.columns,
                    dtype=np.float32,
                )
            if _needs_feature("dow_sin"):
                feats["dow_sin"] = pd.DataFrame(
                    np.tile(
                        np.sin(2.0 * np.pi * dows / 7.0)[:, None],
                        (1, len(c_log.columns)),
                    ),
                    index=c_log.index,
                    columns=c_log.columns,
                    dtype=np.float32,
                )
            if _needs_feature("dow_cos"):
                feats["dow_cos"] = pd.DataFrame(
                    np.tile(
                        np.cos(2.0 * np.pi * dows / 7.0)[:, None],
                        (1, len(c_log.columns)),
                    ),
                    index=c_log.index,
                    columns=c_log.columns,
                    dtype=np.float32,
                )

    if not requested_feature_set or position_sizer_keys.intersection(
        requested_feature_set
    ):
        tprint("Features: adding missing position sizer features")

        atr_base = feats.get(
            "atr_pct_base",
            pd.DataFrame(index=c_log.index, columns=c_log.columns, dtype=np.float32),
        )
        atr_base_values = atr_base.to_numpy(dtype=np.float32, copy=False)
        if atr_base.empty or not np.isfinite(atr_base_values).any():
            atr_base = raw_atr_pct

        ret_1 = None
        bar_range = None
        atr_mean_24 = None
        high_12 = None
        low_12 = None
        high_24 = None
        low_24 = None

        if _needs_feature(
            "bar_direction_entropy",
            "realized_vol_15m_realized_vol_2h",
            "direction_entropy_20",
            "acceleration_of_move",
            "vol_regime_transition",
        ):
            ret_1 = (c_log.shift(-1) / c_log - 1.0).fillna(0.0).astype(np.float32)

        if _needs_feature(
            "micro_range_decay", "range_decay", "choppiness_index_20", "accept_gt66"
        ):
            bar_range = h_raw - l_raw

        if _needs_feature("ATR_spike_ratio", "ATR_ratio_short_long"):
            atr_mean_24 = atr_base.rolling(24, min_periods=1).mean()
        if _needs_feature("ATR_spike_ratio"):
            feats["ATR_spike_ratio"] = (
                (atr_base / (atr_mean_24 + 1e-9)).fillna(1.0).astype(np.float32)
            )
        if _needs_feature("ATR_ratio_short_long"):
            atr_mean_3 = atr_base.rolling(3, min_periods=1).mean()
            feats["ATR_ratio_short_long"] = (
                (atr_mean_3 / (atr_mean_24 + 1e-9)).fillna(1.0).astype(np.float32)
            )
            del atr_mean_3

        if _needs_feature("bar_direction_entropy"):
            feats["bar_direction_entropy"] = ff.apply_to_frame(
                ret_1, ff.binary_entropy_nb, 12
            ).astype(np.float32)

        rv_1 = rv_2 = rv_24 = None
        if _needs_feature("realized_vol_15m_realized_vol_2h", "vol_regime_transition"):
            rv_1 = ff.apply_to_frame(ret_1, ff.realized_vol_nb, 1)
            rv_2 = ff.apply_to_frame(ret_1, ff.realized_vol_nb, 2)
        if _needs_feature("realized_vol_15m_realized_vol_2h"):
            feats["realized_vol_15m_realized_vol_2h"] = (
                (rv_1 / (rv_2 + 1e-9)).fillna(1.0).astype(np.float32)
            )
        if _needs_feature("vol_regime_transition"):
            rv_24 = ff.apply_to_frame(ret_1, ff.realized_vol_nb, 24)
            rv24_mean_48 = rv_24.rolling(48, min_periods=1).mean()
            feats["vol_regime_transition"] = (
                (rv_24 / (rv24_mean_48 + 1e-9)).fillna(1.0).astype(np.float32)
            )

        if _needs_feature("micro_range_decay"):
            feats["micro_range_decay"] = ff.apply_to_frame(
                bar_range, ff.slope_nb, 3
            ).astype(np.float32)

        if _needs_feature("range_decay"):
            range_mean_3 = bar_range.rolling(3, min_periods=1).mean()
            range_mean_6 = bar_range.rolling(6, min_periods=1).mean()
            feats["range_decay"] = (
                (range_mean_3 / (range_mean_6 + 1e-9)).fillna(1.0).astype(np.float32)
            )

        if _needs_feature("close_position_in_range", "close_location_in_bar"):
            h_mat = h_raw.to_numpy(dtype=np.float32)
            l_mat = l_raw.to_numpy(dtype=np.float32)
            c_mat = c_raw.to_numpy(dtype=np.float32)
            res = close_location_in_bar_nb_parallel(h_mat, l_mat, c_mat)
            feats["close_position_in_range"] = pd.DataFrame(
                res, index=h_raw.index, columns=h_raw.columns, dtype=np.float32
            )
            if _needs_feature("close_location_in_bar"):
                feats["close_location_in_bar"] = feats["close_position_in_range"].copy()

        if _needs_feature(
            "distance_to_local_high",
            "distance_to_local_low",
            "bidirectional_range_ratio",
            "dist_local_swing",
        ):
            high_12 = h_raw.rolling(12, min_periods=1).max()
            low_12 = l_raw.rolling(12, min_periods=1).min()
        if _needs_feature("distance_to_local_high"):
            feats["distance_to_local_high"] = (
                ((high_12 - c_raw) / (c_raw + 1e-9)).fillna(0.0).astype(np.float32)
            )
        if _needs_feature("distance_to_local_low"):
            feats["distance_to_local_low"] = (
                ((c_raw - low_12) / (c_raw + 1e-9)).fillna(0.0).astype(np.float32)
            )

        if _needs_feature("distance_to_vwap", "dist_vwap_atr"):
            vwap_24 = ff.numba_rolling_vwap(c_raw, v_raw, 24).astype(
                np.float32, copy=False
            )
            feats["distance_to_vwap"] = (
                ((c_raw - vwap_24) / (vwap_24 + 1e-9)).fillna(0.0).astype(np.float32)
            )
            if _needs_feature("dist_vwap_atr"):
                feats["dist_vwap_atr"] = feats["distance_to_vwap"]

        if _needs_feature("bidirectional_range_ratio"):
            range_12 = high_12 - low_12
            high_3 = h_raw.rolling(3, min_periods=1).max()
            low_3 = l_raw.rolling(3, min_periods=1).min()
            range_3 = high_3 - low_3
            feats["bidirectional_range_ratio"] = (
                (range_3 / (range_12 + 1e-9)).fillna(1.0).astype(np.float32)
            )

        if _needs_feature("bollinger_band_width"):
            sma_20 = c_log.rolling(20, min_periods=1).mean()
            std_20 = c_log.rolling(20, min_periods=1).std()
            feats["bollinger_band_width"] = (
                (2 * std_20 / (sma_20 + 1e-9)).fillna(0.0).astype(np.float32)
            )

        if _needs_feature("choppiness_index_20"):
            tr = np.maximum(
                np.maximum(h_raw - l_raw, (h_raw - c_raw.shift(1)).abs()),
                (l_raw - c_raw.shift(1)).abs(),
            )
            tr_20 = tr.rolling(20, min_periods=1).sum()
            high_max_20 = h_raw.rolling(20, min_periods=1).max()
            low_min_20 = l_raw.rolling(20, min_periods=1).min()
            range_20 = high_max_20 - low_min_20

            range_safe = np.where(range_20 > 1e-9, range_20, 1e-9)
            tr_safe = np.where(np.isfinite(tr_20), tr_20, 1e-9)
            ratio_clean = np.clip(tr_safe / range_safe, 1e-9, None)

            feats["choppiness_index_20"] = pd.DataFrame(
                np.clip(100.0 * np.log(ratio_clean) / np.log(20.0), 0, 100).astype(
                    np.float32
                ),
                index=tr_20.index,
                columns=tr_20.columns,
            )

        if _needs_feature("climax_volume_ratio"):
            vol_mean_24 = v_raw.rolling(24, min_periods=1).mean()
            vol_max_6 = v_raw.rolling(6, min_periods=1).max()
            feats["climax_volume_ratio"] = (
                (vol_max_6 / (vol_mean_24 + 1e-9)).fillna(1.0).astype(np.float32)
            )
        if _needs_feature("volume_z_12", "volume_z_24"):
            vol_mean_12 = v_raw.rolling(12, min_periods=1).mean()
            vol_std_12 = v_raw.rolling(12, min_periods=1).std()
            vol_mean_24 = v_raw.rolling(24, min_periods=1).mean()
            vol_std_24 = v_raw.rolling(24, min_periods=1).std()
            if _needs_feature("volume_z_12"):
                feats["volume_z_12"] = np.where(
                    vol_std_12 > 1e-9,
                    (v_raw - vol_mean_12) / vol_std_12,
                    0.0,
                ).astype(np.float32)
            if _needs_feature("volume_z_24"):
                volume_z_24 = np.where(
                    vol_std_24 > 1e-9,
                    (v_raw - vol_mean_24) / vol_std_24,
                    0.0,
                ).astype(np.float32)
                feats["volume_z_24"] = volume_z_24

        if _needs_feature("dist_ema50_atr"):
            ema_50 = ff.apply_to_frame(c_log, ff.ema_nb, 50)
            feats["dist_ema50_atr"] = ((c_log - ema_50) / (atr_base + 1e-9)).astype(
                np.float32
            )
        if _needs_feature("dist_ema100_atr"):
            ema_100 = ff.apply_to_frame(c_log, ff.ema_nb, 100)
            feats["dist_ema100_atr"] = ((c_log - ema_100) / (atr_base + 1e-9)).astype(
                np.float32
            )
        if _needs_feature("dist_ema200_atr"):
            ema_200 = ff.apply_to_frame(c_log, ff.ema_nb, 200)
            feats["dist_ema200_atr"] = ((c_log - ema_200) / (atr_base + 1e-9)).astype(
                np.float32
            )

        if _needs_feature("dist_rolling_7d_high"):
            high_168 = h_raw.rolling(168, min_periods=1).max()
            feats["dist_rolling_7d_high"] = (
                ((high_168 - c_raw) / (c_raw + 1e-9)).fillna(0.0).astype(np.float32)
            )

        if _needs_feature(
            "dist_prior_day_high", "dist_prior_day_low", "dist_range_mid_atr"
        ):
            high_24 = h_raw.rolling(24, min_periods=1).max()
            low_24 = l_raw.rolling(24, min_periods=1).min()
        if _needs_feature("dist_prior_day_high"):
            feats["dist_prior_day_high"] = (
                ((high_24.shift(1) - c_raw) / (c_raw + 1e-9))
                .fillna(0.0)
                .astype(np.float32)
            )
        if _needs_feature("dist_prior_day_low"):
            feats["dist_prior_day_low"] = (
                ((c_raw - low_24.shift(1)) / (c_raw + 1e-9))
                .fillna(0.0)
                .astype(np.float32)
            )
        if _needs_feature("dist_range_mid_atr"):
            range_mid = (high_24 + low_24) / 2.0
            feats["dist_range_mid_atr"] = (
                (c_raw - range_mid) / (atr_base + 1e-9)
            ).astype(np.float32)

        if _needs_feature("dist_weekly_vwap"):
            vwap_168 = ff.numba_rolling_vwap(c_raw, v_raw, 168).astype(
                np.float32, copy=False
            )
            feats["dist_weekly_vwap"] = (
                ((c_raw - vwap_168) / (vwap_168 + 1e-9)).fillna(0.0).astype(np.float32)
            )

        if _needs_feature("direction_entropy_20"):
            feats["direction_entropy_20"] = ff.apply_to_frame(
                ret_1, ff.binary_entropy_nb, 20
            ).astype(np.float32)

        if _needs_feature("atr_change_rate"):
            feats["atr_change_rate"] = (
                atr_base.pct_change().fillna(0.0).astype(np.float32)
            )

        if _needs_feature("acceleration_of_move"):
            feats["acceleration_of_move"] = ff.apply_to_frame(
                ret_1, ff.slope_nb, 6
            ).astype(np.float32)

        if _needs_feature("accept_gt66"):
            close_in_range = (c_raw - l_raw) / (h_raw - l_raw + 1e-9)
            feats["accept_gt66"] = (
                (close_in_range > 0.66)
                .astype(np.float32)
                .rolling(6, min_periods=1)
                .mean()
                .astype(np.float32)
            )

        if _needs_feature("MACD_histogram"):
            ema_12 = ff.apply_to_frame(c_log, ff.ema_nb, 12)
            ema_26 = ff.apply_to_frame(c_log, ff.ema_nb, 26)
            macd = ema_12 - ema_26
            signal = ff.apply_to_frame(macd, ff.ema_nb, 9)
            feats["MACD_histogram"] = (macd - signal).astype(np.float32)

        if _needs_feature("RSI") and "rsi" not in feats:
            feats["RSI"] = ff.numba_rsi(c_log, 14).astype(np.float32)

        if _needs_feature("volume_zscore_48h"):
            feats["volume_zscore_48h"] = zscore_rolling(v, 48, winsorize=False)

        if _needs_feature("compression_score"):
            rolling_std_short = _roll_std("c_raw", c_raw, 10)
            rolling_std_long = _roll_std("c_raw", c_raw, 48)
            feats["compression_score"] = (
                (rolling_std_short / (rolling_std_long + 1e-12))
                .fillna(0.0)
                .astype(np.float32)
            )

        if _needs_feature("return_autocorr_48"):
            # ⚡ Bolt: Vectorized rolling autocorrelation of returns
            # 🎯 Why: Iterating over DataFrame columns in Python using pd.Series.rolling().apply() is phenomenally slow.
            # 📊 Impact: ~115x speedup for computing `return_autocorr_48` on large datasets.
            ret1h = c_log.diff(1)
            feats["return_autocorr_48"] = (
                ff.numba_rolling_corr(ret1h, ret1h.shift(1), 48)
                .fillna(0.0)
                .astype(np.float32)
            )

        if _needs_feature("variance_ratio_10_48"):
            var_10 = _roll_std("c_raw", c_raw, 10) ** 2
            var_48 = _roll_std("c_raw", c_raw, 48) ** 2
            feats["variance_ratio_10_48"] = (
                (var_10 / (var_48 + 1e-12)).fillna(0.0).astype(np.float32)
            )

        if _needs_feature("volume_trend_48"):
            feats["volume_trend_48"] = ff.apply_to_frame(v_raw, slope_nb, 48).astype(
                np.float32
            )

        if _needs_feature("volume_autocorr_48"):
            feats["volume_autocorr_48"] = _rolling_autocorr_df(v_raw, 48)

        if _needs_feature("volatility_of_volatility_48"):
            roll_std_4h = ff.apply_to_frame(c_raw, rolling_std_nb, 4)
            feats["volatility_of_volatility_48"] = ff.apply_to_frame(
                roll_std_4h, rolling_std_nb, 48
            ).astype(np.float32)

        if _needs_feature("trend_acceleration"):
            # ema_slope_norm is equivalent to ema20_slope_5h
            ema20 = ff.apply_to_frame(c_raw, ema_nb, 20)
            ema20_slope = ema20 - ema20.shift(5)
            ema20_slope_norm = ema20_slope / (atr_base + 1e-12)
            feats["trend_acceleration"] = (
                (ema20_slope_norm - ema20_slope_norm.shift(1))
                .fillna(0.0)
                .astype(np.float32)
            )

        if _needs_feature("volatility_autocorr_48"):
            feats["volatility_autocorr_48"] = _rolling_autocorr_df(atr_base, 48)

        if _needs_feature("dist_local_swing"):
            dist_to_high = (high_12 - c_raw).abs()
            dist_to_low = (c_raw - low_12).abs()
            feats["dist_local_swing"] = np.minimum(dist_to_high, dist_to_low).astype(
                np.float32
            )

        if _needs_feature("dist_ma100_atr"):
            ma_100 = c_log.rolling(100, min_periods=1).mean()
            feats["dist_ma100_atr"] = ((c_log - ma_100) / (atr_base + 1e-9)).astype(
                np.float32
            )

        if _needs_feature("abs_edge_pred") and "abs_edge_pred" not in feats:
            feats["abs_edge_pred"] = pd.DataFrame(
                index=c_log.index, columns=c_log.columns, dtype=np.float32
            ).fillna(0.0)

        del (
            ret_1,
            bar_range,
            atr_mean_24,
            rv_1,
            rv_2,
            rv_24,
            high_12,
            low_12,
            high_24,
            low_24,
        )
        # Now delete base price/volume DataFrames and intermediates after all features are computed
        del o, h, l, c, v
        del atr, dir_s, rv6, rv12, rv_ratio, mkt_gates
        gc.collect()

    tprint(f"Features: done ({len(feats)} keys)")

    # Add dynamically generated peer context and TS pct to skip set
    for k in feats.keys():
        if (
            k.startswith("cs_rank_")
            or k.startswith("cs_rz_")
            or k.startswith("ts_pct_")
        ):
            skip_transform_set.add(k)

    # Add gated feature patterns to skip set (if gated features were enabled)
    if cfg.get("enable_gated_features", False):
        for w in gate_windows:
            for prefix in [
                "s",
                "reject",
                "retest_accept",
                "tf_qual",
                "mr_qual",
                "vol_z",
                "liquidity",
            ]:
                for suffix in [
                    "mean",
                    "std",
                    "z",
                    "pct",
                    "bin3",
                    "gt25",
                    "gt50",
                    "gt66",
                    "gt75",
                ]:
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

    # --- Structural Z-Normalization Features (Pre-calculated for Mask Optimiser) ---
    # These are computationally intensive robust Z-scores (median/MAD) computed once per symbol.
    bph = int(cfg.get("bars_per_hour", 4))
    window_14d = int(14 * 24 * bph)

    if _needs_feature(
        "z_hl_range",
        "z_intrabar_range_atr",
        "z_compression_expansion",
        "z_volume",
        "z_breakout_up_24",
        "z_breakout_dn_24",
        "z_dist_ema_24",
        "z_dist_vwap_24",
        "z_atr_norm_ret_24",
        "z_sm_momentum_24",
        "z_slope_change_24",
        "z_path_efficiency_24",
    ):
        # 1. Volatility / Range
        ast_hl_range = (h_raw - l_raw).astype(np.float32)
        safe_close_vol = (c_raw * atr_base).clip(lower=1e-6).astype(np.float32)
        z_items: list[tuple[str, pd.DataFrame]] = []

        if _needs_feature("z_hl_range"):
            z_items.append(("z_hl_range", ast_hl_range))

        # Intrabar range normalized by ATR-units
        ast_intrabar_range_atr = (ast_hl_range / safe_close_vol).astype(np.float32)
        if _needs_feature("z_intrabar_range_atr"):
            z_items.append(("z_intrabar_range_atr", ast_intrabar_range_atr))

        # Compression/Expansion: Range Spike vs Bollinger Width
        if _needs_feature("z_compression_expansion"):
            bb_width = (_roll_std("close", c_raw, 20) / c_raw.clip(lower=1e-6)).astype(
                np.float32
            )
            ast_comp_exp = (ast_intrabar_range_atr / bb_width.clip(lower=1e-6)).astype(
                np.float32
            )
            z_items.append(("z_compression_expansion", ast_comp_exp))

        # 2. Volume
        if _needs_feature("z_volume"):
            z_items.append(("z_volume", v_raw))

        # 3. Breakout / Structure (using z=24 as standard)
        z_win = 24
        if _needs_feature(
            "z_breakout_up_24", "z_breakout_dn_24", "z_dist_ema_24", "z_dist_vwap_24"
        ):
            trailing_high_24 = _roll_max("high", h_raw, z_win).shift(1)
            trailing_low_24 = _roll_min("low", l_raw, z_win).shift(1)

            if _needs_feature("z_breakout_up_24"):
                ast_breakout_up = ((c_raw - trailing_high_24) / safe_close_vol).astype(
                    np.float32
                )
                z_items.append(("z_breakout_up_24", ast_breakout_up))

            if _needs_feature("z_breakout_dn_24"):
                ast_breakout_dn = ((trailing_low_24 - c_raw) / safe_close_vol).astype(
                    np.float32
                )
                z_items.append(("z_breakout_dn_24", ast_breakout_dn))

            # Stretch Location (EMA/VWAP)
            if _needs_feature("z_dist_ema_24"):
                ema_24 = (ff.numba_ema_nan_safe(c_raw.to_numpy(), z_win)).astype(
                    np.float32
                )
                ema_24 = pd.DataFrame(ema_24, index=c_raw.index, columns=c_raw.columns)
                ast_dist_ema = ((c_raw - ema_24) / safe_close_vol).astype(np.float32)
                z_items.append(("z_dist_ema_24", ast_dist_ema))

            if _needs_feature("z_dist_vwap_24"):
                # VWAP proxy
                sum_v = _roll_sum("vol", v_raw, z_win)
                sum_pv = _roll_sum("pv", (c_raw * v_raw), z_win)
                vwap_24 = (sum_pv / sum_v.clip(lower=1e-6)).astype(np.float32)
                ast_dist_vwap = ((c_raw - vwap_24) / safe_close_vol).astype(np.float32)
                z_items.append(("z_dist_vwap_24", ast_dist_vwap))

        # 4. Momentum (ATR-normalized)
        if _needs_feature("z_atr_norm_ret_24", "z_sm_momentum_24", "z_slope_change_24"):
            ret_24 = (c_raw / c_raw.shift(z_win) - 1.0).astype(np.float32)
            ast_atr_norm_ret = (ret_24 / atr_base.clip(lower=1e-6)).astype(np.float32)

            if _needs_feature("z_atr_norm_ret_24"):
                z_items.append(("z_atr_norm_ret_24", ast_atr_norm_ret))

            if _needs_feature("z_sm_momentum_24"):
                ret_8 = (c_raw / c_raw.shift(8) - 1.0).astype(np.float32)
                z_items.append(
                    ("z_sm_momentum_24", (ret_8 - ret_24).astype(np.float32))
                )

            if _needs_feature("z_slope_change_24"):
                ast_slope_change = (ret_24 - ret_24.shift(1)).astype(np.float32)
                z_items.append(("z_slope_change_24", ast_slope_change))

        # 5. Path Structure (Efficiency Ratio)
        if _needs_feature("z_path_efficiency_24"):
            ast_abs_moves = (c_raw - c_raw.shift(1)).abs().astype(np.float32)
            sum_abs_moves = _roll_sum("abs_moves", ast_abs_moves, z_win)
            net_move = (c_raw - c_raw.shift(z_win)).astype(np.float32)
            ast_path_eff = (net_move / sum_abs_moves.clip(lower=1e-9)).astype(
                np.float32
            )
            z_items.append(("z_path_efficiency_24", ast_path_eff))

        if z_items:
            feats.update(_batch_roll_robust_zscore(z_items, window_14d))

        # Add to skip transform set (Z-scores are already normalized)
        for k in feats:
            if k.startswith("z_"):
                skip_transform_set.add(k)

    del h_raw, l_raw, c_raw, v_raw, atr_base
    gc.collect()

    # Capture shared index/columns ONCE before converting to numpy
    _feat_index = c_log.index

    _feat_columns = list(c_log.columns)

    feat_keys_list = list(feats.keys())
    # --- Optimized Batched CausalTransform ---
    # Merge dynamically identified skip patterns into the skip_transform_set
    for k in feat_keys_list:
        if (
            k.startswith("cs_rank_")
            or k.startswith("cs_rz_")
            or k.startswith("ts_pct_")
        ):
            skip_transform_set.add(k)
        else:
            arr = np.asarray(feats[k], dtype=np.float32)
            if _is_boolean_like_feature(arr):
                skip_transform_set.add(k)

    tprint(
        f"CausalTransform workset: {len(feats) - len(skip_transform_set)} transform, {len(skip_transform_set)} skipped"
    )

    chunk_size = int(cfg.get("transform_chunk_size", 50))
    feats = transformer.transform_batch(
        feats, skip_keys=skip_transform_set, chunk_size=chunk_size
    )

    del transformer
    gc.collect()

    # Final check for Inf/NaN (numpy arrays now)
    tprint("Features: performing final Inf/NaN check")
    for k in feats:
        arr = feats[k]
        if isinstance(arr, np.ndarray):
            if arr.ndim == 2 and arr.shape == feature_shape:
                arr = pd.DataFrame(arr, index=feature_index, columns=feature_columns)
            elif arr.ndim == 1 and arr.shape[0] == len(feature_index):
                arr = pd.DataFrame(
                    np.broadcast_to(arr[:, None], feature_shape),
                    index=feature_index,
                    columns=feature_columns,
                )
            feats[k] = arr

        values = arr.to_numpy() if isinstance(arr, pd.DataFrame) else np.asarray(arr)
        if not np.isfinite(values).all():
            n_bad = int((~np.isfinite(values)).sum())
            tprint(f"  WARNING: {k} has {n_bad} non-finite values, replacing with 0")
            clean = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
            feats[k] = (
                pd.DataFrame(clean, index=arr.index, columns=arr.columns)
                if isinstance(arr, pd.DataFrame)
                else clean
            )

    for k in (
        "atr_pct",
        "rolling_std_4h",
        "trend_persistence",
        "trend_ratio",
    ):
        feats.pop(k, None)

    tprint(f"Features: done ({len(feats)} keys)")
    return feats, _feat_index, _feat_columns


# ============================================================
# Position Sizer V2 Numba/Numpy Feature Builders
# ============================================================


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
def bars_since_flip_nb(sign: np.ndarray) -> np.ndarray:
    out = np.zeros_like(sign, dtype=np.float32)
    n = len(sign)
    if n == 0:
        return out

    prev = sign[0]
    for i in range(1, n):
        cur = sign[i]
        if np.isnan(cur) or np.isnan(prev) or cur != prev:
            out[i] = 0.0
        else:
            out[i] = out[i - 1] + 1.0
        prev = cur
    return out


@njit(cache=True)
def _median_inplace_nb(values: np.ndarray, length: int) -> float:
    if length <= 0:
        return np.nan
    tmp = np.empty(length, dtype=np.float64)
    for i in range(length):
        tmp[i] = values[i]
    tmp.sort()
    mid = length // 2
    if length % 2 == 1:
        return tmp[mid]
    return 0.5 * (tmp[mid - 1] + tmp[mid])


@njit(parallel=True, cache=True)
def _robust_obs_var_per_col_nb(arr: np.ndarray) -> np.ndarray:
    n_rows, n_cols = arr.shape
    out = np.ones(n_cols, dtype=np.float64)
    if n_rows <= 1:
        return out

    for j in prange(n_cols):
        diffs = np.empty(max(n_rows - 1, 1), dtype=np.float64)
        abs_devs = np.empty(max(n_rows - 1, 1), dtype=np.float64)
        count = 0
        prev = arr[0, j]
        for i in range(1, n_rows):
            cur = arr[i, j]
            if np.isfinite(cur) and np.isfinite(prev):
                diffs[count] = cur - prev
                count += 1
            prev = cur

        if count == 0:
            out[j] = 1.0
            continue

        med = _median_inplace_nb(diffs, count)
        abs_count = 0
        for i in range(count):
            val = diffs[i]
            if np.isfinite(val):
                abs_devs[abs_count] = abs(val - med)
                abs_count += 1

        mad = _median_inplace_nb(abs_devs, abs_count)
        sigma = (1.4826 * mad) / np.sqrt(2.0)
        var = max(sigma, 1e-6) ** 2
        out[j] = var if np.isfinite(var) else 1.0

    return out


@njit(parallel=True, cache=True)
def _kalman_local_level_nb(
    y: np.ndarray, q: np.ndarray, r: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    t_len, n_cols = y.shape
    x = np.full((t_len, n_cols), np.nan, dtype=np.float64)
    innov_var = np.full((t_len, n_cols), np.nan, dtype=np.float64)
    p_state = np.full((t_len, n_cols), np.nan, dtype=np.float64)

    for j in prange(n_cols):
        first = y[0, j]
        x_prev = first if np.isfinite(first) else 0.0
        p_prev = r[j]

        q_j = q[j]
        r_j = r[j]

        for t in range(t_len):
            p_pred = p_prev + q_j
            s_t = p_pred + r_j
            innov_var[t, j] = s_t
            y_t = y[t, j]
            if np.isfinite(y_t):
                k_t = p_pred / max(s_t, 1e-12)
                x_new = x_prev + k_t * (y_t - x_prev)
                p_new = (1.0 - k_t) * p_pred
            else:
                x_new = x_prev
                p_new = p_pred
            x[t, j] = x_new
            p_state[t, j] = p_new
            x_prev = x_new
            p_prev = p_new

    return x, innov_var, p_state


@njit(cache=True)
def _corrcoef_1d_nb(x: np.ndarray, y: np.ndarray) -> float:
    n = len(x)
    mean_x = 0.0
    mean_y = 0.0
    for i in range(n):
        mean_x += x[i]
        mean_y += y[i]
    mean_x /= max(n, 1)
    mean_y /= max(n, 1)

    cov = 0.0
    var_x = 0.0
    var_y = 0.0
    for i in range(n):
        dx = x[i] - mean_x
        dy = y[i] - mean_y
        cov += dx * dy
        var_x += dx * dx
        var_y += dy * dy
    if var_x <= 1e-12 or var_y <= 1e-12:
        return 0.0
    return cov / np.sqrt(var_x * var_y)


@njit(parallel=True, cache=True)
def _decile_monotonicity_score_nb(signal: np.ndarray, ret: np.ndarray) -> float:
    n_rows, n_cols = signal.shape

    # Pre-allocate per-thread local arrays to avoid race conditions
    # We create a 2D array [n_rows, 10] to accumulate sums and counts without locks
    sums_local = np.zeros((n_rows, 10), dtype=np.float64)
    counts_local = np.zeros((n_rows, 10), dtype=np.float64)

    for t in prange(n_rows):
        valid_s = np.empty(n_cols, dtype=np.float64)
        valid_r = np.empty(n_cols, dtype=np.float64)
        n_valid = 0
        for j in range(n_cols):
            s = signal[t, j]
            r = ret[t, j]
            if np.isfinite(s) and np.isfinite(r):
                valid_s[n_valid] = s
                valid_r[n_valid] = r
                n_valid += 1

        if n_valid < 20:
            continue

        order = np.argsort(valid_s[:n_valid])
        for rank in range(n_valid):
            idx = order[rank]
            bucket = min((10 * rank) // n_valid, 9)
            sums_local[t, bucket] += valid_r[idx]
            counts_local[t, bucket] += 1.0

    # Aggregate results from local arrays
    sums = np.zeros(10, dtype=np.float64)
    counts = np.zeros(10, dtype=np.float64)
    for t in range(n_rows):
        for i in range(10):
            sums[i] += sums_local[t, i]
            counts[i] += counts_local[t, i]

    means = np.empty(10, dtype=np.float64)
    valid_mean_count = 0
    mean_sum = 0.0
    for i in range(10):
        if counts[i] > 0:
            means[i] = sums[i] / counts[i]
            mean_sum += means[i]
            valid_mean_count += 1
        else:
            means[i] = np.nan

    if valid_mean_count == 0:
        return 0.0

    fill = mean_sum / valid_mean_count
    for i in range(10):
        if not np.isfinite(means[i]):
            means[i] = fill

    mean_level = 0.0
    for i in range(10):
        mean_level += means[i]
    mean_level /= 10.0

    var = 0.0
    for i in range(10):
        diff = means[i] - mean_level
        var += diff * diff
    if var <= 1e-12:
        return 0.0

    x = np.arange(10, dtype=np.float64)
    return _corrcoef_1d_nb(x, means)


@njit(parallel=True, cache=True)
def _rowwise_median_mad_nb(mat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n_rows, n_cols = mat.shape
    med = np.full(n_rows, np.nan, dtype=np.float64)
    mad = np.full(n_rows, np.nan, dtype=np.float64)

    for i in prange(n_rows):
        row_vals = np.empty(n_cols, dtype=np.float64)
        abs_vals = np.empty(n_cols, dtype=np.float64)
        count = 0
        for j in range(n_cols):
            val = mat[i, j]
            if np.isfinite(val):
                row_vals[count] = val
                count += 1
        if count == 0:
            continue

        row_med = _median_inplace_nb(row_vals, count)
        med[i] = row_med

        for j in range(count):
            abs_vals[j] = abs(row_vals[j] - row_med)
        mad[i] = _median_inplace_nb(abs_vals, count)

    return med, mad


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


@njit(parallel=True, cache=True)
def ema_nb_parallel(mat: np.ndarray, window: int) -> np.ndarray:
    n_rows, n_cols = mat.shape
    out = np.empty((n_rows, n_cols), dtype=np.float32)
    for j in prange(n_cols):
        out[:, j] = ema_nb(mat[:, j], window)
    return out


@njit(cache=True)
def rolling_std_nb(x: np.ndarray, window: int) -> np.ndarray:
    out = np.full_like(x, np.nan)
    n = len(x)
    if n == 0 or window <= 0:
        return out

    # Circular buffer to track outgoing values
    buf = np.empty(window, dtype=np.float64)
    buf_valid = np.zeros(window, dtype=np.bool_)
    buf_idx = 0

    K = 0.0
    K_set = False

    sum_d = 0.0
    sum_d_sq = 0.0
    count = 0

    for i in range(n):
        val_in = x[i]
        in_valid = not np.isnan(val_in)

        # Remove outgoing (only if window is full)
        if i >= window:
            out_idx = buf_idx
            if buf_valid[out_idx]:
                d_out = buf[out_idx] - K
                sum_d -= d_out
                sum_d_sq -= d_out * d_out
                count -= 1

        # Add incoming
        if in_valid:
            if not K_set:
                K = val_in
                K_set = True

            d_in = val_in - K
            sum_d += d_in
            sum_d_sq += d_in * d_in
            count += 1

            buf[buf_idx] = val_in
            buf_valid[buf_idx] = True
        else:
            buf_valid[buf_idx] = False

        buf_idx = (buf_idx + 1) % window

        if count > 1:
            var_num = sum_d_sq - (sum_d * sum_d) / count
            if var_num < 0:
                var_num = 0.0
            out[i] = np.float32(np.sqrt(var_num / (count - 1)))
        elif count == 1:
            out[i] = 0.0

    return out


@njit(parallel=True, cache=True)
def rolling_std_nb_parallel(mat: np.ndarray, window: int) -> np.ndarray:
    n_rows, n_cols = mat.shape
    out = np.empty((n_rows, n_cols), dtype=np.float32)
    for j in prange(n_cols):
        out[:, j] = rolling_std_nb(mat[:, j], window)
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
        slice_ret = ret[start : i + 1]

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
def close_location_in_bar_nb(
    high: np.ndarray, low: np.ndarray, close: np.ndarray
) -> np.ndarray:
    out = np.full_like(close, np.nan)
    for i in range(len(close)):
        rng = high[i] - low[i]
        if not np.isnan(rng) and rng > 1e-9:
            out[i] = (close[i] - low[i]) / rng
        else:
            out[i] = 0.5
    return out


@njit(parallel=True, cache=True)
def close_location_in_bar_nb_parallel(
    high: np.ndarray, low: np.ndarray, close: np.ndarray
) -> np.ndarray:
    n_rows, n_cols = close.shape
    out = np.empty((n_rows, n_cols), dtype=np.float32)
    for j in prange(n_cols):
        out[:, j] = close_location_in_bar_nb(high[:, j], low[:, j], close[:, j])
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
def base_pred_summary_nb(
    base_pred_matrix: np.ndarray,
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
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
                var += (val - r_mean) ** 2
                if val < r_min:
                    r_min = val
                if val > r_max:
                    r_max = val
                if val > 0:
                    pos_count += 1
                elif val < 0:
                    neg_count += 1

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


@njit(cache=True)
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
                if x[j] > mx:
                    mx = x[j]
        if valid:
            out[i] = mx
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
                if x[j] < mn:
                    mn = x[j]
        if valid:
            out[i] = mn
    return out


@njit(cache=True)
def slope_nb(x: np.ndarray, window: int) -> np.ndarray:
    out = np.full_like(x, np.nan)
    n = len(x)
    if n < 2 or window < 2:
        return out

    sum_x = 0.0
    sum_y = 0.0
    sum_xy = 0.0
    sum_x2 = 0.0
    count = 0

    for i in range(n):
        val_in = x[i]
        in_valid = not np.isnan(val_in)

        # Add incoming
        if in_valid:
            sum_x += i
            sum_y += val_in
            sum_xy += i * val_in
            sum_x2 += i * i
            count += 1

        # Remove outgoing
        if i >= window:
            out_idx = i - window
            val_out = x[out_idx]
            if not np.isnan(val_out):
                sum_x -= out_idx
                sum_y -= val_out
                sum_xy -= out_idx * val_out
                sum_x2 -= out_idx * out_idx
                count -= 1

        if i >= window - 1 and count > 1:
            mean_x = sum_x / count
            mean_y = sum_y / count
            num = sum_xy - count * mean_x * mean_y
            den = sum_x2 - count * mean_x**2
            if den > 1e-9:
                out[i] = num / den
            else:
                out[i] = 0.0

    return out


@njit(parallel=True, cache=True)
def slope_nb_parallel(mat: np.ndarray, window: int) -> np.ndarray:
    n_rows, n_cols = mat.shape
    out = np.empty((n_rows, n_cols), dtype=np.float32)
    for j in prange(n_cols):
        out[:, j] = slope_nb(mat[:, j], window)
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
    if window < 2 or n < window:
        return out

    # Pre-allocate buffer for valid elements
    buf = np.empty(window, dtype=x.dtype)
    counts = np.zeros(n_bins, dtype=np.float64)

    for i in range(window - 1, n):
        start = i - window + 1

        valid_count = 0
        mn = np.inf
        mx = -np.inf

        for j in range(start, i + 1):
            val = x[j]
            if not np.isnan(val):
                buf[valid_count] = val
                if val < mn:
                    mn = val
                if val > mx:
                    mx = val
                valid_count += 1

        if valid_count > 1:
            if mx > mn:
                # Reset counts
                for b in range(n_bins):
                    counts[b] = 0.0

                step = (mx - mn) / n_bins
                for j in range(valid_count):
                    b = int((buf[j] - mn) / step)
                    if b == n_bins:
                        b -= 1
                    counts[b] += 1

                ent = 0.0
                inv_count = 1.0 / valid_count
                for b in range(n_bins):
                    c = counts[b]
                    if c > 0:
                        p = c * inv_count
                        ent -= p * np.log2(p)
                out[i] = ent
            else:
                out[i] = 0.0
        elif valid_count == 1:
            out[i] = 0.0

    return out


@njit(parallel=True, cache=True)
def entropy_nb_parallel(mat: np.ndarray, window: int, n_bins: int = 5) -> np.ndarray:
    n_rows, n_cols = mat.shape
    out = np.empty((n_rows, n_cols), dtype=np.float32)
    for j in prange(n_cols):
        out[:, j] = entropy_nb(mat[:, j], window, n_bins)
    return out


@njit(cache=True)
def binary_entropy_nb(x: np.ndarray, window: int) -> np.ndarray:
    out = np.full_like(x, np.nan)
    n = len(x)
    if window < 1 or n < window:
        return out

    pos_c = 0
    neg_c = 0
    tot = 0

    for i in range(n):
        val_in = x[i]
        in_valid = not np.isnan(val_in)

        if i >= window:
            val_out = x[i - window]
            if not np.isnan(val_out):
                tot -= 1
                if val_out > 0:
                    pos_c -= 1
                elif val_out < 0:
                    neg_c -= 1

        if in_valid:
            tot += 1
            if val_in > 0:
                pos_c += 1
            elif val_in < 0:
                neg_c += 1

        if i >= window - 1 and tot > 0:
            p_pos = pos_c / tot
            p_neg = neg_c / tot
            ent = 0.0
            if p_pos > 0:
                ent -= p_pos * np.log2(p_pos)
            if p_neg > 0:
                ent -= p_neg * np.log2(p_neg)
            out[i] = ent

    return out


@njit(parallel=True, cache=True)
def binary_entropy_nb_parallel(mat: np.ndarray, window: int) -> np.ndarray:
    n_rows, n_cols = mat.shape
    out = np.empty((n_rows, n_cols), dtype=np.float32)
    for j in prange(n_cols):
        out[:, j] = binary_entropy_nb(mat[:, j], window)
    return out


def build_position_sizer_feature_frame(
    raw_inputs: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    close = np.ascontiguousarray(raw_inputs.get("close", np.empty(0)), dtype=np.float32)
    high = np.ascontiguousarray(raw_inputs.get("high", np.empty(0)), dtype=np.float32)
    low = np.ascontiguousarray(raw_inputs.get("low", np.empty(0)), dtype=np.float32)
    volume = np.ascontiguousarray(
        raw_inputs.get("volume", np.empty(0)), dtype=np.float32
    )
    atr = np.ascontiguousarray(raw_inputs.get("atr", np.empty(0)), dtype=np.float32)
    meta_pred = np.ascontiguousarray(
        raw_inputs.get("meta_oof_pred", np.empty(0)), dtype=np.float32
    )
    base_pred_matrix = np.ascontiguousarray(
        raw_inputs.get("base_oof_pred_matrix", np.empty((len(close), 0))),
        dtype=np.float32,
    )

    n = len(close)
    if n == 0:
        return {}

    # Basic state / Ensembles
    b_mean, b_std, b_min, b_max, b_range, sign_agree, top2_gap = base_pred_summary_nb(
        base_pred_matrix
    )
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
    impulse_range = range_6_abs  # Proxy impulse window as 6 bars (1h=6h)
    range_last_3bars_impulse_range = np.where(
        impulse_range > 1e-9, range_3_abs / impulse_range, 1.0
    ).astype(np.float32)
    volatility_contraction_ratio = np.where(
        rolling_mean_nb(bar_range, 24) > 1e-9,
        rolling_mean_nb(bar_range, 4) / rolling_mean_nb(bar_range, 24),
        1.0,
    ).astype(np.float32)
    atr_decay_rate = slope_nb(atr, 6).astype(
        np.float32
    )  # Using short slope for decay rate

    rv_1 = realized_vol_nb(ret_1, 1)  # ~15m assuming 15m underlying or 1 bar
    rv_2 = realized_vol_nb(ret_1, 2)
    rv_4 = realized_vol_nb(ret_1, 4)
    rv_6 = realized_vol_nb(ret_1, 6)
    rv_12 = realized_vol_nb(ret_1, 12)
    rv_24 = realized_vol_nb(ret_1, 24)
    rv_48 = realized_vol_nb(ret_1, 48)

    realized_vol_15m_2h = np.where(rv_2 > 1e-9, rv_1 / rv_2, 1.0).astype(
        np.float32
    )  # Ratio approximation
    micro_range_decay = slope_nb(bar_range, 3).astype(np.float32)

    wick_ratio_last_bar = np.where(
        bar_range > 1e-9,
        np.minimum(
            high - np.maximum(close, opens := np.roll(close, 1)),
            np.minimum(close, opens) - low,
        )
        / bar_range,
        0.0,
    ).astype(np.float32)
    close_position_in_range = close_location_in_bar_nb(high, low, close).astype(
        np.float32
    )

    # Simple rejection logic (wick > 50% bar)
    rejection_ratio = np.where(
        bar_range > 1e-9,
        (np.maximum(high - np.maximum(close, opens), np.minimum(close, opens) - low))
        / bar_range,
        0.0,
    )
    rejection_ratio = rolling_mean_nb(rejection_ratio, 6).astype(np.float32)

    vol_sum_3 = rolling_sum_nb(volume, 3)
    vol_sum_4 = rolling_sum_nb(volume, 4)
    vol_sum_6 = rolling_sum_nb(volume, 6)
    vol_sum_12 = rolling_sum_nb(volume, 12)
    vol_sum_24 = rolling_sum_nb(volume, 24)

    mean_vol_24 = rolling_mean_nb(volume, 24)

    impulse_participation_volume = np.where(
        mean_vol_24 > 1e-9, vol_sum_6 / (6 * mean_vol_24), 1.0
    ).astype(np.float32)
    terminal_climax_volume = np.where(
        mean_vol_24 > 1e-9, volume / mean_vol_24, 1.0
    ).astype(np.float32)
    post_impulse_persistence = np.where(
        vol_sum_6 > 1e-9, vol_sum_4 / vol_sum_6, 1.0
    ).astype(np.float32)

    reversal_bar_strength = np.where(
        bar_range > 1e-9, (close - opens) / bar_range, 0.0
    ).astype(np.float32)
    bidirectional_range_ratio = np.where(
        rolling_max_nb(high, 12) - rolling_min_nb(low, 12) > 1e-9,
        range_3_abs / (rolling_max_nb(high, 12) - rolling_min_nb(low, 12)),
        1.0,
    ).astype(np.float32)

    momentum_last_3bars_impulse_return = np.where(
        np.abs(ret_6) > 1e-9, ret_3 / ret_6, 0.0
    ).astype(np.float32)
    drift_after_impulse = slope_nb(close, 4).astype(np.float32)
    slope_last_n_bars = slope_nb(close, 6).astype(np.float32)

    impulse_volume_ratio = np.where(
        mean_vol_24 > 1e-9, vol_sum_12 / (12 * mean_vol_24), 1.0
    ).astype(np.float32)
    terminal_volume_ratio = np.where(
        vol_sum_6 > 1e-9, vol_sum_3 / (vol_sum_6 / 2), 1.0
    ).astype(np.float32)
    post_impulse_volume_persistence2 = np.where(
        vol_sum_6 > 1e-9, vol_sum_4 / (vol_sum_6 * 0.66), 1.0
    ).astype(np.float32)
    impulse_volume_slope = slope_nb(volume, 6).astype(np.float32)

    impulse_vol_ratio = np.where(rv_48 > 1e-9, rv_12 / rv_48, 1.0).astype(np.float32)
    impulse_range_atr_ratio = np.where(
        atr > 1e-9, range_6_abs / rolling_mean_nb(atr, 24), 1.0
    ).astype(np.float32)
    vol_compression_ratio = np.where(rv_6 > 1e-9, rv_4 / rv_6, 1.0).astype(np.float32)
    range_decay = np.where(
        rolling_mean_nb(bar_range, 6) > 1e-9,
        rolling_mean_nb(bar_range, 3) / rolling_mean_nb(bar_range, 6),
        1.0,
    ).astype(np.float32)

    # Model 2 additions
    impulse_speed = np.where(range_6_abs > 1e-9, ret_6 / range_6_abs, 0.0).astype(
        np.float32
    )
    impulse_acceleration = slope_nb(ret_1, 6).astype(np.float32)
    wick_cluster_ratio = rolling_mean_nb(wick_ratio_last_bar, 3).astype(np.float32)
    rejection_bar_count = rolling_sum_nb(
        np.where(wick_ratio_last_bar > 0.4, 1.0, 0.0), 6
    ).astype(np.float32)
    atr_spike_ratio = np.where(
        rolling_mean_nb(atr, 24) > 1e-9, atr / rolling_mean_nb(atr, 24), 1.0
    ).astype(np.float32)

    high_12 = rolling_max_nb(high, 12)
    low_12 = rolling_min_nb(low, 12)
    distance_to_local_high = np.where(
        close > 1e-9, (high_12 - close) / close, 0.0
    ).astype(np.float32)
    distance_to_local_low = np.where(
        close > 1e-9, (close - low_12) / close, 0.0
    ).astype(np.float32)

    vwap_val = vwap_nb(close, volume, 24)
    distance_to_vwap = np.where(
        vwap_val > 1e-9, (close - vwap_val) / vwap_val, 0.0
    ).astype(np.float32)

    climax_volume_ratio = np.where(
        mean_vol_24 > 1e-9, rolling_max_nb(volume, 6) / mean_vol_24, 1.0
    ).astype(np.float32)

    vol_countertrend = np.where(np.sign(ret_1) != np.sign(ret_6), volume, 0.0)
    reversal_volume_ratio = np.where(
        vol_sum_6 > 1e-9, rolling_sum_nb(vol_countertrend, 6) / vol_sum_6, 0.0
    ).astype(np.float32)

    vol_wicks = np.where(wick_ratio_last_bar > 0.4, volume, 0.0)
    mean_vol_12 = rolling_mean_nb(volume, 12)
    rejection_volume_ratio = np.where(
        mean_vol_12 > 1e-9, rolling_sum_nb(vol_wicks, 6) / mean_vol_12, 0.0
    ).astype(np.float32)

    terminal_vol_ratio = np.where(
        rv_6 > 1e-9, rv_3 := realized_vol_nb(ret_1, 3) / rv_6, 1.0
    ).astype(np.float32)

    vol_up = rolling_sum_nb(np.where(ret_1 > 0, volume, 0.0), 12)
    vol_down = rolling_sum_nb(np.where(ret_1 < 0, volume, 0.0), 12)
    volatility_asymmetry = np.where(
        vol_up + vol_down > 1e-9, vol_up / (vol_up + vol_down), 0.5
    ).astype(np.float32)

    # Model 3 additions
    vol_regime_transition = np.where(
        rolling_mean_nb(rv_24, 48) > 1e-9, rv_24 / rolling_mean_nb(rv_24, 48), 1.0
    ).astype(np.float32)
    atr_ratio_short_long = np.where(
        rolling_mean_nb(atr, 24) > 1e-9,
        rolling_mean_nb(atr, 3) / rolling_mean_nb(atr, 24),
        1.0,
    ).astype(np.float32)

    bar_direction_entropy = binary_entropy_nb(ret_1, 12).astype(np.float32)
    wick_entropy = entropy_nb(wick_ratio_last_bar, 12).astype(np.float32)
    impulse_breakdown_score = np.where(ret_6 > 1e-9, ret_3 / ret_6, 0.0).astype(
        np.float32
    )  # Same proxy as momentum ratio

    volume_volatility = np.where(
        mean_vol_12 > 1e-9, rolling_std_nb(volume, 12) / mean_vol_12, 0.0
    ).astype(np.float32)
    volume_regime_shift = np.where(
        mean_vol_24 > 1e-9, rolling_mean_nb(volume, 6) / mean_vol_24, 1.0
    ).astype(np.float32)
    volume_entropy = entropy_nb(volume, 12).astype(np.float32)

    return_per_volume = np.where(volume > 1e-9, np.abs(ret_1) / volume, 0.0).astype(
        np.float32
    )

    mean_rv_12 = rolling_mean_nb(rv_12, 12)
    safe_mean_rv_12 = np.where(mean_rv_12 > 1e-9, mean_rv_12, 1.0)
    vol_of_vol = np.where(
        mean_rv_12 > 1e-9, rolling_std_nb(rv_12, 12) / safe_mean_rv_12, 0.0
    ).astype(np.float32)

    mean_rv_16 = rolling_mean_nb(rv_12, 16)
    safe_mean_rv_16 = np.where(mean_rv_16 > 1e-9, mean_rv_16, 1.0)
    vol_regime_shift_4_16 = np.where(
        mean_rv_16 > 1e-9,
        rolling_mean_nb(rv_12, 4) / safe_mean_rv_16,
        1.0,
    ).astype(np.float32)

    mean_bar_range_12 = rolling_mean_nb(bar_range, 12)
    safe_mean_bar_range_12 = np.where(mean_bar_range_12 > 1e-9, mean_bar_range_12, 1.0)
    range_cv = np.where(
        mean_bar_range_12 > 1e-9,
        rolling_std_nb(bar_range, 12) / safe_mean_bar_range_12,
        0.0,
    ).astype(np.float32)

    safe_rv_12 = np.where(rv_12 > 1e-9, rv_12, 1.0)
    return_vol_ratio = np.where(rv_12 > 1e-9, np.abs(ret_1) / safe_rv_12, 0.0).astype(
        np.float32
    )

    # Pre-existing standard features
    ema_12 = ema_nb(close, 12)
    ema_24 = ema_nb(close, 24)
    price_vs_ema_12_z = np.where(ema_12 > 1e-9, (close - ema_12) / ema_12, 0.0).astype(
        np.float32
    )
    price_vs_ema_24_z = np.where(ema_24 > 1e-9, (close - ema_24) / ema_24, 0.0).astype(
        np.float32
    )
    ema_12_minus_ema_24_z = np.where(
        ema_24 > 1e-9, (ema_12 - ema_24) / ema_24, 0.0
    ).astype(np.float32)
    rv_ratio_6_24 = np.where(rv_24 > 1e-9, rv_6 / rv_24, 1.0).astype(np.float32)
    dsv_12 = downside_semivol_nb(ret_1, 12)

    vol_std_24 = rolling_std_nb(volume, 24)
    volume_z_24 = np.where(
        vol_std_24 > 1e-9, (volume - mean_vol_24) / vol_std_24, 0.0
    ).astype(np.float32)
    volume_z_12 = np.where(
        rolling_std_nb(volume, 12) > 1e-9,
        (volume - mean_vol_12) / rolling_std_nb(volume, 12),
        0.0,
    ).astype(np.float32)
    regime_trend = np.ascontiguousarray(
        raw_inputs.get("regime_trend_score", np.zeros(n)), dtype=np.float32
    )
    regime_vol = np.ascontiguousarray(
        raw_inputs.get("regime_vol_score", np.zeros(n)), dtype=np.float32
    )
    regime_liq = np.ascontiguousarray(
        raw_inputs.get("regime_liquidity_score", np.zeros(n)), dtype=np.float32
    )
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
        "oof_rank_among_candidates": np.zeros(
            n, dtype=np.float32
        ),  # Replaced dynamically cross-sectionally
        "ret_1": ret_1,
        "ret_3": ret_3,
        "ret_6": ret_6,
        "ret_12": ret_12,
        "ret_24": ret_24,
        "price_vs_ema_12_z": price_vs_ema_12_z,
        "price_vs_ema_24_z": price_vs_ema_24_z,
        "ema_12_minus_ema_24_z": ema_12_minus_ema_24_z,
        "trend_slope_12_z": slope_last_n_bars,
        "trend_slope_24_z": slope_nb(close, 24).astype(np.float32),
        "range_1_atr": range_1,
        "range_3_atr": range_3,
        "rv_6": rv_6,
        "rv_12": rv_12,
        "rv_24": rv_24,
        "rv_ratio_6_24": rv_ratio_6_24,
        "close_location_in_bar": close_position_in_range,
        "downside_semivol_12": dsv_12,
        "volume_z_12": volume_z_12,
        "volume_z_24": volume_z_24,
        "liquidity_shock_z": liquidity_shock_nb(vol_sum_6, vol_sum_24).astype(
            np.float32
        ),
        "regime_trend_score": regime_trend,
        "regime_vol_score": regime_vol,
        "regime_liquidity_score": regime_liq,
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
        "dow_sin": dow_sin,
        "dow_cos": dow_cos,
        "session_progress": (hod / 24.0).astype(np.float32),
        # --- NEW MODEL 1 ---
        "range_last_3bars_impulse_range": range_last_3bars_impulse_range,
        "volatility_contraction_ratio": volatility_contraction_ratio,
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
        "range_cv": range_cv,
        "return_vol_ratio": return_vol_ratio,
    }

    from extreme_price_movements.config import POSITION_SIZER_V2_FEATURE_CONFIG

    for key in POSITION_SIZER_V2_FEATURE_CONFIG["shared_feature_keys"]:
        if key not in feature_dict:
            feature_dict[key] = np.zeros(n, dtype=np.float32)

    for k in (
        POSITION_SIZER_V2_FEATURE_CONFIG["model1_edge_feature_keys"]
        + POSITION_SIZER_V2_FEATURE_CONFIG["model2_downside_feature_keys"]
        + POSITION_SIZER_V2_FEATURE_CONFIG["model3_uncertainty_feature_keys"]
    ):
        if k not in feature_dict and k not in [
            "edge_pred",
            "downside_pred",
            "edge_minus_downside",
            "abs_edge_pred",
        ]:
            feature_dict[k] = np.zeros(n, dtype=np.float32)

    return feature_dict


def assemble_feature_matrix(
    feature_dict: Dict[str, np.ndarray], keys: List[str]
) -> np.ndarray:
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


def add_cross_sectional_peer_context_features(
    feats: dict[str, pd.DataFrame], min_group_size: int = 5
) -> dict[str, pd.DataFrame]:
    """
    Computes cross-sectional percentile ranks for select features to provide peer context.
    Ranks are strictly causal (computed per-timestamp).
    """
    from .utils import tprint

    tprint("Computing explicit cross-sectional peer-context features...")

    # Candidates specifically requested for cross-sectional ranking
    cs_candidates = {
        "ret6h",
        "volatility_zscore",
        "amihud_z",
        "volume_z_24",
        "vol_z",
        "range_24h_pct",
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
            df_mat = np.ascontiguousarray(df.to_numpy(dtype=np.float64))
            med_arr, mad_arr = _rowwise_median_mad_nb(df_mat)
            med = pd.Series(med_arr, index=df.index, dtype=np.float32)
            mad = pd.Series(mad_arr, index=df.index, dtype=np.float32)

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


def add_time_series_percentile_features(
    feats: dict[str, pd.DataFrame],
    lookback: int = 720,
    min_history_fraction: float = 0.25,
) -> dict[str, pd.DataFrame]:
    """
    Computes rolling causal time-series percentile ranks for select features.
    """
    import extreme_price_movements.fast_funcs as ff

    from .utils import tprint

    tprint("Computing rolling time-series percentile companion features...")

    # Candidates specifically requested for ts percentiles
    ts_pct_candidates = {
        # Price dynamics
        "ret1h",
        "ret6h",
        "impulse",
        "trend_strength_4d",
        # Volatility & range
        "rv_6h",
        "rv_24h",
        "vol_compression_ratio",
        # Activity
        "vol_shock_z",
        "amihud_z",
        # Geometry
        "breakout_24h",
        "dist_ema_fast",
        "wick_ratio",
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
