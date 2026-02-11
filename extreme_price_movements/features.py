import numpy as np
import pandas as pd
import hashlib
import os
import pickle
import re
from joblib import Memory
from extreme_price_movements.utils import tprint, check_inf_nan
from extreme_price_movements.feature_transforms import CausalFeatureTransformer
from extreme_price_movements.time_utils import ensure_utc
from extreme_price_movements.frac_diff_adaptive import find_min_ffd, frac_diff_ffd
from extreme_price_movements.validation import validate_panel
from extreme_price_movements.gated_features import add_accept_gate_features, add_gate_features
import extreme_price_movements.fast_funcs as ff

# Initialize joblib cache
_cache = Memory("./cache/features", verbose=0)

# --- Per-column FFD incremental cache ---
_FFD_COL_CACHE_DIR = "./cache/ffd_columns"

def _sanitize_col_name(name):
    """Make column name filesystem-safe."""
    return re.sub(r'[^\w\-.]', '_', str(name))

def _col_data_hash(arr):
    """Fast hash of column data for cache key."""
    return hashlib.md5(np.ascontiguousarray(arr).tobytes()).hexdigest()[:16]

def zscore_rolling(x: pd.DataFrame, n: int):
    return ff.numba_zscore(x, n)

def rsi(close: pd.DataFrame, n: int):
    return ff.numba_rsi(close, n)

def ema(x: pd.DataFrame, span: int):
    alpha = 2.0 / (span + 1.0)
    return ff.numba_ewma(x, alpha, False)

def atr_percent(high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame, n: int):
    return ff.numba_atr_no_norm(high, low, close, n)

def rolling_mad(df: pd.DataFrame, window: int):
    med = ff.numba_rolling_median(df, window)
    mad = ff.numba_rolling_median((df - med).abs(), window)
    return mad.astype(np.float32)

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

    mkt_close = _transform_price(mkt_close_raw.to_frame(name="c"))["c"]
    mkt_high  = _transform_price(mkt_high_raw.to_frame(name="h"))["h"]
    mkt_low   = _transform_price(mkt_low_raw.to_frame(name="l"))["l"]
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

def compute_features_hourly(panel, mkt_gates, cfg):
    """
    Compute features. Joblib caching removed — features are persisted to parquet
    by save_features, and the joblib serialization doubled peak memory.
    """
    return _compute_features_impl(panel, mkt_gates, cfg)

def _compute_features_impl(panel, mkt_gates, cfg):
    tprint("Features: compute base matrices")
    
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
    
    if len(mkt_gates) == len(new_idx):
        mkt_gates.index = new_idx
    else:
        mkt_gates = mkt_gates.reindex(new_idx)

    # 2. Transform Open
    o_raw = panel.pop("open").astype(np.float32)
    o_raw.index = new_idx
    o = _transform_price(o_raw, _label="open")
    del o_raw
    gc.collect()

    # 3. Transform High
    h_raw = panel.pop("high").astype(np.float32)
    h_raw.index = new_idx
    h = _transform_price(h_raw, _label="high")
    del h_raw
    gc.collect()

    # 4. Transform Low
    l_raw = panel.pop("low").astype(np.float32)
    l_raw.index = new_idx
    l = _transform_price(l_raw, _label="low")
    del l_raw
    gc.collect()

    # 5. Transform Close
    c_raw = panel.pop("close").astype(np.float32)
    c_raw.index = new_idx

    # Compute Proxy Target (User Request 2026-02-11)
    # Forward 3h returns for feature selection skill metric
    fwd_ret_3h = (c_raw.shift(-3) / c_raw - 1.0).fillna(0.0).astype(np.float32)
    target_proxy = fwd_ret_3h

    c = _transform_price(c_raw, _label="close")
    del c_raw # Note: c is needed for Volume transform? No, but needed for features.
    gc.collect()

    # 6. Transform Volume
    v_raw = panel.pop("volume").astype(np.float32)
    v_raw.index = new_idx
    v = _transform_volume(v_raw)
    del v_raw
    gc.collect()
    
    # Clear panel rest
    panel.clear()

    feats = {}
    feats["ret1h"] = c
    feats["ret6h"] = ff.numba_rolling_sum(c, 6)

    for H in [2, 3, 4, 5, 10, 12, 16, 20, 24, 28]:
        feats[f"ret{H}h"] = ff.numba_rolling_sum(c, H)

    feats["range_pct"] = (h - l)
    feats["gap_pct"]   = (o - c.shift(1))

    atr_base = atr_percent(h, l, c, n=cfg["atr_n"])
    feats["atr_pct_base"] = atr_base

    rsi_base = rsi(c, n=cfg["rsi_n"])
    feats["rsi_base"] = rsi_base
    feats["rsi_slope_base"] = rsi_base.diff(cfg["rsi_slope_n"]).astype(np.float32)

    feats["rv_24h"] = ff.numba_rolling_std(feats["ret1h"], 24)
    feats["rv_6h"] = ff.numba_rolling_std(feats["ret1h"], 6)
    feats["rv_12h"] = ff.numba_rolling_std(feats["ret1h"], 12)

    # New Filter Features (Range & Vol Z-score)
    h_24 = ff.numba_rolling_max(h, 24)
    l_24 = ff.numba_rolling_min(l, 24)
    h_12 = ff.numba_rolling_max(h, 12)
    l_12 = ff.numba_rolling_min(l, 12)

    # range_24h_pct is max_h - min_l. inputs are log-FFD, so diff is %-ish.
    # Do NOT divide by c (FFD) as it crosses 0.
    feats["range_24h_pct"] = (h_24 - l_24).astype(np.float32)
    feats["range_12h_pct"] = (h_12 - l_12).astype(np.float32)
    del h_24, l_24, h_12, l_12

    # Volatility Z-score (using Log-ATR robust z-score)
    # Baseline: 90 days. x = log(ATR/Close).
    # Z = (x - Q(0.45)) / (1.4826 * MAD)
    # atr_base is raw ATR (price units), so we normalize by C
    vol_proxy = (atr_base / (c + 1e-12))
    log_vol = np.log(vol_proxy + 1e-9).astype(np.float32)
    feats["volatility_zscore"] = ff.numba_rolling_robust_zscore(
        log_vol, window=24 * 90, quantile=0.45
    ).astype(np.float32)
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
    if "rsi" in feats:
        feats["rsi_lag1"] = feats["rsi"].shift(1).astype(np.float32)
        # RSI Slope 1h (Momentum Turn for long_mr)
        feats["rsi_1h_slope"] = feats["rsi"].diff(1).fillna(0).astype(np.float32)

    # 4. Tail Risk (CVaR Proxy for long_tf)
    # 5th percentile return over 48 hours (2 days)
    # Use Numba-optimized rolling quantile (O(N) vs Pandas O(N log W))
    feats["cvar_5pct"] = ff.numba_rolling_quantile(feats["ret1h"], 48, 0.05).fillna(0).astype(np.float32)

    # 5. Liquidity Shock (Amihud Proxy for long_tf)
    # |Ret| / (Volume * Price). Spikes indicate price moving on thin liquidity.
    illiq_raw = (feats["ret1h"].abs() / ((v * c) + 1e-12)).replace([np.inf, -np.inf], np.nan)
    feats["amihud_illiq"] = ff.numba_rolling_mean(illiq_raw, 24).fillna(0).astype(np.float32)

    # 6. Skew Proxy (Close Location Value Mean)
    if "clv" in feats:
        feats["clv_mean_24"] = ff.numba_rolling_mean(feats["clv"], 24).fillna(0).astype(np.float32)

    # 7. Stabilization / Falling Knife Features (for long_mr)
    # Climax Volume
    feats["vol_z_4h"] = zscore_rolling(v, 4).fillna(0).astype(np.float32)

    # ATR pct change (Volatility Cooling)
    if "atr_pct" in feats:
        feats["atr_pct_change"] = feats["atr_pct"].pct_change().fillna(0).astype(np.float32)

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

    sma_base = ff.numba_rolling_mean(c, cfg["trend_sma_n"])
    feats["trend_pct_base"] = (c - sma_base).astype(np.float32)

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

    vwap_24 = pd.DataFrame(index=c.index, columns=c.columns, dtype=np.float32)
    for col in c.columns:
        p_arr = c[col].to_numpy(dtype=np.float32)
        v_arr = v[col].to_numpy(dtype=np.float32)
        vwap_24[col] = ff._numba_rolling_vwap(p_arr, v_arr, 24)

    feats["dist_vwap_norm"] = ((c - vwap_24) / (atr_base + 1e-12)).astype(np.float32)

    feats["momentum_accel"] = feats["ret1h"].diff().astype(np.float32)

    log_v = v
    mu_lv = ff.numba_rolling_mean(log_v, cfg["volz_n"])
    sd_lv = ff.numba_rolling_std(log_v, cfg["volz_n"])
    feats["rvol_z"] = ((log_v - mu_lv) / (sd_lv + 1e-12)).astype(np.float32)

    vr = v * feats["ret1h"].abs()
    ema_vr = ema(vr, 24)
    feats["vol_range_shock"] = (vr / (ema_vr + 1e-12)).astype(np.float32)

    v_max = ff.numba_rolling_max(v, 24)
    feats["climax_decay"] = (v_max / (v + 1e-12)).astype(np.float32)

    cum_sv = ff.numba_rolling_sum(signed_vol, 24)
    # Correlation uses internal robust logic, but fillna(0) is good
    feats["cumulative_delta_stall"] = ff.numba_rolling_corr(c, cum_sv, 24).fillna(0).astype(np.float32)
    cum_sv_6 = ff.numba_rolling_sum(signed_vol, 6)
    feats["delta_stall_6"] = ff.numba_rolling_corr(c, cum_sv_6, 6).fillna(0).astype(np.float32)

    feats["vol_expansion_ratio"] = (atr_ema_f / (atr_ema_s + 1e-12)).astype(np.float32)

    sig_s = ff.numba_rolling_std(feats["ret1h"], 6)
    sig_m = ff.numba_rolling_std(feats["ret1h"], 18)
    feats["vol_compression"] = (sig_s / (sig_m + 1e-12)).astype(np.float32)

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
        out = base_df.copy()
        out = out.where(~(rr > cfg["rv_ratio_fast_thr"]), fast_df)
        out = out.where(~(rr < cfg["rv_ratio_slow_thr"]), slow_df)
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
    feats["amihud_z"] = ff.numba_rolling_robust_zscore(amihud_log, window=24*30, quantile=0.50).astype(np.float32)
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
    feats["vol_z_30_calm"] = ff.numba_rolling_robust_zscore(np.log(feats["atr_pct_base"] + 1e-9), window=24 * 30, quantile=0.45).astype(np.float32)
    feats["volume_price_corr_10h"] = ff.numba_rolling_corr(feats["ret1h"].abs(), v, 10).fillna(0).astype(np.float32)

    sma_fast = ff.numba_rolling_mean(c, max(24, int(cfg["trend_sma_n"] * 0.5)))
    sma_slow = ff.numba_rolling_mean(c, int(cfg["trend_sma_n"] * 2))
    trend_fast = (c - sma_fast)
    trend_slow = (c - sma_slow)
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

    # --- New Helper Features for Models ---
    dir_s = np.sign(feats["ret24h"])
    dir_s[dir_s == 0] = 1 # fallback

    atr = feats["atr_pct"] + 1e-12
    rv6 = feats["rv_6h"] + 1e-12
    rv12 = feats["rv_12h"] + 1e-12

    for k in [2, 4, 6, 12, 24]:
        rmax = ff.numba_rolling_max(c, k)
        rmin = ff.numba_rolling_min(c, k)

        rmax_s = rmax.shift(1)
        rmin_s = rmin.shift(1)

        donch = dir_s * (c - rmax_s)
        donch = donch.where(dir_s > 0, -1 * (c - rmin_s))
        feats[f"donch_dist_{k}"] = (donch / atr).clip(lower=0).astype(np.float32)

        pb_raw = dir_s * (c - rmax)
        pb_raw = pb_raw.where(dir_s > 0, -1 * (c - rmin))
        feats[f"pullback_{k}"] = (pb_raw / atr).astype(np.float32)

    feats["excess_6h"] = (feats["ret1h"].abs() / rv6).astype(np.float32)
    feats["excess_12h"] = (feats["ret1h"].abs() / rv12).astype(np.float32)

    for k in [2, 4]:
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

    cur_pnl = (c - o_entry) * dir_s
    gb = (mfe - cur_pnl) / (mfe + 1e-12)
    feats["giveback"] = gb.clip(0, 1).shift(1).astype(np.float32)
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

    gate_window = int(cfg.get("accept_gate_window", 64))
    gate_windows = sorted(set([8, gate_window]))
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
    # Train mask: Exclude last 3 hours (where forward target is invalid/0 due to shift)
    train_mask_proxy = pd.Series(True, index=c.index)
    if len(train_mask_proxy) > 3:
        train_mask_proxy.iloc[-3:] = False

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

    feats["tf_qual"] = (s_pct * feats["tf_tape"]).astype(np.float32)
    feats["mr_qual"] = (reject_like * feats["mr_tape"]).astype(np.float32)

    # 4/ Meta
    feats["rv_ratio_6_24"] = (feats["rv_6h"] / (feats["rv_24h"] + 1e-12)).astype(np.float32)

    # Define gates helpers for Meta

    feats["G_EXH_EFFORT"] = (feats["evr_6"] * (feats["vol_z24"] + 1.0) / (feats["progress"] + 1e-12)).fillna(0).astype(np.float32)
    feats["G_EXH_GIVEBACK"] = (feats["giveback"] * (1.0 + feats["donch_dist_12"])).fillna(0).astype(np.float32)
    feats["G_EXH_TAIL_FAIL"] = (feats["tail_against"] * (feats["ft_2"] - feats["ft_4"]).clip(lower=0)).fillna(0).astype(np.float32)

    feats["G_MR_SPIKE"] = (feats["speed"] * feats["excess_6h"] * clv_inv).fillna(0).astype(np.float32)
    feats["G_TF_GRIND"] = (ret_rat * feats["clv_mean_4"] * pb2_inv).astype(np.float32)
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
    
    # Volatility Interaction Context (New)
    feats["dist_ext_x_vol"] = (feats["donch_dist_12"] * feats["vol_z"]).fillna(0).astype(np.float32)
    feats["regime_x_vol"] = (feats["rv_ratio_6_24"] * feats["vol_z"]).fillna(0).astype(np.float32)
    feats["rsi_x_vol"] = ((feats["rsi"] - 50.0) * feats["vol_z"]).fillna(0).astype(np.float32)

    feats["stall_ext_corr"] = (feats["delta_stall_6"] * feats["donch_dist_12"]).astype(np.float32)

    feats["G_META_EXH"] = (feats["overext"] + feats["G_EXH_EFFORT"] + feats["stall_ext"] + feats["G_EXH_GIVEBACK"]).astype(np.float32)
    feats["G_META_TF_QUAL"] = (feats["accept"] * (1.0 - feats["G_META_EXH"].clip(0,1))).astype(np.float32)
    feats["G_META_MR_QUAL"] = (reject_like * (1.0 - feats["overext"].clip(0,1))).astype(np.float32)
    feats["G_META_AMBIG"] = (ambig_term * feats["rv_ratio_6_24"]).astype(np.float32)

    ret_w = feats["ret10h"]
    local_low = ff.numba_rolling_min(l, 10)
    local_high = ff.numba_rolling_max(h, 10)
    draw_num = np.where((ret_w > 0).to_numpy(), (c - local_low).to_numpy(), (c - local_high).to_numpy())
    feats["draw_sym_10h"] = (np.sign(ret_w) * pd.DataFrame(draw_num, index=c.index, columns=c.columns) / (c + 1e-12)).astype(np.float32)
    feats["draw_extreme_10h"] = feats["draw_sym_10h"].abs().astype(np.float32)

    hi_24_prev = ff.numba_rolling_max(h.shift(1), 24)
    lo_24_prev = ff.numba_rolling_min(l.shift(1), 24)
    up_break = c - hi_24_prev
    dn_break = c - lo_24_prev
    choose_up = (up_break.abs() >= dn_break.abs())
    feats["breakout_24h"] = np.where(choose_up, up_break, dn_break).astype(np.float32) / (c + 1e-12)
    feats["breakout_24h"] = feats["breakout_24h"].astype(np.float32)

    abs_net_score = feats["accept"] + reject_like
    feats["meta_abs_net_x_breakout"] = (abs_net_score * feats["breakout_24h"].abs()).astype(np.float32)
    feats["meta_abs_net_x_drawext"] = (abs_net_score * feats["draw_extreme_10h"]).astype(np.float32)
    feats["meta_abs_net_x_vov_ratio"] = (abs_net_score * (feats["vov_ratio"] - 1.0).clip(lower=0)).astype(np.float32)
    feats["meta_alignment"] = (np.sign(feats["accept"] - reject_like) * np.sign(feats["ret5h"])).astype(np.float32)
    feats["meta_signal_x_accel"] = ((feats["accept"] - reject_like) * feats["accel_5h"]).astype(np.float32)

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
    sign_2h = np.sign(feats["ret2h"])
    sign_24h = np.sign(feats["ret24h"])
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
    price_trend = feats["ret6h"]
    feats["vol_price_diverge"] = (np.sign(vol_trend) * np.sign(price_trend) * -1.0).astype(np.float32)

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
    feats["dist_vwap_resid"] = (feats["dist_vwap_norm"] - 0.5 * mkt_trend_z).astype(np.float32)
    feats["dist_ema_fast_resid"] = (feats["dist_ema_fast"] - 0.5 * mkt_trend_z).astype(np.float32)
    feats["trend_pct_resid"] = (feats["trend_pct"] - 0.5 * mkt_trend_z).astype(np.float32)

    # =====================================================================
    # User Requested Features (Report 2026-02-10) - TF/MR/Alpha
    # =====================================================================

    # Base Components
    ema_6 = ema(c, 6)
    ema_24 = ema(c, 24)
    trend_t = ema_6.diff(1).astype(np.float32)
    feats["trend_t"] = trend_t

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

    # Free target_proxy — no longer needed after gated feature selection
    del target_proxy, time_blocks, train_mask_proxy
    gc.collect()

    tprint(f"Features: {len(feats)} features before CausalTransform. Applying transforms...")
    transformer = CausalFeatureTransformer(winsor_qt=0.02, roll_window=24*30)

    skip_transform_set = {
        "sin_hod", "cos_hod", "sin_dow", "cos_dow", "range_24h_pct", "range_12h_pct",
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

    for w in gate_windows:
        for prefix in ["s", "reject", "retest_accept", "tf_qual", "mr_qual", "vol_z", "liquidity"]:
            for suffix in ["mean", "std", "z", "pct", "bin3", "gt25", "gt50", "gt66", "gt75", "gt85", "gt90"]:
                skip_transform_set.add(f"{prefix}_{suffix}_{w}")

    feat_keys_list = list(feats.keys())
    n_transformed, n_skipped = 0, 0
    for i, k in enumerate(feat_keys_list):
        if k in skip_transform_set:
            feats[k] = feats[k].astype(np.float32)
            n_skipped += 1
            continue
        try:
            feats[k] = transformer.transform(feats[k], name=k)
            n_transformed += 1
        except Exception as e:
            tprint(f"Warning: Transform failed for {k}: {e}")
            import traceback
            traceback.print_exc()
            feats[k] = feats[k].astype(np.float32)
        if (i + 1) % 50 == 0:
            tprint(f"  CausalTransform progress: {i+1}/{len(feat_keys_list)}")
    tprint(f"CausalTransform complete: {n_transformed} transformed, {n_skipped} skipped")

    # Final check for Inf/NaN
    tprint("Features: performing final Inf/NaN check")
    for k, v in feats.items():
        check_inf_nan(v, k)

    tprint(f"Features: done ({len(feats)} keys)")
    return feats
