import numpy as np
import pandas as pd
import hashlib
from joblib import Memory
from extreme_price_movements.utils import tprint, check_inf_nan
from extreme_price_movements.feature_transforms import CausalFeatureTransformer
from extreme_price_movements.time_utils import ensure_utc
from extreme_price_movements.frac_diff_adaptive import find_min_ffd, frac_diff_ffd
from extreme_price_movements.validation import validate_panel
import extreme_price_movements.fast_funcs as ff

# Initialize joblib cache
_cache = Memory("./cache/features", verbose=0)

def zscore_rolling(x: pd.DataFrame, n: int):
    return ff.numba_zscore(x, n)

def rsi(close: pd.DataFrame, n: int):
    return ff.numba_rsi(close, n)

def ema(x: pd.DataFrame, span: int):
    alpha = 2.0 / (span + 1.0)
    return ff.apply_to_frame(x, ff._numba_ewma_nan_safe, alpha, False)

def atr_percent(high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame, n: int):
    return ff.numba_atr_no_norm(high, low, close, n)

@_cache.cache
def _transform_price(df):
    tprint("Transforming Prices: Log -> EWMA(5) -> Adaptive FracDiff")
    df_log = np.log(df + 1e-9)
    df_den = ff.apply_to_frame(df_log, ff._numba_ewma_nan_safe, 2.0/6.0, False)
    
    # Apply adaptive FFD per column
    df_fd = pd.DataFrame(index=df.index, columns=df.columns, dtype=np.float32)
    total_cols = len(df_den.columns)
    for i, col in enumerate(df_den.columns):
        if (i+1) % 5 == 0 or (i+1) == total_cols:
             tprint(f"Adaptive FFD: processing {i+1}/{total_cols} - {col}")
        
        series = df_den[col].dropna()
        if len(series) < 100:
            # Fallback to fixed d=0.4 for short series
            d_opt = 0.4
        else:
            # Find minimal d for stationarity
            d_opt, _, _ = find_min_ffd(series, d_range=(0.0, 1.0), step=0.1)
        
        # Apply FFD
        df_fd[col] = frac_diff_ffd(df_den[col], d_opt, thres=1e-5)
    
    tprint(f"Adaptive FFD: d range [{df_fd.min().min():.3f}, {df_fd.max().max():.3f}]")
    return df_fd

@_cache.cache
def _transform_volume(df):
    tprint("Transforming Volume: Log -> EWMA(5)")
    df_log = np.log(df + 1.0)
    df_den = ff.apply_to_frame(df_log, ff._numba_ewma_nan_safe, 2.0/6.0, False)
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
    df["G_TREND"] = (df["mkt_ret24h"].abs() > gate_trend_thr).astype(np.int32)
    df["mkt_rv_ratio"] = df["mkt_rv"] / (df["mkt_rv_med"] + 1e-12)

    float_cols = ["mkt_rv_med", "mkt_rv_ratio"]
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

def _hash_panel(panel):
    """Create hash of panel data for cache key."""
    h = hashlib.md5()
    for key in sorted(panel.keys()):
        h.update(key.encode())
        h.update(panel[key].values.tobytes())
    return h.hexdigest()

def _hash_mkt_gates(mkt_gates):
    """Create hash of market gates for cache key."""
    return hashlib.md5(mkt_gates.values.tobytes()).hexdigest()

@_cache.cache
def _compute_features_cached(panel_hash, mkt_gates_hash, cfg_tuple, panel, mkt_gates):
    """Cached implementation of feature computation."""
    return _compute_features_impl(panel, mkt_gates, dict(cfg_tuple))

def compute_features_hourly(panel, mkt_gates, cfg):
    """
    Compute features with caching to avoid recomputation.
    Uses hash-based cache key for panel and mkt_gates.
    """
    # Create cache keys
    panel_hash = _hash_panel(panel)
    mkt_gates_hash = _hash_mkt_gates(mkt_gates)
    cfg_tuple = tuple(sorted(cfg.items()))
    
    # Call cached implementation
    return _compute_features_cached(panel_hash, mkt_gates_hash, cfg_tuple, panel, mkt_gates)

def _compute_features_impl(panel, mkt_gates, cfg):
    tprint("Features: compute base matrices")
    
    # Validate panel data quality
    validation_results = validate_panel(panel, raise_on_error=False, verbose=False)
    if not validation_results['valid']:
        tprint(f"WARNING: Panel validation failed with {len(validation_results['errors'])} errors")
        for error in validation_results['errors'][:3]:  # Show first 3 errors
            tprint(f"  - {error}")
    
    o_raw, h_raw, l_raw, c_raw, v_raw = panel["open"], panel["high"], panel["low"], panel["close"], panel["volume"]

    new_idx = ensure_utc(pd.DataFrame(index=c_raw.index)).index
    o_raw.index = new_idx
    h_raw.index = new_idx
    l_raw.index = new_idx
    c_raw.index = new_idx
    v_raw.index = new_idx

    if len(mkt_gates) == len(c_raw):
        mkt_gates.index = new_idx
    else:
        mkt_gates = mkt_gates.reindex(new_idx)

    o_raw = o_raw.astype(np.float32)
    h_raw = h_raw.astype(np.float32)
    l_raw = l_raw.astype(np.float32)
    c_raw = c_raw.astype(np.float32)
    v_raw = v_raw.astype(np.float32)

    o = _transform_price(o_raw)
    h = _transform_price(h_raw)
    l = _transform_price(l_raw)
    c = _transform_price(c_raw)
    v = _transform_volume(v_raw)

    feats = {}
    feats["ret1h"] = c
    feats["ret6h"] = ff.numba_rolling_sum(c, 6)

    for H in [2, 3, 4, 12, 16, 20, 24, 28]:
        feats[f"ret{H}h"] = ff.numba_rolling_sum(c, H)

    feats["range_pct"] = (h - l)
    feats["gap_pct"]   = (o - c.shift(1))

    atr_base = atr_percent(h, l, c, n=cfg["atr_n"])
    feats["atr_pct_base"] = atr_base

    rsi_base = rsi(c, n=cfg["rsi_n"])
    feats["rsi_base"] = rsi_base
    feats["rsi_slope_base"] = rsi_base.diff(cfg["rsi_slope_n"]).astype(np.float32)

    feats["rv_24h"] = ff.apply_to_frame(feats["ret1h"], ff._numba_rolling_std_nan_safe, 24)
    feats["rv_6h"] = ff.apply_to_frame(feats["ret1h"], ff._numba_rolling_std_nan_safe, 6)
    feats["rv_12h"] = ff.apply_to_frame(feats["ret1h"], ff._numba_rolling_std_nan_safe, 12)

    feats["qv"] = (c * v).astype(np.float32)
    feats["vol_z24_base"] = zscore_rolling(v, 24)
    feats["vol_z_base"]   = zscore_rolling(v, cfg["volz_n"])

    ema_fast_base = ema(c, cfg["ema_fast"])
    ema_slow_base = ema(c, cfg["ema_slow"])
    feats["dist_ema_fast_base"] = ((c - ema_fast_base) / (atr_base + 1e-12)).astype(np.float32)
    feats["dist_ema_slow_base"] = ((c - ema_slow_base) / (atr_base + 1e-12)).astype(np.float32)

    feats["roc_div"] = (feats["ret1h"] - feats["ret6h"]).astype(np.float32)
    feats["ret1h_z"] = (feats["ret1h"] / (feats["rv_24h"] + 1e-12)).astype(np.float32)

    body = (c - o).abs()
    upper_wick = (h - c.where(c >= o, o)).clip(lower=0)
    lower_wick = (c.where(c <= o, o) - l).clip(lower=0)
    feats["body_pct"] = body.astype(np.float32)
    feats["wick_body_ratio"] = ((upper_wick + lower_wick) / (body + 1e-12)).astype(np.float32)

    feats["vol_price_spread"] = (v / ((h - l) + 1e-12)).astype(np.float32)

    prev_close = c.shift(1)
    tr_1 = (h - l)
    tr_2 = (h - prev_close).abs()
    tr_3 = (l - prev_close).abs()
    tr = np.maximum(tr_1, np.maximum(tr_2, tr_3))
    atr_tr = ff.apply_to_frame(tr, ff._numba_ewma_nan_safe, 1.0/cfg["atr_n"], False)
    feats["atr_expansion"] = (tr / (atr_tr + 1e-12)).astype(np.float32)

    sma_base = ff.apply_to_frame(c, ff._numba_rolling_mean_nan_safe, cfg["trend_sma_n"])
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
    feats["efficiency"] = ff.apply_to_frame(eff, ff._numba_rolling_mean_nan_safe, 12)

    skew_ser = feats["ret1h"].skew(axis=1)
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
    fvg_bull = (l_prev2 - h).clip(lower=0) / (c + 1e-12)
    fvg_bear = (l - h_prev2).clip(lower=0) / (c + 1e-12)
    feats["fvg"] = (fvg_bull - fvg_bear).astype(np.float32)

    feats["churn"] = (v / ((c - o).abs() + 1e-12)).astype(np.float32)
    feats["slope"] = ((ema_fast_base - ema_slow_base) / (atr_base + 1e-12)).astype(np.float32)

    t_snr_num = ema(feats["ret1h"], 6).abs()
    t_snr_den = ff.apply_to_frame(feats["ret1h"], ff._numba_rolling_std_nan_safe, 24)
    feats["trend_snr"] = (t_snr_num / (t_snr_den + 1e-12)).astype(np.float32)

    feats["v_power"] = (v / (c.abs() + 1e-12)).astype(np.float32)
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
    mu_lv = ff.apply_to_frame(log_v, ff._numba_rolling_mean_nan_safe, cfg["volz_n"])
    sd_lv = ff.apply_to_frame(log_v, ff._numba_rolling_std_nan_safe, cfg["volz_n"])
    feats["rvol_z"] = ((log_v - mu_lv) / (sd_lv + 1e-12)).astype(np.float32)

    vr = v * feats["ret1h"].abs()
    ema_vr = ema(vr, 24)
    feats["vol_range_shock"] = (vr / (ema_vr + 1e-12)).astype(np.float32)

    v_max = ff.numba_rolling_max(v, 24)
    feats["climax_decay"] = (v_max / (v + 1e-12)).astype(np.float32)

    cum_sv = ff.numba_rolling_sum(signed_vol, 24)
    feats["cumulative_delta_stall"] = ff.numba_rolling_corr(c, cum_sv, 24).fillna(0).astype(np.float32)
    cum_sv_6 = ff.numba_rolling_sum(signed_vol, 6)
    feats["delta_stall_6"] = ff.numba_rolling_corr(c, cum_sv_6, 6).fillna(0).astype(np.float32)

    feats["vol_expansion_ratio"] = (atr_ema_f / (atr_ema_s + 1e-12)).astype(np.float32)

    sig_s = ff.apply_to_frame(feats["ret1h"], ff._numba_rolling_std_nan_safe, 6)
    sig_m = ff.apply_to_frame(feats["ret1h"], ff._numba_rolling_std_nan_safe, 18)
    feats["vol_compression"] = (sig_s / (sig_m + 1e-12)).astype(np.float32)

    rv_ratio = mkt_gates["mkt_rv_ratio"].reindex(c.index).astype(np.float32)
    feats["mkt_rv_ratio"] = rv_ratio

    def pick_by_rv(fast_df, base_df, slow_df):
        rr = pd.DataFrame(np.repeat(rv_ratio.to_numpy()[:,None], base_df.shape[1], axis=1),
                          index=base_df.index, columns=base_df.columns).astype(np.float32)
        out = base_df.copy()
        out = out.where(~(rr > cfg["rv_ratio_fast_thr"]), fast_df)
        out = out.where(~(rr < cfg["rv_ratio_slow_thr"]), slow_df)
        return out.astype(np.float32)

    rsi_fast = rsi(c, max(2, int(cfg["rsi_n"] * 0.5)))
    rsi_slow = rsi(c, int(cfg["rsi_n"] * 2))
    feats["rsi"] = pick_by_rv(rsi_fast, rsi_base, rsi_slow)

    atr_fast = atr_percent(h, l, c, max(2, int(cfg["atr_n"] * 0.5)))
    atr_slow = atr_percent(h, l, c, int(cfg["atr_n"] * 2))
    feats["atr_pct"] = pick_by_rv(atr_fast, atr_base, atr_slow)

    volz_fast = zscore_rolling(v, max(24, int(cfg["volz_n"] * 0.5)))
    volz_slow = zscore_rolling(v, int(cfg["volz_n"] * 2))
    feats["vol_z"] = pick_by_rv(volz_fast, feats["vol_z_base"], volz_slow)

    sma_fast = ff.apply_to_frame(c, ff._numba_rolling_mean_nan_safe, max(24, int(cfg["trend_sma_n"] * 0.5)))
    sma_slow = ff.apply_to_frame(c, ff._numba_rolling_mean_nan_safe, int(cfg["trend_sma_n"] * 2))
    trend_fast = (c - sma_fast)
    trend_slow = (c - sma_slow)
    feats["trend_pct"] = pick_by_rv(trend_fast, feats["trend_pct_base"], trend_slow)

    ema_fast_f = ema(c, max(4, int(cfg["ema_fast"] * 0.5)))
    ema_fast_s = ema(c, int(cfg["ema_fast"] * 2))
    dist_fast_f = (c - ema_fast_f) / (feats["atr_pct"] + 1e-12)
    dist_fast_s = (c - ema_fast_s) / (feats["atr_pct"] + 1e-12)
    feats["dist_ema_fast"] = pick_by_rv(dist_fast_f, feats["dist_ema_fast_base"], dist_fast_s)

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

    clv_raw = ((2 * c - h - l) / ((h - l) + 1e-9))
    feats["clv"] = clv_raw.astype(np.float32)
    feats["clv_mean_2"] = ff.apply_to_frame(feats["clv"], ff._numba_rolling_mean_nan_safe, 2).astype(np.float32)
    feats["clv_mean_4"] = ff.apply_to_frame(feats["clv"], ff._numba_rolling_mean_nan_safe, 4).astype(np.float32)

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

    feats["mfe_4h"] = (mfe / atr).astype(np.float32)
    feats["mae_4h"] = (mae / atr).astype(np.float32)

    cur_pnl = (c - o_entry) * dir_s
    gb = (mfe - cur_pnl) / (mfe + 1e-12)
    feats["giveback"] = gb.clip(0, 1).astype(np.float32)

    # --- COMPOSITE / INTERACTION FEATURES ---

    # 1/ Exhaustion
    feats["overext"] = (feats["donch_dist_12"] * feats["excess_6h"]).astype(np.float32)
    feats["overext_weak"] = (feats["donch_dist_12"] * (1.0 - feats["clv_mean_4"].clip(lower=0))).astype(np.float32)
    feats["effort_gate"] = (feats["evr_6"] * (feats["vol_z24"] + 1.0) / (feats["progress"] + 1e-12)).astype(np.float32)
    feats["stall_ext"] = (feats["donch_dist_12"] * (1.0 - feats["delta_stall_6"])).astype(np.float32)
    feats["tail_fail"] = (feats["tail_against"] * (feats["ft_2"] - feats["ft_4"]).clip(lower=0)).astype(np.float32)

    pb_avg = (feats["pullback_2"] + feats["pullback_4"]) / 2.0
    fail_term = (feats["failure_2"] + 0.5 * feats["failure_4"])
    feats["reject"] = ((1.0 - feats["clv_mean_4"].clip(lower=0)) * pb_avg * fail_term).astype(np.float32)

    feats["impulse_ratio_24"] = (feats["ret1h"].abs() / (feats["ret24h"].abs() + 1e-12)).astype(np.float32)
    feats["impulse_ratio_12"] = (feats["ret1h"].abs() / (feats["ret12h"].abs() + 1e-12)).astype(np.float32)
    feats["accel"] = (feats["ret1h"] - feats["ret1h"].shift(1)).abs() / (feats["rv_6h"] + 1e-12)
    feats["blowoff_risk"] = (feats["impulse_ratio_24"] * feats["accel"] * feats["donch_dist_12"]).astype(np.float32)

    # 2/ Spike Anatomy / Regime
    s_max = feats["ret16h"].abs()
    for k in [20, 24, 28]:
        s_max = np.maximum(s_max, feats[f"ret{k}h"].abs())
    feats["S"] = (dir_s * s_max).astype(np.float32)

    feats["coherence_24"] = (dir_s * (feats["ret6h"] + feats["ret12h"] + feats["ret24h"]) / (feats["rv_24h"] + 1e-12)).astype(np.float32)
    turb = mkt_gates["mkt_rv_ratio"].reindex(c.index).astype(np.float32)
    mkt_ret6h_s = mkt_gates["mkt_ret6h"].reindex(c.index).astype(np.float32)
    tape_align = (dir_s * mkt_ret6h_s)
    feats["tf_tape"] = (tape_align.clip(lower=0) / (1.0 + turb)).astype(np.float32)
    feats["mr_tape"] = ((-tape_align).clip(lower=0) / (1.0 + turb)).astype(np.float32)

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
    feats["accept"] = (ft2_pos * clv4_pos * pb2_inv).astype(np.float32)
    feats["retest_accept"] = (ft4_pos * clv4_pos * pb4_inv).astype(np.float32)

    feats["tf_qual"] = (feats["accept"] * feats["tf_tape"]).astype(np.float32)

    # feats["reject"] already computed
    feats["mr_qual"] = (feats["reject"] * feats["mr_tape"]).astype(np.float32)
    feats["retrace_12"] = (-feats["pullback_12"]).astype(np.float32)

    # 4/ Meta
    feats["rv_ratio_6_24"] = (feats["rv_6h"] / (feats["rv_24h"] + 1e-12)).astype(np.float32)

    # Define gates helpers for Meta
    accept2 = feats["G_TF_ACCEPT2"] = (ft2_pos * clv4_pos * pb2_inv).astype(np.float32)
    # Re-define Gate vars if they were not defined yet (G_TF_ACCEPT2 was defined in GATES section in previous version, now I define it here or use it)
    # Actually I haven't defined GATES section yet in this rewritten version!

    # Let's define the GATES first as they are features too
    feats["G_EXH_STALL_EXT"] = (feats["donch_dist_12"] * (1.0 - feats["delta_stall_6"])).astype(np.float32)
    feats["G_EXH_EFFORT"] = (feats["evr_6"] * (feats["vol_z24"] + 1.0) / (feats["progress"] + 1e-12)).astype(np.float32)
    feats["G_EXH_BLOWOFF"] = (feats["donch_dist_12"] * feats["excess_6h"]).astype(np.float32)
    feats["G_EXH_GIVEBACK"] = (feats["giveback"] * (1.0 + feats["donch_dist_12"])).astype(np.float32)
    feats["G_EXH_REJECT"] = ((1.0 - feats["clv_mean_4"]) * feats["pullback_4"].abs()).astype(np.float32)
    feats["G_EXH_TAIL_FAIL"] = (feats["tail_against"] * (feats["ft_2"] - feats["ft_4"]).clip(lower=0)).astype(np.float32)
    feats["G_TF_ACCEPT"] = (ft2_pos * clv4_pos).astype(np.float32)
    feats["G_TF_ACCEPT2"] = accept2 # Defined above
    feats["G_MR_REJECT"] = (fail_sum * clv_inv * pb_avg_abs).astype(np.float32)
    feats["G_MR_OVEREXT"] = (feats["donch_dist_12"] * (1.0 - ft2_pos)).astype(np.float32)
    feats["G_MR_SPIKE"] = (feats["speed"] * feats["excess_6h"] * clv_inv).astype(np.float32)
    feats["G_TF_GRIND"] = (ret_rat * feats["clv_mean_4"] * pb2_inv).astype(np.float32)
    feats["G_MR_TAIL"] = (feats["tail_against"] * (1.0 + feats["donch_dist_6"])).astype(np.float32)
    feats["G_TF_RETEST_OK"] = (ft4_pos * clv4_pos * pb4_inv).astype(np.float32)

    # Meta Features using Gates
    reject = feats["G_MR_REJECT"]
    ambig_term = (1.0 - np.maximum(accept2, reject))
    feats["ambig"] = (ambig_term * feats["rv_ratio_6_24"]).astype(np.float32)

    feats["stage_tf"] = (feats["accept"] * feats["coherence_24"]).astype(np.float32)
    feats["stage_blowoff"] = (feats["blowoff_risk"] + feats["effort_gate"] + feats["stall_ext"]).astype(np.float32)
    feats["stage_mr"] = (feats["reject"] * (1.0 + feats["overext"])).astype(np.float32)
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
    feats["stall_ext"] = (feats["delta_stall_6"] * feats["donch_dist_12"]).astype(np.float32)

    feats["G_META_EXH"] = (feats["G_EXH_BLOWOFF"] + feats["G_EXH_EFFORT"] + feats["G_EXH_STALL_EXT"] + feats["G_EXH_GIVEBACK"]).astype(np.float32)
    feats["G_META_TF_QUAL"] = (feats["G_TF_ACCEPT2"] * (1.0 - feats["G_META_EXH"].clip(0,1))).astype(np.float32)
    feats["G_META_MR_QUAL"] = (feats["G_MR_REJECT"] * (1.0 - feats["G_EXH_BLOWOFF"].clip(0,1))).astype(np.float32)
    feats["G_META_AMBIG"] = (ambig_term * feats["rv_ratio_6_24"]).astype(np.float32)

    feats["spike_score"] = (feats["speed"] * feats["excess_6h"]).astype(np.float32)
    feats["grind_score"] = (ret_rat * feats["clv_mean_4"]).astype(np.float32)
    coh_norm = feats["coh"].clip(0,1)
    feats["chop_score"] = (feats["rv_ratio_6_24"] * (1.0 - coh_norm)).astype(np.float32)

    tprint("Features: Applying Causal Transforms (Log + Winsor + ZScore)")
    transformer = CausalFeatureTransformer(winsor_qt=0.02, roll_window=24*30)

    skip_transform = ["sin_hod", "cos_hod", "sin_dow", "cos_dow"]

    for k in feats.keys():
        tprint(f"Generating feature: {k}")
        if k in skip_transform:
            feats[k] = feats[k].astype(np.float32)
            continue
        try:
            feats[k] = transformer.transform(feats[k], name=k)
        except Exception as e:
            tprint(f"Warning: Transform failed for {k}: {e}")
            import traceback
            traceback.print_exc()
            feats[k] = feats[k].astype(np.float32)

    # Final check for Inf/NaN
    tprint("Features: performing final Inf/NaN check")
    for k, v in feats.items():
        check_inf_nan(v, k)

    tprint(f"Features: done ({len(feats)} keys)")
    return feats
