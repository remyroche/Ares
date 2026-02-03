import numpy as np
import pandas as pd
from extreme_price_movements.utils import tprint
from extreme_price_movements.feature_transforms import CausalFeatureTransformer
from extreme_price_movements.time_utils import ensure_utc
import extreme_price_movements.fast_funcs as ff

def zscore_rolling(x: pd.DataFrame, n: int):
    # Use Numba implementation
    return ff.numba_zscore(x, n)

def rsi(close: pd.DataFrame, n: int):
    # Use Numba implementation
    return ff.numba_rsi(close, n)

def ema(x: pd.DataFrame, span: int):
    alpha = 2.0 / (span + 1.0)
    return ff.apply_to_frame(x, ff._numba_ewma, alpha, False)

def atr_percent(high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame, n: int):
    # Use Numba implementation
    return ff.numba_atr(high, low, close, n)

def time_sin_cos(index: pd.DatetimeIndex):
    hod = index.hour.to_numpy()
    dow = index.dayofweek.to_numpy()
    sin_hod = np.sin(2*np.pi*hod/24.0)
    cos_hod = np.cos(2*np.pi*hod/24.0)
    sin_dow = np.sin(2*np.pi*dow/7.0)
    cos_dow = np.cos(2*np.pi*dow/7.0)
    return sin_hod, cos_hod, sin_dow, cos_dow

def compute_market_features(panel, basket_syms, trend_sma_hours=24*14):
    c = panel["close"]
    h = panel["high"]
    l = panel["low"]
    v = panel["volume"]

    basket = [s for s in basket_syms if s in c.columns]
    if not basket:
        basket = list(c.columns)

    mkt_close = c[basket].mean(axis=1)
    mkt_high  = h[basket].mean(axis=1)
    mkt_low   = l[basket].mean(axis=1)
    mkt_vol   = v[basket].mean(axis=1)

    # Note: numba_pct_change calculates (x[t] - x[t-N]) / x[t-N].
    # This uses information up to time t, so it is safe for predicting returns from t onwards.
    mkt_ret24h_df = ff.numba_pct_change(mkt_close.to_frame(), 24)
    mkt_ret24h = mkt_ret24h_df[mkt_ret24h_df.columns[0]]

    mkt_ret6h_df  = ff.numba_pct_change(mkt_close.to_frame(), 6)
    mkt_ret6h = mkt_ret6h_df[mkt_ret6h_df.columns[0]]

    sma_df = ff.numba_rolling_mean(mkt_close.to_frame(), trend_sma_hours)
    sma = sma_df[sma_df.columns[0]]

    mkt_trend = (mkt_close / (sma + 1e-12) - 1.0)

    mkt_ret1h_df = ff.numba_pct_change(mkt_close.to_frame(), 1)
    mkt_ret1h = mkt_ret1h_df[mkt_ret1h_df.columns[0]]

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
    # Explicit float32 downcast to reduce memory and ensure consistency
    return mkt_df.astype(np.float32)

def add_regime_gates(mkt_df: pd.DataFrame, gate_vol_lookback_hours: int, gate_trend_thr: float):
    df = mkt_df.copy()
    # No look-ahead bias: Rolling median uses window ending at t.
    rv_med_df = ff.numba_rolling_median(df[["mkt_rv"]], gate_vol_lookback_hours)
    df["mkt_rv_med"] = rv_med_df["mkt_rv"]

    # Gates are calculated at time t using information up to t.
    df["G_VOL"] = (df["mkt_rv"] > df["mkt_rv_med"]).astype(np.int32)
    df["G_TREND"] = (df["mkt_ret24h"].abs() > gate_trend_thr).astype(np.int32)
    df["mkt_rv_ratio"] = df["mkt_rv"] / (df["mkt_rv_med"] + 1e-12)

    # Ensure float columns are float32
    float_cols = ["mkt_rv_med", "mkt_rv_ratio"]
    for c in float_cols:
        df[c] = df[c].astype(np.float32)

    return df

def compute_funding_proxy(panel, mkt_df):
    c = panel["close"]
    h = panel["high"]
    l = panel["low"]
    v = panel["volume"]

    c_ma = ff.numba_rolling_mean(c, 24)
    dist = (c / (c_ma + 1e-12)) - 1.0

    mkt_close_df = mkt_df[["mkt_close"]]
    mkt_ma_df = ff.numba_rolling_mean(mkt_close_df, 24)
    mkt_dist = (mkt_df["mkt_close"] / (mkt_ma_df["mkt_close"] + 1e-12)) - 1.0

    relative_premium = dist.sub(mkt_dist, axis=0)

    candle_pos = (c - l) / ((h - l) + 1e-9)
    vol_z = zscore_rolling(v, 24) # Uses numba now
    intensity = (candle_pos - 0.5) * vol_z

    return (relative_premium + (0.5 * intensity)).astype(np.float32)

def compute_features_hourly(panel, mkt_gates, cfg):
    tprint("Features: compute base matrices")
    o, h, l, c, v = panel["open"], panel["high"], panel["low"], panel["close"], panel["volume"]

    # Ensure all have same UTC index
    new_idx = ensure_utc(pd.DataFrame(index=c.index)).index
    o.index = new_idx
    h.index = new_idx
    l.index = new_idx
    c.index = new_idx
    v.index = new_idx

    # Align mkt_gates index to UTC
    if len(mkt_gates) == len(c):
        mkt_gates.index = new_idx
    else:
        mkt_gates = mkt_gates.reindex(new_idx)

    # Enforce float32 inputs
    o = o.astype(np.float32)
    h = h.astype(np.float32)
    l = l.astype(np.float32)
    c = c.astype(np.float32)
    v = v.astype(np.float32)

    feats = {}
    # Ret calc
    feats["ret1h"] = ff.numba_pct_change(c, 1).astype(np.float32)
    feats["ret6h"] = ff.numba_pct_change(c, 6).astype(np.float32)

    for H in [12,16,20,24,28]:
        feats[f"ret{H}h"] = ff.numba_pct_change(c, H).astype(np.float32)

    feats["range_pct"] = ((h - l) / (c + 1e-12)).astype(np.float32)
    feats["gap_pct"]   = ((o - c.shift(1)) / (c.shift(1) + 1e-12)).astype(np.float32)

    atr_base = atr_percent(h, l, c, n=cfg["atr_n"]) # Uses Numba
    feats["atr_pct_base"] = atr_base

    rsi_base = rsi(c, n=cfg["rsi_n"]) # Uses Numba
    feats["rsi_base"] = rsi_base
    feats["rsi_slope_base"] = rsi_base.diff(cfg["rsi_slope_n"]).astype(np.float32)

    feats["rv_24h"] = ff.apply_to_frame(feats["ret1h"], ff._numba_rolling_std_nan_safe, 24)

    feats["qv"] = (c * v).astype(np.float32)
    feats["vol_z24_base"] = zscore_rolling(v, 24) # Uses Numba
    feats["vol_z_base"]   = zscore_rolling(v, cfg["volz_n"]) # Uses Numba

    ema_fast_base = ema(c, cfg["ema_fast"]) # Uses Numba
    ema_slow_base = ema(c, cfg["ema_slow"]) # Uses Numba
    feats["dist_ema_fast_base"] = ((c / (ema_fast_base + 1e-12) - 1.0) / (atr_base + 1e-12)).astype(np.float32)
    feats["dist_ema_slow_base"] = ((c / (ema_slow_base + 1e-12) - 1.0) / (atr_base + 1e-12)).astype(np.float32)

    feats["roc_div"] = (feats["ret1h"] - feats["ret6h"]).astype(np.float32)
    feats["ret1h_z"] = (feats["ret1h"] / (feats["rv_24h"] + 1e-12)).astype(np.float32)

    body = (c - o).abs()
    upper_wick = (h - c.where(c >= o, o)).clip(lower=0)
    lower_wick = (c.where(c <= o, o) - l).clip(lower=0)
    feats["body_pct"] = (body / (c + 1e-12)).astype(np.float32)
    feats["wick_body_ratio"] = ((upper_wick + lower_wick) / (body + 1e-12)).astype(np.float32)

    feats["vol_price_spread"] = (v / ((h - l) + 1e-12)).astype(np.float32)

    prev_close = c.shift(1)

    tr_1 = (h - l)
    tr_2 = (h - prev_close).abs()
    tr_3 = (l - prev_close).abs()
    tr = np.maximum(tr_1, np.maximum(tr_2, tr_3))

    atr_tr = ff.apply_to_frame(tr, ff._numba_ewma, 1.0/cfg["atr_n"], False)
    feats["atr_expansion"] = (tr / (atr_tr + 1e-12)).astype(np.float32)

    sma_base = ff.apply_to_frame(c, ff._numba_rolling_mean_nan_safe, cfg["trend_sma_n"])
    feats["trend_pct_base"] = ((c / (sma_base + 1e-12)) - 1.0).astype(np.float32)

    hod = pd.Series(v.index.hour, index=v.index)
    rvol_denom = ff.numba_grouped_rolling_mean(v, hod, int(cfg["rvol_days"]*24))
    feats["rvol_hod_base"] = (v / (rvol_denom + 1e-12)).astype(np.float32)

    feats["funding_proxy"] = compute_funding_proxy(panel, mkt_gates)

    sin_hod, cos_hod, sin_dow, cos_dow = time_sin_cos(c.index)
    feats["sin_hod"] = pd.DataFrame(np.repeat(sin_hod[:,None], c.shape[1], axis=1), index=c.index, columns=c.columns).astype(np.float32)
    feats["cos_hod"] = pd.DataFrame(np.repeat(cos_hod[:,None], c.shape[1], axis=1), index=c.index, columns=c.columns).astype(np.float32)
    feats["sin_dow"] = pd.DataFrame(np.repeat(sin_dow[:,None], c.shape[1], axis=1), index=c.index, columns=c.columns).astype(np.float32)
    feats["cos_dow"] = pd.DataFrame(np.repeat(cos_dow[:,None], c.shape[1], axis=1), index=c.index, columns=c.columns).astype(np.float32)

    # --- EXISTING EXTRA FEATURES ---
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

    feats["v_power"] = (v / (c.diff().abs() + 1e-12)).astype(np.float32)
    feats["signed_vol"] = signed_vol.astype(np.float32)

    # --- NEW REQUESTED FEATURES ---

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

    log_v = np.log(v + 1.0)
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

    feats["vol_expansion_ratio"] = (atr_ema_f / (atr_ema_s + 1e-12)).astype(np.float32)

    sig_s = ff.apply_to_frame(feats["ret1h"], ff._numba_rolling_std_nan_safe, 6)
    sig_m = ff.apply_to_frame(feats["ret1h"], ff._numba_rolling_std_nan_safe, 18)
    feats["vol_compression"] = (sig_s / (sig_m + 1e-12)).astype(np.float32)

    rv_ratio = mkt_gates["mkt_rv_ratio"].reindex(c.index).astype(np.float32)
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
    trend_fast = (c / (sma_fast + 1e-12)) - 1.0
    trend_slow = (c / (sma_slow + 1e-12)) - 1.0
    feats["trend_pct"] = pick_by_rv(trend_fast, feats["trend_pct_base"], trend_slow)

    ema_fast_f = ema(c, max(4, int(cfg["ema_fast"] * 0.5)))
    ema_fast_s = ema(c, int(cfg["ema_fast"] * 2))
    dist_fast_f = (c / (ema_fast_f + 1e-12) - 1.0) / (feats["atr_pct"] + 1e-12)
    dist_fast_s = (c / (ema_fast_s + 1e-12) - 1.0) / (feats["atr_pct"] + 1e-12)
    feats["dist_ema_fast"] = pick_by_rv(dist_fast_f, feats["dist_ema_fast_base"], dist_fast_s)

    feats["vol_z24"] = feats["vol_z24_base"]
    feats["rsi_slope"] = feats["rsi"].diff(cfg["rsi_slope_n"]).astype(np.float32)
    feats["a_funding_proxy"] = feats["funding_proxy"]

    tprint("Features: Applying Causal Transforms (Log + Winsor + ZScore)")
    transformer = CausalFeatureTransformer(winsor_qt=0.02, roll_window=24*30)

    skip_transform = ["sin_hod", "cos_hod", "sin_dow", "cos_dow"]

    for k in feats.keys():
        if k in skip_transform:
            feats[k] = feats[k].astype(np.float32)
            continue
        try:
            feats[k] = transformer.transform(feats[k])
        except Exception as e:
            tprint(f"Warning: Transform failed for {k}: {e}")
            feats[k] = feats[k].astype(np.float32)

    tprint(f"Features: done ({len(feats)} keys)")
    return feats
