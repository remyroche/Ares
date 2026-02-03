import numpy as np
import pandas as pd
from extreme_price_movements.utils import tprint
from extreme_price_movements.feature_transforms import CausalFeatureTransformer
from extreme_price_movements.time_utils import ensure_utc

def zscore_rolling(x: pd.DataFrame, n: int):
    mu = x.rolling(n).mean()
    sd = x.rolling(n).std(ddof=0)
    return (x - mu) / (sd + 1e-12)

def rsi(close: pd.DataFrame, n: int):
    delta = close.diff()
    up = delta.clip(lower=0)
    dn = (-delta).clip(lower=0)
    rs = up.ewm(alpha=1/n, adjust=False).mean() / (dn.ewm(alpha=1/n, adjust=False).mean() + 1e-12)
    return 100 - (100 / (1 + rs))

def ema(x: pd.DataFrame, span: int):
    return x.ewm(span=span, adjust=False).mean()

def atr_percent(high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame, n: int):
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=0).groupby(level=0).max()
    atr = tr.ewm(alpha=1/n, adjust=False).mean()
    return atr / (close + 1e-12)

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

    mkt_ret24h = mkt_close.pct_change(24)
    mkt_ret6h  = mkt_close.pct_change(6)

    sma = mkt_close.rolling(trend_sma_hours).mean()
    mkt_trend = (mkt_close / (sma + 1e-12) - 1.0)

    mkt_ret1h = mkt_close.pct_change()
    mkt_rv = mkt_ret1h.rolling(24).std(ddof=0)

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
    return mkt_df

def add_regime_gates(mkt_df: pd.DataFrame, gate_vol_lookback_hours: int, gate_trend_thr: float):
    df = mkt_df.copy()
    df["mkt_rv_med"] = df["mkt_rv"].rolling(gate_vol_lookback_hours).median()
    df["G_VOL"] = (df["mkt_rv"] > df["mkt_rv_med"]).astype(int)
    df["G_TREND"] = (df["mkt_ret24h"].abs() > gate_trend_thr).astype(int)
    df["mkt_rv_ratio"] = df["mkt_rv"] / (df["mkt_rv_med"] + 1e-12)
    return df

def compute_funding_proxy(panel, mkt_df):
    c = panel["close"]
    h = panel["high"]
    l = panel["low"]
    v = panel["volume"]

    dist = (c / (c.rolling(24).mean() + 1e-12)) - 1.0
    mkt_dist = (mkt_df["mkt_close"] / (mkt_df["mkt_close"].rolling(24).mean() + 1e-12)) - 1.0
    relative_premium = dist.sub(mkt_dist, axis=0)

    candle_pos = (c - l) / ((h - l) + 1e-9)
    vol_mu = v.rolling(24).mean()
    vol_sd = v.rolling(24).std(ddof=0)
    vol_z = (v - vol_mu) / (vol_sd + 1e-12)
    intensity = (candle_pos - 0.5) * vol_z

    return (relative_premium + (0.5 * intensity)).astype(np.float32)

def compute_features_hourly(panel, mkt_gates, cfg):
    tprint("Features: compute base matrices")
    o, h, l, c, v = panel["open"], panel["high"], panel["low"], panel["close"], panel["volume"]
    c.index = ensure_utc(pd.DataFrame(index=c.index)).index

    feats = {}
    feats["ret1h"]   = c.pct_change()
    feats["ret6h"]   = c.pct_change(6)
    for H in [12,16,20,24,28]:
        feats[f"ret{H}h"] = c.pct_change(H)

    feats["range_pct"] = (h - l) / (c + 1e-12)
    feats["gap_pct"]   = (o - c.shift(1)) / (c.shift(1) + 1e-12)

    atr_base = atr_percent(h, l, c, n=cfg["atr_n"])
    feats["atr_pct_base"] = atr_base

    rsi_base = rsi(c, n=cfg["rsi_n"])
    feats["rsi_base"] = rsi_base
    feats["rsi_slope_base"] = rsi_base.diff(cfg["rsi_slope_n"])

    feats["rv_24h"] = feats["ret1h"].rolling(24).std(ddof=0)

    feats["qv"] = (c * v)
    feats["vol_z24_base"] = zscore_rolling(v, 24)
    feats["vol_z_base"]   = zscore_rolling(v, cfg["volz_n"])

    ema_fast_base = ema(c, cfg["ema_fast"])
    ema_slow_base = ema(c, cfg["ema_slow"])
    feats["dist_ema_fast_base"] = (c / (ema_fast_base + 1e-12) - 1.0) / (atr_base + 1e-12)
    feats["dist_ema_slow_base"] = (c / (ema_slow_base + 1e-12) - 1.0) / (atr_base + 1e-12)

    feats["roc_div"] = feats["ret1h"] - feats["ret6h"]
    feats["ret1h_z"] = feats["ret1h"] / (feats["rv_24h"] + 1e-12)

    body = (c - o).abs()
    upper_wick = (h - c.where(c >= o, o)).clip(lower=0)
    lower_wick = (c.where(c <= o, o) - l).clip(lower=0)
    feats["body_pct"] = body / (c + 1e-12)
    feats["wick_body_ratio"] = (upper_wick + lower_wick) / (body + 1e-12)

    feats["vol_price_spread"] = v / ((h - l) + 1e-12)

    prev_close = c.shift(1)
    tr = pd.concat([(h - l), (h - prev_close).abs(), (l - prev_close).abs()], axis=0).groupby(level=0).max()
    atr_tr = tr.ewm(alpha=1/cfg["atr_n"], adjust=False).mean()
    feats["atr_expansion"] = tr / (atr_tr + 1e-12)

    sma_base = c.rolling(cfg["trend_sma_n"]).mean()
    feats["trend_pct_base"] = (c / (sma_base + 1e-12)) - 1.0

    hod = pd.Series(v.index.hour, index=v.index)
    feats["rvol_hod_base"] = v / (v.groupby(hod).transform(lambda s: s.rolling(cfg["rvol_days"]*24, min_periods=24).mean()) + 1e-12)

    feats["funding_proxy"] = compute_funding_proxy(panel, mkt_gates)

    sin_hod, cos_hod, sin_dow, cos_dow = time_sin_cos(c.index)
    feats["sin_hod"] = pd.DataFrame(np.repeat(sin_hod[:,None], c.shape[1], axis=1), index=c.index, columns=c.columns).astype(np.float32)
    feats["cos_hod"] = pd.DataFrame(np.repeat(cos_hod[:,None], c.shape[1], axis=1), index=c.index, columns=c.columns).astype(np.float32)
    feats["sin_dow"] = pd.DataFrame(np.repeat(sin_dow[:,None], c.shape[1], axis=1), index=c.index, columns=c.columns).astype(np.float32)
    feats["cos_dow"] = pd.DataFrame(np.repeat(cos_dow[:,None], c.shape[1], axis=1), index=c.index, columns=c.columns).astype(np.float32)

    # --- EXISTING EXTRA FEATURES ---
    signed_vol = v * np.sign(c - o)
    sv_abs = signed_vol.abs()
    ewma_sv_fast = signed_vol.ewm(span=6, adjust=False).mean()
    ewma_sv_slow = sv_abs.ewm(span=24, adjust=False).mean()
    feats["flow_persistence"] = ewma_sv_fast / (ewma_sv_slow + 1e-12)
    feats["flow_ratio"] = feats["flow_persistence"]

    eff = (c - o).abs() / ((h - l) + 1e-9)
    feats["efficiency"] = eff.rolling(12).mean()

    skew_ser = feats["ret1h"].skew(axis=1)
    feats["skew"] = pd.DataFrame(np.repeat(skew_ser.values[:,None], c.shape[1], axis=1), index=c.index, columns=c.columns).astype(np.float32)

    r = feats["ret1h"]
    r2 = r**2
    up_sq = r2.where(r > 0, 0.0)
    dn_sq = r2.where(r < 0, 0.0)
    up_vol = up_sq.ewm(span=24, adjust=False).mean()
    dn_vol = dn_sq.ewm(span=24, adjust=False).mean()
    feats["up_vol"] = up_vol
    feats["dn_vol"] = dn_vol
    feats["vol_asym"] = up_vol - dn_vol

    l_prev2 = l.shift(2)
    h_prev2 = h.shift(2)
    fvg_bull = (l_prev2 - h).clip(lower=0) / (c + 1e-12)
    fvg_bear = (l - h_prev2).clip(lower=0) / (c + 1e-12)
    feats["fvg"] = fvg_bull - fvg_bear

    feats["churn"] = v / ((c - o).abs() + 1e-12)
    feats["slope"] = (ema_fast_base - ema_slow_base) / (atr_base + 1e-12)
    feats["trend_snr"] = feats["ret1h"].ewm(span=6, adjust=False).mean().abs() / (feats["ret1h"].rolling(24).std() + 1e-12)
    feats["v_power"] = v / (c.diff().abs() + 1e-12)
    feats["signed_vol"] = signed_vol

    # --- NEW REQUESTED FEATURES ---

    atr_ema_f = atr_base.ewm(span=6, adjust=False).mean()
    atr_ema_s = atr_base.ewm(span=24, adjust=False).mean()
    feats["atr_slope"] = (atr_ema_f - atr_ema_s) / (atr_ema_s + 1e-12)

    pv = c * v
    sum_pv = pv.rolling(24).sum()
    sum_v = v.rolling(24).sum()
    vwap_24 = sum_pv / (sum_v + 1e-12)
    feats["dist_vwap_norm"] = (c - vwap_24) / (atr_base + 1e-12)

    feats["momentum_accel"] = feats["ret1h"].diff()

    log_v = np.log(v + 1.0)
    mu_lv = log_v.rolling(cfg["volz_n"]).mean()
    sd_lv = log_v.rolling(cfg["volz_n"]).std(ddof=0)
    feats["rvol_z"] = (log_v - mu_lv) / (sd_lv + 1e-12)

    vr = v * feats["ret1h"].abs()
    ema_vr = vr.ewm(span=24, adjust=False).mean()
    feats["vol_range_shock"] = vr / (ema_vr + 1e-12)

    v_max = v.rolling(24).max()
    feats["climax_decay"] = v_max / (v + 1e-12)

    cum_sv = signed_vol.rolling(24).sum()
    feats["cumulative_delta_stall"] = c.rolling(24).corr(cum_sv).fillna(0)

    feats["vol_expansion_ratio"] = atr_ema_f / (atr_ema_s + 1e-12)

    sig_s = feats["ret1h"].rolling(6).std()
    sig_m = feats["ret1h"].rolling(18).std()
    feats["vol_compression"] = sig_s / (sig_m + 1e-12)

    # Adaptive Windows
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

    sma_fast = c.rolling(max(24, int(cfg["trend_sma_n"] * 0.5))).mean()
    sma_slow = c.rolling(int(cfg["trend_sma_n"] * 2)).mean()
    trend_fast = (c / (sma_fast + 1e-12)) - 1.0
    trend_slow = (c / (sma_slow + 1e-12)) - 1.0
    feats["trend_pct"] = pick_by_rv(trend_fast, feats["trend_pct_base"], trend_slow)

    ema_fast_f = ema(c, max(4, int(cfg["ema_fast"] * 0.5)))
    ema_fast_s = ema(c, int(cfg["ema_fast"] * 2))
    dist_fast_f = (c / (ema_fast_f + 1e-12) - 1.0) / (feats["atr_pct"] + 1e-12)
    dist_fast_s = (c / (ema_fast_s + 1e-12) - 1.0) / (feats["atr_pct"] + 1e-12)
    feats["dist_ema_fast"] = pick_by_rv(dist_fast_f, feats["dist_ema_fast_base"], dist_fast_s)

    feats["vol_z24"] = feats["vol_z24_base"]
    feats["rsi_slope"] = feats["rsi"].diff(cfg["rsi_slope_n"])
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

class StatefulFeatureCalculator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.buffer_panel = None
        # Max lookback: need enough for rolling windows.
        # trend_sma_n = 24*14 ~ 336
        # rvol_days = 14 * 24 = 336
        # train_lookback = 24*30 = 720
        # Safe buffer: 24*45
        self.max_lookback = 24 * 45

    def update(self, new_panel):
        """
        Updates the internal buffer with new_panel (which might be just 1 row or a chunk)
        and returns the features corresponding to the *new* rows only (or the whole buffer features if needed).
        Currently we return the *last* row features if new_panel is 1 row.
        """
        if self.buffer_panel is None:
            self.buffer_panel = new_panel
        else:
            # Append per column
            for k in self.buffer_panel:
                if k in new_panel:
                    combined = pd.concat([self.buffer_panel[k], new_panel[k]])
                    combined = combined[~combined.index.duplicated(keep='last')]
                    combined = combined.sort_index()
                    if len(combined) > self.max_lookback:
                        combined = combined.iloc[-self.max_lookback:]
                    self.buffer_panel[k] = combined

        # Compute features on the buffer
        # This is not O(1), it's O(buffer_size).
        # But buffer_size (1000) << full history (24000).

        mkt_df = compute_market_features(self.buffer_panel, self.cfg["market_basket"])
        mkt_gates = add_regime_gates(mkt_df, self.cfg["gate_vol_lookback_hours"], self.cfg["gate_trend_thr"])

        # We only need features for the new timestamps.
        # compute_features_hourly transforms everything.
        # We can optimize compute_features_hourly to take start_idx?
        # But rolling needs previous data.
        # So we compute all, then slice.

        feats = compute_features_hourly(self.buffer_panel, mkt_gates, self.cfg)

        return feats, mkt_gates
