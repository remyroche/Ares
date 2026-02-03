import numpy as np
import pandas as pd
from utils import tprint

def zscore(x: pd.DataFrame, n: int):
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
    # (3) time encoding: hour-of-day + day-of-week
    hod = index.hour.to_numpy()
    dow = index.dayofweek.to_numpy()
    sin_hod = np.sin(2*np.pi*hod/24.0)
    cos_hod = np.cos(2*np.pi*hod/24.0)
    sin_dow = np.sin(2*np.pi*dow/7.0)
    cos_dow = np.cos(2*np.pi*dow/7.0)
    return sin_hod, cos_hod, sin_dow, cos_dow

def compute_market_features(panel, basket_syms, trend_sma_hours):
    c = panel["close"]
    h = panel["high"]
    l = panel["low"]
    v = panel["volume"]

    basket = [s for s in basket_syms if s in c.columns]
    if not basket:
        raise ValueError("Market basket symbols missing.")

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

    # (4) volatility ratio used for adaptive window selection
    df["mkt_rv_ratio"] = df["mkt_rv"] / (df["mkt_rv_med"] + 1e-12)
    return df

def compute_funding_proxy(panel, mkt_df):
    """
    (5) Funding proxy computed vectorized.
    Uses per-symbol OHLCV wide matrices and market composite OHLCV from mkt_df.
    """
    c = panel["close"]
    h = panel["high"]
    l = panel["low"]
    v = panel["volume"]

    # 1) relative premium vs market (distance from 24h mean, standardized by subtraction)
    dist = (c / (c.rolling(24).mean() + 1e-12)) - 1.0
    mkt_dist = (mkt_df["mkt_close"] / (mkt_df["mkt_close"].rolling(24).mean() + 1e-12)) - 1.0
    relative_premium = dist.sub(mkt_dist, axis=0)

    # 2) buying intensity: candle position * vol z
    candle_pos = (c - l) / ((h - l) + 1e-9)
    vol_mu = v.rolling(24).mean()
    vol_sd = v.rolling(24).std(ddof=0)
    vol_z = (v - vol_mu) / (vol_sd + 1e-12)
    intensity = (candle_pos - 0.5) * vol_z

    return (relative_premium + (0.5 * intensity)).astype(np.float32)

def compute_features_hourly(panel, mkt_gates, cfg):
    """
    Vectorized wide-matrix features.
    Adds:
      - multiple ret horizons (1): 12/16/20/24/28
      - time sin/cos (3)
      - funding proxy (5)
      - adaptive fast/base/slow windows selection (4) for selected features
    """
    tprint("Features: compute base matrices")
    o, h, l, c, v = panel["open"], panel["high"], panel["low"], panel["close"], panel["volume"]

    feats = {}
    # returns
    feats["ret1h"]   = c.pct_change()
    feats["ret6h"]   = c.pct_change(6)
    for H in [12,16,20,24,28]:
        feats[f"ret{H}h"] = c.pct_change(H)

    # range/gap
    feats["range_pct"] = (h - l) / (c + 1e-12)
    feats["gap_pct"]   = (o - c.shift(1)) / (c.shift(1) + 1e-12)

    # base ATR/RSI etc
    atr_base = atr_percent(h, l, c, n=cfg["atr_n"])
    feats["atr_pct_base"] = atr_base

    rsi_base = rsi(c, n=cfg["rsi_n"])
    feats["rsi_base"] = rsi_base
    feats["rsi_slope_base"] = rsi_base.diff(cfg["rsi_slope_n"])

    # volatility
    feats["rv_24h"] = feats["ret1h"].rolling(24).std(ddof=0)

    # volume features
    feats["qv"] = (c * v)
    feats["vol_z24_base"] = zscore(v, 24)
    feats["vol_z_base"]   = zscore(v, cfg["volz_n"])

    # EMA distances
    ema_fast_base = ema(c, cfg["ema_fast"])
    ema_slow_base = ema(c, cfg["ema_slow"])
    feats["dist_ema_fast_base"] = (c / (ema_fast_base + 1e-12) - 1.0) / (atr_base + 1e-12)
    feats["dist_ema_slow_base"] = (c / (ema_slow_base + 1e-12) - 1.0) / (atr_base + 1e-12)

    # ROC divergence + z-score
    feats["roc_div"] = feats["ret1h"] - feats["ret6h"]
    feats["ret1h_z"] = feats["ret1h"] / (feats["rv_24h"] + 1e-12)

    # candle anatomy
    body = (c - o).abs()
    upper_wick = (h - c.where(c >= o, o)).clip(lower=0)
    lower_wick = (c.where(c <= o, o) - l).clip(lower=0)
    feats["body_pct"] = body / (c + 1e-12)
    feats["wick_body_ratio"] = (upper_wick + lower_wick) / (body + 1e-12)

    feats["vol_price_spread"] = v / ((h - l) + 1e-12)

    # ATR expansion
    prev_close = c.shift(1)
    tr = pd.concat([(h - l), (h - prev_close).abs(), (l - prev_close).abs()], axis=0).groupby(level=0).max()
    atr_tr = tr.ewm(alpha=1/cfg["atr_n"], adjust=False).mean()
    feats["atr_expansion"] = tr / (atr_tr + 1e-12)

    # trend pct (SMA)
    sma_base = c.rolling(cfg["trend_sma_n"]).mean()
    feats["trend_pct_base"] = (c / (sma_base + 1e-12)) - 1.0

    # RVOL (hour-of-day)
    hod = pd.Series(v.index.hour, index=v.index)
    feats["rvol_hod_base"] = v / (v.groupby(hod).transform(lambda s: s.rolling(cfg["rvol_days"]*24, min_periods=24).mean()) + 1e-12)

    # funding proxy (vectorized)
    feats["funding_proxy"] = compute_funding_proxy(panel, mkt_gates)

    # (3) time sin/cos features (broadcast across symbols)
    sin_hod, cos_hod, sin_dow, cos_dow = time_sin_cos(c.index)
    feats["sin_hod"] = pd.DataFrame(np.repeat(sin_hod[:,None], c.shape[1], axis=1), index=c.index, columns=c.columns).astype(np.float32)
    feats["cos_hod"] = pd.DataFrame(np.repeat(cos_hod[:,None], c.shape[1], axis=1), index=c.index, columns=c.columns).astype(np.float32)
    feats["sin_dow"] = pd.DataFrame(np.repeat(sin_dow[:,None], c.shape[1], axis=1), index=c.index, columns=c.columns).astype(np.float32)
    feats["cos_dow"] = pd.DataFrame(np.repeat(cos_dow[:,None], c.shape[1], axis=1), index=c.index, columns=c.columns).astype(np.float32)

    # (4) Vol-adjusted lookbacks (vectorized selection among fast/base/slow)
    # Compute fast/slow variants, then select per timestamp using mkt_rv_ratio buckets.
    rv_ratio = mkt_gates["mkt_rv_ratio"].reindex(c.index).astype(np.float32)

    def pick_by_rv(fast_df, base_df, slow_df):
        # broadcast rv_ratio to columns
        rr = pd.DataFrame(np.repeat(rv_ratio.to_numpy()[:,None], base_df.shape[1], axis=1),
                          index=base_df.index, columns=base_df.columns).astype(np.float32)
        out = base_df.copy()
        out = out.where(~(rr > cfg["rv_ratio_fast_thr"]), fast_df)
        out = out.where(~(rr < cfg["rv_ratio_slow_thr"]), slow_df)
        return out.astype(np.float32)

    # rsi variants
    rsi_fast = rsi(c, max(2, int(cfg["rsi_n"] * 0.5)))
    rsi_slow = rsi(c, int(cfg["rsi_n"] * 2))
    feats["rsi"] = pick_by_rv(rsi_fast, rsi_base, rsi_slow)

    # atr variants
    atr_fast = atr_percent(h, l, c, max(2, int(cfg["atr_n"] * 0.5)))
    atr_slow = atr_percent(h, l, c, int(cfg["atr_n"] * 2))
    feats["atr_pct"] = pick_by_rv(atr_fast, atr_base, atr_slow)

    # vol_z variants
    volz_fast = zscore(v, max(24, int(cfg["volz_n"] * 0.5)))
    volz_slow = zscore(v, int(cfg["volz_n"] * 2))
    feats["vol_z"] = pick_by_rv(volz_fast, feats["vol_z_base"], volz_slow)

    # trend variants (SMA)
    sma_fast = c.rolling(max(24, int(cfg["trend_sma_n"] * 0.5))).mean()
    sma_slow = c.rolling(int(cfg["trend_sma_n"] * 2)).mean()
    trend_fast = (c / (sma_fast + 1e-12)) - 1.0
    trend_slow = (c / (sma_slow + 1e-12)) - 1.0
    feats["trend_pct"] = pick_by_rv(trend_fast, feats["trend_pct_base"], trend_slow)

    # EMA distance variants (fast span changes)
    ema_fast_f = ema(c, max(4, int(cfg["ema_fast"] * 0.5)))
    ema_fast_s = ema(c, int(cfg["ema_fast"] * 2))
    dist_fast_f = (c / (ema_fast_f + 1e-12) - 1.0) / (feats["atr_pct"] + 1e-12)
    dist_fast_s = (c / (ema_fast_s + 1e-12) - 1.0) / (feats["atr_pct"] + 1e-12)
    feats["dist_ema_fast"] = pick_by_rv(dist_fast_f, feats["dist_ema_fast_base"], dist_fast_s)

    # Keep base variants for debugging if desired
    feats["vol_z24"] = feats["vol_z24_base"]
    feats["rsi_slope"] = feats["rsi"].diff(cfg["rsi_slope_n"])
    feats["a_funding_proxy"] = feats["funding_proxy"]

    # downcast everything to float32
    for k in list(feats.keys()):
        feats[k] = feats[k].astype(np.float32)

    tprint(f"Features: done ({len(feats)} keys)")
    return feats
