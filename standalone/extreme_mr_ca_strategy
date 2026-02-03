"""
FULL UPDATED UNIFIED SCRIPT (Parquet persistence + incremental fetch + float32 downcast + 7-day API chunks)

Binance Spot (hourly) + daily Cross-Margin universe filter + MR/TF ElasticNet regression blend
+ RuleCleaner (corr > 0.8, keep by |coef|) + Peak Exhaustion Score (ML, trained hourly on rolling window)
+ TP/SL monitored on hourly bars (simulation)

Key requirements satisfied:
1/ Keep trend monitoring vs mean reversion and long vs short logic (MR + TF + regime mixing).
2/ No leakage/lookahead (strict training windows; features at ts_sig; labels strictly future).
3/ No VWAP entry (entry is next-hour open only).
4/ Hourly monitoring (TP/SL on hourly bars; sim uses hourly bars).
5/ margin_spot_symbols refreshed daily (refresh_margin_universe_daily).
6/ RuleCleaner to remove correlated features >0.8 using ElasticNet coef magnitude.
7/ API limit: OHLCV fetch split into 7-day chunks (hard enforced).
8/ Peak exhaustion score driven by ML on same datasets, trained hourly on rolling window.
   (Implemented as logistic regression ElasticNet; retrained each decision point using last N hours of data.)
+ Data persistence layer saves OHLCV to Parquet and only fetches new hourly bars on subsequent runs.
+ Downcast to 32-bit, save on disk, chunk fetch; vectorize core operations (wide matrices).

Dependencies:
  pip install ccxt pandas numpy requests scikit-learn pyarrow
"""

import os
import time
import math
import requests
import numpy as np
import pandas as pd
import ccxt

from dataclasses import dataclass
from collections import deque

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet, LogisticRegression


# =========================
# Binance Margin universe (Cross Margin)
# =========================

BINANCE_API = "https://api.binance.com"

def fetch_binance_cross_margin_pairs():
    """GET /sapi/v1/margin/allPairs (MARKET_DATA). No signature required."""
    r = requests.get(f"{BINANCE_API}/sapi/v1/margin/allPairs", timeout=30)
    r.raise_for_status()
    return r.json()

def margin_pairs_to_spot_symbols(margin_pairs_json, quote="USDT"):
    """Convert Binance symbols (e.g., BTCUSDT) -> ccxt (BTC/USDT). Keep only quote=USDT."""
    out = set()
    for row in margin_pairs_json:
        s = row.get("symbol", "")
        if not s.endswith(quote):
            continue
        base = s[:-len(quote)]
        if base:
            out.add(f"{base}/{quote}")
    return sorted(out)

@dataclass
class MarginUniverseCache:
    symbols: list[str]
    asof_day: pd.Timestamp

def refresh_margin_universe_daily(cache: MarginUniverseCache | None, quote="USDT") -> MarginUniverseCache:
    """
    (5) Refresh margin universe daily.
    Call at startup and once per UTC day in live mode.
    """
    today = pd.Timestamp.utcnow().tz_localize("UTC").floor("D")
    if cache is not None and cache.asof_day == today:
        return cache
    pairs = fetch_binance_cross_margin_pairs()
    syms = margin_pairs_to_spot_symbols(pairs, quote=quote)
    return MarginUniverseCache(symbols=syms, asof_day=today)


# =========================
# Exchange
# =========================

def make_spot_exchange():
    ex = ccxt.binance({"enableRateLimit": True})
    ex.load_markets()
    return ex


# =========================
# Data fetch — forced 7-day chunks
# =========================

def _fetch_ohlcv_paged(exchange, symbol, since_ms, until_ms, timeframe="1h", limit=1000):
    """
    Fetch with ccxt paging but bounded by [since_ms, until_ms).
    """
    out = []
    since = since_ms
    while True:
        batch = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
        if not batch:
            break

        for row in batch:
            ts = row[0]
            if ts < since_ms:
                continue
            if ts >= until_ms:
                break
            out.append(row)

        last = batch[-1][0]
        if last >= until_ms - 1:
            break

        since = last + 1
        if len(batch) < limit:
            break
        time.sleep(exchange.rateLimit / 1000)

    if not out:
        return pd.DataFrame(columns=["ts","open","high","low","close","volume"]).set_index(
            pd.DatetimeIndex([], tz="UTC", name="ts")
        )

    df = pd.DataFrame(out, columns=["ts","open","high","low","close","volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.drop_duplicates("ts").set_index("ts").sort_index()
    return df

def fetch_ohlcv_all_7d_chunks(exchange, symbol, since_ms, timeframe="1h", limit=1000):
    """
    (7) API limit: split fetches into 7-day chunks (hard enforced).
    """
    chunk_ms = int(pd.Timedelta(days=7).total_seconds() * 1000)
    now_ms = int(pd.Timestamp.utcnow().value // 10**6)

    dfs = []
    start = since_ms
    while start < now_ms:
        end = min(start + chunk_ms, now_ms)
        df = _fetch_ohlcv_paged(exchange, symbol, start, end, timeframe=timeframe, limit=limit)
        if len(df):
            dfs.append(df)
        start = end
        time.sleep(exchange.rateLimit / 1000)

    if not dfs:
        return pd.DataFrame(columns=["open","high","low","close","volume"]).set_index(
            pd.DatetimeIndex([], tz="UTC", name="ts")
        )

    out = pd.concat(dfs).sort_index()
    out = out[~out.index.duplicated(keep="last")]
    return out


# =========================
# Parquet persistence layer (per-symbol)
# =========================

class OHLCVStore:
    """
    Saves per-symbol OHLCV to Parquet and only fetches new hourly bars on subsequent runs.
    Downcasts OHLCV to float32 for memory/disk.
    """

    def __init__(self, root_dir="data", timeframe="1h"):
        self.root_dir = root_dir
        self.timeframe = timeframe
        self.ohlcv_dir = os.path.join(root_dir, "ohlcv")
        os.makedirs(self.ohlcv_dir, exist_ok=True)

    def _sym_path(self, symbol: str) -> str:
        safe = symbol.replace("/", "_")
        return os.path.join(self.ohlcv_dir, f"{safe}.parquet")

    def _downcast(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        out = df.copy()
        for col in ["open","high","low","close","volume"]:
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors="coerce").astype(np.float32)
        return out

    def load(self, symbol: str) -> pd.DataFrame:
        path = self._sym_path(symbol)
        if not os.path.exists(path):
            return pd.DataFrame(columns=["open","high","low","close","volume"]).set_index(
                pd.DatetimeIndex([], tz="UTC", name="ts")
            )
        df = pd.read_parquet(path)
        if "ts" in df.columns:
            df["ts"] = pd.to_datetime(df["ts"], utc=True)
            df = df.set_index("ts")
        df = df.sort_index()
        return self._downcast(df)

    def save(self, symbol: str, df: pd.DataFrame) -> None:
        if df.empty:
            return
        df = df.sort_index()
        df = self._downcast(df)
        out = df.reset_index()
        # ensure column name "ts"
        if out.columns[0] != "ts":
            out = out.rename(columns={out.columns[0]: "ts"})
        out.to_parquet(self._sym_path(symbol), index=False)

    def update_symbol(self, exchange, symbol: str, since_ms: int) -> pd.DataFrame:
        """
        Load existing parquet; fetch only missing bars (still using 7d chunks); append; dedupe; save.
        """
        existing = self.load(symbol)
        if existing.empty:
            fresh = fetch_ohlcv_all_7d_chunks(exchange, symbol, since_ms, timeframe=self.timeframe, limit=1000)
            fresh = self._downcast(fresh)
            self.save(symbol, fresh)
            return fresh

        last_ts = existing.index.max()
        next_ts = last_ts + pd.Timedelta(hours=1)
        next_ms = int(next_ts.value // 10**6)

        now_ms = int(pd.Timestamp.utcnow().value // 10**6)
        if next_ms >= now_ms:
            return existing

        fresh = fetch_ohlcv_all_7d_chunks(exchange, symbol, next_ms, timeframe=self.timeframe, limit=1000)
        if fresh is None or fresh.empty:
            return existing

        fresh = self._downcast(fresh)
        merged = pd.concat([existing, fresh]).sort_index()
        merged = merged[~merged.index.duplicated(keep="last")]

        self.save(symbol, merged)
        return merged


# =========================
# Panel builder (wide matrices)
# =========================

def to_panel(dfs_by_symbol: dict[str, pd.DataFrame]):
    keys = ["open","high","low","close","volume"]
    panel = {}
    for k in keys:
        panel[k] = pd.concat([df[k].rename(sym) for sym, df in dfs_by_symbol.items()], axis=1).sort_index()
    return panel

def downcast_panel_float32(panel: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    for k in panel:
        panel[k] = panel[k].astype(np.float32)
    return panel

def assert_basket_present(panel_close: pd.DataFrame, market_basket: list[str]):
    missing = [s for s in market_basket if s not in panel_close.columns]
    if missing:
        raise ValueError(f"Market basket symbols missing from fetched data: {missing}")


# =========================
# Feature engineering (hourly) — strictly causal
# =========================

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

def atr_percent(high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame, n: int):
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=0).groupby(level=0).max()
    atr = tr.ewm(alpha=1/n, adjust=False).mean()
    return atr / (close + 1e-12)

def compute_features_hourly(panel, cfg):
    o, h, l, c, v = panel["open"], panel["high"], panel["low"], panel["close"], panel["volume"]
    feats = {}

    feats["ret1h"]   = c.pct_change()
    feats["ret6h"]   = c.pct_change(6)
    feats["ret24h"]  = c.pct_change(24)

    feats["range_pct"] = (h - l) / (c + 1e-12)
    feats["gap_pct"]   = (o - c.shift(1)) / (c.shift(1) + 1e-12)
    feats["atr_pct"]   = atr_percent(h, l, c, n=cfg["atr_n"])

    feats["rsi"]     = rsi(c, n=cfg["rsi_n"])
    feats["qv"]      = (c * v)  # quote-volume proxy
    feats["vol_z"]   = zscore(v, n=cfg["volz_n"])

    sma = c.rolling(cfg["trend_sma_n"]).mean()
    feats["trend_pct"] = (c / (sma + 1e-12)) - 1.0

    feats["rv_24h"] = feats["ret1h"].rolling(24).std(ddof=0)
    feats["body_pct"] = (c - o).abs() / (c + 1e-12)

    return feats


# =========================
# Market features + regime gates
# =========================

def compute_market_features(panel_close, feats_ret1h, basket_syms, trend_sma_hours):
    basket = [s for s in basket_syms if s in panel_close.columns]
    if not basket:
        raise ValueError("Market basket symbols missing from data columns.")

    mkt_ret24h = panel_close[basket].pct_change(24).mean(axis=1)
    mkt_ret6h  = panel_close[basket].pct_change(6).mean(axis=1)

    sma = panel_close[basket].rolling(trend_sma_hours).mean()
    mkt_trend = (panel_close[basket] / (sma + 1e-12) - 1.0).mean(axis=1)

    mkt_rv = feats_ret1h[basket].mean(axis=1).rolling(24).std(ddof=0)

    return pd.DataFrame({
        "mkt_ret24h": mkt_ret24h,
        "mkt_ret6h": mkt_ret6h,
        "mkt_trend": mkt_trend,
        "mkt_rv": mkt_rv
    })

def add_regime_gates(mkt_df: pd.DataFrame, cfg):
    df = mkt_df.copy()
    df["mkt_rv_med"] = df["mkt_rv"].rolling(cfg["gate_vol_lookback_hours"]).median()
    df["G_VOL"] = (df["mkt_rv"] > df["mkt_rv_med"]).astype(int)
    df["G_TREND"] = (df["mkt_ret24h"].abs() > cfg["gate_trend_thr"]).astype(int)
    return df

def apply_interaction_toggles(df: pd.DataFrame, causal_cols, gate_cols, drop_raw=True):
    out = df.copy()
    for g in gate_cols:
        for col in causal_cols:
            out[f"{col}_{g}_0"] = out[col] * (1 - out[g])
            out[f"{col}_{g}_1"] = out[col] * out[g]
    if drop_raw:
        out = out.drop(columns=list(causal_cols), errors="ignore")
    return out


# =========================
# Training universe selection
# =========================

def select_extreme_movers(ret_series: pd.Series, pct: float = 0.05, min_n: int = 5, max_n: int = 20):
    r = ret_series.dropna().sort_values()
    if len(r) == 0:
        return [], []

    neg = r[r < 0]
    pos = r[r > 0]

    top = []
    bot = []

    if len(pos) > 0:
        pos_cut = pos.quantile(1 - pct)
        top = pos[pos >= pos_cut].sort_values(ascending=False).index.tolist()
    if len(neg) > 0:
        neg_cut = neg.quantile(pct)
        bot = neg[neg <= neg_cut].sort_values(ascending=True).index.tolist()

    top = top[:max_n]
    bot = bot[:max_n]

    if len(top) < min_n and len(pos) >= min_n:
        top = pos.sort_values(ascending=False).head(min_n).index.tolist()
    if len(bot) < min_n and len(neg) >= min_n:
        bot = neg.sort_values(ascending=True).head(min_n).index.tolist()

    return top, bot


# =========================
# Execution
# =========================

def entry_price_next_hour_open(panel_open, ts_entry, symbol):
    try:
        px = panel_open.loc[ts_entry, symbol]
        return float(px) if pd.notna(px) and px > 0 else np.nan
    except Exception:
        return np.nan

def simulate_trade_hourly(o_s, h_s, l_s, c_s, entry_ts, entry_px, side, tp, sl, max_hold_hours):
    if np.isnan(entry_px) or entry_px <= 0:
        return 0.0, entry_ts, "no_entry"

    if side == "long":
        tp_px = entry_px * (1 + tp)
        sl_px = entry_px * (1 - sl)
    else:
        tp_px = entry_px * (1 - tp)
        sl_px = entry_px * (1 + sl)

    end_ts = entry_ts + pd.Timedelta(hours=max_hold_hours)
    path = o_s.loc[entry_ts:end_ts].index
    if len(path) == 0:
        return 0.0, entry_ts, "no_path"

    for ts in path:
        hh = h_s.loc[ts]; ll = l_s.loc[ts]; cc = c_s.loc[ts]
        if np.isnan(hh) or np.isnan(ll) or np.isnan(cc):
            continue

        if side == "long":
            hit_tp = hh >= tp_px
            hit_sl = ll <= sl_px
            if hit_tp and hit_sl:
                return (sl_px / entry_px) - 1.0, ts, "sl_same_hour"
            if hit_tp:
                return (tp_px / entry_px) - 1.0, ts, "tp"
            if hit_sl:
                return (sl_px / entry_px) - 1.0, ts, "sl"
        else:
            hit_tp = ll <= tp_px
            hit_sl = hh >= sl_px
            if hit_tp and hit_sl:
                return (entry_px / sl_px) - 1.0, ts, "sl_same_hour"
            if hit_tp:
                return (entry_px / tp_px) - 1.0, ts, "tp"
            if hit_sl:
                return (entry_px / sl_px) - 1.0, ts, "sl"

    last_ts = path[-1]
    last_close = c_s.loc[last_ts]
    if side == "long":
        return (last_close / entry_px) - 1.0, last_ts, "time_exit"
    else:
        return (entry_px / last_close) - 1.0, last_ts, "time_exit"


# =========================
# Models
# =========================

def make_elasticnet_reg(alpha=1e-3, l1_ratio=0.2):
    return Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("reg", ElasticNet(
            alpha=float(alpha),
            l1_ratio=float(l1_ratio),
            fit_intercept=True,
            max_iter=5000,
            random_state=42
        ))
    ])

def make_exhaustion_model(C=1.0, l1_ratio=0.3):
    """
    Peak exhaustion model: logistic regression with elastic-net penalty.
    Trained hourly on a rolling window of hourly samples.
    """
    return Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("clf", LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            l1_ratio=float(l1_ratio),
            C=float(C),
            max_iter=2000,
            random_state=42
        ))
    ])

def map_pred_to_score(pred_ret, mode="tanh", scale=10.0):
    x = pred_ret * scale
    if mode == "tanh":
        return float(np.tanh(max(0.0, x)))  # [0,1)
    if mode == "relu":
        return float(max(0.0, x))
    raise ValueError("mode must be 'tanh' or 'relu'")

def mix_predictions(mr_pred, tf_pred, G_VOL, G_TREND, cfg):
    w_mr = cfg["mix_base_mr"] + cfg["mix_add_mr_on_vol"] * int(G_VOL)
    w_tf = cfg["mix_base_tf"] + cfg["mix_add_tf_on_trend"] * int(G_TREND)
    s = w_mr + w_tf
    if s <= 0:
        return mr_pred
    return (w_mr/s) * mr_pred + (w_tf/s) * tf_pred


# =========================
# Coef persistence
# =========================

class CoefPersistence:
    def __init__(self, window=60, nonzero_eps=1e-10):
        self.window = int(window)
        self.nonzero_eps = float(nonzero_eps)
        self.coef_hist = deque(maxlen=self.window)
        self.feature_names = None

    def update(self, model_pipeline, feat_cols):
        reg = model_pipeline.named_steps["reg"]
        coefs = np.asarray(reg.coef_).ravel().copy()
        if self.feature_names is None:
            self.feature_names = list(feat_cols)
        else:
            if list(feat_cols) != self.feature_names:
                self.feature_names = list(feat_cols)
                self.coef_hist.clear()
        self.coef_hist.append(coefs)

    def stable_feature_mask(self, min_nonzero_rate=0.3, min_sign_consistency=0.7):
        if self.feature_names is None or len(self.coef_hist) < 2:
            return None
        W = np.vstack(self.coef_hist)
        nz = (np.abs(W) > self.nonzero_eps).astype(int)
        nonzero_rate = nz.mean(axis=0)

        pos = (W > self.nonzero_eps).sum(axis=0)
        neg = (W < -self.nonzero_eps).sum(axis=0)
        denom = (pos + neg).astype(float) + 1e-12
        sign_consistency = np.maximum(pos, neg) / denom

        stable = (nonzero_rate >= min_nonzero_rate) & (sign_consistency >= min_sign_consistency)
        return stable

    def model_stability_score(self, min_nonzero_rate=0.3, min_sign_consistency=0.7):
        m = self.stable_feature_mask(min_nonzero_rate, min_sign_consistency)
        if m is None:
            return 0.0
        return float(m.mean())


# =========================
# RuleCleaner (corr > thr; keep by |coef|)
# =========================

class RuleCleaner:
    def __init__(self, corr_thr=0.8):
        self.corr_thr = float(corr_thr)
        self.keep_cols_ = None

    def fit(self, X_df: pd.DataFrame, coef_by_col: dict):
        cols = list(X_df.columns)
        if len(cols) <= 1:
            self.keep_cols_ = cols
            return self

        corr = X_df.corr().abs()
        np.fill_diagonal(corr.values, 0.0)

        strength = pd.Series({c: abs(float(coef_by_col.get(c, 0.0))) for c in cols})
        ordered = strength.sort_values(ascending=False).index.tolist()

        keep = []
        dropped = set()
        for c in ordered:
            if c in dropped:
                continue
            keep.append(c)
            high = corr.index[corr[c] > self.corr_thr].tolist()
            for h in high:
                dropped.add(h)

        self.keep_cols_ = keep
        return self

    def transform(self, X_df: pd.DataFrame) -> pd.DataFrame:
        if self.keep_cols_ is None:
            return X_df
        cols = [c for c in self.keep_cols_ if c in X_df.columns]
        return X_df[cols].copy()


# =========================
# Daily sample builders (MR and TF)
# =========================

def build_daily_samples_regression_with_gates(panel, feats, mkt_df_with_gates, cfg, universe_syms):
    c = panel["close"]
    days = pd.Index(sorted(set(c.index.floor("D"))))
    rows = []

    qv_day = feats["qv"].groupby(feats["qv"].index.floor("D")).sum()

    for di in range(len(days) - 2):
        d = days[di]
        d_next = days[di + 1]

        mask_d = (c.index.floor("D") == d)
        if not mask_d.any():
            continue
        ts_sig = c.index[mask_d][-1]

        mask_next = (c.index.floor("D") == d_next)
        if not mask_next.any():
            continue
        ts_entry = c.index[mask_next][0]

        ts_exit = ts_entry + pd.Timedelta(hours=cfg["label_horizon_hours"])
        if ts_exit not in c.index:
            continue

        if d not in qv_day.index:
            continue
        qv = qv_day.loc[d].dropna().sort_values(ascending=False)
        uni = [s for s in qv.head(cfg["universe_topN"]).index if s in universe_syms]
        if not uni:
            continue

        if ts_sig not in mkt_df_with_gates.index:
            continue
        mkt_row = mkt_df_with_gates.loc[ts_sig]

        top_syms, bot_syms = select_extreme_movers(
            ret_series=feats["ret24h"].loc[ts_sig, uni],
            pct=cfg["train_extreme_pct"],
            min_n=cfg["train_extreme_min"],
            max_n=cfg["train_extreme_max"],
        )
        train_syms = set(top_syms) | set(bot_syms)
        if not train_syms:
            continue

        uni_ret24 = feats["ret24h"].loc[ts_sig, uni].dropna()
        if len(uni_ret24) == 0:
            continue
        breadth_pos = float((uni_ret24 > 0).mean())
        breadth_neg = float((uni_ret24 < 0).mean())
        uni_mom     = float(uni_ret24.mean())
        uni_disp    = float(uni_ret24.std(ddof=0))

        # vectorized label base for this day
        c0 = c.loc[ts_entry, list(train_syms)]
        c1 = c.loc[ts_exit, list(train_syms)]
        y = (c1 / (c0 + 1e-12) - 1.0).astype(np.float32)

        for sym in train_syms:
            if sym not in c.columns:
                continue
            if pd.isna(y.get(sym, np.nan)):
                continue

            try:
                x = {
                    "day": d,
                    "ts_sig": ts_sig,
                    "symbol": sym,

                    "a_ret24h": float(feats["ret24h"].loc[ts_sig, sym]),
                    "a_ret6h":  float(feats["ret6h"].loc[ts_sig, sym]),
                    "a_atr":    float(feats["atr_pct"].loc[ts_sig, sym]),
                    "a_rsi":    float(feats["rsi"].loc[ts_sig, sym]),
                    "a_volz":   float(feats["vol_z"].loc[ts_sig, sym]),
                    "a_trend":  float(feats["trend_pct"].loc[ts_sig, sym]),
                    "a_rv24":   float(feats["rv_24h"].loc[ts_sig, sym]),
                    "a_range":  float(feats["range_pct"].loc[ts_sig, sym]),
                    "a_gap":    float(feats["gap_pct"].loc[ts_sig, sym]),
                    "a_body":   float(feats["body_pct"].loc[ts_sig, sym]),

                    "mkt_ret24h": float(mkt_row["mkt_ret24h"]),
                    "mkt_ret6h":  float(mkt_row["mkt_ret6h"]),
                    "mkt_trend":  float(mkt_row["mkt_trend"]),
                    "mkt_rv":     float(mkt_row["mkt_rv"]),

                    "G_VOL": int(mkt_row["G_VOL"]),
                    "G_TREND": int(mkt_row["G_TREND"]),

                    "u_breadth_pos": breadth_pos,
                    "u_breadth_neg": breadth_neg,
                    "u_mom":         uni_mom,
                    "u_disp":        uni_disp,

                    "y": float(y[sym]),
                }
            except Exception:
                continue

            rows.append(x)

    df = pd.DataFrame(rows)
    if df.empty:
        return df, []

    df = apply_interaction_toggles(
        df,
        causal_cols=cfg["causal_cols"],
        gate_cols=["G_VOL", "G_TREND"],
        drop_raw=cfg["drop_raw_causal"]
    )
    feat_cols = [c for c in df.columns if c not in ("day","ts_sig","symbol","y")]
    df = df.dropna(subset=feat_cols + ["y"])
    return df, feat_cols

def build_daily_samples_trend_follow(panel, feats, mkt_df_with_gates, cfg, universe_syms):
    c = panel["close"]
    days = pd.Index(sorted(set(c.index.floor("D"))))
    rows = []

    qv_day = feats["qv"].groupby(feats["qv"].index.floor("D")).sum()

    for di in range(len(days) - 2):
        d = days[di]
        d_next = days[di + 1]

        mask_d = (c.index.floor("D") == d)
        if not mask_d.any():
            continue
        ts_sig = c.index[mask_d][-1]

        mask_next = (c.index.floor("D") == d_next)
        if not mask_next.any():
            continue
        ts_entry = c.index[mask_next][0]

        ts_exit = ts_entry + pd.Timedelta(hours=cfg["label_horizon_hours"])
        if ts_exit not in c.index:
            continue

        if d not in qv_day.index:
            continue
        qv = qv_day.loc[d].dropna().sort_values(ascending=False)
        uni = [s for s in qv.head(cfg["universe_topN"]).index if s in universe_syms]
        if not uni:
            continue

        if ts_sig not in mkt_df_with_gates.index:
            continue
        mkt_row = mkt_df_with_gates.loc[ts_sig]

        trend_ser = feats["trend_pct"].loc[ts_sig, uni].dropna()
        if len(trend_ser) == 0:
            continue

        k = max(cfg["tf_train_min"], int(len(trend_ser) * cfg["tf_train_pct"]))
        k = min(k, cfg["tf_train_max"])

        pos = trend_ser.sort_values(ascending=False).head(k).index.tolist()
        neg = trend_ser.sort_values(ascending=True).head(k).index.tolist()
        train_syms = set(pos) | set(neg)
        if not train_syms:
            continue

        uni_ret24 = feats["ret24h"].loc[ts_sig, uni].dropna()
        if len(uni_ret24) == 0:
            continue
        breadth_pos = float((uni_ret24 > 0).mean())
        breadth_neg = float((uni_ret24 < 0).mean())
        uni_mom     = float(uni_ret24.mean())
        uni_disp    = float(uni_ret24.std(ddof=0))

        c0 = c.loc[ts_entry, list(train_syms)]
        c1 = c.loc[ts_exit, list(train_syms)]
        y = (c1 / (c0 + 1e-12) - 1.0).astype(np.float32)

        for sym in train_syms:
            if sym not in c.columns:
                continue
            if pd.isna(y.get(sym, np.nan)):
                continue

            try:
                x = {
                    "day": d,
                    "ts_sig": ts_sig,
                    "symbol": sym,

                    "a_ret24h": float(feats["ret24h"].loc[ts_sig, sym]),
                    "a_ret6h":  float(feats["ret6h"].loc[ts_sig, sym]),
                    "a_atr":    float(feats["atr_pct"].loc[ts_sig, sym]),
                    "a_rsi":    float(feats["rsi"].loc[ts_sig, sym]),
                    "a_volz":   float(feats["vol_z"].loc[ts_sig, sym]),
                    "a_trend":  float(feats["trend_pct"].loc[ts_sig, sym]),
                    "a_rv24":   float(feats["rv_24h"].loc[ts_sig, sym]),
                    "a_range":  float(feats["range_pct"].loc[ts_sig, sym]),
                    "a_gap":    float(feats["gap_pct"].loc[ts_sig, sym]),
                    "a_body":   float(feats["body_pct"].loc[ts_sig, sym]),

                    "mkt_ret24h": float(mkt_row["mkt_ret24h"]),
                    "mkt_ret6h":  float(mkt_row["mkt_ret6h"]),
                    "mkt_trend":  float(mkt_row["mkt_trend"]),
                    "mkt_rv":     float(mkt_row["mkt_rv"]),

                    "G_VOL": int(mkt_row["G_VOL"]),
                    "G_TREND": int(mkt_row["G_TREND"]),

                    "u_breadth_pos": breadth_pos,
                    "u_breadth_neg": breadth_neg,
                    "u_mom":         uni_mom,
                    "u_disp":        uni_disp,

                    "y": float(y[sym]),
                }
            except Exception:
                continue

            rows.append(x)

    df = pd.DataFrame(rows)
    if df.empty:
        return df, []

    df = apply_interaction_toggles(
        df,
        causal_cols=cfg["causal_cols"],
        gate_cols=["G_VOL", "G_TREND"],
        drop_raw=cfg["drop_raw_causal"]
    )
    feat_cols = [c for c in df.columns if c not in ("day","ts_sig","symbol","y")]
    df = df.dropna(subset=feat_cols + ["y"])
    return df, feat_cols


# =========================
# Peak exhaustion samples (trained hourly on rolling window)
# =========================

def build_hourly_exhaustion_train_frame(panel, feats, mkt_df_with_gates, cfg, ts_end, lookback_hours, universe_syms, feat_cols_expected):
    """
    Returns (X_df, y) for exhaustion model training over [ts_start, ts_end] hours,
    using only symbols in universe_syms (kept modest for performance).
    Vectorized label approximation:
      - direction = sign(ret24h(t))
      - future_max/min in [t..t+H] computed via reverse-rolling
      - exhaustion if adverse move within horizon exceeds threshold:
          direction>0: (future_min/future_max - 1) <= -thr
          direction<0: (future_max/future_min - 1) >=  thr
    """
    c = panel["close"]
    idx = c.index

    ts_start = ts_end - pd.Timedelta(hours=int(lookback_hours))
    if ts_start not in idx:
        # snap to nearest available
        ts_start = idx[idx.get_indexer([ts_start], method="backfill")[0]]

    # need horizon beyond ts_end for labels
    H = int(cfg["exh_horizon_hours"])
    ts_ext_end = ts_end + pd.Timedelta(hours=H)
    if ts_ext_end not in idx:
        # cannot label the most recent tail; shrink end
        # move ts_end back so that ts_end+H exists
        valid = idx[idx <= (idx.max() - pd.Timedelta(hours=H))]
        if len(valid) == 0:
            return None, None
        ts_end = valid.max()
        ts_ext_end = ts_end + pd.Timedelta(hours=H)

    syms = [s for s in universe_syms if s in c.columns]
    if not syms:
        return None, None

    # slice close with horizon extension
    close_ext = c.loc[ts_start:ts_ext_end, syms].astype(np.float32)
    close_win = c.loc[ts_start:ts_end, syms].astype(np.float32)

    # direction proxy at each hour in window
    dir_mat = np.sign(feats["ret24h"].loc[ts_start:ts_end, syms].astype(np.float32).values)
    # avoid zeros
    dir_mat[dir_mat == 0] = np.nan

    # future max/min over [t..t+H] (inclusive)
    rev = close_ext.iloc[::-1]
    future_max_ext = rev.rolling(H + 1, min_periods=H + 1).max().iloc[::-1]
    future_min_ext = rev.rolling(H + 1, min_periods=H + 1).min().iloc[::-1]
    future_max = future_max_ext.loc[ts_start:ts_end]
    future_min = future_min_ext.loc[ts_start:ts_end]

    # adverse move measures
    # long direction: need big drop somewhere -> approx by min/max ratio
    ratio_long = (future_min / (future_max + 1e-12)) - 1.0  # negative
    # short direction: need big spike up -> approx by max/min ratio
    ratio_short = (future_max / (future_min + 1e-12)) - 1.0  # positive

    thr = float(cfg["exh_reversal_thr"])
    y_long = (ratio_long.values <= -thr).astype(np.int8)
    y_short = (ratio_short.values >=  thr).astype(np.int8)

    dir_pos = (dir_mat > 0).astype(bool)
    dir_neg = (dir_mat < 0).astype(bool)
    y = np.full_like(y_long, fill_value=-1, dtype=np.int8)
    y[dir_pos] = y_long[dir_pos]
    y[dir_neg] = y_short[dir_neg]

    # build feature frame (same raw features + market + gates) then interactionize
    # We assemble wide, then stack (manageable: lookback_hours * len(syms))
    t_index = close_win.index

    # market/gates at each hour
    mkt = mkt_df_with_gates.loc[t_index, ["mkt_ret24h","mkt_ret6h","mkt_trend","mkt_rv","G_VOL","G_TREND"]].copy()
    # broadcast market cols across syms via repeat in stacking
    # asset features
    feat_map = {
        "a_ret24h": feats["ret24h"].loc[t_index, syms],
        "a_ret6h":  feats["ret6h"].loc[t_index, syms],
        "a_atr":    feats["atr_pct"].loc[t_index, syms],
        "a_rsi":    feats["rsi"].loc[t_index, syms],
        "a_volz":   feats["vol_z"].loc[t_index, syms],
        "a_trend":  feats["trend_pct"].loc[t_index, syms],
        "a_rv24":   feats["rv_24h"].loc[t_index, syms],
        "a_range":  feats["range_pct"].loc[t_index, syms],
        "a_gap":    feats["gap_pct"].loc[t_index, syms],
        "a_body":   feats["body_pct"].loc[t_index, syms],
    }

    # stack to long
    parts = []
    for name, mat in feat_map.items():
        s = mat.stack(dropna=False).rename(name)
        parts.append(s)

    X = pd.concat(parts, axis=1)
    X.index.names = ["ts", "symbol"]

    # add market features / gates to each (ts, symbol)
    for col in ["mkt_ret24h","mkt_ret6h","mkt_trend","mkt_rv","G_VOL","G_TREND"]:
        X[col] = X.index.get_level_values("ts").map(mkt[col]).astype(np.float32)

    # attach y
    y_df = pd.DataFrame(y, index=t_index, columns=syms).stack(dropna=False).rename("y_exh")
    y_df = y_df.reindex(X.index)
    X = X.join(y_df)

    # drop invalid direction rows and NaNs
    X = X[X["y_exh"] >= 0]
    X = X.dropna()

    # interactionize
    X_reset = X.reset_index()
    X_int = apply_interaction_toggles(
        X_reset,
        causal_cols=cfg["causal_cols"],
        gate_cols=["G_VOL","G_TREND"],
        drop_raw=cfg["drop_raw_causal"]
    )

    # final feature cols
    feat_cols = [c for c in X_int.columns if c not in ("ts","symbol","y_exh")]
    # enforce same feature space as daily models
    if feat_cols != feat_cols_expected:
        # strict to avoid silent mismatch
        return None, None

    y_out = X_int["y_exh"].astype(int).to_numpy()
    X_out = X_int[feat_cols]

    # downcast X to float32
    X_out = X_out.astype(np.float32)
    return X_out, y_out


# =========================
# Backtest (3y) — MR/TF + RuleCleaner + Exhaustion
# =========================

def backtest_last_3y(panel, feats, mkt_df_with_gates, cfg, margin_symbols, market_basket):
    o, h, l, c = panel["open"], panel["high"], panel["low"], panel["close"]

    # enforce margin universe overlap
    symbols = sorted(set(c.columns).intersection(set(margin_symbols)))
    if not symbols:
        raise ValueError("No overlap between fetched OHLCV symbols and margin-market symbols.")
    # enforce basket presence (gates dependency)
    assert_basket_present(c, market_basket)

    # daily samples for MR/TF
    samples_mr, feat_cols_mr = build_daily_samples_regression_with_gates(panel, feats, mkt_df_with_gates, cfg, universe_syms=symbols)
    samples_tf, feat_cols_tf = build_daily_samples_trend_follow(panel, feats, mkt_df_with_gates, cfg, universe_syms=symbols)
    if samples_mr.empty or samples_tf.empty:
        raise ValueError("Samples missing (MR or TF). Adjust universe/history or selection criteria.")
    if feat_cols_mr != feat_cols_tf:
        raise ValueError("Feature columns differ between MR and TF sample builders. Keep them aligned.")
    feat_cols = feat_cols_mr

    last_day = c.index.floor("D").max()
    start_day = last_day - pd.Timedelta(days=365*3)
    days = pd.Index(sorted(set(c.index.floor("D"))))
    days = days[days >= start_day]

    equity = 1.0
    eq_curve = []
    trades = []

    daily_borrow = cfg["borrow_apr"] / 365.0
    fee_rt = cfg["fee_bps"] / 1e4

    coef_persist_mr = CoefPersistence(window=cfg["coef_persist_window"])
    coef_persist_tf = CoefPersistence(window=cfg["coef_persist_window"])
    cleaner_mr = RuleCleaner(corr_thr=cfg["ruleclean_corr_thr"])
    cleaner_tf = RuleCleaner(corr_thr=cfg["ruleclean_corr_thr"])

    for di in range(len(days) - 2):
        d = days[di]
        d_next = days[di + 1]

        # signal ts = last hour of day d
        mask_d = (c.index.floor("D") == d)
        if not mask_d.any():
            eq_curve.append((d_next, equity))
            continue
        ts_sig = c.index[mask_d][-1]

        # entry ts = first hour of next day
        mask_next = (c.index.floor("D") == d_next)
        if not mask_next.any():
            eq_curve.append((d_next, equity))
            continue
        ts_entry = c.index[mask_next][0]

        # rolling training window (STRICTLY past)
        train_start = d - pd.Timedelta(days=cfg["train_days"])
        train_end   = d - pd.Timedelta(days=1)

        train_mr = samples_mr[(samples_mr["day"] >= train_start) & (samples_mr["day"] <= train_end)]
        train_tf = samples_tf[(samples_tf["day"] >= train_start) & (samples_tf["day"] <= train_end)]
        if len(train_mr) < cfg["min_train_samples_mr"] or len(train_tf) < cfg["min_train_samples_tf"]:
            eq_curve.append((d_next, equity))
            continue

        # Train MR
        model_mr = make_elasticnet_reg(alpha=cfg["alpha_mr"], l1_ratio=cfg["l1_ratio_mr"])
        Xtr_mr_df = train_mr[feat_cols].copy()
        ytr_mr = train_mr["y"].to_numpy(dtype=np.float32, copy=False)
        model_mr.fit(Xtr_mr_df.to_numpy(dtype=np.float32, copy=False), ytr_mr)
        coef_persist_mr.update(model_mr, feat_cols)

        mr_coefs = dict(zip(feat_cols, model_mr.named_steps["reg"].coef_.ravel().tolist()))
        cleaner_mr.fit(Xtr_mr_df, mr_coefs)

        stable_mask_mr = coef_persist_mr.stable_feature_mask(
            min_nonzero_rate=cfg["min_feat_nonzero_rate"],
            min_sign_consistency=cfg["min_feat_sign_consistency"]
        )
        stability_mr = coef_persist_mr.model_stability_score(
            min_nonzero_rate=cfg["min_feat_nonzero_rate"],
            min_sign_consistency=cfg["min_feat_sign_consistency"]
        )

        # Train TF
        model_tf = make_elasticnet_reg(alpha=cfg["alpha_tf"], l1_ratio=cfg["l1_ratio_tf"])
        Xtr_tf_df = train_tf[feat_cols].copy()
        ytr_tf = train_tf["y"].to_numpy(dtype=np.float32, copy=False)
        model_tf.fit(Xtr_tf_df.to_numpy(dtype=np.float32, copy=False), ytr_tf)
        coef_persist_tf.update(model_tf, feat_cols)

        tf_coefs = dict(zip(feat_cols, model_tf.named_steps["reg"].coef_.ravel().tolist()))
        cleaner_tf.fit(Xtr_tf_df, tf_coefs)

        stable_mask_tf = coef_persist_tf.stable_feature_mask(
            min_nonzero_rate=cfg["min_feat_nonzero_rate"],
            min_sign_consistency=cfg["min_feat_sign_consistency"]
        )
        stability_tf = coef_persist_tf.model_stability_score(
            min_nonzero_rate=cfg["min_feat_nonzero_rate"],
            min_sign_consistency=cfg["min_feat_sign_consistency"]
        )

        # Skip if both too unstable
        if (stability_mr < cfg["min_model_stability_to_trade"]) and (stability_tf < cfg["min_model_stability_to_trade"]):
            eq_curve.append((d_next, equity))
            continue

        # today's candidate rows
        today_mr = samples_mr[samples_mr["day"] == d].copy()
        today_tf = samples_tf[samples_tf["day"] == d].copy()
        if today_mr.empty or today_tf.empty:
            eq_curve.append((d_next, equity))
            continue

        today = pd.merge(
            today_mr[["symbol","G_VOL","G_TREND"] + feat_cols],
            today_tf[["symbol"] + feat_cols],
            on="symbol",
            suffixes=("_mr", "_tf"),
            how="inner"
        )
        if today.empty:
            eq_curve.append((d_next, equity))
            continue

        # blended stability -> effective cap
        G_VOL_day = int(today["G_VOL"].iloc[0])
        G_TREND_day = int(today["G_TREND"].iloc[0])
        w_mr = cfg["mix_base_mr"] + cfg["mix_add_mr_on_vol"] * G_VOL_day
        w_tf = cfg["mix_base_tf"] + cfg["mix_add_tf_on_trend"] * G_TREND_day
        s = w_mr + w_tf
        blended_stability = stability_mr if s <= 0 else (w_mr/s)*stability_mr + (w_tf/s)*stability_tf
        target = max(cfg["target_model_stability"], 1e-6)
        stability_scale = min(1.0, max(0.0, blended_stability / target))
        effective_gross_cap = cfg["wallet_gross_cap"] * stability_scale

        # Predict MR with RuleCleaner + persistence mask
        X_mr_df = today[[c_ + "_mr" for c_ in feat_cols]].copy()
        X_mr_df.columns = feat_cols
        X_mr_df = cleaner_mr.transform(X_mr_df).astype(np.float32)
        X_mr = X_mr_df.to_numpy(dtype=np.float32, copy=False)
        if stable_mask_mr is not None:
            kept = list(X_mr_df.columns)
            kept_idx = [feat_cols.index(k) for k in kept]
            mask_kept = stable_mask_mr[kept_idx]
            X_mr[:, ~mask_kept] = 0.0
        pred_mr = model_mr.predict(X_mr).astype(np.float32)

        # Predict TF
        X_tf_df = today[[c_ + "_tf" for c_ in feat_cols]].copy()
        X_tf_df.columns = feat_cols
        X_tf_df = cleaner_tf.transform(X_tf_df).astype(np.float32)
        X_tf = X_tf_df.to_numpy(dtype=np.float32, copy=False)
        if stable_mask_tf is not None:
            kept = list(X_tf_df.columns)
            kept_idx = [feat_cols.index(k) for k in kept]
            mask_kept = stable_mask_tf[kept_idx]
            X_tf[:, ~mask_kept] = 0.0
        pred_tf = model_tf.predict(X_tf).astype(np.float32)

        # Blend prediction
        today["pred_mr"] = pred_mr
        today["pred_tf"] = pred_tf
        today["pred"] = [
            mix_predictions(mr, tf, gv, gt, cfg)
            for mr, tf, gv, gt in zip(pred_mr, pred_tf, today["G_VOL"].values, today["G_TREND"].values)
        ]

        # (8) Peak exhaustion model trained hourly on rolling window ending at ts_sig
        # Train on symbols in today's merged set (keeps it bounded and consistent)
        exh_syms = today["symbol"].unique().tolist()
        X_exh, y_exh = build_hourly_exhaustion_train_frame(
            panel=panel,
            feats=feats,
            mkt_df_with_gates=mkt_df_with_gates,
            cfg=cfg,
            ts_end=ts_sig,
            lookback_hours=cfg["exh_train_lookback_hours"],
            universe_syms=exh_syms,
            feat_cols_expected=feat_cols,
        )
        if X_exh is None or y_exh is None or len(y_exh) < cfg["min_exh_samples"] or len(np.unique(y_exh)) < 2:
            eq_curve.append((d_next, equity))
            continue

        exh_model = make_exhaustion_model(C=cfg["exh_C"], l1_ratio=cfg["exh_l1_ratio"])
        exh_model.fit(X_exh.to_numpy(dtype=np.float32, copy=False), y_exh)

        # Exhaustion probability for today's picks (use MR-view features as "state")
        X_exh_today = today[[c_ + "_mr" for c_ in feat_cols]].copy()
        X_exh_today.columns = feat_cols
        X_exh_today = X_exh_today.astype(np.float32)
        p_exh = exh_model.predict_proba(X_exh_today.to_numpy(dtype=np.float32, copy=False))[:, 1].astype(np.float32)
        today["p_exhaust"] = p_exh

        # block & downweight
        today = today[today["p_exhaust"] < cfg["exh_block_thr"]]
        if today.empty:
            eq_curve.append((d_next, equity))
            continue

        long_df  = today[today["pred"] >= cfg["thr_long"]].copy()
        short_df = today[today["pred"] <= cfg["thr_short"]].copy()

        if not long_df.empty:
            long_df["score_raw"] = long_df["pred"].apply(lambda x: map_pred_to_score(float(x), cfg["score_map"], cfg["score_scale"]))
            long_df["score"] = long_df["score_raw"] * ((1.0 - long_df["p_exhaust"]) ** cfg["exh_score_pow"])
            long_df = long_df.sort_values("score", ascending=False).head(cfg["k_long"])

        if not short_df.empty:
            short_df["score_raw"] = (-short_df["pred"]).apply(lambda x: map_pred_to_score(float(x), cfg["score_map"], cfg["score_scale"]))
            short_df["score"] = short_df["score_raw"] * ((1.0 - short_df["p_exhaust"]) ** cfg["exh_score_pow"])
            short_df = short_df.sort_values("score", ascending=False).head(cfg["k_short"])

        if long_df.empty and short_df.empty:
            eq_curve.append((d_next, equity))
            continue

        picks = []
        for _, r in long_df.iterrows():
            picks.append((r["symbol"], "long", float(r["score"]), float(r["pred"]), float(r["p_exhaust"])))
        for _, r in short_df.iterrows():
            picks.append((r["symbol"], "short", float(r["score"]), float(r["pred"]), float(r["p_exhaust"])))

        total_score = sum(p[2] for p in picks)
        if total_score <= 0:
            eq_curve.append((d_next, equity))
            continue

        weights = [(sym, side, effective_gross_cap * (score / total_score), pred, pexh)
                   for sym, side, score, pred, pexh in picks]

        pnl = 0.0
        for sym, side, w, pred, pexh in weights:
            entry_px = entry_price_next_hour_open(o, ts_entry, sym)
            if np.isnan(entry_px) or entry_px <= 0:
                continue

            rr, exit_ts, why = simulate_trade_hourly(
                o_s=o[sym], h_s=h[sym], l_s=l[sym], c_s=c[sym],
                entry_ts=ts_entry,
                entry_px=entry_px,
                side=side,
                tp=cfg["tp"],
                sl=cfg["sl"],
                max_hold_hours=cfg["hold_hours"]
            )

            if side == "short":
                rr -= daily_borrow * (cfg["hold_hours"] / 24.0)

            rr -= 2.0 * fee_rt
            pnl += w * rr

            trades.append({
                "day": d,
                "ts_sig": ts_sig,
                "entry_ts": ts_entry,
                "exit_ts": exit_ts,
                "symbol": sym,
                "side": side,
                "weight": w,
                "pred_return": pred,
                "p_exhaust": pexh,
                "ret": rr,
                "pnl_contrib": w * rr,
                "entry_px": float(entry_px),
                "exit_reason": why,
                "stability_mr": stability_mr,
                "stability_tf": stability_tf,
                "blended_stability": blended_stability,
                "effective_gross_cap": effective_gross_cap
            })

        equity *= (1 + pnl)
        eq_curve.append((d_next, equity))

    eq = pd.Series({d: e for d, e in eq_curve}).sort_index()
    trades_df = pd.DataFrame(trades)

    if len(eq) > 2:
        dr = eq.pct_change().dropna()
        ann = 365.0
        sharpe = (dr.mean() / (dr.std(ddof=0) + 1e-12)) * math.sqrt(ann)
        cagr = eq.iloc[-1] ** (ann / len(eq)) - 1.0
        max_dd = (eq / eq.cummax() - 1.0).min()
    else:
        sharpe = np.nan; cagr = np.nan; max_dd = np.nan

    stats = {
        "total_return": float(eq.iloc[-1] - 1.0) if len(eq) else np.nan,
        "cagr": float(cagr),
        "sharpe": float(sharpe),
        "max_dd": float(max_dd),
        "n_trades": int(len(trades_df)) if not trades_df.empty else 0,
    }
    return eq, trades_df, stats


# =========================
# Universe builder (always include market basket)
# =========================

def build_fetch_universe(margin_symbols: list[str], market_basket: list[str], M: int = 350) -> list[str]:
    base = [s for s in margin_symbols if s.endswith("/USDT")]
    chosen = base[:M]
    return sorted(set(chosen).union(set(market_basket)))


# =========================
# MAIN
# =========================

if __name__ == "__main__":
    cfg = {
        # data/universe
        "universe_topN": 200,

        # label/training (daily MR/TF)
        "label_horizon_hours": 24,
        "train_days": 180,

        # MR extreme mover filter
        "train_extreme_pct": 0.05,
        "train_extreme_min": 5,
        "train_extreme_max": 20,

        # TF selection
        "tf_train_pct": 0.05,
        "tf_train_min": 5,
        "tf_train_max": 20,

        # gates
        "gate_vol_lookback_hours": 24*14,
        "gate_trend_thr": 0.02,

        # interaction features
        "causal_cols": ["a_ret24h","a_rsi","a_volz","a_atr","a_trend","a_rv24"],
        "drop_raw_causal": True,

        # MR/TF hyperparams
        "alpha_mr": 5e-4,
        "l1_ratio_mr": 0.30,
        "min_train_samples_mr": 1500,

        "alpha_tf": 5e-4,
        "l1_ratio_tf": 0.30,
        "min_train_samples_tf": 1500,

        # mixture weights
        "mix_base_mr": 1.0,
        "mix_base_tf": 1.0,
        "mix_add_mr_on_vol": 1.0,
        "mix_add_tf_on_trend": 1.0,

        # candidate thresholds
        "thr_long":  0.010,
        "thr_short": -0.010,

        # selection cap
        "k_long": 12,
        "k_short": 12,

        # sizing
        "wallet_gross_cap": 0.25,
        "score_map": "tanh",
        "score_scale": 15.0,

        # stability gating
        "coef_persist_window": 60,
        "min_feat_nonzero_rate": 0.30,
        "min_feat_sign_consistency": 0.70,
        "min_model_stability_to_trade": 0.15,
        "target_model_stability": 0.40,

        # RuleCleaner
        "ruleclean_corr_thr": 0.80,

        # risk mgmt / costs
        "tp": 0.06,
        "sl": 0.04,
        "hold_hours": 48,
        "fee_bps": 10.0,
        "borrow_apr": 0.20,

        # features
        "atr_n": 14,
        "rsi_n": 14,
        "volz_n": 24*7,
        "trend_sma_n": 24*14,

        # Exhaustion model
        "exh_horizon_hours": 24,
        "exh_reversal_thr": 0.04,
        "exh_train_lookback_hours": 24*30,  # train on last 30 days of hours
        "min_exh_samples": 5000,
        "exh_C": 1.0,
        "exh_l1_ratio": 0.30,
        "exh_block_thr": 0.70,
        "exh_score_pow": 1.5,
    }

    market_basket = ["BTC/USDT","ETH/USDT","AVAX/USDT","SOL/USDT","XRP/USDT"]

    ex = make_spot_exchange()

    # daily refresh margin universe (req #5)
    mu_cache = refresh_margin_universe_daily(None, quote="USDT")
    margin_symbols = mu_cache.symbols

    # build fetch universe with basket hard included
    syms = build_fetch_universe(margin_symbols, market_basket, M=350)

    # persistence store (Parquet)
    store = OHLCVStore(root_dir="data", timeframe="1h")

    # since (4 years)
    since = (pd.Timestamp.utcnow().tz_localize("UTC") - pd.Timedelta(days=365*4)).floor("D")
    since_ms = int(since.value // 10**6)

    dfs = {}
    for s in syms:
        try:
            df = store.update_symbol(ex, s, since_ms)
            # require enough history
            if len(df) >= 24 * 365 * 2:
                dfs[s] = df
        except Exception:
            continue

    panel = to_panel(dfs)

    # align common columns (keep basket symbols even if others drop)
    common = set(panel["close"].columns)
    for k in panel:
        common &= set(panel[k].columns)
    common = sorted(common)
    for k in panel:
        panel[k] = panel[k][common].dropna(how="all")

    # enforce basket presence for gates
    assert_basket_present(panel["close"], market_basket)

    # downcast wide panel to float32
    panel = downcast_panel_float32(panel)

    # compute features + market gates
    feats = compute_features_hourly(panel, cfg)
    mkt_df = compute_market_features(panel["close"], feats["ret1h"], market_basket, trend_sma_hours=24*14)
    mkt_df = add_regime_gates(mkt_df, cfg)

    # run backtest
    eq, trades, stats = backtest_last_3y(panel, feats, mkt_df, cfg, margin_symbols, market_basket)

    print("\nSTATS:", stats)
    print("Equity last:", float(eq.iloc[-1]) if len(eq) else None)

    print("\nTRADES SAMPLE:")
    if not trades.empty:
        print(trades.head(10).to_string(index=False))
    else:
        print("(no trades)")
