import numpy as np
import pandas as pd

from utils import tprint
from models import make_elasticnet_reg, make_exhaustion_model

# ---------- Interaction toggles ----------
def apply_interaction_toggles(df: pd.DataFrame, causal_cols, gate_cols, drop_raw=True):
    out = df.copy()
    for g in gate_cols:
        for col in causal_cols:
            out[f"{col}_{g}_0"] = out[col] * (1 - out[g])
            out[f"{col}_{g}_1"] = out[col] * out[g]
    if drop_raw:
        out = out.drop(columns=list(causal_cols), errors="ignore")
    return out

# ---------- RuleCleaner ----------
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

# ---------- Coef persistence ----------
from collections import deque
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

    def per_symbol_stability(self, kept_cols: list[str], feat_cols_all: list[str], stable_mask_all):
        if stable_mask_all is None:
            return 0.0
        idx = [feat_cols_all.index(c) for c in kept_cols if c in feat_cols_all]
        if not idx:
            return 0.0
        return float(stable_mask_all[idx].mean())

# ---------- Exhaustion training/pred ----------
def build_exhaustion_Xy(panel, feats, mkt_gates, cfg, ts_end, lookback_hours, syms):
    """
    Train on [ts_end-lookback .. ts_end-1]; label uses future horizon from each t in that window.
    """
    c = panel["close"]
    idx = c.index
    H = int(cfg["exh_horizon_hours"])
    ts_train_end = ts_end - pd.Timedelta(hours=1)
    ts_start = ts_train_end - pd.Timedelta(hours=int(lookback_hours))

    if ts_train_end not in idx:
        return None, None, None
    if (ts_train_end + pd.Timedelta(hours=H)) not in idx:
        return None, None, None

    syms = [s for s in syms if s in c.columns]
    if not syms:
        return None, None, None

    ts_ext_end = ts_train_end + pd.Timedelta(hours=H)
    close_ext = c.loc[ts_start:ts_ext_end, syms].astype(np.float32)
    close_win = c.loc[ts_start:ts_train_end, syms].astype(np.float32)
    t_index = close_win.index

    dir_mat = np.sign(feats["ret24h"].loc[t_index, syms].astype(np.float32).values)
    dir_mat[dir_mat == 0] = np.nan

    rev = close_ext.iloc[::-1]
    fmax_ext = rev.rolling(H + 1, min_periods=H + 1).max().iloc[::-1]
    fmin_ext = rev.rolling(H + 1, min_periods=H + 1).min().iloc[::-1]
    fmax = fmax_ext.loc[t_index]
    fmin = fmin_ext.loc[t_index]

    thr = float(cfg["exh_reversal_thr"])
    ratio_long = (fmin / (fmax + 1e-12)) - 1.0
    ratio_short = (fmax / (fmin + 1e-12)) - 1.0

    y_long = (ratio_long.values <= -thr).astype(np.int8)
    y_short = (ratio_short.values >=  thr).astype(np.int8)

    y = np.full_like(y_long, fill_value=-1, dtype=np.int8)
    y[dir_mat > 0] = y_long[dir_mat > 0]
    y[dir_mat < 0] = y_short[dir_mat < 0]

    # build X from configured keys (includes sin/cos time, funding proxy, etc.)
    parts = []
    for fk in cfg["exh_feature_keys"]:
        mat = feats[fk].loc[t_index, syms]
        parts.append(mat.stack(dropna=False).rename(fk))
    X = pd.concat(parts, axis=1)
    X.index.names = ["ts","symbol"]

    # add market features/gates
    mg = mkt_gates.loc[t_index, ["mkt_ret24h","mkt_ret6h","mkt_trend","mkt_rv","G_VOL","G_TREND"]]
    for col in mg.columns:
        X[col] = X.index.get_level_values("ts").map(mg[col]).astype(np.float32)

    y_ser = pd.DataFrame(y, index=t_index, columns=syms).stack(dropna=False).rename("y_exh")
    y_ser = y_ser.reindex(X.index)
    X = X.join(y_ser)

    X = X[X["y_exh"] >= 0].dropna()
    y_out = X["y_exh"].astype(int).to_numpy()
    X_out = X.drop(columns=["y_exh"]).astype(np.float32)
    return X_out, y_out, list(X_out.columns)

def compute_p_exhaustion_at_t(panel, feats, mkt_gates, cfg, ts, syms):
    Xtr, ytr, cols = build_exhaustion_Xy(panel, feats, mkt_gates, cfg, ts, cfg["exh_train_lookback_hours"], syms)
    if Xtr is None or len(ytr) < cfg["min_exh_samples"] or len(np.unique(ytr)) < 2:
        return pd.Series(index=syms, data=np.nan, dtype=np.float32)

    model = make_exhaustion_model(C=cfg["exh_C"], l1_ratio=cfg["exh_l1_ratio"])
    model.fit(Xtr.to_numpy(dtype=np.float32, copy=False), ytr)

    # build X at ts
    t_index = pd.DatetimeIndex([ts], tz="UTC")
    syms2 = [s for s in syms if s in panel["close"].columns]
    parts = []
    for fk in cfg["exh_feature_keys"]:
        parts.append(feats[fk].loc[t_index, syms2].stack(dropna=False).rename(fk))
    Xp = pd.concat(parts, axis=1)
    Xp.index.names = ["ts","symbol"]

    mg = mkt_gates.loc[t_index, ["mkt_ret24h","mkt_ret6h","mkt_trend","mkt_rv","G_VOL","G_TREND"]]
    for col in mg.columns:
        Xp[col] = Xp.index.get_level_values("ts").map(mg[col]).astype(np.float32)

    for c in cols:
        if c not in Xp.columns:
            Xp[c] = np.nan
    Xp = Xp[cols].dropna()
    if Xp.empty:
        return pd.Series(index=syms, data=np.nan, dtype=np.float32)

    p = model.predict_proba(Xp.to_numpy(dtype=np.float32, copy=False))[:, 1].astype(np.float32)
    out = pd.Series(p, index=Xp.index.get_level_values("symbol"))
    return out.reindex(syms).astype(np.float32)

# ---------- Hourly training set (merged cross-section) ----------
def build_hourly_training_set(panel, feats, mkt_gates, cfg, syms, ts_train_end, p_exh_hist, label_horizon_hours: int):
    """
    Builds hourly samples (merged cross-section). Label uses chosen horizon H.
    Adds sin/cos time features via feats['sin_hod',...] already included in feats.
    """
    o, c = panel["open"], panel["close"]
    idx = c.index
    H = int(label_horizon_hours)

    ts_start = ts_train_end - pd.Timedelta(hours=int(cfg["train_lookback_hours"]))
    ts_start = idx[idx.get_indexer([ts_start], method="backfill")[0]]

    valid_end = idx[idx <= (idx.max() - pd.Timedelta(hours=H+1))]
    t_all = idx[(idx >= ts_start) & (idx <= ts_train_end)]
    t_all = t_all[t_all.isin(valid_end)]

    rows = []
    for t in t_all:
        t_entry = t + pd.Timedelta(hours=1)
        t_exit  = t_entry + pd.Timedelta(hours=H)
        if t_entry not in idx or t_exit not in idx:
            continue

        sel = [s for s in syms if s in c.columns]
        dev = feats["ret1h_z"].loc[t, sel].dropna()
        if dev.empty:
            continue
        n = len(dev)
        k = max(cfg["train_extreme_min"], int(n * cfg["train_extreme_pct_hourly"]))
        k = min(k, cfg["train_extreme_max"])
        top_syms = dev.sort_values(ascending=False).head(k).index.tolist()
        bot_syms = dev.sort_values(ascending=True).head(k).index.tolist()
        train_syms = list(set(top_syms) | set(bot_syms))

        if t not in mkt_gates.index:
            continue
        mr = mkt_gates.loc[t]

        entry_open = o.loc[t_entry, train_syms]
        exit_close = c.loc[t_exit, train_syms]
        y = (exit_close / (entry_open + 1e-12) - 1.0).astype(np.float32)

        t_exh_lag = t - pd.Timedelta(hours=1)
        for sym in train_syms:
            if pd.isna(y.get(sym, np.nan)):
                continue
            p_lag = np.nan
            if t_exh_lag in p_exh_hist.index and sym in p_exh_hist.columns:
                p_lag = p_exh_hist.loc[t_exh_lag, sym]

            try:
                rows.append({
                    "ts": t,
                    "symbol": sym,

                    # returns horizons (1)
                    "a_ret12h": float(feats["ret12h"].loc[t, sym]),
                    "a_ret16h": float(feats["ret16h"].loc[t, sym]),
                    "a_ret20h": float(feats["ret20h"].loc[t, sym]),
                    "a_ret24h": float(feats["ret24h"].loc[t, sym]),
                    "a_ret28h": float(feats["ret28h"].loc[t, sym]),

                    "a_ret6h":  float(feats["ret6h"].loc[t, sym]),
                    "a_ret1h_z": float(feats["ret1h_z"].loc[t, sym]),
                    "a_atr":    float(feats["atr_pct"].loc[t, sym]),
                    "a_rsi":    float(feats["rsi"].loc[t, sym]),
                    "a_volz":   float(feats["vol_z"].loc[t, sym]),
                    "a_trend":  float(feats["trend_pct"].loc[t, sym]),
                    "a_rv24":   float(feats["rv_24h"].loc[t, sym]),
                    "a_range":  float(feats["range_pct"].loc[t, sym]),
                    "a_gap":    float(feats["gap_pct"].loc[t, sym]),
                    "a_body":   float(feats["body_pct"].loc[t, sym]),
                    "a_dist_ema_fast": float(feats["dist_ema_fast"].loc[t, sym]),
                    "a_dist_ema_slow": float(feats["dist_ema_slow"].loc[t, sym]),
                    "a_roc_div": float(feats["roc_div"].loc[t, sym]),
                    "a_vol_price_spread": float(feats["vol_price_spread"].loc[t, sym]),
                    "a_funding_proxy": float(feats["a_funding_proxy"].loc[t, sym]),

                    # time features (3)
                    "sin_hod": float(feats["sin_hod"].loc[t, sym]),
                    "cos_hod": float(feats["cos_hod"].loc[t, sym]),
                    "sin_dow": float(feats["sin_dow"].loc[t, sym]),
                    "cos_dow": float(feats["cos_dow"].loc[t, sym]),

                    # market
                    "mkt_ret24h": float(mr["mkt_ret24h"]),
                    "mkt_ret6h":  float(mr["mkt_ret6h"]),
                    "mkt_trend":  float(mr["mkt_trend"]),
                    "mkt_rv":     float(mr["mkt_rv"]),

                    # gates
                    "G_VOL": int(mr["G_VOL"]),
                    "G_TREND": int(mr["G_TREND"]),

                    # exhaustion as input
                    "p_exh_lag1": float(p_lag) if pd.notna(p_lag) else np.nan,

                    # label
                    "y": float(y[sym]),
                })
            except Exception:
                continue

    df = pd.DataFrame(rows)
    if df.empty:
        return df, []

    df = df.dropna()
    df = apply_interaction_toggles(
        df,
        causal
