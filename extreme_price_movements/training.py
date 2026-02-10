import numpy as np
import pandas as pd
from .utils import tprint
from .model_race import ModelRace
from .meta_model import MetaModel
from .exhaustion import ExhaustionModel
from .feature_selection_extreme_events import mdi_feature_selection_v3
from .candidates import select_trade_candidates_hourly, select_trade_candidates_vectorized
import extreme_price_movements.fast_funcs as ff
from .labeling import compute_trailing_atr_labels, compute_triple_barrier_labels
from .sample_weights import build_label_time_ranges, compute_sample_weights_with_uniqueness
from sklearn.mixture import GaussianMixture
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
from .optimise_tpsl_ratio import (
    run_tp_sl_selection_fast,
    calibrate_atr_base_pct,
    compute_vol_z_log_mad,
    PurgedKFold,
    scaled_atr_pct,
)
from .trap_specialist import build_trap_dataset, train_trap_from_dataset, compute_trap_oof_predictions
from .gamma_specialist import build_gamma_dataset, train_gamma_from_dataset
from .model_scoring import precision_at_k, avg_trades_per_day, ece_at_mask, topk_mask, calibration_curve_bins, calibration_profile, ic_cross_sectional
from .metrics import calculate_selection_score
from .gate_metrics import compute_stage_gate_metrics


def _aggregate_alpha_oof_metrics(y_true, probs, returns, sample_weight=None, groups=None):
    """Aggregate tradability and calibration metrics for alpha models."""
    metrics = {}
    y_bin = (np.asarray(y_true) >= 0.5).astype(int)
    probs = np.asarray(probs, dtype=float)
    returns = np.asarray(returns, dtype=float)
    sw = np.asarray(sample_weight, dtype=float) if sample_weight is not None else None

    mask = np.isfinite(probs) & np.isfinite(returns)
    if y_true is not None:
        mask &= np.isfinite(y_bin)
    if sw is not None:
        sw = sw[: len(mask)]
        mask &= np.isfinite(sw)

    if mask.sum() < 10:
        return metrics

    y_m = y_bin[mask]
    p_m = probs[mask]
    r_m = returns[mask]
    w_m = sw[mask] if sw is not None else None
    # Selection score already returns composite metrics we need
    sel_metrics = calculate_selection_score(y_m, p_m, r_m, sample_weight=w_m)
    metrics.update({
        "auc": float(sel_metrics.get("AUC", np.nan)),
        "ic": float(sel_metrics.get("IC", np.nan)),
        "sharpe": float(sel_metrics.get("Sharpe", np.nan)),
        "win_rate": float(sel_metrics.get("WinRate", np.nan)),
        "avg_return": float(sel_metrics.get("AvgReturn", np.nan)),
        "n_trades": int(sel_metrics.get("Trades", 0)),
        "rank_ic": float(sel_metrics.get("RankIC", np.nan)),
        "max_dd": float(sel_metrics.get("MaxDD", np.nan)),
        "sortino": float(sel_metrics.get("Sortino", np.nan)),
        "calmar": float(sel_metrics.get("Calmar", np.nan)),
    })

    if len(np.unique(y_m)) > 1:
        metrics["auc_raw"] = float(roc_auc_score(y_m, p_m))

    if groups is not None and mask.sum() > 0:
        grp = np.asarray(groups)[mask]
        ic = ic_cross_sectional(p_m, r_m, groups=grp)
        metrics["ic_cs"] = float(ic)

    return metrics

def compute_per_regime_metrics(y_true, y_prob, df, sample_weight=None, global_prev=None):
    """Compute BSS and AUC per regime bucket from OOF predictions (all unweighted).
    
    Args:
        y_true: labels (n,) — will be binarized at 0.5
        y_prob: predicted probabilities (n,)
        df: DataFrame with __regime_*__ columns
        sample_weight: IGNORED (kept for API compat). All metrics unweighted.
        global_prev: global prevalence for BSS_global baseline comparison
    
    Returns:
        dict: {regime_name: {bucket_label: {bss, bss_global, auc, brier, n}}}
    """
    from sklearn.metrics import roc_auc_score, brier_score_loss
    
    regime_cols = [c for c in df.columns if c.startswith("__regime_") and c.endswith("__")]
    if not regime_cols:
        return {}
    
    # Compute global prevalence if not provided
    y_bin_all = (np.asarray(y_true) >= 0.5).astype(np.int8)
    if global_prev is None:
        global_prev = float(np.mean(y_bin_all))
    bs_ref_global = global_prev * (1.0 - global_prev)
    bs_ref_global = max(bs_ref_global, 1e-6)
    
    bucket_labels = {0: "low", 1: "mid", 2: "high"}
    results = {}
    
    for rc in regime_cols:
        regime_name = rc.replace("__regime_", "").replace("__", "")
        regime_vals = df[rc].values if rc in df.columns else None
        if regime_vals is None:
            continue
        
        regime_results = {}
        for bucket_id, bucket_name in bucket_labels.items():
            mask = regime_vals == bucket_id
            n_bucket = int(mask.sum())
            if n_bucket < 20:
                regime_results[bucket_name] = {"bss": 0.0, "bss_global": 0.0, "auc": 0.5, "brier": 0.0, "n": n_bucket}
                continue
            
            y_b = y_bin_all[mask]
            p_b = np.clip(y_prob[mask], 1e-7, 1 - 1e-7)
            
            # BSS with bucket-specific baseline
            bss = 0.0
            bss_global = 0.0
            brier_basic = 0.0
            try:
                prev_bucket = float(np.mean(y_b))
                brier_basic = float(brier_score_loss(y_b, p_b))
                # Bucket-baseline BSS
                if 0.02 < prev_bucket < 0.98:
                    bs_ref_bucket = prev_bucket * (1.0 - prev_bucket)
                    bs_ref_bucket = max(bs_ref_bucket, 1e-6)
                    bss = 1.0 - (brier_basic / bs_ref_bucket)
                    if not np.isfinite(bss):
                        bss = 0.0
                # Global-baseline BSS (comparable across buckets)
                bss_global = 1.0 - (brier_basic / bs_ref_global)
                if not np.isfinite(bss_global):
                    bss_global = 0.0
            except Exception:
                bss = 0.0
                bss_global = 0.0
            
            # AUC
            auc = 0.5
            try:
                if len(np.unique(y_b)) > 1:
                    auc = float(roc_auc_score(y_b, p_b))
            except Exception:
                auc = 0.5
            
            regime_results[bucket_name] = {
                "bss": round(bss, 4), "bss_global": round(bss_global, 4),
                "auc": round(auc, 4), "brier": round(brier_basic, 4), "n": n_bucket
            }
        
        results[regime_name] = regime_results
    
    return results


def _fast_lookup(feat_df, event_ts, event_sym):
    """Fast extraction of values at (ts, sym) positions using numpy indexing.
    Returns 1D array of values. NaN where lookup fails."""
    row_idx = feat_df.index.get_indexer(event_ts)
    col_idx = feat_df.columns.get_indexer(event_sym)
    vals = feat_df.values
    # Mark invalid positions
    valid = (row_idx >= 0) & (col_idx >= 0)
    out = np.full(len(event_ts), np.nan, dtype=np.float32)
    if valid.any():
        out[valid] = vals[row_idx[valid], col_idx[valid]]
    return out


def apply_interaction_toggles(df: pd.DataFrame, causal_cols, gate_cols, drop_raw=True):
    out = df.copy()
    for g in gate_cols:
        if g not in out.columns:
            continue
        for col in causal_cols:
            if col in out.columns:
                out[f"{col}_{g}_0"] = out[col] * (1 - out[g])
                out[f"{col}_{g}_1"] = out[col] * out[g]
    if drop_raw:
        out = out.drop(columns=[c for c in causal_cols if c in out.columns], errors="ignore")
    return out



def _winsorize_and_unit_mean(arr, lo_q=0.05, hi_q=0.95, clip_min=0.75, clip_max=1.25):
    a = np.asarray(arr, dtype=np.float64)
    if a.size == 0:
        return a
    valid = np.isfinite(a)
    if not valid.any():
        return np.ones_like(a)
    v = a[valid]
    lo = np.quantile(v, lo_q)
    hi = np.quantile(v, hi_q)
    out = np.clip(a, lo, hi)
    out = np.clip(out, clip_min, clip_max)
    mu = np.nanmean(out)
    if not np.isfinite(mu) or mu <= 1e-12:
        return np.ones_like(out)
    return out / mu


def _normalize_cross_sectional(ts_vals, weights):
    w = np.asarray(weights, dtype=np.float64).copy()
    ts = np.asarray(ts_vals)
    if len(w) == 0:
        return w
    for t in np.unique(ts):
        m = ts == t
        s = np.sum(w[m])
        if s > 0:
            w[m] = w[m] / s
        else:
            w[m] = 1.0 / max(1, m.sum())
    return w


def _sigmoid(x):
    x = np.asarray(x, dtype=np.float64)
    x = np.clip(x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-x))


def _compute_atr_scale(atr_pct_df, cfg):
    fast_hl = int(cfg.get('atr_norm_fast_hl_hours', 24))
    slow_hl = int(cfg.get('atr_norm_slow_hl_hours', 24*5))
    global_hl = int(cfg.get('atr_norm_global_hl_hours', 24*5))
    warmup_h = int(cfg.get('atr_norm_warmup_hours', 24*10))
    g_lo, g_hi = cfg.get('atr_norm_clip_global', [0.7, 1.5])
    m_lo, m_hi = cfg.get('atr_norm_clip_scale', [0.6, 2.5])

    atr = atr_pct_df.astype(np.float64)
    ewm_fast = atr.ewm(halflife=fast_hl, min_periods=12).mean()
    ewm_slow = atr.ewm(halflife=slow_hl, min_periods=12).mean()
    atr_ewm = atr.ewm(halflife=global_hl, min_periods=12).mean()

    warmup_n = min(len(atr_ewm), max(24, warmup_h))
    atr_ref = float(np.nanmedian(atr_ewm.iloc[:warmup_n].values)) if warmup_n > 0 else float(np.nanmedian(atr_ewm.values))
    if not np.isfinite(atr_ref) or atr_ref <= 1e-9:
        atr_ref = float(np.nanmedian(atr.values))
    atr_ref = max(atr_ref, 1e-6)

    local = atr / (ewm_fast + 1e-12)
    global_raw = np.sqrt(ewm_slow / atr_ref)
    global_mult = np.clip(global_raw, g_lo, g_hi)
    atr_scale = np.clip(local * global_mult, m_lo, m_hi)
    return atr_scale.astype(np.float32), atr_ref

def compute_weights_logic(df, cfg, model_kind):
    tprint(f"Entering function: compute_weights_logic in training.py")
    from .model_mr import compute_mr_weights
    from .model_tf import compute_tf_weights
    if model_kind == "mr": return compute_mr_weights(df, cfg)
    else: return compute_tf_weights(df, cfg)

def build_exhaustion_Xy(panel, feats, mkt_gates, cfg, ts_end, lookback_hours, syms, trend_filter=None):
    tprint(f"Entering function: build_exhaustion_Xy in training.py")
    c = panel["close"]
    idx = c.index
    H = int(cfg["exh_horizon_hours"])
    ts_train_end = ts_end - pd.Timedelta(hours=1)
    ts_start = ts_train_end - pd.Timedelta(hours=int(lookback_hours))
    if ts_train_end not in idx: return None, None, None, None
    mask = (idx >= ts_start) & (idx <= ts_train_end + pd.Timedelta(hours=H))
    idx_slice = idx[mask]
    valid_syms = [s for s in syms if s in c.columns]
    # Also filter against feature columns to avoid KeyError on newly listed symbols
    for fk in cfg.get("exh_feature_keys", []):
        if fk in feats:
            valid_syms = [s for s in valid_syms if s in feats[fk].columns]
    if not valid_syms: return None, None, None, None
    t_index = idx[(idx >= ts_start) & (idx <= ts_train_end)]
    current = c.loc[t_index, valid_syms]

    label_type = cfg.get("exh_label_type", "simple")
    tprint(f"Exhaustion Label Type: {label_type}")
    if label_type == "peak":
        use_atr = cfg.get("exh_use_atr", True)
        if use_atr and "atr_pct" in feats:
             atr_full = feats["atr_pct"] * panel["close"]
             near_k = float(cfg.get("exh_atr_near_k", 0.5))
             rev_k = float(cfg.get("exh_atr_rev_k", 2.0))
        else:
             atr_full = panel["close"]
             near_k = float(cfg.get("exh_near_thr", 0.01))
             rev_k = float(cfg.get("exh_rev_thr_pct", 0.04))

        common_idx = panel["close"].index.intersection(atr_full.index)
        c_full = panel["close"].loc[common_idx]
        a_full = atr_full.loc[common_idx]

        max_near = float(cfg.get("exh_near_dist_cap_pct", 0.02))
        min_rev = float(cfg.get("exh_rev_dist_floor_pct", 0.005))

        l_short, w_short = ff.compute_peak_labels_and_weights(c_full, a_full, H, near_k, rev_k, True, max_near, min_rev)
        l_long, w_long = ff.compute_peak_labels_and_weights(c_full, a_full, H, near_k, rev_k, False, max_near, min_rev)

        l_short_s = l_short.reindex(index=t_index, columns=valid_syms)
        l_long_s = l_long.reindex(index=t_index, columns=valid_syms)
        w_short_s = w_short.reindex(index=t_index, columns=valid_syms)
        w_long_s = w_long.reindex(index=t_index, columns=valid_syms)

        is_short_rev = l_short_s.fillna(0) > 0.5
        is_long_rev = l_long_s.fillna(0) > 0.5

    else:
        close_sub = c.loc[idx_slice, valid_syms].astype(np.float32)
        rev_close = close_sub.iloc[::-1]
        fmax = rev_close.rolling(H).max().shift(1).iloc[::-1]
        fmin = rev_close.rolling(H).min().shift(1).iloc[::-1]
        fmax = fmax.loc[t_index]; fmin = fmin.loc[t_index]
        thr = float(cfg["exh_reversal_thr"])
        is_short_rev = ((fmin / (current + 1e-12)) - 1.0) <= -thr
        is_long_rev = ((fmax / (current + 1e-12)) - 1.0) >= thr
    ret24 = feats["ret24h"].loc[t_index, valid_syms]
    dir_mat = np.sign(ret24).fillna(0).astype(np.int8)
    y = np.zeros(current.shape, dtype=np.int8)
    w = np.ones(current.shape, dtype=np.float32)

    mask_up = (dir_mat > 0)
    if mask_up.values.any():
        y[mask_up] = is_short_rev.values[mask_up].astype(np.int8)
        if cfg.get("exh_label_type") == "peak":
             w[mask_up] = w_short_s.values[mask_up].astype(np.float32)

    mask_dn = (dir_mat < 0)
    if mask_dn.values.any():
        y[mask_dn] = is_long_rev.values[mask_dn].astype(np.int8)
        if cfg.get("exh_label_type") == "peak":
             w[mask_dn] = w_long_s.values[mask_dn].astype(np.float32)

    if cfg.get("exh_label_type") == "peak":
        mask_boosted = w > 1.0
        if mask_boosted.sum() > 10:
             boosted_vals = w[mask_boosted]
             cap = np.quantile(boosted_vals, 0.80)
             w[w > cap] = cap
    X_parts = []
    # Exhaustion features are specific now
    for k in cfg.get("exh_feature_keys", cfg.get("exh_feature_keys_legacy", [])):
        if k in feats:
            X_parts.append(feats[k].loc[t_index, valid_syms].stack(future_stack=True).rename(k))
    X = pd.concat(X_parts, axis=1)
    X.index.names = ["ts", "symbol"]
    if trend_filter:
        trend_vals = feats["trend_pct"].loc[t_index, valid_syms].stack(future_stack=True)
        common_idx = X.index.intersection(trend_vals.index)
        X = X.loc[common_idx]; trend_vals = trend_vals.loc[common_idx]
        if trend_filter == "up": keep = trend_vals > 0
        else: keep = trend_vals <= 0
        X = X[keep]
        y_ser = pd.DataFrame(y, index=t_index, columns=valid_syms).stack(future_stack=True).rename("y").reindex(X.index)
        y_arr = y_ser.values.astype(int)
        
        w_ser = pd.DataFrame(w, index=t_index, columns=valid_syms).stack(future_stack=True).rename("w").reindex(X.index)
        w_arr = w_ser.values.astype(np.float32)
    else:
        y_df = pd.DataFrame(y, index=t_index, columns=valid_syms).stack(future_stack=True).rename("y")
        w_df = pd.DataFrame(w, index=t_index, columns=valid_syms).stack(future_stack=True).rename("w")
        X = X.join(y_df).join(w_df).dropna()
        y_arr = X.pop("y").astype(int).values
        w_arr = X.pop("w").astype(np.float32).values

    # Class imbalance correction: inverse-frequency weighting
    n_pos = (y_arr == 1).sum()
    n_neg = (y_arr == 0).sum()
    if n_pos > 0 and n_neg > 0:
        n_total = n_pos + n_neg
        w_pos = n_total / (2.0 * n_pos)
        w_neg = n_total / (2.0 * n_neg)
        class_mult = np.where(y_arr == 1, w_pos, w_neg).astype(np.float32)
        w_arr = w_arr * class_mult

    tprint(f"Exhaustion X shape: {X.shape}, y shape: {y_arr.shape}")
    tprint(f"Exhaustion class dist: {np.bincount(y_arr)}")

    mg = mkt_gates.loc[t_index, ["mkt_ret24h", "mkt_ret6h", "mkt_trend", "mkt_rv", "G_VOL", "G_TREND"]]
    for col in mg.columns:
        vals = pd.Series(X.index.get_level_values("ts").map(mg[col]).values, index=X.index)
        if vals.std() > 1e-9:
            X[col] = vals

    return X, y_arr, w_arr, list(X.columns)

def compute_p_exhaustion_at_t(panel, feats, mkt_gates, cfg, ts, syms, models=None):
    tprint(f"Entering function: compute_p_exhaustion_at_t in training.py")
    t_index = pd.DatetimeIndex([ts], tz="UTC")
    valid_syms = [s for s in syms if s in panel["close"].columns]
    trend_vals = feats["trend_pct"].loc[ts, valid_syms]
    up_syms = trend_vals[trend_vals > 0].index.tolist()
    dn_syms = trend_vals[trend_vals <= 0].index.tolist()
    tprint(f"compute_p_exhaustion_at_t: {len(up_syms)} up, {len(dn_syms)} down")

    out_probs = pd.Series(index=syms, dtype=float).fillna(0.0)
    lookback = cfg["exh_train_lookback_hours"]
    if up_syms:
        if models and "up" in models: model_up = models["up"]
        else:
            tprint("Training UP model...")
            X, y, w, _ = build_exhaustion_Xy(panel, feats, mkt_gates, cfg, ts, lookback, valid_syms, trend_filter="up")
            if X is not None and len(y) > 100:
                model_up = ExhaustionModel()
                model_up.fit(X, y, sample_weight=w)
            else:
                tprint("Not enough data for UP model.")
                model_up = None
        if model_up and model_up.model:
            Xp = _build_pred_X(feats, mkt_gates, cfg, ts, up_syms, feature_key="exh_feature_keys")
            if not Xp.empty:
                # No manual scaling needed with calibration
                probs = model_up.predict_proba(Xp)
                out_probs.loc[up_syms] = probs
    if dn_syms:
        if models and "down" in models: model_dn = models["down"]
        else:
            tprint("Training DOWN model...")
            X, y, w, _ = build_exhaustion_Xy(panel, feats, mkt_gates, cfg, ts, lookback, valid_syms, trend_filter="down")
            if X is not None and len(y) > 100:
                model_dn = ExhaustionModel()
                model_dn.fit(X, y, sample_weight=w)
            else:
                tprint("Not enough data for DOWN model.")
                model_dn = None
        if model_dn and model_dn.model:
            Xp = _build_pred_X(feats, mkt_gates, cfg, ts, dn_syms, feature_key="exh_feature_keys")
            if not Xp.empty:
                preds = model_dn.predict_proba(Xp)
                # No manual scaling needed
                out_probs.loc[dn_syms] = preds
    return out_probs.fillna(0.0)

def _build_pred_X(feats, mkt_gates, cfg, ts, syms, feature_key="exh_feature_keys"):
    tprint(f"Entering function: _build_pred_X in training.py")
    t_index = pd.DatetimeIndex([ts], tz="UTC")
    X_parts = []

    keys = cfg.get(feature_key, [])

    for k in keys:
        if k in feats:
            X_parts.append(feats[k].loc[t_index, syms].stack(future_stack=True).rename(k))
    if not X_parts: return pd.DataFrame()

    Xp = pd.concat(X_parts, axis=1)
    Xp.index.names = ["ts", "symbol"]
    mg = mkt_gates.loc[t_index, ["mkt_ret24h", "mkt_ret6h", "mkt_trend", "mkt_rv", "G_VOL", "G_TREND"]]
    for col in mg.columns:
        Xp[col] = Xp.index.get_level_values("ts").map(mg[col])
    return Xp.fillna(0)

def generate_exhaustion_history(panel, feats, mkt_gates, cfg, ts_end, lookback_hours, syms):
    tprint(f"Entering function: generate_exhaustion_history in training.py")
    train_end = ts_end - pd.Timedelta(hours=lookback_hours)
    train_len = cfg["exh_train_lookback_hours"]
    tprint("Generating UP history...")
    X_up, y_up, w_up, _ = build_exhaustion_Xy(panel, feats, mkt_gates, cfg, train_end, train_len, syms, trend_filter="up")
    model_up = None
    arr_oof_up = None
    if X_up is not None and len(y_up) > 100:
        model_up = ExhaustionModel()
        model_up.fit(X_up, y_up, sample_weight=w_up)
        # OOF Predictions for UP
        tprint("Generating OOF predictions for UP model...")
        oof_preds, _ = model_up.compute_oof_predictions(X_up, y_up)
        # Unstack to align with (ts, symbol) grid
        s_oof = pd.Series(oof_preds, index=X_up.index)
        # We need this to match the prediction window structure later
        # We'll delay unstacking until we have valid_syms and t_idx defined below

    tprint("Generating DOWN history...")
    X_dn, y_dn, w_dn, _ = build_exhaustion_Xy(panel, feats, mkt_gates, cfg, train_end, train_len, syms, trend_filter="down")
    model_dn = None
    arr_oof_dn = None
    if X_dn is not None and len(y_dn) > 100:
        model_dn = ExhaustionModel()
        model_dn.fit(X_dn, y_dn, sample_weight=w_dn)
        # OOF Predictions for DOWN
        tprint("Generating OOF predictions for DOWN model...")
        oof_preds, _ = model_dn.compute_oof_predictions(X_dn, y_dn)
        s_oof_dn = pd.Series(oof_preds, index=X_dn.index)

    # --- Fast vectorized prediction over full window ---
    t_idx = pd.date_range(train_end, ts_end, freq='h', tz="UTC")
    t_idx = t_idx[t_idx.isin(panel["close"].index)]
    valid_syms = [s for s in syms if s in panel["close"].columns]
    n_t, n_s = len(t_idx), len(valid_syms)
    tprint(f"Exhaustion prediction window: {n_t} timestamps x {n_s} symbols = {n_t * n_s} cells")

    # Prepare OOF arrays (n_t, n_s) aligned to prediction window
    if model_up and 's_oof' in locals():
        tprint("Aligning UP OOF predictions to grid...")
        df_oof = s_oof.unstack(level="symbol").reindex(index=t_idx, columns=valid_syms)
        arr_oof_up = df_oof.values.astype(np.float32) # contains NaNs where OOF missing

    if model_dn and 's_oof_dn' in locals():
        tprint("Aligning DOWN OOF predictions to grid...")
        df_oof = s_oof_dn.unstack(level="symbol").reindex(index=t_idx, columns=valid_syms)
        arr_oof_dn = df_oof.values.astype(np.float32)

    # Build feature+gate arrays as 2D (n_t, n_features) per symbol, predict per-symbol
    keys = cfg.get("exh_feature_keys", [])
    mkt_cols = ["mkt_ret24h", "mkt_ret6h", "mkt_trend", "mkt_rv", "G_VOL", "G_TREND"]
    mg = mkt_gates.reindex(t_idx)[mkt_cols].fillna(0).values.astype(np.float32)  # (n_t, 6)

    # Pre-extract feature arrays aligned to t_idx: dict[key] -> DataFrame(n_t, n_s)
    feat_aligned = {}
    for k in keys:
        if k in feats:
            feat_aligned[k] = feats[k].reindex(index=t_idx, columns=valid_syms).fillna(0).values.astype(np.float32)

    # Trend values for direction gating
    if "trend_pct" in feats:
        trend_arr = feats["trend_pct"].reindex(index=t_idx, columns=valid_syms).fillna(0).values
    else:
        trend_arr = np.ones((n_t, n_s), dtype=np.float32)

    # Predict per-symbol (avoids 1.6M-row MultiIndex entirely)
    result = np.zeros((n_t, n_s), dtype=np.float32)
    n_feat_keys = len(feat_aligned)
    n_cols = n_feat_keys + len(mkt_cols)

    for j, sym in enumerate(valid_syms):
        # Build X for this symbol: (n_t, n_feat_keys + n_mkt_cols)
        x_parts = []
        for k in keys:
            if k in feat_aligned:
                x_parts.append(feat_aligned[k][:, j:j+1])
        if x_parts:
            x_feat = np.hstack(x_parts)  # (n_t, n_feat_keys)
        else:
            x_feat = np.zeros((n_t, 0), dtype=np.float32)
        X_sym = np.hstack([x_feat, mg])  # (n_t, n_cols)
        X_sym_df = pd.DataFrame(X_sym, columns=keys[:x_feat.shape[1]] + mkt_cols)

        p_up_sym = np.zeros(n_t, dtype=np.float32)
        if model_up and model_up.model:
            # 1. Fitted prediction (fallback)
            preds = model_up.predict_proba(X_sym_df)
            
            # 2. Overlay OOF predictions where available
            if arr_oof_up is not None:
                oof_col = arr_oof_up[:, j]
                valid_oof = ~np.isnan(oof_col)
                if valid_oof.any():
                    # OOF is already calibrated
                    preds[valid_oof] = oof_col[valid_oof]
            p_up_sym = preds

        p_dn_sym = np.zeros(n_t, dtype=np.float32)
        if model_dn and model_dn.model:
            # 1. Fitted prediction
            preds = model_dn.predict_proba(X_sym_df)
            
            # 2. Overlay OOF predictions
            if arr_oof_dn is not None:
                oof_col = arr_oof_dn[:, j]
                valid_oof = ~np.isnan(oof_col)
                if valid_oof.any():
                    preds[valid_oof] = oof_col[valid_oof]
            p_dn_sym = preds

        result[:, j] = np.where(trend_arr[:, j] > 0, p_up_sym, p_dn_sym)

    res_df = pd.DataFrame(result, index=t_idx, columns=valid_syms)
    res_df = res_df.reindex(columns=syms).fillna(0.0)
    return res_df

def _build_pred_X_window(feats, mkt_gates, cfg, t_idx, syms, feature_key="exh_feature_keys"):
    tprint(f"Entering function: _build_pred_X_window in training.py")
    X_parts = []
    keys = cfg.get(feature_key, [])
    for k in keys:
        if k in feats:
            X_parts.append(feats[k].reindex(index=t_idx, columns=syms).stack(future_stack=True).rename(k))
    Xp = pd.concat(X_parts, axis=1)
    Xp.index.names = ["ts", "symbol"]
    mg = mkt_gates.reindex(t_idx)[["mkt_ret24h", "mkt_ret6h", "mkt_trend", "mkt_rv", "G_VOL", "G_TREND"]]
    for col in mg.columns:
        Xp[col] = Xp.index.get_level_values("ts").map(mg[col])
    return Xp.fillna(0)

def build_hourly_training_set_and_weights(
    panel, feats, mkt_gates, cfg, syms, ts_end, p_exh_hist, H, model_kind,
    trend_filter=None, feature_key=None, extra_feature_keys=None,
    label_method="atr", fixed_tp=0.05, fixed_sl=0.025, side="long",
    _cached_cand_mask=None, _cached_tb=None
):
    tprint(f"Entering function: build_hourly_training_set_and_weights in training.py")
    c = panel["close"]
    idx = c.index

    if _cached_tb is not None:
        tb_labels, tb_returns = _cached_tb
    elif label_method == "triple_barrier":
        # Dynamic Barrier Logic
        if "atr_pct" in feats:
            atr_pct = feats["atr_pct"]

            window_size = 24 * 30

            tprint("Computing dynamic barriers...")
            atr_base = atr_pct.rolling(window_size, min_periods=24).median()
            atr_std = atr_pct.rolling(window_size, min_periods=24).std()

            z_score = (atr_pct - atr_base) / (atr_std + 1e-12)

            b_pct_vals = scaled_atr_pct(
                atr_pct.values,
                z_score.values,
                atr_base.values,
                z_max=3.0,
                lo=0.03,
                hi=0.06
            )

            barrier_pct = pd.DataFrame(b_pct_vals, index=atr_pct.index, columns=atr_pct.columns)

            tp_df = barrier_pct
            sl_df = 0.5 * barrier_pct

            tprint(f"Labeling: Dynamic Triple Barrier (Mean TP={tp_df.mean().mean():.4f}, Side={side})")

            tb_labels, tb_returns = compute_triple_barrier_labels(
                panel, tp_df, sl_df, H, side=side
            )
        else:
            tprint("Warning: atr_pct not found for dynamic barriers. Falling back to fixed.")
            tprint(f"Labeling: Fixed Triple Barrier (TP={fixed_tp}, SL={fixed_sl}, Side={side})")
            tb_labels, tb_returns = compute_triple_barrier_labels(
                panel, fixed_tp, fixed_sl, H, side=side
            )

    else:
        # Default ATR logic
        k_sl = cfg.get("train_k_sl", 2.0)
        k_pt = cfg.get("train_k_pt", 2.0)
        k_tp = cfg.get("train_k_tp", 1.0)

        if "atr_pct" in feats:
            atr_df = feats["atr_pct"]
        else:
            tprint("Warning: atr_pct not found, using default 1% ATR for labeling")
            atr_df = pd.DataFrame(0.01, index=c.index, columns=c.columns)

        tb_labels, tb_returns = compute_trailing_atr_labels(
            panel, atr_df,
            k_sl=k_sl, k_pt=k_pt, k_tp=k_tp,
            horizon_hours=H
        )

    if _cached_cand_mask is not None:
        cand_mask = _cached_cand_mask
    else:
        cand_mask = select_trade_candidates_vectorized(panel, feats, pct=cfg["trade_extreme_pct"], metric=cfg["trade_deviation_metric"])
    if cand_mask is None:
        tprint("No candidates mask returned.")
        return None, None, None, None, None
    tprint(f"Candidates found: {cand_mask.sum().sum()}")

    ts_start = ts_end - pd.Timedelta(hours=int(cfg["train_lookback_hours"]))
    # Slice to time window first, then apply subsample filter
    ts_end_adj = ts_end - pd.Timedelta(hours=H+8)
    window_cand = cand_mask.loc[(cand_mask.index >= ts_start) & (cand_mask.index <= ts_end_adj)]
    # Subsample: keep only every 5th hour
    window_cand = window_cand[window_cand.index.hour % 5 == 0]

    if feature_key:
        feat_keys = cfg.get(feature_key, [])
    else:
        feat_keys = cfg.get("causal_cols", [])

    if extra_feature_keys:
        # Add extra keys, preserving uniqueness
        feat_keys = list(set(feat_keys) | set(extra_feature_keys))

    # --- Vectorized event extraction using numpy ---
    valid_syms = [s for s in syms if s in window_cand.columns and s in tb_labels.columns]
    if not valid_syms or window_cand.empty:
        tprint("No rows generated for training set.")
        return None, None, None, None, None, None

    sub_mask = window_cand[valid_syms]
    rows_idx, cols_idx = np.where(sub_mask.values)
    tprint(f"Candidate events: {len(rows_idx)}")
    if len(rows_idx) == 0:
        tprint("No rows generated for training set.")
        return None, None, None, None, None, None

    event_ts = sub_mask.index[rows_idx]
    event_sym = np.array(valid_syms)[cols_idx]
    entry_ts = event_ts + pd.Timedelta(hours=1)

    # Filter: entry_ts must be in tb_labels index
    entry_valid = entry_ts.isin(tb_labels.index)
    event_ts = event_ts[entry_valid]
    event_sym = event_sym[entry_valid]
    entry_ts = event_ts + pd.Timedelta(hours=1)

    if len(event_ts) == 0:
        tprint("No rows generated for training set.")
        return None, None, None, None, None, None

    # Trend filter
    if trend_filter and "trend_pct" in feats:
        trend_vals = _fast_lookup(feats["trend_pct"], event_ts, event_sym)
        trend_vals = np.nan_to_num(trend_vals, nan=0.0)
        trend_dir = np.sign(trend_vals)
        if trend_filter == "up":
            keep = trend_dir > 0
        else:
            keep = trend_dir <= 0
        event_ts = event_ts[keep]
        event_sym = event_sym[keep]
        entry_ts = event_ts + pd.Timedelta(hours=1)

    if len(event_ts) == 0:
        tprint("No rows generated for training set.")
        return None, None, None, None, None, None

    tprint(f"Events after trend filter: {len(event_ts)}")

    # --- Fast numpy positional lookups (avoid stack/reindex) ---
    # Extract TB labels/returns at entry time
    lbl_vals = _fast_lookup(tb_labels, entry_ts, event_sym)
    ret_vals = _fast_lookup(tb_returns, entry_ts, event_sym)

    # PnL computation
    if label_method == "triple_barrier":
        pnl = ret_vals
    else:
        if "trend_pct" in feats:
            tv = _fast_lookup(feats["trend_pct"], event_ts, event_sym)
            tv = np.nan_to_num(tv, nan=0.0)
        else:
            tv = np.ones(len(event_ts), dtype=np.float32)
        td = np.sign(tv)
        td[td == 0] = 1.0
        if model_kind == "tf":
            trade_dir = td
        elif model_kind == "mr":
            trade_dir = -td
        else:
            trade_dir = np.ones(len(event_ts))
        pnl = ret_vals * trade_dir

    # Quantile-based labels: top 30% -> 1, bottom 30% -> 0, middle 40% dropped
    q_lo = float(cfg.get("label_quantile_lo", 0.30))
    q_hi = float(cfg.get("label_quantile_hi", 0.70))
    pnl_lo = np.quantile(pnl[np.isfinite(pnl)], q_lo) if np.sum(np.isfinite(pnl)) > 10 else 0.0
    pnl_hi = np.quantile(pnl[np.isfinite(pnl)], q_hi) if np.sum(np.isfinite(pnl)) > 10 else 0.0
    keep_mask = (pnl <= pnl_lo) | (pnl >= pnl_hi)
    tprint(f"Quantile labels: lo_thr={pnl_lo:.6f}, hi_thr={pnl_hi:.6f}, "
           f"kept={keep_mask.sum()}/{len(pnl)} ({keep_mask.mean()*100:.0f}%)")

    # Apply mask — drop ambiguous middle samples
    event_ts = event_ts[keep_mask]
    event_sym = event_sym[keep_mask]
    entry_ts = event_ts + pd.Timedelta(hours=1)
    pnl = pnl[keep_mask]
    lbl_vals = lbl_vals[keep_mask]
    ret_vals = ret_vals[keep_mask]

    if len(event_ts) == 0:
        tprint("No rows after quantile filtering.")
        return None, None, None, None, None, None

    y_hard = (pnl >= pnl_hi).astype(np.float32)
    tprint(f"Hard label dist: 0={int((y_hard==0).sum())} ({(y_hard==0).mean()*100:.1f}%), "
           f"1={int((y_hard==1).sum())} ({(y_hard==1).mean()*100:.1f}%)")

    # Soft labels from path quality: q = MFEtp - c*MAEsl; p = sigmoid(k*q)
    # Then blend with dynamic alpha(q): y* = (1-alpha)*y + alpha*p
    # NOTE: alpha_max capped at 0.15 to prevent soft labels from pushing
    # true positives below 0.5 (which ModelRace uses as binarization threshold).
    use_soft_labels = bool(cfg.get("label_use_soft", True))
    if use_soft_labels:
        c_soft = float(cfg.get("label_soft_c", 1.0))
        k_soft = float(cfg.get("label_soft_k", 2.0))
        alpha_min = float(cfg.get("label_soft_alpha_min", 0.02))
        alpha_max = float(cfg.get("label_soft_alpha_max", 0.15))
        alpha_s = float(cfg.get("label_soft_alpha_s", 3.0))
        q0 = float(cfg.get("label_soft_q0", 0.5))

        if "mfe_4h" in feats:
            mfe_raw = np.nan_to_num(_fast_lookup(feats["mfe_4h"], event_ts, event_sym), nan=0.0)
        else:
            mfe_raw = np.maximum(pnl, 0.0)
        if "mae_4h" in feats:
            mae_raw = np.nan_to_num(_fast_lookup(feats["mae_4h"], event_ts, event_sym), nan=0.0)
        else:
            mae_raw = np.maximum(-pnl, 0.0)

        if "atr_pct" in feats:
            tp_scale = np.nan_to_num(_fast_lookup(feats["atr_pct"], event_ts, event_sym), nan=0.02)
        else:
            tp_scale = np.full(len(event_ts), 0.02, dtype=np.float32)
        tp_scale = np.clip(np.abs(tp_scale), 1e-4, None)
        sl_scale = np.clip(0.5 * tp_scale, 1e-4, None)

        mfe_tp = np.maximum(mfe_raw, 0.0) / tp_scale
        mae_sl = np.maximum(mae_raw, 0.0) / sl_scale
        q = mfe_tp - c_soft * mae_sl
        p_soft = _sigmoid(k_soft * q)
        alpha = alpha_min + (alpha_max - alpha_min) * _sigmoid(alpha_s * (np.abs(q) - q0))
        alpha = np.clip(alpha, 0.0, 1.0)
        y_bin = ((1.0 - alpha) * y_hard) + (alpha * p_soft)
        y_bin = np.clip(y_bin, 0.0, 1.0).astype(np.float32)
    else:
        y_bin = y_hard

    # Base weight from move magnitude
    pa = np.abs(np.nan_to_num(_fast_lookup(feats["ret24h"], event_ts, event_sym), nan=0.0))
    w_base = np.log1p(pa)

    # Mild class-balance multiplier (inverse-freq with sqrt exponent + hard cap)
    p_pos = float(np.mean(y_bin)) if len(y_bin) else 0.5
    p_pos = float(np.clip(p_pos, 1e-4, 1 - 1e-4))
    w1 = (0.5 / p_pos) ** 0.5
    w0 = (0.5 / (1.0 - p_pos)) ** 0.5
    w_class = np.where(y_bin >= 0.5, w1, w0)
    w_class = np.clip(w_class, 0.85, 1.25)

    # Consensus weight from geometry votes (timeouts ignored)
    n_tp = np.nan_to_num(_fast_lookup(feats.get("__geom_n_tp__", pd.DataFrame(0, index=tb_labels.index, columns=tb_labels.columns)), event_ts, event_sym), nan=0.0)
    n_sl = np.nan_to_num(_fast_lookup(feats.get("__geom_n_sl__", pd.DataFrame(0, index=tb_labels.index, columns=tb_labels.columns)), event_ts, event_sym), nan=0.0)
    n_res = n_tp + n_sl
    c_cons = (n_tp - n_sl) / np.maximum(1.0, n_res)
    A = float(cfg.get("consensus_amp", 0.25))
    k_cons = float(cfg.get("consensus_k", 2.0))
    w_consensus = 1.0 + A * np.tanh(k_cons * c_cons)

    beta = float(cfg.get("consensus_beta", 0.20))
    w_base = _winsorize_and_unit_mean(w_base, clip_min=0.75, clip_max=1.25)
    w_class = _winsorize_and_unit_mean(w_class, clip_min=0.75, clip_max=1.25)
    w_consensus = _winsorize_and_unit_mean(w_consensus, clip_min=0.75, clip_max=1.25)

    w_mix = ((1.0 - beta) * (w_base * w_class)) + (beta * w_consensus)
    p95 = np.nanpercentile(w_mix, 95) if len(w_mix) else 1.0
    w_mix = np.clip(w_mix, 0.0, max(p95, 1e-6))
    w_mix = _normalize_cross_sectional(event_ts, w_mix)
    w_mix = w_mix / max(np.nanmean(w_mix), 1e-12)
    weights_raw = w_mix.astype(np.float32)

    # Build feature DataFrame
    # event_ts is a DatetimeIndex, event_sym is a numpy array
    ts_arr = event_ts.values if hasattr(event_ts, 'values') else event_ts
    sym_arr = event_sym.values if hasattr(event_sym, 'values') else event_sym
    parts = {"ts": ts_arr, "symbol": sym_arr, "y_bin": y_bin, "y_ret": pnl.astype(np.float32), "w": weights_raw.astype(np.float32)}

    # Store barrier_pct for risk-adjusted meta model target
    if "atr_pct" in feats:
        barrier_vals = _fast_lookup(feats["atr_pct"], event_ts, event_sym)
        barrier_vals = np.nan_to_num(barrier_vals, nan=0.02).astype(np.float32)
        parts["__barrier_pct__"] = np.clip(barrier_vals, 0.005, None)

    parts["__n_tp__"] = n_tp.astype(np.float32)
    parts["__n_sl__"] = n_sl.astype(np.float32)
    parts["__n_res__"] = n_res.astype(np.float32)
    parts["__w_consensus__"] = w_consensus.astype(np.float32)

    # p_exh_lag1
    lag_ts = event_ts - pd.Timedelta(hours=1)
    if p_exh_hist is not None:
        parts["p_exh_lag1"] = np.nan_to_num(_fast_lookup(p_exh_hist, lag_ts, event_sym), nan=0.0).astype(np.float32)
    else:
        parts["p_exh_lag1"] = np.zeros(len(event_ts), dtype=np.float32)

    # Feature columns — fast lookup
    for k in feat_keys:
        if k == "p_exh_lag1":
            continue
        if k in feats:
            parts[k] = _fast_lookup(feats[k], event_ts, event_sym)

    # Market gates
    parts["G_VOL"] = mkt_gates["G_VOL"].reindex(event_ts).values
    parts["G_TREND"] = mkt_gates["G_TREND"].reindex(event_ts).values

    df = pd.DataFrame(parts)

    # Drop constant market gates (fix for Low Variation warning)
    for g in ["G_VOL", "G_TREND"]:
        if g in df.columns and df[g].nunique() <= 1:
            if df[g].std() < 1e-9:
                df.drop(columns=[g], inplace=True)
    # Drop rows only where critical columns are NaN; fill feature NaNs with 0
    critical_cols = ["ts", "symbol", "y_bin", "y_ret", "w"]
    df = df.dropna(subset=[c for c in critical_cols if c in df.columns])
    feat_cols = [c for c in df.columns if c not in critical_cols]
    if feat_cols:
        df[feat_cols] = df[feat_cols].fillna(0)
    if df.empty:
        tprint("No rows generated for training set.")
        return None, None, None, None, None, None
    tprint(f"Final training set size: {len(df)}")

    # Build label time ranges for uniqueness weighting
    entry_times = df["ts"].values
    exit_times = entry_times + pd.Timedelta(hours=H)  # H is the horizon
    label_times = build_label_time_ranges(
        pd.DatetimeIndex(entry_times),
        pd.DatetimeIndex(exit_times)
    )
    
    # Extract time grid from panel for accurate uniqueness (Improvement #3)
    # This ensures we measure uniqueness on the actual price bars, not just event boundaries.
    ts_min = pd.Timestamp(label_times["t_start"].min())
    ts_max = pd.Timestamp(label_times["t_end"].max())
    full_idx = panel["close"].index
    # Ensure tz compatibility
    if full_idx.tz is not None and ts_min.tzinfo is None:
        ts_min = ts_min.tz_localize(full_idx.tz)
        ts_max = ts_max.tz_localize(full_idx.tz)
    elif full_idx.tz is None and ts_min.tzinfo is not None:
        ts_min = ts_min.tz_localize(None)
        ts_max = ts_max.tz_localize(None)
    time_grid = full_idx[(full_idx >= ts_min) & (full_idx <= ts_max)]

    # Compute sample weights with uniqueness (AFML Chapter 4)
    base_weights = df["w"].values
    returns = df["y_ret"].values
    weights = compute_sample_weights_with_uniqueness(
        label_times=label_times,
        returns=returns,
        base_weights=base_weights,
        time_grid=time_grid
    )
    
    tprint(f"Applied uniqueness weighting: mean={weights.mean():.3f}, std={weights.std():.3f}")
    df.drop(columns=["w"], inplace=True)

    # --- Regime columns for per-regime BSS/AUC reporting ---
    # 6 regime dimensions: vol_12h, vol_48h, volume_12h, volume_48h, trend_12h, trend_48h
    # Each bucketed into 3 terciles (low/mid/high)
    _regime_map = {
        "__regime_vol_12h__": "rv_12h",
        "__regime_vol_48h__": "rv_24h",         # rv_24h is closest proxy for 48h
        "__regime_volume_12h__": "vol_z_base",   # volume z-score (short horizon)
        "__regime_volume_48h__": "vol_z24_base", # volume z-score (longer horizon)
        "__regime_trend_12h__": "ret6h",         # 6h return as 12h trend proxy
        "__regime_trend_48h__": "trend_pct_base",# trend pct as 48h trend proxy
    }
    for regime_col, src_col in _regime_map.items():
        if src_col in df.columns:
            vals = df[src_col].values.astype(np.float64)
            valid = np.isfinite(vals)
            if valid.sum() > 10:
                try:
                    terciles = np.nanpercentile(vals[valid], [33.3, 66.7])
                    buckets = np.full(len(vals), 1, dtype=np.int8)  # default mid
                    buckets[vals <= terciles[0]] = 0  # low
                    buckets[vals >= terciles[1]] = 2  # high
                    df[regime_col] = buckets
                except Exception:
                    df[regime_col] = np.int8(1)
            else:
                df[regime_col] = np.int8(1)
        else:
            # Try to look up from feats directly
            if src_col in feats:
                raw_vals = _fast_lookup(feats[src_col], event_ts, event_sym)
                raw_vals = np.nan_to_num(raw_vals, nan=0.0).astype(np.float64)
                if len(raw_vals) > 10:
                    try:
                        terciles = np.nanpercentile(raw_vals, [33.3, 66.7])
                        buckets = np.full(len(raw_vals), 1, dtype=np.int8)
                        buckets[raw_vals <= terciles[0]] = 0
                        buckets[raw_vals >= terciles[1]] = 2
                        df[regime_col] = buckets
                    except Exception:
                        df[regime_col] = np.int8(1)
                else:
                    df[regime_col] = np.int8(1)
            else:
                df[regime_col] = np.int8(1)

    # Preserve raw meta feature columns BEFORE interaction toggles destroy them.
    # At inference time (engine.py), the meta model receives raw feature names,
    # so we must train on raw names too. Prefix with __meta_raw__ to avoid
    # collision with toggled columns and to survive drop_raw=True.
    meta_keys_cfg = cfg.get("meta_feature_keys", [])
    for mk in meta_keys_cfg:
        if mk in df.columns:
            df[f"__meta_raw__{mk}"] = df[mk].values

    df = apply_interaction_toggles(df, feat_keys, ["G_VOL", "G_TREND"], drop_raw=cfg["drop_raw_causal"])
    y_bin = df.pop("y_bin").values.astype(np.float32)
    y_ret = df.pop("y_ret").values.astype(np.float32)

    X_out = df.drop(columns=["ts", "symbol"], errors="ignore").astype(np.float32)
    X_out.index = df.index

    df_meta = df[["ts", "symbol"]] if "ts" in df.columns else pd.DataFrame(index=df.index)

    return X_out, y_bin, y_ret, list(X_out.columns), weights, df_meta

def train_spike_anatomy_model(panel, feats, mkt_gates, cfg, syms, ts_end, _cached_cand_mask=None, mode=None):
    tprint(f"Entering function: train_spike_anatomy_model in training.py")
    if _cached_cand_mask is not None:
        cand_mask = _cached_cand_mask
    else:
        cand_mask = select_trade_candidates_vectorized(panel, feats, pct=cfg["trade_extreme_pct"], metric=cfg["trade_deviation_metric"])
    if cand_mask is None: return None

    # Slice to time window FIRST (fast index slice), then filter
    ts_start = ts_end - pd.Timedelta(hours=int(cfg["train_lookback_hours"]))
    window_mask = cand_mask.loc[(cand_mask.index >= ts_start) & (cand_mask.index <= ts_end)]
    if window_mask.empty or not window_mask.any(axis=None):
        tprint("Spike Anatomy: no candidates in window.")
        return None

    # Filter by mode (best/worst) if requested
    metric_name = cfg["trade_deviation_metric"]
    if mode in ["best", "worst"] and metric_name in feats:
        metric_df = feats[metric_name].reindex(index=window_mask.index, columns=window_mask.columns)
        if mode == "best":
            mode_mask = metric_df > 0
        else:
            mode_mask = metric_df < 0
        window_mask = window_mask & mode_mask

    if window_mask.empty or not window_mask.any(axis=None):
        tprint(f"Spike Anatomy ({mode}): no candidates in window.")
        return None

    keys = cfg.get("spike_feature_keys", [])
    available_keys = [k for k in keys if k in feats]
    if not available_keys:
        tprint("No spike features available.")
        return None

    # Restrict to valid syms
    valid_syms = [s for s in syms if s in window_mask.columns]
    sub_mask = window_mask[valid_syms]

    # Use numpy to find True positions (much faster than stack)
    rows_idx, cols_idx = np.where(sub_mask.values)
    tprint(f"Spike Anatomy events: {len(rows_idx)}")
    if len(rows_idx) == 0:
        return None

    event_ts_vals = sub_mask.index[rows_idx]
    event_sym_vals = np.array(valid_syms)[cols_idx]

    # Extract features using fast numpy positional indexing
    # Optimization: Precompute indices once
    # Assumes all feature DataFrames share the same index/columns
    # Validate strict alignment for optimized lookup
    # We rely on all features having identical Index/Columns implies row_idx/col_idx are valid for all.
    # Ref: panel["close"] is the ground truth for alignment
    ref_df = panel["close"]
    
    # Make a shallow copy of feats to avoid side effects if we modify
    feats = feats.copy()

    for k in available_keys:
        if k in feats:
            f_df = feats[k]
            if (len(f_df.index) != len(ref_df.index) or 
                not f_df.index.equals(ref_df.index) or 
                not f_df.columns.equals(ref_df.columns)):
                 
                 tprint(f"Warning: Feature {k} structure mismatch in Spike Anatomy. Aligning to panel...")
                 # Reindex to match panel (fill NaNs with 0 to prevent crash)
                 feats[k] = f_df.reindex(index=ref_df.index, columns=ref_df.columns, fill_value=0.0)
    
    # Re-fetch ref from potentially aligned features (or just use panel index for row_idx)
    # But current code uses ref_df.index.get_indexer.
    # Let's use ref_df (panel["close"]) for indexing.
    
    row_idx = ref_df.index.get_indexer(event_ts_vals)
    col_idx = ref_df.columns.get_indexer(event_sym_vals)
    
    valid = (row_idx >= 0) & (col_idx >= 0)
    
    data = {}
    for k in available_keys:
        vals = feats[k].values
        out = np.full(len(event_ts_vals), np.nan, dtype=np.float32)
        if valid.any():
            out[valid] = vals[row_idx[valid], col_idx[valid]]
        data[k] = out

    events_mi = pd.MultiIndex.from_arrays([event_ts_vals, event_sym_vals])
    df = pd.DataFrame(data, index=events_mi).dropna()
    tprint(f"Spike Anatomy dataset shape: {df.shape}")
    return df if not df.empty else None

def generate_label_datasets(panel, feats, mkt_gates, cfg, syms, ts, p_exh_hist):
    tprint(f"Entering function: generate_label_datasets in training.py")
    datasets = {}

    # Pre-compute shared expensive operations once
    tprint("Pre-computing candidate mask (shared across all steps)...")
    cached_cand_mask = select_trade_candidates_vectorized(
        panel, feats, pct=cfg["trade_extreme_pct"], metric=cfg["trade_deviation_metric"]
    )

    # Apply OOS holdout: exclude last N days from training labels
    oos_days = cfg.get("oos_holdout_days", 0)
    if oos_days > 0 and cached_cand_mask is not None:
        cutoff = ts - pd.Timedelta(days=oos_days)
        n_before = cached_cand_mask.sum().sum()
        cached_cand_mask = cached_cand_mask.loc[cached_cand_mask.index <= cutoff]
        n_after = cached_cand_mask.sum().sum()
        tprint(f"OOS holdout: excluded last {oos_days} days (cutoff={cutoff}). Candidates: {n_before} -> {n_after}")

    # 1. Spike Anatomy (2 GMM models: Best & Worst)
    for mode in ["best", "worst"]:
        spike_df = train_spike_anatomy_model(panel, feats, mkt_gates, cfg, syms, ts, _cached_cand_mask=cached_cand_mask, mode=mode)
        if spike_df is not None:
            datasets[f"spike_anatomy_{mode}"] = spike_df

    # 2. Alpha Models (MR/TF)
    trade_sides = ["long", "short"]
    kinds = ["mr", "tf"]
    horizons = cfg["label_horizons_hours"]

    # Pre-compute triple barrier labels per (H, side) — shared across MR/TF
    tb_cache = {}

    if "atr_pct" in feats:
        atr_pct_df = feats["atr_pct"]
        atr_scale_df, atr_ref = _compute_atr_scale(atr_pct_df, cfg)
        tprint(f"ATR normalization: atr_ref={atr_ref:.6f}, scale_mean={float(np.nanmean(atr_scale_df.values)):.4f}")
    else:
        atr_pct_df = None
        atr_scale_df = None

    fee_pct = float(cfg.get("label_round_trip_fee_pct", 0.5)) / 100.0
    tp_vals = [float(x) / 100.0 for x in cfg.get("label_tp_values_pct", [5.0, 3.5, 2.0])]
    sl_vals = [float(x) / 100.0 for x in cfg.get("label_sl_values_pct", [0.5, 1.0, 2.0])]
    min_net_rr = float(cfg.get("label_min_net_rr", 1.5))
    min_tp_hit = float(cfg.get("label_min_tp_hit_rate", 0.02))
    max_timeout = float(cfg.get("label_max_timeout_rate", 0.90))

    geom_cache = {}
    for H in horizons:
        for side in trade_sides:
            tprint(f"Pre-computing geometry labels H={H} side={side}...")

            if atr_scale_df is None:
                atr_scale_local = pd.DataFrame(1.0, index=panel["close"].index, columns=panel["close"].columns)
            else:
                atr_scale_local = atr_scale_df.reindex(index=panel["close"].index, columns=panel["close"].columns).fillna(1.0)

            geom_runs = []
            for tpv in tp_vals:
                for slv in sl_vals:
                    net_rr = tpv / max(slv + fee_pct, 1e-9)
                    if net_rr < min_net_rr:
                        continue
                    tp_df = np.clip(atr_scale_local * tpv, 0.001, None)
                    sl_df = np.clip(atr_scale_local * slv, 0.001, None)
                    lbl, ret = compute_triple_barrier_labels(panel, tp_df, sl_df, H, side=side)
                    tp_hit = float((lbl.values == 1).sum()) / max(1, lbl.size)
                    to_rate = float((lbl.values == 0).sum()) / max(1, lbl.size)
                    if tp_hit < min_tp_hit or to_rate > max_timeout:
                        continue
                    geom_runs.append((lbl, ret, tpv, slv))

            if not geom_runs:
                tprint(f"No valid geometry for H={H} side={side}; using fallback.")
                lbl, ret = compute_triple_barrier_labels(panel, 0.05, 0.01, H, side=side)
                geom_runs = [(lbl, ret, 0.05, 0.01)]

            labels_stack = np.stack([x[0].values for x in geom_runs], axis=0)
            rets_stack = np.stack([x[1].values for x in geom_runs], axis=0)
            n_tp_df = pd.DataFrame(np.sum(labels_stack == 1, axis=0), index=panel["close"].index, columns=panel["close"].columns)
            n_sl_df = pd.DataFrame(np.sum(labels_stack == -1, axis=0), index=panel["close"].index, columns=panel["close"].columns)
            n_to_df = pd.DataFrame(np.sum(labels_stack == 0, axis=0), index=panel["close"].index, columns=panel["close"].columns)

            agg_ret = np.nanmean(rets_stack, axis=0)
            agg_lbl = np.where(n_tp_df.values > n_sl_df.values, 1, np.where(n_sl_df.values > n_tp_df.values, -1, 0)).astype(np.int8)
            tb_labels = pd.DataFrame(agg_lbl, index=panel["close"].index, columns=panel["close"].columns)
            tb_returns = pd.DataFrame(agg_ret, index=panel["close"].index, columns=panel["close"].columns)
            tb_cache[(H, side)] = (tb_labels, tb_returns)
            geom_cache[(H, side)] = {"n_tp": n_tp_df, "n_sl": n_sl_df, "n_to": n_to_df, "n_geom": len(geom_runs)}

    if geom_cache:
        any_key = next(iter(geom_cache.keys()))
        feats["__geom_n_tp__"] = geom_cache[any_key]["n_tp"]
        feats["__geom_n_sl__"] = geom_cache[any_key]["n_sl"]
        feats["__geom_n_to__"] = geom_cache[any_key]["n_to"]

    for side in trade_sides:
        for k in kinds:
            # Determine Candidate Filter based on Strategy
            cand_filter = "unknown"
            if side == "long":
                if k == "mr": cand_filter = "worst" # ret < 0
                else: cand_filter = "best"          # ret > 0
            else: # short
                if k == "mr": cand_filter = "best"  # ret > 0
                else: cand_filter = "worst"         # ret < 0

            trend_filter = "up" if cand_filter == "best" else "down"

            feat_key = "tf_feature_keys" if k == "tf" else "mr_feature_keys"

            fixed_tp = 0.05
            fixed_sl = 0.025

            for H in horizons:
                tprint(f"Generating labels for {side} {k} ({cand_filter}) H={H}...")

                if (H, side) in geom_cache:
                    feats["__geom_n_tp__"] = geom_cache[(H, side)]["n_tp"]
                    feats["__geom_n_sl__"] = geom_cache[(H, side)]["n_sl"]
                    feats["__geom_n_to__"] = geom_cache[(H, side)]["n_to"]

                # Optimization: We include meta keys here so they are present in the dataframe
                # for the meta model later. However, we must filter them out when training
                # the alpha model itself (in train_models_from_artifacts).
                X, y, y_ret, cols, w, meta_idx = build_hourly_training_set_and_weights(
                    panel, feats, mkt_gates, cfg, syms, ts, p_exh_hist, H, k,
                    trend_filter=trend_filter, feature_key=feat_key,
                    extra_feature_keys=cfg.get("meta_feature_keys", []),
                    label_method="triple_barrier",
                    fixed_tp=fixed_tp, fixed_sl=fixed_sl, side=side,
                    _cached_cand_mask=cached_cand_mask,
                    _cached_tb=tb_cache[(H, side)]
                )

                if X is not None:
                    df_out = X.copy()
                    df_out["__y_bin__"] = y
                    df_out["__y_ret__"] = y_ret
                    df_out["__w__"] = w

                    if meta_idx is not None:
                        df_out["__ts__"] = meta_idx["ts"]
                        df_out["__symbol__"] = meta_idx["symbol"]

                    datasets[f"train_{side}_{k}_{H}"] = df_out

    # 3. Exhaustion Models
    lookback = cfg["exh_train_lookback_hours"]
    directions = ["up", "down"]
    for d in directions:
        tprint(f"Generating exhaustion training set for {d}...")
        X, y, w, cols = build_exhaustion_Xy(panel, feats, mkt_gates, cfg, ts, lookback, syms, trend_filter=d)
        if X is not None:
            df_out = X.copy()
            df_out["__y__"] = y
            df_out["__w__"] = w
            datasets[f"exh_{d}"] = df_out.reset_index()

    # 4. Specialist Models (Trap & Gamma)
    tprint("Generating specialist training sets...")
    trap_df = build_trap_dataset(panel, feats, cfg, syms)
    if trap_df is not None:
        datasets["trap_model"] = trap_df

    gamma_df = build_gamma_dataset(panel, feats, cfg, syms)
    if gamma_df is not None:
        datasets["gamma_model"] = gamma_df

    return datasets

def train_specialist_models(panel, feats, mkt_gates, cfg, syms, ts_end):
    """
    Train Trap and Gamma specialist models.
    
    Args:
        panel: Dictionary with OHLCV DataFrames
        feats: Dictionary of feature DataFrames
        mkt_gates: Market regime gates DataFrame
        cfg: Configuration dictionary
        syms: List of symbols to train on
        ts_end: End timestamp for training window
    
    Returns:
        Dictionary with trained specialist models
    """
    from .trap_specialist import train_trap_specialist
    from .gamma_specialist import train_gamma_specialist
    
    tprint("=" * 60)
    tprint("TRAINING SPECIALIST MODELS")
    tprint("=" * 60)
    
    specialist_models = {}
    
    # 1. Trap Specialist (GMM-based quality filter)
    try:
        trap_model = train_trap_specialist(panel, feats, cfg, syms, ts_end)
        specialist_models["trap_model"] = trap_model
    except Exception as e:
        tprint(f"ERROR: Trap Specialist training failed: {e}")
        specialist_models["trap_model"] = None
    
    # 2. Gamma Specialist (ExtraTrees regression for volatility)
    try:
        gamma_model = train_gamma_specialist(panel, feats, cfg, syms, ts_end)
        specialist_models["gamma_model"] = gamma_model
    except Exception as e:
        tprint(f"ERROR: Gamma Specialist training failed: {e}")
        specialist_models["gamma_model"] = None
    
    tprint("=" * 60)
    tprint("SPECIALIST TRAINING COMPLETE")
    tprint("=" * 60)
    
    return specialist_models


def _compute_and_print_meta_metrics(y_true, y_pred, name, groups=None, y_raw_ret=None):
    """Print meta model diagnostics with economically meaningful metrics.

    IC_target: rank corr of pred vs transformed target (what model optimizes)
    IC_raw:    rank corr of pred vs raw returns (what trading PnL depends on)
    R²:        OOS R-squared of pred vs raw returns (variance explained)
    WinRate@k: fraction with raw_ret > 0 in top k% by pred score
    AvgRet@k:  mean raw return in top k% (bps)
    CVaR@k:    mean of worst 20% raw returns in top k% (downside tail)
    Sharpe@10: mean/std of per-timestamp Top10 avg returns (signal stability)
    GtP@10:    Gain-to-Pain = sum(gains) / sum(|losses|) for Top10 per-ts returns
    Tail sanity: top10 vs bot10 realized raw returns (detects sign flips)
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    ic_target = ic_cross_sectional(y_pred, y_true, groups=groups)
    ic_raw = ic_cross_sectional(y_pred, y_raw_ret, groups=groups) if y_raw_ret is not None else float('nan')

    if groups is not None:
        trades_day_10 = avg_trades_per_day(y_pred, 0.10, np.asarray(groups))
        trades_day_30 = avg_trades_per_day(y_pred, 0.30, np.asarray(groups))
    else:
        trades_day_10 = float(len(y_pred) * 0.10)
        trades_day_30 = float(len(y_pred) * 0.30)

    # Top10 and Bot10 masks for tail sanity
    m_top10 = topk_mask(y_pred, 0.10, groups=groups)
    m_bot10 = topk_mask(-y_pred, 0.10, groups=groups)  # lowest predictions

    has_raw = y_raw_ret is not None
    r = np.asarray(y_raw_ret, dtype=float) if has_raw else y_true

    # R² of predictions vs raw returns (OOS variance explained)
    if has_raw:
        ss_res = float(np.sum((y_raw_ret - y_pred) ** 2))
        ss_tot = float(np.sum((y_raw_ret - np.mean(y_raw_ret)) ** 2))
        r_squared = 1.0 - ss_res / max(ss_tot, 1e-12)
    else:
        r_squared = float('nan')

    # WinRate@k: fraction with raw_ret > 0 (economically meaningful)
    base_win = float(np.mean(r > 0))
    win10 = float(np.mean(r[m_top10] > 0)) if m_top10.any() else float('nan')
    win40_m = topk_mask(y_pred, 0.40, groups=groups)
    win40 = float(np.mean(r[win40_m] > 0)) if win40_m.any() else float('nan')

    # AvgRet@k in bps (×10000)
    avg_ret_top10 = float(np.mean(r[m_top10])) * 10000 if m_top10.any() else float('nan')
    avg_ret_bot10 = float(np.mean(r[m_bot10])) * 10000 if m_bot10.any() else float('nan')

    # CVaR@10: mean of worst 20% of raw returns in top decile
    if m_top10.any() and has_raw:
        top10_rets = r[m_top10]
        n_worst = max(1, int(0.20 * len(top10_rets)))
        cvar10 = float(np.mean(np.sort(top10_rets)[:n_worst])) * 10000
    else:
        cvar10 = float('nan')

    # Sharpe@10 and Gain-to-Pain@10: per-timestamp Top10 avg return stability
    sharpe10 = float('nan')
    gtp10 = float('nan')
    n_ts_top10 = 0
    if m_top10.any() and has_raw and groups is not None:
        g = np.asarray(groups)
        ts_rets = []
        for t in np.unique(g):
            tm = (g == t) & m_top10
            if tm.any():
                ts_rets.append(float(np.mean(r[tm])))
        if len(ts_rets) >= 5:
            ts_arr = np.array(ts_rets)
            n_ts_top10 = len(ts_arr)
            mu_ts = float(np.mean(ts_arr))
            std_ts = float(np.std(ts_arr, ddof=1))
            sharpe10 = mu_ts / max(std_ts, 1e-12)
            gains = float(np.sum(ts_arr[ts_arr > 0]))
            pains = float(np.sum(np.abs(ts_arr[ts_arr < 0])))
            gtp10 = gains / max(pains, 1e-12)

    # Target-space top10 mean (for debugging)
    tgt_top10 = float(np.mean(y_true[m_top10])) if m_top10.any() else float('nan')
    pred_top10 = float(np.mean(y_pred[m_top10])) if m_top10.any() else float('nan')

    tprint(
        f"  {name}: IC_target={ic_target:.4f} IC_raw={ic_raw:.4f} R²={r_squared:.6f} "
        f"WinRate@10={win10:.4f} WinRate@40={win40:.4f} (base={base_win:.4f}) "
        f"AvgTrades/Day@10={trades_day_10:.2f}"
    )
    tprint(
        f"  {name} Top10: AvgRet={avg_ret_top10:+.2f}bps CVaR={cvar10:+.2f}bps "
        f"Sharpe={sharpe10:.4f} GtP={gtp10:.4f} (n_ts={n_ts_top10})"
    )
    tprint(
        f"  {name} Bot10: AvgRet={avg_ret_bot10:+.2f}bps  "
        f"(Top10-Bot10 spread={avg_ret_top10 - avg_ret_bot10:+.2f}bps)"
    )

def train_models_from_artifacts(datasets, cfg):
    tprint(f"Entering function: train_models_from_artifacts in training.py")
    directions = ["up", "down"]
    kinds = ["mr", "tf"]
    final_models = {}

    alpha_gate_results = []
    meta_gate_results = []

    # 1. Train Spike Models (Best & Worst)
    spike_models = {}
    for mode in ["best", "worst"]:
        key = f"spike_anatomy_{mode}"
        if key in datasets:
            tprint(f"Training Spike Model ({mode})...")
            df_spike = datasets[key]
            # Ensure numeric-only, drop any index/meta columns
            df_spike_num = df_spike.select_dtypes(include=[np.number])
            if isinstance(df_spike_num.index, pd.MultiIndex):
                df_spike_num = df_spike_num.reset_index(drop=True)
            df_spike_num = df_spike_num.dropna()
            # Drop near-zero-variance columns that cause singular covariance
            col_std = df_spike_num.std()
            keep_cols = col_std[col_std > 1e-6].index
            df_spike_num = df_spike_num[keep_cols]
            n_comp = min(4, max(1, len(df_spike_num) // 100))
            tprint(f"Spike GMM ({mode}): {len(df_spike_num)} samples, {df_spike_num.shape[1]} features, {n_comp} components")
            if len(df_spike_num) >= 50 and df_spike_num.shape[1] >= 2:
                # Standardize before fitting to avoid ill-conditioned covariance
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(df_spike_num)
                for n_try in [n_comp, max(1, n_comp // 2), 1]:
                    try:
                        gmm = GaussianMixture(n_components=n_try, covariance_type='diag', reg_covar=1e-2, random_state=42)
                        gmm.fit(X_scaled)
                        spike_models[mode] = {"gmm": gmm, "scaler": scaler, "columns": list(df_spike_num.columns)}
                        tprint(f"Spike GMM ({mode}) fitted with {n_try} components.")

                        # Generate OOF Scores (Log-Likelihood)
                        # We use the full set as OOF here simplistically since GMM is unsupervised density estimation
                        scores = gmm.score_samples(X_scaled)
                        
                        # Save OOF artifact
                        # Align with original index
                        df_oof = pd.DataFrame(scores, index=df_spike_num.index, columns=["score"])
                        # If index was MultiIndex, restore it? 
                        # We reset_index earlier for df_spike_num if it was MI.
                        # But typically datasets[key] comes loaded via pd.read_parquet which might have lost MI if not handled.
                        # `load_artifact_df` usually returns default range index unless instructed.
                        # Let's check datasets[key] structure.
                        # If `df_spike` had index, we want to align `df_oof` to it.
                        # df_spike_num was reset-indexed if MI. 
                        # "if isinstance(df_spike_num.index, pd.MultiIndex): df_spike_num = ..."
                        # Actually we used `df_spike_num` for fitting.
                        # Let's align `df_oof` to `df_spike` (original df with metadata).

                        # If df_spike has columns "ts", "symbol", use them as index for OOF.
                        if "ts" in df_spike.columns and "symbol" in df_spike.columns:
                            df_oof["ts"] = df_spike["ts"]
                            df_oof["symbol"] = df_spike["symbol"]
                            # Reordering
                        
                        # Save
                        run_id = datasets[key].attrs.get("run_id") 
                        # datasets loaded via `load_artifact_df` might not track run_id implicitly?
                        # `train_models_from_artifacts` doesn't know run_id from args.
                        # But `pipeline_steps` knows.
                        # We can't save here easily without run_id.
                        # Actually, we can return spike OOFs and let caller save?
                        # Or pass run_id?
                        
                        # Let's store OOF in spike_models payload?
                        spike_models[mode]["oof_scores"] = df_oof

                        break
                    except ValueError as e:
                        tprint(f"Spike GMM ({mode}) failed with {n_try} components: {e}")
                        continue

    # 1.5 Train Specialist Models & Generate OOF (Moved before Alpha Models to provide features)
    specialist_models = {
        "trap_model": None,
        "gamma_model": None,
    }

    specialist_oof_lookup = {} # key: (ts, symbol) -> dict of scores

    # Train Trap Model
    if "trap_model" in datasets:
        trap_df = datasets["trap_model"]
        tprint("Training Trap Specialist from artifact...")
        try:
            m = train_trap_from_dataset(trap_df, cfg)
            specialist_models["trap_model"] = m

            # Generate scores via direct prediction (fast; OOF CV on 8M rows is prohibitive)
            from .trap_specialist import TRAP_FEATURE_KEYS
            if m is not None and "gmm" in m and "scaler" in m:
                X_trap = trap_df[TRAP_FEATURE_KEYS].values.astype(np.float32)
                y_trap = trap_df["y_quality"].values.astype(np.float32)
                X_scaled = m["scaler"].transform(X_trap)
                cluster_labels = m["gmm"].predict(X_scaled)
                # Map clusters to quality scores using semantic ordering
                cluster_means = []
                for k in range(m["gmm"].n_components):
                    mask = cluster_labels == k
                    cluster_means.append(float(y_trap[mask].mean()) if mask.sum() > 0 else 0.0)
                trap_scores = np.array([cluster_means[l] for l in cluster_labels], dtype=np.float32)
                oof_series = pd.Series(trap_scores, index=pd.MultiIndex.from_frame(trap_df[["ts", "symbol"]]))
                specialist_oof_lookup["trap_score"] = oof_series
                tprint(f"  Trap scores generated: mean={trap_scores.mean():.3f}, std={trap_scores.std():.3f}")

        except Exception as e:
            tprint(f"Trap Specialist training failed: {e}")
            import traceback; traceback.print_exc()

    # Train Gamma Model
    if "gamma_model" in datasets:
        gamma_df = datasets["gamma_model"]
        tprint("Training Gamma Specialist from artifact...")
        try:
            m = train_gamma_from_dataset(gamma_df, cfg)
            specialist_models["gamma_model"] = m

            # Generate scores via direct prediction (fast)
            if m is not None:
                from .gamma_specialist import GAMMA_FEATURE_KEYS
                X_gamma = gamma_df[GAMMA_FEATURE_KEYS]
                gamma_scores = m.predict(X_gamma).astype(np.float32)
                oof_series = pd.Series(gamma_scores, index=pd.MultiIndex.from_frame(gamma_df[["ts", "symbol"]]))
                specialist_oof_lookup["gamma_score"] = oof_series
                tprint(f"  Gamma scores generated: mean={gamma_scores.mean():.3f}, std={gamma_scores.std():.3f}")

        except Exception as e:
            tprint(f"Gamma Specialist training failed: {e}")
            import traceback; traceback.print_exc()

    # 2. Train Alpha Models
    # directions (up/down) replaced by sides (long/short)
    trade_sides = ["long", "short"]
    kinds = ["mr", "tf"]
    final_models = {}

    for side in trade_sides:
        final_models[side] = {}
        for k in kinds:
            best_ic = -1.0; best_m = None
            horizons = cfg["label_horizons_hours"]

            for H in horizons:
                key = f"train_{side}_{k}_{H}"
                if key not in datasets: continue

                df = datasets[key]
                if df.empty or len(df) < cfg["min_train_samples"] // 4:
                    continue

                # Schema assertion: verify dataset has required columns
                required_cols = {"__y_bin__", "__y_ret__", "__w__"}
                missing = required_cols - set(df.columns)
                assert not missing, f"Dataset {key} missing columns: {missing}"

                # Inject Specialist OOF Features
                # df has ts, symbol. We can map.
                # Note: df is a copy from datasets[key], so modifying it is safe for this loop.
                # But we should ensure we don't modify the original artifact if cached.
                # Currently datasets[key] is loaded from disk once.
                # We make a copy below for training, but let's inject before.

                # Check if ts/symbol columns exist
                if "ts" in df.columns and "symbol" in df.columns:
                    mi = pd.MultiIndex.from_frame(df[["ts", "symbol"]])
                    for feat_name, oof_ser in specialist_oof_lookup.items():
                        # Map values. Fill NaN with 0.5 (neutral) or median?
                        # Trap: 0=Trap, 1=Quality. Gamma: Volatility.
                        # Use 0.0 for missing gamma? 0.0 for Trap?
                        # Trap: mean ~ 0.5. Gamma: mean ~ 0.5.
                        # Let's fill with median of OOF to avoid bias.
                        fill_val = oof_ser.median() if not oof_ser.empty else 0.0

                        # Reindex to align
                        vals = oof_ser.reindex(mi).fillna(fill_val).values
                        df[feat_name] = vals.astype(np.float32)
                        tprint(f"  Injected {feat_name} into {key} (coverage: {np.mean(~np.isnan(vals)):.1%})")

                y = df["__y_bin__"].values.astype(np.float32)
                y_ret = df["__y_ret__"].values.astype(np.float32)
                w_raw = df["__w__"].values.astype(np.float32)
                # Temper weights with sqrt to reduce n_eff collapse from skewed uniqueness weights
                # sqrt preserves relative ordering but compresses the tail
                w = np.sqrt(np.clip(w_raw, 0.0, None))

                drop_cols = ["__y_bin__", "__y_ret__", "__w__", "__ts__", "__symbol__", "__barrier_pct__"]
                X = df.drop(columns=[c for c in drop_cols if c in df.columns])

                # Filter features strictly for the Alpha Model (exclude meta-only features)
                # We need to know which feature_key was used.
                # k is "mr" or "tf"
                feat_key_name = "tf_feature_keys" if k == "tf" else "mr_feature_keys"
                allowed_keys = set(cfg.get(feat_key_name, []))

                # Also include "causal_cols" if feature_key fallback logic was used, but
                # here we know we used the explicit keys.
                # Note: build_hourly_training_set_and_weights adds gate columns G_VOL/G_TREND and p_exh_lag1.
                # We should allow those too.
                # And interaction toggles? apply_interaction_toggles creates columns like "col_G_0".
                # If "col" is in allowed_keys, "col_G_0" should be allowed.

                # Simpler approach: Filter base columns.
                # But X has interaction columns already.
                # We can't easily filter interaction columns by exact name match.
                # Heuristic: Check if base feature part of the column name is in allowed_keys.

                valid_cols = []
                # Always keep market gates/lags/specialist scores if they are standard inputs
                std_inputs = {"p_exh_lag1", "G_VOL", "G_TREND", "mkt_ret24h", "mkt_ret6h", "mkt_trend", "mkt_rv", "trap_score", "gamma_score"}

                for c in X.columns:
                    # Check exact match
                    if c in allowed_keys or c in std_inputs:
                        valid_cols.append(c)
                        continue
                    # Check interaction pattern: {col}_{gate}_0 or {col}_{gate}_1
                    # We assume gate is G_VOL or G_TREND
                    is_inter = False
                    for g in ["G_VOL", "G_TREND"]:
                        if f"_{g}_0" in c or f"_{g}_1" in c:
                             base = c.split(f"_{g}_")[0]
                             if base in allowed_keys:
                                 valid_cols.append(c)
                                 is_inter = True
                                 break
                    if is_inter: continue

                if not valid_cols:
                     tprint(f"Warning: No valid columns found for {side} {k} after filtering. Using all.")
                     valid_cols = list(X.columns)

                X = X[valid_cols]
                cols = list(X.columns)

                # Derive strategy context for logging
                # long+mr = buy dips (candidates with worst recent returns)
                # long+tf = buy momentum (candidates with best recent returns)
                # short+mr = sell rips (candidates with best recent returns)
                # short+tf = sell weakness (candidates with worst recent returns)
                if side == "long":
                    cand_filter = "bottom_ret" if k == "mr" else "top_ret"
                else:
                    cand_filter = "top_ret" if k == "mr" else "bottom_ret"

                tprint(f"Training {side}_{k} [{key}] cand={cand_filter} H={H} (n={len(X)})...")

                # --- Integrated MDI Feature Selection ---
                # Fix: Don't feed raw 300+ features to ModelRace. Select top signal first.
                tprint(f"Running MDI Feature Selection for {side} {k}...")
                
                # Base model for MDI (ExtraTrees)
                from sklearn.ensemble import ExtraTreesRegressor
                mdi_base = ExtraTreesRegressor(n_estimators=500, min_samples_leaf=50, max_features='sqrt', n_jobs=-1, random_state=42)
                
                sel_res = mdi_feature_selection_v3(
                    X, y,
                    base_model=mdi_base,
                    sample_weight=w,
                    end_features=60,
                    cumulative_cap=0.99,
                    min_share=0.0005,
                    min_features=30,
                    max_features_pct=0.8
                )
                
                selected_feats = sel_res.selected_features
                tprint(f"MDI selected {len(selected_feats)} features (from {X.shape[1]}).")
                
                X_sel = X[selected_feats]
                cols = list(selected_feats)
                
                y_hard_check = (y >= 0.5).astype(int)
                tprint(f"  Class dist: 0={int((y_hard_check==0).sum())} ({(y_hard_check==0).mean()*100:.1f}%), "
                       f"1={int((y_hard_check==1).sum())} ({(y_hard_check==1).mean()*100:.1f}%)")

                race = ModelRace(kind=k, n_splits=5)
                groups = df["__ts__"].values if "__ts__" in df.columns else None
                race.fit(X_sel, y, sample_weight=w, returns=y_ret, groups=groups)
                score = race.metrics.get(race.best_model_name, -1.0)
                dm = race.detailed_metrics.get(race.best_model_name, {})
                # Race CV AUC = fold-averaged AUC during model selection
                # OOF AUC = AUC on full post-calibration OOF vector (canonical)
                oof_auc_canonical = 0.5
                if race.oof_probs is not None:
                    y_bin_canon = (y >= 0.5).astype(np.int8)
                    if len(np.unique(y_bin_canon)) > 1:
                        from sklearn.metrics import roc_auc_score as _roc_auc
                        oof_auc_canonical = float(_roc_auc(y_bin_canon, race.oof_probs))
                # --- Alpha model OOF diagnostics (all metrics from same post-calibration oof) ---
                per_regime = {}
                oof_bss = dm.get('BSS', 0)  # fallback to race BSS
                _bs_oof = float('nan')
                _prev_global = float('nan')
                if race.oof_probs is not None:
                    oof = race.oof_probs
                    y_bin_oof = (y >= 0.5).astype(np.int8)
                    # Recompute BSS from post-calibration OOF (same probs as all other metrics)
                    from sklearn.metrics import brier_score_loss as _bsl
                    _prev_global = float(np.mean(y_bin_oof))
                    _p_clip = np.clip(oof, 1e-7, 1 - 1e-7)
                    try:
                        _bs_oof = float(_bsl(y_bin_oof, _p_clip))
                        _bs_ref_global = float(_bsl(y_bin_oof, np.full_like(_p_clip, _prev_global)))
                        _bs_ref_global = max(_bs_ref_global, 1e-6)
                        oof_bss = 1.0 - (_bs_oof / _bs_ref_global)
                        if not np.isfinite(oof_bss):
                            oof_bss = 0.0
                    except Exception:
                        _bs_oof = 0.0
                        oof_bss = 0.0

                tprint(f"Finished {side} {k} H={H}: Winner={race.best_model_name}, Score={score:.4f}, "
                       f"RcAUC={dm.get('AUC',0):.4f}, OOF_AUC={oof_auc_canonical:.4f}, "
                       f"RcIC={dm.get('IC',0):.4f}, RcBSS={dm.get('BSS',0):.4f}, "
                       f"OOF_Brier={_bs_oof:.4f}, OOF_BSS={oof_bss:.4f}")

                if race.oof_probs is not None:
                    tprint(f"  OOF probs [post-cal]: mean={np.mean(oof):.4f}, std={np.std(oof):.4f}, "
                           f"min={np.min(oof):.4f}, max={np.max(oof):.4f}")
                    tprint(f"  OOF Brier={_bs_oof:.4f}, prev={_prev_global:.4f}, BSS={oof_bss:.4f}")
                    # OOF-based return correlation (key signal quality metric)
                    if np.std(oof) > 1e-9 and np.std(y_ret) > 1e-9:
                        oof_ret_corr = float(np.corrcoef(oof, y_ret)[0, 1])
                        tprint(f"  OOF-return correlation: {oof_ret_corr:.4f}")
                    # Calibration: mean predicted prob vs actual positive rate
                    tprint(f"  Calibration: mean_pred={np.mean(oof):.4f} vs actual_rate={_prev_global:.4f}")
                    g = df["__ts__"].values if "__ts__" in df.columns else None
                    # All eval metrics UNWEIGHTED (Option B: weights only for training loss)
                    m10 = topk_mask(oof, 0.10, groups=g)
                    ece10 = ece_at_mask(y_bin_oof, oof, m10, n_bins=10)
                    curve = calibration_curve_bins(y_bin_oof, oof, n_bins=10)
                    profile = calibration_profile(curve)
                    p10 = precision_at_k(y_bin_oof, oof, 0.10, groups=g)
                    p40 = precision_at_k(y_bin_oof, oof, 0.40, groups=g)
                    tpd10 = avg_trades_per_day(oof, 0.10, g) if g is not None else float(len(oof) * 0.10)
                    tpd30 = avg_trades_per_day(oof, 0.30, g) if g is not None else float(len(oof) * 0.30)
                    tprint(
                        f"  AlphaPrec@10={p10:.4f} AlphaPrec@40={p40:.4f} "
                        f"AvgTrades/Day@10={tpd10:.2f} AvgTrades/Day@30={tpd30:.2f}"
                    )
                    tprint(f"  Alpha calibration: ECE@10={ece10:.4f} profile={profile}")

                    # --- Per-regime BSS/AUC (unweighted, with both bucket and global baselines) ---
                    per_regime = compute_per_regime_metrics(y, oof, df, global_prev=_prev_global)
                    if per_regime:
                        tprint(f"  Per-regime BSS/AUC/Brier ({len(per_regime)} dimensions):")
                        for rname, rbuckets in per_regime.items():
                            parts = []
                            for bl, bm in rbuckets.items():
                                bss_g = bm.get('bss_global', 0)
                                parts.append(f"{bl}(n={bm['n']}): BSS={bm['bss']:.3f} BSS_g={bss_g:.3f} AUC={bm['auc']:.3f} Brier={bm.get('brier',0):.4f}")
                            tprint(f"    {rname}: {' | '.join(parts)}")
                    if "__n_res__" in df.columns:
                        n_res_vals = np.clip(df["__n_res__"].values.astype(float), 0.0, None)
                        try:
                            from sklearn.metrics import roc_auc_score
                            if len(np.unique(y_bin_oof)) > 1:
                                auc_all = float(roc_auc_score(y_bin_oof, oof))
                            else:
                                auc_all = 0.5
                        except Exception:
                            auc_all = 0.5
                        resolved_w = n_res_vals / max(np.mean(n_res_vals), 1e-9)
                        try:
                            auc_res = float(roc_auc_score(y_bin_oof, oof, sample_weight=resolved_w)) if len(np.unique(y_bin_oof)) > 1 else 0.5
                        except Exception:
                            auc_res = 0.5
                        tprint(f"  AUC reporting ({side}/{k}/H={H}): raw={auc_all:.4f}, resolved-weighted={auc_res:.4f}")

                alpha_diag = {}
                if race.best_model_name in race.detailed_metrics:
                    dm_best = race.detailed_metrics[race.best_model_name]
                    alpha_diag = {
                        "prec10": float(dm_best.get("Prec10", np.nan)),
                        "prec40": float(dm_best.get("Prec40", np.nan)),
                        "ece_top10": float(dm_best.get("ece_top10", np.nan)),
                        "calibration_profile": dm_best.get("calibration_profile", "n/a"),
                    }
                if race.oof_probs is not None:
                    groups_v = df["__ts__"].values if "__ts__" in df.columns else None
                    alpha_diag["avg_trades_day_10"] = float(avg_trades_per_day(race.oof_probs, 0.10, groups_v)) if groups_v is not None else float(len(race.oof_probs) * 0.10)
                    alpha_diag["avg_trades_day_30"] = float(avg_trades_per_day(race.oof_probs, 0.30, groups_v)) if groups_v is not None else float(len(race.oof_probs) * 0.30)
                    oof_metrics = _aggregate_alpha_oof_metrics(y, race.oof_probs, y_ret, sample_weight=w, groups=groups_v)
                    alpha_diag.update(oof_metrics)

                if score > best_ic:
                    best_ic = score
                    best_m = {"model": race, "H": H, "feat_cols": cols, "per_regime": per_regime, "alpha_diag": alpha_diag}

            # --- Stage Gate Check (Alpha) ---
            if best_m is not None:
                race_best = best_m["model"]
                cv_prec10 = race_best.detailed_metrics.get(race_best.best_model_name, {}).get("CV_Prec10", 1.0)

                # We need OOF probs and targets from the best model (which corresponds to 'race')
                # But 'race' is the ModelRace instance. Its oof_probs are for the best model.
                # And we need y, y_ret from the dataset used for best_m["H"]
                best_H = best_m["H"]
                best_key = f"train_{side}_{k}_{best_H}"
                if best_key in datasets:
                    df_best = datasets[best_key]
                    y_best = df_best["__y_bin__"].values
                    y_ret_best = df_best["__y_ret__"].values
                    oof_best = race_best.oof_probs

                    # Ensure alignment (ModelRace OOF should align with input y)
                    if oof_best is not None and len(oof_best) == len(y_best):
                        gate_res = compute_stage_gate_metrics(
                            y_best, oof_best, y_ret_best,
                            model_type="classifier",
                            cv_prec10=cv_prec10
                        )
                        gate_res["Model"] = f"{side}_{k}"
                        alpha_gate_results.append(gate_res)

            final_models[side][k] = best_m

    # 3. Train Meta Models (One per Alpha Model: Side x Kind)
    # Using OOF predictions from Alpha Models to train Meta Model (re-calibration/sizing)
    # The Alpha models (ModelRace) already generate OOF predictions via CV.
    meta_models = {}
    for side in trade_sides:
        for k in kinds:
            # Check if alpha model exists
            conf = final_models[side].get(k)
            if not conf:
                tprint(f"Meta {side}_{k}: skipped (missing alpha model)")
                continue

            H = conf["H"]
            key = f"train_{side}_{k}_{H}"
            if key not in datasets:
                tprint(f"Meta {side}_{k}: skipped (missing dataset)")
                continue

            df = datasets[key].copy()

            # Ensure index meta columns exist (though not strictly needed if we use df directly)
            if "__ts__" not in df.columns or "__symbol__" not in df.columns:
                 # If missing, we might still proceed if row order is preserved (it should be)
                 pass

            # Use the OOF probs from the race (generated via CV)
            race = conf["model"]
            if race.oof_probs is None:
                tprint(f"Meta {side}_{k}: skipped (no OOF probs found)")
                continue

            p_oof = race.oof_probs
            y_ret = df["__y_ret__"].values

            if len(p_oof) != len(y_ret):
                tprint(f"Meta {side}_{k}: mismatch length OOF={len(p_oof)} vs y_ret={len(y_ret)}")
                continue

            # Prepare meta features using RAW feature names (not interaction-toggled).
            # At inference time (engine.py), the meta model receives raw feature names
            # like "ambig", not toggled names like "ambig_G_VOL_0".
            # Raw values were preserved as __meta_raw__{name} columns in
            # build_hourly_training_set_and_weights before interaction toggles.
            configured_meta = cfg.get("meta_feature_keys", [])
            feat_cols = []
            raw_prefix = "__meta_raw__"
            for mk in configured_meta:
                prefixed = f"{raw_prefix}{mk}"
                if prefixed in df.columns:
                    feat_cols.append(mk)  # use raw name for column in X_feats
            feat_cols = list(dict.fromkeys(feat_cols))  # dedupe preserving order
            if not feat_cols:
                tprint(f"Meta {side}_{k}: skipped (no raw meta features found in dataset)")
                continue

            # Build X_feats with raw feature names (matching inference column names)
            X_feats = pd.DataFrame(index=df.index)
            for mk in feat_cols:
                X_feats[mk] = df[f"{raw_prefix}{mk}"].values
            X_feats = X_feats.fillna(0.0)

            # Cross-sectional rank percentile target.
            # For each timestamp, rank raw returns across symbols and convert to
            # percentile [0, 1]. This is the cleanest target for a meta layer
            # whose job is cross-sectional selection:
            #   - Immune to market-wide drift (rank is relative)
            #   - No tail warping (unlike log/winsor transforms)
            #   - IC on rank_pct ≡ IC on raw returns (monotone transform)
            #   - Ridge learns "which features predict relative outperformance"
            _ts_meta = df["__ts__"].values if "__ts__" in df.columns else None
            if _ts_meta is not None:
                y_target = np.zeros(len(y_ret), dtype=np.float64)
                for t in np.unique(_ts_meta):
                    _tmask = _ts_meta == t
                    n_t = _tmask.sum()
                    if n_t > 1:
                        from scipy.stats import rankdata
                        y_target[_tmask] = (rankdata(y_ret[_tmask]) - 1) / (n_t - 1)
                    else:
                        y_target[_tmask] = 0.5
            else:
                from scipy.stats import rankdata
                y_target = (rankdata(y_ret) - 1) / max(len(y_ret) - 1, 1)
            tprint(f"  Using rank_pct target: mean={np.mean(y_target):.4f}, "
                   f"std={np.std(y_target):.4f}, range=[{np.min(y_target):.4f}, {np.max(y_target):.4f}]")

            # Meta weights: magnitude + MAE/MFE proxy + consensus; remove 20% lowest n_res
            n_res = df.get("__n_res__", pd.Series(np.ones(len(df)), index=df.index)).values.astype(np.float64)
            keep = n_res >= np.quantile(n_res, 0.20)
            if keep.sum() < 100:
                keep = np.ones(len(df), dtype=bool)

            df = df.loc[keep].copy()
            X_feats = X_feats.loc[keep].copy()
            y_target = y_target[keep]
            p_oof = p_oof[keep]
            n_res = n_res[keep]

            # Confidence weight: upweight samples where alpha model is more decisive
            # (further from 0.5). Meta should focus on "strong calls" where gating matters.
            conf_w = np.abs(p_oof - 0.5) * 2.0  # [0, 1] range
            conf_w = 0.8 + 0.4 * (conf_w / max(np.mean(conf_w), 1e-9))
            mag = _winsorize_and_unit_mean(conf_w, clip_min=0.75, clip_max=1.25)

            cons = np.clip(df.get("__w_consensus__", pd.Series(np.ones(len(df)), index=df.index)).values.astype(np.float64), 0.75, 1.25)
            cons = _winsorize_and_unit_mean(cons, clip_min=0.75, clip_max=1.25)

            w_meta = mag * cons
            alpha_pow = float(np.clip(cfg.get("meta_weight_alpha", 0.5), 0.3, 0.7))
            w_meta = np.power(np.clip(w_meta, 1e-9, None), alpha_pow)
            ts_vals = df["__ts__"].values if "__ts__" in df.columns else np.arange(len(df))
            w_meta = _normalize_cross_sectional(ts_vals, w_meta)
            w_meta = w_meta / max(np.mean(w_meta), 1e-12)

            tprint(f"Meta {side}_{k}: Training on {len(df)} samples with {len(feat_cols)} raw meta features...")
            tprint(f"  Meta feature names: {feat_cols[:10]}{'...' if len(feat_cols) > 10 else ''}")
            tprint(f"  Target: mean={np.mean(y_target):.6f}, std={np.std(y_target):.6f}, "
                   f"min={np.min(y_target):.4f}, max={np.max(y_target):.4f}")
            tprint(f"  OOF probs input: mean={np.mean(p_oof):.4f}, std={np.std(p_oof):.4f}")

            meta = MetaModel()
            X_meta = meta.prepare_meta_features(p_oof, X_feats, pred_col_name="pred_logit")

            # Add feature interactions: pred_logit × key features + regime buckets
            pred_logit = X_meta["pred_logit"].values
            for interact_feat in ["vol_z", "mkt_rv_ratio", "ambig", "exh_qual", "trend_pct"]:
                if interact_feat in X_meta.columns:
                    X_meta[f"pred_x_{interact_feat}"] = pred_logit * X_meta[interact_feat].values
            # Regime bucket interactions: pred_logit × vol/trend regime indicators
            for regime_col in ["G_VOL", "G_TREND"]:
                raw_regime = f"{raw_prefix}{regime_col}"
                if raw_regime in df.columns:
                    rv = df[raw_regime].values
                    for bucket in [0, 1, 2]:
                        X_meta[f"pred_x_{regime_col}_{bucket}"] = pred_logit * (rv == bucket).astype(float)

            meta_groups = df["__ts__"].values if "__ts__" in df.columns else None
            meta.fit(X_meta, y_target, sample_weight=w_meta, groups=meta_groups)

            # --- Meta model OOF diagnostics ---
            if meta.oof_probs is not None:
                m_oof = meta.oof_probs
                tprint(f"  Meta OOF preds: mean={np.mean(m_oof):.6f}, std={np.std(m_oof):.6f}, "
                       f"min={np.min(m_oof):.6f}, max={np.max(m_oof):.6f}")

                meta_groups = df["__ts__"].values if "__ts__" in df.columns else None
                y_ret_filtered = df["__y_ret__"].values if "__y_ret__" in df.columns else y_target
                _compute_and_print_meta_metrics(y_target, m_oof, name=f"Meta {side}_{k}",
                                                groups=meta_groups, y_raw_ret=y_ret_filtered)

            if meta.selected_features:
                tprint(f"  Meta selected features ({len(meta.selected_features)}): {meta.selected_features[:8]}{'...' if len(meta.selected_features) > 8 else ''}")

            meta_models[f"{side}_{k}"] = meta
            tprint(f"Meta {side}_{k}: fitted.")

            # --- Stage Gate Check (Meta) ---
            if meta.oof_probs is not None:
                # y_target is the rank percentile target used for training meta
                # y_ret_filtered is the raw return (for downside checks)
                y_ret_filtered = df["__y_ret__"].values if "__y_ret__" in df.columns else y_target

                gate_res = compute_stage_gate_metrics(
                    y_target, meta.oof_probs, y_ret_filtered,
                    model_type="quantile_meta"
                )
                gate_res["Model"] = f"{side}_{k}"
                meta_gate_results.append(gate_res)

    # 4. Train Exhaustion Models
    exh_models = {}
    for d in directions:
        key = f"exh_{d}"
        if key in datasets:
            df = datasets[key]
            if len(df) > 100:
                y = df["__y__"].values.astype(int)
                w_raw = df["__w__"].values.astype(np.float64)
                p95_w = np.percentile(w_raw, 95)
                w = np.clip(w_raw, 0.0, max(p95_w, 1e-6))
                # Drop meta/targets
                # In `build_exhaustion_Xy`, X columns are features + G_VOL/G_TREND
                # The reset_index added `ts` `symbol`.
                drop_cols = ["__y__", "__w__", "ts", "symbol"]
                X = df.drop(columns=[c for c in drop_cols if c in df.columns])

                n_pos = int((y==1).sum())
                n_neg = int((y==0).sum())
                prevalence = n_pos / max(1, len(y))
                n_eff_exh = (np.sum(w) ** 2) / np.sum(w ** 2)
                tprint(f"Exhaustion {d}: {len(X)} samples, {X.shape[1]} features, class dist: 0={n_neg} 1={n_pos} (prev={prevalence:.4f}), n_eff={n_eff_exh:.0f}")
                m = ExhaustionModel()
                m.fit(X, y, sample_weight=w)
                if m.model is not None:
                    tprint(f"Exhaustion {d}: fitted, {len(m.selected_features)} selected features")
                    # Evaluate with PR-AUC (appropriate for extreme imbalance)
                    try:
                        from sklearn.metrics import average_precision_score, precision_score
                        X_eval = X[m.selected_features].fillna(0.0)
                        p_exh = m.model.predict_proba(X_eval)[:, 1]
                        pr_auc = average_precision_score(y, p_exh, sample_weight=w)
                        # Precision at top-K (K = 2 * n_pos)
                        k = min(2 * max(n_pos, 1), len(y))
                        top_k_idx = np.argsort(p_exh)[-k:]
                        prec_at_k = float(np.mean(y[top_k_idx]))
                        tprint(f"Exhaustion {d}: PR-AUC={pr_auc:.4f} (baseline={prevalence:.4f}), Prec@{k}={prec_at_k:.4f}")
                    except Exception as e:
                        tprint(f"Exhaustion {d}: eval error: {e}")
                else:
                    tprint(f"Exhaustion {d}: fitting failed (model is None)")
                exh_models[d] = m
            else:
                tprint(f"Exhaustion {d}: skipped (only {len(df)} samples)")
                exh_models[d] = None
        else:
            tprint(f"Exhaustion {d}: no dataset found")
            exh_models[d] = None

    # --- Stage Gate Reporting ---
    tprint("\n=== Stage Gate Report: Alpha Models (Classifiers) ===")
    alpha_pass_count = 0
    if alpha_gate_results:
        df_alpha_gate = pd.DataFrame(alpha_gate_results)
        cols_order = ["Model", "passed", "PR_AUC", "Brier_Imp", "Lift_k", "CV_Prec_k"]
        # Print main columns
        tprint(df_alpha_gate[ [c for c in cols_order if c in df_alpha_gate.columns] ].to_string(index=False))
        alpha_pass_count = df_alpha_gate["passed"].sum()
    else:
        tprint("No Alpha models evaluated.")

    tprint(f"\nAlpha Stage: {alpha_pass_count}/4 passed.")
    if alpha_pass_count < 2:
        tprint("WARNING: Alpha Stage FAILED (< 2 models passed).")

    tprint("\n=== Stage Gate Report: Meta Models (Quantile) ===")
    meta_pass_count = 0
    if meta_gate_results:
        df_meta_gate = pd.DataFrame(meta_gate_results)
        cols_order = ["Model", "passed", "Coverage_Diff", "Pinball_Imp", "Spearman_IC", "Pass_Spread", "Pass_Downside"]
        tprint(df_meta_gate[ [c for c in cols_order if c in df_meta_gate.columns] ].to_string(index=False))
        meta_pass_count = df_meta_gate["passed"].sum()
    else:
        tprint("No Meta models evaluated.")

    tprint(f"\nMeta Stage: {meta_pass_count}/4 passed.")
    if meta_pass_count < 2:
        tprint("WARNING: Meta Stage FAILED (< 2 models passed).")

    return {
        "alpha_models": final_models,
        "alpha_oof_metrics": {f"{side}_{kind}": final_models.get(side, {}).get(kind, {}).get("alpha_diag", {}) for side in trade_sides for kind in kinds},
        "exh_models": exh_models,
        "meta_models": meta_models,
        "spike_models": spike_models,
        "specialist_models": specialist_models,
    }

def select_best_horizon(panel, feats, mkt_gates, cfg, syms, ts, p_exh_hist):
    # DEPRECATED / LEGACY WRAPPER
    # This function is kept for backward compatibility if needed,
    # but strictly we should use the new split.
    # We can implement it by calling the new functions in sequence.
    datasets = generate_label_datasets(panel, feats, mkt_gates, cfg, syms, ts, p_exh_hist)
    return train_models_from_artifacts(datasets, cfg)

def optimize_risk_params(panel, feats, mkt_gates, cfg, train_syms, ts, p_exh_hist, alpha_models):
    tprint("Entering function: optimize_risk_params in training.py (High Throughput Selection)")

    granular_risk = {}

    # 1. Prepare shared price data
    # We need to process all candidates from the training history.
    # We can reuse select_trade_candidates_vectorized logic but we want ALL potential signals.
    # Or just use the signals that were actually generated by the strategy logic?
    # The selection script expects X (features) and prices.

    # Extract panel data
    open_df = panel["open"]
    high_df = panel["high"]
    low_df = panel["low"]
    close_df = panel["close"]

    # ATR stats
    if "atr_pct" not in feats:
        tprint("ATR pct missing, skipping optimization")
        return cfg

    atr_pct_df = feats["atr_pct"]
    window_base = 24 * 30

    tprint("Computing ATR baselines for optimization...")
    atr_base_df = atr_pct_df.rolling(window_base, min_periods=24).median().bfill()
    # Using the fast numpy functions if possible, but pandas is easier for alignment here
    # For Z, we need a robust one.
    # Let's use the one from fast_funcs if available or re-implement simple robust Z.
    atr_std_df = atr_pct_df.rolling(window_base, min_periods=24).std().bfill()
    z_df = (atr_pct_df - atr_base_df) / (atr_std_df + 1e-12)

    # 2. Iterate over strategies (buckets)
    trade_sides = ["long", "short"]
    kinds = ["mr", "tf"]

    # We need to gather events for each bucket.
    # This is non-trivial because `optimize_risk_params` is usually called on a small simulation window.
    # But `run_tp_sl_selection_fast` is designed for training time selection on historical data.
    # Assuming `ts` is the end of training.

    # We will scan the last N days (e.g. 90 or 180) for candidates.
    lookback_days = 90
    ts_start = ts - pd.Timedelta(days=lookback_days)

    # Select candidates
    cand_mask = select_trade_candidates_vectorized(panel, feats, pct=cfg["trade_extreme_pct"], metric=cfg["trade_deviation_metric"])
    if cand_mask is None:
        tprint("No candidates found.")
        return cfg

    final_mask = cand_mask.loc[(cand_mask.index >= ts_start) & (cand_mask.index <= ts)]

    valid_ts = final_mask[final_mask.any(axis=1)].index

    # Pre-fetch numpy arrays for the full period (aligned)
    # We need a unified index and columns
    # Let's align everything to `close_df`

    # Flatten everything to 1D arrays of events?
    # run_tp_sl_selection_fast takes 1D arrays for open, high, etc?
    # No, it takes full arrays and event indices.
    # BUT, if we have multiple assets, we can concatenate them?
    # Or run per asset? No, pooled.
    # To pool, we can concatenate all asset time series end-to-end, and adjust event indices.
    # That's one way.
    # Or simply: run_tp_sl_selection_fast expects single instrument arrays.
    # "Assumptions: Single instrument arrays, time-aligned."
    # So we cannot pass the whole panel directly.
    # We must flatten the panel into a long format (Time, Symbol) -> single timeline?
    # No, time-alignment breaks if we concatenate.
    # Effectively, we treat the dataset as one long series of (t, asset) observations.
    # We can concatenate the columns of the panel into one giant 1D array.
    # And compute event indices relative to this giant array.
    # Yes, that works if we insert NaNs or gaps between assets to prevent window crossover.
    # Or just careful indexing.
    # Let's concatenate with a small buffer of NaNs between assets.

    tprint("Flattening panel data for pooled optimization...")

    # 15m precision is NOT used in the optimizer — the build_event_cache_15m
    # requires a contiguous 15m array aligned 4:1 with the full 1h panel,
    # which would require downloading 90 days of 15m data per symbol (~400 symbols).
    # 1h resolution is sufficient for grid search; 15m is used in backtest engine per-trade.
    use_15m = False
    exchange = None

    assets = close_df.columns
    # Collect arrays (1h resolution — always needed for features/ATR)
    big_open = []
    big_high = []
    big_low = []
    big_close = []
    big_atr = []
    big_z = []
    big_atr_base = []
    big_X = [] # Features

    # 15m resolution arrays (optional)
    big_open_15m = []
    big_high_15m = []
    big_low_15m = []
    big_close_15m = []
    has_15m_data = False

    asset_offsets = {}
    asset_offsets_15m = {}
    current_offset = 0
    current_offset_15m = 0
    buffer_size = 100 # larger than horizon
    buffer_size_15m = 400 # 100 hours * 4

    # We need features too.
    # Let's pick a standard set of features for X
    feat_keys = cfg.get("causal_cols", [])
    if not feat_keys:
        # Fallback
        feat_keys = ["trend_pct", "vol_pct", "ret24h"]

    for sym in assets:
        if sym not in atr_pct_df.columns: continue

        # Get data chunks
        o = open_df[sym].values.astype(np.float32)
        h = high_df[sym].values.astype(np.float32)
        l = low_df[sym].values.astype(np.float32)
        c = close_df[sym].values.astype(np.float32)

        a = np.nan_to_num(atr_pct_df[sym].reindex(close_df.index).values.astype(np.float32), nan=0.01)
        b = np.nan_to_num(atr_base_df[sym].reindex(close_df.index).values.astype(np.float32), nan=0.01)
        z_v = np.nan_to_num(z_df[sym].reindex(close_df.index).values.astype(np.float32), nan=0.0)

        # Features
        # Gather into (T, F) — reindex to panel index to ensure length match
        panel_idx = close_df.index
        x_list = []
        for k in feat_keys:
            if k in feats and sym in feats[k].columns:
                x_list.append(np.nan_to_num(feats[k][sym].reindex(panel_idx).values.astype(np.float32), nan=0.0))
            else:
                x_list.append(np.zeros(len(c), dtype=np.float32))
        x_arr = np.stack(x_list, axis=1) if x_list else np.zeros((len(c), 1), dtype=np.float32)

        # Append 1h data
        big_open.append(o)
        big_high.append(h)
        big_low.append(l)
        big_close.append(c)
        big_atr.append(a)
        big_atr_base.append(b)
        big_z.append(z_v)
        big_X.append(x_arr)

        asset_offsets[sym] = current_offset
        current_offset += len(c) + buffer_size

        # Add buffer
        nan_buf = np.full(buffer_size, np.nan, dtype=np.float32)
        nan_buf_x = np.full((buffer_size, len(feat_keys)), np.nan, dtype=np.float32)

        big_open.append(nan_buf)
        big_high.append(nan_buf)
        big_low.append(nan_buf)
        big_close.append(nan_buf)
        big_atr.append(nan_buf)
        big_atr_base.append(nan_buf)
        big_z.append(nan_buf)
        big_X.append(nan_buf_x)

        # Download/load 15m data for this asset
        if use_15m:
            try:
                from extreme_price_movements.hf_data_loader import get_15m_ohlcv
                ccxt_sym = sym if "/" in sym else sym.replace("USDT", "/USDT")
                # Download full optimization window in one shot
                window_hours = int((ts - ts_start).total_seconds() / 3600) + 48
                ts_start_utc = pd.Timestamp(ts_start)
                if ts_start_utc.tzinfo is None:
                    ts_start_utc = ts_start_utc.tz_localize('UTC')
                else:
                    ts_start_utc = ts_start_utc.tz_convert('UTC')

                df_15m = get_15m_ohlcv(exchange, ccxt_sym, ts_start_utc, window_hours)
                if not df_15m.empty and len(df_15m) >= len(c) * 3:  # at least 75% coverage
                    o15 = df_15m['open'].values.astype(np.float32)
                    h15 = df_15m['high'].values.astype(np.float32)
                    l15 = df_15m['low'].values.astype(np.float32)
                    c15 = df_15m['close'].values.astype(np.float32)

                    big_open_15m.append(o15)
                    big_high_15m.append(h15)
                    big_low_15m.append(l15)
                    big_close_15m.append(c15)

                    asset_offsets_15m[sym] = current_offset_15m
                    current_offset_15m += len(c15) + buffer_size_15m

                    nan_buf_15m = np.full(buffer_size_15m, np.nan, dtype=np.float32)
                    big_open_15m.append(nan_buf_15m)
                    big_high_15m.append(nan_buf_15m)
                    big_low_15m.append(nan_buf_15m)
                    big_close_15m.append(nan_buf_15m)
                else:
                    tprint(f"  {sym}: 15m data insufficient ({len(df_15m) if not df_15m.empty else 0} bars), using 1h")
            except Exception as e:
                tprint(f"  {sym}: 15m download failed: {e}")

    # Concatenate 1h arrays
    full_open = np.concatenate(big_open)
    full_high = np.concatenate(big_high)
    full_low = np.concatenate(big_low)
    full_close = np.concatenate(big_close)
    full_atr = np.concatenate(big_atr)
    full_atr_base = np.concatenate(big_atr_base)
    full_z = np.concatenate(big_z)
    full_X = np.concatenate(big_X, axis=0)

    # Concatenate 15m arrays if available
    full_open_15m = full_high_15m = full_low_15m = full_close_15m = None
    if use_15m and big_open_15m:
        full_open_15m = np.concatenate(big_open_15m)
        full_high_15m = np.concatenate(big_high_15m)
        full_low_15m = np.concatenate(big_low_15m)
        full_close_15m = np.concatenate(big_close_15m)
        has_15m_data = True
        tprint(f"15m data: {len(asset_offsets_15m)}/{len(asset_offsets)} assets, {len(full_close_15m)} total bars")

    # Now iterate strategies and collect event indices
    for side in trade_sides:
        for k in kinds:
            # Filter logic
            cand_filter = "unknown"
            if side == "long":
                if k == "mr": cand_filter = "worst"
                else: cand_filter = "best"
            else:
                if k == "mr": cand_filter = "best"
                else: cand_filter = "worst"

            trend_filter = "up" if cand_filter == "best" else "down"

            # Collect indices (1h and optionally 15m)
            indices = []
            time_indices = []  # parallel array: actual time position (shared across assets)
            indices_15m = [] if has_15m_data else None

            # Iterate valid timestamps
            for t in valid_ts:
                # Get candidates at t
                row = final_mask.loc[t]
                cands = row[row].index.intersection(assets)

                # Check trend
                trend_vals = feats["trend_pct"].loc[t, cands]

                for sym in cands:
                    tv = trend_vals[sym]
                    tdir = np.sign(tv) if tv != 0 else 1.0

                    if trend_filter == "up" and tdir <= 0: continue
                    if trend_filter == "down" and tdir > 0: continue

                    # Found a candidate
                    # Get index in full arrays
                    try:
                        tidx = close_df.index.get_loc(t)
                    except KeyError:
                        continue

                    flat_idx = asset_offsets[sym] + tidx
                    indices.append(flat_idx)
                    time_indices.append(tidx)  # true temporal coordinate (same for all assets at time t)

                    # Also collect 15m index if available for this asset
                    if has_15m_data and sym in asset_offsets_15m:
                        flat_idx_15m = asset_offsets_15m[sym] + tidx
                        indices_15m.append(flat_idx_15m)

            indices = np.array(indices, dtype=np.int32)
            time_indices = np.array(time_indices, dtype=np.int64)
            if len(indices) > 0:
                mean_z = float(np.mean(full_z[indices]))
                # Compute actual barrier_pct (vol-scaled, clipped fraction) for meaningful logging
                _b_pct = scaled_atr_pct(
                    full_atr[indices], full_z[indices],
                    full_atr_base[indices], z_max=3.0, lo=0.03, hi=0.06
                )
                mean_barrier = float(np.nanmean(_b_pct))
                tprint(f"Bucket {side} {k} ({cand_filter}): {len(indices)} events | Mean Barrier%={mean_barrier*100:.2f} | Mean VolZ={mean_z:.2f}")
            else:
                tprint(f"Bucket {side} {k} ({cand_filter}): 0 events")

            if len(indices) < 50:
                tprint("Not enough events, using defaults.")
                default_risk = {
                    "tp_mult": cfg.get("tp_mult", 1.0),
                    "sl_mult": cfg.get("sl_mult", 0.5),
                    "trail_mult": cfg.get("trail_mult", 0.5),
                    "vol_lo": cfg.get("vol_lo", 0.03),
                    "vol_hi": cfg.get("vol_hi", 0.06),
                    "vol_z_max": cfg.get("vol_z_max", 3.0),
                    "max_hold_hours": 12 if k == "mr" else 24,
                    "k_sl": cfg["risk_k_sl"],
                    "k_trail_start": cfg["risk_k_trail_start"],
                    "k_trail_dist": cfg["risk_k_trail_dist"],
                }
                granular_risk[f"risk_{side}_{k}"] = default_risk
                granular_risk[f"risk_{k}_{cand_filter}"] = default_risk
                continue

            # Run optimization
            # Note: run_tp_sl_selection_fast selects tp_mult and sl_mult for TRIPLE BARRIER.
            # But the system might use Trailing ATR logic at execution time?
            # If we switch to Triple Barrier execution, we use these.
            # If we use Trailing ATR, we map them: k_sl = sl_mult (approx).
            # The prompt implies we want to find optimal "TP:SL ratio" and levels.

            # Prepare optional 15m arrays for this bucket
            _15m_kwargs = {}
            if has_15m_data and indices_15m and len(indices_15m) == len(indices):
                _15m_kwargs = {
                    "open_15m": full_open_15m,
                    "high_15m": full_high_15m,
                    "low_15m": full_low_15m,
                    "close_15m": full_close_15m,
                }
                tprint(f"  Passing 15m data to optimizer ({len(indices)} events)")

            summary = run_tp_sl_selection_fast(
                X=full_X,
                open_=full_open,
                high=full_high,
                low=full_low,
                close=full_close,
                atr_pct=full_atr,
                z=full_z,
                atr_base_pct=full_atr_base,
                event_idx=indices,
                horizon=24,
                max_events=2000,
                tp_mult_grid=[0.4, 0.5, 0.6, 0.8, 1.0, 1.25, 1.5],
                sl_mult_grid=[0.10, 0.15, 0.18, 0.25, 0.30, 0.40, 0.50],
                trail_mult_grid=[0.15, 0.25, 0.35, 0.50],
                lo_grid=[0.01, 0.02, 0.03, 0.04],
                hi_grid=[0.05, 0.06, 0.07],
                z_max_grid=[2.5, 3.0, 3.5],
                entry_mode="next_open",
                event_time_idx=time_indices,
                **_15m_kwargs
            )

            # Enforce hard constraints on optimized values
            _opt_bp_risk = 0.5 * (summary.final_lo + summary.final_hi)
            _abs_tp_risk = summary.final_tp_mult * _opt_bp_risk
            _ratio_risk = summary.final_tp_mult / max(summary.final_sl_mult, 0.01)
            min_ratio = float(cfg.get("min_tp_sl_ratio", 1.5))
            min_tp_abs = float(cfg.get("min_tp_abs_pct", 0.02))
            _tp_m = summary.final_tp_mult
            _sl_m = summary.final_sl_mult
            if _abs_tp_risk < min_tp_abs:
                _tp_m = min_tp_abs / max(_opt_bp_risk, 0.01)
                tprint(f"  Enforced min TP: tp_mult raised to {_tp_m:.2f} (abs {_tp_m*_opt_bp_risk*100:.1f}%)")
            if _tp_m / max(_sl_m, 0.01) < min_ratio:
                _sl_m = _tp_m / min_ratio
                tprint(f"  Enforced min ratio: sl_mult lowered to {_sl_m:.2f}")
            summary.final_tp_mult = _tp_m
            summary.final_sl_mult = _sl_m
            tprint(f"Optimized {side} {k}: TP={summary.final_tp_mult:.2f} ({summary.final_tp_mult*_opt_bp_risk*100:.1f}%), "
                   f"SL={summary.final_sl_mult:.2f} ({summary.final_sl_mult*_opt_bp_risk*100:.1f}%), "
                   f"ratio={summary.final_tp_mult/max(summary.final_sl_mult,0.01):.1f}x, "
                   f"Trail={summary.final_trail_mult:.2f}, Lo={summary.final_lo:.2f}, Hi={summary.final_hi:.2f}, Zmax={summary.final_z_max:.2f}")

            if summary.outer_results:
                avg_auc = np.mean([r.test_auc for r in summary.outer_results])
                avg_ic = np.mean([r.test_ic for r in summary.outer_results])
                avg_pnl = np.mean([r.test_pnl for r in summary.outer_results])
                tprint(f"  Avg Test Metrics: AUC={avg_auc:.4f}, IC={avg_ic:.4f}, PnL={avg_pnl:.4f}")

                pairs = [(r.chosen_tp_mult, r.chosen_sl_mult, r.chosen_trail_mult, r.chosen_lo, r.chosen_hi, r.chosen_z_max) for r in summary.outer_results]
                tprint(f"  Stability (Chosen Configs): {pairs}")

            # Per-bucket max hold hours: MR = shorter (reversion is fast), TF = longer
            bucket_hold = 12 if k == "mr" else 24

            # Per-bucket profit-protection anchored to EMPIRICAL MFE quantiles
            # (not ATR%, which is wildly mis-scaled at ~57%)
            mfe_s = summary.empirical_mfe_stats or {}
            mfe_med = mfe_s.get("mfe_median", 0.01)
            mfe_p25 = mfe_s.get("mfe_p25", 0.005)
            mfe_p75 = mfe_s.get("mfe_p75", 0.02)
            mae_med = mfe_s.get("mae_median", 0.01)
            mae_p75 = mfe_s.get("mae_p75", 0.02)

            # BE threshold: trigger at p25 of MFE (protect early — most losers saw at least this much profit)
            # MR: tighter (p25), TF: slightly wider (between p25 and median)
            if k == "mr":
                be_pct = max(0.003, mfe_p25)
            else:
                be_pct = max(0.003, 0.5 * (mfe_p25 + mfe_med))

            # Profit-lock: trigger at p25 MFE (early protection — most winners reach this)
            lock_pct = max(0.005, mfe_p25)
            # Lock amount: lock 50% of p25 MFE as real profit
            lock_amt = max(0.002, 0.50 * mfe_p25)

            # Max giveback: exit if return drops more than 50-65% from peak MFE
            # MR: tighter giveback (reversion is fast), TF: slightly wider
            giveback_frac = 0.50 if k == "mr" else 0.60
            giveback_pct = max(0.003, giveback_frac * mfe_med)

            # Max loss: hard cap at mae_p75 (75th percentile of adverse excursion)
            max_loss = max(0.01, min(0.05, mae_p75))

            tprint(f"  {side}_{k} profit-protection (empirical MFE): "
                   f"MFE p25={mfe_p25*100:.2f}% med={mfe_med*100:.2f}% p75={mfe_p75*100:.2f}% | "
                   f"MAE med={mae_med*100:.2f}% p75={mae_p75*100:.2f}% | "
                   f"BE@{be_pct*100:.2f}% Lock@{lock_pct*100:.2f}% LockAmt={lock_amt*100:.2f}% "
                   f"Giveback={giveback_pct*100:.2f}% MaxLoss={max_loss*100:.2f}%")

            bucket_risk = {
                "tp_mult": summary.final_tp_mult,
                "sl_mult": summary.final_sl_mult,
                "trail_mult": summary.final_trail_mult,
                "vol_lo": summary.final_lo,
                "vol_hi": summary.final_hi,
                "vol_z_max": summary.final_z_max,
                "max_hold_hours": bucket_hold,
                "k_sl": summary.final_sl_mult,
                "k_trail_start": summary.final_tp_mult,
                "k_trail_dist": summary.final_trail_mult,
                # Profit-protection (anchored to empirical MFE quantiles)
                "be_threshold_pct": be_pct,
                "profit_lock_pct": lock_pct,
                "profit_lock_amount": lock_amt,
                "giveback_pct": giveback_pct,
                "max_loss_pct": max_loss,
            }
            granular_risk[f"risk_{side}_{k}"] = bucket_risk
            granular_risk[f"risk_{k}_{cand_filter}"] = bucket_risk

    best_params = {
        "k_sl": cfg["risk_k_sl"],
        "k_trail_start": cfg["risk_k_trail_start"],
        "k_trail_dist": cfg["risk_k_trail_dist"],
        "granular_risk": granular_risk
    }

    return best_params
