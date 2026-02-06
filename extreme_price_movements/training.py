import numpy as np
import pandas as pd
from extreme_price_movements.utils import tprint
from extreme_price_movements.model_race import ModelRace
from extreme_price_movements.meta_model import MetaModel
from extreme_price_movements.exhaustion import ExhaustionModel
from extreme_price_movements.exhaustion import ExhaustionModel
from extreme_price_movements.feature_selection_extreme_events import mdi_feature_selection_v3
from extreme_price_movements.candidates import select_trade_candidates_hourly, select_trade_candidates_vectorized
import extreme_price_movements.fast_funcs as ff
from extreme_price_movements.labeling import compute_trailing_atr_labels, compute_triple_barrier_labels
from extreme_price_movements.sample_weights import build_label_time_ranges, compute_sample_weights_with_uniqueness
from sklearn.mixture import GaussianMixture
from extreme_price_movements.optimise_tpsl_ratio import (
    run_tp_sl_selection_fast,
    calibrate_atr_base_pct,
    compute_vol_z_log_mad,
    PurgedKFold,
)

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
    tprint(f"Entering function: apply_interaction_toggles in training.py")
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

def compute_weights_logic(df, cfg, model_kind):
    tprint(f"Entering function: compute_weights_logic in training.py")
    from extreme_price_movements.model_mr import compute_mr_weights
    from extreme_price_movements.model_tf import compute_tf_weights
    if model_kind == "mr": return compute_mr_weights(df, cfg)
    else: return compute_tf_weights(df, cfg)

def scaled_atr_pct(
    atr_pct: float,
    z: float,
    atr_base_pct: float,
    *,
    z_max: float = 3.0,
    lo: float = 0.03,
    hi: float = 0.06,
    eps: float = 1e-12,
):
    """
    ATR-informed, shock-scaled, bounded barrier percent.
    Vectorized using NumPy.
    """
    # 1) Shock control
    shock = np.clip(z, 0.0, z_max)

    # 2) Dynamic multiplier 'a' so that:
    #    atr_base_pct * (1 + a*z_max) ≈ hi
    a = (hi / np.maximum(atr_base_pct, eps) - 1.0) / z_max

    # 3) Multiplicative scaling
    raw = atr_pct * (1.0 + a * shock)

    # 4) Enforce cross-asset low/high targets
    return np.clip(raw, lo, hi)

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
                model_up = ExhaustionModel(C=cfg["exh_C"], l1_ratio=cfg["exh_l1_ratio"])
                model_up.fit(X, y, sample_weight=w)
            else:
                tprint("Not enough data for UP model.")
                model_up = None
        if model_up and model_up.model:
            Xp = _build_pred_X(feats, mkt_gates, cfg, ts, up_syms, feature_key="exh_feature_keys")
            if not Xp.empty:
                probs = model_up.predict_proba(Xp)
                probs = np.clip(probs * 2.0, 0.0, 1.0)
                out_probs.loc[up_syms] = probs
    if dn_syms:
        if models and "down" in models: model_dn = models["down"]
        else:
            tprint("Training DOWN model...")
            X, y, w, _ = build_exhaustion_Xy(panel, feats, mkt_gates, cfg, ts, lookback, valid_syms, trend_filter="down")
            if X is not None and len(y) > 100:
                model_dn = ExhaustionModel(C=cfg["exh_C"], l1_ratio=cfg["exh_l1_ratio"])
                model_dn.fit(X, y, sample_weight=w)
            else:
                tprint("Not enough data for DOWN model.")
                model_dn = None
        if model_dn and model_dn.model:
            Xp = _build_pred_X(feats, mkt_gates, cfg, ts, dn_syms, feature_key="exh_feature_keys")
            if not Xp.empty:
                probs = model_dn.predict_proba(Xp)
                out_probs.loc[dn_syms] = probs
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
        model_up = ExhaustionModel(C=cfg["exh_C"], l1_ratio=cfg["exh_l1_ratio"])
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
        model_dn = ExhaustionModel(C=cfg["exh_C"], l1_ratio=cfg["exh_l1_ratio"])
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
            preds = np.clip(preds * 2.0, 0.0, 1.0)

            # 2. Overlay OOF predictions where available
            if arr_oof_up is not None:
                oof_col = arr_oof_up[:, j]
                valid_oof = ~np.isnan(oof_col)
                if valid_oof.any():
                    # Apply same scaling to OOF
                    preds[valid_oof] = np.clip(oof_col[valid_oof] * 2.0, 0.0, 1.0)
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
    trend_filter=None, feature_key=None,
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
    # Subsample: keep only every 4th hour
    window_cand = window_cand[window_cand.index.hour % 4 == 0]

    if feature_key:
        feat_keys = cfg.get(feature_key, [])
    else:
        feat_keys = cfg.get("causal_cols", [])

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

    y_bin = (pnl > 0).astype(np.int8)

    # Weights from ret24h
    pa = np.abs(np.nan_to_num(_fast_lookup(feats["ret24h"], event_ts, event_sym), nan=0.0))
    weights_raw = np.log(1 + pa)

    # Class imbalance correction: inverse-frequency weighting
    n_pos = (y_bin == 1).sum()
    n_neg = (y_bin == 0).sum()
    if n_pos > 0 and n_neg > 0:
        n_total = n_pos + n_neg
        w_pos = n_total / (2.0 * n_pos)
        w_neg = n_total / (2.0 * n_neg)
        class_mult = np.where(y_bin == 1, w_pos, w_neg).astype(np.float32)
        weights_raw = weights_raw * class_mult

    # Volatility-based weight multiplier: range_24h * vol_z, bounded [0.5, 2.0]
    if "range_24h_pct" in feats and "volatility_zscore" in feats:
        r24 = np.abs(np.nan_to_num(_fast_lookup(feats["range_24h_pct"], event_ts, event_sym), nan=0.0))
        vz = np.abs(np.nan_to_num(_fast_lookup(feats["volatility_zscore"], event_ts, event_sym), nan=0.0))
        vol_mult = np.clip(r24 * vz, 0.5, 2.0).astype(np.float32)
        weights_raw = weights_raw * vol_mult

    # Build feature DataFrame
    # event_ts is a DatetimeIndex, event_sym is a numpy array
    ts_arr = event_ts.values if hasattr(event_ts, 'values') else event_ts
    sym_arr = event_sym.values if hasattr(event_sym, 'values') else event_sym
    parts = {"ts": ts_arr, "symbol": sym_arr, "y_bin": y_bin, "y_ret": pnl.astype(np.float32), "w": weights_raw.astype(np.float32)}

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

    df = apply_interaction_toggles(df, feat_keys, ["G_VOL", "G_TREND"], drop_raw=cfg["drop_raw_causal"])
    y_bin = df.pop("y_bin").values.astype(int)
    y_ret = df.pop("y_ret").values.astype(np.float32)

    X_out = df.drop(columns=["ts", "symbol"], errors="ignore").astype(np.float32)
    X_out.index = df.index

    df_meta = df[["ts", "symbol"]] if "ts" in df.columns else pd.DataFrame(index=df.index)

    return X_out, y_bin, y_ret, list(X_out.columns), weights, df_meta

def train_spike_anatomy_model(panel, feats, mkt_gates, cfg, syms, ts_end, _cached_cand_mask=None):
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
    ref_key = available_keys[0]
    ref_df = feats[ref_key]

    # Validate strict alignment for optimized lookup
    # We rely on all features having identical Index/Columns implies row_idx/col_idx are valid for all.
    for k in available_keys[1:]:
        if k in feats and (len(feats[k].index) != len(ref_df.index) or not feats[k].index.equals(ref_df.index) or not feats[k].columns.equals(ref_df.columns)):
             tprint(f"Warning: Feature {k} structure mismatch in Spike Anatomy. alignment_check=FAIL")
             # Fallback to slower safe lookup or skip? 
             # Given this is a critical optimization, we return None (fail safe) or reindex (slow).
             # Let's fail safe to alert the user of data corruption.
             return None
    
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

    # 1. Spike Anatomy
    spike_df = train_spike_anatomy_model(panel, feats, mkt_gates, cfg, syms, ts, _cached_cand_mask=cached_cand_mask)
    if spike_df is not None:
        datasets["spike_anatomy"] = spike_df

    # 2. Alpha Models (MR/TF)
    trade_sides = ["long", "short"]
    kinds = ["mr", "tf"]
    horizons = cfg["label_horizons_hours"]

    # Pre-compute triple barrier labels per (H, side) — shared across MR/TF
    # Use run_tp_sl_selection_fast to independently optimise TP and SL multipliers
    tb_cache = {}

    if "atr_pct" in feats:
        atr_pct_df = feats["atr_pct"]
        window_size = 24 * 30
        atr_base_df = atr_pct_df.rolling(window_size, min_periods=24).median()
        atr_std_df = atr_pct_df.rolling(window_size, min_periods=24).std()
        z_score_df = (atr_pct_df - atr_base_df) / (atr_std_df + 1e-12)
        b_pct_vals = scaled_atr_pct(atr_pct_df.values, z_score_df.values, atr_base_df.values, z_max=3.0, lo=0.03, hi=0.06)
        barrier_pct_df = pd.DataFrame(b_pct_vals, index=atr_pct_df.index, columns=atr_pct_df.columns)
    else:
        atr_pct_df = None
        barrier_pct_df = None

    for H in horizons:
        for side in trade_sides:
            tprint(f"Pre-computing triple barrier labels H={H} side={side}...")

            if atr_pct_df is None:
                # Fallback: fixed barriers
                tb_labels, tb_returns = compute_triple_barrier_labels(panel, 0.05, 0.025, H, side=side)
                tb_cache[(H, side)] = (tb_labels, tb_returns)
                continue

            # --- Optimise TP/SL via run_tp_sl_selection_fast ---
            # Use representative asset from market basket for speed
            close_df = panel["close"]
            opt_feat_keys = cfg.get("causal_cols", [])[:10]
            tp_mult, sl_mult = 1.0, 0.5  # defaults

            # Pick representative asset: first market basket member with data
            rep_sym = None
            for s in cfg.get("market_basket", []):
                if s in close_df.columns and s in atr_pct_df.columns:
                    rep_sym = s
                    break
            if rep_sym is None and len(close_df.columns) > 0:
                rep_sym = close_df.columns[0]

            if rep_sym is not None:
                panel_idx = close_df.index
                o = panel["open"][rep_sym].values.astype(np.float32)
                h = panel["high"][rep_sym].values.astype(np.float32)
                l = panel["low"][rep_sym].values.astype(np.float32)
                c = close_df[rep_sym].values.astype(np.float32)
                # Align ATR arrays to panel index
                a = np.nan_to_num(atr_pct_df[rep_sym].reindex(panel_idx).values.astype(np.float32), nan=0.01)
                ab = np.nan_to_num(atr_base_df[rep_sym].reindex(panel_idx).values.astype(np.float32), nan=0.01)
                zv = np.nan_to_num(z_score_df[rep_sym].reindex(panel_idx).values.astype(np.float32), nan=0.0)

                # Features for X — align to panel index
                panel_idx = close_df.index
                x_list = []
                for fk in opt_feat_keys:
                    if fk in feats and rep_sym in feats[fk].columns:
                        aligned = feats[fk][rep_sym].reindex(panel_idx).values.astype(np.float32)
                        x_list.append(np.nan_to_num(aligned, nan=0.0))
                    else:
                        x_list.append(np.zeros(len(c), dtype=np.float32))
                x_arr = np.stack(x_list, axis=1) if x_list else np.zeros((len(c), 1), dtype=np.float32)

                # Event indices from candidate mask
                ts_start_opt = ts - pd.Timedelta(hours=int(cfg["train_lookback_hours"]))
                if cached_cand_mask is not None and rep_sym in cached_cand_mask.columns:
                    cand_series = cached_cand_mask[rep_sym]
                    cand_series = cand_series[(cand_series.index >= ts_start_opt) & (cand_series.index <= ts)]
                    event_ts_idx = cand_series[cand_series].index
                    event_indices = close_df.index.get_indexer(event_ts_idx)
                    event_indices = event_indices[event_indices >= 0].astype(np.int32)
                else:
                    # Fallback: use every 4th hour as events
                    mask = (close_df.index >= ts_start_opt) & (close_df.index <= ts)
                    event_indices = np.where(mask)[0][::4].astype(np.int32)

                tprint(f"TP/SL optimization: {len(event_indices)} events on {rep_sym} for H={H} side={side}")

                if len(event_indices) >= 50:
                    try:
                        summary = run_tp_sl_selection_fast(
                            X=x_arr,
                            open_=o, high=h, low=l, close=c,
                            atr_pct=a, z=zv, atr_base_pct=ab,
                            event_idx=event_indices,
                            horizon=H,
                            max_events=3000,
                            tp_mult_grid=[0.6, 0.8, 1.0, 1.25, 1.5],
                            sl_mult_grid=[0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
                            entry_mode="next_open",
                        )
                        tp_mult = summary.final_tp_mult
                        sl_mult = summary.final_sl_mult
                        tprint(f"Optimised TP/SL for H={H} side={side}: tp_mult={tp_mult:.2f}, sl_mult={sl_mult:.2f}")
                    except Exception as e:
                        tprint(f"TP/SL optimization failed for H={H} side={side}: {e}. Using defaults.")
                else:
                    tprint(f"Not enough events ({len(event_indices)}) for TP/SL optimization. Using defaults.")

            tp_df = tp_mult * barrier_pct_df
            sl_df = sl_mult * barrier_pct_df
            tb_labels, tb_returns = compute_triple_barrier_labels(panel, tp_df, sl_df, H, side=side)
            tb_cache[(H, side)] = (tb_labels, tb_returns)

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

                X, y, y_ret, cols, w, meta_idx = build_hourly_training_set_and_weights(
                    panel, feats, mkt_gates, cfg, syms, ts, p_exh_hist, H, k,
                    trend_filter=trend_filter, feature_key=feat_key,
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

    return datasets

def train_models_from_artifacts(datasets, cfg):
    tprint(f"Entering function: train_models_from_artifacts in training.py")
    directions = ["up", "down"]
    kinds = ["mr", "tf"]
    final_models = {}

    # 1. Train Spike Model
    spike_model = None
    if "spike_anatomy" in datasets:
        tprint("Training Spike Model...")
        df_spike = datasets["spike_anatomy"]
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
        tprint(f"Spike GMM: {len(df_spike_num)} samples, {df_spike_num.shape[1]} features, {n_comp} components")
        if len(df_spike_num) >= 50 and df_spike_num.shape[1] >= 2:
            # Standardize before fitting to avoid ill-conditioned covariance
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(df_spike_num)
            for n_try in [n_comp, max(1, n_comp // 2), 1]:
                try:
                    gmm = GaussianMixture(n_components=n_try, covariance_type='diag', reg_covar=1e-2, random_state=42)
                    gmm.fit(X_scaled)
                    spike_model = {"gmm": gmm, "scaler": scaler, "columns": list(df_spike_num.columns)}
                    tprint(f"Spike GMM fitted with {n_try} components.")
                    break
                except ValueError as e:
                    tprint(f"Spike GMM failed with {n_try} components: {e}")
                    continue

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

                y = df["__y_bin__"].values.astype(int)
                y_ret = df["__y_ret__"].values.astype(np.float32)
                w = df["__w__"].values.astype(np.float32)

                drop_cols = ["__y_bin__", "__y_ret__", "__w__", "__ts__", "__symbol__"]
                X = df.drop(columns=[c for c in drop_cols if c in df.columns])
                cols = list(X.columns)

                # Derive strategy context for logging
                cand_filter = "unknown"
                if side == "long":
                    if k == "mr": cand_filter = "worst" # ret < 0
                    else: cand_filter = "best"          # ret > 0
                else: # short
                    if k == "mr": cand_filter = "best"  # ret > 0
                    else: cand_filter = "worst"         # ret < 0

                tprint(f"Training {side} {k} ({cand_filter}) H={H} (n={len(X)})...")

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
                    cumulative_cap=0.99,
                    min_share=0.0005,
                    min_features=10,
                    max_features_pct=0.8
                )
                
                selected_feats = sel_res.selected_features
                tprint(f"MDI selected {len(selected_feats)} features (from {X.shape[1]}).")
                
                X_sel = X[selected_feats]
                cols = list(selected_feats)
                
                tprint(f"  Class dist: 0={int((y==0).sum())} ({(y==0).mean()*100:.1f}%), 1={int((y==1).sum())} ({(y==1).mean()*100:.1f}%)")

                race = ModelRace(kind=k, n_splits=3)
                race.fit(X_sel, y, sample_weight=w, returns=y_ret)
                score = race.metrics.get(race.best_model_name, -1.0)
                dm = race.detailed_metrics.get(race.best_model_name, {})
                tprint(f"Finished {side} {k} H={H}: Winner={race.best_model_name}, Score={score:.4f}, AUC={dm.get('AUC',0):.4f}, IC={dm.get('IC',0):.4f}, BSS={dm.get('BSS',0):.4f}")

                if score > best_ic:
                    best_ic = score
                    best_m = {"model": race, "H": H, "feat_cols": cols}

            final_models[side][k] = best_m

    # 3. Train Meta Models (using OOF predictions from alpha models on union of samples)
    meta_models = {}
    for side in trade_sides:
        mr_conf = final_models[side]["mr"]
        tf_conf = final_models[side]["tf"]
        if not mr_conf or not tf_conf:
            tprint(f"Meta {side}: skipped (missing MR or TF model)")
            meta_models[side] = None; continue

        H_mr = mr_conf["H"]
        H_tf = tf_conf["H"]

        key_mr = f"train_{side}_mr_{H_mr}"
        key_tf = f"train_{side}_tf_{H_tf}"

        if key_mr not in datasets or key_tf not in datasets:
            tprint(f"Meta {side}: skipped (missing datasets)")
            meta_models[side] = None; continue

        df_mr = datasets[key_mr].copy()
        df_tf = datasets[key_tf].copy()

        if "__ts__" not in df_mr.columns or "__symbol__" not in df_mr.columns:
            tprint(f"Meta {side}: skipped (missing meta columns)")
            meta_models[side] = None; continue

        # Build index for both
        idx_mr = pd.MultiIndex.from_frame(df_mr[["__ts__", "__symbol__"]])
        idx_tf = pd.MultiIndex.from_frame(df_tf[["__ts__", "__symbol__"]])
        df_mr.index = idx_mr
        df_tf.index = idx_tf

        # Union of all samples from both origins
        all_idx = idx_mr.union(idx_tf)
        tprint(f"Meta {side}: MR={len(idx_mr)}, TF={len(idx_tf)}, union={len(all_idx)}, overlap={len(idx_mr.intersection(idx_tf))}")

        if len(all_idx) < 200:
            tprint(f"Meta {side}: skipped (union too small: {len(all_idx)})")
            meta_models[side] = None; continue

        # Get OOF predictions from alpha models
        mr_race = mr_conf["model"]
        tf_race = tf_conf["model"]

        # For samples IN the training set: use OOF predictions
        # For samples NOT in the training set: use fitted model predictions
        drop_cols = ["__y_bin__", "__y_ret__", "__w__", "__ts__", "__symbol__"]

        p_mr_all = np.full(len(all_idx), 0.5, dtype=np.float32)
        p_tf_all = np.full(len(all_idx), 0.5, dtype=np.float32)
        y_ret_all = np.zeros(len(all_idx), dtype=np.float32)
        w_all = np.ones(len(all_idx), dtype=np.float32)

        # Map all_idx positions
        all_idx_list = list(all_idx)

        # MR OOF predictions for MR samples
        if mr_race.oof_probs is not None:
            for i, idx_val in enumerate(all_idx_list):
                if idx_val in idx_mr:
                    pos_in_mr = idx_mr.get_loc(idx_val)
                    if isinstance(pos_in_mr, int):
                        p_mr_all[i] = mr_race.oof_probs[pos_in_mr]
                    elif hasattr(pos_in_mr, '__iter__'):
                        p_mr_all[i] = mr_race.oof_probs[pos_in_mr[0]] if len(pos_in_mr) > 0 else 0.5

        # TF OOF predictions for TF samples
        if tf_race.oof_probs is not None:
            for i, idx_val in enumerate(all_idx_list):
                if idx_val in idx_tf:
                    pos_in_tf = idx_tf.get_loc(idx_val)
                    if isinstance(pos_in_tf, int):
                        p_tf_all[i] = tf_race.oof_probs[pos_in_tf]
                    elif hasattr(pos_in_tf, '__iter__'):
                        p_tf_all[i] = tf_race.oof_probs[pos_in_tf[0]] if len(pos_in_tf) > 0 else 0.5

        # For samples not in MR set, predict with fitted MR model
        mr_missing_mask = ~np.array([idx_val in idx_mr for idx_val in all_idx_list])
        if mr_missing_mask.any():
            missing_in_tf = [all_idx_list[i] for i in range(len(all_idx_list)) if mr_missing_mask[i]]
            X_missing = df_tf.loc[missing_in_tf].drop(columns=[c for c in drop_cols if c in df_tf.columns])
            try:
                p_mr_all[mr_missing_mask] = mr_race.predict(X_missing[mr_conf["feat_cols"]])
            except Exception:
                pass  # Keep 0.5 default

        # For samples not in TF set, predict with fitted TF model
        tf_missing_mask = ~np.array([idx_val in idx_tf for idx_val in all_idx_list])
        if tf_missing_mask.any():
            missing_in_mr = [all_idx_list[i] for i in range(len(all_idx_list)) if tf_missing_mask[i]]
            X_missing = df_mr.loc[missing_in_mr].drop(columns=[c for c in drop_cols if c in df_mr.columns])
            try:
                p_tf_all[tf_missing_mask] = tf_race.predict(X_missing[tf_conf["feat_cols"]])
            except Exception:
                pass  # Keep 0.5 default

        # y_ret: use TF returns where available, else MR returns
        for i, idx_val in enumerate(all_idx_list):
            if idx_val in idx_tf:
                pos = idx_tf.get_loc(idx_val)
                pos = pos if isinstance(pos, int) else (pos[0] if hasattr(pos, '__iter__') else pos)
                y_ret_all[i] = df_tf.iloc[pos]["__y_ret__"]
                w_all[i] = df_tf.iloc[pos]["__w__"]
            elif idx_val in idx_mr:
                pos = idx_mr.get_loc(idx_val)
                pos = pos if isinstance(pos, int) else (pos[0] if hasattr(pos, '__iter__') else pos)
                y_ret_all[i] = df_mr.iloc[pos]["__y_ret__"]
                w_all[i] = df_mr.iloc[pos]["__w__"]

        tprint(f"Meta {side}: p_mr mean={np.mean(p_mr_all):.4f} std={np.std(p_mr_all):.4f}")
        tprint(f"Meta {side}: p_tf mean={np.mean(p_tf_all):.4f} std={np.std(p_tf_all):.4f}")

        # Build shared feature matrix for meta model from label datasets
        # Find columns common to both MR and TF datasets (excluding meta cols)
        mr_feat_cols = set(df_mr.columns) - set(drop_cols)
        tf_feat_cols = set(df_tf.columns) - set(drop_cols)
        shared_feat_cols = sorted(mr_feat_cols & tf_feat_cols)
        tprint(f"Meta {side}: {len(shared_feat_cols)} shared feature columns available")

        # Extract shared features for all samples (prefer TF source, fallback MR)
        meta_feat_rows = []
        for i, idx_val in enumerate(all_idx_list):
            if idx_val in idx_tf:
                pos = idx_tf.get_loc(idx_val)
                pos = pos if isinstance(pos, int) else (pos[0] if hasattr(pos, '__iter__') else pos)
                meta_feat_rows.append(df_tf.iloc[pos][shared_feat_cols].values)
            elif idx_val in idx_mr:
                pos = idx_mr.get_loc(idx_val)
                pos = pos if isinstance(pos, int) else (pos[0] if hasattr(pos, '__iter__') else pos)
                meta_feat_rows.append(df_mr.iloc[pos][shared_feat_cols].values)
            else:
                meta_feat_rows.append(np.zeros(len(shared_feat_cols)))

        df_meta_feats = pd.DataFrame(meta_feat_rows, columns=shared_feat_cols).fillna(0.0).astype(np.float32)
        tprint(f"Meta {side}: df_meta_feats shape={df_meta_feats.shape}")

        meta = MetaModel()
        X_meta_final = meta.prepare_meta_features(p_tf_all, p_mr_all, df_meta_feats)
        tprint(f"Meta {side}: X_meta shape={X_meta_final.shape}, columns={list(X_meta_final.columns[:5])}...")
        meta.fit(X_meta_final, y_ret_all)
        meta_models[side] = meta
        tprint(f"Meta {side}: fitted on {len(all_idx)} samples, selected={meta.selected_features}")

    # 4. Train Exhaustion Models
    exh_models = {}
    for d in directions:
        key = f"exh_{d}"
        if key in datasets:
            df = datasets[key]
            if len(df) > 100:
                y = df["__y__"].values.astype(int)
                w = df["__w__"].values.astype(np.float32)
                # Drop meta/targets
                # In `build_exhaustion_Xy`, X columns are features + G_VOL/G_TREND
                # The reset_index added `ts` `symbol`.
                drop_cols = ["__y__", "__w__", "ts", "symbol"]
                X = df.drop(columns=[c for c in drop_cols if c in df.columns])

                tprint(f"Exhaustion {d}: {len(X)} samples, {X.shape[1]} features, class dist: 0={int((y==0).sum())} 1={int((y==1).sum())}")
                m = ExhaustionModel(C=cfg["exh_C"], l1_ratio=cfg["exh_l1_ratio"])
                m.fit(X, y, sample_weight=w)
                if m.model is not None:
                    tprint(f"Exhaustion {d}: fitted, {len(m.selected_features)} selected features")
                else:
                    tprint(f"Exhaustion {d}: fitting failed (model is None)")
                exh_models[d] = m
            else:
                tprint(f"Exhaustion {d}: skipped (only {len(df)} samples)")
                exh_models[d] = None
        else:
            tprint(f"Exhaustion {d}: no dataset found")
            exh_models[d] = None

    return {"alpha_models": final_models, "exh_models": exh_models, "meta_models": meta_models, "spike_model": spike_model}

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
    atr_base_df = atr_pct_df.rolling(window_base, min_periods=24).median().fillna(method='bfill')
    # Using the fast numpy functions if possible, but pandas is easier for alignment here
    # For Z, we need a robust one.
    # Let's use the one from fast_funcs if available or re-implement simple robust Z.
    atr_std_df = atr_pct_df.rolling(window_base, min_periods=24).std().fillna(method='bfill')
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

    mask = (cand_mask.index >= ts_start) & (cand_mask.index <= ts)
    final_mask = cand_mask & pd.Series(mask, index=cand_mask.index).fillna(False)

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

    assets = close_df.columns
    # Collect arrays
    big_open = []
    big_high = []
    big_low = []
    big_close = []
    big_atr = []
    big_z = []
    big_atr_base = []
    big_X = [] # Features

    asset_offsets = {}
    current_offset = 0
    buffer_size = 100 # larger than horizon

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

        a = atr_pct_df[sym].values.astype(np.float32)
        b = atr_base_df[sym].values.astype(np.float32)
        z_v = z_df[sym].values.astype(np.float32)

        # Features
        # Gather into (T, F)
        x_list = []
        for k in feat_keys:
            if k in feats:
                x_list.append(feats[k][sym].values.astype(np.float32))
            else:
                x_list.append(np.zeros(len(c), dtype=np.float32))
        x_arr = np.stack(x_list, axis=1)

        # Append
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

    # Concatenate
    full_open = np.concatenate(big_open)
    full_high = np.concatenate(big_high)
    full_low = np.concatenate(big_low)
    full_close = np.concatenate(big_close)
    full_atr = np.concatenate(big_atr)
    full_atr_base = np.concatenate(big_atr_base)
    full_z = np.concatenate(big_z)
    full_X = np.concatenate(big_X, axis=0)

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

            # Collect indices
            indices = []

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
                    # t is timestamp. We need integer index in the asset array.
                    # Assuming all assets have same index 'idx' (from panel)
                    # We can map t to integer index in panel

                    # idx is sorted? Yes.
                    # Find integer location
                    try:
                        time_idx = idx.get_loc(t)
                    except KeyError:
                        continue

                    flat_idx = asset_offsets[sym] + time_idx
                    indices.append(flat_idx)

            indices = np.array(indices, dtype=np.int32)
            tprint(f"Bucket {side} {k} ({cand_filter}): {len(indices)} events")

            if len(indices) < 50:
                tprint("Not enough events, using defaults.")
                granular_risk[f"risk_{side}_{k}"] = {
                    "k_sl": cfg["risk_k_sl"],
                    "k_trail_start": cfg["risk_k_trail_start"],
                    "k_trail_dist": cfg["risk_k_trail_dist"],
                    # "tp_mult": ... we store optimized params
                }
                continue

            # Run optimization
            # Note: run_tp_sl_selection_fast selects tp_mult and sl_mult for TRIPLE BARRIER.
            # But the system might use Trailing ATR logic at execution time?
            # If we switch to Triple Barrier execution, we use these.
            # If we use Trailing ATR, we map them: k_sl = sl_mult (approx).
            # The prompt implies we want to find optimal "TP:SL ratio" and levels.

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
                horizon=24, # Fixed horizon for now
                max_events=2000, # Cap for speed
                tp_mult_grid=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0], # Wider grid
                sl_mult_grid=[0.5, 1.0, 1.5, 2.0, 2.5],
                entry_mode="next_open"
            )

            tprint(f"Optimized {side} {k}: TP_mult={summary.final_tp_mult:.2f}, SL_mult={summary.final_sl_mult:.2f}")

            # Store in granular risk
            # We map these to the config keys used by Triple Barrier execution
            granular_risk[f"risk_{side}_{k}"] = {
                "tp_mult": summary.final_tp_mult,
                "sl_mult": summary.final_sl_mult,
                # Fallback keys for legacy
                "k_sl": 2.0,
                "k_trail_start": 2.0,
                "k_trail_dist": 1.0
            }

    # Best params for default (not really used if granular is active)
    best_params = {
        "k_sl": cfg["risk_k_sl"],
        "k_trail_start": cfg["risk_k_trail_start"],
        "k_trail_dist": cfg["risk_k_trail_dist"],
        "granular_risk": granular_risk
    }

    return best_params
