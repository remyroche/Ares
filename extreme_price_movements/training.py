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
                        break
                    except ValueError as e:
                        tprint(f"Spike GMM ({mode}) failed with {n_try} components: {e}")
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

            # Prepare meta features
            # Extract features from df (excluding internal cols)
            drop_cols = ["__y_bin__", "__y_ret__", "__w__", "__ts__", "__symbol__"]
            feat_cols = [c for c in df.columns if c not in drop_cols]
            X_feats = df[feat_cols].fillna(0.0) #.astype(np.float32)

            tprint(f"Meta {side}_{k}: Training on {len(df)} samples...")
            meta = MetaModel()
            # Use single prediction input (OOF)
            X_meta = meta.prepare_meta_features(p_oof, X_feats, pred_col_name="pred_logit")
            meta.fit(X_meta, y_ret)

            # Save separately
            meta_models[f"{side}_{k}"] = meta
            tprint(f"Meta {side}_{k}: fitted.")

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

    return {"alpha_models": final_models, "exh_models": exh_models, "meta_models": meta_models, "spike_models": spike_models}

def select_best_horizon(panel, feats, mkt_gates, cfg, syms, ts, p_exh_hist):
    # DEPRECATED / LEGACY WRAPPER
    # This function is kept for backward compatibility if needed,
    # but strictly we should use the new split.
    # We can implement it by calling the new functions in sequence.
    datasets = generate_label_datasets(panel, feats, mkt_gates, cfg, syms, ts, p_exh_hist)
    return train_models_from_artifacts(datasets, cfg)

def optimize_risk_params(panel, feats, mkt_gates, cfg, train_syms, ts, p_exh_hist, alpha_models, meta_models):
    tprint("Entering function: optimize_risk_params in training.py (High Throughput Selection w/ Meta Scores)")

    granular_risk = {}

    # 1. Prepare Data for Optimization (Flattened for Numba)
    open_df = panel["open"]
    high_df = panel["high"]
    low_df = panel["low"]
    close_df = panel["close"]

    if "atr_pct" not in feats:
        return cfg

    atr_pct_df = feats["atr_pct"]
    window_base = 24 * 30
    atr_base_df = atr_pct_df.rolling(window_base, min_periods=24).median().fillna(method='bfill')
    atr_std_df = atr_pct_df.rolling(window_base, min_periods=24).std().fillna(method='bfill')
    z_df = (atr_pct_df - atr_base_df) / (atr_std_df + 1e-12)

    # Flatten
    assets = close_df.columns
    big_open = []; big_high = []; big_low = []; big_close = []
    big_atr = []; big_z = []; big_atr_base = []; big_X = []

    asset_offsets = {}
    current_offset = 0
    buffer_size = 100

    # Base Features for Optimization X
    feat_keys = cfg.get("causal_cols", [])
    if not feat_keys: feat_keys = ["trend_pct", "vol_pct", "ret24h"]

    for sym in assets:
        if sym not in atr_pct_df.columns: continue

        # Data
        c = close_df[sym].values.astype(np.float32)
        n = len(c)
        big_open.append(open_df[sym].values.astype(np.float32))
        big_high.append(high_df[sym].values.astype(np.float32))
        big_low.append(low_df[sym].values.astype(np.float32))
        big_close.append(c)

        big_atr.append(atr_pct_df[sym].values.astype(np.float32))
        big_atr_base.append(atr_base_df[sym].values.astype(np.float32))
        big_z.append(z_df[sym].values.astype(np.float32))

        # X
        x_list = []
        for k in feat_keys:
            if k in feats: x_list.append(feats[k][sym].values.astype(np.float32))
            else: x_list.append(np.zeros(n, dtype=np.float32))
        big_X.append(np.stack(x_list, axis=1) if x_list else np.zeros((n, 1), dtype=np.float32))

        asset_offsets[sym] = current_offset
        current_offset += n + buffer_size

        # Buffer
        nan_buf = np.full(buffer_size, np.nan, dtype=np.float32)
        big_open.append(nan_buf); big_high.append(nan_buf); big_low.append(nan_buf); big_close.append(nan_buf)
        big_atr.append(nan_buf); big_z.append(nan_buf); big_atr_base.append(nan_buf)
        big_X.append(np.full((buffer_size, len(feat_keys)), np.nan, dtype=np.float32))

    full_open = np.concatenate(big_open)
    full_high = np.concatenate(big_high)
    full_low = np.concatenate(big_low)
    full_close = np.concatenate(big_close)
    full_atr = np.concatenate(big_atr)
    full_atr_base = np.concatenate(big_atr_base)
    full_z = np.concatenate(big_z)
    full_X = np.concatenate(big_X, axis=0)

    # 2. Identify Candidates & Score Them to Filter Events
    lookback_days = 90
    ts_start = ts - pd.Timedelta(days=lookback_days)

    cand_mask = select_trade_candidates_vectorized(panel, feats, pct=cfg["trade_extreme_pct"], metric=cfg["trade_deviation_metric"])
    if cand_mask is None: return cfg

    mask = (cand_mask.index >= ts_start) & (cand_mask.index <= ts)
    final_mask = cand_mask & pd.Series(mask, index=cand_mask.index).fillna(False)

    # Extract Events for Batch Scoring
    # We need to construct a DataFrame similar to what engine.py does
    # ts, symbol, and all feature columns

    # Find active cells
    active_ts_idx, active_col_idx = np.where(final_mask.values)
    if len(active_ts_idx) == 0: return cfg

    event_ts = final_mask.index[active_ts_idx]
    event_sym = final_mask.columns[active_col_idx]

    tprint(f"Risk Opt: Scoring {len(event_ts)} candidate events...")

    # Build Feature Matrix for these events
    # We need all keys required by models
    req_keys = set()
    for k_list in ["mr_feature_keys", "tf_feature_keys", "spike_feature_keys", "meta_feature_keys", "causal_cols"]:
        if k_list in cfg: req_keys.update(cfg[k_list])

    data_parts = {"ts": event_ts, "symbol": event_sym}

    # Fast lookup for market gates
    mg = mkt_gates.reindex(event_ts)
    for c in ["mkt_ret24h", "mkt_ret6h", "mkt_trend", "mkt_rv", "G_VOL", "G_TREND"]:
        data_parts[c] = mg[c].values

    # Fast lookup for features
    for k in req_keys:
        if k in feats:
            data_parts[k] = _fast_lookup(feats[k], event_ts, event_sym)

    df_events = pd.DataFrame(data_parts).fillna(0.0)

    # Trend Classification
    if "trend_pct" in df_events.columns:
        df_events["kind"] = np.where(df_events["trend_pct"] > 0, "best", "worst")
    else:
        df_events["kind"] = "worst" # fallback

    # 3. Batch Predict & Classify
    buckets = {"long_tf": [], "short_mr": [], "long_mr": [], "short_tf": []}

    # We process by kind (best/worst)
    for kind, grp in df_events.groupby("kind"):
        if kind == "best":
            m_long = alpha_models.get("long", {}).get("tf", {})
            m_short = alpha_models.get("short", {}).get("mr", {})
            meta_long = meta_models.get("long_tf")
            meta_short = meta_models.get("short_mr")

            feat_keys_long = cfg.get("tf_feature_keys", cfg["causal_cols"])
            feat_keys_short = cfg.get("mr_feature_keys", cfg["causal_cols"])

            # Buckets to target
            target_long = "long_tf"
            target_short = "short_mr"
        else:
            m_long = alpha_models.get("long", {}).get("mr", {})
            m_short = alpha_models.get("short", {}).get("tf", {})
            meta_long = meta_models.get("long_mr")
            meta_short = meta_models.get("short_tf")

            feat_keys_long = cfg.get("mr_feature_keys", cfg["causal_cols"])
            feat_keys_short = cfg.get("tf_feature_keys", cfg["causal_cols"])

            target_long = "long_mr"
            target_short = "short_tf"

        if not m_long or not m_short: continue

        # Predict Long
        grp_long = apply_interaction_toggles(grp.copy(), feat_keys_long, ["G_VOL","G_TREND"], drop_raw=cfg["drop_raw_causal"])
        cols_l = m_long["feat_cols"]
        X_long = pd.DataFrame(0.0, index=grp.index, columns=cols_l)
        avail_l = [c for c in cols_l if c in grp_long.columns]
        X_long[avail_l] = grp_long[avail_l].fillna(0.0)
        p_long = m_long["model"].predict(X_long)

        s_long = (p_long - 0.5) * 0.1
        if meta_long:
            grp_num = grp.select_dtypes(include=[np.number]).copy()
            grp_t = apply_interaction_toggles(grp_num, feat_keys_long, ["G_VOL", "G_TREND"], drop_raw=False)
            X_m = meta_long.prepare_meta_features(p_long, grp_t, pred_col_name="pred_logit")
            if meta_long.selected_features:
                 for c in meta_long.selected_features:
                     if c not in X_m.columns: X_m[c] = 0.0
            s_long = meta_long.predict(X_m)

        # Predict Short
        grp_short = apply_interaction_toggles(grp.copy(), feat_keys_short, ["G_VOL","G_TREND"], drop_raw=cfg["drop_raw_causal"])
        cols_s = m_short["feat_cols"]
        X_short = pd.DataFrame(0.0, index=grp.index, columns=cols_s)
        avail_s = [c for c in cols_s if c in grp_short.columns]
        X_short[avail_s] = grp_short[avail_s].fillna(0.0)
        p_short = m_short["model"].predict(X_short)

        s_short = (p_short - 0.5) * 0.1
        if meta_short:
            grp_num = grp.select_dtypes(include=[np.number]).copy()
            grp_t = apply_interaction_toggles(grp_num, feat_keys_short, ["G_VOL", "G_TREND"], drop_raw=False)
            X_m = meta_short.prepare_meta_features(p_short, grp_t, pred_col_name="pred_logit")
            if meta_short.selected_features:
                 for c in meta_short.selected_features:
                     if c not in X_m.columns: X_m[c] = 0.0
            s_short = meta_short.predict(X_m)

        # Net Score
        net_score = s_long - s_short
        thr = cfg["thr_long"]

        # Filter and Collect Indices
        # We need to map back to (ts, sym) and then to flat_idx

        long_idxs = grp.index[net_score > thr]
        short_idxs = grp.index[net_score < -thr]

        # Helper to convert df rows to flat indices
        def get_flat_indices(df_subset):
            f_idxs = []
            for t, s in zip(df_subset["ts"], df_subset["symbol"]):
                if s not in asset_offsets: continue
                try:
                    t_idx = close_df.index.get_loc(t)
                    f_idxs.append(asset_offsets[s] + t_idx)
                except: continue
            return f_idxs

        buckets[target_long].extend(get_flat_indices(grp.loc[long_idxs]))
        buckets[target_short].extend(get_flat_indices(grp.loc[short_idxs]))

    # 4. Run Optimization per Bucket
    for bucket_key, indices in buckets.items():
        if not indices: continue

        # Parse key: "long_tf" -> side="long", k="tf"
        parts = bucket_key.split("_")
        side, k = parts[0], parts[1]

        idx_arr = np.array(indices, dtype=np.int32)
        tprint(f"Risk Opt Bucket {bucket_key}: {len(idx_arr)} events")

        if len(idx_arr) < 20:
             # Default
             granular_risk[f"risk_{bucket_key}"] = {
                "tp_mult": 1.0, "sl_mult": 0.5,
                "k_sl": 2.0, "k_trail_start": 2.0, "k_trail_dist": 1.0
             }
             continue

        summary = run_tp_sl_selection_fast(
            X=full_X,
            open_=full_open, high=full_high, low=full_low, close=full_close,
            atr_pct=full_atr, z=full_z, atr_base_pct=full_atr_base,
            event_idx=idx_arr,
            horizon=24,
            max_events=2000,
            tp_mult_grid=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
            sl_mult_grid=[0.5, 1.0, 1.5, 2.0, 2.5],
            entry_mode="next_open"
        )

        tprint(f"Optimized {bucket_key}: TP={summary.final_tp_mult:.2f}, SL={summary.final_sl_mult:.2f}")
        granular_risk[f"risk_{bucket_key}"] = {
            "tp_mult": summary.final_tp_mult,
            "sl_mult": summary.final_sl_mult,
            "k_sl": 2.0, "k_trail_start": 2.0, "k_trail_dist": 1.0
        }

    best_params = {
        "k_sl": cfg["risk_k_sl"],
        "k_trail_start": cfg["risk_k_trail_start"],
        "k_trail_dist": cfg["risk_k_trail_dist"],
        "granular_risk": granular_risk
    }
    return best_params
