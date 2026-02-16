import numpy as np
import pandas as pd
from .utils import tprint
from .model_race import ModelRace
from .meta_model import MetaModel, MetaClassifierModel
from .exhaustion import ExhaustionModel
from .feature_selection_extreme_events import mdi_feature_selection_v3
from .candidates import select_trade_candidates_hourly, select_trade_candidates_vectorized
import extreme_price_movements.fast_funcs as ff
from .labeling import compute_trailing_atr_labels, compute_triple_barrier_labels
from .sample_weights import build_label_time_ranges, compute_sample_weights_with_uniqueness, compute_mfe_mae_weights
from .sample_weight_optimization import (
    combine_weights_safely,
    compute_vol_weights,
    compute_liquidity_weights,
    compute_distance_to_barrier_weights,
    compute_recency_weights,
    optimize_component_weights,
    log_weight_statistics,
    select_test_feature_frame,
)
from .offline_optimisers.params_store import apply_offline_optimizer_best_params
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

import os
import json
from datetime import datetime, timezone

# =============================================================================
# UNIFIED BARRIER FACTORY - Canonical TP/SL geometry (best-of-both pipelines)
# =============================================================================
def _coerce_feature_to_panel_df(x, panel, name: str, fill_value: float = np.nan):
    """Ensure feature is a DataFrame aligned to panel['close'] index/columns."""
    close = panel["close"]
    if isinstance(x, pd.DataFrame):
        return x.reindex(index=close.index, columns=close.columns)
    if isinstance(x, np.ndarray):
        if x.shape == close.shape:
            tprint(f"Warning: feature '{name}' provided as ndarray; coercing to DataFrame aligned to panel.")
            return pd.DataFrame(x, index=close.index, columns=close.columns)
        tprint(
            f"Warning: feature '{name}' ndarray shape {x.shape} mismatches panel {close.shape}; "
            f"using fill_value={fill_value}."
        )
        return pd.DataFrame(fill_value, index=close.index, columns=close.columns)
    if x is None:
        return pd.DataFrame(fill_value, index=close.index, columns=close.columns)
    tprint(f"Warning: feature '{name}' has unexpected type {type(x)}; using fill_value={fill_value}.")
    return pd.DataFrame(fill_value, index=close.index, columns=close.columns)


def compute_barrier_factory(
    atr_pct: pd.DataFrame,
    window_size: int = 24 * 30,
    k_tp: float = 1.0,
    sl_base_mult: float = 0.5,
    horizon: int = 4,
    H_base: int = 4,
    disp_floor: float = 0.1,
    z_max: float = 3.0,
    k_reg: float = 0.3,
    m_lo: float = 0.7,
    m_hi: float = 1.5,
    sl_lo: float = 0.4,
    sl_hi: float = 0.7,
    z_gate: float = 1.0,
    tp_lo: float = 0.02,   # Lower bound for TP (2%)
    tp_hi: float = 0.06,  # Upper bound for TP (6%)
    return_components: bool = False,
) -> tuple:
    """
    Canonical barrier factory - unified TP/SL geometry for both pipelines.
    
    Formula:
        tp_raw = k_tp * atr_pct * m(z) * sqrt(H / H_base)
        tp = clamp(tp_raw, tp_lo, tp_hi)  # Match old scaled_atr_pct behavior
        sl = sl_base_mult * sl_mult(z) * tp
        
    Where:
        base = median(atr_pct, window)
        disp = max(MAD(atr_pct, window), disp_floor * base)
        z = (atr_pct - base) / disp  (robust z-score)
        m(z) = exp(k_reg * clip(z, -z_max, z_max)) then clipped to [m_lo, m_hi]
        sl_mult(z) = sl_lo for z < z_gate, interpolates to sl_hi for z >= z_gate
    
    Note: z_gate is used ONLY for SL adaptation, not for event selection.
    Event gating should be applied separately in the candidate filtering stage.
    
    TP bounds (tp_lo, tp_hi) match the old scaled_atr_pct behavior:
    - Old: barrier = scaled_atr_pct(atr, z, base, lo=0.02, hi=0.06) 
    - New: barrier = clamp(k_tp * atr_pct * m(z) * sqrt(H), 0.02, 0.06)
    
    Returns:
        (tp_df, sl_df) or (tp_df, sl_df, diagnostics) if return_components=True
    """
    import numpy as np
    
    # 1. Compute base and dispersion (P1's robust approach)
    atr_median = atr_pct.rolling(window_size, min_periods=24).median()
    
    # MAD with floor
    def _rolling_mad(x, window):
        roll_med = x.rolling(window, min_periods=24).median()
        return (x - roll_med).abs().rolling(window, min_periods=24).median()
    
    atr_mad = _rolling_mad(atr_pct, window_size)
    atr_disp = np.maximum(atr_mad, disp_floor * atr_median)
    
    # 2. Robust z-score (P1's approach in P2's log-ratio friendly form)
    z_score = (atr_pct - atr_median) / (atr_disp + 1e-12)
    z_clipped = np.clip(z_score, -z_max, z_max)
    
    # 3. Regime multiplier m(z) - smooth exponential with bounds (P1's mapping)
    m_raw = np.exp(k_reg * z_clipped)
    m_clipped = np.clip(m_raw, m_lo, m_hi)
    
    # 4. Horizon scaling (P2's sqrt scaling)
    h_scale = np.sqrt(horizon / H_base)
    
    # 5. Raw TP = k_tp * ATR% * m(z) * sqrt(H/H_base)
    tp_raw = k_tp * atr_pct * m_clipped * h_scale
    
    # 6. Apply bounds to match old scaled_atr_pct behavior (Issue: barriers too large without bounds!)
    tp_vals = np.clip(tp_raw, tp_lo, tp_hi)
    
    # Debug: print bounds being used
    import sys
    print(f"DEBUG barrier_factory: tp_lo={tp_lo}, tp_hi={tp_hi}, tp_raw_mean={np.nanmean(tp_raw):.4f}, tp_clipped_mean={np.nanmean(tp_vals):.4f}", file=sys.stderr)
    
    # 7. SL ratio adapts to regime (P1's adaptive SL)
    # One-sided: only adapt when z > z_gate (high vol regime)
    # Below z_gate: uses sl_lo (quiet market - not traded if gating at z_gate for events)
    # Above z_gate: interpolates to sl_hi (volatile market - the regime we trade)
    z_norm = np.clip((z_clipped - z_gate) / (z_max - z_gate), 0, 1)  # 0 to 1 for z in [z_gate, z_max]
    sl_mult = sl_lo + (sl_hi - sl_lo) * z_norm
    
    # SL = sl_base_mult * sl_mult * TP
    sl_vals = sl_base_mult * sl_mult * tp_vals
    
    tp_df = pd.DataFrame(tp_vals, index=atr_pct.index, columns=atr_pct.columns)
    sl_df = pd.DataFrame(sl_vals, index=atr_pct.index, columns=atr_pct.columns)
    
    if return_components:
        # Per-asset diagnostics for cross-asset portability check
        asset_diagnostics = {}
        for col in atr_pct.columns:
            col_idx = atr_pct.columns.get_loc(col)
            m_vals = m_clipped.values[:, col_idx]
            sl_m = sl_mult.values[:, col_idx]
            atr_col = atr_pct[col].values
            tp_col = tp_vals.values[:, col_idx]
            
            # TP in ATR units: avoid division by zero
            tp_atr_ratio = np.divide(tp_col, atr_col, out=np.full_like(tp_col, np.nan), where=atr_col != 0)
            
            asset_diagnostics[col] = {
                "m_at_m_lo_pct": float(np.mean(m_vals == m_lo)),
                "m_at_m_hi_pct": float(np.mean(m_vals == m_hi)),
                "sl_at_sl_lo_pct": float(np.mean(sl_m == sl_lo)),
                "sl_at_sl_hi_pct": float(np.mean(sl_m == sl_hi)),
                "tp_atr_units": float(np.nanmean(tp_atr_ratio)),
            }
        
        diagnostics = {
            "z_mean": float(np.nanmean(z_clipped.values)),
            "z_p10": float(np.nanpercentile(z_clipped.values, 10)),
            "z_p90": float(np.nanpercentile(z_clipped.values, 90)),
            "z_below_gate_pct": float(np.mean(z_clipped.values < z_gate)),
            "z_above_gate_pct": float(np.mean(z_clipped.values >= z_gate)),
            "m_mean": float(np.nanmean(m_clipped.values)),
            "m_p10": float(np.nanpercentile(m_clipped.values, 10)),
            "m_p90": float(np.nanpercentile(m_clipped.values, 90)),
            "m_at_m_lo_pct": float(np.mean(m_clipped.values == m_lo)),
            "m_at_m_hi_pct": float(np.mean(m_clipped.values == m_hi)),
            "sl_mult_mean": float(np.nanmean(sl_mult)),
            "sl_at_sl_lo_pct": float(np.mean(sl_mult == sl_lo)),
            "sl_at_sl_hi_pct": float(np.mean(sl_mult == sl_hi)),
            "tp_mean": float(np.nanmean(tp_vals)),
            "sl_mean": float(np.nanmean(sl_vals)),
            "clip_low_pct": float(np.mean(m_clipped.values == m_lo)),
            "clip_high_pct": float(np.mean(m_clipped.values == m_hi)),
            "asset_diagnostics": asset_diagnostics,
        }
        return tp_df, sl_df, diagnostics
    
    return tp_df, sl_df



def _emoji(pass_flag):
    return "✅" if bool(pass_flag) else "⚠️"


TOPK_GATE_FRAC = 0.20
TOPK_INFO_FRACS = (0.10, 0.30)


def _get_training_candidate_config(cfg):
    return (
        cfg.get("train_extreme_pct_hourly", cfg.get("trade_extreme_pct", 0.05)),
        cfg.get("train_min_range_pct", 0.07),
        cfg.get("train_min_vol_zscore", 1.6),
    )


def _resolve_training_cfg_with_offline_optimisers(cfg):
    """Apply persisted offline-optimiser best params onto cfg with cfg values as fallback."""
    try:
        return apply_offline_optimizer_best_params(cfg)
    except Exception as exc:
        tprint(f"Warning: failed to load offline optimiser params; using cfg defaults ({exc})")
        return cfg


def _mad(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0
    med = np.median(x)
    return float(np.median(np.abs(x - med)) + 1e-12)


def _safe_spearman(x, y):
    try:
        v = spearmanr(x, y).correlation
        return float(v) if np.isfinite(v) else 0.0
    except Exception:
        return 0.0


def _avg_trades_per_day_global(scores, k_frac, timestamps):
    """Average selected trades/day using GLOBAL top-k and full day span.

    This avoids per-timestamp `max(1, ceil(k*group_size))` behavior that can
    collapse @10 and @30 to similar counts when groups are sparse.
    """
    s = np.asarray(scores, dtype=float)
    if s.size == 0:
        return 0.0
    k = max(1, int(np.ceil(float(k_frac) * s.size)))
    if timestamps is None:
        return float(k)
    ts = np.asarray(timestamps)
    if ts.size == 0:
        return float(k)
    days_all = np.array([np.datetime64(t, 'D') for t in ts])
    n_days = np.unique(days_all).size
    return float(k / max(n_days, 1))


def _evaluate_target(score, y, cost=0.005):
    """Composite scorer for meta target selection."""
    df_ev = pd.DataFrame({"score": np.asarray(score, dtype=float),
                          "y": np.asarray(y, dtype=float)}).dropna()
    if len(df_ev) < 30:
        return -999.0
    df_ev["rank"] = df_ev["score"].rank(pct=True)
    df_ev["y_rank"] = df_ev["y"].rank(pct=True)
    top_mask = df_ev["rank"] >= 0.7
    top = df_ev[top_mask]
    mean_net_top30 = top["y"].mean() - cost if len(top) > 0 else -cost
    true_top30 = df_ev["y_rank"] >= 0.7
    precision = (true_top30 & top_mask).sum() / max(top_mask.sum(), 1)
    lift30 = precision / 0.30
    if len(top) > 1:
        ic_t30 = _safe_spearman(top["score"].values, top["y"].values)
    else:
        ic_t30 = 0.0
    df_ev["decile"] = (df_ev["rank"] * 10).astype(int).clip(upper=9)
    q_means = df_ev.groupby("decile")["y"].mean()
    if len(q_means) >= 10:
        spread_q10_q5 = float(q_means.iloc[9] - q_means.iloc[4])
        spread_q10_q1 = float(q_means.iloc[9] - q_means.iloc[0])
    else:
        spread_q10_q5, spread_q10_q1 = 0.0, 0.0
    top_returns = top["y"] - cost if len(top) > 0 else pd.Series([0.0])
    stability = float(top_returns.mean() / top_returns.std()) if top_returns.std() > 0 else 0.0
    composite = (2.0 * mean_net_top30 + 1.5 * (lift30 - 1.0) + 1.5 * ic_t30
                 + 1.0 * spread_q10_q5 + 1.0 * spread_q10_q1 + 2.0 * stability)
    return float(composite)


def _build_target_variants(y_ret_raw, vol_proxy=None):
    """Build 4 base target variants x 2 normalizations (raw + semivol) = 8 targets."""
    from scipy.stats import rankdata
    from scipy.special import expit as _sigmoid

    n = len(y_ret_raw)
    y = np.asarray(y_ret_raw, dtype=np.float64)
    fin = np.isfinite(y)

    # Base 1: Global rank percentile
    rk = np.full(n, 0.5, dtype=np.float64)
    if fin.sum() > 1:
        rk[fin] = (rankdata(y[fin]) - 1) / max(fin.sum() - 1, 1)

    # Base 2: Soft Top-30% Score (smooth gate)
    _temp = 0.08
    t_soft30 = _sigmoid((rk - 0.70) / _temp)

    # Base 3: Quantile-Binned Midpoint (denoised rank)
    n_qbins = 20
    t_qbin = np.full(n, 0.5, dtype=np.float64)
    if fin.sum() > n_qbins:
        edges = np.percentile(y[fin], np.linspace(0, 100, n_qbins + 1))
        edges[0] -= 1e-12; edges[-1] += 1e-12
        bins = np.clip(np.digitize(y, edges) - 1, 0, n_qbins - 1)
        t_qbin = (bins + 0.5) / n_qbins

    # Base 4: Tail-Amplified Percentile (top-30% emphasis)
    t_tail = rk + 0.5 * np.maximum(0.0, rk - 0.70)

    bases = {"rank_pct": rk, "soft_top30": t_soft30, "qbin_mid": t_qbin, "tail_amp": t_tail}

    # Vol proxy for semi-vol normalization
    if vol_proxy is not None and np.isfinite(vol_proxy).sum() > 10:
        vp = np.clip(np.asarray(vol_proxy, dtype=np.float64), 1e-9, None)
        vp_med = float(np.nanmedian(vp[np.isfinite(vp)]))
        has_vol = True
    else:
        vp = np.ones(n, dtype=np.float64)
        vp_med = 1.0
        has_vol = False

    targets = {}
    for bname, bt in bases.items():
        # 1. Raw
        targets[bname] = bt.astype(np.float32)

        # 2. Semi-vol normalization (power scaling 0.5)
        semi_scale = np.power(vp_med / np.clip(vp, 1e-9, None), 0.5) if has_vol else np.ones(n)
        semi_scale = np.clip(semi_scale, 0.5, 2.0)
        targets[f"{bname}_semivol"] = ((bt - 0.5) * semi_scale + 0.5).astype(np.float32)

    return targets


def _detailed_oof_metrics(oof, y_ret, cost=0.005, n_rolling=5):
    """Compute global + top-30% metrics from OOF predictions vs raw returns.
    Includes: net top-30% after cost, turnover proxy, rolling IC stability.
    Returns dict with keys for the summary table."""
    from scipy.stats import spearmanr
    s = np.asarray(oof, dtype=float)
    y = np.asarray(y_ret, dtype=float)
    m = np.isfinite(s) & np.isfinite(y)
    s, y = s[m], y[m]
    n = len(s)
    if n < 10:
        return {}
    # Global metrics
    ic_g = float(spearmanr(s, y).statistic) if n > 2 else 0.0
    rk = (pd.Series(s).rank(pct=True)).values
    # Decile spread
    dec = (rk * 10).astype(int).clip(0, 9)
    q_means = pd.Series(y).groupby(dec).mean()
    spread_10_1 = float(q_means.get(9, 0) - q_means.get(0, 0)) if len(q_means) >= 2 else 0.0
    # Top-30% metrics
    t30 = rk >= 0.70
    n_t30 = int(t30.sum())
    if n_t30 < 3:
        return {"IC_global": ic_g, "Spread_10v1": spread_10_1,
                "n": n, "n_top30": 0}
    y_t30 = y[t30]
    s_t30 = s[t30]
    ic_t30 = float(spearmanr(s_t30, y_t30).statistic) if n_t30 > 2 else 0.0
    mean_ret_t30 = float(np.mean(y_t30))
    mean_net_t30 = mean_ret_t30 - cost
    std_t30 = max(float(np.std(y_t30)), 1e-9)
    sharpe_t30 = mean_net_t30 / std_t30
    # Sortino (downside deviation only)
    downside = y_t30[y_t30 < 0]
    dd = float(np.sqrt(np.mean(downside**2))) if len(downside) > 0 else 1e-9
    sortino_t30 = mean_net_t30 / max(dd, 1e-9)
    # Lift@30: precision of model's top-30% vs true top-30%
    y_rk = (pd.Series(y).rank(pct=True)).values
    true_t30 = y_rk >= 0.70
    precision = float((true_t30 & t30).sum()) / max(float(t30.sum()), 1)
    lift30 = precision / 0.30
    # Bottom-30% for spread
    b30 = rk <= 0.30
    mean_ret_b30 = float(np.mean(y[b30])) if b30.sum() > 0 else 0.0
    spread_t30_b30 = mean_ret_t30 - mean_ret_b30

    # Turnover proxy: fraction of top-30% that changes between consecutive subperiods
    chunk = max(n // n_rolling, 30)
    turnover = 0.0
    if n >= 2 * chunk:
        prev_set = None
        turn_vals = []
        for i in range(0, n - chunk + 1, chunk):
            _s_chunk = s[i:i+chunk]
            _rk_chunk = (pd.Series(_s_chunk).rank(pct=True)).values
            _top_idx = set(np.where(_rk_chunk >= 0.70)[0] + i)
            if prev_set is not None and len(prev_set) > 0:
                overlap = len(_top_idx & prev_set)
                union = len(_top_idx | prev_set)
                turn_vals.append(1.0 - overlap / max(union, 1))
            prev_set = _top_idx
        turnover = float(np.mean(turn_vals)) if turn_vals else 0.0

    # Rolling IC stability: std of IC across subperiods / mean IC
    rolling_ics = []
    if n >= 2 * chunk:
        for i in range(0, n - chunk + 1, chunk):
            _s_c = s[i:i+chunk]
            _y_c = y[i:i+chunk]
            if len(_s_c) > 5:
                _ic_c = float(spearmanr(_s_c, _y_c).statistic)
                rolling_ics.append(_ic_c)
    if len(rolling_ics) >= 2:
        _ic_mean = float(np.mean(rolling_ics))
        _ic_std = float(np.std(rolling_ics))
        stability = _ic_mean / max(_ic_std, 1e-9) if _ic_std > 1e-9 else 10.0
    else:
        stability = 0.0

    return {
        "IC_global": ic_g, "Spread_10v1": spread_10_1,
        "IC_top30": ic_t30, "Mean_ret_t30": mean_ret_t30,
        "Mean_net_t30": mean_net_t30, "Sharpe_t30": sharpe_t30,
        "Sortino_t30": sortino_t30,
        "Lift@30": lift30, "Spread_t30vb30": spread_t30_b30,
        "Turnover": turnover, "Stability": stability,
        "n": n, "n_top30": n_t30,
    }


def _run_target_race(X_meta_np, y_ret_raw, vol_proxy, w_meta, side_k_label):
    """Target race: ExtraTrees on ALL target variants directly (no Ridge pre-screen).
    Returns (best_target_name, best_target_array, race_log).
    """
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.preprocessing import RobustScaler

    targets = _build_target_variants(y_ret_raw, vol_proxy=vol_proxy)
    Xv = np.asarray(X_meta_np, dtype=np.float32)
    n = len(y_ret_raw)
    log_lines = []
    log_lines.append(f"  Target race ({side_k_label}): {len(targets)} variants, n={n}")

    pkf = PurgedKFold(n_splits=3, purge=5, embargo=2)
    _time_idx = np.arange(n, dtype=np.float64)

    _scaler = RobustScaler().fit(Xv)
    Xv_scaled = _scaler.transform(Xv).astype(np.float32)

    def _cv_oof(model_fn, tgt, sw):
        """Run 3-fold purged CV, return OOF predictions."""
        oof = np.full(n, np.nan, dtype=np.float64)
        tgt_f = tgt.astype(np.float64)
        fin = np.isfinite(tgt_f)
        if fin.sum() < 50:
            return None
        fill = float(np.nanmedian(tgt_f[fin]))
        tgt_f = np.where(fin, tgt_f, fill)
        for tr, va in pkf.split(_time_idx):
            m = model_fn()
            sw_tr = sw[tr] if sw is not None else None
            m.fit(Xv_scaled[tr], tgt_f[tr], sample_weight=sw_tr)
            oof[va] = m.predict(Xv_scaled[va])
        return oof

    # ExtraTrees on ALL target variants directly
    et_scores = {}
    et_metrics = {}
    for tname, tgt in targets.items():
        def _make_et():
            return ExtraTreesRegressor(
                n_estimators=200, max_depth=6, min_samples_leaf=30,
                max_features="sqrt", n_jobs=3, random_state=42)
        oof = _cv_oof(_make_et, tgt, w_meta)
        if oof is None:
            continue
        mask = np.isfinite(oof)
        if mask.sum() < 30:
            continue
        comp = _evaluate_target(oof[mask], y_ret_raw[mask])
        dm = _detailed_oof_metrics(oof[mask], y_ret_raw[mask])
        et_scores[tname] = comp
        et_metrics[tname] = dm
        log_lines.append(f"    ET  {tname:25s} composite={comp:.4f}")

    if not et_scores:
        log_lines.append("  Target race: all ET evaluations failed, using rank_pct")
        return "rank_pct", targets["rank_pct"], log_lines

    sorted_et = sorted(et_scores.items(), key=lambda x: x[1], reverse=True)
    best_name = sorted_et[0][0]
    log_lines.append(f"  Target race winner: {best_name}")

    # ── Summary table ──
    _hdr = (f"  {'Target':25s} {'Comp':>7s} {'IC_g':>7s} "
            f"{'IC_t30':>7s} {'Lift@30':>7s} {'Ret_t30':>9s} "
            f"{'Net_t30':>9s} {'Shrp_t30':>8s} {'Spr10v1':>9s} {'Spr_tb':>9s}")
    log_lines.append(f"  {'─'*110}")
    log_lines.append(_hdr)
    log_lines.append(f"  {'─'*110}")
    for tname, comp in sorted_et:
        dm = et_metrics.get(tname, {})
        _win = " ★" if tname == best_name else ""
        log_lines.append(
            f"  {tname:25s} {comp:>7.4f} "
            f"{dm.get('IC_global',0):>7.4f} {dm.get('IC_top30',0):>7.4f} "
            f"{dm.get('Lift@30',0):>7.3f} {dm.get('Mean_ret_t30',0):>9.6f} "
            f"{dm.get('Mean_net_t30',0):>9.6f} {dm.get('Sharpe_t30',0):>8.3f} "
            f"{dm.get('Spread_10v1',0):>9.6f} {dm.get('Spread_t30vb30',0):>9.6f}{_win}")
    log_lines.append(f"  {'─'*110}")

    return best_name, targets[best_name], log_lines


def _build_bin_mono_metrics(y_true, score, n_bins=10):
    y_true = np.asarray(y_true, dtype=float)
    score = np.asarray(score, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(score)
    y_true, score = y_true[mask], score[mask]
    if y_true.size < 20:
        return {"rho_bin_med": 0.0, "top_gt_mid_gt_bot": False, "top20_bot50_spread": 0.0}
    q = np.nanquantile(score, np.linspace(0, 1, n_bins + 1))
    q[0] -= 1e-12
    q[-1] += 1e-12
    bins = np.clip(np.digitize(score, q[1:-1]), 0, n_bins - 1)
    medians = []
    idxs = []
    for b in range(n_bins):
        m = bins == b
        if m.any():
            idxs.append(b)
            medians.append(float(np.median(y_true[m])))
    rho = _safe_spearman(idxs, medians) if len(medians) >= 3 else 0.0
    top = float(np.median(y_true[score >= np.nanquantile(score, 0.90)]))
    mid = float(np.median(y_true[(score >= np.nanquantile(score, 0.45)) & (score <= np.nanquantile(score, 0.55))]))
    bot = float(np.median(y_true[score <= np.nanquantile(score, 0.10)]))
    spread = float(np.median(y_true[score >= np.nanquantile(score, 0.80)]) - np.median(y_true[score <= np.nanquantile(score, 0.50)]))
    return {
        "rho_bin_med": rho,
        "top_gt_mid_gt_bot": bool((top > mid) and (mid > bot)),
        "top20_bot50_spread": spread,
    }


def _fold_stats_from_groups(y, pred, groups, fn):
    if groups is None:
        return []
    g = np.asarray(groups)
    out = []
    for gg in np.unique(g):
        m = g == gg
        if np.sum(m) < 10:
            continue
        out.append(fn(y[m], pred[m]))
    return out


def _base_model_report_entry(model_name, side, kind, dm, y_bin, oof_probs, y_ret, groups, y_lbl=None):
    prev = float(np.mean(y_bin))
    prev = float(np.clip(prev, 1e-7, 1 - 1e-7))
    base_brier = prev * (1.0 - prev)
    base_ll = -(prev*np.log(prev) + (1-prev)*np.log(1-prev))
    # Recompute Brier/LL from OOF predictions (consistent with y_bin)
    from sklearn.metrics import brier_score_loss, log_loss as _log_loss
    p_clip = np.clip(oof_probs, 1e-7, 1 - 1e-7)
    brier = float(brier_score_loss(y_bin, p_clip))
    ll = float(_log_loss(y_bin, p_clip))
    brier_imp = (base_brier - brier) / max(base_brier, 1e-9)
    ll_imp = (base_ll - ll) / max(base_ll, 1e-9)

    k_frac = TOPK_GATE_FRAC
    k_n = max(1, int(len(y_bin) * k_frac))
    idx = np.argsort(oof_probs)[-k_n:]
    prec_k = float(np.mean(y_bin[idx]))
    lift_k = prec_k / max(prev, 1e-9)
    prec_lift_abs = prec_k - prev

    # Timeout Rate Analysis (if raw labels available)
    timeout_metrics = {}
    if y_lbl is not None:
        # 0 = timeout, 1 = profit, -1 = stop loss
        # Check timeout rate in top 20%
        to_k = float(np.mean(y_lbl[idx] == 0))
        timeout_metrics["timeout_rate_at_20pct"] = to_k
        # Check timeout rate in bottom 50%
        n_bot = max(1, int(len(y_bin) * 0.50))
        idx_bot = np.argsort(oof_probs)[:n_bot]
        to_bot = float(np.mean(y_lbl[idx_bot] == 0))
        timeout_metrics["timeout_rate_at_bot50pct"] = to_bot

    # Informational (non-gating) precision/lift at 10% and 30%
    info_metrics = {}
    for _frac in TOPK_INFO_FRACS:
        _n = max(1, int(len(y_bin) * _frac))
        _idx = np.argsort(oof_probs)[-_n:]
        _prec = float(np.mean(y_bin[_idx]))
        info_metrics[f"prec_at_{int(_frac*100)}pct"] = _prec
        info_metrics[f"lift_at_{int(_frac*100)}pct"] = _prec / max(prev, 1e-9)

    fold_imp = [x for x in dm.get("fold_logloss_imp", []) if np.isfinite(x)]
    pos_fold_ratio = float(np.mean(np.array(fold_imp) > 0.0)) if fold_imp else 0.0
    worst_fold_imp = float(np.min(fold_imp)) if fold_imp else -1.0

    # Bootstrap CV(Prec@20) from OOF predictions
    # This computes the coefficient of variation (std/mean) of precision@20% across bootstrap samples.
    # Note: This is NOT cross-validation - it measures bootstrap stability of the precision estimate.
    n_boot = 50
    rng = np.random.RandomState(42)
    prec_samples = []
    n_total = len(y_bin)
    for _ in range(n_boot):
        idx_b = rng.choice(n_total, size=n_total, replace=True)
        _n_k = max(1, int(n_total * k_frac))
        top_idx = np.argsort(oof_probs[idx_b])[-_n_k:]
        p_k_b = float(np.mean(y_bin[idx_b][top_idx]))
        prec_samples.append(p_k_b)
    prec_arr = np.array(prec_samples)
    bootstrap_prec20_cv = float(np.std(prec_arr) / (np.mean(prec_arr) + 1e-9))

    from sklearn.metrics import average_precision_score
    # Ensure binary labels and safe probs
    y_bin_calc = (np.asarray(y_bin) >= 0.5).astype(int)
    oof_probs_safe = np.nan_to_num(oof_probs, nan=0.0)
    try:
        pr_auc = float(average_precision_score(y_bin_calc, oof_probs_safe)) if len(np.unique(y_bin_calc)) > 1 else 0.0
    except Exception:
        pr_auc = 0.0

    # Prevalence-aware PR-AUC threshold (matching gate_metrics.py logic)
    # Threshold = max(1.25 * prev, prev + 0.05)
    # We remove the 0.50 floor because for low-prevalence (e.g. 0.35), 0.45 is a good score.
    prev_for_threshold = float(np.mean(y_bin_calc))
    pr_auc_threshold = max(1.25 * prev_for_threshold, prev_for_threshold + 0.05)

    # Diagnostic: PR-AUC below prevalence indicates model is worse than random
    if pr_auc < prev_for_threshold:
        tprint(f"WARNING: PR-AUC ({pr_auc:.4f}) < Prevalence ({prev_for_threshold:.4f}) for {model_name} - model is worse than random!")
        tprint(f"  Lift@20%: {lift_k:.4f}")
        # Check if labels might be inverted
        if lift_k < 1.0 and prec_lift_abs < 0:
            tprint(f"  CRITICAL: Lift < 1.0 and precision lift negative - possible label inversion!")

    checks = {
        "pr_auc_ge_threshold": pr_auc >= pr_auc_threshold,
        "pr_auc_ge_random": pr_auc >= prev_for_threshold,
        "brier_and_logloss_improve_ge_2pct": bool((brier_imp >= 0.02) and (ll_imp >= 0.02)),
        "liftk_and_preck_lift": bool((lift_k >= 1.2) and ((prec_lift_abs >= 0.025) or ((lift_k - 1.0) >= 0.05))),
        "bootstrap_prec20_cv_le_0_30": bootstrap_prec20_cv <= 0.30,
        "delta_logloss_le_minus_0_5pct": ll_imp >= 0.005,
        "logloss_improves_in_ge_70pct_folds": pos_fold_ratio >= 0.70,
        "worst_fold_delta_logloss_ge_0_5pct_improve": worst_fold_imp >= -0.005,
    }

    # Raw AUC and IC for summary table
    from sklearn.metrics import roc_auc_score as _roc_auc_score
    try:
        raw_auc = float(_roc_auc_score(y_bin, p_clip)) if len(np.unique(y_bin)) > 1 else 0.5
    except Exception:
        raw_auc = 0.5
    raw_ic = _safe_spearman(oof_probs, y_ret)

    metrics = {
        "auc": raw_auc,
        "ic": raw_ic,
        "logloss": float(ll),
        "pr_auc": pr_auc,
        "pr_auc_threshold": pr_auc_threshold,
        "brier_improvement": float(brier_imp),
        "logloss_improvement": float(ll_imp),
        "lift_at_20pct": float(lift_k),
        "precision_lift_abs": float(prec_lift_abs),
        "bootstrap_prec20_cv": bootstrap_prec20_cv,
        "fold_logloss_improvement_ratio": pos_fold_ratio,
        "worst_fold_logloss_improvement": worst_fold_imp,
        **info_metrics,
        **timeout_metrics,
    }

    return {
        "model": model_name,
        "side": side,
        "kind": kind,
        "score": float(dm.get("rank_score", dm.get("score", 0.0))),
        "checks": {k: {"pass": bool(v), "emoji": _emoji(v)} for k, v in checks.items()},
        "metrics": metrics,
        "passed": bool(all(checks.values())),
    }


def _meta_report_entry(name, meta_model, y_target, y_ret, base_score, groups,
                       y_per_horizon=None):
    pred = int(getattr(meta_model, "score_sign", 1)) * np.asarray(meta_model.oof_probs, dtype=float)
    y_target = np.asarray(y_target, dtype=float)
    y_ret = np.asarray(y_ret, dtype=float)
    if pred.size != y_target.size:
        n = min(pred.size, y_target.size)
        pred, y_target, y_ret = pred[:n], y_target[:n], y_ret[:n]
        if groups is not None:
            groups = np.asarray(groups)[:n]
        if y_per_horizon is not None:
            y_per_horizon = {h: v[:n] for h, v in y_per_horizon.items()}

    tau = 0.85
    is_quantile = bool(meta_model.model and meta_model.model.get("pool") == "quantile")

    def pinball(y, q, a=tau):
        e = y - q
        return float(np.mean(np.maximum(a*e, (a-1.0)*e)))

    cov = float(np.mean(y_target <= pred))
    pb = pinball(y_target, pred)
    base_q = float(np.quantile(y_target, tau))
    pb_base = pinball(y_target, np.full_like(y_target, base_q))
    pb_imp = (pb_base - pb) / max(pb_base, 1e-9)

    fold_pb_imp = []
    fold_sign = []
    if groups is not None:
        g = np.asarray(groups)
        for gg in np.unique(g):
            m = g == gg
            if np.sum(m) < 20:
                continue
            pbf = pinball(y_target[m], pred[m])
            bqf = float(np.quantile(y_target[m], tau))
            pbbf = pinball(y_target[m], np.full(np.sum(m), bqf))
            impf = (pbbf - pbf) / max(pbbf, 1e-9)
            fold_pb_imp.append(float(impf))
            fold_sign.append(float(impf) >= 0.02)

    ic = _safe_spearman(pred, y_target)
    fold_ics = _fold_stats_from_groups(y_target, pred, groups, _safe_spearman) if groups is not None else []
    pos_ic_ratio = float(np.mean(np.array(fold_ics) > 0)) if fold_ics else 0.0
    stable_sign = True
    if fold_ics:
        signs = np.sign(fold_ics)
        stable_sign = bool(np.mean(signs == np.sign(np.mean(fold_ics))) >= 0.7)

    mono = _build_bin_mono_metrics(y_ret, pred, n_bins=10)
    mad_y = _mad(y_ret)
    spread_ok = (mono["top20_bot50_spread"] >= 0.25 * mad_y) or (mono["top20_bot50_spread"] > 0)

    k20 = max(1, int(TOPK_GATE_FRAC * len(pred)))
    idx_meta = np.argsort(pred)[-k20:]
    idx_meta_flip = np.argsort(-pred)[-k20:]
    idx_base = np.argsort(base_score)[-k20:]
    def es_tail(v):
        s = np.asarray(v, dtype=float)
        if s.size == 0:
            return 0.0
        q = np.quantile(s, 0.15)
        tail = s[s <= q]
        return float(np.mean(tail)) if tail.size else float(q)
    es_meta = es_tail(y_ret[idx_meta])
    es_base = es_tail(y_ret[idx_base])
    es_ok = es_meta <= 1.1 * es_base if es_base > 0 else es_meta >= 1.1 * es_base

    def strat_metrics(sel_idx):
        r = y_ret[sel_idx]
        net = float(np.sum(r))
        dn = r[r < 0]
        sortino = float(np.mean(r) / (np.std(dn) + 1e-9)) if dn.size else 0.0
        return net, sortino
    net_meta, sort_meta = strat_metrics(idx_meta)
    net_meta_flip, sort_meta_flip = strat_metrics(idx_meta_flip)
    net_base, sort_base = strat_metrics(idx_base)

    # Informational (non-gating): Top10 and Top30 policy slices
    info_policy = {}
    for _frac in TOPK_INFO_FRACS:
        _k = max(1, int(_frac * len(pred)))
        _idx_m = np.argsort(pred)[-_k:]
        _idx_b = np.argsort(base_score)[-_k:]
        _nm, _sm = strat_metrics(_idx_m)
        _nb, _sb = strat_metrics(_idx_b)
        tag = f"{int(_frac*100)}"
        info_policy[f"net_return_meta_top{tag}"] = _nm
        info_policy[f"net_return_base_top{tag}"] = _nb
        info_policy[f"sortino_meta_top{tag}"] = _sm
        info_policy[f"sortino_base_top{tag}"] = _sb

    checks = {}
    if is_quantile:
        checks.update({
            "coverage_tau": abs(cov - tau) <= 0.05,
            "pinball_improve_ge_2pct": pb_imp >= 0.02,
            "pinball_improve_ge_2of3_folds": (np.mean(fold_sign) >= (2/3)) if fold_sign else False,
        })
    else:
        # proxy robust-loss/bias checks based on oof residuals
        res = pred - y_target
        loss = float(np.mean(np.abs(res)))
        loss_base = float(np.mean(np.abs(np.full_like(y_target, np.median(y_target)) - y_target)))
        loss_imp = (loss_base - loss) / max(loss_base, 1e-9)
        fold_loss = _fold_stats_from_groups(y_target, pred, groups, lambda yt, pr: (np.mean(np.abs(np.median(yt)-yt)) - np.mean(np.abs(pr-yt))) / max(np.mean(np.abs(np.median(yt)-yt)),1e-9)) if groups is not None else []
        mean_err = float(np.mean(res))
        bias_fold = _fold_stats_from_groups(y_target, pred, groups, lambda yt, pr: np.mean(pr-yt)) if groups is not None else []
        mad_t = _mad(y_target)
        checks.update({
            "robust_loss_ge_2pct": loss_imp >= 0.02,
            "robust_loss_ge_2of3_folds": (np.mean(np.array(fold_loss) > 0) >= 2/3) if fold_loss else False,
            "robust_loss_worst_fold_ge_1pct": (np.min(fold_loss) >= 0.01) if fold_loss else False,
            "bias_overall": abs(mean_err) <= 0.05 * mad_t,
            "bias_per_fold": (np.max(np.abs(bias_fold)) <= 0.07 * mad_t) if bias_fold else False,
        })

    checks.update({
        "spearman_ic_ge_0_03": ic >= 0.03,
        "ic_stable_sign": stable_sign,
        "ic_pos_ge_70pct_folds": pos_ic_ratio >= 0.70,
        "bin_monotonicity_ge_0_9": mono["rho_bin_med"] >= 0.90,
        "top_mid_bottom_ordering": mono["top_gt_mid_gt_bot"],
        "top20_bottom50_spread": spread_ok,
        "es20_meta_vs_base": es_ok,
        "net_return_vs_no_meta": net_meta > net_base,
        "sortino_vs_no_meta": sort_meta > sort_base,
    })

    # Compute comprehensive OOF metrics for reporting table (against meta target)
    from .meta_model import MetaModel as _MM
    _oof_metrics = _MM._compute_oof_metrics(pred, y_target,
                                            y_per_horizon=y_per_horizon)

    return {
        "model": name,
        "model_type": meta_model.model.get("kind") if meta_model.model else None,
        "passed": bool(all(checks.values())),
        "checks": {k: {"pass": bool(v), "emoji": _emoji(v)} for k, v in checks.items()},
        "metrics": {
            "coverage": cov,
            "pinball_improvement": pb_imp,
            "spearman_ic": ic,
            "ic_positive_fold_ratio": pos_ic_ratio,
            "bin_spearman": mono["rho_bin_med"],
            "top20_bottom50_spread": mono["top20_bot50_spread"],
            "es20_meta": es_meta,
            "es20_base": es_base,
            "net_return_meta": net_meta,
            "net_return_meta_if_flipped": net_meta_flip,
            "net_return_base": net_base,
            "sortino_meta": sort_meta,
            "sortino_meta_if_flipped": sort_meta_flip,
            "sortino_base": sort_base,
            "direction_flip_improves": bool((net_meta_flip > net_meta) and (sort_meta_flip > sort_meta)),
            **info_policy,
            **_oof_metrics,
        },
    }


def save_training_gate_report(report_payload, cfg, run_id=None):
    reports_dir = os.path.join("extreme_price_movements", "reports")
    os.makedirs(reports_dir, exist_ok=True)
    rid = run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(reports_dir, f"training_gate_report_{rid}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=2)
    return out_path


def print_training_gate_report(report_payload):
    tprint("\n" + "=" * 100)
    tprint("MODEL QUALITY REPORT")
    tprint("=" * 100)

    # ── Base model table ─────────────────────────────────────────────────
    base_items = report_payload.get("base_models", [])
    # Only show race winners (non-winners have no OOF metrics)
    winners = [it for it in base_items if it.get("is_winner", False)]
    if not winners:
        winners = base_items  # fallback: show all if no winner flag
    if winners:
        tprint("\n--- BASE MODELS (Race Winners) ---")
        hdr = f"{'Model':<32} {'AUC':>7} {'IC':>7} {'LogLoss':>8} {'PR-AUC':>7} {'Lift@20':>8} {'BrierImp':>9}"
        tprint(hdr)
        tprint("-" * len(hdr))
        for it in winners:
            m = it.get("metrics", {})
            name = it.get("model", "?")
            auc = m.get("auc", float("nan"))
            ic = m.get("ic", float("nan"))
            ll = m.get("logloss", float("nan"))
            prauc = m.get("pr_auc", float("nan"))
            lift = m.get("lift_at_20pct", float("nan"))
            brimp = m.get("brier_improvement", float("nan"))
            tprint(f"{name:<32} {auc:>7.4f} {ic:>7.4f} {ll:>8.4f} {prauc:>7.4f} {lift:>8.3f} {brimp:>8.1%}")
    else:
        tprint("\n--- BASE MODELS: none ---")

    # ── Meta model table (regressors) ───────────────────────────────────
    meta_items = report_payload.get("meta_models", [])
    reg_items = [it for it in meta_items if not it.get("model", "").endswith("_clf")]
    clf_items = [it for it in meta_items if it.get("model", "").endswith("_clf")]

    if reg_items:
        tprint("\n--- META MODELS (Regressors) ---")
        hdr = f"{'Model':<20} {'IC':>7} {'IC_mh':>7} {'IC_t30':>7} {'ECE_t30':>8} {'Sprd10':>8} {'Sprd30':>8} {'Win_t30':>8} {'Loss_t30':>9} {'N':>7}"
        tprint(hdr)
        tprint("-" * len(hdr))
        for it in reg_items:
            m = it.get("metrics", it)
            name = it.get("model", "?")
            ic = m.get("ic", m.get("spearman_ic", float("nan")))
            icmh = m.get("ic_mh", float("nan"))
            ic30 = m.get("ic_top30", float("nan"))
            ece = m.get("ece_top30", float("nan"))
            s10 = m.get("spread10", float("nan"))
            s30 = m.get("spread30", float("nan"))
            wr30 = m.get("win_rate_top30", float("nan"))
            rl30 = m.get("robust_loss_top30", float("nan"))
            nt = m.get("n", 0)
            tprint(f"{name:<20} {ic:>7.4f} {icmh:>7.4f} {ic30:>7.4f} {ece:>8.4f} {s10:>8.6f} {s30:>8.6f} {wr30:>7.1%} {rl30:>8.1%} {nt:>7d}")
    else:
        tprint("\n--- META MODELS (Regressors): none ---")

    # ── Meta classifier table ─────────────────────────────────────────
    if clf_items:
        tprint("\n--- META MODELS (Classifiers) ---")
        hdr = f"{'Model':<20} {'Winner':<16} {'Thr':>5} {'PR-AUC':>7} {'Lift@26':>8} {'Sortino':>8} {'PnL_bps':>8} {'MaxDD':>8} {'WinRate':>8} {'Trd/Day':>8}"
        tprint(hdr)
        tprint("-" * len(hdr))
        for it in clf_items:
            m = it.get("metrics", {})
            name = it.get("model", "?")
            winner = m.get("clf_winner", "?")
            thr = m.get("clf_threshold_pct", 0)
            prauc = m.get("clf_pr_auc", float("nan"))
            lift26 = m.get("clf_lift_26", float("nan"))
            sortino = m.get("clf_sortino", float("nan"))
            pnl = m.get("clf_pnl_total_bps", float("nan"))
            maxdd = m.get("clf_max_dd_bps", float("nan"))
            wr = m.get("clf_win_rate", float("nan"))
            tpd = m.get("clf_avg_trades_day", float("nan"))
            tprint(f"{name:<20} {winner:<16} {thr:>4d}% {prauc:>7.4f} {lift26:>8.3f} {sortino:>8.3f} {pnl:>8.1f} {maxdd:>8.1f} {wr:>7.1%} {tpd:>8.1f}")
    else:
        tprint("\n--- META MODELS (Classifiers): none ---")

    tprint("=" * 100)


class AlphaHorizonEnsemble:
    """Average probabilities across multiple horizon-specific alpha models."""
    def __init__(self, members):
        # members: list of dict(model, feat_cols, H, weight, oof_probs)
        self.members = members
        self.oof_probs = None
        if members:
            oofs = [m.get("oof_probs") for m in members if m.get("oof_probs") is not None]
            if oofs:
                self.oof_probs = np.mean(np.vstack(oofs), axis=0).astype(np.float32)

    def predict_proba(self, X):
        preds = []
        for m in self.members:
            mdl = m["model"]
            cols = m["feat_cols"]
            if hasattr(X, "reindex"):
                Xi = X.reindex(columns=cols, fill_value=0.0)
            else:
                Xi = X
            p = mdl.predict_proba(Xi)[:, 1]
            preds.append(np.asarray(p, dtype=np.float64))
        if not preds:
            n = len(X) if hasattr(X, "__len__") else 0
            p1 = np.full(n, 0.5, dtype=np.float64)
        else:
            p1 = np.mean(np.vstack(preds), axis=0)
        p1 = np.clip(p1, 1e-6, 1 - 1e-6)
        return np.column_stack([1.0 - p1, p1])



def compute_meta_target(ret2: np.ndarray, ret4: np.ndarray, ret8: np.ndarray, groups=None) -> np.ndarray:
    """Build a per-trade meta target as winsorized log-return.

    Uses weighted average of per-horizon log-returns [0.40, 0.35, 0.25] for [H2, H4, H8],
    then winsorizes at [5th, 95th] percentile to reduce outlier noise while preserving
    per-trade magnitude information.

    This target is NOT cross-sectional rank (we want to trade every profitable asset,
    not just the best one). It preserves the key property: bigger positive return = better trade.
    The `groups` argument is kept for backward compatibility but intentionally ignored.
    """

    def _log1p_ret(x: np.ndarray) -> np.ndarray:
        v = np.asarray(x, dtype=float)
        v = np.clip(v, -0.999999, None)
        out = np.log1p(v)
        if not np.all(np.isfinite(out)):
            finite = np.isfinite(out)
            if finite.any():
                fill = float(np.nanmedian(out[finite]))
                out = np.where(finite, out, fill)
            else:
                out = np.zeros_like(out, dtype=float)
        return out.astype(np.float32)

    r2 = _log1p_ret(ret2)
    r4 = _log1p_ret(ret4)
    r8 = _log1p_ret(ret8)
    raw = (0.40 * r2 + 0.35 * r4 + 0.25 * r8).astype(np.float64)

    # Asymmetric winsorization: hard clip downside at p5, sqrt-compress upside above p90
    # Rationale: downside outliers are noise (stop-losses), but upside tails
    # contain real signal about trade quality that we want to preserve.
    # sqrt compression (not tanh) preserves monotonic ordering within the tail,
    # which is critical for IC_t30 (ranking quality among top predictions).
    finite = np.isfinite(raw)
    if finite.sum() > 10:
        lo = float(np.percentile(raw[finite], 5))
        hi = float(np.percentile(raw[finite], 90))
        # Hard clip downside
        raw = np.where(raw < lo, lo, raw)
        # Sqrt-compress upside: values above p90 are compressed but ordering preserved
        # sqrt(x) is monotonic and never saturates, unlike tanh
        above = raw > hi
        if above.any():
            scale = max(float(np.std(raw[finite])), 1e-9)
            excess = (raw[above] - hi) / scale  # normalized excess
            raw[above] = hi + scale * np.sqrt(excess)  # sqrt preserves ordering

    return raw.astype(np.float32)


def build_horizon_prediction_features(conf: dict, X_eval: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=X_eval.index)
    models_by_h = conf.get("models_by_h", {}) if conf else {}
    for H in [2, 4, 8]:
        if H in models_by_h:
            m = models_by_h[H]
            Xi = X_eval.reindex(columns=m.get("feat_cols", []), fill_value=0.0)
            out[f"pred_H{H}"] = m["model"].predict_proba(Xi)[:, 1]
        else:
            out[f"pred_H{H}"] = 0.5
    return out

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
        from sklearn.metrics import log_loss as _log_loss
        p_clipped = np.clip(p_m, 1e-6, 1 - 1e-6)
        metrics["logloss"] = float(_log_loss(y_m, p_clipped))

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

    atr = atr_pct_df.astype(np.float32)
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


def _strategy_bucket_context(trade_side: str, model_kind: str) -> tuple:
    """Return (candidate_bucket, move_bucket, strategy_label) for (trade_side, model_kind)."""
    if trade_side == "long":
        cand_filter = "worst" if model_kind == "mr" else "best"
    else:
        cand_filter = "best" if model_kind == "mr" else "worst"
    move_bucket = "up" if cand_filter == "best" else "down"
    if trade_side == "long" and model_kind == "mr":
        strategy_label = "buy_dips"
    elif trade_side == "long" and model_kind == "tf":
        strategy_label = "buy_momentum"
    elif trade_side == "short" and model_kind == "mr":
        strategy_label = "sell_rips"
    else:
        strategy_label = "sell_weakness"
    return cand_filter, move_bucket, strategy_label

def build_exhaustion_Xy(panel, feats, mkt_gates, cfg, ts_end, lookback_hours, syms, trend_filter=None, model_direction=None):
    """
    Build exhaustion training data.
    
    Parameters
    ----------
    model_direction : str, optional
        'up' for UP model (predicts long reversals during downtrends)
        'down' for DOWN model (predicts short reversals during uptrends)
        If None, uses legacy behavior (deprecated).
    
    Label Assignment Logic (FIXED):
    - UP model (model_direction='up'): Looks for LONG reversals during DOWNTRENDS
      - Uses trend_filter='down' to select downtrending samples
      - Assigns is_long_rev labels (long reversal happened)
    - DOWN model (model_direction='down'): Looks for SHORT reversals during UPTRENDS
      - Uses trend_filter='up' to select uptrending samples  
      - Assigns is_short_rev labels (short reversal happened)
    """
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
    ret24 = feats["ret24h"].reindex(index=t_index, columns=valid_syms)
    dir_mat = np.sign(ret24).fillna(0).astype(np.int8)
    y = np.zeros(current.shape, dtype=np.int8)
    w = np.ones(current.shape, dtype=np.float32)

    # FIXED: Label assignment based on model_direction
    # UP model (model_direction='up'): Predicts LONG reversals during DOWNTRENDS
    # DOWN model (model_direction='down'): Predicts SHORT reversals during UPTRENDS
    if model_direction == 'up':
        # UP model: Look for long reversals (is_long_rev) during downtrends (dir_mat < 0)
        mask_dn = (dir_mat < 0)
        if mask_dn.values.any():
            y[mask_dn] = is_long_rev.values[mask_dn].astype(np.int8)
            if cfg.get("exh_label_type") == "peak":
                w[mask_dn] = w_long_s.values[mask_dn].astype(np.float32)
        tprint(f"UP model: Using LONG reversal labels during DOWNTRENDS")
    elif model_direction == 'down':
        # DOWN model: Look for short reversals (is_short_rev) during uptrends (dir_mat > 0)
        mask_up = (dir_mat > 0)
        if mask_up.values.any():
            y[mask_up] = is_short_rev.values[mask_up].astype(np.int8)
            if cfg.get("exh_label_type") == "peak":
                w[mask_up] = w_short_s.values[mask_up].astype(np.float32)
        tprint(f"DOWN model: Using SHORT reversal labels during UPTRENDS")
    else:
        # Legacy behavior (deprecated - kept for backward compatibility)
        # This was the buggy behavior that caused the imbalance
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
            X_parts.append(feats[k].reindex(index=t_index, columns=valid_syms).stack(future_stack=True).rename(k))
    X = pd.concat(X_parts, axis=1)
    X.index.names = ["ts", "symbol"]
    if trend_filter:
        trend_vals = feats["trend_pct"].reindex(index=t_index, columns=valid_syms).stack(future_stack=True)
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
        common_idx = X.index.intersection(y_df.index).intersection(w_df.index)
        X = X.loc[common_idx].copy()
        X["y"] = y_df.reindex(common_idx).values
        X["w"] = w_df.reindex(common_idx).values
        X = X.dropna()
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
        _std = vals.std()
        if pd.notna(_std) and _std > 1e-9:
            X[col] = vals

    return X, y_arr, w_arr, list(X.columns)

def compute_p_exhaustion_at_t(panel, feats, mkt_gates, cfg, ts, syms, models=None):
    tprint(f"Entering function: compute_p_exhaustion_at_t in training.py")
    t_index = pd.DatetimeIndex([ts], tz="UTC")
    valid_syms = [s for s in syms if s in panel["close"].columns]
    trend_vals = feats["trend_pct"].reindex(columns=valid_syms).loc[ts] if ts in feats["trend_pct"].index else pd.Series(0.0, index=valid_syms)
    up_syms = trend_vals[trend_vals > 0].index.tolist()
    dn_syms = trend_vals[trend_vals <= 0].index.tolist()
    tprint(f"compute_p_exhaustion_at_t: {len(up_syms)} up, {len(dn_syms)} down")

    out_probs = pd.Series(index=syms, dtype=float).fillna(0.0)
    lookback = cfg["exh_train_lookback_hours"]
    if up_syms:
        if models and "up" in models: model_up = models["up"]
        else:
            tprint("Training UP model...")
            X, y, w, _ = build_exhaustion_Xy(panel, feats, mkt_gates, cfg, ts, lookback, valid_syms, model_direction="up")
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
            X, y, w, _ = build_exhaustion_Xy(panel, feats, mkt_gates, cfg, ts, lookback, valid_syms, model_direction="down")
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
            X_parts.append(feats[k].reindex(index=t_index, columns=syms).stack(future_stack=True).rename(k))
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
    X_up, y_up, w_up, _ = build_exhaustion_Xy(panel, feats, mkt_gates, cfg, train_end, train_len, syms, model_direction="up")
    model_up = None
    arr_oof_up = None
    if X_up is not None and len(y_up) > 100:
        model_up = ExhaustionModel()
        model_up.fit(X_up, y_up, sample_weight=w_up)
        # OOF Predictions for UP
        tprint("Generating OOF predictions for UP model...")
        oof_preds, _ = model_up.compute_oof_predictions(X_up, y_up)
        # Unstack to align with (ts, symbol) grid
        # We need this to match the prediction window structure later
        # We'll delay unstacking until we have valid_syms and t_idx defined below

    tprint("Generating DOWN history...")
    X_dn, y_dn, w_dn, _ = build_exhaustion_Xy(panel, feats, mkt_gates, cfg, train_end, train_len, syms, model_direction="down")
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

def _optimize_training_sample_weights(
    df: pd.DataFrame,
    X_frame: pd.DataFrame,
    y_ret: np.ndarray,
    label_times: pd.DataFrame,
    base_weights: np.ndarray,
    cfg: dict,
    stage: str,
    extra_components: dict | None = None,
) -> np.ndarray:
    """Optimize sample-weight component blend using constrained CV objective."""
    if not bool(cfg.get("sample_weight_opt_enable", True)):
        return np.asarray(base_weights, dtype=np.float32)

    n = len(base_weights)
    if n < int(cfg.get("sample_weight_opt_min_samples", 400)):
        return np.asarray(base_weights, dtype=np.float32)

    components: dict[str, np.ndarray] = {"base": np.asarray(base_weights, dtype=float)}
    ts_vals = pd.to_datetime(df["ts"]).values

    if "rv_24h" in df.columns:
        components["vol_cs"] = compute_vol_weights(
            df["rv_24h"].values, ts_vals,
            direction=cfg.get("sample_weight_vol_direction", "downweight_high"),
            power=float(cfg.get("sample_weight_vol_power", 0.5)),
            min_group_size=int(cfg.get("sample_weight_vol_min_group_size", 20)),
        )

    if "volume_usd_24h" in df.columns:
        components["liquidity"] = compute_liquidity_weights(df["volume_usd_24h"].values)
    elif "quote_volume_24h" in df.columns:
        components["liquidity"] = compute_liquidity_weights(df["quote_volume_24h"].values)

    era = pd.to_datetime(df["ts"]).dt.to_period("M").astype(str).values
    bar_idx = np.arange(n, dtype=int)
    components["recency"] = compute_recency_weights(
        bar_idx, era,
        half_life_bars=int(cfg.get("sample_weight_recency_half_life_bars", 24 * 30)),
        min_era_neff_ratio=float(cfg.get("sample_weight_recency_min_era_neff_ratio", 0.2)),
    )

    if extra_components:
        for k, v in extra_components.items():
            if v is None:
                continue
            arr = np.asarray(v, dtype=float)
            if len(arr) == n:
                components[k] = arr

    label_intervals = np.column_stack([
        pd.to_datetime(label_times["t_start"]).values.astype("datetime64[ns]"),
        pd.to_datetime(label_times["t_end"]).values.astype("datetime64[ns]"),
    ])

    X_frame = select_test_feature_frame(X_frame)
    X_np = np.asarray(X_frame, dtype=np.float32)

    fixed_component_alphas = cfg.get("sample_weight_component_alphas")
    stage_l = str(stage).lower()
    if "meta" in stage_l:
        fixed_component_alphas = cfg.get("sample_weight_component_alphas_meta", fixed_component_alphas)
    elif "base" in stage_l or "alpha" in stage_l:
        fixed_component_alphas = cfg.get("sample_weight_component_alphas_base", fixed_component_alphas)
    if isinstance(fixed_component_alphas, dict) and fixed_component_alphas:
        resolved_alphas = {
            str(name): float(fixed_component_alphas.get(name, 1.0))
            for name in components.keys()
        }
        optimized_weights = combine_weights_safely(
            components,
            resolved_alphas,
            min_n_eff_ratio=float(cfg.get("sample_weight_opt_min_n_eff_ratio", 0.30)),
        )
        tprint(f"[{stage}] using persisted sample-weight component alphas={resolved_alphas}")
        log_weight_statistics(optimized_weights, era, f"{stage}_persisted_alphas")
        return np.asarray(optimized_weights, dtype=np.float32)

    res = optimize_component_weights(
        X=X_np,
        y_ret=np.asarray(y_ret, dtype=float),
        label_intervals=label_intervals,
        components=components,
        production_model=cfg.get("sample_weight_opt_model_family", "ExtraTrees"),
        n_trials=int(cfg.get("sample_weight_opt_trials", 16)),
        n_splits=int(cfg.get("sample_weight_opt_n_splits", 5)),
        embargo_bars=int(cfg.get("sample_weight_opt_embargo_bars", 10)),
        min_n_eff_ratio=float(cfg.get("sample_weight_opt_min_n_eff_ratio", 0.30)),
        max_top1pct=float(cfg.get("sample_weight_opt_max_top1pct", 0.10)),
        random_state=int(cfg.get("seed", 42)),
    )

    tprint(f"[{stage}] sample-weight optimization objective={res.objective_value:.5f} alphas={res.component_alphas}")
    log_weight_statistics(res.optimized_weights, era, f"{stage}_optimized")
    return np.asarray(res.optimized_weights, dtype=np.float32)


def build_hourly_training_set_and_weights(
    panel, feats, mkt_gates, cfg, syms, ts_end, p_exh_hist, H, model_kind,
    trend_filter=None, feature_key=None, extra_feature_keys=None,
    label_method="atr", fixed_tp=0.05, fixed_sl=0.025, side="long",
    _cached_cand_mask=None, _cached_tb=None, _tb_cache=None
):
    tprint(f"Entering function: build_hourly_training_set_and_weights in training.py")
    c = panel["close"]
    idx = c.index

    train_pct, train_min_range, train_min_vol = _get_training_candidate_config(cfg)
    if _cached_cand_mask is not None:
        cand_mask = _cached_cand_mask
    else:
        cand_mask = select_trade_candidates_vectorized(
            panel,
            feats,
            pct=train_pct,
            metric=cfg["trade_deviation_metric"],
            min_range_pct=train_min_range,
            min_vol_zscore=train_min_vol,
        )
    if cand_mask is None:
        tprint("No candidates mask returned.")
        return None, None, None, None, None, None
    tprint(f"Candidates found: {cand_mask.sum().sum()}")

    ts_start = ts_end - pd.Timedelta(hours=int(cfg["train_lookback_hours"]))
    # Slice to time window first, then apply subsample filter
    ts_end_adj = ts_end - pd.Timedelta(hours=H+8)
    window_cand = cand_mask.loc[(cand_mask.index >= ts_start) & (cand_mask.index <= ts_end_adj)]
    if window_cand.empty:
        tprint(
            "No rows generated for training set: "
            f"valid_syms=0, window_cand_shape={window_cand.shape}, "
            f"ts_window=[{ts_start}, {ts_end_adj}]"
        )
        return None, None, None, None, None, None

    # Early symbol precheck before any barrier computation.
    valid_syms = [s for s in syms if s in window_cand.columns and s in c.columns]
    if not valid_syms:
        tprint(
            "No rows generated for training set: "
            f"valid_syms={len(valid_syms)}, window_cand_shape={window_cand.shape}, "
            f"ts_window=[{ts_start}, {ts_end_adj}]"
        )
        return None, None, None, None, None, None

    if _cached_tb is not None:
        tb_labels, tb_returns = _cached_tb
    elif label_method == "triple_barrier":
        # Use unified barrier factory (canonical TP/SL geometry)
        if "atr_pct" in feats:
            atr_pct = _coerce_feature_to_panel_df(feats["atr_pct"], panel, "atr_pct", fill_value=0.01)

            # Get config parameters for barrier factory
            k_tp = float(cfg.get("barrier_k_tp", 1.0))
            sl_base_mult = float(cfg.get("barrier_sl_base_mult", 0.5))
            disp_floor = float(cfg.get("barrier_disp_floor", 0.1))
            z_max = float(cfg.get("barrier_z_max", 3.0))
            k_reg = float(cfg.get("barrier_k_reg", 0.3))
            m_lo = float(cfg.get("barrier_m_lo", 0.7))
            m_hi = float(cfg.get("barrier_m_hi", 1.5))
            sl_lo = float(cfg.get("barrier_sl_lo", 0.4))
            sl_hi = float(cfg.get("barrier_sl_hi", 0.7))
            z_gate = float(cfg.get("barrier_z_gate", 1.0))
            H_base = float(cfg.get("label_horizon_base", 4))
            tp_lo = float(cfg.get("barrier_tp_lo_h2", 0.015)) if int(H) == 2 else float(cfg.get("barrier_tp_lo", 0.02))
            tp_hi = float(cfg.get("barrier_tp_hi", 0.06))
            tb_cache_key = (
                int(H),
                str(side),
                float(k_tp),
                float(sl_base_mult),
                float(disp_floor),
                float(z_max),
                float(k_reg),
                float(m_lo),
                float(m_hi),
                float(sl_lo),
                float(sl_hi),
                float(z_gate),
                float(tp_lo),
                float(tp_hi),
            )
            if _tb_cache is not None and tb_cache_key in _tb_cache:
                tb_labels, tb_returns = _tb_cache[tb_cache_key]
                tprint("Using cached barriers/triple-barrier labels")
            else:
                tprint("Computing barriers using unified factory...")
                tp_df, sl_df, diag = compute_barrier_factory(
                    atr_pct=atr_pct,
                    window_size=24 * 30,
                    k_tp=k_tp,
                    sl_base_mult=sl_base_mult,
                    horizon=H,
                    H_base=H_base,
                    disp_floor=disp_floor,
                    z_max=z_max,
                    k_reg=k_reg,
                    m_lo=m_lo,
                    m_hi=m_hi,
                    sl_lo=sl_lo,
                    sl_hi=sl_hi,
                    z_gate=z_gate,
                    tp_lo=tp_lo,
                    tp_hi=tp_hi,
                    return_components=True,
                )
                tprint(
                    f"Labeling: Unified Barrier Factory (Mean TP={diag['tp_mean']:.4f}, "
                    f"SL={diag['sl_mean']:.4f}, m∈[{diag['m_p10']:.2f},{diag['m_p90']:.2f}], "
                    f"m_at bounds: lo={diag['m_at_m_lo_pct']:.1%}, hi={diag['m_at_m_hi_pct']:.1%}, "
                    f"z_gate: below={diag['z_below_gate_pct']:.1%}, above={diag['z_above_gate_pct']:.1%}, "
                    f"sl_mult: lo={diag['sl_at_sl_lo_pct']:.1%}, hi={diag['sl_at_sl_hi_pct']:.1%})"
                )
                tb_labels, tb_returns = compute_triple_barrier_labels(
                    panel, tp_df, sl_df, H, side=side
                )
                if _tb_cache is not None:
                    _tb_cache[tb_cache_key] = (tb_labels, tb_returns)

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

    # Subsample: disabled - use all hours for maximum signal
    # window_cand = window_cand[window_cand.index.hour % 3 == 0]

    if feature_key:
        feat_keys = cfg.get(feature_key, [])
    else:
        feat_keys = cfg.get("causal_cols", [])

    if extra_feature_keys:
        # Add extra keys, preserving uniqueness
        feat_keys = list(set(feat_keys) | set(extra_feature_keys))

    # --- Vectorized event extraction using numpy ---
    valid_syms = [s for s in valid_syms if s in tb_labels.columns]
    if not valid_syms or window_cand.empty:
        tprint(
            "No rows generated for training set: "
            f"valid_syms={len(valid_syms)}, window_cand_shape={window_cand.shape}, "
            f"ts_window=[{ts_start}, {ts_end_adj}]"
        )
        return None, None, None, None, None, None

    # Pre-filter candidates to where entry_ts is present in tb_labels index.
    # Use UTC-ns comparison to avoid tz-aware vs tz-naive mismatch causing false-empty alignment.
    try:
        cand_ns = pd.to_datetime(window_cand.index, utc=True).view("i8")
        valid_entry_ns = pd.to_datetime(tb_labels.index, utc=True).view("i8") - int(pd.Timedelta(hours=1).value)
        align_mask = np.isin(cand_ns, valid_entry_ns)
        if align_mask.any():
            window_cand_aligned = window_cand.iloc[align_mask]
        else:
            tprint(
                "No rows after strict entry alignment; falling back to unaligned candidates "
                "and relying on entry_valid post-filter."
            )
            window_cand_aligned = window_cand
    except Exception:
        valid_entry_times = tb_labels.index - pd.Timedelta(hours=1)
        window_cand_aligned = window_cand[window_cand.index.isin(valid_entry_times)]
        if window_cand_aligned.empty:
            tprint(
                "No rows after aligning to tb_labels index (fallback path); "
                "using unaligned candidates and relying on entry_valid post-filter."
            )
            window_cand_aligned = window_cand

    sub_mask = window_cand_aligned[valid_syms]
    rows_idx, cols_idx = np.where(sub_mask.values)
    tprint(f"Candidate events: {len(rows_idx)}")
    if len(rows_idx) == 0:
        tprint(
            "No rows generated for training set: candidate event extraction returned 0 "
            f"(sub_mask_shape={sub_mask.shape})"
        )
        return None, None, None, None, None, None

    event_ts = sub_mask.index[rows_idx]
    event_sym = np.array(valid_syms)[cols_idx]
    entry_ts = event_ts + pd.Timedelta(hours=1)

    # Diagnostic: log gap rate (should now be 0% after pre-filtering)
    n_pre_entry = len(event_ts)
    # Robust timestamp matching across tz-aware/naive representations.
    try:
        entry_ns = pd.to_datetime(entry_ts, utc=True).view("i8")
        tb_ns = pd.to_datetime(tb_labels.index, utc=True).view("i8")
        entry_valid = np.isin(entry_ns, tb_ns)
    except Exception:
        entry_valid = entry_ts.isin(tb_labels.index)
    n_entry_drop = int((~entry_valid).sum())
    gap_rate = n_entry_drop / n_pre_entry if n_pre_entry > 0 else 0.0
    if n_entry_drop > 0:
        tprint(f"Entry alignment drop (H={H}): removed {n_entry_drop}/{n_pre_entry} events missing in tb_labels index")
        tprint(f"  Gap rate: {gap_rate*100:.1f}% | tb_labels range: {tb_labels.index.min()} to {tb_labels.index.max()}")
        tprint(f"  Sample missing hours: {entry_ts[~entry_valid][:5].tolist()}")
    event_ts = event_ts[entry_valid]
    event_sym = event_sym[entry_valid]
    entry_ts = event_ts + pd.Timedelta(hours=1)

    if len(event_ts) == 0:
        tprint(
            "No rows generated for training set: "
            f"all events dropped by entry alignment (pre={n_pre_entry}, drop={n_entry_drop})"
        )
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
        n_pre_trend = len(event_ts)
        n_trend_drop = int((~keep).sum())
        if n_trend_drop > 0:
            tprint(f"Trend filter drop ({trend_filter}): removed {n_trend_drop}/{n_pre_trend} events")
        event_ts = event_ts[keep]
        event_sym = event_sym[keep]
        entry_ts = event_ts + pd.Timedelta(hours=1)

    if len(event_ts) == 0:
        tprint(
            "No rows generated for training set: "
            f"trend_filter='{trend_filter}' removed all events"
        )
        return None, None, None, None, None, None

    tprint(f"Events after trend filter: {len(event_ts)}")

    # --- Fast numpy positional lookups (avoid stack/reindex) ---
    # Extract TB labels/returns at entry time
    lbl_vals = _fast_lookup(tb_labels, entry_ts, event_sym)
    ret_vals = _fast_lookup(tb_returns, entry_ts, event_sym)

    # PnL computation
    # For triple_barrier: ret_vals already encodes the correct directional return
    # (long returns for side="long", short returns for side="short").
    # No trade_dir adjustment needed — the side is baked into the barrier labels.
    pnl = ret_vals

    # Quantile-based labels + weighting.
    q_lo = float(cfg.get("label_quantile_lo", 0.30))
    q_hi = float(cfg.get("label_quantile_hi", 0.70))
    pnl_lo = np.quantile(pnl[np.isfinite(pnl)], q_lo) if np.sum(np.isfinite(pnl)) > 10 else 0.0
    pnl_hi = np.quantile(pnl[np.isfinite(pnl)], q_hi) if np.sum(np.isfinite(pnl)) > 10 else 0.0
    quantile_mode = str(cfg.get("label_quantile_mode", "weighted_union")).lower()
    w_quant = np.ones(len(pnl), dtype=np.float32)

    if quantile_mode == "hard_filter":
        keep_mask = (pnl <= pnl_lo) | (pnl >= pnl_hi)
        n_kept = int(keep_mask.sum())
        n_drop = int(len(pnl) - n_kept)
        tprint(f"Quantile labels [hard_filter]: lo_thr={pnl_lo:.6f}, hi_thr={pnl_hi:.6f}, "
               f"kept={n_kept}/{len(pnl)} ({keep_mask.mean()*100:.0f}%), dropped={n_drop}")

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
    else:
        # Weighted-union mode: keep all samples, emphasize tails continuously.
        ranks = pd.Series(pnl).rank(method="average", pct=True).values.astype(np.float32)
        dist = np.abs(ranks - 0.5) * 2.0  # 0 at median, 1 at extremes
        tail_floor = float(np.clip(cfg.get("label_quantile_weight_floor", 0.35), 0.0, 1.0))
        tail_gamma = float(np.clip(cfg.get("label_quantile_weight_gamma", 1.5), 0.5, 4.0))
        w_quant = tail_floor + (1.0 - tail_floor) * np.power(np.clip(dist, 0.0, 1.0), tail_gamma)
        tprint(
            f"Quantile labels [weighted_union]: lo_thr={pnl_lo:.6f}, hi_thr={pnl_hi:.6f}, "
            f"kept={len(pnl)}/{len(pnl)} (100%), w_quant(mean={w_quant.mean():.3f}, p10={np.quantile(w_quant, 0.10):.3f}, p90={np.quantile(w_quant, 0.90):.3f})"
        )

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
    w_base = w_base * w_quant

    # MFE/MAE-based weighting (Report 2026-02-12)
    # Weight by how "decisive" the price movement was relative to barriers
    # r_mfe = MFE/TP, r_mae = MAE/SL, d = max(r_mfe, r_mae)
    # w_mfe_mae = w_min + (1-w_min) * clip(d/tau, 0, 1)
    # This weights samples by excursion quality, not speed or net R:R
    
    # Get MFE/MAE from features (already computed in features.py)
    if "mfe_4h" in feats:
        mfe_vals = np.nan_to_num(_fast_lookup(feats["mfe_4h"], event_ts, event_sym), nan=0.0)
    else:
        mfe_vals = np.maximum(pnl, 0.0)
    if "mae_4h" in feats:
        mae_vals = np.nan_to_num(_fast_lookup(feats["mae_4h"], event_ts, event_sym), nan=0.0)
    else:
        mae_vals = np.maximum(-pnl, 0.0)
    
    # Get barrier distances (TP/SL) from ATR
    if "atr_pct" in feats:
        barrier_vals = np.nan_to_num(_fast_lookup(feats["atr_pct"], event_ts, event_sym), nan=0.02)
    else:
        barrier_vals = np.full(len(event_ts), 0.02, dtype=np.float32)
    tp_vals = np.clip(np.abs(barrier_vals), 1e-4, None)
    sl_vals = np.clip(0.5 * tp_vals, 1e-4, None)
    
    # Timeout detection: lbl_vals == 0 means timeout
    is_timeout = (lbl_vals == 0)
    
    # Compute MFE/MAE weights
    w_mfe_mae = compute_mfe_mae_weights(
        mfe=mfe_vals,
        mae=mae_vals,
        tp=tp_vals,
        sl=sl_vals,
        is_timeout=is_timeout,
        touch_margin=None,  # Not available in training data
        w_min=float(cfg.get("mfe_mae_w_min", 0.5)),
        tau=float(cfg.get("mfe_mae_tau", 1.0)),
        cost_floor=float(cfg.get("mfe_mae_cost_floor", 0.001))
    )
    
    # Multiply into base weight (before normalization)
    w_base = w_base * w_mfe_mae
    tprint(f"MFE/MAE weighting: mean={w_mfe_mae.mean():.3f}, p10={np.quantile(w_mfe_mae, 0.10):.3f}, p90={np.quantile(w_mfe_mae, 0.90):.3f}")

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
    # Store raw triple-barrier label (-1, 0, 1) for timeout analysis
    parts = {
        "ts": ts_arr,
        "symbol": sym_arr,
        "y_bin": y_bin,
        "y_ret": pnl.astype(np.float32),
        "w": weights_raw.astype(np.float32),
        "__y_lbl__": lbl_vals.astype(np.int8)
    }

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
        tprint(
            "No rows generated for training set: "
            "DataFrame empty after critical-column drop/fill"
        )
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
    
    # Extract selection metric values for event scoring
    # Use range_16h_pct - the high/low percentage difference used in candidate selection
    selection_metric_name = "range_16h_pct"
    selection_metric_values = None
    if selection_metric_name in feats:
        # Look up the metric values at event times and symbols
        selection_metric_values = _fast_lookup(feats[selection_metric_name], df["ts"].values, df["symbol"].values)
        selection_metric_values = np.nan_to_num(selection_metric_values, nan=0.0)
        tprint(f"Extracted selection metric '{selection_metric_name}' for event scoring")
    else:
        tprint(f"Warning: Selection metric '{selection_metric_name}' not found in features, using fallback")
    
    weights = compute_sample_weights_with_uniqueness(
        label_times=label_times,
        returns=returns,
        base_weights=base_weights,
        time_grid=time_grid,
        selection_metric=selection_metric_values
    )

    # Distance-to-barrier component (saturating alternative from sample-weight optimization plan)
    dist_component = None
    if bool(cfg.get("sample_weight_use_distance_component", True)):
        entry_px = np.ones(len(df), dtype=float)
        up_bar = 1.0 + np.clip(tp_vals[:len(df)], 1e-6, None)
        dn_bar = 1.0 - np.clip(sl_vals[:len(df)], 1e-6, None)
        atr_proxy = np.clip(np.abs(barrier_vals[:len(df)]), 1e-6, None)
        dist_component = compute_distance_to_barrier_weights(
            entry_prices=entry_px,
            upper_barriers=up_bar,
            lower_barriers=dn_bar,
            atr_past=atr_proxy,
            k=float(cfg.get("sample_weight_distance_k", 0.5)),
            min_dist=float(cfg.get("sample_weight_distance_min_dist", 0.5)),
            form=str(cfg.get("sample_weight_distance_form", "inverse")),
        )

    feature_cols_for_opt = [
        c for c in df.columns
        if c not in {"ts", "symbol", "y_bin", "y_ret", "w"} and np.issubdtype(df[c].dtype, np.number)
    ]
    if feature_cols_for_opt:
        weights = _optimize_training_sample_weights(
            df=df,
            X_frame=df[feature_cols_for_opt].fillna(0.0),
            y_ret=returns,
            label_times=label_times,
            base_weights=weights,
            cfg=cfg,
            stage="base",
            extra_components={"distance": dist_component} if dist_component is not None else None,
        )

    tprint(f"Applied uniqueness+optimized weighting: mean={weights.mean():.3f}, std={weights.std():.3f}")
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

def _get_bucket_label_config(cfg, side, kind):
    """Get per-bucket TP/SL/min_net_rr with fallback to global values."""
    bucket = f"{side}_{kind}"
    tp_key = f"label_tp_values_pct_{bucket}"
    sl_key = f"label_sl_values_pct_{bucket}"
    rr_key = f"label_min_net_rr_{bucket}"
    tp_vals = [float(x) / 100.0 for x in cfg.get(tp_key, cfg.get("label_tp_values_pct", [3.0, 4.0, 5.0, 6.0]))]
    sl_vals = [float(x) / 100.0 for x in cfg.get(sl_key, cfg.get("label_sl_values_pct", [0.5, 1.0, 2.0]))]
    min_net_rr = float(cfg.get(rr_key, cfg.get("label_min_net_rr", 1.2)))
    return tp_vals, sl_vals, min_net_rr


def build_grid_aggregated_tb_cache(panel, feats, cfg, horizons, trade_sides):
    """Build grid-aggregated triple-barrier labels shared across MR/TF for each (H, side)."""
    tb_cache = {}    # (H, side) -> (tb_labels, tb_returns)
    geom_cache = {}  # (H, side) -> {"n_tp", "n_sl", "n_to", "n_geom"}

    if "atr_pct" in feats:
        atr_pct_df = _coerce_feature_to_panel_df(feats["atr_pct"], panel, "atr_pct", fill_value=0.02)
    else:
        atr_pct_df = None

    fee_pct = float(cfg.get("label_round_trip_fee_pct", 0.5)) / 100.0
    min_tp_hit = float(cfg.get("label_min_tp_hit_rate", 0.02))
    min_tp_hit_h2 = float(cfg.get("label_min_tp_hit_rate_h2", 0.01))
    max_timeout = float(cfg.get("label_max_timeout_rate", 0.90))
    max_timeout_h2 = float(cfg.get("label_max_timeout_rate_h2", 0.97))

    # Global TP/SL values interpreted as multipliers relative to ATR%.
    tp_mults = cfg.get("barrier_k_tp_grid", [0.8, 1.0, 1.25, 1.6, 2.0, 2.5])
    sl_base_mults = cfg.get("barrier_sl_base_grid", [0.5, 1.0, 1.5])
    min_net_rr = float(cfg.get("label_min_net_rr", 0.9))
    min_net_rr_h2 = float(cfg.get("label_min_net_rr_h2", 0.75))
    min_events_h2 = int(cfg.get("label_min_events_h2", 50))
    h2_rescue_topk = int(cfg.get("label_h2_rescue_topk", 3))

    # Unified barrier factory params (shared across all geometries)
    disp_floor = float(cfg.get("barrier_disp_floor", 0.1))
    z_max = float(cfg.get("barrier_z_max", 3.0))
    k_reg = float(cfg.get("barrier_k_reg", 0.3))
    m_lo = float(cfg.get("barrier_m_lo", 0.7))
    m_hi = float(cfg.get("barrier_m_hi", 1.5))
    sl_lo = float(cfg.get("barrier_sl_lo", 0.4))
    sl_hi = float(cfg.get("barrier_sl_hi", 0.7))
    z_gate = float(cfg.get("barrier_z_gate", 1.0))
    H_base = float(cfg.get("label_horizon_base", 4))
    tp_lo = float(cfg.get("barrier_tp_lo", 0.02))
    tp_lo_h2 = float(cfg.get("barrier_tp_lo_h2", 0.015))
    tp_hi = float(cfg.get("barrier_tp_hi", 0.06))

    # Cache raw triple barrier results per (H, side, k_tp, sl_base_mult) to avoid recomputation
    _raw_tb_cache = {}

    for H in horizons:
        for side in trade_sides:
            min_tp_hit_eff = min_tp_hit_h2 if int(H) == 2 else min_tp_hit
            tp_lo_eff = tp_lo_h2 if int(H) == 2 else tp_lo
            max_timeout_eff = max_timeout_h2 if int(H) == 2 else max_timeout
            min_net_rr_eff = min_net_rr_h2 if int(H) == 2 else min_net_rr
            min_events_eff = min_events_h2 if int(H) == 2 else 100
            reject_counts = {
                "rr": 0,
                "n_events": 0,
                "tp_hit": 0,
                "timeout": 0,
            }
            total_geoms = 0
            if atr_pct_df is None:
                atr_pct_local = pd.DataFrame(0.02, index=panel["close"].index, columns=panel["close"].columns)
            else:
                atr_pct_local = atr_pct_df.fillna(0.02)

            tprint(f"Pre-computing geometry labels H={H} side={side} (k_tp={tp_mults}, sl_base={sl_base_mults})...")

            geom_runs = []
            relaxed_pool = []
            for k_tp in tp_mults:
                for sl_base_mult in sl_base_mults:
                    total_geoms += 1
                    tp_df, sl_df = compute_barrier_factory(
                        atr_pct=atr_pct_local,
                        window_size=24 * 30,
                        k_tp=k_tp,
                        sl_base_mult=sl_base_mult,
                        horizon=H,
                        H_base=H_base,
                        disp_floor=disp_floor,
                        z_max=z_max,
                        k_reg=k_reg,
                        m_lo=m_lo,
                        m_hi=m_hi,
                        sl_lo=sl_lo,
                        sl_hi=sl_hi,
                        z_gate=z_gate,
                        tp_lo=tp_lo_eff,
                        tp_hi=tp_hi,
                    )

                    net_rr = k_tp / max(sl_base_mult + fee_pct / (k_tp + 1e-9), 1e-9)
                    if net_rr < min_net_rr_eff:
                        reject_counts["rr"] += 1
                        continue
                    raw_key = (H, side, round(float(k_tp), 4), round(float(sl_base_mult), 4))
                    if raw_key not in _raw_tb_cache:
                        lbl, ret = compute_triple_barrier_labels(panel, tp_df, sl_df, H, side=side)
                        _raw_tb_cache[raw_key] = (lbl, ret)
                    lbl, ret = _raw_tb_cache[raw_key]

                    n_events = lbl.size
                    tp_hit = float((lbl.values == 1).sum()) / max(1, n_events)
                    sl_hit = float((lbl.values == -1).sum()) / max(1, n_events)
                    to_rate = float((lbl.values == 0).sum()) / max(1, n_events)

                    if n_events < min_events_eff:
                        reject_counts["n_events"] += 1
                        continue
                    relaxed_pool.append((lbl, ret, k_tp, sl_base_mult, net_rr, tp_hit, sl_hit, to_rate, n_events))
                    if tp_hit < min_tp_hit_eff:
                        reject_counts["tp_hit"] += 1
                        continue
                    if to_rate > max_timeout_eff:
                        reject_counts["timeout"] += 1
                        continue

                    rr_weight = float(np.clip(net_rr / 1.2, 0.0, 1.0))
                    geom_runs.append((lbl, ret, k_tp, sl_base_mult, rr_weight, tp_hit, sl_hit, to_rate, n_events))

            if not geom_runs:
                # H=2 rescue: keep top-K geometries by tp_hit if strict gates still reject all.
                if int(H) == 2 and relaxed_pool:
                    relaxed_pool.sort(key=lambda x: (x[5], -x[7]), reverse=True)  # tp_hit desc, timeout asc
                    picked = relaxed_pool[: max(1, h2_rescue_topk)]
                    for (lbl, ret, k_tp, sl_base_mult, net_rr, tp_hit, sl_hit, to_rate, n_events) in picked:
                        rr_weight = float(np.clip(net_rr / 1.2, 0.0, 1.0))
                        geom_runs.append((lbl, ret, k_tp, sl_base_mult, rr_weight, tp_hit, sl_hit, to_rate, n_events))
                    tprint(
                        f"H=2 rescue accepted {len(geom_runs)} geometries "
                        f"(best tp_hit={picked[0][5]:.4f}, timeout={picked[0][7]:.4f})"
                    )

            if not geom_runs:
                tprint(f"No valid geometry for H={H} side={side}; using fallback.")
                tprint(
                    "Geometry rejection breakdown: "
                    f"H={H}, side={side}, total={total_geoms}, "
                    f"rr={reject_counts['rr']}, n_events={reject_counts['n_events']}, "
                    f"tp_hit={reject_counts['tp_hit']}, timeout={reject_counts['timeout']}, "
                    f"min_tp_hit={min_tp_hit_eff:.4f}, tp_lo={tp_lo_eff:.4f}, "
                    f"max_timeout={max_timeout_eff:.4f}, min_rr={min_net_rr_eff:.4f}, "
                    f"min_events={min_events_eff}"
                )
                tp_df, sl_df = compute_barrier_factory(
                    atr_pct=atr_pct_local,
                    window_size=24 * 30,
                    k_tp=1.0,
                    sl_base_mult=0.5,
                    horizon=H,
                    H_base=H_base,
                    disp_floor=disp_floor,
                    z_max=z_max,
                    k_reg=k_reg,
                    m_lo=m_lo,
                    m_hi=m_hi,
                    sl_lo=sl_lo,
                    sl_hi=sl_hi,
                    z_gate=z_gate,
                    tp_lo=tp_lo_eff,
                    tp_hi=tp_hi,
                )
                lbl, ret = compute_triple_barrier_labels(panel, tp_df, sl_df, H, side=side)
                geom_runs = [(lbl, ret, 1.0, 0.5, 1.0, 0.1, 0.1, 0.8, lbl.size)]
            else:
                # Diagnostics: print accepted geometry count + params to trace what survives gates.
                geom_desc = []
                for _, _, k_tp_v, sl_base_v, rr_w, tp_hit_v, sl_hit_v, to_rate_v, n_ev_v in geom_runs:
                    geom_desc.append(
                        f"(k_tp={k_tp_v:.2f}, sl={sl_base_v:.2f}, tp_hit={tp_hit_v:.3f}, "
                        f"to={to_rate_v:.3f}, n={int(n_ev_v)}, w={rr_w:.3f})"
                    )
                tprint(
                    f"Accepted geometries H={H} side={side}: {len(geom_runs)} | "
                    + "; ".join(geom_desc)
                )

            labels_stack = np.stack([x[0].values for x in geom_runs], axis=0)
            rets_stack = np.stack([x[1].values for x in geom_runs], axis=0)
            rr_weights_raw = np.array([x[4] for x in geom_runs], dtype=np.float32)

            rr_weights = np.sqrt(rr_weights_raw)
            rr_weights = rr_weights / (rr_weights.mean() + 1e-12)

            w_tp = np.zeros_like(labels_stack[0], dtype=np.float32)
            w_sl = np.zeros_like(labels_stack[0], dtype=np.float32)
            w_to = np.zeros_like(labels_stack[0], dtype=np.float32)
            for gi in range(len(geom_runs)):
                w_tp += rr_weights[gi] * (labels_stack[gi] == 1).astype(np.float32)
                w_sl += rr_weights[gi] * (labels_stack[gi] == -1).astype(np.float32)
                w_to += rr_weights[gi] * (labels_stack[gi] == 0).astype(np.float32)
            n_tp_df = pd.DataFrame(w_tp, index=panel["close"].index, columns=panel["close"].columns)
            n_sl_df = pd.DataFrame(w_sl, index=panel["close"].index, columns=panel["close"].columns)
            n_to_df = pd.DataFrame(w_to, index=panel["close"].index, columns=panel["close"].columns)

            w_sum = rr_weights.sum()
            agg_ret = np.average(rets_stack, axis=0, weights=rr_weights) if w_sum > 0 else np.nanmean(rets_stack, axis=0)
            agg_lbl = np.where(n_tp_df.values > n_sl_df.values, 1, np.where(n_sl_df.values > n_tp_df.values, -1, 0)).astype(np.int8)
            tb_labels = pd.DataFrame(agg_lbl, index=panel["close"].index, columns=panel["close"].columns)
            tb_returns = pd.DataFrame(agg_ret, index=panel["close"].index, columns=panel["close"].columns)

            tb_cache[(H, side)] = (tb_labels, tb_returns)
            geom_cache[(H, side)] = {"n_tp": n_tp_df, "n_sl": n_sl_df, "n_to": n_to_df, "n_geom": len(geom_runs)}

    return tb_cache, geom_cache

def generate_label_datasets(panel, feats, mkt_gates, cfg, syms, ts, p_exh_hist):
    tprint(f"Entering function: generate_label_datasets in training.py")
    datasets = {}

    # Pre-compute shared expensive operations once
    tprint("Pre-computing candidate mask (shared across all steps)...")
    train_pct, _train_min_range, _train_min_vol = _get_training_candidate_config(cfg)
    cached_cand_mask = select_trade_candidates_vectorized(
        panel,
        feats,
        pct=train_pct,
        metric=cfg["trade_deviation_metric"],
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

    # Pre-compute triple-barrier labels with geometry-grid aggregation per (H, side)
    tb_cache, geom_cache = build_grid_aggregated_tb_cache(
        panel=panel,
        feats=feats,
        cfg=cfg,
        horizons=horizons,
        trade_sides=trade_sides,
    )

    if geom_cache:
        any_key = next(iter(geom_cache.keys()))
        feats["__geom_n_tp__"] = geom_cache[any_key]["n_tp"]
        feats["__geom_n_sl__"] = geom_cache[any_key]["n_sl"]
        feats["__geom_n_to__"] = geom_cache[any_key]["n_to"]

    for side in trade_sides:
        for k in kinds:
            trade_side = side
            cand_filter, move_bucket, strategy_label = _strategy_bucket_context(trade_side, k)
            trend_filter = move_bucket

            feat_key = "tf_feature_keys" if k == "tf" else "mr_feature_keys"

            fixed_tp = 0.05
            fixed_sl = 0.025

            for H in horizons:
                tprint(
                    f"Generating labels: trade_side={trade_side}, kind={k}, "
                    f"move_bucket={move_bucket}, candidate_bucket={cand_filter}, "
                    f"strategy={strategy_label}, H={H}"
                )

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
        X, y, w, _ = build_exhaustion_Xy(panel, feats, mkt_gates, cfg, ts, lookback, syms, trend_filter=d)
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

    IC_target: global rank corr of pred vs transformed target (all time+assets)
    IC_raw:    global rank corr of pred vs raw returns (all time+assets)
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

    ic_target = _safe_spearman(y_pred, y_true)
    ic_raw = _safe_spearman(y_pred, y_raw_ret) if y_raw_ret is not None else float('nan')

    trades_day_10 = _avg_trades_per_day_global(y_pred, 0.10, np.asarray(groups) if groups is not None else None)
    trades_day_30 = _avg_trades_per_day_global(y_pred, 0.30, np.asarray(groups) if groups is not None else None)

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

def train_meta_models_from_artifacts(datasets, cfg, alpha_models):
    """Train only meta models from datasets and pre-trained alpha models."""
    import time as _time
    _t0_meta = _time.monotonic()
    tprint("train_meta_models_from_artifacts: starting")
    trade_sides = ["long", "short"]
    kinds = ["mr", "tf"]
    meta_models = {}
    meta_gate_results = []
    _bucket_y_ret = {}  # per-bucket raw returns for OOF saving

    def _collect_horizon_oof(side_name, kind_name):
        conf_local = alpha_models.get(side_name, {}).get(kind_name) if alpha_models else None
        if not conf_local:
            return {}, {2: "missing_alpha_bucket", 4: "missing_alpha_bucket", 8: "missing_alpha_bucket"}
        models_by_h_local = conf_local.get("models_by_h", {})
        out = {}
        skip = {}
        for h_local in [2, 4, 8]:
            ds_key_local = f"train_{side_name}_{kind_name}_{h_local}"
            if ds_key_local not in datasets:
                skip[h_local] = "missing_dataset"
                continue
            race_local = models_by_h_local.get(h_local, {}).get("model") if h_local in models_by_h_local else None
            if race_local is None:
                skip[h_local] = "missing_alpha_model"
                continue
            if race_local.oof_probs is None:
                skip[h_local] = "missing_oof_probs"
                continue
            df_local = datasets[ds_key_local]
            oof_local = np.asarray(race_local.oof_probs, dtype=float)
            if len(oof_local) != len(df_local):
                skip[h_local] = f"oof_len_mismatch:{len(oof_local)}!={len(df_local)}"
                continue
            out[h_local] = (df_local, oof_local)
        return out, skip

    def _align_oof_to_union(df_union, source_df, source_oof):
        if "__ts__" in df_union.columns and "__symbol__" in df_union.columns and "__ts__" in source_df.columns and "__symbol__" in source_df.columns:
            src_lookup = {
                kk: i for i, kk in enumerate(zip(source_df["__ts__"].values, source_df["__symbol__"].values))
            }
            union_keys = list(zip(df_union["__ts__"].values, df_union["__symbol__"].values))
            src_idx = np.array([src_lookup.get(kk, -1) for kk in union_keys], dtype=int)
            valid = src_idx >= 0
            aligned = np.full(len(df_union), 0.5, dtype=np.float32)
            aligned[valid] = source_oof[src_idx[valid]].astype(np.float32)
            return aligned
        n_use = min(len(df_union), len(source_oof))
        aligned = np.full(len(df_union), 0.5, dtype=np.float32)
        aligned[:n_use] = source_oof[:n_use].astype(np.float32)
        return aligned

    for side in trade_sides:
        for k in kinds:
            trade_side = side
            cand_filter, move_bucket, strategy_label = _strategy_bucket_context(trade_side, k)
            tprint(
                f"Meta bucket context: trade_side={trade_side}, kind={k}, "
                f"move_bucket={move_bucket}, candidate_bucket={cand_filter}, strategy={strategy_label}"
            )
            conf = alpha_models.get(side, {}).get(k) if alpha_models else None
            if not conf:
                tprint(f"Meta {side}_{k}: skipped (missing alpha model)")
                continue

            # Primary horizon OOF set (this bucket's own alpha models)
            horizon_dfs, horizon_skip_reasons = _collect_horizon_oof(side, k)
            for h, reason in horizon_skip_reasons.items():
                if reason.startswith("oof_len_mismatch"):
                    ds_key_dbg = f"train_{side}_{k}_{h}"
                    _df_len = len(datasets.get(ds_key_dbg, [])) if ds_key_dbg in datasets else "na"
                    _oof_len = reason.split(":")[-1].split("!=")[0] if ":" in reason else "na"
                    tprint(f"Meta {side}_{k} H={h}: OOF length mismatch ({_oof_len} vs {_df_len}), skipping horizon")

            if not horizon_dfs:
                tprint(f"Meta {side}_{k}: skipped (no horizon with valid OOF)")
                if horizon_skip_reasons:
                    tprint(f"  Horizon availability diagnostics: {horizon_skip_reasons}")
                continue
            tprint(f"Meta {side}_{k}: {len(horizon_dfs)} horizons available: {sorted(horizon_dfs.keys())} ({_time.monotonic()-_t0_meta:.1f}s)")
            if horizon_skip_reasons:
                tprint(f"  Horizons excluded: {horizon_skip_reasons}")

            # Use the largest horizon dataset as the base (for meta feature columns)
            base_H = max(horizon_dfs.keys(), key=lambda h: len(horizon_dfs[h][0]))
            df_base = horizon_dfs[base_H][0]

            # Build (ts, symbol) index for union + diagnostics
            tprint(f"  Building union key index ({len(df_base)} base rows)...")
            if "__ts__" in df_base.columns and "__symbol__" in df_base.columns:
                base_keys = list(zip(df_base["__ts__"].values, df_base["__symbol__"].values))
                union_keys = []
                seen_keys = set()
                key_to_idx_by_h = {}

                for kk in base_keys:
                    if kk not in seen_keys:
                        union_keys.append(kk)
                        seen_keys.add(kk)

                for h in sorted(horizon_dfs.keys()):
                    df_h, _ = horizon_dfs[h]
                    ts_vals = df_h["__ts__"]
                    sym_vals = df_h["__symbol__"]
                    null_key_rows = int((ts_vals.isna() | sym_vals.isna()).sum())
                    h_keys = list(zip(ts_vals.values, sym_vals.values))
                    uniq_keys = len(set(h_keys))
                    dup_keys = len(h_keys) - uniq_keys
                    if null_key_rows > 0 or dup_keys > 0:
                        tprint(
                            f"    H{h} key hygiene: rows={len(df_h)}, unique={uniq_keys}, duplicates={dup_keys}, null_keys={null_key_rows}"
                        )
                    key_to_idx_by_h[h] = {kxy: i for i, kxy in enumerate(h_keys)}
                    for kk in h_keys:
                        if kk not in seen_keys:
                            union_keys.append(kk)
                            seen_keys.add(kk)

                inter_count = sum(1 for kk in union_keys if all(kk in key_to_idx_by_h[h] for h in horizon_dfs.keys()))
                tprint(f"  Union: {len(union_keys)} unique samples across horizons (intersection={inter_count})")
                for h in sorted(horizon_dfs.keys()):
                    present = np.array([kk in key_to_idx_by_h[h] for kk in union_keys], dtype=bool)
                    miss_cnt = int((~present).sum())
                    coverage = float(present.mean() * 100.0)
                    tprint(f"    H{h} coverage on union: {present.sum()}/{len(union_keys)} ({coverage:.1f}%), missing={miss_cnt}")

                base_lookup = {kk: i for i, kk in enumerate(base_keys)}
                for h in sorted(horizon_dfs.keys()):
                    if h == base_H:
                        continue
                    h_key_set = set(key_to_idx_by_h[h].keys())
                    base_key_set = set(base_lookup.keys())
                    base_only = len(base_key_set - h_key_set)
                    h_only = len(h_key_set - base_key_set)
                    if base_only > 0 or h_only > 0:
                        tprint(
                            f"    H{h} vs base H{base_H}: base_only={base_only}, h_only={h_only}"
                        )

                df = df_base.iloc[[base_lookup[kk] for kk in union_keys if kk in base_lookup]].copy()

                missing_keys = [kk for kk in union_keys if kk not in base_lookup]
                if missing_keys:
                    donor_horizons = sorted(horizon_dfs.keys(), key=lambda hh: (-len(horizon_dfs[hh][0]), hh))
                    extra_rows = []
                    for kk in missing_keys:
                        for h in donor_horizons:
                            h_idx = key_to_idx_by_h[h].get(kk, -1)
                            if h_idx >= 0:
                                extra_rows.append(horizon_dfs[h][0].iloc[h_idx].reindex(df_base.columns))
                                break
                    if extra_rows:
                        df_extra = pd.DataFrame(extra_rows, columns=df_base.columns)
                        df = pd.concat([df, df_extra], axis=0, ignore_index=True)

                df = df.reset_index(drop=True)
                tprint(f"  Union dataset built: {len(df)} rows ({_time.monotonic()-_t0_meta:.1f}s)")
            else:
                # Fallback: use min-length truncation
                min_len = min(len(df_h) for df_h, _ in horizon_dfs.values())
                keep_mask = np.ones(len(df_base), dtype=bool)
                keep_mask[min_len:] = False
                df = df_base.loc[keep_mask].reset_index(drop=True).copy()
                tprint(f"  No ts/symbol columns; truncating to {min_len} samples")

            if len(df) < 100:
                tprint(f"Meta {side}_{k}: skipped (only {len(df)} union samples)")
                continue

            # Build OOF predictions for each horizon, aligned to common samples
            tprint(f"  Aligning OOF predictions across horizons...")
            pred_h = pd.DataFrame(index=df.index)
            p_oof_avg_parts = []
            for h in sorted(horizon_dfs.keys()):
                df_h, oof_h = horizon_dfs[h]
                p_h = _align_oof_to_union(df, df_h, oof_h)
                pred_h[f"pred_{k}_H{h}"] = p_h
                pred_h[f"pred_H{h}"] = p_h
                p_oof_avg_parts.append(p_h)

            # Ensure side-level meta models see all same-side base outputs (TF + MR)
            for k_other in kinds:
                other_horizon_dfs, _ = _collect_horizon_oof(side, k_other)
                for h in [2, 4, 8]:
                    col_name = f"pred_{k_other}_H{h}"
                    if col_name in pred_h.columns:
                        continue
                    if h in other_horizon_dfs:
                        df_h_other, oof_h_other = other_horizon_dfs[h]
                        pred_h[col_name] = _align_oof_to_union(df, df_h_other, oof_h_other)
                    else:
                        pred_h[col_name] = np.full(len(df), 0.5, dtype=np.float32)
            tprint(
                f"  Meta {side}_{k}: added same-side TF/MR base OOF features "
                f"(cols={len([c for c in pred_h.columns if c.startswith('pred_')])})"
            )

            y_ret = df["__y_ret__"].values

            tprint(f"  OOF alignment done ({_time.monotonic()-_t0_meta:.1f}s). Building meta features...")

            # Build y_target from per-horizon returns (aligned to common samples)
            _has_keys = "__ts__" in df.columns and "__symbol__" in df.columns
            if _has_keys:
                _common_keys = list(zip(df["__ts__"].values, df["__symbol__"].values))
            _h_lookup_cache = {}

            def _ret_for_h_aligned(h):
                ds_key = f"train_{side}_{k}_{h}"
                for source_df in [datasets.get(ds_key), horizon_dfs.get(h, (None,))[0]]:
                    if source_df is None:
                        continue
                    if not (_has_keys and "__ts__" in source_df.columns and "__symbol__" in source_df.columns):
                        continue
                    cache_id = id(source_df)
                    if cache_id not in _h_lookup_cache:
                        _h_lookup_cache[cache_id] = {
                            kk: i for i, kk in enumerate(
                                zip(source_df["__ts__"].values, source_df["__symbol__"].values)
                            )
                        }
                    h_lookup = _h_lookup_cache[cache_id]
                    h_idx = np.array([h_lookup.get(ck, -1) for ck in _common_keys])
                    valid = h_idx >= 0
                    if not valid.any():
                        continue
                    ret = np.zeros(len(df), dtype=np.float32)
                    ret[valid] = source_df["__y_ret__"].values[h_idx[valid]].astype(np.float32)
                    return ret
                return y_ret.astype(np.float32)

            tprint(f"  Computing meta target ({_time.monotonic()-_t0_meta:.1f}s)...")
            _r2, _r4, _r8 = _ret_for_h_aligned(2), _ret_for_h_aligned(4), _ret_for_h_aligned(8)
            # Per-horizon returns for multi-barrier classifier labels
            _y_per_h = {2: _r2, 4: _r4, 8: _r8}

            # Per-horizon IC diagnostics
            _oof_by_h = {}
            for h in sorted(horizon_dfs.keys()):
                _oof_by_h[h] = pred_h[f"pred_H{h}"].values
            for _hh in sorted(_y_per_h.keys()):
                if _hh not in _oof_by_h:
                    continue
                _ic_h = _safe_spearman(_oof_by_h[_hh], _y_per_h[_hh])
                tprint(f"    IC(oof_H{_hh}, r_H{_hh}) = {_ic_h:.4f}")

            # p_oof: average OOF across all horizons (used for filtering & diagnostics)
            p_oof = np.mean(p_oof_avg_parts, axis=0)

            # Vol proxy for normalization variants (barrier_pct if available)
            _vol_proxy = df["__barrier_pct__"].values.astype(np.float64) if "__barrier_pct__" in df.columns else None

            configured_meta = cfg.get("meta_feature_keys", [])
            raw_prefix = "__meta_raw__"
            feat_cols = [mk for mk in configured_meta if f"{raw_prefix}{mk}" in df.columns]
            feat_cols = list(dict.fromkeys(feat_cols))
            exclude_key = f"meta_feature_exclude_{k}"
            exclude_set = set(cfg.get(exclude_key, []))
            if exclude_set:
                n_before = len(feat_cols)
                feat_cols = [f for f in feat_cols if f not in exclude_set]
                tprint(f"  Meta feature exclusion ({k}): {n_before} -> {len(feat_cols)} features ({n_before - len(feat_cols)} excluded)")
            if not feat_cols:
                tprint(f"Meta {side}_{k}: skipped (no raw meta features found in dataset)")
                continue

            X_feats = pd.DataFrame(index=df.index)
            for mk in feat_cols:
                X_feats[mk] = df[f"{raw_prefix}{mk}"].values
            X_feats = X_feats.fillna(0.0).astype(np.float32)

            n_res = df.get("__n_res__", pd.Series(np.ones(len(df)), index=df.index)).values.astype(np.float32)
            keep = n_res >= np.quantile(n_res, 0.20)
            if keep.sum() < 100:
                keep = np.ones(len(df), dtype=bool)

            df = df.loc[keep].reset_index(drop=True).copy()
            X_feats = X_feats.loc[keep].reset_index(drop=True).copy()
            pred_h = pred_h.loc[keep].reset_index(drop=True).copy()
            if _vol_proxy is not None:
                _vol_proxy = _vol_proxy[keep]
            p_oof = p_oof[keep]
            n_res = n_res[keep]
            _y_per_h = {h: v[keep] for h, v in _y_per_h.items()}

            # Build per-horizon logit features (all 3 horizons as separate inputs)
            from scipy.special import logit as _logit_fn
            from scipy.special import expit as _sigmoid
            _logit_parts = []
            for h in sorted(horizon_dfs.keys()):
                _p_h = np.clip(pred_h[f"pred_H{h}"].values.astype(float), 1e-4, 1 - 1e-4)
                _lg_h = np.clip(_logit_fn(_p_h), -4.0, 4.0)
                X_feats[f"pred_logit_H{h}"] = _lg_h.astype(np.float32)
                _logit_parts.append(_lg_h)
            pred_logit_avg = np.mean(_logit_parts, axis=0).astype(np.float32)
            X_feats["pred_logit"] = pred_logit_avg

            # Also keep raw pred_H columns as features
            X_feats = pd.concat([X_feats, pred_h], axis=1)

            # 4 disagreement features per base kind (computed independently on TF and MR OOFs)
            for kind_name in ["tf", "mr"]:
                p2 = pred_h[f"pred_{kind_name}_H2"].values.astype(np.float32)
                p4 = pred_h[f"pred_{kind_name}_H4"].values.astype(np.float32)
                p8 = pred_h[f"pred_{kind_name}_H8"].values.astype(np.float32)
                stack = np.vstack([p2, p4, p8]).T.astype(np.float32)
                pair_abs = (np.abs(p2 - p4) + np.abs(p2 - p8) + np.abs(p4 - p8)) / 3.0
                vote_p = (stack > 0.5).mean(axis=1).astype(np.float32)
                X_feats[f"disagree_{kind_name}_std"] = np.std(stack, axis=1, dtype=np.float32).astype(np.float32)
                X_feats[f"disagree_{kind_name}_range"] = (np.max(stack, axis=1) - np.min(stack, axis=1)).astype(np.float32)
                X_feats[f"disagree_{kind_name}_pair_abs"] = pair_abs.astype(np.float32)
                X_feats[f"disagree_{kind_name}_vote_mix"] = (4.0 * vote_p * (1.0 - vote_p)).astype(np.float32)
                if kind_name == "tf":
                    agree_tf_avg = (1.0 - np.clip(pair_abs, 0.0, 1.0)).astype(np.float32)
                else:
                    agree_mr_avg = (1.0 - np.clip(pair_abs, 0.0, 1.0)).astype(np.float32)

            # Additional 4 cross-kind features requested:
            # 1) average agreement TF - average agreement MR
            # 2-4) TF - MR per horizon (H2/H4/H8)
            X_feats["agree_tf_minus_mr_avg"] = (
                agree_tf_avg - agree_mr_avg
            ).astype(np.float32)
            for h in [2, 4, 8]:
                X_feats[f"tf_minus_mr_H{h}"] = (
                    pred_h[f"pred_tf_H{h}"].values.astype(np.float32) - pred_h[f"pred_mr_H{h}"].values.astype(np.float32)
                ).astype(np.float32)

            X_meta_base = X_feats.fillna(0.0)

            # Add interaction features
            pred_logit = pred_logit_avg
            for interact_feat in ["vol_z", "mkt_rv_ratio", "ambig", "exh_qual", "trend_pct",
                                  "trend_t", "trend_z_t", "spike_score", "grind_score", "chop_score"]:
                if interact_feat in X_meta_base.columns:
                    X_meta_base[f"pred_x_{interact_feat}"] = pred_logit * X_meta_base[interact_feat].values

            # Regime interactions: G_VOL, G_TREND (binary)
            for regime_col in ["G_VOL", "G_TREND"]:
                raw_regime = f"{raw_prefix}{regime_col}"
                if raw_regime in df.columns:
                    rv = df[raw_regime].values
                    for rbucket in [0, 1, 2]:
                        X_meta_base[f"pred_x_{regime_col}_{rbucket}"] = pred_logit * (rv == rbucket).astype(float)

            # Granular regime interactions: vol, volume, trend (3-state, 12h & 48h)
            for regime_raw_col in ["__regime_vol_12h__", "__regime_vol_48h__",
                                   "__regime_volume_12h__", "__regime_volume_48h__",
                                   "__regime_trend_12h__", "__regime_trend_48h__"]:
                if regime_raw_col in df.columns:
                    rv = df[regime_raw_col].values
                    _rname = regime_raw_col.replace("__regime_", "").replace("__", "")
                    for rbucket in [0, 1, 2]:
                        X_meta_base[f"pred_x_{_rname}_{rbucket}"] = pred_logit * (rv == rbucket).astype(float)

            # Cross-timeframe context features
            if "trend_slope_48h" in X_meta_base.columns and "trend_slope_120h" in X_meta_base.columns:
                _ts48 = X_meta_base["trend_slope_48h"].values
                _ts120 = X_meta_base["trend_slope_120h"].values
                X_meta_base["trend_slope_ratio_48_120"] = np.where(
                    np.abs(_ts120) > 1e-9, _ts48 / np.clip(np.abs(_ts120), 1e-9, None), 0.0).astype(np.float32)
            if "__regime_vol_12h__" in df.columns and "__regime_vol_48h__" in df.columns:
                _v12 = df["__regime_vol_12h__"].values
                _v48 = df["__regime_vol_48h__"].values
                X_meta_base["vol_regime_agree"] = (_v12 == _v48).astype(np.float32)
                X_meta_base["vol_regime_diff"] = (_v12 - _v48).astype(np.float32)
            if "__regime_trend_12h__" in df.columns and "__regime_trend_48h__" in df.columns:
                _t12 = df["__regime_trend_12h__"].values
                _t48 = df["__regime_trend_48h__"].values
                X_meta_base["trend_regime_agree"] = (_t12 == _t48).astype(np.float32)
                X_meta_base["trend_regime_diff"] = (_t12 - _t48).astype(np.float32)

            meta_groups = df["__ts__"].values if "__ts__" in df.columns else None

            # ══════════════════════════════════════════════════════════════
            # PER-HORIZON REGRESSOR TRAINING (H2, H4, H8)
            # ══════════════════════════════════════════════════════════════
            _available_horizons = sorted(_y_per_h.keys())
            tprint(f"  Training per-horizon regressors for {side}_{k}: H={_available_horizons}")

            for _h in _available_horizons:
                _h_label = f"{side}_{k}_H{_h}"
                y_ret_raw_h = _y_per_h[_h].astype(np.float64)

                # Sample weights: magnitude sigmoid (moderate top-30% upweight) + MFE/MAE quality
                # Each source is normalized to mean=1 before combining so neither dominates.
                _alpha_w = float(cfg.get("meta_weight_sigmoid_alpha", 0.5))
                _y_abs = np.abs(y_ret_raw_h)
                _fin_w = np.isfinite(_y_abs)
                _q70 = float(np.percentile(_y_abs[_fin_w], 70))
                _s_w = max(float(np.std(_y_abs[_fin_w])), 1e-9)
                # sigmoid centered at p70: top-30% get ~1.25-1.5x, bottom-70% get ~1.0x
                w_mag = 1.0 + _alpha_w * _sigmoid((_y_abs - _q70) / _s_w)
                w_mag = w_mag / max(float(np.mean(w_mag)), 1e-12)  # normalize to mean=1

                _mfe_col = f"__meta_raw__mfe_{_h}h"
                _mae_col = f"__meta_raw__mae_{_h}h"
                _bp = df["__barrier_pct__"].values if "__barrier_pct__" in df.columns else None
                if _mfe_col in df.columns and _mae_col in df.columns and _bp is not None:
                    _mfe_v = np.nan_to_num(df[_mfe_col].values, nan=0.0).astype(np.float64)
                    _mae_v = np.nan_to_num(df[_mae_col].values, nan=0.0).astype(np.float64)
                    _bp_v = np.clip(_bp.astype(np.float64), 1e-6, None)
                    _d_exc = np.maximum(np.abs(_mfe_v) / _bp_v, np.abs(_mae_v) / _bp_v)
                    _tau_exc = float(cfg.get("meta_mfe_mae_tau", 1.0))
                    w_exc = 0.5 + 0.5 * np.clip(_d_exc / _tau_exc, 0.0, 1.0)
                else:
                    w_exc = np.ones(len(df), dtype=np.float64)
                w_exc = w_exc / max(float(np.mean(w_exc)), 1e-12)  # normalize to mean=1

                w_meta_h = (w_mag * w_exc).astype(np.float64)
                w_meta_h = w_meta_h / max(float(np.mean(w_meta_h)), 1e-12)  # final mean=1
                # Guard n_eff: clip extreme weights so n_eff >= 30% of N
                _n_eff = float(np.sum(w_meta_h) ** 2 / max(np.sum(w_meta_h ** 2), 1e-12))
                if _n_eff < 0.3 * len(w_meta_h):
                    _clip_hi = float(np.percentile(w_meta_h, 95))
                    w_meta_h = np.clip(w_meta_h, 0.0, _clip_hi)
                    w_meta_h = w_meta_h / max(float(np.mean(w_meta_h)), 1e-12)
                    _n_eff_new = float(np.sum(w_meta_h) ** 2 / max(np.sum(w_meta_h ** 2), 1e-12))
                    tprint(f"    {_h_label} n_eff clipped: {_n_eff:.0f} -> {_n_eff_new:.0f} (N={len(w_meta_h)})")

                if bool(cfg.get("sample_weight_opt_enable", True)) and "__ts__" in df.columns:
                    _meta_ts = pd.to_datetime(df["__ts__"]) 
                    _meta_label_times = pd.DataFrame({
                        "t_start": _meta_ts,
                        "t_end": _meta_ts + pd.Timedelta(hours=int(_h)),
                    })
                    _meta_extra = {
                        "magnitude": w_mag,
                        "excursion": w_exc,
                    }
                    if _vol_proxy is not None and len(_vol_proxy) == len(w_meta_h):
                        _meta_extra["vol_cs"] = compute_vol_weights(_vol_proxy, _meta_ts.values)
                    w_meta_h = _optimize_training_sample_weights(
                        df=pd.DataFrame({"ts": _meta_ts}),
                        X_frame=X_meta_base.select_dtypes(include=[np.number]).fillna(0.0),
                        y_ret=y_ret_raw_h,
                        label_times=_meta_label_times,
                        base_weights=w_meta_h,
                        cfg={
                            **cfg,
                            "sample_weight_opt_trials": int(cfg.get("meta_sample_weight_opt_trials", cfg.get("sample_weight_opt_trials", 16))),
                        },
                        stage=f"meta_reg_{_h_label}",
                        extra_components=_meta_extra,
                    )
                w_meta_h = w_meta_h.astype(np.float32)

                # Target race for this horizon
                tprint(f"  Running target race for {_h_label} ({_time.monotonic()-_t0_meta:.1f}s)...")
                _tgt_name, y_target_h, _tgt_log = _run_target_race(
                    X_meta_base.to_numpy(dtype=np.float32), y_ret_raw_h, _vol_proxy, w_meta_h, _h_label)
                for _ll in _tgt_log:
                    tprint(_ll)
                _yt_fin = y_target_h[np.isfinite(y_target_h)]
                tprint(f"  Winning target '{_tgt_name}': n={len(y_target_h)}, "
                       f"mean={np.mean(_yt_fin):.6f}, std={np.std(_yt_fin):.6f}")

                # Fit MetaModel for this horizon
                meta_h = MetaModel()
                meta_h.strategy_name = _h_label
                tprint(f"  Fitting MetaModel {_h_label} (n={len(df)}, feats={X_meta_base.shape[1]}) ({_time.monotonic()-_t0_meta:.1f}s)...")
                meta_h.fit(X_meta_base, y_target_h, sample_weight=w_meta_h, groups=meta_groups,
                           y_per_horizon=_y_per_h)
                meta_models[_h_label] = meta_h
                _bucket_y_ret[_h_label] = y_ret_raw_h.copy()
                tprint(f"Meta {_h_label}: fitted ({_time.monotonic()-_t0_meta:.1f}s).")

                # Orientation safeguard for MR buckets
                if meta_h.oof_probs is not None:
                    y_ret_filtered = df["__y_ret__"].values if "__y_ret__" in df.columns else y_target_h

                    def _top_spread(yv, sv, frac=0.10):
                        n = len(yv)
                        if n <= 2:
                            return 0.0
                        ksel = max(1, int(frac * n))
                        it = np.argsort(sv)[-ksel:]
                        ib = np.argsort(sv)[:ksel]
                        return float(np.mean(yv[it]) - np.mean(yv[ib]))

                    pred_oof = np.asarray(meta_h.oof_probs, dtype=float)
                    ic_pos = _safe_spearman(pred_oof, y_ret_filtered)
                    ic_neg = _safe_spearman(-pred_oof, y_ret_filtered)
                    sp_pos = _top_spread(y_ret_filtered, pred_oof, frac=0.10)
                    sp_neg = _top_spread(y_ret_filtered, -pred_oof, frac=0.10)

                    meta_h.score_sign = 1
                    if k == "mr" and ((ic_neg > ic_pos + 1e-4) and (sp_neg > sp_pos + 1e-6)):
                        meta_h.score_sign = -1
                        tprint(f"Meta {_h_label}: orientation flipped (IC {ic_pos:.4f}->{ic_neg:.4f})")

                    pred_for_gate = meta_h.score_sign * pred_oof
                    gate_type = "meta_regression"
                    gate_res = compute_stage_gate_metrics(y_target_h, pred_for_gate, y_ret_filtered, model_type=gate_type)
                    gate_res["Model"] = _h_label
                    gate_res["Model_Type"] = gate_type
                    gate_res["Score_Sign"] = int(meta_h.score_sign)
                    gate_res["IC_Pos"] = float(ic_pos)
                    gate_res["IC_Neg"] = float(ic_neg)
                    gate_res["Spread10_Pos"] = float(sp_pos)
                    gate_res["Spread10_Neg"] = float(sp_neg)
                    meta_gate_results.append(gate_res)

            # ══════════════════════════════════════════════════════════════
            # CLASSIFIER TRAINING (single per bucket, uses all horizons)
            # ══════════════════════════════════════════════════════════════
            # Magnitude sigmoid: moderate top-30% upweight (same alpha as regressors)
            # Each source normalized to mean=1 before combining so neither dominates.
            _alpha_clf = float(cfg.get("meta_weight_sigmoid_alpha", 0.5))
            _y_avg_abs = np.mean([np.abs(_y_per_h[h]) for h in _available_horizons], axis=0)
            _fin_w = np.isfinite(_y_avg_abs)
            _q70_c = float(np.percentile(_y_avg_abs[_fin_w], 70))
            _s_c = max(float(np.std(_y_avg_abs[_fin_w])), 1e-9)
            w_mag_clf = 1.0 + _alpha_clf * _sigmoid((_y_avg_abs - _q70_c) / _s_c)
            w_mag_clf = w_mag_clf / max(float(np.mean(w_mag_clf)), 1e-12)  # normalize to mean=1

            # MFE/MAE quality weighting for classifier (average across horizons)
            _bp_clf = df["__barrier_pct__"].values if "__barrier_pct__" in df.columns else None
            w_exc_clf = np.ones(len(df), dtype=np.float64)
            if _bp_clf is not None:
                _exc_parts = []
                for _hc in _available_horizons:
                    _mfe_col_c = f"__meta_raw__mfe_{_hc}h"
                    _mae_col_c = f"__meta_raw__mae_{_hc}h"
                    if _mfe_col_c in df.columns and _mae_col_c in df.columns:
                        _mfe_vc = np.nan_to_num(df[_mfe_col_c].values, nan=0.0).astype(np.float64)
                        _mae_vc = np.nan_to_num(df[_mae_col_c].values, nan=0.0).astype(np.float64)
                        _bp_vc = np.clip(_bp_clf.astype(np.float64), 1e-6, None)
                        _exc_parts.append(np.maximum(np.abs(_mfe_vc) / _bp_vc, np.abs(_mae_vc) / _bp_vc))
                if _exc_parts:
                    _d_exc_clf = np.mean(_exc_parts, axis=0)
                    _tau_exc_clf = float(cfg.get("meta_mfe_mae_tau", 1.0))
                    w_exc_clf = 0.5 + 0.5 * np.clip(_d_exc_clf / _tau_exc_clf, 0.0, 1.0)
            w_exc_clf = w_exc_clf / max(float(np.mean(w_exc_clf)), 1e-12)  # normalize to mean=1

            w_meta_clf = (w_mag_clf * w_exc_clf).astype(np.float64)
            w_meta_clf = w_meta_clf / max(float(np.mean(w_meta_clf)), 1e-12)  # final mean=1
            # Guard n_eff: clip extreme weights so n_eff >= 30% of N
            _n_eff_clf = float(np.sum(w_meta_clf) ** 2 / max(np.sum(w_meta_clf ** 2), 1e-12))
            if _n_eff_clf < 0.3 * len(w_meta_clf):
                _clip_hi_clf = float(np.percentile(w_meta_clf, 95))
                w_meta_clf = np.clip(w_meta_clf, 0.0, _clip_hi_clf)
                w_meta_clf = w_meta_clf / max(float(np.mean(w_meta_clf)), 1e-12)
                _n_eff_clf_new = float(np.sum(w_meta_clf) ** 2 / max(np.sum(w_meta_clf ** 2), 1e-12))
                tprint(f"    {side}_{k}_clf n_eff clipped: {_n_eff_clf:.0f} -> {_n_eff_clf_new:.0f} (N={len(w_meta_clf)})")

            if bool(cfg.get("sample_weight_opt_enable", True)) and "__ts__" in df.columns:
                _meta_ts_c = pd.to_datetime(df["__ts__"]) 
                _mid_h_c = 4 if 4 in _available_horizons else _available_horizons[len(_available_horizons)//2]
                _meta_label_times_c = pd.DataFrame({
                    "t_start": _meta_ts_c,
                    "t_end": _meta_ts_c + pd.Timedelta(hours=int(_mid_h_c)),
                })
                w_meta_clf = _optimize_training_sample_weights(
                    df=pd.DataFrame({"ts": _meta_ts_c}),
                    X_frame=X_meta_base.select_dtypes(include=[np.number]).fillna(0.0),
                    y_ret=_y_per_h[_mid_h_c].astype(np.float64),
                    label_times=_meta_label_times_c,
                    base_weights=w_meta_clf,
                    cfg={
                        **cfg,
                        "sample_weight_opt_trials": int(cfg.get("meta_sample_weight_opt_trials", cfg.get("sample_weight_opt_trials", 16))),
                    },
                    stage=f"meta_clf_{side}_{k}",
                    extra_components={"magnitude": w_mag_clf, "excursion": w_exc_clf},
                )
            w_meta_clf = w_meta_clf.astype(np.float32)

            # Use the middle horizon's target for the classifier's y_ret argument
            _mid_h = 4 if 4 in _y_per_h else _available_horizons[len(_available_horizons)//2]
            y_target_clf = _y_per_h[_mid_h].astype(np.float64)

            tprint(f"  Fitting MetaClassifierModel {side}_{k} ({_time.monotonic()-_t0_meta:.1f}s)...")
            meta_clf = MetaClassifierModel()
            meta_clf.strategy_name = f"{side}_{k}"
            meta_clf.fit(X_meta_base, y_target_clf, sample_weight=w_meta_clf, groups=meta_groups,
                         y_per_horizon=_y_per_h)
            meta_models[f"{side}_{k}_clf"] = meta_clf
            _bucket_y_ret[f"{side}_{k}_clf"] = y_target_clf.copy()
            tprint(f"Meta {side}_{k}_clf: fitted ({_time.monotonic()-_t0_meta:.1f}s).")

    # ── Comprehensive per-model summary table (global + top-30%) ──
    tprint(f"\n{'═'*170}")
    tprint(f"  META MODEL SUMMARY TABLE (Global + Top-30%)")
    tprint(f"{'═'*170}")
    _tbl_hdr = (f"  {'Model':22s} {'Type':5s} {'Winner':18s} {'IC_g':>7s} "
                f"{'IC_t30':>7s} {'Lift@30':>7s} {'Net_t30':>9s} "
                f"{'Shrp_t30':>8s} {'Sort_t30':>8s} {'Spr10v1':>9s} "
                f"{'Turnover':>8s} {'Stabil':>7s}")
    tprint(_tbl_hdr)
    tprint(f"  {'─'*168}")
    for key, meta_obj in meta_models.items():
        if not hasattr(meta_obj, 'oof_probs') or meta_obj.oof_probs is None:
            continue
        is_clf = key.endswith("_clf")
        _mtype = "clf" if is_clf else "reg"
        _winner_name = getattr(meta_obj, 'model', {}).get('name', '?') if hasattr(meta_obj, 'model') and isinstance(getattr(meta_obj, 'model', None), dict) else '?'
        if _winner_name and len(_winner_name) > 18:
            _winner_name = _winner_name[:18]
        _bret_key = key
        if _bret_key in _bucket_y_ret:
            _bret = _bucket_y_ret[_bret_key]
            if len(_bret) == len(meta_obj.oof_probs):
                _dm = _detailed_oof_metrics(meta_obj.oof_probs, _bret)
                tprint(
                    f"  {key:22s} {_mtype:5s} {_winner_name:18s} "
                    f"{_dm.get('IC_global',0):>7.4f} {_dm.get('IC_top30',0):>7.4f} "
                    f"{_dm.get('Lift@30',0):>7.3f} {_dm.get('Mean_net_t30',0):>9.6f} "
                    f"{_dm.get('Sharpe_t30',0):>8.3f} {_dm.get('Sortino_t30',0):>8.3f} "
                    f"{_dm.get('Spread_10v1',0):>9.6f} "
                    f"{_dm.get('Turnover',0):>8.3f} {_dm.get('Stability',0):>7.2f}")
    tprint(f"{'═'*170}\n")

    # Save meta OOF predictions for ridge_position_sizer
    # Include trade context (return, direction) for position sizing
    _run_id = cfg.get("run_id", "default")
    meta_oof_dir = os.path.join(cfg.get("data_root", "data"), "artifacts", _run_id, "meta_oof")
    os.makedirs(meta_oof_dir, exist_ok=True)
    
    for key, meta in meta_models.items():
        if hasattr(meta, 'oof_probs') and meta.oof_probs is not None:
            # Parse side from key (e.g., "long_mr_H2" -> "long", "short_tf_clf" -> "short")
            parts = key.split("_")
            side_parsed = parts[0] if parts else "long"
            is_long = 1 if side_parsed == "long" else 0
            
            meta_oof_path = os.path.join(meta_oof_dir, f"meta_oof_{key}.parquet")
            oof_df = pd.DataFrame({
                "oof_pred": meta.oof_probs,
                "index": range(len(meta.oof_probs)),
                "is_long": is_long,
            })
            
            # Attach raw returns from per-bucket storage (key stored directly)
            if key in _bucket_y_ret:
                _bret = _bucket_y_ret[key]
                if len(_bret) == len(meta.oof_probs):
                    oof_df["return"] = _bret
            
            oof_df.to_parquet(meta_oof_path, index=False)
            tprint(f"Saved meta OOF predictions for {key} to {meta_oof_path}")
    
    tprint(f"train_meta_models_from_artifacts: done ({_time.monotonic()-_t0_meta:.1f}s), {len(meta_models)} meta models")
    return meta_models, meta_gate_results

def train_models_from_artifacts(datasets, cfg, train_meta=True):
    tprint(f"Entering function: train_models_from_artifacts in training.py")
    cfg = _resolve_training_cfg_with_offline_optimisers(cfg)
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
            per_h_models = {}
            feature_selection_by_h = {}
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

                # Explicit strategy context for logging:
                # trade_side (long/short) is separate from move_bucket (up/down).
                trade_side = side
                cand_filter, move_bucket, strategy_label = _strategy_bucket_context(trade_side, k)
                cand_filter_pretty = "top_ret" if cand_filter == "best" else "bottom_ret"

                tprint(
                    f"Training {side}_{k} [{key}] trade_side={trade_side} "
                    f"move_bucket={move_bucket} candidate_bucket={cand_filter_pretty} "
                    f"strategy={strategy_label} H={H} (n={len(X)})..."
                )

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
                feature_selection_by_h[H] = list(selected_feats)
                tprint(f"MDI selected {len(selected_feats)} features (from {X.shape[1]}) for H={H}.")
                
                X_sel = X[selected_feats]
                cols = list(selected_feats)
                
                y_hard_check = (y >= 0.5).astype(int)
                tprint(f"  Class dist: 0={int((y_hard_check==0).sum())} ({(y_hard_check==0).mean()*100:.1f}%), "
                       f"1={int((y_hard_check==1).sum())} ({(y_hard_check==1).mean()*100:.1f}%)")

                race = ModelRace(kind=k, n_splits=3)
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
                    p30 = precision_at_k(y_bin_oof, oof, 0.30, groups=g)
                    tpd10 = _avg_trades_per_day_global(oof, 0.10, g)
                    tpd30 = _avg_trades_per_day_global(oof, 0.30, g)
                    tprint(
                        f"  AlphaPrec@10={p10:.4f} AlphaPrec@30={p30:.4f} "
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
                    alpha_diag["avg_trades_day_10"] = float(_avg_trades_per_day_global(race.oof_probs, 0.10, groups_v))
                    alpha_diag["avg_trades_day_30"] = float(_avg_trades_per_day_global(race.oof_probs, 0.30, groups_v))
                    oof_metrics = _aggregate_alpha_oof_metrics(y, race.oof_probs, y_ret, sample_weight=w, groups=groups_v)
                    alpha_diag.update(oof_metrics)

                if score > best_ic:
                    best_ic = score
                    best_m = {"model": race, "H": H, "feat_cols": cols, "per_regime": per_regime, "alpha_diag": alpha_diag}

                per_h_models[H] = {
                    "model": race,
                    "H": H,
                    "feat_cols": cols,
                    "per_regime": per_regime,
                    "alpha_diag": alpha_diag,
                    "score": score,
                }

            # --- Multi-horizon deployment: all horizons are kept ---
            # best_m serves as the "primary" (highest score) for backward compat.
            # models_by_h stores all trained horizons for inference averaging.
            if best_m is not None:
                best_m["models_by_h"] = {h: {"model": v["model"], "feat_cols": v["feat_cols"], "H": v["H"], "selected_features": feature_selection_by_h.get(h, v["feat_cols"])} for h, v in per_h_models.items()}
                tprint(f"  {side}_{k}: Deploying {len(per_h_models)} horizons: {sorted(per_h_models.keys())} (primary H={best_m['H']})")

            # --- Save OOF predictions as lightweight parquet for fast meta loading ---
            _run_id = cfg.get("run_id", "default")
            oof_dir = os.path.join(cfg["data_root"], "artifacts", _run_id, "oof")
            os.makedirs(oof_dir, exist_ok=True)
            for _h, _v in per_h_models.items():
                _race = _v["model"]
                if _race.oof_probs is not None:
                    _oof_path = os.path.join(oof_dir, f"oof_{side}_{k}_H{_h}.parquet")
                    pd.DataFrame({"oof_prob": _race.oof_probs}).to_parquet(_oof_path, index=False)

            # --- Save each ModelRace in native format (fast load) ---
            models_dir = os.path.join(cfg["data_root"], "artifacts", _run_id, "models", "native")
            for _h, _v in per_h_models.items():
                _race = _v["model"]
                _model_dir = os.path.join(models_dir, f"{side}_{k}_H{_h}")
                _race.save_native(_model_dir)
                import json as _json
                with open(os.path.join(_model_dir, "columns.json"), "w") as _cf:
                    _json.dump({"feat_cols": _v.get("feat_cols", []), "selected_features": _v.get("selected_features", _v.get("feat_cols", []))}, _cf)
            # Strip for pickle fallback
            for _h, _v in per_h_models.items():
                _v["model"].strip_for_serialization()

            # --- Stage Gate Check (Alpha) — per horizon ---
            for _gate_H, _gate_v in per_h_models.items():
                _gate_race = _gate_v["model"]
                _gate_key = f"train_{side}_{k}_{_gate_H}"
                if _gate_key not in datasets or _gate_race.oof_probs is None:
                    continue
                _gate_df = datasets[_gate_key]
                _gate_y = _gate_df["__y_bin__"].values
                _gate_yret = _gate_df["__y_ret__"].values
                _gate_oof = _gate_race.oof_probs
                if len(_gate_oof) != len(_gate_y):
                    continue
                # Bootstrap CV(Prec@20)
                _yb_hard = (_gate_y >= 0.5).astype(np.float64)
                _rng = np.random.RandomState(42)
                _prec_samples = []
                _n = len(_yb_hard)
                _kf = 0.20
                for _ in range(50):
                    _ib = _rng.choice(_n, size=_n, replace=True)
                    _nk = max(1, int(_n * _kf))
                    top_idx = np.argsort(_gate_oof[_ib])[-_nk:]
                    p_k_b = float(np.mean(_yb_hard[_ib][top_idx]))
                    _prec_samples.append(p_k_b)
                _pa = np.array(_prec_samples)
                _cv20 = float(np.std(_pa) / (np.mean(_pa) + 1e-9))
                tprint(f"  {side}_{k} H={_gate_H}: Bootstrap CV(Prec@20)={_cv20:.3f}")

                gate_res = compute_stage_gate_metrics(
                    _gate_y, _gate_oof, _gate_yret,
                    model_type="classifier",
                    cv_prec10=_cv20
                )
                gate_res["Model"] = f"{side}_{k}_H{_gate_H}"
                alpha_gate_results.append(gate_res)

            final_models[side][k] = best_m

    # Save base models intermediate for train_meta mode
    _run_id = cfg.get("run_id", "default")
    _intermediate_path = os.path.join(cfg["data_root"], "artifacts", _run_id, "base_models_intermediate.pkl")
    os.makedirs(os.path.dirname(_intermediate_path), exist_ok=True)
    import pickle as _pkl_save
    _base_bundle = {
        "alpha_models": final_models,
        "spike_models": spike_models,
        "specialist_models": specialist_models,
    }
    with open(_intermediate_path, "wb") as _f:
        _pkl_save.dump(_base_bundle, _f)
    tprint(f"Base models intermediate saved to {_intermediate_path}")

    # 3. Train Meta Models (One per Alpha Model: Side x Kind)
    if train_meta:
        meta_models, meta_gate_results = train_meta_models_from_artifacts(datasets, cfg, final_models)
    else:
        meta_models, meta_gate_results = {}, []

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

    n_alpha_models = len(alpha_gate_results) if alpha_gate_results else 0
    alpha_half = max(1, n_alpha_models // 2)
    tprint(f"\nAlpha Stage: {alpha_pass_count}/{n_alpha_models} passed (need {alpha_half}).")
    if alpha_pass_count < alpha_half:
        tprint(f"WARNING: Alpha Stage FAILED (< {alpha_half} models passed).")

    tprint("\n=== Stage Gate Report: Meta Models (Quantile) ===")
    meta_pass_count = 0
    if meta_gate_results:
        df_meta_gate = pd.DataFrame(meta_gate_results)
        cols_order = ["Model", "passed", "Coverage_Diff", "Pinball_Imp", "Spearman_IC", "Pass_Spread", "Pass_Downside"]
        tprint(df_meta_gate[ [c for c in cols_order if c in df_meta_gate.columns] ].to_string(index=False))
        meta_pass_count = df_meta_gate["passed"].sum()
    else:
        tprint("No Meta models evaluated.")

    n_meta_models = len(meta_gate_results) if meta_gate_results else 0
    meta_half = max(1, n_meta_models // 2)
    tprint(f"\nMeta Stage: {meta_pass_count}/{n_meta_models} passed (need {meta_half}).")
    if meta_pass_count < meta_half:
        tprint(f"WARNING: Meta Stage FAILED (< {meta_half} models passed).")

    # Extended per-model quality report (base + meta) — per horizon per bucket.
    base_quality_rows = []
    for side in trade_sides:
        for kind in kinds:
            conf = final_models.get(side, {}).get(kind)
            if not conf:
                continue
            models_by_h = conf.get("models_by_h", {})
            if not models_by_h:
                # Fallback: single-model legacy format
                models_by_h = {conf.get("H", 4): {"model": conf["model"], "feat_cols": conf["feat_cols"]}}
            for H_rep, h_info in models_by_h.items():
                ds_key = f"train_{side}_{kind}_{H_rep}"
                if ds_key not in datasets:
                    continue
                race = h_info["model"]
                if race.oof_probs is None:
                    continue
                dfm = datasets[ds_key]
                y_bin = (dfm["__y_bin__"].values >= 0.5).astype(int)
                y_ret = dfm["__y_ret__"].values.astype(float)
                y_lbl = dfm["__y_lbl__"].values.astype(int) if "__y_lbl__" in dfm.columns else None
                groups = dfm["__ts__"].values if "__ts__" in dfm.columns else None
                for cand_name, dm in race.detailed_metrics.items():
                    # Use per-model OOF predictions from detailed_metrics, not the winner's OOF
                    oof_probs = dm.get("oof_probs")
                    if oof_probs is None:
                        # Fallback to winner's OOF if per-model not available (legacy)
                        oof_probs = np.asarray(race.oof_probs, dtype=float)
                    else:
                        oof_probs = np.asarray(oof_probs, dtype=float)
                    n = min(len(y_bin), len(oof_probs), len(y_ret))
                    y_bin_model, oof_probs_model, y_ret_model = y_bin[:n], oof_probs[:n], y_ret[:n]
                    y_lbl_model = y_lbl[:n] if y_lbl is not None else None
                    if groups is not None:
                        groups_model = np.asarray(groups)[:n]
                    else:
                        groups_model = None
                    entry = _base_model_report_entry(
                        model_name=f"{side}_{kind}_H{H_rep}:{cand_name}",
                        side=side,
                        kind=kind,
                        dm=dm,
                        y_bin=y_bin_model,
                        oof_probs=oof_probs_model,
                        y_ret=y_ret_model,
                        groups=groups_model,
                        y_lbl=y_lbl_model,
                    )
                    entry["H"] = H_rep
                    entry["is_winner"] = (cand_name == race.best_model_name)
                    base_quality_rows.append(entry)

    meta_quality_rows = []
    for key, meta in meta_models.items():
        if not key or "_" not in key:
            continue

        # Handle classifier keys like "long_mr_clf" vs regressor keys like "long_mr"
        is_clf = key.endswith("_clf")
        base_key = key[:-4] if is_clf else key  # strip "_clf" suffix
        parts = base_key.split("_", 1)
        if len(parts) != 2:
            continue
        side, kind = parts

        # For classifier models, build a simplified report entry from their report_rows
        if is_clf and isinstance(meta, MetaClassifierModel):
            # Find the best record from the classifier race
            best_rec = None
            for rec in getattr(meta, "report_rows", []):
                if best_rec is None or rec.get("composite_score", 0) > best_rec.get("composite_score", 0):
                    best_rec = rec
            clf_metrics = {}
            if best_rec:
                clf_metrics = {
                    "clf_winner": best_rec.get("model", "?"),
                    "clf_threshold_pct": best_rec.get("threshold_pct", 0),
                    "clf_pr_auc": best_rec.get("pr_auc", 0),
                    "clf_lift_26": best_rec.get("lift_26", 1.0),
                    "clf_sortino": best_rec.get("sortino", 0),
                    "clf_pnl_total_bps": best_rec.get("pnl_total_bps", 0),
                    "clf_max_dd_bps": best_rec.get("max_dd_bps", 0),
                    "clf_win_rate": best_rec.get("win_rate", 0),
                    "clf_avg_trades_day": best_rec.get("avg_trades_day", 0),
                    "clf_prec_13": best_rec.get("prec_13", 0),
                    "clf_prec_26": best_rec.get("prec_26", 0),
                    "clf_prec_39": best_rec.get("prec_39", 0),
                    "clf_ic_stability_mean": best_rec.get("ic_stability_mean", 0),
                }
            meta_quality_rows.append({
                "model": key, "passed": True, "metrics": clf_metrics,
            })
            continue

        conf = final_models.get(side, {}).get(kind)
        if not conf:
            continue
        models_by_h = conf.get("models_by_h", {})
        # Collect available horizon OOFs (same logic as train_meta)
        _h_oofs = {}
        for h in [2, 4, 8]:
            ds_key = f"train_{side}_{kind}_{h}"
            if ds_key not in datasets:
                continue
            race_h = models_by_h.get(h, {}).get("model") if h in models_by_h else None
            if race_h is None or race_h.oof_probs is None:
                continue
            df_h = datasets[ds_key]
            oof_h = np.asarray(race_h.oof_probs, dtype=float)
            if len(oof_h) == len(df_h):
                _h_oofs[h] = (df_h, oof_h)
        if not _h_oofs:
            continue
        # Use largest-H dataset as base, average OOF across horizons
        _base_H = max(_h_oofs.keys(), key=lambda h: len(_h_oofs[h][0]))
        dfm = _h_oofs[_base_H][0]
        y_ret = dfm["__y_ret__"].values.astype(float)
        # Average base_score from all horizon OOFs (truncated to base length)
        _oof_parts = []
        for h, (df_h, oof_h) in _h_oofs.items():
            _oof_parts.append(oof_h[:len(dfm)] if len(oof_h) >= len(dfm) else np.pad(oof_h, (0, len(dfm) - len(oof_h)), constant_values=0.5))
        base_score = np.mean(_oof_parts, axis=0)
        def _aligned_ret(h):
            """Get __y_ret__ for horizon h, aligned to len(dfm)."""
            k = f"train_{side}_{kind}_{h}"
            if k not in datasets:
                return y_ret.copy()
            arr = datasets[k]["__y_ret__"].values.astype(float)
            if len(arr) >= len(dfm):
                return arr[:len(dfm)]
            return np.pad(arr, (0, len(dfm) - len(arr)), constant_values=0.0)
        _r2_rpt, _r4_rpt, _r8_rpt = _aligned_ret(2), _aligned_ret(4), _aligned_ret(8)
        y_target = compute_meta_target(_r2_rpt, _r4_rpt, _r8_rpt, groups=None)
        _y_per_h_rpt = {2: _r2_rpt, 4: _r4_rpt, 8: _r8_rpt}
        groups = dfm["__ts__"].values if "__ts__" in dfm.columns else None
        n = min(len(y_ret), len(y_target), len(base_score), len(meta.oof_probs) if meta.oof_probs is not None else 0)
        if n <= 10:
            continue
        y_ret, y_target, base_score = y_ret[:n], y_target[:n], base_score[:n]
        _y_per_h_rpt = {h: v[:n] for h, v in _y_per_h_rpt.items()}
        if groups is not None:
            groups = np.asarray(groups)[:n]
        meta_quality_rows.append(_meta_report_entry(key, meta, y_target, y_ret, base_score, groups,
                                                    y_per_horizon=_y_per_h_rpt))

    winners_base = sorted([r for r in base_quality_rows if r.get("is_winner")], key=lambda x: x.get("score", -1e9), reverse=True)
    others_base = sorted([r for r in base_quality_rows if not r.get("is_winner")], key=lambda x: x.get("score", -1e9), reverse=True)
    winners_meta = sorted([r for r in meta_quality_rows if r.get("passed")], key=lambda x: x.get("metrics", {}).get("spearman_ic", -1e9), reverse=True)
    others_meta = sorted([r for r in meta_quality_rows if not r.get("passed")], key=lambda x: x.get("metrics", {}).get("spearman_ic", -1e9), reverse=True)

    gate_report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_models": winners_base + others_base,
        "meta_models": winners_meta + others_meta,
        "winners": {
            "base": [r["model"] for r in winners_base],
            "meta": [r["model"] for r in winners_meta],
        },
    }
    print_training_gate_report(gate_report)

    return {
        "alpha_models": final_models,
        "alpha_oof_metrics": {f"{side}_{kind}": final_models.get(side, {}).get(kind, {}).get("alpha_diag", {}) for side in trade_sides for kind in kinds},
        "exh_models": exh_models,
        "meta_models": meta_models,
        "spike_models": spike_models,
        "specialist_models": specialist_models,
        "quality_gate_report": gate_report,
    }


def select_best_horizon(panel, feats, mkt_gates, cfg, syms, ts):
    # DEPRECATED / LEGACY WRAPPER
    # This function is kept for backward compatibility if needed,
    # but strictly we should use the new split.
    # We can implement it by calling the new functions in sequence.
    datasets = generate_label_datasets(panel, feats, mkt_gates, cfg, syms, ts, None)
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
            trade_side = side
            cand_filter, move_bucket, strategy_label = _strategy_bucket_context(trade_side, k)
            trend_filter = move_bucket

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
                trend_vals = feats["trend_pct"].reindex(columns=cands).loc[t] if t in feats["trend_pct"].index else pd.Series(0.0, index=cands)

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
                tprint(
                    f"Bucket trade_side={trade_side} kind={k} move_bucket={move_bucket} "
                    f"candidate_bucket={cand_filter} strategy={strategy_label}: {len(indices)} events | "
                    f"Mean Barrier%={mean_barrier*100:.2f} | Mean VolZ={mean_z:.2f}"
                )
            else:
                tprint(
                    f"Bucket trade_side={trade_side} kind={k} move_bucket={move_bucket} "
                    f"candidate_bucket={cand_filter} strategy={strategy_label}: 0 events"
                )

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
            min_ratio = float(cfg.get("min_tp_sl_ratio", 1.2))
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
