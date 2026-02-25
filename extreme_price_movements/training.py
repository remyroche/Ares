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
from .sample_weights import (
    build_label_time_ranges,
    compute_sample_weights_with_uniqueness,
    drawdown_aware_weights,
    compute_mfe_mae_weights,
    compute_cell_weights_neg_mass_renorm,
    NegMassRenormCfg,
)
from .sample_weight_optimization import (
    combine_weights_safely,
    compute_vol_weights,
    compute_liquidity_weights,
    compute_distance_to_barrier_weights,
    compute_recency_weights,
    sample_weight_tp_classifier,
    sample_weight_meta_regression,
    optimize_component_weights,
    log_weight_statistics,
    select_test_feature_frame,
)
from .offline_optimisers.params_store import apply_offline_optimizer_best_params, load_tbm_geometry_grid, CANDIDATE_BEST_PARAMS_CSV
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
from .barrier_geometry import make_effective_tp
from .policy_ml import (
    build_base_tp_vs_sl,
    load_best_policy_params_from_optimise,
    MetaClassifierSelectionConfig,
    policy_rollout_ml,
)
from .production_admissibility import ProdGates, production_admissibility_report
from .gate_metrics import compute_stage_gate_metrics

import os
import json
from typing import Optional
from datetime import datetime, timezone
from types import SimpleNamespace

# ... (omitting helper functions like _coerce_feature_to_panel_df to save space, assuming they exist above) ...
# =============================================================================
# UNIFIED BARRIER FACTORY - Canonical TP/SL geometry (best-of-both pipelines)
# =============================================================================
# ... (omitting _compute_dynamic_horizon_frame, _compute_barrier_base, compute_barrier_factory) ...
# Assuming user has context, I will insert the changed function directly.

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


def _compute_dynamic_horizon_frame(
    atr_pct: pd.DataFrame,
    base_horizon: float,
    cfg: dict,
    _base: dict | None = None,
) -> pd.DataFrame | None:
    if not bool(cfg.get("use_dynamic_horizon", False)):
        return None

    if _base is not None and "z_clipped" in _base:
        z = _base["z_clipped"]
    else:
        window = int(cfg.get("barrier_atr_window", 24 * 30))
        disp_floor = float(cfg.get("barrier_disp_floor", 0.1))
        z_max = float(cfg.get("barrier_z_max", 3.0))

        atr_median = atr_pct.rolling(window, min_periods=24).median()
        atr_mad = (atr_pct - atr_median).abs().rolling(window, min_periods=24).median()
        atr_disp = np.maximum(atr_mad, disp_floor * atr_median)
        z = np.clip((atr_pct - atr_median) / (atr_disp + 1e-12), -z_max, z_max)

    z_lo = float(cfg.get("dynamic_horizon_z_lo", -1.0))
    z_hi = float(cfg.get("dynamic_horizon_z_hi", 2.0))
    max_scale_add = float(cfg.get("dynamic_horizon_max_scale_add", 0.5))

    fraction = np.clip((z - z_lo) / (z_hi - z_lo + 1e-9), 0.0, 1.0)
    scale = 1.0 + max_scale_add * fraction

    return scale * float(base_horizon)


def _compute_barrier_base(
    atr_pct: pd.DataFrame,
    window_size: int,
    disp_floor: float,
    z_max: float,
    k_reg: float,
    m_lo: float,
    m_hi: float,
    sl_lo: float,
    sl_hi: float,
    z_gate: float,
    use_standalone_sl: bool = False,
) -> dict:
    atr_median = atr_pct.rolling(window_size, min_periods=24).median()
    atr_mad = (atr_pct - atr_median).abs().rolling(window_size, min_periods=24).median()
    atr_disp = np.maximum(atr_mad, disp_floor * atr_median)
    z_score = (atr_pct - atr_median) / (atr_disp + 1e-12)
    z_clipped = np.clip(z_score, -z_max, z_max)
    m_clipped = np.clip(np.exp(k_reg * z_clipped), m_lo, m_hi)
    z_norm = np.clip((z_clipped - z_gate) / (z_max - z_gate), 0, 1)
    sl_mult = sl_lo + (sl_hi - sl_lo) * z_norm
    return {"atr_median": atr_median, "z_clipped": z_clipped, "m_clipped": m_clipped, "sl_mult": sl_mult, "use_standalone_sl": use_standalone_sl}


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
    tp_lo: float = 0.02,
    tp_hi: float = 0.06,
    return_components: bool = False,
    _base: dict | None = None,
    use_standalone_sl: bool = False,
) -> tuple:
    if _base is not None:
        m_clipped = _base["m_clipped"]
        sl_mult   = _base["sl_mult"]
    else:
        atr_median = atr_pct.rolling(window_size, min_periods=24).median()
        atr_mad = (atr_pct - atr_median).abs().rolling(window_size, min_periods=24).median()
        atr_disp = np.maximum(atr_mad, disp_floor * atr_median)
        z_clipped = np.clip((atr_pct - atr_median) / (atr_disp + 1e-12), -z_max, z_max)
        m_clipped = np.clip(np.exp(k_reg * z_clipped), m_lo, m_hi)
        z_norm = np.clip((z_clipped - z_gate) / (z_max - z_gate), 0, 1)
        sl_mult = sl_lo + (sl_hi - sl_lo) * z_norm

    tp_raw = k_tp * atr_pct * m_clipped
    tp_vals = make_effective_tp(
        tp_raw,
        horizon=horizon,
        horizon_scaling="sqrt",
        lo=float(tp_lo),
        hi=float(tp_hi),
        horizon_alpha=0.5,
        horizon_base=float(H_base),
    )
    
    sl_raw_standalone = sl_base_mult * atr_pct * m_clipped
    sl_vals_standalone = make_effective_tp(
        sl_raw_standalone,
        horizon=horizon,
        horizon_scaling="sqrt",
        lo=float(sl_lo),
        hi=float(sl_hi),
        horizon_alpha=0.5,
        horizon_base=float(H_base),
    )
    
    sl_vals_compounded = sl_base_mult * sl_mult * tp_vals
    sl_vals = np.minimum(sl_vals_standalone, sl_vals_compounded)
    
    tp_df = pd.DataFrame(tp_vals, index=atr_pct.index, columns=atr_pct.columns)
    sl_df = pd.DataFrame(sl_vals, index=atr_pct.index, columns=atr_pct.columns)
    
    if return_components:
        asset_diagnostics = {}
        for col in atr_pct.columns:
            col_idx = atr_pct.columns.get_loc(col)
            m_vals = m_clipped.values[:, col_idx]
            sl_m = sl_mult.values[:, col_idx]
            atr_col = atr_pct[col].values
            tp_col = tp_vals.values[:, col_idx]
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
    try:
        return apply_offline_optimizer_best_params(cfg)
    except Exception as exc:
        tprint(f"Warning: failed to load offline optimiser params; using cfg defaults ({exc})")
        return cfg


def _build_optimal_candidate_mask(panel, feats, cfg):
    strict = bool(cfg.get("require_offline_candidate_ranges", True))
    cfg_resolved = _resolve_training_cfg_with_offline_optimisers(cfg)

    if strict and (not CANDIDATE_BEST_PARAMS_CSV.exists()):
        raise RuntimeError(
            "Candidate best-params CSV missing; strict range mode forbids generating events outside offline-optimal ranges. "
            f"Expected: {CANDIDATE_BEST_PARAMS_CSV}"
        )

    train_pct, train_min_range, train_min_vol = _get_training_candidate_config(cfg_resolved)
    sign_min = float(cfg_resolved.get("min_feat_sign_consistency", 0.80))

    tprint(
        "Building strict candidate mask from offline-optimal conditions: "
        f"pct={float(train_pct):.4f}, min_range_pct={float(train_min_range):.4f}, "
        f"min_vol_zscore={float(train_min_vol):.4f}, min_feat_sign_consistency={sign_min:.3f}, "
        f"strict={strict}"
    )
    cand_mask = select_trade_candidates_vectorized(
        panel,
        feats,
        pct=train_pct,
        metric=cfg_resolved["trade_deviation_metric"],
        min_range_pct=train_min_range,
        min_vol_zscore=train_min_vol,
        sign_consistency_min=sign_min,
    )
    return cand_mask, cfg_resolved


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
    # ... existing implementation ...
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
    from scipy.stats import rankdata
    from scipy.special import expit as _sigmoid

    n = len(y_ret_raw)
    y = np.asarray(y_ret_raw, dtype=np.float64)
    fin = np.isfinite(y)

    rk = np.full(n, 0.5, dtype=np.float64)
    if fin.sum() > 1:
        rk[fin] = (rankdata(y[fin]) - 1) / max(fin.sum() - 1, 1)

    _temp = 0.08
    t_soft30 = _sigmoid((rk - 0.70) / _temp)

    n_qbins = 20
    t_qbin = np.full(n, 0.5, dtype=np.float64)
    if fin.sum() > n_qbins:
        edges = np.percentile(y[fin], np.linspace(0, 100, n_qbins + 1))
        edges[0] -= 1e-12; edges[-1] += 1e-12
        bins = np.clip(np.digitize(y, edges) - 1, 0, n_qbins - 1)
        t_qbin = (bins + 0.5) / n_qbins

    t_tail = rk + 0.5 * np.maximum(0.0, rk - 0.70)

    bases = {"rank_pct": rk, "soft_top30": t_soft30, "qbin_mid": t_qbin, "tail_amp": t_tail}

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
        targets[bname] = bt.astype(np.float32)
        semi_scale = np.power(vp_med / np.clip(vp, 1e-9, None), 0.5) if has_vol else np.ones(n)
        semi_scale = np.clip(semi_scale, 0.5, 2.0)
        targets[f"{bname}_semivol"] = ((bt - 0.5) * semi_scale + 0.5).astype(np.float32)

    return targets


def _detailed_oof_metrics(oof, y_ret, cost=0.005, n_rolling=5):
    # ... existing implementation ...
    from scipy.stats import spearmanr
    s = np.asarray(oof, dtype=float)
    y = np.asarray(y_ret, dtype=float)
    m = np.isfinite(s) & np.isfinite(y)
    s, y = s[m], y[m]
    n = len(s)
    if n < 10:
        return {}
    ic_g = float(spearmanr(s, y).statistic) if n > 2 else 0.0
    rk = (pd.Series(s).rank(pct=True)).values
    dec = (rk * 10).astype(int).clip(0, 9)
    q_means = pd.Series(y).groupby(dec).mean()
    spread_10_1 = float(q_means.get(9, 0) - q_means.get(0, 0)) if len(q_means) >= 2 else 0.0
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
    downside = y_t30[y_t30 < 0]
    dd = float(np.sqrt(np.mean(downside**2))) if len(downside) > 0 else 1e-9
    sortino_t30 = mean_net_t30 / max(dd, 1e-9)
    y_rk = (pd.Series(y).rank(pct=True)).values
    true_t30 = y_rk >= 0.70
    precision = float((true_t30 & t30).sum()) / max(float(t30.sum()), 1)
    lift30 = precision / 0.30
    b30 = rk <= 0.30
    mean_ret_b30 = float(np.mean(y[b30])) if b30.sum() > 0 else 0.0
    spread_t30_b30 = mean_ret_t30 - mean_ret_b30

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

# ... (omitting _run_target_race, _build_bin_mono_metrics, _fold_stats_from_groups, _base_model_report_entry, _meta_report_entry, save_training_gate_report, print_training_gate_report, OUT_SL, OUT_TO, OUT_TP, AlphaHorizonEnsemble, compute_meta_target, build_horizon_prediction_features, _aggregate_alpha_oof_metrics, compute_per_regime_metrics, _fast_lookup, apply_interaction_toggles, _winsorize_and_unit_mean, _normalize_cross_sectional, _sigmoid, _compute_atr_scale, compute_weights_logic, _strategy_bucket_context, build_exhaustion_Xy, compute_p_exhaustion_at_t, _build_pred_X, generate_exhaustion_history, _build_pred_X_window, _optimize_training_sample_weights, build_hourly_training_set_and_weights, train_spike_anatomy_model, _get_bucket_label_config) ...
# To fit within the window, I will just apply the necessary changes to `train_meta_models_from_artifacts` and surrounding helpers if needed.
# The user wants to update `train_meta_models_from_artifacts`.

def _apply_dominance_constraints(w, y_class, sl_class=0, to_class=1):
    """Apply Spec B dominance constraints: SL > TO."""
    w = np.asarray(w, dtype=np.float64)
    is_sl = (y_class == sl_class)
    is_to = (y_class == to_class)
    
    # B2) Per-sample dominance constraint
    if is_sl.any() and is_to.any():
        w_sl = w[is_sl]
        w_to = w[is_to]
        
        # SL median should ideally be above TO 95th
        # But per-sample cap: TO weight capped at representative SL weight
        w_sl_rep = np.percentile(w_sl, 50)
        w[is_to] = np.minimum(w[is_to], w_sl_rep)
        
    # B3) Average-ratio constraint: mean(TO) >= mean(SL) / 10
    if is_sl.any() and is_to.any():
        mean_sl = np.mean(w[is_sl])
        mean_to = np.mean(w[is_to])
        if mean_to < mean_sl / 10.0 and mean_to > 1e-12:
            scale = (mean_sl / 10.0) / mean_to
            scale = min(scale, 3.0) # Cap scale
            w[is_to] *= scale
            
    # B4) Single global normalization
    w = w / max(np.mean(w), 1e-12)
    return w

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
    _aux_head_oof = {}  # per-bucket shared-fold auxiliary head OOF outputs

    # Load optimize policy params dynamically (for policy-aligned targets/selection context)
    _run_id_for_policy = str(cfg.get("run_id", ""))
    _policy_params_blob = load_best_policy_params_from_optimise(cfg.get("data_root", "../data"), _run_id_for_policy) if _run_id_for_policy else {}
    if _policy_params_blob:
        _n_buckets = len(_policy_params_blob.get("buckets", {})) if isinstance(_policy_params_blob, dict) else 0
        tprint(f"Meta training: loaded optimise policy params for run_id={_run_id_for_policy} (buckets={_n_buckets})")
    else:
        tprint("Meta training: optimise policy params not found for current run (using return-derived utility fallback).")

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

    def _train_aux_heads_shared_folds(
        X_num,
        y_u,
        y_mae,
        y_mfe,
        y_dur,
        trade_mask,
        cv_embargo_bars=12,
        bucket_id: str | None = None,
        data_root: str = "data",
        run_id: str = "default",
        dur_censored=None,
    ):
        """Train auxiliary meta heads with shared purged folds and return OOF preds.

        Heads: Utility(U), MAE(q70), MFE(q70), DUR.
        Missing labels are handled via placeholder target + zero sample_weight (no semantic fallback).
        """
        from .purged_cv import PurgedKFold as _PKF
        from sklearn.linear_model import HuberRegressor
        from sklearn.ensemble import ExtraTreesRegressor
        from sklearn.metrics import mean_absolute_error as _mae_loss

        try:
            from lightgbm import LGBMRegressor
        except Exception:
            LGBMRegressor = None
        try:
            from xgboost import XGBRegressor
        except Exception:
            XGBRegressor = None

        _Xdf_in = X_num if isinstance(X_num, pd.DataFrame) else pd.DataFrame(np.asarray(X_num, dtype=float), columns=[f"f_{i}" for i in range(np.asarray(X_num).shape[1])])
        Xv = np.asarray(_Xdf_in.values, dtype=float)
        n = len(Xv)
        tm = np.asarray(trade_mask, dtype=bool) if trade_mask is not None else np.ones(n, dtype=bool)
        tm = tm[:n]

        def _arr(a):
            z = np.asarray(a, dtype=float)
            if len(z) != n:
                out = np.full(n, np.nan, dtype=float)
                m = min(n, len(z))
                out[:m] = z[:m]
                return out
            return z

        y_u_raw = _arr(y_u)
        y_mae_raw = _arr(y_mae)
        y_mfe_raw = _arr(y_mfe)
        y_dur_raw = _arr(y_dur)
        dur_c = np.asarray(dur_censored, dtype=bool)[:n] if dur_censored is not None else np.zeros(n, dtype=bool)

        valid_u = np.isfinite(y_u_raw)
        valid_mae = np.isfinite(y_mae_raw)
        valid_mfe = np.isfinite(y_mfe_raw)
        valid_dur = np.isfinite(y_dur_raw)

        y_u_fit = np.where(valid_u, y_u_raw, 0.0)
        y_mae_fit = np.where(valid_mae, y_mae_raw, 0.0)
        y_mfe_fit = np.where(valid_mfe, y_mfe_raw, 0.0)
        y_dur_fit = np.where(valid_dur, y_dur_raw, 0.0)

        oof_u = np.full(n, np.nan, dtype=float)
        oof_mae_q70 = np.full(n, np.nan, dtype=float)
        oof_mfe = np.full(n, np.nan, dtype=float)
        oof_dur = np.full(n, np.nan, dtype=float)

        _pkf_shared = _PKF(n_splits=3, purge=int(cv_embargo_bars), embargo=int(cv_embargo_bars))
        _splits_shared = list(_pkf_shared.split(Xv))

        # keep selector simple/robust: all numeric features
        idx_u = np.arange(Xv.shape[1], dtype=int)
        idx_q = np.arange(Xv.shape[1], dtype=int)
        idx_mfe = np.arange(Xv.shape[1], dtype=int)
        idx_dur = np.arange(Xv.shape[1], dtype=int)

        def _normalize_clip_weights(w, lo=0.75, hi=1.25):
            w = np.asarray(w, dtype=float)
            w = np.where(np.isfinite(w), w, 0.0)
            w = np.clip(w, 0.0, None)
            pos = w > 0
            if np.any(pos):
                w[pos] = w[pos] / max(float(np.mean(w[pos])), 1e-12)
                w[pos] = np.clip(w[pos], lo, hi)
                w[pos] = w[pos] / max(float(np.mean(w[pos])), 1e-12)
            return w

        def _tail_multiplier(y_fit, w_base, tr_idx):
            w = np.asarray(w_base, dtype=float).copy()
            tr_pos = tr_idx[w[tr_idx] > 0]
            if len(tr_pos) >= 20:
                yt = y_fit[tr_pos]
                p50 = float(np.nanpercentile(yt, 50))
                p95 = float(np.nanpercentile(yt, 95))
                if np.isfinite(p50) and np.isfinite(p95) and p95 > p50:
                    mult = np.clip(y_fit, p50, p95)
                    mult = np.where(np.isfinite(mult), mult, p50)
                    mult = mult / max(p50, 1e-9)
                    w *= np.clip(mult, 0.75, 1.25)
            return _normalize_clip_weights(w)

        def _weighted_mae(y_true, y_pred, w):
            m = np.isfinite(y_true) & np.isfinite(y_pred) & (w > 0)
            if m.sum() == 0:
                return np.inf
            return float(np.sum(np.abs(y_true[m] - y_pred[m]) * w[m]) / max(np.sum(w[m]), 1e-12))

        _u_winners = []
        for tr, va in _splits_shared:
            tr_idx = tr[tm[tr]]
            va_idx = va[tm[va]]
            if len(tr_idx) < 50 or len(va_idx) == 0:
                continue

            Xu_tr = Xv[tr_idx]
            Xu_va = Xv[va_idx]

            # Utility head: weighted model race
            w_u = (valid_u.astype(float) * tm.astype(float))
            alpha_u = float(cfg.get("aux_u_weight_alpha", 0.7))
            mag = np.clip(np.abs(y_u_fit), 1e-9, None) ** alpha_u
            mag = np.where(np.isfinite(mag), mag, 1.0)
            w_u *= np.clip(mag, 0.75, 1.25)
            w_u = _normalize_clip_weights(w_u)
            w_u_tr = w_u[tr_idx]

            util_candidates = []
            util_candidates.append(("huber", HuberRegressor(epsilon=1.35, alpha=1e-3)))
            util_candidates.append(("et", ExtraTreesRegressor(n_estimators=200, max_depth=6, min_samples_leaf=20, random_state=42, n_jobs=2)))
            if XGBRegressor is not None:
                util_candidates.append(("xgb", XGBRegressor(
                    n_estimators=300,
                    max_depth=5,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.0,
                    reg_lambda=5.0,
                    random_state=42,
                    n_jobs=2,
                    num_parallel_tree=8,
                    objective="reg:squarederror",
                )))

            best_u_pred = None
            best_u_name = None
            best_u_score = np.inf
            for nm, mdl in util_candidates:
                try:
                    mdl.fit(Xu_tr[:, idx_u], y_u_fit[tr_idx], sample_weight=w_u_tr)
                    pred_va = mdl.predict(Xu_va[:, idx_u])
                    score = _weighted_mae(y_u_fit[va_idx], pred_va, w_u[va_idx])
                    if score < best_u_score:
                        best_u_score = score
                        best_u_pred = pred_va
                        best_u_name = nm
                except Exception:
                    continue
            if best_u_pred is not None:
                oof_u[va_idx] = best_u_pred
                _u_winners.append(best_u_name or "unknown")

            # MAE q70 head
            w_mae = _normalize_clip_weights(valid_mae.astype(float) * tm.astype(float))
            w_mae = _tail_multiplier(y_mae_fit, w_mae, tr_idx)
            w_mae_tr = w_mae[tr_idx]
            try:
                if LGBMRegressor is not None:
                    m_q = LGBMRegressor(
                        objective="quantile", alpha=float(cfg.get("aux_mae_quantile_alpha", 0.7)),
                        n_estimators=500, learning_rate=0.03, num_leaves=63, min_child_samples=50,
                        subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=2,
                    )
                    m_q.fit(
                        Xu_tr[:, idx_q], y_mae_fit[tr_idx], sample_weight=w_mae_tr,
                        eval_set=[(Xu_va[:, idx_q], y_mae_fit[va_idx])],
                        eval_sample_weight=[w_mae[va_idx]],
                    )
                else:
                    m_q = ExtraTreesRegressor(n_estimators=150, max_depth=5, min_samples_leaf=20, random_state=42, n_jobs=2)
                    m_q.fit(Xu_tr[:, idx_q], y_mae_fit[tr_idx], sample_weight=w_mae_tr)
                oof_mae_q70[va_idx] = m_q.predict(Xu_va[:, idx_q])
            except Exception:
                pass

            # MFE head: quantile objective (mirror MAE)
            w_mfe = _normalize_clip_weights(valid_mfe.astype(float) * tm.astype(float))
            w_mfe = _tail_multiplier(y_mfe_fit, w_mfe, tr_idx)
            w_mfe_tr = w_mfe[tr_idx]
            try:
                if LGBMRegressor is not None:
                    m_mfe = LGBMRegressor(
                        objective="quantile", alpha=float(cfg.get("aux_mfe_quantile_alpha", 0.75)),
                        n_estimators=500, learning_rate=0.03, num_leaves=63, min_child_samples=50,
                        subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=2,
                    )
                    m_mfe.fit(
                        Xu_tr[:, idx_mfe], y_mfe_fit[tr_idx], sample_weight=w_mfe_tr,
                        eval_set=[(Xu_va[:, idx_mfe], y_mfe_fit[va_idx])],
                        eval_sample_weight=[w_mfe[va_idx]],
                    )
                else:
                    m_mfe = ExtraTreesRegressor(n_estimators=150, max_depth=5, min_samples_leaf=20, random_state=42, n_jobs=2)
                    m_mfe.fit(Xu_tr[:, idx_mfe], y_mfe_fit[tr_idx], sample_weight=w_mfe_tr)
                oof_mfe[va_idx] = m_mfe.predict(Xu_va[:, idx_mfe])
            except Exception:
                pass

            # Duration head with censoring down-weight
            w_dur = valid_dur.astype(float) * tm.astype(float)
            if dur_c is not None and len(dur_c) == n:
                w_dur[dur_c] *= float(cfg.get("aux_dur_censor_weight", 0.25))
            w_dur = _normalize_clip_weights(w_dur)
            w_dur_tr = w_dur[tr_idx]
            try:
                m_dur = ExtraTreesRegressor(n_estimators=150, max_depth=5, min_samples_leaf=20, random_state=42, n_jobs=2)
                m_dur.fit(Xu_tr[:, idx_dur], y_dur_fit[tr_idx], sample_weight=w_dur_tr)
                oof_dur[va_idx] = m_dur.predict(Xu_va[:, idx_dur])
            except Exception:
                pass

        def _fill(oof, y_fit):
            o = np.asarray(oof, dtype=float)
            fill = float(np.nanmedian(y_fit[np.isfinite(y_fit)])) if np.isfinite(y_fit).any() else 0.0
            o = np.where(np.isfinite(o), o, fill)
            return o.astype(np.float32)

        _fs_report = {
            "u": {"n_in": int(_Xdf_in.shape[1]), "n_selected": int(len(idx_u))},
            "mae_q70": {"n_in": int(_Xdf_in.shape[1]), "n_selected": int(len(idx_q))},
            "mfe": {"n_in": int(_Xdf_in.shape[1]), "n_selected": int(len(idx_mfe))},
            "dur": {"n_in": int(_Xdf_in.shape[1]), "n_selected": int(len(idx_dur))},
            "config": {
                "weights": "per-head-validity+train-only-tail",
                "dur_censor_weight": float(cfg.get("aux_dur_censor_weight", 0.25)),
                "utility_race_winners": _u_winners,
            },
        }
        try:
            if bucket_id:
                _rp = os.path.join(data_root, "artifacts", run_id, "fs_reports", f"{bucket_id}_cap12")
                os.makedirs(_rp, exist_ok=True)
                import json as _json
                for _h in ["u", "mae_q70", "mfe", "dur"]:
                    with open(os.path.join(_rp, f"{_h}.json"), "w") as _f:
                        _json.dump(_fs_report[_h] | {"head": _h}, _f)
        except Exception as _e_fsrep:
            tprint(f"Warning: failed to persist fs report for {bucket_id}: {_e_fsrep}")

        return {
            "oof_u_hat": _fill(oof_u, y_u_fit),
            "oof_log_mae_q70_hat": _fill(oof_mae_q70, y_mae_fit),
            "oof_log_mfe_hat": _fill(oof_mfe, y_mfe_fit),
            "oof_log_dur_hat": _fill(oof_dur, y_dur_fit),
            "fs_report": _fs_report,
        }

    for side in trade_sides:
        for k in kinds:
            # ... (existing setup code for side/k) ...
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
            # ... (existing union logic) ...
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
            _use_policy_target = bool(cfg.get("meta_use_policy_value_target", True))

            # Use vol_proxy (ATR) for risk-normalized target / classifier fallback labels
            _vol_proxy = df["__barrier_pct__"].values.astype(np.float64) if "__barrier_pct__" in df.columns else None

            if _use_policy_target:
                if "__u_policy_net__" not in df.columns:
                    raise RuntimeError(
                        "meta_use_policy_value_target=True requires '__u_policy_net__' in training artifacts. "
                        "Enable policy_rollout_labeling during label generation and rebuild artifacts."
                    )
                y_target_h = np.asarray(df["__u_policy_net__"].values, dtype=np.float32)
                tprint(
                    f"  META TARGET: policy_value(u_policy) n={len(y_target_h)} "
                    f"mean={float(np.mean(y_target_h)):.6f} std={float(np.std(y_target_h)):.6f}"
                )
                # Keep horizon keys for per-horizon model slots, but target is true policy utility.
                _y_per_h = {h: y_target_h.copy() for h in sorted(horizon_dfs.keys())}
            else:
                _r2, _r4, _r8 = _ret_for_h_aligned(2), _ret_for_h_aligned(4), _ret_for_h_aligned(8)
                y_target_h = compute_meta_target(_r2, _r4, _r8, vol_proxy=_vol_proxy)
                tprint(f"  Using risk-normalized target: n={len(y_target_h)}, mean={float(np.mean(y_target_h)):.6f}, std={float(np.std(y_target_h)):.6f}")
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

            # Memory optimization: use single copy with shared index, avoid redundant copies
            keep_idx = np.where(keep)[0]
            df = df.iloc[keep_idx].reset_index(drop=True)
            X_feats = X_feats.iloc[keep_idx].reset_index(drop=True)
            pred_h = pred_h.iloc[keep_idx].reset_index(drop=True)
            if _vol_proxy is not None:
                _vol_proxy = _vol_proxy[keep]
            p_oof = p_oof[keep]
            n_res = n_res[keep]
            _y_per_h = {h: v[keep] for h, v in _y_per_h.items()}
            y_target_h = y_target_h[keep]
            _trade_mask = np.ones(len(df), dtype=bool)
            if "__trigger_offset_h__" in df.columns:
                _trade_mask = np.abs(np.asarray(df["__trigger_offset_h__"].values, dtype=float)) <= float(cfg.get("trade_mask_abs_hours", 4.0))

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
                # Guard: replace any inf/nan with 0 so downstream sklearn/optuna don't choke
                _y_finite_mask = np.isfinite(y_ret_raw_h)
                if not _y_finite_mask.all():
                    _y_fill = float(np.nanmedian(y_ret_raw_h[_y_finite_mask])) if _y_finite_mask.any() else 0.0
                    y_ret_raw_h = np.where(_y_finite_mask, y_ret_raw_h, _y_fill)

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
                    _w_opt = _optimize_training_sample_weights(
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
                    # If optimizer dropped non-finite rows internally, expand back to full union size
                    _full_n = len(w_meta_h)
                    if len(_w_opt) != _full_n:
                        _fin_mask = np.isfinite(y_ret_raw_h)
                        _w_expanded = np.full(_full_n, float(np.median(_w_opt)), dtype=np.float64)
                        _w_expanded[_fin_mask] = _w_opt[:int(_fin_mask.sum())]
                        w_meta_h = _w_expanded
                    else:
                        w_meta_h = _w_opt
                w_meta_h = w_meta_h.astype(np.float32)


            # Fit MetaModel for this horizon
            meta_h = MetaModel()
            meta_h.strategy_name = _h_label
            tprint(f"  Fitting MetaModel {_h_label} (n={len(df)}, feats={X_meta_base.shape[1]}) ({_time.monotonic()-_t0_meta:.1f}s)...")
            _meta_y_per_h = None if bool(cfg.get("meta_use_policy_value_target", True)) else _y_per_h
            meta_h.fit(X_meta_base, y_target_h, sample_weight=w_meta_h, groups=meta_groups,
                       y_per_horizon=_meta_y_per_h)
            meta_models[_h_label] = meta_h
            _bucket_y_ret[_h_label] = y_ret_raw_h.copy()
            tprint(f"Meta {_h_label}: fitted ({_time.monotonic()-_t0_meta:.1f}s).")

            # Orientation safeguard for MR buckets
            if meta_h.oof_probs is not None:
                y_ret_filtered = (df["__y_ret__"].values if "__y_ret__" in df.columns else y_target_h)
                _mask_eval = np.asarray(_trade_mask, dtype=bool)[:len(y_ret_filtered)]

                def _top_spread(yv, sv, frac=0.10):
                    n = len(yv)
                    if n <= 2:
                        return 0.0
                    ksel = max(1, int(frac * n))
                    it = np.argsort(sv)[-ksel:]
                    ib = np.argsort(sv)[:ksel]
                    return float(np.mean(yv[it]) - np.mean(yv[ib]))

                pred_oof = np.asarray(meta_h.oof_probs, dtype=float)
                ic_pos = _safe_spearman(pred_oof[_mask_eval], y_ret_filtered[_mask_eval])
                ic_neg = _safe_spearman((-pred_oof)[_mask_eval], y_ret_filtered[_mask_eval])
                sp_pos = _top_spread(y_ret_filtered[_mask_eval], pred_oof[_mask_eval], frac=0.10)
                sp_neg = _top_spread(y_ret_filtered[_mask_eval], (-pred_oof)[_mask_eval], frac=0.10)

                meta_h.score_sign = 1
                if k == "mr" and ((ic_neg > ic_pos + 1e-4) and (sp_neg > sp_pos + 1e-6)):
                    meta_h.score_sign = -1
                    tprint(f"Meta {_h_label}: orientation flipped (IC {ic_pos:.4f}->{ic_neg:.4f})")

                pred_for_gate = meta_h.score_sign * pred_oof
                gate_type = "meta_regression"
                gate_res = compute_stage_gate_metrics(y_target_h[_mask_eval], pred_for_gate[_mask_eval], y_ret_filtered[_mask_eval], model_type=gate_type)
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
            if bool(cfg.get("meta_race_include_classifiers", False)):
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

                _use_engine_meta_labels = bool(cfg.get("meta_clf_use_engine_labels", True))
                _y_class_override = None
                if _use_engine_meta_labels:
                    if "__y_outcome__" not in df.columns:
                        raise RuntimeError(
                            "meta_clf_use_engine_labels=True requires '__y_outcome__' in artifacts. "
                            "Regenerate labels with policy_rollout_labeling enabled."
                        )
                    _y_class_override = np.asarray(df["__y_outcome__"].values, dtype=np.int8)
                    tprint(
                        f"  Meta classifier labels source=engine counts="
                        f"SL={int(np.sum(_y_class_override == 0))} "
                        f"TO={int(np.sum(_y_class_override == 1))} "
                        f"TP={int(np.sum(_y_class_override == 2))}"
                    )

                # Spec B: Apply weight dominance constraints to meta-classifier weights
                if _y_class_override is not None:
                    w_meta_clf = _apply_dominance_constraints(w_meta_clf, _y_class_override, sl_class=0, to_class=1)
                    tprint(f"  Applied weight dominance constraints (SL>TO) to {side}_{k}_clf")

                tprint(f"  Fitting MetaClassifierModel {side}_{k} ({_time.monotonic()-_t0_meta:.1f}s)...")
                meta_clf = MetaClassifierModel()
                meta_clf.strategy_name = f"{side}_{k}"
                meta_clf.FEE_PER_ROUND_TRIP = float(cfg.get("label_round_trip_fee_pct", 0.3)) / 100.0
                # Pass vol_proxy for risk-unit thresholding
                _sel_cfg = MetaClassifierSelectionConfig(
                    max_logloss=float(cfg.get("meta_clf_max_logloss", 1.10)),
                    dynamic_utility_from_realized=bool(cfg.get("meta_clf_dynamic_utility_from_realized", True)),
                    u_tp=float(cfg.get("meta_clf_u_tp", 1.0)),
                    u_to=float(cfg.get("meta_clf_u_to", 0.0)),
                    u_sl=float(cfg.get("meta_clf_u_sl", -3.0)),
                    top_frac=float(cfg.get("meta_clf_top_frac", 0.30)),
                    min_top_n=int(cfg.get("meta_clf_min_top_n", 50)),
                    min_lift_vs_baseline=float(cfg.get("meta_clf_min_lift_vs_baseline", 0.0)),
                    require_positive_oof_utility=bool(cfg.get("meta_clf_require_positive_oof_utility", True)),
                )
                if "__u_policy_net__" in df.columns:
                    _realized_u = np.asarray(df["__u_policy_net__"].values, dtype=float)
                else:
                    _realized_u = np.log1p(np.clip(np.asarray(y_target_clf, dtype=float), -0.999999, None))
                # Shared-fold auxiliary heads for sizing/risk features
                try:
                    _u_raw = np.asarray(df["__u_policy_net__"].values, dtype=float) if "__u_policy_net__" in df.columns else np.full(len(df), np.nan, dtype=float)
                    _mae_raw = np.asarray(df["__mae_ret__"].values, dtype=float) if "__mae_ret__" in df.columns else np.full(len(df), np.nan, dtype=float)
                    _mfe_raw = np.asarray(df["__mfe_ret__"].values, dtype=float) if "__mfe_ret__" in df.columns else np.full(len(df), np.nan, dtype=float)
                    _dur_raw = np.asarray(df["__duration__"].values, dtype=float) if "__duration__" in df.columns else np.full(len(df), np.nan, dtype=float)
                    _atr_norm = np.asarray(df["__barrier_pct__"].values, dtype=float) if "__barrier_pct__" in df.columns else np.full(len(df), np.nan, dtype=float)
                    _atr_ok = np.isfinite(_atr_norm) & (_atr_norm > 0)

                    _y_u = _u_raw
                    _mae_norm = np.full(len(df), np.nan, dtype=float)
                    _mfe_norm = np.full(len(df), np.nan, dtype=float)
                    _mae_norm[_atr_ok] = np.clip(_mae_raw[_atr_ok], 0.0, None) / _atr_norm[_atr_ok]
                    _mfe_norm[_atr_ok] = np.clip(_mfe_raw[_atr_ok], 0.0, None) / _atr_norm[_atr_ok]
                    _y_mae = np.log1p(np.clip(_mae_norm, 0.0, None))
                    _y_mfe = np.log1p(np.clip(_mfe_norm, 0.0, None))
                    _y_dur = np.log1p(np.clip(_dur_raw, 0.0, None))
                    _dur_censored = np.asarray(df["__dur_censored__"].values, dtype=bool) if "__dur_censored__" in df.columns else None

                    _aux_head_oof[f"{side}_{k}"] = _train_aux_heads_shared_folds(
                        X_num=X_meta_base.select_dtypes(include=[np.number]).fillna(0.0),
                        y_u=_y_u,
                        y_mae=_y_mae,
                        y_mfe=_y_mfe,
                        y_dur=_y_dur,
                        trade_mask=_trade_mask,
                        cv_embargo_bars=int(cfg.get("cv_embargo_bars", 12)),
                        bucket_id=f"{side}_{k}",
                        data_root=str(cfg.get("data_root", "data")),
                        run_id=str(cfg.get("run_id", "default")),
                        dur_censored=_dur_censored,
                    )
                except Exception as _e_aux:
                    tprint(f"Warning: aux head training failed for {side}_{k}: {_e_aux}")

                meta_clf.fit(
                    X_meta_base,
                    y_target_clf,
                    sample_weight=w_meta_clf,
                    groups=meta_groups,
                    y_per_horizon=None if _y_class_override is not None else _y_per_h,
                    vol_proxy=None if _y_class_override is not None else _vol_proxy,
                    realized_u_policy=_realized_u,
                    selection_cfg=_sel_cfg,
                    y_class_override=_y_class_override,
                    trade_mask=_trade_mask,
                    to_fraction_cap=0.6 # Spec C1
                )
                meta_models[f"{side}_{k}_clf"] = meta_clf
                _bucket_y_ret[f"{side}_{k}_clf"] = y_target_clf.copy()

                # Early invalidation predictor (binary) for downstream gating/sizing
                if "__early_inval__" in df.columns:
                    try:
                        from sklearn.linear_model import LogisticRegression
                        from .purged_cv import PurgedKFold as _PKF
                        _y_ei = np.asarray(df["__early_inval__"].values, dtype=int)
                        _X_ei = X_meta_base.select_dtypes(include=[np.number]).fillna(0.0).to_numpy(dtype=float)
                        _oof_ei = np.full(len(_y_ei), 0.5, dtype=float)
                        _pkf = _PKF(n_splits=3, purge=int(cfg.get("cv_embargo_bars", 12)), embargo=int(cfg.get("cv_embargo_bars", 12)))
                        for _tr, _va in _pkf.split(_X_ei):
                            if len(np.unique(_y_ei[_tr])) < 2:
                                continue
                            _m_ei = LogisticRegression(max_iter=500, class_weight="balanced", random_state=42)
                            _m_ei.fit(_X_ei[_tr], _y_ei[_tr])
                            _oof_ei[_va] = _m_ei.predict_proba(_X_ei[_va])[:, 1]
                        # final fit
                        _m_ei_final = LogisticRegression(max_iter=500, class_weight="balanced", random_state=42)
                        if len(np.unique(_y_ei)) >= 2:
                            _m_ei_final.fit(_X_ei, _y_ei)
                        meta_models[f"{side}_{k}_early_inval"] = SimpleNamespace(
                            oof_probs=np.asarray(_oof_ei, dtype=np.float32),
                            model={"kind": "early_inval_clf", "models": [_m_ei_final], "name": "early_inval"},
                        )
                        _bucket_y_ret[f"{side}_{k}_early_inval"] = y_target_clf.copy()
                        tprint(f"Meta {side}_{k}_early_inval: fitted")
                    except Exception as _e_ei:
                        tprint(f"Warning: early invalidation model failed for {side}_{k}: {_e_ei}")

                tprint(f"Meta {side}_{k}_clf: fitted ({_time.monotonic()-_t0_meta:.1f}s).")
            else:
                tprint(f"Meta {side}_{k}_clf: skipped (meta_race_include_classifiers=False)")

    # ... (rest of the function) ...
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
                if is_clf and np.ndim(meta_obj.oof_probs) == 2:
                    # Align diagnostics with live execution scoring (EV proxy): 2*P(TP) - P(SL)
                    _oof_score = meta_obj.oof_probs[:, 2] * 2.0 - meta_obj.oof_probs[:, 0]
                else:
                    _oof_score = meta_obj.oof_probs
                _dm = _detailed_oof_metrics(_oof_score, _bret, cost=(2.0 * (float(cfg.get("fee_bps", 25.0)) + float(cfg.get("slippage_bps", 0.0))) / 10000.0))
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
            _is_clf_key = key.endswith("_clf")
            _is_ei_key = key.endswith("_early_inval")
            if _is_ei_key:
                _oof_pred_1d = np.asarray(meta.oof_probs, dtype=float)
                oof_df = pd.DataFrame({
                    "oof_pred": _oof_pred_1d,
                    "oof_p_early_inval": _oof_pred_1d,
                    "index": range(len(meta.oof_probs)),
                    "is_long": is_long,
                })
            elif _is_clf_key and np.ndim(meta.oof_probs) == 2:
                p_sl = np.asarray(meta.oof_probs[:, 0], dtype=float)
                p_to = np.asarray(meta.oof_probs[:, 1], dtype=float)
                p_tp = np.asarray(meta.oof_probs[:, 2], dtype=float)
                oof_ev = 2.0 * p_tp - p_sl
                oof_df = pd.DataFrame({
                    "oof_pred": oof_ev,
                    "oof_ev": oof_ev,
                    "oof_p_sl": p_sl,
                    "oof_p_to": p_to,
                    "oof_p_tp": p_tp,
                    "index": range(len(meta.oof_probs)),
                    "is_long": is_long,
                })
            else:
                _oof_pred_1d = np.asarray(meta.oof_probs, dtype=float)
                oof_df = pd.DataFrame({
                    "oof_pred": _oof_pred_1d,
                    "oof_u_hat": _oof_pred_1d,
                    "index": range(len(meta.oof_probs)),
                    "is_long": is_long,
                })
            
            # Attach raw returns from per-bucket storage (key stored directly)
            if key in _bucket_y_ret:
                _bret = _bucket_y_ret[key]
                if len(_bret) == len(meta.oof_probs):
                    oof_df["return"] = _bret

            # Persist policy-aligned utility labels for downstream sizer selection
            if "__u_policy_net__" in df.columns and len(df["__u_policy_net__"]) == len(meta.oof_probs):
                oof_df["u_policy_net"] = np.asarray(df["__u_policy_net__"].values, dtype=np.float32)
            if "__u_policy__" in df.columns and len(df["__u_policy__"]) == len(meta.oof_probs):
                oof_df["u_policy"] = np.asarray(df["__u_policy__"].values, dtype=np.float32)
            if "__y_outcome__" in df.columns and len(df["__y_outcome__"]) == len(meta.oof_probs):
                oof_df["exit_code"] = np.asarray(df["__y_outcome__"].values, dtype=np.int8)
            _bucket_base = "_".join(key.split("_")[:2])
            _aux = _aux_head_oof.get(_bucket_base)
            if isinstance(_aux, dict):
                for _cn in ["oof_u_hat", "oof_log_mae_q70_hat", "oof_log_mfe_hat", "oof_log_dur_hat"]:
                    if _cn in _aux and len(_aux[_cn]) == len(meta.oof_probs):
                        oof_df[_cn] = np.asarray(_aux[_cn], dtype=np.float32)
            if "__early_inval__" in df.columns and len(df["__early_inval__"]) == len(meta.oof_probs):
                oof_df["early_inval"] = np.asarray(df["__early_inval__"].values, dtype=np.int8)
            if "__mae_ret__" in df.columns and len(df["__mae_ret__"]) == len(meta.oof_probs):
                oof_df["mae_ret"] = np.asarray(df["__mae_ret__"].values, dtype=np.float32)
            if "__mfe_ret__" in df.columns and len(df["__mfe_ret__"]) == len(meta.oof_probs):
                oof_df["mfe_ret"] = np.asarray(df["__mfe_ret__"].values, dtype=np.float32)
            if "__duration__" in df.columns and len(df["__duration__"]) == len(meta.oof_probs):
                oof_df["duration"] = np.asarray(df["__duration__"].values, dtype=np.int16)

            oof_df.to_parquet(meta_oof_path, index=False)
            tprint(f"Saved meta OOF predictions for {key} to {meta_oof_path}")
    
    tprint(f"train_meta_models_from_artifacts: done ({_time.monotonic()-_t0_meta:.1f}s), {len(meta_models)} meta models")
    return meta_models, meta_gate_results
