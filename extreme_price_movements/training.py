import gc
import hashlib
import json
import os
import pickle
import platform
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score
from sklearn.mixture import GaussianMixture

import extreme_price_movements.fast_funcs as ff

from .barrier_geometry import make_effective_tp
from .calibration import apply_logit_shift, compute_logit_shift, compute_prevalences
from .candidates import (
    select_trade_candidates_hourly,
    select_trade_candidates_vectorized,
)
from .config import CANON_HORIZONS

# from .spike_anatomy import SpikeAnatomyModel
from .feature_selection_extreme_events import mdi_feature_selection_v3
from .gamma_specialist import build_gamma_dataset, train_gamma_from_dataset
from .gate_metrics import compute_stage_gate_metrics
from .labeling import compute_trailing_atr_labels, compute_triple_barrier_labels
from .meta_model import MetaClassifierModel, MetaModel
from .meta_training.utility_smooth import (
    smooth_utility_from_log_heads,
    smooth_utility_from_log_heads_standardized,
    smooth_utility_loss,
)
from .metrics import calculate_selection_score
from .model_race import ModelRace, _safe_binary_calibrate
from .model_scoring import (
    avg_trades_per_day,
    calibration_curve_bins,
    calibration_profile,
    ece_at_mask,
    ic_cross_sectional,
    precision_at_k,
    topk_mask,
)
from .offline_optimisers.params_store import (
    CANDIDATE_BEST_PARAMS_CSV,
    apply_offline_optimizer_best_params,
    load_tbm_all_params_per_cell,
    load_tbm_all_params_per_side_horizon,
    load_tbm_best_params_per_bucket,
    load_tbm_best_params_per_cell,
    load_tbm_best_params_per_side_horizon,
    load_tbm_geometry_grid,
)
from .optimise_tpsl_ratio import (
    PurgedKFold,
    calibrate_atr_base_pct,
    compute_vol_z_log_mad,
    run_tp_sl_selection_fast,
    scaled_atr_pct,
)
from .path_utils import resolve_reports_dir
from .policy_ml import (
    MetaMoveSelectionConfig,
    build_base_tp_vs_sl,
    load_best_policy_params_from_optimise,
    pick_meta_move_by_topq,
    policy_rollout_ml,
)
from .production_admissibility import ProdGates, production_admissibility_report
from .sample_weight_optimization import (
    combine_weights_safely,
    compute_distance_to_barrier_weights,
    compute_liquidity_weights,
    compute_recency_weights,
    compute_vol_weights,
    log_weight_statistics,
    optimize_component_weights,
    sample_weight_meta_regression,
    sample_weight_tp_classifier,
    select_test_feature_frame,
)
from .sample_weights import (
    NegMassRenormCfg,
    build_label_time_ranges,
    compute_cell_weights_neg_mass_renorm,
    compute_mfe_mae_weights,
    compute_sample_weights_with_uniqueness,
    drawdown_aware_weights,
)
from .strategy_registry import get_strategies, strategy_runtime_horizons
from .trap_specialist import (
    build_trap_dataset,
    compute_trap_oof_predictions,
    train_trap_from_dataset,
)
from .tree_leaf_policy import tree_regularization_params
from .training_utils import build_wide_tight_pair_features
from .utils import tprint

_EXP_INPUT_CLIP = 60.0
_EXP_OUTPUT_MAX = 1e12

# Features used for the miner target residualization.
# Base models must residualize their targets against these and NOT use them as inputs.
TRAINING_RESIDUALIZATION_FEATURE_KEYS: tuple[str, ...] = (
    "ema50_ema200_spread_continuous",
    "atr_change_rate_ts_continuous",
    "bars_in_high_vol_state_log_norm",
    "volatility_of_volatility_48",
    "trend_strength_percentile",
    "volatility_autocorr_48",
)


def _safe_exp_bounded(
    x, exp_clip: float = _EXP_INPUT_CLIP, out_max: float = _EXP_OUTPUT_MAX
):
    """Numerically stable exp with bounded input/output and finite guarantees."""
    x_arr = np.asarray(x, dtype=np.float64)
    x_clip = np.clip(x_arr, -abs(float(exp_clip)), abs(float(exp_clip)))
    out = np.exp(x_clip)
    out = np.where(np.isfinite(out), out, float(out_max))
    return np.clip(out, 0.0, float(out_max))


def _stable_drawdown_proxy(returns: np.ndarray) -> np.ndarray:
    """Compute drawdown from returns without overflowing cumulative products."""
    r = np.asarray(returns, dtype=np.float64)
    if r.size == 0:
        return np.zeros(0, dtype=np.float64)
    r = np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)
    # Keep log1p defined and avoid pathological values dominating the proxy.
    r = np.clip(r, -0.999999, 10.0)
    log_eq = np.cumsum(np.log1p(r))
    log_peak = np.maximum.accumulate(log_eq)
    dd = 1.0 - np.exp(np.clip(log_eq - log_peak, -700.0, 0.0))
    dd = np.where(np.isfinite(dd), dd, 1.0)
    return np.clip(dd, 0.0, 1.0)


def _normalize_oof_timestamps_to_numpy(ts_like) -> np.ndarray:
    ts = pd.to_datetime(ts_like)
    if getattr(ts.dtype, "tz", None) is not None:
        ts = ts.dt.tz_convert("UTC").dt.tz_localize(None)
    return ts.to_numpy(dtype="datetime64[ns]")


def _split_geometry_triplets_into_archetypes(triplets, archetypes=None):
    """Partition validated TBM triplets into tight/wide archetypes.

    The split is deterministic and based on joint TP/SL geometry width. Lower-width
    configurations are `tight`, higher-width are `wide`.
    """
    archetypes = [str(a) for a in (archetypes or ["tight", "wide"]) if str(a)]
    if not triplets:
        return {}
    uniq_triplets = []
    seen = set()
    for t in triplets:
        key = (round(float(t[0]), 6), round(float(t[1]), 6), int(t[2]))
        if key in seen:
            continue
        seen.add(key)
        uniq_triplets.append((float(t[0]), float(t[1]), int(t[2])))
    if len(uniq_triplets) <= 1:
        return {"tight": uniq_triplets}

    scored = []
    for t in uniq_triplets:
        k_tp, sl_tp, atr_win = t
        width_score = float(k_tp) + float(sl_tp)
        asym_score = abs(float(k_tp) - float(sl_tp))
        scored.append((width_score, asym_score, atr_win, t))
    scored.sort(key=lambda x: (x[0], x[1], x[2]))
    ordered = [x[-1] for x in scored]
    n = len(ordered)

    mid = n // 2
    groups = {
        "tight": ordered[:mid],
        "wide": ordered[mid:],
    }
    return {k: v for k, v in groups.items() if k in archetypes and v}


def _cluster_geometry_candidates_hybrid(
    triplets,
    ranked_rows,
    archetypes=None,
    topk=None,
    learnability_weight: float = 0.75,
    geometry_weight: float = 0.25,
):
    """Cluster TBM candidates into tight/wide.

    Wide is the ones with wider SL than the median, tight is tighter SL than the median.
    If SL is identical to the median, it uses TP (k_tp) as a tie-breaker.
    """
    import numpy as np

    archetypes = [str(a) for a in (archetypes or ["tight", "wide"]) if str(a)]
    if not triplets:
        return {}

    uniq_triplets = []
    seen = set()
    for t in triplets:
        key = (round(float(t[0]), 6), round(float(t[1]), 6), int(t[2]))
        if key in seen:
            continue
        seen.add(key)
        uniq_triplets.append((float(t[0]), float(t[1]), int(t[2])))

    if len(uniq_triplets) <= 1:
        return {"tight": uniq_triplets}

    # Group based on sl_as_tp_pct (index 1 of triplet)
    sl_vals = [t[1] for t in uniq_triplets]
    sl_median = np.median(sl_vals)

    tight_group = []
    wide_group = []

    # Pre-calculate median of TP (index 0) strictly for those exactly AT the median SL
    # to use as the tie-breaker
    median_sl_triplets = [t for t in uniq_triplets if t[1] == sl_median]
    tp_median = (
        np.median([t[0] for t in median_sl_triplets]) if median_sl_triplets else 0
    )

    for t in uniq_triplets:
        if t[1] < sl_median:
            tight_group.append(t)
        elif t[1] > sl_median:
            wide_group.append(t)
        else:
            # Tie breaker: compare k_tp (index 0) to median k_tp among the tied group
            if t[0] < tp_median:
                tight_group.append(t)
            elif t[0] > tp_median:
                wide_group.append(t)
            else:
                # If both SL and TP are exactly their respective medians,
                # just balance the groups.
                if len(tight_group) <= len(wide_group):
                    tight_group.append(t)
                else:
                    wide_group.append(t)

    # Handle edge case where there is no split even after tie-breaker
    # (e.g. all identical points somehow, though uniq_triplets handles most of that)
    if not tight_group:
        mid = len(uniq_triplets) // 2
        tight_group = uniq_triplets[:mid]
        wide_group = uniq_triplets[mid:]
    elif not wide_group:
        mid = len(uniq_triplets) // 2
        wide_group = uniq_triplets[mid:]
        tight_group = uniq_triplets[:mid]

    grouped = {
        "tight": tight_group,
        "wide": wide_group,
    }
    return {k: v for k, v in grouped.items() if k in archetypes and v}


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
            tprint(
                f"Warning: feature '{name}' provided as ndarray; coercing to DataFrame aligned to panel."
            )
            return pd.DataFrame(x, index=close.index, columns=close.columns)
        tprint(
            f"Warning: feature '{name}' ndarray shape {x.shape} mismatches panel {close.shape}; "
            f"using fill_value={fill_value}."
        )
        return pd.DataFrame(fill_value, index=close.index, columns=close.columns)
    if x is None:
        return pd.DataFrame(fill_value, index=close.index, columns=close.columns)
    tprint(
        f"Warning: feature '{name}' has unexpected type {type(x)}; using fill_value={fill_value}."
    )
    return pd.DataFrame(fill_value, index=close.index, columns=close.columns)


def _compute_dynamic_horizon_frame(
    atr_pct: pd.DataFrame,
    base_horizon: float,
    cfg: dict,
    _base: dict | None = None,
) -> pd.DataFrame | None:
    """
    Compute dynamic horizon based on ATR regime.
    Formula: H_dyn = H_base * (1.0 + 0.5 * clip((z - z_lo)/(z_hi - z_lo), 0, 1))
    Default z_lo = -1.0, z_hi = 2.0. Maps z=[-1, 2] to scale=[1.0, 1.5].
    """
    if not bool(cfg.get("use_dynamic_horizon", False)):
        return None

    if _base is not None and "z_clipped" in _base:
        z = _base["z_clipped"]
    else:
        # Recompute minimal z-score if base not provided
        window = int(cfg.get("barrier_atr_window", 24 * 30))
        disp_floor = float(cfg.get("barrier_disp_floor", 0.1))
        z_max = float(cfg.get("barrier_z_max", 3.0))

        atr_median = ff.numba_rolling_median(atr_pct, window)
        atr_mad = ff.numba_rolling_mad(atr_pct - atr_median, window).abs()
        atr_disp = np.maximum(atr_mad, disp_floor * atr_median)
        z = np.clip((atr_pct - atr_median) / (atr_disp + 1e-12), -z_max, z_max)

    z_lo = float(cfg.get("dynamic_horizon_z_lo", -1.0))
    z_hi = float(cfg.get("dynamic_horizon_z_hi", 2.0))
    max_scale_add = float(cfg.get("dynamic_horizon_max_scale_add", 0.5))  # +50%

    # Linear interpolation of scale from 1.0 to 1.0 + max_scale_add
    # z <= z_lo -> 0.0 -> scale 1.0
    # z >= z_hi -> 1.0 -> scale 1.5
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
    sl_mult_lo: float,
    sl_mult_hi: float,
    z_gate: float,
    use_standalone_sl: bool = False,
) -> dict:
    """Pre-compute the shared rolling base (median, MAD, z, m, sl_mult) once per
    (atr_window, side, kind, H) so compute_barrier_factory can skip redundant
    rolling operations when sweeping over (k_tp, sl_base_mult) combinations."""
    atr_median = ff.numba_rolling_median(atr_pct, window_size)
    atr_mad = ff.numba_rolling_mad(atr_pct - atr_median, window_size).abs()
    atr_disp = np.maximum(atr_mad, disp_floor * atr_median)
    z_score = (atr_pct - atr_median) / (atr_disp + 1e-12)
    z_clipped = np.clip(z_score, -z_max, z_max)
    m_clipped = np.clip(_safe_exp_bounded(k_reg * z_clipped), m_lo, m_hi)
    if not np.isfinite(np.asarray(m_clipped, dtype=np.float64)).all():
        tprint(
            "WARNING: Non-finite barrier multipliers detected in _compute_barrier_base; sanitizing to bounds."
        )
        if hasattr(m_clipped, "replace"):
            m_clipped = (
                m_clipped.replace([np.inf, -np.inf], np.nan)
                .fillna(float(m_hi))
                .clip(lower=float(m_lo), upper=float(m_hi))
            )
        else:
            m_clipped = np.where(np.isfinite(m_clipped), m_clipped, float(m_hi))
            m_clipped = np.clip(m_clipped, float(m_lo), float(m_hi))
    z_norm = np.clip((z_clipped - z_gate) / (z_max - z_gate), 0, 1)
    sl_mult = sl_mult_lo + (sl_mult_hi - sl_mult_lo) * z_norm
    return {
        "atr_median": atr_median,
        "z_clipped": z_clipped,
        "m_clipped": m_clipped,
        "sl_mult": sl_mult,
        "use_standalone_sl": use_standalone_sl,
    }


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
    sl_mult_lo: float = 0.4,
    sl_mult_hi: float = 0.7,
    sl_lo: float = 0.005,
    sl_hi: float = 0.06,
    z_gate: float = 1.0,
    tp_lo: float = 0.02,
    tp_hi: float = 0.06,
    return_components: bool = False,
    _base: dict | None = None,
    use_standalone_sl: bool = False,
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
    # Use pre-computed base when provided (avoids redundant rolling ops across k_tp/sl sweeps)
    if _base is not None:
        m_clipped = _base["m_clipped"]
        sl_mult = _base["sl_mult"]
    else:
        atr_median = ff.numba_rolling_median(atr_pct, window_size)
        atr_mad = ff.numba_rolling_mad(atr_pct - atr_median, window_size).abs()
        atr_disp = np.maximum(atr_mad, disp_floor * atr_median)
        z_clipped = np.clip((atr_pct - atr_median) / (atr_disp + 1e-12), -z_max, z_max)
        m_clipped = np.clip(_safe_exp_bounded(k_reg * z_clipped), m_lo, m_hi)
        if not np.isfinite(np.asarray(m_clipped, dtype=np.float64)).all():
            tprint(
                "WARNING: Non-finite barrier multipliers detected in compute_barrier_factory; sanitizing to bounds."
            )
            if hasattr(m_clipped, "replace"):
                m_clipped = (
                    m_clipped.replace([np.inf, -np.inf], np.nan)
                    .fillna(float(m_hi))
                    .clip(lower=float(m_lo), upper=float(m_hi))
                )
            else:
                m_clipped = np.where(np.isfinite(m_clipped), m_clipped, float(m_hi))
                m_clipped = np.clip(m_clipped, float(m_lo), float(m_hi))
        z_norm = np.clip((z_clipped - z_gate) / (z_max - z_gate), 0, 1)
        sl_mult = sl_mult_lo + (sl_mult_hi - sl_mult_lo) * z_norm

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

    # User requested to use BOTH standalone SL and compounding with TP distance.
    # We compute both and take the more conservative (tighter) stop-loss of the two.

    # 1. Standalone logic: SL is a direct multiple of ATR.
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

    # 2. Compounded logic (legacy): SL is derived from effective TP.
    sl_vals_compounded = sl_base_mult * sl_mult * tp_vals

    # 3. Fuse the two (take the tighter SL to satisfy both sets of geometry constraints).
    sl_vals = np.minimum(sl_vals_standalone, sl_vals_compounded)

    tp_df = pd.DataFrame(tp_vals, index=atr_pct.index, columns=atr_pct.columns)
    sl_df = pd.DataFrame(sl_vals, index=atr_pct.index, columns=atr_pct.columns)

    if return_components:
        z_arr = np.asarray(z_clipped, dtype=np.float32)
        m_arr = np.asarray(m_clipped, dtype=np.float32)
        sl_mult_arr = np.asarray(sl_mult, dtype=np.float32)
        tp_arr = np.asarray(tp_vals, dtype=np.float32)
        # Per-asset diagnostics for cross-asset portability check
        asset_diagnostics = {}
        for col in atr_pct.columns:
            col_idx = atr_pct.columns.get_loc(col)
            m_vals = m_arr[:, col_idx]
            sl_m = sl_mult_arr[:, col_idx]
            atr_col = atr_pct[col].values
            tp_col = tp_arr[:, col_idx]

            # TP in ATR units: avoid division by zero
            tp_atr_ratio = np.divide(
                tp_col, atr_col, out=np.full_like(tp_col, np.nan), where=atr_col != 0
            )

            asset_diagnostics[col] = {
                "m_at_m_lo_pct": float(np.mean(m_vals == m_lo)),
                "m_at_m_hi_pct": float(np.mean(m_vals == m_hi)),
                "sl_at_sl_lo_pct": float(np.mean(sl_m == sl_lo)),
                "sl_at_sl_hi_pct": float(np.mean(sl_m == sl_hi)),
                "tp_atr_units": float(np.nanmean(tp_atr_ratio)),
            }

        diagnostics = {
            "z_mean": float(np.nanmean(z_arr)),
            "z_p10": float(np.nanpercentile(z_arr, 10)),
            "z_p90": float(np.nanpercentile(z_arr, 90)),
            "z_below_gate_pct": float(np.mean(z_arr < z_gate)),
            "z_above_gate_pct": float(np.mean(z_arr >= z_gate)),
            "m_mean": float(np.nanmean(m_arr)),
            "m_p10": float(np.nanpercentile(m_arr, 10)),
            "m_p90": float(np.nanpercentile(m_arr, 90)),
            "m_at_m_lo_pct": float(np.mean(m_arr == m_lo)),
            "m_at_m_hi_pct": float(np.mean(m_arr == m_hi)),
            "sl_mult_mean": float(np.nanmean(sl_mult)),
            "sl_at_sl_lo_pct": float(np.mean(sl_mult == sl_lo)),
            "sl_at_sl_hi_pct": float(np.mean(sl_mult == sl_hi)),
            "tp_mean": float(np.nanmean(tp_vals)),
            "sl_mean": float(np.nanmean(sl_vals)),
            "clip_low_pct": float(np.mean(m_arr == m_lo)),
            "clip_high_pct": float(np.mean(m_arr == m_hi)),
            "asset_diagnostics": asset_diagnostics,
        }
        return tp_df, sl_df, diagnostics

    return tp_df, sl_df


def _emoji(pass_flag):
    return "✅" if bool(pass_flag) else "⚠️"


TOPK_GATE_FRAC = 0.20
TOPK_INFO_FRACS = (0.10, 0.30)


def _resolve_training_cfg_with_offline_optimisers(cfg):
    """Apply persisted offline-optimiser best params onto cfg with cfg values as fallback."""
    try:
        return apply_offline_optimizer_best_params(cfg)
    except Exception as exc:
        tprint(
            f"Warning: failed to load offline optimiser params; using cfg defaults ({exc})"
        )
        return cfg


def _build_optimal_candidate_mask(panel, feats, cfg):
    """Build candidate mask strictly from persisted offline-optimal threshold conditions."""
    cfg_resolved = _resolve_training_cfg_with_offline_optimisers(cfg)

    from extreme_price_movements.intraday_crypto_library import (
        PERSISTED_INTRADAY_LIBRARY_COLUMNS,
        build_intraday_crypto_library,
    )
    from extreme_price_movements.lgbm_based_mask_generation import (
        CanonicalRuleMaskResolver,
        FeatureProcessor,
        parse_condition_string,
        parse_slot_map,
        split_composite_key,
    )
    from extreme_price_movements.strategy_registry import get_strategies

    # Import legacy mask builder for fallback on legacy mode keys
    try:
        from extreme_price_movements.inference.candidate_selector import (
            _build_mask_for_mode as _build_legacy_mode_mask,
        )
        from extreme_price_movements.inference.candidate_selector import (
            _up_down_zones as _legacy_up_down_zones,
        )
    except Exception:
        _build_legacy_mode_mask = None
        _legacy_up_down_zones = None

    strategies = get_strategies(cfg_resolved)

    tprint(
        f"Building context-based masks from LGBM strategies: {len(strategies)} strategies found."
    )

    close_df = panel["close"]
    n_ts, n_syms = close_df.shape

    idx_flat = np.repeat(close_df.index.to_numpy(), n_syms)
    sym_flat = np.tile(close_df.columns.to_numpy(), n_ts)

    feats_1d = {}
    for k, v in feats.items():
        if hasattr(v, "to_numpy"):
            feats_1d[k] = v.to_numpy(dtype=np.float32).ravel()
        else:
            feats_1d[k] = np.asarray(v, dtype=np.float32).ravel()

    def _extract_rule_feature_names(_canonical_key: str) -> set[str]:
        _out: set[str] = set()
        _parts = split_composite_key(_canonical_key)
        if _parts is not None:
            _out.update(_extract_rule_feature_names(_parts[0]))
            _out.update(_extract_rule_feature_names(_parts[1]))
            return _out
        try:
            _slot_map = parse_slot_map(
                _canonical_key, ("trigger", "location", "regime")
            )
        except Exception:
            return _out
        for _slot_value in _slot_map.values():
            if _slot_value in {"*", "Composite"}:
                continue
            for _cond_str in str(_slot_value).split("&"):
                _parsed = parse_condition_string(_cond_str)
                if _parsed is None:
                    continue
                _feature_name, _, _ = _parsed
                if _feature_name not in {"Composite"}:
                    _out.add(str(_feature_name))
        return _out

    _required_rule_features: set[str] = set()
    for _strat in strategies:
        _required_rule_features.update(
            _extract_rule_feature_names(str(_strat.get("base_event_trigger", "") or ""))
        )

    _missing_rule_features = {
        _k for _k in _required_rule_features if _k and _k not in feats
    }
    if "tail_asymmetry_q90_q10_atr_norm" in _missing_rule_features and "ret1h" in feats:
        _ret1h = feats["ret1h"].astype(np.float32, copy=False)
        _q90 = ff.numba_rolling_quantile(_ret1h, 50, 0.90)
        _q10 = np.abs(ff.numba_rolling_quantile(_ret1h, 50, 0.10))
        _raw = np.log((_q90 + 1e-8) / (_q10 + 1e-8))
        feats["tail_asymmetry_q90_q10_atr_norm"] = np.tanh(_raw).astype(np.float32)
        _missing_rule_features.discard("tail_asymmetry_q90_q10_atr_norm")

    _need_intraday = sorted(
        _missing_rule_features.intersection(set(PERSISTED_INTRADAY_LIBRARY_COLUMNS))
    )
    if _need_intraday:
        _close_df = panel["close"].astype(np.float32, copy=False)
        _session_ids = pd.Series(
            pd.factorize(_close_df.index.normalize())[0].astype(np.int32),
            index=_close_df.index,
            dtype="int32",
        )
        _wide_lib = build_intraday_crypto_library(
            {
                "open": panel["open"].astype(np.float32, copy=False),
                "high": panel["high"].astype(np.float32, copy=False),
                "low": panel["low"].astype(np.float32, copy=False),
                "close": _close_df,
                "volume": panel["volume"].astype(np.float32, copy=False),
                "session_id": _session_ids,
            }
        )
        if isinstance(_wide_lib, dict):
            for _k in _need_intraday:
                _v = _wide_lib.get(_k)
                if isinstance(_v, pd.DataFrame):
                    feats[_k] = _v.astype(np.float32, copy=False)
                elif isinstance(_v, pd.Series):
                    _vals = np.broadcast_to(
                        _v.to_numpy(dtype=np.float32)[:, None], _close_df.shape
                    )
                    feats[_k] = pd.DataFrame(
                        _vals,
                        index=_close_df.index,
                        columns=_close_df.columns,
                        copy=False,
                    ).astype(np.float32, copy=False)

    fp = FeatureProcessor()
    X, metadata, audits = fp.prepare_features(
        feats_1d, idx_flat, sym_flat, cfg_resolved
    )
    tprint("Finished FeatureProcessor.prepare_features.")
    resolver = CanonicalRuleMaskResolver(X, metadata)
    tprint("Finished CanonicalRuleMaskResolver initialization.")

    # Load per-mode mask params for legacy fallback
    _legacy_mode_cfg = dict(cfg_resolved.get("candidate_mask_params_by_mode", {}) or {})

    def _eval_numeric_condition(
        feature_df: pd.DataFrame, operator: str, raw_value: str
    ) -> pd.DataFrame:
        threshold = float(raw_value)
        vals = feature_df.reindex(index=close_df.index, columns=close_df.columns)
        if operator == "<=":
            return vals <= threshold
        if operator == "<":
            return vals < threshold
        if operator == ">":
            return vals > threshold
        if operator == ">=":
            return vals >= threshold
        if operator == "==":
            return vals == threshold
        raise ValueError(f"Unsupported canonical operator: {operator}")

    def _direct_canonical_mask(canonical_key: str) -> pd.DataFrame | None:
        _parts = split_composite_key(canonical_key)
        if _parts is not None:
            _left = _direct_canonical_mask(_parts[0])
            _right = _direct_canonical_mask(_parts[1])
            if _left is None or _right is None:
                return None
            return (_left | _right).fillna(False).astype(bool)

        try:
            slot_map = parse_slot_map(canonical_key, resolver.slot_order)
        except ValueError:
            slot_map = parse_slot_map(canonical_key, ("trigger", "location", "regime"))

        mask_df = pd.DataFrame(True, index=close_df.index, columns=close_df.columns)
        for slot_value in slot_map.values():
            if slot_value in {"*", "Composite"}:
                continue
            for cond_str in slot_value.split("&"):
                parsed = parse_condition_string(cond_str)
                if parsed is None:
                    return None
                feature_name, operator, raw_value = parsed
                feature_df = feats.get(feature_name)
                if not isinstance(feature_df, pd.DataFrame):
                    return None
                mask_df &= _eval_numeric_condition(
                    feature_df, operator, raw_value
                ).fillna(False)

        return mask_df.fillna(False).astype(bool)

    def _legacy_mode_mask(mode_name: str):
        """Build mask for legacy mode keys like price_up_tf using per-mode config."""
        if (
            _build_legacy_mode_mask is None
            or _legacy_up_down_zones is None
            or mode_name not in _legacy_mode_cfg
        ):
            return None
        try:
            tprint(f"Building legacy mask for {mode_name}...")
            up_zone, down_zone = _legacy_up_down_zones(
                feats,
                panel,
                metric=str(cfg_resolved.get("train_candidate_metric") or "ret12h"),
            )
            tprint(f"Got up/down zones for {mode_name}...")
            per_mode = _build_legacy_mode_mask(
                panel,
                feats,
                _legacy_mode_cfg[mode_name],
            )
            tprint(f"Got per_mode legacy mask for {mode_name}...")
            if mode_name in {"price_up_tf", "price_up_mr"}:
                return (up_zone & per_mode).fillna(False).astype(bool)
            else:
                return (down_zone & per_mode).fillna(False).astype(bool)
        except Exception as e:
            tprint(f"[WARNING] Legacy mode mask failed for {mode_name}: {e}")
            return None

    mask_by_strategy = {}
    global_mask = None

    for strat in strategies:
        strat_id = strat["strategy_id"]
        key = strat["base_event_trigger"]

        # Try canonical resolver first, fall back to legacy mode builder for price_* keys
        mask_df = None
        if key.startswith("price_up") or key.startswith("price_down"):
            mask_df = _legacy_mode_mask(key)
            if mask_df is not None:
                tprint(
                    f"[DIAGNOSTIC] Strategy {strat_id} using LEGACY mode mask for {key}: {mask_df.sum().sum():,} active triggers"
                )

        if mask_df is None:
            try:
                mask_1d = resolver.get_mask(key)
                mask_2d = mask_1d.reshape((n_ts, n_syms))
                mask_df = pd.DataFrame(
                    mask_2d, index=close_df.index, columns=close_df.columns
                )
                tprint(
                    f"[DIAGNOSTIC] Strategy {strat_id} using CANONICAL mask for {key}: {mask_df.sum().sum():,} active triggers"
                )
            except KeyError as e:
                mask_df = _direct_canonical_mask(key)
                if mask_df is not None:
                    tprint(
                        f"[DIAGNOSTIC] Strategy {strat_id} using DIRECT canonical mask for {key}: {mask_df.sum().sum():,} active triggers"
                    )
                else:
                    tprint(
                        f"Failed to generate mask for {strat_id} with key {key}: {e}"
                    )
                    mask_df = pd.DataFrame(
                        False, index=close_df.index, columns=close_df.columns
                    )

        mask_by_strategy[strat_id] = mask_df

        if global_mask is None:
            global_mask = mask_df
        else:
            global_mask = global_mask | mask_df

    if global_mask is None:
        global_mask = pd.DataFrame(
            False, index=close_df.index, columns=close_df.columns
        )

    return global_mask, cfg_resolved, mask_by_strategy


def _mad(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0
    med = np.median(x)
    return float(np.median(np.abs(x - med)) + 1e-12)


def _safe_spearman(x, y):
    try:
        x_nan = np.asarray(x, dtype=float)
        y_nan = np.asarray(y, dtype=float)
        mask = np.isfinite(x_nan) & np.isfinite(y_nan)
        if not np.any(mask):
            return 0.0

        # Check variance to avoid ConstantInputWarning
        if np.std(x_nan[mask]) < 1e-12 or np.std(y_nan[mask]) < 1e-12:
            return 0.0

        v = spearmanr(x_nan[mask], y_nan[mask]).correlation
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
    days_all = np.array([np.datetime64(t, "D") for t in ts])
    n_days = np.unique(days_all).size
    return float(k / max(n_days, 1))


def _fit_direct_extratrees_base_model(
    *,
    kind_name,
    X,
    y,
    sample_weight=None,
    returns=None,
    groups=None,
    symbols=None,
    n_splits: int = 2,
    cfg=None,
):
    """Fit the base alpha model directly with ExtraTrees, bypassing ModelRace."""
    import os as _os

    from scipy.stats import rankdata as _rankdata
    from sklearn.base import clone
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

    from .purged_cv import PurgedKFold as _PurgedKFold

    race = ModelRace(kind=kind_name, task="base", n_splits=n_splits)
    y = np.asarray(y, dtype=np.float64)
    y = np.clip(y, 0.0, 1.0)
    y_hard = (y >= 0.5).astype(np.int8)
    returns_arr = y if returns is None else np.asarray(returns, dtype=np.float64)
    sample_weight_arr = (
        None if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
    )
    groups_arr = None if groups is None else np.asarray(groups)
    symbols_arr = None if symbols is None else np.asarray(symbols)

    # Base model fitting only accepts numeric inputs. Keep this boundary clean so
    # timestamp / identifier columns cannot leak through feature selection or
    # grouped variant assembly.
    if hasattr(X, "select_dtypes"):
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        non_numeric_cols = [c for c in X.columns if c not in set(numeric_cols)]
        if non_numeric_cols:
            tprint(
                f"Direct ExtraTrees: Dropping {len(non_numeric_cols)} non-numeric features before fit."
            )
        X = X[numeric_cols].copy()
        if X.empty:
            raise ValueError(
                "Direct ExtraTrees: no numeric features available after filtering."
            )

    if hasattr(X, "iloc"):
        try:
            X_np = X.to_numpy(dtype=np.float32, copy=False)
        except (ValueError, TypeError):
            X_np = X
    else:
        X_np = X

    candidate = race._get_candidates(race_mode=True)["extratrees"]
    inner = candidate.estimator if hasattr(candidate, "estimator") else candidate

    from .post_race_hpo import load_best_base_extratrees_params

    _hpo_dir = (
        _os.path.join(cfg.get("data_root", "data"), "hpo_out")
        if cfg is not None
        else None
    )
    if cfg is not None and "base_hpo_out_dir" in cfg:
        _hpo_dir = cfg["base_hpo_out_dir"]

    hpo_params = None
    if _hpo_dir is not None:
        hpo_params = load_best_base_extratrees_params(_hpo_dir, scope_key=kind_name)
        if hpo_params is None:
            hpo_params = load_best_base_extratrees_params(_hpo_dir)

    if hpo_params is not None:
        p = dict(hpo_params)
        tree_dyn = tree_regularization_params(y_hard, task_type="classification")
        p.setdefault("min_samples_leaf", int(tree_dyn["min_samples_leaf"]))
        p.setdefault("min_samples_split", int(tree_dyn["min_samples_split"]))
        p.setdefault("bootstrap", False)
        p.setdefault("ccp_alpha", 1e-4)
        p.setdefault("max_leaf_nodes", 512)
        inner.set_params(**{k: v for k, v in p.items() if k in inner.get_params()})
        hpo_scope = kind_name
        tprint(
            f"Direct ExtraTrees[{hpo_scope}]: using HPO params: "
            f"max_depth={inner.max_depth}, min_samples_leaf={inner.min_samples_leaf}, "
            f"min_samples_split={inner.min_samples_split}, max_features={inner.max_features}, "
            f"ccp_alpha={inner.ccp_alpha}, min_impurity_decrease={inner.min_impurity_decrease}"
        )
    else:
        tree_dyn = tree_regularization_params(y_hard, task_type="classification")
        min_leaf_dyn = int(tree_dyn["min_samples_leaf"])
        min_split_dyn = int(tree_dyn["min_samples_split"])
        inner.set_params(
            min_samples_leaf=min_leaf_dyn,
            min_samples_split=min_split_dyn,
            bootstrap=False,
            ccp_alpha=1e-4,
            max_leaf_nodes=512,
        )
        tprint(
            f"Direct ExtraTrees: min_samples_leaf={min_leaf_dyn}, min_samples_split={min_split_dyn}"
        )

    purge_samples = race.max_label_horizon_hours + 2
    embargo_samples = max(2, race.max_label_horizon_hours // 2)
    tscv = _PurgedKFold(
        n_splits=n_splits,
        purge=purge_samples,
        embargo=embargo_samples,
        times=groups_arr,
    )
    cached_splits = list(tscv.split(X_np))
    if not cached_splits:
        n_total = len(y_hard)
        split_at = max(1, int(np.floor(0.7 * n_total)))
        if split_at < n_total:
            cached_splits = [
                (
                    np.arange(0, split_at, dtype=np.int32),
                    np.arange(split_at, n_total, dtype=np.int32),
                )
            ]
            tprint(
                "Direct ExtraTrees: fell back to single chronological holdout split."
            )

    oof_probs = np.full(len(y_hard), np.nan, dtype=np.float64)
    for fold_i, (train_idx, val_idx) in enumerate(cached_splits, start=1):
        if train_idx.size == 0 or val_idx.size == 0:
            continue
        X_tr = X_np[train_idx] if isinstance(X_np, np.ndarray) else X.iloc[train_idx]
        X_val = X_np[val_idx] if isinstance(X_np, np.ndarray) else X.iloc[val_idx]
        y_tr = y_hard[train_idx]
        y_val = y_hard[val_idx]
        w_tr = sample_weight_arr[train_idx] if sample_weight_arr is not None else None
        est = clone(candidate)
        race._fit_model(est, X_tr, y_tr, sample_weight=w_tr)
        _proba_out = est.predict_proba(X_val)
        if _proba_out.shape[1] < 2:
            oof_probs[val_idx] = 0.5
            continue
        probs_raw = np.asarray(_proba_out[:, 1], dtype=np.float64)
        if sample_weight_arr is not None:
            w_tr_sum = float(np.sum(w_tr))
            p_weighted = float(np.sum(w_tr * y_tr) / max(w_tr_sum, 1e-12))
        else:
            p_weighted = float(np.mean(y_tr))
        p_unweighted = float(np.mean(y_val))
        delta_logit = compute_logit_shift(p_unweighted, p_weighted, eps=1e-6)
        oof_probs[val_idx] = apply_logit_shift(probs_raw, delta_logit, eps=1e-6)
        tprint(
            f"  Direct ExtraTrees fold {fold_i}/{len(cached_splits)} complete: n_train={train_idx.size}, n_val={val_idx.size}"
        )

    oof_probs = np.nan_to_num(oof_probs, nan=0.5)
    oof_probs_raw = np.asarray(oof_probs, dtype=np.float32).copy()
    p_unweighted_all, p_weighted_all = compute_prevalences(y_hard, sample_weight_arr)
    race._used_sample_weight_ = sample_weight_arr is not None
    race.calibration_state_ = race._build_bias_state(
        p_unweighted_all, p_weighted_all, eps=1e-6
    )
    race.calibration_state_["calibration_method"] = "identity"
    race.calibration_state_["calibration_input"] = "bias_corrected"
    race.calibrator_ = None
    race.platt_calibrator_ = None
    race.oof_probs = np.asarray(oof_probs, dtype=np.float32)
    race.raw_rank_scores_ = (
        (_rankdata(oof_probs) - 1.0) / max(len(oof_probs) - 1, 1)
        if len(oof_probs) > 0
        else np.zeros(0, dtype=np.float64)
    )
    tprint(
        f"OOF predictions: mean={float(np.mean(oof_probs)):.4f}, std={float(np.std(oof_probs)):.4f}"
    )

    try:
        tprint(
            "Running calibration router on OOF predictions (isotonic/platt/identity)..."
        )
        calibrated_oof, race.calibrator_, cal_method = _safe_binary_calibrate(
            oof_probs, y_hard, min_unique=20, min_samples=100
        )
        if isinstance(race.calibration_state_, dict):
            race.calibration_state_["calibration_input"] = "bias_corrected"
            race.calibration_state_["calibration_method"] = cal_method
        tprint(f"Calibration router selected: {cal_method}")

        min_variance = 0.01
        cal_std = float(np.std(calibrated_oof))
        if cal_std < min_variance and len(oof_probs) > 1:
            tprint(
                f"WARNING: Calibrated scores have low variance (std={cal_std:.6f}). Enforcing minimum spread."
            )
            rank_scores = (_rankdata(oof_probs) - 1.0) / max(len(oof_probs) - 1, 1)
            prevalence = float(np.mean(y_hard))
            rank_scores_centered = np.clip(rank_scores - 0.5 + prevalence, 0.05, 0.95)
            calibrated_oof = 0.7 * calibrated_oof + 0.3 * rank_scores_centered
            tprint(
                f"  Blended with rank scores: new std={float(np.std(calibrated_oof)):.6f}"
            )

        raw_brier = float(brier_score_loss(y_hard, np.clip(oof_probs, 1e-7, 1 - 1e-7)))
        cal_brier = float(
            brier_score_loss(y_hard, np.clip(calibrated_oof, 1e-7, 1 - 1e-7))
        )
        tprint(
            f"Calibration ({cal_method}): Brier raw={raw_brier:.4f} -> calibrated={cal_brier:.4f}"
        )

        platt_calibrator = LogisticRegression(random_state=42, max_iter=1000)
        platt_calibrator.fit(calibrated_oof.reshape(-1, 1), y_hard)
        platt_calibrated = platt_calibrator.predict_proba(
            calibrated_oof.reshape(-1, 1)
        )[:, 1]
        platt_brier = float(
            brier_score_loss(y_hard, np.clip(platt_calibrated, 1e-7, 1 - 1e-7))
        )
        if platt_brier < cal_brier - 1e-4:
            race.platt_calibrator_ = platt_calibrator
            tprint(
                f"Platt scaling enabled: Brier improved {cal_brier:.4f} -> {platt_brier:.4f}"
            )
        else:
            race.platt_calibrator_ = None
            tprint(
                f"Platt scaling skipped: no improvement (isotonic={cal_brier:.4f}, platt={platt_brier:.4f})"
            )
        race.oof_probs = np.asarray(calibrated_oof, dtype=np.float32)
    except Exception as exc:
        race.calibrator_ = None
        race.platt_calibrator_ = None
        tprint(f"Direct ExtraTrees calibration fallback to identity: {exc}")

    sel = calculate_selection_score(
        y_hard,
        race.oof_probs,
        returns_arr,
        sample_weight=sample_weight_arr,
        w_bss=0.20,
        w_realized=0.55,
        w_uic=0.25,
    )
    try:
        auc = (
            float(roc_auc_score(y_hard, race.oof_probs))
            if len(np.unique(y_hard)) > 1
            else 0.5
        )
    except Exception:
        auc = 0.5
    try:
        ll = float(log_loss(y_hard, np.clip(race.oof_probs, 1e-7, 1 - 1e-7)))
    except Exception:
        ll = float("nan")
    try:
        acc = float(accuracy_score(y_hard, race.oof_probs >= 0.5))
    except Exception:
        acc = float("nan")
    if (
        returns_arr is not None
        and np.std(race.oof_probs) > 1e-9
        and np.std(returns_arr) > 1e-9
    ):
        try:
            if symbols_arr is not None:
                ic = float(
                    ic_cross_sectional(race.oof_probs, returns_arr, groups=symbols_arr)
                )
                if not np.isfinite(ic):
                    ic = 0.0
            else:
                ic = float(
                    np.corrcoef(_rankdata(race.oof_probs), _rankdata(returns_arr))[0, 1]
                )
        except Exception:
            ic = 0.0
    else:
        ic = 0.0
    top10_mask = topk_mask(
        race.oof_probs, 0.10, groups=groups_arr if groups_arr is not None else None
    )
    top30_mask = topk_mask(
        race.oof_probs, 0.30, groups=groups_arr if groups_arr is not None else None
    )
    curve = calibration_curve_bins(y_hard, race.oof_probs, n_bins=10)
    dm = {
        "score": float(sel.get("Selection_Score", 0.0)),
        "rank_score": float(sel.get("Selection_Score", 0.0)),
        "AUC": float(auc),
        "IC": float(ic),
        "BSS": float(sel.get("BSS", 0.0)),
        "Brier": float(sel.get("Brier", np.nan)),
        "Prec10": float(sel.get("Prec_Top10", np.nan)),
        "Prec20": float(sel.get("Prec_Top20", np.nan)),
        "Prec25": float(sel.get("Prec_Top25", np.nan)),
        "Prec30": float(sel.get("Prec_Top30", np.nan)),
        "Prec40": float(sel.get("Prec_Top40", np.nan)),
        "LogLoss": float(ll),
        "Accuracy": float(acc),
        "ece_top10": float(
            ece_at_mask(
                y_hard,
                race.oof_probs,
                top10_mask,
                n_bins=10,
                w=sample_weight_arr,
            )
        ),
        "ece_top30": float(
            ece_at_mask(
                y_hard,
                race.oof_probs,
                top30_mask,
                n_bins=10,
                w=sample_weight_arr,
            )
        ),
        "calibration_curve": curve,
        "calibration_profile": calibration_profile(curve),
        "oof_probs": np.asarray(race.oof_probs, dtype=np.float32).copy(),
        "oof_raw": np.asarray(oof_probs_raw, dtype=np.float32).copy(),
    }
    degeneracy = _base_oof_degeneracy_summary(oof_probs_raw, race.oof_probs)
    dm["degeneracy"] = degeneracy
    race.is_degenerate_ = bool(degeneracy.get("is_degenerate", False))
    race.degeneracy_info_ = degeneracy
    if race.is_degenerate_:
        tprint(
            f"CRITICAL: Degenerate base classifier [{kind_name}] "
            f"raw_unique={degeneracy['raw']['unique_count']} "
            f"cal_unique={degeneracy['calibrated']['unique_count']} "
            f"raw_std={degeneracy['raw']['std']:.6f} "
            f"cal_std={degeneracy['calibrated']['std']:.6f} "
            f"reasons={degeneracy['reasons']}"
        )
    race.best_model_name = "extratrees"
    race.metrics = {"extratrees": float(sel.get("Selection_Score", 0.0))}
    race.detailed_metrics = {"extratrees": dm}
    race.best_model = clone(candidate)
    race._fit_model(race.best_model, X_np, y_hard, sample_weight=sample_weight_arr)

    try:
        tprint("Post-refit recalibration: generating OOF from refit ExtraTrees...")
        refit_oof = np.full(len(y_hard), np.nan, dtype=np.float64)
        for train_idx, val_idx in cached_splits:
            if train_idx.size == 0 or val_idx.size == 0:
                continue
            X_val_fold = X_np[val_idx] if isinstance(X_np, np.ndarray) else X.iloc[val_idx]
            probs_raw = np.asarray(
                race.best_model.predict_proba(X_val_fold)[:, 1], dtype=np.float64
            )
            y_tr_fold = y_hard[train_idx]
            if sample_weight_arr is not None:
                w_tr_fold = sample_weight_arr[train_idx]
                den = float(np.sum(w_tr_fold))
                p_weighted_fold = float(
                    np.sum(w_tr_fold * y_tr_fold) / max(den, 1e-12)
                )
            else:
                p_weighted_fold = float(np.mean(y_tr_fold))
            p_unweighted_fold = float(np.mean(y_hard[val_idx]))
            delta_logit_fold = compute_logit_shift(
                p_unweighted_fold, p_weighted_fold, eps=1e-6
            )
            refit_oof[val_idx] = apply_logit_shift(
                probs_raw, delta_logit_fold, eps=1e-6
            )
        refit_oof = np.nan_to_num(refit_oof, nan=0.5)

        refit_calibrated, refit_calibrator, refit_cal_method = (
            _safe_binary_calibrate(refit_oof, y_hard, min_unique=20, min_samples=100)
        )
        race.calibrator_ = refit_calibrator
        if isinstance(race.calibration_state_, dict):
            race.calibration_state_["calibration_method"] = refit_cal_method

        cal_brier = float(
            brier_score_loss(
                y_hard, np.clip(refit_calibrated, 1e-7, 1 - 1e-7)
            )
        )
        platt = LogisticRegression(random_state=42, max_iter=1000)
        platt.fit(refit_calibrated.reshape(-1, 1), y_hard)
        platt_pred = platt.predict_proba(refit_calibrated.reshape(-1, 1))[:, 1]
        platt_brier = float(
            brier_score_loss(y_hard, np.clip(platt_pred, 1e-7, 1 - 1e-7))
        )
        if platt_brier < cal_brier - 1e-4:
            race.platt_calibrator_ = platt
            tprint(
                f"Post-refit recalibration: {refit_cal_method} + Platt "
                f"(Brier {cal_brier:.4f} -> {platt_brier:.4f})"
            )
        else:
            race.platt_calibrator_ = None
            tprint(
                f"Post-refit recalibration: {refit_cal_method} "
                f"(Brier {cal_brier:.4f}, Platt skipped)"
            )
    except Exception as _e:
        tprint(
            f"WARNING: post-refit recalibration failed, keeping OOF calibration: {_e}"
        )

    return race


def _evaluate_target(score, y, cost=0.005):
    """Composite scorer for meta target selection."""
    df_ev = pd.DataFrame(
        {"score": np.asarray(score, dtype=float), "y": np.asarray(y, dtype=float)}
    ).dropna()
    if len(df_ev) < 30:
        return -999.0
    df_ev["rank"] = df_ev["score"].rank(pct=True)
    df_ev["y_rank"] = df_ev["y"].rank(pct=True)

    ic_global = _safe_spearman(df_ev["score"].values, df_ev["y"].values)

    df_ev["decile"] = (df_ev["rank"] * 10).astype(int).clip(upper=9)
    q_means = df_ev.groupby("decile")["y"].mean()
    if len(q_means) >= 10:
        spread_q10_q1 = float(q_means.iloc[9] - q_means.iloc[0])
    else:
        spread_q10_q1 = 0.0

    all_returns = df_ev["y"] - cost
    stability = (
        float(all_returns.mean() / all_returns.std()) if all_returns.std() > 0 else 0.0
    )

    k30 = max(1, int(0.30 * len(df_ev)))
    idx_top30 = np.argpartition(df_ev["score"].values, -k30)[-k30:]
    mean_ret_t30 = float(df_ev["y"].values[idx_top30].mean())

    composite = (
        2.0 * ic_global + 3.0 * spread_q10_q1 + 2.0 * stability + 1.0 * mean_ret_t30
    )
    return float(composite)


def _build_target_variants(y_ret_raw, vol_proxy=None):
    """Build 4 base target variants x 2 normalizations (raw + semivol) = 8 targets."""
    from scipy.special import expit as _sigmoid
    from scipy.stats import rankdata

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
        edges[0] -= 1e-12
        edges[-1] += 1e-12
        bins = np.clip(np.digitize(y, edges) - 1, 0, n_qbins - 1)
        t_qbin = (bins + 0.5) / n_qbins

    # Base 4: Tail-Amplified Percentile (top-30% emphasis)
    t_tail = rk + 0.5 * np.maximum(0.0, rk - 0.70)

    bases = {
        "rank_pct": rk,
        "soft_top30": t_soft30,
        "qbin_mid": t_qbin,
    }

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
        semi_scale = (
            np.power(vp_med / np.clip(vp, 1e-9, None), 0.5) if has_vol else np.ones(n)
        )
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
    spread_10_1 = (
        float(q_means.get(9, 0) - q_means.get(0, 0)) if len(q_means) >= 2 else 0.0
    )
    # Top-30% metrics
    t30 = rk >= 0.70
    n_t30 = int(t30.sum())
    if n_t30 < 3:
        return {"IC_global": ic_g, "Spread_10v1": spread_10_1, "n": n, "n_top30": 0}
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
            _s_chunk = s[i : i + chunk]
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
            _s_c = s[i : i + chunk]
            _y_c = y[i : i + chunk]
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
        "IC_global": ic_g,
        "Spread_10v1": spread_10_1,
        "IC_top30": ic_t30,
        "Mean_ret_t30": mean_ret_t30,
        "Mean_net_t30": mean_net_t30,
        "Sharpe_t30": sharpe_t30,
        "Sortino_t30": sortino_t30,
        "Lift@30": lift30,
        "Spread_t30vb30": spread_t30_b30,
        "Turnover": turnover,
        "Stability": stability,
        "n": n,
        "n_top30": n_t30,
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
                n_estimators=200,
                max_depth=6,
                min_samples_leaf=30,
                max_features="sqrt",
                n_jobs=3,
                random_state=42,
            )

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
    _hdr = (
        f"  {'Target':25s} {'Comp':>7s} {'IC_g':>7s} "
        f"{'IC_t30':>7s} {'Lift@30':>7s} {'Ret_t30':>9s} "
        f"{'Net_t30':>9s} {'Shrp_t30':>8s} {'Spr10v1':>9s} {'Spr_tb':>9s}"
    )
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
            f"{dm.get('Spread_10v1',0):>9.6f} {dm.get('Spread_t30vb30',0):>9.6f}{_win}"
        )
    log_lines.append(f"  {'─'*110}")

    return best_name, targets[best_name], log_lines


def _build_bin_mono_metrics(y_true, score, n_bins=10):
    y_true = np.asarray(y_true, dtype=float)
    score = np.asarray(score, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(score)
    y_true, score = y_true[mask], score[mask]
    if y_true.size < 20:
        return {
            "rho_bin_med": 0.0,
            "top_gt_mid_gt_bot": False,
            "top20_bot50_spread": 0.0,
        }
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
    mid = float(
        np.median(
            y_true[
                (score >= np.nanquantile(score, 0.45))
                & (score <= np.nanquantile(score, 0.55))
            ]
        )
    )
    bot = float(np.median(y_true[score <= np.nanquantile(score, 0.10)]))
    spread = float(
        np.median(y_true[score >= np.nanquantile(score, 0.80)])
        - np.median(y_true[score <= np.nanquantile(score, 0.50)])
    )
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


def _base_oof_degeneracy_summary(
    raw_probs,
    calibrated_probs,
    *,
    decimals: int = 8,
    min_unique: int = 3,
    min_std: float = 1e-4,
):
    raw = np.asarray(raw_probs, dtype=np.float64)
    cal = np.asarray(calibrated_probs, dtype=np.float64)
    raw = raw[np.isfinite(raw)]
    cal = cal[np.isfinite(cal)]

    def _stats(arr):
        if arr.size == 0:
            return {
                "n": 0,
                "unique_count": 0,
                "std": 0.0,
                "min": float("nan"),
                "max": float("nan"),
                "range": 0.0,
                "dominant_share": 1.0,
            }
        rounded = np.round(arr, decimals=decimals)
        uniq, counts = np.unique(rounded, return_counts=True)
        dom = float(np.max(counts) / max(arr.size, 1))
        arr_min = float(np.min(arr))
        arr_max = float(np.max(arr))
        return {
            "n": int(arr.size),
            "unique_count": int(len(uniq)),
            "std": float(np.std(arr)),
            "min": arr_min,
            "max": arr_max,
            "range": float(arr_max - arr_min),
            "dominant_share": dom,
        }

    raw_stats = _stats(raw)
    cal_stats = _stats(cal)
    reasons = []
    if cal_stats["unique_count"] < min_unique:
        reasons.append(f"cal_unique_lt_{min_unique}")
    if raw_stats["unique_count"] < min_unique:
        reasons.append(f"raw_unique_lt_{min_unique}")
    if cal_stats["std"] < min_std:
        reasons.append(f"cal_std_lt_{min_std:g}")
    if raw_stats["std"] < min_std:
        reasons.append(f"raw_std_lt_{min_std:g}")
    if cal_stats["dominant_share"] >= 0.995:
        reasons.append("cal_dominant_share_ge_0.995")
    if raw_stats["dominant_share"] >= 0.995:
        reasons.append("raw_dominant_share_ge_0.995")
    return {
        "is_degenerate": bool(reasons),
        "reasons": reasons,
        "raw": raw_stats,
        "calibrated": cal_stats,
        "thresholds": {"min_unique": int(min_unique), "min_std": float(min_std)},
    }


def _base_model_report_entry(
    model_name,
    side,
    kind,
    dm,
    y_bin,
    oof_probs,
    y_ret,
    groups,
    y_lbl=None,
    top5_features=None,
    top10_features=None,
):
    prev = float(np.mean(y_bin))
    prev = float(np.clip(prev, 1e-7, 1 - 1e-7))
    base_brier = prev * (1.0 - prev)
    base_ll = -(prev * np.log(prev) + (1 - prev) * np.log(1 - prev))
    # Recompute Brier/LL from OOF predictions (consistent with y_bin)
    from sklearn.metrics import brier_score_loss
    from sklearn.metrics import log_loss as _log_loss

    p_clip = np.clip(oof_probs, 1e-7, 1 - 1e-7)
    raw_probs = dm.get("oof_raw")
    if raw_probs is None:
        raw_probs = oof_probs
    raw_probs = np.asarray(raw_probs, dtype=float)
    raw_clip = np.clip(raw_probs, 1e-7, 1 - 1e-7)
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
        _rec = float(np.sum(y_bin[_idx]) / max(np.sum(y_bin), 1.0))
        info_metrics[f"prec_at_{int(_frac*100)}pct"] = _prec
        info_metrics[f"recall_at_{int(_frac*100)}pct"] = _rec
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
        pr_auc = (
            float(average_precision_score(y_bin_calc, oof_probs_safe))
            if len(np.unique(y_bin_calc)) > 1
            else 0.0
        )
    except Exception:
        pr_auc = 0.0

    # Prevalence-aware PR-AUC threshold (matching gate_metrics.py logic)
    # Threshold = max(1.25 * prev, prev + 0.05)
    # We remove the 0.50 floor because for low-prevalence (e.g. 0.35), 0.45 is a good score.
    prev_for_threshold = float(np.mean(y_bin_calc))
    pr_auc_threshold = max(1.25 * prev_for_threshold, prev_for_threshold + 0.05)

    # Diagnostic: PR-AUC below prevalence indicates model is worse than random
    if pr_auc < prev_for_threshold:
        tprint(
            f"WARNING: PR-AUC ({pr_auc:.4f}) < Prevalence ({prev_for_threshold:.4f}) for {model_name} - model is worse than random!"
        )
        tprint(f"  Lift@20%: {lift_k:.4f}")
        # Check if labels might be inverted
        if lift_k < 1.0 and prec_lift_abs < 0:
            tprint(
                f"  CRITICAL: Lift < 1.0 and precision lift negative - possible label inversion!"
            )

    checks = {
        "pr_auc_ge_threshold": pr_auc >= pr_auc_threshold,
        "pr_auc_ge_random": pr_auc >= prev_for_threshold,
        "brier_and_logloss_improve_ge_2pct": bool(
            (brier_imp >= 0.02) and (ll_imp >= 0.02)
        ),
        "liftk_and_preck_lift": bool(
            (lift_k >= 1.2) and ((prec_lift_abs >= 0.025) or ((lift_k - 1.0) >= 0.05))
        ),
        "bootstrap_prec20_cv_le_0_30": bootstrap_prec20_cv <= 0.30,
        "delta_logloss_le_minus_0_5pct": ll_imp >= 0.005,
        "logloss_improves_in_ge_70pct_folds": pos_fold_ratio >= 0.70,
        "worst_fold_delta_logloss_ge_0_5pct_improve": worst_fold_imp >= -0.005,
    }

    # Raw AUC and ICs for summary table.
    #
    # NOTE:
    # - For alpha classifiers, AUC is defined against y_bin (TP hit vs not-TP).
    # - Spearman vs y_ret can legitimately have opposite sign when return magnitudes
    #   are asymmetric (e.g., many correctly-ranked TP events but a few large losses).
    #
    # To avoid reporting confusion ("good AUC, negative IC"), we report:
    #   * ic:     rank correlation vs y_bin (classification-consistent IC)
    #   * ic_ret: rank correlation vs y_ret (economic utility orientation)
    from sklearn.metrics import roc_auc_score as _roc_auc_score

    try:
        raw_auc = (
            float(_roc_auc_score(y_bin, raw_clip))
            if len(np.unique(y_bin)) > 1
            else 0.5
        )
    except Exception:
        raw_auc = 0.5
    try:
        calibrated_auc = (
            float(_roc_auc_score(y_bin, p_clip)) if len(np.unique(y_bin)) > 1 else 0.5
        )
    except Exception:
        calibrated_auc = 0.5
    raw_ic_bin = _safe_spearman(oof_probs, y_bin)
    raw_ic_ret = _safe_spearman(oof_probs, y_ret)

    metrics = {
        "auc": calibrated_auc,
        "raw_auc": raw_auc,
        "calibrated_auc": calibrated_auc,
        "ic": raw_ic_bin,
        "ic_ret": raw_ic_ret,
        "brier": float(brier),
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
    degeneracy = dm.get("degeneracy", {})

    return {
        "model": model_name,
        "side": side,
        "kind": kind,
        "score": float(dm.get("rank_score", dm.get("score", 0.0))),
        "checks": {k: {"pass": bool(v), "emoji": _emoji(v)} for k, v in checks.items()},
        "metrics": metrics,
        "raw_auc": raw_auc,
        "calibrated_auc": calibrated_auc,
        "degenerate": bool(degeneracy.get("is_degenerate", False)),
        "degeneracy": degeneracy,
        "passed": bool(all(checks.values())),
        "top5_features": list(top5_features or []),
        "top10_features": list(top10_features or []),
    }


def _meta_report_entry(
    name, meta_model, y_target, y_ret, base_score, groups, y_per_horizon=None
):
    pred = int(getattr(meta_model, "score_sign", 1)) * np.asarray(
        meta_model.oof_probs, dtype=float
    )
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
        return float(np.mean(np.maximum(a * e, (a - 1.0) * e)))

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
    fold_ics = (
        _fold_stats_from_groups(y_target, pred, groups, _safe_spearman)
        if groups is not None
        else []
    )
    pos_ic_ratio = float(np.mean(np.array(fold_ics) > 0)) if fold_ics else 0.0
    stable_sign = True
    if fold_ics:
        signs = np.sign(fold_ics)
        stable_sign = bool(np.mean(signs == np.sign(np.mean(fold_ics))) >= 0.7)

    mono = _build_bin_mono_metrics(y_ret, pred, n_bins=10)
    mad_y = _mad(y_ret)
    spread_ok = (mono["top20_bot50_spread"] >= 0.25 * mad_y) or (
        mono["top20_bot50_spread"] > 0
    )

    # Enforcing strict Top-30% Turnover limits per new requirement
    k30 = max(1, int(0.30 * len(pred)))
    idx_meta = np.argsort(pred)[-k30:]
    idx_meta_flip = np.argsort(-pred)[-k30:]
    idx_base = np.argsort(base_score)[-k30:]

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
        checks.update(
            {
                "coverage_tau": abs(cov - tau) <= 0.05,
                "pinball_improve_ge_2pct": pb_imp >= 0.02,
                "pinball_improve_ge_2of3_folds": (np.mean(fold_sign) >= (2 / 3))
                if fold_sign
                else False,
            }
        )
    else:
        # proxy robust-loss/bias checks based on oof residuals
        res = pred - y_target
        loss = float(np.mean(np.abs(res)))
        loss_base = float(
            np.mean(np.abs(np.full_like(y_target, np.median(y_target)) - y_target))
        )
        loss_imp = (loss_base - loss) / max(loss_base, 1e-9)
        fold_loss = (
            _fold_stats_from_groups(
                y_target,
                pred,
                groups,
                lambda yt, pr: (
                    np.mean(np.abs(np.median(yt) - yt)) - np.mean(np.abs(pr - yt))
                )
                / max(np.mean(np.abs(np.median(yt) - yt)), 1e-9),
            )
            if groups is not None
            else []
        )
        mean_err = float(np.mean(res))
        bias_fold = (
            _fold_stats_from_groups(
                y_target, pred, groups, lambda yt, pr: np.mean(pr - yt)
            )
            if groups is not None
            else []
        )
        mad_t = _mad(y_target)
        checks.update(
            {
                "robust_loss_ge_2pct": loss_imp >= 0.02,
                "robust_loss_ge_2of3_folds": (np.mean(np.array(fold_loss) > 0) >= 2 / 3)
                if fold_loss
                else False,
                "robust_loss_worst_fold_ge_1pct": (np.min(fold_loss) >= 0.01)
                if fold_loss
                else False,
                "bias_overall": abs(mean_err) <= 0.05 * mad_t,
                "bias_per_fold": (np.max(np.abs(bias_fold)) <= 0.07 * mad_t)
                if bias_fold
                else False,
            }
        )

    checks.update(
        {
            "spearman_ic_ge_0_03": ic >= 0.03,
            "ic_stable_sign": stable_sign,
            "ic_pos_ge_70pct_folds": pos_ic_ratio >= 0.70,
            "bin_monotonicity_ge_0_9": mono["rho_bin_med"] >= 0.90,
            "top_mid_bottom_ordering": mono["top_gt_mid_gt_bot"],
            "top20_bottom50_spread": spread_ok,
            "es30_meta_vs_base": es_ok,
            "net_return_vs_no_meta": net_meta > net_base,
            "sortino_vs_no_meta": sort_meta > sort_base,
        }
    )

    # Compute comprehensive OOF metrics for reporting table (against meta target)
    from .meta_model import MetaModel as _MM

    _oof_metrics = _MM._compute_oof_metrics(pred, y_target, y_per_horizon=y_per_horizon)

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
            "es30_meta": es_meta,
            "es30_base": es_base,
            "net_return_meta": net_meta,
            "net_return_meta_if_flipped": net_meta_flip,
            "net_return_base": net_base,
            "sortino_meta": sort_meta,
            "sortino_meta_if_flipped": sort_meta_flip,
            "sortino_base": sort_base,
            "direction_flip_improves": bool(
                (net_meta_flip > net_meta) and (sort_meta_flip > sort_meta)
            ),
            **info_policy,
            **_oof_metrics,
        },
    }


def save_training_gate_report(report_payload, cfg, run_id=None):
    reports_dir = (
        cfg.get("reports_root")
        or os.environ.get("EPM_REPORTS_DIR")
        or os.path.join("extreme_price_movements", "reports")
    )
    os.makedirs(reports_dir, exist_ok=True)
    rid = run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(reports_dir, f"training_gate_report_{rid}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=2)
    return out_path


def save_quality_gate_artifacts(
    report_payload,
    cfg,
    run_id,
    base_quality_rows=None,
    meta_quality_rows=None,
):
    data_root = str(cfg.get("data_root", "data"))
    out_dir = os.path.join(data_root, "artifacts", str(run_id), "quality_reports")
    os.makedirs(out_dir, exist_ok=True)

    gate_path = os.path.join(out_dir, "training_gate_report.json")
    with open(gate_path, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=2)

    def _safe_normalize(rows, filename):
        if not rows:
            return
        try:
            # Use json_normalize which handles nested dicts by creating dot-separated columns
            df = pd.json_normalize(rows)
            # Find any columns that contain un-flattened lists/arrays (can cause ValueError on CSV save)
            for col in df.columns:
                if df[col].apply(lambda x: isinstance(x, (list, np.ndarray))).any():
                    df[col] = df[col].apply(
                        lambda x: str(x) if isinstance(x, (list, np.ndarray)) else x
                    )
            df.to_csv(os.path.join(out_dir, filename), index=False)

            json_filename = filename.replace(".csv", ".json")
            df.to_json(os.path.join(out_dir, json_filename), orient="records", indent=2)
        except Exception as exc:
            tprint(
                f"WARNING: Comprehensive artifact saving failed for {filename}: {exc}. Falling back to basic DataFrame."
            )
            try:
                # Fallback: simple DataFrame constructor (may still fail if columns are inconsistent)
                pd.DataFrame(rows).to_csv(os.path.join(out_dir, filename), index=False)
            except Exception as exc2:
                tprint(
                    f"ERROR: Fallback artifact saving also failed for {filename}: {exc2}"
                )

    _safe_normalize(base_quality_rows, "base_model_detailed_metrics.csv")
    _safe_normalize(meta_quality_rows, "meta_model_detailed_metrics.csv")

    return gate_path


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
        hdr = f"{'Model':<32} {'RawAUC':>7} {'CalAUC':>7} {'IC_bin':>7} {'IC_ret':>7} {'LogLoss':>8} {'PR-AUC':>7} {'Lift@20':>8} {'BrierImp':>9} {'Deg':>4}"
        tprint(hdr)
        tprint("-" * len(hdr))
        for it in winners:
            m = it.get("metrics", {})
            name = it.get("model", "?")
            raw_auc = m.get("raw_auc", m.get("auc", float("nan")))
            cal_auc = m.get("calibrated_auc", m.get("auc", float("nan")))
            ic = m.get("ic", float("nan"))
            ic_ret = m.get("ic_ret", float("nan"))
            ll = m.get("logloss", float("nan"))
            prauc = m.get("pr_auc", float("nan"))
            lift = m.get("lift_at_20pct", float("nan"))
            brimp = m.get("brier_improvement", float("nan"))
            deg = "Y" if it.get("degenerate", False) else "N"
            tprint(
                f"{name:<32} {raw_auc:>7.4f} {cal_auc:>7.4f} {ic:>7.4f} {ic_ret:>7.4f} {ll:>8.4f} {prauc:>7.4f} {lift:>8.3f} {brimp:>8.1%} {deg:>4}"
            )
    blocked = report_payload.get("blocked_strategy_ids", [])
    if blocked:
        tprint(
            f"\nBlocked downstream strategy_ids ({len(blocked)}): "
            + ", ".join(str(x) for x in blocked)
        )
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
            tprint(
                f"{name:<20} {ic:>7.4f} {icmh:>7.4f} {ic30:>7.4f} {ece:>8.4f} {s10:>8.6f} {s30:>8.6f} {wr30:>7.1%} {rl30:>8.1%} {nt:>7d}"
            )
    else:
        tprint("\n--- META MODELS (Regressors): none ---")

    # ── Meta classifier table ─────────────────────────────────────────
    if clf_items:
        tprint("\n--- META MODELS (Move Classifiers) ---")
        hdr = (
            f"{'Model':<20} {'Winner':<16} {'Thr':>5} {'MoveAUC':>8} "
            f"{'PR':>7} {'PRΔ':>7} {'IC':>7} {'Bal0.5':>7} {'Bal*':>7} "
            f"{'ECE10':>7} {'Brier':>7} {'BaseR':>7} {'Slope':>7} "
            f"{'Int':>7} {'Top10L':>8} {'Top10P':>7} {'Top10R':>7} "
            f"{'Top5L':>7} {'Top20L':>7}"
        )
        tprint(hdr)
        tprint("-" * len(hdr))
        for it in clf_items:
            m = it.get("metrics", {})
            name = it.get("model", "?")
            winner = m.get("clf_winner", "?")
            thr = m.get("clf_threshold_pct", 0)
            auc = m.get("move_roc_auc", m.get("roc_auc", float("nan")))
            prauc = m.get("move_pr_auc", m.get("pr_auc", float("nan")))
            base_rate = m.get("move_base_rate", m.get("base_rate", float("nan")))
            pr_delta = m.get("pr_auc_vs_base_rate", float("nan"))
            move_ic = m.get("move_ic", float("nan"))
            bal05 = m.get("move_balanced_accuracy_0p5", m.get("balanced_accuracy_0p5", float("nan")))
            balbest = m.get("move_balanced_accuracy_best", m.get("balanced_accuracy_best", float("nan")))
            ece = m.get("ece_10", m.get("ece", float("nan")))
            slope = m.get("calibration_slope", float("nan"))
            inter = m.get("calibration_intercept", float("nan"))
            top10l = m.get("top10_lift", float("nan"))
            top10p = m.get("top_decile_precision", m.get("top10_hit_rate", float("nan")))
            top10r = m.get("top_decile_recall", float("nan"))
            brier = m.get("brier", float("nan"))
            top5l = m.get("top05_lift", float("nan"))
            top20l = m.get("top20_lift", float("nan"))
            tprint(
                f"{name:<20} {winner:<16} {thr:>4d}% {auc:>8.4f} {prauc:>7.4f} {pr_delta:>7.4f} "
                f"{move_ic:>7.4f} {bal05:>7.3f} {balbest:>7.3f} {ece:>7.3f} {brier:>7.4f} "
                f"{base_rate:>7.3f} {slope:>7.3f} {inter:>7.3f} {top10l:>8.3f} "
                f"{top10p:>7.1%} {top10r:>7.1%} {top5l:>7.3f} {top20l:>7.3f}"
            )
    else:
        tprint("\n--- META MODELS (Move Classifiers): none ---")

    tprint("=" * 100)


OUT_SL = np.int8(0)
OUT_TO = np.int8(1)
OUT_TP = np.int8(2)


def _stable_cfg_subset_hash(
    cfg: dict, horizons: list[int], trade_sides: list[str]
) -> str:
    """Stable hash for TB/geometry cache invalidation across equivalent configs."""
    cache_keys = [
        "label_round_trip_fee_pct",
        "label_min_tp_hit_rate",
        "label_min_tp_hit_rate_h2",
        "label_max_timeout_rate",
        "label_max_timeout_rate_h2",
        "label_min_net_rr",
        "label_min_net_rr_h2",
        "label_min_events_h2",
        "label_h2_rescue_topk",
        "label_h4_rescue_topk",
        "barrier_disp_floor",
        "barrier_z_max",
        "barrier_k_reg",
        "barrier_m_lo",
        "barrier_m_hi",
        "barrier_sl_lo",
        "barrier_sl_hi",
        "barrier_z_gate",
        "label_horizon_base",
        "barrier_tp_lo",
        "barrier_tp_lo_h2",
        "barrier_tp_hi",
        "barrier_k_tp_grid",
        "barrier_sl_base_grid",
        "label_use_production_tp_floor",
        "use_dynamic_horizon",
        "dynamic_horizon_z_lo",
        "dynamic_horizon_z_hi",
        "dynamic_horizon_max_scale_add",
        "use_standalone_sl",
    ]
    payload = {
        "cfg": {k: cfg.get(k) for k in cache_keys},
        "horizons": [int(h) for h in horizons],
        "trade_sides": [str(s) for s in trade_sides],
        "label_logic_version": int(cfg.get("label_logic_version", 2)),
        "py": platform.python_version(),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:16]


def _tb_cache_dir(cfg: dict, run_id: str) -> str:
    return os.path.join(cfg["data_root"], "artifacts", run_id, "labels", "tb_cache")


def _tb_cache_paths(
    cfg: dict, run_id: str, H: int, strategy_id: str, config_hash: str
) -> tuple[str, str, str]:
    root = _tb_cache_dir(cfg, run_id)
    stem = f"H{int(H)}_{str(strategy_id)}_{config_hash}"
    return (
        os.path.join(root, f"tb_{stem}.pkl"),
        os.path.join(root, f"geom_{stem}.pkl"),
        os.path.join(root, f"events_{stem}.npz"),
    )


def _save_event_index_artifact(
    path: str, event_ts, event_sym, symbol_vocab: dict[str, int]
) -> None:
    if event_ts is None or event_sym is None or len(event_ts) == 0:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ts_ns = pd.to_datetime(event_ts, utc=True).view("i8")
    sym_ids = np.asarray(
        [symbol_vocab.get(str(s), -1) for s in event_sym], dtype=np.int32
    )
    np.savez_compressed(path, entry_ts_ns=ts_ns.astype(np.int64), symbol_id=sym_ids)


def _choose_parallel_cells(n_cells: int, cfg: dict) -> int:
    if n_cells <= 1:
        return 1
    max_workers = int(cfg.get("label_parallel_max_workers", 3))
    # Default to single-worker on Apple Silicon because Numba's workqueue backend
    # is not thread-safe under nested Python-thread fan-out.
    if platform.system() == "Darwin" and platform.machine().lower() in {
        "arm64",
        "aarch64",
    }:
        max_workers = min(max_workers, int(cfg.get("label_parallel_max_workers_m1", 1)))
    max_workers = max(1, max_workers)
    return min(max_workers, n_cells)


def _downcast_label_dataset_df(df: pd.DataFrame, copy: bool = True) -> pd.DataFrame:
    """Downcast generated label datasets to compact dtypes (float32/int32) where safe."""
    if not isinstance(df, pd.DataFrame) or df.empty:
        return df
    out = df.copy() if copy else df
    for col in out.columns:
        s = out[col]
        if pd.api.types.is_float_dtype(s.dtype):
            out[col] = s.astype(np.float32, copy=False)
        elif pd.api.types.is_integer_dtype(s.dtype):
            if str(col).startswith("__y") or str(col).endswith("_id"):
                out[col] = s.astype(np.int32, copy=False)
            else:
                out[col] = pd.to_numeric(s, downcast="integer")
    return out


def subsample_symbol_balanced(
    df: pd.DataFrame,
    max_rows: int,
    symbol_col: str = "__symbol__",
    ts_col: str = "__ts__",
    rng_seed: int = 42,
) -> pd.DataFrame:
    if len(df) <= max_rows:
        return df
    if symbol_col not in df.columns:
        return (
            df.sample(n=max_rows, random_state=rng_seed)
            .sort_values(ts_col if ts_col in df.columns else df.columns[0])
            .reset_index(drop=True)
        )
    symbols = df[symbol_col].unique()
    n_symbols = len(symbols)
    per_symbol = max(1, max_rows // n_symbols)
    rng = np.random.RandomState(rng_seed)
    parts: list[pd.DataFrame] = []
    for sym in symbols:
        sym_df = df[df[symbol_col] == sym]
        if len(sym_df) <= per_symbol:
            parts.append(sym_df)
        else:
            parts.append(
                sym_df.sample(n=per_symbol, random_state=rng).reset_index(drop=True)
            )
    out = pd.concat(parts, axis=0, ignore_index=True)
    if len(out) > max_rows:
        out = out.sample(n=max_rows, random_state=rng_seed).reset_index(drop=True)
    if ts_col in out.columns:
        out = out.sort_values(ts_col).reset_index(drop=True)
    return out


def _subsample_indices_time_balanced(
    n_rows: int, max_rows: int, y: Optional[np.ndarray] = None
) -> np.ndarray:
    if n_rows <= max_rows:
        return np.arange(n_rows, dtype=np.int32)
    base_idx = np.linspace(0, n_rows - 1, max_rows, dtype=np.int32)
    if y is None:
        return np.unique(base_idx)
    y_arr = np.asarray(y)
    pos_idx = np.flatnonzero(y_arr >= 0.5)
    neg_idx = np.flatnonzero(y_arr < 0.5)
    keep_pos = min(len(pos_idx), max_rows // 2)
    keep_neg = min(len(neg_idx), max_rows - keep_pos)
    pos_take = (
        pos_idx[np.linspace(0, len(pos_idx) - 1, keep_pos, dtype=np.int32)]
        if keep_pos > 0
        else np.array([], dtype=np.int32)
    )
    neg_take = (
        neg_idx[np.linspace(0, len(neg_idx) - 1, keep_neg, dtype=np.int32)]
        if keep_neg > 0
        else np.array([], dtype=np.int32)
    )
    out = np.unique(np.concatenate([base_idx, pos_take, neg_take]))
    if len(out) > max_rows:
        out = out[np.linspace(0, len(out) - 1, max_rows, dtype=np.int32)]
    return out.astype(np.int32, copy=False)


def _bounded_sample_cap(
    n_rows: int, absolute_cap: int | None = None, pct_cap: float | None = None
) -> int:
    cap = int(n_rows)
    if absolute_cap is not None and int(absolute_cap) > 0:
        cap = min(cap, int(absolute_cap))
    if pct_cap is not None and np.isfinite(pct_cap) and float(pct_cap) > 0.0:
        pct_rows = int(np.ceil(float(n_rows) * min(float(pct_cap), 1.0)))
        if pct_rows > 0:
            cap = min(cap, pct_rows)
    return max(1, int(cap))


def _downcast_tb_triplet(tb_triplet):
    """Normalize TB cache payload dtypes for memory and serialization stability."""
    tb_labels, tb_returns, tb_quality = tb_triplet
    if isinstance(tb_labels, pd.DataFrame):
        tb_labels = tb_labels.astype(np.int8, copy=False)
    if isinstance(tb_returns, pd.DataFrame):
        tb_returns = tb_returns.astype(np.float32, copy=False)
    if isinstance(tb_quality, pd.DataFrame):
        tb_quality = tb_quality.astype(np.float32, copy=False)
    return (tb_labels, tb_returns, tb_quality)


def _downcast_geom_payload(geom_payload: dict):
    """Normalize geometry cache payload dtypes for memory and serialization stability."""
    if not isinstance(geom_payload, dict):
        return geom_payload
    out = dict(geom_payload)
    for k in ("tp_vals", "sl_vals"):
        df = out.get(k)
        if isinstance(df, pd.DataFrame):
            out[k] = df.astype(np.float32, copy=False)
    if "n_geom" in out:
        try:
            out["n_geom"] = int(out["n_geom"])
        except Exception:
            pass
    return out


class AlphaHorizonEnsemble:
    """Average probabilities across multiple horizon-specific alpha models."""

    def __init__(self, members):
        # members: list of dict(model, feat_cols, H, weight, oof_probs)
        self.members = members
        self.oof_probs = None
        if members:
            oofs = [
                m.get("oof_probs") for m in members if m.get("oof_probs") is not None
            ]
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


def compute_meta_target(
    ret_h1: np.ndarray,
    ret_h2: np.ndarray,
    ret_h4: np.ndarray,
    vol_proxy: Optional[np.ndarray] = None,
    groups=None,
) -> np.ndarray:
    """Build a per-trade meta target as risk-normalized, squashed log-return.

    Uses weighted average of per-horizon log-returns [0.25, 0.40, 0.35] for [H1, H2, H4].
    If vol_proxy (ATR) is provided, returns are normalized to risk units (approx Z-score)
    before averaging. Finally, apply monotone squashing (asinh) to handle tails robustly
    without hard clipping.

    This target is NOT cross-sectional rank (we want to trade every profitable asset,
    not just the best one). It preserves the key property: bigger positive return = better trade.
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
        return out.astype(np.float64)

    r1 = _log1p_ret(ret_h1)
    r2 = _log1p_ret(ret_h2)
    r4 = _log1p_ret(ret_h4)

    if vol_proxy is not None:
        assert len(vol_proxy) == len(
            r1
        ), f"vol_proxy length mismatch: {len(vol_proxy)} vs {len(r1)}"
        vp = np.clip(np.asarray(vol_proxy, dtype=float), 1e-4, None)
        # Normalize by expected volatility over horizon H: sigma_H = atr_1h * sqrt(H)
        # Using sqrt(H) scaling assumes diffusive variance.
        n1 = r1 / (vp * np.sqrt(1.0))
        n2 = r2 / (vp * np.sqrt(2.0))
        n4 = r4 / (vp * np.sqrt(4.0))
        raw = 0.25 * n1 + 0.40 * n2 + 0.35 * n4
        # Guard against any residual inf/nan before arcsinh
        raw = np.where(np.isfinite(raw), raw, 0.0)
        raw = np.clip(raw, -1e6, 1e6)

        # Monotone squashing: asinh(x/c) with c=2.0 (risk units)
        # 2.0 risk units = 2 sigma event.
        c = 2.0
        target = np.arcsinh(raw / c)
    else:
        # Fallback to legacy logic if no vol_proxy
        raw = 0.25 * r1 + 0.40 * r2 + 0.35 * r4
        finite = np.isfinite(raw)
        if finite.sum() > 10:
            lo = float(np.percentile(raw[finite], 5))
            hi = float(np.percentile(raw[finite], 90))
            raw = np.where(raw < lo, lo, raw)
            above = raw > hi
            if above.any():
                scale = max(float(np.std(raw[finite])), 1e-9)
                excess = (raw[above] - hi) / scale
                raw[above] = hi + scale * np.sqrt(excess)
        target = raw

    return target.astype(np.float32)


def build_horizon_prediction_features(conf: dict, X_eval: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=X_eval.index)
    models_by_h = conf.get("models_by_h", {}) if conf else {}
    for H in CANON_HORIZONS:
        if H in models_by_h:
            m = models_by_h[H]
            Xi = X_eval.reindex(columns=m.get("feat_cols", []), fill_value=0.0)
            out[f"pred_H{H}"] = m["model"].predict_proba(Xi)[:, 1]
        else:
            out[f"pred_H{H}"] = 0.5
    return out


def _aggregate_alpha_oof_metrics(
    y_true, probs, returns, sample_weight=None, groups=None
):
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
    metrics.update(
        {
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
            "prec30": float(sel_metrics.get("Prec_Top30", np.nan)),
            "lift30": float(sel_metrics.get("Lift_Top30", np.nan)),
        }
    )

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


def compute_per_regime_metrics(
    y_true, y_prob, df, sample_weight=None, global_prev=None
):
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
    from sklearn.metrics import brier_score_loss, roc_auc_score

    regime_cols = [
        c for c in df.columns if c.startswith("__regime_") and c.endswith("__")
    ]
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
                regime_results[bucket_name] = {
                    "bss": 0.0,
                    "bss_global": 0.0,
                    "auc": 0.5,
                    "brier": 0.0,
                    "n": n_bucket,
                }
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
                "bss": round(bss, 4),
                "bss_global": round(bss_global, 4),
                "auc": round(auc, 4),
                "brier": round(brier_basic, 4),
                "n": n_bucket,
            }

        results[regime_name] = regime_results

    return results


def _fast_lookup(feat_df, event_ts, event_sym):
    """Fast extraction of values at (ts, sym) positions using numpy indexing.
    Returns 1D array of values. NaN where lookup fails."""
    feat_index = feat_df.index
    if isinstance(feat_index, pd.DatetimeIndex):
        try:
            if feat_index.tz is None:
                row_keys = pd.DatetimeIndex(pd.to_datetime(event_ts)).tz_localize(None)
            else:
                row_keys = pd.DatetimeIndex(
                    pd.to_datetime(event_ts, utc=True)
                ).tz_convert(feat_index.tz)
            row_idx = feat_index.get_indexer(row_keys)
        except Exception:
            row_idx = feat_index.get_indexer(event_ts)
    else:
        row_idx = feat_index.get_indexer(event_ts)
    col_idx = feat_df.columns.get_indexer(event_sym)
    vals = feat_df.values
    # Mark invalid positions
    valid = (row_idx >= 0) & (col_idx >= 0)
    out = np.full(len(event_ts), np.nan, dtype=np.float32)
    if valid.any():
        out[valid] = vals[row_idx[valid], col_idx[valid]]
    return out


def _datetime_index_to_ns(index: pd.DatetimeIndex) -> np.ndarray:
    if getattr(index, "tz", None) is None:
        return index.view("i8")
    return index.tz_convert("UTC").tz_localize(None).view("i8")


def _event_ts_to_ns(event_ts, target_tz=None) -> np.ndarray:
    ts = pd.to_datetime(event_ts, utc=True)
    if target_tz is not None:
        ts = ts.tz_convert(target_tz)
    else:
        ts = ts.tz_localize(None)
    if getattr(ts, "tz", None) is not None:
        ts = ts.tz_localize(None)
    return ts.view("i8")


def _resolve_row_indexer(
    feat_index,
    event_ts,
    lookup_cache: Optional[dict] = None,
):
    cache_key = ("row", id(feat_index), id(event_ts))
    if lookup_cache is not None and cache_key in lookup_cache:
        return lookup_cache[cache_key]

    if isinstance(feat_index, pd.DatetimeIndex) and feat_index.is_monotonic_increasing:
        idx_ns = _datetime_index_to_ns(feat_index)
        key_ns = _event_ts_to_ns(event_ts, target_tz=getattr(feat_index, "tz", None))
        pos = np.searchsorted(idx_ns, key_ns)
        valid = (pos >= 0) & (pos < len(idx_ns))
        row_idx = np.full(len(key_ns), -1, dtype=np.int32)
        if valid.any():
            exact = idx_ns[pos[valid]] == key_ns[valid]
            row_idx_valid = pos[valid].astype(np.int32, copy=False)
            row_idx[valid] = np.where(exact, row_idx_valid, -1)
    else:
        row_idx = feat_index.get_indexer(event_ts).astype(np.int32, copy=False)

    if lookup_cache is not None:
        lookup_cache[cache_key] = row_idx
    return row_idx


def _resolve_col_indexer(
    columns: pd.Index,
    event_sym,
    lookup_cache: Optional[dict] = None,
):
    cache_key = ("col", id(columns), id(event_sym))
    if lookup_cache is not None and cache_key in lookup_cache:
        return lookup_cache[cache_key]

    sym_arr = np.asarray(event_sym, dtype=object)
    uniq_syms, inv = np.unique(sym_arr, return_inverse=True)
    uniq_idx = columns.get_indexer(uniq_syms).astype(np.int32, copy=False)
    col_idx = uniq_idx[inv]
    if lookup_cache is not None:
        lookup_cache[cache_key] = col_idx
    return col_idx


def _fast_lookup_cached(
    feat_df,
    event_ts,
    event_sym,
    lookup_cache: Optional[dict] = None,
):
    row_idx = _resolve_row_indexer(feat_df.index, event_ts, lookup_cache=lookup_cache)
    col_idx = _resolve_col_indexer(
        feat_df.columns, event_sym, lookup_cache=lookup_cache
    )
    vals = feat_df.values
    valid = (row_idx >= 0) & (col_idx >= 0)
    out = np.full(len(row_idx), np.nan, dtype=np.float32)
    if valid.any():
        out[valid] = vals[row_idx[valid], col_idx[valid]]
    return out


def _fast_series_lookup_cached(
    series: pd.Series,
    event_ts,
    lookup_cache: Optional[dict] = None,
) -> np.ndarray:
    row_idx = _resolve_row_indexer(series.index, event_ts, lookup_cache=lookup_cache)
    vals = np.asarray(series.values)
    out = np.full(len(row_idx), np.nan, dtype=np.float32)
    valid = row_idx >= 0
    if valid.any():
        out[valid] = vals[row_idx[valid]]
    return out


def _align_values_by_ts_symbol_keys(
    union_ts,
    union_sym,
    source_ts,
    source_sym,
    source_vals,
    fill_value=np.nan,
    dtype=np.float32,
) -> np.ndarray:
    union_ts_ns = pd.to_datetime(union_ts, utc=True, errors="coerce").view("i8")
    source_ts_ns = pd.to_datetime(source_ts, utc=True, errors="coerce").view("i8")
    union_sym_arr = np.asarray(union_sym, dtype=object)
    source_sym_arr = np.asarray(source_sym, dtype=object)
    source_vals_arr = np.asarray(source_vals, dtype=dtype)

    if (
        len(union_ts_ns) == 0
        or len(source_ts_ns) == 0
        or len(source_sym_arr) == 0
        or len(source_vals_arr) == 0
    ):
        return np.full(len(union_ts_ns), fill_value, dtype=dtype)

    all_syms = np.concatenate([source_sym_arr, union_sym_arr])
    codes, _ = pd.factorize(all_syms, sort=False)
    source_sym_codes = codes[: len(source_sym_arr)].astype(np.int32, copy=False)
    union_sym_codes = codes[len(source_sym_arr) :].astype(np.int32, copy=False)

    key_dtype = np.dtype([("ts", np.int64), ("sym", np.int32)])
    source_keys = np.empty(len(source_ts_ns), dtype=key_dtype)
    source_keys["ts"] = source_ts_ns.astype(np.int64, copy=False)
    source_keys["sym"] = source_sym_codes
    union_keys = np.empty(len(union_ts_ns), dtype=key_dtype)
    union_keys["ts"] = union_ts_ns.astype(np.int64, copy=False)
    union_keys["sym"] = union_sym_codes

    order = np.argsort(source_keys, kind="mergesort")
    source_keys_sorted = source_keys[order]
    source_vals_sorted = source_vals_arr[order]
    if len(source_keys_sorted) > 1:
        keep = np.ones(len(source_keys_sorted), dtype=bool)
        keep[1:] = source_keys_sorted[1:] != source_keys_sorted[:-1]
        source_keys_sorted = source_keys_sorted[keep]
        source_vals_sorted = source_vals_sorted[keep]

    pos = np.searchsorted(source_keys_sorted, union_keys)
    valid = (pos >= 0) & (pos < len(source_keys_sorted))
    out = np.full(len(union_keys), fill_value, dtype=dtype)
    if valid.any():
        exact = source_keys_sorted[pos[valid]] == union_keys[valid]
        pos_valid = pos[valid]
        if np.any(exact):
            valid_idx = np.flatnonzero(valid)
            out_idx = valid_idx[exact]
            out[out_idx] = source_vals_sorted[pos_valid[exact]]
    return out


def _ts_symbol_string_keys(ts_vals, sym_vals) -> np.ndarray:
    ts_ns = pd.to_datetime(ts_vals, utc=True, errors="coerce").view("i8").astype(str)
    sym_str = np.asarray(sym_vals, dtype=str)
    return np.char.add(np.char.add(ts_ns, "|"), sym_str)


def _unique_preserve_order(arr: np.ndarray) -> np.ndarray:
    if len(arr) == 0:
        return arr
    _, first_idx = np.unique(arr, return_index=True)
    return arr[np.sort(first_idx)]


def _indexer_by_string_keys(
    target_keys: np.ndarray, source_keys: np.ndarray
) -> np.ndarray:
    if len(target_keys) == 0 or len(source_keys) == 0:
        return np.full(len(target_keys), -1, dtype=np.int32)
    uniq_source, first_idx = np.unique(source_keys, return_index=True)
    pos = np.searchsorted(uniq_source, target_keys)
    valid = (pos >= 0) & (pos < len(uniq_source))
    out = np.full(len(target_keys), -1, dtype=np.int32)
    if valid.any():
        exact = uniq_source[pos[valid]] == target_keys[valid]
        pos_valid = pos[valid]
        valid_idx = np.flatnonzero(valid)
        if np.any(exact):
            out_idx = valid_idx[exact]
            out[out_idx] = first_idx[pos_valid[exact]].astype(np.int32, copy=False)
    return out


def _cap_selected_features(
    selected_feats,
    available_cols,
    target_cap: int,
    min_features: int = 1,
):
    available_set = set(available_cols)
    out = []
    seen = set()
    for feat in selected_feats or []:
        if feat in available_set and feat not in seen:
            out.append(str(feat))
            seen.add(feat)
            if len(out) >= target_cap:
                break
    if len(out) >= min_features:
        return out
    fallback = []
    for feat in available_cols:
        if feat not in seen:
            fallback.append(str(feat))
            if len(out) + len(fallback) >= max(
                min_features, min(target_cap, len(available_cols))
            ):
                break
    return (out + fallback)[: max(min_features, min(target_cap, len(available_cols)))]


def _mdi_top_feature_lists(sel_res, selected_feats: list[str]) -> tuple[list[str], list[str]]:
    """Return selector top-5 and top-10 lists, preferring the MDI ranking table."""
    # The selector already returns a ranked feature list in `selected_feats`.
    # Some selector artifacts also expose a `metrics_table`, but that table can
    # be a separate diagnostic view and is not reliable as the source of the
    # final model feature ordering. Use the selected feature order directly.
    ranked = [str(v) for v in selected_feats if isinstance(v, str)]
    if len(ranked) < 10:
        try:
            metrics_table = getattr(sel_res, "metrics_table", None)
            if metrics_table is not None and not getattr(metrics_table, "empty", True):
                df = metrics_table.copy()
                if "final_rank" in df.columns:
                    df = df.sort_values("final_rank", ascending=True)
                elif "final_score" in df.columns:
                    df = df.sort_values("final_score", ascending=False)
                for idx in df.index.tolist():
                    if isinstance(idx, str) and idx not in ranked:
                        ranked.append(str(idx))
                        if len(ranked) >= 10:
                            break
        except Exception:
            pass

    return ranked[:5], ranked[:10]


def apply_interaction_toggles(df: pd.DataFrame, causal_cols, gate_cols, drop_raw=True):
    new_cols = {}
    for g in gate_cols:
        if g not in df.columns:
            continue
        for col in causal_cols:
            if col in df.columns:
                new_cols[f"{col}_{g}_0"] = df[col] * (1 - df[g])
                new_cols[f"{col}_{g}_1"] = df[col] * df[g]

    if new_cols:
        out = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    else:
        out = df.copy()

    if drop_raw:
        out = out.drop(
            columns=[c for c in causal_cols if c in out.columns], errors="ignore"
        )
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
    if isinstance(ts_vals, pd.DatetimeIndex):
        ts = _datetime_index_to_ns(ts_vals).astype(np.int64, copy=False)
    else:
        ts_arr = np.asarray(ts_vals)
        if np.issubdtype(ts_arr.dtype, np.datetime64):
            ts = ts_arr.astype("datetime64[ns]").view("i8").astype(np.int64, copy=False)
        else:
            try:
                ts = (
                    pd.to_datetime(ts_arr, utc=True, errors="coerce")
                    .view("i8")
                    .astype(np.int64, copy=False)
                )
            except Exception:
                ts = ts_arr
    if len(w) == 0:
        return w
    if np.issubdtype(np.asarray(ts).dtype, np.integer):
        uniq, inv = np.unique(np.asarray(ts, dtype=np.int64), return_inverse=True)
        counts = np.bincount(inv).astype(np.int32, copy=False)
        sums = np.bincount(inv, weights=np.where(np.isfinite(w), w, 0.0))
        valid_groups = counts > 0
        scale = np.ones(len(uniq), dtype=np.float64)
        pos = valid_groups & (sums > 0.0)
        scale[pos] = 1.0 / sums[pos]
        scale[valid_groups & ~pos] = 1.0 / counts[valid_groups & ~pos]
        w = w * scale[inv]
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
    fast_hl = int(cfg.get("atr_norm_fast_hl_hours", 24))
    slow_hl = int(cfg.get("atr_norm_slow_hl_hours", 24 * 5))
    global_hl = int(cfg.get("atr_norm_global_hl_hours", 24 * 5))
    warmup_h = int(cfg.get("atr_norm_warmup_hours", 24 * 10))
    g_lo, g_hi = cfg.get("atr_norm_clip_global", [0.7, 1.5])
    m_lo, m_hi = cfg.get("atr_norm_clip_scale", [0.6, 2.5])

    atr = atr_pct_df.astype(np.float32)
    ewm_fast = atr.ewm(halflife=fast_hl, min_periods=12).mean()
    ewm_slow = atr.ewm(halflife=slow_hl, min_periods=12).mean()
    atr_ewm = atr.ewm(halflife=global_hl, min_periods=12).mean()

    warmup_n = min(len(atr_ewm), max(24, warmup_h))
    atr_ref = (
        float(np.nanmedian(atr_ewm.iloc[:warmup_n].values))
        if warmup_n > 0
        else float(np.nanmedian(atr_ewm.values))
    )
    if not np.isfinite(atr_ref) or atr_ref <= 1e-9:
        atr_ref = float(np.nanmedian(atr.values))
    atr_ref = max(atr_ref, 1e-6)

    local = atr / (ewm_fast + 1e-12)
    global_raw = np.sqrt(ewm_slow / atr_ref)
    global_mult = np.clip(global_raw, g_lo, g_hi)
    atr_scale = np.clip(local * global_mult, m_lo, m_hi)
    return atr_scale.astype(np.float32), atr_ref


def compute_weights_logic(df, cfg, strategy=None):
    tprint(f"Entering function: compute_weights_logic in training.py")
    from .model_mr import compute_mr_weights
    from .model_tf import compute_tf_weights

    # Assume TF if strategy is not provided or if it's explicitly TF
    is_mr = strategy.get("is_mr", False) if strategy else False

    if is_mr:
        return compute_mr_weights(df, cfg)
    else:
        return compute_tf_weights(df, cfg)


def _strategy_bucket_context(
    trade_side: str, strategy_id: str, cfg: dict | None = None
) -> tuple:
    """Return (candidate_bucket, move_bucket, strategy_label) for (trade_side, strategy_id).

    Strategy definitions (cfg['strategies']) are authoritative; legacy mapping is fallback.
    """
    side = str(trade_side).lower()
    strat_id = str(strategy_id)

    for strat in get_strategies(cfg or {}):
        s_side = str(strat.get("trade_side", "")).lower()
        s_id = str(strat.get("strategy_id", ""))
        mode = str(strat.get("base_event_trigger", "")).lower()
        if s_side == side and s_id == strat_id:
            explicit_move_bucket = str(strat.get("move_bucket", "")).lower()
            explicit_candidate_bucket = str(strat.get("candidate_bucket", "")).lower()
            if explicit_move_bucket in {"up", "down"}:
                move_bucket = explicit_move_bucket
            elif "price_up" in mode or "price_down" in mode:
                move_bucket = "up" if "price_up" in mode else "down"
            else:
                move_bucket = None
            if explicit_candidate_bucket in {"best", "worst"}:
                cand_filter = explicit_candidate_bucket
            elif move_bucket in {"up", "down"}:
                cand_filter = "best" if move_bucket == "up" else "worst"
            else:
                cand_filter = None
            return cand_filter, move_bucket, s_id

    # Legacy fallback
    # To retain backwards compat for existing keys that might just be "mr" or "tf"
    is_mr = "mr" in strat_id.lower()
    is_tf = "tf" in strat_id.lower()

    if side == "long":
        cand_filter = "worst" if is_mr else "best"
    else:
        cand_filter = "best" if is_mr else "worst"

    move_bucket = "up" if cand_filter == "best" else "down"
    return cand_filter, move_bucket, strat_id


def _trend_direction_keep_mask(trend_vals, trend_filter: str) -> np.ndarray:
    """Return a strict directional mask for trend filtering.

    Notes:
    - Neutral (0.0) trend values are excluded for both directions.
    - Non-finite values are excluded (instead of being coerced to 0.0 and leaking into "down").
    """
    arr = np.asarray(trend_vals, dtype=float)
    finite = np.isfinite(arr)
    if str(trend_filter).lower() == "up":
        return finite & (arr > 0.0)
    return finite & (arr < 0.0)


def _meta_feature_keys_for_kind(
    cfg: dict,
    strategy: dict | None = None,
    kind: str | None = None,
) -> list[str]:
    """Return shared meta features plus optional kind-specific overlay based on strategy config."""
    if strategy and "meta_feature_keys" in strategy and kind is None:
        out = []
        seen = set()
        for k in strategy["meta_feature_keys"]:
            if isinstance(k, str) and k and k not in seen:
                out.append(k)
                seen.add(k)
        return out

    from .training_utils import dedupe_keep_order, get_meta_feature_keys

    if kind is None:
        return dedupe_keep_order(
            get_meta_feature_keys("reg", cfg)
            + get_meta_feature_keys("clf", cfg)
            + get_meta_feature_keys("mfe", cfg)
            + get_meta_feature_keys("mae", cfg)
            + get_meta_feature_keys("asym", cfg)
        )
    kind = str(kind).lower()
    if kind not in {"reg", "clf", "mfe", "mae", "asym"}:
        kind = "clf"
    return dedupe_keep_order(get_meta_feature_keys(kind, cfg))


def _build_meta_move_soft_target(
    abs_ret: np.ndarray,
    vol_proxy: np.ndarray,
    thresholds: list[float] | tuple[float, ...],
    weights: list[float] | tuple[float, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    abs_ret_arr = np.asarray(abs_ret, dtype=np.float64).reshape(-1)
    vol_arr = np.asarray(vol_proxy, dtype=np.float64).reshape(-1)
    thresh = np.asarray(list(thresholds), dtype=np.float64).reshape(-1)
    w = np.asarray(list(weights), dtype=np.float64).reshape(-1)
    if thresh.size == 0:
        thresh = np.asarray([1.0, 1.25, 1.5], dtype=np.float64)
    if w.size == 0:
        w = np.asarray([0.45, 0.35, 0.20], dtype=np.float64)
    if thresh.size != w.size:
        n = min(thresh.size, w.size)
        thresh = thresh[:n]
        w = w[:n]
    if thresh.size == 0:
        thresh = np.asarray([1.0, 1.25, 1.5], dtype=np.float64)
        w = np.asarray([0.45, 0.35, 0.20], dtype=np.float64)
    if not np.isfinite(np.sum(w)) or float(np.sum(w)) <= 0.0:
        w = np.ones_like(thresh, dtype=np.float64)
    w = w / max(float(np.sum(w)), 1e-12)
    vp = np.clip(vol_arr, 1e-9, None)
    ladder = []
    for k in thresh:
        ladder.append((abs_ret_arr > (float(k) * vp)).astype(np.float32))
    stack = np.vstack(ladder).astype(np.float32)
    soft = np.dot(w.astype(np.float32), stack).astype(np.float32)
    hard = (soft >= 0.5).astype(np.int8)
    middle_idx = 1 if len(thresh) > 1 else 0
    move_thr = (float(thresh[middle_idx]) * vp).astype(np.float32)
    return soft, hard, move_thr


# Exhaustion model logic removed per user request.


def _optimize_training_sample_weights(
    df: pd.DataFrame,
    X_frame: pd.DataFrame,
    y_ret: np.ndarray,
    label_times: pd.DataFrame,
    base_weights: np.ndarray,
    cfg: dict,
    stage: str,
    extra_components: dict | None = None,
    strategy: dict | None = None,
) -> np.ndarray:
    """Optimize sample-weight component blend using constrained CV objective."""
    if not bool(cfg.get("sample_weight_opt_enable", True)):
        return np.asarray(base_weights, dtype=np.float32)

    n = len(base_weights)
    if n < int(cfg.get("sample_weight_opt_min_samples", 400)):
        return np.asarray(base_weights, dtype=np.float32)

    components: dict[str, np.ndarray] = {"base": np.asarray(base_weights, dtype=float)}
    ts_vals = pd.to_datetime(df["ts"]).values
    stage_l = str(stage).lower()

    # vol_cs component removed per user request

    # liquidity component removed per user request

    era = pd.to_datetime(df["ts"]).dt.to_period("M").astype(str).values
    bar_idx = np.arange(n, dtype=int)
    components["recency"] = compute_recency_weights(
        bar_idx,
        era,
        half_life_bars=int(cfg.get("sample_weight_recency_half_life_bars", 24 * 30)),
        min_era_neff_ratio=float(
            cfg.get("sample_weight_recency_min_era_neff_ratio", 0.2)
        ),
    )

    # Trade-quality component from excursion quality (MFE/MAE vs TP/SL).
    # Keep as a separate optimizable component (even if base weights already include it)
    # so stage-specific alpha selection can up/down-weight it explicitly.
    if bool(cfg.get("sample_weight_use_trade_quality_component", True)):
        mfe_col = (
            "__mfe__"
            if "__mfe__" in df.columns
            else ("mfe_4h" if "mfe_4h" in df.columns else None)
        )
        mae_col = (
            "__mae__"
            if "__mae__" in df.columns
            else ("mae_4h" if "mae_4h" in df.columns else None)
        )
        if mfe_col is not None and mae_col is not None:
            mfe_v = np.nan_to_num(df[mfe_col].values, nan=0.0).astype(np.float64)
            mae_v = np.nan_to_num(df[mae_col].values, nan=0.0).astype(np.float64)
            if "__tp__" in df.columns:
                tp_v = np.clip(
                    np.abs(
                        np.nan_to_num(df["__tp__"].values, nan=0.02).astype(np.float64)
                    ),
                    1e-4,
                    None,
                )
            elif "__barrier_pct__" in df.columns:
                tp_v = np.clip(
                    np.abs(
                        np.nan_to_num(df["__barrier_pct__"].values, nan=0.02).astype(
                            np.float64
                        )
                    ),
                    1e-4,
                    None,
                )
            else:
                tp_v = np.full(n, 0.02, dtype=np.float64)

            if "__sl__" in df.columns:
                sl_v = np.clip(
                    np.abs(
                        np.nan_to_num(df["__sl__"].values, nan=0.01).astype(np.float64)
                    ),
                    1e-4,
                    None,
                )
            else:
                sl_v = np.clip(0.5 * tp_v, 1e-4, None)

            if "__is_timeout__" in df.columns:
                is_to = np.asarray(df["__is_timeout__"].values, dtype=bool)
            elif "__y_lbl__" in df.columns:
                is_to = np.asarray(df["__y_lbl__"].values) == OUT_TO
            else:
                is_to = np.zeros(n, dtype=bool)

            w_trade_quality = compute_mfe_mae_weights(
                mfe=mfe_v,
                mae=mae_v,
                tp=tp_v,
                sl=sl_v,
                is_timeout=is_to,
                touch_margin=None,
                w_min=float(cfg.get("mfe_mae_w_min", 0.5)),
                tau=float(cfg.get("mfe_mae_tau", 1.0)),
                cost_floor=float(cfg.get("mfe_mae_cost_floor", 0.001)),
            )
            components["trade_quality"] = np.asarray(w_trade_quality, dtype=np.float64)

    # Magnitude/opportunity gates as additive components (not hard replacements).
    # These are economically grounded helpers to downweight low-opportunity rows
    # and saturate tail influence for return regression targets.
    if bool(cfg.get("sample_weight_use_tp_opportunity_component", True)):
        if "__barrier_pct__" in df.columns:
            atr_proxy = np.clip(
                np.nan_to_num(df["__barrier_pct__"].values, nan=0.02).astype(
                    np.float64
                ),
                1e-6,
                None,
            )
        elif "atr_pct" in df.columns:
            atr_proxy = np.clip(
                np.nan_to_num(df["atr_pct"].values, nan=0.02).astype(np.float64),
                1e-6,
                None,
            )
        else:
            atr_proxy = np.full(n, 0.02, dtype=np.float64)

        fee_rt = float(
            cfg.get(
                "sample_weight_fee_rt",
                float(cfg.get("label_round_trip_fee_pct", 0.3)) / 100.0,
            )
        )
        components["tp_opportunity"] = sample_weight_tp_classifier(
            atr_pct_past=atr_proxy,
            fee_rt=fee_rt,
            k=float(cfg.get("sample_weight_tp_k", 1.5)),
            s=cfg.get("sample_weight_tp_softness", None),
            w_min=float(cfg.get("sample_weight_tp_w_min", 0.4)),
        ).astype(np.float64, copy=False)

    if bool(cfg.get("sample_weight_use_meta_magnitude_component", True)):
        if "__barrier_pct__" in df.columns:
            atr_proxy = np.clip(
                np.nan_to_num(df["__barrier_pct__"].values, nan=0.02).astype(
                    np.float64
                ),
                1e-6,
                None,
            )
        elif "atr_pct" in df.columns:
            atr_proxy = np.clip(
                np.nan_to_num(df["atr_pct"].values, nan=0.02).astype(np.float64),
                1e-6,
                None,
            )
        else:
            atr_proxy = None

        fee_rt = float(
            cfg.get(
                "sample_weight_fee_rt",
                float(cfg.get("label_round_trip_fee_pct", 0.3)) / 100.0,
            )
        )
        components["meta_magnitude"] = sample_weight_meta_regression(
            y_ret_net=np.asarray(y_ret, dtype=np.float64),
            atr_pct_past=atr_proxy,
            fee_rt=fee_rt,
            k=float(cfg.get("sample_weight_meta_k", 1.5)),
            s=cfg.get("sample_weight_meta_softness", None),
            w_min=float(cfg.get("sample_weight_meta_w_min", 0.4)),
            alpha=None,
            alpha_quantile=float(cfg.get("sample_weight_meta_alpha_quantile", 0.5)),
        ).astype(np.float64, copy=False)

    if extra_components:
        for k, v in extra_components.items():
            if v is None:
                continue
            arr = np.asarray(v, dtype=float)
            if len(arr) == n:
                components[k] = arr

    label_intervals = np.column_stack(
        [
            pd.to_datetime(label_times["t_start"]).values.astype("datetime64[ns]"),
            pd.to_datetime(label_times["t_end"]).values.astype("datetime64[ns]"),
        ]
    )

    X_frame = select_test_feature_frame(X_frame)
    X_np = np.asarray(X_frame, dtype=np.float32)

    fixed_component_alphas = cfg.get("sample_weight_component_alphas")
    if "meta" in stage_l:
        fixed_component_alphas = cfg.get(
            "sample_weight_component_alphas_meta", fixed_component_alphas
        )
    elif "base" in stage_l or "alpha" in stage_l:
        fixed_component_alphas = cfg.get(
            "sample_weight_component_alphas_base", fixed_component_alphas
        )
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
        tprint(
            f"[{stage}] using persisted sample-weight component alphas={resolved_alphas}"
        )
        log_weight_statistics(optimized_weights, era, f"{stage}_persisted_alphas")
        return np.asarray(optimized_weights, dtype=np.float32)

    opt_max_samples = _bounded_sample_cap(
        len(X_np),
        absolute_cap=int(
            cfg.get(
                "sample_weight_opt_max_samples",
                20000 if "meta" in stage_l else 30000,
            )
        ),
        pct_cap=float(cfg.get("sample_weight_opt_max_pct", 1.0)),
    )
    if len(X_np) > opt_max_samples:
        opt_idx = np.linspace(0, len(X_np) - 1, opt_max_samples, dtype=np.int32)
        X_np = X_np[opt_idx]
        y_ret_opt = np.asarray(y_ret, dtype=float)[opt_idx]
        label_intervals = label_intervals[opt_idx]
        components = {k: np.asarray(v)[opt_idx] for k, v in components.items()}
    else:
        y_ret_opt = np.asarray(y_ret, dtype=float)

    res = optimize_component_weights(
        X=X_np,
        y_ret=y_ret_opt,
        label_intervals=label_intervals,
        components=components,
        production_model=cfg.get("sample_weight_opt_model_family", "ExtraTrees"),
        n_trials=int(cfg.get("sample_weight_opt_trials", 8)),
        n_splits=int(cfg.get("sample_weight_opt_n_splits", 3)),
        embargo_bars=int(cfg.get("sample_weight_opt_embargo_bars", 10)),
        min_n_eff_ratio=float(cfg.get("sample_weight_opt_min_n_eff_ratio", 0.30)),
        max_top1pct=float(cfg.get("sample_weight_opt_max_top1pct", 0.05)),
        random_state=int(cfg.get("seed", 42)),
    )

    tprint(
        f"[{stage}] sample-weight optimization objective={res.objective_value:.5f} alphas={res.component_alphas}"
    )
    log_weight_statistics(res.optimized_weights, era, f"{stage}_optimized")
    return np.asarray(res.optimized_weights, dtype=np.float32)


def build_hourly_training_set_and_weights(
    panel,
    feats,
    mkt_gates,
    cfg,
    syms,
    ts_end,
    p_exh_hist,
    H,
    model_kind,
    trend_filter=None,
    strategy=None,
    feature_key=None,
    extra_feature_keys=None,
    label_method="atr",
    fixed_tp=0.05,
    fixed_sl=0.025,
    side="long",
    _cached_cand_mask=None,
    _cached_tb=None,
    _tb_cache=None,
    _precomputed_events=None,
    _geom_frames=None,
):
    tprint(f"Entering function: build_hourly_training_set_and_weights in training.py")
    empty_out = (None, None, None, None, None, None, None)
    lookup_cache: dict = {}
    c = panel["close"]
    idx = c.index

    if _cached_cand_mask is not None:
        cand_mask = _cached_cand_mask
    else:
        # Strict mode: never generate events outside persisted offline-optimal candidate ranges.
        cand_mask, _, mask_by_strategy = _build_optimal_candidate_mask(
            panel, feats, cfg
        )
    if cand_mask is None:
        tprint("No candidates mask returned.")
        return empty_out
    tprint(f"Candidates found: {cand_mask.sum().sum()}")

    ts_start = ts_end - pd.Timedelta(hours=int(cfg["train_lookback_hours"]))
    # Slice to time window first, then apply subsample filter
    ts_end_adj = ts_end - pd.Timedelta(hours=H + 8)
    window_cand = cand_mask.loc[
        (cand_mask.index >= ts_start) & (cand_mask.index <= ts_end_adj)
    ]
    if window_cand.empty:
        tprint(
            "No rows generated for training set: "
            f"valid_syms=0, window_cand_shape={window_cand.shape}, "
            f"ts_window=[{ts_start}, {ts_end_adj}]"
        )
        return empty_out

    # Early symbol precheck before any barrier computation.
    valid_syms = [s for s in syms if s in window_cand.columns and s in c.columns]
    if not valid_syms:
        tprint(
            "No rows generated for training set: "
            f"valid_syms={len(valid_syms)}, window_cand_shape={window_cand.shape}, "
            f"ts_window=[{ts_start}, {ts_end_adj}]"
        )
        return empty_out

    # 3) Purge Label Noise (Microstructure Filtering)
    # This block was hoisted to 'generate_label_datasets' for 15x operational speedup.
    # The filter is now pre-embedded directly inside the passed `_cached_cand_mask`.

    if _cached_tb is not None:
        tb_labels, tb_returns, tb_quality = _cached_tb
    elif label_method == "triple_barrier":
        # Use unified barrier factory (canonical TP/SL geometry)
        if "atr_pct" in feats:
            atr_pct = _coerce_feature_to_panel_df(
                feats["atr_pct"], panel, "atr_pct", fill_value=0.01
            )

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
            tp_lo = (
                float(cfg.get("barrier_tp_lo_h2", 0.015))
                if int(H) == 2
                else float(cfg.get("barrier_tp_lo", 0.02))
            )
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
                tb_labels, tb_returns, tb_quality = _tb_cache[tb_cache_key]
                tprint("Using cached barriers/triple-barrier labels")
            else:
                # Dynamic horizon scaling (+0% to +50% based on ATR regime)
                dyn_horizon = _compute_dynamic_horizon_frame(atr_pct, float(H), cfg)
                eff_horizon = dyn_horizon if dyn_horizon is not None else float(H)
                if dyn_horizon is not None:
                    h_mean = float(
                        dyn_horizon.values[np.isfinite(dyn_horizon.values)].mean()
                    )
                    tprint(
                        f"Using dynamic horizon scaling (base={H}h -> mean={h_mean:.2f}h)"
                    )

                tprint("Computing barriers using unified factory...")
                tp_df, sl_df, diag = compute_barrier_factory(
                    atr_pct=atr_pct,
                    window_size=int(cfg.get("barrier_atr_window", 24 * 30)),
                    k_tp=k_tp,
                    sl_base_mult=sl_base_mult,
                    horizon=eff_horizon,
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
                    use_standalone_sl=cfg.get("use_standalone_sl", False),
                )
                tprint(
                    f"Labeling: Unified Barrier Factory (Mean TP={diag['tp_mean']:.4f}, "
                    f"SL={diag['sl_mean']:.4f}, m∈[{diag['m_p10']:.2f},{diag['m_p90']:.2f}], "
                    f"m_at bounds: lo={diag['m_at_m_lo_pct']:.1%}, hi={diag['m_at_m_hi_pct']:.1%}, "
                    f"z_gate: below={diag['z_below_gate_pct']:.1%}, above={diag['z_above_gate_pct']:.1%}, "
                    f"sl_mult: lo={diag['sl_at_sl_lo_pct']:.1%}, hi={diag['sl_at_sl_hi_pct']:.1%})"
                )
                _tb_out = compute_triple_barrier_labels(
                    panel,
                    tp_df,
                    sl_df,
                    H,
                    side=side,
                    return_outcomes=True,
                    horizons_frame=dyn_horizon,
                    return_path_stats=False,
                )
                tb_labels, tb_returns, tb_quality = _tb_out[:3]
                if _tb_cache is not None:
                    _tb_cache[tb_cache_key] = (tb_labels, tb_returns, tb_quality)

    else:
        # Default ATR logic (fallback: no quality score)
        k_sl = cfg.get("train_k_sl", 2.0)
        k_pt = cfg.get("train_k_pt", 2.0)
        k_tp = cfg.get("train_k_tp", 1.0)

        if "atr_pct" in feats:
            atr_df = feats["atr_pct"]
        else:
            tprint("Warning: atr_pct not found, using default 1% ATR for labeling")
            atr_df = pd.DataFrame(0.01, index=c.index, columns=c.columns)

        tb_labels, tb_returns = compute_trailing_atr_labels(
            panel, atr_df, k_sl=k_sl, k_pt=k_pt, k_tp=k_tp, horizon_hours=H
        )
        # Mock quality for legacy ATR
        tb_quality = pd.DataFrame(0.5, index=tb_labels.index, columns=tb_labels.columns)

    # Subsample: disabled - use all hours for maximum signal
    # window_cand = window_cand[window_cand.index.hour % 3 == 0]

    if feature_key:
        feat_keys = cfg.get(feature_key, [])
    else:
        feat_keys = cfg.get("causal_cols", [])

    if extra_feature_keys:
        # Add extra keys, preserving uniqueness
        feat_keys = list(set(feat_keys) | set(extra_feature_keys))

    # Add Regime Conditioning Features
    regime_keys = [
        "cusum_strength",
        "move_magnitude_z",
        "cusum_decay",
        "vol_percentile",
        "vol_of_vol",
        "atr_percentile",
        "liquidity_ratio",
    ]
    if bool(cfg.get("use_regime_features", True)):
        feat_keys = list(set(feat_keys) | set(regime_keys))

    # --- Pipeline Hardening: Filter missing features ---
    # Ensure features exist at runtime (skip missing ones per hardening policy)
    _orig_feat_keys = list(feat_keys)
    feat_keys = [k for k in feat_keys if (k in feats or k == "p_exh_lag1")]
    _missing_feats = sorted(list(set(_orig_feat_keys) - set(feat_keys)))
    if _missing_feats:
        tprint(
            f"WARNING: Pipeline Hardening: {len(_missing_feats)} features not available in Parquet store. "
            f"Skipping: {_missing_feats}"
        )

    # --- Vectorized event extraction using numpy ---
    valid_syms = [s for s in valid_syms if s in tb_labels.columns]
    if not valid_syms or window_cand.empty:
        tprint(
            "No rows generated for training set: "
            f"valid_syms={len(valid_syms)}, window_cand_shape={window_cand.shape}, "
            f"ts_window=[{ts_start}, {ts_end_adj}]"
        )
        return empty_out

    # Pre-filter candidates to where entry_ts is present in tb_labels index.
    # Use UTC-ns comparison to avoid tz-aware vs tz-naive mismatch causing false-empty alignment.
    try:
        cand_ns = pd.to_datetime(window_cand.index, utc=True).view("i8")
        valid_entry_ns = pd.to_datetime(tb_labels.index, utc=True).view("i8") - int(
            pd.Timedelta(hours=1).value
        )
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

    if _precomputed_events is not None:
        event_ts, event_sym, entry_ts = _precomputed_events
    else:
        sub_mask = window_cand_aligned[valid_syms]
        rows_idx, cols_idx = np.where(sub_mask.values)
        tprint(f"Candidate events: {len(rows_idx)}")
        if len(rows_idx) == 0:
            tprint(
                "No rows generated for training set: candidate event extraction returned 0 "
                f"(sub_mask_shape={sub_mask.shape})"
            )
            return empty_out

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
        tprint(
            f"Entry alignment drop (H={H}): removed {n_entry_drop}/{n_pre_entry} events missing in tb_labels index"
        )
        tprint(
            f"  Gap rate: {gap_rate*100:.1f}% | tb_labels range: {tb_labels.index.min()} to {tb_labels.index.max()}"
        )
        tprint(f"  Sample missing hours: {entry_ts[~entry_valid][:5].tolist()}")
    event_ts = event_ts[entry_valid]
    event_sym = event_sym[entry_valid]
    entry_ts = event_ts + pd.Timedelta(hours=1)

    if len(event_ts) == 0:
        tprint(
            "No rows generated for training set: "
            f"all events dropped by entry alignment (pre={n_pre_entry}, drop={n_entry_drop})"
        )
        return empty_out

    # Trend filter (skip when already precomputed upstream for this side/kind)
    if _precomputed_events is None and trend_filter and "trend_pct" in feats:
        trend_vals = _fast_lookup_cached(
            feats["trend_pct"], event_ts, event_sym, lookup_cache=lookup_cache
        )
        keep = _trend_direction_keep_mask(trend_vals, trend_filter)
        n_pre_trend = len(event_ts)
        n_trend_drop = int((~keep).sum())
        if n_trend_drop > 0:
            tprint(
                f"Trend filter drop ({trend_filter}): removed {n_trend_drop}/{n_pre_trend} events"
            )
        event_ts = event_ts[keep]
        event_sym = event_sym[keep]
        entry_ts = event_ts + pd.Timedelta(hours=1)

    if len(event_ts) == 0:
        tprint(
            "No rows generated for training set: "
            f"trend_filter='{trend_filter}' removed all events"
        )
        return empty_out

    tprint(f"Events after trend filter: {len(event_ts)}")

    # --- Fast numpy positional lookups (avoid stack/reindex) ---
    # Extract TB labels/returns at entry time
    lbl_vals = _fast_lookup_cached(
        tb_labels, entry_ts, event_sym, lookup_cache=lookup_cache
    )
    ret_vals = _fast_lookup_cached(
        tb_returns, entry_ts, event_sym, lookup_cache=lookup_cache
    )
    qual_vals = _fast_lookup_cached(
        tb_quality, entry_ts, event_sym, lookup_cache=lookup_cache
    )

    # Extract absolute TP/SL levels for dynamic timeout weighting
    if (
        _geom_frames is not None
        and "tp_vals" in _geom_frames
        and "sl_vals" in _geom_frames
    ):
        tp_vals = _fast_lookup_cached(
            _geom_frames["tp_vals"], entry_ts, event_sym, lookup_cache=lookup_cache
        )
        sl_vals = _fast_lookup_cached(
            _geom_frames["sl_vals"], entry_ts, event_sym, lookup_cache=lookup_cache
        )
    else:
        # Fallback context: reuse fixed_tp/fixed_sl or global cfg defaults if geom frames missing
        tp_vals = np.full_like(ret_vals, fixed_tp)
        sl_vals = np.full_like(ret_vals, fixed_sl)

    # PnL computation
    pnl = ret_vals

    # Optional: overwrite labels/returns with engine-identical rollout labels.
    _policy_rollout_enabled = bool(cfg.get("policy_rollout_labeling_enable", True))
    _policy_mfe_vals = None
    _policy_mae_vals = None
    _policy_bars_vals = None
    _policy_bars_to_mfe_vals = None
    if _policy_rollout_enabled and all(
        k in panel for k in ["open", "high", "low", "close"]
    ):
        try:
            _max_hold = int(
                cfg.get(
                    "policy_label_max_hold_hours",
                    cfg.get(
                        "max_hold_hours", max(cfg.get("label_horizons_hours", [8]))
                    ),
                )
            )
            _direction = 1 if str(side).lower() == "long" else -1
            _policy_sl = float(cfg.get("policy_label_sl_atr_mult", 1.2))
            _policy_tp_ratio = float(cfg.get("policy_label_tp_sl_ratio", 2.0))
            _policy_trailing = float(cfg.get("policy_label_trailing_pct", 0.35))
            _policy_cfg = {
                "policy_label_sl_atr_mult": _policy_sl,
                "policy_label_tp_sl_ratio": _policy_tp_ratio,
                "policy_label_trailing_pct": _policy_trailing,
            }
            _idx_cache = {}
            _ret_pol = np.zeros(len(entry_ts), dtype=np.float32)
            _lbl_pol = np.ones(len(entry_ts), dtype=np.int8)
            _mae_pol = np.zeros(len(entry_ts), dtype=np.float32)
            _mfe_pol = np.zeros(len(entry_ts), dtype=np.float32)
            _bars_pol = np.zeros(len(entry_ts), dtype=np.int16)
            _bars_to_mfe_pol = np.zeros(len(entry_ts), dtype=np.int16)
            _atr_panel = feats.get("atr_pct", None)
            for _i, (_ts_e, _sym) in enumerate(zip(entry_ts, event_sym)):
                _sym = str(_sym)
                if _sym not in _idx_cache:
                    _ohlc_sym = pd.DataFrame(
                        {
                            "open": panel["open"][_sym],
                            "high": panel["high"][_sym],
                            "low": panel["low"][_sym],
                            "close": panel["close"][_sym],
                        }
                    ).dropna()
                    if (
                        _atr_panel is not None
                        and hasattr(_atr_panel, "columns")
                        and _sym in _atr_panel.columns
                    ):
                        _atr_s = (
                            _atr_panel[_sym]
                            .reindex(_ohlc_sym.index)
                            .ffill()
                            .fillna(0.02)
                        )
                    else:
                        _atr_s = pd.Series(0.02, index=_ohlc_sym.index)
                    _idx_cache[_sym] = (_ohlc_sym, _atr_s)
                _ohlc_sym, _atr_s = _idx_cache[_sym]
                _entry_ns = (
                    pd.to_datetime([_ts_e], utc=True).tz_localize(None).view("i8")[0]
                )
                _ohlc_ns = _datetime_index_to_ns(_ohlc_sym.index)
                _t0 = int(np.searchsorted(_ohlc_ns, _entry_ns))
                if _t0 >= len(_ohlc_ns) or _ohlc_ns[_t0] != _entry_ns:
                    _t0 = -1
                if _t0 < 0:
                    continue
                _entry_px = float(_ohlc_sym["open"].iloc[_t0])
                _po = policy_rollout_ml(
                    ohlc=_ohlc_sym,
                    atr_pct=_atr_s,
                    t0=_t0,
                    direction=_direction,
                    policy_params=_policy_cfg,
                    max_hold_hours=_max_hold,
                )
                _ret_pol[_i] = float(_po.r_policy)
                _lbl_pol[_i] = np.int8(_po.exit_code)
                _mae_pol[_i] = float(_po.mae)
                _mfe_pol[_i] = float(_po.mfe)
                _bars_pol[_i] = int(_po.bars_held)
                _bars_held = int(max(0, _po.bars_held))
                _bars_to_scan = int(min(_bars_held, _max_hold))
                if _bars_to_scan > 0:
                    _scan_lo = _t0 + 1
                    _scan_hi = min(_t0 + _bars_to_scan, len(_ohlc_sym) - 1)
                    if _scan_hi >= _scan_lo:
                        if _direction > 0:
                            _fav_path = (
                                _ohlc_sym["high"]
                                .iloc[_scan_lo : _scan_hi + 1]
                                .to_numpy(dtype=np.float64)
                                / (_entry_px + 1e-12)
                            ) - 1.0
                        else:
                            _fav_path = (
                                _entry_px
                                / (
                                    _ohlc_sym["low"]
                                    .iloc[_scan_lo : _scan_hi + 1]
                                    .to_numpy(dtype=np.float64)
                                    + 1e-12
                                )
                            ) - 1.0
                        if _fav_path.size:
                            _bars_to_mfe_pol[_i] = np.int16(np.argmax(_fav_path) + 1)
            ret_vals = _ret_pol
            lbl_vals = _lbl_pol
            pnl = ret_vals
            _policy_mae_vals = _mae_pol
            _policy_mfe_vals = _mfe_pol
            _policy_bars_vals = _bars_pol
            _policy_bars_to_mfe_vals = _bars_to_mfe_pol
            tprint(
                f"Policy rollout labels applied: n={len(ret_vals)} direction={'long' if _direction>0 else 'short'} "
                f"sl_atr={_policy_sl:.2f} tp_sl={_policy_tp_ratio:.2f} trailing={_policy_trailing:.2f} "
                f"max_hold={_max_hold}h"
            )
        except Exception as _e_pol:
            tprint(
                f"WARNING: policy rollout labeling failed, falling back to triple-barrier labels: {_e_pol}"
            )

    # New Target Logic: Binary (TP vs Rest) with Outcome-based Weighting
    # Outcomes: 2=TP, 1=TIMEOUT, 0=SL
    ret_vals = np.nan_to_num(
        np.asarray(ret_vals, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0
    )
    qual_vals = np.clip(
        np.nan_to_num(
            np.asarray(qual_vals, dtype=np.float32),
            nan=0.5,
            posinf=1.0,
            neginf=0.0,
        ),
        0.0,
        1.0,
    )
    tp_vals = np.clip(
        np.nan_to_num(
            np.asarray(tp_vals, dtype=np.float32),
            nan=float(fixed_tp),
            posinf=float(fixed_tp),
            neginf=float(fixed_tp),
        ),
        1e-4,
        None,
    )
    sl_vals = np.clip(
        np.nan_to_num(
            np.asarray(sl_vals, dtype=np.float32),
            nan=float(fixed_sl),
            posinf=float(fixed_sl),
            neginf=float(fixed_sl),
        ),
        1e-4,
        None,
    )
    pnl = np.nan_to_num(
        np.asarray(pnl, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0
    )

    valid_outcomes = np.isin(
        lbl_vals, np.array([OUT_SL, OUT_TO, OUT_TP], dtype=lbl_vals.dtype)
    )
    assert bool(
        np.all(valid_outcomes)
    ), f"Unexpected outcome labels found: {np.unique(lbl_vals[~valid_outcomes])}"

    _diag_labels = bool(cfg.get("label_diagnostics_mode", False))
    if _diag_labels:
        _n_all = len(lbl_vals)
        _tp_all = float(np.sum(lbl_vals == OUT_TP)) / max(1, _n_all)
        _sl_all = float(np.sum(lbl_vals == OUT_SL)) / max(1, _n_all)
        _to_all = float(np.sum(lbl_vals == OUT_TO)) / max(1, _n_all)
        tprint(
            f"[LABEL_DIAG][PRE_TO_FILTER] side={side} kind={k} H={H} n={_n_all} "
            f"tp={_tp_all:.3%} sl={_sl_all:.3%} to={_to_all:.3%}"
        )

    # Target: base classifier can be TP-vs-SL only (exclude timeout rows)
    _exclude_to = bool(cfg.get("base_exclude_timeout_from_classifier", True))
    if _exclude_to:
        _y_tmp, _mask_tmp = build_base_tp_vs_sl(lbl_vals)
        keep_rows = np.asarray(_mask_tmp, dtype=bool)
        y_bin = np.asarray(_y_tmp, dtype=np.float32)
        if not np.all(keep_rows):
            event_ts = event_ts[keep_rows]
            event_sym = event_sym[keep_rows]
            entry_ts = entry_ts[keep_rows]
            lbl_vals = lbl_vals[keep_rows]
            ret_vals = ret_vals[keep_rows]
            qual_vals = qual_vals[keep_rows]
            tp_vals = tp_vals[keep_rows]
            sl_vals = sl_vals[keep_rows]
            pnl = pnl[keep_rows]
            if _policy_mfe_vals is not None:
                _policy_mfe_vals = _policy_mfe_vals[keep_rows]
            if _policy_mae_vals is not None:
                _policy_mae_vals = _policy_mae_vals[keep_rows]
            if _policy_bars_vals is not None:
                _policy_bars_vals = _policy_bars_vals[keep_rows]
            if _policy_bars_to_mfe_vals is not None:
                _policy_bars_to_mfe_vals = _policy_bars_to_mfe_vals[keep_rows]
    else:
        # Legacy TP-vs-rest
        y_bin = (lbl_vals == OUT_TP).astype(np.float32)

    if _diag_labels:
        _n_kept = len(lbl_vals)
        _tp_kept = float(np.sum(lbl_vals == OUT_TP)) / max(1, _n_kept)
        _sl_kept = float(np.sum(lbl_vals == OUT_SL)) / max(1, _n_kept)
        _to_kept = float(np.sum(lbl_vals == OUT_TO)) / max(1, _n_kept)
        _pos = float(np.mean(y_bin)) if len(y_bin) else float("nan")
        _neg = 1.0 - _pos if np.isfinite(_pos) else float("nan")
        tprint(
            f"[LABEL_DIAG][POST_TO_FILTER] side={side} kind={k} H={H} n={_n_kept} "
            f"tp={_tp_kept:.3%} sl={_sl_kept:.3%} to={_to_kept:.3%} y_pos={_pos:.3%} y_neg={_neg:.3%}"
        )

    # Weighting based on Quality and Outcome Importance
    # TP (2): w = quality (0.5 to 1.0) -> cleaner wins matter more? Or messy wins less?
    # SL (0): w = 1.0 - quality (0.0=Bad Loss -> w=1.0, 0.5=Near Miss -> w=0.5)
    # TO (1): w = 0.2 (Downweight timeouts)

    w_outcome = np.ones_like(y_bin, dtype=np.float32)
    is_tp = lbl_vals == OUT_TP
    is_sl = lbl_vals == OUT_SL
    is_to = lbl_vals == OUT_TO
    # timeout_weight is now a base scalar for the dynamic formula
    timeout_weight = float(cfg.get("timeout_weight", 0.4))
    w_outcome[is_tp] = qual_vals[is_tp]
    w_outcome[is_sl] = 1.0 - qual_vals[is_sl]

    # Dynamic TO weighting: wTO = 1 + 1.5 * (1 - s)^2 where s = (r - SL)/(TP - SL) normalized to [0,1]
    # Then scaled by timeout_weight (0.2)
    if np.any(is_to):
        r_to = pnl[is_to]
        # Barriers are distances (positive) -> SL level = -sl, TP level = +tp
        sl_dist = sl_vals[is_to]
        tp_dist = tp_vals[is_to]
        denom = tp_dist + sl_dist
        # s = position in range [SL, TP], 0=SL, 1=TP
        s_score = np.clip((r_to + sl_dist) / (denom + 1e-9), 0.0, 1.0)
        # Weight shape: quadratic increase towards SL
        # s=1 (TP) -> w=1.0, s=0 (SL) -> w=2.5
        w_to_dynamic = 1.0 + 1.5 * ((1.0 - s_score) ** 2)

        # Apply base scalar
        w_to_final = w_to_dynamic * timeout_weight

        # Floor constraint: avg(TO) must be >= 0.1 * avg(SL)
        if np.any(is_sl):
            avg_sl = np.mean(w_outcome[is_sl])
            avg_to = np.mean(w_to_final)
            target_avg_to = 0.05 * avg_sl
            if avg_to < target_avg_to and avg_to > 1e-12:
                boost = target_avg_to / avg_to
                w_to_final *= boost
                tprint(
                    f"TO Weight Boosted: avg_to={avg_to:.4f} -> {avg_to*boost:.4f} (target 0.1*SL={target_avg_to:.4f})"
                )

        w_outcome[is_to] = w_to_final

    # Clip weights (configurable; tighter defaults to reduce peaky calibration artifacts)
    w_clip_min = float(cfg.get("outcome_weight_clip_min", 0.5))
    w_clip_max = float(cfg.get("outcome_weight_clip_max", 2.0))
    w_outcome = np.clip(w_outcome, w_clip_min, w_clip_max)

    # Base weight from realized event magnitude, not a fixed-horizon proxy such as ret24h.
    # Winsorize at the 95th percentile, then rescale to [0.5, 1.5] so it remains
    # on the same order of magnitude as the other multiplicative components.
    event_abs_ret = np.abs(np.asarray(ret_vals, dtype=np.float32))
    if event_abs_ret.size:
        mag_q = float(np.nanquantile(event_abs_ret, 0.95))
        if not np.isfinite(mag_q):
            mag_q = 1.0
        mag_q = max(mag_q, 1e-9)
        event_abs_ret_clip = np.clip(event_abs_ret, 0.0, mag_q)
        w_magnitude = 0.5 + event_abs_ret_clip / mag_q
    else:
        mag_q = 1.0
        w_magnitude = np.ones_like(y_bin, dtype=np.float32)
    w_magnitude = np.asarray(w_magnitude, dtype=np.float32)
    w_base = w_magnitude * w_outcome

    # MFE/MAE-based weighting (Report 2026-02-12)
    # Weight by how "decisive" the price movement was relative to barriers
    # r_mfe = MFE/TP, r_mae = MAE/SL, d = max(r_mfe, r_mae)
    # w_mfe_mae = w_min + (1-w_min) * clip(d/tau, 0, 1)
    # This weights samples by excursion quality, not speed or net R:R

    # Get MFE/MAE from policy rollout when enabled (engine-identical), else feature proxies
    if _policy_mfe_vals is not None and _policy_mae_vals is not None:
        mfe_vals = np.asarray(_policy_mfe_vals, dtype=np.float32)
        mae_vals = np.asarray(_policy_mae_vals, dtype=np.float32)
    else:
        if "mfe_4h" in feats:
            mfe_vals = np.nan_to_num(
                _fast_lookup_cached(
                    feats["mfe_4h"], event_ts, event_sym, lookup_cache=lookup_cache
                ),
                nan=0.0,
            )
        else:
            mfe_vals = np.maximum(pnl, 0.0)
        if "mae_4h" in feats:
            mae_vals = np.nan_to_num(
                _fast_lookup_cached(
                    feats["mae_4h"], event_ts, event_sym, lookup_cache=lookup_cache
                ),
                nan=0.0,
            )
        else:
            mae_vals = np.maximum(-pnl, 0.0)

    # Get barrier distances (TP/SL) from ATR
    if "atr_pct" in feats:
        barrier_vals = np.nan_to_num(
            _fast_lookup_cached(
                feats["atr_pct"], event_ts, event_sym, lookup_cache=lookup_cache
            ),
            nan=0.02,
        )
    else:
        barrier_vals = np.full(len(event_ts), 0.02, dtype=np.float32)
    tp_vals = np.clip(np.abs(barrier_vals), 1e-4, None)
    sl_vals = np.clip(0.5 * tp_vals, 1e-4, None)

    # Timeout detection
    is_timeout = lbl_vals == OUT_TO

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
        cost_floor=float(cfg.get("mfe_mae_cost_floor", 0.001)),
    )
    # Align excursion-quality weights to the same [0.5, 1.5] scale as the
    # realized-magnitude multiplier before the later combined normalization step.
    w_mfe_mae = np.clip((2.0 * np.asarray(w_mfe_mae, dtype=np.float32)) - 0.5, 0.5, 1.5)
    w_mfe_mae = np.nan_to_num(w_mfe_mae, nan=1.0, posinf=1.5, neginf=0.5)

    # Multiply into base weight (before normalization)
    w_base = np.nan_to_num(w_base, nan=1.0, posinf=1.5, neginf=0.5).astype(
        np.float32, copy=False
    )
    w_base = w_base * w_mfe_mae
    tprint(
        f"MFE/MAE weighting: mean={w_mfe_mae.mean():.3f}, p10={np.quantile(w_mfe_mae, 0.10):.3f}, p90={np.quantile(w_mfe_mae, 0.90):.3f}"
    )
    tprint(
        "Weight scale check: "
        f"magnitude[q95={mag_q:.6f}] mean={float(np.mean(w_magnitude)):.3f} p05={float(np.quantile(w_magnitude, 0.05)):.3f} p95={float(np.quantile(w_magnitude, 0.95)):.3f}; "
        f"outcome mean={float(np.mean(w_outcome)):.3f} p05={float(np.quantile(w_outcome, 0.05)):.3f} p95={float(np.quantile(w_outcome, 0.95)):.3f}; "
        f"mfe_mae mean={float(np.mean(w_mfe_mae)):.3f} p05={float(np.quantile(w_mfe_mae, 0.05)):.3f} p95={float(np.quantile(w_mfe_mae, 0.95)):.3f}"
    )

    # Tighten the binary target using the label diagnostics already produced by the labeler.
    # Previously weak TP outcomes were only downweighted; they still remained positive in y_bin.
    if bool(cfg.get("base_demote_weak_tp_in_y_bin", True)) and np.any(is_tp):
        min_tp_quality = float(cfg.get("base_min_tp_quality_for_positive", 0.60))
        min_tp_weight = float(cfg.get("base_min_tp_weight_for_positive", 0.85))
        _qual_arr = np.asarray(qual_vals, dtype=np.float32)
        # Only apply quality threshold if qual_vals are plausibly in [0,1] range
        # (mean of TP quality must be > 0.05; otherwise the column is not a proper
        # bound-efficiency score and the threshold is meaningless)
        _qual_tp_mean = float(np.nanmean(_qual_arr[is_tp])) if np.any(is_tp) else 0.0
        _use_qual_gate = _qual_tp_mean > 0.05
        weak_tp_mask = is_tp & (
            (
                (_qual_arr < min_tp_quality)
                if _use_qual_gate
                else np.zeros(len(is_tp), dtype=bool)
            )
            | (np.asarray(w_mfe_mae, dtype=np.float32) < min_tp_weight)
        )
        # Safety: never demote ALL positives — that produces a single-class target
        _would_survive = int(np.sum(is_tp)) - int(np.sum(weak_tp_mask))
        if _would_survive < 1:
            tprint(
                f"Weak TP demotion SKIPPED: would eliminate all {int(np.sum(is_tp))} positives "
                f"(qual_tp_mean={_qual_tp_mean:.3f}, use_qual_gate={_use_qual_gate})"
            )
        elif np.any(weak_tp_mask):
            y_bin = np.asarray(y_bin, dtype=np.float32)
            y_bin[weak_tp_mask] = 0.0
            tprint(
                f"Weak TP demotion in y_bin: demoted={int(np.sum(weak_tp_mask))}/{int(np.sum(is_tp))} "
                f"(min_quality={min_tp_quality:.2f}, min_mfe_mae_w={min_tp_weight:.2f}, "
                f"qual_gate_active={_use_qual_gate})"
            )

    # Mild class-balance multiplier (inverse-freq with sqrt exponent + hard cap)
    p_pos = float(np.mean(y_bin)) if len(y_bin) else 0.5
    p_pos = float(np.clip(p_pos, 1e-4, 1 - 1e-4))
    w1 = (0.5 / p_pos) ** 0.5
    w0 = (0.5 / (1.0 - p_pos)) ** 0.5
    w_class = np.where(y_bin >= 0.5, w1, w0)
    w_class = np.clip(w_class, 0.85, 1.25)

    # Consensus weight from geometry votes (timeouts ignored)
    if _geom_frames is None:
        _geom_tp_df = feats.get(
            "__geom_n_tp__",
            pd.DataFrame(0, index=tb_labels.index, columns=tb_labels.columns),
        )
        _geom_sl_df = feats.get(
            "__geom_n_sl__",
            pd.DataFrame(0, index=tb_labels.index, columns=tb_labels.columns),
        )
    else:
        _geom_tp_df = _geom_frames.get(
            "n_tp", pd.DataFrame(0, index=tb_labels.index, columns=tb_labels.columns)
        )
        _geom_sl_df = _geom_frames.get(
            "n_sl", pd.DataFrame(0, index=tb_labels.index, columns=tb_labels.columns)
        )

    n_tp = np.nan_to_num(
        _fast_lookup_cached(
            _geom_tp_df, event_ts, event_sym, lookup_cache=lookup_cache
        ),
        nan=0.0,
    )
    n_sl = np.nan_to_num(
        _fast_lookup_cached(
            _geom_sl_df, event_ts, event_sym, lookup_cache=lookup_cache
        ),
        nan=0.0,
    )
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

    # Negative mass renormalization (Timeout downweighting) - DISABLED in favor of dynamic wTO logic
    # if bool(cfg.get("use_neg_mass_renorm", True)):
    #     renorm_cfg = NegMassRenormCfg(
    #         w_to_min=float(cfg.get("neg_mass_w_to_min", 0.2)),
    #         w_to_max=float(cfg.get("neg_mass_w_to_max", 1.0)),
    #         rho_pos_over_neg=float(cfg.get("neg_mass_rho", 1.0)),
    #     )
    #     # We treat the whole training set as one 'cell' for renormalization purpose here
    #     cell_ids = np.zeros(len(lbl_vals), dtype=np.int32)
    #
    #     w_mix = compute_cell_weights_neg_mass_renorm(
    #         y=lbl_vals,
    #         cell_id=cell_ids,
    #         base_w=w_mix,
    #         cfg=renorm_cfg,
    #         tp_label=OUT_TP,
    #         sl_label=OUT_SL,
    #         to_label=OUT_TO,
    #     )
    #
    #     n_tp_renorm = (lbl_vals == OUT_TP).sum()
    #     n_sl_renorm = (lbl_vals == OUT_SL).sum()
    #     n_to_renorm = (lbl_vals == OUT_TO).sum()
    #     tprint(f"Renormalized Weights: TP={n_tp_renorm} SL={n_sl_renorm} TO={n_to_renorm}")

    weights_raw = w_mix.astype(np.float32)

    # Build feature DataFrame
    # event_ts is a DatetimeIndex, event_sym is a numpy array
    if isinstance(event_ts, pd.DatetimeIndex):
        ts_arr = event_ts.to_numpy(dtype="datetime64[ns]")
    else:
        ts_arr = (
            pd.to_datetime(event_ts, utc=True, errors="coerce")
            .tz_localize(None)
            .to_numpy(dtype="datetime64[ns]")
        )
    sym_arr = np.asarray(
        event_sym.values if hasattr(event_sym, "values") else event_sym, dtype=object
    )
    _sym_codes, _sym_uniques = pd.factorize(sym_arr, sort=False)
    # Store raw triple-barrier label (-1, 0, 1) for timeout analysis
    _fee_rt = float(cfg.get("policy_fee_rt", 0.003))
    _r_gross = np.asarray(pnl, dtype=np.float32)
    _r_net = ((1.0 + _r_gross.astype(np.float64)) * (1.0 - _fee_rt) - 1.0).astype(
        np.float32
    )
    _dur_vals = np.asarray(
        _policy_bars_vals
        if _policy_bars_vals is not None
        else np.full(len(_r_gross), H, dtype=np.int16),
        dtype=np.int16,
    )
    _bars_to_mfe_vals = np.asarray(
        _policy_bars_to_mfe_vals
        if _policy_bars_to_mfe_vals is not None
        else np.maximum(1, _dur_vals),
        dtype=np.int16,
    )
    _is_mr = str(model_kind).lower() == "mr"
    if _is_mr:
        _mr_sl = np.clip(np.asarray(sl_vals, dtype=np.float64), 1e-6, None)
        _mr_mae_ratio = np.clip(
            np.asarray(mae_vals, dtype=np.float64) / _mr_sl, 0.0, 3.0
        )
        _mr_path_penalty = np.clip(1.0 - np.square(_mr_mae_ratio), 0.0, 1.0)
        _mr_horizon_bars = max(float(cfg.get("mr_utility_horizon_bars", H)), 1.0)
        _mr_velocity_penalty = np.exp(
            -np.clip(_bars_to_mfe_vals.astype(np.float64), 0.0, None) / _mr_horizon_bars
        )
        _u_policy_gross = (
            _r_gross.astype(np.float64) * _mr_path_penalty * _mr_velocity_penalty
        ).astype(np.float32)
        _u_policy_net = (
            _r_net.astype(np.float64) * _mr_path_penalty * _mr_velocity_penalty
        ).astype(np.float32)
        _y_ret_target = _u_policy_net.astype(np.float32)
    else:
        _mr_path_penalty = np.ones(len(_r_gross), dtype=np.float64)
        _mr_velocity_penalty = np.ones(len(_r_gross), dtype=np.float64)
        _u_policy_gross = np.log1p(
            np.clip(_r_gross.astype(np.float64), -0.999999, None)
        ).astype(np.float32)
        _u_policy_net = np.log1p(
            np.clip(_r_net.astype(np.float64), -0.999999, None)
        ).astype(np.float32)
        _y_ret_target = pnl.astype(np.float32)

    parts = {
        "ts": ts_arr,
        "symbol": sym_arr,
        "y_bin": y_bin,
        "y_ret": _y_ret_target,
        "w": weights_raw.astype(np.float32),
        "__y_lbl__": lbl_vals.astype(np.int8),
        "__mfe__": np.asarray(mfe_vals, dtype=np.float32),
        "__mae__": np.asarray(mae_vals, dtype=np.float32),
        "__tp__": np.asarray(tp_vals, dtype=np.float32),
        "__sl__": np.asarray(sl_vals, dtype=np.float32),
        "__is_timeout__": np.asarray(is_timeout, dtype=np.int8),
        "__quality__": np.asarray(qual_vals, dtype=np.float32),
        "__u_policy__": _u_policy_gross,
    }
    # Engine-identical policy labels (gross/net + utility + diagnostics)
    parts["__r_policy_gross__"] = _r_gross
    parts["__r_policy_net__"] = _r_net
    parts["__u_policy_net__"] = _u_policy_net
    parts["__mae_ret__"] = np.asarray(mae_vals, dtype=np.float32)
    parts["__mfe_ret__"] = np.asarray(mfe_vals, dtype=np.float32)
    parts["__bars_to_mfe__"] = _bars_to_mfe_vals
    parts["__mr_path_penalty__"] = np.asarray(_mr_path_penalty, dtype=np.float32)
    parts["__mr_velocity_penalty__"] = np.asarray(
        _mr_velocity_penalty, dtype=np.float32
    )
    parts["__early_inval__"] = np.asarray(
        (lbl_vals == OUT_SL)
        & (
            np.asarray(
                _policy_bars_vals
                if _policy_bars_vals is not None
                else np.zeros(len(lbl_vals)),
                dtype=np.int16,
            )
            <= int(cfg.get("kill_min_bars", 2))
        ),
        dtype=np.int8,
    )

    if _policy_bars_vals is not None:
        parts["__bars_policy__"] = np.asarray(_policy_bars_vals, dtype=np.int16)

    # Store barrier_pct for risk-adjusted meta model target
    if "atr_pct" in feats:
        barrier_vals = _fast_lookup_cached(
            feats["atr_pct"], event_ts, event_sym, lookup_cache=lookup_cache
        )
        barrier_vals = np.nan_to_num(barrier_vals, nan=0.02).astype(np.float32)
        parts["__barrier_pct__"] = np.clip(barrier_vals, 0.005, None)

    parts["__n_tp__"] = n_tp.astype(np.float32)
    parts["__n_sl__"] = n_sl.astype(np.float32)
    parts["__n_res__"] = n_res.astype(np.float32)
    parts["__w_consensus__"] = w_consensus.astype(np.float32)

    lag_ts = ts_arr - np.timedelta64(1, "h")
    _feat_heartbeat_every = max(1, int(cfg.get("label_feature_heartbeat_every", 16)))
    _symbol_chunk_size = max(1, int(cfg.get("label_symbol_chunk_size", 50)))
    _n_unique_symbols = int(len(_sym_uniques))

    def _slice_part(_value, _mask):
        if isinstance(_value, (pd.Index, pd.Series, np.ndarray)):
            return _value[_mask]
        return np.asarray(_value)[_mask]

    def _build_df_chunk(
        _mask: np.ndarray, _chunk_idx: int, _chunk_total: int
    ) -> pd.DataFrame:
        _event_ts = ts_arr[_mask]
        _event_sym = sym_arr[_mask]
        _lag_ts = lag_ts[_mask]
        _parts = {k0: _slice_part(v0, _mask) for k0, v0 in parts.items()}
        # Keep lookup caches local to the chunk. The global cache uses object ids
        # for event arrays, which can be reused across chunks and yield stale
        # row/column indexers with mismatched lengths late in materialization.
        _chunk_lookup_cache = {}

        if p_exh_hist is not None:
            _parts["p_exh_lag1"] = np.nan_to_num(
                _fast_lookup_cached(
                    p_exh_hist, _lag_ts, _event_sym, lookup_cache=_chunk_lookup_cache
                ),
                nan=0.0,
            ).astype(np.float32)
        else:
            _parts["p_exh_lag1"] = np.zeros(len(_event_ts), dtype=np.float32)

        _feat_t0 = time.time()
        for _feat_i, k in enumerate(feat_keys, start=1):
            if k == "p_exh_lag1":
                continue
            if k in feats:
                _parts[k] = _fast_lookup_cached(
                    feats[k], _event_ts, _event_sym, lookup_cache=_chunk_lookup_cache
                )
            if (
                _feat_i == 1
                or _feat_i % _feat_heartbeat_every == 0
                or _feat_i == len(feat_keys)
            ):
                tprint(
                    f"[label hb] H={H} side={side} kind={model_kind} "
                    f"chunk={_chunk_idx}/{_chunk_total} symbols<= {_symbol_chunk_size} "
                    f"features={min(_feat_i, len(feat_keys))}/{len(feat_keys)} "
                    f"events={len(_event_ts):,} elapsed={time.time() - _feat_t0:.1f}s"
                )

        _parts["G_VOL"] = _fast_series_lookup_cached(
            mkt_gates["G_VOL"], _event_ts, lookup_cache=_chunk_lookup_cache
        )
        _parts["G_TREND"] = _fast_series_lookup_cached(
            mkt_gates["G_TREND"], _event_ts, lookup_cache=_chunk_lookup_cache
        )
        return pd.DataFrame(_parts)

    if _n_unique_symbols > _symbol_chunk_size:
        _dfs = []
        _chunk_code_batches = [
            np.arange(i, min(i + _symbol_chunk_size, _n_unique_symbols), dtype=np.int32)
            for i in range(0, _n_unique_symbols, _symbol_chunk_size)
        ]
        tprint(
            f"Label symbol chunking enabled: symbols={_n_unique_symbols} "
            f"chunk_size={_symbol_chunk_size} chunks={len(_chunk_code_batches)}"
        )
        for _chunk_idx, _code_batch in enumerate(_chunk_code_batches, start=1):
            _mask = np.isin(_sym_codes, _code_batch, assume_unique=False)
            if not np.any(_mask):
                continue
            _dfs.append(_build_df_chunk(_mask, _chunk_idx, len(_chunk_code_batches)))
            if bool(cfg.get("label_gc_after_each_chunk", True)):
                gc.collect()
        df = pd.concat(_dfs, axis=0, ignore_index=True) if _dfs else pd.DataFrame(parts)
        del _dfs
    else:
        df = _build_df_chunk(np.ones(len(sym_arr), dtype=bool), 1, 1)

    # Drop constant market gates (fix for Low Variation warning)
    for g in ["G_VOL", "G_TREND"]:
        if g in df.columns and df[g].nunique() <= 1:
            if df[g].std() < 1e-9:
                df.drop(columns=[g], inplace=True)
    # Drop rows only where critical columns are NaN; fill feature NaNs with 0
    # Include all __y_* and diagnostic columns in critical_cols to prevent them from becoming features
    diagnostic_cols = [
        "__y_lbl__",
        "__mfe__",
        "__mae__",
        "__tp__",
        "__sl__",
        "__is_timeout__",
        "__quality__",
        "__barrier_pct__",
        "__n_tp__",
        "__n_sl__",
        "__n_res__",
        "__w_consensus__",
        "__bars_to_mfe__",
        "__mr_path_penalty__",
        "__mr_velocity_penalty__",
        "__regime_vol_12h__",
        "__regime_vol_48h__",
        "__regime_volume_12h__",
        "__regime_volume_48h__",
        "__regime_trend_12h__",
        "__regime_trend_48h__",
    ]
    critical_cols = ["ts", "symbol", "y_bin", "y_ret", "w"] + diagnostic_cols
    df = df.dropna(
        subset=[c for c in ["ts", "symbol", "y_bin", "y_ret", "w"] if c in df.columns]
    )

    # Extract labels before dropping diagnostic cols from features list
    lbl_vals_out = (
        df["__y_lbl__"].values.astype(np.int8) if "__y_lbl__" in df.columns else None
    )

    feat_cols = [c for c in df.columns if c not in critical_cols]
    if feat_cols:
        df[feat_cols] = df[feat_cols].fillna(0)
    if df.empty:
        tprint(
            "No rows generated for training set: "
            "DataFrame empty after critical-column drop/fill"
        )
        return empty_out
    tprint(f"Final training set size: {len(df)}")

    # Quick leakage sanity KPI for regime features vs realized future returns.
    if bool(cfg.get("check_regime_leakage", True)):
        corr_warn_thr = float(cfg.get("regime_corr_warn_thr", 0.35))
        regime_probe = [
            "cusum_strength",
            "move_magnitude_z",
            "cusum_decay",
            "vol_percentile",
            "vol_of_vol",
            "atr_percentile",
            "liquidity_ratio",
        ]
        suspicious = []
        for rk in regime_probe:
            if rk not in df.columns:
                continue
            xv = np.asarray(df[rk].values, dtype=np.float64)
            yv = np.asarray(df["y_ret"].values, dtype=np.float64)
            m = np.isfinite(xv) & np.isfinite(yv)
            if m.sum() < 100:
                continue
            sd_x = float(np.nanstd(xv[m]))
            sd_y = float(np.nanstd(yv[m]))
            if sd_x < 1e-12 or sd_y < 1e-12:
                continue
            corr = float(np.corrcoef(xv[m], yv[m])[0, 1])
            if np.isfinite(corr) and abs(corr) >= corr_warn_thr:
                suspicious.append((rk, corr))
        if suspicious:
            suspicious_txt = ", ".join([f"{k}:{c:+.3f}" for k, c in suspicious])
            tprint(
                f"WARNING: high regime-feature corr with future returns (check leakage/OOS): {suspicious_txt}"
            )

    # Build label time ranges for uniqueness weighting
    entry_times = df["ts"].values
    exit_times = entry_times + pd.Timedelta(hours=H)  # H is the horizon
    label_times = build_label_time_ranges(
        pd.DatetimeIndex(entry_times), pd.DatetimeIndex(exit_times)
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

    # Extract selection metric values for event scoring.
    # Allow various range_pct feature names depending on generation settings.
    valid_range_metrics = [
        "range_pct",
        "range_16h_pct",
        "range_12h_pct",
        "range_8h_pct",
        "range_24h_pct",
    ]
    selection_metric_name = next((m for m in valid_range_metrics if m in feats), None)

    if not selection_metric_name:
        raise RuntimeError(
            f"Missing required selection metric in features. Expected one of {valid_range_metrics}. "
            "Fallback metrics are disabled by policy."
        )

    metric_raw = _fast_lookup_cached(
        feats[selection_metric_name],
        df["ts"].values,
        df["symbol"].values,
        lookup_cache={},
    )
    metric_raw = np.asarray(metric_raw, dtype=np.float32)
    finite = np.isfinite(metric_raw)
    finite_n = int(finite.sum())
    total_n = int(len(metric_raw))
    if finite_n <= 0:
        raise RuntimeError(
            f"Required selection metric '{selection_metric_name}' contains no finite values after alignment."
        )

    min_cov = float(cfg.get("range_pct_min_coverage", 0.95))
    cov = float(finite_n / max(total_n, 1))
    if cov < min_cov:
        raise RuntimeError(
            f"Required selection metric '{selection_metric_name}' has insufficient finite coverage: "
            f"{finite_n}/{total_n} ({cov:.1%}) < required {min_cov:.1%}."
        )

    metric_f = metric_raw[finite]
    m_std = float(np.nanstd(metric_f))
    m_span = float(np.nanpercentile(metric_f, 95) - np.nanpercentile(metric_f, 5))
    if not (m_std > 1e-8 and m_span > 1e-8):
        raise RuntimeError(
            f"Required selection metric '{selection_metric_name}' is near-constant "
            f"(std={m_std:.3e}, p95-p05={m_span:.3e}); aborting."
        )

    selection_metric_values = np.nan_to_num(metric_raw, nan=0.0)
    tprint(
        f"Extracted required selection metric '{selection_metric_name}' for event scoring "
        f"(coverage={cov:.1%})"
    )

    weights = compute_sample_weights_with_uniqueness(
        label_times=label_times,
        returns=returns,
        base_weights=base_weights,
        time_grid=time_grid,
        selection_metric=selection_metric_values,
    )

    # Optional drawdown-aware weighting proxy for faster recovery and lower ulcer behavior.
    if bool(cfg.get("sample_weight_use_drawdown_component", True)):
        dd_proxy = _stable_drawdown_proxy(returns)
        dd_component = drawdown_aware_weights(
            dd_proxy,
            k_dd=float(cfg.get("sample_weight_drawdown_k_dd", 5.0)),
            k_early=float(cfg.get("sample_weight_drawdown_k_early", 2.0)),
            tau=float(cfg.get("sample_weight_drawdown_tau", 24.0)),
        )
        weights = weights * dd_component

    # Distance-to-barrier component removed per user request
    dist_component = None

    feature_cols_for_opt = [
        c
        for c in df.columns
        if c not in {"ts", "symbol", "y_bin", "y_ret", "w"}
        and np.issubdtype(df[c].dtype, np.number)
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
            extra_components=None,  # Removed distance component per user request
            strategy=strategy,
        )

    tprint(
        f"Applied uniqueness+optimized weighting: mean={weights.mean():.3f}, std={weights.std():.3f}"
    )
    df.drop(columns=["w"], inplace=True)

    # --- Regime columns for per-regime BSS/AUC reporting ---
    # 6 regime dimensions: vol_12h, vol_48h, volume_12h, volume_48h, trend_12h, trend_48h
    # Each bucketed into 3 terciles (low/mid/high)
    _regime_map = {
        "__regime_vol_12h__": "rv_12h",
        "__regime_vol_48h__": "rv_24h",  # rv_24h is closest proxy for 48h
        "__regime_volume_12h__": "vol_z_base",  # volume z-score (short horizon)
        "__regime_volume_48h__": "vol_z24_base",  # volume z-score (longer horizon)
        "__regime_trend_12h__": "ret6h",  # 6h return as 12h trend proxy
        "__regime_trend_48h__": "trend_pct_base",  # trend pct as 48h trend proxy
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
    meta_keys_cfg = _meta_feature_keys_for_kind(cfg, strategy)
    _df_ts = df["ts"].values
    _df_sym = df["symbol"].values
    for mk in meta_keys_cfg:
        if mk in df.columns:
            df[f"__meta_raw__{mk}"] = df[mk].values
        elif mk in feats:
            df[f"__meta_raw__{mk}"] = np.nan_to_num(
                _fast_lookup(feats[mk], _df_ts, _df_sym), nan=0.0
            ).astype(np.float32)

    df = apply_interaction_toggles(
        df, feat_keys, ["G_VOL", "G_TREND"], drop_raw=cfg["drop_raw_causal"]
    )
    y_bin = df.pop("y_bin").values.astype(np.float32)
    y_ret = df.pop("y_ret").values.astype(np.float32)

    X_out = df.drop(columns=["ts", "symbol"], errors="ignore").astype(np.float32)
    X_out.index = df.index

    df_meta = (
        df[["ts", "symbol"]] if "ts" in df.columns else pd.DataFrame(index=df.index)
    )

    return X_out, y_bin, y_ret, list(X_out.columns), weights, df_meta, lbl_vals_out


def _get_bucket_label_config(cfg, side, kind):
    """Get per-bucket TP/SL/min_net_rr with fallback to global values."""
    bucket = f"{side}_{kind}"
    tp_key = f"label_tp_values_pct_{bucket}"
    sl_key = f"label_sl_values_pct_{bucket}"
    rr_key = f"label_min_net_rr_{bucket}"
    tp_vals = [
        float(x) / 100.0
        for x in cfg.get(tp_key, cfg.get("label_tp_values_pct", [3.0, 4.0, 5.0, 6.0]))
    ]
    sl_vals = [
        float(x) / 100.0
        for x in cfg.get(sl_key, cfg.get("label_sl_values_pct", [0.5, 1.0, 2.0]))
    ]
    min_net_rr = float(cfg.get(rr_key, cfg.get("label_min_net_rr", 1.2)))
    return tp_vals, sl_vals, min_net_rr


def build_grid_aggregated_tb_cache(panel, feats, cfg, horizons, strategies=None):
    """Build grid-aggregated triple-barrier labels shared across MR/TF for each (H, strategy)."""
    tb_cache = {}  # (H, side) -> (tb_labels, tb_returns)
    geom_cache = {}  # (H, side) -> {"tp_vals", "sl_vals", "n_geom"}

    if "atr_pct" in feats:
        atr_pct_df = _coerce_feature_to_panel_df(
            feats["atr_pct"], panel, "atr_pct", fill_value=0.02
        )
    else:
        atr_pct_df = None

    fee_pct = float(cfg.get("label_round_trip_fee_pct", 0.3)) / 100.0
    min_tp_hit = float(cfg.get("label_min_tp_hit_rate", 0.02))
    min_tp_hit_h2 = float(cfg.get("label_min_tp_hit_rate_h2", 0.01))
    max_timeout = float(cfg.get("label_max_timeout_rate", 0.90))
    max_timeout_h2 = float(cfg.get("label_max_timeout_rate_h2", 0.97))

    # TP/SL geometry grid — loaded once, resolved per (H, side) from per-cell data.
    # compare_tbm_parameters.py saves a per-(bucket, horizon) grid; we take the union
    # of both bucket cells sharing a given (side, H) so all needed geometries are computed.
    _tbm_grid = load_tbm_geometry_grid()
    _per_cell = _tbm_grid.get("per_cell", {})
    _global_k_tp = _tbm_grid["k_tp_grid"] if _tbm_grid["k_tp_grid"] else None
    _global_sl = _tbm_grid["sl_base_grid"] if _tbm_grid["sl_base_grid"] else None
    _grid_source = (
        "tbm_geometry_grid.csv" if _per_cell or _global_k_tp else "hardcoded defaults"
    )
    tprint(
        f"Geometry grid loaded: source={_grid_source}  per_cell_keys={sorted(_per_cell.keys())}"
    )

    def _cell_keys_for_strategy(kind: str | None, side: str, H: int) -> list[str]:
        side_l = str(side).lower()
        side_key = f"{side_l}_H{int(H)}"
        if kind in {"MR", "TF"}:
            return [side_key, f"{kind}_{side_l}_H{int(H)}"]
        return [side_key]

    def _grid_for_cell(kind: str | None, side: str, H: int):
        """Return (k_tp_list, sl_list, tp_abs_lo_pct_or_None) for a specific (kind, side, H) cell.
        kind is 'MR'/'TF' or None; cell_key format is 'MR_long_H4'.
        Falls back to global grid, then cfg overrides, then hardcoded defaults.
        tp_abs_lo_pct_or_None is None when no per-cell value is available (caller uses global tp_lo).
        """
        cell_keys = _cell_keys_for_strategy(kind, side, H)
        cells = [
            _per_cell.get(cell_key) for cell_key in cell_keys if cell_key in _per_cell
        ]
        if cells:
            k_vals = sorted(
                {float(v) for cell in cells for v in (cell.get("k_tp_grid") or [])}
            )
            sl_vals = sorted(
                {float(v) for cell in cells for v in (cell.get("sl_base_grid") or [])}
            )
            tp_abs_vals = [
                float(cell.get("tp_abs_lo_pct"))
                for cell in cells
                if cell.get("tp_abs_lo_pct") is not None
            ]
            if k_vals and sl_vals:
                return (
                    k_vals,
                    sl_vals,
                    min(tp_abs_vals) if tp_abs_vals else None,
                )
        if len(cell_keys) == 1:
            cell = _per_cell.get(cell_keys[0])
        else:
            cell = None
        if cell and cell.get("k_tp_grid") and cell.get("sl_base_grid"):
            return (
                sorted(cell["k_tp_grid"]),
                sorted(cell["sl_base_grid"]),
                cell.get("tp_abs_lo_pct"),
            )
        # Fall back to global grid, then cfg overrides, then hardcoded defaults
        k_fallback = cfg.get(
            "barrier_k_tp_grid", _global_k_tp or [0.8, 1.0, 1.25, 1.6, 2.0, 2.5]
        )
        s_fallback = cfg.get("barrier_sl_base_grid", _global_sl or [0.5, 1.0, 1.5])
        return k_fallback, s_fallback, None

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
    sl_mult_lo = float(cfg.get("barrier_sl_mult_lo", 0.4))
    sl_mult_hi = float(cfg.get("barrier_sl_mult_hi", 0.7))
    sl_lo = float(cfg.get("barrier_sl_lo", 0.005))
    sl_hi = float(cfg.get("barrier_sl_hi", 0.06))
    z_gate = float(cfg.get("barrier_z_gate", 1.0))
    H_base = float(cfg.get("label_horizon_base", 4))
    tp_lo = float(cfg.get("barrier_tp_lo", 0.02))
    tp_lo_h2 = float(cfg.get("barrier_tp_lo_h2", 0.015))
    tp_hi = float(cfg.get("barrier_tp_hi", 0.06))

    # Cache raw triple barrier results per (H, side, k_tp, sl_base_mult) to avoid recomputation.
    _raw_tb_cache = {}
    _raw_tb_payload_cache = {}
    _raw_tb_payload_pinned = {}
    _raw_tb_payload_cache_max_bytes = max(
        0,
        int(float(cfg.get("label_raw_tb_payload_cache_mb", 2048.0)) * 1024 * 1024),
    )
    _kinds = ["mr", "tf"]
    _prod_events_rows = []
    _prod_cell_metrics = {}
    _diag_labels = bool(cfg.get("label_diagnostics_mode", False))

    def _matrix_to_array(_obj, dtype=None):
        if _obj is None:
            return None
        if isinstance(_obj, pd.DataFrame):
            return _obj.to_numpy(dtype=dtype, copy=False)
        _arr = np.asarray(_obj)
        if dtype is not None and _arr.dtype != np.dtype(dtype):
            _arr = _arr.astype(dtype, copy=False)
        return _arr

    def _raw_tb_payload_nbytes(payload: dict) -> int:
        _total = 0
        for _k in ("lbl", "ret", "qual", "tp_vals", "sl_vals"):
            _v = payload.get(_k)
            if _v is None:
                continue
            try:
                _total += int(np.asarray(_v).nbytes)
            except Exception:
                pass
        return int(_total)

    def _trim_raw_tb_payload_cache() -> None:
        if _raw_tb_payload_cache_max_bytes <= 0:
            _raw_tb_payload_cache.clear()
            return
        _total = sum(
            int(_payload.get("nbytes", 0))
            for _payload in _raw_tb_payload_cache.values()
        )
        if _total <= _raw_tb_payload_cache_max_bytes:
            return
        for _key, _payload in sorted(
            _raw_tb_payload_cache.items(),
            key=lambda _item: (
                float(_item[1].get("score", 0.0)),
                float(_item[1].get("ts", 0.0)),
            ),
        ):
            _total -= int(_payload.get("nbytes", 0))
            _raw_tb_payload_cache.pop(_key, None)
            if _total <= _raw_tb_payload_cache_max_bytes:
                break

    def _cache_raw_tb_payload(raw_key: tuple, lbl, ret, qual, tp_vals, sl_vals, score: float) -> None:
        if raw_key is None or _raw_tb_payload_cache_max_bytes <= 0:
            return
        _payload = {
            "lbl": np.ascontiguousarray(_matrix_to_array(lbl, np.int8)),
            "ret": np.ascontiguousarray(_matrix_to_array(ret, np.float32)),
            "qual": np.ascontiguousarray(_matrix_to_array(qual, np.float32)),
            "tp_vals": np.ascontiguousarray(_matrix_to_array(tp_vals, np.float32)),
            "sl_vals": np.ascontiguousarray(_matrix_to_array(sl_vals, np.float32)),
            "score": float(score),
            "ts": time.perf_counter(),
        }
        _payload["nbytes"] = _raw_tb_payload_nbytes(_payload)
        _raw_tb_payload_cache[raw_key] = _payload
        _trim_raw_tb_payload_cache()

    def _refresh_pinned_raw_payloads(_pool, _keep_top_n: int) -> None:
        _raw_tb_payload_pinned.clear()
        if not _pool or _keep_top_n <= 0:
            return
        _top = sorted(
            _pool,
            key=lambda g: (
                float(g.get("rr_weight", 0.0)),
                float(g.get("auc_bound", 0.0)),
                float(g.get("tp_sep_top10", 0.0)),
                float(g.get("bind", 0.0)),
                float(g.get("tp_over_sl", 0.0)),
            ),
            reverse=True,
        )[: max(1, int(_keep_top_n))]
        for _g in _top:
            _rk = _g.get("raw_key")
            if _rk is None:
                continue
            _payload = _raw_tb_payload_cache.get(_rk)
            if _payload is not None:
                _raw_tb_payload_pinned[_rk] = _payload

    # -------------------------------------------------------------------------------------
    # OPTIMIZATION: Hoist ATR and Barrier Base Calculations (15x Speedup)
    # -------------------------------------------------------------------------------------
    # atr_pct and _barrier_base_cache have zero dependency on H (Horizon), side, or kind.
    # Computing them once here prevents calculating a rolling 720-bar median/MAD 16 different times.
    if "atr_pct" in feats:
        atr_pct_df = _coerce_feature_to_panel_df(
            feats["atr_pct"], panel, "atr_pct", fill_value=0.02
        )
        atr_pct_local = atr_pct_df.fillna(0.02)
    else:
        atr_pct_df = None
        atr_pct_local = pd.DataFrame(
            0.02, index=panel["close"].index, columns=panel["close"].columns
        )

    # Gather all unique `atr_window` values used anywhere in the geometry grid
    _all_windows = set()
    for _k, _cell_data in _per_cell.items():
        _triplets = _cell_data.get("validated_triplets", [])
        if _triplets:
            for t in _triplets:
                _all_windows.add(t[2])
        elif _cell_data.get("atr_window"):
            _all_windows.add(_cell_data["atr_window"])
    _all_windows.add(
        _tbm_grid.get("atr_window") or int(cfg.get("barrier_atr_window", 24 * 30))
    )

    _barrier_base_cache: dict = {}
    for _win in sorted(_all_windows):
        tprint(f"Pre-computing global barrier base (window={_win}h)...")
        _barrier_base_cache[_win] = _compute_barrier_base(
            atr_pct_local,
            _win,
            disp_floor,
            z_max,
            k_reg,
            m_lo,
            m_hi,
            sl_mult_lo,
            sl_mult_hi,
            z_gate,
            use_standalone_sl=cfg.get("use_standalone_sl", False),
        )
    # -------------------------------------------------------------------------------------

    def _resolve_cell_min_tp_hit(
        side: str, kind: str, h: int, default_val: float
    ) -> float:
        """Resolve per-cell TP-hit threshold with flexible key fallbacks.

        Supported keys (first found wins), e.g. for kind='mr', side='long', h=4:
          - label_min_tp_hit_rate_mr_long_h4
          - label_min_tp_hit_rate_mr_long
          - label_min_tp_hit_rate_long_mr_h4
          - label_min_tp_hit_rate_long_mr
          - label_min_tp_hit_rate_MR_long_H4
        """
        kind_l = str(kind).lower()
        side_l = str(side).lower()
        key_candidates = [
            f"label_min_tp_hit_rate_{kind_l}_h{int(h)}",
            f"label_min_tp_hit_rate_{kind_l}",
            f"label_min_tp_hit_rate_{kind_l}_h{int(h)}",
            f"label_min_tp_hit_rate_{kind_l}",
            f"label_min_tp_hit_rate_{str(kind).upper()}_H{int(h)}",
        ]
        for _k in key_candidates:
            if _k in cfg:
                try:
                    return float(cfg.get(_k, default_val))
                except Exception:
                    continue
        return float(default_val)

    def _resolve_cell_tp_floor(
        side: str, kind: str, h: int, default_val: float, mode: str
    ) -> float:
        """Resolve per-cell TP floor for mode in {'search','prod'} with fallback keys."""
        kind_l = str(kind).lower()
        side_l = str(side).lower()
        key_candidates = [
            f"barrier_tp_lo_{mode}_{kind_l}_h{int(h)}",
            f"barrier_tp_lo_{mode}_{kind_l}",
            f"barrier_tp_lo_{mode}_{kind_l}_h{int(h)}",
            f"barrier_tp_lo_{mode}_{kind_l}",
            f"barrier_tp_lo_{mode}_{str(kind).upper()}_H{int(h)}",
        ]
        for _k in key_candidates:
            if _k in cfg:
                try:
                    return float(cfg.get(_k, default_val))
                except Exception:
                    continue
        return float(default_val)

    def _co_calibrate_tp_floor(
        tp_floor: float, sl_base_mult: float, tp_hi_local: float
    ) -> float:
        """Co-calibrate TP floor with SL geometry so floors are tuned jointly.

        NOTE: Co-calibration disabled (alpha=0.0) to make TP and SL parameters independent.
        This addresses floor dominance issue by preventing SL-based TP floor increases.
        """
        alpha = float(
            cfg.get("barrier_tp_lo_sl_cocalib_alpha", 0.0)
        )  # Changed from 0.30 to 0.0
        sl_ref = float(cfg.get("barrier_sl_base_mult_ref", 0.5))
        # Higher SL multiple -> require a somewhat higher TP floor to preserve economic edge.
        # DISABLED: No longer scale TP floor based on SL geometry
        scale = 1.0 + alpha * ((float(sl_base_mult) - sl_ref) / max(sl_ref, 1e-9))
        scale = float(np.clip(scale, 0.6, 1.8))
        out = float(tp_floor) * scale
        return float(np.clip(out, 1e-4, tp_hi_local))

    def _quality_metrics_from_proxy(lbl_df, ret_df, qual_df):
        """Compute bound-event quality proxies from current geometry outputs."""
        qual_vals = qual_df.values.astype(np.float64, copy=False).ravel()
        lbl_vals = lbl_df.values.ravel()
        ret_vals = ret_df.values.astype(np.float64, copy=False).ravel()
        finite_q = np.isfinite(qual_vals)
        if finite_q.any():
            qv = qual_vals[finite_q]
            y_tp_q = (lbl_vals[finite_q] == OUT_TP).astype(np.float64)
            y_bound_q = (lbl_vals[finite_q] != OUT_TO).astype(np.float64)
            bmask = y_bound_q.astype(bool)
            if bmask.sum() >= 10:
                qb = qv[bmask]
                yb = y_tp_q[bmask]
                n_pos_b = int(yb.sum())
                n_neg_b = int(len(yb) - n_pos_b)
                if n_pos_b > 0 and n_neg_b > 0:
                    from sklearn.metrics import roc_auc_score

                    auc_bound = float(roc_auc_score(yb, qb))
                else:
                    auc_bound = 0.5
            else:
                auc_bound = 0.5
            if len(qv) >= 10:
                q_thr = float(np.quantile(qv, 0.90))
                top_mask = qv >= q_thr
                tp_top = float(y_tp_q[top_mask].mean()) if top_mask.any() else 0.0
                tp_rest = float(y_tp_q[~top_mask].mean()) if (~top_mask).any() else 0.0
                tp_sep_top10 = tp_top - tp_rest
            else:
                tp_sep_top10 = 0.0
        else:
            auc_bound = 0.5
            tp_sep_top10 = 0.0
        tp_ret = ret_vals[lbl_vals == OUT_TP]
        sl_ret = ret_vals[lbl_vals == OUT_SL]
        if tp_ret.size > 0 and sl_ret.size > 0:
            er_tp = float(np.mean(tp_ret))
            er_sl = float(np.mean(sl_ret))
            tp_over_sl = er_tp / max(abs(er_sl), 1e-9)
        else:
            tp_over_sl = 0.0
        return float(auc_bound), float(tp_sep_top10), float(tp_over_sl)

    def _summarize_geom_triplet(
        *,
        k_tp: float,
        sl_base_mult: float,
        atr_window: int,
        rr_weight: float,
        tp_hit_raw: float,
        sl_hit_raw: float,
        timeout_raw: float,
        n_events_raw: int,
        n_candidates: int,
        n_rr_kept: int,
        auc_bound: float,
        tp_sep_top10: float,
        bind_raw: float,
        tp_over_sl: float,
        tp_guard_target: float,
        tp_emp_base: float,
        tp_floor_share: float,
        tp_ceil_share: float,
        raw_key: tuple,
    ) -> dict:
        return {
            "k_tp": float(k_tp),
            "sl_base_mult": float(sl_base_mult),
            "atr_window": int(atr_window),
            "rr_weight": float(max(rr_weight, 1e-6)),
            "tp_hit": float(tp_hit_raw),
            "sl_hit": float(sl_hit_raw),
            "to_rate": float(timeout_raw),
            "n_events": int(n_events_raw),
            "tp_hit_raw": float(tp_hit_raw),
            "sl_hit_raw": float(sl_hit_raw),
            "timeout_raw": float(timeout_raw),
            "n_raw": int(n_events_raw),
            "tp_hit_kept": float("nan"),
            "sl_hit_kept": float("nan"),
            "timeout_kept": float("nan"),
            "n_candidates": int(n_candidates),
            "n_rr_kept": int(n_rr_kept),
            "auc_bound": float(auc_bound),
            "tp_sep_top10": float(tp_sep_top10),
            "bind": float(bind_raw),
            "bind_raw": float(bind_raw),
            "tp_over_sl": float(tp_over_sl),
            "tp_guard_target": float(tp_guard_target),
            "tp_emp_base": float(tp_emp_base),
            "tp_floor_share": float(tp_floor_share),
            "tp_ceil_share": float(tp_ceil_share),
            "raw_key": raw_key,
        }

    def _materialize_geom_triplet(_g: dict, _side: str, _H: int) -> dict:
        _raw_key = _g.get("raw_key")
        _cached_payload = _raw_tb_payload_pinned.get(_raw_key)
        if _cached_payload is None:
            _cached_payload = _raw_tb_payload_cache.get(_raw_key)
        if _cached_payload is not None:
            return {
                **_g,
                "tp_floor_search": float(_g.get("tp_floor_search", np.nan)),
                "tp_floor_prod": float(_g.get("tp_floor_prod", np.nan)),
                "lbl": _cached_payload["lbl"],
                "ret": _cached_payload["ret"],
                "qual": _cached_payload["qual"],
                "tp_vals": _cached_payload["tp_vals"],
                "sl_vals": _cached_payload["sl_vals"],
            }
        _sl_base = float(_g["sl_base_mult"])
        _tp_lo_eval = _co_calibrate_tp_floor(tp_lo_eff_search, _sl_base, tp_hi)
        _tp_lo_prod_eval = _co_calibrate_tp_floor(tp_lo_eff_prod, _sl_base, tp_hi)
        _tp_lo_final_prod = (
            _tp_lo_prod_eval
            if bool(cfg.get("label_use_production_tp_floor", True))
            else _tp_lo_eval
        )
        _audit_win = int(_g.get("atr_window", _atr_window))
        _audit_base = _barrier_base_cache.get(_audit_win)
        if _audit_base is None:
            _audit_base = _barrier_base_cache.get(
                int(cfg.get("barrier_atr_window", 24 * 30))
            )
        _tp_df_a, _sl_df_a = compute_barrier_factory(
            atr_pct=atr_pct_local,
            window_size=_audit_win,
            k_tp=float(_g["k_tp"]),
            sl_base_mult=_sl_base,
            horizon=_H,
            H_base=H_base,
            disp_floor=disp_floor,
            z_max=z_max,
            k_reg=k_reg,
            m_lo=m_lo,
            m_hi=m_hi,
            sl_mult_lo=sl_mult_lo,
            sl_mult_hi=sl_mult_hi,
            sl_lo=sl_lo,
            sl_hi=sl_hi,
            z_gate=z_gate,
            tp_lo=_tp_lo_eval,
            tp_hi=tp_hi,
            _base=_audit_base,
        )
        _tb_out = compute_triple_barrier_labels(
            panel,
            _tp_df_a,
            _sl_df_a,
            _H,
            side=_side,
            return_outcomes=True,
            return_path_stats=False,
        )
        _lbl_mat, _ret_mat, _qual_mat = _tb_out[:3]
        return {
            **_g,
            "tp_floor_search": float(_tp_lo_eval),
            "tp_floor_prod": float(_tp_lo_final_prod),
            "lbl": np.ascontiguousarray(_matrix_to_array(_lbl_mat, np.int8)),
            "ret": np.ascontiguousarray(_matrix_to_array(_ret_mat, np.float32)),
            "qual": np.ascontiguousarray(_matrix_to_array(_qual_mat, np.float32)),
            "tp_vals": np.ascontiguousarray(_matrix_to_array(_tp_df_a, np.float32)),
            "sl_vals": np.ascontiguousarray(_matrix_to_array(_sl_df_a, np.float32)),
        }

    def _aggregate_geom_runs(_runs):
        if not _runs:
            return None
        rr_weights_raw = np.array([g["rr_weight"] for g in _runs], dtype=np.float32)
        rr_weights = np.sqrt(rr_weights_raw).astype(np.float32)
        rr_weights = rr_weights / (rr_weights.mean() + 1e-12)
        _atr_med = _barrier_base["atr_median"].values
        _atr_pct = atr_pct_local.values
        atr_ratio = np.divide(
            _atr_pct,
            _atr_med,
            out=np.ones_like(_atr_pct),
            where=_atr_med > 1e-12,
        )

        k_tps_raw = np.array([g["k_tp"] for g in _runs], dtype=np.float32)
        k_min, k_max = k_tps_raw.min(), k_tps_raw.max()
        if k_max > k_min:
            k_tps_norm = 2.0 * (k_tps_raw - k_min) / (k_max - k_min) - 1.0
        else:
            k_tps_norm = np.zeros_like(k_tps_raw)

        first_lbl = _matrix_to_array(_runs[0]["lbl"], np.int8)
        _shape = first_lbl.shape
        w_tp = np.zeros(_shape, dtype=np.float32)
        w_sl = np.zeros(_shape, dtype=np.float32)
        w_to = np.zeros(_shape, dtype=np.float32)
        agg_ret_num = np.zeros(_shape, dtype=np.float32)
        agg_qual_num = np.zeros(_shape, dtype=np.float32)
        tp_vals_num = np.zeros(_shape, dtype=np.float32)
        sl_vals_num = np.zeros(_shape, dtype=np.float32)

        for _idx, _g in enumerate(_runs):
            _lbl = _matrix_to_array(_g["lbl"], np.int8)
            _ret = _matrix_to_array(_g["ret"], np.float32)
            _qual = _matrix_to_array(_g["qual"], np.float32)
            _tp_vals = _matrix_to_array(_g["tp_vals"], np.float32)
            _sl_vals = _matrix_to_array(_g["sl_vals"], np.float32)
            _dyn_exp = 0.5 * (atr_ratio - 1.0) * k_tps_norm[_idx]
            _dyn_exp = np.clip(_dyn_exp, -20.0, 20.0)
            _w = rr_weights[_idx] * np.exp(_dyn_exp).astype(np.float32, copy=False)
            _mask_tp = (_lbl == OUT_TP).astype(np.float32, copy=False)
            _mask_sl = (_lbl == OUT_SL).astype(np.float32, copy=False)
            _mask_to = (_lbl == OUT_TO).astype(np.float32, copy=False)
            w_tp += _w * _mask_tp
            w_sl += _w * _mask_sl
            w_to += _w * _mask_to
            agg_ret_num += _w * _ret
            agg_qual_num += _w * _qual
            tp_vals_num += _w * _tp_vals
            sl_vals_num += _w * _sl_vals

        w_sum = w_tp + w_sl + w_to
        _denom_safe = np.where(w_sum > 0, w_sum, 1.0)
        tp_vals_df = pd.DataFrame(
            (tp_vals_num / _denom_safe).astype(np.float32, copy=False),
            index=panel["close"].index,
            columns=panel["close"].columns,
        )
        sl_vals_df = pd.DataFrame(
            (sl_vals_num / _denom_safe).astype(np.float32, copy=False),
            index=panel["close"].index,
            columns=panel["close"].columns,
        )
        denom = np.where(w_sum > 0, w_sum, 1.0)
        agg_ret = (agg_ret_num / denom).astype(np.float32)
        agg_qual = (agg_qual_num / denom).astype(np.float32)
        agg_lbl = np.where(
            w_tp > w_sl,
            OUT_TP,
            np.where(w_sl > w_tp, OUT_SL, OUT_TO),
        ).astype(np.int8)

        return {
            "agg_lbl": agg_lbl,
            "agg_ret": agg_ret,
            "agg_qual": agg_qual,
            "tp_vals_df": tp_vals_df,
            "sl_vals_df": sl_vals_df,
        }

    def _materialize_geom_aggregate(_runs, _cache_key):
        _agg = _aggregate_geom_runs(_runs)
        if _agg is None:
            return

        tb_cache[_cache_key] = (
            pd.DataFrame(
                _agg["agg_lbl"].astype(np.int8, copy=False),
                index=panel["close"].index,
                columns=panel["close"].columns,
            ),
            pd.DataFrame(
                _agg["agg_ret"].astype(np.float32, copy=False),
                index=panel["close"].index,
                columns=panel["close"].columns,
            ),
            pd.DataFrame(
                _agg["agg_qual"].astype(np.float32, copy=False),
                index=panel["close"].index,
                columns=panel["close"].columns,
            ),
        )
        geom_cache[_cache_key] = {
            "tp_vals": _agg["tp_vals_df"],
            "sl_vals": _agg["sl_vals_df"],
            "n_geom": len(_runs),
        }

    # ── ALPHA MODELS (long/short × mr/tf × horizons) ──
    # Note: Using horizons=horizons for explicit control.
    horizons = horizons or list(CANON_HORIZONS)
    strategies = list(strategies) if strategies is not None else get_strategies(cfg)
    for strat in strategies:
        side = strat["trade_side"]
        s_id = strat["strategy_id"]
        # Resolve MR/TF kind from strategy flags, for cell_key lookup in geometry grid
        kind = (
            "MR"
            if strat.get("is_mr", False)
            else ("TF" if strat.get("is_tf", False) else None)
        )
        k_label = s_id  # keep strategy_id for artifact keys
        for H in horizons:
            # _grid_for_cell must come first — _cell_tp_abs_lo is used in floor resolution below.
            # Use the canonical kind key (MR/TF) for grid lookup
            tp_mults, sl_base_mults, _cell_tp_abs_lo = _grid_for_cell(
                kind, side, int(H)
            )
            tprint(
                f"[TB Cache] H={H} strategy={s_id} kind={kind or 'SIDE'} side={side}: looking up geometry grid..."
            )

            _default_tp_hit = min_tp_hit_h2 if int(H) == 2 else min_tp_hit
            min_tp_hit_eff = _resolve_cell_min_tp_hit(
                side=side, kind=k_label, h=int(H), default_val=_default_tp_hit
            )
            # Separate search-vs-production TP floors.
            # Per-cell tp_abs_lo_pct from the geometry grid takes priority over global cfg value —
            # this is the floor the optimizer actually selected for this (kind, side, H) cell.
            _global_tp_lo_base = tp_lo_h2 if int(H) == 2 else tp_lo
            _cell_tp_lo_base = (
                float(_cell_tp_abs_lo)
                if _cell_tp_abs_lo is not None
                else _global_tp_lo_base
            )
            tp_lo_search_default = float(
                cfg.get(
                    "barrier_tp_lo_search_h2"
                    if int(H) == 2
                    else "barrier_tp_lo_search",
                    _cell_tp_lo_base,
                )
            )
            tp_lo_prod_default = float(
                cfg.get(
                    "barrier_tp_lo_prod_h2" if int(H) == 2 else "barrier_tp_lo_prod",
                    max(_cell_tp_lo_base, tp_lo_search_default),
                )
            )
            tp_lo_eff_search = _resolve_cell_tp_floor(
                side=side,
                kind=k_label,
                h=int(H),
                default_val=tp_lo_search_default,
                mode="search",
            )
            tp_lo_eff_prod = _resolve_cell_tp_floor(
                side=side,
                kind=k_label,
                h=int(H),
                default_val=tp_lo_prod_default,
                mode="prod",
            )
            # Optional horizon scaling for TP floors.
            if bool(cfg.get("barrier_tp_lo_scale_with_horizon", True)):
                _h_scale = float(np.sqrt(max(float(H), 1.0) / max(float(H_base), 1.0)))
                tp_lo_eff_search *= _h_scale
                tp_lo_eff_prod *= _h_scale
            # Production floor should not be looser than search floor by default.
            tp_lo_eff_prod = max(tp_lo_eff_prod, tp_lo_eff_search)
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
            clip_stats = {
                "tp_floor_shares": [],
                "tp_ceil_shares": [],
            }
            # ── Per-cell/bucket geometry override (Native downstream integration) ────────
            # If per-cell winners exist (from current optimization run), use the
            # specific winning triplet for this exact bucket-horizon cell.
            # Bucket mapping: (long, tf) -> TF_long, (short, mr) -> MR_short, etc.
            # Use canonical kind prefix for cell key _bname
            _cell_keys_exact = _cell_keys_for_strategy(kind, side, H)
            _cell_key_exact = ",".join(_cell_keys_exact)

            # Load per-cell mapping for a diagnostic tag only; do NOT collapse to a single
            # triplet — let the full geometry grid validated_triplets drive the sweep.
            if kind is None:
                _p_cell_best = load_tbm_best_params_per_side_horizon()
                _p_cell_all = load_tbm_all_params_per_side_horizon()
            else:
                _p_cell_best = load_tbm_best_params_per_cell()
                _p_cell_all = load_tbm_all_params_per_cell()
            _cell_winner = None
            for _ck in _cell_keys_exact:
                _cell_winner = _p_cell_best.get(_ck)
                if _cell_winner:
                    break
            _cell_ranked = []
            for _ck in _cell_keys_exact:
                _cell_ranked.extend(_p_cell_all.get(_ck, []))

            # Check for validated triplets first (geometry grid — top-k per cell).
            _validated_triplets = []
            for _ck in _cell_keys_exact:
                _cell_data = _per_cell.get(_ck, {})
                for _triplet in _cell_data.get("validated_triplets") or []:
                    if _triplet not in _validated_triplets:
                        _validated_triplets.append(_triplet)

            # Diagnostic: log how many validated triplets we're sweeping.
            if _cell_winner:
                _tag = "native_cell" if "cell_key" in _cell_winner else "native_bucket"
                tprint(
                    f"  [{_tag}] {_cell_key_exact}: sweeping {len(_validated_triplets)} "
                    f"validated geometry triplets (strat={s_id}). "
                    f"Best single config: k_tp={_cell_winner.get('k_tp')} sl={_cell_winner.get('sl_as_tp_pct')}"
                )

                if not _validated_triplets:
                    _fallback_win = int(
                        (_cell_winner or {}).get("base_atr_window")
                        or next(
                            (
                                _per_cell.get(_ck, {}).get("atr_window")
                                for _ck in _cell_keys_exact
                                if _per_cell.get(_ck, {}).get("atr_window") is not None
                            ),
                            None,
                        )
                        or _tbm_grid.get("atr_window")
                        or int(cfg.get("barrier_atr_window", 24 * 30))
                    )
                    _fallback_k = (_cell_winner or {}).get("k_tp")
                    _fallback_sl = (_cell_winner or {}).get("sl_as_tp_pct")
                    if _fallback_k is not None and _fallback_sl is not None:
                        if _cell_ranked:
                            _validated_triplets = []
                            for _row in _cell_ranked:
                                _rk = _row.get("k_tp")
                                _rs = _row.get("sl_as_tp_pct")
                                _rw = int(_row.get("base_atr_window", _fallback_win))
                                if _rk is None or _rs is None:
                                    continue
                                _triplet = (float(_rk), float(_rs), _rw)
                                if _triplet not in _validated_triplets:
                                    _validated_triplets.append(_triplet)
                        else:
                            _validated_triplets = [
                                (float(_fallback_k), float(_fallback_sl), _fallback_win)
                            ]
                        tprint(
                            f"  [native_fallback] {_cell_key_exact}: no validated triplets; "
                            f"using {len(_validated_triplets)} persisted per-cell config(s); "
                            f"best k_tp={_fallback_k} sl={_fallback_sl} atr={_fallback_win}"
                        )
                    else:
                        _validated_triplets = [
                            (k, s, _fallback_win)
                            for k in tp_mults
                            for s in sl_base_mults
                        ]

                _cell_windows = sorted(set(t[2] for t in _validated_triplets))
                _keep_all_lte = int(cfg.get("label_geom_keep_all_if_lte_per_cell", 6))
                _keep_top_n = max(1, int(cfg.get("label_geom_keep_topn_per_cell", 6)))
                _skip_geom_eval = len(_validated_triplets) > 0 and len(
                    _validated_triplets
                ) <= max(0, min(_keep_all_lte, _keep_top_n))
                tprint(
                    f"Pre-computing geometry labels H={H} strat={s_id} kind={kind} side={side} "
                    f"(triplets={len(_validated_triplets)}, atr_windows={_cell_windows})..."
                )
                if _skip_geom_eval:
                    tprint(
                        f"Geometry selection bypass: H={H} strat={s_id} side={side} "
                        f"keeping all {len(_validated_triplets)} validated triplets "
                        f"(threshold={min(_keep_all_lte, _keep_top_n)})"
                    )

                geom_runs = []
                relaxed_pool = []
                _cell_start_ts = time.perf_counter()
                _hb_last_ts = _cell_start_ts
                _hb_every = max(1, int(cfg.get("label_geom_heartbeat_every", 4)))
                _hb_secs = max(5.0, float(cfg.get("label_geom_heartbeat_secs", 60.0)))

                # OPTIMIZATION: Cache barrier factory outputs by (atr_window, k_tp, sl_base_mult, H, tp_lo, tp_hi)
                # This avoids redundant computation when same geometry is used across different (side, kind) combinations
                _barrier_factory_cache = (
                    {}
                    if "_barrier_factory_cache" not in dir()
                    else _barrier_factory_cache
                )

                for _triplet_idx, (k_tp, sl_base_mult, _atr_window) in enumerate(
                    _validated_triplets, start=1
                ):
                    _now_ts = time.perf_counter()
                    if (
                        _triplet_idx == 1
                        or _triplet_idx == len(_validated_triplets)
                        or (_triplet_idx % _hb_every == 0)
                        or ((_now_ts - _hb_last_ts) >= _hb_secs)
                    ):
                        tprint(
                            f"[geom hb] strat={s_id} side={side} H={H} "
                            f"triplet={_triplet_idx}/{len(_validated_triplets)} "
                            f"accepted={len(relaxed_pool)} "
                            f"elapsed={_now_ts - _cell_start_ts:.1f}s"
                        )
                        _hb_last_ts = _now_ts
                    _barrier_base = _barrier_base_cache[_atr_window]
                    total_geoms += 1
                    tp_lo_eval = _co_calibrate_tp_floor(
                        tp_lo_eff_search, sl_base_mult, tp_hi
                    )

                    # OPTIMIZATION: Use cached barriers if available
                    _bf_cache_key = (
                        int(_atr_window),
                        round(float(k_tp), 4),
                        round(float(sl_base_mult), 4),
                        int(H),
                        round(float(tp_lo_eval), 6),
                        round(float(tp_hi), 6),
                    )
                    if _bf_cache_key in _barrier_factory_cache:
                        tp_df, sl_df = _barrier_factory_cache[_bf_cache_key]
                    else:
                        tp_df, sl_df = compute_barrier_factory(
                            atr_pct=atr_pct_local,
                            window_size=_atr_window,
                            k_tp=k_tp,
                            sl_base_mult=sl_base_mult,
                            horizon=H,
                            H_base=H_base,
                            disp_floor=disp_floor,
                            z_max=z_max,
                            k_reg=k_reg,
                            m_lo=m_lo,
                            m_hi=m_hi,
                            sl_mult_lo=sl_mult_lo,
                            sl_mult_hi=sl_mult_hi,
                            sl_lo=sl_lo,
                            sl_hi=sl_hi,
                            z_gate=z_gate,
                            tp_lo=tp_lo_eval,
                            tp_hi=tp_hi,
                            _base=_barrier_base,
                        )
                        _barrier_factory_cache[_bf_cache_key] = (tp_df, sl_df)
                    _tp_vals = tp_df.values
                    _tp_floor_share = float(np.mean(_tp_vals <= (tp_lo_eval + 1e-9)))
                    _tp_ceil_share = float(np.mean(_tp_vals >= (tp_hi - 1e-9)))
                    clip_stats["tp_floor_shares"].append(_tp_floor_share)
                    clip_stats["tp_ceil_shares"].append(_tp_ceil_share)

                    net_rr = k_tp / max(sl_base_mult + fee_pct / (k_tp + 1e-9), 1e-9)
                    if net_rr < min_net_rr_eff:
                        reject_counts["rr"] += 1
                        continue
                    raw_key = (
                        int(H),
                        str(side),
                        int(_atr_window),
                        round(float(k_tp), 4),
                        round(float(sl_base_mult), 4),
                        round(float(tp_lo_eval), 6),
                        round(float(tp_hi), 6),
                    )

                    _tb_summary = _raw_tb_cache.get(raw_key)
                    if _tb_summary is None:
                        if (
                            _triplet_idx == 1
                            or _triplet_idx == len(_validated_triplets)
                            or (_triplet_idx % _hb_every == 0)
                            or ((_now_ts - _hb_last_ts) >= _hb_secs)
                        ):
                            tprint(
                                f"[geom raw] strat={s_id} side={side} H={H} "
                                f"triplet={_triplet_idx}/{len(_validated_triplets)} "
                                f"computing raw triple-barrier labels..."
                            )
                        _tb_out = compute_triple_barrier_labels(
                            panel,
                            tp_df,
                            sl_df,
                            H,
                            side=side,
                            return_outcomes=True,
                            return_path_stats=False,
                        )
                        lbl, ret, qual = _tb_out[:3]
                        n_events_raw = int(lbl.size)
                        tp_hit_raw = float((lbl.values == 2).sum()) / max(1, n_events_raw)
                        sl_hit_raw = float((lbl.values == 0).sum()) / max(1, n_events_raw)
                        timeout_raw = float((lbl.values == 1).sum()) / max(1, n_events_raw)
                        auc_bound, tp_sep_top10, tp_over_sl = _quality_metrics_from_proxy(
                            lbl, ret, qual
                        )
                        bind_raw = tp_hit_raw + sl_hit_raw
                        _tp_ts = (lbl.values == OUT_TP).mean(axis=1).astype(np.float64)
                        _tp_roll_win = int(
                            cfg.get("label_tp_hit_guardrail_roll_hours", 24 * 14)
                        )
                        _tp_roll = (
                            pd.Series(_tp_ts)
                            .rolling(
                                _tp_roll_win,
                                min_periods=max(24, _tp_roll_win // 8),
                            )
                            .mean()
                        )
                        _tp_emp_base = (
                            float(
                                _tp_roll.quantile(
                                    float(
                                        cfg.get(
                                            "label_tp_hit_guardrail_quantile", 0.50
                                        )
                                    )
                                )
                            )
                            if _tp_roll.notna().any()
                            else float(tp_hit_raw)
                        )
                        if (
                            _triplet_idx == 1
                            or _triplet_idx == len(_validated_triplets)
                            or (_triplet_idx % _hb_every == 0)
                            or ((_now_ts - _hb_last_ts) >= _hb_secs)
                        ):
                            tprint(
                                f"[geom raw] strat={s_id} side={side} H={H} "
                                f"triplet={_triplet_idx}/{len(_validated_triplets)} "
                                f"raw labels ready ({lbl.size} events)"
                            )
                        _tb_summary = {
                            "n_events_raw": int(n_events_raw),
                            "tp_hit_raw": float(tp_hit_raw),
                            "sl_hit_raw": float(sl_hit_raw),
                            "timeout_raw": float(timeout_raw),
                            "auc_bound": float(auc_bound),
                            "tp_sep_top10": float(tp_sep_top10),
                            "tp_over_sl": float(tp_over_sl),
                            "bind_raw": float(bind_raw),
                            "tp_emp_base": float(_tp_emp_base),
                        }
                        _raw_tb_cache[raw_key] = _tb_summary
                        _cache_raw_tb_payload(
                            raw_key,
                            lbl,
                            ret,
                            qual,
                            tp_df,
                            sl_df,
                            score=float(_tb_summary.get("auc_bound", 0.5)),
                        )
                        del lbl, ret, qual
                    n_events_raw = int(_tb_summary["n_events_raw"])
                    tp_hit_raw = float(_tb_summary["tp_hit_raw"])
                    sl_hit_raw = float(_tb_summary["sl_hit_raw"])
                    timeout_raw = float(_tb_summary["timeout_raw"])

                    # Kept-universe counters are unavailable at this stage (pre candidate/rr filters).
                    n_candidates = -1
                    n_rr_kept = -1

                    if n_events_raw < min_events_eff:
                        reject_counts["n_events"] += 1
                        continue

                    # Track guardrail breaches for diagnostics, but do not hard-reject.
                    if tp_hit_raw < min_tp_hit_eff:
                        reject_counts["tp_hit"] += 1
                    if timeout_raw > max_timeout_eff:
                        reject_counts["timeout"] += 1

                    if _skip_geom_eval:
                        _persisted_match = None
                        for _row in _cell_ranked:
                            try:
                                if (
                                    round(float(_row.get("k_tp", np.nan)), 4)
                                    == round(float(k_tp), 4)
                                    and round(
                                        float(_row.get("sl_as_tp_pct", np.nan)), 4
                                    )
                                    == round(float(sl_base_mult), 4)
                                    and int(_row.get("base_atr_window", _atr_window))
                                    == int(_atr_window)
                                ):
                                    _persisted_match = _row
                                    break
                            except Exception:
                                continue
                        relaxed_pool.append(
                            _summarize_geom_triplet(
                                k_tp=float(k_tp),
                                sl_base_mult=float(sl_base_mult),
                                atr_window=int(_atr_window),
                                rr_weight=1.0,
                                tp_hit_raw=float(tp_hit_raw),
                                sl_hit_raw=float(sl_hit_raw),
                                timeout_raw=float(timeout_raw),
                                n_events_raw=int(n_events_raw),
                                n_candidates=int(n_candidates),
                                n_rr_kept=int(n_rr_kept),
                                auc_bound=float(
                                    (_persisted_match or {}).get(
                                        "cell_auc_bound", float("nan")
                                    )
                                ),
                                tp_sep_top10=float(
                                    (_persisted_match or {}).get(
                                        "cell_tp_sep", float("nan")
                                    )
                                ),
                                bind_raw=float(tp_hit_raw + sl_hit_raw),
                                tp_over_sl=float(
                                    (_persisted_match or {}).get(
                                        "cell_tp_over_sl",
                                        float(k_tp) / max(float(sl_base_mult), 1e-9),
                                    )
                                ),
                                tp_guard_target=float("nan"),
                                tp_emp_base=float("nan"),
                                tp_floor_share=float(_tp_floor_share),
                                tp_ceil_share=float(_tp_ceil_share),
                                raw_key=raw_key,
                            )
                        )
                        _refresh_pinned_raw_payloads(relaxed_pool, _keep_top_n)
                        continue

                    # Composite geometry score (quality-first, TP-hit as guardrail).
                    auc_bound = float(_tb_summary.get("auc_bound", 0.5))
                    tp_sep_top10 = float(_tb_summary.get("tp_sep_top10", 0.0))
                    tp_over_sl = float(_tb_summary.get("tp_over_sl", 0.0))
                    bind_raw = float(
                        _tb_summary.get("bind_raw", tp_hit_raw + sl_hit_raw)
                    )
                    _tp_emp_base = float(_tb_summary.get("tp_emp_base", tp_hit_raw))

                    # Normalize components to stable [0, 1]-ish ranges.
                    auc_term = float(np.clip((auc_bound - 0.5) / 0.2, 0.0, 1.0))
                    sep_term = float(np.clip(tp_sep_top10 / 0.12, 0.0, 1.0))
                    bind_term = float(np.clip(bind_raw, 0.0, 1.0))
                    edge_term = float(np.clip((tp_over_sl - 1.0) / 1.0, 0.0, 1.0))

                    # Guardrails: soft penalties (TP-hit no longer primary hard gate).
                    tp_guard_target = max(
                        min_tp_hit_eff
                        * float(cfg.get("label_tp_hit_guardrail_floor_frac", 0.25)),
                        _tp_emp_base,
                    )
                    tp_guard = float(
                        np.clip(tp_hit_raw / max(tp_guard_target, 1e-9), 0.0, 1.0)
                    )
                    to_guard = float(
                        np.clip(max_timeout_eff / max(timeout_raw, 1e-9), 0.0, 1.0)
                    )
                    rr_guard = float(
                        np.clip(net_rr / max(min_net_rr_eff, 1e-9), 0.0, 2.0)
                    )

                    # Quality-first ranking weights.
                    base_score = (
                        0.45 * auc_term
                        + 0.30 * sep_term
                        + 0.15 * bind_term
                        + 0.10 * edge_term
                    )
                    geom_weight = (
                        base_score
                        * (0.85 + 0.15 * tp_guard)
                        * (0.80 + 0.20 * to_guard)
                        * (0.80 + 0.20 * min(rr_guard, 1.0))
                    )
                    relaxed_pool.append(
                        _summarize_geom_triplet(
                            k_tp=float(k_tp),
                            sl_base_mult=float(sl_base_mult),
                            atr_window=int(_atr_window),
                            rr_weight=float(max(geom_weight, 1e-6)),
                            tp_hit_raw=float(tp_hit_raw),
                            sl_hit_raw=float(sl_hit_raw),
                            timeout_raw=float(timeout_raw),
                            n_events_raw=int(n_events_raw),
                            n_candidates=int(n_candidates),
                            n_rr_kept=int(n_rr_kept),
                            auc_bound=float(auc_bound),
                            tp_sep_top10=float(tp_sep_top10),
                            bind_raw=float(bind_raw),
                            tp_over_sl=float(tp_over_sl),
                            tp_guard_target=float(tp_guard_target),
                            tp_emp_base=float(_tp_emp_base),
                            tp_floor_share=float(_tp_floor_share),
                            tp_ceil_share=float(_tp_ceil_share),
                            raw_key=raw_key,
                        )
                    )
                    _refresh_pinned_raw_payloads(relaxed_pool, _keep_top_n)

                # Keep top-N geometries per (bucket, horizon) by composite score.
                if relaxed_pool:
                    if _skip_geom_eval:
                        geom_runs_proxy = list(relaxed_pool)
                    else:
                        relaxed_pool.sort(
                            key=lambda g: (
                                g["rr_weight"],
                                g["auc_bound"],
                                g["tp_sep_top10"],
                                g["bind"],
                                g["tp_over_sl"],
                            ),
                            reverse=True,
                        )
                        geom_runs_proxy = relaxed_pool[:_keep_top_n]

                    # Materialize only the winners, preferably from the bounded raw payload cache.
                    geom_runs = [
                        _materialize_geom_triplet(_g, side, int(H))
                        for _g in geom_runs_proxy
                    ]
                    _raw_tb_payload_cache.clear()
                    _raw_tb_payload_pinned.clear()

                if not geom_runs:
                    # Rescue path mirrored for H2/H4 with stricter fallbacks on longer horizons.
                    if relaxed_pool and int(H) in (2, 4):
                        rescue_topk_h2 = int(
                            cfg.get("label_h2_rescue_topk", h2_rescue_topk)
                        )
                        rescue_topk_h4 = int(cfg.get("label_h4_rescue_topk", 2))
                        _rescue_topk = rescue_topk_h2 if int(H) == 2 else rescue_topk_h4
                        picked = relaxed_pool[: max(1, _rescue_topk)]
                        geom_runs = picked
                        tprint(
                            f"H={H} rescue accepted {len(geom_runs)} geometries "
                            f"(best tp_hit={picked[0]['tp_hit']:.4f}, timeout={picked[0]['to_rate']:.4f}, "
                            f"auc_bound={picked[0]['auc_bound']:.4f}, tp_sep={picked[0]['tp_sep_top10']*100:.2f}pp)"
                        )

                if not geom_runs:
                    tprint(
                        f"No valid geometry for H={H} side={side} kind={kind} strat={s_id}; using fallback."
                    )
                    tprint(
                        "Geometry rejection breakdown: "
                        f"H={H}, side={side}, kind={kind}, strat={s_id}, total={total_geoms}, "
                        f"rr_rejects={reject_counts['rr']}, n_events_rejects={reject_counts['n_events']}, "
                        f"tp_hit_guardrail_rejects={reject_counts['tp_hit']}, timeout_guardrail_rejects={reject_counts['timeout']}, "
                        f"min_tp_hit={min_tp_hit_eff:.4f}, tp_lo_search={tp_lo_eff_search:.4f}, tp_lo_prod={tp_lo_eff_prod:.4f}, "
                        f"max_timeout={max_timeout_eff:.4f}, min_rr={min_net_rr_eff:.4f}, "
                        f"min_events={min_events_eff}, "
                        f"tp_floor_share_mean={float(np.mean(clip_stats['tp_floor_shares'])) if clip_stats['tp_floor_shares'] else 0.0:.3f}, "
                        f"tp_ceil_share_mean={float(np.mean(clip_stats['tp_ceil_shares'])) if clip_stats['tp_floor_shares'] else 0.0:.3f}, "
                        f"sl_hit_mean={float(np.mean([g['sl_hit'] for g in relaxed_pool])) if relaxed_pool else 0.0:.3f}, "
                        f"no_hit_mean={float(np.mean([max(0.0, 1.0 - g['tp_hit'] - g['sl_hit'] - g['to_rate']) for g in relaxed_pool])) if relaxed_pool else 0.0:.3f}"
                    )
                    _tp_lo_fb = _co_calibrate_tp_floor(tp_lo_eff_prod, 0.5, tp_hi)
                    tp_df, sl_df = compute_barrier_factory(
                        atr_pct=atr_pct_local,
                        window_size=_atr_window,
                        k_tp=1.0,
                        sl_base_mult=0.5,
                        horizon=H,
                        H_base=H_base,
                        disp_floor=disp_floor,
                        z_max=z_max,
                        k_reg=k_reg,
                        m_lo=m_lo,
                        m_hi=m_hi,
                        sl_mult_lo=sl_mult_lo,
                        sl_mult_hi=sl_mult_hi,
                        sl_lo=sl_lo,
                        sl_hi=sl_hi,
                        z_gate=z_gate,
                        tp_lo=_tp_lo_fb,
                        tp_hi=tp_hi,
                        _base=_barrier_base,
                    )
                    _tb_out = compute_triple_barrier_labels(
                        panel,
                        tp_df,
                        sl_df,
                        H,
                        side=side,
                        return_outcomes=True,
                        return_path_stats=False,
                    )
                    lbl, ret, qual = _tb_out[:3]
                    _tp_vals_fb = tp_df.values
                    _n_events_fb = lbl.size
                    _tp_hit_fb = float((lbl.values == OUT_TP).sum()) / max(
                        1, _n_events_fb
                    )
                    _sl_hit_fb = float((lbl.values == OUT_SL).sum()) / max(
                        1, _n_events_fb
                    )
                    _to_rate_fb = float((lbl.values == OUT_TO).sum()) / max(
                        1, _n_events_fb
                    )
                    geom_runs = [
                        {
                            "lbl": lbl,
                            "ret": ret,
                            "qual": qual,
                            "k_tp": 1.0,
                            "sl_base_mult": 0.5,
                            "rr_weight": 1.0,
                            "tp_hit": _tp_hit_fb,
                            "sl_hit": _sl_hit_fb,
                            "to_rate": _to_rate_fb,
                            "n_events": int(_n_events_fb),
                            "auc_bound": 0.5,
                            "tp_sep_top10": 0.0,
                            "bind": _tp_hit_fb + _sl_hit_fb,
                            "tp_over_sl": 1.0,
                            "tp_floor_share": float(
                                np.mean(_tp_vals_fb <= (_tp_lo_fb + 1e-9))
                            ),
                            "tp_ceil_share": float(
                                np.mean(_tp_vals_fb >= (tp_hi - 1e-9))
                            ),
                        }
                    ]
                else:
                    geom_desc = []
                    no_hit_rates = []

                    def _fmt_count(v):
                        try:
                            iv = int(v)
                            return str(iv) if iv >= 0 else "NA"
                        except Exception:
                            return "NA"

                    for _g in geom_runs:
                        _no_hit = max(
                            0.0, 1.0 - _g["tp_hit"] - _g["sl_hit"] - _g["to_rate"]
                        )
                        no_hit_rates.append(_no_hit)
                        geom_desc.append(
                            f"(k_tp={_g['k_tp']:.2f}, sl={_g['sl_base_mult']:.2f}, tp_hit={_g['tp_hit']:.3%}, "
                            f"sl_hit={_g['sl_hit']:.3%}, timeout={_g['to_rate']:.3%}, no_hit={_no_hit:.3%}, "
                            f"n={int(_g['n_events'])}, auc_b={_g['auc_bound']:.3f}, sep={_g['tp_sep_top10']*100:.2f}pp, "
                            f"bind_raw={_g['bind']:.3f}, edge={_g['tp_over_sl']:.2f}, w={_g['rr_weight']:.3f}, "
                            f"n_candidates={_fmt_count(_g.get('n_candidates'))}, n_rr_kept={_fmt_count(_g.get('n_rr_kept'))})"
                        )
                    tprint(
                        f"Accepted geometries H={H} side={side} kind={kind} strat={s_id}: {len(geom_runs)} | "
                        + "; ".join(geom_desc)
                    )
                    tprint(
                        "Geometry diagnostics: "
                        f"H={H}, side={side}, kind={k_label}, total={total_geoms}, "
                        f"rr_rejects={reject_counts['rr']}, n_events_rejects={reject_counts['n_events']}, "
                        f"tp_hit_guardrail_rejects={reject_counts['tp_hit']}, timeout_guardrail_rejects={reject_counts['timeout']}, "
                        f"min_tp_hit={min_tp_hit_eff:.4f}, max_timeout={max_timeout_eff:.4f}, min_rr={min_net_rr_eff:.4f}, "
                        f"tp_floor_share_mean={float(np.mean(clip_stats['tp_floor_shares'])) if clip_stats['tp_floor_shares'] else 0.0:.3f}, "
                        f"tp_ceil_share_mean={float(np.mean(clip_stats['tp_ceil_shares'])) if clip_stats['tp_ceil_shares'] else 0.0:.3f}"
                    )
                    _clip_warn_thr = float(
                        cfg.get("label_tp_floor_clip_warn_threshold", 0.70)
                    )
                    _tp_floor_mean = (
                        float(np.mean(clip_stats["tp_floor_shares"]))
                        if clip_stats["tp_floor_shares"]
                        else 0.0
                    )
                    if _tp_floor_mean > _clip_warn_thr:
                        tprint(
                            f"WARNING: TP floor clipping is high for H={H} side={side} kind={k_label} "
                            f"({100.0*_tp_floor_mean:.1f}% > {100.0*_clip_warn_thr:.1f}%). "
                            f"Consider raising production TP floor or widening geometry."
                        )

                    # Build production-admissibility inputs from the kept top-K geometries for this cell.
                    _prod_geom = _aggregate_geom_runs(geom_runs)
                    _best_g = geom_runs[0]
                    _lbl_mat = _prod_geom["agg_lbl"]
                    _ret_mat = _prod_geom["agg_ret"]
                    _tp_mat = _prod_geom["tp_vals_df"].to_numpy(dtype=np.float32, copy=False)
                    _sl_mat = _prod_geom["sl_vals_df"].to_numpy(dtype=np.float32, copy=False)
                    _valid_prod = (
                        np.isfinite(_ret_mat)
                        & np.isfinite(_tp_mat)
                        & np.isfinite(_sl_mat)
                    )
                    _valid_prod &= np.isin(
                        _lbl_mat, np.array([OUT_SL, OUT_TO, OUT_TP], dtype=np.int8)
                    )
                    _bucket = str(side)
                    _y_prod = _lbl_mat[_valid_prod].astype(np.int8, copy=False)
                    _prod_df = pd.DataFrame(
                        {
                            "bucket": _bucket,
                            "horizon": int(H),
                            "label": _y_prod,
                            "tp": _tp_mat[_valid_prod].astype(np.float32, copy=False),
                            "sl": _sl_mat[_valid_prod].astype(np.float32, copy=False),
                            "payoff": _ret_mat[_valid_prod].astype(
                                np.float32, copy=False
                            ),
                        }
                    )
                    _prod_sample_n = int(cfg.get("prod_adm_sample_n", 500_000))
                    if len(_prod_df) > _prod_sample_n:
                        _prod_df = _prod_df.sample(n=_prod_sample_n, random_state=42)
                    _prod_events_rows.append(_prod_df)
                    del (
                        _prod_df,
                        _lbl_mat,
                        _ret_mat,
                        _tp_mat,
                        _sl_mat,
                        _valid_prod,
                        _y_prod,
                    )
                    _prod_cell_metrics[f"{side}_H{int(H)}"] = {
                        "timeout": float(
                            np.mean(
                                [
                                    float(
                                        g.get(
                                            "timeout_raw",
                                            g.get("to_rate", float("nan")),
                                        )
                                    )
                                    for g in geom_runs
                                ]
                            )
                        ),
                        "auc_label": float(
                            np.mean(
                                [
                                    float(
                                        g.get(
                                            "auc_label",
                                            g.get("auc_bound", float("nan")),
                                        )
                                    )
                                    for g in geom_runs
                                ]
                            )
                        ),
                        "auc_bound": float(
                            np.mean(
                                [float(g.get("auc_bound", float("nan"))) for g in geom_runs]
                            )
                        ),
                        "tp_sep_top10": float(
                            np.mean(
                                [float(g.get("tp_sep_top10", float("nan"))) for g in geom_runs]
                            )
                        ),
                        "ap_lift": float(
                            np.mean(
                                [float(g.get("ap_lift", float("nan"))) for g in geom_runs]
                            )
                        ),
                        "tp_over_sl": float(
                            np.mean(
                                [float(g.get("tp_over_sl", float("nan"))) for g in geom_runs]
                            )
                        ),
                    }

                if _diag_labels:
                    _pre_src = geom_runs if geom_runs else relaxed_pool
                    if _pre_src:
                        _tp_pre = float(
                            np.mean(
                                [float(g.get("tp_hit", float("nan"))) for g in _pre_src]
                            )
                        )
                        _sl_pre = float(
                            np.mean(
                                [float(g.get("sl_hit", float("nan"))) for g in _pre_src]
                            )
                        )
                        _to_pre = float(
                            np.mean(
                                [
                                    float(g.get("to_rate", float("nan")))
                                    for g in _pre_src
                                ]
                            )
                        )
                        tprint(
                            f"[LABEL_DIAG][PRE_GEOM] side={side} kind={kind} strat={s_id} H={H} "
                            f"n_geom={len(_pre_src)} tp={_tp_pre:.3%} sl={_sl_pre:.3%} to={_to_pre:.3%}"
                        )

                _geom_variants = _cluster_geometry_candidates_hybrid(
                    [
                        (
                            g["k_tp"],
                            g["sl_base_mult"],
                            int(g.get("atr_window", _atr_window)),
                        )
                        for g in geom_runs
                    ],
                    ranked_rows=_cell_ranked,
                    archetypes=cfg.get("base_geometry_archetypes", ["tight", "wide"]),
                    topk=cfg.get("base_geometry_grr_topk", 12),
                    learnability_weight=float(
                        cfg.get("base_geometry_learnability_weight", 0.75)
                    ),
                    geometry_weight=float(
                        cfg.get("base_geometry_geometry_weight", 0.25)
                    ),
                )
                _balanced_triplets = _geom_variants.get("balanced", [])
                if _balanced_triplets:
                    _balanced_set = {
                        (round(float(t[0]), 6), round(float(t[1]), 6), int(t[2]))
                        for t in _balanced_triplets
                    }
                    _canonical_runs = [
                        g
                        for g in geom_runs
                        if (
                            round(float(g["k_tp"]), 6),
                            round(float(g["sl_base_mult"]), 6),
                            int(g.get("atr_window", _atr_window)),
                        )
                        in _balanced_set
                    ]
                else:
                    _canonical_runs = list(geom_runs)
                _materialize_geom_aggregate(_canonical_runs, (int(H), s_id))
                for _variant_name, _variant_triplets in _geom_variants.items():
                    if _variant_name == "balanced":
                        # canonical aggregate is now the balanced/default path
                        continue
                    _variant_set = {
                        (round(float(t[0]), 6), round(float(t[1]), 6), int(t[2]))
                        for t in _variant_triplets
                    }
                    _variant_runs = [
                        g
                        for g in geom_runs
                        if (
                            round(float(g["k_tp"]), 6),
                            round(float(g["sl_base_mult"]), 6),
                            int(g.get("atr_window", _atr_window)),
                        )
                        in _variant_set
                    ]
                    if not _variant_runs:
                        continue
                    _materialize_geom_aggregate(
                        _variant_runs, (int(H), s_id, _variant_name)
                    )

                for _g in geom_runs:
                    _g.pop("lbl", None)
                    _g.pop("ret", None)
                    _g.pop("qual", None)
                geom_runs.clear()
                relaxed_pool.clear()
                _raw_tb_cache.clear()
                # Do NOT clear _barrier_base_cache — it is pre-computed globally and reused across cells.

    # Production-aligned admissibility diagnostic (label-step side).
    # The pre-event TB-cache phase still works on dense matrix outputs before the
    # strategy-specific candidate/event materialization path. That makes TP/SL/TO
    # rates look far worse than the actual production candidate set, so running the
    # hard admissibility gate here is misleading. Keep it opt-in only.
    if _prod_events_rows and bool(
        cfg.get("label_prod_admissibility_pre_event_enable", False)
    ):
        _events_prod = pd.concat(_prod_events_rows, ignore_index=True)
        _score_prod = _events_prod["payoff"].to_numpy(dtype=np.float32, copy=False)
        _tp_lo_prod_eval = float(
            cfg.get("barrier_tp_lo_prod", cfg.get("barrier_tp_lo", 0.02))
        )
        _sl_lo_prod_eval = float(
            cfg.get("tbm_sl_abs_lo_pct", cfg.get("barrier_tp_lo", 0.02))
        )
        _gates = ProdGates(
            n_min=int(cfg.get("prod_adm_n_min", 50)),
            bind_cell_min=float(cfg.get("prod_adm_bind_cell_min", 0.38)),
            bind_min=float(cfg.get("prod_adm_bind_min", 0.50)),
            timeout_max=float(cfg.get("prod_adm_timeout_max", 0.60)),
            timeout_range_max=float(cfg.get("prod_adm_timeout_range_max", 0.50)),
            sl_to_tp_max=float(cfg.get("prod_adm_sl_to_tp_max", 3.0)),
            auc_min=float(cfg.get("prod_adm_auc_min", 0.56)),
            auc_bound_min=float(cfg.get("prod_adm_auc_bound_min", 0.52)),
            tp_sep_min=float(cfg.get("prod_adm_tp_sep_min", 0.05)),
            ap_lift_min=float(cfg.get("prod_adm_ap_lift_min", 1.25)),
            tp_over_sl_min=float(cfg.get("prod_adm_tp_over_sl_min", 1.05)),
            tp_floor_bind_max_cell=float(
                cfg.get("prod_adm_tp_floor_bind_max_cell", 0.70)
            ),
            tp_floor_bind_max_agg=float(
                cfg.get("prod_adm_tp_floor_bind_max_agg", 0.65)
            ),
        )
        _prod_report = production_admissibility_report(
            events_prod=_events_prod,
            score_prod=_score_prod,
            bucket_horizon_metrics_prod=_prod_cell_metrics,
            tp_lo_prod=_tp_lo_prod_eval,
            sl_lo_prod=_sl_lo_prod_eval,
            gates=_gates,
        )
        if not bool(_prod_report.get("admissible_tier0", False)):
            tprint(
                "[LABEL_PROD_ADMISSIBILITY] FAIL "
                + " | ".join(_prod_report.get("failures", []))
            )
        else:
            tprint("[LABEL_PROD_ADMISSIBILITY] PASS")
        # Surface potential threshold conflicts with existing config values.
        if float(cfg.get("label_max_timeout_rate", 0.90)) > _gates.timeout_max:
            tprint(
                f"[LABEL_PROD_ADMISSIBILITY] NOTE existing label_max_timeout_rate={float(cfg.get('label_max_timeout_rate')):.3f} "
                f"is looser than prod_adm_timeout_max={_gates.timeout_max:.3f}."
            )
    elif _prod_events_rows:
        tprint(
            "[LABEL_PROD_ADMISSIBILITY] SKIP pre-event gate disabled; "
            "candidate-filtered datasets are the authoritative contract."
        )

    return tb_cache, geom_cache


def generate_label_datasets(
    panel, feats, mkt_gates, cfg, syms, ts, p_exh_hist, horizons=None
):
    tprint(f"Entering function: generate_label_datasets in training.py")
    run_id = pd.Timestamp(ts).strftime("%Y%m%d_%H%M%S")
    datasets = {}
    incremental_persist = bool(cfg.get("label_persist_incremental", False))
    persisted_manifest: dict[str, dict[str, object]] = {}
    base_variant_buffer: dict[tuple[str, int], dict[str, pd.DataFrame]] = {}
    requested_horizons = list(horizons) if horizons else list(CANON_HORIZONS)
    tprint(
        f"Label dataset builder: symbols={len(syms)} horizons={requested_horizons} "
        f"incremental_persist={incremental_persist}"
    )

    def _persist_label_artifact(name: str, df: pd.DataFrame) -> None:
        save_artifact_df(df, cfg["data_root"], run_id, "labels", name)
        persisted_manifest[name] = {
            "file": f"{name}.parquet",
            "rows": int(len(df)),
            "columns": list(df.columns),
        }

    def _maybe_flush_base_variant(strategy_id: str, horizon: int) -> None:
        if not incremental_persist:
            return
        key = (strategy_id, int(horizon))
        buf = base_variant_buffer.get(key)
        if not buf:
            return
        tight_df = buf.get("tight")
        wide_df = buf.get("wide")
        if tight_df is None or wide_df is None:
            return
        base_key = f"train_{strategy_id}_{int(horizon)}"
        base_df = pd.concat([tight_df, wide_df], axis=0, ignore_index=True)
        _persist_label_artifact(base_key, base_df)
        del base_df
        del base_variant_buffer[key]
        if bool(cfg.get("label_gc_after_each_dataset", True)):
            gc.collect()

    # Always resolve + enforce persisted offline-optimal candidate ranges before any event generation.
    cfg = _resolve_training_cfg_with_offline_optimisers(cfg)

    # Pre-compute shared expensive operations once
    tprint("Pre-computing candidate mask (shared across all steps)...")
    cached_cand_mask, cfg, mask_by_strategy = _build_optimal_candidate_mask(
        panel, feats, cfg
    )

    # Pre-calculate Microstructure Noise Filter (Costly rolling operations, shared across all models)
    if bool(cfg.get("use_noise_filter", True)) and cached_cand_mask is not None:
        tprint("Pre-computing Microstructure Noise Filter...")
        from extreme_price_movements.fast_funcs import numba_rolling_mean_parallel
        vol = panel["volume"]
        # Optimization: Replaced slow pandas rolling with Numba parallel equivalent.
        # This solves massive memory spikes and slow dataframe allocations across large panels.
        # Impact: Reduces memory overhead and speeds up noise filter calculation by 5-10x.
        vol_ma = numba_rolling_mean_parallel(vol, 24 * 30)
        # Apply min_periods equivalent logic
        vol_ma.iloc[:23, :] = np.nan
        liq_mask = vol >= (0.25 * vol_ma)

        h = panel["high"]
        l = panel["low"]
        c = panel["close"]
        wick_size = (h - l) / (c + 1e-9)
        wick_thr = float(cfg.get("noise_filter_wick_thr", 0.05))
        wick_mask = wick_size < wick_thr

        noise_mask = liq_mask & wick_mask

        # Embed noise mask into global candidate mask
        n_before = cached_cand_mask.sum().sum()
        cached_cand_mask = cached_cand_mask & noise_mask.reindex(
            cached_cand_mask.index
        ).fillna(False)
        for _sid, _mask_df in list(mask_by_strategy.items()):
            try:
                _aligned_noise = noise_mask.reindex(_mask_df.index).fillna(False)
                mask_by_strategy[_sid] = (_mask_df & _aligned_noise).astype(bool)
            except Exception:
                continue
        n_after = cached_cand_mask.sum().sum()
        tprint(f"Noise filter applied globally. Candidates: {n_before} -> {n_after}")

    # Apply OOS holdout: exclude last N days from training labels
    oos_days = cfg.get("oos_holdout_days", 0)
    if oos_days > 0 and cached_cand_mask is not None:
        cutoff = ts - pd.Timedelta(days=oos_days)
        n_before = cached_cand_mask.sum().sum()
        cached_cand_mask = cached_cand_mask.loc[cached_cand_mask.index <= cutoff]
        for _sid, _mask_df in list(mask_by_strategy.items()):
            try:
                mask_by_strategy[_sid] = _mask_df.loc[_mask_df.index <= cutoff]
            except Exception:
                continue
        n_after = cached_cand_mask.sum().sum()
        tprint(
            f"OOS holdout: excluded last {oos_days} days (cutoff={cutoff}). Candidates: {n_before} -> {n_after}"
        )

    # 1. Spike Anatomy (2 GMM models: Best & Worst)
    # NOTE: Spike Anatomy models removed - GMM training disabled
    # for mode in ["best", "worst"]:
    #     spike_df = train_spike_anatomy_model(
    #         panel,
    #         feats,
    #         mkt_gates,
    #         cfg,
    #         syms,
    #         ts,
    #         _cached_cand_mask=mask_by_strategy.get(k, cached_cand_mask),
    #         mode=mode,
    #     )
    #     if spike_df is not None:
    #         datasets[f"spike_anatomy_{mode}"] = spike_df

    # Pre-compute/load triple-barrier labels with geometry-grid aggregation per (H, strategy)
    tb_cache: dict = {}
    geom_cache: dict = {}
    missing_cells: list[tuple[int, str]] = []

    strategies = get_strategies(cfg)
    strategies_by_id = {s["strategy_id"]: s for s in strategies}
    strategy_horizons = {
        s["strategy_id"]: strategy_runtime_horizons(
            s, cfg, requested_horizons=requested_horizons
        )
        for s in strategies
    }
    active_horizons = sorted({int(h) for hs in strategy_horizons.values() for h in hs})
    tprint(
        f"Label dataset builder: strategies={len(strategies)} active_horizons={active_horizons}"
    )
    # Hash based on horizons + strategy_ids
    cfg_hash = _stable_cfg_subset_hash(
        cfg, active_horizons, [s["strategy_id"] for s in strategies]
    )

    for strat in strategies:
        s_id = strat["strategy_id"]
        for H in strategy_horizons.get(s_id, []):
            p_tb, p_geom, _ = _tb_cache_paths(cfg, run_id, H, s_id, cfg_hash)
            if os.path.exists(p_tb) and os.path.exists(p_geom):
                try:
                    with open(p_tb, "rb") as fh:
                        tb_cache[(int(H), s_id)] = _downcast_tb_triplet(pickle.load(fh))
                    with open(p_geom, "rb") as fh:
                        geom_cache[(int(H), s_id)] = _downcast_geom_payload(
                            pickle.load(fh)
                        )
                    continue
                except Exception as _e_cache:
                    tprint(
                        f"TB cache read failed for H={H} strategy_id={s_id}: {_e_cache}"
                    )
            missing_cells.append((int(H), s_id))

    def _build_and_persist_tb_cell(H: int, s_id: str) -> bool:
        _strat = strategies_by_id.get(s_id)
        if _strat is None:
            return False
        tprint(f"TB cache build: strategy_id={s_id} H={H} starting...")
        _tb_i, _geom_i = build_grid_aggregated_tb_cache(
            panel=panel,
            feats=feats,
            cfg=cfg,
            horizons=[int(H)],
            strategies=[_strat],
        )
        key = (int(H), s_id)
        persisted_base = False
        for _k, _v in list(_tb_i.items()):
            if not isinstance(_k, tuple) or len(_k) < 2:
                continue
            if int(_k[0]) != int(H) or str(_k[1]) != str(s_id):
                continue
            tb_cache[_k] = _downcast_tb_triplet(_v)
            if len(_k) == 2 and _k == key:
                persisted_base = True
                p_tb, p_geom, _ = _tb_cache_paths(cfg, run_id, H, s_id, cfg_hash)
                os.makedirs(os.path.dirname(p_tb), exist_ok=True)
                with open(p_tb, "wb") as fh:
                    pickle.dump(tb_cache[_k], fh, protocol=pickle.HIGHEST_PROTOCOL)
        for _k, _v in list(_geom_i.items()):
            if not isinstance(_k, tuple) or len(_k) < 2:
                continue
            if int(_k[0]) != int(H) or str(_k[1]) != str(s_id):
                continue
            geom_cache[_k] = _downcast_geom_payload(_v)
            if len(_k) == 2 and _k == key:
                p_tb, p_geom, _ = _tb_cache_paths(cfg, run_id, H, s_id, cfg_hash)
                os.makedirs(os.path.dirname(p_geom), exist_ok=True)
                with open(p_geom, "wb") as fh:
                    pickle.dump(geom_cache[_k], fh, protocol=pickle.HIGHEST_PROTOCOL)
        if not persisted_base:
            tprint(f"TB cache build: strategy_id={s_id} H={H} produced no base cell payload.")
            return False
        tprint(f"TB cache build: strategy_id={s_id} H={H} persisted.")
        del _tb_i, _geom_i
        if bool(cfg.get("label_gc_after_each_dataset", True)):
            gc.collect()
        return True

    if missing_cells:
        miss_h = sorted({c[0] for c in missing_cells})
        miss_s_ids = sorted({c[1] for c in missing_cells})
        tprint(
            f"TB cache miss: {len(missing_cells)} cells; recomputing horizons={miss_h} strategy_ids={miss_s_ids}"
        )
        for H, s_id in sorted(set((int(h), sid) for h, sid in missing_cells)):
            _build_and_persist_tb_cell(int(H), s_id)
        tprint(
            f"TB cache refresh complete: cells={len(missing_cells)} now_cached={len(tb_cache)}"
        )
    else:
        tprint(f"TB cache fully satisfied from disk: cells={len(tb_cache)}")

    if bool(cfg.get("base_geometry_train_variants", True)):
        _missing_variant_keys = []
        for strat in strategies:
            s_id = strat["strategy_id"]
            for H in strategy_horizons.get(s_id, []):
                for _variant in cfg.get("base_geometry_archetypes", ["tight", "wide"]):
                    _variant = str(_variant)
                    if (int(H), s_id, _variant) not in tb_cache:
                        _missing_variant_keys.append((int(H), s_id, _variant))
        if _missing_variant_keys:
            tprint(
                f"Materializing grouped base-geometry variants in-memory for {len(_missing_variant_keys)} cells..."
            )
            for _H, _s_id, _variant in sorted(set(_missing_variant_keys)):
                _build_and_persist_tb_cell(int(_H), _s_id)
        else:
            tprint("Base-geometry variant cache already satisfied for all requested cells.")

    def _prepare_events_once_for_strategy(_strategy: dict):
        _prep_lookup_cache: dict = {}
        ts_start = ts - pd.Timedelta(hours=int(cfg["train_lookback_hours"]))
        _hs = strategy_horizons.get(_strategy["strategy_id"], [])
        if not _hs:
            return None
        min_h = int(min(_hs))
        ts_end_adj = ts - pd.Timedelta(hours=min_h + 8)
        _strategy_mask = mask_by_strategy.get(
            _strategy["strategy_id"], cached_cand_mask
        )
        if _strategy_mask is None:
            return None
        window_cand = _strategy_mask.loc[
            (_strategy_mask.index >= ts_start) & (_strategy_mask.index <= ts_end_adj)
        ]
        if window_cand.empty:
            return None

        valid_syms = [
            s for s in syms if s in window_cand.columns and s in panel["close"].columns
        ]
        if not valid_syms:
            return None

        s_id = _strategy["strategy_id"]
        _tb = tb_cache.get((min_h, s_id))
        if _tb is None:
            return None
        tb_labels = _tb[0]

        try:
            cand_ns = pd.to_datetime(window_cand.index, utc=True).view("i8")
            valid_entry_ns = pd.to_datetime(tb_labels.index, utc=True).view("i8") - int(
                pd.Timedelta(hours=1).value
            )
            align_mask = np.isin(cand_ns, valid_entry_ns)
            window_cand_aligned = (
                window_cand.iloc[align_mask] if align_mask.any() else window_cand
            )
        except Exception:
            valid_entry_times = tb_labels.index - pd.Timedelta(hours=1)
            window_cand_aligned = window_cand[window_cand.index.isin(valid_entry_times)]
            if window_cand_aligned.empty:
                window_cand_aligned = window_cand

        sub_mask = window_cand_aligned[valid_syms]
        rows_idx, cols_idx = np.where(sub_mask.values)
        if len(rows_idx) == 0:
            return None

        event_ts = sub_mask.index[rows_idx]
        event_sym = np.array(valid_syms)[cols_idx]
        entry_ts = event_ts + pd.Timedelta(hours=1)

        if "trend_pct" in feats:
            cand_filter, move_bucket, _ = _strategy_bucket_context(
                _strategy["trade_side"], _strategy["strategy_id"], cfg
            )
            del cand_filter
            if move_bucket in {"up", "down"}:
                trend_vals = _fast_lookup_cached(
                    feats["trend_pct"],
                    event_ts,
                    event_sym,
                    lookup_cache=_prep_lookup_cache,
                )
                keep = _trend_direction_keep_mask(trend_vals, move_bucket)
                event_ts = event_ts[keep]
                event_sym = event_sym[keep]
                entry_ts = entry_ts[keep]

        return (event_ts, event_sym, entry_ts)

    precomputed_events = {
        strat["strategy_id"]: _prepare_events_once_for_strategy(strat)
        for strat in strategies
    }

    tasks = []
    _required_geometry_variants = ("tight", "wide")
    for strat in strategies:
        side = strat["trade_side"]
        k = strat["strategy_id"]
        strategy_id = strat["strategy_id"]
        cand_filter, move_bucket, strategy_label = _strategy_bucket_context(
            side, strategy_id, cfg
        )

        if "mr" in strategy_id.lower():
            feat_key = "base_short_feature_keys"
        elif "tf" in strategy_id.lower():
            feat_key = "base_long_feature_keys"
        else:
            feat_key = "base_shared_feature_keys"

        fixed_tp = 0.05
        fixed_sl = 0.025
        extra_feature_keys = _meta_feature_keys_for_kind(cfg, strat)
        for H in strategy_horizons.get(strategy_id, []):
            H_int = int(H)

            if not bool(cfg.get("base_geometry_train_variants", True)):
                tprint(
                    f"Skipping classifier label cell {strategy_id}_H{H_int}: "
                    "base_geometry_train_variants=False but tight/wide are mandatory."
                )
                continue

            _missing_required = [
                _v
                for _v in _required_geometry_variants
                if (H_int, strategy_id, _v) not in tb_cache
            ]
            if _missing_required:
                tprint(
                    f"Skipping classifier label cell {strategy_id}_H{H_int}: "
                    f"missing required geometry variants={_missing_required}."
                )
                continue

            for _variant in _required_geometry_variants:
                tasks.append(
                    (
                        H_int,
                        side,
                        k,
                        strategy_id,
                        _variant,
                        move_bucket,
                        strategy_label,
                        cand_filter,
                        feat_key,
                        extra_feature_keys,
                        fixed_tp,
                        fixed_sl,
                    )
                )

    def _run_one_cell(task):
        (
            H,
            side,
            k,
            strategy_id,
            variant,
            move_bucket,
            strategy_label,
            cand_filter,
            feat_key,
            extra_feature_keys,
            fixed_tp,
            fixed_sl,
        ) = task
        _variant_log = variant if variant is not None else "h1_regressor"
        tprint(
            f"Generating labels: strategy_id={strategy_id}, H={H}, "
            f"variant={_variant_log}"
        )
        _tb_key = (H, strategy_id) if variant is None else (H, strategy_id, variant)
        _geom = geom_cache.get(_tb_key)

        _pre = precomputed_events.get(strategy_id)
        if _pre is not None:
            _ts_ev, _sym_ev, _entry_ev = _pre
            _mask_h = _ts_ev <= (ts - pd.Timedelta(hours=H + 8))
            _pre_h = (_ts_ev[_mask_h], _sym_ev[_mask_h], _entry_ev[_mask_h])
            if len(_pre_h[0]) == 0:
                _pre_h = None
        else:
            _pre_h = None

        (
            X,
            y,
            y_ret,
            cols,
            w,
            meta_idx,
            lbl_vals,
        ) = build_hourly_training_set_and_weights(
            panel,
            feats,
            mkt_gates,
            cfg,
            syms,
            ts,
            p_exh_hist,
            H,
            k,
            trend_filter=move_bucket,
            feature_key=feat_key,
            extra_feature_keys=extra_feature_keys,
            label_method="triple_barrier",
            fixed_tp=fixed_tp,
            fixed_sl=fixed_sl,
            side=side,
            _cached_cand_mask=mask_by_strategy.get(k, cached_cand_mask),
            _cached_tb=tb_cache.get((H, strategy_id))
            if variant is None
            else tb_cache.get((H, strategy_id, variant)),
            _precomputed_events=_pre_h,
            _geom_frames=_geom,
        )

        _, _, p_evt = _tb_cache_paths(cfg, run_id, H, strategy_id, cfg_hash)
        if _pre_h is not None and variant is None:
            _save_event_index_artifact(p_evt, _pre_h[2], _pre_h[1], symbol_vocab)

        return (H, side, k, variant, X, y, y_ret, w, meta_idx, lbl_vals)

    # Enforce maximum of 2 workers as requested for speed/memory tradeoff
    n_workers = min(2, _choose_parallel_cells(len(tasks), cfg))

    # We want to enable parallel execution if n_workers > 1,
    # regardless of legacy label_parallel_enable setting, as long as the user hasn't explicitly disabled it
    parallel_enabled = bool(cfg.get("label_parallel_enable", True)) or n_workers > 1

    if parallel_enabled and n_workers > 1:
        tprint(
            f"Parallel label cell execution enabled: workers={n_workers}, cells={len(tasks)}"
        )
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            futs = [ex.submit(_run_one_cell, t) for t in tasks]
            for fut in as_completed(futs):
                H, side, k, variant, X, y, y_ret, w, meta_idx, lbl_vals = fut.result()
                if X is not None:
                    df_out = X.copy()
                    df_out["__y_bin__"] = y
                    df_out["__y_ret__"] = y_ret
                    df_out["__y_outcome__"] = lbl_vals
                    df_out["__w__"] = w
                    if meta_idx is not None:
                        df_out["__ts__"] = meta_idx["ts"]
                        df_out["__symbol__"] = meta_idx["symbol"]
                    _ds_key = (
                        f"train_{k}_{H}"
                        if variant is None
                        else f"train_{k}_{H}_{variant}"
                    )
                    df_out = _downcast_label_dataset_df(df_out, copy=False)
                    if incremental_persist:
                        _persist_label_artifact(_ds_key, df_out)
                        if variant in {"tight", "wide"}:
                            base_key = (str(k), int(H))
                            base_variant_buffer.setdefault(base_key, {})[
                                variant
                            ] = df_out
                            _maybe_flush_base_variant(str(k), int(H))
                        del df_out
                    else:
                        datasets[_ds_key] = df_out
                    del X, y, y_ret, w, meta_idx, lbl_vals
                    if bool(cfg.get("label_gc_after_each_dataset", True)):
                        gc.collect()
    else:
        for t in tasks:
            H, side, k, variant, X, y, y_ret, w, meta_idx, lbl_vals = _run_one_cell(t)
            if X is not None:
                df_out = X.copy()
                df_out["__y_bin__"] = y
                df_out["__y_ret__"] = y_ret
                df_out["__y_outcome__"] = lbl_vals
                df_out["__w__"] = w
                if meta_idx is not None:
                    df_out["__ts__"] = meta_idx["ts"]
                    df_out["__symbol__"] = meta_idx["symbol"]
                _ds_key = (
                    f"train_{k}_{H}" if variant is None else f"train_{k}_{H}_{variant}"
                )
                df_out = _downcast_label_dataset_df(df_out, copy=False)
                if incremental_persist:
                    _persist_label_artifact(_ds_key, df_out)
                    if variant in {"tight", "wide"}:
                        base_key = (str(k), int(H))
                        base_variant_buffer.setdefault(base_key, {})[variant] = df_out
                        _maybe_flush_base_variant(str(k), int(H))
                    del df_out
                else:
                    datasets[_ds_key] = df_out
                del X, y, y_ret, w, meta_idx, lbl_vals
                if bool(cfg.get("label_gc_after_each_dataset", True)):
                    gc.collect()

    # Synthesize the canonical base training dataset by concatenating required
    # tight/wide variants. We retain the canonical key name for downstream
    # compatibility.
    if not incremental_persist:
        strategies = get_strategies(cfg)
        for strat in strategies:
            k = strat["strategy_id"]
            for H in strategy_horizons.get(k, []):
                H_int = int(H)
                _tight_key = f"train_{k}_{H_int}_tight"
                _wide_key = f"train_{k}_{H_int}_wide"
                _base_key = f"train_{k}_{H_int}"
                if _tight_key not in datasets or _wide_key not in datasets:
                    continue
                _parts = [datasets[_tight_key], datasets[_wide_key]]
                _parts = [p for p in _parts if p is not None and not p.empty]
                if not _parts:
                    continue
                datasets[_base_key] = pd.concat(_parts, axis=0, ignore_index=True)

    tprint("Spike anatomy and specialist model datasets are disabled.")

    for _k, _v in list(datasets.items()):
        if isinstance(_v, pd.DataFrame) and _v.dtypes.eq(np.float64).any():
            datasets[_k] = _downcast_label_dataset_df(_v, copy=False)

    if bool(cfg.get("label_fail_on_duplicate_datasets", True)):
        _fingerprints: dict[tuple[int, str, int], list[str]] = {}
        for _name, _df in datasets.items():
            if not isinstance(_df, pd.DataFrame):
                continue
            if not {"__ts__", "__symbol__", "__y_ret__", "__y_outcome__"}.issubset(
                _df.columns
            ):
                continue
            _parts = _name.split("_")
            if len(_parts) < 3 or _parts[0] != "train":
                continue
            _variant = ""
            _h_token = _parts[-1]
            if _h_token in {"tight", "wide", "balanced"} and len(_parts) >= 4:
                _variant = _h_token
                _h_token = _parts[-2]
            try:
                _h = int(_h_token)
            except Exception:
                continue
            _core = _df[["__ts__", "__symbol__", "__y_ret__", "__y_outcome__"]]
            _fp = int(
                pd.util.hash_pandas_object(_core, index=False)
                .to_numpy(dtype=np.uint64, copy=False)
                .sum(dtype=np.uint64)
            )
            _fingerprints.setdefault((_h, _variant, _fp), []).append(_name)
        _dupes = [sorted(v) for v in _fingerprints.values() if len(v) > 1]
        if _dupes:
            raise RuntimeError(
                "Duplicate label datasets detected across distinct strategies: "
                + "; ".join(",".join(v) for v in _dupes)
            )

    if incremental_persist:
        _manifest_path = os.path.join(
            cfg["data_root"], "artifacts", run_id, "labels", "labels_manifest.json"
        )
        _manifest = {
            "run_id": run_id,
            "datasets": persisted_manifest,
        }
        with open(_manifest_path, "w") as _mf:
            json.dump(_manifest, _mf, indent=2, sort_keys=True)
        tprint(
            f"Wrote labels manifest with {len(persisted_manifest)} entries to {_manifest_path}"
        )
        return {}

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
    from .gamma_specialist import train_gamma_specialist
    from .trap_specialist import train_trap_specialist

    tprint("=" * 60)
    tprint("TRAINING SPECIALIST MODELS")
    tprint("=" * 60)

    specialist_models = {}

    # NOTE: Trap Specialist (GMM) removed from pipeline
    # # 1. Trap Specialist (GMM-based quality filter)
    # NOTE: Trap Specialist (GMM) training disabled
    # try:
    #     trap_model = train_trap_specialist(panel, feats, cfg, syms, ts_end)
    #     specialist_models["trap_model"] = trap_model
    # except Exception as e:
    #     tprint(f"ERROR: Trap Specialist training failed: {e}")
    #     specialist_models["trap_model"] = None

    # 2. Gamma Specialist (ExtraTrees regression for volatility)
    # NOTE: Gamma Specialist (GMM) training disabled
    # try:
    #     gamma_model = train_gamma_specialist(panel, feats, cfg, syms, ts_end)
    #     specialist_models["gamma_model"] = gamma_model
    # except Exception as e:
    #     tprint(f"ERROR: Gamma Specialist training failed: {e}")
    #     specialist_models["gamma_model"] = None

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
    ic_raw = (
        _safe_spearman(y_pred, y_raw_ret) if y_raw_ret is not None else float("nan")
    )

    trades_day_10 = _avg_trades_per_day_global(
        y_pred, 0.10, np.asarray(groups) if groups is not None else None
    )
    trades_day_30 = _avg_trades_per_day_global(
        y_pred, 0.30, np.asarray(groups) if groups is not None else None
    )

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
        r_squared = float("nan")

    # WinRate@k: fraction with raw_ret > 0 (economically meaningful)
    base_win = float(np.mean(r > 0))
    win10 = float(np.mean(r[m_top10] > 0)) if m_top10.any() else float("nan")
    win40_m = topk_mask(y_pred, 0.40, groups=groups)
    win40 = float(np.mean(r[win40_m] > 0)) if win40_m.any() else float("nan")

    # AvgRet@k in bps (×10000)
    avg_ret_top10 = (
        float(np.mean(r[m_top10])) * 10000 if m_top10.any() else float("nan")
    )
    avg_ret_bot10 = (
        float(np.mean(r[m_bot10])) * 10000 if m_bot10.any() else float("nan")
    )

    # CVaR@10: mean of worst 20% of raw returns in top decile
    if m_top10.any() and has_raw:
        top10_rets = r[m_top10]
        n_worst = max(1, int(0.20 * len(top10_rets)))
        cvar10 = float(np.mean(np.sort(top10_rets)[:n_worst])) * 10000
    else:
        cvar10 = float("nan")

    # Sharpe@10 and Gain-to-Pain@10: per-timestamp Top10 avg return stability
    sharpe10 = float("nan")
    gtp10 = float("nan")
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
    tgt_top10 = float(np.mean(y_true[m_top10])) if m_top10.any() else float("nan")
    pred_top10 = float(np.mean(y_pred[m_top10])) if m_top10.any() else float("nan")

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


class AuxHeadWrapper:
    def __init__(
        self,
        model,
        selected_features_idx,
        is_utility=False,
        mae_model=None,
        mfe_model=None,
        util_kwargs=None,
        output_transform=None,
    ):
        self.model = model
        self.selected_features_idx = np.asarray(selected_features_idx, dtype=int)
        self.is_utility = is_utility
        self.mae_model = mae_model
        self.mfe_model = mfe_model
        self.util_kwargs = util_kwargs or {}
        self.output_transform = output_transform or {}
        self.selected_features = (
            []
        )  # To satisfy missing coverage checks if accessed blindly

    def predict(self, X):
        X_vals = X.values if hasattr(X, "values") else np.asarray(X)
        if self.is_utility:
            mae_hat = self.mae_model.predict(X_vals)
            mfe_hat = self.mfe_model.predict(X_vals)
            if bool(self.util_kwargs.get("standardize", False)):
                return smooth_utility_from_log_heads_standardized(
                    log_mfe=mfe_hat,
                    log_mae=mae_hat,
                    tp=float(self.util_kwargs.get("tp", 0.02)),
                    sl=float(self.util_kwargs.get("sl", 0.01)),
                    alpha=float(self.util_kwargs.get("alpha", 6.0)),
                    mfe_mean=float(self.util_kwargs.get("mfe_mean", 0.0)),
                    mfe_std=float(self.util_kwargs.get("mfe_std", 1.0)),
                    mae_mean=float(self.util_kwargs.get("mae_mean", 0.0)),
                    mae_std=float(self.util_kwargs.get("mae_std", 1.0)),
                ).astype(np.float32)
            return smooth_utility_from_log_heads(
                log_mfe=mfe_hat,
                log_mae=mae_hat,
                tp=float(self.util_kwargs.get("tp", 0.02)),
                sl=float(self.util_kwargs.get("sl", 0.01)),
                alpha=float(self.util_kwargs.get("alpha", 6.0)),
            ).astype(np.float32)
        else:
            if self.model is None:
                return np.zeros(len(X_vals), dtype=np.float32)
            X_sel = X_vals[:, self.selected_features_idx]
            pred = self.model.predict(X_sel).astype(np.float32)
            tr_kind = str(self.output_transform.get("kind", "")).lower()
            if tr_kind == "rank_to_log":
                qx = np.asarray(self.output_transform.get("x", []), dtype=float)
                qy = np.asarray(self.output_transform.get("y", []), dtype=float)
                if qx.size >= 2 and qy.size == qx.size:
                    pred = np.interp(np.clip(pred, 0.0, 1.0), qx, qy).astype(np.float32)
            return pred


def build_excursion_targets(
    mfe: np.ndarray,
    mae: np.ndarray,
    atr_h: np.ndarray,
    eps_atr: float = 1e-8,
    eps_asym: float = 1e-6,
) -> dict:
    """Shared helper to robustly compute MFE, MAE, and Asymmetry targets."""
    mfe = np.asarray(mfe, dtype=float)
    mae = np.asarray(mae, dtype=float)
    atr = np.asarray(atr_h, dtype=float)

    atr_safe = np.maximum(atr, eps_atr)

    mfe_norm = np.maximum(mfe / atr_safe, 0.0)
    mae_norm = np.maximum(mae / atr_safe, 0.0)

    mfe_cap = np.clip(mfe_norm, 0.0, 3.0)
    y_mfe = np.log1p(np.maximum(mfe_cap - 0.1, 0.0))
    y_mae = np.log1p(mae_norm)

    # Use a bounded continuous ratio for asymmetry
    c = 0.5  # scaling constant
    y_asym = (mfe_norm - mae_norm) / (mfe_norm + mae_norm + c)

    return {
        "mfe_norm": mfe_norm,
        "mae_norm": mae_norm,
        "y_mfe": y_mfe,
        "y_mae": y_mae,
        "y_asym": y_asym,
    }


def train_meta_models_from_artifacts(
    datasets, cfg, alpha_models, base_variant_models=None
):
    base_variant_models = dict(base_variant_models or {})
    """Train only meta models from datasets and pre-trained alpha models."""
    import time as _time

    _t0_meta = _time.monotonic()
    tprint("train_meta_models_from_artifacts: starting")
    # NOTE: strategy iteration now driven by get_strategies(cfg) - no hardcoded trade_sides/kinds
    alpha_half = max(1, len(alpha_models) // 2)
    meta_models = {}
    meta_gate_results = []
    _bucket_y_ret = {}  # per-bucket raw returns for OOF saving
    _bucket_metadata = {}
    _aux_head_oof = {}  # per-bucket shared-fold auxiliary head OOF outputs
    include_meta_reg = bool(cfg.get("meta_train_regression_bucket_model", False))
    include_meta_clf = bool(cfg.get("meta_clf_enabled", cfg.get("meta_race_include_classifiers", True)))
    require_meta_move_prob = bool(cfg.get("meta_require_classifier_barrier_probs", True))
    if require_meta_move_prob and not include_meta_clf:
        raise RuntimeError(
            "meta_clf_enabled=False, but policy-aligned meta/sizer runs require the binary move head export."
        )
    _ps_regime_cols = [
        str(c)
        for c in cfg.get("position_sizer_regime_feature_keys", [])
        if isinstance(c, str) and c
    ]
    reports_dir = resolve_reports_dir(cfg.get("reports_root"))
    _meta_hpo_out_dir = os.path.join(str(cfg.get("data_root", "data")), "hpo_out")
    tprint(f"Meta training starting with datasets: {list(datasets.keys())}")
    tprint(
        f"Meta training config label_horizons_hours: {cfg.get('label_horizons_hours')}"
    )
    tprint(
        f"Meta training heads-only mode: reg_enabled={include_meta_reg} clf_enabled={include_meta_clf}"
    )

    # Load optimize policy params dynamically (for policy-aligned targets/selection context)
    _run_id_for_policy = str(cfg.get("run_id", ""))
    _policy_params_blob = (
        load_best_policy_params_from_optimise(
            cfg.get("data_root", "../data"), _run_id_for_policy
        )
        if _run_id_for_policy
        else {}
    )
    if _policy_params_blob:
        _n_buckets = (
            len(_policy_params_blob.get("buckets", {}))
            if isinstance(_policy_params_blob, dict)
            else 0
        )
        tprint(
            f"Meta training: loaded optimise policy params for run_id={_run_id_for_policy} (buckets={_n_buckets})"
        )
    else:
        tprint(
            "Meta training: optimise policy params not found for current run (using return-derived utility fallback)."
        )

    _strategy_cfg_map = {s["strategy_id"]: s for s in get_strategies(cfg)}

    def _alpha_conf_for_strategy(trade_side: str, strategy_id: str):
        if not alpha_models:
            return None
        side_bundle = alpha_models.get(trade_side)
        if isinstance(side_bundle, dict) and strategy_id in side_bundle:
            return side_bundle.get(strategy_id)
        direct_bundle = alpha_models.get(strategy_id)
        if isinstance(direct_bundle, dict):
            return direct_bundle
        return None

    def _collect_horizon_oof(trade_side, strategy_id):
        conf_local = _alpha_conf_for_strategy(trade_side, strategy_id)
        _strategy_local = _strategy_cfg_map.get(
            strategy_id, {"strategy_id": strategy_id}
        )
        _expected_horizons = strategy_runtime_horizons(_strategy_local, cfg)
        if not conf_local:
            return {}, {
                **{int(h): "missing_alpha_bucket" for h in _expected_horizons},
            }
        if bool(conf_local.get("downstream_blocked", False)):
            reason = str(
                (conf_local.get("alpha_diag", {}) or {}).get(
                    "blocked_reason", "base_strategy_blocked"
                )
            )
            tprint(
                f"Meta training: skipping blocked strategy_id={strategy_id} "
                f"trade_side={trade_side} reason={reason}"
            )
            return {}, {**{int(h): reason for h in _expected_horizons}}
        models_by_h_local = conf_local.get("models_by_h", {})
        out = {}
        skip = {}
        for h_local in _expected_horizons:
            ds_key_local = f"train_{strategy_id}_{h_local}"
            if ds_key_local not in datasets:
                skip[h_local] = "missing_dataset"
                continue
            race_local = (
                models_by_h_local.get(h_local, {}).get("model")
                if h_local in models_by_h_local
                else None
            )
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
        tprint(
            f"Meta training: horizons found for {strategy_id}: {list(out.keys())} (skip: {skip})"
        )
        return out, skip

    def _align_oof_to_union(df_union, source_df, source_oof):
        if (
            "__ts__" in df_union.columns
            and "__symbol__" in df_union.columns
            and "__ts__" in source_df.columns
            and "__symbol__" in source_df.columns
        ):
            return _align_values_by_ts_symbol_keys(
                df_union["__ts__"].values,
                df_union["__symbol__"].values,
                source_df["__ts__"].values,
                source_df["__symbol__"].values,
                source_oof,
                fill_value=0.5,
                dtype=np.float32,
            )
        n_use = min(len(df_union), len(source_oof))
        aligned = np.full(len(df_union), 0.5, dtype=np.float32)
        aligned[:n_use] = source_oof[:n_use].astype(np.float32)
        return aligned

    def _race_sigma_vector(race, key: str, fallback_len: int) -> np.ndarray:
        if race is None:
            return np.full(fallback_len, np.nan, dtype=np.float32)
        detailed = getattr(race, "detailed_metrics", {}) or {}
        best_name = getattr(race, "best_model_name", None)
        if best_name in detailed and key in detailed[best_name]:
            vals = np.asarray(detailed[best_name][key], dtype=np.float32)
            if len(vals) >= fallback_len:
                return vals[:fallback_len]
            out = np.full(fallback_len, np.nan, dtype=np.float32)
            out[: len(vals)] = vals
            return out
        for dm in detailed.values():
            if key in dm:
                vals = np.asarray(dm[key], dtype=np.float32)
                if len(vals) >= fallback_len:
                    return vals[:fallback_len]
                out = np.full(fallback_len, np.nan, dtype=np.float32)
                out[: len(vals)] = vals
                return out
        return np.full(fallback_len, np.nan, dtype=np.float32)

    def _collect_wide_tight_variant_features(
        side: str,
        kind: str,
        df_union: pd.DataFrame,
        horizon_dfs_local: dict,
    ) -> tuple[pd.DataFrame, list[str]]:
        feature_dict: dict[str, np.ndarray] = {}
        feature_cols: list[str] = []
        for h in sorted(horizon_dfs_local.keys()):
            wide_key = (side, kind, int(h), "wide")
            tight_key = (side, kind, int(h), "tight")
            wide_info = base_variant_models.get(wide_key)
            tight_info = base_variant_models.get(tight_key)
            if not wide_info or not tight_info:
                continue
            wide_race = wide_info.get("model")
            tight_race = tight_info.get("model")
            if wide_race is None or tight_race is None:
                continue
            wide_ds = datasets.get(f"train_{kind}_{int(h)}_wide")
            tight_ds = datasets.get(f"train_{kind}_{int(h)}_tight")
            if wide_ds is None or tight_ds is None:
                continue
            wide_oof = getattr(wide_race, "oof_probs", None)
            tight_oof = getattr(tight_race, "oof_probs", None)
            if wide_oof is None or tight_oof is None:
                continue
            wide_pred = _align_oof_to_union(df_union, wide_ds, wide_oof)
            tight_pred = _align_oof_to_union(df_union, tight_ds, tight_oof)
            wide_sigma = _align_values_by_ts_symbol_keys(
                df_union["__ts__"].values,
                df_union["__symbol__"].values,
                wide_ds["__ts__"].values,
                wide_ds["__symbol__"].values,
                _race_sigma_vector(wide_race, "oof_sigma_trees", len(wide_ds)),
                fill_value=np.nan,
                dtype=np.float32,
            )
            tight_sigma = _align_values_by_ts_symbol_keys(
                df_union["__ts__"].values,
                df_union["__symbol__"].values,
                tight_ds["__ts__"].values,
                tight_ds["__symbol__"].values,
                _race_sigma_vector(tight_race, "oof_sigma_trees", len(tight_ds)),
                fill_value=np.nan,
                dtype=np.float32,
            )
            wide_robust_sigma = _align_values_by_ts_symbol_keys(
                df_union["__ts__"].values,
                df_union["__symbol__"].values,
                wide_ds["__ts__"].values,
                wide_ds["__symbol__"].values,
                _race_sigma_vector(wide_race, "oof_sigma_robust", len(wide_ds)),
                fill_value=np.nan,
                dtype=np.float32,
            )
            tight_robust_sigma = _align_values_by_ts_symbol_keys(
                df_union["__ts__"].values,
                df_union["__symbol__"].values,
                tight_ds["__ts__"].values,
                tight_ds["__symbol__"].values,
                _race_sigma_vector(tight_race, "oof_sigma_robust", len(tight_ds)),
                fill_value=np.nan,
                dtype=np.float32,
            )
            base_name = f"base_H{int(h)}"
            feature_dict[f"{base_name}_wide"] = wide_pred.astype(np.float32)
            feature_dict[f"{base_name}_tight"] = tight_pred.astype(np.float32)
            feature_dict[f"{base_name}_sigma_wide"] = wide_sigma.astype(np.float32)
            feature_dict[f"{base_name}_sigma_tight"] = tight_sigma.astype(np.float32)
            feature_dict[f"{base_name}_robust_sigma_wide"] = wide_robust_sigma.astype(
                np.float32
            )
            feature_dict[f"{base_name}_robust_sigma_tight"] = tight_robust_sigma.astype(
                np.float32
            )
            feature_dict.update(
                build_wide_tight_pair_features(
                    wide_pred,
                    tight_pred,
                    base_name=base_name,
                    sigma_wide=wide_sigma,
                    sigma_tight=tight_sigma,
                    robust_sigma_wide=wide_robust_sigma,
                    robust_sigma_tight=tight_robust_sigma,
                )
            )
            feature_cols.extend(
                [
                    f"{base_name}_wide",
                    f"{base_name}_tight",
                    f"{base_name}_sigma_wide",
                    f"{base_name}_sigma_tight",
                    f"{base_name}_robust_sigma_wide",
                    f"{base_name}_robust_sigma_tight",
                ]
            )
            feature_cols.extend(
                [
                    f"{base_name}_avg",
                    f"{base_name}_diff",
                    f"{base_name}_abs_diff",
                    f"{base_name}_rel_diff",
                    f"{base_name}_sigma_avg",
                    f"{base_name}_cv_wide",
                    f"{base_name}_cv_tight",
                    f"{base_name}_cv_avg",
                    f"{base_name}_agreement_strength",
                    f"{base_name}_reliability",
                ]
            )
        if feature_dict:
            return pd.DataFrame(feature_dict, index=df_union.index), feature_cols
        return pd.DataFrame(index=df_union.index), []

    def _train_aux_heads_shared_folds(
        X_num,
        y_u,
        y_mae,
        y_mfe,
        y_asym,
        trade_mask,
        timestamps,
        cv_embargo_bars=12,
        bucket_id: str | None = None,
        data_root: str = "data",
        run_id: str = "default",
        hpo_out_dir: str | None = None,
    ):
        """Train auxiliary meta heads with shared purged folds and return OOF preds.

        Heads: Utility(U), MAE(q70), MFE(q70).
        Missing labels are handled via placeholder target + zero sample_weight (no semantic fallback).
        """
        from sklearn.ensemble import ExtraTreesRegressor

        from extreme_price_movements.periods_symbols_management import (
            EventSchema,
            SlicePlanner,
            SlicePlannerConfig,
        )

        try:
            from lightgbm import LGBMRegressor
        except Exception:
            LGBMRegressor = None
        _Xdf_in = (
            X_num
            if isinstance(X_num, pd.DataFrame)
            else pd.DataFrame(
                np.asarray(X_num, dtype=float),
                columns=[f"f_{i}" for i in range(np.asarray(X_num).shape[1])],
            )
        )
        Xv = np.asarray(_Xdf_in.values, dtype=float)
        n = len(Xv)
        tm = (
            np.asarray(trade_mask, dtype=bool)
            if trade_mask is not None
            else np.ones(n, dtype=bool)
        )
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
        y_asym_raw = _arr(y_asym)
        valid_mae = np.isfinite(y_mae_raw)
        valid_mfe = np.isfinite(y_mfe_raw)
        valid_asym = np.isfinite(y_asym_raw)

        y_mae_fit = np.where(valid_mae, y_mae_raw, 0.0)
        y_mfe_fit = np.where(valid_mfe, y_mfe_raw, 0.0)
        y_asym_fit = np.where(valid_asym, y_asym_raw, 0.0)

        oof_u = np.full(n, np.nan, dtype=float)
        oof_mae_q70 = np.full(n, np.nan, dtype=float)
        oof_mfe = np.full(n, np.nan, dtype=float)
        oof_asym = np.full(n, np.nan, dtype=float)

        util_cfg_nested = (
            (cfg.get("meta", {}) or {}).get("utility_smooth", {})
            if isinstance(cfg.get("meta", {}), dict)
            else {}
        )
        utility_tp = float(
            util_cfg_nested.get("tp", cfg.get("meta_utility_smooth_tp", 0.02))
        )
        utility_sl = float(
            util_cfg_nested.get("sl", cfg.get("meta_utility_smooth_sl", 0.01))
        )
        utility_alpha = float(
            util_cfg_nested.get("alpha", cfg.get("meta_utility_smooth_alpha", 15.0))
        )
        utility_loss_name = str(
            util_cfg_nested.get("loss", cfg.get("meta_utility_smooth_loss", "huber"))
        ).lower()
        utility_loss_weight = float(
            util_cfg_nested.get(
                "loss_weight", cfg.get("meta_utility_smooth_loss_weight", 1.0)
            )
        )

        # Use SlicePlanner for temporal CV
        events = pd.DataFrame(
            {
                "event_id": np.arange(n, dtype=np.int64),
                "symbol": np.repeat("ALL", n),
                "t0": pd.to_datetime(timestamps, utc=True, errors="coerce"),
                "t1": pd.to_datetime(timestamps, utc=True, errors="coerce")
                + pd.Timedelta(seconds=cv_embargo_bars),
            }
        )
        p_cfg = SlicePlannerConfig.fast_defaults(schema=EventSchema())
        p_cfg = p_cfg.__class__(
            **{
                **p_cfg.__dict__,
                "preset": p_cfg.preset.__class__(
                    preset_name=p_cfg.preset.preset_name,
                    outer=p_cfg.preset.outer,
                    inner=p_cfg.preset.inner.__class__(n_splits=3),
                    sampling=p_cfg.preset.sampling,
                    symbol_policy=p_cfg.preset.symbol_policy,
                    purge_policy=p_cfg.preset.purge_policy,
                ),
                "silent": True,
                "min_rows_per_fold": 1,
                "min_symbols_per_fold": 1,
            }
        )
        bundle = SlicePlanner(p_cfg).build(events)
        _splits_shared = [
            (plan.fit_idx, plan.predict_idx)
            for plan in bundle["consumer_plans"]["ridge_sizer_fit"]
            if plan.tag == "predict_outer_test"
            and plan.fit_idx.size > 0
            and plan.predict_idx.size > 0
        ]
        if not _splits_shared:
            raise ValueError("SlicePlanner failed to generate auxiliary head splits")

        # keep selector simple/robust: all numeric features
        idx_q = np.arange(Xv.shape[1], dtype=int)
        idx_mfe = np.arange(Xv.shape[1], dtype=int)
        idx_u = np.arange(Xv.shape[1], dtype=int)
        idx_asym = np.arange(Xv.shape[1], dtype=int)

        # --- MDI feature selection for MAE and MFE heads (independent per head) ---
        _mdi_n_target = int(cfg.get("aux_head_mdi_n_target", 50))
        _mdi_n_min = int(cfg.get("aux_head_mdi_n_min", 30))
        _mdi_n_max = int(cfg.get("aux_head_mdi_n_max", 60))
        try:
            _tm_idx = np.where(tm)[0]
            _Xdf_tm = _Xdf_in.iloc[_tm_idx].copy()
            _col_names = list(_Xdf_in.columns)
            _col_to_idx = {c: i for i, c in enumerate(_col_names)}

            _fs_report_dir = os.path.join(
                str(cfg.get("data_root", "data")),
                "artifacts",
                str(cfg.get("run_id", "default")),
                "fs_reports",
            )
            _sel_family_map = dict(cfg.get("selector_feature_family_map", {}) or {})
            _prev_run_id_aux = (
                cfg.get("prev_run_id")
                or cfg.get("prior_run_id")
                or cfg.get("warm_start_run_id")
            )

            def _load_prev_selected(_head_key: str):
                if not _prev_run_id_aux:
                    return None
                _p = os.path.join(
                    str(cfg.get("data_root", "data")),
                    "artifacts",
                    str(_prev_run_id_aux),
                    "fs_reports",
                    _head_key,
                    "selected_features.json",
                )
                if not os.path.exists(_p):
                    return None
                try:
                    with open(_p, "r", encoding="utf-8") as _f:
                        return list((json.load(_f) or {}).get("selected_features", []))
                except Exception:
                    return None

            def _run_aux_head_mdi(_head_name, _y_fit, _valid_mask):
                _y_sel = np.asarray(_y_fit, dtype=float)[_tm_idx]
                _w_sel = (np.asarray(_valid_mask, dtype=float) * tm.astype(float))[
                    _tm_idx
                ]
                _clean_mask = np.isfinite(_Xdf_tm.values).all(axis=1) & np.isfinite(
                    _y_sel
                )
                _X_clean = _Xdf_tm[_clean_mask]
                _y_clean = _y_sel[_clean_mask]
                _w_clean = _w_sel[_clean_mask]
                if len(_X_clean) < 200 or _X_clean.shape[1] <= _mdi_n_min:
                    tprint(
                        f"  aux_head MDI[{bucket_id}][{_head_name}]: skipped "
                        f"(n={len(_X_clean)}, feats={_X_clean.shape[1]})"
                    )
                    return np.arange(Xv.shape[1], dtype=int)
                _mdi_max = _bounded_sample_cap(
                    len(_X_clean),
                    absolute_cap=int(
                        cfg.get(
                            "meta_aux_selector_max_samples",
                            cfg.get("meta_selector_max_samples", 30000),
                        )
                    ),
                    pct_cap=float(
                        cfg.get(
                            "meta_aux_selector_max_pct",
                            cfg.get("meta_selector_max_pct", 1.0),
                        )
                    ),
                )
                if len(_X_clean) > _mdi_max > 0:
                    _sel_sub_idx = _subsample_indices_time_balanced(
                        len(_X_clean),
                        _mdi_max,
                        y=(_y_clean if np.isfinite(_y_clean).any() else None),
                    )
                    _X_clean = _X_clean.iloc[_sel_sub_idx]
                    _y_clean = _y_clean[_sel_sub_idx]
                    _w_clean = _w_clean[_sel_sub_idx]
                    tprint(
                        f"  aux_head MDI[{bucket_id}][{_head_name}]: subsampled to {len(_X_clean)} rows"
                    )
                _head_l = str(_head_name).lower()
                if _head_l == "utility":
                    _head_cfg = dict(cfg.get("aux_utility_selector_cfg", {}) or {})
                    _sel_target = "regression"
                    _sel_loss = "huber"
                    _sel_alpha = None
                    _top_metric = str(
                        _head_cfg.get("selector_top_metric", "top30_mean_utility")
                    )
                elif _head_l == "mfe":
                    _head_cfg = dict(cfg.get("aux_mfe_selector_cfg", {}) or {})
                    _sel_target = "regression"
                    _sel_loss = "huber"
                    _sel_alpha = None
                    _top_metric = _head_cfg.get("selector_top_metric", "ic_top")
                else:
                    _head_cfg = dict(cfg.get("aux_mae_selector_cfg", {}) or {})
                    _sel_target = "regression"
                    _sel_loss = "huber"
                    _sel_alpha = None
                    _top_metric = _head_cfg.get("selector_top_metric", "ic_top")
                _mdi_base = ExtraTreesRegressor(
                    n_estimators=int(_head_cfg.get("analysis_n_estimators", 160)),
                    max_depth=5,
                    min_samples_leaf=40,
                    max_features="sqrt",
                    n_jobs=2,
                    random_state=42,
                )
                _head_key = f"aux_{bucket_id}_{_head_l}"
                _mdi_res = mdi_feature_selection_v3(
                    _X_clean,
                    _y_clean,
                    base_model=_mdi_base,
                    sample_weight=_w_clean,
                    selector_y=_y_clean,
                    selector_target=_sel_target,
                    selector_loss=_sel_loss,
                    selector_alpha=_sel_alpha,
                    selector_head_name=_head_key,
                    selector_top_metric=_top_metric,
                    selector_report_dir=_fs_report_dir,
                    selector_prev_selected=_load_prev_selected(_head_key),
                    selector_family_map=_sel_family_map,
                    selector_focus_top_frac=float(
                        _head_cfg.get("selector_focus_top_frac", 1.0)
                    ),
                    selector_emit_report=bool(
                        _head_cfg.get("selector_emit_report", True)
                    ),
                    analysis_n_estimators=int(
                        _head_cfg.get("analysis_n_estimators", 160)
                    ),
                    analysis_max_samples=int(
                        _head_cfg.get("analysis_max_samples", 2500)
                    ),
                    min_samples_leaf_pct=float(
                        _head_cfg.get("min_samples_leaf_pct", 0.02)
                    ),
                    selector_max_missing_frac=float(
                        _head_cfg.get("selector_max_missing_frac", 0.15)
                    ),
                    selector_near_constant_dominance=float(
                        _head_cfg.get("selector_near_constant_dominance", 0.999)
                    ),
                    composite_weights={
                        "top30": float(_head_cfg.get("top30", 0.0)),
                        "global": float(_head_cfg.get("global", 0.55)),
                        "stability": float(_head_cfg.get("stability", 0.25)),
                        "frequency": float(_head_cfg.get("frequency", 0.15)),
                        "interaction": float(_head_cfg.get("interaction", 0.05)),
                    },
                    end_features=min(_mdi_n_target, _mdi_n_max),
                    cumulative_cap=0.99,
                    min_share=0.0005,
                    min_features=_mdi_n_min,
                    max_features_pct=0.8,
                )
                _sel_cols = list(_mdi_res.selected_features)
                _sel_cap = max(_mdi_n_min, min(_mdi_n_target, _mdi_n_max))
                _sel_cols = _cap_selected_features(
                    _sel_cols,
                    _col_names,
                    target_cap=_sel_cap,
                    min_features=_mdi_n_min,
                )
                _sel_idx = np.array(
                    [_col_to_idx[c] for c in _sel_cols if c in _col_to_idx], dtype=int
                )
                if len(_sel_idx) < _mdi_n_min:
                    tprint(
                        f"  aux_head MDI[{bucket_id}][{_head_name}]: too few features selected "
                        f"({len(_sel_idx)}), using all"
                    )
                    return np.arange(Xv.shape[1], dtype=int)
                tprint(
                    f"  aux_head MDI[{bucket_id}][{_head_name}]: selected {len(_sel_idx)}/{Xv.shape[1]} features"
                )
                return _sel_idx

            idx_q = _run_aux_head_mdi("mae_q70", y_mae_fit, valid_mae)
            idx_mfe = _run_aux_head_mdi("mfe", y_mfe_fit, valid_mfe)
            idx_u = _run_aux_head_mdi("utility", y_u_raw, np.isfinite(y_u_raw) & tm)
            idx_asym = _run_aux_head_mdi("asym", y_asym_fit, valid_asym)
        except Exception as _e_mdi:
            tprint(
                f"  aux_head MDI[{bucket_id}]: feature selection failed ({_e_mdi}), using all features"
            )

        def _normalize_clip_weights(w, lo=0.1, hi=3.0, tr_idx=None):
            w = np.asarray(w, dtype=float)
            w = np.where(np.isfinite(w), w, 0.0)
            w = np.clip(w, 0.0, None)
            pos = w > 0
            if np.any(pos):
                if tr_idx is not None:
                    train_pos = np.zeros_like(pos)
                    train_pos[tr_idx] = pos[tr_idx]
                    mean_w = (
                        float(np.mean(w[train_pos]))
                        if np.any(train_pos)
                        else float(np.mean(w[pos]))
                    )
                else:
                    mean_w = float(np.mean(w[pos]))
                w[pos] = w[pos] / max(mean_w, 1e-12)
                w[pos] = np.clip(w[pos], lo, hi)

                # Re-calculate mean after clipping
                if tr_idx is not None:
                    mean_w2 = (
                        float(np.mean(w[train_pos]))
                        if np.any(train_pos)
                        else float(np.mean(w[pos]))
                    )
                else:
                    mean_w2 = float(np.mean(w[pos]))
                w[pos] = w[pos] / max(mean_w2, 1e-12)
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
                    w *= np.clip(mult, 0.1, 3.0)
            return _normalize_clip_weights(w, tr_idx=tr_idx)

        def _tail_multiplier_asymmetric(y_fit, w_base, tr_idx):
            w = np.asarray(w_base, dtype=float).copy()
            tr_pos = tr_idx[w[tr_idx] > 0]
            if len(tr_pos) >= 20:
                yt = y_fit[tr_pos]
                p30 = float(np.nanpercentile(yt, 30))
                p70 = float(np.nanpercentile(yt, 70))
                p95 = float(np.nanpercentile(yt, 95))
                if (
                    np.isfinite(p30)
                    and np.isfinite(p70)
                    and np.isfinite(p95)
                    and p95 > p70
                ):
                    high = np.clip((y_fit - p70) / max(p95 - p70, 1e-9), 0.0, 1.0)
                    low = np.clip(
                        (p30 - y_fit) / max(p30 - np.nanpercentile(yt, 5), 1e-9),
                        0.0,
                        1.0,
                    )
                    w *= 1.0 + 0.25 * high - 0.10 * low
            return _normalize_clip_weights(w)

        def _rank_pct_target(y):
            y = np.asarray(y, dtype=float)
            out = np.full(len(y), 0.5, dtype=float)
            m = np.isfinite(y)
            if np.sum(m) > 1:
                r = pd.Series(y[m]).rank(method="average", pct=True).values
                out[m] = np.clip(r, 0.0, 1.0)
            return out

        def _qbin_mid_target(y, n_bins=20):
            y = np.asarray(y, dtype=float)
            out = np.full(len(y), 0.5, dtype=float)
            m = np.isfinite(y)
            if np.sum(m) <= n_bins:
                return _rank_pct_target(y)
            yv = y[m]
            edges = np.percentile(yv, np.linspace(0.0, 100.0, n_bins + 1))
            edges[0] -= 1e-12
            edges[-1] += 1e-12
            bins = np.clip(np.digitize(y, edges) - 1, 0, n_bins - 1)
            out[m] = (bins[m] + 0.5) / float(n_bins)
            return out

        def _rank_tail_amp_target(y, top_start=0.70, amp=0.50):
            r = _rank_pct_target(y)
            t = np.maximum(0.0, r - float(top_start))
            out = r + float(amp) * t
            return np.clip(out, 0.0, 1.0)

        def _rank_to_log_mapping(y_log_ref):
            y_ref = np.asarray(y_log_ref, dtype=float)
            y_ref = y_ref[np.isfinite(y_ref)]
            if y_ref.size < 10:
                x = np.array([0.0, 1.0], dtype=float)
                med = (
                    float(np.nanmedian(y_log_ref[np.isfinite(y_log_ref)]))
                    if np.isfinite(y_log_ref).any()
                    else 0.0
                )
                y = np.array([med, med], dtype=float)
                return x, y
            y_sorted = np.sort(y_ref)
            x = np.linspace(0.0, 1.0, y_sorted.size, dtype=float)
            return x, y_sorted

        def _head_eval_metrics(y_true_log, pred_log, fold_stats):
            y_true = np.asarray(y_true_log, dtype=float)
            pred = np.asarray(pred_log, dtype=float)
            mask = np.isfinite(y_true) & np.isfinite(pred)
            if np.sum(mask) < 30:
                return {
                    "ic": -1.0,
                    "ic_top30": -1.0,
                    "ic_top20": -1.0,
                    "ic_top10": -1.0,
                    "mono": -1.0,
                    "ece_top30": 1.0,
                    "stability": 0.0,
                    "stability_top30": 1.0,
                    "mse": 9e9,
                    "score": -9e9,
                }
            yt = y_true[mask]
            pr = pred[mask]
            ic = _safe_spearman(pr, yt)

            def _top_ic(_frac):
                _n_top = max(1, int(np.ceil(float(_frac) * len(pr))))
                _idx = np.argpartition(pr, -_n_top)[-_n_top:]
                return _safe_spearman(pr[_idx], yt[_idx]), _idx

            top_frac = float(cfg.get("aux_head_select_top_frac", 0.30))
            ic_top30, idx_top = _top_ic(top_frac)
            ic_top20, _ = _top_ic(0.20)
            ic_top10, _ = _top_ic(0.10)
            ic_top30 = _safe_spearman(pr[idx_top], yt[idx_top])
            dec = np.clip((pd.Series(pr).rank(pct=True).values * 10).astype(int), 0, 9)
            dec_means = pd.Series(yt).groupby(dec).mean()
            if len(dec_means) >= 3:
                mono = _safe_spearman(np.arange(len(dec_means)), dec_means.values)
            else:
                mono = 0.0
            # Top-bucket calibration: ECE on top-30% using 5 score bins.
            ece_top30 = 1.0
            try:
                pr_top = pr[idx_top]
                yt_top = yt[idx_top]
                if len(pr_top) >= 20:
                    pos_thr = float(np.nanmedian(yt_top))
                    if np.isfinite(pos_thr):
                        y_pos = (yt_top >= pos_thr).astype(float)
                        edges = np.quantile(pr_top, np.linspace(0.0, 1.0, 6))
                        edges[0] -= 1e-12
                        edges[-1] += 1e-12
                        ece = 0.0
                        for bi in range(5):
                            lo, hi = edges[bi], edges[bi + 1]
                            if bi == 4:
                                m_bin = (pr_top >= lo) & (pr_top <= hi)
                            else:
                                m_bin = (pr_top >= lo) & (pr_top < hi)
                            if not np.any(m_bin):
                                continue
                            conf = float((bi + 0.5) / 5.0)
                            acc = float(np.mean(y_pos[m_bin]))
                            ece += abs(acc - conf) * (
                                np.sum(m_bin) / max(len(pr_top), 1)
                            )
                        ece_top30 = float(np.clip(ece, 0.0, 1.0))
            except Exception:
                ece_top30 = 1.0
            fold_ics = [
                float(d.get("ic", np.nan))
                for d in fold_stats
                if np.isfinite(d.get("ic", np.nan))
            ]
            fold_ics_top30 = [
                float(d.get("ic_top30", np.nan))
                for d in fold_stats
                if np.isfinite(d.get("ic_top30", np.nan))
            ]
            fold_mses = [
                float(d.get("mse", np.nan))
                for d in fold_stats
                if np.isfinite(d.get("mse", np.nan))
            ]
            stability = float(np.std(fold_ics)) if len(fold_ics) >= 2 else 1.0
            stability_top30 = (
                float(np.std(fold_ics_top30)) if len(fold_ics_top30) >= 2 else 1.0
            )
            mse_val = (
                float(np.mean(fold_mses))
                if len(fold_mses) > 0
                else float(np.mean((pr - yt) ** 2))
            )

            # Rebalance weights to focus less on pure top-tail and more on full-distribution and magnitude calibration
            w_top = float(cfg.get("aux_head_select_w_ic_top", 0.35))
            w_all = float(cfg.get("aux_head_select_w_ic_all", 0.30))
            w_mono = float(cfg.get("aux_head_select_w_mono", 0.15))
            w_stab = float(cfg.get("aux_head_select_w_stability", 0.15))
            w_stab_top30 = float(cfg.get("aux_head_select_w_stability_top30", 0.10))
            w_ece = float(cfg.get("aux_head_select_w_ece_top", 0.15))
            w_mse = float(
                cfg.get("aux_head_select_w_mse", 0.50)
            )  # new penalty term for MSE in log_raw space

            score = float(
                w_top * ic_top30
                + w_all * ic
                + w_mono * mono
                - w_stab * stability
                - w_stab_top30 * stability_top30
                - w_ece * ece_top30
                - w_mse * mse_val
            )
            return {
                "ic": float(ic),
                "ic_top30": float(ic_top30),
                "ic_top20": float(ic_top20),
                "ic_top10": float(ic_top10),
                "mono": float(mono),
                "ece_top30": float(ece_top30),
                "stability": stability,
                "stability_top30": stability_top30,
                "mse": mse_val,
                "score": score,
            }

        def _fit_predict_head(kind, Xu_tr, y_tr, Xu_va, sw_tr, sw_va):
            kind = str(kind).lower()
            if kind in ("ridge", "ridge_optuna"):
                from sklearn.linear_model import Ridge
                from sklearn.preprocessing import RobustScaler

                Xu_tr = np.asarray(Xu_tr, dtype=float)
                Xu_va = np.asarray(Xu_va, dtype=float)
                y_tr = np.asarray(y_tr, dtype=float)
                sw_tr = np.asarray(sw_tr, dtype=float) if sw_tr is not None else None
                sc = RobustScaler()
                Xtr_s = sc.fit_transform(Xu_tr)
                Xva_s = sc.transform(Xu_va)
                best_alpha = float(cfg.get("aux_head_ridge_alpha_default", 1.0))
                if kind == "ridge_optuna":
                    n_trials = int(cfg.get("aux_head_weight_optuna_trials", 12))
                    if n_trials > 0 and len(Xtr_s) >= 120:
                        try:
                            import importlib

                            if importlib.util.find_spec("optuna") is not None:
                                import optuna

                                optuna.logging.set_verbosity(optuna.logging.WARNING)
                                inner = _PKF(
                                    n_splits=int(
                                        cfg.get(
                                            "aux_head_weight_optuna_inner_splits", 3
                                        )
                                    ),
                                    purge=max(1, int(cv_embargo_bars // 2)),
                                    embargo=max(1, int(cv_embargo_bars // 2)),
                                )
                                _time_idx_inner = np.arange(len(Xtr_s), dtype=float)

                                def _obj(trial):
                                    a = trial.suggest_float(
                                        "alpha",
                                        float(
                                            cfg.get("aux_head_ridge_alpha_min", 1e-3)
                                        ),
                                        float(
                                            cfg.get("aux_head_ridge_alpha_max", 100.0)
                                        ),
                                        log=True,
                                    )
                                    vals = []
                                    for itr, iva in inner.split(_time_idx_inner):
                                        if len(itr) < 30 or len(iva) < 10:
                                            continue
                                        m = Ridge(alpha=float(a), random_state=42)
                                        _sw = sw_tr[itr] if sw_tr is not None else None
                                        m.fit(Xtr_s[itr], y_tr[itr], sample_weight=_sw)
                                        p = m.predict(Xtr_s[iva])
                                        vals.append(_safe_spearman(p, y_tr[iva]))
                                    if not vals:
                                        return -1.0
                                    return float(np.nanmean(vals))

                                sampler = optuna.samplers.TPESampler(seed=42)
                                study = optuna.create_study(
                                    direction="maximize", sampler=sampler
                                )
                                study.optimize(
                                    _obj, n_trials=n_trials, show_progress_bar=False
                                )
                                if study.best_trial is not None:
                                    best_alpha = float(
                                        study.best_trial.params.get("alpha", best_alpha)
                                    )
                        except Exception:
                            pass
                m = Ridge(alpha=float(best_alpha), random_state=42)
                m.fit(Xtr_s, y_tr, sample_weight=sw_tr)
                return m.predict(Xva_s)
            if kind == "lgbm" and LGBMRegressor is not None:
                n_est = int(cfg.get("aux_head_ablation_lgbm_estimators", 200))
                m = LGBMRegressor(
                    objective="regression",
                    n_estimators=n_est,
                    learning_rate=0.03,
                    num_leaves=63,
                    min_child_samples=50,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    n_jobs=2,
                )
                m.fit(Xu_tr, y_tr, sample_weight=sw_tr)
                return m.predict(Xu_va)
            # Default comparator: ExtraTrees (single model for target/weight comparison).
            m = ExtraTreesRegressor(
                n_estimators=int(cfg.get("aux_head_ablation_et_estimators", 120)),
                max_depth=6,
                min_samples_leaf=30,
                max_features="sqrt",
                random_state=42,
                n_jobs=2,
            )
            m.fit(Xu_tr, y_tr, sample_weight=sw_tr)
            return m.predict(Xu_va)

        def _head_weight_vector(
            y_fit_log, base_w, weight_name, tr_idx, weight_lambda=1.0
        ):
            _w_name = str(weight_name).lower()
            _w_base = np.asarray(base_w, dtype=float).copy()
            if _w_name == "none":
                return _w_base
            if _w_name == "symmetric_tail":
                _w_tail = _tail_multiplier(y_fit_log, _w_base, tr_idx)
            else:
                _w_tail = _tail_multiplier_asymmetric(y_fit_log, _w_base, tr_idx)
            if _w_name == "top30_tail":
                _thr = (
                    float(np.nanpercentile(y_fit_log[tr_idx], 70))
                    if len(tr_idx) > 20
                    else np.nan
                )
                if np.isfinite(_thr):
                    _w_tail *= 1.0 + 0.25 * (y_fit_log >= _thr).astype(float)
                    _w_tail = _normalize_clip_weights(_w_tail, tr_idx=tr_idx)
            elif _w_name == "asym_magnitude":
                _mag = np.abs(y_fit_log)
                _p50 = (
                    float(np.nanpercentile(_mag[tr_idx], 50))
                    if len(tr_idx) > 20
                    else 0.0
                )
                if np.isfinite(_p50) and _p50 > 1e-9:
                    _w_tail *= np.clip(_mag / _p50, 0.5, 2.0)
                    _w_tail = _normalize_clip_weights(_w_tail, tr_idx=tr_idx)
            _lam = float(np.clip(weight_lambda, 0.0, 1.0))
            _w = _w_base + _lam * (_w_tail - _w_base)
            return _normalize_clip_weights(_w, tr_idx=tr_idx)

        def _available_head_models():
            raw = cfg.get(
                "aux_head_model_race_candidates", ["lgbm", "extratrees", "ridge"]
            )
            if not isinstance(raw, (list, tuple)):
                raw = [raw]
            out = []
            for it in raw:
                k = str(it).lower().strip()
                if k in ("et", "extra_trees"):
                    k = "extratrees"
                if k in ("ridge_optuna",):
                    k = "ridge"
                if k == "lgbm" and LGBMRegressor is None:
                    continue
                if k not in ("lgbm", "extratrees", "ridge"):
                    continue
                if k not in out:
                    out.append(k)
            if not out:
                out = (
                    ["lgbm", "extratrees", "ridge"]
                    if LGBMRegressor is not None
                    else ["extratrees", "ridge"]
                )
            return out

        def _run_head_model_race(
            head_name,
            idx_cols,
            y_fit_log,
            y_fit_train_target,
            base_w,
            target_choice,
            weight_choice,
            weight_lambda,
        ):
            cand_models = _available_head_models()
            race_rows = []
            _train_cap = _bounded_sample_cap(
                n,
                absolute_cap=int(
                    cfg.get("meta_aux_model_race_train_max_samples", 40000)
                ),
                pct_cap=float(cfg.get("meta_aux_model_race_train_max_pct", 1.0)),
            )
            _valid_cap = _bounded_sample_cap(
                n,
                absolute_cap=int(
                    cfg.get("meta_aux_model_race_valid_max_samples", 12000)
                ),
                pct_cap=float(cfg.get("meta_aux_model_race_valid_max_pct", 0.5)),
            )
            for _mk in cand_models:
                _oof_c = np.full(n, np.nan, dtype=float)
                _fold_stats = []
                for tr, va in _splits_shared:
                    tr_idx = tr[tm[tr]]
                    va_idx = va[tm[va]]
                    if len(tr_idx) < 50 or len(va_idx) == 0:
                        continue
                    if len(tr_idx) > _train_cap > 0:
                        tr_idx = tr_idx[
                            _subsample_indices_time_balanced(
                                len(tr_idx),
                                _train_cap,
                                y=y_fit_train_target[tr_idx],
                            )
                        ]
                    if len(va_idx) > _valid_cap > 0:
                        va_idx = va_idx[
                            _subsample_indices_time_balanced(
                                len(va_idx),
                                _valid_cap,
                                y=y_fit_train_target[va_idx],
                            )
                        ]
                    _w = _head_weight_vector(
                        y_fit_log,
                        base_w,
                        weight_choice,
                        tr_idx,
                        weight_lambda=weight_lambda,
                    )
                    try:
                        _pred = np.asarray(
                            _fit_predict_head(
                                _mk,
                                Xv[tr_idx][:, idx_cols],
                                y_fit_train_target[tr_idx],
                                Xv[va_idx][:, idx_cols],
                                _w[tr_idx],
                                _w[va_idx],
                            ),
                            dtype=float,
                        )
                        if target_choice != "log_raw":
                            _qx, _qy = _rank_to_log_mapping(y_fit_log)
                            _pred_log = np.interp(np.clip(_pred, 0.0, 1.0), _qx, _qy)
                        else:
                            _pred_log = _pred
                        _oof_c[va_idx] = _pred_log
                        _n30 = max(1, int(np.ceil(0.30 * len(_pred_log))))
                        _idx30 = np.argpartition(_pred_log, -_n30)[-_n30:]
                        _fold_stats.append(
                            {
                                "ic": _safe_spearman(_pred_log, y_fit_log[va_idx]),
                                "ic_top30": _safe_spearman(
                                    _pred_log[_idx30], y_fit_log[va_idx][_idx30]
                                ),
                            }
                        )
                    except Exception:
                        continue
                _fill = (
                    float(np.nanmedian(y_fit_log[np.isfinite(y_fit_log)]))
                    if np.isfinite(y_fit_log).any()
                    else 0.0
                )
                _oof_c = np.where(np.isfinite(_oof_c), _oof_c, _fill)
                _m = _head_eval_metrics(y_fit_log, _oof_c, _fold_stats)
                race_rows.append({"model": _mk, "metrics": _m, "oof": _oof_c})
            if not race_rows:
                fallback = "lgbm" if LGBMRegressor is not None else "extratrees"
                return fallback, []
            race_rows.sort(key=lambda x: x["metrics"]["score"], reverse=True)
            best = race_rows[0]
            tprint(
                f"  aux_{head_name}_model_race[{bucket_id or 'bucket'}]: winner={best['model']} "
                f"ic={best['metrics']['ic']:.4f} ic_top30={best['metrics']['ic_top30']:.4f} "
                f"ic_top20={best['metrics']['ic_top20']:.4f} ic_top10={best['metrics']['ic_top10']:.4f} "
                f"ece_top30={best['metrics']['ece_top30']:.4f} mono={best['metrics']['mono']:.4f} "
                f"stab={best['metrics']['stability']:.4f} stab_t30={best['metrics']['stability_top30']:.4f}"
            )
            return str(best["model"]), race_rows

        def _eval_head_oof(
            model_kind,
            idx_cols,
            y_fit_log,
            y_train_target,
            target_choice,
            weight_choice,
            base_w,
            weight_lambda=1.0,
        ):
            _oof = np.full(n, np.nan, dtype=float)
            _fold_stats = []
            _train_cap = _bounded_sample_cap(
                n,
                absolute_cap=int(cfg.get("meta_aux_eval_train_max_samples", 40000)),
                pct_cap=float(cfg.get("meta_aux_eval_train_max_pct", 1.0)),
            )
            _valid_cap = _bounded_sample_cap(
                n,
                absolute_cap=int(cfg.get("meta_aux_eval_valid_max_samples", 12000)),
                pct_cap=float(cfg.get("meta_aux_eval_valid_max_pct", 0.5)),
            )
            for tr, va in _splits_shared:
                tr_idx = tr[tm[tr]]
                va_idx = va[tm[va]]
                if len(tr_idx) < 50 or len(va_idx) == 0:
                    continue
                if len(tr_idx) > _train_cap > 0:
                    tr_idx = tr_idx[
                        _subsample_indices_time_balanced(
                            len(tr_idx),
                            _train_cap,
                            y=y_train_target[tr_idx],
                        )
                    ]
                if len(va_idx) > _valid_cap > 0:
                    va_idx = va_idx[
                        _subsample_indices_time_balanced(
                            len(va_idx),
                            _valid_cap,
                            y=y_train_target[va_idx],
                        )
                    ]
                _w = _head_weight_vector(
                    y_fit_log,
                    base_w,
                    weight_choice,
                    tr_idx,
                    weight_lambda=weight_lambda,
                )
                try:
                    _pred = np.asarray(
                        _fit_predict_head(
                            model_kind,
                            Xv[tr_idx][:, idx_cols],
                            y_train_target[tr_idx],
                            Xv[va_idx][:, idx_cols],
                            _w[tr_idx],
                            _w[va_idx],
                        ),
                        dtype=float,
                    )
                    if target_choice != "log_raw":
                        # CRITICAL FIX: compute mapping ONLY on fold training targets to prevent leakage
                        _qx, _qy = _rank_to_log_mapping(y_fit_log[tr_idx])
                        _pred_log = np.interp(np.clip(_pred, 0.0, 1.0), _qx, _qy)
                    else:
                        _pred_log = _pred
                    _oof[va_idx] = _pred_log

                    _va_true_log = y_fit_log[va_idx]
                    _mask_va = np.isfinite(_pred_log) & np.isfinite(_va_true_log)
                    if _mask_va.sum() > 5:
                        _n30 = max(1, int(np.ceil(0.30 * _mask_va.sum())))
                        _idx30 = np.argpartition(_pred_log[_mask_va], -_n30)[-_n30:]
                        _fold_stats.append(
                            {
                                "ic": _safe_spearman(
                                    _pred_log[_mask_va], _va_true_log[_mask_va]
                                ),
                                "ic_top30": _safe_spearman(
                                    _pred_log[_mask_va][_idx30],
                                    _va_true_log[_mask_va][_idx30],
                                ),
                                "mse": float(
                                    np.mean(
                                        (_pred_log[_mask_va] - _va_true_log[_mask_va])
                                        ** 2
                                    )
                                ),
                            }
                        )
                except Exception:
                    continue
            _fill = (
                float(np.nanmedian(y_fit_log[np.isfinite(y_fit_log)]))
                if np.isfinite(y_fit_log).any()
                else 0.0
            )
            _oof = np.where(np.isfinite(_oof), _oof, _fill)
            _m = _head_eval_metrics(y_fit_log, _oof, _fold_stats)
            return _oof, _m

        def _select_target_stage(
            head_name, idx_cols, y_fit_log, target_map, target_variants, base_w
        ):
            rows = []
            for _t_name in [str(v).lower() for v in target_variants]:
                if _t_name not in target_map:
                    continue
                _oof, _m = _eval_head_oof(
                    model_kind="extratrees",
                    idx_cols=idx_cols,
                    y_fit_log=y_fit_log,
                    y_train_target=target_map[_t_name],
                    target_choice=_t_name,
                    weight_choice="none",
                    base_w=base_w,
                )
                rows.append({"target": _t_name, "oof": _oof, "metrics": _m})
            rows.sort(key=lambda x: x["metrics"]["score"], reverse=True)
            if rows:
                best = rows[0]
                tprint(
                    f"  aux_{head_name}_target_stage[{bucket_id or 'bucket'}]: target={best['target']} "
                    f"ic={best['metrics']['ic']:.4f} ic_top30={best['metrics']['ic_top30']:.4f} "
                    f"ic_top20={best['metrics']['ic_top20']:.4f} ic_top10={best['metrics']['ic_top10']:.4f} "
                    f"ece_top30={best['metrics']['ece_top30']:.4f} mono={best['metrics']['mono']:.4f} "
                    f"stab={best['metrics']['stability']:.4f} stab_t30={best['metrics']['stability_top30']:.4f}"
                )
                return best["target"], rows
            fallback_target = "rank_pct" if "rank_pct" in target_map else "log_raw"
            return fallback_target, []

        def _select_weight_stage(
            head_name,
            idx_cols,
            y_fit_log,
            y_train_target,
            target_choice,
            weight_variants,
            base_w,
        ):
            rows = []
            _lambda_grid = cfg.get(
                "aux_head_weight_lambda_grid", [0.1 * i for i in range(1, 11)]
            )
            if not isinstance(_lambda_grid, (list, tuple)):
                _lambda_grid = [0.1 * i for i in range(1, 11)]
            _lambda_grid = [float(v) for v in _lambda_grid if np.isfinite(v) and v > 0]
            if not _lambda_grid:
                _lambda_grid = [1.0]
            for _w_name in [str(v).lower() for v in weight_variants]:
                if _w_name == "none":
                    _oof, _m = _eval_head_oof(
                        model_kind="ridge_optuna",
                        idx_cols=idx_cols,
                        y_fit_log=y_fit_log,
                        y_train_target=y_train_target,
                        target_choice=target_choice,
                        weight_choice=_w_name,
                        base_w=base_w,
                        weight_lambda=0.0,
                    )
                    rows.append(
                        {"weights": _w_name, "lambda": 0.0, "oof": _oof, "metrics": _m}
                    )
                    continue

                _best = None
                _prev_score = -9e9
                for _lam in _lambda_grid:
                    _oof, _m = _eval_head_oof(
                        model_kind="ridge_optuna",
                        idx_cols=idx_cols,
                        y_fit_log=y_fit_log,
                        y_train_target=y_train_target,
                        target_choice=target_choice,
                        weight_choice=_w_name,
                        base_w=base_w,
                        weight_lambda=_lam,
                    )
                    _row = {
                        "weights": _w_name,
                        "lambda": float(_lam),
                        "oof": _oof,
                        "metrics": _m,
                    }
                    if _best is None or _m["score"] > _best["metrics"]["score"]:
                        _best = _row
                    # stop right when score no longer improves
                    if _m["score"] <= (_prev_score + 1e-9):
                        break
                    _prev_score = _m["score"]
                if _best is not None:
                    rows.append(_best)
            rows.sort(key=lambda x: x["metrics"]["score"], reverse=True)
            if rows:
                baseline = next((r for r in rows if r["weights"] == "none"), rows[0])
                min_gain = float(cfg.get("aux_head_weight_min_gain_vs_none", 1e-4))
                top_tol = float(cfg.get("aux_head_weight_topk_tolerance", 1e-4))
                better = []
                for r in rows:
                    if r["weights"] == "none":
                        continue
                    if r["metrics"]["score"] <= (
                        baseline["metrics"]["score"] + min_gain
                    ):
                        continue
                    # Hard reject weighted candidates that degrade top slices vs baseline.
                    if (
                        float(r["metrics"].get("ic_top30", -1.0))
                        < float(baseline["metrics"].get("ic_top30", -1.0)) - top_tol
                        or float(r["metrics"].get("ic_top20", -1.0))
                        < float(baseline["metrics"].get("ic_top20", -1.0)) - top_tol
                        or float(r["metrics"].get("ic_top10", -1.0))
                        < float(baseline["metrics"].get("ic_top10", -1.0)) - top_tol
                    ):
                        continue
                    better.append(r)
                best = (
                    max(better, key=lambda r: r["metrics"]["score"])
                    if better
                    else baseline
                )
                tprint(
                    f"  aux_{head_name}_weight_stage[{bucket_id or 'bucket'}]: weights={best['weights']} lambda={best.get('lambda', 0.0):.2f} "
                    f"ic={best['metrics']['ic']:.4f} ic_top30={best['metrics']['ic_top30']:.4f} "
                    f"ic_top20={best['metrics']['ic_top20']:.4f} ic_top10={best['metrics']['ic_top10']:.4f} "
                    f"ece_top30={best['metrics']['ece_top30']:.4f} mono={best['metrics']['mono']:.4f} "
                    f"stab={best['metrics']['stability']:.4f} stab_t30={best['metrics']['stability_top30']:.4f}"
                )
                return (
                    best["weights"],
                    float(best.get("lambda", 0.0)),
                    best["oof"],
                    rows,
                )
            return "none", 0.0, np.full(n, np.nan, dtype=float), []

        mae_target_variants = [
            str(v).lower()
            for v in list(cfg.get("aux_mae_target_variants", ["log_raw"]))
        ]
        mae_weight_variants = [
            str(v).lower()
            for v in list(
                cfg.get(
                    "aux_mae_weight_variants",
                    ["none", "asymmetric_tail", "symmetric_tail", "top30_tail"],
                )
            )
        ]
        _base_mae_w = _normalize_clip_weights(
            valid_mae.astype(float) * tm.astype(float)
        )
        _mae_targets = {
            "log_raw": y_mae_fit.astype(float),
            "rank_pct": _rank_pct_target(y_mae_fit),
            "qbin_mid": _qbin_mid_target(
                y_mae_fit, n_bins=int(cfg.get("aux_mae_qbin_bins", 20))
            ),
        }
        mae_target_choice, _mae_target_candidates = _select_target_stage(
            head_name="mae_q70",
            idx_cols=idx_q,
            y_fit_log=y_mae_fit,
            target_map=_mae_targets,
            target_variants=mae_target_variants,
            base_w=_base_mae_w,
        )
        _mae_target_train = _mae_targets.get(
            mae_target_choice, _mae_targets["rank_pct"]
        )
        (
            mae_weight_choice,
            mae_weight_lambda,
            oof_mae_q70,
            _mae_weight_candidates,
        ) = _select_weight_stage(
            head_name="mae_q70",
            idx_cols=idx_q,
            y_fit_log=y_mae_fit,
            y_train_target=_mae_target_train,
            target_choice=mae_target_choice,
            weight_variants=mae_weight_variants,
            base_w=_base_mae_w,
        )

        mfe_target_variants = [
            str(v).lower()
            for v in list(cfg.get("aux_mfe_target_variants", ["log_raw"]))
        ]
        mfe_weight_variants = [
            str(v).lower()
            for v in list(
                cfg.get(
                    "aux_mfe_weight_variants",
                    ["none", "asymmetric_tail", "symmetric_tail", "top30_tail"],
                )
            )
        ]
        _base_mfe_w = _normalize_clip_weights(
            valid_mfe.astype(float) * tm.astype(float)
        )
        _mfe_targets = {
            "log_raw": y_mfe_fit.astype(float),
            "rank_pct": _rank_pct_target(y_mfe_fit),
            "qbin_mid": _qbin_mid_target(
                y_mfe_fit, n_bins=int(cfg.get("aux_mfe_qbin_bins", 20))
            ),
        }
        mfe_target_choice, _mfe_target_candidates = _select_target_stage(
            head_name="mfe",
            idx_cols=idx_mfe,
            y_fit_log=y_mfe_fit,
            target_map=_mfe_targets,
            target_variants=mfe_target_variants,
            base_w=_base_mfe_w,
        )
        _mfe_target_train = _mfe_targets.get(
            mfe_target_choice, _mfe_targets["rank_pct"]
        )
        (
            mfe_weight_choice,
            mfe_weight_lambda,
            oof_mfe,
            _mfe_weight_candidates,
        ) = _select_weight_stage(
            head_name="mfe",
            idx_cols=idx_mfe,
            y_fit_log=y_mfe_fit,
            y_train_target=_mfe_target_train,
            target_choice=mfe_target_choice,
            weight_variants=mfe_weight_variants,
            base_w=_base_mfe_w,
        )

        # ASYM HEAD UPGRADE
        asym_target_variants = [
            str(v).lower()
            for v in list(
                cfg.get("aux_asym_target_variants", ["log_raw"])
            )
        ]
        asym_weight_variants = [
            str(v).lower()
            for v in list(
                cfg.get(
                    "aux_asym_weight_variants",
                    ["none", "asymmetric_tail", "symmetric_tail", "top30_tail"],
                )
            )
        ]
        _base_asym_w = _normalize_clip_weights(
            valid_asym.astype(float) * tm.astype(float)
        )
        _asym_targets = {
            "log_raw": y_asym_fit.astype(float),
            "rank_pct": _rank_pct_target(y_asym_fit),
            "qbin_mid": _qbin_mid_target(
                y_asym_fit, n_bins=int(cfg.get("aux_asym_qbin_bins", 20))
            ),
        }
        asym_target_choice, _asym_target_candidates = _select_target_stage(
            head_name="asym",
            idx_cols=idx_asym,
            y_fit_log=y_asym_fit,
            target_map=_asym_targets,
            target_variants=asym_target_variants,
            base_w=_base_asym_w,
        )
        _asym_target_train = _asym_targets.get(
            asym_target_choice, _asym_targets["rank_pct"]
        )
        (
            asym_weight_choice,
            asym_weight_lambda,
            oof_asym,
            _asym_weight_candidates,
        ) = _select_weight_stage(
            head_name="asym",
            idx_cols=idx_asym,
            y_fit_log=y_asym_fit,
            y_train_target=_asym_target_train,
            target_choice=asym_target_choice,
            weight_variants=asym_weight_variants,
            base_w=_base_asym_w,
        )

        def _fill(oof, y_fit):
            o = np.asarray(oof, dtype=float)
            fill = (
                float(np.nanmedian(y_fit[np.isfinite(y_fit)]))
                if np.isfinite(y_fit).any()
                else 0.0
            )
            o = np.where(np.isfinite(o), o, fill)
            return o.astype(np.float32)

        _fs_report = {
            "u": {"n_in": int(_Xdf_in.shape[1]), "n_selected": int(len(idx_u))},
            "mae_q70": {
                "n_in": int(_Xdf_in.shape[1]),
                "n_selected": int(len(idx_q)),
                "target_choice": mae_target_choice,
                "weight_choice": mae_weight_choice,
                "weight_lambda": float(mae_weight_lambda),
            },
            "mfe": {
                "n_in": int(_Xdf_in.shape[1]),
                "n_selected": int(len(idx_mfe)),
                "target_choice": mfe_target_choice,
                "weight_choice": mfe_weight_choice,
                "weight_lambda": float(mfe_weight_lambda),
            },
            "asym": {
                "n_in": int(_Xdf_in.shape[1]),
                "n_selected": int(len(idx_asym)),
                "target_choice": asym_target_choice,
                "weight_choice": asym_weight_choice,
                "weight_lambda": float(asym_weight_lambda),
            },
            "config": {
                "weights": "per-head-validity+train-only-tail",
                "utility_head": "deterministic_from_mfe_mae",
            },
        }
        if _mae_target_candidates:
            _fs_report["mae_q70"]["target_stage_top"] = [
                {
                    "target": c["target"],
                    "ic": float(c["metrics"]["ic"]),
                    "ic_top30": float(c["metrics"]["ic_top30"]),
                    "ic_top20": float(c["metrics"].get("ic_top20", np.nan)),
                    "ic_top10": float(c["metrics"].get("ic_top10", np.nan)),
                    "ece_top30": float(c["metrics"]["ece_top30"]),
                    "mono": float(c["metrics"]["mono"]),
                    "stability": float(c["metrics"]["stability"]),
                    "stability_top30": float(
                        c["metrics"].get("stability_top30", np.nan)
                    ),
                    "score": float(c["metrics"]["score"]),
                }
                for c in _mae_target_candidates[:5]
            ]
        if _mae_weight_candidates:
            _fs_report["mae_q70"]["weight_stage_top"] = [
                {
                    "weights": c["weights"],
                    "lambda": float(c.get("lambda", 0.0)),
                    "ic": float(c["metrics"]["ic"]),
                    "ic_top30": float(c["metrics"]["ic_top30"]),
                    "ic_top20": float(c["metrics"].get("ic_top20", np.nan)),
                    "ic_top10": float(c["metrics"].get("ic_top10", np.nan)),
                    "ece_top30": float(c["metrics"]["ece_top30"]),
                    "mono": float(c["metrics"]["mono"]),
                    "stability": float(c["metrics"]["stability"]),
                    "stability_top30": float(
                        c["metrics"].get("stability_top30", np.nan)
                    ),
                    "score": float(c["metrics"]["score"]),
                }
                for c in _mae_weight_candidates[:5]
            ]
        if _mfe_target_candidates:
            _fs_report["mfe"]["target_stage_top"] = [
                {
                    "target": c["target"],
                    "ic": float(c["metrics"]["ic"]),
                    "ic_top30": float(c["metrics"]["ic_top30"]),
                    "ic_top20": float(c["metrics"].get("ic_top20", np.nan)),
                    "ic_top10": float(c["metrics"].get("ic_top10", np.nan)),
                    "ece_top30": float(c["metrics"]["ece_top30"]),
                    "mono": float(c["metrics"]["mono"]),
                    "stability": float(c["metrics"]["stability"]),
                    "stability_top30": float(
                        c["metrics"].get("stability_top30", np.nan)
                    ),
                    "score": float(c["metrics"]["score"]),
                }
                for c in _mfe_target_candidates[:5]
            ]
        if _mfe_weight_candidates:
            _fs_report["mfe"]["weight_stage_top"] = [
                {
                    "weights": c["weights"],
                    "lambda": float(c.get("lambda", 0.0)),
                    "ic": float(c["metrics"]["ic"]),
                    "ic_top30": float(c["metrics"]["ic_top30"]),
                    "ic_top20": float(c["metrics"].get("ic_top20", np.nan)),
                    "ic_top10": float(c["metrics"].get("ic_top10", np.nan)),
                    "ece_top30": float(c["metrics"]["ece_top30"]),
                    "mono": float(c["metrics"]["mono"]),
                    "stability": float(c["metrics"]["stability"]),
                    "stability_top30": float(
                        c["metrics"].get("stability_top30", np.nan)
                    ),
                    "score": float(c["metrics"]["score"]),
                }
                for c in _mfe_weight_candidates[:5]
            ]
        if _asym_target_candidates:
            _fs_report["asym"]["target_stage_top"] = [
                {
                    "target": c["target"],
                    "ic": float(c["metrics"]["ic"]),
                    "ic_top30": float(c["metrics"]["ic_top30"]),
                    "ic_top20": float(c["metrics"].get("ic_top20", np.nan)),
                    "ic_top10": float(c["metrics"].get("ic_top10", np.nan)),
                    "ece_top30": float(c["metrics"]["ece_top30"]),
                    "mono": float(c["metrics"]["mono"]),
                    "stability": float(c["metrics"]["stability"]),
                    "stability_top30": float(
                        c["metrics"].get("stability_top30", np.nan)
                    ),
                    "score": float(c["metrics"]["score"]),
                }
                for c in _asym_target_candidates[:5]
            ]
        if _asym_weight_candidates:
            _fs_report["asym"]["weight_stage_top"] = [
                {
                    "weights": c["weights"],
                    "lambda": float(c.get("lambda", 0.0)),
                    "ic": float(c["metrics"]["ic"]),
                    "ic_top30": float(c["metrics"]["ic_top30"]),
                    "ic_top20": float(c["metrics"].get("ic_top20", np.nan)),
                    "ic_top10": float(c["metrics"].get("ic_top10", np.nan)),
                    "ece_top30": float(c["metrics"]["ece_top30"]),
                    "mono": float(c["metrics"]["mono"]),
                    "stability": float(c["metrics"]["stability"]),
                    "stability_top30": float(
                        c["metrics"].get("stability_top30", np.nan)
                    ),
                    "score": float(c["metrics"]["score"]),
                }
                for c in _asym_weight_candidates[:5]
            ]

        try:
            if bucket_id:
                _rp = os.path.join(
                    data_root, "artifacts", run_id, "fs_reports", f"{bucket_id}_cap12"
                )
                os.makedirs(_rp, exist_ok=True)
                import json as _json

                for _h in ["u", "mae_q70", "mfe"]:
                    with open(os.path.join(_rp, f"{_h}.json"), "w") as _f:
                        _json.dump(_fs_report[_h] | {"head": _h}, _f)
        except Exception as _e_fsrep:
            tprint(f"Warning: failed to persist fs report for {bucket_id}: {_e_fsrep}")

        oof_log_mae_q70_hat = _fill(oof_mae_q70, y_mae_fit)
        oof_log_mfe_hat = _fill(oof_mfe, y_mfe_fit)
        _q_tp = float(cfg.get("meta_utility_smooth_tp_quantile", 0.60))
        _q_sl = float(cfg.get("meta_utility_smooth_sl_quantile", 0.60))
        _blend = float(cfg.get("meta_utility_smooth_quantile_blend", 0.50))
        _tp_min = float(cfg.get("meta_utility_smooth_tp_min", 0.003))
        _tp_max = float(cfg.get("meta_utility_smooth_tp_max", 0.250))
        _sl_min = float(cfg.get("meta_utility_smooth_sl_min", 0.002))
        _sl_max = float(cfg.get("meta_utility_smooth_sl_max", 0.250))

        _mfe_true = np.expm1(np.asarray(y_mfe_fit, dtype=float))
        _mae_true = np.expm1(np.asarray(y_mae_fit, dtype=float))
        _mfe_pred = np.expm1(np.asarray(oof_log_mfe_hat, dtype=float))
        _mae_pred = np.expm1(np.asarray(oof_log_mae_q70_hat, dtype=float))
        _mfe_true = _mfe_true[np.isfinite(_mfe_true)]
        _mae_true = _mae_true[np.isfinite(_mae_true)]
        _mfe_pred = _mfe_pred[np.isfinite(_mfe_pred)]
        _mae_pred = _mae_pred[np.isfinite(_mae_pred)]
        if _mfe_true.size > 20 and _mfe_pred.size > 20:
            utility_tp = float(
                np.clip(
                    _blend * np.quantile(_mfe_true, _q_tp)
                    + (1.0 - _blend) * np.quantile(_mfe_pred, _q_tp),
                    _tp_min,
                    _tp_max,
                )
            )
        if _mae_true.size > 20 and _mae_pred.size > 20:
            utility_sl = float(
                np.clip(
                    _blend * np.quantile(_mae_true, _q_sl)
                    + (1.0 - _blend) * np.quantile(_mae_pred, _q_sl),
                    _sl_min,
                    _sl_max,
                )
            )

        utility_use_z = bool(cfg.get("meta_utility_smooth_use_zscore", True))

        # Calculate full data stats ONLY for final wrapping
        _mfe_mean = (
            float(np.nanmean(y_mfe_fit)) if np.isfinite(y_mfe_fit).any() else 0.0
        )
        _mfe_std = float(np.nanstd(y_mfe_fit)) if np.isfinite(y_mfe_fit).any() else 1.0
        _mae_mean = (
            float(np.nanmean(y_mae_fit)) if np.isfinite(y_mae_fit).any() else 0.0
        )
        _mae_std = float(np.nanstd(y_mae_fit)) if np.isfinite(y_mae_fit).any() else 1.0
        _mfe_std = max(_mfe_std, 1e-6)
        _mae_std = max(_mae_std, 1e-6)

        def _utility_from_logs_fold(
            _log_mfe, _log_mae, _alpha, _mf_mean, _mf_std, _ma_mean, _ma_std
        ):
            if utility_use_z:
                return smooth_utility_from_log_heads_standardized(
                    log_mfe=_log_mfe,
                    log_mae=_log_mae,
                    tp=utility_tp,
                    sl=utility_sl,
                    alpha=_alpha,
                    mfe_mean=_mf_mean,
                    mfe_std=_mf_std,
                    mae_mean=_ma_mean,
                    mae_std=_ma_std,
                )
            return smooth_utility_from_log_heads(
                log_mfe=_log_mfe,
                log_mae=_log_mae,
                tp=utility_tp,
                sl=utility_sl,
                alpha=_alpha,
            )

        def _utility_from_logs(_log_mfe, _log_mae, _alpha):
            # Used for wrapper and target calculation.
            return _utility_from_logs_fold(
                _log_mfe, _log_mae, _alpha, _mfe_mean, _mfe_std, _mae_mean, _mae_std
            )

        _alpha_grid = cfg.get(
            "meta_utility_smooth_alpha_grid", [2.0, 4.0, 6.0, 8.0, 10.0, 12.0]
        )
        if not isinstance(_alpha_grid, (list, tuple)):
            _alpha_grid = [utility_alpha, 6.0]
        _alpha_grid = [float(a) for a in _alpha_grid if np.isfinite(a) and a > 0]
        if not _alpha_grid:
            _alpha_grid = [utility_alpha]

        _best_alpha = float(utility_alpha)
        _best_alpha_score = -9e9
        _u_eval = np.asarray(y_u_raw, dtype=float)

        for _a in _alpha_grid:
            _u_tmp = np.full(n, np.nan, dtype=float)
            for tr, va in _splits_shared:
                tr_idx = tr[tm[tr]]
                va_idx = va[tm[va]]
                if len(tr_idx) < 5 or len(va_idx) == 0:
                    continue
                _fold_mfe_mean = (
                    float(np.nanmean(y_mfe_fit[tr_idx]))
                    if np.isfinite(y_mfe_fit[tr_idx]).any()
                    else 0.0
                )
                _fold_mfe_std = (
                    float(np.nanstd(y_mfe_fit[tr_idx]))
                    if np.isfinite(y_mfe_fit[tr_idx]).any()
                    else 1.0
                )
                _fold_mae_mean = (
                    float(np.nanmean(y_mae_fit[tr_idx]))
                    if np.isfinite(y_mae_fit[tr_idx]).any()
                    else 0.0
                )
                _fold_mae_std = (
                    float(np.nanstd(y_mae_fit[tr_idx]))
                    if np.isfinite(y_mae_fit[tr_idx]).any()
                    else 1.0
                )

                _u_tmp[va_idx] = _utility_from_logs_fold(
                    oof_log_mfe_hat[va_idx],
                    oof_log_mae_q70_hat[va_idx],
                    _a,
                    _fold_mfe_mean,
                    max(_fold_mfe_std, 1e-6),
                    _fold_mae_mean,
                    max(_fold_mae_std, 1e-6),
                )

            _m_eval = np.isfinite(_u_tmp) & np.isfinite(_u_eval)
            if np.sum(_m_eval) < 30:
                continue
            _corr = _safe_spearman(_u_tmp[_m_eval], _u_eval[_m_eval])
            _dyn = float(np.nanstd(_u_tmp[_m_eval]))
            _score = float(_corr + 0.10 * np.log1p(max(_dyn, 0.0)))
            if _score > _best_alpha_score:
                _best_alpha_score = _score
                _best_alpha = float(_a)
        utility_alpha = _best_alpha

        # IMPORTANT: utility OOF is derived strictly from OOF MAE/MFE predictions AND fold-specific stats
        # (never from in-fold/final fits) to prevent leakage into ridge-sizer training.
        oof_u_hat = np.full(n, np.nan, dtype=float)
        for tr, va in _splits_shared:
            tr_idx = tr[tm[tr]]
            va_idx = va[tm[va]]
            if len(tr_idx) < 5 or len(va_idx) == 0:
                continue
            _fold_mfe_mean = (
                float(np.nanmean(y_mfe_fit[tr_idx]))
                if np.isfinite(y_mfe_fit[tr_idx]).any()
                else 0.0
            )
            _fold_mfe_std = (
                float(np.nanstd(y_mfe_fit[tr_idx]))
                if np.isfinite(y_mfe_fit[tr_idx]).any()
                else 1.0
            )
            _fold_mae_mean = (
                float(np.nanmean(y_mae_fit[tr_idx]))
                if np.isfinite(y_mae_fit[tr_idx]).any()
                else 0.0
            )
            _fold_mae_std = (
                float(np.nanstd(y_mae_fit[tr_idx]))
                if np.isfinite(y_mae_fit[tr_idx]).any()
                else 1.0
            )

            oof_u_hat[va_idx] = _utility_from_logs_fold(
                oof_log_mfe_hat[va_idx],
                oof_log_mae_q70_hat[va_idx],
                utility_alpha,
                _fold_mfe_mean,
                max(_fold_mfe_std, 1e-6),
                _fold_mae_mean,
                max(_fold_mae_std, 1e-6),
            )

        # Fill missing OOF values with global utility using median filling if needed
        oof_u_hat = np.where(
            np.isfinite(oof_u_hat),
            oof_u_hat,
            _utility_from_logs(oof_log_mfe_hat, oof_log_mae_q70_hat, utility_alpha),
        ).astype(np.float32)
        u_target = np.asarray(
            _utility_from_logs(y_mfe_fit, y_mae_fit, utility_alpha),
            dtype=float,
        )
        u_loss = smooth_utility_loss(oof_u_hat, u_target, loss=utility_loss_name)
        u_corr = (
            float(spearmanr(oof_u_hat, u_target).correlation)
            if np.isfinite(u_target).sum() > 5
            else float("nan")
        )
        u_corr_realized = (
            float(spearmanr(oof_u_hat, y_u_raw).correlation)
            if np.isfinite(y_u_raw).sum() > 5
            else float("nan")
        )
        tprint(
            f"  utility_smooth[{bucket_id or 'bucket'}]: tp={utility_tp:.4f} sl={utility_sl:.4f} alpha={utility_alpha:.2f} z={utility_use_z} "
            f"loss={u_loss:.6f} weighted={utility_loss_weight * u_loss:.6f} corr_target={u_corr:.4f} corr_realized={u_corr_realized:.4f} "
            f"mfe_hat_mean={float(np.nanmean(oof_log_mfe_hat)):.4f} mae_hat_mean={float(np.nanmean(oof_log_mae_q70_hat)):.4f} "
            f"u_hat_mean={float(np.nanmean(oof_u_hat)):.5f} u_target_mean={float(np.nanmean(u_target)):.5f}"
        )

        def _mae_train_target_full():
            return y_mae_fit

        def _mae_train_weights_full():
            w = _normalize_clip_weights(valid_mae.astype(float) * tm.astype(float))
            return _head_weight_vector(
                y_mae_fit,
                w,
                mae_weight_choice,
                np.arange(n),
                weight_lambda=mae_weight_lambda,
            )

        def _mfe_train_target_full():
            return y_mfe_fit

        def _mfe_train_weights_full():
            w = _normalize_clip_weights(valid_mfe.astype(float) * tm.astype(float))
            return _head_weight_vector(
                y_mfe_fit,
                w,
                mfe_weight_choice,
                np.arange(n),
                weight_lambda=mfe_weight_lambda,
            )

        def _asym_train_target_full():
            return y_asym_fit

        def _asym_train_weights_full():
            w = _normalize_clip_weights(valid_asym.astype(float) * tm.astype(float))
            return _head_weight_vector(
                y_asym_fit,
                w,
                asym_weight_choice,
                np.arange(n),
                weight_lambda=asym_weight_lambda,
            )

        _mae_ref_x, _mae_ref_y = _rank_to_log_mapping(y_mae_fit)
        mae_output_transform = {"kind": "identity"}
        if mae_target_choice != "log_raw":
            mae_output_transform = {
                "kind": "rank_to_log",
                "x": _mae_ref_x,
                "y": _mae_ref_y,
            }
        _mfe_ref_x, _mfe_ref_y = _rank_to_log_mapping(y_mfe_fit)
        mfe_output_transform = {"kind": "identity"}
        if mfe_target_choice != "log_raw":
            mfe_output_transform = {
                "kind": "rank_to_log",
                "x": _mfe_ref_x,
                "y": _mfe_ref_y,
            }

        _asym_ref_x, _asym_ref_y = _rank_to_log_mapping(y_asym_fit)
        asym_output_transform = {"kind": "identity"}
        if asym_target_choice != "log_raw":
            asym_output_transform = {
                "kind": "rank_to_log",
                "x": _asym_ref_x,
                "y": _asym_ref_y,
            }

        mae_final_model_kind = "xgb_parallel_forest"
        mfe_final_model_kind = "xgb_parallel_forest"
        asym_final_model_kind = "xgb_parallel_forest"
        _fs_report["mae_q70"]["final_model_choice"] = mae_final_model_kind
        _fs_report["mfe"]["final_model_choice"] = mfe_final_model_kind
        _fs_report["asym"]["final_model_choice"] = asym_final_model_kind

        if bool(cfg.get("train_full_inference_models", False)):
            tprint("Meta heads: robust mode => full inference retrain enabled")
        else:
            tprint("Meta heads: fast mode => skipping full inference retrain")

        # Train final models on the full dataset for out-of-sample inference
        # Controlled by pipeline planner mode: robust=True enables full inference retrain.
        m_mae_final = None
        m_mfe_final = None
        do_full_inference_retrain = bool(cfg.get("train_full_inference_models", False))
        if len(idx_q) > 0 and np.any(valid_mae & tm) and do_full_inference_retrain:
            y_mae_final_fit = _mae_train_target_full()
            w_mae_full = _mae_train_weights_full()
            m_mae_final = _configure_meta_reg(f"mae_q70_final_{bucket_id}", "aux_mae_selector_cfg")
            m_mae_final.fit(
                pd.DataFrame(Xv[:, idx_q], columns=[f"f_{i}" for i in range(len(idx_q))]),
                y_mae_final_fit,
                sample_weight=w_mae_full,
            )
        if len(idx_mfe) > 0 and np.any(valid_mfe & tm) and do_full_inference_retrain:
            y_mfe_final_fit = _mfe_train_target_full()
            w_mfe_full = _mfe_train_weights_full()
            m_mfe_final = _configure_meta_reg(f"mfe_final_{bucket_id}", "aux_mfe_selector_cfg")
            m_mfe_final.fit(
                pd.DataFrame(Xv[:, idx_mfe], columns=[f"f_{i}" for i in range(len(idx_mfe))]),
                y_mfe_final_fit,
                sample_weight=w_mfe_full,
            )

        _heads_out = {
            "oof_u_hat": oof_u_hat,
            "oof_log_mae_q70_hat": oof_log_mae_q70_hat,
            "oof_log_mfe_hat": oof_log_mfe_hat,
            "oof_asym_hat": oof_asym,
            "oof_u_target": u_target.astype(np.float32),
            "utility_smooth_metrics": {
                "loss": float(u_loss),
                "loss_weight": float(utility_loss_weight),
                "loss_name": utility_loss_name,
                "corr_spearman": float(u_corr),
                "corr_realized_u": float(u_corr_realized),
                "tp_calibrated": float(utility_tp),
                "sl_calibrated": float(utility_sl),
                "alpha_calibrated": float(utility_alpha),
                "use_zscore": bool(utility_use_z),
                "mfe_mean": float(_mfe_mean),
                "mfe_std": float(_mfe_std),
                "mae_mean": float(_mae_mean),
                "mae_std": float(_mae_std),
            },
            "fs_report": _fs_report,
        }

        m_asym_final = None
        if len(idx_asym) > 0 and np.any(valid_asym & tm):
            _asym_full_w = _normalize_clip_weights(
                valid_asym.astype(float) * tm.astype(float)
            )
            _asym_full_w = _head_weight_vector(
                y_asym_fit,
                _asym_full_w,
                asym_weight_choice,
                np.arange(len(y_asym_fit)),
                weight_lambda=asym_weight_lambda,
            )
            m_asym_final = _configure_meta_reg(f"asym_final_{bucket_id}", "aux_asym_selector_cfg")
            m_asym_final.fit(
                pd.DataFrame(Xv[:, idx_asym], columns=[f"f_{i}" for i in range(len(idx_asym))]),
                _asym_train_target_full(),
                sample_weight=_asym_full_w,
            )

        heads_meta = {
            "mae_q70": AuxHeadWrapper(
                model=m_mae_final,
                selected_features_idx=idx_q,
                output_transform=mae_output_transform,
            ),
            "mfe": AuxHeadWrapper(
                model=m_mfe_final,
                selected_features_idx=idx_mfe,
                output_transform=mfe_output_transform,
            ),
            "asym": AuxHeadWrapper(
                model=m_asym_final,
                selected_features_idx=idx_asym,
                output_transform=asym_output_transform,
            ),
        }
        heads_meta["utility"] = AuxHeadWrapper(
            model=None,
            selected_features_idx=idx_u,
            is_utility=True,
            mae_model=heads_meta["mae_q70"],
            mfe_model=heads_meta["mfe"],
            util_kwargs={
                "tp": utility_tp,
                "sl": utility_sl,
                "alpha": utility_alpha,
                "standardize": bool(utility_use_z),
                "mfe_mean": float(_mfe_mean),
                "mfe_std": float(_mfe_std),
                "mae_mean": float(_mae_mean),
                "mae_std": float(_mae_std),
            },
        )

        return _heads_out, heads_meta

    _meta_pipeline_version = str(
        cfg.get("meta_training_pipeline_version", "legacy")
    ).lower()
    _use_aligned_map_v2 = _meta_pipeline_version in {"aligned_map_v2", "v2", "aligned"}

    def _collect_bucket_metadata_frame(_df_local):
        _md = {}
        for _cn in [
            "timestamp",
            "symbol",
            "asset",
            "__ts__",
            "__symbol__",
            "__y_bin__",
            "__y_ret__",
            "__u_policy_net__",
            "__u_policy__",
            "__y_outcome__",
            "exit_code",
            "__mae_ret__",
            "__mfe_ret__",
            "__bars_to_mfe__",
            "__barrier_pct__",
            "__early_inval__",
            "__mr_path_penalty__",
            "__mr_velocity_penalty__",
        ]:
            if _cn in _df_local.columns:
                _md[_cn] = _df_local[_cn].values
        for _cn in _ps_regime_cols:
            if _cn in _df_local.columns:
                _md[_cn] = _df_local[_cn].values
        return _md

    def _meta_map_weights(_target_vec, _df_local, _trade_mask_local):
        _t = np.asarray(_target_vec, dtype=float)
        _t = np.nan_to_num(_t, nan=0.0, posinf=0.0, neginf=0.0)
        _abs = np.abs(_t)
        if np.isfinite(_abs).any():
            _q = float(np.nanpercentile(_abs[np.isfinite(_abs)], 70))
            _s = max(float(np.nanstd(_abs[np.isfinite(_abs)])), 1e-6)
            _w = 1.0 + 0.5 * _sigmoid((_abs - _q) / _s)
        else:
            _w = np.ones(len(_t), dtype=float)
        if "__barrier_pct__" in _df_local.columns:
            _bp = np.clip(
                np.asarray(_df_local["__barrier_pct__"].values, dtype=float), 1e-6, None
            )
            _w *= 0.9 + 0.1 * np.clip(np.nanmedian(_bp) / _bp, 0.5, 1.5)
        _w *= np.asarray(_trade_mask_local, dtype=float)
        _w = np.where(np.isfinite(_w), _w, 0.0)
        _pos = _w > 0
        if np.any(_pos):
            _w[_pos] = _w[_pos] / max(float(np.mean(_w[_pos])), 1e-12)
            _w[_pos] = np.clip(
                _w[_pos],
                float(cfg.get("meta_map_weight_clip_lo", 0.5)),
                float(cfg.get("meta_map_weight_clip_hi", 1.5)),
            )
            _w[_pos] = _w[_pos] / max(float(np.mean(_w[_pos])), 1e-12)
        return _w.astype(np.float32)

    def _tbm_proxy_target(_ret_h, _mfe_h, _mae_h, _tp_pct, _sl_pct):
        _ret_h = np.asarray(_ret_h, dtype=float)
        _mfe_h = np.asarray(_mfe_h, dtype=float)
        _mae_h = np.asarray(_mae_h, dtype=float)
        _out = np.clip(_ret_h, -float(_sl_pct), float(_tp_pct))
        _hit_tp = _mfe_h >= float(_tp_pct)
        _hit_sl = _mae_h >= float(_sl_pct)
        _out = np.where(_hit_tp & ~_hit_sl, float(_tp_pct), _out)
        _out = np.where(_hit_sl & ~_hit_tp, -float(_sl_pct), _out)
        _both = _hit_tp & _hit_sl
        _out = np.where(_both & (_ret_h >= 0.0), float(_tp_pct), _out)
        _out = np.where(_both & (_ret_h < 0.0), -float(_sl_pct), _out)
        return _out.astype(np.float32)

    def _tbm_proxy_target_class(
        _mfe_h, _mae_h, _t_mfe_h, _t_mae_h, _tp_pct, _sl_pct, _horizon_hours
    ):
        """Generate classification labels for TBM: 0=SL, 1=TO, 2=TP
        Uses time-to-barrier for strict first-touch ordering."""
        if _t_mfe_h is None or _t_mae_h is None:
            raise RuntimeError(
                "time_to_mfe/time_to_mae required for strict TBM ordering in _tbm_proxy_target_class"
            )

        _mfe_h = np.asarray(_mfe_h, dtype=float)
        _mae_h = np.asarray(_mae_h, dtype=float)
        _t_mfe_h = np.asarray(_t_mfe_h, dtype=float)
        _t_mae_h = np.asarray(_t_mae_h, dtype=float)
        _horizon_hours = float(_horizon_hours) + 1e-5

        _hit_tp = (_mfe_h >= float(_tp_pct)) & (_t_mfe_h <= _horizon_hours)
        _hit_sl = (_mae_h >= float(_sl_pct)) & (_t_mae_h <= _horizon_hours)
        _both = _hit_tp & _hit_sl

        # 0 = SL hit (and not TP)
        # 1 = Timeout (neither SL nor TP hit)
        # 2 = TP hit (and not SL)
        _out = np.ones(len(_mfe_h), dtype=np.int8)  # Default to TO (1)
        _out[_hit_sl & ~_hit_tp] = 0  # SL
        _out[_hit_tp & ~_hit_sl] = 2  # TP
        _out[_both & (_t_mfe_h < _t_mae_h)] = 2  # Both hit, TP first -> TP
        _out[_both & (_t_mfe_h >= _t_mae_h)] = 0  # Both hit, SL first or tied -> SL
        return _out

    def _fit_aligned_meta_map_heads(
        *,
        side,
        k,
        df,
        X_meta_base,
        meta_groups,
        trade_mask,
        ret_for_h,
        bucket_horizons,
    ):
        _bucket_key = f"{side}_{k}"
        _bucket_n = int(len(df))
        _bucket_horizons = sorted({int(h) for h in (bucket_horizons or [])})
        _native_bucket_h = (
            int(_bucket_horizons[0]) if len(_bucket_horizons) == 1 else None
        )
        _md = _collect_bucket_metadata_frame(df)
        _selector_prev_run = (
            cfg.get("prev_run_id")
            or cfg.get("prior_run_id")
            or cfg.get("warm_start_run_id")
        )
        _fs_dir = os.path.join(
            str(cfg.get("data_root", "data")),
            "artifacts",
            str(cfg.get("run_id", "default")),
            "fs_reports",
        )

        def _prev_selected(_head_key):
            if not _selector_prev_run:
                return None
            _p = os.path.join(
                str(cfg.get("data_root", "data")),
                "artifacts",
                str(_selector_prev_run),
                "fs_reports",
                f"meta_{_head_key}",
                "selected_features.json",
            )
            if not os.path.exists(_p):
                return None
            try:
                with open(_p, "r", encoding="utf-8") as _f:
                    return list((json.load(_f) or {}).get("selected_features", []))
            except Exception:
                return None

        def _configure_meta_reg(_head_name, _selector_cfg_key):
            _m = MetaModel(reports_dir=reports_dir)
            _m.strategy_name = _head_name
            _m.candidate_mode = "xgb_parallel_forest"
            _m.disable_hpo = bool(
                cfg.get("meta_parallel_forest_disable_hpo", False)
            )
            _m.hpo_out_dir = _meta_hpo_out_dir
            _mcw = float(cfg.get("meta_parallel_forest_min_child_weight", 40.0))
            _m.xgb_parallel_forest_params = {
                "objective": "reg:squarederror",
                "n_estimators": int(cfg.get("meta_parallel_forest_rounds", 100)),
                "num_parallel_tree": int(
                    cfg.get("meta_parallel_forest_num_parallel_tree", 20)
                ),
                "max_depth": int(cfg.get("meta_parallel_forest_max_depth", 5)),
                "learning_rate": float(
                    cfg.get("meta_parallel_forest_learning_rate", 0.05)
                ),
                "subsample": 0.75,
                "colsample_bytree": 0.75,
                "reg_alpha": float(cfg.get("meta_parallel_forest_reg_alpha", 2.0)),
                "reg_lambda": float(cfg.get("meta_parallel_forest_reg_lambda", 15.0)),
                "min_child_weight": max(1.0, float(_mcw)),
                "gamma": float(cfg.get("meta_parallel_forest_gamma", 1.5)),
                "tree_method": "hist",
                "random_state": 42,
                "n_jobs": 3,
                "verbosity": 0,
                "early_stopping_rounds": int(
                    cfg.get("meta_parallel_forest_early_stopping_rounds", 20)
                ),
            }
            _m.selector_cfg = dict(cfg.get(_selector_cfg_key, {}) or {})
            _m.selector_report_dir = _fs_dir
            _m.selector_prev_selected = _prev_selected(_head_name)
            _m.selector_family_map = dict(
                cfg.get("selector_feature_family_map", {}) or {}
            )
            _m.selector_target_override = "regression"
            _m.selector_loss_override = "huber"
            return _m

        def _configure_meta_clf(_head_name, _selector_cfg_key):
            _m = MetaClassifierModel(reports_dir=reports_dir)
            _m.strategy_name = _head_name
            _m.candidate_mode = "xgb_parallel_forest"
            _m.disable_hpo = bool(
                cfg.get("meta_parallel_forest_disable_hpo", False)
            )
            _m.hpo_out_dir = _meta_hpo_out_dir
            _mcw = float(cfg.get("meta_parallel_forest_min_child_weight", 40.0))
            _m.xgb_parallel_forest_params = {
                "objective": "binary:logistic",
                "n_estimators": int(cfg.get("meta_parallel_forest_rounds", 100)),
                "num_parallel_tree": int(
                    cfg.get("meta_parallel_forest_num_parallel_tree", 20)
                ),
                "max_depth": int(cfg.get("meta_parallel_forest_max_depth", 5)),
                "learning_rate": float(
                    cfg.get("meta_parallel_forest_learning_rate", 0.05)
                ),
                "subsample": 0.75,
                "colsample_bytree": 0.75,
                "reg_alpha": float(cfg.get("meta_parallel_forest_reg_alpha", 2.0)),
                "reg_lambda": float(cfg.get("meta_parallel_forest_reg_lambda", 15.0)),
                "min_child_weight": max(1.0, float(_mcw)),
                "gamma": float(cfg.get("meta_parallel_forest_gamma", 1.5)),
                "tree_method": "hist",
                "random_state": 42,
                "n_jobs": 3,
                "verbosity": 0,
                "eval_metric": "logloss",
                "early_stopping_rounds": int(
                    cfg.get("meta_parallel_forest_early_stopping_rounds", 20)
                ),
            }
            _m.selector_cfg = dict(cfg.get(_selector_cfg_key, {}) or {})
            _m.selector_report_dir = _fs_dir
            _m.selector_prev_selected = _prev_selected(_head_name)
            _m.selector_family_map = dict(
                cfg.get("selector_feature_family_map", {}) or {}
            )
            _sel_cfg = MetaMoveSelectionConfig(
                min_roc_auc=float(cfg.get("meta_move_min_roc_auc", 0.56)),
                min_pr_auc=float(cfg.get("meta_move_min_pr_auc", 0.0)),
                min_balanced_accuracy=float(cfg.get("meta_move_min_bal_acc", 0.0)),
                min_ic=float(cfg.get("meta_move_min_ic", 0.0)),
                top_frac=float(cfg.get("meta_move_top_frac", 0.10)),
                min_top_n=int(cfg.get("meta_move_min_top_n", 50)),
                min_lift_vs_baseline=float(cfg.get("meta_move_min_lift_vs_baseline", 0.0)),
                require_positive_top_lift=bool(cfg.get("meta_move_require_positive_top_lift", True)),
                require_positive_base_rate=bool(cfg.get("meta_move_require_positive_base_rate", True)),
            )
            return _m, _sel_cfg

        def _align_bucket_vec(_arr, *, fill_value=np.nan, dtype=np.float32):
            _v = np.asarray(_arr, dtype=dtype)
            if len(_v) == _bucket_n:
                return _v
            _out = np.full(_bucket_n, fill_value, dtype=dtype)
            _m = min(_bucket_n, len(_v))
            if _m > 0:
                _out[:_m] = _v[:_m]
            return _out

        def _native_excursion_cols(_h: int):
            _mfe_col = f"__meta_raw__mfe_{int(_h)}h"
            _mae_col = f"__meta_raw__mae_{int(_h)}h"
            _t_mfe_col = f"__meta_raw__t_mfe_{int(_h)}h"
            _t_mae_col = f"__meta_raw__t_mae_{int(_h)}h"

            if (
                _mfe_col in df.columns
                and _mae_col in df.columns
                and _t_mfe_col in df.columns
                and _t_mae_col in df.columns
            ):
                return (
                    np.asarray(df[_mfe_col].values, dtype=np.float32),
                    np.asarray(df[_mae_col].values, dtype=np.float32),
                    np.asarray(df[_t_mfe_col].values, dtype=np.float32),
                    np.asarray(df[_t_mae_col].values, dtype=np.float32),
                )
            if (
                _native_bucket_h is not None
                and int(_h) == _native_bucket_h
                and "__mfe_ret__" in df.columns
                and "__mae_ret__" in df.columns
            ):
                _t_mfe = (
                    np.asarray(df["__t_mfe__"].values, dtype=np.float32)
                    if "__t_mfe__" in df.columns
                    else None
                )
                _t_mae = (
                    np.asarray(df["__t_mae__"].values, dtype=np.float32)
                    if "__t_mae__" in df.columns
                    else None
                )
                return (
                    np.asarray(df["__mfe_ret__"].values, dtype=np.float32),
                    np.asarray(df["__mae_ret__"].values, dtype=np.float32),
                    _t_mfe,
                    _t_mae,
                )
            return None

        # Geometry-specific multiclass classifiers are disabled.
        # We now train a single binary move head below so downstream gating
        # uses a stable p_move signal instead of TP/TIME/SL probabilities.

        _bp = (
            np.clip(
                np.asarray(df["__barrier_pct__"].values, dtype=np.float32), 1e-6, None
            )
            if "__barrier_pct__" in df.columns
            else np.full(len(df), 0.02, dtype=np.float32)
        )
        _mae_horizons = [
            int(h)
            for h in cfg.get("meta_map_mae_horizons", [2, 4])
            if int(h) in _bucket_horizons
        ] or list(_bucket_horizons)
        for _h in _mae_horizons:
            _exc = _native_excursion_cols(int(_h))
            if _exc is None:
                continue
            _target_src = _exc[1]
            _target = np.log1p(
                np.clip(np.asarray(_target_src, dtype=np.float32) / _bp, 0.0, None)
            )
            _weights = _meta_map_weights(_target, df, trade_mask)
            _head_name = f"{_bucket_key}_mae_h{int(_h)}"
            _model = _configure_meta_reg(_head_name, "aux_mae_selector_cfg")
            _model.fit(
                X_meta_base,
                _target,
                sample_weight=_weights,
                groups=meta_groups,
                y_per_horizon=None,
            )
            meta_models[_head_name] = _model
            _bucket_y_ret[_head_name] = np.asarray(_target, dtype=float)
            _bucket_metadata[_head_name] = _md
            tprint(f"Meta {_head_name}: fitted aligned MAE head")

        _mfe_horizons = [
            int(h)
            for h in cfg.get("meta_map_mfe_horizons", [2, 4])
            if int(h) in _bucket_horizons
        ] or list(_bucket_horizons)
        for _h in _mfe_horizons:
            _exc = _native_excursion_cols(int(_h))
            if _exc is None:
                continue
            _target_src = _exc[0]
            _target = np.log1p(
                np.clip(np.asarray(_target_src, dtype=np.float32) / _bp, 0.0, None)
            )
            _weights = _meta_map_weights(_target, df, trade_mask)
            _head_name = f"{_bucket_key}_mfe_h{int(_h)}"
            _model = _configure_meta_reg(_head_name, "aux_mfe_selector_cfg")
            _model.fit(
                X_meta_base,
                _target,
                sample_weight=_weights,
                groups=meta_groups,
                y_per_horizon=None,
            )
            meta_models[_head_name] = _model
            _bucket_y_ret[_head_name] = np.asarray(_target, dtype=float)
            _bucket_metadata[_head_name] = _md
            tprint(f"Meta {_head_name}: fitted aligned MFE head")

        _asym_horizons = [
            int(h)
            for h in cfg.get("meta_map_mfe_horizons", [2, 4])
            if int(h) in _bucket_horizons
        ] or list(_bucket_horizons)
        for _h in _asym_horizons:
            _exc = _native_excursion_cols(int(_h))
            if _exc is None:
                continue
            _mfe_src, _mae_src = _exc[0], _exc[1]
            _bp_asym = (
                np.clip(
                    np.asarray(df["__barrier_pct__"].values, dtype=np.float32),
                    1e-6,
                    None,
                )
                if "__barrier_pct__" in df.columns
                else np.full(len(df), 0.02, dtype=np.float32)
            )
            _eps_asym = np.float32(1e-6)
            _mfe_norm = np.maximum(
                np.asarray(_mfe_src, dtype=np.float32) / _bp_asym, 0.0
            )
            _mae_norm = np.maximum(
                np.asarray(_mae_src, dtype=np.float32) / _bp_asym, 0.0
            )
            _target_asym = (
                np.log(_mfe_norm + _eps_asym) - np.log(_mae_norm + _eps_asym)
            ).astype(np.float32)
            _target_asym = np.where(
                np.isfinite(_target_asym), _target_asym, 0.0
            ).astype(np.float32)
            _weights_asym = _meta_map_weights(_target_asym, df, trade_mask)
            _head_name_asym = f"{_bucket_key}_asym_h{int(_h)}"
            _model_asym = _configure_meta_reg(_head_name_asym, "aux_asym_selector_cfg")
            _model_asym.fit(
                X_meta_base,
                _target_asym,
                sample_weight=_weights_asym,
                groups=meta_groups,
                y_per_horizon=None,
            )
            meta_models[_head_name_asym] = _model_asym
            _bucket_y_ret[_head_name_asym] = np.asarray(_target_asym, dtype=float)
            _bucket_metadata[_head_name_asym] = _md
            tprint(f"Meta {_head_name_asym}: fitted aligned asym head")

        if include_meta_clf:
            _mid_h = (
                _bucket_horizons[len(_bucket_horizons) // 2] if _bucket_horizons else 4
            )
            _y_mid = _align_bucket_vec(
                ret_for_h(int(_mid_h)), fill_value=np.nan, dtype=np.float64
            )
            _weights_clf = _meta_map_weights(_y_mid, df, trade_mask)
            _move_thresholds = tuple(
                float(x) for x in cfg.get("meta_clf_move_thresholds", [1.0, 1.25, 1.5])
            )
            _move_weights = tuple(
                float(x) for x in cfg.get("meta_clf_move_weights", [0.45, 0.35, 0.20])
            )
            _bp_move = _bp if "_bp" in locals() else np.asarray(
                df["__barrier_pct__"].values, dtype=np.float64
            ) if "__barrier_pct__" in df.columns else np.full(len(df), 0.02, dtype=np.float64)
            _y_move_soft, _y_move, _move_thr = _build_meta_move_soft_target(
                abs_ret=np.abs(_y_mid),
                vol_proxy=_bp_move,
                thresholds=_move_thresholds,
                weights=_move_weights,
            )
            tprint(
                f"  Meta move target (aligned map): H={_mid_h} "
                f"thresholds={list(_move_thresholds)} weights={list(_move_weights)} "
                f"base_rate={float(np.mean(_y_move)):.4f} soft_mean={float(np.mean(_y_move_soft)):.4f}"
            )
            _clf = MetaClassifierModel(reports_dir=reports_dir)
            _clf.strategy_name = _bucket_key
            _clf.candidate_mode = "xgb_parallel_forest"
            _clf.xgb_parallel_forest_params = {
                "objective": "binary:logistic",
                "n_estimators": int(cfg.get("meta_parallel_forest_rounds", 100)),
                "num_parallel_tree": int(
                    cfg.get("meta_parallel_forest_num_parallel_tree", 20)
                ),
                "max_depth": int(cfg.get("meta_parallel_forest_max_depth", 5)),
                "learning_rate": float(
                    cfg.get("meta_parallel_forest_learning_rate", 0.05)
                ),
                "subsample": 0.75,
                "colsample_bytree": 0.75,
                "reg_alpha": float(cfg.get("meta_parallel_forest_reg_alpha", 2.0)),
                "reg_lambda": float(cfg.get("meta_parallel_forest_reg_lambda", 15.0)),
                "min_child_weight": float(
                    cfg.get("meta_parallel_forest_min_child_weight", 40.0)
                ),
                "gamma": float(cfg.get("meta_parallel_forest_gamma", 1.5)),
                "tree_method": "hist",
                "random_state": 42,
                "n_jobs": 3,
                "verbosity": 0,
                "eval_metric": "logloss",
                "early_stopping_rounds": int(
                    cfg.get("meta_parallel_forest_early_stopping_rounds", 20)
                ),
            }
            _clf.selector_cfg = dict(cfg.get("meta_selector_cfg", {}) or {})
            _clf.selector_report_dir = _fs_dir
            _clf.selector_prev_selected = _prev_selected(_bucket_key)
            _clf.selector_family_map = dict(
                cfg.get("selector_feature_family_map", {}) or {}
            )
            _sel_cfg = MetaMoveSelectionConfig(
                min_roc_auc=float(cfg.get("meta_move_min_roc_auc", 0.56)),
                min_pr_auc=float(cfg.get("meta_move_min_pr_auc", 0.0)),
                min_balanced_accuracy=float(cfg.get("meta_move_min_bal_acc", 0.0)),
                min_ic=float(cfg.get("meta_move_min_ic", 0.0)),
                top_frac=float(cfg.get("meta_move_top_frac", 0.10)),
                min_top_n=int(cfg.get("meta_move_min_top_n", 50)),
                min_lift_vs_baseline=float(cfg.get("meta_move_min_lift_vs_baseline", 0.0)),
                require_positive_top_lift=bool(cfg.get("meta_move_require_positive_top_lift", True)),
                require_positive_base_rate=bool(cfg.get("meta_move_require_positive_base_rate", True)),
            )
            _clf.fit(
                X_meta_base,
                _y_mid,
                sample_weight=_weights_clf,
                groups=meta_groups,
                y_per_horizon={
                    int(h): _align_bucket_vec(
                        ret_for_h(int(h)), fill_value=np.nan, dtype=np.float64
                    )
                    for h in _bucket_horizons
                },
                vol_proxy=_bp_move,
                realized_u_policy=np.abs(_y_mid),
                selection_cfg=_sel_cfg,
                trade_mask=np.asarray(trade_mask, dtype=bool),
                move_thresholds=_move_thresholds,
                move_weights=_move_weights,
                use_class_weight_multiplier=bool(
                    cfg.get("meta_clf_use_class_weight_multiplier", True)
                ),
                max_class_weight=float(cfg.get("meta_clf_max_class_weight", 10.0)),
                use_calibration=bool(cfg.get("meta_clf_use_calibration", True)),
                move_horizon=int(_mid_h),
            )
            meta_models[f"{_bucket_key}_clf"] = _clf
            _bucket_y_ret[f"{_bucket_key}_clf"] = _y_mid.copy()
            _bucket_metadata[f"{_bucket_key}_clf"] = _md
            tprint(f"Meta {_bucket_key}_clf: fitted aligned move head")

    strategies = get_strategies(cfg)
    for strat in strategies:
        side = strat["trade_side"]
        k = strat["strategy_id"]
        trade_side = side
        cand_filter, move_bucket, strategy_label = _strategy_bucket_context(
            trade_side, k, cfg
        )
        tprint(
            f"Meta bucket context: trade_side={trade_side}, kind={k}, "
            f"move_bucket={move_bucket}, candidate_bucket={cand_filter}, strategy={strategy_label}"
        )
        conf = _alpha_conf_for_strategy(side, k)
        if not conf:
            tprint(f"Meta {k}: skipped (missing alpha model)")
            continue

        # Primary horizon OOF set (this bucket's own alpha models)
        horizon_dfs, horizon_skip_reasons = _collect_horizon_oof(side, k)
        for h, reason in horizon_skip_reasons.items():
            if reason.startswith("oof_len_mismatch"):
                ds_key_dbg = f"train_{k}_{h}"
                _df_len = (
                    len(datasets.get(ds_key_dbg, []))
                    if ds_key_dbg in datasets
                    else "na"
                )
                _oof_len = (
                    reason.split(":")[-1].split("!=")[0] if ":" in reason else "na"
                )
                tprint(
                    f"Meta {k} H={h}: OOF length mismatch ({_oof_len} vs {_df_len}), skipping horizon"
                )

        if not horizon_dfs:
            tprint(f"Meta {k}: skipped (no horizon with valid OOF)")
            if horizon_skip_reasons:
                tprint(f"  Horizon availability diagnostics: {horizon_skip_reasons}")
            continue
        tprint(
            f"Meta {k}: {len(horizon_dfs)} horizons available: {sorted(horizon_dfs.keys())} ({_time.monotonic()-_t0_meta:.1f}s)"
        )
        _available_horizons = sorted(horizon_dfs.keys())
        if horizon_skip_reasons:
            tprint(f"  Horizons excluded: {horizon_skip_reasons}")

        # Use the largest horizon dataset as the base (for meta feature columns)
        base_H = max(horizon_dfs.keys(), key=lambda h: len(horizon_dfs[h][0]))
        df_base = horizon_dfs[base_H][0]

        # Build (ts, symbol) index for union + diagnostics
        tprint(f"  Building union key index ({len(df_base)} base rows)...")
        if "__ts__" in df_base.columns and "__symbol__" in df_base.columns:
            base_keys = _ts_symbol_string_keys(
                df_base["__ts__"].values, df_base["__symbol__"].values
            )
            union_keys_parts = [_unique_preserve_order(base_keys)]
            key_rows_by_h = {}
            key_set_by_h = {}

            for h in sorted(horizon_dfs.keys()):
                df_h, _ = horizon_dfs[h]
                ts_vals = df_h["__ts__"]
                sym_vals = df_h["__symbol__"]
                null_key_rows = int((ts_vals.isna() | sym_vals.isna()).sum())
                h_keys = _ts_symbol_string_keys(ts_vals.values, sym_vals.values)
                uniq_h_keys = np.unique(h_keys)
                uniq_keys = len(uniq_h_keys)
                dup_keys = len(h_keys) - uniq_keys
                if null_key_rows > 0 or dup_keys > 0:
                    tprint(
                        f"    H{h} key hygiene: rows={len(df_h)}, unique={uniq_keys}, duplicates={dup_keys}, null_keys={null_key_rows}"
                    )
                key_rows_by_h[h] = h_keys
                key_set_by_h[h] = uniq_h_keys
                union_keys_parts.append(_unique_preserve_order(h_keys))

            union_keys = _unique_preserve_order(np.concatenate(union_keys_parts))
            inter_mask = np.ones(len(union_keys), dtype=bool)
            for h in sorted(horizon_dfs.keys()):
                inter_mask &= np.isin(union_keys, key_set_by_h[h], assume_unique=False)
            inter_count = int(inter_mask.sum())
            tprint(
                f"  Union: {len(union_keys)} unique samples across horizons (intersection={inter_count})"
            )
            for h in sorted(horizon_dfs.keys()):
                present = np.isin(union_keys, key_set_by_h[h], assume_unique=False)
                miss_cnt = int((~present).sum())
                coverage = float(present.mean() * 100.0)
                tprint(
                    f"    H{h} coverage on union: {present.sum()}/{len(union_keys)} ({coverage:.1f}%), missing={miss_cnt}"
                )

            base_lookup = _indexer_by_string_keys(union_keys, base_keys)
            base_key_set = np.unique(base_keys)
            for h in sorted(horizon_dfs.keys()):
                if h == base_H:
                    continue
                h_key_set = key_set_by_h[h]
                base_only = int(np.setdiff1d(base_key_set, h_key_set).size)
                h_only = int(np.setdiff1d(h_key_set, base_key_set).size)
                if base_only > 0 or h_only > 0:
                    tprint(
                        f"    H{h} vs base H{base_H}: base_only={base_only}, h_only={h_only}"
                    )

            df = df_base.iloc[base_lookup[base_lookup >= 0]].copy()

            missing_mask = base_lookup < 0
            if missing_mask.any():
                missing_keys = union_keys[missing_mask]
                donor_horizons = sorted(
                    horizon_dfs.keys(),
                    key=lambda hh: (-len(horizon_dfs[hh][0]), hh),
                )
                extra_rows = []
                unresolved = np.asarray(missing_keys)
                for h in donor_horizons:
                    if len(unresolved) == 0:
                        break
                    h_idx = _indexer_by_string_keys(unresolved, key_rows_by_h[h])
                    found = h_idx >= 0
                    if found.any():
                        for row_idx in h_idx[found]:
                            extra_rows.append(
                                horizon_dfs[h][0]
                                .iloc[int(row_idx)]
                                .reindex(df_base.columns)
                            )
                        unresolved = unresolved[~found]
                if extra_rows:
                    df_extra = pd.DataFrame(extra_rows, columns=df_base.columns)
                    df = pd.concat([df, df_extra], axis=0, ignore_index=True)

            df = df.reset_index(drop=True)
            tprint(
                f"  Union dataset built: {len(df)} rows ({_time.monotonic()-_t0_meta:.1f}s)"
            )
        else:
            # Fallback: use min-length truncation
            min_len = min(len(df_h) for df_h, _ in horizon_dfs.values())
            keep_mask = np.ones(len(df_base), dtype=bool)
            keep_mask[min_len:] = False
            df = df_base.loc[keep_mask].reset_index(drop=True).copy()
            tprint(f"  No ts/symbol columns; truncating to {min_len} samples")

        if len(df) < 100:
            tprint(f"Meta {k}: skipped (only {len(df)} union samples)")
            continue

        _meta_cap = int(cfg.get("meta_fit_max_samples", 50000))
        if _meta_cap > 0 and len(df) > _meta_cap:
            _pre_cap = len(df)
            df = subsample_symbol_balanced(df, _meta_cap)
            tprint(
                f"  Meta {k}: symbol-balanced subsample {_pre_cap} -> {len(df)} rows (cap={_meta_cap})"
            )

        # Build OOF predictions for each horizon, aligned to common samples
        tprint(f"  Aligning OOF predictions across horizons...")
        pred_h = pd.DataFrame(index=df.index)
        pred_h_diag = pd.DataFrame(index=df.index)
        p_oof_avg_parts = []
        for h in sorted(horizon_dfs.keys()):
            df_h, oof_h = horizon_dfs[h]
            p_h = _align_oof_to_union(df, df_h, oof_h)
            p_h_diag = _align_values_by_ts_symbol_keys(
                df["__ts__"].values,
                df["__symbol__"].values,
                df_h["__ts__"].values,
                df_h["__symbol__"].values,
                oof_h,
                fill_value=np.nan,
                dtype=np.float32,
            )
            pred_h[f"pred_{k}_H{h}"] = p_h
            pred_h[f"pred_H{h}"] = p_h
            pred_h_diag[f"pred_H{h}"] = p_h_diag
            p_oof_avg_parts.append(p_h)

        tprint(
            f"  Meta {k}: own-strategy OOF features only "
            f"(cols={len([c for c in pred_h.columns if c.startswith('pred_')])})"
        )

        y_ret = df["__y_ret__"].values

        tprint(
            f"  OOF alignment done ({_time.monotonic()-_t0_meta:.1f}s). Building meta features..."
        )

        # Build y_target from per-horizon returns (aligned to common samples)
        _has_keys = "__ts__" in df.columns and "__symbol__" in df.columns
        _h_lookup_cache = {}

        def _ret_for_h_aligned(h):
            ds_key = f"train_{k}_{h}"
            for source_df in [datasets.get(ds_key), horizon_dfs.get(h, (None,))[0]]:
                if source_df is None:
                    continue
                if not (
                    _has_keys
                    and "__ts__" in source_df.columns
                    and "__symbol__" in source_df.columns
                ):
                    continue
                cache_id = id(source_df)
                if cache_id not in _h_lookup_cache:
                    _h_lookup_cache[cache_id] = _align_values_by_ts_symbol_keys(
                        df["__ts__"].values,
                        df["__symbol__"].values,
                        source_df["__ts__"].values,
                        source_df["__symbol__"].values,
                        source_df["__y_ret__"].values,
                        fill_value=np.nan,
                        dtype=np.float32,
                    )
                _ret_vals = _h_lookup_cache[cache_id]
                valid = np.isfinite(_ret_vals)
                if not valid.any():
                    continue
                ret = np.zeros(len(df), dtype=np.float32)
                ret[valid] = _ret_vals[valid]
                return ret
            return y_ret.astype(np.float32)

        tprint(f"  Computing meta target ({_time.monotonic()-_t0_meta:.1f}s)...")
        _use_policy_target = bool(cfg.get("meta_use_policy_value_target", True))

        # Use vol_proxy (ATR) for risk-normalized target / classifier fallback labels
        _vol_proxy = (
            df["__barrier_pct__"].values.astype(np.float64)
            if "__barrier_pct__" in df.columns
            else None
        )

        if _use_policy_target:
            if "__u_policy_net__" not in df.columns:
                tprint(
                    "  META TARGET: '__u_policy_net__' missing; falling back to return-derived utility target."
                )
                _use_policy_target = False
        if _use_policy_target:
            y_target_h = np.asarray(df["__u_policy_net__"].values, dtype=np.float32)
            tprint(
                f"  META TARGET: policy_value(u_policy) n={len(y_target_h)} "
                f"mean={float(np.mean(y_target_h)):.6f} std={float(np.std(y_target_h)):.6f}"
            )
            # Keep horizon keys for per-horizon model slots, but target is true policy utility.
            _y_per_h = {h: y_target_h.copy() for h in sorted(horizon_dfs.keys())}
        else:
            _rets = [_ret_for_h_aligned(int(h)) for h in CANON_HORIZONS]
            if len(_rets) >= 3:
                y_target_h = compute_meta_target(*_rets[:3], vol_proxy=_vol_proxy)
                tprint(
                    f"  Using risk-normalized target: n={len(y_target_h)}, mean={float(np.mean(y_target_h)):.6f}, std={float(np.std(y_target_h)):.6f}"
                )
            else:
                y_target_h = np.asarray(
                    df["__y_ret__"].values if "__y_ret__" in df.columns else _rets[0],
                    dtype=np.float32,
                )
                tprint(
                    f"  Using direct return target fallback: n={len(y_target_h)}, mean={float(np.mean(y_target_h)):.6f}, std={float(np.std(y_target_h)):.6f}"
                )
            # Per-horizon returns for multi-barrier classifier labels
            _y_per_h = {int(h): r for h, r in zip(CANON_HORIZONS, _rets)}

        # Per-horizon IC diagnostics
        _oof_by_h = {}
        for h in sorted(horizon_dfs.keys()):
            _oof_by_h[h] = pred_h_diag[f"pred_H{h}"].values
        for _hh in sorted(_y_per_h.keys()):
            if _hh not in _oof_by_h:
                continue
            _ic_h = _safe_spearman(_oof_by_h[_hh], _y_per_h[_hh])
            tprint(f"    IC(oof_H{_hh}, r_H{_hh}) = {_ic_h:.4f}")

        # p_oof: average OOF across all horizons (used for filtering & diagnostics)
        p_oof = np.mean(p_oof_avg_parts, axis=0)

        # Vol proxy for normalization variants (barrier_pct if available)
        _vol_proxy = (
            df["__barrier_pct__"].values.astype(np.float64)
            if "__barrier_pct__" in df.columns
            else None
        )

        configured_meta = _meta_feature_keys_for_kind(cfg, strat, kind="clf")
        raw_prefix = "__meta_raw__"
        feat_cols = [mk for mk in configured_meta if f"{raw_prefix}{mk}" in df.columns]
        feat_cols = list(dict.fromkeys(feat_cols))
        missing_meta_keys = [
            mk for mk in configured_meta if f"{raw_prefix}{mk}" not in df.columns
        ]
        tprint(
            f"  Meta {k}: resolved raw meta keys {len(feat_cols)}/{len(configured_meta)} "
            f"(missing={len(missing_meta_keys)}); raw source is limited to configured meta keys"
        )
        if missing_meta_keys:
            tprint(
                f"    Meta {k}: missing raw meta keys (first 20): {missing_meta_keys[:20]}"
            )
        exclude_key = f"meta_feature_exclude_{k}"
        exclude_set = set(cfg.get(exclude_key, []))
        if exclude_set:
            n_before = len(feat_cols)
            feat_cols = [f for f in feat_cols if f not in exclude_set]
            tprint(
                f"  Meta feature exclusion ({k}): {n_before} -> {len(feat_cols)} features ({n_before - len(feat_cols)} excluded)"
            )
        if not feat_cols:
            tprint(f"Meta {k}: skipped (no raw meta features found in dataset)")
            continue

        # Bulk initialize to avoid fragmentation PerformanceWarnings
        _feat_dict = {}
        for mk in feat_cols:
            _col_val = df[f"{raw_prefix}{mk}"]
            if isinstance(_col_val, pd.DataFrame):
                # Handle unexpected duplicate column names by taking the first one
                _feat_dict[mk] = _col_val.iloc[:, 0].values
            else:
                _feat_dict[mk] = _col_val.values

        X_feats = (
            pd.DataFrame(_feat_dict, index=df.index).fillna(0.0).astype(np.float32)
        )

        n_res = df.get(
            "__n_res__", pd.Series(np.ones(len(df)), index=df.index)
        ).values.astype(np.float32)
        _trade_mask = np.ones(len(df), dtype=bool)
        if "__trigger_offset_h__" in df.columns:
            _trade_mask = np.abs(
                np.asarray(df["__trigger_offset_h__"].values, dtype=float)
            ) <= float(cfg.get("trade_mask_abs_hours", 4.0))

        # Build per-horizon logit features in bulk
        from scipy.special import expit as _sigmoid
        from scipy.special import logit as _logit_fn

        _logit_data = {}
        _logit_parts = []
        for h in sorted(horizon_dfs.keys()):
            _p_h = np.clip(pred_h[f"pred_H{h}"].values.astype(float), 1e-4, 1 - 1e-4)
            _lg_h = np.clip(_logit_fn(_p_h), -4.0, 4.0).astype(np.float32)
            _logit_data[f"pred_logit_H{h}"] = _lg_h
            _logit_parts.append(_lg_h)

        _logit_avg = np.mean(_logit_parts, axis=0).astype(np.float32)
        _logit_data["pred_logit"] = _logit_avg

        # Combine everything in one shot
        X_feats = pd.concat(
            [X_feats, pd.DataFrame(_logit_data, index=X_feats.index), pred_h],
            axis=1,
        )

        variant_feat_df, variant_feat_cols = _collect_wide_tight_variant_features(
            side, k, df, horizon_dfs
        )
        if not variant_feat_df.empty:
            X_feats = pd.concat([X_feats, variant_feat_df], axis=1)

        # Disagreement features over this strategy's own horizon OOFs only.
        _diag_feats = {}
        horizon_preds = []
        own_horizons_for_diag = []
        for h in CANON_HORIZONS:
            col = f"pred_{k}_H{int(h)}"
            if col in pred_h.columns:
                horizon_preds.append(pred_h[col].values.astype(np.float32))
                own_horizons_for_diag.append(int(h))
        if horizon_preds:
            stack = np.vstack(horizon_preds).T.astype(np.float32)
            pair_terms = []
            for i in range(len(horizon_preds)):
                for j in range(i + 1, len(horizon_preds)):
                    pair_terms.append(np.abs(horizon_preds[i] - horizon_preds[j]))
            pair_abs = (
                np.mean(np.vstack(pair_terms), axis=0)
                if pair_terms
                else np.zeros(len(stack), dtype=np.float32)
            )
            vote_p = (stack > 0.5).mean(axis=1).astype(np.float32)
            _diag_feats["disagree_self_std"] = np.std(
                stack, axis=1, dtype=np.float32
            ).astype(np.float32)
            _diag_feats["disagree_self_range"] = (
                np.max(stack, axis=1) - np.min(stack, axis=1)
            ).astype(np.float32)
            _diag_feats["disagree_self_pair_abs"] = pair_abs.astype(np.float32)
            _diag_feats["disagree_self_vote_mix"] = (
                4.0 * vote_p * (1.0 - vote_p)
            ).astype(np.float32)
            _diag_feats["agree_self_avg"] = (1.0 - np.clip(pair_abs, 0.0, 1.0)).astype(
                np.float32
            )
            for i in range(len(horizon_preds)):
                for j in range(i + 1, len(horizon_preds)):
                    hi = own_horizons_for_diag[i]
                    hj = own_horizons_for_diag[j]
                    _diag_feats[f"pred_H{hi}_minus_H{hj}"] = (
                        horizon_preds[i] - horizon_preds[j]
                    ).astype(np.float32)

        tprint(
            f"  Meta {k}: engineered additions pred_cols={len(_logit_data)} "
            f"diag_cols={len(_diag_feats)} total_meta_base_before_context="
            f"{len(X_feats.columns) + len(_diag_feats)}"
        )

        X_feats = pd.concat(
            [X_feats, pd.DataFrame(_diag_feats, index=X_feats.index)], axis=1
        )

        X_meta_base = X_feats.fillna(0.0)

        # pred_logit interaction terms removed by design.
        # Root-cause analysis showed they can amplify incorrect MAE directionality.
        pred_logit = _logit_avg

        # Cross-timeframe context features
        if (
            "trend_slope_48h" in X_meta_base.columns
            and "trend_slope_120h" in X_meta_base.columns
        ):
            _ts48 = X_meta_base["trend_slope_48h"].values
            _ts120 = X_meta_base["trend_slope_120h"].values
            X_meta_base["trend_slope_ratio_48_120"] = np.where(
                np.abs(_ts120) > 1e-9,
                _ts48 / np.clip(np.abs(_ts120), 1e-9, None),
                0.0,
            ).astype(np.float32)
        if "__regime_vol_12h__" in df.columns and "__regime_vol_48h__" in df.columns:
            _v12 = df["__regime_vol_12h__"].values
            _v48 = df["__regime_vol_48h__"].values
            X_meta_base["vol_regime_agree"] = (_v12 == _v48).astype(np.float32)
            X_meta_base["vol_regime_diff"] = (_v12 - _v48).astype(np.float32)
        if (
            "__regime_trend_12h__" in df.columns
            and "__regime_trend_48h__" in df.columns
        ):
            _t12 = df["__regime_trend_12h__"].values
            _t48 = df["__regime_trend_48h__"].values
            X_meta_base["trend_regime_agree"] = (_t12 == _t48).astype(np.float32)
            X_meta_base["trend_regime_diff"] = (_t12 - _t48).astype(np.float32)

        meta_model_cols = []
        meta_model_cols.extend(
            [c for c in pred_h.columns if c in X_meta_base.columns and c.startswith(f"pred_{k}_H")]
        )
        meta_model_cols.extend([c for c in variant_feat_cols if c in X_meta_base.columns])
        meta_model_cols.extend([c for c in feat_cols if c in X_meta_base.columns])
        meta_model_cols = list(dict.fromkeys(meta_model_cols))
        if not meta_model_cols:
            raise RuntimeError(
                f"Meta {k}: no configured meta keys survived into the model input frame"
            )
        meta_extras = [c for c in X_meta_base.columns if c not in meta_model_cols]
        tprint(
            f"  Meta {k}: model input restricted to {len(meta_model_cols)} configured meta keys "
            f"(dropped extras={len(meta_extras)})"
        )
        if meta_extras:
            tprint(f"    Meta {k}: dropped non-meta extras (first 20): {meta_extras[:20]}")
        X_meta_models = X_meta_base.loc[:, meta_model_cols].copy()

        meta_groups = df["__ts__"].values if "__ts__" in df.columns else None

        _ran_aligned_map_v2 = False
        if _use_aligned_map_v2:
            tprint(
                f"  Using aligned meta map v2 for {k}: "
                f"TBM/MAE/MFE map heads + classifier"
            )
            _fit_aligned_meta_map_heads(
                side=side,
                k=k,
                df=df,
                X_meta_base=X_meta_models,
                meta_groups=meta_groups,
                trade_mask=_trade_mask,
                ret_for_h=_ret_for_h_aligned,
                bucket_horizons=_available_horizons,
            )
            _ran_aligned_map_v2 = True

        # ══════════════════════════════════════════════════════════════
        # MAIN META REGRESSOR (bucket-level)
        # ══════════════════════════════════════════════════════════════
        # Pick a representative horizon from CANON_HORIZONS (prefer largest available)
        _h_main = (
            sorted([h for h in CANON_HORIZONS if h in _y_per_h])[-1]
            if any(h in _y_per_h for h in CANON_HORIZONS)
            else sorted(_y_per_h.keys())[-1]
        )
        _h_label = f"{k}_reg"
        y_ret_raw_main = y_target_h.astype(np.float64)

        # Guard: replace any inf/nan with 0 so downstream sklearn/optuna don't choke
        _y_finite_mask = np.isfinite(y_ret_raw_main)
        if not _y_finite_mask.all():
            _y_fill = (
                float(np.nanmedian(y_ret_raw_main[_y_finite_mask]))
                if _y_finite_mask.any()
                else 0.0
            )
            y_ret_raw_main = np.where(_y_finite_mask, y_ret_raw_main, _y_fill)

        # Sample weights: magnitude sigmoid (very slight top-40% upweight) + MFE/MAE quality
        # Use main horizon for quality indicators
        _alpha_w = float(cfg.get("meta_weight_sigmoid_alpha", 0.2))
        _y_abs = np.abs(y_ret_raw_main)
        _fin_w = np.isfinite(_y_abs)
        _q60 = float(np.percentile(_y_abs[_fin_w], 60))
        _s_w = max(float(np.std(_y_abs[_fin_w])), 1e-9)
        # sigmoid centered at p60: top-40% get ~1.1-1.2x, bottom-60% get ~1.0x
        w_mag = 1.0 + _alpha_w * _sigmoid((_y_abs - _q60) / _s_w)
        w_mag = w_mag / max(float(np.mean(w_mag)), 1e-12)  # normalize to mean=1

        _mfe_col = f"__meta_raw__mfe_{_h_main}h"
        _mae_col = f"__meta_raw__mae_{_h_main}h"
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

        _n_res_max = (
            float(np.nanmax(n_res)) if len(n_res) and np.nanmax(n_res) > 0 else 1.0
        )
        w_n_res = 0.5 + 1.5 * np.sqrt(
            np.clip(n_res.astype(np.float64) / _n_res_max, 0.0, 1.0)
        )
        w_meta_main = (w_mag * w_exc * w_n_res).astype(np.float64)
        w_meta_main = w_meta_main / max(
            float(np.mean(w_meta_main)), 1e-12
        )  # final mean=1
        # Guard n_eff: clip extreme weights so n_eff >= 30% of N
        _n_eff = float(np.sum(w_meta_main) ** 2 / max(np.sum(w_meta_main**2), 1e-12))
        if _n_eff < 0.3 * len(w_meta_main):
            _clip_hi = float(np.percentile(w_meta_main, 95))
            w_meta_main = np.clip(w_meta_main, 0.0, _clip_hi)
            w_meta_main = w_meta_main / max(float(np.mean(w_meta_main)), 1e-12)
            _n_eff_new = float(
                np.sum(w_meta_main) ** 2 / max(np.sum(w_meta_main**2), 1e-12)
            )
            tprint(
                f"    {_h_label} n_eff clipped: {_n_eff:.0f} -> {_n_eff_new:.0f} (N={len(w_meta_main)})"
            )

        if bool(cfg.get("sample_weight_opt_enable", True)) and "__ts__" in df.columns:
            _meta_ts = pd.to_datetime(df["__ts__"])
            _meta_label_times = pd.DataFrame(
                {
                    "t_start": _meta_ts,
                    "t_end": _meta_ts + pd.Timedelta(hours=int(_h_main)),
                }
            )
            _meta_extra = {
                "magnitude": w_mag,
                "excursion": w_exc,
            }
            # vol_cs component removed per user request
            _w_opt = _optimize_training_sample_weights(
                df=pd.DataFrame({"ts": _meta_ts}),
                X_frame=X_meta_models.select_dtypes(include=[np.number]).fillna(0.0),
                y_ret=y_ret_raw_main,
                label_times=_meta_label_times,
                base_weights=w_meta_main,
                cfg={
                    **cfg,
                    "sample_weight_opt_trials": int(
                        cfg.get(
                            "meta_sample_weight_opt_trials",
                            cfg.get("sample_weight_opt_trials", 16),
                        )
                    ),
                },
                stage=f"meta_{_h_label}",
                extra_components=_meta_extra,
            )
            # If optimizer dropped non-finite rows internally, expand back to full union size
            _full_n = len(w_meta_main)
            if len(_w_opt) != _full_n:
                _fin_mask = np.isfinite(y_ret_raw_main)
                _w_expanded = np.full(
                    _full_n, float(np.median(_w_opt)), dtype=np.float64
                )
                _w_expanded[_fin_mask] = _w_opt[: int(_fin_mask.sum())]
                w_meta_main = _w_expanded
            else:
                w_meta_main = _w_opt
        w_meta_main = w_meta_main.astype(np.float32)

        # Fit bucket-level regressor only when explicitly enabled (default disabled).
        if include_meta_reg:
            meta_reg = MetaModel(reports_dir=reports_dir)
            meta_reg.strategy_name = _h_label
            meta_reg.candidate_mode = "xgb_parallel_forest"
            meta_reg.disable_hpo = bool(
                cfg.get("meta_parallel_forest_disable_hpo", False)
            )
            meta_reg.hpo_out_dir = _meta_hpo_out_dir
            meta_reg.xgb_parallel_forest_params = {
                "objective": "reg:squarederror",
                "n_estimators": int(cfg.get("meta_parallel_forest_rounds", 100)),
                "num_parallel_tree": int(
                    cfg.get("meta_parallel_forest_num_parallel_tree", 20)
                ),
                "max_depth": int(cfg.get("meta_parallel_forest_max_depth", 5)),
                "learning_rate": float(
                    cfg.get("meta_parallel_forest_learning_rate", 0.05)
                ),
                "subsample": 0.75,
                "colsample_bytree": 0.75,
                "reg_alpha": float(cfg.get("meta_parallel_forest_reg_alpha", 2.0)),
                "reg_lambda": float(cfg.get("meta_parallel_forest_reg_lambda", 15.0)),
                "min_child_weight": float(
                    cfg.get("meta_parallel_forest_min_child_weight", 40.0)
                ),
                "gamma": float(cfg.get("meta_parallel_forest_gamma", 1.5)),
                "tree_method": "hist",
                "random_state": 42,
                "n_jobs": 3,
                "verbosity": 0,
                "early_stopping_rounds": int(
                    cfg.get("meta_parallel_forest_early_stopping_rounds", 20)
                ),
            }
            _meta_sel_cfg = dict(cfg.get("meta_selector_cfg", {}) or {})
            _meta_fs_dir = os.path.join(
                str(cfg.get("data_root", "data")),
                "artifacts",
                str(cfg.get("run_id", "default")),
                "fs_reports",
            )
            _meta_prev_sel = None
            _meta_prev_run = (
                cfg.get("prev_run_id")
                or cfg.get("prior_run_id")
                or cfg.get("warm_start_run_id")
            )
            if _meta_prev_run:
                _meta_prev_path = os.path.join(
                    str(cfg.get("data_root", "data")),
                    "artifacts",
                    str(_meta_prev_run),
                    "fs_reports",
                    f"meta_{_h_label}",
                    "selected_features.json",
                )
                if os.path.exists(_meta_prev_path):
                    try:
                        with open(_meta_prev_path, "r", encoding="utf-8") as _f:
                            _meta_prev_sel = list(
                                (json.load(_f) or {}).get("selected_features", [])
                            )
                    except Exception:
                        _meta_prev_sel = None
            meta_reg.selector_cfg = _meta_sel_cfg
            meta_reg.selector_report_dir = _meta_fs_dir
            meta_reg.selector_prev_selected = _meta_prev_sel
            meta_reg.selector_family_map = dict(
                cfg.get("selector_feature_family_map", {}) or {}
            )
            tprint(
                f"  Fitting MetaModel {_h_label} (n={len(df)}, feats={X_meta_base.shape[1]}) ({_time.monotonic()-_t0_meta:.1f}s)..."
            )
            # REG target: risk-normalized realized return log(1 + ret_h / barrier).
            # Distinct from MFE (upside excursion only) and asym (MFE/MAE ratio):
            # this captures signed directional edge per unit of volatility.
            _ret_reg = _ret_for_h_aligned(int(_h_main))
            _bp_reg = (
                np.clip(df["__barrier_pct__"].values.astype(np.float32), 1e-6, None)
                if "__barrier_pct__" in df.columns
                else np.full(len(df), 0.02, dtype=np.float32)
            )
            _y_reg = np.sign(_ret_reg) * np.log1p(
                np.abs(_ret_reg.astype(np.float32)) / _bp_reg
            )
            _y_reg = np.where(np.isfinite(_y_reg), _y_reg, 0.0).astype(np.float32)
            tprint(
                f"  REG target: log(1+|ret_h{_h_main}|/barrier)*sign "
                f"mean={float(np.mean(_y_reg)):.4f} std={float(np.std(_y_reg)):.4f}"
            )
            meta_reg.fit(
                X_meta_models,
                _y_reg,
                sample_weight=w_meta_main,
                groups=meta_groups,
                y_per_horizon=None,
            )
            meta_models[_h_label] = meta_reg
            _bucket_y_ret[_h_label] = y_ret_raw_main.copy()
            tprint(f"Meta {_h_label}: fitted ({_time.monotonic()-_t0_meta:.1f}s).")

            # Orientation safeguard for MR buckets
            if meta_reg.oof_probs is not None:
                y_ret_filtered = (
                    df["__y_ret__"].values if "__y_ret__" in df.columns else y_target_h
                )
                _mask_eval = np.asarray(_trade_mask, dtype=bool)[: len(y_ret_filtered)]

                def _top_spread(yv, sv, frac=0.10):
                    n = len(yv)
                    if n <= 2:
                        return 0.0
                    ksel = max(1, int(frac * n))
                    it = np.argsort(sv)[-ksel:]
                    ib = np.argsort(sv)[:ksel]
                    return float(np.mean(yv[it]) - np.mean(yv[ib]))

                pred_oof = np.asarray(meta_reg.oof_probs, dtype=float)
                ic_pos = _safe_spearman(
                    pred_oof[_mask_eval], y_ret_filtered[_mask_eval]
                )
                ic_neg = _safe_spearman(
                    (-pred_oof)[_mask_eval], y_ret_filtered[_mask_eval]
                )
                sp_pos = _top_spread(
                    y_ret_filtered[_mask_eval], pred_oof[_mask_eval], frac=0.10
                )
                sp_neg = _top_spread(
                    y_ret_filtered[_mask_eval], (-pred_oof)[_mask_eval], frac=0.10
                )

                meta_reg.score_sign = 1
                is_mr = "mr" in k.lower()
                if is_mr and ((ic_neg > ic_pos + 1e-4) and (sp_neg > sp_pos + 1e-6)):
                    meta_reg.score_sign = -1
                    tprint(
                        f"Meta {_h_label}: orientation flipped (IC {ic_pos:.4f}->{ic_neg:.4f})"
                    )

                pred_for_gate = meta_reg.score_sign * pred_oof
                gate_type = "meta_regression"
                gate_res = compute_stage_gate_metrics(
                    y_target_h[_mask_eval],
                    pred_for_gate[_mask_eval],
                    y_ret_filtered[_mask_eval],
                    model_type=gate_type,
                )
                gate_res["Model"] = _h_label
                gate_res["Model_Type"] = gate_type
                gate_res["Score_Sign"] = int(meta_reg.score_sign)
                gate_res["IC_Pos"] = float(ic_pos)
                gate_res["IC_Neg"] = float(ic_neg)
                gate_res["Spread10_Pos"] = float(sp_pos)
                gate_res["Spread10_Neg"] = float(sp_neg)
                meta_gate_results.append(gate_res)

                # Store metadata for this regressor bucket
                _md = {}
                for _cn in [
                    "timestamp",
                    "symbol",
                    "asset",
                    "__ts__",
                    "__symbol__",
                    "__y_bin__",
                    "__y_ret__",
                    "__u_policy_net__",
                    "__u_policy__",
                    "__y_outcome__",
                    "exit_code",
                    "__mae_ret__",
                    "__mfe_ret__",
                    "__bars_to_mfe__",
                    "__barrier_pct__",
                    "__early_inval__",
                    "__mr_path_penalty__",
                    "__mr_velocity_penalty__",
                ]:
                    if _cn in df.columns:
                        _md[_cn] = df[_cn].values
                for _cn in _ps_regime_cols:
                    if _cn in df.columns:
                        _md[_cn] = df[_cn].values
                _bucket_metadata[_h_label] = _md
        else:
            tprint(
                f"Meta {_h_label}: skipped (meta_train_regression_bucket_model=False)"
            )

        # Keep the shared-fold auxiliary heads running even in aligned_map_v2.
        # Those heads feed _aux_head_oof and the downstream meta_oof parquet
        # exports that the sizer expects. The classifier branch below is
        # skipped separately to avoid retraining and overwriting the aligned
        # classifier that was already fit above.

        # ══════════════════════════════════════════════════════════════
        # CLASSIFIER & AUXILIARY HEAD TRAINING (single per bucket, uses all horizons)
        # ══════════════════════════════════════════════════════════════

        # --- 1. ALWAYS Train Auxiliary Heads (MFE, MAE, Utility) ---
        # These are critical for the position sizer and reporting, even if classifiers are skipped.
        # Shared-fold auxiliary heads for sizing/risk features
        try:
            _u_raw = (
                np.asarray(df["__u_policy_net__"].values, dtype=float)
                if "__u_policy_net__" in df.columns
                else np.full(len(df), np.nan, dtype=float)
            )
            _mae_raw = (
                np.asarray(df["__mae_ret__"].values, dtype=float)
                if "__mae_ret__" in df.columns
                else np.full(len(df), np.nan, dtype=float)
            )
            _mfe_raw = (
                np.asarray(df["__mfe_ret__"].values, dtype=float)
                if "__mfe_ret__" in df.columns
                else np.full(len(df), np.nan, dtype=float)
            )
            _atr_norm = (
                np.asarray(df["__barrier_pct__"].values, dtype=float)
                if "__barrier_pct__" in df.columns
                else np.full(len(df), np.nan, dtype=float)
            )
            _atr_ok = np.isfinite(_atr_norm) & (_atr_norm > 0)

            _y_u = _u_raw

            _targets = build_excursion_targets(_mfe_raw, _mae_raw, _atr_norm)
            _y_mfe = _targets["y_mfe"]
            _y_mae = _targets["y_mae"]
            _y_asym = _targets["y_asym"]

            _aux_results, _aux_meta = _train_aux_heads_shared_folds(
                X_num=X_meta_models.select_dtypes(include=[np.number]).fillna(0.0),
                y_u=_y_u,
                y_mae=_y_mae,
                y_mfe=_y_mfe,
                y_asym=_y_asym,
                trade_mask=_trade_mask,
                timestamps=df["__ts__"].values,
                cv_embargo_bars=int(cfg.get("cv_embargo_bars", 12)),
                bucket_id=k,
                data_root=str(cfg.get("data_root", "data")),
                run_id=str(cfg.get("run_id", "default")),
                hpo_out_dir=_meta_hpo_out_dir,
            )
            _aux_head_oof[k] = _aux_results

            # ELBOW: Elevate aux heads to standard meta models for reporting and OOF persistence
            for _hn, _hm in _aux_meta.items():
                _hkey = f"{side}_{k}_{_hn}"
                meta_models[_hkey] = _hm
                if _hn == "utility":
                    _bucket_y_ret[_hkey] = np.asarray(_y_u, dtype=float)
                elif _hn == "mae_q70":
                    _bucket_y_ret[_hkey] = np.asarray(_y_mae, dtype=float)
                elif _hn == "mfe":
                    _bucket_y_ret[_hkey] = np.asarray(_y_mfe, dtype=float)
                elif _hn == "asym":
                    _bucket_y_ret[_hkey] = np.asarray(_y_asym, dtype=float)
                else:
                    _bucket_y_ret[_hkey] = np.asarray(_y_u, dtype=float)

            # Store metadata for aux heads using standard base names
            _md = {}
            for _cn in [
                "timestamp",
                "symbol",
                "asset",
                "__ts__",
                "__symbol__",
                "__y_bin__",
                "__y_ret__",
                "__u_policy_net__",
                "__u_policy__",
                "__y_outcome__",
                "exit_code",
                "__mae_ret__",
                "__mfe_ret__",
                "__bars_to_mfe__",
                "__barrier_pct__",
                "__early_inval__",
                "__mr_path_penalty__",
                "__mr_velocity_penalty__",
            ]:
                if _cn in df.columns:
                    _md[_cn] = df[_cn].values
            for _cn in _ps_regime_cols:
                if _cn in df.columns:
                    _md[_cn] = df[_cn].values
            for _hn in ["utility", "mae_q70", "mfe", "asym"]:
                _bucket_metadata[f"{k}_{_hn}"] = _md

        except Exception as _e_aux:
            tprint(f"Warning: aux head training failed for {side}_{k}: {_e_aux}")

        # --- 2. Optional Meta Move-Classifier Training ---
        if include_meta_clf and not _ran_aligned_map_v2:
            # Magnitude sigmoid: very slight top-40% upweight (same alpha as regressors)
            # Each source normalized to mean=1 before combining so neither dominates.
            _alpha_clf = float(cfg.get("meta_weight_sigmoid_alpha", 0.2))
            _y_avg_abs = np.mean(
                [np.abs(_y_per_h[h]) for h in _available_horizons], axis=0
            )
            _fin_w = np.isfinite(_y_avg_abs)
            _q60_c = float(np.percentile(_y_avg_abs[_fin_w], 60))
            _s_c = max(float(np.std(_y_avg_abs[_fin_w])), 1e-9)
            w_mag_clf = 1.0 + _alpha_clf * _sigmoid((_y_avg_abs - _q60_c) / _s_c)
            w_mag_clf = w_mag_clf / max(
                float(np.mean(w_mag_clf)), 1e-12
            )  # normalize to mean=1

            # MFE/MAE quality weighting for classifier (average across horizons)
            _bp_clf = (
                df["__barrier_pct__"].values
                if "__barrier_pct__" in df.columns
                else None
            )
            w_exc_clf = np.ones(len(df), dtype=np.float64)
            if _bp_clf is not None:
                _exc_parts = []
                for _hc in _available_horizons:
                    _mfe_col_c = f"__meta_raw__mfe_{_hc}h"
                    _mae_col_c = f"__meta_raw__mae_{_hc}h"
                    if _mfe_col_c in df.columns and _mae_col_c in df.columns:
                        _mfe_vc = np.nan_to_num(df[_mfe_col_c].values, nan=0.0).astype(
                            np.float64
                        )
                        _mae_vc = np.nan_to_num(df[_mae_col_c].values, nan=0.0).astype(
                            np.float64
                        )
                        _bp_vc = np.clip(_bp_clf.astype(np.float64), 1e-6, None)
                        _exc_parts.append(
                            np.maximum(
                                np.abs(_mfe_vc) / _bp_vc, np.abs(_mae_vc) / _bp_vc
                            )
                        )
                if _exc_parts:
                    _d_exc_clf = np.mean(_exc_parts, axis=0)
                    _tau_exc_clf = float(cfg.get("meta_mfe_mae_tau", 1.0))
                    w_exc_clf = 0.5 + 0.5 * np.clip(_d_exc_clf / _tau_exc_clf, 0.0, 1.0)
            w_exc_clf = w_exc_clf / max(
                float(np.mean(w_exc_clf)), 1e-12
            )  # normalize to mean=1

            w_meta_clf = (w_mag_clf * w_exc_clf * w_n_res).astype(np.float64)
            w_meta_clf = w_meta_clf / max(
                float(np.mean(w_meta_clf)), 1e-12
            )  # final mean=1
            # Guard n_eff: clip extreme weights so n_eff >= 30% of N
            _n_eff_clf = float(
                np.sum(w_meta_clf) ** 2 / max(np.sum(w_meta_clf**2), 1e-12)
            )
            if _n_eff_clf < 0.3 * len(w_meta_clf):
                _clip_hi_clf = float(np.percentile(w_meta_clf, 95))
                w_meta_clf = np.clip(w_meta_clf, 0.0, _clip_hi_clf)
                w_meta_clf = w_meta_clf / max(float(np.mean(w_meta_clf)), 1e-12)
                _n_eff_clf_new = float(
                    np.sum(w_meta_clf) ** 2 / max(np.sum(w_meta_clf**2), 1e-12)
                )
                tprint(
                    f"    {k}_clf n_eff clipped: {_n_eff_clf:.0f} -> {_n_eff_clf_new:.0f} (N={len(w_meta_clf)})"
                )

            if (
                bool(cfg.get("sample_weight_opt_enable", True))
                and "__ts__" in df.columns
            ):
                _meta_ts_c = pd.to_datetime(df["__ts__"])
                _mid_h_c = (
                    4
                    if 4 in _available_horizons
                    else _available_horizons[len(_available_horizons) // 2]
                )
                _meta_label_times_c = pd.DataFrame(
                    {
                        "t_start": _meta_ts_c,
                        "t_end": _meta_ts_c + pd.Timedelta(hours=int(_mid_h_c)),
                    }
                )
                w_meta_clf = _optimize_training_sample_weights(
                    df=pd.DataFrame({"ts": _meta_ts_c}),
                    X_frame=X_meta_models.select_dtypes(include=[np.number]).fillna(0.0),
                    y_ret=_y_per_h[_mid_h_c].astype(np.float64),
                    label_times=_meta_label_times_c,
                    base_weights=w_meta_clf,
                    cfg={
                        **cfg,
                        "sample_weight_opt_trials": int(
                            cfg.get(
                                "meta_sample_weight_opt_trials",
                                cfg.get("sample_weight_opt_trials", 16),
                            )
                        ),
                    },
                    stage=f"meta_clf_{k}",
                    extra_components={
                        "magnitude": w_mag_clf,
                        "excursion": w_exc_clf,
                    },
                    strategy=strat,
                )
            w_meta_clf = w_meta_clf.astype(np.float32)

            _mid_h = 4 if 4 in _y_per_h else _available_horizons[len(_available_horizons) // 2]
            _move_vol_proxy = _vol_proxy
            if _move_vol_proxy is None:
                for _cand_col in [
                    "atr_24h",
                    "realized_volatility_24h",
                    "asset_vol_level",
                    "__regime_vol_24h__",
                    "__regime_vol_12h__",
                    "__barrier_pct__",
                ]:
                    if _cand_col in df.columns:
                        _move_vol_proxy = np.asarray(df[_cand_col].values, dtype=np.float64)
                        tprint(f"  Meta classifier vol proxy fallback: {_cand_col}")
                        break
            if _move_vol_proxy is None:
                raise RuntimeError("meta_clf_move requires a causal vol proxy column")

            _move_thresholds = tuple(
                float(x) for x in cfg.get("meta_clf_move_thresholds", [1.0, 1.25, 1.5])
            )
            _move_weights = tuple(
                float(x) for x in cfg.get("meta_clf_move_weights", [0.45, 0.35, 0.20])
            )
            y_target_clf = _y_per_h[_mid_h].astype(np.float64)
            _y_move_soft, _y_move, _move_thr = _build_meta_move_soft_target(
                abs_ret=np.abs(y_target_clf),
                vol_proxy=_move_vol_proxy,
                thresholds=_move_thresholds,
                weights=_move_weights,
            )
            _vol_valid = np.isfinite(_move_vol_proxy) & (_move_vol_proxy > 1e-9)
            tprint(
                f"  Meta move target: H={_mid_h} thresholds={list(_move_thresholds)} "
                f"weights={list(_move_weights)} base_rate={float(np.mean(_y_move)):.4f} "
                f"soft_mean={float(np.mean(_y_move_soft)):.4f} "
                f"vol_valid={int(_vol_valid.sum())}/{len(_vol_valid)}"
            )

            tprint(
                f"  Fitting MetaClassifierModel {side}_{k} ({_time.monotonic()-_t0_meta:.1f}s)..."
            )
            meta_clf = MetaClassifierModel(reports_dir=reports_dir)
            meta_clf.strategy_name = k
            meta_clf.candidate_mode = "xgb_parallel_forest"
            meta_clf.disable_hpo = bool(cfg.get("meta_parallel_forest_disable_hpo", False))
            meta_clf.hpo_out_dir = _meta_hpo_out_dir
            meta_clf.xgb_parallel_forest_params = {
                "objective": "binary:logistic",
                "n_estimators": int(cfg.get("meta_parallel_forest_rounds", 100)),
                "num_parallel_tree": int(cfg.get("meta_parallel_forest_num_parallel_tree", 20)),
                "max_depth": int(cfg.get("meta_parallel_forest_max_depth", 5)),
                "learning_rate": float(cfg.get("meta_parallel_forest_learning_rate", 0.05)),
                "subsample": 0.75,
                "colsample_bytree": 0.75,
                "reg_alpha": float(cfg.get("meta_parallel_forest_reg_alpha", 2.0)),
                "reg_lambda": float(cfg.get("meta_parallel_forest_reg_lambda", 15.0)),
                "min_child_weight": float(cfg.get("meta_parallel_forest_min_child_weight", 40.0)),
                "gamma": float(cfg.get("meta_parallel_forest_gamma", 1.5)),
                "tree_method": "hist",
                "random_state": 42,
                "n_jobs": 3,
                "verbosity": 0,
                "eval_metric": "logloss",
                "early_stopping_rounds": int(cfg.get("meta_parallel_forest_early_stopping_rounds", 20)),
            }
            meta_clf.FEE_PER_ROUND_TRIP = float(cfg.get("label_round_trip_fee_pct", 0.3)) / 100.0
            _sel_cfg = MetaMoveSelectionConfig(
                min_roc_auc=float(cfg.get("meta_move_min_roc_auc", 0.56)),
                min_pr_auc=float(cfg.get("meta_move_min_pr_auc", 0.0)),
                min_balanced_accuracy=float(cfg.get("meta_move_min_bal_acc", 0.0)),
                min_ic=float(cfg.get("meta_move_min_ic", 0.0)),
                top_frac=float(cfg.get("meta_move_top_frac", 0.10)),
                min_top_n=int(cfg.get("meta_move_min_top_n", 50)),
                min_lift_vs_baseline=float(cfg.get("meta_move_min_lift_vs_baseline", 0.0)),
                require_positive_top_lift=bool(cfg.get("meta_move_require_positive_top_lift", True)),
                require_positive_base_rate=bool(cfg.get("meta_move_require_positive_base_rate", True)),
            )

            meta_clf.fit(
                X_meta_models,
                y_target_clf,
                sample_weight=w_meta_clf,
                groups=meta_groups,
                y_per_horizon=_y_per_h,
                vol_proxy=_move_vol_proxy,
                realized_u_policy=np.abs(y_target_clf),
                selection_cfg=_sel_cfg,
                trade_mask=_trade_mask,
                move_thresholds=_move_thresholds,
                move_weights=_move_weights,
                use_class_weight_multiplier=bool(cfg.get("meta_clf_use_class_weight_multiplier", True)),
                max_class_weight=float(cfg.get("meta_clf_max_class_weight", 10.0)),
                use_calibration=bool(cfg.get("meta_clf_use_calibration", True)),
                move_horizon=int(_mid_h),
            )
            meta_models[f"{k}_clf"] = meta_clf
            _bucket_y_ret[f"{k}_clf"] = y_target_clf.copy()

            _md = {}
            for _cn in [
                "timestamp",
                "symbol",
                "asset",
                "__ts__",
                "__symbol__",
                "__y_bin__",
                "__y_ret__",
                "__u_policy_net__",
                "__u_policy__",
                "exit_code",
                "__mae_ret__",
                "__mfe_ret__",
                "__bars_to_mfe__",
                "__barrier_pct__",
                "__early_inval__",
                "__mr_path_penalty__",
                "__mr_velocity_penalty__",
            ]:
                if _cn in df.columns:
                    _md[_cn] = df[_cn].values
            for _cn in _ps_regime_cols:
                if _cn in df.columns:
                    _md[_cn] = df[_cn].values
            _bucket_metadata[f"{k}_clf"] = _md

            tprint(f"Meta {k}_clf: fitted ({_time.monotonic()-_t0_meta:.1f}s).")
        else:
            tprint(f"Meta {k}_clf: skipped (meta_clf_enabled=False)")

        # --- 3. ALWAYS Train Early Invalidation head for downstream consumers ---
        if "__early_inval__" in df.columns:
            try:
                from sklearn.linear_model import LogisticRegression

                from extreme_price_movements.periods_symbols_management import (
                    EventSchema,
                    SlicePlanner,
                    SlicePlannerConfig,
                )

                _y_ei = np.asarray(df["__early_inval__"].values, dtype=int)
                _X_ei = (
                    X_meta_models.select_dtypes(include=[np.number])
                    .fillna(0.0)
                    .to_numpy(dtype=float)
                )
                _oof_ei = np.full(len(_y_ei), 0.5, dtype=float)

                # Use SlicePlanner for temporal CV
                n_ei = len(_y_ei)
                events = pd.DataFrame(
                    {
                        "event_id": np.arange(n_ei, dtype=np.int64),
                        "symbol": np.repeat("ALL", n_ei),
                        "t0": pd.to_datetime(df["__ts__"], utc=True, errors="coerce"),
                        "t1": pd.to_datetime(df["__ts__"], utc=True, errors="coerce")
                        + pd.Timedelta(seconds=int(cfg.get("cv_embargo_bars", 12))),
                    }
                )
                p_cfg = SlicePlannerConfig.fast_defaults(schema=EventSchema())
                p_cfg = p_cfg.__class__(
                    **{
                        **p_cfg.__dict__,
                        "preset": p_cfg.preset.__class__(
                            preset_name=p_cfg.preset.preset_name,
                            outer=p_cfg.preset.outer,
                            inner=p_cfg.preset.inner.__class__(n_splits=3),
                            sampling=p_cfg.preset.sampling,
                            symbol_policy=p_cfg.preset.symbol_policy,
                            purge_policy=p_cfg.preset.purge_policy,
                        ),
                        "silent": True,
                        "min_rows_per_fold": 1,
                        "min_symbols_per_fold": 1,
                    }
                )
                bundle = SlicePlanner(p_cfg).build(events)
                splits = [
                    (plan.fit_idx, plan.predict_idx)
                    for plan in bundle["consumer_plans"]["ridge_sizer_fit"]
                    if plan.tag == "predict_outer_test"
                    and plan.fit_idx.size > 0
                    and plan.predict_idx.size > 0
                ]
                if not splits:
                    raise ValueError(
                        "SlicePlanner failed to generate early invalidation splits"
                    )

                for _tr, _va in splits:
                    if len(np.unique(_y_ei[_tr])) < 2:
                        continue
                    _m_ei = LogisticRegression(
                        max_iter=1000, class_weight="balanced", random_state=42
                    )
                    _m_ei.fit(_X_ei[_tr], _y_ei[_tr])
                    _oof_ei[_va] = _m_ei.predict_proba(_X_ei[_va])[:, 1]
                _m_ei_final = LogisticRegression(
                    max_iter=1000, class_weight="balanced", random_state=42
                )
                if len(np.unique(_y_ei)) >= 2:
                    _m_ei_final.fit(_X_ei, _y_ei)
                meta_models[f"{side}_{k}_early_inval"] = SimpleNamespace(
                    oof_probs=np.asarray(_oof_ei, dtype=np.float32),
                    model={
                        "kind": "early_inval_clf",
                        "models": [_m_ei_final],
                        "name": "early_inval",
                    },
                )
                _bucket_y_ret[f"{k}_early_inval"] = np.asarray(_y_ei, dtype=float)
                _md = {}
                for _cn in [
                    "timestamp",
                    "symbol",
                    "asset",
                    "__ts__",
                    "__symbol__",
                    "__y_bin__",
                    "__y_ret__",
                    "__u_policy_net__",
                    "__u_policy__",
                    "__y_outcome__",
                    "exit_code",
                    "__mae_ret__",
                    "__mfe_ret__",
                    "__bars_to_mfe__",
                    "__barrier_pct__",
                    "__early_inval__",
                    "__mr_path_penalty__",
                    "__mr_velocity_penalty__",
                ]:
                    if _cn in df.columns:
                        _md[_cn] = df[_cn].values
                for _cn in _ps_regime_cols:
                    if _cn in df.columns:
                        _md[_cn] = df[_cn].values
                _bucket_metadata[f"{k}_early_inval"] = _md
                tprint(f"Meta {k}_early_inval: fitted")
            except Exception as _e_ei:
                tprint(
                    f"Warning: early invalidation model failed for {side}_{k}: {_e_ei}"
                )

    # ── Downstream heads summary table (fed to users/sizers) ──
    tprint(f"\n{'═'*148}")
    tprint(
        "  META HEADS SUMMARY TABLE (Utility / MAE / MFE / Asym / Early-Inval / TBM Corr)"
    )
    tprint(f"{'═'*148}")
    _tbl_hdr = (
        f"  {'Bucket':16s} {'U_IC':>7s} {'MAE_IC':>7s} {'MFE_IC':>7s} {'ASYM_IC':>7s} "
        f"{'MAE~SL':>7s} {'MFE~TP':>7s} {'ASY~TP':>7s} {'ASY~SL':>7s} "
        f"{'EI_AUC':>7s} {'EI_P@10':>8s} {'N':>8s}"
    )
    tprint(_tbl_hdr)
    tprint(f"  {'─'*146}")
    strategies = get_strategies(cfg)
    for strat in strategies:
        side = strat["trade_side"]
        k = strat["strategy_id"]
        _b = f"{side}_{k}"
        _aux = _aux_head_oof.get(_b)
        u_ic, mae_ic, mfe_ic, asym_ic = (
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
        )
        corr_mae_sl, corr_mfe_tp, corr_asym_tp, corr_asym_sl = (
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
        )
        n_samp = 0
        if isinstance(_aux, dict):
            u_ic = _safe_spearman(
                _aux.get("oof_u_hat", []), _bucket_y_ret.get(f"{_b}_utility", [])
            )
            mae_ic = _safe_spearman(
                _aux.get("oof_log_mae_q70_hat", []),
                _bucket_y_ret.get(f"{_b}_mae_q70", []),
            )
            mfe_ic = _safe_spearman(
                _aux.get("oof_log_mfe_hat", []), _bucket_y_ret.get(f"{_b}_mfe", [])
            )
            asym_ic = _safe_spearman(
                _aux.get("oof_asym_hat", []), _bucket_y_ret.get(f"{_b}_asym", [])
            )
            n_samp = len(_aux.get("oof_u_hat", []))

            # TBM Correlation Diagnostics
            _md = _bucket_metadata.get(f"{_b}_utility", {})
            if "__y_outcome__" in _md:
                _outcomes = np.asarray(_md["__y_outcome__"])[:n_samp]
                if len(_outcomes) == n_samp:
                    is_sl = (_outcomes == 0).astype(int)
                    is_tp = (_outcomes == 2).astype(int)
                    if len(np.unique(is_sl)) > 1:
                        corr_mae_sl = _safe_spearman(
                            _aux.get("oof_log_mae_q70_hat", []), is_sl
                        )
                        corr_asym_sl = _safe_spearman(
                            _aux.get("oof_asym_hat", []), is_sl
                        )
                    if len(np.unique(is_tp)) > 1:
                        corr_mfe_tp = _safe_spearman(
                            _aux.get("oof_log_mfe_hat", []), is_tp
                        )
                        corr_asym_tp = _safe_spearman(
                            _aux.get("oof_asym_hat", []), is_tp
                        )

        ei_auc, ei_p10 = float("nan"), float("nan")
        _ei_key = f"{_b}_early_inval"
        _ei = meta_models.get(_ei_key)
        _ei_md = _bucket_metadata.get(_ei_key, {})
        if (
            _ei is not None
            and hasattr(_ei, "oof_probs")
            and _ei.oof_probs is not None
            and "__early_inval__" in _ei_md
        ):
            _p = np.asarray(_ei.oof_probs, dtype=float)
            _y = np.asarray(_ei_md["__early_inval__"], dtype=float)[: len(_p)]
            _m = np.isfinite(_p) & np.isfinite(_y)
            if np.sum(_m) >= 10 and len(np.unique(_y[_m])) > 1:
                _n_pos = int(np.sum(_y[_m] == 1))
                _n_neg = int(np.sum(_y[_m] == 0))
                if _n_pos > 0 and _n_neg > 0:
                    _ranks = pd.Series(_p[_m]).rank(method="average").to_numpy(float)
                    _u = _ranks[_y[_m] == 1].sum() - _n_pos * (_n_pos + 1) / 2.0
                    ei_auc = float(_u / (_n_pos * _n_neg))
                _k10 = max(1, int(np.ceil(0.10 * np.sum(_m))))
                _idx10 = np.argsort(_p[_m])[-_k10:]
                ei_p10 = float(np.mean(_y[_m][_idx10]))

        tprint(
            f"  {_b:16s} {u_ic:>7.4f} {mae_ic:>7.4f} {mfe_ic:>7.4f} {asym_ic:>7.4f} "
            f"{corr_mae_sl:>7.4f} {corr_mfe_tp:>7.4f} {corr_asym_tp:>7.4f} {corr_asym_sl:>7.4f} "
            f"{ei_auc:>7.4f} {ei_p10:>8.4f} {n_samp:>8d}"
        )
    tprint(f"{'═'*148}\n")

    # Save meta OOF predictions for ridge_position_sizer
    # Include trade context (return, direction) for position sizing
    _run_id = cfg.get("run_id", "default")
    meta_oof_dir = os.path.join(
        cfg.get("data_root", "data"), "artifacts", _run_id, "meta_oof"
    )
    os.makedirs(meta_oof_dir, exist_ok=True)
    # Make the artifact directory self-consistent for this run.
    # Re-runs should not leave stale classifier/regression heads behind.
    try:
        import glob as _glob

        for _p in _glob.glob(os.path.join(meta_oof_dir, "meta_oof_*.parquet")):
            try:
                os.remove(_p)
            except Exception:
                pass
    except Exception:
        pass
    # Remove stale legacy outputs for disabled heads.
    try:
        import glob as _glob

        _stale = []
        if not include_meta_reg:
            _stale += _glob.glob(os.path.join(meta_oof_dir, "meta_oof_*_reg.parquet"))
        if not include_meta_clf:
            _stale += _glob.glob(os.path.join(meta_oof_dir, "meta_oof_*_clf.parquet"))
        for _p in _stale:
            try:
                os.remove(_p)
            except Exception:
                pass
    except Exception:
        pass
    import re

    _allowed_meta_suffixes = ["_utility", "_mae_q70", "_mfe", "_asym", "_early_inval"]
    if include_meta_reg:
        _allowed_meta_suffixes.append("_reg")
    if include_meta_clf:
        _allowed_meta_suffixes.append("_clf")
    _allowed_meta_suffixes = tuple(_allowed_meta_suffixes)
    _aligned_map_pat = re.compile(
        r".*_(tbm_500_250|tbm_250_125|mae|mfe|asym)_h\d+$"
    )

    def _fill_nonfinite_oof_vector(
        values, global_neutral: float = 0.0, method: str = "median"
    ):
        _arr = np.asarray(values, dtype=np.float64).reshape(-1).copy()
        _finite = np.isfinite(_arr)
        if _finite.all():
            return _arr
        if _finite.any():
            if method == "mean":
                _fill = float(np.nanmean(_arr[_finite]))
            else:
                _fill = float(np.nanmedian(_arr[_finite]))
        else:
            _fill = float(global_neutral)
            tprint(f"    Fallback imputation triggered. Using global neutral: {_fill}")
        _arr[~_finite] = _fill
        return _arr

    def validate_meta_oof_schema(df: pd.DataFrame, key: str):
        required_cols = [
            "index",
            "timestamp",
            "symbol",
            "is_long",
            "return",
            "exit_code",
            "u_policy_net",
        ]

        # Determine expected prediction columns based on key
        if key.endswith("_utility"):
            required_cols.extend(["oof_u_hat"])
        elif key.endswith("_mae_q70"):
            required_cols.extend(["oof_log_mae_q70_hat", "oof_mae_q70_hat"])
        elif key.endswith("_mfe"):
            required_cols.extend(["oof_log_mfe_hat", "oof_mfe_hat"])
        elif key.endswith("_asym"):
            required_cols.extend(["oof_asym_hat"])
        elif key.endswith("_clf") or key.endswith("_move"):
            required_cols.extend(["oof_p_move"])
        elif key.endswith("_early_inval"):
            required_cols.extend(["oof_p_early_inval"])

        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            tprint(f"    SCHEMA WARNING: {key} missing columns: {missing}")

        for c in required_cols:
            if c in df.columns:
                if not pd.api.types.is_numeric_dtype(df[c]) and c not in [
                    "timestamp",
                    "symbol",
                ]:
                    tprint(f"    SCHEMA WARNING: {key} column {c} is not numeric")

        if "oof_p_move" in df.columns:
            p = np.asarray(df["oof_p_move"], dtype=float)
            if np.any((p < -1e-6) | (p > 1 + 1e-6)):
                tprint(f"    SCHEMA WARNING: {key} has out-of-range move probabilities")

    # Collect calibration metrics if configured
    _validate_calib = bool(cfg.get("META_VALIDATE_CALIBRATION", False))
    _calib_report = {}
    _export_summary = {}

    def record_export_stats(df, key):
        stats = {}
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]) and col not in [
                "timestamp",
                "symbol",
                "is_long",
                "index",
                "return",
                "exit_code",
                "u_policy_net",
            ]:
                arr = df[col].values
                valid = np.isfinite(arr)
                stats[col] = {
                    "mean": float(np.mean(arr[valid])) if np.any(valid) else None,
                    "std": float(np.std(arr[valid])) if np.any(valid) else None,
                    "min": float(np.min(arr[valid])) if np.any(valid) else None,
                    "max": float(np.max(arr[valid])) if np.any(valid) else None,
                    "missing_fraction": float(1.0 - np.mean(valid)),
                }
        _export_summary[key] = stats

    def compute_prob_calibration(y_true, y_prob, n_bins=10):
        from sklearn.metrics import brier_score_loss

        valid = np.isfinite(y_true) & np.isfinite(y_prob)
        if np.sum(valid) < n_bins:
            return {}
        y_t, y_p = y_true[valid], y_prob[valid]
        brier = float(brier_score_loss(y_t, y_p))

        # ECE and Reliability
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        binids = np.digitize(y_p, bins) - 1
        binids = np.clip(binids, 0, n_bins - 1)

        ece = 0.0
        reliability = []
        for i in range(n_bins):
            mask = binids == i
            if np.any(mask):
                mean_p = float(np.mean(y_p[mask]))
                mean_t = float(np.mean(y_t[mask]))
                count = int(np.sum(mask))
                ece += np.abs(mean_p - mean_t) * (count / len(y_p))
                reliability.append(
                    {"bin": i, "mean_pred": mean_p, "mean_true": mean_t, "count": count}
                )

        return {
            "brier_score": brier,
            "ece": float(ece),
            "reliability_curve": reliability,
        }

    def record_calibration(df, key):
        if not _validate_calib:
            return

        calib_stats = {}
        if "oof_p_move" in df.columns and "y_move" in df.columns:
            calib_stats["move_calibration"] = compute_prob_calibration(
                np.asarray(df["y_move"].values, dtype=float),
                np.asarray(df["oof_p_move"].values, dtype=float),
            )

        if "early_inval" in df.columns and "oof_p_early_inval" in df.columns:
            ei_true = df["early_inval"].values
            calib_stats["early_inval_calibration"] = compute_prob_calibration(
                ei_true, df["oof_p_early_inval"].values
            )

        if calib_stats:
            _calib_report[key] = calib_stats

    def _trim_1d(values, n: int):
        return np.asarray(values).reshape(-1)[: int(n)]

    def compute_meta_expected_value(p_tp, p_sl, ratio):
        return ratio * p_tp - p_sl

    for key, meta in meta_models.items():
        if not key.endswith(_allowed_meta_suffixes) and not _aligned_map_pat.match(key):
            continue
        if hasattr(meta, "oof_probs") and meta.oof_probs is not None:
            # Parse side from key (e.g., "long_mr_H2" -> "long", "short_tf_clf" -> "short")
            parts = key.split("_")
            side_parsed = parts[0] if parts else "long"
            is_long = 1 if side_parsed == "long" else 0

            meta_oof_path = os.path.join(meta_oof_dir, f"meta_oof_{key}.parquet")
            _is_clf_key = key.endswith("_clf") or key.endswith("_move")
            _is_tbm_multiclass_key = False
            _is_ei_key = key.endswith("_early_inval")
            if _is_ei_key:
                _oof_pred_1d = _fill_nonfinite_oof_vector(
                    meta.oof_probs, global_neutral=0.5, method="mean"
                )
                _n_meta = len(_oof_pred_1d)
                oof_df = pd.DataFrame(
                    {
                        "oof_pred": _trim_1d(_oof_pred_1d, _n_meta),
                        "oof_p_early_inval": _trim_1d(_oof_pred_1d, _n_meta),
                        "index": range(_n_meta),
                        "is_long": is_long,
                    }
                )
            elif _is_clf_key and np.ndim(meta.oof_probs) == 1:
                _p_move = _fill_nonfinite_oof_vector(
                    meta.oof_probs, global_neutral=0.5, method="mean"
                )
                _n_meta = len(_p_move)
                oof_df = pd.DataFrame(
                    {
                        "oof_pred": _trim_1d(_p_move, _n_meta),
                        "oof_p_move": _trim_1d(_p_move, _n_meta),
                        "index": range(_n_meta),
                        "is_long": is_long,
                    }
                )
                if hasattr(meta, "y_move") and getattr(meta, "y_move", None) is not None:
                    oof_df["y_move"] = _trim_1d(np.asarray(meta.y_move, dtype=float), _n_meta)
                if hasattr(meta, "y_move_soft") and getattr(meta, "y_move_soft", None) is not None:
                    oof_df["y_move_soft"] = _trim_1d(
                        np.asarray(meta.y_move_soft, dtype=float), _n_meta
                    )
                if hasattr(meta, "move_threshold") and getattr(meta, "move_threshold", None) is not None:
                    oof_df["move_threshold"] = _trim_1d(
                        np.asarray(meta.move_threshold, dtype=float), _n_meta
                    )
                if hasattr(meta, "oof_probs_raw") and getattr(meta, "oof_probs_raw", None) is not None:
                    _p_move_raw = _fill_nonfinite_oof_vector(
                        meta.oof_probs_raw, global_neutral=0.5, method="mean"
                    )
                    oof_df["oof_p_move_raw"] = _trim_1d(_p_move_raw, _n_meta)
                _robust_sigma = getattr(meta, "oof_robust_sigma", None)
                if _robust_sigma is not None:
                    _robust_sigma = _fill_nonfinite_oof_vector(
                        _robust_sigma, global_neutral=np.nan, method="mean"
                    )
                    oof_df["robust_sigma_meta_clf"] = _trim_1d(
                        _robust_sigma, _n_meta
                    )
                    oof_df["cv_meta_clf"] = _trim_1d(
                        _robust_sigma / (np.abs(_p_move) + 1e-9), _n_meta
                    )
            elif _is_clf_key and np.ndim(meta.oof_probs) == 2 and meta.oof_probs.shape[1] == 3:
                p_sl = _fill_nonfinite_oof_vector(
                    meta.oof_probs[:, 0], global_neutral=1.0 / 3.0, method="mean"
                )
                p_to = _fill_nonfinite_oof_vector(
                    meta.oof_probs[:, 1], global_neutral=1.0 / 3.0, method="mean"
                )
                p_tp = _fill_nonfinite_oof_vector(
                    meta.oof_probs[:, 2], global_neutral=1.0 / 3.0, method="mean"
                )
                _n_meta = min(len(p_sl), len(p_to), len(p_tp))
                oof_df = pd.DataFrame(
                    {
                        "oof_pred": _trim_1d(p_tp, _n_meta),
                        "oof_p_move": _trim_1d(p_tp, _n_meta),
                        "index": range(_n_meta),
                        "is_long": is_long,
                    }
                )
                if hasattr(meta, "y_move") and getattr(meta, "y_move", None) is not None:
                    oof_df["y_move"] = _trim_1d(np.asarray(meta.y_move, dtype=float), _n_meta)
                if hasattr(meta, "y_move_soft") and getattr(meta, "y_move_soft", None) is not None:
                    oof_df["y_move_soft"] = _trim_1d(
                        np.asarray(meta.y_move_soft, dtype=float), _n_meta
                    )
                if hasattr(meta, "move_threshold") and getattr(meta, "move_threshold", None) is not None:
                    oof_df["move_threshold"] = _trim_1d(
                        np.asarray(meta.move_threshold, dtype=float), _n_meta
                    )
                _robust_sigma = getattr(meta, "oof_robust_sigma", None)
                if _robust_sigma is not None:
                    _robust_sigma = _fill_nonfinite_oof_vector(
                        _robust_sigma, global_neutral=np.nan, method="mean"
                    )
                    oof_df["robust_sigma_meta_clf"] = _trim_1d(
                        _robust_sigma, _n_meta
                    )
                    oof_df["cv_meta_clf"] = _trim_1d(
                        _robust_sigma / (np.abs(p_tp) + 1e-9), _n_meta
                    )
            elif _is_tbm_multiclass_key and np.ndim(meta.oof_probs) == 2:
                p_sl = _fill_nonfinite_oof_vector(
                    meta.oof_probs[:, 0], global_neutral=1.0 / 3.0, method="mean"
                )
                p_to = _fill_nonfinite_oof_vector(
                    meta.oof_probs[:, 1], global_neutral=1.0 / 3.0, method="mean"
                )
                p_tp = _fill_nonfinite_oof_vector(
                    meta.oof_probs[:, 2], global_neutral=1.0 / 3.0, method="mean"
                )
                _n_meta = min(len(p_sl), len(p_to), len(p_tp))
                oof_df = pd.DataFrame(
                    {
                        "oof_pred": _trim_1d(p_tp, _n_meta),
                        "oof_p_move": _trim_1d(p_tp, _n_meta),
                        "index": range(_n_meta),
                        "is_long": is_long,
                    }
                )
                if hasattr(meta, "y_move") and getattr(meta, "y_move", None) is not None:
                    oof_df["y_move"] = _trim_1d(np.asarray(meta.y_move, dtype=float), _n_meta)
                if hasattr(meta, "y_move_soft") and getattr(meta, "y_move_soft", None) is not None:
                    oof_df["y_move_soft"] = _trim_1d(
                        np.asarray(meta.y_move_soft, dtype=float), _n_meta
                    )
                if hasattr(meta, "move_threshold") and getattr(meta, "move_threshold", None) is not None:
                    oof_df["move_threshold"] = _trim_1d(
                        np.asarray(meta.move_threshold, dtype=float), _n_meta
                    )
            else:
                _oof_pred_1d = _fill_nonfinite_oof_vector(
                    meta.oof_probs, global_neutral=0.0, method="median"
                )
                _n_meta = len(_oof_pred_1d)
                oof_df = pd.DataFrame(
                    {
                        "oof_pred": _trim_1d(_oof_pred_1d, _n_meta),
                        "index": range(_n_meta),
                        "is_long": is_long,
                    }
                )
                if key.endswith("_reg"):
                    _score_sign = int(getattr(meta, "score_sign", 1))
                    oof_df["score_sign"] = _score_sign
                    oof_df["oof_pred_oriented"] = _score_sign * _trim_1d(
                        _oof_pred_1d, _n_meta
                    )
                if key.endswith("_utility"):
                    oof_df["oof_u_hat"] = _trim_1d(_oof_pred_1d, _n_meta)
                    _robust_sigma = getattr(meta, "oof_robust_sigma", None)
                    if _robust_sigma is not None:
                        _robust_sigma = _fill_nonfinite_oof_vector(
                            _robust_sigma, global_neutral=np.nan, method="mean"
                        )
                        oof_df["robust_sigma_meta_reg"] = _trim_1d(
                            _robust_sigma, _n_meta
                        )
                        oof_df["cv_meta_reg"] = _trim_1d(
                            _robust_sigma / (np.abs(_oof_pred_1d) + 1e-9), _n_meta
                        )
                elif key.endswith("_mae_q70"):
                    oof_df["oof_log_mae_q70_hat"] = _trim_1d(_oof_pred_1d, _n_meta)
                    oof_df["oof_mae_q70_hat"] = np.expm1(
                        _trim_1d(_oof_pred_1d, _n_meta)
                    )
                elif key.endswith("_mfe"):
                    oof_df["oof_log_mfe_hat"] = _trim_1d(_oof_pred_1d, _n_meta)
                    oof_df["oof_mfe_hat"] = np.expm1(_trim_1d(_oof_pred_1d, _n_meta))
                elif key.endswith("_asym"):
                    oof_df["oof_asym_hat"] = _trim_1d(_oof_pred_1d, _n_meta)
                if hasattr(meta, "y_move") and getattr(meta, "y_move", None) is not None:
                    oof_df["y_move"] = _trim_1d(np.asarray(meta.y_move, dtype=float), _n_meta)
                if hasattr(meta, "move_threshold") and getattr(meta, "move_threshold", None) is not None:
                    oof_df["move_threshold"] = _trim_1d(
                        np.asarray(meta.move_threshold, dtype=float), _n_meta
                    )

            # Attach metadata from bucket-specific storage
            _md = _bucket_metadata.get(key, {})
            _n_meta = min(_n_meta, len(oof_df))

            if _md:
                if "__ts__" in _md:
                    oof_df["timestamp"] = pd.to_datetime(_md["__ts__"]).values[:_n_meta]
                elif "timestamp" in _md:
                    oof_df["timestamp"] = pd.to_datetime(_md["timestamp"]).values[
                        :_n_meta
                    ]

                if "__symbol__" in _md:
                    oof_df["symbol"] = _md["__symbol__"][:_n_meta]
                elif "symbol" in _md:
                    oof_df["symbol"] = _md["symbol"][:_n_meta]
                elif "asset" in _md:
                    oof_df["symbol"] = _md["asset"][:_n_meta]

                if "__y_ret__" in _md:
                    oof_df["return"] = _md["__y_ret__"][:_n_meta]
                elif "return" in _md:
                    oof_df["return"] = _md["return"][:_n_meta]
                if "__barrier_pct__" in _md:
                    oof_df["barrier_pct"] = np.asarray(_md["__barrier_pct__"], dtype=float)[:_n_meta]
                if "__y_bin__" in _md:
                    oof_df["y_bin"] = np.asarray(_md["__y_bin__"], dtype=float)[
                        :_n_meta
                    ]
                elif "y_bin" in _md:
                    oof_df["y_bin"] = np.asarray(_md["y_bin"], dtype=float)[:_n_meta]

                _bucket_base = "_".join(key.split("_")[:2])

                if "__u_policy_net__" in _md:
                    oof_df["u_policy_net"] = _md["__u_policy_net__"][:_n_meta]
                elif f"{_bucket_base}_utility" in _bucket_y_ret:
                    oof_df["u_policy_net"] = _bucket_y_ret[f"{_bucket_base}_utility"][
                        :_n_meta
                    ]
                if "__u_policy__" in _md:
                    oof_df["u_policy"] = _md["__u_policy__"][:_n_meta]
                if "__y_outcome__" in _md:
                    oof_df["exit_code"] = _md["__y_outcome__"][:_n_meta]
                elif "exit_code" in _md:
                    oof_df["exit_code"] = _md["exit_code"][:_n_meta]

                _bucket_base = "_".join(key.split("_")[:2])

                if "__mae_ret__" in _md:
                    oof_df["mae_ret"] = _md["__mae_ret__"][:_n_meta]
                elif f"{_bucket_base}_mae_q70" in _bucket_y_ret:
                    # In case it is saved in _bucket_y_ret rather than _md
                    oof_df["mae_ret"] = _bucket_y_ret[f"{_bucket_base}_mae_q70"][
                        :_n_meta
                    ]
                if "__mfe_ret__" in _md:
                    oof_df["mfe_ret"] = _md["__mfe_ret__"][:_n_meta]
                elif f"{_bucket_base}_mfe" in _bucket_y_ret:
                    oof_df["mfe_ret"] = _bucket_y_ret[f"{_bucket_base}_mfe"][:_n_meta]
                if "__bars_to_mfe__" in _md:
                    oof_df["bars_to_mfe"] = _md["__bars_to_mfe__"][:_n_meta]
                if "__mr_path_penalty__" in _md:
                    oof_df["mr_path_penalty"] = _md["__mr_path_penalty__"][:_n_meta]
                if "__mr_velocity_penalty__" in _md:
                    oof_df["mr_velocity_penalty"] = _md["__mr_velocity_penalty__"][
                        :_n_meta
                    ]
                if "__early_inval__" in _md:
                    oof_df["early_inval"] = _md["__early_inval__"][:_n_meta]
                for _cn in _ps_regime_cols:
                    if _cn in _md:
                        oof_df[_cn] = _md[_cn][:_n_meta]

                if "oof_p_move" in oof_df.columns:
                    if hasattr(meta, "y_move") and getattr(meta, "y_move", None) is not None:
                        y_move = np.asarray(meta.y_move, dtype=float)[:_n_meta]
                        oof_df["y_move"] = y_move
                    elif hasattr(meta, "y_move_soft") and getattr(meta, "y_move_soft", None) is not None:
                        y_soft = np.asarray(meta.y_move_soft, dtype=float)[:_n_meta]
                        y_move = (y_soft >= 0.5).astype(float)
                        oof_df["y_move"] = y_move
                    elif "return" in oof_df.columns:
                        move_k = float(getattr(meta, "label_threshold", 1.25))
                        if "barrier_pct" in oof_df.columns:
                            move_thr = move_k * np.asarray(oof_df["barrier_pct"], dtype=float)
                        else:
                            move_thr = np.full(len(oof_df), float(move_k), dtype=float)
                        y_move = (
                            np.abs(np.asarray(oof_df["return"], dtype=float)) > move_thr
                        ).astype(float)
                        oof_df["y_move"] = y_move
                        oof_df["move_threshold"] = np.asarray(move_thr, dtype=float)

            _bucket_base = "_".join(key.split("_")[:2])
            _aux = _aux_head_oof.get(_bucket_base)
            if isinstance(_aux, dict):
                for _cn in [
                    "oof_u_hat",
                    "oof_log_mae_q70_hat",
                    "oof_log_mfe_hat",
                    "oof_asym_hat",
                ]:
                    if _cn in _aux and len(_aux[_cn]) == _n_meta:
                        oof_df[_cn] = _fill_nonfinite_oof_vector(
                            _aux[_cn], neutral=0.0
                        ).astype(np.float32, copy=False)

            if key.endswith("_reg") and "oof_pred_oriented" not in oof_df.columns:
                _score_sign = int(getattr(meta, "score_sign", 1))
                oof_df["score_sign"] = _score_sign
                oof_df["oof_pred_oriented"] = _score_sign * np.asarray(
                    oof_df["oof_pred"], dtype=float
                )

            record_export_stats(oof_df, key)
            record_calibration(oof_df, key)
            validate_meta_oof_schema(oof_df, key)
            oof_df.to_parquet(meta_oof_path, index=False)
            tprint(f"Saved meta OOF predictions for {key} to {meta_oof_path}")

    # Save auxiliary head OOF predictions as separate parquet files
    meta_scale_contract = {
        "oof_u_hat": "arcsinh(log-return normalized by ATR)",
        "oof_log_mae_q70_hat": "log1p(MAE / ATR)",
        "oof_mae_q70_hat": "MAE / ATR",
        "oof_log_mfe_hat": "log1p(MFE / ATR)",
        "oof_mfe_hat": "MFE / ATR",
        "oof_asym_hat": "log(MFE / MAE)",
        "oof_p_move": "probability of a materially large move",
        "oof_p_move_raw": "raw probability before calibration",
        "oof_p_early_inval": "probability",
        "oof_ev": "expected payoff multiple",
    }
    import json

    with open(os.path.join(meta_oof_dir, "meta_export_summary.json"), "w") as _f:
        json.dump(_export_summary, _f, indent=2)
    if _validate_calib:
        with open(
            os.path.join(meta_oof_dir, "meta_calibration_report.json"), "w"
        ) as _f:
            json.dump(_calib_report, _f, indent=2)
    with open(os.path.join(meta_oof_dir, "meta_scale_contract.json"), "w") as _f:
        json.dump(meta_scale_contract, _f, indent=2)

    for bucket_base, aux_data in _aux_head_oof.items():
        if not isinstance(aux_data, dict):
            continue
        # Get metadata for this bucket
        _md = _bucket_metadata.get(bucket_base, {})
        _n = len(aux_data.get("oof_u_hat", []))
        if _n == 0:
            continue
        # Parse side from bucket_base (e.g., "long_mr" -> "long")
        parts = bucket_base.split("_")
        side_parsed = parts[0] if parts else "long"
        is_long = 1 if side_parsed == "long" else 0

        # Save each aux head as a separate parquet file.
        # If the explicit asym head is unavailable for a bucket, derive a
        # stable proxy from the existing MFE/MAE aux heads so the Ridge stack
        # still receives a dedicated asymmetry feature file.
        for head_name in ["utility", "mae_q70", "mfe", "asym"]:
            if head_name == "utility":
                oof_key = "oof_u_hat"
            elif head_name == "mae_q70":
                oof_key = "oof_log_mae_q70_hat"
            elif head_name == "mfe":
                oof_key = "oof_log_mfe_hat"
            elif head_name == "asym":
                oof_key = "oof_asym_hat"
            else:
                continue

            if oof_key not in aux_data:
                if head_name == "asym" and all(
                    k in aux_data for k in ("oof_log_mfe_hat", "oof_log_mae_q70_hat")
                ):
                    _oof_pred = (
                        np.asarray(aux_data["oof_log_mfe_hat"], dtype=float)
                        - np.asarray(aux_data["oof_log_mae_q70_hat"], dtype=float)
                    ).astype(np.float32, copy=False)
                    aux_data["oof_asym_hat"] = _oof_pred
                    tprint(
                        f"  WARNING: Derived missing asym aux head for {bucket_base} from MFE-MAE spread."
                    )
                else:
                    continue

            _oof_pred = _fill_nonfinite_oof_vector(
                aux_data[oof_key], global_neutral=0.0, method="median"
            )
            oof_df = pd.DataFrame(
                {
                    "oof_pred": _oof_pred,
                    "index": range(_n),
                    "is_long": is_long,
                }
            )
            if head_name == "utility":
                oof_df["oof_u_hat"] = _oof_pred
            elif head_name == "mae_q70":
                oof_df["oof_log_mae_q70_hat"] = _oof_pred
                oof_df["oof_mae_q70_hat"] = np.expm1(_oof_pred)
            elif head_name == "mfe":
                oof_df["oof_log_mfe_hat"] = _oof_pred
                oof_df["oof_mfe_hat"] = np.expm1(_oof_pred)
            elif head_name == "asym":
                oof_df["oof_asym_hat"] = _oof_pred

            # Attach metadata
            if _md:
                if "__ts__" in _md:
                    oof_df["timestamp"] = pd.to_datetime(_md["__ts__"]).values[:_n]
                elif "timestamp" in _md:
                    oof_df["timestamp"] = pd.to_datetime(_md["timestamp"]).values[:_n]
                if "__symbol__" in _md:
                    oof_df["symbol"] = _md["__symbol__"][:_n]
                elif "symbol" in _md:
                    oof_df["symbol"] = _md["symbol"][:_n]
                elif "asset" in _md:
                    oof_df["symbol"] = _md["asset"][:_n]
                if "__y_ret__" in _md:
                    oof_df["return"] = _md["__y_ret__"][:_n]
                elif "return" in _md:
                    oof_df["return"] = _md["return"][:_n]
                if "__y_bin__" in _md:
                    oof_df["y_bin"] = np.asarray(_md["__y_bin__"], dtype=float)[:_n]
                elif "y_bin" in _md:
                    oof_df["y_bin"] = np.asarray(_md["y_bin"], dtype=float)[:_n]

            meta_oof_path = os.path.join(
                meta_oof_dir, f"meta_oof_{bucket_base}_{head_name}.parquet"
            )
            record_export_stats(oof_df, f"{bucket_base}_{head_name}")
            record_calibration(oof_df, f"{bucket_base}_{head_name}")
            validate_meta_oof_schema(oof_df, f"{bucket_base}_{head_name}")
            oof_df.to_parquet(meta_oof_path, index=False)
            tprint(
                f"Saved meta OOF predictions for aux head {bucket_base}_{head_name} to {meta_oof_path}"
            )

    # Train EV-decomposition probabilistic/quantile heads in meta step so they share
    # the same meta feature keys and selector pipeline context.
    if bool(cfg.get("ev_decomposition_enabled", False)) and bool(
        cfg.get("ev_decomposition_train_in_meta", True)
    ):
        try:
            from extreme_price_movements.position_sizer.training_orchestrator import (
                train_position_sizer_models as _train_ps_models,
            )

            _ps_buckets = {}
            _strats = get_strategies(cfg)
            for _strat in _strats:
                _b = f"{_strat['trade_side']}_{_strat['strategy_id']}"
                _util_key = f"{_b}_utility"
                _md = _bucket_metadata.get(_util_key, {})
                _aux = _aux_head_oof.get(_b, {})
                if not isinstance(_aux, dict):
                    continue
                _n = len(np.asarray(_aux.get("oof_u_hat", []), dtype=float))
                if _n <= 0:
                    continue
                _oof = pd.DataFrame(
                    {
                        "score": np.asarray(
                            _aux.get("oof_u_hat", np.zeros(_n)), dtype=float
                        ),
                        "oof_u_hat": np.asarray(
                            _aux.get("oof_u_hat", np.zeros(_n)), dtype=float
                        ),
                        "oof_log_mae_q70_hat": np.asarray(
                            _aux.get("oof_log_mae_q70_hat", np.zeros(_n)), dtype=float
                        ),
                        "oof_log_mfe_hat": np.asarray(
                            _aux.get("oof_log_mfe_hat", np.zeros(_n)), dtype=float
                        ),
                        "oof_asym_hat": np.asarray(
                            _aux.get("oof_asym_hat", np.zeros(_n)), dtype=float
                        ),
                    }
                )
                _out = pd.DataFrame(
                    {
                        "return": np.asarray(
                            _md.get("__y_ret__", np.zeros(_n)), dtype=float
                        )[:_n],
                        "y_bin": np.asarray(
                            _md.get("__y_bin__", np.zeros(_n)), dtype=float
                        )[:_n],
                        "mfe_ret": np.asarray(
                            _md.get("__mfe_ret__", np.zeros(_n)), dtype=float
                        )[:_n],
                        "mae_ret": np.asarray(
                            _md.get("__mae_ret__", np.zeros(_n)), dtype=float
                        )[:_n],
                        "timestamp": pd.to_datetime(_md.get("__ts__", np.arange(_n))),
                        "symbol": np.asarray(
                            _md.get("__symbol__", np.array(["UNK"] * _n)), dtype=object
                        )[:_n],
                    }
                )
                for _cn in _ps_regime_cols:
                    if _cn in _md:
                        _oof[_cn] = np.asarray(_md[_cn])[:_n]
                _ps_buckets[_b] = {"oof": _oof, "outcomes": _out}
            if _ps_buckets:
                _ps_rows = []
                for _b, _pack in _ps_buckets.items():
                    _oof = _pack.get("oof")
                    _out = _pack.get("outcomes")
                    if _oof is None or _out is None or len(_oof) != len(_out):
                        continue
                    _df = pd.DataFrame(index=np.arange(len(_oof)))
                    _df["score"] = np.asarray(
                        _oof.get("score", pd.Series(np.zeros(len(_oof)))).values,
                        dtype=float,
                    )
                    _df["pnl_label"] = np.asarray(
                        _out.get("return", pd.Series(np.zeros(len(_out)))).values,
                        dtype=float,
                    )
                    _df["y_bin"] = np.asarray(
                        _out.get("y_bin", pd.Series(np.zeros(len(_out)))).values,
                        dtype=float,
                    )
                    _df["mfe"] = np.asarray(
                        _out.get("mfe_ret", pd.Series(np.zeros(len(_out)))).values,
                        dtype=float,
                    )
                    _df["mae"] = np.asarray(
                        _out.get("mae_ret", pd.Series(np.zeros(len(_out)))).values,
                        dtype=float,
                    )
                    _df["timestamp"] = np.asarray(
                        _out.get("timestamp", pd.Series(np.arange(len(_out)))).values
                    )
                    _df["symbol"] = np.asarray(
                        _out.get("symbol", pd.Series(["UNK"] * len(_out))).values
                    )
                    _df["bucket"] = _b
                    for _c in _oof.columns:
                        if _c in _df.columns:
                            continue
                        try:
                            _v = pd.to_numeric(_oof[_c], errors="coerce").astype(float)
                            if np.isfinite(_v).any():
                                _df[_c] = _v.values
                        except Exception:
                            continue
                    _ps_rows.append(_df)
                if _ps_rows:
                    _ps_df = pd.concat(_ps_rows, axis=0, ignore_index=True)
                    if "timestamp" in _ps_df.columns:
                        _ps_df = _ps_df.sort_values("timestamp").reset_index(drop=True)
                    _forbid = {
                        "pnl_label",
                        "mfe",
                        "mae",
                        "timestamp",
                        "symbol",
                        "bucket",
                        "return",
                        "is_long",
                        "index",
                    }
                    _priority = [
                        str(c)
                        for c in cfg.get("position_sizer_feature_priority", ["score"])
                        if isinstance(c, str)
                    ]
                    _num_cols = [
                        c
                        for c in _ps_df.columns
                        if c not in _forbid and pd.api.types.is_numeric_dtype(_ps_df[c])
                    ]
                    _feature_cols = []
                    for _c in _priority + _ps_regime_cols + _num_cols:
                        if _c in _num_cols and _c not in _feature_cols:
                            _feature_cols.append(_c)
                    if not _feature_cols:
                        _feature_cols = (
                            ["score"] if "score" in _ps_df.columns else ["pnl_label"]
                        )
                    _ps_train = _train_ps_models(
                        ps_df=_ps_df, feature_cols=_feature_cols, cfg=cfg
                    )
                    _ev_diag = dict(_ps_train.get("diagnostics", {}) or {})
                    _ev_train_cfg = dict(_ps_train.get("training_config", {}) or {})
                    if _ev_train_cfg:
                        tprint(
                            "Meta EV decomposition models trained: "
                            f"pwin={_ev_train_cfg.get('pwin_base_engine')} "
                            f"quant={_ev_train_cfg.get('quantile_base_engine')} "
                            f"reg={_ev_train_cfg.get('regularization_level')} "
                            f"cal={_ev_train_cfg.get('calibrator_method')}"
                        )
                    from extreme_price_movements.position_sizer.runtime import (
                        EVDecompositionBundle,
                        compute_schema_hash,
                        make_bundle_metadata,
                    )

                    _git_sha = ""
                    try:
                        import subprocess as _sp

                        _git_sha = _sp.check_output(
                            ["git", "rev-parse", "--short", "HEAD"], text=True
                        ).strip()
                    except Exception:
                        _git_sha = ""
                    _q_cfg = {
                        "exp_win_quantile": float(_ps_train["exp_win_quantile"]),
                        "risk_loss_quantile": float(_ps_train["risk_loss_quantile"]),
                        "costs_mode": str(_ps_train["costs_mode"]),
                    }
                    _schema_hash = compute_schema_hash(_feature_cols, extra=_q_cfg)
                    _meta = make_bundle_metadata(git_sha=_git_sha)
                    _bundle = EVDecompositionBundle(
                        feature_cols=list(_feature_cols),
                        pwin_model=_ps_train["pwin_model"],
                        win_model=_ps_train["win_model"],
                        loss_model=_ps_train["loss_model"],
                        tp_sl_defaults=None,
                        config={
                            "backend": "ev_decomposition",
                            "soft_label_enabled": bool(
                                _ps_train.get("soft_label_enabled", False)
                            ),
                            "calibration_scope": str(
                                cfg.get("position_sizer_calibration_scope", "regime")
                            ),
                            "exp_win_quantile": float(_ps_train["exp_win_quantile"]),
                            "risk_loss_quantile": float(
                                _ps_train["risk_loss_quantile"]
                            ),
                            "costs_mode": str(_ps_train["costs_mode"]),
                        },
                        version="v1",
                        bundle_version=1,
                        created_at=_meta.get("created_at", ""),
                        git_sha=_meta.get("git_sha", ""),
                        schema_hash=_schema_hash,
                    )
                    _ps_out = os.path.join(
                        cfg.get("data_root", "data"),
                        "artifacts",
                        _run_id,
                        "ev_decomposition",
                    )
                    os.makedirs(_ps_out, exist_ok=True)
                    _bundle_path = os.path.join(_ps_out, "ev_decomposition_bundle.pkl")
                    with open(_bundle_path, "wb") as _f:
                        pickle.dump(_bundle, _f)
                    _manifest = {
                        "bundle_path": _bundle_path,
                        "feature_cols": _feature_cols,
                        "version": "v1",
                        "bundle_version": 1,
                    }
                    with open(
                        os.path.join(_ps_out, "ev_decomposition_bundle.json"), "w"
                    ) as _f:
                        json.dump(_manifest, _f, indent=2)
                    with open(
                        os.path.join(
                            _ps_out, "ev_decomposition_training_diagnostics.json"
                        ),
                        "w",
                    ) as _f:
                        json.dump(
                            {"training_config": _ev_train_cfg, "diagnostics": _ev_diag},
                            _f,
                            indent=2,
                        )
                    _state_path = os.path.join(
                        cfg.get("data_root", "data"),
                        "artifacts",
                        _run_id,
                        "models",
                        "trained_state.pkl",
                    )
                    if os.path.exists(_state_path):
                        try:
                            with open(_state_path, "rb") as _f:
                                _state = pickle.load(_f)
                            if isinstance(_state, dict):
                                _state["ev_decomposition"] = dict(_manifest)
                                with open(_state_path, "wb") as _f:
                                    pickle.dump(_state, _f)
                        except Exception as _e_state:
                            tprint(
                                f"Warning: failed to update trained_state with ev_decomposition manifest: {_e_state}"
                            )
                    tprint(
                        "Meta training: EV decomposition bundle trained from meta buckets."
                    )
        except Exception as _e_ps_meta:
            tprint(f"Warning: meta EV decomposition training failed: {_e_ps_meta}")

    tprint(
        f"train_meta_models_from_artifacts: done ({_time.monotonic()-_t0_meta:.1f}s), {len(meta_models)} meta models"
    )

    # Print selected features for each meta model
    tprint("\n=== META MODEL SELECTED FEATURES (Top 20) ===")
    for key, meta in meta_models.items():
        if hasattr(meta, "selected_features") and meta.selected_features:
            feats = (
                meta.selected_features[:20]
                if len(meta.selected_features) > 20
                else meta.selected_features
            )
            tprint(f"\n{key} ({len(meta.selected_features)} total features):")
            for i, f in enumerate(feats, 1):
                tprint(f"  {i:2d}. {f}")
    tprint("=== END META FEATURES ===\n")

    return meta_models, meta_gate_results


def train_models_from_artifacts(datasets, cfg, train_meta=True, train_base=True):
    tprint(f"Entering function: train_models_from_artifacts in training.py")
    tprint(f"train_base={train_base}, train_meta={train_meta}")
    cfg = _resolve_training_cfg_with_offline_optimisers(cfg)
    final_models = {}
    spike_models = {}
    exh_models = {}
    degenerate_strategy_ids = set()

    alpha_gate_results = []
    meta_gate_results = []

    specialist_models = {
        "trap_model": None,
        "gamma_model": None,
    }

    specialist_oof_lookup = {}  # key: (ts, symbol) -> dict of scores
    tprint("Spike anatomy and specialist models are disabled.")
    gc.collect()

    # 2. Train Alpha Models
    final_models = {}
    base_variant_models = {}
    min_base_fit_features = int(cfg.get("base_min_fit_features_guard", 8))

    def _log_feature_matrix_state(label, X_df, selected=None):
        cols_now = list(X_df.columns) if hasattr(X_df, "columns") else []
        n_cols = len(cols_now)
        n_unique = len(set(cols_now))
        tprint(
            f"{label}: rows={len(X_df)} cols={n_cols} unique_cols={n_unique}"
        )
        if cols_now:
            tprint(f"  {label} columns sample: {cols_now[:12]}")
        if selected is not None:
            selected = list(selected)
            tprint(
                f"{label}: selected_features={len(selected)} final_fit_features={len(selected)}"
            )
            if selected:
                tprint(f"  {label} selected sample: {selected[:12]}")

    def _feature_guard_ok(label, cols):
        n_cols = len(list(cols))
        if n_cols < min_base_fit_features:
            tprint(
                f"CRITICAL: {label} has only {n_cols} usable features "
                f"(guard={min_base_fit_features}). Skipping as broken config."
            )
            return False
        return True

    def _train_base_variant_dataset(
        side_name, kind_name, horizon, dataset_key, df_variant, strategy=None
    ):
        if (
            df_variant is None
            or df_variant.empty
            or len(df_variant) < cfg["min_train_samples"] // 4
        ):
            return None
        y = df_variant["__y_bin__"].values.astype(np.float32)
        y_ret = df_variant["__y_ret__"].values.astype(np.float32)
        w_raw = df_variant["__w__"].values.astype(np.float32)
        w = np.sqrt(np.clip(w_raw, 0.0, None))
        variant_sel_max = int(cfg.get("base_variant_selector_max_samples", 25000))
        variant_fit_max = int(cfg.get("base_variant_fit_max_samples", 100000))

        drop_cols = [
            "__y_bin__",
            "__y_ret__",
            "__w__",
            "__ts__",
            "__symbol__",
            "__barrier_pct__",
        ]
        X = df_variant.drop(columns=[c for c in drop_cols if c in df_variant.columns])
        leak_cols = [c for c in X.columns if str(c).startswith("__")]
        if leak_cols:
            tprint(
                f"  Dropping {len(leak_cols)} internal label columns from base variant features: {leak_cols}"
            )
            X = X.drop(columns=leak_cols, errors="ignore")

        # Use strategy dict to extract feature keys
        allowed_keys = set(strategy.get("feature_keys", [])) if strategy else set()
        std_inputs = {
            "trap_score",
            "gamma_score",
            "mkt_ret6h",
            "mkt_ret24h",
            "mkt_rv",
            "mkt_trend",
            "G_VOL",
            "G_TREND",
        }

        valid_cols = []
        for c in X.columns:
            if c in allowed_keys or c in std_inputs:
                valid_cols.append(c)
                continue
            for g in ["G_VOL", "G_TREND"]:
                if f"_{g}_0" in c or f"_{g}_1" in c:
                    base = c.split(f"_{g}_")[0]
                    if base in allowed_keys:
                        valid_cols.append(c)
                        break

        # --- Pipeline Hardening: Ensure valid_cols strictly exist in X ---
        _missing_from_X = [c for c in valid_cols if c not in X.columns]
        if _missing_from_X:
            tprint(
                f"  WARNING: Base Model Selector: Skipping {len(_missing_from_X)} allowed features missing from X: {_missing_from_X}"
            )
            valid_cols = [c for c in valid_cols if c in X.columns]

        if not valid_cols:
            valid_cols = list(X.columns)
        X = X[valid_cols]
        _log_feature_matrix_state(f"Base variant pre-MDI [{dataset_key}]", X)
        if not _feature_guard_ok(f"Base variant pre-MDI [{dataset_key}]", X.columns):
            return None
        _sel_idx = _subsample_indices_time_balanced(len(X), variant_sel_max, y)
        _fit_idx = _subsample_indices_time_balanced(len(X), variant_fit_max, y)
        from sklearn.ensemble import ExtraTreesRegressor

        _base_sel_cfg = dict(cfg.get("base_selector_cfg", {}) or {})
        mdi_base = ExtraTreesRegressor(
            n_estimators=int(_base_sel_cfg.get("analysis_n_estimators", 192)),
            max_depth=5,
            min_samples_leaf=64,
            max_features="sqrt",
            n_jobs=2,
            random_state=42,
        )
        _selector_head = f"base_{dataset_key}"
        _selector_report_dir = os.path.join(
            str(cfg.get("data_root", "data")),
            "artifacts",
            str(cfg.get("run_id", "default")),
            "fs_reports",
        )
        sel_res = mdi_feature_selection_v3(
            X.iloc[_sel_idx],
            y[_sel_idx],
            candidate_cols=list(X.columns),
            base_model=mdi_base,
            sample_weight=w[_sel_idx],
            selector_y=y[_sel_idx],
            selector_target=str(cfg.get("base_mdi_selector_target", "classification")),
            selector_loss=str(cfg.get("base_mdi_selector_loss", "binary_logloss")),
            selector_head_name=_selector_head,
            selector_report_dir=_selector_report_dir,
            selector_prev_selected=None,
            selector_family_map=dict(cfg.get("selector_feature_family_map", {}) or {}),
            selector_focus_top_frac=float(
                _base_sel_cfg.get("selector_focus_top_frac", 1.0)
            ),
            selector_top_metric=_base_sel_cfg.get("selector_top_metric", "ic"),
            selector_frequency_hit_mode=str(
                _base_sel_cfg.get("selector_frequency_hit_mode", "relative")
            ),
            selector_frequency_hit_quantile=float(
                _base_sel_cfg.get("selector_frequency_hit_quantile", 0.80)
            ),
            selector_frequency_hit_abs=float(
                _base_sel_cfg.get("selector_frequency_hit_abs", 1e-6)
            ),
            selector_interaction_mode=str(
                _base_sel_cfg.get("selector_interaction_mode", "tree_path_lift")
            ),
            selector_interaction_topk_pairs=int(
                _base_sel_cfg.get("selector_interaction_topk_pairs", 100)
            ),
            selector_interaction_max_pairs_per_feature=int(
                _base_sel_cfg.get("selector_interaction_max_pairs_per_feature", 8)
            ),
            selector_interaction_corr_penalty=bool(
                _base_sel_cfg.get("selector_interaction_corr_penalty", True)
            ),
            selector_family_penalty=bool(
                _base_sel_cfg.get("selector_family_penalty", True)
            ),
            selector_emit_report=bool(_base_sel_cfg.get("selector_emit_report", True)),
            analysis_n_estimators=int(_base_sel_cfg.get("analysis_n_estimators", 192)),
            analysis_max_samples=int(_base_sel_cfg.get("analysis_max_samples", 3000)),
            min_samples_leaf_pct=float(
                _base_sel_cfg.get("min_samples_leaf_pct", 0.015)
            ),
            selector_max_missing_frac=float(
                _base_sel_cfg.get("selector_max_missing_frac", 0.15)
            ),
            selector_near_constant_dominance=float(
                _base_sel_cfg.get("selector_near_constant_dominance", 0.999)
            ),
            selector_hysteresis_margin=float(
                _base_sel_cfg.get("selector_hysteresis_margin", 0.05)
            ),
            selector_min_overlap=float(_base_sel_cfg.get("selector_min_overlap", 0.70)),
            composite_weights={
                "top30": float(_base_sel_cfg.get("top30", 0.0)),
                "global": float(_base_sel_cfg.get("global", 0.55)),
                "stability": float(_base_sel_cfg.get("stability", 0.25)),
                "frequency": float(_base_sel_cfg.get("frequency", 0.15)),
                "interaction": float(_base_sel_cfg.get("interaction", 0.05)),
            },
            end_features=60,
            cumulative_cap=0.99,
            min_share=0.0005,
            min_features=30,
            max_features_pct=0.8,
        )
        selected_feats = _cap_selected_features(
            list(sel_res.selected_features),
            list(X.columns),
            target_cap=int(cfg.get("base_variant_mdi_target_features", 60)),
            min_features=int(cfg.get("base_variant_mdi_min_features", 30)),
        )
        top5_feats, top10_feats = _mdi_top_feature_lists(sel_res, selected_feats)
        _log_feature_matrix_state(
            f"Base variant post-MDI [{dataset_key}]",
            X.iloc[_fit_idx][selected_feats] if selected_feats else X.iloc[_fit_idx][[]],
            selected_feats,
        )
        if not _feature_guard_ok(
            f"Base variant post-MDI [{dataset_key}]",
            selected_feats,
        ):
            return None
        X_sel = X.iloc[_fit_idx][selected_feats]
        groups = (
            df_variant["__ts__"].values[_fit_idx]
            if "__ts__" in df_variant.columns
            else None
        )
        symbols = (
            df_variant["__symbol__"].values[_fit_idx]
            if "__symbol__" in df_variant.columns
            else None
        )
        race = _fit_direct_extratrees_base_model(
            kind_name=kind_name,
            X=X_sel,
            y=y[_fit_idx],
            sample_weight=w[_fit_idx],
            returns=y_ret[_fit_idx],
            groups=groups,
            symbols=symbols,
            n_splits=2,
            cfg=cfg,
        )
        return {
            "model": race,
            "H": int(horizon),
            "feat_cols": selected_feats,
            "selected_features": selected_feats,
            "top5_features": top5_feats,
            "top10_features": top10_feats,
            "dataset_key": dataset_key,
        }

    if train_base:
        from extreme_price_movements.strategy_registry import get_strategies

        strats = get_strategies(cfg)
        strategy_horizons = {
            s["strategy_id"]: strategy_runtime_horizons(s, cfg) for s in strats
        }
        for s in strats:
            s_side = s["trade_side"]
            if s_side not in final_models:
                final_models[s_side] = {}

        for strategy in strats:
            for side, k in [(strategy["trade_side"], strategy["strategy_id"])]:
                best_ic = -1.0
                best_m = None
                per_h_models = {}
                deployable_h_models = {}
                feature_selection_by_h = {}
                for H in strategy_horizons.get(k, []):
                    key = f"train_{k}_{H}"
                    if key not in datasets:
                        continue

                    df = datasets[key]
                    if df.empty or len(df) < cfg["min_train_samples"] // 4:
                        continue

                    _base_cap = int(cfg.get("base_fit_max_samples", 150000))
                    if _base_cap > 0 and len(df) > _base_cap:
                        df = subsample_symbol_balanced(df, _base_cap)
                        tprint(
                            f"  {key}: symbol-balanced subsample -> {len(df)} rows (cap={_base_cap})"
                        )

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
                            tprint(
                                f"  Injected {feat_name} into {key} (coverage: {np.mean(~np.isnan(vals)):.1%})"
                            )

                    y = df["__y_bin__"].values.astype(np.float32)
                    if len(np.unique((y >= 0.5).astype(int))) < 2:
                        tprint(
                            f"  Skipping {key}: single-class target (all {'positive' if np.mean(y)>=0.5 else 'negative'})"
                        )
                        continue
                    y_ret = df["__y_ret__"].values.astype(np.float32)
                    w_raw = df["__w__"].values.astype(np.float32)
                    # Temper weights with sqrt to reduce n_eff collapse from skewed uniqueness weights
                    # sqrt preserves relative ordering but compresses the tail
                    w = np.sqrt(np.clip(w_raw, 0.0, None))

                    drop_cols = [
                        "__y_bin__",
                        "__y_ret__",
                        "__w__",
                        "__ts__",
                        "__symbol__",
                        "__barrier_pct__",
                    ]
                    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
                    leak_cols = [c for c in X.columns if str(c).startswith("__")]
                    if leak_cols:
                        tprint(
                            f"  Dropping {len(leak_cols)} internal label columns from base features: {leak_cols}"
                        )
                        X = X.drop(columns=leak_cols, errors="ignore")

                    # Filter features strictly for the Alpha Model (exclude meta-only features)
                    # We need to know which feature_key was used.
                    # k is strategy_id
                    allowed_keys = set(strategy.get("feature_keys", []))

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
                    std_inputs = {
                        "p_exh_lag1",
                        "G_VOL",
                        "G_TREND",
                        "mkt_ret24h",
                        "mkt_ret6h",
                        "mkt_trend",
                        "mkt_rv",
                        "trap_score",
                        "gamma_score",
                    }

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
                        if is_inter:
                            continue

                    if not valid_cols:
                        tprint(
                            f"Warning: No valid columns found for {side} {k} after filtering. Using all."
                        )
                        valid_cols = list(X.columns)

                    X = X[valid_cols]
                    _log_feature_matrix_state(f"Base pre-MDI [{key}]", X)
                    if not _feature_guard_ok(f"Base pre-MDI [{key}]", X.columns):
                        continue
                    base_sel_max = int(cfg.get("base_selector_max_samples", 30000))
                    base_fit_max = int(cfg.get("base_fit_max_samples", 150000))
                    _sel_idx = _subsample_indices_time_balanced(len(X), base_sel_max, y)
                    _fit_idx = _subsample_indices_time_balanced(len(X), base_fit_max, y)
                    cols = list(X.columns)

                    # Explicit strategy context for logging:
                    # trade_side (long/short) is separate from move_bucket (up/down).
                    trade_side = side
                    cand_filter, move_bucket, strategy_label = _strategy_bucket_context(
                        trade_side, k, cfg
                    )
                    cand_filter_pretty = (
                        "top_ret" if cand_filter == "best" else "bottom_ret"
                    )

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

                    _base_sel_cfg = dict(cfg.get("base_selector_cfg", {}) or {})
                    mdi_base = ExtraTreesRegressor(
                        n_estimators=int(
                            _base_sel_cfg.get("analysis_n_estimators", 192)
                        ),
                        max_depth=5,
                        min_samples_leaf=64,
                        max_features="sqrt",
                        n_jobs=2,
                        random_state=42,
                    )
                    _selector_head = f"base_{side}_{k}_H{H}"
                    _selector_report_dir = os.path.join(
                        str(cfg.get("data_root", "data")),
                        "artifacts",
                        str(cfg.get("run_id", "default")),
                        "fs_reports",
                    )
                    _prev_sel = None
                    _prev_run_id = (
                        cfg.get("prev_run_id")
                        or cfg.get("prior_run_id")
                        or cfg.get("warm_start_run_id")
                    )
                    if _prev_run_id:
                        _prev_path = os.path.join(
                            str(cfg.get("data_root", "data")),
                            "artifacts",
                            str(_prev_run_id),
                            "fs_reports",
                            _selector_head,
                            "selected_features.json",
                        )
                        if os.path.exists(_prev_path):
                            try:
                                with open(_prev_path, "r", encoding="utf-8") as _f:
                                    _prev_sel = list(
                                        (json.load(_f) or {}).get(
                                            "selected_features", []
                                        )
                                    )
                            except Exception:
                                _prev_sel = None

                    sel_res = mdi_feature_selection_v3(
                        X.iloc[_sel_idx],
                        y[_sel_idx],
                        candidate_cols=list(X.columns),
                        base_model=mdi_base,
                        sample_weight=w[_sel_idx],
                        selector_y=y[_sel_idx],
                        selector_target=str(
                            cfg.get("base_mdi_selector_target", "classification")
                        ),
                        selector_loss=str(
                            cfg.get("base_mdi_selector_loss", "binary_logloss")
                        ),
                        selector_head_name=_selector_head,
                        selector_report_dir=_selector_report_dir,
                        selector_prev_selected=_prev_sel,
                        selector_family_map=dict(
                            cfg.get("selector_feature_family_map", {}) or {}
                        ),
                        selector_focus_top_frac=float(
                            _base_sel_cfg.get("selector_focus_top_frac", 1.0)
                        ),
                        selector_top_metric=_base_sel_cfg.get(
                            "selector_top_metric", "ic"
                        ),
                        selector_frequency_hit_mode=str(
                            _base_sel_cfg.get("selector_frequency_hit_mode", "relative")
                        ),
                        selector_frequency_hit_quantile=float(
                            _base_sel_cfg.get("selector_frequency_hit_quantile", 0.80)
                        ),
                        selector_frequency_hit_abs=float(
                            _base_sel_cfg.get("selector_frequency_hit_abs", 1e-6)
                        ),
                        selector_interaction_mode=str(
                            _base_sel_cfg.get(
                                "selector_interaction_mode", "tree_path_lift"
                            )
                        ),
                        selector_interaction_topk_pairs=int(
                            _base_sel_cfg.get("selector_interaction_topk_pairs", 100)
                        ),
                        selector_interaction_max_pairs_per_feature=int(
                            _base_sel_cfg.get(
                                "selector_interaction_max_pairs_per_feature", 8
                            )
                        ),
                        selector_interaction_corr_penalty=bool(
                            _base_sel_cfg.get("selector_interaction_corr_penalty", True)
                        ),
                        selector_family_penalty=bool(
                            _base_sel_cfg.get("selector_family_penalty", True)
                        ),
                        selector_emit_report=bool(
                            _base_sel_cfg.get("selector_emit_report", True)
                        ),
                        analysis_n_estimators=int(
                            _base_sel_cfg.get("analysis_n_estimators", 192)
                        ),
                        analysis_max_samples=int(
                            _base_sel_cfg.get("analysis_max_samples", 3000)
                        ),
                        min_samples_leaf_pct=float(
                            _base_sel_cfg.get("min_samples_leaf_pct", 0.015)
                        ),
                        selector_max_missing_frac=float(
                            _base_sel_cfg.get("selector_max_missing_frac", 0.15)
                        ),
                        selector_near_constant_dominance=float(
                            _base_sel_cfg.get("selector_near_constant_dominance", 0.999)
                        ),
                        selector_hysteresis_margin=float(
                            _base_sel_cfg.get("selector_hysteresis_margin", 0.05)
                        ),
                        selector_min_overlap=float(
                            _base_sel_cfg.get("selector_min_overlap", 0.70)
                        ),
                        composite_weights={
                            "top30": float(_base_sel_cfg.get("top30", 0.0)),
                            "global": float(_base_sel_cfg.get("global", 0.55)),
                            "stability": float(_base_sel_cfg.get("stability", 0.25)),
                            "frequency": float(_base_sel_cfg.get("frequency", 0.15)),
                            "interaction": float(
                                _base_sel_cfg.get("interaction", 0.05)
                            ),
                        },
                        end_features=60,
                        cumulative_cap=0.99,
                        min_share=0.0005,
                        min_features=30,
                        max_features_pct=0.8,
                    )

                    selected_feats = _cap_selected_features(
                        list(sel_res.selected_features),
                        list(X.columns),
                        target_cap=int(cfg.get("base_mdi_target_features", 60)),
                        min_features=int(cfg.get("base_mdi_min_features", 30)),
                    )
                    top5_feats, top10_feats = _mdi_top_feature_lists(
                        sel_res, selected_feats
                    )
                    feature_selection_by_h[H] = list(selected_feats)
                    tprint(
                        f"MDI selected {len(selected_feats)} features (from {X.shape[1]}) for H={H}."
                    )
                    _log_feature_matrix_state(
                        f"Base post-MDI [{key}]",
                        X.iloc[_fit_idx][selected_feats] if selected_feats else X.iloc[_fit_idx][[]],
                        selected_feats,
                    )
                    if not _feature_guard_ok(f"Base post-MDI [{key}]", selected_feats):
                        continue

                    X_sel = X.iloc[_fit_idx][selected_feats]
                    y_fit = y[_fit_idx]
                    y_ret_fit = y_ret[_fit_idx]
                    w_fit = w[_fit_idx]
                    cols = list(selected_feats)

                    y_hard_check = (y_fit >= 0.5).astype(int)
                    tprint(
                        f"  Class dist: 0={int((y_hard_check==0).sum())} ({(y_hard_check==0).mean()*100:.1f}%), "
                        f"1={int((y_hard_check==1).sum())} ({(y_hard_check==1).mean()*100:.1f}%)"
                    )

                    groups = (
                        df["__ts__"].values[_fit_idx]
                        if "__ts__" in df.columns
                        else None
                    )
                    symbols = (
                        df["__symbol__"].values[_fit_idx]
                        if "__symbol__" in df.columns
                        else None
                    )
                    race = _fit_direct_extratrees_base_model(
                        kind_name=k,
                        X=X_sel,
                        y=y_fit,
                        sample_weight=w_fit,
                        returns=y_ret_fit,
                        groups=groups,
                        symbols=symbols,
                        n_splits=2,
                        cfg=cfg,
                    )
                    score = race.metrics.get(race.best_model_name, -1.0)
                    dm = race.detailed_metrics.get(race.best_model_name, {})
                    degeneracy_info = dm.get("degeneracy", {})
                    is_degenerate = bool(degeneracy_info.get("is_degenerate", False))
                    # Race CV AUC = fold-averaged AUC during model selection
                    # OOF AUC = AUC on full post-calibration OOF vector (canonical)
                    oof_auc_canonical = 0.5
                    if race.oof_probs is not None:
                        y_bin_canon = (y >= 0.5).astype(np.int8)
                        if len(np.unique(y_bin_canon)) > 1:
                            from sklearn.metrics import roc_auc_score as _roc_auc

                            oof_auc_canonical = float(
                                _roc_auc(y_bin_canon, race.oof_probs)
                            )
                    # --- Alpha model OOF diagnostics (all metrics from same post-calibration oof) ---
                    per_regime = {}
                    oof_bss = dm.get("BSS", 0)  # fallback to race BSS
                    _bs_oof = float("nan")
                    _prev_global = float("nan")
                    if race.oof_probs is not None:
                        oof = race.oof_probs
                        y_bin_oof = (y >= 0.5).astype(np.int8)
                        # Recompute BSS from post-calibration OOF (same probs as all other metrics)
                        from sklearn.metrics import brier_score_loss as _bsl

                        _prev_global = float(np.mean(y_bin_oof))
                        _p_clip = np.clip(oof, 1e-7, 1 - 1e-7)
                        try:
                            _bs_oof = float(_bsl(y_bin_oof, _p_clip))
                            _bs_ref_global = float(
                                _bsl(y_bin_oof, np.full_like(_p_clip, _prev_global))
                            )
                            _bs_ref_global = max(_bs_ref_global, 1e-6)
                            oof_bss = 1.0 - (_bs_oof / _bs_ref_global)
                            if not np.isfinite(oof_bss):
                                oof_bss = 0.0
                        except Exception:
                            _bs_oof = 0.0
                            oof_bss = 0.0

                    tprint(
                        f"Finished {side} {k} H={H}: Winner={race.best_model_name}, Score={score:.4f}, "
                        f"RcAUC={dm.get('AUC',0):.4f}, OOF_AUC={oof_auc_canonical:.4f}, "
                        f"RcIC={dm.get('IC',0):.4f}, RcBSS={dm.get('BSS',0):.4f}, "
                        f"OOF_Brier={_bs_oof:.4f}, OOF_BSS={oof_bss:.4f}"
                    )
                    if is_degenerate:
                        tprint(
                            f"  CRITICAL: base model degenerate for {side}/{k}/H={H}; "
                            f"reasons={degeneracy_info.get('reasons', [])} "
                            f"raw={degeneracy_info.get('raw', {})} "
                            f"cal={degeneracy_info.get('calibrated', {})}"
                        )
                        tprint(
                            f"  Degenerate config context ({side}/{k}/H={H}): "
                            f"selected={len(selected_feats)} top10={top10_feats[:10]}"
                        )

                    if race.oof_probs is not None:
                        tprint(
                            f"  OOF probs [post-cal]: mean={np.mean(oof):.4f}, std={np.std(oof):.4f}, "
                            f"min={np.min(oof):.4f}, max={np.max(oof):.4f}"
                        )
                        tprint(
                            f"  OOF Brier={_bs_oof:.4f}, prev={_prev_global:.4f}, BSS={oof_bss:.4f}"
                        )
                        # OOF-based return correlation (key signal quality metric)
                        if np.std(oof) > 1e-9 and np.std(y_ret) > 1e-9:
                            oof_ret_corr = float(np.corrcoef(oof, y_ret)[0, 1])
                            tprint(f"  OOF-return correlation: {oof_ret_corr:.4f}")
                        # Calibration: mean predicted prob vs actual positive rate
                        tprint(
                            f"  Calibration: mean_pred={np.mean(oof):.4f} vs actual_rate={_prev_global:.4f}"
                        )
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
                        tprint(
                            f"  Alpha calibration: ECE@10={ece10:.4f} profile={profile}"
                        )

                        # --- Per-regime BSS/AUC (unweighted, with both bucket and global baselines) ---
                        per_regime = compute_per_regime_metrics(
                            y, oof, df, global_prev=_prev_global
                        )
                        if per_regime:
                            tprint(
                                f"  Per-regime BSS/AUC/Brier ({len(per_regime)} dimensions):"
                            )
                            for rname, rbuckets in per_regime.items():
                                parts = []
                                for bl, bm in rbuckets.items():
                                    bss_g = bm.get("bss_global", 0)
                                    parts.append(
                                        f"{bl}(n={bm['n']}): BSS={bm['bss']:.3f} BSS_g={bss_g:.3f} AUC={bm['auc']:.3f} Brier={bm.get('brier',0):.4f}"
                                    )
                                tprint(f"    {rname}: {' | '.join(parts)}")
                        if "__n_res__" in df.columns:
                            n_res_vals = np.clip(
                                df["__n_res__"].values.astype(float), 0.0, None
                            )
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
                                auc_res = (
                                    float(
                                        roc_auc_score(
                                            y_bin_oof, oof, sample_weight=resolved_w
                                        )
                                    )
                                    if len(np.unique(y_bin_oof)) > 1
                                    else 0.5
                                )
                            except Exception:
                                auc_res = 0.5
                            tprint(
                                f"  AUC reporting ({side}/{k}/H={H}): raw={auc_all:.4f}, resolved-weighted={auc_res:.4f}"
                            )

                    alpha_diag = {}
                    if race.best_model_name in race.detailed_metrics:
                        dm_best = race.detailed_metrics[race.best_model_name]
                        alpha_diag = {
                            "prec10": float(dm_best.get("Prec10", np.nan)),
                            "prec40": float(dm_best.get("Prec40", np.nan)),
                            "ece_top10": float(dm_best.get("ece_top10", np.nan)),
                            "calibration_profile": dm_best.get(
                                "calibration_profile", "n/a"
                            ),
                            "degenerate": bool(
                                dm_best.get("degeneracy", {}).get("is_degenerate", False)
                            ),
                            "degeneracy_reasons": list(
                                dm_best.get("degeneracy", {}).get("reasons", [])
                            ),
                        }
                    if race.oof_probs is not None:
                        groups_v = (
                            df["__ts__"].values[_fit_idx]
                            if "__ts__" in df.columns
                            else None
                        )
                        alpha_diag["avg_trades_day_10"] = float(
                            _avg_trades_per_day_global(race.oof_probs, 0.10, groups_v)
                        )
                        alpha_diag["avg_trades_day_30"] = float(
                            _avg_trades_per_day_global(race.oof_probs, 0.30, groups_v)
                        )
                        oof_metrics = _aggregate_alpha_oof_metrics(
                            y_fit,
                            race.oof_probs,
                            y_ret_fit,
                            sample_weight=w_fit,
                            groups=groups_v,
                        )
                        alpha_diag.update(oof_metrics)

                    # Economic gate: positive realized expectancy on top-k OOF selection
                    _econ_ok = True
                    _econ_mean = float("nan")
                    _econ_top_frac = float(
                        cfg.get("base_oof_expectancy_top_frac", 0.30)
                    )
                    if race.oof_probs is not None and bool(
                        cfg.get("base_require_positive_oof_expectancy", True)
                    ):
                        _k_top = max(
                            1, int(np.ceil(_econ_top_frac * len(race.oof_probs)))
                        )
                        _idx_top = np.argsort(race.oof_probs)[-_k_top:]
                        _econ_mean = float(
                            np.mean(np.asarray(y_ret_fit, dtype=float)[_idx_top])
                        )
                        _econ_ok = bool(_econ_mean > 0.0)
                        tprint(
                            f"  Base economic gate ({side}/{k}/H={H}): top{int(_econ_top_frac*100)} mean_ret={_econ_mean:.6f} pass={_econ_ok}"
                        )

                    _deployable = bool(_econ_ok) and not is_degenerate
                    _score_for_select = score if _deployable else -1e12
                    if _score_for_select > best_ic:
                        best_ic = _score_for_select
                        best_m = {
                            "model": race,
                            "H": H,
                            "feat_cols": cols,
                            "per_regime": per_regime,
                            "alpha_diag": alpha_diag,
                        }

                    per_h_models[H] = {
                        "model": race,
                        "H": H,
                        "feat_cols": cols,
                        "per_regime": per_regime,
                        "alpha_diag": alpha_diag,
                        "top5_features": top5_feats,
                        "top10_features": top10_feats,
                        "score": score,
                        "econ_ok": bool(_econ_ok),
                        "degenerate": is_degenerate,
                        "degeneracy": degeneracy_info,
                        "deployable": _deployable,
                        "econ_top_mean_ret": float(_econ_mean)
                        if np.isfinite(_econ_mean)
                        else float("nan"),
                    }
                    if _deployable:
                        deployable_h_models[H] = per_h_models[H]

            # --- Multi-horizon deployment: all horizons are kept ---
            # best_m serves as the "primary" (highest score) for backward compat.
            # models_by_h stores all trained horizons for inference averaging.
            strategy_degenerate = bool(per_h_models) and not bool(deployable_h_models)
            if best_m is None and per_h_models and not strategy_degenerate:
                _fallback_h = max(
                    deployable_h_models,
                    key=lambda h: deployable_h_models[h].get("score", -1e12),
                )
                best_m = deployable_h_models[_fallback_h]
                tprint(
                    f"  WARNING: {side}_{k}: all deployable horizons failed selection score gate; "
                    f"falling back to best deployable horizon H={_fallback_h}"
                )
            if best_m is not None:
                best_m["models_by_h"] = {
                    h: {
                        "model": v["model"],
                        "feat_cols": v["feat_cols"],
                        "H": v["H"],
                        "selected_features": feature_selection_by_h.get(
                            h, v["feat_cols"]
                        ),
                        "top5_features": v.get("top5_features", [])[:5],
                        "top10_features": v.get("top10_features", [])[:10],
                    }
                    for h, v in deployable_h_models.items()
                }
                best_m["all_models_by_h"] = {
                    h: {
                        "model": v["model"],
                        "feat_cols": v["feat_cols"],
                        "H": v["H"],
                        "selected_features": feature_selection_by_h.get(
                            h, v["feat_cols"]
                        ),
                        "top5_features": v.get("top5_features", [])[:5],
                        "top10_features": v.get("top10_features", [])[:10],
                        "degenerate": bool(v.get("degenerate", False)),
                        "degeneracy": v.get("degeneracy", {}),
                        "deployable": bool(v.get("deployable", False)),
                    }
                    for h, v in per_h_models.items()
                }
                best_m["downstream_blocked"] = False
                tprint(
                    f"  {side}_{k}: Deploying {len(deployable_h_models)} horizons: {sorted(deployable_h_models.keys())} "
                    f"(primary H={best_m['H']})"
                )
            elif strategy_degenerate:
                degenerate_strategy_ids.add(k)
                _fallback_h = max(
                    per_h_models, key=lambda h: per_h_models[h].get("score", -1e12)
                )
                best_m = dict(per_h_models[_fallback_h])
                best_m["models_by_h"] = {}
                best_m["all_models_by_h"] = {
                    h: {
                        "model": v["model"],
                        "feat_cols": v["feat_cols"],
                        "H": v["H"],
                        "selected_features": feature_selection_by_h.get(
                            h, v["feat_cols"]
                        ),
                        "top5_features": v.get("top5_features", [])[:5],
                        "top10_features": v.get("top10_features", [])[:10],
                        "degenerate": bool(v.get("degenerate", False)),
                        "degeneracy": v.get("degeneracy", {}),
                        "deployable": False,
                    }
                    for h, v in per_h_models.items()
                }
                best_m["downstream_blocked"] = True
                best_m.setdefault("alpha_diag", {})
                best_m["alpha_diag"]["degenerate_strategy"] = True
                best_m["alpha_diag"]["blocked_reason"] = "all_base_horizons_degenerate"
                tprint(
                    f"CRITICAL: Blocking downstream training for {side}_{k}; "
                    f"all base horizons are degenerate."
                )

            # --- Save OOF predictions for fast meta loading + richer diagnostics ---
            # Memory optimization: build payload incrementally, avoid intermediate arrays
            _run_id = cfg.get("run_id", "default")
            oof_dir = os.path.join(cfg["data_root"], "artifacts", _run_id, "oof")
            os.makedirs(oof_dir, exist_ok=True)
            for _h, _v in deployable_h_models.items():
                _race = _v["model"]
                if _race.oof_probs is None:
                    continue

                _oof_path = os.path.join(oof_dir, f"oof_{k}_H{_h}.parquet")
                _gate_key = f"train_{k}_{_h}"
                _df_oof = datasets.get(_gate_key)
                _n = int(len(_race.oof_probs))
                if _df_oof is not None:
                    _n = min(_n, int(len(_df_oof)))

                # Use float32 for OOF probs to halve memory vs float64
                _payload = {
                    "oof_prob": np.asarray(_race.oof_probs, dtype=np.float32)[:_n],
                    "index": np.arange(_n, dtype=np.int32),
                }
                _dm_best = (
                    _race.detailed_metrics.get(_race.best_model_name, {})
                    if hasattr(_race, "detailed_metrics")
                    else {}
                )
                for _sigma_key in ("oof_sigma_trees", "oof_sigma_robust"):
                    _sigma_vals = _dm_best.get(_sigma_key)
                    if _sigma_vals is not None and len(_sigma_vals) >= _n:
                        _payload[_sigma_key] = np.asarray(
                            _sigma_vals, dtype=np.float32
                        )[:_n]

                if _df_oof is not None:
                    if "__ts__" in _df_oof.columns:
                        _payload["timestamp"] = _normalize_oof_timestamps_to_numpy(
                            _df_oof["__ts__"]
                        )[:_n]
                    elif "timestamp" in _df_oof.columns:
                        _payload["timestamp"] = _normalize_oof_timestamps_to_numpy(
                            _df_oof["timestamp"]
                        )[:_n]

                    if "__symbol__" in _df_oof.columns:
                        _payload["symbol"] = (
                            _df_oof["__symbol__"].astype(str).values[:_n]
                        )
                    elif "symbol" in _df_oof.columns:
                        _payload["symbol"] = _df_oof["symbol"].astype(str).values[:_n]
                    elif "asset" in _df_oof.columns:
                        _payload["symbol"] = _df_oof["asset"].astype(str).values[:_n]

                    if "__y_bin__" in _df_oof.columns:
                        _payload["y_bin"] = np.asarray(
                            _df_oof["__y_bin__"], dtype=np.float32
                        )[:_n]
                    if "__y_ret__" in _df_oof.columns:
                        _payload["y_ret"] = np.asarray(
                            _df_oof["__y_ret__"], dtype=np.float32
                        )[:_n]

                # Only store oof_raw if explicitly configured (saves ~50% storage per model)
                if cfg.get("save_oof_raw", False):
                    _dm_best = (
                        _race.detailed_metrics.get(_race.best_model_name, {})
                        if hasattr(_race, "detailed_metrics")
                        else {}
                    )
                    _oof_raw = _dm_best.get("oof_raw")
                    if _oof_raw is not None and len(_oof_raw) >= _n:
                        _payload["oof_raw"] = np.asarray(_oof_raw, dtype=np.float32)[
                            :_n
                        ]

                pd.DataFrame(_payload).to_parquet(_oof_path, index=False)

            # --- Save each ModelRace in native format (fast load) ---
            models_dir = os.path.join(
                cfg["data_root"], "artifacts", _run_id, "models", "native"
            )
            for _h, _v in deployable_h_models.items():
                _race = _v["model"]
                _model_dir = os.path.join(models_dir, f"{side}_{k}_H{_h}")
                _race.save_native(_model_dir)
                import json as _json

                with open(os.path.join(_model_dir, "columns.json"), "w") as _cf:
                    _json.dump(
                        {
                            "feat_cols": _v.get("feat_cols", []),
                            "selected_features": _v.get(
                                "selected_features", _v.get("feat_cols", [])
                            ),
                        },
                        _cf,
                    )
            # Strip for pickle fallback
            for _h, _v in deployable_h_models.items():
                _v["model"].strip_for_serialization()

            # --- Stage Gate Check (Alpha) — per horizon ---
            for _gate_H, _gate_v in deployable_h_models.items():
                _gate_race = _gate_v["model"]
                _gate_key = f"train_{k}_{_gate_H}"
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
                for _ in range(25):
                    _ib = _rng.choice(_n, size=_n, replace=True)
                    _nk = max(1, int(_n * _kf))
                    top_idx = np.argsort(_gate_oof[_ib])[-_nk:]
                    p_k_b = float(np.mean(_yb_hard[_ib][top_idx]))
                    _prec_samples.append(p_k_b)
                _pa = np.array(_prec_samples)
                _cv20 = float(np.std(_pa) / (np.mean(_pa) + 1e-9))
                tprint(f"  {side}_{k} H={_gate_H}: Bootstrap CV(Prec@20)={_cv20:.3f}")

                gate_res = compute_stage_gate_metrics(
                    _gate_y,
                    _gate_oof,
                    _gate_yret,
                    model_type="classifier",
                    cv_prec10=_cv20,
                )
                gate_res["Model"] = f"{side}_{k}_H{_gate_H}"
                alpha_gate_results.append(gate_res)

            if best_m is not None:
                final_models[side][k] = best_m

        if bool(cfg.get("base_geometry_train_variants", True)):
            tprint("Training grouped base-geometry variant models (tight/wide)...")
            _run_id = cfg.get("run_id", "default")
            oof_dir = os.path.join(cfg["data_root"], "artifacts", _run_id, "oof")
            os.makedirs(oof_dir, exist_ok=True)
            strategies = get_strategies(cfg)
            for strat in strategies:
                side = strat["trade_side"]
                k = strat["strategy_id"]
                for H in strategy_horizons.get(k, []):
                    for variant in cfg.get(
                        "base_geometry_archetypes", ["tight", "wide"]
                    ):
                        variant = str(variant)
                        ds_key = f"train_{k}_{H}_{variant}"
                        if ds_key not in datasets:
                            continue
                        tprint(f"Training grouped base variant {ds_key}...")
                        _variant_fit = _train_base_variant_dataset(
                            side, k, H, ds_key, datasets[ds_key], strategy=strat
                        )
                        if _variant_fit is None:
                            tprint(f"  Skipped {ds_key}: insufficient data")
                            continue
                        base_variant_models[(side, k, int(H), variant)] = _variant_fit
                        _race = _variant_fit["model"]
                        if _race.oof_probs is None:
                            continue
                        _df_oof = datasets[ds_key]
                        _n = min(int(len(_race.oof_probs)), int(len(_df_oof)))
                        _payload = {
                            "oof_prob": np.asarray(_race.oof_probs, dtype=np.float32)[
                                :_n
                            ],
                            "index": np.arange(_n, dtype=np.int32),
                        }
                        _dm_best = (
                            _race.detailed_metrics.get(_race.best_model_name, {})
                            if hasattr(_race, "detailed_metrics")
                            else {}
                        )
                        for _sigma_key in ("oof_sigma_trees", "oof_sigma_robust"):
                            _sigma_vals = _dm_best.get(_sigma_key)
                            if _sigma_vals is not None and len(_sigma_vals) >= _n:
                                _payload[_sigma_key] = np.asarray(
                                    _sigma_vals, dtype=np.float32
                                )[:_n]
                        if "__ts__" in _df_oof.columns:
                            _payload["timestamp"] = _normalize_oof_timestamps_to_numpy(
                                _df_oof["__ts__"]
                            )[:_n]
                        if "__symbol__" in _df_oof.columns:
                            _payload["symbol"] = (
                                _df_oof["__symbol__"].astype(str).values[:_n]
                            )
                        if "__y_bin__" in _df_oof.columns:
                            _payload["y_bin"] = np.asarray(
                                _df_oof["__y_bin__"], dtype=np.float32
                            )[:_n]
                        if "__y_ret__" in _df_oof.columns:
                            _payload["y_ret"] = np.asarray(
                                _df_oof["__y_ret__"], dtype=np.float32
                            )[:_n]
                            _oof_path = os.path.join(
                                oof_dir, f"oof_{side}_{k}_H{int(H)}_{variant}.parquet"
                            )
                            pd.DataFrame(_payload).to_parquet(_oof_path, index=False)
                            tprint(f"  Saved grouped base OOF to {_oof_path}")

    # Save base models intermediate for train_meta mode (only when we actually trained them)
    if train_base:
        _run_id = cfg.get("run_id", "default")
        _intermediate_path = os.path.join(
            cfg["data_root"], "artifacts", _run_id, "base_models_intermediate.pkl"
        )
        os.makedirs(os.path.dirname(_intermediate_path), exist_ok=True)
        import pickle as _pkl_save

        _base_bundle = {
            "alpha_models": final_models,
            "base_variant_models": base_variant_models,
            "spike_models": spike_models,
            "specialist_models": specialist_models,
            "blocked_strategy_ids": sorted(str(s) for s in degenerate_strategy_ids),
        }
        with open(_intermediate_path, "wb") as _f:
            _pkl_save.dump(_base_bundle, _f)
        tprint(f"Base models intermediate saved to {_intermediate_path}")

    # Load base models if train_base=False (meta-only mode)
    if not train_base:
        import pickle as _pkl_load

        _run_id = cfg.get("run_id", "default")
        _intermediate_path = os.path.join(
            cfg["data_root"], "artifacts", _run_id, "base_models_intermediate.pkl"
        )
        if not os.path.exists(_intermediate_path):
            tprint(
                f"ERROR: Base models intermediate not found at {_intermediate_path}. Cannot train meta models without base models."
            )
            return None
        with open(_intermediate_path, "rb") as _f:
            _base_bundle = _pkl_load.load(_f)
        final_models = _base_bundle.get("alpha_models", {})
        base_variant_models = _base_bundle.get("base_variant_models", {})
        spike_models = _base_bundle.get("spike_models", {})
        specialist_models = _base_bundle.get("specialist_models", {})
        tprint(f"Loaded base models from {_intermediate_path}")

    # 3. Train Meta Models (One per Alpha Model: Side x Kind)
    if train_meta:
        meta_models, meta_gate_results = train_meta_models_from_artifacts(
            datasets, cfg, final_models, base_variant_models=base_variant_models
        )
    else:
        meta_models, meta_gate_results = {}, []

    # Exhaustion Models (already trained at step 1.6)

    # --- Stage Gate Reporting ---
    tprint("\n=== Stage Gate Report: Alpha Models (Classifiers) ===")
    alpha_pass_count = 0
    if alpha_gate_results:
        df_alpha_gate = pd.DataFrame(alpha_gate_results)
        cols_order = ["Model", "passed", "PR_AUC", "Brier_Imp", "Lift_k", "CV_Prec_k"]
        # Print main columns
        tprint(
            df_alpha_gate[
                [c for c in cols_order if c in df_alpha_gate.columns]
            ].to_string(index=False)
        )
        alpha_pass_count = df_alpha_gate["passed"].sum()
    else:
        tprint("No Alpha models evaluated.")

    n_alpha_models = len(alpha_gate_results) if alpha_gate_results else 0
    alpha_half = max(1, n_alpha_models // 2)
    tprint(
        f"\nAlpha Stage: {alpha_pass_count}/{n_alpha_models} passed (need {alpha_half})."
    )
    if alpha_pass_count < alpha_half:
        tprint(f"WARNING: Alpha Stage FAILED (< {alpha_half} models passed).")

    tprint("\n=== Stage Gate Report: Meta Models (Quantile) ===")
    meta_pass_count = 0
    if meta_gate_results:
        df_meta_gate = pd.DataFrame(meta_gate_results)
        cols_order = [
            "Model",
            "passed",
            "Coverage_Diff",
            "Pinball_Imp",
            "Spearman_IC",
            "Pass_Spread",
            "Pass_Downside",
        ]
        tprint(
            df_meta_gate[
                [c for c in cols_order if c in df_meta_gate.columns]
            ].to_string(index=False)
        )
        meta_pass_count = df_meta_gate["passed"].sum()
    else:
        tprint("No Meta models evaluated.")

    n_meta_models = len(meta_gate_results) if meta_gate_results else 0
    meta_half = max(1, n_meta_models // 2)
    tprint(
        f"\nMeta Stage: {meta_pass_count}/{n_meta_models} passed (need {meta_half})."
    )
    if meta_pass_count < meta_half:
        tprint(f"WARNING: Meta Stage FAILED (< {meta_half} models passed).")

    # Extended per-model quality report (base + meta) — per horizon per bucket.
    base_quality_rows = []
    strategies = get_strategies(cfg)
    for strat in strategies:
        side = strat["trade_side"]
        kind = strat["strategy_id"]
        conf = final_models.get(side, {}).get(kind)
        if not conf:
            continue
        models_by_h = conf.get("models_by_h", {})
        if not models_by_h:
            # Fallback: single-model legacy format
            models_by_h = {
                conf.get("H", 4): {
                    "model": conf["model"],
                    "feat_cols": conf["feat_cols"],
                    "top5_features": conf.get("top5_features", []),
                    "top10_features": conf.get("top10_features", []),
                    "deployable": not bool(conf.get("downstream_blocked", False)),
                }
            }
        for H_rep, h_info in models_by_h.items():
            ds_key = f"train_{kind}_{H_rep}"
            if ds_key not in datasets:
                continue
            race = h_info["model"]
            if race.oof_probs is None:
                continue
            dfm = datasets[ds_key]
            y_bin = (dfm["__y_bin__"].values >= 0.5).astype(int)
            y_ret = dfm["__y_ret__"].values.astype(float)
            y_lbl = (
                dfm["__y_lbl__"].values.astype(int)
                if "__y_lbl__" in dfm.columns
                else None
            )
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
                y_bin_model, oof_probs_model, y_ret_model = (
                    y_bin[:n],
                    oof_probs[:n],
                    y_ret[:n],
                )
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
                    top5_features=h_info.get("top5_features", []),
                    top10_features=h_info.get("top10_features", []),
                )
                entry["H"] = H_rep
                entry["is_winner"] = cand_name == race.best_model_name
                entry["downstream_blocked"] = bool(conf.get("downstream_blocked", False))
                entry["variant"] = "primary"
                # Canonical primary rows are kept internally but are not part of the
                # reportable geometry-variant surface. The report should focus on
                # the explicit _tight / _wide models only.
                continue

    for (side, kind, horizon, variant), variant_info in (base_variant_models or {}).items():
        race = variant_info.get("model")
        if race is None or race.oof_probs is None:
            continue
        ds_key = f"train_{kind}_{int(horizon)}_{variant}"
        if ds_key not in datasets:
            continue
        dfm = datasets[ds_key]
        y_bin = (dfm["__y_bin__"].values >= 0.5).astype(int)
        y_ret = dfm["__y_ret__"].values.astype(float)
        y_lbl = (
            dfm["__y_lbl__"].values.astype(int)
            if "__y_lbl__" in dfm.columns
            else None
        )
        groups = dfm["__ts__"].values if "__ts__" in dfm.columns else None
        for cand_name, dm in race.detailed_metrics.items():
            oof_probs = dm.get("oof_probs")
            if oof_probs is None:
                oof_probs = np.asarray(race.oof_probs, dtype=float)
            else:
                oof_probs = np.asarray(oof_probs, dtype=float)
            n = min(len(y_bin), len(oof_probs), len(y_ret))
            entry = _base_model_report_entry(
                model_name=f"{side}_{kind}_H{int(horizon)}_{variant}:{cand_name}",
                side=side,
                kind=kind,
                dm=dm,
                y_bin=y_bin[:n],
                oof_probs=oof_probs[:n],
                y_ret=y_ret[:n],
                groups=np.asarray(groups)[:n] if groups is not None else None,
                y_lbl=y_lbl[:n] if y_lbl is not None else None,
                top5_features=variant_info.get("top5_features", []),
                top10_features=variant_info.get("top10_features", []),
            )
            entry["H"] = int(horizon)
            entry["variant"] = str(variant)
            entry["is_winner"] = cand_name == race.best_model_name
            entry["downstream_blocked"] = False
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
                if best_rec is None or rec.get("composite_score", 0) > best_rec.get(
                    "composite_score", 0
                ):
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
            meta_quality_rows.append(
                {
                    "model": key,
                    "passed": True,
                    "metrics": clf_metrics,
                }
            )
            continue

        conf = final_models.get(side, {}).get(kind)
        if not conf:
            continue
        models_by_h = conf.get("models_by_h", {})
        # Collect available horizon OOFs (same logic as train_meta)
        _h_oofs = {}
        for h in CANON_HORIZONS:
            ds_key = f"train_{kind}_{h}"
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
            _oof_parts.append(
                oof_h[: len(dfm)]
                if len(oof_h) >= len(dfm)
                else np.pad(oof_h, (0, len(dfm) - len(oof_h)), constant_values=0.5)
            )
        base_score = np.mean(_oof_parts, axis=0)

        def _aligned_ret(h):
            """Get __y_ret__ for horizon h, aligned to len(dfm)."""
            k = f"train_{kind}_{h}"
            if k not in datasets:
                return y_ret.copy()
            arr = datasets[k]["__y_ret__"].values.astype(float)
            if len(arr) >= len(dfm):
                return arr[: len(dfm)]
            return np.pad(arr, (0, len(dfm) - len(arr)), constant_values=0.0)

        _r2_rpt, _r4_rpt, _r8_rpt = _aligned_ret(2), _aligned_ret(4), _aligned_ret(8)

        # Extract vol_proxy from base df (largest horizon usually) for reporting metrics
        # Note: 'dfm' is the largest horizon dataset.
        _vol_proxy_rpt = (
            dfm["__barrier_pct__"].values.astype(float)
            if "__barrier_pct__" in dfm.columns
            else None
        )
        if _vol_proxy_rpt is not None and len(_vol_proxy_rpt) < len(dfm):
            _vol_proxy_rpt = np.pad(
                _vol_proxy_rpt,
                (0, len(dfm) - len(_vol_proxy_rpt)),
                constant_values=0.02,
            )

        y_target = compute_meta_target(
            _r2_rpt, _r4_rpt, _r8_rpt, vol_proxy=_vol_proxy_rpt, groups=None
        )
        _y_per_h_rpt = {2: _r2_rpt, 4: _r4_rpt, 8: _r8_rpt}
        groups = dfm["__ts__"].values if "__ts__" in dfm.columns else None
        n = min(
            len(y_ret),
            len(y_target),
            len(base_score),
            len(meta.oof_probs) if meta.oof_probs is not None else 0,
        )
        if n <= 10:
            continue
        y_ret, y_target, base_score = y_ret[:n], y_target[:n], base_score[:n]
        _y_per_h_rpt = {h: v[:n] for h, v in _y_per_h_rpt.items()}
        if groups is not None:
            groups = np.asarray(groups)[:n]
        meta_quality_rows.append(
            _meta_report_entry(
                key,
                meta,
                y_target,
                y_ret,
                base_score,
                groups,
                y_per_horizon=_y_per_h_rpt,
            )
        )

    winners_base = sorted(
        [r for r in base_quality_rows if r.get("is_winner")],
        key=lambda x: x.get("score", -1e9),
        reverse=True,
    )
    others_base = sorted(
        [r for r in base_quality_rows if not r.get("is_winner")],
        key=lambda x: x.get("score", -1e9),
        reverse=True,
    )
    winners_meta = sorted(
        [r for r in meta_quality_rows if r.get("passed")],
        key=lambda x: x.get("metrics", {}).get("spearman_ic", -1e9),
        reverse=True,
    )
    others_meta = sorted(
        [r for r in meta_quality_rows if not r.get("passed")],
        key=lambda x: x.get("metrics", {}).get("spearman_ic", -1e9),
        reverse=True,
    )

    gate_report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_models": winners_base + others_base,
        "meta_models": winners_meta + others_meta,
        "blocked_strategy_ids": sorted(
            {
                r.get("kind")
                for r in base_quality_rows
                if bool(r.get("downstream_blocked", False))
            }
        ),
        "winners": {
            "base": [r["model"] for r in winners_base],
            "meta": [r["model"] for r in winners_meta],
        },
    }
    _run_id = cfg.get("run_id") or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    save_quality_gate_artifacts(
        gate_report,
        cfg,
        _run_id,
        base_quality_rows=base_quality_rows,
        meta_quality_rows=meta_quality_rows,
    )
    print_training_gate_report(gate_report)

    # Build alpha metrics correctly from dynamic strategies
    alpha_metrics = {}
    for side, side_models in final_models.items():
        for kind, kind_model in side_models.items():
            alpha_metrics[f"{side}_{kind}"] = (
                kind_model.get("alpha_diag", {}) if kind_model is not None else {}
            )

    return {
        "alpha_models": final_models,
        "alpha_oof_metrics": alpha_metrics,
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


def optimize_risk_params(
    panel, feats, mkt_gates, cfg, train_syms, ts, p_exh_hist, alpha_models
):
    tprint(
        "Entering function: optimize_risk_params in training.py (High Throughput Selection)"
    )

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

    # We need to gather events for each bucket.
    # This is non-trivial because `optimize_risk_params` is usually called on a small simulation window.
    # But `run_tp_sl_selection_fast` is designed for training time selection on historical data.
    # Assuming `ts` is the end of training.

    # We will scan the last N days (e.g. 90 or 180) for candidates.
    lookback_days = 90
    # Select candidates
    cand_mask, _, mask_by_strategy = _build_optimal_candidate_mask(panel, feats, cfg)
    if cand_mask is None:
        tprint("No candidates found.")
        return cfg

    # Ensure ts is a Timestamp and handle potential timezone mismatch with index
    ts = pd.Timestamp(ts)
    if cand_mask.index.tz is not None:
        if ts.tz is None:
            ts = ts.tz_localize("UTC").tz_convert(cand_mask.index.tz)
        else:
            ts = ts.tz_convert(cand_mask.index.tz)
    else:
        if ts.tz is not None:
            ts = ts.tz_localize(None)

    ts_start = ts - pd.Timedelta(days=lookback_days)

    # Select candidates
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

    # Memory optimization: limit pooled optimization to a representative subset (top 150 assets)
    # processing 600+ assets over 90 days causes OOM during flattening.
    assets = [sym for sym in close_df.columns if sym in atr_pct_df.columns]
    missing_atr_count = len(close_df.columns) - len(assets)
    if missing_atr_count > 0:
        tprint(
            f"Skipping {missing_atr_count} symbols without atr_pct in feature cache "
            f"(eligible={len(assets)})."
        )
    if len(assets) > 150:
        tprint(
            f"Limiting risk optimization pooled assets from {len(assets)} to 150 to prevent OOM."
        )
        # Assuming assets are already somewhat volume-ordered or just pick a stable subset
        assets = assets[:150]
    if not assets:
        tprint(
            "No ATR-eligible assets available for risk optimization; skipping optimization step."
        )
        return cfg

    # Collect arrays (1h resolution — always needed for features/ATR)
    big_open = []
    big_high = []
    big_low = []
    big_close = []
    big_atr = []
    big_z = []
    big_atr_base = []
    big_X = []  # Features

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
    buffer_size = 100  # larger than horizon
    buffer_size_15m = 400  # 100 hours * 4

    # We need features too.
    # Let's pick a standard set of features for X
    feat_keys = cfg.get("causal_cols", [])
    if not feat_keys:
        # Fallback
        feat_keys = ["trend_pct", "vol_pct", "ret24h"]

    for sym in assets:
        if sym not in atr_pct_df.columns:
            continue

        # Get data chunks
        o = open_df[sym].values.astype(np.float32)
        h = high_df[sym].values.astype(np.float32)
        l = low_df[sym].values.astype(np.float32)
        c = close_df[sym].values.astype(np.float32)

        a = np.nan_to_num(
            atr_pct_df[sym].reindex(close_df.index).values.astype(np.float32), nan=0.01
        )
        b = np.nan_to_num(
            atr_base_df[sym].reindex(close_df.index).values.astype(np.float32), nan=0.01
        )
        z_v = np.nan_to_num(
            z_df[sym].reindex(close_df.index).values.astype(np.float32), nan=0.0
        )

        # Features
        # Gather into (T, F) — reindex to panel index to ensure length match
        panel_idx = close_df.index
        x_list = []
        for k in feat_keys:
            if k in feats and sym in feats[k].columns:
                x_list.append(
                    np.nan_to_num(
                        feats[k][sym].reindex(panel_idx).values.astype(np.float32),
                        nan=0.0,
                    )
                )
            else:
                x_list.append(np.zeros(len(c), dtype=np.float32))
        x_arr = (
            np.stack(x_list, axis=1)
            if x_list
            else np.zeros((len(c), 1), dtype=np.float32)
        )

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
                    ts_start_utc = ts_start_utc.tz_localize("UTC")
                else:
                    ts_start_utc = ts_start_utc.tz_convert("UTC")

                df_15m = get_15m_ohlcv(exchange, ccxt_sym, ts_start_utc, window_hours)
                if (
                    not df_15m.empty and len(df_15m) >= len(c) * 3
                ):  # at least 75% coverage
                    o15 = df_15m["open"].values.astype(np.float32)
                    h15 = df_15m["high"].values.astype(np.float32)
                    l15 = df_15m["low"].values.astype(np.float32)
                    c15 = df_15m["close"].values.astype(np.float32)

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
                    tprint(
                        f"  {sym}: 15m data insufficient ({len(df_15m) if not df_15m.empty else 0} bars), using 1h"
                    )
            except Exception as e:
                tprint(f"  {sym}: 15m download failed: {e}")

    if not big_open:
        tprint("No pooled optimization arrays were built; skipping optimization step.")
        return cfg

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
        tprint(
            f"15m data: {len(asset_offsets_15m)}/{len(asset_offsets)} assets, {len(full_close_15m)} total bars"
        )

    # Now iterate strategies and collect event indices
    strategies = get_strategies(cfg)
    for strat in strategies:
        side = strat["trade_side"]
        k = strat["strategy_id"]
        trade_side = side
        cand_filter, move_bucket, strategy_label = _strategy_bucket_context(
            trade_side, k, cfg
        )
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
            trend_vals = (
                feats["trend_pct"].reindex(columns=cands).loc[t]
                if t in feats["trend_pct"].index
                else pd.Series(0.0, index=cands)
            )

            for sym in cands:
                tv = trend_vals[sym]
                if not _trend_direction_keep_mask([tv], trend_filter)[0]:
                    continue

                # Found a candidate
                # Get index in full arrays
                try:
                    tidx = close_df.index.get_loc(t)
                except KeyError:
                    continue

                flat_idx = asset_offsets[sym] + tidx
                indices.append(flat_idx)
                time_indices.append(
                    tidx
                )  # true temporal coordinate (same for all assets at time t)

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
                full_atr[indices],
                full_z[indices],
                full_atr_base[indices],
                z_max=3.0,
                lo=0.03,
                hi=0.06,
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
            is_mr_default = strat.get("is_mr", False)
            default_risk = {
                "tp_mult": cfg.get("tp_mult", 1.0),
                "sl_mult": cfg.get("sl_mult", 0.5),
                "trail_mult": cfg.get("trail_mult", 0.5),
                "vol_lo": cfg.get("vol_lo", 0.03),
                "vol_hi": cfg.get("vol_hi", 0.06),
                "vol_z_max": cfg.get("vol_z_max", 3.0),
                "max_hold_hours": 12 if is_mr_default else 24,
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

        _tp_opt = str(cfg.get("tp_sl_search_optimizer", "legacy")).lower()
        if _tp_opt == "new":
            from types import SimpleNamespace as _SNS

            from .position_sizer.tp_sl_selection import (
                CompositeObjectiveConfig as _COCfg,
            )
            from .position_sizer.tp_sl_selection import (
                select_best_tp_sl as _select_best_tp_sl,
            )

            _cocfg = _COCfg(
                mar=float(cfg.get("objective_mar", 0.0)),
                eps_log=float(cfg.get("objective_eps_log", 1e-12)),
                eps_sortino=float(cfg.get("objective_eps_sortino", 1e-12)),
                mode=str(cfg.get("objective_composite_mode", "hard_gate")),
                q_top=float(cfg.get("objective_composite_q_top", 0.95)),
                selection=str(cfg.get("objective_composite_selection", "min_std")),
                min_trades_per_fold=int(
                    cfg.get("tp_sl_search_min_trades_per_fold", 200)
                ),
                elg_scale=float(cfg.get("objective_scaling_elg_scale", 10000.0)),
                mnpt_scale=float(cfg.get("objective_scaling_mnpt_scale", 10000.0)),
                elg_min=float(cfg.get("objective_clipping_elg_min", -1.0)),
                elg_max=float(cfg.get("objective_clipping_elg_max", 1.0)),
                sortino_min=float(cfg.get("objective_clipping_sortino_min", -10.0)),
                sortino_max=float(cfg.get("objective_clipping_sortino_max", 10.0)),
                mnpt_min=float(cfg.get("objective_clipping_mnpt_min", -1.0)),
                mnpt_max=float(cfg.get("objective_clipping_mnpt_max", 1.0)),
            )
            _sel = _select_best_tp_sl(
                open_=full_open,
                close=full_close,
                event_idx=indices,
                timestamps=time_indices,
                tp_mult_grid=cfg.get("tp_sl_search_k_tp_grid", [0.8, 1.0, 1.25, 1.5]),
                sl_mult_grid=cfg.get("tp_sl_search_k_sl_grid", [0.1, 0.15, 0.25, 0.4]),
                cfg=_cocfg,
            )
            _best = _sel.get("best") or {}
            _cand = _best.get(
                "candidate", (cfg.get("tp_mult", 1.0), cfg.get("sl_mult", 0.5))
            )
            summary = _SNS(
                final_tp_mult=float(_cand[0]),
                final_sl_mult=float(_cand[1]),
                final_trail_mult=float(cfg.get("trail_mult", 0.25)),
                final_lo=float(cfg.get("vol_lo", 0.02)),
                final_hi=float(cfg.get("vol_hi", 0.06)),
                final_z_max=float(cfg.get("vol_z_max", 3.0)),
                outer_results=[],
            )
        else:
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
                fee_bps=float(cfg.get("fee_bps", 25.0)),
                **_15m_kwargs,
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
            tprint(
                f"  Enforced min TP: tp_mult raised to {_tp_m:.2f} (abs {_tp_m*_opt_bp_risk*100:.1f}%)"
            )
        if _tp_m / max(_sl_m, 0.01) < min_ratio:
            _sl_m = _tp_m / min_ratio
            tprint(f"  Enforced min ratio: sl_mult lowered to {_sl_m:.2f}")
        summary.final_tp_mult = _tp_m
        summary.final_sl_mult = _sl_m
        tprint(
            f"Optimized {side} {k}: TP={summary.final_tp_mult:.2f} ({summary.final_tp_mult*_opt_bp_risk*100:.1f}%), "
            f"SL={summary.final_sl_mult:.2f} ({summary.final_sl_mult*_opt_bp_risk*100:.1f}%), "
            f"ratio={summary.final_tp_mult/max(summary.final_sl_mult,0.01):.1f}x, "
            f"Trail={summary.final_trail_mult:.2f}, Lo={summary.final_lo:.2f}, Hi={summary.final_hi:.2f}, Zmax={summary.final_z_max:.2f}"
        )

        if summary.outer_results:
            avg_auc = np.mean([r.test_auc for r in summary.outer_results])
            avg_ic = np.mean([r.test_ic for r in summary.outer_results])
            avg_pnl = np.mean([r.test_pnl for r in summary.outer_results])
            tprint(
                f"  Avg Test Metrics: AUC={avg_auc:.4f}, IC={avg_ic:.4f}, PnL={avg_pnl:.4f}"
            )

            pairs = [
                (
                    r.chosen_tp_mult,
                    r.chosen_sl_mult,
                    r.chosen_trail_mult,
                    r.chosen_lo,
                    r.chosen_hi,
                    r.chosen_z_max,
                )
                for r in summary.outer_results
            ]
            tprint(f"  Stability (Chosen Configs): {pairs}")

        # Per-bucket max hold hours: MR = shorter (reversion is fast), TF = longer
        is_mr = strat.get("is_mr", False)
        bucket_hold = 12 if is_mr else 24

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
        if is_mr:
            be_pct = max(0.003, mfe_p25)
        else:
            be_pct = max(0.003, 0.5 * (mfe_p25 + mfe_med))

        # Profit-lock: trigger at p25 MFE (early protection — most winners reach this)
        lock_pct = max(0.005, mfe_p25)
        # Lock amount: lock 50% of p25 MFE as real profit
        lock_amt = max(0.002, 0.50 * mfe_p25)

        # Max giveback: exit if return drops more than 50-65% from peak MFE
        # MR: tighter giveback (reversion is fast), TF: slightly wider
        giveback_frac = 0.50 if is_mr else 0.60
        giveback_pct = max(0.003, giveback_frac * mfe_med)

        # Max loss: hard cap at mae_p75 (75th percentile of adverse excursion)
        max_loss = max(0.01, min(0.05, mae_p75))

        tprint(
            f"  {side}_{k} profit-protection (empirical MFE): "
            f"MFE p25={mfe_p25*100:.2f}% med={mfe_med*100:.2f}% p75={mfe_p75*100:.2f}% | "
            f"MAE med={mae_med*100:.2f}% p75={mae_p75*100:.2f}% | "
            f"BE@{be_pct*100:.2f}% Lock@{lock_pct*100:.2f}% LockAmt={lock_amt*100:.2f}% "
            f"Giveback={giveback_pct*100:.2f}% MaxLoss={max_loss*100:.2f}%"
        )

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
        "granular_risk": granular_risk,
    }

    return best_params
