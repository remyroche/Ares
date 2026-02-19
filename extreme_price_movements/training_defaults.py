"""Shared training-parameter defaults used by offline optimisers.

This module centralises how defaults are derived so offline optimisation scripts
stay aligned with the training pipeline parameters/fallbacks.
"""
from __future__ import annotations

from typing import Any, Dict

from .config import CFG


def _cfg_or_default(cfg: Dict[str, Any] | None) -> Dict[str, Any]:
    return cfg if isinstance(cfg, dict) else CFG


def get_candidate_filter_defaults(cfg: Dict[str, Any] | None = None) -> Dict[str, float]:
    c = _cfg_or_default(cfg)
    return {
        "train_extreme_pct_hourly": float(c.get("train_extreme_pct_hourly", c.get("trade_extreme_pct", 0.05))),
        "train_min_range_pct": float(c.get("train_min_range_pct", 0.07)),
        "train_min_vol_zscore": float(c.get("train_min_vol_zscore", 1.6)),
        "min_feat_sign_consistency": float(c.get("min_feat_sign_consistency", 0.70)),
    }


def get_barrier_factory_defaults(cfg: Dict[str, Any] | None = None) -> Dict[str, float]:
    c = _cfg_or_default(cfg)
    return {
        "barrier_k_tp": float(c.get("barrier_k_tp", 1.0)),
        "barrier_sl_base_mult": float(c.get("barrier_sl_base_mult", 0.5)),
        "barrier_disp_floor": float(c.get("barrier_disp_floor", 0.1)),
        "barrier_z_max": float(c.get("barrier_z_max", 3.0)),
        "barrier_k_reg": float(c.get("barrier_k_reg", 0.3)),
        "barrier_m_lo": float(c.get("barrier_m_lo", 0.7)),
        "barrier_m_hi": float(c.get("barrier_m_hi", 1.5)),
        "barrier_sl_lo": float(c.get("barrier_sl_lo", 0.4)),
        "barrier_sl_hi": float(c.get("barrier_sl_hi", 0.7)),
        "barrier_z_gate": float(c.get("barrier_z_gate", 1.0)),
        "barrier_tp_lo": float(c.get("barrier_tp_lo", 0.02)),
        "barrier_tp_hi": float(c.get("barrier_tp_hi", 0.06)),
        "label_horizon_base": float(c.get("label_horizon_base", 4.0)),
    }


def get_sample_weight_opt_defaults(cfg: Dict[str, Any] | None = None) -> Dict[str, Any]:
    c = _cfg_or_default(cfg)
    return {
        "sample_weight_opt_model_family": str(c.get("sample_weight_opt_model_family", "ExtraTrees")),
        "sample_weight_opt_n_splits": int(c.get("sample_weight_opt_n_splits", 5)),
        "sample_weight_opt_embargo_bars": int(c.get("sample_weight_opt_embargo_bars", 10)),
        "sample_weight_opt_min_n_eff_ratio": float(c.get("sample_weight_opt_min_n_eff_ratio", 0.30)),
        "sample_weight_opt_max_top1pct": float(c.get("sample_weight_opt_max_top1pct", 0.10)),
        "sample_weight_vol_power": float(c.get("sample_weight_vol_power", 0.5)),
        "sample_weight_distance_k": float(c.get("sample_weight_distance_k", 0.5)),
        "sample_weight_distance_min_dist": float(c.get("sample_weight_distance_min_dist", 0.5)),
        "sample_weight_recency_half_life_bars": int(c.get("sample_weight_recency_half_life_bars", 24 * 30)),
    }


def get_target_race_model_defaults(cfg: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Training-aligned defaults for target-race OOF model stack."""
    c = _cfg_or_default(cfg)
    return {
        "ridge_screen_alpha": float(c.get("target_race_ridge_alpha", 0.5)),
        "ridge_screen_top_frac": float(c.get("target_race_ridge_top_frac", 0.25)),
        "et_params": {
            "n_estimators": int(c.get("target_race_et_n_estimators", 200)),
            "max_depth": int(c.get("target_race_et_max_depth", 6)),
            "min_samples_leaf": int(c.get("target_race_et_min_samples_leaf", 30)),
            "max_features": c.get("target_race_et_max_features", "sqrt"),
            "n_jobs": int(c.get("target_race_et_n_jobs", 3)),
            "random_state": int(c.get("target_race_et_random_state", 42)),
        },
    }


def get_tbm_optimizer_defaults(cfg: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Defaults for non-optimized TBM optimizer knobs sourced from runtime training config."""
    c = _cfg_or_default(cfg)
    barrier = get_barrier_factory_defaults(c)
    return {
        "tp_abs_lo_pct": float(c.get("tbm_tp_abs_lo_pct", barrier["barrier_tp_lo"])),
        "tp_abs_hi_pct": float(c.get("tbm_tp_abs_hi_pct", barrier["barrier_tp_hi"])),
        "sl_abs_lo_pct": float(c.get("tbm_sl_abs_lo_pct", barrier["barrier_tp_lo"])),
        "sl_abs_hi_pct": float(c.get("tbm_sl_abs_hi_pct", barrier["barrier_tp_hi"])),
        "horizon_base": int(c.get("tbm_horizon_base", barrier["label_horizon_base"])),
        "horizon_alpha": float(c.get("tbm_horizon_alpha", 0.5)),
        "tp_abs_pct": float(c.get("tbm_tp_abs_pct", 0.02)),
        "tp_base_pct": float(c.get("tbm_tp_base_pct", 0.02)),
        "base_atr_window": int(c.get("tbm_base_atr_window", 168)),
        "fee_pct": float(c.get("tbm_fee_pct", c.get("fee_pct", 0.5))),
        "slip_buffer": float(c.get("tbm_slip_buffer", c.get("slip_buffer", 0.1))),
    }


def get_sample_weight_eval_model_defaults(cfg: Dict[str, Any] | None = None) -> Dict[str, Dict[str, Any]]:
    """Evaluation-model defaults used by sample-weight optimisation CV scoring."""
    c = _cfg_or_default(cfg)
    return {
        "extratrees": {
            "n_estimators": int(c.get("sample_weight_eval_et_n_estimators", 50)),
            "max_depth": int(c.get("sample_weight_eval_et_max_depth", 6)),
            "min_samples_leaf": int(c.get("sample_weight_eval_et_min_samples_leaf", 50)),
            "max_features": c.get("sample_weight_eval_et_max_features", "sqrt"),
            "n_jobs": int(c.get("sample_weight_eval_et_n_jobs", 2)),
        },
        "randomforest": {
            "n_estimators": int(c.get("sample_weight_eval_rf_n_estimators", 80)),
            "max_depth": int(c.get("sample_weight_eval_rf_max_depth", 6)),
            "min_samples_leaf": int(c.get("sample_weight_eval_rf_min_samples_leaf", 50)),
            "max_features": c.get("sample_weight_eval_rf_max_features", "sqrt"),
            "n_jobs": int(c.get("sample_weight_eval_rf_n_jobs", 2)),
        },
    }
