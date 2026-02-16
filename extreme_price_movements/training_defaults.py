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
