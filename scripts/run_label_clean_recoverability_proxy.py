#!/usr/bin/env python3
"""Rare-event clean-recoverability proxy for label candidates.

This is a pre-production diagnostic. It asks whether causal features can learn
the clean executable positive class itself, before running full base/meta model
training. For each validation month it fits cheap month-forward tree proxies on
prior months only, then evaluates selected rows after policy costs.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_label_dual_target_execution_smoke import _rank_pct, _selection_weekly_rows
from scripts.run_label_economic_proxy_ablation import LABEL_ARMS, _label_targets
from scripts.run_label_feature_store_model_smoke import _add_delta_fields, _fit_predict, _month_model_frame
from scripts.run_label_quality_proxy_diagnostics import (
    DEFAULT_FEATURE_DIR,
    DEFAULT_FEATURE_LIST_CSV,
    DEFAULT_LABELS_DIR,
    _decile_diagnostics,
    _feature_columns,
    _json_safe,
    _load_feature_store_columns,
    _load_labels,
    _path_metrics,
    _read_feature_list,
    _safe_mean,
    _safe_quantile,
    _selection_metrics,
    _spearman,
)


DEFAULT_OUTPUT_DIR = Path("data_perp/reports/label_clean_recoverability_proxy_v1")
DEFAULT_LABEL_ARMS = (
    "S49_clean_recoverable_tail_rank",
    "S50_s30_clean_recoverable_tail",
)
DEFAULT_TARGET_MODES = ("hard_clean", "soft_hard_blend")
DEFAULT_WEIGHT_MODES = ("balanced", "dirty_focus", "hard_vs_dirty")
DEFAULT_DIRTY_PENALTIES = (0.0, 0.25, 0.50)
DEFAULT_TOP_FRACS = (0.0025, 0.005)
DEFAULT_SEEDS = (42, 7301, 999)
DERIVED_FEATURE_MODES = ("none", "clean_recoverability_v1", "path_risk_v2")


def _parse_csv(value: str | None, default: tuple[str, ...]) -> list[str]:
    if value is None or not str(value).strip():
        return list(default)
    lowered = str(value).strip().lower()
    if lowered == "all":
        return []
    return [part.strip() for part in str(value).split(",") if part.strip()]


def _parse_int_csv(value: str | None, default: tuple[int, ...]) -> list[int]:
    if value is None or not str(value).strip():
        return list(default)
    return [int(part.strip()) for part in str(value).split(",") if part.strip()]


def _parse_float_csv(value: str | None, default: tuple[float, ...]) -> list[float]:
    if value is None or not str(value).strip():
        return list(default)
    return [float(part.strip()) for part in str(value).split(",") if part.strip()]


def _baseline_row(valid_metrics: pd.DataFrame) -> dict[str, float]:
    return {
        "period_baseline_mean_u": _safe_mean(valid_metrics["u_policy_net"]),
        "period_baseline_hit_u": _safe_mean(valid_metrics["u_policy_net"] > 0.0),
        "period_baseline_q10_u": _safe_quantile(valid_metrics["u_policy_net"], 0.10),
    }


def _target_for_mode(target: pd.DataFrame, mode: str) -> pd.Series:
    hard = pd.to_numeric(target["target_hard"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    soft = pd.to_numeric(target["target_soft"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    if mode == "hard_clean":
        return hard
    if mode == "soft_hard_blend":
        return (0.65 * hard + 0.35 * soft).clip(0.0, 1.0)
    if mode == "hard_gated_soft":
        return soft.where(hard > 0.0, 0.0).clip(0.0, 1.0)
    raise ValueError(f"Unknown target mode: {mode}")


def _dirty_target(metrics: pd.DataFrame) -> pd.Series:
    return (
        (pd.to_numeric(metrics["u_policy_net"], errors="coerce").fillna(0.0) <= 0.0)
        | (pd.to_numeric(metrics["mae_norm"], errors="coerce").fillna(0.0) >= 1.0)
        | (pd.to_numeric(metrics["barrier"], errors="coerce").fillna(0.0) > 0.025)
        | (metrics["is_timeout"].astype(bool))
    ).astype(float)


def _feature_series(frame: pd.DataFrame, name: str) -> pd.Series | None:
    if name not in frame.columns:
        return None
    values = pd.to_numeric(frame[name], errors="coerce").replace([np.inf, -np.inf], np.nan)
    if int(values.notna().sum()) < 100:
        return None
    return values


def _fixed_tanh(values: pd.Series, *, center: float = 0.0, scale: float = 1.0) -> pd.Series:
    scale = float(scale)
    if abs(scale) < 1e-6:
        scale = 1e-6 if scale >= 0.0 else -1e-6
    return pd.Series(
        np.tanh((pd.to_numeric(values, errors="coerce") - float(center)) / scale),
        index=values.index,
    )


def _add_derived_clean_recoverability_features(frame: pd.DataFrame, *, mode: str) -> pd.DataFrame:
    if mode == "none":
        return pd.DataFrame(index=frame.index)
    if mode not in DERIVED_FEATURE_MODES:
        raise ValueError(f"Unknown derived feature mode: {mode}")

    base: dict[str, pd.Series] = {}

    def add_base(name: str, source: pd.Series | None) -> None:
        if source is None:
            return
        values = pd.to_numeric(source, errors="coerce").replace([np.inf, -np.inf], np.nan)
        if int(values.notna().sum()) < 100 or int(values.nunique(dropna=True)) < 3:
            return
        base[name] = values.astype(np.float32)

    adx10 = _feature_series(frame, "adx_10")
    adx7 = _feature_series(frame, "adx_7")
    oi_rank = _feature_series(frame, "oi_rank")
    zscore = _feature_series(frame, "zscore_price_200")
    prev_week_loc = _feature_series(frame, "loc_prev_week_range_pos_24")
    local_loc = _feature_series(frame, "loc_range_pos_24")
    resist = _feature_series(frame, "distance_to_resistance_daily_vwap_atr")
    support = _feature_series(frame, "distance_to_support_daily_vwap_atr")
    compression = _feature_series(frame, "atr_compression_ratio")
    entry_7d = _feature_series(frame, "oiw_pos_delta_entry_dist_7d_atr")
    entry_14d = _feature_series(frame, "oiw_pos_delta_entry_dist_14d_atr")
    dn_vol = _feature_series(frame, "dn_vol")
    decel = _feature_series(frame, "decel_8")
    memory = _feature_series(frame, "memory_asymmetry_1ATR")
    climax = _feature_series(frame, "climax_vol_12")

    add_base("cr_adx10_high_fixed", _fixed_tanh(adx10, center=4.0, scale=1.0) if adx10 is not None else None)
    add_base("cr_adx7_high_fixed", _fixed_tanh(adx7, center=4.0, scale=1.0) if adx7 is not None else None)
    add_base("cr_oi_rank_low", _fixed_tanh(oi_rank, center=0.25, scale=-0.20) if oi_rank is not None else None)
    add_base("cr_zscore_low", _fixed_tanh(zscore, center=0.0, scale=-2.0) if zscore is not None else None)
    add_base(
        "cr_prev_week_loc_low",
        _fixed_tanh(prev_week_loc, center=0.15, scale=-0.20) if prev_week_loc is not None else None,
    )
    add_base("cr_local_loc_low", _fixed_tanh(local_loc, center=0.15, scale=-0.20) if local_loc is not None else None)
    add_base("cr_resistance_room", _fixed_tanh(resist, center=0.0, scale=2.0) if resist is not None else None)
    if support is not None:
        add_base("cr_support_abs_near", -pd.to_numeric(support, errors="coerce").abs().clip(upper=5.0) / 5.0)
    if support is not None and resist is not None:
        add_base("cr_vwap_room_asym", _fixed_tanh(resist - support, center=0.0, scale=2.0))
    add_base("cr_compression_high", _fixed_tanh(compression, center=0.0, scale=1.0) if compression is not None else None)
    add_base("cr_entry_7d_high", _fixed_tanh(entry_7d, center=0.0, scale=2.0) if entry_7d is not None else None)
    add_base("cr_entry_14d_high", _fixed_tanh(entry_14d, center=0.0, scale=2.0) if entry_14d is not None else None)
    add_base("cr_dn_vol_pressure", _fixed_tanh(dn_vol, center=0.0, scale=1.0) if dn_vol is not None else None)
    add_base("cr_decel_high", _fixed_tanh(decel, center=0.0, scale=1.0) if decel is not None else None)
    add_base("cr_memory_high", _fixed_tanh(memory, center=1.0, scale=1.0) if memory is not None else None)
    add_base("cr_climax_high", _fixed_tanh(climax, center=0.0, scale=1.0) if climax is not None else None)

    interaction_specs = [
        ("cr_adx10_x_oi_low", "cr_adx10_high_fixed", "cr_oi_rank_low"),
        ("cr_adx10_x_zscore_low", "cr_adx10_high_fixed", "cr_zscore_low"),
        ("cr_adx10_x_prev_week_low", "cr_adx10_high_fixed", "cr_prev_week_loc_low"),
        ("cr_adx10_x_entry7d", "cr_adx10_high_fixed", "cr_entry_7d_high"),
        ("cr_adx10_x_resistance_room", "cr_adx10_high_fixed", "cr_resistance_room"),
        ("cr_adx10_x_support_near", "cr_adx10_high_fixed", "cr_support_abs_near"),
        ("cr_adx10_x_compression", "cr_adx10_high_fixed", "cr_compression_high"),
        ("cr_adx10_x_dnvol", "cr_adx10_high_fixed", "cr_dn_vol_pressure"),
        ("cr_adx10_x_decel", "cr_adx10_high_fixed", "cr_decel_high"),
        ("cr_adx10_x_memory", "cr_adx10_high_fixed", "cr_memory_high"),
        ("cr_oi_low_x_zscore_low", "cr_oi_rank_low", "cr_zscore_low"),
        ("cr_oi_low_x_prev_week_low", "cr_oi_rank_low", "cr_prev_week_loc_low"),
        ("cr_zscore_low_x_resistance_room", "cr_zscore_low", "cr_resistance_room"),
        ("cr_vwap_asym_x_oi_low", "cr_vwap_room_asym", "cr_oi_rank_low"),
        ("cr_entry7d_x_compression", "cr_entry_7d_high", "cr_compression_high"),
    ]
    for name, left, right in interaction_specs:
        if left in base and right in base:
            add_base(name, base[left] * base[right])

    if mode == "path_risk_v2":
        trend_r2 = _feature_series(frame, "trend_r2_24")
        trend_z = _feature_series(frame, "trend_z_t")
        trend_accel = _feature_series(frame, "trend_acceleration")
        range_12h = _feature_series(frame, "range_12h_pct")
        range_24h = _feature_series(frame, "range_24h_pct")
        range_pct = _feature_series(frame, "range_pct")
        vol_compression = _feature_series(frame, "vol_compression")
        wick = _feature_series(frame, "wick_ratio_4h_max")
        body = _feature_series(frame, "body_pct")
        rejection = _feature_series(frame, "rejection_proxy")
        spread_abs = _feature_series(frame, "spread_proxy_abs_return_bps_robust_z")
        spread_hl = _feature_series(frame, "spread_proxy_hl_range_bps_robust_z")
        dir_edge = _feature_series(frame, "dir_path_edge_2h")
        dir_risk_long = _feature_series(frame, "dir_path_risk_long_2h")
        dir_risk_short = _feature_series(frame, "dir_path_risk_short_2h")
        dir_risk_skew = _feature_series(frame, "dir_path_risk_skew_2h")
        leverage = _feature_series(frame, "leverage_build_score")
        flow = _feature_series(frame, "flow_ratio")
        churn = _feature_series(frame, "churn")
        stall = _feature_series(frame, "cumulative_delta_stall")
        memory2 = _feature_series(frame, "memory_asymmetry_2ATR")
        memory3 = _feature_series(frame, "memory_asymmetry_3ATR")
        ema20_dist = _feature_series(frame, "dist_ema20_atr")
        ema200_dist = _feature_series(frame, "dist_ema200_atr")
        swing_dist = _feature_series(frame, "dist_local_swing")
        prior_low_dist = _feature_series(frame, "dist_prior_day_low")
        breakout = _feature_series(frame, "breakout_24h")
        confirmed = _feature_series(frame, "breakout_confirmed")
        innovation = _feature_series(frame, "innovation_z_x_zr_3h")
        jump = _feature_series(frame, "jump_intensity")

        add_base("pr_trend_r2_high", _fixed_tanh(trend_r2, center=0.35, scale=0.18) if trend_r2 is not None else None)
        add_base("pr_trend_r2_low", _fixed_tanh(trend_r2, center=0.20, scale=-0.15) if trend_r2 is not None else None)
        if trend_z is not None:
            trend_z_num = pd.to_numeric(trend_z, errors="coerce")
            add_base("pr_trend_z_abs_low", -trend_z_num.abs().clip(upper=5.0) / 5.0)
            add_base("pr_trend_z_abs_high", trend_z_num.abs().clip(upper=5.0) / 5.0)
        add_base("pr_trend_accel_high", _fixed_tanh(trend_accel, center=0.0, scale=1.0) if trend_accel is not None else None)
        add_base("pr_range12_low", _fixed_tanh(range_12h, center=0.0, scale=-1.0) if range_12h is not None else None)
        add_base("pr_range24_low", _fixed_tanh(range_24h, center=0.0, scale=-1.0) if range_24h is not None else None)
        add_base("pr_range_pct_low", _fixed_tanh(range_pct, center=0.0, scale=-1.0) if range_pct is not None else None)
        add_base(
            "pr_vol_compression_high",
            _fixed_tanh(vol_compression, center=0.0, scale=1.0) if vol_compression is not None else None,
        )
        add_base("pr_wick_low", _fixed_tanh(wick, center=1.0, scale=-0.75) if wick is not None else None)
        add_base("pr_wick_high", _fixed_tanh(wick, center=1.0, scale=0.75) if wick is not None else None)
        add_base("pr_body_high", _fixed_tanh(body, center=0.45, scale=0.20) if body is not None else None)
        add_base("pr_rejection_low", _fixed_tanh(rejection, center=0.0, scale=-1.0) if rejection is not None else None)
        add_base("pr_spread_abs_low", _fixed_tanh(spread_abs, center=0.0, scale=-1.5) if spread_abs is not None else None)
        add_base("pr_spread_abs_high", _fixed_tanh(spread_abs, center=0.0, scale=1.5) if spread_abs is not None else None)
        add_base("pr_spread_hl_low", _fixed_tanh(spread_hl, center=0.0, scale=-1.5) if spread_hl is not None else None)
        add_base("pr_spread_hl_high", _fixed_tanh(spread_hl, center=0.0, scale=1.5) if spread_hl is not None else None)
        add_base("pr_dir_edge_high", _fixed_tanh(dir_edge, center=0.0, scale=1.0) if dir_edge is not None else None)
        add_base(
            "pr_dir_long_risk_low",
            _fixed_tanh(dir_risk_long, center=0.0, scale=-1.0) if dir_risk_long is not None else None,
        )
        add_base(
            "pr_dir_short_risk_low",
            _fixed_tanh(dir_risk_short, center=0.0, scale=-1.0) if dir_risk_short is not None else None,
        )
        if dir_risk_skew is not None:
            skew = pd.to_numeric(dir_risk_skew, errors="coerce")
            add_base("pr_dir_risk_skew_abs_low", -skew.abs().clip(upper=5.0) / 5.0)
        add_base("pr_leverage_low", _fixed_tanh(leverage, center=0.0, scale=-1.0) if leverage is not None else None)
        add_base("pr_leverage_high", _fixed_tanh(leverage, center=0.0, scale=1.0) if leverage is not None else None)
        add_base("pr_flow_abs_low", -pd.to_numeric(flow, errors="coerce").abs().clip(upper=5.0) / 5.0 if flow is not None else None)
        add_base("pr_churn_low", _fixed_tanh(churn, center=0.0, scale=-1.0) if churn is not None else None)
        add_base("pr_stall_low", _fixed_tanh(stall, center=0.0, scale=-1.0) if stall is not None else None)
        add_base("pr_memory2_high", _fixed_tanh(memory2, center=1.0, scale=1.0) if memory2 is not None else None)
        add_base("pr_memory3_high", _fixed_tanh(memory3, center=1.0, scale=1.0) if memory3 is not None else None)
        if ema20_dist is not None:
            add_base("pr_ema20_abs_low", -pd.to_numeric(ema20_dist, errors="coerce").abs().clip(upper=5.0) / 5.0)
        if ema200_dist is not None:
            add_base("pr_ema200_abs_low", -pd.to_numeric(ema200_dist, errors="coerce").abs().clip(upper=8.0) / 8.0)
        if swing_dist is not None:
            add_base("pr_swing_abs_low", -pd.to_numeric(swing_dist, errors="coerce").abs().clip(upper=5.0) / 5.0)
        add_base("pr_prior_low_near", _fixed_tanh(prior_low_dist, center=0.0, scale=-2.0) if prior_low_dist is not None else None)
        add_base("pr_breakout_high", _fixed_tanh(breakout, center=0.0, scale=1.0) if breakout is not None else None)
        add_base(
            "pr_breakout_confirmed_high",
            _fixed_tanh(confirmed, center=0.0, scale=1.0) if confirmed is not None else None,
        )
        add_base("pr_innovation_abs_low", -pd.to_numeric(innovation, errors="coerce").abs().clip(upper=5.0) / 5.0 if innovation is not None else None)
        add_base("pr_jump_low", _fixed_tanh(jump, center=0.0, scale=-1.0) if jump is not None else None)

        path_interactions = [
            ("pr_clean_trend_low_spread", "pr_trend_r2_high", "pr_spread_hl_low"),
            ("pr_clean_trend_body", "pr_trend_r2_high", "pr_body_high"),
            ("pr_clean_adx_spread", "cr_adx10_high_fixed", "pr_spread_hl_low"),
            ("pr_clean_adx_wick", "cr_adx10_high_fixed", "pr_wick_low"),
            ("pr_clean_dir_edge_spread", "pr_dir_edge_high", "pr_spread_abs_low"),
            ("pr_clean_dir_edge_body", "pr_dir_edge_high", "pr_body_high"),
            ("pr_clean_compression_spread", "pr_vol_compression_high", "pr_spread_hl_low"),
            ("pr_clean_compression_wick", "pr_vol_compression_high", "pr_wick_low"),
            ("pr_clean_low_flow_churn", "pr_flow_abs_low", "pr_churn_low"),
            ("pr_clean_low_leverage_stall", "pr_leverage_low", "pr_stall_low"),
            ("pr_clean_room_low_spread", "cr_resistance_room", "pr_spread_hl_low"),
            ("pr_clean_support_low_wick", "cr_support_abs_near", "pr_wick_low"),
            ("pr_dirty_chop_spread", "pr_trend_r2_low", "pr_spread_hl_high"),
            ("pr_dirty_chop_wick", "pr_trend_r2_low", "pr_wick_high"),
            ("pr_dirty_leverage_spread", "pr_leverage_high", "pr_spread_hl_high"),
            ("pr_dirty_rejection_wick", "pr_wick_high", "pr_rejection_low"),
            ("pr_dirty_jump_spread", "pr_jump_low", "pr_spread_abs_high"),
            ("pr_low_zscore_clean_path", "cr_zscore_low", "pr_spread_hl_low"),
            ("pr_low_zscore_body", "cr_zscore_low", "pr_body_high"),
            ("pr_low_zscore_dir_edge", "cr_zscore_low", "pr_dir_edge_high"),
        ]
        for name, left, right in path_interactions:
            if left in base and right in base:
                add_base(name, base[left] * base[right])

    if not base:
        return pd.DataFrame(index=frame.index)
    return pd.DataFrame(
        {name: values.astype(np.float32) for name, values in base.items()},
        index=frame.index,
    )


def _effective_sample_size(weights: pd.Series) -> float:
    values = pd.to_numeric(weights, errors="coerce").fillna(0.0).clip(lower=0.0)
    denom = float(np.square(values).sum())
    if denom <= 0.0:
        return 0.0
    return float(np.square(values.sum()) / denom)


def _clean_weights(
    *,
    clean_hard: pd.Series,
    dirty: pd.Series,
    mode: str,
    max_weight: float,
    min_weight: float,
) -> pd.Series:
    clean = pd.to_numeric(clean_hard, errors="coerce").fillna(0.0) > 0.5
    dirty_bool = pd.to_numeric(dirty, errors="coerce").fillna(0.0) > 0.5
    pos = int(clean.sum())
    neg = int((~clean).sum())
    balance = min(float(max_weight), float(neg / max(pos, 1)))
    weights = pd.Series(1.0, index=clean_hard.index, dtype=np.float64)
    if mode == "balanced":
        weights.loc[clean] = balance
    elif mode == "dirty_focus":
        weights.loc[clean] = balance
        weights.loc[(~clean) & dirty_bool] = min(float(max_weight), 2.5)
        weights.loc[(~clean) & (~dirty_bool)] = 0.50
    elif mode == "hard_vs_dirty":
        weights.loc[clean] = balance
        weights.loc[(~clean) & dirty_bool] = min(float(max_weight), max(2.0, 0.50 * balance))
        weights.loc[(~clean) & (~dirty_bool)] = 0.25
    else:
        raise ValueError(f"Unknown weight mode: {mode}")
    weights = weights.clip(lower=float(min_weight), upper=float(max_weight))
    mean = float(weights.mean())
    if math.isfinite(mean) and mean > 0.0:
        weights = weights / mean
    return weights.astype(np.float32)


def _seed_average_predict(
    *,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    w_train: pd.Series,
    x_valid: pd.DataFrame,
    seeds: list[int],
) -> tuple[pd.Series, float, float]:
    preds = [
        _fit_predict(
            x_train=x_train,
            y_train=y_train,
            w_train=w_train,
            x_valid=x_valid,
            seed=seed,
        )
        for seed in seeds
    ]
    matrix = np.vstack(preds)
    pred = np.mean(matrix, axis=0).astype(np.float32)
    std = np.std(matrix, axis=0).astype(np.float32) if len(preds) > 1 else np.zeros_like(pred)
    return pd.Series(pred), float(np.mean(std)), float(np.percentile(std, 90))


def _run_month(
    *,
    frame: pd.DataFrame,
    metrics: pd.DataFrame,
    targets: dict[str, pd.DataFrame],
    features: list[str],
    month: str,
    label_arms: list[str],
    target_modes: list[str],
    weight_modes: list[str],
    dirty_penalties: list[float],
    top_fracs: list[float],
    seeds: list[int],
    train_lookback_months: int | None,
    max_weight: float,
    min_weight: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    month_period = frame["__ts__"].dt.to_period("M").astype(str)
    train_mask = month_period < month
    if train_lookback_months is not None and int(train_lookback_months) > 0:
        prior_months = sorted(month_period[train_mask].dropna().unique())
        keep_months = set(prior_months[-int(train_lookback_months) :])
        train_mask = train_mask & month_period.isin(keep_months)
    valid_mask = month_period == month
    if int(train_mask.sum()) < 500 or int(valid_mask.sum()) < 100:
        return [], [], [
            {
                "period": month,
                "skipped": True,
                "train_rows": int(train_mask.sum()),
                "valid_rows": int(valid_mask.sum()),
            }
        ]

    x_train, x_valid = _month_model_frame(
        frame,
        train_mask=train_mask,
        valid_mask=valid_mask,
        features=features,
    )
    train = frame.loc[train_mask].copy()
    valid = frame.loc[valid_mask].copy().reset_index(drop=True)
    train_metrics = metrics.loc[train_mask].copy()
    valid_metrics = metrics.loc[valid_mask].copy().reset_index(drop=True)
    dirty_train = _dirty_target(train_metrics)
    dirty_valid = _dirty_target(valid_metrics).reset_index(drop=True)
    baseline = _baseline_row(valid_metrics)
    monthly_rows: list[dict[str, Any]] = []
    weekly_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []

    for label_arm in label_arms:
        target_train = targets[label_arm].loc[train_mask].copy()
        target_valid = targets[label_arm].loc[valid_mask].copy().reset_index(drop=True)
        clean_hard_train = pd.to_numeric(target_train["target_hard"], errors="coerce").fillna(0.0)
        clean_hard_valid = pd.to_numeric(target_valid["target_hard"], errors="coerce").fillna(0.0)
        for target_mode in target_modes:
            y_train = _target_for_mode(target_train, target_mode)
            for weight_mode in weight_modes:
                weights = _clean_weights(
                    clean_hard=clean_hard_train,
                    dirty=dirty_train,
                    mode=weight_mode,
                    max_weight=max_weight,
                    min_weight=min_weight,
                )
                clean_pred, clean_seed_std_mean, clean_seed_std_p90 = _seed_average_predict(
                    x_train=x_train,
                    y_train=y_train,
                    w_train=weights,
                    x_valid=x_valid,
                    seeds=seeds,
                )
                dirty_weights = _clean_weights(
                    clean_hard=dirty_train,
                    dirty=dirty_train,
                    mode="balanced",
                    max_weight=max_weight,
                    min_weight=min_weight,
                )
                dirty_pred, dirty_seed_std_mean, dirty_seed_std_p90 = _seed_average_predict(
                    x_train=x_train,
                    y_train=dirty_train,
                    w_train=dirty_weights,
                    x_valid=x_valid,
                    seeds=seeds,
                )
                clean_pred = clean_pred.reset_index(drop=True)
                dirty_pred = dirty_pred.reset_index(drop=True)
                clean_rank = _rank_pct(clean_pred)
                dirty_rank = _rank_pct(dirty_pred)
                diagnostic_rows.append(
                    {
                        "period": month,
                        "label_arm": label_arm,
                        "target_mode": target_mode,
                        "weight_arm": weight_mode,
                        "train_rows": int(train_mask.sum()),
                        "valid_rows": int(valid_mask.sum()),
                        "model_feature_count": int(len(features)),
                        "train_clean_hard_rate": _safe_mean(clean_hard_train),
                        "valid_clean_hard_rate": _safe_mean(clean_hard_valid),
                        "train_dirty_rate": _safe_mean(dirty_train),
                        "valid_dirty_rate": _safe_mean(dirty_valid),
                        "weight_mean": _safe_mean(weights),
                        "weight_p90": _safe_quantile(weights, 0.90),
                        "weight_p99": _safe_quantile(weights, 0.99),
                        "weight_effective_n": _effective_sample_size(weights),
                        "weight_effective_frac": _effective_sample_size(weights) / float(len(weights))
                        if len(weights)
                        else float("nan"),
                        "clean_ic_u": _spearman(clean_pred, valid_metrics["u_policy_net"]),
                        "clean_ic_label_soft": _spearman(clean_pred, target_valid["target_soft"]),
                        "clean_ic_label_hard": _spearman(clean_pred, clean_hard_valid),
                        "dirty_ic_actual": _spearman(dirty_pred, dirty_valid),
                        "clean_seed_std_mean": clean_seed_std_mean,
                        "clean_seed_std_p90": clean_seed_std_p90,
                        "dirty_seed_std_mean": dirty_seed_std_mean,
                        "dirty_seed_std_p90": dirty_seed_std_p90,
                    }
                )
                for dirty_penalty in dirty_penalties:
                    score = clean_rank - float(dirty_penalty) * dirty_rank
                    decile = _decile_diagnostics(score, valid_metrics["u_policy_net"])
                    for top_frac in top_fracs:
                        arm = (
                            f"{label_arm}::{weight_mode}::{target_mode}"
                            f"::dirty{dirty_penalty:.2f}"
                        )
                        row = _selection_metrics(
                            frame=valid,
                            metrics=valid_metrics,
                            target=target_valid,
                            score=score,
                            arm=arm,
                            selector="clean_recoverability_proxy_seed_ensemble_oos",
                            period=month,
                            top_frac=float(top_frac),
                        )
                        _add_delta_fields(row, baseline)
                        row.update(
                            {
                                "label_arm": label_arm,
                                "weight_arm": weight_mode,
                                "target_mode": target_mode,
                                "selection_mode": "clean_recoverability_proxy",
                                "mae_penalty": float(dirty_penalty),
                                "wide_penalty": 0.0,
                                "timeout_penalty": 0.0,
                                "mae_keep_frac": 1.0,
                                "wide_keep_frac": 1.0,
                                "timeout_keep_frac": 1.0,
                                "model_feature_count": int(len(features)),
                                "model_features": ",".join(features),
                                "score_ic_u": _spearman(score, valid_metrics["u_policy_net"]),
                                "score_ic_label": _spearman(score, target_valid["target_soft"]),
                                "score_ic_label_hard": _spearman(score, clean_hard_valid),
                                "dirty_ic_actual": _spearman(dirty_pred, dirty_valid),
                                **decile,
                            }
                        )
                        monthly_rows.append(row)
                        for weekly_row in _selection_weekly_rows(
                            frame=valid,
                            metrics=valid_metrics,
                            target=target_valid,
                            score=score,
                            arm=arm,
                            selector="clean_recoverability_proxy_seed_ensemble_oos",
                            period=month,
                            top_frac=float(top_frac),
                        ):
                            weekly_row.update(
                                {
                                    "label_arm": label_arm,
                                    "weight_arm": weight_mode,
                                    "target_mode": target_mode,
                                    "selection_mode": "clean_recoverability_proxy",
                                    "mae_penalty": float(dirty_penalty),
                                    "wide_penalty": 0.0,
                                    "timeout_penalty": 0.0,
                                    "mae_keep_frac": 1.0,
                                    "wide_keep_frac": 1.0,
                                    "timeout_keep_frac": 1.0,
                                    "model_feature_count": int(len(features)),
                                    "model_features": ",".join(features),
                                    "score_ic_u": _spearman(score, valid_metrics["u_policy_net"]),
                                    "score_ic_label": _spearman(score, target_valid["target_soft"]),
                                    "score_ic_label_hard": _spearman(score, clean_hard_valid),
                                }
                            )
                            weekly_rows.append(weekly_row)
    return monthly_rows, weekly_rows, diagnostic_rows


def _aggregate(monthly: pd.DataFrame) -> pd.DataFrame:
    if monthly.empty:
        return monthly
    rows: list[dict[str, Any]] = []
    group_cols = [
        "arm",
        "label_arm",
        "weight_arm",
        "target_mode",
        "selection_mode",
        "mae_penalty",
        "top_frac",
    ]
    for key, group in monthly.groupby(group_cols, observed=True, dropna=False):
        key_dict = dict(zip(group_cols, key))
        mean_u = pd.to_numeric(group["mean_u"], errors="coerce")
        selected_rows = pd.to_numeric(group["selected_rows"], errors="coerce")
        rows.append(
            {
                **key_dict,
                "months": int(group["period"].nunique()),
                "positive_months": int((mean_u > 0.0).sum()),
                "mean_u": _safe_mean(mean_u),
                "worst_month_mean_u": float(mean_u.min()) if len(mean_u.dropna()) else float("nan"),
                "hit_u": _safe_mean(group["hit_u"]),
                "q10_u": _safe_mean(group["q10_u"]),
                "target_top_hard_rate": _safe_mean(group["target_top_hard_rate"]),
                "score_ic_u": _safe_mean(group["score_ic_u"]),
                "score_ic_label": _safe_mean(group["score_ic_label"]),
                "score_ic_label_hard": _safe_mean(group["score_ic_label_hard"]),
                "dirty_ic_actual": _safe_mean(group["dirty_ic_actual"]),
                "bad_mae_1r_rate": _safe_mean(group["bad_mae_1r_rate"]),
                "wide_barrier_25bps_rate": _safe_mean(group["wide_barrier_25bps_rate"]),
                "timeout_rate": _safe_mean(group["timeout_rate"]),
                "top_symbol_share": _safe_mean(group["top_symbol_share"]),
                "mean_selected_rows": _safe_mean(selected_rows),
                "min_selected_rows": int(selected_rows.min()) if len(selected_rows.dropna()) else 0,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["top_frac", "mean_u", "worst_month_mean_u"],
        ascending=[True, False, False],
    )


def _write_markdown(output_dir: Path, aggregate: pd.DataFrame, manifest: dict[str, Any]) -> Path:
    path = output_dir / "label_clean_recoverability_proxy.md"

    def table(frame: pd.DataFrame, cols: list[str], limit: int | None = None) -> str:
        if frame.empty:
            return "No rows."
        view = frame[[col for col in cols if col in frame.columns]].copy()
        if limit is not None:
            view = view.head(limit)
        for col in view.columns:
            if pd.api.types.is_float_dtype(view[col]):
                view[col] = view[col].map(lambda value: f"{float(value):.4f}" if pd.notna(value) else "")
        return view.to_markdown(index=False)

    cols = [
        "label_arm",
        "weight_arm",
        "target_mode",
        "mae_penalty",
        "top_frac",
        "months",
        "positive_months",
        "mean_u",
        "worst_month_mean_u",
        "target_top_hard_rate",
        "score_ic_u",
        "score_ic_label_hard",
        "dirty_ic_actual",
        "bad_mae_1r_rate",
        "timeout_rate",
        "mean_selected_rows",
        "top_symbol_share",
    ]
    lines = [
        "# Label Clean-Recoverability Proxy",
        "",
        "Scope: cheap month-forward rare-event proxy over causal feature-store features. This is not production LightGBM training or a final OOS claim.",
        "",
        f"Rows: `{manifest['rows']}`",
        f"Symbols: `{manifest['symbols']}`",
        f"Feature count: `{manifest['feature_count']}`",
        f"Months: `{','.join(manifest['months'])}`",
        "",
    ]
    for frac in manifest["top_fracs"]:
        subset = aggregate[aggregate["top_frac"].eq(float(frac))].sort_values(
            ["mean_u", "worst_month_mean_u"],
            ascending=[False, False],
        )
        lines.extend([f"## Top {float(frac):.2%}", "", table(subset, cols, limit=40), ""])
    lines.extend(
        [
            "## Outputs",
            "",
            f"- Monthly: `{manifest['outputs']['monthly']}`",
            f"- Weekly: `{manifest['outputs']['weekly']}`",
            f"- Aggregate: `{manifest['outputs']['aggregate']}`",
            f"- Diagnostics: `{manifest['outputs']['diagnostics']}`",
            f"- Manifest: `{manifest['outputs']['manifest']}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_proxy(
    *,
    labels_path: Path,
    output_dir: Path,
    feature_dir: Path,
    feature_list_csv: Path,
    max_feature_store_features: int | None,
    months: list[str],
    label_arms: list[str],
    target_modes: list[str],
    weight_modes: list[str],
    dirty_penalties: list[float],
    top_fracs: list[float],
    seeds: list[int],
    train_lookback_months: int | None,
    max_weight: float,
    min_weight: float,
    derived_feature_mode: str,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = _load_labels(labels_path)
    selected_features = _read_feature_list(feature_list_csv, max_features=max_feature_store_features)
    feature_matrix, feature_store_report = _load_feature_store_columns(
        frame,
        feature_dir=feature_dir,
        selected_features=selected_features,
    )
    for col in feature_matrix.columns:
        frame[col] = feature_matrix[col].to_numpy(dtype=np.float32, copy=False)
    derived_matrix = _add_derived_clean_recoverability_features(
        frame,
        mode=str(derived_feature_mode),
    )
    derived_features = list(derived_matrix.columns)
    if derived_features:
        frame = pd.concat([frame, derived_matrix], axis=1).copy()

    metrics = _path_metrics(frame)
    features = _feature_columns(frame)
    targets = _label_targets(frame, metrics)
    if not label_arms:
        label_arms = list(LABEL_ARMS)
    missing_labels = sorted(set(label_arms) - set(targets))
    if missing_labels:
        raise ValueError(f"Unknown label arms: {missing_labels}")

    available_months = sorted(frame["__ts__"].dt.to_period("M").dropna().astype(str).unique())
    eval_months = months or available_months[1:]
    monthly_rows: list[dict[str, Any]] = []
    weekly_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    for month in eval_months:
        rows, weeks, diagnostics = _run_month(
            frame=frame,
            metrics=metrics,
            targets=targets,
            features=features,
            month=str(month),
            label_arms=label_arms,
            target_modes=target_modes,
            weight_modes=weight_modes,
            dirty_penalties=dirty_penalties,
            top_fracs=top_fracs,
            seeds=seeds,
            train_lookback_months=train_lookback_months,
            max_weight=max_weight,
            min_weight=min_weight,
        )
        monthly_rows.extend(rows)
        weekly_rows.extend(weeks)
        diagnostic_rows.extend(diagnostics)

    monthly = pd.DataFrame(monthly_rows)
    weekly = pd.DataFrame(weekly_rows)
    diagnostics = pd.DataFrame(diagnostic_rows)
    aggregate = _aggregate(monthly)
    paths = {
        "monthly": output_dir / "label_clean_recoverability_proxy_monthly.csv",
        "weekly": output_dir / "label_clean_recoverability_proxy_weekly.csv",
        "aggregate": output_dir / "label_clean_recoverability_proxy_aggregate.csv",
        "diagnostics": output_dir / "label_clean_recoverability_proxy_diagnostics.csv",
        "manifest": output_dir / "manifest.json",
    }
    monthly.to_csv(paths["monthly"], index=False)
    weekly.to_csv(paths["weekly"], index=False)
    aggregate.to_csv(paths["aggregate"], index=False)
    diagnostics.to_csv(paths["diagnostics"], index=False)
    manifest = {
        "scope": "clean_recoverability_proxy_not_full_policy_training",
        "labels_path": str(labels_path),
        "output_dir": str(output_dir),
        "rows": int(len(frame)),
        "timestamp_min": frame["__ts__"].min(),
        "timestamp_max": frame["__ts__"].max(),
        "symbols": int(frame["__symbol__"].nunique(dropna=True)),
        "feature_count": int(len(features)),
        "derived_feature_mode": str(derived_feature_mode),
        "derived_feature_count": int(len(derived_features)),
        "derived_features": derived_features,
        "feature_dir": str(feature_dir),
        "feature_list_csv": str(feature_list_csv),
        "max_feature_store_features": max_feature_store_features,
        "feature_store": feature_store_report,
        "months": [str(v) for v in eval_months],
        "label_arms": label_arms,
        "target_modes": target_modes,
        "weight_modes": weight_modes,
        "dirty_penalties": [float(v) for v in dirty_penalties],
        "top_fracs": [float(v) for v in top_fracs],
        "model": {
            "type": "ExtraTreesRegressor",
            "seeds": [int(v) for v in seeds],
            "seed_count": int(len(seeds)),
            "train_lookback_months": int(train_lookback_months)
            if train_lookback_months is not None
            else None,
            "max_weight": float(max_weight),
            "min_weight": float(min_weight),
        },
        "outputs": {key: str(value) for key, value in paths.items()},
    }
    paths["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    markdown = _write_markdown(output_dir, aggregate, manifest)
    manifest["outputs"]["markdown"] = str(markdown)
    paths["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-path", type=Path, default=DEFAULT_LABELS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--feature-list-csv", type=Path, default=DEFAULT_FEATURE_LIST_CSV)
    parser.add_argument("--max-feature-store-features", type=int, default=None)
    parser.add_argument("--months", default="2026-04,2026-05,2026-06")
    parser.add_argument("--label-arms", default=",".join(DEFAULT_LABEL_ARMS))
    parser.add_argument("--target-modes", default=",".join(DEFAULT_TARGET_MODES))
    parser.add_argument("--weight-modes", default=",".join(DEFAULT_WEIGHT_MODES))
    parser.add_argument("--dirty-penalties", default=",".join(str(v) for v in DEFAULT_DIRTY_PENALTIES))
    parser.add_argument("--top-fracs", default=",".join(str(v) for v in DEFAULT_TOP_FRACS))
    parser.add_argument("--seeds", default=",".join(str(v) for v in DEFAULT_SEEDS))
    parser.add_argument("--train-lookback-months", type=int, default=None)
    parser.add_argument("--max-weight", type=float, default=12.0)
    parser.add_argument("--min-weight", type=float, default=0.10)
    parser.add_argument("--derived-feature-mode", choices=DERIVED_FEATURE_MODES, default="none")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = run_proxy(
        labels_path=args.labels_path,
        output_dir=args.output_dir,
        feature_dir=args.feature_dir,
        feature_list_csv=args.feature_list_csv,
        max_feature_store_features=args.max_feature_store_features,
        months=_parse_csv(args.months, ()),
        label_arms=_parse_csv(args.label_arms, DEFAULT_LABEL_ARMS),
        target_modes=_parse_csv(args.target_modes, DEFAULT_TARGET_MODES),
        weight_modes=_parse_csv(args.weight_modes, DEFAULT_WEIGHT_MODES),
        dirty_penalties=_parse_float_csv(args.dirty_penalties, DEFAULT_DIRTY_PENALTIES),
        top_fracs=_parse_float_csv(args.top_fracs, DEFAULT_TOP_FRACS),
        seeds=_parse_int_csv(args.seeds, DEFAULT_SEEDS),
        train_lookback_months=args.train_lookback_months,
        max_weight=float(args.max_weight),
        min_weight=float(args.min_weight),
        derived_feature_mode=str(args.derived_feature_mode),
    )
    print(json.dumps(_json_safe(manifest), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
