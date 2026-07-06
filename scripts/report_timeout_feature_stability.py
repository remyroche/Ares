#!/usr/bin/env python3
"""Feature-level timeout and holding-time stability diagnostic.

This is a pre-training label QA report. It does not train models. It asks
whether any current causal feature has stable month-forward signal for lower
timeout / shorter holding time while preserving utility and adverse-path
quality.
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

from scripts.run_label_quality_proxy_diagnostics import (  # noqa: E402
    DEFAULT_FEATURE_DIR,
    DEFAULT_FEATURE_LIST_CSV,
    DEFAULT_LABELS_DIR,
    _feature_columns,
    _json_safe,
    _load_feature_store_columns,
    _load_labels,
    _path_metrics,
    _rank_top_indices,
    _read_feature_list,
    _safe_mean,
    _safe_quantile,
    _selection_metrics,
)
from scripts.run_soft_label_economic_proxy_ablation import (  # noqa: E402
    DEFAULT_EVENT_FEATURE_STORE_FEATURES,
    DEFAULT_PRIOR_WINDOWS_DAYS,
    DEFAULT_STATE_PATH_PRIOR_FEATURES,
    _causal_outcome_prior_features,
    _causal_state_path_prior_features,
    _event_confirmation_features,
    _parse_csv,
    _parse_float_csv,
)


DEFAULT_OUTPUT_DIR = Path("data_perp/reports/timeout_feature_stability_stage50_v1")
DEFAULT_FIT_MONTHS = ("2026-04", "2026-05")
DEFAULT_HOLDOUT_MONTH = "2026-06"
DEFAULT_TOP_FRACS = (0.0025, 0.005, 0.01)
DEFAULT_SLOW_TRADE_SOURCE_FEATURES = (
    "adx_7",
    "adx_10",
    "adx_14",
    "adx_di_plus_14",
    "adx_di_minus_14",
    "atr_compression_ratio",
    "atr_expansion",
    "atr_percentile",
    "body_pct",
    "bollinger_band_width",
    "breakout_24h",
    "breakout_confirmed",
    "churn",
    "compression_score",
    "cumulative_delta_stall",
    "delta_stall_6",
    "distance_to_resistance_daily_vwap_atr",
    "distance_to_support_daily_vwap_atr",
    "dist_ema20_atr",
    "dist_ema50_atr",
    "dist_ema200_atr",
    "dist_ema_fast_base",
    "dist_ema_slow_base",
    "dist_from_high_48h",
    "dist_from_low_48h",
    "dist_local_swing",
    "dist_prior_day_high",
    "dist_prior_day_low",
    "flow_persistence",
    "flow_ratio",
    "loc_bb_channel_pos_48",
    "loc_ema_stack_pos_24",
    "loc_range_pos_24",
    "loc_range_pos_48",
    "loc_session_pos_24",
    "loc_swing_range_pos_24",
    "log_quote_volume",
    "memory_asymmetry_1ATR",
    "memory_asymmetry_2ATR",
    "memory_asymmetry_3ATR",
    "progress",
    "quote_volume_z_30d",
    "range_12h_pct",
    "range_24h_pct",
    "range_norm_12",
    "range_norm_24",
    "range_pct",
    "rejection_proxy",
    "shock_12h",
    "shock_decay",
    "shock_vol_ratio",
    "spread_proxy_abs_return_bps_robust_z",
    "spread_proxy_close_location_robust_z",
    "spread_proxy_hl_range_bps_robust_z",
    "spread_proxy_lower_wick_bps_robust_z",
    "spread_proxy_upper_wick_bps_robust_z",
    "spread_proxy_wick_to_range_robust_z",
    "stall",
    "stall_x_flow",
    "time_since_event_extreme_12h",
    "trap_strength",
    "trend_acceleration",
    "trend_age_hours",
    "trend_alignment_1_3_6",
    "trend_dispersion_3_6_12",
    "trend_r2_24",
    "trend_r2_48",
    "trend_slope_48h",
    "trend_snr",
    "trend_strength_percentile",
    "trend_t",
    "trend_z_t",
    "vol_compression",
    "vol_expansion_ratio",
    "vol_of_vol_cp_logstd_8_32",
    "vol_price_spread",
    "vol_regime_shift",
    "vol_shock_asym_4_12",
    "vol_shock_asym_8_24",
    "wick_body_ratio",
    "wick_ratio",
    "wick_ratio_4h_max",
    "zr_1h_x_volume_z_24h",
    "zr_3h_x_volume_z_24h",
    "zr_6h_x_range_z_24h",
    "zr_12h_x_range_z_48h",
)


def _safe_numeric(values: Any) -> pd.Series:
    if isinstance(values, pd.Series):
        return pd.to_numeric(values, errors="coerce")
    return pd.to_numeric(pd.Series(values), errors="coerce")


def _feature_series(frame: pd.DataFrame, name: str) -> pd.Series | None:
    if name not in frame.columns:
        return None
    values = _safe_numeric(frame[name]).replace([np.inf, -np.inf], np.nan)
    if int(values.notna().sum()) < 100 or int(values.nunique(dropna=True)) < 3:
        return None
    return values


def _fixed_tanh(values: pd.Series, *, center: float = 0.0, scale: float = 1.0) -> pd.Series:
    scale = float(scale)
    if abs(scale) < 1e-6:
        scale = 1e-6 if scale >= 0.0 else -1e-6
    return pd.Series(
        np.tanh((_safe_numeric(values) - float(center)) / scale),
        index=values.index,
    )


def _abs_low(values: pd.Series, *, clip: float = 5.0) -> pd.Series:
    clip = max(float(clip), 1e-6)
    return -_safe_numeric(values).abs().clip(upper=clip) / clip


def _abs_high(values: pd.Series, *, clip: float = 5.0) -> pd.Series:
    clip = max(float(clip), 1e-6)
    return _safe_numeric(values).abs().clip(upper=clip) / clip


def _add_slow_trade_diagnostic_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Add fixed, causal diagnostic features for slow/non-decisive trade state."""

    base: dict[str, pd.Series] = {}

    def add(name: str, source: pd.Series | np.ndarray | None) -> None:
        if source is None:
            return
        values = _safe_numeric(source).replace([np.inf, -np.inf], np.nan)
        if len(values) != len(frame):
            values = pd.Series(values.to_numpy() if isinstance(values, pd.Series) else values, index=frame.index)
        else:
            values = pd.Series(values.to_numpy(dtype=np.float32, copy=False), index=frame.index)
        if int(values.notna().sum()) < 100 or int(values.nunique(dropna=True)) < 3:
            return
        base[name] = values.astype(np.float32)

    def s(name: str) -> pd.Series | None:
        return _feature_series(frame, name)

    def first_available(*names: str) -> pd.Series | None:
        for name in names:
            values = s(name)
            if values is not None:
                return values
        return None

    ts = pd.to_datetime(frame["__ts__"], errors="coerce", utc=True)
    hour = ts.dt.hour.astype(float) + ts.dt.minute.astype(float) / 60.0
    dayofweek = ts.dt.dayofweek.astype(float)
    add("time_diag_utc_hour_sin", np.sin(2.0 * np.pi * hour / 24.0))
    add("time_diag_utc_hour_cos", np.cos(2.0 * np.pi * hour / 24.0))
    add("time_diag_dayofweek_sin", np.sin(2.0 * np.pi * dayofweek / 7.0))
    add("time_diag_dayofweek_cos", np.cos(2.0 * np.pi * dayofweek / 7.0))

    trend_r2 = s("trend_r2_24")
    trend_r2_48 = s("trend_r2_48")
    trend_snr = s("trend_snr")
    trend_z = s("trend_z_t")
    trend_age = s("trend_age_hours")
    trend_accel = s("trend_acceleration")
    trend_align = s("trend_alignment_1_3_6")
    adx7 = s("adx_7")
    adx10 = s("adx_10")
    adx14 = s("adx_14")
    compression = s("atr_compression_ratio")
    vol_compression = s("vol_compression")
    compression_score = s("compression_score")
    atr_expansion = s("atr_expansion")
    atr_pct = s("atr_percentile")
    range12 = s("range_12h_pct")
    range24 = s("range_24h_pct")
    range_norm12 = s("range_norm_12")
    range_norm24 = s("range_norm_24")
    spread_abs = s("spread_proxy_abs_return_bps_robust_z")
    spread_hl = s("spread_proxy_hl_range_bps_robust_z")
    spread_wick = s("spread_proxy_wick_to_range_robust_z")
    spread_close = s("spread_proxy_close_location_robust_z")
    wick = first_available("wick_ratio_4h_max", "wick_ratio")
    wick_body = s("wick_body_ratio")
    body = s("body_pct")
    rejection = s("rejection_proxy")
    flow = s("flow_ratio")
    flow_persist = s("flow_persistence")
    churn = s("churn")
    stall = first_available("cumulative_delta_stall", "stall")
    stall_flow = s("stall_x_flow")
    trap = s("trap_strength")
    shock = s("shock_12h")
    shock_decay = s("shock_decay")
    shock_vol = s("shock_vol_ratio")
    event_age = s("time_since_event_extreme_12h")
    progress = s("progress")
    breakout = s("breakout_24h")
    breakout_confirmed = s("breakout_confirmed")
    dist_ema20 = s("dist_ema20_atr")
    dist_ema50 = s("dist_ema50_atr")
    dist_ema200 = s("dist_ema200_atr")
    dist_swing = s("dist_local_swing")
    dist_prior_high = s("dist_prior_day_high")
    dist_prior_low = s("dist_prior_day_low")
    support = s("distance_to_support_daily_vwap_atr")
    resistance = s("distance_to_resistance_daily_vwap_atr")
    session_pos = s("loc_session_pos_24")
    loc_range = s("loc_range_pos_24")
    loc_range48 = s("loc_range_pos_48")
    loc_bb = s("loc_bb_channel_pos_48")
    log_volume = s("log_quote_volume")
    quote_volume_z = s("quote_volume_z_30d")
    vol_price_spread = s("vol_price_spread")
    vol_of_vol = s("vol_of_vol_cp_logstd_8_32")
    zrv = s("zr_1h_x_volume_z_24h")
    zrv3 = s("zr_3h_x_volume_z_24h")
    zr_range6 = s("zr_6h_x_range_z_24h")
    zr_range12 = s("zr_12h_x_range_z_48h")

    add("slow_diag_trend_r2_low", _fixed_tanh(trend_r2, center=0.20, scale=-0.15) if trend_r2 is not None else None)
    add("fast_diag_trend_r2_high", _fixed_tanh(trend_r2, center=0.35, scale=0.18) if trend_r2 is not None else None)
    add("slow_diag_trend48_low", _fixed_tanh(trend_r2_48, center=0.20, scale=-0.15) if trend_r2_48 is not None else None)
    add("fast_diag_trend_snr_high", _fixed_tanh(trend_snr, center=0.0, scale=1.0) if trend_snr is not None else None)
    add("slow_diag_trend_z_abs_low", _abs_low(trend_z, clip=5.0) if trend_z is not None else None)
    add("fast_diag_trend_z_abs_high", _abs_high(trend_z, clip=5.0) if trend_z is not None else None)
    add("slow_diag_trend_age_high", _fixed_tanh(trend_age, center=24.0, scale=24.0) if trend_age is not None else None)
    add("fast_diag_trend_accel_high", _fixed_tanh(trend_accel, center=0.0, scale=1.0) if trend_accel is not None else None)
    add("fast_diag_trend_align_high", _fixed_tanh(trend_align, center=0.0, scale=1.0) if trend_align is not None else None)
    add("slow_diag_adx7_low", _fixed_tanh(adx7, center=4.0, scale=-1.0) if adx7 is not None else None)
    add("fast_diag_adx7_high", _fixed_tanh(adx7, center=4.0, scale=1.0) if adx7 is not None else None)
    add("slow_diag_adx10_low", _fixed_tanh(adx10, center=4.0, scale=-1.0) if adx10 is not None else None)
    add("fast_diag_adx10_high", _fixed_tanh(adx10, center=4.0, scale=1.0) if adx10 is not None else None)
    add("fast_diag_adx14_high", _fixed_tanh(adx14, center=4.0, scale=1.0) if adx14 is not None else None)
    add("slow_diag_atr_compression_high", _fixed_tanh(compression, center=0.0, scale=1.0) if compression is not None else None)
    add("slow_diag_vol_compression_high", _fixed_tanh(vol_compression, center=0.0, scale=1.0) if vol_compression is not None else None)
    add("slow_diag_compression_score_high", _fixed_tanh(compression_score, center=0.0, scale=1.0) if compression_score is not None else None)
    add("fast_diag_atr_expansion_high", _fixed_tanh(atr_expansion, center=0.0, scale=1.0) if atr_expansion is not None else None)
    add("fast_diag_atr_percentile_high", _fixed_tanh(atr_pct, center=0.55, scale=0.20) if atr_pct is not None else None)
    add("slow_diag_range12_low", _fixed_tanh(range12, center=0.0, scale=-1.0) if range12 is not None else None)
    add("slow_diag_range24_low", _fixed_tanh(range24, center=0.0, scale=-1.0) if range24 is not None else None)
    add("fast_diag_range12_high", _fixed_tanh(range12, center=0.0, scale=1.0) if range12 is not None else None)
    add("fast_diag_range24_high", _fixed_tanh(range24, center=0.0, scale=1.0) if range24 is not None else None)
    add("slow_diag_range_norm12_low", _fixed_tanh(range_norm12, center=0.0, scale=-1.0) if range_norm12 is not None else None)
    add("slow_diag_range_norm24_low", _fixed_tanh(range_norm24, center=0.0, scale=-1.0) if range_norm24 is not None else None)
    add("slow_diag_spread_abs_high", _fixed_tanh(spread_abs, center=0.0, scale=1.5) if spread_abs is not None else None)
    add("fast_diag_spread_abs_low", _fixed_tanh(spread_abs, center=0.0, scale=-1.5) if spread_abs is not None else None)
    add("slow_diag_spread_hl_high", _fixed_tanh(spread_hl, center=0.0, scale=1.5) if spread_hl is not None else None)
    add("fast_diag_spread_hl_low", _fixed_tanh(spread_hl, center=0.0, scale=-1.5) if spread_hl is not None else None)
    add("slow_diag_wick_to_range_high", _fixed_tanh(spread_wick, center=0.0, scale=1.5) if spread_wick is not None else None)
    add("slow_diag_close_location_abs_high", _abs_high(spread_close, clip=5.0) if spread_close is not None else None)
    add("slow_diag_wick_high", _fixed_tanh(wick, center=1.0, scale=0.75) if wick is not None else None)
    add("fast_diag_wick_low", _fixed_tanh(wick, center=1.0, scale=-0.75) if wick is not None else None)
    add("slow_diag_wick_body_high", _fixed_tanh(wick_body, center=1.0, scale=0.75) if wick_body is not None else None)
    add("fast_diag_body_high", _fixed_tanh(body, center=0.45, scale=0.20) if body is not None else None)
    add("slow_diag_body_low", _fixed_tanh(body, center=0.25, scale=-0.20) if body is not None else None)
    add("slow_diag_rejection_high", _fixed_tanh(rejection, center=0.0, scale=1.0) if rejection is not None else None)
    add("slow_diag_flow_abs_low", _abs_low(flow, clip=5.0) if flow is not None else None)
    add("fast_diag_flow_persist_high", _fixed_tanh(flow_persist, center=0.0, scale=1.0) if flow_persist is not None else None)
    add("slow_diag_churn_high", _fixed_tanh(churn, center=0.0, scale=1.0) if churn is not None else None)
    add("slow_diag_stall_high", _fixed_tanh(stall, center=0.0, scale=1.0) if stall is not None else None)
    add("slow_diag_stall_flow_high", _fixed_tanh(stall_flow, center=0.0, scale=1.0) if stall_flow is not None else None)
    add("slow_diag_trap_high", _fixed_tanh(trap, center=0.0, scale=1.0) if trap is not None else None)
    add("fast_diag_shock_high", _fixed_tanh(shock, center=0.0, scale=1.0) if shock is not None else None)
    add("slow_diag_shock_decay_high", _fixed_tanh(shock_decay, center=0.0, scale=1.0) if shock_decay is not None else None)
    add("slow_diag_shock_vol_high", _fixed_tanh(shock_vol, center=0.0, scale=1.0) if shock_vol is not None else None)
    add("fast_diag_event_fresh", _fixed_tanh(event_age, center=3.0, scale=-3.0) if event_age is not None else None)
    add("slow_diag_event_stale", _fixed_tanh(event_age, center=8.0, scale=3.0) if event_age is not None else None)
    add("slow_diag_progress_late", _fixed_tanh(progress, center=0.55, scale=0.25) if progress is not None else None)
    add("fast_diag_breakout_high", _fixed_tanh(breakout, center=0.0, scale=1.0) if breakout is not None else None)
    add(
        "fast_diag_breakout_confirmed_high",
        _fixed_tanh(breakout_confirmed, center=0.0, scale=1.0) if breakout_confirmed is not None else None,
    )
    add("slow_diag_ema20_abs_low", _abs_low(dist_ema20, clip=5.0) if dist_ema20 is not None else None)
    add("slow_diag_ema50_abs_low", _abs_low(dist_ema50, clip=5.0) if dist_ema50 is not None else None)
    add("slow_diag_ema200_abs_low", _abs_low(dist_ema200, clip=8.0) if dist_ema200 is not None else None)
    add("slow_diag_swing_abs_low", _abs_low(dist_swing, clip=5.0) if dist_swing is not None else None)
    add("slow_diag_prior_high_near", _fixed_tanh(dist_prior_high, center=0.0, scale=-2.0) if dist_prior_high is not None else None)
    add("slow_diag_prior_low_near", _fixed_tanh(dist_prior_low, center=0.0, scale=-2.0) if dist_prior_low is not None else None)
    add("slow_diag_support_abs_near", _abs_low(support, clip=5.0) if support is not None else None)
    add("slow_diag_resistance_abs_near", _abs_low(resistance, clip=5.0) if resistance is not None else None)
    if support is not None and resistance is not None:
        nearest_level = pd.concat([_safe_numeric(support).abs(), _safe_numeric(resistance).abs()], axis=1).min(axis=1)
        add("slow_diag_nearest_vwap_level_near", -nearest_level.clip(upper=5.0) / 5.0)
        add("fast_diag_vwap_room_abs_high", nearest_level.clip(upper=5.0) / 5.0)
    add("time_diag_session_pos", session_pos if session_pos is not None else None)
    add("slow_diag_range_mid_abs", _abs_low(loc_range - 0.5, clip=0.5) if loc_range is not None else None)
    add("slow_diag_range48_mid_abs", _abs_low(loc_range48 - 0.5, clip=0.5) if loc_range48 is not None else None)
    add("slow_diag_bb_mid_abs", _abs_low(loc_bb - 0.5, clip=0.5) if loc_bb is not None else None)
    add("fast_diag_volume_z_high", _fixed_tanh(quote_volume_z, center=0.0, scale=1.0) if quote_volume_z is not None else None)
    add("slow_diag_volume_low", _fixed_tanh(log_volume, center=0.0, scale=-1.0) if log_volume is not None else None)
    add("slow_diag_vol_price_spread_high", _fixed_tanh(vol_price_spread, center=0.0, scale=1.0) if vol_price_spread is not None else None)
    add("slow_diag_vol_of_vol_high", _fixed_tanh(vol_of_vol, center=0.0, scale=1.0) if vol_of_vol is not None else None)
    add("fast_diag_volume_impulse_high", _fixed_tanh(zrv, center=0.0, scale=1.0) if zrv is not None else None)
    add("fast_diag_volume3_impulse_high", _fixed_tanh(zrv3, center=0.0, scale=1.0) if zrv3 is not None else None)
    add("fast_diag_range_volume6_high", _fixed_tanh(zr_range6, center=0.0, scale=1.0) if zr_range6 is not None else None)
    add("fast_diag_range_volume12_high", _fixed_tanh(zr_range12, center=0.0, scale=1.0) if zr_range12 is not None else None)

    interactions = [
        ("slow_diag_chop_congestion", "slow_diag_trend_r2_low", "slow_diag_adx10_low"),
        ("slow_diag_chop_range_low", "slow_diag_chop_congestion", "slow_diag_range24_low"),
        ("slow_diag_compression_stall", "slow_diag_atr_compression_high", "slow_diag_stall_high"),
        ("slow_diag_compression_low_range", "slow_diag_atr_compression_high", "slow_diag_range24_low"),
        ("slow_diag_spread_wick_trap", "slow_diag_spread_hl_high", "slow_diag_wick_high"),
        ("slow_diag_spread_rejection_trap", "slow_diag_spread_hl_high", "slow_diag_rejection_high"),
        ("slow_diag_level_chop", "slow_diag_nearest_vwap_level_near", "slow_diag_chop_congestion"),
        ("slow_diag_midrange_chop", "slow_diag_range_mid_abs", "slow_diag_chop_congestion"),
        ("slow_diag_event_stale_chop", "slow_diag_event_stale", "slow_diag_chop_congestion"),
        ("slow_diag_low_flow_stall", "slow_diag_flow_abs_low", "slow_diag_stall_high"),
        ("slow_diag_low_volume_chop", "slow_diag_volume_low", "slow_diag_chop_congestion"),
        ("fast_diag_decisive_trend", "fast_diag_adx10_high", "fast_diag_trend_r2_high"),
        ("fast_diag_decisive_body", "fast_diag_decisive_trend", "fast_diag_body_high"),
        ("fast_diag_decisive_clean_spread", "fast_diag_decisive_trend", "fast_diag_spread_hl_low"),
        ("fast_diag_clean_breakout", "fast_diag_breakout_confirmed_high", "fast_diag_spread_hl_low"),
        ("fast_diag_impulse_fresh", "fast_diag_shock_high", "fast_diag_event_fresh"),
        ("fast_diag_impulse_body", "fast_diag_impulse_fresh", "fast_diag_body_high"),
        ("fast_diag_range_volume_clean", "fast_diag_range_volume6_high", "fast_diag_spread_hl_low"),
        ("fast_diag_trend_volume_clean", "fast_diag_decisive_trend", "fast_diag_volume_z_high"),
    ]
    for name, left, right in interactions:
        if left in base and right in base:
            add(name, base[left] * base[right])

    out = pd.DataFrame({name: values.astype(np.float32) for name, values in base.items()}, index=frame.index)
    report = {
        "enabled": True,
        "source_features_requested": int(len(DEFAULT_SLOW_TRADE_SOURCE_FEATURES)),
        "derived_features": int(len(out.columns)),
        "feature_names": list(out.columns),
        "causality": "fixed transforms of observable feature-store columns plus timestamp cycles; no labels, future path metrics, or global normalization",
    }
    return out, report


def _rank_array(values: Any) -> np.ndarray:
    return _safe_numeric(values).rank(method="average").to_numpy(dtype=np.float64, copy=True)


def _rank_corr(left_rank: np.ndarray, right_rank: np.ndarray) -> float:
    mask = np.isfinite(left_rank) & np.isfinite(right_rank)
    if int(mask.sum()) < 5:
        return float("nan")
    left = left_rank[mask]
    right = right_rank[mask]
    left_std = float(left.std(ddof=0))
    right_std = float(right.std(ddof=0))
    if left_std <= 0.0 or right_std <= 0.0:
        return float("nan")
    left_centered = left - float(left.mean())
    right_centered = right - float(right.mean())
    return float(np.mean(left_centered * right_centered) / (left_std * right_std))


def _feature_family(feature: str) -> str:
    if feature.startswith("slow_diag_"):
        return "slow_diag"
    if feature.startswith("fast_diag_"):
        return "fast_diag"
    if feature.startswith("time_diag_"):
        return "time_diag"
    if feature.startswith("prior_xs_state_"):
        return "prior_xs_state"
    if feature.startswith("prior_symbol_"):
        return "prior_symbol"
    if feature.startswith("prior_global_"):
        return "prior_global"
    if feature.startswith("event_xs_") or feature.startswith("event_"):
        return "event"
    if feature.startswith("xs_rank_"):
        return "xs_rank"
    if feature.startswith("__meta_raw__"):
        return "meta_raw"
    if "_G_" in feature:
        return "regime_split"
    return "base_feature"


def _fmt(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        number = float(value)
    except Exception:
        return ""
    if not math.isfinite(number):
        return ""
    if abs(number) >= 1000:
        return f"{number:,.0f}"
    return f"{number:.4f}".rstrip("0").rstrip(".")


def _table(frame: pd.DataFrame, cols: list[str], *, limit: int | None = None) -> str:
    if frame.empty:
        return "_No rows._"
    view = frame[[col for col in cols if col in frame.columns]].copy()
    if limit is not None:
        view = view.head(limit)
    for col in view.columns:
        if pd.api.types.is_numeric_dtype(view[col]):
            view[col] = view[col].map(_fmt)
    return view.to_markdown(index=False)


def _weighted_mean(frame: pd.DataFrame, value_col: str, weight_col: str = "selected_rows") -> float:
    if frame.empty or value_col not in frame.columns or weight_col not in frame.columns:
        return float("nan")
    values = _safe_numeric(frame[value_col])
    weights = _safe_numeric(frame[weight_col]).fillna(0.0)
    mask = values.notna() & weights.gt(0.0)
    if not bool(mask.any()):
        return float("nan")
    return float(np.average(values[mask], weights=weights[mask]))


def _month_summary(prefix: str, frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {
            f"{prefix}_months": 0,
            f"{prefix}_positive_months": 0,
            f"{prefix}_mean_month_u": float("nan"),
            f"{prefix}_worst_month_u": float("nan"),
            f"{prefix}_bad_mae_1r_rate": float("nan"),
            f"{prefix}_p90_mae_norm": float("nan"),
            f"{prefix}_wide_25bps_rate": float("nan"),
            f"{prefix}_timeout_rate": float("nan"),
            f"{prefix}_selected_rows": 0,
        }
    mean_u = _safe_numeric(frame["mean_u"])
    return {
        f"{prefix}_months": int(frame["period"].astype(str).nunique()),
        f"{prefix}_positive_months": int(mean_u.gt(0.0).sum()),
        f"{prefix}_mean_month_u": _safe_mean(mean_u),
        f"{prefix}_worst_month_u": float(mean_u.min()) if mean_u.notna().any() else float("nan"),
        f"{prefix}_selected_rows": int(_safe_numeric(frame["selected_rows"]).fillna(0.0).sum()),
        f"{prefix}_bad_mae_1r_rate": _weighted_mean(frame, "bad_mae_1r_rate"),
        f"{prefix}_p90_mae_norm": _weighted_mean(frame, "p90_mae_norm"),
        f"{prefix}_wide_25bps_rate": _weighted_mean(frame, "wide_barrier_25bps_rate"),
        f"{prefix}_timeout_rate": _weighted_mean(frame, "timeout_rate"),
        f"{prefix}_q10_u": _weighted_mean(frame, "q10_u"),
    }


def _is_clean(row: pd.Series, prefix: str) -> bool:
    return bool(
        row.get(f"{prefix}_mean_month_u", float("nan")) > 0.0
        and row.get(f"{prefix}_worst_month_u", float("nan")) > 0.0
        and row.get(f"{prefix}_bad_mae_1r_rate", float("nan")) <= 0.40
        and row.get(f"{prefix}_p90_mae_norm", float("nan")) <= 4.0
        and row.get(f"{prefix}_wide_25bps_rate", float("nan")) <= 0.05
        and row.get(f"{prefix}_timeout_rate", float("nan")) <= 0.20
    )


def _targets(metrics: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "utility": _safe_numeric(metrics["u_policy_net"]),
        "timeout": metrics["is_timeout"].astype(float),
        "bad_mae": (_safe_numeric(metrics["mae_norm"]) >= 1.0).astype(float),
        "wide_25": (_safe_numeric(metrics["barrier"]) > 0.025).astype(float),
        "bars_policy": _safe_numeric(metrics["bars_policy"]),
        "bars_to_mfe": _safe_numeric(metrics["bars_to_mfe"]),
    }


def _load_frame(
    *,
    labels_path: Path,
    feature_dir: Path,
    feature_list_csv: Path,
    max_feature_store_features: int | None,
    include_causal_outcome_priors: bool,
    include_causal_state_path_priors: bool,
    include_event_confirmation_features: bool,
    include_slow_trade_diagnostic_features: bool,
    prior_windows_days: list[float],
    prior_embargo_hours: float,
    state_path_prior_features: list[str],
    event_feature_store_features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict[str, Any]]:
    frame = _load_labels(labels_path)
    selected_features = _read_feature_list(feature_list_csv, max_features=max_feature_store_features)
    if include_slow_trade_diagnostic_features:
        selected_features = list(dict.fromkeys(list(selected_features) + list(DEFAULT_SLOW_TRADE_SOURCE_FEATURES)))
    if include_event_confirmation_features:
        selected_features = list(dict.fromkeys(list(selected_features) + list(event_feature_store_features)))
    feature_matrix, feature_store_report = _load_feature_store_columns(
        frame,
        feature_dir=feature_dir,
        selected_features=selected_features,
    )
    if not feature_matrix.empty:
        frame = pd.concat(
            [
                frame.drop(columns=[col for col in feature_matrix.columns if col in frame.columns]),
                feature_matrix.astype(np.float32, copy=False),
            ],
            axis=1,
        ).copy()
    metrics = _path_metrics(frame)
    reports: dict[str, Any] = {"feature_store": feature_store_report}
    if include_causal_outcome_priors:
        outcome_priors, reports["causal_outcome_priors"] = _causal_outcome_prior_features(
            frame,
            metrics,
            windows_days=prior_windows_days,
            embargo_hours=prior_embargo_hours,
        )
        frame = pd.concat([frame, outcome_priors.astype(np.float32, copy=False)], axis=1).copy()
    else:
        reports["causal_outcome_priors"] = {"enabled": False}
    if include_causal_state_path_priors:
        state_priors, reports["causal_state_path_priors"] = _causal_state_path_prior_features(
            frame,
            metrics,
            state_features=state_path_prior_features,
            windows_days=prior_windows_days,
            embargo_hours=prior_embargo_hours,
        )
        frame = pd.concat([frame, state_priors.astype(np.float32, copy=False)], axis=1).copy()
    else:
        reports["causal_state_path_priors"] = {"enabled": False}
    if include_event_confirmation_features:
        event_features, reports["event_confirmation_features"] = _event_confirmation_features(
            frame,
            event_features=event_feature_store_features,
        )
        frame = pd.concat([frame, event_features.astype(np.float32, copy=False)], axis=1).copy()
    else:
        reports["event_confirmation_features"] = {"enabled": False}
    if include_slow_trade_diagnostic_features:
        slow_features, reports["slow_trade_diagnostic_features"] = _add_slow_trade_diagnostic_features(frame)
        if not slow_features.empty:
            frame = pd.concat([frame, slow_features.astype(np.float32, copy=False)], axis=1).copy()
    else:
        reports["slow_trade_diagnostic_features"] = {"enabled": False}
    features = _feature_columns(frame)
    return frame, metrics, features, reports


def _feature_ic_rows(
    frame: pd.DataFrame,
    metrics: pd.DataFrame,
    features: list[str],
    *,
    months: list[str],
    fit_months: list[str],
    holdout_month: str,
) -> pd.DataFrame:
    month_series = frame["__ts__"].dt.to_period("M").astype(str)
    target_map = _targets(metrics)
    target_ranks: dict[tuple[str, str], np.ndarray] = {}
    for period in months:
        mask = month_series.eq(period).to_numpy(dtype=bool, copy=False)
        for name, values in target_map.items():
            target_ranks[(period, name)] = _rank_array(values.loc[mask].reset_index(drop=True))

    period_rows: list[dict[str, Any]] = []
    feature_arrays: dict[str, dict[str, np.ndarray]] = {}
    for feature in features:
        values = _safe_numeric(frame[feature])
        per_period: dict[str, np.ndarray] = {}
        for period in months:
            mask = month_series.eq(period).to_numpy(dtype=bool, copy=False)
            feature_values = values.loc[mask].reset_index(drop=True)
            rank = _rank_array(feature_values)
            per_period[period] = rank
            row: dict[str, Any] = {
                "feature": feature,
                "feature_family": _feature_family(feature),
                "period": period,
                "finite_frac": float(feature_values.notna().mean()),
                "nunique": int(feature_values.nunique(dropna=True)),
            }
            for target_name in target_map:
                row[f"ic_{target_name}"] = _rank_corr(rank, target_ranks[(period, target_name)])
            period_rows.append(row)
        feature_arrays[feature] = per_period

    period_ic = pd.DataFrame(period_rows)
    summary_rows: list[dict[str, Any]] = []
    for feature, group in period_ic.groupby("feature", sort=False):
        fit_group = group[group["period"].astype(str).isin(fit_months)].copy()
        holdout = group[group["period"].astype(str).eq(str(holdout_month))].copy()
        raw_fit_timeout = _safe_mean(fit_group["ic_timeout"])
        direction = -1.0 if math.isfinite(raw_fit_timeout) and raw_fit_timeout > 0.0 else 1.0

        def directed_mean(frame_part: pd.DataFrame, col: str) -> float:
            return direction * _safe_mean(frame_part[col])

        def directed_max(frame_part: pd.DataFrame, col: str) -> float:
            values = direction * _safe_numeric(frame_part[col])
            return float(values.max()) if values.notna().any() else float("nan")

        def directed_min(frame_part: pd.DataFrame, col: str) -> float:
            values = direction * _safe_numeric(frame_part[col])
            return float(values.min()) if values.notna().any() else float("nan")

        all_group = group.copy()
        row = {
            "feature": feature,
            "feature_family": str(group["feature_family"].iloc[0]),
            "direction": "high" if direction > 0.0 else "low",
            "fit_months": ",".join(fit_months),
            "holdout_month": holdout_month,
            "fit_timeout_ic": directed_mean(fit_group, "ic_timeout"),
            "fit_timeout_ic_max": directed_max(fit_group, "ic_timeout"),
            "holdout_timeout_ic": directed_mean(holdout, "ic_timeout"),
            "all_timeout_ic_max": directed_max(all_group, "ic_timeout"),
            "fit_utility_ic": directed_mean(fit_group, "ic_utility"),
            "fit_utility_ic_min": directed_min(fit_group, "ic_utility"),
            "holdout_utility_ic": directed_mean(holdout, "ic_utility"),
            "fit_bad_mae_ic": directed_mean(fit_group, "ic_bad_mae"),
            "fit_bad_mae_ic_max": directed_max(fit_group, "ic_bad_mae"),
            "holdout_bad_mae_ic": directed_mean(holdout, "ic_bad_mae"),
            "fit_wide_ic": directed_mean(fit_group, "ic_wide_25"),
            "holdout_wide_ic": directed_mean(holdout, "ic_wide_25"),
            "fit_bars_policy_ic": directed_mean(fit_group, "ic_bars_policy"),
            "holdout_bars_policy_ic": directed_mean(holdout, "ic_bars_policy"),
            "fit_bars_to_mfe_ic": directed_mean(fit_group, "ic_bars_to_mfe"),
            "holdout_bars_to_mfe_ic": directed_mean(holdout, "ic_bars_to_mfe"),
            "mean_finite_frac": _safe_mean(group["finite_frac"]),
            "min_nunique": int(_safe_numeric(group["nunique"]).min()),
        }
        row["anti_timeout_fit_pass"] = bool(row["fit_timeout_ic_max"] <= 0.0)
        row["anti_timeout_holdout_pass"] = bool(row["holdout_timeout_ic"] <= 0.0)
        row["utility_fit_pass"] = bool(row["fit_utility_ic_min"] >= 0.0)
        row["utility_holdout_pass"] = bool(row["holdout_utility_ic"] >= 0.0)
        row["bad_mae_fit_pass"] = bool(row["fit_bad_mae_ic_max"] <= 0.0)
        row["bad_mae_holdout_pass"] = bool(row["holdout_bad_mae_ic"] <= 0.0)
        row["stable_timeout_feature"] = bool(
            row["anti_timeout_fit_pass"]
            and row["anti_timeout_holdout_pass"]
            and row["utility_fit_pass"]
            and row["bad_mae_fit_pass"]
        )
        row["stable_full_feature"] = bool(
            row["stable_timeout_feature"]
            and row["utility_holdout_pass"]
            and row["bad_mae_holdout_pass"]
        )
        row["stability_score"] = float(
            -2.0 * (row["fit_timeout_ic"] if math.isfinite(row["fit_timeout_ic"]) else 0.0)
            -2.0 * (row["holdout_timeout_ic"] if math.isfinite(row["holdout_timeout_ic"]) else 0.0)
            + (row["fit_utility_ic"] if math.isfinite(row["fit_utility_ic"]) else 0.0)
            + (row["holdout_utility_ic"] if math.isfinite(row["holdout_utility_ic"]) else 0.0)
            - (row["fit_bad_mae_ic"] if math.isfinite(row["fit_bad_mae_ic"]) else 0.0)
            - (row["holdout_bad_mae_ic"] if math.isfinite(row["holdout_bad_mae_ic"]) else 0.0)
        )
        summary_rows.append(row)
    return period_ic, pd.DataFrame(summary_rows)


def _selection_rows(
    frame: pd.DataFrame,
    metrics: pd.DataFrame,
    feature_summary: pd.DataFrame,
    *,
    top_fracs: list[float],
    fit_months: list[str],
    holdout_month: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    month_series = frame["__ts__"].dt.to_period("M").astype(str)
    months = sorted(set(fit_months + [holdout_month]))
    monthly_rows: list[dict[str, Any]] = []
    dummy_targets = pd.DataFrame(
        {
            "target_soft": pd.Series(0.0, index=frame.index),
            "target_hard": pd.Series(0.0, index=frame.index),
        }
    )
    for _, row in feature_summary.iterrows():
        feature = str(row["feature"])
        direction = 1.0 if str(row["direction"]) == "high" else -1.0
        score_all = direction * _safe_numeric(frame[feature])
        for period in months:
            mask = month_series.eq(period)
            if int(mask.sum()) < 100:
                continue
            valid_frame = frame.loc[mask].reset_index(drop=True)
            valid_metrics = metrics.loc[mask].reset_index(drop=True)
            valid_target = dummy_targets.loc[mask].reset_index(drop=True)
            valid_score = score_all.loc[mask].reset_index(drop=True)
            for top_frac in top_fracs:
                metric_row = _selection_metrics(
                    frame=valid_frame,
                    metrics=valid_metrics,
                    target=valid_target,
                    score=valid_score,
                    arm=feature,
                    selector="single_feature_timeout_direction",
                    period=period,
                    top_frac=float(top_frac),
                )
                metric_row["feature"] = feature
                metric_row["feature_family"] = row["feature_family"]
                metric_row["direction"] = row["direction"]
                monthly_rows.append(metric_row)
    monthly = pd.DataFrame(monthly_rows)
    if monthly.empty:
        return monthly, pd.DataFrame()

    aggregate_rows: list[dict[str, Any]] = []
    for (feature, top_frac), group in monthly.groupby(["feature", "top_frac"], dropna=False, sort=False):
        fit = group[group["period"].astype(str).isin(fit_months)].copy()
        holdout = group[group["period"].astype(str).eq(str(holdout_month))].copy()
        if fit.empty or holdout.empty:
            continue
        out = {
            "feature": str(feature),
            "feature_family": str(group["feature_family"].iloc[0]),
            "direction": str(group["direction"].iloc[0]),
            "top_frac": float(top_frac),
        }
        out.update(_month_summary("fit", fit))
        out.update(_month_summary("holdout", holdout))
        out["fit_clean_pass"] = _is_clean(pd.Series(out), "fit")
        out["holdout_clean_standalone_pass"] = _is_clean(pd.Series(out), "holdout")
        out["holdout_clean_pass"] = bool(out["fit_clean_pass"] and out["holdout_clean_standalone_pass"])
        out["positive_dirty_holdout"] = bool(out["holdout_mean_month_u"] > 0.0 and not out["holdout_clean_pass"])
        out["path_risk_score"] = float(
            (out["holdout_mean_month_u"] if math.isfinite(out["holdout_mean_month_u"]) else 0.0)
            + 0.25 * (out["holdout_q10_u"] if math.isfinite(out["holdout_q10_u"]) else 0.0)
            - 0.020 * (out["holdout_bad_mae_1r_rate"] if math.isfinite(out["holdout_bad_mae_1r_rate"]) else 0.0)
            - 0.003 * (out["holdout_p90_mae_norm"] if math.isfinite(out["holdout_p90_mae_norm"]) else 0.0)
            - 0.010 * (out["holdout_timeout_rate"] if math.isfinite(out["holdout_timeout_rate"]) else 0.0)
        )
        aggregate_rows.append(out)
    return monthly, pd.DataFrame(aggregate_rows)


def _family_summary(feature_summary: pd.DataFrame, selection_summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for family, group in feature_summary.groupby("feature_family", dropna=False, sort=False):
        selected = selection_summary[selection_summary["feature_family"].astype(str).eq(str(family))]
        rows.append(
            {
                "feature_family": family,
                "features": int(len(group)),
                "stable_timeout_features": int(group["stable_timeout_feature"].sum()),
                "stable_full_features": int(group["stable_full_feature"].sum()),
                "mean_holdout_timeout_ic": _safe_mean(group["holdout_timeout_ic"]),
                "best_stability_score": float(_safe_numeric(group["stability_score"]).max()),
                "single_feature_fit_clean": int(selected["fit_clean_pass"].sum()) if not selected.empty else 0,
                "single_feature_holdout_clean": int(selected["holdout_clean_pass"].sum()) if not selected.empty else 0,
                "best_holdout_mean_u": float(_safe_numeric(selected["holdout_mean_month_u"]).max())
                if not selected.empty
                else float("nan"),
                "best_holdout_timeout_rate": float(_safe_numeric(selected["holdout_timeout_rate"]).min())
                if not selected.empty
                else float("nan"),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["stable_full_features", "stable_timeout_features", "best_stability_score"],
        ascending=[False, False, False],
    )


def _write_markdown(
    *,
    output_dir: Path,
    feature_summary: pd.DataFrame,
    selection_summary: pd.DataFrame,
    family_summary: pd.DataFrame,
    manifest: dict[str, Any],
) -> Path:
    path = output_dir / "timeout_feature_stability_report.md"
    stable = feature_summary.sort_values(
        ["stable_full_feature", "stable_timeout_feature", "stability_score"],
        ascending=[False, False, False],
    )
    best_selectors = selection_summary.sort_values(
        ["holdout_clean_pass", "positive_dirty_holdout", "path_risk_score"],
        ascending=[False, False, False],
    )
    slow_report = manifest.get("feature_reports", {}).get("slow_trade_diagnostic_features", {})
    lines = [
        "# Timeout Feature Stability",
        "",
        "Scope: feature-level diagnostic only. No model training, Optuna, policy geometry, or inference artifact changes.",
        "",
        f"Fit months: `{','.join(manifest['fit_months'])}`. Holdout month: `{manifest['holdout_month']}`.",
        f"Features evaluated: `{manifest['feature_count']}`.",
    ]
    if bool(slow_report.get("enabled", False)):
        lines.extend(
            [
                f"Slow-trade diagnostic features: `{slow_report.get('derived_features', 0)}`.",
                "Slow-trade diagnostic causality: fixed transforms of observable feature-store columns plus timestamp cycles.",
            ]
        )
    lines.extend(
        [
        "",
        "## Counts",
        "",
        f"- Stable timeout features: `{manifest['stable_timeout_features']}`",
        f"- Stable full features: `{manifest['stable_full_features']}`",
        f"- Single-feature clean fit+holdout rows: `{manifest['single_feature_holdout_clean_rows']}`",
        f"- Single-feature positive but dirty holdout rows: `{manifest['single_feature_positive_dirty_rows']}`",
        "",
        "## Best Feature IC Stability",
        "",
        _table(
            stable,
            [
                "feature",
                "feature_family",
                "direction",
                "fit_timeout_ic",
                "holdout_timeout_ic",
                "fit_utility_ic",
                "holdout_utility_ic",
                "fit_bad_mae_ic",
                "holdout_bad_mae_ic",
                "stability_score",
                "stable_timeout_feature",
                "stable_full_feature",
            ],
            limit=30,
        ),
        "",
        "## Best Single-Feature Selectors",
        "",
        _table(
            best_selectors,
            [
                "feature",
                "feature_family",
                "direction",
                "top_frac",
                "fit_mean_month_u",
                "fit_bad_mae_1r_rate",
                "fit_p90_mae_norm",
                "fit_timeout_rate",
                "holdout_mean_month_u",
                "holdout_bad_mae_1r_rate",
                "holdout_p90_mae_norm",
                "holdout_timeout_rate",
                "holdout_clean_pass",
                "path_risk_score",
            ],
            limit=30,
        ),
        "",
        "## Feature Families",
        "",
        _table(
            family_summary,
            [
                "feature_family",
                "features",
                "stable_timeout_features",
                "stable_full_features",
                "mean_holdout_timeout_ic",
                "single_feature_fit_clean",
                "single_feature_holdout_clean",
                "best_holdout_mean_u",
                "best_holdout_timeout_rate",
            ],
        ),
        "",
        "## Interpretation",
        "",
        "- If stable timeout features are scarce or no single-feature selector passes fit+holdout, the current feature stack lacks a freezeable holding-time control signal.",
        "- If stable timeout features exist but selectors are economically dirty, use them as auxiliary gates or new feature candidates, not as standalone labels.",
        "- A positive holdout utility row with high timeout is not sufficient evidence for training; it reproduces the Stage49 failure mode.",
        "",
        "## Outputs",
        "",
        f"- Feature period IC: `{manifest['outputs']['feature_period_ic']}`",
        f"- Feature stability: `{manifest['outputs']['feature_stability']}`",
        f"- Single-feature monthly selection: `{manifest['outputs']['single_feature_monthly']}`",
        f"- Single-feature fit/holdout: `{manifest['outputs']['single_feature_fit_holdout']}`",
        f"- Family summary: `{manifest['outputs']['family_summary']}`",
        f"- Manifest: `{manifest['outputs']['manifest']}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_report(
    *,
    labels_path: Path,
    feature_dir: Path,
    feature_list_csv: Path,
    output_dir: Path,
    max_feature_store_features: int | None,
    fit_months: list[str],
    holdout_month: str,
    top_fracs: list[float],
    include_causal_outcome_priors: bool,
    include_causal_state_path_priors: bool,
    include_event_confirmation_features: bool,
    include_slow_trade_diagnostic_features: bool,
    prior_windows_days: list[float],
    prior_embargo_hours: float,
    state_path_prior_features: list[str],
    event_feature_store_features: list[str],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame, metrics, features, reports = _load_frame(
        labels_path=labels_path,
        feature_dir=feature_dir,
        feature_list_csv=feature_list_csv,
        max_feature_store_features=max_feature_store_features,
        include_causal_outcome_priors=include_causal_outcome_priors,
        include_causal_state_path_priors=include_causal_state_path_priors,
        include_event_confirmation_features=include_event_confirmation_features,
        include_slow_trade_diagnostic_features=include_slow_trade_diagnostic_features,
        prior_windows_days=prior_windows_days,
        prior_embargo_hours=prior_embargo_hours,
        state_path_prior_features=state_path_prior_features,
        event_feature_store_features=event_feature_store_features,
    )
    month_series = frame["__ts__"].dt.to_period("M").astype(str)
    months = sorted(set(fit_months + [holdout_month]))
    missing_months = sorted(set(months) - set(month_series.dropna().unique()))
    if missing_months:
        raise ValueError(f"Requested months are absent from labels: {missing_months}")

    period_ic, feature_summary = _feature_ic_rows(
        frame,
        metrics,
        features,
        months=months,
        fit_months=fit_months,
        holdout_month=holdout_month,
    )
    monthly, selection_summary = _selection_rows(
        frame,
        metrics,
        feature_summary,
        top_fracs=top_fracs,
        fit_months=fit_months,
        holdout_month=holdout_month,
    )
    family_summary = _family_summary(feature_summary, selection_summary)

    paths = {
        "feature_period_ic": output_dir / "timeout_feature_period_ic.csv",
        "feature_stability": output_dir / "timeout_feature_stability.csv",
        "single_feature_monthly": output_dir / "timeout_single_feature_monthly.csv",
        "single_feature_fit_holdout": output_dir / "timeout_single_feature_fit_holdout.csv",
        "family_summary": output_dir / "timeout_feature_family_summary.csv",
        "manifest": output_dir / "manifest.json",
    }
    period_ic.to_csv(paths["feature_period_ic"], index=False)
    feature_summary.to_csv(paths["feature_stability"], index=False)
    monthly.to_csv(paths["single_feature_monthly"], index=False)
    selection_summary.to_csv(paths["single_feature_fit_holdout"], index=False)
    family_summary.to_csv(paths["family_summary"], index=False)

    manifest = {
        "scope": "timeout_feature_stability",
        "labels_path": str(labels_path),
        "feature_dir": str(feature_dir),
        "feature_list_csv": str(feature_list_csv),
        "output_dir": str(output_dir),
        "rows": int(len(frame)),
        "symbols": int(frame["__symbol__"].nunique(dropna=True)),
        "timestamp_min": frame["__ts__"].min(),
        "timestamp_max": frame["__ts__"].max(),
        "fit_months": list(fit_months),
        "holdout_month": str(holdout_month),
        "top_fracs": [float(v) for v in top_fracs],
        "feature_count": int(len(features)),
        "stable_timeout_features": int(feature_summary["stable_timeout_feature"].sum()),
        "stable_full_features": int(feature_summary["stable_full_feature"].sum()),
        "single_feature_fit_clean_rows": int(selection_summary["fit_clean_pass"].sum())
        if not selection_summary.empty
        else 0,
        "single_feature_holdout_clean_rows": int(selection_summary["holdout_clean_pass"].sum())
        if not selection_summary.empty
        else 0,
        "single_feature_positive_dirty_rows": int(selection_summary["positive_dirty_holdout"].sum())
        if not selection_summary.empty
        else 0,
        "feature_reports": reports,
        "outputs": {key: str(value) for key, value in paths.items()},
    }
    markdown = _write_markdown(
        output_dir=output_dir,
        feature_summary=feature_summary,
        selection_summary=selection_summary,
        family_summary=family_summary,
        manifest=manifest,
    )
    manifest["outputs"]["markdown"] = str(markdown)
    paths["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-path", type=Path, default=DEFAULT_LABELS_DIR)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--feature-list-csv", type=Path, default=DEFAULT_FEATURE_LIST_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-feature-store-features", type=int, default=None)
    parser.add_argument("--fit-months", type=_parse_csv, default=",".join(DEFAULT_FIT_MONTHS))
    parser.add_argument("--holdout-month", type=str, default=DEFAULT_HOLDOUT_MONTH)
    parser.add_argument("--top-fracs", type=_parse_float_csv, default=",".join(str(v) for v in DEFAULT_TOP_FRACS))
    parser.add_argument("--include-causal-outcome-priors", action="store_true")
    parser.add_argument("--include-causal-state-path-priors", action="store_true")
    parser.add_argument("--include-event-confirmation-features", action="store_true")
    parser.add_argument("--include-slow-trade-diagnostic-features", action="store_true")
    parser.add_argument("--prior-windows-days", type=_parse_float_csv, default=",".join(str(v) for v in DEFAULT_PRIOR_WINDOWS_DAYS))
    parser.add_argument("--prior-embargo-hours", type=float, default=24.0)
    parser.add_argument(
        "--state-path-prior-features",
        type=_parse_csv,
        default=",".join(DEFAULT_STATE_PATH_PRIOR_FEATURES),
    )
    parser.add_argument(
        "--event-feature-store-features",
        type=_parse_csv,
        default=",".join(DEFAULT_EVENT_FEATURE_STORE_FEATURES),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = run_report(
        labels_path=args.labels_path,
        feature_dir=args.feature_dir,
        feature_list_csv=args.feature_list_csv,
        output_dir=args.output_dir,
        max_feature_store_features=args.max_feature_store_features,
        fit_months=list(args.fit_months),
        holdout_month=str(args.holdout_month),
        top_fracs=list(args.top_fracs),
        include_causal_outcome_priors=bool(args.include_causal_outcome_priors),
        include_causal_state_path_priors=bool(args.include_causal_state_path_priors),
        include_event_confirmation_features=bool(args.include_event_confirmation_features),
        include_slow_trade_diagnostic_features=bool(args.include_slow_trade_diagnostic_features),
        prior_windows_days=list(args.prior_windows_days),
        prior_embargo_hours=float(args.prior_embargo_hours),
        state_path_prior_features=list(args.state_path_prior_features),
        event_feature_store_features=list(args.event_feature_store_features),
    )
    print(json.dumps(_json_safe(manifest), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
