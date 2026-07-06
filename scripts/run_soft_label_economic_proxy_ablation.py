#!/usr/bin/env python3
"""Soft-label economic proxy ablation before base/meta training.

This is a label QA tool, not a model-training tool. It compares soft-label
definitions on a fixed candidate universe with:

1. oracle label-sort, which tests whether the label ranks executable outcomes;
2. prior-month feature proxies, which tests whether that ranking is learnable.

Both selectors are judged by Apr-May fit selection and June holdout economic
gates so labels are evaluated inside the intended execution envelope.
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
    LABEL_ARMS,
    _feature_columns,
    _json_safe,
    _load_feature_store_columns,
    _load_labels,
    _make_targets,
    _path_metrics,
    _rank_top_indices,
    _read_feature_list,
    _safe_mean,
    _safe_quantile,
    _selection_metrics,
    _sigmoid,
    _spearman,
)


DEFAULT_OUTPUT_DIR = Path("data_perp/reports/soft_label_economic_proxy_ablation_s10_policy_net_v1")
DEFAULT_TOP_FRACS = (0.005, 0.010, 0.030, 0.050, 0.100, 0.300)
DEFAULT_FIT_MONTHS = ("2026-04", "2026-05")
DEFAULT_HOLDOUT_MONTH = "2026-06"
PROXY_OBJECTIVES = ("target_ic", "economic_ic", "economic_score")
DEFAULT_PRIOR_WINDOWS_DAYS = (7.0, 14.0, 30.0)
DEFAULT_STATE_PATH_PRIOR_FEATURES = (
    "zscore_price_200",
    "atr_compression_ratio",
    "loc_prev_week_range_pos_24",
    "loc_range_pos_24",
    "oi_rank",
    "range_24h_pct",
    "spread_proxy_hl_range_bps_robust_z",
    "spread_proxy_abs_return_bps_robust_z",
    "distance_to_support_daily_vwap_atr",
    "distance_to_resistance_daily_vwap_atr",
    "adx_7",
    "breakout_24h",
    "body_pct",
)
DEFAULT_EVENT_FEATURE_STORE_FEATURES = (
    "time_since_event_extreme_12h",
    "second_leg_accel_1h",
    "second_leg_accel_2h",
    "second_leg_accel_vol_1h",
    "shock_12h",
    "shock_vol_ratio",
    "shock_decay",
    "dist_from_low_event_12h",
    "dist_from_low_48h",
    "dist_from_high_48h",
    "pullback_2",
    "pullback_4",
    "pullback_8",
    "pullback_48",
    "progress",
    "speed",
    "breakout_24h",
    "breakout_confirmed",
    "breakout_soft",
    "pct_breakout_t",
    "climax_range_24",
    "climax_vol_12",
    "jump_intensity",
    "impulse",
    "impulse_ratio_24",
    "vw_breakout",
    "retest_quality",
    "rejection_proxy",
    "volume_trend_alignment",
    "dip_volume_profile",
    "volume_capitulation",
    "spread_proxy_hl_range_bps_robust_z",
    "spread_proxy_abs_return_bps_robust_z",
    "median_spread_bps",
    "xasset_ob_liquidity_peer_resid",
    "distance_to_support_daily_vwap_atr",
    "distance_to_resistance_daily_vwap_atr",
    "up_barrier_pressure_daily_vwap",
    "down_barrier_pressure_daily_vwap",
    "oi_rank",
    "oi_chg_2h",
    "oi_up_agree",
    "leverage_build_score",
    "oiw_pos_delta_entry_dist_1d_atr",
    "oiw_pos_delta_entry_dist_7d_atr",
    "oiw_pos_delta_entry_dist_14d_atr",
    "dist_oiw_abs_delta_12h_atr",
    "dist_oiw_signed_delta_12h_atr",
    "range_12h_pct",
    "range_24h_pct",
    "range_pct",
    "body_pct",
    "wick_ratio_4h_max",
    "loc_range_pos_24",
    "loc_prev_week_range_pos_24",
)


EXTRA_ARM_DESCRIPTIONS = {
    "E1_strict_clean_utility": "utility gated by strict MAE/barrier/time cleanliness",
    "E2_bounded_clean_utility": "utility gated by bounded MAE/barrier/time cleanliness",
    "E3_mfe_mae_ratio_clean": "utility gated by MFE/MAE ratio and clean path",
    "E4_fast_clean_mfe": "utility gated by fast 1R MFE and clean MAE",
    "E5_margin_clean_utility": "strict clean utility requiring an explicit net edge",
    "E6_low_barrier_clean": "strict clean utility with lower barrier preference",
    "E7_timestamp_rank_clean": "timestamp-local rank blend of strict clean utility",
    "E8_low_mae_net_utility": "positive net utility with a strict low-MAE path envelope",
    "E9_low_mae_mfe_ratio": "positive utility requiring low MAE and strong MFE/MAE ratio",
    "E10_fast_low_mae_rank": "timestamp-local rank of fast low-MAE positive utility",
    "E11_low_mae_margin_utility": "low-MAE clean utility requiring a larger net edge",
    "E12_quiet_low_mae_mfe_ratio": "E9 with a decision-time quiet-continuation event-intensity gate",
    "E13_oi_location_low_mae": "E9 with decision-time OI and range-location confirmation",
    "E14_run_entry_low_mae": "E9 de-clustered to prefer early rows in adjacent clean symbol runs",
    "E15_quiet_oi_low_mae": "E9 with quiet-continuation plus OI/location confirmation",
    "E16_quiet_oi_run_entry": "E9 with quiet-continuation, OI/location confirmation, and run-entry de-clustering",
    "E17_loud_barrier_low_mae": "E9 restricted to loud events with an explicit low-barrier economic envelope",
    "E18_loud_range_efficiency": "loud-event utility gated by MFE/MAE efficiency, speed, and bounded barriers",
    "E19_loud_liquid_low_barrier": "E18 with decision-time liquidity and low-barrier context confirmation",
    "E20_loud_run_entry_barrier": "E17 de-clustered to prefer early rows in adjacent loud-event clean runs",
    "E21_bounded_high_mae_efficiency": "positive utility inside a bounded high-MAE, MFE/MAE-efficient path envelope",
    "E22_lowz_rebound_bounded": "E21 with decision-time low-zscore/ATR-compression rebound context",
    "E23_loud_bounded_event_efficiency": "E21 with loud-event context for a bounded high-MAE event family",
    "E24_rebound_run_entry_bounded": "E22 de-clustered to prefer early rows in adjacent rebound runs",
    "E25_fast_decisive_low_mae": "fast no-timeout low-MAE path requiring early MFE and positive net utility",
    "E26_fast_decisive_rank": "timestamp-local rank of E25 to force sparse fast-path containment",
    "E27_fast_margin_no_timeout": "fast no-timeout path requiring a larger net utility margin",
    "E28_ultra_low_mae_fast": "ultra-low MAE and low barrier fast-path target",
    "E29_fast_loud_clean": "E25 restricted to loud events after enforcing fast clean path quality",
    "E30_fast_rebound_clean": "E25 restricted to low-zscore/rebound context after enforcing fast clean path quality",
    "E31_fast_run_entry_clean": "E25 de-clustered to prefer first rows in fast clean symbol runs",
    "E32_fast_low_timeout_rank": "timestamp-local rank blend of strict fast/no-timeout and timeout-averse path score",
    "E33_timeout_primary_decisive_path": "no-timeout decisive-path target with utility as a secondary term",
    "E34_timeout_primary_rank": "timestamp-local rank of no-timeout decisive-path quality",
    "E35_timeout_primary_utility": "positive utility only inside a no-timeout decisive-path envelope",
    "E36_timeout_primary_margin": "larger net edge only inside a no-timeout decisive-path envelope",
    "E37_timeout_primary_low_mae": "no-timeout decisive-path target with an explicit low-MAE gate",
    "E38_timeout_primary_loud": "no-timeout decisive-path target restricted to loud event context",
    "E39_timeout_primary_liquid_lowbarrier": "no-timeout decisive-path target with liquidity and low-barrier context",
    "E40_timeout_primary_rebound": "no-timeout decisive-path target restricted to rebound context",
}


def _parse_csv(value: str) -> list[str]:
    return [part.strip() for part in str(value).split(",") if part.strip()]


def _parse_float_csv(value: str) -> list[float]:
    return [float(part.strip()) for part in str(value).split(",") if part.strip()]


def _safe_numeric(values: Any) -> pd.Series:
    if isinstance(values, pd.Series):
        return pd.to_numeric(values, errors="coerce")
    return pd.to_numeric(pd.Series(values), errors="coerce")


def _safe_min(values: Any) -> float:
    series = _safe_numeric(values).dropna()
    return float(series.min()) if len(series) else float("nan")


def _safe_max(values: Any) -> float:
    series = _safe_numeric(values).dropna()
    return float(series.max()) if len(series) else float("nan")


def _weighted_mean(frame: pd.DataFrame, value_col: str, weight_col: str) -> float:
    if frame.empty or value_col not in frame.columns or weight_col not in frame.columns:
        return float("nan")
    values = _safe_numeric(frame[value_col])
    weights = _safe_numeric(frame[weight_col]).fillna(0.0)
    mask = values.notna() & weights.gt(0.0)
    if not bool(mask.any()):
        return float("nan")
    return float(np.average(values[mask], weights=weights[mask]))


def _rank_pct(values: pd.Series, *, high_good: bool = True) -> pd.Series:
    ranks = _safe_numeric(values).rank(method="average", pct=True)
    if not high_good:
        ranks = 1.0 - ranks
    return ranks.clip(0.0, 1.0)


def _xs_rank_feature(frame: pd.DataFrame, feature: str, *, high_good: bool = True) -> pd.Series:
    values = _safe_numeric(frame[feature])
    ranks = values.groupby(frame["__ts__"], dropna=False).rank(method="average", pct=True)
    if not high_good:
        ranks = 1.0 - ranks
    return ranks.clip(0.0, 1.0)


def _mean_available(parts: list[pd.Series], index: pd.Index) -> pd.Series:
    if not parts:
        return pd.Series(np.nan, index=index, dtype=np.float32)
    return pd.concat(parts, axis=1).mean(axis=1).astype(np.float32)


def _neutral_series(index: pd.Index, value: float = 0.5) -> pd.Series:
    return pd.Series(float(value), index=index, dtype=np.float32)


def _xs_rank_or_neutral(frame: pd.DataFrame, feature: str, *, high_good: bool = True) -> pd.Series:
    if feature not in frame.columns:
        return _neutral_series(frame.index)
    ranked = _xs_rank_feature(frame, feature, high_good=high_good)
    return ranked.fillna(0.5).astype(np.float32)


def _feature_or_neutral(frame: pd.DataFrame, feature: str) -> pd.Series:
    if feature not in frame.columns:
        return _neutral_series(frame.index)
    return _safe_numeric(frame[feature]).fillna(0.5).clip(0.0, 1.0).astype(np.float32)


def _run_entry_multiplier(
    frame: pd.DataFrame,
    base_soft: pd.Series,
    *,
    high_threshold: float = 0.92,
    gap_hours: float = 2.0,
    continuation_weight: float = 0.25,
) -> pd.Series:
    """Downweight repeated adjacent high-label rows from the same symbol.

    The label still uses future outcomes, but this avoids teaching the proxy
    that every row in an ex-post clean run is equally good. It is a label
    shaping diagnostic, not an inference feature.
    """

    out = pd.Series(1.0, index=frame.index, dtype=np.float32)
    work = pd.DataFrame(
        {
            "__ts__": pd.to_datetime(frame["__ts__"], errors="coerce"),
            "__symbol__": frame["__symbol__"].astype(str),
            "__soft__": _safe_numeric(base_soft).fillna(0.0),
            "__idx__": np.arange(len(frame), dtype=np.int64),
        },
        index=frame.index,
    ).sort_values(["__symbol__", "__ts__"], kind="mergesort")
    gap = pd.Timedelta(hours=float(gap_hours))
    for _, group in work.groupby("__symbol__", sort=False):
        prev_high_ts: pd.Timestamp | None = None
        for _, row in group.iterrows():
            ts = row["__ts__"]
            pos = int(row["__idx__"])
            is_high = bool(row["__soft__"] >= float(high_threshold))
            if is_high and prev_high_ts is not None and pd.notna(ts) and ts - prev_high_ts <= gap:
                out.iloc[pos] = float(continuation_weight)
            if is_high and pd.notna(ts):
                prev_high_ts = ts
    return out


def _event_confirmation_features(
    frame: pd.DataFrame,
    *,
    event_features: list[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    present = [feature for feature in event_features if feature in frame.columns]
    if not present:
        return pd.DataFrame(index=frame.index), {
            "enabled": True,
            "requested_features": list(event_features),
            "retained_features": [],
            "feature_count": 0,
            "mean_finite_frac": 0.0,
            "min_finite_frac": 0.0,
        }

    out: dict[str, pd.Series] = {}

    def add_rank(feature: str, *, high_good: bool = True) -> pd.Series | None:
        if feature not in frame.columns:
            return None
        suffix = "hi" if high_good else "lo"
        name = f"event_xs_{suffix}_{feature}"
        ranked = _xs_rank_feature(frame, feature, high_good=high_good).astype(np.float32)
        out[name] = ranked
        return ranked

    high_rank_features = [
        "second_leg_accel_1h",
        "second_leg_accel_2h",
        "second_leg_accel_vol_1h",
        "shock_12h",
        "shock_vol_ratio",
        "breakout_24h",
        "breakout_confirmed",
        "breakout_soft",
        "pct_breakout_t",
        "climax_range_24",
        "climax_vol_12",
        "jump_intensity",
        "impulse",
        "impulse_ratio_24",
        "vw_breakout",
        "retest_quality",
        "rejection_proxy",
        "volume_trend_alignment",
        "dip_volume_profile",
        "volume_capitulation",
        "progress",
        "speed",
        "oi_rank",
        "oi_chg_2h",
        "oi_up_agree",
        "leverage_build_score",
        "range_12h_pct",
        "range_24h_pct",
        "range_pct",
        "body_pct",
        "wick_ratio_4h_max",
        "loc_range_pos_24",
        "loc_prev_week_range_pos_24",
        "xasset_ob_liquidity_peer_resid",
    ]
    low_rank_features = [
        "time_since_event_extreme_12h",
        "spread_proxy_hl_range_bps_robust_z",
        "spread_proxy_abs_return_bps_robust_z",
        "median_spread_bps",
        "distance_to_support_daily_vwap_atr",
        "distance_to_resistance_daily_vwap_atr",
        "up_barrier_pressure_daily_vwap",
        "down_barrier_pressure_daily_vwap",
        "pullback_2",
        "pullback_4",
        "pullback_8",
        "pullback_48",
        "dist_from_low_event_12h",
        "dist_from_low_48h",
        "dist_from_high_48h",
        "oiw_pos_delta_entry_dist_1d_atr",
        "oiw_pos_delta_entry_dist_7d_atr",
        "oiw_pos_delta_entry_dist_14d_atr",
        "dist_oiw_abs_delta_12h_atr",
        "dist_oiw_signed_delta_12h_atr",
    ]
    ranks_hi = {
        feature: add_rank(feature, high_good=True)
        for feature in high_rank_features
        if feature in present
    }
    ranks_lo = {
        feature: add_rank(feature, high_good=False)
        for feature in low_rank_features
        if feature in present
    }

    idx = frame.index
    fresh = _mean_available(
        [rank for name, rank in ranks_lo.items() if name in {"time_since_event_extreme_12h"} and rank is not None],
        idx,
    )
    impulse = _mean_available(
        [
            ranks_hi[name]
            for name in [
                "second_leg_accel_1h",
                "second_leg_accel_2h",
                "second_leg_accel_vol_1h",
                "shock_12h",
                "shock_vol_ratio",
                "impulse",
                "impulse_ratio_24",
                "jump_intensity",
                "progress",
                "speed",
            ]
            if name in ranks_hi and ranks_hi[name] is not None
        ],
        idx,
    )
    confirmation = _mean_available(
        [
            ranks_hi[name]
            for name in [
                "breakout_24h",
                "breakout_confirmed",
                "breakout_soft",
                "pct_breakout_t",
                "vw_breakout",
                "retest_quality",
                "volume_trend_alignment",
                "rejection_proxy",
            ]
            if name in ranks_hi and ranks_hi[name] is not None
        ],
        idx,
    )
    liquidity = _mean_available(
        [
            rank
            for name, rank in ranks_lo.items()
            if name
            in {
                "spread_proxy_hl_range_bps_robust_z",
                "spread_proxy_abs_return_bps_robust_z",
                "median_spread_bps",
            }
            and rank is not None
        ]
        + [
            ranks_hi[name]
            for name in ["xasset_ob_liquidity_peer_resid"]
            if name in ranks_hi and ranks_hi[name] is not None
        ],
        idx,
    )
    oi_context = _mean_available(
        [
            ranks_hi[name]
            for name in ["oi_rank", "oi_chg_2h", "oi_up_agree", "leverage_build_score"]
            if name in ranks_hi and ranks_hi[name] is not None
        ]
        + [
            rank
            for name, rank in ranks_lo.items()
            if name
            in {
                "oiw_pos_delta_entry_dist_1d_atr",
                "oiw_pos_delta_entry_dist_7d_atr",
                "oiw_pos_delta_entry_dist_14d_atr",
                "dist_oiw_abs_delta_12h_atr",
                "dist_oiw_signed_delta_12h_atr",
            }
            and rank is not None
        ],
        idx,
    )
    low_barrier_context = _mean_available(
        [
            rank
            for name, rank in ranks_lo.items()
            if name
            in {
                "distance_to_support_daily_vwap_atr",
                "distance_to_resistance_daily_vwap_atr",
                "up_barrier_pressure_daily_vwap",
                "down_barrier_pressure_daily_vwap",
            }
            and rank is not None
        ],
        idx,
    )
    pullback_control = _mean_available(
        [
            rank
            for name, rank in ranks_lo.items()
            if name
            in {
                "pullback_2",
                "pullback_4",
                "pullback_8",
                "pullback_48",
                "dist_from_low_event_12h",
                "dist_from_low_48h",
                "dist_from_high_48h",
            }
            and rank is not None
        ],
        idx,
    )

    composite_parts = {
        "event_freshness": fresh,
        "event_impulse": impulse,
        "event_confirmation": confirmation,
        "event_liquidity_quality": liquidity,
        "event_oi_context": oi_context,
        "event_low_barrier_context": low_barrier_context,
        "event_pullback_control": pullback_control,
    }
    for name, values in composite_parts.items():
        out[name] = values.astype(np.float32)

    def product(name: str, left: pd.Series, right: pd.Series) -> None:
        out[name] = (left.fillna(0.5) * right.fillna(0.5)).astype(np.float32)

    product("event_fresh_impulse", fresh, impulse)
    product("event_confirmed_impulse", confirmation, impulse)
    product("event_confirmed_liquid_impulse", confirmation, liquidity * impulse)
    product("event_oi_confirmed_impulse", oi_context, confirmation * impulse)
    product("event_lowbarrier_confirmed", low_barrier_context, confirmation)
    product("event_pullback_confirmed", pullback_control, confirmation)
    product("event_clean_breakout_context", liquidity, low_barrier_context * confirmation)

    features = pd.DataFrame(out, index=frame.index)
    finite = features.notna().mean()
    return features, {
        "enabled": True,
        "requested_features": list(event_features),
        "retained_features": present,
        "feature_count": int(features.shape[1]),
        "mean_finite_frac": float(finite.mean()) if len(finite) else 0.0,
        "min_finite_frac": float(finite.min()) if len(finite) else 0.0,
        "composite_features": [
            "event_freshness",
            "event_impulse",
            "event_confirmation",
            "event_liquidity_quality",
            "event_oi_context",
            "event_low_barrier_context",
            "event_pullback_control",
            "event_fresh_impulse",
            "event_confirmed_impulse",
            "event_confirmed_liquid_impulse",
            "event_oi_confirmed_impulse",
            "event_lowbarrier_confirmed",
            "event_pullback_confirmed",
            "event_clean_breakout_context",
        ],
    }


def _causal_outcome_prior_features(
    frame: pd.DataFrame,
    metrics: pd.DataFrame,
    *,
    windows_days: list[float],
    embargo_hours: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    ts = pd.to_datetime(frame["__ts__"], utc=True, errors="coerce")
    ts_ns = ts.astype("int64").to_numpy(dtype=np.int64, copy=False)
    embargo_ns = int(float(embargo_hours) * 60.0 * 60.0 * 1_000_000_000)
    u = _safe_numeric(metrics["u_policy_net"]).to_numpy(dtype=np.float64, copy=False)
    mae = _safe_numeric(metrics["mae_norm"]).to_numpy(dtype=np.float64, copy=False)
    barrier = _safe_numeric(metrics["barrier"]).to_numpy(dtype=np.float64, copy=False)
    timeout = _safe_numeric(metrics["is_timeout"].astype(float)).fillna(1.0).to_numpy(dtype=np.float64, copy=False)
    mfe = _safe_numeric(metrics["mfe_norm"]).to_numpy(dtype=np.float64, copy=False)
    mfe_mae = np.divide(
        mfe,
        np.clip(mae, 0.25, None),
        out=np.full_like(mfe, np.nan, dtype=np.float64),
        where=np.isfinite(mfe) & np.isfinite(mae),
    )
    values = {
        "mean_u": u,
        "hit_u": (u > 0.0).astype(float),
        "bad_mae": (mae >= 1.0).astype(float),
        "clean": (
            (u > 0.0)
            & (mae <= 1.0)
            & (barrier <= 0.025)
            & (timeout <= 0.0)
        ).astype(float),
        "bounded": (
            (u > 0.0)
            & (mae <= 1.0)
            & (barrier <= 0.035)
            & (mfe_mae >= 1.25)
            & (timeout <= 0.0)
        ).astype(float),
    }

    out: dict[str, np.ndarray] = {}

    def fill_scope(scope: str, positions: np.ndarray) -> None:
        if len(positions) == 0:
            return
        order = np.argsort(ts_ns[positions], kind="mergesort")
        sorted_positions = positions[order]
        sorted_ts = ts_ns[sorted_positions]
        for window_days in windows_days:
            window_ns = int(float(window_days) * 24.0 * 60.0 * 60.0 * 1_000_000_000)
            right = np.searchsorted(sorted_ts, sorted_ts - embargo_ns, side="right")
            left = np.searchsorted(sorted_ts, sorted_ts - embargo_ns - window_ns, side="left")
            counts = (right - left).astype(np.float64)
            count_col = f"prior_{scope}_count_{window_days:g}d"
            out.setdefault(count_col, np.full(len(frame), np.nan, dtype=np.float32))
            out[count_col][sorted_positions] = counts.astype(np.float32)
            for metric_name, raw_values in values.items():
                sorted_values = raw_values[sorted_positions].astype(np.float64, copy=False)
                finite = np.isfinite(sorted_values)
                cumulative_values = np.concatenate([[0.0], np.cumsum(np.where(finite, sorted_values, 0.0))])
                cumulative_counts = np.concatenate([[0.0], np.cumsum(finite.astype(np.float64))])
                numerator = cumulative_values[right] - cumulative_values[left]
                denominator = cumulative_counts[right] - cumulative_counts[left]
                means = np.divide(
                    numerator,
                    denominator,
                    out=np.full_like(numerator, np.nan, dtype=np.float64),
                    where=denominator > 0.0,
                )
                col = f"prior_{scope}_{metric_name}_{window_days:g}d"
                out.setdefault(col, np.full(len(frame), np.nan, dtype=np.float32))
                out[col][sorted_positions] = means.astype(np.float32)

    all_positions = np.arange(len(frame), dtype=np.int64)
    fill_scope("global", all_positions)
    for _, idx in frame.groupby("__symbol__", sort=False).indices.items():
        fill_scope("symbol", np.asarray(idx, dtype=np.int64))

    priors = pd.DataFrame(out, index=frame.index)
    finite = priors.notna().mean()
    return priors, {
        "enabled": True,
        "embargo_hours": float(embargo_hours),
        "windows_days": [float(v) for v in windows_days],
        "feature_count": int(priors.shape[1]),
        "mean_finite_frac": float(finite.mean()) if len(finite) else 0.0,
        "min_finite_frac": float(finite.min()) if len(finite) else 0.0,
    }


def _causal_state_path_prior_features(
    frame: pd.DataFrame,
    metrics: pd.DataFrame,
    *,
    state_features: list[str],
    windows_days: list[float],
    embargo_hours: float,
    min_bucket_rows: int = 20,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    present_features = [feature for feature in state_features if feature in frame.columns]
    if not present_features:
        return pd.DataFrame(index=frame.index), {
            "enabled": True,
            "embargo_hours": float(embargo_hours),
            "windows_days": [float(v) for v in windows_days],
            "requested_state_features": list(state_features),
            "retained_state_features": [],
            "feature_count": 0,
            "mean_finite_frac": 0.0,
            "min_finite_frac": 0.0,
        }

    ts = pd.to_datetime(frame["__ts__"], utc=True, errors="coerce")
    ts_ns = ts.astype("int64").to_numpy(dtype=np.int64, copy=False)
    valid_ts = ts.notna().to_numpy(dtype=bool, copy=False)
    embargo_ns = int(float(embargo_hours) * 60.0 * 60.0 * 1_000_000_000)
    u = _safe_numeric(metrics["u_policy_net"]).to_numpy(dtype=np.float64, copy=False)
    mae = _safe_numeric(metrics["mae_norm"]).to_numpy(dtype=np.float64, copy=False)
    barrier = _safe_numeric(metrics["barrier"]).to_numpy(dtype=np.float64, copy=False)
    timeout = _safe_numeric(metrics["is_timeout"].astype(float)).to_numpy(dtype=np.float64, copy=False)
    mfe = _safe_numeric(metrics["mfe_norm"]).to_numpy(dtype=np.float64, copy=False)
    mfe_mae = np.divide(
        mfe,
        np.clip(mae, 0.25, None),
        out=np.full_like(mfe, np.nan, dtype=np.float64),
        where=np.isfinite(mfe) & np.isfinite(mae),
    )
    bad_mae = np.where(np.isfinite(mae), (mae >= 1.0).astype(float), np.nan)
    wide_25 = np.where(np.isfinite(barrier), (barrier > 0.025).astype(float), np.nan)
    timeout_value = np.where(np.isfinite(timeout), timeout, np.nan)
    clean = np.where(
        np.isfinite(u) & np.isfinite(mae) & np.isfinite(barrier) & np.isfinite(timeout),
        ((u > 0.0) & (mae <= 1.0) & (barrier <= 0.025) & (timeout <= 0.0)).astype(float),
        np.nan,
    )
    bounded = np.where(
        np.isfinite(u) & np.isfinite(mae) & np.isfinite(barrier) & np.isfinite(timeout) & np.isfinite(mfe_mae),
        (
            (u > 0.0)
            & (mae <= 1.0)
            & (barrier <= 0.035)
            & (mfe_mae >= 1.25)
            & (timeout <= 0.0)
        ).astype(float),
        np.nan,
    )
    values = {
        "mean_u": u,
        "hit_u": np.where(np.isfinite(u), (u > 0.0).astype(float), np.nan),
        "bad_mae": bad_mae,
        "wide_25": wide_25,
        "timeout": timeout_value,
        "mae_norm": mae,
        "mfe_mae": mfe_mae,
        "clean": clean,
        "bounded": bounded,
    }
    out: dict[str, np.ndarray] = {}

    def fill_positions(prefix: str, positions: np.ndarray) -> None:
        if len(positions) < int(min_bucket_rows):
            return
        order = np.argsort(ts_ns[positions], kind="mergesort")
        sorted_positions = positions[order]
        sorted_ts = ts_ns[sorted_positions]
        for window_days in windows_days:
            window_ns = int(float(window_days) * 24.0 * 60.0 * 60.0 * 1_000_000_000)
            right = np.searchsorted(sorted_ts, sorted_ts - embargo_ns, side="right")
            left = np.searchsorted(sorted_ts, sorted_ts - embargo_ns - window_ns, side="left")
            counts = (right - left).astype(np.float64)
            count_col = f"{prefix}_count_{window_days:g}d"
            out.setdefault(count_col, np.full(len(frame), np.nan, dtype=np.float32))
            out[count_col][sorted_positions] = counts.astype(np.float32)
            for metric_name, raw_values in values.items():
                sorted_values = raw_values[sorted_positions].astype(np.float64, copy=False)
                finite = np.isfinite(sorted_values)
                cumulative_values = np.concatenate([[0.0], np.cumsum(np.where(finite, sorted_values, 0.0))])
                cumulative_counts = np.concatenate([[0.0], np.cumsum(finite.astype(np.float64))])
                numerator = cumulative_values[right] - cumulative_values[left]
                denominator = cumulative_counts[right] - cumulative_counts[left]
                means = np.divide(
                    numerator,
                    denominator,
                    out=np.full_like(numerator, np.nan, dtype=np.float64),
                    where=denominator > 0.0,
                )
                col = f"{prefix}_{metric_name}_{window_days:g}d"
                out.setdefault(col, np.full(len(frame), np.nan, dtype=np.float32))
                out[col][sorted_positions] = means.astype(np.float32)

    bucket_counts: dict[str, dict[str, int]] = {}
    for feature in present_features:
        values_ser = _safe_numeric(frame[feature])
        xs_rank = values_ser.groupby(frame["__ts__"], dropna=False).rank(method="average", pct=True)
        rank_col = f"xs_rank_{feature}"
        out[rank_col] = xs_rank.to_numpy(dtype=np.float32, copy=True)
        rank_values = xs_rank.to_numpy(dtype=np.float64, copy=False)
        buckets = np.full(len(frame), -1, dtype=np.int8)
        buckets[(rank_values > 0.0) & (rank_values <= (1.0 / 3.0))] = 0
        buckets[(rank_values > (1.0 / 3.0)) & (rank_values <= (2.0 / 3.0))] = 1
        buckets[rank_values > (2.0 / 3.0)] = 2
        feature_counts: dict[str, int] = {}
        for bucket in (0, 1, 2):
            positions = np.flatnonzero((buckets == bucket) & valid_ts)
            feature_counts[str(bucket)] = int(len(positions))
            fill_positions(f"prior_xs_state_{feature}_b{bucket}", positions.astype(np.int64, copy=False))
        bucket_counts[feature] = feature_counts

    priors = pd.DataFrame(out, index=frame.index)
    finite = priors.notna().mean()
    return priors, {
        "enabled": True,
        "embargo_hours": float(embargo_hours),
        "windows_days": [float(v) for v in windows_days],
        "requested_state_features": list(state_features),
        "retained_state_features": present_features,
        "bucket_counts": bucket_counts,
        "feature_count": int(priors.shape[1]),
        "mean_finite_frac": float(finite.mean()) if len(finite) else 0.0,
        "min_finite_frac": float(finite.min()) if len(finite) else 0.0,
    }


def _target_frame(soft: pd.Series, hard: pd.Series) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "target_soft": _safe_numeric(soft).clip(0.0, 1.0),
            "target_hard": pd.Series(hard, index=soft.index).fillna(False).astype(float),
        },
        index=soft.index,
    )


def _rank_array(values: Any) -> np.ndarray:
    return _safe_numeric(values).rank(method="average").to_numpy(dtype=np.float64, copy=True)


def _rank_corr(left_rank: np.ndarray, right_rank: np.ndarray) -> float:
    mask = np.isfinite(left_rank) & np.isfinite(right_rank)
    if int(mask.sum()) < 3:
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


def _extra_targets(frame: pd.DataFrame, metrics: pd.DataFrame) -> dict[str, pd.DataFrame]:
    u = _safe_numeric(metrics["u_policy_net"]).fillna(-0.02)
    mae = _safe_numeric(metrics["mae_norm"]).fillna(10.0)
    mfe = _safe_numeric(metrics["mfe_norm"]).fillna(0.0)
    barrier = _safe_numeric(metrics["barrier"]).fillna(0.0)
    bars = _safe_numeric(metrics["bars_to_mfe"]).fillna(24.0)
    timeout = _safe_numeric(metrics["is_timeout"].astype(float)).fillna(1.0).clip(0.0, 1.0)
    mfe_mae = (mfe / mae.clip(lower=0.25)).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(upper=10.0)

    utility = pd.Series(_sigmoid((u - 0.0015) / 0.008), index=metrics.index).clip(0.0, 1.0)
    margin_utility = pd.Series(_sigmoid((u - 0.0040) / 0.007), index=metrics.index).clip(0.0, 1.0)

    strict_clean = (
        pd.Series(_sigmoid((0.75 - mae) / 0.20), index=metrics.index)
        * pd.Series(_sigmoid((0.022 - barrier) / 0.005), index=metrics.index)
        * pd.Series(_sigmoid((10.0 - bars) / 4.0), index=metrics.index)
        * (1.0 - timeout)
    ).clip(0.0, 1.0)
    bounded_clean = (
        pd.Series(_sigmoid((1.00 - mae) / 0.25), index=metrics.index)
        * pd.Series(_sigmoid((0.030 - barrier) / 0.006), index=metrics.index)
        * pd.Series(_sigmoid((16.0 - bars) / 5.0), index=metrics.index)
        * (1.0 - timeout)
    ).clip(0.0, 1.0)
    ratio_clean = (
        pd.Series(_sigmoid((mfe_mae - 1.35) / 0.35), index=metrics.index)
        * pd.Series(_sigmoid((mfe - 1.25) / 0.35), index=metrics.index)
        * pd.Series(_sigmoid((0.95 - mae) / 0.25), index=metrics.index)
        * (1.0 - timeout)
    ).clip(0.0, 1.0)
    fast_clean = (
        pd.Series(_sigmoid((mfe - 1.0) / 0.30), index=metrics.index)
        * pd.Series(_sigmoid((8.0 - bars) / 3.0), index=metrics.index)
        * pd.Series(_sigmoid((0.90 - mae) / 0.25), index=metrics.index)
        * (1.0 - timeout)
    ).clip(0.0, 1.0)
    low_barrier = (
        strict_clean
        * pd.Series(_sigmoid((0.018 - barrier) / 0.004), index=metrics.index)
    ).clip(0.0, 1.0)
    low_mae_clean = (
        pd.Series(_sigmoid((0.60 - mae) / 0.15), index=metrics.index)
        * pd.Series(_sigmoid((0.022 - barrier) / 0.005), index=metrics.index)
        * pd.Series(_sigmoid((12.0 - bars) / 4.0), index=metrics.index)
        * (1.0 - timeout)
    ).clip(0.0, 1.0)
    low_mae_ratio = (
        pd.Series(_sigmoid((0.70 - mae) / 0.18), index=metrics.index)
        * pd.Series(_sigmoid((mfe_mae - 1.50) / 0.30), index=metrics.index)
        * pd.Series(_sigmoid((mfe - 1.25) / 0.30), index=metrics.index)
        * (1.0 - timeout)
    ).clip(0.0, 1.0)

    loud_intensity = _mean_available(
        [
            _xs_rank_or_neutral(frame, "speed", high_good=True),
            _xs_rank_or_neutral(frame, "shock_12h", high_good=True),
            _xs_rank_or_neutral(frame, "shock_vol_ratio", high_good=True),
            _xs_rank_or_neutral(frame, "breakout_24h", high_good=True),
            _xs_rank_or_neutral(frame, "progress", high_good=True),
            _xs_rank_or_neutral(frame, "range_24h_pct", high_good=True),
            _xs_rank_or_neutral(frame, "impulse_ratio_24", high_good=True),
        ],
        frame.index,
    ).fillna(0.5)
    quiet_continuation = pd.Series(
        _sigmoid((0.72 - loud_intensity) / 0.12),
        index=metrics.index,
    ).clip(0.0, 1.0)
    oi_location_score = _mean_available(
        [
            _xs_rank_or_neutral(frame, "oi_up_agree", high_good=True),
            _xs_rank_or_neutral(frame, "oi_chg_2h", high_good=True),
            _xs_rank_or_neutral(frame, "loc_prev_week_range_pos_24", high_good=True),
            _xs_rank_or_neutral(frame, "loc_range_pos_24", high_good=True),
            _feature_or_neutral(frame, "event_xs_lo_oiw_pos_delta_entry_dist_1d_atr"),
            _feature_or_neutral(frame, "event_xs_lo_oiw_pos_delta_entry_dist_7d_atr"),
            _feature_or_neutral(frame, "event_xs_lo_oiw_pos_delta_entry_dist_14d_atr"),
        ],
        frame.index,
    ).fillna(0.5)
    oi_location = pd.Series(
        _sigmoid((oi_location_score - 0.56) / 0.12),
        index=metrics.index,
    ).clip(0.0, 1.0)
    liquidity_score = _mean_available(
        [
            _feature_or_neutral(frame, "event_xs_lo_spread_proxy_hl_range_bps_robust_z"),
            _feature_or_neutral(frame, "event_xs_lo_spread_proxy_abs_return_bps_robust_z"),
            _feature_or_neutral(frame, "event_xs_lo_median_spread_bps"),
            _xs_rank_or_neutral(frame, "xasset_ob_liquidity_peer_resid", high_good=True),
        ],
        frame.index,
    ).fillna(0.5)
    low_barrier_context_score = _mean_available(
        [
            _feature_or_neutral(frame, "event_xs_lo_distance_to_support_daily_vwap_atr"),
            _feature_or_neutral(frame, "event_xs_lo_distance_to_resistance_daily_vwap_atr"),
            _feature_or_neutral(frame, "event_xs_lo_up_barrier_pressure_daily_vwap"),
            _feature_or_neutral(frame, "event_xs_lo_down_barrier_pressure_daily_vwap"),
        ],
        frame.index,
    ).fillna(0.5)
    loud_gate = pd.Series(
        _sigmoid((loud_intensity - 0.70) / 0.08),
        index=metrics.index,
    ).clip(0.0, 1.0)
    barrier_gate_25 = pd.Series(
        _sigmoid((0.025 - barrier) / 0.004),
        index=metrics.index,
    ).clip(0.0, 1.0)
    barrier_gate_30 = pd.Series(
        _sigmoid((0.030 - barrier) / 0.005),
        index=metrics.index,
    ).clip(0.0, 1.0)
    fast_gate = pd.Series(
        _sigmoid((8.0 - bars) / 3.0),
        index=metrics.index,
    ).clip(0.0, 1.0)
    liquid_gate = pd.Series(
        _sigmoid((liquidity_score - 0.55) / 0.10),
        index=metrics.index,
    ).clip(0.0, 1.0)
    low_barrier_context_gate = pd.Series(
        _sigmoid((low_barrier_context_score - 0.55) / 0.10),
        index=metrics.index,
    ).clip(0.0, 1.0)
    low_zscore_score = _xs_rank_or_neutral(frame, "zscore_price_200", high_good=False).fillna(0.5)
    low_atr_compression_score = _xs_rank_or_neutral(frame, "atr_compression_ratio", high_good=False).fillna(0.5)
    low_range_location_score = _xs_rank_or_neutral(frame, "loc_range_pos_24", high_good=False).fillna(0.5)
    rebound_context_score = _mean_available(
        [
            low_zscore_score,
            low_atr_compression_score,
            low_range_location_score,
            oi_location_score,
            liquidity_score,
        ],
        frame.index,
    ).fillna(0.5)
    rebound_gate = pd.Series(
        _sigmoid((rebound_context_score - 0.58) / 0.10),
        index=metrics.index,
    ).clip(0.0, 1.0)

    strict_utility = (utility * strict_clean).clip(0.0, 1.0)
    rank = strict_utility.groupby(frame["__ts__"], dropna=False).rank(method="average", pct=True)
    rank = rank.fillna(strict_utility.rank(method="average", pct=True)).clip(0.0, 1.0)
    timestamp_rank_clean = (0.50 * strict_utility + 0.50 * rank).clip(0.0, 1.0)
    low_mae_utility = (utility * low_mae_clean).clip(0.0, 1.0)
    low_mae_rank = low_mae_utility.groupby(frame["__ts__"], dropna=False).rank(method="average", pct=True)
    low_mae_rank = low_mae_rank.fillna(low_mae_utility.rank(method="average", pct=True)).clip(0.0, 1.0)
    fast_low_mae_rank = (0.45 * low_mae_utility + 0.55 * low_mae_rank).clip(0.0, 1.0)
    low_mae_ratio_utility = (utility * low_mae_ratio).clip(0.0, 1.0)
    run_entry = _run_entry_multiplier(
        frame,
        low_mae_ratio_utility,
        high_threshold=0.92,
        gap_hours=2.0,
        continuation_weight=0.25,
    )
    loud_range_efficiency = (
        pd.Series(_sigmoid((mfe_mae - 2.0) / 0.40), index=metrics.index)
        * pd.Series(_sigmoid((mfe - 2.0) / 0.50), index=metrics.index)
        * pd.Series(_sigmoid((0.80 - mae) / 0.20), index=metrics.index)
        * fast_gate
        * (1.0 - timeout)
    ).clip(0.0, 1.0)
    loud_barrier_low_mae = (low_mae_ratio_utility * loud_gate * barrier_gate_25).clip(0.0, 1.0)
    loud_range_utility = (
        utility
        * loud_gate
        * barrier_gate_30
        * loud_range_efficiency
    ).clip(0.0, 1.0)
    loud_liquid_low_barrier = (
        loud_range_utility
        * liquid_gate
        * low_barrier_context_gate
    ).clip(0.0, 1.0)
    loud_run_entry = _run_entry_multiplier(
        frame,
        loud_barrier_low_mae,
        high_threshold=0.90,
        gap_hours=2.0,
        continuation_weight=0.25,
    )
    bounded_high_mae_envelope = (
        pd.Series(_sigmoid((4.0 - mae) / 0.80), index=metrics.index)
        * pd.Series(_sigmoid((mfe_mae - 1.25) / 0.30), index=metrics.index)
        * pd.Series(_sigmoid((mfe - 1.50) / 0.45), index=metrics.index)
        * pd.Series(_sigmoid((0.035 - barrier) / 0.006), index=metrics.index)
        * pd.Series(_sigmoid((24.0 - bars) / 6.0), index=metrics.index)
        * (1.0 - timeout)
    ).clip(0.0, 1.0)
    bounded_high_mae_utility = (utility * bounded_high_mae_envelope).clip(0.0, 1.0)
    lowz_rebound_bounded = (bounded_high_mae_utility * rebound_gate).clip(0.0, 1.0)
    loud_bounded_event = (
        bounded_high_mae_utility
        * loud_gate
        * barrier_gate_30
    ).clip(0.0, 1.0)
    rebound_run_entry = _run_entry_multiplier(
        frame,
        lowz_rebound_bounded,
        high_threshold=0.85,
        gap_hours=2.0,
        continuation_weight=0.25,
    )
    fast_decisive_envelope = (
        pd.Series(_sigmoid((6.0 - bars) / 2.0), index=metrics.index)
        * pd.Series(_sigmoid((0.60 - mae) / 0.14), index=metrics.index)
        * pd.Series(_sigmoid((0.022 - barrier) / 0.004), index=metrics.index)
        * pd.Series(_sigmoid((mfe - 1.20) / 0.25), index=metrics.index)
        * pd.Series(_sigmoid((mfe_mae - 1.75) / 0.35), index=metrics.index)
        * (1.0 - timeout)
    ).clip(0.0, 1.0)
    fast_decisive = (utility * fast_decisive_envelope).clip(0.0, 1.0)
    fast_decisive_rank = fast_decisive.groupby(frame["__ts__"], dropna=False).rank(
        method="average",
        pct=True,
    )
    fast_decisive_rank = fast_decisive_rank.fillna(fast_decisive.rank(method="average", pct=True)).clip(0.0, 1.0)
    fast_decisive_rank_blend = (0.40 * fast_decisive + 0.60 * fast_decisive_rank).clip(0.0, 1.0)
    fast_margin = (margin_utility * fast_decisive_envelope).clip(0.0, 1.0)
    ultra_fast_envelope = (
        pd.Series(_sigmoid((4.0 - bars) / 1.5), index=metrics.index)
        * pd.Series(_sigmoid((0.42 - mae) / 0.10), index=metrics.index)
        * pd.Series(_sigmoid((0.018 - barrier) / 0.003), index=metrics.index)
        * pd.Series(_sigmoid((mfe - 1.10) / 0.20), index=metrics.index)
        * pd.Series(_sigmoid((mfe_mae - 2.00) / 0.35), index=metrics.index)
        * (1.0 - timeout)
    ).clip(0.0, 1.0)
    ultra_fast = (utility * ultra_fast_envelope).clip(0.0, 1.0)
    fast_clean_run_entry = _run_entry_multiplier(
        frame,
        fast_decisive,
        high_threshold=0.82,
        gap_hours=2.0,
        continuation_weight=0.20,
    )
    timeout_averse_path = (
        fast_decisive_envelope
        * pd.Series(_sigmoid((0.35 - mae) / 0.12), index=metrics.index)
        * pd.Series(_sigmoid((4.0 - bars) / 2.0), index=metrics.index)
        * (1.0 - timeout)
    ).clip(0.0, 1.0)
    timeout_averse_rank = timeout_averse_path.groupby(frame["__ts__"], dropna=False).rank(
        method="average",
        pct=True,
    )
    timeout_averse_rank = timeout_averse_rank.fillna(timeout_averse_path.rank(method="average", pct=True)).clip(0.0, 1.0)
    fast_low_timeout_rank = (0.30 * fast_decisive + 0.70 * timeout_averse_rank).clip(0.0, 1.0)
    no_timeout_decisive_path = (
        pd.Series(_sigmoid((8.0 - bars) / 2.5), index=metrics.index)
        * pd.Series(_sigmoid((mfe - 1.00) / 0.25), index=metrics.index)
        * pd.Series(_sigmoid((mfe_mae - 1.25) / 0.35), index=metrics.index)
        * pd.Series(_sigmoid((0.030 - barrier) / 0.006), index=metrics.index)
        * (1.0 - timeout)
    ).clip(0.0, 1.0)
    timeout_primary_decisive = (
        no_timeout_decisive_path
        * (0.70 + 0.30 * utility)
    ).clip(0.0, 1.0)
    timeout_primary_rank = no_timeout_decisive_path.groupby(frame["__ts__"], dropna=False).rank(
        method="average",
        pct=True,
    )
    timeout_primary_rank = timeout_primary_rank.fillna(no_timeout_decisive_path.rank(method="average", pct=True)).clip(
        0.0,
        1.0,
    )
    timeout_primary_rank_blend = (
        0.25 * no_timeout_decisive_path
        + 0.75 * timeout_primary_rank
    ).clip(0.0, 1.0)
    timeout_primary_utility = (utility * no_timeout_decisive_path).clip(0.0, 1.0)
    timeout_primary_margin = (margin_utility * no_timeout_decisive_path).clip(0.0, 1.0)
    timeout_primary_low_mae = (
        no_timeout_decisive_path
        * pd.Series(_sigmoid((0.80 - mae) / 0.20), index=metrics.index)
        * (0.65 + 0.35 * utility)
    ).clip(0.0, 1.0)
    timeout_primary_loud = (timeout_primary_decisive * loud_gate).clip(0.0, 1.0)
    timeout_primary_liquid_lowbarrier = (
        timeout_primary_decisive
        * liquid_gate
        * low_barrier_context_gate
    ).clip(0.0, 1.0)
    timeout_primary_rebound = (timeout_primary_decisive * rebound_gate).clip(0.0, 1.0)

    hard_clean = (
        (u > 0.0)
        & (mae <= 1.0)
        & (barrier <= 0.025)
        & (bars <= 14.0)
        & (timeout <= 0.0)
    )
    hard_low_mae = (
        (u > 0.0)
        & (mae <= 0.70)
        & (barrier <= 0.025)
        & (bars <= 12.0)
        & (timeout <= 0.0)
    )
    hard_low_mae_ratio = hard_low_mae & (mfe_mae >= 1.50)
    hard_quiet = hard_low_mae_ratio & (quiet_continuation >= 0.50) & (loud_intensity <= 0.78)
    hard_oi_location = hard_low_mae_ratio & (oi_location >= 0.50)
    hard_run_entry = hard_low_mae_ratio & (run_entry >= 0.50)
    hard_loud_barrier = hard_low_mae_ratio & (loud_intensity >= 0.70) & (barrier <= 0.025)
    hard_loud_efficiency = (
        (u > 0.0)
        & (loud_intensity >= 0.70)
        & (mfe >= 2.0)
        & (mae <= 0.80)
        & (mfe_mae >= 2.0)
        & (barrier <= 0.030)
        & (bars <= 12.0)
        & (timeout <= 0.0)
    )
    hard_loud_liquid_low_barrier = (
        hard_loud_efficiency
        & (liquidity_score >= 0.55)
        & (low_barrier_context_score >= 0.55)
    )
    hard_bounded_high_mae = (
        (u > 0.0)
        & (mae <= 4.0)
        & (mfe >= 1.50)
        & (mfe_mae >= 1.25)
        & (barrier <= 0.035)
        & (bars <= 24.0)
        & (timeout <= 0.0)
    )
    hard_lowz_rebound_bounded = hard_bounded_high_mae & (rebound_context_score >= 0.58)
    hard_loud_bounded_event = hard_bounded_high_mae & (loud_intensity >= 0.70) & (barrier <= 0.030)
    hard_fast_decisive = (
        (u > 0.0)
        & (mae <= 0.60)
        & (barrier <= 0.022)
        & (mfe >= 1.20)
        & (mfe_mae >= 1.75)
        & (bars <= 6.0)
        & (timeout <= 0.0)
    )
    hard_fast_margin = hard_fast_decisive & (u > 0.0040)
    hard_ultra_fast = (
        (u > 0.0)
        & (mae <= 0.42)
        & (barrier <= 0.018)
        & (mfe >= 1.10)
        & (mfe_mae >= 2.00)
        & (bars <= 4.0)
        & (timeout <= 0.0)
    )
    hard_fast_loud = hard_fast_decisive & (loud_intensity >= 0.70)
    hard_fast_rebound = hard_fast_decisive & (rebound_context_score >= 0.58)
    hard_fast_run_entry = hard_fast_decisive & (fast_clean_run_entry >= 0.50)
    hard_timeout_primary_decisive = (
        (u > 0.0)
        & (mfe >= 1.00)
        & (mfe_mae >= 1.25)
        & (barrier <= 0.030)
        & (bars <= 8.0)
        & (timeout <= 0.0)
    )
    hard_timeout_primary_rank = hard_timeout_primary_decisive & (timeout_primary_rank >= 0.70)
    hard_timeout_primary_margin = hard_timeout_primary_decisive & (u > 0.0040)
    hard_timeout_primary_low_mae = hard_timeout_primary_decisive & (mae <= 0.80) & (barrier <= 0.025)
    hard_timeout_primary_loud = hard_timeout_primary_decisive & (loud_intensity >= 0.70)
    hard_timeout_primary_liquid_lowbarrier = (
        hard_timeout_primary_decisive
        & (liquidity_score >= 0.55)
        & (low_barrier_context_score >= 0.55)
    )
    hard_timeout_primary_rebound = hard_timeout_primary_decisive & (rebound_context_score >= 0.58)
    quiet_oi = (quiet_continuation * oi_location).clip(0.0, 1.0)
    return {
        "E1_strict_clean_utility": _target_frame(strict_utility, hard_clean & (mae <= 0.75)),
        "E2_bounded_clean_utility": _target_frame((utility * bounded_clean).clip(0.0, 1.0), hard_clean),
        "E3_mfe_mae_ratio_clean": _target_frame((utility * ratio_clean).clip(0.0, 1.0), hard_clean & (mfe_mae >= 1.25)),
        "E4_fast_clean_mfe": _target_frame((utility * fast_clean).clip(0.0, 1.0), hard_clean & (bars <= 8.0)),
        "E5_margin_clean_utility": _target_frame((margin_utility * strict_clean).clip(0.0, 1.0), hard_clean & (u > 0.0040)),
        "E6_low_barrier_clean": _target_frame((utility * low_barrier).clip(0.0, 1.0), hard_clean & (barrier <= 0.018)),
        "E7_timestamp_rank_clean": _target_frame(timestamp_rank_clean, timestamp_rank_clean >= 0.70),
        "E8_low_mae_net_utility": _target_frame(low_mae_utility, hard_low_mae),
        "E9_low_mae_mfe_ratio": _target_frame(low_mae_ratio_utility, hard_low_mae_ratio),
        "E10_fast_low_mae_rank": _target_frame(fast_low_mae_rank, fast_low_mae_rank >= 0.75),
        "E11_low_mae_margin_utility": _target_frame((margin_utility * low_mae_clean).clip(0.0, 1.0), hard_low_mae & (u > 0.0040)),
        "E12_quiet_low_mae_mfe_ratio": _target_frame(
            (low_mae_ratio_utility * quiet_continuation).clip(0.0, 1.0),
            hard_quiet,
        ),
        "E13_oi_location_low_mae": _target_frame(
            (low_mae_ratio_utility * oi_location).clip(0.0, 1.0),
            hard_oi_location,
        ),
        "E14_run_entry_low_mae": _target_frame(
            (low_mae_ratio_utility * run_entry).clip(0.0, 1.0),
            hard_run_entry,
        ),
        "E15_quiet_oi_low_mae": _target_frame(
            (low_mae_ratio_utility * quiet_oi).clip(0.0, 1.0),
            hard_quiet & hard_oi_location,
        ),
        "E16_quiet_oi_run_entry": _target_frame(
            (low_mae_ratio_utility * quiet_oi * run_entry).clip(0.0, 1.0),
            hard_quiet & hard_oi_location & hard_run_entry,
        ),
        "E17_loud_barrier_low_mae": _target_frame(
            loud_barrier_low_mae,
            hard_loud_barrier,
        ),
        "E18_loud_range_efficiency": _target_frame(
            loud_range_utility,
            hard_loud_efficiency,
        ),
        "E19_loud_liquid_low_barrier": _target_frame(
            loud_liquid_low_barrier,
            hard_loud_liquid_low_barrier,
        ),
        "E20_loud_run_entry_barrier": _target_frame(
            (loud_barrier_low_mae * loud_run_entry).clip(0.0, 1.0),
            hard_loud_barrier & (loud_run_entry >= 0.50),
        ),
        "E21_bounded_high_mae_efficiency": _target_frame(
            bounded_high_mae_utility,
            hard_bounded_high_mae,
        ),
        "E22_lowz_rebound_bounded": _target_frame(
            lowz_rebound_bounded,
            hard_lowz_rebound_bounded,
        ),
        "E23_loud_bounded_event_efficiency": _target_frame(
            loud_bounded_event,
            hard_loud_bounded_event,
        ),
        "E24_rebound_run_entry_bounded": _target_frame(
            (lowz_rebound_bounded * rebound_run_entry).clip(0.0, 1.0),
            hard_lowz_rebound_bounded & (rebound_run_entry >= 0.50),
        ),
        "E25_fast_decisive_low_mae": _target_frame(
            fast_decisive,
            hard_fast_decisive,
        ),
        "E26_fast_decisive_rank": _target_frame(
            fast_decisive_rank_blend,
            hard_fast_decisive & (fast_decisive_rank >= 0.70),
        ),
        "E27_fast_margin_no_timeout": _target_frame(
            fast_margin,
            hard_fast_margin,
        ),
        "E28_ultra_low_mae_fast": _target_frame(
            ultra_fast,
            hard_ultra_fast,
        ),
        "E29_fast_loud_clean": _target_frame(
            (fast_decisive * loud_gate).clip(0.0, 1.0),
            hard_fast_loud,
        ),
        "E30_fast_rebound_clean": _target_frame(
            (fast_decisive * rebound_gate).clip(0.0, 1.0),
            hard_fast_rebound,
        ),
        "E31_fast_run_entry_clean": _target_frame(
            (fast_decisive * fast_clean_run_entry).clip(0.0, 1.0),
            hard_fast_run_entry,
        ),
        "E32_fast_low_timeout_rank": _target_frame(
            fast_low_timeout_rank,
            hard_fast_decisive & (timeout_averse_rank >= 0.75),
        ),
        "E33_timeout_primary_decisive_path": _target_frame(
            timeout_primary_decisive,
            hard_timeout_primary_decisive,
        ),
        "E34_timeout_primary_rank": _target_frame(
            timeout_primary_rank_blend,
            hard_timeout_primary_rank,
        ),
        "E35_timeout_primary_utility": _target_frame(
            timeout_primary_utility,
            hard_timeout_primary_decisive,
        ),
        "E36_timeout_primary_margin": _target_frame(
            timeout_primary_margin,
            hard_timeout_primary_margin,
        ),
        "E37_timeout_primary_low_mae": _target_frame(
            timeout_primary_low_mae,
            hard_timeout_primary_low_mae,
        ),
        "E38_timeout_primary_loud": _target_frame(
            timeout_primary_loud,
            hard_timeout_primary_loud,
        ),
        "E39_timeout_primary_liquid_lowbarrier": _target_frame(
            timeout_primary_liquid_lowbarrier,
            hard_timeout_primary_liquid_lowbarrier,
        ),
        "E40_timeout_primary_rebound": _target_frame(
            timeout_primary_rebound,
            hard_timeout_primary_rebound,
        ),
    }


def _all_targets(frame: pd.DataFrame, metrics: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    targets = _make_targets(frame, metrics)
    descriptions = {arm.name: arm.description for arm in LABEL_ARMS}
    extras = _extra_targets(frame, metrics)
    targets.update(extras)
    descriptions.update(EXTRA_ARM_DESCRIPTIONS)
    return targets, descriptions


def _proxy_score(
    *,
    train: pd.DataFrame,
    valid: pd.DataFrame,
    features: list[str],
    target_train: pd.Series,
    metrics_train: pd.DataFrame | None = None,
    top_k: int,
    proxy_objective: str = "target_ic",
    min_target_ic: float = 0.0,
    min_utility_ic: float = 0.0,
    max_bad_mae_ic: float = 0.0,
    max_wide_ic: float = 0.0,
    max_timeout_ic: float = 0.0,
    utility_weight: float = 1.0,
    bad_mae_weight: float = 1.0,
    wide_weight: float = 0.5,
    timeout_weight: float = 0.5,
) -> tuple[pd.Series, dict[str, Any]]:
    proxy_objective = str(proxy_objective)
    if proxy_objective not in PROXY_OBJECTIVES:
        raise ValueError(f"Unknown proxy objective: {proxy_objective}")
    rows: list[dict[str, Any]] = []
    y = _safe_numeric(target_train)
    target_rank = _rank_array(y)
    utility = pd.Series(np.nan, index=train.index)
    bad_mae = pd.Series(np.nan, index=train.index)
    wide = pd.Series(np.nan, index=train.index)
    timeout = pd.Series(np.nan, index=train.index)
    if metrics_train is not None:
        utility = _safe_numeric(metrics_train["u_policy_net"]).reindex(train.index)
        bad_mae = (_safe_numeric(metrics_train["mae_norm"]).reindex(train.index) >= 1.0).astype(float)
        wide = (_safe_numeric(metrics_train["barrier"]).reindex(train.index) > 0.025).astype(float)
        timeout = _safe_numeric(metrics_train["is_timeout"].astype(float)).reindex(train.index)
    utility_rank = _rank_array(utility)
    bad_mae_rank = _rank_array(bad_mae)
    wide_rank = _rank_array(wide)
    timeout_rank = _rank_array(timeout)
    for feature in features:
        feature_rank = _rank_array(train[feature])
        target_ic = _rank_corr(feature_rank, target_rank)
        if not math.isfinite(target_ic):
            continue
        if proxy_objective == "target_ic":
            rows.append(
                {
                    "feature": feature,
                    "ic": float(target_ic),
                    "abs_ic": abs(float(target_ic)),
                    "ranking_score": abs(float(target_ic)),
                    "target_ic": float(target_ic),
                    "utility_ic": float("nan"),
                    "bad_mae_ic": float("nan"),
                    "wide_ic": float("nan"),
                    "timeout_ic": float("nan"),
                    "rank_high_good": bool(float(target_ic) >= 0.0),
                }
            )
            continue

        utility_ic = _rank_corr(feature_rank, utility_rank)
        bad_mae_ic = _rank_corr(feature_rank, bad_mae_rank)
        wide_ic = _rank_corr(feature_rank, wide_rank)
        timeout_ic = _rank_corr(feature_rank, timeout_rank)
        if not all(math.isfinite(v) for v in (utility_ic, bad_mae_ic, wide_ic, timeout_ic)):
            continue
        if target_ic < float(min_target_ic):
            continue
        if proxy_objective == "economic_ic" and (
            utility_ic < float(min_utility_ic)
            or bad_mae_ic > float(max_bad_mae_ic)
            or wide_ic > float(max_wide_ic)
            or timeout_ic > float(max_timeout_ic)
        ):
            continue
        ranking_score = (
            float(target_ic)
            + float(utility_weight) * float(utility_ic)
            - float(bad_mae_weight) * float(bad_mae_ic)
            - float(wide_weight) * float(wide_ic)
            - float(timeout_weight) * float(timeout_ic)
        )
        if not math.isfinite(ranking_score):
            continue
        rows.append(
            {
                "feature": feature,
                "ic": float(target_ic),
                "abs_ic": abs(float(target_ic)),
                "ranking_score": float(ranking_score),
                "target_ic": float(target_ic),
                "utility_ic": float(utility_ic),
                "bad_mae_ic": float(bad_mae_ic),
                "wide_ic": float(wide_ic),
                "timeout_ic": float(timeout_ic),
                "rank_high_good": True,
            }
        )
    if not rows:
        return pd.Series(np.nan, index=valid.index), {
            "proxy_objective": proxy_objective,
            "proxy_features": [],
            "proxy_candidate_count": 0,
            "proxy_top_abs_ic": float("nan"),
            "proxy_mean_top_abs_ic": float("nan"),
            "proxy_top_ranking_score": float("nan"),
            "proxy_mean_ranking_score": float("nan"),
            "proxy_mean_train_target_ic": float("nan"),
            "proxy_mean_train_utility_ic": float("nan"),
            "proxy_mean_train_bad_mae_ic": float("nan"),
            "proxy_mean_train_wide_ic": float("nan"),
            "proxy_mean_train_timeout_ic": float("nan"),
        }
    chosen = pd.DataFrame(rows).sort_values("ranking_score", ascending=False).head(int(top_k))
    parts: list[pd.Series] = []
    for _, row in chosen.iterrows():
        parts.append(_rank_pct(valid[str(row["feature"])], high_good=bool(row["rank_high_good"])).fillna(0.5))
    score = pd.concat(parts, axis=1).mean(axis=1) if parts else pd.Series(np.nan, index=valid.index)

    def chosen_mean(column: str) -> float:
        values = _safe_numeric(chosen[column]) if column in chosen.columns else pd.Series(dtype=float)
        return _safe_mean(values)

    return score.reindex(valid.index), {
        "proxy_objective": proxy_objective,
        "proxy_features": chosen["feature"].astype(str).tolist(),
        "proxy_candidate_count": int(len(rows)),
        "proxy_top_abs_ic": float(chosen["abs_ic"].iloc[0]) if len(chosen) else float("nan"),
        "proxy_mean_top_abs_ic": float(chosen["abs_ic"].mean()) if len(chosen) else float("nan"),
        "proxy_top_ranking_score": float(chosen["ranking_score"].iloc[0]) if len(chosen) else float("nan"),
        "proxy_mean_ranking_score": chosen_mean("ranking_score"),
        "proxy_mean_train_target_ic": chosen_mean("target_ic"),
        "proxy_mean_train_utility_ic": chosen_mean("utility_ic"),
        "proxy_mean_train_bad_mae_ic": chosen_mean("bad_mae_ic"),
        "proxy_mean_train_wide_ic": chosen_mean("wide_ic"),
        "proxy_mean_train_timeout_ic": chosen_mean("timeout_ic"),
    }


def _selection_row(
    *,
    frame: pd.DataFrame,
    metrics: pd.DataFrame,
    target: pd.DataFrame,
    score: pd.Series,
    arm: str,
    selector: str,
    period: str,
    top_frac: float,
    diag: dict[str, Any],
) -> dict[str, Any]:
    row = _selection_metrics(
        frame=frame,
        metrics=metrics,
        target=target,
        score=score,
        arm=arm,
        selector=selector,
        period=period,
        top_frac=top_frac,
    )
    idx = _rank_top_indices(score, float(top_frac))
    selected = metrics.iloc[idx] if len(idx) else metrics.iloc[:0]
    if len(selected):
        mfe_mae = (selected["mfe_norm"] / selected["mae_norm"].clip(lower=0.25)).replace(
            [np.inf, -np.inf],
            np.nan,
        ).clip(upper=10.0)
        row["mean_mfe_mae_ratio"] = _safe_mean(mfe_mae)
        row["clean_row_rate"] = _safe_mean(
            (selected["u_policy_net"] > 0.0)
            & (selected["mae_norm"] <= 1.0)
            & (selected["barrier"] <= 0.025)
            & (selected["is_timeout"].astype(float) <= 0.0)
        )
        row["bounded_row_rate"] = _safe_mean(
            (selected["u_policy_net"] > 0.0)
            & (selected["mae_norm"] <= 1.0)
            & (selected["barrier"] <= 0.035)
            & (mfe_mae >= 1.25)
            & (selected["is_timeout"].astype(float) <= 0.0)
        )
    else:
        row["mean_mfe_mae_ratio"] = float("nan")
        row["clean_row_rate"] = float("nan")
        row["bounded_row_rate"] = float("nan")
    row.update(diag)
    return row


def _selected_ledger_rows(
    *,
    frame: pd.DataFrame,
    metrics: pd.DataFrame,
    target: pd.DataFrame,
    score: pd.Series,
    arm: str,
    selector: str,
    period: str,
    top_frac: float,
) -> list[dict[str, Any]]:
    idx = _rank_top_indices(score, float(top_frac))
    if len(idx) == 0:
        return []
    frame_reset = frame.reset_index(drop=True)
    metrics_reset = metrics.reset_index(drop=True)
    target_reset = target.reset_index(drop=True)
    score_reset = _safe_numeric(score.reset_index(drop=True))
    selected_rows = int(len(idx))
    out: list[dict[str, Any]] = []
    for rank, pos in enumerate(idx, start=1):
        ts = pd.to_datetime(frame_reset.loc[pos, "__ts__"], errors="coerce")
        metric_row = metrics_reset.loc[pos]
        mae_norm = float(metric_row["mae_norm"]) if pd.notna(metric_row["mae_norm"]) else float("nan")
        mfe_norm = float(metric_row["mfe_norm"]) if pd.notna(metric_row["mfe_norm"]) else float("nan")
        barrier = float(metric_row["barrier"]) if pd.notna(metric_row["barrier"]) else float("nan")
        mfe_mae = (
            float(min(mfe_norm / max(mae_norm, 0.25), 10.0))
            if math.isfinite(mfe_norm) and math.isfinite(mae_norm)
            else float("nan")
        )
        out.append(
            {
                "timestamp": ts.isoformat() if pd.notna(ts) else "",
                "symbol": str(frame_reset.loc[pos, "__symbol__"]),
                "period": str(period),
                "week": ts.to_period("W-SUN").strftime("%Y-%m-%d/%Y-%m-%d") if pd.notna(ts) else "",
                "arm": str(arm),
                "label_arm": str(arm),
                "weight_arm": "W0_proxy",
                "selection_mode": str(selector),
                "mae_penalty": 0.0,
                "wide_penalty": 0.0,
                "timeout_penalty": 0.0,
                "mae_keep_frac": 1.0,
                "wide_keep_frac": 1.0,
                "timeout_keep_frac": 1.0,
                "top_frac": float(top_frac),
                "selected_rank": int(rank),
                "selected_rows": selected_rows,
                "score": float(score_reset.iloc[pos]) if pd.notna(score_reset.iloc[pos]) else float("nan"),
                "target_soft": float(target_reset.loc[pos, "target_soft"]) if pd.notna(target_reset.loc[pos, "target_soft"]) else float("nan"),
                "target_hard": float(target_reset.loc[pos, "target_hard"]) if pd.notna(target_reset.loc[pos, "target_hard"]) else float("nan"),
                "u_policy_net": float(metric_row["u_policy_net"]) if pd.notna(metric_row["u_policy_net"]) else float("nan"),
                "ret_net": float(metric_row["ret_net"]) if pd.notna(metric_row["ret_net"]) else float("nan"),
                "mfe_norm": mfe_norm,
                "mae_norm": mae_norm,
                "mfe_mae_ratio": mfe_mae,
                "barrier": barrier,
                "bars_to_mfe": float(metric_row["bars_to_mfe"]) if pd.notna(metric_row["bars_to_mfe"]) else float("nan"),
                "bad_mae_1r": bool(mae_norm >= 1.0) if math.isfinite(mae_norm) else False,
                "wide_barrier_25bps": bool(barrier > 0.025) if math.isfinite(barrier) else False,
                "wide_barrier_35bps": bool(barrier > 0.035) if math.isfinite(barrier) else False,
                "is_timeout": bool(metric_row["is_timeout"]),
            }
        )
    return out


def _monthly_weekly_rows(
    *,
    valid_frame: pd.DataFrame,
    valid_metrics: pd.DataFrame,
    valid_target: pd.DataFrame,
    score: pd.Series,
    arm: str,
    selector: str,
    month: str,
    top_fracs: list[float],
    diag: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    monthly_rows: list[dict[str, Any]] = []
    weekly_rows: list[dict[str, Any]] = []
    frame_reset = valid_frame.reset_index(drop=True)
    metrics_reset = valid_metrics.reset_index(drop=True)
    target_reset = valid_target.reset_index(drop=True)
    score_reset = score.reset_index(drop=True)
    for frac in top_fracs:
        monthly_rows.append(
            _selection_row(
                frame=frame_reset,
                metrics=metrics_reset,
                target=target_reset,
                score=score_reset,
                arm=arm,
                selector=selector,
                period=str(month),
                top_frac=float(frac),
                diag=diag,
            )
        )
        weeks = frame_reset["__ts__"].dt.to_period("W-SUN").astype(str)
        for week, ids in pd.Series(np.arange(len(frame_reset)), index=frame_reset.index).groupby(weeks, dropna=False):
            pos = ids.to_numpy(dtype=np.int64)
            if len(pos) < 20:
                continue
            week_row = _selection_row(
                frame=frame_reset.iloc[pos].reset_index(drop=True),
                metrics=metrics_reset.iloc[pos].reset_index(drop=True),
                target=target_reset.iloc[pos].reset_index(drop=True),
                score=score_reset.iloc[pos].reset_index(drop=True),
                arm=arm,
                selector=selector,
                period=str(month),
                top_frac=float(frac),
                diag=diag,
            )
            week_row["week"] = str(week)
            week_row["week_selected_rows"] = int(week_row["selected_rows"])
            weekly_rows.append(week_row)
    return monthly_rows, weekly_rows


def _summarize_month(prefix: str, frame: pd.DataFrame) -> dict[str, Any]:
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
            f"{prefix}_mean_mfe_mae_ratio": float("nan"),
            f"{prefix}_clean_row_rate": float("nan"),
            f"{prefix}_bounded_row_rate": float("nan"),
        }
    mean_u = _safe_numeric(frame["mean_u"])
    return {
        f"{prefix}_months": int(frame["period"].astype(str).nunique()),
        f"{prefix}_positive_months": int(mean_u.gt(0.0).sum()),
        f"{prefix}_mean_month_u": _safe_mean(mean_u),
        f"{prefix}_worst_month_u": _safe_min(mean_u),
        f"{prefix}_selected_rows": int(_safe_numeric(frame["selected_rows"]).sum()),
        f"{prefix}_bad_mae_1r_rate": _weighted_mean(frame, "bad_mae_1r_rate", "selected_rows"),
        f"{prefix}_p90_mae_norm": _weighted_mean(frame, "p90_mae_norm", "selected_rows"),
        f"{prefix}_wide_25bps_rate": _weighted_mean(frame, "wide_barrier_25bps_rate", "selected_rows"),
        f"{prefix}_timeout_rate": _weighted_mean(frame, "timeout_rate", "selected_rows"),
        f"{prefix}_mean_mfe_mae_ratio": _weighted_mean(frame, "mean_mfe_mae_ratio", "selected_rows"),
        f"{prefix}_clean_row_rate": _weighted_mean(frame, "clean_row_rate", "selected_rows"),
        f"{prefix}_bounded_row_rate": _weighted_mean(frame, "bounded_row_rate", "selected_rows"),
        f"{prefix}_max_top_symbol_share": _safe_max(frame["top_symbol_share"]),
    }


def _summarize_week(prefix: str, frame: pd.DataFrame, *, min_week_rows: int) -> dict[str, Any]:
    if frame.empty:
        return {
            f"{prefix}_weeks": 0,
            f"{prefix}_material_weeks": 0,
            f"{prefix}_material_positive_week_rate": float("nan"),
            f"{prefix}_q25_week_u": float("nan"),
            f"{prefix}_worst_week_u": float("nan"),
        }
    mean_u = _safe_numeric(frame["mean_u"])
    selected_rows = _safe_numeric(frame["week_selected_rows"]).fillna(0.0)
    material = selected_rows >= int(min_week_rows)
    positive = mean_u > 0.0
    return {
        f"{prefix}_weeks": int(len(frame)),
        f"{prefix}_material_weeks": int(material.sum()),
        f"{prefix}_material_positive_week_rate": float((positive & material).sum() / material.sum())
        if int(material.sum())
        else float("nan"),
        f"{prefix}_q25_week_u": _safe_quantile(mean_u[material], 0.25) if int(material.sum()) else float("nan"),
        f"{prefix}_worst_week_u": _safe_min(mean_u[material]) if int(material.sum()) else float("nan"),
    }


def _fit_holdout_summary(
    *,
    monthly: pd.DataFrame,
    weekly: pd.DataFrame,
    fit_months: list[str],
    holdout_month: str,
    min_week_rows: int,
    min_fit_material_weeks: int,
    min_holdout_material_weeks: int,
    min_fit_positive_week_rate: float,
    min_holdout_positive_week_rate: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_cols = ["arm", "selector", "top_frac"]
    for key, group in monthly.groupby(group_cols, observed=True, dropna=False):
        arm, selector, frac = key
        week_group = weekly[
            weekly["arm"].astype(str).eq(str(arm))
            & weekly["selector"].astype(str).eq(str(selector))
            & _safe_numeric(weekly["top_frac"]).eq(float(frac))
        ].copy()
        fit_month = group[group["period"].astype(str).isin(fit_months)].copy()
        holdout_monthly = group[group["period"].astype(str).eq(str(holdout_month))].copy()
        fit_week = week_group[week_group["period"].astype(str).isin(fit_months)].copy()
        holdout_week = week_group[week_group["period"].astype(str).eq(str(holdout_month))].copy()
        if fit_month.empty or holdout_monthly.empty:
            continue

        row: dict[str, Any] = {
            "arm": str(arm),
            "selector": str(selector),
            "top_frac": float(frac),
        }
        row.update(_summarize_month("fit", fit_month))
        row.update(_summarize_month("holdout", holdout_monthly))
        row.update(_summarize_week("fit", fit_week, min_week_rows=min_week_rows))
        row.update(_summarize_week("holdout", holdout_week, min_week_rows=min_week_rows))

        fit_sign = (
            row["fit_months"] == len(fit_months)
            and row["fit_positive_months"] == len(fit_months)
            and row["fit_worst_month_u"] > 0.0
            and row["fit_material_weeks"] >= min_fit_material_weeks
            and row["fit_material_positive_week_rate"] >= min_fit_positive_week_rate
        )
        holdout_sign = (
            row["holdout_mean_month_u"] > 0.0
            and row["holdout_material_weeks"] >= min_holdout_material_weeks
            and row["holdout_material_positive_week_rate"] >= min_holdout_positive_week_rate
        )
        fit_clean = (
            fit_sign
            and row["fit_bad_mae_1r_rate"] <= 0.50
            and row["fit_p90_mae_norm"] <= 3.0
            and row["fit_timeout_rate"] <= 0.20
            and row["fit_wide_25bps_rate"] <= 0.30
        )
        holdout_clean = (
            holdout_sign
            and row["holdout_bad_mae_1r_rate"] <= 0.50
            and row["holdout_p90_mae_norm"] <= 3.0
            and row["holdout_timeout_rate"] <= 0.20
            and row["holdout_wide_25bps_rate"] <= 0.30
        )
        fit_bounded = (
            fit_sign
            and row["fit_bad_mae_1r_rate"] <= 0.80
            and row["fit_p90_mae_norm"] <= 4.0
            and row["fit_timeout_rate"] <= 0.20
            and row["fit_wide_25bps_rate"] <= 0.35
            and row["fit_mean_mfe_mae_ratio"] >= 1.25
        )
        holdout_bounded = (
            holdout_sign
            and row["holdout_bad_mae_1r_rate"] <= 0.80
            and row["holdout_p90_mae_norm"] <= 4.0
            and row["holdout_timeout_rate"] <= 0.20
            and row["holdout_wide_25bps_rate"] <= 0.35
            and row["holdout_mean_mfe_mae_ratio"] >= 1.25
        )
        row["fit_sign_pass"] = bool(fit_sign)
        row["holdout_sign_pass"] = bool(holdout_sign)
        row["fit_clean_pass"] = bool(fit_clean)
        row["holdout_clean_standalone_pass"] = bool(holdout_clean)
        row["holdout_clean_pass"] = bool(fit_clean and holdout_clean)
        row["fit_bounded_pass"] = bool(fit_bounded)
        row["holdout_bounded_standalone_pass"] = bool(holdout_bounded)
        row["holdout_bounded_pass"] = bool(fit_bounded and holdout_bounded)
        row["positive_dirty_holdout"] = bool(holdout_sign and not holdout_bounded)
        row["path_risk_score"] = float(
            (row["holdout_mean_month_u"] if pd.notna(row["holdout_mean_month_u"]) else 0.0)
            + 0.50 * (row["holdout_q25_week_u"] if pd.notna(row["holdout_q25_week_u"]) else 0.0)
            + 0.002 * (row["holdout_mean_mfe_mae_ratio"] if pd.notna(row["holdout_mean_mfe_mae_ratio"]) else 0.0)
            - 0.020 * (row["holdout_bad_mae_1r_rate"] if pd.notna(row["holdout_bad_mae_1r_rate"]) else 0.0)
            - 0.003 * (row["holdout_p90_mae_norm"] if pd.notna(row["holdout_p90_mae_norm"]) else 0.0)
        )
        rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(
        ["holdout_clean_pass", "holdout_bounded_pass", "positive_dirty_holdout", "path_risk_score"],
        ascending=[False, False, False, False],
    )


def _label_summary(
    *,
    frame: pd.DataFrame,
    metrics: pd.DataFrame,
    targets: dict[str, pd.DataFrame],
    descriptions: dict[str, str],
    features: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for arm, target in targets.items():
        soft = target["target_soft"]
        feature_ics = []
        for feature in features:
            ic = _spearman(frame[feature], soft)
            if math.isfinite(ic):
                feature_ics.append(abs(ic))
        feature_ics = sorted(feature_ics, reverse=True)
        rows.append(
            {
                "arm": arm,
                "description": descriptions.get(arm, ""),
                "soft_mean": _safe_mean(soft),
                "soft_std": float(_safe_numeric(soft).std(ddof=0)),
                "soft_p90": _safe_quantile(soft, 0.90),
                "soft_high_sat_rate": _safe_mean(soft >= 0.95),
                "hard_rate": _safe_mean(target["target_hard"]),
                "ic_soft_vs_u": _spearman(soft, metrics["u_policy_net"]),
                "ic_soft_vs_mae": _spearman(soft, metrics["mae_norm"]),
                "ic_soft_vs_mfe": _spearman(soft, metrics["mfe_norm"]),
                "feature_top_abs_ic": feature_ics[0] if feature_ics else float("nan"),
                "feature_mean_top_abs_ic": float(np.mean(feature_ics[:12])) if feature_ics else float("nan"),
                "feature_n_abs_ic_ge_002": int(np.sum(np.asarray(feature_ics) >= 0.02)) if feature_ics else 0,
                "feature_n_abs_ic_ge_005": int(np.sum(np.asarray(feature_ics) >= 0.05)) if feature_ics else 0,
            }
        )
    return pd.DataFrame(rows)


def _write_markdown(
    *,
    output_dir: Path,
    label_summary: pd.DataFrame,
    fit_holdout: pd.DataFrame,
    proxy_ic: pd.DataFrame,
    manifest: dict[str, Any],
) -> Path:
    path = output_dir / "soft_label_economic_proxy_ablation.md"

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

    counts = (
        fit_holdout.groupby("selector", observed=True)
        .agg(
            rows=("arm", "size"),
            fit_clean=("fit_clean_pass", "sum"),
            holdout_clean=("holdout_clean_pass", "sum"),
            fit_bounded=("fit_bounded_pass", "sum"),
            holdout_bounded=("holdout_bounded_pass", "sum"),
            positive_dirty=("positive_dirty_holdout", "sum"),
        )
        .reset_index()
        if not fit_holdout.empty
        else pd.DataFrame()
    )
    proxy_rows = fit_holdout[fit_holdout["selector"].eq("feature_ic_proxy")].copy()
    oracle_rows = fit_holdout[fit_holdout["selector"].eq("oracle_label_sort")].copy()
    proxy_best = proxy_rows.sort_values(
        ["holdout_clean_pass", "holdout_bounded_pass", "positive_dirty_holdout", "path_risk_score"],
        ascending=[False, False, False, False],
    )
    oracle_best = oracle_rows.sort_values(
        ["holdout_clean_pass", "holdout_bounded_pass", "positive_dirty_holdout", "path_risk_score"],
        ascending=[False, False, False, False],
    )
    lines = [
        "# Soft Label Economic Proxy Ablation",
        "",
        "Scope: fixed candidate universe, soft-label tests only. No LightGBM, Optuna, policy geometry, or base/meta training is run.",
        "",
        f"Fit months: `{','.join(manifest['fit_months'])}`. Holdout month: `{manifest['holdout_month']}`.",
        "",
        "Economic gates require positive Apr-May fit, weekly fit stability, then June holdout sign. Clean adds bad-MAE <= 50% and p90 MAE <= 3R. Bounded adds bad-MAE <= 80%, p90 MAE <= 4R, MFE/MAE >= 1.25, timeout <= 20%, and wide-barrier control.",
        "",
        "## Gate Counts",
        "",
        table(counts, ["selector", "rows", "fit_clean", "holdout_clean", "fit_bounded", "holdout_bounded", "positive_dirty"]),
        "",
        "## Best Feature-Proxy Rows",
        "",
        table(
            proxy_best,
            [
                "arm",
                "top_frac",
                "fit_sign_pass",
                "fit_clean_pass",
                "fit_bounded_pass",
                "holdout_sign_pass",
                "holdout_clean_pass",
                "holdout_bounded_pass",
                "positive_dirty_holdout",
                "fit_mean_month_u",
                "holdout_mean_month_u",
                "holdout_bad_mae_1r_rate",
                "holdout_p90_mae_norm",
                "holdout_mean_mfe_mae_ratio",
                "holdout_material_positive_week_rate",
                "path_risk_score",
            ],
            limit=30,
        ),
        "",
        "## Best Oracle Label-Sort Rows",
        "",
        table(
            oracle_best,
            [
                "arm",
                "top_frac",
                "fit_clean_pass",
                "fit_bounded_pass",
                "holdout_clean_pass",
                "holdout_bounded_pass",
                "fit_mean_month_u",
                "holdout_mean_month_u",
                "holdout_bad_mae_1r_rate",
                "holdout_p90_mae_norm",
                "holdout_mean_mfe_mae_ratio",
                "path_risk_score",
            ],
            limit=30,
        ),
        "",
        "## Label Shape And Static Feature Association",
        "",
        table(
            label_summary.sort_values(["feature_mean_top_abs_ic", "ic_soft_vs_u"], ascending=[False, False]),
            [
                "arm",
                "soft_mean",
                "soft_std",
                "hard_rate",
                "ic_soft_vs_u",
                "ic_soft_vs_mae",
                "ic_soft_vs_mfe",
                "feature_top_abs_ic",
                "feature_mean_top_abs_ic",
                "feature_n_abs_ic_ge_002",
            ],
            limit=30,
        ),
        "",
        "## Monthly Proxy ICs",
        "",
        table(
            proxy_ic.sort_values(["period", "oos_ic_u"], ascending=[True, False]),
            [
                "period",
                "arm",
                "proxy_objective",
                "oos_ic_target",
                "oos_ic_u",
                "oos_ic_bad_mae",
                "oos_ic_wide",
                "oos_ic_timeout",
                "proxy_candidate_count",
                "proxy_top_ranking_score",
                "proxy_mean_train_target_ic",
                "proxy_mean_train_utility_ic",
                "proxy_mean_train_bad_mae_ic",
                "proxy_mean_train_wide_ic",
                "proxy_mean_train_timeout_ic",
                "proxy_top_abs_ic",
                "proxy_mean_top_abs_ic",
                "proxy_features",
            ],
            limit=80,
        ),
        "",
        "## Outputs",
        "",
        f"- Label summary: `{manifest['outputs']['label_summary']}`",
        f"- Monthly selection: `{manifest['outputs']['monthly']}`",
        f"- Weekly selection: `{manifest['outputs']['weekly']}`",
        f"- Monthly proxy IC: `{manifest['outputs']['proxy_ic']}`",
        f"- Fit/holdout summary: `{manifest['outputs']['fit_holdout']}`",
        f"- Manifest: `{manifest['outputs']['manifest']}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_ablation(
    *,
    labels_path: Path,
    output_dir: Path,
    feature_dir: Path,
    feature_list_csv: Path,
    max_feature_store_features: int | None,
    proxy_top_k: int,
    proxy_objective: str,
    proxy_min_target_ic: float,
    proxy_min_utility_ic: float,
    proxy_max_bad_mae_ic: float,
    proxy_max_wide_ic: float,
    proxy_max_timeout_ic: float,
    proxy_utility_weight: float,
    proxy_bad_mae_weight: float,
    proxy_wide_weight: float,
    proxy_timeout_weight: float,
    top_fracs: list[float],
    fit_months: list[str],
    holdout_month: str,
    min_week_rows: int,
    arms: list[str] | None,
    include_causal_outcome_priors: bool,
    include_causal_state_path_priors: bool,
    include_event_confirmation_features: bool,
    include_slow_trade_diagnostic_features: bool,
    prior_windows_days: list[float],
    prior_embargo_hours: float,
    state_path_prior_features: list[str],
    event_feature_store_features: list[str],
    write_selected_ledger: bool,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = _load_labels(labels_path)
    selected_features = _read_feature_list(feature_list_csv, max_features=max_feature_store_features)
    slow_trade_report: dict[str, Any] = {"enabled": False}
    if include_slow_trade_diagnostic_features:
        from scripts.report_timeout_feature_stability import (  # noqa: WPS433
            DEFAULT_SLOW_TRADE_SOURCE_FEATURES,
            _add_slow_trade_diagnostic_features,
        )

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
    prior_report: dict[str, Any] = {"enabled": False}
    if include_causal_outcome_priors:
        prior_features, prior_report = _causal_outcome_prior_features(
            frame,
            metrics,
            windows_days=prior_windows_days,
            embargo_hours=prior_embargo_hours,
        )
        frame = pd.concat([frame, prior_features.astype(np.float32, copy=False)], axis=1).copy()
    state_path_prior_report: dict[str, Any] = {"enabled": False}
    if include_causal_state_path_priors:
        state_path_prior_features_frame, state_path_prior_report = _causal_state_path_prior_features(
            frame,
            metrics,
            state_features=state_path_prior_features,
            windows_days=prior_windows_days,
            embargo_hours=prior_embargo_hours,
        )
        frame = pd.concat(
            [frame, state_path_prior_features_frame.astype(np.float32, copy=False)],
            axis=1,
        ).copy()
    event_confirmation_report: dict[str, Any] = {"enabled": False}
    if include_event_confirmation_features:
        event_features, event_confirmation_report = _event_confirmation_features(
            frame,
            event_features=event_feature_store_features,
        )
        frame = pd.concat([frame, event_features.astype(np.float32, copy=False)], axis=1).copy()
    if include_slow_trade_diagnostic_features:
        slow_features, slow_trade_report = _add_slow_trade_diagnostic_features(frame)
        if not slow_features.empty:
            frame = pd.concat([frame, slow_features.astype(np.float32, copy=False)], axis=1).copy()
    targets, descriptions = _all_targets(frame, metrics)
    if arms:
        missing = sorted(set(arms) - set(targets))
        if missing:
            raise ValueError(f"Unknown arm(s): {missing}")
        targets = {arm: targets[arm] for arm in arms}
        descriptions = {arm: descriptions.get(arm, "") for arm in arms}
    features = _feature_columns(frame)
    label_summary = _label_summary(
        frame=frame,
        metrics=metrics,
        targets=targets,
        descriptions=descriptions,
        features=features,
    )

    month_series = frame["__ts__"].dt.to_period("M").astype(str)
    months = sorted(month_series.dropna().unique())
    monthly_rows: list[dict[str, Any]] = []
    weekly_rows: list[dict[str, Any]] = []
    proxy_ic_rows: list[dict[str, Any]] = []
    selected_ledger_rows: list[dict[str, Any]] = []

    for month in months[1:]:
        train_mask = month_series < str(month)
        valid_mask = month_series == str(month)
        if int(train_mask.sum()) < 500 or int(valid_mask.sum()) < 100:
            continue
        train = frame.loc[train_mask].copy()
        valid = frame.loc[valid_mask].copy()
        valid_metrics = metrics.loc[valid_mask].copy()
        for arm, target in targets.items():
            train_target = target.loc[train_mask, "target_soft"]
            valid_target = target.loc[valid_mask].copy()
            train_metrics = metrics.loc[train_mask].copy()
            proxy_score, proxy_diag = _proxy_score(
                train=train,
                valid=valid,
                features=features,
                target_train=train_target,
                metrics_train=train_metrics,
                top_k=proxy_top_k,
                proxy_objective=proxy_objective,
                min_target_ic=proxy_min_target_ic,
                min_utility_ic=proxy_min_utility_ic,
                max_bad_mae_ic=proxy_max_bad_mae_ic,
                max_wide_ic=proxy_max_wide_ic,
                max_timeout_ic=proxy_max_timeout_ic,
                utility_weight=proxy_utility_weight,
                bad_mae_weight=proxy_bad_mae_weight,
                wide_weight=proxy_wide_weight,
                timeout_weight=proxy_timeout_weight,
            )
            proxy_ic_rows.append(
                {
                    "period": str(month),
                    "arm": arm,
                    "description": descriptions.get(arm, ""),
                    "oos_ic_target": _spearman(proxy_score, valid_target["target_soft"]),
                    "oos_ic_u": _spearman(proxy_score, valid_metrics["u_policy_net"]),
                    "oos_ic_bad_mae": _spearman(proxy_score, (valid_metrics["mae_norm"] >= 1.0).astype(float)),
                    "oos_ic_wide": _spearman(proxy_score, (valid_metrics["barrier"] > 0.025).astype(float)),
                    "oos_ic_timeout": _spearman(proxy_score, valid_metrics["is_timeout"].astype(float)),
                    "proxy_objective": proxy_diag.get("proxy_objective"),
                    "proxy_candidate_count": proxy_diag.get("proxy_candidate_count"),
                    "proxy_top_abs_ic": proxy_diag.get("proxy_top_abs_ic"),
                    "proxy_mean_top_abs_ic": proxy_diag.get("proxy_mean_top_abs_ic"),
                    "proxy_top_ranking_score": proxy_diag.get("proxy_top_ranking_score"),
                    "proxy_mean_ranking_score": proxy_diag.get("proxy_mean_ranking_score"),
                    "proxy_mean_train_target_ic": proxy_diag.get("proxy_mean_train_target_ic"),
                    "proxy_mean_train_utility_ic": proxy_diag.get("proxy_mean_train_utility_ic"),
                    "proxy_mean_train_bad_mae_ic": proxy_diag.get("proxy_mean_train_bad_mae_ic"),
                    "proxy_mean_train_wide_ic": proxy_diag.get("proxy_mean_train_wide_ic"),
                    "proxy_mean_train_timeout_ic": proxy_diag.get("proxy_mean_train_timeout_ic"),
                    "proxy_features": ",".join(proxy_diag.get("proxy_features", [])),
                }
            )
            for selector, score, diag in (
                (
                    "oracle_label_sort",
                    valid_target["target_soft"],
                    {
                        "oos_ic_target": 1.0,
                        "oos_ic_u": _spearman(valid_target["target_soft"], valid_metrics["u_policy_net"]),
                        "oos_ic_bad_mae": _spearman(valid_target["target_soft"], (valid_metrics["mae_norm"] >= 1.0).astype(float)),
                        "oos_ic_wide": _spearman(valid_target["target_soft"], (valid_metrics["barrier"] > 0.025).astype(float)),
                        "oos_ic_timeout": _spearman(valid_target["target_soft"], valid_metrics["is_timeout"].astype(float)),
                        "proxy_objective": "oracle_label_sort",
                        "proxy_candidate_count": float("nan"),
                        "proxy_top_abs_ic": float("nan"),
                        "proxy_mean_top_abs_ic": float("nan"),
                        "proxy_top_ranking_score": float("nan"),
                        "proxy_mean_ranking_score": float("nan"),
                        "proxy_mean_train_target_ic": float("nan"),
                        "proxy_mean_train_utility_ic": float("nan"),
                        "proxy_mean_train_bad_mae_ic": float("nan"),
                        "proxy_mean_train_wide_ic": float("nan"),
                        "proxy_mean_train_timeout_ic": float("nan"),
                        "proxy_features": "",
                    },
                ),
                (
                    "feature_ic_proxy",
                    proxy_score,
                    {
                        "oos_ic_target": _spearman(proxy_score, valid_target["target_soft"]),
                        "oos_ic_u": _spearman(proxy_score, valid_metrics["u_policy_net"]),
                        "oos_ic_bad_mae": _spearman(proxy_score, (valid_metrics["mae_norm"] >= 1.0).astype(float)),
                        "oos_ic_wide": _spearman(proxy_score, (valid_metrics["barrier"] > 0.025).astype(float)),
                        "oos_ic_timeout": _spearman(proxy_score, valid_metrics["is_timeout"].astype(float)),
                        "proxy_objective": proxy_diag.get("proxy_objective"),
                        "proxy_candidate_count": proxy_diag.get("proxy_candidate_count"),
                        "proxy_top_abs_ic": proxy_diag.get("proxy_top_abs_ic"),
                        "proxy_mean_top_abs_ic": proxy_diag.get("proxy_mean_top_abs_ic"),
                        "proxy_top_ranking_score": proxy_diag.get("proxy_top_ranking_score"),
                        "proxy_mean_ranking_score": proxy_diag.get("proxy_mean_ranking_score"),
                        "proxy_mean_train_target_ic": proxy_diag.get("proxy_mean_train_target_ic"),
                        "proxy_mean_train_utility_ic": proxy_diag.get("proxy_mean_train_utility_ic"),
                        "proxy_mean_train_bad_mae_ic": proxy_diag.get("proxy_mean_train_bad_mae_ic"),
                        "proxy_mean_train_wide_ic": proxy_diag.get("proxy_mean_train_wide_ic"),
                        "proxy_mean_train_timeout_ic": proxy_diag.get("proxy_mean_train_timeout_ic"),
                        "proxy_features": ",".join(proxy_diag.get("proxy_features", [])),
                    },
                ),
            ):
                m_rows, w_rows = _monthly_weekly_rows(
                    valid_frame=valid,
                    valid_metrics=valid_metrics,
                    valid_target=valid_target,
                    score=score,
                    arm=arm,
                    selector=selector,
                    month=str(month),
                    top_fracs=top_fracs,
                    diag=diag,
                )
                monthly_rows.extend(m_rows)
                weekly_rows.extend(w_rows)
    monthly = pd.DataFrame(monthly_rows)
    weekly = pd.DataFrame(weekly_rows)
    proxy_ic = pd.DataFrame(proxy_ic_rows)
    if write_selected_ledger:
        # Build the selected ledger from monthly rows in one pass after scores are computed.
        selected_ledger_rows = []
        for month in months[1:]:
            train_mask = month_series < str(month)
            valid_mask = month_series == str(month)
            if int(train_mask.sum()) < 500 or int(valid_mask.sum()) < 100:
                continue
            train = frame.loc[train_mask].copy()
            valid = frame.loc[valid_mask].copy()
            valid_metrics = metrics.loc[valid_mask].copy().reset_index(drop=True)
            for arm, target in targets.items():
                proxy_score, _ = _proxy_score(
                    train=train,
                    valid=valid,
                    features=features,
                    target_train=target.loc[train_mask, "target_soft"],
                    metrics_train=metrics.loc[train_mask].copy(),
                    top_k=proxy_top_k,
                    proxy_objective=proxy_objective,
                    min_target_ic=proxy_min_target_ic,
                    min_utility_ic=proxy_min_utility_ic,
                    max_bad_mae_ic=proxy_max_bad_mae_ic,
                    max_wide_ic=proxy_max_wide_ic,
                    max_timeout_ic=proxy_max_timeout_ic,
                    utility_weight=proxy_utility_weight,
                    bad_mae_weight=proxy_bad_mae_weight,
                    wide_weight=proxy_wide_weight,
                    timeout_weight=proxy_timeout_weight,
                )
                valid_target = target.loc[valid_mask].copy().reset_index(drop=True)
                valid_reset = valid.reset_index(drop=True)
                proxy_score = proxy_score.reset_index(drop=True)
                for top_frac in top_fracs:
                    selected_ledger_rows.extend(
                        _selected_ledger_rows(
                            frame=valid_reset,
                            metrics=valid_metrics,
                            target=valid_target,
                            score=proxy_score,
                            arm=arm,
                            selector="feature_ic_proxy",
                            period=str(month),
                            top_frac=float(top_frac),
                        )
                    )
    fit_holdout = _fit_holdout_summary(
        monthly=monthly,
        weekly=weekly,
        fit_months=fit_months,
        holdout_month=holdout_month,
        min_week_rows=min_week_rows,
        min_fit_material_weeks=4,
        min_holdout_material_weeks=2,
        min_fit_positive_week_rate=0.55,
        min_holdout_positive_week_rate=0.50,
    )

    paths = {
        "label_summary": output_dir / "soft_label_summary.csv",
        "monthly": output_dir / "soft_label_monthly_selection.csv",
        "weekly": output_dir / "soft_label_weekly_selection.csv",
        "proxy_ic": output_dir / "soft_label_monthly_proxy_ic.csv",
        "fit_holdout": output_dir / "soft_label_fit_holdout.csv",
        "selected_ledger": output_dir / "soft_label_feature_proxy_selected_ledger.csv",
        "manifest": output_dir / "manifest.json",
    }
    label_summary.to_csv(paths["label_summary"], index=False)
    monthly.to_csv(paths["monthly"], index=False)
    weekly.to_csv(paths["weekly"], index=False)
    proxy_ic.to_csv(paths["proxy_ic"], index=False)
    fit_holdout.to_csv(paths["fit_holdout"], index=False)
    pd.DataFrame(selected_ledger_rows).to_csv(paths["selected_ledger"], index=False)

    manifest = {
        "labels_path": str(labels_path),
        "output_dir": str(output_dir),
        "rows": int(len(frame)),
        "timestamp_min": frame["__ts__"].min(),
        "timestamp_max": frame["__ts__"].max(),
        "symbols": int(frame["__symbol__"].nunique(dropna=True)),
        "feature_dir": str(feature_dir),
        "feature_list_csv": str(feature_list_csv),
        "feature_store": feature_store_report,
        "causal_outcome_priors": prior_report,
        "causal_state_path_priors": state_path_prior_report,
        "event_confirmation_features": event_confirmation_report,
        "slow_trade_diagnostic_features": slow_trade_report,
        "features": features,
        "feature_count": int(len(features)),
        "proxy_top_k": int(proxy_top_k),
        "proxy_objective": str(proxy_objective),
        "proxy_min_target_ic": float(proxy_min_target_ic),
        "proxy_min_utility_ic": float(proxy_min_utility_ic),
        "proxy_max_bad_mae_ic": float(proxy_max_bad_mae_ic),
        "proxy_max_wide_ic": float(proxy_max_wide_ic),
        "proxy_max_timeout_ic": float(proxy_max_timeout_ic),
        "proxy_utility_weight": float(proxy_utility_weight),
        "proxy_bad_mae_weight": float(proxy_bad_mae_weight),
        "proxy_wide_weight": float(proxy_wide_weight),
        "proxy_timeout_weight": float(proxy_timeout_weight),
        "top_fracs": [float(v) for v in top_fracs],
        "fit_months": [str(v) for v in fit_months],
        "holdout_month": str(holdout_month),
        "arms": list(targets.keys()),
        "rows_monthly": int(len(monthly)),
        "rows_weekly": int(len(weekly)),
        "rows_fit_holdout": int(len(fit_holdout)),
        "rows_selected_ledger": int(len(selected_ledger_rows)),
        "feature_proxy_fit_clean_pass_rows": int(
            fit_holdout[
                fit_holdout["selector"].eq("feature_ic_proxy")
                & fit_holdout["fit_clean_pass"]
            ].shape[0]
        )
        if not fit_holdout.empty
        else 0,
        "feature_proxy_holdout_clean_pass_rows": int(
            fit_holdout[
                fit_holdout["selector"].eq("feature_ic_proxy")
                & fit_holdout["holdout_clean_pass"]
            ].shape[0]
        )
        if not fit_holdout.empty
        else 0,
        "feature_proxy_fit_bounded_pass_rows": int(
            fit_holdout[
                fit_holdout["selector"].eq("feature_ic_proxy")
                & fit_holdout["fit_bounded_pass"]
            ].shape[0]
        )
        if not fit_holdout.empty
        else 0,
        "feature_proxy_holdout_bounded_pass_rows": int(
            fit_holdout[
                fit_holdout["selector"].eq("feature_ic_proxy")
                & fit_holdout["holdout_bounded_pass"]
            ].shape[0]
        )
        if not fit_holdout.empty
        else 0,
        "outputs": {key: str(value) for key, value in paths.items()},
    }
    markdown = _write_markdown(
        output_dir=output_dir,
        label_summary=label_summary,
        fit_holdout=fit_holdout,
        proxy_ic=proxy_ic,
        manifest=manifest,
    )
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
    parser.add_argument("--proxy-top-k", type=int, default=12)
    parser.add_argument("--proxy-objective", choices=PROXY_OBJECTIVES, default="target_ic")
    parser.add_argument("--proxy-min-target-ic", type=float, default=0.0)
    parser.add_argument("--proxy-min-utility-ic", type=float, default=0.0)
    parser.add_argument("--proxy-max-bad-mae-ic", type=float, default=0.0)
    parser.add_argument("--proxy-max-wide-ic", type=float, default=0.0)
    parser.add_argument("--proxy-max-timeout-ic", type=float, default=0.0)
    parser.add_argument("--proxy-utility-weight", type=float, default=1.0)
    parser.add_argument("--proxy-bad-mae-weight", type=float, default=1.0)
    parser.add_argument("--proxy-wide-weight", type=float, default=0.5)
    parser.add_argument("--proxy-timeout-weight", type=float, default=0.5)
    parser.add_argument("--top-fracs", type=_parse_float_csv, default=",".join(str(v) for v in DEFAULT_TOP_FRACS))
    parser.add_argument("--fit-months", type=_parse_csv, default=",".join(DEFAULT_FIT_MONTHS))
    parser.add_argument("--holdout-month", type=str, default=DEFAULT_HOLDOUT_MONTH)
    parser.add_argument("--min-week-rows", type=int, default=2)
    parser.add_argument("--arms", type=_parse_csv, default=None)
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
    parser.add_argument("--write-selected-ledger", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = run_ablation(
        labels_path=args.labels_path,
        output_dir=args.output_dir,
        feature_dir=args.feature_dir,
        feature_list_csv=args.feature_list_csv,
        max_feature_store_features=args.max_feature_store_features,
        proxy_top_k=int(args.proxy_top_k),
        proxy_objective=str(args.proxy_objective),
        proxy_min_target_ic=float(args.proxy_min_target_ic),
        proxy_min_utility_ic=float(args.proxy_min_utility_ic),
        proxy_max_bad_mae_ic=float(args.proxy_max_bad_mae_ic),
        proxy_max_wide_ic=float(args.proxy_max_wide_ic),
        proxy_max_timeout_ic=float(args.proxy_max_timeout_ic),
        proxy_utility_weight=float(args.proxy_utility_weight),
        proxy_bad_mae_weight=float(args.proxy_bad_mae_weight),
        proxy_wide_weight=float(args.proxy_wide_weight),
        proxy_timeout_weight=float(args.proxy_timeout_weight),
        top_fracs=list(args.top_fracs),
        fit_months=list(args.fit_months),
        holdout_month=str(args.holdout_month),
        min_week_rows=int(args.min_week_rows),
        arms=args.arms,
        include_causal_outcome_priors=bool(args.include_causal_outcome_priors),
        include_causal_state_path_priors=bool(args.include_causal_state_path_priors),
        include_event_confirmation_features=bool(args.include_event_confirmation_features),
        include_slow_trade_diagnostic_features=bool(args.include_slow_trade_diagnostic_features),
        prior_windows_days=list(args.prior_windows_days),
        prior_embargo_hours=float(args.prior_embargo_hours),
        state_path_prior_features=list(args.state_path_prior_features),
        event_feature_store_features=list(args.event_feature_store_features),
        write_selected_ledger=bool(args.write_selected_ledger),
    )
    print(json.dumps(_json_safe(manifest), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
