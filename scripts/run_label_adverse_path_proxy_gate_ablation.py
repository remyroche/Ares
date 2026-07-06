#!/usr/bin/env python3
"""Proxy-only adverse-path gate ablation for label learnability.

This is a pre-training diagnostic. It does not fit LightGBM, Optuna, policy
geometry, or the ExtraTrees smoke model. It tests whether causal feature proxies
can improve the recovery of clean S61/S62-style oracle rows while preserving
post-cost economics.
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

from scripts.report_label_proxy_quality_with_economic_limits import (  # noqa: E402
    DEFAULT_COMBINE_LABEL_WEIGHT,
    DEFAULT_FEATURE_LIST_CSV,
    DEFAULT_LABELS_PATH,
    DEFAULT_PROXY_TOP_K,
    DEFAULT_TOP_FRACS,
    _add_delta,
    _baseline,
    _parse_csv,
    _parse_float_csv,
    _score_proxy,
    _slice_week_positions,
    _top_gate,
)
from scripts.run_label_economic_proxy_ablation import _label_targets  # noqa: E402
from scripts.run_label_quality_proxy_diagnostics import (  # noqa: E402
    DEFAULT_FEATURE_DIR,
    _decile_diagnostics,
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
    _spearman,
)
from scripts.run_soft_label_economic_proxy_ablation import (  # noqa: E402
    DEFAULT_EVENT_FEATURE_STORE_FEATURES,
    DEFAULT_PRIOR_WINDOWS_DAYS,
    DEFAULT_STATE_PATH_PRIOR_FEATURES,
    _causal_outcome_prior_features,
    _causal_state_path_prior_features,
    _event_confirmation_features,
)
from scripts.run_soft_label_candidate_source_ablation import (  # noqa: E402
    _build_sources,
    _source_context,
)


DEFAULT_OUTPUT_DIR = Path(
    "data_perp/reports/label_adverse_path_proxy_gate_ablation_v1"
)
DEFAULT_LABEL_ARMS = (
    "S60_tpnet_severe_adverse_veto_path",
    "S61_tpnet_strict_adverse_veto_rank",
    "S62_tpnet_clean_dirty_contrast_rank",
)
DEFAULT_GATE_KEEP_FRACS = (0.30, 0.50, 0.70)
DEFAULT_RISK_PENALTIES = (0.25, 0.50, 1.00)
DEFAULT_SOURCE_NAMES = ("all",)
DEFAULT_FIT_MONTHS = ("2026-04", "2026-05")
DEFAULT_HOLDOUT_MONTH = "2026-06"


def _safe_numeric(values: Any) -> pd.Series:
    if isinstance(values, pd.Series):
        return pd.to_numeric(values, errors="coerce")
    return pd.to_numeric(pd.Series(values), errors="coerce")


def _mfe_mae(metrics: pd.DataFrame) -> pd.Series:
    out = metrics["mfe_norm"] / metrics["mae_norm"].clip(lower=0.25)
    return out.replace([np.inf, -np.inf], np.nan).clip(upper=10.0)


def _xs_rank_or_neutral(frame: pd.DataFrame, feature: str, *, high_good: bool = True) -> pd.Series:
    if feature not in frame.columns:
        return pd.Series(0.5, index=frame.index, dtype=np.float32)
    values = _safe_numeric(frame[feature])
    ranks = values.groupby(frame["__ts__"], dropna=False).rank(method="average", pct=True)
    if not high_good:
        ranks = 1.0 - ranks
    return ranks.fillna(0.5).clip(0.0, 1.0).astype(np.float32)


def _mean_available(parts: list[pd.Series], index: pd.Index) -> pd.Series:
    if not parts:
        return pd.Series(0.5, index=index, dtype=np.float32)
    return pd.concat(parts, axis=1).mean(axis=1).fillna(0.5).clip(0.0, 1.0).astype(np.float32)


def _adverse_path_composite_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Decision-time composites for adverse-path observability.

    These use only current-row feature values. Cross-sectional ranks are formed
    within the prediction timestamp, matching the existing event-confirmation
    feature style and avoiding future outcomes.
    """

    idx = frame.index

    def hi(*features: str) -> list[pd.Series]:
        return [_xs_rank_or_neutral(frame, feature, high_good=True) for feature in features if feature in frame.columns]

    def lo(*features: str) -> list[pd.Series]:
        return [_xs_rank_or_neutral(frame, feature, high_good=False) for feature in features if feature in frame.columns]

    liquid = _mean_available(
        lo(
            "spread_proxy_hl_range_bps_robust_z",
            "spread_proxy_abs_return_bps_robust_z",
            "spread_proxy_lower_wick_bps_robust_z",
            "spread_proxy_upper_wick_bps_robust_z",
            "spread_proxy_wick_to_range_robust_z",
            "median_spread_bps",
            "vol_price_spread",
        )
        + hi("xasset_ob_liquidity_peer_resid", "quote_volume_z_30d", "log_quote_volume"),
        idx,
    )
    event_impulse = _mean_available(
        hi(
            "shock_12h",
            "shock_vol_ratio",
            "impulse",
            "impulse_ratio_24",
            "jump_intensity",
            "second_leg_accel_1h",
            "second_leg_accel_2h",
            "second_leg_accel_vol_1h",
            "progress",
            "speed",
        ),
        idx,
    )
    confirmed_breakout = _mean_available(
        hi(
            "breakout_24h",
            "breakout_confirmed",
            "breakout_soft",
            "pct_breakout_t",
            "vw_breakout",
            "retest_quality",
            "trend_retest_success_rate",
            "volume_trend_alignment",
        ),
        idx,
    )
    exhaustion = _mean_available(
        hi(
            "climax_range_24",
            "climax_vol_12",
            "climax",
            "climax_decay",
            "trend_overextension_z",
            "trend_age_hours",
            "range_12h_pct",
            "range_24h_pct",
            "range_pct",
            "vol_expansion_ratio",
            "atr_expansion",
            "wick_ratio_4h_max",
            "wick_body_ratio",
            "wick_ratio",
        ),
        idx,
    )
    reversal_pressure = _mean_available(
        hi(
            "rejection_proxy",
            "impulse_reversal",
            "impulse_reversal_short",
            "stall",
            "stall_x_flow",
            "delta_stall_6",
            "cumulative_delta_stall",
            "mr_climax",
            "mr_failure",
            "tail_fail",
        ),
        idx,
    )
    low_barrier_pressure = _mean_available(
        lo(
            "up_barrier_pressure_daily_vwap",
            "down_barrier_pressure_daily_vwap",
            "distance_to_support_daily_vwap_atr",
            "distance_to_resistance_daily_vwap_atr",
            "dist_ema20_atr",
            "dist_ema50_atr",
            "dist_ema200_atr",
            "dist_ma100_atr",
        ),
        idx,
    )
    location_middle = _mean_available(
        [
            1.0 - (2.0 * (_xs_rank_or_neutral(frame, feature, high_good=True) - 0.5).abs()).clip(0.0, 1.0)
            for feature in [
                "loc_range_pos_24",
                "loc_range_pos_48",
                "loc_prev_week_range_pos_24",
                "loc_session_pos_24",
                "loc_swing_range_pos_24",
                "loc_ema_stack_pos_24",
                "loc_ema_stack_pos_48",
            ]
            if feature in frame.columns
        ],
        idx,
    )
    pullback_control = _mean_available(
        lo(
            "pullback_2",
            "pullback_4",
            "pullback_8",
            "pullback_48",
            "pullback_depth",
            "loc_pullback_depth_24",
            "loc_pullback_depth_48",
            "dist_from_low_event_12h",
        ),
        idx,
    )
    trend_quality = _mean_available(
        hi(
            "adx_7",
            "adx_10",
            "adx_14",
            "adx_di_plus_14",
            "trend_alignment_1_3_6",
            "trend_strength_percentile",
            "trend_r2_24",
            "trend_r2_48",
            "trend_t",
            "trend_z_t",
            "slope",
        )
        + lo("choppiness_cp_z_8_32_96", "binned_return_entropy_24", "directional_entropy_20", "price_entropy_cp_absratio_8_32"),
        idx,
    )
    oi_leverage_pressure = _mean_available(
        hi(
            "oi_rank",
            "oi_chg_2h",
            "oi_up_agree",
            "leverage_build_score",
            "abs_ret_per_oi_z_24h",
            "dist_oiw_intensity_12h_atr",
            "dist_oiw_z_delta_12h_atr",
            "dist_oiw_z_delta_96h_atr",
        ),
        idx,
    )
    fresh_event = _mean_available(
        lo("time_since_event_extreme_12h")
        + hi("dist_from_low_event_12h", "breakout_min"),
        idx,
    )
    compression_breakout = _mean_available(
        hi("atr_compression_ratio", "vol_compression", "compression_score", "breakout_confirmed", "breakout_24h")
        + lo("spread_proxy_hl_range_bps_robust_z", "spread_proxy_abs_return_bps_robust_z"),
        idx,
    )

    out: dict[str, pd.Series] = {
        "ap_liquid_context": liquid,
        "ap_event_impulse": event_impulse,
        "ap_confirmed_breakout": confirmed_breakout,
        "ap_exhaustion_pressure": exhaustion,
        "ap_reversal_pressure": reversal_pressure,
        "ap_low_barrier_pressure": low_barrier_pressure,
        "ap_location_middle": location_middle,
        "ap_pullback_control": pullback_control,
        "ap_trend_quality": trend_quality,
        "ap_oi_leverage_pressure": oi_leverage_pressure,
        "ap_fresh_event": fresh_event,
        "ap_compression_breakout": compression_breakout,
    }

    def product(name: str, left: pd.Series, right: pd.Series) -> None:
        out[name] = (left.fillna(0.5) * right.fillna(0.5)).clip(0.0, 1.0).astype(np.float32)

    product("ap_liquid_confirmed_breakout", liquid, confirmed_breakout)
    product("ap_clean_continuation_context", liquid, confirmed_breakout * trend_quality * low_barrier_pressure)
    product("ap_fresh_liquid_breakout", fresh_event, liquid * confirmed_breakout)
    product("ap_pullback_confirmed_liquid", pullback_control, confirmed_breakout * liquid)
    product("ap_compression_breakout_liquid", compression_breakout, liquid)
    product("ap_overheated_event_risk", event_impulse, exhaustion)
    product("ap_overheated_reversal_risk", exhaustion, reversal_pressure)
    product("ap_oi_impulse_risk", oi_leverage_pressure, event_impulse)
    product("ap_dirty_pressure_composite", exhaustion, reversal_pressure * (1.0 - liquid.fillna(0.5)))

    features = pd.DataFrame(out, index=frame.index).astype(np.float32, copy=False)
    finite = features.notna().mean()
    source_features = sorted(
        {
            feature
            for feature in frame.columns
            if any(
                token in feature
                for token in (
                    "spread",
                    "liquidity",
                    "barrier",
                    "distance_to",
                    "impulse",
                    "shock",
                    "breakout",
                    "pullback",
                    "wick",
                    "body",
                    "range",
                    "loc_",
                    "adx",
                    "oi",
                    "volume",
                    "rejection",
                    "trend",
                    "stall",
                    "climax",
                    "entropy",
                    "compression",
                )
            )
        }
    )
    return features, {
        "enabled": True,
        "feature_count": int(features.shape[1]),
        "feature_names": list(features.columns),
        "source_feature_count": int(len(source_features)),
        "source_features_used_or_available": source_features[:200],
        "mean_finite_frac": float(finite.mean()) if len(finite) else 0.0,
        "min_finite_frac": float(finite.min()) if len(finite) else 0.0,
    }


def _path_targets(metrics: pd.DataFrame) -> dict[str, pd.Series]:
    mfe_mae = _mfe_mae(metrics)
    timeout = metrics["is_timeout"].astype(float).fillna(1.0)
    strict_clean = (
        (metrics["u_policy_net"] > 0.0)
        & (metrics["mae_norm"] <= 0.85)
        & (metrics["barrier"] <= 0.024)
        & (mfe_mae >= 1.35)
        & (timeout <= 0.0)
    ).astype(float)
    bounded = (
        (metrics["u_policy_net"] > 0.0)
        & (metrics["mae_norm"] <= 1.0)
        & (metrics["barrier"] <= 0.035)
        & (mfe_mae >= 1.25)
        & (timeout <= 0.0)
    ).astype(float)
    bad_mae = (metrics["mae_norm"] >= 1.0).astype(float)
    wide_25 = (metrics["barrier"] > 0.025).astype(float)
    dirty = (
        (metrics["mae_norm"] >= 1.0)
        | (metrics["barrier"] > 0.035)
        | (mfe_mae < 1.25)
        | (timeout > 0.0)
        | (metrics["u_policy_net"] <= 0.0)
    ).astype(float)
    quality = (
        0.35 * strict_clean
        + 0.25 * bounded
        + 0.20 * (1.0 - bad_mae)
        + 0.10 * (1.0 - wide_25)
        + 0.10 * (1.0 - timeout.clip(0.0, 1.0))
    ).clip(0.0, 1.0)
    return {
        "strict_clean": strict_clean.astype(float),
        "bounded": bounded.astype(float),
        "bad_mae": bad_mae.astype(float),
        "wide_25": wide_25.astype(float),
        "timeout": timeout.clip(0.0, 1.0).astype(float),
        "dirty": dirty.astype(float),
        "path_quality": quality.astype(float),
    }


def _top_mask(score: pd.Series, frac: float) -> pd.Series:
    out = pd.Series(False, index=score.index)
    idx = _rank_top_indices(score, frac)
    if len(idx):
        out.iloc[idx] = True
    return out


def _score_period(
    *,
    frame: pd.DataFrame,
    metrics: pd.DataFrame,
    target: pd.DataFrame,
    score: pd.Series,
    oracle_score: pd.Series,
    period_type: str,
    period: str,
    month: str,
    source: str,
    selector: str,
    label_arm: str,
    top_frac: float,
    label_score: pd.Series,
    clean_score: pd.Series,
    risk_score: pd.Series,
    selector_features: str,
) -> dict[str, Any]:
    score = _safe_numeric(score).reset_index(drop=True)
    frame = frame.reset_index(drop=True)
    metrics = metrics.reset_index(drop=True)
    target = target.reset_index(drop=True)
    oracle_score = _safe_numeric(oracle_score).reset_index(drop=True)
    label_score = _safe_numeric(label_score).reset_index(drop=True)
    clean_score = _safe_numeric(clean_score).reset_index(drop=True)
    risk_score = _safe_numeric(risk_score).reset_index(drop=True)

    row = _selection_metrics(
        frame=frame,
        metrics=metrics,
        target=target,
        score=score,
        arm=f"{selector}::{label_arm}",
        selector=selector,
        period=period,
        top_frac=top_frac,
    )
    _add_delta(row, _baseline(metrics))

    selected = _top_mask(score, top_frac)
    oracle = _top_mask(oracle_score, top_frac)
    recovered = selected & oracle
    row.update(
        {
            "period_type": period_type,
            "month": month,
            "source": source,
            "label_arm": label_arm,
            "selector_features": selector_features,
            "score_ic_u": _spearman(score, metrics["u_policy_net"]),
            "score_ic_label": _spearman(score, label_score),
            "score_ic_clean": _spearman(score, clean_score),
            "score_ic_risk": _spearman(score, risk_score),
            "oracle_top_rows": int(oracle.sum()),
            "oracle_recovered_rows": int(recovered.sum()),
            "oracle_recovery_rate": float(recovered.sum() / oracle.sum()) if int(oracle.sum()) else 0.0,
            "selected_oracle_overlap_rate": (
                float(recovered.sum() / selected.sum()) if int(selected.sum()) else 0.0
            ),
        }
    )
    row.update(_decile_diagnostics(score, metrics["u_policy_net"]))
    return row


def _selector_scores(
    *,
    label_score: pd.Series,
    clean_score: pd.Series,
    quality_score: pd.Series,
    bad_mae_score: pd.Series,
    dirty_score: pd.Series,
    timeout_score: pd.Series,
    gate_keep_fracs: list[float],
    risk_penalties: list[float],
    combine_label_weight: float,
) -> list[tuple[str, pd.Series]]:
    label = _safe_numeric(label_score).reset_index(drop=True)
    clean = _safe_numeric(clean_score).reset_index(drop=True).fillna(0.5).clip(0.0, 1.0)
    quality = _safe_numeric(quality_score).reset_index(drop=True).fillna(0.5).clip(0.0, 1.0)
    bad = _safe_numeric(bad_mae_score).reset_index(drop=True).fillna(0.5).clip(0.0, 1.0)
    dirty = _safe_numeric(dirty_score).reset_index(drop=True).fillna(0.5).clip(0.0, 1.0)
    timeout = _safe_numeric(timeout_score).reset_index(drop=True).fillna(0.5).clip(0.0, 1.0)
    risk = ((bad + dirty + timeout) / 3.0).clip(0.0, 1.0)
    safe = ((clean + quality + (1.0 - bad) + (1.0 - dirty) + (1.0 - timeout)) / 5.0).clip(0.0, 1.0)

    out: list[tuple[str, pd.Series]] = [
        ("label_proxy_oos", label),
        (
            f"label{combine_label_weight:.2f}_clean{1.0 - combine_label_weight:.2f}_blend",
            (combine_label_weight * label + (1.0 - combine_label_weight) * clean),
        ),
        (
            f"label{combine_label_weight:.2f}_path_quality{1.0 - combine_label_weight:.2f}_blend",
            (combine_label_weight * label + (1.0 - combine_label_weight) * quality),
        ),
        (
            f"label{combine_label_weight:.2f}_safe{1.0 - combine_label_weight:.2f}_blend",
            (combine_label_weight * label + (1.0 - combine_label_weight) * safe),
        ),
    ]
    for penalty in risk_penalties:
        out.extend(
            [
                (f"label_minus_badmae_{penalty:.2f}", label - float(penalty) * bad),
                (f"label_minus_dirty_{penalty:.2f}", label - float(penalty) * dirty),
                (f"label_minus_composite_risk_{penalty:.2f}", label - float(penalty) * risk),
            ]
        )
    for keep_frac in gate_keep_fracs:
        clean_gate = _top_gate(clean, keep_frac)
        quality_gate = _top_gate(quality, keep_frac)
        safe_gate = _top_gate(safe, keep_frac)
        low_bad_gate = _top_gate(-bad, keep_frac)
        low_dirty_gate = _top_gate(-dirty, keep_frac)
        low_timeout_gate = _top_gate(-timeout, keep_frac)
        low_composite_gate = _top_gate(-risk, keep_frac)
        out.extend(
            [
                (f"clean_gate{keep_frac:.2f}_then_label", label.where(clean_gate)),
                (f"path_quality_gate{keep_frac:.2f}_then_label", label.where(quality_gate)),
                (f"safe_gate{keep_frac:.2f}_then_label", label.where(safe_gate)),
                (f"low_badmae_gate{keep_frac:.2f}_then_label", label.where(low_bad_gate)),
                (f"low_dirty_gate{keep_frac:.2f}_then_label", label.where(low_dirty_gate)),
                (f"low_timeout_gate{keep_frac:.2f}_then_label", label.where(low_timeout_gate)),
                (f"low_composite_risk_gate{keep_frac:.2f}_then_label", label.where(low_composite_gate)),
                (
                    f"clean_lowrisk_gate{keep_frac:.2f}_then_label",
                    label.where(clean_gate & low_composite_gate),
                ),
                (
                    f"quality_lowrisk_gate{keep_frac:.2f}_then_label",
                    label.where(quality_gate & low_composite_gate),
                ),
            ]
        )
    return out


def _aggregate(rows: pd.DataFrame, *, min_material_selected_rows: int) -> pd.DataFrame:
    if rows.empty:
        return rows
    out_rows: list[dict[str, Any]] = []
    group_cols = [
        col
        for col in ["period_type", "source", "selector", "label_arm", "confusion_arm", "top_frac"]
        if col in rows.columns
    ]
    for key, group in rows.groupby(group_cols, dropna=False, observed=True):
        if not isinstance(key, tuple):
            key = (key,)
        key_values = dict(zip(group_cols, key))
        period_type = key_values.get("period_type")
        selector = key_values.get("selector")
        label_arm = key_values.get("label_arm")
        top_frac = key_values.get("top_frac")
        selected_rows = _safe_numeric(group["selected_rows"])
        mean_return = _safe_numeric(group["mean_return_net"])
        sum_return = (mean_return * selected_rows).replace([np.inf, -np.inf], np.nan)
        sum_return_plus10 = ((mean_return - 0.0010) * selected_rows).replace([np.inf, -np.inf], np.nan)
        material = group[selected_rows.ge(int(min_material_selected_rows)).to_numpy()].copy()
        material_return = _safe_numeric(material["mean_return_net"]) if len(material) else pd.Series(dtype=float)
        material_rows = _safe_numeric(material["selected_rows"]) if len(material) else pd.Series(dtype=float)
        material_sum_return_plus10 = ((material_return - 0.0010) * material_rows).replace(
            [np.inf, -np.inf],
            np.nan,
        )
        periods = int(len(group))
        material_periods = int(len(material))
        row = {
            "period_type": period_type,
            "selector": selector,
            "label_arm": label_arm,
            "top_frac": float(top_frac),
            "periods": periods,
            "positive_return_periods": int((mean_return > 0.0).sum()),
            "positive_return_period_rate": float((mean_return > 0.0).mean()) if periods else float("nan"),
            "material_periods": material_periods,
            "material_period_rate": material_periods / periods if periods else float("nan"),
            "positive_material_return_periods": int((material_return > 0.0).sum()) if material_periods else 0,
            "positive_material_return_period_rate": (
                float((material_return > 0.0).mean()) if material_periods else float("nan")
            ),
            "mean_return_net": _safe_mean(mean_return),
            "worst_period_return_net": _safe_quantile(mean_return, 0.0),
            "q25_period_return_net": _safe_quantile(mean_return, 0.25),
            "sum_return_net": float(sum_return.sum(skipna=True)),
            "sum_return_net_plus10bps": float(sum_return_plus10.sum(skipna=True)),
            "material_sum_return_net_plus10bps": (
                float(material_sum_return_plus10.sum(skipna=True)) if material_periods else 0.0
            ),
            "mean_u": _safe_mean(group["mean_u"]),
            "worst_period_mean_u": _safe_quantile(group["mean_u"], 0.0),
            "hit_u": _safe_mean(group["hit_u"]),
            "q10_u": _safe_mean(group["q10_u"]),
            "delta_mean_u_vs_period": _safe_mean(group["delta_mean_u_vs_period"]),
            "score_ic_u": _safe_mean(group["score_ic_u"]),
            "score_ic_label": _safe_mean(group["score_ic_label"]),
            "score_ic_clean": _safe_mean(group["score_ic_clean"]),
            "score_ic_risk": _safe_mean(group["score_ic_risk"]),
            "decile_spearman_u": _safe_mean(group["decile_spearman_u"]),
            "top_bottom_decile_spread_u": _safe_mean(group["top_bottom_decile_spread_u"]),
            "bad_mae_1r_rate": _safe_mean(group["bad_mae_1r_rate"]),
            "p90_mae_norm": _safe_mean(group["p90_mae_norm"]),
            "wide_barrier_25bps_rate": _safe_mean(group["wide_barrier_25bps_rate"]),
            "timeout_rate": _safe_mean(group["timeout_rate"]),
            "strict_clean_row_rate": _safe_mean(group["strict_clean_row_rate"]),
            "bounded_row_rate": _safe_mean(group["bounded_row_rate"]),
            "mean_mfe_mae_ratio": _safe_mean(group["mean_mfe_mae_ratio"]),
            "oracle_top_rows": int(_safe_numeric(group["oracle_top_rows"]).sum(skipna=True)),
            "oracle_recovered_rows": int(_safe_numeric(group["oracle_recovered_rows"]).sum(skipna=True)),
            "mean_oracle_recovery_rate": _safe_mean(group["oracle_recovery_rate"]),
            "mean_selected_oracle_overlap_rate": _safe_mean(group["selected_oracle_overlap_rate"]),
            "mean_selected_rows": _safe_mean(selected_rows),
            "min_selected_rows": int(selected_rows.min()) if len(selected_rows.dropna()) else 0,
            "top_symbol_share": _safe_mean(group["top_symbol_share"]),
            "selector_features": str(group["selector_features"].dropna().iloc[0])
            if group["selector_features"].dropna().size
            else "",
        }
        for optional_col in ("source", "confusion_arm"):
            if optional_col in key_values:
                row[optional_col] = key_values[optional_col]
        row["overall_oracle_recovery_rate"] = (
            row["oracle_recovered_rows"] / row["oracle_top_rows"] if row["oracle_top_rows"] else 0.0
        )
        monthly_stable = (
            row["positive_return_period_rate"] >= 1.0
            and row["worst_period_return_net"] > 0.0
        )
        weekly_stable = (
            row["positive_material_return_period_rate"] >= 0.60
            and row["q25_period_return_net"] > 0.0
            and row["material_sum_return_net_plus10bps"] > 0.0
        )
        stable = monthly_stable if period_type == "month" else weekly_stable
        row["acceptance_gate"] = (
            bool(row["mean_return_net"] > 0.0)
            and bool(row["sum_return_net_plus10bps"] > 0.0)
            and bool(stable)
            and bool(row["score_ic_u"] > 0.0)
            and bool(row["bad_mae_1r_rate"] <= 0.40)
            and bool(row["p90_mae_norm"] <= 4.00)
            and bool(row["wide_barrier_25bps_rate"] <= 0.05)
            and bool(row["overall_oracle_recovery_rate"] >= 0.25)
        )
        out_rows.append(row)
    out = pd.DataFrame(out_rows)
    return out.sort_values(
        ["acceptance_gate", "period_type", "top_frac", "sum_return_net_plus10bps"],
        ascending=[False, True, True, False],
    )


def _safe_min(values: Any) -> float:
    series = _safe_numeric(values).dropna()
    return float(series.min()) if len(series) else float("nan")


def _weighted_mean(frame: pd.DataFrame, value_col: str, weight_col: str) -> float:
    if frame.empty or value_col not in frame.columns or weight_col not in frame.columns:
        return float("nan")
    values = _safe_numeric(frame[value_col])
    weights = _safe_numeric(frame[weight_col]).fillna(0.0)
    mask = values.notna() & weights.gt(0.0)
    if not bool(mask.any()):
        return float("nan")
    return float(np.average(values[mask], weights=weights[mask]))


def _summarize_month(prefix: str, frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {
            f"{prefix}_months": 0,
            f"{prefix}_positive_months": 0,
            f"{prefix}_mean_month_return_net": float("nan"),
            f"{prefix}_worst_month_return_net": float("nan"),
            f"{prefix}_selected_rows": 0,
            f"{prefix}_bad_mae_1r_rate": float("nan"),
            f"{prefix}_p90_mae_norm": float("nan"),
            f"{prefix}_wide_25bps_rate": float("nan"),
            f"{prefix}_timeout_rate": float("nan"),
            f"{prefix}_strict_clean_row_rate": float("nan"),
            f"{prefix}_bounded_row_rate": float("nan"),
            f"{prefix}_mean_oracle_recovery_rate": float("nan"),
            f"{prefix}_score_ic_u": float("nan"),
        }
    returns = _safe_numeric(frame["mean_return_net"])
    return {
        f"{prefix}_months": int(frame["month"].astype(str).nunique()),
        f"{prefix}_positive_months": int(returns.gt(0.0).sum()),
        f"{prefix}_mean_month_return_net": _safe_mean(returns),
        f"{prefix}_worst_month_return_net": _safe_min(returns),
        f"{prefix}_selected_rows": int(_safe_numeric(frame["selected_rows"]).sum(skipna=True)),
        f"{prefix}_bad_mae_1r_rate": _weighted_mean(frame, "bad_mae_1r_rate", "selected_rows"),
        f"{prefix}_p90_mae_norm": _weighted_mean(frame, "p90_mae_norm", "selected_rows"),
        f"{prefix}_wide_25bps_rate": _weighted_mean(frame, "wide_barrier_25bps_rate", "selected_rows"),
        f"{prefix}_timeout_rate": _weighted_mean(frame, "timeout_rate", "selected_rows"),
        f"{prefix}_strict_clean_row_rate": _weighted_mean(frame, "strict_clean_row_rate", "selected_rows"),
        f"{prefix}_bounded_row_rate": _weighted_mean(frame, "bounded_row_rate", "selected_rows"),
        f"{prefix}_mean_oracle_recovery_rate": _safe_mean(frame["oracle_recovery_rate"]),
        f"{prefix}_score_ic_u": _safe_mean(frame["score_ic_u"]),
    }


def _summarize_week(prefix: str, frame: pd.DataFrame, *, min_week_rows: int) -> dict[str, Any]:
    if frame.empty:
        return {
            f"{prefix}_weeks": 0,
            f"{prefix}_material_weeks": 0,
            f"{prefix}_material_positive_week_rate": float("nan"),
            f"{prefix}_q25_week_return_net": float("nan"),
            f"{prefix}_worst_week_return_net": float("nan"),
        }
    returns = _safe_numeric(frame["mean_return_net"])
    selected_rows = _safe_numeric(frame["selected_rows"]).fillna(0.0)
    material = selected_rows >= int(min_week_rows)
    positive = returns > 0.0
    return {
        f"{prefix}_weeks": int(len(frame)),
        f"{prefix}_material_weeks": int(material.sum()),
        f"{prefix}_material_positive_week_rate": float((positive & material).sum() / material.sum())
        if int(material.sum())
        else float("nan"),
        f"{prefix}_q25_week_return_net": _safe_quantile(returns[material], 0.25) if int(material.sum()) else float("nan"),
        f"{prefix}_worst_week_return_net": _safe_min(returns[material]) if int(material.sum()) else float("nan"),
    }


def _fit_holdout_summary(
    period_rows: pd.DataFrame,
    *,
    fit_months: list[str],
    holdout_month: str,
    min_week_rows: int,
    min_fit_material_weeks: int,
    min_holdout_material_weeks: int,
    min_fit_positive_week_rate: float,
    min_holdout_positive_week_rate: float,
) -> pd.DataFrame:
    if period_rows.empty:
        return pd.DataFrame()
    monthly = period_rows[period_rows["period_type"].eq("month")].copy()
    weekly = period_rows[period_rows["period_type"].eq("week")].copy()
    if monthly.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    group_cols = ["source", "selector", "label_arm", "top_frac"]
    for key, group in monthly.groupby(group_cols, observed=True, dropna=False):
        source, selector, label_arm, top_frac = key
        week_group = weekly[
            weekly["source"].astype(str).eq(str(source))
            & weekly["selector"].astype(str).eq(str(selector))
            & weekly["label_arm"].astype(str).eq(str(label_arm))
            & _safe_numeric(weekly["top_frac"]).eq(float(top_frac))
        ].copy()
        fit_month = group[group["month"].astype(str).isin(fit_months)].copy()
        holdout_monthly = group[group["month"].astype(str).eq(str(holdout_month))].copy()
        fit_week = week_group[week_group["month"].astype(str).isin(fit_months)].copy()
        holdout_week = week_group[week_group["month"].astype(str).eq(str(holdout_month))].copy()
        if fit_month.empty or holdout_monthly.empty:
            continue

        row: dict[str, Any] = {
            "source": str(source),
            "selector": str(selector),
            "label_arm": str(label_arm),
            "top_frac": float(top_frac),
        }
        row.update(_summarize_month("fit", fit_month))
        row.update(_summarize_month("holdout", holdout_monthly))
        row.update(_summarize_week("fit", fit_week, min_week_rows=min_week_rows))
        row.update(_summarize_week("holdout", holdout_week, min_week_rows=min_week_rows))

        fit_sign = (
            row["fit_months"] == len(fit_months)
            and row["fit_positive_months"] == len(fit_months)
            and row["fit_worst_month_return_net"] > 0.0
            and row["fit_material_weeks"] >= int(min_fit_material_weeks)
            and row["fit_material_positive_week_rate"] >= float(min_fit_positive_week_rate)
        )
        holdout_sign = (
            row["holdout_positive_months"] >= 1
            and row["holdout_mean_month_return_net"] > 0.0
            and row["holdout_material_weeks"] >= int(min_holdout_material_weeks)
            and row["holdout_material_positive_week_rate"] >= float(min_holdout_positive_week_rate)
        )
        fit_economic = (
            fit_sign
            and row["fit_score_ic_u"] > 0.0
            and row["fit_bad_mae_1r_rate"] <= 0.40
            and row["fit_p90_mae_norm"] <= 4.00
            and row["fit_wide_25bps_rate"] <= 0.05
        )
        holdout_economic = (
            holdout_sign
            and row["holdout_score_ic_u"] > 0.0
            and row["holdout_bad_mae_1r_rate"] <= 0.40
            and row["holdout_p90_mae_norm"] <= 4.00
            and row["holdout_wide_25bps_rate"] <= 0.05
        )
        fit_oracle = row["fit_mean_oracle_recovery_rate"] >= 0.25
        holdout_oracle = row["holdout_mean_oracle_recovery_rate"] >= 0.25
        row["fit_sign_pass"] = bool(fit_sign)
        row["holdout_sign_pass"] = bool(holdout_sign)
        row["fit_economic_pass"] = bool(fit_economic)
        row["holdout_economic_pass"] = bool(holdout_economic)
        row["fit_oracle_recovery_pass"] = bool(fit_oracle)
        row["holdout_oracle_recovery_pass"] = bool(holdout_oracle)
        row["trainworthy_pass"] = bool(fit_economic and holdout_economic and fit_oracle and holdout_oracle)
        row["positive_dirty_holdout"] = bool(holdout_sign and not holdout_economic)
        row["holdout_risk_score"] = float(
            (row["holdout_mean_month_return_net"] if pd.notna(row["holdout_mean_month_return_net"]) else 0.0)
            + 0.50 * (row["holdout_q25_week_return_net"] if pd.notna(row["holdout_q25_week_return_net"]) else 0.0)
            - 0.020 * (row["holdout_bad_mae_1r_rate"] if pd.notna(row["holdout_bad_mae_1r_rate"]) else 0.0)
            - 0.003 * (row["holdout_p90_mae_norm"] if pd.notna(row["holdout_p90_mae_norm"]) else 0.0)
        )
        rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(
        ["trainworthy_pass", "holdout_economic_pass", "fit_economic_pass", "positive_dirty_holdout", "holdout_risk_score"],
        ascending=[False, False, False, False, False],
    )


def _table(frame: pd.DataFrame, cols: list[str], limit: int | None = None) -> str:
    if frame.empty:
        return "No rows."
    view = frame[[col for col in cols if col in frame.columns]].copy()
    if limit is not None:
        view = view.head(limit)
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda value: f"{float(value):.4f}" if pd.notna(value) else "")
    return view.to_markdown(index=False)


def _write_markdown(
    output_dir: Path,
    aggregate: pd.DataFrame,
    period_rows: pd.DataFrame,
    fit_holdout: pd.DataFrame,
    manifest: dict[str, Any],
) -> Path:
    path = output_dir / "label_adverse_path_proxy_gate_ablation.md"
    cols = [
        "acceptance_gate",
        "period_type",
        "source",
        "selector",
        "label_arm",
        "top_frac",
        "periods",
        "positive_return_period_rate",
        "mean_return_net",
        "worst_period_return_net",
        "sum_return_net_plus10bps",
        "score_ic_u",
        "score_ic_clean",
        "score_ic_risk",
        "bad_mae_1r_rate",
        "p90_mae_norm",
        "strict_clean_row_rate",
        "bounded_row_rate",
        "timeout_rate",
        "overall_oracle_recovery_rate",
        "mean_selected_rows",
    ]
    period_cols = [
        "period",
        "source",
        "selector",
        "label_arm",
        "top_frac",
        "selected_rows",
        "mean_return_net",
        "score_ic_u",
        "bad_mae_1r_rate",
        "p90_mae_norm",
        "strict_clean_row_rate",
        "timeout_rate",
        "oracle_recovery_rate",
    ]
    fit_cols = [
        "trainworthy_pass",
        "fit_economic_pass",
        "holdout_economic_pass",
        "fit_oracle_recovery_pass",
        "holdout_oracle_recovery_pass",
        "source",
        "selector",
        "label_arm",
        "top_frac",
        "fit_mean_month_return_net",
        "holdout_mean_month_return_net",
        "fit_bad_mae_1r_rate",
        "holdout_bad_mae_1r_rate",
        "fit_p90_mae_norm",
        "holdout_p90_mae_norm",
        "fit_wide_25bps_rate",
        "holdout_wide_25bps_rate",
        "fit_mean_oracle_recovery_rate",
        "holdout_mean_oracle_recovery_rate",
        "fit_material_positive_week_rate",
        "holdout_material_positive_week_rate",
        "fit_selected_rows",
        "holdout_selected_rows",
    ]
    lines = [
        "# Label Adverse-Path Proxy Gate Ablation",
        "",
        "Scope: proxy-only development diagnostic. No LightGBM, Optuna, policy optimization, or tree smoke model is run.",
        "",
        "The acceptance gate requires positive post-cost returns after +10 bps stress, positive score IC, bad-MAE <= 40%, p90 MAE <= 4R, wide-barrier <= 5%, and at least 25% oracle-clean recovery.",
        "",
        f"Labels: `{', '.join(manifest['label_arms'])}`",
        f"Sources: `{', '.join(manifest['sources'])}`",
        f"Top fractions: `{manifest['top_fracs']}`",
        f"Fit months: `{', '.join(manifest['fit_months'])}`. Holdout month: `{manifest['holdout_month']}`.",
        f"Gate keep fractions: `{manifest['gate_keep_fracs']}`",
        f"Risk penalties: `{manifest['risk_penalties']}`",
        f"Causal outcome priors: `{manifest['include_causal_outcome_priors']}`",
        f"Causal state-path priors: `{manifest['include_causal_state_path_priors']}`",
        f"Event confirmation features: `{manifest['include_event_confirmation_features']}`",
        f"Adverse-path composites: `{manifest['include_adverse_path_composites']}`",
        "",
    ]
    for period_type in ("month", "week"):
        subset = aggregate[aggregate["period_type"].eq(period_type)].copy()
        lines.extend(
            [
                f"## {period_type.title()} Aggregate",
                "",
                _table(
                    subset.sort_values(
                        ["acceptance_gate", "overall_oracle_recovery_rate", "sum_return_net_plus10bps"],
                        ascending=[False, False, False],
                    ),
                    cols,
                    limit=40,
                ),
                "",
            ]
        )
    lines.extend(
        [
            "## Fit / Holdout Gate",
            "",
            "Train-worthy rows require April-May fit economics, June holdout economics, positive score IC, bad-MAE <= 40%, p90 MAE <= 4R, wide-barrier <= 5%, and >= 25% oracle recovery on both fit and holdout.",
            "",
            _table(
                fit_holdout.sort_values(
                    ["trainworthy_pass", "holdout_economic_pass", "fit_economic_pass", "holdout_risk_score"],
                    ascending=[False, False, False, False],
                )
                if not fit_holdout.empty
                else fit_holdout,
                fit_cols,
                limit=60,
            ),
            "",
        ]
    )
    focus = period_rows[
        period_rows["period_type"].eq("month")
        & period_rows["selector"].isin(
            [
                "label_proxy_oos",
                "safe_gate0.30_then_label",
                "low_composite_risk_gate0.30_then_label",
                "clean_lowrisk_gate0.30_then_label",
                "label_minus_composite_risk_0.50",
            ]
        )
    ].copy()
    lines.extend(
        [
            "## Month Detail Focus",
            "",
            _table(
                focus.sort_values(["period", "source", "label_arm", "selector", "top_frac"]),
                period_cols,
                limit=160,
            ),
            "",
            "## Outputs",
            "",
            f"- Period rows: `{manifest['outputs']['period_rows']}`",
            f"- Aggregate: `{manifest['outputs']['aggregate']}`",
            f"- Fit/holdout: `{manifest['outputs']['fit_holdout']}`",
            f"- Manifest: `{manifest['outputs']['manifest']}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_report(
    *,
    labels_path: Path,
    output_dir: Path,
    feature_dir: Path,
    feature_list_csv: Path,
    max_feature_store_features: int | None,
    label_arms: list[str],
    top_fracs: list[float],
    gate_keep_fracs: list[float],
    risk_penalties: list[float],
    proxy_top_k: int,
    combine_label_weight: float,
    include_causal_outcome_priors: bool,
    include_causal_state_path_priors: bool,
    include_event_confirmation_features: bool,
    include_adverse_path_composites: bool,
    prior_windows_days: list[float],
    prior_embargo_hours: float,
    state_path_prior_features: list[str],
    event_feature_store_features: list[str],
    min_material_selected_rows: int,
    sources: list[str] | None,
    run_gap_hours: float,
    min_train_source_rows: int,
    min_valid_source_rows: int,
    fit_months: list[str],
    holdout_month: str,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = _load_labels(labels_path)
    selected_features = _read_feature_list(feature_list_csv, max_features=max_feature_store_features)
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
    outcome_prior_report: dict[str, Any] = {"enabled": False}
    if include_causal_outcome_priors:
        prior_features, outcome_prior_report = _causal_outcome_prior_features(
            frame,
            metrics,
            windows_days=prior_windows_days,
            embargo_hours=prior_embargo_hours,
        )
        frame = pd.concat([frame, prior_features.astype(np.float32, copy=False)], axis=1).copy()

    state_path_prior_report: dict[str, Any] = {"enabled": False}
    if include_causal_state_path_priors:
        state_prior_features, state_path_prior_report = _causal_state_path_prior_features(
            frame,
            metrics,
            state_features=state_path_prior_features,
            windows_days=prior_windows_days,
            embargo_hours=prior_embargo_hours,
        )
        frame = pd.concat([frame, state_prior_features.astype(np.float32, copy=False)], axis=1).copy()

    event_confirmation_report: dict[str, Any] = {"enabled": False}
    if include_event_confirmation_features:
        event_features, event_confirmation_report = _event_confirmation_features(
            frame,
            event_features=event_feature_store_features,
        )
        frame = pd.concat([frame, event_features.astype(np.float32, copy=False)], axis=1).copy()

    adverse_path_composite_report: dict[str, Any] = {"enabled": False}
    if include_adverse_path_composites:
        adverse_path_features, adverse_path_composite_report = _adverse_path_composite_features(frame)
        frame = pd.concat([frame, adverse_path_features.astype(np.float32, copy=False)], axis=1).copy()

    source_context = _source_context(frame)
    overlap = [col for col in source_context.columns if col in frame.columns]
    if overlap:
        frame = frame.drop(columns=overlap)
    frame = pd.concat([frame, source_context.astype(np.float32, copy=False)], axis=1)
    source_masks_all = _build_sources(frame, source_context, run_gap_hours=float(run_gap_hours))
    requested_sources = list(sources) if sources else list(DEFAULT_SOURCE_NAMES)
    unknown_sources = sorted(set(requested_sources).difference(source_masks_all))
    if unknown_sources:
        raise ValueError(f"Unknown sources: {unknown_sources}. Available: {sorted(source_masks_all)}")
    selected_sources = {
        source: source_masks_all[source].fillna(False).astype(bool).reindex(frame.index, fill_value=False)
        for source in requested_sources
    }

    features = _feature_columns(frame)
    targets = _label_targets(frame, metrics)
    unknown = sorted(set(label_arms).difference(targets))
    if unknown:
        raise ValueError(f"Unknown label arms: {unknown}")
    path_targets = _path_targets(metrics)
    month_period = frame["__ts__"].dt.to_period("M").astype(str)
    months = sorted(m for m in month_period.dropna().unique().tolist() if m >= "2026-04")

    rows: list[dict[str, Any]] = []
    for source_name, source_mask in selected_sources.items():
        for month in months:
            train_mask = month_period.lt(str(month)) & source_mask
            valid_mask = month_period.eq(str(month)) & source_mask
            if int(train_mask.sum()) < int(min_train_source_rows) or int(valid_mask.sum()) < int(min_valid_source_rows):
                continue
            train = frame.loc[train_mask].copy()
            valid_source = frame.loc[valid_mask].copy()
            valid = valid_source.reset_index(drop=True)
            valid_metrics = metrics.loc[valid_mask].copy().reset_index(drop=True)
            valid_indices = np.arange(len(valid), dtype=np.int64)

            path_scores: dict[str, pd.Series] = {}
            path_feature_names: dict[str, str] = {}
            for path_name, target_series in path_targets.items():
                score, diag = _score_proxy(
                    train=train,
                    valid=valid_source,
                    features=features,
                    y_train=target_series.loc[train_mask],
                    proxy_top_k=proxy_top_k,
                )
                path_scores[path_name] = score.reset_index(drop=True)
                path_feature_names[path_name] = ",".join(diag.get("proxy_features", []))

            for label_arm in label_arms:
                target = targets[label_arm]
                target_valid = target.loc[valid_mask].copy().reset_index(drop=True)
                label_score, label_diag = _score_proxy(
                    train=train,
                    valid=valid_source,
                    features=features,
                    y_train=target.loc[train_mask, "target_soft"],
                    proxy_top_k=proxy_top_k,
                )
                label_score = label_score.reset_index(drop=True)
                selector_specs = _selector_scores(
                    label_score=label_score,
                    clean_score=path_scores["strict_clean"],
                    quality_score=path_scores["path_quality"],
                    bad_mae_score=path_scores["bad_mae"],
                    dirty_score=path_scores["dirty"],
                    timeout_score=path_scores["timeout"],
                    gate_keep_fracs=gate_keep_fracs,
                    risk_penalties=risk_penalties,
                    combine_label_weight=combine_label_weight,
                )
                feature_summary = (
                    "label="
                    + ",".join(label_diag.get("proxy_features", []))
                    + "; clean="
                    + path_feature_names.get("strict_clean", "")
                    + "; bad_mae="
                    + path_feature_names.get("bad_mae", "")
                    + "; dirty="
                    + path_feature_names.get("dirty", "")
                )
                period_slices = [("month", month, valid_indices)]
                period_slices.extend(("week", week, pos) for week, pos in _slice_week_positions(valid))
                for selector, score in selector_specs:
                    score = _safe_numeric(score).reset_index(drop=True)
                    for period_type, period, pos in period_slices:
                        local_frame = valid.iloc[pos].reset_index(drop=True)
                        local_metrics = valid_metrics.iloc[pos].reset_index(drop=True)
                        local_target = target_valid.iloc[pos].reset_index(drop=True)
                        local_score = score.iloc[pos].reset_index(drop=True)
                        local_label_score = label_score.iloc[pos].reset_index(drop=True)
                        local_clean_score = path_scores["strict_clean"].iloc[pos].reset_index(drop=True)
                        local_risk_score = path_scores["dirty"].iloc[pos].reset_index(drop=True)
                        local_oracle = target_valid["target_soft"].iloc[pos].reset_index(drop=True)
                        for top_frac in top_fracs:
                            rows.append(
                                _score_period(
                                    frame=local_frame,
                                    metrics=local_metrics,
                                    target=local_target,
                                    score=local_score,
                                    oracle_score=local_oracle,
                                    period_type=period_type,
                                    period=str(period),
                                    month=str(month),
                                    source=str(source_name),
                                    selector=str(selector),
                                    label_arm=str(label_arm),
                                    top_frac=float(top_frac),
                                    label_score=local_label_score,
                                    clean_score=local_clean_score,
                                    risk_score=local_risk_score,
                                    selector_features=feature_summary,
                                )
                            )

    period_rows = pd.DataFrame(rows)
    aggregate = _aggregate(period_rows, min_material_selected_rows=min_material_selected_rows)
    fit_holdout = _fit_holdout_summary(
        period_rows,
        fit_months=fit_months,
        holdout_month=holdout_month,
        min_week_rows=min_material_selected_rows,
        min_fit_material_weeks=4,
        min_holdout_material_weeks=2,
        min_fit_positive_week_rate=0.55,
        min_holdout_positive_week_rate=0.50,
    )

    paths = {
        "period_rows": output_dir / "label_adverse_path_proxy_gate_period_rows.csv",
        "aggregate": output_dir / "label_adverse_path_proxy_gate_aggregate.csv",
        "fit_holdout": output_dir / "label_adverse_path_proxy_gate_fit_holdout.csv",
        "manifest": output_dir / "manifest.json",
    }
    period_rows.to_csv(paths["period_rows"], index=False)
    aggregate.to_csv(paths["aggregate"], index=False)
    fit_holdout.to_csv(paths["fit_holdout"], index=False)

    manifest = {
        "scope": "proxy_only_adverse_path_gate_ablation",
        "labels_path": str(labels_path),
        "output_dir": str(output_dir),
        "rows": int(len(frame)),
        "timestamp_min": frame["__ts__"].min(),
        "timestamp_max": frame["__ts__"].max(),
        "symbols": int(frame["__symbol__"].nunique(dropna=True)),
        "feature_dir": str(feature_dir),
        "feature_list_csv": str(feature_list_csv),
        "max_feature_store_features": max_feature_store_features,
        "feature_store": feature_store_report,
        "include_causal_outcome_priors": bool(include_causal_outcome_priors),
        "include_causal_state_path_priors": bool(include_causal_state_path_priors),
        "include_event_confirmation_features": bool(include_event_confirmation_features),
        "include_adverse_path_composites": bool(include_adverse_path_composites),
        "prior_windows_days": [float(v) for v in prior_windows_days],
        "prior_embargo_hours": float(prior_embargo_hours),
        "causal_outcome_priors": outcome_prior_report,
        "causal_state_path_priors": state_path_prior_report,
        "event_confirmation_features": event_confirmation_report,
        "adverse_path_composites": adverse_path_composite_report,
        "feature_count": int(len(features)),
        "label_arms": list(label_arms),
        "sources": list(selected_sources),
        "run_gap_hours": float(run_gap_hours),
        "top_fracs": [float(v) for v in top_fracs],
        "gate_keep_fracs": [float(v) for v in gate_keep_fracs],
        "risk_penalties": [float(v) for v in risk_penalties],
        "proxy_top_k": int(proxy_top_k),
        "combine_label_weight": float(combine_label_weight),
        "min_material_selected_rows": int(min_material_selected_rows),
        "min_train_source_rows": int(min_train_source_rows),
        "min_valid_source_rows": int(min_valid_source_rows),
        "fit_months": [str(value) for value in fit_months],
        "holdout_month": str(holdout_month),
        "months": months,
        "outputs": {key: str(value) for key, value in paths.items()},
    }
    markdown = _write_markdown(output_dir, aggregate, period_rows, fit_holdout, manifest)
    manifest["outputs"]["markdown"] = str(markdown)
    paths["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-path", type=Path, default=DEFAULT_LABELS_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--feature-list-csv", type=Path, default=DEFAULT_FEATURE_LIST_CSV)
    parser.add_argument("--max-feature-store-features", type=int, default=None)
    parser.add_argument("--label-arms", type=lambda value: _parse_csv(value, DEFAULT_LABEL_ARMS), default=",".join(DEFAULT_LABEL_ARMS))
    parser.add_argument("--sources", type=lambda value: _parse_csv(value, DEFAULT_SOURCE_NAMES), default=",".join(DEFAULT_SOURCE_NAMES))
    parser.add_argument("--run-gap-hours", type=float, default=6.0)
    parser.add_argument("--top-fracs", type=lambda value: _parse_float_csv(value, DEFAULT_TOP_FRACS), default=",".join(str(v) for v in DEFAULT_TOP_FRACS))
    parser.add_argument("--gate-keep-fracs", type=lambda value: _parse_float_csv(value, DEFAULT_GATE_KEEP_FRACS), default=",".join(str(v) for v in DEFAULT_GATE_KEEP_FRACS))
    parser.add_argument("--risk-penalties", type=lambda value: _parse_float_csv(value, DEFAULT_RISK_PENALTIES), default=",".join(str(v) for v in DEFAULT_RISK_PENALTIES))
    parser.add_argument("--proxy-top-k", type=int, default=DEFAULT_PROXY_TOP_K)
    parser.add_argument("--combine-label-weight", type=float, default=DEFAULT_COMBINE_LABEL_WEIGHT)
    parser.add_argument("--include-causal-outcome-priors", action="store_true")
    parser.add_argument("--include-causal-state-path-priors", action="store_true")
    parser.add_argument("--include-event-confirmation-features", action="store_true")
    parser.add_argument("--include-adverse-path-composites", action="store_true")
    parser.add_argument("--prior-windows-days", type=lambda value: _parse_float_csv(value, DEFAULT_PRIOR_WINDOWS_DAYS), default=",".join(str(v) for v in DEFAULT_PRIOR_WINDOWS_DAYS))
    parser.add_argument("--prior-embargo-hours", type=float, default=24.0)
    parser.add_argument(
        "--state-path-prior-features",
        type=lambda value: _parse_csv(value, DEFAULT_STATE_PATH_PRIOR_FEATURES),
        default=",".join(DEFAULT_STATE_PATH_PRIOR_FEATURES),
    )
    parser.add_argument(
        "--event-feature-store-features",
        type=lambda value: _parse_csv(value, DEFAULT_EVENT_FEATURE_STORE_FEATURES),
        default=",".join(DEFAULT_EVENT_FEATURE_STORE_FEATURES),
    )
    parser.add_argument("--min-material-selected-rows", type=int, default=5)
    parser.add_argument("--min-train-source-rows", type=int, default=500)
    parser.add_argument("--min-valid-source-rows", type=int, default=100)
    parser.add_argument("--fit-months", type=lambda value: _parse_csv(value, DEFAULT_FIT_MONTHS), default=",".join(DEFAULT_FIT_MONTHS))
    parser.add_argument("--holdout-month", type=str, default=DEFAULT_HOLDOUT_MONTH)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = run_report(
        labels_path=args.labels_path,
        output_dir=args.output_dir,
        feature_dir=args.feature_dir,
        feature_list_csv=args.feature_list_csv,
        max_feature_store_features=args.max_feature_store_features,
        label_arms=list(args.label_arms),
        top_fracs=list(args.top_fracs),
        gate_keep_fracs=list(args.gate_keep_fracs),
        risk_penalties=list(args.risk_penalties),
        proxy_top_k=int(args.proxy_top_k),
        combine_label_weight=float(args.combine_label_weight),
        include_causal_outcome_priors=bool(args.include_causal_outcome_priors),
        include_causal_state_path_priors=bool(args.include_causal_state_path_priors),
        include_event_confirmation_features=bool(args.include_event_confirmation_features),
        include_adverse_path_composites=bool(args.include_adverse_path_composites),
        prior_windows_days=list(args.prior_windows_days),
        prior_embargo_hours=float(args.prior_embargo_hours),
        state_path_prior_features=list(args.state_path_prior_features),
        event_feature_store_features=list(args.event_feature_store_features),
        min_material_selected_rows=int(args.min_material_selected_rows),
        sources=list(args.sources) if args.sources else None,
        run_gap_hours=float(args.run_gap_hours),
        min_train_source_rows=int(args.min_train_source_rows),
        min_valid_source_rows=int(args.min_valid_source_rows),
        fit_months=list(args.fit_months),
        holdout_month=str(args.holdout_month),
    )
    print(
        json.dumps(
            _json_safe(
                {
                    key: value
                    for key, value in manifest.items()
                    if key not in {"feature_store", "causal_state_path_priors"}
                }
            ),
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
