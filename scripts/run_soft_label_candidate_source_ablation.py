#!/usr/bin/env python3
"""Candidate-source split diagnostics for soft labels.

This is a label QA tool, not model training. It keeps the dense candidate
universe fixed, defines decision-time source masks such as quiet continuation
or loud event rows, then tests whether a simple out-of-time feature proxy can
rank profitable labels inside each source.
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
    _read_feature_list,
    _safe_mean,
    _safe_quantile,
    _spearman,
)
from scripts.run_soft_label_economic_proxy_ablation import (  # noqa: E402
    DEFAULT_EVENT_FEATURE_STORE_FEATURES,
    DEFAULT_FIT_MONTHS,
    DEFAULT_HOLDOUT_MONTH,
    DEFAULT_TOP_FRACS,
    _all_targets,
    _event_confirmation_features,
    _fit_holdout_summary,
    _mean_available,
    _monthly_weekly_rows,
    _parse_csv,
    _parse_float_csv,
    _proxy_score,
    _rank_pct,
    _selection_row,
    _xs_rank_or_neutral,
)


DEFAULT_OUTPUT_DIR = Path("data_perp/reports/soft_label_candidate_source_ablation_v1")
DEFAULT_ARMS = ("E9_low_mae_mfe_ratio", "E14_run_entry_low_mae")
DEFAULT_PRIOR_WINDOWS_DAYS = (7.0, 14.0, 30.0)


def _safe_numeric(values: Any) -> pd.Series:
    if isinstance(values, pd.Series):
        return pd.to_numeric(values, errors="coerce")
    return pd.to_numeric(pd.Series(values), errors="coerce")


def _causal_time_edge_prior_features(
    frame: pd.DataFrame,
    metrics: pd.DataFrame,
    *,
    windows_days: list[float],
    embargo_hours: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build decision-time rolling priors for early-edge/no-timeout behavior."""

    ts = pd.to_datetime(frame["__ts__"], utc=True, errors="coerce")
    valid_ts = ts.notna().to_numpy(dtype=bool, copy=False)
    ts_ns = ts.astype("int64").to_numpy(dtype=np.int64, copy=False)
    embargo_ns = int(float(embargo_hours) * 60.0 * 60.0 * 1_000_000_000)

    u = _safe_numeric(metrics["u_policy_net"]).to_numpy(dtype=np.float64, copy=False)
    mae = _safe_numeric(metrics["mae_norm"]).to_numpy(dtype=np.float64, copy=False)
    barrier = _safe_numeric(metrics["barrier"]).to_numpy(dtype=np.float64, copy=False)
    mfe = _safe_numeric(metrics["mfe_norm"]).to_numpy(dtype=np.float64, copy=False)
    bars = _safe_numeric(metrics["bars_to_mfe"]).to_numpy(dtype=np.float64, copy=False)
    timeout = _safe_numeric(metrics["is_timeout"].astype(float)).to_numpy(dtype=np.float64, copy=False)
    mfe_mae = np.divide(
        mfe,
        np.clip(mae, 0.25, None),
        out=np.full_like(mfe, np.nan, dtype=np.float64),
        where=np.isfinite(mfe) & np.isfinite(mae),
    )
    finite_path = (
        np.isfinite(u)
        & np.isfinite(mae)
        & np.isfinite(barrier)
        & np.isfinite(mfe)
        & np.isfinite(bars)
        & np.isfinite(timeout)
    )
    early_edge = np.where(
        finite_path,
        ((u > 0.0) & (mfe >= 1.0) & (bars <= 6.0) & (timeout <= 0.0)).astype(float),
        np.nan,
    )
    early_clean = np.where(
        finite_path & np.isfinite(mfe_mae),
        (
            (u > 0.0)
            & (mae <= 1.0)
            & (barrier <= 0.025)
            & (mfe >= 1.0)
            & (mfe_mae >= 1.25)
            & (bars <= 6.0)
            & (timeout <= 0.0)
        ).astype(float),
        np.nan,
    )
    late_or_timeout = np.where(
        finite_path,
        ((timeout > 0.0) | (bars >= 16.0)).astype(float),
        np.nan,
    )
    fast_bad_mae = np.where(
        finite_path,
        ((mae >= 1.0) & (bars <= 6.0)).astype(float),
        np.nan,
    )
    values = {
        "mean_u": u,
        "timeout": np.where(np.isfinite(timeout), timeout, np.nan),
        "bars_to_mfe": np.where(np.isfinite(bars), bars, np.nan),
        "early_edge": early_edge,
        "early_clean": early_clean,
        "late_or_timeout": late_or_timeout,
        "fast_bad_mae": fast_bad_mae,
    }

    out: dict[str, np.ndarray] = {}

    def fill_scope(scope: str, positions: np.ndarray) -> None:
        positions = positions[valid_ts[positions]]
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
            count_col = f"prior_{scope}_time_edge_count_{window_days:g}d"
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
                col = f"prior_{scope}_time_edge_{metric_name}_{window_days:g}d"
                out.setdefault(col, np.full(len(frame), np.nan, dtype=np.float32))
                out[col][sorted_positions] = means.astype(np.float32)

    all_positions = np.arange(len(frame), dtype=np.int64)
    fill_scope("global", all_positions)
    for _, idx in frame.groupby("__symbol__", sort=False).indices.items():
        fill_scope("symbol", np.asarray(idx, dtype=np.int64))

    priors = pd.DataFrame(out, index=frame.index)
    for scope in ("global", "symbol"):
        for window_days in windows_days:
            suffix = f"{window_days:g}d"
            early_col = f"prior_{scope}_time_edge_early_clean_{suffix}"
            edge_col = f"prior_{scope}_time_edge_early_edge_{suffix}"
            timeout_col = f"prior_{scope}_time_edge_timeout_{suffix}"
            late_col = f"prior_{scope}_time_edge_late_or_timeout_{suffix}"
            if early_col in priors and timeout_col in priors:
                priors[f"prior_{scope}_time_edge_clean_minus_timeout_{suffix}"] = (
                    _safe_numeric(priors[early_col]) - _safe_numeric(priors[timeout_col])
                ).astype(np.float32)
            if edge_col in priors and late_col in priors:
                priors[f"prior_{scope}_time_edge_edge_minus_late_{suffix}"] = (
                    _safe_numeric(priors[edge_col]) - _safe_numeric(priors[late_col])
                ).astype(np.float32)

    finite = priors.notna().mean()
    return priors, {
        "enabled": True,
        "embargo_hours": float(embargo_hours),
        "windows_days": [float(v) for v in windows_days],
        "feature_count": int(priors.shape[1]),
        "mean_finite_frac": float(finite.mean()) if len(finite) else 0.0,
        "min_finite_frac": float(finite.min()) if len(finite) else 0.0,
    }


def _economic_guard_proxy_score(
    *,
    train: pd.DataFrame,
    valid: pd.DataFrame,
    features: list[str],
    target_train: pd.Series,
    train_metrics: pd.DataFrame,
    top_k: int,
) -> tuple[pd.Series, dict[str, Any]]:
    """Select features whose past label direction agrees with past economics."""

    y = _safe_numeric(target_train)
    train_u = _safe_numeric(train_metrics["u_policy_net"])
    train_bad_mae = (_safe_numeric(train_metrics["mae_norm"]) >= 1.0).astype(float)
    rows: list[dict[str, Any]] = []
    for feature in features:
        label_ic = _spearman(train[feature], y)
        if not math.isfinite(label_ic) or abs(float(label_ic)) <= 0.0:
            continue
        orient = 1.0 if float(label_ic) >= 0.0 else -1.0
        utility_ic = _spearman(train[feature], train_u)
        bad_mae_ic = _spearman(train[feature], train_bad_mae)
        if not math.isfinite(utility_ic) or not math.isfinite(bad_mae_ic):
            continue
        oriented_utility_ic = orient * float(utility_ic)
        oriented_bad_mae_ic = orient * float(bad_mae_ic)
        if oriented_utility_ic <= 0.0 or oriented_bad_mae_ic >= 0.0:
            continue
        econ_score = (
            abs(float(label_ic))
            + 0.75 * oriented_utility_ic
            + 0.50 * (-oriented_bad_mae_ic)
        )
        rows.append(
            {
                "feature": feature,
                "ic": float(label_ic),
                "abs_ic": abs(float(label_ic)),
                "utility_ic": float(utility_ic),
                "bad_mae_ic": float(bad_mae_ic),
                "oriented_utility_ic": oriented_utility_ic,
                "oriented_bad_mae_ic": oriented_bad_mae_ic,
                "econ_score": float(econ_score),
            }
        )

    if not rows:
        return pd.Series(np.nan, index=valid.index), {
            "proxy_features": [],
            "proxy_top_abs_ic": float("nan"),
            "proxy_mean_top_abs_ic": float("nan"),
            "proxy_mean_econ_score": float("nan"),
            "proxy_guard_candidates": 0,
        }

    chosen = pd.DataFrame(rows).sort_values("econ_score", ascending=False).head(int(top_k))
    parts: list[pd.Series] = []
    for _, row in chosen.iterrows():
        parts.append(_rank_pct(valid[str(row["feature"])], high_good=float(row["ic"]) >= 0.0).fillna(0.5))
    score = pd.concat(parts, axis=1).mean(axis=1) if parts else pd.Series(np.nan, index=valid.index)
    return score.reindex(valid.index), {
        "proxy_features": chosen["feature"].astype(str).tolist(),
        "proxy_top_abs_ic": float(chosen["abs_ic"].iloc[0]) if len(chosen) else float("nan"),
        "proxy_mean_top_abs_ic": float(chosen["abs_ic"].mean()) if len(chosen) else float("nan"),
        "proxy_mean_econ_score": float(chosen["econ_score"].mean()) if len(chosen) else float("nan"),
        "proxy_guard_candidates": int(len(rows)),
    }


def _time_edge_guard_proxy_score(
    *,
    train: pd.DataFrame,
    valid: pd.DataFrame,
    features: list[str],
    target_train: pd.Series,
    train_metrics: pd.DataFrame,
    top_k: int,
) -> tuple[pd.Series, dict[str, Any]]:
    """Select features whose past direction agrees with faster, non-timeout edge."""

    y = _safe_numeric(target_train)
    train_u = _safe_numeric(train_metrics["u_policy_net"])
    train_bad_mae = (_safe_numeric(train_metrics["mae_norm"]) >= 1.0).astype(float)
    train_timeout = _safe_numeric(train_metrics["is_timeout"].astype(float)).fillna(1.0)
    train_bars = _safe_numeric(train_metrics["bars_to_mfe"]).fillna(24.0)
    train_early_edge = (
        (_safe_numeric(train_metrics["mfe_norm"]) >= 1.0)
        & train_bars.le(6.0)
        & train_timeout.le(0.0)
    ).astype(float)
    rows: list[dict[str, Any]] = []
    for feature in features:
        label_ic = _spearman(train[feature], y)
        if not math.isfinite(label_ic) or abs(float(label_ic)) <= 0.0:
            continue
        orient = 1.0 if float(label_ic) >= 0.0 else -1.0
        utility_ic = _spearman(train[feature], train_u)
        bad_mae_ic = _spearman(train[feature], train_bad_mae)
        timeout_ic = _spearman(train[feature], train_timeout)
        bars_ic = _spearman(train[feature], train_bars)
        early_edge_ic = _spearman(train[feature], train_early_edge)
        if not all(math.isfinite(v) for v in (utility_ic, bad_mae_ic, timeout_ic, bars_ic, early_edge_ic)):
            continue
        oriented_utility_ic = orient * float(utility_ic)
        oriented_bad_mae_ic = orient * float(bad_mae_ic)
        oriented_timeout_ic = orient * float(timeout_ic)
        oriented_bars_ic = orient * float(bars_ic)
        oriented_early_edge_ic = orient * float(early_edge_ic)
        if (
            oriented_utility_ic <= 0.0
            or oriented_bad_mae_ic >= 0.0
            or oriented_timeout_ic >= 0.0
            or (oriented_bars_ic >= 0.0 and oriented_early_edge_ic <= 0.0)
        ):
            continue
        time_edge_score = (
            abs(float(label_ic))
            + 0.75 * oriented_utility_ic
            + 0.50 * (-oriented_bad_mae_ic)
            + 0.85 * (-oriented_timeout_ic)
            + 0.45 * max(0.0, -oriented_bars_ic)
            + 0.45 * max(0.0, oriented_early_edge_ic)
        )
        rows.append(
            {
                "feature": feature,
                "ic": float(label_ic),
                "abs_ic": abs(float(label_ic)),
                "utility_ic": float(utility_ic),
                "bad_mae_ic": float(bad_mae_ic),
                "timeout_ic": float(timeout_ic),
                "bars_ic": float(bars_ic),
                "early_edge_ic": float(early_edge_ic),
                "oriented_utility_ic": oriented_utility_ic,
                "oriented_bad_mae_ic": oriented_bad_mae_ic,
                "oriented_timeout_ic": oriented_timeout_ic,
                "oriented_bars_ic": oriented_bars_ic,
                "oriented_early_edge_ic": oriented_early_edge_ic,
                "time_edge_score": float(time_edge_score),
            }
        )

    if not rows:
        return pd.Series(np.nan, index=valid.index), {
            "proxy_features": [],
            "proxy_top_abs_ic": float("nan"),
            "proxy_mean_top_abs_ic": float("nan"),
            "proxy_mean_time_edge_score": float("nan"),
            "proxy_time_edge_candidates": 0,
        }

    chosen = pd.DataFrame(rows).sort_values("time_edge_score", ascending=False).head(int(top_k))
    parts: list[pd.Series] = []
    for _, row in chosen.iterrows():
        parts.append(_rank_pct(valid[str(row["feature"])], high_good=float(row["ic"]) >= 0.0).fillna(0.5))
    score = pd.concat(parts, axis=1).mean(axis=1) if parts else pd.Series(np.nan, index=valid.index)
    return score.reindex(valid.index), {
        "proxy_features": chosen["feature"].astype(str).tolist(),
        "proxy_top_abs_ic": float(chosen["abs_ic"].iloc[0]) if len(chosen) else float("nan"),
        "proxy_mean_top_abs_ic": float(chosen["abs_ic"].mean()) if len(chosen) else float("nan"),
        "proxy_mean_time_edge_score": float(chosen["time_edge_score"].mean()) if len(chosen) else float("nan"),
        "proxy_time_edge_candidates": int(len(rows)),
        "proxy_mean_train_timeout_ic": _safe_mean(chosen["timeout_ic"]),
        "proxy_mean_train_bars_ic": _safe_mean(chosen["bars_ic"]),
        "proxy_mean_train_early_edge_ic": _safe_mean(chosen["early_edge_ic"]),
    }


def _source_context(frame: pd.DataFrame) -> pd.DataFrame:
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
    oi_location = _mean_available(
        [
            _xs_rank_or_neutral(frame, "oi_up_agree", high_good=True),
            _xs_rank_or_neutral(frame, "oi_chg_2h", high_good=True),
            _xs_rank_or_neutral(frame, "loc_prev_week_range_pos_24", high_good=True),
            _xs_rank_or_neutral(frame, "loc_range_pos_24", high_good=True),
            _safe_numeric(frame.get("event_xs_lo_oiw_pos_delta_entry_dist_1d_atr", 0.5)).fillna(0.5),
            _safe_numeric(frame.get("event_xs_lo_oiw_pos_delta_entry_dist_7d_atr", 0.5)).fillna(0.5),
            _safe_numeric(frame.get("event_xs_lo_oiw_pos_delta_entry_dist_14d_atr", 0.5)).fillna(0.5),
        ],
        frame.index,
    ).fillna(0.5)
    liquidity = _mean_available(
        [
            _safe_numeric(frame.get("event_xs_lo_spread_proxy_hl_range_bps_robust_z", 0.5)).fillna(0.5),
            _safe_numeric(frame.get("event_xs_lo_spread_proxy_abs_return_bps_robust_z", 0.5)).fillna(0.5),
            _safe_numeric(frame.get("event_xs_lo_median_spread_bps", 0.5)).fillna(0.5),
            _xs_rank_or_neutral(frame, "xasset_ob_liquidity_peer_resid", high_good=True),
        ],
        frame.index,
    ).fillna(0.5)
    low_barrier = _mean_available(
        [
            _safe_numeric(frame.get("event_xs_lo_distance_to_support_daily_vwap_atr", 0.5)).fillna(0.5),
            _safe_numeric(frame.get("event_xs_lo_distance_to_resistance_daily_vwap_atr", 0.5)).fillna(0.5),
            _safe_numeric(frame.get("event_xs_lo_up_barrier_pressure_daily_vwap", 0.5)).fillna(0.5),
            _safe_numeric(frame.get("event_xs_lo_down_barrier_pressure_daily_vwap", 0.5)).fillna(0.5),
        ],
        frame.index,
    ).fillna(0.5)
    low_zscore = _xs_rank_or_neutral(frame, "zscore_price_200", high_good=False).fillna(0.5)
    low_atr_compression = _xs_rank_or_neutral(frame, "atr_compression_ratio", high_good=False).fillna(0.5)
    low_range_location = _xs_rank_or_neutral(frame, "loc_range_pos_24", high_good=False).fillna(0.5)
    rebound_context = _mean_available(
        [
            low_zscore,
            low_atr_compression,
            low_range_location,
            oi_location,
            liquidity,
        ],
        frame.index,
    ).fillna(0.5)
    out = pd.DataFrame(index=frame.index)
    out["source_loud_intensity"] = loud_intensity.astype(np.float32)
    out["source_quiet_score"] = (1.0 - loud_intensity).clip(0.0, 1.0).astype(np.float32)
    out["source_oi_location"] = oi_location.astype(np.float32)
    out["source_liquidity"] = liquidity.astype(np.float32)
    out["source_low_barrier"] = low_barrier.astype(np.float32)
    out["source_low_zscore"] = low_zscore.astype(np.float32)
    out["source_low_atr_compression"] = low_atr_compression.astype(np.float32)
    out["source_low_range_location"] = low_range_location.astype(np.float32)
    out["source_rebound_context"] = rebound_context.astype(np.float32)
    out["source_event_quality"] = (
        0.40 * out["source_oi_location"]
        + 0.30 * out["source_liquidity"]
        + 0.30 * out["source_low_barrier"]
    ).clip(0.0, 1.0).astype(np.float32)
    time_edge_prior_quality = _mean_available(
        [
            _xs_rank_or_neutral(frame, "prior_symbol_time_edge_early_clean_30d", high_good=True),
            _xs_rank_or_neutral(frame, "prior_symbol_time_edge_early_edge_30d", high_good=True),
            _xs_rank_or_neutral(frame, "prior_symbol_time_edge_clean_minus_timeout_30d", high_good=True),
            _xs_rank_or_neutral(frame, "prior_symbol_time_edge_edge_minus_late_30d", high_good=True),
            _xs_rank_or_neutral(frame, "prior_symbol_time_edge_timeout_30d", high_good=False),
            _xs_rank_or_neutral(frame, "prior_symbol_time_edge_bars_to_mfe_30d", high_good=False),
            _xs_rank_or_neutral(frame, "prior_global_time_edge_early_clean_14d", high_good=True),
            _xs_rank_or_neutral(frame, "prior_global_time_edge_early_edge_14d", high_good=True),
            _xs_rank_or_neutral(frame, "prior_global_time_edge_clean_minus_timeout_14d", high_good=True),
            _xs_rank_or_neutral(frame, "prior_global_time_edge_edge_minus_late_14d", high_good=True),
            _xs_rank_or_neutral(frame, "prior_global_time_edge_timeout_14d", high_good=False),
            _xs_rank_or_neutral(frame, "prior_global_time_edge_bars_to_mfe_14d", high_good=False),
        ],
        frame.index,
    ).fillna(0.5)
    out["source_time_edge_prior_quality"] = time_edge_prior_quality.astype(np.float32)
    adverse_prior_safety = _mean_available(
        [
            _xs_rank_or_neutral(frame, "prior_symbol_time_edge_fast_bad_mae_30d", high_good=False),
            _xs_rank_or_neutral(frame, "prior_symbol_time_edge_late_or_timeout_30d", high_good=False),
            _xs_rank_or_neutral(frame, "prior_symbol_time_edge_timeout_30d", high_good=False),
            _xs_rank_or_neutral(frame, "prior_symbol_time_edge_clean_minus_timeout_30d", high_good=True),
            _xs_rank_or_neutral(frame, "prior_symbol_time_edge_early_clean_30d", high_good=True),
            _xs_rank_or_neutral(frame, "prior_global_time_edge_fast_bad_mae_14d", high_good=False),
            _xs_rank_or_neutral(frame, "prior_global_time_edge_late_or_timeout_14d", high_good=False),
            _xs_rank_or_neutral(frame, "prior_global_time_edge_timeout_14d", high_good=False),
            _xs_rank_or_neutral(frame, "prior_global_time_edge_clean_minus_timeout_14d", high_good=True),
            _xs_rank_or_neutral(frame, "prior_global_time_edge_early_clean_14d", high_good=True),
        ],
        frame.index,
    ).fillna(0.5)
    out["source_adverse_prior_safety"] = adverse_prior_safety.astype(np.float32)
    out["source_dual_prior_quality"] = (
        0.50 * out["source_time_edge_prior_quality"]
        + 0.50 * out["source_adverse_prior_safety"]
    ).clip(0.0, 1.0).astype(np.float32)
    return out


def _run_entry_mask(
    frame: pd.DataFrame,
    base_mask: pd.Series,
    *,
    gap_hours: float,
) -> pd.Series:
    mask = base_mask.fillna(False).astype(bool).reindex(frame.index, fill_value=False)
    out = pd.Series(False, index=frame.index)
    work = pd.DataFrame(
        {
            "__ts__": pd.to_datetime(frame["__ts__"], errors="coerce"),
            "__symbol__": frame["__symbol__"].astype(str),
            "__mask__": mask.to_numpy(dtype=bool, copy=False),
            "__idx__": np.arange(len(frame), dtype=np.int64),
        },
        index=frame.index,
    ).sort_values(["__symbol__", "__ts__"], kind="mergesort")
    gap = pd.Timedelta(hours=float(gap_hours))
    for _, group in work.groupby("__symbol__", sort=False):
        prev_ts: pd.Timestamp | None = None
        prev_active = False
        for _, row in group.iterrows():
            active = bool(row["__mask__"])
            ts = row["__ts__"]
            if active and (not prev_active or prev_ts is None or pd.isna(ts) or ts - prev_ts > gap):
                out.iloc[int(row["__idx__"])] = True
            prev_active = active
            if active and pd.notna(ts):
                prev_ts = ts
            elif not active:
                prev_ts = None
    return out


def _build_sources(
    frame: pd.DataFrame,
    context: pd.DataFrame,
    *,
    run_gap_hours: float,
) -> dict[str, pd.Series]:
    loud = context["source_loud_intensity"]
    quiet = context["source_quiet_score"]
    oi = context["source_oi_location"]
    liq = context["source_liquidity"]
    low_barrier = context["source_low_barrier"]
    event_quality = context["source_event_quality"]
    time_edge_prior_quality = context["source_time_edge_prior_quality"]
    adverse_prior_safety = context["source_adverse_prior_safety"]
    dual_prior_quality = context["source_dual_prior_quality"]
    low_zscore = context["source_low_zscore"]
    low_atr_compression = context["source_low_atr_compression"]
    rebound_context = context["source_rebound_context"]

    quiet_loose = quiet >= 0.30
    quiet_mid = quiet >= 0.38
    quiet_strict = quiet >= 0.45
    quiet_oi = quiet_mid & (oi >= 0.55)
    quiet_quality = quiet_mid & (event_quality >= 0.54)
    loud_event = loud >= 0.72
    loud_quality = loud_event & (liq >= 0.55) & (low_barrier >= 0.52)
    any_event_quality = ((loud >= 0.62) | (oi >= 0.62)) & (event_quality >= 0.52)
    low_zscore_rebound = low_zscore >= 0.67
    low_atr_rebound = low_atr_compression >= 0.67
    rebound_mid = rebound_context >= 0.58
    rebound_strict = rebound_context >= 0.64
    rebound_event_quality = rebound_mid & (((loud >= 0.55) | (oi >= 0.55)) & (event_quality >= 0.50))
    lowz_event_quality = low_zscore_rebound & (((loud >= 0.55) | (oi >= 0.55)) & (event_quality >= 0.50))
    event_confirmation = _safe_numeric(frame.get("event_confirmation", 0.5)).fillna(0.5).clip(0.0, 1.0)
    event_impulse = _safe_numeric(frame.get("event_impulse", 0.5)).fillna(0.5).clip(0.0, 1.0)
    event_liquidity_quality = _safe_numeric(frame.get("event_liquidity_quality", 0.5)).fillna(0.5).clip(0.0, 1.0)
    event_low_barrier_context = _safe_numeric(frame.get("event_low_barrier_context", 0.5)).fillna(0.5).clip(0.0, 1.0)
    event_confirmed_impulse = _safe_numeric(frame.get("event_confirmed_impulse", 0.25)).fillna(0.25).clip(0.0, 1.0)
    event_lowbarrier_confirmed = _safe_numeric(frame.get("event_lowbarrier_confirmed", 0.25)).fillna(0.25).clip(0.0, 1.0)
    event_clean_breakout_context = (
        _safe_numeric(frame.get("event_clean_breakout_context", 0.25)).fillna(0.25).clip(0.0, 1.0)
    )
    event_confirmed_liquid_impulse = (
        _safe_numeric(frame.get("event_confirmed_liquid_impulse", 0.125)).fillna(0.125).clip(0.0, 1.0)
    )
    confirmed_lowbarrier_quality = (
        (event_confirmation >= 0.55)
        & (event_low_barrier_context >= 0.55)
        & (event_liquidity_quality >= 0.50)
    )
    confirmed_impulse_lowbarrier = (
        (event_confirmed_impulse >= 0.32)
        & (event_lowbarrier_confirmed >= 0.30)
        & (liq >= 0.50)
    )
    clean_breakout_quality = (
        (event_clean_breakout_context >= 0.30)
        & (event_confirmed_liquid_impulse >= 0.20)
        & (event_impulse >= 0.50)
    )
    quiet_confirmed_lowbarrier = quiet_mid & confirmed_lowbarrier_quality
    time_edge_prior_confirmed_lowbarrier_quality = confirmed_lowbarrier_quality & (time_edge_prior_quality >= 0.58)
    time_edge_prior_confirmed_impulse_lowbarrier = confirmed_impulse_lowbarrier & (time_edge_prior_quality >= 0.58)
    time_edge_prior_clean_breakout_quality = clean_breakout_quality & (time_edge_prior_quality >= 0.58)
    time_edge_prior_quiet_confirmed_lowbarrier = quiet_confirmed_lowbarrier & (time_edge_prior_quality >= 0.58)
    confirmed_lowbarrier_family = confirmed_lowbarrier_quality | quiet_confirmed_lowbarrier
    confirmed_lowbarrier_impulse_family = (
        confirmed_lowbarrier_quality | confirmed_impulse_lowbarrier | quiet_confirmed_lowbarrier
    )
    confirmed_event_quality_family = (
        confirmed_lowbarrier_quality
        | confirmed_impulse_lowbarrier
        | clean_breakout_quality
        | quiet_confirmed_lowbarrier
    )
    time_edge_prior_confirmed_family = (
        time_edge_prior_confirmed_lowbarrier_quality
        | time_edge_prior_confirmed_impulse_lowbarrier
        | time_edge_prior_clean_breakout_quality
        | time_edge_prior_quiet_confirmed_lowbarrier
    )
    dual_prior_confirmed_lowbarrier_quality = (
        confirmed_lowbarrier_quality
        & (time_edge_prior_quality >= 0.56)
        & (adverse_prior_safety >= 0.56)
        & (dual_prior_quality >= 0.58)
    )
    dual_prior_confirmed_impulse_lowbarrier = (
        confirmed_impulse_lowbarrier
        & (time_edge_prior_quality >= 0.56)
        & (adverse_prior_safety >= 0.56)
        & (dual_prior_quality >= 0.58)
    )
    dual_prior_clean_breakout_quality = (
        clean_breakout_quality
        & (time_edge_prior_quality >= 0.56)
        & (adverse_prior_safety >= 0.56)
        & (dual_prior_quality >= 0.58)
    )
    dual_prior_quiet_confirmed_lowbarrier = (
        quiet_confirmed_lowbarrier
        & (time_edge_prior_quality >= 0.56)
        & (adverse_prior_safety >= 0.56)
        & (dual_prior_quality >= 0.58)
    )
    strict_dual_prior_confirmed_lowbarrier_quality = dual_prior_confirmed_lowbarrier_quality & (dual_prior_quality >= 0.62)
    strict_dual_prior_confirmed_impulse_lowbarrier = dual_prior_confirmed_impulse_lowbarrier & (dual_prior_quality >= 0.62)

    sources: dict[str, pd.Series] = {
        "all": pd.Series(True, index=frame.index),
        "quiet_loose": quiet_loose,
        "quiet_mid": quiet_mid,
        "quiet_strict": quiet_strict,
        "quiet_oi": quiet_oi,
        "quiet_quality": quiet_quality,
        "loud_event": loud_event,
        "loud_quality": loud_quality,
        "any_event_quality": any_event_quality,
        "low_zscore_rebound": low_zscore_rebound,
        "low_atr_rebound": low_atr_rebound,
        "rebound_mid": rebound_mid,
        "rebound_strict": rebound_strict,
        "rebound_event_quality": rebound_event_quality,
        "lowz_event_quality": lowz_event_quality,
        "confirmed_lowbarrier_quality": confirmed_lowbarrier_quality,
        "confirmed_impulse_lowbarrier": confirmed_impulse_lowbarrier,
        "clean_breakout_quality": clean_breakout_quality,
        "quiet_confirmed_lowbarrier": quiet_confirmed_lowbarrier,
        "confirmed_lowbarrier_family": confirmed_lowbarrier_family,
        "confirmed_lowbarrier_impulse_family": confirmed_lowbarrier_impulse_family,
        "confirmed_event_quality_family": confirmed_event_quality_family,
        "time_edge_prior_confirmed_lowbarrier_quality": time_edge_prior_confirmed_lowbarrier_quality,
        "time_edge_prior_confirmed_impulse_lowbarrier": time_edge_prior_confirmed_impulse_lowbarrier,
        "time_edge_prior_clean_breakout_quality": time_edge_prior_clean_breakout_quality,
        "time_edge_prior_quiet_confirmed_lowbarrier": time_edge_prior_quiet_confirmed_lowbarrier,
        "time_edge_prior_confirmed_family": time_edge_prior_confirmed_family,
        "dual_prior_confirmed_lowbarrier_quality": dual_prior_confirmed_lowbarrier_quality,
        "dual_prior_confirmed_impulse_lowbarrier": dual_prior_confirmed_impulse_lowbarrier,
        "dual_prior_clean_breakout_quality": dual_prior_clean_breakout_quality,
        "dual_prior_quiet_confirmed_lowbarrier": dual_prior_quiet_confirmed_lowbarrier,
        "strict_dual_prior_confirmed_lowbarrier_quality": strict_dual_prior_confirmed_lowbarrier_quality,
        "strict_dual_prior_confirmed_impulse_lowbarrier": strict_dual_prior_confirmed_impulse_lowbarrier,
    }
    sources["run_entry_quiet_mid"] = _run_entry_mask(frame, quiet_mid, gap_hours=run_gap_hours)
    sources["run_entry_quiet_quality"] = _run_entry_mask(frame, quiet_quality, gap_hours=run_gap_hours)
    sources["run_entry_loud_event"] = _run_entry_mask(frame, loud_event, gap_hours=run_gap_hours)
    sources["run_entry_any_event_quality"] = _run_entry_mask(frame, any_event_quality, gap_hours=run_gap_hours)
    sources["run_entry_rebound_mid"] = _run_entry_mask(frame, rebound_mid, gap_hours=run_gap_hours)
    sources["run_entry_rebound_event_quality"] = _run_entry_mask(
        frame,
        rebound_event_quality,
        gap_hours=run_gap_hours,
    )
    sources["run_entry_confirmed_lowbarrier_quality"] = _run_entry_mask(
        frame,
        confirmed_lowbarrier_quality,
        gap_hours=run_gap_hours,
    )
    sources["run_entry_confirmed_impulse_lowbarrier"] = _run_entry_mask(
        frame,
        confirmed_impulse_lowbarrier,
        gap_hours=run_gap_hours,
    )
    sources["run_entry_clean_breakout_quality"] = _run_entry_mask(
        frame,
        clean_breakout_quality,
        gap_hours=run_gap_hours,
    )
    sources["run_entry_quiet_confirmed_lowbarrier"] = _run_entry_mask(
        frame,
        quiet_confirmed_lowbarrier,
        gap_hours=run_gap_hours,
    )
    sources["run_entry_confirmed_lowbarrier_family"] = _run_entry_mask(
        frame,
        confirmed_lowbarrier_family,
        gap_hours=run_gap_hours,
    )
    sources["run_entry_confirmed_lowbarrier_impulse_family"] = _run_entry_mask(
        frame,
        confirmed_lowbarrier_impulse_family,
        gap_hours=run_gap_hours,
    )
    sources["run_entry_confirmed_event_quality_family"] = _run_entry_mask(
        frame,
        confirmed_event_quality_family,
        gap_hours=run_gap_hours,
    )
    sources["run_entry_time_edge_prior_confirmed_lowbarrier_quality"] = _run_entry_mask(
        frame,
        time_edge_prior_confirmed_lowbarrier_quality,
        gap_hours=run_gap_hours,
    )
    sources["run_entry_time_edge_prior_confirmed_impulse_lowbarrier"] = _run_entry_mask(
        frame,
        time_edge_prior_confirmed_impulse_lowbarrier,
        gap_hours=run_gap_hours,
    )
    sources["run_entry_time_edge_prior_clean_breakout_quality"] = _run_entry_mask(
        frame,
        time_edge_prior_clean_breakout_quality,
        gap_hours=run_gap_hours,
    )
    sources["run_entry_time_edge_prior_quiet_confirmed_lowbarrier"] = _run_entry_mask(
        frame,
        time_edge_prior_quiet_confirmed_lowbarrier,
        gap_hours=run_gap_hours,
    )
    sources["run_entry_time_edge_prior_confirmed_family"] = _run_entry_mask(
        frame,
        time_edge_prior_confirmed_family,
        gap_hours=run_gap_hours,
    )
    sources["run_entry_dual_prior_confirmed_lowbarrier_quality"] = _run_entry_mask(
        frame,
        dual_prior_confirmed_lowbarrier_quality,
        gap_hours=run_gap_hours,
    )
    sources["run_entry_dual_prior_confirmed_impulse_lowbarrier"] = _run_entry_mask(
        frame,
        dual_prior_confirmed_impulse_lowbarrier,
        gap_hours=run_gap_hours,
    )
    sources["run_entry_dual_prior_clean_breakout_quality"] = _run_entry_mask(
        frame,
        dual_prior_clean_breakout_quality,
        gap_hours=run_gap_hours,
    )
    sources["run_entry_dual_prior_quiet_confirmed_lowbarrier"] = _run_entry_mask(
        frame,
        dual_prior_quiet_confirmed_lowbarrier,
        gap_hours=run_gap_hours,
    )
    sources["run_entry_strict_dual_prior_confirmed_lowbarrier_quality"] = _run_entry_mask(
        frame,
        strict_dual_prior_confirmed_lowbarrier_quality,
        gap_hours=run_gap_hours,
    )
    sources["run_entry_strict_dual_prior_confirmed_impulse_lowbarrier"] = _run_entry_mask(
        frame,
        strict_dual_prior_confirmed_impulse_lowbarrier,
        gap_hours=run_gap_hours,
    )
    return {name: mask.fillna(False).astype(bool) for name, mask in sources.items()}


def _source_summary(
    *,
    frame: pd.DataFrame,
    metrics: pd.DataFrame,
    context: pd.DataFrame,
    sources: dict[str, pd.Series],
) -> pd.DataFrame:
    month = frame["__ts__"].dt.to_period("M").astype(str)
    rows: list[dict[str, Any]] = []
    for source, mask in sources.items():
        selected = metrics.loc[mask]
        selected_frame = frame.loc[mask]
        selected_context = context.loc[mask]
        monthly_counts = selected_frame.groupby(month.loc[mask], dropna=False).size().to_dict()
        rows.append(
            {
                "source": source,
                "rows": int(mask.sum()),
                "row_frac": float(mask.mean()),
                "symbols": int(selected_frame["__symbol__"].nunique()) if len(selected_frame) else 0,
                "mean_u": _safe_mean(selected.get("u_policy_net")),
                "hit_u": _safe_mean(selected.get("u_policy_net") > 0.0) if len(selected) else float("nan"),
                "bad_mae_1r_rate": _safe_mean(selected.get("mae_norm") >= 1.0) if len(selected) else float("nan"),
                "wide_25bps_rate": _safe_mean(selected.get("barrier") > 0.025) if len(selected) else float("nan"),
                "timeout_rate": _safe_mean(selected.get("is_timeout").astype(float)) if len(selected) else float("nan"),
                "mean_loud_intensity": _safe_mean(selected_context.get("source_loud_intensity")),
                "mean_oi_location": _safe_mean(selected_context.get("source_oi_location")),
                "mean_event_quality": _safe_mean(selected_context.get("source_event_quality")),
                "mean_time_edge_prior_quality": _safe_mean(selected_context.get("source_time_edge_prior_quality")),
                "mean_adverse_prior_safety": _safe_mean(selected_context.get("source_adverse_prior_safety")),
                "mean_dual_prior_quality": _safe_mean(selected_context.get("source_dual_prior_quality")),
                "mean_low_zscore": _safe_mean(selected_context.get("source_low_zscore")),
                "mean_low_atr_compression": _safe_mean(selected_context.get("source_low_atr_compression")),
                "mean_rebound_context": _safe_mean(selected_context.get("source_rebound_context")),
                "rows_2026_04": int(monthly_counts.get("2026-04", 0)),
                "rows_2026_05": int(monthly_counts.get("2026-05", 0)),
                "rows_2026_06": int(monthly_counts.get("2026-06", 0)),
            }
        )
    return pd.DataFrame(rows).sort_values(["rows"], ascending=False)


def _source_fit_holdout_summary(
    *,
    monthly: pd.DataFrame,
    weekly: pd.DataFrame,
    fit_months: list[str],
    holdout_month: str,
    min_week_rows: int,
) -> pd.DataFrame:
    if monthly.empty:
        return pd.DataFrame()
    parts: list[pd.DataFrame] = []
    for source in sorted(monthly["source"].astype(str).unique()):
        monthly_source = monthly[monthly["source"].astype(str).eq(source)].copy()
        weekly_source = weekly[weekly["source"].astype(str).eq(source)].copy()
        summary = _fit_holdout_summary(
            monthly=monthly_source,
            weekly=weekly_source,
            fit_months=fit_months,
            holdout_month=holdout_month,
            min_week_rows=min_week_rows,
            min_fit_material_weeks=4,
            min_holdout_material_weeks=2,
            min_fit_positive_week_rate=0.55,
            min_holdout_positive_week_rate=0.50,
        )
        if summary.empty:
            continue
        summary.insert(0, "source", source)
        parts.append(summary)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


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
    *,
    output_dir: Path,
    source_summary: pd.DataFrame,
    fit_holdout: pd.DataFrame,
    proxy_ic: pd.DataFrame,
    manifest: dict[str, Any],
) -> Path:
    path = output_dir / "soft_label_candidate_source_ablation.md"
    counts = (
        fit_holdout.groupby(["source", "selector"], observed=True)
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
    proxy_best = (
        fit_holdout[fit_holdout["selector"].eq("feature_ic_proxy")]
        .sort_values(
            ["holdout_clean_pass", "holdout_bounded_pass", "holdout_mean_month_u", "path_risk_score"],
            ascending=[False, False, False, False],
        )
        if not fit_holdout.empty
        else pd.DataFrame()
    )
    econ_best = (
        fit_holdout[fit_holdout["selector"].eq("economic_guard_proxy")]
        .sort_values(
            ["holdout_clean_pass", "holdout_bounded_pass", "holdout_mean_month_u", "path_risk_score"],
            ascending=[False, False, False, False],
        )
        if not fit_holdout.empty
        else pd.DataFrame()
    )
    time_edge_best = (
        fit_holdout[fit_holdout["selector"].eq("time_edge_guard_proxy")]
        .sort_values(
            ["holdout_clean_pass", "holdout_bounded_pass", "holdout_mean_month_u", "path_risk_score"],
            ascending=[False, False, False, False],
        )
        if not fit_holdout.empty
        else pd.DataFrame()
    )
    oracle_best = (
        fit_holdout[fit_holdout["selector"].eq("oracle_label_sort")]
        .sort_values(
            ["holdout_clean_pass", "holdout_bounded_pass", "holdout_mean_month_u", "path_risk_score"],
            ascending=[False, False, False, False],
        )
        if not fit_holdout.empty
        else pd.DataFrame()
    )
    lines = [
        "# Soft Label Candidate Source Ablation",
        "",
        "Scope: fixed dense candidate universe, source masks from decision-time features, no model training.",
        "",
        f"Fit months: `{','.join(manifest['fit_months'])}`. Holdout month: `{manifest['holdout_month']}`.",
        f"Proxy top-k: `{manifest['proxy_top_k']}`. Run-entry gap hours: `{manifest['run_gap_hours']}`.",
        "",
        "## Source Coverage",
        "",
        _table(
            source_summary,
            [
                "source",
                "rows",
                "rows_2026_04",
                "rows_2026_05",
                "rows_2026_06",
                "mean_u",
                "hit_u",
                "bad_mae_1r_rate",
                "mean_loud_intensity",
                "mean_event_quality",
                "mean_time_edge_prior_quality",
                "mean_adverse_prior_safety",
                "mean_dual_prior_quality",
                "mean_low_zscore",
                "mean_rebound_context",
            ],
            limit=40,
        ),
        "",
        "## Gate Counts",
        "",
        _table(counts, ["source", "selector", "rows", "fit_clean", "holdout_clean", "fit_bounded", "holdout_bounded", "positive_dirty"], limit=80),
        "",
        "## Best Feature-Proxy Source Rows",
        "",
        _table(
            proxy_best,
            [
                "source",
                "label_arm",
                "top_frac",
                "fit_sign_pass",
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
            limit=40,
        ),
        "",
        "## Best Economic-Guard Proxy Source Rows",
        "",
        _table(
            econ_best,
            [
                "source",
                "label_arm",
                "top_frac",
                "fit_sign_pass",
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
            limit=40,
        ),
        "",
        "## Best Time-Edge Guard Proxy Source Rows",
        "",
        _table(
            time_edge_best,
            [
                "source",
                "label_arm",
                "top_frac",
                "fit_sign_pass",
                "holdout_sign_pass",
                "holdout_clean_pass",
                "holdout_bounded_pass",
                "positive_dirty_holdout",
                "fit_mean_month_u",
                "holdout_mean_month_u",
                "holdout_bad_mae_1r_rate",
                "holdout_p90_mae_norm",
                "holdout_timeout_rate",
                "holdout_mean_mfe_mae_ratio",
                "holdout_material_positive_week_rate",
                "path_risk_score",
            ],
            limit=40,
        ),
        "",
        "## Best Oracle Source Rows",
        "",
        _table(
            oracle_best,
            [
                "source",
                "label_arm",
                "top_frac",
                "holdout_clean_pass",
                "holdout_bounded_pass",
                "fit_mean_month_u",
                "holdout_mean_month_u",
                "holdout_bad_mae_1r_rate",
                "holdout_p90_mae_norm",
                "holdout_mean_mfe_mae_ratio",
                "path_risk_score",
            ],
            limit=40,
        ),
        "",
        "## June Proxy ICs",
        "",
        _table(
            proxy_ic[proxy_ic["period"].astype(str).eq(str(manifest["holdout_month"]))].sort_values("oos_ic_u", ascending=False),
            [
                "selector",
                "source",
                "label_arm",
                "oos_ic_target",
                "oos_ic_u",
                "oos_ic_bad_mae",
                "oos_ic_timeout",
                "oos_ic_bars_to_mfe",
                "proxy_features",
            ],
            limit=80,
        ),
        "",
        "## Outputs",
        "",
        f"- Source summary: `{manifest['outputs']['source_summary']}`",
        f"- Monthly selection: `{manifest['outputs']['monthly']}`",
        f"- Weekly selection: `{manifest['outputs']['weekly']}`",
        f"- Proxy IC: `{manifest['outputs']['proxy_ic']}`",
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
    top_fracs: list[float],
    fit_months: list[str],
    holdout_month: str,
    min_week_rows: int,
    arms: list[str],
    sources: list[str] | None,
    event_feature_store_features: list[str],
    run_gap_hours: float,
    min_train_source_rows: int,
    min_valid_source_rows: int,
    include_causal_time_edge_priors: bool,
    prior_windows_days: list[float],
    prior_embargo_hours: float,
    include_economic_guard_proxy: bool,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = _load_labels(labels_path)
    selected_features = _read_feature_list(feature_list_csv, max_features=max_feature_store_features)
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

    event_features, event_report = _event_confirmation_features(
        frame,
        event_features=event_feature_store_features,
    )
    if not event_features.empty:
        frame = pd.concat([frame, event_features.astype(np.float32, copy=False)], axis=1).copy()
    metrics = _path_metrics(frame)
    time_edge_prior_report: dict[str, Any] = {"enabled": False}
    if include_causal_time_edge_priors:
        time_edge_prior_features, time_edge_prior_report = _causal_time_edge_prior_features(
            frame,
            metrics,
            windows_days=prior_windows_days,
            embargo_hours=prior_embargo_hours,
        )
        frame = pd.concat([frame, time_edge_prior_features.astype(np.float32, copy=False)], axis=1).copy()
    context = _source_context(frame)
    frame = pd.concat([frame, context.astype(np.float32, copy=False)], axis=1).copy()
    targets, descriptions = _all_targets(frame, metrics)
    missing_arms = sorted(set(arms) - set(targets))
    if missing_arms:
        raise ValueError(f"Unknown arm(s): {missing_arms}")
    targets = {arm: targets[arm] for arm in arms}
    descriptions = {arm: descriptions.get(arm, "") for arm in arms}

    source_masks = _build_sources(frame, context, run_gap_hours=run_gap_hours)
    if sources:
        missing_sources = sorted(set(sources) - set(source_masks))
        if missing_sources:
            raise ValueError(f"Unknown source(s): {missing_sources}")
        source_masks = {source: source_masks[source] for source in sources}

    features = _feature_columns(frame)
    month_series = frame["__ts__"].dt.to_period("M").astype(str)
    months = sorted(month_series.dropna().unique())
    monthly_rows: list[dict[str, Any]] = []
    weekly_rows: list[dict[str, Any]] = []
    proxy_ic_rows: list[dict[str, Any]] = []

    for source_name, source_mask in source_masks.items():
        for month in months[1:]:
            train_mask = month_series.lt(str(month)) & source_mask
            valid_mask = month_series.eq(str(month)) & source_mask
            if int(train_mask.sum()) < int(min_train_source_rows) or int(valid_mask.sum()) < int(min_valid_source_rows):
                continue
            train = frame.loc[train_mask].copy()
            valid = frame.loc[valid_mask].copy()
            train_metrics = metrics.loc[train_mask].copy()
            valid_metrics = metrics.loc[valid_mask].copy()
            for label_arm, target in targets.items():
                combined_arm = f"{source_name}::{label_arm}"
                train_target = target.loc[train_mask, "target_soft"]
                valid_target = target.loc[valid_mask].copy()
                proxy_score, proxy_diag = _proxy_score(
                    train=train,
                    valid=valid,
                    features=features,
                    target_train=train_target,
                    top_k=proxy_top_k,
                )
                selector_scores: list[tuple[str, pd.Series, dict[str, Any]]] = [
                    (
                        "oracle_label_sort",
                        valid_target["target_soft"],
                        {
                            "oos_ic_target": 1.0,
                            "oos_ic_u": _spearman(valid_target["target_soft"], valid_metrics["u_policy_net"]),
                            "oos_ic_bad_mae": _spearman(valid_target["target_soft"], (valid_metrics["mae_norm"] >= 1.0).astype(float)),
                            "oos_ic_timeout": _spearman(valid_target["target_soft"], valid_metrics["is_timeout"].astype(float)),
                            "oos_ic_bars_to_mfe": _spearman(valid_target["target_soft"], valid_metrics["bars_to_mfe"]),
                            "proxy_top_abs_ic": float("nan"),
                            "proxy_mean_top_abs_ic": float("nan"),
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
                            "oos_ic_timeout": _spearman(proxy_score, valid_metrics["is_timeout"].astype(float)),
                            "oos_ic_bars_to_mfe": _spearman(proxy_score, valid_metrics["bars_to_mfe"]),
                            "proxy_top_abs_ic": proxy_diag.get("proxy_top_abs_ic"),
                            "proxy_mean_top_abs_ic": proxy_diag.get("proxy_mean_top_abs_ic"),
                            "proxy_features": ",".join(proxy_diag.get("proxy_features", [])),
                        },
                    ),
                ]
                if include_economic_guard_proxy:
                    econ_score, econ_diag = _economic_guard_proxy_score(
                        train=train,
                        valid=valid,
                        features=features,
                        target_train=train_target,
                        train_metrics=train_metrics,
                        top_k=proxy_top_k,
                    )
                    selector_scores.append(
                        (
                            "economic_guard_proxy",
                            econ_score,
                            {
                                "oos_ic_target": _spearman(econ_score, valid_target["target_soft"]),
                                "oos_ic_u": _spearman(econ_score, valid_metrics["u_policy_net"]),
                                "oos_ic_bad_mae": _spearman(econ_score, (valid_metrics["mae_norm"] >= 1.0).astype(float)),
                                "oos_ic_timeout": _spearman(econ_score, valid_metrics["is_timeout"].astype(float)),
                                "oos_ic_bars_to_mfe": _spearman(econ_score, valid_metrics["bars_to_mfe"]),
                                "proxy_top_abs_ic": econ_diag.get("proxy_top_abs_ic"),
                                "proxy_mean_top_abs_ic": econ_diag.get("proxy_mean_top_abs_ic"),
                                "proxy_features": ",".join(econ_diag.get("proxy_features", [])),
                                "proxy_mean_econ_score": econ_diag.get("proxy_mean_econ_score"),
                                "proxy_guard_candidates": econ_diag.get("proxy_guard_candidates"),
                            },
                        )
                    )
                    time_score, time_diag = _time_edge_guard_proxy_score(
                        train=train,
                        valid=valid,
                        features=features,
                        target_train=train_target,
                        train_metrics=train_metrics,
                        top_k=proxy_top_k,
                    )
                    selector_scores.append(
                        (
                            "time_edge_guard_proxy",
                            time_score,
                            {
                                "oos_ic_target": _spearman(time_score, valid_target["target_soft"]),
                                "oos_ic_u": _spearman(time_score, valid_metrics["u_policy_net"]),
                                "oos_ic_bad_mae": _spearman(time_score, (valid_metrics["mae_norm"] >= 1.0).astype(float)),
                                "oos_ic_timeout": _spearman(time_score, valid_metrics["is_timeout"].astype(float)),
                                "oos_ic_bars_to_mfe": _spearman(time_score, valid_metrics["bars_to_mfe"]),
                                "proxy_top_abs_ic": time_diag.get("proxy_top_abs_ic"),
                                "proxy_mean_top_abs_ic": time_diag.get("proxy_mean_top_abs_ic"),
                                "proxy_features": ",".join(time_diag.get("proxy_features", [])),
                                "proxy_mean_time_edge_score": time_diag.get("proxy_mean_time_edge_score"),
                                "proxy_time_edge_candidates": time_diag.get("proxy_time_edge_candidates"),
                                "proxy_mean_train_timeout_ic": time_diag.get("proxy_mean_train_timeout_ic"),
                                "proxy_mean_train_bars_ic": time_diag.get("proxy_mean_train_bars_ic"),
                                "proxy_mean_train_early_edge_ic": time_diag.get("proxy_mean_train_early_edge_ic"),
                            },
                        )
                    )

                for selector, score, diag in selector_scores:
                    proxy_ic_rows.append(
                        {
                            "selector": selector,
                            "source": source_name,
                            "arm": combined_arm,
                            "label_arm": label_arm,
                            "description": descriptions.get(label_arm, ""),
                            "period": str(month),
                            "train_rows": int(train_mask.sum()),
                            "valid_rows": int(valid_mask.sum()),
                            "oos_ic_target": diag.get("oos_ic_target"),
                            "oos_ic_u": diag.get("oos_ic_u"),
                            "oos_ic_bad_mae": diag.get("oos_ic_bad_mae"),
                            "oos_ic_timeout": diag.get("oos_ic_timeout"),
                            "oos_ic_bars_to_mfe": diag.get("oos_ic_bars_to_mfe"),
                            "proxy_top_abs_ic": diag.get("proxy_top_abs_ic"),
                            "proxy_mean_top_abs_ic": diag.get("proxy_mean_top_abs_ic"),
                            "proxy_features": diag.get("proxy_features", ""),
                            "proxy_mean_econ_score": diag.get("proxy_mean_econ_score", float("nan")),
                            "proxy_guard_candidates": diag.get("proxy_guard_candidates", float("nan")),
                            "proxy_mean_time_edge_score": diag.get("proxy_mean_time_edge_score", float("nan")),
                            "proxy_time_edge_candidates": diag.get("proxy_time_edge_candidates", float("nan")),
                            "proxy_mean_train_timeout_ic": diag.get("proxy_mean_train_timeout_ic", float("nan")),
                            "proxy_mean_train_bars_ic": diag.get("proxy_mean_train_bars_ic", float("nan")),
                            "proxy_mean_train_early_edge_ic": diag.get("proxy_mean_train_early_edge_ic", float("nan")),
                        }
                    )
                    m_rows, w_rows = _monthly_weekly_rows(
                        valid_frame=valid,
                        valid_metrics=valid_metrics,
                        valid_target=valid_target,
                        score=score,
                        arm=combined_arm,
                        selector=selector,
                        month=str(month),
                        top_fracs=top_fracs,
                        diag=diag,
                    )
                    for row in m_rows:
                        row["source"] = source_name
                        row["label_arm"] = label_arm
                        row["source_train_rows"] = int(train_mask.sum())
                        row["source_valid_rows"] = int(valid_mask.sum())
                    for row in w_rows:
                        row["source"] = source_name
                        row["label_arm"] = label_arm
                        row["source_train_rows"] = int(train_mask.sum())
                        row["source_valid_rows"] = int(valid_mask.sum())
                    monthly_rows.extend(m_rows)
                    weekly_rows.extend(w_rows)

    monthly = pd.DataFrame(monthly_rows)
    weekly = pd.DataFrame(weekly_rows)
    proxy_ic = pd.DataFrame(proxy_ic_rows)
    source_summary = _source_summary(
        frame=frame,
        metrics=metrics,
        context=context,
        sources=source_masks,
    )
    fit_holdout = _source_fit_holdout_summary(
        monthly=monthly,
        weekly=weekly,
        fit_months=fit_months,
        holdout_month=holdout_month,
        min_week_rows=min_week_rows,
    )
    if not fit_holdout.empty:
        fit_holdout["label_arm"] = fit_holdout["arm"].astype(str).str.split("::", n=1).str[-1]

    paths = {
        "source_summary": output_dir / "candidate_source_summary.csv",
        "monthly": output_dir / "candidate_source_monthly_selection.csv",
        "weekly": output_dir / "candidate_source_weekly_selection.csv",
        "proxy_ic": output_dir / "candidate_source_proxy_ic.csv",
        "fit_holdout": output_dir / "candidate_source_fit_holdout.csv",
        "manifest": output_dir / "manifest.json",
    }
    source_summary.to_csv(paths["source_summary"], index=False)
    monthly.to_csv(paths["monthly"], index=False)
    weekly.to_csv(paths["weekly"], index=False)
    proxy_ic.to_csv(paths["proxy_ic"], index=False)
    fit_holdout.to_csv(paths["fit_holdout"], index=False)

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
        "event_confirmation_features": event_report,
        "feature_count": int(len(features)),
        "proxy_top_k": int(proxy_top_k),
        "top_fracs": [float(v) for v in top_fracs],
        "fit_months": [str(v) for v in fit_months],
        "holdout_month": str(holdout_month),
        "arms": list(targets.keys()),
        "sources": list(source_masks.keys()),
        "run_gap_hours": float(run_gap_hours),
        "min_train_source_rows": int(min_train_source_rows),
        "min_valid_source_rows": int(min_valid_source_rows),
        "include_causal_time_edge_priors": bool(include_causal_time_edge_priors),
        "causal_time_edge_priors": time_edge_prior_report,
        "include_economic_guard_proxy": bool(include_economic_guard_proxy),
        "rows_monthly": int(len(monthly)),
        "rows_weekly": int(len(weekly)),
        "rows_fit_holdout": int(len(fit_holdout)),
        "feature_proxy_holdout_clean_pass_rows": int(
            fit_holdout[
                fit_holdout["selector"].eq("feature_ic_proxy")
                & fit_holdout["holdout_clean_pass"]
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
        "economic_guard_holdout_clean_pass_rows": int(
            fit_holdout[
                fit_holdout["selector"].eq("economic_guard_proxy")
                & fit_holdout["holdout_clean_pass"]
            ].shape[0]
        )
        if not fit_holdout.empty
        else 0,
        "economic_guard_holdout_bounded_pass_rows": int(
            fit_holdout[
                fit_holdout["selector"].eq("economic_guard_proxy")
                & fit_holdout["holdout_bounded_pass"]
            ].shape[0]
        )
        if not fit_holdout.empty
        else 0,
        "time_edge_guard_holdout_clean_pass_rows": int(
            fit_holdout[
                fit_holdout["selector"].eq("time_edge_guard_proxy")
                & fit_holdout["holdout_clean_pass"]
            ].shape[0]
        )
        if not fit_holdout.empty
        else 0,
        "time_edge_guard_holdout_bounded_pass_rows": int(
            fit_holdout[
                fit_holdout["selector"].eq("time_edge_guard_proxy")
                & fit_holdout["holdout_bounded_pass"]
            ].shape[0]
        )
        if not fit_holdout.empty
        else 0,
        "outputs": {key: str(value) for key, value in paths.items()},
    }
    markdown = _write_markdown(
        output_dir=output_dir,
        source_summary=source_summary,
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
    parser.add_argument("--proxy-top-k", type=int, default=4)
    parser.add_argument("--top-fracs", type=_parse_float_csv, default=",".join(str(v) for v in DEFAULT_TOP_FRACS[:5]))
    parser.add_argument("--fit-months", type=_parse_csv, default=",".join(DEFAULT_FIT_MONTHS))
    parser.add_argument("--holdout-month", type=str, default=DEFAULT_HOLDOUT_MONTH)
    parser.add_argument("--min-week-rows", type=int, default=2)
    parser.add_argument("--arms", type=_parse_csv, default=",".join(DEFAULT_ARMS))
    parser.add_argument("--sources", type=_parse_csv, default=None)
    parser.add_argument(
        "--event-feature-store-features",
        type=_parse_csv,
        default=",".join(DEFAULT_EVENT_FEATURE_STORE_FEATURES),
    )
    parser.add_argument("--run-gap-hours", type=float, default=2.0)
    parser.add_argument("--min-train-source-rows", type=int, default=200)
    parser.add_argument("--min-valid-source-rows", type=int, default=30)
    parser.add_argument("--include-causal-time-edge-priors", action="store_true")
    parser.add_argument("--prior-windows-days", type=_parse_float_csv, default=",".join(str(v) for v in DEFAULT_PRIOR_WINDOWS_DAYS))
    parser.add_argument("--prior-embargo-hours", type=float, default=24.0)
    parser.add_argument("--include-economic-guard-proxy", action="store_true")
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
        top_fracs=list(args.top_fracs),
        fit_months=list(args.fit_months),
        holdout_month=str(args.holdout_month),
        min_week_rows=int(args.min_week_rows),
        arms=list(args.arms),
        sources=list(args.sources) if args.sources else None,
        event_feature_store_features=list(args.event_feature_store_features),
        run_gap_hours=float(args.run_gap_hours),
        min_train_source_rows=int(args.min_train_source_rows),
        min_valid_source_rows=int(args.min_valid_source_rows),
        include_causal_time_edge_priors=bool(args.include_causal_time_edge_priors),
        prior_windows_days=list(args.prior_windows_days),
        prior_embargo_hours=float(args.prior_embargo_hours),
        include_economic_guard_proxy=bool(args.include_economic_guard_proxy),
    )
    print(json.dumps(_json_safe(manifest), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
